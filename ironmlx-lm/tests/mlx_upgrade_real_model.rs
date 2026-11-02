//! Opt-in, sequential real-checkpoint gate for dependency upgrades.
//! Set IRONMLX_UPGRADE_MODEL to an immutable Dense or MoE snapshot. Compare
//! B1 and B4 on the same fixed token trajectories, including cache updates.
use ironmlx_lm::core::model_input::{build_decode_position_ids, build_position_ids_batched};
use ironmlx_lm::core::{Loader, Model};
use ironmlx_lm::models::{Qwen35Model, Qwen35MoeModel};
use mlx::{Array, Dtype};

fn prefill(
    model: &dyn Model,
    ids: &Array,
    batch: i32,
    cache: &mut [ironmlx_lm::core::cache::layer::LayerCache],
) -> Array {
    // Match the scheduler contract, including the MoE N-1 + 1 morphology.
    // A raw tensor prefill is not a qualification of that serving path.
    if model.requires_split_batched_prefill_for_token_parity() {
        let prefix = ids.slice(&[0, 0][..], &[batch, 3][..]).unwrap();
        let pos = build_position_ids_batched(&vec![3; batch as usize], 3).unwrap();
        let _ = logits(
            model
                .forward_on(&prefix, &pos, None, None, Some(cache), ().into())
                .unwrap(),
        );
        let last = ids.slice(&[0, 3][..], &[batch, 4][..]).unwrap();
        let pos = build_decode_position_ids(&vec![3; batch as usize]).unwrap();
        model
            .forward_on(&last, &pos, None, None, Some(cache), ().into())
            .unwrap()
    } else {
        let pos = build_position_ids_batched(&vec![4; batch as usize], 4).unwrap();
        model
            .forward_on(ids, &pos, None, None, Some(cache), ().into())
            .unwrap()
    }
}

fn logits(a: Array) -> Vec<f32> {
    let values = a.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|x| x.is_finite()));
    values
}

#[test]
#[ignore = "requires IRONMLX_UPGRADE_MODEL and a real GPU checkpoint"]
fn dense_or_moe_prefill_and_cached_decode_b1_b4() {
    let path = std::env::var("IRONMLX_UPGRADE_MODEL").expect("explicit snapshot required");
    if std::env::var("IRONMLX_UPGRADE_M5_PROFILE").as_deref() == Ok("1") {
        use ironmlx_core::m5_profile::{self, AneMode, M5ProfileMode, M5ProfileTarget};
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(&path).join("config.json")).unwrap(),
        )
        .unwrap();
        let target = if config["model_type"] == "qwen3_5_moe" {
            M5ProfileTarget::Qwen36Moe
        } else {
            M5ProfileTarget::Qwen35Dense
        };
        let profile = m5_profile::install_for_target_with_ane(
            M5ProfileMode::Auto,
            AneMode::Off,
            true,
            target,
        );
        assert!(
            profile.is_active(),
            "M5 profile qualification requires an eligible GPU"
        );
    }
    let loader = Loader::open(std::path::Path::new(&path)).unwrap();
    let is_moe = loader.config_raw_value()["model_type"] == "qwen3_5_moe";
    let model: Box<dyn Model> = match loader.config_raw_value()["model_type"].as_str().unwrap() {
        "qwen3_5" => Box::new(Qwen35Model::from_loader(&loader).unwrap()),
        "qwen3_5_moe" => Box::new(Qwen35MoeModel::from_loader(&loader).unwrap()),
        other => panic!("unsupported qualification model {other}"),
    };
    drop(loader);
    let prompts: [[i32; 4]; 4] = [
        [100, 200, 300, 400],
        [600, 700, 800, 900],
        [1000, 1100, 1200, 1300],
        [2000, 2100, 2200, 2300],
    ];
    let mut reference = Vec::new();
    for prompt in &prompts {
        let mut cache = model.make_cache(1, 256, Dtype::Bfloat16).unwrap();
        let ids = Array::try_from((prompt.as_slice(), &[1, 4][..])).unwrap();
        let prefill = logits(prefill(model.as_ref(), &ids, 1, &mut cache));
        // Fixed continuation keeps both routes on an identical trajectory.
        let next = Array::try_from((&[42i32][..], &[1, 1][..])).unwrap();
        let pos = build_decode_position_ids(&[4]).unwrap();
        let decode = logits(
            model
                .forward_on(&next, &pos, None, None, Some(&mut cache), ().into())
                .unwrap(),
        );
        reference.push((prefill, decode));
    }
    let mut cache = model.make_cache(4, 256, Dtype::Bfloat16).unwrap();
    let tokens: Vec<i32> = prompts.into_iter().flatten().collect();
    let ids = Array::try_from((tokens.as_slice(), &[4, 4][..])).unwrap();
    let prefill = logits(prefill(model.as_ref(), &ids, 4, &mut cache));
    let next = Array::try_from((&[42i32; 4][..], &[4, 1][..])).unwrap();
    let pos = build_decode_position_ids(&[4; 4]).unwrap();
    let decode = logits(
        model
            .forward_on(&next, &pos, None, None, Some(&mut cache), ().into())
            .unwrap(),
    );
    let vocab = reference[0].0.len();
    if let Ok(destination) = std::env::var("IRONMLX_UPGRADE_CAPTURE") {
        std::fs::write(
            destination,
            serde_json::to_vec(&serde_json::json!({
                "snapshot": path, "b1": reference, "b4_prefill": prefill, "b4_decode": decode
            }))
            .unwrap(),
        )
        .unwrap();
    }
    for (stage, actual) in [(0, prefill), (1, decode)] {
        assert_eq!(actual.len(), 4 * vocab);
        for (row, expected) in reference.iter().enumerate() {
            let expected = if stage == 0 { &expected.0 } else { &expected.1 };
            let actual = &actual[row * vocab..(row + 1) * vocab];
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rms = (actual
                .iter()
                .zip(expected)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / vocab as f64)
                .sqrt();
            eprintln!("{path}: stage={stage} row={row} max_abs={max_abs:.5} rms={rms:.5}");
            if is_moe {
                assert_eq!(
                    actual, expected,
                    "MoE request isolation changed B1/B4 products"
                );
            } else {
                assert!(max_abs < 1.0 && rms < 0.15, "B1/B4 logits diverged");
            }
        }
    }
}
