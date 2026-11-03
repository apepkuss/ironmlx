//! Independent request rows, ragged prefill, inactive rows and slot adoption.
//! Exact B1/B4 behavior is required before scheduler batching is enabled.
use ironmlx_lm::{
    core::{
        cache::layer::{adopt_layer_cache_rows, LayerCache},
        model_input::{build_batch_attention_mask, build_batch_linear_mask},
        tensor_file::ModelTensorFiles,
        Loader, Model,
    },
    models::qwen4_exp::text_model::{ExpertPagingOptions, FlashTextModel},
};
use mlx::{Array, Dtype};
use std::{path::Path, sync::Arc};

fn exact(label: &str, a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape(), "{label}");
    assert_eq!(a.dtype(), b.dtype(), "{label}");
    assert_eq!(
        a.equal(b)
            .unwrap()
            .reshape([-1])
            .unwrap()
            .all(-1, false)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true],
        "{label}"
    );
}
fn positions(cache: &[LayerCache], seq: i32) -> Array {
    let starts = offsets(cache);
    let one: Vec<i32> = starts
        .iter()
        .flat_map(|start| (0..seq).map(move |i| start + i))
        .collect();
    Array::try_from((one.repeat(3).as_slice(), [3, starts.len() as i32, seq])).unwrap()
}
fn ids(tokens: &[u32], batch: i32, seq: i32) -> Array {
    Array::try_from((tokens, [batch, seq])).unwrap()
}
fn offsets(cache: &[LayerCache]) -> Vec<i32> {
    let LayerCache::Flash(first) = &cache[0] else {
        panic!("Flash cache")
    };
    for layer in cache {
        let LayerCache::Flash(layer) = layer else {
            panic!("Flash cache")
        };
        assert_eq!(layer.offsets(), first.offsets());
    }
    first.offsets().to_vec()
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT; defaults to a forced 64 MiB expert cache"]
fn flash_ragged_b4_matches_b1_and_preserves_inactive_and_adopted_rows() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("snapshot required");
    mlx::memory::set_cache_limit(128 * 1024 * 1024);
    mlx::memory::set_wired_limit(24usize << 30).unwrap();
    let loader = Loader::open_flash_text_paged(Path::new(&path), 8u64 << 30).unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&path)).unwrap());
    let model = FlashTextModel::from_loader_with_paging(
        &loader,
        files,
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        Some(ExpertPagingOptions {
            cache_bytes: 64 * 1024 * 1024,
            staging_bytes: 16 * 1024 * 1024,
            pack_bytes: 64 * 1024 * 1024,
        }),
    )
    .unwrap();
    drop(loader);
    let prompts = [
        vec![100, 200, 248044, 300],
        vec![100, 200, 248044, 301, 302, 303],
        vec![400, 401],
        vec![500, 248044, 501, 502, 503],
    ];
    let lengths: Vec<i32> = prompts.iter().map(|p| p.len() as i32).collect();
    let max = *lengths.iter().max().unwrap();
    let mut padded = vec![0; 4 * max as usize];
    for (row, prompt) in prompts.iter().enumerate() {
        padded[row * max as usize..row * max as usize + prompt.len()].copy_from_slice(prompt);
    }
    let mut batch = model.make_cache(4, 32, Dtype::Bfloat16).unwrap();
    let attention = build_batch_attention_mask(&lengths, max, Dtype::Bfloat16).unwrap();
    let linear = build_batch_linear_mask(&lengths, max).unwrap();
    let prefill_started = std::time::Instant::now();
    let actual = model
        .batched_prefill(
            &ids(&padded, 4, max),
            &positions(&batch, max),
            &attention,
            &linear,
            &lengths,
            Some(&mut batch),
            ().into(),
        )
        .unwrap();
    actual.eval().unwrap();
    eprintln!(
        "TIMING prefill_b4_ms={:.3}",
        prefill_started.elapsed().as_secs_f64() * 1000.0
    );
    let mut references = Vec::new();
    for (row, prompt) in prompts.iter().enumerate() {
        let mut cache = model.make_cache(1, 32, Dtype::Bfloat16).unwrap();
        let expected = model
            .forward_on(
                &ids(prompt, 1, prompt.len() as i32),
                &positions(&cache, prompt.len() as i32),
                None,
                None,
                Some(&mut cache),
                ().into(),
            )
            .unwrap();
        let row_output = actual
            .slice(
                &[row as i32, 0, 0][..],
                &[row as i32 + 1, 1, model.config.vocab_size as i32][..],
            )
            .unwrap();
        exact("ragged prefill B1/B4", &row_output, &expected);
        references.push(cache);
    }
    assert_eq!(offsets(&batch), lengths);
    let tokens = [700, 701, 702, 703];
    let post_lengths: Vec<i32> = lengths.iter().map(|n| n + 1).collect();
    let decode_mask = ironmlx_lm::core::model_input::build_per_row_decode_mask(
        &post_lengths,
        *post_lengths.iter().max().unwrap(),
        Dtype::Bfloat16,
    )
    .unwrap();
    let decode_started = std::time::Instant::now();
    let actual = model
        .forward_on(
            &ids(&tokens, 4, 1),
            &positions(&batch, 1),
            Some(&[1, 1, 1, 1]),
            Some(&decode_mask),
            Some(&mut batch),
            ().into(),
        )
        .unwrap();
    actual.eval().unwrap();
    eprintln!(
        "TIMING decode_b4_ms={:.3}",
        decode_started.elapsed().as_secs_f64() * 1000.0
    );
    for row in 0..4 {
        let expected = model
            .forward_on(
                &ids(&tokens[row..row + 1], 1, 1),
                &positions(&references[row], 1),
                None,
                None,
                Some(&mut references[row]),
                ().into(),
            )
            .unwrap();
        exact(
            "cached decode B1/B4",
            &actual
                .slice(
                    &[row as i32, 0, 0][..],
                    &[row as i32 + 1, 1, model.config.vocab_size as i32][..],
                )
                .unwrap(),
            &expected,
        );
    }
    assert_eq!(
        offsets(&batch),
        lengths.iter().map(|n| n + 1).collect::<Vec<_>>()
    );
    if std::env::var("IRONMLX_FLASH_PAGED_BATCHING").as_deref() != Ok("off") {
        let stats = model.paging_telemetry().unwrap().snapshot();
        assert_eq!(
            stats.experts.phases[0].request_width, 4,
            "prefill must preserve the B4 input width"
        );
        assert_eq!(
            stats.experts.phases[1].request_width, 4,
            "decode must preserve the B4 input width"
        );
        assert!(stats
            .experts
            .phases
            .iter()
            .all(|p| p.route_count >= p.hits + p.misses));
        assert!(stats.experts.charged_bytes <= stats.experts.budget_bytes);
        assert_eq!(
            stats.experts.pinned_bytes, 0,
            "GPU consumers have completed"
        );
        eprintln!("paging stats: {}", serde_json::to_string(&stats).unwrap());
    }
    let mut adopted = model.make_cache(4, 64, Dtype::Bfloat16).unwrap();
    adopt_layer_cache_rows(&mut adopted, &batch, 0, 2).unwrap();
    assert_eq!(offsets(&adopted), [lengths[2] + 1, 0, 0, 0]);
    let mask = Array::try_from((&[true, false, false, false][..], [4, 1])).unwrap();
    let actual = model
        .forward_on(
            &ids(&[812, 813, 814, 815], 4, 1),
            &positions(&adopted, 1),
            None,
            Some(&mask),
            Some(&mut adopted),
            ().into(),
        )
        .unwrap();
    let expected = model
        .forward_on(
            &ids(&[812], 1, 1),
            &positions(&references[2], 1),
            None,
            None,
            Some(&mut references[2]),
            ().into(),
        )
        .unwrap();
    exact(
        "adopted row continuation",
        &actual
            .slice(&[0, 0, 0][..], &[1, 1, model.config.vocab_size as i32][..])
            .unwrap(),
        &expected,
    );
    assert_eq!(offsets(&adopted), [lengths[2] + 2, 0, 0, 0]);
    let before: Vec<_> = adopted.iter().map(LayerCache::snapshot).collect();
    let positions_before = offsets(&adopted);
    assert!(model
        .forward_on(
            &ids(&[9, 10, 11, model.config.vocab_size], 4, 1),
            &positions(&adopted, 1),
            None,
            None,
            Some(&mut adopted),
            ().into()
        )
        .is_err());
    assert_eq!(offsets(&adopted), positions_before);
    let mut expected_state = model.make_cache(4, 64, Dtype::Bfloat16).unwrap();
    for (c, snapshot) in expected_state.iter_mut().zip(&before) {
        c.restore(snapshot).unwrap();
    }
    for (a, b) in adopted.iter().zip(&expected_state) {
        for (a, b) in a.state_arrays().into_iter().zip(b.state_arrays()) {
            exact("failed batch state", a, b);
        }
    }
    for layer in &mut adopted {
        layer.reset().unwrap();
    }
    assert_eq!(offsets(&adopted), [0, 0, 0, 0]);
    eprintln!(
        "ragged B4, inactive rows, slot adoption, rejection and reset: exact; memory={:?}",
        mlx::memory::snapshot()
    );
}

#[test]
#[ignore = "requires actual Flash checkpoint; direct native PagedMoe B2/B4"]
#[cfg(feature = "test-support")]
fn flash_paged_moe_direct_b2_b4_merges_expert_work() {
    use ironmlx_lm::{
        core::paged_experts::{ExpertTableSpec, PagedExperts},
        models::qwen4_exp::{config::FlashTextConfig, PagedMoe},
    };
    use std::sync::Mutex;
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let loader = Loader::open_flash_text_paged(Path::new(&path), 8u64 << 30).unwrap();
    let cfg = FlashTextConfig::from_root(loader.config_raw_value()).unwrap();
    let prefix = "model.layers.0.mlp";
    let quant = loader
        .quant_meta_for(&format!("{prefix}.switch_mlp.gate_proj"))
        .unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&path)).unwrap());
    let store = Arc::new(Mutex::new(
        PagedExperts::new(
            files,
            vec![ExpertTableSpec {
                prefix: format!("language_model.{prefix}.switch_mlp"),
                experts: cfg.num_experts as u32,
                hidden: cfg.hidden_size,
                intermediate: cfg.moe_intermediate_size,
                quant,
            }],
            64 << 20,
            16 << 20,
            64 << 20,
        )
        .unwrap(),
    ));
    let moe = PagedMoe::from_weights(&loader, prefix, &cfg, Arc::clone(&store), 0).unwrap();
    for seq in [2, 127, 205, 1024] {
        let values: Vec<f32> = (0..cfg.hidden_size)
            .map(|n| ((n % 113) as f32 - 56.0) / 113.0)
            .collect::<Vec<_>>()
            .repeat(seq as usize);
        let one = Array::try_from((values.as_slice(), [1, seq, cfg.hidden_size]))
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap();
        let expected = moe.forward_on(&one, ().into(), 0).unwrap();
        expected.eval().unwrap();
        let before = store.lock().unwrap().stats();
        for width in [2, 4] {
            let input = mlx::ops::shape::concatenate(&vec![&one; width], 0).unwrap();
            let output = moe.forward_on(&input, ().into(), 0).unwrap();
            output.eval().unwrap();
            exact(
                &format!("direct native MoE B{width} S{seq}"),
                &output,
                &mlx::ops::shape::concatenate(&vec![&expected; width], 0).unwrap(),
            );
        }
        let cache = store.lock().unwrap();
        assert_eq!(
            cache.stats().misses,
            before.misses,
            "identical requests must share resident experts"
        );
        assert_eq!(cache.telemetry().snapshot().experts.phases[0].max_width, 4);
    }
    assert!(moe
        .forward_on(
            &Array::zeros([0, 2, cfg.hidden_size], Dtype::Bfloat16).unwrap(),
            ().into(),
            0
        )
        .is_err());
}
