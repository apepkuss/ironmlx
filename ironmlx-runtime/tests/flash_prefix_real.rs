//! Opt-in cross-process Flash codec qualification with the actual checkpoint.
//! Writer and reader are separate processes; reader checks full-vocabulary
//! logits against uninterrupted B1 continuations at native B1/B2/B4 widths.
use ironmlx_lm::{
    core::{
        cache::layer::{
            adopt_layer_cache_rows, prefix_entry_for_row, prefix_key_spec_for_caches,
            restore_prefix_entry_for_rows, LayerCache,
        },
        Model,
    },
    models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest},
};
use ironmlx_runtime::core::cache::PagedPrefixStore;
use mlx::{Array, Dtype};
use std::{collections::HashMap, path::Path};
fn positions(cache: &[LayerCache], width: i32) -> Array {
    let LayerCache::Flash(cache) = &cache[0] else {
        unreachable!()
    };
    let p: Vec<i32> = cache
        .offsets()
        .iter()
        .flat_map(|start| (0..width).map(move |i| start + i))
        .collect();
    Array::try_from((
        p.repeat(3).as_slice(),
        [3, cache.offsets().len() as i32, width],
    ))
    .unwrap()
}
fn exact(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert!(
        a.equal(b)
            .unwrap()
            .reshape([-1])
            .unwrap()
            .all(-1, false)
            .unwrap()
            .item::<bool>()
            .unwrap(),
        "full-vocabulary logits differ"
    );
}
#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and actual Apple GPU; allocates a bounded 2 GiB expert cache"]
fn flash_prefix_cross_process_exact_b1_b2_b4() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let root = std::env::var("IRONMLX_FLASH_PREFIX_ARTIFACTS").unwrap();
    let root = Path::new(&root);
    std::fs::create_dir_all(root).unwrap();
    let len = std::env::var("IRONMLX_FLASH_PREFIX_LEN")
        .ok()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(4097);
    let role = std::env::var("IRONMLX_FLASH_PREFIX_ROLE").unwrap_or_default();
    if role.is_empty() {
        for role in ["write", "read"] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "flash_prefix_cross_process_exact_b1_b2_b4",
                    "--nocapture",
                ])
                .env("IRONMLX_FLASH_PREFIX_ROLE", role)
                .output()
                .unwrap();
            std::fs::write(
                root.join(format!("{role}.log")),
                [&result.stdout[..], &result.stderr[..]].concat(),
            )
            .unwrap();
            assert!(
                result.status.success(),
                "{role} failed: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        return;
    }
    mlx::memory::set_cache_limit(128 << 20);
    mlx::memory::set_wired_limit(48usize << 30).unwrap();
    let mut request = FlashLoadRequest::new(64u64 << 30, 4, len + 64, 2048, 0);
    request.expert_residency = ironmlx_lm::core::residency::ExpertResidency::Ssd;
    request.expert_cache_limit_bytes = 2u64 << 30;
    let plan = FlashLoadPlan::inspect(Path::new(&snapshot), &request).unwrap();
    let loader = plan.open_loader(Path::new(&snapshot)).unwrap();
    let model = plan.build_model(&loader).unwrap();
    drop(loader);
    let tokens: Vec<u32> = (0..len)
        .map(|i| {
            if i == 31 {
                model.config.eos_token_id
            } else {
                1000 + ((i * 13) % 10000) as u32
            }
        })
        .collect();
    let mut cache = model
        .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
        .unwrap();
    let spec = prefix_key_spec_for_caches(
        "qualified-flash",
        &tokens,
        len as i32,
        Some("ordinary-text-v1"),
        128,
        &cache,
    )
    .unwrap()
    .unwrap();
    let store = PagedPrefixStore::new(root.join("store")).with_max_bytes(8usize << 30);
    if role == "write" {
        let started = std::time::Instant::now();
        for chunk in tokens.chunks(2048) {
            let output = model
                .forward_on(
                    &Array::try_from((chunk, [1, chunk.len() as i32])).unwrap(),
                    &positions(&cache, chunk.len() as i32),
                    None,
                    None,
                    Some(&mut cache),
                    ().into(),
                )
                .unwrap();
            output.eval().unwrap();
        }
        let prefill_s = started.elapsed().as_secs_f64();
        let telemetry = model.paging_telemetry().unwrap().snapshot();
        let (entry, cached_len) = prefix_entry_for_row(&cache, 0).unwrap().unwrap();
        assert_eq!(cached_len, len as i32);
        let save = std::time::Instant::now();
        store.save(&spec, &entry).unwrap();
        let save_s = save.elapsed().as_secs_f64();
        let mut reference = HashMap::new();
        for row in 0..4 {
            let mut independent = model
                .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
                .unwrap();
            adopt_layer_cache_rows(&mut independent, &cache, 0, 0).unwrap();
            for step in 0..8 {
                let token = [700 + row * 17 + step];
                let output = model
                    .forward_on(
                        &Array::try_from((token.as_slice(), [1, 1])).unwrap(),
                        &positions(&independent, 1),
                        None,
                        None,
                        Some(&mut independent),
                        ().into(),
                    )
                    .unwrap();
                output.eval().unwrap();
                reference.insert(format!("row{row}_step{step}"), output);
            }
        }
        mlx::io::save_safetensors(
            &root.join("reference.safetensors").to_string_lossy(),
            &reference,
            &HashMap::new(),
        )
        .unwrap();
        std::fs::write(root.join("write.json"), serde_json::to_vec_pretty(&serde_json::json!({"prefill_s":prefill_s,"save_s":save_s,"payload_bytes":spec.payload_bytes(),"paging":telemetry})).unwrap()).unwrap();
    } else {
        let start = std::time::Instant::now();
        let entry = store.load(&spec).unwrap().expect("cross-process hit");
        let load_s = start.elapsed().as_secs_f64();
        let (reference, _) =
            mlx::io::load_safetensors(&root.join("reference.safetensors").to_string_lossy())
                .unwrap();
        for batch in [1, 2, 4] {
            let mut cache = model
                .make_cache(batch, (len + 64) as i32, Dtype::Bfloat16)
                .unwrap();
            restore_prefix_entry_for_rows(
                &mut cache,
                &entry,
                &(0..batch as usize).collect::<Vec<_>>(),
                len as i32,
            )
            .unwrap();
            for step in 0..8 {
                let tokens: Vec<u32> = (0..batch).map(|row| 700 + row as u32 * 17 + step).collect();
                let output = model
                    .forward_on(
                        &Array::try_from((tokens.as_slice(), [batch, 1])).unwrap(),
                        &positions(&cache, 1),
                        Some(&vec![1; batch as usize]),
                        None,
                        Some(&mut cache),
                        ().into(),
                    )
                    .unwrap();
                output.eval().unwrap();
                for row in 0..batch {
                    let actual = output
                        .slice(
                            &[row, 0, 0][..],
                            &[row + 1, 1, model.config.vocab_size as i32][..],
                        )
                        .unwrap();
                    exact(&actual, &reference[&format!("row{row}_step{step}")]);
                }
            }
        }
        std::fs::write(root.join("read.json"), serde_json::to_vec_pretty(&serde_json::json!({"load_s":load_s,"exact_logits_widths":[1,2,4],"decode_steps":8,"paging":model.paging_telemetry().unwrap().snapshot()})).unwrap()).unwrap();
    }
}
