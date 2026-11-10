//! Independent request rows, ragged prefill, inactive rows and slot adoption.
//! Exact B1/B4 behavior is required before scheduler batching is enabled.
use ironmlx_lm::{
    core::{
        cache::layer::LayerCache,
        model_input::{build_batch_attention_mask, build_batch_linear_mask},
        Model,
    },
    models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest},
};
use mlx::{Array, Dtype};
use std::path::Path;

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
#[ignore = "requires actual Flash checkpoint; long B4 paging correctness and phase timings"]
fn flash_long_paged_b4_matches_independent_rows_across_sparse_boundary() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    mlx::memory::set_cache_limit(128 << 20);
    let mut request = FlashLoadRequest::new(48u64 << 30, 4, 4096, 2049, 0);
    request.expert_residency = ironmlx_lm::core::residency::ExpertResidency::Ssd;
    request.expert_cache_limit_bytes = 2u64 << 30;
    request.execution =
        ironmlx_lm::models::qwen4_exp::text_model::FlashExecutionOptions::from_environment()
            .unwrap();
    let plan = FlashLoadPlan::inspect(Path::new(&path), &request).unwrap();
    let loader = plan.open_loader(Path::new(&path)).unwrap();
    let model = plan.build_model(&loader).unwrap();
    drop(loader);
    let lengths = [63, 127, 1024, 2049];
    let max = 2049;
    let prompts: Vec<Vec<u32>> = lengths
        .iter()
        .enumerate()
        .map(|(row, &len)| {
            (0..len)
                .map(|n| 1000 + ((n * 13 + row as i32 * 71) % 10000) as u32)
                .collect()
        })
        .collect();
    let mut padded = vec![0; 4 * max as usize];
    for (r, p) in prompts.iter().enumerate() {
        padded[r * max as usize..r * max as usize + p.len()].copy_from_slice(p);
    }
    let mut batch = model.make_cache(4, 4096, Dtype::Bfloat16).unwrap();
    let mask = build_batch_linear_mask(&lengths, max).unwrap();
    let attention = build_batch_attention_mask(&lengths, max, Dtype::Bfloat16).unwrap();
    let started = std::time::Instant::now();
    let output = model
        .batched_prefill(
            &ids(&padded, 4, max),
            &positions(&batch, max),
            &attention,
            &mask,
            &lengths,
            Some(&mut batch),
            ().into(),
        )
        .unwrap();
    output.eval().unwrap();
    eprintln!(
        "LONG_B4_PREFILL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!(
        "LONG_B4_STATS={}",
        serde_json::to_string(&model.paging_telemetry().unwrap().snapshot()).unwrap()
    );
    let mut refs = Vec::new();
    for (r, p) in prompts.iter().enumerate() {
        let mut cache = model.make_cache(1, 4096, Dtype::Bfloat16).unwrap();
        let expected = model
            .forward_on(
                &ids(p, 1, p.len() as i32),
                &positions(&cache, p.len() as i32),
                None,
                None,
                Some(&mut cache),
                ().into(),
            )
            .unwrap();
        exact(
            "long paged B4 prefill",
            &output
                .slice(
                    &[r as i32, 0, 0][..],
                    &[r as i32 + 1, 1, model.config.vocab_size as i32][..],
                )
                .unwrap(),
            &expected,
        );
        refs.push(cache);
    }
    for tokens in [[701, 702, 703, 704], [705, 706, 707, 708]] {
        let started = std::time::Instant::now();
        let output = model
            .forward_on(
                &ids(&tokens, 4, 1),
                &positions(&batch, 1),
                None,
                None,
                Some(&mut batch),
                ().into(),
            )
            .unwrap();
        output.eval().unwrap();
        eprintln!(
            "LONG_B4_DECODE_MS={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        for (r, cache) in refs.iter_mut().enumerate() {
            let expected = model
                .forward_on(
                    &ids(&tokens[r..r + 1], 1, 1),
                    &positions(cache, 1),
                    None,
                    None,
                    Some(cache),
                    ().into(),
                )
                .unwrap();
            exact(
                "long paged B4 continuation",
                &output
                    .slice(
                        &[r as i32, 0, 0][..],
                        &[r as i32 + 1, 1, model.config.vocab_size as i32][..],
                    )
                    .unwrap(),
                &expected,
            );
        }
    }
    assert_eq!(
        offsets(&batch),
        lengths.iter().map(|n| n + 2).collect::<Vec<_>>()
    );
    let stats = model.paging_telemetry().unwrap().snapshot();
    assert!(stats.experts.charged_bytes <= stats.experts.budget_bytes);
    eprintln!(
        "LONG_B4_FINAL_STATS={} memory={:?}",
        serde_json::to_string(&stats).unwrap(),
        mlx::memory::snapshot()
    );
}
