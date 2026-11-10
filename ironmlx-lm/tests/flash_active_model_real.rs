//! Full target quality and within-tier batching gate; opt-in actual hardware.
use ironmlx_lm::{
    core::{active_pages::ActivePagesConfig, cache::layer::LayerCache, Model},
    models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest},
};
use mlx::{Array, Dtype};
use std::path::Path;

fn positions(cache: &[LayerCache], seq: i32) -> Array {
    let LayerCache::Flash(layer) = &cache[0] else {
        unreachable!()
    };
    let p: Vec<_> = layer
        .offsets()
        .iter()
        .flat_map(|&start| start..start + seq)
        .collect();
    Array::try_from((
        p.repeat(3).as_slice(),
        [3, layer.offsets().len() as i32, seq],
    ))
    .unwrap()
}
fn run(
    model: &impl Model,
    cache: &mut [LayerCache],
    tokens: &[u32],
    batch: i32,
    seq: i32,
) -> Array {
    let result = model
        .forward_on(
            &Array::try_from((tokens, [batch, seq])).unwrap(),
            &positions(cache, seq),
            None,
            None,
            Some(cache),
            ().into(),
        )
        .unwrap();
    result.eval().unwrap();
    result
}
fn exact(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    assert!(a
        .equal(b)
        .unwrap()
        .reshape([-1])
        .unwrap()
        .all(-1, false)
        .unwrap()
        .item::<bool>()
        .unwrap());
}
fn quality(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    let a = a.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
    let b = b.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
    let mut max = 0f32;
    let mut peak = 0f32;
    let mut err = 0f64;
    let mut norm = 0f64;
    for (&a, &b) in a.iter().zip(&b) {
        assert!(a.is_finite() && b.is_finite());
        max = max.max((a - b).abs());
        peak = peak.max(a.abs());
        err += ((a - b) as f64).powi(2);
        norm += (a as f64).powi(2);
    }
    let rms = (err / norm.max(1e-20)).sqrt();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    eprintln!(
        "full target SSD max_abs={max} peak={peak} relative_rms={rms} argmax_ram={} argmax_ssd={}",
        argmax(&a),
        argmax(&b)
    );
    assert!(
        max <= peak * 0.03 + 1e-4 && rms <= 0.01,
        "full target numerical quality regression"
    );
    assert_eq!(
        argmax(&a),
        argmax(&b),
        "fixed-trajectory greedy output changed"
    );
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and actual Apple GPU; optional IRONMLX_FLASH_QUALIFY_FULL=1"]
fn flash_active_full_vocab_quality_and_b2_b4_isolation() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let len = std::env::var("IRONMLX_FLASH_ACTIVE_CONTEXT")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(4099);
    let full = std::env::var("IRONMLX_FLASH_QUALIFY_FULL").as_deref() == Ok("1");
    mlx::memory::set_cache_limit(128 << 20);
    mlx::memory::set_wired_limit(if full { 96usize << 30 } else { 24usize << 30 }).unwrap();
    let mut request = FlashLoadRequest::new(
        if full { 110u64 << 30 } else { 24u64 << 30 },
        4,
        len + 64,
        2048,
        0,
    );
    request.expert_residency = if full {
        ironmlx_lm::core::residency::ExpertResidency::Full
    } else {
        ironmlx_lm::core::residency::ExpertResidency::Ssd
    };
    request.expert_cache_limit_bytes = 2 << 30;
    request.execution =
        ironmlx_lm::models::qwen4_exp::text_model::FlashExecutionOptions::from_environment()
            .unwrap();
    let plan = FlashLoadPlan::inspect(Path::new(&path), &request).unwrap();
    let loader = plan.open_loader(Path::new(&path)).unwrap();
    let model = plan.build_model(&loader).unwrap();
    drop(loader);
    let prompt: Vec<u32> = (0..len).map(|i| 1000 + ((i * 13) % 10000) as u32).collect();
    let mut cache = model
        .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
        .unwrap();
    let mut reference = Vec::new();
    for chunk in prompt.chunks(2048) {
        let out = run(&model, &mut cache, chunk, 1, chunk.len() as i32);
        if chunk.len() != 2048 {
            reference.push(out);
        }
    }
    for token in 700..708 {
        reference.push(run(&model, &mut cache, &[token], 1, 1));
    }
    drop(cache);
    let root = std::env::temp_dir().join(format!("flash-model-active-{}", uuid::Uuid::new_v4()));
    let model = model
        .with_active_state(ActivePagesConfig {
            root: root.clone(),
            cache_bytes: 64 << 10,
            pack_bytes: 8 << 20,
            disk_bytes: 2 << 30,
            min_free_bytes: 0,
        })
        .unwrap();
    let regular_cost = model.scheduler_resident_cache_cost().unwrap();
    let draft_cost = ironmlx_lm::models::dflash2::DFlash2Target::dflash2_target_cache_cost(&model);
    assert_eq!(
        regular_cost,
        (
            draft_cost.bytes_per_token,
            draft_cost.fixed_bytes_per_sequence
        )
    );
    assert_eq!(regular_cost, (16384, 3usize << 30));
    let mut prefix = model
        .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
        .unwrap();
    let mut current = Vec::new();
    for chunk in prompt.chunks(2048) {
        let out = run(&model, &mut prefix, chunk, 1, chunk.len() as i32);
        if chunk.len() != 2048 {
            current.push(out);
        }
    }
    {
        let mut cache = model
            .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
            .unwrap();
        ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(&mut cache, &prefix, 0, 0).unwrap();
        for token in 700..708 {
            current.push(run(&model, &mut cache, &[token], 1, 1));
        }
        assert_eq!(reference.len(), current.len());
        for (a, b) in reference.iter().zip(&current) {
            quality(a, b);
            exact(a, b);
        }
    }
    for width in [2usize, 4] {
        let mut batch = model
            .make_cache(width as i32, (len + 64) as i32, Dtype::Bfloat16)
            .unwrap();
        let mut singles = Vec::new();
        for row in 0..width {
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(&mut batch, &prefix, row, 0)
                .unwrap();
            let mut cache = model
                .make_cache(1, (len + 64) as i32, Dtype::Bfloat16)
                .unwrap();
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(&mut cache, &prefix, 0, 0)
                .unwrap();
            singles.push(cache);
        }
        for step in 0..4 {
            let tokens: Vec<u32> = (0..width)
                .map(|row| 700 + (row * 17 + step) as u32)
                .collect();
            let actual = run(&model, &mut batch, &tokens, width as i32, 1);
            for row in 0..width {
                let expected = run(&model, &mut singles[row], &tokens[row..row + 1], 1, 1);
                exact(
                    &actual
                        .slice(
                            [row as i32, 0, 0],
                            [row as i32 + 1, 1, model.config.vocab_size as i32],
                        )
                        .unwrap(),
                    &expected,
                );
            }
        }
    }
    drop(prefix);
    // Layout-only rejection must happen before attempting to materialize a
    // long optional prefix from SSD. No long KV tensors are allocated here.
    let oversized = model.make_cache(1, 32768, Dtype::Bfloat16).unwrap();
    assert!(ironmlx_lm::core::cache::layer::prefix_key_spec_for_caches(
        "active-test",
        &vec![1; 32768],
        32768,
        None,
        128,
        &oversized
    )
    .unwrap()
    .is_none());
    drop(oversized);
    let paging = model.paging_telemetry().unwrap().snapshot();
    if !full {
        assert!(
            paging
                .experts
                .phases
                .iter()
                .any(|phase| phase.max_width >= 4 && phase.microbatches > 0),
            "SSD experts must execute native B4 microbatches"
        );
    }
    let stats = paging.active_state.unwrap();
    eprintln!("full model active state {stats:?}");
    assert!(stats.written_bytes > 0 && stats.read_bytes > 0);
    assert_eq!(stats.failures, 0);
    assert_eq!(stats.live_files, 0);
    assert_eq!(stats.disk_bytes, 0);
    drop(model);
    std::fs::remove_dir(root).unwrap();
}

#[test]
#[ignore = "requires actual checkpoint metadata; does not allocate model weights"]
fn flash_active_memory_capacity_plans() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    for (physical, batch, cap) in [(32u64, 1usize, 32768usize), (64, 2, 32768), (128, 4, 32768)] {
        let available = (physical - 8) << 30;
        let mut request = FlashLoadRequest::new(available, batch, cap, 2048, 0);
        request.expert_residency = ironmlx_lm::core::residency::ExpertResidency::Ssd;
        request.active_pages = Some(ActivePagesConfig::for_memory(
            std::env::temp_dir().join("flash-plan-only"),
            available,
        ));
        let plan = FlashLoadPlan::inspect(Path::new(&path), &request).unwrap();
        eprintln!(
            "SIMULATED physical_gib={physical} batch={batch} cap={cap} {}",
            serde_json::to_string(&plan).unwrap()
        );
        assert!(plan.residency.peak_bytes <= available);
    }
    let mut ram = FlashLoadRequest::new(24u64 << 30, 1, 131072, 2048, 0);
    ram.expert_residency = ironmlx_lm::core::residency::ExpertResidency::Ssd;
    assert!(FlashLoadPlan::inspect(Path::new(&path), &ram).is_err());
    ram.active_pages = Some(ActivePagesConfig::for_memory(
        std::env::temp_dir().join("flash-plan-only"),
        24u64 << 30,
    ));
    let tiered = FlashLoadPlan::inspect(Path::new(&path), &ram).unwrap();
    eprintln!(
        "SIMULATED 32GiB B1 128K: RAM reservation rejected, tiered peak={} admitted",
        tiered.residency.peak_bytes
    );
    ram.active_pages.as_mut().unwrap().pack_bytes = 4 << 20;
    assert!(FlashLoadPlan::inspect(Path::new(&path), &ram)
        .unwrap_err()
        .to_string()
        .contains("warm-history rollback"));
    ram.active_pages.as_mut().unwrap().pack_bytes = 8 << 20;
    ram.active_pages.as_mut().unwrap().cache_bytes = 16 << 20;
    assert!(FlashLoadPlan::inspect(Path::new(&path), &ram)
        .unwrap_err()
        .to_string()
        .contains("warm-history rollback"));
}
