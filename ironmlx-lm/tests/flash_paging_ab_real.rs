//! Fixed-trajectory paging controls; cache/reader policies are selected per process.
use ironmlx_lm::core::Model;
use ironmlx_lm::models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest};
use mlx::Array;
use std::{path::Path, time::Instant};

fn exact(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(
        a.equal(b)
            .unwrap()
            .reshape([-1])
            .unwrap()
            .all(-1, false)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true]
    );
}

#[test]
#[ignore = "requires actual Flash snapshot; scan retention and I/O controls, not cold-drive proof"]
fn flash_paging_scan_and_replay_controls() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let cache_mib = std::env::var("FLASH_AB_EXPERT_MIB")
        .ok()
        .map(|v| v.parse::<u64>().unwrap())
        .unwrap_or(4096);
    let scan_tokens = std::env::var("FLASH_AB_SCAN_TOKENS")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(2048);
    mlx::memory::set_cache_limit(128 << 20);
    let mut request = FlashLoadRequest::new(48 << 30, 1, 4096, scan_tokens, 0);
    request.expert_residency = ironmlx_lm::core::residency::ExpertResidency::Ssd;
    request.expert_cache_limit_bytes = cache_mib << 20;
    let plan = FlashLoadPlan::inspect(Path::new(&path), &request).unwrap();
    let loader = plan.open_loader(Path::new(&path)).unwrap();
    let model = plan.build_model(&loader).unwrap();
    drop(loader);
    let mut row = model.new_cache(4096).unwrap();
    let seed: Vec<_> = (0..63).map(|n| 1000 + n * 13).collect();
    model
        .forward_hidden_tokens_on(&seed, &mut row, &[], ().into())
        .unwrap();
    let before = row.snapshot();
    let reference = model
        .forward_tokens_on(&[701], &mut row, &[], ().into())
        .unwrap()
        .logits;
    row.restore(&before).unwrap();
    let hot = model
        .forward_tokens_on(&[701], &mut row, &[], ().into())
        .unwrap()
        .logits;
    exact(&reference, &hot);
    let telemetry = model.paging_telemetry().unwrap();
    let stats_before_scan = telemetry.snapshot();
    let scan: Vec<_> = (0..scan_tokens)
        .map(|n| 10000 + ((n * 17) % 10000) as u32)
        .collect();
    let mut scanner = model.new_cache(4096).unwrap();
    let clock = Instant::now();
    model
        .forward_hidden_tokens_on(&scan, &mut scanner, &[], ().into())
        .unwrap()
        .hidden
        .eval()
        .unwrap();
    let scan_seconds = clock.elapsed().as_secs_f64();
    let stats_after_scan = telemetry.snapshot();
    row.restore(&before).unwrap();
    let clock = Instant::now();
    let replay = model
        .forward_tokens_on(&[701], &mut row, &[], ().into())
        .unwrap()
        .logits;
    let replay_seconds = clock.elapsed().as_secs_f64();
    exact(&reference, &replay);
    let stats_after_replay = telemetry.snapshot();
    eprintln!(
        "PAGING_AB={}",
        serde_json::json!({
            "scan_seconds": scan_seconds, "replay_seconds": replay_seconds,
            "before_scan": stats_before_scan, "after_scan": stats_after_scan,
            "after_replay": stats_after_replay, "mlx_peak_bytes": mlx::memory::snapshot().peak_bytes,
            "scan_tokens": scan_tokens, "cache_mib": cache_mib,
        })
    );
}
