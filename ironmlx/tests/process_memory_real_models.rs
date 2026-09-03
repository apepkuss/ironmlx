#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ironmlx::core::process_memory::{
    global_process_memory_governor, native_memory_telemetry, ColdMaterializationTracker,
    MaterializationComponents, StaticMemoryEstimate,
};
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::Loader;
use mlx::Array;

fn fixture_image() -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_vl/coco_sample.jpg"),
    )
    .expect("read real image fixture")
}

fn model_path(env_name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var(env_name)
            .unwrap_or_else(|_| panic!("{env_name} must point to a real multimodal checkpoint")),
    );
    assert!(
        path.join("config.json").is_file(),
        "invalid model path: {path:?}"
    );
    path
}

fn calibrate<M: DenseVlMethods>(
    label: &str,
    model: &M,
    vision_cold_bytes: usize,
    pixel_values: &[Array],
    grid_thw: &[(i32, i32, i32)],
) {
    let estimated_bytes = model
        .estimate_vision_prefill_peak_bytes(pixel_values, grid_thw)
        .expect("architecture-aware vision peak estimate");
    assert!(estimated_bytes > 0 && estimated_bytes != usize::MAX);

    // The loader eagerly evaluates graph transformations, but mmap-backed raw
    // weights are first physically touched by an actual tower execution. Keep
    // that cold materialization delta separate from the steady-state transient
    // peak that the per-request estimator is responsible for bounding.
    mlx::transforms::clear_cache();
    let cold_tracker = ColdMaterializationTracker::new(StaticMemoryEstimate {
        vision_cold_bytes,
        ..StaticMemoryEstimate::default()
    });
    let cold_guard = cold_tracker
        .begin(
            MaterializationComponents {
                text: false,
                vision: true,
                speculative: false,
            },
            &global_process_memory_governor(),
        )
        .expect("reserve real cold vision liability");
    let cold_cache_limit_bytes = global_process_memory_governor()
        .snapshot()
        .mlx_cache_limit_bytes;
    let cold_baseline_mlx = mlx::memory::snapshot();
    let cold_baseline = native_memory_telemetry()
        .current_usage_bytes()
        .expect("authoritative cold baseline usage");
    let cold_peak = Arc::new(AtomicUsize::new(cold_baseline));
    let cold_stop = Arc::new(AtomicBool::new(false));
    let sampler_peak = Arc::clone(&cold_peak);
    let sampler_stop = Arc::clone(&cold_stop);
    let cold_sampler = std::thread::spawn(move || {
        while !sampler_stop.load(Ordering::Acquire) {
            if let Some(usage) = native_memory_telemetry().current_usage_bytes() {
                sampler_peak.fetch_max(usage, Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    let warmup = model
        .compute_vision_embeds(pixel_values, grid_thw, mlx::StreamOrDevice::default())
        .expect("construct cold real vision graph");
    mlx::transforms::eval(&[&warmup]).expect("materialize cold real vision graph");
    if let Some(usage) = native_memory_telemetry().current_usage_bytes() {
        cold_peak.fetch_max(usage, Ordering::Relaxed);
    }
    cold_stop.store(true, Ordering::Release);
    cold_sampler.join().expect("join cold footprint sampler");
    cold_guard.commit();
    let cold_after_mlx = mlx::memory::snapshot();
    let cold_materialization_growth_bytes = cold_peak
        .load(Ordering::Relaxed)
        .saturating_sub(cold_baseline);
    drop(warmup);
    mlx::transforms::clear_cache();

    let governor = global_process_memory_governor();
    let transient_reservation = governor
        .try_reserve_prefill(estimated_bytes, "real_vision_prefill")
        .expect("reserve transient peak and possible MLX cache growth");
    let prefill_reserved_bytes = transient_reservation.bytes();
    let cache_growth_liability_bytes = prefill_reserved_bytes.saturating_sub(estimated_bytes);
    let baseline_telemetry = native_memory_telemetry();
    let baseline_usage = baseline_telemetry
        .current_usage_bytes()
        .expect("authoritative baseline usage");
    let baseline_mlx = mlx::memory::snapshot();
    let peak_usage = Arc::new(AtomicUsize::new(baseline_usage));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler_peak = Arc::clone(&peak_usage);
    let sampler_stop = Arc::clone(&stop);
    let sampler = std::thread::spawn(move || {
        while !sampler_stop.load(Ordering::Acquire) {
            if let Some(usage) = native_memory_telemetry().current_usage_bytes() {
                sampler_peak.fetch_max(usage, Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    let started = Instant::now();
    let embeds = model
        .compute_vision_embeds(pixel_values, grid_thw, mlx::StreamOrDevice::default())
        .expect("construct real vision graph");
    mlx::transforms::eval(&[&embeds]).expect("materialize real vision graph");
    let elapsed = started.elapsed();
    if let Some(usage) = native_memory_telemetry().current_usage_bytes() {
        peak_usage.fetch_max(usage, Ordering::Relaxed);
    }
    stop.store(true, Ordering::Release);
    sampler.join().expect("join footprint sampler");
    transient_reservation.commit();

    let after_mlx = mlx::memory::snapshot();
    let observed_peak_bytes = peak_usage
        .load(Ordering::Relaxed)
        .saturating_sub(baseline_usage);
    let mlx_active_growth_bytes = after_mlx
        .active_bytes
        .saturating_sub(baseline_mlx.active_bytes);
    println!(
        "{{\"model\":\"{label}\",\"estimated_bytes\":{estimated_bytes},\"cache_growth_liability_bytes\":{cache_growth_liability_bytes},\"prefill_reserved_bytes\":{prefill_reserved_bytes},\"vision_cold_bytes\":{vision_cold_bytes},\"cold_cache_limit_bytes\":{cold_cache_limit_bytes},\"observed_peak_bytes\":{observed_peak_bytes},\"cold_materialization_growth_bytes\":{cold_materialization_growth_bytes},\"cold_baseline_mlx_active_bytes\":{},\"cold_after_mlx_active_bytes\":{},\"cold_mlx_active_growth_bytes\":{},\"cold_baseline_mlx_cache_bytes\":{},\"cold_after_mlx_cache_bytes\":{},\"baseline_usage_bytes\":{baseline_usage},\"baseline_mlx_active_bytes\":{},\"after_mlx_active_bytes\":{},\"mlx_active_growth_bytes\":{mlx_active_growth_bytes},\"elapsed_ms\":{},\"grid_thw\":\"{grid_thw:?}\"}}",
        cold_baseline_mlx.active_bytes,
        cold_after_mlx.active_bytes,
        cold_after_mlx
            .active_bytes
            .saturating_sub(cold_baseline_mlx.active_bytes),
        cold_baseline_mlx.cache_bytes,
        cold_after_mlx.cache_bytes,
        baseline_mlx.active_bytes,
        after_mlx.active_bytes,
        elapsed.as_millis()
    );
    assert!(
        observed_peak_bytes > 0,
        "real vision run produced no footprint growth"
    );
    assert!(
        observed_peak_bytes <= prefill_reserved_bytes,
        "{label} prefill admission under-reserved: observed={observed_peak_bytes} transient={estimated_bytes} cache_liability={cache_growth_liability_bytes} reserved={prefill_reserved_bytes}"
    );
    assert!(
        cold_materialization_growth_bytes
            <= vision_cold_bytes.saturating_add(estimated_bytes),
        "{label} cold admission under-reserved: observed={cold_materialization_growth_bytes} cold={vision_cold_bytes} transient={estimated_bytes}"
    );
    drop(embeds);
    mlx::transforms::clear_cache();
}

fn vision_tensor_bytes(loader: &Loader) -> usize {
    const PREFIXES: &[&str] = &[
        "vision_tower.",
        "vision_embedder.",
        "embed_vision.",
        "model.encoder.vision_tower.",
        "model.encoder.embed_vision.",
        "vit_merger.",
        "merger.",
    ];
    loader
        .keys()
        .filter(|key| PREFIXES.iter().any(|prefix| key.starts_with(prefix)))
        .map(|key| {
            let tensor = loader.tensor(key).expect("existing loader tensor");
            tensor.size().saturating_mul(tensor.dtype().byte_size())
        })
        .fold(0usize, usize::saturating_add)
}

#[test]
#[ignore = "requires IRONMLX_MEMORY_GOVERNOR_QWEN_MODEL"]
fn qwen35_real_vision_peak_is_covered() {
    let loader = Loader::open_multimodal(&model_path("IRONMLX_MEMORY_GOVERNOR_QWEN_MODEL"))
        .expect("open Qwen3.5 multimodal checkpoint");
    let model =
        ironmlx::models::qwen3_5::Qwen35Model::from_loader(&loader).expect("load Qwen3.5 model");
    let (pixels, grid_h, grid_w) =
        ironmlx::models::qwen3_5::image_processor::preprocess(&fixture_image())
            .expect("preprocess Qwen3.5 image");
    calibrate(
        "qwen3.5",
        &model,
        vision_tensor_bytes(&loader),
        &[pixels],
        &[(1, grid_h, grid_w)],
    );
}

#[test]
#[ignore = "requires IRONMLX_MEMORY_GOVERNOR_GEMMA4_MODEL"]
fn gemma4_real_vision_peak_is_covered() {
    let loader = Loader::open_multimodal(&model_path("IRONMLX_MEMORY_GOVERNOR_GEMMA4_MODEL"))
        .expect("open Gemma4 multimodal checkpoint");
    let config =
        ironmlx::models::gemma4::Gemma4Config::from_loader(&loader).expect("parse Gemma4 config");
    let vision_config = config.vision_config.as_ref().expect("Gemma4 vision config");
    let processed =
        ironmlx::models::gemma4::image_processor::preprocess(&fixture_image(), vision_config)
            .expect("preprocess Gemma4 image");
    let model = ironmlx::models::gemma4::Gemma4Model::from_loader_with_config(&loader, config)
        .expect("load Gemma4 model");
    calibrate(
        "gemma4",
        &model,
        vision_tensor_bytes(&loader),
        &[processed.pixel_values],
        &[(1, processed.grid_h, processed.grid_w)],
    );
}

#[test]
#[ignore = "requires IRONMLX_MEMORY_GOVERNOR_MINICPM_MODEL"]
fn minicpmv46_real_vision_peak_is_covered() {
    let loader = Loader::open_multimodal(&model_path("IRONMLX_MEMORY_GOVERNOR_MINICPM_MODEL"))
        .expect("open MiniCPM-V-4.6 multimodal checkpoint");
    let model = ironmlx::models::minicpmv4_6::MiniCpmV46Model::from_loader(&loader)
        .expect("load MiniCPM-V-4.6 model");
    let (pixels, grid_h, grid_w) =
        ironmlx::models::minicpmv4_6::image_processor::preprocess(&fixture_image())
            .expect("preprocess MiniCPM-V-4.6 image");
    calibrate(
        "minicpm-v-4.6",
        &model,
        vision_tensor_bytes(&loader),
        &[pixels],
        &[(1, grid_h, grid_w)],
    );
}
