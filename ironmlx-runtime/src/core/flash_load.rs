//! Flash load admission shared by the native App engine and CLI entrypoints.
use anyhow::Result;
use ironmlx_lm::models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest};
use std::path::Path;

pub fn is_flash(path: &Path) -> Result<bool> {
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path.join("config.json"))?)?;
    Ok(config["model_type"] == "qwen4_exp")
}
pub fn plan(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
) -> Result<FlashLoadPlan> {
    plan_with_occupied(path, batch, cap, chunk, draft_bytes, limit, 0)
}

/// EnginePool's planned resident ceilings include bounded expert/PLE caches
/// which may not have materialized yet. The process sample remains a floor.
pub(crate) fn available_bytes(minimum_occupied: usize, limit: Option<usize>) -> Result<usize> {
    let physical = super::memory_budget::system_total_ram_bytes();
    let memory = mlx::memory::snapshot();
    let telemetry = super::process_memory::global_process_memory_governor().sample_process();
    anyhow::ensure!(
        !telemetry.telemetry_degraded
            && !matches!(
                telemetry.pressure_level,
                super::process_memory::PressureLevel::Hard
                    | super::process_memory::PressureLevel::Emergency
            ),
        "Flash loading requires available process and host memory telemetry without hard pressure"
    );
    // Both observed process use (other loaded models) and outstanding
    // reservations reduce this load's envelope. Never count reclaimable
    // filesystem pages as an additional copy of physical memory.
    let occupied = telemetry
        .current_usage_bytes
        .max(memory.active_bytes)
        .max(minimum_occupied)
        .saturating_add(telemetry.reserved_bytes);
    let ceiling = physical
        .saturating_sub(8usize << 30)
        .min(
            memory
                .max_recommended_bytes
                .unwrap_or(physical)
                .saturating_sub(2usize << 30),
        )
        .min(if telemetry.hard_watermark_bytes > 0 {
            telemetry.hard_watermark_bytes
        } else {
            physical
        });
    let available = ceiling
        .saturating_sub(occupied)
        .min(limit.unwrap_or(usize::MAX));
    Ok(available)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_with_occupied(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
    minimum_occupied: usize,
) -> Result<FlashLoadPlan> {
    plan_with_active_occupied(
        path,
        batch,
        cap,
        chunk,
        draft_bytes,
        limit,
        minimum_occupied,
        &super::cache::ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_with_active_occupied(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
    minimum_occupied: usize,
    active: &super::cache::ActiveKvOffloadConfig,
) -> Result<FlashLoadPlan> {
    let available = available_bytes(minimum_occupied, limit)?;
    let mut request = FlashLoadRequest::new(available as u64, batch, cap, chunk, draft_bytes);
    if active.enabled {
        request.active_pages = Some(
            ironmlx_lm::core::active_pages::ActivePagesConfig::for_memory(
                active.root.join("flash"),
                available as u64,
            ),
        );
    }
    FlashLoadPlan::inspect(path, &request.with_environment()?)
}
