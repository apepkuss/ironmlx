//! Keeping model weights GPU-resident.
//!
//! macOS may compress memory that is not resident. When memory is tight it
//! takes the weights between requests, and every decode window then has to
//! decompress them again: generation slows several times over. Resident
//! weights cannot be taken.
//!
//! Engines loaded through a [`Loader`](ironmlx_lm::core::Loader) pin exactly
//! their weight buffers (`mlx::memory::pin_resident`). The DFlash2 server
//! pins every buffer allocated after its prewarm ([`pin_all_allocated`]):
//! only the weights are allocated then, including the copies its models
//! derive from the checkpoint. Either way a pin ends by itself when its
//! buffer is freed, so unloading a model needs no bookkeeping.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use serde::Serialize;

/// Interval at which the residency request is renewed: macOS drops residency
/// within seconds of GPU idleness despite a standing request.
pub const RESIDENCY_REFRESH_MS: u32 = 500;

/// `--resident-weights`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidentWeightsMode {
    /// Resident when the weights fit ([`fits_auto`]).
    #[default]
    Auto,
    /// Resident whatever their size.
    On,
    Off,
}

impl std::str::FromStr for ResidentWeightsMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => Err(format!("invalid value {other:?}: expected auto, on or off")),
        }
    }
}

/// What was decided for one model's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidentDecision {
    Resident,
    /// `--resident-weights off`.
    Off,
    /// `auto`, and the weights are too large a share of the working set.
    TooLarge,
    /// The request to MLX failed; the weights are not resident.
    Failed,
}

/// In `auto`, weights stay resident only while everything resident is at most
/// two thirds of the device's recommended working set: resident memory cannot
/// be compressed or swapped, so the rest of the machine needs the remainder.
pub fn fits_auto(resident_bytes: usize, max_recommended_bytes: usize) -> bool {
    max_recommended_bytes > 0 && (resident_bytes as u128) * 3 <= (max_recommended_bytes as u128) * 2
}

/// `already_resident_bytes`: what earlier loads made resident.
pub fn decide(
    mode: ResidentWeightsMode,
    weight_bytes: usize,
    already_resident_bytes: usize,
    max_recommended_bytes: usize,
) -> ResidentDecision {
    match mode {
        ResidentWeightsMode::Off => ResidentDecision::Off,
        ResidentWeightsMode::On => ResidentDecision::Resident,
        ResidentWeightsMode::Auto => {
            if fits_auto(
                already_resident_bytes.saturating_add(weight_bytes),
                max_recommended_bytes,
            ) {
                ResidentDecision::Resident
            } else {
                ResidentDecision::TooLarge
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResidentModel {
    pub decision: ResidentDecision,
    pub weight_bytes: usize,
    /// Bytes this load made resident. Smaller than `weight_bytes` when the
    /// model keeps part of its weights in buffers of its own.
    pub resident_bytes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ResidentWeightsHealth {
    /// Bytes pinned right now; falls when a pinned model is unloaded.
    pub pinned_bytes: usize,
    /// What each model's most recent load decided. A model stays listed
    /// after it is unloaded; `pinned_bytes` is what is resident now.
    pub last_load: BTreeMap<String, ResidentModel>,
}

#[derive(Default)]
struct Registry {
    /// The mode of the most recent load, which weights loaded later (a
    /// vision encoder on its first image) follow.
    mode: Option<ResidentWeightsMode>,
    models: BTreeMap<String, ResidentModel>,
}

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn start_refresh() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(
        || match mlx::memory::start_residency_refresh(RESIDENCY_REFRESH_MS) {
            Ok(()) => tracing::info!(
                interval_ms = RESIDENCY_REFRESH_MS,
                "weight residency refresh started"
            ),
            Err(error) => tracing::warn!(%error, "weight residency refresh not started"),
        },
    );
}

fn record(model_id: &str, mode: ResidentWeightsMode, model: ResidentModel) {
    let mut registry = registry();
    registry.mode = Some(mode);
    registry.models.insert(model_id.to_owned(), model);
}

/// Decide for `weight_bytes` and, when they are to be resident, `pin` them.
fn pin_with(
    mode: ResidentWeightsMode,
    model_id: &str,
    weight_bytes: usize,
    pin: impl FnOnce() -> mlx::Result<usize>,
) -> ResidentDecision {
    let max_recommended_bytes = mlx::memory::snapshot().max_recommended_bytes.unwrap_or(0);
    let mut decision = decide(
        mode,
        weight_bytes,
        mlx::memory::pinned_bytes(),
        max_recommended_bytes,
    );
    let mut resident_bytes = 0;
    let mut pin_us = 0;
    if decision == ResidentDecision::Resident {
        let started = std::time::Instant::now();
        let pinned = pin();
        pin_us = started.elapsed().as_micros() as u64;
        match pinned {
            Ok(bytes) => {
                resident_bytes = bytes;
                start_refresh();
            }
            Err(error) => {
                tracing::warn!(model = model_id, %error, "weights not made resident");
                decision = ResidentDecision::Failed;
            }
        }
    }
    tracing::info!(
        model = model_id,
        ?decision,
        weight_bytes,
        resident_bytes,
        max_recommended_bytes,
        pin_us,
        "resident weights"
    );
    record(
        model_id,
        mode,
        ResidentModel {
            decision,
            weight_bytes,
            resident_bytes,
        },
    );
    decision
}

/// Pin the weights a loader has just evaluated, before or after the model is
/// built from it: the model's layers share these buffers. `model_id` names
/// the load in [`health`] (a model's draft or MTP head gets a name of its
/// own).
pub fn pin_loaded_weights<'a>(
    mode: ResidentWeightsMode,
    model_id: &str,
    weights: impl IntoIterator<Item = &'a mlx::Array>,
    weight_bytes: usize,
) -> ResidentDecision {
    pin_with(mode, model_id, weight_bytes, || {
        let arrays: Vec<&mlx::Array> = weights.into_iter().collect();
        mlx::memory::pin_resident(&arrays)
    })
}

/// Pin every buffer allocated now, after releasing the allocator cache: for
/// a server whose only allocations at this point are its models' weights,
/// including the copies they derived from the checkpoint at prewarm.
pub fn pin_all_allocated(mode: ResidentWeightsMode, model_id: &str) -> ResidentDecision {
    mlx::clear_cache();
    let weight_bytes = mlx::memory::snapshot().active_bytes;
    pin_with(mode, model_id, weight_bytes, mlx::memory::pin_all_resident)
}

/// After a warmup, pin again everything allocated for `model_id` when its
/// load made it resident: a model may replace weights with layouts it builds
/// on its first forward (the M5 affine4 tiles), and the pins of the replaced
/// buffers ended when they were freed. Nothing for a model not recorded as
/// resident.
pub fn repin_all_allocated(model_id: &str) {
    let resident = registry()
        .models
        .get(model_id)
        .is_some_and(|model| model.decision == ResidentDecision::Resident);
    if !resident {
        return;
    }
    mlx::clear_cache();
    let started = std::time::Instant::now();
    match mlx::memory::pin_all_resident() {
        Ok(bytes) => {
            let pinned_bytes = mlx::memory::pinned_bytes();
            tracing::info!(
                model = model_id,
                newly_pinned_bytes = bytes,
                pinned_bytes,
                active_bytes = mlx::memory::snapshot().active_bytes,
                pin_us = started.elapsed().as_micros() as u64,
                "resident weights pinned again after warmup"
            );
            if let Some(model) = registry().models.get_mut(model_id) {
                model.resident_bytes = pinned_bytes;
            }
        }
        Err(error) => tracing::warn!(model = model_id, %error, "weights not pinned again"),
    }
}

/// Pin weights a model loads after it started serving (a vision encoder on
/// its first image) under the mode of the most recent load; nothing when no
/// load has been recorded.
pub fn pin_later_weights<'a>(
    model_id: &str,
    weights: impl IntoIterator<Item = &'a mlx::Array>,
    weight_bytes: usize,
) -> ResidentDecision {
    let Some(mode) = registry().mode else {
        return ResidentDecision::Off;
    };
    pin_loaded_weights(mode, model_id, weights, weight_bytes)
}

pub fn health() -> ResidentWeightsHealth {
    let registry = registry();
    ResidentWeightsHealth {
        pinned_bytes: mlx::memory::pinned_bytes(),
        last_load: registry.models.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    #[test]
    fn auto_keeps_weights_resident_up_to_two_thirds_of_the_working_set() {
        // A 32 GB Mac: the recommended working set is about 26.8 GB.
        let max = 26_800_603_136;
        assert!(fits_auto(16_398_820_992, max));
        assert!(fits_auto(max / 3 * 2, max));
        assert!(!fits_auto(max / 3 * 2 + 3, max));
        assert!(!fits_auto(20 * GIB, max));
        // Unknown working set: nothing is assumed to fit.
        assert!(!fits_auto(GIB, 0));
    }

    #[test]
    fn decision_follows_the_mode_and_counts_what_is_already_resident() {
        let max = 30 * GIB;
        use ResidentDecision::{Off, Resident, TooLarge};
        use ResidentWeightsMode as Mode;
        assert_eq!(decide(Mode::Auto, 16 * GIB, 0, max), Resident);
        assert_eq!(decide(Mode::Auto, 21 * GIB, 0, max), TooLarge);
        // A second model: both together must fit.
        assert_eq!(decide(Mode::Auto, 8 * GIB, 16 * GIB, max), TooLarge);
        assert_eq!(decide(Mode::Auto, 4 * GIB, 16 * GIB, max), Resident);
        assert_eq!(decide(Mode::On, 29 * GIB, 16 * GIB, max), Resident);
        assert_eq!(decide(Mode::Off, GIB, 0, max), Off);
    }

    #[test]
    fn mode_parses_its_three_values() {
        assert_eq!("auto".parse(), Ok(ResidentWeightsMode::Auto));
        assert_eq!("on".parse(), Ok(ResidentWeightsMode::On));
        assert_eq!("off".parse(), Ok(ResidentWeightsMode::Off));
        assert!("yes".parse::<ResidentWeightsMode>().is_err());
    }
}
