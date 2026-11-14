//! Keeping model weights GPU-resident.
//!
//! macOS may compress memory that is not resident. When memory is tight it
//! takes the weights between requests, and every decode window then has to
//! decompress them again: generation slows several times over. Resident
//! weights cannot be taken.
//!
//! Engines loaded through a [`Loader`](ironmlx_lm::core::Loader) pin exactly
//! their weight buffers (`mlx::memory::pin_resident`); the pin ends by itself
//! when a buffer is freed, so unloading a model needs no bookkeeping. The
//! DFlash2 server, which loads one model for the life of the process, uses the
//! wired limit instead ([`record_wired_limit`]).

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
    /// The DFlash2 server's wired limit, 0 elsewhere.
    pub wired_limit_bytes: usize,
    /// What each model's most recent load decided. A model stays listed
    /// after it is unloaded; `pinned_bytes` is what is resident now.
    pub last_load: BTreeMap<String, ResidentModel>,
}

#[derive(Default)]
struct Registry {
    wired_limit_bytes: usize,
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

fn record(model_id: &str, model: ResidentModel) {
    let mut registry = registry();
    registry.models.insert(model_id.to_owned(), model);
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
    let max_recommended_bytes = mlx::memory::snapshot().max_recommended_bytes.unwrap_or(0);
    let mut decision = decide(
        mode,
        weight_bytes,
        mlx::memory::pinned_bytes(),
        max_recommended_bytes,
    );
    let mut resident_bytes = 0;
    if decision == ResidentDecision::Resident {
        let arrays: Vec<&mlx::Array> = weights.into_iter().collect();
        match mlx::memory::pin_resident(&arrays) {
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
        "resident weights"
    );
    record(
        model_id,
        ResidentModel {
            decision,
            weight_bytes,
            resident_bytes,
        },
    );
    decision
}

/// The DFlash2 server made `limit_bytes` resident through the wired limit
/// (`decision` says whether it did).
pub fn record_wired_limit(model_id: &str, decision: ResidentDecision, limit_bytes: usize) {
    let resident_bytes = if decision == ResidentDecision::Resident {
        limit_bytes
    } else {
        0
    };
    registry().wired_limit_bytes = resident_bytes;
    record(
        model_id,
        ResidentModel {
            decision,
            weight_bytes: limit_bytes,
            resident_bytes,
        },
    );
    if decision == ResidentDecision::Resident {
        start_refresh();
    }
}

pub fn health() -> ResidentWeightsHealth {
    let registry = registry();
    ResidentWeightsHealth {
        pinned_bytes: mlx::memory::pinned_bytes(),
        wired_limit_bytes: registry.wired_limit_bytes,
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
