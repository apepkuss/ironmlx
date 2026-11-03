//! Shared ANE policy. Hardware qualification is independent of GPU profiles,
//! draft decoding and the model currently selected in the App.
pub use crate::m5_profile::AneMode;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    OnceLock,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Dense27B,
    FlashNext,
    Other,
}
impl Model {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense27B => "qwen38-27b",
            Self::FlashNext => "qwen38-flash-next",
            Self::Other => "unadapted",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardware {
    pub chip: Option<String>,
    pub qualified: bool,
    pub available: bool,
    pub reason: &'static str,
    pub runtime_error: Option<String>,
}
/// Exact measured chip identities, not a GPU-generation or future-chip guess.
pub fn qualified_chip(chip: &str) -> bool {
    chip == "Apple M5 Max"
}
pub fn resolve_hardware(chip: Option<String>, runtime: Result<(), String>) -> Hardware {
    let qualified = chip.as_deref().is_some_and(qualified_chip);
    let available = qualified && runtime.is_ok();
    let reason = if chip.is_none() {
        "hardware_unknown"
    } else if !qualified {
        "machine_unqualified"
    } else if !available {
        "runtime_unavailable"
    } else {
        "eligible"
    };
    let runtime_error = if qualified { runtime.err() } else { None };
    Hardware {
        chip,
        qualified,
        available,
        reason,
        runtime_error,
    }
}
pub fn hardware() -> &'static Hardware {
    static HW: OnceLock<Hardware> = OnceLock::new();
    HW.get_or_init(|| {
        let chip = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|v| v.status.success())
            .and_then(|v| String::from_utf8(v.stdout).ok())
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty());
        // Unqualified devices never open the private runtime.
        let runtime = if chip.as_deref().is_some_and(qualified_chip) {
            ironmlx_ane::availability()
        } else {
            Err("not qualified".into())
        };
        resolve_hardware(chip, runtime)
    })
}
static FIXED_MODEL: OnceLock<Model> = OnceLock::new();
pub fn set_fixed_model(model: Model) {
    let _ = FIXED_MODEL.set(model);
}
pub fn fixed_model() -> Option<Model> {
    FIXED_MODEL.get().copied()
}
static MODE: OnceLock<AneMode> = OnceLock::new();
pub fn install(mode: AneMode) {
    let _ = MODE.set(mode);
}
pub fn mode() -> AneMode {
    MODE.get().copied().unwrap_or(AneMode::Auto)
}
pub fn model_adapted(model: Model) -> bool {
    model == Model::Dense27B
}
pub fn eligible(model: Model) -> bool {
    mode() == AneMode::Auto && model_adapted(model) && hardware().available && !retired(model)
}
/// Stable arithmetic contract; retirement does not change the cache namespace
/// within a process. A fresh GPU-only process cannot restore this namespace.
pub fn dense_execution_fingerprint() -> Option<String> {
    (mode() == AneMode::Auto && model_adapted(Model::Dense27B) && hardware().available).then(|| {
        format!(
            "ane-dense-w8a8-v2;units={};pipeline={};maxpos={}",
            configured_dense_setting(crate::m5_profile::settings::DENSE_PREFILL_ANE_FFN_UNITS)
                .unwrap(),
            configured_dense_setting(crate::m5_profile::settings::DENSE_PREFILL_ANE_FFN_PIPELINE)
                .unwrap(),
            configured_dense_setting(
                crate::m5_profile::settings::DENSE_PREFILL_ANE_FFN_MAX_POSITION
            )
            .unwrap()
        )
    })
}
static DENSE_FAILED: AtomicBool = AtomicBool::new(false);
static FLASH_FAILED: AtomicBool = AtomicBool::new(false);
static DENSE_SURFACES: AtomicU64 = AtomicU64::new(0);
static FLASH_SURFACES: AtomicU64 = AtomicU64::new(0);
pub fn retired(model: Model) -> bool {
    match model {
        Model::Dense27B => &DENSE_FAILED,
        Model::FlashNext => &FLASH_FAILED,
        Model::Other => return false,
    }
    .load(Ordering::Relaxed)
}
pub fn retire(model: Model) {
    match model {
        Model::Dense27B => &DENSE_FAILED,
        Model::FlashNext => &FLASH_FAILED,
        Model::Other => return,
    }
    .store(true, Ordering::Relaxed);
}
fn surface_counter(model: Model) -> &'static AtomicU64 {
    match model {
        Model::Dense27B => &DENSE_SURFACES,
        Model::FlashNext => &FLASH_SURFACES,
        Model::Other => panic!("no ANE context for unadapted model"),
    }
}
pub fn retain_surfaces(model: Model, bytes: usize) {
    surface_counter(model).fetch_add(bytes as u64, Ordering::Relaxed);
}
pub fn release_surfaces(model: Model, bytes: usize) {
    surface_counter(model).fetch_sub(bytes as u64, Ordering::Relaxed);
}
pub fn surface_bytes(model: Model) -> u64 {
    if model == Model::Other {
        0
    } else {
        surface_counter(model).load(Ordering::Relaxed)
    }
}
pub fn reason(model: Model) -> &'static str {
    if mode() == AneMode::Off {
        "disabled"
    } else if !hardware().available {
        hardware().reason
    } else if retired(model) {
        "runtime_failed"
    } else if !model_adapted(model) {
        "model_not_adapted"
    } else {
        "eligible"
    }
}
static DENSE_JOBS: AtomicU64 = AtomicU64::new(0);
static FLASH_JOBS: AtomicU64 = AtomicU64::new(0);
pub fn record_completed(model: Model) {
    match model {
        Model::Dense27B => &DENSE_JOBS,
        Model::FlashNext => &FLASH_JOBS,
        Model::Other => return,
    }
    .fetch_add(1, Ordering::Relaxed);
}
pub fn completed(model: Model) -> u64 {
    match model {
        Model::Dense27B => &DENSE_JOBS,
        Model::FlashNext => &FLASH_JOBS,
        Model::Other => return 0,
    }
    .load(Ordering::Relaxed)
}
/// Legacy variable names remain accepted, with policy and --ane off taking
/// precedence over explicit variables. Other model families never inherit them.
pub fn dense_setting(name: &str) -> Option<String> {
    let configured = configured_dense_setting(name)?;
    Some(if eligible(Model::Dense27B) {
        configured
    } else {
        "0".into()
    })
}
fn configured_dense_setting(name: &str) -> Option<String> {
    use crate::m5_profile::settings as s;
    // Half-chunk pipelining consumes the Dense tiled GPU layout. Ordinary
    // inference and profile-off use the independently qualified native split.
    let tiled = crate::m5_profile::installed().is_some_and(|p| {
        p.is_active() && p.target == crate::m5_profile::M5ProfileTarget::Qwen35Dense
    });
    let fallback = match name {
        s::DENSE_PREFILL_ANE_FFN_UNITS => {
            if tiled {
                "12"
            } else {
                "9"
            }
        }
        s::DENSE_PREFILL_ANE_FFN_PIPELINE => {
            if tiled {
                "1"
            } else {
                "0"
            }
        }
        s::DENSE_PREFILL_ANE_FFN_MAX_POSITION => "65536",
        _ => return None,
    };
    Some(std::env::var(name).unwrap_or_else(|_| fallback.into()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualification_does_not_guess_from_gpu_generation() {
        for chip in [
            "Apple M1 Pro",
            "Apple M2 Max",
            "Apple M3 Max",
            "Apple M4 Max",
            "Apple M5",
            "Apple M5 Pro",
            "Apple M6 Max",
            "",
        ] {
            let h = resolve_hardware(Some(chip.into()), Ok(()));
            assert!(!h.qualified && !h.available);
        }
        assert!(resolve_hardware(Some("Apple M5 Max".into()), Ok(())).available);
        assert_eq!(resolve_hardware(None, Ok(())).reason, "hardware_unknown");
        let h = resolve_hardware(Some("Apple M5 Max".into()), Err("ABI mismatch".into()));
        assert!(h.qualified && !h.available);
        assert_eq!(h.reason, "runtime_unavailable");
    }
    #[test]
    fn model_qualifications_are_separate() {
        assert!(model_adapted(Model::Dense27B));
        assert!(!model_adapted(Model::FlashNext));
        assert!(!model_adapted(Model::Other));
    }
}
