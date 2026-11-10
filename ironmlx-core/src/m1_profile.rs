//! Hardware-qualified, engine-owned M1 GPU plans. Never installs global settings.
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum M1ProfileMode {
    Off,
    #[default]
    Auto,
}

impl std::str::FromStr for M1ProfileMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            _ => Err("M1 profile must be auto or off".into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum M1Device {
    Pro,
    Max,
}

impl M1Device {
    /// Real device identity, cached independently of the profile mode. Reading
    /// metadata does not prepare any M1 kernels, weights or GPU workspaces.
    pub fn current() -> Option<Self> {
        static DEVICE: OnceLock<Option<M1Device>> = OnceLock::new();
        *DEVICE.get_or_init(|| {
            let architecture = mlx::metal::architecture().ok()?;
            if !architecture.starts_with("applegpu_g13") {
                return None;
            }
            let memory = mlx::memory::snapshot();
            Self::detect(memory.device_name.as_deref()?, &architecture)
        })
    }

    pub fn detect(name: &str, architecture: &str) -> Option<Self> {
        // Check the real Metal architecture too: a name/diagnostic override
        // alone must never route M5 (or another GPU) to this plan.
        let suffix = architecture.strip_prefix("applegpu_g13")?;
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        match name {
            "Apple M1 Pro" => Some(Self::Pro),
            "Apple M1 Max" => Some(Self::Max),
            _ => None,
        }
    }

    pub fn qmv_rows_per_group(self) -> i32 {
        match self {
            Self::Pro => 2,
            Self::Max => 4,
        }
    }
}

#[derive(Debug)]
pub struct M1ExecutionPlan {
    pub device: Option<M1Device>,
    pub status: &'static str,
    narrow: AtomicU64,
    prefill: AtomicU64,
    fallback: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct M1PlanSnapshot {
    pub status: &'static str,
    pub device: Option<M1Device>,
    pub fingerprint: String,
    pub qmv_rows_per_group: Option<i32>,
    pub narrow_projection_calls: u64,
    pub prefill_projection_calls: u64,
    pub fallback_projection_calls: u64,
    pub ane: &'static str,
}

impl M1ExecutionPlan {
    /// Unsupported hardware owns no M1 plan. On M1 Pro/Max an explicit `off`
    /// remains observable without activating dispatch or cache namespaces.
    pub fn resolve_for_mode(
        mode: M1ProfileMode,
        qualified: bool,
        device: Option<M1Device>,
    ) -> Option<Self> {
        let device = device?;
        let mut plan = Self::resolve(qualified, Some(device));
        if mode == M1ProfileMode::Off {
            plan.status = "off";
        }
        Some(plan)
    }

    pub fn resolve(qualified: bool, device: Option<M1Device>) -> Self {
        Self {
            device,
            status: if device.is_none() {
                "unsupported_device"
            } else if !qualified {
                "unsupported_model"
            } else {
                "active"
            },
            narrow: AtomicU64::new(0),
            prefill: AtomicU64::new(0),
            fallback: AtomicU64::new(0),
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == "active"
    }

    pub fn record_narrow(&self) {
        self.narrow.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_prefill(&self) {
        self.prefill.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_fallback(&self) {
        self.fallback.fetch_add(1, Ordering::Relaxed);
    }
    pub fn fingerprint(&self) -> String {
        format!(
            "m1-gpu-v1;device={:?};status={};qmv=mlx-product-stable;nv={};prefill=mlx-affine4;glue=exact;ane=off",
            self.device, self.status, self.device.map_or(0, M1Device::qmv_rows_per_group)
        )
    }
    pub fn snapshot(&self) -> M1PlanSnapshot {
        M1PlanSnapshot {
            status: self.status,
            device: self.device,
            fingerprint: self.fingerprint(),
            qmv_rows_per_group: self.device.map(M1Device::qmv_rows_per_group),
            narrow_projection_calls: self.narrow.load(Ordering::Relaxed),
            prefill_projection_calls: self.prefill.load(Ordering::Relaxed),
            fallback_projection_calls: self.fallback.load(Ordering::Relaxed),
            ane: "not_implemented",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_real_m1_pro_and_max_are_eligible() {
        assert_eq!(
            M1Device::detect("Apple M1 Pro", "applegpu_g13p"),
            Some(M1Device::Pro)
        );
        assert_eq!(
            M1Device::detect("Apple M1 Max", "applegpu_g13s"),
            Some(M1Device::Max)
        );
        for (name, arch) in [
            ("Apple M5 Max", "applegpu_g17s"),
            ("Apple M1 Pro", "applegpu_g17s"),
            ("Apple M1", "applegpu_g13g"),
            ("Apple M1 Max", "applegpu_g130s"),
            ("Apple M2 Pro", "applegpu_g14p"),
        ] {
            assert_eq!(M1Device::detect(name, arch), None);
        }
    }
    #[test]
    fn plans_keep_counters_and_fingerprints_independent() {
        let pro = M1ExecutionPlan::resolve(true, Some(M1Device::Pro));
        let max = M1ExecutionPlan::resolve(true, Some(M1Device::Max));
        pro.record_narrow();
        assert_eq!(pro.snapshot().narrow_projection_calls, 1);
        assert_eq!(max.snapshot().narrow_projection_calls, 0);
        assert_ne!(pro.fingerprint(), max.fingerprint());
        assert!(!M1ExecutionPlan::resolve(false, Some(M1Device::Pro)).is_active());
        assert!(!M1ExecutionPlan::resolve(true, None).is_active());
        assert_eq!(M1ProfileMode::default(), M1ProfileMode::Auto);
    }

    #[test]
    fn hardware_gates_plans_independently_of_mode_and_model() {
        for mode in [M1ProfileMode::Auto, M1ProfileMode::Off] {
            for qualified in [false, true] {
                assert!(M1ExecutionPlan::resolve_for_mode(mode, qualified, None).is_none());
                for device in [M1Device::Pro, M1Device::Max] {
                    let plan =
                        M1ExecutionPlan::resolve_for_mode(mode, qualified, Some(device)).unwrap();
                    assert_eq!(plan.device, Some(device));
                    assert_eq!(plan.is_active(), mode == M1ProfileMode::Auto && qualified);
                    assert_eq!(
                        plan.status,
                        if mode == M1ProfileMode::Off {
                            "off"
                        } else if qualified {
                            "active"
                        } else {
                            "unsupported_model"
                        }
                    );
                }
            }
        }
    }
}
