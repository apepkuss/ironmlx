//! Capacity planning before weights are materialized. The caller supplies
//! usable physical/process memory after system and other-model reservations.
//! A plan is not admission to an unadapted loader: the model must implement
//! bounded paging and provide its minimum complete expert working set.
use anyhow::{anyhow, ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertResidency {
    Auto,
    Full,
    Ssd,
}

#[derive(Clone, Debug)]
pub struct ResidencyRequest {
    pub mode: ExpertResidency,
    pub usable_bytes: u64,
    /// Non-expert resident weights, excluding optional unloaded vision weights.
    pub fixed_weight_bytes: u64,
    pub expert_weight_bytes: u64,
    /// Zero means expert paging is not implemented/qualified for this model.
    pub minimum_expert_bytes: u64,
    /// Bounds the SSD-mode hot cache; None uses the remaining model budget.
    pub expert_cache_limit_bytes: Option<u64>,
    pub ple_cache_bytes: u64,
    pub draft_bytes: u64,
    pub kv_and_recurrent_bytes: u64,
    /// Includes prefill activations, packed-layout padding, staging buffers,
    /// simultaneous GPU/ANE buffers and in-flight loads retained by execution.
    pub temporary_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResidencyPlan {
    pub mode: ExpertResidency,
    pub non_expert_bytes: u64,
    pub resident_expert_bytes: u64,
    pub peak_bytes: u64,
    pub usable_bytes: u64,
}

impl ResidencyRequest {
    pub fn plan(&self) -> Result<ResidencyPlan> {
        let non_expert = [
            self.fixed_weight_bytes,
            self.ple_cache_bytes,
            self.draft_bytes,
            self.kv_and_recurrent_bytes,
            self.temporary_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |n, bytes| n.checked_add(bytes))
        .ok_or_else(|| anyhow!("non-expert memory budget overflow"))?;
        ensure!(
            non_expert <= self.usable_bytes,
            "non-expert working set requires {non_expert} bytes, usable budget is {}",
            self.usable_bytes
        );
        let available = self.usable_bytes - non_expert;
        let full_fits = self.expert_weight_bytes <= available;
        let (mode, resident_expert_bytes) = match self.mode {
            ExpertResidency::Full | ExpertResidency::Auto if full_fits => {
                (ExpertResidency::Full, self.expert_weight_bytes)
            }
            ExpertResidency::Full => {
                return Err(anyhow!(
                    "full expert residency requires {} bytes, remaining budget is {available}",
                    self.expert_weight_bytes
                ))
            }
            _ => {
                ensure!(self.expert_weight_bytes > 0 && self.minimum_expert_bytes > 0,
                    "SSD expert residency requires an adapted MoE model and a known minimum working set");
                ensure!(
                    self.minimum_expert_bytes <= self.expert_weight_bytes,
                    "minimum expert working set exceeds all expert weights"
                );
                let cache = self
                    .expert_cache_limit_bytes
                    .unwrap_or(available)
                    .min(available)
                    .min(self.expert_weight_bytes);
                ensure!(
                    cache >= self.minimum_expert_bytes,
                    "expert cache budget {cache} is below the minimum working set {}",
                    self.minimum_expert_bytes
                );
                (ExpertResidency::Ssd, cache)
            }
        };
        Ok(ResidencyPlan {
            mode,
            non_expert_bytes: non_expert,
            resident_expert_bytes,
            peak_bytes: non_expert + resident_expert_bytes,
            usable_bytes: self.usable_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;
    fn request() -> ResidencyRequest {
        ResidencyRequest {
            mode: ExpertResidency::Auto,
            usable_bytes: 56 * GIB,
            fixed_weight_bytes: 4 * GIB,
            expert_weight_bytes: 70 * GIB,
            minimum_expert_bytes: 2 * GIB,
            expert_cache_limit_bytes: Some(24 * GIB),
            ple_cache_bytes: GIB,
            draft_bytes: 2 * GIB,
            kv_and_recurrent_bytes: 8 * GIB,
            temporary_bytes: 3 * GIB,
        }
    }
    #[test]
    fn auto_uses_a_bounded_expert_cache_when_all_weights_do_not_fit() {
        let plan = request().plan().unwrap();
        assert_eq!(plan.mode, ExpertResidency::Ssd);
        assert_eq!(plan.non_expert_bytes, 18 * GIB);
        assert_eq!(plan.resident_expert_bytes, 24 * GIB);
        assert_eq!(plan.peak_bytes, 42 * GIB);
    }
    #[test]
    fn auto_prefers_full_residency_when_the_complete_plan_fits() {
        let mut r = request();
        r.usable_bytes = 100 * GIB;
        let p = r.plan().unwrap();
        assert_eq!(p.mode, ExpertResidency::Full);
        assert_eq!(p.peak_bytes, 88 * GIB);
        r.mode = ExpertResidency::Ssd;
        assert_eq!(r.plan().unwrap().resident_expert_bytes, 24 * GIB);
    }
    #[test]
    fn rejects_an_unadapted_dense_or_moe_paging_fallback() {
        let mut r = request();
        r.minimum_expert_bytes = 0;
        assert!(r.plan().is_err());
        r.expert_weight_bytes = 0;
        r.mode = ExpertResidency::Ssd;
        assert!(r.plan().is_err());
        r.mode = ExpertResidency::Auto;
        r.fixed_weight_bytes = 100 * GIB;
        assert!(r.plan().is_err());
    }
    #[test]
    fn rejects_insufficient_fixed_minimum_and_full_resident_budgets() {
        let mut r = request();
        r.mode = ExpertResidency::Full;
        assert!(r.plan().is_err());
        r.mode = ExpertResidency::Ssd;
        r.expert_cache_limit_bytes = Some(GIB);
        assert!(r.plan().is_err());
        r.expert_cache_limit_bytes = None;
        r.usable_bytes = 19 * GIB;
        assert!(r.plan().is_err());
        r.usable_bytes = 17 * GIB;
        assert!(r.plan().is_err());
    }
    #[test]
    fn rejects_overflow_and_clamps_cache_to_real_weights_and_available_memory() {
        let mut r = request();
        r.fixed_weight_bytes = u64::MAX;
        assert!(r.plan().is_err());
        let mut r = request();
        r.expert_cache_limit_bytes = None;
        assert_eq!(r.plan().unwrap().peak_bytes, r.usable_bytes);
        r.mode = ExpertResidency::Ssd;
        r.usable_bytes = 100 * GIB;
        assert_eq!(
            r.plan().unwrap().resident_expert_bytes,
            r.expert_weight_bytes
        );
    }
}
