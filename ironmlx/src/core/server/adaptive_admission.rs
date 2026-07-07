pub(crate) const GEMMA4_DRAFTER_ADAPTIVE_PHYSICAL_B_MAX: usize = 4;
pub(crate) const QWEN_MTP_ADAPTIVE_PHYSICAL_B_MAX: usize = 4;

const ADAPTIVE_LATENCY_BATCH_LIMIT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdmissionRequestShape {
    pub(crate) prompt_len: usize,
    pub(crate) prefill_chunk_size: usize,
}

impl AdmissionRequestShape {
    pub(crate) fn uses_multi_chunk_prefill(self) -> bool {
        self.prefill_chunk_size > 0 && self.prompt_len > self.prefill_chunk_size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveAdmissionMode {
    Disabled,
    Gemma4Drafter,
    QwenMtp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdaptiveAdmissionPolicy {
    mode: AdaptiveAdmissionMode,
}

impl AdaptiveAdmissionPolicy {
    pub(crate) fn disabled() -> Self {
        Self {
            mode: AdaptiveAdmissionMode::Disabled,
        }
    }

    pub(crate) fn gemma4_drafter() -> Self {
        Self {
            mode: AdaptiveAdmissionMode::Gemma4Drafter,
        }
    }

    pub(crate) fn qwen_mtp() -> Self {
        Self {
            mode: AdaptiveAdmissionMode::QwenMtp,
        }
    }

    pub(crate) fn fresh_batch_limit(
        self,
        request: AdmissionRequestShape,
        model_limit: usize,
        b_max: usize,
    ) -> usize {
        let model_limit = model_limit.clamp(1, b_max.max(1));
        match self.mode {
            AdaptiveAdmissionMode::Disabled => model_limit,
            AdaptiveAdmissionMode::Gemma4Drafter | AdaptiveAdmissionMode::QwenMtp => {
                if request.uses_multi_chunk_prefill() {
                    1
                } else {
                    model_limit
                        .min(ADAPTIVE_LATENCY_BATCH_LIMIT)
                        .clamp(1, b_max.max(1))
                }
            }
        }
    }

    pub(crate) fn can_start_rolling_mid_admit(
        self,
        request: AdmissionRequestShape,
        active_count: usize,
        model_limit: usize,
        b_max: usize,
    ) -> bool {
        if active_count >= b_max {
            return false;
        }
        let model_limit = model_limit.clamp(1, b_max.max(1));
        match self.mode {
            AdaptiveAdmissionMode::Disabled => {
                active_count < model_limit || request.uses_multi_chunk_prefill()
            }
            AdaptiveAdmissionMode::Gemma4Drafter if request.uses_multi_chunk_prefill() => true,
            AdaptiveAdmissionMode::QwenMtp if request.uses_multi_chunk_prefill() => {
                active_count == 0
            }
            AdaptiveAdmissionMode::Gemma4Drafter | AdaptiveAdmissionMode::QwenMtp => {
                active_count < model_limit.min(ADAPTIVE_LATENCY_BATCH_LIMIT)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_preserves_model_limits() {
        let policy = AdaptiveAdmissionPolicy::disabled();
        let request = AdmissionRequestShape {
            prompt_len: 8192,
            prefill_chunk_size: 2048,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 4);
        assert!(policy.can_start_rolling_mid_admit(request, 3, 4, 4));
    }

    #[test]
    fn gemma4_drafter_short_fresh_batch_uses_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
    }

    #[test]
    fn gemma4_drafter_long_chunked_fresh_batch_starts_single_request() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 24_576,
            prefill_chunk_size: 2048,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 1);
    }

    #[test]
    fn gemma4_drafter_long_chunked_mid_admit_can_use_physical_slots() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 24_576,
            prefill_chunk_size: 2048,
        };

        assert!(policy.can_start_rolling_mid_admit(request, 3, 4, 4));
        assert!(!policy.can_start_rolling_mid_admit(request, 4, 4, 4));
    }

    #[test]
    fn gemma4_drafter_non_chunked_mid_admit_stays_with_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
        };

        assert!(policy.can_start_rolling_mid_admit(request, 1, 4, 4));
        assert!(!policy.can_start_rolling_mid_admit(request, 2, 4, 4));
    }

    #[test]
    fn qwen_mtp_long_chunked_fresh_batch_starts_single_request() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 1);
    }

    #[test]
    fn qwen_mtp_long_chunked_mid_admit_stays_single_request() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
        };

        assert!(policy.can_start_rolling_mid_admit(request, 0, 4, 4));
        assert!(!policy.can_start_rolling_mid_admit(request, 1, 4, 4));
        assert!(!policy.can_start_rolling_mid_admit(request, 4, 4, 4));
    }

    #[test]
    fn qwen_mtp_short_requests_stay_with_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
        assert!(policy.can_start_rolling_mid_admit(request, 1, 4, 4));
        assert!(!policy.can_start_rolling_mid_admit(request, 2, 4, 4));
    }
}
