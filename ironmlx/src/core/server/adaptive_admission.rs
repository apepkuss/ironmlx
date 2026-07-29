pub(crate) const GEMMA4_DRAFTER_ADAPTIVE_PHYSICAL_B_MAX: usize = 4;
pub(crate) const QWEN_MTP_ADAPTIVE_PHYSICAL_B_MAX: usize = 2;

const ADAPTIVE_LATENCY_BATCH_LIMIT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdmissionRequestShape {
    pub(crate) prompt_len: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) decode_cadence_mid_chunk_cap: usize,
    pub(crate) speculative_pipelinable: bool,
}

impl AdmissionRequestShape {
    pub(crate) fn uses_multi_chunk_prefill(self) -> bool {
        self.prefill_chunk_size > 0 && self.prompt_len > self.prefill_chunk_size
    }

    #[cfg(test)]
    fn rolling_prefill_required_decode_steps(self) -> usize {
        let requested_chunk = if self.prefill_chunk_size == 0 {
            self.prompt_len.max(1)
        } else {
            self.prefill_chunk_size.max(1)
        };
        let effective_chunk = requested_chunk
            .min(self.decode_cadence_mid_chunk_cap.max(1))
            .max(1);
        let chunks = self.prompt_len.saturating_add(effective_chunk - 1) / effective_chunk;
        chunks.saturating_mul(ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK)
    }
}

pub(crate) const ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveAdmissionMode {
    Disabled,
    Gemma4Drafter,
    PromptLookup,
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

    pub(crate) fn prompt_lookup() -> Self {
        Self {
            mode: AdaptiveAdmissionMode::PromptLookup,
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
            AdaptiveAdmissionMode::Gemma4Drafter | AdaptiveAdmissionMode::PromptLookup => {
                if request.uses_multi_chunk_prefill() {
                    1
                } else {
                    model_limit
                        .min(ADAPTIVE_LATENCY_BATCH_LIMIT)
                        .clamp(1, b_max.max(1))
                }
            }
            AdaptiveAdmissionMode::QwenMtp => {
                if request.uses_multi_chunk_prefill() {
                    if request.speculative_pipelinable {
                        model_limit
                            .min(ADAPTIVE_LATENCY_BATCH_LIMIT)
                            .clamp(1, b_max.max(1))
                    } else {
                        1
                    }
                } else {
                    model_limit
                        .min(ADAPTIVE_LATENCY_BATCH_LIMIT)
                        .clamp(1, b_max.max(1))
                }
            }
        }
    }

    pub(crate) fn can_join_fresh_batch(
        self,
        first: AdmissionRequestShape,
        candidate: AdmissionRequestShape,
    ) -> bool {
        match self.mode {
            AdaptiveAdmissionMode::Disabled => true,
            AdaptiveAdmissionMode::Gemma4Drafter | AdaptiveAdmissionMode::PromptLookup => {
                first.uses_multi_chunk_prefill() == candidate.uses_multi_chunk_prefill()
            }
            AdaptiveAdmissionMode::QwenMtp => {
                if first.uses_multi_chunk_prefill() != candidate.uses_multi_chunk_prefill() {
                    return false;
                }
                if first.uses_multi_chunk_prefill() {
                    first.speculative_pipelinable && candidate.speculative_pipelinable
                } else {
                    true
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
        _available_decode_steps: usize,
    ) -> bool {
        if active_count >= b_max {
            return false;
        }
        let model_limit = model_limit.clamp(1, b_max.max(1));
        match self.mode {
            AdaptiveAdmissionMode::Disabled => {
                active_count < model_limit || request.uses_multi_chunk_prefill()
            }
            AdaptiveAdmissionMode::Gemma4Drafter | AdaptiveAdmissionMode::PromptLookup
                if request.uses_multi_chunk_prefill() =>
            {
                true
            }
            AdaptiveAdmissionMode::QwenMtp if request.uses_multi_chunk_prefill() => false,
            AdaptiveAdmissionMode::Gemma4Drafter
            | AdaptiveAdmissionMode::PromptLookup
            | AdaptiveAdmissionMode::QwenMtp => {
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
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 4);
        assert!(policy.can_start_rolling_mid_admit(request, 3, 4, 4, 0));
    }

    #[test]
    fn gemma4_drafter_short_fresh_batch_uses_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
    }

    #[test]
    fn gemma4_drafter_long_chunked_fresh_batch_starts_single_request() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 24_576,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 1);
    }

    #[test]
    fn gemma4_drafter_long_chunked_mid_admit_can_use_physical_slots() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 24_576,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert!(policy.can_start_rolling_mid_admit(request, 3, 4, 4, 0));
        assert!(!policy.can_start_rolling_mid_admit(request, 4, 4, 4, 0));
    }

    #[test]
    fn prompt_lookup_serializes_long_fresh_prefill_and_allows_rolling_admit() {
        let policy = AdaptiveAdmissionPolicy::prompt_lookup();
        let request = AdmissionRequestShape {
            prompt_len: 8192,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 1);
        assert!(policy.can_start_rolling_mid_admit(request, 1, 4, 4, 0));
        assert!(policy.can_join_fresh_batch(request, request));
    }

    #[test]
    fn prompt_lookup_keeps_short_fresh_prefill_batched() {
        let policy = AdaptiveAdmissionPolicy::prompt_lookup();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
    }

    #[test]
    fn gemma4_drafter_non_chunked_mid_admit_stays_with_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::gemma4_drafter();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert!(policy.can_start_rolling_mid_admit(request, 1, 4, 4, 0));
        assert!(!policy.can_start_rolling_mid_admit(request, 2, 4, 4, 0));
    }

    #[test]
    fn qwen_mtp_pipelinable_long_chunked_fresh_batch_uses_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
    }

    #[test]
    fn qwen_mtp_non_pipelinable_long_chunked_fresh_batch_starts_single_request() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: false,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 1);
    }

    #[test]
    fn qwen_mtp_long_chunked_mid_admit_does_not_enter_decode_hot_path() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        let available_decode_steps = request.rolling_prefill_required_decode_steps();
        assert!(!policy.can_start_rolling_mid_admit(request, 0, 4, 4, available_decode_steps));
        assert!(!policy.can_start_rolling_mid_admit(request, 1, 4, 4, available_decode_steps));
        assert!(!policy.can_start_rolling_mid_admit(request, 3, 4, 4, available_decode_steps));
        assert!(!policy.can_start_rolling_mid_admit(request, 4, 4, 4, available_decode_steps));
    }

    #[test]
    fn qwen_mtp_long_chunked_mid_admit_rejects_512_token_decode_budget() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 19_785,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(request.rolling_prefill_required_decode_steps(), 312);
        assert!(!policy.can_start_rolling_mid_admit(request, 1, 4, 4, 64));
        assert!(!policy.can_start_rolling_mid_admit(request, 1, 4, 4, 312));
    }

    #[test]
    fn qwen_mtp_short_requests_stay_with_latency_cap() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let request = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert_eq!(policy.fresh_batch_limit(request, 4, 4), 2);
        assert!(policy.can_start_rolling_mid_admit(request, 1, 4, 4, 0));
        assert!(!policy.can_start_rolling_mid_admit(request, 2, 4, 4, 0));
    }

    #[test]
    fn qwen_mtp_fresh_batch_compatibility_separates_long_and_short_requests() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let long = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };
        let short = AdmissionRequestShape {
            prompt_len: 512,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };

        assert!(policy.can_join_fresh_batch(long, long));
        assert!(policy.can_join_fresh_batch(short, short));
        assert!(!policy.can_join_fresh_batch(long, short));
        assert!(!policy.can_join_fresh_batch(short, long));
    }

    #[test]
    fn qwen_mtp_fresh_batch_compatibility_rejects_non_pipelinable_long_candidate() {
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let long = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: true,
        };
        let non_pipelinable_long = AdmissionRequestShape {
            prompt_len: 32_768,
            prefill_chunk_size: 2048,
            decode_cadence_mid_chunk_cap: 256,
            speculative_pipelinable: false,
        };

        assert!(!policy.can_join_fresh_batch(long, non_pipelinable_long));
    }
}
