//! Synchronous single-request MTP execution, independent of CLI or HTTP output.

use std::collections::VecDeque;

use anyhow::Context;
use ironmlx_lm::core::speculative_model::MtpSpeculativeModel;
use ironmlx_lm::core::tokenizer::{DecodeStream, Tokenizer};
use ironmlx_lm::core::vision::DenseVlMethods;

use super::generation_types::{GenerateEvent, GenerateRequest};
use super::scheduler::{Phase, Scheduler, StepEvent};
use super::speculative::MtpSpeculativeConfig;
use crate::Result;

fn mtp_scheduler_effective_cap_for_cli(request_cap: usize, model_max_context: i32) -> usize {
    let request_cap =
        request_cap.max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize);
    let headroom_cap =
        ((request_cap as f64) / crate::core::memory_budget::SOFT_LIMIT_FRAC).ceil() as usize;
    let headroom_cap = headroom_cap.max(request_cap);
    let model_max_context = usize::try_from(model_max_context).unwrap_or(0);
    if model_max_context == 0 {
        headroom_cap
    } else {
        headroom_cap.min(model_max_context)
    }
}

pub struct MtpSchedulerGenerationStream<'m, M: MtpSpeculativeModel> {
    model: &'m M,
    mtp: &'m M::MtpHead,
    scheduler: Scheduler<M>,
    cfg: MtpSpeculativeConfig,
    pending: VecDeque<StepEvent>,
    did_prefill: bool,
    finished: bool,
    detok: DecodeStream<'m>,
}

impl<'m, M: MtpSpeculativeModel + DenseVlMethods> MtpSchedulerGenerationStream<'m, M> {
    pub fn new(
        model: &'m M,
        mtp: &'m M::MtpHead,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Self> {
        let request_cap = request
            .prompt_ids
            .len()
            .saturating_add(request.max_new_tokens)
            .max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize);
        let meta = model.model_meta();
        let effective_cap =
            mtp_scheduler_effective_cap_for_cli(request_cap, meta.max_position_embeddings);
        let mut scheduler = Scheduler::<M>::new(1, effective_cap, meta)
            .context("creating MTP generation scheduler")?;
        scheduler
            .admit(request)
            .context("admitting MTP generation request")?;
        Ok(Self {
            model,
            mtp,
            scheduler,
            cfg,
            pending: VecDeque::new(),
            did_prefill: false,
            finished: false,
            detok: tokenizer.decode_stream(true),
        })
    }

    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        loop {
            if let Some(ev) = self.pending.pop_front() {
                let text = self.detok.step(ev.token)?.unwrap_or_default();
                if ev.finish_reason.is_some() {
                    self.finished = true;
                }
                return Ok(Some(GenerateEvent {
                    token: ev.token,
                    text,
                    finish_reason: ev.finish_reason,
                }));
            }
            if !self.did_prefill {
                self.pending.extend(
                    self.scheduler
                        .prefill_admitted_mtp_single(self.model, self.mtp, self.cfg)?,
                );
                self.did_prefill = true;
            } else if self.scheduler.phase() == Phase::Decoding {
                self.pending
                    .extend(self.scheduler.step_mtp_single(self.model, self.mtp)?);
            } else {
                self.finished = true;
                return Ok(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mtp_scheduler_effective_cap_for_cli_leaves_soft_limit_headroom() {
        let cap = mtp_scheduler_effective_cap_for_cli(365, 1_000);
        assert_eq!(cap, 430);
        assert!(
            ((cap as f64) * crate::core::memory_budget::SOFT_LIMIT_FRAC) >= 365.0,
            "cap={cap} should leave enough soft-limit budget for request"
        );

        let min_cap = mtp_scheduler_effective_cap_for_cli(1, 1_000);
        assert_eq!(min_cap, 302);

        let clamped = mtp_scheduler_effective_cap_for_cli(900, 1_000);
        assert_eq!(clamped, 1_000);
    }
}
