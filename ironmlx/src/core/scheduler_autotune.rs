//! Diagnose-only scheduler/autotune reporting.
//!
//! This module intentionally does not mutate runtime configuration. It turns
//! the current `serve` parameters, model metadata, and model-level scheduler
//! policy hooks into a startup report that can guide later benchmark sweeps.

use std::fmt::Write;

use crate::core::memory_budget::{
    kv_bytes_per_token, ModelMeta, SAFETY_MARGIN_BYTES, SOFT_LIMIT_FRAC,
};
use crate::core::Model;

const PROMPT_LIMIT_SAMPLES: [usize; 4] = [512, 1024, 2048, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptBatchLimit {
    pub prompt_len: usize,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerAutotuneInput {
    pub model_name: String,
    pub meta: ModelMeta,
    pub prefill_chunk_size: usize,
    pub b_max: usize,
    pub admission_deadline_ms: u64,
    pub admission_queue_max: usize,
    pub requested_max_cache_cap: usize,
    pub effective_cap_max: usize,
    pub total_ram_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationLevel {
    Info,
    Warning,
}

impl RecommendationLevel {
    fn as_str(self) -> &'static str {
        match self {
            RecommendationLevel::Info => "INFO",
            RecommendationLevel::Warning => "WARN",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerAutotuneRecommendation {
    pub level: RecommendationLevel,
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerAutotuneDerived {
    pub kv_bytes_per_token: usize,
    pub available_budget_bytes: usize,
    pub reserved_kv_bytes: usize,
    pub soft_limit_bytes: usize,
    pub max_b_max_for_effective_cap: usize,
    pub max_cache_cap_for_b_max: usize,
    pub requested_cache_cap_was_capped: bool,
}

#[derive(Debug, Clone)]
pub struct SchedulerAutotuneReport {
    pub diagnose_only: bool,
    pub input: SchedulerAutotuneInput,
    pub derived: SchedulerAutotuneDerived,
    pub prompt_batch_limits: Vec<PromptBatchLimit>,
    pub recommendations: Vec<SchedulerAutotuneRecommendation>,
}

pub fn prompt_batch_limits_for_model<M: Model>(b_max: usize) -> Vec<PromptBatchLimit> {
    PROMPT_LIMIT_SAMPLES
        .iter()
        .copied()
        .map(|prompt_len| PromptBatchLimit {
            prompt_len,
            limit: M::fresh_prefill_batch_limit(prompt_len, b_max).clamp(1, b_max),
        })
        .collect()
}

pub fn build_scheduler_autotune_report(
    input: SchedulerAutotuneInput,
    prompt_batch_limits: Vec<PromptBatchLimit>,
) -> SchedulerAutotuneReport {
    let derived = derive_stats(&input);
    let recommendations = build_recommendations(&input, &derived, &prompt_batch_limits);
    SchedulerAutotuneReport {
        diagnose_only: true,
        input,
        derived,
        prompt_batch_limits,
        recommendations,
    }
}

impl SchedulerAutotuneReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "scheduler/autotune report (diagnose-only; no runtime parameters changed)"
        )
        .unwrap();
        writeln!(out, "model: {}", self.input.model_name).unwrap();
        writeln!(
            out,
            "current: prefill_chunk_size={} b_max={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} effective_cap_max={}",
            self.input.prefill_chunk_size,
            self.input.b_max,
            self.input.admission_deadline_ms,
            self.input.admission_queue_max,
            self.input.requested_max_cache_cap,
            self.input.effective_cap_max
        )
        .unwrap();
        writeln!(
            out,
            "memory: total={} model_weights={} available_budget={} kv_bytes_per_token={} reserved_kv={} soft_limit={}",
            format_bytes(self.input.total_ram_bytes),
            format_bytes(self.input.meta.weight_bytes),
            format_bytes(self.derived.available_budget_bytes),
            self.derived.kv_bytes_per_token,
            format_bytes(self.derived.reserved_kv_bytes),
            format_bytes(self.derived.soft_limit_bytes),
        )
        .unwrap();
        writeln!(
            out,
            "capacity: max_b_max_for_effective_cap={} max_cache_cap_for_b_max={}",
            self.derived.max_b_max_for_effective_cap, self.derived.max_cache_cap_for_b_max
        )
        .unwrap();
        if self.prompt_batch_limits.is_empty() {
            writeln!(out, "model_prefill_batch_limits: (none)").unwrap();
        } else {
            let limits = self
                .prompt_batch_limits
                .iter()
                .map(|sample| format!("PP{}=>{}", sample.prompt_len, sample.limit))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "model_prefill_batch_limits: {limits}").unwrap();
        }
        writeln!(out, "recommendations:").unwrap();
        if self.recommendations.is_empty() {
            writeln!(
                out,
                "- INFO no_recommendations: current settings look consistent"
            )
            .unwrap();
        } else {
            for item in &self.recommendations {
                writeln!(
                    out,
                    "- {} {}: {}",
                    item.level.as_str(),
                    item.code,
                    item.message
                )
                .unwrap();
            }
        }
        out
    }
}

fn derive_stats(input: &SchedulerAutotuneInput) -> SchedulerAutotuneDerived {
    let kv_token_bytes = kv_bytes_per_token(&input.meta);
    let available_budget = input
        .total_ram_bytes
        .saturating_sub(input.meta.weight_bytes)
        .saturating_sub(SAFETY_MARGIN_BYTES);
    let reserved_kv = input
        .b_max
        .saturating_mul(input.effective_cap_max)
        .saturating_mul(kv_token_bytes);
    let soft_limit = ((reserved_kv as f64) * SOFT_LIMIT_FRAC) as usize;
    let bytes_per_full_cap_slot = input.effective_cap_max.saturating_mul(kv_token_bytes);
    let max_b_for_cap = available_budget
        .checked_div(bytes_per_full_cap_slot)
        .unwrap_or(0);
    let bytes_per_b_token = input.b_max.saturating_mul(kv_token_bytes);
    let max_cap_for_b = available_budget.checked_div(bytes_per_b_token).unwrap_or(0);

    SchedulerAutotuneDerived {
        kv_bytes_per_token: kv_token_bytes,
        available_budget_bytes: available_budget,
        reserved_kv_bytes: reserved_kv,
        soft_limit_bytes: soft_limit,
        max_b_max_for_effective_cap: max_b_for_cap,
        max_cache_cap_for_b_max: max_cap_for_b,
        requested_cache_cap_was_capped: input.requested_max_cache_cap > input.effective_cap_max,
    }
}

fn build_recommendations(
    input: &SchedulerAutotuneInput,
    derived: &SchedulerAutotuneDerived,
    prompt_batch_limits: &[PromptBatchLimit],
) -> Vec<SchedulerAutotuneRecommendation> {
    let mut items = Vec::new();

    if derived.reserved_kv_bytes > derived.available_budget_bytes {
        items.push(warn(
            "memory_budget_overrun",
            format!(
                "reserved KV ({}) exceeds available budget ({}); lower --b-max or --max-cache-cap before enabling this profile",
                format_bytes(derived.reserved_kv_bytes),
                format_bytes(derived.available_budget_bytes)
            ),
        ));
    } else if derived.available_budget_bytes > 0
        && derived.reserved_kv_bytes >= ((derived.available_budget_bytes as f64) * 0.8) as usize
    {
        items.push(warn(
            "memory_budget_tight",
            format!(
                "reserved KV ({}) is close to available budget ({}); long prompts and concurrency may hit runtime admission limits",
                format_bytes(derived.reserved_kv_bytes),
                format_bytes(derived.available_budget_bytes)
            ),
        ));
    }

    if derived.requested_cache_cap_was_capped {
        items.push(info(
            "max_cache_cap_capped",
            format!(
                "requested max_cache_cap={} is capped by model context to {}",
                input.requested_max_cache_cap, input.effective_cap_max
            ),
        ));
    }

    if input.b_max == 1 {
        items.push(info(
            "single_request_mode",
            "b_max=1 optimizes single-request latency; concurrent agent traffic will queue"
                .to_string(),
        ));
    } else {
        items.push(info(
            "concurrent_mode",
            format!(
                "b_max={} enables concurrent scheduler slots; validate TTFT/ITL under agent-style long prompts",
                input.b_max
            ),
        ));
    }

    if input.prefill_chunk_size == 0 {
        items.push(warn(
            "prefill_chunking_disabled",
            "prefill chunking is disabled; long prompts can block decode cadence and queued-request TTFT"
                .to_string(),
        ));
    } else if input.prefill_chunk_size < 512 {
        items.push(info(
            "prefill_chunking_fine",
            format!(
                "prefill_chunk_size={} is fine-grained; benchmark chunk overhead before keeping it",
                input.prefill_chunk_size
            ),
        ));
    } else if input.prefill_chunk_size > 4096 {
        items.push(info(
            "prefill_chunking_coarse",
            format!(
                "prefill_chunk_size={} is coarse; queued-request TTFT may rise for long prompts",
                input.prefill_chunk_size
            ),
        ));
    }

    if input.admission_queue_max == 0 {
        items.push(warn(
            "admission_queue_disabled",
            "admission_queue_max=0 rejects requests immediately when scheduler slots are saturated"
                .to_string(),
        ));
    }

    if input.admission_deadline_ms > 10 {
        items.push(warn(
            "admission_deadline_high",
            format!(
                "admission_deadline_ms={} can increase first-batch TTFT; validate against agent latency goals",
                input.admission_deadline_ms
            ),
        ));
    }

    if prompt_batch_limits
        .iter()
        .any(|sample| sample.limit < input.b_max)
    {
        items.push(info(
            "model_prefill_limit_active",
            "model fresh-prefill batch limit is below b_max for at least one sampled prompt length; treat it as part of the performance path"
                .to_string(),
        ));
    }

    items
}

fn info(code: &'static str, message: String) -> SchedulerAutotuneRecommendation {
    SchedulerAutotuneRecommendation {
        level: RecommendationLevel::Info,
        code,
        message,
    }
}

fn warn(code: &'static str, message: String) -> SchedulerAutotuneRecommendation {
    SchedulerAutotuneRecommendation {
        level: RecommendationLevel::Warning,
        code,
        message,
    }
}

fn format_bytes(bytes: usize) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", (bytes as f64) / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", (bytes as f64) / MIB)
    } else {
        format!("{bytes} B")
    }
}
