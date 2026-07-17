//! Diagnose-only scheduler/autotune reporting.
//!
//! This module intentionally does not mutate runtime configuration. It turns
//! the current `serve` parameters, model metadata, and model-level scheduler
//! policy hooks into a startup report that can guide later benchmark sweeps.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};

use crate::core::memory_budget::{
    kv_bytes_per_token, ModelMeta, SAFETY_MARGIN_BYTES, SOFT_LIMIT_FRAC,
};
use crate::core::Model;
use serde::{Deserialize, Serialize};

const PROMPT_LIMIT_SAMPLES: [usize; 4] = [512, 1024, 2048, 8192];
pub const SCHEDULER_AUTOTUNE_SCHEMA_VERSION: u32 = 5;

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
    pub decode_cadence_mid_chunk_cap: usize,
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
            "current: prefill_chunk_size={} b_max={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} effective_cap_max={} decode_cadence_mid_chunk_cap={}",
            self.input.prefill_chunk_size,
            self.input.b_max,
            self.input.admission_deadline_ms,
            self.input.admission_queue_max,
            self.input.requested_max_cache_cap,
            self.input.effective_cap_max,
            self.input.decode_cadence_mid_chunk_cap
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

    if input.b_max > 1
        && input.prefill_chunk_size > 0
        && input.decode_cadence_mid_chunk_cap < input.prefill_chunk_size
    {
        items.push(info(
            "decode_cadence_cap_active",
            format!(
                "decode_cadence_mid_chunk_cap={} can split rolling mid-admit prefill chunks below prefill_chunk_size={} when active decode rows exist",
                input.decode_cadence_mid_chunk_cap, input.prefill_chunk_size
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneObjective {
    pub ttft_p95_weight: f64,
    pub itl_p95_weight: f64,
    pub e2e_p95_weight: f64,
    pub throughput_weight: f64,
}

impl SchedulerAutotuneObjective {
    pub fn agent_default() -> Self {
        Self {
            ttft_p95_weight: 0.40,
            itl_p95_weight: 0.35,
            e2e_p95_weight: 0.20,
            throughput_weight: 0.05,
        }
    }

    fn normalized(self) -> Self {
        let sum = self.ttft_p95_weight
            + self.itl_p95_weight
            + self.e2e_p95_weight
            + self.throughput_weight;
        if sum <= f64::EPSILON {
            return Self::agent_default();
        }
        Self {
            ttft_p95_weight: self.ttft_p95_weight / sum,
            itl_p95_weight: self.itl_p95_weight / sum,
            e2e_p95_weight: self.e2e_p95_weight / sum,
            throughput_weight: self.throughput_weight / sum,
        }
    }
}

impl Default for SchedulerAutotuneObjective {
    fn default() -> Self {
        Self::agent_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerAutotuneProfileConfig {
    pub b_max: usize,
    pub prefill_chunk_size: usize,
    pub admission_deadline_ms: u64,
    pub admission_queue_max: usize,
    pub max_cache_cap: usize,
    pub decode_cadence_mid_chunk_cap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerExecutionModel {
    RollingV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerSpeculativeMode {
    Disabled,
    QwenMtp,
    Gemma4Drafter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerKvQuantization {
    None,
    Turbo3,
    Turbo4,
    K3V4,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerWeightQuantizationContext {
    pub mode: String,
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerSpeculativeContext {
    pub mode: SchedulerSpeculativeMode,
    pub draft_model_fingerprint: Option<String>,
    pub draft_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerPrefixCacheContext {
    pub enabled: bool,
    pub block_size: Option<usize>,
    pub max_pages: Option<usize>,
    pub lru_max_bytes: Option<usize>,
    pub ssd_max_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerActiveKvContext {
    pub enabled: bool,
    pub resident_cap_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeContext {
    pub execution_model: SchedulerExecutionModel,
    pub model_architecture: String,
    pub model_fingerprint: String,
    pub weight_quantization: SchedulerWeightQuantizationContext,
    pub speculative: SchedulerSpeculativeContext,
    pub kv_quantization: SchedulerKvQuantization,
    pub prefix_cache: SchedulerPrefixCacheContext,
    pub active_kv: SchedulerActiveKvContext,
    pub logical_kv_cap_tokens: usize,
    pub memory_limit_total_bytes: Option<usize>,
    pub memory_limit_model_bytes: Option<usize>,
}

impl SchedulerAutotuneRuntimeContext {
    pub fn local_default(logical_kv_cap_tokens: usize) -> Self {
        Self {
            execution_model: SchedulerExecutionModel::RollingV1,
            model_architecture: "unknown".to_string(),
            model_fingerprint: "runtime-default".to_string(),
            weight_quantization: SchedulerWeightQuantizationContext {
                mode: "unknown".to_string(),
                fingerprint: "runtime-default".to_string(),
            },
            speculative: SchedulerSpeculativeContext {
                mode: SchedulerSpeculativeMode::Disabled,
                draft_model_fingerprint: None,
                draft_tokens: None,
            },
            kv_quantization: SchedulerKvQuantization::None,
            prefix_cache: SchedulerPrefixCacheContext {
                enabled: false,
                block_size: None,
                max_pages: None,
                lru_max_bytes: None,
                ssd_max_bytes: None,
            },
            active_kv: SchedulerActiveKvContext {
                enabled: false,
                resident_cap_tokens: None,
            },
            logical_kv_cap_tokens,
            memory_limit_total_bytes: None,
            memory_limit_model_bytes: None,
        }
    }

    pub fn fingerprint(&self) -> String {
        let encoded = serde_json::to_vec(self)
            .expect("scheduler runtime context contains only serializable fields");
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in encoded {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerAutotuneCacheState {
    Cold,
    Warm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneMtpMetrics {
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub windows: u64,
    pub fallback_prefill_count: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub main_rollback_us: u64,
    pub prefill_cache_commit_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeHealth {
    pub healthy: bool,
    pub status: String,
    pub request_completion_ok: bool,
    pub admission_queue_full_count_delta: u64,
    pub memory_budget_exceeded_count_delta: u64,
    pub active_kv_degraded: bool,
    pub active_kv_swap_error_count_delta: u64,
    pub logical_kv_cap_tokens: usize,
    pub resident_kv_cap_tokens: usize,
    pub mtp: Option<SchedulerAutotuneMtpMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneMeasurement {
    pub config: SchedulerAutotuneProfileConfig,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub concurrency: usize,
    pub cache_state: SchedulerAutotuneCacheState,
    pub ttft_ms_p95: f64,
    pub itl_ms_p95: f64,
    pub early_itl_ms_p95: f64,
    pub e2e_s_p95: f64,
    pub tokens_per_sec: f64,
    pub memory_budget_ok: bool,
    pub cached_tokens_warning: bool,
    pub runtime_health: SchedulerAutotuneRuntimeHealth,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerAutotuneScenario {
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub concurrency: usize,
    pub cache_state: SchedulerAutotuneCacheState,
}

impl From<&SchedulerAutotuneMeasurement> for SchedulerAutotuneScenario {
    fn from(value: &SchedulerAutotuneMeasurement) -> Self {
        Self {
            prompt_len: value.prompt_len,
            max_new_tokens: value.max_new_tokens,
            concurrency: value.concurrency,
            cache_state: value.cache_state,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneCalibrationInput {
    pub schema_version: u32,
    pub model_name: String,
    pub hardware_label: String,
    pub runtime_context: SchedulerAutotuneRuntimeContext,
    pub objective: SchedulerAutotuneObjective,
    pub measurements: Vec<SchedulerAutotuneMeasurement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneCandidateScore {
    pub config: SchedulerAutotuneProfileConfig,
    pub score: f64,
    pub scenario_count: usize,
    pub mean_ttft_norm: f64,
    pub mean_itl_norm: f64,
    pub mean_early_itl_norm: f64,
    pub mean_e2e_norm: f64,
    pub mean_throughput_norm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneSelectionNote {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRejectedCandidate {
    pub config: SchedulerAutotuneProfileConfig,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneScenarioOverride {
    pub scenario: SchedulerAutotuneScenario,
    pub config: SchedulerAutotuneProfileConfig,
    pub score: f64,
    pub baseline_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerAutotuneSelectionProfile {
    Balanced,
    #[default]
    AgentLongPrompt,
}

impl SchedulerAutotuneSelectionProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::AgentLongPrompt => "agent-long-prompt",
        }
    }

    fn scenario_weight(self, scenario: &SchedulerAutotuneScenario) -> f64 {
        match self {
            Self::Balanced => 1.0,
            Self::AgentLongPrompt => {
                if scenario.prompt_len >= 4096 {
                    3.0
                } else {
                    1.0
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedulerAutotuneSelectionOptions {
    pub profile: SchedulerAutotuneSelectionProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneProfileSelection {
    pub diagnose_only: bool,
    pub model_name: String,
    pub hardware_label: String,
    pub runtime_context: SchedulerAutotuneRuntimeContext,
    pub selection_profile: SchedulerAutotuneSelectionProfile,
    pub objective: SchedulerAutotuneObjective,
    pub scenarios: Vec<SchedulerAutotuneScenario>,
    pub selected: Option<SchedulerAutotuneCandidateScore>,
    pub candidates: Vec<SchedulerAutotuneCandidateScore>,
    pub scenario_overrides: Vec<SchedulerAutotuneScenarioOverride>,
    pub rejected: Vec<SchedulerAutotuneRejectedCandidate>,
    pub warnings: Vec<SchedulerAutotuneSelectionNote>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeProfileMetadata {
    pub created_at_unix_ms: u64,
    pub ironmlx_version: String,
    pub selection_profile: SchedulerAutotuneSelectionProfile,
    pub objective: SchedulerAutotuneObjective,
    pub scenario_coverage: Vec<SchedulerAutotuneScenario>,
    pub selected_score: f64,
    pub candidate_count: usize,
    pub rejected_count: usize,
    pub selection_warnings: Vec<SchedulerAutotuneSelectionNote>,
}

impl SchedulerAutotuneRuntimeProfileMetadata {
    pub fn synthetic(created_at_unix_ms: u64) -> Self {
        Self {
            created_at_unix_ms,
            ironmlx_version: env!("CARGO_PKG_VERSION").to_string(),
            selection_profile: SchedulerAutotuneSelectionProfile::AgentLongPrompt,
            objective: SchedulerAutotuneObjective::agent_default(),
            scenario_coverage: Vec::new(),
            selected_score: 1.0,
            candidate_count: 0,
            rejected_count: 0,
            selection_warnings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeProfile {
    pub schema_version: u32,
    pub model_name: String,
    pub hardware_label: String,
    pub runtime_context: SchedulerAutotuneRuntimeContext,
    pub config: SchedulerAutotuneProfileConfig,
    pub rules: Vec<SchedulerAutotuneRuntimeRule>,
    pub metadata: SchedulerAutotuneRuntimeProfileMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeRule {
    pub when: SchedulerAutotuneRuntimeRuleCondition,
    pub config: SchedulerAutotuneProfileConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeRuleCondition {
    pub prompt_len_gte: usize,
    pub max_new_tokens_gte: usize,
    pub effective_concurrency_gte: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerAutotuneRuntimeRequest {
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub effective_concurrency: usize,
}

impl SchedulerAutotuneRuntimeProfile {
    pub fn select_config(
        &self,
        request: SchedulerAutotuneRuntimeRequest,
    ) -> SchedulerAutotuneProfileConfig {
        self.rules
            .iter()
            .find(|rule| rule.when.matches(request))
            .map(|rule| rule.config)
            .unwrap_or(self.config)
    }
}

impl SchedulerAutotuneRuntimeRuleCondition {
    fn matches(self, request: SchedulerAutotuneRuntimeRequest) -> bool {
        request.prompt_len >= self.prompt_len_gte
            && request.max_new_tokens >= self.max_new_tokens_gte
            && request.effective_concurrency >= self.effective_concurrency_gte
    }

    fn specificity_key(self) -> (usize, usize, usize) {
        (
            self.prompt_len_gte,
            self.max_new_tokens_gte,
            self.effective_concurrency_gte,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerAutotuneProfileHealthStatus {
    Healthy,
    Warning,
    Invalid,
}

impl SchedulerAutotuneProfileHealthStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Warning => "warning",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerAutotuneProfileHealthLevel {
    Info,
    Warning,
    Error,
}

impl SchedulerAutotuneProfileHealthLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerAutotuneProfileHealthNote {
    pub level: SchedulerAutotuneProfileHealthLevel,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerAutotuneProfileHealthReport {
    pub status: SchedulerAutotuneProfileHealthStatus,
    pub model_name: String,
    pub hardware_label: String,
    pub profile_context_fingerprint: String,
    pub expected_context_fingerprint: String,
    pub created_at_unix_ms: u64,
    pub max_age_days: u64,
    pub notes: Vec<SchedulerAutotuneProfileHealthNote>,
}

impl SchedulerAutotuneProfileHealthReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        writeln!(out, "scheduler/autotune profile health").unwrap();
        writeln!(out, "status: {}", self.status.as_str()).unwrap();
        writeln!(out, "model: {}", self.model_name).unwrap();
        writeln!(out, "hardware: {}", self.hardware_label).unwrap();
        writeln!(
            out,
            "runtime_context: profile={} expected={}",
            self.profile_context_fingerprint, self.expected_context_fingerprint
        )
        .unwrap();
        writeln!(out, "created_at_unix_ms: {}", self.created_at_unix_ms).unwrap();
        writeln!(out, "max_age_days: {}", self.max_age_days).unwrap();
        writeln!(out, "notes:").unwrap();
        for note in &self.notes {
            writeln!(
                out,
                "- {} {}: {}",
                note.level.as_str(),
                note.code,
                note.message
            )
            .unwrap();
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerAutotuneProfileHealthInput<'a> {
    pub profile: &'a SchedulerAutotuneRuntimeProfile,
    pub expected_model_name: &'a str,
    pub expected_hardware_label: &'a str,
    pub expected_runtime_context: &'a SchedulerAutotuneRuntimeContext,
    pub current_ironmlx_version: &'a str,
    pub now_unix_ms: u64,
    pub max_age_days: u64,
}

pub fn evaluate_scheduler_autotune_profile_health(
    input: SchedulerAutotuneProfileHealthInput<'_>,
) -> SchedulerAutotuneProfileHealthReport {
    const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000;

    let profile = input.profile;
    let mut notes = Vec::new();

    if profile.schema_version != SCHEDULER_AUTOTUNE_SCHEMA_VERSION {
        notes.push(profile_health_error(
            "schema_version_mismatch",
            format!(
                "profile schema_version={} does not match expected {}",
                profile.schema_version, SCHEDULER_AUTOTUNE_SCHEMA_VERSION
            ),
        ));
    }

    if profile.hardware_label != input.expected_hardware_label {
        notes.push(profile_health_error(
            "hardware_label_mismatch",
            format!(
                "profile hardware_label={} does not match current hardware_label={}",
                profile.hardware_label, input.expected_hardware_label
            ),
        ));
    }

    if profile.runtime_context != *input.expected_runtime_context {
        notes.push(profile_health_error(
            "runtime_context_mismatch",
            format!(
                "profile runtime context {} does not match current runtime context {}",
                profile.runtime_context.fingerprint(),
                input.expected_runtime_context.fingerprint()
            ),
        ));
    }

    if profile.model_name != input.expected_model_name {
        notes.push(profile_health_warning(
            "model_name_mismatch",
            format!(
                "profile model_name={} differs from current model_name={}; exact model path store matches may still be valid",
                profile.model_name, input.expected_model_name
            ),
        ));
    }

    if profile.metadata.ironmlx_version != input.current_ironmlx_version {
        notes.push(profile_health_warning(
            "ironmlx_version_changed",
            format!(
                "profile was created by ironmlx {} but current version is {}; recalibration is recommended",
                profile.metadata.ironmlx_version, input.current_ironmlx_version
            ),
        ));
    }

    let max_age_ms = input.max_age_days.saturating_mul(MS_PER_DAY);
    if max_age_ms > 0
        && input
            .now_unix_ms
            .saturating_sub(profile.metadata.created_at_unix_ms)
            > max_age_ms
    {
        notes.push(profile_health_warning(
            "profile_stale",
            format!(
                "profile age exceeds {} day(s); rerun scheduler-autotune calibrate for a fresh baseline",
                input.max_age_days
            ),
        ));
    }

    let scenarios = &profile.metadata.scenario_coverage;
    if scenarios.is_empty() {
        notes.push(profile_health_warning(
            "no_scenario_coverage",
            "profile metadata has no calibration scenario coverage".to_string(),
        ));
    } else {
        if !scenarios.iter().any(|scenario| scenario.prompt_len >= 1024) {
            notes.push(profile_health_warning(
                "no_long_prompt_coverage",
                "profile has no PP>=1024 calibration scenario; agent workloads commonly use long prompts".to_string(),
            ));
        }
        if !scenarios.iter().any(|scenario| scenario.concurrency > 1) {
            notes.push(profile_health_warning(
                "no_concurrent_coverage",
                "profile has no concurrency>1 calibration scenario; queued-request TTFT was not validated".to_string(),
            ));
        }
        if profile.runtime_context.prefix_cache.enabled
            && !scenarios
                .iter()
                .any(|scenario| scenario.cache_state == SchedulerAutotuneCacheState::Warm)
        {
            notes.push(profile_health_warning(
                "no_warm_prefix_cache_coverage",
                "profile enables prefix cache but has no warm-cache calibration scenario"
                    .to_string(),
            ));
        }
    }

    for warning in &profile.metadata.selection_warnings {
        notes.push(profile_health_warning(
            format!("selection_warning_{}", warning.code),
            warning.message.clone(),
        ));
    }

    let status = if notes
        .iter()
        .any(|note| note.level == SchedulerAutotuneProfileHealthLevel::Error)
    {
        SchedulerAutotuneProfileHealthStatus::Invalid
    } else if notes
        .iter()
        .any(|note| note.level == SchedulerAutotuneProfileHealthLevel::Warning)
    {
        SchedulerAutotuneProfileHealthStatus::Warning
    } else {
        notes.push(profile_health_info(
            "profile_healthy",
            "profile metadata matches current model, hardware, version, age, and agent coverage requirements".to_string(),
        ));
        SchedulerAutotuneProfileHealthStatus::Healthy
    };

    SchedulerAutotuneProfileHealthReport {
        status,
        model_name: profile.model_name.clone(),
        hardware_label: profile.hardware_label.clone(),
        profile_context_fingerprint: profile.runtime_context.fingerprint(),
        expected_context_fingerprint: input.expected_runtime_context.fingerprint(),
        created_at_unix_ms: profile.metadata.created_at_unix_ms,
        max_age_days: input.max_age_days,
        notes,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerAutotuneMergeOptions {
    pub require_complete_coverage: bool,
}

impl Default for SchedulerAutotuneMergeOptions {
    fn default() -> Self {
        Self {
            require_complete_coverage: true,
        }
    }
}

pub fn merge_scheduler_autotune_calibrations(
    inputs: Vec<SchedulerAutotuneCalibrationInput>,
    options: SchedulerAutotuneMergeOptions,
) -> Result<SchedulerAutotuneCalibrationInput> {
    let mut iter = inputs.into_iter();
    let Some(mut merged) = iter.next() else {
        bail!("at least one calibration input is required");
    };

    validate_single_calibration(&merged, "input[0]")?;
    let objective = merged.objective.normalized();
    merged.objective = objective;

    for (idx, mut input) in iter.enumerate() {
        let label = format!("input[{}]", idx + 1);
        validate_single_calibration(&input, &label)?;
        input.objective = input.objective.normalized();

        if input.model_name != merged.model_name {
            bail!(
                "{label} model_name mismatch: expected {}, got {}",
                merged.model_name,
                input.model_name
            );
        }
        if input.hardware_label != merged.hardware_label {
            bail!(
                "{label} hardware_label mismatch: expected {}, got {}",
                merged.hardware_label,
                input.hardware_label
            );
        }
        if input.runtime_context != merged.runtime_context {
            bail!(
                "{label} runtime_context mismatch: expected {}, got {}",
                merged.runtime_context.fingerprint(),
                input.runtime_context.fingerprint()
            );
        }
        if input.objective != objective {
            bail!("{label} objective mismatch");
        }

        merged.measurements.extend(input.measurements);
    }

    if options.require_complete_coverage {
        validate_complete_scenario_coverage(&merged.measurements)?;
    }

    Ok(merged)
}

pub fn select_scheduler_autotune_profile(
    input: SchedulerAutotuneCalibrationInput,
) -> SchedulerAutotuneProfileSelection {
    select_scheduler_autotune_profile_with_options(
        input,
        SchedulerAutotuneSelectionOptions::default(),
    )
}

pub fn select_scheduler_autotune_profile_with_options(
    input: SchedulerAutotuneCalibrationInput,
    options: SchedulerAutotuneSelectionOptions,
) -> SchedulerAutotuneProfileSelection {
    let objective = input.objective.normalized();
    let runtime_context = input.runtime_context;
    let mut grouped: BTreeMap<SchedulerAutotuneProfileConfig, Vec<SchedulerAutotuneMeasurement>> =
        BTreeMap::new();
    for row in input.measurements {
        grouped.entry(row.config).or_default().push(row);
    }

    let mut rejected = Vec::new();
    let mut eligible: BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeMap<SchedulerAutotuneScenario, SchedulerAutotuneMeasurement>,
    > = BTreeMap::new();

    for (config, rows) in grouped {
        if rows.iter().any(|row| !row.memory_budget_ok) {
            rejected.push(rejected_candidate(
                config,
                "memory_budget_unsafe",
                "one or more calibration rows reported memory_budget_ok=false",
            ));
            continue;
        }
        if rows.iter().any(|row| {
            row.cache_state == SchedulerAutotuneCacheState::Cold && row.cached_tokens_warning
        }) {
            rejected.push(rejected_candidate(
                config,
                "cached_tokens_present",
                "one or more calibration rows reported cached_tokens_warning=true",
            ));
            continue;
        }
        if rows.iter().any(|row| !row.runtime_health.healthy) {
            rejected.push(rejected_candidate(
                config,
                "runtime_health_unsafe",
                "one or more calibration rows reported an unhealthy runtime delta",
            ));
            continue;
        }
        if rows
            .iter()
            .any(|row| !row.runtime_health.request_completion_ok)
        {
            rejected.push(rejected_candidate(
                config,
                "request_completion_failed",
                "one or more calibration rows did not complete benchmark requests",
            ));
            continue;
        }
        if runtime_context.speculative.mode != SchedulerSpeculativeMode::Disabled
            && rows.iter().all(|row| {
                row.runtime_health
                    .mtp
                    .as_ref()
                    .is_none_or(|mtp| mtp.drafted_tokens == 0)
            })
        {
            rejected.push(rejected_candidate(
                config,
                "speculative_path_inactive",
                "runtime context enables speculative decoding but no draft tokens were observed",
            ));
            continue;
        }

        let mut scenario_rows = BTreeMap::new();
        for row in rows {
            scenario_rows.insert(SchedulerAutotuneScenario::from(&row), row);
        }
        eligible.insert(config, scenario_rows);
    }

    let required_scenarios: BTreeSet<SchedulerAutotuneScenario> = eligible
        .values()
        .flat_map(|rows| rows.keys().cloned())
        .collect();

    let mut complete: BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeMap<SchedulerAutotuneScenario, SchedulerAutotuneMeasurement>,
    > = BTreeMap::new();
    for (config, rows) in eligible {
        let missing: Vec<SchedulerAutotuneScenario> = required_scenarios
            .iter()
            .filter(|scenario| !rows.contains_key(*scenario))
            .cloned()
            .collect();
        if missing.is_empty() {
            complete.insert(config, rows);
        } else {
            rejected.push(rejected_candidate(
                config,
                "missing_scenario_coverage",
                format!("candidate is missing {} scenario(s)", missing.len()),
            ));
        }
    }

    let mut warnings = coverage_warnings(&required_scenarios, &runtime_context);
    if complete.len() == 1 {
        warnings.push(selection_note(
            "single_candidate",
            "only one complete candidate remained after filtering; treat selection as validation, not comparison",
        ));
    }

    let candidates =
        score_complete_candidates(&complete, &required_scenarios, objective, options.profile);
    let selected = candidates.first().cloned();
    let scenario_overrides = selected
        .as_ref()
        .map(|selected| {
            build_scenario_overrides(&complete, &required_scenarios, objective, &selected.config)
        })
        .unwrap_or_default();
    if selected.is_none() {
        warnings.push(selection_note(
            "no_valid_profile",
            "no complete memory-safe profile candidate could be selected",
        ));
    }

    SchedulerAutotuneProfileSelection {
        diagnose_only: true,
        model_name: input.model_name,
        hardware_label: input.hardware_label,
        runtime_context,
        selection_profile: options.profile,
        objective,
        scenarios: required_scenarios.iter().cloned().collect(),
        selected,
        candidates,
        scenario_overrides,
        rejected,
        warnings,
    }
}

impl SchedulerAutotuneProfileSelection {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "scheduler/autotune profile selection (diagnose-only; no runtime parameters changed)"
        )
        .unwrap();
        writeln!(out, "model: {}", self.model_name).unwrap();
        writeln!(out, "hardware: {}", self.hardware_label).unwrap();
        writeln!(
            out,
            "selection_profile: {}",
            self.selection_profile.as_str()
        )
        .unwrap();
        writeln!(
            out,
            "objective: ttft_p95={:.2} itl_p95={:.2} e2e_p95={:.2} throughput={:.2}",
            self.objective.ttft_p95_weight,
            self.objective.itl_p95_weight,
            self.objective.e2e_p95_weight,
            self.objective.throughput_weight
        )
        .unwrap();

        match &self.selected {
            Some(selected) => {
                writeln!(
                    out,
                    "selected: b_max={} prefill_chunk_size={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} decode_cadence_mid_chunk_cap={} score={:.4}",
                    selected.config.b_max,
                    selected.config.prefill_chunk_size,
                    selected.config.admission_deadline_ms,
                    selected.config.admission_queue_max,
                    selected.config.max_cache_cap,
                    selected.config.decode_cadence_mid_chunk_cap,
                    selected.score
                )
                .unwrap();
            }
            None => {
                writeln!(out, "selected: (none)").unwrap();
            }
        }

        writeln!(out, "candidates:").unwrap();
        for item in &self.candidates {
            writeln!(
                out,
                "- b_max={} chunk={} deadline_ms={} queue_max={} cap={} decode_cadence_cap={} score={:.4} scenarios={} early_itl_norm={:.4}",
                item.config.b_max,
                item.config.prefill_chunk_size,
                item.config.admission_deadline_ms,
                item.config.admission_queue_max,
                item.config.max_cache_cap,
                item.config.decode_cadence_mid_chunk_cap,
                item.score,
                item.scenario_count,
                item.mean_early_itl_norm,
            )
            .unwrap();
        }

        writeln!(out, "scenario_overrides:").unwrap();
        if self.scenario_overrides.is_empty() {
            writeln!(out, "- none").unwrap();
        } else {
            for item in &self.scenario_overrides {
                writeln!(
                    out,
                    "- PP>={} TG>={} C>={}: b_max={} chunk={} deadline_ms={} queue_max={} cap={} decode_cadence_cap={} score={:.4} baseline_score={:.4}",
                    item.scenario.prompt_len,
                    item.scenario.max_new_tokens,
                    item.scenario.concurrency,
                    item.config.b_max,
                    item.config.prefill_chunk_size,
                    item.config.admission_deadline_ms,
                    item.config.admission_queue_max,
                    item.config.max_cache_cap,
                    item.config.decode_cadence_mid_chunk_cap,
                    item.score,
                    item.baseline_score
                )
                .unwrap();
            }
        }

        writeln!(out, "rejected:").unwrap();
        if self.rejected.is_empty() {
            writeln!(out, "- none").unwrap();
        } else {
            for item in &self.rejected {
                writeln!(
                    out,
                    "- {} b_max={} chunk={} deadline_ms={} decode_cadence_cap={}: {}",
                    item.code,
                    item.config.b_max,
                    item.config.prefill_chunk_size,
                    item.config.admission_deadline_ms,
                    item.config.decode_cadence_mid_chunk_cap,
                    item.message
                )
                .unwrap();
            }
        }

        writeln!(out, "warnings:").unwrap();
        if self.warnings.is_empty() {
            writeln!(out, "- none").unwrap();
        } else {
            for item in &self.warnings {
                writeln!(out, "- {}: {}", item.code, item.message).unwrap();
            }
        }
        out
    }
}

pub fn build_scheduler_autotune_runtime_profile(
    selection: &SchedulerAutotuneProfileSelection,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    build_scheduler_autotune_runtime_profile_at(selection, unix_time_ms())
}

pub fn build_scheduler_autotune_runtime_profile_at(
    selection: &SchedulerAutotuneProfileSelection,
    created_at_unix_ms: u64,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    let selected = selection
        .selected
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("selected scheduler profile is required"))?;

    Ok(SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: selection.model_name.clone(),
        hardware_label: selection.hardware_label.clone(),
        runtime_context: selection.runtime_context.clone(),
        config: selected.config,
        rules: runtime_rules_from_overrides(&selection.scenario_overrides),
        metadata: SchedulerAutotuneRuntimeProfileMetadata {
            created_at_unix_ms,
            ironmlx_version: env!("CARGO_PKG_VERSION").to_string(),
            selection_profile: selection.selection_profile,
            objective: selection.objective,
            scenario_coverage: selection.scenarios.clone(),
            selected_score: selected.score,
            candidate_count: selection.candidates.len(),
            rejected_count: selection.rejected.len(),
            selection_warnings: selection.warnings.clone(),
        },
    })
}

fn runtime_rules_from_overrides(
    overrides: &[SchedulerAutotuneScenarioOverride],
) -> Vec<SchedulerAutotuneRuntimeRule> {
    let mut rules = overrides
        .iter()
        .filter(|item| item.scenario.cache_state == SchedulerAutotuneCacheState::Cold)
        .map(|item| SchedulerAutotuneRuntimeRule {
            when: SchedulerAutotuneRuntimeRuleCondition {
                prompt_len_gte: item.scenario.prompt_len,
                max_new_tokens_gte: item.scenario.max_new_tokens,
                effective_concurrency_gte: item.scenario.concurrency,
            },
            config: item.config,
        })
        .collect::<Vec<_>>();
    rules.sort_by_key(|rule| std::cmp::Reverse(rule.when.specificity_key()));
    compress_runtime_rules(&mut rules);
    rules
}

fn compress_runtime_rules(rules: &mut Vec<SchedulerAutotuneRuntimeRule>) {
    let probes = runtime_rule_probe_requests(rules);
    let mut idx = 0;
    while idx < rules.len() {
        let mut candidate = rules.clone();
        candidate.remove(idx);
        if runtime_rules_equivalent_on_probes(rules, &candidate, &probes) {
            *rules = candidate;
        } else {
            idx += 1;
        }
    }
}

fn runtime_rule_probe_requests(
    rules: &[SchedulerAutotuneRuntimeRule],
) -> Vec<SchedulerAutotuneRuntimeRequest> {
    let mut prompt_lens = BTreeSet::new();
    let mut max_new_tokens = BTreeSet::new();
    let mut concurrency = BTreeSet::new();
    for rule in rules {
        insert_runtime_probe_values(&mut prompt_lens, rule.when.prompt_len_gte);
        insert_runtime_probe_values(&mut max_new_tokens, rule.when.max_new_tokens_gte);
        insert_runtime_probe_values(&mut concurrency, rule.when.effective_concurrency_gte);
    }

    let mut probes = Vec::new();
    for prompt_len in prompt_lens {
        for max_new_tokens in &max_new_tokens {
            for effective_concurrency in &concurrency {
                probes.push(SchedulerAutotuneRuntimeRequest {
                    prompt_len,
                    max_new_tokens: *max_new_tokens,
                    effective_concurrency: *effective_concurrency,
                });
            }
        }
    }
    probes
}

fn insert_runtime_probe_values(values: &mut BTreeSet<usize>, threshold: usize) {
    values.insert(threshold);
    if threshold > 0 {
        values.insert(threshold - 1);
    }
}

fn runtime_rules_equivalent_on_probes(
    before: &[SchedulerAutotuneRuntimeRule],
    after: &[SchedulerAutotuneRuntimeRule],
    probes: &[SchedulerAutotuneRuntimeRequest],
) -> bool {
    probes.iter().all(|request| {
        runtime_rules_select(before, *request) == runtime_rules_select(after, *request)
    })
}

fn runtime_rules_select(
    rules: &[SchedulerAutotuneRuntimeRule],
    request: SchedulerAutotuneRuntimeRequest,
) -> Option<SchedulerAutotuneProfileConfig> {
    rules
        .iter()
        .find(|rule| rule.when.matches(request))
        .map(|rule| rule.config)
}

fn score_complete_candidates(
    complete: &BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeMap<SchedulerAutotuneScenario, SchedulerAutotuneMeasurement>,
    >,
    required_scenarios: &BTreeSet<SchedulerAutotuneScenario>,
    objective: SchedulerAutotuneObjective,
    selection_profile: SchedulerAutotuneSelectionProfile,
) -> Vec<SchedulerAutotuneCandidateScore> {
    let mut best_by_scenario: BTreeMap<SchedulerAutotuneScenario, ScenarioBest> = BTreeMap::new();
    for scenario in required_scenarios {
        let mut best = ScenarioBest::default();
        for rows in complete.values() {
            if let Some(row) = rows.get(scenario) {
                best.observe(row);
            }
        }
        best_by_scenario.insert(scenario.clone(), best);
    }

    let mut scored = Vec::new();
    for (config, rows) in complete {
        let mut score_sum = 0.0;
        let mut scenario_weight_sum = 0.0;
        let mut ttft_norm_sum = 0.0;
        let mut itl_norm_sum = 0.0;
        let mut early_itl_norm_sum = 0.0;
        let mut e2e_norm_sum = 0.0;
        let mut throughput_norm_sum = 0.0;
        for scenario in required_scenarios {
            let Some(row) = rows.get(scenario) else {
                continue;
            };
            let Some(best) = best_by_scenario.get(scenario) else {
                continue;
            };
            let ttft_norm = row.ttft_ms_p95 / nonzero(best.ttft_ms_p95);
            let itl_norm = row.itl_ms_p95 / nonzero(best.itl_ms_p95);
            let early_itl_norm = row.early_itl_ms_p95 / nonzero(best.early_itl_ms_p95);
            let e2e_norm = row.e2e_s_p95 / nonzero(best.e2e_s_p95);
            let throughput_norm = nonzero(best.tokens_per_sec) / nonzero(row.tokens_per_sec);
            let scenario_score = objective.ttft_p95_weight * ttft_norm
                + objective.itl_p95_weight * itl_norm
                + objective.e2e_p95_weight * e2e_norm
                + objective.throughput_weight * throughput_norm;
            let scenario_weight = selection_profile.scenario_weight(scenario);
            score_sum += scenario_weight * scenario_score;
            scenario_weight_sum += scenario_weight;
            ttft_norm_sum += scenario_weight * ttft_norm;
            itl_norm_sum += scenario_weight * itl_norm;
            early_itl_norm_sum += scenario_weight * early_itl_norm;
            e2e_norm_sum += scenario_weight * e2e_norm;
            throughput_norm_sum += scenario_weight * throughput_norm;
        }
        let n = if scenario_weight_sum > f64::EPSILON {
            scenario_weight_sum
        } else {
            1.0
        };
        scored.push(SchedulerAutotuneCandidateScore {
            config: *config,
            score: score_sum / n,
            scenario_count: required_scenarios.len(),
            mean_ttft_norm: ttft_norm_sum / n,
            mean_itl_norm: itl_norm_sum / n,
            mean_early_itl_norm: early_itl_norm_sum / n,
            mean_e2e_norm: e2e_norm_sum / n,
            mean_throughput_norm: throughput_norm_sum / n,
        });
    }

    scored.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.config.cmp(&b.config))
    });
    scored
}

fn build_scenario_overrides(
    complete: &BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeMap<SchedulerAutotuneScenario, SchedulerAutotuneMeasurement>,
    >,
    required_scenarios: &BTreeSet<SchedulerAutotuneScenario>,
    objective: SchedulerAutotuneObjective,
    baseline_config: &SchedulerAutotuneProfileConfig,
) -> Vec<SchedulerAutotuneScenarioOverride> {
    let mut overrides = Vec::new();
    for scenario in required_scenarios {
        let Some(baseline_row) = complete
            .get(baseline_config)
            .and_then(|rows| rows.get(scenario))
        else {
            continue;
        };

        let mut best = ScenarioBest::default();
        for rows in complete.values() {
            if let Some(row) = rows.get(scenario) {
                best.observe(row);
            }
        }

        let baseline_score = scenario_score(baseline_row, &best, objective);
        let mut scenario_winner: Option<(SchedulerAutotuneProfileConfig, f64)> = None;
        for (config, rows) in complete {
            if !request_runtime_switchable(baseline_config, config) {
                continue;
            }
            let Some(row) = rows.get(scenario) else {
                continue;
            };
            let score = scenario_score(row, &best, objective);
            match scenario_winner {
                Some((winner_config, winner_score))
                    if score > winner_score
                        || ((score - winner_score).abs() <= f64::EPSILON
                            && *config >= winner_config) => {}
                _ => scenario_winner = Some((*config, score)),
            }
        }

        let Some((winner_config, winner_score)) = scenario_winner else {
            continue;
        };
        if winner_config != *baseline_config && winner_score < baseline_score {
            overrides.push(SchedulerAutotuneScenarioOverride {
                scenario: scenario.clone(),
                config: winner_config,
                score: winner_score,
                baseline_score,
            });
        }
    }
    overrides
}

fn request_runtime_switchable(
    baseline: &SchedulerAutotuneProfileConfig,
    candidate: &SchedulerAutotuneProfileConfig,
) -> bool {
    baseline.b_max == candidate.b_max
        && baseline.admission_deadline_ms == candidate.admission_deadline_ms
        && baseline.admission_queue_max == candidate.admission_queue_max
        && baseline.max_cache_cap == candidate.max_cache_cap
}

fn scenario_score(
    row: &SchedulerAutotuneMeasurement,
    best: &ScenarioBest,
    objective: SchedulerAutotuneObjective,
) -> f64 {
    let ttft_norm = row.ttft_ms_p95 / nonzero(best.ttft_ms_p95);
    let itl_norm = row.itl_ms_p95 / nonzero(best.itl_ms_p95);
    let e2e_norm = row.e2e_s_p95 / nonzero(best.e2e_s_p95);
    let throughput_norm = nonzero(best.tokens_per_sec) / nonzero(row.tokens_per_sec);
    objective.ttft_p95_weight * ttft_norm
        + objective.itl_p95_weight * itl_norm
        + objective.e2e_p95_weight * e2e_norm
        + objective.throughput_weight * throughput_norm
}

#[derive(Debug, Clone, Copy)]
struct ScenarioBest {
    ttft_ms_p95: f64,
    itl_ms_p95: f64,
    early_itl_ms_p95: f64,
    e2e_s_p95: f64,
    tokens_per_sec: f64,
}

impl Default for ScenarioBest {
    fn default() -> Self {
        Self {
            ttft_ms_p95: f64::INFINITY,
            itl_ms_p95: f64::INFINITY,
            early_itl_ms_p95: f64::INFINITY,
            e2e_s_p95: f64::INFINITY,
            tokens_per_sec: 0.0,
        }
    }
}

impl ScenarioBest {
    fn observe(&mut self, row: &SchedulerAutotuneMeasurement) {
        self.ttft_ms_p95 = self.ttft_ms_p95.min(nonzero(row.ttft_ms_p95));
        self.itl_ms_p95 = self.itl_ms_p95.min(nonzero(row.itl_ms_p95));
        self.early_itl_ms_p95 = self.early_itl_ms_p95.min(nonzero(row.early_itl_ms_p95));
        self.e2e_s_p95 = self.e2e_s_p95.min(nonzero(row.e2e_s_p95));
        self.tokens_per_sec = self.tokens_per_sec.max(nonzero(row.tokens_per_sec));
    }
}

fn coverage_warnings(
    required_scenarios: &BTreeSet<SchedulerAutotuneScenario>,
    runtime_context: &SchedulerAutotuneRuntimeContext,
) -> Vec<SchedulerAutotuneSelectionNote> {
    let mut warnings = Vec::new();
    if !required_scenarios
        .iter()
        .any(|scenario| scenario.prompt_len >= 1024)
    {
        warnings.push(selection_note(
            "no_long_prompt_coverage",
            "calibration input has no PP>=1024 scenario; agent workloads commonly use long prompts",
        ));
    }
    if !required_scenarios
        .iter()
        .any(|scenario| scenario.concurrency > 1)
    {
        warnings.push(selection_note(
            "no_concurrent_coverage",
            "calibration input has no concurrency>1 scenario; queued-request TTFT cannot be assessed",
        ));
    }
    if runtime_context.prefix_cache.enabled
        && !required_scenarios
            .iter()
            .any(|scenario| scenario.cache_state == SchedulerAutotuneCacheState::Warm)
    {
        warnings.push(selection_note(
            "no_warm_prefix_cache_coverage",
            "runtime context enables prefix cache but calibration has no warm-cache scenario",
        ));
    }
    warnings
}

fn validate_single_calibration(
    input: &SchedulerAutotuneCalibrationInput,
    label: &str,
) -> Result<()> {
    if input.schema_version != SCHEDULER_AUTOTUNE_SCHEMA_VERSION {
        bail!(
            "{label} schema_version mismatch: expected {}, got {}",
            SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            input.schema_version
        );
    }
    if input.model_name.trim().is_empty() {
        bail!("{label} model_name must not be empty");
    }
    if input.hardware_label.trim().is_empty() {
        bail!("{label} hardware_label must not be empty");
    }
    if input.runtime_context.model_architecture.trim().is_empty() {
        bail!("{label} runtime_context.model_architecture must not be empty");
    }
    if input.runtime_context.model_fingerprint.trim().is_empty() {
        bail!("{label} runtime_context.model_fingerprint must not be empty");
    }
    if input.measurements.is_empty() {
        bail!("{label} measurements must not be empty");
    }
    Ok(())
}

fn validate_complete_scenario_coverage(
    measurements: &[SchedulerAutotuneMeasurement],
) -> Result<()> {
    let mut scenarios_by_config: BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeSet<SchedulerAutotuneScenario>,
    > = BTreeMap::new();
    for row in measurements {
        scenarios_by_config
            .entry(row.config)
            .or_default()
            .insert(SchedulerAutotuneScenario::from(row));
    }

    let mut iter = scenarios_by_config.into_iter();
    let Some((baseline_config, baseline_scenarios)) = iter.next() else {
        return Ok(());
    };

    for (config, scenarios) in iter {
        if scenarios != baseline_scenarios {
            let missing = baseline_scenarios.difference(&scenarios).count();
            let extra = scenarios.difference(&baseline_scenarios).count();
            bail!(
                "scenario coverage mismatch for b_max={} prefill_chunk_size={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} decode_cadence_mid_chunk_cap={} against baseline b_max={} prefill_chunk_size={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} decode_cadence_mid_chunk_cap={}: missing {} scenario(s), extra {} scenario(s)",
                config.b_max,
                config.prefill_chunk_size,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                baseline_config.b_max,
                baseline_config.prefill_chunk_size,
                baseline_config.admission_deadline_ms,
                baseline_config.admission_queue_max,
                baseline_config.max_cache_cap,
                baseline_config.decode_cadence_mid_chunk_cap,
                missing,
                extra
            );
        }
    }
    Ok(())
}

fn rejected_candidate(
    config: SchedulerAutotuneProfileConfig,
    code: &'static str,
    message: impl Into<String>,
) -> SchedulerAutotuneRejectedCandidate {
    SchedulerAutotuneRejectedCandidate {
        config,
        code: code.to_string(),
        message: message.into(),
    }
}

fn selection_note(
    code: &'static str,
    message: impl Into<String>,
) -> SchedulerAutotuneSelectionNote {
    SchedulerAutotuneSelectionNote {
        code: code.to_string(),
        message: message.into(),
    }
}

fn profile_health_info(
    code: impl Into<String>,
    message: impl Into<String>,
) -> SchedulerAutotuneProfileHealthNote {
    SchedulerAutotuneProfileHealthNote {
        level: SchedulerAutotuneProfileHealthLevel::Info,
        code: code.into(),
        message: message.into(),
    }
}

fn profile_health_warning(
    code: impl Into<String>,
    message: impl Into<String>,
) -> SchedulerAutotuneProfileHealthNote {
    SchedulerAutotuneProfileHealthNote {
        level: SchedulerAutotuneProfileHealthLevel::Warning,
        code: code.into(),
        message: message.into(),
    }
}

fn profile_health_error(
    code: impl Into<String>,
    message: impl Into<String>,
) -> SchedulerAutotuneProfileHealthNote {
    SchedulerAutotuneProfileHealthNote {
        level: SchedulerAutotuneProfileHealthLevel::Error,
        code: code.into(),
        message: message.into(),
    }
}

fn nonzero(value: f64) -> f64 {
    if value.is_finite() && value > 1e-9 {
        value
    } else {
        1e-9
    }
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}
