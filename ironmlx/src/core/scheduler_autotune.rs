//! Diagnose-only scheduler/autotune reporting.
//!
//! This module intentionally does not mutate runtime configuration. It turns
//! the current `serve` parameters, model metadata, and model-level scheduler
//! policy hooks into a startup report that can guide later benchmark sweeps.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::core::memory_budget::{
    kv_bytes_per_token, ModelMeta, SAFETY_MARGIN_BYTES, SOFT_LIMIT_FRAC,
};
use crate::core::Model;
use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneMeasurement {
    pub config: SchedulerAutotuneProfileConfig,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub concurrency: usize,
    pub ttft_ms_p95: f64,
    pub itl_ms_p95: f64,
    pub e2e_s_p95: f64,
    pub tokens_per_sec: f64,
    #[serde(default = "default_true")]
    pub memory_budget_ok: bool,
    #[serde(default)]
    pub cached_tokens_warning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchedulerAutotuneScenario {
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub concurrency: usize,
}

impl From<&SchedulerAutotuneMeasurement> for SchedulerAutotuneScenario {
    fn from(value: &SchedulerAutotuneMeasurement) -> Self {
        Self {
            prompt_len: value.prompt_len,
            max_new_tokens: value.max_new_tokens,
            concurrency: value.concurrency,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneCalibrationInput {
    pub schema_version: u32,
    pub model_name: String,
    pub hardware_label: String,
    #[serde(default)]
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
pub struct SchedulerAutotuneProfileSelection {
    pub diagnose_only: bool,
    pub model_name: String,
    pub hardware_label: String,
    pub objective: SchedulerAutotuneObjective,
    pub selected: Option<SchedulerAutotuneCandidateScore>,
    pub candidates: Vec<SchedulerAutotuneCandidateScore>,
    pub rejected: Vec<SchedulerAutotuneRejectedCandidate>,
    pub warnings: Vec<SchedulerAutotuneSelectionNote>,
}

pub fn select_scheduler_autotune_profile(
    input: SchedulerAutotuneCalibrationInput,
) -> SchedulerAutotuneProfileSelection {
    let objective = input.objective.normalized();
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
        if rows.iter().any(|row| row.cached_tokens_warning) {
            rejected.push(rejected_candidate(
                config,
                "cached_tokens_present",
                "one or more calibration rows reported cached_tokens_warning=true",
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

    let mut warnings = coverage_warnings(&required_scenarios);
    if complete.len() == 1 {
        warnings.push(selection_note(
            "single_candidate",
            "only one complete candidate remained after filtering; treat selection as validation, not comparison",
        ));
    }

    let candidates = score_complete_candidates(&complete, &required_scenarios, objective);
    let selected = candidates.first().cloned();
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
        objective,
        selected,
        candidates,
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
                    "selected: b_max={} prefill_chunk_size={} admission_deadline_ms={} admission_queue_max={} max_cache_cap={} score={:.4}",
                    selected.config.b_max,
                    selected.config.prefill_chunk_size,
                    selected.config.admission_deadline_ms,
                    selected.config.admission_queue_max,
                    selected.config.max_cache_cap,
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
                "- b_max={} chunk={} deadline_ms={} queue_max={} cap={} score={:.4} scenarios={}",
                item.config.b_max,
                item.config.prefill_chunk_size,
                item.config.admission_deadline_ms,
                item.config.admission_queue_max,
                item.config.max_cache_cap,
                item.score,
                item.scenario_count,
            )
            .unwrap();
        }

        writeln!(out, "rejected:").unwrap();
        if self.rejected.is_empty() {
            writeln!(out, "- none").unwrap();
        } else {
            for item in &self.rejected {
                writeln!(
                    out,
                    "- {} b_max={} chunk={} deadline_ms={}: {}",
                    item.code,
                    item.config.b_max,
                    item.config.prefill_chunk_size,
                    item.config.admission_deadline_ms,
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

fn score_complete_candidates(
    complete: &BTreeMap<
        SchedulerAutotuneProfileConfig,
        BTreeMap<SchedulerAutotuneScenario, SchedulerAutotuneMeasurement>,
    >,
    required_scenarios: &BTreeSet<SchedulerAutotuneScenario>,
    objective: SchedulerAutotuneObjective,
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
        let mut ttft_norm_sum = 0.0;
        let mut itl_norm_sum = 0.0;
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
            let e2e_norm = row.e2e_s_p95 / nonzero(best.e2e_s_p95);
            let throughput_norm = nonzero(best.tokens_per_sec) / nonzero(row.tokens_per_sec);
            let scenario_score = objective.ttft_p95_weight * ttft_norm
                + objective.itl_p95_weight * itl_norm
                + objective.e2e_p95_weight * e2e_norm
                + objective.throughput_weight * throughput_norm;
            score_sum += scenario_score;
            ttft_norm_sum += ttft_norm;
            itl_norm_sum += itl_norm;
            e2e_norm_sum += e2e_norm;
            throughput_norm_sum += throughput_norm;
        }
        let n = required_scenarios.len().max(1) as f64;
        scored.push(SchedulerAutotuneCandidateScore {
            config: *config,
            score: score_sum / n,
            scenario_count: required_scenarios.len(),
            mean_ttft_norm: ttft_norm_sum / n,
            mean_itl_norm: itl_norm_sum / n,
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

#[derive(Debug, Clone, Copy)]
struct ScenarioBest {
    ttft_ms_p95: f64,
    itl_ms_p95: f64,
    e2e_s_p95: f64,
    tokens_per_sec: f64,
}

impl Default for ScenarioBest {
    fn default() -> Self {
        Self {
            ttft_ms_p95: f64::INFINITY,
            itl_ms_p95: f64::INFINITY,
            e2e_s_p95: f64::INFINITY,
            tokens_per_sec: 0.0,
        }
    }
}

impl ScenarioBest {
    fn observe(&mut self, row: &SchedulerAutotuneMeasurement) {
        self.ttft_ms_p95 = self.ttft_ms_p95.min(nonzero(row.ttft_ms_p95));
        self.itl_ms_p95 = self.itl_ms_p95.min(nonzero(row.itl_ms_p95));
        self.e2e_s_p95 = self.e2e_s_p95.min(nonzero(row.e2e_s_p95));
        self.tokens_per_sec = self.tokens_per_sec.max(nonzero(row.tokens_per_sec));
    }
}

fn coverage_warnings(
    required_scenarios: &BTreeSet<SchedulerAutotuneScenario>,
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
    warnings
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

fn nonzero(value: f64) -> f64 {
    if value.is_finite() && value > 1e-9 {
        value
    } else {
        1e-9
    }
}

fn default_true() -> bool {
    true
}
