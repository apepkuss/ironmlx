//! Offline Gemma4 draft-cap calibration from bounded benchmark observations.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::speculative::{MtpDraftCapContextBucket, MtpDraftCapObservation};

pub const MTP_DRAFT_CAP_CALIBRATION_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Deserialize, Serialize)]
pub struct MtpDraftCapBenchInput {
    pub meta: MtpDraftCapBenchMeta,
    pub records: Vec<MtpDraftCapBenchRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MtpDraftCapBenchMeta {
    pub backend: String,
    pub mode: String,
    pub speculative_source: Option<String>,
    pub model_dir: String,
    pub mtp_model_dir: Option<String>,
    pub mtp_draft_tokens: Option<usize>,
    pub mtp_trace_windows: usize,
    pub prompt_file: String,
    pub prompt_tokens: usize,
    pub scheduler_prompt_files: Vec<String>,
    pub scheduler_prompt_tokens: Vec<usize>,
    pub scheduler_batch_width: usize,
    pub max_tokens: usize,
    pub prefill_chunk_size: usize,
    pub kv_quant: String,
    pub paged_prefix_cache_dir: Option<String>,
    pub paged_prefix_cache_block_size: i32,
    pub paged_prefix_cache_max_pages: Option<i32>,
    pub active_kv_offload: bool,
    pub b_max: usize,
    pub effective_cap_max: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub device_name: Option<String>,
    pub ironmlx_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MtpDraftCapBenchRecord {
    pub valid: bool,
    pub generated_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: Option<String>,
    pub scheduler_requests: Vec<MtpDraftCapBenchRequest>,
    pub mtp_stats: Option<MtpDraftCapBenchStats>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MtpDraftCapBenchRequest {
    pub request_index: usize,
    pub prompt_file: String,
    pub generated_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: Option<String>,
    pub valid: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MtpDraftCapBenchStats {
    pub draft_cap_observations: Vec<MtpDraftCapObservation>,
    pub draft_cap_observation_dropped_windows: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MtpDraftCapCalibrationReport {
    pub schema_version: u32,
    pub runtime: MtpDraftCapRuntimeContext,
    pub input_files: usize,
    pub valid_records: usize,
    pub min_windows: usize,
    pub min_records: usize,
    pub min_improvement_percent: f64,
    pub ignored: MtpDraftCapIgnoredCoverage,
    pub recommendations: Vec<MtpDraftCapRegimeRecommendation>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MtpDraftCapRuntimeContext {
    pub mode: String,
    pub model_dir: String,
    pub drafter_dir: String,
    pub device_name: String,
    pub ironmlx_version: String,
    pub prompt_file: String,
    pub prompt_tokens: usize,
    pub scheduler_prompt_files: Vec<String>,
    pub scheduler_prompt_tokens: Vec<usize>,
    pub scheduler_batch_width: usize,
    pub max_tokens: usize,
    pub prefill_chunk_size: usize,
    pub kv_quant: String,
    pub mtp_trace_windows: usize,
    pub b_max: usize,
    pub effective_cap_max: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MtpDraftCapCalibrationConfig {
    pub min_windows: usize,
    pub min_records: usize,
    pub min_improvement_percent: f64,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct MtpDraftCapIgnoredCoverage {
    pub invalid_records: usize,
    pub missing_observation_records: usize,
    pub mixed_context_windows: usize,
    pub zero_time_windows: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MtpDraftCapRegimeRecommendation {
    pub batch_width: usize,
    pub context_bucket: MtpDraftCapContextBucket,
    pub status: MtpDraftCapRecommendationStatus,
    pub best_observed_cap: Option<usize>,
    pub recommended_cap: Option<usize>,
    pub candidates: Vec<MtpDraftCapCandidate>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MtpDraftCapRecommendationStatus {
    Recommended,
    InsufficientCapCoverage,
    InsufficientWindows,
    InsufficientRecords,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MtpDraftCapCandidate {
    pub cap: usize,
    pub records: usize,
    pub windows: usize,
    pub full_cap_windows: usize,
    pub adaptive_lowered_windows: usize,
    pub mixed_depth_windows: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub rollback_count: usize,
    pub acceptance_rate: f64,
    pub rollback_rate: f64,
    pub mean_draft_tokens_per_window: f64,
    pub mean_window_us: f64,
    pub committed_tokens_per_second: f64,
    pub total_us: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub main_rollback_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateKey {
    batch_width: usize,
    context_bucket: MtpDraftCapContextBucket,
    cap: usize,
}

struct CandidateAccumulator {
    observation: MtpDraftCapObservation,
    records: usize,
    full_cap_windows: usize,
    adaptive_lowered_windows: usize,
    mixed_depth_windows: usize,
}

impl CandidateAccumulator {
    fn new(observation: MtpDraftCapObservation) -> Self {
        let (full_cap_windows, adaptive_lowered_windows, mixed_depth_windows) =
            depth_coverage(&observation);
        Self {
            observation,
            records: 0,
            full_cap_windows,
            adaptive_lowered_windows,
            mixed_depth_windows,
        }
    }

    fn add_observation(&mut self, observation: &MtpDraftCapObservation) {
        let (full_cap_windows, adaptive_lowered_windows, mixed_depth_windows) =
            depth_coverage(observation);
        add_observation(&mut self.observation, observation);
        self.full_cap_windows = self.full_cap_windows.saturating_add(full_cap_windows);
        self.adaptive_lowered_windows = self
            .adaptive_lowered_windows
            .saturating_add(adaptive_lowered_windows);
        self.mixed_depth_windows = self.mixed_depth_windows.saturating_add(mixed_depth_windows);
    }

    fn merge_record(&mut self, record: &Self) {
        add_observation(&mut self.observation, &record.observation);
        self.full_cap_windows = self
            .full_cap_windows
            .saturating_add(record.full_cap_windows);
        self.adaptive_lowered_windows = self
            .adaptive_lowered_windows
            .saturating_add(record.adaptive_lowered_windows);
        self.mixed_depth_windows = self
            .mixed_depth_windows
            .saturating_add(record.mixed_depth_windows);
    }
}

fn depth_coverage(observation: &MtpDraftCapObservation) -> (usize, usize, usize) {
    if observation.min_draft_tokens == observation.configured_max_draft_tokens
        && observation.max_draft_tokens == observation.configured_max_draft_tokens
    {
        (observation.windows, 0, 0)
    } else if observation.max_draft_tokens < observation.configured_max_draft_tokens {
        (0, observation.windows, 0)
    } else {
        (0, 0, observation.windows)
    }
}

pub fn calibrate_mtp_draft_cap(
    inputs: Vec<MtpDraftCapBenchInput>,
    config: MtpDraftCapCalibrationConfig,
) -> Result<MtpDraftCapCalibrationReport> {
    if inputs.is_empty() {
        bail!("at least one benchmark input is required");
    }
    if config.min_windows == 0 {
        bail!("min_windows must be greater than zero");
    }
    if config.min_records == 0 {
        bail!("min_records must be greater than zero");
    }
    if !config.min_improvement_percent.is_finite() || config.min_improvement_percent < 0.0 {
        bail!("min_improvement_percent must be finite and non-negative");
    }

    let first = inputs.first().expect("inputs checked non-empty");
    validate_input(first, "input[0]")?;
    let drafter_dir = first
        .meta
        .mtp_model_dir
        .clone()
        .context("input[0] is missing mtp_model_dir")?;
    let device_name = first
        .meta
        .device_name
        .clone()
        .context("input[0] is missing device_name")?;
    for (idx, input) in inputs.iter().enumerate().skip(1) {
        let label = format!("input[{idx}]");
        validate_input(input, &label)?;
        ensure_same_runtime(&first.meta, &input.meta, &label)?;
    }

    let mut ignored = MtpDraftCapIgnoredCoverage::default();
    let mut valid_records = 0;
    let mut expected_outputs: Option<Vec<(Vec<u32>, Option<String>)>> = None;
    let mut grouped: BTreeMap<CandidateKey, CandidateAccumulator> = BTreeMap::new();
    for (input_idx, input) in inputs.iter().enumerate() {
        for (record_idx, record) in input.records.iter().enumerate() {
            if !record.valid {
                ignored.invalid_records = ignored.invalid_records.saturating_add(1);
                continue;
            }
            validate_record_output(
                record,
                &input.meta,
                input_idx,
                record_idx,
                &mut expected_outputs,
            )?;
            valid_records += 1;
            let Some(stats) = record.mtp_stats.as_ref() else {
                ignored.missing_observation_records =
                    ignored.missing_observation_records.saturating_add(1);
                continue;
            };
            if stats.draft_cap_observation_dropped_windows > 0 {
                bail!(
                    "benchmark observation table overflowed by {} window(s); split the workload before calibration",
                    stats.draft_cap_observation_dropped_windows
                );
            }
            if stats.draft_cap_observations.is_empty() {
                ignored.missing_observation_records =
                    ignored.missing_observation_records.saturating_add(1);
            }
            let mut record_grouped = BTreeMap::<CandidateKey, CandidateAccumulator>::new();
            for observation in &stats.draft_cap_observations {
                validate_observation(observation, input_idx, record_idx)?;
                if observation.batch_width > input.meta.scheduler_batch_width {
                    bail!(
                        "input[{input_idx}].records[{record_idx}] observation batch_width {} exceeds scheduler_batch_width {}",
                        observation.batch_width,
                        input.meta.scheduler_batch_width
                    );
                }
                if Some(observation.configured_max_draft_tokens) != input.meta.mtp_draft_tokens {
                    bail!(
                        "input[{input_idx}].records[{record_idx}] observation cap {} does not match metadata cap {:?}",
                        observation.configured_max_draft_tokens,
                        input.meta.mtp_draft_tokens
                    );
                }
                if observation.mixed_context_buckets {
                    ignored.mixed_context_windows = ignored
                        .mixed_context_windows
                        .saturating_add(observation.windows);
                    continue;
                }
                if observation.total_us == 0 {
                    ignored.zero_time_windows = ignored
                        .zero_time_windows
                        .saturating_add(observation.windows);
                    continue;
                }
                let key = CandidateKey {
                    batch_width: observation.batch_width,
                    context_bucket: observation.context_bucket,
                    cap: observation.configured_max_draft_tokens,
                };
                record_grouped
                    .entry(key)
                    .and_modify(|current| current.add_observation(observation))
                    .or_insert_with(|| CandidateAccumulator::new(observation.clone()));
            }
            for (key, record_accumulator) in record_grouped {
                grouped
                    .entry(key)
                    .and_modify(|current| {
                        current.merge_record(&record_accumulator);
                        current.records = current.records.saturating_add(1);
                    })
                    .or_insert(CandidateAccumulator {
                        observation: record_accumulator.observation,
                        records: 1,
                        full_cap_windows: record_accumulator.full_cap_windows,
                        adaptive_lowered_windows: record_accumulator.adaptive_lowered_windows,
                        mixed_depth_windows: record_accumulator.mixed_depth_windows,
                    });
            }
        }
    }

    let mut regimes: BTreeMap<(usize, MtpDraftCapContextBucket), Vec<MtpDraftCapCandidate>> =
        BTreeMap::new();
    for (key, accumulator) in grouped {
        regimes
            .entry((key.batch_width, key.context_bucket))
            .or_default()
            .push(candidate_from_observation(
                key.cap,
                accumulator.records,
                accumulator.full_cap_windows,
                accumulator.adaptive_lowered_windows,
                accumulator.mixed_depth_windows,
                accumulator.observation,
            ));
    }

    let recommendations = regimes
        .into_iter()
        .map(|((batch_width, context_bucket), mut candidates)| {
            candidates.sort_by_key(|candidate| candidate.cap);
            let window_eligible = candidates
                .iter()
                .filter(|candidate| candidate.windows >= config.min_windows)
                .collect::<Vec<_>>();
            let eligible = window_eligible
                .iter()
                .copied()
                .filter(|candidate| candidate.records >= config.min_records)
                .collect::<Vec<_>>();
            let best_observed_cap = eligible
                .iter()
                .copied()
                .max_by(|left, right| {
                    left.committed_tokens_per_second
                        .total_cmp(&right.committed_tokens_per_second)
                        .then_with(|| right.cap.cmp(&left.cap))
                })
                .map(|candidate| candidate.cap);
            let (status, recommended_cap) = if candidates.len() < 2 {
                (
                    MtpDraftCapRecommendationStatus::InsufficientCapCoverage,
                    None,
                )
            } else if window_eligible.len() < 2 {
                (MtpDraftCapRecommendationStatus::InsufficientWindows, None)
            } else if eligible.len() < 2 {
                (MtpDraftCapRecommendationStatus::InsufficientRecords, None)
            } else {
                let best = eligible
                    .iter()
                    .copied()
                    .max_by(|left, right| {
                        left.committed_tokens_per_second
                            .total_cmp(&right.committed_tokens_per_second)
                            .then_with(|| right.cap.cmp(&left.cap))
                    })
                    .expect("eligible candidates checked non-empty");
                let multiplier = 1.0 + config.min_improvement_percent / 100.0;
                let recommended = eligible
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        candidate.committed_tokens_per_second * multiplier
                            >= best.committed_tokens_per_second
                    })
                    .min_by_key(|candidate| candidate.cap)
                    .expect("best candidate always satisfies improvement threshold");
                (
                    MtpDraftCapRecommendationStatus::Recommended,
                    Some(recommended.cap),
                )
            };
            MtpDraftCapRegimeRecommendation {
                batch_width,
                context_bucket,
                status,
                best_observed_cap,
                recommended_cap,
                candidates,
            }
        })
        .collect();

    Ok(MtpDraftCapCalibrationReport {
        schema_version: MTP_DRAFT_CAP_CALIBRATION_SCHEMA_VERSION,
        runtime: MtpDraftCapRuntimeContext {
            mode: first.meta.mode.clone(),
            model_dir: first.meta.model_dir.clone(),
            drafter_dir,
            device_name,
            ironmlx_version: first.meta.ironmlx_version.clone(),
            prompt_file: first.meta.prompt_file.clone(),
            prompt_tokens: first.meta.prompt_tokens,
            scheduler_prompt_files: first.meta.scheduler_prompt_files.clone(),
            scheduler_prompt_tokens: first.meta.scheduler_prompt_tokens.clone(),
            scheduler_batch_width: first.meta.scheduler_batch_width,
            max_tokens: first.meta.max_tokens,
            prefill_chunk_size: first.meta.prefill_chunk_size,
            kv_quant: first.meta.kv_quant.clone(),
            mtp_trace_windows: first.meta.mtp_trace_windows,
            b_max: first.meta.b_max,
            effective_cap_max: first.meta.effective_cap_max,
            warmup_runs: first.meta.warmup_runs,
            measured_runs: first.meta.measured_runs,
        },
        input_files: inputs.len(),
        valid_records,
        min_windows: config.min_windows,
        min_records: config.min_records,
        min_improvement_percent: config.min_improvement_percent,
        ignored,
        recommendations,
    })
}

fn validate_observation(
    observation: &MtpDraftCapObservation,
    input_idx: usize,
    record_idx: usize,
) -> Result<()> {
    let label = format!("input[{input_idx}].records[{record_idx}] observation");
    if observation.batch_width == 0 || observation.windows == 0 {
        bail!("{label} must have non-zero batch_width and windows");
    }
    if !observation.windows.is_multiple_of(observation.batch_width) {
        bail!(
            "{label} windows {} must be divisible by batch_width {}",
            observation.windows,
            observation.batch_width
        );
    }
    if observation.min_draft_tokens == 0
        || observation.min_draft_tokens > observation.max_draft_tokens
        || observation.max_draft_tokens > observation.configured_max_draft_tokens
    {
        bail!("{label} has an invalid draft-token range");
    }
    let min_drafted = observation
        .windows
        .saturating_mul(observation.min_draft_tokens);
    let max_drafted = observation
        .windows
        .saturating_mul(observation.max_draft_tokens);
    if !(min_drafted..=max_drafted).contains(&observation.drafted_tokens) {
        bail!(
            "{label} drafted_tokens {} is outside [{min_drafted}, {max_drafted}]",
            observation.drafted_tokens
        );
    }
    if observation.accepted_draft_tokens > observation.drafted_tokens {
        bail!("{label} accepts more tokens than it drafted");
    }
    let max_committed_tokens = observation
        .accepted_draft_tokens
        .saturating_add(observation.windows);
    if !(observation.windows..=max_committed_tokens).contains(&observation.committed_tokens) {
        bail!(
            "{label} committed_tokens {} is outside [{}, {max_committed_tokens}]",
            observation.committed_tokens,
            observation.windows
        );
    }
    if observation.rollback_count > observation.windows {
        bail!("{label} rollback_count exceeds windows");
    }
    Ok(())
}

fn validate_input(input: &MtpDraftCapBenchInput, label: &str) -> Result<()> {
    let meta = &input.meta;
    if meta.backend != "ironmlx-core" {
        bail!("{label} backend must be ironmlx-core, got {}", meta.backend);
    }
    if !matches!(meta.mode.as_str(), "mtp-text" | "scheduler-text") {
        bail!(
            "{label} mode must be mtp-text or scheduler-text, got {}",
            meta.mode
        );
    }
    if meta.speculative_source.as_deref() != Some("gemma4-drafter") {
        bail!(
            "{label} speculative_source must be gemma4-drafter, got {:?}",
            meta.speculative_source
        );
    }
    if meta.mtp_model_dir.is_none() {
        bail!("{label} is missing mtp_model_dir");
    }
    if meta.mtp_draft_tokens.is_none() {
        bail!("{label} is missing mtp_draft_tokens");
    }
    if meta.scheduler_batch_width == 0 {
        bail!("{label} scheduler_batch_width must be greater than zero");
    }
    if meta.scheduler_batch_width > meta.b_max {
        bail!(
            "{label} scheduler_batch_width {} exceeds b_max {}",
            meta.scheduler_batch_width,
            meta.b_max
        );
    }
    if meta.mode == "mtp-text" && meta.scheduler_batch_width != 1 {
        bail!(
            "{label} mtp-text requires scheduler_batch_width 1, got {}",
            meta.scheduler_batch_width
        );
    }
    if meta.scheduler_prompt_files.len() != meta.scheduler_batch_width {
        bail!(
            "{label} scheduler_prompt_files count {} does not match scheduler_batch_width {}",
            meta.scheduler_prompt_files.len(),
            meta.scheduler_batch_width
        );
    }
    if meta.scheduler_prompt_tokens.len() != meta.scheduler_batch_width {
        bail!(
            "{label} scheduler_prompt_tokens count {} does not match scheduler_batch_width {}",
            meta.scheduler_prompt_tokens.len(),
            meta.scheduler_batch_width
        );
    }
    if meta.scheduler_prompt_files.first() != Some(&meta.prompt_file)
        || meta.scheduler_prompt_tokens.first() != Some(&meta.prompt_tokens)
    {
        bail!("{label} primary prompt metadata does not match scheduler prompt metadata");
    }
    if meta.device_name.as_deref().is_none_or(str::is_empty) {
        bail!("{label} is missing device_name");
    }
    if meta.ironmlx_version.is_empty() {
        bail!("{label} is missing ironmlx_version");
    }
    if meta.paged_prefix_cache_dir.is_some() {
        bail!("{label} must disable paged prefix cache for draft-cap calibration");
    }
    if meta.active_kv_offload {
        bail!("{label} must disable active KV offload for draft-cap calibration");
    }
    if input.records.len() != meta.measured_runs {
        bail!(
            "{label} record count {} does not match measured_runs {}",
            input.records.len(),
            meta.measured_runs
        );
    }
    Ok(())
}

fn validate_record_output(
    record: &MtpDraftCapBenchRecord,
    meta: &MtpDraftCapBenchMeta,
    input_idx: usize,
    record_idx: usize,
    expected_outputs: &mut Option<Vec<(Vec<u32>, Option<String>)>>,
) -> Result<()> {
    let label = format!("input[{input_idx}].records[{record_idx}]");
    if record.generated_tokens == 0 {
        bail!("{label} generated zero tokens");
    }
    if record.generated_tokens != record.generated_token_ids.len() {
        bail!(
            "{label} generated_tokens {} does not match generated_token_ids length {}",
            record.generated_tokens,
            record.generated_token_ids.len()
        );
    }
    if record.generated_tokens != meta.max_tokens
        || record.finish_reason.as_deref() != Some("length")
    {
        bail!(
            "{label} valid output must finish by length with exactly {} tokens",
            meta.max_tokens
        );
    }
    let outputs = match meta.mode.as_str() {
        "mtp-text" => {
            if !record.scheduler_requests.is_empty() {
                bail!("{label} mtp-text record must not contain scheduler_requests");
            }
            vec![(
                record.generated_token_ids.clone(),
                record.finish_reason.clone(),
            )]
        }
        "scheduler-text" => {
            if record.scheduler_requests.len() != meta.scheduler_batch_width {
                bail!(
                    "{label} scheduler request count {} does not match scheduler_batch_width {}",
                    record.scheduler_requests.len(),
                    meta.scheduler_batch_width
                );
            }
            let mut outputs = Vec::with_capacity(record.scheduler_requests.len());
            for (expected_index, request) in record.scheduler_requests.iter().enumerate() {
                let request_label = format!("{label}.scheduler_requests[{expected_index}]");
                if request.request_index != expected_index {
                    bail!(
                        "{request_label} request_index {} does not match position {expected_index}",
                        request.request_index
                    );
                }
                if request.prompt_file != meta.scheduler_prompt_files[expected_index] {
                    bail!(
                        "{request_label} prompt_file mismatch: expected {:?}, got {:?}",
                        meta.scheduler_prompt_files[expected_index],
                        request.prompt_file
                    );
                }
                if !request.valid {
                    bail!("{request_label} is marked invalid inside a valid benchmark record");
                }
                if request.generated_tokens == 0 {
                    bail!("{request_label} generated zero tokens");
                }
                if request.generated_tokens != request.generated_token_ids.len() {
                    bail!(
                        "{request_label} generated_tokens {} does not match generated_token_ids length {}",
                        request.generated_tokens,
                        request.generated_token_ids.len()
                    );
                }
                if request.generated_tokens != meta.max_tokens
                    || request.finish_reason.as_deref() != Some("length")
                {
                    bail!(
                        "{request_label} valid output must finish by length with exactly {} tokens",
                        meta.max_tokens
                    );
                }
                outputs.push((
                    request.generated_token_ids.clone(),
                    request.finish_reason.clone(),
                ));
            }
            if outputs[0].0 != record.generated_token_ids
                || outputs[0].1.as_deref() != record.finish_reason.as_deref()
            {
                bail!("{label} representative output does not match scheduler request 0");
            }
            outputs
        }
        mode => bail!("{label} has unsupported mode {mode:?} after input validation"),
    };

    if let Some(expected) = expected_outputs {
        if outputs != *expected {
            bail!("{label} outputs differ from the first valid greedy benchmark record");
        }
    } else {
        *expected_outputs = Some(outputs);
    }
    Ok(())
}

fn ensure_same_runtime(
    expected: &MtpDraftCapBenchMeta,
    actual: &MtpDraftCapBenchMeta,
    label: &str,
) -> Result<()> {
    macro_rules! require_equal {
        ($field:ident) => {
            if actual.$field != expected.$field {
                bail!(
                    "{label} {} mismatch: expected {:?}, got {:?}",
                    stringify!($field),
                    expected.$field,
                    actual.$field
                );
            }
        };
    }
    require_equal!(mode);
    require_equal!(model_dir);
    require_equal!(mtp_model_dir);
    require_equal!(mtp_trace_windows);
    require_equal!(prompt_file);
    require_equal!(prompt_tokens);
    require_equal!(scheduler_prompt_files);
    require_equal!(scheduler_prompt_tokens);
    require_equal!(scheduler_batch_width);
    require_equal!(max_tokens);
    require_equal!(prefill_chunk_size);
    require_equal!(kv_quant);
    require_equal!(paged_prefix_cache_dir);
    require_equal!(paged_prefix_cache_block_size);
    require_equal!(paged_prefix_cache_max_pages);
    require_equal!(active_kv_offload);
    require_equal!(b_max);
    require_equal!(effective_cap_max);
    require_equal!(warmup_runs);
    require_equal!(measured_runs);
    require_equal!(device_name);
    require_equal!(ironmlx_version);
    Ok(())
}

fn add_observation(current: &mut MtpDraftCapObservation, other: &MtpDraftCapObservation) {
    current.windows = current.windows.saturating_add(other.windows);
    current.drafted_tokens = current.drafted_tokens.saturating_add(other.drafted_tokens);
    current.accepted_draft_tokens = current
        .accepted_draft_tokens
        .saturating_add(other.accepted_draft_tokens);
    current.committed_tokens = current
        .committed_tokens
        .saturating_add(other.committed_tokens);
    current.rollback_count = current.rollback_count.saturating_add(other.rollback_count);
    current.total_us = current.total_us.saturating_add(other.total_us);
    current.draft_forward_us = current
        .draft_forward_us
        .saturating_add(other.draft_forward_us);
    current.verify_forward_us = current
        .verify_forward_us
        .saturating_add(other.verify_forward_us);
    current.projection_us = current.projection_us.saturating_add(other.projection_us);
    current.sampling_us = current.sampling_us.saturating_add(other.sampling_us);
    current.main_rollback_us = current
        .main_rollback_us
        .saturating_add(other.main_rollback_us);
    current.decode_cache_commit_us = current
        .decode_cache_commit_us
        .saturating_add(other.decode_cache_commit_us);
    current.cache_restore_us = current
        .cache_restore_us
        .saturating_add(other.cache_restore_us);
}

fn candidate_from_observation(
    cap: usize,
    records: usize,
    full_cap_windows: usize,
    adaptive_lowered_windows: usize,
    mixed_depth_windows: usize,
    observation: MtpDraftCapObservation,
) -> MtpDraftCapCandidate {
    let committed_tokens = observation.committed_tokens;
    MtpDraftCapCandidate {
        cap,
        records,
        windows: observation.windows,
        full_cap_windows,
        adaptive_lowered_windows,
        mixed_depth_windows,
        drafted_tokens: observation.drafted_tokens,
        accepted_draft_tokens: observation.accepted_draft_tokens,
        committed_tokens,
        rollback_count: observation.rollback_count,
        acceptance_rate: observation.accepted_draft_tokens as f64
            / observation.drafted_tokens.max(1) as f64,
        rollback_rate: observation.rollback_count as f64 / observation.windows.max(1) as f64,
        mean_draft_tokens_per_window: observation.drafted_tokens as f64
            / observation.windows.max(1) as f64,
        mean_window_us: observation.total_us as f64 / observation.windows.max(1) as f64,
        committed_tokens_per_second: committed_tokens as f64 * 1_000_000.0
            / observation.total_us.max(1) as f64,
        total_us: observation.total_us,
        draft_forward_us: observation.draft_forward_us,
        verify_forward_us: observation.verify_forward_us,
        projection_us: observation.projection_us,
        sampling_us: observation.sampling_us,
        main_rollback_us: observation.main_rollback_us,
        decode_cache_commit_us: observation.decode_cache_commit_us,
        cache_restore_us: observation.cache_restore_us,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        cap: usize,
        windows: usize,
        committed_per_window: usize,
        us: u64,
    ) -> MtpDraftCapObservation {
        MtpDraftCapObservation {
            configured_max_draft_tokens: cap,
            min_draft_tokens: cap,
            max_draft_tokens: cap,
            batch_width: 1,
            context_bucket: MtpDraftCapContextBucket::UpTo8k,
            mixed_context_buckets: false,
            windows,
            drafted_tokens: windows * cap,
            accepted_draft_tokens: windows * committed_per_window.saturating_sub(1),
            committed_tokens: windows * committed_per_window,
            rollback_count: windows / 4,
            total_us: us,
            draft_forward_us: us / 4,
            verify_forward_us: us / 2,
            projection_us: us / 10,
            sampling_us: us / 20,
            main_rollback_us: us / 20,
            decode_cache_commit_us: 0,
            cache_restore_us: 0,
        }
    }

    fn input(cap: usize, observation: MtpDraftCapObservation) -> MtpDraftCapBenchInput {
        MtpDraftCapBenchInput {
            meta: MtpDraftCapBenchMeta {
                backend: "ironmlx-core".to_string(),
                mode: "mtp-text".to_string(),
                speculative_source: Some("gemma4-drafter".to_string()),
                model_dir: "/models/gemma4".to_string(),
                mtp_model_dir: Some("/models/gemma4-drafter".to_string()),
                mtp_draft_tokens: Some(cap),
                mtp_trace_windows: 0,
                prompt_file: "/prompts/calibration.txt".to_string(),
                prompt_tokens: 4_096,
                scheduler_prompt_files: vec!["/prompts/calibration.txt".to_string()],
                scheduler_prompt_tokens: vec![4_096],
                scheduler_batch_width: 1,
                max_tokens: 3,
                prefill_chunk_size: 2_048,
                kv_quant: "none".to_string(),
                paged_prefix_cache_dir: None,
                paged_prefix_cache_block_size: 256,
                paged_prefix_cache_max_pages: None,
                active_kv_offload: false,
                b_max: 2,
                effective_cap_max: 8_192,
                warmup_runs: 1,
                measured_runs: 3,
                device_name: Some("Apple Test GPU".to_string()),
                ironmlx_version: "0.0.1".to_string(),
            },
            records: (0..3)
                .map(|_| MtpDraftCapBenchRecord {
                    valid: true,
                    generated_tokens: 3,
                    generated_token_ids: vec![11, 12, 13],
                    finish_reason: Some("length".to_string()),
                    scheduler_requests: Vec::new(),
                    mtp_stats: Some(MtpDraftCapBenchStats {
                        draft_cap_observations: vec![observation.clone()],
                        draft_cap_observation_dropped_windows: 0,
                    }),
                })
                .collect(),
        }
    }

    fn config() -> MtpDraftCapCalibrationConfig {
        MtpDraftCapCalibrationConfig {
            min_windows: 32,
            min_records: 3,
            min_improvement_percent: 3.0,
        }
    }

    fn scheduler_input(
        cap: usize,
        mut observation: MtpDraftCapObservation,
    ) -> MtpDraftCapBenchInput {
        observation.batch_width = 2;
        let mut input = input(cap, observation);
        input.meta.mode = "scheduler-text".to_string();
        input.meta.scheduler_prompt_files = vec![
            "/prompts/short.txt".to_string(),
            "/prompts/long.txt".to_string(),
        ];
        input.meta.scheduler_prompt_tokens = vec![512, 4_096];
        input.meta.scheduler_batch_width = 2;
        input.meta.prompt_file = "/prompts/short.txt".to_string();
        input.meta.prompt_tokens = 512;
        for record in &mut input.records {
            record.scheduler_requests = vec![
                MtpDraftCapBenchRequest {
                    request_index: 0,
                    prompt_file: "/prompts/short.txt".to_string(),
                    generated_tokens: 3,
                    generated_token_ids: vec![11, 12, 13],
                    finish_reason: Some("length".to_string()),
                    valid: true,
                },
                MtpDraftCapBenchRequest {
                    request_index: 1,
                    prompt_file: "/prompts/long.txt".to_string(),
                    generated_tokens: 3,
                    generated_token_ids: vec![21, 22, 23],
                    finish_reason: Some("length".to_string()),
                    valid: true,
                },
            ];
        }
        input
    }

    #[test]
    fn calibration_recommends_highest_observed_committed_rate() {
        let cap1 = observation(1, 64, 2, 640_000);
        let cap2 = observation(2, 64, 3, 768_000);
        let report = calibrate_mtp_draft_cap(vec![input(1, cap1), input(2, cap2)], config())
            .expect("calibration");
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.recommendations.len(), 1);
        let regime = &report.recommendations[0];
        assert_eq!(regime.status, MtpDraftCapRecommendationStatus::Recommended);
        assert_eq!(regime.recommended_cap, Some(2));
    }

    #[test]
    fn calibration_rejects_cross_device_inputs() {
        let mut second = input(2, observation(2, 64, 3, 768_000));
        second.meta.device_name = Some("Different GPU".to_string());
        let error = calibrate_mtp_draft_cap(
            vec![input(1, observation(1, 64, 2, 640_000)), second],
            config(),
        )
        .expect_err("cross-device inputs must fail");
        assert!(error.to_string().contains("device_name mismatch"));
    }

    #[test]
    fn calibration_includes_adaptive_lowered_windows_in_configured_cap_policy() {
        let mut lowered = observation(1, 64, 2, 640_000);
        lowered.configured_max_draft_tokens = 2;
        let report =
            calibrate_mtp_draft_cap(vec![input(2, lowered)], config()).expect("calibration report");
        let regime = &report.recommendations[0];
        let candidate = &regime.candidates[0];
        assert_eq!(candidate.cap, 2);
        assert_eq!(candidate.adaptive_lowered_windows, 192);
        assert_eq!(candidate.full_cap_windows, 0);
        assert_eq!(candidate.mean_draft_tokens_per_window, 1.0);
        assert_eq!(
            regime.status,
            MtpDraftCapRecommendationStatus::InsufficientCapCoverage
        );
    }

    #[test]
    fn calibration_keeps_lower_cap_when_gain_is_below_threshold() {
        let cap1 = observation(1, 64, 2, 640_000);
        let cap2 = observation(2, 64, 2, 630_000);
        let report = calibrate_mtp_draft_cap(vec![input(1, cap1), input(2, cap2)], config())
            .expect("calibration");
        let regime = &report.recommendations[0];
        assert_eq!(regime.best_observed_cap, Some(2));
        assert_eq!(regime.recommended_cap, Some(1));
    }

    #[test]
    fn calibration_rejects_output_divergence_between_caps() {
        let first = input(1, observation(1, 64, 2, 640_000));
        let mut second = input(2, observation(2, 64, 3, 768_000));
        second.records[1].generated_token_ids[2] = 99;

        let error = calibrate_mtp_draft_cap(vec![first, second], config())
            .expect_err("divergent greedy output must fail calibration");
        assert!(error
            .to_string()
            .contains("outputs differ from the first valid greedy benchmark record"));
    }

    #[test]
    fn calibration_rejects_truncated_measured_records() {
        let mut truncated = input(1, observation(1, 64, 2, 640_000));
        truncated.records.pop();

        let error = calibrate_mtp_draft_cap(vec![truncated], config())
            .expect_err("record count must match benchmark metadata");
        assert!(error
            .to_string()
            .contains("record count 2 does not match measured_runs 3"));
    }

    #[test]
    fn calibration_accepts_scheduler_batch_outputs_aligned_by_request() {
        let cap1 = scheduler_input(1, observation(1, 64, 2, 640_000));
        let cap2 = scheduler_input(2, observation(2, 64, 3, 768_000));

        let report =
            calibrate_mtp_draft_cap(vec![cap1, cap2], config()).expect("scheduler calibration");

        assert_eq!(report.runtime.mode, "scheduler-text");
        assert_eq!(report.runtime.scheduler_batch_width, 2);
        assert_eq!(report.recommendations[0].batch_width, 2);
        assert_eq!(report.recommendations[0].recommended_cap, Some(2));
    }

    #[test]
    fn calibration_rejects_scheduler_output_divergence_by_request() {
        let first = scheduler_input(1, observation(1, 64, 2, 640_000));
        let mut second = scheduler_input(2, observation(2, 64, 3, 768_000));
        second.records[0].scheduler_requests[1].generated_token_ids[2] = 99;

        let error = calibrate_mtp_draft_cap(vec![first, second], config())
            .expect_err("per-request output divergence must fail");
        assert!(error
            .to_string()
            .contains("outputs differ from the first valid greedy benchmark record"));
    }

    #[test]
    fn calibration_rejects_scheduler_record_without_request_outputs() {
        let mut batch = scheduler_input(1, observation(1, 64, 2, 640_000));
        batch.records[0].scheduler_requests.clear();

        let error = calibrate_mtp_draft_cap(vec![batch], config())
            .expect_err("batched records require per-request outputs");
        assert!(error
            .to_string()
            .contains("scheduler request count 0 does not match scheduler_batch_width 2"));
    }

    #[test]
    fn calibration_rejects_scheduler_b1_without_request_output() {
        let mut batch = scheduler_input(1, observation(1, 64, 2, 640_000));
        batch.meta.scheduler_prompt_files.truncate(1);
        batch.meta.scheduler_prompt_tokens.truncate(1);
        batch.meta.scheduler_batch_width = 1;
        for record in &mut batch.records {
            record.scheduler_requests.truncate(1);
            for observation in &mut record
                .mtp_stats
                .as_mut()
                .expect("stats")
                .draft_cap_observations
            {
                observation.batch_width = 1;
            }
        }
        batch.records[0].scheduler_requests.clear();

        let error = calibrate_mtp_draft_cap(vec![batch], config())
            .expect_err("scheduler B1 requires per-request output");
        assert!(error
            .to_string()
            .contains("scheduler request count 0 does not match scheduler_batch_width 1"));
    }

    #[test]
    fn bench_input_rejects_missing_scheduler_contract_fields() {
        let make_input = || input(1, observation(1, 64, 2, 640_000));

        let mut missing_prompt_files = serde_json::to_value(make_input()).expect("serialize input");
        missing_prompt_files["meta"]
            .as_object_mut()
            .expect("meta object")
            .remove("scheduler_prompt_files");
        let error = serde_json::from_value::<MtpDraftCapBenchInput>(missing_prompt_files)
            .expect_err("scheduler_prompt_files is required");
        assert!(error.to_string().contains("scheduler_prompt_files"));

        let mut missing_requests = serde_json::to_value(make_input()).expect("serialize input");
        missing_requests["records"][0]
            .as_object_mut()
            .expect("record object")
            .remove("scheduler_requests");
        let error = serde_json::from_value::<MtpDraftCapBenchInput>(missing_requests)
            .expect_err("scheduler_requests is required");
        assert!(error.to_string().contains("scheduler_requests"));

        let mut missing_observations = serde_json::to_value(make_input()).expect("serialize input");
        missing_observations["records"][0]["mtp_stats"]
            .as_object_mut()
            .expect("mtp_stats object")
            .remove("draft_cap_observations");
        let error = serde_json::from_value::<MtpDraftCapBenchInput>(missing_observations)
            .expect_err("draft_cap_observations is required");
        assert!(error.to_string().contains("draft_cap_observations"));
    }

    #[test]
    fn calibration_keeps_scheduler_width_shrink_as_separate_regime() {
        let mut batch_observation = observation(1, 64, 2, 640_000);
        batch_observation.batch_width = 2;
        let mut single_observation = observation(1, 32, 2, 320_000);
        single_observation.context_bucket = MtpDraftCapContextBucket::UpTo2k;
        let mut batch = scheduler_input(1, batch_observation);
        for record in &mut batch.records {
            record
                .mtp_stats
                .as_mut()
                .expect("stats")
                .draft_cap_observations
                .push(single_observation.clone());
        }

        let report = calibrate_mtp_draft_cap(vec![batch], config()).expect("calibration");

        assert_eq!(report.recommendations.len(), 2);
        assert_eq!(report.recommendations[0].batch_width, 1);
        assert_eq!(report.recommendations[1].batch_width, 2);
    }

    #[test]
    fn calibration_rejects_observation_wider_than_admitted_batch() {
        let mut too_wide = observation(1, 66, 2, 660_000);
        too_wide.batch_width = 3;
        let mut batch = scheduler_input(1, too_wide.clone());
        for record in &mut batch.records {
            record
                .mtp_stats
                .as_mut()
                .expect("stats")
                .draft_cap_observations[0] = too_wide.clone();
        }

        let error = calibrate_mtp_draft_cap(vec![batch], config())
            .expect_err("observation width must fit admitted batch");
        assert!(error
            .to_string()
            .contains("observation batch_width 3 exceeds scheduler_batch_width 2"));
    }
}
