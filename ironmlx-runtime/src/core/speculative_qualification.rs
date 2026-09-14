use std::{
    collections::{HashMap, VecDeque},
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::{
    core::{sampler::Sampler, scheduler_autotune::SchedulerAutotuneRuntimeProfile},
    Result,
};

const SCHEMA_VERSION: u32 = 1;
const BASELINE_SAMPLES: usize = 8;
const PROBE_SAMPLES: usize = 8;
const MIN_GAIN_BPS: u64 = 300;
const REJECTED_INITIAL_COOLDOWN_TOKENS: u64 = 512;
const REJECTED_MAX_COOLDOWN_TOKENS: u64 = 32 * 1_024;
const REVALIDATE_TOKENS: u64 = 512;
const PROFILE_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NeuralExactSource {
    QwenMtp,
    Gemma4Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NeuralExactAction {
    Ordinary,
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct NeuralExactRegime {
    source: NeuralExactSource,
    context_bucket_tokens: usize,
    sampler: SamplerFingerprint,
}

impl NeuralExactRegime {
    pub(crate) fn new(source: NeuralExactSource, context_tokens: usize, sampler: Sampler) -> Self {
        Self {
            source,
            context_bucket_tokens: context_bucket(context_tokens),
            sampler: SamplerFingerprint::from(sampler),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct SamplerFingerprint {
    temperature_bits: u32,
    top_k: Option<i32>,
    top_p_bits: Option<u32>,
    min_p_bits: Option<u32>,
    repetition_penalty_bits: Option<u32>,
    frequency_penalty_bits: Option<u32>,
    presence_penalty_bits: Option<u32>,
}

impl From<Sampler> for SamplerFingerprint {
    fn from(sampler: Sampler) -> Self {
        Self {
            temperature_bits: sampler.temperature.to_bits(),
            top_k: sampler.top_k,
            top_p_bits: sampler.top_p.map(f32::to_bits),
            min_p_bits: sampler.min_p.map(f32::to_bits),
            repetition_penalty_bits: sampler.repetition_penalty.map(f32::to_bits),
            frequency_penalty_bits: sampler.frequency_penalty.map(f32::to_bits),
            presence_penalty_bits: sampler.presence_penalty.map(f32::to_bits),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NeuralExactQualificationRuntimeConfig {
    context_fingerprint: String,
    profile_path: PathBuf,
    source: NeuralExactSource,
}

impl NeuralExactQualificationRuntimeConfig {
    pub(crate) fn for_scheduler_profile(
        profile: &SchedulerAutotuneRuntimeProfile,
        source: NeuralExactSource,
    ) -> Result<Self> {
        let context_fingerprint = qualification_context_fingerprint(profile, source)?;
        let home = dirs::home_dir()
            .context("locating home directory for neural exact qualification profiles")?;
        Ok(Self {
            profile_path: home
                .join(".ironmlx")
                .join("neural-exact-qualifications")
                .join("profiles")
                .join(format!("{context_fingerprint}.json")),
            context_fingerprint,
            source,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(source: NeuralExactSource, fingerprint: &str, path: PathBuf) -> Self {
        Self {
            context_fingerprint: fingerprint.to_string(),
            profile_path: path,
            source,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct NeuralExactQualificationStats {
    pub ordinary_cost_samples: u64,
    pub exact_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub exact_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub profile_loads: u64,
    pub profile_write_requests: u64,
    pub profile_writes: u64,
    pub profile_write_failures: u64,
    pub profile_write_coalesces: u64,
}

#[derive(Debug)]
pub(crate) struct NeuralExactCostController {
    runtime: NeuralExactQualificationRuntimeConfig,
    regimes: HashMap<NeuralExactRegime, RegimeState>,
    writer: ProfileWriter,
    stats: NeuralExactQualificationStats,
}

#[derive(Debug)]
struct RegimeState {
    phase: Phase,
    last_evidence: Option<Evidence>,
    next_rejected_cooldown_tokens: u64,
}

#[derive(Debug)]
enum Phase {
    Baseline {
        samples: Vec<u64>,
    },
    Probe {
        baseline_cost_per_token_ns: u64,
        samples: Vec<u64>,
        counters: QualificationCounters,
    },
    Qualified {
        baseline_cost_per_token_ns: u64,
        rolling_exact_samples: VecDeque<u64>,
        rolling_counters: VecDeque<QualificationCounters>,
        tokens_until_revalidate: u64,
    },
    Rejected {
        ordinary_samples: VecDeque<u64>,
        cooldown_tokens: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(crate) struct NeuralExactSampleCounters {
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub exact_windows: u64,
    pub residual_corrections: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct QualificationCounters {
    drafted_tokens: u64,
    accepted_tokens: u64,
    exact_windows: u64,
    residual_corrections: u64,
}

impl QualificationCounters {
    fn accumulate(&mut self, delta: NeuralExactSampleCounters) {
        self.drafted_tokens = self.drafted_tokens.saturating_add(delta.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(delta.accepted_tokens);
        self.exact_windows = self.exact_windows.saturating_add(delta.exact_windows);
        self.residual_corrections = self
            .residual_corrections
            .saturating_add(delta.residual_corrections);
    }

    fn accumulate_counters(&mut self, delta: Self) {
        self.drafted_tokens = self.drafted_tokens.saturating_add(delta.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(delta.accepted_tokens);
        self.exact_windows = self.exact_windows.saturating_add(delta.exact_windows);
        self.residual_corrections = self
            .residual_corrections
            .saturating_add(delta.residual_corrections);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Decision {
    Qualified,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Evidence {
    regime: NeuralExactRegime,
    decision: Decision,
    baseline_cost_per_token_ns: u64,
    exact_cost_per_token_ns: u64,
    estimated_gain_bps: i64,
    baseline_samples: usize,
    exact_samples: usize,
    counters: QualificationCounters,
    rejected_cooldown_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Profile {
    schema_version: u32,
    context_fingerprint: String,
    source: NeuralExactSource,
    updated_at_unix_ms: u64,
    entries: Vec<Evidence>,
}

#[derive(Debug)]
struct ProfileWriter {
    mailbox: Arc<ProfileMailbox>,
    counters: Arc<ProfileWriterCounters>,
    worker: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct ProfileWriterCounters {
    writes: AtomicU64,
    failures: AtomicU64,
}

#[derive(Debug)]
struct ProfileMailbox {
    state: Mutex<ProfileMailboxState>,
    wake: Condvar,
}

#[derive(Debug, Default)]
struct ProfileMailboxState {
    pending: Option<Profile>,
    closed: bool,
}

impl ProfileWriter {
    fn new(path: PathBuf) -> Result<Self> {
        let mailbox = Arc::new(ProfileMailbox {
            state: Mutex::new(ProfileMailboxState::default()),
            wake: Condvar::new(),
        });
        let worker_mailbox = Arc::clone(&mailbox);
        let counters = Arc::new(ProfileWriterCounters::default());
        let worker_counters = Arc::clone(&counters);
        let worker = std::thread::Builder::new()
            .name("neural-exact-profile-writer".to_string())
            .spawn(move || loop {
                let profile = {
                    let mut state = worker_mailbox
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    while state.pending.is_none() && !state.closed {
                        state = worker_mailbox
                            .wake
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    match state.pending.take() {
                        Some(profile) => profile,
                        None if state.closed => break,
                        None => continue,
                    }
                };
                if let Err(error) = persist_profile(&path, &profile) {
                    worker_counters.failures.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        target: "ironmlx::speculative",
                        path = %path.display(),
                        error = %error,
                        "failed to persist neural exact qualification profile"
                    );
                } else {
                    worker_counters.writes.fetch_add(1, Ordering::Relaxed);
                }
            })
            .context("spawning neural exact qualification profile writer")?;
        Ok(Self {
            mailbox,
            counters,
            worker: Some(worker),
        })
    }

    fn queue_latest(&self, profile: Profile) -> bool {
        let mut state = self
            .mailbox
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let replaced = state.pending.replace(profile).is_some();
        self.mailbox.wake.notify_one();
        replaced
    }
}

impl Drop for ProfileWriter {
    fn drop(&mut self) {
        {
            let mut state = self
                .mailbox
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
            self.mailbox.wake.notify_one();
        }
        if self
            .worker
            .take()
            .is_some_and(|worker| worker.join().is_err())
        {
            tracing::warn!(
                target: "ironmlx::speculative",
                "neural exact qualification profile writer panicked during shutdown"
            );
        }
    }
}

impl NeuralExactCostController {
    pub(crate) fn new(runtime: NeuralExactQualificationRuntimeConfig) -> Result<Self> {
        let mut stats = NeuralExactQualificationStats::default();
        let regimes = match load_profile(&runtime) {
            Ok(Some(profile)) => {
                stats.profile_loads = 1;
                profile
                    .entries
                    .into_iter()
                    .map(|evidence| {
                        let (phase, next_rejected_cooldown_tokens) = match evidence.decision {
                            Decision::Qualified => (
                                Phase::Qualified {
                                    baseline_cost_per_token_ns: evidence.baseline_cost_per_token_ns,
                                    rolling_exact_samples: VecDeque::new(),
                                    rolling_counters: VecDeque::new(),
                                    tokens_until_revalidate: REVALIDATE_TOKENS,
                                },
                                REJECTED_INITIAL_COOLDOWN_TOKENS,
                            ),
                            Decision::Rejected => (
                                Phase::Rejected {
                                    ordinary_samples: VecDeque::new(),
                                    cooldown_tokens: evidence.rejected_cooldown_tokens,
                                },
                                next_rejected_cooldown(evidence.rejected_cooldown_tokens),
                            ),
                        };
                        (
                            evidence.regime,
                            RegimeState {
                                phase,
                                last_evidence: Some(evidence),
                                next_rejected_cooldown_tokens,
                            },
                        )
                    })
                    .collect()
            }
            Ok(None) => HashMap::new(),
            Err(error) => {
                tracing::warn!(
                    target: "ironmlx::speculative",
                    path = %runtime.profile_path.display(),
                    error = %error,
                    "ignoring invalid neural exact qualification profile"
                );
                HashMap::new()
            }
        };
        let writer = ProfileWriter::new(runtime.profile_path.clone())?;
        let mut controller = Self {
            runtime,
            regimes,
            writer,
            stats,
        };
        controller.refresh_gauges();
        Ok(controller)
    }

    pub(crate) fn next_action(&mut self, regime: NeuralExactRegime) -> NeuralExactAction {
        match self
            .regimes
            .entry(regime)
            .or_insert_with(initial_regime_state)
            .phase
        {
            Phase::Probe { .. } | Phase::Qualified { .. } => NeuralExactAction::Exact,
            Phase::Baseline { .. } | Phase::Rejected { .. } => NeuralExactAction::Ordinary,
        }
    }

    pub(crate) fn record_sample(
        &mut self,
        regime: NeuralExactRegime,
        action: NeuralExactAction,
        elapsed_ns: u64,
        committed_tokens: usize,
        exact_counters: NeuralExactSampleCounters,
    ) {
        if committed_tokens == 0 {
            return;
        }
        let cost_per_token_ns = elapsed_ns / committed_tokens as u64;
        let progress_tokens = committed_tokens as u64;
        let state = self
            .regimes
            .entry(regime)
            .or_insert_with(initial_regime_state);
        let mut persist = false;
        match &mut state.phase {
            Phase::Baseline { samples } if action == NeuralExactAction::Ordinary => {
                self.stats.ordinary_cost_samples =
                    self.stats.ordinary_cost_samples.saturating_add(1);
                self.stats.ordinary_cost_us = self
                    .stats
                    .ordinary_cost_us
                    .saturating_add(elapsed_ns / 1_000);
                samples.push(cost_per_token_ns);
                if samples.len() >= BASELINE_SAMPLES {
                    state.phase = Phase::Probe {
                        baseline_cost_per_token_ns: median(samples),
                        samples: Vec::with_capacity(PROBE_SAMPLES),
                        counters: QualificationCounters::default(),
                    };
                }
            }
            Phase::Probe {
                baseline_cost_per_token_ns,
                samples,
                counters,
            } if action == NeuralExactAction::Exact => {
                self.stats.exact_cost_samples = self.stats.exact_cost_samples.saturating_add(1);
                self.stats.exact_cost_us =
                    self.stats.exact_cost_us.saturating_add(elapsed_ns / 1_000);
                samples.push(cost_per_token_ns);
                counters.accumulate(exact_counters);
                if samples.len() >= PROBE_SAMPLES {
                    let baseline = *baseline_cost_per_token_ns;
                    let exact = median(samples);
                    let decision = decision(baseline, exact);
                    let cooldown = if decision == Decision::Rejected {
                        state.next_rejected_cooldown_tokens
                    } else {
                        0
                    };
                    state.last_evidence = Some(Evidence {
                        regime,
                        decision,
                        baseline_cost_per_token_ns: baseline,
                        exact_cost_per_token_ns: exact,
                        estimated_gain_bps: estimated_gain_bps(baseline, exact),
                        baseline_samples: BASELINE_SAMPLES,
                        exact_samples: PROBE_SAMPLES,
                        counters: *counters,
                        rejected_cooldown_tokens: cooldown,
                    });
                    state.phase = if decision == Decision::Qualified {
                        state.next_rejected_cooldown_tokens = REJECTED_INITIAL_COOLDOWN_TOKENS;
                        Phase::Qualified {
                            baseline_cost_per_token_ns: baseline,
                            rolling_exact_samples: VecDeque::new(),
                            rolling_counters: VecDeque::new(),
                            tokens_until_revalidate: REVALIDATE_TOKENS,
                        }
                    } else {
                        state.next_rejected_cooldown_tokens = next_rejected_cooldown(cooldown);
                        Phase::Rejected {
                            ordinary_samples: VecDeque::new(),
                            cooldown_tokens: cooldown,
                        }
                    };
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                    persist = true;
                }
            }
            Phase::Qualified {
                baseline_cost_per_token_ns,
                rolling_exact_samples,
                rolling_counters,
                tokens_until_revalidate,
            } if action == NeuralExactAction::Exact => {
                self.stats.exact_cost_samples = self.stats.exact_cost_samples.saturating_add(1);
                self.stats.exact_cost_us =
                    self.stats.exact_cost_us.saturating_add(elapsed_ns / 1_000);
                rolling_exact_samples.push_back(cost_per_token_ns);
                let mut counter_delta = QualificationCounters::default();
                counter_delta.accumulate(exact_counters);
                rolling_counters.push_back(counter_delta);
                while rolling_exact_samples.len() > PROBE_SAMPLES {
                    rolling_exact_samples.pop_front();
                    rolling_counters.pop_front();
                }
                *tokens_until_revalidate = tokens_until_revalidate.saturating_sub(progress_tokens);
                let drifted = rolling_exact_samples.len() == PROBE_SAMPLES
                    && decision(
                        *baseline_cost_per_token_ns,
                        median_deque(rolling_exact_samples),
                    ) == Decision::Rejected;
                if drifted {
                    let exact = median_deque(rolling_exact_samples);
                    let counters = rolling_counters.iter().copied().fold(
                        QualificationCounters::default(),
                        |mut total, delta| {
                            total.accumulate_counters(delta);
                            total
                        },
                    );
                    let cooldown = state.next_rejected_cooldown_tokens;
                    state.last_evidence = Some(Evidence {
                        regime,
                        decision: Decision::Rejected,
                        baseline_cost_per_token_ns: *baseline_cost_per_token_ns,
                        exact_cost_per_token_ns: exact,
                        estimated_gain_bps: estimated_gain_bps(*baseline_cost_per_token_ns, exact),
                        baseline_samples: BASELINE_SAMPLES,
                        exact_samples: PROBE_SAMPLES,
                        counters,
                        rejected_cooldown_tokens: cooldown,
                    });
                    state.next_rejected_cooldown_tokens = next_rejected_cooldown(cooldown);
                    state.phase = Phase::Rejected {
                        ordinary_samples: VecDeque::new(),
                        cooldown_tokens: cooldown,
                    };
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                    persist = true;
                } else if *tokens_until_revalidate == 0 {
                    state.phase = Phase::Baseline {
                        samples: Vec::with_capacity(BASELINE_SAMPLES),
                    };
                }
            }
            Phase::Rejected {
                ordinary_samples,
                cooldown_tokens,
            } if action == NeuralExactAction::Ordinary => {
                self.stats.ordinary_cost_samples =
                    self.stats.ordinary_cost_samples.saturating_add(1);
                self.stats.ordinary_cost_us = self
                    .stats
                    .ordinary_cost_us
                    .saturating_add(elapsed_ns / 1_000);
                ordinary_samples.push_back(cost_per_token_ns);
                while ordinary_samples.len() > BASELINE_SAMPLES {
                    ordinary_samples.pop_front();
                }
                *cooldown_tokens = cooldown_tokens.saturating_sub(progress_tokens);
                if *cooldown_tokens == 0 && ordinary_samples.len() == BASELINE_SAMPLES {
                    state.phase = Phase::Probe {
                        baseline_cost_per_token_ns: median_deque(ordinary_samples),
                        samples: Vec::with_capacity(PROBE_SAMPLES),
                        counters: QualificationCounters::default(),
                    };
                }
            }
            _ => {}
        }
        self.refresh_gauges();
        if persist {
            self.queue_profile_write();
        }
    }

    pub(crate) fn stats(&self) -> NeuralExactQualificationStats {
        NeuralExactQualificationStats {
            profile_writes: self.writer.counters.writes.load(Ordering::Relaxed),
            profile_write_failures: self.writer.counters.failures.load(Ordering::Relaxed),
            ..self.stats
        }
    }

    fn refresh_gauges(&mut self) {
        self.stats.qualified_regimes_current = self
            .regimes
            .values()
            .filter(|state| matches!(state.phase, Phase::Qualified { .. }))
            .count() as u64;
        self.stats.rejected_regimes_current = self
            .regimes
            .values()
            .filter(|state| matches!(state.phase, Phase::Rejected { .. }))
            .count() as u64;
    }

    fn queue_profile_write(&mut self) {
        let profile = Profile {
            schema_version: SCHEMA_VERSION,
            context_fingerprint: self.runtime.context_fingerprint.clone(),
            source: self.runtime.source,
            updated_at_unix_ms: unix_time_ms(),
            entries: self
                .regimes
                .values()
                .filter_map(|state| state.last_evidence.clone())
                .collect(),
        };
        if self.writer.queue_latest(profile) {
            self.stats.profile_write_coalesces =
                self.stats.profile_write_coalesces.saturating_add(1);
        }
        self.stats.profile_write_requests = self.stats.profile_write_requests.saturating_add(1);
    }
}

fn initial_regime_state() -> RegimeState {
    RegimeState {
        phase: Phase::Baseline {
            samples: Vec::with_capacity(BASELINE_SAMPLES),
        },
        last_evidence: None,
        next_rejected_cooldown_tokens: REJECTED_INITIAL_COOLDOWN_TOKENS,
    }
}

fn decision(baseline: u64, exact: u64) -> Decision {
    if exact.saturating_mul(10_000) <= baseline.saturating_mul(10_000 - MIN_GAIN_BPS) {
        Decision::Qualified
    } else {
        Decision::Rejected
    }
}

fn estimated_gain_bps(baseline: u64, exact: u64) -> i64 {
    if baseline == 0 {
        return 0;
    }
    let delta = i128::from(baseline) - i128::from(exact);
    (delta.saturating_mul(10_000) / i128::from(baseline))
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn next_rejected_cooldown(current: u64) -> u64 {
    current.saturating_mul(4).clamp(
        REJECTED_INITIAL_COOLDOWN_TOKENS,
        REJECTED_MAX_COOLDOWN_TOKENS,
    )
}

fn context_bucket(tokens: usize) -> usize {
    tokens.max(1).next_power_of_two()
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_deque(values: &VecDeque<u64>) -> u64 {
    let mut values = values.iter().copied().collect::<Vec<_>>();
    median(&mut values)
}

fn qualification_context_fingerprint(
    profile: &SchedulerAutotuneRuntimeProfile,
    source: NeuralExactSource,
) -> Result<String> {
    let encoded = serde_json::to_vec(&(
        SCHEMA_VERSION,
        env!("CARGO_PKG_VERSION"),
        source,
        &profile.hardware_label,
        profile.runtime_context.fingerprint(),
        profile.config,
        &profile.rules,
    ))?;
    Ok(fnv1a_hex(&encoded))
}

fn load_profile(runtime: &NeuralExactQualificationRuntimeConfig) -> Result<Option<Profile>> {
    let bytes = match std::fs::read(&runtime.profile_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let profile: Profile = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        profile.schema_version == SCHEMA_VERSION,
        "profile schema {} != {}",
        profile.schema_version,
        SCHEMA_VERSION
    );
    anyhow::ensure!(
        profile.context_fingerprint == runtime.context_fingerprint,
        "profile context fingerprint mismatch"
    );
    anyhow::ensure!(profile.source == runtime.source, "profile source mismatch");
    anyhow::ensure!(
        unix_time_ms().saturating_sub(profile.updated_at_unix_ms) <= PROFILE_TTL_MS,
        "profile expired"
    );
    Ok(Some(profile))
}

fn persist_profile(path: &Path, profile: &Profile) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("qualification profile path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let encoded = serde_json::to_vec_pretty(profile)?;
    let temp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let mut temp = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp_path)
        .with_context(|| format!("creating {}", temp_path.display()))?;
    temp.write_all(&encoded)?;
    temp.write_all(b"\n")?;
    temp.sync_all()?;
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("renaming {} to {}", temp_path.display(), path.display()))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(name: &str) -> NeuralExactQualificationRuntimeConfig {
        let path = std::env::temp_dir().join(format!(
            "ironmlx-neural-exact-qualification-{name}-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        NeuralExactQualificationRuntimeConfig::for_test(NeuralExactSource::QwenMtp, name, path)
    }

    fn regime() -> NeuralExactRegime {
        NeuralExactRegime::new(
            NeuralExactSource::QwenMtp,
            1_024,
            Sampler::greedy().with_temperature(0.7).with_top_p(0.9),
        )
    }

    #[test]
    fn rejects_a_slower_exact_regime_and_returns_to_ordinary() {
        let mut controller = NeuralExactCostController::new(runtime("reject")).unwrap();
        let regime = regime();
        for _ in 0..BASELINE_SAMPLES {
            assert_eq!(controller.next_action(regime), NeuralExactAction::Ordinary);
            controller.record_sample(
                regime,
                NeuralExactAction::Ordinary,
                100_000,
                10,
                NeuralExactSampleCounters::default(),
            );
        }
        for _ in 0..PROBE_SAMPLES {
            assert_eq!(controller.next_action(regime), NeuralExactAction::Exact);
            controller.record_sample(
                regime,
                NeuralExactAction::Exact,
                170_000,
                10,
                NeuralExactSampleCounters {
                    drafted_tokens: 4,
                    accepted_tokens: 2,
                    exact_windows: 2,
                    residual_corrections: 1,
                },
            );
        }
        assert_eq!(controller.next_action(regime), NeuralExactAction::Ordinary);
        let stats = controller.stats();
        assert_eq!(stats.rejected_regimes_current, 1);
        assert_eq!(stats.qualification_changes, 1);
    }

    #[test]
    fn qualifies_a_faster_exact_regime_and_rejects_later_drift() {
        let mut controller = NeuralExactCostController::new(runtime("qualify")).unwrap();
        let regime = regime();
        for _ in 0..BASELINE_SAMPLES {
            controller.record_sample(
                regime,
                NeuralExactAction::Ordinary,
                100_000,
                10,
                NeuralExactSampleCounters::default(),
            );
        }
        for _ in 0..PROBE_SAMPLES {
            controller.record_sample(
                regime,
                NeuralExactAction::Exact,
                80_000,
                10,
                NeuralExactSampleCounters::default(),
            );
        }
        assert_eq!(controller.next_action(regime), NeuralExactAction::Exact);
        assert_eq!(controller.stats().qualified_regimes_current, 1);
        for _ in 0..PROBE_SAMPLES {
            controller.record_sample(
                regime,
                NeuralExactAction::Exact,
                140_000,
                10,
                NeuralExactSampleCounters::default(),
            );
        }
        assert_eq!(controller.next_action(regime), NeuralExactAction::Ordinary);
        assert_eq!(controller.stats().rejected_regimes_current, 1);
    }

    #[test]
    fn persists_and_reloads_a_rejected_regime() {
        let runtime = runtime("reload");
        let profile_path = runtime.profile_path.clone();
        let regime = regime();
        {
            let mut controller = NeuralExactCostController::new(runtime.clone()).unwrap();
            for _ in 0..BASELINE_SAMPLES {
                controller.record_sample(
                    regime,
                    NeuralExactAction::Ordinary,
                    100_000,
                    10,
                    NeuralExactSampleCounters::default(),
                );
            }
            for _ in 0..PROBE_SAMPLES {
                controller.record_sample(
                    regime,
                    NeuralExactAction::Exact,
                    170_000,
                    10,
                    NeuralExactSampleCounters::default(),
                );
            }
        }

        let mut reloaded = NeuralExactCostController::new(runtime).unwrap();
        assert_eq!(reloaded.stats().profile_loads, 1);
        assert_eq!(reloaded.stats().rejected_regimes_current, 1);
        assert_eq!(reloaded.next_action(regime), NeuralExactAction::Ordinary);
        std::fs::remove_file(profile_path).ok();
    }

    #[test]
    fn sampler_seed_does_not_split_a_cost_regime() {
        let sampler = Sampler::greedy().with_temperature(0.7).with_top_p(0.9);
        assert_eq!(
            NeuralExactRegime::new(NeuralExactSource::QwenMtp, 1_024, sampler.with_seed(1)),
            NeuralExactRegime::new(NeuralExactSource::QwenMtp, 1_024, sampler.with_seed(2))
        );
    }
}
