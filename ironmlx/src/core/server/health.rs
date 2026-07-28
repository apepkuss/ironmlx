//! `/healthz` JSON endpoint (B1-p2.5 G3). Snapshot of scheduler /
//! memory / model state for monitoring + load balancer health probes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use crate::core::cache::{ActiveKvOffloadHealth, ActiveKvOffloadSharedStats};
use crate::core::memory_budget::system_total_ram_bytes;
use crate::core::prompt_lookup::{PromptLookupConfig, PromptLookupStats};

#[derive(Debug, Serialize)]
pub enum HealthStatus {
    #[serde(rename = "healthy")]
    Healthy,
    #[serde(rename = "degraded")]
    Degraded,
    #[serde(rename = "down")]
    Down,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub max_position_embeddings: i32,
}

#[derive(Debug, Serialize)]
pub struct SchedulerInfo {
    pub b_max: usize,
    pub b_active: usize,
    pub b_queued: usize,
    pub queue_max: usize,
    pub admit_count: u64,
    pub batch_count: u64,
    pub admission_queue_full_count: u64,
    pub memory_budget_exceeded_count: u64,
}

#[derive(Debug, Serialize)]
pub struct MemoryInfo {
    pub total_ram_bytes: usize,
    pub free_ram_bytes: usize,
    pub kv_cache_active_bytes: usize,
    pub kv_cache_soft_limit_bytes: usize,
    pub kv_cache_logical_cap_tokens: usize,
    pub kv_cache_resident_cap_tokens: usize,
    pub kv_cache_budget_policy: String,
    pub mlx_total_bytes: Option<usize>,
    pub mlx_max_recommended_bytes: Option<usize>,
    pub mlx_active_bytes: usize,
    pub mlx_cache_bytes: usize,
    pub mlx_peak_bytes: usize,
    pub mlx_memory_limit_bytes: usize,
    pub process_governor: crate::core::process_memory::MemoryGovernorSnapshot,
    pub prefix_store: crate::core::cache::AsyncPrefixStoreStats,
    pub immutable_prefix_blocks: crate::core::server::scheduler_actor::ImmutablePrefixBlockHealth,
}

#[derive(Debug, Serialize)]
pub struct MtpHealthInfo {
    pub enabled: bool,
    pub requested_draft_tokens: Option<usize>,
    /// Runtime cap after applying cache and scheduler safety constraints.
    pub draft_tokens: Option<usize>,
    pub prefill_count: u64,
    pub step_count: u64,
    pub fallback_prefill_count: u64,
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub windows: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub draft_host_sync_count: u64,
    pub draft_host_sync_us: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub main_rollback_us: u64,
    pub cache_commit_us: u64,
    pub prefill_cache_commit_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
    pub sampled_exact_qualification: NeuralExactQualificationHealth,
}

#[derive(Debug, Default, Serialize)]
pub struct NeuralExactQualificationHealth {
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

impl From<crate::core::speculative_qualification::NeuralExactQualificationStats>
    for NeuralExactQualificationHealth
{
    fn from(stats: crate::core::speculative_qualification::NeuralExactQualificationStats) -> Self {
        Self {
            ordinary_cost_samples: stats.ordinary_cost_samples,
            exact_cost_samples: stats.exact_cost_samples,
            ordinary_cost_us: stats.ordinary_cost_us,
            exact_cost_us: stats.exact_cost_us,
            qualified_regimes_current: stats.qualified_regimes_current,
            rejected_regimes_current: stats.rejected_regimes_current,
            qualification_changes: stats.qualification_changes,
            profile_loads: stats.profile_loads,
            profile_write_requests: stats.profile_write_requests,
            profile_writes: stats.profile_writes,
            profile_write_failures: stats.profile_write_failures,
            profile_write_coalesces: stats.profile_write_coalesces,
        }
    }
}

#[derive(Clone)]
pub struct MtpHealthConfig {
    enabled: bool,
    requested_draft_tokens: Option<usize>,
    draft_tokens: Option<usize>,
    prefill_count: Arc<AtomicU64>,
    step_count: Arc<AtomicU64>,
    fallback_prefill_count: Arc<AtomicU64>,
    drafted_tokens: Arc<AtomicU64>,
    accepted_draft_tokens: Arc<AtomicU64>,
    windows: Arc<AtomicU64>,
    exact_sampling_windows: Arc<AtomicU64>,
    exact_acceptance_draws: Arc<AtomicU64>,
    exact_residual_corrections: Arc<AtomicU64>,
    exact_bonus_samples: Arc<AtomicU64>,
    draft_forward_us: Arc<AtomicU64>,
    verify_forward_us: Arc<AtomicU64>,
    projection_us: Arc<AtomicU64>,
    sampling_us: Arc<AtomicU64>,
    draft_host_sync_count: Arc<AtomicU64>,
    draft_host_sync_us: Arc<AtomicU64>,
    verify_accept_host_sync_count: Arc<AtomicU64>,
    verify_accept_host_sync_us: Arc<AtomicU64>,
    main_rollback_us: Arc<AtomicU64>,
    cache_commit_us: Arc<AtomicU64>,
    prefill_cache_commit_us: Arc<AtomicU64>,
    decode_cache_commit_us: Arc<AtomicU64>,
    cache_restore_us: Arc<AtomicU64>,
    neural_exact_qualification_stats: Arc<
        std::sync::Mutex<crate::core::speculative_qualification::NeuralExactQualificationStats>,
    >,
}

impl MtpHealthConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            requested_draft_tokens: None,
            draft_tokens: None,
            prefill_count: Arc::new(AtomicU64::new(0)),
            step_count: Arc::new(AtomicU64::new(0)),
            fallback_prefill_count: Arc::new(AtomicU64::new(0)),
            drafted_tokens: Arc::new(AtomicU64::new(0)),
            accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
            windows: Arc::new(AtomicU64::new(0)),
            exact_sampling_windows: Arc::new(AtomicU64::new(0)),
            exact_acceptance_draws: Arc::new(AtomicU64::new(0)),
            exact_residual_corrections: Arc::new(AtomicU64::new(0)),
            exact_bonus_samples: Arc::new(AtomicU64::new(0)),
            draft_forward_us: Arc::new(AtomicU64::new(0)),
            verify_forward_us: Arc::new(AtomicU64::new(0)),
            projection_us: Arc::new(AtomicU64::new(0)),
            sampling_us: Arc::new(AtomicU64::new(0)),
            draft_host_sync_count: Arc::new(AtomicU64::new(0)),
            draft_host_sync_us: Arc::new(AtomicU64::new(0)),
            verify_accept_host_sync_count: Arc::new(AtomicU64::new(0)),
            verify_accept_host_sync_us: Arc::new(AtomicU64::new(0)),
            main_rollback_us: Arc::new(AtomicU64::new(0)),
            cache_commit_us: Arc::new(AtomicU64::new(0)),
            prefill_cache_commit_us: Arc::new(AtomicU64::new(0)),
            decode_cache_commit_us: Arc::new(AtomicU64::new(0)),
            cache_restore_us: Arc::new(AtomicU64::new(0)),
            neural_exact_qualification_stats: Arc::new(std::sync::Mutex::new(
                crate::core::speculative_qualification::NeuralExactQualificationStats::default(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enabled(
        requested_draft_tokens: usize,
        draft_tokens: usize,
        prefill_count: Arc<AtomicU64>,
        step_count: Arc<AtomicU64>,
        fallback_prefill_count: Arc<AtomicU64>,
        drafted_tokens: Arc<AtomicU64>,
        accepted_draft_tokens: Arc<AtomicU64>,
        windows: Arc<AtomicU64>,
        exact_sampling_windows: Arc<AtomicU64>,
        exact_acceptance_draws: Arc<AtomicU64>,
        exact_residual_corrections: Arc<AtomicU64>,
        exact_bonus_samples: Arc<AtomicU64>,
        draft_forward_us: Arc<AtomicU64>,
        verify_forward_us: Arc<AtomicU64>,
        projection_us: Arc<AtomicU64>,
        sampling_us: Arc<AtomicU64>,
        draft_host_sync_count: Arc<AtomicU64>,
        draft_host_sync_us: Arc<AtomicU64>,
        verify_accept_host_sync_count: Arc<AtomicU64>,
        verify_accept_host_sync_us: Arc<AtomicU64>,
        main_rollback_us: Arc<AtomicU64>,
        cache_commit_us: Arc<AtomicU64>,
        prefill_cache_commit_us: Arc<AtomicU64>,
        decode_cache_commit_us: Arc<AtomicU64>,
        cache_restore_us: Arc<AtomicU64>,
        neural_exact_qualification_stats: Arc<
            std::sync::Mutex<crate::core::speculative_qualification::NeuralExactQualificationStats>,
        >,
    ) -> Self {
        Self {
            enabled: true,
            requested_draft_tokens: Some(requested_draft_tokens),
            draft_tokens: Some(draft_tokens),
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
            windows,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            draft_forward_us,
            verify_forward_us,
            projection_us,
            sampling_us,
            draft_host_sync_count,
            draft_host_sync_us,
            verify_accept_host_sync_count,
            verify_accept_host_sync_us,
            main_rollback_us,
            cache_commit_us,
            prefill_cache_commit_us,
            decode_cache_commit_us,
            cache_restore_us,
            neural_exact_qualification_stats,
        }
    }

    fn snapshot(&self) -> MtpHealthInfo {
        MtpHealthInfo {
            enabled: self.enabled,
            requested_draft_tokens: self.requested_draft_tokens,
            draft_tokens: self.draft_tokens,
            prefill_count: self.prefill_count.load(Ordering::Relaxed),
            step_count: self.step_count.load(Ordering::Relaxed),
            fallback_prefill_count: self.fallback_prefill_count.load(Ordering::Relaxed),
            drafted_tokens: self.drafted_tokens.load(Ordering::Relaxed),
            accepted_draft_tokens: self.accepted_draft_tokens.load(Ordering::Relaxed),
            windows: self.windows.load(Ordering::Relaxed),
            exact_sampling_windows: self.exact_sampling_windows.load(Ordering::Relaxed),
            exact_acceptance_draws: self.exact_acceptance_draws.load(Ordering::Relaxed),
            exact_residual_corrections: self.exact_residual_corrections.load(Ordering::Relaxed),
            exact_bonus_samples: self.exact_bonus_samples.load(Ordering::Relaxed),
            draft_forward_us: self.draft_forward_us.load(Ordering::Relaxed),
            verify_forward_us: self.verify_forward_us.load(Ordering::Relaxed),
            projection_us: self.projection_us.load(Ordering::Relaxed),
            sampling_us: self.sampling_us.load(Ordering::Relaxed),
            draft_host_sync_count: self.draft_host_sync_count.load(Ordering::Relaxed),
            draft_host_sync_us: self.draft_host_sync_us.load(Ordering::Relaxed),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .load(Ordering::Relaxed),
            verify_accept_host_sync_us: self.verify_accept_host_sync_us.load(Ordering::Relaxed),
            main_rollback_us: self.main_rollback_us.load(Ordering::Relaxed),
            cache_commit_us: self.cache_commit_us.load(Ordering::Relaxed),
            prefill_cache_commit_us: self.prefill_cache_commit_us.load(Ordering::Relaxed),
            decode_cache_commit_us: self.decode_cache_commit_us.load(Ordering::Relaxed),
            cache_restore_us: self.cache_restore_us.load(Ordering::Relaxed),
            sampled_exact_qualification: (*self
                .neural_exact_qualification_stats
                .lock()
                .expect("neural exact qualification stats mutex poisoned"))
            .into(),
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct PromptLookupHealthInfo {
    pub enabled: bool,
    pub min_ngram: Option<usize>,
    pub max_ngram: Option<usize>,
    pub max_draft_tokens: Option<usize>,
    pub history_window_tokens: Option<usize>,
    pub max_index_entries: Option<usize>,
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub zero_accept_windows: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub propose_us: u64,
    pub index_build_us: u64,
    pub index_update_us: u64,
    pub index_entries_current: u64,
    pub index_entries_peak: u64,
    pub index_evictions: u64,
    pub verify_round_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub exact_batched_verify_windows: u64,
    pub sequential_verify_windows: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub rollback_count: u64,
    pub rollback_us: u64,
    pub mtp_shadow_commit_windows: u64,
    pub mtp_shadow_commit_tokens: u64,
    pub mtp_shadow_commit_us: u64,
    pub miss_fast_path_steps: u64,
    pub ordinary_cost_samples: u64,
    pub lookup_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub lookup_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub qualification_profile_loads: u64,
    pub qualification_profile_writes: u64,
    pub qualification_profile_write_drops: u64,
    pub hybrid_neural_windows: u64,
    pub hybrid_lookup_windows: u64,
    pub hybrid_source_switches: u64,
    pub hybrid_lookup_miss_fallbacks: u64,
    pub hybrid_neural_rebases: u64,
    pub hybrid_neural_rebase_us: u64,
}

impl PromptLookupHealthInfo {
    pub fn aggregate(snapshots: impl IntoIterator<Item = Self>) -> Self {
        let mut aggregate = Self::default();
        let mut config: Option<(usize, usize, usize, usize, usize)> = None;
        let mut config_mismatch = false;
        for snapshot in snapshots {
            aggregate.enabled |= snapshot.enabled;
            if snapshot.enabled {
                let current = snapshot
                    .min_ngram
                    .zip(snapshot.max_ngram)
                    .zip(snapshot.max_draft_tokens)
                    .zip(snapshot.history_window_tokens)
                    .zip(snapshot.max_index_entries)
                    .map(
                        |((((min_ngram, max_ngram), max_draft_tokens), history), entries)| {
                            (min_ngram, max_ngram, max_draft_tokens, history, entries)
                        },
                    );
                match (config, current) {
                    (None, Some(current)) => config = Some(current),
                    (Some(expected), Some(current)) if expected == current => {}
                    _ => config_mismatch = true,
                }
            }
            aggregate.queries += snapshot.queries;
            aggregate.hits += snapshot.hits;
            aggregate.misses += snapshot.misses;
            aggregate.drafted_tokens += snapshot.drafted_tokens;
            aggregate.accepted_tokens += snapshot.accepted_tokens;
            aggregate.rejected_tokens += snapshot.rejected_tokens;
            aggregate.zero_accept_windows += snapshot.zero_accept_windows;
            aggregate.propose_us += snapshot.propose_us;
            aggregate.index_build_us += snapshot.index_build_us;
            aggregate.index_update_us += snapshot.index_update_us;
            aggregate.index_entries_current += snapshot.index_entries_current;
            aggregate.index_entries_peak += snapshot.index_entries_peak;
            aggregate.index_evictions += snapshot.index_evictions;
            aggregate.verify_round_us += snapshot.verify_round_us;
            aggregate.verify_forward_us += snapshot.verify_forward_us;
            aggregate.projection_us += snapshot.projection_us;
            aggregate.exact_batched_verify_windows += snapshot.exact_batched_verify_windows;
            aggregate.sequential_verify_windows += snapshot.sequential_verify_windows;
            aggregate.verify_accept_host_sync_count += snapshot.verify_accept_host_sync_count;
            aggregate.verify_accept_host_sync_us += snapshot.verify_accept_host_sync_us;
            aggregate.rollback_count += snapshot.rollback_count;
            aggregate.rollback_us += snapshot.rollback_us;
            aggregate.mtp_shadow_commit_windows += snapshot.mtp_shadow_commit_windows;
            aggregate.mtp_shadow_commit_tokens += snapshot.mtp_shadow_commit_tokens;
            aggregate.mtp_shadow_commit_us += snapshot.mtp_shadow_commit_us;
            aggregate.miss_fast_path_steps += snapshot.miss_fast_path_steps;
            aggregate.ordinary_cost_samples += snapshot.ordinary_cost_samples;
            aggregate.lookup_cost_samples += snapshot.lookup_cost_samples;
            aggregate.ordinary_cost_us += snapshot.ordinary_cost_us;
            aggregate.lookup_cost_us += snapshot.lookup_cost_us;
            aggregate.qualified_regimes_current += snapshot.qualified_regimes_current;
            aggregate.rejected_regimes_current += snapshot.rejected_regimes_current;
            aggregate.qualification_changes += snapshot.qualification_changes;
            aggregate.qualification_profile_loads += snapshot.qualification_profile_loads;
            aggregate.qualification_profile_writes += snapshot.qualification_profile_writes;
            aggregate.qualification_profile_write_drops +=
                snapshot.qualification_profile_write_drops;
            aggregate.hybrid_neural_windows += snapshot.hybrid_neural_windows;
            aggregate.hybrid_lookup_windows += snapshot.hybrid_lookup_windows;
            aggregate.hybrid_source_switches += snapshot.hybrid_source_switches;
            aggregate.hybrid_lookup_miss_fallbacks += snapshot.hybrid_lookup_miss_fallbacks;
            aggregate.hybrid_neural_rebases += snapshot.hybrid_neural_rebases;
            aggregate.hybrid_neural_rebase_us += snapshot.hybrid_neural_rebase_us;
        }
        if !config_mismatch {
            if let Some((min_ngram, max_ngram, max_draft_tokens, history, entries)) = config {
                aggregate.min_ngram = Some(min_ngram);
                aggregate.max_ngram = Some(max_ngram);
                aggregate.max_draft_tokens = Some(max_draft_tokens);
                aggregate.history_window_tokens = Some(history);
                aggregate.max_index_entries = Some(entries);
            }
        }
        aggregate
    }
}

#[derive(Clone)]
pub struct PromptLookupHealthConfig {
    config: Option<PromptLookupConfig>,
    stats: Arc<Mutex<Option<PromptLookupStats>>>,
}

impl PromptLookupHealthConfig {
    pub fn disabled() -> Self {
        Self {
            config: None,
            stats: Arc::new(Mutex::new(None)),
        }
    }

    pub fn enabled(
        config: PromptLookupConfig,
        stats: Arc<Mutex<Option<PromptLookupStats>>>,
    ) -> Self {
        Self {
            config: Some(config),
            stats,
        }
    }

    fn snapshot(&self) -> PromptLookupHealthInfo {
        let stats = self
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unwrap_or_default();
        PromptLookupHealthInfo {
            enabled: self.config.is_some(),
            min_ngram: self.config.map(|config| config.min_ngram),
            max_ngram: self.config.map(|config| config.max_ngram),
            max_draft_tokens: self.config.map(|config| config.max_draft_tokens),
            history_window_tokens: self.config.map(|config| config.history_window_tokens),
            max_index_entries: self.config.map(|config| config.max_index_entries),
            queries: stats.queries,
            hits: stats.hits,
            misses: stats.misses,
            drafted_tokens: stats.drafted_tokens,
            accepted_tokens: stats.accepted_tokens,
            rejected_tokens: stats.rejected_tokens,
            zero_accept_windows: stats.zero_accept_windows,
            exact_sampling_windows: stats.exact_sampling_windows,
            exact_acceptance_draws: stats.exact_acceptance_draws,
            exact_residual_corrections: stats.exact_residual_corrections,
            exact_bonus_samples: stats.exact_bonus_samples,
            propose_us: stats.propose_us,
            index_build_us: stats.index_build_us,
            index_update_us: stats.index_update_us,
            index_entries_current: stats.index_entries_current,
            index_entries_peak: stats.index_entries_peak,
            index_evictions: stats.index_evictions,
            verify_round_us: stats.verify_round_us,
            verify_forward_us: stats.verify_forward_us,
            projection_us: stats.projection_us,
            exact_batched_verify_windows: stats.exact_batched_verify_windows,
            sequential_verify_windows: stats.sequential_verify_windows,
            verify_accept_host_sync_count: stats.verify_accept_host_sync_count,
            verify_accept_host_sync_us: stats.verify_accept_host_sync_us,
            rollback_count: stats.rollback_count,
            rollback_us: stats.rollback_us,
            mtp_shadow_commit_windows: stats.mtp_shadow_commit_windows,
            mtp_shadow_commit_tokens: stats.mtp_shadow_commit_tokens,
            mtp_shadow_commit_us: stats.mtp_shadow_commit_us,
            miss_fast_path_steps: stats.miss_fast_path_steps,
            ordinary_cost_samples: stats.ordinary_cost_samples,
            lookup_cost_samples: stats.lookup_cost_samples,
            ordinary_cost_us: stats.ordinary_cost_us,
            lookup_cost_us: stats.lookup_cost_us,
            qualified_regimes_current: stats.qualified_regimes_current,
            rejected_regimes_current: stats.rejected_regimes_current,
            qualification_changes: stats.qualification_changes,
            qualification_profile_loads: stats.qualification_profile_loads,
            qualification_profile_writes: stats.qualification_profile_writes,
            qualification_profile_write_drops: stats.qualification_profile_write_drops,
            hybrid_neural_windows: stats.hybrid_neural_windows,
            hybrid_lookup_windows: stats.hybrid_lookup_windows,
            hybrid_source_switches: stats.hybrid_source_switches,
            hybrid_lookup_miss_fallbacks: stats.hybrid_lookup_miss_fallbacks,
            hybrid_neural_rebases: stats.hybrid_neural_rebases,
            hybrid_neural_rebase_us: stats.hybrid_neural_rebase_us,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthSnapshot {
    pub status: HealthStatus,
    pub uptime_secs: u64,
    pub model: ModelInfo,
    pub scheduler: SchedulerInfo,
    pub memory: MemoryInfo,
    pub mtp: MtpHealthInfo,
    pub prompt_lookup: PromptLookupHealthInfo,
    pub active_kv_offload: ActiveKvOffloadHealth,
    pub device_name: Option<String>,
    pub version: &'static str,
}

pub struct SchedulerHealthCollector {
    pub start_time: Instant,
    pub b_max: usize,
    pub queue_max: usize,
    pub model_name: String,
    pub max_position_embeddings: i32,
    pub b_active: Arc<AtomicU64>,
    pub b_queued: Arc<AtomicU64>,
    pub admit_count: Arc<AtomicU64>,
    pub batch_count: Arc<AtomicU64>,
    pub admission_queue_full_count: Arc<AtomicU64>,
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    pub kv_cache_soft_limit_bytes: usize,
    pub kv_cache_logical_cap_tokens: usize,
    pub kv_cache_resident_cap_tokens: usize,
    pub kv_cache_budget_policy: String,
    pub mtp: MtpHealthConfig,
    pub prompt_lookup: PromptLookupHealthConfig,
    pub active_kv_offload: ActiveKvOffloadSharedStats,
    pub immutable_prefix_blocks:
        crate::core::server::scheduler_actor::ImmutablePrefixBlockSharedStats,
}

impl SchedulerHealthCollector {
    pub fn snapshot(&self) -> HealthSnapshot {
        let uptime_secs = self.start_time.elapsed().as_secs();
        let total_ram_bytes = system_total_ram_bytes();
        let free_ram_bytes = system_free_ram_bytes();
        let b_active = self.b_active.load(Ordering::Relaxed) as usize;
        let b_queued = self.b_queued.load(Ordering::Relaxed) as usize;
        let admission_full = self.admission_queue_full_count.load(Ordering::Relaxed);
        let mb_exceeded = self.memory_budget_exceeded_count.load(Ordering::Relaxed);
        let kv_active = self.kv_cache_active_bytes.load(Ordering::Relaxed);
        let mlx_memory = mlx::memory::snapshot();
        let process_governor =
            crate::core::process_memory::global_process_memory_governor().sample_process();
        let prefix_store = crate::core::cache::process_async_prefix_store_queue().stats();

        let active_kv_offload = self.active_kv_offload.snapshot();
        let mut status = classify_status(
            b_queued,
            self.queue_max,
            free_ram_bytes,
            kv_active,
            self.kv_cache_soft_limit_bytes,
        );
        if active_kv_offload.degraded {
            status = HealthStatus::Degraded;
        }
        if crate::core::cache::process_async_prefix_store_queue().is_backpressured() {
            status = HealthStatus::Degraded;
        }
        if process_governor.pressure_level == crate::core::process_memory::PressureLevel::Emergency
        {
            status = HealthStatus::Down;
        } else if process_governor.telemetry_degraded
            || process_governor.pressure_level != crate::core::process_memory::PressureLevel::Normal
        {
            status = HealthStatus::Degraded;
        }
        HealthSnapshot {
            status,
            uptime_secs,
            model: ModelInfo {
                name: self.model_name.clone(),
                max_position_embeddings: self.max_position_embeddings,
            },
            scheduler: SchedulerInfo {
                b_max: self.b_max,
                b_active,
                b_queued,
                queue_max: self.queue_max,
                admit_count: self.admit_count.load(Ordering::Relaxed),
                batch_count: self.batch_count.load(Ordering::Relaxed),
                admission_queue_full_count: admission_full,
                memory_budget_exceeded_count: mb_exceeded,
            },
            memory: MemoryInfo {
                total_ram_bytes,
                free_ram_bytes,
                kv_cache_active_bytes: kv_active,
                kv_cache_soft_limit_bytes: self.kv_cache_soft_limit_bytes,
                kv_cache_logical_cap_tokens: self.kv_cache_logical_cap_tokens,
                kv_cache_resident_cap_tokens: self.kv_cache_resident_cap_tokens,
                kv_cache_budget_policy: self.kv_cache_budget_policy.clone(),
                mlx_total_bytes: mlx_memory.total_bytes,
                mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
                mlx_active_bytes: mlx_memory.active_bytes,
                mlx_cache_bytes: mlx_memory.cache_bytes,
                mlx_peak_bytes: mlx_memory.peak_bytes,
                mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
                process_governor,
                prefix_store,
                immutable_prefix_blocks: self.immutable_prefix_blocks.snapshot(),
            },
            mtp: self.mtp.snapshot(),
            prompt_lookup: self.prompt_lookup.snapshot(),
            active_kv_offload,
            device_name: mlx_memory.device_name,
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

pub fn classify_status(
    b_queued: usize,
    queue_max: usize,
    free_ram_bytes: usize,
    kv_cache_active_bytes: usize,
    kv_cache_soft_limit_bytes: usize,
) -> HealthStatus {
    let queue_high = queue_max > 0 && b_queued >= queue_max / 2;
    let mem_low = free_ram_bytes < (1024 * 1024 * 1024);
    let budget_near = kv_cache_soft_limit_bytes > 0
        && kv_cache_active_bytes >= ((kv_cache_soft_limit_bytes as f64) * 0.9) as usize;
    if queue_high || mem_low || budget_near {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

pub fn system_free_ram_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                let mut page_size = 16_384_usize;
                let mut pages_free = 0_usize;
                for line in s.lines() {
                    if let Some(rest) =
                        line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
                    {
                        if let Some(num) = rest.split(' ').next() {
                            if let Ok(p) = num.parse::<usize>() {
                                page_size = p;
                            }
                        }
                    }
                    if let Some(rest) = line.strip_prefix("Pages free:") {
                        let t = rest.trim().trim_end_matches('.');
                        if let Ok(n) = t.parse::<usize>() {
                            pages_free = n;
                        }
                    }
                }
                if pages_free > 0 {
                    return pages_free * page_size;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    4 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Arc;

    #[test]
    fn classify_healthy_when_all_green() {
        let s = classify_status(0, 32, 8 * 1024 * 1024 * 1024, 1_000_000, 10_000_000);
        assert!(matches!(s, HealthStatus::Healthy));
    }

    #[test]
    fn classify_degraded_when_queue_half_full() {
        let s = classify_status(16, 32, 8 * 1024 * 1024 * 1024, 0, 1);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_free_ram_low() {
        let s = classify_status(0, 32, 500_000_000, 0, 1);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_budget_near_soft_limit() {
        let s = classify_status(0, 32, 8 * 1024 * 1024 * 1024, 9_500_000, 10_000_000);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    fn test_collector(mtp: MtpHealthConfig) -> SchedulerHealthCollector {
        test_collector_with_active_kv(
            mtp,
            ActiveKvOffloadSharedStats::new(&crate::core::cache::ActiveKvOffloadConfig::disabled()),
        )
    }

    fn test_collector_with_active_kv(
        mtp: MtpHealthConfig,
        active_kv_offload: ActiveKvOffloadSharedStats,
    ) -> SchedulerHealthCollector {
        SchedulerHealthCollector {
            start_time: Instant::now(),
            b_max: 1,
            queue_max: 8,
            model_name: "test-model".to_string(),
            max_position_embeddings: 4096,
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admit_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: Arc::new(AtomicU64::new(0)),
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
            kv_cache_logical_cap_tokens: 262_144,
            kv_cache_resident_cap_tokens: 1_024,
            kv_cache_budget_policy: "active_kv_offload".to_string(),
            mtp,
            prompt_lookup: PromptLookupHealthConfig::disabled(),
            active_kv_offload,
            immutable_prefix_blocks:
                crate::core::server::scheduler_actor::ImmutablePrefixBlockSharedStats::new(false),
        }
    }

    #[test]
    fn snapshot_memory_reports_budget_policy_and_caps() {
        let collector = test_collector(MtpHealthConfig::disabled());
        collector.admit_count.store(3, Ordering::Relaxed);
        collector.batch_count.store(2, Ordering::Relaxed);
        let snapshot = collector.snapshot();

        assert_eq!(snapshot.memory.kv_cache_logical_cap_tokens, 262_144);
        assert_eq!(snapshot.memory.kv_cache_resident_cap_tokens, 1_024);
        assert_eq!(snapshot.memory.kv_cache_budget_policy, "active_kv_offload");
        assert_eq!(snapshot.scheduler.admit_count, 3);
        assert_eq!(snapshot.scheduler.batch_count, 2);
    }

    #[test]
    fn snapshot_mtp_reports_disabled_config() {
        let snapshot = test_collector(MtpHealthConfig::disabled()).snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.requested_draft_tokens, None);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 0);
        assert_eq!(snapshot.mtp.drafted_tokens, 0);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 0);
        assert_eq!(snapshot.mtp.windows, 0);
        assert_eq!(snapshot.mtp.draft_forward_us, 0);
        assert_eq!(snapshot.mtp.verify_forward_us, 0);
        assert_eq!(snapshot.mtp.projection_us, 0);
        assert_eq!(snapshot.mtp.sampling_us, 0);
        assert_eq!(snapshot.mtp.main_rollback_us, 0);
        assert_eq!(snapshot.mtp.cache_commit_us, 0);
        assert_eq!(snapshot.mtp.cache_restore_us, 0);
    }

    #[test]
    fn snapshot_prompt_lookup_reports_config_and_live_stats() {
        let config = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 5,
            max_draft_tokens: 3,
            history_window_tokens: 4096,
            max_index_entries: 8192,
        };
        let stats = PromptLookupStats {
            queries: 11,
            hits: 7,
            misses: 4,
            drafted_tokens: 19,
            accepted_tokens: 13,
            rejected_tokens: 6,
            exact_sampling_windows: 5,
            exact_acceptance_draws: 14,
            exact_residual_corrections: 3,
            exact_bonus_samples: 2,
            verify_round_us: 17,
            verify_accept_host_sync_count: 7,
            rollback_count: 2,
            miss_fast_path_steps: 3,
            ordinary_cost_samples: 8,
            lookup_cost_samples: 9,
            ordinary_cost_us: 10,
            lookup_cost_us: 11,
            exact_batched_verify_windows: 3,
            sequential_verify_windows: 4,
            qualified_regimes_current: 1,
            rejected_regimes_current: 2,
            qualification_changes: 4,
            qualification_profile_loads: 1,
            qualification_profile_writes: 5,
            qualification_profile_write_drops: 1,
            hybrid_neural_windows: 12,
            hybrid_lookup_windows: 6,
            hybrid_source_switches: 4,
            hybrid_lookup_miss_fallbacks: 3,
            hybrid_neural_rebases: 2,
            hybrid_neural_rebase_us: 29,
            ..PromptLookupStats::default()
        };
        let published = Arc::new(Mutex::new(Some(stats)));
        let mut collector = test_collector(MtpHealthConfig::disabled());
        collector.prompt_lookup = PromptLookupHealthConfig::enabled(config, published);

        let snapshot = collector.snapshot();

        assert!(snapshot.prompt_lookup.enabled);
        assert_eq!(snapshot.prompt_lookup.min_ngram, Some(2));
        assert_eq!(snapshot.prompt_lookup.max_ngram, Some(5));
        assert_eq!(snapshot.prompt_lookup.max_draft_tokens, Some(3));
        assert_eq!(snapshot.prompt_lookup.queries, 11);
        assert_eq!(snapshot.prompt_lookup.hits, 7);
        assert_eq!(snapshot.prompt_lookup.misses, 4);
        assert_eq!(snapshot.prompt_lookup.drafted_tokens, 19);
        assert_eq!(snapshot.prompt_lookup.accepted_tokens, 13);
        assert_eq!(snapshot.prompt_lookup.rejected_tokens, 6);
        assert_eq!(snapshot.prompt_lookup.exact_sampling_windows, 5);
        assert_eq!(snapshot.prompt_lookup.exact_acceptance_draws, 14);
        assert_eq!(snapshot.prompt_lookup.exact_residual_corrections, 3);
        assert_eq!(snapshot.prompt_lookup.exact_bonus_samples, 2);
        assert_eq!(snapshot.prompt_lookup.verify_round_us, 17);
        assert_eq!(snapshot.prompt_lookup.verify_accept_host_sync_count, 7);
        assert_eq!(snapshot.prompt_lookup.rollback_count, 2);
        assert_eq!(snapshot.prompt_lookup.miss_fast_path_steps, 3);
        assert_eq!(snapshot.prompt_lookup.ordinary_cost_samples, 8);
        assert_eq!(snapshot.prompt_lookup.lookup_cost_samples, 9);
        assert_eq!(snapshot.prompt_lookup.ordinary_cost_us, 10);
        assert_eq!(snapshot.prompt_lookup.lookup_cost_us, 11);
        assert_eq!(snapshot.prompt_lookup.exact_batched_verify_windows, 3);
        assert_eq!(snapshot.prompt_lookup.sequential_verify_windows, 4);
        assert_eq!(snapshot.prompt_lookup.qualified_regimes_current, 1);
        assert_eq!(snapshot.prompt_lookup.rejected_regimes_current, 2);
        assert_eq!(snapshot.prompt_lookup.qualification_changes, 4);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_loads, 1);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_writes, 5);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_write_drops, 1);
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_windows, 12);
        assert_eq!(snapshot.prompt_lookup.hybrid_lookup_windows, 6);
        assert_eq!(snapshot.prompt_lookup.hybrid_source_switches, 4);
        assert_eq!(snapshot.prompt_lookup.hybrid_lookup_miss_fallbacks, 3);
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_rebases, 2);
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_rebase_us, 29);
    }

    #[test]
    fn snapshot_mtp_reports_enabled_config_and_live_counters() {
        let prefill_count = Arc::new(AtomicU64::new(7));
        let step_count = Arc::new(AtomicU64::new(11));
        let fallback_prefill_count = Arc::new(AtomicU64::new(13));
        let drafted_tokens = Arc::new(AtomicU64::new(17));
        let accepted_draft_tokens = Arc::new(AtomicU64::new(19));
        let windows = Arc::new(AtomicU64::new(23));
        let exact_sampling_windows = Arc::new(AtomicU64::new(5));
        let exact_acceptance_draws = Arc::new(AtomicU64::new(12));
        let exact_residual_corrections = Arc::new(AtomicU64::new(3));
        let exact_bonus_samples = Arc::new(AtomicU64::new(2));
        let draft_forward_us = Arc::new(AtomicU64::new(29));
        let verify_forward_us = Arc::new(AtomicU64::new(31));
        let projection_us = Arc::new(AtomicU64::new(37));
        let sampling_us = Arc::new(AtomicU64::new(41));
        let draft_host_sync_count = Arc::new(AtomicU64::new(0));
        let draft_host_sync_us = Arc::new(AtomicU64::new(0));
        let verify_accept_host_sync_count = Arc::new(AtomicU64::new(23));
        let verify_accept_host_sync_us = Arc::new(AtomicU64::new(42));
        let main_rollback_us = Arc::new(AtomicU64::new(43));
        let cache_commit_us = Arc::new(AtomicU64::new(47));
        let prefill_cache_commit_us = Arc::new(AtomicU64::new(19));
        let decode_cache_commit_us = Arc::new(AtomicU64::new(28));
        let cache_restore_us = Arc::new(AtomicU64::new(53));
        let neural_exact_qualification_stats = Arc::new(std::sync::Mutex::new(
            crate::core::speculative_qualification::NeuralExactQualificationStats {
                ordinary_cost_samples: 8,
                exact_cost_samples: 5,
                rejected_regimes_current: 1,
                qualification_changes: 1,
                ..Default::default()
            },
        ));
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            1,
            prefill_count.clone(),
            step_count.clone(),
            fallback_prefill_count.clone(),
            drafted_tokens.clone(),
            accepted_draft_tokens.clone(),
            windows.clone(),
            exact_sampling_windows.clone(),
            exact_acceptance_draws.clone(),
            exact_residual_corrections.clone(),
            exact_bonus_samples.clone(),
            draft_forward_us.clone(),
            verify_forward_us.clone(),
            projection_us.clone(),
            sampling_us.clone(),
            draft_host_sync_count.clone(),
            draft_host_sync_us.clone(),
            verify_accept_host_sync_count.clone(),
            verify_accept_host_sync_us.clone(),
            main_rollback_us.clone(),
            cache_commit_us.clone(),
            prefill_cache_commit_us.clone(),
            decode_cache_commit_us.clone(),
            cache_restore_us.clone(),
            neural_exact_qualification_stats.clone(),
        ))
        .snapshot();

        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(1));

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, Some(1));
        assert_eq!(snapshot.mtp.prefill_count, 7);
        assert_eq!(snapshot.mtp.step_count, 11);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 13);
        assert_eq!(snapshot.mtp.drafted_tokens, 17);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 19);
        assert_eq!(snapshot.mtp.windows, 23);
        assert_eq!(snapshot.mtp.draft_forward_us, 29);
        assert_eq!(snapshot.mtp.verify_forward_us, 31);
        assert_eq!(snapshot.mtp.projection_us, 37);
        assert_eq!(snapshot.mtp.sampling_us, 41);
        assert_eq!(snapshot.mtp.draft_host_sync_count, 0);
        assert_eq!(snapshot.mtp.draft_host_sync_us, 0);
        assert_eq!(snapshot.mtp.verify_accept_host_sync_count, 23);
        assert_eq!(snapshot.mtp.verify_accept_host_sync_us, 42);
        assert_eq!(snapshot.mtp.main_rollback_us, 43);
        assert_eq!(snapshot.mtp.cache_commit_us, 47);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 19);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 28);
        assert_eq!(snapshot.mtp.cache_restore_us, 53);
        assert_eq!(
            snapshot
                .mtp
                .sampled_exact_qualification
                .ordinary_cost_samples,
            8
        );
        assert_eq!(
            snapshot.mtp.sampled_exact_qualification.exact_cost_samples,
            5
        );
        assert_eq!(
            snapshot
                .mtp
                .sampled_exact_qualification
                .rejected_regimes_current,
            1
        );

        prefill_count.store(13, Ordering::Relaxed);
        step_count.store(17, Ordering::Relaxed);
        fallback_prefill_count.store(23, Ordering::Relaxed);
        drafted_tokens.store(29, Ordering::Relaxed);
        accepted_draft_tokens.store(31, Ordering::Relaxed);
        windows.store(37, Ordering::Relaxed);
        exact_sampling_windows.store(7, Ordering::Relaxed);
        exact_acceptance_draws.store(18, Ordering::Relaxed);
        exact_residual_corrections.store(4, Ordering::Relaxed);
        exact_bonus_samples.store(3, Ordering::Relaxed);
        draft_forward_us.store(41, Ordering::Relaxed);
        verify_forward_us.store(43, Ordering::Relaxed);
        projection_us.store(47, Ordering::Relaxed);
        sampling_us.store(53, Ordering::Relaxed);
        verify_accept_host_sync_count.store(37, Ordering::Relaxed);
        verify_accept_host_sync_us.store(61, Ordering::Relaxed);
        main_rollback_us.store(59, Ordering::Relaxed);
        cache_commit_us.store(61, Ordering::Relaxed);
        prefill_cache_commit_us.store(29, Ordering::Relaxed);
        decode_cache_commit_us.store(32, Ordering::Relaxed);
        cache_restore_us.store(67, Ordering::Relaxed);
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            2,
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
            windows,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            draft_forward_us,
            verify_forward_us,
            projection_us,
            sampling_us,
            draft_host_sync_count,
            draft_host_sync_us,
            verify_accept_host_sync_count,
            verify_accept_host_sync_us,
            main_rollback_us,
            cache_commit_us,
            prefill_cache_commit_us,
            decode_cache_commit_us,
            cache_restore_us,
            neural_exact_qualification_stats,
        ))
        .snapshot();

        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 13);
        assert_eq!(snapshot.mtp.step_count, 17);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 23);
        assert_eq!(snapshot.mtp.drafted_tokens, 29);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 31);
        assert_eq!(snapshot.mtp.windows, 37);
        assert_eq!(snapshot.mtp.exact_sampling_windows, 7);
        assert_eq!(snapshot.mtp.exact_acceptance_draws, 18);
        assert_eq!(snapshot.mtp.exact_residual_corrections, 4);
        assert_eq!(snapshot.mtp.exact_bonus_samples, 3);
        assert_eq!(snapshot.mtp.draft_forward_us, 41);
        assert_eq!(snapshot.mtp.verify_forward_us, 43);
        assert_eq!(snapshot.mtp.projection_us, 47);
        assert_eq!(snapshot.mtp.sampling_us, 53);
        assert_eq!(snapshot.mtp.main_rollback_us, 59);
        assert_eq!(snapshot.mtp.cache_commit_us, 61);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 29);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 32);
        assert_eq!(snapshot.mtp.cache_restore_us, 67);
    }

    #[test]
    fn snapshot_degraded_when_active_kv_reports_error() {
        let active_kv_offload = ActiveKvOffloadSharedStats::new(
            &crate::core::cache::ActiveKvOffloadConfig::enabled(std::env::temp_dir()),
        );
        active_kv_offload.record_error();

        let snapshot =
            test_collector_with_active_kv(MtpHealthConfig::disabled(), active_kv_offload)
                .snapshot();

        assert!(matches!(snapshot.status, HealthStatus::Degraded));
        assert!(snapshot.active_kv_offload.degraded);
    }

    #[test]
    fn health_memory_serializes_mlx_allocator_fields() {
        let snapshot = HealthSnapshot {
            status: HealthStatus::Healthy,
            uptime_secs: 7,
            model: ModelInfo {
                name: "test-model".to_string(),
                max_position_embeddings: 4096,
            },
            scheduler: SchedulerInfo {
                b_max: 8,
                b_active: 1,
                b_queued: 0,
                queue_max: 16,
                admit_count: 1,
                batch_count: 1,
                admission_queue_full_count: 0,
                memory_budget_exceeded_count: 0,
            },
            memory: MemoryInfo {
                total_ram_bytes: 64,
                free_ram_bytes: 32,
                kv_cache_active_bytes: 16,
                kv_cache_soft_limit_bytes: 24,
                kv_cache_logical_cap_tokens: 128,
                kv_cache_resident_cap_tokens: 64,
                kv_cache_budget_policy: "full_resident".to_string(),
                mlx_total_bytes: Some(55),
                mlx_max_recommended_bytes: Some(66),
                mlx_active_bytes: 11,
                mlx_cache_bytes: 22,
                mlx_peak_bytes: 33,
                mlx_memory_limit_bytes: 44,
                process_governor: crate::core::process_memory::MemoryGovernorSnapshot::default(),
                prefix_store: crate::core::cache::AsyncPrefixStoreStats::default(),
                immutable_prefix_blocks:
                    crate::core::server::scheduler_actor::ImmutablePrefixBlockHealth::default(),
            },
            mtp: MtpHealthInfo {
                enabled: false,
                requested_draft_tokens: None,
                draft_tokens: None,
                prefill_count: 0,
                step_count: 0,
                fallback_prefill_count: 0,
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                windows: 0,
                exact_sampling_windows: 0,
                exact_acceptance_draws: 0,
                exact_residual_corrections: 0,
                exact_bonus_samples: 0,
                draft_forward_us: 0,
                verify_forward_us: 0,
                projection_us: 0,
                sampling_us: 0,
                draft_host_sync_count: 0,
                draft_host_sync_us: 0,
                verify_accept_host_sync_count: 0,
                verify_accept_host_sync_us: 0,
                main_rollback_us: 0,
                cache_commit_us: 0,
                prefill_cache_commit_us: 0,
                decode_cache_commit_us: 0,
                cache_restore_us: 0,
                sampled_exact_qualification: NeuralExactQualificationHealth::default(),
            },
            prompt_lookup: PromptLookupHealthInfo::default(),
            active_kv_offload: ActiveKvOffloadHealth::disabled(),
            device_name: Some("Apple Test GPU".to_string()),
            version: "test",
        };

        let value = serde_json::to_value(snapshot).expect("serialize health snapshot");
        assert_eq!(value["memory"]["mlx_total_bytes"], 55);
        assert_eq!(value["memory"]["mlx_max_recommended_bytes"], 66);
        assert_eq!(value["memory"]["mlx_active_bytes"], 11);
        assert_eq!(value["memory"]["mlx_cache_bytes"], 22);
        assert_eq!(value["memory"]["mlx_peak_bytes"], 33);
        assert_eq!(value["memory"]["mlx_memory_limit_bytes"], 44);
        assert_eq!(value["memory"]["kv_cache_logical_cap_tokens"], 128);
        assert_eq!(value["memory"]["kv_cache_resident_cap_tokens"], 64);
        assert_eq!(value["memory"]["kv_cache_budget_policy"], "full_resident");
        assert_eq!(value["device_name"], "Apple Test GPU");
    }
}
