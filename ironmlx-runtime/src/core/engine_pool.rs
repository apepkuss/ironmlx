//! Native model pool: registry, load/unload, leases, memory admission and idle eviction.
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use serde::Serialize;
use tokio::sync::{oneshot, watch, Mutex, Notify};

use crate::core::cache::{PagedPrefixCacheConfig, PrefixLruCacheConfig};
use crate::core::load_error::{classify_load_error, LoadFailed, RequestLoadFailed};
use crate::core::load_registry::{
    LoadActivity, LoadBegin, LoadCancelled, LoadHandle, LoadInProgress, LoadKind, LoadRegistry,
    LoadStage, LoadState, LoadTracker, LoadTriggerKind, LoadVariant, LoadWarning, WeightComponent,
};
use crate::core::process_memory::{
    global_process_memory_governor, MemoryReservation, PressureLevel, StaticMemoryEstimate,
};
use crate::core::prompt_lookup::PromptLookupConfig;
use crate::core::speculative::{MtpDraftTokensArg, MtpSpeculativeConfig};
use crate::Result;
use ironmlx_core::sampler::Sampler;
use ironmlx_image::models::QwenImage21Pipeline;
use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::speculative_model::MtpSpeculativeModel;
use {ironmlx_lm::core::loader::Loader, ironmlx_lm::core::tokenizer::Tokenizer};
use {
    ironmlx_lm::models::DiffusionGemmaConfig, ironmlx_lm::models::DiffusionGemmaGenerationConfig,
    ironmlx_lm::models::DiffusionGemmaModel, ironmlx_lm::models::Gemma4Model,
    ironmlx_lm::models::Glm4MoeLiteModel, ironmlx_lm::models::LlamaModel,
    ironmlx_lm::models::MiniCpmV46Model, ironmlx_lm::models::ModelArchitecture,
    ironmlx_lm::models::Qwen35Model, ironmlx_lm::models::Qwen35MoeModel,
    ironmlx_lm::models::Qwen36MoeModel,
};

use crate::core::engine_state::{
    CausalEngine as AppState, Gemma4DrafterEngine as Gemma4DrafterAppState,
};
use crate::core::runtime_health as health;
use ironmlx_lm::core::vision_input::VisionInputConfig;

pub use crate::core::runtime_config::*;

/// Bytes of the weight files under `model_paths`, each file counted once.
pub fn estimated_weight_bytes_for_roots<'a>(
    model_paths: impl IntoIterator<Item = &'a std::path::Path>,
) -> Result<usize> {
    let roots: Vec<PathBuf> = model_paths.into_iter().map(PathBuf::from).collect();
    let mut pending = roots.clone();
    let mut observed_files = HashSet::new();
    let mut bytes = 0usize;
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("reading model directory {}", directory.display()))?
        {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            // Hugging Face snapshots expose weight shards as symlinks into
            // the blob store. `DirEntry::file_type` describes the symlink,
            // while `metadata` follows it to the authoritative file size.
            let metadata = std::fs::metadata(entry.path())?;
            if !metadata.is_file() {
                continue;
            }
            let path = entry.path();
            let is_weight = matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("safetensors" | "gguf" | "bin")
            );
            if is_weight {
                let authoritative_path = std::fs::canonicalize(&path)
                    .with_context(|| format!("resolving model weight {}", path.display()))?;
                if observed_files.insert(authoritative_path) {
                    bytes = bytes.saturating_add(metadata.len() as usize);
                }
            }
        }
    }
    if bytes == 0 {
        let roots = roots
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!("cannot estimate model load memory: no weight files found under {roots}");
    }
    Ok(bytes)
}

#[cfg(test)]
fn estimated_model_weight_bytes(model_path: &std::path::Path) -> Result<usize> {
    estimated_weight_bytes_for_roots([model_path])
}

fn estimated_engine_weight_bytes(model: &EngineModelConfig) -> Result<usize> {
    let weights = estimated_weight_bytes_for_roots(
        std::iter::once(model.path.as_path())
            .chain(
                model
                    .mtp
                    .iter()
                    .map(|settings| settings.model_dir.as_path()),
            )
            .chain(
                model
                    .audio
                    .iter()
                    .map(|audio| audio.derived_resources.as_path()),
            ),
    )?;
    let weights = weights
        .checked_mul(model.decision.unwrap_or_default().dtype.weight_multiplier())
        .ok_or_else(|| anyhow::anyhow!("model load size overflow"))?;
    // Audio additionally retains verified FSTs, dictionary bytes and their Rust
    // indexes. Count auxiliary weights and reserve conservative frontend growth
    // before loading, while actual resident use is sampled by the same governor.
    weights
        .checked_add(if model.audio.is_some() {
            1024 * 1024 * 1024
        } else {
            0
        })
        .ok_or_else(|| anyhow::anyhow!("model load size overflow"))
}

#[derive(Clone)]
pub struct EnginePoolState {
    inner: Arc<EnginePoolInner>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineLoadedModelInfo {
    pub decision: Option<ironmlx_decision::DecisionSettings>,
    pub decision_metrics: Option<crate::core::decision_execution::DecisionMetricsSnapshot>,
    pub embedding_metrics: Option<crate::core::embedding_execution::EmbeddingMetricsSnapshot>,
    pub id: String,
    pub path: String,
    pub architecture: String,
    pub capabilities: EngineModelCapabilities,
    pub runtime_state: EngineRuntimeState,
    pub scheduler: Option<&'static str>,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub queue_capacity: usize,
    pub usage: crate::core::runtime_usage::ModelRuntimeUsageSnapshot,
    pub active_kv_offload: Option<crate::core::cache::ActiveKvOffloadHealth>,
    pub is_default: bool,
    pub pinned: bool,
    pub max_position_embeddings: i32,
    pub mtp_model_dir: Option<String>,
    pub mtp_draft_tokens: Option<usize>,
    pub prompt_lookup: Option<PromptLookupConfig>,
    /// Native reasoning contract of the loaded chat template; `None` when the
    /// model exposes no supported native reasoning channel.
    pub native_output_dialect: Option<ironmlx_lm::core::native_output::NativeOutputDialect>,
    /// Scheduler configuration in effect; `None` without a causal scheduler.
    pub scheduler_profile: Option<crate::core::scheduler_resolution::SchedulerProfileKind>,
}

struct EnginePoolInner {
    asr_settings: Mutex<HashMap<String, super::asr_execution::AsrSettings>>,
    registry: Mutex<EngineRegistry>,
    slots: Mutex<HashMap<String, Arc<EngineSlot>>>,
    runtime: EngineRuntimeOptions,
    capacity_policy: EnginePoolCapacityPolicy,
    load_gate: Mutex<()>,
    load_registry: Arc<LoadRegistry>,
    #[cfg(any(test, feature = "test-support"))]
    test_load_hook: std::sync::Mutex<Option<Arc<TestLoadHook>>>,
}

impl EnginePoolInner {
    fn new(
        registry: EngineRegistry,
        slots: HashMap<String, Arc<EngineSlot>>,
        runtime: EngineRuntimeOptions,
        capacity_policy: EnginePoolCapacityPolicy,
    ) -> Self {
        Self {
            asr_settings: Mutex::new(HashMap::new()),
            registry: Mutex::new(registry),
            slots: Mutex::new(slots),
            runtime,
            capacity_policy,
            load_gate: Mutex::new(()),
            load_registry: LoadRegistry::new(),
            #[cfg(any(test, feature = "test-support"))]
            test_load_hook: std::sync::Mutex::new(None),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    fn test_load_hook(&self) -> Option<Arc<TestLoadHook>> {
        self.test_load_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

struct EngineSlot {
    /// Cached at registration; loaded engines supply their actual admission cap.
    configured_context_window: Option<usize>,
    model: EngineModelConfig,
    runtime: EngineRuntimeOptions,
    active_requests: Arc<AtomicUsize>,
    pinned: AtomicBool,
    state: Mutex<EngineSlotState>,
    notify: Notify,
    /// The load this slot is running, from registration until the slot state
    /// records its result. Changed only while `state` is locked.
    pending_load: std::sync::Mutex<Option<PendingSlotLoad>>,
}

/// How the caller of a load is identified in the load registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLoadOptions {
    /// Client-chosen registry id; generated when absent.
    pub load_id: Option<String>,
    pub trigger: LoadTriggerKind,
    /// Reported with the load's result when it succeeds.
    pub warning: Option<LoadWarning>,
}

impl EngineLoadOptions {
    pub fn new(trigger: LoadTriggerKind) -> Self {
        Self {
            load_id: None,
            trigger,
            warning: None,
        }
    }
}

/// A request waited for a load of a slot that the model no longer uses (a
/// reload replaced it, or restored the slot it replaced); resolve the model's
/// slot again.
#[derive(Debug, thiserror::Error)]
#[error("engine slot was replaced while waiting for its load")]
pub struct EngineSlotReplaced {
    /// The awaited load was cancelled: the request may still be served by an
    /// engine that is already loaded, but must not start a load itself.
    pub load_cancelled: bool,
}

/// Stable identity of what a model configuration loads. `pinned` and
/// `default` only affect pool policy, so two requests that differ only there
/// load the same engine.
pub fn engine_model_config_fingerprint(model: &EngineModelConfig) -> String {
    let mut normalized = model.clone();
    normalized.pinned = false;
    normalized.default = false;
    // FNV-1a over the Debug form: every field type is plain data without
    // hash-ordered collections, so equal configurations print identically.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in format!("{normalized:?}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn load_variant(model: &EngineModelConfig) -> LoadVariant {
    if model.mtp.is_some() {
        LoadVariant::Mtp
    } else {
        LoadVariant::Plain
    }
}

/// A slot load other callers can join. The outcome is published once the
/// slot state holds the result.
#[derive(Clone)]
struct PendingSlotLoad {
    /// Registry id of the load; it finishes there only after a reload has
    /// settled which slot the model keeps.
    load_id: String,
    outcome: watch::Receiver<Option<SlotLoadOutcome>>,
}

#[derive(Debug, Clone)]
enum SlotLoadOutcome {
    Loaded,
    Cancelled,
    /// Classified like the starter's error, so joiners answer the same way.
    Failed(LoadFailed),
}

impl SlotLoadOutcome {
    fn failed(error: &anyhow::Error) -> Self {
        Self::Failed(LoadFailed::from_error(error))
    }

    fn aborted() -> Self {
        Self::Failed(LoadFailed {
            code: None,
            message: MODEL_LOAD_ABORTED.to_string(),
        })
    }
}

const MODEL_LOAD_ABORTED: &str = "model load ended without a result";

impl PendingSlotLoad {
    async fn wait(mut self) -> Result<()> {
        let outcome = match self.outcome.wait_for(Option::is_some).await {
            Ok(outcome) => outcome.clone(),
            Err(_) => None,
        };
        match outcome {
            Some(SlotLoadOutcome::Loaded) => Ok(()),
            Some(SlotLoadOutcome::Cancelled) => Err(LoadCancelled.into()),
            Some(SlotLoadOutcome::Failed(failed)) => Err(failed.into()),
            None => Err(anyhow::anyhow!(MODEL_LOAD_ABORTED)),
        }
    }
}

fn is_load_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<LoadCancelled>().is_some()
}

/// Records the end of one slot load exactly once: the slot state, clearing
/// the pending load and waking joiners change together under the state lock.
/// Dropped without settling (a panic), it returns the slot to `Unloaded`.
struct SlotLoadSettler {
    slot: Arc<EngineSlot>,
    outcome: Option<watch::Sender<Option<SlotLoadOutcome>>>,
}

impl SlotLoadSettler {
    async fn settle(&mut self, state: EngineSlotState, outcome: SlotLoadOutcome) {
        let slot = self.slot.clone();
        let mut guard = slot.state.lock().await;
        *guard = state;
        self.publish(outcome);
    }

    /// Cancelled loads leave the slot unloaded, not failed, so the next
    /// request or load starts afresh.
    async fn settle_cancelled(&mut self) {
        let slot = self.slot.clone();
        let mut guard = slot.state.lock().await;
        let load_attempts = guard.load_attempts();
        *guard = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Manual,
            last_error: None,
            changed_unix_ms: unix_time_ms(),
            load_attempts,
        };
        self.publish(SlotLoadOutcome::Cancelled);
    }

    async fn settle_failed(&mut self, load_attempts: u64, error: &anyhow::Error) {
        let failure = LoadFailed::from_error(error);
        self.settle(
            EngineSlotState::Failed {
                last_error: format!("{error:#}"),
                failure: failure.clone(),
                failed_unix_ms: unix_time_ms(),
                load_attempts,
            },
            SlotLoadOutcome::Failed(failure),
        )
        .await;
    }

    /// Call with the slot state locked.
    fn publish(&mut self, outcome: SlotLoadOutcome) {
        self.slot.pending_load().take();
        self.slot.notify.notify_waiters();
        if let Some(sender) = self.outcome.take() {
            sender.send_replace(Some(outcome));
        }
    }
}

impl Drop for SlotLoadSettler {
    fn drop(&mut self) {
        let Some(sender) = self.outcome.take() else {
            return;
        };
        fn abort(slot: &EngineSlot, state: &mut EngineSlotState) {
            if matches!(
                state,
                EngineSlotState::Unloaded { .. } | EngineSlotState::Loading { .. }
            ) {
                let load_attempts = state.load_attempts();
                *state = EngineSlotState::Unloaded {
                    reason: EngineUnloadReason::Manual,
                    last_error: Some(MODEL_LOAD_ABORTED.to_string()),
                    changed_unix_ms: unix_time_ms(),
                    load_attempts,
                };
            }
            slot.pending_load().take();
            slot.notify.notify_waiters();
        }
        tracing::error!(model = %self.slot.model.id, "model load ended without settling its slot");
        let outcome = Some(SlotLoadOutcome::aborted());
        match self.slot.state.try_lock() {
            Ok(mut state) => {
                abort(&self.slot, &mut state);
                sender.send_replace(outcome);
            }
            Err(_) => {
                let slot = self.slot.clone();
                let settle = async move {
                    let mut state = slot.state.lock().await;
                    abort(&slot, &mut state);
                    sender.send_replace(outcome);
                };
                match tokio::runtime::Handle::try_current() {
                    Ok(runtime) => {
                        runtime.spawn(settle);
                    }
                    Err(_) => tracing::error!("no runtime to settle an aborted model load"),
                }
            }
        }
    }
}

/// Run a synchronous loader or model-construction section without stalling
/// the other tasks of a multi-threaded runtime. The section stays on the
/// current thread, which MLX weight evaluation requires.
fn run_load_section<T>(section: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) if runtime.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(section)
        }
        _ => section(),
    }
}

enum EngineSlotState {
    Unloaded {
        reason: EngineUnloadReason,
        last_error: Option<String>,
        changed_unix_ms: u64,
        load_attempts: u64,
    },
    Loading {
        started_unix_ms: u64,
        load_attempts: u64,
    },
    Loaded {
        engine: Arc<EngineVariant>,
        snapshot_use_lease: Option<Arc<ModelSnapshotUseLease>>,
        loaded_unix_ms: u64,
        last_used_unix_ms: u64,
        load_attempts: u64,
        request_count: u64,
    },
    Draining {
        started_unix_ms: u64,
        load_attempts: u64,
        request_count: u64,
    },
    Failed {
        last_error: String,
        /// What later requests are told, classified like the first answer.
        failure: LoadFailed,
        failed_unix_ms: u64,
        load_attempts: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineUnloadReason {
    Startup,
    Evicted,
    Manual,
    Ttl,
    CapacityRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineRuntimeState {
    Disabled,
    Unloaded,
    Loading,
    Loaded,
    Draining,
    Failed,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineLoadTrigger {
    Request,
    Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnginePoolCapacityPolicy {
    EvictLruIdle,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnginePoolCapacityDecision {
    Continue,
    Reject,
    TryEvictLruIdle,
}

fn decide_engine_pool_capacity(
    policy: EnginePoolCapacityPolicy,
    max_loaded_models: usize,
    loaded_count: usize,
) -> EnginePoolCapacityDecision {
    if loaded_count < max_loaded_models {
        return EnginePoolCapacityDecision::Continue;
    }
    match policy {
        EnginePoolCapacityPolicy::EvictLruIdle => EnginePoolCapacityDecision::TryEvictLruIdle,
        EnginePoolCapacityPolicy::Reject => EnginePoolCapacityDecision::Reject,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvictionCandidate {
    id: String,
    last_used_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelTtlCandidate {
    id: String,
    state: EngineRuntimeState,
    active_requests: usize,
    pinned: bool,
    last_used_unix_ms: Option<u64>,
}

fn select_lru_eviction_candidate(mut candidates: Vec<EvictionCandidate>) -> Option<String> {
    candidates.sort_by(|left, right| {
        left.last_used_unix_ms
            .cmp(&right.last_used_unix_ms)
            .then_with(|| left.id.cmp(&right.id))
    });
    candidates.into_iter().next().map(|candidate| candidate.id)
}

fn select_model_ttl_unload_candidates(
    mut candidates: Vec<ModelTtlCandidate>,
    now_unix_ms: u64,
    ttl: Duration,
) -> Vec<String> {
    let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
    candidates.sort_by(|left, right| left.id.cmp(&right.id));
    candidates
        .into_iter()
        .filter(|candidate| {
            candidate.state == EngineRuntimeState::Loaded
                && candidate.active_requests == 0
                && !candidate.pinned
                && candidate
                    .last_used_unix_ms
                    .is_some_and(|last_used| now_unix_ms.saturating_sub(last_used) >= ttl_ms)
        })
        .map(|candidate| candidate.id)
        .collect()
}

fn model_ttl_sweep_interval(ttl: Duration) -> Duration {
    let seconds = ttl.as_secs().saturating_div(4).clamp(5, 60);
    Duration::from_secs(seconds)
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

impl EngineSlotState {
    fn load_attempts(&self) -> u64 {
        match self {
            Self::Unloaded { load_attempts, .. }
            | Self::Loading { load_attempts, .. }
            | Self::Loaded { load_attempts, .. }
            | Self::Draining { load_attempts, .. }
            | Self::Failed { load_attempts, .. } => *load_attempts,
        }
    }

    fn runtime_snapshot(&self) -> EngineSlotRuntimeSnapshot {
        match self {
            Self::Unloaded {
                reason,
                last_error,
                changed_unix_ms,
                load_attempts,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Unloaded,
                unload_reason: Some(*reason),
                last_error: last_error.clone(),
                changed_unix_ms: Some(*changed_unix_ms),
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: 0,
            },
            Self::Loading {
                started_unix_ms,
                load_attempts,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Loading,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: Some(*started_unix_ms),
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: 0,
            },
            Self::Loaded {
                loaded_unix_ms,
                last_used_unix_ms,
                load_attempts,
                request_count,
                ..
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Loaded,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: Some(*loaded_unix_ms),
                last_used_unix_ms: Some(*last_used_unix_ms),
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: *request_count,
            },
            Self::Draining {
                started_unix_ms,
                load_attempts,
                request_count,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Draining,
                unload_reason: Some(EngineUnloadReason::Manual),
                last_error: None,
                changed_unix_ms: Some(*started_unix_ms),
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: *request_count,
            },
            Self::Failed {
                last_error,
                failed_unix_ms,
                load_attempts,
                ..
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Failed,
                unload_reason: None,
                last_error: Some(last_error.clone()),
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: Some(*failed_unix_ms),
                load_attempts: *load_attempts,
                request_count: 0,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct EngineSlotRuntimeSnapshot {
    state: EngineRuntimeState,
    unload_reason: Option<EngineUnloadReason>,
    last_error: Option<String>,
    changed_unix_ms: Option<u64>,
    load_started_unix_ms: Option<u64>,
    loaded_unix_ms: Option<u64>,
    last_used_unix_ms: Option<u64>,
    failed_unix_ms: Option<u64>,
    load_attempts: u64,
    request_count: u64,
}

#[derive(Clone)]
pub enum EngineVariant {
    Asr(super::asr_execution::AsrRuntime),
    Alignment(super::alignment_execution::AlignmentRuntime),
    Diarization(super::diarization_execution::DiarizationRuntime),
    Embedding(super::embedding_execution::EmbeddingRuntime),
    Decision(super::decision_execution::DecisionRuntime),
    Audio(super::audio_execution::AudioRuntime),
    Qwen35(AppState<Qwen35Model>),
    Qwen35Moe(AppState<Qwen35MoeModel>),
    Qwen36Moe(AppState<Qwen36MoeModel>),
    Gemma4(AppState<Gemma4Model>),
    Gemma4Drafter(Box<Gemma4DrafterAppState>),
    Glm4MoeLite(AppState<Glm4MoeLiteModel>),
    Llama(AppState<LlamaModel>),
    MiniCpmV46(AppState<MiniCpmV46Model>),
    DiffusionGemma(crate::core::diffusion_execution::DiffusionGemmaRuntime),
    QwenImage(crate::core::qwen_image_execution::QwenImageRuntime),
    /// Engine produced by the test load hook; holds no model.
    #[cfg(any(test, feature = "test-support"))]
    Test(TestEngine),
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone)]
pub struct TestEngine {
    model_weight_bytes: usize,
}

pub struct EngineLease {
    engine: Arc<EngineVariant>,
    default_max_output_tokens: Option<usize>,
    active_requests: Option<Arc<AtomicUsize>>,
    _snapshot_use_lease: Option<Arc<ModelSnapshotUseLease>>,
}

impl std::fmt::Debug for EngineLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineLease").finish_non_exhaustive()
    }
}

impl EngineLease {
    fn new(
        engine: Arc<EngineVariant>,
        default_max_output_tokens: Option<usize>,
        active_requests: Option<Arc<AtomicUsize>>,
        snapshot_use_lease: Option<Arc<ModelSnapshotUseLease>>,
    ) -> Self {
        Self {
            engine,
            default_max_output_tokens,
            active_requests,
            _snapshot_use_lease: snapshot_use_lease,
        }
    }

    pub fn engine(&self) -> &EngineVariant {
        &self.engine
    }

    pub fn default_max_output_tokens(&self) -> Option<usize> {
        self.default_max_output_tokens
    }
}

#[derive(Debug)]
struct ModelSnapshotUseLease {
    file: File,
}

impl ModelSnapshotUseLease {
    fn acquire(model_path: &std::path::Path) -> Result<Option<Self>> {
        let canonical = std::fs::canonicalize(model_path)
            .with_context(|| format!("resolving model path {}", model_path.display()))?;
        let Some(commit) = canonical.file_name().and_then(|value| value.to_str()) else {
            return Ok(None);
        };
        if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(None);
        }
        let Some(snapshots) = canonical.parent() else {
            return Ok(None);
        };
        if snapshots.file_name().and_then(|value| value.to_str()) != Some("snapshots") {
            return Ok(None);
        }
        let Some(repository) = snapshots.parent() else {
            return Ok(None);
        };
        let lock_directory = repository.join(".locks");
        std::fs::create_dir_all(&lock_directory).with_context(|| {
            format!(
                "creating model snapshot use lock directory {}",
                lock_directory.display()
            )
        })?;
        let lock_path = lock_directory.join(format!("{}.use.lock", commit.to_ascii_lowercase()));
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("opening model snapshot use lock {}", lock_path.display()))?;
        // SAFETY: `file` owns a valid descriptor for the lifetime of this lease.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            bail!(
                "acquiring model snapshot use lock {}: {}",
                lock_path.display(),
                error
            );
        }
        Ok(Some(Self { file }))
    }
}

impl Drop for ModelSnapshotUseLease {
    fn drop(&mut self) {
        // SAFETY: `self.file` remains open until after `drop` returns.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        if let Some(active_requests) = &self.active_requests {
            active_requests.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl EnginePoolState {
    pub async fn new(config: EnginePoolConfig, runtime: EngineRuntimeOptions) -> Result<Self> {
        let registry = EngineRegistry::new(config.registry_manifest())?;
        config.validate_enabled_model_architectures()?;
        let mut slots = HashMap::with_capacity(config.models.len());
        for model in config.models {
            let pinned = model.pinned;
            let slot = Arc::new(EngineSlot::new_unloaded(model, runtime.clone(), pinned));
            slots.insert(slot.model.id.clone(), slot);
        }
        let state = Self {
            inner: Arc::new(EnginePoolInner::new(
                registry,
                slots,
                runtime,
                EnginePoolCapacityPolicy::EvictLruIdle,
            )),
        };
        state.preload().await?;
        Ok(state)
    }

    pub fn new_dynamic(
        runtime: EngineRuntimeOptions,
        max_loaded_models: Option<usize>,
    ) -> Result<Self> {
        let registry = EngineRegistry::empty(max_loaded_models)?;
        Ok(Self {
            inner: Arc::new(EnginePoolInner::new(
                registry,
                HashMap::new(),
                runtime,
                EnginePoolCapacityPolicy::Reject,
            )),
        })
    }

    pub fn start_model_ttl_sweeper(&self) {
        let Some(ttl) = self.inner.runtime.model_ttl else {
            return;
        };
        let pool = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(model_ttl_sweep_interval(ttl));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let unloaded = pool.unload_expired_model_ttl_once(unix_time_ms()).await;
                if !unloaded.is_empty() {
                    tracing::info!(
                        "ironmlx EnginePool TTL unloaded idle models count={} ids={}",
                        unloaded.len(),
                        unloaded.join(",")
                    );
                }
            }
        });
    }

    pub fn start_memory_governor_monitor(&self) {
        let weak = Arc::downgrade(&self.inner);
        let interval_duration = global_process_memory_governor().config().poll_interval;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let snapshot = global_process_memory_governor().sample_process();
                if snapshot.pressure_level == PressureLevel::Normal {
                    continue;
                }

                // MLX cache reclaim is idempotent and does not invalidate
                // request KV or model weights.
                mlx::transforms::clear_cache();
                // Only what usage itself is over the recovery watermark:
                // a request's reservations are served where they are made,
                // and taking more evicts prefixes its follow-ups restore.
                let governor = global_process_memory_governor();
                let needed_bytes = governor.recovery_overage(&governor.refresh_process());
                match crate::core::cache::reclaim_process_prefix_lru_bytes(needed_bytes, None) {
                    Ok(reclaimed) if reclaimed.bytes > 0 => {
                        tracing::info!(
                            needed_bytes,
                            reclaimed_bytes = reclaimed.bytes,
                            covered_entries = reclaimed.covered_entries,
                            other_entries = reclaimed.other_entries,
                            ?snapshot.pressure_level,
                            "memory governor shrank process hot prefix cache"
                        );
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "memory governor hot-cache shrink failed");
                    }
                }
                // Cache reclaim may have resolved the peak. Re-sample before
                // selecting any newly idle model, and after every eviction.
                let governor = global_process_memory_governor();
                let mut snapshot = governor.refresh_process();
                if snapshot.pressure_level.rank() >= PressureLevel::Hard.rank() {
                    tracing::info!(
                        level = ?snapshot.pressure_level,
                        current_bytes = snapshot.current_usage_bytes,
                        ceiling_bytes = snapshot.effective_ceiling_bytes,
                        reserved_bytes = snapshot.reserved_bytes,
                        model_load_reserved_bytes = snapshot.model_load_reserved_bytes,
                        recovery_bytes = (snapshot.effective_ceiling_bytes as f64 * governor.config().recovery_ratio) as usize,
                        sample_sequence = snapshot.sample_sequence,
                        eviction_needed = idle_eviction_needed(snapshot, governor.config()),
                        "memory governor post-reclaim eviction decision"
                    );
                }
                while idle_eviction_needed(snapshot, governor.config()) {
                    let evict_all = snapshot.pressure_level == PressureLevel::Emergency;
                    match inner.evict_lru_idle_engine("memory-pressure").await {
                        Ok(true) => {
                            snapshot = governor.refresh_process();
                            if !evict_all {
                                break;
                            }
                        }
                        Ok(false) => break,
                        Err(error) => {
                            tracing::warn!(%error, "memory governor idle-model eviction failed");
                            break;
                        }
                    }
                }
            }
        });
    }

    pub(crate) async fn unload_expired_model_ttl_once(&self, now_unix_ms: u64) -> Vec<String> {
        let Some(ttl) = self.inner.runtime.model_ttl else {
            return Vec::new();
        };
        let slots_snapshot = self.inner.slots.lock().await.clone();
        let mut candidates = Vec::with_capacity(slots_snapshot.len());
        for (id, slot) in &slots_snapshot {
            if slot.model.load_policy == EngineLoadPolicy::Preload {
                continue;
            }
            candidates.push(slot.model_ttl_candidate(id).await);
        }
        let ids = select_model_ttl_unload_candidates(candidates, now_unix_ms, ttl);
        let mut unloaded = Vec::new();
        for id in ids {
            let Some(slot) = slots_snapshot.get(&id) else {
                continue;
            };
            if slot.unload_if_ttl_expired(now_unix_ms, ttl).await {
                unloaded.push(id);
            }
        }
        unloaded
    }

    async fn preload(&self) -> Result<()> {
        let models = self.inner.registry.lock().await.models().to_vec();
        for model in models {
            if model.load_policy != EngineLoadPolicy::Preload {
                continue;
            }
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots
                    .get(&model.id)
                    .cloned()
                    .with_context(|| format!("missing engine slot `{}`", model.id))?
            };
            slot.ensure_loaded(
                &self.inner,
                EngineLoadTrigger::Control,
                EngineLoadOptions::new(LoadTriggerKind::Startup),
            )
            .await
            .with_context(|| {
                format!(
                    "preloading engine model `{}` from {}",
                    model.id,
                    model.path.display()
                )
            })?;
        }
        Ok(())
    }

    pub async fn is_audio_model(&self, requested: Option<&str>) -> Result<bool> {
        let id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(requested)?
            .to_owned();
        let slots = self.inner.slots.lock().await;
        Ok(slots.get(&id).is_some_and(|slot| {
            slot.model.audio.is_some()
                || matches!(
                    slot.model.capabilities.runtime_kind,
                    "asr" | "alignment" | "diarization"
                )
        }))
    }

    pub async fn is_diarization_model(&self, id: &str) -> Result<bool> {
        self.inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(id))?;
        Ok(self
            .inner
            .slots
            .lock()
            .await
            .get(id)
            .is_some_and(|slot| slot.model.capabilities.runtime_kind == "diarization"))
    }

    pub async fn is_alignment_model(&self, id: &str) -> Result<bool> {
        self.inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(id))?;
        Ok(self
            .inner
            .slots
            .lock()
            .await
            .get(id)
            .is_some_and(|slot| slot.model.capabilities.runtime_kind == "alignment"))
    }

    pub async fn is_asr_model(&self, id: &str) -> Result<bool> {
        self.inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(id))?;
        Ok(self
            .inner
            .slots
            .lock()
            .await
            .get(id)
            .is_some_and(|slot| slot.model.capabilities.runtime_kind == "asr"))
    }

    pub async fn asr_settings(&self, id: &str) -> Result<super::asr_execution::AsrSettings> {
        anyhow::ensure!(
            self.is_asr_model(id).await?,
            "ASR settings require an ASR model"
        );
        Ok(self
            .inner
            .asr_settings
            .lock()
            .await
            .get(id)
            .copied()
            .unwrap_or_default())
    }

    pub async fn set_asr_settings(
        &self,
        id: &str,
        settings: super::asr_execution::AsrSettings,
    ) -> Result<()> {
        settings.validate()?;
        anyhow::ensure!(
            self.is_asr_model(id).await?,
            "ASR settings require an ASR model"
        );
        self.inner
            .asr_settings
            .lock()
            .await
            .insert(id.to_owned(), settings);
        Ok(())
    }

    pub async fn is_embedding_model(&self, id: &str) -> bool {
        let slots = self.inner.slots.lock().await;
        slots
            .get(id)
            .is_some_and(|slot| slot.model.capabilities.runtime_kind == "embedding")
    }

    pub async fn is_decision_model(&self, requested: Option<&str>) -> Result<bool> {
        let id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(requested)?
            .to_owned();
        Ok(self
            .inner
            .slots
            .lock()
            .await
            .get(&id)
            .is_some_and(|slot| slot.model.capabilities.runtime_kind == "decision"))
    }

    pub async fn decision_model_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self
            .inner
            .slots
            .lock()
            .await
            .values()
            .filter(|slot| {
                slot.model.capabilities.runtime_kind == "decision"
                    && slot.model.load_policy != EngineLoadPolicy::Disabled
            })
            .map(|slot| slot.model.id.clone())
            .collect();
        ids.sort();
        ids
    }

    pub async fn resolve_engine(&self, requested: Option<&str>) -> Result<(String, EngineLease)> {
        let mut may_start_load = true;
        loop {
            let model_id = self
                .inner
                .registry
                .lock()
                .await
                .resolve_model_id(requested)?
                .to_string();
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots
                    .get(&model_id)
                    .cloned()
                    .with_context(|| format!("missing engine slot `{model_id}`"))?
            };
            // A reload that ended while this request waited swapped the slot
            // (or restored the previous one): serve from the current slot. A
            // cancelled load is not restarted by the requests that waited on
            // it.
            let result = slot
                .lease_or_load(
                    &self.inner,
                    EngineLoadTrigger::Request,
                    EngineLoadOptions::new(LoadTriggerKind::Request),
                    may_start_load,
                )
                .await;
            if let Some(replaced) = result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<EngineSlotReplaced>())
            {
                may_start_load &= !replaced.load_cancelled;
                continue;
            }
            return match result {
                Ok(engine) => Ok((model_id, engine)),
                Err(error) => {
                    // Classified before the model id joins the chain, the
                    // same way joiners and the failed slot classify it.
                    let classified = classify_load_error(&error);
                    let slot_failed =
                        matches!(*slot.state.lock().await, EngineSlotState::Failed { .. });
                    Err(error.context(RequestLoadFailed {
                        model_id,
                        classified,
                        slot_failed,
                    }))
                }
            };
        }
    }

    /// Loads in progress and recently finished, shared by every load path.
    pub fn load_registry(&self) -> Arc<LoadRegistry> {
        self.inner.load_registry.clone()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn install_test_load_hook(&self) -> Arc<TestLoadHook> {
        let hook = Arc::new(TestLoadHook::new());
        *self
            .inner
            .test_load_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook.clone());
        hook
    }

    /// Wait for `active`, a load of the same model already in progress, when
    /// it loads the configuration identified by `config_fingerprint`;
    /// otherwise refuse with [`LoadInProgress`].
    pub async fn join_in_progress_load(
        &self,
        active: &LoadActivity,
        config_fingerprint: &str,
    ) -> Result<LoadActivity> {
        if active.config_fingerprint != config_fingerprint {
            return Err(LoadInProgress {
                load_id: active.load_id.clone(),
            }
            .into());
        }
        let finished = self
            .inner
            .load_registry
            .wait(&active.load_id)
            .await
            .with_context(|| format!("model load `{}` is no longer tracked", active.load_id))?;
        match finished.state {
            LoadState::Loaded => Ok(finished),
            LoadState::Cancelled => Err(LoadCancelled.into()),
            _ => Err(LoadFailed::from_recorded(
                finished.code.as_deref(),
                finished.message.as_deref(),
            )
            .into()),
        }
    }

    pub async fn load_model(&self, requested: &str) -> Result<EngineModelControlResult> {
        self.load_model_with(
            requested,
            EngineLoadOptions::new(LoadTriggerKind::OpenAiControl),
        )
        .await
    }

    pub async fn load_model_with(
        &self,
        requested: &str,
        options: EngineLoadOptions,
    ) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.ensure_loaded(&self.inner, EngineLoadTrigger::Control, options)
            .await
            .with_context(|| format!("loading engine model `{model_id}`"))?;
        Ok(slot.control_result().await)
    }

    pub async fn unload_model(&self, requested: &str) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.unload().await?;
        Ok(slot.control_result().await)
    }

    pub async fn reload_dynamic_model(
        &self,
        model: EngineModelConfig,
        set_default: bool,
    ) -> Result<EngineModelControlResult> {
        self.reload_dynamic_model_with(
            model,
            set_default,
            EngineLoadOptions::new(LoadTriggerKind::Admin),
        )
        .await
    }

    /// Load `model` into a new slot, replacing the registered one only when
    /// the load succeeds. The swap, load and rollback run in their own task:
    /// a dropped caller neither cancels the load nor leaves the pool half
    /// swapped.
    pub async fn reload_dynamic_model_with(
        &self,
        model: EngineModelConfig,
        set_default: bool,
        options: EngineLoadOptions,
    ) -> Result<EngineModelControlResult> {
        let pool = self.clone();
        tokio::spawn(async move { pool.run_dynamic_reload(model, set_default, options).await })
            .await
            .map_err(|error| anyhow::anyhow!("model reload task failed: {error}"))?
    }

    async fn run_dynamic_reload(
        &self,
        model: EngineModelConfig,
        set_default: bool,
        options: EngineLoadOptions,
    ) -> Result<EngineModelControlResult> {
        validate_engine_model_config(&model)?;
        if set_default
            && matches!(
                model.capabilities.runtime_kind,
                "embedding" | "asr" | "alignment" | "diarization"
            )
        {
            bail!("embedding and ASR models cannot be the default generation model");
        }
        if model.load_policy == EngineLoadPolicy::Disabled {
            bail!("engine model `{}` is disabled", model.id);
        }
        let model_id = model.id.clone();
        let config_fingerprint = engine_model_config_fingerprint(&model);
        let current_slot = self.inner.slots.lock().await.get(&model_id).cloned();
        let replaces_loaded = match &current_slot {
            Some(slot) => slot.is_loaded().await,
            None => false,
        };
        #[cfg(any(test, feature = "test-support"))]
        if let Some(hook) = self.inner.test_load_hook() {
            hook.pause(TestLoadPause::BeforeRegistration).await;
        }
        // Registered before the slot swap: a second request for this model
        // joins or is refused instead of replacing the slot being loaded.
        let handle = match self.inner.load_registry.try_begin(LoadBegin {
            load_id: options.load_id,
            model: model_id.clone(),
            variant: load_variant(&model),
            kind: if replaces_loaded {
                LoadKind::Reload
            } else {
                LoadKind::Load
            },
            trigger: options.trigger,
            config_fingerprint: config_fingerprint.clone(),
            warning: options.warning,
        }) {
            Ok(handle) => handle,
            Err(active) => {
                self.join_in_progress_load(&active, &config_fingerprint)
                    .await?;
                let slot = self
                    .inner
                    .slots
                    .lock()
                    .await
                    .get(&model_id)
                    .cloned()
                    .with_context(|| format!("missing engine slot `{model_id}`"))?;
                return Ok(slot.control_result().await);
            }
        };

        let new_slot = Arc::new(EngineSlot::new_unloaded(
            model.clone(),
            self.inner.runtime.clone(),
            model.pinned,
        ));
        // Requests that find the new slot join this load.
        let (outcome, outcome_receiver) = watch::channel(None);
        *new_slot.pending_load() = Some(PendingSlotLoad {
            load_id: handle.load_id().to_string(),
            outcome: outcome_receiver,
        });

        let old_slot = self.inner.slots.lock().await.remove(&model_id);
        let previous_default = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let was_previous_default = previous_default.as_deref() == Some(model_id.as_str());
        let registered = {
            let mut registry = self.inner.registry.lock().await;
            registry.remove_model(&model_id);
            registry
                .upsert_model(model.manifest_view(), set_default || was_previous_default)
                .and_then(|()| {
                    if previous_default.is_none()
                        && matches!(
                            model.capabilities.runtime_kind,
                            "embedding" | "asr" | "alignment" | "diarization"
                        )
                    {
                        registry.restore_default_model(None)?;
                    }
                    Ok(())
                })
        };
        if let Err(error) = registered {
            let error = anyhow::Error::from(error);
            let restored = self
                .restore_replaced_slot(&model_id, old_slot, was_previous_default, previous_default)
                .await;
            handle.finish_failed(&error);
            restored?;
            return Err(error);
        }
        self.inner
            .slots
            .lock()
            .await
            .insert(model_id.clone(), new_slot.clone());

        let tracker = handle.tracker();
        match new_slot
            .run_load(&self.inner, &tracker, outcome, EngineLoadTrigger::Control)
            .await
        {
            Ok(lease) => {
                // A control load holds no request; release the engine
                // reference so the model stays evictable.
                drop(lease);
                handle.finish_loaded();
                Ok(new_slot.control_result().await)
            }
            Err(error) => {
                // A failed or cancelled reload restores the previous slot, so
                // the model stays loaded with its earlier configuration.
                let restored = self
                    .restore_replaced_slot(
                        &model_id,
                        old_slot,
                        was_previous_default,
                        previous_default,
                    )
                    .await;
                handle.finish_failed(&error);
                restored?;
                Err(error)
            }
        }
    }

    async fn restore_replaced_slot(
        &self,
        model_id: &str,
        old_slot: Option<Arc<EngineSlot>>,
        was_previous_default: bool,
        previous_default: Option<String>,
    ) -> Result<()> {
        self.inner.slots.lock().await.remove(model_id);
        {
            let mut registry = self.inner.registry.lock().await;
            registry.remove_model(model_id);
            if let Some(old_slot) = old_slot.as_ref() {
                registry.upsert_model(old_slot.model.manifest_view(), was_previous_default)?;
            }
            registry.restore_default_model(previous_default)?;
        }
        if let Some(old_slot) = old_slot {
            self.inner
                .slots
                .lock()
                .await
                .insert(model_id.to_string(), old_slot);
        }
        Ok(())
    }

    pub async fn register_dynamic_model(
        &self,
        model: EngineModelConfig,
        set_default: bool,
        pinned: Option<bool>,
    ) -> Result<EngineModelControlResult> {
        validate_engine_model_config(&model)?;
        if set_default
            && matches!(
                model.capabilities.runtime_kind,
                "embedding" | "asr" | "alignment" | "diarization"
            )
        {
            bail!("embedding and ASR models cannot be the default generation model");
        }
        let model_id = model.id.clone();
        let previous_default = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let existing_slot = self.inner.slots.lock().await.get(&model_id).cloned();
        let slot = match existing_slot {
            Some(existing) if existing.is_loaded_or_loading().await => {
                // Discovery registrations omit pinned so a stale scan cannot
                // overwrite a restore or an explicit pin/unpin operation.
                if let Some(pinned) = pinned {
                    existing.set_pinned(pinned);
                }
                existing
            }
            existing => {
                let pinned = pinned
                    .unwrap_or_else(|| existing.as_ref().is_some_and(|slot| slot.is_pinned()));
                let slot = Arc::new(EngineSlot::new_unloaded(
                    model.clone(),
                    self.inner.runtime.clone(),
                    pinned,
                ));
                self.inner
                    .slots
                    .lock()
                    .await
                    .insert(model_id.clone(), slot.clone());
                slot
            }
        };
        {
            let mut registry = self.inner.registry.lock().await;
            registry.upsert_model(model.manifest_view(), set_default)?;
            if previous_default.is_none() && !set_default {
                registry.restore_default_model(None)?;
            }
        }
        Ok(slot.control_result().await)
    }

    pub async fn unload_dynamic_model(&self, requested: &str) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.unload().await?;
        Ok(slot.control_result().await)
    }

    pub async fn set_model_pinned(
        &self,
        requested: &str,
        pinned: bool,
    ) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        if !slot.is_loaded().await {
            bail!("engine model `{model_id}` is not loaded");
        }
        slot.set_pinned(pinned);
        Ok(slot.control_result().await)
    }

    pub async fn set_default_model(&self, requested: &str) -> Result<()> {
        if self.is_embedding_model(requested).await
            || self.is_asr_model(requested).await?
            || self.is_alignment_model(requested).await?
            || self.is_diarization_model(requested).await?
        {
            bail!("embedding, ASR and alignment models cannot be the default generation model");
        }
        self.inner
            .registry
            .lock()
            .await
            .set_default_model(requested)?;
        Ok(())
    }

    pub async fn is_model_registered(&self, id: &str) -> bool {
        self.inner.registry.lock().await.model(id).is_some()
    }

    pub async fn is_model_loaded(&self, id: &str) -> bool {
        let slot = self.inner.slots.lock().await.get(id).cloned();
        match slot {
            Some(slot) => matches!(&*slot.state.lock().await, EngineSlotState::Loaded { .. }),
            None => false,
        }
    }

    pub async fn pending_requests(&self, id: &str) -> Option<usize> {
        let slot = self.inner.slots.lock().await.get(id).cloned();
        match slot {
            Some(slot) => slot.pending_requests().await,
            None => None,
        }
    }

    pub async fn loaded_model_infos(&self) -> Vec<EngineLoadedModelInfo> {
        let default_model = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let slots = self.inner.slots.lock().await.clone();
        let mut models = Vec::new();
        for (id, slot) in slots {
            let loaded = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    _ => None,
                }
            };
            let Some(engine) = loaded else {
                continue;
            };
            let health = engine.loaded_health();
            let usage = engine.runtime_usage_snapshot();
            let decision_metrics = match &health {
                LoadedEngineHealth::Decision { metrics, .. } => Some(*metrics),
                _ => None,
            };
            let embedding_metrics = match &health {
                LoadedEngineHealth::Embedding { metrics, .. } => Some(*metrics),
                _ => None,
            };
            let active_kv_offload = match &health {
                LoadedEngineHealth::Causal(snapshot) if snapshot.active_kv_offload.enabled => {
                    Some(snapshot.active_kv_offload.clone())
                }
                LoadedEngineHealth::Causal(_)
                | LoadedEngineHealth::DiffusionGemma { .. }
                | LoadedEngineHealth::ImageGeneration { .. }
                | LoadedEngineHealth::Audio { .. }
                | LoadedEngineHealth::Decision { .. }
                | LoadedEngineHealth::Embedding { .. } => None,
            };
            let (scheduler, active_requests, queued_requests, queue_capacity) = match &health {
                LoadedEngineHealth::Causal(snapshot) => (
                    Some("continuous_batching"),
                    snapshot.scheduler.b_active,
                    snapshot.scheduler.b_queued,
                    snapshot.scheduler.queue_max,
                ),
                LoadedEngineHealth::DiffusionGemma {
                    scheduler,
                    active_requests,
                    queued_requests,
                    queue_capacity,
                }
                | LoadedEngineHealth::ImageGeneration {
                    scheduler,
                    active_requests,
                    queued_requests,
                    queue_capacity,
                }
                | LoadedEngineHealth::Audio {
                    scheduler,
                    active_requests,
                    queued_requests,
                    queue_capacity,
                }
                | LoadedEngineHealth::Embedding {
                    scheduler,
                    active_requests,
                    queued_requests,
                    queue_capacity,
                    ..
                }
                | LoadedEngineHealth::Decision {
                    scheduler,
                    active_requests,
                    queued_requests,
                    queue_capacity,
                    ..
                } => (
                    Some(*scheduler),
                    *active_requests,
                    *queued_requests,
                    *queue_capacity,
                ),
            };
            models.push(EngineLoadedModelInfo {
                decision: slot.model.decision,
                decision_metrics,
                embedding_metrics,
                id: id.clone(),
                path: slot.model.path.to_string_lossy().into_owned(),
                architecture: engine.architecture().to_string(),
                capabilities: slot.model.capabilities.clone(),
                runtime_state: EngineRuntimeState::Loaded,
                scheduler,
                active_requests,
                queued_requests,
                queue_capacity,
                usage,
                active_kv_offload,
                is_default: default_model.as_deref() == Some(id.as_str()),
                max_position_embeddings: health.max_position_embeddings(),
                pinned: slot.is_pinned(),
                mtp_model_dir: slot
                    .model
                    .mtp
                    .as_ref()
                    .map(|mtp| mtp.model_dir.to_string_lossy().into_owned()),
                mtp_draft_tokens: slot.model.mtp.as_ref().and_then(|mtp| mtp.draft_tokens),
                prompt_lookup: slot.model.prompt_lookup,
                native_output_dialect: engine.native_output_dialect(),
                scheduler_profile: slot.model.scheduler_profile_kind,
            });
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    pub async fn loaded_causal_health_snapshots(&self) -> Vec<health::HealthSnapshot> {
        let slots = self.inner.slots.lock().await.clone();
        let mut snapshots = Vec::new();
        for slot in slots.values() {
            let loaded = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    _ => None,
                }
            };
            let Some(engine) = loaded else {
                continue;
            };
            if let LoadedEngineHealth::Causal(snapshot) = engine.loaded_health() {
                snapshots.push(*snapshot);
            }
        }
        snapshots
    }

    pub async fn clear_shared_prompt_lookup(
        &self,
        requested: Option<&str>,
    ) -> Result<(usize, usize)> {
        let requested_id = match requested {
            Some(requested) => Some(
                self.inner
                    .registry
                    .lock()
                    .await
                    .resolve_model_id(Some(requested))?
                    .to_string(),
            ),
            None => None,
        };
        let slots = self.inner.slots.lock().await.clone();
        let mut engines = Vec::new();
        for (id, slot) in slots {
            if requested_id
                .as_deref()
                .is_some_and(|requested| requested != id)
            {
                continue;
            }
            let engine = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    _ => None,
                }
            };
            if let Some(engine) = engine {
                engines.push(engine);
            }
        }
        if requested_id.is_some() && engines.is_empty() {
            bail!(
                "engine model `{}` is not loaded",
                requested_id.as_deref().unwrap_or_default()
            );
        }
        let mut cleared_entries = 0_usize;
        for engine in &engines {
            cleared_entries =
                cleared_entries.saturating_add(engine.clear_shared_prompt_lookup().await?);
        }
        Ok((engines.len(), cleared_entries))
    }

    pub async fn model_snapshots(&self) -> Vec<EngineModelSnapshot> {
        let mut data = Vec::new();
        let models = self.inner.registry.lock().await.servable_models_owned();
        for model in models {
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots.get(&model.id).cloned()
            };
            let (snapshot, context_window, default_max_output_tokens) = match slot {
                Some(slot) => {
                    let state = slot.state.lock().await;
                    let context_window = match &*state {
                        EngineSlotState::Loaded { engine, .. } => engine.context_window(),
                        _ => slot.configured_context_window,
                    };
                    (
                        state.runtime_snapshot(),
                        context_window,
                        slot.model.default_max_output_tokens,
                    )
                }
                None => (
                    EngineSlotRuntimeSnapshot {
                        state: EngineRuntimeState::Missing,
                        unload_reason: None,
                        last_error: Some("engine slot missing".to_string()),
                        changed_unix_ms: None,
                        load_started_unix_ms: None,
                        loaded_unix_ms: None,
                        last_used_unix_ms: None,
                        failed_unix_ms: None,
                        load_attempts: 0,
                        request_count: 0,
                    },
                    None,
                    None,
                ),
            };
            data.push(EngineModelSnapshot {
                id: model.id.clone(),
                context_window,
                // Causal serving has no independent output-only cap in model
                // metadata. Publish the deployment default when configured,
                // otherwise a conservative budget that leaves input room.
                max_output_tokens: super::model_capacity::advertised_max_output_tokens(
                    context_window,
                    default_max_output_tokens,
                ),
                load_policy: model.load_policy,
                state: snapshot.state,
                unload_reason: snapshot.unload_reason,
                last_error: snapshot.last_error,
                changed_unix_ms: snapshot.changed_unix_ms,
                load_started_unix_ms: snapshot.load_started_unix_ms,
                loaded_unix_ms: snapshot.loaded_unix_ms,
                last_used_unix_ms: snapshot.last_used_unix_ms,
                failed_unix_ms: snapshot.failed_unix_ms,
                load_attempts: snapshot.load_attempts,
                request_count: snapshot.request_count,
            });
        }
        data
    }

    pub async fn health_snapshot(&self) -> EnginePoolHealth {
        let mut models = Vec::new();
        let registry = self.inner.registry.lock().await;
        let default_model = registry.default_model().map(str::to_string);
        let max_loaded_models = registry.max_loaded_models();
        let registry_models = registry.models().to_vec();
        drop(registry);
        for model in registry_models {
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots.get(&model.id).cloned()
            };
            let Some(slot) = slot else {
                models.push(EngineModelHealth {
                    id: model.id.clone(),
                    load_policy: model.load_policy,
                    pinned: false,
                    state: EngineRuntimeState::Missing,
                    unload_reason: None,
                    last_error: Some("engine slot missing".to_string()),
                    loaded_unix_ms: None,
                    last_used_unix_ms: None,
                    load_started_unix_ms: None,
                    failed_unix_ms: None,
                    load_attempts: 0,
                    request_count: 0,
                    loaded: None,
                });
                continue;
            };
            models.push(slot.health_snapshot().await);
        }
        let loaded_models = models
            .iter()
            .filter(|model| model.state == EngineRuntimeState::Loaded)
            .count();
        let failed_models = models
            .iter()
            .filter(|model| model.state == EngineRuntimeState::Failed)
            .count();
        let process_governor = global_process_memory_governor().sample_process();
        let prefix_store = crate::core::cache::process_async_prefix_store_queue().stats();
        let mut initial_reasons = Vec::new();
        if failed_models > 0 {
            initial_reasons.push(health::HealthDegradedReason::ModelFailed);
        }
        if crate::core::cache::process_async_prefix_store_queue().is_backpressured() {
            initial_reasons.push(health::HealthDegradedReason::PrefixStoreBackpressured);
        }
        let (status, degraded_reasons) =
            health::classify_process_status(&process_governor, initial_reasons);
        EnginePoolHealth {
            status,
            degraded_reasons,
            mode: "engine_pool",
            default_model,
            max_loaded_models,
            loaded_models,
            models,
            process_governor,
            prefix_store,
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// A controlled telemetry pulse for lifecycle regression, never a real
/// allocation or a production setting. Subsequent native samples remain real.
#[cfg(any(test, feature = "test-support"))]
fn diagnostic_model_load_pressure_pulse() {
    static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if std::env::var("IRONMLX_DIAGNOSTIC_MODEL_LOAD_PRESSURE_PULSE").as_deref() != Ok("1")
        || FIRED.swap(true, Ordering::SeqCst)
    {
        return;
    }
    let governor = global_process_memory_governor();
    let before = governor.refresh_process();
    let mut pulse = crate::core::process_memory::native_memory_telemetry();
    let peak = before
        .effective_ceiling_bytes
        .saturating_add(governor.config().emergency_overage_bytes);
    pulse.phys_footprint_bytes = Some(peak);
    pulse.mlx_active_bytes = Some(peak);
    let injected = governor.update(pulse);
    tracing::warn!(native_current_bytes=before.current_usage_bytes,injected_current_bytes=injected.current_usage_bytes,ceiling_bytes=injected.effective_ceiling_bytes,level=?injected.pressure_level,"controlled model-load telemetry pulse; no corresponding native allocation");
}

/// Admission and cache limits retain their hysteresis. Destructive model
/// reclamation stops once a fresh authoritative sample has already crossed
/// the same recovery watermark; a cooling-down grade alone is insufficient.
fn idle_eviction_needed(
    snapshot: crate::core::process_memory::MemoryGovernorSnapshot,
    config: crate::core::process_memory::MemoryGovernorConfig,
) -> bool {
    if snapshot.pressure_level.rank() < PressureLevel::Hard.rank() {
        return false;
    }
    if snapshot.telemetry_degraded || snapshot.effective_ceiling_bytes == 0 {
        return true;
    }
    let observed = snapshot.current_usage_bytes.saturating_add(
        snapshot
            .reserved_bytes
            .saturating_sub(snapshot.model_load_reserved_bytes),
    );
    observed > (snapshot.effective_ceiling_bytes as f64 * config.recovery_ratio) as usize
}

impl EnginePoolInner {
    async fn ensure_capacity_for(
        &self,
        model: &EngineModelConfig,
    ) -> Result<Option<MemoryReservation>> {
        let target_id = model.id.as_str();
        let estimated_weight_bytes = estimated_engine_weight_bytes(model)?;
        // A target that cannot fit by itself must not evict existing models
        // while attempting to reclaim capacity for an impossible load.
        self.runtime
            .memory_limits
            .check_model_memory_limit(target_id, estimated_weight_bytes)?;
        self.runtime
            .memory_limits
            .check_total_memory_limit(target_id, estimated_weight_bytes)?;
        let governor = global_process_memory_governor();
        let first_snapshot = governor.sample_process();
        let first_reservation = if first_snapshot.pressure_level == PressureLevel::Normal {
            governor.try_reserve_model_load(estimated_weight_bytes)
        } else {
            Err(
                crate::core::process_memory::MemoryReservationError::Pressure {
                    purpose: "model_load",
                    level: first_snapshot.pressure_level,
                },
            )
        };
        let memory_reservation = match first_reservation {
            Ok(reservation) => Some(reservation),
            Err(first_error) => {
                // One reclaim/retry attempt. Every action is idempotent:
                // clearing an empty MLX cache is a no-op, the prefix cache
                // gives up only what is still missing, and idle eviction
                // rechecks ownership while holding the slot lock.
                mlx::transforms::clear_cache();
                let needed_bytes = governor.reservation_shortfall(
                    &governor.refresh_process(),
                    estimated_weight_bytes,
                    crate::core::process_memory::ReservationNeed::Admission,
                );
                let reclaimed =
                    crate::core::cache::reclaim_process_prefix_lru_bytes(needed_bytes, None)?;
                if reclaimed.bytes > 0 {
                    mlx::transforms::clear_cache();
                    // The release takes a few samples to show.
                    governor.refresh_after_release(&mut |snapshot| {
                        snapshot.pressure_level == PressureLevel::Normal
                    });
                    tracing::info!(
                        model = target_id,
                        needed_bytes,
                        reclaimed_bytes = reclaimed.bytes,
                        "hot prefix cache made room for a model load"
                    );
                }
                let _ = self.evict_lru_idle_engine(target_id).await?;
                let retry_snapshot = governor.refresh_process();
                if retry_snapshot.pressure_level != PressureLevel::Normal {
                    bail!(
                        "process memory governor rejected model load `{target_id}` after one reclaim retry: first={first_error}; pressure={:?}",
                        retry_snapshot.pressure_level
                    );
                }
                match governor.try_reserve_model_load(estimated_weight_bytes) {
                    Ok(reservation) => Some(reservation),
                    Err(retry_error) => {
                        bail!(
                            "process memory governor rejected model load `{target_id}` after one reclaim retry: first={first_error}; retry={retry_error}"
                        )
                    }
                }
            }
        };
        let max_loaded_models = self.registry.lock().await.max_loaded_models();
        if let Some(max_loaded_models) = max_loaded_models {
            let loaded_count = self.loaded_count().await;
            match decide_engine_pool_capacity(self.capacity_policy, max_loaded_models, loaded_count)
            {
                EnginePoolCapacityDecision::Continue => {}
                EnginePoolCapacityDecision::Reject => {
                    bail!(
                        "engine pool capacity reached: max_loaded_models={max_loaded_models}, unload an existing model before loading `{target_id}`"
                    );
                }
                EnginePoolCapacityDecision::TryEvictLruIdle => {
                    if !self.evict_lru_idle_engine(target_id).await? {
                        bail!(
                            "engine pool capacity reached: max_loaded_models={max_loaded_models}, no idle lazy engine can be evicted"
                        );
                    }
                }
            }
        }

        // Reject known oversize loads before constructing MLX arrays. The
        // post-load check remains necessary for transformations whose live
        // allocation differs from the on-disk weight estimate.
        if self
            .runtime
            .memory_limits
            .model_memory_limit_bytes
            .is_some()
        {
            let estimated_loaded_model_bytes = self
                .loaded_model_weight_bytes_excluding(target_id)
                .await
                .saturating_add(estimated_weight_bytes);
            self.runtime
                .memory_limits
                .check_model_memory_limit(target_id, estimated_loaded_model_bytes)?;
        }
        if self
            .runtime
            .memory_limits
            .total_memory_limit_bytes
            .is_some()
        {
            let estimated_active_bytes = mlx::memory::snapshot()
                .active_bytes
                .saturating_add(estimated_weight_bytes);
            self.runtime
                .memory_limits
                .check_total_memory_limit(target_id, estimated_active_bytes)?;
        }
        Ok(memory_reservation)
    }

    async fn loaded_count(&self) -> usize {
        let mut loaded = 0;
        let slots = self.slots.lock().await;
        for slot in slots.values() {
            if matches!(
                &*slot.state.lock().await,
                EngineSlotState::Loaded { .. } | EngineSlotState::Draining { .. }
            ) {
                loaded += 1;
            }
        }
        loaded
    }

    async fn check_memory_limits_after_load(
        &self,
        target_id: &str,
        engine: &EngineVariant,
    ) -> Result<()> {
        let loaded_model_bytes = self
            .loaded_model_weight_bytes_excluding(target_id)
            .await
            .saturating_add(engine.model_weight_bytes());
        self.runtime
            .memory_limits
            .check_model_memory_limit(target_id, loaded_model_bytes)?;

        let mlx_active_bytes = mlx::memory::snapshot().active_bytes;
        self.runtime
            .memory_limits
            .check_total_memory_limit(target_id, mlx_active_bytes)?;
        Ok(())
    }

    async fn loaded_model_weight_bytes_excluding(&self, exclude_id: &str) -> usize {
        let slots_snapshot = self.slots.lock().await.clone();
        let mut total = 0usize;
        for (id, slot) in slots_snapshot {
            if id == exclude_id {
                continue;
            }
            let engine = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    EngineSlotState::Draining { .. } => None,
                    _ => None,
                }
            };
            if let Some(engine) = engine {
                total = total.saturating_add(engine.model_weight_bytes());
            }
        }
        total
    }

    async fn evict_lru_idle_engine(&self, target_id: &str) -> Result<bool> {
        let mut candidates = Vec::new();
        let slots_snapshot = self.slots.lock().await.clone();
        for (id, slot) in &slots_snapshot {
            if id == target_id
                || slot.model.load_policy == EngineLoadPolicy::Preload
                || slot.is_pinned()
            {
                continue;
            }
            let guard = slot.state.lock().await;
            let EngineSlotState::Loaded {
                engine,
                last_used_unix_ms,
                ..
            } = &*guard
            else {
                continue;
            };
            if slot.active_requests.load(Ordering::SeqCst) == 0 && Arc::strong_count(engine) == 1 {
                candidates.push(EvictionCandidate {
                    id: id.clone(),
                    last_used_unix_ms: *last_used_unix_ms,
                });
            }
        }

        let Some(evict_id) = select_lru_eviction_candidate(candidates) else {
            return Ok(false);
        };
        let Some(slot) = self.slots.lock().await.get(&evict_id).cloned() else {
            return Ok(false);
        };
        let mut guard = slot.state.lock().await;
        let load_attempts = guard.load_attempts();
        let EngineSlotState::Loaded { engine, .. } = &*guard else {
            return Ok(false);
        };
        if slot.is_pinned()
            || slot.active_requests.load(Ordering::SeqCst) != 0
            || Arc::strong_count(engine) != 1
        {
            return Ok(false);
        }
        tracing::info!(
            model = %evict_id,
            target = target_id,
            active_requests = slot.active_requests.load(Ordering::SeqCst),
            engine_references = Arc::strong_count(engine),
            pinned = slot.is_pinned(),
            "memory eviction candidate rechecked under slot lock"
        );
        *guard = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Evicted,
            last_error: None,
            changed_unix_ms: unix_time_ms(),
            load_attempts,
        };
        crate::core::cache::cancel_process_async_prefix_store_model(&evict_id);
        tracing::info!(
            "ironmlx EnginePool evicted idle model id={} for target={}",
            evict_id,
            target_id
        );
        slot.notify.notify_waiters();
        Ok(true)
    }
}

impl EngineSlot {
    fn new_unloaded(model: EngineModelConfig, runtime: EngineRuntimeOptions, pinned: bool) -> Self {
        Self {
            configured_context_window: super::model_capacity::configured_context_window(&model),
            model,
            runtime,
            active_requests: Arc::new(AtomicUsize::new(0)),
            pinned: AtomicBool::new(pinned),
            state: Mutex::new(EngineSlotState::Unloaded {
                reason: EngineUnloadReason::Startup,
                last_error: None,
                changed_unix_ms: unix_time_ms(),
                load_attempts: 0,
            }),
            notify: Notify::new(),
            pending_load: std::sync::Mutex::new(None),
        }
    }

    fn pending_load(&self) -> std::sync::MutexGuard<'_, Option<PendingSlotLoad>> {
        self.pending_load
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::SeqCst)
    }

    fn set_pinned(&self, pinned: bool) {
        self.pinned.store(pinned, Ordering::SeqCst);
    }

    async fn is_loaded_or_loading(&self) -> bool {
        let state = self.state.lock().await;
        matches!(
            &*state,
            EngineSlotState::Loaded { .. }
                | EngineSlotState::Loading { .. }
                | EngineSlotState::Draining { .. }
        ) || self.pending_load().is_some()
    }

    async fn is_loaded(&self) -> bool {
        matches!(&*self.state.lock().await, EngineSlotState::Loaded { .. })
    }

    /// Whether the pool still serves this model from this slot.
    async fn is_current_slot(self: &Arc<Self>, pool: &EnginePoolInner) -> bool {
        pool.slots
            .lock()
            .await
            .get(&self.model.id)
            .is_some_and(|slot| Arc::ptr_eq(slot, self))
    }

    /// Return a lease on the loaded engine, loading it first when needed.
    ///
    /// The load runs in its own task, registered in the pool's load registry
    /// before it queues for the load gate. Callers only wait for its result:
    /// dropping the returned future never cancels a load, and concurrent
    /// callers join the load already running for this slot. While another
    /// load of the same model is registered (a reload swapping slots), this
    /// slot waits for it instead of starting a second one.
    async fn ensure_loaded(
        self: &Arc<Self>,
        pool: &Arc<EnginePoolInner>,
        trigger: EngineLoadTrigger,
        options: EngineLoadOptions,
    ) -> Result<EngineLease> {
        self.lease_or_load(pool, trigger, options, true).await
    }

    /// [`Self::ensure_loaded`]; without `may_start_load` the slot is served
    /// only when loaded or by joining a load, and otherwise answers
    /// [`LoadCancelled`].
    async fn lease_or_load(
        self: &Arc<Self>,
        pool: &Arc<EnginePoolInner>,
        trigger: EngineLoadTrigger,
        options: EngineLoadOptions,
        may_start_load: bool,
    ) -> Result<EngineLease> {
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            bail!("engine model `{}` is disabled", self.model.id);
        }

        enum Next<'a> {
            Wait(tokio::sync::futures::Notified<'a>),
            Join(PendingSlotLoad),
            JoinRegistered(String),
            Started(oneshot::Receiver<Result<EngineLease>>),
        }
        let mut load_id = options.load_id;
        loop {
            let next = {
                let mut state = self.state.lock().await;
                match &mut *state {
                    EngineSlotState::Loaded {
                        engine,
                        snapshot_use_lease,
                        last_used_unix_ms,
                        request_count,
                        ..
                    } => {
                        let active_requests = if trigger == EngineLoadTrigger::Request {
                            *last_used_unix_ms = unix_time_ms();
                            *request_count = request_count.saturating_add(1);
                            self.active_requests.fetch_add(1, Ordering::SeqCst);
                            Some(self.active_requests.clone())
                        } else {
                            None
                        };
                        return Ok(EngineLease::new(
                            engine.clone(),
                            self.model.default_max_output_tokens,
                            active_requests,
                            snapshot_use_lease.clone(),
                        ));
                    }
                    EngineSlotState::Draining { .. } => {
                        if self.active_requests.load(Ordering::SeqCst) == 0 {
                            let load_attempts = state.load_attempts();
                            *state = EngineSlotState::Unloaded {
                                reason: EngineUnloadReason::Manual,
                                last_error: None,
                                changed_unix_ms: unix_time_ms(),
                                load_attempts,
                            };
                        } else {
                            bail!("engine model `{}` is currently unloading", self.model.id);
                        }
                    }
                    EngineSlotState::Failed {
                        last_error,
                        failure,
                        ..
                    } if trigger == EngineLoadTrigger::Request && self.pending_load().is_none() => {
                        let failed_state =
                            format!("engine model `{}` is in failed state", self.model.id);
                        // Answer like the request that saw the load fail.
                        // The failure travels as `LoadFailed`, so the model
                        // id in the text cannot change its class; an
                        // unclassified one keeps the raw chain as its text.
                        let context = if failure.code.is_some() {
                            format!("{failed_state}: {last_error}")
                        } else {
                            failed_state
                        };
                        return Err(anyhow::Error::from(failure.clone()).context(context));
                    }
                    EngineSlotState::Loading { .. }
                    | EngineSlotState::Unloaded { .. }
                    | EngineSlotState::Failed { .. } => {}
                }
                let mut pending = self.pending_load();
                if let Some(load) = pending.as_ref() {
                    Next::Join(load.clone())
                } else if matches!(&*state, EngineSlotState::Loading { .. }) {
                    Next::Wait(self.notify.notified())
                } else if !may_start_load {
                    return Err(LoadCancelled.into());
                } else {
                    match pool.load_registry.try_begin(LoadBegin {
                        load_id: load_id.clone(),
                        model: self.model.id.clone(),
                        variant: load_variant(&self.model),
                        kind: LoadKind::Load,
                        trigger: options.trigger,
                        config_fingerprint: engine_model_config_fingerprint(&self.model),
                        warning: None,
                    }) {
                        Ok(handle) => {
                            load_id = None;
                            let (load, result) = self.spawn_load(pool, handle, trigger);
                            *pending = Some(load);
                            Next::Started(result)
                        }
                        Err(active) => Next::JoinRegistered(active.load_id),
                    }
                }
            };
            match next {
                Next::Wait(notified) => notified.await,
                Next::Join(load) => {
                    #[cfg(any(test, feature = "test-support"))]
                    if let Some(hook) = pool.test_load_hook() {
                        hook.note_join();
                    }
                    let load_id = load.load_id.clone();
                    let joined = load.wait().await;
                    if trigger == EngineLoadTrigger::Request {
                        // A reload finishes in the registry once it has kept
                        // or restored a slot; a request then needs that slot.
                        let _ = pool.load_registry.wait(&load_id).await;
                        if !self.is_current_slot(pool).await {
                            return Err(EngineSlotReplaced {
                                load_cancelled: joined.as_ref().is_err_and(is_load_cancelled),
                            }
                            .into());
                        }
                    }
                    joined?;
                }
                Next::JoinRegistered(active) => {
                    let finished = pool.load_registry.wait(&active).await;
                    if trigger == EngineLoadTrigger::Request && !self.is_current_slot(pool).await {
                        return Err(EngineSlotReplaced {
                            load_cancelled: finished
                                .as_ref()
                                .is_some_and(|load| load.state == LoadState::Cancelled),
                        }
                        .into());
                    }
                    match finished {
                        Some(load) if load.state == LoadState::Cancelled => {
                            return Err(LoadCancelled.into())
                        }
                        Some(load) if load.state == LoadState::Failed => {
                            return Err(LoadFailed::from_recorded(
                                load.code.as_deref(),
                                load.message.as_deref(),
                            )
                            .into())
                        }
                        _ => {}
                    }
                }
                Next::Started(result) => {
                    return result
                        .await
                        .unwrap_or_else(|_| Err(anyhow::anyhow!(MODEL_LOAD_ABORTED)));
                }
            }
        }
    }

    /// Start `handle`'s load of this slot in a task the caller does not own.
    /// Joiners wait on the returned pending load; the starter receives its
    /// lease (taken as the slot becomes loaded) or the original error.
    fn spawn_load(
        self: &Arc<Self>,
        pool: &Arc<EnginePoolInner>,
        handle: LoadHandle,
        trigger: EngineLoadTrigger,
    ) -> (PendingSlotLoad, oneshot::Receiver<Result<EngineLease>>) {
        let (outcome, outcome_receiver) = watch::channel(None);
        let (result_sender, result) = oneshot::channel();
        let load_id = handle.load_id().to_string();
        let slot = self.clone();
        let pool = pool.clone();
        tokio::spawn(async move {
            let tracker = handle.tracker();
            let loaded = slot.run_load(&pool, &tracker, outcome, trigger).await;
            match &loaded {
                Ok(_) => handle.finish_loaded(),
                Err(error) => handle.finish_failed(error),
            }
            let _ = result_sender.send(loaded);
        });
        (
            PendingSlotLoad {
                load_id,
                outcome: outcome_receiver,
            },
            result,
        )
    }

    /// Run one load of this slot to its end and record the result in the
    /// slot. Cancellation is honoured while queued for the load gate and at
    /// each checkpoint until the engine starts; a cancelled load leaves the
    /// slot `Unloaded`, and its memory reservation is released.
    ///
    /// Returns the starter's lease, taken together with `Loaded`: a request
    /// trigger already counts as an active request when the slot becomes
    /// loaded, so the next queued load cannot evict it as idle.
    async fn run_load(
        self: &Arc<Self>,
        pool: &EnginePoolInner,
        tracker: &LoadTracker,
        outcome: watch::Sender<Option<SlotLoadOutcome>>,
        trigger: EngineLoadTrigger,
    ) -> Result<EngineLease> {
        let mut settler = SlotLoadSettler {
            slot: self.clone(),
            outcome: Some(outcome),
        };
        let _load_gate = tokio::select! {
            biased;
            () = tracker.cancelled() => {
                settler.settle_cancelled().await;
                return Err(LoadCancelled.into());
            }
            gate = pool.load_gate.lock() => gate,
        };
        tracker.enter_stage(LoadStage::PreparingMemory);

        let load_reservation = match pool.ensure_capacity_for(&self.model).await {
            Ok(reservation) => reservation,
            Err(error) => {
                let message = format!("{error:#}");
                let load_attempts = self.state.lock().await.load_attempts();
                settler
                    .settle(
                        EngineSlotState::Unloaded {
                            reason: EngineUnloadReason::CapacityRejected,
                            last_error: Some(message),
                            changed_unix_ms: unix_time_ms(),
                            load_attempts,
                        },
                        SlotLoadOutcome::failed(&error),
                    )
                    .await;
                return Err(error);
            }
        };
        #[cfg(any(test, feature = "test-support"))]
        if let Some(hook) = pool.test_load_hook() {
            hook.pause(TestLoadPause::AfterPreparingMemory).await;
        }
        if let Err(error) = tracker.checkpoint() {
            drop(load_reservation);
            settler.settle_cancelled().await;
            return Err(error);
        }

        let load_attempts = {
            let mut state = self.state.lock().await;
            let load_attempts = state.load_attempts().saturating_add(1);
            *state = EngineSlotState::Loading {
                started_unix_ms: unix_time_ms(),
                load_attempts,
            };
            self.notify.notify_waiters();
            load_attempts
        };

        let snapshot_use_lease = match ModelSnapshotUseLease::acquire(&self.model.path) {
            Ok(lease) => lease.map(Arc::new),
            Err(error) => {
                drop(load_reservation);
                settler.settle_failed(load_attempts, &error).await;
                return Err(error);
            }
        };
        #[cfg(any(test, feature = "test-support"))]
        let loaded = match pool.test_load_hook() {
            Some(hook) => hook.load(&self.model, tracker).await,
            None => load_engine_variant(&self.model, &self.runtime, tracker).await,
        };
        #[cfg(not(any(test, feature = "test-support")))]
        let loaded = load_engine_variant(&self.model, &self.runtime, tracker).await;
        // Every engine passes this point before it counts as loaded, so an
        // accepted cancel never ends in a loaded slot.
        let loaded = loaded.and_then(|engine| {
            before_engine_start(tracker)?;
            Ok(engine)
        });
        let engine = match loaded {
            Ok(engine) => engine,
            Err(error) => {
                drop(load_reservation);
                if is_load_cancelled(&error) {
                    settler.settle_cancelled().await;
                } else {
                    settler.settle_failed(load_attempts, &error).await;
                }
                return Err(error);
            }
        };
        if let Err(error) = pool
            .check_memory_limits_after_load(&self.model.id, &engine)
            .await
        {
            drop(load_reservation);
            settler.settle_failed(load_attempts, &error).await;
            return Err(error);
        }
        #[cfg(any(test, feature = "test-support"))]
        diagnostic_model_load_pressure_pulse();
        if let Some(reservation) = load_reservation {
            reservation.commit();
        }
        global_process_memory_governor().sample_process();
        let now = unix_time_ms();
        let engine = Arc::new(engine);
        let active_requests = (trigger == EngineLoadTrigger::Request).then(|| {
            self.active_requests.fetch_add(1, Ordering::SeqCst);
            self.active_requests.clone()
        });
        let lease = EngineLease::new(
            engine.clone(),
            self.model.default_max_output_tokens,
            active_requests,
            snapshot_use_lease.clone(),
        );
        settler
            .settle(
                EngineSlotState::Loaded {
                    engine,
                    snapshot_use_lease,
                    loaded_unix_ms: now,
                    last_used_unix_ms: now,
                    load_attempts,
                    request_count: u64::from(trigger == EngineLoadTrigger::Request),
                },
                SlotLoadOutcome::Loaded,
            )
            .await;
        Ok(lease)
    }

    async fn model_ttl_candidate(&self, id: &str) -> ModelTtlCandidate {
        let snapshot = self.runtime_snapshot().await;
        ModelTtlCandidate {
            id: id.to_string(),
            state: snapshot.state,
            active_requests: self.active_requests.load(Ordering::SeqCst),
            pinned: self.is_pinned(),
            last_used_unix_ms: snapshot.last_used_unix_ms,
        }
    }

    async fn unload_if_ttl_expired(&self, now_unix_ms: u64, ttl: Duration) -> bool {
        if self.model.load_policy == EngineLoadPolicy::Preload
            || self.is_pinned()
            || self.active_requests.load(Ordering::SeqCst) > 0
        {
            return false;
        }
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        let mut state = self.state.lock().await;
        let load_attempts = state.load_attempts();
        let EngineSlotState::Loaded {
            last_used_unix_ms, ..
        } = &*state
        else {
            return false;
        };
        if now_unix_ms.saturating_sub(*last_used_unix_ms) < ttl_ms
            || self.active_requests.load(Ordering::SeqCst) > 0
        {
            return false;
        }
        *state = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Ttl,
            last_error: None,
            changed_unix_ms: now_unix_ms,
            load_attempts,
        };
        crate::core::cache::cancel_process_async_prefix_store_model(&self.model.id);
        self.notify.notify_waiters();
        true
    }

    async fn unload(self: &Arc<Self>) -> Result<()> {
        if self.model.load_policy == EngineLoadPolicy::Preload {
            bail!(
                "engine model `{}` uses preload policy and cannot be unloaded",
                self.model.id
            );
        }
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            bail!("engine model `{}` is disabled", self.model.id);
        }

        let mut state = self.state.lock().await;
        let load_attempts = state.load_attempts();
        match &*state {
            EngineSlotState::Loaded { request_count, .. } => {
                if self.active_requests.load(Ordering::SeqCst) > 0 {
                    *state = EngineSlotState::Draining {
                        started_unix_ms: unix_time_ms(),
                        load_attempts,
                        request_count: *request_count,
                    };
                    self.notify.notify_waiters();
                    self.schedule_finish_draining();
                    return Ok(());
                }
            }
            EngineSlotState::Loading { .. } => {
                bail!("engine model `{}` is currently loading", self.model.id);
            }
            EngineSlotState::Draining { .. } => {
                if self.active_requests.load(Ordering::SeqCst) > 0 {
                    self.schedule_finish_draining();
                    return Ok(());
                }
            }
            EngineSlotState::Unloaded { .. } | EngineSlotState::Failed { .. } => {}
        }
        *state = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Manual,
            last_error: None,
            changed_unix_ms: unix_time_ms(),
            load_attempts,
        };
        crate::core::cache::cancel_process_async_prefix_store_model(&self.model.id);
        self.notify.notify_waiters();
        Ok(())
    }

    fn schedule_finish_draining(self: &Arc<Self>) {
        let slot = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                if slot.active_requests.load(Ordering::SeqCst) == 0 {
                    let mut state = slot.state.lock().await;
                    if matches!(&*state, EngineSlotState::Draining { .. }) {
                        let load_attempts = state.load_attempts();
                        *state = EngineSlotState::Unloaded {
                            reason: EngineUnloadReason::Manual,
                            last_error: None,
                            changed_unix_ms: unix_time_ms(),
                            load_attempts,
                        };
                        crate::core::cache::cancel_process_async_prefix_store_model(&slot.model.id);
                        slot.notify.notify_waiters();
                    }
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    }

    async fn runtime_snapshot(&self) -> EngineSlotRuntimeSnapshot {
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            return EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Disabled,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: 0,
                request_count: 0,
            };
        }
        self.state.lock().await.runtime_snapshot()
    }

    async fn pending_requests(&self) -> Option<usize> {
        let engine = {
            let state = self.state.lock().await;
            match &*state {
                EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                _ => None,
            }
        };
        engine.map(|engine| engine.pending_requests())
    }

    async fn control_result(&self) -> EngineModelControlResult {
        let snapshot = self.runtime_snapshot().await;
        EngineModelControlResult {
            id: self.model.id.clone(),
            load_policy: self.model.load_policy,
            pinned: self.is_pinned(),
            state: snapshot.state,
            unload_reason: snapshot.unload_reason,
            last_error: snapshot.last_error,
            load_attempts: snapshot.load_attempts,
            request_count: snapshot.request_count,
        }
    }

    async fn health_snapshot(&self) -> EngineModelHealth {
        let snapshot = self.runtime_snapshot().await;
        let loaded = if snapshot.state == EngineRuntimeState::Loaded {
            let guard = self.state.lock().await;
            match &*guard {
                EngineSlotState::Loaded { engine, .. } => Some(engine.loaded_health()),
                _ => None,
            }
        } else {
            None
        };
        EngineModelHealth {
            id: self.model.id.clone(),
            load_policy: self.model.load_policy,
            pinned: self.is_pinned(),
            state: snapshot.state,
            unload_reason: snapshot.unload_reason,
            last_error: snapshot.last_error,
            loaded_unix_ms: snapshot.loaded_unix_ms,
            last_used_unix_ms: snapshot.last_used_unix_ms,
            load_started_unix_ms: snapshot.load_started_unix_ms,
            failed_unix_ms: snapshot.failed_unix_ms,
            load_attempts: snapshot.load_attempts,
            request_count: snapshot.request_count,
            loaded,
        }
    }
}

impl EngineVariant {
    fn context_window(&self) -> Option<usize> {
        let cap = match self {
            Self::Qwen35(state) => state.effective_cap_max,
            Self::Qwen35Moe(state) => state.effective_cap_max,
            Self::Qwen36Moe(state) => state.effective_cap_max,
            Self::Gemma4(state) => state.effective_cap_max,
            Self::Gemma4Drafter(state) => state.base.effective_cap_max,
            Self::Glm4MoeLite(state) => state.effective_cap_max,
            Self::Llama(state) => state.effective_cap_max,
            Self::MiniCpmV46(state) => state.effective_cap_max,
            Self::DiffusionGemma(_)
            | Self::QwenImage(_)
            | Self::Audio(_)
            | Self::Asr(_)
            | Self::Alignment(_)
            | Self::Diarization(_)
            | Self::Decision(_)
            | Self::Embedding(_) => return None,
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => return None,
        };
        (cap > 0).then_some(cap)
    }

    async fn clear_shared_prompt_lookup(&self) -> Result<usize> {
        match self {
            Self::Qwen35(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::Qwen35Moe(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::Qwen36Moe(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::Gemma4(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::Gemma4Drafter(state) => {
                state
                    .base
                    .request_execution
                    .clear_shared_prompt_lookup()
                    .await
            }
            Self::Glm4MoeLite(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::Llama(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::MiniCpmV46(state) => state.request_execution.clear_shared_prompt_lookup().await,
            Self::DiffusionGemma(_)
            | Self::QwenImage(_)
            | Self::Audio(_)
            | Self::Asr(_)
            | Self::Alignment(_)
            | Self::Diarization(_)
            | Self::Decision(_)
            | Self::Embedding(_) => Ok(0),
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => Ok(0),
        }
    }

    fn loaded_health(&self) -> LoadedEngineHealth {
        match self {
            Self::Diarization(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Audio {
                    scheduler: "serial_diarization",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::diarization_execution::DIARIZATION_QUEUE_CAPACITY,
                }
            }
            Self::Alignment(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Audio {
                    scheduler: "serial_alignment",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::alignment_execution::ALIGNMENT_QUEUE_CAPACITY,
                }
            }
            Self::Asr(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Audio {
                    scheduler: "serial_asr",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::asr_execution::ASR_QUEUE_CAPACITY,
                }
            }
            Self::Embedding(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Embedding {
                    scheduler: "serial_embedding",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::embedding_execution::EMBEDDING_QUEUE_CAPACITY,
                    metrics: state.metrics(),
                }
            }
            Self::Decision(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Decision {
                    scheduler: "serial_decision",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::decision_execution::DECISION_QUEUE_CAPACITY,
                    metrics: state.metrics(),
                }
            }
            Self::Audio(state) => {
                let (active_requests, queued_requests) = state.active_and_queued();
                LoadedEngineHealth::Audio {
                    scheduler: "serial_audio",
                    active_requests,
                    queued_requests,
                    queue_capacity: super::audio_execution::AUDIO_QUEUE_CAPACITY,
                }
            }
            Self::Qwen35(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Qwen35Moe(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Qwen36Moe(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Gemma4(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Gemma4Drafter(state) => {
                LoadedEngineHealth::Causal(Box::new(state.base.health_collector.snapshot()))
            }
            Self::Glm4MoeLite(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Llama(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::MiniCpmV46(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::DiffusionGemma(state) => {
                let stats = state.lane.stats();
                LoadedEngineHealth::DiffusionGemma {
                    scheduler: "serial_block_diffusion",
                    active_requests: stats.active_requests,
                    queued_requests: stats.queued_requests,
                    queue_capacity: stats.queue_capacity,
                }
            }
            Self::QwenImage(state) => {
                let stats = state.lane.stats();
                LoadedEngineHealth::ImageGeneration {
                    scheduler: "serial_image_generation",
                    active_requests: stats.active_requests,
                    queued_requests: stats.queued_requests,
                    queue_capacity: stats.queue_capacity,
                }
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => LoadedEngineHealth::Audio {
                scheduler: "test",
                active_requests: 0,
                queued_requests: 0,
                queue_capacity: 0,
            },
        }
    }

    fn native_output_dialect(
        &self,
    ) -> Option<ironmlx_lm::core::native_output::NativeOutputDialect> {
        match self {
            Self::Qwen35(state) => state.tokenizer.native_output_dialect(),
            Self::Qwen35Moe(state) => state.tokenizer.native_output_dialect(),
            Self::Qwen36Moe(state) => state.tokenizer.native_output_dialect(),
            Self::Gemma4(state) => state.tokenizer.native_output_dialect(),
            Self::Gemma4Drafter(state) => state.base.tokenizer.native_output_dialect(),
            Self::Glm4MoeLite(state) => state.tokenizer.native_output_dialect(),
            Self::Llama(state) => state.tokenizer.native_output_dialect(),
            Self::MiniCpmV46(state) => state.tokenizer.native_output_dialect(),
            Self::DiffusionGemma(state) => state.tokenizer.native_output_dialect(),
            Self::QwenImage(_)
            | Self::Audio(_)
            | Self::Asr(_)
            | Self::Alignment(_)
            | Self::Diarization(_)
            | Self::Decision(_)
            | Self::Embedding(_) => None,
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => None,
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            Self::Qwen35(_) => "qwen3_5",
            Self::Qwen35Moe(_) => "qwen3_5_moe",
            Self::Qwen36Moe(_) => "qwen3_6_moe",
            Self::Gemma4(_) => "gemma4",
            Self::Gemma4Drafter(_) => "gemma4",
            Self::Glm4MoeLite(_) => "glm4_moe_lite",
            Self::Llama(_) => "llama",
            Self::MiniCpmV46(_) => "minicpmv4_6",
            Self::DiffusionGemma(_) => "diffusion_gemma",
            Self::QwenImage(_) => "qwen_image_2_1",
            Self::Audio(_) => "indextts25",
            Self::Asr(_) => "qwen3_asr",
            Self::Alignment(_) => "qwen3_forced_aligner",
            Self::Diarization(_) => "nemotron_diarization",
            Self::Decision(_) => "laya_multilingual_mlx",
            Self::Embedding(_) => "embedding_gemma2",
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => "test",
        }
    }

    fn model_weight_bytes(&self) -> usize {
        match self {
            Self::Qwen35(state) => state.model_weight_bytes,
            Self::Qwen35Moe(state) => state.model_weight_bytes,
            Self::Qwen36Moe(state) => state.model_weight_bytes,
            Self::Gemma4(state) => state.model_weight_bytes,
            Self::Gemma4Drafter(state) => state.base.model_weight_bytes,
            Self::Glm4MoeLite(state) => state.model_weight_bytes,
            Self::Llama(state) => state.model_weight_bytes,
            Self::MiniCpmV46(state) => state.model_weight_bytes,
            Self::DiffusionGemma(state) => state.model_weight_bytes,
            Self::QwenImage(state) => state.model_weight_bytes,
            Self::Audio(state) => state.model_weight_bytes(),
            Self::Asr(state) => state.model_weight_bytes(),
            Self::Alignment(state) => state.model_weight_bytes(),
            Self::Diarization(state) => state.model_weight_bytes(),
            Self::Decision(state) => state.model_weight_bytes(),
            Self::Embedding(state) => state.model_weight_bytes(),
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(state) => state.model_weight_bytes,
        }
    }

    fn pending_requests(&self) -> usize {
        match self {
            Self::Diarization(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Alignment(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Asr(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Embedding(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Decision(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Audio(state) => {
                let (active, queued) = state.active_and_queued();
                active + queued
            }
            Self::Qwen35(state) => causal_pending_requests(state),
            Self::Qwen35Moe(state) => causal_pending_requests(state),
            Self::Qwen36Moe(state) => causal_pending_requests(state),
            Self::Gemma4(state) => causal_pending_requests(state),
            Self::Gemma4Drafter(state) => causal_pending_requests(&state.base),
            Self::Glm4MoeLite(state) => causal_pending_requests(state),
            Self::Llama(state) => causal_pending_requests(state),
            Self::MiniCpmV46(state) => causal_pending_requests(state),
            Self::DiffusionGemma(state) => {
                let stats = state.lane.stats();
                stats.active_requests.saturating_add(stats.queued_requests)
            }
            Self::QwenImage(state) => {
                let stats = state.lane.stats();
                stats.active_requests.saturating_add(stats.queued_requests)
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => 0,
        }
    }

    fn runtime_usage_snapshot(&self) -> crate::core::runtime_usage::ModelRuntimeUsageSnapshot {
        match self {
            Self::Qwen35(state) => state.runtime_usage.snapshot(true),
            Self::Qwen35Moe(state) => state.runtime_usage.snapshot(true),
            Self::Qwen36Moe(state) => state.runtime_usage.snapshot(true),
            Self::Gemma4(state) => state.runtime_usage.snapshot(true),
            Self::Gemma4Drafter(state) => state.base.runtime_usage.snapshot(true),
            Self::Glm4MoeLite(state) => state.runtime_usage.snapshot(true),
            Self::Llama(state) => state.runtime_usage.snapshot(true),
            Self::MiniCpmV46(state) => state.runtime_usage.snapshot(true),
            Self::DiffusionGemma(state) => state.runtime_usage.snapshot(false),
            Self::QwenImage(state) => state.runtime_usage.snapshot(false),
            Self::Audio(_) => Default::default(),
            Self::Asr(state) => state.usage(),
            Self::Alignment(state) => state.usage(),
            Self::Diarization(state) => state.usage(),
            Self::Decision(state) => state.usage(),
            Self::Embedding(state) => state.usage(),
            #[cfg(any(test, feature = "test-support"))]
            Self::Test(_) => Default::default(),
        }
    }
}

fn causal_pending_requests<M>(state: &AppState<M>) -> usize
where
    M: Model + ironmlx_lm::core::vision::DenseVlMethods + Send + 'static,
{
    let snapshot = state.health_collector.snapshot();
    let model_locked = usize::from(state.model.try_lock().is_err());
    snapshot
        .scheduler
        .b_active
        .saturating_add(snapshot.scheduler.b_queued)
        .saturating_add(model_locked)
}

#[derive(Debug)]
struct ResolvedEngineMtpConfig {
    model_dir: PathBuf,
    draft_tokens: usize,
}

fn resolve_engine_mtp_config(
    model: &EngineModelConfig,
    architecture: ModelArchitecture,
    raw_config: &serde_json::Value,
) -> Result<Option<ResolvedEngineMtpConfig>> {
    let Some(settings) = model.mtp.as_ref() else {
        return Ok(None);
    };
    match architecture {
        ModelArchitecture::Qwen35Dense
        | ModelArchitecture::Qwen35Moe
        | ModelArchitecture::Gemma4 => {}
        _ => bail!(
            "engine model `{}` configures MTP for a non-Qwen/Gemma4 architecture",
            model.id
        ),
    }
    if !settings.model_dir.exists() {
        bail!(
            "engine model `{}` mtp_model_dir must point to a local directory (got '{}')",
            model.id,
            settings.model_dir.display()
        );
    }
    let validation = crate::core::model_validation::validate_mtp_pair(
        &model.path,
        &settings.model_dir,
        settings.draft_tokens,
    )?;
    if !validation.compatible {
        bail!(
            "engine model `{}` MTP validation failed: {}: {}",
            model.id,
            validation.reason_code,
            validation.message
        );
    }
    let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
        raw_config,
        settings
            .draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
    Ok(Some(ResolvedEngineMtpConfig {
        model_dir: settings.model_dir.clone(),
        draft_tokens,
    }))
}

/// Logs how long each phase of a model load took, at the phase boundaries
/// that load-progress stages are reported at.
pub struct LoadStageTimer {
    model: String,
    started: Instant,
    stage_started: Instant,
}

impl LoadStageTimer {
    pub fn start(model: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            model: model.into(),
            started: now,
            stage_started: now,
        }
    }

    /// Log `stage` as finished now and start timing the next stage.
    pub fn finish(&mut self, stage: &'static str) {
        let now = Instant::now();
        tracing::info!(
            model = %self.model,
            stage,
            elapsed_ms = now.duration_since(self.stage_started).as_millis() as u64,
            total_ms = now.duration_since(self.started).as_millis() as u64,
            "model load stage finished"
        );
        self.stage_started = now;
    }
}

/// Directory estimates of the weights this load will read, shown until each
/// loader reports its exact size.
fn plan_weight_progress(model: &EngineModelConfig, tracker: &LoadTracker) {
    if let Ok(bytes) = estimated_weight_bytes_for_roots([model.path.as_path()]) {
        tracker.set_component_estimate(WeightComponent::Main, bytes as u64);
    }
    if let Some(mtp) = &model.mtp {
        if let Ok(bytes) = estimated_weight_bytes_for_roots([mtp.model_dir.as_path()]) {
            tracker.set_component_estimate(WeightComponent::Draft, bytes as u64);
        }
    }
}

/// The main weights are in memory; tokenizer and model construction follow.
fn after_main_weights(tracker: &LoadTracker) {
    tracker.enter_stage(LoadStage::BuildingModel);
}

/// Cancellation point after the model is built, before draft weights are read.
fn before_draft_weights(tracker: &LoadTracker) -> Result<()> {
    tracker.checkpoint()?;
    tracker.enter_stage(LoadStage::ReadingDraftWeights);
    Ok(())
}

/// Last cancellation point: once the scheduler actor starts the load runs to
/// its end.
fn before_engine_start(tracker: &LoadTracker) -> Result<()> {
    tracker.begin_finishing(LoadStage::StartingEngine)
}

async fn load_engine_variant(
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    tracker: &LoadTracker,
) -> Result<EngineVariant> {
    tracker.enter_stage(LoadStage::ReadingWeights);
    if model.capabilities.runtime_kind == "diarization" {
        return Ok(EngineVariant::Diarization(
            super::diarization_execution::DiarizationRuntime::load(model.path.clone()).await?,
        ));
    }
    if model.capabilities.runtime_kind == "alignment" {
        return Ok(EngineVariant::Alignment(
            super::alignment_execution::AlignmentRuntime::load(model.path.clone()).await?,
        ));
    }
    if model.capabilities.runtime_kind == "asr" {
        return Ok(EngineVariant::Asr(
            super::asr_execution::AsrRuntime::load(model.path.clone()).await?,
        ));
    }
    if model.capabilities.runtime_kind == "embedding" {
        return Ok(EngineVariant::Embedding(
            super::embedding_execution::EmbeddingRuntime::load(model.path.clone()).await?,
        ));
    }
    if model.capabilities.runtime_kind == "decision" {
        return Ok(EngineVariant::Decision(
            super::decision_execution::DecisionRuntime::load(
                model.path.clone(),
                model.decision.unwrap_or_default(),
            )
            .await?,
        ));
    }
    if let Some(audio) = &model.audio {
        return Ok(EngineVariant::Audio(
            super::audio_execution::AudioRuntime::load(model.path.clone(), audio.clone()).await?,
        ));
    }
    if model.capabilities.runtime_kind == "image_generation" {
        let model_weight_bytes = estimated_weight_bytes_for_roots([model.path.as_path()])?;
        let pipeline = run_load_section(|| QwenImage21Pipeline::load(&model.path))
            .with_context(|| format!("QwenImage21Pipeline::load {}", model.path.display()))?;
        return Ok(EngineVariant::QwenImage(
            crate::core::qwen_image_execution::QwenImageRuntime::new(
                pipeline,
                model.id.clone(),
                model_weight_bytes,
            ),
        ));
    }
    plan_weight_progress(model, tracker);
    let observer = tracker.observer();
    let loader = run_load_section(|| Loader::open_multimodal_with(&model.path, &observer))
        .with_context(|| format!("Loader::open_multimodal {}", model.path.display()))?;
    after_main_weights(tracker);
    let component_bytes = loader.loaded_tensor_component_bytes();
    let static_memory_estimate =
        StaticMemoryEstimate::from_allocated_weights(component_bytes.text, component_bytes.vision);
    let tokenizer =
        run_load_section(|| Tokenizer::from_loader(&loader)).context("Tokenizer::from_loader")?;
    let model_type = loader
        .config_raw_value()
        .get("model_type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;
    let architecture = ModelArchitecture::from_model_type(model_type)?;
    if architecture == ModelArchitecture::DiffusionGemma {
        let cfg = DiffusionGemmaConfig::from_loader(&loader)
            .context("DiffusionGemmaConfig::from_loader")?;
        let vision_config = cfg
            .vision_config
            .clone()
            .ok_or_else(|| anyhow::anyhow!("DiffusionGemma config has no vision_config"))?;
        let image_token_id = cfg.image_token_id;
        let generation_config = DiffusionGemmaGenerationConfig::from_loader(&loader)
            .context("DiffusionGemmaGenerationConfig::from_loader")?;
        let model_impl = run_load_section(|| DiffusionGemmaModel::from_loader(&loader))
            .context("DiffusionGemmaModel::from_loader")?;
        before_engine_start(tracker)?;
        let state = crate::core::diffusion_execution::build_diffusion_gemma_app_state(
            model_impl,
            tokenizer,
            generation_config,
            model.id.clone(),
            static_memory_estimate.total_cold_bytes(),
            VisionInputConfig::DiffusionGemma {
                vision_config,
                image_token_id,
            },
        );
        return Ok(EngineVariant::DiffusionGemma(state));
    }
    let mtp_config = resolve_engine_mtp_config(model, architecture, loader.config_raw_value())?;
    let prompt_lookup = model
        .prompt_lookup
        .map(PromptLookupConfig::validate)
        .transpose()?;
    let scheduler_config = model
        .scheduler_runtime_profile
        .as_ref()
        .context("causal engine model is missing scheduler runtime profile")?
        .config;
    let paged_prefix_cache = runtime.paged_prefix_cache_config(&model.id, scheduler_config)?;
    let prefix_lru_cache = runtime.prefix_lru_cache_config(paged_prefix_cache.as_ref())?;

    let vision_input = Some(VisionInputConfig::from_causal_loader(
        architecture,
        &loader,
        runtime.max_image_pixels,
    )?);
    match architecture {
        ModelArchitecture::Qwen35Dense => {
            let mut model_impl = run_load_section(|| Qwen35Model::from_loader(&loader))
                .context("Qwen35Model::from_loader")?;
            model_impl.configure_m1_profile(runtime.m1_profile);
            build_qwen35_engine(
                model_impl,
                tokenizer,
                model,
                runtime,
                static_memory_estimate,
                paged_prefix_cache,
                prefix_lru_cache,
                mtp_config,
                prompt_lookup,
                vision_input,
                tracker,
            )
            .await
        }
        ModelArchitecture::Qwen35Moe => {
            if ironmlx_lm::models::is_qwen36_moe_config(loader.config_raw_value()) {
                let model_impl = run_load_section(|| Qwen36MoeModel::from_loader(&loader))
                    .context("Qwen36MoeModel::from_loader")?;
                build_qwen36_moe_engine(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    static_memory_estimate,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    mtp_config,
                    prompt_lookup,
                    vision_input,
                    tracker,
                )
                .await
            } else {
                let model_impl = run_load_section(|| Qwen35MoeModel::from_loader(&loader))
                    .context("Qwen35MoeModel::from_loader")?;
                build_qwen35_moe_engine(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    static_memory_estimate,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    mtp_config,
                    prompt_lookup,
                    vision_input,
                    tracker,
                )
                .await
            }
        }
        ModelArchitecture::Gemma4 => {
            let model_impl = run_load_section(|| Gemma4Model::from_loader(&loader))
                .context("Gemma4Model::from_loader")?;
            if let Some(mtp_config) = mtp_config {
                let state = build_gemma4_drafter_causal_state(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    static_memory_estimate,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    mtp_config,
                    prompt_lookup,
                    vision_input,
                    tracker,
                )
                .await?;
                Ok(EngineVariant::Gemma4Drafter(Box::new(state)))
            } else if let Some(prompt_lookup) = prompt_lookup {
                let state = build_prompt_lookup_causal_state(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    static_memory_estimate,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    prompt_lookup,
                    vision_input,
                    tracker,
                )
                .await?;
                Ok(EngineVariant::Gemma4(state))
            } else {
                let state = build_plain_causal_state(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    static_memory_estimate,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    vision_input,
                    tracker,
                )
                .await?;
                Ok(EngineVariant::Gemma4(state))
            }
        }
        ModelArchitecture::Glm4MoeLite => {
            let model_impl = run_load_section(|| Glm4MoeLiteModel::from_loader(&loader))
                .context("Glm4MoeLiteModel::from_loader")?;
            let state = build_plain_or_prompt_lookup_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                static_memory_estimate,
                paged_prefix_cache,
                prefix_lru_cache,
                prompt_lookup,
                vision_input,
                tracker,
            )
            .await?;
            Ok(EngineVariant::Glm4MoeLite(state))
        }
        ModelArchitecture::Llama => {
            let model_impl = run_load_section(|| LlamaModel::from_loader(&loader))
                .context("LlamaModel::from_loader")?;
            let state = build_plain_or_prompt_lookup_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                static_memory_estimate,
                paged_prefix_cache,
                prefix_lru_cache,
                prompt_lookup,
                vision_input,
                tracker,
            )
            .await?;
            Ok(EngineVariant::Llama(state))
        }
        ModelArchitecture::MiniCpmV46 => {
            let model_impl =
                run_load_section(|| ironmlx_lm::models::minicpmv4_6::model_from_loader(&loader))
                    .context("minicpmv4_6::model_from_loader")?;
            let state = build_plain_or_prompt_lookup_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                static_memory_estimate,
                paged_prefix_cache,
                prefix_lru_cache,
                prompt_lookup,
                vision_input,
                tracker,
            )
            .await?;
            Ok(EngineVariant::MiniCpmV46(state))
        }
        ModelArchitecture::DiffusionGemma => unreachable!("handled before causal runtime setup"),
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen35_engine(
    model_impl: Qwen35Model,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen35(state))
    } else {
        let state = build_plain_or_prompt_lookup_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen35(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen35_moe_engine(
    model_impl: Qwen35MoeModel,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen35Moe(state))
    } else {
        let state = build_plain_or_prompt_lookup_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen35Moe(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen36_moe_engine(
    model_impl: Qwen36MoeModel,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen36Moe(state))
    } else {
        let state = build_plain_or_prompt_lookup_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await?;
        Ok(EngineVariant::Qwen36Moe(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_plain_or_prompt_lookup_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<AppState<M>>
where
    M: Model + ironmlx_lm::core::vision::DenseVlMethods + Send + 'static,
{
    if let Some(prompt_lookup) = prompt_lookup {
        build_prompt_lookup_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            prompt_lookup,
            vision_input,
            tracker,
        )
        .await
    } else {
        build_plain_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            static_memory_estimate,
            paged_prefix_cache,
            prefix_lru_cache,
            vision_input,
            tracker,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_plain_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<AppState<M>>
where
    M: Model + ironmlx_lm::core::vision::DenseVlMethods + Send + 'static,
{
    before_engine_start(tracker)?;
    let profile = model
        .scheduler_runtime_profile
        .clone()
        .context("causal engine model is missing scheduler runtime profile")?;
    let state = crate::core::engine_state::build_plain_app_state(
        model_impl,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

#[allow(clippy::too_many_arguments)]
async fn build_prompt_lookup_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    prompt_lookup: PromptLookupConfig,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<AppState<M>>
where
    M: Model + ironmlx_lm::core::vision::DenseVlMethods + Send + 'static,
{
    before_engine_start(tracker)?;
    let profile = model
        .scheduler_runtime_profile
        .clone()
        .context("causal engine model is missing scheduler runtime profile")?;
    let state = crate::core::engine_state::build_prompt_lookup_app_state(
        model_impl,
        prompt_lookup,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

#[allow(clippy::too_many_arguments)]
async fn build_mtp_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    mut static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: ResolvedEngineMtpConfig,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<AppState<M>>
where
    M: Model + ironmlx_lm::core::vision::DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    before_draft_weights(tracker)?;
    let observer = tracker.observer();
    let mtp_loader =
        run_load_section(|| Loader::open_mtp_with(&mtp_config.model_dir, &observer))
            .with_context(|| format!("Loader::open_mtp {}", mtp_config.model_dir.display()))?;
    static_memory_estimate.set_allocated_speculative_weights(mtp_loader.loaded_tensor_bytes());
    let mtp = run_load_section(|| model_impl.load_mtp_head(&mtp_loader))
        .with_context(|| format!("loading MTP head from {}", mtp_config.model_dir.display()))?;
    before_engine_start(tracker)?;
    let profile = model
        .scheduler_runtime_profile
        .clone()
        .context("causal engine model is missing scheduler runtime profile")?;
    let state = crate::core::engine_state::build_mtp_app_state(
        model_impl,
        mtp,
        mtp_config.draft_tokens,
        prompt_lookup,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

#[allow(clippy::too_many_arguments)]
async fn build_gemma4_drafter_causal_state(
    model_impl: Gemma4Model,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EngineRuntimeOptions,
    mut static_memory_estimate: StaticMemoryEstimate,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: ResolvedEngineMtpConfig,
    prompt_lookup: Option<PromptLookupConfig>,
    vision_input: Option<VisionInputConfig>,
    tracker: &LoadTracker,
) -> Result<Gemma4DrafterAppState> {
    before_draft_weights(tracker)?;
    let observer = tracker.observer();
    let drafter_loader =
        run_load_section(|| Loader::open_gemma4_drafter_with(&mtp_config.model_dir, &observer))
            .with_context(|| {
                format!(
                    "Loader::open_gemma4_drafter {}",
                    mtp_config.model_dir.display()
                )
            })?;
    static_memory_estimate.set_allocated_speculative_weights(drafter_loader.loaded_tensor_bytes());
    let drafter = run_load_section(|| {
        ironmlx_lm::models::gemma4::Gemma4AssistantModel::from_loader(&drafter_loader)
    })
    .with_context(|| {
        format!(
            "loading Gemma4 assistant drafter from {}",
            mtp_config.model_dir.display()
        )
    })?;
    before_engine_start(tracker)?;
    let profile = model
        .scheduler_runtime_profile
        .clone()
        .context("causal engine model is missing scheduler runtime profile")?;
    let state = crate::core::engine_state::build_gemma4_drafter_app_state(
        model_impl,
        drafter,
        mtp_config.draft_tokens,
        prompt_lookup,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

#[derive(Debug, Serialize)]
pub struct EngineModelControlResult {
    pub id: String,
    load_policy: EngineLoadPolicy,
    pinned: bool,
    pub state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    unload_reason: Option<EngineUnloadReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    load_attempts: u64,
    request_count: u64,
}

#[derive(Debug, Serialize)]
pub struct EnginePoolHealth {
    status: health::HealthStatus,
    degraded_reasons: Vec<health::HealthDegradedReason>,
    mode: &'static str,
    default_model: Option<String>,
    max_loaded_models: Option<usize>,
    loaded_models: usize,
    models: Vec<EngineModelHealth>,
    process_governor: crate::core::process_memory::MemoryGovernorSnapshot,
    prefix_store: crate::core::cache::AsyncPrefixStoreStats,
    version: &'static str,
}

#[derive(Debug, Serialize)]
struct EngineModelHealth {
    id: String,
    load_policy: EngineLoadPolicy,
    pinned: bool,
    state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    unload_reason: Option<EngineUnloadReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_started_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed_unix_ms: Option<u64>,
    load_attempts: u64,
    request_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded: Option<LoadedEngineHealth>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LoadedEngineHealth {
    Embedding {
        metrics: super::embedding_execution::EmbeddingMetricsSnapshot,
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
    },
    Decision {
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
        metrics: super::decision_execution::DecisionMetricsSnapshot,
    },
    Audio {
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
    },
    Causal(Box<health::HealthSnapshot>),
    DiffusionGemma {
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
    },
    ImageGeneration {
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
    },
}

impl LoadedEngineHealth {
    fn max_position_embeddings(&self) -> i32 {
        match self {
            Self::Causal(snapshot) => snapshot.model.max_position_embeddings,
            Self::DiffusionGemma { .. }
            | Self::ImageGeneration { .. }
            | Self::Audio { .. }
            | Self::Decision { .. }
            | Self::Embedding { .. } => 0,
        }
    }
}

/// Points where [`TestLoadHook`] can hold a load so a test can act on it.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestLoadPause {
    /// A reload that has not registered in the load registry yet.
    BeforeRegistration,
    AfterPreparingMemory,
    ReadingWeights,
    BeforeDraftWeights,
    BeforeEngineStart,
    AfterEngineStart,
}

/// Replaces weight loading in tests: walks the stages and cancellation points
/// of a real load (through the same stage helpers) without reading weights,
/// and can pause at a [`TestLoadPause`] until released.
#[cfg(any(test, feature = "test-support"))]
pub struct TestLoadHook {
    pause_at: std::sync::Mutex<Option<TestLoadPause>>,
    reached_sender: tokio::sync::mpsc::UnboundedSender<TestLoadPause>,
    reached: Mutex<tokio::sync::mpsc::UnboundedReceiver<TestLoadPause>>,
    release: Notify,
    loads_started: AtomicUsize,
    fail_with: std::sync::Mutex<Option<fn() -> anyhow::Error>>,
    joins: watch::Sender<usize>,
}

#[cfg(any(test, feature = "test-support"))]
impl TestLoadHook {
    fn new() -> Self {
        let (reached_sender, reached) = tokio::sync::mpsc::unbounded_channel();
        Self {
            pause_at: std::sync::Mutex::new(None),
            reached_sender,
            reached: Mutex::new(reached),
            release: Notify::new(),
            loads_started: AtomicUsize::new(0),
            fail_with: std::sync::Mutex::new(None),
            joins: watch::channel(0).0,
        }
    }

    /// Wait until `count` callers in all have joined a slot load another
    /// caller started.
    pub async fn wait_joins(&self, count: usize) {
        let mut joins = self.joins.subscribe();
        tokio::time::timeout(
            Duration::from_secs(10),
            joins.wait_for(|joined| *joined >= count),
        )
        .await
        .expect("callers join the load")
        .expect("hook alive");
    }

    fn note_join(&self) {
        self.joins.send_modify(|joined| *joined += 1);
    }

    /// Fail later loads with this error once they have read the weights.
    pub fn fail_with(&self, error: Option<fn() -> anyhow::Error>) {
        *self.fail_with.lock().unwrap() = error;
    }

    pub fn pause_at(&self, pause: Option<TestLoadPause>) {
        *self.pause_at.lock().unwrap() = pause;
    }

    /// The pause point a load is now held at.
    pub async fn wait_reached(&self) -> TestLoadPause {
        tokio::time::timeout(Duration::from_secs(10), self.reached.lock().await.recv())
            .await
            .expect("load reaches its pause point")
            .expect("hook alive")
    }

    /// Let the held load continue.
    pub fn release(&self) {
        self.release.notify_one();
    }

    /// Loads that got past the load gate and memory preparation.
    pub fn loads_started(&self) -> usize {
        self.loads_started.load(Ordering::SeqCst)
    }

    async fn pause(&self, at: TestLoadPause) {
        let paused = *self.pause_at.lock().unwrap() == Some(at);
        if paused {
            self.reached_sender
                .send(at)
                .expect("test holds the receiver");
            self.release.notified().await;
        }
    }

    async fn load(
        &self,
        model: &EngineModelConfig,
        tracker: &LoadTracker,
    ) -> Result<EngineVariant> {
        self.loads_started.fetch_add(1, Ordering::SeqCst);
        tracker.enter_stage(LoadStage::ReadingWeights);
        tracker.set_component_estimate(WeightComponent::Main, 1024);
        if model.mtp.is_some() {
            tracker.set_component_estimate(WeightComponent::Draft, 128);
        }
        let observer = tracker.observer();
        observer.begin_weights(WeightComponent::Main, 1000);
        observer.add_weight_bytes(500);
        self.pause(TestLoadPause::ReadingWeights).await;
        let fail_with = *self.fail_with.lock().unwrap();
        if let Some(error) = fail_with {
            return Err(error());
        }
        // The loader checks for cancellation after every weight chunk.
        tracker.checkpoint()?;
        observer.add_weight_bytes(500);
        after_main_weights(tracker);
        if model.mtp.is_some() {
            self.pause(TestLoadPause::BeforeDraftWeights).await;
            before_draft_weights(tracker)?;
            observer.begin_weights(WeightComponent::Draft, 100);
            observer.add_weight_bytes(100);
            tracker.checkpoint()?;
        }
        self.pause(TestLoadPause::BeforeEngineStart).await;
        before_engine_start(tracker)?;
        self.pause(TestLoadPause::AfterEngineStart).await;
        Ok(EngineVariant::Test(TestEngine {
            model_weight_bytes: 1000,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::EngineModelCapabilities;
    use ironmlx_lm::models::ModelArchitecture;
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::core::runtime_config::SamplingDefaults;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
        SchedulerAutotuneRuntimeProfileMetadata, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    use super::{
        decide_engine_pool_capacity, estimated_engine_weight_bytes, estimated_model_weight_bytes,
        unix_time_ms, validate_engine_model_config, EngineLoadPolicy, EngineLoadTrigger,
        EngineModelConfig, EngineModelManifest, EngineMtpSettings, EnginePoolCapacityDecision,
        EnginePoolCapacityPolicy, EnginePoolConfig, EnginePoolManifest, EnginePoolMemoryLimits,
        EnginePoolState, EngineRegistry, EngineRegistryError, EngineRuntimeOptions,
        EngineRuntimeState, EngineSlot, EngineSlotState, ModelSnapshotUseLease, ModelTtlCandidate,
    };
    use tokio::sync::{Mutex, Notify};

    fn model(id: &str) -> EngineModelManifest {
        EngineModelManifest {
            audio: None,
            decision: None,
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}")),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            scheduler_profile: None,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: None,
        }
    }

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 512,
            admission_deadline_ms: 5,
            admission_queue_max: 4,
            max_cache_cap: 1024,
            decode_cadence_mid_chunk_cap: 128,
        }
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            runtime_context:
                crate::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(
                    1024,
                ),
            config: profile_config(),
            rules: Vec::new(),
            metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
        }
    }

    fn runtime_config() -> EngineRuntimeOptions {
        EngineRuntimeOptions {
            m1_profile: ironmlx_core::m1_profile::M1ProfileMode::Off,
            kv_cache_turboquant_bits: None,
            scheduler_autotune_report: false,
            paged_prefix_cache: None,
            prefix_lru_cache_max_bytes: None,
            max_image_pixels: None,
            model_ttl: None,
            memory_limits: EnginePoolMemoryLimits::default(),
            active_kv_offload: crate::core::cache::ActiveKvOffloadConfig::disabled(),
        }
    }

    fn model_config(id: &str, path: &Path, load_policy: EngineLoadPolicy) -> EngineModelConfig {
        EngineModelConfig {
            audio: None,
            decision: None,
            id: id.to_string(),
            path: path.to_path_buf(),
            load_policy,
            default: false,
            pinned: false,
            scheduler_runtime_profile: Some(runtime_profile()),
            scheduler_profile_kind: None,
            mtp: None,
            prompt_lookup: None,
            sampling_defaults: SamplingDefaults::default(),
            default_max_output_tokens: None,
            capabilities: EngineModelCapabilities::for_architecture(
                ModelArchitecture::Qwen35Dense,
                false,
            ),
        }
    }

    fn write_minimal_model_config(model_type: &str) -> PathBuf {
        // Wall-clock nanoseconds alone collide between parallel tests (macOS
        // reports microseconds), letting one test overwrite another's model.
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-engine-pool-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .expect("write config");
        dir
    }

    #[test]
    fn recovered_pressure_does_not_evict_but_unrecovered_pressure_does() {
        use crate::core::process_memory::{
            HostVmStatistics, MemoryGovernorConfig, MemoryTelemetry, PressureLevel,
            ProcessMemoryGovernor,
        };
        let gib = |x: usize| x * 1024 * 1024 * 1024;
        let config = MemoryGovernorConfig::default();
        let governor = ProcessMemoryGovernor::new(config).unwrap();
        let sample = |used| MemoryTelemetry {
            total_ram_bytes: gib(32),
            phys_footprint_bytes: Some(gib(used)),
            mlx_active_bytes: Some(gib(used)),
            mlx_cache_bytes: Some(0),
            metal_limit_bytes: Some(gib(24)),
            vm: Some(HostVmStatistics {
                free_bytes: gib(8),
                inactive_bytes: gib(2),
                active_bytes: gib(4),
                wired_bytes: gib(6),
            }),
        };
        let peak = governor.update(sample(28));
        assert_eq!(peak.pressure_level, PressureLevel::Emergency);
        assert!(super::idle_eviction_needed(peak, config));
        let not_recovered = governor.update(sample(21));
        assert!(
            super::idle_eviction_needed(not_recovered, config),
            "hysteresis must still reclaim above recovery watermark"
        );
        let recovered = governor.update(sample(14));
        println!(
            "RECOVERY_TRACE peak={peak:?} unrecovered={not_recovered:?} recovered={recovered:?}"
        );
        assert_eq!(recovered.pressure_level, PressureLevel::Emergency);
        assert!(!super::idle_eviction_needed(recovered, config), "fresh usage below recovery watermark must not evict an idle model merely because the pressure grade is cooling down");
    }

    #[test]
    fn audio_load_estimate_covers_auxiliary_weights_and_frontend() {
        let root = write_minimal_model_config("unused_for_audio");
        let derived = root.join("derived");
        std::fs::create_dir_all(&derived).unwrap();
        std::fs::write(root.join("model.safetensors"), [0; 17]).unwrap();
        std::fs::write(derived.join("auxiliary.safetensors"), [0; 23]).unwrap();
        let mut config = model_config("tts", &root, EngineLoadPolicy::Lazy);
        config.audio = Some(super::super::audio_execution::AudioModelResources {
            derived_resources: derived,
            resource_lock: root.join("lock.json"),
            wetext_fsts: root.join("fsts"),
            unidic_dir: root.join("unidic"),
            execution: Default::default(),
        });
        config.scheduler_runtime_profile = None;
        config.capabilities = EngineModelCapabilities::audio();
        assert_eq!(
            estimated_engine_weight_bytes(&config).unwrap(),
            40 + 1024 * 1024 * 1024
        );
        assert!(validate_engine_model_config(&config).is_ok());
        config.sampling_defaults.temperature = Some(0.5);
        assert!(validate_engine_model_config(&config).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn idle_eviction_preserves_recovery_hysteresis_reservations_and_fail_closed() {
        use crate::core::process_memory::{
            MemoryGovernorConfig, MemoryGovernorSnapshot, PressureLevel,
        };
        let config = MemoryGovernorConfig::default();
        let gib = 1024 * 1024 * 1024;
        let base = MemoryGovernorSnapshot {
            pressure_level: PressureLevel::Hard,
            current_usage_bytes: 14 * gib,
            effective_ceiling_bytes: 24 * gib,
            telemetry_degraded: false,
            ..Default::default()
        };
        let watermark = (base.effective_ceiling_bytes as f64 * config.recovery_ratio) as usize;
        assert!(!super::idle_eviction_needed(base, config));
        assert!(!super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                current_usage_bytes: watermark,
                ..base
            },
            config
        ));
        assert!(super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                current_usage_bytes: watermark + 1,
                ..base
            },
            config
        ));
        assert!(super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                reserved_bytes: 8 * gib,
                ..base
            },
            config
        ));
        assert!(!super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                reserved_bytes: 8 * gib,
                model_load_reserved_bytes: 8 * gib,
                ..base
            },
            config
        ));
        assert!(super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                telemetry_degraded: true,
                ..base
            },
            config
        ));
        assert!(super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                effective_ceiling_bytes: 0,
                ..base
            },
            config
        ));
        assert!(!super::idle_eviction_needed(
            MemoryGovernorSnapshot {
                pressure_level: PressureLevel::Soft,
                current_usage_bytes: 23 * gib,
                ..base
            },
            config
        ));
        println!("EVICTION_BOUNDARIES recovery={watermark} equality=keep above=evict general_reservation=evict pending_load_subset=keep degraded=fail_closed");
    }

    #[test]
    fn snapshot_use_lease_blocks_exclusive_cleanup_until_released() {
        let root = write_minimal_model_config("qwen3_5");
        std::fs::remove_dir_all(&root).expect("reset temp root");
        let commit = "a".repeat(40);
        let snapshot = root
            .join("huggingface")
            .join("org--model")
            .join("snapshots")
            .join(&commit);
        std::fs::create_dir_all(&snapshot).expect("create snapshot");

        let lease = ModelSnapshotUseLease::acquire(&snapshot)
            .expect("acquire snapshot use lease")
            .expect("new layout must use a lease");
        let lock_path = snapshot
            .parent()
            .expect("snapshots")
            .parent()
            .expect("repository")
            .join(".locks")
            .join(format!("{commit}.use.lock"));
        let contender = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .expect("open contender");
        // SAFETY: `contender` owns a valid descriptor for the duration of this check.
        let blocked = unsafe { libc::flock(contender.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_ne!(blocked, 0);

        drop(lease);
        // SAFETY: `contender` remains open until the end of the test.
        let acquired = unsafe { libc::flock(contender.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(acquired, 0);
        // SAFETY: `contender` still owns the locked descriptor.
        unsafe {
            libc::flock(contender.as_raw_fd(), libc::LOCK_UN);
        }
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn engine_model_accepts_neural_and_prompt_lookup_sources_together() {
        let model_dir = write_minimal_model_config("qwen3_5");
        let mut config = model_config("dual-source", &model_dir, EngineLoadPolicy::Lazy);
        config.mtp = Some(EngineMtpSettings {
            model_dir: model_dir.clone(),
            draft_tokens: Some(2),
        });
        config.prompt_lookup = Some(crate::core::prompt_lookup::PromptLookupConfig::default());

        validate_engine_model_config(&config).expect("hybrid sources are valid");

        std::fs::remove_dir_all(model_dir).expect("cleanup");
    }

    #[test]
    #[cfg(unix)]
    fn model_weight_estimate_follows_hugging_face_snapshot_symlinks() {
        use std::os::unix::fs::symlink;

        let model_dir = write_minimal_model_config("qwen3_5");
        let blob = model_dir.join("weight-blob");
        std::fs::write(&blob, vec![0_u8; 4096]).expect("write weight blob");
        symlink(&blob, model_dir.join("model.safetensors")).expect("symlink weight shard");

        assert_eq!(estimated_model_weight_bytes(&model_dir).unwrap(), 4096);
    }

    #[test]
    fn engine_weight_estimate_includes_mtp_weights_without_double_counting_roots() {
        let base_dir = write_minimal_model_config("qwen3_5");
        let mtp_dir = write_minimal_model_config("qwen3_5_mtp");
        std::fs::write(base_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("write base weights");
        std::fs::write(mtp_dir.join("model.safetensors"), vec![0_u8; 1024])
            .expect("write mtp weights");
        let mut config = model_config("qwen", &base_dir, EngineLoadPolicy::Lazy);
        config.mtp = Some(EngineMtpSettings {
            model_dir: mtp_dir.clone(),
            draft_tokens: Some(2),
        });

        assert_eq!(estimated_engine_weight_bytes(&config).unwrap(), 5120);

        config.mtp = Some(EngineMtpSettings {
            model_dir: base_dir.clone(),
            draft_tokens: Some(2),
        });
        assert_eq!(estimated_engine_weight_bytes(&config).unwrap(), 4096);

        std::fs::remove_dir_all(base_dir).ok();
        std::fs::remove_dir_all(mtp_dir).ok();
    }

    #[test]
    fn registry_rejects_duplicate_model_ids() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), model("alpha")],
        };

        let err = EngineRegistry::new(manifest).expect_err("duplicate ids must be rejected");

        assert!(matches!(
            err,
            EngineRegistryError::DuplicateModelId { id } if id == "alpha"
        ));
    }

    #[test]
    fn registry_uses_single_enabled_model_as_implicit_default() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha")],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        assert_eq!(registry.resolve_model_id(None).expect("default"), "alpha");
        assert_eq!(
            registry.resolve_model_id(Some("alpha")).expect("explicit"),
            "alpha"
        );
    }

    #[test]
    fn registry_requires_default_when_multiple_models_are_enabled() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), model("beta")],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let err = registry
            .resolve_model_id(None)
            .expect_err("ambiguous default must fail");

        assert!(matches!(err, EngineRegistryError::AmbiguousDefault));
    }

    #[test]
    fn registry_routes_missing_request_to_declared_default_model() {
        let mut beta = model("beta");
        beta.default = true;
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        assert_eq!(registry.resolve_model_id(None).expect("default"), "beta");
    }

    #[test]
    fn registry_rejects_unknown_and_disabled_models() {
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let missing = registry
            .resolve_model_id(Some("missing"))
            .expect_err("unknown model");
        let disabled = registry
            .resolve_model_id(Some("beta"))
            .expect_err("disabled model");

        assert!(matches!(
            missing,
            EngineRegistryError::UnknownModel { id } if id == "missing"
        ));
        assert!(matches!(
            disabled,
            EngineRegistryError::ModelDisabled { id } if id == "beta"
        ));
    }

    #[test]
    fn registry_openai_model_list_excludes_disabled_models() {
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let ids: Vec<_> = registry
            .servable_models()
            .iter()
            .map(|model| model.id.as_str())
            .collect();

        assert_eq!(ids, vec!["alpha"]);
    }

    #[test]
    fn registry_rejects_manifest_with_no_enabled_models() {
        let mut alpha = model("alpha");
        alpha.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![alpha],
        };

        let err = EngineRegistry::new(manifest).expect_err("no enabled models must fail");

        assert!(matches!(err, EngineRegistryError::NoEnabledModels));
    }

    #[test]
    fn registry_rejects_zero_engine_pool_capacity() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: Some(0),
            models: vec![model("alpha")],
        };

        let err = EngineRegistry::new(manifest).expect_err("zero capacity must fail");

        assert!(matches!(err, EngineRegistryError::InvalidMaxLoadedModels));
    }

    #[test]
    fn registry_rejects_preload_count_above_capacity() {
        let mut alpha = model("alpha");
        alpha.load_policy = EngineLoadPolicy::Preload;
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Preload;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: Some(1),
            models: vec![alpha, beta],
        };

        let err = EngineRegistry::new(manifest).expect_err("preload capacity must fail");

        assert!(matches!(
            err,
            EngineRegistryError::PreloadCapacityExceeded {
                preload_count: 2,
                max_loaded_models: 1
            }
        ));
    }

    #[test]
    fn registry_dynamic_upsert_does_not_replace_existing_default_without_explicit_request() {
        let mut registry = EngineRegistry::empty(None).expect("registry");

        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry.upsert_model(model("beta"), false).expect("beta");

        assert_eq!(registry.default_model(), Some("alpha"));
        assert!(registry.model("alpha").expect("alpha").default);
        assert!(!registry.model("beta").expect("beta").default);
    }

    #[test]
    fn registry_dynamic_upsert_preserves_current_default_when_replacing_same_model() {
        let mut registry = EngineRegistry::empty(None).expect("registry");

        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry
            .upsert_model(model("alpha"), false)
            .expect("replace alpha");

        assert_eq!(registry.default_model(), Some("alpha"));
        assert!(registry.model("alpha").expect("alpha").default);
    }

    #[test]
    fn dynamic_engine_pool_rejects_capacity_instead_of_evicting_models() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");

        assert_eq!(pool.inner.capacity_policy, EnginePoolCapacityPolicy::Reject);
    }

    #[test]
    fn dynamic_engine_pool_capacity_decision_rejects_when_full() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::Reject, 3, 3),
            EnginePoolCapacityDecision::Reject
        );
    }

    #[test]
    fn manifest_engine_pool_capacity_decision_keeps_lru_eviction_when_full() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::EvictLruIdle, 3, 3),
            EnginePoolCapacityDecision::TryEvictLruIdle
        );
    }

    #[test]
    fn engine_pool_capacity_decision_continues_when_below_capacity() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::Reject, 3, 2),
            EnginePoolCapacityDecision::Continue
        );
    }

    #[test]
    fn engine_pool_model_memory_limit_rejects_weight_sum_above_limit() {
        let limits = super::EnginePoolMemoryLimits {
            total_memory_limit_bytes: None,
            model_memory_limit_bytes: Some(10),
        };

        let error = limits
            .check_model_memory_limit("alpha", 11)
            .expect_err("model bytes above limit must fail");

        assert!(format!("{error:#}").contains("engine pool model memory limit exceeded"));
    }

    #[test]
    fn engine_pool_total_memory_limit_rejects_active_bytes_above_limit() {
        let limits = super::EnginePoolMemoryLimits {
            total_memory_limit_bytes: Some(10),
            model_memory_limit_bytes: None,
        };

        let error = limits
            .check_total_memory_limit("alpha", 11)
            .expect_err("active bytes above limit must fail");

        assert!(format!("{error:#}").contains("engine pool total memory limit exceeded"));
    }

    #[tokio::test]
    async fn engine_pool_memory_limits_reject_before_materialization() {
        for (model_limit, total_limit, draft_bytes, expected_error) in [
            (
                Some(511),
                None,
                0,
                "engine pool model memory limit exceeded",
            ),
            (
                None,
                Some(511),
                0,
                "engine pool total memory limit exceeded",
            ),
            (
                Some(768),
                None,
                512,
                "engine pool model memory limit exceeded",
            ),
        ] {
            let mut runtime = runtime_config();
            runtime.memory_limits = EnginePoolMemoryLimits {
                model_memory_limit_bytes: model_limit,
                total_memory_limit_bytes: total_limit,
            };
            let pool = EnginePoolState::new_dynamic(runtime, None).expect("pool");
            let model_dir = write_minimal_model_config("qwen3_5");
            // Deliberately invalid weights: reaching the loader would produce
            // a format error instead of the expected admission rejection.
            std::fs::write(model_dir.join("model.safetensors"), vec![0_u8; 512])
                .expect("write weight size fixture");
            let mut config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
            let draft_dir = (draft_bytes > 0).then(|| {
                let dir = write_minimal_model_config("qwen3_5_mtp");
                std::fs::write(dir.join("model.safetensors"), vec![0_u8; draft_bytes])
                    .expect("write draft size fixture");
                config.mtp = Some(EngineMtpSettings {
                    model_dir: dir.clone(),
                    draft_tokens: Some(2),
                });
                dir
            });
            pool.register_dynamic_model(config, false, None)
                .await
                .expect("register model");

            let error = pool.load_model("alpha").await.expect_err("reject load");
            assert!(format!("{error:#}").contains(expected_error));
            let slot = pool.inner.slots.lock().await["alpha"].clone();
            let health = slot.health_snapshot().await;
            assert_eq!(health.state, EngineRuntimeState::Unloaded);
            assert_eq!(health.load_attempts, 0);
            assert!(pool.loaded_model_infos().await.is_empty());

            std::fs::remove_dir_all(model_dir).expect("cleanup model fixture");
            if let Some(dir) = draft_dir {
                std::fs::remove_dir_all(dir).expect("cleanup draft fixture");
            }
        }
    }

    #[tokio::test]
    async fn dynamic_register_adds_lazy_unloaded_model_without_loading() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);

        let result = pool
            .register_dynamic_model(config, false, None)
            .await
            .expect("register dynamic model");

        assert_eq!(result.id, "alpha");
        assert_eq!(result.state, EngineRuntimeState::Unloaded);
        assert_eq!(result.load_attempts, 0);
        assert!(pool.loaded_model_infos().await.is_empty());
        assert_eq!(pool.inner.registry.lock().await.default_model(), None);
        assert!(
            !pool
                .inner
                .registry
                .lock()
                .await
                .model("alpha")
                .expect("registered model")
                .default
        );

        let list = pool.model_snapshots().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "alpha");
        assert_eq!(list[0].state, EngineRuntimeState::Unloaded);
    }

    #[tokio::test]
    async fn discovery_capacity_is_cached_and_refreshed_on_registration() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).unwrap();
        let model_dir = write_minimal_model_config("qwen3_5");
        std::fs::write(
            model_dir.join("config.json"),
            r#"{
            "model_type":"qwen3_5", "text_config":{"max_position_embeddings":8192}
        }"#,
        )
        .unwrap();
        let mut config = model_config("capacity", &model_dir, EngineLoadPolicy::Lazy);
        config.default_max_output_tokens = Some(256);
        pool.register_dynamic_model(config.clone(), false, None)
            .await
            .unwrap();
        let list = pool.model_snapshots().await;
        assert_eq!(list[0].context_window, Some(1024));
        assert_eq!(list[0].max_output_tokens, Some(256));
        assert_eq!(list[0].load_attempts, 0);

        // Discovery does not reread files or load weights on each request.
        std::fs::write(
            model_dir.join("config.json"),
            r#"{
            "model_type":"qwen3_5", "text_config":{"max_position_embeddings":4096}
        }"#,
        )
        .unwrap();
        assert_eq!(pool.model_snapshots().await[0].context_window, Some(1024));
        config
            .scheduler_runtime_profile
            .as_mut()
            .unwrap()
            .config
            .max_cache_cap = 16384;
        config.default_max_output_tokens = Some(1024);
        pool.register_dynamic_model(config, false, None)
            .await
            .unwrap();
        let list = pool.model_snapshots().await;
        assert_eq!(list[0].context_window, Some(4096));
        assert_eq!(list[0].max_output_tokens, Some(1024));
        assert_eq!(list[0].state, EngineRuntimeState::Unloaded);
        std::fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn discovery_capacity_requires_readable_metadata_and_a_causal_profile() {
        let model_dir = write_minimal_model_config("qwen3_5");
        let mut config = model_config("capacity", &model_dir, EngineLoadPolicy::Lazy);
        let capacity = crate::core::model_capacity::configured_context_window;
        assert_eq!(capacity(&config), None);
        std::fs::write(
            model_dir.join("config.json"),
            r#"{
            "model_type":"qwen3_5", "text_config":{"max_position_embeddings":8192}
        }"#,
        )
        .unwrap();
        assert_eq!(capacity(&config), Some(1024));
        config.capabilities = EngineModelCapabilities::audio();
        assert_eq!(capacity(&config), None);
        config.capabilities =
            EngineModelCapabilities::for_architecture(ModelArchitecture::DiffusionGemma, false);
        assert_eq!(capacity(&config), None);
        config.capabilities =
            EngineModelCapabilities::for_architecture(ModelArchitecture::Qwen35Dense, false);
        config.scheduler_runtime_profile = None;
        assert_eq!(capacity(&config), None);
        config.scheduler_runtime_profile = Some(runtime_profile());
        std::fs::write(model_dir.join("config.json"), "invalid json").unwrap();
        assert_eq!(capacity(&config), None);
        std::fs::remove_dir_all(model_dir).unwrap();
        assert_eq!(capacity(&config), None);
    }

    #[tokio::test]
    async fn discovery_registration_preserves_pins_during_restore() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
        pool.register_dynamic_model(config.clone(), false, Some(true))
            .await
            .expect("register pinned model for restore");

        // Discovery must preserve intent both before and during loading.
        pool.register_dynamic_model(config.clone(), false, None)
            .await
            .expect("discover unloaded model");
        let slot = pool.inner.slots.lock().await["alpha"].clone();
        assert!(slot.is_pinned());
        *slot.state.lock().await = EngineSlotState::Loading {
            started_unix_ms: unix_time_ms(),
            load_attempts: 1,
        };
        pool.register_dynamic_model(config.clone(), false, None)
            .await
            .expect("stale discovery during restore");
        assert!(slot.is_pinned());
        assert!(std::sync::Arc::ptr_eq(
            &slot,
            &pool.inner.slots.lock().await["alpha"]
        ));

        // Explicit changes still work, and a later discovery cannot undo them.
        pool.register_dynamic_model(config.clone(), false, Some(false))
            .await
            .expect("explicit unpin");
        assert!(!slot.is_pinned());
        pool.register_dynamic_model(config.clone(), false, None)
            .await
            .expect("discover unpinned model");
        assert!(!slot.is_pinned());
        pool.register_dynamic_model(config, false, Some(true))
            .await
            .expect("explicit pin");
        assert!(slot.is_pinned());
        std::fs::remove_dir_all(model_dir).expect("cleanup model fixture");
    }

    #[tokio::test]
    async fn dynamic_unload_keeps_model_registered_for_lazy_reload() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
        pool.register_dynamic_model(config, true, None)
            .await
            .expect("register dynamic model");

        let result = pool
            .unload_dynamic_model("alpha")
            .await
            .expect("unload dynamic model");

        assert_eq!(result.id, "alpha");
        assert_eq!(result.state, EngineRuntimeState::Unloaded);
        assert_eq!(
            pool.inner
                .registry
                .lock()
                .await
                .resolve_model_id(Some("alpha"))
                .expect("registered model"),
            "alpha"
        );
    }

    #[tokio::test]
    async fn draining_model_rejects_new_request_until_unload_finishes() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let slot = std::sync::Arc::new(EngineSlot {
            configured_context_window: None,
            model: model_config("alpha", &model_dir, EngineLoadPolicy::Lazy),
            runtime: runtime_config(),
            active_requests: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            pinned: std::sync::atomic::AtomicBool::new(false),
            state: Mutex::new(EngineSlotState::Draining {
                started_unix_ms: unix_time_ms(),
                load_attempts: 1,
                request_count: 1,
            }),
            notify: Notify::new(),
            pending_load: std::sync::Mutex::new(None),
        });

        let error = slot
            .ensure_loaded(
                &pool.inner,
                EngineLoadTrigger::Request,
                super::EngineLoadOptions::new(super::LoadTriggerKind::Request),
            )
            .await
            .expect_err("draining model must reject new requests");

        assert!(format!("{error:#}").contains("currently unloading"));
    }

    #[test]
    fn registry_restore_default_model_preserves_previous_default_after_failed_dynamic_load() {
        let mut registry = EngineRegistry::empty(None).expect("registry");
        registry.upsert_model(model("alpha"), true).expect("alpha");
        let previous_default = registry.default_model().map(str::to_string);

        registry.upsert_model(model("beta"), true).expect("beta");
        assert_eq!(registry.default_model(), Some("beta"));

        registry.remove_model("beta");
        registry
            .restore_default_model(previous_default)
            .expect("restore default");

        assert_eq!(registry.default_model(), Some("alpha"));
    }

    #[test]
    fn registry_restore_default_model_can_restore_ambiguous_no_default_state() {
        let mut registry = EngineRegistry::empty(None).expect("registry");
        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry.restore_default_model(None).expect("clear default");
        registry.upsert_model(model("beta"), true).expect("beta");
        assert_eq!(registry.default_model(), Some("beta"));

        registry.remove_model("beta");
        registry
            .restore_default_model(None)
            .expect("restore no default");

        assert_eq!(registry.default_model(), None);
        assert_eq!(
            registry.resolve_model_id(None).expect("single model"),
            "alpha"
        );
    }

    #[test]
    fn eviction_candidate_selection_uses_lru_order() {
        let chosen = super::select_lru_eviction_candidate(vec![
            super::EvictionCandidate {
                id: "recent".to_string(),
                last_used_unix_ms: 30,
            },
            super::EvictionCandidate {
                id: "oldest".to_string(),
                last_used_unix_ms: 10,
            },
            super::EvictionCandidate {
                id: "middle".to_string(),
                last_used_unix_ms: 20,
            },
        ]);

        assert_eq!(chosen.as_deref(), Some("oldest"));
    }

    #[test]
    fn model_ttl_candidates_require_loaded_idle_expired_models() {
        let candidates = super::select_model_ttl_unload_candidates(
            vec![
                ModelTtlCandidate {
                    id: "expired".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 0,
                    pinned: false,
                    last_used_unix_ms: Some(1_000),
                },
                ModelTtlCandidate {
                    id: "active".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 1,
                    pinned: false,
                    last_used_unix_ms: Some(1_000),
                },
                ModelTtlCandidate {
                    id: "pinned".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 0,
                    pinned: true,
                    last_used_unix_ms: Some(1_000),
                },
                ModelTtlCandidate {
                    id: "fresh".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 0,
                    pinned: false,
                    last_used_unix_ms: Some(9_000),
                },
                ModelTtlCandidate {
                    id: "unloaded".to_string(),
                    state: EngineRuntimeState::Unloaded,
                    active_requests: 0,
                    pinned: false,
                    last_used_unix_ms: Some(1_000),
                },
            ],
            10_000,
            Duration::from_secs(5),
        );

        assert_eq!(candidates, vec!["expired"]);
    }

    #[test]
    fn model_ttl_unload_reason_serializes_as_ttl() {
        let value = serde_json::to_value(super::EngineUnloadReason::Ttl).expect("ttl reason");

        assert_eq!(value, serde_json::json!("ttl"));
    }

    #[tokio::test]
    async fn engine_pool_state_accepts_gemma4_unified_model_type_before_lazy_load() {
        let model_dir = write_minimal_model_config("gemma4_unified");
        let config = EnginePoolConfig {
            default_model: Some("gemma4-unified".to_string()),
            max_loaded_models: Some(1),
            models: vec![model_config(
                "gemma4-unified",
                &model_dir,
                EngineLoadPolicy::Lazy,
            )],
        };

        EnginePoolState::new(config, runtime_config())
            .await
            .expect("gemma4_unified lazy model type should be accepted");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires QWEN35_MODEL pointing to a real checkpoint"]
    #[serial_test::serial(mlx_metal)]
    async fn real_engine_pool_pressure_path_evicts_idle_lru_before_second_model_load() {
        let model_dir = PathBuf::from(
            std::env::var("QWEN35_MODEL")
                .expect("QWEN35_MODEL must point to a real Qwen3.5 checkpoint"),
        );
        let config = EnginePoolConfig {
            default_model: Some("alpha".to_string()),
            max_loaded_models: Some(1),
            models: vec![
                model_config("alpha", &model_dir, EngineLoadPolicy::Lazy),
                model_config("beta", &model_dir, EngineLoadPolicy::Lazy),
            ],
        };
        let pool = EnginePoolState::new(config, runtime_config())
            .await
            .expect("engine pool");

        assert_eq!(
            pool.load_model("alpha").await.expect("load alpha").state,
            EngineRuntimeState::Loaded
        );
        assert_eq!(
            pool.load_model("beta").await.expect("load beta").state,
            EngineRuntimeState::Loaded
        );

        let models = pool.model_snapshots().await;
        let alpha = models
            .iter()
            .find(|model| model.id == "alpha")
            .expect("alpha state");
        let beta = models
            .iter()
            .find(|model| model.id == "beta")
            .expect("beta state");
        assert_eq!(alpha.state, EngineRuntimeState::Unloaded);
        assert_eq!(
            alpha.unload_reason,
            Some(super::EngineUnloadReason::Evicted)
        );
        assert_eq!(beta.state, EngineRuntimeState::Loaded);
    }

    mod load_lifecycle {
        use super::{model_config, runtime_config, write_minimal_model_config};
        use crate::core::engine_pool::{
            EngineLoadOptions, EngineLoadPolicy, EngineModelConfig, EngineMtpSettings,
            EnginePoolState, EngineRuntimeState, EngineSlotRuntimeSnapshot, EngineUnloadReason,
            EngineVariant, TestLoadHook, TestLoadPause,
        };
        use crate::core::load_registry::{
            LoadActivity, LoadCancelError, LoadCancelled, LoadInProgress, LoadKind, LoadStage,
            LoadState, LoadTriggerKind,
        };
        use crate::core::process_memory::global_process_memory_governor;
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::time::Duration;

        const WAIT: Duration = Duration::from_secs(10);

        struct Fixture {
            pool: EnginePoolState,
            hook: Arc<TestLoadHook>,
            config: EngineModelConfig,
            dirs: Vec<PathBuf>,
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                for dir in &self.dirs {
                    let _ = std::fs::remove_dir_all(dir);
                }
            }
        }

        /// A registered, unloaded model "alpha" whose loads run the test load
        /// hook instead of reading real weights.
        async fn fixture(mtp: bool) -> Fixture {
            let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
            let hook = pool.install_test_load_hook();
            let model_dir = write_minimal_model_config("qwen3_5");
            std::fs::write(model_dir.join("model.safetensors"), vec![0_u8; 1024])
                .expect("weight size fixture");
            let mut dirs = vec![model_dir.clone()];
            let mut config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
            if mtp {
                let draft_dir = write_minimal_model_config("qwen3_5_mtp");
                std::fs::write(draft_dir.join("model.safetensors"), vec![0_u8; 128])
                    .expect("draft size fixture");
                config.mtp = Some(EngineMtpSettings {
                    model_dir: draft_dir.clone(),
                    draft_tokens: Some(2),
                });
                dirs.push(draft_dir);
            }
            pool.register_dynamic_model(config.clone(), false, None)
                .await
                .expect("register alpha");
            Fixture {
                pool,
                hook,
                config,
                dirs,
            }
        }

        fn is_load_cancelled(error: &anyhow::Error) -> bool {
            error.downcast_ref::<LoadCancelled>().is_some()
        }

        async fn slot_snapshot(pool: &EnginePoolState, id: &str) -> EngineSlotRuntimeSnapshot {
            let slot = pool.inner.slots.lock().await[id].clone();
            slot.runtime_snapshot().await
        }

        async fn loaded_engine(pool: &EnginePoolState, id: &str) -> Arc<EngineVariant> {
            let slot = pool.inner.slots.lock().await[id].clone();
            let state = slot.state.lock().await;
            match &*state {
                super::EngineSlotState::Loaded { engine, .. } => engine.clone(),
                _ => panic!("`{id}` is not loaded"),
            }
        }

        async fn wait_for_load(pool: &EnginePoolState, model: &str) -> LoadActivity {
            tokio::time::timeout(WAIT, async {
                loop {
                    if let Some(load) = pool.load_registry().in_progress_for(model) {
                        return load;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .expect("load registered")
        }

        fn model_load_reserved_bytes() -> usize {
            global_process_memory_governor()
                .snapshot()
                .model_load_reserved_bytes
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn loaded_models_report_their_scheduler_configuration() {
            use crate::core::scheduler_resolution::SchedulerProfileKind;
            let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
            let _hook = pool.install_test_load_hook();
            let mut dirs = Vec::new();
            for (id, kind) in [
                ("tuned", Some(SchedulerProfileKind::Profile)),
                ("plain", Some(SchedulerProfileKind::Default)),
            ] {
                let model_dir = write_minimal_model_config("qwen3_5");
                std::fs::write(model_dir.join("model.safetensors"), vec![0_u8; 1024])
                    .expect("weight size fixture");
                let mut config = model_config(id, &model_dir, EngineLoadPolicy::Lazy);
                config.scheduler_profile_kind = kind;
                pool.register_dynamic_model(config, false, None)
                    .await
                    .expect("register model");
                pool.load_model(id).await.expect("load model");
                dirs.push(model_dir);
            }
            let infos = pool.loaded_model_infos().await;
            let kind_of = |id: &str| {
                infos
                    .iter()
                    .find(|info| info.id == id)
                    .expect("loaded model")
                    .scheduler_profile
            };
            assert_eq!(kind_of("tuned"), Some(SchedulerProfileKind::Profile));
            assert_eq!(kind_of("plain"), Some(SchedulerProfileKind::Default));
            for dir in dirs {
                let _ = std::fs::remove_dir_all(dir);
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn cancel_while_queued_never_takes_load_gate() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            let gate = pool.inner.load_gate.lock().await;
            let load = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("alpha").await }
            });
            let queued = wait_for_load(&pool, "alpha").await;
            assert_eq!(queued.state, LoadState::Queued);
            assert_eq!(queued.trigger, LoadTriggerKind::OpenAiControl);

            pool.load_registry()
                .cancel(&queued.load_id)
                .expect("cancel queued");
            // The gate is still held here: the load must finish without it.
            let error = tokio::time::timeout(WAIT, load)
                .await
                .expect("queued load stops while the gate is held")
                .expect("join")
                .expect_err("cancelled load");
            assert!(is_load_cancelled(&error), "{error:#}");
            drop(gate);

            let finished = pool.load_registry().wait(&queued.load_id).await.unwrap();
            assert_eq!(finished.state, LoadState::Cancelled);
            assert_eq!(finished.completed_stages.len(), 1);
            assert_eq!(finished.completed_stages[0].stage, LoadStage::Queued);
            assert_eq!(fixture.hook.loads_started(), 0);
            let slot = slot_snapshot(&pool, "alpha").await;
            assert_eq!(slot.state, EngineRuntimeState::Unloaded);
            assert_eq!(slot.load_attempts, 0);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn reload_after_cancel_succeeds_at_every_checkpoint() {
            for checkpoint in [
                None,
                Some(TestLoadPause::AfterPreparingMemory),
                Some(TestLoadPause::ReadingWeights),
                Some(TestLoadPause::BeforeDraftWeights),
                Some(TestLoadPause::BeforeEngineStart),
            ] {
                let fixture = fixture(true).await;
                let pool = fixture.pool.clone();
                let reserved_before = model_load_reserved_bytes();
                let gate = match checkpoint {
                    None => Some(pool.inner.load_gate.lock().await),
                    Some(pause) => {
                        fixture.hook.pause_at(Some(pause));
                        None
                    }
                };
                let load = tokio::spawn({
                    let pool = pool.clone();
                    async move { pool.load_model("alpha").await }
                });
                if let Some(pause) = checkpoint {
                    assert_eq!(fixture.hook.wait_reached().await, pause);
                }
                let active = wait_for_load(&pool, "alpha").await;
                pool.load_registry()
                    .cancel(&active.load_id)
                    .unwrap_or_else(|error| panic!("{checkpoint:?}: cancel: {error}"));
                if checkpoint.is_some() {
                    fixture.hook.release();
                }
                let error = tokio::time::timeout(WAIT, load)
                    .await
                    .expect("cancelled load stops")
                    .expect("join")
                    .expect_err("cancelled load");
                assert!(is_load_cancelled(&error), "{checkpoint:?}: {error:#}");
                drop(gate);

                let slot = slot_snapshot(&pool, "alpha").await;
                assert_eq!(slot.state, EngineRuntimeState::Unloaded, "{checkpoint:?}");
                assert_eq!(slot.unload_reason, Some(EngineUnloadReason::Manual));
                assert_eq!(slot.last_error, None, "{checkpoint:?}");
                let finished = pool.load_registry().wait(&active.load_id).await.unwrap();
                assert_eq!(finished.state, LoadState::Cancelled, "{checkpoint:?}");
                assert_eq!(
                    model_load_reserved_bytes(),
                    reserved_before,
                    "{checkpoint:?}: reservation released"
                );

                fixture.hook.pause_at(None);
                let reloaded = tokio::time::timeout(WAIT, pool.load_model("alpha"))
                    .await
                    .expect("reload finishes")
                    .unwrap_or_else(|error| panic!("{checkpoint:?}: reload: {error:#}"));
                assert_eq!(reloaded.state, EngineRuntimeState::Loaded, "{checkpoint:?}");
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn cancel_after_engine_start_returns_finishing() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            fixture.hook.pause_at(Some(TestLoadPause::AfterEngineStart));
            let load = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("alpha").await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::AfterEngineStart
            );
            let active = wait_for_load(&pool, "alpha").await;
            assert_eq!(active.stage, LoadStage::StartingEngine);
            assert_eq!(
                pool.load_registry().cancel(&active.load_id),
                Err(LoadCancelError::Finishing)
            );
            fixture.hook.release();

            let loaded = tokio::time::timeout(WAIT, load)
                .await
                .expect("load finishes")
                .expect("join")
                .expect("load is no longer cancellable");
            assert_eq!(loaded.state, EngineRuntimeState::Loaded);
            let finished = pool.load_registry().wait(&active.load_id).await.unwrap();
            assert_eq!(finished.state, LoadState::Loaded);
            assert_eq!(finished.bytes_done, finished.bytes_total);
            let stages: Vec<LoadStage> = finished
                .completed_stages
                .iter()
                .map(|stage| stage.stage)
                .collect();
            assert_eq!(
                stages,
                [
                    LoadStage::Queued,
                    LoadStage::PreparingMemory,
                    LoadStage::ReadingWeights,
                    LoadStage::BuildingModel,
                    LoadStage::StartingEngine,
                ]
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn cancel_reload_restores_previous_engine() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            pool.load_model("alpha").await.expect("initial load");
            let old_slot = pool.inner.slots.lock().await["alpha"].clone();
            let old_engine = loaded_engine(&pool, "alpha").await;

            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let mut changed = fixture.config.clone();
            changed.default_max_output_tokens = Some(77);
            let reload = tokio::spawn({
                let pool = pool.clone();
                async move { pool.reload_dynamic_model(changed, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let active = wait_for_load(&pool, "alpha").await;
            assert_eq!(active.kind, LoadKind::Reload);
            assert_eq!(active.trigger, LoadTriggerKind::Admin);
            pool.load_registry()
                .cancel(&active.load_id)
                .expect("cancel reload");
            fixture.hook.release();

            let error = tokio::time::timeout(WAIT, reload)
                .await
                .expect("reload stops")
                .expect("join")
                .expect_err("cancelled reload");
            assert!(is_load_cancelled(&error), "{error:#}");
            assert!(Arc::ptr_eq(
                &pool.inner.slots.lock().await["alpha"],
                &old_slot
            ));
            assert!(pool
                .loaded_model_infos()
                .await
                .iter()
                .any(|model| model.id == "alpha"));
            let (_, lease) = pool.resolve_engine(Some("alpha")).await.expect("request");
            assert!(std::ptr::eq(lease.engine(), &*old_engine));
            let finished = pool.load_registry().wait(&active.load_id).await.unwrap();
            assert_eq!(finished.state, LoadState::Cancelled);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn request_during_cancelled_reload_is_served_by_restored_engine() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            pool.load_model("alpha").await.expect("initial load");
            let old_engine = loaded_engine(&pool, "alpha").await;

            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let mut changed = fixture.config.clone();
            changed.default_max_output_tokens = Some(77);
            let reload = tokio::spawn({
                let pool = pool.clone();
                async move { pool.reload_dynamic_model(changed, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            // The request finds the reload's new slot and waits for it.
            let request = tokio::spawn({
                let pool = pool.clone();
                async move {
                    pool.resolve_engine(Some("alpha"))
                        .await
                        .map(|(_, lease)| lease)
                }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!request.is_finished());

            let active = wait_for_load(&pool, "alpha").await;
            pool.load_registry()
                .cancel(&active.load_id)
                .expect("cancel reload");
            fixture.hook.pause_at(None);
            fixture.hook.release();
            let error = tokio::time::timeout(WAIT, reload)
                .await
                .expect("reload stops")
                .expect("join")
                .expect_err("cancelled reload");
            assert!(is_load_cancelled(&error), "{error:#}");

            let lease = tokio::time::timeout(WAIT, request)
                .await
                .expect("request finishes")
                .expect("join")
                .unwrap_or_else(|error| panic!("request is served: {error:#}"));
            assert!(std::ptr::eq(lease.engine(), &*old_engine));
            assert_eq!(fixture.hook.loads_started(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn replaced_slot_does_not_load_after_the_reload_succeeds() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            let replaced = pool.inner.slots.lock().await["alpha"].clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let reload = tokio::spawn({
                let pool = pool.clone();
                let config = fixture.config.clone();
                async move { pool.reload_dynamic_model(config, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let stale = tokio::spawn({
                let pool = pool.inner.clone();
                async move {
                    replaced
                        .ensure_loaded(
                            &pool,
                            super::EngineLoadTrigger::Request,
                            EngineLoadOptions::new(LoadTriggerKind::Request),
                        )
                        .await
                        .map(|_| ())
                }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!stale.is_finished());
            fixture.hook.pause_at(None);
            fixture.hook.release();
            tokio::time::timeout(WAIT, reload)
                .await
                .expect("reload finishes")
                .expect("join")
                .expect("reloaded");
            let error = tokio::time::timeout(WAIT, stale)
                .await
                .expect("stale request ends with the reload")
                .expect("join")
                .expect_err("the stale slot is no longer the model's");
            assert!(
                error
                    .downcast_ref::<crate::core::engine_pool::EngineSlotReplaced>()
                    .is_some(),
                "{error:#}"
            );
            // The request resolves again and is served by the reloaded engine.
            pool.resolve_engine(Some("alpha")).await.expect("request");
            assert_eq!(fixture.hook.loads_started(), 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn duplicate_load_same_config_joins() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let first = tokio::spawn({
                let pool = pool.clone();
                let config = fixture.config.clone();
                async move { pool.reload_dynamic_model(config, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            // `pinned` and `set_default` do not change what gets loaded.
            let mut same = fixture.config.clone();
            same.pinned = true;
            let second = tokio::spawn({
                let pool = pool.clone();
                async move { pool.reload_dynamic_model(same, true).await }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!second.is_finished());
            assert_eq!(pool.load_registry().list().len(), 1);
            fixture.hook.release();

            for load in [first, second] {
                let result = tokio::time::timeout(WAIT, load)
                    .await
                    .expect("load finishes")
                    .expect("join")
                    .expect("joined load succeeds");
                assert_eq!(result.state, EngineRuntimeState::Loaded);
            }
            assert_eq!(fixture.hook.loads_started(), 1);
            let loads = pool.load_registry().list();
            assert_eq!(loads.len(), 1);
            assert_eq!(loads[0].state, LoadState::Loaded);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn duplicate_load_different_config_conflicts() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let first = tokio::spawn({
                let pool = pool.clone();
                let config = fixture.config.clone();
                async move { pool.reload_dynamic_model(config, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let active = wait_for_load(&pool, "alpha").await;
            let mut other = fixture.config.clone();
            other.default_max_output_tokens = Some(99);
            let error = pool
                .reload_dynamic_model(other, false)
                .await
                .expect_err("different configuration conflicts");
            let conflict = error
                .downcast_ref::<LoadInProgress>()
                .unwrap_or_else(|| panic!("load_in_progress error: {error:#}"));
            assert_eq!(conflict.load_id, active.load_id);
            fixture.hook.release();

            let loaded = tokio::time::timeout(WAIT, first)
                .await
                .expect("first load finishes")
                .expect("join")
                .expect("first load is unaffected");
            assert_eq!(loaded.state, EngineRuntimeState::Loaded);
            assert_eq!(pool.load_registry().list().len(), 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn request_triggered_load_is_registered() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            let (_, lease) = pool.resolve_engine(Some("alpha")).await.expect("request");
            drop(lease);
            let loads = pool.load_registry().list();
            assert_eq!(loads.len(), 1);
            assert_eq!(loads[0].model, "alpha");
            assert_eq!(loads[0].trigger, LoadTriggerKind::Request);
            assert_eq!(loads[0].kind, LoadKind::Load);
            assert_eq!(loads[0].state, LoadState::Loaded);
        }

        /// Single-threaded runtime: when the load finishes, the queued load
        /// woken by the released gate runs before the request that started
        /// the first load, exactly the order that let it evict an idle model.
        #[tokio::test(flavor = "current_thread")]
        #[serial_test::serial(model_load_governor)]
        async fn request_triggered_load_holds_its_lease_when_loaded() {
            let mut dirs = Vec::new();
            let mut models = Vec::new();
            for id in ["alpha", "beta"] {
                let dir = write_minimal_model_config("qwen3_5");
                std::fs::write(dir.join("model.safetensors"), vec![0_u8; 1024])
                    .expect("weight size fixture");
                models.push(model_config(id, &dir, EngineLoadPolicy::Lazy));
                dirs.push(dir);
            }
            let config = super::EnginePoolConfig {
                default_model: Some("alpha".to_string()),
                max_loaded_models: Some(1),
                models,
            };
            let pool = EnginePoolState::new(config, runtime_config())
                .await
                .expect("pool");
            let hook = pool.install_test_load_hook();
            hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let request = tokio::spawn({
                let pool = pool.clone();
                async move {
                    pool.resolve_engine(Some("alpha"))
                        .await
                        .map(|(_, lease)| lease)
                }
            });
            assert_eq!(hook.wait_reached().await, TestLoadPause::ReadingWeights);
            hook.pause_at(None);
            let control = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("beta").await }
            });
            assert_eq!(wait_for_load(&pool, "beta").await.state, LoadState::Queued);
            hook.release();

            let lease = tokio::time::timeout(WAIT, request)
                .await
                .expect("request finishes")
                .expect("join")
                .expect("request gets alpha");
            // The full pool cannot evict alpha while its request holds the
            // lease, so the queued load is refused instead of thrashing.
            let error = tokio::time::timeout(WAIT, control)
                .await
                .expect("queued load finishes")
                .expect("join")
                .expect_err("pool is full of busy models");
            assert!(
                format!("{error:#}").contains("engine pool capacity reached"),
                "{error:#}"
            );
            assert_eq!(hook.loads_started(), 1);
            let alpha = slot_snapshot(&pool, "alpha").await;
            assert_eq!(alpha.state, EngineRuntimeState::Loaded);
            assert_eq!(alpha.request_count, 1);
            drop(lease);
            for dir in dirs {
                let _ = std::fs::remove_dir_all(dir);
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn request_during_load_joins_it() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let control = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("alpha").await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let request = tokio::spawn({
                let pool = pool.clone();
                async move { pool.resolve_engine(Some("alpha")).await.map(|_| ()) }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!request.is_finished());
            fixture.hook.release();
            tokio::time::timeout(WAIT, control)
                .await
                .expect("control load")
                .expect("join")
                .expect("loaded");
            tokio::time::timeout(WAIT, request)
                .await
                .expect("request")
                .expect("join")
                .expect("request joins the load");
            assert_eq!(fixture.hook.loads_started(), 1);
            assert_eq!(pool.load_registry().list().len(), 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn replaced_slot_waits_for_the_registered_reload() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            let replaced = pool.inner.slots.lock().await["alpha"].clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let reload = tokio::spawn({
                let pool = pool.clone();
                let config = fixture.config.clone();
                async move { pool.reload_dynamic_model(config, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            // A request that resolved the slot before the swap must not start
            // a second load of the model.
            let stale = tokio::spawn({
                let pool = pool.inner.clone();
                async move {
                    replaced
                        .ensure_loaded(
                            &pool,
                            super::EngineLoadTrigger::Request,
                            EngineLoadOptions::new(LoadTriggerKind::Request),
                        )
                        .await
                        .map(|_| ())
                }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!stale.is_finished());
            assert_eq!(pool.load_registry().list().len(), 1);

            let active = wait_for_load(&pool, "alpha").await;
            pool.load_registry()
                .cancel(&active.load_id)
                .expect("cancel");
            fixture.hook.release();
            let error = tokio::time::timeout(WAIT, stale)
                .await
                .expect("stale request ends with the reload")
                .expect("join")
                .expect_err("reload was cancelled");
            assert!(is_load_cancelled(&error), "{error:#}");
            let error = reload.await.expect("join").expect_err("cancelled");
            assert!(is_load_cancelled(&error), "{error:#}");
            assert_eq!(fixture.hook.loads_started(), 1);
        }

        /// A caller that joined a load reports its failure with the code and
        /// message the load's own caller got.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn joined_load_keeps_the_capability_error_code() {
            use crate::core::load_error::classify_load_error;
            use crate::core::model_management::{
                ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
                DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
            };
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();
            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            fixture.hook.fail_with(Some(|| {
                ModelCapabilityError {
                    code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
                    message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
                }
                .into()
            }));
            let initiator = tokio::spawn({
                let pool = pool.clone();
                let config = fixture.config.clone();
                async move { pool.reload_dynamic_model(config, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let active = wait_for_load(&pool, "alpha").await;
            let joiner = tokio::spawn({
                let pool = pool.clone();
                let fingerprint =
                    crate::core::engine_pool::engine_model_config_fingerprint(&fixture.config);
                async move { pool.join_in_progress_load(&active, &fingerprint).await }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!joiner.is_finished());
            fixture.hook.release();

            let initiator = tokio::time::timeout(WAIT, initiator)
                .await
                .expect("load ends")
                .expect("join")
                .expect_err("load fails");
            let joiner = tokio::time::timeout(WAIT, joiner)
                .await
                .expect("joiner ends")
                .expect("join")
                .expect_err("joined load fails");
            let initiator = classify_load_error(&initiator);
            assert_eq!(initiator.code, Some(DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE));
            assert_eq!(classify_load_error(&joiner), initiator);
        }

        /// A cancelled first load answers the requests waiting on it with the
        /// cancel, whichever slot they found; only a later request loads.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn request_waiting_on_cancelled_first_load_is_not_reloaded() {
            use crate::core::load_error::{classify_load_error, LOAD_CANCELLED_CODE};
            for via_reload in [false, true] {
                let fixture = fixture(false).await;
                let pool = fixture.pool.clone();
                let old_slot = pool.inner.slots.lock().await["alpha"].clone();
                fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
                let load = tokio::spawn({
                    let pool = pool.clone();
                    let config = fixture.config.clone();
                    async move {
                        if via_reload {
                            pool.reload_dynamic_model(config, false).await.map(|_| ())
                        } else {
                            pool.load_model("alpha").await.map(|_| ())
                        }
                    }
                });
                assert_eq!(
                    fixture.hook.wait_reached().await,
                    TestLoadPause::ReadingWeights
                );
                // One request finds the model's current slot, one the slot it
                // had when the load started.
                let current = tokio::spawn({
                    let pool = pool.clone();
                    async move { pool.resolve_engine(Some("alpha")).await.map(|_| ()) }
                });
                let stale = tokio::spawn({
                    let pool = pool.inner.clone();
                    async move {
                        old_slot
                            .ensure_loaded(
                                &pool,
                                super::EngineLoadTrigger::Request,
                                EngineLoadOptions::new(LoadTriggerKind::Request),
                            )
                            .await
                            .map(|_| ())
                    }
                });
                tokio::time::sleep(Duration::from_millis(50)).await;
                assert!(!current.is_finished());
                assert!(!stale.is_finished());

                let active = wait_for_load(&pool, "alpha").await;
                pool.load_registry()
                    .cancel(&active.load_id)
                    .expect("cancel first load");
                fixture.hook.pause_at(None);
                fixture.hook.release();
                let error = tokio::time::timeout(WAIT, load)
                    .await
                    .expect("load stops")
                    .expect("join")
                    .expect_err("cancelled load");
                assert!(is_load_cancelled(&error), "{error:#}");
                for (name, waiter) in [("current", current), ("stale", stale)] {
                    let error = tokio::time::timeout(WAIT, waiter)
                        .await
                        .unwrap_or_else(|_| panic!("{name} request ends with the cancel"))
                        .expect("join")
                        .expect_err("the waited load was cancelled");
                    assert!(is_load_cancelled(&error), "{name}: {error:#}");
                    assert_eq!(
                        classify_load_error(&error).code,
                        Some(LOAD_CANCELLED_CODE),
                        "{name}"
                    );
                }
                assert_eq!(fixture.hook.loads_started(), 1, "reload: {via_reload}");
                assert_eq!(
                    slot_snapshot(&pool, "alpha").await.state,
                    EngineRuntimeState::Unloaded
                );

                // A request made after the cancel loads the model.
                pool.resolve_engine(Some("alpha"))
                    .await
                    .expect("later request loads");
                assert_eq!(fixture.hook.loads_started(), 2, "reload: {via_reload}");
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[serial_test::serial(model_load_governor)]
        async fn dropped_caller_future_does_not_cancel_load() {
            let fixture = fixture(false).await;
            let pool = fixture.pool.clone();

            fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
            let caller = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("alpha").await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let active = wait_for_load(&pool, "alpha").await;
            caller.abort();
            assert!(caller.await.expect_err("aborted").is_cancelled());
            fixture.hook.release();
            let finished = tokio::time::timeout(WAIT, pool.load_registry().wait(&active.load_id))
                .await
                .expect("load continues")
                .expect("known load");
            assert_eq!(finished.state, LoadState::Loaded);
            assert_eq!(
                slot_snapshot(&pool, "alpha").await.state,
                EngineRuntimeState::Loaded
            );

            // A dropped reload caller still swaps in the new engine.
            let mut changed = fixture.config.clone();
            changed.default_max_output_tokens = Some(55);
            let caller = tokio::spawn({
                let pool = pool.clone();
                async move { pool.reload_dynamic_model(changed, false).await }
            });
            assert_eq!(
                fixture.hook.wait_reached().await,
                TestLoadPause::ReadingWeights
            );
            let active = wait_for_load(&pool, "alpha").await;
            caller.abort();
            let _ = caller.await;
            fixture.hook.release();
            let finished = tokio::time::timeout(WAIT, pool.load_registry().wait(&active.load_id))
                .await
                .expect("reload continues")
                .expect("known load");
            assert_eq!(finished.state, LoadState::Loaded);
            let slot = pool.inner.slots.lock().await["alpha"].clone();
            assert_eq!(slot.model.default_max_output_tokens, Some(55));
            assert_eq!(
                slot.runtime_snapshot().await.state,
                EngineRuntimeState::Loaded
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[ignore = "requires QWEN35_MODEL pointing to a real checkpoint"]
        #[serial_test::serial(mlx_metal)]
        async fn real_model_cancel_during_weights_then_load_reports_progress() {
            let model_dir = PathBuf::from(
                std::env::var("QWEN35_MODEL")
                    .expect("QWEN35_MODEL must point to a real Qwen3.5 checkpoint"),
            );
            let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
            pool.register_dynamic_model(
                model_config("alpha", &model_dir, EngineLoadPolicy::Lazy),
                false,
                None,
            )
            .await
            .expect("register alpha");

            let load = tokio::spawn({
                let pool = pool.clone();
                async move { pool.load_model("alpha").await }
            });
            let reading = tokio::time::timeout(WAIT, async {
                loop {
                    if let Some(load) = pool.load_registry().in_progress_for("alpha") {
                        if matches!(
                            load.stage,
                            LoadStage::ReadingWeights | LoadStage::BuildingModel
                        ) {
                            return load;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("load reaches its weights");
            pool.load_registry()
                .cancel(&reading.load_id)
                .expect("cancel while reading weights");
            let error = load.await.expect("join").expect_err("cancelled load");
            assert!(is_load_cancelled(&error), "{error:#}");
            assert_eq!(
                slot_snapshot(&pool, "alpha").await.state,
                EngineRuntimeState::Unloaded
            );

            let loaded = pool.load_model("alpha").await.expect("load after cancel");
            assert_eq!(loaded.state, EngineRuntimeState::Loaded);
            let finished = pool
                .load_registry()
                .list()
                .into_iter()
                .find(|load| load.state == LoadState::Loaded)
                .expect("finished load listed");
            assert!(finished.bytes_total > 0);
            assert_eq!(finished.bytes_done, finished.bytes_total);
            let stages: Vec<LoadStage> = finished
                .completed_stages
                .iter()
                .map(|stage| stage.stage)
                .collect();
            assert_eq!(
                stages,
                [
                    LoadStage::Queued,
                    LoadStage::PreparingMemory,
                    LoadStage::ReadingWeights,
                    LoadStage::BuildingModel,
                    LoadStage::StartingEngine,
                ]
            );
        }

        #[test]
        fn config_fingerprint_ignores_pin_and_default_only() {
            let model_dir = PathBuf::from("/models/alpha");
            let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
            let fingerprint = super::super::engine_model_config_fingerprint(&config);
            let mut same = config.clone();
            same.pinned = true;
            same.default = true;
            assert_eq!(
                super::super::engine_model_config_fingerprint(&same),
                fingerprint
            );
            let mut other = config;
            other.default_max_output_tokens = Some(1);
            assert_ne!(
                super::super::engine_model_config_fingerprint(&other),
                fingerprint
            );
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EngineModelSnapshot {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    pub load_policy: EngineLoadPolicy,
    pub state: EngineRuntimeState,
    pub unload_reason: Option<EngineUnloadReason>,
    pub last_error: Option<String>,
    pub changed_unix_ms: Option<u64>,
    pub load_started_unix_ms: Option<u64>,
    pub loaded_unix_ms: Option<u64>,
    pub last_used_unix_ms: Option<u64>,
    pub failed_unix_ms: Option<u64>,
    pub load_attempts: u64,
    pub request_count: u64,
}
