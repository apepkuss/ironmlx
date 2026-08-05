//! Process-wide memory pressure control for the serving runtime.
//!
//! The KV budget in [`crate::core::memory_budget`] is a capacity invariant.
//! This module is the runtime control loop: it measures the macOS process
//! footprint and host VM state, combines those signals with the Metal working
//! set limit, and turns the resulting headroom into pressure and reservation
//! decisions shared by every scheduler in a process.

use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;
use thiserror::Error;

use crate::core::memory_budget::{kv_bytes_per_token, ModelMeta};

const GIB: usize = 1024 * 1024 * 1024;
const MIB: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    Normal,
    Soft,
    Hard,
    Emergency,
}

/// Static, metadata-only estimate of mmap-backed weights that may become part
/// of the process footprint on first use. These values are derived from the
/// sanitized loader tensor table and never require a warmup forward.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StaticMemoryEstimate {
    pub text_cold_bytes: usize,
    pub vision_cold_bytes: usize,
    pub speculative_cold_bytes: usize,
}

impl StaticMemoryEstimate {
    pub fn total_cold_bytes(self) -> usize {
        self.text_cold_bytes
            .saturating_add(self.vision_cold_bytes)
            .saturating_add(self.speculative_cold_bytes)
    }
}

/// Components touched by one real execution step. Fresh requests include the
/// base text model; mid-admission can select only a still-cold vision component
/// because the text model is already active. Components remain independent so
/// one execution cannot mark unrelated weights warm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterializationComponents {
    pub text: bool,
    pub vision: bool,
    pub speculative: bool,
}

impl MaterializationComponents {
    pub const fn text() -> Self {
        Self {
            text: true,
            vision: false,
            speculative: false,
        }
    }

    pub const fn for_request(vision: bool, speculative: bool) -> Self {
        Self {
            text: true,
            vision,
            speculative,
        }
    }

    pub fn requested_bytes(self, estimate: StaticMemoryEstimate) -> usize {
        self.bytes(estimate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentMaterializationState {
    Cold,
    Warming,
    Warm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComponentMaterializationStates {
    text: ComponentMaterializationState,
    vision: ComponentMaterializationState,
    speculative: ComponentMaterializationState,
}

impl Default for ComponentMaterializationStates {
    fn default() -> Self {
        Self {
            text: ComponentMaterializationState::Cold,
            vision: ComponentMaterializationState::Cold,
            speculative: ComponentMaterializationState::Cold,
        }
    }
}

/// Engine-lifetime tracker for first-use materialization. All execution paths
/// sharing a model also share this tracker. A component can have only one
/// Warming owner; followers wait and re-check instead of reserving it twice.
#[derive(Debug)]
pub struct ColdMaterializationTracker {
    estimate: StaticMemoryEstimate,
    states: Mutex<ComponentMaterializationStates>,
    state_changed: Condvar,
}

impl ColdMaterializationTracker {
    pub fn new(estimate: StaticMemoryEstimate) -> Arc<Self> {
        Arc::new(Self {
            estimate,
            states: Mutex::new(ComponentMaterializationStates::default()),
            state_changed: Condvar::new(),
        })
    }

    pub fn estimate(&self) -> StaticMemoryEstimate {
        self.estimate
    }

    pub fn state(&self, component: MaterializationComponent) -> ComponentMaterializationState {
        let states = self
            .states
            .lock()
            .expect("cold materialization state poisoned");
        component.state(&states)
    }

    pub fn begin(
        self: &Arc<Self>,
        components: MaterializationComponents,
        governor: &SharedProcessMemoryGovernor,
    ) -> Result<ColdMaterializationGuard, MemoryReservationError> {
        self.begin_inner(components, governor, true)
    }

    fn begin_inner(
        self: &Arc<Self>,
        components: MaterializationComponents,
        governor: &SharedProcessMemoryGovernor,
        refresh_telemetry: bool,
    ) -> Result<ColdMaterializationGuard, MemoryReservationError> {
        let mut states = self
            .states
            .lock()
            .expect("cold materialization state poisoned");
        while components.any_warming(&states) {
            states = self
                .state_changed
                .wait(states)
                .expect("cold materialization state poisoned while waiting");
        }

        let owned = components.cold_subset(&states, self.estimate);
        owned.mark(&mut states, ComponentMaterializationState::Warming);
        drop(states);

        let bytes = owned.bytes(self.estimate);
        let reservation = if bytes == 0 {
            None
        } else {
            if refresh_telemetry {
                governor.sample_process();
            }
            match governor.try_reserve_cold_materialization(bytes) {
                Ok(reservation) => Some(reservation),
                Err(_) => {
                    mlx::transforms::clear_cache();
                    if refresh_telemetry {
                        governor.refresh_process();
                    }
                    match governor.try_reserve_cold_materialization(bytes) {
                        Ok(reservation) => Some(reservation),
                        Err(error) => {
                            self.rollback(owned);
                            return Err(error);
                        }
                    }
                }
            }
        };

        Ok(ColdMaterializationGuard {
            tracker: Arc::clone(self),
            governor: Arc::clone(governor),
            owned,
            reservation,
            refresh_telemetry,
            committed: false,
        })
    }

    fn rollback(&self, owned: MaterializationComponents) {
        let mut states = self
            .states
            .lock()
            .expect("cold materialization state poisoned");
        owned.mark(&mut states, ComponentMaterializationState::Cold);
        self.state_changed.notify_all();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationComponent {
    Text,
    Vision,
    Speculative,
}

impl MaterializationComponent {
    fn state(self, states: &ComponentMaterializationStates) -> ComponentMaterializationState {
        match self {
            Self::Text => states.text,
            Self::Vision => states.vision,
            Self::Speculative => states.speculative,
        }
    }
}

impl MaterializationComponents {
    fn any_warming(self, states: &ComponentMaterializationStates) -> bool {
        (self.text && states.text == ComponentMaterializationState::Warming)
            || (self.vision && states.vision == ComponentMaterializationState::Warming)
            || (self.speculative && states.speculative == ComponentMaterializationState::Warming)
    }

    fn cold_subset(
        self,
        states: &ComponentMaterializationStates,
        estimate: StaticMemoryEstimate,
    ) -> Self {
        Self {
            text: self.text
                && estimate.text_cold_bytes > 0
                && states.text == ComponentMaterializationState::Cold,
            vision: self.vision
                && estimate.vision_cold_bytes > 0
                && states.vision == ComponentMaterializationState::Cold,
            speculative: self.speculative
                && estimate.speculative_cold_bytes > 0
                && states.speculative == ComponentMaterializationState::Cold,
        }
    }

    fn mark(
        self,
        states: &mut ComponentMaterializationStates,
        state: ComponentMaterializationState,
    ) {
        if self.text {
            states.text = state;
        }
        if self.vision {
            states.vision = state;
        }
        if self.speculative {
            states.speculative = state;
        }
    }

    fn bytes(self, estimate: StaticMemoryEstimate) -> usize {
        usize::from(self.text)
            .saturating_mul(estimate.text_cold_bytes)
            .saturating_add(usize::from(self.vision).saturating_mul(estimate.vision_cold_bytes))
            .saturating_add(
                usize::from(self.speculative).saturating_mul(estimate.speculative_cold_bytes),
            )
    }
}

/// RAII ownership of one or more Warming components and their admission
/// reservation. Success is explicit; every error, cancellation, or unwind
/// returns owned components to Cold and releases the reservation exactly once.
#[derive(Debug)]
pub struct ColdMaterializationGuard {
    tracker: Arc<ColdMaterializationTracker>,
    governor: SharedProcessMemoryGovernor,
    owned: MaterializationComponents,
    reservation: Option<MemoryReservation>,
    refresh_telemetry: bool,
    committed: bool,
}

impl ColdMaterializationGuard {
    pub fn reserved_bytes(&self) -> usize {
        self.reservation
            .as_ref()
            .map_or(0, MemoryReservation::bytes)
    }

    pub fn commit(mut self) {
        if self.refresh_telemetry {
            self.governor.refresh_process();
        }
        if let Some(reservation) = self.reservation.take() {
            reservation.commit();
        }
        let mut states = self
            .tracker
            .states
            .lock()
            .expect("cold materialization state poisoned");
        self.owned
            .mark(&mut states, ComponentMaterializationState::Warm);
        self.tracker.state_changed.notify_all();
        self.committed = true;
    }
}

impl Drop for ColdMaterializationGuard {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.reservation.take();
        self.tracker.rollback(self.owned);
    }
}

impl PressureLevel {
    pub(crate) fn rank(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Soft => 1,
            Self::Hard => 2,
            Self::Emergency => 3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryGovernorConfig {
    pub static_reserve_bytes: usize,
    pub active_reclaim_ratio: f64,
    pub soft_ratio: f64,
    pub hard_ratio: f64,
    pub prefill_headroom_ratio: f64,
    pub recovery_ratio: f64,
    pub promotion_samples: u32,
    pub recovery_samples: u32,
    pub emergency_overage_bytes: usize,
    pub minimum_prefill_chunk_tokens: usize,
    pub poll_interval: Duration,
    pub telemetry_stale_after: Duration,
    /// Normal-pressure MLX allocator cache budget as a fraction of the
    /// effective process ceiling, bounded by the byte limits below.
    pub mlx_cache_ratio: f64,
    pub mlx_cache_min_bytes: usize,
    /// Upper bound while any model component is performing first-use mmap
    /// materialization. This is process-global rather than per Engine.
    pub mlx_cache_cold_max_bytes: usize,
    pub mlx_cache_max_bytes: usize,
}

impl Default for MemoryGovernorConfig {
    fn default() -> Self {
        Self {
            static_reserve_bytes: 6 * GIB,
            active_reclaim_ratio: 0.5,
            soft_ratio: 0.90,
            hard_ratio: 0.95,
            prefill_headroom_ratio: 0.90,
            recovery_ratio: 0.85,
            promotion_samples: 2,
            recovery_samples: 3,
            emergency_overage_bytes: 2 * GIB,
            minimum_prefill_chunk_tokens: 128,
            poll_interval: Duration::from_millis(250),
            telemetry_stale_after: Duration::from_secs(2),
            mlx_cache_ratio: 0.05,
            mlx_cache_min_bytes: 128 * MIB,
            mlx_cache_cold_max_bytes: 512 * MIB,
            mlx_cache_max_bytes: 2 * GIB,
        }
    }
}

impl MemoryGovernorConfig {
    pub fn validate(self) -> Result<Self, MemoryGovernorConfigError> {
        if self.static_reserve_bytes == 0 {
            return Err(MemoryGovernorConfigError::ZeroStaticReserve);
        }
        if !(0.0..=1.0).contains(&self.active_reclaim_ratio) {
            return Err(MemoryGovernorConfigError::InvalidActiveReclaimRatio);
        }
        if !(0.0 < self.recovery_ratio
            && self.recovery_ratio < self.soft_ratio
            && self.soft_ratio < self.hard_ratio
            && self.hard_ratio <= 1.0)
        {
            return Err(MemoryGovernorConfigError::InvalidWatermarks);
        }
        if !(0.0 < self.prefill_headroom_ratio && self.prefill_headroom_ratio <= 1.0) {
            return Err(MemoryGovernorConfigError::InvalidPrefillHeadroom);
        }
        if self.promotion_samples == 0 || self.recovery_samples == 0 {
            return Err(MemoryGovernorConfigError::ZeroSampleCount);
        }
        if self.minimum_prefill_chunk_tokens == 0 {
            return Err(MemoryGovernorConfigError::ZeroMinimumPrefillChunk);
        }
        if self.poll_interval.is_zero() || self.telemetry_stale_after < self.poll_interval {
            return Err(MemoryGovernorConfigError::InvalidPollingWindow);
        }
        if !(0.0 < self.mlx_cache_ratio && self.mlx_cache_ratio <= 1.0)
            || self.mlx_cache_min_bytes == 0
            || self.mlx_cache_min_bytes > self.mlx_cache_cold_max_bytes
            || self.mlx_cache_cold_max_bytes > self.mlx_cache_max_bytes
            || self.mlx_cache_min_bytes > self.mlx_cache_max_bytes
        {
            return Err(MemoryGovernorConfigError::InvalidMlxCacheBudget);
        }
        Ok(self)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MemoryGovernorConfigError {
    #[error("memory governor static reserve must be greater than zero")]
    ZeroStaticReserve,
    #[error("memory governor active reclaim ratio must be between zero and one")]
    InvalidActiveReclaimRatio,
    #[error("memory governor watermarks must satisfy 0 < recovery < soft < hard <= 1")]
    InvalidWatermarks,
    #[error("memory governor prefill headroom ratio must satisfy 0 < ratio <= 1")]
    InvalidPrefillHeadroom,
    #[error("memory governor promotion and recovery sample counts must be non-zero")]
    ZeroSampleCount,
    #[error("memory governor minimum prefill chunk must be non-zero")]
    ZeroMinimumPrefillChunk,
    #[error("memory governor telemetry stale window must cover at least one poll interval")]
    InvalidPollingWindow,
    #[error(
        "memory governor MLX cache budget must have 0 < ratio <= 1 and 0 < min <= cold max <= max"
    )]
    InvalidMlxCacheBudget,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostVmStatistics {
    pub free_bytes: usize,
    pub active_bytes: usize,
    pub inactive_bytes: usize,
    pub wired_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryTelemetry {
    pub total_ram_bytes: usize,
    pub phys_footprint_bytes: Option<usize>,
    pub vm: Option<HostVmStatistics>,
    pub mlx_active_bytes: Option<usize>,
    pub mlx_cache_bytes: Option<usize>,
    pub metal_limit_bytes: Option<usize>,
}

impl MemoryTelemetry {
    pub fn current_usage_bytes(self) -> Option<usize> {
        match (self.phys_footprint_bytes, self.mlx_active_bytes) {
            (Some(phys), Some(active)) => Some(phys.max(active)),
            (Some(phys), None) => Some(phys),
            (None, Some(active)) => Some(active),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MemoryGovernorSnapshot {
    pub pressure_level: PressureLevel,
    pub current_usage_bytes: usize,
    /// Host memory that the governor considers reclaimable without exceeding
    /// its dynamic process ceiling: free + inactive + the configured fraction
    /// of active pages. `None` means host VM telemetry is unavailable.
    pub available_ram_bytes: Option<usize>,
    pub reserved_bytes: usize,
    pub effective_ceiling_bytes: usize,
    pub static_ceiling_bytes: usize,
    pub dynamic_ceiling_bytes: Option<usize>,
    pub metal_ceiling_bytes: Option<usize>,
    pub soft_watermark_bytes: usize,
    pub hard_watermark_bytes: usize,
    pub mlx_cache_bytes: usize,
    pub mlx_cache_limit_bytes: usize,
    pub cold_materialization_reservations: usize,
    pub prefill_reservations: usize,
    pub prefill_cache_growth_reserved_bytes: usize,
    pub telemetry_degraded: bool,
    pub sample_sequence: u64,
}

impl Default for MemoryGovernorSnapshot {
    fn default() -> Self {
        Self {
            pressure_level: PressureLevel::Normal,
            current_usage_bytes: 0,
            available_ram_bytes: None,
            reserved_bytes: 0,
            effective_ceiling_bytes: 0,
            static_ceiling_bytes: 0,
            dynamic_ceiling_bytes: None,
            metal_ceiling_bytes: None,
            soft_watermark_bytes: 0,
            hard_watermark_bytes: 0,
            mlx_cache_bytes: 0,
            mlx_cache_limit_bytes: 0,
            cold_materialization_reservations: 0,
            prefill_reservations: 0,
            prefill_cache_growth_reserved_bytes: 0,
            telemetry_degraded: true,
            sample_sequence: 0,
        }
    }
}

#[derive(Debug)]
struct GovernorState {
    snapshot: MemoryGovernorSnapshot,
    candidate_level: PressureLevel,
    candidate_samples: u32,
    recovery_samples: u32,
    last_sample_at: Option<Instant>,
}

#[derive(Debug, Default)]
struct MlxCacheControllerState {
    original_limit_bytes: Option<usize>,
    applied_limit_bytes: usize,
}

#[derive(Debug, Default)]
struct MlxCacheController {
    state: Mutex<MlxCacheControllerState>,
}

impl MlxCacheController {
    fn apply_limit(&self, requested_bytes: usize, observed_cache_bytes: usize) -> usize {
        let mut state = self.state.lock().expect("MLX cache controller poisoned");
        let original = match state.original_limit_bytes {
            Some(limit) => limit,
            None => {
                // MLX has no cache-limit getter. Setting zero is allocation-free
                // and returns the previous limit; restore the bounded value
                // before releasing the process-global controller lock.
                let limit = mlx::memory::set_cache_limit(0);
                state.original_limit_bytes = Some(limit);
                limit
            }
        };
        let applied = requested_bytes.min(original);
        if state.applied_limit_bytes != applied {
            mlx::memory::set_cache_limit(applied);
            state.applied_limit_bytes = applied;
        }
        if applied == 0 || observed_cache_bytes > applied {
            mlx::transforms::clear_cache();
        }
        applied
    }
}

static GLOBAL_MLX_CACHE_CONTROLLER: OnceLock<MlxCacheController> = OnceLock::new();

#[derive(Debug)]
pub struct ProcessMemoryGovernor {
    config: MemoryGovernorConfig,
    sample_gate: Mutex<()>,
    state: Mutex<GovernorState>,
    manage_mlx_cache: bool,
}

pub type SharedProcessMemoryGovernor = Arc<ProcessMemoryGovernor>;

static GLOBAL_PROCESS_MEMORY_GOVERNOR: OnceLock<SharedProcessMemoryGovernor> = OnceLock::new();

pub fn global_process_memory_governor() -> SharedProcessMemoryGovernor {
    Arc::clone(GLOBAL_PROCESS_MEMORY_GOVERNOR.get_or_init(|| {
        let governor = ProcessMemoryGovernor::shared_default();
        governor.sample_process();
        governor
    }))
}

impl ProcessMemoryGovernor {
    pub fn new(config: MemoryGovernorConfig) -> Result<Self, MemoryGovernorConfigError> {
        Self::new_inner(config, false)
    }

    fn new_inner(
        config: MemoryGovernorConfig,
        manage_mlx_cache: bool,
    ) -> Result<Self, MemoryGovernorConfigError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            sample_gate: Mutex::new(()),
            state: Mutex::new(GovernorState {
                snapshot: MemoryGovernorSnapshot::default(),
                candidate_level: PressureLevel::Normal,
                candidate_samples: 0,
                recovery_samples: 0,
                last_sample_at: None,
            }),
            manage_mlx_cache,
        })
    }

    pub fn shared_default() -> SharedProcessMemoryGovernor {
        Arc::new(
            Self::new_inner(MemoryGovernorConfig::default(), true)
                .expect("valid default governor config"),
        )
    }

    pub fn config(&self) -> MemoryGovernorConfig {
        self.config
    }

    pub fn sample_process(&self) -> MemoryGovernorSnapshot {
        let _sample_guard = self
            .sample_gate
            .lock()
            .expect("memory governor sample gate poisoned");
        {
            let state = self.state.lock().expect("memory governor state poisoned");
            if state
                .last_sample_at
                .is_some_and(|sampled| sampled.elapsed() < self.config.poll_interval)
            {
                return state.snapshot;
            }
        }
        self.update(native_memory_telemetry())
    }

    /// Force an authoritative native sample after this process has performed
    /// a reclaim or materialization action. Unlike the periodic sampler this
    /// bypasses the poll interval so retry and commit decisions cannot reuse a
    /// pre-action footprint.
    pub fn refresh_process(&self) -> MemoryGovernorSnapshot {
        let _sample_guard = self
            .sample_gate
            .lock()
            .expect("memory governor sample gate poisoned");
        self.update(native_memory_telemetry())
    }

    pub fn update(&self, telemetry: MemoryTelemetry) -> MemoryGovernorSnapshot {
        let mut state = self.state.lock().expect("memory governor state poisoned");
        let current = telemetry.current_usage_bytes().unwrap_or(0);
        let total = telemetry.total_ram_bytes;
        let static_ceiling = total
            .saturating_sub(self.config.static_reserve_bytes)
            .max(1);
        let available_ram_bytes = telemetry.vm.map(|vm| {
            vm.free_bytes
                .saturating_add(vm.inactive_bytes)
                .saturating_add(
                    ((vm.active_bytes as f64) * self.config.active_reclaim_ratio) as usize,
                )
                .max(1)
        });
        let dynamic_ceiling =
            available_ram_bytes.map(|available| current.saturating_add(available).max(1));
        let metal_ceiling = telemetry.metal_limit_bytes.filter(|limit| *limit > 0);
        let mut effective_ceiling = static_ceiling;
        if let Some(dynamic) = dynamic_ceiling {
            effective_ceiling = effective_ceiling.min(dynamic);
        }
        if let Some(metal) = metal_ceiling {
            effective_ceiling = effective_ceiling.min(metal);
        }

        let telemetry_degraded = telemetry.phys_footprint_bytes.is_none()
            || telemetry.vm.is_none()
            || telemetry.metal_limit_bytes.is_none();
        let soft = ratio_bytes(effective_ceiling, self.config.soft_ratio);
        let hard = ratio_bytes(effective_ceiling, self.config.hard_ratio);
        let observed = classify_observed_pressure(
            current.saturating_add(state.snapshot.reserved_bytes),
            effective_ceiling,
            soft,
            hard,
            self.config.emergency_overage_bytes,
            telemetry.current_usage_bytes().is_some(),
        );
        let recovery = ratio_bytes(effective_ceiling, self.config.recovery_ratio);
        if telemetry_degraded {
            // A partial sample cannot prove that the process is safe. Enter the
            // fail-safe soft state immediately so admission pauses without
            // waiting for the normal promotion debounce window.
            if state.snapshot.pressure_level == PressureLevel::Normal {
                state.snapshot.pressure_level = PressureLevel::Soft;
            }
            state.candidate_level = state.snapshot.pressure_level;
            state.candidate_samples = 0;
            state.recovery_samples = 0;
        } else {
            let current_with_reservations = current.saturating_add(state.snapshot.reserved_bytes);
            advance_pressure_state(
                &mut state,
                observed,
                current_with_reservations,
                recovery,
                self.config,
            );
        }

        state.snapshot.current_usage_bytes = current;
        state.snapshot.available_ram_bytes = available_ram_bytes;
        state.snapshot.effective_ceiling_bytes = effective_ceiling;
        state.snapshot.static_ceiling_bytes = static_ceiling;
        state.snapshot.dynamic_ceiling_bytes = dynamic_ceiling;
        state.snapshot.metal_ceiling_bytes = metal_ceiling;
        state.snapshot.soft_watermark_bytes = soft;
        state.snapshot.hard_watermark_bytes = hard;
        state.snapshot.mlx_cache_bytes = telemetry.mlx_cache_bytes.unwrap_or(0);
        state.snapshot.telemetry_degraded = telemetry_degraded;
        state.snapshot.sample_sequence = state.snapshot.sample_sequence.wrapping_add(1);
        state.last_sample_at = Some(Instant::now());
        self.apply_mlx_cache_budget_locked(&mut state);
        state.snapshot
    }

    pub fn snapshot(&self) -> MemoryGovernorSnapshot {
        let mut state = self.state.lock().expect("memory governor state poisoned");
        if state
            .last_sample_at
            .is_none_or(|sampled| sampled.elapsed() > self.config.telemetry_stale_after)
        {
            state.snapshot.telemetry_degraded = true;
            if state.snapshot.pressure_level == PressureLevel::Normal {
                state.snapshot.pressure_level = PressureLevel::Soft;
            }
            self.apply_mlx_cache_budget_locked(&mut state);
        }
        state.snapshot
    }

    pub fn admission_paused(&self) -> bool {
        self.snapshot().pressure_level != PressureLevel::Normal
    }

    pub fn try_reserve(
        self: &Arc<Self>,
        bytes: usize,
        purpose: &'static str,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        self.try_reserve_inner(bytes, purpose, ReservationKind::General)
    }

    fn try_reserve_cold_materialization(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        self.try_reserve_inner(
            bytes,
            "cold_materialization",
            ReservationKind::ColdMaterialization,
        )
    }

    /// Reserve both the modeled transient peak and any allocator-cache growth
    /// permitted by the current global cap. The latter is zero once the cache
    /// is already populated and remains process-global across Engines.
    pub fn try_reserve_prefill(
        self: &Arc<Self>,
        transient_bytes: usize,
        purpose: &'static str,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        self.try_reserve_inner(transient_bytes, purpose, ReservationKind::Prefill)
    }

    fn try_reserve_inner(
        self: &Arc<Self>,
        bytes: usize,
        purpose: &'static str,
        kind: ReservationKind,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        let mut state = self.state.lock().expect("memory governor state poisoned");
        let telemetry_stale = state
            .last_sample_at
            .is_none_or(|sampled| sampled.elapsed() > self.config.telemetry_stale_after);
        if telemetry_stale {
            state.snapshot.telemetry_degraded = true;
            if state.snapshot.pressure_level == PressureLevel::Normal {
                state.snapshot.pressure_level = PressureLevel::Soft;
            }
            self.apply_mlx_cache_budget_locked(&mut state);
        }
        let snapshot = state.snapshot;
        if snapshot.effective_ceiling_bytes == 0 || snapshot.telemetry_degraded || telemetry_stale {
            return Err(MemoryReservationError::TelemetryUnavailable { purpose });
        }
        if snapshot.pressure_level.rank() >= PressureLevel::Hard.rank() {
            return Err(MemoryReservationError::Pressure {
                purpose,
                level: snapshot.pressure_level,
            });
        }
        let target = ratio_bytes(
            snapshot.effective_ceiling_bytes,
            self.config.prefill_headroom_ratio,
        );
        let cache_growth_liability =
            if kind == ReservationKind::Prefill && snapshot.prefill_reservations == 0 {
                prefill_cache_growth_liability(snapshot)
            } else {
                0
            };
        let reserved_bytes = bytes.saturating_add(cache_growth_liability);
        let projected = snapshot
            .current_usage_bytes
            .saturating_add(snapshot.reserved_bytes)
            .saturating_add(reserved_bytes);
        if projected > target {
            return Err(MemoryReservationError::InsufficientHeadroom {
                purpose,
                requested_bytes: reserved_bytes,
                projected_bytes: projected,
                target_bytes: target,
            });
        }
        state.snapshot.reserved_bytes =
            state.snapshot.reserved_bytes.saturating_add(reserved_bytes);
        if kind == ReservationKind::ColdMaterialization {
            state.snapshot.cold_materialization_reservations = state
                .snapshot
                .cold_materialization_reservations
                .saturating_add(1);
        } else if kind == ReservationKind::Prefill {
            state.snapshot.prefill_reservations =
                state.snapshot.prefill_reservations.saturating_add(1);
            state.snapshot.prefill_cache_growth_reserved_bytes = state
                .snapshot
                .prefill_cache_growth_reserved_bytes
                .saturating_add(cache_growth_liability);
        }
        self.apply_mlx_cache_budget_locked(&mut state);
        Ok(MemoryReservation {
            governor: Arc::clone(self),
            bytes: reserved_bytes,
            cache_growth_liability_bytes: cache_growth_liability,
            kind,
            committed: false,
        })
    }

    fn release_reservation(
        &self,
        bytes: usize,
        cache_growth_liability_bytes: usize,
        kind: ReservationKind,
    ) {
        let mut state = self.state.lock().expect("memory governor state poisoned");
        if kind == ReservationKind::Prefill {
            let transient_bytes = bytes.saturating_sub(cache_growth_liability_bytes);
            state.snapshot.reserved_bytes = state
                .snapshot
                .reserved_bytes
                .saturating_sub(transient_bytes);
            state.snapshot.prefill_reservations =
                state.snapshot.prefill_reservations.saturating_sub(1);
            if state.snapshot.prefill_reservations == 0 {
                state.snapshot.reserved_bytes = state
                    .snapshot
                    .reserved_bytes
                    .saturating_sub(state.snapshot.prefill_cache_growth_reserved_bytes);
                state.snapshot.prefill_cache_growth_reserved_bytes = 0;
            }
        } else {
            state.snapshot.reserved_bytes = state.snapshot.reserved_bytes.saturating_sub(bytes);
        }
        if kind == ReservationKind::ColdMaterialization {
            state.snapshot.cold_materialization_reservations = state
                .snapshot
                .cold_materialization_reservations
                .saturating_sub(1);
        }
        self.apply_mlx_cache_budget_locked(&mut state);
    }

    fn apply_mlx_cache_budget_locked(&self, state: &mut GovernorState) {
        let snapshot = state.snapshot;
        let safe_target = ratio_bytes(
            snapshot.effective_ceiling_bytes,
            self.config.prefill_headroom_ratio,
        );
        let non_cache_usage = snapshot
            .current_usage_bytes
            .saturating_sub(snapshot.mlx_cache_bytes);
        let unreserved_headroom = safe_target
            .saturating_sub(non_cache_usage)
            .saturating_sub(snapshot.reserved_bytes);
        let normal_budget = ratio_bytes(
            snapshot.effective_ceiling_bytes,
            self.config.mlx_cache_ratio,
        )
        .clamp(
            self.config.mlx_cache_min_bytes,
            self.config.mlx_cache_max_bytes,
        )
        .min(unreserved_headroom);
        let desired = if snapshot.telemetry_degraded {
            0
        } else {
            match snapshot.pressure_level {
                PressureLevel::Normal => {
                    if snapshot.cold_materialization_reservations > 0 {
                        normal_budget.min(self.config.mlx_cache_cold_max_bytes)
                    } else {
                        normal_budget
                    }
                }
                PressureLevel::Soft => {
                    (normal_budget / 2).min(self.config.mlx_cache_cold_max_bytes)
                }
                PressureLevel::Hard | PressureLevel::Emergency => 0,
            }
        };
        let desired = if snapshot.prefill_reservations > 0 {
            desired.min(
                snapshot
                    .mlx_cache_bytes
                    .saturating_add(snapshot.prefill_cache_growth_reserved_bytes),
            )
        } else {
            desired
        };
        state.snapshot.mlx_cache_limit_bytes = if self.manage_mlx_cache {
            GLOBAL_MLX_CACHE_CONTROLLER
                .get_or_init(MlxCacheController::default)
                .apply_limit(desired, snapshot.mlx_cache_bytes)
        } else {
            desired
        };
    }

    pub fn plan_prefill_chunk(
        self: &Arc<Self>,
        requested_tokens: usize,
        kv_len: usize,
        batch_size: usize,
        meta: &ModelMeta,
    ) -> Result<PrefillChunkPlan, PrefillGuardError> {
        if requested_tokens == 0 {
            return Err(PrefillGuardError::ZeroRequestedTokens);
        }
        let snapshot = self.snapshot();
        if snapshot.effective_ceiling_bytes == 0 || snapshot.telemetry_degraded {
            return Err(PrefillGuardError::TelemetryUnavailable);
        }
        let target = ratio_bytes(
            snapshot.effective_ceiling_bytes,
            self.config.prefill_headroom_ratio,
        );
        let cache_growth_liability = prefill_cache_growth_liability(snapshot);
        let baseline = snapshot
            .current_usage_bytes
            .saturating_add(snapshot.reserved_bytes)
            .saturating_add(cache_growth_liability);
        let minimum = requested_tokens.min(self.config.minimum_prefill_chunk_tokens.max(1));
        let minimum_bytes = estimate_prefill_peak_growth(meta, minimum, kv_len, batch_size);
        if baseline.saturating_add(minimum_bytes) > target {
            return Err(PrefillGuardError::MinimumChunkUnsafe {
                minimum_tokens: minimum,
                estimated_bytes: baseline.saturating_add(minimum_bytes),
                target_bytes: target,
            });
        }

        let requested_bytes =
            estimate_prefill_peak_growth(meta, requested_tokens, kv_len, batch_size);
        let selected = if baseline.saturating_add(requested_bytes) <= target {
            requested_tokens
        } else {
            largest_safe_chunk(minimum, requested_tokens, |tokens| {
                baseline.saturating_add(estimate_prefill_peak_growth(
                    meta, tokens, kv_len, batch_size,
                )) <= target
            })
        };
        let reserved_bytes = estimate_prefill_peak_growth(meta, selected, kv_len, batch_size);
        let reservation = self
            .try_reserve_prefill(reserved_bytes, "prefill_chunk")
            .map_err(PrefillGuardError::Reservation)?;
        let cache_growth_liability_bytes = reservation.bytes().saturating_sub(reserved_bytes);
        Ok(PrefillChunkPlan {
            requested_tokens,
            selected_tokens: selected,
            estimated_peak_growth_bytes: reserved_bytes,
            cache_growth_liability_bytes,
            target_bytes: target,
            reservation,
        })
    }
}

fn advance_pressure_state(
    state: &mut GovernorState,
    observed: PressureLevel,
    current: usize,
    recovery_watermark: usize,
    config: MemoryGovernorConfig,
) {
    let level = state.snapshot.pressure_level;
    if observed.rank() > level.rank() {
        state.recovery_samples = 0;
        if observed == PressureLevel::Emergency {
            state.snapshot.pressure_level = PressureLevel::Emergency;
            state.candidate_samples = 0;
            return;
        }
        if state.candidate_level == observed {
            state.candidate_samples = state.candidate_samples.saturating_add(1);
        } else {
            state.candidate_level = observed;
            state.candidate_samples = 1;
        }
        if state.candidate_samples >= config.promotion_samples {
            state.snapshot.pressure_level = observed;
            state.candidate_samples = 0;
        }
        return;
    }

    state.candidate_samples = 0;
    if current <= recovery_watermark {
        state.recovery_samples = state.recovery_samples.saturating_add(1);
        if state.recovery_samples >= config.recovery_samples {
            state.snapshot.pressure_level = match level {
                PressureLevel::Emergency => PressureLevel::Hard,
                PressureLevel::Hard => PressureLevel::Soft,
                PressureLevel::Soft | PressureLevel::Normal => PressureLevel::Normal,
            };
            state.recovery_samples = 0;
        }
    } else {
        state.recovery_samples = 0;
    }
}

fn classify_observed_pressure(
    current: usize,
    ceiling: usize,
    soft: usize,
    hard: usize,
    emergency_overage: usize,
    usage_valid: bool,
) -> PressureLevel {
    if !usage_valid || ceiling == 0 {
        return PressureLevel::Soft;
    }
    if current >= ceiling.saturating_add(emergency_overage) {
        PressureLevel::Emergency
    } else if current >= hard {
        PressureLevel::Hard
    } else if current >= soft {
        PressureLevel::Soft
    } else {
        PressureLevel::Normal
    }
}

fn ratio_bytes(bytes: usize, ratio: f64) -> usize {
    ((bytes as f64) * ratio) as usize
}

fn prefill_cache_growth_liability(snapshot: MemoryGovernorSnapshot) -> usize {
    if snapshot.prefill_reservations > 0 {
        return 0;
    }
    snapshot
        .mlx_cache_limit_bytes
        .saturating_sub(snapshot.mlx_cache_bytes)
}

pub fn estimate_prefill_peak_growth(
    meta: &ModelMeta,
    query_tokens: usize,
    kv_len: usize,
    batch_size: usize,
) -> usize {
    let batch = batch_size.max(1);
    let query = query_tokens.max(1);
    let context = kv_len.saturating_add(query).max(1);
    let kv_growth = kv_bytes_per_token(meta)
        .saturating_mul(query)
        .saturating_mul(batch);
    let score_peak = (meta.num_attention_heads.max(1) as usize)
        .saturating_mul(query)
        .saturating_mul(context)
        .saturating_mul(4)
        .saturating_mul(batch);
    let hidden_peak = (meta.hidden_size.max(1) as usize)
        .saturating_mul(query)
        .saturating_mul(2)
        .saturating_mul(batch);
    kv_growth
        .saturating_add(score_peak)
        .saturating_add(hidden_peak)
}

fn largest_safe_chunk(
    minimum: usize,
    requested: usize,
    mut fits: impl FnMut(usize) -> bool,
) -> usize {
    let mut low = minimum;
    let mut high = requested;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

#[derive(Debug, Error)]
pub enum MemoryReservationError {
    #[error("memory telemetry unavailable while reserving {purpose}")]
    TelemetryUnavailable { purpose: &'static str },
    #[error("memory pressure {level:?} blocks reservation for {purpose}")]
    Pressure {
        purpose: &'static str,
        level: PressureLevel,
    },
    #[error(
        "insufficient memory headroom for {purpose}: request={requested_bytes} projected={projected_bytes} target={target_bytes}"
    )]
    InsufficientHeadroom {
        purpose: &'static str,
        requested_bytes: usize,
        projected_bytes: usize,
        target_bytes: usize,
    },
}

#[derive(Debug)]
pub struct MemoryReservation {
    governor: SharedProcessMemoryGovernor,
    bytes: usize,
    cache_growth_liability_bytes: usize,
    kind: ReservationKind,
    committed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationKind {
    General,
    ColdMaterialization,
    Prefill,
}

impl MemoryReservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn commit(mut self) {
        self.governor
            .release_reservation(self.bytes, self.cache_growth_liability_bytes, self.kind);
        self.committed = true;
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.governor.release_reservation(
                self.bytes,
                self.cache_growth_liability_bytes,
                self.kind,
            );
        }
    }
}

#[derive(Debug)]
pub struct PrefillChunkPlan {
    pub requested_tokens: usize,
    pub selected_tokens: usize,
    pub estimated_peak_growth_bytes: usize,
    pub cache_growth_liability_bytes: usize,
    pub target_bytes: usize,
    pub reservation: MemoryReservation,
}

#[derive(Debug, Error)]
pub enum PrefillGuardError {
    #[error("prefill chunk request must contain at least one token")]
    ZeroRequestedTokens,
    #[error(
        "authoritative memory telemetry is unavailable; prefill admission is fail-safe closed"
    )]
    TelemetryUnavailable,
    #[error(
        "minimum prefill chunk is unsafe: tokens={minimum_tokens} estimated={estimated_bytes} target={target_bytes}"
    )]
    MinimumChunkUnsafe {
        minimum_tokens: usize,
        estimated_bytes: usize,
        target_bytes: usize,
    },
    #[error(transparent)]
    Reservation(#[from] MemoryReservationError),
}

pub fn native_memory_telemetry() -> MemoryTelemetry {
    let mlx = mlx::memory::snapshot();
    let metal_limit_bytes = [Some(mlx.memory_limit_bytes), mlx.max_recommended_bytes]
        .into_iter()
        .flatten()
        .filter(|value| *value > 0)
        .min();
    MemoryTelemetry {
        total_ram_bytes: crate::core::memory_budget::system_total_ram_bytes(),
        phys_footprint_bytes: macos_phys_footprint_bytes(),
        vm: macos_host_vm_statistics(),
        mlx_active_bytes: Some(mlx.active_bytes),
        mlx_cache_bytes: Some(mlx.cache_bytes),
        metal_limit_bytes,
    }
}

#[repr(C)]
#[derive(Default)]
struct RusageInfoV4 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    tail: [u64; 27],
}

fn macos_phys_footprint_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "proc")]
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut libc::c_void) -> i32;
        }
        let mut info = RusageInfoV4::default();
        // SAFETY: `info` has the public rusage_info_v4 layout and remains valid
        // for the duration of the synchronous kernel call.
        let rc = unsafe {
            proc_pid_rusage(
                std::process::id() as i32,
                4,
                (&mut info as *mut RusageInfoV4).cast(),
            )
        };
        if rc == 0 && info.phys_footprint > 0 {
            usize::try_from(info.phys_footprint).ok()
        } else {
            None
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn macos_host_vm_statistics() -> Option<HostVmStatistics> {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn mach_host_self() -> u32;
            static mach_task_self_: u32;
            fn mach_port_deallocate(task: u32, name: u32) -> i32;
            fn host_page_size(host: u32, page_size: *mut u32) -> i32;
            fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        }
        let host = unsafe { mach_host_self() };
        let result = (|| {
            let mut page_size = 0_u32;
            // SAFETY: valid out pointer and host port returned by mach_host_self.
            if unsafe { host_page_size(host, &mut page_size) } != 0 || page_size == 0 {
                return None;
            }
            let mut words = [0_i32; 64];
            let mut count = words.len() as u32;
            // HOST_VM_INFO64 = 4. The leading free/active/inactive/wire counters
            // are stable natural_t fields; a max-sized buffer avoids depending on
            // the SDK-specific tail of vm_statistics64_data_t.
            let rc = unsafe { host_statistics64(host, 4, words.as_mut_ptr(), &mut count) };
            if rc != 0 || count < 4 {
                return None;
            }
            let bytes = |pages: i32| (pages as u32 as usize).saturating_mul(page_size as usize);
            Some(HostVmStatistics {
                free_bytes: bytes(words[0]),
                active_bytes: bytes(words[1]),
                inactive_bytes: bytes(words[2]),
                wired_bytes: bytes(words[3]),
            })
        })();
        // SAFETY: `mach_host_self` returns an owned send right reference for
        // the current task. Balance it on every success and failure path.
        let _ = unsafe { mach_port_deallocate(mach_task_self_, host) };
        result
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gib(value: usize) -> usize {
        value * GIB
    }

    fn config() -> MemoryGovernorConfig {
        MemoryGovernorConfig {
            static_reserve_bytes: gib(4),
            promotion_samples: 2,
            recovery_samples: 2,
            emergency_overage_bytes: gib(1),
            minimum_prefill_chunk_tokens: 16,
            ..MemoryGovernorConfig::default()
        }
    }

    fn telemetry(usage: usize) -> MemoryTelemetry {
        MemoryTelemetry {
            total_ram_bytes: gib(32),
            phys_footprint_bytes: Some(usage),
            vm: Some(HostVmStatistics {
                free_bytes: gib(8),
                inactive_bytes: gib(2),
                active_bytes: gib(8),
                wired_bytes: gib(8),
            }),
            mlx_active_bytes: Some(usage.saturating_sub(1)),
            mlx_cache_bytes: Some(0),
            metal_limit_bytes: Some(gib(24)),
        }
    }

    #[test]
    fn effective_ceiling_is_minimum_of_static_dynamic_and_metal() {
        let governor = ProcessMemoryGovernor::new(config()).unwrap();
        let snapshot = governor.update(telemetry(gib(10)));
        assert_eq!(snapshot.available_ram_bytes, Some(gib(14)));
        assert_eq!(snapshot.static_ceiling_bytes, gib(28));
        assert_eq!(snapshot.dynamic_ceiling_bytes, Some(gib(24)));
        assert_eq!(snapshot.metal_ceiling_bytes, Some(gib(24)));
        assert_eq!(snapshot.effective_ceiling_bytes, gib(24));
    }

    #[test]
    fn mlx_cache_budget_is_bounded_and_pressure_driven() {
        let mut cfg = config();
        cfg.promotion_samples = 1;
        cfg.mlx_cache_ratio = 0.50;
        cfg.mlx_cache_min_bytes = 64 * MIB;
        cfg.mlx_cache_max_bytes = 512 * MIB;
        let governor = ProcessMemoryGovernor::new(cfg).unwrap();

        let normal = governor.update(telemetry(gib(4)));
        assert_eq!(normal.mlx_cache_limit_bytes, 512 * MIB);

        let mut soft_sample = telemetry(gib(22));
        soft_sample.vm = Some(HostVmStatistics {
            free_bytes: gib(2),
            inactive_bytes: 0,
            active_bytes: gib(2),
            wired_bytes: gib(8),
        });
        let soft = governor.update(soft_sample);
        assert_eq!(soft.pressure_level, PressureLevel::Soft);
        assert!(soft.mlx_cache_limit_bytes <= 256 * MIB);

        let mut degraded = telemetry(gib(4));
        degraded.vm = None;
        let fail_safe = governor.update(degraded);
        assert_eq!(fail_safe.mlx_cache_limit_bytes, 0);
    }

    #[test]
    fn reservations_reduce_cache_budget_without_allocating_memory() {
        let mut cfg = config();
        cfg.mlx_cache_ratio = 1.0;
        cfg.mlx_cache_min_bytes = 1;
        cfg.mlx_cache_max_bytes = gib(8);
        let governor = Arc::new(ProcessMemoryGovernor::new(cfg).unwrap());
        let mut sample = telemetry(gib(20));
        sample.mlx_active_bytes = Some(gib(20));
        sample.mlx_cache_bytes = Some(0);
        let before = governor.update(sample);
        let reservation = governor.try_reserve(gib(1), "cache_budget_test").unwrap();
        let during = governor.snapshot();
        assert_eq!(during.current_usage_bytes, before.current_usage_bytes);
        assert_eq!(during.reserved_bytes, gib(1));
        assert_eq!(
            during.mlx_cache_limit_bytes,
            before.mlx_cache_limit_bytes.saturating_sub(gib(1))
        );
        drop(reservation);
        let after = governor.snapshot();
        assert_eq!(after.reserved_bytes, 0);
        assert_eq!(after.mlx_cache_limit_bytes, before.mlx_cache_limit_bytes);
    }

    #[test]
    fn cold_materialization_clamps_cache_until_last_global_reservation_releases() {
        let mut cfg = config();
        cfg.mlx_cache_ratio = 1.0;
        cfg.mlx_cache_min_bytes = 128 * MIB;
        cfg.mlx_cache_cold_max_bytes = 512 * MIB;
        cfg.mlx_cache_max_bytes = gib(2);
        let governor = Arc::new(ProcessMemoryGovernor::new(cfg).unwrap());
        let before = governor.update(telemetry(gib(4)));
        assert_eq!(before.mlx_cache_limit_bytes, gib(2));
        assert_eq!(before.cold_materialization_reservations, 0);

        let first = governor
            .try_reserve_cold_materialization(128 * MIB)
            .unwrap();
        let second = governor
            .try_reserve_cold_materialization(256 * MIB)
            .unwrap();
        let both = governor.snapshot();
        assert_eq!(both.cold_materialization_reservations, 2);
        assert_eq!(both.reserved_bytes, 384 * MIB);
        assert_eq!(both.mlx_cache_limit_bytes, 512 * MIB);

        drop(first);
        let one = governor.snapshot();
        assert_eq!(one.cold_materialization_reservations, 1);
        assert_eq!(one.reserved_bytes, 256 * MIB);
        assert_eq!(one.mlx_cache_limit_bytes, 512 * MIB);

        second.commit();
        let restored = governor.snapshot();
        assert_eq!(restored.cold_materialization_reservations, 0);
        assert_eq!(restored.reserved_bytes, 0);
        assert_eq!(restored.mlx_cache_limit_bytes, gib(2));
    }

    #[test]
    fn failed_cold_materialization_reservation_does_not_leak_cache_clamp() {
        let mut cfg = config();
        cfg.mlx_cache_ratio = 1.0;
        cfg.mlx_cache_cold_max_bytes = 512 * MIB;
        cfg.mlx_cache_max_bytes = gib(2);
        let governor = Arc::new(ProcessMemoryGovernor::new(cfg).unwrap());
        let mut sample = telemetry(gib(4));
        sample.vm = None;
        governor.update(sample);

        assert!(matches!(
            governor.try_reserve_cold_materialization(128 * MIB),
            Err(MemoryReservationError::TelemetryUnavailable { .. })
        ));
        let snapshot = governor.snapshot();
        assert_eq!(snapshot.cold_materialization_reservations, 0);
        assert_eq!(snapshot.reserved_bytes, 0);
    }

    #[test]
    fn concurrent_prefills_reserve_global_cache_growth_once_until_last_release() {
        let mut cfg = config();
        cfg.mlx_cache_ratio = 1.0;
        cfg.mlx_cache_min_bytes = 128 * MIB;
        cfg.mlx_cache_cold_max_bytes = 512 * MIB;
        cfg.mlx_cache_max_bytes = gib(2);
        let governor = Arc::new(ProcessMemoryGovernor::new(cfg).unwrap());
        governor.update(telemetry(gib(4)));

        let first = governor
            .try_reserve_prefill(128 * MIB, "first_prefill")
            .unwrap();
        let after_first = governor.snapshot();
        assert_eq!(after_first.prefill_reservations, 1);
        assert_eq!(after_first.prefill_cache_growth_reserved_bytes, gib(2));
        assert_eq!(after_first.reserved_bytes, gib(2) + 128 * MIB);

        let second = governor
            .try_reserve_prefill(256 * MIB, "second_prefill")
            .unwrap();
        let concurrent = governor.snapshot();
        assert_eq!(concurrent.prefill_reservations, 2);
        assert_eq!(concurrent.prefill_cache_growth_reserved_bytes, gib(2));
        assert_eq!(concurrent.reserved_bytes, gib(2) + 384 * MIB);
        assert_eq!(second.bytes(), 256 * MIB);

        drop(first);
        let transferred = governor.snapshot();
        assert_eq!(transferred.prefill_reservations, 1);
        assert_eq!(transferred.prefill_cache_growth_reserved_bytes, gib(2));
        assert_eq!(transferred.reserved_bytes, gib(2) + 256 * MIB);

        second.commit();
        let released = governor.snapshot();
        assert_eq!(released.prefill_reservations, 0);
        assert_eq!(released.prefill_cache_growth_reserved_bytes, 0);
        assert_eq!(released.reserved_bytes, 0);
    }

    #[test]
    fn concurrent_reservations_share_one_process_cache_budget() {
        let mut cfg = config();
        cfg.mlx_cache_ratio = 1.0;
        cfg.mlx_cache_min_bytes = 1;
        cfg.mlx_cache_max_bytes = gib(8);
        let governor = Arc::new(ProcessMemoryGovernor::new(cfg).unwrap());
        let mut sample = telemetry(gib(20));
        sample.mlx_active_bytes = Some(gib(20));
        sample.mlx_cache_bytes = Some(0);
        let before = governor.update(sample);
        let barrier = Arc::new(std::sync::Barrier::new(5));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let governor = Arc::clone(&governor);
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let reservation = governor
                    .try_reserve(128 * MIB, "cross_engine_cache_budget")
                    .unwrap();
                barrier.wait();
                barrier.wait();
                drop(reservation);
            }));
        }
        barrier.wait();
        let during = governor.snapshot();
        assert_eq!(during.reserved_bytes, 512 * MIB);
        assert_eq!(
            during.mlx_cache_limit_bytes,
            before.mlx_cache_limit_bytes.saturating_sub(512 * MIB)
        );
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn missing_authoritative_signal_fails_safe_closed() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        let mut sample = telemetry(gib(4));
        sample.phys_footprint_bytes = None;
        sample.vm = None;
        governor.update(sample);
        let snapshot = governor.snapshot();
        assert!(snapshot.telemetry_degraded);
        assert_eq!(snapshot.pressure_level, PressureLevel::Soft);
        assert!(matches!(
            governor.try_reserve(1, "test"),
            Err(MemoryReservationError::TelemetryUnavailable { .. })
        ));
    }

    #[test]
    fn stale_authoritative_sample_fails_safe_closed_at_reservation_boundary() {
        let governor = Arc::new(
            ProcessMemoryGovernor::new(MemoryGovernorConfig {
                poll_interval: Duration::from_millis(1),
                telemetry_stale_after: Duration::from_millis(2),
                ..config()
            })
            .unwrap(),
        );
        governor.update(telemetry(gib(4)));
        std::thread::sleep(Duration::from_millis(5));

        let snapshot = governor.snapshot();
        assert!(snapshot.telemetry_degraded);
        assert_eq!(snapshot.pressure_level, PressureLevel::Soft);
        assert!(matches!(
            governor.try_reserve(1, "stale_test"),
            Err(MemoryReservationError::TelemetryUnavailable { .. })
        ));
    }

    #[test]
    fn promotion_requires_consecutive_samples_and_recovery_has_hysteresis() {
        let governor = ProcessMemoryGovernor::new(config()).unwrap();
        governor.update(telemetry(gib(23)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Normal);
        governor.update(telemetry(gib(23)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Hard);

        governor.update(telemetry(gib(8)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Hard);
        governor.update(telemetry(gib(8)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Soft);
        governor.update(telemetry(gib(8)));
        governor.update(telemetry(gib(8)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Normal);
    }

    #[test]
    fn emergency_promotion_is_immediate_but_recovers_stepwise() {
        let governor = ProcessMemoryGovernor::new(config()).unwrap();
        governor.update(telemetry(gib(26)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Emergency);
        governor.update(telemetry(gib(8)));
        governor.update(telemetry(gib(8)));
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Hard);
    }

    #[test]
    fn reservations_are_conserved_on_drop_and_commit() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        governor.update(telemetry(gib(4)));
        {
            let reservation = governor.try_reserve(1024, "test").unwrap();
            assert_eq!(governor.snapshot().reserved_bytes, 1024);
            drop(reservation);
        }
        assert_eq!(governor.snapshot().reserved_bytes, 0);
        let reservation = governor.try_reserve(2048, "test").unwrap();
        reservation.commit();
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn cold_materialization_tracks_components_independently() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        let tracker = ColdMaterializationTracker::new(StaticMemoryEstimate {
            text_cold_bytes: 1_024,
            vision_cold_bytes: 2_048,
            speculative_cold_bytes: 4_096,
        });

        let text = tracker
            .begin_inner(MaterializationComponents::text(), &governor, false)
            .unwrap();
        assert_eq!(text.reserved_bytes(), 1_024);
        assert_eq!(
            tracker.state(MaterializationComponent::Text),
            ComponentMaterializationState::Warming
        );
        text.commit();
        assert_eq!(
            tracker.state(MaterializationComponent::Text),
            ComponentMaterializationState::Warm
        );
        assert_eq!(
            tracker.state(MaterializationComponent::Vision),
            ComponentMaterializationState::Cold
        );

        let vision = tracker
            .begin_inner(
                MaterializationComponents {
                    text: true,
                    vision: true,
                    speculative: false,
                },
                &governor,
                false,
            )
            .unwrap();
        assert_eq!(vision.reserved_bytes(), 2_048);
        vision.commit();
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn cold_materialization_drop_rolls_back_state_and_reservation() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        let tracker = ColdMaterializationTracker::new(StaticMemoryEstimate {
            text_cold_bytes: 4_096,
            ..StaticMemoryEstimate::default()
        });

        let guard = tracker
            .begin_inner(MaterializationComponents::text(), &governor, false)
            .unwrap();
        assert_eq!(governor.snapshot().reserved_bytes, 4_096);
        drop(guard);
        assert_eq!(governor.snapshot().reserved_bytes, 0);
        assert_eq!(
            tracker.state(MaterializationComponent::Text),
            ComponentMaterializationState::Cold
        );
    }

    #[test]
    fn cold_materialization_reservation_failure_restores_cold_state() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        let mut sample = telemetry(gib(4));
        sample.phys_footprint_bytes = None;
        sample.vm = None;
        governor.update(sample);
        let tracker = ColdMaterializationTracker::new(StaticMemoryEstimate {
            text_cold_bytes: 4_096,
            ..StaticMemoryEstimate::default()
        });

        assert!(matches!(
            tracker.begin_inner(MaterializationComponents::text(), &governor, false),
            Err(MemoryReservationError::TelemetryUnavailable { .. })
        ));
        assert_eq!(
            tracker.state(MaterializationComponent::Text),
            ComponentMaterializationState::Cold
        );
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_first_use_reserves_each_component_once() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        let tracker = ColdMaterializationTracker::new(StaticMemoryEstimate {
            text_cold_bytes: 8_192,
            ..StaticMemoryEstimate::default()
        });
        let first = tracker
            .begin_inner(MaterializationComponents::text(), &governor, false)
            .unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let follower_tracker = Arc::clone(&tracker);
        let follower_governor = Arc::clone(&governor);
        let follower = std::thread::spawn(move || {
            let guard = follower_tracker
                .begin_inner(MaterializationComponents::text(), &follower_governor, false)
                .unwrap();
            sender.send(guard.reserved_bytes()).unwrap();
            guard.commit();
        });

        assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
        assert_eq!(governor.snapshot().reserved_bytes, 8_192);
        first.commit();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 0);
        follower.join().unwrap();
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_reservations_preserve_global_budget_conservation() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        let barrier = Arc::new(std::sync::Barrier::new(9));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let governor = Arc::clone(&governor);
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let reservation = governor.try_reserve(1024, "concurrent").unwrap();
                barrier.wait();
                barrier.wait();
                drop(reservation);
            }));
        }
        barrier.wait();
        assert_eq!(governor.snapshot().reserved_bytes, 8 * 1024);
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_admission_never_overcommits_effective_headroom() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(4)));
        let barrier = Arc::new(std::sync::Barrier::new(33));
        let admitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..32 {
            let governor = Arc::clone(&governor);
            let barrier = Arc::clone(&barrier);
            let admitted = Arc::clone(&admitted);
            workers.push(std::thread::spawn(move || {
                let reservation = governor.try_reserve(gib(1), "concurrent_admission").ok();
                if reservation.is_some() {
                    admitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();
                barrier.wait();
                drop(reservation);
            }));
        }
        barrier.wait();
        let admitted = admitted.load(std::sync::atomic::Ordering::Relaxed);
        let snapshot = governor.snapshot();
        assert!(admitted > 0 && admitted < 32);
        assert_eq!(snapshot.reserved_bytes, admitted * gib(1));
        assert!(
            snapshot
                .current_usage_bytes
                .saturating_add(snapshot.reserved_bytes)
                <= ratio_bytes(
                    snapshot.effective_ceiling_bytes,
                    governor.config.prefill_headroom_ratio
                )
        );
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn prefill_plan_shrinks_and_minimum_chunk_refuses_safely() {
        let governor = Arc::new(ProcessMemoryGovernor::new(config()).unwrap());
        governor.update(telemetry(gib(20)));
        governor.update(telemetry(gib(20)));
        let meta = ModelMeta {
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            hidden_size: 4096,
            head_dim: Some(128),
            weight_bytes: 0,
            max_position_embeddings: 131_072,
            spatial_merge_size: 2,
        };
        let plan = governor
            .plan_prefill_chunk(4096, 32_768, 1, &meta)
            .expect("some chunk should fit");
        assert!(plan.selected_tokens < 4096);
        assert!(plan.selected_tokens >= 16);
        drop(plan);

        governor.update(telemetry(gib(23)));
        governor.update(telemetry(gib(23)));
        assert!(matches!(
            governor.plan_prefill_chunk(4096, 131_072, 1, &meta),
            Err(PrefillGuardError::MinimumChunkUnsafe { .. })
                | Err(PrefillGuardError::Reservation(
                    MemoryReservationError::Pressure { .. }
                ))
        ));
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[serial_test::serial(mlx_metal)]
    fn native_telemetry_tracks_physically_touched_process_memory() {
        let before = native_memory_telemetry();
        let before_footprint = before
            .phys_footprint_bytes
            .expect("macOS phys_footprint must be available");
        assert!(before.vm.is_some(), "macOS VM statistics must be available");
        assert!(
            before.metal_limit_bytes.is_some(),
            "Metal memory limit must be available"
        );

        let mut pressure = vec![0_u8; 64 * 1024 * 1024];
        for byte in pressure.iter_mut().step_by(4096) {
            *byte = 0xa5;
        }
        std::hint::black_box(&pressure);

        let after = native_memory_telemetry();
        let after_footprint = after
            .phys_footprint_bytes
            .expect("macOS phys_footprint must remain available");
        assert!(
            after_footprint >= before_footprint,
            "phys_footprint regressed under a live allocation: before={before_footprint} after={after_footprint}"
        );
        let governor = ProcessMemoryGovernor::new(MemoryGovernorConfig::default()).unwrap();
        let snapshot = governor.update(after);
        assert!(!snapshot.telemetry_degraded);
        assert!(snapshot.effective_ceiling_bytes > 0);
        drop(pressure);
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[serial_test::serial(mlx_metal)]
    fn real_physical_pressure_drives_emergency_and_stepwise_recovery() {
        let before = native_memory_telemetry();
        let baseline = before
            .current_usage_bytes()
            .expect("macOS process usage must be available");
        let total = before.total_ram_bytes;
        let recovery_ratio = 0.85_f64;
        let headroom = 16 * 1024 * 1024;
        let emergency_overage = 8 * 1024 * 1024;
        let desired_ceiling =
            ((baseline as f64 / recovery_ratio).ceil() as usize).saturating_add(headroom);
        assert!(
            desired_ceiling < total,
            "test process footprint leaves no safe room below total RAM"
        );

        let governor = ProcessMemoryGovernor::new(MemoryGovernorConfig {
            static_reserve_bytes: total - desired_ceiling,
            active_reclaim_ratio: 1.0,
            soft_ratio: 0.90,
            hard_ratio: 0.95,
            prefill_headroom_ratio: 0.90,
            recovery_ratio,
            promotion_samples: 1,
            recovery_samples: 1,
            emergency_overage_bytes: emergency_overage,
            minimum_prefill_chunk_tokens: 1,
            poll_interval: Duration::from_millis(1),
            telemetry_stale_after: Duration::from_secs(5),
            mlx_cache_ratio: 0.01,
            mlx_cache_min_bytes: 128 * MIB,
            mlx_cache_cold_max_bytes: 512 * MIB,
            mlx_cache_max_bytes: 512 * MIB,
        })
        .expect("valid real-pressure governor config");
        let initial = governor.update(before);
        assert_eq!(initial.pressure_level, PressureLevel::Normal);
        assert_eq!(initial.effective_ceiling_bytes, desired_ceiling);

        let allocation_bytes = desired_ceiling
            .saturating_add(emergency_overage)
            .saturating_add(headroom)
            .saturating_sub(baseline);
        assert!(
            allocation_bytes <= GIB,
            "real-pressure test refuses an unexpectedly large allocation: {allocation_bytes} bytes"
        );
        // SAFETY: anonymous private mapping with a validated non-zero length;
        // every touched byte remains within the mapping and munmap balances it.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                allocation_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(mapping, libc::MAP_FAILED, "mmap pressure allocation failed");
        let bytes = mapping.cast::<u8>();
        for offset in (0..allocation_bytes).step_by(4096) {
            // SAFETY: offset is bounded by allocation_bytes and the mapping is writable.
            unsafe { bytes.add(offset).write_volatile(0xa5) };
        }

        let pressured = governor.update(native_memory_telemetry());
        assert_eq!(pressured.pressure_level, PressureLevel::Emergency);
        assert!(
            pressured.current_usage_bytes
                >= pressured
                    .effective_ceiling_bytes
                    .saturating_add(emergency_overage),
            "real footprint did not cross emergency threshold: {pressured:?}"
        );

        // SAFETY: mapping and length exactly match the successful mmap above.
        assert_eq!(unsafe { libc::munmap(mapping, allocation_bytes) }, 0);
        let recovery_watermark = ratio_bytes(desired_ceiling, recovery_ratio);
        let deadline = Instant::now() + Duration::from_secs(5);
        let recovered_telemetry = loop {
            let sample = native_memory_telemetry();
            if sample
                .current_usage_bytes()
                .is_some_and(|usage| usage <= recovery_watermark)
            {
                break sample;
            }
            assert!(
                Instant::now() < deadline,
                "physical footprint did not recover below hysteresis watermark"
            );
            std::thread::sleep(Duration::from_millis(10));
        };

        assert_eq!(
            governor.update(recovered_telemetry).pressure_level,
            PressureLevel::Hard
        );
        assert_eq!(
            governor.update(recovered_telemetry).pressure_level,
            PressureLevel::Soft
        );
        assert_eq!(
            governor.update(recovered_telemetry).pressure_level,
            PressureLevel::Normal
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[serial_test::serial(mlx_metal)]
    fn native_sampling_advances_only_once_per_poll_interval() {
        let governor = ProcessMemoryGovernor::new(MemoryGovernorConfig {
            poll_interval: Duration::from_millis(50),
            telemetry_stale_after: Duration::from_secs(1),
            ..MemoryGovernorConfig::default()
        })
        .unwrap();

        let first = governor.sample_process();
        let immediate = governor.sample_process();
        assert_eq!(immediate.sample_sequence, first.sample_sequence);

        std::thread::sleep(Duration::from_millis(60));
        let next_interval = governor.sample_process();
        assert_eq!(
            next_interval.sample_sequence,
            first.sample_sequence.wrapping_add(1)
        );
    }
}
