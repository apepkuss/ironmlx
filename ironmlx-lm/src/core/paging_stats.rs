//! Model-owned paging telemetry. Snapshots never take the expert/PLE store lock
//! or evaluate MLX work. Counters are observational, not admission accounting.
use serde::Serialize;
use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy)]
pub enum PagingPhase {
    Prefill = 0,
    Decode = 1,
    Verify = 2,
}
thread_local! {
    static PHASE: Cell<PagingPhase> = const { Cell::new(PagingPhase::Prefill) };
    static PHASE_DEPTH:Cell<u32>=const {Cell::new(0)};
}
pub(crate) fn phase_hint() -> Option<PagingPhase> {
    PHASE_DEPTH.with(|d| (d.get() > 0).then(current_phase))
}
pub fn current_phase() -> PagingPhase {
    PHASE.with(|p| p.get())
}
pub fn is_prefill() -> bool {
    PHASE.with(|p| matches!(p.get(), PagingPhase::Prefill))
}
pub struct PhaseGuard(PagingPhase, u32, std::marker::PhantomData<std::rc::Rc<()>>);
impl PhaseGuard {
    pub fn enter(phase: PagingPhase) -> Self {
        Self(
            PHASE.with(|p| p.replace(phase)),
            PHASE_DEPTH.with(|d| {
                let old = d.get();
                d.set(old.saturating_add(1));
                old
            }),
            std::marker::PhantomData,
        )
    }
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        PHASE.with(|p| p.set(self.0));
        PHASE_DEPTH.with(|d| d.set(self.1));
    }
}

#[derive(Default)]
pub struct CacheCounters {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub read_bytes: AtomicU64,
    pub read_us: AtomicU64,
    pub pack_us: AtomicU64,
    pub lock_wait_us: AtomicU64,
    pub route_count: AtomicU64,
    pub batches: AtomicU64,
    pub microbatches: AtomicU64,
    pub budget_splits: AtomicU64,
    pub arithmetic_splits: AtomicU64,
    pub max_width: AtomicU64,
    pub request_width: AtomicU64,
    pub scan_admissions: AtomicU64,
    pub prefetch_hits: AtomicU64,
    pub prefetch_scheduled: AtomicU64,
    pub prefetch_unused: AtomicU64,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CachePhaseSnapshot {
    pub phase: &'static str,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: Option<f64>,
    pub read_bytes: u64,
    pub read_us: u64,
    pub pack_us: u64,
    pub lock_wait_us: u64,
    pub route_count: u64,
    pub batches: u64,
    pub microbatches: u64,
    pub budget_splits: u64,
    pub arithmetic_splits: u64,
    pub max_width: u64,
    pub request_width: u64,
    pub scan_admissions: u64,
    pub prefetch_hits: u64,
    pub prefetch_scheduled: u64,
    pub prefetch_unused: u64,
}
impl CacheCounters {
    fn snapshot(&self, phase: &'static str) -> CachePhaseSnapshot {
        let hits = self.hits.load(Relaxed);
        let misses = self.misses.load(Relaxed);
        CachePhaseSnapshot {
            phase,
            hits,
            misses,
            hit_rate: (hits + misses > 0).then(|| hits as f64 / (hits + misses) as f64),
            evictions: self.evictions.load(Relaxed),
            read_bytes: self.read_bytes.load(Relaxed),
            read_us: self.read_us.load(Relaxed),
            pack_us: self.pack_us.load(Relaxed),
            lock_wait_us: self.lock_wait_us.load(Relaxed),
            route_count: self.route_count.load(Relaxed),
            batches: self.batches.load(Relaxed),
            microbatches: self.microbatches.load(Relaxed),
            budget_splits: self.budget_splits.load(Relaxed),
            arithmetic_splits: self.arithmetic_splits.load(Relaxed),
            max_width: self.max_width.load(Relaxed),
            request_width: self.request_width.load(Relaxed),
            scan_admissions: self.scan_admissions.load(Relaxed),
            prefetch_hits: self.prefetch_hits.load(Relaxed),
            prefetch_scheduled: self.prefetch_scheduled.load(Relaxed),
            prefetch_unused: self.prefetch_unused.load(Relaxed),
        }
    }
}
#[derive(Default)]
pub struct CacheTelemetry {
    pub budget_bytes: AtomicU64,
    pub charged_bytes: AtomicU64,
    pub payload_bytes: AtomicU64,
    pub peak_charged_bytes: AtomicU64,
    pub entries: AtomicU64,
    pub pinned_bytes: AtomicU64,
    pub pending_bytes: AtomicU64,
    phases: [CacheCounters; 3],
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheSnapshot {
    pub budget_bytes: u64,
    pub charged_bytes: u64,
    pub payload_bytes: u64,
    pub peak_charged_bytes: u64,
    pub entries: u64,
    pub pinned_bytes: u64,
    pub pending_bytes: u64,
    pub phases: Vec<CachePhaseSnapshot>,
}
impl CacheTelemetry {
    pub fn current(&self) -> &CacheCounters {
        &self.phases[PHASE.with(|p| p.get()) as usize]
    }
    pub fn phase(&self, phase: PagingPhase) -> &CacheCounters {
        &self.phases[phase as usize]
    }
    pub fn snapshot(&self) -> CacheSnapshot {
        CacheSnapshot {
            budget_bytes: self.budget_bytes.load(Relaxed),
            charged_bytes: self.charged_bytes.load(Relaxed),
            payload_bytes: self.payload_bytes.load(Relaxed),
            peak_charged_bytes: self.peak_charged_bytes.load(Relaxed),
            entries: self.entries.load(Relaxed),
            pinned_bytes: self.pinned_bytes.load(Relaxed),
            pending_bytes: self.pending_bytes.load(Relaxed),
            phases: ["prefill", "decode", "verify"]
                .into_iter()
                .enumerate()
                .map(|(i, p)| self.phases[i].snapshot(p))
                .collect(),
        }
    }
}
pub struct PagingTelemetry {
    instance_id: String,
    started: Instant,
    started_unix_ms: u64,
    pub mode: &'static str,
    pub row_fallback_disabled: AtomicU64,
    pub row_fallback_unqualified: AtomicU64,
    pub row_fallback_verify_order: AtomicU64,
    pub experts: CacheTelemetry,
    pub ple: CacheTelemetry,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PagingFallback {
    pub reason: &'static str,
    pub calls: u64,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PagingSnapshot {
    pub instance_id: String,
    pub fallbacks: Vec<PagingFallback>,
    pub mode: &'static str,
    pub started_unix_ms: u64,
    pub sampled_unix_ms: u64,
    pub elapsed_ms: u64,
    pub experts: CacheSnapshot,
    pub ple: CacheSnapshot,
}
pub fn elapsed_us(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u64::MAX as u128) as u64
}
impl PagingTelemetry {
    pub fn new(paged: bool) -> Self {
        Self {
            instance_id: instance_id(),
            started: Instant::now(),
            started_unix_ms: unix_ms(),
            mode: if paged { "ssd" } else { "full" },
            row_fallback_disabled: AtomicU64::new(0),
            row_fallback_unqualified: AtomicU64::new(0),
            row_fallback_verify_order: AtomicU64::new(0),
            experts: CacheTelemetry::default(),
            ple: CacheTelemetry::default(),
        }
    }
    pub fn snapshot(&self) -> PagingSnapshot {
        PagingSnapshot {
            fallbacks: vec![
                PagingFallback {
                    reason: "gpu_not_qualified",
                    calls: self.row_fallback_unqualified.load(Relaxed),
                },
                PagingFallback {
                    reason: "native_batching_disabled",
                    calls: self.row_fallback_disabled.load(Relaxed),
                },
                PagingFallback {
                    reason: "token_major_verifier",
                    calls: self.row_fallback_verify_order.load(Relaxed),
                },
            ],
            instance_id: self.instance_id.clone(),
            mode: self.mode,
            started_unix_ms: self.started_unix_ms,
            sampled_unix_ms: unix_ms(),
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            experts: self.experts.snapshot(),
            ple: self.ple.snapshot(),
        }
    }
}
fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_isolation_and_empty_rate() {
        assert!(phase_hint().is_none());
        let s = PagingTelemetry::new(true);
        assert_eq!(s.snapshot().experts.phases[0].hit_rate, None);
        {
            let _guard = PhaseGuard::enter(PagingPhase::Verify);
            assert!(matches!(phase_hint(), Some(PagingPhase::Verify)));
            s.experts.current().hits.fetch_add(2, Relaxed);
        }
        s.experts.current().misses.fetch_add(1, Relaxed);
        let p = s.snapshot();
        assert_eq!(p.experts.phases[2].hits, 2);
        assert_eq!(p.experts.phases[0].misses, 1);
        assert_eq!(p.experts.phases[2].hit_rate, Some(1.0));
        assert_ne!(
            p.instance_id,
            PagingTelemetry::new(true).snapshot().instance_id
        );
    }
}

fn instance_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT.fetch_add(1, Relaxed)
    )
}
