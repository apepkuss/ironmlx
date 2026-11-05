//! Flash load admission shared by the native App engine and CLI entrypoints.
use anyhow::Result;
use ironmlx_lm::models::qwen4_exp::load_plan::{FlashLoadPlan, FlashLoadRequest};
use std::path::Path;
use std::time::{Duration, Instant};

pub fn is_flash(path: &Path) -> Result<bool> {
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path.join("config.json"))?)?;
    Ok(config["model_type"] == "qwen4_exp")
}
pub fn plan(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
) -> Result<FlashLoadPlan> {
    plan_with_occupied(path, batch, cap, chunk, draft_bytes, limit, 0)
}

/// EnginePool's planned resident ceilings include bounded expert/PLE caches
/// which may not have materialized yet. The process sample remains a floor.
pub(crate) fn available_bytes(minimum_occupied: usize, limit: Option<usize>) -> Result<usize> {
    Ok(admission_budgets(minimum_occupied, limit)?.now)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmissionBudgets {
    now: usize,
    /// The same budget once the host limit no longer binds. Never admitted
    /// directly; it only separates "not yet" from "never" on this machine.
    host_recovered: usize,
}

fn admission_budgets(minimum_occupied: usize, limit: Option<usize>) -> Result<AdmissionBudgets> {
    let physical = super::memory_budget::system_total_ram_bytes();
    let memory = mlx::memory::snapshot();
    // A just-unloaded model may have released its buffers within the poll
    // interval. Admission must not reuse that model's cached footprint.
    let governor = super::process_memory::global_process_memory_governor();
    let telemetry = governor.refresh_process();
    anyhow::ensure!(
        !telemetry.telemetry_degraded
            && !matches!(
                telemetry.pressure_level,
                super::process_memory::PressureLevel::Hard
                    | super::process_memory::PressureLevel::Emergency
            ),
        "Flash loading requires available process and host memory telemetry without hard pressure"
    );
    // Both observed process use (other loaded models) and outstanding
    // reservations reduce this load's envelope. Never count reclaimable
    // filesystem pages as an additional copy of physical memory.
    let occupied = telemetry
        .current_usage_bytes
        .max(memory.active_bytes)
        .max(minimum_occupied)
        .saturating_add(telemetry.reserved_bytes);
    let device_ceiling = physical.saturating_sub(8usize << 30).min(
        memory
            .max_recommended_bytes
            .unwrap_or(physical)
            .saturating_sub(2usize << 30),
    );
    let budget = |watermark: usize| {
        device_ceiling
            .min(if watermark > 0 { watermark } else { physical })
            .saturating_sub(occupied)
            .min(limit.unwrap_or(usize::MAX))
    };
    Ok(AdmissionBudgets {
        now: budget(telemetry.hard_watermark_bytes),
        host_recovered: budget(governor.hard_watermark_without_host_limit(&telemetry)),
    })
}

/// One Flash load's admission inputs, re-planned against fresh telemetry.
pub(crate) struct FlashPlanRequest<'a> {
    pub path: &'a Path,
    pub batch: usize,
    pub cap: usize,
    pub chunk: usize,
    pub draft_bytes: u64,
    pub limit: Option<usize>,
    pub minimum_occupied: usize,
    pub active: &'a super::cache::ActiveKvOffloadConfig,
}

impl FlashPlanRequest<'_> {
    fn inspect(&self, available: usize) -> Result<FlashLoadPlan> {
        let mut request = FlashLoadRequest::new(
            available as u64,
            self.batch,
            self.cap,
            self.chunk,
            self.draft_bytes,
        );
        if self.active.enabled {
            request.active_pages = Some(
                ironmlx_lm::core::active_pages::ActivePagesConfig::for_memory(
                    self.active.root.join("flash"),
                    available as u64,
                ),
            );
        }
        FlashLoadPlan::inspect(self.path, &request.with_environment()?)
    }

    pub(crate) fn admission(&self) -> FlashAdmission<'_> {
        FlashAdmission {
            request: self,
            wait: None,
        }
    }
}

/// A process that just exited keeps its pages until the kernel finishes tearing
/// it down: on M5 Max 128 GiB wired memory falls for 0.3-0.5 s after `waitpid`
/// returns, so an immediate successor sees a host that is only temporarily full.
pub(crate) const ADMISSION_POLL: Duration = Duration::from_millis(100);
const ADMISSION_DEADLINE: Duration = Duration::from_secs(30);
/// Reclaim that stops is no longer "temporary". A loadable, weaker plan
/// waits less than a load that would otherwise fail.
pub(crate) const ADMISSION_PATIENCE_REJECTED: Duration = Duration::from_secs(5);
const ADMISSION_PATIENCE_DEGRADED: Duration = Duration::from_millis(1500);
const ADMISSION_PROGRESS_BYTES: usize = 256 << 20;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WaitVerdict {
    Continue,
    Expired,
}

/// Bounded by a total deadline and by how long the budget may stop growing.
#[derive(Debug)]
pub(crate) struct RecoveryWait {
    started: Instant,
    last_progress: Instant,
    first_budget: usize,
    best_budget: usize,
}

impl RecoveryWait {
    pub(crate) fn new(budget: usize, now: Instant) -> Self {
        Self {
            started: now,
            last_progress: now,
            first_budget: budget,
            best_budget: budget,
        }
    }

    pub(crate) fn observe(
        &mut self,
        budget: usize,
        patience: Duration,
        now: Instant,
    ) -> WaitVerdict {
        if budget >= self.best_budget.saturating_add(ADMISSION_PROGRESS_BYTES) {
            self.best_budget = budget;
            self.last_progress = now;
        }
        if now - self.started >= ADMISSION_DEADLINE || now - self.last_progress >= patience {
            WaitVerdict::Expired
        } else {
            WaitVerdict::Continue
        }
    }
}

/// How one planning attempt relates to the budget after host recovery.
enum AdmissionAttempt<P> {
    Ready(P),
    /// Would fit once the host recovers. `degraded` is the weaker plan that
    /// fits right now, when Auto found one.
    HostShort {
        error: Option<anyhow::Error>,
        degraded: Option<P>,
    },
    Rejected(anyhow::Error),
}

fn classify_attempt<P>(
    budgets: AdmissionBudgets,
    inspect: impl Fn(usize) -> Result<P>,
    residency_rank: impl Fn(&P) -> u8,
) -> AdmissionAttempt<P> {
    match inspect(budgets.now) {
        Ok(plan) if budgets.host_recovered <= budgets.now => AdmissionAttempt::Ready(plan),
        Ok(plan) => match inspect(budgets.host_recovered) {
            // Auto would move experts to SSD for this process's lifetime. A
            // smaller SSD cache alone is the normal state of a busy small Mac.
            Ok(better) if residency_rank(&better) > residency_rank(&plan) => {
                AdmissionAttempt::HostShort {
                    error: None,
                    degraded: Some(plan),
                }
            }
            _ => AdmissionAttempt::Ready(plan),
        },
        Err(error) => match inspect(budgets.host_recovered) {
            Ok(_) => AdmissionAttempt::HostShort {
                error: Some(error),
                degraded: None,
            },
            // Model errors and machines too small for this load do not recover.
            Err(_) => AdmissionAttempt::Rejected(error),
        },
    }
}

pub(crate) enum AdmissionStep {
    Done(Result<FlashLoadPlan>),
    /// Temporarily short of host memory; poll again after this delay.
    Wait(Duration),
}

/// Admission that tolerates a host still reclaiming another process's memory.
/// Every poll re-plans against a fresh native sample, so nothing is admitted
/// beyond what the unchanged plan accepts. The caller owns sleeping, and thus
/// cancellation.
pub(crate) struct FlashAdmission<'a> {
    request: &'a FlashPlanRequest<'a>,
    wait: Option<RecoveryWait>,
}

impl FlashAdmission<'_> {
    pub(crate) fn poll(&mut self) -> AdmissionStep {
        let budgets = match admission_budgets(self.request.minimum_occupied, self.request.limit) {
            Ok(budgets) => budgets,
            Err(error) => return AdmissionStep::Done(Err(error)),
        };
        let attempt = classify_attempt(
            budgets,
            |available| self.request.inspect(available),
            |plan: &FlashLoadPlan| match plan.residency.mode {
                ironmlx_lm::core::residency::ExpertResidency::Ssd => 0,
                _ => 1,
            },
        );
        match attempt {
            AdmissionAttempt::Ready(plan) => {
                if let Some(wait) = &self.wait {
                    tracing::info!(
                        waited_ms = wait.started.elapsed().as_millis() as u64,
                        first_budget = wait.first_budget,
                        budget = budgets.now,
                        "Flash admission recovered with host memory"
                    );
                }
                AdmissionStep::Done(Ok(plan))
            }
            AdmissionAttempt::Rejected(error) => AdmissionStep::Done(Err(error)),
            AdmissionAttempt::HostShort { error, degraded } => {
                let now = Instant::now();
                let patience = if degraded.is_some() {
                    ADMISSION_PATIENCE_DEGRADED
                } else {
                    ADMISSION_PATIENCE_REJECTED
                };
                let wait = self.wait.get_or_insert_with(|| {
                    tracing::info!(
                        budget = budgets.now,
                        budget_after_recovery = budgets.host_recovered,
                        "Flash admission waiting for host memory to recover"
                    );
                    RecoveryWait::new(budgets.now, now)
                });
                if wait.observe(budgets.now, patience, now) == WaitVerdict::Continue {
                    return AdmissionStep::Wait(ADMISSION_POLL);
                }
                let waited_ms = wait.started.elapsed().as_millis() as u64;
                match (degraded, error) {
                    (Some(plan), _) => {
                        tracing::warn!(
                            waited_ms,
                            budget = budgets.now,
                            budget_after_recovery = budgets.host_recovered,
                            "Flash host memory did not recover; using the plan that fits now"
                        );
                        AdmissionStep::Done(Ok(plan))
                    }
                    (None, Some(error)) => AdmissionStep::Done(Err(error.context(format!(
                        "host memory did not recover within {waited_ms} ms (budget {} -> {} bytes)",
                        wait.first_budget, budgets.now
                    )))),
                    (None, None) => unreachable!("host-short attempts keep a plan or an error"),
                }
            }
        }
    }
}

/// Blocking admission for process startup, where SIGTERM is the cancellation.
pub fn plan_waiting_for_host_recovery(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    active: &super::cache::ActiveKvOffloadConfig,
) -> Result<FlashLoadPlan> {
    let request = FlashPlanRequest {
        path,
        batch,
        cap,
        chunk,
        draft_bytes,
        limit: None,
        minimum_occupied: 0,
        active,
    };
    let mut admission = request.admission();
    loop {
        match admission.poll() {
            AdmissionStep::Done(result) => return result,
            AdmissionStep::Wait(delay) => std::thread::sleep(delay),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_with_occupied(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
    minimum_occupied: usize,
) -> Result<FlashLoadPlan> {
    plan_with_active_occupied(
        path,
        batch,
        cap,
        chunk,
        draft_bytes,
        limit,
        minimum_occupied,
        &super::cache::ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_with_active_occupied(
    path: &Path,
    batch: usize,
    cap: usize,
    chunk: usize,
    draft_bytes: u64,
    limit: Option<usize>,
    minimum_occupied: usize,
    active: &super::cache::ActiveKvOffloadConfig,
) -> Result<FlashLoadPlan> {
    let available = available_bytes(minimum_occupied, limit)?;
    let mut request = FlashLoadRequest::new(available as u64, batch, cap, chunk, draft_bytes);
    if active.enabled {
        request.active_pages = Some(
            ironmlx_lm::core::active_pages::ActivePagesConfig::for_memory(
                active.root.join("flash"),
                available as u64,
            ),
        );
    }
    FlashLoadPlan::inspect(path, &request.with_environment()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets(now: usize, host_recovered: usize) -> AdmissionBudgets {
        AdmissionBudgets {
            now,
            host_recovered,
        }
    }

    // Plans are their residency rank: full needs 100 bytes, Auto falls back to SSD.
    fn strict(available: usize) -> Result<u8> {
        anyhow::ensure!(available >= 100, "needs 100, has {available}");
        Ok(1)
    }

    fn auto(available: usize) -> Result<u8> {
        Ok(u8::from(available >= 100))
    }

    fn state<P>(attempt: AdmissionAttempt<P>) -> &'static str {
        match attempt {
            AdmissionAttempt::Ready(_) => "ready",
            AdmissionAttempt::HostShort {
                degraded: Some(_), ..
            } => "degraded",
            AdmissionAttempt::HostShort { .. } => "short",
            AdmissionAttempt::Rejected(_) => "rejected",
        }
    }

    #[test]
    fn only_host_shortage_that_would_fit_after_recovery_can_wait() {
        let size = |p: &u8| *p;
        assert_eq!(
            state(classify_attempt(budgets(120, 120), strict, size)),
            "ready"
        );
        assert_eq!(
            state(classify_attempt(budgets(120, 200), strict, size)),
            "ready"
        );
        assert_eq!(
            state(classify_attempt(budgets(40, 120), strict, size)),
            "short"
        );
        // Too big for this machine: failing now must not turn into waiting.
        assert_eq!(
            state(classify_attempt(budgets(40, 80), strict, size)),
            "rejected"
        );
        // Model and config errors fail the same way under any budget.
        let broken = |_: usize| -> Result<u8> { anyhow::bail!("corrupt model") };
        assert_eq!(
            state(classify_attempt(budgets(40, 120), broken, size)),
            "rejected"
        );
    }

    #[test]
    fn auto_waits_before_settling_for_a_weaker_plan_but_never_for_a_small_machine() {
        let size = |p: &u8| *p;
        assert_eq!(
            state(classify_attempt(budgets(40, 120), auto, size)),
            "degraded"
        );
        assert_eq!(
            state(classify_attempt(budgets(120, 200), auto, size)),
            "ready"
        );
        // SSD either way once capacity itself is the limit.
        assert_eq!(
            state(classify_attempt(budgets(40, 80), auto, size)),
            "ready"
        );
    }

    #[test]
    fn recovery_wait_is_bounded_by_stalled_reclaim_and_by_the_deadline() {
        let start = Instant::now();
        let patience = ADMISSION_PATIENCE_REJECTED;
        let mut wait = RecoveryWait::new(1 << 30, start);
        assert_eq!(
            wait.observe(1 << 30, patience, start),
            WaitVerdict::Continue
        );
        assert_eq!(
            wait.observe(1 << 30, patience, start + patience),
            WaitVerdict::Expired,
            "a budget that stopped growing is not temporary"
        );

        // Growth below the progress threshold does not extend the wait.
        let mut wait = RecoveryWait::new(1 << 30, start);
        let noise = (1 << 30) + ADMISSION_PROGRESS_BYTES - 1;
        assert_eq!(
            wait.observe(noise, patience, start + patience),
            WaitVerdict::Expired
        );

        // Real progress extends it, but never past the deadline.
        let mut wait = RecoveryWait::new(1 << 30, start);
        let mut budget = 1usize << 30;
        let mut at = start;
        let step = Duration::from_secs(4);
        loop {
            at += step;
            budget += 1 << 30;
            if wait.observe(budget, patience, at) == WaitVerdict::Expired {
                break;
            }
        }
        assert!(at - start >= ADMISSION_DEADLINE);
        assert!(at - start < ADMISSION_DEADLINE + step);
    }
}
