//! Draft-budget selection for the qualified Qwen3.6 MoE DFlash2 B1 linear path.
//!
//! DFlash2 proposes a whole block with one draft forward, so the draft cost
//! changes little with the budget while the target verify widens with it. The
//! window-cost policy therefore picks the budget whose complete window (draft,
//! verify, projection, sampling, commit/rollback) costs the least per committed
//! token, from two estimates:
//! - acceptance per draft depth, pooled over every speculative window with
//!   exponential forgetting per drafted window: only new drafted evidence
//!   replaces old evidence, an ordinary window neither adds nor ages it. A
//!   rejection at depth `k` lowers only the estimate for depth `k`, so one
//!   partial accept does not disqualify a wider budget;
//! - the measured window time per budget, linearly interpolated or
//!   extrapolated across measured budgets. Timing noise only adds time (first
//!   use of a shape, contention), so a sample below the estimate replaces it
//!   while higher samples move it by an EWMA step. A drafting window right
//!   after prefill or an ordinary window also drafts over the target context
//!   those produced; that one-time cost is not the budget's steady cost, so
//!   such a window contributes acceptance but no cost sample.
//!
//! Exploration is bounded. Each cost regime starts with a short calibration at
//! the widest budget, budget 1, their midpoint and the ordinary window, two
//! cost samples each: the first window at a width can include a one-off
//! process-wide cost (first use of a shape), which the second, lower sample
//! replaces.
//! Afterwards single-window probes of a neighbouring budget run on an interval
//! that doubles while probes do not change the choice. Probe directions
//! alternate, so a budget-0 probe cannot crowd out upward exploration.
//!
//! Budget 0 is not final. While it is chosen, recovery probes draft
//! `RECOVERY_PROBE_WINDOWS` consecutive windows at the best drafting budget
//! (the first consumes the accumulated context, the next ones measure the
//! cost) whenever an exploration credit covers their predicted worst case. Ordinary windows earn
//! `RECOVERY_SHARE` of their time as credit; each probe window spends its measured time beyond what
//! ordinary decoding would have taken for the same tokens, so a probe that pays off earns credit and the
//! next one follows at once. A probe also runs after `MAX_PROBE_INTERVAL` ordinary windows in a row,
//! whatever the credit, which bounds the raw target context the stream keeps for the draft cache. So
//! the exploration at budget 0 is at most `RECOVERY_SHARE` of the ordinary time + the measured cost of
//! one credit-started probe + the measured cost of the forced probes (at most one per interval), which
//! the credit does not cover: the price of the context bound when drafting windows are expensive.
//!
//! The Qwen MTP policy (`QwenMtpDraftPolicyState`) is unchanged and still
//! serves every other DFlash2 target and execution shape.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, Weak};

use anyhow::{anyhow, Result};

use super::speculative::{
    MtpDraftBudgetChange, MtpDraftPolicyRegime, MtpDraftPolicyWindow, QwenMtpDraftPolicyState,
};

/// Diagnostic override of the budget policy for targets qualified for the
/// window-cost policy: `legacy` keeps the shared Qwen MTP policy.
pub(crate) const BUDGET_POLICY_SETTING: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_BUDGET_POLICY";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DFlash2BudgetPolicyKind {
    /// The shared Qwen MTP policy.
    Legacy,
    /// The DFlash2 window-cost policy.
    WindowCost,
}

impl DFlash2BudgetPolicyKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::WindowCost => "window-cost",
        }
    }

    /// Policy for a stream: the window-cost policy only for targets qualified
    /// for it on the linear path (no flat tree configured), unless the
    /// diagnostic override asks for the legacy policy.
    pub(crate) fn resolve(qualified_target: bool, tree_configured: bool) -> Result<Self> {
        Self::select(
            std::env::var(BUDGET_POLICY_SETTING).ok().as_deref(),
            qualified_target,
            tree_configured,
        )
    }

    fn select(
        requested: Option<&str>,
        qualified_target: bool,
        tree_configured: bool,
    ) -> Result<Self> {
        match requested.map(str::trim) {
            None | Some("") => Ok(if qualified_target && !tree_configured {
                Self::WindowCost
            } else {
                Self::Legacy
            }),
            Some("legacy") => Ok(Self::Legacy),
            Some("window-cost") if qualified_target && !tree_configured => Ok(Self::WindowCost),
            Some("window-cost") => Err(anyhow!(
                "{BUDGET_POLICY_SETTING}=window-cost needs a qualified Qwen3.6 MoE target on the linear path"
            )),
            Some(other) => Err(anyhow!(
                "{BUDGET_POLICY_SETTING} must be `legacy` or `window-cost`, got `{other}`"
            )),
        }
    }
}

/// Why a window ran at its budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DFlash2WindowRole {
    Exploit,
    Calibrate,
    Probe,
}

#[derive(Debug, Clone)]
pub(crate) enum DFlash2DraftBudgetPolicy {
    Legacy(QwenMtpDraftPolicyState),
    WindowCost(DFlash2WindowCostPolicy),
}

impl DFlash2DraftBudgetPolicy {
    #[cfg(test)]
    pub(crate) fn new(kind: DFlash2BudgetPolicyKind, max_draft_tokens: usize) -> Self {
        Self::with_memory(kind, max_draft_tokens, None)
    }

    /// Policy for one stream. With a `domain`, the window-cost policy shares
    /// what does not depend on the request with the other streams of the
    /// same target/drafter pair (see [`DFlash2PolicyDomain`]).
    pub(crate) fn with_memory(
        kind: DFlash2BudgetPolicyKind,
        max_draft_tokens: usize,
        domain: Option<DFlash2PolicyDomain>,
    ) -> Self {
        match kind {
            DFlash2BudgetPolicyKind::Legacy => {
                Self::Legacy(QwenMtpDraftPolicyState::new(max_draft_tokens))
            }
            DFlash2BudgetPolicyKind::WindowCost => Self::WindowCost(
                DFlash2WindowCostPolicy::with_memory(max_draft_tokens, domain),
            ),
        }
    }

    pub(crate) fn kind(&self) -> DFlash2BudgetPolicyKind {
        match self {
            Self::Legacy(_) => DFlash2BudgetPolicyKind::Legacy,
            Self::WindowCost(_) => DFlash2BudgetPolicyKind::WindowCost,
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        match self {
            Self::Legacy(policy) => policy.current_budget(),
            Self::WindowCost(policy) => policy.next_budget(),
        }
    }

    /// Role of the window that runs at [`Self::current_budget`].
    pub(crate) fn window_role(&self) -> DFlash2WindowRole {
        match self {
            Self::Legacy(policy) => {
                if policy.probe_budget() == Some(policy.current_budget()) {
                    DFlash2WindowRole::Probe
                } else {
                    DFlash2WindowRole::Exploit
                }
            }
            Self::WindowCost(policy) => policy.role,
        }
    }

    pub(crate) fn observe(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        match self {
            Self::Legacy(policy) => policy.observe_external_window(window),
            Self::WindowCost(policy) => policy.observe(window),
        }
    }

    pub(crate) fn acceptance_ewma(&self) -> Option<f64> {
        match self {
            Self::Legacy(policy) => policy.acceptance_ewma(),
            Self::WindowCost(policy) => policy.acceptance_ewma,
        }
    }

    /// The legacy policy treats a settled budget 0 as final.
    pub(crate) fn uses_ordinary_decode(&self) -> bool {
        match self {
            Self::Legacy(policy) => policy.uses_ordinary_decode(),
            Self::WindowCost(_) => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn should_maintain_mtp_cache(&self) -> bool {
        match self {
            Self::Legacy(policy) => policy.should_maintain_mtp_cache(),
            Self::WindowCost(policy) => policy.next_budget() > 0,
        }
    }
}

/// A loaded target/drafter pair. Window costs are a property of the pair,
/// the execution regime and the hardware, not of the request, so the streams
/// of one pair share them: a new request reuses the measured costs instead of
/// calibrating them again. Each stream also starts its depth acceptance from a
/// small pooled prior of the earlier streams instead of a constant.
///
/// The pair is the two loaded instances (`DFlash2Instance` of the target and
/// the drafter), not their addresses: its memory lives while both instances
/// do, a reload starts from scratch, and the memory of an unloaded pair is
/// dropped at the next access.
#[derive(Debug, Clone)]
pub(crate) struct DFlash2PolicyDomain {
    target: Weak<()>,
    draft: Weak<()>,
}

impl DFlash2PolicyDomain {
    pub(crate) fn new(target: Weak<()>, draft: Weak<()>) -> Self {
        Self { target, draft }
    }

    fn is_live(&self) -> bool {
        self.target.strong_count() > 0 && self.draft.strong_count() > 0
    }

    /// Unique among the pairs in the memory: every entry keeps weak handles
    /// to its instances, so their allocations (and addresses) cannot be
    /// reused while the entry exists.
    fn key(&self) -> (usize, usize) {
        (self.target.as_ptr() as usize, self.draft.as_ptr() as usize)
    }
}

#[derive(Default)]
struct PairMemory {
    cost: HashMap<(MtpDraftPolicyRegime, usize), Vec<CostStat>>,
    depth: HashMap<usize, Vec<DepthStat>>,
}

struct PairEntry {
    domain: DFlash2PolicyDomain,
    memory: PairMemory,
}

#[derive(Default)]
struct SharedPolicyMemory {
    pairs: HashMap<(usize, usize), PairEntry>,
}

impl SharedPolicyMemory {
    /// The memory of a live pair, after dropping the pairs that were unloaded.
    fn pair(&mut self, domain: &DFlash2PolicyDomain) -> Option<&mut PairMemory> {
        self.pairs.retain(|_, entry| entry.domain.is_live());
        if !domain.is_live() {
            return None;
        }
        Some(
            &mut self
                .pairs
                .entry(domain.key())
                .or_insert_with(|| PairEntry {
                    domain: domain.clone(),
                    memory: PairMemory::default(),
                })
                .memory,
        )
    }
}

fn shared_memory() -> std::sync::MutexGuard<'static, SharedPolicyMemory> {
    static MEMORY: OnceLock<Mutex<SharedPolicyMemory>> = OnceLock::new();
    MEMORY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Pairs held in the shared memory after dropping unloaded ones.
#[cfg(test)]
fn shared_pair_count() -> usize {
    let mut memory = shared_memory();
    memory.pairs.retain(|_, entry| entry.domain.is_live());
    memory.pairs.len()
}

#[derive(Debug, Clone, Copy, Default)]
struct DepthStat {
    reached: f64,
    accepted: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CostStat {
    window_us: f64,
    samples: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DFlash2WindowCostPolicy {
    max_draft_tokens: usize,
    /// Budget chosen for exploitation.
    current: usize,
    /// Budget of the next window: `current`, or a calibration/probe budget.
    next: usize,
    role: DFlash2WindowRole,
    regime: Option<MtpDraftPolicyRegime>,
    /// Acceptance of depth `k + 1`.
    depth: Vec<DepthStat>,
    /// Window cost of budget `b` in the active regime.
    cost: Vec<CostStat>,
    windows_since_probe: usize,
    probe_interval: usize,
    probe_up_next: bool,
    acceptance_ewma: Option<f64>,
    /// Drafted width of the previous window; `None` before the first.
    previous_width: Option<usize>,
    /// Consecutive ordinary windows up to the last observed one.
    ordinary_run: usize,
    /// Exploration time available to recovery probes while budget 0 is chosen.
    recovery_credit_us: f64,
    /// Budget and remaining windows of the recovery probe in progress.
    recovery_budget: usize,
    recovery_windows_left: usize,
    domain: Option<DFlash2PolicyDomain>,
}

impl DFlash2WindowCostPolicy {
    const COST_ALPHA: f64 = 0.3;
    const ACCEPTANCE_ALPHA: f64 = 0.35;
    /// Forgetting applied to depth statistics per speculative window.
    const DEPTH_DECAY: f64 = 0.95;
    /// Pseudo-count of the depth prior: the previous depth's estimate, or one
    /// half for depth 1.
    const DEPTH_PRIOR_WEIGHT: f64 = 2.0;
    const FIRST_DEPTH_PRIOR: f64 = 0.5;
    /// Relative cost improvement required to leave the current budget.
    const SWITCH_MARGIN: f64 = 0.03;
    /// Cost samples per calibration budget.
    const CALIBRATION_SAMPLES: usize = 2;
    const INITIAL_PROBE_INTERVAL: usize = 8;
    const MAX_PROBE_INTERVAL: usize = 64;
    /// Share of ordinary-window time that credit-started recovery probes may spend at budget 0
    /// (probes forced by the context bound come on top); below the switch margin, so this
    /// exploration costs less than the hysteresis the policy already accepts.
    const RECOVERY_SHARE: f64 = 0.02;
    /// Windows per recovery probe: one consumes the accumulated context, the
    /// rest measure the budget's steady cost.
    const RECOVERY_PROBE_WINDOWS: usize = 2;
    /// Forgetting of the pooled depth statistics per speculative window.
    const SHARED_DEPTH_DECAY: f64 = 0.99;
    /// Depth-1 windows the pooled prior is worth in a new stream.
    const SHARED_PRIOR_WINDOWS: f64 = 4.0;

    #[cfg(test)]
    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        Self::with_memory(max_draft_tokens, None)
    }

    pub(crate) fn with_memory(
        max_draft_tokens: usize,
        domain: Option<DFlash2PolicyDomain>,
    ) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        let mut depth = vec![DepthStat::default(); max_draft_tokens];
        if let Some(pooled) = domain.as_ref().and_then(|domain| {
            shared_memory()
                .pair(domain)?
                .depth
                .get(&max_draft_tokens)
                .cloned()
        }) {
            let scale = if pooled[0].reached > Self::SHARED_PRIOR_WINDOWS {
                Self::SHARED_PRIOR_WINDOWS / pooled[0].reached
            } else {
                1.0
            };
            for (depth, pooled) in depth.iter_mut().zip(&pooled) {
                depth.reached = pooled.reached * scale;
                depth.accepted = pooled.accepted * scale;
            }
        }
        let mut policy = Self {
            max_draft_tokens,
            current: max_draft_tokens,
            next: max_draft_tokens,
            role: DFlash2WindowRole::Calibrate,
            regime: None,
            depth,
            cost: vec![CostStat::default(); max_draft_tokens + 1],
            windows_since_probe: 0,
            probe_interval: Self::INITIAL_PROBE_INTERVAL,
            probe_up_next: true,
            acceptance_ewma: None,
            previous_width: None,
            ordinary_run: 0,
            recovery_credit_us: 0.0,
            recovery_budget: 1,
            recovery_windows_left: 0,
            domain,
        };
        policy.plan_next(false);
        policy
    }

    pub(crate) fn next_budget(&self) -> usize {
        self.next.min(self.max_draft_tokens)
    }

    /// Calibration budgets in run order: the widest budget (deepest
    /// acceptance evidence), budget 1, their midpoint (a cost line through
    /// three points) and one ordinary window (budget 0 cannot be predicted
    /// from speculative windows).
    fn calibration_budgets(&self) -> Vec<usize> {
        let max = self.max_draft_tokens;
        let mut unique = Vec::new();
        for budget in [max, 1, max.div_ceil(2), 0] {
            if !unique.contains(&budget) {
                unique.push(budget);
            }
        }
        unique
    }

    fn pending_calibration(&self) -> Option<usize> {
        self.calibration_budgets()
            .into_iter()
            .find(|&budget| self.cost[budget].samples < Self::CALIBRATION_SAMPLES)
    }

    pub(crate) fn observe(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        let before = self.next_budget();
        let observed_role = self.role;
        let regime = window.regime();
        if self.regime != Some(regime) {
            // Window costs depend on the context bucket and execution shape;
            // acceptance follows the content and is kept.
            self.regime = Some(regime);
            self.cost = self
                .domain
                .as_ref()
                .and_then(|domain| {
                    shared_memory()
                        .pair(domain)?
                        .cost
                        .get(&(regime, self.max_draft_tokens))
                        .cloned()
                })
                .unwrap_or_else(|| vec![CostStat::default(); self.max_draft_tokens + 1]);
            self.windows_since_probe = 0;
            self.probe_interval = Self::INITIAL_PROBE_INTERVAL;
            self.recovery_windows_left = 0;
        }
        let width = window.attempted_draft_tokens.min(self.max_draft_tokens);
        let measured = window.measured_window_us() as f64;
        let catch_up = width > 0 && self.previous_width.is_none_or(|previous| previous == 0);
        if !catch_up {
            let cost = &mut self.cost[width];
            cost.window_us = if cost.samples == 0 || measured < cost.window_us {
                measured
            } else {
                measured.mul_add(Self::COST_ALPHA, cost.window_us * (1.0 - Self::COST_ALPHA))
            };
            cost.samples = cost.samples.saturating_add(1);
            let cost = *cost;
            if let Some(domain) = &self.domain {
                if let Some(pair) = shared_memory().pair(domain) {
                    pair.cost
                        .entry((regime, self.max_draft_tokens))
                        .or_insert_with(|| vec![CostStat::default(); self.max_draft_tokens + 1])
                        [width] = cost;
                }
            }
        }
        if width > 0 {
            let accepted = window.accepted_draft_tokens.min(width);
            self.record_depths(width, accepted);
            if let Some(domain) = &self.domain {
                if let Some(pair) = shared_memory().pair(domain) {
                    let pooled = pair
                        .depth
                        .entry(self.max_draft_tokens)
                        .or_insert_with(|| vec![DepthStat::default(); self.max_draft_tokens]);
                    Self::accumulate_depths(pooled, Self::SHARED_DEPTH_DECAY, width, accepted);
                }
            }
        }
        if self.current == 0 {
            if width == 0 {
                self.recovery_credit_us += Self::RECOVERY_SHARE * measured;
            } else if self.cost[0].samples > 0 {
                // Time beyond what ordinary windows would have taken for the
                // same tokens; negative when the probe paid off.
                self.recovery_credit_us -=
                    measured - window.committed_tokens as f64 * self.cost[0].window_us;
            }
        }
        self.ordinary_run = if width == 0 { self.ordinary_run + 1 } else { 0 };
        self.previous_width = Some(width);
        self.plan_next(observed_role == DFlash2WindowRole::Probe);
        let after = self.next_budget();
        MtpDraftBudgetChange {
            reduced: after < before,
            increased: after > before,
        }
    }

    fn accumulate_depths(depths: &mut [DepthStat], decay: f64, width: usize, accepted: usize) {
        for depth in depths.iter_mut() {
            depth.reached *= decay;
            depth.accepted *= decay;
        }
        // Positions past the first rejection were never compared.
        for (index, depth) in depths.iter_mut().enumerate().take(width) {
            depth.reached += 1.0;
            if index < accepted {
                depth.accepted += 1.0;
            } else {
                break;
            }
        }
    }

    fn record_depths(&mut self, width: usize, accepted: usize) {
        Self::accumulate_depths(&mut self.depth, Self::DEPTH_DECAY, width, accepted);
        let acceptance = accepted as f64 / width as f64;
        self.acceptance_ewma = Some(match self.acceptance_ewma {
            Some(previous) => acceptance.mul_add(
                Self::ACCEPTANCE_ALPHA,
                previous * (1.0 - Self::ACCEPTANCE_ALPHA),
            ),
            None => acceptance,
        });
    }

    /// Estimated acceptance of each depth, each shrunk towards the estimate
    /// of the depth before it.
    fn depth_acceptance(&self) -> Vec<f64> {
        let mut prior = Self::FIRST_DEPTH_PRIOR;
        self.depth
            .iter()
            .map(|depth| {
                let estimate = (depth.accepted + Self::DEPTH_PRIOR_WEIGHT * prior)
                    / (depth.reached + Self::DEPTH_PRIOR_WEIGHT);
                prior = estimate;
                estimate
            })
            .collect()
    }

    /// Expected committed tokens of a window with `budget` drafted positions:
    /// the accepted prefix plus the target's own token.
    fn expected_committed(acceptance: &[f64], budget: usize) -> f64 {
        let mut survive = 1.0;
        let mut committed = 1.0;
        for probability in acceptance.iter().take(budget) {
            survive *= probability;
            committed += survive;
        }
        committed
    }

    /// Window cost of `budget`: measured, or interpolated/extrapolated from
    /// measured speculative budgets. Budget 0 runs a different (ordinary)
    /// window and is only ever measured.
    fn predicted_window_us(&self, budget: usize) -> Option<f64> {
        let measured = |b: usize| (self.cost[b].samples > 0).then_some(self.cost[b].window_us);
        if let Some(value) = measured(budget) {
            return Some(value);
        }
        if budget == 0 {
            return None;
        }
        let points: Vec<(f64, f64)> = (1..=self.max_draft_tokens)
            .filter_map(|b| measured(b).map(|cost| (b as f64, cost)))
            .collect();
        let x = budget as f64;
        let below = points.iter().rev().find(|(b, _)| *b < x);
        let above = points.iter().find(|(b, _)| *b > x);
        let line =
            |(x0, y0): (f64, f64), (x1, y1): (f64, f64)| y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        match (below, above) {
            (Some(&low), Some(&high)) => Some(line(low, high)),
            (Some(_), None) if points.len() >= 2 => {
                Some(line(points[points.len() - 2], points[points.len() - 1]))
            }
            (None, Some(_)) if points.len() >= 2 => Some(line(points[0], points[1])),
            _ => None,
        }
        .map(|cost: f64| cost.max(0.0))
    }

    /// Predicted cost per committed token for every budget with a cost estimate.
    fn scores(&self) -> Vec<Option<f64>> {
        let acceptance = self.depth_acceptance();
        (0..=self.max_draft_tokens)
            .map(|budget| {
                self.predicted_window_us(budget)
                    .map(|cost| cost / Self::expected_committed(&acceptance, budget))
            })
            .collect()
    }

    fn best_budget(scores: &[Option<f64>]) -> Option<usize> {
        scores
            .iter()
            .enumerate()
            .filter_map(|(budget, score)| score.map(|score| (budget, score)))
            .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
            .map(|(budget, _)| budget)
    }

    fn plan_next(&mut self, after_probe: bool) {
        if let Some(budget) = self.pending_calibration() {
            self.next = budget;
            self.role = DFlash2WindowRole::Calibrate;
            return;
        }
        let scores = self.scores();
        if let Some(best) = Self::best_budget(&scores) {
            let current_score = scores[self.current];
            let switch = match (scores[best], current_score) {
                (Some(best_score), Some(current_score)) => {
                    best != self.current && best_score < current_score * (1.0 - Self::SWITCH_MARGIN)
                }
                (Some(_), None) => true,
                _ => false,
            };
            if switch {
                if best == 0 {
                    self.recovery_credit_us = 0.0;
                }
                self.current = best;
                self.probe_interval = Self::INITIAL_PROBE_INTERVAL;
                self.recovery_windows_left = 0;
            } else if after_probe && self.current > 0 {
                self.probe_interval = (self.probe_interval * 2).min(Self::MAX_PROBE_INTERVAL);
            }
        }
        if after_probe {
            self.windows_since_probe = 0;
        } else {
            self.windows_since_probe = self.windows_since_probe.saturating_add(1);
        }
        if self.current == 0 {
            self.plan_recovery(&scores);
            return;
        }
        match self.probe_target() {
            Some(probe) if self.windows_since_probe >= self.probe_interval => {
                self.next = probe;
                self.role = DFlash2WindowRole::Probe;
            }
            _ => {
                self.next = self.current;
                self.role = DFlash2WindowRole::Exploit;
            }
        }
    }

    /// Next window while budget 0 is chosen: the rest of a recovery probe, a
    /// new recovery probe at the best drafting budget when the credit covers
    /// its worst case (no draft accepted) or the ordinary run reached its
    /// bound, otherwise an ordinary window.
    fn plan_recovery(&mut self, scores: &[Option<f64>]) {
        if self.recovery_windows_left > 0 {
            self.recovery_windows_left -= 1;
            self.next = self.recovery_budget;
            self.role = DFlash2WindowRole::Probe;
            return;
        }
        let candidate = Self::best_budget(&scores[1..]).map_or(1, |index| index + 1);
        let worst_case_us = match (self.predicted_window_us(candidate), self.cost[0].samples) {
            (Some(window_us), samples) if samples > 0 => {
                Self::RECOVERY_PROBE_WINDOWS as f64 * (window_us - self.cost[0].window_us).max(0.0)
            }
            _ => f64::INFINITY,
        };
        self.recovery_credit_us = self.recovery_credit_us.min(worst_case_us);
        if self.recovery_credit_us >= worst_case_us || self.ordinary_run >= Self::MAX_PROBE_INTERVAL
        {
            self.recovery_budget = candidate;
            self.recovery_windows_left = Self::RECOVERY_PROBE_WINDOWS - 1;
            self.next = candidate;
            self.role = DFlash2WindowRole::Probe;
        } else {
            self.next = 0;
            self.role = DFlash2WindowRole::Exploit;
        }
    }

    /// Neighbour of a non-zero current budget, alternating direction; the
    /// widest budget is left through the one below it.
    fn probe_target(&mut self) -> Option<usize> {
        let up = (self.current < self.max_draft_tokens).then_some(self.current + 1);
        let down = self.current.checked_sub(1);
        let target = match (up, down) {
            (Some(up), Some(down)) => {
                if self.probe_up_next {
                    up
                } else {
                    down
                }
            }
            (Some(up), None) => up,
            (None, Some(down)) => down,
            (None, None) => return None,
        };
        if self.windows_since_probe >= self.probe_interval {
            self.probe_up_next = !self.probe_up_next;
        }
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic acceptance and cost source for policy simulations: a
    /// window drafting `budget` positions accepts the leading positions whose
    /// uniform draw falls below that depth's acceptance, and costs a fixed time
    /// per budget.
    pub(crate) struct Scenario {
        pub acceptance: Vec<f64>,
        pub window_us: Vec<u64>,
        state: u64,
    }

    impl Scenario {
        pub(crate) fn new(acceptance: &[f64], window_us: &[u64], seed: u64) -> Self {
            Self {
                acceptance: acceptance.to_vec(),
                window_us: window_us.to_vec(),
                state: seed,
            }
        }

        fn uniform(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 11) as f64 / (1_u64 << 53) as f64
        }

        pub(crate) fn window(&mut self, budget: usize) -> MtpDraftPolicyWindow {
            let mut accepted = 0;
            while accepted < budget && self.uniform() < self.acceptance[accepted] {
                accepted += 1;
            }
            MtpDraftPolicyWindow::from_measured_components(
                budget,
                accepted,
                accepted + 1,
                self.window_us[budget],
                256,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        }

        /// Expected cost per committed token of a fixed budget.
        pub(crate) fn fixed_cost_per_token(&self, budget: usize) -> f64 {
            self.window_us[budget] as f64
                / DFlash2WindowCostPolicy::expected_committed(&self.acceptance, budget)
        }
    }

    pub(crate) struct Run {
        pub budgets: Vec<usize>,
        pub roles: Vec<DFlash2WindowRole>,
        pub total_us: u64,
        pub committed: usize,
    }

    impl Run {
        pub(crate) fn cost_per_token(&self) -> f64 {
            self.total_us as f64 / self.committed as f64
        }

        pub(crate) fn histogram(&self, max: usize) -> Vec<usize> {
            let mut counts = vec![0; max + 1];
            for &budget in &self.budgets {
                counts[budget] += 1;
            }
            counts
        }
    }

    /// Drives `policy` for `windows` windows of `scenario`.
    pub(crate) fn simulate(
        policy: &mut DFlash2DraftBudgetPolicy,
        scenario: &mut Scenario,
        windows: usize,
    ) -> Run {
        let mut run = Run {
            budgets: Vec::new(),
            roles: Vec::new(),
            total_us: 0,
            committed: 0,
        };
        for _ in 0..windows {
            let budget = policy.current_budget();
            run.roles.push(policy.window_role());
            let window = scenario.window(budget);
            run.budgets.push(budget);
            run.total_us += window.total_us;
            run.committed += window.committed_tokens;
            policy.observe(window);
        }
        run
    }

    /// Window times proportional to the measured M5 Max affine4 MoE windows
    /// (ordinary 7.7 ms, b1 10.7, b3 14.4, b7 23.9 ms; others interpolated).
    /// They only shape the scenarios; the policy reads no constant time.
    const WINDOW_US: [u64; 8] = [
        7_700, 10_700, 12_550, 14_400, 16_775, 19_150, 21_525, 23_900,
    ];
    /// Per-depth acceptance measured on the code-2 prompt (fixed b7).
    const CODE: [f64; 7] = [0.90, 0.89, 0.82, 0.84, 0.96, 0.84, 1.0];
    /// Low-acceptance text: draft tokens are mostly rejected.
    const LOW: [f64; 7] = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30];
    /// Moderate acceptance (knowledge-like answers).
    const MID: [f64; 7] = [0.60, 0.55, 0.50, 0.50, 0.50, 0.50, 0.50];

    fn oracle(scenario: &Scenario) -> (usize, f64) {
        (0..scenario.window_us.len())
            .map(|budget| (budget, scenario.fixed_cost_per_token(budget)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .expect("budgets")
    }

    fn window_cost(budgets: usize) -> DFlash2DraftBudgetPolicy {
        DFlash2DraftBudgetPolicy::new(DFlash2BudgetPolicyKind::WindowCost, budgets)
    }

    use ironmlx_lm::models::dflash2::DFlash2Instance;

    fn pair(target: &DFlash2Instance, draft: &DFlash2Instance) -> Option<DFlash2PolicyDomain> {
        Some(DFlash2PolicyDomain::new(
            target.downgrade(),
            draft.downgrade(),
        ))
    }

    fn stream(domain: Option<DFlash2PolicyDomain>) -> DFlash2DraftBudgetPolicy {
        DFlash2DraftBudgetPolicy::with_memory(DFlash2BudgetPolicyKind::WindowCost, 7, domain)
    }

    /// Whether a stream of `domain` calibrates after its first window (the
    /// first one has no regime yet and runs at the widest budget).
    fn calibrates(domain: Option<DFlash2PolicyDomain>, windows: usize) -> bool {
        let mut scenario = Scenario::new(&CODE, &WINDOW_US, 3);
        let mut policy = stream(domain);
        let run = simulate(&mut policy, &mut scenario, windows);
        run.roles[1..].contains(&DFlash2WindowRole::Calibrate)
    }

    #[test]
    #[serial_test::serial(dflash2_policy_memory)]
    fn a_new_stream_of_the_same_pair_reuses_measured_costs() {
        let (target, draft) = (DFlash2Instance::new(), DFlash2Instance::new());
        assert!(calibrates(pair(&target, &draft), 40));
        // Once its regime is known every cost of the pair is already measured.
        assert!(!calibrates(pair(&target, &draft), 40));
        // A different pair (another target with the same drafter) starts
        // from scratch.
        let other = DFlash2Instance::new();
        assert!(calibrates(pair(&other, &draft), 4));
    }

    #[test]
    #[serial_test::serial(dflash2_policy_memory)]
    fn the_pair_identity_moves_with_the_models() {
        struct Loaded {
            _weights: Vec<u8>,
            instance: DFlash2Instance,
        }
        let target = Loaded {
            _weights: vec![0; 64],
            instance: DFlash2Instance::new(),
        };
        let draft = DFlash2Instance::new();
        assert!(calibrates(pair(&target.instance, &draft), 40));
        let moved = Box::new(target);
        let mut held = vec![moved];
        held.reserve(1024);
        assert!(!calibrates(pair(&held[0].instance, &draft), 40));
    }

    #[test]
    #[serial_test::serial(dflash2_policy_memory)]
    fn a_reloaded_pair_calibrates_again_and_unloaded_pairs_are_dropped() {
        let before = shared_pair_count();
        let mut seen = std::collections::HashSet::new();
        let mut reused = 0;
        for _ in 0..64 {
            // Each load measures its own costs: it never sees those of the
            // previous load, even when the allocator hands it the same address.
            let (target, draft) = (DFlash2Instance::new(), DFlash2Instance::new());
            let address = (target.downgrade().as_ptr(), draft.downgrade().as_ptr());
            reused += usize::from(!seen.insert(address));
            assert!(calibrates(pair(&target, &draft), 40));
            assert!(!calibrates(pair(&target, &draft), 40));
            drop((target, draft));
        }
        // Unloaded pairs leave the memory, so the allocator reuses their
        // addresses; none of those loads inherited anything (asserted above).
        eprintln!("loads at an earlier load's addresses: {reused} of 64");
        assert!(shared_pair_count() <= before);
        // A stream that outlives its models no longer reads or writes the
        // memory.
        let (target, draft) = (DFlash2Instance::new(), DFlash2Instance::new());
        let domain = pair(&target, &draft);
        drop(target);
        assert!(calibrates(domain, 40));
        assert!(shared_pair_count() <= before);
    }

    #[test]
    fn early_partial_accepts_do_not_pin_the_budget_to_one() {
        // The first two windows reject early (7/2 then 3/0 under the legacy
        // policy). The window-cost policy keeps wider budgets in play.
        let mut scenario = Scenario::new(&CODE, &WINDOW_US, 7);
        let mut policy = window_cost(7);
        let mut forced = [2_usize, 0].into_iter();
        for _ in 0..2 {
            let budget = policy.current_budget();
            let accepted = forced.next().expect("forced").min(budget);
            policy.observe(MtpDraftPolicyWindow::from_measured_components(
                budget,
                accepted,
                accepted + 1,
                WINDOW_US[budget],
                256,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ));
        }
        let run = simulate(&mut policy, &mut scenario, 120);
        let tail = &run.budgets[run.budgets.len() - 60..];
        let at_one = tail.iter().filter(|&&budget| budget == 1).count();
        assert!(at_one < tail.len() / 4, "budget stays at 1: {tail:?}");
        let (best, best_cost) = oracle(&scenario);
        assert!(best >= 3, "scenario optimum {best}");
        assert!(
            run.cost_per_token() <= best_cost * 1.08,
            "cost/token {:.0} vs oracle b{best} {best_cost:.0}",
            run.cost_per_token()
        );
    }

    #[test]
    fn upward_probes_are_not_crowded_out_by_zero_probes() {
        // Start where the legacy policy ends up: budget 1 with good acceptance.
        let mut scenario = Scenario::new(&CODE, &WINDOW_US, 11);
        let mut policy = window_cost(7);
        let run = simulate(&mut policy, &mut scenario, 150);
        let probes: Vec<usize> = run
            .roles
            .iter()
            .zip(&run.budgets)
            .filter(|(role, _)| **role == DFlash2WindowRole::Probe)
            .map(|(_, &budget)| budget)
            .collect();
        let zero = probes.iter().filter(|&&budget| budget == 0).count();
        assert!(zero <= probes.len() / 2 + 1, "probes {probes:?}");
        let histogram = run.histogram(7);
        assert!(
            histogram[3..].iter().sum::<usize>() > run.budgets.len() / 2,
            "budget histogram {histogram:?}"
        );
    }

    #[test]
    fn budget_zero_is_chosen_for_low_acceptance_and_re_explored() {
        // Per seed: probes keep leaving 0, their count stays bounded and the
        // cost stays near the best fixed budget. Across seeds the policy
        // spends most windows on 0 (a single seed can hit an acceptance
        // streak during which budget 1 really is cheaper).
        let mut ordinary_windows = Vec::new();
        for seed in 0..16_u64 {
            let mut scenario = Scenario::new(&LOW, &WINDOW_US, seed);
            let mut policy = window_cost(7);
            let run = simulate(&mut policy, &mut scenario, 200);
            let histogram = run.histogram(7);
            ordinary_windows.push(histogram[0]);
            let settled = run
                .budgets
                .iter()
                .position(|&budget| budget == 0)
                .expect("reaches budget 0");
            let later_probes = run.roles[settled..]
                .iter()
                .zip(&run.budgets[settled..])
                .filter(|(role, budget)| **role == DFlash2WindowRole::Probe && **budget > 0)
                .count();
            assert!(
                later_probes >= 2,
                "seed {seed}: budget 0 is final: {later_probes} probes"
            );
            let probes = run
                .roles
                .iter()
                .filter(|role| **role == DFlash2WindowRole::Probe)
                .count();
            assert!(
                probes <= run.budgets.len() / 8,
                "seed {seed}: {probes} probes"
            );
            let (_, best_cost) = oracle(&scenario);
            assert!(
                run.cost_per_token() <= best_cost * 1.08,
                "seed {seed}: histogram {histogram:?}"
            );
        }
        ordinary_windows.sort_unstable();
        assert!(
            ordinary_windows[ordinary_windows.len() / 2] > 150,
            "ordinary windows per seed {ordinary_windows:?}"
        );
    }

    #[test]
    fn budget_zero_always_returns_to_drafting_within_the_probe_bound() {
        // The stream keeps the raw target context of ordinary windows until
        // the next drafting window consumes it; this bounds that context.
        for seed in 0..8 {
            let mut scenario = Scenario::new(&[0.0; 7], &WINDOW_US, seed);
            let mut policy = window_cost(7);
            let run = simulate(&mut policy, &mut scenario, 400);
            let mut gap = 0;
            let mut longest = 0;
            for &budget in &run.budgets {
                gap = if budget == 0 { gap + 1 } else { 0 };
                longest = longest.max(gap);
            }
            assert!(
                longest <= DFlash2WindowCostPolicy::MAX_PROBE_INTERVAL + 1,
                "seed {seed}: {longest} ordinary windows without drafting"
            );
            assert!(
                run.histogram(7)[0] > 350,
                "seed {seed}: {:?}",
                run.histogram(7)
            );
        }
    }

    #[test]
    fn budget_zero_is_left_when_acceptance_recovers() {
        let mut low = Scenario::new(&LOW, &WINDOW_US, 3);
        let mut policy = window_cost(7);
        simulate(&mut policy, &mut low, 120);
        assert_eq!(policy.current_budget(), 0);
        let mut code = Scenario::new(&CODE, &WINDOW_US, 3);
        let run = simulate(&mut policy, &mut code, 200);
        let tail = &run.budgets[run.budgets.len() - 50..];
        assert!(
            tail.iter().filter(|&&budget| budget >= 2).count() > 40,
            "tail {tail:?}"
        );
    }

    #[test]
    fn a_cheaper_wide_budget_is_not_excluded_by_moderate_acceptance() {
        // Acceptance below the legacy 0.85 gate, yet budget 2 is the cheapest
        // per committed token.
        let mut scenario = Scenario::new(&MID, &WINDOW_US, 17);
        let (best, best_cost) = oracle(&scenario);
        assert!(best >= 1);
        let mut policy = window_cost(7);
        let run = simulate(&mut policy, &mut scenario, 150);
        assert!(
            run.cost_per_token() <= best_cost * 1.08,
            "cost/token {:.0} vs oracle b{best} {best_cost:.0}, histogram {:?}",
            run.cost_per_token(),
            run.histogram(7)
        );
    }

    #[test]
    fn policy_selection_keeps_legacy_outside_the_qualified_linear_path() {
        use DFlash2BudgetPolicyKind::{Legacy, WindowCost};
        let select = DFlash2BudgetPolicyKind::select;
        assert_eq!(select(None, true, false).unwrap(), WindowCost);
        assert_eq!(select(Some(""), true, false).unwrap(), WindowCost);
        assert_eq!(select(None, false, false).unwrap(), Legacy);
        assert_eq!(select(None, true, true).unwrap(), Legacy);
        assert_eq!(select(Some("legacy"), true, false).unwrap(), Legacy);
        assert_eq!(
            select(Some("window-cost"), true, false).unwrap(),
            WindowCost
        );
        assert!(select(Some("window-cost"), false, false).is_err());
        assert!(select(Some("window-cost"), true, true).is_err());
        assert!(select(Some("b3"), true, false).is_err());
    }

    /// Draft-side cost of consuming one pending target-context position,
    /// from the real p4 run: a b1 window right after an ordinary stretch
    /// takes ~1.1 ms more than a steady b1 window (~64 extra positions).
    const CONTEXT_US_PER_POSITION: u64 = 17;
    /// Prompt positions consumed by the first drafting window.
    const PROMPT_POSITIONS: u64 = 100;

    /// Windows of a scripted run, with the time and tokens of each.
    struct Trace {
        budgets: Vec<usize>,
        roles: Vec<DFlash2WindowRole>,
        window_us: Vec<u64>,
        committed: Vec<usize>,
    }

    impl Trace {
        fn cost_per_token(&self, from: usize, to: usize) -> f64 {
            let us: u64 = self.window_us[from..to].iter().sum();
            let tokens: usize = self.committed[from..to].iter().sum();
            us as f64 / tokens as f64
        }

        fn longest_ordinary_run(&self) -> usize {
            let (mut run, mut longest) = (0, 0);
            for &budget in &self.budgets {
                run = if budget == 0 { run + 1 } else { 0 };
                longest = longest.max(run);
            }
            longest
        }

        /// First window at or after `change` from which at least 90% of the
        /// next 50 windows draft, with the tokens, windows and time spent
        /// between `change` and it.
        fn sustained_drafting_after(&self, change: usize) -> Option<(usize, usize, usize, u64)> {
            (change..self.budgets.len()).find_map(|start| {
                let span = &self.budgets[start..(start + 50).min(self.budgets.len())];
                let drafting = span.iter().filter(|&&budget| budget > 0).count();
                (span.len() >= 10 && drafting * 10 >= span.len() * 9).then(|| {
                    (
                        start,
                        self.committed[change..start].iter().sum(),
                        start - change,
                        self.window_us[change..start].iter().sum(),
                    )
                })
            })
        }

        fn report(&self, name: &str) -> String {
            let compact: String = self
                .budgets
                .iter()
                .map(|&budget| char::from(b'0' + budget as u8))
                .collect();
            format!(
                "{name}: histogram {:?} probes {} longest_ordinary_run {} budgets {compact}",
                {
                    let mut counts = vec![0; 8];
                    for &budget in &self.budgets {
                        counts[budget] += 1;
                    }
                    counts
                },
                self.roles
                    .iter()
                    .filter(|role| **role == DFlash2WindowRole::Probe)
                    .count(),
                self.longest_ordinary_run(),
            )
        }
    }

    /// Drives the window-cost policy for `windows` windows. `acceptance(step)`
    /// gives the per-depth acceptance and `cost(step, budget, first_use)` the
    /// window time before context consumption; drafting windows also pay for
    /// every pending context position (the previous drafting window's
    /// committed tokens plus each ordinary window since, or the prompt).
    fn run_script(
        windows: usize,
        seed: u64,
        acceptance: impl Fn(usize) -> &'static [f64],
        cost: impl Fn(usize, usize, bool) -> u64,
    ) -> Trace {
        let mut policy = window_cost(7);
        let mut rng = Scenario::new(&[0.0; 7], &WINDOW_US, seed);
        let mut used = [false; 8];
        let mut pending = PROMPT_POSITIONS;
        let mut trace = Trace {
            budgets: Vec::new(),
            roles: Vec::new(),
            window_us: Vec::new(),
            committed: Vec::new(),
        };
        for step in 0..windows {
            let budget = policy.current_budget();
            trace.roles.push(policy.window_role());
            let rates = acceptance(step);
            let mut accepted = 0;
            while accepted < budget && rng.uniform() < rates[accepted] {
                accepted += 1;
            }
            let mut window_us = cost(step, budget, !used[budget]);
            used[budget] = true;
            if budget > 0 {
                window_us += CONTEXT_US_PER_POSITION * pending;
                pending = accepted as u64 + 1;
            } else {
                pending += 1;
            }
            trace.budgets.push(budget);
            trace.window_us.push(window_us);
            trace.committed.push(accepted + 1);
            policy.observe(fixed_window(budget, accepted, window_us));
        }
        trace
    }

    fn best_fixed_cost(acceptance: &[f64]) -> f64 {
        oracle(&Scenario::new(acceptance, &WINDOW_US, 0)).1
    }

    /// Ordinary windows and cost against the best fixed budget over 32 seeds
    /// of the constant scenarios (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn budget_policy_seed_sweep() {
        for (name, acceptance) in [("LOW", &LOW), ("MID", &MID), ("CODE", &CODE)] {
            let mut ordinary_windows = Vec::new();
            let mut ratios = Vec::new();
            for seed in 0..32_u64 {
                let mut scenario = Scenario::new(acceptance, &WINDOW_US, seed);
                let mut policy = window_cost(7);
                let run = simulate(&mut policy, &mut scenario, 200);
                ordinary_windows.push(run.histogram(7)[0]);
                ratios.push(run.cost_per_token() / oracle(&scenario).1);
            }
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            let worst = ratios.iter().copied().fold(0.0, f64::max);
            println!(
                "{name} ordinary_windows_per_seed {ordinary_windows:?} cost_ratio mean {mean:.3} worst {worst:.3}"
            );
        }
    }

    #[test]
    fn recovery_s1_first_use_cost_spike_is_remeasured() {
        // The first window at each budget >= 2 takes 5x its steady time (the
        // first-use spikes seen in the real run); later windows are normal.
        let trace = run_script(
            300,
            21,
            |_| &CODE,
            |_, budget, first_use| {
                if first_use && budget >= 2 {
                    WINDOW_US[budget] * 5
                } else {
                    WINDOW_US[budget]
                }
            },
        );
        let ratio = trace.cost_per_token(100, 300) / best_fixed_cost(&CODE);
        eprintln!("{} cost_ratio_100_300 {ratio:.3}", trace.report("S1"));
        assert!(ratio <= 1.10, "S1: {ratio:.3} x the best fixed budget");
    }

    #[test]
    fn recovery_s2_low_then_high_acceptance_resumes_drafting() {
        let change = 200;
        let trace = run_script(
            500,
            22,
            |step| if step < change { &LOW } else { &CODE },
            |_, budget, _| WINDOW_US[budget],
        );
        // Precondition: the low-acceptance text settled the policy on budget 0.
        let before = &trace.budgets[change - 50..change];
        assert!(
            before.iter().filter(|&&budget| budget == 0).count() >= 40,
            "S2: not settled on budget 0 before the change: {before:?}"
        );
        let sustained = trace.sustained_drafting_after(change);
        let ratio = trace.cost_per_token(change, 500) / best_fixed_cost(&CODE);
        eprintln!(
            "{} sustained_drafting(start,tokens,windows,us) {sustained:?} cost_ratio_after {ratio:.3}",
            trace.report("S2")
        );
        let (_, tokens, _, _) = sustained.expect("S2: drafting never resumed");
        assert!(tokens <= 128, "S2: resumed after {tokens} tokens");
        assert!(ratio <= 1.15, "S2: {ratio:.3} x the best fixed budget");
    }

    #[test]
    fn recovery_s3_transient_cost_interference_is_forgotten() {
        // Windows 60..160 are slowed down: ordinary windows 1.5x, drafting
        // windows 2.6x (the ratios of the slow p4 run).
        let (from, to) = (60, 160);
        let trace = run_script(
            to + 300,
            23,
            |_| &MID,
            |step, budget, _| {
                let base = WINDOW_US[budget] as f64;
                let factor = match (step >= from && step < to, budget) {
                    (false, _) => 1.0,
                    (true, 0) => 1.5,
                    (true, _) => 2.6,
                };
                (base * factor) as u64
            },
        );
        let sustained = trace.sustained_drafting_after(to);
        let ratio = trace.cost_per_token(to, to + 300) / best_fixed_cost(&MID);
        eprintln!(
            "{} sustained_drafting(start,tokens,windows,us) {sustained:?} cost_ratio_after {ratio:.3}",
            trace.report("S3")
        );
        let (_, tokens, _, _) = sustained.expect("S3: drafting never resumed");
        assert!(tokens <= 128, "S3: resumed after {tokens} tokens");
        assert!(ratio <= 1.10, "S3: {ratio:.3} x the best fixed budget");
    }

    #[test]
    fn recovery_s4_sustained_low_acceptance_keeps_ordinary_decoding() {
        const NONE: [f64; 7] = [0.0; 7];
        for (name, acceptance, seed) in [("S4-low", &LOW, 24_u64), ("S4-none", &NONE, 25)] {
            let trace = run_script(600, seed, |_| acceptance, |_, budget, _| WINDOW_US[budget]);
            let ratio = trace.cost_per_token(100, 600) / WINDOW_US[0] as f64;
            eprintln!("{} cost_ratio_vs_ordinary {ratio:.3}", trace.report(name));
            assert!(ratio <= 1.03, "{name}: {ratio:.3} x ordinary decoding");
            assert!(
                trace.longest_ordinary_run() <= DFlash2WindowCostPolicy::MAX_PROBE_INTERVAL,
                "{name}: {} ordinary windows in a row",
                trace.longest_ordinary_run()
            );
        }
    }

    #[test]
    fn recovery_s5_cold_start_spikes_do_not_hold_budget_zero() {
        // A cold process: the first window at every drafted width takes 6x its
        // steady time (79.7-84.2 ms first b1/b2/b4 windows in the cold p1 run).
        // Moderate acceptance first (the answer), then code-like (the copy).
        let change = 60;
        let trace = run_script(
            300,
            26,
            |step| if step < change { &MID } else { &CODE },
            |_, budget, first_use| {
                if first_use && budget >= 1 {
                    WINDOW_US[budget] * 6
                } else {
                    WINDOW_US[budget]
                }
            },
        );
        let ordinary = trace.budgets[..150]
            .iter()
            .filter(|&&budget| budget == 0)
            .count();
        let ratio = trace.cost_per_token(change, 300) / best_fixed_cost(&CODE);
        eprintln!(
            "{} ordinary_in_first_150 {ordinary} cost_ratio_60_300 {ratio:.3}",
            trace.report("S5")
        );
        assert!(
            ordinary * 4 <= 150,
            "S5: {ordinary} ordinary windows of the first 150"
        );
        assert!(ratio <= 1.15, "S5: {ratio:.3} x the best fixed budget");
    }

    /// Exploration spent while budget 0 is chosen, split by what started each
    /// recovery probe.
    struct ExplorationAccount {
        /// Time of ordinary windows once the policy has settled on budget 0.
        ordinary_us: u64,
        /// Drafting time beyond what ordinary windows would have taken for the
        /// same tokens, summed over every probe window after settling.
        probe_overhead_us: f64,
        /// Probes that started after `MAX_PROBE_INTERVAL` ordinary windows in a
        /// row, and all others.
        forced_probes: usize,
        other_probes: usize,
        /// Largest overhead of one probe (its consecutive drafting windows).
        largest_probe_us: f64,
        longest_ordinary_run: usize,
    }

    fn exploration_account(trace: &Trace, settled: usize, ordinary_us: f64) -> ExplorationAccount {
        let mut account = ExplorationAccount {
            ordinary_us: 0,
            probe_overhead_us: 0.0,
            forced_probes: 0,
            other_probes: 0,
            largest_probe_us: 0.0,
            longest_ordinary_run: 0,
        };
        let (mut run, mut probe_us) = (0, 0.0_f64);
        for index in settled..trace.budgets.len() {
            if trace.budgets[index] == 0 {
                account.ordinary_us += trace.window_us[index];
                if probe_us != 0.0 {
                    account.largest_probe_us = account.largest_probe_us.max(probe_us);
                    probe_us = 0.0;
                }
                run += 1;
                account.longest_ordinary_run = account.longest_ordinary_run.max(run);
                continue;
            }
            if run > 0 || index == settled {
                if run >= DFlash2WindowCostPolicy::MAX_PROBE_INTERVAL {
                    account.forced_probes += 1;
                } else {
                    account.other_probes += 1;
                }
            }
            run = 0;
            let overhead =
                trace.window_us[index] as f64 - trace.committed[index] as f64 * ordinary_us;
            account.probe_overhead_us += overhead;
            probe_us += overhead;
        }
        account.largest_probe_us = account.largest_probe_us.max(probe_us);
        account
    }

    /// Exploration overhead at budget 0 when drafting cannot pay off, with
    /// realistic and with expensive drafting windows (a draft forward that
    /// dominates the window), against the stated 2% share (run with
    /// `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn exploration_overhead_counterexamples() {
        const NONE: [f64; 7] = [0.0; 7];
        // Draft forward ~28 ms: every drafting window costs 30-43 ms.
        const EXPENSIVE_US: [u64; 8] = [
            7_700, 30_000, 32_000, 34_000, 36_000, 38_000, 40_000, 43_000,
        ];
        for (name, acceptance, costs) in [
            ("none/realistic", &NONE, &WINDOW_US),
            ("low/realistic", &LOW, &WINDOW_US),
            ("none/expensive-draft", &NONE, &EXPENSIVE_US),
            ("low/expensive-draft", &LOW, &EXPENSIVE_US),
        ] {
            let trace = run_script(2_000, 31, |_| acceptance, |_, budget, _| costs[budget]);
            let settled = trace
                .budgets
                .iter()
                .enumerate()
                .skip(8)
                .find(|(_, &budget)| budget == 0)
                .map(|(index, _)| index)
                .expect("settles on 0");
            let account = exploration_account(&trace, settled, costs[0] as f64);
            println!(
                "{name}: ordinary_ms {:.0} probe_overhead_ms {:.0} share {:.2}% forced_probes {} other_probes {} largest_probe_ms {:.1} longest_ordinary_run {}",
                account.ordinary_us as f64 / 1e3,
                account.probe_overhead_us / 1e3,
                100.0 * account.probe_overhead_us / account.ordinary_us as f64,
                account.forced_probes,
                account.other_probes,
                account.largest_probe_us / 1e3,
                account.longest_ordinary_run,
            );
        }
    }

    /// S3 (MID acceptance, windows 60..160 slowed: ordinary 1.5x, drafting
    /// 2.6x) exactly as `recovery_s3_transient_cost_interference_is_forgotten`
    /// runs it, with the policy state observable after every window and an
    /// optional diagnostic intervention on that state after window 159.
    const S3_FROM: usize = 60;
    const S3_TO: usize = 160;
    const S3_WINDOWS: usize = S3_TO + 300;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum S3Intervention {
        None,
        /// Cost estimates back to their values before the interference.
        RestoreCosts,
        /// Depth acceptance statistics back to their values before it.
        RestoreAcceptance,
        /// Exploration credit large enough for a recovery probe at once.
        FullCredit,
    }

    fn s3_cost(step: usize, budget: usize) -> u64 {
        let base = WINDOW_US[budget] as f64;
        let factor = match ((S3_FROM..S3_TO).contains(&step), budget) {
            (false, _) => 1.0,
            (true, 0) => 1.5,
            (true, _) => 2.6,
        };
        (base * factor) as u64
    }

    fn window_cost_state(policy: &DFlash2DraftBudgetPolicy) -> &DFlash2WindowCostPolicy {
        match policy {
            DFlash2DraftBudgetPolicy::WindowCost(policy) => policy,
            DFlash2DraftBudgetPolicy::Legacy(_) => unreachable!(),
        }
    }

    fn window_cost_state_mut(
        policy: &mut DFlash2DraftBudgetPolicy,
    ) -> &mut DFlash2WindowCostPolicy {
        match policy {
            DFlash2DraftBudgetPolicy::WindowCost(policy) => policy,
            DFlash2DraftBudgetPolicy::Legacy(_) => unreachable!(),
        }
    }

    /// Runs S3 for `seed`; calls `inspect(step, budget, role, accepted,
    /// window_us, pending_after, policy)` after each window.
    fn s3_run(
        seed: u64,
        intervention: S3Intervention,
        mut inspect: impl FnMut(
            usize,
            usize,
            DFlash2WindowRole,
            usize,
            u64,
            u64,
            &DFlash2DraftBudgetPolicy,
        ),
    ) -> Trace {
        let mut policy = window_cost(7);
        let mut rng = Scenario::new(&[0.0; 7], &WINDOW_US, seed);
        let mut used = [false; 8];
        let mut pending = PROMPT_POSITIONS;
        let mut trace = Trace {
            budgets: Vec::new(),
            roles: Vec::new(),
            window_us: Vec::new(),
            committed: Vec::new(),
        };
        let mut before: Option<(Vec<CostStat>, Vec<DepthStat>)> = None;
        for step in 0..S3_WINDOWS {
            let budget = policy.current_budget();
            let role = policy.window_role();
            trace.roles.push(role);
            let mut accepted = 0;
            while accepted < budget && rng.uniform() < MID[accepted] {
                accepted += 1;
            }
            let mut window_us = s3_cost(step, budget);
            used[budget] = true;
            if budget > 0 {
                window_us += CONTEXT_US_PER_POSITION * pending;
                pending = accepted as u64 + 1;
            } else {
                pending += 1;
            }
            trace.budgets.push(budget);
            trace.window_us.push(window_us);
            trace.committed.push(accepted + 1);
            policy.observe(fixed_window(budget, accepted, window_us));
            if step + 1 == S3_FROM {
                let state = window_cost_state(&policy);
                before = Some((state.cost.clone(), state.depth.clone()));
            }
            if step + 1 == S3_TO && intervention != S3Intervention::None {
                let (cost, depth) = before.clone().expect("state before the interference");
                let state = window_cost_state_mut(&mut policy);
                match intervention {
                    S3Intervention::RestoreCosts => state.cost = cost,
                    S3Intervention::RestoreAcceptance => state.depth = depth,
                    S3Intervention::FullCredit => state.recovery_credit_us = f64::MAX / 4.0,
                    S3Intervention::None => {}
                }
                state.plan_next(false);
            }
            inspect(step, budget, role, accepted, window_us, pending, &policy);
        }
        trace
    }

    /// Complete 50-window spans only (the registered text).
    fn sustained_complete_after(trace: &Trace, change: usize) -> Option<(usize, usize)> {
        (change..trace.budgets.len().saturating_sub(49)).find_map(|start| {
            let drafting = trace.budgets[start..start + 50]
                .iter()
                .filter(|&&b| b > 0)
                .count();
            (drafting * 10 >= 50 * 9).then(|| (start, trace.committed[change..start].iter().sum()))
        })
    }

    fn s3_zero_runs(trace: &Trace, from: usize, to: usize) -> (usize, usize, usize, usize) {
        // (ordinary windows, longest ordinary run, drafting windows, returns to 0 after drafting)
        let window = &trace.budgets[from..to];
        let (mut run, mut longest, mut reentries) = (0, 0, 0);
        for (i, &b) in window.iter().enumerate() {
            run = if b == 0 { run + 1 } else { 0 };
            longest = longest.max(run);
            if b == 0 && i > 0 && window[i - 1] > 0 {
                reentries += 1;
            }
        }
        let zeros = window.iter().filter(|&&b| b == 0).count();
        (zeros, longest, window.len() - zeros, reentries)
    }

    /// Per-seed classification of S3 over seeds 0..32 for the frozen policy
    /// (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn s3_seed_classification() {
        for seed in 0..32_u64 {
            let trace = s3_run(seed, S3Intervention::None, |_, _, _, _, _, _, _| {});
            let reference = run_script(
                S3_WINDOWS,
                seed,
                |_| &MID,
                |step, budget, _| s3_cost(step, budget),
            );
            assert_eq!(
                trace.budgets, reference.budgets,
                "seed {seed}: replicated loop differs"
            );
            let original = trace.sustained_drafting_after(S3_TO);
            let registered = sustained_complete_after(&trace, S3_TO);
            let ratio = trace.cost_per_token(S3_TO, S3_WINDOWS) / best_fixed_cost(&MID);
            let (_, during_longest, _, _) = s3_zero_runs(&trace, S3_FROM, S3_TO);
            let (zeros, longest, drafting, reentries) = s3_zero_runs(&trace, S3_TO, S3_WINDOWS);
            let at_end = trace.budgets[S3_TO];
            println!(
                "S3SEED {{\"seed\":{seed},\"original_recovered\":{},\"original_start\":{},\"original_tokens\":{},\"registered_recovered\":{},\"registered_start\":{},\"registered_tokens\":{},\"cost_ratio\":{ratio:.4},\"longest_zero_during\":{during_longest},\"budget_after_interference\":{at_end},\"zeros_after\":{zeros},\"longest_zero_after\":{longest},\"drafting_after\":{drafting},\"returns_to_zero_after\":{reentries}}}",
                original.is_some(),
                original.map_or(-1, |o| o.0 as i64),
                original.map_or(-1, |o| o.1 as i64),
                registered.is_some(),
                registered.map_or(-1, |r| r.0 as i64),
                registered.map_or(-1, |r| r.1 as i64),
            );
        }
    }

    /// Full decision trace of S3 for the seeds in `S3_TRACE_SEEDS` (comma
    /// separated; run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn s3_window_trace() {
        let seeds: Vec<u64> = std::env::var("S3_TRACE_SEEDS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|seed| seed.trim().parse().ok())
            .collect();
        for seed in seeds {
            let mut committed_total = 0;
            s3_run(
                seed,
                S3Intervention::None,
                |step, budget, role, accepted, window_us, pending, policy| {
                    committed_total += accepted + 1;
                    let state = window_cost_state(policy);
                    let next_role = policy.window_role();
                    let trigger = match next_role {
                        DFlash2WindowRole::Probe if state.current == 0 => {
                            if state.recovery_windows_left + 1
                                == DFlash2WindowCostPolicy::RECOVERY_PROBE_WINDOWS
                            {
                                if state.ordinary_run >= DFlash2WindowCostPolicy::MAX_PROBE_INTERVAL
                                {
                                    "forced-bound"
                                } else {
                                    "credit"
                                }
                            } else {
                                "probe-continuation"
                            }
                        }
                        DFlash2WindowRole::Probe => "neighbour-interval",
                        DFlash2WindowRole::Calibrate => "calibration",
                        DFlash2WindowRole::Exploit => "-",
                    };
                    let costs: Vec<String> = state
                        .cost
                        .iter()
                        .map(|c| format!("[{:.0},{}]", c.window_us, c.samples))
                        .collect();
                    let depth: Vec<String> = state
                        .depth_acceptance()
                        .iter()
                        .map(|p| format!("{p:.3}"))
                        .collect();
                    let scores: Vec<String> = state
                        .scores()
                        .iter()
                        .map(|s| s.map_or("null".to_string(), |s| format!("{s:.0}")))
                        .collect();
                    println!(
                    "S3TRACE {{\"seed\":{seed},\"step\":{step},\"budget\":{budget},\"role\":\"{role:?}\",\"accepted\":{accepted},\"committed\":{},\"cum_committed\":{committed_total},\"window_us\":{window_us},\"pending_after\":{pending},\"cost\":[{}],\"depth_acceptance\":[{}],\"scores\":[{}],\"current\":{},\"next\":{},\"next_role\":\"{next_role:?}\",\"next_trigger\":\"{trigger}\",\"credit_us\":{:.0},\"ordinary_run\":{},\"recovery_left\":{},\"probe_interval\":{},\"windows_since_probe\":{}}}",
                    accepted + 1,
                    costs.join(","),
                    depth.join(","),
                    scores.join(","),
                    state.current,
                    policy.current_budget(),
                    state.recovery_credit_us,
                    state.ordinary_run,
                    state.recovery_windows_left,
                    state.probe_interval,
                    state.windows_since_probe,
                );
                },
            );
        }
    }

    /// Single interventions on the policy state after the interference, for
    /// the seeds in `S3_TRACE_SEEDS` (diagnostic; run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn s3_state_interventions() {
        let seeds: Vec<u64> = std::env::var("S3_TRACE_SEEDS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|seed| seed.trim().parse().ok())
            .collect();
        for seed in seeds {
            for intervention in [
                S3Intervention::None,
                S3Intervention::RestoreCosts,
                S3Intervention::RestoreAcceptance,
                S3Intervention::FullCredit,
            ] {
                let trace = s3_run(seed, intervention, |_, _, _, _, _, _, _| {});
                let original = trace.sustained_drafting_after(S3_TO);
                let registered = sustained_complete_after(&trace, S3_TO);
                let ratio = trace.cost_per_token(S3_TO, S3_WINDOWS) / best_fixed_cost(&MID);
                let (zeros, longest, drafting, reentries) = s3_zero_runs(&trace, S3_TO, S3_WINDOWS);
                println!(
                    "S3INTERVENTION seed {seed} {intervention:?}: original {:?} registered {:?} cost_ratio {ratio:.3} zeros_after {zeros} longest_zero_after {longest} drafting_after {drafting} returns_to_zero {reentries} budgets_after {}",
                    original.map(|o| (o.0, o.1)),
                    registered,
                    trace.budgets[S3_TO..]
                        .iter()
                        .map(|&b| char::from(b'0' + b as u8))
                        .collect::<String>()
                );
            }
        }
    }

    /// Replaying the observed windows of S3 seed 3 into a fresh policy gives
    /// the same decisions: the policy is a deterministic function of its
    /// observations (diagnostic; run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn s3_seed3_replay_is_deterministic() {
        let mut observed = Vec::new();
        let mut decided = Vec::new();
        s3_run(
            3,
            S3Intervention::None,
            |_, budget, _, accepted, window_us, _, policy| {
                observed.push((budget, accepted, window_us));
                decided.push(policy.current_budget());
            },
        );
        let mut replay = window_cost(7);
        for (index, &(budget, accepted, window_us)) in observed.iter().enumerate() {
            assert_eq!(
                replay.current_budget(),
                budget,
                "window {index}: replay chose another budget"
            );
            replay.observe(fixed_window(budget, accepted, window_us));
            assert_eq!(
                replay.current_budget(),
                decided[index],
                "window {index}: next budget differs"
            );
        }
        println!(
            "S3REPLAY seed 3: {} windows replayed with identical decisions",
            observed.len()
        );
    }

    /// Deterministic acceptance with MID-like means: the first rejected depth of
    /// window `k` cycles through a fixed 40-entry pattern (P(accept depth 1)
    /// 0.600, depth 2 0.325, depth 3 0.175, depth 4 0.075, depth 5 0.025).
    fn patterned_accepted(k: usize, budget: usize) -> usize {
        const FIRST_REJECTED: [usize; 40] = [
            0, 1, 0, 2, 0, 1, 3, 0, 1, 2, 0, 1, 4, 0, 2, 1, 0, 3, 1, 0, 0, 1, 0, 2, 0, 1, 3, 0, 1,
            2, 0, 1, 4, 0, 2, 1, 0, 3, 1, 5,
        ];
        FIRST_REJECTED[k % FIRST_REJECTED.len()].min(budget)
    }

    /// Minimal reproduction attempt of the S3 seed-3 failure (diagnostic input,
    /// not a natural run): after a normal calibration, a slowdown is fed so
    /// that the narrow budgets that pay off under MID-like acceptance (1, 2)
    /// collect many slowed samples while budgets 4 and 7 collect one each;
    /// then normal costs and the patterned acceptance run closed loop.
    /// `inflate_narrow` and `inflate_wide` switch those parts off.
    fn minimal_s3_repro(
        inflate_narrow: bool,
        inflate_wide: bool,
    ) -> (Vec<usize>, usize, Vec<usize>, Trace) {
        let mut policy = window_cost(7);
        let mut k = 0;
        let mut feed = |policy: &mut DFlash2DraftBudgetPolicy, budget: usize, factor: f64| {
            let accepted = patterned_accepted(k, budget);
            k += 1;
            policy.observe(fixed_window(
                budget,
                accepted,
                (WINDOW_US[budget] as f64 * factor) as u64,
            ));
        };
        for budget in [7, 7, 7, 1, 1, 4, 4, 0, 0] {
            feed(&mut policy, budget, 1.0);
        }
        for _ in 0..12 {
            feed(&mut policy, 1, if inflate_narrow { 2.6 } else { 1.0 });
            feed(&mut policy, 2, if inflate_narrow { 2.6 } else { 1.0 });
        }
        feed(&mut policy, 4, if inflate_wide { 2.6 } else { 1.0 });
        feed(&mut policy, 7, if inflate_wide { 2.6 } else { 1.0 });
        for _ in 0..30 {
            feed(&mut policy, 0, 1.5);
        }
        let state = window_cost_state(&policy);
        let costs: Vec<usize> = state.cost.iter().map(|c| c.window_us as usize).collect();
        let current_after_slowdown = state.current;
        let mut trace = Trace {
            budgets: Vec::new(),
            roles: Vec::new(),
            window_us: Vec::new(),
            committed: Vec::new(),
        };
        let mut probes = Vec::new();
        for _ in 0..300 {
            let budget = policy.current_budget();
            let role = policy.window_role();
            let accepted = patterned_accepted(k, budget);
            k += 1;
            if role == DFlash2WindowRole::Probe {
                probes.push(budget);
            }
            trace.budgets.push(budget);
            trace.roles.push(role);
            trace.window_us.push(WINDOW_US[budget]);
            trace.committed.push(accepted + 1);
            policy.observe(fixed_window(budget, accepted, WINDOW_US[budget]));
        }
        (costs, current_after_slowdown, probes, trace)
    }

    /// The minimal reproduction attempt and its reductions (diagnostic; run
    /// with `--ignored --nocapture`). Under the patterned acceptance budget 4
    /// commits 2.175 tokens per window (7.71 ms/token, ordinary 7.7 ms) and
    /// budgets 1 and 2 cost ~6.69 and ~6.52 ms/token.
    #[test]
    #[ignore = "diagnostic report"]
    fn s3_minimal_reproduction() {
        let cases = [
            (
                "uneven slowdown: budgets 1, 2 many samples; 4, 7 one each",
                true,
                true,
            ),
            ("without slowing budgets 1 and 2", false, true),
            ("without slowing budgets 4 and 7", true, false),
        ];
        for (name, narrow, wide) in cases {
            let (costs, current, probes, trace) = minimal_s3_repro(narrow, wide);
            let sustained = sustained_complete_after(&trace, 0);
            let ratio = trace.cost_per_token(0, trace.budgets.len())
                / (WINDOW_US[2] as f64 / (1.0 + 0.600 + 0.325));
            let zeros = trace.budgets.iter().filter(|&&b| b == 0).count();
            let mut distinct = probes.clone();
            distinct.sort_unstable();
            distinct.dedup();
            println!(
                "S3MIN {name}: costs after slowdown {costs:?}; exploit budget after slowdown {current}; probe budgets {distinct:?} ({} probe windows); sustained drafting (complete 50) {sustained:?}; ordinary windows {zeros}/300; cost vs patterned b2 {ratio:.3}; budgets {}",
                probes.len(),
                trace
                    .budgets
                    .iter()
                    .map(|&b| char::from(b'0' + b as u8))
                    .collect::<String>()
            );
        }
    }

    fn fixed_window(budget: usize, accepted: usize, window_us: u64) -> MtpDraftPolicyWindow {
        MtpDraftPolicyWindow::from_measured_components(
            budget,
            accepted.min(budget),
            accepted.min(budget) + 1,
            window_us,
            256,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    }

    /// Step-by-step decisions of the legacy (shared Qwen MTP) policy on
    /// crafted window sequences, one per reported failure mode (run with
    /// `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn legacy_policy_decision_trace() {
        let trace = |name: &str,
                     windows: &mut dyn FnMut(usize, usize) -> MtpDraftPolicyWindow,
                     steps: usize| {
            let mut policy = DFlash2DraftBudgetPolicy::new(DFlash2BudgetPolicyKind::Legacy, 7);
            let mut budgets = Vec::new();
            for step in 0..steps {
                let budget = policy.current_budget();
                let role = policy.window_role();
                budgets.push(format!(
                    "{budget}{}",
                    if role == DFlash2WindowRole::Probe {
                        "p"
                    } else {
                        ""
                    }
                ));
                policy.observe(windows(step, budget));
            }
            println!("{name}: {}", budgets.join(" "));
        };
        // 1. Two early partial accepts (7/2, then 0 accepted), then every
        //    draft accepted: the budget falls to 1 within two windows.
        trace(
            "1 early-partial-collapse",
            &mut |step, budget| {
                let accepted = match step {
                    0 => 2,
                    1 => 0,
                    _ => budget,
                };
                fixed_window(budget, accepted, WINDOW_US[budget])
            },
            60,
        );
        // 2. At budget 1 every draft is accepted and wider budgets are cheaper
        //    per token (code costs), yet probes go to 0 and 2 is not tried.
        trace(
            "2 budget-1-full-accept",
            &mut |step, budget| {
                let accepted = match step {
                    0 => 2,
                    1 => 0,
                    _ => budget,
                };
                fixed_window(budget, accepted, WINDOW_US[budget])
            },
            200,
        );
        // 3. Budget 0 wins the probe (low acceptance), then acceptance
        //    recovers: the budget never leaves 0.
        trace(
            "3 settled-zero",
            &mut |step, budget| {
                let accepted = if step < 40 { 0 } else { budget };
                fixed_window(budget, accepted, WINDOW_US[budget])
            },
            200,
        );
        // 4. Moderate acceptance (deterministic pattern): budget 2 is cheaper
        //    per token than 1, but the 0.85 acceptance gate excludes it.
        let pattern = [1_usize, 0, 2, 1, 0, 2, 1, 1, 0, 2];
        trace(
            "4 moderate-acceptance",
            &mut |step, budget| {
                let accepted = pattern[step % pattern.len()];
                fixed_window(budget, accepted, WINDOW_US[budget])
            },
            120,
        );
    }

    /// Diagnostic comparison of the legacy and window-cost policies on the
    /// same deterministic scenarios (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic report"]
    fn budget_policy_simulation_report() {
        for (name, acceptance) in [("code", &CODE), ("mid", &MID), ("low", &LOW)] {
            let probe = Scenario::new(acceptance, &WINDOW_US, 0);
            let (best, best_cost) = oracle(&probe);
            println!("scenario {name}: oracle fixed b{best} {best_cost:.0} us/token");
            for budget in 0..=7 {
                println!(
                    "  fixed b{budget}: {:.0} us/token",
                    probe.fixed_cost_per_token(budget)
                );
            }
            for kind in [
                DFlash2BudgetPolicyKind::Legacy,
                DFlash2BudgetPolicyKind::WindowCost,
            ] {
                for seed in [1_u64, 2, 3] {
                    let mut scenario = Scenario::new(acceptance, &WINDOW_US, seed);
                    let mut policy = DFlash2DraftBudgetPolicy::new(kind, 7);
                    let run = simulate(&mut policy, &mut scenario, 130);
                    let probes = run
                        .roles
                        .iter()
                        .zip(&run.budgets)
                        .filter(|(role, _)| **role == DFlash2WindowRole::Probe)
                        .map(|(_, budget)| budget.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    println!(
                        "  {:<11} seed {seed}: {:.0} us/token ({:.2}x oracle) histogram {:?} first32 {:?} probes [{probes}]",
                        kind.name(),
                        run.cost_per_token(),
                        run.cost_per_token() / best_cost,
                        run.histogram(7),
                        &run.budgets[..32],
                    );
                }
            }
        }
    }
}
