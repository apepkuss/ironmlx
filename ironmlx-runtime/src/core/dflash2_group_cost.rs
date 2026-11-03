//! Measured ordinary/speculative cost for one loaded paged target actor.
use std::collections::HashMap;

use anyhow::{bail, Result};

use super::speculative::MtpDraftCapContextBucket;

pub(crate) const SETTING: &str = "IRONMLX_FLASH_DFLASH_GROUP_POLICY";

pub(crate) fn enabled(qualified: bool) -> Result<bool> {
    if !qualified {
        return Ok(false);
    }
    select(
        qualified,
        std::env::var(SETTING).ok().as_deref(),
        [
            ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET,
            ironmlx_core::m5_profile::settings::DFLASH2_RAGGED_BUDGET,
            ironmlx_core::m5_profile::settings::DFLASH2_RAGGED_SOLO_BUDGET,
            ironmlx_core::m5_profile::settings::DFLASH2_RAGGED_PROBE_INTERVAL,
            super::dflash2_budget::BUDGET_POLICY_SETTING,
            ironmlx_core::m5_profile::settings::DFLASH2_RAGGED_ORDINARY_MIN_WIDTH,
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some()),
    )
}

fn select(qualified: bool, requested: Option<&str>, explicit_budget: bool) -> Result<bool> {
    match requested.map(str::trim) {
        None | Some("") | Some("auto") => Ok(qualified && !explicit_budget),
        Some("off") => Ok(false),
        Some(_) => bail!("{SETTING} must be auto or off"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Regime {
    width: usize,
    context: MtpDraftCapContextBucket,
    draft: usize,
}

impl Regime {
    pub(crate) fn new(width: usize, context: usize, draft: usize) -> Self {
        Self {
            width,
            context: MtpDraftCapContextBucket::for_tokens(context),
            draft,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Choice {
    pub ordinary: bool,
    pub probe: bool,
    pub reset: bool,
}

#[derive(Default)]
struct Cost {
    samples: usize,
    us_per_token: f64,
}

impl Cost {
    fn observe(&mut self, us: f64, tokens: usize) {
        let sample = us / tokens as f64;
        self.us_per_token = if self.samples == 0 {
            sample
        } else {
            0.3_f64.mul_add(sample, 0.7 * self.us_per_token)
        };
        self.samples = self.samples.saturating_add(1);
    }
}

#[derive(Default)]
struct State {
    ordinary: Cost,
    draft: Cost,
    ordinary_run: usize,
    draft_run: usize,
    recovery_left: usize,
    credit_us: f64,
    last_us: Option<u64>,
}

#[derive(Default)]
pub(crate) struct GroupCostPolicy {
    states: HashMap<Regime, State>,
}

impl GroupCostPolicy {
    // Bound retained draft context and retry when the workload changes.
    const MAX_ORDINARY_WINDOWS: usize = 64;
    const IDLE_RESET_US: u64 = 60_000_000;

    pub(crate) fn choose(&mut self, key: Regime, now_us: u64, remaining: usize) -> Choice {
        let state = self.states.entry(key).or_default();
        let reset = state
            .last_us
            .is_some_and(|last| now_us.saturating_sub(last) > Self::IDLE_RESET_US);
        if reset {
            *state = State::default();
        }
        let probe_fits = remaining > 2 * key.draft.saturating_add(1);
        let baseline_probe = state.draft_run >= Self::MAX_ORDINARY_WINDOWS;
        let ordinary = if state.ordinary.samples < 2 || baseline_probe {
            true
        } else if state.draft.samples < 2 || state.recovery_left > 0 {
            !probe_fits
        } else if state.draft.us_per_token < state.ordinary.us_per_token * 0.97 {
            false
        } else {
            // Two windows include catching up the accumulated draft context.
            let probe_cost = state.draft.us_per_token * key.width as f64 * 2.0;
            if probe_fits
                && (state.ordinary_run >= Self::MAX_ORDINARY_WINDOWS
                    || state.credit_us >= probe_cost)
            {
                state.recovery_left = 2;
                false
            } else {
                true
            }
        };
        Choice {
            ordinary,
            probe: baseline_probe
                || (!ordinary
                    && (state.draft.samples < 2
                        || state.recovery_left > 0
                        || state.ordinary.samples < 2)),
            reset,
        }
    }

    pub(crate) fn observe(
        &mut self,
        key: Regime,
        choice: Choice,
        window_us: u64,
        build_us: u64,
        emitted: usize,
        now_us: u64,
    ) {
        if emitted == 0 || window_us == 0 {
            return;
        }
        let state = self.states.entry(key).or_default();
        // Group creation is common to both routes and tracked separately.
        let us = window_us.saturating_sub(build_us).max(1) as f64;
        state.last_us = Some(now_us);
        if choice.ordinary {
            state.ordinary.observe(us, emitted);
            state.ordinary_run = state.ordinary_run.saturating_add(1);
            state.draft_run = 0;
            state.credit_us += us * 0.02;
        } else {
            state.draft.observe(us, emitted);
            state.ordinary_run = 0;
            state.draft_run = state.draft_run.saturating_add(1);
            state.recovery_left = state.recovery_left.saturating_sub(1);
            let extra = (us - state.ordinary.us_per_token * emitted as f64).max(0.0);
            state.credit_us = (state.credit_us - extra).max(0.0);
        }
        // Short requests cannot save an unlimited probe burst for a later one.
        let probe_credit_cap = if state.draft.samples > 0 {
            state.draft.us_per_token * key.width as f64 * 2.0
        } else {
            state.ordinary.us_per_token * key.width as f64 * 2.0 * (key.draft + 1) as f64
        };
        state.credit_us = state.credit_us.min(probe_credit_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(p: &mut GroupCostPolicy, key: Regime, draft_cost: u64) -> Choice {
        let c = p.choose(key, 1, 256);
        let (us, emitted) = if c.ordinary {
            (100, 4)
        } else {
            (draft_cost, 8)
        };
        p.observe(key, c, us, 0, emitted, 1);
        c
    }

    #[test]
    fn costly_drafts_fall_back_and_recover_when_cost_improves() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        assert!(run(&mut p, key, 600).ordinary);
        assert!(run(&mut p, key, 600).ordinary);
        assert!(run(&mut p, key, 600).probe);
        assert!(run(&mut p, key, 600).probe);
        for _ in 0..60 {
            assert!(run(&mut p, key, 600).ordinary);
        }
        for _ in 0..600 {
            run(&mut p, key, 50);
        }
        assert!(!p.choose(key, 1, 256).ordinary);
    }

    #[test]
    fn profitable_drafts_and_width_context_and_idle_are_independent() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        for _ in 0..5 {
            run(&mut p, key, 100);
        }
        assert!(!p.choose(key, 1, 256).ordinary);
        assert!(p.choose(Regime::new(2, 100, 3), 1, 256).ordinary);
        assert!(p.choose(Regime::new(4, 9000, 3), 1, 256).ordinary);
        let reset = p.choose(key, GroupCostPolicy::IDLE_RESET_US + 2, 256);
        assert!(reset.reset && reset.ordinary);
    }

    #[test]
    fn short_outputs_do_not_spend_their_horizon_on_calibration() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        for remaining in (1..=7).rev() {
            let c = p.choose(key, 1, remaining);
            assert!(c.ordinary && !c.probe);
            p.observe(key, c, 100, 0, 4, 1);
        }
        assert!(p.choose(key, 1, 32).probe);
    }

    #[test]
    fn short_request_credit_cannot_trigger_a_later_probe_burst() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        for _ in 0..1000 {
            let c = p.choose(key, 1, 7);
            assert!(c.ordinary);
            p.observe(key, c, 100, 0, 4, 1);
        }
        assert!(run(&mut p, key, 600).probe);
        assert!(run(&mut p, key, 600).probe);
        for _ in 0..16 {
            assert!(run(&mut p, key, 600).ordinary);
        }
    }

    #[test]
    fn profitable_execution_rechecks_the_ordinary_baseline() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        for _ in 0..10 {
            run(&mut p, key, 100);
        }
        for _ in 0..600 {
            let c = p.choose(key, 1, 256);
            p.observe(
                key,
                c,
                if c.ordinary { 20 } else { 100 },
                0,
                if c.ordinary { 4 } else { 8 },
                1,
            );
        }
        assert!(p.choose(key, 1, 256).ordinary);
    }

    #[test]
    fn failed_windows_do_not_train_and_manual_budgets_win() {
        let mut p = GroupCostPolicy::default();
        let key = Regime::new(4, 100, 3);
        for _ in 0..10 {
            let c = p.choose(key, 1, 256);
            p.observe(key, c, 100, 0, 0, 1);
        }
        assert!(p.choose(key, 1, 256).ordinary);
        assert!(select(true, Some("auto"), false).unwrap());
        assert!(!select(true, Some("auto"), true).unwrap());
        assert!(!select(false, None, false).unwrap());
        assert!(!select(true, Some("off"), false).unwrap());
        assert!(select(true, Some("on"), false).is_err());
    }
}
