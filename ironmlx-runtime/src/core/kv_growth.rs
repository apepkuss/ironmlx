//! Charging dense K/V as it grows.
//!
//! A request is admitted for the K/V its prompt needs plus a lookahead, not
//! for its whole `max_new_tokens`. Before each decode step the runtime
//! extends a row's charge one growth step ahead of its context. When the
//! memory budget or the process-memory governor refuses an extension, the
//! prefix cache yields first; if the extension is still refused, the row's
//! output budget shrinks to what its current charge holds and the row ends
//! with `finish_reason = "length"`.

use std::sync::atomic::{AtomicU64, Ordering};

use ironmlx_lm::core::cache::kv_growth_capacity;

/// Minimum tokens a row's charge stays ahead of its context. Must exceed the
/// widest step a row writes so a refused extension still leaves room for
/// the next token.
pub(crate) const KV_GROWTH_MIN_LOOKAHEAD_TOKENS: usize = 256;

/// Rows whose output budget shrank because an extension was refused.
static KV_GROWTH_DENIED_TOTAL: AtomicU64 = AtomicU64::new(0);

pub fn kv_growth_denied_total() -> u64 {
    KV_GROWTH_DENIED_TOTAL.load(Ordering::Relaxed)
}

pub(crate) fn record_kv_growth_denied() {
    KV_GROWTH_DENIED_TOTAL.fetch_add(1, Ordering::Relaxed);
}

/// Lookahead for a runtime whose widest step writes `max_step_tokens`. A
/// row's K/V may run up to one step ahead of the tokens it has emitted, and
/// the next step writes up to one more, so the lookahead covers two steps
/// and leaves a token of room.
pub(crate) fn kv_growth_lookahead(max_step_tokens: usize) -> usize {
    KV_GROWTH_MIN_LOOKAHEAD_TOKENS.max(max_step_tokens.saturating_mul(2).saturating_add(1))
}

/// Tokens charged at admission: the prompt plus the lookahead, rounded up to
/// a growth step, never above the request's own capacity.
pub(crate) fn initial_charge_tokens(
    prompt_tokens: usize,
    request_cap: usize,
    lookahead: usize,
) -> usize {
    kv_growth_capacity(prompt_tokens.saturating_add(lookahead), request_cap)
}

/// The charge a row at `context_tokens` must grow to before its next step,
/// or `None` when its current charge still covers the lookahead or already
/// covers the whole request.
pub(crate) fn extended_charge_tokens(
    context_tokens: usize,
    charged_tokens: usize,
    request_cap: usize,
    lookahead: usize,
) -> Option<usize> {
    if charged_tokens >= request_cap {
        return None;
    }
    let needed = context_tokens.saturating_add(lookahead);
    if needed <= charged_tokens {
        return None;
    }
    Some(kv_growth_capacity(needed, request_cap).max(charged_tokens))
}

/// Output budget of a row whose extension was refused: what the current
/// charge holds beyond the prompt, at least one token past what it has
/// generated, never above the requested budget.
pub(crate) fn clamped_max_new_tokens(
    prompt_tokens: usize,
    generated_tokens: usize,
    charged_tokens: usize,
    scratch_tokens: usize,
    requested_max_new_tokens: usize,
) -> usize {
    charged_tokens
        .saturating_sub(prompt_tokens)
        .saturating_sub(scratch_tokens)
        .max(generated_tokens.saturating_add(1))
        .min(requested_max_new_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmlx_lm::core::cache::kv_growth_step_tokens;

    fn step() -> usize {
        usize::try_from(kv_growth_step_tokens()).unwrap()
    }

    #[test]
    fn admission_charges_the_prompt_step_not_the_output_budget() {
        let lookahead = kv_growth_lookahead(1);
        let prompt = 1_000;
        let charged = initial_charge_tokens(prompt, prompt + 32_768, lookahead);
        assert_eq!(charged, step());
        // A short request is charged no more than it can use.
        assert_eq!(initial_charge_tokens(10, 300, lookahead), 300);
    }

    #[test]
    fn extension_waits_for_the_lookahead_and_stops_at_the_request_cap() {
        let lookahead = kv_growth_lookahead(1);
        let cap = 3 * step();
        let charged = step();
        assert_eq!(
            extended_charge_tokens(charged - lookahead, charged, cap, lookahead),
            None
        );
        assert_eq!(
            extended_charge_tokens(charged - lookahead + 1, charged, cap, lookahead),
            Some(2 * step())
        );
        assert_eq!(
            extended_charge_tokens(cap - 1, cap, cap, lookahead),
            None,
            "a row charged for its whole request never extends"
        );
        assert_eq!(
            extended_charge_tokens(cap - 10, 2 * step(), cap, lookahead),
            Some(cap)
        );
    }

    #[test]
    fn lookahead_exceeds_the_widest_step() {
        assert_eq!(kv_growth_lookahead(8), KV_GROWTH_MIN_LOOKAHEAD_TOKENS);
        assert_eq!(kv_growth_lookahead(200), 401);
    }

    #[test]
    fn refused_extension_shrinks_the_budget_to_the_charge() {
        let prompt = 1_000;
        let charged = step();
        // Room left within the charge becomes the remaining budget.
        assert_eq!(
            clamped_max_new_tokens(prompt, 2_900, charged, 0, 32_768),
            charged - prompt
        );
        // Verify scratch must fit inside the charge as well.
        assert_eq!(
            clamped_max_new_tokens(prompt, 2_900, charged, 16, 32_768),
            charged - prompt - 16
        );
        // Never below one more token, never above the request.
        assert_eq!(
            clamped_max_new_tokens(prompt, 5_000, charged, 0, 32_768),
            5_001
        );
        assert_eq!(clamped_max_new_tokens(prompt, 10, charged, 0, 200), 200);
    }
}
