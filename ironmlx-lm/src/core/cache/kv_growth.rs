//! Growth step of dense K/V buffers.
//!
//! A dense K/V buffer is allocated for the tokens it holds and grows by one
//! step when a write crosses its current capacity; `cap` stays the hard
//! maximum. Allocating the whole `prompt + max_new_tokens` up front reserves
//! output that most requests never generate.

use std::sync::OnceLock;

/// Tokens added to a dense K/V buffer each time it grows.
///
/// One step of Qwen3.8-27B is 256 MB. A grow copies the buffer once; at a
/// 30K context that is about 2 GB read and written, paid once per 4096
/// generated tokens. The step stays at or above the 256-token width below
/// which Metal attention takes its slow path.
pub const KV_GROWTH_STEP_TOKENS: i32 = 4096;

/// Smallest step the diagnostic override accepts.
pub const MIN_KV_GROWTH_STEP_TOKENS: i32 = 256;

/// Diagnostic override of the growth step, used by tests that need a
/// request to cross many step boundaries. Not a tuning knob.
pub const KV_GROWTH_STEP_ENV: &str = "IRONMLX_DIAGNOSTIC_KV_GROWTH_STEP";

/// The growth step in effect: [`KV_GROWTH_STEP_TOKENS`] unless the
/// diagnostic override is set to a value of at least
/// [`MIN_KV_GROWTH_STEP_TOKENS`].
pub fn kv_growth_step_tokens() -> i32 {
    static STEP: OnceLock<i32> = OnceLock::new();
    *STEP.get_or_init(|| {
        std::env::var(KV_GROWTH_STEP_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<i32>().ok())
            .filter(|step| *step >= MIN_KV_GROWTH_STEP_TOKENS)
            .unwrap_or(KV_GROWTH_STEP_TOKENS)
    })
}

/// Step for a dense K/V buffer whose hard maximum is `cap`: the growth
/// step, or `cap` itself when the buffer is smaller than one step.
pub fn kv_growth_step(cap: i32) -> i32 {
    cap.min(kv_growth_step_tokens()).max(1)
}

/// Capacity a dense buffer grows to when it must hold `tokens`: the next
/// step boundary at or above `tokens`, never above `cap`.
pub fn kv_growth_capacity(tokens: usize, cap: usize) -> usize {
    let step = usize::try_from(kv_growth_step_tokens()).unwrap_or(1).max(1);
    tokens.div_ceil(step).saturating_mul(step).min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_never_exceeds_cap() {
        assert_eq!(kv_growth_step(100), 100);
        assert_eq!(kv_growth_step(i32::MAX), kv_growth_step_tokens());
    }

    #[test]
    fn capacity_rounds_up_to_a_step_and_stops_at_cap() {
        let step = usize::try_from(kv_growth_step_tokens()).unwrap();
        assert_eq!(kv_growth_capacity(1, 1_000_000), step);
        assert_eq!(kv_growth_capacity(step, 1_000_000), step);
        assert_eq!(kv_growth_capacity(step + 1, 1_000_000), 2 * step);
        assert_eq!(kv_growth_capacity(step + 1, step + 7), step + 7);
        assert_eq!(kv_growth_capacity(0, 10), 0);
    }
}
