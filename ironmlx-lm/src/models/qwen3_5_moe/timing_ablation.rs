//! Timing attribution only. A benchmark may zero the output of selected
//! components on its own thread to measure their share of a forward; outputs
//! computed while any flag is set are not valid model outputs and must never
//! reach a generation path. All flags are off unless set explicitly.

use std::cell::Cell;

pub const ROUTED_EXPERTS: u32 = 1;
pub const SHARED_EXPERT: u32 = 1 << 1;
pub const FULL_ATTENTION: u32 = 1 << 2;
pub const LINEAR_ATTENTION: u32 = 1 << 3;
/// GDN conv, recurrence and gated norm only; the input and output
/// projections still run (on zeros).
pub const GDN_CORE: u32 = 1 << 4;
/// Full-attention cache update and SDPA only; the q/k/v and o projections
/// still run (on zeros).
pub const ATTN_CORE: u32 = 1 << 5;
/// Read by benchmarks only: skip the LM-head projection after a forward.
pub const LM_HEAD: u32 = 1 << 6;

thread_local! {
    static FLAGS: Cell<u32> = const { Cell::new(0) };
}

/// Set the ablated components for this thread (0 restores normal execution).
pub fn set(flags: u32) {
    FLAGS.with(|cell| cell.set(flags));
}

pub fn skips(component: u32) -> bool {
    FLAGS.with(|cell| cell.get() & component != 0)
}
