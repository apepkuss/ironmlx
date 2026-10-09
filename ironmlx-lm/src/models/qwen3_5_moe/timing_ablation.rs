//! Timing attribution only. A benchmark may zero the output of selected
//! components on its own thread to measure their share of a forward; outputs
//! computed while any flag is set are not valid model outputs and must never
//! reach a generation path. All flags are off unless set explicitly.

use std::cell::Cell;

pub const ROUTED_EXPERTS: u32 = 1;
pub const SHARED_EXPERT: u32 = 1 << 1;
pub const FULL_ATTENTION: u32 = 1 << 2;
pub const LINEAR_ATTENTION: u32 = 1 << 3;

thread_local! {
    static FLAGS: Cell<u32> = const { Cell::new(0) };
}

/// Set the ablated components for this thread (0 restores normal execution).
pub fn set(flags: u32) {
    FLAGS.with(|cell| cell.set(flags));
}

pub(crate) fn skips(component: u32) -> bool {
    FLAGS.with(|cell| cell.get() & component != 0)
}
