//! Scoped routing for batch-invariant quantized projections.
//!
//! Some MLX QMM shapes produce slightly different results when the leading
//! batch dimension changes. Callers arm a narrow model context, and only the
//! affected projection blocks opt their `Linear` layers into per-row QMM.

use std::cell::Cell;

thread_local! {
    static CONTEXT_DEPTH: Cell<u32> = const { Cell::new(0) };
    static LINEAR_DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct ContextScope;

impl Drop for ContextScope {
    fn drop(&mut self) {
        CONTEXT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn context_scope() -> ContextScope {
    CONTEXT_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    ContextScope
}

pub(crate) fn context_is_armed() -> bool {
    CONTEXT_DEPTH.with(|depth| depth.get() > 0)
}

pub(crate) struct LinearScope;

impl Drop for LinearScope {
    fn drop(&mut self) {
        LINEAR_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn linear_scope() -> LinearScope {
    LINEAR_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    LinearScope
}

pub(crate) fn linear_is_armed() -> bool {
    LINEAR_DEPTH.with(|depth| depth.get() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_restore_nested_thread_local_state() {
        assert!(!context_is_armed());
        assert!(!linear_is_armed());
        {
            let _context = context_scope();
            let _nested_context = context_scope();
            let _linear = linear_scope();
            assert!(context_is_armed());
            assert!(linear_is_armed());
        }
        assert!(!context_is_armed());
        assert!(!linear_is_armed());
    }
}
