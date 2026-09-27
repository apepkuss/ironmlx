//! Scoped routing for DFlash2 draft-only projection fusion.
//!
//! Draft logits are proposals rather than committed target logits, so their
//! small-op graph may use the fused QKV and gate/up layouts without changing
//! the exact target-verification contract.  Keeping this scope separate from
//! product-stable target QMM prevents accidental changes to ordinary model
//! execution.

use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn scope() -> Scope {
    DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Scope
}

pub(crate) fn is_armed() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_is_nested_and_thread_local() {
        assert!(!is_armed());
        {
            let _outer = scope();
            let _inner = scope();
            assert!(is_armed());
        }
        assert!(!is_armed());
    }
}
