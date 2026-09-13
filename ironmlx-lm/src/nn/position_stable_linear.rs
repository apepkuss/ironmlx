//! Scoped full-precision Linear execution that preserves repeated Q=1 shapes.

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
    fn scope_restores_nested_thread_local_state() {
        assert!(!is_armed());
        {
            let _outer = scope();
            let _inner = scope();
            assert!(is_armed());
        }
        assert!(!is_armed());
    }
}
