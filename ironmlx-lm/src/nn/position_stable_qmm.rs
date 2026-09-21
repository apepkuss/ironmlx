//! Scoped routing for quantized projections whose output must retain the
//! numerical morphology of independent Q=1 calls at every sequence position.

use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static EXACT_AFFINE8_B4_Q2_DEPTH: Cell<u32> = const { Cell::new(0) };
    static DFLASH2_BULK_ATTENTION_DEPTH: Cell<u32> = const { Cell::new(0) };
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

pub(crate) struct ExactAffine8B4Q2Scope;

impl Drop for ExactAffine8B4Q2Scope {
    fn drop(&mut self) {
        EXACT_AFFINE8_B4_Q2_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn exact_affine8_b4_q2_scope() -> ExactAffine8B4Q2Scope {
    EXACT_AFFINE8_B4_Q2_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    ExactAffine8B4Q2Scope
}

pub(crate) fn exact_affine8_b4_q2_is_armed() -> bool {
    EXACT_AFFINE8_B4_Q2_DEPTH.with(|depth| depth.get() > 0)
}

pub(crate) struct DFlash2BulkAttentionScope;

impl Drop for DFlash2BulkAttentionScope {
    fn drop(&mut self) {
        DFLASH2_BULK_ATTENTION_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn dflash2_bulk_attention_scope() -> DFlash2BulkAttentionScope {
    DFLASH2_BULK_ATTENTION_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    DFlash2BulkAttentionScope
}

pub(crate) fn dflash2_bulk_attention_is_armed() -> bool {
    DFLASH2_BULK_ATTENTION_DEPTH.with(|depth| depth.get() > 0)
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

    #[test]
    fn exact_affine8_b4_q2_scope_restores_nested_thread_local_state() {
        assert!(!exact_affine8_b4_q2_is_armed());
        {
            let _outer = exact_affine8_b4_q2_scope();
            let _inner = exact_affine8_b4_q2_scope();
            assert!(exact_affine8_b4_q2_is_armed());
        }
        assert!(!exact_affine8_b4_q2_is_armed());
    }

    #[test]
    fn dflash2_bulk_attention_scope_restores_nested_thread_local_state() {
        assert!(!dflash2_bulk_attention_is_armed());
        {
            let _outer = dflash2_bulk_attention_scope();
            let _inner = dflash2_bulk_attention_scope();
            assert!(dflash2_bulk_attention_is_armed());
        }
        assert!(!dflash2_bulk_attention_is_armed());
    }
}
