//! Scoped routing for product-stable affine quantized projections.
//!
//! Qwen MTP uses this for its complete draft head and draft-logit projection.
//! The underlying MLX primitive preserves the single-row accumulation tree
//! while evaluating multiple rows in one dispatch.

use std::cell::Cell;

use mlx::{Array, StreamOrDevice};

use crate::Result;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static AFFINE8_WIDE_DEPTH: Cell<u32> = const { Cell::new(0) };
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

pub(crate) struct Affine8WideScope {
    _product_stable: Scope,
}

impl Drop for Affine8WideScope {
    fn drop(&mut self) {
        AFFINE8_WIDE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn affine8_wide_scope() -> Affine8WideScope {
    AFFINE8_WIDE_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Affine8WideScope {
        _product_stable: scope(),
    }
}

pub(crate) fn affine8_wide_is_armed() -> bool {
    AFFINE8_WIDE_DEPTH.with(|depth| depth.get() > 0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_on(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
    target: StreamOrDevice,
) -> Result<Array> {
    if bits == 8 && affine8_wide_is_armed() {
        Ok(
            mlx::quantization::quantized_matmul_product_stable_affine8_wide_on(
                x,
                weight,
                scales,
                biases,
                transpose,
                Some(group_size),
                Some(bits),
                mode,
                target,
            )?,
        )
    } else {
        Ok(mlx::quantization::quantized_matmul_product_stable_on(
            x,
            weight,
            scales,
            biases,
            transpose,
            Some(group_size),
            Some(bits),
            mode,
            target,
        )?)
    }
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
    fn affine8_wide_scope_arms_product_stable_and_restores_state() {
        assert!(!is_armed());
        assert!(!affine8_wide_is_armed());
        {
            let _wide = affine8_wide_scope();
            assert!(is_armed());
            assert!(affine8_wide_is_armed());
        }
        assert!(!is_armed());
        assert!(!affine8_wide_is_armed());
    }
}
