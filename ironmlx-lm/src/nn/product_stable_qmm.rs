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
    } else if let Some(max_nv) = wide_rows(x, weight, biases, transpose, group_size, bits, mode) {
        Ok(mlx::quantization::qmv_fast_wide_on(
            x,
            weight,
            scales,
            biases.expect("checked"),
            max_nv,
            target,
        )?)
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

/// Most rows per threadgroup of MLX's product-stable `qmv_fast_wide` kernel
/// for the operands MLX would send to it, under `moe_fast_path` (MLX itself
/// uses two).
fn wide_rows(
    x: &Array,
    weight: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Option<i32> {
    if !super::moe_fast_path::is_armed() {
        return None;
    }
    let k = *x.shape().as_slice().last()?;
    if k <= 0 {
        return None;
    }
    let w = weight.shape();
    let rows = x.size() / usize::try_from(k).ok()?;
    (bits == 4
        && group_size == 64
        && mode == "affine"
        && transpose
        && rows > 2
        && x.dtype() == mlx::Dtype::Bfloat16
        && biases.is_some_and(|b| b.dtype() == mlx::Dtype::Bfloat16)
        && w.as_slice().len() == 2
        && w.as_slice()[1] * 8 == k
        && k % 512 == 0
        && w.as_slice()[0] % 8 == 0)
        .then_some(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn wide_rows_declines_zero_width_inputs() {
        let _armed = super::super::moe_fast_path::scope();
        let x = Array::zeros(&[3, 0][..], mlx::Dtype::Bfloat16).unwrap();
        let w = Array::zeros(&[32, 0][..], mlx::Dtype::Uint32).unwrap();
        let b = Array::zeros(&[32, 8][..], mlx::Dtype::Bfloat16).unwrap();
        assert_eq!(wide_rows(&x, &w, Some(&b), true, 64, 4, "affine"), None);
        let x = Array::zeros(&[3, 512][..], mlx::Dtype::Bfloat16).unwrap();
        let w = Array::zeros(&[32, 64][..], mlx::Dtype::Uint32).unwrap();
        assert_eq!(wide_rows(&x, &w, Some(&b), true, 64, 4, "affine"), Some(5));
    }

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
    #[serial_test::serial(mlx_metal)]
    fn wider_row_groups_keep_every_product_stable_affine4_bit() {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
        }
        let key = mlx::random::key(5).unwrap();
        let normal = |shape: &[i32], scale: f64| {
            let a = mlx::random::normal()
                .shape(shape)
                .dtype(mlx::Dtype::Float32)
                .scale(scale)
                .key(&key)
                .sample()
                .unwrap();
            mlx::ops::cast::astype(&a, mlx::Dtype::Bfloat16).unwrap()
        };
        let bits = |a: &Array| -> Vec<u32> {
            mlx::ops::cast::astype(a, mlx::Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        for (n, k) in [(1024, 2048), (520, 512)] {
            let q = mlx::quantization::quantize(
                &normal(&[n, k], 0.05),
                Some(64),
                Some(4),
                "affine",
                None,
            )
            .unwrap();
            for rows in 1..=8 {
                let x = normal(&[1, rows, k], 2.0);
                let run = || {
                    forward_on(
                        &x,
                        &q[0],
                        &q[1],
                        Some(&q[2]),
                        true,
                        64,
                        4,
                        "affine",
                        StreamOrDevice::default(),
                    )
                    .unwrap()
                };
                let mlx_bits = bits(&run());
                let _fast = super::super::moe_fast_path::scope();
                assert_eq!(bits(&run()), mlx_bits, "N={n} K={k} rows={rows}");
            }
        }
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
