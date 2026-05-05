//! Matrix multiplication.
//!
//! `matmul(a, b)` covers all NumPy-/MLX-style matmul cases:
//!
//! - 2D × 2D: standard matrix product `[M, K] @ [K, N] → [M, N]`
//! - Batched: `[B..., M, K] @ [B..., K, N] → [B..., M, N]`
//! - Broadcasting on batch dims: `[B, 1, M, K] @ [1, H, K, N] → [B, H, M, N]`
//!
//! MLX handles all dispatch internally; this is a single FFI thin wrapper.

use crate::{Array, Error, Result};

/// Matrix multiplication. See module docs for shape rules.
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_matmul(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
