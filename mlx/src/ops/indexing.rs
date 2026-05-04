//! Indexing ops: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`.

use crate::{broadcast, Array, Error, Result};

/// Element-wise conditional select: `cond ? x : y`, with NumPy broadcasting
/// across all three operands.
///
/// `cond` is typically a `bool` array but MLX accepts any numeric dtype
/// (non-zero is treated as true).
///
/// Trailing underscore in the name avoids the Rust `where` keyword.
pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    // Validate broadcast compatibility in two steps: cond+x, then result+y.
    // This produces structured Error::BroadcastMismatch instead of opaque MLX strings.
    let cond_x = broadcast::broadcast_shape(&cond.shape(), &x.shape())?;
    broadcast::broadcast_shape(&cond_x, &y.shape())?;
    let inner = mlx_sys::array::ffi::array_where(cond.as_inner(), x.as_inner(), y.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
