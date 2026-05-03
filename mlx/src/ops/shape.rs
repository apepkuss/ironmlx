//! Shape ops: reshape, transpose family, broadcast_to, concatenate, stack, split.

use smallvec::SmallVec;

use crate::{Array, Error, Result};

/// Reshape an array to the given shape. A single `-1` in the shape is replaced
/// by the inferred size; multiple `-1`s or a non-divisible product return
/// `Err(Error::Mlx)`.
pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    let total: usize = a.size();
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    let resolved: SmallVec<[i32; 8]> = match neg_count {
        0 => shape.iter().copied().collect(),
        1 => {
            let known: usize = shape
                .iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .product();
            if known == 0 || total % known != 0 {
                return Err(Error::Mlx(format!(
                    "reshape: cannot infer -1 dim — total {total} not divisible by product {known} of remaining dims {shape:?}"
                )));
            }
            let inferred = (total / known) as i32;
            shape
                .iter()
                .map(|&d| if d == -1 { inferred } else { d })
                .collect()
        }
        _ => {
            return Err(Error::Mlx(format!(
                "reshape: at most one -1 placeholder allowed, got {neg_count} in {shape:?}"
            )))
        }
    };
    let inner =
        mlx_sys::array::ffi::array_reshape(a.as_inner(), &resolved).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
