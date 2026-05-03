//! NumPy-style broadcasting shape inference.
//!
//! Used by binary operators (`Add`/`Sub`/`Mul`/`Div`) before dispatching to MLX
//! to produce structured `Error::BroadcastMismatch` errors with `lhs`/`rhs`
//! fields, instead of relying on MLX's English exception strings.
//!
//! The same algorithm will be reused by P1b2 reductions (computing keepdim
//! shapes) and `broadcast_to` op.

use smallvec::SmallVec;

use crate::{Error, Result};

/// Compute the broadcast result shape of two operand shapes per NumPy rules.
///
/// Returns `Err(Error::BroadcastMismatch)` if the shapes are incompatible.
pub fn broadcast_shape(lhs: &[i32], rhs: &[i32]) -> Result<SmallVec<[i32; 8]>> {
    let n = lhs.len().max(rhs.len());
    let mut out = SmallVec::<[i32; 8]>::with_capacity(n);
    for i in 0..n {
        // Right-align: treat missing leading dims as 1.
        let a = lhs.get(lhs.len().wrapping_sub(n - i)).copied().unwrap_or(1);
        let b = rhs.get(rhs.len().wrapping_sub(n - i)).copied().unwrap_or(1);
        let dim = match (a, b) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(Error::BroadcastMismatch {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            }
        };
        out.push(dim);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_shapes() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn missing_leading_dim_is_one() {
        // [2, 3] vs [3] → [2, 3] (rhs treated as [1, 3])
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn one_dim_expands_in_middle() {
        // [2, 1, 4] vs [3, 4] → right-align: [2, 1, 4] vs [_, 3, 4] → [2, 3, 4]
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 4]).unwrap().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn scalar_broadcasts_to_anything() {
        // empty shape (scalar) vs [2, 3] → [2, 3]
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap().as_slice(), &[2, 3]);
        assert_eq!(broadcast_shape(&[2, 3], &[]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn both_scalars() {
        let result = broadcast_shape(&[], &[]).unwrap();
        assert_eq!(result.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn incompatible_dim_errors() {
        // [2, 3] vs [2, 4] → mismatch at axis 1 (neither is 1)
        let err = broadcast_shape(&[2, 3], &[2, 4]).unwrap_err();
        match err {
            Error::BroadcastMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![2, 4]);
            }
            other => panic!("expected BroadcastMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rank_mismatch_with_incompatible_dim() {
        // [3] vs [2, 4] → right-align: [_, 3] vs [2, 4] → mismatch at axis 1
        let err = broadcast_shape(&[3], &[2, 4]).unwrap_err();
        assert!(matches!(err, Error::BroadcastMismatch { .. }));
    }
}
