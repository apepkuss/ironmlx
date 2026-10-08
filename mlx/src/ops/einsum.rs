//! Einstein summation with MLX's native contraction order and layout.
use crate::{Array, Error, Result, StreamOrDevice};
pub fn einsum(equation: &str, operands: &[&Array]) -> Result<Array> {
    einsum_on(equation, operands, ())
}
pub fn einsum_on(
    equation: &str,
    operands: &[&Array],
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let mut arrays = mlx_sys::compile::ffi::array_vec_new();
    for operand in operands {
        mlx_sys::compile::ffi::array_vec_push(arrays.pin_mut(), operand.as_inner());
    }
    let (has, device_only, device, index) = target.into().encode();
    // SAFETY: operands remain borrowed throughout the contraction call; C++
    // retains reference-counted arrays in the returned graph. Exceptions propagate.
    let inner = unsafe {
        mlx_sys::einsum::ffi::ops_einsum(equation, &arrays, has, device_only, device, index)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contractions_preserve_results_and_recover_from_bad_equations() -> Result<()> {
        let a: Array = (&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], (2, 3)).try_into()?;
        let b: Array = (&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], (3, 2)).try_into()?;
        let result = einsum_on("ij,jk->ik", &[&a, &b], crate::Device::cpu())?;
        assert_eq!(result.shape().as_slice(), [2, 2]);
        assert_eq!(result.to_vec::<f32>()?, vec![22.0, 28.0, 49.0, 64.0]);
        assert!(einsum("ij,jk->ix", &[&a, &b]).is_err());
        let c: Array = (&[1.0f32, 2.0, 3.0][..], (3,)).try_into()?;
        assert_eq!(
            einsum_on("i,i,i->", &[&c, &c, &c], crate::Device::cpu())?.item::<f32>()?,
            36.0
        );
        Ok(())
    }
}
