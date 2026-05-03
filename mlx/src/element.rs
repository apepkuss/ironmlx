//! `Element` is the type-class for Rust types that map to MLX dtypes.
//!
//! Sealed (impossible to impl outside this crate) so that downstream code
//! cannot construct an arbitrary `T -> Dtype` mapping that would violate
//! FFI type safety.

use crate::{Array, Dtype, Error, Result};

mod sealed {
    pub trait Sealed {}
}

pub trait Element: sealed::Sealed + Copy + Send + 'static {
    const DTYPE: Dtype;

    #[doc(hidden)]
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array>;

    #[doc(hidden)]
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>>;

    #[doc(hidden)]
    fn array_item(arr: &Array) -> Result<Self>;
}

// === Implementations ===

impl sealed::Sealed for bool {}
impl Element for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        let bytes: Vec<u8> = slice.iter().map(|&b| b as u8).collect();
        let inner = mlx_sys::array::ffi::array_from_bool(&bytes, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
        arr.eval()?;  // implicit eval per spec A8
        let bytes = mlx_sys::array::ffi::array_to_vec_bool(arr.as_inner()).map_err(Error::from)?;
        Ok(bytes.into_iter().map(|b| b != 0).collect())
    }
    fn array_item(arr: &Array) -> Result<Self> {
        mlx_sys::array::ffi::array_item_bool(arr.as_inner()).map_err(Error::from)
    }
}

macro_rules! element_impl_simple {
    ($T:ty, $dt:expr, $shim_from:ident, $shim_item:ident, $shim_to_vec:ident) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
                let inner = mlx_sys::array::ffi::$shim_from(slice, shape).map_err(Error::from)?;
                Ok(Array::from_inner(inner))
            }
            fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
                arr.eval()?;
                let raw = mlx_sys::array::ffi::$shim_to_vec(arr.as_inner()).map_err(Error::from)?;
                Ok(raw.into_iter().collect::<Vec<_>>())
            }
            fn array_item(arr: &Array) -> Result<Self> {
                mlx_sys::array::ffi::$shim_item(arr.as_inner()).map_err(Error::from)
            }
        }
    };
}

element_impl_simple!(u8, Dtype::Uint8, array_from_u8, array_item_u8, array_to_vec_u8);
element_impl_simple!(i8, Dtype::Int8, array_from_i8, array_item_i8, array_to_vec_i8);
element_impl_simple!(i16, Dtype::Int16, array_from_i16, array_item_i16, array_to_vec_i16);
element_impl_simple!(i32, Dtype::Int32, array_from_i32, array_item_i32, array_to_vec_i32);
element_impl_simple!(i64, Dtype::Int64, array_from_i64, array_item_i64, array_to_vec_i64);
element_impl_simple!(f32, Dtype::Float32, array_from_f32, array_item_f32, array_to_vec_f32);
element_impl_simple!(f64, Dtype::Float64, array_from_f64, array_item_f64, array_to_vec_f64);

// f16/bf16 reinterpret through u16.
impl sealed::Sealed for half::f16 {}
impl Element for half::f16 {
    const DTYPE: Dtype = Dtype::Float16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_f16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
        arr.eval()?;
        let raw = mlx_sys::array::ffi::array_to_vec_f16(arr.as_inner()).map_err(Error::from)?;
        Ok(raw.into_iter().map(half::f16::from_bits).collect())
    }
    fn array_item(arr: &Array) -> Result<Self> {
        let bits = mlx_sys::array::ffi::array_item_f16(arr.as_inner()).map_err(Error::from)?;
        Ok(half::f16::from_bits(bits))
    }
}

impl sealed::Sealed for half::bf16 {}
impl Element for half::bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_bf16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
        arr.eval()?;
        let raw = mlx_sys::array::ffi::array_to_vec_bf16(arr.as_inner()).map_err(Error::from)?;
        Ok(raw.into_iter().map(half::bf16::from_bits).collect())
    }
    fn array_item(arr: &Array) -> Result<Self> {
        let bits = mlx_sys::array::ffi::array_item_bf16(arr.as_inner()).map_err(Error::from)?;
        Ok(half::bf16::from_bits(bits))
    }
}

const _: () = {
    assert!(std::mem::size_of::<half::f16>() == 2);
    assert!(std::mem::size_of::<half::bf16>() == 2);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_const_matches_for_each_element() {
        assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
        assert_eq!(<u8 as Element>::DTYPE, Dtype::Uint8);
        assert_eq!(<i8 as Element>::DTYPE, Dtype::Int8);
        assert_eq!(<i16 as Element>::DTYPE, Dtype::Int16);
        assert_eq!(<i32 as Element>::DTYPE, Dtype::Int32);
        assert_eq!(<i64 as Element>::DTYPE, Dtype::Int64);
        assert_eq!(<half::f16 as Element>::DTYPE, Dtype::Float16);
        assert_eq!(<half::bf16 as Element>::DTYPE, Dtype::Bfloat16);
        assert_eq!(<f32 as Element>::DTYPE, Dtype::Float32);
        assert_eq!(<f64 as Element>::DTYPE, Dtype::Float64);
    }
}
