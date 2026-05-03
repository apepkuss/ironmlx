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

    /// Construct a new array from a slice of `Self`.
    /// Implementation is per-dtype because cxx FFI is monomorphized on T.
    #[doc(hidden)]
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array>;

    /// Read all elements out as a `Vec<Self>`. Implicitly evals if needed.
    #[doc(hidden)]
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>>;

    /// Read the single scalar element. Caller already verified `arr.size() == 1`.
    #[doc(hidden)]
    fn array_item(arr: &Array) -> Result<Self>;
}

// === Implementations ===
//
// Pattern: each type's array_from delegates to the corresponding shim
// function, with bool/f16/bf16 doing transparent reinterpret as needed.

impl sealed::Sealed for bool {}
impl Element for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // cxx::Slice doesn't accept &[bool]; convert to &[u8] (each true → 1, false → 0).
        let bytes: Vec<u8> = slice.iter().map(|&b| b as u8).collect();
        let inner = mlx_sys::array::ffi::array_from_bool(&bytes, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("filled in by Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("filled in by Task 10") }
}

macro_rules! element_impl_simple {
    ($T:ty, $dt:expr, $shim_from:ident) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
                let inner = mlx_sys::array::ffi::$shim_from(slice, shape).map_err(Error::from)?;
                Ok(Array::from_inner(inner))
            }
            fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("filled in by Task 11") }
            fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("filled in by Task 10") }
        }
    };
}

element_impl_simple!(u8, Dtype::Uint8, array_from_u8);
element_impl_simple!(i8, Dtype::Int8, array_from_i8);
element_impl_simple!(i16, Dtype::Int16, array_from_i16);
element_impl_simple!(i32, Dtype::Int32, array_from_i32);
element_impl_simple!(i64, Dtype::Int64, array_from_i64);
element_impl_simple!(f32, Dtype::Float32, array_from_f32);
element_impl_simple!(f64, Dtype::Float64, array_from_f64);

// f16/bf16 reinterpret through &[u16] (half::f16 is repr(transparent) over u16).
impl sealed::Sealed for half::f16 {}
impl Element for half::f16 {
    const DTYPE: Dtype = Dtype::Float16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // SAFETY: half::f16 is #[repr(transparent)] over u16 (documented invariant of the
        // half crate), and the shim function takes a u16 slice that it reinterprets to
        // mlx::core::float16_t (also a 2-byte POD with identical bit layout).
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_f16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("filled in by Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("filled in by Task 10") }
}

impl sealed::Sealed for half::bf16 {}
impl Element for half::bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // SAFETY: half::bf16 is #[repr(transparent)] over u16 (documented invariant of the
        // half crate); shim reinterprets to mlx::core::bfloat16_t (identical 2-byte layout).
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_bf16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("filled in by Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("filled in by Task 10") }
}

// Compile-time guarantee that f16/bf16 are 2 bytes (matching mlx::core::float16_t/bfloat16_t).
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
