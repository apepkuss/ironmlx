//! `Element` is the type-class for Rust types that map to MLX dtypes.
//!
//! Sealed (impossible to impl outside this crate) so that downstream code
//! cannot construct an arbitrary `T -> Dtype` mapping that would violate
//! FFI type safety.

use crate::{Array, Dtype, Result};

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

macro_rules! element_stub {
    ($T:ty, $dt:expr) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(_slice: &[Self], _shape: &[i32]) -> Result<Array> {
                unimplemented!("filled in by Task 9")
            }
            fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> {
                unimplemented!("filled in by Task 11")
            }
            fn array_item(_arr: &Array) -> Result<Self> {
                unimplemented!("filled in by Task 10")
            }
        }
    };
}

element_stub!(bool, Dtype::Bool);
element_stub!(u8, Dtype::Uint8);
element_stub!(i8, Dtype::Int8);
element_stub!(i16, Dtype::Int16);
element_stub!(i32, Dtype::Int32);
element_stub!(i64, Dtype::Int64);
element_stub!(half::f16, Dtype::Float16);
element_stub!(half::bf16, Dtype::Bfloat16);
element_stub!(f32, Dtype::Float32);
element_stub!(f64, Dtype::Float64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_const_matches_for_each_element() {
        // Compile-time check that every Element type has the right DTYPE constant.
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
