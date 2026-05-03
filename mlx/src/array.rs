use cxx::UniquePtr;

use crate::{Dtype, Error, Result};

/// An MLX array. Cheap to clone (MLX internally refcounts the storage).
pub struct Array(UniquePtr<mlx_sys::array::ffi::MlxArray>);

impl Array {
    /// Create an array filled with zeros of the given shape and dtype.
    /// The result is lazy — call [`Array::eval`] before reading the data.
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Self> {
        let inner = mlx_sys::array::ffi::array_zeros(shape, dtype.as_u8())
            .map_err(Error::from)?;
        Ok(Array(inner))
    }

    /// The shape of the array. `[]` denotes a scalar.
    pub fn shape(&self) -> Vec<i32> {
        mlx_sys::array::ffi::array_shape(&self.0)
    }

    /// The dtype of the array.
    pub fn dtype(&self) -> Dtype {
        // The shim only ever returns values produced by static_cast<uint8_t>(Dtype::Val),
        // so a missing variant means MLX was upgraded with a new dtype — surface it as a panic
        // (this is a programmer error, not a runtime condition).
        let raw = mlx_sys::array::ffi::array_dtype(&self.0);
        Dtype::from_u8(raw).expect("MLX returned unknown Dtype::Val — mlx-sys/mlx version mismatch")
    }

    pub fn ndim(&self) -> usize {
        mlx_sys::array::ffi::array_ndim(&self.0)
    }

    pub fn size(&self) -> usize {
        mlx_sys::array::ffi::array_size(&self.0)
    }

    /// Force evaluation of the lazy graph backing this array.
    pub fn eval(&self) -> Result<()> {
        mlx_sys::transforms::ffi::eval_one(&self.0).map_err(Error::from)
    }

    /// Hidden raw FFI access for advanced users and internal tests.
    #[doc(hidden)]
    pub fn as_inner(&self) -> &mlx_sys::array::ffi::MlxArray {
        &self.0
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        Array(mlx_sys::array::ffi::array_clone(&self.0))
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // CRITICAL: Debug must NOT trigger eval. Read shape/dtype/availability
        // through the cheap getters that the spec guarantees do not eval.
        let evaluated = mlx_sys::array::ffi::array_is_available(&self.0);
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("evaluated", &evaluated)
            .finish()
    }
}
