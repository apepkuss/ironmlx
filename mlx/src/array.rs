use cxx::UniquePtr;

use crate::{Dtype, Element, Error, Result};

/// An MLX array. Cheap to clone (MLX internally refcounts the storage).
pub struct Array(UniquePtr<mlx_sys::array::ffi::MlxArray>);

impl Array {
    /// Construct from a raw cxx UniquePtr. Internal use only — the safe API
    /// is `Array::from_slice<T>` / `Array::zeros` / etc.
    pub(crate) fn from_inner(inner: cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>) -> Self {
        Array(inner)
    }

    /// Construct an array from a slice of `T` and a shape.
    ///
    /// Returns `Err(Error::ShapeMismatch)` if `slice.len()` does not equal
    /// `shape.iter().product()` (or 1 for empty/scalar shapes).
    pub fn from_slice<T: Element>(slice: &[T], shape: &[i32]) -> Result<Array> {
        let expected: usize = shape.iter().map(|&d| d as usize).product();
        let expected = if shape.is_empty() { 1 } else { expected };
        if slice.len() != expected {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![slice.len() as i32],
            });
        }
        T::array_from(slice, shape)
    }

    /// Read this array as a single scalar of type `T`.
    ///
    /// Returns `Err` if the array is not a scalar (size != 1) or if its
    /// dtype does not match `T::DTYPE`. Implicitly evaluates the array.
    pub fn item<T: Element>(&self) -> Result<T> {
        if self.size() != 1 {
            return Err(Error::Mlx(format!(
                "item() called on non-scalar array (size={}, shape={:?})",
                self.size(),
                self.shape().as_slice()
            )));
        }
        if self.dtype() != T::DTYPE {
            return Err(Error::DtypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype(),
            });
        }
        T::array_item(self)
    }

    /// Create an array filled with zeros of the given shape and dtype.
    /// The result is lazy — call [`Array::eval`] before reading the data.
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Self> {
        let inner = mlx_sys::array::ffi::array_zeros(shape, dtype.as_u8())
            .map_err(Error::from)?;
        Ok(Array(inner))
    }

    /// The shape of the array. `[]` denotes a scalar.
    ///
    /// Returns a `SmallVec` with 8 inline slots — zero allocation for
    /// the common case of ≤ 8-dimensional tensors.
    pub fn shape(&self) -> smallvec::SmallVec<[i32; 8]> {
        let raw = mlx_sys::array::ffi::array_shape(&self.0);
        smallvec::SmallVec::from_vec(raw)
    }

    /// The size along the given dimension. Supports negative indexing
    /// (`-1` is the last dim).
    ///
    /// Panics if `dim` is out of range.
    pub fn shape_at(&self, dim: i32) -> i32 {
        let s = self.shape();
        let n = s.len() as i32;
        let idx = if dim < 0 { dim + n } else { dim };
        assert!(idx >= 0 && idx < n, "shape_at({dim}): out of range for ndim={n}");
        s[idx as usize]
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
