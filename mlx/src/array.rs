use cxx::UniquePtr;

use crate::{Dtype, Element, Error, Result};

/// An MLX array. Cheap to clone (MLX internally refcounts the storage).
pub struct Array(UniquePtr<mlx_sys::array::ffi::MlxArray>);

impl Array {
    /// Low-level FFI escape hatch — wrap an existing cxx UniquePtr<MlxArray> as a safe Array. Use the high-level constructors (Array::from_slice / zeros / etc.) for normal code.
    #[doc(hidden)]
    pub fn from_inner(inner: cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>) -> Self {
        Array(inner)
    }

    /// Construct an array from a slice of `T` and a shape.
    ///
    /// Returns `Err(Error::Mlx(...))` if `shape` contains a negative dimension,
    /// or `Err(Error::ShapeMismatch)` if `slice.len()` does not equal
    /// `shape.iter().product()` (the empty-shape product is 1, denoting a scalar).
    pub fn from_slice<T: Element>(slice: &[T], shape: &[i32]) -> Result<Array> {
        // Reject negative dims early — `d as usize` would wrap to usize::MAX
        // and the subsequent .product() would either overflow-panic in debug
        // or wrap silently in release.
        if let Some(&d) = shape.iter().find(|&&d| d < 0) {
            return Err(Error::Mlx(format!(
                "from_slice: negative dimension {d} in shape {shape:?}"
            )));
        }
        // Empty shape → empty product = 1 → scalar (1 element). The branch
        // for shape.is_empty() is unnecessary because i32::product on an
        // empty iterator already returns 1.
        let expected: usize = shape.iter().map(|&d| d as usize).product();
        if slice.len() != expected {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![slice.len() as i32],
            });
        }
        T::array_from(slice, shape)
    }

    /// Copy all elements out as a `Vec<T>`. Implicitly evaluates if needed.
    ///
    /// Returns `Err(Error::DtypeMismatch)` if the array's dtype does not
    /// match `T::DTYPE`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DtypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype(),
            });
        }
        T::array_to_vec(self)
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
        let inner = mlx_sys::array::ffi::array_zeros(shape, dtype.as_u8()).map_err(Error::from)?;
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
        assert!(
            idx >= 0 && idx < n,
            "shape_at({dim}): out of range for ndim={n}"
        );
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

    /// Asynchronously evaluate this array. See [`crate::transforms::async_eval`].
    ///
    /// The returned future does not borrow `self` — submission runs
    /// synchronously before the future is constructed, and the future
    /// then owns a refcount-share clone of the array on which it waits.
    pub fn async_eval(&self) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
        crate::transforms::async_eval(&[self])
    }

    /// Low-level FFI escape hatch — borrow the underlying cxx MlxArray. Use the high-level methods for normal code.
    #[doc(hidden)]
    pub fn as_inner(&self) -> &mlx_sys::array::ffi::MlxArray {
        &self.0
    }

    /// Element-wise natural exponential. See [`crate::ops::exp`].
    pub fn exp(&self) -> Result<Array> {
        crate::ops::exp(self)
    }

    /// Element-wise natural logarithm. See [`crate::ops::log`].
    pub fn log(&self) -> Result<Array> {
        crate::ops::log(self)
    }

    /// Element-wise square root. See [`crate::ops::sqrt`].
    pub fn sqrt(&self) -> Result<Array> {
        crate::ops::sqrt(self)
    }

    /// Element-wise hyperbolic tangent. See [`crate::ops::tanh`].
    pub fn tanh(&self) -> Result<Array> {
        crate::ops::tanh(self)
    }

    /// Element-wise sigmoid. See [`crate::ops::sigmoid`].
    pub fn sigmoid(&self) -> Result<Array> {
        crate::ops::sigmoid(self)
    }

    /// Element-wise x^2. See [`crate::ops::square`].
    pub fn square(&self) -> Result<Array> {
        crate::ops::square(self)
    }

    /// Element-wise 1/sqrt(x). See [`crate::ops::rsqrt`].
    pub fn rsqrt(&self) -> Result<Array> {
        crate::ops::rsqrt(self)
    }

    /// Element-wise error function. See [`crate::ops::erf`].
    pub fn erf(&self) -> Result<Array> {
        crate::ops::erf(self)
    }

    /// Element-wise 1/x. See [`crate::ops::reciprocal`].
    pub fn reciprocal(&self) -> Result<Array> {
        crate::ops::reciprocal(self)
    }

    /// Sum over the specified axes. See [`crate::ops::sum`].
    pub fn sum<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::sum(self, axes, keepdim)
    }

    /// Mean over the specified axes. See [`crate::ops::mean`].
    pub fn mean<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::mean(self, axes, keepdim)
    }

    /// Maximum over the specified axes. See [`crate::ops::max`].
    pub fn max<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::max(self, axes, keepdim)
    }

    /// Minimum over the specified axes. See [`crate::ops::min`].
    pub fn min<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::min(self, axes, keepdim)
    }

    /// Indices of the maximum values along the specified axis. See [`crate::ops::argmax`].
    pub fn argmax<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::argmax(self, axes, keepdim)
    }

    /// Reshape this array. See [`crate::ops::reshape`].
    pub fn reshape(&self, shape: &[i32]) -> Result<Array> {
        crate::ops::reshape(self, shape)
    }

    /// Reverse all axes. See [`crate::ops::transpose`].
    pub fn transpose(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Shorthand for [`Array::transpose`]. Standard convention in matrix code.
    pub fn t(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Permute axes per the given permutation. See [`crate::ops::transpose_axes`].
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        crate::ops::transpose_axes(self, axes)
    }

    /// Broadcast to the given shape. See [`crate::ops::broadcast_to`].
    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        crate::ops::broadcast_to(self, shape)
    }

    /// Matrix multiplication. See [`crate::ops::matmul()`] for shape rules.
    pub fn matmul(&self, rhs: &Array) -> Result<Array> {
        crate::ops::matmul(self, rhs)
    }

    /// Use `self` as the condition mask, selecting from `x` where true and `y` where false.
    /// See [`crate::ops::where_`].
    pub fn where_(&self, x: &Array, y: &Array) -> Result<Array> {
        crate::ops::where_(self, x, y)
    }

    /// Take values along `axis`. See [`crate::ops::take`].
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take(self, indices, axis)
    }

    /// Per-axis gather (PyTorch `torch.gather`). See [`crate::ops::take_along_axis`].
    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take_along_axis(self, indices, axis)
    }

    /// Slice with stride 1. See [`crate::ops::slice`].
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Result<Array> {
        crate::ops::slice(self, start, stop)
    }

    /// Slice with explicit strides. See [`crate::ops::slice_strided`].
    pub fn slice_strided(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
        crate::ops::slice_strided(self, start, stop, strides)
    }

    /// N-dimensional gather. See [`crate::ops::gather`].
    pub fn gather(&self, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
        crate::ops::gather(self, indices, axes, slice_sizes)
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

// SAFETY: MLX's `mlx::core::array` is internally backed by
// `std::shared_ptr<ArrayDesc>`. The shared_ptr refcount is atomic, so
// transferring ownership across threads is safe (the destructor in the
// receiving thread can decrement the refcount).
//
// We do NOT impl Sync because MLX's "const" methods (set_status,
// attach_event, is_available's lazy→available transition) mutate the
// underlying ArrayDesc without synchronization. Two threads holding
// `&Array` to the same array would race. To share an Array between
// threads, clone it (cheap MLX refcount) or wrap it in
// `Arc<Mutex<Array>>`. See README "Threading" section.
unsafe impl Send for Array {}
