use cxx::UniquePtr;

use crate::{Dtype, Element, Error, IntoShape, Result, Shape};

/// An MLX array. Cheap to clone (MLX internally refcounts the storage).
pub struct Array(UniquePtr<mlx_sys::array::ffi::MlxArray>);

impl Array {
    /// Low-level FFI escape hatch — wrap an existing cxx UniquePtr<MlxArray> as a safe Array. Use the high-level constructors (Array::try_from / zeros / etc.) for normal code.
    #[doc(hidden)]
    pub fn from_inner(inner: cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>) -> Self {
        Array(inner)
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
                "item() called on non-scalar array (size={}, shape={})",
                self.size(),
                self.shape()
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
    pub fn zeros<S: IntoShape>(shape: S, dtype: Dtype) -> Result<Self> {
        let shape = shape.into_shape();
        let inner = mlx_sys::array::ffi::array_zeros(shape.as_slice(), dtype.as_u8())
            .map_err(Error::from)?;
        Ok(Array(inner))
    }

    /// The shape of the array. `[]` denotes a scalar.
    pub fn shape(&self) -> Shape {
        let raw = mlx_sys::array::ffi::array_shape(&self.0);
        Shape::from(raw)
    }

    /// The size along the given dimension. Supports negative indexing
    /// (`-1` is the last dim).
    ///
    /// Panics if `dim` is out of range.
    pub fn shape_at(&self, dim: i32) -> i32 {
        let s = self.shape();
        let n = s.rank() as i32;
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

    /// Asynchronously evaluate this array. See [`crate::transforms::async_eval_fut`].
    ///
    /// The returned future does not borrow `self` — submission runs
    /// synchronously before the future is constructed, and the future
    /// then owns a refcount-share clone of the array on which it waits.
    pub fn async_eval(&self) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
        crate::transforms::async_eval_fut(&[self])
    }

    /// Low-level FFI escape hatch — borrow the underlying cxx MlxArray. Use the high-level methods for normal code.
    #[doc(hidden)]
    pub fn as_inner(&self) -> &mlx_sys::array::ffi::MlxArray {
        &self.0
    }

    /// Low-level FFI escape hatch — consume an Array and return the inner cxx UniquePtr<MlxArray>.
    #[doc(hidden)]
    pub fn into_inner(self) -> cxx::UniquePtr<mlx_sys::array::ffi::MlxArray> {
        self.0
    }

    /// Element-wise natural exponential. See [`crate::ops::exp`].
    pub fn exp(&self) -> Result<Array> {
        crate::ops::exp(self)
    }

    /// Stream-targeted variant of [`Array::exp`].
    pub fn exp_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::exp_on(self, target)
    }

    /// Element-wise natural logarithm. See [`crate::ops::log`].
    pub fn log(&self) -> Result<Array> {
        crate::ops::log(self)
    }

    /// Stream-targeted variant of [`Array::log`].
    pub fn log_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::log_on(self, target)
    }

    /// Element-wise square root. See [`crate::ops::sqrt`].
    pub fn sqrt(&self) -> Result<Array> {
        crate::ops::sqrt(self)
    }

    /// Stream-targeted variant of [`Array::sqrt`].
    pub fn sqrt_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::sqrt_on(self, target)
    }

    /// Element-wise hyperbolic tangent. See [`crate::ops::tanh`].
    pub fn tanh(&self) -> Result<Array> {
        crate::ops::tanh(self)
    }

    /// Stream-targeted variant of [`Array::tanh`].
    pub fn tanh_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::tanh_on(self, target)
    }

    /// Element-wise sigmoid. See [`crate::ops::sigmoid`].
    pub fn sigmoid(&self) -> Result<Array> {
        crate::ops::sigmoid(self)
    }

    /// Stream-targeted variant of [`Array::sigmoid`].
    pub fn sigmoid_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::sigmoid_on(self, target)
    }

    /// Element-wise x^2. See [`crate::ops::square`].
    pub fn square(&self) -> Result<Array> {
        crate::ops::square(self)
    }

    /// Stream-targeted variant of [`Array::square`].
    pub fn square_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::square_on(self, target)
    }

    /// Element-wise 1/sqrt(x). See [`crate::ops::rsqrt`].
    pub fn rsqrt(&self) -> Result<Array> {
        crate::ops::rsqrt(self)
    }

    /// Stream-targeted variant of [`Array::rsqrt`].
    pub fn rsqrt_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::rsqrt_on(self, target)
    }

    /// Element-wise error function. See [`crate::ops::erf`].
    pub fn erf(&self) -> Result<Array> {
        crate::ops::erf(self)
    }

    /// Stream-targeted variant of [`Array::erf`].
    pub fn erf_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::erf_on(self, target)
    }

    /// Element-wise 1/x. See [`crate::ops::reciprocal`].
    pub fn reciprocal(&self) -> Result<Array> {
        crate::ops::reciprocal(self)
    }

    /// Stream-targeted variant of [`Array::reciprocal`].
    pub fn reciprocal_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::unary::reciprocal_on(self, target)
    }

    /// Sum over the specified axes. See [`crate::ops::sum`].
    pub fn sum<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::sum(self, axes, keepdim)
    }

    /// Stream-targeted variant of [`Array::sum`].
    pub fn sum_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        keepdim: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::reduction::sum_on(self, axes, keepdim, target)
    }

    /// Mean over the specified axes. See [`crate::ops::mean`].
    pub fn mean<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::mean(self, axes, keepdim)
    }

    /// Stream-targeted variant of [`Array::mean`].
    pub fn mean_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        keepdim: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::reduction::mean_on(self, axes, keepdim, target)
    }

    /// Maximum over the specified axes. See [`crate::ops::max`].
    pub fn max<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::max(self, axes, keepdim)
    }

    /// Stream-targeted variant of [`Array::max`].
    pub fn max_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        keepdim: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::reduction::max_on(self, axes, keepdim, target)
    }

    /// Minimum over the specified axes. See [`crate::ops::min`].
    pub fn min<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::min(self, axes, keepdim)
    }

    /// Stream-targeted variant of [`Array::min`].
    pub fn min_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        keepdim: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::reduction::min_on(self, axes, keepdim, target)
    }

    /// Indices of the maximum values along the specified axis. See [`crate::ops::argmax`].
    pub fn argmax<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::argmax(self, axes, keepdim)
    }

    /// Stream-targeted variant of [`Array::argmax`].
    pub fn argmax_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        keepdim: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::reduction::argmax_on(self, axes, keepdim, target)
    }

    /// Reshape this array. See [`crate::ops::reshape`].
    pub fn reshape<S: IntoShape>(&self, shape: S) -> Result<Array> {
        crate::ops::reshape(self, shape)
    }

    /// Stream-targeted variant of [`Array::reshape`].
    pub fn reshape_on<S: IntoShape>(
        &self,
        shape: S,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::shape::reshape_on(self, shape, target)
    }

    /// Reverse all axes. See [`crate::ops::transpose`].
    pub fn transpose(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Stream-targeted variant of [`Array::transpose`].
    pub fn transpose_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::shape::transpose_on(self, target)
    }

    /// Shorthand for [`Array::transpose`]. Standard convention in matrix code.
    pub fn t(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Stream-targeted variant of [`Array::t`].
    pub fn t_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::shape::transpose_on(self, target)
    }

    /// Permute axes per the given permutation. See [`crate::ops::transpose_axes`].
    pub fn transpose_axes<S: IntoShape>(&self, axes: S) -> Result<Array> {
        crate::ops::transpose_axes(self, axes)
    }

    /// Stream-targeted variant of [`Array::transpose_axes`].
    pub fn transpose_axes_on<S: IntoShape>(
        &self,
        axes: S,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::shape::transpose_axes_on(self, axes, target)
    }

    /// Broadcast to the given shape. See [`crate::ops::broadcast_to`].
    pub fn broadcast_to<S: IntoShape>(&self, shape: S) -> Result<Array> {
        crate::ops::broadcast_to(self, shape)
    }

    /// Stream-targeted variant of [`Array::broadcast_to`].
    pub fn broadcast_to_on<S: IntoShape>(
        &self,
        shape: S,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::shape::broadcast_to_on(self, shape, target)
    }

    /// Matrix multiplication. See [`crate::ops::matmul()`] for shape rules.
    pub fn matmul(&self, rhs: &Array) -> Result<Array> {
        crate::ops::matmul(self, rhs)
    }

    /// Stream-targeted variant of [`Array::matmul`].
    pub fn matmul_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::matmul::matmul_on(self, rhs, target)
    }

    /// Use `self` as the condition mask, selecting from `x` where true and `y` where false.
    /// See [`crate::ops::where_`].
    pub fn where_(&self, x: &Array, y: &Array) -> Result<Array> {
        crate::ops::where_(self, x, y)
    }

    /// Stream-targeted variant of [`Array::where_`].
    pub fn where_on(
        &self,
        x: &Array,
        y: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::where_on(self, x, y, target)
    }

    /// Element-wise addition. Returns an error on shape/dtype mismatch.
    /// For an infallible panic-on-err variant, use the `+` operator.
    pub fn try_add(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::add(self, rhs)
    }

    /// Stream-targeted variant of [`Array::try_add`].
    pub fn try_add_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::add_on(self, rhs, target)
    }

    /// Element-wise subtraction. Returns an error on shape/dtype mismatch.
    /// For an infallible panic-on-err variant, use the `-` operator.
    pub fn try_sub(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::subtract(self, rhs)
    }

    /// Stream-targeted variant of [`Array::try_sub`].
    pub fn try_sub_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::subtract_on(self, rhs, target)
    }

    /// Element-wise multiplication. Returns an error on shape/dtype mismatch.
    /// For an infallible panic-on-err variant, use the `*` operator.
    pub fn try_mul(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::multiply(self, rhs)
    }

    /// Stream-targeted variant of [`Array::try_mul`].
    pub fn try_mul_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::multiply_on(self, rhs, target)
    }

    /// Element-wise division. Returns an error on shape/dtype mismatch.
    /// For an infallible panic-on-err variant, use the `/` operator.
    pub fn try_div(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::divide(self, rhs)
    }

    /// Stream-targeted variant of [`Array::try_div`].
    pub fn try_div_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::divide_on(self, rhs, target)
    }

    /// Element-wise negation. Returns an error if the operation fails.
    /// For an infallible panic-on-err variant, use the unary `-` operator.
    pub fn try_neg(&self) -> Result<Array> {
        crate::ops::binary::negative(self)
    }

    /// Stream-targeted variant of [`Array::try_neg`].
    pub fn try_neg_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::binary::negative_on(self, target)
    }

    /// Take values along `axis`. See [`crate::ops::take`].
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take(self, indices, axis)
    }

    /// Stream-targeted variant of [`Array::take`].
    pub fn take_on(
        &self,
        indices: &Array,
        axis: i32,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::take_on(self, indices, axis, target)
    }

    /// Per-axis gather (PyTorch `torch.gather`). See [`crate::ops::take_along_axis`].
    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take_along_axis(self, indices, axis)
    }

    /// Stream-targeted variant of [`Array::take_along_axis`].
    pub fn take_along_axis_on(
        &self,
        indices: &Array,
        axis: i32,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::take_along_axis_on(self, indices, axis, target)
    }

    /// Slice with stride 1. See [`crate::ops::slice`].
    pub fn slice<S1: IntoShape, S2: IntoShape>(&self, start: S1, stop: S2) -> Result<Array> {
        crate::ops::slice(self, start, stop)
    }

    /// Stream-targeted variant of [`Array::slice`].
    pub fn slice_on<S1: IntoShape, S2: IntoShape>(
        &self,
        start: S1,
        stop: S2,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::slice_on(self, start, stop, target)
    }

    /// Slice with explicit strides. See [`crate::ops::slice_strided`].
    pub fn slice_strided<S1: IntoShape, S2: IntoShape, S3: IntoShape>(
        &self,
        start: S1,
        stop: S2,
        strides: S3,
    ) -> Result<Array> {
        crate::ops::slice_strided(self, start, stop, strides)
    }

    /// Stream-targeted variant of [`Array::slice_strided`].
    pub fn slice_strided_on<S1: IntoShape, S2: IntoShape, S3: IntoShape>(
        &self,
        start: S1,
        stop: S2,
        strides: S3,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::slice_strided_on(self, start, stop, strides, target)
    }

    /// N-dimensional gather. See [`crate::ops::gather`].
    pub fn gather(&self, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
        crate::ops::gather(self, indices, axes, slice_sizes)
    }

    /// Stream-targeted variant of [`gather`].
    pub fn gather_on(
        &self,
        indices: &[&Array],
        axes: &[i32],
        slice_sizes: &[i32],
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::indexing::gather_on(self, indices, axes, slice_sizes, target)
    }

    // === P5.5 comparison ops ===

    /// Element-wise `self == rhs`. See [`crate::ops::equal`].
    pub fn equal(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::equal(self, rhs)
    }

    /// Stream-targeted variant of [`Array::equal`].
    pub fn equal_on(&self, rhs: &Array, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::binary::equal_on(self, rhs, target)
    }

    /// Element-wise `self != rhs`. See [`crate::ops::not_equal`].
    pub fn not_equal(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::not_equal(self, rhs)
    }

    /// Stream-targeted variant of [`Array::not_equal`].
    pub fn not_equal_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::not_equal_on(self, rhs, target)
    }

    /// Element-wise `self < rhs`. See [`crate::ops::less`].
    pub fn less(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::less(self, rhs)
    }

    /// Stream-targeted variant of [`Array::less`].
    pub fn less_on(&self, rhs: &Array, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::binary::less_on(self, rhs, target)
    }

    /// Element-wise `self <= rhs`. See [`crate::ops::less_equal`].
    pub fn less_equal(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::less_equal(self, rhs)
    }

    /// Stream-targeted variant of [`Array::less_equal`].
    pub fn less_equal_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::less_equal_on(self, rhs, target)
    }

    /// Element-wise `self > rhs`. See [`crate::ops::greater`].
    pub fn greater(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::greater(self, rhs)
    }

    /// Stream-targeted variant of [`Array::greater`].
    pub fn greater_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::greater_on(self, rhs, target)
    }

    /// Element-wise `self >= rhs`. See [`crate::ops::greater_equal`].
    pub fn greater_equal(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::greater_equal(self, rhs)
    }

    /// Stream-targeted variant of [`Array::greater_equal`].
    pub fn greater_equal_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::greater_equal_on(self, rhs, target)
    }

    // === P5.5 element-wise max/min ===

    /// Element-wise `max(self, rhs)`. See [`crate::ops::maximum`].
    pub fn maximum(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::maximum(self, rhs)
    }

    /// Stream-targeted variant of [`Array::maximum`].
    pub fn maximum_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::maximum_on(self, rhs, target)
    }

    /// Element-wise `min(self, rhs)`. See [`crate::ops::minimum`].
    pub fn minimum(&self, rhs: &Array) -> Result<Array> {
        crate::ops::binary::minimum(self, rhs)
    }

    /// Stream-targeted variant of [`Array::minimum`].
    pub fn minimum_on(
        &self,
        rhs: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::minimum_on(self, rhs, target)
    }

    // === P5.5 clip ===

    /// Clamp to `[a_min, a_max]`. Pass `None` for either bound to leave it
    /// unbounded. See [`crate::ops::clip`].
    pub fn clip(&self, a_min: Option<&Array>, a_max: Option<&Array>) -> Result<Array> {
        crate::ops::binary::clip(self, a_min, a_max)
    }

    /// Stream-targeted variant of [`Array::clip`].
    pub fn clip_on(
        &self,
        a_min: Option<&Array>,
        a_max: Option<&Array>,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::binary::clip_on(self, a_min, a_max, target)
    }

    // === P5.5 softmax ===

    /// Softmax along the given axes. See [`crate::ops::softmax`].
    pub fn softmax<A: crate::ops::IntoAxes>(&self, axes: A, precise: bool) -> Result<Array> {
        crate::ops::unary::softmax(self, axes, precise)
    }

    /// Stream-targeted variant of [`Array::softmax`].
    pub fn softmax_on<A: crate::ops::IntoAxes>(
        &self,
        axes: A,
        precise: bool,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::unary::softmax_on(self, axes, precise, target)
    }

    // === P5.5 sort family ===

    /// Sort along `axis` (ascending). See [`crate::ops::sort::sort`].
    pub fn sort(&self, axis: i32) -> Result<Array> {
        crate::ops::sort::sort(self, axis)
    }

    /// Stream-targeted variant of [`Array::sort`].
    pub fn sort_on(&self, axis: i32, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::sort::sort_on(self, axis, target)
    }

    /// Indices that would sort `self` along `axis`. See [`crate::ops::argsort`].
    pub fn argsort(&self, axis: i32) -> Result<Array> {
        crate::ops::sort::argsort(self, axis)
    }

    /// Stream-targeted variant of [`Array::argsort`].
    pub fn argsort_on(&self, axis: i32, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::sort::argsort_on(self, axis, target)
    }

    /// Partial sort placing `kth`-smallest at position `kth`. See [`crate::ops::partition`].
    pub fn partition(&self, kth: i32, axis: i32) -> Result<Array> {
        crate::ops::sort::partition(self, kth, axis)
    }

    /// Stream-targeted variant of [`Array::partition`].
    pub fn partition_on(
        &self,
        kth: i32,
        axis: i32,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::sort::partition_on(self, kth, axis, target)
    }

    /// Indices form of [`Array::partition`]. See [`crate::ops::argpartition`].
    pub fn argpartition(&self, kth: i32, axis: i32) -> Result<Array> {
        crate::ops::sort::argpartition(self, kth, axis)
    }

    /// Stream-targeted variant of [`Array::argpartition`].
    pub fn argpartition_on(
        &self,
        kth: i32,
        axis: i32,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::sort::argpartition_on(self, kth, axis, target)
    }

    /// Top-k values along `axis` (values only — not sorted; see [`crate::ops::topk`]).
    pub fn topk(&self, k: i32, axis: i32) -> Result<Array> {
        crate::ops::sort::topk(self, k, axis)
    }

    /// Stream-targeted variant of [`Array::topk`].
    pub fn topk_on(
        &self,
        k: i32,
        axis: i32,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::sort::topk_on(self, k, axis, target)
    }

    // === P5.5 astype (dtype conversion) ===

    /// Convert this array to a new array with the given [`Dtype`]. See
    /// [`crate::ops::cast::astype`] for the free-fn form.
    pub fn astype(&self, dtype: Dtype) -> Result<Array> {
        crate::ops::cast::astype(self, dtype)
    }

    /// Stream-targeted variant of [`Array::astype`].
    pub fn astype_on(
        &self,
        dtype: Dtype,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::cast::astype_on(self, dtype, target)
    }

    // === P5.5 constructors: `_like` family + tril/triu ===

    /// Array with the same shape and dtype as `self`, filled with ones. See
    /// [`crate::ops::constructors::ones_like`].
    pub fn ones_like(&self) -> Result<Array> {
        crate::ops::constructors::ones_like(self)
    }

    /// Stream-targeted variant of [`Array::ones_like`].
    pub fn ones_like_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::constructors::ones_like_on(self, target)
    }

    /// Array with the same shape and dtype as `self`, filled with zeros. See
    /// [`crate::ops::constructors::zeros_like`].
    pub fn zeros_like(&self) -> Result<Array> {
        crate::ops::constructors::zeros_like(self)
    }

    /// Stream-targeted variant of [`Array::zeros_like`].
    pub fn zeros_like_on(&self, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::constructors::zeros_like_on(self, target)
    }

    /// Array with the same shape (and dtype) as `self`, filled with `vals`
    /// (broadcast as needed). See [`crate::ops::constructors::full_like`].
    pub fn full_like(&self, vals: &Array) -> Result<Array> {
        crate::ops::constructors::full_like(self, vals)
    }

    /// Stream-targeted variant of [`Array::full_like`].
    pub fn full_like_on(
        &self,
        vals: &Array,
        target: impl Into<crate::StreamOrDevice>,
    ) -> Result<Array> {
        crate::ops::constructors::full_like_on(self, vals, target)
    }

    /// Lower triangular part of `self` (zero out elements strictly above
    /// the diagonal offset by `k`). See [`crate::ops::constructors::tril`].
    pub fn tril(&self, k: i32) -> Result<Array> {
        crate::ops::constructors::tril(self, k)
    }

    /// Stream-targeted variant of [`Array::tril`].
    pub fn tril_on(&self, k: i32, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::constructors::tril_on(self, k, target)
    }

    /// Upper triangular part of `self` (zero out elements strictly below
    /// the diagonal offset by `k`). See [`crate::ops::constructors::triu`].
    pub fn triu(&self, k: i32) -> Result<Array> {
        crate::ops::constructors::triu(self, k)
    }

    /// Stream-targeted variant of [`Array::triu`].
    pub fn triu_on(&self, k: i32, target: impl Into<crate::StreamOrDevice>) -> Result<Array> {
        crate::ops::constructors::triu_on(self, k, target)
    }
}

/// Construct an Array from a slice of `T` and any [`IntoShape`].
///
/// Returns `Err(Error::Mlx)` if the shape contains a negative dim, or
/// `Err(Error::ShapeMismatch)` if `slice.len()` does not equal the shape's
/// element count.
impl<T: Element, S: IntoShape> TryFrom<(&[T], S)> for Array {
    type Error = Error;
    fn try_from((slice, shape): (&[T], S)) -> Result<Array> {
        let shape = shape.into_shape();
        if let Some(&d) = shape.iter().find(|&&d| d < 0) {
            return Err(Error::Mlx(format!(
                "Array::try_from: negative dimension {d} in shape {shape}"
            )));
        }
        let expected: usize = shape.numel();
        if slice.len() != expected {
            return Err(Error::ShapeMismatch {
                expected: shape,
                actual: Shape::from(slice.len() as i32),
            });
        }
        T::array_from(slice, shape.as_slice())
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

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array<{}>{}", self.dtype(), self.shape())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_slice_and_tuple_shape() {
        let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
        assert_eq!(a.shape().as_slice(), &[2, 2]);
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn try_from_array_shape() {
        let a: Array = (&[1.0_f32, 2.0][..], [2]).try_into().unwrap();
        assert_eq!(a.shape().as_slice(), &[2]);
    }

    #[test]
    fn try_from_size_mismatch_errors() {
        let r: Result<Array> = (&[1.0_f32, 2.0][..], (3,)).try_into();
        assert!(matches!(r, Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn shape_returns_shape_type() {
        let a = Array::zeros((2, 3), Dtype::Float32).unwrap();
        let s: Shape = a.shape();
        assert_eq!(s.rank(), 2);
        assert_eq!(s.numel(), 6);
        assert_eq!(format!("{s}"), "[2, 3]");
    }

    #[test]
    fn clone_is_refcount_share() {
        let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
        let b = a.clone();
        assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn add_operator_works() {
        let a: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
        let b: Array = (&[10.0_f32, 20.0, 30.0][..], (3,)).try_into().unwrap();
        let c = &a + &b;
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn sub_mul_div_operators_work() {
        let a: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
        let b: Array = (&[2.0_f32, 4.0][..], (2,)).try_into().unwrap();
        assert_eq!((&a - &b).to_vec::<f32>().unwrap(), vec![8.0, 16.0]);
        assert_eq!((&a * &b).to_vec::<f32>().unwrap(), vec![20.0, 80.0]);
        assert_eq!((&a / &b).to_vec::<f32>().unwrap(), vec![5.0, 5.0]);
    }

    #[test]
    fn neg_operator_works() {
        let a: Array = (&[1.0_f32, -2.0][..], (2,)).try_into().unwrap();
        assert_eq!((-&a).to_vec::<f32>().unwrap(), vec![-1.0, 2.0]);
    }

    #[test]
    fn into_shape_threads_through_zeros_and_reshape() {
        let a = Array::zeros([2, 3], Dtype::Float32).unwrap();
        let b = a.reshape((3, 2)).unwrap();
        assert_eq!(b.shape().as_slice(), &[3, 2]);
    }

    #[test]
    #[should_panic(expected = "Array + Array failed:")]
    fn add_operator_panics_on_shape_mismatch() {
        let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
        let b: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
        let _ = &a + &b;
    }

    #[test]
    fn array_display_uses_dtype_short_name() {
        let a = Array::zeros((2, 3), Dtype::Float32).unwrap();
        assert_eq!(format!("{a}"), "Array<f32>[2, 3]");
    }

    #[test]
    fn try_add_returns_result() {
        let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
        let b: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
        let c = a.try_add(&b).unwrap();
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![11.0, 22.0]);
    }

    #[test]
    fn try_add_returns_err_on_shape_mismatch() {
        let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
        let b: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
        assert!(a.try_add(&b).is_err());
    }

    #[test]
    fn try_sub_mul_div_neg_work() {
        let a: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
        let b: Array = (&[2.0_f32, 4.0][..], (2,)).try_into().unwrap();
        assert_eq!(
            a.try_sub(&b).unwrap().to_vec::<f32>().unwrap(),
            vec![8.0, 16.0]
        );
        assert_eq!(
            a.try_mul(&b).unwrap().to_vec::<f32>().unwrap(),
            vec![20.0, 80.0]
        );
        assert_eq!(
            a.try_div(&b).unwrap().to_vec::<f32>().unwrap(),
            vec![5.0, 5.0]
        );
        assert_eq!(
            a.try_neg().unwrap().to_vec::<f32>().unwrap(),
            vec![-10.0, -20.0]
        );
    }
}
