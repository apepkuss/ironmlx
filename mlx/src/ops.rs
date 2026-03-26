use mlx_sys as sys;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{Result, check};
use crate::stream::Stream;
use crate::vector::VectorArray;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Run a binary op of the form `int mlx_xxx(mlx_array* res, a, b, stream)`.
macro_rules! binary_op {
    ($fn:ident, $a:expr, $b:expr, $stream:expr) => {{
        let mut res = Array::new_empty();
        check(unsafe { sys::$fn(res.as_raw_mut(), $a.as_raw(), $b.as_raw(), $stream.as_raw()) })?;
        Ok(res)
    }};
}

/// Run a unary op of the form `int mlx_xxx(mlx_array* res, a, stream)`.
macro_rules! unary_op {
    ($fn:ident, $a:expr, $stream:expr) => {{
        let mut res = Array::new_empty();
        check(unsafe { sys::$fn(res.as_raw_mut(), $a.as_raw(), $stream.as_raw()) })?;
        Ok(res)
    }};
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

pub fn add(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_add, a, b, stream)
}

pub fn subtract(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_subtract, a, b, stream)
}

pub fn multiply(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_multiply, a, b, stream)
}

pub fn divide(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_divide, a, b, stream)
}

pub fn matmul(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_matmul, a, b, stream)
}

// ── Elementwise unary ────────────────────────────────────────────────────────

pub fn abs(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_abs, a, stream)
}

pub fn neg(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_negative, a, stream)
}

pub fn exp(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_exp, a, stream)
}

pub fn log(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_log, a, stream)
}

pub fn sqrt(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_sqrt, a, stream)
}

pub fn square(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_square, a, stream)
}

// ── Reductions ───────────────────────────────────────────────────────────────

/// Sum along `axes`. Pass an empty slice to sum over all axes.
pub fn sum(a: &Array, axes: &[i32], keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    if axes.is_empty() {
        check(unsafe { sys::mlx_sum(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw()) })?;
    } else {
        check(unsafe {
            sys::mlx_sum_axes(
                res.as_raw_mut(),
                a.as_raw(),
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_raw(),
            )
        })?;
    }
    Ok(res)
}

/// Mean along `axes`. Pass an empty slice to average over all axes.
pub fn mean(a: &Array, axes: &[i32], keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    if axes.is_empty() {
        check(unsafe { sys::mlx_mean(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw()) })?;
    } else {
        check(unsafe {
            sys::mlx_mean_axes(
                res.as_raw_mut(),
                a.as_raw(),
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_raw(),
            )
        })?;
    }
    Ok(res)
}

// ── Activations ──────────────────────────────────────────────────────────────

/// Softmax along the given axes (use `&[-1]` for the last axis).
pub fn softmax(a: &Array, axes: &[i32], stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_softmax_axes(
            res.as_raw_mut(),
            a.as_raw(),
            axes.as_ptr(),
            axes.len(),
            false, // precise
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// ReLU: max(a, 0) — implemented as maximum(a, zeros_like(a)).
pub fn relu(a: &Array, stream: &Stream) -> Result<Array> {
    let mut zeros = Array::new_empty();
    check(unsafe { sys::mlx_zeros_like(zeros.as_raw_mut(), a.as_raw(), stream.as_raw()) })?;
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_maximum(
            res.as_raw_mut(),
            a.as_raw(),
            zeros.as_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

// ── Shape manipulation ────────────────────────────────────────────────────────

pub fn reshape(a: &Array, shape: &[i32], stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_reshape(
            res.as_raw_mut(),
            a.as_raw(),
            shape.as_ptr(),
            shape.len(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Reverse all dimension order (equivalent to numpy.transpose with no args).
pub fn transpose(a: &Array, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_transpose(res.as_raw_mut(), a.as_raw(), stream.as_raw()) })?;
    Ok(res)
}

/// Transpose with explicit axis permutation.
pub fn transpose_axes(a: &Array, axes: &[i32], stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_transpose_axes(
            res.as_raw_mut(),
            a.as_raw(),
            axes.as_ptr(),
            axes.len(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

// ── Shape operations ────────────────────────────────────────────────────────

/// Concatenate arrays along `axis`.
pub fn concatenate(arrays: &VectorArray, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_concatenate_axis(res.as_raw_mut(), arrays.as_raw(), axis, stream.as_raw())
    })?;
    Ok(res)
}

/// Add a size-one dimension at `axis`.
pub fn expand_dims(a: &Array, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_expand_dims(res.as_raw_mut(), a.as_raw(), axis, stream.as_raw()) })?;
    Ok(res)
}

/// Add size-one dimensions at multiple `axes`.
pub fn expand_dims_axes(a: &Array, axes: &[i32], stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_expand_dims_axes(
            res.as_raw_mut(),
            a.as_raw(),
            axes.as_ptr(),
            axes.len(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Remove all size-one dimensions.
pub fn squeeze(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_squeeze, a, stream)
}

/// Remove the size-one dimension at `axis`.
pub fn squeeze_axis(a: &Array, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_squeeze_axis(res.as_raw_mut(), a.as_raw(), axis, stream.as_raw()) })?;
    Ok(res)
}

/// Slice an array with `start`, `stop`, and `strides` per dimension.
pub fn slice(
    a: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_slice(
            res.as_raw_mut(),
            a.as_raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Split an array into `num_splits` equal parts along `axis`.
pub fn split(a: &Array, num_splits: i32, axis: i32, stream: &Stream) -> Result<VectorArray> {
    let mut raw = unsafe { sys::mlx_vector_array_new() };
    check(unsafe { sys::mlx_split(&mut raw, a.as_raw(), num_splits, axis, stream.as_raw()) })?;
    Ok(VectorArray::from_raw(raw))
}

/// Repeat array `repeats` times along `axis`.
pub fn repeat_axis(a: &Array, repeats: i32, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_repeat_axis(res.as_raw_mut(), a.as_raw(), repeats, axis, stream.as_raw())
    })?;
    Ok(res)
}

/// Split an array at the given `indices` along `axis`.
pub fn split_at_indices(
    a: &Array,
    indices: &[i32],
    axis: i32,
    stream: &Stream,
) -> Result<VectorArray> {
    let mut raw = unsafe { sys::mlx_vector_array_new() };
    check(unsafe {
        sys::mlx_split_sections(
            &mut raw,
            a.as_raw(),
            indices.as_ptr(),
            indices.len(),
            axis,
            stream.as_raw(),
        )
    })?;
    Ok(VectorArray::from_raw(raw))
}

/// Stack arrays along `axis`.
pub fn stack(arrays: &VectorArray, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_stack_axis(res.as_raw_mut(), arrays.as_raw(), axis, stream.as_raw())
    })?;
    Ok(res)
}

/// Gather elements along `axis` using `indices`.
pub fn take_axis(a: &Array, indices: &Array, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_take_axis(
            res.as_raw_mut(),
            a.as_raw(),
            indices.as_raw(),
            axis,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Gather elements from a flattened array using `indices`.
pub fn take(a: &Array, indices: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_take, a, indices, stream)
}

/// Reinterpret the binary data of `a` as `dtype`.
pub fn view(a: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_view(
            res.as_raw_mut(),
            a.as_raw(),
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Broadcast `a` to the given `shape`.
pub fn broadcast_to(a: &Array, shape: &[i32], stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_broadcast_to(
            res.as_raw_mut(),
            a.as_raw(),
            shape.as_ptr(),
            shape.len(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

// ── Math / activation ───────────────────────────────────────────────────────

pub fn sigmoid(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_sigmoid, a, stream)
}

pub fn tanh(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_tanh, a, stream)
}

pub fn rsqrt(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_rsqrt, a, stream)
}

pub fn power(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_power, a, b, stream)
}

pub fn cos(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_cos, a, stream)
}

pub fn sin(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_sin, a, stream)
}

pub fn maximum(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_maximum, a, b, stream)
}

pub fn minimum(a: &Array, b: &Array, stream: &Stream) -> Result<Array> {
    binary_op!(mlx_minimum, a, b, stream)
}

/// Element-wise selection: choose from `x` where `condition` is true, else `y`.
pub fn where_(condition: &Array, x: &Array, y: &Array, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_where(
            res.as_raw_mut(),
            condition.as_raw(),
            x.as_raw(),
            y.as_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Clip values to `[min, max]`. Either bound may be `None`.
pub fn clip(a: &Array, min: Option<&Array>, max: Option<&Array>, stream: &Stream) -> Result<Array> {
    let null_arr = unsafe { std::mem::zeroed::<sys::mlx_array>() };
    let min_raw = min.map_or(null_arr, |x| x.as_raw());
    let max_raw = max.map_or(null_arr, |x| x.as_raw());
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_clip(
            res.as_raw_mut(),
            a.as_raw(),
            min_raw,
            max_raw,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

pub fn erf(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_erf, a, stream)
}

/// Sort along `axis`.
pub fn sort_axis(a: &Array, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_sort_axis(res.as_raw_mut(), a.as_raw(), axis, stream.as_raw()) })?;
    Ok(res)
}

/// Return indices that sort along `axis`.
pub fn argsort_axis(a: &Array, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_argsort_axis(res.as_raw_mut(), a.as_raw(), axis, stream.as_raw()) })?;
    Ok(res)
}

/// Cumulative sum along `axis`.
pub fn cumsum(
    a: &Array,
    axis: i32,
    reverse: bool,
    inclusive: bool,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_cumsum(
            res.as_raw_mut(),
            a.as_raw(),
            axis,
            reverse,
            inclusive,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

// ── Reductions ──────────────────────────────────────────────────────────────

/// Max along `axes`. Pass an empty slice to reduce over all axes.
pub fn max(a: &Array, axes: &[i32], keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    if axes.is_empty() {
        check(unsafe { sys::mlx_max(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw()) })?;
    } else {
        check(unsafe {
            sys::mlx_max_axes(
                res.as_raw_mut(),
                a.as_raw(),
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_raw(),
            )
        })?;
    }
    Ok(res)
}

/// Min along `axes`. Pass an empty slice to reduce over all axes.
pub fn min(a: &Array, axes: &[i32], keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    if axes.is_empty() {
        check(unsafe { sys::mlx_min(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw()) })?;
    } else {
        check(unsafe {
            sys::mlx_min_axes(
                res.as_raw_mut(),
                a.as_raw(),
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_raw(),
            )
        })?;
    }
    Ok(res)
}

/// Index of the maximum value along `axis`.
pub fn argmax(a: &Array, axis: i32, keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_argmax_axis(
            res.as_raw_mut(),
            a.as_raw(),
            axis,
            keep_dims,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Index of the minimum value along `axis`.
pub fn argmin(a: &Array, axis: i32, keep_dims: bool, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_argmin_axis(
            res.as_raw_mut(),
            a.as_raw(),
            axis,
            keep_dims,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Return the top-k elements along `axis`.
pub fn topk(a: &Array, k: i32, axis: i32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_topk_axis(res.as_raw_mut(), a.as_raw(), k, axis, stream.as_raw()) })?;
    Ok(res)
}

// ── Constructors ────────────────────────────────────────────────────────────

/// Cast `a` to `dtype`.
pub fn astype(a: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_astype(
            res.as_raw_mut(),
            a.as_raw(),
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Create an array of zeros with the given `shape` and `dtype`.
pub fn zeros(shape: &[i32], dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_zeros(
            res.as_raw_mut(),
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Create an array of ones with the given `shape` and `dtype`.
pub fn ones(shape: &[i32], dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_ones(
            res.as_raw_mut(),
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Create an array filled with `value` of the given `shape` and `dtype`.
pub fn full(shape: &[i32], value: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_full(
            res.as_raw_mut(),
            shape.as_ptr(),
            shape.len(),
            value.as_raw(),
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Create an array of zeros with the same shape and dtype as `a`.
pub fn zeros_like(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_zeros_like, a, stream)
}

/// Create an array of ones with the same shape and dtype as `a`.
pub fn ones_like(a: &Array, stream: &Stream) -> Result<Array> {
    unary_op!(mlx_ones_like, a, stream)
}

/// Create a 1-D array with values from `start` to `stop` with `step`.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_arange(
            res.as_raw_mut(),
            start,
            stop,
            step,
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

// ── Quantized ────────────────────────────────────────────────────────────

/// Convert `Option<i32>` to the C `mlx_optional_int` struct.
fn optional_int(v: Option<i32>) -> sys::mlx_optional_int_ {
    match v {
        Some(value) => sys::mlx_optional_int_ {
            value,
            has_value: true,
        },
        None => sys::mlx_optional_int_ {
            value: 0,
            has_value: false,
        },
    }
}

/// Dequantize a packed weight matrix back to full precision.
///
/// - `w`: quantized weight (packed uint32)
/// - `scales`: per-group scale factors
/// - `biases`: per-group biases (optional)
/// - `group_size`: quantization group size (default 64)
/// - `bits`: quantization bit-width (default 4)
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let biases_raw = match biases {
        Some(b) => b.as_raw(),
        None => unsafe { std::mem::zeroed() },
    };
    let mode_c = std::ffi::CString::new(mode).unwrap();
    let dtype = sys::mlx_optional_dtype_ {
        value: 0,
        has_value: false,
    };
    check(unsafe {
        sys::mlx_dequantize(
            res.as_raw_mut(),
            w.as_raw(),
            scales.as_raw(),
            biases_raw,
            optional_int(group_size),
            optional_int(bits),
            mode_c.as_ptr(),
            dtype,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Quantized matrix multiplication.
///
/// Performs `x @ dequantize(w, scales, biases)^T` using the quantized weight
/// representation directly, avoiding full dequantization.
///
/// Supported modes: `"affine"` (2/3/4/6/8 bit), `"mxfp8"`, `"nvfp4"`.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let biases_raw = match biases {
        Some(b) => b.as_raw(),
        None => unsafe { std::mem::zeroed() },
    };
    let mode_c = std::ffi::CString::new(mode).unwrap();
    check(unsafe {
        sys::mlx_quantized_matmul(
            res.as_raw_mut(),
            x.as_raw(),
            w.as_raw(),
            scales.as_raw(),
            biases_raw,
            transpose,
            optional_int(group_size),
            optional_int(bits),
            mode_c.as_ptr(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Create `num` evenly-spaced values from `start` to `stop`.
pub fn linspace(start: f64, stop: f64, num: i32, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_linspace(
            res.as_raw_mut(),
            start,
            stop,
            num,
            dtype.to_raw(),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Write `values` into `a` at positions given by `indices` along `axis`.
///
/// Like NumPy/PyTorch `put_along_axis` / `scatter_`: `a`, `indices`, and
/// `values` must have the same number of dimensions. Broadcasting is supported.
pub fn put_along_axis(
    a: &Array,
    indices: &Array,
    values: &Array,
    axis: i32,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_put_along_axis(
            res.as_raw_mut(),
            a.as_raw(),
            indices.as_raw(),
            values.as_raw(),
            axis as std::os::raw::c_int,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}
