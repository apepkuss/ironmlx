use mlx_sys as sys;

use crate::array::Array;
use crate::error::{check, Result};
use crate::stream::Stream;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Run a binary op of the form `int mlx_xxx(mlx_array* res, a, b, stream)`.
macro_rules! binary_op {
    ($fn:ident, $a:expr, $b:expr, $stream:expr) => {{
        let mut res = Array::new_empty();
        check(unsafe {
            sys::$fn(
                res.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                $stream.as_raw(),
            )
        })?;
        Ok(res)
    }};
}

/// Run a unary op of the form `int mlx_xxx(mlx_array* res, a, stream)`.
macro_rules! unary_op {
    ($fn:ident, $a:expr, $stream:expr) => {{
        let mut res = Array::new_empty();
        check(unsafe {
            sys::$fn(
                res.as_raw_mut(),
                $a.as_raw(),
                $stream.as_raw(),
            )
        })?;
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
        check(unsafe {
            sys::mlx_sum(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw())
        })?;
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
        check(unsafe {
            sys::mlx_mean(res.as_raw_mut(), a.as_raw(), keep_dims, stream.as_raw())
        })?;
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
