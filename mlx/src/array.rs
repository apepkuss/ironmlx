use mlx_sys as sys;

use crate::dtype::Dtype;
use crate::error::{check, Result};
use crate::stream::Stream;

/// An N-dimensional array (lazy, reference-counted internally by MLX).
pub struct Array(sys::mlx_array);

// ── Constructors ──────────────────────────────────────────────────────────────

impl Array {
    /// An uninitialised array handle (internal use).
    pub(crate) fn new_empty() -> Self {
        Array(unsafe { sys::mlx_array_new() })
    }

    /// Scalar `bool`.
    pub fn from_bool(v: bool) -> Self {
        Array(unsafe { sys::mlx_array_new_bool(v) })
    }

    /// Scalar `i32`.
    pub fn from_int(v: i32) -> Self {
        Array(unsafe { sys::mlx_array_new_int(v) })
    }

    /// Scalar `f32`.
    pub fn from_float(v: f32) -> Self {
        Array(unsafe { sys::mlx_array_new_float(v) })
    }

    /// 1-D array from a slice of `f32`.
    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i32];
        Array(unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                sys::mlx_dtype_MLX_FLOAT32,
            )
        })
    }

    /// 1-D array from a slice of `i32`.
    pub fn from_slice_i32(data: &[i32]) -> Self {
        let shape = [data.len() as i32];
        Array(unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                sys::mlx_dtype_MLX_INT32,
            )
        })
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    pub fn dtype(&self) -> Dtype {
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.0) })
    }

    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.0) as usize }
    }

    pub fn size(&self) -> usize {
        unsafe { sys::mlx_array_size(self.0) }
    }

    pub fn nbytes(&self) -> usize {
        unsafe { sys::mlx_array_nbytes(self.0) }
    }

    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        let ptr = unsafe { sys::mlx_array_shape(self.0) };
        if ptr.is_null() || ndim == 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, ndim) }.to_vec()
    }

    // ── Evaluation ───────────────────────────────────────────────────────────

    /// Force evaluation of any deferred computation.
    pub fn eval(&self) -> Result<()> {
        check(unsafe { sys::mlx_array_eval(self.0) })
    }

    // ── Data access (requires prior eval) ────────────────────────────────────

    pub fn item_f32(&self) -> Result<f32> {
        self.eval()?;
        let mut out: f32 = 0.0;
        check(unsafe { sys::mlx_array_item_float32(&mut out, self.0) })?;
        Ok(out)
    }

    pub fn item_i32(&self) -> Result<i32> {
        self.eval()?;
        let mut out: i32 = 0;
        check(unsafe { sys::mlx_array_item_int32(&mut out, self.0) })?;
        Ok(out)
    }

    /// Copy evaluated `f32` data into a `Vec`.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.eval()?;
        let n = self.size();
        let ptr = unsafe { sys::mlx_array_data_float32(self.0) };
        if ptr.is_null() {
            return Err(crate::error::Error::Mlx("null data pointer".into()));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_array {
        self.0
    }

    /// Internal: wrap a raw `mlx_array` returned by a C op.
    pub(crate) fn from_raw(raw: sys::mlx_array) -> Self {
        Array(raw)
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0) };
    }
}

// ── Formatting ───────────────────────────────────────────────────────────────

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Array(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype()
        )
    }
}
