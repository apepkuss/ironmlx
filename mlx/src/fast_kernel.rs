//! Safe Rust wrapper for mlx-c's Metal kernel API (`mlx_fast_metal_kernel`).
//!
//! Enables execution of custom Metal shaders through the MLX runtime,
//! required for architectures like GatedDeltaNet that use custom GPU kernels.

use std::ffi::CString;

use mlx_sys as sys;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{Error, Result, check};
use crate::stream::Stream;
use crate::vector::VectorArray;

// ── MetalKernelConfig ────────────────────────────────────────────────────────

/// Configuration for a Metal kernel execution, including output shapes,
/// grid/threadgroup dimensions, and template arguments.
pub struct MetalKernelConfig {
    raw: sys::mlx_fast_metal_kernel_config,
}

impl MetalKernelConfig {
    /// Create a new, empty kernel configuration.
    pub fn new() -> Self {
        MetalKernelConfig {
            raw: unsafe { sys::mlx_fast_metal_kernel_config_new() },
        }
    }

    /// Add an output argument with the given shape and dtype.
    pub fn add_output_arg(&self, shape: &[i32], dtype: Dtype) -> Result<()> {
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.raw,
                shape.as_ptr(),
                shape.len(),
                dtype.to_raw(),
            )
        })
    }

    /// Set the compute grid dimensions (number of threadgroups).
    pub fn set_grid(&self, grid: [i32; 3]) -> Result<()> {
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_set_grid(self.raw, grid[0], grid[1], grid[2])
        })
    }

    /// Set the threadgroup dimensions (threads per threadgroup).
    pub fn set_thread_group(&self, tg: [i32; 3]) -> Result<()> {
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_set_thread_group(self.raw, tg[0], tg[1], tg[2])
        })
    }

    /// Set the initial value for output buffers.
    pub fn set_init_value(&self, value: f32) -> Result<()> {
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(self.raw, value) })
    }

    /// Enable or disable verbose logging for kernel compilation.
    pub fn set_verbose(&self, verbose: bool) -> Result<()> {
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_verbose(self.raw, verbose) })
    }

    /// Add a dtype template argument.
    pub fn add_template_arg_dtype(&self, name: &str, dtype: Dtype) -> Result<()> {
        let c_name = CString::new(name).map_err(|e| Error::Mlx(e.to_string()))?;
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                self.raw,
                c_name.as_ptr(),
                dtype.to_raw(),
            )
        })
    }

    /// Add an integer template argument.
    pub fn add_template_arg_int(&self, name: &str, value: i32) -> Result<()> {
        let c_name = CString::new(name).map_err(|e| Error::Mlx(e.to_string()))?;
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_int(self.raw, c_name.as_ptr(), value)
        })
    }

    /// Add a boolean template argument.
    pub fn add_template_arg_bool(&self, name: &str, value: bool) -> Result<()> {
        let c_name = CString::new(name).map_err(|e| Error::Mlx(e.to_string()))?;
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                self.raw,
                c_name.as_ptr(),
                value,
            )
        })
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_fast_metal_kernel_config {
        self.raw
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.raw) };
    }
}

impl Default for MetalKernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── VectorString (internal helper) ───────────────────────────────────────────

/// Thin RAII wrapper around `mlx_vector_string`, used internally to pass
/// string slices to the C API.
struct VectorString(sys::mlx_vector_string);

impl VectorString {
    /// Build a vector of strings from a slice of `&str`.
    fn from_strs(strs: &[&str]) -> Result<Self> {
        let c_strs: Vec<CString> = strs
            .iter()
            .map(|s| CString::new(*s).map_err(|e| Error::Mlx(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let mut ptrs: Vec<*const std::os::raw::c_char> =
            c_strs.iter().map(|s| s.as_ptr()).collect();
        Ok(VectorString(unsafe {
            sys::mlx_vector_string_new_data(ptrs.as_mut_ptr(), ptrs.len())
        }))
    }

    fn as_raw(&self) -> sys::mlx_vector_string {
        self.0
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        unsafe { sys::mlx_vector_string_free(self.0) };
    }
}

// ── MetalKernel ──────────────────────────────────────────────────────────────

/// A compiled Metal kernel that can be applied to arrays.
///
/// Wraps `mlx_fast_metal_kernel`. Build one from Metal shader source via
/// [`MetalKernel::new`], then call [`MetalKernel::apply`] to execute it.
pub struct MetalKernel {
    raw: sys::mlx_fast_metal_kernel,
}

// SAFETY: MetalKernel wraps a compiled GPU kernel object that is immutable
// after creation. The MLX runtime handles GPU synchronization internally.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    /// Create a new Metal kernel from shader source.
    ///
    /// # Arguments
    ///
    /// * `name` - Kernel function name in the Metal source.
    /// * `input_names` - Names of the input arrays (must match kernel parameters).
    /// * `output_names` - Names of the output arrays.
    /// * `source` - Metal shader source code.
    /// * `header` - Additional header source prepended to the shader.
    /// * `ensure_row_contiguous` - If `true`, inputs are made row-contiguous before dispatch.
    /// * `atomic_outputs` - If `true`, outputs use atomic operations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        let c_name = CString::new(name).map_err(|e| Error::Mlx(e.to_string()))?;
        let c_source = CString::new(source).map_err(|e| Error::Mlx(e.to_string()))?;
        let c_header = CString::new(header).map_err(|e| Error::Mlx(e.to_string()))?;
        let c_input_names = VectorString::from_strs(input_names)?;
        let c_output_names = VectorString::from_strs(output_names)?;

        let raw = unsafe {
            sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                c_input_names.as_raw(),
                c_output_names.as_raw(),
                c_source.as_ptr(),
                c_header.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };

        if raw.ctx.is_null() {
            return Err(Error::Mlx("failed to create Metal kernel".into()));
        }

        Ok(MetalKernel { raw })
    }

    /// Execute the kernel with the given inputs and configuration.
    ///
    /// Returns the output arrays produced by the kernel.
    pub fn apply(
        &self,
        inputs: &[&Array],
        config: &MetalKernelConfig,
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let input_vec = VectorArray::from_arrays(inputs);

        // Use a raw handle for the out-parameter, then wrap it in VectorArray
        // for RAII cleanup.
        let mut raw_output = unsafe { sys::mlx_vector_array_new() };

        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                &mut raw_output,
                self.raw,
                input_vec.as_raw(),
                config.as_raw(),
                stream.as_raw(),
            )
        })?;

        let output_vec = VectorArray::from_raw(raw_output);
        let n = output_vec.len();
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(output_vec.get(i)?);
        }
        Ok(results)
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}
