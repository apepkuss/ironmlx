//! Wrapper for MLX custom Metal kernel API (`mlx::core::fast::metal_kernel`).
//!
//! Allows registering and executing custom Metal shaders that integrate
//! seamlessly with MLX's lazy evaluation and compute graph.

use mlx_sys as sys;
use std::ffi::CString;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{Result, check};
use crate::stream::Stream;
use crate::vector::VectorArray;

/// A compiled custom Metal kernel.
pub struct MetalKernel(sys::mlx_fast_metal_kernel);

impl MetalKernel {
    /// Create a new Metal kernel from shader source.
    ///
    /// - `name`: kernel function name in the Metal source
    /// - `input_names`: names of input arguments (must match shader params)
    /// - `output_names`: names of output arguments
    /// - `source`: Metal shader source code
    /// - `header`: optional header code prepended to source
    /// - `ensure_row_contiguous`: ensure inputs are row-contiguous before kernel
    /// - `atomic_outputs`: whether outputs use atomic operations
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Self {
        let c_name = CString::new(name).unwrap();
        let c_source = CString::new(source).unwrap();
        let c_header = CString::new(header).unwrap();

        let input_vs = unsafe {
            let vs = sys::mlx_vector_string_new();
            for n in input_names {
                let cn = CString::new(*n).unwrap();
                sys::mlx_vector_string_append_value(vs, cn.as_ptr());
            }
            vs
        };

        let output_vs = unsafe {
            let vs = sys::mlx_vector_string_new();
            for n in output_names {
                let cn = CString::new(*n).unwrap();
                sys::mlx_vector_string_append_value(vs, cn.as_ptr());
            }
            vs
        };

        let kernel = unsafe {
            sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                input_vs,
                output_vs,
                c_source.as_ptr(),
                c_header.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };

        unsafe {
            sys::mlx_vector_string_free(input_vs);
            sys::mlx_vector_string_free(output_vs);
        }

        MetalKernel(kernel)
    }

    /// Execute the kernel with given inputs and configuration.
    pub fn apply(
        &self,
        inputs: &[&Array],
        config: &MetalKernelConfig,
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let input_va = VectorArray::from_arrays(inputs);
        let mut output_va = VectorArray::new();

        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                output_va.as_raw_mut(),
                self.0,
                input_va.as_raw(),
                config.0,
                stream.as_raw(),
            )
        })?;

        Ok(output_va.to_vec())
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) };
    }
}

/// Configuration for a Metal kernel execution.
pub struct MetalKernelConfig(sys::mlx_fast_metal_kernel_config);

impl Default for MetalKernelConfig {
    fn default() -> Self {
        MetalKernelConfig(unsafe { sys::mlx_fast_metal_kernel_config_new() })
    }
}

impl MetalKernelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an output argument with given shape and dtype.
    pub fn add_output(&self, shape: &[i32], dtype: Dtype) -> &Self {
        unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.0,
                shape.as_ptr(),
                shape.len(),
                dtype.to_raw(),
            );
        }
        self
    }

    /// Set the compute grid dimensions.
    pub fn set_grid(&self, x: i32, y: i32, z: i32) -> &Self {
        unsafe { sys::mlx_fast_metal_kernel_config_set_grid(self.0, x, y, z) };
        self
    }

    /// Set the thread group dimensions.
    pub fn set_thread_group(&self, x: i32, y: i32, z: i32) -> &Self {
        unsafe { sys::mlx_fast_metal_kernel_config_set_thread_group(self.0, x, y, z) };
        self
    }

    /// Set initial value for output buffers.
    pub fn set_init_value(&self, value: f32) -> &Self {
        unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(self.0, value) };
        self
    }

    /// Enable verbose mode (print kernel source on compilation).
    pub fn set_verbose(&self, verbose: bool) -> &Self {
        unsafe { sys::mlx_fast_metal_kernel_config_set_verbose(self.0, verbose) };
        self
    }

    /// Add a dtype template argument.
    pub fn add_template_dtype(&self, name: &str, dtype: Dtype) -> &Self {
        let cn = CString::new(name).unwrap();
        unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                self.0,
                cn.as_ptr(),
                dtype.to_raw(),
            );
        }
        self
    }

    /// Add an integer template argument.
    pub fn add_template_int(&self, name: &str, value: i32) -> &Self {
        let cn = CString::new(name).unwrap();
        unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_int(self.0, cn.as_ptr(), value);
        }
        self
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}
