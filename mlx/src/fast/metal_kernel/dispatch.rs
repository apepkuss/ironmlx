//! Typestate-protected dispatch builder. Compile-time enforces all 5
//! mandatory fields (inputs, output_shapes, output_dtypes, grid,
//! threadgroup) are set before `.dispatch()` is callable.
//!
//! Marker layout: `DispatchBuilder<I, OS, OD, G, TG>` where each parameter is
//! either `Unset` or `Set`. Setters move the relevant marker from `Unset` to
//! `Set`. The terminal `dispatch()` is only callable when all five are `Set`.

use std::marker::PhantomData;
use std::sync::Arc;

use mlx_sys::fast::ffi::TemplateArgC;

use crate::{Array, ArrayVec, Dtype, Error, Result, Shape, StreamOrDevice};

use super::MetalKernelInner;

/// Marker: builder field has not been set.
pub struct Unset;

/// Marker: builder field has been set.
pub struct Set;

/// cxx-safe template argument. Maps to MLX's
/// `std::variant<int, bool, Dtype>`.
#[derive(Debug, Clone)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}

impl TemplateArg {
    fn to_c(&self, name: &str) -> TemplateArgC {
        match self {
            TemplateArg::Int(v) => TemplateArgC {
                name: name.to_string(),
                kind: 0,
                int_val: *v,
                bool_val: false,
                dtype_val: 0,
            },
            TemplateArg::Bool(v) => TemplateArgC {
                name: name.to_string(),
                kind: 1,
                int_val: 0,
                bool_val: *v,
                dtype_val: 0,
            },
            TemplateArg::Dtype(d) => TemplateArgC {
                name: name.to_string(),
                kind: 2,
                int_val: 0,
                bool_val: false,
                dtype_val: d.as_u8(),
            },
        }
    }
}

/// Typestate-protected dispatch builder.
pub struct DispatchBuilder<I, OS, OD, G, TG> {
    kernel: Arc<MetalKernelInner>,

    inputs: Option<cxx::UniquePtr<mlx_sys::compile::ffi::ArrayVec>>,
    output_shapes: Option<Vec<Shape>>,
    output_dtypes: Option<Vec<Dtype>>,
    grid: Option<(i32, i32, i32)>,
    threadgroup: Option<(i32, i32, i32)>,

    template_args: Vec<(String, TemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
    target: StreamOrDevice,

    _markers: PhantomData<(I, OS, OD, G, TG)>,
}

impl DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
    pub(crate) fn new(kernel: Arc<MetalKernelInner>) -> Self {
        Self {
            kernel,
            inputs: None,
            output_shapes: None,
            output_dtypes: None,
            grid: None,
            threadgroup: None,
            template_args: Vec::new(),
            init_value: None,
            verbose: false,
            target: StreamOrDevice::Default,
            _markers: PhantomData,
        }
    }
}

// === 5 mandatory setters (each transitions one marker Unset -> Set) ===

impl<OS, OD, G, TG> DispatchBuilder<Unset, OS, OD, G, TG> {
    /// Set the input arrays. Required.
    pub fn inputs(self, arrays: &[&Array]) -> DispatchBuilder<Set, OS, OD, G, TG> {
        let mut vec = mlx_sys::compile::ffi::array_vec_new();
        for a in arrays {
            mlx_sys::compile::ffi::array_vec_push(vec.pin_mut(), a.as_inner());
        }
        DispatchBuilder {
            kernel: self.kernel,
            inputs: Some(vec),
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: self.grid,
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OD, G, TG> DispatchBuilder<I, Unset, OD, G, TG> {
    /// Set the output shapes. Required. Length must match the kernel's
    /// declared output count (verified at runtime in `dispatch()`).
    pub fn output_shapes(self, shapes: &[Shape]) -> DispatchBuilder<I, Set, OD, G, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: Some(shapes.to_vec()),
            output_dtypes: self.output_dtypes,
            grid: self.grid,
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OS, G, TG> DispatchBuilder<I, OS, Unset, G, TG> {
    /// Set the output dtypes. Required. Length must match the kernel's
    /// declared output count (verified at runtime in `dispatch()`).
    pub fn output_dtypes(self, dtypes: &[Dtype]) -> DispatchBuilder<I, OS, Set, G, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: Some(dtypes.to_vec()),
            grid: self.grid,
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OS, OD, TG> DispatchBuilder<I, OS, OD, Unset, TG> {
    /// Set GPU dispatch grid (x, y, z). Required.
    pub fn grid(self, gx: i32, gy: i32, gz: i32) -> DispatchBuilder<I, OS, OD, Set, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: Some((gx, gy, gz)),
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OS, OD, G> DispatchBuilder<I, OS, OD, G, Unset> {
    /// Set GPU threadgroup size (x, y, z). Required.
    pub fn threadgroup(self, tx: i32, ty: i32, tz: i32) -> DispatchBuilder<I, OS, OD, G, Set> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: self.grid,
            threadgroup: Some((tx, ty, tz)),
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

// === optional setters (don't change markers) ===

impl<I, OS, OD, G, TG> DispatchBuilder<I, OS, OD, G, TG> {
    /// Add an `int` template argument.
    pub fn template_int(mut self, name: impl Into<String>, v: i32) -> Self {
        self.template_args.push((name.into(), TemplateArg::Int(v)));
        self
    }

    /// Add a `bool` template argument.
    pub fn template_bool(mut self, name: impl Into<String>, v: bool) -> Self {
        self.template_args.push((name.into(), TemplateArg::Bool(v)));
        self
    }

    /// Add a `Dtype` template argument.
    pub fn template_dtype(mut self, name: impl Into<String>, v: Dtype) -> Self {
        self.template_args
            .push((name.into(), TemplateArg::Dtype(v)));
        self
    }

    /// Set initial value for atomic outputs (only meaningful if kernel was
    /// built with `atomic_outputs(true)`).
    pub fn init_value(mut self, v: f32) -> Self {
        self.init_value = Some(v);
        self
    }

    /// Enable verbose Metal compile logging.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Set target stream/device.
    pub fn stream(mut self, target: impl Into<StreamOrDevice>) -> Self {
        self.target = target.into();
        self
    }
}

// === dispatch() — only callable with all markers Set ===

impl DispatchBuilder<Set, Set, Set, Set, Set> {
    /// Execute the kernel and return outputs as `ArrayVec`. Take individual
    /// outputs via `arr_vec.take_at(i)` in the order declared in
    /// `MetalKernelBuilder::outputs(...)`.
    pub fn dispatch(self) -> Result<ArrayVec> {
        let input_vec = self.inputs.expect("typestate: inputs Set");
        let output_shapes = self.output_shapes.expect("typestate: output_shapes Set");
        let output_dtypes = self.output_dtypes.expect("typestate: output_dtypes Set");
        let grid = self.grid.expect("typestate: grid Set");
        let threadgroup = self.threadgroup.expect("typestate: threadgroup Set");

        // Sanity: counts match
        if output_shapes.len() != self.kernel.output_count {
            return Err(Error::Mlx(format!(
                "MetalKernel dispatch: output_shapes count {} != declared outputs {}",
                output_shapes.len(),
                self.kernel.output_count,
            )));
        }
        if output_dtypes.len() != self.kernel.output_count {
            return Err(Error::Mlx(format!(
                "MetalKernel dispatch: output_dtypes count {} != declared outputs {}",
                output_dtypes.len(),
                self.kernel.output_count,
            )));
        }

        // Build ShapesVec
        let mut shapes_vec = mlx_sys::fast::ffi::shapes_vec_new();
        for s in &output_shapes {
            mlx_sys::fast::ffi::shapes_vec_push(shapes_vec.pin_mut(), s.as_slice());
        }

        // Encode template args
        let template_c: Vec<TemplateArgC> = self
            .template_args
            .iter()
            .map(|(name, val)| val.to_c(name))
            .collect();

        // Encode dtypes
        let dtype_reprs: Vec<u8> = output_dtypes.iter().map(|d| d.as_u8()).collect();

        // Encode stream
        let (has_stream, dev_only, dev_t, idx) = self.target.encode();

        let (init_has, init_v) = match self.init_value {
            Some(v) => (true, v),
            None => (false, 0.0),
        };

        let raw_outputs = mlx_sys::fast::ffi::metal_kernel_dispatch(
            &self.kernel.handle,
            &input_vec,
            &shapes_vec,
            &dtype_reprs,
            grid.0,
            grid.1,
            grid.2,
            threadgroup.0,
            threadgroup.1,
            threadgroup.2,
            &template_c,
            init_has,
            init_v,
            self.verbose,
            has_stream,
            dev_only,
            dev_t,
            idx,
        )
        .map_err(Error::from)?;

        Ok(ArrayVec::from_inner(raw_outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::MetalKernel;

    fn trivial_kernel() -> MetalKernel {
        MetalKernel::builder("trivial_add_one")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
            .build()
            .expect("kernel compiles")
    }

    #[test]
    fn template_arg_int_to_c() {
        let c = TemplateArg::Int(42).to_c("Dk");
        assert_eq!(c.kind, 0);
        assert_eq!(c.int_val, 42);
    }

    #[test]
    fn template_arg_bool_to_c() {
        let c = TemplateArg::Bool(true).to_c("Vec");
        assert_eq!(c.kind, 1);
        assert!(c.bool_val);
    }

    #[test]
    fn template_arg_dtype_to_c() {
        let c = TemplateArg::Dtype(Dtype::Float16).to_c("InT");
        assert_eq!(c.kind, 2);
        assert_eq!(c.dtype_val, Dtype::Float16.as_u8());
    }

    #[test]
    fn typestate_setters_traverse_to_dispatchable() {
        let k = trivial_kernel();
        let x: Array = (&[0.0_f32, 0.0, 0.0, 0.0][..], (4_i32,))
            .try_into()
            .unwrap();
        let mut outputs = k
            .dispatch_builder()
            .inputs(&[&x])
            .output_shapes(&[Shape::from((4_i32,))])
            .output_dtypes(&[Dtype::Float32])
            .grid(4, 1, 1)
            .threadgroup(4, 1, 1)
            .dispatch()
            .expect("dispatch ok");
        let y = outputs.take_at(0).expect("take");
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }
}
