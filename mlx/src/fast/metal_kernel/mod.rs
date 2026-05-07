//! Custom Metal kernel binding for Apple Silicon. Wraps
//! `mlx::core::fast::metal_kernel`.
//!
//! Two-phase API:
//! 1. **Build** — `MetalKernel::builder(name).inputs(...).outputs(...).source(...).build()?`
//!    creates a callable kernel handle (the actual Metal source compilation
//!    happens lazily at first dispatch). Cheap to clone (`Arc` internally).
//! 2. **Dispatch** — `kernel.dispatch_builder().inputs(...).grid(...).threadgroup(...).dispatch()?`
//!    executes the kernel. Mandatory fields enforced at compile time via
//!    typestate (see `dispatch.rs`).
//!
//! See P3a spec § 3 for design rationale.

use std::sync::Arc;

use crate::{Error, Result};

mod dispatch;

pub use dispatch::{DispatchBuilder, Set, TemplateArg, Unset};

/// Compiled Metal kernel handle. Cheap to clone (Arc-shared inner).
pub struct MetalKernel {
    inner: Arc<MetalKernelInner>,
}

// T5 will read handle and output_count in dispatch(); allow dead_code until then.
#[allow(dead_code)]
pub(crate) struct MetalKernelInner {
    pub(crate) handle: cxx::UniquePtr<mlx_sys::fast::ffi::MetalKernelInner>,
    pub(crate) output_count: usize,
}

// SAFETY: cxx::UniquePtr<MetalKernelInner> wraps a C++ object holding
// `std::function<...>`; immutable after construction. The MLX
// CustomKernelFunction is intended to be called from any thread (the
// kernel itself is stateless; per-dispatch state is in arguments).
// Mark Send+Sync to allow Arc-share across threads.
unsafe impl Send for MetalKernelInner {}
unsafe impl Sync for MetalKernelInner {}

impl Clone for MetalKernel {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MetalKernel {
    /// Start building a kernel with the given name.
    pub fn builder(name: impl Into<String>) -> MetalKernelBuilder {
        MetalKernelBuilder {
            name: name.into(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            source: String::new(),
            header: String::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
    }

    /// Begin a dispatch invocation. Returns a typestate-protected builder
    /// where 5 mandatory fields (inputs / output_shapes / output_dtypes /
    /// grid / threadgroup) must be set before `.dispatch()` is callable.
    pub fn dispatch_builder(&self) -> DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
        DispatchBuilder::new(self.inner.clone())
    }

    /// Access the underlying inner Arc (used by dispatch builder; not part
    /// of the public API).
    // T5 will call inner_arc() in DispatchBuilder tests; allow dead_code until then.
    #[allow(dead_code)]
    pub(crate) fn inner_arc(&self) -> &Arc<MetalKernelInner> {
        &self.inner
    }
}

/// Build-time configuration for a Metal kernel.
pub struct MetalKernelBuilder {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    source: String,
    header: String,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
}

impl MetalKernelBuilder {
    /// Set input parameter names.
    pub fn inputs(mut self, names: &[&str]) -> Self {
        self.input_names = names.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set output parameter names. Number of outputs is fixed at build time
    /// and must match the size of `output_shapes` / `output_dtypes` passed at
    /// dispatch time (verified at runtime in `dispatch()`).
    pub fn outputs(mut self, names: &[&str]) -> Self {
        self.output_names = names.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set the Metal kernel source code (function body — not a full
    /// `kernel void f(...)` declaration; MLX wraps this).
    pub fn source(mut self, src: impl Into<String>) -> Self {
        self.source = src.into();
        self
    }

    /// Set an optional Metal header included before the kernel source.
    pub fn header(mut self, hdr: impl Into<String>) -> Self {
        self.header = hdr.into();
        self
    }

    /// Whether MLX should ensure inputs are row-contiguous before passing.
    /// Default `true`.
    pub fn ensure_row_contiguous(mut self, v: bool) -> Self {
        self.ensure_row_contiguous = v;
        self
    }

    /// Whether outputs should be initialized for atomic accumulation.
    /// Default `false`.
    pub fn atomic_outputs(mut self, v: bool) -> Self {
        self.atomic_outputs = v;
        self
    }

    /// Compile the kernel.
    pub fn build(self) -> Result<MetalKernel> {
        if self.input_names.is_empty() {
            return Err(Error::Mlx(
                "MetalKernelBuilder: must call inputs(...) before build()".to_owned(),
            ));
        }
        if self.output_names.is_empty() {
            return Err(Error::Mlx(
                "MetalKernelBuilder: must call outputs(...) before build()".to_owned(),
            ));
        }
        if self.source.is_empty() {
            return Err(Error::Mlx(
                "MetalKernelBuilder: must call source(...) before build()".to_owned(),
            ));
        }
        let output_count = self.output_names.len();
        let handle = mlx_sys::fast::ffi::metal_kernel_build(
            &self.name,
            &self.input_names,
            &self.output_names,
            &self.source,
            &self.header,
            self.ensure_row_contiguous,
            self.atomic_outputs,
        )
        .map_err(Error::from)?;
        Ok(MetalKernel {
            inner: Arc::new(MetalKernelInner {
                handle,
                output_count,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_rejects_empty_inputs() {
        let r = MetalKernel::builder("k")
            .outputs(&["y"])
            .source("y[0] = 1.0;")
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn builder_rejects_empty_outputs() {
        let r = MetalKernel::builder("k")
            .inputs(&["x"])
            .source("y[0] = 1.0;")
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn builder_rejects_empty_source() {
        let r = MetalKernel::builder("k")
            .inputs(&["x"])
            .outputs(&["y"])
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn build_succeeds_with_valid_inputs() {
        let r = MetalKernel::builder("trivial_add_one")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
            .build();
        assert!(r.is_ok(), "build should succeed: {:?}", r.err());
    }

    #[test]
    fn clone_is_arc_share() {
        let k = MetalKernel::builder("k")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("y[0] = 1.0;")
            .build()
            .unwrap();
        let k2 = k.clone();
        assert!(Arc::ptr_eq(k.inner_arc(), k2.inner_arc()));
    }
}
