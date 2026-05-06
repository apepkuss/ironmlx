//! MLX `compile()` — JIT-trace Rust closures into MLX graphs for fused
//! execution. Tasks 2/3 add the closure-binding surface; this file currently
//! only exposes the global controls and the `CompileMode` enum.

/// Global compile mode. Mirrors `mlx::core::CompileMode`.
#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    /// Compile is fully disabled; functions run eagerly.
    Disabled = 0,
    /// Compile, but skip the simplify pass.
    NoSimplify = 1,
    /// Compile, but skip kernel fusion.
    NoFuse = 2,
    /// Full compile (default).
    #[default]
    Enabled = 3,
}

/// Globally disable MLX compile. Equivalent to `set_compile_mode(Disabled)`.
pub fn disable_compile() {
    mlx_sys::compile::ffi::disable_compile();
}

/// Globally enable MLX compile. Equivalent to `set_compile_mode(Enabled)`.
pub fn enable_compile() {
    mlx_sys::compile::ffi::enable_compile();
}

/// Set the global compile mode.
pub fn set_compile_mode(mode: CompileMode) {
    mlx_sys::compile::ffi::set_compile_mode(mode as u8);
}

// ===== CompiledFn + compile() =====

use crate::{Array, Error, Result};
use cxx::UniquePtr;

/// A compiled MLX function. Wraps an opaque `std::function` produced by
/// `mlx::core::compile`. Drop releases the underlying trace.
pub struct CompiledFn {
    inner: UniquePtr<mlx_sys::compile::ffi::CompiledFn>,
}

impl CompiledFn {
    /// Replay the compiled graph on the given inputs.
    ///
    /// Errors surface from any of: the user closure returning `Err` (or
    /// panicking, caught and converted to `Err`), MLX trace failures, or
    /// shape mismatches at replay time. On error, no partial outputs are
    /// returned.
    pub fn invoke(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        let mut in_vec = mlx_sys::compile::ffi::array_vec_new();
        for a in inputs {
            mlx_sys::compile::ffi::array_vec_push(in_vec.pin_mut(), a.as_inner());
        }

        let mut out_vec =
            mlx_sys::compile::ffi::compiled_fn_invoke(&self.inner, &in_vec).map_err(Error::from)?;

        let n = mlx_sys::compile::ffi::array_vec_count(&out_vec);
        let mut outs: Vec<Array> = Vec::with_capacity(n);
        // Drain front-to-back: each take_at removes index 0.
        for _ in 0..n {
            let a = mlx_sys::compile::ffi::array_vec_take_at(out_vec.pin_mut(), 0)
                .map_err(Error::from)?;
            outs.push(Array::from_inner(a));
        }
        Ok(outs)
    }
}

/// JIT-compile a Rust closure into an MLX traced graph.
///
/// The closure is invoked once at trace time (and again on shape changes
/// when `shapeless=false`). Every MLX op the closure runs is recorded;
/// subsequent calls to [`CompiledFn::invoke`] replay the optimized graph
/// without re-running the closure.
///
/// The closure must be `Send + 'static`. `Sync` is intentionally NOT
/// required because `mlx::Array` is `!Sync`; MLX invokes the closure
/// sequentially per `CompiledFn`, so non-`Sync` captures are safe.
///
/// Returning `Err` from the closure (or panicking) yields a Rust `Err`
/// from `compile()` or `invoke()`; the panic is caught by cxx, never
/// aborts the process.
///
/// Errors returned by the closure are converted to a string at the FFI
/// boundary, so structured `Error` variants (`ShapeMismatch`,
/// `DtypeMismatch`, `BroadcastMismatch`) emerge as `Error::Mlx(String)`
/// from `compile()` / `invoke()`. Closure callers that need structured
/// errors should panic with a typed payload instead.
pub fn compile<F>(f: F, shapeless: bool) -> Result<CompiledFn>
where
    // `Send + 'static` only: `Sync` is intentionally NOT required, so users
    // can capture `Array` (which is `Send` but `!Sync`, see `mlx/src/array.rs`)
    // directly in a `move` closure. MLX's compiled-function machinery invokes
    // the closure sequentially per `CompiledFn`, so a non-`Sync` body is safe.
    // Concurrent replay of the same `CompiledFn` from multiple threads is
    // not currently supported by MLX core in any case.
    F: Fn(&[&Array]) -> Result<Vec<Array>> + Send + 'static,
{
    // Adapt user closure (Array-level) to bridge closure (MlxArray-level).
    let bridge_fn: Box<mlx_sys::compile::CallbackFn> =
        Box::new(move |refs: &[&mlx_sys::array::ffi::MlxArray]| {
            // Build ephemeral safe Arrays around each input. array_clone is
            // a cheap MLX refcount copy so this does not duplicate buffers.
            let temp: Vec<Array> = refs
                .iter()
                .map(|m| Array::from_inner(mlx_sys::array::ffi::array_clone(m)))
                .collect();
            let borrows: Vec<&Array> = temp.iter().collect();

            let outs = f(&borrows).map_err(|e| e.to_string())?;

            // Convert Vec<Array> → Vec<UniquePtr<MlxArray>> by consuming each
            // Array (Array::into_inner moves out the cxx ptr).
            Ok(outs.into_iter().map(Array::into_inner).collect())
        });

    let cb = mlx_sys::compile::make_callback(bridge_fn);
    let inner = mlx_sys::compile::ffi::compile_with_callback(cb, shapeless).map_err(Error::from)?;
    Ok(CompiledFn { inner })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_compile_mode_is_enabled() {
        assert_eq!(CompileMode::default(), CompileMode::Enabled);
    }
}
