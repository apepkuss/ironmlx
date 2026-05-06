//! Bridge for MLX compile subsystem.
//!
//! Closure binding via `extern "Rust" CompileCallback`. The shim wraps
//! the Rust callback in a `shared_ptr<rust::Box<CompileCallback>>` so that
//! the lambda passed to `mlx::core::compile` is `CopyConstructible` (a
//! requirement of `std::function`).

use crate::bridge::array::ffi::MlxArray;
use cxx::UniquePtr;

/// Type of the bridge-level user closure stored inside [`CompileCallback`].
///
/// Inputs come in as `&[&MlxArray]` (cheap-cloned by the bridge from a C++
/// `ArrayVec`). Outputs are returned as `Vec<UniquePtr<MlxArray>>`, which the
/// bridge then pushes into a fresh `ArrayVec` for the C++ side.
///
/// Errors are stringly-typed at this layer so the bridge stays decoupled
/// from the safe `mlx` crate's `Error` enum (which would create a crate
/// dependency cycle: `mlx-sys` ← `mlx`).
/// Internal — used by the safe `mlx::compile` adapter; not stable API.
#[doc(hidden)]
pub type CallbackFn = dyn Fn(&[&MlxArray]) -> Result<Vec<UniquePtr<MlxArray>>, String> + Send;

/// Wraps a user-provided Rust closure for use as an MLX trace target.
///
/// The closure runs once per trace (more if `shapeless=false` and the
/// shape changes) and must build an MLX graph from the inputs — every
/// op called on the inputs / capture variables is recorded by MLX.
///
/// Returning `Err` (or panicking) from the closure surfaces as a Rust
/// `Err` from `compile()` or `CompiledFn::invoke()`. cxx auto-translates
/// panics via `catch_unwind` because [`Self::invoke`] returns a `Result`.
// NOTE: The Rust safe API enforces single-threaded replay via `&CompiledFn`
// being `!Sync`. Direct sys-crate users must not concurrently invoke a
// CompiledFn — the user closure is only required to be `Send`, not `Sync`.
pub struct CompileCallback {
    f: Box<CallbackFn>,
}

impl CompileCallback {
    /// Materialize the C++-supplied input `ArrayVec` into a Rust slice of
    /// `&MlxArray`, dispatch the user closure, and pack outputs back into
    /// a new `ArrayVec`. Errors propagate as `cxx::Exception` to the C++
    /// caller (which re-raises a `std::exception`).
    ///
    /// We wrap the closure call in `catch_unwind` so that a Rust panic
    /// inside the user closure is converted to a `String` Err rather than
    /// aborting the process. cxx 1.0 aborts on un-caught panics across
    /// `extern "Rust"`, so this is mandatory for panic safety.
    fn invoke(&self, inputs: &ffi::ArrayVec) -> Result<UniquePtr<ffi::ArrayVec>, String> {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let n = ffi::array_vec_count(inputs);
        let owned: Vec<UniquePtr<MlxArray>> = (0..n)
            .map(|i| ffi::array_vec_get_at(inputs, i).map_err(|e| e.what().to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&MlxArray> = owned.iter().map(|p| &**p).collect();

        let outputs = match catch_unwind(AssertUnwindSafe(|| (self.f)(&refs))) {
            Ok(Ok(v)) => v,
            Ok(Err(s)) => return Err(s),
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "panic in compile callback".to_string()
                };
                return Err(format!("compile callback panicked: {msg}"));
            }
        };

        let mut out_vec = ffi::array_vec_new();
        for a in &outputs {
            ffi::array_vec_push(out_vec.pin_mut(), a);
        }
        Ok(out_vec)
    }
}

/// Construct a `CompileCallback` from a boxed bridge-level closure. Used
/// by the safe `mlx` crate to build the callback before passing it across
/// cxx into [`ffi::compile_with_callback`].
pub fn make_callback(f: Box<CallbackFn>) -> Box<CompileCallback> {
    Box::new(CompileCallback { f })
}

#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    extern "Rust" {
        type CompileCallback;
        // Result-returning so cxx wraps panics in catch_unwind. Without
        // Result, a Rust panic would abort the process.
        fn invoke(self: &CompileCallback, inputs: &ArrayVec) -> Result<UniquePtr<ArrayVec>>;
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ArrayVec;
        type CompiledFn;

        // ===== Global controls =====
        fn disable_compile();
        fn enable_compile();
        fn set_compile_mode(mode: u8);

        // ===== ArrayVec =====
        fn array_vec_new() -> UniquePtr<ArrayVec>;
        fn array_vec_count(v: &ArrayVec) -> usize;
        fn array_vec_get_at(v: &ArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_take_at(v: Pin<&mut ArrayVec>, i: usize) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_push(v: Pin<&mut ArrayVec>, a: &MlxArray);

        // ===== CompiledFn =====
        fn compile_with_callback(
            cb: Box<CompileCallback>,
            shapeless: bool,
        ) -> Result<UniquePtr<CompiledFn>>;
        fn compiled_fn_invoke(cf: &CompiledFn, inputs: &ArrayVec) -> Result<UniquePtr<ArrayVec>>;

        // ===== Compile cache control =====
        fn compile_clear_cache();
    }
}
