//! MLX `compile()` — JIT-trace Rust closures into MLX graphs for fused
//! execution. Tasks 2/3 add the closure-binding surface; this file currently
//! only exposes the global controls and the `CompileMode` enum.

/// Global compile mode. Mirrors `mlx::core::CompileMode`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Compile is fully disabled; functions run eagerly.
    Disabled = 0,
    /// Compile, but skip the simplify pass.
    NoSimplify = 1,
    /// Compile, but skip kernel fusion.
    NoFuse = 2,
    /// Full compile (default).
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
