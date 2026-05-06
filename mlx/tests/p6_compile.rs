//! Integration tests for mlx::compile — JIT compilation of Rust closures.

use mlx::compile::{disable_compile, enable_compile, set_compile_mode, CompileMode};

#[test]
fn compile_mode_setters() {
    // Round-trip every variant; the calls must not panic and must leave
    // compile in a usable (enabled) state for subsequent tests.
    set_compile_mode(CompileMode::Disabled);
    set_compile_mode(CompileMode::NoSimplify);
    set_compile_mode(CompileMode::NoFuse);
    set_compile_mode(CompileMode::Enabled);
    disable_compile();
    enable_compile();
}
