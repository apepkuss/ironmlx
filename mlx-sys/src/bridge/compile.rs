//! Bridge for MLX compile subsystem.
//!
//! P6 Task 1 surface: global controls only. Tasks 2/3 add ArrayVec, the
//! extern "Rust" CompileCallback, CompiledFn, and the compile entry point.

#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        // ===== Global controls =====
        fn disable_compile();
        fn enable_compile();
        fn set_compile_mode(mode: u8);
    }
}
