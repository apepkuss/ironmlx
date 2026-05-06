//! Bridge for MLX compile subsystem.
//!
//! P6 Task 2 surface: global controls + ArrayVec opaque carrier. Task 3
//! adds the extern "Rust" CompileCallback, CompiledFn, and the compile
//! entry point.

#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ArrayVec;

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
    }
}
