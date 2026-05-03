#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");

        /// Opaque holder for `mlx::core::array`. Internally refcounted by MLX.
        type MlxArray;

        fn array_zeros(shape: &[i32], dtype: u8) -> UniquePtr<MlxArray>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
    }
}
