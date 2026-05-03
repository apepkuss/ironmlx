#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");

        /// Opaque holder for `mlx::core::array`. Internally refcounted by MLX.
        type MlxArray;

        fn array_zeros(shape: &[i32], dtype: u8) -> Result<UniquePtr<MlxArray>>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
        fn array_ndim(a: &MlxArray) -> usize;
        fn array_size(a: &MlxArray) -> usize;
        fn array_dtype(a: &MlxArray) -> u8;
        fn array_clone(a: &MlxArray) -> UniquePtr<MlxArray>;
        fn array_is_available(a: &MlxArray) -> bool;
    }
}
