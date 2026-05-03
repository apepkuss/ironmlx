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

        // from_slice family — Result-wrapped per the shim throw rule
        // (MLX may throw on shape×dtype size mismatch).
        fn array_from_bool(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u8(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i8(data: &[i8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i16(data: &[i16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i32(data: &[i32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i64(data: &[i64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_bf16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f32(data: &[f32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f64(data: &[f64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
    }
}
