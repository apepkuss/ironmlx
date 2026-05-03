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

        // item family — Result-wrapped (MLX item<T>() may throw if dtype mismatches
        // or eval fails for any reason).
        fn array_item_bool(a: &MlxArray) -> Result<bool>;
        fn array_item_u8(a: &MlxArray) -> Result<u8>;
        fn array_item_i8(a: &MlxArray) -> Result<i8>;
        fn array_item_i16(a: &MlxArray) -> Result<i16>;
        fn array_item_i32(a: &MlxArray) -> Result<i32>;
        fn array_item_i64(a: &MlxArray) -> Result<i64>;
        fn array_item_f16(a: &MlxArray) -> Result<u16>;
        fn array_item_bf16(a: &MlxArray) -> Result<u16>;
        fn array_item_f32(a: &MlxArray) -> Result<f32>;
        fn array_item_f64(a: &MlxArray) -> Result<f64>;

        // to_vec family — Result-wrapped (data() can throw if storage isn't
        // available, e.g. eval hasn't been called).
        fn array_to_vec_bool(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_u8(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_i8(a: &MlxArray) -> Result<Vec<i8>>;
        fn array_to_vec_i16(a: &MlxArray) -> Result<Vec<i16>>;
        fn array_to_vec_i32(a: &MlxArray) -> Result<Vec<i32>>;
        fn array_to_vec_i64(a: &MlxArray) -> Result<Vec<i64>>;
        fn array_to_vec_f16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_bf16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_f32(a: &MlxArray) -> Result<Vec<f32>>;
        fn array_to_vec_f64(a: &MlxArray) -> Result<Vec<f64>>;

        // Binary ops (P1b1) — Result-wrapped per the shim throw rule.
        fn array_add(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_subtract(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_multiply(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_divide(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;

        // Unary ops (P1b1) — Result-wrapped (MLX may throw on dtype not supported,
        // e.g. sqrt on integer types).
        fn array_negative(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_exp(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_log(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_sqrt(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_tanh(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_sigmoid(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_square(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_rsqrt(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_erf(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_reciprocal(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
    }
}
