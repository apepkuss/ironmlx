//! Bridge for MLX quantization subsystem.
//!
//! Quantize returns std::vector<array>, which cxx 1.0 doesn't support
//! as Vec<UniquePtr<T>>. Wrapped as opaque QuantizeResult with
//! count() + take_at(idx) free functions. Single-use semantics:
//! take_at(idx) twice throws (matches P2c take_by_name pattern).
//!
//! Optional encodings:
//! - Option<i32>   → (bool has_value, i32 value)  (P2b rope pattern)
//! - Option<Dtype> → (bool has_dtype, u8 dtype_repr)
//! - Option<&Array>→ *const MlxArray (nullptr = None)  (P2b/P2c pattern)
//! - &str mode     → rust::Str  (P2b sdpa pattern)

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/quantization.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type QuantizeResult;

        // ===== quantize result accessors =====
        fn quantize_result_count(r: &QuantizeResult) -> usize;
        fn quantize_result_take_at(
            r: Pin<&mut QuantizeResult>,
            idx: usize,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== quantize =====
        unsafe fn quantize(
            w: &MlxArray,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            global_scale: *const MlxArray,
        ) -> Result<UniquePtr<QuantizeResult>>;

        // ===== dequantize =====
        unsafe fn dequantize(
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            global_scale: *const MlxArray,
            has_dtype: bool,
            dtype_repr: u8,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
