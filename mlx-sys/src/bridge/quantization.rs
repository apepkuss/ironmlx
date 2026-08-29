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
//!
//! Each fn carries 4 trailing `StreamOrDevice` args (P5.7) — same encoding
//! as the array bridge: `(has_target, is_device_only, device_type, stream_index)`.

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
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
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
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== quantized_matmul =====
        unsafe fn quantized_matmul(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn quantized_matmul_batch_isolated(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn quantized_matmul_product_stable(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn quantized_matmul_product_stable_affine8_wide(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn quantized_matmul_bench_ms(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            runs: usize,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<Vec<f64>>;

        // ===== qqmm =====
        unsafe fn qqmm(
            x: &MlxArray,
            w: &MlxArray,
            w_scales: *const MlxArray,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            global_scale_x: *const MlxArray,
            global_scale_w: *const MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== gather_qmm =====
        unsafe fn gather_qmm(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            transpose: bool,
            has_group_size: bool,
            group_size: i32,
            has_bits: bool,
            bits: i32,
            mode: &str,
            sorted_indices: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== FP8 =====
        fn from_fp8(
            x: &MlxArray,
            dtype_repr: u8,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn to_fp8(
            x: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
