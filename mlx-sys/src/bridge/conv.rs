//! Bridge for MLX convolution ops (1D, 2D, and transposed 1D).
//!
//! Each fn carries 4 trailing `StreamOrDevice` args (P5.7) — same encoding
//! as the array bridge: `(has_target, is_device_only, device_type, stream_index)`.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/conv.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;

        unsafe fn ops_conv1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn ops_conv2d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_h: i32,
            dilation_w: i32,
            groups: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn ops_conv_transpose1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            output_padding: i32,
            groups: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
