//! Real Fourier transform bridge.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/fft.h");
        type MlxArray = crate::bridge::array::ffi::MlxArray;
        unsafe fn ops_real_fft(
            input: &MlxArray,
            n: i32,
            axis: i32,
            inverse: bool,
            norm: u8,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
