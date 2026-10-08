//! Einstein contraction through MLX's native contraction planner.
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/einsum.h");
        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ArrayVec = crate::bridge::compile::ffi::ArrayVec;
        unsafe fn ops_einsum(
            equation: &str,
            operands: &ArrayVec,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
