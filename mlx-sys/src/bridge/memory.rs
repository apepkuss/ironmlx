//! Bridge for MLX allocator memory counters.

// The safety contract of the unsafe functions (`pin_resident`,
// `unpin_resident`) is documented in the safe wrappers in `mlx::memory`.
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/memory.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;

        fn get_active_memory() -> usize;
        fn get_cache_memory() -> usize;
        fn get_peak_memory() -> usize;
        fn reset_peak_memory();
        fn get_memory_limit() -> usize;
        fn set_cache_limit(limit: usize) -> usize;
        fn set_wired_limit(limit: usize) -> Result<usize>;
        fn start_residency_refresh(interval_ms: u32) -> Result<()>;
        unsafe fn pin_resident(arrays: &[*const MlxArray]) -> Result<usize>;
        unsafe fn unpin_resident(arrays: &[*const MlxArray]) -> Result<usize>;
        fn get_pinned_memory() -> usize;
        fn get_memory_size() -> Result<usize>;
        fn get_max_recommended_memory() -> Result<usize>;
        fn get_device_name() -> Result<String>;
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{
        get_active_memory, get_cache_memory, get_device_name, get_max_recommended_memory,
        get_memory_limit, get_memory_size, get_peak_memory, get_pinned_memory, pin_resident,
        reset_peak_memory, set_cache_limit, set_wired_limit, start_residency_refresh,
        unpin_resident,
    };
}
