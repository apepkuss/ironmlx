//! MLX allocator memory counters.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub active_bytes: usize,
    pub cache_bytes: usize,
    pub peak_bytes: usize,
    pub memory_limit_bytes: usize,
    pub total_bytes: Option<usize>,
    pub max_recommended_bytes: Option<usize>,
    pub device_name: Option<String>,
}

pub fn snapshot() -> MemorySnapshot {
    MemorySnapshot {
        active_bytes: mlx_sys::memory::ffi::get_active_memory(),
        cache_bytes: mlx_sys::memory::ffi::get_cache_memory(),
        peak_bytes: mlx_sys::memory::ffi::get_peak_memory(),
        memory_limit_bytes: mlx_sys::memory::ffi::get_memory_limit(),
        total_bytes: mlx_sys::memory::ffi::get_memory_size().ok(),
        max_recommended_bytes: mlx_sys::memory::ffi::get_max_recommended_memory().ok(),
        device_name: mlx_sys::memory::ffi::get_device_name().ok(),
    }
}
