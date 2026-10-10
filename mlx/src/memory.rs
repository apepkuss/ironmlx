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

/// Set the allocator cache limit and return the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    mlx_sys::memory::ffi::set_cache_limit(limit)
}

/// Set the wired (GPU-resident) memory limit and return the previous limit.
/// Allocations up to the limit are kept resident through Metal residency
/// sets. Fails when the limit exceeds the device's recommended working set.
pub fn set_wired_limit(limit: usize) -> crate::Result<usize> {
    Ok(mlx_sys::memory::ffi::set_wired_limit(limit)?)
}

/// Renew the residency request of the wired allocations every `interval_ms`
/// for the rest of the process. macOS drops residency shortly after the GPU
/// goes idle despite a standing request. Later calls are no-ops.
pub fn start_residency_refresh(interval_ms: u32) -> crate::Result<()> {
    Ok(mlx_sys::memory::ffi::start_residency_refresh(interval_ms)?)
}
