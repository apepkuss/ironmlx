//! MLX allocator memory counters.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The highest peak of the windows [`begin_peak_window`] has closed.
static CLOSED_WINDOWS_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

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
        peak_bytes: peak_window_bytes().max(CLOSED_WINDOWS_PEAK_BYTES.load(Ordering::Relaxed)),
        memory_limit_bytes: mlx_sys::memory::ffi::get_memory_limit(),
        total_bytes: mlx_sys::memory::ffi::get_memory_size().ok(),
        max_recommended_bytes: mlx_sys::memory::ffi::get_max_recommended_memory().ok(),
        device_name: mlx_sys::memory::ffi::get_device_name().ok(),
    }
}

/// Start measuring the peak of active memory afresh: [`peak_window_bytes`]
/// then reports the peak reached since this call. [`snapshot`] keeps
/// reporting the peak since the process started.
pub fn begin_peak_window() {
    CLOSED_WINDOWS_PEAK_BYTES.fetch_max(peak_window_bytes(), Ordering::Relaxed);
    mlx_sys::memory::ffi::reset_peak_memory();
}

/// Peak active memory since the last [`begin_peak_window`].
pub fn peak_window_bytes() -> usize {
    mlx_sys::memory::ffi::get_peak_memory()
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

/// Keep the memory of evaluated arrays GPU-resident and return the bytes
/// newly pinned. A pinned buffer stays resident whatever the wired limit is
/// and does not use its budget; the pin ends when the buffer is unpinned or
/// freed. Arrays that are not evaluated, and small buffers that share the
/// allocator's heap, are skipped.
pub fn pin_resident(arrays: &[&crate::Array]) -> crate::Result<usize> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: each pointer borrows a live `&Array` for the duration of this
    // call; the shim copies the array handles before using them.
    Ok(unsafe { mlx_sys::memory::ffi::pin_resident(&raw) }?)
}

/// Undo [`pin_resident`] and return the bytes unpinned.
pub fn unpin_resident(arrays: &[&crate::Array]) -> crate::Result<usize> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: as in `pin_resident`.
    Ok(unsafe { mlx_sys::memory::ffi::unpin_resident(&raw) }?)
}

/// Bytes currently pinned by [`pin_resident`].
pub fn pinned_bytes() -> usize {
    mlx_sys::memory::ffi::get_pinned_memory()
}
