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

pub fn try_snapshot() -> crate::Result<MemorySnapshot> {
    Ok(MemorySnapshot {
        active_bytes: mlx_sys::memory::ffi::get_active_memory()?,
        cache_bytes: mlx_sys::memory::ffi::get_cache_memory()?,
        peak_bytes: mlx_sys::memory::ffi::get_peak_memory()?,
        memory_limit_bytes: mlx_sys::memory::ffi::get_memory_limit()?,
        total_bytes: mlx_sys::memory::ffi::get_memory_size().ok(),
        max_recommended_bytes: mlx_sys::memory::ffi::get_max_recommended_memory().ok(),
        device_name: mlx_sys::memory::ffi::get_device_name().ok(),
    })
}

/// # Panics
/// When the MLX device cannot initialize. Zeroed counters would read as free
/// memory; entry points call [`crate::metal::verify_runtime`] to report this
/// as an error before any snapshot.
pub fn snapshot() -> MemorySnapshot {
    try_snapshot().unwrap_or_else(|error| panic!("MLX memory counters unavailable: {error}"))
}

/// Set the allocator cache limit and return the previous limit.
///
/// # Panics
/// When the MLX device cannot initialize, as [`snapshot`] does.
pub fn set_cache_limit(limit: usize) -> usize {
    mlx_sys::memory::ffi::set_cache_limit(limit)
        .unwrap_or_else(|error| panic!("MLX cache limit unavailable: {error}"))
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

/// A standing Metal residency request for only these evaluated weights.
/// Dropping the lease ends its requests before releasing its storage handles.
/// It does not change the allocator's wired limit. Bind it to each execution
/// stream before use; dropping/suspending it removes its queue registrations.
/// While any lease is active its request is renewed every 500 ms, because
/// macOS drops residency within seconds of GPU idleness; no GPU work is sent.
pub struct WeightResidency(cxx::UniquePtr<mlx_sys::memory::ffi::ScopedResidency>);

// The immutable weight handles move with the lease, like Array. Metal set
// mutations and destruction are serialized by the shim's registry mutex.
unsafe impl Send for WeightResidency {}

impl WeightResidency {
    pub fn new(weights: &[&crate::Array], budget_bytes: usize) -> crate::Result<Self> {
        let pointers: Vec<_> = weights.iter().map(|a| a.as_inner() as *const _).collect();
        // SAFETY: all pointers reference arrays that live through this call;
        // the shim validates availability and retains their storage handles.
        Ok(Self(unsafe {
            mlx_sys::memory::ffi::make_scoped_residency(&pointers, budget_bytes)?
        }))
    }

    pub fn bytes(&self) -> usize {
        mlx_sys::memory::ffi::scoped_residency_bytes(self.0.as_ref().expect("residency lease"))
    }

    pub fn bind_on(&self, target: impl Into<crate::StreamOrDevice>) -> crate::Result<bool> {
        let (has, device_only, device, stream) = target.into().encode();
        Ok(mlx_sys::memory::ffi::scoped_residency_bind(
            self.0.as_ref().expect("residency lease"),
            has,
            device_only,
            device,
            stream,
        )?)
    }

    /// Bind the lease to the calling thread's default queue and wait for one
    /// submission on it. Run right after the request: left unattached, the
    /// system re-establishes the requested pages at a later GPU submission,
    /// which stalls whichever work happens to come first. One-shot, not a
    /// keepalive. Returns false, without GPU work, while the lease is paused.
    pub fn activate_on_current_queue(&self) -> crate::Result<bool> {
        if !self.bind_on(())? {
            return Ok(false);
        }
        let probe = crate::Array::zeros([1], crate::Dtype::Uint32)?;
        crate::ops::binary::add(&probe, &probe)?.eval()?;
        crate::transforms::synchronize()?;
        Ok(true)
    }
}

pub use mlx_sys::memory::ffi::ScopedResidencyStats as WeightResidencyStats;

pub fn weight_residency_stats() -> WeightResidencyStats {
    mlx_sys::memory::ffi::scoped_residency_stats()
}

/// Suspend only owned weight requests under pressure. Allocations stay valid
/// for ordinary GPU execution. Recovery reissues these requests, not a timer.
pub fn set_weight_residency_enabled(enabled: bool) -> crate::Result<()> {
    Ok(mlx_sys::memory::ffi::set_scoped_residency_enabled(enabled)?)
}
