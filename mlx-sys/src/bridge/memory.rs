//! Bridge for MLX allocator memory counters.

#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ScopedResidencyStats {
        leases: usize,
        reserved_bytes: usize,
        requested_bytes: usize,
        queue_attachments: usize,
    }
    unsafe extern "C++" {
        include!("cxx_mlx_shim/memory.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ScopedResidency;
        unsafe fn make_scoped_residency(
            arrays: &[*const MlxArray],
            budget_bytes: usize,
        ) -> Result<UniquePtr<ScopedResidency>>;
        fn scoped_residency_bytes(lease: &ScopedResidency) -> usize;
        fn scoped_residency_stats() -> ScopedResidencyStats;
        fn set_scoped_residency_enabled(enabled: bool) -> Result<()>;
        #[allow(clippy::too_many_arguments)]
        fn scoped_residency_bind(
            lease: &ScopedResidency,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<bool>;

        // These initialize the Metal device on first use, which throws when
        // its metallib cannot be loaded.
        fn get_active_memory() -> Result<usize>;
        fn get_cache_memory() -> Result<usize>;
        fn get_peak_memory() -> Result<usize>;
        fn get_memory_limit() -> Result<usize>;
        fn set_cache_limit(limit: usize) -> Result<usize>;
        fn set_wired_limit(limit: usize) -> Result<usize>;
        fn start_residency_refresh(interval_ms: u32) -> Result<()>;
        fn get_memory_size() -> Result<usize>;
        fn get_max_recommended_memory() -> Result<usize>;
        fn get_device_name() -> Result<String>;
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{
        get_active_memory, get_cache_memory, get_device_name, get_max_recommended_memory,
        get_memory_limit, get_memory_size, get_peak_memory, make_scoped_residency,
        scoped_residency_bind, scoped_residency_bytes, scoped_residency_stats, set_cache_limit,
        set_scoped_residency_enabled, set_wired_limit, start_residency_refresh, ScopedResidency,
        ScopedResidencyStats,
    };
}
