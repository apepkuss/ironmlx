//! Bridge for MLX allocator memory counters.

#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/memory.h");

        fn get_active_memory() -> usize;
        fn get_cache_memory() -> usize;
        fn get_peak_memory() -> usize;
        fn get_memory_limit() -> usize;
        fn set_cache_limit(limit: usize) -> usize;
        fn get_memory_size() -> Result<usize>;
        fn get_max_recommended_memory() -> Result<usize>;
        fn get_device_name() -> Result<String>;
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{
        get_active_memory, get_cache_memory, get_device_name, get_max_recommended_memory,
        get_memory_limit, get_memory_size, get_peak_memory, set_cache_limit,
    };
}
