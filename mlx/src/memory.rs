use mlx_sys as sys;

use crate::error::{check, Result};

/// Get the currently allocated memory in bytes.
pub fn get_active_memory() -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_get_active_memory(&mut res) })?;
    Ok(res)
}

/// Get the current cache memory in bytes.
pub fn get_cache_memory() -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_get_cache_memory(&mut res) })?;
    Ok(res)
}

/// Get the peak memory usage in bytes.
pub fn get_peak_memory() -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_get_peak_memory(&mut res) })?;
    Ok(res)
}

/// Get the current memory limit in bytes.
pub fn get_memory_limit() -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_get_memory_limit(&mut res) })?;
    Ok(res)
}

/// Reset the peak memory counter.
pub fn reset_peak_memory() -> Result<()> {
    check(unsafe { sys::mlx_reset_peak_memory() })
}

/// Set the memory limit in bytes, returning the previous limit.
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_set_memory_limit(&mut res, limit) })?;
    Ok(res)
}

/// Set the cache limit in bytes, returning the previous limit.
pub fn set_cache_limit(limit: usize) -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_set_cache_limit(&mut res, limit) })?;
    Ok(res)
}

/// Set the wired memory limit in bytes, returning the previous limit.
pub fn set_wired_limit(limit: usize) -> Result<usize> {
    let mut res: usize = 0;
    check(unsafe { sys::mlx_set_wired_limit(&mut res, limit) })?;
    Ok(res)
}

/// Clear the memory cache.
pub fn clear_cache() -> Result<()> {
    check(unsafe { sys::mlx_clear_cache() })
}
