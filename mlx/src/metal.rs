use mlx_sys as sys;

use crate::error::{Result, check};

/// Check whether a Metal GPU device is available.
pub fn is_available() -> Result<bool> {
    let mut res: bool = false;
    check(unsafe { sys::mlx_metal_is_available(&mut res) })?;
    Ok(res)
}

/// Start a Metal GPU capture to the given file path.
pub fn start_capture(path: &str) -> Result<()> {
    let c_path =
        std::ffi::CString::new(path).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
    check(unsafe { sys::mlx_metal_start_capture(c_path.as_ptr()) })
}

/// Stop an active Metal GPU capture.
pub fn stop_capture() -> Result<()> {
    check(unsafe { sys::mlx_metal_stop_capture() })
}

/// Set the maximum number of operations per Metal command buffer.
/// Lower values improve stability for large prompts at a small throughput cost.
pub fn set_max_ops_per_buffer(val: i32) -> Result<()> {
    check(unsafe { sys::mlx_metal_set_max_ops_per_buffer(val) })
}

/// Set the maximum MB of data per Metal command buffer.
pub fn set_max_mb_per_buffer(val: i32) -> Result<()> {
    check(unsafe { sys::mlx_metal_set_max_mb_per_buffer(val) })
}
