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
