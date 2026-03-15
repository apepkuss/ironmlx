use mlx_sys as sys;

use crate::error::{check, Result};
use crate::stream::Stream;
use crate::vector::{MapStringToArray, MapStringToString};

/// Load arrays and metadata from a safetensors file.
pub fn load_safetensors(
    path: &str,
    stream: &Stream,
) -> Result<(MapStringToArray, MapStringToString)> {
    let mut arrays = MapStringToArray::new();
    let mut metadata = MapStringToString::new();
    let c_path =
        std::ffi::CString::new(path).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
    check(unsafe {
        sys::mlx_load_safetensors(
            arrays.as_raw_mut(),
            metadata.as_raw_mut(),
            c_path.as_ptr(),
            stream.as_raw(),
        )
    })?;
    Ok((arrays, metadata))
}

/// Save arrays and metadata to a safetensors file.
pub fn save_safetensors(
    path: &str,
    params: &MapStringToArray,
    metadata: &MapStringToString,
) -> Result<()> {
    let c_path =
        std::ffi::CString::new(path).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
    check(unsafe { sys::mlx_save_safetensors(c_path.as_ptr(), params.as_raw(), metadata.as_raw()) })
}
