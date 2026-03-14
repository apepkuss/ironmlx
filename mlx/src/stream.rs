use mlx_sys as sys;

use crate::device::Device;

/// An MLX execution stream (async computation context).
pub struct Stream(sys::mlx_stream);

impl Stream {
    /// Create a new stream on the given device.
    pub fn new(device: &Device) -> Self {
        let raw = unsafe { sys::mlx_stream_new_device(device.as_raw()) };
        Stream(raw)
    }

    /// Return the default stream for the given device.
    pub fn default_stream(device: &Device) -> Self {
        let raw = unsafe { sys::mlx_default_stream(device.as_raw()) };
        Stream(raw)
    }

    /// Block until all operations on this stream have completed.
    pub fn synchronize(&self) {
        unsafe { sys::mlx_synchronize(self.0) };
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_stream {
        self.0
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::mlx_stream_free(self.0) };
    }
}
