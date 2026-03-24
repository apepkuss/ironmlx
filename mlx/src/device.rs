use mlx_sys as sys;

/// Compute device: CPU or GPU (Metal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Gpu,
}

impl DeviceType {
    fn to_raw(self) -> sys::mlx_device_type {
        match self {
            DeviceType::Cpu => sys::mlx_device_type__MLX_CPU,
            DeviceType::Gpu => sys::mlx_device_type__MLX_GPU,
        }
    }

    fn from_raw(raw: sys::mlx_device_type) -> Self {
        match raw {
            sys::mlx_device_type__MLX_CPU => DeviceType::Cpu,
            sys::mlx_device_type__MLX_GPU => DeviceType::Gpu,
            _ => panic!("Unknown mlx_device_type: {}", raw),
        }
    }
}

/// A handle to an MLX compute device.
pub struct Device(sys::mlx_device);

impl Device {
    /// Create a device of the given type (index 0).
    pub fn new(kind: DeviceType) -> Self {
        let raw = unsafe { sys::mlx_device_new_type(kind.to_raw(), 0) };
        Device(raw)
    }

    pub fn cpu() -> Self {
        Self::new(DeviceType::Cpu)
    }

    pub fn gpu() -> Self {
        Self::new(DeviceType::Gpu)
    }

    pub fn device_type(&self) -> DeviceType {
        let mut kind: sys::mlx_device_type = sys::mlx_device_type__MLX_CPU;
        unsafe { sys::mlx_device_get_type(&mut kind, self.0) };
        DeviceType::from_raw(kind)
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_device {
        self.0
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { sys::mlx_device_free(self.0) };
    }
}

/// Device information retrieved from the MLX backend.
pub struct DeviceInfo(sys::mlx_device_info);

impl DeviceInfo {
    /// Query device info for the given device.
    pub fn get(device: &Device) -> Option<Self> {
        unsafe {
            let mut info = sys::mlx_device_info_new();
            if sys::mlx_device_info_get(&mut info, device.as_raw()) == 0 && !info.ctx.is_null() {
                Some(DeviceInfo(info))
            } else {
                sys::mlx_device_info_free(info);
                None
            }
        }
    }

    /// Get a size_t value by key (e.g. "memory_size", "max_recommended_working_set_size").
    pub fn get_size(&self, key: &str) -> Option<usize> {
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut value: usize = 0;
        let rc = unsafe { sys::mlx_device_info_get_size(&mut value, self.0, c_key.as_ptr()) };
        if rc == 0 { Some(value) } else { None }
    }

    /// Get a string value by key (e.g. "device_name", "architecture").
    pub fn get_string(&self, key: &str) -> Option<String> {
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut ptr: *const std::os::raw::c_char = std::ptr::null();
        let rc = unsafe { sys::mlx_device_info_get_string(&mut ptr, self.0, c_key.as_ptr()) };
        if rc == 0 && !ptr.is_null() {
            Some(
                unsafe { std::ffi::CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        } else {
            None
        }
    }
}

impl Drop for DeviceInfo {
    fn drop(&mut self) {
        unsafe { sys::mlx_device_info_free(self.0) };
    }
}
