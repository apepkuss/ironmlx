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

    /// List all available keys.
    pub fn keys(&self) -> Vec<String> {
        unsafe {
            let mut keys = sys::mlx_vector_string_new();
            if sys::mlx_device_info_get_keys(&mut keys, self.0) != 0 {
                sys::mlx_vector_string_free(keys);
                return vec![];
            }
            let len = sys::mlx_vector_string_size(keys);
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                let mut ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
                if sys::mlx_vector_string_get(&mut ptr, keys, i) == 0 && !ptr.is_null() {
                    result.push(
                        std::ffi::CStr::from_ptr(ptr as *const _)
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
            sys::mlx_vector_string_free(keys);
            result
        }
    }

    /// Check if a key holds a string value (vs size_t).
    pub fn is_string(&self, key: &str) -> bool {
        let c_key = match std::ffi::CString::new(key) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let mut result = false;
        unsafe { sys::mlx_device_info_is_string(&mut result, self.0, c_key.as_ptr()) };
        result
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
