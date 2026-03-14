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
            DeviceType::Cpu => sys::mlx_device_type_MLX_CPU,
            DeviceType::Gpu => sys::mlx_device_type_MLX_GPU,
        }
    }

    fn from_raw(raw: sys::mlx_device_type) -> Self {
        match raw {
            sys::mlx_device_type_MLX_CPU => DeviceType::Cpu,
            sys::mlx_device_type_MLX_GPU => DeviceType::Gpu,
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
        let mut kind: sys::mlx_device_type = sys::mlx_device_type_MLX_CPU;
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
