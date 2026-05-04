//! Device types and queries.
//!
//! [`Device`] is a POD struct (8 bytes) describing where MLX should run
//! computations. On macOS Apple Silicon, the default is the GPU (Metal).
//! [`Device::cpu()`] and [`Device::gpu`] are convenience constructors;
//! the type itself is `Copy + PartialEq + Eq + Debug`.
//!
//! ## Layout note
//!
//! The cxx bridge's `Device` struct stores `device_type` as `i32` (a
//! wire-compatible representation of MLX's `enum class : int`). To keep the
//! safe-layer API strongly-typed, [`Device`] here is a **separate** struct
//! whose `device_type` field is the native Rust [`DeviceType`] enum. Both
//! structs have identical layout (i32 + i32), and `From` impls in both
//! directions perform a trivial field copy at the FFI boundary.

pub use mlx_sys::stream::ffi::DeviceType;

/// Where MLX should run computations. POD value type, cheap to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    pub device_type: DeviceType,
    pub index: i32,
}

impl Device {
    /// CPU device (always index 0; MLX exposes only one CPU "device").
    pub const fn cpu() -> Self {
        Device {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// GPU device with the given index. On macOS Apple Silicon there is
    /// typically only one GPU (index 0).
    pub const fn gpu(index: i32) -> Self {
        Device {
            device_type: DeviceType::Gpu,
            index,
        }
    }
}

impl From<mlx_sys::stream::ffi::Device> for Device {
    fn from(d: mlx_sys::stream::ffi::Device) -> Self {
        let device_type = match d.device_type {
            0 => DeviceType::Cpu,
            1 => DeviceType::Gpu,
            other => panic!("unknown MLX DeviceType discriminant: {other}"),
        };
        Device {
            device_type,
            index: d.index,
        }
    }
}

impl From<Device> for mlx_sys::stream::ffi::Device {
    fn from(d: Device) -> Self {
        mlx_sys::stream::ffi::Device {
            device_type: d.device_type as i32,
            index: d.index,
        }
    }
}

/// Get the current thread's default device (where ops execute by default).
pub fn default_device() -> Device {
    mlx_sys::stream::ffi::default_device().into()
}

/// Set the current thread's default device. Subsequent ops on this thread
/// will execute on `d` unless an explicit stream/device override is provided.
///
/// This is **thread-local** in MLX — setting on thread A does not affect
/// thread B.
pub fn set_default_device(d: Device) {
    mlx_sys::stream::ffi::set_default_device(d.into());
}

/// Returns `true` if MLX has the given device available on this system.
pub fn is_available(d: Device) -> bool {
    mlx_sys::stream::ffi::is_available(d.into())
}

/// Number of devices of the given type available on this system.
pub fn device_count(t: DeviceType) -> i32 {
    mlx_sys::stream::ffi::device_count(t as i32)
}
