//! Bridge for MLX Device/Stream types and async transforms.
//!
//! Uses cxx shared structs (Device, Stream) for zero-overhead POD value
//! passing. Layout binary-compatible with mlx::core::Device/Stream (same
//! field order, same underlying types). Conversion to/from MLX native types
//! happens in the C++ shim via field-by-field copy.
//!
//! ## DeviceType note
//!
//! cxx shared enums are emitted as `#[repr(transparent)] struct { pub repr: T }`
//! on the Rust side, which does NOT support the `as` primitive cast syntax.
//! To keep the public API ergonomic (`d.device_type as i32`,
//! `DeviceType::Gpu as i32`), `DeviceType` is defined as a native Rust
//! `#[repr(i32)]` enum outside the cxx bridge, and `Device.device_type` uses
//! `i32` as the wire type (layout-identical). The public `ffi` module
//! re-exports `DeviceType` so callers use the canonical `ffi::DeviceType` path.

/// Mirror of `mlx::core::Device::DeviceType`. Native Rust `#[repr(i32)]` enum
/// so that `as i32` primitive casting works. Values match MLX's declaration
/// order in mlx/device.h (cpu=0, gpu=1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
}

// cxx::bridge generates `unsafe fn` declarations for our pointer-slice variants
// (async_eval_many). The Safety contract is documented in the safe Rust wrapper
// (`mlx::transforms::async_eval`); cxx doesn't propagate doc comments from
// inside the bridge macro.
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    /// Mirror of `mlx::core::Device`. POD, 8 bytes, layout-compatible with MLX.
    /// `device_type` stored as `i32` (same layout as `DeviceType`'s `#[repr(i32)]`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Device {
        pub device_type: i32,
        pub index: i32,
    }

    /// Mirror of `mlx::core::Stream`. POD, 12 bytes, layout-compatible with MLX.
    /// Streams must be obtained via `default_stream()` / `new_stream()` —
    /// constructing one with arbitrary `index` is undefined behavior in MLX.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Stream {
        pub index: i32,
        pub device: Device,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/stream.h");

        // Reuse the MlxArray opaque type from the array bridge for async_eval.
        type MlxArray = crate::bridge::array::ffi::MlxArray;

        // === Device queries ===
        fn default_device() -> Device;
        fn set_default_device(d: Device);
        fn is_available(d: Device) -> bool;
        fn device_count(t: i32) -> i32;

        // === Stream lifecycle ===
        fn default_stream(d: Device) -> Stream;
        fn new_stream(d: Device) -> Result<Stream>;
        fn set_default_stream(s: Stream);
        fn get_streams() -> Vec<Stream>;
        fn clear_streams();

        // === Transforms ===
        unsafe fn async_eval_many(arrays: &[*const MlxArray]) -> Result<()>;
        fn synchronize() -> Result<()>;
        fn synchronize_stream(s: Stream) -> Result<()>;
    }
}

/// Public FFI module. Re-exports the cxx bridge items plus `DeviceType`
/// (defined outside the bridge as a native Rust enum to support `as i32` casting).
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    pub use super::ffi_bridge::{
        async_eval_many, clear_streams, default_device, default_stream, device_count, get_streams,
        is_available, new_stream, set_default_device, set_default_stream, synchronize,
        synchronize_stream, Device, MlxArray, Stream,
    };
    pub use super::DeviceType;
}
