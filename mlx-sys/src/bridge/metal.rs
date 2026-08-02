//! Bridge for `mlx::core::metal::{start,stop}_capture` (debug/profiling) and
//! `mlx::core::metal::device_info()['architecture']` (device-arch lookup
//! tables — used by ironmlx-p8a-stage9 to pick per-chip kernel tiles).
//!
//! All functions in this bridge can throw on the C++ side (driver /
//! capture-manager failure for capture; missing Metal backend or unexpected
//! variant type for architecture), so they're declared `Result<T>` per the
//! project's "shim-can-throw → cxx Result" rule (`bridge/mod.rs`).

#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/metal.h");

        /// Open an Xcode-compatible `.gputrace` file at `path` and start
        /// capturing every Metal command submitted on the default device.
        ///
        /// Throws on driver / capture-manager failure: missing Xcode
        /// entitlement (`MTL_CAPTURE_ENABLED=1`), path not writable, capture
        /// already running, or no Metal device available.
        fn start_capture(path: &str) -> Result<()>;

        /// Finalize the in-progress capture. Throws if no capture is active.
        fn stop_capture() -> Result<()>;

        /// Configure the exact MLX metallib loaded by this process.
        fn set_metallib_path(path: &str) -> Result<()>;

        /// Return the Metal device's architecture name (e.g. `"apple_g13s"`
        /// for an M1 Pro 16-core GPU, `"apple_g15p"` for M3 Pro), as
        /// reported by `MTLDevice.architecture.name` and exposed via
        /// `mlx::core::metal::device_info()['architecture']`.
        ///
        /// Throws if the Metal backend isn't available on this system or
        /// the entry's variant is unexpectedly not a string (defensive —
        /// MLX always stores it as `std::string`).
        fn device_architecture() -> Result<String>;
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{
        device_architecture, set_metallib_path, start_capture, stop_capture,
    };
}
