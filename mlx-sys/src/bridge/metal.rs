//! Bridge for `mlx::core::metal::{start,stop}_capture` (debug/profiling).
//!
//! Used by ironmlx-p8a-stage4 to produce Xcode-readable `.gputrace` bundles
//! for per-kernel Metal profiling. Both functions can throw on driver /
//! capture-manager failure, so they're declared `Result<()>` in the bridge
//! per the project's "shim-can-throw → cxx Result" rule (`bridge/mod.rs`).

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
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{start_capture, stop_capture};
}
