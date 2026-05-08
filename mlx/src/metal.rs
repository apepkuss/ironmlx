//! Metal capture (debug / profiling) — wraps `mlx::core::metal::start_capture`
//! and `stop_capture` from MLX C++.
//!
//! Used to produce Xcode-readable `.gputrace` bundles for per-kernel Metal
//! profiling. The output file can be opened in Xcode to inspect each kernel's
//! GPU time, occupancy, argument binding, etc.
//!
//! # Requirements
//!
//! Capture requires the `MTL_CAPTURE_ENABLED=1` environment variable at the
//! time the process starts (Apple's gating for debug-mode capture). Without
//! it, `start_capture` throws "Metal Capture is not enabled".
//!
//! # Example
//!
//! ```no_run
//! use mlx::metal::capture;
//!
//! # fn main() -> mlx::Result<()> {
//! capture::start("/tmp/my_run.gputrace")?;
//! // ... run MLX ops you want to profile ...
//! capture::stop()?;
//! # Ok(())
//! # }
//! ```
//!
//! Open the resulting `.gputrace` in Xcode (File → Open) to view per-kernel
//! timing.

use crate::{Error, Result};

/// Open an Xcode-compatible `.gputrace` file at `path` and start capturing
/// every Metal command submitted on the default device until [`stop`] is
/// called.
///
/// Errors if Metal capture is not enabled (set `MTL_CAPTURE_ENABLED=1` before
/// process start), the path is not writable, or a capture is already running.
pub fn start(path: &str) -> Result<()> {
    mlx_sys::metal::ffi::start_capture(path).map_err(Error::from)
}

/// Finalize the in-progress capture started by [`start`]. Errors if no
/// capture is active.
pub fn stop() -> Result<()> {
    mlx_sys::metal::ffi::stop_capture().map_err(Error::from)
}
