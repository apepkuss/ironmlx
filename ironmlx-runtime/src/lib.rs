//! Inference scheduling, model lifecycle, memory and shared cache services.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("ironmlx-runtime only supports macOS on Apple Silicon (aarch64-apple-darwin)");
pub mod core;
pub use anyhow::{Error, Result};
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub use core::scheduler_actor::test_support;
