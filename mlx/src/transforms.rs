//! Computation graph transforms (sync + async evaluation).
//!
//! For lazy `Array::eval()` see `mlx/src/array.rs`. This module adds:
//!
//! - [`synchronize`] — block on the current thread's default stream
//! - [`synchronize_stream`] — block on a specific stream
//! - [`async_eval`] — submit + return a runtime-agnostic Future (Task 5)

use crate::{Error, Result, Stream};

/// Block the current thread until all queued work on the current thread's
/// **default stream** completes. To synchronize on a specific stream
/// (regardless of which stream is currently the default), use
/// [`synchronize_stream`].
pub fn synchronize() -> Result<()> {
    mlx_sys::stream::ffi::synchronize().map_err(Error::from)
}

/// Block the current thread until all queued work on the **given stream**
/// completes, regardless of which thread queued the work or which stream
/// is currently the default.
pub fn synchronize_stream(s: Stream) -> Result<()> {
    mlx_sys::stream::ffi::synchronize_stream(s.into()).map_err(Error::from)
}
