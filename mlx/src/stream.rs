//! Stream lifecycle and management.
//!
//! `Stream` is a POD struct (12 bytes) representing an MLX execution stream.
//! Ops queued on the same stream execute in order; ops on different streams
//! may run concurrently. Streams are bound to a specific [`Device`].
//!
//! **Construction**: only obtain `Stream` values via [`default_stream`] or
//! [`new_stream`] — arbitrary indices do not correspond to real MLX stream
//! workers, and `synchronize_stream` / op dispatch on a fabricated index
//! throws inside MLX. The struct is `#[non_exhaustive]` so external callers
//! cannot construct one with a struct literal.
//!
//! Like [`crate::Device`], `Stream` is a safe-layer wrapper over
//! `mlx_sys::stream::ffi::Stream` with the strongly-typed [`Device`] field.

use crate::{Device, Error, Result};

/// MLX execution stream. POD value type, cheap to copy. Cannot be
/// constructed externally — use [`default_stream`] or [`new_stream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stream {
    pub index: i32,
    pub device: Device,
}

impl From<mlx_sys::stream::ffi::Stream> for Stream {
    fn from(s: mlx_sys::stream::ffi::Stream) -> Self {
        Stream {
            index: s.index,
            device: s.device.into(),
        }
    }
}

impl From<Stream> for mlx_sys::stream::ffi::Stream {
    fn from(s: Stream) -> Self {
        mlx_sys::stream::ffi::Stream {
            index: s.index,
            device: s.device.into(),
        }
    }
}

/// Get the default stream for the given device on the current thread.
pub fn default_stream(d: Device) -> Stream {
    mlx_sys::stream::ffi::default_stream(d.into()).into()
}

/// Create a new stream on the given device. The returned stream has a
/// fresh, unique index.
pub fn new_stream(d: Device) -> Result<Stream> {
    mlx_sys::stream::ffi::new_stream(d.into())
        .map(Into::into)
        .map_err(Error::from)
}

/// Make the stream the default for its device on the current thread.
/// Subsequent ops on this thread that target the stream's device will use
/// `s` unless explicitly overridden.
pub fn set_default_stream(s: Stream) {
    mlx_sys::stream::ffi::set_default_stream(s.into());
}

/// Return all streams currently registered on this thread (across all devices).
pub fn get_streams() -> Vec<Stream> {
    mlx_sys::stream::ffi::get_streams()
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Destroy all streams created in the current thread. The default stream
/// will be recreated lazily on the next access.
pub fn clear_streams() {
    mlx_sys::stream::ffi::clear_streams();
}
