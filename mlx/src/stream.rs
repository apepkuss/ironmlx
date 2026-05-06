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

/// Target stream or device for an op. Mirrors MLX C++
/// `mlx::core::StreamOrDevice` (`std::variant<monostate, Stream, Device>`).
///
/// Used by `*_on` op variants. Construct via:
/// - [`StreamOrDevice::default()`] or `().into()` — use MLX's current default
/// - `stream.into()` — explicit stream
/// - `device.into()` — that device's default stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamOrDevice {
    /// Use MLX's current default stream for the calling thread.
    #[default]
    Default,
    /// Use the specified stream.
    Stream(Stream),
    /// Use the default stream of the specified device.
    Device(crate::Device),
}

impl From<Stream> for StreamOrDevice {
    fn from(s: Stream) -> Self {
        StreamOrDevice::Stream(s)
    }
}

impl From<crate::Device> for StreamOrDevice {
    fn from(d: crate::Device) -> Self {
        StreamOrDevice::Device(d)
    }
}

impl From<()> for StreamOrDevice {
    fn from(_: ()) -> Self {
        StreamOrDevice::Default
    }
}

impl StreamOrDevice {
    /// Encode for FFI: `(has_target, is_device_only, device_type_repr, stream_index)`.
    /// `Default` → `(false, false, 0, 0)`. `Device` → `(true, true, dt, 0)`.
    /// `Stream` → `(true, false, dt, idx)`.
    ///
    /// Consumed by op `_on` variants in subsequent P5.7 tasks.
    pub(crate) fn encode(self) -> (bool, bool, u8, i32) {
        match self {
            StreamOrDevice::Default => (false, false, 0, 0),
            StreamOrDevice::Device(d) => (true, true, d.device_type as u8, 0),
            StreamOrDevice::Stream(s) => (true, false, s.device.device_type as u8, s.index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_or_device_default_is_default_variant() {
        assert_eq!(StreamOrDevice::default(), StreamOrDevice::Default);
    }

    #[test]
    fn stream_or_device_from_unit() {
        let s: StreamOrDevice = ().into();
        assert_eq!(s, StreamOrDevice::Default);
    }

    #[test]
    fn stream_or_device_from_stream() {
        let s = default_stream(crate::Device::cpu());
        let target: StreamOrDevice = s.into();
        assert_eq!(target, StreamOrDevice::Stream(s));
    }

    #[test]
    fn stream_or_device_from_device() {
        let target: StreamOrDevice = crate::Device::cpu().into();
        assert_eq!(target, StreamOrDevice::Device(crate::Device::cpu()));
    }

    #[test]
    fn encode_default() {
        let (has, dev_only, dt, idx) = StreamOrDevice::Default.encode();
        assert!(!has);
        assert!(!dev_only);
        assert_eq!(dt, 0);
        assert_eq!(idx, 0);
    }

    #[test]
    fn encode_device_only() {
        let target = StreamOrDevice::Device(crate::Device::cpu());
        let (has, dev_only, dt, idx) = target.encode();
        assert!(has);
        assert!(dev_only);
        assert_eq!(dt, crate::DeviceType::Cpu as u8);
        assert_eq!(idx, 0);
    }

    #[test]
    fn encode_full_stream() {
        let s = default_stream(crate::Device::cpu());
        let target = StreamOrDevice::Stream(s);
        let (has, dev_only, dt, idx) = target.encode();
        assert!(has);
        assert!(!dev_only);
        assert_eq!(dt, s.device.device_type as u8);
        assert_eq!(idx, s.index);
    }
}
