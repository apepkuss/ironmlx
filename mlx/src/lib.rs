pub mod array;
mod array_ops;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod stream;

pub use array::Array;
pub use device::{Device, DeviceType};
pub use dtype::Dtype;
pub use error::{Error, Result};
pub use stream::Stream;

use std::sync::OnceLock;

static INIT: OnceLock<()> = OnceLock::new();

/// Initialise the MLX runtime (installs error handler, etc.).
/// Called automatically on first use; safe to call multiple times.
pub fn init() {
    INIT.get_or_init(|| {
        error::install_error_handler();
    });
}

/// Return the default stream for the given device type.
///
/// Convenience wrapper around `Stream::default_stream(&Device::new(kind))`.
pub fn default_stream(kind: DeviceType) -> Stream {
    Stream::default_stream(&Device::new(kind))
}
