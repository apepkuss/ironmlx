pub mod array;
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
