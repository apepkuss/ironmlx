//! Safe Rust bindings to Apple MLX.
//!
//! # Quickstart
//!
//! ```no_run
//! use mlx::{Array, Dtype, Result};
//! use mlx::random;
//!
//! # fn main() -> Result<()> {
//! let x = random::uniform().shape((2, 3)).sample()?;
//! let y = random::normal().shape((2, 3)).sample()?;
//! let z = &x + &y;
//! # Ok(())
//! # }
//! ```
//!
//! # Threading
//!
//! [`Array`] is `Send` but not `Sync`. To share an array between threads,
//! clone it (cheap MLX refcount).

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod array;
mod broadcast;
mod device;
mod dtype;
mod element;
mod error;
mod ops_impl;
mod shape;
mod stream;

pub mod compile;
pub mod fast;
pub mod io;
pub mod ops;
pub mod quantization;
pub mod random;
pub mod transforms;

pub use array::Array;
pub use device::{
    default_device, device_count, is_available, set_default_device, Device, DeviceType,
};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use shape::{IntoShape, Shape};
pub use stream::{
    clear_streams, default_stream, get_streams, new_stream, set_default_stream, Stream,
    StreamOrDevice,
};
