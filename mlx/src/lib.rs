//! Safe Rust bindings to Apple MLX.
//!
//! # Quickstart
//!
//! ```no_run
//! use mlx::{Array, Dtype};
//!
//! # fn main() -> mlx::Result<()> {
//! let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])?;
//! let v: Vec<f32> = a.to_vec()?;
//! assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
//! # Ok(())
//! # }
//! ```
//!
//! # Threading
//!
//! [`Array`] is `Send` but not `Sync`. To share an array between threads,
//! clone it (cheap MLX refcount). See the README for details.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod array;
mod broadcast;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;
mod stream;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use device::{default_device, device_count, is_available, set_default_device, Device, DeviceType};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
pub use stream::{clear_streams, default_stream, get_streams, new_stream, set_default_stream, Stream};
