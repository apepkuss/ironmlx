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
pub mod compile;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;
mod shape;
mod stream;
pub mod transforms;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use compile::{
    compile, disable_compile, enable_compile, set_compile_mode, CompileMode, CompiledFn,
};
pub use device::{
    default_device, device_count, is_available, set_default_device, Device, DeviceType,
};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
pub use shape::{IntoShape, Shape};
pub use stream::{
    clear_streams, default_stream, get_streams, new_stream, set_default_stream, Stream,
};
pub use transforms::{async_eval, synchronize, synchronize_stream};

pub mod fast;
pub use fast::{layer_norm, rms_norm, rope, rope_with_array_offset, scaled_dot_product_attention};

pub mod io;
pub use io::{
    load_gguf, load_npy, load_npy_from_reader, load_safetensors, load_safetensors_from_reader,
    save_gguf, save_npy, save_npy_to_writer, save_safetensors, save_safetensors_to_writer,
    GGUFMetaData, Reader, Writer,
};

pub mod quantization;
pub use quantization::{
    dequantize, from_fp8, gather_qmm, qqmm, quantize, quantized_matmul, to_fp8,
};

pub mod random;
pub use random::{
    bernoulli, bernoulli_default, bits, categorical, categorical_n, categorical_shaped, gumbel,
    key, laplace, multivariate_normal, normal, permutation, permutation_arange, randint, seed,
    split, split_n, truncated_normal, truncated_normal_default, uniform, uniform_default,
};
