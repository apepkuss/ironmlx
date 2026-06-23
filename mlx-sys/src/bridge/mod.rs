//! Each MLX C++ subsystem gets its own bridge module.
//!
//! **Rule (P1a onward):** Any shim function that can throw a C++ exception
//! MUST be declared `Result<T>` in its `#[cxx::bridge]` block. cxx wraps
//! the throw as `cxx::Exception`, which our `From<cxx::Exception> for Error`
//! impl converts to `Error::Mlx(String)`. Without this, a thrown exception
//! propagates through a non-`Result` cxx function as `std::terminate` —
//! the process aborts instead of yielding a recoverable Rust error.
//!
//! Pure getters (no throw paths) may stay as plain return types.

pub mod array;
pub mod compile;
pub mod conv;
pub mod fast;
pub mod io;
pub mod memory;
pub mod metal;
pub mod quantization;
pub mod random;
pub mod stream;
pub mod transforms;
