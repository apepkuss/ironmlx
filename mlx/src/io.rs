//! File and stream IO for MLX arrays.
//!
//! - safetensors: tensor + string metadata; file path or Reader/Writer
//! - gguf: tensor + variant metadata; file path only (upstream limitation)
//! - npy: single array; file path or Reader/Writer
//!
//! Reader / Writer are opaque handles wrapping MLX io::Reader/Writer.
//! Backends: file path + in-memory (B-lite). No Rust-implemented IO callbacks.

use std::pin::Pin;

use crate::{Error, Result};

/// Opaque IO reader handle. Backed by file (`open_file`) or memory (`from_bytes`).
pub struct Reader(cxx::UniquePtr<mlx_sys::io::ffi::MlxReader>);

/// Opaque IO writer handle. Backed by file (`create_file`) or memory (`memory`).
/// Memory writers can be drained via [`Writer::into_bytes`].
pub struct Writer(cxx::UniquePtr<mlx_sys::io::ffi::MlxWriter>);

impl Reader {
    /// Open a file for reading (uses MLX's parallel file reader internally).
    pub fn open_file(path: &str) -> Result<Self> {
        let inner = mlx_sys::io::ffi::open_file_reader(path).map_err(Error::from)?;
        Ok(Reader(inner))
    }

    /// Construct an in-memory reader from a byte slice (data is copied).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Reader(mlx_sys::io::ffi::open_memory_reader(bytes))
    }

    #[allow(dead_code)] // Will be used by load_*_from_reader in Tasks 2/4
    pub(crate) fn pin_mut(&mut self) -> Pin<&mut mlx_sys::io::ffi::MlxReader> {
        self.0.pin_mut()
    }
}

impl Writer {
    /// Open a file for writing (truncates if exists).
    pub fn create_file(path: &str) -> Result<Self> {
        let inner = mlx_sys::io::ffi::create_file_writer(path).map_err(Error::from)?;
        Ok(Writer(inner))
    }

    /// Construct an in-memory writer. Drain via [`Writer::into_bytes`] after writes.
    pub fn memory() -> Self {
        Writer(mlx_sys::io::ffi::create_memory_writer())
    }

    /// Drain the in-memory buffer. Returns `Err` if this is a file writer.
    /// Consumes the writer (memory buffer is moved out).
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        mlx_sys::io::ffi::writer_into_bytes(self.0).map_err(Error::from)
    }

    #[allow(dead_code)] // Will be used by save_*_to_writer in Tasks 2/4
    pub(crate) fn pin_mut(&mut self) -> Pin<&mut mlx_sys::io::ffi::MlxWriter> {
        self.0.pin_mut()
    }
}
