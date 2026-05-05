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

    pub(crate) fn pin_mut(&mut self) -> Pin<&mut mlx_sys::io::ffi::MlxWriter> {
        self.0.pin_mut()
    }
}

use std::collections::HashMap;

use crate::Array;

// ===== safetensors =====

/// Load tensors + string metadata from a `.safetensors` file.
pub fn load_safetensors(path: &str) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let mut result = mlx_sys::io::ffi::load_safetensors_file(path).map_err(Error::from)?;
    safetensors_decompose(&mut result)
}

/// Load tensors + string metadata from a Reader.
pub fn load_safetensors_from_reader(
    reader: &mut Reader,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let mut result =
        mlx_sys::io::ffi::load_safetensors_reader(reader.pin_mut()).map_err(Error::from)?;
    safetensors_decompose(&mut result)
}

fn safetensors_decompose(
    result: &mut cxx::UniquePtr<mlx_sys::io::ffi::SafetensorsLoadResult>,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let names = mlx_sys::io::ffi::safetensors_tensor_names(result);
    let mut tensors: HashMap<String, Array> = HashMap::with_capacity(names.len());
    for name in names {
        let array_ptr = mlx_sys::io::ffi::safetensors_take_tensor_by_name(result.pin_mut(), &name)
            .map_err(Error::from)?;
        tensors.insert(name, Array::from_inner(array_ptr));
    }
    let meta_names = mlx_sys::io::ffi::safetensors_metadata_names(result);
    let meta_values = mlx_sys::io::ffi::safetensors_metadata_values(result);
    let metadata: HashMap<String, String> = meta_names.into_iter().zip(meta_values).collect();
    Ok((tensors, metadata))
}

/// Save tensors + metadata to a `.safetensors` file.
pub fn save_safetensors(
    path: &str,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> Result<()> {
    let builder = build_safetensors_builder(tensors, metadata);
    mlx_sys::io::ffi::save_safetensors_file(path, &builder).map_err(Error::from)
}

/// Save tensors + metadata to a Writer.
pub fn save_safetensors_to_writer(
    writer: &mut Writer,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> Result<()> {
    let builder = build_safetensors_builder(tensors, metadata);
    mlx_sys::io::ffi::save_safetensors_writer(writer.pin_mut(), &builder).map_err(Error::from)
}

fn build_safetensors_builder(
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> cxx::UniquePtr<mlx_sys::io::ffi::SafetensorsSaveBuilder> {
    let mut builder = mlx_sys::io::ffi::new_safetensors_save_builder();
    for (name, array) in tensors {
        mlx_sys::io::ffi::safetensors_builder_add_tensor(builder.pin_mut(), name, array.as_inner());
    }
    for (key, value) in metadata {
        mlx_sys::io::ffi::safetensors_builder_add_metadata(builder.pin_mut(), key, value);
    }
    builder
}
