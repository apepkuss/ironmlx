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

// ===== GGUF =====

/// GGUF metadata value. Mirrors `mlx::core::GGUFMetaData` minus monostate
/// (the empty variant is silently dropped during load).
#[derive(Debug)]
pub enum GGUFMetaData {
    Array(Array),
    String(String),
    StringList(Vec<String>),
}

/// Load tensors + GGUF metadata from a `.gguf` file.
pub fn load_gguf(path: &str) -> Result<(HashMap<String, Array>, HashMap<String, GGUFMetaData>)> {
    let mut result = mlx_sys::io::ffi::load_gguf_file(path).map_err(Error::from)?;
    gguf_decompose(&mut result)
}

fn gguf_decompose(
    result: &mut cxx::UniquePtr<mlx_sys::io::ffi::GGUFLoadResult>,
) -> Result<(HashMap<String, Array>, HashMap<String, GGUFMetaData>)> {
    // tensors
    let tensor_names = mlx_sys::io::ffi::gguf_tensor_names(result);
    let mut tensors: HashMap<String, Array> = HashMap::with_capacity(tensor_names.len());
    for name in tensor_names {
        let array_ptr = mlx_sys::io::ffi::gguf_take_tensor_by_name(result.pin_mut(), &name)
            .map_err(Error::from)?;
        tensors.insert(name, Array::from_inner(array_ptr));
    }

    // metadata: 三类合并
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();

    // array metadata
    let arr_names = mlx_sys::io::ffi::gguf_array_meta_names(result);
    for name in arr_names {
        let array_ptr = mlx_sys::io::ffi::gguf_take_array_meta_by_name(result.pin_mut(), &name)
            .map_err(Error::from)?;
        metadata.insert(name, GGUFMetaData::Array(Array::from_inner(array_ptr)));
    }

    // string metadata
    let str_names = mlx_sys::io::ffi::gguf_string_meta_names(result);
    let str_values = mlx_sys::io::ffi::gguf_string_meta_values(result);
    for (name, value) in str_names.into_iter().zip(str_values) {
        metadata.insert(name, GGUFMetaData::String(value));
    }

    // string list metadata: 解 packed
    let list_names = mlx_sys::io::ffi::gguf_string_list_meta_names(result);
    let packed = mlx_sys::io::ffi::gguf_string_list_meta_values_packed(result);
    let lengths = mlx_sys::io::ffi::gguf_string_list_meta_lengths(result);
    let mut idx: usize = 0;
    for (name, len) in list_names.into_iter().zip(lengths) {
        let len = len as usize;
        let strings: Vec<String> = packed[idx..idx + len].to_vec();
        idx += len;
        metadata.insert(name, GGUFMetaData::StringList(strings));
    }

    Ok((tensors, metadata))
}

/// Save tensors + GGUF metadata to a `.gguf` file.
pub fn save_gguf(
    path: &str,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, GGUFMetaData>,
) -> Result<()> {
    let mut builder = mlx_sys::io::ffi::new_gguf_save_builder();
    for (name, array) in tensors {
        mlx_sys::io::ffi::gguf_builder_add_tensor(builder.pin_mut(), name, array.as_inner());
    }
    for (key, value) in metadata {
        match value {
            GGUFMetaData::Array(a) => {
                mlx_sys::io::ffi::gguf_builder_add_array_meta(builder.pin_mut(), key, a.as_inner())
            }
            GGUFMetaData::String(s) => {
                mlx_sys::io::ffi::gguf_builder_add_string_meta(builder.pin_mut(), key, s)
            }
            GGUFMetaData::StringList(items) => {
                mlx_sys::io::ffi::gguf_builder_begin_string_list_meta(builder.pin_mut(), key)
                    .map_err(Error::from)?;
                for item in items {
                    mlx_sys::io::ffi::gguf_builder_push_string_list_meta(builder.pin_mut(), item)
                        .map_err(Error::from)?;
                }
                mlx_sys::io::ffi::gguf_builder_end_string_list_meta(builder.pin_mut())
                    .map_err(Error::from)?;
            }
        }
    }
    mlx_sys::io::ffi::save_gguf_file(path, &builder).map_err(Error::from)
}
