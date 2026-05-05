//! Bridge for MLX IO (load/save: safetensors, gguf, npy + Reader/Writer streams).
//!
//! Map decomposition: shim returns opaque LoadResult types, Rust calls
//! parallel name/value getters and rebuilds HashMap on the safe layer.
//!
//! Save direction: opaque SaveBuilder accumulates entries via add_* calls;
//! single save_*_file/writer call commits.
//!
//! Reader/Writer: opaque MlxReader/MlxWriter wrap shared_ptr<io::Reader/Writer>.
//! B-lite = file + memory backends only; no Rust trait callbacks.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/io.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type MlxReader;
        type MlxWriter;

        // ===== Reader / Writer 工厂 =====
        fn open_file_reader(path: &str) -> Result<UniquePtr<MlxReader>>;
        fn open_memory_reader(data: &[u8]) -> UniquePtr<MlxReader>;
        fn create_file_writer(path: &str) -> Result<UniquePtr<MlxWriter>>;
        fn create_memory_writer() -> UniquePtr<MlxWriter>;
        fn writer_into_bytes(writer: UniquePtr<MlxWriter>) -> Result<Vec<u8>>;

        type SafetensorsLoadResult;
        type SafetensorsSaveBuilder;

        // ===== safetensors =====
        fn load_safetensors_file(path: &str) -> Result<UniquePtr<SafetensorsLoadResult>>;
        fn load_safetensors_reader(
            reader: Pin<&mut MlxReader>,
        ) -> Result<UniquePtr<SafetensorsLoadResult>>;
        fn safetensors_tensor_names(r: &SafetensorsLoadResult) -> Vec<String>;
        fn safetensors_take_tensor_by_name(
            r: Pin<&mut SafetensorsLoadResult>,
            name: &str,
        ) -> Result<UniquePtr<MlxArray>>;
        fn safetensors_metadata_names(r: &SafetensorsLoadResult) -> Vec<String>;
        fn safetensors_metadata_values(r: &SafetensorsLoadResult) -> Vec<String>;

        fn new_safetensors_save_builder() -> UniquePtr<SafetensorsSaveBuilder>;
        fn safetensors_builder_add_tensor(
            b: Pin<&mut SafetensorsSaveBuilder>,
            name: &str,
            array: &MlxArray,
        );
        fn safetensors_builder_add_metadata(
            b: Pin<&mut SafetensorsSaveBuilder>,
            key: &str,
            value: &str,
        );
        fn save_safetensors_file(path: &str, builder: &SafetensorsSaveBuilder) -> Result<()>;
        fn save_safetensors_writer(
            writer: Pin<&mut MlxWriter>,
            builder: &SafetensorsSaveBuilder,
        ) -> Result<()>;

        type GGUFLoadResult;
        type GGUFSaveBuilder;

        // ===== GGUF =====
        fn load_gguf_file(path: &str) -> Result<UniquePtr<GGUFLoadResult>>;
        fn gguf_tensor_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_take_tensor_by_name(
            r: Pin<&mut GGUFLoadResult>,
            name: &str,
        ) -> Result<UniquePtr<MlxArray>>;
        fn gguf_array_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_take_array_meta_by_name(
            r: Pin<&mut GGUFLoadResult>,
            name: &str,
        ) -> Result<UniquePtr<MlxArray>>;
        fn gguf_string_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_meta_values(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_values_packed(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_lengths(r: &GGUFLoadResult) -> Vec<u64>;

        fn new_gguf_save_builder() -> UniquePtr<GGUFSaveBuilder>;
        fn gguf_builder_add_tensor(b: Pin<&mut GGUFSaveBuilder>, name: &str, array: &MlxArray);
        fn gguf_builder_add_array_meta(b: Pin<&mut GGUFSaveBuilder>, key: &str, array: &MlxArray);
        fn gguf_builder_add_string_meta(b: Pin<&mut GGUFSaveBuilder>, key: &str, value: &str);
        fn gguf_builder_begin_string_list_meta(
            b: Pin<&mut GGUFSaveBuilder>,
            key: &str,
        ) -> Result<()>;
        fn gguf_builder_push_string_list_meta(
            b: Pin<&mut GGUFSaveBuilder>,
            value: &str,
        ) -> Result<()>;
        fn gguf_builder_end_string_list_meta(b: Pin<&mut GGUFSaveBuilder>) -> Result<()>;
        fn save_gguf_file(path: &str, builder: &GGUFSaveBuilder) -> Result<()>;

        // ===== npy =====
        fn load_npy_file(path: &str) -> Result<UniquePtr<MlxArray>>;
        fn load_npy_reader(reader: Pin<&mut MlxReader>) -> Result<UniquePtr<MlxArray>>;
        fn save_npy_file(path: &str, array: &MlxArray) -> Result<()>;
        fn save_npy_writer(writer: Pin<&mut MlxWriter>, array: &MlxArray) -> Result<()>;
    }
}
