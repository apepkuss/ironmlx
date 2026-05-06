//! Integration tests for mlx::io — file and stream IO for MLX arrays.

use mlx::io::{Reader, Writer};

#[test]
fn reader_open_file_nonexistent_returns_err() {
    let result = Reader::open_file("/nonexistent/path/should-not-exist.safetensors");
    assert!(result.is_err());
}

#[test]
fn reader_from_bytes_constructs_ok() {
    // 空数据可构造（数据有效性由后续 load 验证）
    let _r = Reader::from_bytes(&[]);
    let _r2 = Reader::from_bytes(&[1, 2, 3]);
}

#[test]
fn writer_memory_into_bytes_empty() {
    let writer = Writer::memory();
    let bytes = writer.into_bytes().expect("memory writer into_bytes");
    assert_eq!(bytes, Vec::<u8>::new());
}

#[test]
fn writer_create_file_invalid_path_returns_err() {
    let result = Writer::create_file("/nonexistent_dir_xyz/should-fail.bin");
    assert!(result.is_err());
}

#[test]
fn writer_file_into_bytes_returns_err() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let writer = Writer::create_file(tmp.path().to_str().unwrap()).expect("create_file");
    let result = writer.into_bytes();
    assert!(result.is_err(), "into_bytes on file writer should err");
}

use mlx::io;
use mlx::Array;
use std::collections::HashMap;

fn make_test_tensors() -> HashMap<String, Array> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "alpha".to_string(),
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[2, 2][..])).expect("alpha"),
    );
    tensors.insert(
        "beta".to_string(),
        Array::try_from((&[10.0_f32, 20.0][..], &[2][..])).expect("beta"),
    );
    tensors
}

fn make_test_metadata() -> HashMap<String, String> {
    let mut meta = HashMap::new();
    meta.insert("model".to_string(), "test-model".to_string());
    meta.insert("version".to_string(), "1.0".to_string());
    meta
}

#[test]
fn safetensors_round_trip_file() {
    // 注：MLX 的 save_safetensors 在路径无 .safetensors 后缀时会自动追加，
    // 因此测试必须使用 .safetensors 结尾路径，否则 load 找不到文件。
    let dir = tempfile::tempdir().expect("tempdir");
    let path_buf = dir.path().join("round_trip.safetensors");
    let path = path_buf.to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata = make_test_metadata();

    io::save_safetensors(path, &tensors, &metadata).expect("save");

    let (loaded_tensors, loaded_meta) = io::load_safetensors(path).expect("load");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert_eq!(loaded_meta, metadata);

    // 数值一致
    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha to_vec");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
    let beta_in: Vec<f32> = tensors["beta"].to_vec().expect("beta to_vec");
    let beta_out: Vec<f32> = loaded_tensors["beta"].to_vec().expect("beta out");
    assert_eq!(beta_in, beta_out);
}

#[test]
fn safetensors_round_trip_memory() {
    let tensors = make_test_tensors();
    let metadata = make_test_metadata();

    let mut writer = io::Writer::memory();
    io::save_safetensors_to_writer(&mut writer, &tensors, &metadata).expect("save to writer");
    let bytes = writer.into_bytes().expect("into_bytes");
    assert!(!bytes.is_empty(), "memory writer should have written bytes");

    let mut reader = io::Reader::from_bytes(&bytes);
    let (loaded_tensors, loaded_meta) =
        io::load_safetensors_from_reader(&mut reader).expect("load from reader");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert_eq!(loaded_meta, metadata);

    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
}

#[test]
fn safetensors_load_nonexistent_file_returns_err() {
    let result = io::load_safetensors("/nonexistent/path/foo.safetensors");
    assert!(result.is_err());
}

#[test]
fn safetensors_empty_metadata_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path_buf = dir.path().join("empty_meta.safetensors");
    let path = path_buf.to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata: HashMap<String, String> = HashMap::new();

    io::save_safetensors(path, &tensors, &metadata).expect("save");
    let (loaded_tensors, loaded_meta) = io::load_safetensors(path).expect("load");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert!(loaded_meta.is_empty());
}

use mlx::io::GGUFMetaData;

#[test]
fn gguf_round_trip_tensor_only() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tensor_only.gguf");
    let path_str = path.to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata: HashMap<String, GGUFMetaData> = HashMap::new();

    io::save_gguf(path_str, &tensors, &metadata).expect("save");
    let (loaded_tensors, loaded_meta) = io::load_gguf(path_str).expect("load");

    assert_eq!(loaded_tensors.len(), tensors.len());
    assert!(loaded_meta.is_empty());
    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
}

#[test]
fn gguf_round_trip_with_string_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("string_meta.gguf");
    let path_str = path.to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    metadata.insert(
        "model".to_string(),
        GGUFMetaData::String("test-model".to_string()),
    );
    metadata.insert(
        "version".to_string(),
        GGUFMetaData::String("1.0".to_string()),
    );

    io::save_gguf(path_str, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path_str).expect("load");

    assert_eq!(loaded_meta.len(), 2);
    match &loaded_meta["model"] {
        GGUFMetaData::String(s) => assert_eq!(s, "test-model"),
        _ => panic!("expected String variant for 'model'"),
    }
    match &loaded_meta["version"] {
        GGUFMetaData::String(s) => assert_eq!(s, "1.0"),
        _ => panic!("expected String variant for 'version'"),
    }
}

#[test]
fn gguf_round_trip_with_string_list_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("string_list_meta.gguf");
    let path_str = path.to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    metadata.insert(
        "tags".to_string(),
        GGUFMetaData::StringList(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    );

    io::save_gguf(path_str, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path_str).expect("load");

    match &loaded_meta["tags"] {
        GGUFMetaData::StringList(list) => {
            assert_eq!(
                list,
                &vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
        _ => panic!("expected StringList variant for 'tags'"),
    }
}

#[test]
fn gguf_round_trip_with_array_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("array_meta.gguf");
    let path_str = path.to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    let scale_array = Array::try_from((&[2.5_f32, 3.5][..], &[2][..])).expect("scale");
    metadata.insert("scale".to_string(), GGUFMetaData::Array(scale_array));

    io::save_gguf(path_str, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path_str).expect("load");

    match &loaded_meta["scale"] {
        GGUFMetaData::Array(arr) => {
            let v: Vec<f32> = arr.to_vec().expect("array meta to_vec");
            assert_eq!(v, vec![2.5_f32, 3.5]);
        }
        _ => panic!("expected Array variant for 'scale'"),
    }
}

#[test]
fn gguf_load_nonexistent_file_returns_err() {
    let result = io::load_gguf("/nonexistent/path/foo.gguf");
    assert!(result.is_err());
}

#[test]
fn npy_round_trip_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("array.npy");
    let path_str = path.to_str().unwrap();
    let array = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("a");

    io::save_npy(path_str, &array).expect("save_npy");
    let loaded = io::load_npy(path_str).expect("load_npy");

    assert_eq!(loaded.shape().as_slice(), &[2, 3]);
    let v: Vec<f32> = loaded.to_vec().expect("to_vec");
    assert_eq!(v, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn npy_round_trip_memory() {
    let array = Array::try_from((&[10.0_f32, 20.0, 30.0][..], &[3][..])).expect("a");

    let mut writer = io::Writer::memory();
    io::save_npy_to_writer(&mut writer, &array).expect("save");
    let bytes = writer.into_bytes().expect("into_bytes");
    assert!(!bytes.is_empty());

    let mut reader = io::Reader::from_bytes(&bytes);
    let loaded = io::load_npy_from_reader(&mut reader).expect("load");

    assert_eq!(loaded.shape().as_slice(), &[3]);
    let v: Vec<f32> = loaded.to_vec().expect("to_vec");
    assert_eq!(v, vec![10.0_f32, 20.0, 30.0]);
}

#[test]
fn npy_load_nonexistent_file_returns_err() {
    let result = io::load_npy("/nonexistent/path/foo.npy");
    assert!(result.is_err());
}

#[test]
fn submodule_path_works() {
    // 验证通过 mlx::io::* 子模块路径访问 P2c 公开 API
    let _r = mlx::io::Reader::from_bytes(&[]);
    let _w = mlx::io::Writer::memory();
    // GGUFMetaData 类型可访问
    let _meta = mlx::io::GGUFMetaData::String("test".to_string());

    // load_safetensors 函数可达（不需要真正成功调用）
    let result = mlx::io::load_safetensors("/nonexistent/foo.safetensors");
    assert!(result.is_err());
}
