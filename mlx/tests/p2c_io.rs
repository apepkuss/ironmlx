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
