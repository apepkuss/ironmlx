use mlx::{Array, Dtype, Element};

#[test]
fn u16_round_trip() {
    let data: Vec<u16> = vec![1, 100, 65535, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint16);
    let back: Vec<u16> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u32_round_trip() {
    let data: Vec<u32> = vec![1, 1_000_000, u32::MAX, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint32);
    let back: Vec<u32> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u64_round_trip() {
    let data: Vec<u64> = vec![1, 1_000_000_000_000, u64::MAX, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint64);
    let back: Vec<u64> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u32_item_scalar() {
    let arr = Array::from_slice(&[42_u32], &[]).expect("from_slice");
    assert_eq!(arr.item::<u32>().expect("item"), 42);
}

#[test]
fn dtype_const_for_new_types() {
    assert_eq!(<u16 as Element>::DTYPE, Dtype::Uint16);
    assert_eq!(<u32 as Element>::DTYPE, Dtype::Uint32);
    assert_eq!(<u64 as Element>::DTYPE, Dtype::Uint64);
}

#[test]
fn shape_validation_for_new_types() {
    // Length mismatch should produce ShapeMismatch (Rust-side check)
    let result = Array::from_slice(&[1_u32, 2, 3], &[5]);
    assert!(matches!(result, Err(mlx::Error::ShapeMismatch { .. })));
}
