use mlx::{Array, Dtype, Error};

#[test]
fn from_slice_f32_round_trip() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_slice(&data, &[2, 3]).expect("from_slice");
    assert_eq!(arr.shape().as_slice(), &[2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_slice_i32_round_trip() {
    let data = vec![10_i32, 20, 30];
    let arr = Array::from_slice(&data, &[3]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Int32);
    assert_eq!(arr.size(), 3);
}

#[test]
fn from_slice_f16_round_trip() {
    let data = vec![half::f16::from_f32(1.5), half::f16::from_f32(2.5)];
    let arr = Array::from_slice(&data, &[2]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Float16);
    assert_eq!(arr.size(), 2);
}

#[test]
fn from_slice_bool_round_trip() {
    let data = vec![true, false, true, false];
    let arr = Array::from_slice(&data, &[2, 2]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Bool);
    assert_eq!(arr.size(), 4);
}

#[test]
fn from_slice_shape_mismatch_returns_err() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let result = Array::from_slice(&data, &[2, 3]);
    match result {
        Err(Error::ShapeMismatch { expected, actual }) => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn from_slice_empty_shape_is_scalar() {
    let data = vec![42.0_f32];
    let arr = Array::from_slice(&data, &[]).expect("from_slice scalar");
    assert_eq!(arr.size(), 1);
    assert_eq!(arr.ndim(), 0);
}

#[test]
fn item_f32_round_trip() {
    let arr = Array::from_slice(&[42.0_f32], &[]).expect("from_slice");
    let v = arr.item::<f32>().expect("item");
    assert_eq!(v, 42.0);
}

#[test]
fn item_dtype_mismatch_returns_err() {
    let arr = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let result = arr.item::<i32>();
    match result {
        Err(Error::DtypeMismatch { expected, actual }) => {
            assert_eq!(expected, Dtype::Int32);
            assert_eq!(actual, Dtype::Float32);
        }
        other => panic!("expected DtypeMismatch, got {other:?}"),
    }
}

#[test]
fn item_non_scalar_returns_err() {
    let arr = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let result = arr.item::<f32>();
    assert!(matches!(result, Err(Error::Mlx(_))));
}
