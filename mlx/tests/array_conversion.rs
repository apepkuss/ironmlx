use mlx::{Array, Dtype, Error};

#[test]
fn from_slice_f32_round_trip() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::try_from((&data[..], &[2, 3][..])).expect("try_from");
    assert_eq!(arr.shape().as_slice(), &[2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_slice_i32_round_trip() {
    let data = vec![10_i32, 20, 30];
    let arr = Array::try_from((&data[..], &[3][..])).expect("try_from");
    assert_eq!(arr.dtype(), Dtype::Int32);
    assert_eq!(arr.size(), 3);
}

#[test]
fn from_slice_f16_round_trip() {
    let data = vec![half::f16::from_f32(1.5), half::f16::from_f32(2.5)];
    let arr = Array::try_from((&data[..], &[2][..])).expect("try_from");
    assert_eq!(arr.dtype(), Dtype::Float16);
    assert_eq!(arr.size(), 2);
}

#[test]
fn from_slice_bool_round_trip() {
    let data = vec![true, false, true, false];
    let arr = Array::try_from((&data[..], &[2, 2][..])).expect("try_from");
    assert_eq!(arr.dtype(), Dtype::Bool);
    assert_eq!(arr.size(), 4);
}

#[test]
fn from_slice_shape_mismatch_returns_err() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let result = Array::try_from((&data[..], &[2, 3][..]));
    match result {
        Err(Error::ShapeMismatch { expected, actual }) => {
            assert_eq!(expected.as_slice(), &[2, 3]);
            assert_eq!(actual.as_slice(), &[3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn from_slice_empty_shape_is_scalar() {
    let data = vec![42.0_f32];
    let arr = Array::try_from((&data[..], &[][..])).expect("from_slice scalar");
    assert_eq!(arr.size(), 1);
    assert_eq!(arr.ndim(), 0);
}

#[test]
fn item_f32_round_trip() {
    let arr = Array::try_from((&[42.0_f32][..], &[][..])).expect("try_from");
    let v = arr.item::<f32>().expect("item");
    assert_eq!(v, 42.0);
}

#[test]
fn item_dtype_mismatch_returns_err() {
    let arr = Array::try_from((&[1.0_f32][..], &[][..])).expect("try_from");
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
    let arr = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("try_from");
    let result = arr.item::<f32>();
    assert!(matches!(result, Err(Error::Mlx(_))));
}

#[test]
fn to_vec_f32_round_trip() {
    let original = vec![1.0_f32, 2.0, 3.0, 4.0];
    let arr = Array::try_from((&original[..], &[2, 2][..])).expect("try_from");
    let read_back = arr.to_vec::<f32>().expect("to_vec");
    assert_eq!(read_back, original);
}

#[test]
fn to_vec_implicit_eval() {
    // Lazy zeros — should NOT need explicit eval before to_vec.
    let arr = Array::zeros(&[3], Dtype::Float32).expect("zeros");
    let v = arr.to_vec::<f32>().expect("to_vec triggers eval");
    assert_eq!(v, vec![0.0_f32, 0.0, 0.0]);
}

#[test]
fn to_vec_f16_bit_pattern_preserved() {
    // Specific bit patterns (NaN-ish, denormal) round-trip exactly.
    let original: Vec<half::f16> = vec![
        half::f16::from_f32(1.5),
        half::f16::from_f32(-2.25),
        half::f16::from_bits(0x7C01), // signaling NaN-ish bit pattern
        half::f16::from_bits(0x0001), // denormal
    ];
    let arr = Array::try_from((&original[..], &[4][..])).expect("try_from");
    let read_back = arr.to_vec::<half::f16>().expect("to_vec");
    for (i, (a, b)) in original.iter().zip(read_back.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit pattern mismatch at index {i}"
        );
    }
}

#[test]
fn to_vec_dtype_mismatch_returns_err() {
    let arr = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("try_from");
    let result = arr.to_vec::<i32>();
    match result {
        Err(Error::DtypeMismatch { expected, actual }) => {
            assert_eq!(expected, Dtype::Int32);
            assert_eq!(actual, Dtype::Float32);
        }
        other => panic!("expected DtypeMismatch, got {other:?}"),
    }
}

#[test]
fn to_vec_bool_round_trip() {
    let original = vec![true, false, true];
    let arr = Array::try_from((&original[..], &[3][..])).expect("try_from");
    let read_back = arr.to_vec::<bool>().expect("to_vec");
    assert_eq!(read_back, original);
}

#[test]
fn item_implicit_eval_on_lazy_scalar() {
    // Regression: `mlx::core::array::item<T>() const` throws on lazy arrays.
    // Per spec A8, item<T> must implicitly eval (same contract as to_vec).
    let arr = Array::zeros(&[], Dtype::Float32).expect("zeros");
    let v = arr
        .item::<f32>()
        .expect("item must trigger implicit eval on lazy");
    assert_eq!(v, 0.0);
}

#[test]
fn from_slice_negative_dim_returns_err() {
    // Regression: shape elements like -1 (sometimes used as a placeholder
    // semantically) would wrap to usize::MAX in the size product and either
    // panic in debug or compute a wrong expected size in release.
    let data = vec![1.0_f32, 2.0, 3.0];
    let result = Array::try_from((&data[..], &[-1, 3][..]));
    match result {
        Err(Error::Mlx(msg)) => assert!(msg.contains("negative dimension"), "msg: {msg}"),
        other => panic!("expected Error::Mlx with 'negative dimension', got {other:?}"),
    }
}

#[test]
fn raw_snapshot_preserves_bf16_float_and_integer_bits_without_borrowing() {
    for (dtype, bytes) in [
        (Dtype::Bfloat16, vec![1, 0, 1, 127, 128, 255, 0, 128]),
        (Dtype::Float32, vec![1, 0, 128, 127, 1, 0, 0, 128]),
        (Dtype::Uint32, vec![255, 255, 255, 255, 0, 0, 0, 0]),
    ] {
        let mut owned = bytes.clone();
        let a = Array::from_bytes(
            &owned,
            &[i32::try_from(bytes.len() / dtype.byte_size()).unwrap()],
            dtype,
        )
        .unwrap();
        owned.fill(0); // constructor must own its storage
        a.eval().unwrap();
        let mut read = vec![0; bytes.len()];
        for (index, chunk) in read.chunks_mut(2).enumerate() {
            a.read_bytes_into(index * 2, chunk).unwrap();
        }
        assert_eq!(read, bytes);
        assert!(a.read_bytes_into(bytes.len(), &mut [0]).is_err());
    }
    assert!(Array::from_bytes(&[2], &[], Dtype::Bool).is_err());
    assert_eq!(
        Array::from_bytes(&[0, 1], &[2], Dtype::Bool)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [false, true]
    );
    let empty = Array::from_bytes(&[], &[0, 3], Dtype::Bfloat16).unwrap();
    empty.eval().unwrap();
    assert_eq!(empty.shape().as_slice(), &[0, 3]);
    assert!(Array::from_bytes(&[0; 3], &[1], Dtype::Float32).is_err());
    assert!(Array::from_bytes(&[], &[-1], Dtype::Float32).is_err());
    let matrix = Array::try_from((&[1_i32, 2, 3, 4, 5, 6][..], &[2, 3][..])).unwrap();
    let offset_row = matrix.slice(&[1, 0][..], &[2, 3][..]).unwrap();
    offset_row.eval().unwrap();
    let mut row_bytes = [0; 12];
    offset_row.read_bytes_into(0, &mut row_bytes).unwrap();
    assert_eq!(
        Array::from_bytes(&row_bytes, &[3], Dtype::Int32)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [4, 5, 6]
    );
    let transpose = matrix.transpose_axes([1, 0]).unwrap();
    transpose.eval().unwrap();
    assert!(transpose.read_bytes_into(0, &mut [0; 24]).is_err());
    let contiguous = transpose.contiguous(false).unwrap();
    contiguous.eval().unwrap();
    let mut data = [0; 24];
    contiguous.read_bytes_into(0, &mut data).unwrap();
    let restored = Array::from_bytes(&data, &[3, 2], Dtype::Int32).unwrap();
    assert_eq!(restored.to_vec::<i32>().unwrap(), [1, 4, 2, 5, 3, 6]);
}
