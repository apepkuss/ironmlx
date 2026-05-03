use mlx::{Array, Dtype, Error};

#[test]
fn reshape_explicit_shape() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).expect("from_slice");
    let r = a.reshape(&[2, 3]).expect("reshape");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_minus_one_inferred_at_end() {
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let r = a.reshape(&[2, -1]).expect("reshape inferred");
    assert_eq!(r.shape().as_slice(), &[2, 12]);
}

#[test]
fn reshape_minus_one_inferred_in_middle() {
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let r = a.reshape(&[2, -1, 4]).expect("reshape inferred middle");
    assert_eq!(r.shape().as_slice(), &[2, 3, 4]);
}

#[test]
fn reshape_no_minus_one() {
    let a = Array::from_slice(&[0.0_f32; 6], &[6]).expect("from_slice");
    let r = a.reshape(&[2, 3]).expect("reshape");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
}

#[test]
fn reshape_multiple_minus_ones_errors() {
    let a = Array::from_slice(&[0.0_f32; 24], &[24]).expect("from_slice");
    let result = a.reshape(&[-1, -1, 4]);
    match result {
        Err(Error::Mlx(msg)) => assert!(msg.contains("at most one -1"), "msg: {msg}"),
        other => panic!("expected Error::Mlx, got {other:?}"),
    }
}

#[test]
fn reshape_indivisible_minus_one_errors() {
    // 24 elements / 5 = not integer
    let a = Array::from_slice(&[0.0_f32; 24], &[24]).expect("from_slice");
    let result = a.reshape(&[5, -1]);
    match result {
        Err(Error::Mlx(msg)) => assert!(msg.contains("not divisible") || msg.contains("infer"), "msg: {msg}"),
        other => panic!("expected Error::Mlx, got {other:?}"),
    }
}

#[test]
fn reshape_total_size_mismatch_propagates_from_mlx() {
    let a = Array::from_slice(&[0.0_f32; 6], &[6]).expect("from_slice");
    // Asking for 8 elements when we have 6 → MLX rejects
    let result = a.reshape(&[2, 4]);
    assert!(matches!(result, Err(Error::Mlx(_))));
    let _ = Dtype::Float32;  // silence unused import
}

#[test]
fn transpose_2d_swaps_rows_cols() {
    // [[1, 2, 3], [4, 5, 6]] (2x3) transposed → [[1, 4], [2, 5], [3, 6]] (3x2)
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let t = a.transpose().expect("transpose");
    assert_eq!(t.shape().as_slice(), &[3, 2]);
    assert_eq!(t.to_vec::<f32>().expect("to_vec"), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn t_method_alias_for_transpose() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let t1 = a.t().expect("t");
    let t2 = a.transpose().expect("transpose");
    assert_eq!(
        t1.to_vec::<f32>().expect("to_vec"),
        t2.to_vec::<f32>().expect("to_vec")
    );
}

#[test]
fn transpose_axes_permute() {
    // [2, 3, 4] permuted by [2, 0, 1] → [4, 2, 3]
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let t = a.transpose_axes(&[2, 0, 1]).expect("transpose_axes");
    assert_eq!(t.shape().as_slice(), &[4, 2, 3]);
}

#[test]
fn broadcast_to_expands_singleton_dim() {
    // [3] broadcast to [2, 3] should replicate the row twice
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let b = a.broadcast_to(&[2, 3]).expect("broadcast_to");
    assert_eq!(b.shape().as_slice(), &[2, 3]);
    assert_eq!(b.to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn broadcast_to_incompatible_shape_errors() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let result = a.broadcast_to(&[2, 4]);
    assert!(matches!(result, Err(Error::Mlx(_))));
}
