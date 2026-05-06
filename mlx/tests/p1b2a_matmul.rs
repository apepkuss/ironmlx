use mlx::{Array, Error};

#[test]
fn matmul_2d() {
    // [2, 3] @ [3, 4] → [2, 4]
    // a = [[1, 2, 3], [4, 5, 6]], b = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    // a @ b = [[1, 2, 3, 0], [4, 5, 6, 0]]
    let a =
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let b = Array::try_from((
        &[
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ][..],
        &[3, 4][..],
    ))
    .expect("try_from");
    let c = a.matmul(&b).expect("matmul");
    assert_eq!(c.shape().as_slice(), &[2, 4]);
    assert_eq!(
        c.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]
    );
}

#[test]
fn matmul_3d_batched() {
    // [B=2, S=3, D=4] @ [B=2, D=4, M=5] → [B=2, S=3, M=5]
    let a = Array::try_from((&[0.0_f32; 24][..], &[2, 3, 4][..])).expect("try_from");
    let b = Array::try_from((&[0.0_f32; 40][..], &[2, 4, 5][..])).expect("try_from");
    let c = a.matmul(&b).expect("matmul");
    assert_eq!(c.shape().as_slice(), &[2, 3, 5]);
}

#[test]
fn matmul_attention_shape() {
    // [B=2, H=4, S=8, D=16] @ [B=2, H=4, D=16, S=8] → [B=2, H=4, S=8, S=8]
    let q = Array::try_from((&[0.0_f32; 1024][..], &[2, 4, 8, 16][..])).expect("try_from");
    let k = Array::try_from((&[0.0_f32; 1024][..], &[2, 4, 16, 8][..])).expect("try_from");
    let scores = q.matmul(&k).expect("matmul");
    assert_eq!(scores.shape().as_slice(), &[2, 4, 8, 8]);
}

#[test]
fn matmul_using_t_for_attention() {
    // Q @ K.t() in attention pattern.
    // For 4D: [B, H, S, D] @ [B, H, S, D].t() = [B, H, S, D] @ reversed = wrong shape.
    // .t() reverses ALL dims, so for proper attention we need transpose_axes(&[0, 1, 3, 2])
    let q = Array::try_from((&[0.0_f32; 1024][..], &[2, 4, 8, 16][..])).expect("try_from");
    let k = Array::try_from((&[0.0_f32; 1024][..], &[2, 4, 8, 16][..])).expect("try_from");
    let kt = k.t().expect("k.t()");
    assert_eq!(kt.shape().as_slice(), &[16, 8, 4, 2]); // .t() reverses ALL dims
                                                       // For a proper attention pattern we'd need transpose_axes(&[0, 1, 3, 2])
    let kt_proper = k.transpose_axes(&[0, 1, 3, 2]).expect("transpose_axes");
    assert_eq!(kt_proper.shape().as_slice(), &[2, 4, 16, 8]);
    let scores = q.matmul(&kt_proper).expect("matmul");
    assert_eq!(scores.shape().as_slice(), &[2, 4, 8, 8]);
}

#[test]
fn matmul_inner_dim_mismatch_errors() {
    let a = Array::try_from((&[0.0_f32; 6][..], &[2, 3][..])).expect("try_from");
    let b = Array::try_from((&[0.0_f32; 8][..], &[4, 2][..])).expect("try_from"); // inner dim 3 != 4
    let result = a.matmul(&b);
    assert!(matches!(result, Err(Error::Mlx(_))));
}
