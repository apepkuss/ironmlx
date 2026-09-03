use mlx::ops::{self, All};
use mlx::{Array, Dtype, Error};

#[test]
fn sum_all_axes_returns_scalar() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[2, 2][..])).expect("try_from");
    let s = ops::sum(&a, All, false).expect("sum_all");
    assert_eq!(s.size(), 1);
    assert_eq!(s.shape().as_slice(), &[] as &[i32]);
    assert!((s.item::<f32>().expect("item") - 10.0).abs() < 1e-6);
}

#[test]
fn sum_single_axis_negative_index() {
    let a =
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let s = ops::sum(&a, -1, false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2]);
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![6.0, 15.0]);
}

#[test]
fn sum_single_axis_keepdim() {
    let a =
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let s = ops::sum(&a, -1, true).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2, 1]);
}

#[test]
fn sum_multi_axis_slice_form() {
    let a = Array::try_from((&[1.0_f32; 24][..], &[2, 3, 4][..])).expect("try_from");
    let s = ops::sum(&a, &[0, 2][..], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
    let v = s.to_vec::<f32>().expect("to_vec");
    assert_eq!(v, vec![8.0_f32, 8.0, 8.0]);
}

#[test]
fn sum_multi_axis_vec_form() {
    let a = Array::try_from((&[1.0_f32; 24][..], &[2, 3, 4][..])).expect("try_from");
    let s = ops::sum(&a, vec![0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_multi_axis_array_literal_form() {
    let a = Array::try_from((&[1.0_f32; 24][..], &[2, 3, 4][..])).expect("try_from");
    let s = ops::sum(&a, [0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_method_matches_free_fn() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[2, 2][..])).expect("try_from");
    let by_method = a.sum(-1, false).expect("method");
    let by_freefn = ops::sum(&a, -1, false).expect("free fn");
    assert_eq!(
        by_method.to_vec::<f32>().expect("method to_vec"),
        by_freefn.to_vec::<f32>().expect("freefn to_vec")
    );
}

#[test]
fn sum_dtype_preserved_for_integers() {
    let a = Array::try_from((&[1_i32, 2, 3][..], &[3][..])).expect("try_from");
    let s = ops::sum(&a, All, false).expect("sum");
    assert_eq!(s.dtype(), Dtype::Int32);
    assert_eq!(s.item::<i32>().expect("item"), 6);
}

#[test]
fn mean_basic() {
    let a = Array::try_from((&[2.0_f32, 4.0, 6.0, 8.0][..], &[2, 2][..])).expect("try_from");
    let m = ops::mean(&a, All, false).expect("mean");
    assert!((m.item::<f32>().expect("item") - 5.0).abs() < 1e-6);

    let m2 = ops::mean(&a, -1, false).expect("mean axis");
    assert_eq!(m2.to_vec::<f32>().expect("to_vec"), vec![3.0_f32, 7.0]);
}

#[test]
fn max_basic() {
    let a = Array::try_from((&[1.0_f32, 5.0, 3.0, 2.0][..], &[2, 2][..])).expect("try_from");
    assert_eq!(
        ops::max(&a, All, false)
            .expect("max")
            .item::<f32>()
            .expect("item"),
        5.0
    );

    let m = ops::max(&a, -1, false).expect("max axis");
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![5.0_f32, 3.0]);
}

#[test]
fn min_basic() {
    let a = Array::try_from((&[1.0_f32, 5.0, 3.0, 2.0][..], &[2, 2][..])).expect("try_from");
    assert_eq!(
        ops::min(&a, All, false)
            .expect("min")
            .item::<f32>()
            .expect("item"),
        1.0
    );

    let m = ops::min(&a, -1, false).expect("min axis");
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![1.0_f32, 2.0]);
}

#[test]
fn argmax_basic() {
    // [[1, 5, 3], [2, 4, 6]] → argmax(-1) = [1, 2]
    let a =
        Array::try_from((&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let am = ops::argmax(&a, -1, false).expect("argmax");
    assert_eq!(am.dtype(), Dtype::Uint32);
    assert_eq!(am.shape().as_slice(), &[2_i32]);
    // P1b2b: u32 is now an Element, value assertion enabled
    assert_eq!(am.to_vec::<u32>().expect("to_vec"), vec![1_u32, 2]);
}

#[test]
fn argmax_all_returns_flat_index() {
    // The single max in [1, 5, 3, 2, 4, 6] is at flat index 5
    let a =
        Array::try_from((&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let am = ops::argmax(&a, All, false).expect("argmax all");
    assert_eq!(am.dtype(), Dtype::Uint32);
    assert_eq!(am.size(), 1);
    assert_eq!(am.shape().as_slice(), &[] as &[i32]);
    // P1b2b: u32 is now an Element, value assertion enabled
    assert_eq!(am.item::<u32>().expect("item"), 5);
}

#[test]
fn argmax_multi_axis_rejected() {
    // MLX doesn't support multi-axis argmax; Rust returns a structured error.
    let a = Array::try_from((&[1.0_f32; 24][..], &[2, 3, 4][..])).expect("try_from");
    let result = ops::argmax(&a, &[0, 1][..], false);
    match result {
        Err(Error::Mlx(msg)) => {
            assert!(msg.contains("does not support multi-axis"), "msg: {msg}");
        }
        other => panic!("expected Error::Mlx, got {other:?}"),
    }
}

#[test]
fn sum_empty_axes_slice_is_no_op() {
    // Empty `&[]` passes through to MLX's multi-axes sum with empty list.
    // MLX's behaviour: empty axes is a no-op, returning the original shape.
    // This test pins the actual MLX semantics so future MLX changes are caught.
    let a =
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let result = ops::sum(&a, &[][..], false);
    match result {
        Ok(s) => {
            // MLX returns the original array unchanged for empty-axes reduction
            assert_eq!(s.shape().as_slice(), &[2, 3]);
            assert_eq!(
                s.to_vec::<f32>().expect("to_vec"),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
        }
        Err(Error::Mlx(msg)) => {
            // Acceptable too — pin whatever MLX does.
            panic!("MLX rejected empty-axes sum: {msg}. Update test if behaviour changed.");
        }
        other => panic!("unexpected result type: {other:?}"),
    }
}

#[test]
fn reduction_methods_match_free_fns() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[2, 2][..])).expect("try_from");
    assert_eq!(
        a.mean(All, false)
            .expect("mean")
            .item::<f32>()
            .expect("item"),
        ops::mean(&a, All, false)
            .expect("mean")
            .item::<f32>()
            .expect("item"),
    );
    assert_eq!(
        a.max(All, false).expect("max").item::<f32>().expect("item"),
        ops::max(&a, All, false)
            .expect("max")
            .item::<f32>()
            .expect("item"),
    );
    assert_eq!(
        a.min(All, false).expect("min").item::<f32>().expect("item"),
        ops::min(&a, All, false)
            .expect("min")
            .item::<f32>()
            .expect("item"),
    );
}
