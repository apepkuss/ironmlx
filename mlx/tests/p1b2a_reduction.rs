use mlx::{ops, All, Array, Dtype};

#[test]
fn sum_all_axes_returns_scalar() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let s = ops::sum(&a, All, false).expect("sum_all");
    assert_eq!(s.size(), 1);
    assert_eq!(s.shape().as_slice(), &[] as &[i32]);
    assert!((s.item::<f32>().expect("item") - 10.0).abs() < 1e-6);
}

#[test]
fn sum_single_axis_negative_index() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = ops::sum(&a, -1, false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2]);
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![6.0, 15.0]);
}

#[test]
fn sum_single_axis_keepdim() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = ops::sum(&a, -1, true).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2, 1]);
}

#[test]
fn sum_multi_axis_slice_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, &[0, 2][..], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
    let v = s.to_vec::<f32>().expect("to_vec");
    assert_eq!(v, vec![8.0_f32, 8.0, 8.0]);
}

#[test]
fn sum_multi_axis_vec_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, vec![0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_multi_axis_array_literal_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, [0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_method_matches_free_fn() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let by_method = a.sum(-1, false).expect("method");
    let by_freefn = ops::sum(&a, -1, false).expect("free fn");
    assert_eq!(
        by_method.to_vec::<f32>().expect("method to_vec"),
        by_freefn.to_vec::<f32>().expect("freefn to_vec")
    );
}

#[test]
fn sum_dtype_preserved_for_integers() {
    let a = Array::from_slice(&[1_i32, 2, 3], &[3]).expect("from_slice");
    let s = ops::sum(&a, All, false).expect("sum");
    assert_eq!(s.dtype(), Dtype::Int32);
    assert_eq!(s.item::<i32>().expect("item"), 6);
}
