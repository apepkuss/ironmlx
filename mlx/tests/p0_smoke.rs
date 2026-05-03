use mlx::{Array, Dtype};

#[test]
fn p0_end_to_end() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32);
    assert_eq!(arr.shape(), vec![2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.size(), 6);
    arr.eval().expect("eval should succeed");
}

#[test]
fn empty_shape_is_scalar() {
    let arr = Array::zeros(&[], Dtype::Int32);
    assert_eq!(arr.shape(), Vec::<i32>::new());
    assert_eq!(arr.ndim(), 0);
    assert_eq!(arr.size(), 1);
}
