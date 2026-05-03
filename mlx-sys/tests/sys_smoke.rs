use mlx_sys::array::ffi;

// Mirror of mlx::core::Dtype::Val. Verified against mlx/dtype.h:
// bool_=0, uint8=1, uint16=2, uint32=3, uint64=4, int8=5, int16=6, int32=7,
// int64=8, float16=9, float32=10, float64=11, bfloat16=12, complex64=13.
const FLOAT32: u8 = 10;

#[test]
fn zeros_then_read_shape() {
    let arr = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros should succeed");
    let shape = ffi::array_shape(&arr);
    assert_eq!(shape, vec![2, 3]);
}

#[test]
fn zeros_scalar_has_empty_shape() {
    let arr = ffi::array_zeros(&[], FLOAT32).expect("zeros should succeed");
    assert_eq!(ffi::array_shape(&arr), Vec::<i32>::new());
}

#[test]
fn zeros_metadata() {
    let arr = ffi::array_zeros(&[2, 3, 4], FLOAT32).expect("zeros should succeed");
    assert_eq!(ffi::array_ndim(&arr), 3);
    assert_eq!(ffi::array_size(&arr), 24);
    assert_eq!(ffi::array_dtype(&arr), FLOAT32);
}

#[test]
fn zeros_then_eval() {
    let arr = mlx_sys::array::ffi::array_zeros(&[8], FLOAT32).expect("zeros should succeed");
    mlx_sys::transforms::ffi::eval_one(&arr).expect("eval should succeed");
}
