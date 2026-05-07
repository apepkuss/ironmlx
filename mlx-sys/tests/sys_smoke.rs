use mlx_sys::array::ffi;

// Mirror of mlx::core::Dtype::Val. Verified against mlx/dtype.h:
// bool_=0, uint8=1, uint16=2, uint32=3, uint64=4, int8=5, int16=6, int32=7,
// int64=8, float16=9, float32=10, float64=11, bfloat16=12, complex64=13.
const FLOAT32: u8 = 10;
#[allow(dead_code)]
const UINT32: u8 = 3;

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

#[test]
fn binary_add_links() {
    let a = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    // P5.7: array_add takes 4 trailing StreamOrDevice params (default = no target).
    let _c =
        mlx_sys::array::ffi::array_add(&a, &b, false, false, 0, 0).expect("add should succeed");
}

#[test]
fn unary_exp_links() {
    let a = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    // P5.7: array_exp takes 4 trailing StreamOrDevice params (default = no target).
    let _e = mlx_sys::array::ffi::array_exp(&a, false, false, 0, 0).expect("exp should succeed");
}

#[test]
fn reduction_sum_links() {
    let a = ffi::array_zeros(&[3, 4], FLOAT32).expect("zeros");
    let _s = mlx_sys::array::ffi::array_sum_all(&a, false, false, false, 0, 0).expect("sum_all");
    let _s2 =
        mlx_sys::array::ffi::array_sum_axis(&a, 0, false, false, false, 0, 0).expect("sum_axis");
    let axes: Vec<i32> = vec![0, 1];
    let _s3 = mlx_sys::array::ffi::array_sum_axes(&a, &axes, false, false, false, 0, 0)
        .expect("sum_axes");
}

#[test]
fn shape_ops_link() {
    let a = ffi::array_zeros(&[6, 4], FLOAT32).expect("zeros");
    // P5.7: shape ops take 4 trailing StreamOrDevice params (default = no target).
    let _r =
        mlx_sys::array::ffi::array_reshape(&a, &[2, 3, 4], false, false, 0, 0).expect("reshape");
    let _t = mlx_sys::array::ffi::array_transpose(&a, false, false, 0, 0).expect("transpose");
    let _ta = mlx_sys::array::ffi::array_transpose_axes(&a, &[1, 0], false, false, 0, 0)
        .expect("transpose_axes");
    let _b = mlx_sys::array::ffi::array_broadcast_to(&a, &[2, 6, 4], false, false, 0, 0)
        .expect("broadcast_to");
}

#[test]
fn matmul_links() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[3, 4], FLOAT32).expect("zeros");
    // P5.7: array_matmul takes 4 trailing StreamOrDevice params.
    let _c = mlx_sys::array::ffi::array_matmul(&a, &b, false, false, 0, 0).expect("matmul");
}

#[test]
fn split_n_links_returns_vec() {
    let a = ffi::array_zeros(&[6, 4], FLOAT32).expect("zeros");
    let v = mlx_sys::array::ffi::array_split_n(&a, 3, 0, false, false, 0, 0).expect("split_n");
    assert_eq!(mlx_sys::array::ffi::split_result_len(&v), 3);
    let _first = mlx_sys::array::ffi::split_result_at(&v, 0).expect("split_result_at");
}

#[test]
fn concatenate_links_with_raw_ptr_slice() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    // Raw pointers cross the bridge as &[*const MlxArray] (cxx 1.0 limitation:
    // can't directly bridge &[&MlxArray]).
    let raw_ptrs: Vec<*const mlx_sys::array::ffi::MlxArray> =
        vec![&*a as *const _, &*b as *const _];
    let _c = unsafe {
        mlx_sys::array::ffi::array_concatenate(
            std::slice::from_raw_parts(raw_ptrs.as_ptr(), raw_ptrs.len()),
            0,
            false,
            false,
            0,
            0,
        )
    }
    .expect("concatenate");
}

#[test]
fn dtype_extension_u32_links() {
    let data: Vec<u32> = vec![1, 2, 3, 4];
    let _arr = mlx_sys::array::ffi::array_from_u32(&data, &[4]).expect("from_u32");
}

#[test]
fn indexing_ops_link() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let cond = ffi::array_zeros(&[2, 3], 0).expect("zeros bool"); // bool dtype = 0
    let b = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    // P5.7: indexing ops take 4 trailing StreamOrDevice params.
    let _w = mlx_sys::array::ffi::array_where(&cond, &a, &b, false, false, 0, 0).expect("where");
    let _s =
        mlx_sys::array::ffi::array_slice_strided(&a, &[0, 0], &[2, 3], &[1, 1], false, false, 0, 0)
            .expect("slice_strided");
    let indices = mlx_sys::array::ffi::array_from_u32(&[0_u32, 2], &[2]).expect("from_u32");
    let _t = mlx_sys::array::ffi::array_take(&a, &indices, 1, false, false, 0, 0).expect("take");
}

#[test]
fn device_default_links() {
    let d = mlx_sys::stream::ffi::default_device();
    // On macOS Apple Silicon the default is GPU (DeviceType::Gpu = 1)
    assert_eq!(
        d.device_type as i32,
        mlx_sys::stream::ffi::DeviceType::Gpu as i32
    );
    assert_eq!(d.index, 0);
    assert!(mlx_sys::stream::ffi::is_available(d));
}

#[test]
fn stream_default_and_new_links() {
    let d = mlx_sys::stream::ffi::default_device();
    let default_stream = mlx_sys::stream::ffi::default_stream(d);
    let new_stream = mlx_sys::stream::ffi::new_stream(d).expect("new_stream should succeed");
    assert_ne!(
        default_stream.index, new_stream.index,
        "new stream should have different index"
    );
    assert_eq!(new_stream.device.index, d.index);
}

#[test]
fn eval_many_links() {
    use mlx_sys::stream::ffi;
    // Empty slice — no-op but confirms ABI link.
    let empty: Vec<*const ffi::MlxArray> = vec![];
    unsafe { ffi::eval_many(&empty).expect("eval_many ABI") };
}

#[test]
fn async_eval_many_links() {
    use mlx_sys::stream::ffi;
    let empty: Vec<*const ffi::MlxArray> = vec![];
    unsafe { ffi::async_eval_many(&empty).expect("async_eval_many ABI") };
}

#[test]
fn compile_clear_cache_links() {
    use mlx_sys::compile::ffi;
    ffi::compile_clear_cache();
}

#[test]
fn shapes_vec_links() {
    use mlx_sys::fast::ffi;
    let mut v = ffi::shapes_vec_new();
    assert_eq!(ffi::shapes_vec_count(&v), 0);
    ffi::shapes_vec_push(v.pin_mut(), &[2, 3, 4]);
    ffi::shapes_vec_push(v.pin_mut(), &[8]);
    assert_eq!(ffi::shapes_vec_count(&v), 2);
}
