use ironmlx::{Array, Device, Stream, init};

fn gpu_stream() -> Stream {
    init();
    Stream::default_stream(&Device::gpu())
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[test]
fn test_gpu_smoke() {
    let s = gpu_stream();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = ironmlx::ops::add(&a, &b, &s).unwrap();
    assert!((c.item_f32().unwrap() - 7.0).abs() < 1e-6);
}

#[test]
fn test_cpu_smoke() {
    init();
    let s = Stream::default_stream(&Device::cpu());
    let a = Array::from_int(10);
    let b = Array::from_int(5);
    let c = ironmlx::ops::subtract(&a, &b, &s).unwrap();
    assert_eq!(c.item_i32().unwrap(), 5);
}

// ── default_stream() ──────────────────────────────────────────────────────────

#[test]
fn test_default_gpu_stream() {
    ironmlx::init();
    let s = ironmlx::default_stream(ironmlx::DeviceType::Gpu);
    let a = Array::from_float(2.0);
    let r = ironmlx::ops::square(&a, &s).unwrap();
    assert!((r.item_f32().unwrap() - 4.0).abs() < 1e-6);
}

// ── softmax / relu ────────────────────────────────────────────────────────────

#[test]
fn test_softmax() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0]);
    let b = ironmlx::ops::softmax(&a, &[-1], &s).unwrap();
    let v = b.to_vec_f32().unwrap();
    let total: f32 = v.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax should sum to 1, got {total}"
    );
    assert!(
        v[2] > v[1] && v[1] > v[0],
        "larger input → larger probability"
    );
}

#[test]
fn test_relu() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[-2.0, 0.0, 3.0]);
    let b = ironmlx::ops::relu(&a, &s).unwrap();
    let v = b.to_vec_f32().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-6, "negative → 0");
    assert!((v[1] - 0.0).abs() < 1e-6, "zero → 0");
    assert!((v[2] - 3.0).abs() < 1e-6, "positive → unchanged");
}

// ── sum / mean with axes ──────────────────────────────────────────────────────

#[test]
fn test_sum_all() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0]);
    let r = ironmlx::ops::sum(&a, &[], false, &s).unwrap();
    assert!((r.item_f32().unwrap() - 6.0).abs() < 1e-5);
}

#[test]
fn test_sum_axis() {
    let s = gpu_stream();
    // shape [2, 3], sum along axis 0 → shape [3]
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx::ops::sum(&a, &[0], false, &s).unwrap();
    assert_eq!(r.shape(), vec![3]);
    let v = r.to_vec_f32().unwrap();
    assert!((v[0] - 5.0).abs() < 1e-5);
    assert!((v[1] - 7.0).abs() < 1e-5);
    assert!((v[2] - 9.0).abs() < 1e-5);
}

#[test]
fn test_sum_keepdims() {
    let s = gpu_stream();
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx::ops::sum(&a, &[1], true, &s).unwrap();
    assert_eq!(r.shape(), vec![2, 1]);
}

#[test]
fn test_mean_axis() {
    let s = gpu_stream();
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx::ops::mean(&a, &[1], false, &s).unwrap();
    assert_eq!(r.shape(), vec![2]);
    let v = r.to_vec_f32().unwrap();
    assert!((v[0] - 2.0).abs() < 1e-5); // mean of [1,2,3]
    assert!((v[1] - 5.0).abs() < 1e-5); // mean of [4,5,6]
}

// ── reshape / transpose ───────────────────────────────────────────────────────

#[test]
fn test_reshape() {
    let s = gpu_stream();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    let b = ironmlx::ops::reshape(&a, &[3, 2], &s).unwrap();
    assert_eq!(b.shape(), vec![3, 2]);
    assert_eq!(b.size(), 6);
    assert_eq!(b.to_vec_f32().unwrap(), data);
}

#[test]
fn test_transpose() {
    let s = gpu_stream();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    let b = ironmlx::ops::transpose(&a, &s).unwrap();
    assert_eq!(b.shape(), vec![3, 2]);
}

#[test]
fn test_transpose_axes() {
    let s = gpu_stream();
    // shape [2, 3, 4] → transpose axes [2, 0, 1] → shape [4, 2, 3]
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let a = Array::from_slice_f32_shape(&data, &[2, 3, 4]);
    let b = ironmlx::ops::transpose_axes(&a, &[2, 0, 1], &s).unwrap();
    assert_eq!(b.shape(), vec![4, 2, 3]);
}

// ── Operator overloading ──────────────────────────────────────────────────────

#[test]
fn test_operator_add() {
    ironmlx::init();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = (&a + &b).unwrap();
    assert!((c.item_f32().unwrap() - 7.0).abs() < 1e-6);
}

#[test]
fn test_operator_sub() {
    ironmlx::init();
    let a = Array::from_float(9.0);
    let b = Array::from_float(4.0);
    let c = (&a - &b).unwrap();
    assert!((c.item_f32().unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_operator_mul() {
    ironmlx::init();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = (&a * &b).unwrap();
    assert!((c.item_f32().unwrap() - 12.0).abs() < 1e-6);
}

#[test]
fn test_operator_div() {
    ironmlx::init();
    let a = Array::from_float(8.0);
    let b = Array::from_float(2.0);
    let c = (&a / &b).unwrap();
    assert!((c.item_f32().unwrap() - 4.0).abs() < 1e-6);
}

// ── Array enhancements ────────────────────────────────────────────────────────

#[test]
fn test_array_clone() {
    let a = Array::from_float(1.5);
    let b = a.clone();
    assert!((b.item_f32().unwrap() - 1.5).abs() < 1e-6);
}

#[test]
fn test_to_vec_i32() {
    let data = vec![1i32, 2, 3, 4];
    let a = Array::from_slice_i32(&data);
    assert_eq!(a.to_vec_i32().unwrap(), data);
}

#[test]
fn test_array_2d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    assert_eq!(a.shape(), vec![2, 3]);
    assert_eq!(a.size(), 6);
    assert_eq!(a.to_vec_f32().unwrap(), data);
}
