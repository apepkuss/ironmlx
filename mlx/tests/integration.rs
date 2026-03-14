use ironmlx::{init, Array, Device, Stream};

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
