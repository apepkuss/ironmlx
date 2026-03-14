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
