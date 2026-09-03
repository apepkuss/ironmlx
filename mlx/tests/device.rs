use mlx::{Device, DeviceType};

#[test]
fn cpu_constructor() {
    let d = Device::cpu();
    assert_eq!(d.device_type, DeviceType::Cpu);
    assert_eq!(d.index, 0);
}

#[test]
fn gpu_constructor() {
    let d = Device::gpu(0);
    assert_eq!(d.device_type, DeviceType::Gpu);
    assert_eq!(d.index, 0);

    let d2 = Device::gpu(3);
    assert_eq!(d2.index, 3);
}

#[test]
fn device_equality_and_copy() {
    let a = Device::gpu(0);
    let b = a; // Copy
    assert_eq!(a, b);
    assert_ne!(a, Device::cpu());
    assert_ne!(Device::gpu(0), Device::gpu(1));
}

#[test]
fn default_device_is_gpu_on_apple_silicon() {
    // On macOS Apple Silicon the default device is the GPU.
    let d = mlx::default_device();
    assert_eq!(d.device_type, DeviceType::Gpu);
}

#[test]
fn cpu_and_gpu_both_available() {
    assert!(mlx::is_available(Device::cpu()));
    assert!(mlx::is_available(Device::gpu(0)));
}

#[test]
fn gpu_device_count_at_least_one() {
    assert!(mlx::device_count(DeviceType::Gpu) >= 1);
    // CPU "count" semantics: MLX returns 1 for CPU (single logical device).
    assert!(mlx::device_count(DeviceType::Cpu) >= 1);
}

#[test]
fn set_default_device_round_trip() {
    let original = mlx::default_device();
    mlx::set_default_device(Device::cpu());
    assert_eq!(mlx::default_device(), Device::cpu());
    // Restore so other tests aren't affected.
    mlx::set_default_device(original);
    assert_eq!(mlx::default_device(), original);
}
