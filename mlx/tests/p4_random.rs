//! Integration tests for mlx::random — PRNG + distributions.

use mlx::random::{key, seed, split, split_n};
use mlx::Array;

#[test]
fn key_is_deterministic_for_same_seed() {
    let k1 = key(42).expect("key 42");
    let k2 = key(42).expect("key 42 again");
    let v1: Vec<u32> = k1.to_vec().expect("k1 to_vec");
    let v2: Vec<u32> = k2.to_vec().expect("k2 to_vec");
    assert_eq!(v1, v2, "key(42) must be deterministic");
}

#[test]
fn key_differs_for_different_seeds() {
    let k1 = key(42).expect("key 42");
    let k2 = key(43).expect("key 43");
    let v1: Vec<u32> = k1.to_vec().expect("k1");
    let v2: Vec<u32> = k2.to_vec().expect("k2");
    assert_ne!(v1, v2, "different seeds should produce different keys");
}

#[test]
fn split_returns_two_distinct_subkeys() {
    let k = key(42).expect("key");
    let (a, b) = split(&k).expect("split");
    let va: Vec<u32> = a.to_vec().expect("a");
    let vb: Vec<u32> = b.to_vec().expect("b");
    let vk: Vec<u32> = k.to_vec().expect("k");
    assert_ne!(va, vb, "split sub-keys must differ");
    assert_ne!(va, vk, "sub-key 0 must differ from parent");
    assert_ne!(vb, vk, "sub-key 1 must differ from parent");
}

#[test]
fn split_n_returns_n_keys() {
    let k = key(42).expect("key");
    let keys = split_n(&k, 5).expect("split_n");
    assert_eq!(keys.shape().as_slice()[0], 5, "first dim must be num=5");
}

#[test]
fn seed_global_is_callable() {
    // Calling seed() should not error. We don't compare global state across calls
    // because subsequent ops in this test process may also touch the default key.
    seed(123);
    seed(456);
}

use mlx::random::{bits, normal, randint, uniform, uniform_default};
use mlx::Dtype;

#[test]
fn bits_returns_uint32_with_shape() {
    let k = key(42).expect("key");
    let b = bits(&[10], 4, Some(&k)).expect("bits");
    assert_eq!(b.shape().as_slice(), &[10]);
    let v: Vec<u32> = b.to_vec().expect("to_vec");
    assert_eq!(v.len(), 10);
}

#[test]
fn uniform_default_in_zero_to_one() {
    let k = key(42).expect("key");
    let u = uniform_default(&[100], Dtype::Float32, Some(&k)).expect("uniform");
    assert_eq!(u.shape().as_slice(), &[100]);
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0.0 && *x < 1.0, "uniform value {x} not in [0, 1)");
    }
}

#[test]
fn uniform_with_low_high_in_range() {
    let k = key(42).expect("key");
    let low = Array::from_slice(&[2.0_f32], &[]).expect("low");
    let high = Array::from_slice(&[5.0_f32], &[]).expect("high");
    let u = uniform(&low, &high, &[100], Dtype::Float32, Some(&k)).expect("uniform");
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 2.0 && *x < 5.0, "uniform value {x} not in [2, 5)");
    }
}

#[test]
fn normal_finite_and_centered() {
    let k = key(42).expect("key");
    let n = normal(&[1000], Dtype::Float32, None, None, Some(&k)).expect("normal");
    assert_eq!(n.shape().as_slice(), &[1000]);
    let v: Vec<f32> = n.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite value: {x}");
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    assert!(
        mean.abs() < 0.2,
        "normal mean {mean} not near 0 (loose tolerance for N=1000)"
    );
}

#[test]
fn randint_in_range_and_int32() {
    let k = key(42).expect("key");
    let low = Array::from_slice(&[0_i32], &[]).expect("low");
    let high = Array::from_slice(&[10_i32], &[]).expect("high");
    let r = randint(&low, &high, &[100], Dtype::Int32, Some(&k)).expect("randint");
    let v: Vec<i32> = r.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0 && *x < 10, "randint value {x} not in [0, 10)");
    }
}
