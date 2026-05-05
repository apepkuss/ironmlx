//! Integration tests for mlx::random — PRNG + distributions.

use mlx::random::{key, seed, split, split_n};

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
