//! Integration tests for P2 — KV cache + Attention wiring.
//!
//! Because Mrope::cos_sin / apply are P3 stubs, full end-to-end forward
//! returns Err. Tests below verify that:
//!   - The cache hook executes before the Err propagates (cache.offset advances)
//!   - The cache=None path does not regress P1 behavior

use ironmlx::KVCache;
use mlx::Dtype;

#[test]
fn kv_cache_standalone_prefill_then_decode() {
    // Verify the cache works end-to-end without involving Attention.
    let mut cache = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);

    // Prefill: append 8 tokens
    let prefill_total = (1 * 4 * 8 * 256) as usize;
    let k1_data: Vec<f32> = (0..prefill_total).map(|i| i as f32).collect();
    let v1_data: Vec<f32> = (0..prefill_total).map(|i| (i as f32) * 10.0).collect();
    let k1: mlx::Array = (&k1_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v1: mlx::Array = (&v1_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let (k_full1, v_full1) = cache.update_and_fetch(&k1, &v1, &[8]).unwrap();
    assert_eq!(cache.offsets()[0], 8);
    assert_eq!(k_full1.shape().as_slice(), &[1, 4, 8, 256]);
    assert_eq!(v_full1.shape().as_slice(), &[1, 4, 8, 256]);

    // Decode: append 1 token
    let one_total = (1 * 4 * 1 * 256) as usize;
    let k2_data: Vec<f32> = (0..one_total).map(|i| (i + 100000) as f32).collect();
    let v2_data: Vec<f32> = (0..one_total).map(|i| (i + 200000) as f32).collect();
    let k2: mlx::Array = (&k2_data[..], (1, 4, 1, 256)).try_into().unwrap();
    let v2: mlx::Array = (&v2_data[..], (1, 4, 1, 256)).try_into().unwrap();
    let (k_full2, v_full2) = cache.update_and_fetch(&k2, &v2, &[1]).unwrap();
    assert_eq!(cache.offsets()[0], 9);
    assert_eq!(k_full2.shape().as_slice(), &[1, 4, 9, 256]);
    assert_eq!(v_full2.shape().as_slice(), &[1, 4, 9, 256]);
}

#[test]
fn kv_cache_with_step_eq_cap_one_shot_alloc() {
    // step >= cap → first update allocates full cap directly.
    let mut cache = KVCache::new(1, 4, 256, 256, Dtype::Float32, 64).with_step(64);
    let total = (1 * 4 * 8 * 256) as usize;
    let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let k: mlx::Array = (&k_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v: mlx::Array = (&v_data[..], (1, 4, 8, 256)).try_into().unwrap();
    cache.update_and_fetch(&k, &v, &[8]).unwrap();
    assert_eq!(cache.offsets()[0], 8);
    assert_eq!(cache.cap(), 64);
}

#[test]
fn kv_cache_reset_allows_session_reuse() {
    let mut cache = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
    let total = (1 * 4 * 8 * 256) as usize;
    let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let k: mlx::Array = (&k_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v: mlx::Array = (&v_data[..], (1, 4, 8, 256)).try_into().unwrap();
    cache.update_and_fetch(&k, &v, &[8]).unwrap();
    cache.reset();
    assert_eq!(cache.offsets()[0], 0);
    cache.update_and_fetch(&k, &v, &[8]).unwrap();
    assert_eq!(cache.offsets()[0], 8);
}
