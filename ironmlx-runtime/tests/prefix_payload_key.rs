use ironmlx_lm::core::cache::KVCache;
use ironmlx_lm::core::cache::PrefixLayerKind;
use ironmlx_lm::core::cache::TurboQuantKVBits;
use ironmlx_lm::nn::prefix_key_spec_for_caches;
use ironmlx_runtime::core::cache::PagedPrefixStore;
use mlx::{Array, Dtype};
use serial_test::serial;
fn turboquant_full_cache(bits: TurboQuantKVBits) -> LayerCache {
    let mut kv = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
        .with_step(16)
        .with_turboquant(bits)
        .expect("enable turboquant");
    let k_data: Vec<f32> = (0..(2 * 4 * 8))
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let v_data: Vec<f32> = (0..(2 * 4 * 8))
        .map(|i| ((i as f32) * 0.023).cos())
        .collect();
    let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
        .try_into()
        .unwrap();
    kv.update_and_fetch(&k, &v, &[4])
        .expect("write turboquant prefix");
    LayerCache::Full(kv)
}
#[test]
#[serial(mlx_metal)]
fn prefix_key_spec_for_turboquant_full_cache_uses_packed_profile() {
    let caches_k3v3 = vec![turboquant_full_cache(TurboQuantKVBits::K3V3)];
    let spec_k3v3 = prefix_key_spec_for_caches("model", &[1, 2, 3, 4], 4, None, 16, &caches_k3v3)
        .expect("prefix spec")
        .expect("TurboQuant prefix spec");

    assert_eq!(
        spec_k3v3.main_layers[0].kind,
        PrefixLayerKind::FullTurboQuantPacked
    );
    assert_eq!(spec_k3v3.main_layers[0].tensors[0].shape, vec![1, 2, 4, 1]);
    assert_eq!(spec_k3v3.main_layers[0].tensors[1].shape, vec![1, 2, 4]);
    assert_eq!(spec_k3v3.main_layers[0].tensors[2].shape, vec![1, 2, 4, 1]);
    assert_eq!(spec_k3v3.main_layers[0].tensors[3].shape, vec![1, 2, 4]);

    let caches_k3v4 = vec![turboquant_full_cache(TurboQuantKVBits::K3V4)];
    let spec_k3v4 = prefix_key_spec_for_caches("model", &[1, 2, 3, 4], 4, None, 16, &caches_k3v4)
        .expect("prefix spec")
        .expect("TurboQuant prefix spec");

    assert_ne!(
        PagedPrefixStore::key_for(&spec_k3v3),
        PagedPrefixStore::key_for(&spec_k3v4)
    );
}

use ironmlx_lm::core::cache::layer::LayerCache;
