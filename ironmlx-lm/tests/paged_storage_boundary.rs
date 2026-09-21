//! A model-side paged cache can offload and restore through a non-filesystem backend.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ironmlx_lm::core::cache::page_storage::{KvPageStore, KvPageStoreFactory};
use ironmlx_lm::core::cache::{PagedKVCache, PagedKvHotColdConfig};
use ironmlx_lm::Result;
use mlx::{Array, Dtype};

#[derive(Debug, Default)]
struct Counters {
    saves: AtomicUsize,
    loads: AtomicUsize,
    resets: AtomicUsize,
    drops: AtomicUsize,
}

#[derive(Debug)]
struct MemoryFactory(Arc<Counters>);

impl KvPageStoreFactory for MemoryFactory {
    fn open(&self) -> Result<Box<dyn KvPageStore>> {
        Ok(Box::new(MemoryStore {
            counters: self.0.clone(),
            tensors: Mutex::new(HashMap::new()),
            next_key: AtomicUsize::new(0),
        }))
    }
}

#[derive(Debug)]
struct MemoryStore {
    counters: Arc<Counters>,
    tensors: Mutex<HashMap<PathBuf, (Array, Array)>>,
    next_key: AtomicUsize,
}

impl KvPageStore for MemoryStore {
    fn location(&self) -> PathBuf {
        PathBuf::from("memory")
    }
    fn segment_path(&self, start_page: i32, page_count: i32) -> PathBuf {
        let key = self.next_key.fetch_add(1, Ordering::Relaxed);
        PathBuf::from(format!("memory/{start_page}-{page_count}-{key}"))
    }
    fn save(&self, key: &Path, k: &Array, v: &Array) -> Result<()> {
        mlx::transforms::eval(&[k, v])?;
        self.tensors
            .lock()
            .unwrap()
            .insert(key.into(), (k.clone(), v.clone()));
        self.counters.saves.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn load(&self, key: &Path) -> Result<(Array, Array)> {
        self.counters.loads.fetch_add(1, Ordering::Relaxed);
        self.tensors
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing memory segment {}", key.display()))
    }
    fn size(&self, key: &Path) -> usize {
        self.tensors.lock().unwrap().get(key).map_or(0, |(k, v)| {
            (k.shape().as_slice().iter().product::<i32>()
                + v.shape().as_slice().iter().product::<i32>()) as usize
                * 4
        })
    }
    fn remove(&self, key: &Path) -> Result<()> {
        self.tensors.lock().unwrap().remove(key);
        Ok(())
    }
    fn reset(&self) -> Result<()> {
        self.tensors.lock().unwrap().clear();
        self.counters.resets.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for MemoryStore {
    fn drop(&mut self) {
        self.counters.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn injected_storage_restores_shared_prefix_and_owns_cleanup() {
    let counters = Arc::new(Counters::default());
    let config =
        PagedKvHotColdConfig::new(Arc::new(MemoryFactory(counters.clone())), 1, 1).unwrap();
    let mut cache = PagedKVCache::new(2, 1, 2, 2, Dtype::Float32, 12, 2, 16).unwrap();
    cache.enable_hot_cold_tiering(config).unwrap();
    let keys: Vec<f32> = (0..20).map(|value| value as f32).collect();
    let values: Vec<f32> = keys.iter().map(|value| value + 100.0).collect();
    let k: Array = (keys.as_slice(), (5, 1, 2, 2)).try_into().unwrap();
    let v: Array = (values.as_slice(), (5, 1, 2, 2)).try_into().unwrap();
    let mut offsets = vec![0, 0];
    cache
        .restore_prefix_pages_for_rows_on(&k, &v, &mut offsets, &[0, 1], 9, ())
        .unwrap();
    assert!(cache.hot_cold_summary().unwrap().offloaded_pages > 0);
    let (actual_k, actual_v) = cache.materialize_prefix_on(&offsets, 9, ()).unwrap();
    assert_eq!(
        actual_k.to_vec::<f32>().unwrap(),
        [&keys[..18], &keys[..18]].concat()
    );
    assert_eq!(
        actual_v.to_vec::<f32>().unwrap(),
        [&values[..18], &values[..18]].concat()
    );
    assert!(counters.saves.load(Ordering::Relaxed) > 0);
    assert!(counters.loads.load(Ordering::Relaxed) > 0);
    cache.clear();
    assert_eq!(cache.hot_cold_summary().unwrap().offloaded_pages, 0);
    assert_eq!(counters.resets.load(Ordering::Relaxed), 1);
    drop(cache);
    assert_eq!(counters.drops.load(Ordering::Relaxed), 1);
}
