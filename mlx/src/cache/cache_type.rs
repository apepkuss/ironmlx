/// Supported KV cache types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// Standard KV cache — supports block slicing for paged storage.
    KVCache,
    /// Rotating/sliding window KV cache — NOT sliceable, needs boundary snapshot.
    RotatingKVCache,
    /// Quantized KV cache — NOT sliceable.
    QuantizedKVCache,
    /// Raw arrays cache (e.g., for SSM states like GatedDeltaNet) — NOT sliceable.
    ArraysCache,
}

impl CacheType {
    /// Whether this cache type supports block-level slicing for paged storage.
    pub fn is_sliceable(&self) -> bool {
        matches!(self, CacheType::KVCache)
    }

    /// Whether this cache type needs boundary snapshots for SSD offloading.
    pub fn needs_boundary_snapshot(&self) -> bool {
        !self.is_sliceable()
    }
}

/// Per-layer cache configuration.
#[derive(Debug, Clone)]
pub struct LayerCacheConfig {
    pub cache_type: CacheType,
    pub layer_index: usize,
}

/// Model-level cache configuration describing the cache type per layer.
#[derive(Debug, Clone)]
pub struct ModelCacheConfig {
    pub num_layers: usize,
    pub layer_configs: Vec<LayerCacheConfig>,
    pub is_hybrid: bool,
}

impl ModelCacheConfig {
    /// Create a uniform config where all layers use the same cache type.
    pub fn uniform(num_layers: usize, cache_type: CacheType) -> Self {
        let layer_configs: Vec<LayerCacheConfig> = (0..num_layers)
            .map(|i| LayerCacheConfig {
                cache_type,
                layer_index: i,
            })
            .collect();
        Self {
            num_layers,
            layer_configs,
            is_hybrid: false,
        }
    }

    /// Create a hybrid config (e.g., Qwen3.5 with mixed GatedDeltaNet + Attention layers).
    pub fn hybrid(layer_configs: Vec<LayerCacheConfig>) -> Self {
        let num_layers = layer_configs.len();
        let first_type = layer_configs.first().map(|c| c.cache_type);
        let is_hybrid = layer_configs
            .iter()
            .any(|c| Some(c.cache_type) != first_type);
        Self {
            num_layers,
            layer_configs,
            is_hybrid,
        }
    }

    /// Get indices of layers that support block slicing.
    pub fn sliceable_layers(&self) -> Vec<usize> {
        self.layer_configs
            .iter()
            .filter(|c| c.cache_type.is_sliceable())
            .map(|c| c.layer_index)
            .collect()
    }

    /// Get indices of layers that need boundary snapshots.
    pub fn non_sliceable_layers(&self) -> Vec<usize> {
        self.layer_configs
            .iter()
            .filter(|c| !c.cache_type.is_sliceable())
            .map(|c| c.layer_index)
            .collect()
    }

    /// Count of sliceable layers.
    pub fn sliceable_count(&self) -> usize {
        self.layer_configs
            .iter()
            .filter(|c| c.cache_type.is_sliceable())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_is_sliceable() {
        assert!(CacheType::KVCache.is_sliceable());
        assert!(!CacheType::KVCache.needs_boundary_snapshot());
    }

    #[test]
    fn rotating_kv_cache_not_sliceable() {
        assert!(!CacheType::RotatingKVCache.is_sliceable());
        assert!(CacheType::RotatingKVCache.needs_boundary_snapshot());
    }

    #[test]
    fn quantized_kv_cache_not_sliceable() {
        assert!(!CacheType::QuantizedKVCache.is_sliceable());
        assert!(CacheType::QuantizedKVCache.needs_boundary_snapshot());
    }

    #[test]
    fn arrays_cache_not_sliceable() {
        assert!(!CacheType::ArraysCache.is_sliceable());
        assert!(CacheType::ArraysCache.needs_boundary_snapshot());
    }

    #[test]
    fn uniform_config() {
        let config = ModelCacheConfig::uniform(32, CacheType::KVCache);
        assert_eq!(config.num_layers, 32);
        assert!(!config.is_hybrid);
        assert_eq!(config.sliceable_layers().len(), 32);
        assert_eq!(config.non_sliceable_layers().len(), 0);
        assert_eq!(config.sliceable_count(), 32);
    }

    #[test]
    fn hybrid_config() {
        // Simulate Qwen3.5: alternating GatedDeltaNet (ArraysCache) and Attention (KVCache)
        let layer_configs: Vec<LayerCacheConfig> = (0..8)
            .map(|i| LayerCacheConfig {
                cache_type: if i % 2 == 0 {
                    CacheType::ArraysCache
                } else {
                    CacheType::KVCache
                },
                layer_index: i,
            })
            .collect();

        let config = ModelCacheConfig::hybrid(layer_configs);
        assert_eq!(config.num_layers, 8);
        assert!(config.is_hybrid);
        assert_eq!(config.sliceable_layers(), vec![1, 3, 5, 7]);
        assert_eq!(config.non_sliceable_layers(), vec![0, 2, 4, 6]);
        assert_eq!(config.sliceable_count(), 4);
    }

    #[test]
    fn hybrid_same_type_not_hybrid() {
        let layer_configs: Vec<LayerCacheConfig> = (0..4)
            .map(|i| LayerCacheConfig {
                cache_type: CacheType::KVCache,
                layer_index: i,
            })
            .collect();

        let config = ModelCacheConfig::hybrid(layer_configs);
        assert!(!config.is_hybrid);
    }
}
