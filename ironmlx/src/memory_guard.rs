use ironmlx_core::memory;

/// Memory guard that monitors GPU memory usage and enforces limits.
#[allow(dead_code)]
pub struct MemoryGuard {
    limit_bytes: usize,
    warn_threshold: f64,
}

impl MemoryGuard {
    pub fn new(limit_bytes: usize, warn_threshold: f64) -> Self {
        if limit_bytes > 0 {
            let _ = memory::set_memory_limit(limit_bytes);
        }
        Self {
            limit_bytes,
            warn_threshold,
        }
    }

    /// Get current memory stats (best-effort, returns 0 on error).
    #[allow(dead_code)]
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            active_bytes: memory::get_active_memory().unwrap_or(0),
            cache_bytes: memory::get_cache_memory().unwrap_or(0),
            peak_bytes: memory::get_peak_memory().unwrap_or(0),
            limit_bytes: self.limit_bytes,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub active_bytes: usize,
    pub cache_bytes: usize,
    pub peak_bytes: usize,
    #[allow(dead_code)]
    pub limit_bytes: usize,
}
