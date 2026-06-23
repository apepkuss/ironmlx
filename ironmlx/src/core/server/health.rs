//! `/healthz` JSON endpoint (B1-p2.5 G3). Snapshot of scheduler /
//! memory / model state for monitoring + load balancer health probes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::core::memory_budget::system_total_ram_bytes;

#[derive(Debug, Serialize)]
pub enum HealthStatus {
    #[serde(rename = "healthy")]
    Healthy,
    #[serde(rename = "degraded")]
    Degraded,
    #[serde(rename = "down")]
    Down,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub max_position_embeddings: i32,
}

#[derive(Debug, Serialize)]
pub struct SchedulerInfo {
    pub b_max: usize,
    pub b_active: usize,
    pub b_queued: usize,
    pub queue_max: usize,
    pub admission_queue_full_count: u64,
    pub memory_budget_exceeded_count: u64,
}

#[derive(Debug, Serialize)]
pub struct MemoryInfo {
    pub total_ram_bytes: usize,
    pub free_ram_bytes: usize,
    pub kv_cache_active_bytes: usize,
    pub kv_cache_soft_limit_bytes: usize,
    pub mlx_total_bytes: Option<usize>,
    pub mlx_max_recommended_bytes: Option<usize>,
    pub mlx_active_bytes: usize,
    pub mlx_cache_bytes: usize,
    pub mlx_peak_bytes: usize,
    pub mlx_memory_limit_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct MtpHealthInfo {
    pub enabled: bool,
    pub draft_tokens: Option<usize>,
    pub prefill_count: u64,
    pub step_count: u64,
    pub fallback_prefill_count: u64,
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
}

#[derive(Clone)]
pub struct MtpHealthConfig {
    enabled: bool,
    draft_tokens: Option<usize>,
    prefill_count: Arc<AtomicU64>,
    step_count: Arc<AtomicU64>,
    fallback_prefill_count: Arc<AtomicU64>,
    drafted_tokens: Arc<AtomicU64>,
    accepted_draft_tokens: Arc<AtomicU64>,
}

impl MtpHealthConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            draft_tokens: None,
            prefill_count: Arc::new(AtomicU64::new(0)),
            step_count: Arc::new(AtomicU64::new(0)),
            fallback_prefill_count: Arc::new(AtomicU64::new(0)),
            drafted_tokens: Arc::new(AtomicU64::new(0)),
            accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn enabled(
        draft_tokens: usize,
        prefill_count: Arc<AtomicU64>,
        step_count: Arc<AtomicU64>,
        fallback_prefill_count: Arc<AtomicU64>,
        drafted_tokens: Arc<AtomicU64>,
        accepted_draft_tokens: Arc<AtomicU64>,
    ) -> Self {
        Self {
            enabled: true,
            draft_tokens: Some(draft_tokens),
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
        }
    }

    fn snapshot(&self) -> MtpHealthInfo {
        MtpHealthInfo {
            enabled: self.enabled,
            draft_tokens: self.draft_tokens,
            prefill_count: self.prefill_count.load(Ordering::Relaxed),
            step_count: self.step_count.load(Ordering::Relaxed),
            fallback_prefill_count: self.fallback_prefill_count.load(Ordering::Relaxed),
            drafted_tokens: self.drafted_tokens.load(Ordering::Relaxed),
            accepted_draft_tokens: self.accepted_draft_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthSnapshot {
    pub status: HealthStatus,
    pub uptime_secs: u64,
    pub model: ModelInfo,
    pub scheduler: SchedulerInfo,
    pub memory: MemoryInfo,
    pub mtp: MtpHealthInfo,
    pub device_name: Option<String>,
    pub version: &'static str,
}

pub struct SchedulerHealthCollector {
    pub start_time: Instant,
    pub b_max: usize,
    pub queue_max: usize,
    pub model_name: String,
    pub max_position_embeddings: i32,
    pub b_active: Arc<AtomicU64>,
    pub b_queued: Arc<AtomicU64>,
    pub admission_queue_full_count: Arc<AtomicU64>,
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    pub kv_cache_soft_limit_bytes: usize,
    pub mtp: MtpHealthConfig,
}

impl SchedulerHealthCollector {
    pub fn snapshot(&self) -> HealthSnapshot {
        let uptime_secs = self.start_time.elapsed().as_secs();
        let total_ram_bytes = system_total_ram_bytes();
        let free_ram_bytes = system_free_ram_bytes();
        let b_active = self.b_active.load(Ordering::Relaxed) as usize;
        let b_queued = self.b_queued.load(Ordering::Relaxed) as usize;
        let admission_full = self.admission_queue_full_count.load(Ordering::Relaxed);
        let mb_exceeded = self.memory_budget_exceeded_count.load(Ordering::Relaxed);
        let kv_active = self.kv_cache_active_bytes.load(Ordering::Relaxed);
        let mlx_memory = mlx::memory::snapshot();

        let status = classify_status(
            b_queued,
            self.queue_max,
            free_ram_bytes,
            kv_active,
            self.kv_cache_soft_limit_bytes,
        );

        HealthSnapshot {
            status,
            uptime_secs,
            model: ModelInfo {
                name: self.model_name.clone(),
                max_position_embeddings: self.max_position_embeddings,
            },
            scheduler: SchedulerInfo {
                b_max: self.b_max,
                b_active,
                b_queued,
                queue_max: self.queue_max,
                admission_queue_full_count: admission_full,
                memory_budget_exceeded_count: mb_exceeded,
            },
            memory: MemoryInfo {
                total_ram_bytes,
                free_ram_bytes,
                kv_cache_active_bytes: kv_active,
                kv_cache_soft_limit_bytes: self.kv_cache_soft_limit_bytes,
                mlx_total_bytes: mlx_memory.total_bytes,
                mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
                mlx_active_bytes: mlx_memory.active_bytes,
                mlx_cache_bytes: mlx_memory.cache_bytes,
                mlx_peak_bytes: mlx_memory.peak_bytes,
                mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
            },
            mtp: self.mtp.snapshot(),
            device_name: mlx_memory.device_name,
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

pub fn classify_status(
    b_queued: usize,
    queue_max: usize,
    free_ram_bytes: usize,
    kv_cache_active_bytes: usize,
    kv_cache_soft_limit_bytes: usize,
) -> HealthStatus {
    let queue_high = queue_max > 0 && b_queued >= queue_max / 2;
    let mem_low = free_ram_bytes < (1024 * 1024 * 1024);
    let budget_near = kv_cache_soft_limit_bytes > 0
        && kv_cache_active_bytes >= ((kv_cache_soft_limit_bytes as f64) * 0.9) as usize;
    if queue_high || mem_low || budget_near {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

pub fn system_free_ram_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                let mut page_size = 16_384_usize;
                let mut pages_free = 0_usize;
                for line in s.lines() {
                    if let Some(rest) =
                        line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
                    {
                        if let Some(num) = rest.split(' ').next() {
                            if let Ok(p) = num.parse::<usize>() {
                                page_size = p;
                            }
                        }
                    }
                    if let Some(rest) = line.strip_prefix("Pages free:") {
                        let t = rest.trim().trim_end_matches('.');
                        if let Ok(n) = t.parse::<usize>() {
                            pages_free = n;
                        }
                    }
                }
                if pages_free > 0 {
                    return pages_free * page_size;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    4 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Arc;

    #[test]
    fn classify_healthy_when_all_green() {
        let s = classify_status(0, 32, 8 * 1024 * 1024 * 1024, 1_000_000, 10_000_000);
        assert!(matches!(s, HealthStatus::Healthy));
    }

    #[test]
    fn classify_degraded_when_queue_half_full() {
        let s = classify_status(16, 32, 8 * 1024 * 1024 * 1024, 0, 1);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_free_ram_low() {
        let s = classify_status(0, 32, 500_000_000, 0, 1);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_budget_near_soft_limit() {
        let s = classify_status(0, 32, 8 * 1024 * 1024 * 1024, 9_500_000, 10_000_000);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    fn test_collector(mtp: MtpHealthConfig) -> SchedulerHealthCollector {
        SchedulerHealthCollector {
            start_time: Instant::now(),
            b_max: 1,
            queue_max: 8,
            model_name: "test-model".to_string(),
            max_position_embeddings: 4096,
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: Arc::new(AtomicU64::new(0)),
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
            mtp,
        }
    }

    #[test]
    fn snapshot_mtp_reports_disabled_config() {
        let snapshot = test_collector(MtpHealthConfig::disabled()).snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 0);
        assert_eq!(snapshot.mtp.drafted_tokens, 0);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 0);
    }

    #[test]
    fn snapshot_mtp_reports_enabled_config_and_live_counters() {
        let prefill_count = Arc::new(AtomicU64::new(7));
        let step_count = Arc::new(AtomicU64::new(11));
        let fallback_prefill_count = Arc::new(AtomicU64::new(13));
        let drafted_tokens = Arc::new(AtomicU64::new(17));
        let accepted_draft_tokens = Arc::new(AtomicU64::new(19));
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            prefill_count.clone(),
            step_count.clone(),
            fallback_prefill_count.clone(),
            drafted_tokens.clone(),
            accepted_draft_tokens.clone(),
        ))
        .snapshot();

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 7);
        assert_eq!(snapshot.mtp.step_count, 11);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 13);
        assert_eq!(snapshot.mtp.drafted_tokens, 17);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 19);

        prefill_count.store(13, Ordering::Relaxed);
        step_count.store(17, Ordering::Relaxed);
        fallback_prefill_count.store(23, Ordering::Relaxed);
        drafted_tokens.store(29, Ordering::Relaxed);
        accepted_draft_tokens.store(31, Ordering::Relaxed);
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
        ))
        .snapshot();

        assert_eq!(snapshot.mtp.prefill_count, 13);
        assert_eq!(snapshot.mtp.step_count, 17);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 23);
        assert_eq!(snapshot.mtp.drafted_tokens, 29);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 31);
    }

    #[test]
    fn health_memory_serializes_mlx_allocator_fields() {
        let snapshot = HealthSnapshot {
            status: HealthStatus::Healthy,
            uptime_secs: 7,
            model: ModelInfo {
                name: "test-model".to_string(),
                max_position_embeddings: 4096,
            },
            scheduler: SchedulerInfo {
                b_max: 8,
                b_active: 1,
                b_queued: 0,
                queue_max: 16,
                admission_queue_full_count: 0,
                memory_budget_exceeded_count: 0,
            },
            memory: MemoryInfo {
                total_ram_bytes: 64,
                free_ram_bytes: 32,
                kv_cache_active_bytes: 16,
                kv_cache_soft_limit_bytes: 24,
                mlx_total_bytes: Some(55),
                mlx_max_recommended_bytes: Some(66),
                mlx_active_bytes: 11,
                mlx_cache_bytes: 22,
                mlx_peak_bytes: 33,
                mlx_memory_limit_bytes: 44,
            },
            mtp: MtpHealthInfo {
                enabled: false,
                draft_tokens: None,
                prefill_count: 0,
                step_count: 0,
                fallback_prefill_count: 0,
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
            },
            device_name: Some("Apple Test GPU".to_string()),
            version: "test",
        };

        let value = serde_json::to_value(snapshot).expect("serialize health snapshot");
        assert_eq!(value["memory"]["mlx_total_bytes"], 55);
        assert_eq!(value["memory"]["mlx_max_recommended_bytes"], 66);
        assert_eq!(value["memory"]["mlx_active_bytes"], 11);
        assert_eq!(value["memory"]["mlx_cache_bytes"], 22);
        assert_eq!(value["memory"]["mlx_peak_bytes"], 33);
        assert_eq!(value["memory"]["mlx_memory_limit_bytes"], 44);
        assert_eq!(value["device_name"], "Apple Test GPU");
    }
}
