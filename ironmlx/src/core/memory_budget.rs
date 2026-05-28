//! Memory budget estimation for the batched scheduler. Computes
//! GQA-aware KV cache bytes from `ModelMeta` and validates
//! `b_max × effective_cap_max × per_token_kv_bytes` against system
//! RAM minus model footprint and safety margin.
//!
//! Used at `Scheduler::new` (startup validation) and `admit_inner`
//! (runtime admission gate). See spec
//! `docs/superpowers/specs/2026-05-18-b1-p2-5-production-hardening-design.md`
//! §4.1 for the design rationale.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct ModelMeta {
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub hidden_size: i32,
    pub head_dim: Option<i32>,
    pub weight_bytes: usize,
    /// Maximum sequence length the model supports. Used by `serve()` for
    /// computing `effective_cap_max = min(--max-cache-cap CLI, max_position_embeddings)`.
    /// P5a-T5: added here so `serve<M>()` can read it from the `Model` trait
    /// without requiring a concrete model-specific `config()` method.
    pub max_position_embeddings: i32,
    /// VL vision spatial merge size (= VisionConfig.spatial_merge_size).
    /// Defaults to 2 for text-only models (unused when no images present).
    /// P5a-T5: carried here so generic HTTP handlers don't need a
    /// model-specific `config()` method.
    pub spatial_merge_size: i32,
}

impl ModelMeta {
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

pub const SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const SOFT_LIMIT_FRAC: f64 = 0.85;

pub fn kv_bytes_per_token(meta: &ModelMeta) -> usize {
    (meta.num_hidden_layers as usize)
        * (meta.num_key_value_heads as usize)
        * (meta.effective_head_dim() as usize)
        * 2  // K + V
        * 2 // bf16
}

pub fn kv_cache_bytes(b: usize, cap: usize, meta: &ModelMeta) -> usize {
    b * cap * kv_bytes_per_token(meta)
}

pub fn system_total_ram_bytes() -> usize {
    if let Ok(s) = std::env::var("IRONMLX_TOTAL_RAM_BYTES") {
        if let Ok(n) = s.parse::<usize>() {
            return n;
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                if let Ok(n) = s.trim().parse::<usize>() {
                    return n;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024
}

pub fn available_budget_bytes(meta: &ModelMeta) -> usize {
    system_total_ram_bytes()
        .saturating_sub(meta.weight_bytes)
        .saturating_sub(SAFETY_MARGIN_BYTES)
}

#[derive(Debug, Error)]
#[error(
    "memory budget exceeded: b_max={b_max} × effective_cap_max={cap} × \
     {bytes_per_token} bytes/token = {requested_bytes} bytes > available {available_bytes} \
     (total RAM {total_ram_bytes} - model {model_weight_bytes} - safety margin 2147483648). \
     Lower --b-max or --max-cache-cap."
)]
pub struct MemoryBudgetError {
    pub b_max: usize,
    pub cap: usize,
    pub bytes_per_token: usize,
    pub requested_bytes: usize,
    pub available_bytes: usize,
    pub total_ram_bytes: usize,
    pub model_weight_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct BudgetState {
    soft_limit: usize,
    active: Arc<AtomicUsize>,
}

impl BudgetState {
    pub fn new(total_budget: usize) -> Self {
        Self {
            soft_limit: ((total_budget as f64) * SOFT_LIMIT_FRAC) as usize,
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn soft_limit(&self) -> usize {
        self.soft_limit
    }

    pub fn active_bytes(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    pub fn shared_active(&self) -> Arc<AtomicUsize> {
        self.active.clone()
    }

    /// 试图把 `requested` 加到 active；若加后超 soft_limit 则返回 Err。
    pub fn try_admit(&self, requested: usize) -> Result<(), (usize, usize, usize)> {
        let cur = self.active.load(Ordering::Relaxed);
        if cur + requested > self.soft_limit {
            return Err((cur, requested, self.soft_limit));
        }
        self.active.fetch_add(requested, Ordering::Relaxed);
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        self.active.fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub fn validate_startup_budget(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
) -> Result<BudgetState, MemoryBudgetError> {
    let bytes_per_token = kv_bytes_per_token(meta);
    let requested = b_max * effective_cap_max * bytes_per_token;
    let available = available_budget_bytes(meta);
    if requested > available {
        return Err(MemoryBudgetError {
            b_max,
            cap: effective_cap_max,
            bytes_per_token,
            requested_bytes: requested,
            available_bytes: available,
            total_ram_bytes: system_total_ram_bytes(),
            model_weight_bytes: meta.weight_bytes,
        });
    }
    Ok(BudgetState::new(requested))
}

/// Realistic Qwen3.5-4B-like ModelMeta for tests.
#[doc(hidden)]
pub fn test_meta_qwen35() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 28,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 4096,
        head_dim: None,
        weight_bytes: 3 * 1024 * 1024 * 1024,
        max_position_embeddings: 32768,
        spatial_merge_size: 2,
    }
}

/// Realistic Qwen3.5-35B-A3B-4bit ModelMeta for tests.
///
/// Values from real snapshot text_config (verified P5b T0). The
/// `weight_bytes` is computed via the MoE-aware formula and rounded
/// to 17 GiB which matches `Qwen35MoeModel::approx_weight_bytes`
/// closely for the published config.
#[doc(hidden)]
pub fn test_meta_qwen35_moe() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 40,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        hidden_size: 2048,
        head_dim: Some(256),
        // approx: attn (4 * 2048^2 * 40 / 2) ≈ 335 MB
        //         routed (3 * 256 * 2048 * 512 * 40 / 2) ≈ 16.1 GB
        //         shared (3 * 2048 * 512 * 40 / 2) ≈ 63 MB
        //         embed + lm_head (2 * 248320 * 2048 / 2) ≈ 0.5 GB
        // total ≈ 17 GB
        weight_bytes: 17 * 1024 * 1024 * 1024,
        max_position_embeddings: 262144,
        spatial_merge_size: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn total_ram_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_total_ram_bytes<T>(bytes: &str, f: impl FnOnce() -> T) -> T {
        let _guard = total_ram_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", bytes);

        struct ClearTotalRamEnv;
        impl Drop for ClearTotalRamEnv {
            fn drop(&mut self) {
                std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
            }
        }
        let _clear = ClearTotalRamEnv;

        f()
    }

    fn meta() -> ModelMeta {
        test_meta_qwen35()
    }

    #[test]
    fn kv_bytes_per_token_gqa_aware() {
        // 28 × 8 × 128 (4096/32) × 2 × 2 = 114688
        assert_eq!(kv_bytes_per_token(&meta()), 114_688);
    }

    #[test]
    fn kv_cache_bytes_scales_with_b_and_cap() {
        let bytes = kv_cache_bytes(1, 1024, &meta());
        assert_eq!(bytes, 1024 * 114_688);
        assert_eq!(kv_cache_bytes(2, 1024, &meta()), 2 * bytes);
    }

    #[test]
    fn validate_within_budget_ok() {
        with_total_ram_bytes("34359738368", || {
            let st = validate_startup_budget(1, 4096, &meta()).expect("should fit");
            assert!(st.soft_limit() > 0);
        });
    }

    #[test]
    fn validate_over_budget_err() {
        with_total_ram_bytes("8589934592", || {
            let err = validate_startup_budget(4, 32768, &meta())
                .expect_err("4 × 32768 × 114688 should exceed 8 - 3 - 2 = 3 GiB budget");
            let msg = format!("{err}");
            assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
            assert!(msg.contains("Lower --b-max"), "msg: {msg}");
        });
    }

    #[test]
    fn budget_state_admit_release_round_trip() {
        let st = BudgetState::new(1_000_000);
        assert_eq!(st.active_bytes(), 0);
        st.try_admit(500_000).expect("under soft limit (850k)");
        assert_eq!(st.active_bytes(), 500_000);
        let err = st.try_admit(400_000);
        assert!(err.is_err(), "should reject above soft limit");
        assert_eq!(
            st.active_bytes(),
            500_000,
            "rejected admit leaves state unchanged"
        );
        st.release(500_000);
        assert_eq!(st.active_bytes(), 0);
    }

    #[test]
    fn moe_kv_bytes_per_token_matches_gqa_formula() {
        let m = test_meta_qwen35_moe();
        // 40 layers × 2 KV heads × 256 head_dim × 2 (K+V) × 2 (bf16) = 81920 bytes/token
        let expected = 40 * 2 * 256 * 2 * 2;
        assert_eq!(kv_bytes_per_token(&m), expected as usize);
    }

    #[test]
    fn moe_validate_budget_realistic_32gb_fits() {
        with_total_ram_bytes("34359738368", || {
            let st = validate_startup_budget(1, 8192, &test_meta_qwen35_moe())
                .expect("32GB host should fit 1 stream × 8K context for MoE");
            assert!(st.soft_limit() > 0);
        });
    }

    #[test]
    fn moe_validate_budget_rejects_overcommit_16gb() {
        with_total_ram_bytes("17179869184", || {
            // 16 GB - 17 GB weights - 2 GB safety margin = negative budget,
            // any cap must be rejected.
            let err = validate_startup_budget(1, 4096, &test_meta_qwen35_moe())
                .expect_err("16GB host cannot fit 17GB MoE weights");
            let msg = format!("{err}");
            assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
        });
    }
}
