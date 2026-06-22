//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::Context;
use axum::{routing::get, routing::post, Router};
use tokio::sync::Mutex;

use crate::core::cache::{PagedPrefixCacheConfig, PrefixLruCacheConfig, TurboQuantKVBits};
use crate::core::model::Model;
use crate::core::scheduler::DenseVlMethods;
use crate::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SchedulerAutotuneRuntimeRequest,
};
use crate::core::speculative::MtpSpeculativeModel;
use crate::core::tokenizer::Tokenizer;
use crate::Result;

pub mod anthropic;
pub mod chat_format;
pub mod diffusion_gemma;
pub mod health;
pub(crate) mod openai;
pub mod scheduler_actor;
pub mod vision;

#[derive(Clone)]
pub enum VisionInputConfig {
    Qwen {
        spatial_merge_size: i32,
    },
    Gemma4 {
        vision_config: crate::models::gemma4::Gemma4VisionConfig,
    },
    DiffusionGemma {
        vision_config: crate::models::gemma4::Gemma4VisionConfig,
        image_token_id: Option<i32>,
    },
    MiniCpmV46 {
        /// Effective image-token downsample = 4 (VitMerger 2×2 × Merger 2×2).
        spatial_merge_size: i32,
    },
}

/// HTTP server shared state. The model is wrapped in a tokio Mutex —
/// concurrent requests serialize behind the lock (P4 single-stream contract).
///
/// 3b-2 adds `scheduler_handle`; short prompts, including VL prompts, route
/// through the SchedulerActor. Long text prompts keep using GenerationStream
/// unless model limits or paged prefix cache require SchedulerActor features.
///
/// P5a-T5: AppState is now generic over `M: Model + DenseVlMethods + Send +
/// 'static`. CLI call sites pass either `Qwen35Model` or `Qwen35MoeModel`
/// based on the checkpoint `model_type`.
///
/// `Clone` is implemented manually so the derive macro doesn't emit an
/// unwanted `M: Clone` bound — all fields clone without needing `M: Clone`
/// because `Arc<Mutex<M>>` and `Arc<...>` are `Clone` unconditionally.
pub struct AppState<M: Model + DenseVlMethods + Send + 'static> {
    pub model: Arc<Mutex<M>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    /// Default prefill chunk size (max tokens per prefill forward). `0`
    /// disables chunking. Applied to every `GenerateRequest` constructed
    /// by the request handlers.
    pub prefill_chunk_size: usize,
    pub vision_input: VisionInputConfig,
    /// SchedulerActor handle. Routed to by short-prompt requests. See
    /// `serve_via_scheduler_*` in `openai.rs`.
    pub scheduler_handle: scheduler_actor::SchedulerActorHandle,
    /// True when the SchedulerActor was started with paged SSD prefix cache.
    pub paged_prefix_cache_enabled: bool,
    /// Maximum concurrent in-flight requests routed to the SchedulerActor.
    pub b_max: usize,
    /// Admission-window deadline (milliseconds) — drain-window timeout.
    pub admission_deadline_ms: u64,
    /// FIFO admission queue capacity.
    pub admission_queue_max: usize,
    /// Effective cap_max = min(--max-cache-cap CLI flag, model.config.max_position_embeddings).
    /// Per-request `prompt_len + max_new_tokens` exceeding this returns HTTP 413. B1-p2.3f.
    pub effective_cap_max: usize,
    /// Runtime scheduler profile. Base config is applied at boot; rules may
    /// select request-level chunk/cadence settings after tokenization.
    pub scheduler_runtime_profile: Arc<SchedulerAutotuneRuntimeProfile>,
    /// Optional TurboQuant K/V bit-widths for full-attention KV cache reads.
    pub kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    /// Health snapshot collector for `/healthz`. Holds shared Arc atomics
    /// wired to the SchedulerActor driver loop + BudgetState. B1-p2.5 G3.
    pub health_collector: Arc<health::SchedulerHealthCollector>,
}

impl<M: Model + DenseVlMethods + Send + 'static> Clone for AppState<M> {
    fn clone(&self) -> Self {
        AppState {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            model_id: self.model_id.clone(),
            prefill_chunk_size: self.prefill_chunk_size,
            vision_input: self.vision_input.clone(),
            scheduler_handle: self.scheduler_handle.clone(),
            paged_prefix_cache_enabled: self.paged_prefix_cache_enabled,
            b_max: self.b_max,
            admission_deadline_ms: self.admission_deadline_ms,
            admission_queue_max: self.admission_queue_max,
            effective_cap_max: self.effective_cap_max,
            scheduler_runtime_profile: self.scheduler_runtime_profile.clone(),
            kv_cache_turboquant_bits: self.kv_cache_turboquant_bits,
            health_collector: self.health_collector.clone(),
        }
    }
}

impl<M: Model + DenseVlMethods + Send + 'static> AppState<M> {
    pub(crate) fn scheduler_request_config(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> SchedulerAutotuneProfileConfig {
        let active = self.scheduler_handle.b_active.load(Ordering::Relaxed) as usize;
        let queued = self.scheduler_handle.b_queued.load(Ordering::Relaxed) as usize;
        self.scheduler_runtime_profile
            .select_config(SchedulerAutotuneRuntimeRequest {
                prompt_len,
                max_new_tokens,
                effective_concurrency: active.saturating_add(queued).saturating_add(1),
            })
    }
}

pub(crate) fn should_route_to_scheduler<M: Model>(
    prompt_len: usize,
    prefill_chunk_size: usize,
    b_max: usize,
    paged_prefix_cache_enabled: bool,
) -> bool {
    if paged_prefix_cache_enabled {
        return true;
    }
    if prefill_chunk_size == 0 || prompt_len <= prefill_chunk_size {
        return true;
    }
    M::fresh_prefill_batch_limit(prompt_len, b_max) < b_max
}

trait SchedulerActorSpawner<M>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool;

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle>;
}

struct PlainSchedulerActorSpawner {
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
}

impl<M> SchedulerActorSpawner<M> for PlainSchedulerActorSpawner
where
    M: Model + DenseVlMethods + Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        if let Some(config) = self.paged_prefix_cache {
            Ok(
                scheduler_actor::spawn_scheduler_actor_with_paged_prefix_cache(
                    model,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    config,
                    self.prefix_lru_cache,
                )?,
            )
        } else {
            Ok(scheduler_actor::spawn_scheduler_actor(
                model,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
            )?)
        }
    }
}

struct MtpSchedulerActorSpawner<H> {
    mtp: H,
    mtp_draft_tokens: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
}

impl<M> SchedulerActorSpawner<M> for MtpSchedulerActorSpawner<M::MtpHead>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        Ok(scheduler_actor::spawn_scheduler_actor_with_mtp(
            model,
            self.mtp,
            self.mtp_draft_tokens,
            b_max,
            admission_deadline,
            admission_queue_max,
            effective_cap_max,
            decode_cadence_mid_chunk_cap,
            meta,
            self.paged_prefix_cache,
            self.prefix_lru_cache,
        )?)
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn serve<M>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize, // 3f
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    serve_inner(
        model,
        tokenizer,
        model_id,
        host,
        port,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        None,
        PlainSchedulerActorSpawner {
            paged_prefix_cache,
            prefix_lru_cache,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_mtp<M>(
    model: M,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
) -> Result<()>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    serve_inner(
        model,
        tokenizer,
        model_id,
        host,
        port,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        Some(mtp_draft_tokens),
        MtpSchedulerActorSpawner {
            mtp,
            mtp_draft_tokens,
            paged_prefix_cache,
            prefix_lru_cache,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn serve_inner<M, S>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    mtp_health_draft_tokens: Option<usize>,
    scheduler_actor_spawner: S,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
    S: SchedulerActorSpawner<M>,
{
    let model = Arc::new(Mutex::new(model));
    let admission_deadline = std::time::Duration::from_millis(admission_deadline_ms);

    // 3f + P5a-T5: extract ModelMeta (which now carries max_position_embeddings)
    // inside a single async lock guard so serve<M>() doesn't need a concrete
    // model-specific `config()` method. `blocking_lock` would panic here because
    // `serve` runs inside a Tokio runtime (tests S5 of 3d / 3f-T4 caught this).
    let meta = {
        let guard = model.lock().await;
        guard.model_meta()
    };
    let model_max_context: usize = meta.max_position_embeddings.max(0) as usize;
    let effective_cap_max = max_cache_cap.min(model_max_context);
    if max_cache_cap > model_max_context {
        tracing::warn!(
            "max_cache_cap CLI flag {} exceeds model_max_context {} — capping at {}",
            max_cache_cap,
            model_max_context,
            model_max_context
        );
    }
    if scheduler_autotune_report {
        let report = crate::core::scheduler_autotune::build_scheduler_autotune_report(
            crate::core::scheduler_autotune::SchedulerAutotuneInput {
                model_name: model_id.clone(),
                meta,
                prefill_chunk_size,
                b_max,
                admission_deadline_ms,
                admission_queue_max,
                requested_max_cache_cap: max_cache_cap,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                total_ram_bytes: crate::core::memory_budget::system_total_ram_bytes(),
            },
            crate::core::scheduler_autotune::prompt_batch_limits_for_model::<M>(b_max),
        );
        tracing::info!(
            target: "ironmlx::scheduler_autotune",
            "\n{}",
            report.render_text()
        );
    }

    let paged_prefix_cache_enabled = scheduler_actor_spawner.paged_prefix_cache_enabled();
    let scheduler_handle = scheduler_actor_spawner.spawn(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
    )?;
    let vision_input = vision_input_override.unwrap_or(VisionInputConfig::Qwen {
        spatial_merge_size: meta.spatial_merge_size,
    });

    let mtp_health = mtp_health_draft_tokens
        .map(|draft_tokens| {
            health::MtpHealthConfig::enabled(
                draft_tokens,
                scheduler_handle.mtp_prefill_count.clone(),
                scheduler_handle.mtp_step_count.clone(),
            )
        })
        .unwrap_or_else(health::MtpHealthConfig::disabled);
    let health_collector = build_health_collector(
        model_id.clone(),
        model_max_context,
        b_max,
        admission_queue_max,
        &scheduler_handle,
        mtp_health,
    );

    let state = AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        vision_input,
        scheduler_handle,
        paged_prefix_cache_enabled,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        effective_cap_max, // 3f
        scheduler_runtime_profile: Arc::new(scheduler_runtime_profile),
        kv_cache_turboquant_bits,
        health_collector,
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing socket addr {host}:{port}"))?;
    tracing::info!("ironmlx server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_health_collector(
    model_id: String,
    model_max_context: usize,
    b_max: usize,
    admission_queue_max: usize,
    scheduler_handle: &scheduler_actor::SchedulerActorHandle,
    mtp: health::MtpHealthConfig,
) -> Arc<health::SchedulerHealthCollector> {
    Arc::new(health::SchedulerHealthCollector {
        start_time: std::time::Instant::now(),
        b_max,
        queue_max: admission_queue_max,
        model_name: model_id,
        max_position_embeddings: model_max_context as i32,
        b_active: scheduler_handle.b_active.clone(),
        b_queued: scheduler_handle.b_queued.clone(),
        admission_queue_full_count: scheduler_handle.admission_queue_full_count.clone(),
        memory_budget_exceeded_count: scheduler_handle.memory_budget_exceeded_count.clone(),
        kv_cache_active_bytes: scheduler_handle.kv_cache_active_bytes.clone(),
        kv_cache_soft_limit_bytes: scheduler_handle.kv_cache_soft_limit_bytes,
        mtp,
    })
}

/// GET /healthz — returns a JSON HealthSnapshot. Reads only Arc atomics;
/// no lock contention with the model or SchedulerActor. B1-p2.5 G3.
async fn healthz_handler<M>(
    axum::extract::State(state): axum::extract::State<AppState<M>>,
) -> axum::Json<health::HealthSnapshot>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    axum::Json(state.health_collector.snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::time::Duration;

    use mlx::{Array, Dtype, StreamOrDevice};
    use tokio::sync::mpsc;
    use tokio::time::sleep;

    use crate::nn::LayerCache;

    struct DefaultRouteModel;
    struct LimitedRouteModel;

    impl Model for DefaultRouteModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_on(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn batched_prefill(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_text_hidden(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl Model for LimitedRouteModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_on(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn batched_prefill(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_text_hidden(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn fresh_prefill_batch_limit(_prompt_len: usize, b_max: usize) -> usize
        where
            Self: Sized,
        {
            b_max.min(2)
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    #[test]
    fn route_keeps_unlimited_model_long_prompt_on_generation_stream() {
        assert!(!should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, false
        ));
    }

    #[test]
    fn route_uses_scheduler_for_model_limited_chunked_long_prompt() {
        assert!(should_route_to_scheduler::<LimitedRouteModel>(
            4096, 2048, 4, false
        ));
    }

    #[test]
    fn route_uses_scheduler_for_long_prompt_when_paged_prefix_cache_enabled() {
        assert!(should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, true,
        ));
    }

    fn test_scheduler_handle() -> scheduler_actor::SchedulerActorHandle {
        let (cmd_tx, _cmd_rx) = mpsc::channel(1);
        let queue_rejected = Arc::new(AtomicU64::new(0));
        scheduler_actor::SchedulerActorHandle {
            cmd_tx,
            admit_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
            saturate_triggered: Arc::new(AtomicU64::new(0)),
            queue_depth_peak: Arc::new(AtomicUsize::new(0)),
            queue_rejected: queue_rejected.clone(),
            mtp_prefill_count: Arc::new(AtomicU64::new(0)),
            mtp_step_count: Arc::new(AtomicU64::new(0)),
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: queue_rejected,
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
        }
    }

    #[test]
    fn health_collector_mtp_disabled_without_server_mtp_config() {
        let handle = test_scheduler_handle();
        let collector = build_health_collector(
            "test-model".to_string(),
            4096,
            1,
            8,
            &handle,
            health::MtpHealthConfig::disabled(),
        );
        let snapshot = collector.snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
    }

    #[test]
    fn health_collector_mtp_enabled_uses_scheduler_actor_counters() {
        let handle = test_scheduler_handle();
        handle.mtp_prefill_count.store(3, Ordering::Relaxed);
        handle.mtp_step_count.store(5, Ordering::Relaxed);
        let collector = build_health_collector(
            "test-model".to_string(),
            4096,
            1,
            8,
            &handle,
            health::MtpHealthConfig::enabled(
                2,
                handle.mtp_prefill_count.clone(),
                handle.mtp_step_count.clone(),
            ),
        );
        let snapshot = collector.snapshot();

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 3);
        assert_eq!(snapshot.mtp.step_count, 5);
    }

    /// Verify two concurrent task acquisitions of the same Mutex serialize.
    /// We don't construct a real Qwen35Model — Mutex<()> exhibits the same
    /// serialization semantics, and that's the load-bearing contract here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mutex_serializes_concurrent_acquirers() {
        let m = Arc::new(Mutex::new(()));
        let m1 = m.clone();
        let m2 = m.clone();

        let timeline: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let t1 = timeline.clone();
        let t2 = timeline.clone();

        let h1 = tokio::spawn(async move {
            let _g = m1.lock().await;
            t1.lock().await.push("1-start");
            sleep(Duration::from_millis(50)).await;
            t1.lock().await.push("1-end");
        });
        sleep(Duration::from_millis(5)).await; // ensure h1 grabs lock first
        let h2 = tokio::spawn(async move {
            let _g = m2.lock().await;
            t2.lock().await.push("2-start");
            t2.lock().await.push("2-end");
        });

        let _ = h1.await;
        let _ = h2.await;

        let tl = timeline.lock().await;
        assert_eq!(*tl, vec!["1-start", "1-end", "2-start", "2-end"]);
    }
}
