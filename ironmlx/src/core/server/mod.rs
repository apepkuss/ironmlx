//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

#[cfg(test)]
use std::sync::atomic::Ordering;
#[cfg(test)]
use std::sync::Arc;

use axum::{extract::State, routing::get, routing::post, Json, Router};
use serde::Serialize;
#[cfg(test)]
use tokio::sync::Mutex;

use crate::core::cache::{
    ActiveKvOffloadConfig, PagedPrefixCacheConfig, PrefixLruCacheConfig, TurboQuantKVBits,
};
use crate::core::model::Model;
#[cfg(test)]
use crate::core::sampler::Sampler;
use crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile;
use crate::core::speculative_model::MtpSpeculativeModel;
use crate::core::tokenizer::Tokenizer;
use crate::core::vision::DenseVlMethods;
use crate::Result;

pub mod anthropic;
pub(crate) mod api_error;
pub(crate) mod api_transport;
pub mod chat_format;
pub mod diffusion_gemma;
pub mod engine;
pub mod health;
pub mod model_manager;
pub(crate) mod openai;
pub(crate) mod responses;
pub use crate::core::scheduler_actor;
#[cfg(test)]
mod scheduler_disconnect_tests;
pub mod security;
pub(crate) mod structured_output;
pub mod vision;

pub(crate) use crate::core::task_execution::RequestAdmissionError;
pub use crate::core::task_execution::RequestExecutionHandle;

pub use crate::core::runtime_config::SamplingDefaults;

pub use crate::core::vision_input::VisionInputConfig;

pub use crate::core::engine_state::CausalEngine as AppState;
pub(crate) use crate::core::engine_state::Gemma4DrafterEngine as Gemma4DrafterAppState;
use crate::core::engine_state::*;

#[allow(clippy::too_many_arguments)]
pub async fn serve<M>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize, // 3f
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    force_scheduler: bool,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
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
        static_memory_estimate,
        None,
        PlainSchedulerActorSpawner {
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
            force_scheduler,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_prompt_lookup<M>(
    model: M,
    cfg: crate::core::prompt_lookup::PromptLookupConfig,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let qualification =
        crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
        )?;
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
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
        static_memory_estimate,
        None,
        PromptLookupSchedulerActorSpawner {
            cfg,
            qualification,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_mtp<M>(
    model: M,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let prompt_lookup = prompt_lookup
        .map(|cfg| -> Result<_> {
            let qualification = crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile_with_baseline(
                &scheduler_runtime_profile,
                crate::core::prompt_lookup::PromptLookupQualificationBaseline::QwenMtp,
            )?;
            Ok((cfg.validate()?, qualification))
        })
        .transpose()?;
    let exact_qualification =
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
            crate::core::speculative_qualification::NeuralExactSource::QwenMtp,
        )?;
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
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
        static_memory_estimate,
        Some(MtpHealthDraftTokens {
            requested: mtp_draft_tokens,
            effective: mtp_draft_tokens,
        }),
        MtpSchedulerActorSpawner {
            mtp,
            mtp_draft_tokens,
            exact_qualification,
            prompt_lookup,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_gemma4_drafter(
    model: crate::models::Gemma4Model,
    drafter: crate::models::gemma4::Gemma4AssistantModel,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()> {
    let state = build_gemma4_drafter_app_state(
        model,
        drafter,
        mtp_draft_tokens,
        prompt_lookup,
        tokenizer,
        model_id,
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
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        active_kv_offload,
    )
    .await?;

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(gemma4_drafter_healthz_handler))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(gemma4_drafter_clear_prompt_lookup_handler),
        )
        .route(
            "/v1/chat/completions",
            post(openai::gemma4_drafter_chat_completions),
        )
        .route("/v1/responses", post(responses::gemma4_drafter_responses))
        .route("/v1/messages", post(anthropic::gemma4_drafter_messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx Gemma4 drafter server").await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn serve_with_dflash2<M>(
    model: M,
    draft: crate::models::DFlash2DraftModel,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    tensor_batch_max_width: usize,
    admission_queue_max: usize,
    max_cache_cap: usize,
    block_size: usize,
    draft_quantization_bits: Option<i32>,
    prefix_cache: Option<crate::core::cache::PrefixLruCacheConfig>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + crate::models::dflash2::DFlash2Target + Send + 'static,
{
    let state = build_dflash2_engine(
        model,
        draft,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        tensor_batch_max_width,
        admission_queue_max,
        max_cache_cap,
        block_size,
        draft_quantization_bits,
        prefix_cache,
        scheduler_runtime_profile,
        static_memory_estimate,
    )
    .await?;
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route("/v1/models", get(dflash2_models_handler::<M>))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(clear_prompt_lookup_handler),
        )
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx DFlash2 server").await
}

#[derive(Debug, Serialize, PartialEq)]
struct DFlash2ModelList {
    object: &'static str,
    data: Vec<DFlash2ModelInfo>,
}

#[derive(Debug, Serialize, PartialEq)]
struct DFlash2ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

fn dflash2_model_list(model_id: &str) -> DFlash2ModelList {
    DFlash2ModelList {
        object: "list",
        data: vec![DFlash2ModelInfo {
            id: model_id.to_owned(),
            object: "model",
            created: 0,
            owned_by: "ironmlx",
        }],
    }
}

async fn dflash2_models_handler<M>(State(state): State<AppState<M>>) -> Json<DFlash2ModelList>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    Json(dflash2_model_list(&state.model_id))
}

#[allow(clippy::too_many_arguments)]
async fn serve_inner<M, S>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
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
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    mtp_health_draft_tokens: Option<MtpHealthDraftTokens>,
    scheduler_actor_spawner: S,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
    S: SchedulerActorSpawner<M>,
{
    let state = build_app_state(
        model,
        tokenizer,
        model_id,
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
        static_memory_estimate,
        mtp_health_draft_tokens,
        scheduler_actor_spawner,
    )
    .await?;
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(clear_prompt_lookup_handler),
        )
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx server").await
}

/// GET /healthz — returns a JSON HealthSnapshot. Reads only Arc atomics;
/// no lock contention with the model or SchedulerActor. B1-p2.5 G3.
#[derive(Debug, Serialize)]
struct SingleActorHealthResponse<T> {
    #[serde(flatten)]
    health: T,
    mode: &'static str,
    models: [(); 0],
}

impl<T> SingleActorHealthResponse<T> {
    fn new(health: T) -> Self {
        Self {
            health,
            mode: "single",
            models: [],
        }
    }
}

async fn healthz_handler<M>(
    axum::extract::State(state): axum::extract::State<AppState<M>>,
) -> axum::Json<SingleActorHealthResponse<health::HealthSnapshot>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    axum::Json(SingleActorHealthResponse::new(
        state.health_collector.snapshot(),
    ))
}

async fn gemma4_drafter_healthz_handler(
    axum::extract::State(state): axum::extract::State<Gemma4DrafterAppState>,
) -> axum::Json<SingleActorHealthResponse<health::HealthSnapshot>> {
    axum::Json(SingleActorHealthResponse::new(
        state.base.health_collector.snapshot(),
    ))
}

async fn clear_prompt_lookup_handler<M>(
    axum::extract::State(state): axum::extract::State<AppState<M>>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>)
where
    M: Model + DenseVlMethods + Send + 'static,
{
    clear_prompt_lookup_response(&state.request_execution, &state.model_id).await
}

async fn gemma4_drafter_clear_prompt_lookup_handler(
    axum::extract::State(state): axum::extract::State<Gemma4DrafterAppState>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    clear_prompt_lookup_response(&state.base.request_execution, &state.base.model_id).await
}

async fn clear_prompt_lookup_response(
    request_execution: &RequestExecutionHandle,
    model_id: &str,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    match request_execution.clear_shared_prompt_lookup().await {
        Ok(cleared_entries) => (
            axum::http::StatusCode::OK,
            axum::Json(serde_json::json!({
                "success": true,
                "status": "cleared",
                "model": model_id,
                "cleared_models": 1,
                "cleared_entries": cleared_entries,
            })),
        ),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "success": false,
                "status": "error",
                "model": model_id,
                "cleared_models": 0,
                "cleared_entries": 0,
                "error": error.to_string(),
            })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::time::Duration;

    use mlx::{Array, Dtype, StreamOrDevice};
    use tokio::sync::mpsc;
    use tokio::time::sleep;

    use crate::core::cache::layer::LayerCache;

    #[test]
    fn effective_model_weight_bytes_uses_loaded_tensor_bytes_when_larger() {
        assert_eq!(effective_model_weight_bytes(1_024, 4_096), 4_096);
    }

    #[test]
    fn effective_model_weight_bytes_keeps_meta_estimate_when_larger() {
        assert_eq!(effective_model_weight_bytes(4_096, 1_024), 4_096);
    }

    #[test]
    fn dflash2_model_list_exposes_only_the_public_target_identifier() {
        let list = dflash2_model_list("mlx-community/Qwen3.8-27B-4bit");
        let json = serde_json::to_value(list).expect("serialize model list");

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["data"][0]["id"], "mlx-community/Qwen3.8-27B-4bit");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "ironmlx");
    }

    #[test]
    fn single_actor_health_response_exposes_app_runtime_contract() {
        let response = SingleActorHealthResponse::new(serde_json::json!({
            "status": "healthy",
            "model": {"name": "test-model"},
        }));
        let json = serde_json::to_value(response).expect("serialize health response");

        assert_eq!(json["status"], "healthy");
        assert_eq!(json["model"]["name"], "test-model");
        assert_eq!(json["mode"], "single");
        assert_eq!(json["models"], serde_json::json!([]));
    }

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

        fn model_meta(&self) -> crate::core::model::ModelMeta {
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

        fn model_meta(&self) -> crate::core::model::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    #[test]
    fn route_keeps_unlimited_model_long_prompt_on_generation_stream() {
        assert!(!should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, false, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_model_limited_chunked_long_prompt() {
        assert!(should_route_to_scheduler::<LimitedRouteModel>(
            4096, 2048, 4, false, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_long_prompt_when_paged_prefix_cache_enabled() {
        assert!(should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, true, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_long_prompt_when_greedy_scheduler_is_forced() {
        assert!(should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 1, false, true,
        ));
    }

    #[test]
    fn prompt_lookup_sampler_validation_accepts_exact_non_greedy() {
        assert!(validate_prompt_lookup_sampler(true, Sampler::greedy()).is_ok());
        assert!(
            validate_prompt_lookup_sampler(false, Sampler::greedy().with_temperature(0.7)).is_ok()
        );
        assert!(
            validate_prompt_lookup_sampler(true, Sampler::greedy().with_temperature(0.7)).is_ok()
        );
        let sampler = Sampler {
            temperature: f32::NAN,
            ..Sampler::greedy()
        };
        assert_eq!(
            validate_prompt_lookup_sampler(true, sampler)
                .unwrap_err()
                .to_string(),
            "sampling temperature must be finite"
        );
    }

    fn test_scheduler_handle() -> scheduler_actor::SchedulerActorHandle {
        let (cmd_tx, _cmd_rx) = mpsc::channel(1);
        let (control_tx, _control_rx) = mpsc::channel(1);
        let queue_rejected = Arc::new(AtomicU64::new(0));
        scheduler_actor::SchedulerActorHandle {
            cmd_tx,
            control_tx,
            cold_materialization_tracker: Arc::new(std::sync::OnceLock::new()),
            runtime_usage: Arc::new(
                crate::core::runtime_usage::ModelRuntimeUsageCounters::default(),
            ),
            admit_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
            saturate_triggered: Arc::new(AtomicU64::new(0)),
            queue_depth_peak: Arc::new(AtomicUsize::new(0)),
            queue_rejected: queue_rejected.clone(),
            mtp_prefill_count: Arc::new(AtomicU64::new(0)),
            mtp_step_count: Arc::new(AtomicU64::new(0)),
            mtp_fallback_prefill_count: Arc::new(AtomicU64::new(0)),
            mtp_drafted_tokens: Arc::new(AtomicU64::new(0)),
            mtp_accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
            mtp_windows: Arc::new(AtomicU64::new(0)),
            mtp_multi_token_windows: Arc::new(AtomicU64::new(0)),
            mtp_exact_sampling_windows: Arc::new(AtomicU64::new(0)),
            mtp_exact_acceptance_draws: Arc::new(AtomicU64::new(0)),
            mtp_exact_residual_corrections: Arc::new(AtomicU64::new(0)),
            mtp_exact_bonus_samples: Arc::new(AtomicU64::new(0)),
            mtp_draft_forward_us: Arc::new(AtomicU64::new(0)),
            mtp_verify_forward_us: Arc::new(AtomicU64::new(0)),
            mtp_projection_us: Arc::new(AtomicU64::new(0)),
            mtp_sampling_us: Arc::new(AtomicU64::new(0)),
            mtp_draft_host_sync_count: Arc::new(AtomicU64::new(0)),
            mtp_draft_host_sync_us: Arc::new(AtomicU64::new(0)),
            mtp_verify_accept_host_sync_count: Arc::new(AtomicU64::new(0)),
            mtp_verify_accept_host_sync_us: Arc::new(AtomicU64::new(0)),
            mtp_main_rollback_us: Arc::new(AtomicU64::new(0)),
            mtp_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_prefill_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_decode_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_cache_restore_us: Arc::new(AtomicU64::new(0)),
            prompt_lookup_published_stats: Arc::new(std::sync::Mutex::new(None)),
            neural_exact_qualification_stats: Arc::new(std::sync::Mutex::new(
                crate::core::speculative_qualification::NeuralExactQualificationStats::default(),
            )),
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: queue_rejected,
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
            kv_cache_logical_cap_tokens: 1,
            kv_cache_resident_cap_tokens: 1,
            kv_cache_budget_policy: "full_resident",
            active_kv_offload: crate::core::cache::ActiveKvOffloadSharedStats::new(
                &crate::core::cache::ActiveKvOffloadConfig::disabled(),
            ),
            immutable_prefix_blocks: scheduler_actor::ImmutablePrefixBlockSharedStats::new(false),
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
            health::PromptLookupHealthConfig::disabled(),
        );
        let snapshot = collector.snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 0);
        assert_eq!(snapshot.mtp.drafted_tokens, 0);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 0);
    }

    #[test]
    fn health_collector_mtp_enabled_uses_scheduler_actor_counters() {
        let handle = test_scheduler_handle();
        handle.mtp_prefill_count.store(3, Ordering::Relaxed);
        handle.mtp_step_count.store(5, Ordering::Relaxed);
        handle
            .mtp_fallback_prefill_count
            .store(7, Ordering::Relaxed);
        handle.mtp_drafted_tokens.store(11, Ordering::Relaxed);
        handle
            .mtp_accepted_draft_tokens
            .store(13, Ordering::Relaxed);
        handle.mtp_windows.store(17, Ordering::Relaxed);
        handle.mtp_multi_token_windows.store(13, Ordering::Relaxed);
        handle.mtp_draft_forward_us.store(19, Ordering::Relaxed);
        handle.mtp_verify_forward_us.store(23, Ordering::Relaxed);
        handle.mtp_projection_us.store(29, Ordering::Relaxed);
        handle.mtp_sampling_us.store(31, Ordering::Relaxed);
        handle.mtp_main_rollback_us.store(37, Ordering::Relaxed);
        handle.mtp_cache_commit_us.store(41, Ordering::Relaxed);
        handle
            .mtp_prefill_cache_commit_us
            .store(17, Ordering::Relaxed);
        handle
            .mtp_decode_cache_commit_us
            .store(24, Ordering::Relaxed);
        handle.mtp_cache_restore_us.store(43, Ordering::Relaxed);
        let collector = build_health_collector(
            "test-model".to_string(),
            4096,
            1,
            8,
            &handle,
            health::MtpHealthConfig::enabled(
                2,
                2,
                handle.mtp_prefill_count.clone(),
                handle.mtp_step_count.clone(),
                handle.mtp_fallback_prefill_count.clone(),
                handle.mtp_drafted_tokens.clone(),
                handle.mtp_accepted_draft_tokens.clone(),
                handle.mtp_windows.clone(),
                handle.mtp_multi_token_windows.clone(),
                handle.mtp_exact_sampling_windows.clone(),
                handle.mtp_exact_acceptance_draws.clone(),
                handle.mtp_exact_residual_corrections.clone(),
                handle.mtp_exact_bonus_samples.clone(),
                handle.mtp_draft_forward_us.clone(),
                handle.mtp_verify_forward_us.clone(),
                handle.mtp_projection_us.clone(),
                handle.mtp_sampling_us.clone(),
                handle.mtp_draft_host_sync_count.clone(),
                handle.mtp_draft_host_sync_us.clone(),
                handle.mtp_verify_accept_host_sync_count.clone(),
                handle.mtp_verify_accept_host_sync_us.clone(),
                handle.mtp_main_rollback_us.clone(),
                handle.mtp_cache_commit_us.clone(),
                handle.mtp_prefill_cache_commit_us.clone(),
                handle.mtp_decode_cache_commit_us.clone(),
                handle.mtp_cache_restore_us.clone(),
                handle.neural_exact_qualification_stats.clone(),
            ),
            health::PromptLookupHealthConfig::disabled(),
        );
        let snapshot = collector.snapshot();

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 3);
        assert_eq!(snapshot.mtp.step_count, 5);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 7);
        assert_eq!(snapshot.mtp.drafted_tokens, 11);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 13);
        assert_eq!(snapshot.mtp.windows, 17);
        assert_eq!(snapshot.mtp.multi_token_windows, 13);
        assert_eq!(snapshot.mtp.draft_forward_us, 19);
        assert_eq!(snapshot.mtp.verify_forward_us, 23);
        assert_eq!(snapshot.mtp.projection_us, 29);
        assert_eq!(snapshot.mtp.sampling_us, 31);
        assert_eq!(snapshot.mtp.main_rollback_us, 37);
        assert_eq!(snapshot.mtp.cache_commit_us, 41);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 17);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 24);
        assert_eq!(snapshot.mtp.cache_restore_us, 43);
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

pub mod image_input;
