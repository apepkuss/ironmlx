//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

#[cfg(test)]
use std::sync::Arc;

use axum::{extract::State, routing::get, routing::post, Json, Router};
use serde::Serialize;
#[cfg(test)]
use tokio::sync::Mutex;

use crate::Result;
#[cfg(test)]
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::speculative_model::MtpSpeculativeModel;
use ironmlx_lm::core::tokenizer::Tokenizer;
use ironmlx_lm::core::vision::DenseVlMethods;
use ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile;
use {
    ironmlx_lm::core::cache::turboquant_kv::TurboQuantKVBits,
    ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig,
    ironmlx_runtime::core::cache::prefix_store::PagedPrefixCacheConfig,
    ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig,
};

pub mod anthropic;
pub(crate) mod api_error;
pub(crate) mod api_transport;
pub(crate) mod audio;
pub mod chat_format;
pub mod diffusion_gemma;
pub mod engine;
pub(crate) mod systemone;
use ironmlx_runtime::core::runtime_health as health;
pub mod model_manager;
pub(crate) mod openai;
pub(crate) mod responses;
#[cfg(test)]
mod scheduler_disconnect_tests;
pub mod security;
pub(crate) mod structured_output;
pub mod vision;
pub(crate) mod voices;

pub(crate) use ironmlx_runtime::core::task_execution::RequestAdmissionError;
use ironmlx_runtime::core::task_execution::RequestExecutionHandle;

use ironmlx_lm::core::vision_input::VisionInputConfig;

pub use ironmlx_runtime::core::engine_state::CausalEngine as AppState;
pub(crate) use ironmlx_runtime::core::engine_state::Gemma4DrafterEngine as Gemma4DrafterAppState;
use ironmlx_runtime::core::engine_state::*;

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
    static_memory_estimate: ironmlx_runtime::core::process_memory::StaticMemoryEstimate,
    force_scheduler: bool,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let state = build_plain_app_state_with_force_scheduler(
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
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        active_kv_offload,
        force_scheduler,
    )
    .await?;
    serve_inner(state, network_config).await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_prompt_lookup<M>(
    model: M,
    cfg: ironmlx_runtime::core::prompt_lookup::PromptLookupConfig,
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
    static_memory_estimate: ironmlx_runtime::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let state = build_prompt_lookup_app_state(
        model,
        cfg,
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
    serve_inner(state, network_config).await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_mtp<M>(
    model: M,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<ironmlx_runtime::core::prompt_lookup::PromptLookupConfig>,
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
    static_memory_estimate: ironmlx_runtime::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let state = build_mtp_app_state(
        model,
        mtp,
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
    serve_inner(state, network_config).await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_gemma4_drafter(
    model: ironmlx_lm::models::Gemma4Model,
    drafter: ironmlx_lm::models::gemma4::Gemma4AssistantModel,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<ironmlx_runtime::core::prompt_lookup::PromptLookupConfig>,
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
    static_memory_estimate: ironmlx_runtime::core::process_memory::StaticMemoryEstimate,
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
    draft: ironmlx_lm::models::DFlash2DraftModel,
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
    prefix_cache: Option<ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    static_memory_estimate: ironmlx_runtime::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + ironmlx_lm::models::dflash2::DFlash2Target + Send + 'static,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

fn dflash2_model_list(model_id: &str, effective_cap_max: usize) -> DFlash2ModelList {
    let capacity = (effective_cap_max > 0).then_some(effective_cap_max);
    DFlash2ModelList {
        object: "list",
        data: vec![DFlash2ModelInfo {
            id: model_id.to_owned(),
            context_window: capacity,
            max_output_tokens: capacity,
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
    Json(dflash2_model_list(&state.model_id, state.effective_cap_max))
}

async fn serve_inner<M>(
    state: AppState<M>,
    network_config: security::ServerNetworkConfig,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
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
    use std::time::Duration;

    use mlx::{Array, Dtype, StreamOrDevice};
    use tokio::time::sleep;

    use ironmlx_lm::core::cache::layer::LayerCache;

    #[test]
    fn dflash2_model_list_exposes_only_the_public_target_identifier() {
        let list = dflash2_model_list("mlx-community/Qwen3.8-27B-4bit", 65536);
        let json = serde_json::to_value(list).expect("serialize model list");

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["data"][0]["id"], "mlx-community/Qwen3.8-27B-4bit");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "ironmlx");
        assert_eq!(json["data"][0]["context_window"], 65536);
        assert_eq!(json["data"][0]["max_output_tokens"], 65536);
        let unknown = serde_json::to_value(dflash2_model_list("unknown", 0)).unwrap();
        assert!(unknown["data"][0].get("context_window").is_none());
        assert!(unknown["data"][0].get("max_output_tokens").is_none());
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

        fn model_meta(&self) -> ironmlx_lm::core::model::ModelMeta {
            ironmlx_runtime::core::memory_budget::test_meta_qwen35()
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

        fn model_meta(&self) -> ironmlx_lm::core::model::ModelMeta {
            ironmlx_runtime::core::memory_budget::test_meta_qwen35()
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
