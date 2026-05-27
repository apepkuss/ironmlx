//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::{routing::get, routing::post, Router};
use tokio::sync::Mutex;

use crate::core::model::Model;
use crate::core::scheduler::DenseVlMethods;
use crate::core::tokenizer::Tokenizer;
use crate::Result;

pub mod anthropic;
pub mod chat_format;
pub mod health;
mod openai;
pub mod scheduler_actor;

/// HTTP server shared state. The model is wrapped in a tokio Mutex —
/// concurrent requests serialize behind the lock (P4 single-stream contract).
///
/// 3b-2 adds `scheduler_handle`; short prompts, including VL prompts, route
/// through the SchedulerActor, while long prompts fall back to GenerationStream.
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
    /// SchedulerActor handle. Routed to by short-prompt requests. See
    /// `serve_via_scheduler_*` in `openai.rs`.
    pub scheduler_handle: scheduler_actor::SchedulerActorHandle,
    /// Maximum concurrent in-flight requests routed to the SchedulerActor.
    pub b_max: usize,
    /// Admission-window deadline (milliseconds) — drain-window timeout.
    pub admission_deadline_ms: u64,
    /// FIFO admission queue capacity.
    pub admission_queue_max: usize,
    /// Effective cap_max = min(--max-cache-cap CLI flag, model.config.max_position_embeddings).
    /// Per-request `prompt_len + max_new_tokens` exceeding this returns HTTP 413. B1-p2.3f.
    pub effective_cap_max: usize,
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
            scheduler_handle: self.scheduler_handle.clone(),
            b_max: self.b_max,
            admission_deadline_ms: self.admission_deadline_ms,
            admission_queue_max: self.admission_queue_max,
            effective_cap_max: self.effective_cap_max,
            health_collector: self.health_collector.clone(),
        }
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
    max_cache_cap: usize,              // 3f
    p5h_measurement_eval_probes: bool, // P5h+1 T1
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // P5h+1 T1: install the measurement-eval-probes flag in the global
    // BEFORE the SchedulerActor / GenerationStream code paths run. Setter
    // is feature-gated; feature-off builds discard the boolean to keep the
    // CLI plumbing uniform without requiring `#[cfg]` at the call site.
    #[cfg(feature = "p5h-profile")]
    crate::core::p5h::set_measurement_eval_probes_active(p5h_measurement_eval_probes);
    #[cfg(not(feature = "p5h-profile"))]
    let _ = p5h_measurement_eval_probes;

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

    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        meta,
    )?;

    // B1-p2.5 G3: Build SchedulerHealthCollector from shared Arc atomics
    // exposed by SchedulerActorHandle. max_position_embeddings already resolved
    // into model_max_context above (i32 → usize). Re-read from model_max_context.
    let health_collector = Arc::new(health::SchedulerHealthCollector {
        start_time: std::time::Instant::now(),
        b_max,
        queue_max: admission_queue_max,
        model_name: model_id.clone(),
        max_position_embeddings: model_max_context as i32,
        b_active: scheduler_handle.b_active.clone(),
        b_queued: scheduler_handle.b_queued.clone(),
        admission_queue_full_count: scheduler_handle.admission_queue_full_count.clone(),
        memory_budget_exceeded_count: scheduler_handle.memory_budget_exceeded_count.clone(),
        kv_cache_active_bytes: scheduler_handle.kv_cache_active_bytes.clone(),
        kv_cache_soft_limit_bytes: scheduler_handle.kv_cache_soft_limit_bytes,
    });

    let state = AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        scheduler_handle,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        effective_cap_max, // 3f
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
    use std::time::Duration;
    use tokio::time::sleep;

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
