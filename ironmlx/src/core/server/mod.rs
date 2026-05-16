//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::{routing::get, routing::post, Router};
use tokio::sync::Mutex;

use crate::core::tokenizer::Tokenizer;
use crate::models::Qwen35Model;
use crate::Result;

pub mod anthropic;
pub mod chat_format;
mod openai;
pub mod scheduler_actor;

#[derive(Clone)]
/// HTTP server shared state. The model is wrapped in a tokio Mutex —
/// concurrent requests serialize behind the lock (P4 single-stream contract).
///
/// 3b-2 adds `scheduler_handle` so text-only short-prompt requests can be
/// routed through the SchedulerActor; VL / long-prompt requests still
/// take the GenerationStream path that holds the model lock directly.
pub struct AppState {
    pub model: Arc<Mutex<Qwen35Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    /// Default prefill chunk size (max tokens per prefill forward). `0`
    /// disables chunking. Applied to every `GenerateRequest` constructed
    /// by the request handlers.
    pub prefill_chunk_size: usize,
    /// SchedulerActor handle. Routed to by text-only short-prompt
    /// requests. See `serve_via_scheduler_*` in `openai.rs`.
    pub scheduler_handle: scheduler_actor::SchedulerActorHandle,
    /// Maximum concurrent in-flight requests routed to the SchedulerActor.
    pub b_max: usize,
    /// Admission-window deadline (milliseconds) — drain-window timeout.
    pub admission_deadline_ms: u64,
    /// FIFO admission queue capacity.
    pub admission_queue_max: usize,
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    model: Qwen35Model,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
) -> Result<()> {
    let model = Arc::new(Mutex::new(model));
    let admission_deadline = std::time::Duration::from_millis(admission_deadline_ms);
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
    );
    let state = AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        scheduler_handle,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
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
