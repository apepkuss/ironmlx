use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::state::{AppState, LogEntry};
use ironmlx_core::generate::SamplerConfig;

#[derive(Serialize)]
pub struct SettingsResponse {
    host: String,
    port: u16,
    memory_limit_gb: f64,
    cache_max_size_gb: f64,
    max_num_seqs: usize,
    temperature: f32,
    top_p: f32,
    api_key_set: bool,
    log_level: String,
}

#[derive(Deserialize)]
pub struct UpdateSettingsRequest {
    pub memory_limit_gb: Option<f64>,
    pub cache_max_size_gb: Option<f64>,
    pub max_num_seqs: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub api_key: Option<String>,
    pub log_level: Option<String>,
}

pub async fn get_settings(State(state): State<Arc<AppState>>) -> Json<SettingsResponse> {
    let config = state.config.read().unwrap();
    Json(SettingsResponse {
        host: config.host.clone(),
        port: config.port,
        memory_limit_gb: config.memory_limit_gb,
        cache_max_size_gb: config.cache_max_size_gb,
        max_num_seqs: config.max_num_seqs,
        temperature: config.temperature,
        top_p: config.top_p,
        api_key_set: config.api_key.is_some(),
        log_level: config.log_level.clone(),
    })
}

pub async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateSettingsRequest>,
) -> StatusCode {
    let mut config = state.config.write().unwrap();
    if let Some(v) = req.temperature {
        config.temperature = v;
    }
    if let Some(v) = req.top_p {
        config.top_p = v;
    }
    if let Some(v) = req.memory_limit_gb {
        config.memory_limit_gb = v;
    }
    if let Some(v) = req.cache_max_size_gb {
        config.cache_max_size_gb = v;
    }
    if let Some(v) = req.max_num_seqs {
        config.max_num_seqs = v;
    }
    if let Some(ref v) = req.api_key {
        config.api_key = if v.is_empty() { None } else { Some(v.clone()) };
    }
    if let Some(ref v) = req.log_level {
        config.log_level = v.clone();
    }
    StatusCode::OK
}

#[derive(Deserialize)]
pub struct AuthRequest {
    pub api_key: String,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub success: bool,
}

pub async fn auth(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AuthRequest>,
) -> Json<AuthResponse> {
    let config = state.config.read().unwrap();
    let success = match &config.api_key {
        Some(key) => req.api_key == *key,
        None => true,
    };
    Json(AuthResponse { success })
}

// ── Logs ─────────────────────────────────────────────────────────────────────

/// GET /admin/api/logs — return the in-memory log buffer as JSON.
pub async fn get_logs(State(state): State<Arc<AppState>>) -> Json<Vec<LogEntry>> {
    Json(state.log_buffer.snapshot())
}

// ── Benchmark ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct BenchmarkRequest {
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize)]
pub struct BenchmarkResponse {
    pub model: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub tokens_per_sec: f64,
}

/// POST /admin/api/benchmark — run a benchmark and return timing results.
pub async fn run_benchmark(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BenchmarkRequest>,
) -> Result<Json<BenchmarkResponse>, (StatusCode, String)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let prompt = req
        .prompt
        .unwrap_or_else(|| "Explain the theory of relativity.".to_string());
    let max_tokens = req.max_tokens.unwrap_or(50);

    let tokens = entry
        .tokenizer
        .encode(&prompt)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let prompt_len = tokens.len();

    let request_id = format!("bench_{}", uuid::Uuid::new_v4().simple());
    let sampler = SamplerConfig::greedy();

    state.log_buffer.push(
        "info",
        &format!(
            "Benchmark started: model={}, prompt_tokens={}, max_tokens={}",
            entry.model_id, prompt_len, max_tokens
        ),
    );

    let t0 = std::time::Instant::now();
    let mut token_rx = entry
        .engine
        .add_request(
            request_id,
            tokens,
            sampler,
            max_tokens,
            entry.eos_token_id,
            None,
        )
        .await;

    let mut count = 0usize;
    let mut ttft: Option<f64> = None;

    while let Some(output) = token_rx.recv().await {
        count += 1;
        if ttft.is_none() {
            ttft = Some(t0.elapsed().as_secs_f64() * 1000.0);
        }
        if output.finish_reason.is_some() {
            break;
        }
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.unwrap_or(0.0);
    let decode_ms = total_ms - ttft_ms;
    let tps = if decode_ms > 0.0 && count > 1 {
        (count - 1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    state.log_buffer.push(
        "info",
        &format!(
            "Benchmark done: {} tokens, {:.0}ms, {:.1} tok/s",
            count, total_ms, tps
        ),
    );

    Ok(Json(BenchmarkResponse {
        model: entry.model_id.clone(),
        prompt_tokens: prompt_len,
        generated_tokens: count,
        ttft_ms,
        total_ms,
        tokens_per_sec: tps,
    }))
}
