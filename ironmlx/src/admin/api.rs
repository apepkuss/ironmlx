use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::state::{AppState, BenchmarkResult, DownloadStatus, LogEntry};
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
    hf_endpoint: String,
    chat_template_override: Option<String>,
    model_aliases: HashMap<String, String>,
    log_buffer_size: usize,
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
    pub hf_endpoint: Option<String>,
    pub chat_template_override: Option<String>,
    pub model_aliases: Option<HashMap<String, String>>,
    pub log_buffer_size: Option<usize>,
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
        hf_endpoint: config.hf_endpoint.clone(),
        chat_template_override: config.chat_template_override.clone(),
        model_aliases: config.model_aliases.clone(),
        log_buffer_size: config.log_buffer_size,
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
    if let Some(ref v) = req.hf_endpoint {
        config.hf_endpoint = v.clone();
    }
    if let Some(ref v) = req.chat_template_override {
        config.chat_template_override = if v.is_empty() { None } else { Some(v.clone()) };
    }
    if let Some(ref v) = req.model_aliases {
        config.model_aliases = v.clone();
    }
    let new_log_cap = if let Some(v) = req.log_buffer_size {
        config.log_buffer_size = v;
        Some(v)
    } else {
        None
    };
    drop(config);
    if let Some(cap) = new_log_cap {
        state.log_buffer.set_capacity(cap);
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
    let aliases = {
        let config = state.config.read().unwrap();
        config.model_aliases.clone()
    };
    let entry = state
        .pool
        .get_with_aliases(req.model.as_deref(), &aliases)
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

    // Store result in history
    {
        let result = BenchmarkResult {
            timestamp: chrono::Utc::now().timestamp(),
            model: entry.model_id.clone(),
            prompt_tokens: prompt_len,
            gen_tokens: count,
            ttft_ms,
            tok_per_sec: tps,
            total_ms,
        };
        state.benchmark_history.lock().unwrap().push(result);
    }

    Ok(Json(BenchmarkResponse {
        model: entry.model_id.clone(),
        prompt_tokens: prompt_len,
        generated_tokens: count,
        ttft_ms,
        total_ms,
        tokens_per_sec: tps,
    }))
}

/// GET /admin/api/benchmark/history — return all benchmark results.
pub async fn get_benchmark_history(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<BenchmarkResult>> {
    Json(state.benchmark_history.lock().unwrap().clone())
}

/// DELETE /admin/api/benchmark/history — clear benchmark history.
pub async fn clear_benchmark_history(State(state): State<Arc<AppState>>) -> StatusCode {
    state.benchmark_history.lock().unwrap().clear();
    StatusCode::OK
}

// ── HuggingFace Model Search & Download ─────────────────────────────────────

#[derive(Deserialize)]
pub struct HfSearchRequest {
    pub query: String,
}

#[derive(Serialize, Deserialize)]
pub struct HfModelInfo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// POST /admin/api/models/search — search HuggingFace for MLX models.
pub async fn hf_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HfSearchRequest>,
) -> Result<Json<Vec<HfModelInfo>>, (StatusCode, String)> {
    let endpoint = {
        let config = state.config.read().unwrap();
        config.hf_endpoint.clone()
    };

    let url = format!(
        "{}/api/models?search={}&filter=mlx&limit=10&sort=downloads",
        endpoint,
        urlencoding::encode(&req.query)
    );

    let resp = reqwest::get(&url).await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("HF API request failed: {e}"),
        )
    })?;

    let models: Vec<HfModelInfo> = resp
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("HF API parse failed: {e}")))?;

    Ok(Json(models))
}

#[derive(Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
}

/// POST /admin/api/models/download — start downloading a HuggingFace model.
pub async fn hf_download(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HfDownloadRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let repo_id = req.repo_id.clone();

    // Check if already downloading
    {
        let downloads = state.downloads.lock().unwrap();
        if let Some(status) = downloads.get(&repo_id)
            && status.status == "downloading"
        {
            return Err((
                StatusCode::CONFLICT,
                format!("{} is already downloading", repo_id),
            ));
        }
    }

    // Set initial status
    {
        let mut downloads = state.downloads.lock().unwrap();
        downloads.insert(
            repo_id.clone(),
            DownloadStatus {
                repo_id: repo_id.clone(),
                status: "downloading".to_string(),
                progress_pct: 0.0,
                error: None,
            },
        );
    }

    let endpoint = {
        let config = state.config.read().unwrap();
        config.hf_endpoint.clone()
    };

    state
        .log_buffer
        .push("info", &format!("Starting download: {}", repo_id));

    // Spawn background download thread
    let state_clone = Arc::clone(&state);
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            let models_dir = crate::config::ironmlx_root().join("models");
            let _ = std::fs::create_dir_all(&models_dir);
            let api = hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(models_dir)
                .with_endpoint(endpoint.clone())
                .build()
                .map_err(|e| format!("Failed to build HF API: {e}"))?;

            let repo = api.model(repo_id.clone());

            // Download the repo info to get file list
            let info_url = format!("{}/api/models/{}", endpoint, repo_id);
            let resp = reqwest::blocking::get(&info_url)
                .map_err(|e| format!("Failed to fetch model info: {e}"))?;
            let info: serde_json::Value = resp
                .json()
                .map_err(|e| format!("Failed to parse model info: {e}"))?;

            let siblings = info
                .get("siblings")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "No files found in model repo".to_string())?;

            let total_files = siblings.len();
            for (i, sibling) in siblings.iter().enumerate() {
                let filename = sibling
                    .get("rfilename")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if filename.is_empty() {
                    continue;
                }

                repo.get(filename)
                    .map_err(|e| format!("Failed to download {}: {e}", filename))?;

                // Update progress
                let pct = ((i + 1) as f32 / total_files as f32) * 100.0;
                let mut downloads = state_clone.downloads.lock().unwrap();
                if let Some(status) = downloads.get_mut(&repo_id) {
                    status.progress_pct = pct;
                }
            }

            Ok(())
        })();

        let mut downloads = state_clone.downloads.lock().unwrap();
        if let Some(status) = downloads.get_mut(&repo_id) {
            match result {
                Ok(()) => {
                    status.status = "completed".to_string();
                    status.progress_pct = 100.0;
                    state_clone
                        .log_buffer
                        .push("info", &format!("Download completed: {}", repo_id));
                }
                Err(e) => {
                    status.status = "failed".to_string();
                    status.error = Some(e.clone());
                    state_clone
                        .log_buffer
                        .push("error", &format!("Download failed: {} — {}", repo_id, e));
                }
            }
        }
    });

    Ok(StatusCode::ACCEPTED)
}

/// GET /admin/api/models/downloads — get status of all downloads.
pub async fn hf_downloads(State(state): State<Arc<AppState>>) -> Json<Vec<DownloadStatus>> {
    let downloads = state.downloads.lock().unwrap();
    Json(downloads.values().cloned().collect())
}
