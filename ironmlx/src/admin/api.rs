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
    /// Desired prompt token count (e.g. 1024, 4096). Overrides `prompt` text.
    pub prompt_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    /// Number of concurrent requests (1 = single, 2/4/8 = batch test).
    pub batch_size: Option<usize>,
}

#[derive(Serialize)]
pub struct BenchmarkResponse {
    pub model: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub batch_size: usize,
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub tg_tps: f64,
    pub pp_tps: f64,
    pub total_ms: f64,
    pub total_throughput: f64,
}

/// Build a prompt with approximately `target_tokens` tokens by repeating a
/// short sentence. The tokenizer is used to verify the actual token count.
fn build_prompt_with_token_count(
    tokenizer: &ironmlx_core::generate::Tokenizer,
    target_tokens: usize,
) -> Result<(String, Vec<i32>), String> {
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    let sentence_tokens = tokenizer.encode(sentence).map_err(|e| e.to_string())?.len();
    // Estimate repeats needed, add 20% buffer
    let repeats = (target_tokens / sentence_tokens.max(1)) + 2;
    let long_prompt = sentence.repeat(repeats);
    let mut tokens: Vec<i32> = tokenizer.encode(&long_prompt).map_err(|e| e.to_string())?;
    tokens.truncate(target_tokens);
    // Decode back to text for the actual prompt
    let prompt = tokenizer.decode(&tokens).map_err(|e| e.to_string())?;
    let final_tokens: Vec<i32> = tokenizer.encode(&prompt).map_err(|e| e.to_string())?;
    Ok((prompt, final_tokens))
}

/// Run a single benchmark request and return (prompt_len, gen_count, ttft_ms, total_ms).
async fn run_single_bench(
    entry: &crate::engine_pool::EngineEntry,
    tokens: Vec<i32>,
    max_tokens: usize,
) -> Result<(usize, usize, f64, f64), String> {
    let prompt_len = tokens.len();
    let request_id = format!("bench_{}", uuid::Uuid::new_v4().simple());
    let sampler = SamplerConfig::greedy();

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
    Ok((prompt_len, count, ttft_ms, total_ms))
}

/// POST /admin/api/benchmark — run a benchmark and return timing results.
///
/// Supports single-request and batch (concurrent) modes.
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

    let max_tokens = req.max_tokens.unwrap_or(128);
    let batch_size = req.batch_size.unwrap_or(1).max(1);

    // Build prompt tokens
    let (prompt_len, tokens) = if let Some(target) = req.prompt_tokens {
        let (_prompt_text, toks) = build_prompt_with_token_count(&entry.tokenizer, target)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        (toks.len(), toks)
    } else {
        let prompt = req
            .prompt
            .unwrap_or_else(|| "Explain the theory of relativity.".to_string());
        let toks = entry
            .tokenizer
            .encode(&prompt)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        (toks.len(), toks)
    };

    // Memory safety check before starting benchmark
    if batch_size > 1 {
        let active_mem = ironmlx_core::memory::get_active_memory().unwrap_or(0) as f64;
        let mem_limit = ironmlx_core::memory::get_memory_limit().unwrap_or(0) as f64;
        let total_mem = ironmlx_core::memory::get_memory_size().unwrap_or(0) as f64;
        let effective_limit = if mem_limit > 0.0 { mem_limit } else { total_mem * 0.9 };
        let available = effective_limit - active_mem;

        // Estimate KV cache memory per sequence:
        // 2 (K+V) * num_layers * n_kv_heads * (prompt_tokens + max_tokens) * head_dim * dtype_size
        // Use model info to estimate; fallback to ~2MB per 1K tokens for 4B model
        let estimated_per_seq_mb = (prompt_len as f64 + max_tokens as f64) * 2.0; // rough: ~2MB/1K tokens
        let total_estimated_mb = estimated_per_seq_mb * batch_size as f64;
        let available_mb = available / 1_048_576.0;

        if total_estimated_mb > available_mb * 0.8 {
            let msg = format!(
                "Insufficient GPU memory for {}x batch (pp{}). Estimated need: {:.0}MB, available: {:.0}MB. Try reducing batch size or prompt length.",
                batch_size, prompt_len, total_estimated_mb, available_mb
            );
            state.log_buffer.push("warning", &msg);
            return Err((StatusCode::SERVICE_UNAVAILABLE, msg));
        }
    }

    state.log_buffer.push(
        "info",
        &format!(
            "Benchmark started: model={}, prompt_tokens={}, max_tokens={}, batch={}",
            entry.model_id, prompt_len, max_tokens, batch_size
        ),
    );

    let t0 = std::time::Instant::now();

    if batch_size <= 1 {
        // Single request benchmark
        let (pl, gen_count, ttft_ms, total_ms) = run_single_bench(&entry, tokens, max_tokens)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

        let decode_ms = total_ms - ttft_ms;
        let tg_tps = if decode_ms > 0.0 && gen_count > 1 {
            (gen_count - 1) as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let tpot_ms = if tg_tps > 0.0 { 1000.0 / tg_tps } else { 0.0 };
        let pp_tps = if ttft_ms > 0.0 {
            pl as f64 / (ttft_ms / 1000.0)
        } else {
            0.0
        };
        let total_throughput = if total_ms > 0.0 {
            (pl + gen_count) as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };

        let resp = BenchmarkResponse {
            model: entry.model_id.clone(),
            prompt_tokens: pl,
            generated_tokens: gen_count,
            batch_size: 1,
            ttft_ms,
            tpot_ms,
            tg_tps,
            pp_tps,
            total_ms,
            total_throughput,
        };

        state.log_buffer.push(
            "info",
            &format!(
                "Benchmark done: pp{}→tg{}, TTFT={:.0}ms, tg={:.1}tok/s, pp={:.0}tok/s",
                pl, gen_count, ttft_ms, tg_tps, pp_tps
            ),
        );

        state
            .benchmark_history
            .lock()
            .unwrap()
            .push(BenchmarkResult {
                timestamp: chrono::Utc::now().timestamp(),
                model: entry.model_id.clone(),
                prompt_tokens: pl,
                gen_tokens: gen_count,
                batch_size: 1,
                ttft_ms,
                tpot_ms,
                tg_tps,
                pp_tps,
                total_ms,
                total_throughput,
            });

        state.total_tokens.fetch_add(
            (pl + gen_count) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(Json(resp))
    } else {
        // Batch benchmark: launch batch_size concurrent requests
        let mut handles = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let e = Arc::clone(&entry);
            let t = tokens.clone();
            let mt = max_tokens;
            handles.push(tokio::spawn(
                async move { run_single_bench(&e, t, mt).await },
            ));
        }

        let mut total_gen = 0usize;
        let mut sum_ttft = 0.0f64;
        let mut max_total = 0.0f64;
        for h in handles {
            let (_, gen_count, ttft_ms, total_ms) = h
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
            total_gen += gen_count;
            sum_ttft += ttft_ms;
            if total_ms > max_total {
                max_total = total_ms;
            }
        }

        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let avg_ttft = sum_ttft / batch_size as f64;
        let decode_ms = wall_ms - avg_ttft;
        let tg_tps = if decode_ms > 0.0 && total_gen > batch_size {
            (total_gen - batch_size) as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let tpot_ms = if tg_tps > 0.0 {
            batch_size as f64 * 1000.0 / tg_tps
        } else {
            0.0
        };
        let pp_tps = if avg_ttft > 0.0 {
            (prompt_len * batch_size) as f64 / (avg_ttft / 1000.0)
        } else {
            0.0
        };
        let total_tokens = (prompt_len * batch_size) + total_gen;
        let total_throughput = if wall_ms > 0.0 {
            total_tokens as f64 / (wall_ms / 1000.0)
        } else {
            0.0
        };

        let resp = BenchmarkResponse {
            model: entry.model_id.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: total_gen / batch_size,
            batch_size,
            ttft_ms: avg_ttft,
            tpot_ms,
            tg_tps,
            pp_tps,
            total_ms: wall_ms,
            total_throughput,
        };

        state.log_buffer.push(
            "info",
            &format!(
                "Batch benchmark done: {}x pp{}→tg{}, wall={:.0}ms, tg={:.1}tok/s, pp={:.0}tok/s",
                batch_size,
                prompt_len,
                total_gen / batch_size,
                wall_ms,
                tg_tps,
                pp_tps
            ),
        );

        state
            .benchmark_history
            .lock()
            .unwrap()
            .push(BenchmarkResult {
                timestamp: chrono::Utc::now().timestamp(),
                model: entry.model_id.clone(),
                prompt_tokens: prompt_len,
                gen_tokens: total_gen / batch_size,
                batch_size,
                ttft_ms: avg_ttft,
                tpot_ms,
                tg_tps,
                pp_tps,
                total_ms: wall_ms,
                total_throughput,
            });

        let total_tokens = (prompt_len * batch_size) + total_gen;
        state.total_tokens.fetch_add(
            total_tokens as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(Json(resp))
    }
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

// ── Local Model Management ──────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LocalModelInfo {
    pub repo_id: String,
    pub loaded: bool,
}

/// GET /admin/api/models/local — list all locally downloaded models.
pub async fn list_local_models(State(state): State<Arc<AppState>>) -> Json<Vec<LocalModelInfo>> {
    let models_dir = crate::config::ironmlx_root().join("models");
    let mut models = Vec::new();

    // Scan ~/.ironmlx/models/ for directories matching "models--org--name"
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("models--") && entry.file_type().is_ok_and(|t| t.is_dir()) {
                // Convert "models--org--name" to "org/name"
                let repo_id = name
                    .strip_prefix("models--")
                    .unwrap_or(&name)
                    .replacen("--", "/", 1);

                // Check if this model is currently loaded
                let loaded = state.pool.get(Some(&repo_id)).is_ok();

                models.push(LocalModelInfo { repo_id, loaded });
            }
        }
    }

    models.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    Json(models)
}

#[derive(Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
}

/// POST /admin/api/models/load — load a local model into the engine.
/// Unloads the current model first (single-model mode).
pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Check if already loaded
    if state.pool.get(Some(&req.model)).is_ok() {
        return Ok(Json(
            serde_json::json!({ "status": "already_loaded", "model": req.model }),
        ));
    }

    // Unload current model(s)
    let loaded: Vec<String> = state.pool.loaded_model_ids().into_iter().collect();
    for id in &loaded {
        let _ = state.pool.unload_model(id);
    }

    state.log_buffer.push(
        "info",
        &format!("Loading model for benchmark: {}", req.model),
    );

    // Read config for cache parameters
    let config = state.config.read().unwrap().clone();
    let hot_cache_bytes = if config.hot_cache_max_size_gb > 0.0 {
        (config.hot_cache_max_size_gb * 1_073_741_824.0) as u64
    } else {
        // Auto: GPU memory / 4
        ironmlx_core::memory::get_memory_size().unwrap_or(0) as u64 / 4
    };
    let cold_cache_bytes = (config.cache_max_size_gb * 1_073_741_824.0) as u64;

    // Load the model
    state
        .pool
        .load_model(
            &req.model,
            hot_cache_bytes,
            cold_cache_bytes,
            config.cache_dir.as_deref(),
            config.max_num_seqs,
            0, // auto init_cache_blocks
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    state
        .log_buffer
        .push("info", &format!("Model loaded: {}", req.model));

    Ok(Json(
        serde_json::json!({ "status": "loaded", "model": req.model }),
    ))
}

/// POST /admin/api/models/unload — unload a model.
pub async fn unload_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    state
        .pool
        .unload_model(&req.model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(
        serde_json::json!({ "status": "unloaded", "model": req.model }),
    ))
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
