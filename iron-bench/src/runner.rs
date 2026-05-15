//! Per-cell driver: warmup + N timed runs of `run_chat_completion`.
//!
//! Each cell is one (target, prompt_len, max_tokens) combination. Warmup runs
//! materialize MLX compile graphs / allocate caches; their timings are
//! discarded. Timed runs are collected as `RunOutcome`s and reduced by the
//! `report` module.

use std::time::SystemTime;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::client::{run_chat_completion, RequestResult};
use crate::prompt::synthesize_prompt;

#[derive(Debug)]
pub struct CellResult {
    pub target_name: String,
    #[allow(dead_code)]
    pub target_url: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub runs: Vec<RunOutcome>,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub run_idx: usize,
    pub prompt_tokens_local: usize,
    pub result: RequestResult,
}

/// (v2 concurrent mode) One worker iteration's outcome.
#[derive(Debug)]
pub struct RequestOutcome {
    pub worker_id: usize,
    pub prompt_tokens_local: usize,
    pub result: RequestResult,
}

/// (v2 concurrent mode) Per-cell result: N workers ran for `duration` seconds,
/// produced `outcomes` requests in aggregate.
#[derive(Debug)]
pub struct ConcurrentCellResult {
    pub target_name: String,
    #[allow(dead_code)]
    pub target_url: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub concurrent: usize,
    /// Wall-clock start of the timed phase (after warmup). Used to compute
    /// aggregate tokens/s and req/s precisely.
    pub cell_start: std::time::Instant,
    /// Wall-clock end of the timed phase (after all workers joined).
    pub cell_end: std::time::Instant,
    pub outcomes: Vec<RequestOutcome>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_cell(
    client: &reqwest::Client,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup: usize,
    runs: usize,
    tokenizer: &Tokenizer,
) -> Result<CellResult> {
    eprintln!("[{target_name}] PP={pp} TG={tg}: warmup x{warmup} ...");
    for w in 0..warmup {
        let nonce = nonce_seed() ^ (w as u64);
        let (prompt, _) = synthesize_prompt(tokenizer, pp, nonce)?;
        let _ = run_chat_completion(client, target_url, model, &prompt, tg).await?;
    }

    eprintln!("[{target_name}] PP={pp} TG={tg}: timed runs x{runs} ...");
    let mut outcomes = Vec::with_capacity(runs);
    for i in 0..runs {
        let nonce = nonce_seed() ^ ((i as u64) << 8);
        let (prompt, prompt_tokens_local) = synthesize_prompt(tokenizer, pp, nonce)?;
        let result = run_chat_completion(client, target_url, model, &prompt, tg).await?;

        let ttft_ms = result.timings.ttft().as_secs_f64() * 1000.0;
        let gen_secs = result.timings.gen_duration().as_secs_f64().max(1e-9);
        let tg_count = result
            .server_completion_tokens
            .map(|n| n as f64)
            .unwrap_or(result.chunk_count as f64);
        let tg_tps = tg_count / gen_secs;
        eprintln!(
            "  [{target_name}] run {}/{runs}: TTFT={ttft_ms:.1}ms TG={tg_tps:.1} tok/s prompt={prompt_tokens_local}",
            i + 1
        );

        outcomes.push(RunOutcome {
            run_idx: i,
            prompt_tokens_local,
            result,
        });
    }

    Ok(CellResult {
        target_name: target_name.into(),
        target_url: target_url.into(),
        pp_target: pp,
        tg_target: tg,
        runs: outcomes,
    })
}

/// (v2 concurrent mode) Drive a single cell with `concurrent` workers for
/// `warmup_duration` (discarded) then `duration` (timed) wall-clock seconds.
///
/// Each worker independently fires `run_chat_completion` -> awaits response ->
/// repeats with a fresh nonce, until the deadline. Outcomes from all workers
/// are flattened into `ConcurrentCellResult.outcomes` for the reporter.
///
/// `client` and `tokenizer` are shared via `Arc` to avoid per-worker resource
/// duplication (HTTP connection pool reuse + tokenizer load amortization).
#[allow(clippy::too_many_arguments)]
pub async fn run_cell_concurrent(
    client: std::sync::Arc<reqwest::Client>,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup_duration: std::time::Duration,
    duration: std::time::Duration,
    concurrent: usize,
    tokenizer: std::sync::Arc<Tokenizer>,
) -> Result<ConcurrentCellResult> {
    use std::time::Instant;

    eprintln!(
        "[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: warmup {warmup_duration:?} ..."
    );

    // === 1. Warmup phase: N workers run for warmup_duration, discard outcomes. ===
    if !warmup_duration.is_zero() {
        let warmup_deadline = Instant::now() + warmup_duration;
        let mut warmup_handles = Vec::with_capacity(concurrent);
        for worker_id in 0..concurrent {
            let client_w = client.clone();
            let tokenizer_w = tokenizer.clone();
            let url = target_url.to_string();
            let model_w = model.to_string();
            warmup_handles.push(tokio::spawn(async move {
                let mut nonce = nonce_seed() ^ (worker_id as u64);
                while Instant::now() < warmup_deadline {
                    let (prompt, _) = crate::prompt::synthesize_prompt(&tokenizer_w, pp, nonce)?;
                    let _ =
                        crate::client::run_chat_completion(&client_w, &url, &model_w, &prompt, tg)
                            .await?;
                    nonce = nonce.wrapping_add(1);
                }
                Ok::<(), anyhow::Error>(())
            }));
        }
        for h in warmup_handles {
            h.await??;
        }
    }

    eprintln!("[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: timed {duration:?} ...");

    // === 2. Timed phase: N workers, duration, collect outcomes. ===
    let cell_start = Instant::now();
    let timed_deadline = cell_start + duration;
    let mut timed_handles = Vec::with_capacity(concurrent);
    for worker_id in 0..concurrent {
        let client_w = client.clone();
        let tokenizer_w = tokenizer.clone();
        let url = target_url.to_string();
        let model_w = model.to_string();
        timed_handles.push(tokio::spawn(async move {
            let mut outcomes: Vec<RequestOutcome> = Vec::new();
            // Distinct nonce space per worker: high 16 bits = worker_id,
            // low 48 bits = wrapping counter. No collisions across workers
            // until each worker has fired 2^48 requests (effectively never).
            let mut nonce = nonce_seed() ^ ((worker_id as u64) << 48);
            while Instant::now() < timed_deadline {
                let (prompt, prompt_local) =
                    crate::prompt::synthesize_prompt(&tokenizer_w, pp, nonce)?;
                let result =
                    crate::client::run_chat_completion(&client_w, &url, &model_w, &prompt, tg)
                        .await?;
                outcomes.push(RequestOutcome {
                    worker_id,
                    prompt_tokens_local: prompt_local,
                    result,
                });
                nonce = nonce.wrapping_add(1);
            }
            Ok::<Vec<RequestOutcome>, anyhow::Error>(outcomes)
        }));
    }

    let mut all_outcomes: Vec<RequestOutcome> = Vec::new();
    for h in timed_handles {
        all_outcomes.extend(h.await??);
    }
    let cell_end = Instant::now();

    eprintln!(
        "[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: {} requests completed",
        all_outcomes.len()
    );

    Ok(ConcurrentCellResult {
        target_name: target_name.into(),
        target_url: target_url.into(),
        pp_target: pp,
        tg_target: tg,
        concurrent,
        cell_start,
        cell_end,
        outcomes: all_outcomes,
    })
}

fn nonce_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
