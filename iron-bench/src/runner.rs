//! Per-cell driver: warmup + N timed runs of `run_chat_completion`.
//!
//! Each cell is one (target, prompt_len, max_tokens) combination. Warmup runs
//! materialize MLX compile graphs / allocate caches; their timings are
//! discarded. Timed runs are collected as `RunOutcome`s and reduced by the
//! `report` module.

// T3 exports used by report (T4) — allow dead_code until T4 wires reduce_cell.
#![allow(dead_code)]

use std::time::SystemTime;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::client::{run_chat_completion, RequestResult};
use crate::prompt::synthesize_prompt;

#[derive(Debug)]
pub struct CellResult {
    pub target_name: String,
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

fn nonce_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
