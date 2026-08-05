//! B1-p2.3f integration: long-prompt admit + decode.
//!
//! Validates the three-tier cap model end-to-end:
//!   1. Server boots with default --max-cache-cap = 32768.
//!   2. effective_cap_max = min(32768, Qwen3.5-4B model_max_context = 262144) = 32768.
//!   3. A request with prompt_len ≈ 10240 + max_new = 20 has cap_needed
//!      = 10260 ≤ 32768 → admits successfully.
//!   4. prefill_admitted_inner lazy-allocates cache with cap = 10260
//!      (slots_max bound at effective_cap_max), enabling long-prompt
//!      decode to completion.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::Qwen35Model;
use tokio::sync::Mutex;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN35_MODEL") {
        return PathBuf::from(p);
    }
    let glob = format!(
        "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
        std::env::var("HOME").unwrap()
    );
    std::fs::read_dir(&glob)
        .expect("snapshots dir")
        .filter_map(|e| e.ok())
        .next()
        .expect("snapshot")
        .path()
}

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let p = model_path();
    let loader = Loader::open_multimodal(&p).expect("Loader::open_multimodal");
    let tok = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader(&loader).expect("model");
    (Arc::new(Mutex::new(model)), Arc::new(tok))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy: needs QWEN35_MODEL
async fn admit_long_prompt_pp10k() {
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();

    // Build a long prompt: repeat phrase ~900 times to reach ≥ 8200 tokens.
    let phrase = "Hello world, this is a long test prompt. ";
    let raw = phrase.repeat(900);
    let prompt_ids = tokenizer.encode(&raw, false).expect("encode");
    let prompt_len = prompt_ids.len();
    eprintln!("[3f] long prompt encoded to {prompt_len} tokens");
    assert!(
        prompt_len >= 8200,
        "expected ≥ 8200 tokens (proves cap > 8192 needed); got {prompt_len}"
    );
    assert!(
        prompt_len <= 16384,
        "test prompt should fit comfortably under default cap_max=32768; got {prompt_len}"
    );

    let handle = spawn_scheduler_actor(
        model.clone(),
        /* b_max */ 4,
        /* admission_deadline */ Duration::from_millis(5),
        /* admission_queue_max */ 32,
        /* effective_cap_max */ 32768,
        /* decode_cadence_mid_chunk_cap */ 256,
        meta,
    )
    .expect("spawn");

    let max_new = 20_usize;
    let req = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        constraint: None,
    };

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: req,
            reply_tx,
        })
        .await
        .expect("cmd_tx.send");

    let admit_reply = reply_rx
        .await
        .expect("reply_rx await")
        .expect("admit ok — long prompt under cap_max");

    let mut event_rx = admit_reply.event_rx;
    let mut tokens: Vec<u32> = Vec::new();
    let mut finish_reason: Option<&'static str> = None;
    while let Some(ev) = event_rx.recv().await {
        tokens.push(ev.token);
        if let Some(reason) = ev.finish_reason {
            finish_reason = Some(reason);
            break;
        }
    }

    eprintln!(
        "[3f] decode produced {} tokens (max_new={}), finish_reason={:?}",
        tokens.len(),
        max_new,
        finish_reason
    );
    assert_eq!(
        tokens.len(),
        max_new,
        "expected exactly max_new tokens, got {} (proves cache cap fit prompt + decode)",
        tokens.len()
    );
    assert_eq!(
        finish_reason,
        Some("length"),
        "expected finish_reason=length, got {finish_reason:?}"
    );

    drop(handle);
}
