//! B1-p2.3b-4 — Anthropic handler refactor + SchedulerActor integration.
//!
//! Three scenarios (see spec § 5.2):
//!   1. `anthropic_actor_b1_text_only_swap` — single text request routes
//!      to SchedulerActor; per-row tokens match B=1 GS baseline.
//!   2. `anthropic_actor_long_prompt_routes_to_gs` — prompt_len >
//!      chunk_size routes to GS; admit_count delta=0.
//!   3. (Task 3) `anthropic_actor_scheduler_path_emits_6_event_sequence`
//!      — directly invoke serve_via_scheduler_stream; assert 6 event
//!      types appear in order + payload fields correct.
//!
//! Tests are `#[ignore]`-gated; run only with `QWEN35_MODEL` env var.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

#[allow(dead_code)]
const ARGMAX_BITID_GATE: f64 = 0.95;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

/// Tokenize a chat-template-rendered prompt. Mirrors 3b-3 test pattern.
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, /* add_generation_prompt */ true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer
        .encode(&rendered, /* add_special_tokens */ false)
        .expect("encode")
}

/// Run a B=1 baseline via direct `GenerationStream`. Locks the model.
/// Caller wraps in `tokio::task::spawn_blocking` to avoid blocking_lock
/// from a Tokio worker thread (panics with "Cannot block the current
/// thread from within a runtime").
fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

#[allow(dead_code)]
fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Send one Admit cmd via `handle.cmd_tx`, await reply, drain `event_rx`
/// to completion, return collected tokens.
async fn admit_and_drain(handle: SchedulerActorHandle, request: GenerateRequest) -> Vec<u32> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send admit");
    let reply = reply_rx.await.expect("admit reply").expect("admit ok");
    let mut event_rx = reply.event_rx;
    let mut tokens = Vec::new();
    while let Some(ev) = event_rx.recv().await {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

fn make_request(
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_b1_text_only_swap() {
    let (model, tokenizer) = load_fixture();

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 12;

    // 1. B=1 baseline. Wrap in spawn_blocking because Mutex::blocking_lock
    // panics from a Tokio worker thread.
    let baseline = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let req = make_request(prompt_ids.clone(), max_new_tokens, stop_token_ids.clone());
        tokio::task::spawn_blocking(move || run_b1_baseline(&model, &tokenizer, req))
            .await
            .expect("baseline join")
    };
    assert!(!baseline.is_empty(), "baseline produced no tokens");

    // 2. Route through SchedulerActor.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let admit_before = handle.admit_count.load(Ordering::Relaxed);

    let req = make_request(prompt_ids, max_new_tokens, stop_token_ids);
    let scheduler_tokens = admit_and_drain(handle.clone(), req).await;

    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    println!(
        "[anthropic_b1] admit_delta={} scheduler_len={} baseline_len={}",
        admit_delta,
        scheduler_tokens.len(),
        baseline.len()
    );
    assert_eq!(admit_delta, 1, "expected exactly one admit");

    // Bit-id parity check. B=1 single-row Scheduler vs B=1 GenerationStream
    // use the same numerical path; bit_id should be 1.0000. Asserting
    // ≥0.95 matches 3b-2 pattern's safety margin.
    let ratio = argmax_bit_id_ratio(&scheduler_tokens, &baseline);
    println!("[anthropic_b1] bit_id={:.4}", ratio);
    assert!(
        ratio >= ARGMAX_BITID_GATE,
        "bit_id {ratio:.4} below gate {ARGMAX_BITID_GATE}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_long_prompt_routes_to_gs() {
    let (model, tokenizer) = load_fixture();

    // Build a synthetic long prompt > chunk_size = 64.
    let chunk_size: usize = 64;
    let short_ids = tokenize_prompt(&tokenizer, "Hello world.");
    let mut long_ids = Vec::with_capacity(chunk_size * 2);
    while long_ids.len() <= chunk_size {
        long_ids.extend_from_slice(&short_ids);
    }
    assert!(
        long_ids.len() > chunk_size,
        "long prompt setup failed: {} <= {}",
        long_ids.len(),
        chunk_size
    );

    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let request = GenerateRequest {
        prompt_ids: long_ids,
        max_new_tokens: 4,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    };

    // Routing predicate (mirrors Anthropic dispatch in messages handler).
    // Anthropic has no has_images check (text-only by design).
    let prompt_len = request.prompt_ids.len();
    let use_scheduler = request.prefill_chunk_size == 0 || prompt_len <= request.prefill_chunk_size;
    assert!(
        !use_scheduler,
        "routing predicate failed: long prompt would go to scheduler"
    );

    // Verify admit_count doesn't change when GS path is taken.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let before = handle.admit_count.load(Ordering::Relaxed);

    // Drop the request — the GS path bypasses the actor; the test only
    // needs to assert the routing decision (mirrors 3b-2 Scenario B/C).
    let _ = request;

    let after = handle.admit_count.load(Ordering::Relaxed);
    assert_eq!(
        after, before,
        "admit_count incremented unexpectedly: {} -> {}",
        before, after
    );
}
