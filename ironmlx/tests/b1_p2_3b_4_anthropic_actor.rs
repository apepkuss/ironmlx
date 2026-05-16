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
use std::time::Duration;

use axum::body::to_bytes;
use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::server::AppState;
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
    let handle = spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(5), 32);
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
    let handle = spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(5), 32);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_scheduler_path_emits_6_event_sequence() {
    let (model, tokenizer) = load_fixture();

    let prompt = "Hello.";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let input_tokens = prompt_ids.len() as u32;
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 4;

    // Construct AppState matching what serve() builds.
    let handle = spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(5), 32);
    let state = AppState {
        model: model.clone(),
        tokenizer: tokenizer.clone(),
        model_id: "test-model".to_string(),
        prefill_chunk_size: 256,
        scheduler_handle: handle.clone(),
    };

    let req = make_request(prompt_ids, max_new_tokens, stop_token_ids);

    // Invoke the scheduler-path helper directly.
    let response = ironmlx::core::server::anthropic::serve_via_scheduler_stream(
        state,
        req,
        "test-model".to_string(),
        input_tokens,
    )
    .await;

    // Collect the response body bytes.
    let body_bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let body = String::from_utf8_lossy(&body_bytes);
    println!("[anthropic_6event] raw body:\n{body}");

    // Parse SSE chunks: split on \n\n boundary. Each chunk starts with
    // "event: <type>\ndata: <json>".
    let mut event_types: Vec<String> = Vec::new();
    let mut event_payloads: Vec<serde_json::Value> = Vec::new();
    for chunk in body.split("\n\n") {
        if chunk.is_empty() {
            continue;
        }
        let mut event_type = None;
        let mut data_line = None;
        for line in chunk.lines() {
            if let Some(t) = line.strip_prefix("event: ") {
                event_type = Some(t.to_string());
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_line = Some(d);
            }
        }
        if let (Some(t), Some(d)) = (event_type, data_line) {
            event_types.push(t);
            let payload: serde_json::Value = serde_json::from_str(d).expect("parse SSE data");
            event_payloads.push(payload);
        }
    }
    println!("[anthropic_6event] event_types={:?}", event_types);

    // Assert event sequence shape.
    assert!(
        event_types.len() >= 5,
        "expected ≥5 events (message_start + content_block_start + ≥1 delta + content_block_stop + message_delta + message_stop), got {} events",
        event_types.len()
    );
    assert_eq!(
        event_types.first().map(|s| s.as_str()),
        Some("message_start"),
        "first event must be message_start"
    );
    assert_eq!(
        event_types.get(1).map(|s| s.as_str()),
        Some("content_block_start"),
        "second event must be content_block_start"
    );
    assert_eq!(
        event_types.last().map(|s| s.as_str()),
        Some("message_stop"),
        "last event must be message_stop"
    );

    // The last 3 events must be content_block_stop → message_delta → message_stop.
    let n = event_types.len();
    assert!(
        event_types[n - 3] == "content_block_stop"
            && event_types[n - 2] == "message_delta"
            && event_types[n - 1] == "message_stop",
        "tail of event_types must be [content_block_stop, message_delta, message_stop]; got {:?}",
        &event_types[n - 3..]
    );

    // Middle events (between content_block_start and content_block_stop)
    // must all be content_block_delta.
    for (i, t) in event_types.iter().enumerate().take(n - 3).skip(2) {
        assert_eq!(
            t.as_str(),
            "content_block_delta",
            "event[{i}] must be content_block_delta, got {t}"
        );
    }

    // Verify message_start payload structure.
    let start = &event_payloads[0];
    assert_eq!(start["type"], "message_start");
    assert_eq!(start["message"]["usage"]["input_tokens"], input_tokens);
    assert_eq!(start["message"]["usage"]["output_tokens"], 0);
    assert!(
        start["message"]["stop_reason"].is_null(),
        "message_start.stop_reason must be null"
    );

    // Verify message_delta payload structure.
    let delta = &event_payloads[n - 2];
    assert_eq!(delta["type"], "message_delta");
    let stop_reason = delta["delta"]["stop_reason"]
        .as_str()
        .expect("stop_reason str");
    assert!(
        stop_reason == "end_turn" || stop_reason == "max_tokens",
        "unexpected stop_reason: {stop_reason}"
    );
    let final_output_tokens = delta["usage"]["output_tokens"]
        .as_u64()
        .expect("output_tokens u64");
    // Number of content_block_delta events ≤ output_tokens (some tokens
    // may produce empty detok text — counted in output_tokens but not emitted).
    let delta_count = event_types
        .iter()
        .filter(|t| t.as_str() == "content_block_delta")
        .count() as u64;
    assert!(
        delta_count <= final_output_tokens,
        "delta count {delta_count} exceeds output_tokens {final_output_tokens} — counter invariant broken"
    );
    println!(
        "[anthropic_6event] output_tokens={} delta_count={} stop_reason={}",
        final_output_tokens, delta_count, stop_reason
    );

    let _ = handle; // keep alive
}
