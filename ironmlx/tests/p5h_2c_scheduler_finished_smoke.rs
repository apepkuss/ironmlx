//! P5h+2.c regression smoke — verify SchedulerActor no longer triggers
//! `step illegal in Finished phase` ERROR for max_new_tokens=1 requests.
//!
//! The bug (P5h+2.b root cause): `prefill_admitted` transitions
//! `phase = Finished` for `max_new_tokens=1` requests; the rolling loop's
//! biased `tokio::select!` falls through to `RollingEvent::Step` when
//! cmd_rx is empty; `sched.step()` rejects the Finished phase; actor
//! logs ERROR + `evict_all` + restarts outer loop. Per-request cycle
//! polluted P5h+2.b acceptance with 1116 ERROR lines per cell.
//!
//! P5h+2.c fix: actor-side pre-event finalization at rolling-loop top
//! evicts the Finished batch before any event pick. This test sends 3
//! sequential max_new_tokens=1 requests and asserts the p5h-profile
//! `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` counter stays at 0.

#![cfg(feature = "p5h-profile")]

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{oneshot, Mutex};

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
    STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_request(prompt_ids: Vec<u32>, stop_token_ids: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: 1,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 256,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        p5h_trace: None,
        p5h_root_span: None,
    }
}

async fn admit_and_expect_single_finished_event(
    handle: &SchedulerActorHandle,
    request: GenerateRequest,
) {
    let (reply_tx, reply_rx) = oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("cmd send");
    let reply = reply_rx.await.expect("admit reply").expect("admit OK");
    let mut event_rx = reply.event_rx;
    let ev = tokio::time::timeout(Duration::from_secs(60), event_rx.recv())
        .await
        .expect("event timeout")
        .expect("first event");
    assert!(
        ev.finish_reason.is_some(),
        "max_new_tokens=1 should finish on the prefill event"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "p5h+2-c — actor smoke proving Finished-phase ERROR branch not hit"]
async fn test_scheduler_actor_max_tokens_1_no_finished_phase_error() {
    // Reset counter at test start (other tests in suite may have touched it).
    STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT.store(0, Ordering::Relaxed);

    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let loader = Loader::open(Path::new(&model_dir)).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let meta = model.model_meta();
    let model = Arc::new(Mutex::new(model));
    let handle = spawn_scheduler_actor(model, 4, Duration::from_millis(5), 32, 32768, 256, meta)
        .expect("spawn_scheduler_actor");

    let prompt_ids = tokenize_prompt(&tokenizer, "Say one short word.");
    let stop_token_ids = tokenizer.eos_token_ids().to_vec();

    // Send 3 sequential max_new_tokens=1 admit cmds. Each should complete
    // (first token sampled) without triggering the Finished-phase step
    // error in the rolling loop.
    for i in 0..3 {
        let request = make_request(prompt_ids.clone(), stop_token_ids.clone());
        admit_and_expect_single_finished_event(&handle, request).await;
        eprintln!("[p5h+2-c smoke] request {i} completed");

        // Brief pause so the actor's rolling loop has a chance to attempt
        // a Step before next admit arrives (this is when the bug would
        // fire — `cmd_rx.recv()` empty → biased fall-through to Step).
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Final pause to ensure the actor has fully settled (any post-step
    // ERROR would have fired by now).
    tokio::time::sleep(Duration::from_millis(200)).await;

    let hit = STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        hit, 0,
        "expected 0 `step illegal in Finished phase` errors, got {hit}; \
         P5h+2.c fix regressed — actor is calling step() in Phase::Finished"
    );

    // Shutdown actor cleanly.
    drop(handle);
}
