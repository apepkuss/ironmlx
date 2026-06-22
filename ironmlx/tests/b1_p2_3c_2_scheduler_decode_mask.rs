//! B1-p2.3c-2 — Scheduler decode-mask activation integration test.
//!
//! Single scenario: scheduler_per_row_finish_different_steps
//!
//! Verifies the per-row decode mask correctly handles ragged cache
//! offsets when rows finish at different decode steps. B=2 with same
//! prompt, max_new_tokens=[3, 8]: row 0 finishes with 'length' at
//! decode step 2 (3rd token total = 1 prefill + 2 decode), row 1
//! continues until step 7 (8th token total).
//!
//! Bit-id parity vs B=1 GenerationStream baselines (per-row) is the
//! primary correctness gate; per-row finish-step is asserted via the
//! step-event sequence (no test seam into Scheduler internals).
//!
//! Test is `#[ignore]`-gated; run only with QWEN35_MODEL env var.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{Phase, Scheduler};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

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
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

/// Run a B=1 baseline via direct `GenerationStream`. Locks the model.
fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&*model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    loop {
        match stream.next_token().expect("next_token") {
            Some(ev) => {
                tokens.push(ev.token);
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            None => break,
        }
    }
    tokens
}

fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn scheduler_per_row_finish_different_steps() {
    let (model, tokenizer) = load_fixture();

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let max_new_a: usize = 3;
    let max_new_b: usize = 8;

    // B=1 baselines: same prompt, different max_new_tokens. Wrap in
    // spawn_blocking because tokio::sync::Mutex::blocking_lock() panics
    // when called from a Tokio worker thread driving async tasks (3b-3
    // learned lesson).
    let baseline_a = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt_ids = prompt_ids.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(
                &model,
                &tokenizer,
                make_request(prompt_ids, max_new_a, stop),
            )
        })
        .await
        .expect("baseline A join")
    };
    let baseline_b = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt_ids = prompt_ids.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(
                &model,
                &tokenizer,
                make_request(prompt_ids, max_new_b, stop),
            )
        })
        .await
        .expect("baseline B join")
    };

    assert_eq!(
        baseline_a.len(),
        max_new_a,
        "baseline A should produce exactly max_new_a tokens (got {})",
        baseline_a.len()
    );
    assert!(
        baseline_b.len() >= max_new_a,
        "baseline B should produce at least max_new_a tokens (got {})",
        baseline_b.len()
    );

    // Scheduler drive must also run on spawn_blocking — blocking_lock on
    // the model mutex from inside a tokio worker thread would panic.
    let prompt_ids_outer = prompt_ids.clone();
    let stop_outer = stop.clone();

    let (tokens_a, tokens_b, finish_step_a) = tokio::task::spawn_blocking(move || {
        let model_guard = model.blocking_lock();

        let mut sched =
            Scheduler::<ironmlx::models::Qwen35Model>::new(2, 32768, model_guard.model_meta())
                .expect("scheduler startup");
        let id_a = sched
            .admit(make_request(
                prompt_ids_outer.clone(),
                max_new_a,
                stop_outer.clone(),
            ))
            .expect("admit a");
        let _id_b = sched
            .admit(make_request(prompt_ids_outer, max_new_b, stop_outer))
            .expect("admit b");

        // Prefill emits 1 token per row.
        let prefill_events = sched.prefill_admitted(&*model_guard).expect("prefill");
        assert_eq!(
            prefill_events.len(),
            2,
            "prefill should emit 1 event per row"
        );

        let mut tokens_a: Vec<u32> = Vec::new();
        let mut tokens_b: Vec<u32> = Vec::new();
        for ev in &prefill_events {
            if ev.id == id_a {
                tokens_a.push(ev.token);
            } else {
                tokens_b.push(ev.token);
            }
        }

        // Decode loop. Track which step row a finishes at.
        let mut finish_step_a: Option<usize> = None;
        let mut step_count = 0usize;
        while sched.phase() != Phase::Finished {
            let events = sched.step(&*model_guard).expect("step");
            step_count += 1;

            for ev in &events {
                if ev.id == id_a {
                    tokens_a.push(ev.token);
                    if ev.finish_reason.is_some() && finish_step_a.is_none() {
                        finish_step_a = Some(step_count);
                    }
                } else {
                    tokens_b.push(ev.token);
                }
            }
        }

        (tokens_a, tokens_b, finish_step_a)
    })
    .await
    .expect("scheduler join");

    println!(
        "[per_row_finish] tokens_a={tokens_a:?}, tokens_b={tokens_b:?}, finish_step_a={finish_step_a:?}"
    );
    println!("[per_row_finish] baseline_a={baseline_a:?}, baseline_b={baseline_b:?}");

    assert_eq!(
        tokens_a.len(),
        max_new_a,
        "row a should produce exactly max_new_a tokens (got {})",
        tokens_a.len()
    );
    assert!(
        tokens_b.len() >= max_new_a,
        "row b should produce at least max_new_a tokens (got {})",
        tokens_b.len()
    );
    // Row a finishes on the decode step where it produces its max_new_a-th
    // token. Prefill provides token 1; decode step 1 provides token 2;
    // decode step 2 provides token 3 + finish. So finish_step_a should be
    // max_new_a - 1 = 2.
    assert_eq!(
        finish_step_a,
        Some(max_new_a - 1),
        "row a should finish on decode step {} (max_new_a - 1)",
        max_new_a - 1
    );

    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    println!(
        "[per_row_finish] bit-id row a vs baseline_a = {:.4}; row b vs baseline_b = {:.4}",
        ratio_a, ratio_b
    );
    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "row a bit-id {} < {}",
        ratio_a,
        ARGMAX_BITID_GATE
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "row b bit-id {} < {}",
        ratio_b,
        ARGMAX_BITID_GATE
    );
}
