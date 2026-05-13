//! B1-p2.3b-1 — Scheduler::prefill_admitted + Scheduler::step end-to-end.
//!
//! Three scenarios (see spec § 5.2):
//!   A. `b1_p2_3b_1_b2_happy`     — B=2 mixed-length prompts, both same
//!                                  `max_new_tokens`. Verify each row's
//!                                  tokens match B=1 baseline argmax
//!                                  bit-id ≥ 0.95.
//!   B. `b1_p2_3b_1_b4_happy`     — B=4 (Task 4).
//!   C. `b1_p2_3b_1_mixed_finish` — B=2 with unequal `max_new_tokens`
//!                                  (Task 4).
//!
//! Test gated `#[ignore]`; runs only with `QWEN35_MODEL` env var.
//! All tests use greedy sampling (no temperature / top_k) so per-row bit-
//! id comparison is meaningful; sampler.rs Sampler::greedy() reproduces
//! the B=1 GenerationStream's argmax exactly.

use std::path::Path;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{Phase, Scheduler};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

const ARGMAX_BITID_GATE: f64 = 0.95;

/// Argmax bit-id ratio between two token streams. Returns the fraction
/// of positions where both streams emit the same token, computed over
/// `min(a.len(), b.len())` positions.
fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Tokenize a prompt with the chat template applied (enable_thinking=false).
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

/// Run a single-stream B=1 baseline for one prompt — returns the generated tokens.
fn run_b1_baseline(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
) -> Vec<u32> {
    let req = GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    };
    let mut stream = GenerationStream::new(model, tokenizer, req).expect("new stream");
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

#[test]
#[ignore]
fn b1_p2_3b_1_b2_happy() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    let prompt_a = "Explain in one sentence what a transformer is.";
    let prompt_b = "Tell me a 16-word story about a robot that loves clouds.";
    let prompt_a_ids = tokenize_prompt(&tokenizer, prompt_a);
    let prompt_b_ids = tokenize_prompt(&tokenizer, prompt_b);

    println!(
        "[b2_happy] prompt_a len={}, prompt_b len={}",
        prompt_a_ids.len(),
        prompt_b_ids.len()
    );

    let max_new_tokens: usize = 16;
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    // 1. B=1 reference for each prompt.
    let baseline_a = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_a_ids.clone(),
        max_new_tokens,
        stop_token_ids.clone(),
    );
    let baseline_b = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_b_ids.clone(),
        max_new_tokens,
        stop_token_ids.clone(),
    );
    assert!(!baseline_a.is_empty(), "baseline A produced no tokens");
    assert!(!baseline_b.is_empty(), "baseline B produced no tokens");
    println!(
        "[b2_happy] baseline_a tokens={:?}",
        &baseline_a[..baseline_a.len().min(8)]
    );
    println!(
        "[b2_happy] baseline_b tokens={:?}",
        &baseline_b[..baseline_b.len().min(8)]
    );

    // 2. Scheduler B=2 run.
    let mut sched = Scheduler::new(2);
    assert_eq!(sched.phase(), Phase::Idle);

    let id_a = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_a_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit A");
    let id_b = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_b_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit B");

    assert_eq!(sched.phase(), Phase::Admitting);
    let prefill_events = sched.prefill_admitted(&model).expect("prefill_admitted");
    assert!(
        !prefill_events.is_empty(),
        "prefill should emit ≥1 event per row"
    );
    assert_eq!(sched.phase(), Phase::Decoding);

    let mut tokens_a: Vec<u32> = Vec::new();
    let mut tokens_b: Vec<u32> = Vec::new();
    for ev in prefill_events {
        if ev.id == id_a {
            tokens_a.push(ev.token);
        } else if ev.id == id_b {
            tokens_b.push(ev.token);
        } else {
            panic!("unexpected event id from prefill {ev:?}");
        }
    }
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            if ev.id == id_a {
                tokens_a.push(ev.token);
            } else if ev.id == id_b {
                tokens_b.push(ev.token);
            } else {
                panic!("unexpected event id {ev:?}");
            }
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // 3. Compare against baselines.
    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    println!(
        "[b2_happy] row_a: scheduler={} baseline={} bit_id={:.4}",
        tokens_a.len(),
        baseline_a.len(),
        ratio_a
    );
    println!(
        "[b2_happy] row_b: scheduler={} baseline={} bit_id={:.4}",
        tokens_b.len(),
        baseline_b.len(),
        ratio_b
    );
    println!("[b2_happy] scheduler_a tokens={:?}", &tokens_a);
    println!("[b2_happy] scheduler_b tokens={:?}", &tokens_b);
    println!("[b2_happy] baseline_a  tokens={:?}", &baseline_a);
    println!("[b2_happy] baseline_b  tokens={:?}", &baseline_b);
    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "row A argmax bit-id {ratio_a:.4} below gate {ARGMAX_BITID_GATE}"
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "row B argmax bit-id {ratio_b:.4} below gate {ARGMAX_BITID_GATE}"
    );

    // 4. Cache reuse: evict_all → Idle, then admit + prefill again.
    sched.evict_all().expect("evict_all");
    assert_eq!(sched.phase(), Phase::Idle);
    assert_eq!(sched.active_count(), 0);

    let id_c = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_a_ids,
            max_new_tokens: 4,
            sampler: Sampler::greedy(),
            stop_token_ids,
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit C");
    let prefill_events_2 = sched.prefill_admitted(&model).expect("prefill_admitted #2");
    let mut tokens_c: Vec<u32> = Vec::new();
    for ev in prefill_events_2 {
        if ev.id == id_c {
            tokens_c.push(ev.token);
        }
    }
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step #2");
        for ev in events {
            if ev.id == id_c {
                tokens_c.push(ev.token);
            }
        }
    }
    println!("[b2_happy] cache-reuse tokens_c={:?}", &tokens_c);
    assert!(
        !tokens_c.is_empty() && tokens_c.len() <= 4,
        "cache-reuse second batch produced {} tokens (expected 1..=4)",
        tokens_c.len()
    );
    let _ = (id_a, id_b);
}
