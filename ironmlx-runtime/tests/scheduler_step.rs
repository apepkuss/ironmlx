//! B1-p2.3b-1 — Scheduler::prefill_admitted + Scheduler::step end-to-end.
//!
//! Three scenarios (see spec § 5.2):
//!   A. `b1_p2_3b_1_b2_happy`     — B=2 mixed-length prompts, both same
//!                                  `max_new_tokens`. Verify each row's
//!                                  tokens match the B=1 baseline (see
//!                                  `assert_matches_baseline`).
//!   B. `b1_p2_3b_1_b4_happy`     — B=4 (Task 4).
//!   C. `b1_p2_3b_1_mixed_finish` — B=2 with unequal `max_new_tokens`
//!                                  (Task 4).
//!
//! Test gated `#[ignore]`; runs only with `QWEN35_MODEL` env var.
//! All tests use greedy sampling (no temperature / top_k) so per-row bit-
//! id comparison is meaningful; sampler.rs Sampler::greedy() reproduces
//! the B=1 GenerationStream's argmax exactly.

use std::path::Path;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::model_input::build_position_ids;
use ironmlx_lm::core::Loader;
use ironmlx_lm::core::Message;
use ironmlx_lm::core::Tokenizer;
use ironmlx_lm::models::qwen3_5::Qwen35Model;
use ironmlx_runtime::core::generate::GenerationStream;
use ironmlx_runtime::core::generation_types::GenerateRequest;
use ironmlx_runtime::core::scheduler::Phase;
use ironmlx_runtime::core::scheduler::Scheduler;

/// A batched row and its B=1 baseline may part at a greedy near-tie: batch
/// width changes the bf16 arithmetic, and two candidates a spacing or two
/// apart swap order. Up to this many bf16 spacings (at the logits'
/// magnitude) between the two tokens counts as a near-tie.
const NEAR_TIE_BF16_SPACINGS: f32 = 2.0;

/// Distance between neighbouring bf16 values around `x`.
fn bf16_spacing(x: f32) -> f32 {
    2_f32.powi(x.abs().max(f32::MIN_POSITIVE).log2().floor() as i32 - 7)
}

/// B=1 logits for the token after `prompt` + `generated`, computed the way
/// the baseline decodes: prefill the prompt, then one token at a time.
fn next_token_logits(model: &Qwen35Model, prompt: &[u32], generated: &[u32]) -> Vec<f32> {
    let p = i32::try_from(prompt.len()).expect("prompt length");
    let total = p + i32::try_from(generated.len()).expect("generated length");
    let mut cache = model
        .make_cache(1, total + 8, model.cache_dtype())
        .expect("make_cache");
    let ids: mlx::Array = (prompt, &[1_i32, p][..]).try_into().expect("prompt ids");
    let mut logits = model
        .forward_on(
            &ids,
            &build_position_ids(0, p).expect("positions"),
            Some(&[p]),
            None,
            Some(&mut cache),
            (),
        )
        .expect("prefill");
    for (offset, &token) in generated.iter().enumerate() {
        let position = p + i32::try_from(offset).expect("offset");
        let one: mlx::Array = (&[token][..], &[1_i32, 1][..])
            .try_into()
            .expect("token id");
        logits = model
            .forward_on(
                &one,
                &build_position_ids(position, 1).expect("position"),
                Some(&[1]),
                None,
                Some(&mut cache),
                (),
            )
            .expect("decode");
    }
    mlx::ops::cast::astype(&logits, mlx::Dtype::Float32)
        .expect("cast")
        .to_vec()
        .expect("logits")
}

/// The scheduler's tokens for a row must equal the B=1 baseline up to the
/// shorter stream's end, or up to a first divergence at a near-tie, after
/// which the two continuations are no longer comparable.
fn assert_matches_baseline(
    model: &Qwen35Model,
    label: &str,
    prompt: &[u32],
    got: &[u32],
    want: &[u32],
) {
    assert!(!got.is_empty(), "{label}: scheduler produced no tokens");
    let Some(k) = got.iter().zip(want).position(|(a, b)| a != b) else {
        println!(
            "{label}: identical to the baseline over {} tokens",
            got.len().min(want.len())
        );
        return;
    };
    let logits = next_token_logits(model, prompt, &want[..k]);
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let gap = (logits[got[k] as usize] - logits[want[k] as usize]).abs();
    let spacing = bf16_spacing(top);
    println!(
        "{label}: diverges at token {k} ({} vs baseline {}): logits {:.4} vs {:.4}, top {top:.4}, \
         gap {gap:.4} = {:.1} bf16 spacings",
        got[k],
        want[k],
        logits[got[k] as usize],
        logits[want[k] as usize],
        gap / spacing
    );
    assert!(
        gap <= NEAR_TIE_BF16_SPACINGS * spacing,
        "{label}: tokens part at {k} where the candidates are {gap:.4} apart \
         (more than {NEAR_TIE_BF16_SPACINGS} bf16 spacings of {spacing}); not a near-tie"
    );
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
        priority: Default::default(),
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        constraint: None,
    };
    let mut stream = GenerationStream::new(model, tokenizer, req).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
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

    // Factual short prompts chosen to minimize greedy near-tied argmax
    // positions (which cascade under bf16 ULP noise — see close-out).
    let prompt_a = "What is the capital of France?";
    let prompt_b = "Name three primary colors used in painting.";
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
    let mut sched = Scheduler::new(2, 32768, model.model_meta()).expect("scheduler startup");
    assert_eq!(sched.phase(), Phase::Idle);

    let id_a = sched
        .admit(GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_a_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            constraint: None,
        })
        .expect("admit A");
    let id_b = sched
        .admit(GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_b_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            constraint: None,
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
    println!("[b2_happy] scheduler_a tokens={:?}", tokens_a);
    println!("[b2_happy] scheduler_b tokens={:?}", tokens_b);
    println!("[b2_happy] baseline_a  tokens={:?}", baseline_a);
    println!("[b2_happy] baseline_b  tokens={:?}", baseline_b);
    assert_matches_baseline(
        &model,
        "[b2_happy] row A",
        &prompt_a_ids,
        &tokens_a,
        &baseline_a,
    );
    assert_matches_baseline(
        &model,
        "[b2_happy] row B",
        &prompt_b_ids,
        &tokens_b,
        &baseline_b,
    );

    // 4. Cache reuse: evict_all → Idle, then admit + prefill again.
    sched.evict_all().expect("evict_all");
    assert_eq!(sched.phase(), Phase::Idle);
    assert_eq!(sched.active_count(), 0);

    let id_c = sched
        .admit(GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_a_ids,
            max_new_tokens: 4,
            sampler: Sampler::greedy(),
            stop_token_ids,
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            constraint: None,
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
    println!("[b2_happy] cache-reuse tokens_c={:?}", tokens_c);
    assert!(
        !tokens_c.is_empty() && tokens_c.len() <= 4,
        "cache-reuse second batch produced {} tokens (expected 1..=4)",
        tokens_c.len()
    );
    let _ = (id_a, id_b);
}

#[test]
#[ignore]
fn b1_p2_3b_1_b4_happy() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    let prompts = [
        "What is two plus two?",
        "Name one color of the sky during sunset.",
        "Write a single-sentence definition of gravity.",
        "How many continents are there on Earth?",
    ];
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenize_prompt(&tokenizer, p))
        .collect();

    let max_new_tokens: usize = 12;
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    // 1. Four B=1 baselines.
    let baselines: Vec<Vec<u32>> = prompt_ids
        .iter()
        .map(|p| {
            run_b1_baseline(
                &model,
                &tokenizer,
                p.clone(),
                max_new_tokens,
                stop_token_ids.clone(),
            )
        })
        .collect();
    for (i, b) in baselines.iter().enumerate() {
        assert!(!b.is_empty(), "baseline row {i} produced no tokens");
    }

    // 2. Scheduler B=4 run (Option A: prefill_admitted returns Vec<StepEvent>).
    let mut sched = Scheduler::new(4, 32768, model.model_meta()).expect("scheduler startup");
    let ids: Vec<_> = prompt_ids
        .iter()
        .map(|p| {
            sched
                .admit(GenerateRequest {
                    priority: Default::default(),
                    prompt_ids: p.clone(),
                    max_new_tokens,
                    sampler: Sampler::greedy(),
                    stop_token_ids: stop_token_ids.clone(),
                    prefill_chunk_size: 0,
                    decode_cadence_mid_chunk_cap: 256,
                    kv_cache_turboquant_bits: None,
                    pixel_values: None,
                    image_grid_thw: None,
                    image_spatial_merge_size: 2,
                    image_token_id: 248056,
                    constraint: None,
                })
                .expect("admit")
        })
        .collect();

    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); 4];
    let prefill_events = sched.prefill_admitted(&model).expect("prefill_admitted");
    assert!(
        !prefill_events.is_empty(),
        "prefill should emit ≥1 event per row"
    );
    for ev in prefill_events {
        let row = ids
            .iter()
            .position(|id| *id == ev.id)
            .expect("unknown event id from prefill");
        tokens[row].push(ev.token);
    }
    assert_eq!(sched.phase(), Phase::Decoding);

    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            let row = ids
                .iter()
                .position(|id| *id == ev.id)
                .expect("unknown event id");
            tokens[row].push(ev.token);
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // 3. Compare each row against its baseline.
    for (i, (got, want)) in tokens.iter().zip(baselines.iter()).enumerate() {
        assert_matches_baseline(
            &model,
            &format!("[b4_happy] row {i}"),
            &prompt_ids[i],
            got,
            want,
        );
    }

    sched.evict_all().expect("evict_all");
    assert_eq!(sched.phase(), Phase::Idle);
}

#[test]
#[ignore]
fn b1_p2_3b_1_mixed_finish() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    // Same prompt for both rows to isolate the mixed-finish effect on
    // emission timing (each row is still compared with its own baseline).
    let prompt = "Describe a sunny day.";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);

    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    // Row 0 caps at 8 tokens, row 1 at 24 tokens.
    let max_a: usize = 8;
    let max_b: usize = 24;

    // B=1 baselines for each cap.
    let baseline_a = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_ids.clone(),
        max_a,
        stop_token_ids.clone(),
    );
    let baseline_b = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_ids.clone(),
        max_b,
        stop_token_ids.clone(),
    );

    let mut sched = Scheduler::new(2, 32768, model.model_meta()).expect("scheduler startup");
    let id_a = sched
        .admit(GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_ids.clone(),
            max_new_tokens: max_a,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            constraint: None,
        })
        .expect("admit A");
    let id_b = sched
        .admit(GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_ids.clone(),
            max_new_tokens: max_b,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            constraint: None,
        })
        .expect("admit B");

    let mut events_a: Vec<(u32, Option<&'static str>)> = Vec::new();
    let mut events_b: Vec<(u32, Option<&'static str>)> = Vec::new();

    let prefill_events = sched.prefill_admitted(&model).expect("prefill_admitted");
    for ev in prefill_events {
        if ev.id == id_a {
            events_a.push((ev.token, ev.finish_reason));
        } else if ev.id == id_b {
            events_b.push((ev.token, ev.finish_reason));
        }
    }

    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            if ev.id == id_a {
                events_a.push((ev.token, ev.finish_reason));
            } else if ev.id == id_b {
                events_b.push((ev.token, ev.finish_reason));
            }
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // Row A: at most max_a events; last one has finish_reason Some.
    // (Baseline may finish earlier on EOS; allow ≤ max_a as long as the
    // last event carries a finish_reason.)
    assert!(
        events_a.len() <= max_a,
        "row A emitted {} events, exceeds cap {}",
        events_a.len(),
        max_a
    );
    assert!(
        events_a.last().expect("row A non-empty").1.is_some(),
        "row A last event missing finish_reason: {:?}",
        events_a.last()
    );
    // Row B: at most max_b events, last has finish_reason.
    assert!(
        events_b.len() <= max_b,
        "row B emitted {} events, exceeds cap {}",
        events_b.len(),
        max_b
    );
    assert!(
        events_b.last().expect("row B non-empty").1.is_some(),
        "row B last event missing finish_reason: {:?}",
        events_b.last()
    );
    // Once row A finished, no further row-A events show up. This is
    // implicit in the iteration above (we only collect per-step events),
    // but cross-check explicitly:
    let a_finish_idx = events_a
        .iter()
        .position(|(_, r)| r.is_some())
        .expect("row A finish position");
    assert_eq!(
        a_finish_idx + 1,
        events_a.len(),
        "row A emitted events after finish: {:?}",
        events_a
    );
    let b_finish_idx = events_b
        .iter()
        .position(|(_, r)| r.is_some())
        .expect("row B finish position");
    assert_eq!(
        b_finish_idx + 1,
        events_b.len(),
        "row B emitted events after finish: {:?}",
        events_b
    );

    // Each row against its baseline (only valid up to whichever ends first).
    let tokens_a: Vec<u32> = events_a.iter().map(|(t, _)| *t).collect();
    let tokens_b: Vec<u32> = events_b.iter().map(|(t, _)| *t).collect();
    assert_matches_baseline(
        &model,
        "[mixed_finish] row A",
        &prompt_ids,
        &tokens_a,
        &baseline_a,
    );
    assert_matches_baseline(
        &model,
        "[mixed_finish] row B",
        &prompt_ids,
        &tokens_b,
        &baseline_b,
    );

    sched.evict_all().expect("evict_all");
}
