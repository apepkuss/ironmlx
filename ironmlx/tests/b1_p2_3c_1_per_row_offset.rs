//! B1-p2.3c-1 — Per-row KV cache offset (cache + model API) integration tests.
//!
//! Five scenarios per spec §5.3:
//!   1. uniform_lens_matches_lockstep_baseline (Task 5)
//!   2. ragged_lens_offsets_diverge (Task 5)
//!   3. zero_len_skips_row (Task 5)
//!   4. decode_with_ragged_offsets (Task 6)
//!   5. invalid_args_return_err (Task 6)
//!
//! Tests are `#[ignore]`-gated; run only with QWEN35_MODEL env var.

use std::path::Path;
use std::sync::Arc;

use mlx::{Array, Dtype};
use tokio::sync::Mutex;

use ironmlx::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_position_ids_batched, GenerateRequest, GenerationStream,
};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;
use ironmlx::nn::LayerCache;

const ARGMAX_BITID_GATE: f64 = 0.95;
const DECODE_STEPS: usize = 8;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

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

fn make_request(prompt_ids: Vec<u32>, max_new_tokens: usize, stop: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: stop,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

/// Run a B=1 baseline via direct GenerationStream. Locks the model.
/// Caller must wrap in tokio::task::spawn_blocking when invoked from a
/// Tokio async context (blocking_lock panics on worker threads otherwise).
fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
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

/// Build a right-padded `[B, max_len]` int32 input_ids tensor + matching
/// position_ids + attention mask + linear mask. Right-pad layout per
/// B1-p2.3c-1 Task 4: real tokens at columns [0..prompt_lens[i]], pad at
/// [prompt_lens[i]..max_len]. Returns
/// (input_ids, position_ids, attn_mask, linear_mask, prompt_lens).
fn build_batched_prefill_inputs(prompts: &[Vec<u32>]) -> (Array, Array, Array, Array, Vec<i32>) {
    let b = prompts.len();
    let prompt_lens: Vec<i32> = prompts.iter().map(|p| p.len() as i32).collect();
    let max_len = *prompt_lens.iter().max().expect("non-empty prompts");
    let s = max_len as usize;

    // Right-pad: real tokens at [0..L_i], pad zeros at [L_i..s].
    let mut flat: Vec<i32> = vec![0; b * s];
    for (row, p) in prompts.iter().enumerate() {
        for (j, &tok) in p.iter().enumerate() {
            flat[row * s + j] = tok as i32;
        }
    }
    let input_ids: Array = (&flat[..], &[b as i32, max_len][..])
        .try_into()
        .expect("input_ids");

    let pos_ids = build_position_ids_batched(&prompt_lens, max_len).expect("pos_ids");
    let attn_mask =
        build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16).expect("attn_mask");
    let linear_mask = build_batch_linear_mask(&prompt_lens, max_len).expect("linear_mask");

    (input_ids, pos_ids, attn_mask, linear_mask, prompt_lens)
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 1: uniform_lens_matches_lockstep_baseline
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn per_row_offset_uniform_lens_matches_lockstep_baseline() {
    let (model, tokenizer) = load_fixture();

    let prompt_a = "What is the capital of France?";
    let prompt_b = "Name three primary colors used in painting.";
    let prompt_a_ids = tokenize_prompt(&tokenizer, prompt_a);
    let prompt_b_ids = tokenize_prompt(&tokenizer, prompt_b);
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    // B=1 baselines via spawn_blocking (3b-3 pattern — avoid blocking_lock panic).
    let baseline_a = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let req = make_request(prompt_a_ids.clone(), DECODE_STEPS, stop.clone());
        tokio::task::spawn_blocking(move || run_b1_baseline(&model, &tokenizer, req))
            .await
            .expect("baseline A")
    };
    let baseline_b = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let req = make_request(prompt_b_ids.clone(), DECODE_STEPS, stop.clone());
        tokio::task::spawn_blocking(move || run_b1_baseline(&model, &tokenizer, req))
            .await
            .expect("baseline B")
    };

    let prompt_a_ids_outer = prompt_a_ids.clone();
    let prompt_b_ids_outer = prompt_b_ids.clone();

    // Batched prefill + per-row decode via direct model API.
    let (batched_a, batched_b) = tokio::task::spawn_blocking(move || {
        let model_guard = model.blocking_lock();
        let prompts = vec![prompt_a_ids_outer, prompt_b_ids_outer];
        let (input_ids, pos_ids, attn_mask, linear_mask, prompt_lens) =
            build_batched_prefill_inputs(&prompts);
        let max_len = *prompt_lens.iter().max().unwrap();
        let cap = max_len + DECODE_STEPS as i32 + 1;
        let mut cache: Vec<LayerCache> = model_guard
            .make_cache(2, cap, Dtype::Bfloat16)
            .expect("make_cache B=2");

        let logits = model_guard
            .batched_prefill(
                &input_ids,
                &pos_ids,
                &attn_mask,
                &linear_mask,
                &prompt_lens,
                Some(&mut cache),
                (),
            )
            .expect("batched_prefill");

        // After per-row prefill (right-pad), row i's cache occupies [0..prompt_lens[i]].
        // Smoke-check the first Full-attention layer's offsets.
        let mut full_seen = false;
        for cell in &cache {
            if let LayerCache::Full(kv) = cell {
                assert_eq!(
                    kv.offsets(),
                    &prompt_lens[..],
                    "Full cache offsets should equal prompt_lens after per-row prefill"
                );
                full_seen = true;
                break;
            }
        }
        assert!(full_seen, "expected at least one Full layer in cache");

        // Linear (GatedDelta) layer offsets — same expectation on real-text prompts
        // (Scenario 2 asserts this with synthetic IDs; this covers the real-prompt path).
        let mut linear_seen = false;
        for cell in &cache {
            if let LayerCache::Linear(gdc) = cell {
                assert_eq!(
                    gdc.offsets(),
                    &prompt_lens[..],
                    "Linear cache offsets should equal prompt_lens after per-row prefill"
                );
                linear_seen = true;
                break;
            }
        }
        assert!(linear_seen, "expected at least one Linear layer in cache");

        // Sample first token per row from prefill logits.
        // logits shape: [B, 1, vocab] (slice_last_and_project per-row collapsed).
        let vocab = logits.shape().as_slice()[2];
        let mut tokens_a: Vec<u32> = Vec::with_capacity(DECODE_STEPS + 1);
        let mut tokens_b: Vec<u32> = Vec::with_capacity(DECODE_STEPS + 1);

        for b_idx in 0..2_usize {
            let row = mlx::ops::indexing::slice(
                &logits,
                &[b_idx as i32, 0_i32, 0_i32][..],
                &[b_idx as i32 + 1, 1_i32, vocab][..],
            )
            .expect("slice row");
            let flat = row.reshape(&[vocab][..]).expect("reshape row");
            let v: Vec<f32> = mlx::ops::cast::astype(&flat, Dtype::Float32)
                .expect("astype f32")
                .to_vec()
                .expect("to_vec");
            let arg = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap();
            if b_idx == 0 {
                tokens_a.push(arg);
            } else {
                tokens_b.push(arg);
            }
        }

        // Decode loop. per_row_lens = [1, 1] per step.
        for _ in 0..DECODE_STEPS {
            let last = [*tokens_a.last().expect("a"), *tokens_b.last().expect("b")];
            let next_input: Array = (&last[..], &[2_i32, 1_i32][..]).try_into().expect("next");
            // Each row's position for the next forward = prompt_lens[i] + tokens.len() - 1.
            // Entering the loop, tokens_{a,b}.len() == 1 (first token from prefill), so
            // step 0's positions = prompt_lens[i] (the slot where the new token lands).
            let per_row_pos: Vec<i32> = vec![
                prompt_lens[0] + tokens_a.len() as i32 - 1,
                prompt_lens[1] + tokens_b.len() as i32 - 1,
            ];
            let pos_ids = build_decode_position_ids(&per_row_pos).expect("pos");
            let step_logits = model_guard
                .forward_on(&next_input, &pos_ids, Some(&[1, 1]), Some(&mut cache), ())
                .expect("forward_on decode");
            for b_idx in 0..2_usize {
                let row = mlx::ops::indexing::slice(
                    &step_logits,
                    &[b_idx as i32, 0_i32, 0_i32][..],
                    &[b_idx as i32 + 1, 1_i32, vocab][..],
                )
                .expect("slice");
                let flat = row.reshape(&[vocab][..]).expect("reshape");
                let v: Vec<f32> = mlx::ops::cast::astype(&flat, Dtype::Float32)
                    .expect("astype f32")
                    .to_vec()
                    .expect("to_vec");
                let arg = v
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap();
                if b_idx == 0 {
                    tokens_a.push(arg);
                } else {
                    tokens_b.push(arg);
                }
            }
        }
        (tokens_a, tokens_b)
    })
    .await
    .expect("batched join");

    let ratio_a = argmax_bit_id_ratio(&batched_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&batched_b, &baseline_b);
    println!(
        "[uniform_lens] row 0 bit-id={:.4}, row 1 bit-id={:.4}",
        ratio_a, ratio_b
    );
    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "row 0 bit-id {} < {}",
        ratio_a,
        ARGMAX_BITID_GATE
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "row 1 bit-id {} < {}",
        ratio_b,
        ARGMAX_BITID_GATE
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 2: ragged_lens_offsets_diverge
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn per_row_offset_ragged_lens_offsets_diverge() {
    let (model, _tokenizer) = load_fixture();

    tokio::task::spawn_blocking(move || {
        let model_guard = model.blocking_lock();
        // Synthetic prompts: row 0 is 8 tokens, row 1 is 16 tokens.
        // We're not asserting bit-id parity — only cache offset divergence.
        let prompt_a: Vec<u32> = (10u32..18).collect();
        let prompt_b: Vec<u32> = (20u32..36).collect();
        let prompts = vec![prompt_a, prompt_b];
        let (input_ids, pos_ids, attn_mask, linear_mask, prompt_lens) =
            build_batched_prefill_inputs(&prompts);
        assert_eq!(prompt_lens, &[8, 16]);

        let max_len = 16;
        let cap = max_len + 8;
        let mut cache: Vec<LayerCache> = model_guard
            .make_cache(2, cap, Dtype::Bfloat16)
            .expect("make_cache");

        let _logits = model_guard
            .batched_prefill(
                &input_ids,
                &pos_ids,
                &attn_mask,
                &linear_mask,
                &prompt_lens,
                Some(&mut cache),
                (),
            )
            .expect("batched_prefill");

        // Full layer cache offsets — should be [8, 16] per row.
        let mut full_seen = 0;
        for cell in &cache {
            if let LayerCache::Full(kv) = cell {
                assert_eq!(
                    kv.offsets(),
                    &[8_i32, 16],
                    "Full layer cache offsets should be ragged [8, 16]"
                );
                full_seen += 1;
            }
        }
        assert!(full_seen > 0, "expected at least one Full layer in cache");

        // Linear (GatedDelta) layer cache offsets — same expectation.
        let mut linear_seen = 0;
        for cell in &cache {
            if let LayerCache::Linear(gdc) = cell {
                assert_eq!(
                    gdc.offsets(),
                    &[8_i32, 16],
                    "Linear layer cache offsets should be ragged [8, 16]"
                );
                linear_seen += 1;
            }
        }
        assert!(
            linear_seen > 0,
            "expected at least one Linear layer in cache"
        );
    })
    .await
    .expect("ragged join");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 3: zero_len_skips_row
// ────────────────────────────────────────────────────────────────────────────

/// Scenario 3 tests the cache-level zero-skip invariant directly.
///
/// `batched_prefill` internally computes `last_positions[i] = per_row_lens[i] - 1`,
/// which yields -1 when `per_row_lens[i] == 0`. That hits the bounds check in
/// `per_row_slice_last` and returns `Err`. The model API does not support
/// `per_row_lens[i] == 0` as an end-to-end argument.
///
/// The actual zero-skip behavior lives in `KVCache::update_and_fetch` (Strategy A
/// write loop skips rows with `per_row_lens[i] == 0`) and
/// `GatedDeltaCache::advance` (adds 0 to `offsets[i]`). Both are tested here
/// by calling the cache APIs directly — this is the correct layer to verify
/// the invariant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn per_row_offset_zero_len_skips_row() {
    use ironmlx::core::cache::GatedDeltaCache;
    use ironmlx::KVCache;

    tokio::task::spawn_blocking(move || {
        // ── KVCache: row 0 len=0 (skip), row 1 len=12 (write) ──────────────
        {
            // B=2, 4 kv heads, head_dim=8, cap=20.
            let mut kv = KVCache::new(2, 4, 8, 8, Dtype::Bfloat16, 20).with_step(20);

            // k/v shape [B, n_kv_heads, T_q, head_dim] with T_q=12 (max of per_row_lens).
            let k_data: Vec<f32> = vec![1.0; 2 * 4 * 12 * 8];
            let v_data: Vec<f32> = vec![2.0; 2 * 4 * 12 * 8];
            let k: Array = (&k_data[..], &[2_i32, 4, 12, 8][..]).try_into().expect("k");
            let v: Array = (&v_data[..], &[2_i32, 4, 12, 8][..]).try_into().expect("v");

            // per_row_lens = [0, 12]: row 0 is skipped, row 1 writes 12 tokens.
            let per_row_lens = [0_i32, 12];
            kv.update_and_fetch(&k, &v, &per_row_lens)
                .expect("update_and_fetch");

            assert_eq!(
                kv.offsets(),
                &[0_i32, 12],
                "KVCache row 0 should stay 0 (skipped), row 1 should be 12"
            );

            println!(
                "[zero_len] KVCache offsets after [0,12] write: {:?}",
                kv.offsets()
            );
        }

        // ── GatedDeltaCache::advance: row 0 advance 0, row 1 advance 12 ────
        {
            // Params (b=2, kernel_size=4, conv_dim=8, hv=4, dv=8, dk=8, Bfloat16, cap=20)
            // mirror gated_delta.rs unit test helper `make_cache_b` — arbitrary small
            // values for testing the per-row offset invariant, not Qwen3.5 production
            // sizes.
            let mut gdc = GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 20)
                .expect("gdc new");

            gdc.advance(&[0_i32, 12]).expect("advance [0, 12]");

            assert_eq!(
                gdc.offsets(),
                &[0_i32, 12],
                "GatedDeltaCache row 0 should stay 0 (no advance), row 1 should be 12"
            );

            println!(
                "[zero_len] GatedDeltaCache offsets after advance [0,12]: {:?}",
                gdc.offsets()
            );
        }
    })
    .await
    .expect("zero_len join");
}
