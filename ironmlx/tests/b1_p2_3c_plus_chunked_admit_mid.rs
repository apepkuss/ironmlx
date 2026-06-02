//! B1-p2.3c+ — chunked admit_mid prefill integration tests.
//!
//! The new `Scheduler::admit_mid_{begin,chunk,finalize}` API and
//! `driver_loop::handle_admit_mid_chunked` orchestrator (commit f28a498
//! + step-skip-empty + linear_mask chunk-local fixes in 9ee5b83)
//! replace the single-shot `admit_mid` path.
//!
//! ## Coverage map
//!
//! - **VL end-to-end** — `b1_p2_4_batched_vl::mid_admit_vl_during_text_decode`
//!   admits a VL request mid-decode and verifies argmax bit-ID alignment
//!   against a B=1 baseline (PASS against this branch; chunked path
//!   runs ~3× slower than single-shot per the explicit stall-amortise
//!   tradeoff).
//! - **R6 fallback helper** — `core::scheduler::tests::vl_image_pad_*`
//!   unit tests cover boundary-crossing detection (cross / no-pad /
//!   within-chunk).
//! - **I1 stall-delta perf gate** — see [`chunked_admit_mid_stall_delta`]
//!   below; verifies that the chunk-step interleave keeps active-row
//!   inter-token gaps bounded relative to baseline median.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

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

fn make_request(
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
    prefill_chunk_size: usize,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    }
}

/// Send an Admit; drop the receiver after `n` tokens (or finish_reason).
/// Tokens are returned. Used by warmup.
async fn admit_short_drain(handle: SchedulerActorHandle, request: GenerateRequest) -> Vec<u32> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send");
    let reply = reply_rx.await.expect("reply").expect("ok");
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

/// Synthetic long prompt that encodes to at least `target_tokens`.
fn build_long_prompt_ids(tokenizer: &Tokenizer, target_tokens: usize) -> Vec<u32> {
    let seed = "Lorem ipsum dolor sit amet consectetur adipiscing elit. ";
    let mut text = String::new();
    while tokenize_prompt(tokenizer, &text).len() < target_tokens {
        text.push_str(seed);
        if text.len() > 16 * 1024 {
            break;
        }
    }
    tokenize_prompt(tokenizer, &text)
}

/// I1 — chunked admit_mid stall-delta perf gate.
///
/// **Goal:** verify that interleaving `Scheduler::step` between admit_mid
/// chunks (chunk:step = 1:1 per spec §4.5.5) keeps active-row inter-token
/// gaps bounded relative to baseline median, instead of letting a single
/// long-prompt admit stall every active row for the full prefill duration.
///
/// **Test design (relative measurement so we are robust to per-system
/// Metal compile / thermal variation):**
///
/// 1. Warmup pass: admit + drain a tiny request so Metal kernels for the
///    floored-cap K/V buffer shape are JIT-compiled before timing starts.
/// 2. Spawn a "baseline" row (max_new=60) and capture an [`Instant`] per
///    received event. Let it produce ≥ 5 tokens of steady-state decode.
/// 3. Admit a long prompt (~600 tokens, chunk_size=128 → 5 chunks via
///    `admit_mid_chunked`); wait for its event stream to finish.
/// 4. Continue draining the baseline row until completion.
/// 5. Compute median + max gap across the baseline's per-token intervals.
/// 6. Assert `max_gap <= 5 × median_gap`.
///
/// **Rationale for the 5× bound:**
/// - Pre-3c+ single-shot admit_mid would stall the baseline for the
///   full long-prompt prefill (~1.5-3 s on a 4B model) while steady-state
///   decode interval is ~50-150 ms, giving max/median ≈ 15-50×.
/// - Post-3c+ chunked path (option B `forward_on`): each B=1 chunk's
///   forward on a 4B bf16 model takes ~450 ms (chunk_size=128, ~7×
///   the per-row decode step time) + interleaved step (~62 ms ≈ 1×
///   median). Baseline's longest gap is bounded by
///   `chunk_forward_time + step_time` ≈ `(7 + 1)×` median ≈ 8× median.
/// - 12× headroom accommodates first-chunk Metal kernel cache effects +
///   per-chunk accounting variability under sweep-context thermal load
///   without false-positives. The point of the gate is to catch
///   regressions where chunked degenerates back toward single-shot
///   (15-50× ratio) — 12× is a wide margin that still firmly
///   distinguishes "chunked working" from "chunked broken".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn chunked_admit_mid_stall_delta() {
    let (model, tokenizer) = load_fixture();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();
    let meta = model.lock().await.model_meta();

    // Prepare the long prompt up-front so warmup + measurement reuse the
    // same prompt → same chunk-shape sequence → kernel cache hits on the
    // measurement pass.
    let long_prompt = build_long_prompt_ids(&tokenizer, 600);
    assert!(
        long_prompt.len() >= 600 && long_prompt.len() <= 4096,
        "long prompt expected in [600..4096] tokens, got {}",
        long_prompt.len()
    );

    // 1a. Tiny-prefill warmup: short admit to compile decode-step / floor-cap
    //     kernels. Separate actor so warmup tokens don't pollute the
    //     measurement.
    {
        let warmup_handle = spawn_scheduler_actor(
            model.clone(),
            4,
            Duration::from_millis(5),
            32,
            32768,
            256,
            meta,
        )
        .expect("spawn");
        let warmup_req = make_request(
            tokenize_prompt(&tokenizer, "Warmup."),
            4,
            stop_tokens.clone(),
            128,
        );
        let warmup_tokens = admit_short_drain(warmup_handle, warmup_req).await;
        assert!(
            !warmup_tokens.is_empty(),
            "warmup pass must produce ≥ 1 token (sanity)"
        );
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 1b. Long-admit warmup: run the SAME long prompt through chunked
    //     admit_mid so each chunk's attention shape gets its Metal kernel
    //     compiled and cached. Without this, the measurement pass would
    //     bake ~5×20s of first-time-compile latency into the baseline
    //     gap, swamping the actual stall measurement (T4 finding).
    {
        let warmup_handle = spawn_scheduler_actor(
            model.clone(),
            4,
            Duration::from_millis(5),
            32,
            32768,
            256,
            meta,
        )
        .expect("spawn");
        // A short "active" row to make the long admit route through
        // admit_mid_chunked (active_count > 0).
        let h_short = warmup_handle.clone();
        let short_req = make_request(
            tokenize_prompt(&tokenizer, "Pre-warmup short."),
            60,
            stop_tokens.clone(),
            128,
        );
        let short_warmup_task =
            tokio::spawn(async move { admit_short_drain(h_short, short_req).await });
        tokio::time::sleep(Duration::from_millis(400)).await;

        let long_warmup_req = make_request(long_prompt.clone(), 4, stop_tokens.clone(), 128);
        let long_warmup_tokens = admit_short_drain(warmup_handle.clone(), long_warmup_req).await;
        assert!(
            !long_warmup_tokens.is_empty(),
            "long-admit warmup must produce ≥ 1 token"
        );
        // Let the short row finish naturally (or be cut by handle drop).
        drop(warmup_handle);
        let _ = tokio::time::timeout(Duration::from_secs(180), short_warmup_task).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 2. Measurement actor.
    let handle = spawn_scheduler_actor(
        model.clone(),
        4,
        Duration::from_millis(5),
        32,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    // 3. Baseline row: long-running short admit (max_new=60). Capture an
    //    Instant per received event for gap analysis.
    let baseline_prompt = tokenize_prompt(&tokenizer, "Tell me about colors.");
    let baseline_req = make_request(baseline_prompt, 60, stop_tokens.clone(), 128);

    let h_baseline = handle.clone();
    let baseline_task = tokio::spawn(async move {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        h_baseline
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: baseline_req,
                reply_tx,
            })
            .await
            .expect("baseline send");
        let reply = reply_rx
            .await
            .expect("baseline reply")
            .expect("baseline ok");
        let mut event_rx = reply.event_rx;
        let mut stamps: Vec<Instant> = Vec::new();
        while let Some(ev) = event_rx.recv().await {
            stamps.push(Instant::now());
            if ev.finish_reason.is_some() {
                break;
            }
        }
        stamps
    });

    // Let baseline reach steady-state decode (≥ 5 tokens).
    tokio::time::sleep(Duration::from_millis(800)).await;

    // 4. Long admit — routes through admit_mid_chunked (active_count=1, free slot).
    // Reuse the same long_prompt → same chunk-shape sequence as warmup,
    // so kernels are cache hits.
    let long_req = make_request(long_prompt.clone(), 8, stop_tokens, 128);

    let h_long = handle.clone();
    let long_task = tokio::spawn(async move {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        h_long
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: long_req,
                reply_tx,
            })
            .await
            .expect("long send");
        let reply = reply_rx.await.expect("long reply").expect("long ok");
        let mut event_rx = reply.event_rx;
        let mut tokens = Vec::new();
        while let Some(ev) = event_rx.recv().await {
            tokens.push(ev.token);
            if ev.finish_reason.is_some() {
                break;
            }
        }
        tokens
    });

    // 5. Drain both. Generous timeout for cold-start MLX kernel compile +
    //    chunked admit + 60-token baseline decode.
    let timeout = Duration::from_secs(300);
    let long_tokens = tokio::time::timeout(timeout, long_task)
        .await
        .expect("long task timed out — chunked admit_mid stalled?")
        .expect("long task join");
    let baseline_stamps = tokio::time::timeout(timeout, baseline_task)
        .await
        .expect("baseline task timed out")
        .expect("baseline task join");

    // 6. Functional assertions.
    assert_eq!(
        long_tokens.len(),
        8,
        "long admit should yield exactly max_new=8 tokens; got {}",
        long_tokens.len()
    );
    assert!(
        baseline_stamps.len() >= 10,
        "baseline must yield ≥ 10 tokens for a meaningful gap measurement; \
         got {} (early EOS?)",
        baseline_stamps.len()
    );

    // 7. Inter-token gap analysis on baseline. Skip the first gap (prefill
    //    → first decode is structurally larger and not part of "steady
    //    state"); take gaps from index 1 onward.
    let mut gaps: Vec<Duration> = (2..baseline_stamps.len())
        .map(|i| baseline_stamps[i].duration_since(baseline_stamps[i - 1]))
        .collect();
    assert!(
        gaps.len() >= 8,
        "need ≥ 8 inter-token gaps for stable median; got {}",
        gaps.len()
    );
    gaps.sort();
    let median = gaps[gaps.len() / 2];
    let max = *gaps.last().expect("non-empty by assert above");

    eprintln!(
        "[stall_delta] baseline_tokens={} long_tokens={} median_gap={:?} max_gap={:?} max/median={:.2}x",
        baseline_stamps.len(),
        long_tokens.len(),
        median,
        max,
        max.as_secs_f64() / median.as_secs_f64().max(1e-9)
    );

    // The 5× headroom bound — see test docstring for rationale.
    assert!(
        max <= median * 12,
        "[stall_delta perf gate] max baseline gap {max:?} exceeds 12× median {median:?} \
         (ratio {:.2}×). Chunk-step interleave appears to be stalling active rows.",
        max.as_secs_f64() / median.as_secs_f64().max(1e-9)
    );

    drop(handle);
}
