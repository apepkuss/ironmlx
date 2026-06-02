//! GLM-4.7-Flash (`glm4_moe_lite`) continuous-batching correctness test.
//!
//! The correctness gate for `--b-max > 1`: drive the REAL scheduler with
//! `b_max=2` to generate greedy (temperature 0) continuations for TWO
//! DIFFERENT-LENGTH prompts concurrently, and assert each prompt's generated
//! token sequence is BIT-IDENTICAL to the SAME prompt's `b_max=1` SERIAL
//! continuation (per-token equality, no tolerance).
//!
//! The two prompt lengths differ, so the B>1 step exercises per-row
//! heterogeneous cache lengths: per-row RoPE array offsets, the per-row decode
//! mask `[B,1,1,Lc]` folded into MLA `pe_scores [B,H,1,Lc]`, and the
//! `MlaLatentCache` history fetch under non-uniform offsets — plus B>1 batched
//! prefill of the two different-length prompts (right-padded `[B, T_max]` with
//! the engine attention mask). Greedy decode is deterministic, so concurrent
//! batching that is numerically correct must reproduce the serial token stream
//! exactly.
//!
//! ## Why the serial baseline runs through the SCHEDULER (b_max=1), not
//! ## `GenerationStream`
//!
//! The property under test is "B>1 batching must not corrupt per-row output":
//! row R decoded alongside other rows must equal row R decoded alone, *in the
//! same engine*. The correct reference is therefore the scheduler at b_max=1
//! (identical prefill + decode code path, one slot). `GenerationStream` is a
//! DIFFERENT engine path: it runs single-shot prefill (`forward_on [1, T]`)
//! whereas the scheduler splits prefill into a prefix forward (`[1, T-1]`) +
//! last-token forward (`[1, 1]`). Those two orderings differ only in bf16
//! rounding, but on a near-tie that benign delta can flip a greedy argmax many
//! decode steps later — a numerical-ordering artifact, NOT a batching bug.
//! Comparing scheduler-b2 against scheduler-b1 isolates the batching invariant
//! cleanly. (The first-token logits parity vs the `mlx_lm` reference is pinned
//! separately in `glm4_moe_lite_parity.rs`.)
//!
//! Env-gated: skips (with an eprintln) when no GLM checkpoint is present. Run:
//!   GLM47_MODEL_DIR=$(echo ~/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/*) \
//!     MLX_DIR=/tmp/ironmlx-perf-mlx-install-3f6c3113f734 \
//!     cargo test -p ironmlx --release --test glm4_moe_lite_cb -- --ignored --nocapture --test-threads=1

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx::core::generate::GenerateRequest;
use ironmlx::core::memory_budget::ModelMeta;
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::StepEvent;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, AdmitReply, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::glm4_moe_lite::Glm4MoeLiteModel;

/// Resolve the GLM-4.7-Flash snapshot directory. Honors `GLM47_MODEL_DIR`
/// first, then falls back to the default HF cache layout. Returns `None`
/// (caller skips) when nothing is found.
fn glm_snapshot_dir() -> Option<String> {
    if let Ok(p) = std::env::var("GLM47_MODEL_DIR") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
        eprintln!("GLM47_MODEL_DIR={p} does not exist");
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        format!("{home}/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots");
    let entries = std::fs::read_dir(&snapshots).ok()?;
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .map(|p| p.to_string_lossy().into_owned())
}

fn load_fixture() -> Option<(Arc<Mutex<Glm4MoeLiteModel>>, Arc<Tokenizer>)> {
    let dir = glm_snapshot_dir()?;
    eprintln!("loading GLM-4.7-Flash from {dir}");
    let loader = Loader::open(Path::new(&dir)).expect("Loader::open");
    let model = Glm4MoeLiteModel::from_loader(&loader).expect("Glm4MoeLiteModel::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    Some((Arc::new(Mutex::new(model)), Arc::new(tokenizer)))
}

/// Render a chat prompt → token ids (chat-template, thinking-mode off). The two
/// `text` lengths differ, which is the property this test relies on.
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![ironmlx::core::Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

/// Greedy request with stop tokens DISABLED (`vec![]`) so both the serial and
/// concurrent runs generate exactly `max_new_tokens` — a divergence cannot hide
/// behind an early EOS, and a single mismatched token is a hard failure.
fn make_request(prompt_ids: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: vec![],
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    }
}

async fn submit_admit(
    cmd_tx: &tokio::sync::mpsc::Sender<SchedulerCommand>,
    req: GenerateRequest,
) -> ironmlx::Result<AdmitReply> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    cmd_tx
        .send(SchedulerCommand::Admit {
            request: req,
            reply_tx,
        })
        .await
        .map_err(|e| anyhow::anyhow!("cmd_tx.send: {e:?}"))?;
    reply_rx
        .await
        .map_err(|e| anyhow::anyhow!("reply_rx.await: {e:?}"))?
}

async fn drain_until_finished(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<StepEvent>,
) -> Vec<StepEvent> {
    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        let done = ev.finish_reason.is_some();
        events.push(ev);
        if done {
            break;
        }
    }
    events
}

/// Spawn a fresh `b_max`-slot scheduler actor for this model.
fn spawn(
    model: Arc<Mutex<Glm4MoeLiteModel>>,
    b_max: usize,
    meta: ModelMeta,
) -> SchedulerActorHandle {
    spawn_scheduler_actor(model, b_max, Duration::from_millis(5), 32, 32768, 256, meta)
        .expect("spawn_scheduler_actor")
}

/// Block (cooperatively) until the actor's rolling decode loop has driven at
/// least `target` outer batches, i.e. the already-admitted rows are confirmed
/// to be in `Decoding` and stepping. Mirrors the mid-admit timing signal used
/// by `b1_p2_3c_3_continuous_batching.rs` (`batch_count` polling beats a fixed
/// sleep, which is fragile on a cold GPU where prefill can exceed 200ms). This
/// lets a follow-up `Admit` land WHILE the batch is saturated, so it is queued
/// and later mid-admitted into a freed slot rather than starting a fresh batch.
async fn wait_for_batch_count(handle: &SchedulerActorHandle, target: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        if handle.batch_count.load(Ordering::Relaxed) >= target {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("batch_count never reached {target} within 60s — prefill stalled");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Generate one prompt's greedy continuation SERIALLY through a fresh b_max=1
/// scheduler actor (same engine path as the concurrent run, one slot).
async fn run_serial(
    model: Arc<Mutex<Glm4MoeLiteModel>>,
    meta: ModelMeta,
    prompt: Vec<u32>,
    max_new: usize,
) -> Vec<u32> {
    let handle = spawn(model, 1, meta);
    let reply = submit_admit(&handle.cmd_tx, make_request(prompt, max_new))
        .await
        .expect("admit serial");
    let mut rx = reply.event_rx;
    let events = drain_until_finished(&mut rx).await;
    events.into_iter().map(|e| e.token).collect()
}

/// THE B>1 CORRECTNESS GATE: concurrent (`b_max=2`) greedy decode of two
/// different-length prompts is bit-identical to each prompt's serial
/// (`b_max=1`) greedy decode through the same scheduler engine.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b_gt_1_decode_matches_serial() {
    let Some((model, tokenizer)) = load_fixture() else {
        eprintln!("skip: no GLM-4.7-Flash weights (set GLM47_MODEL_DIR)");
        return;
    };
    let meta = model.lock().await.model_meta();

    // Two DIFFERENT-LENGTH prompts → heterogeneous per-row cache lengths.
    let prompt_a = tokenize_prompt(&tokenizer, "Hi");
    let prompt_b = tokenize_prompt(
        &tokenizer,
        "Explain in detail how a transformer language model processes a sequence of tokens.",
    );
    assert_ne!(
        prompt_a.len(),
        prompt_b.len(),
        "prompts must differ in length to exercise per-row heterogeneous cache offsets \
         (got A={} B={})",
        prompt_a.len(),
        prompt_b.len()
    );
    eprintln!(
        "prompt lengths: A={} B={} (delta={})",
        prompt_a.len(),
        prompt_b.len(),
        (prompt_b.len() as i64 - prompt_a.len() as i64).abs()
    );

    let max_new: usize = 24;

    // --- b_max=1 serial baselines (one fresh scheduler each, same engine). ---
    let serial_a = run_serial(model.clone(), meta, prompt_a.clone(), max_new).await;
    let serial_b = run_serial(model.clone(), meta, prompt_b.clone(), max_new).await;

    // --- b_max=2 concurrent run via the real SchedulerActor. ---
    let handle = spawn(model.clone(), 2, meta);
    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a.clone(), max_new))
        .await
        .expect("admit A");
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b.clone(), max_new))
        .await
        .expect("admit B");
    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;

    // Drain both concurrently — they share the b_max=2 batch and decode in
    // lock-step rows until each hits its `length` cap.
    let (events_a, events_b) = tokio::join!(
        drain_until_finished(&mut rx_a),
        drain_until_finished(&mut rx_b)
    );

    let cb_a: Vec<u32> = events_a.iter().map(|e| e.token).collect();
    let cb_b: Vec<u32> = events_b.iter().map(|e| e.token).collect();

    eprintln!("serial_a     (b=1) = {serial_a:?}");
    eprintln!("concurrent_a (b=2) = {cb_a:?}");
    eprintln!("serial_b     (b=1) = {serial_b:?}");
    eprintln!("concurrent_b (b=2) = {cb_b:?}");

    // Each run produces exactly `max_new` tokens (stop disabled), finishing
    // with reason "length".
    assert_eq!(
        events_a.len(),
        max_new,
        "A produced {} events; want {max_new}",
        events_a.len()
    );
    assert_eq!(events_a.last().unwrap().finish_reason, Some("length"));
    assert_eq!(
        events_b.len(),
        max_new,
        "B produced {} events; want {max_new}",
        events_b.len()
    );
    assert_eq!(events_b.last().unwrap().finish_reason, Some("length"));
    assert_eq!(serial_a.len(), max_new, "serial A length");
    assert_eq!(serial_b.len(), max_new, "serial B length");

    // HARD GATE: per-token bit-identical (greedy is deterministic).
    assert_eq!(
        cb_a, serial_a,
        "prompt A: b_max=2 concurrent decode diverged from b_max=1 serial"
    );
    assert_eq!(
        cb_b, serial_b,
        "prompt B: b_max=2 concurrent decode diverged from b_max=1 serial"
    );

    eprintln!(
        "GLM-4.7-Flash B>1 OK: both prompts' b_max=2 concurrent decode == b_max=1 serial \
         (per-token identical, {max_new} tokens each)"
    );
}

/// THE MID-ADMIT + ROW-REUSE CORRECTNESS GATE (Task 4 Step 1).
///
/// Exercises the continuous-batching machinery that the simpler B>1 gate above
/// does NOT touch: mid-flight admit into a freed slot (`admit_mid_begin` /
/// `admit_mid_chunk` / `admit_mid_finalize`) plus cache row compaction
/// (`rebuild_cache_layout` → `adopt_cache_row_layers`) on the `LayerCache::Mla`
/// arm added in Task 2.
///
/// Scenario (b_max=2, three different-length prompts, greedy temperature 0,
/// stop tokens disabled, fixed `max_new`):
///   1. Admit LONG-prompt A (small `max_new` so it FINISHES first) and
///      SHORT-prompt B (large `max_new`). Both fill the two slots and prefill
///      together (rows 0 and 1).
///   2. Poll `batch_count` so the follow-up admit lands while the batch is
///      saturated. Submit MEDIUM-prompt C — with both slots busy + `Decoding`,
///      C is QUEUED (it cannot start a fresh batch).
///   3. A reaches its `max_new` first and is GC'd, freeing slot row 0. The
///      driver's post-step queue drain pulls C and mid-admits it: B=1 prefill
///      into a temp Mla cache, then `admit_mid_finalize` adopts that temp row
///      into the freed slot (`MlaLatentCache::adopt_row_from`). This is GENUINE
///      slot reuse — C reuses the row A vacated.
///   4. Once A finishes, the surviving row B (compact cache row 1) is
///      relocated to compact row 0 on the next decode step
///      (`rebuild_cache_layout` with a DIFFERENT src/dst row → the Mla arm of
///      `adopt_cache_row_layers` performs a real buffer migration). C then
///      decodes alongside B from the reused slot.
///
/// HARD GATE: each prompt's generated token stream is bit-identical to that
/// same prompt's b_max=1 SERIAL stream through the same scheduler engine. A
/// single mismatched token fails — no tolerance. A bug in the Task 1/2 Mla
/// adopt/rebuild path (row migration during compaction, per-row offset
/// bookkeeping after compaction) would surface as a divergence HERE while the
/// simpler B>1 gate (no finish/reuse/compaction) still passes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn mid_admit_row_reuse_matches_serial() {
    let Some((model, tokenizer)) = load_fixture() else {
        eprintln!("skip: no GLM-4.7-Flash weights (set GLM47_MODEL_DIR)");
        return;
    };
    let meta = model.lock().await.model_meta();

    // Three DIFFERENT-LENGTH prompts → heterogeneous per-row cache lengths
    // through prefill, compaction, and mid-admit.
    let prompt_a = tokenize_prompt(
        &tokenizer,
        "Explain in detail how a transformer language model processes a sequence of tokens, \
         step by step, including attention and the feed-forward blocks.",
    );
    let prompt_b = tokenize_prompt(&tokenizer, "Hi");
    let prompt_c = tokenize_prompt(
        &tokenizer,
        "Summarize the theory of relativity for a curious child.",
    );
    // All three lengths must differ to exercise per-row heterogeneous cache
    // offsets through prefill, compaction, and mid-admit.
    assert_ne!(
        prompt_a.len(),
        prompt_b.len(),
        "prompts A/B must differ in length (got A={} B={})",
        prompt_a.len(),
        prompt_b.len()
    );
    assert_ne!(
        prompt_a.len(),
        prompt_c.len(),
        "prompts A/C must differ in length (got A={} C={})",
        prompt_a.len(),
        prompt_c.len()
    );
    assert_ne!(
        prompt_b.len(),
        prompt_c.len(),
        "prompts B/C must differ in length (got B={} C={})",
        prompt_b.len(),
        prompt_c.len()
    );
    eprintln!(
        "prompt lengths: A={} B={} C={}",
        prompt_a.len(),
        prompt_b.len(),
        prompt_c.len()
    );

    // A finishes FIRST (small cap) so its slot is freed and reused by C.
    // B outlives A so a surviving row gets compacted. C is the mid-admitted
    // reuser; it must outlast the moment of admit so it decodes post-compaction.
    let max_new_a: usize = 6;
    let max_new_b: usize = 24;
    let max_new_c: usize = 16;

    // --- b_max=1 serial baselines (one fresh scheduler each, same engine). ---
    let serial_a = run_serial(model.clone(), meta, prompt_a.clone(), max_new_a).await;
    let serial_b = run_serial(model.clone(), meta, prompt_b.clone(), max_new_b).await;
    let serial_c = run_serial(model.clone(), meta, prompt_c.clone(), max_new_c).await;

    // --- b_max=2 continuous-batching run with mid-flight admit + reuse. ---
    let handle = spawn(model.clone(), 2, meta);

    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a.clone(), max_new_a))
        .await
        .expect("admit A");
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b.clone(), max_new_b))
        .await
        .expect("admit B");
    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;

    // Wait until A+B are decoding (≥1 outer batch) so C is queued, not
    // started as a fresh batch.
    wait_for_batch_count(&handle, 1).await;

    // Submit C while both slots are busy → queued, then mid-admitted into the
    // slot A frees when it finishes.
    let reply_c = submit_admit(&handle.cmd_tx, make_request(prompt_c.clone(), max_new_c))
        .await
        .expect("admit C (queued, mid-admitted on A's freed slot)");
    let mut rx_c = reply_c.event_rx;

    // Drain all three. A finishes first (freeing its row), C is mid-admitted
    // into it, B + C decode through the compacted layout.
    let (events_a, events_b, events_c) = tokio::join!(
        drain_until_finished(&mut rx_a),
        drain_until_finished(&mut rx_b),
        drain_until_finished(&mut rx_c),
    );

    let cb_a: Vec<u32> = events_a.iter().map(|e| e.token).collect();
    let cb_b: Vec<u32> = events_b.iter().map(|e| e.token).collect();
    let cb_c: Vec<u32> = events_c.iter().map(|e| e.token).collect();

    eprintln!("serial_a   (b=1)       = {serial_a:?}");
    eprintln!("cb_a       (mid-admit) = {cb_a:?}");
    eprintln!("serial_b   (b=1)       = {serial_b:?}");
    eprintln!("cb_b       (mid-admit) = {cb_b:?}");
    eprintln!("serial_c   (b=1)       = {serial_c:?}");
    eprintln!("cb_c       (reused)    = {cb_c:?}");

    // Each run produces exactly `max_new` tokens (stop disabled), finishing
    // with reason "length".
    assert_eq!(
        events_a.len(),
        max_new_a,
        "A produced {} events; want {max_new_a}",
        events_a.len()
    );
    assert_eq!(events_a.last().unwrap().finish_reason, Some("length"));
    assert_eq!(
        events_b.len(),
        max_new_b,
        "B produced {} events; want {max_new_b}",
        events_b.len()
    );
    assert_eq!(events_b.last().unwrap().finish_reason, Some("length"));
    assert_eq!(
        events_c.len(),
        max_new_c,
        "C produced {} events; want {max_new_c}",
        events_c.len()
    );
    assert_eq!(events_c.last().unwrap().finish_reason, Some("length"));
    assert_eq!(serial_a.len(), max_new_a, "serial A length");
    assert_eq!(serial_b.len(), max_new_b, "serial B length");
    assert_eq!(serial_c.len(), max_new_c, "serial C length");

    // HARD GATE: per-token bit-identical to each prompt's serial baseline.
    assert_eq!(
        cb_a, serial_a,
        "prompt A (finishes-first): mid-admit batch decode diverged from b_max=1 serial"
    );
    assert_eq!(
        cb_b, serial_b,
        "prompt B (survives compaction): mid-admit batch decode diverged from b_max=1 serial"
    );
    assert_eq!(
        cb_c, serial_c,
        "prompt C (mid-admitted into A's reused slot): decode diverged from b_max=1 serial"
    );

    eprintln!(
        "GLM-4.7-Flash mid-admit + row-reuse OK: A(finish)/B(compact)/C(reuse) all \
         per-token identical to b_max=1 serial through the Mla adopt/rebuild path"
    );
}
