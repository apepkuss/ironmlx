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
    spawn_scheduler_actor(model, b_max, Duration::from_millis(5), 32, 32768, meta)
        .expect("spawn_scheduler_actor")
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
