//! B1-p2.5 production hardening integration tests.
//!
//! G1: startup memory budget validation — MemoryBudgetError when
//!     IRONMLX_TOTAL_RAM_BYTES is too small.
//! G2: runtime admission gate — 3rd admit rejected when scheduler is
//!     at b_max with queue_max=0.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let loader = Loader::open(Path::new(&model_dir)).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

/// Startup budget gate: force IRONMLX_TOTAL_RAM_BYTES to 4 GiB so that
/// b_max=4 × cap=32768 × 114688 bytes/token ≈ 14 GiB vastly exceeds the
/// available budget (4 GiB − model weight − 2 GiB safety margin).
/// Expects spawn_scheduler_actor to return MemoryBudgetError.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_5_startup_rejects_overcommit() {
    // 4 GiB RAM override — far too small for 4 × 32768 × 114688 ≈ 14 GiB
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "4294967296");

    let (model, _tok) = load_fixture();
    let meta = model.lock().await.model_meta();
    let result = spawn_scheduler_actor(model, 4, Duration::from_millis(5), 32, 32768, meta);

    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");

    let err = result
        .err()
        .expect("expected MemoryBudgetError on overcommit");
    let msg = format!("{err}");
    assert!(
        msg.contains("memory budget exceeded"),
        "unexpected msg: {msg}"
    );
    assert!(msg.contains("Lower"), "hint missing in msg: {msg}");
}

/// Admission gate rejects when full: b_max=2 + queue_max=0.
/// Send 3 admits into a 2-slot scheduler simultaneously (all go into the
/// channel before the driver can drain them); driver handles admits 1+2 in
/// drain_window — both succeed; admit 3 arrives while saturated →
/// QueueFull error (queue_max=0 → immediate reject).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_5_admission_gate_rejects_when_full() {
    // 16 GiB — comfortable for startup validation (b_max=2, cap=2048,
    // 2 × 2048 × 114688 ≈ 448 MB << 16 − weight − 2 GiB available).
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "17179869184");

    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    // admission_queue_max=0: any admit arriving when scheduler is full
    // is immediately rejected with QueueFull (no queuing).
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        0, // queue_max=0 → immediate reject on saturation
        2048,
        meta,
    )
    .expect("spawn ok with 16 GiB budget");

    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");

    let prompt = tokenizer.encode("Hello", false).unwrap();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();
    let make = || GenerateRequest {
        prompt_ids: prompt.clone(),
        max_new_tokens: 512,
        sampler: Sampler::greedy(),
        stop_token_ids: stop_tokens.clone(),
        prefill_chunk_size: 128,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    };

    // Send all 3 admits into the channel before the driver can drain any
    // of them. Channel capacity is 64 so all sends are instant.
    let (tx0, rx0) = tokio::sync::oneshot::channel();
    let (tx1, rx1) = tokio::sync::oneshot::channel();
    let (tx2, rx2) = tokio::sync::oneshot::channel();

    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make(),
            reply_tx: tx0,
        })
        .await
        .unwrap();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make(),
            reply_tx: tx1,
        })
        .await
        .unwrap();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make(),
            reply_tx: tx2,
        })
        .await
        .unwrap();

    // Wait for all 3 replies. The driver processes first_cmd (admit 0)
    // immediately, then drain_window picks up admits 1 and 2.
    // When active_count reaches b_max=2, saturated=true; admit 2 goes to
    // enqueue_or_reject with queue_max=0 → QueueFull error.
    let _r0 = rx0.await.expect("reply 0").expect("admit 0 ok");
    let _r1 = rx1.await.expect("reply 1").expect("admit 1 ok");
    let admit_err = rx2
        .await
        .expect("reply 2")
        .err()
        .expect("3rd admit should fail");

    let msg = format!("{admit_err}");
    assert!(
        msg.contains("memory budget") || msg.contains("scheduler full") || msg.contains("queue"),
        "unexpected error message: {msg}"
    );

    drop(handle);
}
