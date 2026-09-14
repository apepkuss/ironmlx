//! B1-p2.3d integration scenarios for admission queue + config exposure.
//!
//! Scenarios drive `spawn_scheduler_actor` directly (S1/S3/S4) and via
//! the HTTP server bound to a random localhost port (S2/S5).
//!
//! Reference fixtures: `tests/fixtures/qwen35_vl/multi_image/` (unused
//! here — text-only suite).

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::model_input::IMAGE_TOKEN_ID;
use ironmlx_lm::models::Qwen35Model;
use ironmlx_runtime::core::generation_types::GenerateRequest;
use tokio::sync::Mutex;
use {
    ironmlx_lm::core::chat_template::Message, ironmlx_lm::core::loader::Loader,
    ironmlx_lm::core::tokenizer::Tokenizer,
};
use {
    ironmlx_runtime::core::scheduler_actor::spawn_scheduler_actor,
    ironmlx_runtime::core::scheduler_actor::SchedulerCommand,
};

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN35_MODEL") {
        return PathBuf::from(p);
    }
    let glob = format!(
        "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
        std::env::var("HOME").unwrap()
    );
    std::fs::read_dir(&glob)
        .expect("snapshots dir")
        .filter_map(|e| e.ok())
        .next()
        .expect("snapshot")
        .path()
}

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let p = model_path();
    let loader = Loader::open_multimodal(&p).expect("Loader::open_multimodal");
    let tok = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader(&loader).expect("model");
    (Arc::new(Mutex::new(model)), Arc::new(tok))
}

fn make_req(tokenizer: &Tokenizer, text: &str, max_new: usize) -> GenerateRequest {
    make_req_with_stop(tokenizer, text, max_new, tokenizer.eos_token_ids().to_vec())
}

/// Variant with explicit stop_token_ids (pass vec![] to disable EOS stopping).
fn make_req_with_stop(
    tokenizer: &Tokenizer,
    text: &str,
    max_new: usize,
    stop_token_ids: Vec<u32>,
) -> GenerateRequest {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .unwrap();
    let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        constraint: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy: needs QWEN35_MODEL
async fn queue_drains_fifo_at_bmax2_c4() {
    // b_max=2, queue_max=8; submit 4 requests back-to-back. All 4 must
    // complete; queue_depth_peak >= 2 (2 had to queue).
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        8,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    let texts = ["Hello", "World", "Goodbye", "Farewell"];
    let mut replies = Vec::new();
    for t in texts {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 8),
                reply_tx: tx,
            })
            .await
            .unwrap();
        replies.push(rx);
    }

    // Drain all 4 — each must reach a finish_reason.
    let mut finishes = 0;
    for rx in replies {
        let reply = rx.await.expect("rx").expect("admit ok");
        let mut event_rx = reply.event_rx;
        while let Some(ev) = event_rx.recv().await {
            if ev.finish_reason.is_some() {
                finishes += 1;
                break;
            }
        }
    }
    assert_eq!(finishes, 4, "expected 4 finishes, got {finishes}");

    let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
    assert!(peak >= 2, "expected queue_depth_peak >= 2, got {peak}");
    let rejected = handle.queue_rejected.load(Ordering::Relaxed);
    assert_eq!(rejected, 0, "expected zero rejections, got {rejected}");

    drop(handle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn queue_overflow_returns_err_via_actor() {
    // b_max=2, queue_max=3; submit 6 requests. The first 2 are admitted
    // immediately (both in flight). Then we wait for the rolling decode
    // loop to start. Then we fire 4 more: first 3 enqueue (queue_max=3),
    // 4th (D) overflows → Err("admission queue full").
    //
    // max_new=1024 for A+B ensures they stay in Decoding long enough that
    // the saturation burst sees active_count == b_max. On Metal GPU, decode
    // is ~150 ms/step regardless of Rust opt-level; 1024 tokens ≈ 150s.
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        3,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    // stop_token_ids: vec![] (disable EOS) so A/B don't short-circuit on
    // "Hello"/"World" + cold GPU; ensures the burst below finds A/B still
    // active in Decoding. Pre-P5 relied on accumulated thermal load to slow
    // decode — fragile, replaced with explicit no-EOS.
    let (tx1, _rx1) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req_with_stop(&tokenizer, "Hello", 1024, vec![]),
            reply_tx: tx1,
        })
        .await
        .unwrap();
    let (tx2, _rx2) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req_with_stop(&tokenizer, "World", 1024, vec![]),
            reply_tx: tx2,
        })
        .await
        .unwrap();

    // Poll until batch_count >= 1 (prefill has started, A B slots allocated)
    // then sleep 200ms to allow at least one rolling decode Step to fire,
    // confirming A B are in Decoding before the saturation burst.
    let deadline_bat = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        if handle.batch_count.load(Ordering::Relaxed) >= 1 {
            break;
        }
        if tokio::time::Instant::now() > deadline_bat {
            panic!("batch_count never reached 1 within 60s");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Wait for at least one decode Step tick (confirms rolling loop active).
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Saturation burst: 4 requests → first 3 enqueue, 4th (D) overflows.
    let mut later_rxs = Vec::new();
    for t in ["A", "B", "C", "D"] {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 8),
                reply_tx: tx,
            })
            .await
            .unwrap();
        later_rxs.push(rx);
    }

    // The 4th of these (= 6th overall) must reject. 30s covers worst-case
    // decode-step latency under GPU contention from prior tests.
    let last_reply = tokio::time::timeout(Duration::from_secs(30), later_rxs.pop().unwrap())
        .await
        .expect("last_reply timeout (D should have been rejected immediately)")
        .expect("oneshot recv");
    match last_reply {
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("admission queue full"),
                "expected 'admission queue full', got: {msg}"
            );
        }
        Ok(_) => {
            let admit_after = handle.admit_count.load(Ordering::Relaxed);
            let batch_after = handle.batch_count.load(Ordering::Relaxed);
            let rejected = handle.queue_rejected.load(Ordering::Relaxed);
            let queue_peak = handle.queue_depth_peak.load(Ordering::Relaxed);
            panic!(
                "expected Err for 6th admit (D), got Ok; admit_count={admit_after} \
                 batch_count={batch_after} queue_rejected={rejected} queue_depth_peak={queue_peak}"
            );
        }
    }

    // Verify queue_rejected counter incremented.
    let rejected = handle.queue_rejected.load(Ordering::Relaxed);
    assert!(
        rejected >= 1,
        "expected queue_rejected >= 1, got {rejected}"
    );

    // Drop handle — driver_loop shuts down, in-flight A+B are discarded
    // (no need to drain 1024 tokens in tests).
    drop(handle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn admission_deadline_config_observed() {
    // admission_deadline_ms = 30 (vs. default 5). Two admits arriving 20ms
    // apart should land in the same batch (drain_window covers both).
    // batch_count should be 1 (not 2).
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle = spawn_scheduler_actor(
        model.clone(),
        4,
        Duration::from_millis(30),
        32,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    let batch_before = handle.batch_count.load(Ordering::Relaxed);

    let (tx1, rx1) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "first", 5),
            reply_tx: tx1,
        })
        .await
        .unwrap();

    // Sleep 20ms — still within the 30ms admission window. The driver_loop
    // has issued the deadline timer; the second admit lands while the
    // first batch is still in the drain_window.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (tx2, rx2) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "second", 5),
            reply_tx: tx2,
        })
        .await
        .unwrap();

    // Drain both replies.
    let r1 = rx1.await.unwrap().unwrap();
    let r2 = rx2.await.unwrap().unwrap();
    for mut rx in [r1.event_rx, r2.event_rx] {
        while let Some(ev) = rx.recv().await {
            if ev.finish_reason.is_some() {
                break;
            }
        }
    }

    let batch_delta = handle.batch_count.load(Ordering::Relaxed) - batch_before;
    assert_eq!(
        batch_delta, 1,
        "expected single batch (deadline=30ms covers both admits), got {batch_delta}"
    );

    drop(handle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn b_max_config_8_no_queue() {
    // B1-p2.5 budget gate: this test intentionally uses b_max=8 × cap=32768
    // (32 GiB nominal KV cache) that would exceed a real 32 GiB Mac's budget.
    // Override IRONMLX_TOTAL_RAM_BYTES to simulate a 64 GiB machine so the
    // budget validation passes. EnvGuard Drop cleans up even on panic.
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
        }
    }
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "68719476736"); // 64 GiB
    let _guard = EnvGuard;

    // b_max=8 + admission_deadline_ms=50: 6 concurrent admits all fit in
    // one batch (queue stays empty).
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle = spawn_scheduler_actor(
        model.clone(),
        8,
        Duration::from_millis(50),
        32,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    let texts = ["a", "b", "c", "d", "e", "f"];
    let mut rxs = Vec::new();
    for t in texts {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 5),
                reply_tx: tx,
            })
            .await
            .unwrap();
        rxs.push(rx);
    }

    for rx in rxs {
        let r = rx.await.unwrap().unwrap();
        let mut e = r.event_rx;
        while let Some(ev) = e.recv().await {
            if ev.finish_reason.is_some() {
                break;
            }
        }
    }

    let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
    assert_eq!(
        peak, 0,
        "expected queue_depth_peak == 0 (b_max=8 absorbs 6 admits), got {peak}"
    );

    drop(handle);
}
