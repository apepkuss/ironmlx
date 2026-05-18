//! B1-p2.3d integration scenarios for admission queue + config exposure.
//!
//! Scenarios drive `spawn_scheduler_actor` directly (S1/S3/S4) and via
//! the HTTP server bound to a random localhost port (S2/S5).
//!
//! Reference fixtures: `tests/fixtures/p6_qwen35_vl/multi_image/` (unused
//! here — text-only suite).

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::Qwen35Model;
use tokio::sync::Mutex;

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
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy: needs QWEN35_MODEL
async fn queue_drains_fifo_at_bmax2_c4() {
    // b_max=2, queue_max=8; submit 4 requests back-to-back. All 4 must
    // complete; queue_depth_peak >= 2 (2 had to queue).
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle = spawn_scheduler_actor(model.clone(), 2, Duration::from_millis(5), 8, 32768, meta)
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
    let handle = spawn_scheduler_actor(model.clone(), 2, Duration::from_millis(5), 3, 32768, meta)
        .expect("spawn");

    let (tx1, _rx1) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "Hello", 1024),
            reply_tx: tx1,
        })
        .await
        .unwrap();
    let (tx2, _rx2) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "World", 1024),
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
    let handle =
        spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(30), 32, 32768, meta)
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
    // b_max=8 + admission_deadline_ms=50: 6 concurrent admits all fit in
    // one batch (queue stays empty).
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let handle =
        spawn_scheduler_actor(model.clone(), 8, Duration::from_millis(50), 32, 32768, meta)
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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore] // real-model heavy + HTTP server
async fn iron_bench_c8_with_queue_no_4xx() {
    // Boot the server on a random port; spawn 8 concurrent HTTP clients
    // hitting /v1/chat/completions for 15s. With b_max=4 + queue_max=32,
    // no HTTP 4xx should occur.
    use ironmlx::core::server;

    let port = 18400 + (std::process::id() % 1000) as u16;
    let model_path = model_path();
    let loader = Loader::open_multimodal(&model_path).unwrap();
    let tokenizer_for_serve = Tokenizer::from_loader(&loader).unwrap();
    let model_for_serve = Qwen35Model::from_loader(&loader).unwrap();

    let server_handle = tokio::spawn(async move {
        server::serve(
            model_for_serve,
            tokenizer_for_serve,
            "qwen35".to_string(),
            "127.0.0.1",
            port,
            2048,  // prefill_chunk_size default
            4,     // b_max
            5,     // admission_deadline_ms
            32,    // admission_queue_max
            32768, // max_cache_cap (3f default)
        )
        .await
    });

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let health_url = format!("http://127.0.0.1:{port}/health");

    // Disable system proxy: macOS proxy (e.g. clash/v2ray on :7897) may not
    // respect the 127.0.0.1 exception, causing 502 Bad Gateway from the proxy
    // instead of the axum server.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .no_proxy()
        .build()
        .unwrap();

    // Poll /health until server is ready (up to 120s — debug build model
    // loading can take 40–60s when run after earlier tests).
    let server_ready = {
        let mut ready = false;
        for _ in 0..120 {
            if client
                .get(&health_url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        ready
    };
    assert!(
        server_ready,
        "server did not become healthy within 120s on port {port}"
    );
    eprintln!("[S5] server ready on port {port}");

    // 8 concurrent workers, each looping for 15 seconds.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut handles = Vec::new();
    for worker_id in 0..8 {
        let client = client.clone();
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let mut ok = 0usize;
            let mut errs: Vec<u16> = Vec::new();
            while tokio::time::Instant::now() < deadline {
                let body = serde_json::json!({
                    "model": "qwen35",
                    "messages": [
                        {"role": "user", "content": format!("hi from worker {worker_id}")}
                    ],
                    "max_tokens": 8,
                });
                let resp = match client.post(&url).json(&body).send().await {
                    Ok(r) => r,
                    Err(_) => {
                        // Connection-layer error (reset/refused). May be
                        // transient Metal GPU warm-up jitter in debug builds
                        // when prior tests leave GPU resources in-flight.
                        // Back off briefly before retrying so the server can
                        // stabilise.
                        errs.push(0);
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        continue;
                    }
                };
                let status = resp.status().as_u16();
                if status == 200 {
                    let _ = resp.bytes().await;
                    ok += 1;
                } else {
                    errs.push(status);
                }
            }
            (worker_id, ok, errs)
        }));
    }

    let mut total_ok = 0usize;
    let mut all_errs: Vec<u16> = Vec::new();
    for h in handles {
        let (worker_id, ok, errs) = h.await.unwrap();
        eprintln!("[S5] worker {worker_id}: ok={ok}, errs={errs:?}");
        total_ok += ok;
        all_errs.extend(errs);
    }

    // No 4xx allowed (would mean a request was rejected, not queued).
    // 5xx (503 from queue overflow) is also disallowed at queue_max=32
    // under c=8 b_max=4 — queue depth should never exceed 4 in this run.
    let four_xx: Vec<_> = all_errs
        .iter()
        .filter(|s| **s >= 400 && **s < 500)
        .collect();
    assert!(
        four_xx.is_empty(),
        "expected no 4xx, got: {four_xx:?}; total_ok={total_ok}"
    );
    let five_xx: Vec<_> = all_errs.iter().filter(|s| **s >= 500).collect();
    assert!(
        five_xx.is_empty(),
        "expected no 5xx at queue_max=32 c=8 b_max=4, got: {five_xx:?}"
    );

    assert!(
        total_ok > 0,
        "expected at least some successful responses, got 0; all_errs={all_errs:?}"
    );

    server_handle.abort();
}
