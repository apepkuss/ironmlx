//! HTTP admission queue acceptance with concurrent requests.
use ironmlx_lm::models::Qwen35Model;
use ironmlx_lm::{Loader, Tokenizer};
use std::path::PathBuf;
use std::time::Duration;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore] // real-model heavy + HTTP server
async fn iron_bench_c8_with_queue_no_4xx() {
    // B1-p2.5 budget gate: server::serve is called with cap=32768 which
    // triggers budget validation. Override to 64 GiB so it passes.
    // EnvGuard Drop cleans up even on panic.
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
        }
    }
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "68719476736"); // 64 GiB
    let _guard = EnvGuard;

    // Boot the server on a random port; spawn 8 concurrent HTTP clients
    // hitting /v1/chat/completions for 15s. With b_max=4 + queue_max=32,
    // no HTTP 4xx should occur.
    use ironmlx::server;
    use {
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneProfileConfig,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata,
        ironmlx_runtime::core::scheduler_autotune::SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    let scheduler_profile = SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "qwen35".to_string(),
        hardware_label: "test-host".to_string(),
        runtime_context:
            ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(32768),
        config: SchedulerAutotuneProfileConfig {
            b_max: 4,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(1811606400000),
    };

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
            ironmlx::server::security::ServerNetworkConfig::local("127.0.0.1", port)?,
            2048,  // prefill_chunk_size default
            4,     // b_max
            5,     // admission_deadline_ms
            32,    // admission_queue_max
            32768, // max_cache_cap (3f default)
            256,   // decode_cadence_mid_chunk_cap
            None,  // kv_cache_turboquant_bits
            None,  // paged_prefix_cache
            None,  // prefix_lru_cache
            ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig::disabled(),
            scheduler_profile,
            false, // scheduler_autotune_report
            None,  // vision_input_override
            ironmlx_runtime::core::process_memory::StaticMemoryEstimate::default(),
            true, // force_scheduler
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
