//! Integration smoke test for iron-bench v2 concurrent mode.
//!
//! Launches an in-process mock SSE server that emits 5 OpenAI-compatible
//! `data: {...}` chunks per request after 4 ms intervals, then `data: [DONE]`.
//! Invokes the iron-bench binary against the mock with `--concurrent 2
//! --duration 1 --warmup-duration 0` and verifies the JSON output contains the
//! expected concurrent-mode fields.
//!
//! Self-skips gracefully when `iron-bench/tests/fixtures/tokenizer.json` is
//! absent (CI / machines without the fixture staged).  See
//! `iron-bench/tests/fixtures/.gitignore` for how to stage the fixture locally.

use std::{convert::Infallible, process::Command, time::Duration};

use axum::{response::sse::Event, routing::post, Router};
use tokio::net::TcpListener;
use tokio_stream::wrappers::ReceiverStream;

// ---------------------------------------------------------------------------
// Mock SSE handler
// ---------------------------------------------------------------------------

async fn mock_sse_handler() -> axum::response::sse::Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(16);

    tokio::spawn(async move {
        // 5 content chunks at 4 ms intervals.
        for i in 0u32..5 {
            tokio::time::sleep(Duration::from_millis(4)).await;
            let body = serde_json::json!({
                "id": "mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "delta": { "content": format!("tok{i}") },
                    "finish_reason": null,
                }],
            });
            let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
        }

        // Final chunk: finish_reason + usage (stream_options.include_usage contract).
        let usage_body = serde_json::json!({
            "id": "mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 5,
                "cached_tokens": 0,
            },
        });
        let _ = tx
            .send(Ok(Event::default().data(usage_body.to_string())))
            .await;

        // Closing [DONE] sentinel.
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    axum::response::sse::Sse::new(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_smoke_against_mock_server() {
    // ------------------------------------------------------------------
    // 1. Check tokenizer fixture — self-skip if absent.
    // ------------------------------------------------------------------
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let tokenizer_path = fixture_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "[smoke] tokenizer fixture not found at {} — skipping.\n\
             Stage it with:\n  \
             SNAP=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)\n  \
             cp \"${{SNAP}}tokenizer.json\" iron-bench/tests/fixtures/tokenizer.json",
            tokenizer_path.display()
        );
        return;
    }

    // ------------------------------------------------------------------
    // 2. Bind mock SSE server on a random port.
    // ------------------------------------------------------------------
    let app = Router::new().route("/v1/chat/completions", post(mock_sse_handler));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });

    // ------------------------------------------------------------------
    // 3. Invoke iron-bench binary (cargo sets CARGO_BIN_EXE_iron-bench).
    // ------------------------------------------------------------------
    let bench_bin = env!("CARGO_BIN_EXE_iron-bench");
    let url = format!("http://{addr}");
    let model_dir = fixture_dir.to_str().expect("utf-8 fixture_dir");

    let output = Command::new(bench_bin)
        .args([
            "--target",
            &format!("mock={url}"),
            "--model-dir",
            model_dir,
            "--model",
            "mock",
            "--prompt-len",
            "16",
            "--max-tokens",
            "5",
            "--concurrent",
            "2",
            "--duration",
            "1",
            "--warmup-duration",
            "0",
            "--format",
            "json",
        ])
        .output()
        .expect("spawn iron-bench");

    assert!(
        output.status.success(),
        "iron-bench v2 exited non-zero:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // ------------------------------------------------------------------
    // 4. Parse and assert JSON shape.
    // ------------------------------------------------------------------
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not valid JSON: {e}\nstdout was: {stdout}\nstderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    });

    // Top-level mode fields.
    assert_eq!(json["mode"], "concurrent", "json={json}");
    assert_eq!(json["concurrent"], 2, "json={json}");

    // At least one cell with concurrent workers.
    let cells = json["cells"].as_array().expect("cells must be an array");
    assert!(!cells.is_empty(), "cells must not be empty; json={json}");

    let cell = &cells[0];
    assert_eq!(cell["concurrent"], 2, "cell={cell}");

    let n_requests = cell["n_requests"]
        .as_u64()
        .unwrap_or_else(|| panic!("n_requests must be a u64; cell={cell}"));
    assert!(
        n_requests > 0,
        "at least one request must have completed; cell={cell}"
    );

    // Percentile latency fields.
    assert!(
        cell["ttft_ms"]["p50"].is_number(),
        "ttft_ms.p50 must be present; cell={cell}"
    );
    assert!(
        cell["ttft_ms"]["p95"].is_number(),
        "ttft_ms.p95 must be present; cell={cell}"
    );
    assert!(
        cell["itl_ms"]["p99"].is_number(),
        "itl_ms.p99 must be present; cell={cell}"
    );
    assert!(
        cell["early_itl_ms"]["p95"].is_number(),
        "early_itl_ms.p95 must be present; cell={cell}"
    );
    assert_eq!(
        cell["early_itl_ms"]["first_n"], 8,
        "early_itl first_n must be exported; cell={cell}"
    );

    // Aggregate throughput.
    assert!(
        cell["aggregate"]["tokens_per_sec"].is_number(),
        "aggregate.tokens_per_sec must be present; cell={cell}"
    );

    // Per-worker breakdown.
    let per_worker_req = cell["per_worker"]["req_count"]
        .as_array()
        .unwrap_or_else(|| panic!("per_worker.req_count must be an array; cell={cell}"));
    assert_eq!(
        per_worker_req.len(),
        2,
        "should have exactly 2 workers; cell={cell}"
    );
    assert!(
        per_worker_req.iter().all(|v| v.as_u64().unwrap_or(0) > 0),
        "each worker should have completed at least one request; cell={cell}"
    );
}
