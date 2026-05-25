//! Verifies `--inter-run-cooldown-secs N` adds inter-run sleep only between
//! measured sequential runs, and is rejected for concurrent mode.
//!
//! Mirrors `concurrent_smoke.rs`: launches an in-process OpenAI-compatible
//! SSE mock server and invokes the built iron-bench binary.

use std::{
    convert::Infallible,
    process::Command,
    time::{Duration, Instant},
};

use axum::{response::sse::Event, routing::post, Router};
use tokio::net::TcpListener;
use tokio_stream::wrappers::ReceiverStream;

async fn mock_sse_handler() -> axum::response::sse::Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let body = serde_json::json!({
            "id": "mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": { "content": "tok" },
                "finish_reason": null,
            }],
        });
        let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
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
                "completion_tokens": 1,
                "cached_tokens": 0,
            },
        });
        let _ = tx
            .send(Ok(Event::default().data(usage_body.to_string())))
            .await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    axum::response::sse::Sse::new(ReceiverStream::new(rx))
}

fn tokenizer_fixture_dir() -> Option<std::path::PathBuf> {
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    if fixture_dir.join("tokenizer.json").exists() {
        Some(fixture_dir)
    } else {
        eprintln!(
            "[cooldown] tokenizer fixture missing at {}; skipping timing assertion",
            fixture_dir.join("tokenizer.json").display()
        );
        None
    }
}

fn run_iron_bench(bench_bin: &str, url: &str, model_dir: &str, cooldown_secs: &str) -> Duration {
    let start = Instant::now();
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
            "1",
            "--runs",
            "2",
            "--warmup",
            "0",
            "--inter-run-cooldown-secs",
            cooldown_secs,
            "--format",
            "csv",
        ])
        .output()
        .expect("spawn iron-bench");
    let elapsed = start.elapsed();
    assert!(
        output.status.success(),
        "iron-bench exited non-zero:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    elapsed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cooldown_inserts_sleep_between_measured_runs_only() {
    let Some(fixture_dir) = tokenizer_fixture_dir() else {
        return;
    };
    let app = Router::new().route("/v1/chat/completions", post(mock_sse_handler));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });

    let bench_bin = env!("CARGO_BIN_EXE_iron-bench");
    let url = format!("http://{addr}");
    let model_dir = fixture_dir.to_str().expect("utf-8 fixture_dir");

    let no_cooldown = run_iron_bench(bench_bin, &url, model_dir, "0");
    let one_second = run_iron_bench(bench_bin, &url, model_dir, "1");
    let delta = one_second.saturating_sub(no_cooldown);
    assert!(
        delta >= Duration::from_millis(800),
        "expected cooldown=1 to add roughly one inter-run sleep; \
         no_cooldown={no_cooldown:?} cooldown_1s={one_second:?} delta={delta:?}",
    );
}

#[test]
fn cooldown_rejects_concurrent_mode_when_nonzero() {
    let bin = env!("CARGO_BIN_EXE_iron-bench");
    let out = Command::new(bin)
        .args([
            "--target",
            "bogus=http://127.0.0.1:1",
            "--model",
            "x",
            "--model-dir",
            "/tmp/nonexistent",
            "--prompt-len",
            "16",
            "--max-tokens",
            "1",
            "--concurrent",
            "2",
            "--duration",
            "1",
            "--warmup-duration",
            "0",
            "--inter-run-cooldown-secs",
            "1",
            "--format",
            "csv",
        ])
        .output()
        .expect("iron-bench spawn");
    assert_ne!(out.status.code(), Some(0), "expected non-zero exit");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("inter-run-cooldown-secs")
            && (combined.contains("concurrent") || combined.contains("v2")),
        "expected validation error mentioning inter-run-cooldown-secs + concurrent/v2; got: {combined}"
    );
}
