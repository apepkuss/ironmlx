//! P4 HTTP smoke test — boots the server on a random port and exercises
//! all four request paths (OpenAI/Anthropic × stream/non-stream).
//!
//! Requires `QWEN35_MODEL` env var pointing to a real Qwen3.5-4B-MLX-4bit dir.
//!
//! Run with:
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//!   QWEN35_MODEL=/path/to/checkpoint \
//!   cargo test --release --ignored -p ironmlx -- p4_http_smoke -- --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Duration;

use ironmlx::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use ironmlx::core::server;
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::Qwen35Model;

fn scheduler_profile() -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "qwen3.5-4b".to_string(),
        hardware_label: "test-host".to_string(),
        config: SchedulerAutotuneProfileConfig {
            b_max: 4,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
    }
}

async fn boot_server(port: u16) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let model_dir = PathBuf::from(
        std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL must be set for p4_http_smoke"),
    );
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let model_id = "qwen3.5-4b".to_string();

    // serve() signature gained 5 args across 3b-2 / 3d / 3f phases; p4 smoke uses defaults.
    tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            model_id,
            "127.0.0.1",
            port,
            /* prefill_chunk_size */ 2048,
            /* b_max */ 4,
            /* admission_deadline_ms */ 5,
            /* admission_queue_max */ 32,
            /* max_cache_cap */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            scheduler_profile(),
            /* scheduler_autotune_report */ false,
            /* p5h_measurement_eval_probes */ false,
            /* vision_input_override */ None,
        )
        .await
    })
}

async fn alloc_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL pointing to real checkpoint"]
async fn p4_http_smoke() {
    let port = alloc_port().await;
    let _server = boot_server(port).await;
    // Wait for server to bind.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        // Bypass any system proxy so we hit our local server directly.
        .no_proxy()
        .build()
        .unwrap();

    // Confirm server is up via health endpoint.
    let health = client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .expect("health check send");
    assert_eq!(health.status(), 200, "health check failed");

    // 1. OpenAI non-streaming
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 16,
            "stream": false
        }))
        .send()
        .await
        .expect("oai non-stream send");
    let status = resp.status();
    let body_text = resp.text().await.expect("oai non-stream body text");
    assert_eq!(status, 200, "oai non-stream status");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("oai non-stream json");
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty(), "oai non-stream content empty");

    // 2. OpenAI streaming
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "stream": true
        }))
        .send()
        .await
        .expect("oai stream send");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("oai stream body");
    assert!(body.contains("data: "), "oai SSE missing data: prefix");
    assert!(body.contains("[DONE]"), "oai SSE missing [DONE]");

    // 3. Anthropic non-streaming
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
            "stream": false
        }))
        .send()
        .await
        .expect("ant non-stream send");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("ant non-stream json");
    let text = body["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty(), "ant non-stream text empty");

    // 4. Anthropic streaming
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "stream": true
        }))
        .send()
        .await
        .expect("ant stream send");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("ant stream body");
    assert!(
        body.contains("event: message_start"),
        "ant SSE missing message_start"
    );
    assert!(
        body.contains("event: content_block_delta"),
        "ant SSE missing content_block_delta"
    );
    assert!(
        body.contains("event: message_stop"),
        "ant SSE missing message_stop"
    );

    // _server handle drops at end of test → tokio aborts the task.
}
