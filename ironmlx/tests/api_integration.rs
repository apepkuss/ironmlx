//! API integration tests — require Qwen3-0.6B-4bit model and a running server.
//!
//! These tests start the ironmlx server as a subprocess on port 18999,
//! run all tests against it, then shut it down.
//!
//! Run with: cargo test -p ironmlx -- --ignored --test-threads=1

use serde_json::Value;
use std::path::Path;
use std::process::{Child, Command, Stdio};

const QWEN3_MODEL_DIR: &str = "/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8";
const TEST_PORT: u16 = 18999;

struct TestServer {
    child: Child,
    base_url: String,
}

impl TestServer {
    async fn start() -> Option<Self> {
        if !Path::new(QWEN3_MODEL_DIR).exists() {
            return None;
        }

        let binary = env!("CARGO_BIN_EXE_ironmlx");
        let child = Command::new(binary)
            .args(["--model", QWEN3_MODEL_DIR, "--port", &TEST_PORT.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;

        let base_url = format!("http://127.0.0.1:{}", TEST_PORT);
        let client = reqwest::Client::new();

        // Wait for server to be ready (up to 30s)
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if let Ok(resp) = client.get(format!("{}/health", base_url)).send().await {
                if resp.status() == 200 {
                    return Some(Self { child, base_url });
                }
            }
        }

        None
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
#[ignore]
async fn test_api_endpoints() {
    let server = match TestServer::start().await {
        Some(s) => s,
        None => {
            eprintln!("Skipping API tests: model not available");
            return;
        }
    };

    let client = reqwest::Client::new();

    // 1. Health check
    {
        let resp = client
            .get(format!("{}/health", server.base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        eprintln!("  ✓ /health");
    }

    // 2. List models
    {
        let resp = client
            .get(format!("{}/v1/models", server.base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["object"], "list");
        let models = body["data"].as_array().unwrap();
        assert!(!models.is_empty());
        assert!(models[0]["id"].as_str().unwrap().contains("Qwen3"));
        eprintln!("  ✓ /v1/models");
    }

    // 3. Chat completions (sync)
    {
        let resp = client
            .post(format!("{}/v1/chat/completions", server.base_url))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 20,
                "temperature": 0.0,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert!(!body["choices"].as_array().unwrap().is_empty());
        assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
        eprintln!("  ✓ /v1/chat/completions (sync)");
    }

    // 4. Chat completions (SSE streaming)
    {
        let resp = client
            .post(format!("{}/v1/chat/completions", server.base_url))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 30,
                "temperature": 0.0,
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let text = resp.text().await.unwrap();
        assert!(text.contains("data: "), "should have SSE data lines");
        assert!(text.contains("[DONE]"), "should end with [DONE]");

        let mut found_content = false;
        for line in text.lines() {
            if let Some(json_str) = line.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    continue;
                }
                let chunk: Value = serde_json::from_str(json_str).unwrap();
                assert_eq!(chunk["object"], "chat.completion.chunk");
                if chunk["choices"][0]["delta"]["content"].is_string()
                    || chunk["choices"][0]["delta"]["reasoning_content"].is_string()
                {
                    found_content = true;
                }
            }
        }
        assert!(found_content, "should have content in at least one chunk");
        eprintln!("  ✓ /v1/chat/completions (SSE stream)");
    }

    // 5. Thinking mode — reasoning_content field
    {
        let resp = client
            .post(format!("{}/v1/chat/completions", server.base_url))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
                "temperature": 0.0,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();

        let message = &body["choices"][0]["message"];
        // Qwen3 should produce reasoning_content
        assert!(
            message["reasoning_content"].is_string(),
            "Qwen3 should produce reasoning_content"
        );
        eprintln!("  ✓ /v1/chat/completions (thinking mode)");
    }

    eprintln!("\n  All API tests passed!");
    // TestServer drops here, killing the subprocess
}
