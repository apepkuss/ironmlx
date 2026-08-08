//! Real-model HTTP acceptance for Anthropic extended/adaptive thinking.
//!
//! Run with:
//! ```text
//! QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
//!   cargo test --release -p ironmlx --test anthropic_thinking_http \
//!   -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Duration;

use ironmlx::core::cache::ActiveKvOffloadConfig;
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
        model_name: "qwen3.5-4b".to_owned(),
        hardware_label: "test-host".to_owned(),
        runtime_context:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(32768),
        config: SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                1811606400000,
            ),
    }
}

async fn alloc_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let port = listener.local_addr().expect("listener address").port();
    drop(listener);
    port
}

async fn boot_server(port: u16) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let model_dir = PathBuf::from(
        std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL must point to a real checkpoint"),
    );
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            "qwen3.5-4b".to_owned(),
            server::security::ServerNetworkConfig::local("127.0.0.1", port)?,
            2048,
            2,
            5,
            32,
            32768,
            256,
            None,
            None,
            None,
            ActiveKvOffloadConfig::disabled(),
            scheduler_profile(),
            false,
            None,
            Default::default(),
            true,
        )
        .await
    })
}

async fn wait_until_healthy(client: &reqwest::Client, port: u16) {
    for _ in 0..100 {
        if client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not become healthy");
}

fn adaptive_request(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "qwen3.5-4b",
        "messages": [{
            "role": "user",
            "content": "Compute 17 * 19 and explain briefly."
        }],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
        "temperature": 0,
        "max_tokens": 64,
        "stream": stream
    })
}

fn thinking_structured_request(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "qwen3.5-4b",
        "messages": [{
            "role": "user",
            "content": "Compute 17 * 19. Return the result in the requested JSON schema."
        }],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"product": {"type": "integer"}},
                    "required": ["product"],
                    "additionalProperties": false
                }
            }
        },
        "temperature": 0,
        "max_tokens": 128,
        "stream": stream
    })
}

fn structured_text(body: &serde_json::Value) -> serde_json::Value {
    let block = body["content"]
        .as_array()
        .expect("content blocks")
        .iter()
        .find(|block| block["type"] == "text")
        .expect("structured text block");
    serde_json::from_str(block["text"].as_str().expect("structured text")).expect("structured JSON")
}

fn structured_sse_text(sse: &str) -> serde_json::Value {
    let mut text = String::new();
    for frame in sse.split("\n\n") {
        let Some(data) = frame.lines().find_map(|line| line.strip_prefix("data: ")) else {
            continue;
        };
        let value: serde_json::Value = serde_json::from_str(data).expect("SSE JSON");
        if value["delta"]["type"] == "text_delta" {
            text.push_str(value["delta"]["text"].as_str().expect("text delta"));
        }
    }
    serde_json::from_str(&text).expect("structured SSE JSON")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL pointing to a real Qwen3.5 checkpoint"]
async fn anthropic_thinking_real_http_acceptance() {
    let port = alloc_port().await;
    let server = boot_server(port).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .no_proxy()
        .build()
        .expect("HTTP client");
    wait_until_healthy(&client, port).await;
    let endpoint = format!("http://127.0.0.1:{port}/v1/messages");

    let response = client
        .post(&endpoint)
        .json(&adaptive_request(false))
        .send()
        .await
        .expect("adaptive sync request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("adaptive sync body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert!(!body["content"][0]["thinking"]
        .as_str()
        .expect("thinking text")
        .is_empty());
    assert!(body["content"][0]["signature"]
        .as_str()
        .expect("thinking signature")
        .starts_with("ironmlx-v1:"));
    assert!(body["usage"]["output_tokens_details"]["thinking_tokens"]
        .as_u64()
        .is_some_and(|tokens| tokens > 0));

    let mut history = body["content"].clone();
    let response = client
        .post(&endpoint)
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [
                {"role": "user", "content": "Compute 17 * 19."},
                {"role": "assistant", "content": history.take()},
                {"role": "user", "content": "Now answer only with the number."}
            ],
            "thinking": {"type": "adaptive"},
            "max_tokens": 32,
            "temperature": 0
        }))
        .send()
        .await
        .expect("thinking history request");
    assert_eq!(response.status(), 200);

    let response = client
        .post(&endpoint)
        .json(&adaptive_request(true))
        .send()
        .await
        .expect("adaptive SSE request");
    assert_eq!(response.status(), 200);
    let sse = response.text().await.expect("adaptive SSE body");
    let thinking_delta = sse.find("\"type\":\"thinking_delta\"").unwrap();
    let signature_delta = sse.find("\"type\":\"signature_delta\"").unwrap();
    assert!(thinking_delta < signature_delta);
    assert!(sse.contains("\"output_tokens_details\":{\"thinking_tokens\":"));
    assert!(sse
        .trim_end()
        .ends_with("data: {\"type\":\"message_stop\"}"));

    let response = client
        .post(&endpoint)
        .json(&thinking_structured_request(false))
        .send()
        .await
        .expect("thinking plus structured output request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response
        .json()
        .await
        .expect("thinking plus structured output body");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(structured_text(&body), serde_json::json!({"product": 323}));

    let response = client
        .post(&endpoint)
        .json(&thinking_structured_request(true))
        .send()
        .await
        .expect("thinking plus structured output SSE request");
    assert_eq!(response.status(), 200);
    let sse = response
        .text()
        .await
        .expect("thinking plus structured output SSE body");
    let thinking_delta = sse.find("\"type\":\"thinking_delta\"").unwrap();
    let text_delta = sse.find("\"type\":\"text_delta\"").unwrap();
    assert!(thinking_delta < text_delta);
    assert_eq!(
        structured_sse_text(&sse),
        serde_json::json!({"product": 323})
    );

    let response = client
        .post(&endpoint)
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{
                "role": "user",
                "content": "Use the multiply tool for 17 times 19."
            }],
            "thinking": {"type": "adaptive"},
            "tools": [{
                "name": "multiply",
                "description": "Multiply two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"],
                    "additionalProperties": false
                }
            }],
            "tool_choice": {"type": "any"},
            "max_tokens": 128,
            "temperature": 0
        }))
        .send()
        .await
        .expect("thinking with tools request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("thinking with tools body");
    let blocks = body["content"].as_array().expect("content blocks");
    assert_eq!(blocks[0]["type"], "thinking");
    assert!(blocks.iter().any(|block| block["type"] == "tool_use"));

    let response = client
        .post(&endpoint)
        .json(&serde_json::json!({
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "thinking": {"type": "disabled"},
            "max_tokens": 16,
            "temperature": 0
        }))
        .send()
        .await
        .expect("disabled thinking request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("disabled thinking body");
    assert!(body["content"]
        .as_array()
        .expect("content blocks")
        .iter()
        .all(|block| block["type"] != "thinking"));

    server.abort();
}
