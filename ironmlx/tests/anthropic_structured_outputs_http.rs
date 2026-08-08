//! Real-model HTTP acceptance for Anthropic Messages Structured Outputs.
//!
//! Run with:
//! ```text
//! LLAMA32_MODEL=/path/to/Llama-3.2-1B-Instruct-4bit \
//!   cargo test --release -p ironmlx --test anthropic_structured_outputs_http \
//!   -- --ignored --test-threads=1
//! ```

use std::path::Path;
use std::time::Duration;

use ironmlx::core::cache::ActiveKvOffloadConfig;
use ironmlx::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use ironmlx::core::server;
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::LlamaModel;

fn scheduler_profile() -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "llama-3.2-1b".to_owned(),
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

async fn boot_server(port: u16, model_dir: &Path) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let loader = Loader::open(model_dir).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = LlamaModel::from_loader(&loader).expect("LlamaModel::from_loader");
    tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            "llama-3.2-1b".to_owned(),
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

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer", "enum": [1, 2, 3]}
        },
        "required": ["city", "days"],
        "additionalProperties": false
    })
}

fn output_config() -> serde_json::Value {
    serde_json::json!({
        "format": {"type": "json_schema", "schema": output_schema()}
    })
}

fn base_request() -> serde_json::Value {
    serde_json::json!({
        "model": "llama-3.2-1b",
        "messages": [{
            "role": "user",
            "content": "Return city Tokyo and days 2. Output only the requested JSON."
        }],
        "temperature": 0,
        "max_tokens": 64,
        "output_config": output_config()
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

fn assert_structured_message(body: &serde_json::Value) {
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["content"][0]["type"], "text");
    let content: serde_json::Value = serde_json::from_str(
        body["content"][0]["text"]
            .as_str()
            .expect("structured text block"),
    )
    .expect("structured JSON");
    assert_eq!(content, serde_json::json!({"city": "Tokyo", "days": 2}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires LLAMA32_MODEL pointing to a real Llama 3.1/3.2 Instruct checkpoint"]
async fn anthropic_structured_outputs_real_http_acceptance() {
    let model_dir = std::env::var("LLAMA32_MODEL").expect("LLAMA32_MODEL must be set");
    let port = alloc_port().await;
    let server = boot_server(port, Path::new(&model_dir)).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .no_proxy()
        .build()
        .expect("HTTP client");
    wait_until_healthy(&client, port).await;
    let endpoint = format!("http://127.0.0.1:{port}/v1/messages");

    let response = client
        .post(&endpoint)
        .json(&base_request())
        .send()
        .await
        .expect("structured sync request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("structured sync body");
    assert_structured_message(&body);

    let mut truncated = base_request();
    truncated["max_tokens"] = serde_json::json!(1);
    let response = client
        .post(&endpoint)
        .json(&truncated)
        .send()
        .await
        .expect("truncated request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("truncated body");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], 1);

    let mut streaming = base_request();
    streaming["stream"] = serde_json::json!(true);
    let response = client
        .post(&endpoint)
        .json(&streaming)
        .send()
        .await
        .expect("structured SSE request");
    assert_eq!(response.status(), 200);
    let sse = response.text().await.expect("structured SSE body");
    assert!(sse.contains("event: message_start"));
    assert!(sse.contains("\"type\":\"text_delta\""));
    assert!(sse.contains("\"stop_reason\":\"end_turn\""));
    assert!(sse
        .trim_end()
        .ends_with("data: {\"type\":\"message_stop\"}"));

    let mut truncated_stream = base_request();
    truncated_stream["max_tokens"] = serde_json::json!(1);
    truncated_stream["stream"] = serde_json::json!(true);
    let response = client
        .post(&endpoint)
        .json(&truncated_stream)
        .send()
        .await
        .expect("truncated structured SSE request");
    assert_eq!(response.status(), 200);
    let sse = response
        .text()
        .await
        .expect("truncated structured SSE body");
    assert!(sse.contains("\"stop_reason\":\"max_tokens\""));
    assert!(sse
        .trim_end()
        .ends_with("data: {\"type\":\"message_stop\"}"));

    let mut with_tools = base_request();
    with_tools["messages"][0]["content"] =
        serde_json::json!("Call get_weather for Tokyo, or return the requested JSON directly.");
    with_tools["tools"] = serde_json::json!([{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        },
        "strict": true
    }]);
    with_tools["tool_choice"] = serde_json::json!({"type": "auto"});
    let response = client
        .post(&endpoint)
        .json(&with_tools)
        .send()
        .await
        .expect("tools plus structured output request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("combined response");
    match body["stop_reason"].as_str() {
        Some("tool_use") => {
            assert_eq!(
                body["content"].as_array().unwrap().last().unwrap()["type"],
                "tool_use"
            );
        }
        Some("end_turn") => assert_structured_message(&body),
        other => panic!("unexpected combined stop reason: {other:?}"),
    }

    with_tools["messages"][0]["content"] =
        serde_json::json!("Return city Tokyo and days 2. Output only JSON.");
    with_tools["tool_choice"] = serde_json::json!({"type": "none"});
    let response = client
        .post(&endpoint)
        .json(&with_tools)
        .send()
        .await
        .expect("disabled tools plus structured output request");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("tool-none response");
    assert_structured_message(&body);

    for invalid in [
        serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": [],
            "max_tokens": 8,
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        }),
        serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": [],
            "max_tokens": 8,
            "output_format": {"type": "json_schema", "schema": output_schema()}
        }),
        serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": [{"role": "assistant", "content": "{"}],
            "max_tokens": 8,
            "output_config": output_config()
        }),
        serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "max_tokens": 8,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "oneOf": [
                            {"type": "object"},
                            {"type": "array"}
                        ]
                    }
                }
            }
        }),
    ] {
        let response = client
            .post(&endpoint)
            .json(&invalid)
            .send()
            .await
            .expect("invalid structured request");
        assert_eq!(response.status(), 400);
        let body: serde_json::Value = response.json().await.expect("Anthropic error body");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    server.abort();
}
