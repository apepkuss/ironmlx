//! Real-checkpoint Responses reasoning acceptance for every public effort.
//!
//! Run with:
//! ```text
//! QWEN38_MODEL=/path/to/Qwen3.8-27B-8bit \
//!   cargo test --release -p ironmlx --test qwen38_reasoning_http \
//!   -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine;
use ironmlx::core::cache::ActiveKvOffloadConfig;
use ironmlx::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use ironmlx::core::server;
use ironmlx::core::{Loader, QuantMode, Tokenizer};
use ironmlx::models::Qwen35Model;

const MAX_CACHE_CAP: usize = 4096;
const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn scheduler_profile(model_id: &str) -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: model_id.to_owned(),
        hardware_label: "qwen38-reasoning-http-test".to_owned(),
        runtime_context:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(
                MAX_CACHE_CAP,
            ),
        config: SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: MAX_CACHE_CAP,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                1818374400000,
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

async fn boot_server(port: u16) -> (tokio::task::JoinHandle<anyhow::Result<()>>, String) {
    let model_dir = PathBuf::from(
        std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL must point to a real checkpoint"),
    );
    let loader = Loader::open_multimodal(&model_dir).expect("Loader::open_multimodal");
    let quantization = loader.quant_meta().expect("Qwen3.8 quantization metadata");
    assert_eq!(quantization.mode, QuantMode::Affine);
    assert_eq!(quantization.group_size, 64);
    assert!(
        matches!(quantization.bits, 4 | 8),
        "QWEN38_MODEL must point to the affine 4-bit or 8-bit checkpoint"
    );
    let model_id = format!("mlx-community/Qwen3.8-27B-{}bit", quantization.bits);
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let served_model_id = model_id.clone();
    let profile = scheduler_profile(&model_id);
    let server = tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            served_model_id,
            server::security::ServerNetworkConfig::local("127.0.0.1", port)?,
            2048,
            1,
            5,
            32,
            MAX_CACHE_CAP,
            256,
            None,
            None,
            None,
            ActiveKvOffloadConfig::disabled(),
            profile,
            false,
            None,
            Default::default(),
            false,
        )
        .await
    });
    (server, model_id)
}

async fn wait_until_healthy(client: &reqwest::Client, port: u16) {
    for _ in 0..100 {
        if client
            .get(format!("http://127.0.0.1:{port}/healthz"))
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

fn request(model_id: &str, effort: Option<&str>, stream: bool) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model_id,
        "input": "用一句话回答：1+1等于几？推理尽量简短。",
        "store": false,
        "max_output_tokens": 256,
        "temperature": 0,
        "stream": stream
    });
    if let Some(effort) = effort {
        body["reasoning"] = serde_json::json!({"effort": effort, "summary": "none"});
    }
    body
}

fn output_text(response: &serde_json::Value) -> Option<&str> {
    response["output"]
        .as_array()?
        .iter()
        .find(|item| item["type"] == "message")?["content"]
        .as_array()?
        .iter()
        .find(|part| part["type"] == "output_text")?["text"]
        .as_str()
}

fn reasoning_text(response: &serde_json::Value) -> Option<&str> {
    response["output"]
        .as_array()?
        .iter()
        .find(|item| item["type"] == "reasoning")?["content"]
        .as_array()?
        .iter()
        .find(|part| part["type"] == "reasoning_text")?["text"]
        .as_str()
}

fn assert_completed_response(
    response: &serde_json::Value,
    requested_effort: Option<&str>,
    expect_reasoning: bool,
) {
    assert_eq!(response["status"], "completed", "response: {response:#}");
    assert!(response["incomplete_details"].is_null());
    assert_eq!(
        response["reasoning"]["effort"],
        requested_effort.unwrap_or("none")
    );
    assert!(
        output_text(response).is_some_and(|text| !text.trim().is_empty()),
        "missing final output text: {response:#}"
    );

    let reasoning_tokens = response["usage"]["output_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .expect("reasoning token usage");
    if expect_reasoning {
        assert!(
            reasoning_text(response).is_some_and(|text| !text.trim().is_empty()),
            "missing reasoning item: {response:#}"
        );
        assert!(
            reasoning_tokens > 0,
            "missing reasoning usage: {response:#}"
        );
    } else {
        assert!(
            reasoning_text(response).is_none(),
            "unexpected reasoning item"
        );
        assert_eq!(reasoning_tokens, 0);
    }
}

async fn send_sync(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    effort: Option<&str>,
) -> serde_json::Value {
    let response = client
        .post(endpoint)
        .json(&request(model_id, effort, false))
        .send()
        .await
        .unwrap_or_else(|error| panic!("send sync effort {effort:?}: {error:#}"));
    let status = response.status();
    let text = response.text().await.expect("sync response body");
    assert_eq!(status, 200, "sync effort {effort:?}: {text}");
    serde_json::from_str(&text).expect("sync response JSON")
}

fn sse_events(body: &str) -> Vec<serde_json::Value> {
    body.split("\n\n")
        .filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE event JSON"))
        .collect()
}

async fn send_stream(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    effort: Option<&str>,
) -> serde_json::Value {
    let response = client
        .post(endpoint)
        .json(&request(model_id, effort, true))
        .send()
        .await
        .unwrap_or_else(|error| panic!("send SSE effort {effort:?}: {error:#}"));
    let status = response.status();
    let text = response.text().await.expect("SSE response body");
    assert_eq!(status, 200, "SSE effort {effort:?}: {text}");
    let events = sse_events(&text);
    let event_types: Vec<&str> = events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect();
    assert!(event_types.contains(&"response.created"));
    assert!(event_types.contains(&"response.output_text.delta"));
    assert!(event_types.contains(&"response.completed"));
    if effort.is_some_and(|effort| effort != "none") {
        assert!(event_types.contains(&"response.reasoning_text.delta"));
        assert!(event_types.contains(&"response.reasoning_text.done"));
    } else {
        assert!(!event_types.contains(&"response.reasoning_text.delta"));
        assert!(!event_types.contains(&"response.reasoning_text.done"));
    }
    events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("response.completed event")["response"]
        .clone()
}

async fn post_json(
    client: &reqwest::Client,
    endpoint: &str,
    body: serde_json::Value,
    context: &str,
) -> serde_json::Value {
    let response = client
        .post(endpoint)
        .json(&body)
        .send()
        .await
        .unwrap_or_else(|error| panic!("{context}: {error:#}"));
    let status = response.status();
    let text = response
        .text()
        .await
        .unwrap_or_else(|error| panic!("{context} response body: {error:#}"));
    assert_eq!(status, 200, "{context}: {text}");
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{context} JSON: {error:#}\n{text}"))
}

async fn assert_structured_output(client: &reqwest::Client, endpoint: &str, model_id: &str) {
    let response = post_json(
        client,
        endpoint,
        serde_json::json!({
            "model": model_id,
            "input": "Return a JSON object whose answer is the integer result of 1+1.",
            "reasoning": {"effort": "none", "summary": "none"},
            "text": {"format": {
                "type": "json_schema",
                "name": "arithmetic_answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "strict": true
            }},
            "store": false,
            "max_output_tokens": 64,
            "temperature": 0,
            "stream": false
        }),
        "structured output",
    )
    .await;
    let text = output_text(&response).expect("structured output text");
    let value: serde_json::Value = serde_json::from_str(text).expect("structured output JSON");
    assert!(value["answer"].is_i64(), "structured output: {value}");
}

async fn assert_tool_round_trip(client: &reqwest::Client, endpoint: &str, model_id: &str) {
    let tools = serde_json::json!([{
        "type": "function",
        "name": "weather",
        "description": "Return the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        },
        "strict": true
    }]);
    let first = post_json(
        client,
        endpoint,
        serde_json::json!({
            "model": model_id,
            "input": "Call the weather tool for Tokyo.",
            "tools": tools.clone(),
            "tool_choice": {"type": "function", "name": "weather"},
            "parallel_tool_calls": false,
            "reasoning": {"effort": "none", "summary": "none"},
            "store": false,
            "max_output_tokens": 128,
            "temperature": 0,
            "stream": false
        }),
        "tool call",
    )
    .await;
    let call = first["output"]
        .as_array()
        .expect("tool output items")
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function_call output");
    assert_eq!(call["name"], "weather");
    let arguments: serde_json::Value = serde_json::from_str(
        call["arguments"]
            .as_str()
            .expect("function_call arguments string"),
    )
    .expect("function_call arguments JSON");
    assert!(arguments["city"].is_string(), "tool arguments: {arguments}");
    let call_id = call["call_id"].as_str().expect("function_call call_id");

    let second = post_json(
        client,
        endpoint,
        serde_json::json!({
            "model": model_id,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Call the weather tool for Tokyo."}]},
                call.clone(),
                {"type": "function_call_output", "call_id": call_id, "output": "22 C and sunny"}
            ],
            "tools": tools,
            "tool_choice": "none",
            "parallel_tool_calls": false,
            "reasoning": {"effort": "none", "summary": "none"},
            "store": false,
            "max_output_tokens": 128,
            "temperature": 0,
            "stream": false
        }),
        "tool result round trip",
    )
    .await;
    assert!(
        output_text(&second).is_some_and(|text| !text.trim().is_empty()),
        "missing tool-result answer: {second:#}"
    );
}

async fn assert_image_input(client: &reqwest::Client, endpoint: &str, model_id: &str) {
    let image_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_vl/coco_sample.jpg");
    let image = std::fs::read(&image_path).expect("read COCO image fixture");
    let image_url = format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(image)
    );
    let response = post_json(
        client,
        endpoint,
        serde_json::json!({
            "model": model_id,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                    {"type": "input_text", "text": "Describe this image in one short sentence."}
                ]
            }],
            "reasoning": {"effort": "none", "summary": "none"},
            "store": false,
            "max_output_tokens": 64,
            "temperature": 0,
            "stream": false
        }),
        "image input",
    )
    .await;
    assert!(
        output_text(&response).is_some_and(|text| !text.trim().is_empty()),
        "missing image answer: {response:#}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN38_MODEL pointing to mlx-community/Qwen3.8-27B-4bit or Qwen3.8-27B-8bit"]
async fn qwen38_all_reasoning_efforts_real_http_acceptance() {
    let port = alloc_port().await;
    let (server, model_id) = boot_server(port).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .no_proxy()
        .build()
        .expect("HTTP client");
    wait_until_healthy(&client, port).await;
    let endpoint = format!("http://127.0.0.1:{port}/v1/responses");

    let default = send_sync(&client, &endpoint, &model_id, None).await;
    assert_completed_response(&default, None, false);

    let mut medium = None;
    for effort in EFFORTS {
        let response = send_sync(&client, &endpoint, &model_id, Some(effort)).await;
        assert_completed_response(&response, Some(effort), effort != "none");
        if effort == "medium" {
            medium = Some(response);
        }
    }

    let default = send_stream(&client, &endpoint, &model_id, None).await;
    assert_completed_response(&default, None, false);
    for effort in EFFORTS {
        let response = send_stream(&client, &endpoint, &model_id, Some(effort)).await;
        assert_completed_response(&response, Some(effort), effort != "none");
    }

    let mut history = medium.expect("medium response")["output"]
        .as_array()
        .expect("medium output items")
        .clone();
    history.insert(
        0,
        serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "1+1等于几？"}]
        }),
    );
    history.push(serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "现在只回答这个结果加1。"}]
    }));
    let response = client
        .post(&endpoint)
        .json(&serde_json::json!({
            "model": &model_id,
            "input": history,
            "reasoning": {"effort": "medium", "summary": "none"},
            "store": false,
            "max_output_tokens": 256,
            "temperature": 0,
            "stream": false
        }))
        .send()
        .await
        .expect("send reasoning history request");
    let status = response.status();
    let text = response.text().await.expect("reasoning history body");
    assert_eq!(status, 200, "reasoning history: {text}");
    let response: serde_json::Value =
        serde_json::from_str(&text).expect("reasoning history response JSON");
    assert_completed_response(&response, Some("medium"), true);

    assert_structured_output(&client, &endpoint, &model_id).await;
    assert_tool_round_trip(&client, &endpoint, &model_id).await;
    assert_image_input(&client, &endpoint, &model_id).await;

    server.abort();
    let _ = server.await;
}
