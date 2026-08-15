//! Real-checkpoint Responses reasoning acceptance for every public effort.
//!
//! Run with:
//! ```text
//! QWEN38_MODEL=/path/to/Qwen3.8-27B-4bit \
//!   cargo test --release -p ironmlx --test qwen38_reasoning_http \
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

const MODEL_ID: &str = "Qwen3.8-27B-4bit";
const MAX_CACHE_CAP: usize = 4096;
const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn scheduler_profile() -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: MODEL_ID.to_owned(),
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

async fn boot_server(port: u16) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let model_dir = PathBuf::from(
        std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL must point to a real checkpoint"),
    );
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            MODEL_ID.to_owned(),
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
            scheduler_profile(),
            false,
            None,
            Default::default(),
            false,
        )
        .await
    })
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

fn request(effort: Option<&str>, stream: bool) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": MODEL_ID,
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
    match requested_effort {
        Some(effort) => assert_eq!(response["reasoning"]["effort"], effort),
        None => assert!(response["reasoning"]["effort"].is_null()),
    }
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
    effort: Option<&str>,
) -> serde_json::Value {
    let response = client
        .post(endpoint)
        .json(&request(effort, false))
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
    effort: Option<&str>,
) -> serde_json::Value {
    let response = client
        .post(endpoint)
        .json(&request(effort, true))
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
    if effort != Some("none") {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN38_MODEL pointing to mlx-community/Qwen3.8-27B-4bit"]
async fn qwen38_all_reasoning_efforts_real_http_acceptance() {
    let port = alloc_port().await;
    let server = boot_server(port).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .no_proxy()
        .build()
        .expect("HTTP client");
    wait_until_healthy(&client, port).await;
    let endpoint = format!("http://127.0.0.1:{port}/v1/responses");

    let default = send_sync(&client, &endpoint, None).await;
    assert_completed_response(&default, None, true);

    let mut medium = None;
    for effort in EFFORTS {
        let response = send_sync(&client, &endpoint, Some(effort)).await;
        assert_completed_response(&response, Some(effort), effort != "none");
        if effort == "medium" {
            medium = Some(response);
        }
    }

    let default = send_stream(&client, &endpoint, None).await;
    assert_completed_response(&default, None, true);
    for effort in EFFORTS {
        let response = send_stream(&client, &endpoint, Some(effort)).await;
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
            "model": MODEL_ID,
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

    server.abort();
    let _ = server.await;
}
