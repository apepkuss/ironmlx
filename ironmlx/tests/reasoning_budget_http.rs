//! Real-model acceptance for automatic reasoning reservation. Run once per
//! checkpoint (Qwen and Gemma) with IRONMLX_REASONING_MODEL and --ignored.
//! Optional IRONMLX_REASONING_MTP_MODEL adds a compatible MTP drafter.
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn boot(model: &str, scheduled: bool) -> (Server, reqwest::Client, String) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let mut command = Command::new(env!("CARGO_BIN_EXE_ironmlx"));
    command.args([
        "serve",
        "--model",
        model,
        "--model-id",
        "budget-test",
        "--port",
        &port.to_string(),
        "--max-sequences",
        "1",
        "--max-cache-cap",
        "8192",
        "--prefill-chunk-size",
        "2048",
    ]);
    if scheduled {
        command.arg("--force-scheduler");
    }
    if let Ok(drafter) = std::env::var("IRONMLX_REASONING_MTP_MODEL") {
        command.args(["--mtp-model-dir", &drafter, "--mtp-draft-tokens", "3"]);
    }
    let mut server = Server(
        command
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    for _ in 0..600 {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited before health check"
        );
        if client
            .get(format!("{base}/healthz"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            return (server, client, base);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("model server did not become healthy");
}

async fn response(client: &reqwest::Client, base: &str, body: Value) -> (Value, Vec<Value>) {
    let streaming = body["stream"] == true;
    let response = client
        .post(format!("{base}/v1/responses"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let wire = response.text().await.unwrap();
    assert!(status.is_success(), "HTTP {status}: {wire}");
    if !streaming {
        return (serde_json::from_str(&wire).unwrap(), vec![]);
    }
    let events: Vec<Value> = wire
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let terminal = events.last().expect("terminal SSE event");
    assert!(
        matches!(
            terminal["type"].as_str(),
            Some("response.completed" | "response.incomplete")
        ),
        "{wire}"
    );
    (terminal["response"].clone(), events)
}

fn text(response: &Value) -> String {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().unwrap())
        .filter_map(|part| part["text"].as_str())
        .collect()
}

#[tokio::test]
#[ignore = "requires IRONMLX_REASONING_MODEL real Qwen/Gemma checkpoint and Metal"]
async fn real_reasoning_budget_preserves_answer_and_replay() {
    let model = std::env::var("IRONMLX_REASONING_MODEL").unwrap();
    for scheduled in [false, true] {
        let (_server, client, base) = boot(&model, scheduled).await;
        for stream in [false, true] {
            let (result, events) = response(
                &client,
                &base,
                json!({
                    "model":"budget-test", "input":"What is 2+2? Answer with only the number.",
                    "reasoning":{"effort":"high"}, "max_output_tokens":256,
                    "temperature":0, "stream":stream, "store":false
                }),
            )
            .await;
            eprintln!("scheduled={scheduled} stream={stream}: {result}");
            assert_eq!(result["status"], "completed");
            assert!(result["incomplete_details"].is_null());
            assert_eq!(text(&result).trim(), "4", "{result}");
            assert!(!text(&result).contains("</think>") && !text(&result).contains("<channel|>"));
            let used = result["usage"]["output_tokens"].as_u64().unwrap();
            let thought = result["usage"]["output_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap();
            assert!(used <= 256 && thought > 0 && thought < used, "{result}");
            for item in result["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|i| i["type"] == "reasoning")
            {
                assert_eq!(item["status"], "completed");
                if stream {
                    assert!(events
                        .iter()
                        .any(|event| event["type"] == "response.output_item.done"
                            && event["item"] == *item));
                }
            }
            let mut history = vec![json!({"role":"user", "content":"What is 2+2?"})];
            history.extend(result["output"].as_array().unwrap().clone());
            history.push(
                json!({"role":"user", "content":"What is 3+3? Answer with only the number."}),
            );
            let (continued, _) = response(
                &client,
                &base,
                json!({"model":"budget-test", "input":history,
                "max_output_tokens":64, "temperature":0, "store":false}),
            )
            .await;
            assert_eq!(continued["status"], "completed");
            assert_eq!(text(&continued).trim(), "6", "{continued}");
        }
        let (structured, _) = response(
            &client,
            &base,
            json!({"model":"budget-test",
            "input":"What is 2+2? Return JSON with answer 4.", "reasoning":{"effort":"high"},
            "max_output_tokens":128, "temperature":0, "store":false,
            "text":{"format":{"type":"json_schema", "name":"answer", "strict":true,
                "schema":{"type":"object", "properties":{"answer":{"const":4}},
                    "required":["answer"], "additionalProperties":false}}}}),
        )
        .await;
        assert_eq!(structured["status"], "completed", "{structured}");
        assert_eq!(
            serde_json::from_str::<Value>(&text(&structured)).unwrap(),
            json!({"answer":4})
        );

        // An adversarial request can still exceed the whole-response limit:
        // reservation promises a native transition, not answer correctness or
        // instruction following. Preserve truthful truncation and replay.
        let (long, long_events) = response(&client, &base, json!({
            "model":"budget-test", "input":"In your reasoning, count from 1 to 100, then calculate 2+2. Your final answer must be only the number 4.",
            "reasoning":{"effort":"high"}, "max_output_tokens":128,
            "temperature":0, "store":false, "stream":true
        })).await;
        eprintln!("long reasoning scheduled={scheduled}: {long}");
        assert!(
            !text(&long).trim().is_empty(),
            "answer space was not used: {long}"
        );
        assert!(long["usage"]["output_tokens"].as_u64().unwrap() <= 128);
        if long["status"] == "incomplete" {
            assert_eq!(long["incomplete_details"]["reason"], "max_output_tokens");
        } else {
            assert_eq!(long["status"], "completed");
        }
        for item in long["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i["type"] == "reasoning")
        {
            assert_eq!(item["status"], "completed");
            assert!(long_events.iter().any(
                |event| event["type"] == "response.output_item.done" && event["item"] == *item
            ));
        }
        let mut history = vec![json!({"role":"user", "content":"What is 2+2?"})];
        history.extend(long["output"].as_array().unwrap().clone());
        history.push(json!({"role":"user", "content":"What is 3+3? Answer with only the number."}));
        let (resumed, _) = response(
            &client,
            &base,
            json!({"model":"budget-test",
            "input":history, "max_output_tokens":64, "store":false, "temperature":0}),
        )
        .await;
        assert_eq!(resumed["status"], "completed", "{resumed}");
        assert_eq!(text(&resumed).trim(), "6", "{resumed}");

        let (tiny, _) = response(
            &client,
            &base,
            json!({"model":"budget-test",
            "input":"Explain how to multiply 123 by 456.", "reasoning":{"effort":"high"},
            "max_output_tokens":1, "store":false, "temperature":0}),
        )
        .await;
        assert_eq!(tiny["status"], "incomplete", "{tiny}");
        assert_eq!(tiny["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(tiny["usage"]["output_tokens"], 1);

        let (tool, _) = response(
            &client,
            &base,
            json!({"model":"budget-test",
            "input":"Think about the question then call answer with value 4.",
            "reasoning":{"effort":"high"}, "max_output_tokens":256,
            "tools":[{"type":"function", "name":"answer", "parameters":{
                "type":"object", "properties":{"value":{"type":"integer","const":4}},
                "required":["value"], "additionalProperties":false}}],
            "tool_choice":"required", "parallel_tool_calls":false,
            "store":false, "temperature":0}),
        )
        .await;
        assert_eq!(tool["status"], "completed", "{tool}");
        let call = tool["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("tool call after reasoning");
        assert_eq!(call["name"], "answer");
        assert_eq!(
            serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap(),
            json!({"value":4})
        );
        assert!(tool["usage"]["output_tokens"].as_u64().unwrap() <= 256);

        // The shared preparation helper also protects the other two APIs.
        for (endpoint, body) in [
            (
                "chat/completions",
                json!({"model":"budget-test", "messages":[{
                "role":"user", "content":"What is 2+2? Answer with only the number."}],
                "chat_template_kwargs":{"enable_thinking":true}, "max_tokens":128, "temperature":0}),
            ),
            (
                "messages",
                json!({"model":"budget-test", "messages":[{
                "role":"user", "content":"What is 2+2? Answer with only the number."}],
                "thinking":{"type":"adaptive"}, "max_tokens":128, "temperature":0}),
            ),
        ] {
            let response = client
                .post(format!("{base}/v1/{endpoint}"))
                .json(&body)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let wire = response.text().await.unwrap();
            assert!(status.is_success(), "{endpoint}: HTTP {status}: {wire}");
            let result: Value = serde_json::from_str(&wire).unwrap();
            if endpoint == "messages" {
                assert!(result["usage"]["output_tokens"].as_u64().unwrap() <= 128);
                assert!(
                    result["content"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|block| block["type"] == "text"
                            && !block["text"].as_str().unwrap().trim().is_empty()),
                    "{wire}"
                );
            } else {
                assert!(result["usage"]["completion_tokens"].as_u64().unwrap() <= 128);
                assert!(
                    matches!(
                        result["choices"][0]["finish_reason"].as_str(),
                        Some("length" | "stop")
                    ),
                    "{wire}"
                );
            }
        }
    }
}
