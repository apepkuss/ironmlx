//! Real-checkpoint acceptance for Chat output budgets and request-scoped penalties.
//!
//! Set CHAT_PARAMETERS_MODEL to a local causal checkpoint. Optional
//! CHAT_PARAMETERS_MTP_MODEL / CHAT_PARAMETERS_DFLASH2_MODEL select speculative
//! serving in the fixed-server test. CHAT_PARAMETERS_BINARY can pin an external
//! production binary instead of Cargo’s integration-test binary.
//! The model-default test uses App daemon
//! routing and should use a small checkpoint because it registers three aliases.

#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;

use std::path::Path;
use std::process::{Child, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

struct Server {
    child: Child,
    root: tempfile::TempDir,
    url: String,
    client: reqwest::Client,
}

impl Server {
    async fn start(model: Option<&Path>) -> Self {
        let root = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut command = if let Some(binary) = std::env::var_os("CHAT_PARAMETERS_BINARY") {
            let mut command = std::process::Command::new(binary);
            command.arg("--mlx-metallib").arg(
                Path::new(&std::env::var_os("MLX_DIR").expect("MLX_DIR must be set"))
                    .join("lib/mlx.metallib"),
            );
            command
        } else {
            ironmlx_process::command()
        };
        command
            .args([
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--max-cache-cap",
                "4096",
                "--max-sequences",
                "2",
                "--prefill-chunk-size",
                "0",
                "--m5-dflash2-profile",
                "off",
                "--voice-profile-dir",
            ])
            .arg(root.path().join("voices"));
        if let Some(model) = model {
            command
                .arg("--model")
                .arg(model)
                .args(["--model-id", "chat-test"]);
            if let Some(mtp) = std::env::var_os("CHAT_PARAMETERS_MTP_MODEL") {
                command
                    .arg("--mtp-model-dir")
                    .arg(mtp)
                    .args(["--mtp-draft-tokens", "2"]);
            }
            if let Some(dflash) = std::env::var_os("CHAT_PARAMETERS_DFLASH2_MODEL") {
                command
                    .arg("--dflash2-model-dir")
                    .arg(dflash)
                    .args(["--dflash2-block-size", "8"]);
            }
        }
        let log = std::fs::File::create(root.path().join("server.log")).unwrap();
        command.stdout(Stdio::null()).stderr(Stdio::from(log));
        let child = command.spawn().unwrap();
        let mut server = Self {
            child,
            root,
            url: format!("http://127.0.0.1:{port}"),
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap(),
        };
        for _ in 0..720 {
            assert!(
                server.child.try_wait().unwrap().is_none(),
                "{}",
                server.log()
            );
            if server
                .client
                .get(format!("{}/healthz", server.url))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return server;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        panic!("server startup timeout: {}", server.log());
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.root.path().join("server.log")).unwrap()
    }

    async fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .json(body)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        let text = response.text().await.unwrap();
        (
            status,
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        )
    }

    async fn chat(&self, model: Option<&str>, fields: Value, expected: u64) -> Value {
        let mut body = chat_body(model);
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let (status, response) = self.post("/v1/chat/completions", &body).await;
        assert_eq!(
            status,
            200,
            "request={body}; response={response}; {}",
            self.log()
        );
        assert_eq!(
            response["usage"]["completion_tokens"], expected,
            "{response}"
        );
        assert_eq!(
            response["choices"][0]["finish_reason"], "length",
            "{response}"
        );
        response
    }

    async fn stream(&self, model: Option<&str>, fields: Value, expected: u64) {
        let mut body = chat_body(model);
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage":true});
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let text = response.text().await.unwrap();
        assert!(text.contains("data: [DONE]"), "{text}");
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect();
        assert!(
            !chunks.iter().any(|chunk| chunk.get("error").is_some()),
            "{text}"
        );
        let usage = chunks.iter().find_map(|chunk| chunk.get("usage")).unwrap();
        assert_eq!(usage["completion_tokens"], expected, "{text}");
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["finish_reason"] == "length"),
            "{text}"
        );
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn chat_body(model: Option<&str>) -> Value {
    let mut body = json!({
        "messages":[{"role":"user","content":"List consecutive integers from 1 to 50, separated by commas."}],
        "temperature":0, "ignore_eos":true,
        "chat_template_kwargs":{"enable_thinking":false}
    });
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    body
}

fn model_dir() -> std::path::PathBuf {
    std::env::var_os("CHAT_PARAMETERS_MODEL")
        .expect("CHAT_PARAMETERS_MODEL must be set")
        .into()
}

#[tokio::test]
#[ignore = "requires CHAT_PARAMETERS_MODEL pointing to a small local causal checkpoint"]
async fn chat_model_defaults_follow_routed_model_and_preserve_responses() {
    let server = Server::start(None).await;
    let model = model_dir();
    for (id, budget, default) in [
        ("short", Some(12), true),
        ("long", Some(20), false),
        ("fallback", None, false),
    ] {
        let (status, response) = server
            .post(
                "/admin/api/models/register",
                &json!({
                    "model":id,"model_dir":model,"set_default":default,
                    "max_cache_cap":4096,"default_max_output_tokens":budget
                }),
            )
            .await;
        assert_eq!(status, 200, "{response}");
    }
    server.chat(None, json!({}), 12).await;
    server
        .chat(
            Some("long"),
            json!({"max_tokens":null,"max_completion_tokens":null}),
            20,
        )
        .await;
    server.chat(Some("short"), json!({}), 12).await;
    server.chat(Some("fallback"), json!({}), 256).await;
    server
        .chat(
            Some("long"),
            json!({"max_completion_tokens":7,"max_tokens":3}),
            7,
        )
        .await;
    server.chat(Some("long"), json!({"max_tokens":6}), 6).await;
    server.stream(Some("short"), json!({}), 12).await;
    server
        .stream(
            Some("long"),
            json!({"max_completion_tokens":9,"max_tokens":4}),
            9,
        )
        .await;
    for id in ["short", "long"] {
        let (status, response) = server
            .post(
                "/v1/responses",
                &json!({
                    "model":id,"input":"Count from 1 to 50."
                }),
            )
            .await;
        assert_eq!(status, 200, "{response}");
        let expected = if id == "short" { 12 } else { 20 };
        assert_eq!(response["max_output_tokens"], expected, "{response}");
        assert!(
            response["usage"]["output_tokens"].as_u64().unwrap() <= expected,
            "{response}"
        );
        let (status, response) = server
            .post(
                "/v1/responses",
                &json!({
                    "model":id,"input":"Count from 1 to 50.","max_output_tokens":5
                }),
            )
            .await;
        assert_eq!(status, 200, "{response}");
        assert_eq!(response["max_output_tokens"], 5, "{response}");
        assert!(
            response["usage"]["output_tokens"].as_u64().unwrap() <= 5,
            "{response}"
        );
    }
}

#[tokio::test]
#[ignore = "requires CHAT_PARAMETERS_MODEL and optional local MTP / DFlash2 draft"]
async fn chat_fixed_serving_budgets_penalties_streams_and_capacity() {
    let server = Server::start(Some(&model_dir())).await;
    let baseline = server
        .chat(Some("chat-test"), json!({"max_completion_tokens":24}), 24)
        .await;
    server.chat(None, json!({"max_tokens":7}), 7).await;
    server
        .chat(None, json!({"max_completion_tokens":11,"max_tokens":4}), 11)
        .await;
    server
        .chat(
            None,
            json!({"max_tokens":6,"max_completion_tokens":null}),
            6,
        )
        .await;
    server
        .chat(
            None,
            json!({"max_completion_tokens":null,"max_tokens":null}),
            256,
        )
        .await;
    let (penalized, encouraged) = tokio::join!(
        server.chat(
            None,
            json!({"max_completion_tokens":24,
            "frequency_penalty":2,"presence_penalty":2}),
            24
        ),
        server.chat(
            None,
            json!({"max_completion_tokens":24,
            "frequency_penalty":-2,"presence_penalty":-2}),
            24
        ),
    );
    assert_ne!(
        penalized["choices"][0]["message"],
        encouraged["choices"][0]["message"]
    );
    let repeated = server
        .chat(
            None,
            json!({"max_completion_tokens":24,
        "frequency_penalty":0,"presence_penalty":null}),
            24,
        )
        .await;
    assert_eq!(
        baseline["choices"][0]["message"],
        repeated["choices"][0]["message"]
    );
    server
        .stream(
            None,
            json!({"max_completion_tokens":13,"max_tokens":5,
        "frequency_penalty":0.5,"presence_penalty":-0.25}),
            13,
        )
        .await;
    server.stream(None, json!({"max_tokens":10}), 10).await;
    server
        .chat(
            None,
            json!({"max_completion_tokens":16,"temperature":0.7,"seed":42,
        "frequency_penalty":0.3,"presence_penalty":-0.2}),
            16,
        )
        .await;
    server
        .stream(
            None,
            json!({"max_completion_tokens":16,"temperature":0.7,"seed":42,
        "frequency_penalty":-0.3,"presence_penalty":0.2}),
            16,
        )
        .await;
    for fields in [
        json!({"max_completion_tokens":0,"max_tokens":8}),
        json!({"max_completion_tokens":8,"max_tokens":0}),
        json!({"frequency_penalty":2.01}),
        json!({"presence_penalty":-2.01}),
        json!({"max_completion_tokens":"8"}),
        json!({"unexpected":true}),
    ] {
        let mut body = chat_body(None);
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let (status, response) = server.post("/v1/chat/completions", &body).await;
        assert_eq!(status, 400, "{response}");
        assert_eq!(response["error"]["type"], "invalid_request_error");
    }
    let mut body = chat_body(None);
    body["max_completion_tokens"] = json!(4096);
    body["max_tokens"] = json!(1);
    let (status, response) = server.post("/v1/chat/completions", &body).await;
    assert_eq!(status, 413, "{response}");
    assert_eq!(
        response["error"]["details"]["requested_max_output_tokens"],
        4096
    );
    let health: Value = server
        .client
        .get(format!("{}/healthz", server.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    if std::env::var_os("CHAT_PARAMETERS_MTP_MODEL").is_some() {
        assert_eq!(health["mtp"]["enabled"], true, "{health}");
        assert!(
            health["mtp"]["drafted_tokens"].as_u64().unwrap() > 0,
            "{health}"
        );
    }
    if std::env::var_os("CHAT_PARAMETERS_DFLASH2_MODEL").is_some() {
        assert_eq!(health["dflash2"]["enabled"], true, "{health}");
        assert!(
            health["dflash2"]["drafted_tokens"].as_u64().unwrap() > 0,
            "{health}"
        );
    }
    println!("chat parameter acceptance passed; health={health}");
}
