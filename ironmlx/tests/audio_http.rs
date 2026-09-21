//! Real-model speech transport and lifecycle acceptance (explicit local resources).
//!
//! IRONMLX_AUDIO_HTTP_MODEL points to JSON {"path": <snapshot>, "audio":
//! {"derived_resources": ..., "resource_lock": ..., "wetext_fsts": ...,
//! "unidic_dir": ...}}. IRONMLX_AUDIO_HTTP_REFERENCE is a 1–60 second WAV.
//! IRONMLX_AUDIO_HTTP_LLM is a local supported LLM snapshot for coexistence.
//! Run with MLX_ENABLE_TF32=0 and cargo test -p ironmlx --release --test audio_http
//! -- --ignored --nocapture --test-threads=1. No downloads or resource conversion.
//! Set IRONMLX_AUDIO_HTTP_BUNDLE to an assembled IronMLX.app to exercise its
//! helper and metallib instead. That mode removes all MLX_/DYLD_ overrides and
//! runs outside the checkout; the executable supplies its own precision default.
#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;
use base64::Engine;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    process::{Child, Stdio},
    time::{Duration, Instant},
};

struct Server {
    child: Child,
    client: reqwest::Client,
    base: String,
    root: PathBuf,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Server {
    async fn start() -> Self {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let root =
            std::env::temp_dir().join(format!("ironmlx-audio-http-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut command = if let Some(bundle) = std::env::var_os("IRONMLX_AUDIO_HTTP_BUNDLE") {
            let bundle = PathBuf::from(bundle).canonicalize().unwrap();
            let helper = bundle.join("Contents/Helpers/ironmlx");
            let metallib = bundle.join("Contents/Resources/mlx.metallib");
            assert!(helper.is_file() && metallib.is_file(), "incomplete Bundle");
            let mut command = std::process::Command::new(&helper);
            for (key, _) in std::env::vars_os() {
                if key.to_string_lossy().starts_with("MLX_")
                    || key.to_string_lossy().starts_with("DYLD_")
                {
                    command.env_remove(key);
                }
            }
            command.arg("--mlx-metallib").arg(&metallib);
            command.current_dir(&root);
            eprintln!(
                "Bundle helper: {}; metallib: {}",
                helper.display(),
                metallib.display()
            );
            command
        } else {
            let mut command = ironmlx_process::command();
            command.env("MLX_ENABLE_TF32", "0");
            command
        };
        let child = command
            .args(["serve", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(root.join("server.log")).unwrap())
            .spawn()
            .unwrap();
        let server = Self {
            child,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(330))
                .build()
                .unwrap(),
            base: format!("http://127.0.0.1:{port}"),
            root,
        };
        for _ in 0..200 {
            if server
                .client
                .get(format!("{}/health", server.base))
                .send()
                .await
                .is_ok()
            {
                return server;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("server startup failed: {}", server.root.display());
    }
    async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await
            .unwrap()
    }
    async fn memory(&self) -> Value {
        let health: Value = self
            .client
            .get(format!("{}/healthz", self.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        health["memory"].clone()
    }
    async fn models(&self) -> Vec<Value> {
        self.client
            .get(format!("{}/admin/api/models/loaded", self.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }
    async fn load(&self, fixture: &Value, id: &str, policy: Value) {
        let mut audio = fixture["audio"].clone();
        audio["execution"] = policy;
        let started = Instant::now();
        let response = self
            .post(
                "/admin/api/models/load",
                &json!({"model":id,"model_dir":fixture["path"],"audio":audio}),
            )
            .await;
        let status = response.status();
        let body = response.text().await.unwrap();
        assert!(status.is_success(), "load {id}: {status} {body}");
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["success"],
            true,
            "{body}"
        );
        eprintln!("load {id}: {:?}", started.elapsed());
    }
    async fn unload(&self, id: &str) {
        let response = self
            .post("/admin/api/models/unload", &json!({"model":id}))
            .await;
        assert!(response.status().is_success());
        for _ in 0..500 {
            if !self.models().await.iter().any(|m| m["id"] == id) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("model {id} did not unload");
    }
    async fn idle(&self, id: &str) {
        for _ in 0..500 {
            if self
                .models()
                .await
                .iter()
                .any(|m| m["id"] == id && m["active_requests"] == 0 && m["queued_requests"] == 0)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("model {id} did not release execution");
    }
}
fn request(id: &str, reference: &str, text: &str, stream: bool) -> Value {
    json!({"model":id,"input":text,"ref_audio":reference,"stream":stream,"response_format":if stream {"pcm"} else {"wav"}})
}
fn headers(response: &reqwest::Response, stream: bool) {
    assert_eq!(response.status(), 200);
    let h = response.headers();
    assert_eq!(
        h["content-type"],
        if stream { "audio/pcm" } else { "audio/wav" }
    );
    assert_eq!(h["x-audio-sample-rate"], "22050");
    assert_eq!(h["x-audio-channels"], "1");
    assert_eq!(h["x-audio-sample-format"], "s16le");
    assert!(h.contains_key("x-request-id"));
    if stream {
        assert!(!h.contains_key("content-length"));
        assert_eq!(h["x-ironmlx-streaming-granularity"], "segment");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires pinned audio resources, reference WAV, and local LLM snapshot"]
async fn real_speech_http_lifecycle() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(std::env::var("IRONMLX_AUDIO_HTTP_MODEL").unwrap()).unwrap(),
    )
    .unwrap();
    let reference = base64::engine::general_purpose::STANDARD
        .encode(std::fs::read(std::env::var("IRONMLX_AUDIO_HTTP_REFERENCE").unwrap()).unwrap());
    let server = Server::start().await;
    let initial_memory = server.memory().await;
    // Five incomplete bodies exhaust bounded input admission without starting
    // any native decode. Disconnecting releases every body reservation.
    let address = server.base.strip_prefix("http://").unwrap();
    let mut incomplete = Vec::new();
    use tokio::io::AsyncWriteExt;
    for _ in 0..5 {
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket.write_all(b"POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1000000\r\n\r\n{").await.unwrap();
        incomplete.push(socket);
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    let overloaded = server.post("/v1/audio/speech", &json!({})).await;
    assert_eq!(overloaded.status(), 503);
    drop(incomplete);
    for _ in 0..100 {
        if server.memory().await["process_governor"]["reserved_bytes"] == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        server.memory().await["process_governor"]["reserved_bytes"],
        0
    );

    eprintln!("server logs: {}", server.root.display());
    server
        .load(&fixture, "tts", json!({"segment_tokens":12}))
        .await;
    let short = "你好，这是语音测试。今天的天气很好。";
    let long = short.repeat(12);
    let response = server
        .post(
            "/v1/audio/speech",
            &request("tts", &reference, short, false),
        )
        .await;
    headers(&response, false);
    let length = response.headers()["content-length"]
        .to_str()
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let wav = response.bytes().await.unwrap();
    assert_eq!(wav.len(), length);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(
        u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize + 8,
        wav.len()
    );
    assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 22050);
    assert!(wav[44..].iter().any(|b| *b != 0));
    let started = Instant::now();
    let mut response = server
        .post("/v1/audio/speech", &request("tts", &reference, &long, true))
        .await;
    headers(&response, true);
    let first = started.elapsed();
    let mut pcm = Vec::new();
    while let Some(chunk) = response.chunk().await.unwrap() {
        pcm.extend_from_slice(&chunk);
    }
    assert!(started.elapsed() > first + Duration::from_millis(100));
    assert!(!pcm.is_empty() && pcm.len() % 2 == 0);
    assert_ne!(&pcm[..4], b"RIFF");
    // A client can split each transport chunk on arbitrary byte boundaries.
    let reassembled: Vec<u8> = pcm.chunks(777).flatten().copied().collect();
    assert_eq!(pcm, reassembled);
    eprintln!(
        "PCM first={first:?} total={:?} bytes={}",
        started.elapsed(),
        pcm.len()
    );
    server.idle("tts").await;

    for (body, status) in [
        (
            json!({"model":"tts","input":"x","ref_audio":reference,"voice":"x"}),
            400,
        ),
        (request("tts", &reference, " ", false), 400),
        (
            request("tts", "data:audio/wav;base64,AAAA", "x", false),
            400,
        ),
        (request("unknown", &reference, "x", false), 404),
        (request("tts", &reference, &"字".repeat(22000), false), 413),
    ] {
        let r = server.post("/v1/audio/speech", &body).await;
        let actual = r.status();
        let text = r.text().await.unwrap();
        assert_eq!(actual.as_u16(), status, "{text}");
        assert!(serde_json::from_str::<Value>(&text).unwrap()["error"].is_object());
    }
    let r = server
        .post(
            "/v1/chat/completions",
            &json!({"model":"tts","messages":[{"role":"user","content":"Hello"}]}),
        )
        .await;
    assert_eq!(r.status(), 400);

    // A disconnected consumer must cancel the active session and permit unload.
    let response = server
        .post(
            "/v1/audio/speech",
            &request("tts", &reference, &short.repeat(100), true),
        )
        .await;
    headers(&response, true);
    drop(response);
    server.idle("tts").await;

    // Exercise the full native lane: one executing plus four queued requests.
    let response = server
        .post(
            "/v1/audio/speech",
            &request("tts", &reference, &short.repeat(100), true),
        )
        .await;
    headers(&response, true);
    let mut queued = Vec::new();
    for expected in 1..=4 {
        let client = server.client.clone();
        let url = format!("{}/v1/audio/speech", server.base);
        let body = request("tts", &reference, short, true);
        queued.push(tokio::spawn(async move {
            client.post(url).json(&body).send().await
        }));
        let mut reached = false;
        for _ in 0..200 {
            if server
                .models()
                .await
                .iter()
                .any(|m| m["id"] == "tts" && m["queued_requests"] == expected)
            {
                reached = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(reached, "queue did not reach {expected}");
    }
    let overloaded = server
        .post("/v1/audio/speech", &request("tts", &reference, short, true))
        .await;
    assert_eq!(overloaded.status(), 503);
    assert!(overloaded.headers().contains_key("retry-after"));
    for task in queued {
        task.abort();
        let _ = task.await;
    }
    drop(response);
    server.idle("tts").await;
    server.unload("tts").await;

    server
        .load(
            &fixture,
            "queue-timeout",
            json!({"queue_timeout_ms":20,"segment_tokens":12}),
        )
        .await;
    let active = server
        .post(
            "/v1/audio/speech",
            &request("queue-timeout", &reference, &short.repeat(100), true),
        )
        .await;
    headers(&active, true);
    let timeout = server
        .post(
            "/v1/audio/speech",
            &request("queue-timeout", &reference, short, true),
        )
        .await;
    assert_eq!(timeout.status(), 504);
    drop(active);
    server.idle("queue-timeout").await;
    server.unload("queue-timeout").await;

    server
        .load(
            &fixture,
            "slow-consumer",
            json!({"slow_consumer_timeout_ms":200,"segment_tokens":12}),
        )
        .await;
    let mut slow = server
        .post(
            "/v1/audio/speech",
            &request("slow-consumer", &reference, &short.repeat(100), true),
        )
        .await;
    headers(&slow, true);
    let started = Instant::now();
    // Do not poll the body: reqwest/hyper and the TCP receive window eventually
    // fill, propagating pressure to the bounded native output channel.
    loop {
        if server
            .models()
            .await
            .iter()
            .any(|m| m["id"] == "slow-consumer" && m["active_requests"] == 0)
        {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "slow consumer failed to stop worker"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut failed = false;
    loop {
        match slow.chunk().await {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(_) => {
                failed = true;
                break;
            }
        }
    }
    assert!(failed, "slow consumer produced normal EOF");
    eprintln!("backpressure cancellation {:?}", started.elapsed());
    server.unload("slow-consumer").await;

    server
        .load(
            &fixture,
            "first-timeout",
            json!({"first_audio_timeout_ms":1,"segment_tokens":12}),
        )
        .await;
    let r = server
        .post(
            "/v1/audio/speech",
            &request("first-timeout", &reference, short, true),
        )
        .await;
    assert_eq!(r.status(), 504);
    server.idle("first-timeout").await;
    server.unload("first-timeout").await;

    // A bounded output policy triggers a genuine model generation failure after
    // earlier segments have already crossed the HTTP boundary.
    server
        .load(
            &fixture,
            "output-limit",
            json!({"max_output_frames":200000,"segment_tokens":12}),
        )
        .await;
    let mut r = server
        .post(
            "/v1/audio/speech",
            &request("output-limit", &reference, &long, true),
        )
        .await;
    headers(&r, true);
    let mut failed = false;
    let mut bytes = 0;
    loop {
        match r.chunk().await {
            Ok(Some(b)) => bytes += b.len(),
            Ok(None) => break,
            Err(_) => {
                failed = true;
                break;
            }
        }
    }
    assert!(
        failed && bytes > 0,
        "post-200 generation failure became normal EOF"
    );
    server.idle("output-limit").await;
    server.unload("output-limit").await;

    server
        .load(
            &fixture,
            "execution-timeout",
            json!({"execution_timeout_ms":8000,"segment_tokens":12}),
        )
        .await;
    let mut r = server
        .post(
            "/v1/audio/speech",
            &request("execution-timeout", &reference, &long, true),
        )
        .await;
    headers(&r, true);
    let mut failed = false;
    loop {
        match r.chunk().await {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(_) => {
                failed = true;
                break;
            }
        }
    }
    assert!(failed, "post-200 execution timeout became normal EOF");
    server.idle("execution-timeout").await;
    server.unload("execution-timeout").await;

    // Both model families run in the same process and memory governor, on their
    // own registered streams. The LLM response must remain usable during TTS.
    server
        .load(&fixture, "coexist", json!({"segment_tokens":12}))
        .await;
    let llm = std::env::var("IRONMLX_AUDIO_HTTP_LLM").unwrap();
    let r = server
        .post(
            "/admin/api/models/load",
            &json!({"model":"llm","model_dir":llm,"max_cache_cap":2048}),
        )
        .await;
    let status = r.status();
    let text = r.text().await.unwrap();
    assert!(status.is_success(), "{text}");
    let mut audio = server
        .post(
            "/v1/audio/speech",
            &request("coexist", &reference, &long, true),
        )
        .await;
    headers(&audio, true);
    let chat=server.post("/v1/chat/completions",&json!({"model":"llm","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0})).await;
    let status = chat.status();
    let text = chat.text().await.unwrap();
    assert_eq!(status, 200, "{text}");
    let chat: Value = serde_json::from_str(&text).unwrap();
    assert!(
        chat["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "{chat}"
    );
    let mut bytes = 0;
    while let Some(chunk) = audio.chunk().await.unwrap() {
        bytes += chunk.len();
    }
    assert!(bytes > 0);
    server.idle("coexist").await;
    server.unload("coexist").await;
    server.unload("llm").await;
    assert!(server.models().await.is_empty());
    // The existing causal scheduler owns its model on a detached thread and
    // exits after its command sender closes. Observe that real completion,
    // rather than equating an empty model registry with an already joined actor.
    let reclaim_started = Instant::now();
    let allowed_active = initial_memory["mlx_active_bytes"].as_u64().unwrap() + 64 * 1024 * 1024;
    let immediate_memory = server.memory().await;
    let mut final_memory = immediate_memory.clone();
    while final_memory["mlx_active_bytes"].as_u64().unwrap() > allowed_active
        || final_memory["process_governor"]["reserved_bytes"] != 0
    {
        assert!(
            reclaim_started.elapsed() < Duration::from_secs(10),
            "active allocations or reservations survived actor shutdown: {final_memory}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        final_memory = server.memory().await;
    }
    eprintln!(
        "unload reclaim: initial_active={} immediate_active={} final_active={} waited={:?}",
        initial_memory["mlx_active_bytes"],
        immediate_memory["mlx_active_bytes"],
        final_memory["mlx_active_bytes"],
        reclaim_started.elapsed()
    );
    eprintln!("final memory: {final_memory}");
    eprintln!("real speech HTTP lifecycle passed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires pinned audio resources and reference WAV"]
async fn real_speech_unload_defers_until_worker_stops() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(std::env::var("IRONMLX_AUDIO_HTTP_MODEL").unwrap()).unwrap(),
    )
    .unwrap();
    let reference = base64::engine::general_purpose::STANDARD
        .encode(std::fs::read(std::env::var("IRONMLX_AUDIO_HTTP_REFERENCE").unwrap()).unwrap());
    let server = Server::start().await;
    server
        .load(&fixture, "tts", json!({"segment_tokens":12}))
        .await;
    let mut output = server
        .post(
            "/v1/audio/speech",
            &request(
                "tts",
                &reference,
                &"你好，这是语音测试。今天的天气很好。".repeat(100),
                true,
            ),
        )
        .await;
    headers(&output, true);
    let response = server
        .post("/admin/api/models/unload", &json!({"model":"tts"}))
        .await;
    assert_eq!(response.status(), 200);
    let result: Value = response.json().await.unwrap();
    assert_eq!(result["status"], "unload_deferred", "{result}");
    let list: Value = server
        .client
        .get(format!("{}/v1/models", server.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["data"][0]["state"], "draining");
    // An active request still owns native computation memory while the registry
    // drains, and remains able to return audio until its consumer disconnects.
    assert!(
        server.memory().await["process_governor"]["reserved_bytes"]
            .as_u64()
            .unwrap()
            >= 32 * 1024 * 1024 * 1024
    );
    assert!(!output.chunk().await.unwrap().unwrap().is_empty());
    drop(output);
    let started = Instant::now();
    loop {
        let list: Value = server
            .client
            .get(format!("{}/v1/models", server.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let memory = server.memory().await;
        if list["data"][0]["state"] == "unloaded"
            && memory["process_governor"]["reserved_bytes"] == 0
            && memory["mlx_active_bytes"].as_u64().unwrap() < 64 * 1024 * 1024
        {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "draining worker retained resources: {list} {memory}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    eprintln!(
        "active audio unload completed safely after disconnect: {:?}",
        started.elapsed()
    );
}
