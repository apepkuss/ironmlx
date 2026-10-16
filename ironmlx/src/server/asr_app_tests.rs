use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn json_request(app: &Router, path: &str, value: Value) -> Value {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(value.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}
async fn transcribe(
    app: &Router,
    model: &str,
    audio: &[u8],
    extra: &[(&str, &str)],
) -> (StatusCode, Vec<u8>) {
    let boundary = "ironmlx-asr-test";
    let mut data = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
    )
    .into_bytes();
    for (name, value) in extra {
        data.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    }
    data.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    data.extend_from_slice(audio);
    data.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(data))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.headers().contains_key("x-request-id"));
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, bytes.to_vec())
}

#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires both local Qwen3-ASR checkpoints and reference Chinese WAV"]
async fn qwen3_asr_real_app_lifecycle_and_api() {
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        args.max_loaded_models,
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let voices_dir = tempfile::tempdir().unwrap();
    let voices = crate::server::voices::VoiceStore::open(voices_dir.path().to_path_buf()).unwrap();
    let app = app_router(manager.clone(), voices);
    let audio = std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap();
    for (size, var) in [
        ("0.6B", "IRONMLX_ASR_SMALL_DIR"),
        ("1.7B", "IRONMLX_ASR_LARGE_DIR"),
    ] {
        let model = format!("mlx-community/Qwen3-ASR-{size}-8bit");
        let path = std::env::var(var).unwrap();
        let response = json_request(
            &app,
            "/admin/api/models/register",
            json!({"model":model,"model_dir":path}),
        )
        .await;
        assert_eq!(response["success"], true, "{response}");
        let response = json_request(
            &app,
            "/admin/api/models/load",
            json!({"model":model,"model_dir":path}),
        )
        .await;
        assert_eq!(response["success"], true, "{response}");
        let loaded = response["loaded_models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == model)
            .unwrap();
        assert_eq!(loaded["runtime_kind"], "asr");
        assert_eq!(loaded["supports_audio"], true);
        assert_eq!(loaded["supports_kv_cache"], false);
        assert_eq!(
            manager.pool.asr_settings(&model).await.unwrap(),
            ironmlx_runtime::core::asr_execution::AsrSettings::default()
        );
        let settings_path = "/admin/api/models/asr";
        json_request(&app, settings_path, json!({"model":model,"settings":{"max_audio_duration_seconds":3,"max_output_tokens":1024}})).await;
        let (status, _) = transcribe(&app, &model, &audio, &[]).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "Per-model duration was not enforced"
        );
        json_request(&app, settings_path, json!({"model":model,"settings":{"max_audio_duration_seconds":7200,"max_output_tokens":1}})).await;
        let (status, bytes) = transcribe(&app, &model, &audio, &[]).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["code"],
            "asr_output_limit_exceeded"
        );
        json_request(&app, settings_path, json!({"model":model,"settings":{"max_audio_duration_seconds":7200,"max_output_tokens":1024}})).await;
        let (status, bytes) = transcribe(&app, &model, &audio, &[]).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["text"], "甚至出现交易几乎停滞的情况。");
        let (status, bytes) = transcribe(
            &app,
            &model,
            &audio,
            &[("language", "zh"), ("response_format", "text")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "甚至出现交易几乎停滞的情况。"
        );
        for fields in [
            vec![("stream", "true"), ("response_format", "text")],
            vec![("response_format", "srt")],
            vec![("language", "not-a-language")],
            vec![("temperature", "0.5")],
            vec![("model", "duplicate")],
        ] {
            let (status, bytes) = transcribe(&app, &model, &audio, &fields).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            assert!(
                serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["message"].is_string()
            );
        }
        for invalid in [b"not audio".as_slice(), b"".as_slice()] {
            let (status, _) = transcribe(&app, &model, invalid, &[]).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
        for path in ["/v1/chat/completions", "/v1/audio/speech"] {
            let response = app.clone().oneshot(
                Request::builder().method("POST").uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"model":model,"messages":[{"role":"user","content":"hello"}],"input":"hello","voice":"default"}).to_string())).unwrap()
            ).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
        use ironmlx_audio::AudioIo;
        let io = ironmlx_audio::io::NativeAudioIo;
        let source = io.decode(&audio, &Default::default()).unwrap();
        let mut long = source.clone();
        long.samples = source.samples.repeat(300);
        let long_wav = io.encode_wav(&long).unwrap();
        let ((status, bytes), ()) = tokio::join!(
            transcribe(&app, &model, &long_wav, &[]),
            realtime_case(&app, &model, &source)
        );
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let long_result: Value = serde_json::from_slice(&bytes).unwrap();
        let transcript = long_result["text"].as_str().unwrap();
        assert!(transcript.matches("交易").count() >= 270, "{transcript}");
        println!(
            "{model}: long recording {:.2}s, recognized {} repeated utterances",
            long.samples.len() as f64 / long.format.sample_rate as f64,
            transcript.matches("交易").count()
        );
        let mut stream_audio = source.clone();
        stream_audio.samples = source.samples.repeat(10);
        let stream_wav = io.encode_wav(&stream_audio).unwrap();
        let (status, bytes) = transcribe(&app, &model, &stream_wav, &[("stream", "true")]).await;
        assert_eq!(status, StatusCode::OK);
        let events = String::from_utf8(bytes).unwrap();
        assert!(events.contains("transcript.text.delta"), "{events}");
        assert!(events.contains("transcript.text.done"), "{events}");
        assert!(!events.contains("event: error"), "{events}");
        let (status, _) = transcribe(&app, "missing-model", &audio, &[]).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        json_request(&app, settings_path, json!({"model":model,"settings":{"max_audio_duration_seconds":1200,"max_output_tokens":1024}})).await;
        let response = json_request(&app, "/admin/api/models/unload", json!({"model":model})).await;
        assert_eq!(response["success"], true, "{response}");
    }
}

struct NoCancel;
impl ironmlx_audio::SessionControl for NoCancel {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}

async fn realtime_case(app: &Router, model: &str, audio: &ironmlx_audio::PcmBuffer) {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = app.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (mut socket, _) = connect_async(format!(
        "ws://127.0.0.1:{port}/v1/realtime?intent=transcription&model={model}"
    ))
    .await
    .unwrap();
    let created = socket.next().await.unwrap().unwrap();
    assert!(created.into_text().unwrap().contains("session.created"));
    socket.send(Message::Text(json!({"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":model,"language":"zh"},"turn_detection":null}}}}).to_string())).await.unwrap();
    assert!(socket
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap()
        .contains("session.updated"));
    let samples = ironmlx_audio::signal::resample_mono(
        &audio.samples.repeat(9),
        audio.format.sample_rate,
        24000,
        &NoCancel,
    )
    .unwrap();
    let bytes = samples
        .iter()
        .flat_map(|x| ((x.clamp(-1., 1.) * 32767.) as i16).to_le_bytes())
        .collect::<Vec<_>>();
    use base64::{engine::general_purpose::STANDARD, Engine};
    let mut partial = false;
    let mut deltas = String::new();
    let mut completed = Vec::new();
    let mut first_partial = None;
    let streaming_started = std::time::Instant::now();
    for chunk in bytes.chunks(24000) {
        socket
            .send(Message::Text(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk)})
                    .to_string(),
            ))
            .await
            .unwrap();
        // Send at audio cadence and verify a recognition event before commit.
        let until = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
        while let Ok(Some(Ok(message))) = tokio::time::timeout_at(until, socket.next()).await {
            let event: Value = serde_json::from_str(&message.into_text().unwrap()).unwrap();
            assert_ne!(event["type"], "error", "{event}");
            assert_ne!(
                event["type"], "conversation.item.input_audio_transcription.failed",
                "{event}"
            );
            if event["type"] == "conversation.item.input_audio_transcription.partial" {
                partial = true;
                first_partial.get_or_insert(streaming_started.elapsed());
            }
            if event["type"] == "conversation.item.input_audio_transcription.delta" {
                deltas.push_str(event["delta"].as_str().unwrap());
            }
            if event["type"] == "conversation.item.input_audio_transcription.completed" {
                assert_eq!(event["is_final"], false);
                completed.push(event["transcript"].as_str().unwrap().to_owned());
            }
        }
    }
    assert!(
        !completed.is_empty(),
        "Continuous input did not automatically complete a rolling window"
    );
    assert!(partial, "No incremental recognition before commit");
    socket
        .send(Message::Text(
            json!({"type":"input_audio_buffer.commit"}).to_string(),
        ))
        .await
        .unwrap();
    let final_text = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            let event: Value =
                serde_json::from_str(&socket.next().await.unwrap().unwrap().into_text().unwrap())
                    .unwrap();
            assert_ne!(event["type"], "error", "{event}");
            assert_ne!(
                event["type"], "conversation.item.input_audio_transcription.failed",
                "{event}"
            );
            if event["type"] == "conversation.item.input_audio_transcription.delta" {
                deltas.push_str(event["delta"].as_str().unwrap());
            }
            if event["type"] == "conversation.item.input_audio_transcription.completed" {
                let text = event["transcript"].as_str().unwrap().to_owned();
                if event["is_final"] == true {
                    break text;
                }
                completed.push(text);
            }
        }
    })
    .await
    .unwrap();
    completed.push(final_text);
    let text = completed.join("");
    assert!(text.matches("交易").count() >= 8, "{text}");
    assert_eq!(deltas, text);
    println!(
        "{model}: realtime first partial {:?}, {} completed rolling items, {} utterances",
        first_partial.unwrap(),
        completed.len(),
        text.matches("交易").count()
    );
    socket
        .send(Message::Text(
            json!({"type":"input_audio_buffer.clear"}).to_string(),
        ))
        .await
        .unwrap();
    assert!(socket
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap()
        .contains("input_audio_buffer.cleared"));
    socket
        .send(Message::Text(
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(&bytes[..4*24000])})
                .to_string(),
        ))
        .await
        .unwrap();
    socket.close(None).await.unwrap();
    // A disconnected session must release its model lease, including pending GPU work.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    server.abort();
}
