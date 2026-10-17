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
    audio_request(app, "/v1/audio/transcriptions", model, audio, extra).await
}
async fn audio_request(
    app: &Router,
    path: &str,
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
                .uri(path)
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

#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires local ASR checkpoints and reference Chinese WAV"]
async fn qwen3_asr_server_vad_queues_speech_and_ignores_silence() {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};
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
    let app = app_router(manager, voices);
    let audio = ironmlx_audio::AudioIo::decode(
        &ironmlx_audio::io::NativeAudioIo,
        &std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap(),
        &Default::default(),
    )
    .unwrap();
    let samples = ironmlx_audio::signal::resample_mono(
        &audio.samples,
        audio.format.sample_rate,
        24000,
        &NoCancel,
    )
    .unwrap();
    let speech: Vec<u8> = samples
        .iter()
        .flat_map(|x| ((x.clamp(-1., 1.) * 32767.) as i16).to_le_bytes())
        .collect();
    for (size, var) in [
        ("0.6B", "IRONMLX_ASR_SMALL_DIR"),
        ("1.7B", "IRONMLX_ASR_LARGE_DIR"),
    ] {
        let model = format!("mlx-community/Qwen3-ASR-{size}-8bit");
        json_request(
            &app,
            "/admin/api/models/register",
            json!({"model":model,"model_dir":std::env::var(var).unwrap()}),
        )
        .await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let router = app.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let (mut ws, _) = connect_async(format!(
            "ws://127.0.0.1:{port}/v1/realtime?intent=transcription&model={model}"
        ))
        .await
        .unwrap();
        ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(json!({"type":"session.update","session":{"turn_detection":{"type":"server_vad"},"input_audio_transcription":{"language":"zh"}}}).to_string())).await.unwrap();
        let updated: Value =
            serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
        assert_eq!(updated["type"], "session.updated");
        // Pure silence must not allocate an ASR turn or trigger inference.
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;48000])})
                .to_string(),
        ))
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(150), ws.next())
                .await
                .is_err()
        );
        let mut input = speech.clone();
        input.extend(vec![0u8; 48000]);
        let repeated = input.repeat(2);
        for chunk in repeated.chunks(9600) {
            ws.send(Message::Text(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk)})
                    .to_string(),
            ))
            .await
            .unwrap();
        }
        let mut starts = Vec::new();
        let mut stops = Vec::new();
        let mut commits = Vec::new();
        let mut finals = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(90), async {
            while finals.len() < 2 {
                let e: Value =
                    serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap())
                        .unwrap();
                assert_ne!(e["type"], "error", "{e}");
                assert_ne!(
                    e["type"], "conversation.item.input_audio_transcription.failed",
                    "{e}"
                );
                match e["type"].as_str().unwrap() {
                    "input_audio_buffer.speech_started" => starts.push(e),
                    "input_audio_buffer.speech_stopped" => stops.push(e),
                    "input_audio_buffer.committed" => commits.push(e),
                    "conversation.item.input_audio_transcription.completed" => finals.push(e),
                    _ => {}
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(starts.len(), 2);
        assert_eq!(stops.len(), 2);
        assert_eq!(commits.len(), 2);
        for i in 0..2 {
            assert_eq!(starts[i]["item_id"], stops[i]["item_id"]);
            assert_eq!(starts[i]["item_id"], commits[i]["item_id"]);
            assert_eq!(starts[i]["item_id"], finals[i]["item_id"]);
            assert_eq!(finals[i]["is_final"], true);
            assert!(
                finals[i]["transcript"].as_str().unwrap().contains("交易"),
                "{}",
                finals[i]
            );
            assert!(
                stops[i]["audio_end_ms"].as_u64().unwrap()
                    > starts[i]["audio_start_ms"].as_u64().unwrap()
            );
        }
        assert_eq!(commits[1]["previous_item_id"], commits[0]["item_id"]);
        println!("{model}: {finals:?}");
        // Clearing resets VAD state; silence after clear remains silent.
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.clear"}).to_string(),
        ))
        .await
        .unwrap();
        let cleared: Value =
            serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
        assert_eq!(cleared["type"], "input_audio_buffer.cleared");
        // Manual commit remains available with VAD enabled; flush the short last frame.
        ws.send(Message::Text(json!({"type":"session.update","session":{"turn_detection":{"type":"server_vad","silence_duration_ms":3000}}}).to_string())).await.unwrap();
        let updated: Value =
            serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
        assert_eq!(updated["type"], "session.updated");
        for chunk in speech.chunks(9600) {
            ws.send(Message::Text(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk)})
                    .to_string(),
            ))
            .await
            .unwrap();
        }
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.commit"}).to_string(),
        ))
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(60), async {
            loop {
                let e: Value =
                    serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap())
                        .unwrap();
                assert_ne!(e["type"], "error", "{e}");
                assert_ne!(
                    e["type"], "conversation.item.input_audio_transcription.failed",
                    "{e}"
                );
                if e["type"] == "conversation.item.input_audio_transcription.completed" {
                    assert_eq!(e["is_final"], true);
                    assert!(e["transcript"].as_str().unwrap().contains("交易"), "{e}");
                    break;
                }
            }
        })
        .await
        .unwrap();
        // Disabling VAD restores the existing explicit-commit transport.
        ws.send(Message::Text(
            json!({"type":"session.update","session":{"turn_detection":null}}).to_string(),
        ))
        .await
        .unwrap();
        let updated: Value =
            serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
        assert_eq!(updated["type"], "session.updated");
        assert!(updated["session"]["audio"]["input"]["turn_detection"].is_null());
        for chunk in speech.chunks(9600) {
            ws.send(Message::Text(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk)})
                    .to_string(),
            ))
            .await
            .unwrap();
        }
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.commit"}).to_string(),
        ))
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(60), async {
            loop {
                let e: Value =
                    serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap())
                        .unwrap();
                assert_ne!(e["type"], "error", "{e}");
                assert_ne!(
                    e["type"], "conversation.item.input_audio_transcription.failed",
                    "{e}"
                );
                if e["type"] == "conversation.item.input_audio_transcription.completed" {
                    assert_eq!(e["is_final"], true);
                    assert!(e["transcript"].as_str().unwrap().contains("交易"), "{e}");
                    break;
                }
            }
        })
        .await
        .unwrap();
        // Clear cancels in-flight and queued turns; no stale completion may follow its acknowledgement.
        ws.send(Message::Text(
            json!({"type":"session.update","session":{"turn_detection":{"type":"server_vad"}}})
                .to_string(),
        ))
        .await
        .unwrap();
        let updated: Value =
            serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
        assert_eq!(updated["type"], "session.updated");
        for chunk in repeated.chunks(9600) {
            ws.send(Message::Text(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk)})
                    .to_string(),
            ))
            .await
            .unwrap();
        }
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.clear"}).to_string(),
        ))
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            loop {
                let e: Value =
                    serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap())
                        .unwrap();
                assert_ne!(e["type"], "error", "{e}");
                if e["type"] == "input_audio_buffer.cleared" {
                    break;
                }
            }
        })
        .await
        .unwrap();
        ws.send(Message::Text(
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;48000])})
                .to_string(),
        ))
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), ws.next())
                .await
                .is_err()
        );
        ws.close(None).await.unwrap();
        server.abort();
        let _ = server.await;
        json_request(&app, "/admin/api/models/unload", json!({"model":model})).await;
    }
}

#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires both ASR models, ForcedAligner and reference WAV"]
async fn qwen3_alignment_and_timestamp_workflow() {
    use ironmlx_audio::AudioIo;
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        Some(2),
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let voices = crate::server::voices::VoiceStore::open(dir.path().to_owned()).unwrap();
    let app = app_router(manager, voices);
    let audio = std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap();
    let aligner = crate::server::transcriptions::FIXED_ALIGNER;
    json_request(
        &app,
        "/admin/api/models/register",
        json!({"model":aligner,"model_dir":std::env::var("IRONMLX_ALIGNER_DIR").unwrap()}),
    )
    .await;
    let (status, bytes) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &audio,
        &[("text", "甚至出现交易几乎停滞的情况。"), ("language", "zh")],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let aligned: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(aligned["words"].as_array().unwrap().len(), 13);
    assert_eq!(aligned["words"][0]["start"], 0.4);
    for extra in [
        vec![("text", "hello")],
        vec![("text", "hello"), ("language", "xx")],
        vec![("text", "hello"), ("language", "en"), ("stream", "true")],
    ] {
        let (status, _) =
            audio_request(&app, "/v1/audio/alignments", aligner, &audio, &extra).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    let (status, _) = transcribe(&app, aligner, &audio, &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for (size, var) in [
        ("0.6B", "IRONMLX_ASR_SMALL_DIR"),
        ("1.7B", "IRONMLX_ASR_LARGE_DIR"),
    ] {
        let model = format!("mlx-community/Qwen3-ASR-{size}-8bit");
        json_request(
            &app,
            "/admin/api/models/register",
            json!({"model":model,"model_dir":std::env::var(var).unwrap()}),
        )
        .await;
        let (status, bytes) = transcribe(
            &app,
            &model,
            &audio,
            &[
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let result: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(result["text"].as_str().unwrap().contains("交易"));
        assert_eq!(result["words"].as_array().unwrap().len(), 13);
        assert!(!result["segments"].as_array().unwrap().is_empty());
        println!("{size}: {result}");
        let (status, _) = transcribe(
            &app,
            &model,
            &audio,
            &[("timestamp_granularities[]", "word")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = transcribe(
            &app,
            &model,
            &audio,
            &[("response_format", "verbose_json"), ("stream", "true")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        if size == "0.6B" {
            let io = ironmlx_audio::io::NativeAudioIo;
            let mut pcm = io.decode(&audio, &Default::default()).unwrap();
            pcm.samples = pcm.samples.repeat(8);
            let long = io.encode_wav(&pcm).unwrap();
            let (status, bytes) = transcribe(
                &app,
                &model,
                &long,
                &[
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                ],
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            let result: Value = serde_json::from_slice(&bytes).unwrap();
            let words = result["words"].as_array().unwrap();
            assert!(words.last().unwrap()["end"].as_f64().unwrap() > 30.);
            assert!(words
                .windows(2)
                .all(|w| w[0]["end"].as_f64().unwrap() <= w[1]["start"].as_f64().unwrap() + 0.001));
            println!(
                "Long recording: {} aligned units, final time {}",
                words.len(),
                words.last().unwrap()["end"]
            );
        }
        json_request(&app, "/admin/api/models/unload", json!({"model":model})).await;
    }
    json_request(&app, "/admin/api/models/unload", json!({"model":aligner})).await;
}

#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires local ASR and ForcedAligner checkpoints"]
async fn qwen3_timestamp_pool_evicts_between_workflow_steps() {
    use ironmlx_runtime::core::engine_pool::{
        EngineLoadPolicy, EngineModelManifest, EnginePoolConfig,
    };
    use ironmlx_runtime::core::model_management::build_engine_model_config_for_pool;
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let mut models = Vec::new();
    for (id, var) in [
        ("mlx-community/Qwen3-ASR-0.6B-8bit", "IRONMLX_ASR_SMALL_DIR"),
        (
            crate::server::transcriptions::FIXED_ALIGNER,
            "IRONMLX_ALIGNER_DIR",
        ),
    ] {
        let manifest = EngineModelManifest {
            id: id.into(),
            path: std::env::var(var).unwrap().into(),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            audio: None,
            decision: None,
            scheduler_profile: None,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: None,
        };
        models.push(
            build_engine_model_config_for_pool(
                &SchedulerResolutionOptions::from(&args),
                manifest,
                None,
                "",
            )
            .unwrap(),
        );
    }
    let pool = EnginePoolState::new(
        EnginePoolConfig {
            models,
            default_model: None,
            max_loaded_models: Some(1),
        },
        crate::cli::serve::engine_runtime_config(&args)
            .unwrap()
            .options,
    )
    .await
    .unwrap();
    let app = crate::server::engine::engine_pool_router().with_state(pool.clone());
    let audio = std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap();
    let (status, bytes) = transcribe(
        &app,
        "mlx-community/Qwen3-ASR-0.6B-8bit",
        &audio,
        &[
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let result: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(result["words"].as_array().unwrap().len(), 13);
    assert!(
        !pool
            .is_model_loaded("mlx-community/Qwen3-ASR-0.6B-8bit")
            .await
    );
    assert!(
        pool.is_model_loaded(crate::server::transcriptions::FIXED_ALIGNER)
            .await
    );
    // Exercise the standalone route on the pool surface too.
    let (status, _) = audio_request(
        &app,
        "/v1/audio/alignments",
        crate::server::transcriptions::FIXED_ALIGNER,
        &audio,
        &[("text", "甚至出现交易几乎停滞的情况。"), ("language", "zh")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires local ASR, ForcedAligner and reference WAV"]
async fn qwen3_alignment_rejects_limits_and_cancels_cleanly() {
    use ironmlx_audio::{AudioIo, PcmBuffer, PcmFormat};
    use std::time::Duration;
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        Some(2),
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let voices = crate::server::voices::VoiceStore::open(dir.path().to_owned()).unwrap();
    let app = app_router(manager.clone(), voices);
    let audio = std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap();
    let asr = "mlx-community/Qwen3-ASR-0.6B-8bit";
    let aligner = crate::server::transcriptions::FIXED_ALIGNER;
    json_request(
        &app,
        "/admin/api/models/register",
        json!({"model":asr,"model_dir":std::env::var("IRONMLX_ASR_SMALL_DIR").unwrap()}),
    )
    .await;
    let (status, bytes) =
        transcribe(&app, asr, &audio, &[("response_format", "verbose_json")]).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(
        !manager.pool.is_model_loaded(asr).await,
        "Missing dependency must fail before loading ASR"
    );
    let (status, bytes) = transcribe(&app, asr, &audio, &[]).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(serde_json::from_slice::<Value>(&bytes).unwrap()["text"]
        .as_str()
        .unwrap()
        .contains("交易"));
    json_request(
        &app,
        "/admin/api/models/register",
        json!({"model":aligner,"model_dir":std::env::var("IRONMLX_ALIGNER_DIR").unwrap()}),
    )
    .await;
    for extra in [
        vec![("text", "!!!"), ("language", "zh")],
        vec![("text", "hello"), ("language", "en"), ("temperature", "0")],
        vec![("text", "hello"), ("text", "duplicate"), ("language", "en")],
    ] {
        let (status, bytes) =
            audio_request(&app, "/v1/audio/alignments", aligner, &audio, &extra).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
    }
    let oversized_text = "啊".repeat(2049);
    let (status, _) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &audio,
        &[("text", &oversized_text), ("language", "zh")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let io = ironmlx_audio::io::NativeAudioIo;
    // Valid standalone inputs must not inherit the TTS reference resampler's 60s cap.
    let mut extended = io.decode(&audio, &Default::default()).unwrap();
    assert_eq!(
        extended.format,
        PcmFormat {
            sample_rate: 16000,
            channels: 1
        }
    );
    extended.samples.resize(61 * 16000, 0.);
    let extended = io.encode_wav(&extended).unwrap();
    let (status, bytes) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &extended,
        &[("text", "甚至出现交易几乎停滞的情况。"), ("language", "zh")],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let result: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(result["duration"], 61.);
    assert_eq!(result["words"].as_array().unwrap().len(), 13);
    let mut pcm = PcmBuffer {
        format: PcmFormat {
            sample_rate: 16000,
            channels: 1,
        },
        samples: vec![0.; 301 * 16000],
    };
    let long = io.encode_wav(&pcm).unwrap();
    let (status, _) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &long,
        &[("text", "你好"), ("language", "zh")],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "Audio duration limit was not enforced"
    );
    pcm.samples.truncate(300 * 16000);
    let long = io.encode_wav(&pcm).unwrap();
    let context_text = "啊".repeat(2048);
    let (status, bytes) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &long,
        &[("text", &context_text), ("language", "zh")],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(
        String::from_utf8_lossy(&bytes).contains("8192"),
        "Expected combined audio/text context rejection"
    );
    let request_app = app.clone();
    let pending = tokio::spawn(async move {
        audio_request(
            &request_app,
            "/v1/audio/alignments",
            aligner,
            &long,
            &[("text", "你好"), ("language", "zh")],
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while manager.pool.pending_requests(aligner).await.unwrap_or(0) == 0 {
            assert!(
                !pending.is_finished(),
                "Cancellation request unexpectedly finished before worker admission"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    pending.abort();
    assert!(pending.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(10), async {
        while manager.pool.pending_requests(aligner).await.unwrap_or(0) != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    // Cancellation must release admission resources and leave the worker usable.
    let (status, bytes) = audio_request(
        &app,
        "/v1/audio/alignments",
        aligner,
        &audio,
        &[("text", "甚至出现交易几乎停滞的情况。"), ("language", "zh")],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let (status, bytes) =
        transcribe(&app, asr, &audio, &[("response_format", "verbose_json")]).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let result: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(!result["segments"].as_array().unwrap().is_empty());
    assert!(
        result.get("words").is_none(),
        "Default granularity is segment only"
    );
}
