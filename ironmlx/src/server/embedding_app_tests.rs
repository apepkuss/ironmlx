use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use base64::Engine as _;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn request(app: &Router, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let r = if let Some(body) = body {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    } else {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    };
    let response = app.clone().oneshot(r).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
#[ignore = "requires local BF16 and affine4 EmbeddingGemma 2 checkpoints"]
async fn embedding_real_app_lifecycle_and_api() {
    let dir = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        args.max_loaded_models,
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let voice_dir =
        std::env::temp_dir().join(format!("ironmlx-embedding-voices-{}", uuid::Uuid::new_v4()));
    let voices = crate::server::voices::VoiceStore::open(voice_dir.clone()).unwrap();
    let app = app_router(manager.clone(), voices);
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-lm/tests/fixtures/embedding_gemma2/reference.json"
    ))
    .unwrap();
    for (precision, variable) in [
        ("bf16", "IRONMLX_EMBEDDING_BF16_DIR"),
        ("4bit", "IRONMLX_EMBEDDING_AFFINE4_DIR"),
    ] {
        let path = std::env::var(variable).unwrap();
        let model = format!("mlx-community/embeddinggemma-2-{precision}");
        let load = json!({"model":model,"model_dir":path});
        let (status, result) =
            request(&app, "/admin/api/models/register", Some(load.clone())).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        let (status, result) = request(&app, "/admin/api/models/load", Some(load.clone())).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_eq!(result["loaded_models"][0]["runtime_kind"], "embedding");
        assert_eq!(result["loaded_models"][0]["supports_kv_cache"], false);
        assert_eq!(result["loaded_models"][0]["supports_vision"], true);
        assert_eq!(result["loaded_models"][0]["supports_audio"], true);
        let base = json!({"model":model,"input":fixture["inputs"]});
        let (status, full) = request(&app, "/v1/embeddings", Some(base.clone())).await;
        assert_eq!(status, StatusCode::OK, "{full}");
        assert_eq!(full["object"], "list");
        assert_eq!(full["model"], model);
        assert_eq!(full["usage"]["prompt_tokens"], 698);
        for (i, row) in full["data"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row["index"], i);
            let expected = fixture[precision]["embeddings"][i].as_array().unwrap();
            for (a, b) in row["embedding"].as_array().unwrap().iter().zip(expected) {
                assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 0.0001);
            }
        }
        let (status, health) = request(&app, "/healthz", None).await;
        assert_eq!(status, StatusCode::OK);
        let metrics = &health["models"][0]["embedding_metrics"];
        assert_eq!(metrics["completed_requests"], 1);
        assert_eq!(metrics["failed_requests"], 0);
        assert_eq!(metrics["recent_completed_requests"], 1);
        assert!(metrics["latency_ms_p50"].as_f64().unwrap() > 0.0);
        let input_rate = metrics["input_tokens_per_second"].as_f64().unwrap();
        let vector_rate = metrics["vectors_per_second"].as_f64().unwrap();
        assert!(
            (input_rate / vector_rate - 698.0 / full["data"].as_array().unwrap().len() as f64)
                .abs()
                < 1e-8
        );
        let (status, encoded) = request(
            &app,
            "/v1/embeddings",
            Some(json!({"model":model,"input":fixture["inputs"][0],"encoding_format":"base64"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{encoded}");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let expected: Vec<f32> =
            serde_json::from_value(full["data"][0]["embedding"].clone()).unwrap();
        assert_eq!(decoded, expected);
        for dimensions in [128, 256, 512] {
            let (status, reduced) = request(
                &app,
                "/v1/embeddings",
                Some(json!({"model":model,"input":fixture["inputs"][0],"dimensions":dimensions})),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{reduced}");
            let vector = reduced["data"][0]["embedding"].as_array().unwrap();
            assert_eq!(vector.len(), dimensions);
            assert!(
                (vector
                    .iter()
                    .map(|v| v.as_f64().unwrap().powi(2))
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 0.00001
            );
        }
        let image_fixture: Value = serde_json::from_str(include_str!(
            "../../../ironmlx-lm/tests/fixtures/embedding_gemma2/image-reference.json"
        ))
        .unwrap();
        let images = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ironmlx-lm/tests/fixtures/embedding_gemma2/images");
        let input: Vec<Value> = image_fixture["inputs"].as_array().unwrap().iter().map(|sample| {
            if sample.is_string() {return sample.clone();}
            let content: Vec<_> = sample["content"].as_array().unwrap().iter().map(|part| {
                if let Some(text) = part["text"].as_str() {json!({"type":"text","text":text})} else {
                    let bytes = std::fs::read(images.join(part["image"].as_str().unwrap())).unwrap();
                    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
                    json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{data}")}})
                }
            }).collect();
            json!({"content":content})
        }).collect();
        let base = json!({"model":model,"input":input});
        let (status, full) = request(&app, "/v1/embeddings", Some(base.clone())).await;
        assert_eq!(status, StatusCode::OK, "{full}");
        assert_eq!(full["usage"]["prompt_tokens"], 1636);
        for (i, row) in full["data"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row["index"], i);
            for (a, b) in row["embedding"].as_array().unwrap().iter().zip(
                image_fixture[precision]["embeddings"][i]
                    .as_array()
                    .unwrap(),
            ) {
                assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 0.0001);
            }
        }
        let (status, single) = request(
            &app,
            "/v1/embeddings",
            Some(
                json!({"model":model,"input":input[0],"encoding_format":"base64","dimensions":256}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{single}");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(single["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(vector.len(), 256);
        assert!((vector.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.00001);
        let part = input[0]["content"][0].clone();
        let (status, _) = request(
            &app,
            "/v1/embeddings",
            Some(json!({"model":model,"input":{"content":vec![part;9]}})),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        // A valid image larger than Axum's default 2 MiB limit remains supported.
        let mut large_png = std::fs::read(images.join("red-circle.png")).unwrap();
        large_png.resize(3 * 1024 * 1024, 0);
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(large_png)
        );
        let (status, large) = request(&app,"/v1/embeddings",Some(json!({"model":model,"input":{"content":[{"type":"image_url","image_url":{"url":url}}]}}))).await;
        assert_eq!(status, StatusCode::OK, "{large}");
        assert_eq!(large["data"][0]["embedding"], full["data"][0]["embedding"]);
        let (status,_) = request(&app,"/v1/embeddings",Some(json!({"model":model,"input":"x".repeat(super::super::security::MAX_REQUEST_BODY_BYTES)}))).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        for bad in [
            json!({"input":[]}),
            json!({"input":" "}),
            json!({"input":"hello","dimensions":129}),
            json!({"input":"hello","encoding_format":"hex"}),
            json!({"input":[1,2,3]}),
            json!({"input":vec!["x";33]}),
            json!({"input":{"content":[]}}),
            json!({"input":{"content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}}),
            json!({"input":{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,broken"}}]}}),
            json!({"input":{"content":[{"type":"audio","audio":"data:audio/wav;base64,AAAA"}]}}),
            json!({"input":{"content":[{"type":"input_audio","input_audio":{"data":"invalid","format":"wav"}}]}}),
            json!({"input":{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"aac"}}]}}),
            json!({"input":{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"flac","url":"https://example.com"}}]}}),
            json!({"input":{"content":[{"type":"image_url","image_url":{"url":"data:image/svg+xml;base64,AAAA"}}]}}),
            json!({"input":{"content":[{"type":"text","text":"<|image|>"}]}}),
            json!({"input":{"content":[{"type":"text","text":"a ".repeat(8192)}, input[0]["content"][0]]}}),
        ] {
            let mut bad = bad;
            bad["model"] = model.clone().into();
            let (status, _) = request(&app, "/v1/embeddings", Some(bad)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
        let (status, loaded) = request(&app, "/admin/api/models/loaded", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(loaded[0]["embedding_metrics"]["failed_requests"], 2);
        let audio_dir = images.parent().unwrap().join("audio");
        let audio_part = |format: &str| json!({"type":"input_audio","input_audio":{"format":format,"data":base64::engine::general_purpose::STANDARD.encode(std::fs::read(audio_dir.join(format!("sunny.{format}"))).unwrap())}});
        let audio_base = json!({"model":model,"input":{"content":[audio_part("wav")]}});
        let (status, audio_full) = request(&app, "/v1/embeddings", Some(audio_base.clone())).await;
        assert_eq!(status, StatusCode::OK, "{audio_full}");
        let audio_reference: Value = serde_json::from_str(include_str!(
            "../../../ironmlx-lm/tests/fixtures/embedding_gemma2/audio-reference.json"
        ))
        .unwrap();
        assert_eq!(
            audio_full["usage"]["prompt_tokens"],
            audio_reference[precision][0]["tokens"]
        );
        for (a, b) in audio_full["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .zip(
                audio_reference[precision][0]["embedding"]
                    .as_array()
                    .unwrap(),
            )
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 0.005);
        }
        for format in ["flac", "mp3"] {
            let (status,encoded)=request(&app,"/v1/embeddings",Some(json!({"model":model,"input":{"content":[{"type":"text","text":"Recorded speech: "},audio_part(format)]},"dimensions":256,"encoding_format":"base64"}))).await;
            assert_eq!(status, StatusCode::OK, "{encoded}");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded["data"][0]["embedding"].as_str().unwrap())
                .unwrap();
            assert_eq!(bytes.len(), 256 * 4);
        }
        let (status, _) = request(
            &app,
            "/v1/embeddings",
            Some(json!({"model":model,"input":{"content":vec![audio_part("wav");9]}})),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        let (a, b, c) = tokio::join!(
            request(&app, "/v1/embeddings", Some(audio_base.clone())),
            request(&app, "/v1/embeddings", Some(audio_base.clone())),
            request(&app, "/v1/embeddings", Some(audio_base.clone()))
        );
        for (status, response) in [a, b, c] {
            assert_eq!(status, StatusCode::OK, "{response}");
            assert_eq!(response, audio_full);
        }
        let (status, _) = request(
            &app,
            "/v1/chat/completions",
            Some(json!({"model":model,"messages":[{"role":"user","content":"hello"}]})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = request(
            &app,
            "/admin/api/models/default",
            Some(json!({"model":model})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (a, b, c) = tokio::join!(
            request(&app, "/v1/embeddings", Some(base.clone())),
            request(&app, "/v1/embeddings", Some(base.clone())),
            request(&app, "/v1/embeddings", Some(base.clone()))
        );
        for (status, response) in [a, b, c] {
            assert_eq!(status, StatusCode::OK, "{response}");
            assert_eq!(response, full);
        }
        let (status, _) = request(
            &app,
            "/admin/api/models/unload",
            Some(json!({"model":model})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, reloaded) = request(&app, "/v1/embeddings", Some(base)).await;
        assert_eq!(status, StatusCode::OK, "{reloaded}");
        assert_eq!(reloaded, full);
        let (_, health) = request(&app, "/healthz", None).await;
        assert_eq!(
            health["models"][0]["embedding_metrics"]["completed_requests"],
            1
        );
        assert_eq!(
            health["models"][0]["embedding_metrics"]["failed_requests"],
            0
        );
        let (status, audio_reloaded) = request(&app, "/v1/embeddings", Some(audio_base)).await;
        assert_eq!(status, StatusCode::OK, "{audio_reloaded}");
        assert_eq!(audio_reloaded, audio_full);

        // Cancelling an HTTP caller must not release a live audio GPU job's
        // lease. Unload drains it, then a new request can lazily load again.
        let wav = ironmlx_audio::AudioIo::encode_wav(
            &ironmlx_audio::io::NativeAudioIo,
            &ironmlx_audio::PcmBuffer {
                format: ironmlx_audio::PcmFormat {
                    sample_rate: 16000,
                    channels: 1,
                },
                samples: vec![0.0; 480000],
            },
        )
        .unwrap();
        let long = json!({"model":model,"input":{"content":[{"type":"input_audio","input_audio":{"format":"wav","data":base64::engine::general_purpose::STANDARD.encode(wav)}}]}});
        let caller_app = app.clone();
        let caller =
            tokio::spawn(async move { request(&caller_app, "/v1/embeddings", Some(long)).await });
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            let (_, loaded) = request(&app, "/admin/api/models/loaded", None).await;
            if loaded[0]["active_requests"].as_u64() == Some(1) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "audio job did not become active"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        let (status, draining) = request(
            &app,
            "/admin/api/models/unload",
            Some(json!({"model":model})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{draining}");
        assert_eq!(draining["status"], "unload_deferred", "{draining}");
        loop {
            let snapshots = manager.pool.model_snapshots().await;
            if snapshots.iter().any(|snapshot| {
                snapshot.id == model
                    && snapshot.state
                        == ironmlx_runtime::core::engine_pool::EngineRuntimeState::Unloaded
            }) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "cancelled audio job did not drain"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let (status, after_cancel) = request(
            &app,
            "/v1/embeddings",
            Some(json!({"model":model,"input":{"content":[audio_part("wav")]}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{after_cancel}");
        assert_eq!(after_cancel, audio_full);
        let (status, _) = request(
            &app,
            "/admin/api/models/unload",
            Some(json!({"model":model})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    drop(app);
    drop(manager);
    std::fs::remove_dir_all(voice_dir).unwrap();
}
