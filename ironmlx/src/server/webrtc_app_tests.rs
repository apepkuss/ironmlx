//! Actual Chromium audio track -> Opus/RTP -> native inference -> DataChannel acceptance.
use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use serde_json::{json, Value};
use sha2::Digest;
use std::{
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};
use tower::ServiceExt;
struct NoCancel;
impl ironmlx_audio::SessionControl for NoCancel {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}
struct Browser(Child);
impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
async fn admin(app: &Router, path: &str, value: Value) -> Value {
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
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap()
}
#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires Qwen3-ASR checkpoint and native MLX"]
async fn webrtc_unconnected_calls_enforce_capacity_and_release_on_hangup_or_timeout() {
    use rtc::rtp_transceiver::{
        rtp_sender::RtpCodecKind, RTCRtpTransceiverDirection, RTCRtpTransceiverInit,
    };
    use webrtc::peer_connection::{
        MediaEngine, PeerConnection, PeerConnectionBuilder, PeerConnectionEventHandler,
    };
    struct OfferHandler;
    #[async_trait::async_trait]
    impl PeerConnectionEventHandler for OfferHandler {}
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        Some(1),
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let pool = manager.pool.clone();
    let temp = tempfile::tempdir().unwrap();
    let admin_app = app_router(
        manager,
        crate::server::voices::VoiceStore::open(temp.path().join("voices")).unwrap(),
    );
    let model = "mlx-community/Qwen3-ASR-0.6B-8bit";
    admin(
        &admin_app,
        "/admin/api/models/register",
        json!({"model":model,"model_dir":std::env::var("IRONMLX_ASR_SMALL_DIR").unwrap()}),
    )
    .await;
    // Exercise the standalone inference router as well as the App router above.
    let app = crate::server::engine::engine_pool_router().with_state(pool.clone());
    let mut media = MediaEngine::default();
    media.register_default_codecs().unwrap();
    let peer = PeerConnectionBuilder::new()
        .with_media_engine(media)
        .with_handler(Arc::new(OfferHandler))
        .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
        .build()
        .await
        .unwrap();
    peer.add_transceiver_from_kind(
        RtpCodecKind::Audio,
        Some(RTCRtpTransceiverInit {
            direction: RTCRtpTransceiverDirection::Sendonly,
            send_encodings: vec![Default::default()],
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    peer.create_data_channel("oai-events", None).await.unwrap();
    let offer = peer.create_offer(None).await.unwrap().sdp;
    let mut locations = Vec::new();
    for index in 0..9 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/realtime/calls?model={model}"))
                    .header("content-type", "application/sdp")
                    .body(Body::from(offer.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        if index == 8 {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        } else {
            let status = response.status();
            let location = response.headers().get("location").cloned();
            let bytes = to_bytes(response.into_body(), 128 * 1024).await.unwrap();
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            assert!(bytes.starts_with(b"v=0"));
            locations.push(location.unwrap().to_str().unwrap().to_owned());
        }
    }
    assert_eq!(crate::server::webrtc::active_call_count(), 8);
    for location in locations.iter().take(7) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("{location}/hangup"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    // The final peer never receives an answer or opens its DataChannel. Its
    // handshake timeout must release the model lease and call registry entry.
    tokio::time::timeout(Duration::from_secs(40), async {
        while crate::server::webrtc::active_call_count() != 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    for location in locations {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("{location}/hangup"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
    peer.close().await.unwrap();
    pool.unload_model(model).await.unwrap();
    assert!(!pool.is_model_loaded(model).await);
}
#[tokio::test]
#[serial_test::serial(mlx_metal)]
#[ignore = "requires Chrome, both ASR models, ForcedAligner, Nemotron and local speech WAV"]
async fn webrtc_chrome_media_transcription_vad_and_speaker_labels() {
    let root = std::env::var("MLX_DIR").unwrap();
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    let args = super::tests::serve_args();
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        Some(4),
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap();
    let pool = manager.pool.clone();
    let temp = tempfile::tempdir().unwrap();
    let app = app_router(
        manager,
        crate::server::voices::VoiceStore::open(temp.path().join("voices")).unwrap(),
    );
    let models = [
        ("mlx-community/Qwen3-ASR-0.6B-8bit", "IRONMLX_ASR_SMALL_DIR"),
        ("mlx-community/Qwen3-ASR-1.7B-8bit", "IRONMLX_ASR_LARGE_DIR"),
        (
            crate::server::transcriptions::FIXED_ALIGNER,
            "IRONMLX_ALIGNER_DIR",
        ),
        (
            ironmlx_audio::diarization::MODEL_ID,
            "IRONMLX_DIARIZATION_DIR",
        ),
    ];
    for (model, key) in models {
        admin(
            &app,
            "/admin/api/models/register",
            json!({"model":model,"model_dir":std::env::var(key).unwrap()}),
        )
        .await;
    }
    let pcm = ironmlx_audio::AudioIo::decode(
        &ironmlx_audio::io::NativeAudioIo,
        &std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO").unwrap()).unwrap(),
        &Default::default(),
    )
    .unwrap();
    let mut samples = vec![0.; 2 * 48000];
    samples.extend(
        ironmlx_audio::signal::resample_mono(
            &pcm.samples,
            pcm.format.sample_rate,
            48000,
            &NoCancel,
        )
        .unwrap(),
    );
    samples.extend(vec![0.; 2 * 48000]);
    let pcm = ironmlx_audio::PcmBuffer {
        format: ironmlx_audio::PcmFormat {
            sample_rate: 48000,
            channels: 1,
        },
        samples,
    };
    let mut bytes = Vec::new();
    ironmlx_audio::io::write_wav(&pcm, &mut bytes).unwrap();
    let (report_tx, mut report_rx) = tokio::sync::mpsc::channel::<Value>(1);
    let report_tx = Arc::new(report_tx);
    let ice: Value = serde_json::from_str(
        &std::env::var("IRONMLX_WEBRTC_TEST_ICE").unwrap_or_else(|_| "{}".into()),
    )
    .unwrap();
    let fixture = axum::Router::new()
        .route(
            "/__ice",
            get(move || {
                let ice = ice.clone();
                async move { Json(ice) }
            }),
        )
        .route(
            "/__audio.wav",
            get(move || {
                let bytes = bytes.clone();
                async move { ([(axum::http::header::CONTENT_TYPE, "audio/wav")], bytes) }
            }),
        )
        .route(
            "/__test",
            get(|| async {
                axum::response::Html(include_str!("../../tests/fixtures/webrtc/browser.html"))
            }),
        )
        .route(
            "/__client.js",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../../../examples/webrtc-transcription-client.js"),
                )
            }),
        )
        .route(
            "/__report",
            post(move |Json(report): Json<Value>| {
                let tx = report_tx.clone();
                async move {
                    let _ = tx.send(report).await;
                    StatusCode::OK
                }
            }),
        );
    let api = crate::server::security::lan_api_router(
        app.clone(),
        sha2::Sha256::digest(b"imx_webrtc_test").into(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        axum::serve(listener, fixture.merge(api)).await.unwrap();
    });
    let executable = std::env::var("IRONMLX_CHROME_PATH")
        .unwrap_or_else(|_| "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into());
    let log = std::fs::File::create(temp.path().join("chrome.log")).unwrap();
    let _browser = Browser(
        Command::new(executable)
            .args([
                "--headless=new",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ])
            .arg(format!(
                "--user-data-dir={}",
                temp.path().join("profile").display()
            ))
            .arg(format!("http://localhost:{port}/__test"))
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap(),
    );
    let report = tokio::time::timeout(Duration::from_secs(150), report_rx.recv()).await;
    let chrome_log = std::fs::read_to_string(temp.path().join("chrome.log")).unwrap_or_default();
    assert!(report.is_ok(), "Browser report timed out. {chrome_log}");
    let report = report.unwrap().expect("report receiver closed");
    if let Ok(path) = std::env::var("IRONMLX_WEBRTC_EVIDENCE") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    assert_eq!(report["ok"], true, "{report}\n{chrome_log}");
    let phases = report["phases"].as_array().unwrap();
    assert_eq!(phases.len(), 3);
    for phase in phases {
        assert!(
            phase["outbound"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["packetsSent"].as_u64().unwrap_or(0) > 20),
            "No actual browser audio RTP: {phase}"
        );
        assert!(phase["result"]["transcript"]
            .as_str()
            .unwrap()
            .contains("交易"));
        if phase["diarization"] == true {
            let segments = phase["result"]["segments"].as_array().unwrap();
            assert_eq!(
                segments
                    .iter()
                    .map(|s| s["text"].as_str().unwrap())
                    .collect::<String>(),
                phase["result"]["transcript"].as_str().unwrap()
            );
        } else {
            assert!(phase["eventTypes"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| !t.as_str().unwrap().contains("diariz")));
        }
        println!(
            "Chrome audio track: Qwen3-ASR-{}, diarization={}, VAD={}, transcript={}",
            phase["size"], phase["diarization"], phase["vad"], phase["result"]["transcript"]
        );
    }
    for (id, _) in models {
        tokio::time::timeout(Duration::from_secs(10), async {
            while pool.pending_requests(id).await.unwrap_or(0) != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        admin(&app, "/admin/api/models/unload", json!({"model":id})).await;
    }
    server.abort();
}
