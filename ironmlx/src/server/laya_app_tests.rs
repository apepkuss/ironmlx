use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use ironmlx_runtime::core::engine_pool::EngineVariant;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = ironmlx_decision::contract::MULTILINGUAL_MODEL_ID;

fn manager() -> ModelManager {
    let args = super::tests::serve_args();
    ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        args.max_loaded_models,
        SchedulerResolutionOptions::from(&args),
    )
    .unwrap()
}

fn test_app_router(manager: ModelManager) -> (Router, std::path::PathBuf) {
    let voice_dir = std::env::temp_dir().join(format!(
        "ironmlx-laya-route-voices-{}",
        uuid::Uuid::new_v4()
    ));
    let voices = crate::server::voices::VoiceStore::open(voice_dir.clone()).unwrap();
    (app_router(manager, voices), voice_dir)
}

async fn request(router: &Router, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let req = if let Some(body) = body {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    } else {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    };
    let response = router.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())),
    )
}

#[tokio::test]
async fn laya_app_registration_and_endpoint_boundaries() {
    let root = std::env::temp_dir().join(format!("ironmlx-laya-route-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("mlx_config.json"),
        json!({
            "format":"laya-mlx", "format_version":1, "repository":MODEL
        })
        .to_string(),
    )
    .unwrap();
    let (app, voice_dir) = test_app_router(manager());
    let (status, registered) = request(
        &app,
        "/admin/api/models/register",
        Some(json!({"model":MODEL,"model_dir":root})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{registered}");
    let (_, list) = request(&app, "/v1/models", None).await;
    assert_eq!(list["models"][0]["name"], MODEL);
    assert_eq!(list["data"][0]["id"], MODEL);
    assert_eq!(list["data"][0]["state"], "unloaded");
    let (status, _) = request(
        &app,
        "/v1/chat/completions",
        Some(json!({"model":MODEL,"messages":[{"role":"user","content":"hello"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = request(
        &app,
        "/v1/systemone",
        Some(json!({"model":MODEL,"state":"x","questions":{}})),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _) = request(
        &app,
        "/admin/api/models/register",
        Some(json!({"model":MODEL,"model_dir":root,"temperature":0.7})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for decision in [
        json!({"batch_size":0}),
        json!({"batch_size":257}),
        json!({"dtype":"int8"}),
        json!({"device":"cuda"}),
        json!({"pad_to_multiple":0}),
        json!({"pad_to_multiple":1025}),
    ] {
        let (status, _) = request(
            &app,
            "/admin/api/models/register",
            Some(json!({"model":MODEL,"model_dir":root,"decision":decision})),
        )
        .await;
        assert!(status.is_client_error());
    }
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(voice_dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires LAYA_MODEL_DIR and LAYA_METALLIB for real App lifecycle inference"]
#[serial_test::serial(mlx_metal)]
async fn laya_real_app_load_predict_drain_unload_reload() {
    let path = std::env::var("LAYA_MODEL_DIR").unwrap();
    mlx::metal::set_metallib_path(&std::env::var("LAYA_METALLIB").unwrap()).unwrap();
    let manager = manager();
    let (app, voice_dir) = test_app_router(manager.clone());
    let load = json!({"model":MODEL,"model_dir":path,"set_default":true});
    let (status, body) = request(&app, "/admin/api/models/load", Some(load.clone())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["loaded_models"][0]["runtime_kind"], "decision");
    let input: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-decision/tests/fixtures/laya-reference/0.request.json"
    ))
    .unwrap();
    let expected: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-decision/tests/fixtures/laya-reference/0.reference.json"
    ))
    .unwrap();
    let (status, output) = request(&app, "/v1/systemone", Some(input.clone())).await;
    assert_eq!(status, StatusCode::OK, "{output}");
    assert_eq!(output["usage"], expected["usage"]);
    assert_eq!(
        output["answers"]["department"]["choice"],
        expected["answers"]["department"]["choice"]
    );
    assert!(
        (output["answers"]["refund"]["noul"].as_f64().unwrap()
            - expected["answers"]["refund"]["noul"].as_f64().unwrap())
        .abs()
            < 0.001
    );
    let (a, b, c) = tokio::join!(
        request(&app, "/v1/systemone", Some(input.clone())),
        request(&app, "/v1/systemone", Some(input.clone())),
        request(&app, "/v1/systemone", Some(input.clone())),
    );
    for (status, body) in [a, b, c] {
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, output);
    }
    let (status, loaded) = request(&app, "/admin/api/models/loaded", None).await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    let metrics = &loaded[0]["decision_metrics"];
    assert_eq!(metrics["completed_requests"], 4);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["recent_completed_requests"], 4);
    assert!(metrics["latency_ms_p50"].as_f64().unwrap() > 0.0);
    assert!(metrics["input_tokens_per_second"].as_f64().unwrap() > 0.0);
    assert!(metrics["questions_per_second"].as_f64().unwrap() > 0.0);
    assert!(metrics["last_request_unix_ms"].as_u64().unwrap() > 0);
    let (status, body) = request(&app, "/admin/api/models/pin", Some(json!({"model":MODEL}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["loaded_models"][0]["pinned"], true);
    let (_, lease) = manager.pool.resolve_engine(Some(MODEL)).await.unwrap();
    let (status, body) = request(
        &app,
        "/admin/api/models/unload",
        Some(json!({"model":MODEL})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "unload_deferred");
    let EngineVariant::Decision(runtime) = lease.engine() else {
        panic!("decision worker");
    };
    let runtime = runtime.clone();
    runtime
        .predict(serde_json::from_value(input.clone()).unwrap(), lease)
        .await
        .unwrap();
    drop(runtime);
    for _ in 0..100 {
        if manager.pool.loaded_model_infos().await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(manager.pool.loaded_model_infos().await.is_empty());
    let (status, body) = request(&app, "/admin/api/models/load", Some(load)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = request(&app, "/v1/systemone", Some(input)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, output);
    let settings = json!({"dtype":"float32","batch_size":2,"cache_prompts":true,
        "device":"cpu","compile":true,"pad_to_multiple":16});
    let (status, body) = request(
        &app,
        "/admin/api/models/load",
        Some(json!({
            "model":MODEL,"model_dir":path,"reload_when_idle":true,"decision":settings
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["loaded_models"][0]["decision"], settings);
    let fp32: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-decision/tests/fixtures/laya-reference/0.cpu.compiled.float32.reference.json"
    ))
    .unwrap();
    let input: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-decision/tests/fixtures/laya-reference/0.request.json"
    ))
    .unwrap();
    let (status, actual) = request(&app, "/v1/systemone", Some(input)).await;
    assert_eq!(status, StatusCode::OK, "{actual}");
    assert!(
        (actual["answers"]["urgency"]["score"].as_f64().unwrap()
            - fp32["answers"]["urgency"]["score"].as_f64().unwrap())
        .abs()
            < 0.001
    );
    request(
        &app,
        "/admin/api/models/unload",
        Some(json!({"model":MODEL})),
    )
    .await;
    assert!(manager.pool.loaded_model_infos().await.is_empty());
    std::fs::remove_dir_all(voice_dir).unwrap();
}

#[tokio::test]
async fn laya_app_lan_authentication_precedes_inference() {
    use sha2::{Digest, Sha256};
    let (router, voice_dir) = test_app_router(manager());
    let app = crate::server::security::lan_api_router(router, Sha256::digest(b"test-key").into());
    for (key, expected) in [
        (None, StatusCode::UNAUTHORIZED),
        (Some("wrong"), StatusCode::UNAUTHORIZED),
        (Some("test-key"), StatusCode::UNPROCESSABLE_ENTITY),
    ] {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/systemone")
            .header("content-type", "application/json");
        if let Some(key) = key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::from("{}")).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert!(body["detail"].is_string());
    }
    std::fs::remove_dir_all(voice_dir).unwrap();
}
