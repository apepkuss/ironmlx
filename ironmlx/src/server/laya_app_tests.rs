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
    let manifest = serde_json::from_value(json!({"id":MODEL,"path":root})).unwrap();
    let config = ironmlx_runtime::core::model_management::build_engine_model_config_for_pool(
        &SchedulerResolutionOptions::from(&super::tests::serve_args()),
        manifest,
        None,
        "test",
    )
    .expect("LAYA manifest only requires mlx_config.json");
    assert_eq!(config.capabilities.runtime_kind, "decision");
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

async fn post_raw(router: &Router, path: &str, body: &str) -> (StatusCode, Option<String>, Value) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let retry = response
        .headers()
        .get("retry-after")
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, retry, serde_json::from_slice(&bytes).unwrap())
}

fn decisions_request() -> Value {
    json!({"model": MODEL, "input": "{\"message\": \"发票被重复扣款，请退款。\"}", "questions": [
        {"type": "score", "name": "urgency", "instructions": "How urgent is this?",
         "levels": [{"label": "can wait"}, {"label": "soon"}, {"label": "today"}]},
        {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
         "choices": [{"value": "billing", "description": "refunds"},
                     {"value": "technical", "description": "bugs"}]},
        {"type": "predicate", "name": "refund", "instructions": "Is a refund requested?"}
    ]})
}

#[tokio::test]
async fn decisions_routes_render_openai_errors_on_app_and_engine_pool() {
    let root =
        std::env::temp_dir().join(format!("ironmlx-laya-decisions-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("mlx_config.json"),
        json!({"format":"laya-mlx", "format_version":1, "repository":MODEL}).to_string(),
    )
    .unwrap();
    let manager = manager();
    let (app, voice_dir) = test_app_router(manager.clone());
    let (status, _, body) = post_raw(&app, "/v1/decisions", &decisions_request().to_string()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    let (status, registered) = request(
        &app,
        "/admin/api/models/register",
        Some(json!({"model":MODEL,"model_dir":root})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{registered}");
    let routers = [
        app,
        crate::server::engine::engine_pool_router().with_state(manager.pool.clone()),
    ];
    for router in &routers {
        let cases = [
            ("{", "invalid_json", None),
            (
                r#"{"model":"aac6fef/laya-multilingual-mlx","input":"x","questions":[]}"#,
                "invalid_value",
                Some("questions"),
            ),
            (
                r#"{"model":"aac6fef/laya-multilingual-mlx","input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],"questions":[{"type":"predicate","instructions":"?"}]}"#,
                "unsupported_value",
                Some("input[0].content[0].type"),
            ),
            (
                r#"{"model":"missing/model","input":"x","questions":[{"type":"predicate","instructions":"?"}]}"#,
                "model_not_found",
                Some("model"),
            ),
        ];
        for (body, code, param) in cases {
            let (status, retry, json) = post_raw(router, "/v1/decisions", body).await;
            assert!(status.is_client_error(), "{body}: {status}");
            assert!(retry.is_none());
            assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
            assert_eq!(json["error"]["code"], code, "{json}");
            if let Some(param) = param {
                assert_eq!(json["error"]["param"], param);
            }
        }
        // The registered directory has no weights, so the on-demand load fails.
        let (status, retry, json) =
            post_raw(router, "/v1/decisions", &decisions_request().to_string()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{json}");
        assert_eq!(retry.as_deref(), Some("5"));
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["code"], "model_load_failed");
    }
    // System One keeps its own error contract on the same pool.
    let (status, _, json) = post_raw(
        &routers[0],
        "/v1/systemone",
        &json!({"model":MODEL,"state":"x","questions":{}}).to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(json["detail"].is_string());
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(voice_dir).unwrap();
}

#[tokio::test]
async fn decisions_lan_authentication_uses_openai_errors() {
    use sha2::{Digest, Sha256};
    let (router, voice_dir) = test_app_router(manager());
    let app = crate::server::security::lan_api_router(router, Sha256::digest(b"test-key").into());
    for (key, expected, code) in [
        (None, StatusCode::UNAUTHORIZED, "auth_invalid"),
        (Some("local"), StatusCode::UNAUTHORIZED, "auth_invalid"),
        (
            Some("test-key"),
            StatusCode::BAD_REQUEST,
            "missing_required_parameter",
        ),
    ] {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/decisions")
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
        assert_eq!(body["error"]["code"], code, "{body}");
    }
    std::fs::remove_dir_all(voice_dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires LAYA_MODEL_DIR and LAYA_METALLIB for real App lifecycle inference"]
#[serial_test::serial(mlx_metal)]
async fn laya_real_decisions_match_system_one_and_share_lifecycle() {
    let path = std::env::var("LAYA_MODEL_DIR").unwrap();
    mlx::metal::set_metallib_path(&std::env::var("LAYA_METALLIB").unwrap()).unwrap();
    let manager = manager();
    let (app, voice_dir) = test_app_router(manager.clone());
    let pool_router = crate::server::engine::engine_pool_router().with_state(manager.pool.clone());
    let (status, body) = request(
        &app,
        "/admin/api/models/register",
        Some(json!({"model":MODEL,"model_dir":path})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The first Decisions request loads the registered model on demand.
    let (status, decision) = request(&app, "/v1/decisions", Some(decisions_request())).await;
    assert_eq!(status, StatusCode::OK, "{decision}");
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../ironmlx-decision/tests/fixtures/laya-reference/0.request.json"
    ))
    .unwrap();
    let (status, system_one) = request(&app, "/v1/systemone", Some(fixture.clone())).await;
    assert_eq!(status, StatusCode::OK, "{system_one}");
    let s1 = &system_one["answers"];
    assert_eq!(decision["model"], MODEL);
    assert_eq!(
        decision["usage"]["input_tokens"],
        system_one["usage"]["input_tokens"]
    );
    assert_eq!(
        decision["usage"]["total_tokens"],
        system_one["usage"]["input_tokens"]
    );
    assert_eq!(decision["usage"]["output_tokens"], 0);
    assert_eq!(
        decision["usage"]["input_tokens_details"]["cached_tokens"],
        0
    );
    assert_eq!(
        decision["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
    let answers = decision["answers"].as_array().unwrap();
    assert_eq!(answers[0]["name"], "urgency");
    assert_eq!(answers[0]["score"], s1["urgency"]["score"]);
    assert_eq!(answers[0]["confidence"], s1["urgency"]["confidence"]);
    for index in 0..3 {
        assert_eq!(
            answers[0]["probabilities"][index]["probability"],
            s1["urgency"]["probabilities"][index.to_string()]
        );
    }
    assert_eq!(answers[1]["choice"], s1["department"]["choice"]);
    assert_eq!(answers[1]["confidence"], s1["department"]["confidence"]);
    assert_eq!(
        answers[1]["probabilities"][0]["probability"],
        s1["department"]["probabilities"]["billing"]
    );
    assert_eq!(answers[2]["probability"], s1["refund"]["noul"]);

    // Concurrent requests on both APIs and both routers share one worker.
    let (a, b, c, d, e) = tokio::join!(
        request(&app, "/v1/decisions", Some(decisions_request())),
        request(&pool_router, "/v1/decisions", Some(decisions_request())),
        request(&app, "/v1/decisions", Some(decisions_request())),
        request(&app, "/v1/systemone", Some(fixture.clone())),
        request(&pool_router, "/v1/systemone", Some(fixture.clone())),
    );
    for (status, body) in [a, b, c] {
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, decision);
    }
    for (status, body) in [d, e] {
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, system_one);
    }
    let (status, loaded) = request(&app, "/admin/api/models/loaded", None).await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    let metrics = &loaded[0]["decision_metrics"];
    assert_eq!(metrics["completed_requests"], 7);
    assert_eq!(metrics["failed_requests"], 0);
    assert!(metrics["input_tokens_per_second"].as_f64().unwrap() > 0.0);
    assert!(metrics["questions_per_second"].as_f64().unwrap() > 0.0);
    let (_, models) = request(&app, "/v1/models", None).await;
    assert_eq!(models["data"][0]["request_count"], 7, "{models}");

    // An in-flight lease defers unload until Decisions work finishes.
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
    let prepared = ironmlx_decision::openai::parse_request(&decisions_request()).unwrap();
    let scores = runtime
        .score(prepared.scoring.clone(), lease)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(prepared.response(&scores).unwrap()).unwrap(),
        decision
    );
    drop(runtime);
    for _ in 0..100 {
        if manager.pool.loaded_model_infos().await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(manager.pool.loaded_model_infos().await.is_empty());

    // A later request reloads on demand with identical results.
    let (status, body) = request(&pool_router, "/v1/decisions", Some(decisions_request())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, decision);
    request(
        &app,
        "/admin/api/models/unload",
        Some(json!({"model":MODEL})),
    )
    .await;
    assert!(manager.pool.loaded_model_infos().await.is_empty());
    std::fs::remove_dir_all(voice_dir).unwrap();
}
