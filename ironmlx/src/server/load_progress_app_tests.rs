//! Model load progress and cancellation over the admin API. Loads run the
//! runtime's test load hook, which walks the real stages without weights and
//! can be held at a known stage.

use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use ironmlx_runtime::core::engine_pool::{TestLoadHook, TestLoadPause};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const WAIT: Duration = Duration::from_secs(10);

async fn request(
    app: &Router,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(body) => builder
            .header("content-type", "application/json")
            .body(Body::from(body.to_string())),
        None => builder.body(Body::empty()),
    }
    .unwrap();
    let response = tokio::time::timeout(WAIT, app.clone().oneshot(request))
        .await
        .unwrap_or_else(|_| panic!("{method} {path} did not respond"))
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn post(app: &Router, path: &str, body: Value) -> (StatusCode, Value) {
    request(app, "POST", path, Some(body)).await
}

async fn get(app: &Router, path: &str) -> (StatusCode, Value) {
    request(app, "GET", path, None).await
}

struct Fixture {
    manager: ModelManager,
    app: Router,
    hook: Arc<TestLoadHook>,
    dirs: Vec<PathBuf>,
    profile_store_root: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for dir in &self.dirs {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn model_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ironmlx-load-progress-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
    std::fs::write(dir.join("model.safetensors"), vec![0_u8; 1024]).unwrap();
    dir
}

fn fixture() -> Fixture {
    fixture_with(|_| {})
}

fn fixture_with(configure: impl FnOnce(&mut crate::cli::serve::ServeArgs)) -> Fixture {
    let mut args = super::tests::serve_args();
    configure(&mut args);
    // Scheduler profiles are looked up in a store of this test's own.
    let profile_store_root =
        std::env::temp_dir().join(format!("ironmlx-load-profiles-{}", uuid::Uuid::new_v4()));
    let mut scheduler_options = SchedulerResolutionOptions::from(&args);
    scheduler_options.scheduler_profile_store_root = Some(profile_store_root.clone());
    let manager = ModelManager::new(
        crate::cli::serve::engine_runtime_config(&args).unwrap(),
        args.max_loaded_models,
        scheduler_options,
    )
    .unwrap();
    let hook = manager.pool.install_test_load_hook();
    let voice_dir =
        std::env::temp_dir().join(format!("ironmlx-load-voices-{}", uuid::Uuid::new_v4()));
    let voices = crate::server::voices::VoiceStore::open(voice_dir.clone()).unwrap();
    let app = app_router(manager.clone(), voices);
    Fixture {
        manager,
        app,
        hook,
        dirs: vec![voice_dir, profile_store_root.clone()],
        profile_store_root,
    }
}

impl Fixture {
    fn add_model(&mut self) -> PathBuf {
        let dir = model_dir();
        self.dirs.push(dir.clone());
        dir
    }

    async fn load(&self, model: &str, dir: &Path, extra: Value) -> (StatusCode, Value) {
        let mut body = json!({"model": model, "model_dir": dir});
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        post(&self.app, "/admin/api/models/load", body).await
    }

    async fn loads(&self) -> Vec<Value> {
        let (status, body) = get(&self.app, "/admin/api/models/loads").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["loads"].as_array().unwrap().clone()
    }

    async fn wait_for_state(&self, load_id: &str, state: &str) -> Value {
        tokio::time::timeout(WAIT, async {
            loop {
                if let Some(load) = self
                    .loads()
                    .await
                    .into_iter()
                    .find(|load| load["load_id"] == load_id && load["state"] == state)
                {
                    return load;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("load {load_id} reaches {state}"))
    }
}

fn spawn_load(app: &Router, body: Value) -> tokio::task::JoinHandle<(StatusCode, Value)> {
    let app = app.clone();
    tokio::spawn(async move { post(&app, "/admin/api/models/load", body).await })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_loads_endpoint_lists_queued_and_loading() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    let beta = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));

    let (status, started) = fixture
        .load(
            "alpha",
            &alpha,
            json!({"async": true, "load_id": "alpha-load"}),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{started}");
    fixture.hook.wait_reached().await;

    // A control-plane load of another model waits behind the load gate.
    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/register",
        json!({"model": "beta", "model_dir": beta}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let queued = tokio::spawn({
        let pool = fixture.manager.pool.clone();
        async move { pool.load_model("beta").await }
    });
    tokio::time::timeout(WAIT, async {
        while fixture.loads().await.len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both loads listed");

    let loads = fixture.loads().await;
    let alpha_load = loads.iter().find(|load| load["model"] == "alpha").unwrap();
    assert_eq!(alpha_load["state"], "loading");
    assert_eq!(alpha_load["stage"], "reading_weights");
    assert_eq!(alpha_load["trigger"], "admin");
    assert_eq!(alpha_load["bytes_done"], 500);
    let beta_load = loads.iter().find(|load| load["model"] == "beta").unwrap();
    assert_eq!(beta_load["state"], "queued");
    assert_eq!(beta_load["trigger"], "openai_control");

    fixture.hook.pause_at(None);
    fixture.hook.release();
    queued.await.unwrap().expect("beta loads");
    fixture.wait_for_state("alpha-load", "loaded").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_async_load_returns_202_and_finishes_in_registry() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));

    let (status, body) = fixture.load("alpha", &alpha, json!({"async": true})).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let load_id = body["load_id"].as_str().expect("load id").to_string();
    assert!(
        matches!(body["state"].as_str(), Some("queued" | "loading")),
        "{body}"
    );
    fixture.hook.wait_reached().await;
    let loads = fixture.loads().await;
    assert_eq!(loads.len(), 1);
    assert_eq!(loads[0]["load_id"], load_id.as_str());
    assert_eq!(loads[0]["state"], "loading");

    fixture.hook.pause_at(None);
    fixture.hook.release();
    let finished = fixture.wait_for_state(&load_id, "loaded").await;
    assert_eq!(finished["model"], "alpha");
    assert!(finished["finished_unix_ms"].is_u64());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_cancel_endpoint_status_codes() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();

    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/loads/missing/cancel",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["code"], "load_not_found");

    // Past the point of no return: the engine is started.
    fixture.hook.pause_at(Some(TestLoadPause::AfterEngineStart));
    let (status, body) = fixture
        .load(
            "alpha",
            &alpha,
            json!({"async": true, "load_id": "finishing"}),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    fixture.hook.wait_reached().await;
    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/loads/finishing/cancel",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_finishing");

    fixture.hook.pause_at(None);
    fixture.hook.release();
    fixture.wait_for_state("finishing", "loaded").await;
    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/loads/finishing/cancel",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_finished");

    // A load that is still reading weights accepts the cancel.
    let beta = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
    let (status, body) = fixture
        .load("beta", &beta, json!({"async": true, "load_id": "reading"}))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    fixture.hook.wait_reached().await;
    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/loads/reading/cancel",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body, json!({"load_id": "reading", "state": "cancelling"}));
    fixture.hook.pause_at(None);
    fixture.hook.release();
    fixture.wait_for_state("reading", "cancelled").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_sync_load_cancelled_returns_409() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));

    let sync = spawn_load(
        &fixture.app,
        json!({"model": "alpha", "model_dir": alpha, "load_id": "doomed"}),
    );
    fixture.hook.wait_reached().await;
    let (status, body) = post(
        &fixture.app,
        "/admin/api/models/loads/doomed/cancel",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    fixture.hook.pause_at(None);
    fixture.hook.release();

    let (status, body) = tokio::time::timeout(WAIT, sync).await.unwrap().unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_cancelled");
    assert_eq!(body["load_id"], "doomed");
    assert_eq!(body["success"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_client_disconnect_does_not_cancel_load() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));

    let abandoned = tokio::time::timeout(
        Duration::from_millis(300),
        fixture.load("alpha", &alpha, json!({"load_id": "abandoned"})),
    )
    .await;
    assert!(abandoned.is_err(), "the held load keeps the request open");
    fixture.hook.wait_reached().await;
    let loads = fixture.loads().await;
    assert_eq!(loads[0]["load_id"], "abandoned");
    assert_eq!(loads[0]["state"], "loading");

    fixture.hook.pause_at(None);
    fixture.hook.release();
    fixture.wait_for_state("abandoned", "loaded").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_load_in_progress_conflict_includes_load_id() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));

    let (status, body) = fixture
        .load("alpha", &alpha, json!({"async": true, "load_id": "first"}))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    fixture.hook.wait_reached().await;

    // The same configuration joins the running load.
    let joined = spawn_load(&fixture.app, json!({"model": "alpha", "model_dir": alpha}));
    // A different configuration is refused with the running load's id.
    let (status, body) = fixture
        .load("alpha", &alpha, json!({"max_cache_cap": 4096}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_in_progress");
    assert_eq!(body["load_id"], "first");
    assert_eq!(fixture.loads().await.len(), 1);

    // An async request for the same configuration joins as well.
    let (status, body) = fixture.load("alpha", &alpha, json!({"async": true})).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["load_id"], "first");

    fixture.hook.pause_at(None);
    fixture.hook.release();
    let (status, body) = tokio::time::timeout(WAIT, joined).await.unwrap().unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "loaded");
    assert_eq!(body["load_id"], "first");
    assert_eq!(fixture.loads().await.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_client_supplied_load_id_is_used() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();

    let (status, body) = fixture
        .load("alpha", &alpha, json!({"load_id": "my-load.1"}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "loaded");
    assert_eq!(body["load_id"], "my-load.1");
    let loads = fixture.loads().await;
    assert_eq!(loads[0]["load_id"], "my-load.1");
    assert_eq!(loads[0]["state"], "loaded");

    // Without a client id the response still names the load.
    let beta = fixture.add_model();
    let (status, body) = fixture.load("beta", &beta, json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let generated = body["load_id"].as_str().expect("generated load id");
    assert!(!generated.is_empty());
    assert!(fixture
        .loads()
        .await
        .iter()
        .any(|load| load["load_id"] == generated));

    // Ids are path segments and must stay unique among known loads.
    let gamma = fixture.add_model();
    let (status, body) = fixture
        .load("gamma", &gamma, json!({"load_id": "bad id/"}))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "invalid_load_id");
    let (status, body) = fixture
        .load("gamma", &gamma, json!({"load_id": "my-load.1"}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_id_in_use");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_async_failure_keeps_the_error_classification() {
    let mut fixture = fixture_with(|args| args.max_loaded_models = Some(1));
    let alpha = fixture.add_model();
    let beta = fixture.add_model();
    let (status, body) = fixture.load("alpha", &alpha, json!({"pinned": true})).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The pool is full of a pinned model: the load is refused after it
    // registered, so only the registry carries the reason.
    let (status, body) = fixture
        .load(
            "beta",
            &beta,
            json!({"async": true, "load_id": "beta-load"}),
        )
        .await;
    assert!(
        status == StatusCode::ACCEPTED || status == StatusCode::SERVICE_UNAVAILABLE,
        "{status} {body}"
    );
    let failed = fixture.wait_for_state("beta-load", "failed").await;
    assert_eq!(failed["code"], MAX_LOADED_MODELS_REACHED_CODE, "{failed}");
    assert_eq!(
        failed["message"], MAX_LOADED_MODELS_REACHED_MESSAGE,
        "{failed}"
    );

    // A synchronous request refused the same way keeps its status and body.
    let (status, body) = fixture.load("beta", &beta, json!({})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["code"], MAX_LOADED_MODELS_REACHED_CODE);
    assert_eq!(body["error"], MAX_LOADED_MODELS_REACHED_MESSAGE);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_joined_load_answers_like_the_initiator() {
    use ironmlx_runtime::core::model_management::{
        ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
        DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
    };
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
    fixture.hook.fail_with(Some(|| {
        ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        }
        .into()
    }));

    let initiator = spawn_load(
        &fixture.app,
        json!({"model": "alpha", "model_dir": alpha, "load_id": "first"}),
    );
    fixture.hook.wait_reached().await;
    let joined = spawn_load(&fixture.app, json!({"model": "alpha", "model_dir": alpha}));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!joined.is_finished());
    fixture.hook.release();

    let (status, body) = tokio::time::timeout(WAIT, initiator)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE);
    assert_eq!(body["error"], DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE);
    let (joined_status, joined_body) = tokio::time::timeout(WAIT, joined).await.unwrap().unwrap();
    assert_eq!(joined_status, status, "{joined_body}");
    assert_eq!(joined_body["code"], body["code"], "{joined_body}");
    assert_eq!(joined_body["error"], body["error"], "{joined_body}");
    let failed = fixture.wait_for_state("first", "failed").await;
    assert_eq!(
        failed["code"], DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
        "{failed}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_default_scheduler_load_has_no_warning() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    let (status, body) = fixture
        .load("alpha", &alpha, json!({"async": true, "load_id": "plain"}))
        .await;
    assert!(
        status == StatusCode::ACCEPTED || status == StatusCode::OK,
        "{status} {body}"
    );
    let loaded = fixture.wait_for_state("plain", "loaded").await;
    assert!(loaded["warning_code"].is_null(), "{loaded}");
    assert!(loaded["warning"].is_null(), "{loaded}");
    let (status, models) = get(&fixture.app, "/admin/api/models/loaded").await;
    assert_eq!(status, StatusCode::OK, "{models}");
    let alpha = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "alpha")
        .unwrap_or_else(|| panic!("alpha is loaded: {models}"));
    assert_eq!(alpha["scheduler_profile"], "default", "{alpha}");
}

/// Stores a profile for the model at `model_dir` made on `hardware_label`
/// for a runtime context that is not the load's.
fn persist_profile_elsewhere(fixture: &Fixture, model_dir: &Path, hardware_label: &str) {
    use ironmlx_runtime::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeContext,
        SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeProfileMetadata,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };
    use ironmlx_runtime::core::scheduler_profile_store::SchedulerProfileStore;
    let profile = SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "alpha".to_string(),
        hardware_label: hardware_label.to_string(),
        runtime_context: SchedulerAutotuneRuntimeContext::local_default(32768),
        config: SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
    };
    SchedulerProfileStore::from_root(&fixture.profile_store_root)
        .persist_profile(model_dir, &profile)
        .expect("persist profile");
}

async fn load_alpha_async(fixture: &Fixture, alpha: &Path, load_id: &str) -> (Value, Value) {
    let (status, body) = fixture
        .load("alpha", alpha, json!({"async": true, "load_id": load_id}))
        .await;
    assert!(
        status == StatusCode::ACCEPTED || status == StatusCode::OK,
        "{status} {body}"
    );
    let loaded = fixture.wait_for_state(load_id, "loaded").await;
    let (status, models) = get(&fixture.app, "/admin/api/models/loaded").await;
    assert_eq!(status, StatusCode::OK, "{models}");
    let model = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "alpha")
        .cloned()
        .unwrap_or_else(|| panic!("alpha is loaded: {models}"));
    (loaded, model)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_async_load_reports_a_profile_made_on_other_hardware() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    persist_profile_elsewhere(&fixture, &alpha, "another-host");
    let (loaded, model) = load_alpha_async(&fixture, &alpha, "warned").await;
    assert_eq!(
        loaded["warning_code"], SCHEDULER_PROFILE_HARDWARE_MISMATCH_WARNING_CODE,
        "{loaded}"
    );
    assert_eq!(
        loaded["warning"], SCHEDULER_PROFILE_HARDWARE_MISMATCH_WARNING,
        "{loaded}"
    );
    assert_eq!(model["scheduler_profile"], "default", "{model}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_settings_mismatch_is_shown_without_a_warning() {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    let host =
        ironmlx_runtime::core::scheduler_profile_store::detect_scheduler_profile_hardware_label();
    persist_profile_elsewhere(&fixture, &alpha, &host);
    let (loaded, model) = load_alpha_async(&fixture, &alpha, "settings").await;
    assert!(loaded["warning_code"].is_null(), "{loaded}");
    assert_eq!(
        model["scheduler_profile"], "default_settings_mismatch",
        "{model}"
    );
}

/// Two requests for one model race past the up-front in-progress check: the
/// later one registers first. The other request must answer for its own load,
/// never with the id of a load whose settings differ.
async fn race_against_registered_load(different_config: bool) -> (StatusCode, Value) {
    let mut fixture = fixture();
    let alpha = fixture.add_model();
    // Held after the up-front check and before registering.
    fixture
        .hook
        .pause_at(Some(TestLoadPause::BeforeRegistration));
    let mut first = json!({"model": "alpha", "model_dir": alpha, "async": true, "load_id": "mine"});
    if different_config {
        first["max_cache_cap"] = json!(4096);
    }
    let first = spawn_load(&fixture.app, first);
    assert_eq!(
        fixture.hook.wait_reached().await,
        TestLoadPause::BeforeRegistration
    );

    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
    let (status, body) = fixture
        .load("alpha", &alpha, json!({"async": true, "load_id": "other"}))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["load_id"], "other");
    assert_eq!(
        fixture.hook.wait_reached().await,
        TestLoadPause::ReadingWeights
    );
    tokio::time::sleep(Duration::from_millis(50)).await;
    if different_config {
        assert!(!first.is_finished(), "answered before its own load ran");
    }

    // Let the first request register (or join), then the other load finish.
    fixture.hook.release();
    tokio::time::sleep(Duration::from_millis(50)).await;
    fixture.hook.pause_at(None);
    fixture.hook.release();
    let answer = tokio::time::timeout(WAIT, first).await.unwrap().unwrap();
    fixture.wait_for_state("other", "loaded").await;
    answer
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_async_start_refuses_another_configs_load() {
    let (status, body) = race_against_registered_load(true).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "load_in_progress");
    assert_eq!(body["load_id"], "other");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_async_start_joins_the_same_configs_load() {
    let (status, body) = race_against_registered_load(false).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["load_id"], "other");
}

#[tokio::test]
async fn load_progress_panicked_load_task_is_an_internal_error() {
    let panicked = tokio::spawn(async { panic!("load task panicked") })
        .await
        .expect_err("task panics");
    let error = AdminError::load_task_failed(panicked);
    assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(error.code, Some("load_task_failed"));
}

const INFERENCE_PATHS: [&str; 2] = ["/v1/chat/completions", "/v1/messages"];

fn inference_body(model: &str) -> Value {
    json!({
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

/// An inference request's answer, with the Retry-After header if any.
#[derive(Debug)]
struct Answer {
    status: StatusCode,
    retry_after: Option<String>,
    body: Value,
}

async fn infer(app: &Router, path: &str, model: &str) -> Answer {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(inference_body(model).to_string()))
        .unwrap();
    let response = tokio::time::timeout(WAIT, app.clone().oneshot(request))
        .await
        .unwrap_or_else(|_| panic!("POST {path} did not respond"))
        .unwrap();
    let status = response.status();
    let retry_after = response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    Answer {
        status,
        retry_after,
        body: serde_json::from_slice(&bytes).unwrap(),
    }
}

fn spawn_infer(
    app: &Router,
    path: &'static str,
    model: &'static str,
) -> tokio::task::JoinHandle<Answer> {
    let app = app.clone();
    tokio::spawn(async move { infer(&app, path, model).await })
}

impl Fixture {
    async fn register(&mut self, model: &str) {
        let dir = self.add_model();
        let (status, body) = post(
            &self.app,
            "/admin/api/models/register",
            json!({"model": model, "model_dir": dir}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
}

/// Error code in either protocol's envelope; the envelope must match `path`.
fn inference_error_code(path: &str, body: &Value) -> String {
    if path == "/v1/messages" {
        assert_eq!(body["type"], "error", "{body}");
        assert!(body["request_id"].is_string(), "{body}");
    } else {
        assert!(body["error"]["type"].is_string(), "{body}");
    }
    body["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

fn reload_required(message: &str) -> String {
    format!(
        "{message} {}",
        crate::server::api_error::RELOAD_REQUIRED_MESSAGE
    )
}

/// An inference request starts the lazy load of `alpha`, held while reading
/// weights; a second request joins that load. The load is then cancelled, or
/// fails with `failure`. Returns the starter's and the joiner's answers.
async fn lazy_load_answers(
    path: &'static str,
    failure: Option<fn() -> anyhow::Error>,
) -> [Answer; 2] {
    let mut fixture = fixture();
    fixture.register("alpha").await;
    fixture.hook.pause_at(Some(TestLoadPause::ReadingWeights));
    fixture.hook.fail_with(failure);

    let starter = spawn_infer(&fixture.app, path, "alpha");
    fixture.hook.wait_reached().await;
    let joiner = spawn_infer(&fixture.app, path, "alpha");
    fixture.hook.wait_joins(1).await;
    let loads = fixture.loads().await;
    assert_eq!(loads.len(), 1, "the joiner shares the load");
    assert_eq!(loads[0]["trigger"], "request");
    let load_id = loads[0]["load_id"].as_str().unwrap().to_owned();

    if failure.is_none() {
        let (status, body) = post(
            &fixture.app,
            &format!("/admin/api/models/loads/{load_id}/cancel"),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    }
    fixture.hook.pause_at(None);
    fixture.hook.release();

    let starter = tokio::time::timeout(WAIT, starter).await.unwrap().unwrap();
    let joiner = tokio::time::timeout(WAIT, joiner).await.unwrap().unwrap();
    let state = if failure.is_none() {
        "cancelled"
    } else {
        "failed"
    };
    fixture.wait_for_state(&load_id, state).await;
    assert_eq!(fixture.hook.loads_started(), 1, "one load served both");
    [starter, joiner]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_inference_request_on_a_cancelled_lazy_load_is_409() {
    for path in INFERENCE_PATHS {
        for answer in lazy_load_answers(path, None).await {
            let body = &answer.body;
            assert_eq!(answer.status, StatusCode::CONFLICT, "{path} {body}");
            assert_eq!(answer.retry_after, None, "{path}");
            assert_eq!(inference_error_code(path, body), "load_cancelled", "{path}");
            assert_eq!(
                body["error"]["message"],
                ironmlx_runtime::core::load_error::LOAD_CANCELLED_MESSAGE,
                "{path}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_inference_request_on_a_misconfigured_lazy_load_is_400() {
    use ironmlx_runtime::core::model_management::{
        ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
        DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
    };
    fn capability() -> anyhow::Error {
        ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        }
        .into()
    }
    for path in INFERENCE_PATHS {
        for answer in lazy_load_answers(path, Some(capability)).await {
            let body = &answer.body;
            assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{path} {body}");
            assert_eq!(answer.retry_after, None, "{path}");
            assert_eq!(
                inference_error_code(path, body),
                DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
                "{path}"
            );
            assert_eq!(
                body["error"]["message"],
                reload_required(DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE),
                "{path}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_inference_request_refused_for_capacity_is_503_with_its_code() {
    for path in INFERENCE_PATHS {
        let mut fixture = fixture_with(|args| args.max_loaded_models = Some(1));
        let alpha = fixture.add_model();
        let (status, body) = fixture.load("alpha", &alpha, json!({"pinned": true})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        fixture.register("beta").await;

        // Refused before loading: beta stays unloaded and the next request
        // tries again, so the answer keeps Retry-After and its own advice.
        for _ in 0..2 {
            let answer = infer(&fixture.app, path, "beta").await;
            let body = &answer.body;
            assert_eq!(
                answer.status,
                StatusCode::SERVICE_UNAVAILABLE,
                "{path} {body}"
            );
            assert_eq!(answer.retry_after.as_deref(), Some("5"), "{path}");
            assert_eq!(
                inference_error_code(path, body),
                MAX_LOADED_MODELS_REACHED_CODE,
                "{path}"
            );
            assert_eq!(
                body["error"]["message"], MAX_LOADED_MODELS_REACHED_MESSAGE,
                "{path}"
            );
        }
    }
}

/// The first request's lazy load of `model` fails with `failure`; a later
/// request finds the model in its failed state. Returns both answers.
async fn failed_model_answers(
    path: &'static str,
    model: &str,
    failure: fn() -> anyhow::Error,
) -> [Answer; 2] {
    let mut fixture = fixture();
    fixture.register(model).await;
    fixture.hook.fail_with(Some(failure));
    let first = infer(&fixture.app, path, model).await;
    let later = infer(&fixture.app, path, model).await;
    assert_eq!(
        fixture.hook.loads_started(),
        1,
        "the later request does not reload"
    );
    [first, later]
}

/// Answers compared without the per-response Anthropic request id.
fn without_request_id(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(object) = body.as_object_mut() {
        object.remove("request_id");
    }
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_later_request_to_a_failed_model_answers_like_the_first() {
    use ironmlx_runtime::core::load_error::{
        GPU_MEMORY_INSUFFICIENT_CODE, GPU_MEMORY_INSUFFICIENT_MESSAGE,
    };
    use ironmlx_runtime::core::model_management::{
        ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
        DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
    };
    fn capability() -> anyhow::Error {
        ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        }
        .into()
    }
    fn out_of_memory() -> anyhow::Error {
        anyhow::anyhow!("Metal OOM: command buffer failed")
    }
    let cases: [(fn() -> anyhow::Error, StatusCode, &str, &str); 2] = [
        (
            capability,
            StatusCode::BAD_REQUEST,
            DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        ),
        (
            out_of_memory,
            StatusCode::SERVICE_UNAVAILABLE,
            GPU_MEMORY_INSUFFICIENT_CODE,
            GPU_MEMORY_INSUFFICIENT_MESSAGE,
        ),
    ];
    for path in INFERENCE_PATHS {
        for (failure, status, code, message) in cases {
            let [first, later] = failed_model_answers(path, "alpha", failure).await;
            let body = &first.body;
            assert_eq!(first.status, status, "{path} {body}");
            assert_eq!(inference_error_code(path, body), code, "{path}");
            // Requests never reload a failed model: no Retry-After, and the
            // message says how to get it loaded.
            assert_eq!(first.retry_after, None, "{path}");
            assert_eq!(body["error"]["message"], reload_required(message), "{path}");
            assert_eq!(later.status, first.status, "{path} {}", later.body);
            assert_eq!(later.retry_after, first.retry_after, "{path}");
            inference_error_code(path, &later.body);
            assert_eq!(
                without_request_id(&later.body),
                without_request_id(&first.body),
                "{path}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial(model_load_governor)]
async fn load_progress_later_request_to_a_model_failed_for_an_unknown_reason_is_503() {
    let reload = format!(" {}", crate::server::api_error::RELOAD_REQUIRED_MESSAGE);
    fn unknown() -> anyhow::Error {
        anyhow::anyhow!("weights for bigscience/bloom are truncated")
    }
    // The model id must not classify the error either.
    for model in ["alpha", "foo-oom"] {
        for path in INFERENCE_PATHS {
            for answer in failed_model_answers(path, model, unknown).await {
                let body = &answer.body;
                assert_eq!(
                    answer.status,
                    StatusCode::SERVICE_UNAVAILABLE,
                    "{model} {path} {body}"
                );
                // The model is left failed and requests do not load it
                // again, classified or not.
                assert_eq!(answer.retry_after, None, "{model} {path}");
                assert_eq!(
                    inference_error_code(path, body),
                    "engine_unavailable",
                    "{model} {path}"
                );
                let message = body["error"]["message"].as_str().unwrap();
                assert!(
                    message.starts_with(&format!("loading engine model `{model}`: "))
                        && message.contains("weights for bigscience/bloom are truncated")
                        && message.ends_with(&reload),
                    "{model} {path} {body}"
                );
            }
        }
    }
}
