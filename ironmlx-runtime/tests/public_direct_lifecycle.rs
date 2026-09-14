//! Native direct execution must retain its lock until the event consumer finishes.
#![cfg(feature = "test-support")]

use std::time::Duration;

use ironmlx_lm::{core::loader::TokenizerConfig, Tokenizer};
use ironmlx_runtime::core::cache::ActiveKvOffloadConfig;
use ironmlx_runtime::core::direct_execution::spawn_direct_with_events;
use ironmlx_runtime::core::engine_state::build_plain_app_state;
use ironmlx_runtime::core::generation_types::{GenerateEvent, GenerateRequest};
use ironmlx_runtime::core::scheduler_autotune::*;
use ironmlx_runtime::test_support::SchedulerActorFakeModel;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial(mlx_metal)]
async fn consumer_early_exit_releases_direct_worker_model_lock() {
    let dir = std::env::temp_dir().join(format!("ironmlx-direct-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    std::fs::write(
        &path,
        serde_json::json!({
            "version":"1.0", "truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}
        })
        .to_string(),
    )
    .unwrap();
    let config: TokenizerConfig =
        serde_json::from_value(serde_json::json!({"chat_template":"hello"})).unwrap();
    let tokenizer = Tokenizer::from_files(&path, &config).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let profile = SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "direct-test".into(),
        hardware_label: "test".into(),
        runtime_context: SchedulerAutotuneRuntimeContext::local_default(256),
        config: SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 16,
            admission_deadline_ms: 5,
            admission_queue_max: 4,
            max_cache_cap: 256,
            decode_cadence_mid_chunk_cap: 16,
        },
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
    };
    let state = build_plain_app_state(
        SchedulerActorFakeModel,
        tokenizer,
        "direct-test".into(),
        16,
        1,
        5,
        4,
        256,
        16,
        None,
        profile,
        false,
        None,
        None,
        None,
        Default::default(),
        ActiveKvOffloadConfig::disabled(),
    )
    .await
    .unwrap();
    let model = state.model.clone();
    let request = GenerateRequest {
        prompt_ids: vec![1],
        max_new_tokens: 2,
        ..ironmlx_runtime::test_support::mk_req(1)
    };
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = spawn_direct_with_events(
        state,
        request,
        Some(vec![
            GenerateEvent {
                token: 1,
                text: "hello".into(),
                finish_reason: None,
            },
            GenerateEvent {
                token: 1,
                text: "hello".into(),
                finish_reason: Some("length"),
            },
        ]),
        move |initialized| {
            let mut generation = initialized.unwrap();
            assert_eq!(generation.next_token().unwrap().unwrap().text, "hello");
            assert!(generation.commit_memory());
            assert!(!generation.commit_memory());
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            // A disconnected consumer stops without reading the final event.
        },
    );
    tokio::time::timeout(Duration::from_secs(10), started_rx)
        .await
        .unwrap()
        .unwrap();
    assert!(
        model.try_lock().is_err(),
        "the worker must retain the lock through consumer finalization"
    );
    release_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(10), worker)
        .await
        .unwrap()
        .unwrap();
    assert!(
        model.try_lock().is_ok(),
        "early consumer exit must release the model lock"
    );
}
