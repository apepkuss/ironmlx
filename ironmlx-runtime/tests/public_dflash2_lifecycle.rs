//! Real-model acceptance of the public runtime boundary without HTTP or ironmlx.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::models::{DFlash2DraftModel, Qwen35Model};
use ironmlx_lm::{Loader, Tokenizer};
use ironmlx_runtime::core::engine_state::build_dflash2_engine;
use ironmlx_runtime::core::generation_types::{GenerateRequest, RequestPriority};
use ironmlx_runtime::core::process_memory::StaticMemoryEstimate;
use ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext;
use ironmlx_runtime::core::scheduler_resolution::default_scheduler_runtime_profile;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires QWEN38_MODEL and DFLASH2_MODEL real checkpoints"]
async fn dflash2_public_runtime_cancels_recovers_and_releases_model_ownership() {
    let model_dir = PathBuf::from(std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL"));
    let draft_dir = PathBuf::from(std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL"));
    let loader = Loader::open(&model_dir).expect("target loader");
    let model = Qwen35Model::from_loader(&loader).expect("target model");
    let tokenizer = Tokenizer::from_loader(&loader).expect("tokenizer");
    let draft_loader = Loader::open_dflash2(&draft_dir).expect("draft loader");
    let draft = DFlash2DraftModel::from_loader(&draft_loader, model.config(), Some(4))
        .expect("draft model");
    let estimate = StaticMemoryEstimate {
        text_cold_bytes: loader.loaded_tensor_bytes(),
        speculative_cold_bytes: draft_loader.loaded_tensor_bytes(),
        ..Default::default()
    };
    drop(loader);
    drop(draft_loader);
    let state = build_dflash2_engine(
        model,
        draft,
        tokenizer,
        "dflash2-lifecycle".to_string(),
        2048,
        2,
        5,
        2,
        1,
        4096,
        4,
        Some(4),
        None,
        default_scheduler_runtime_profile(SchedulerAutotuneRuntimeContext::local_default(4096)),
        estimate,
    )
    .await
    .expect("native engine");
    assert!(state.request_execution.is_dflash2());
    let weak_model = Arc::downgrade(&state.model);
    let weak_tokenizer = Arc::downgrade(&state.tokenizer);
    let prompt_ids = state
        .tokenizer
        .encode(
            "Write all integers from one to one thousand, separated by commas.",
            true,
        )
        .expect("prompt");
    let request = |max_new_tokens| GenerateRequest {
        priority: Default::default(),
        prompt_ids: prompt_ids.clone(),
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    };
    let mut cancelled = state
        .request_execution
        .admit(request(2048))
        .await
        .unwrap_or_else(|_| panic!("first native admission"));
    let first = tokio::time::timeout(Duration::from_secs(60), cancelled.event_rx.recv())
        .await
        .expect("first token timeout")
        .expect("first token");
    assert!(
        first.finish_reason.is_none(),
        "cancel during active generation"
    );
    drop(cancelled);
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let health = state.health_collector.snapshot();
            if state.request_execution.active_and_queued() == (0, 0)
                && health.memory.kv_cache_active_bytes == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancel releases admission and KV reservation");

    let mut recovered = state
        .request_execution
        .admit(request(8))
        .await
        .unwrap_or_else(|_| panic!("recovery admission"));
    let mut emitted = 0;
    let mut finished = false;
    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(event) = recovered.event_rx.recv().await {
            emitted += 1;
            if event.finish_reason.is_some() {
                finished = true;
                break;
            }
        }
    })
    .await
    .expect("recovery finishes");
    assert!(emitted > 0 && finished);
    assert!(state.health_collector.snapshot().dflash2.enabled);
    drop(recovered);
    drop(state);
    tokio::time::timeout(Duration::from_secs(30), async {
        while weak_model.upgrade().is_some() || weak_tokenizer.upgrade().is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("last engine handle drop stops worker and releases model/tokenizer ownership");
    mlx::clear_cache();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires QWEN38_MODEL and DFLASH2_MODEL real checkpoints"]
#[serial_test::serial(mlx_metal)]
async fn dflash2_foreground_preempts_and_resumes_background_without_replay() {
    let model_dir = PathBuf::from(std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL"));
    let draft_dir = PathBuf::from(std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL"));
    let mut loader = Loader::open(&model_dir).expect("target loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader_dflash2(&mut loader).expect("DFlash2 target model");
    let draft_loader = Loader::open_dflash2(&draft_dir).expect("draft loader");
    let draft = DFlash2DraftModel::from_loader(&draft_loader, model.config(), Some(4))
        .expect("draft model");
    let estimate = StaticMemoryEstimate {
        text_cold_bytes: loader.loaded_tensor_bytes(),
        speculative_cold_bytes: draft_loader.loaded_tensor_bytes(),
        ..Default::default()
    };
    drop(loader);
    drop(draft_loader);
    let state = build_dflash2_engine(
        model,
        draft,
        tokenizer,
        "dflash2-priority".to_string(),
        2048,
        2,
        1,
        2,
        4,
        4096,
        8,
        Some(4),
        None,
        default_scheduler_runtime_profile(SchedulerAutotuneRuntimeContext::local_default(4096)),
        estimate,
    )
    .await
    .expect("native engine");
    let prompt_ids = state
        .tokenizer
        .encode(
            "Write all integers from one to one thousand, separated by commas.",
            true,
        )
        .expect("prompt");
    let request = |priority, max_new_tokens| GenerateRequest {
        priority,
        prompt_ids: prompt_ids.clone(),
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    };

    let mut background = state
        .request_execution
        .admit(request(RequestPriority::Background, 64))
        .await
        .unwrap_or_else(|_| panic!("background admission"));
    tokio::time::timeout(Duration::from_secs(60), background.event_rx.recv())
        .await
        .expect("background first-token timeout")
        .expect("background first token");

    let mut foreground = state
        .request_execution
        .admit(request(RequestPriority::Foreground, 4))
        .await
        .unwrap_or_else(|_| panic!("foreground admission"));
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if state
                .health_collector
                .snapshot()
                .scheduler
                .background_paused
                == 1
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("background pause published");
    while background.event_rx.try_recv().is_ok() {}

    let mut foreground_finished = false;
    while let Some(event) =
        tokio::time::timeout(Duration::from_secs(30), foreground.event_rx.recv())
            .await
            .expect("foreground event timeout")
    {
        assert!(
            background.event_rx.try_recv().is_err(),
            "background advanced while foreground owned the execution lane"
        );
        if event.finish_reason.is_some() {
            foreground_finished = true;
            break;
        }
    }
    assert!(foreground_finished);

    let mut background_finished = false;
    while let Some(event) =
        tokio::time::timeout(Duration::from_secs(60), background.event_rx.recv())
            .await
            .expect("background continuation timeout")
    {
        if event.finish_reason.is_some() {
            background_finished = true;
            break;
        }
    }
    assert!(background_finished);
    let health = state.health_collector.snapshot();
    assert_eq!(health.scheduler.background_paused, 0);
    assert_eq!(health.scheduler.background_preemptions, 1);
    assert_eq!(health.scheduler.background_resumes, 1);
    drop(state);
    mlx::clear_cache();
}
