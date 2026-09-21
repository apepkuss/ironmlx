//! Real-model acceptance of the public runtime boundary without HTTP or ironmlx.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::models::{DFlash2DraftModel, Qwen35Model};
use ironmlx_lm::{Loader, Tokenizer};
use ironmlx_runtime::core::engine_state::build_dflash2_engine;
use ironmlx_runtime::core::generation_types::GenerateRequest;
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
