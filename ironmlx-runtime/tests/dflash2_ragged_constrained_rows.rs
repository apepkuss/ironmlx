//! Ragged (multi-request) DFlash2 windows take constrained rows: four
//! concurrent Qwen3.6 MoE requests under the native reasoning budget (what a
//! chat request with thinking enabled carries) decode in ragged windows, and
//! every output keeps the budget's framing.

use std::path::PathBuf;
use std::time::Duration;

use ironmlx_core::m5_profile::{self, M5ProfileMode, M5ProfileTarget};
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::models::dflash2::DFlash2Target;
use ironmlx_lm::models::{DFlash2DraftModel, Qwen35MoeModel};
use ironmlx_lm::{Loader, Tokenizer};
use ironmlx_runtime::core::engine_state::build_dflash2_engine;
use ironmlx_runtime::core::generation_types::GenerateRequest;
use ironmlx_runtime::core::process_memory::StaticMemoryEstimate;
use ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext;
use ironmlx_runtime::core::scheduler_resolution::default_scheduler_runtime_profile;

const PROMPTS: [&str; 4] = [
    "Write a Rust function that parses a semantic version string into its three numeric parts.",
    "Explain why the sky is blue in three sentences.",
    "Summarize the plot of a heist film in which the plan goes wrong halfway through.",
    "List four differences between TCP and UDP.",
];
const NEW_TOKENS: usize = 160;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN36_MOE_MODEL and DFLASH2_MOE_MODEL real checkpoints"]
async fn ragged_windows_take_rows_under_the_native_reasoning_budget() {
    let model_dir = PathBuf::from(std::env::var("QWEN36_MOE_MODEL").expect("QWEN36_MOE_MODEL"));
    let draft_dir = PathBuf::from(std::env::var("DFLASH2_MOE_MODEL").expect("DFLASH2_MOE_MODEL"));
    m5_profile::install_for_target(M5ProfileMode::Auto, true, M5ProfileTarget::Qwen36Moe);
    let loader = Loader::open(&model_dir).expect("target loader");
    let model = Qwen35MoeModel::from_loader(&loader).expect("target model");
    let tokenizer = Tokenizer::from_loader(&loader).expect("tokenizer");
    let draft_loader = Loader::open_dflash2(&draft_dir).expect("draft loader");
    let draft = DFlash2DraftModel::from_loader(&draft_loader, model.dflash2_target_spec(), Some(4))
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
        "dflash2-ragged-constrained".to_string(),
        2048,
        4,
        0,
        1,
        4,
        8192,
        8,
        Some(4),
        None,
        default_scheduler_runtime_profile(SchedulerAutotuneRuntimeContext::local_default(8192)),
        estimate,
        ironmlx_lm::core::vision_input::VisionInputConfig::TextOnly,
    )
    .await
    .expect("native engine");
    assert!(state.request_execution.is_dflash2());
    // The native reasoning budget a chat request with thinking enabled gets.
    let native = state
        .tokenizer
        .native_output_decoder_config(None)
        .expect("native output config")
        .expect("Qwen3.6 has a native output dialect");
    assert!(native.reasoning_enabled, "thinking is on by default");
    let request = |text: &str| GenerateRequest {
        priority: Default::default(),
        prompt_ids: state.tokenizer.encode(text, true).expect("prompt"),
        max_new_tokens: NEW_TOKENS,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: ironmlx_lm::core::model_input::IMAGE_TOKEN_ID,
        constraint: state
            .tokenizer
            .apply_reasoning_budget(None, Some(native), NEW_TOKENS)
            .expect("reasoning budget"),
    };
    let before = state
        .health_collector
        .snapshot()
        .dflash2
        .ragged_linear
        .windows;
    let mut admitted = Vec::new();
    for text in PROMPTS {
        admitted.push(
            state
                .request_execution
                .admit(request(text))
                .await
                .unwrap_or_else(|_| panic!("admission")),
        );
    }
    let mut outputs = Vec::new();
    for mut handle in admitted {
        let mut tokens = Vec::new();
        tokio::time::timeout(Duration::from_secs(180), async {
            while let Some(event) = handle.event_rx.recv().await {
                tokens.push(event.token);
                if event.finish_reason.is_some() {
                    break;
                }
            }
        })
        .await
        .expect("generation finishes");
        assert!(!tokens.is_empty());
        outputs.push(state.tokenizer.decode(&tokens, false).expect("decode"));
    }
    let ragged = state
        .health_collector
        .snapshot()
        .dflash2
        .ragged_linear
        .windows
        - before;
    assert!(
        ragged > 0,
        "constrained rows did not decode in ragged windows"
    );
    for (prompt, output) in PROMPTS.iter().zip(&outputs) {
        assert!(
            output.contains("</think>"),
            "the reasoning budget did not close the thought for {prompt:?}: {output:?}"
        );
    }
}
