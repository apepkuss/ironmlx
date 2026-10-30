//! Confidence verify (M5 profile setting `DFLASH2_CONFIDENCE_VERIFY`) on the
//! real Qwen3.6 MoE DFlash2 pair: verifying only the confident draft prefix
//! emits the same token ids as the window-cost budget policy.

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

const PROMPTS: [&str; 3] = [
    "Write a Rust function that parses a semantic version string into its three numeric parts and \
     explain each step.",
    "Explain why the sky is blue, then list three common misconceptions about it.",
    "Summarize the plot of a heist film in which the plan goes wrong halfway through.",
];
const NEW_TOKENS: usize = 96;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires QWEN36_MOE_MODEL and DFLASH2_MOE_MODEL real checkpoints"]
async fn confidence_verify_emits_the_window_cost_policy_tokens() {
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
        "dflash2-confidence-verify".to_string(),
        2048,
        1,
        0,
        1,
        1,
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
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|text| state.tokenizer.encode(text, true).expect("prompt"))
        .collect();

    // The setting is read when a request's stream is created, so the two
    // policies alternate on one engine: 0 keeps the window-cost policy.
    let mut outputs: Vec<Vec<Vec<u32>>> = Vec::new();
    for value in ["0", "500", "0", "500"] {
        std::env::set_var(m5_profile::settings::DFLASH2_CONFIDENCE_VERIFY, value);
        let mut run = Vec::new();
        for prompt_ids in &prompts {
            let mut admitted = state
                .request_execution
                .admit(GenerateRequest {
                    priority: Default::default(),
                    prompt_ids: prompt_ids.clone(),
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
                    constraint: None,
                })
                .await
                .unwrap_or_else(|_| panic!("admission"));
            let mut tokens = Vec::new();
            tokio::time::timeout(Duration::from_secs(120), async {
                while let Some(event) = admitted.event_rx.recv().await {
                    tokens.push(event.token);
                    if event.finish_reason.is_some() {
                        break;
                    }
                }
            })
            .await
            .expect("generation finishes");
            assert!(!tokens.is_empty(), "no tokens for {value}");
            run.push(tokens);
        }
        outputs.push(run);
    }
    std::env::remove_var(m5_profile::settings::DFLASH2_CONFIDENCE_VERIFY);
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let reference = &outputs[0][i];
        for (run, value) in outputs.iter().zip(["0", "500", "0", "500"]) {
            assert_eq!(
                &run[i], reference,
                "token ids differ with DFLASH2_CONFIDENCE_VERIFY={value} for prompt {prompt:?}"
            );
        }
    }
}
