//! B1-p2.3e.1b — vectorized configured sampler perf gate.
//!
//! 4-row concurrent admit batch with all rows using
//! `temperature=0.7, top_p=0.9, repetition_penalty=1.1` — guaranteed
//! to hit configured_pipeline (not the 3e.1a fast path). Measures
//! per-row median inter-token gap, asserts:
//!   - per-row medians within 2× (batched-step lockstep)
//!   - max median ≤ 250 ms (configured pipeline budget; 3e.1a fast
//!     path was 64.7 ms argmax; configured adds ~50-100 ms for the
//!     7 ops + categorical)

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let loader = Loader::open(Path::new(&model_dir)).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_configured_request(
    prompt_ids: Vec<u32>,
    max_new: usize,
    stop: Vec<u32>,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_repetition_penalty(1.1)
            .with_seed(42),
        stop_token_ids: stop,
        prefill_chunk_size: 128,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3e_1b_configured_decode_speedup() {
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();
    let handle = spawn_scheduler_actor(
        model.clone(),
        4,
        Duration::from_millis(5),
        32,
        32768,
        256,
        meta,
    )
    .expect("spawn");

    let prompts = [
        "Write a short essay on the history of Italian cuisine.",
        "Explain the principles of quantum entanglement in simple terms.",
        "Describe the most important inventions of the 20th century.",
        "Tell a creative short story about a robot who learns to paint.",
    ];

    let mut tasks: Vec<tokio::task::JoinHandle<Vec<Instant>>> = Vec::new();
    for p in prompts {
        let ids = tokenize_prompt(&tokenizer, p);
        let req = make_configured_request(ids, 50, stop_tokens.clone());
        let h = handle.clone();
        tasks.push(tokio::spawn(async move {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            h.cmd_tx
                .send(SchedulerCommand::Admit {
                    request: req,
                    reply_tx,
                })
                .await
                .expect("send");
            let reply = reply_rx.await.expect("reply").expect("ok");
            let mut event_rx = reply.event_rx;
            let mut stamps: Vec<Instant> = Vec::new();
            while let Some(ev) = event_rx.recv().await {
                stamps.push(Instant::now());
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            stamps
        }));
    }

    let mut all_stamps: Vec<Vec<Instant>> = Vec::new();
    for t in tasks {
        let s = tokio::time::timeout(Duration::from_secs(240), t)
            .await
            .expect("timeout")
            .expect("join");
        assert!(s.len() >= 10, "row needs ≥ 10 tokens; got {}", s.len());
        all_stamps.push(s);
    }

    let mut all_medians: Vec<Duration> = Vec::new();
    for stamps in &all_stamps {
        let mut gaps: Vec<Duration> = (2..stamps.len())
            .map(|i| stamps[i].duration_since(stamps[i - 1]))
            .collect();
        gaps.sort();
        all_medians.push(gaps[gaps.len() / 2]);
    }

    let max_median = all_medians.iter().max().copied().unwrap();
    let min_median = all_medians.iter().min().copied().unwrap();

    eprintln!(
        "[3e.1b perf gate] per-row medians: {:?} | max={:?} min={:?} ratio={:.2}x",
        all_medians,
        max_median,
        min_median,
        max_median.as_secs_f64() / min_median.as_secs_f64().max(1e-9)
    );

    assert!(
        max_median <= min_median * 2,
        "per-row median spread > 2×: {:?} (lockstep broken?)",
        all_medians
    );

    assert!(
        max_median <= Duration::from_millis(250),
        "max median {max_median:?} exceeds 250 ms — configured_pipeline regression?"
    );

    drop(handle);
}
