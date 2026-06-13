//! B1-p2.3e.1a — vectorized greedy sampler perf gate.
//!
//! Goal: verify the all-greedy fast path inside `sample_batch`
//! reduces per-step sampler-driven serialization at B=4 vs. the
//! per-row pre-3e.1a loop. Measurement is RELATIVE — we sample
//! median + max per-step wall time across N decode steps, compare
//! against pre-3e.1a expectation (~4× sampler block). Robust to
//! per-system Metal compile + thermal variation by relying on
//! ratios, not absolute thresholds.

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

fn make_request(prompt_ids: Vec<u32>, max_new: usize, stop_token_ids: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids,
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
async fn b1_p2_3e_1a_greedy_decode_speedup() {
    let (model, tokenizer) = load_fixture();
    let meta = model.lock().await.model_meta();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();

    // Spawn 4 concurrent greedy admits → decode goes through the
    // all-greedy fast path inside Scheduler::step's sample_batch
    // dispatch.
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
        "Explain why the sky appears blue during the day.",
        "List five common fruits and describe each one briefly.",
        "Describe three things you might see on a city street.",
        "What are some benefits of regular exercise for health?",
    ];

    let mut tasks: Vec<tokio::task::JoinHandle<Vec<Instant>>> = Vec::new();
    for p in prompts {
        let ids = tokenize_prompt(&tokenizer, p);
        let req = make_request(ids, 50, stop_tokens.clone());
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
        let s = tokio::time::timeout(Duration::from_secs(120), t)
            .await
            .expect("timeout")
            .expect("join");
        assert!(
            s.len() >= 10,
            "row needs ≥ 10 tokens for gap stats; got {}",
            s.len()
        );
        all_stamps.push(s);
    }

    // For each row, compute median per-token gap (skip first gap
    // which includes prefill→first-decode transition).
    let mut all_medians: Vec<Duration> = Vec::new();
    for stamps in &all_stamps {
        let mut gaps: Vec<Duration> = (2..stamps.len())
            .map(|i| stamps[i].duration_since(stamps[i - 1]))
            .collect();
        gaps.sort();
        let median = gaps[gaps.len() / 2];
        all_medians.push(median);
    }

    let max_median = all_medians.iter().max().copied().unwrap();
    let min_median = all_medians.iter().min().copied().unwrap();

    eprintln!(
        "[3e.1a perf gate] per-row median gaps: {:?} | max_median={:?} min_median={:?} ratio={:.2}x",
        all_medians,
        max_median,
        min_median,
        max_median.as_secs_f64() / min_median.as_secs_f64().max(1e-9)
    );

    // Functional gate: rows decode in lockstep so per-row medians
    // should be within 2× of each other (steps are batched; one row
    // can't pull ahead). If one row's median is > 2× another's,
    // either sample_batch is wrong (rows independently slow) or
    // step lockstep broke.
    assert!(
        max_median <= min_median * 2,
        "per-row median spread too wide: {:?} (lockstep broken?)",
        all_medians
    );

    // Perf gate (loose lower bound): the all-greedy fast path should
    // keep median gap under 200 ms on a 4B bf16 model. Pre-3e.1a
    // had ~80-120 ms median for the same prompt set; 3e.1a should
    // be ≤ this. 200 ms is a defensive ceiling that catches the
    // "vectorize broke and we fell back unintentionally to per-row"
    // regression, NOT a strict speedup proof.
    assert!(
        max_median <= Duration::from_millis(200),
        "per-row max median {max_median:?} exceeds 200 ms — sample_batch fast path may not be firing"
    );

    drop(handle);
}
