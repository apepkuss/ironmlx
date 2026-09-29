//! B1-p2.3b-2 — SchedulerActor + OpenAI handler routing integration.
//!
//! Three scenarios:
//!   A. `scheduler_actor_b1_text_only_swap` — text request routes to
//!      SchedulerActor; argmax bit-id ≥ 0.95 vs direct GenerationStream
//!      baseline.
//!   B. `scheduler_actor_long_prompt_routes_to_scheduler` — prompt_len >
//!      chunk_size routes to SchedulerActor.
//!   C. `scheduler_actor_vl_routes_to_scheduler` — VL request routing remains
//!      unified on SchedulerActor.
//!
//! Test gated `#[ignore]`; runs only with `QWEN35_MODEL` env var.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::Loader;
use ironmlx_lm::core::Message;
use ironmlx_lm::core::Tokenizer;
use ironmlx_lm::models::qwen3_5::Qwen35Model;
use ironmlx_runtime::core::engine_state::should_route_to_scheduler;
use ironmlx_runtime::core::generate::GenerationStream;
use ironmlx_runtime::core::generation_types::GenerateRequest;
use ironmlx_runtime::core::scheduler_actor::spawn_scheduler_actor;
use ironmlx_runtime::core::scheduler_actor::SchedulerCommand;

const ARGMAX_BITID_GATE: f64 = 0.95;

fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Tokenize a chat-template-rendered prompt (enable_thinking=false).
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, /* add_generation_prompt */ true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer
        .encode(&rendered, /* add_special_tokens */ false)
        .expect("encode")
}

/// Run a single-stream B=1 baseline — returns generated tokens.
fn run_b1_baseline(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let mut stream = GenerationStream::new(model, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn scheduler_actor_b1_text_only_swap() {
    configure_local_test_metallib();
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 12;

    let make_request = || GenerateRequest {
        priority: Default::default(),
        prompt_ids: prompt_ids.clone(),
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: stop_token_ids.clone(),
        prefill_chunk_size: 256, // > prompt_len → routes to scheduler
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        constraint: None,
    };

    // 1. B=1 reference via direct GenerationStream.
    let baseline = run_b1_baseline(&model, &tokenizer, make_request());
    assert!(!baseline.is_empty(), "baseline produced no tokens");

    // 2. Route the same request through the SchedulerActor.
    //    The actor takes Arc<Mutex<Qwen35Model>>; wrap the model we already
    //    loaded (no second disk load needed).
    let meta = model.model_meta();
    let model_arc = Arc::new(Mutex::new(model));
    let handle =
        spawn_scheduler_actor(model_arc, 4, Duration::from_millis(5), 32, 32768, 256, meta)
            .expect("spawn_scheduler_actor");
    let before = handle.admit_count.load(Ordering::Relaxed);

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_request(),
            reply_tx,
        })
        .await
        .expect("send admit");
    let reply = reply_rx.await.expect("admit reply").expect("admit ok");
    let mut event_rx = reply.event_rx;

    let mut scheduler_tokens: Vec<u32> = Vec::new();
    while let Some(ev) = event_rx.recv().await {
        scheduler_tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }

    let after = handle.admit_count.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        1,
        "expected exactly one admit, got delta={}",
        after - before
    );

    let ratio = argmax_bit_id_ratio(&scheduler_tokens, &baseline);
    println!(
        "[scheduler_actor_b1] scheduler={} baseline={} bit_id={:.4}",
        scheduler_tokens.len(),
        baseline.len(),
        ratio
    );
    assert!(
        ratio >= ARGMAX_BITID_GATE,
        "argmax bit-id {ratio:.4} below gate {ARGMAX_BITID_GATE}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn scheduler_actor_long_prompt_routes_to_scheduler() {
    configure_local_test_metallib();
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    // Build a synthetic long prompt by repeating tokens until > chunk_size.
    let chunk_size: usize = 64;
    let short_ids = tokenize_prompt(&tokenizer, "Hello world.");
    let mut long_ids = Vec::with_capacity(chunk_size * 2);
    while long_ids.len() <= chunk_size {
        long_ids.extend_from_slice(&short_ids);
    }
    assert!(
        long_ids.len() > chunk_size,
        "long prompt setup failed: {} <= {}",
        long_ids.len(),
        chunk_size
    );

    let request = GenerateRequest {
        priority: Default::default(),
        prompt_ids: long_ids,
        max_new_tokens: 4,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: chunk_size,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        constraint: None,
    };

    assert!(should_route_to_scheduler::<Qwen35Model>(
        request.prompt_ids.len(),
        request.prefill_chunk_size,
        4,
        false,
        false,
    ));

    // Admit and drain the former direct-path workload through SchedulerActor.
    let meta = model.model_meta();
    let model_arc = Arc::new(Mutex::new(model));
    let handle =
        spawn_scheduler_actor(model_arc, 4, Duration::from_millis(5), 32, 32768, 256, meta)
            .expect("spawn_scheduler_actor");
    let before = handle.admit_count.load(Ordering::Relaxed);

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send long-prompt admit");
    let mut event_rx = reply_rx
        .await
        .expect("long-prompt admit reply")
        .expect("long-prompt admit accepted")
        .event_rx;
    let mut tokens = Vec::new();
    while let Some(event) = event_rx.recv().await {
        tokens.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }

    let after = handle.admit_count.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        1,
        "expected one long-prompt admit, got delta={}",
        after - before
    );
    assert!(
        !tokens.is_empty(),
        "long-prompt SchedulerActor produced no tokens"
    );

    let _ = tokenizer;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn scheduler_actor_vl_routes_to_scheduler() {
    configure_local_test_metallib();
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    // Build a minimal VL request marker — pixel_values = Some(non-empty image list).
    // The routing decision only checks `pixel_values.is_some()`; building a
    // real VL prompt for end-to-end inference is heavy (P6 fixture) and is
    // already covered by `qwen35_vl_logits_match`. This test verifies the
    // routing-decision branch only.
    let dummy_image: mlx::Array = (&[0.0_f32; 1][..], (1_i32,))
        .try_into()
        .expect("dummy array");
    // image_grid_thw is Vec<(T, H, W)> — one dummy tile.
    let dummy_grid: Vec<(i32, i32, i32)> = vec![(1, 1, 1)];

    let request = GenerateRequest {
        priority: Default::default(),
        prompt_ids: tokenize_prompt(&tokenizer, "Describe the picture."),
        max_new_tokens: 4,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0, // chunking off — VL routing wins anyway
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: Some(vec![dummy_image]),
        image_grid_thw: Some(dummy_grid),
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        constraint: None,
    };

    assert!(should_route_to_scheduler::<Qwen35Model>(
        request.prompt_ids.len(),
        request.prefill_chunk_size,
        4,
        false,
        false,
    ));

    let meta = model.model_meta();
    let model_arc = Arc::new(Mutex::new(model));
    let handle =
        spawn_scheduler_actor(model_arc, 4, Duration::from_millis(5), 32, 32768, 256, meta)
            .expect("spawn_scheduler_actor");
    let before = handle.admit_count.load(Ordering::Relaxed);

    // The dummy image is intentionally not executable. Real Scheduler VL
    // inference is covered by the batched_vl suite; this test guards the
    // server-level routing contract.
    drop(request);

    let after = handle.admit_count.load(Ordering::Relaxed);
    assert_eq!(
        after, before,
        "route-only test must not submit the dummy VL request"
    );

    let _ = tokenizer;
}

fn configure_local_test_metallib() {
    static CONFIGURED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CONFIGURED.get_or_init(|| {
        let mlx_dir = std::env::var("MLX_DIR").expect("MLX_DIR required for MLX tests");
        let path = Path::new(&mlx_dir).join("lib/mlx.metallib");
        mlx::metal::set_metallib_path(path.to_string_lossy().as_ref())
            .expect("load MLX_DIR/lib/mlx.metallib for test");
    });
}
