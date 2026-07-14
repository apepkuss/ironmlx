//! Real-checkpoint production gate for Gemma4 MoE.
//!
//! Skipped unless IRONMLX_TEST_REAL_GEMMA4_MOE=1 because the 26B A4B
//! checkpoint is large.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ironmlx::core::cache::{ActiveKvOffloadConfig, PagedPrefixCacheConfig, TurboQuantKVBits};
use ironmlx::core::generate::{GenerateRequest, GenerationStream, IMAGE_TOKEN_ID};
use ironmlx::core::scheduler::StepEvent;
use ironmlx::core::server::chat_format::render_and_encode;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv,
    SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::server::vision::{
    derive_image_token_and_merge, expand_decoded_messages, DecodedMessage, DecodedPart,
};
use ironmlx::core::server::VisionInputConfig;
use ironmlx::core::{QuantMode, Sampler, Tokenizer};
use ironmlx::models::{Gemma4Config, Gemma4Model};
use ironmlx::Loader;
use tokio::sync::{mpsc, Mutex};

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join(format!(
        ".ironmlx/models/models--mlx-community--{repo}/snapshots"
    ));
    let entries = std::fs::read_dir(&base).ok()?;
    entries.flatten().find_map(|entry| {
        let path = entry.path();
        path.is_dir().then_some(path)
    })
}

fn should_run() -> bool {
    std::env::var_os("IRONMLX_TEST_REAL_GEMMA4_MOE").as_deref() == Some("1".as_ref())
}

fn coco_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/p6_qwen35_vl/coco_sample.jpg")
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("ironmlx-{name}-{}-{nanos}", std::process::id()))
}

fn assert_gemma4_moe_loader_contract(loader: &Loader) {
    let global = loader.quant_meta().expect("global quant meta");
    assert_eq!(global.mode, QuantMode::Affine);
    assert_eq!(global.bits, 4);
    assert_eq!(global.group_size, 64);

    let cfg = loader.config_raw_value();
    assert_eq!(cfg["model_type"].as_str(), Some("gemma4"));
    assert_eq!(cfg["text_config"]["enable_moe_block"].as_bool(), Some(true));
    assert_eq!(cfg["text_config"]["num_experts"].as_i64(), Some(128));
    assert_eq!(cfg["text_config"]["top_k_experts"].as_i64(), Some(8));
    assert_eq!(
        cfg["text_config"]["moe_intermediate_size"].as_i64(),
        Some(704)
    );

    let mut router_layers = 0usize;
    let mut expert_layers = 0usize;
    for layer in 0..30 {
        let router = format!("model.layers.{layer}.router.proj");
        let router_meta = loader
            .quant_meta_for(&router)
            .unwrap_or_else(|| panic!("{router}: missing quant meta"));
        assert_eq!(router_meta.mode, QuantMode::Affine, "{router}");
        assert_eq!(router_meta.bits, 8, "{router}");
        assert_eq!(router_meta.group_size, 64, "{router}");
        assert!(loader.contains(&format!("{router}.weight")));
        assert!(loader.contains(&format!("{router}.scales")));
        assert!(loader.contains(&format!("{router}.biases")));
        assert!(loader.contains(&format!("model.layers.{layer}.router.scale")));
        assert!(loader.contains(&format!("model.layers.{layer}.router.per_expert_scale")));
        router_layers += 1;

        for name in ["gate_proj", "up_proj", "down_proj"] {
            let prefix = format!("model.layers.{layer}.experts.switch_glu.{name}");
            let meta = loader
                .quant_meta_for(&prefix)
                .unwrap_or_else(|| panic!("{prefix}: missing quant meta"));
            assert_eq!(meta.mode, QuantMode::Affine, "{prefix}");
            assert_eq!(meta.bits, 4, "{prefix}");
            assert_eq!(meta.group_size, 64, "{prefix}");
            assert!(loader.contains(&format!("{prefix}.weight")));
            assert!(loader.contains(&format!("{prefix}.scales")));
            assert!(loader.contains(&format!("{prefix}.biases")));
        }
        expert_layers += 1;
    }

    assert_eq!(router_layers, 30);
    assert_eq!(expert_layers, 30);
}

fn assert_short_text_generation(model: &Gemma4Model, tokenizer: &Tokenizer) {
    let request = make_text_request(tokenizer, "Hello", 1);
    let mut stream =
        GenerationStream::new_text_only(model, tokenizer, request).expect("GenerationStream");
    let event = stream
        .next_token()
        .expect("decode next token")
        .expect("one generated event");
    assert!(
        event.finish_reason.is_some(),
        "max_new_tokens=1 must finish"
    );
}

fn make_text_request(
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> GenerateRequest {
    let prompt_ids = tokenizer
        .encode(prompt, true)
        .expect("tokenizer encode")
        .into_iter()
        .collect::<Vec<_>>();
    assert!(!prompt_ids.is_empty());

    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    }
}

fn make_image_request(
    tokenizer: &Tokenizer,
    cfg: &Gemma4Config,
    prompt: &str,
    max_new_tokens: usize,
) -> GenerateRequest {
    let vision_config = cfg
        .vision_config
        .clone()
        .expect("Gemma4 MoE checkpoint must include vision_config");
    let vision_input = VisionInputConfig::Gemma4 {
        vision_config: vision_config.clone(),
    };
    let (image_token_id, spatial_merge_size) =
        derive_image_token_and_merge(&vision_input, tokenizer);
    let image = std::fs::read(coco_fixture()).expect("read COCO fixture image");
    let messages = vec![DecodedMessage {
        role: "user".to_owned(),
        parts: vec![
            DecodedPart::Text(prompt.to_owned()),
            DecodedPart::Image(image),
        ],
    }];
    let (flat_messages, pixel_values, image_grid_thw) =
        expand_decoded_messages(messages, &vision_input).expect("expand Gemma4 MoE image prompt");
    let pixel_values = pixel_values.expect("image prompt must produce pixel_values");
    assert_eq!(pixel_values.len(), image_grid_thw.len());
    assert!(!pixel_values.is_empty());

    let prompt_ids =
        render_and_encode(tokenizer, &flat_messages, None).expect("render Gemma4 image prompt");
    assert!(
        prompt_ids
            .iter()
            .any(|&id| i32::try_from(id).ok() == Some(image_token_id)),
        "Gemma4 image prompt must contain image placeholder tokens"
    );

    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: Some(pixel_values),
        image_grid_thw: Some(image_grid_thw),
        image_spatial_merge_size: spatial_merge_size,
        image_token_id,
    }
}

fn assert_short_image_generation(model: &Gemma4Model, tokenizer: &Tokenizer, cfg: &Gemma4Config) {
    let request = make_image_request(tokenizer, cfg, "Describe the image briefly.", 1);
    let mut stream = GenerationStream::new(model, tokenizer, request)
        .expect("Gemma4 MoE image GenerationStream");
    let event = stream
        .next_token()
        .expect("decode next image token")
        .expect("one generated image event");
    assert!(
        event.finish_reason.is_some(),
        "max_new_tokens=1 must finish"
    );
}

fn with_turboquant(mut request: GenerateRequest, bits: TurboQuantKVBits) -> GenerateRequest {
    request.kv_cache_turboquant_bits = Some(bits);
    request
}

fn assert_two_generated_tokens(
    mut stream: GenerationStream<'_, Gemma4Model>,
    label: &str,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(2);
    let mut finish_reason = None;
    for step in 0..2 {
        let event = stream
            .next_token()
            .unwrap_or_else(|err| panic!("{label}: decode step {step} failed: {err:#}"))
            .unwrap_or_else(|| panic!("{label}: decode step {step} returned no event"));
        tokens.push(event.token);
        finish_reason = event.finish_reason;
    }
    assert!(
        finish_reason.is_some(),
        "{label}: max_new_tokens=2 must finish on second token"
    );
    tokens
}

async fn admit_and_drain(handle: SchedulerActorHandle, request: GenerateRequest) -> Vec<u32> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send scheduler admit");
    let reply = reply_rx
        .await
        .expect("scheduler admit reply")
        .expect("admit ok");
    let mut event_rx = reply.event_rx;
    let mut tokens = Vec::new();
    while let Some(event) = event_rx.recv().await {
        tokens.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

async fn admit_and_first_event(
    handle: SchedulerActorHandle,
    request: GenerateRequest,
    label: &str,
) -> (mpsc::UnboundedReceiver<StepEvent>, StepEvent) {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .unwrap_or_else(|err| panic!("{label}: send scheduler admit failed: {err}"));
    let reply = tokio::time::timeout(Duration::from_secs(120), reply_rx)
        .await
        .unwrap_or_else(|_| panic!("{label}: scheduler admit reply timed out"))
        .unwrap_or_else(|err| panic!("{label}: scheduler admit reply channel failed: {err}"))
        .unwrap_or_else(|err| panic!("{label}: scheduler admit failed: {err:#}"));
    let mut event_rx = reply.event_rx;
    let first = tokio::time::timeout(Duration::from_secs(120), event_rx.recv())
        .await
        .unwrap_or_else(|_| panic!("{label}: first event timed out"))
        .unwrap_or_else(|| panic!("{label}: first event channel closed"));
    (event_rx, first)
}

async fn drain_existing_rx_until_finish(
    mut event_rx: mpsc::UnboundedReceiver<StepEvent>,
    label: &str,
) -> Vec<u32> {
    let mut tokens = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(120), event_rx.recv())
            .await
            .unwrap_or_else(|_| panic!("{label}: event timed out"))
            .unwrap_or_else(|| panic!("{label}: event channel closed before finish"));
        tokens.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

async fn assert_concurrent_batch(
    handle: &SchedulerActorHandle,
    request_a: GenerateRequest,
    request_b: GenerateRequest,
    label: &str,
) {
    let admit_before = handle.admit_count.load(Ordering::Relaxed);
    let batch_before = handle.batch_count.load(Ordering::Relaxed);
    let (tokens_a, tokens_b) = tokio::join!(
        admit_and_drain(handle.clone(), request_a),
        admit_and_drain(handle.clone(), request_b)
    );

    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    let batch_delta = handle.batch_count.load(Ordering::Relaxed) - batch_before;
    eprintln!(
        "{label}: admit_delta={admit_delta} batch_delta={batch_delta} len_a={} len_b={}",
        tokens_a.len(),
        tokens_b.len()
    );
    assert_eq!(admit_delta, 2, "{label}: expected 2 admitted requests");
    assert_eq!(
        batch_delta, 1,
        "{label}: expected both requests to share one b_max=2 batch"
    );
    assert_eq!(tokens_a.len(), 1, "{label}: request A token count");
    assert_eq!(tokens_b.len(), 1, "{label}: request B token count");
}

#[test]
fn gemma4_moe_affine4_real_checkpoint_loads_and_generates_when_requested() {
    if !should_run() {
        eprintln!("IRONMLX_TEST_REAL_GEMMA4_MOE=1 not set; skipping real Gemma4 MoE gate");
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-26b-a4b-it-4bit") else {
        eprintln!("gemma-4-26b-a4b-it-4bit cache absent; skipping");
        return;
    };

    let loader =
        Loader::open_multimodal(&dir).expect("Loader::open_multimodal Gemma4 MoE affine 4bit");
    assert_gemma4_moe_loader_contract(&loader);
    let cfg = Gemma4Config::from_loader(&loader).expect("Gemma4Config::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader");
    assert!(model.config().enable_moe_block);
    assert_short_text_generation(&model, &tokenizer);
    assert_short_image_generation(&model, &tokenizer, &cfg);
}

#[test]
fn gemma4_moe_turboquant_kv_real_text_and_image_when_requested() {
    if !should_run() {
        eprintln!(
            "IRONMLX_TEST_REAL_GEMMA4_MOE=1 not set; skipping real Gemma4 MoE TurboQuant KV gate"
        );
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-26b-a4b-it-4bit") else {
        eprintln!("gemma-4-26b-a4b-it-4bit cache absent; skipping");
        return;
    };

    let loader =
        Loader::open_multimodal(&dir).expect("Loader::open_multimodal Gemma4 MoE affine 4bit");
    assert_gemma4_moe_loader_contract(&loader);
    let cfg = Gemma4Config::from_loader(&loader).expect("Gemma4Config::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader");
    assert!(model.config().enable_moe_block);

    for (label, bits) in [
        ("turbo3", TurboQuantKVBits::K3V3),
        ("turbo4", TurboQuantKVBits::K4V4),
        ("k3v4", TurboQuantKVBits::K3V4),
    ] {
        let text_request = with_turboquant(
            make_text_request(&tokenizer, "Write two short words:", 2),
            bits,
        );
        let text_stream = GenerationStream::new_text_only(&model, &tokenizer, text_request)
            .unwrap_or_else(|err| panic!("{label}: text GenerationStream failed: {err:#}"));
        let text_tokens =
            assert_two_generated_tokens(text_stream, &format!("gemma4_moe_text_{label}"));

        let image_request = with_turboquant(
            make_image_request(&tokenizer, &cfg, "Describe the image in two words.", 2),
            bits,
        );
        let image_stream = GenerationStream::new(&model, &tokenizer, image_request)
            .unwrap_or_else(|err| panic!("{label}: image GenerationStream failed: {err:#}"));
        let image_tokens =
            assert_two_generated_tokens(image_stream, &format!("gemma4_moe_image_{label}"));

        eprintln!(
            "gemma4_moe_turboquant_kv_{label}: text_tokens={text_tokens:?} image_tokens={image_tokens:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gemma4_moe_affine4_real_scheduler_bmax2_text_and_image_when_requested() {
    if !should_run() {
        eprintln!(
            "IRONMLX_TEST_REAL_GEMMA4_MOE=1 not set; skipping real Gemma4 MoE concurrency gate"
        );
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-26b-a4b-it-4bit") else {
        eprintln!("gemma-4-26b-a4b-it-4bit cache absent; skipping");
        return;
    };

    let loader =
        Loader::open_multimodal(&dir).expect("Loader::open_multimodal Gemma4 MoE affine 4bit");
    assert_gemma4_moe_loader_contract(&loader);
    let cfg = Gemma4Config::from_loader(&loader).expect("Gemma4Config::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader");
    assert!(model.config().enable_moe_block);

    let meta = model.model_meta();
    let handle = spawn_scheduler_actor(
        Arc::new(Mutex::new(model)),
        2,
        Duration::from_millis(50),
        32,
        2048,
        256,
        meta,
    )
    .expect("spawn Gemma4 MoE b_max=2 scheduler");

    assert_concurrent_batch(
        &handle,
        make_text_request(&tokenizer, "Hello", 1),
        make_text_request(&tokenizer, "Name one color.", 1),
        "gemma4_moe_text_bmax2",
    )
    .await;

    assert_concurrent_batch(
        &handle,
        make_image_request(&tokenizer, &cfg, "Describe the image briefly.", 1),
        make_image_request(&tokenizer, &cfg, "Name one visible object.", 1),
        "gemma4_moe_image_bmax2",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gemma4_moe_active_kv_offload_real_text_park_restore_when_requested() {
    if !should_run() {
        eprintln!(
            "IRONMLX_TEST_REAL_GEMMA4_MOE=1 not set; skipping real Gemma4 MoE Active KV gate"
        );
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-26b-a4b-it-4bit") else {
        eprintln!("gemma-4-26b-a4b-it-4bit cache absent; skipping");
        return;
    };

    let loader =
        Loader::open_multimodal(&dir).expect("Loader::open_multimodal Gemma4 MoE affine 4bit");
    assert_gemma4_moe_loader_contract(&loader);
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader");
    assert!(model.config().enable_moe_block);

    let prefix_root = unique_temp_dir("gemma4-moe-active-kv-prefix");
    let active_kv_root = unique_temp_dir("gemma4-moe-active-kv-offload");
    let prefix_config = PagedPrefixCacheConfig::new(&prefix_root, "gemma4-moe-active-kv", 16, 4096)
        .expect("prefix config");
    let active_kv_config = ActiveKvOffloadConfig::enabled(active_kv_root.clone());
    let meta = model.model_meta();
    let handle = spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv(
        Arc::new(Mutex::new(model)),
        1,
        Duration::from_millis(5),
        4,
        4096,
        256,
        meta,
        prefix_config,
        None,
        active_kv_config,
    )
    .expect("spawn Gemma4 MoE Active KV scheduler");

    let first_request = make_text_request(
        &tokenizer,
        "Continue this sequence with short tokens: one, two, three, four,",
        16,
    );
    let (first_rx, first_event) =
        admit_and_first_event(handle.clone(), first_request, "gemma4_moe_active_kv_first").await;
    assert!(
        first_event.finish_reason.is_none(),
        "first request must still be active before forcing Active KV park/restore"
    );

    let second_tokens = admit_and_drain(
        handle.clone(),
        make_text_request(&tokenizer, "Reply with one short word.", 1),
    )
    .await;
    assert_eq!(second_tokens.len(), 1, "second request should finish");

    let first_tail = drain_existing_rx_until_finish(first_rx, "gemma4_moe_active_kv_first").await;
    assert!(
        !first_tail.is_empty(),
        "first request should resume after Active KV restore"
    );

    let health = handle.active_kv_offload.snapshot();
    assert!(health.enabled, "Active KV must be enabled");
    assert!(
        health.active,
        "Active KV should become active after park/restore"
    );
    assert!(!health.degraded, "Active KV must not degrade: {health:?}");
    assert!(
        health.swap_out_count >= 1,
        "expected at least one Active KV swap out: {health:?}"
    );
    assert!(
        health.swap_in_count >= 1,
        "expected at least one Active KV swap in: {health:?}"
    );
    assert_eq!(health.swap_error_count, 0, "Active KV errors: {health:?}");
    assert_eq!(health.parked_requests, 0, "parked request leak: {health:?}");

    drop(handle);
    std::fs::remove_dir_all(prefix_root).ok();
    std::fs::remove_dir_all(active_kv_root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gemma4_moe_turboquant_kv_active_kv_offload_real_text_park_restore_when_requested() {
    if !should_run() {
        eprintln!(
            "IRONMLX_TEST_REAL_GEMMA4_MOE=1 not set; skipping real Gemma4 MoE TurboQuant+Active KV gate"
        );
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-26b-a4b-it-4bit") else {
        eprintln!("gemma-4-26b-a4b-it-4bit cache absent; skipping");
        return;
    };

    for (label, bits) in [
        ("turbo3", TurboQuantKVBits::K3V3),
        ("turbo4", TurboQuantKVBits::K4V4),
        ("k3v4", TurboQuantKVBits::K3V4),
    ] {
        let loader =
            Loader::open_multimodal(&dir).expect("Loader::open_multimodal Gemma4 MoE affine 4bit");
        assert_gemma4_moe_loader_contract(&loader);
        let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
        let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader");
        assert!(model.config().enable_moe_block);

        let prefix_root = unique_temp_dir(&format!("gemma4-moe-{label}-tq-active-kv-prefix"));
        let active_kv_root = unique_temp_dir(&format!("gemma4-moe-{label}-tq-active-kv-offload"));
        let prefix_config = PagedPrefixCacheConfig::new(
            &prefix_root,
            &format!("gemma4-moe-{label}-tq-active-kv"),
            16,
            4096,
        )
        .expect("prefix config");
        let active_kv_config = ActiveKvOffloadConfig::enabled(active_kv_root.clone());
        let meta = model.model_meta();
        let handle = spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv(
            Arc::new(Mutex::new(model)),
            1,
            Duration::from_millis(5),
            4,
            4096,
            256,
            meta,
            prefix_config,
            None,
            active_kv_config,
        )
        .unwrap_or_else(|err| {
            panic!("{label}: spawn Gemma4 MoE TurboQuant+Active KV scheduler: {err:#}")
        });

        let first_request = with_turboquant(
            make_text_request(
                &tokenizer,
                &format!(
                    "Continue this {label} TurboQuant sequence with short tokens: one, two, three, four,"
                ),
                16,
            ),
            bits,
        );
        let (first_rx, first_event) = admit_and_first_event(
            handle.clone(),
            first_request,
            &format!("gemma4_moe_{label}_tq_active_kv_first"),
        )
        .await;
        assert!(
            first_event.finish_reason.is_none(),
            "{label}: first request must still be active before forcing Active KV park/restore"
        );

        let second_tokens = admit_and_drain(
            handle.clone(),
            with_turboquant(
                make_text_request(&tokenizer, "Reply with one short word.", 1),
                bits,
            ),
        )
        .await;
        assert_eq!(
            second_tokens.len(),
            1,
            "{label}: second request should finish"
        );

        let first_tail = drain_existing_rx_until_finish(
            first_rx,
            &format!("gemma4_moe_{label}_tq_active_kv_first"),
        )
        .await;
        assert!(
            !first_tail.is_empty(),
            "{label}: first request should resume after Active KV restore"
        );

        let health = handle.active_kv_offload.snapshot();
        assert!(health.enabled, "{label}: Active KV must be enabled");
        assert!(
            health.active,
            "{label}: Active KV should become active after park/restore"
        );
        assert!(
            !health.degraded,
            "{label}: Active KV must not degrade: {health:?}"
        );
        assert!(
            health.swap_out_count >= 1,
            "{label}: expected at least one Active KV swap out: {health:?}"
        );
        assert!(
            health.swap_in_count >= 1,
            "{label}: expected at least one Active KV swap in: {health:?}"
        );
        assert_eq!(
            health.swap_error_count, 0,
            "{label}: Active KV errors: {health:?}"
        );
        assert_eq!(
            health.parked_requests, 0,
            "{label}: parked request leak: {health:?}"
        );

        drop(handle);
        std::fs::remove_dir_all(prefix_root).ok();
        std::fs::remove_dir_all(active_kv_root).ok();
    }
}
