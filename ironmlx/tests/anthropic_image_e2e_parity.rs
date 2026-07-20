//! End-to-end Anthropic image parity (transitivity): Anthropic /v1/messages image
//! completion must equal the OpenAI /v1/chat/completions completion token-for-token
//! under the SAME image + prompt + greedy decode.  OpenAI is already validated vs
//! mlx-vlm per architecture, so this transitively anchors Anthropic to mlx-vlm.
//!
//! Env vars (each a checkpoint snapshot dir):
//!   QWEN35_VL_DENSE_MODEL, QWEN35_VL_MOE_MODEL, GEMMA4_MODEL, MINICPMV46_MODEL
//!
//! Run (export MLX env first), one architecture at a time:
//!   source ~/.local/mlx/mlx-env.sh && QWEN35_VL_DENSE_MODEL=... \
//!     cargo test --release -p ironmlx --test anthropic_image_e2e_parity \
//!       -- --ignored e2e_qwen35_vl_dense --nocapture
//!
//! ## Verification status (2026-06-02, this branch) — ALL 4 ARCHITECTURES PASS
//!
//! All four real VLM architectures pass e2e parity (Anthropic == OpenAI,
//! token-identical, non-empty) under both endpoints: Qwen3.5-VL dense,
//! Qwen3.5-VL MoE, Gemma4, MiniCPM-V-4.6.
//!
//! Two PRE-EXISTING serve-path backend bugs (NOT introduced by the Anthropic
//! image feature — only manifest on the multi-threaded scheduler serve path, so
//! the feature surfaced them) were root-caused and fixed in this branch:
//!   * MiniCPM-V (fixed in c6ae50e): `MiniCpmV46Vision::from_loader` never
//!     eagerly eval'd the SigLIP weights (unlike qwen's `VisionTower`), so the
//!     lazy weight graph stayed tagged with the construction thread's MLX stream;
//!     the scheduler-actor prefill thread then failed `to_vec(vision_embeds)`
//!     with "There is no Stream(gpu, 1) in current thread" → scheduler poisoned.
//!     Fix: mirror qwen — add `collect_weights` + `MiniCpmV46Vision::eval_weights`.
//!   * Gemma4 (fixed in 5ddef44): the b==1 split-prefill ran a prefix chunk then
//!     a last-token chunk against the SAME KVCache, dropping the prefix hidden
//!     without `eval` → the prefix's lazy cache writes were never committed → the
//!     last chunk's read-after-write fused into one lazy graph MLX intermittently
//!     miscomputed to all-NaN logits (argmax 0 = <pad> → empty). ~35-55%
//!     intermittent, all-KVCache models only (qwen/minicpmv hybrid cache dodged
//!     it). Fix: `eval` barrier on the prefix hidden at both VL + text split sites.
//!   Both bugs were in feature-predating model/scheduler code; the OpenAI endpoint
//!   hit them identically, confirming the Anthropic wire layer was always sound.

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine;
use ironmlx::core::model::Model;
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use ironmlx::core::server::{self, VisionInputConfig};
use ironmlx::core::{Loader, Tokenizer};

// ---------------------------------------------------------------------------
// Image fixture
// ---------------------------------------------------------------------------

/// Load the shared COCO fixture and base64-encode it as a JPEG.
/// The path matches the fixture used by all p6 and minicpmv46 VL generators.
fn coco_b64() -> String {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/p6_qwen35_vl/coco_sample.jpg"
    ))
    .expect("read coco_sample.jpg");
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ---------------------------------------------------------------------------
// Port allocation
// ---------------------------------------------------------------------------

async fn alloc_port() -> u16 {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

// ---------------------------------------------------------------------------
// Server boot helper
// ---------------------------------------------------------------------------

fn scheduler_profile() -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "e2e".to_string(),
        hardware_label: "test-host".to_string(),
        runtime_context:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(32768),
        config: SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata:
            ironmlx::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                1811606400000,
            ),
    }
}

fn boot<M>(
    model: M,
    tokenizer: Tokenizer,
    port: u16,
    vision: Option<VisionInputConfig>,
) -> tokio::task::JoinHandle<anyhow::Result<()>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    tokio::spawn(async move {
        server::serve(
            model,
            tokenizer,
            "e2e".to_string(),
            "127.0.0.1",
            port,
            /* prefill_chunk_size */ 2048,
            /* b_max */ 1,
            /* admission_deadline_ms */ 5,
            /* admission_queue_max */ 32,
            /* max_cache_cap */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            /* kv_cache_turboquant_bits */ None,
            /* paged_prefix_cache */ None,
            /* prefix_lru_cache */ None,
            /* active_kv_offload */ ironmlx::core::cache::ActiveKvOffloadConfig::disabled(),
            scheduler_profile(),
            /* scheduler_autotune_report */ false,
            /* vision_input_override */ vision,
            /* static_memory_estimate */ Default::default(),
        )
        .await
    })
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .no_proxy()
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Request body builders
// ---------------------------------------------------------------------------

fn anthropic_body(b64: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "e2e",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64
                }}
            ]
        }]
    })
}

fn openai_body(b64: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "e2e",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/jpeg;base64,{b64}")
                }}
            ]
        }]
    })
}

// ---------------------------------------------------------------------------
// Response text extractors
// ---------------------------------------------------------------------------

async fn anthropic_text(c: &reqwest::Client, port: u16, b64: &str) -> String {
    let r: serde_json::Value = c
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .json(&anthropic_body(b64))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    r["content"][0]["text"].as_str().unwrap_or("").to_string()
}

async fn openai_text(c: &reqwest::Client, port: u16, b64: &str) -> String {
    let r: serde_json::Value = c
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&openai_body(b64))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    r["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// Server-ready poll helper
// ---------------------------------------------------------------------------

async fn wait_ready(c: &reqwest::Client, port: u16) {
    for _ in 0..100 {
        if let Ok(r) = c
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
        {
            if r.status() == 200 {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("server on port {port} not ready after 10s");
}

// ---------------------------------------------------------------------------
// Transitivity assertion
// ---------------------------------------------------------------------------

async fn assert_transitive_parity(port: u16) {
    let c = client();
    // Poll /health until the server is accepting connections.
    wait_ready(&c, port).await;
    let b64 = coco_b64();
    let a = anthropic_text(&c, port, &b64).await;
    let o = openai_text(&c, port, &b64).await;
    println!("anthropic: {a:?}");
    println!("openai:    {o:?}");
    assert!(!a.is_empty(), "anthropic completion must not be empty");
    assert!(!o.is_empty(), "openai completion must not be empty");
    assert_eq!(
        a, o,
        "Anthropic and OpenAI completions diverged for the same image+prompt"
    );
}

// ---------------------------------------------------------------------------
// Qwen3.5-4B dense (smallest/fastest — primary harness validation)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_VL_DENSE_MODEL checkpoint"]
async fn e2e_qwen35_vl_dense() {
    let dir = PathBuf::from(std::env::var("QWEN35_VL_DENSE_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::Qwen35Model::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    // None → server falls back to VisionInputConfig::Qwen{spatial_merge_size from ModelMeta}
    let _s = boot(model, tok, port, None);
    assert_transitive_parity(port).await;
}

// ---------------------------------------------------------------------------
// Qwen3.5-35B MoE
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_VL_MOE_MODEL checkpoint"]
async fn e2e_qwen35_vl_moe() {
    let dir = PathBuf::from(std::env::var("QWEN35_VL_MOE_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::Qwen35MoeModel::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let _s = boot(model, tok, port, None);
    assert_transitive_parity(port).await;
}

// ---------------------------------------------------------------------------
// Gemma4
// ---------------------------------------------------------------------------

// Fixed in 5ddef44 (scheduler b==1 split-prefill eval barrier). Verified 20/20
// non-empty on both endpoints with a real COCO description, 0 NaN. See module docs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MODEL checkpoint"]
async fn e2e_gemma4() {
    let dir = PathBuf::from(std::env::var("GEMMA4_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let cfg = ironmlx::models::Gemma4Config::from_loader(&loader).unwrap();
    let vision = cfg
        .vision_config
        .map(|vision_config| VisionInputConfig::Gemma4 { vision_config });
    let model = ironmlx::models::Gemma4Model::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let _s = boot(model, tok, port, vision);
    assert_transitive_parity(port).await;
}

// ---------------------------------------------------------------------------
// MiniCPM-V-4.6  (transitivity check)
// ---------------------------------------------------------------------------

// Fixed in c6ae50e (eager eval of SigLIP vision weights at construction). Verified
// both endpoints return a real COCO description, token-identical. See module docs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MINICPMV46_MODEL checkpoint"]
async fn e2e_minicpmv46() {
    let dir = PathBuf::from(std::env::var("MINICPMV46_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let vision = Some(VisionInputConfig::MiniCpmV46 {
        spatial_merge_size: 4,
    });
    let _s = boot(model, tok, port, vision);
    assert_transitive_parity(port).await;
}
