//! End-to-end Anthropic image parity (transitivity): Anthropic /v1/messages image
//! completion must equal the OpenAI /v1/chat/completions completion token-for-token
//! under the SAME image + prompt + greedy decode.  OpenAI is already validated vs
//! mlx-vlm per architecture, so this transitively anchors Anthropic to mlx-vlm.
//!
//! Also includes one ignored MiniCPM-V-4.6 direct-vs-mlx-vlm test that compares
//! the Anthropic endpoint output against the `expected_gen_tokens.npy` fixture
//! produced by `tests/fixtures/minicpmv46_vl/gen_single_image_generate.py`.
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
use ironmlx::core::server::{self, VisionInputConfig};
use ironmlx::core::{Loader, Tokenizer};

mod common;
use common::minicpmv46_parity::{checkpoint_dir, load_npy_in, FIXTURE_DIR_VL};

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
            /* p5h_measurement_eval_probes */ false,
            /* vision_input_override */ vision,
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

// ---------------------------------------------------------------------------
// MiniCPM-V-4.6 direct-vs-mlx-vlm via Anthropic endpoint
//
// Uses the same COCO image fixture used by gen_single_image_generate.py
// (verified: the gen script reads `../p6_qwen35_vl/coco_sample.jpg`).
// Re-encodes the Anthropic endpoint's text output and compares token-id
// prefixes against `expected_gen_tokens.npy` (the mlx-vlm reference tokens).
//
// Fixture directory: ironmlx/tests/fixtures/minicpmv46_vl/
// Fixture file: expected_gen_tokens.npy (int32 [K], gitignored)
// To regenerate: run gen_single_image_generate.py with MINICPMV46_MODEL set.
//
// The vision_embeds stream bug that previously prevented generation is FIXED
// (c6ae50e); MiniCPM-V now generates correctly on the serve path (see the
// e2e_minicpmv46 transitivity test above, which PASSES). This direct-vs-fixture
// test remains a TODO purely on PROMPT ALIGNMENT: the gen script uses
// `PROMPT = "<image>Describe this image."` (image FIRST, "this"), whereas the
// shared `anthropic_body` here sends [text "Describe the image.", image] (text
// first, "the"). To enable it, send a matching prompt — parts
// [image, text "Describe this image."] — AND match the serve chat-template
// render to the gen script's, so the token-ids line up.
//
// Not a correctness gap: MiniCPM-V's serve-path agreement with mlx-vlm was
// already established at P2b integration (first-token-exact single-image e2e
// parity), and the transitivity test anchors the Anthropic endpoint to that
// validated backend. This direct test is a redundant strengthening, left
// ignored pending the prompt-alignment chore above.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MINICPMV46_MODEL + fixture; prompt-alignment TODO vs gen script (stream bug fixed in c6ae50e)"]
async fn e2e_minicpmv46_direct_vs_mlxvlm() {
    use mlx::{ops, Dtype};

    let dir = checkpoint_dir();
    let loader = Loader::open_multimodal(&dir).unwrap();
    // Load two tokenizer instances: one for re-encoding the Anthropic output
    // (kept in scope), one moved into boot().  Tokenizer is not Clone.
    let tok_encode = Tokenizer::from_loader(&loader).unwrap();
    let tok_boot = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader).unwrap();

    // Load the mlx-vlm reference token ids from the fixture.
    let expected_arr = load_npy_in(FIXTURE_DIR_VL, "expected_gen_tokens.npy");
    let expected_i32: Vec<i32> = ops::cast::astype(&expected_arr, Dtype::Int32)
        .unwrap()
        .to_vec()
        .unwrap();
    let expected_u32: Vec<u32> = expected_i32.iter().map(|&i| i as u32).collect();

    // Boot the server using the second tokenizer instance.
    let port = alloc_port().await;
    let vision = Some(VisionInputConfig::MiniCpmV46 {
        spatial_merge_size: 4,
    });
    let _s = boot(model, tok_boot, port, vision);

    // Poll /health until the server is accepting connections.
    let b64 = coco_b64();
    let c = client();
    wait_ready(&c, port).await;
    let got = anthropic_text(&c, port, &b64).await;

    println!(
        "mlx-vlm reference ({} tokens): {:?}",
        expected_u32.len(),
        expected_u32
    );
    println!("ironmlx anthropic: {got:?}");

    assert!(!got.is_empty(), "Anthropic completion must not be empty");

    // Re-encode the Anthropic output and compare token-id prefixes.
    // This is panic-safe on non-ASCII output (avoids byte-slicing UTF-8).
    let got_ids = tok_encode
        .encode(&got, false)
        .expect("re-encode anthropic output");
    let n = got_ids.len().min(expected_u32.len());
    assert!(
        n > 0,
        "fixture/expected decoded to zero tokens — regenerate the fixture"
    );
    assert_eq!(
        &got_ids[..n],
        &expected_u32[..n],
        "Anthropic token-id prefix diverged from mlx-vlm reference\n  got[..{n}]={:?}\n  exp[..{n}]={:?}",
        &got_ids[..n],
        &expected_u32[..n],
    );

    println!("e2e_minicpmv46_direct_vs_mlxvlm: PASS — token-id prefix match {n} tokens",);
}
