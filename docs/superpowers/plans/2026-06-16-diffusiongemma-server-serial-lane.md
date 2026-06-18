# DiffusionGemma Server Serial Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ironmlx serve` support for DiffusionGemma text-only and image-text-to-text requests through a dedicated serial block-diffusion lane.

**Architecture:** DiffusionGemma will not implement `DenseVlMethods` and will not enter the causal `SchedulerActor`. `serve.rs` will branch to a new concrete `server::diffusion_gemma` runtime, with OpenAI and Anthropic non-streaming handlers that reuse existing message decoding, chat template rendering, and vision preprocessing before calling `generate_text` or `generate_image_text` behind a model mutex.

**Tech Stack:** Rust, axum, tokio, MLX Rust bindings, existing ironmlx tokenizer/loader/server modules.

---

## File Structure

- Create `ironmlx/src/core/server/diffusion_gemma.rs`: concrete AppState, routes, OpenAI/Anthropic non-streaming handlers, event aggregation helpers, unsupported streaming response.
- Modify `ironmlx/src/core/server/mod.rs`: export the new module and add `VisionInputConfig::DiffusionGemma`.
- Modify `ironmlx/src/core/server/vision.rs`: route DiffusionGemma through Gemma4-compatible placeholder/preprocess and derive image token id from DiffusionGemma config.
- Modify `ironmlx/src/core/server/anthropic.rs`: expose the already-existing Anthropic request/decode pieces to sibling server modules.
- Modify `ironmlx/src/cli/serve.rs`: remove the DiffusionGemma hard reject and branch to the new server lane.
- Add or extend tests in the touched modules.

## Task 1: Add DiffusionGemma VisionInputConfig

**Files:**
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/src/core/server/vision.rs`

- [ ] **Step 1: Write the failing test**

Add this case to `vision::tests::derive_image_token_and_merge_returns_correct_merge_size`:

```rust
let vision_config = crate::models::gemma4::Gemma4VisionConfig {
    model_type: "gemma4_vision".to_string(),
    hidden_size: 1152,
    intermediate_size: 4304,
    num_hidden_layers: 27,
    num_attention_heads: 16,
    num_key_value_heads: 16,
    head_dim: 72,
    global_head_dim: None,
    hidden_activation: "gelu_pytorch_tanh".to_string(),
    rms_norm_eps: 1e-6,
    max_position_embeddings: 8192,
    attention_bias: false,
    attention_dropout: 0.0,
    layer_types: None,
    rope_parameters: None,
    default_output_length: 256,
    patch_size: 14,
    position_embedding_size: 8192,
    pooling_kernel_size: 5,
    use_clipped_linears: false,
    standardize: false,
};
let (tok_id, merge) = derive_image_token_and_merge(
    &VisionInputConfig::DiffusionGemma {
        vision_config,
        image_token_id: Some(77_777),
    },
    &tok,
);
assert_eq!(merge, 5, "DiffusionGemma pooling_kernel_size");
assert_eq!(tok_id, 88_888, "DiffusionGemma tokenizer image token id wins");
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx derive_image_token_and_merge_returns_correct_merge_size -- --nocapture
```

Expected: compile failure because `VisionInputConfig::DiffusionGemma` does not exist.

- [ ] **Step 3: Implement the variant and matches**

Add this enum variant:

```rust
DiffusionGemma {
    vision_config: crate::models::gemma4::Gemma4VisionConfig,
    image_token_id: Option<i32>,
},
```

Update all `VisionInputConfig` matches in `vision.rs` so DiffusionGemma uses the Gemma4 placeholder and `gemma4::image_processor::preprocess`:

```rust
VisionInputConfig::DiffusionGemma {
    vision_config,
    image_token_id,
} => (
    tokenizer
        .token_to_id("<|image|>")
        .map(|id| id as i32)
        .or(*image_token_id)
        .unwrap_or(258_880),
    vision_config.pooling_kernel_size,
),
```

- [ ] **Step 4: Run the test and verify pass**

Run the same cargo test command. Expected: PASS.

## Task 2: Add Pure DiffusionGemma Server Helpers

**Files:**
- Create: `ironmlx/src/core/server/diffusion_gemma.rs`
- Modify: `ironmlx/src/core/server/mod.rs`

- [ ] **Step 1: Write failing unit tests**

Create tests for event aggregation and Anthropic stop reason mapping:

```rust
#[test]
fn collect_events_joins_text_and_uses_finish_reason() {
    let events = vec![
        DiffusionGemmaGenerateEvent {
            token: 1,
            text: "hel".to_string(),
            finish_reason: None,
        },
        DiffusionGemmaGenerateEvent {
            token: 2,
            text: "lo".to_string(),
            finish_reason: None,
        },
        DiffusionGemmaGenerateEvent {
            token: 0,
            text: String::new(),
            finish_reason: Some("length"),
        },
    ];

    let completion = collect_events(events, "stop");
    assert_eq!(completion.content, "hello");
    assert_eq!(completion.finish_reason, "length");
    assert_eq!(completion.completion_tokens, 3);
}

#[test]
fn anthropic_stop_reason_maps_openai_reasons() {
    assert_eq!(anthropic_stop_reason("stop"), "end_turn");
    assert_eq!(anthropic_stop_reason("length"), "max_tokens");
    assert_eq!(anthropic_stop_reason("other"), "other");
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx diffusion_gemma::tests::collect_events -- --nocapture
```

Expected: compile failure because the module/helper functions are missing.

- [ ] **Step 3: Implement helpers**

Create the module with:

```rust
struct CompletionParts {
    content: String,
    finish_reason: &'static str,
    completion_tokens: u32,
}

fn collect_events(
    events: Vec<crate::models::diffusion_gemma::generation::DiffusionGemmaGenerateEvent>,
    default_finish: &'static str,
) -> CompletionParts {
    let mut content = String::new();
    let mut finish_reason = default_finish;
    let mut completion_tokens = 0_u32;
    for event in events {
        content.push_str(&event.text);
        completion_tokens += 1;
        if let Some(reason) = event.finish_reason {
            finish_reason = reason;
            break;
        }
    }
    CompletionParts {
        content,
        finish_reason,
        completion_tokens,
    }
}

fn anthropic_stop_reason(reason: &'static str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        other => other,
    }
}
```

Add `pub mod diffusion_gemma;` to `server/mod.rs`.

- [ ] **Step 4: Run tests and verify pass**

Run the same cargo test command. Expected: PASS.

## Task 3: Implement DiffusionGemma HTTP Handlers

**Files:**
- Modify: `ironmlx/src/core/server/diffusion_gemma.rs`
- Modify: `ironmlx/src/core/server/anthropic.rs`

- [ ] **Step 1: Expose Anthropic decode pieces**

Change:

```rust
struct AnthropicMessage
fn decode_anthropic_messages(...)
```

to:

```rust
pub(crate) struct AnthropicMessage
pub(crate) fn decode_anthropic_messages(...)
```

and make `MessagesRequest.messages` `pub(crate)`.

- [ ] **Step 2: Add DiffusionGemma state and request preparation**

Implement:

```rust
#[derive(Clone)]
pub struct DiffusionGemmaAppState {
    pub model: Arc<Mutex<DiffusionGemmaModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub generation_config: DiffusionGemmaGenerationConfig,
    pub model_id: String,
    pub vision_input: VisionInputConfig,
}
```

Add preparation helpers that return `prompt_ids`, optional image tensors, `image_grid_thw`, and image token id. OpenAI uses `openai::decode_openai_messages`; Anthropic uses `anthropic::decode_anthropic_messages`.

- [ ] **Step 3: Add `stream: true` rejection**

At the top of both handlers:

```rust
if req.stream {
    return unsupported_streaming_response();
}
```

The response body must be JSON with `unsupported_feature`.

- [ ] **Step 4: Add blocking generation path**

In a `spawn_blocking` closure, lock the model and call:

```rust
crate::models::diffusion_gemma::generate_text(...)
crate::models::diffusion_gemma::generate_image_text(...)
```

Choose image generation when `pixel_values` and `image_grid_thw` are both present.

- [ ] **Step 5: Return endpoint-compatible JSON**

OpenAI response uses:

```json
{"object":"chat.completion","choices":[{"message":{"role":"assistant","content":"..."}}]}
```

Anthropic response uses:

```json
{"type":"message","role":"assistant","content":[{"type":"text","text":"..."}]}
```

## Task 4: Add DiffusionGemma Server Entrypoint

**Files:**
- Modify: `ironmlx/src/core/server/diffusion_gemma.rs`
- Modify: `ironmlx/src/cli/serve.rs`

- [ ] **Step 1: Implement `serve_diffusion_gemma`**

Build an axum router with:

```rust
Router::new()
    .route("/health", get(|| async { "ok" }))
    .route("/healthz", get(|| async { axum::Json(serde_json::json!({"status":"ok"})) }))
    .route("/v1/chat/completions", post(openai_chat_completions))
    .route("/v1/messages", post(anthropic_messages))
    .with_state(state)
```

- [ ] **Step 2: Wire CLI branch**

In `serve.rs`, remove the hard reject and in the DiffusionGemma match arm:

```rust
let cfg = crate::models::DiffusionGemmaConfig::from_loader(&loader)?;
let vision_config = cfg
    .vision_config
    .clone()
    .ok_or_else(|| anyhow::anyhow!("DiffusionGemma config has no vision_config"))?;
let vision_input = server::VisionInputConfig::DiffusionGemma {
    vision_config,
    image_token_id: cfg.image_token_id,
};
let generation_config =
    crate::models::DiffusionGemmaGenerationConfig::from_loader(&loader)?;
let model = crate::models::DiffusionGemmaModel::from_loader(&loader)?;
serve_diffusion_gemma_model(model, tokenizer, generation_config, &args, vision_input)
```

Add a small helper mirroring `serve_with_model` but without scheduler args.

## Task 5: Verification

**Files:**
- No source changes unless tests reveal defects.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
```

- [ ] **Step 2: Focused tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx diffusion_gemma -- --nocapture
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx server::vision -- --nocapture
```

- [ ] **Step 3: Lints and build**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

- [ ] **Step 4: CLI HTTP smoke**

Start:

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model /Users/xin/.ironmlx/models/mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --host 127.0.0.1 \
  --port 18080
```

Verify:

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8}'

curl -s http://127.0.0.1:18080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8}'
```

Expected: HTTP 200 JSON responses with non-empty assistant text.

Check streaming rejection:

```bash
curl -i -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hello"}],"max_tokens":1}'
```

Expected: HTTP 400 with `unsupported_feature`.
