# Anthropic `/v1/messages` 图像支持 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 ironmlx 的 Anthropic `/v1/messages` 端点支持原生 base64 图像请求，覆盖全部 4 个真 VLM 架构（Qwen3.5-VL dense/moe、MiniCPM-V-4.6、Gemma4）。

**Architecture:** 方案 A「归一化到解码后字节」。新建 `server/vision.rs`，持有 wire-agnostic 的 `DecodedMessage` 类型与共享核心 `expand_decoded_messages`（per-model preprocess + 占位符重写）。OpenAI 与 Anthropic 端点各自把自己的 wire 格式解码成 `DecodedMessage` 后调用同一核心；`forward_vl` 后端零改动。Anthropic 端点新增私有 base64-source schema 并删除现状误拦 `image_url` 的逻辑（同时修掉真 Anthropic image block 触发 serde 422 的缺陷）。

**Tech Stack:** Rust、axum、serde、mlx（mlx-sys FFI）、base64 crate、reqwest（OpenAI 端 URL fetch 用）。参考 spec：[docs/superpowers/specs/2026-06-01-anthropic-image-support-design.md](../specs/2026-06-01-anthropic-image-support-design.md)。

---

## 文件结构

**新建：**
- `ironmlx/src/core/server/vision.rs` — 中立类型 `DecodedPart` / `DecodedMessage`；共享核心 `expand_decoded_messages`；helper `derive_image_token_and_merge`、`qwen_placeholder`、`gemma4_placeholder`。
- `ironmlx/tests/anthropic_image_byte_parity.rs` — OpenAI↔Anthropic 归一化 byte-parity（CI 可跑 + 1 个 ignored Gemma4）。
- `ironmlx/tests/anthropic_image_e2e_parity.rs` — 4 架构端到端（ignored，需 checkpoint env）。

**修改：**
- `ironmlx/src/core/server/mod.rs` — 注册 `pub mod vision;`。
- `ironmlx/src/core/server/openai.rs` — `expand_image_parts_in_messages` 拆为 `decode_openai_messages`（wire→bytes）+ 调 `vision::expand_decoded_messages`；删除已搬走的 `qwen_placeholder` / `gemma4_placeholder` / `flatten_content_with_placeholders`；`image_token_id` / `spatial_merge_size` 派生改用 `vision::derive_image_token_and_merge`。
- `ironmlx/src/core/server/anthropic.rs` — 新增私有 `AnthropicContent` / `AnthropicContentPart` / `AnthropicImageSource` / `AnthropicMessage`；`MessagesRequest.messages` 改用之；新增 `decode_anthropic_messages`（base64→bytes）；handler 调共享核心 + `GenerateRequest` 传真值；删除 `image_url` 拒绝段。

**前置环境（运行 Rust 检测 / 真模型测试时）：** 按 memory `reference_mlx_build_env` 每条命令显式 export `MLX_DIR` + `MLX_METAL_PATH` + `DYLD_LIBRARY_PATH`，否则 mlx-sys build.rs panic。

---

## Task 1: 新建 `server/vision.rs` — 中立类型 + 共享归一化核心

**Files:**
- Create: `ironmlx/src/core/server/vision.rs`
- Modify: `ironmlx/src/core/server/mod.rs`（加 `pub mod vision;`）

- [ ] **Step 1: 注册模块**

在 `ironmlx/src/core/server/mod.rs` 的模块声明区（现有 `pub mod anthropic;` / `pub mod chat_format;` / `mod openai;` 附近）加一行：

```rust
pub mod vision;
```

- [ ] **Step 2: 写 `vision.rs` 骨架 + 中立类型 + 共享核心 + helper（含失败前的内联测试）**

创建 `ironmlx/src/core/server/vision.rs`，完整内容：

```rust
//! Wire-agnostic vision normalization shared by the OpenAI and Anthropic HTTP
//! handlers. Each endpoint decodes ITS OWN wire format (OpenAI `image_url`,
//! Anthropic `image`+`source`) into the neutral [`DecodedMessage`] structure;
//! [`expand_decoded_messages`] then runs the per-model preprocess + placeholder
//! rewrite shared by both. The `forward_vl` backend downstream is untouched.

use mlx::Array;

use crate::core::server::chat_format::{ChatMessage, Content};
use crate::core::server::VisionInputConfig;
use crate::core::tokenizer::Tokenizer;
use crate::models::{gemma4, qwen3_5};

/// A single content part after the endpoint's wire format has been decoded to
/// raw bytes (protocol-agnostic).
pub enum DecodedPart {
    Text(String),
    Image(Vec<u8>),
}

/// A message after wire-format decoding. Both endpoints normalize into this
/// before calling [`expand_decoded_messages`].
pub struct DecodedMessage {
    pub role: String,
    pub parts: Vec<DecodedPart>,
}

/// Qwen3.5-VL placeholder: `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>`.
fn qwen_placeholder(n: usize) -> String {
    let mut s = String::from("<|vision_start|>");
    for _ in 0..n {
        s.push_str("<|image_pad|>");
    }
    s.push_str("<|vision_end|>");
    s
}

/// Gemma4 placeholder: `<|image>` + N × `<|image|>` + `<image|>`.
fn gemma4_placeholder(n: usize) -> String {
    let mut s = String::from("<|image>");
    for _ in 0..n {
        s.push_str("<|image|>");
    }
    s.push_str("<image|>");
    s
}

/// Derive `(image_token_id, spatial_merge_size)` for the active model. Both
/// endpoints need `image_token_id` to populate `GenerateRequest`, so this is
/// the single shared source of the derivation (was inline in `openai.rs`).
pub fn derive_image_token_and_merge(
    vision_input: &VisionInputConfig,
    tokenizer: &Tokenizer,
) -> (i32, i32) {
    match vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => (
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID),
            *spatial_merge_size,
        ),
        VisionInputConfig::Gemma4 { vision_config } => (
            tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .unwrap_or(258_880),
            vision_config.pooling_kernel_size,
        ),
        VisionInputConfig::MiniCpmV46 { spatial_merge_size } => (
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(248_056),
            *spatial_merge_size,
        ),
    }
}

/// For each `DecodedPart::Image`, run the `vision_input`-specific preprocess
/// (Qwen / Gemma4 / MiniCpmV46), collect `pixel_values` + `grid_thw`, and
/// rewrite every message to plain text with placeholder tokens inserted at the
/// image positions. Wire- and endpoint-agnostic.
///
/// Returns `(flat_text_messages, pixel_values, image_grid_thw)`:
/// - `flat_text_messages` feeds `render_and_encode`,
/// - `pixel_values` is `None` when there are no images (eagerly `eval`'d before
///   return so the tensors are safe to cross into `spawn_blocking`),
/// - `image_grid_thw` has one entry per image (MiniCPM-V multi-slice: one per slice).
pub fn expand_decoded_messages(
    messages: Vec<DecodedMessage>,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>)> {
    let spatial_merge_size = match vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => *spatial_merge_size,
        VisionInputConfig::Gemma4 { vision_config } => vision_config.pooling_kernel_size,
        VisionInputConfig::MiniCpmV46 { spatial_merge_size } => *spatial_merge_size,
    };
    if spatial_merge_size <= 0 {
        return Err(anyhow::anyhow!(
            "expand_decoded_messages: spatial_merge_size must be > 0 (got {spatial_merge_size})"
        ));
    }

    let mut all_pixel_values: Vec<Array> = Vec::new();
    let mut grid_thw: Vec<(i32, i32, i32)> = Vec::new();
    let mut placeholders: Vec<String> = Vec::new();

    // First pass: preprocess every image part in order.
    for msg in &messages {
        for part in &msg.parts {
            if let DecodedPart::Image(img_bytes) = part {
                match vision_input {
                    VisionInputConfig::Qwen { .. } => {
                        let (pv, gh, gw) = qwen3_5::image_processor::preprocess(img_bytes)?;
                        let n = ((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize;
                        placeholders.push(qwen_placeholder(n));
                        all_pixel_values.push(pv);
                        grid_thw.push((1, gh, gw));
                    }
                    VisionInputConfig::Gemma4 { vision_config } => {
                        let processed =
                            gemma4::image_processor::preprocess(img_bytes, vision_config)?;
                        placeholders.push(gemma4_placeholder(processed.soft_tokens));
                        grid_thw.push((1, processed.grid_h, processed.grid_w));
                        all_pixel_values.push(processed.pixel_values);
                    }
                    VisionInputConfig::MiniCpmV46 { .. } => {
                        // Multi-slice (LLaVA-UHD): single source of truth for the
                        // divisibility guard, token count, and placeholder.
                        let parts = crate::models::minicpmv4_6::preprocess_sliced_to_parts(
                            img_bytes,
                            spatial_merge_size,
                        )?;
                        all_pixel_values.extend(parts.pixel_values);
                        grid_thw.extend(parts.grid_thw);
                        placeholders.push(parts.placeholder);
                    }
                }
            }
        }
    }

    // Second pass: rewrite messages to plain text with placeholders in-order.
    let mut placeholders = placeholders.into_iter();
    let flat_messages: Vec<ChatMessage> = messages
        .into_iter()
        .map(|msg| {
            let mut out = String::new();
            for part in msg.parts {
                match part {
                    DecodedPart::Text(text) => out.push_str(&text),
                    DecodedPart::Image(_) => {
                        out.push_str(&placeholders.next().unwrap_or_default());
                    }
                }
            }
            ChatMessage {
                role: msg.role,
                content: Content::Text(out),
            }
        })
        .collect();

    let pixel_values = if all_pixel_values.is_empty() {
        None
    } else {
        // Eagerly materialize on this thread before the tensor crosses into
        // spawn_blocking, where a different worker thread's default MLX stream
        // cannot evaluate this thread's lazy graph.
        for pv in &all_pixel_values {
            mlx::transforms::eval(&[pv])?;
        }
        Some(all_pixel_values)
    };

    Ok((flat_messages, pixel_values, grid_thw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_qwen_wraps_n_pads() {
        assert_eq!(
            qwen_placeholder(2),
            "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[test]
    fn placeholder_gemma4_wraps_n_pads() {
        assert_eq!(gemma4_placeholder(2), "<|image><|image|><|image|><image|>");
    }

    #[test]
    fn text_only_message_passes_through_with_no_pixels() {
        let msgs = vec![DecodedMessage {
            role: "user".to_string(),
            parts: vec![DecodedPart::Text("hello".to_string())],
        }];
        let (flat, pv, grid) =
            expand_decoded_messages(msgs, &VisionInputConfig::Qwen { spatial_merge_size: 2 })
                .unwrap();
        assert_eq!(flat.len(), 1);
        match &flat[0].content {
            Content::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Content::Text"),
        }
        assert!(pv.is_none());
        assert!(grid.is_empty());
    }
}
```

- [ ] **Step 3: 运行单测确认通过**

按 memory `reference_mlx_build_env` export 环境后运行：

```bash
cargo test -p ironmlx --lib core::server::vision -- --nocapture
```

Expected: `placeholder_qwen_wraps_n_pads`、`placeholder_gemma4_wraps_n_pads`、`text_only_message_passes_through_with_no_pixels` 三个 PASS。

- [ ] **Step 4: Commit**

```bash
git add ironmlx/src/core/server/vision.rs ironmlx/src/core/server/mod.rs
git commit -m "feat(server): add vision.rs neutral DecodedMessage + shared expand_decoded_messages"
```

---

## Task 2: `openai.rs` 重构走共享核心（行为不变）

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs`

- [ ] **Step 1: 把 `expand_image_parts_in_messages` 改为 `decode_openai_messages` + 共享核心**

在 `openai.rs` 中，将现有 `expand_image_parts_in_messages`（约 `openai.rs:209-296`）**整体替换**为下面两个函数。`decode_openai_messages` 负责 wire→bytes（保留 `data:` / `http(s)` 双路），随后委托共享核心：

```rust
use crate::core::server::vision::{expand_decoded_messages, DecodedMessage, DecodedPart};

/// Decode every OpenAI `image_url` content part into raw bytes, producing the
/// wire-agnostic `DecodedMessage` list consumed by the shared vision core.
pub async fn decode_openai_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
) -> anyhow::Result<Vec<DecodedMessage>> {
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match msg.content {
            Content::Text(t) => parts.push(DecodedPart::Text(t)),
            Content::Parts(ps) => {
                for p in ps {
                    match p {
                        ContentPart::Text { text } => parts.push(DecodedPart::Text(text)),
                        ContentPart::ImageUrl { image_url } => {
                            let bytes = decode_image_url(&image_url.url, client).await?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: msg.role,
            parts,
        });
    }
    Ok(out)
}

/// Walk `messages`, decode + preprocess every `image_url`, and rewrite to
/// text-with-placeholder. Signature unchanged so the handler call site is
/// untouched; the body now delegates to the shared `vision` core.
pub async fn expand_image_parts_in_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>)> {
    let decoded = decode_openai_messages(messages, client).await?;
    expand_decoded_messages(decoded, vision_input)
}
```

- [ ] **Step 2: 删除已搬走的 helper**

删除 `openai.rs` 中现已迁移到 `vision.rs` 的三个私有函数：`qwen_placeholder`（约 `openai.rs:298-305`）、`gemma4_placeholder`（约 `openai.rs:307-314`）、`flatten_content_with_placeholders`（约 `openai.rs:316-335`）。

- [ ] **Step 3: handler 内 `image_token_id` / `spatial_merge_size` 派生改用共享 helper**

在 `chat_completions`（约 `openai.rs:414-440`）中，把那段 `let (image_token_id, spatial_merge_size) = match &state.vision_input { ... }` 整体替换为：

```rust
    let (image_token_id, spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.vision_input,
            &state.tokenizer,
        );
```

- [ ] **Step 4: 处理 import 警告**

`openai.rs` 顶部若 `use crate::models::{gemma4, qwen3_5};` 在删除上述 helper 后变为未使用，则删除该 `use`（`gemma4` / `qwen3_5` 的 preprocess 现在只在 `vision.rs` 调用）。其余 import（`Content`, `ContentPart`, `mlx::Array`）仍被 `decode_openai_messages` / 签名使用，保留。

- [ ] **Step 5: 编译 + 跑现有 OpenAI VL 回归测试确认零漂移**

```bash
cargo build -p ironmlx
cargo test -p ironmlx --lib core::server::openai -- --nocapture
```

Expected: 编译通过；`openai` 模块内现有单测全部 PASS（chat_format / openai 的 content 解析测试不变）。

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/server/openai.rs
git commit -m "refactor(server): route openai vision through shared expand_decoded_messages"
```

---

## Task 3: Anthropic 私有 wire 类型 + serde 单测

**Files:**
- Modify: `ironmlx/src/core/server/anthropic.rs`

- [ ] **Step 1: 写失败的 serde 单测**

在 `anthropic.rs` 末尾的 `#[cfg(test)] mod tests`（若不存在则新建）加入：

```rust
#[cfg(test)]
mod wire_tests {
    use super::*;

    #[test]
    fn parses_native_base64_image_block() {
        let body = r#"
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
            ]
        }"#;
        let m: AnthropicMessage = serde_json::from_str(body).unwrap();
        assert_eq!(m.role, "user");
        let parts = match m.content {
            AnthropicContent::Parts(p) => p,
            _ => panic!("expected Parts"),
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], AnthropicContentPart::Text { .. }));
        match &parts[1] {
            AnthropicContentPart::Image { source } => {
                let AnthropicImageSource::Base64 { media_type, data } = source;
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "aGVsbG8=");
            }
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn parses_plain_string_content() {
        let m: AnthropicMessage =
            serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
        assert!(matches!(m.content, AnthropicContent::Text(ref t) if t == "hi"));
    }

    #[test]
    fn rejects_openai_image_url_shape() {
        // OpenAI `image_url` shape must NOT parse on the Anthropic endpoint.
        let body = r#"
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]
        }"#;
        assert!(serde_json::from_str::<AnthropicMessage>(body).is_err());
    }
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cargo test -p ironmlx --lib core::server::anthropic::wire_tests
```

Expected: 编译失败（`AnthropicMessage` 等类型未定义）。

- [ ] **Step 3: 定义私有 wire 类型 + 改 `MessagesRequest`**

在 `anthropic.rs` 的 `MessagesRequest` 定义上方加入私有类型，并把 `MessagesRequest.messages` 的类型从 `Vec<ChatMessage>` 改为 `Vec<AnthropicMessage>`：

```rust
/// Anthropic native image source — base64 only (URL source is out of scope).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
}

/// Anthropic native content block.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentPart {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

/// Anthropic message content: plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Parts(Vec<AnthropicContentPart>),
}

/// Anthropic message (private wire type; not shared with the OpenAI endpoint).
#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}
```

`MessagesRequest`：

```rust
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,   // was Vec<ChatMessage>
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}
```

> 注意：`media_type` 字段被解析但**不强校验**（spec 决策）；实际格式由下游 image crate 自识别。`AnthropicImageSource::Base64 { media_type, .. }` 中 `media_type` 仅占位读取，Task 4 解构时用 `..` 忽略它——为避免 `dead_code` 警告，保持该字段 `pub`-less 且被测试读取（`wire_tests` 已读 `media_type`）。

- [ ] **Step 4: 运行确认 serde 单测通过**

```bash
cargo test -p ironmlx --lib core::server::anthropic::wire_tests
```

Expected: 三个测试 PASS（`parses_native_base64_image_block`、`parses_plain_string_content`、`rejects_openai_image_url_shape`）。

> 此时 `anthropic.rs` 的 handler 仍引用旧的 `ChatMessage`/`Content`/`ContentPart`，编译会因 `MessagesRequest.messages` 类型变更而**报错**——这是预期的，Task 4 修复 handler。本 Step 仅验证新类型的 serde 行为（用 `--lib core::server::anthropic::wire_tests` 单独跑，整 crate 编译留待 Task 4）。若环境一定要整 crate 编译通过才能跑测试，可临时把 handler 内 `req.messages` 用法注释，Task 4 恢复。

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/core/server/anthropic.rs
git commit -m "feat(server): add Anthropic native base64 image wire types + serde tests"
```

---

## Task 4: Anthropic handler 图像接入 + 删除拒绝逻辑

**Files:**
- Modify: `ironmlx/src/core/server/anthropic.rs`

- [ ] **Step 1: 改 import**

`anthropic.rs` 顶部 import 调整：删除 `use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};` 中不再需要的 `Content` / `ContentPart`（保留 `render_and_encode`、`ChatMessage`——`expand_decoded_messages` 返回 `Vec<ChatMessage>`）。新增：

```rust
use base64::Engine;
use crate::core::server::vision::{expand_decoded_messages, DecodedMessage, DecodedPart};
```

- [ ] **Step 2: 新增 `decode_anthropic_messages`（base64→bytes）**

在 `messages` handler 之前加入：

```rust
/// Decode Anthropic native content blocks into the wire-agnostic
/// `DecodedMessage` list. base64 source is decoded in-process (no network);
/// `media_type` is informational and not validated.
fn decode_anthropic_messages(messages: Vec<AnthropicMessage>) -> anyhow::Result<Vec<DecodedMessage>> {
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match m.content {
            AnthropicContent::Text(t) => parts.push(DecodedPart::Text(t)),
            AnthropicContent::Parts(ps) => {
                for p in ps {
                    match p {
                        AnthropicContentPart::Text { text } => {
                            parts.push(DecodedPart::Text(text))
                        }
                        AnthropicContentPart::Image { source } => {
                            let AnthropicImageSource::Base64 { data, .. } = source;
                            let bytes = base64::engine::general_purpose::STANDARD
                                .decode(data.as_bytes())
                                .map_err(|e| anyhow::anyhow!("image base64 decode: {e}"))?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: m.role,
            parts,
        });
    }
    Ok(out)
}
```

- [ ] **Step 3: 替换 handler 中的拒绝段 + flat_messages 段 + GenerateRequest 装配**

在 `messages` handler 中：

1. **删除** 现有「拒绝 image content parts」整段（约 `anthropic.rs:170-190`，`for m in &req.messages { if let Content::Parts ... return 400 }`）。
2. **删除** 现有 `let flat_messages: Vec<ChatMessage> = req.messages.into_iter().map(...)` 整段（约 `anthropic.rs:192-216`）。
3. **替换为**：

```rust
    // Decode Anthropic wire format → neutral DecodedMessage (base64 → bytes).
    let decoded = match decode_anthropic_messages(req.messages) {
        Ok(d) => d,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("image decode: {e}")).into_response();
        }
    };

    // Shared per-model preprocess + placeholder rewrite.
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_decoded_messages(decoded, &state.vision_input) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("image decode/preprocess: {e}"),
                )
                    .into_response();
            }
        };
    let image_grid_thw_opt = if image_grid_thw.is_empty() {
        None
    } else {
        Some(image_grid_thw)
    };
    let (image_token_id, image_spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.vision_input,
            &state.tokenizer,
        );
```

4. 在 `GenerateRequest` 构造（约 `anthropic.rs:232-248`）中，把图像四字段从硬编码 `None` / 占位值改为真值：

```rust
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: state.prefill_chunk_size,
        pixel_values,
        image_grid_thw: image_grid_thw_opt,
        image_spatial_merge_size,
        image_token_id,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    };
```

> `render_and_encode(&state.tokenizer, &flat_messages, None)` 调用（约 `anthropic.rs:220`）保持不变——`flat_messages` 仍是 `Vec<ChatMessage>`。

- [ ] **Step 4: 编译整 crate + 跑 anthropic 模块测试**

```bash
cargo build -p ironmlx
cargo test -p ironmlx --lib core::server::anthropic
```

Expected: 整 crate 编译通过（Task 3 遗留的类型不匹配在此修复）；`wire_tests` + 现有 anthropic 单测全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/core/server/anthropic.rs
git commit -m "feat(server): wire Anthropic /v1/messages base64 images into shared vision core"
```

---

## Task 5: byte-parity 集成测试（OpenAI ↔ Anthropic 归一化逐位一致）

**Files:**
- Create: `ironmlx/tests/anthropic_image_byte_parity.rs`

测试核心：同一图像字节，经 OpenAI 解码路径（`decode_openai_messages`，`data:` base64）与 Anthropic 解码路径（`decode_anthropic_messages`，raw base64）后，`expand_decoded_messages` 产出逐位一致。Qwen / MiniCpmV46 变体的 `image_processor::preprocess` 是纯函数（无需模型权重），CI 可跑；Gemma4 变体需 `vision_config`，标 `#[ignore]` 用 checkpoint 加载。

> 为让测试可调，`decode_anthropic_messages` 与 `AnthropicMessage` 等类型需对 integration test 可见。在 `anthropic.rs` 把 `decode_anthropic_messages`、`AnthropicMessage`、`AnthropicContent`、`AnthropicContentPart`、`AnthropicImageSource` 的可见性从私有提升为 `pub(crate)`，并通过 `ironmlx` crate 的 test-only 重导出暴露——若 crate 未对外导出 `core::server::anthropic`，则在 `anthropic.rs` 加 `#[cfg(test)]` 不够（integration test 是独立 crate）。**实现选择**：把 byte-parity 断言写成 `anthropic.rs` 内的 `#[cfg(test)] mod parity_tests`（与 wire_tests 同文件），直接访问私有项，无需提升可见性。下面给出该内联形式。

- [ ] **Step 1: 在 `anthropic.rs` 写 byte-parity 测试（失败态）**

在 `anthropic.rs` 的 `#[cfg(test)]` 区加入。需要一张合成图像字节——用一个最小合法 PNG 的 base64（10×10 红色，预编码常量）：

```rust
#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::core::server::chat_format::{ChatMessage, Content, ContentPart, ImageUrl};
    use crate::core::server::openai::decode_openai_messages;
    use crate::core::server::vision::expand_decoded_messages;
    use crate::core::server::VisionInputConfig;

    // 10x10 red PNG, base64 (no `data:` prefix).
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR4nGP8z8Dwn4EIwDiqkL4KAV6+Av0Ojo0kAAAAAElFTkSuQmCC";

    fn anthropic_one_image() -> Vec<AnthropicMessage> {
        vec![AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContent::Parts(vec![
                AnthropicContentPart::Text {
                    text: "what is this?".to_string(),
                },
                AnthropicContentPart::Image {
                    source: AnthropicImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: RED_PNG_B64.to_string(),
                    },
                },
            ]),
        }]
    }

    fn openai_one_image() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: Content::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: format!("data:image/png;base64,{RED_PNG_B64}"),
                    },
                },
            ]),
        }]
    }

    async fn run_parity(vision_input: VisionInputConfig) {
        let client = reqwest::Client::new();
        // OpenAI path: data: URL → bytes → shared core.
        let openai_decoded = decode_openai_messages(openai_one_image(), &client)
            .await
            .unwrap();
        let (o_flat, o_pv, o_grid) =
            expand_decoded_messages(openai_decoded, &vision_input).unwrap();
        // Anthropic path: raw base64 → bytes → shared core.
        let anthropic_decoded = decode_anthropic_messages(anthropic_one_image()).unwrap();
        let (a_flat, a_pv, a_grid) =
            expand_decoded_messages(anthropic_decoded, &vision_input).unwrap();

        // flat text identical.
        assert_eq!(o_flat.len(), a_flat.len());
        for (o, a) in o_flat.iter().zip(a_flat.iter()) {
            let (ot, at) = match (&o.content, &a.content) {
                (Content::Text(o), Content::Text(a)) => (o, a),
                _ => panic!("expected flat Content::Text"),
            };
            assert_eq!(ot, at, "flat text mismatch");
        }
        // grid identical.
        assert_eq!(o_grid, a_grid, "grid_thw mismatch");
        // pixel_values byte-identical.
        let o_pv = o_pv.expect("openai pixels");
        let a_pv = a_pv.expect("anthropic pixels");
        assert_eq!(o_pv.len(), a_pv.len(), "pixel tensor count");
        for (o, a) in o_pv.iter().zip(a_pv.iter()) {
            // Read f32 data the same way tests/common::to_f32_vec does
            // (astype + to_vec); `.as_slice()` is for shape dims, not data.
            let od: Vec<f32> = mlx::ops::cast::astype(o, mlx::Dtype::Float32)
                .unwrap()
                .to_vec()
                .unwrap();
            let ad: Vec<f32> = mlx::ops::cast::astype(a, mlx::Dtype::Float32)
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(od, ad, "pixel_values byte mismatch");
        }
    }

    #[tokio::test]
    async fn byte_parity_qwen() {
        run_parity(VisionInputConfig::Qwen {
            spatial_merge_size: 2,
        })
        .await;
    }

    #[tokio::test]
    async fn byte_parity_minicpmv46() {
        run_parity(VisionInputConfig::MiniCpmV46 {
            spatial_merge_size: 4,
        })
        .await;
    }
}
```

> `decode_openai_messages` 须对 crate 内测试可见——在 Task 2 已是 `pub`，但 `openai` 模块在 `mod.rs` 是 `mod openai;`（私有）。把它改为 `pub(crate) mod openai;`（或 `pub mod openai;`）以便 `anthropic.rs` 内的测试 `use crate::core::server::openai::decode_openai_messages`。`ImageUrl` 类型需在 `chat_format.rs` 为 `pub`（现状已是 `pub struct ImageUrl`）。

- [ ] **Step 2: 调整 `openai` 模块可见性**

在 `mod.rs` 把 `mod openai;` 改为 `pub(crate) mod openai;`。

- [ ] **Step 3: 运行确认通过**

```bash
cargo test -p ironmlx --lib core::server::anthropic::parity_tests -- --nocapture
```

Expected: `byte_parity_qwen`、`byte_parity_minicpmv46` PASS（pixel_values / grid / flat 文本逐位一致）。

> f32 取数契约已对齐 `tests/common/minicpmv46_parity.rs::to_f32_vec`（`ops::cast::astype(a, Dtype::Float32).to_vec()`）；`.as_slice()` 仅用于 `shape()` 维度，不用于取数据。

- [ ] **Step 4: Commit**

```bash
git add ironmlx/src/core/server/anthropic.rs ironmlx/src/core/server/mod.rs
git commit -m "test(server): OpenAI<->Anthropic vision byte-parity (qwen + minicpmv46 variants)"
```

---

## Task 6: 端到端 parity（4 架构传递性 + MiniCPM-V 直连 mlx-vlm）

**Files:**
- Create: `ironmlx/tests/anthropic_image_e2e_parity.rs`

策略（spec §6 传递性机制）：boot server → Anthropic `/v1/messages` 单图请求生成的 completion，断言与 OpenAI `/v1/chat/completions` 同图像同 prompt（greedy、同 max_tokens）逐 token（逐字符）一致。4 架构各一 `#[ignore]` 测试，env var 指向 checkpoint。MiniCPM-V 额外断言生成首段 == mlx-vlm `expected_gen_tokens.npy` 的 decode 文本。

- [ ] **Step 1: 写测试文件（boot helper + 请求 helper + 4 架构 + mlx-vlm 直连）**

创建 `ironmlx/tests/anthropic_image_e2e_parity.rs`：

```rust
//! End-to-end Anthropic image parity: Anthropic /v1/messages image completion
//! must match the OpenAI /v1/chat/completions completion token-for-token under
//! the SAME image + prompt + greedy decode (transitivity: the OpenAI endpoint
//! is already validated vs mlx-vlm per architecture). MiniCPM-V additionally
//! checks the generated prefix directly against the mlx-vlm fixture.
//!
//! Env vars (each points at a checkpoint snapshot dir):
//!   QWEN35_VL_DENSE_MODEL, QWEN35_VL_MOE_MODEL, GEMMA4_MODEL, MINICPMV46_MODEL
//!
//! Run (per memory reference_mlx_build_env, export MLX_DIR/MLX_METAL_PATH/DYLD first):
//! ```text
//! MINICPMV46_MODEL=/path/to/snapshot \
//!   cargo test --release -p ironmlx --test anthropic_image_e2e_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Duration;

use ironmlx::core::server::{self, VisionInputConfig};
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::model::Model;
use ironmlx::core::{Loader, Tokenizer};

const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR4nGP8z8Dwn4EIwDiqkL4KAV6+Av0Ojo0kAAAAAElFTkSuQmCC";

async fn alloc_port() -> u16 {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

/// Boot a server for `model` on `port` with the given vision override.
fn boot<M>(model: M, tokenizer: Tokenizer, port: u16, vision: Option<VisionInputConfig>)
    -> tokio::task::JoinHandle<anyhow::Result<()>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    tokio::spawn(async move {
        server::serve(
            model, tokenizer, "e2e".to_string(), "127.0.0.1", port,
            /* prefill_chunk_size */ 2048, /* b_max */ 1,
            /* admission_deadline_ms */ 5, /* admission_queue_max */ 32,
            /* max_cache_cap */ 32768, /* p5h_measurement_eval_probes */ false,
            vision,
        ).await
    })
}

fn anthropic_body() -> serde_json::Value {
    serde_json::json!({
        "model": "e2e",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": RED_PNG_B64}}
            ]
        }]
    })
}

fn openai_body() -> serde_json::Value {
    serde_json::json!({
        "model": "e2e",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{RED_PNG_B64}")}}
            ]
        }]
    })
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .no_proxy()
        .build()
        .unwrap()
}

/// Anthropic completion text from non-streaming /v1/messages.
async fn anthropic_text(c: &reqwest::Client, port: u16) -> String {
    let r: serde_json::Value = c
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .json(&anthropic_body())
        .send().await.unwrap()
        .json().await.unwrap();
    r["content"][0]["text"].as_str().unwrap_or("").to_string()
}

/// OpenAI completion text from non-streaming /v1/chat/completions.
async fn openai_text(c: &reqwest::Client, port: u16) -> String {
    let r: serde_json::Value = c
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&openai_body())
        .send().await.unwrap()
        .json().await.unwrap();
    r["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string()
}

/// Shared assertion: Anthropic completion == OpenAI completion for the same image.
async fn assert_transitive_parity(port: u16) {
    let c = client();
    tokio::time::sleep(Duration::from_millis(800)).await;
    let a = anthropic_text(&c, port).await;
    let o = openai_text(&c, port).await;
    assert!(!a.is_empty(), "anthropic completion empty");
    assert_eq!(a, o, "Anthropic and OpenAI completions diverged for same image");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_VL_DENSE_MODEL"]
async fn e2e_qwen35_vl_dense() {
    let dir = PathBuf::from(std::env::var("QWEN35_VL_DENSE_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::Qwen35Model::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let _s = boot(model, tok, port, None); // None → fallback VisionInputConfig::Qwen
    assert_transitive_parity(port).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_VL_MOE_MODEL"]
async fn e2e_qwen35_vl_moe() {
    let dir = PathBuf::from(std::env::var("QWEN35_VL_MOE_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::Qwen35MoeModel::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let _s = boot(model, tok, port, None);
    assert_transitive_parity(port).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MODEL"]
async fn e2e_gemma4() {
    let dir = PathBuf::from(std::env::var("GEMMA4_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let cfg = ironmlx::models::Gemma4Config::from_loader(&loader).unwrap();
    let vision = cfg.vision_config.map(|vc| VisionInputConfig::Gemma4 { vision_config: vc });
    let model = ironmlx::models::Gemma4Model::from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let _s = boot(model, tok, port, vision);
    assert_transitive_parity(port).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MINICPMV46_MODEL"]
async fn e2e_minicpmv46() {
    let dir = PathBuf::from(std::env::var("MINICPMV46_MODEL").unwrap());
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let vision = Some(VisionInputConfig::MiniCpmV46 { spatial_merge_size: 4 });
    let _s = boot(model, tok, port, vision);
    assert_transitive_parity(port).await;
}
```

- [ ] **Step 2: 跑 4 架构端到端（有 checkpoint 时）**

逐个架构跑（串行，按 memory `feedback_serial_perf_experiments` 一次一个 server）。示例（MiniCPM-V）：

```bash
MINICPMV46_MODEL=$HOME/.ironmlx/models/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/<sha> \
  cargo test --release -p ironmlx --test anthropic_image_e2e_parity -- --ignored e2e_minicpmv46 --nocapture
```

Qwen dense / moe / Gemma4 同理用 `QWEN35_VL_DENSE_MODEL` / `QWEN35_VL_MOE_MODEL` / `GEMMA4_MODEL` 指向对应 snapshot。

Expected: 每个架构 `assert_transitive_parity` PASS（Anthropic completion == OpenAI completion，且非空）。

- [ ] **Step 3: MiniCPM-V 直连 mlx-vlm 断言**

在 `anthropic_image_e2e_parity.rs` 加入直连 mlx-vlm 的测试。复用现有 fixture `expected_gen_tokens.npy`（由 [gen_single_image_generate.py](../../../ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_generate.py) 生成）：用该 fixture 的**输入图像/prompt**驱动 Anthropic 请求，断言生成 token 的解码文本以 mlx-vlm reference 的解码文本为前缀。

```rust
mod common;
use common::minicpmv46_parity::{checkpoint_dir, load_npy_in, FIXTURE_DIR_VL};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MINICPMV46_MODEL + minicpmv46_vl fixtures"]
async fn e2e_minicpmv46_vs_mlxvlm() {
    // mlx-vlm reference greedy tokens (astype + to_vec, per tests/common style).
    let expected = load_npy_in(FIXTURE_DIR_VL, "expected_gen_tokens.npy");
    let expected_i32: Vec<i32> = mlx::ops::cast::astype(&expected, mlx::Dtype::Int32)
        .unwrap()
        .to_vec()
        .unwrap();
    let expected_ids: Vec<u32> = expected_i32.into_iter().map(|x| x as u32).collect();

    let dir = checkpoint_dir(); // honors MINICPMV46_MODEL
    let loader = Loader::open_multimodal(&dir).unwrap();
    let tok = Tokenizer::from_loader(&loader).unwrap();
    let expected_text = tok.decode(&expected_ids, /* skip_special */ true).unwrap();

    let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader).unwrap();
    let port = alloc_port().await;
    let vision = Some(VisionInputConfig::MiniCpmV46 { spatial_merge_size: 4 });
    let _s = boot(model, tok, port, vision);

    let c = client();
    tokio::time::sleep(Duration::from_millis(800)).await;
    let got = anthropic_text(&c, port).await;

    // Anthropic-generated prefix must match the mlx-vlm reference prefix.
    let n = expected_text.len().min(got.len());
    assert_eq!(
        &got[..n], &expected_text[..n],
        "Anthropic generation diverged from mlx-vlm reference"
    );
}
```

> 若 fixture 的输入图像并非上面的合成 `RED_PNG_B64`（fixture 用真实测试图），则 `anthropic_body()` 在此测试需改用 fixture 对应的原图 base64。实现时：从 fixture 目录读取该图（gen 脚本的输入图路径，见脚本头部注释），base64 编码后填入请求。若 fixture 仅存预处理后的 `input_pixel_values.npy` 而无原图，则此直连断言降级为「Anthropic vs OpenAI 文本一致」已由 `e2e_minicpmv46` 覆盖，本测试改为跳过并 `log` 说明（按 memory `feedback no silent caps` 显式记录）。

- [ ] **Step 4: 跑 mlx-vlm 直连断言**

```bash
source ~/.local/mlx/mlx-env.sh
MINICPMV46_MODEL=/path/to/MiniCPM-V-4.6-4bit/snapshots/<sha> \
  cargo test --release -p ironmlx --test anthropic_image_e2e_parity -- --ignored e2e_minicpmv46_vs_mlxvlm --nocapture
```

Expected: PASS（Anthropic 生成前缀 == mlx-vlm reference 前缀）。

- [ ] **Step 5: Commit**

```bash
git add ironmlx/tests/anthropic_image_e2e_parity.rs
git commit -m "test(server): Anthropic image e2e parity (4 archs transitive + minicpmv46 vs mlx-vlm)"
```

---

## Task 7: Close-out — 全量 Rust 检测 + 回归

**Files:** 无新增（验收 + 修复期）

- [ ] **Step 1: 格式化**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
```

Expected: 第二条无输出（格式干净）。

- [ ] **Step 2: clippy**

```bash
cargo +nightly clippy --all-features --workspace -- -D warnings
```

Expected: 无 warning/error。若有，修复（常见：`dead_code` on `media_type`——已被 `wire_tests` 读，应无；未使用 import——删除）。

- [ ] **Step 3: release 编译**

```bash
cargo build --release
```

Expected: 成功。

- [ ] **Step 4: 全量 lib 测试（CI 无模型）**

```bash
cargo test -p ironmlx --lib
```

Expected: 全绿，含 `vision::tests`、`anthropic::wire_tests`、`anthropic::parity_tests`（qwen + minicpmv46 byte-parity）。

- [ ] **Step 5: ignored e2e（有 checkpoint 时，串行）**

逐架构跑 Task 6 的 4 个传递性测试 + minicpmv46 直连。全部 PASS 即满足 spec §7 验收 1-2。

- [ ] **Step 6: Commit（若 Step 1-3 有修复）**

```bash
git add -A
git commit -m "chore(server): fmt + clippy clean for Anthropic image support"
```

---

## Self-Review（写完后对照 spec 自查）

**Spec coverage：**
- spec §3.2 vision.rs 中立类型 + 共享核心 → Task 1 ✓
- spec §3.3 OpenAI 端点重构（行为不变）→ Task 2 ✓
- spec §3.4 Anthropic 私有 wire 类型 + handler 接入 + 删除拒绝段 + GenerateRequest 真值 → Task 3 + 4 ✓
- spec §4 错误处理（base64 失败 400 / preprocess 400 / 误发 image_url serde 失败）→ Task 3 Step 1（`rejects_openai_image_url_shape`）+ Task 4 Step 2 错误转换 ✓
- spec §6 byte-parity（Qwen/MiniCpmV46 CI + Gemma4 ignored）→ Task 5 ✓（Gemma4 byte-parity 见下「已知缺口」）
- spec §6 端到端传递性 + MiniCPM-V 直连 → Task 6 ✓
- spec §7 验收 6（fmt/clippy/build）→ Task 7 ✓

**已知缺口（需 Boss 知晓）：**
- Task 5 byte-parity 内联实现仅覆盖 **Qwen + MiniCpmV46** 两个 CI-可跑变体；**Gemma4 变体的 byte-parity** 因需 `Gemma4VisionConfig`（无法廉价合成）未写成独立 ignored 测试——但 Gemma4 归一化正确性由 Task 6 `e2e_gemma4`（传递性，需 checkpoint）覆盖。若要补 Gemma4 纯 byte-parity，需在 Task 5 增加一个 `#[ignore]` 测试用 `GEMMA4_MODEL` 加载 `vision_config`。

**Placeholder scan：** 无 TBD/TODO；每个改码 step 含完整代码。Task 6 Step 3 的 fixture-原图 fallback 有显式降级说明（非 placeholder，是 honest 的条件分支 + `log`）。

**Type consistency：** `DecodedMessage`/`DecodedPart`/`expand_decoded_messages`/`derive_image_token_and_merge` 全程同名；`AnthropicMessage`/`AnthropicContent`/`AnthropicContentPart`/`AnthropicImageSource::Base64 { media_type, data }` 在 Task 3 定义、Task 4/5 使用一致；`GenerateRequest` 字段名与 `generate.rs` 定义（`pixel_values`/`image_grid_thw`/`image_spatial_merge_size`/`image_token_id`）一致。
