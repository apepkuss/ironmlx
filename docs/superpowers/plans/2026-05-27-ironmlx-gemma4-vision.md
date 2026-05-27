# Gemma4 Vision Implementation Plan

> Required design reference: [docs/superpowers/specs/2026-05-27-ironmlx-gemma4-vision-design.md](../specs/2026-05-27-ironmlx-gemma4-vision-design.md).

**Goal:** 基于 Gemma4 Dense text commit，支持本地 `gemma-4-e4b-it-4bit` 的单 image + text -> text 推理，CLI 和 OpenAI-compatible server 都可用。

**Architecture:** 新增 Gemma4 image processor、vision tower 和 multimodal projection；复用现有 `GenerateRequest` / `GenerationStream` / Scheduler 的 VL 扩展点，但按 Gemma4 token 和 pixel 流程生成 placeholder 与 vision embeddings。

**Scope:** 单图、Dense、image+text。多图、audio、video、MoE 不进入本轮。

## Task 0: 事实校验与文档

- [x] 核对本地 checkpoint 中的 `vision_config`、`vision_tower.*`、`embed_vision.*` 和 special tokens。
- [x] 对照 `mlx-vlm/mlx_vlm/models/gemma4` 梳理数据流。
- [x] 写入设计 spec 和实施 plan。
- [x] 自审 Mermaid 语法与实施边界。

## Task 1: Config 与 loader 边界

**Files:**
- `ironmlx/src/models/gemma4/config.rs`
- `ironmlx/src/core/loader.rs`

- [x] 增加 `Gemma4VisionConfig`，解析 patch size、pooling kernel、position embedding size、clipping、standardize。
- [x] `Gemma4Config` 记录 image boundary token、vision soft token 上限和 optional vision_config。
- [x] `Loader::open_multimodal` 保留 vision tower 和 `embed_vision`，但仍丢弃 audio tower 和 `embed_audio`。
- [x] 增加 loader 单元测试覆盖 multimodal load 不保留 Gemma4 audio tower。

## Task 2: Gemma4 image processor

**Files:**
- `ironmlx/src/models/gemma4/image_processor.rs`

- [x] 实现 RGB decode。
- [x] 实现 Gemma4 aspect-ratio resize。
- [x] 输出 `[1,3,H,W]` float32 pixel values。
- [x] 返回 soft token count 和 `(1, grid_h, grid_w)`。
- [x] 单元测试 resize 维度是 `pooling_kernel_size * patch_size` 的倍数，soft token count 正确。

## Task 3: Vision tower 与 projection

**Files:**
- `ironmlx/src/models/gemma4/vision.rs`

- [x] 实现 `ClippableLinear`，支持 direct prefix 和 `.linear` 子前缀。
- [x] 实现 patch embedder、2D position embedding、bidirectional mask。
- [x] 实现 multidimensional RoPE。
- [x] 实现 16-layer ViT block。
- [x] 实现 position-aware avg pool。
- [x] 实现 `MultimodalEmbedder`：RMSNorm(no scale) + projection。
- [x] 使用真实 checkpoint CLI/server smoke 覆盖加载验证。

## Task 4: Text 注入与 Gemma4Model VL 方法

**Files:**
- `ironmlx/src/models/gemma4/text_model.rs`
- `ironmlx/src/models/gemma4/model.rs`

- [x] 暴露 text model 的 `forward_embeddings_on`。
- [x] per-layer input 支持传入屏蔽 image token 后的 token ids。
- [x] `Gemma4Model` 加载 optional `vision_tower` 和 `embed_vision`。
- [x] 实现 `compute_vision_embeds`。
- [x] 实现 `forward_vl_chunk` 和 `batched_prefill_vl`。
- [x] 保持 text-only `forward_on` 遇到 image/audio token 时明确报错。

## Task 5: CLI 与 server 接入

**Files:**
- `ironmlx/src/cli/generate.rs`
- `ironmlx/src/cli/serve.rs`
- `ironmlx/src/core/server/openai.rs`
- `ironmlx/src/core/server/chat_format.rs`

- [x] `generate` 增加 `--image`，单图限制。
- [x] Gemma4 image 请求使用 `Loader::open_multimodal`。
- [x] server 根据模型输入类型选择 Qwen 或 Gemma4 placeholder style。
- [x] Gemma4 server image_url 使用 Gemma4 processor，不复用 Qwen patch processor。
- [x] 多图请求返回清晰 400。

## Task 6: 验证

- [x] `cargo fmt`
- [x] `cargo +nightly fmt --all -- --check`
- [x] `cargo +nightly clippy --all-features --workspace -- -D warnings`
- [x] `cargo build --release`
- [x] Gemma4 text-only smoke regression。
- [x] Gemma4 CLI image smoke。
- [x] Gemma4 server image_url smoke。

## Completion Criteria

- [x] `gemma-4-e4b-it-4bit` 可以处理单张图片和文本问题。
- [x] text-only Gemma4 路径保持可用。
- [x] Qwen VL placeholder 行为由单元测试锁定，真实 Qwen VL smoke 未纳入本轮。
- [x] 所有 Rust gate 通过，或明确记录外部阻塞原因。
