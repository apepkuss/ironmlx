# Gemma4 Dense Text-Only Implementation Plan

> For agentic workers: implement task-by-task. Keep this checklist current as work progresses. Required design reference: [docs/superpowers/specs/2026-05-27-ironmlx-gemma4-dense-design.md](../specs/2026-05-27-ironmlx-gemma4-dense-design.md).

**Goal:** 在 `codex/gemma-4-moe` worktree 中实现 `model_type=gemma4` 且 `text_config.enable_moe_block=false` 的 Dense text-only 推理路径，并用本地 `gemma-4-e4b-it-4bit` checkpoint 做 smoke 验证。

**Architecture:** 新增 `ironmlx/src/models/gemma4/`，复用基础 `Loader`、`Linear`、`Embedding`、`RmsNorm`、`KVCache` 和 `Model` trait；Gemma4 特有的 per-layer input、GeGLU、RoPE、sliding/full attention、KV sharing 和 tied output 均放在 Gemma4 模块内。

**Tech Stack:** Rust / cxx-mlx / Apple Silicon Metal / safetensors mmap / 4-bit affine quantized weights。

## Pre-flight

- [x] 确认 worktree：`/Volumes/Dev/cxx-mlx-gemma-4-moe`
- [x] 确认分支：`codex/gemma-4-moe`
- [x] 确认本地模型存在：`~/.ironmlx/models/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9`
- [x] 确认 `gemma-4-e4b-it-4bit` 是 Dense multimodal checkpoint，本阶段只实现 language model text-only path

## Task 0: 文档与事实校验

**Files:**
- `docs/superpowers/specs/2026-05-27-ironmlx-gemma4-dense-design.md`
- `docs/superpowers/plans/2026-05-27-ironmlx-gemma4-dense.md`

- [x] 写入设计 spec，明确 scope、out-of-scope、第一性原则和高性能数据流。
- [x] 写入实施 plan，拆分为可执行任务。
- [x] 自审 spec/plan，修正与本仓库现有 trait/cache/scheduler 不一致的描述。
- [x] 校验 Mermaid 语法。

## Task 1: Loader sanitize 根因修复

**Files:**
- `ironmlx/src/core/loader.rs`

- [x] 增加模型感知辅助函数，识别 `model_type=gemma4` 或 `text_config.model_type=gemma4_text`。
- [x] 在 conv/norm 检测前，text-only loader 丢弃 `vision_tower.*`、`audio_tower.*`、`embed_vision.*`、`embed_audio.*`。
- [x] 将 Qwen3.5 RMSNorm `+1.0` shift 限定在 Qwen3.5 语义下，Gemma4 不允许被 audio conv 触发。
- [x] 保持 `language_model.` prefix strip 和 tied `lm_head` drop 逻辑。
- [x] 添加 loader 单元测试覆盖 Gemma4 audio conv 不触发 norm shift。

## Task 2: Gemma4 config 与模块骨架

**Files:**
- `ironmlx/src/models/gemma4/mod.rs`
- `ironmlx/src/models/gemma4/config.rs`
- `ironmlx/src/models/mod.rs`

- [x] 新增 `Gemma4Config` 和 `Gemma4TextConfig`，从嵌套 `text_config` 解析。
- [x] 校验 `enable_moe_block=false`，否则报错并提示 MoE 不在本任务范围。
- [x] 解析 `layer_types`、`rope_parameters`、`num_kv_shared_layers`、`hidden_size_per_layer_input`、`final_logit_softcapping`。
- [x] 提供 helper：`layer_kind(i)`、`kv_dim_for_layer(i)`、`first_kv_shared_layer_idx()`、`previous_kv_layer(i)`。
- [x] 添加 config 单元测试，使用真实 config fixture 或 inline JSON。

## Task 3: Gemma4 基础算子

**Files:**
- `ironmlx/src/models/gemma4/ops.rs`
- `ironmlx/src/models/gemma4/mlp.rs`

- [x] 实现无权重 RMSNorm helper，调用 `mlx::fast::rms_norm_on(x, None, eps, target)`。
- [x] 实现 GELU approximate helper，复用现有 MLX op。
- [x] 实现 `Gemma4GeGluMlp`：fused gate/up load，forward 后 split，`gelu(gate) * up`，再 down。
- [x] 单元测试 fused gate/up shape 和错误 key 报错。

## Task 4: RoPE、mask 与 attention

**Files:**
- `ironmlx/src/models/gemma4/rope.rs`
- `ironmlx/src/models/gemma4/attention.rs`

- [x] 实现 sliding default RoPE：base 10000，full head rotation。
- [x] 实现 full proportional RoPE：base 1000000，partial rotary factor 0.25，左右半头分段旋转。
- [x] 实现 full causal/ragged additive mask 接入。
- [x] 实现 sliding window additive mask，窗口大小来自 `sliding_window`。
- [x] 实现 `Gemma4Attention`：q/k/v/o projection，q/k/v norm，scale=1.0，cache update，shared K/V reuse。
- [x] 单元测试 mask 边界：第 0、511、512、513 token 的可见范围。

## Task 5: Decoder layer、TextModel 与 KV sharing

**Files:**
- `ironmlx/src/models/gemma4/decoder_layer.rs`
- `ironmlx/src/models/gemma4/text_model.rs`

- [x] 实现 `Gemma4DecoderLayer` 的 attention、GeGLU FFN、per-layer input block、layer scalar。
- [x] 实现 text embedding scaling 和 per-layer input 计算。
- [x] `make_cache` 只为 pre-shared layers 创建 `KVCache`，减少后 18 层 cache 内存。
- [x] forward 中按真实 layer index 映射 cache slot，并为 shared layers 复用最近同类型 K/V。
- [x] 单元测试 `previous_kv_layer` 对 e4b 42 层 pattern 的映射。

## Task 6: Top-level model 与入口 dispatch

**Files:**
- `ironmlx/src/models/gemma4/model.rs`
- `ironmlx/src/cli/generate.rs`
- `ironmlx/src/cli/serve.rs`

- [x] 实现 `Gemma4Model::from_loader`。
- [x] 实现 `core::Model`：`make_cache`、`forward_on`、`batched_prefill`、`forward_text_hidden`、`model_meta`、`num_hidden_layers`。
- [x] 实现 tied embedding output projection和 final logit softcap。
- [x] `model_meta().num_hidden_layers` 使用 24 个 cache-bearing layers；`num_hidden_layers()` 返回 42 个 decoder layers。
- [x] 为 `DenseVlMethods` 提供 text-only stub，VL 调用返回明确错误。
- [x] CLI 和 server dispatch 增加 `model_type=gemma4`。

## Task 7: 验证与修复

- [x] `cargo fmt`
- [x] `cargo +nightly fmt --all -- --check`
- [x] `cargo +nightly clippy --all-features --workspace -- -D warnings`
- [x] `cargo build --release`
- [x] 真实模型 smoke：用 `gemma-4-e4b-it-4bit` 运行短 prompt greedy 生成。
- [x] 自审 `git diff`，修复发现的问题。

## Completion Criteria

- [x] `model_type=gemma4` 可以被 CLI 加载并生成文本。
- [x] Gemma4 audio/vision 权重不会污染 text-only loader sanitize。
- [x] Dense Gemma4 forward 包含 per-layer input、GeGLU、sliding/full attention、KV sharing 和 final logit softcap。
- [x] 所有要求的 Rust 检查通过，或明确记录无法通过的外部原因。
