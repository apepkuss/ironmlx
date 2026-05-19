# P5a — Trait Model Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 `core::model::Model` trait，使 `Qwen35Model` 实现该 trait；将 `Scheduler` / `GenerationStream` / `SchedulerActor` / `AppState` 改为 generic over `M: Model`；dense 路径所有既有 unit + integration 测试通过、性能不退化。

**Architecture:** 编译期 monomorphization、零运行时开销。VL 相关方法不进 trait（保留为 `Qwen35Model` inherent），P5a 不动 VL code path 类型签名。

**Tech Stack:** Rust 1.94 / mlx (cxx-mlx wrapper) / Apple Silicon Metal。

**Spec reference:** [docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md](../specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md) §3.1 / §3.9 / §3.11

---

## Pre-flight

### Step 0.1: Branch + clean state 验证

- [ ] 确认在 `ironmlx-p5-moe` 分支

Run: `git -C /Users/xin/workspace/ironmlx-backend branch --show-current`
Expected output: `ironmlx-p5-moe`

- [ ] 确认 working tree clean

Run: `git -C /Users/xin/workspace/ironmlx-backend status --short`
Expected: 空输出

### Step 0.2: 基线 build + test 红绿确认

- [ ] 跑 release build 确认基线绿

Run:
```
cargo build --release -p ironmlx
```
Expected: build 成功无 warning（按 CLAUDE.md `-D warnings` 隐含要求）

- [ ] 跑 lib unit test 全集合作为基线

Run:
```
cargo test -p ironmlx --lib --release
```
Expected: 全 PASS（记下数量，比如 `test result: ok. NNN passed`，作为后续 regression 对照）

---

## Task 1: 定义 `core::model::Model` trait

**Files:**
- Create: `ironmlx/src/core/model.rs`
- Modify: `ironmlx/src/core/mod.rs`
- Test: `ironmlx/src/core/model.rs`（tests 模块）

- [ ] **Step 1.1: 创建 trait 文件**

Create `ironmlx/src/core/model.rs`:
```rust
//! Trait abstracting the inference model used by [`crate::core::scheduler::Scheduler`],
//! [`crate::core::generate::GenerationStream`], and [`crate::core::server::SchedulerActor`].
//!
//! VL-related methods (`forward_vl_chunk` / `batched_prefill_vl` / `compute_vision_embeds`)
//! intentionally remain inherent on concrete models; see spec §3.1 / §3.9.

use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::memory_budget::ModelMeta;
use crate::nn::LayerCache;
use crate::Result;

pub trait Model {
    fn make_cache(
        &self,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<Vec<LayerCache>>;

    #[allow(clippy::too_many_arguments)]
    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    fn model_meta(&self) -> ModelMeta;

    fn num_hidden_layers(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::Model;

    // Compile-time sanity: trait is object-safe-NEUTRAL (we don't require it),
    // and is `Send + Sync` when the model arrays are (mlx::Array is Send+Sync).
    fn _assert_trait_signature_exists<M: Model>(_: &M) {}
}
```

- [ ] **Step 1.2: 把 mod 暴露到 core**

Modify `ironmlx/src/core/mod.rs`:
```rust
pub mod cache;
pub mod chat_template;
pub mod generate;
pub mod loader;
pub mod memory_budget;
pub mod model;          // ← 新增
pub mod sampler;
pub mod scheduler;
pub mod server;
pub mod tokenizer;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use generate::{build_position_ids, GenerateEvent, GenerateRequest, GenerationStream};
pub use loader::{EosTokenId, Loader, QuantMeta, QuantMode, TokenizerConfig};
pub use model::Model;   // ← 新增
pub use sampler::Sampler;
pub use scheduler::{Phase, RequestId, RequestState, Scheduler, SchedulerError, StepEvent};
pub use tokenizer::Tokenizer;
```

- [ ] **Step 1.3: 验证编译通过（尚无实现者）**

Run:
```
cargo build -p ironmlx
```
Expected: build 成功，trait 自身无 implementor 也合法。

- [ ] **Step 1.4: 验证 lib test 不退化**

Run: `cargo test -p ironmlx --lib --release model::tests`
Expected: 1 test PASS (assert trait signature exists)

- [ ] **Step 1.5: Commit T1**

```
git add ironmlx/src/core/model.rs ironmlx/src/core/mod.rs
git commit -m "$(cat <<'EOF'
feat(p5a-t1): introduce core::model::Model trait

Defines the 5-method trait abstracting inference models used by
Scheduler / GenerationStream / SchedulerActor. VL methods stay
inherent on concrete model types (out of scope for the trait).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Qwen35Model` 实现 `Model` trait

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/model.rs`（在文件末尾新增 `impl Model for Qwen35Model` 块）

- [ ] **Step 2.1: 写 impl block**

Append to `ironmlx/src/models/qwen3_5/model.rs`（在最后 `}` 之前不动；新 impl 在 module-level 添加，紧跟现有 `impl Qwen35Model {...}` 之后）:
```rust
impl crate::core::model::Model for Qwen35Model {
    fn make_cache(
        &self,
        batch: i32,
        cap: i32,
        dtype: mlx::Dtype,
    ) -> crate::Result<Vec<crate::nn::LayerCache>> {
        Qwen35Model::make_cache(self, batch, cap, dtype)
    }

    fn forward_on(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35Model::forward_on(self, input_ids, position_ids, per_row_lens, decode_mask, cache, target)
    }

    fn batched_prefill(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35Model::batched_prefill(
            self, input_ids, position_ids, attention_mask, linear_attention_mask,
            per_row_lens, cache, target,
        )
    }

    fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        Qwen35Model::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}
```

- [ ] **Step 2.2: 编译验证**

Run: `cargo build -p ironmlx`
Expected: 成功；可能出现 `unused_imports` warning（暂时无碍，T3 会消化）

- [ ] **Step 2.3: 单测验证 inherent 路径仍工作**

Run: `cargo test -p ironmlx --lib --release qwen3_5::model::tests`
Expected: 全部 PASS（既有 make_cache_layer_kinds_match_partition 等）

- [ ] **Step 2.4: Commit T2**

```
git add ironmlx/src/models/qwen3_5/model.rs
git commit -m "$(cat <<'EOF'
feat(p5a-t2): Qwen35Model implements core::Model

Forwards the 5 trait methods (make_cache / forward_on /
batched_prefill / model_meta / num_hidden_layers) to existing
inherent methods. Inherent methods retained verbatim; VL inherent
methods unchanged (out of trait scope).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `Scheduler` 改 generic `<M: Model>`

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`（结构体 + impl block 全部加 generic 参数）

- [ ] **Step 3.1: 修改 struct 定义**

In `ironmlx/src/core/scheduler.rs`，找到 `pub struct Scheduler { ... }` 定义（line ~150 区间）。

替换 import 添加：
```rust
use crate::core::model::Model;
use std::marker::PhantomData;
```

替换 struct 头：
```rust
pub struct Scheduler<M: Model> {
    // ...原有字段不变...
    _marker: PhantomData<fn(&M) -> ()>,   // M 出现在 step()/prefill_admitted()/admit() 入参签名，
                                          // PhantomData 保证编译期类型绑定且无 dropck/auto-trait 副作用
}
```

- [ ] **Step 3.2: impl block 加 generic**

把所有 `impl Scheduler { ... }` 改为 `impl<M: Model> Scheduler<M> { ... }`，所有原签名为 `model: &Qwen35Model` 的方法改为 `model: &M`。具体处理（按 grep 结果）：
- `pub fn prefill_admitted(&mut self, model: &Qwen35Model)` → `model: &M`
- `fn prefill_admitted_inner(&mut self, model: &Qwen35Model)` → `model: &M`
- `pub fn step(&mut self, model: &Qwen35Model)` → `model: &M`
- `fn step_inner(&mut self, model: &Qwen35Model)` → `model: &M`
- 其他 `model: &Qwen35Model` 参数（line ~1238 / 1283 / 1410 / 1536）逐一替换

`Scheduler::new` 内的 `PhantomData::<fn(&M) -> ()>` 初始化加上（与 `_marker` 字段对应）。

把所有 `model.make_cache(...) / model.forward_on(...) / model.batched_prefill(...) / model.batched_prefill_vl(...) / model.compute_vision_embeds(...)` 调用中**仅 trait 方法**保持不变（trait 中暴露）；**VL 方法**（`batched_prefill_vl` / `compute_vision_embeds` / `forward_vl_chunk`）的所在路径下，方法体内需要用 trait object 的 downcast 或新增 trait extension —— P5a **不动** VL code path 类型签名，参见步骤 3.3。

- [ ] **Step 3.3: VL code path 守护策略**

`scheduler.rs` 内 line 838 `model.batched_prefill_vl(...)` 和 line 1375 / 1478 `model.compute_vision_embeds(...) / forward_vl_chunk(...)` 是 VL 路径。P5a 阶段处理：

把这些调用所在的整个方法 / 代码段（沿用现有方法名 `prefill_admitted_vl_inner` 等如存在；或现有 fn 内的 VL 分支）做参数 `M: Model` 时的 trait bound 扩展：

定义一个本地 trait extension，仅给 dense 用：
```rust
// 在 ironmlx/src/core/scheduler.rs 文件顶部 use 后
pub(crate) trait DenseVlMethods {
    fn batched_prefill_vl(
        &self, /* 与 Qwen35Model::batched_prefill_vl 同签名 */
    ) -> crate::Result<mlx::Array>;
    fn compute_vision_embeds(
        &self,
        pixel_values: &mlx::Array,
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;
    fn forward_vl_chunk(
        &self, /* 与 Qwen35Model::forward_vl_chunk 同签名 */
    ) -> crate::Result<mlx::Array>;
}

impl DenseVlMethods for crate::models::qwen3_5::Qwen35Model {
    // delegate 全部 to inherent methods
}
```

并把 VL 分支所在的方法签名改为 `where M: Model + DenseVlMethods`。MoE model（P5b 引入的 Qwen35MoeModel）**不实现** `DenseVlMethods`，从而尝试用 MoE model 走 VL endpoint 会编译失败（这是 desired，因为 P5 D2 显式排除 MoE VL）。

> **NOTE**: `DenseVlMethods` 完整签名照搬 `Qwen35Model::batched_prefill_vl` / `compute_vision_embeds` / `forward_vl_chunk` 的现有签名；按 `model.rs` line 256-323 / 462-543 复制即可。

- [ ] **Step 3.4: 修复因 generic 化导致的所有调用方编译错误**

Run: `cargo build -p ironmlx`
Expected：会有一批 type 错误，主要来自 `Scheduler::new` / `spawn_scheduler_actor` / 测试 fixture。按 compiler 报错位置逐一加 `::<Qwen35Model>` turbofish 或上下文推断。

修复完成后再 build。

- [ ] **Step 3.5: 跑 scheduler 单测**

Run:
```
cargo test -p ironmlx --lib --release scheduler::tests
cargo test -p ironmlx --lib --release scheduler:: -- --skip integration
```
Expected: PASS（与基线数量一致）

- [ ] **Step 3.6: Commit T3**

```
git add ironmlx/src/core/scheduler.rs
git commit -m "$(cat <<'EOF'
refactor(p5a-t3): Scheduler<M: Model> generic over model type

Scheduler now parameterized over M: Model. Non-VL methods
(step / prefill_admitted / admit) accept &M. VL methods are
guarded by a local DenseVlMethods extension trait implemented
only by Qwen35Model — attempting to instantiate Scheduler<Qwen35MoeModel>
on a VL code path is now a compile error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `GenerationStream` 改 generic `<'m, M: Model>`

**Files:**
- Modify: `ironmlx/src/core/generate.rs`

- [ ] **Step 4.1: struct + impl 加 generic 参数**

In `ironmlx/src/core/generate.rs` line ~84:
```rust
// 改前
pub struct GenerationStream<'m> {
    model: &'m Qwen35Model,
    // ...
}

// 改后
pub struct GenerationStream<'m, M: crate::core::Model> {
    model: &'m M,
    // ...
}
```

`impl<'m> GenerationStream<'m> { ... }` → `impl<'m, M: crate::core::Model> GenerationStream<'m, M> { ... }`。

`GenerationStream::new` 签名 `pub fn new(model: &'m Qwen35Model, ...)` → `pub fn new(model: &'m M, ...)`。

`model.forward_on(...)` 调用保持不变（trait 暴露）。

VL 入口（line ~975 `compute_vision_embeds` / line ~1028 `forward_vl_chunk` / line ~1044 `forward_on`）的处理：
- line 1044 `model.forward_on` 在 trait 内，不动
- line 975 / 1028 是 VL 路径，加 `where M: crate::core::Model + crate::core::scheduler::DenseVlMethods` 守护（与 scheduler 同模式）

- [ ] **Step 4.2: 修编译错误**

Run: `cargo build -p ironmlx`
Expected: 同 T3，按错逐一修。

- [ ] **Step 4.3: Lib test 不退化**

Run: `cargo test -p ironmlx --lib --release generate::tests`
Expected: PASS

- [ ] **Step 4.4: Commit T4**

```
git add ironmlx/src/core/generate.rs
git commit -m "$(cat <<'EOF'
refactor(p5a-t4): GenerationStream<'m, M: Model> generic over model

Stream borrows &'m M instead of &'m Qwen35Model. VL chunk path
guarded by DenseVlMethods trait bound (P5a-t3 contract).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `SchedulerActor` + `AppState` generic 化

**Files:**
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`

- [ ] **Step 5.1: SchedulerActor 加 generic**

In `ironmlx/src/core/server/scheduler_actor.rs` line ~141:
```rust
// 改前
pub struct SchedulerActor {
    model: Arc<Mutex<Qwen35Model>>,
    // ...
}

// 改后
pub struct SchedulerActor<M: crate::core::Model + Send + Sync + 'static> {
    model: Arc<Mutex<M>>,
    // ...
}
```

`impl SchedulerActor { ... }` → `impl<M: crate::core::Model + Send + Sync + 'static> SchedulerActor<M> { ... }`。

`spawn_scheduler_actor(model: Arc<Mutex<Qwen35Model>>, ...)` → `spawn_scheduler_actor<M: crate::core::Model + Send + Sync + 'static>(model: Arc<Mutex<M>>, ...)`。

凡 `&Qwen35Model` 参数 → `&M`；trait 中暴露的方法不需改调用点。

- [ ] **Step 5.2: AppState generic**

In `ironmlx/src/core/server/mod.rs` line ~31:
```rust
// 改前
pub struct AppState {
    pub model: Arc<Mutex<Qwen35Model>>,
    // ...
}

// 改后
pub struct AppState<M: crate::core::Model + Send + Sync + 'static> {
    pub model: Arc<Mutex<M>>,
    // ...
}
```

axum router 内 `Router::new().with_state(state: AppState)` 同样 generic 化；handler 的 `State(state): State<AppState>` → `State<AppState<M>>`。

VL 路由 endpoint（multipart image 处理）改 trait bound 加 `+ DenseVlMethods`。

- [ ] **Step 5.3: 修编译错误**

Run: `cargo build -p ironmlx`
按错误修。可能涉及 turbofish 在 `serve()` 启动路径上的传递。

- [ ] **Step 5.4: Server unit + integration test 验证**

Run:
```
cargo test -p ironmlx --lib --release server::
cargo test -p ironmlx --release --tests p4_http_smoke 2>&1 | head -30
```
Expected: 单测 PASS；HTTP smoke 在缺 checkpoint 环境会 skip（`#[ignore]`），无 panic / 编译错。

- [ ] **Step 5.5: Commit T5**

```
git add ironmlx/src/core/server/
git commit -m "$(cat <<'EOF'
refactor(p5a-t5): SchedulerActor<M> + AppState<M> generic

HTTP server + actor now parameterized over M: Model + Send + Sync
+ 'static. VL endpoints additionally require M: DenseVlMethods.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CLI 入口保持硬接 `Qwen35Model`（P5a 不引入 dispatch）

**Files:**
- Modify (validate-only): `ironmlx/src/cli/generate.rs`
- Modify (validate-only): `ironmlx/src/cli/serve.rs`

P5a 不在 cli 层引入 `model_type` 分发（那是 P5c 工作）。但 generic 化之后 CLI 调用 `Qwen35Model::from_loader` + 把 model 传给 `GenerationStream::new` / `Scheduler<Qwen35Model>::new` 需要让 compiler 推断成功。

- [ ] **Step 6.1: 验证 cli build 成功**

Run: `cargo build -p ironmlx --release`
Expected: 成功；如有 type 推断 ambiguity，在 `GenerationStream::<Qwen35Model>::new(...)` 加 turbofish。

- [ ] **Step 6.2: 不需要 commit（只在必要时改 turbofish），合并进 T7 final commit**

---

## Task 7: dense E2E 回归 + 工具链 hygiene

**Files:** N/A（cross-cutting verification）

- [ ] **Step 7.1: 完整 lib + tests build**

Run:
```
cargo build --release -p ironmlx
cargo test -p ironmlx --lib --release
```
Expected: lib 全部 PASS，数量等于基线（Step 0.2 记下的 NNN）。

- [ ] **Step 7.2: 跑 P4 集成测试 (`#[ignore]` 但可手动跑)**

Run（需要 IRONMLX_MODEL_DIR 指向 ~/.ironmlx/models/Qwen3.5-4B-MLX-4bit 的 snapshot 目录）:
```
export IRONMLX_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots | head -1)
cargo test -p ironmlx --release --test p4_http_smoke -- --ignored --nocapture 2>&1 | tail -30
```
Expected: PASS，server 启动 + 单 prompt token stream 完成。

如果 4B-MLX-4bit 不在本地，此步可跳过但需要在 close-out 注明手测留待 P5d 一起做。

- [ ] **Step 7.3: 跑 b1_p2_3b_3 concurrent generation smoke**

Run:
```
cargo test -p ironmlx --release --test b1_p2_3b_3_concurrent_gs 2>&1 | tail -20
```
Expected: PASS

- [ ] **Step 7.4: 跑 rust 工具链 hygiene**

Run（依 CLAUDE.md 4 命令）:
```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```
Expected: 全 pass，clippy 零 warning。

- [ ] **Step 7.5: 写 close-out 章节追加到 commit message 或 plan**

Final commit:
```
git add -A
git commit -m "$(cat <<'EOF'
chore(p5a): close-out — dense regression + hygiene

All dense unit + integration tests pass (NNN tests, identical to
P5a baseline). cargo fmt / nightly fmt check / clippy -D warnings /
release build all green.

P5a sub-phase complete. P5b can proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## P5a 闭环条件

- [ ] `cargo test -p ironmlx --lib --release` 数量与 P5a 起点基线一致
- [ ] `cargo +nightly clippy --all-features --workspace -- -D warnings` 零 warning
- [ ] `cargo build --release` 成功
- [ ] 至少一个 P4 集成测试（如 p4_http_smoke）手测 PASS
- [ ] `git log ironmlx-p5-moe --oneline` 看到 7 个 P5a commits

满足全部 → P5b 启动。

---

## Self-Review Notes

- ✓ Spec coverage：trait Model (§3.1) / Qwen35Model 实现 (§3.4) / scheduler+generate+server generic (§3.9) 全部覆盖
- ✓ VL 守护策略明确（DenseVlMethods extension trait）
- ✓ CLI dispatch 留给 P5c（P5a 不改）
- ✓ Task 数 = 7 + Pre-flight，符合 [feedback_task_breakdown_bounded](../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_task_breakdown_bounded.md) 5-7 范围
