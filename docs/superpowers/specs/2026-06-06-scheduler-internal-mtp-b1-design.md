# Scheduler Internal MTP B1 Design

**目标：** 在 `Scheduler` 内部实现第一阶段 MTP speculative decoding，范围限定为 `b_max=1`、text-only、Qwen dense/MoE 模型，使用已有 offsets-only rollback primitives，并保持每次 `step` 只发一个 `StepEvent` 的 scheduler 事件合同。

**决策摘要：**

- 采用 `b_max=1` first 的 scheduler-internal MTP 路线。
- 不直接做 multi-row batch MTP。
- 不在本阶段接入 `SchedulerActor` 服务路径。
- `Scheduler<M>` 继续保持 `M: Model`，不把整个 scheduler 泛型边界改成 `M: MtpSpeculativeModel`。
- 在 scheduler 内新增可选 MTP runtime state；MTP 专用方法单独加 `M: MtpSpeculativeModel + DenseVlMethods` bound。

---

## 1. 背景

当前分支已有两个 checkpoint：

- `b7322cd feat: add qwen mtp speculative decoding`
- `eedde89 feat: gate scheduler-text mtp bench path`

第一阶段已经完成：

- Qwen dense/MoE MTP head 加载。
- `MtpTextGenerationStream` 单请求 speculative decoding。
- `KVCacheSnapshot`、`MtpCacheSnapshot`、`LayerCacheSnapshot` offsets-only rollback。
- `ironmlx-core-bench --mode scheduler-text --b-max 1 --mtp-model-dir ...` 的 bench gating。

目前 `scheduler-text` 的 MTP bench path 仍然委托到单请求 `MtpTextGenerationStream`，不是 `Scheduler` 自身的 decode loop。下一阶段要把 MTP 的 draft/verify/accept 语义放进 scheduler 内部，但只先做单请求窗口。

---

## 2. 当前边界

### 2.1 Scheduler 的现有合同

`Scheduler::prefill_admitted` 与 `Scheduler::step` 目前满足这些合同：

- `prefill_admitted` 为每个 admitted row 采样第一个 token，并为每个 row 发一个 `StepEvent`。
- `step` 在 `Phase::Decoding` 下对每个 active row 最多推进一个 token。
- `StepEvent` 是 per-token 事件，服务层依赖这个节奏做 SSE / unary 输出。
- mid-admit 预留 row 通过 `generated_tokens.is_empty()` 被 `step` 排除，直到 `admit_mid_finalize` 采样 first token。
- compact cache layout 用 `cache_rows` 映射 model batch row 到 scheduler slot row。

这些合同本阶段不改变。

### 2.2 MTP stream 的现有语义

`MtpTextGenerationStream` 已有稳定单请求流程：

- prefill prompt 到 main cache。
- 从 prompt last hidden 投影并采样 first token。
- 用 MTP head draft tokens。
- main model verify `[current_token] + draft_tokens`。
- 根据 `resolve_speculative_tokens` 得到 accepted prefix / corrected token / bonus token。
- mismatch 时 restore main cache 到 verify 前 snapshot，再 replay accepted prefix。
- MTP cache 每个 draft window 后 restore 到 draft 前 snapshot。
- pending token queue 每次对外只吐一个 token。

本阶段 scheduler-internal MTP 应复用这个语义，而不是重新定义 speculative decoding。

### 2.3 Rollback 策略

现有 rollback primitives 已符合 Boss 偏好的高性能策略：

- `KVCacheSnapshot` 只保存 per-row logical offsets。
- `MtpCacheSnapshot` 只保存每层 `KVCacheSnapshot`。
- `LayerCacheSnapshot` 对 Full / Linear / Mla 都是轻量 checkpoint。
- rollback 不复制 dense K/V buffer，不清 stale data；后续 mask 只读取 logical offset 范围。

本阶段继续使用 offsets-only rollback。任何需要复制 K/V buffer 的方案都不进入本阶段。

---

## 3. Scope

### 3.1 In Scope

- 新增 scheduler-internal MTP runtime state。
- 新增 `Scheduler` 的 MTP 专用 prefill/step 方法。
- 限制 `b_max == 1`。
- 限制 text-only request。
- 限制 Qwen dense/MoE text 模型，即实现了 `MtpSpeculativeModel` 的模型。
- bench `scheduler-text --b-max 1 --mtp-model-dir` 改为使用真正的 scheduler-internal MTP 方法。
- 输出 `MtpSpeculativeStats`，保持 bench JSON 中 `mtp_stats` 可见。
- 单元测试覆盖 gating、pending-token event cadence、rollback stats。
- 真实 Qwen3.5-4B base + MTP smoke。

### 3.2 Out Of Scope

- multi-row MTP batch。
- `SchedulerActor` 服务路径启用 MTP。
- VL request 的 MTP。
- 非 Qwen 模型的 MTP。
- 非 greedy sampler。
- 改变 `StepEvent` 结构。
- 改变当前普通 `prefill_admitted` / `step` 行为。

---

## 4. Architecture

### 4.1 Scheduler State

在 `scheduler.rs` 内新增 private runtime state：

```rust
struct SchedulerMtpState {
    cfg: MtpSpeculativeConfig,
    mtp_cache: MtpCache,
    pending_tokens: VecDeque<u32>,
    last_hidden: Array,
    stats: MtpSpeculativeStats,
}
```

`Scheduler<M>` 新增字段：

```rust
mtp_state: Option<SchedulerMtpState>,
```

理由：

- `MtpCache`、`Array`、stats 都不依赖 `M::MtpHead`。
- `Scheduler<M>` 可以继续保持 `M: Model`。
- 非 MTP scheduler 只持有 `None`。
- MTP 专用方法通过参数接收 `&M::MtpHead`，因此无需把 head 存入 scheduler。

### 4.2 Public/Internal Methods

新增 scheduler 方法：

```rust
pub fn prefill_admitted_mtp_single(
    &mut self,
    model: &M,
    mtp: &M::MtpHead,
    cfg: MtpSpeculativeConfig,
) -> Result<Vec<StepEvent>>
where
    M: MtpSpeculativeModel + DenseVlMethods;

pub fn step_mtp_single(
    &mut self,
    model: &M,
    mtp: &M::MtpHead,
) -> Result<Vec<StepEvent>>
where
    M: MtpSpeculativeModel;

pub fn mtp_stats(&self) -> Option<MtpSpeculativeStats>;
```

Method contracts：

- `prefill_admitted_mtp_single` 只允许 `b_max == 1`、一个 admitted row、text-only、`cfg.max_draft_tokens > 0`。
- `step_mtp_single` 只允许 `Phase::Decoding` 且 `mtp_state.is_some()`。
- 两个方法都遵守 poison-on-error 规则，与现有 `prefill_admitted` / `step` 一致。
- 每次调用最多返回一个 `StepEvent`。

### 4.3 Prefill Flow

`prefill_admitted_mtp_single` 的流程：

1. 校验 scope：`b_max == 1`、active row 数量为 1、request text-only、sampler pipelinable。
2. 分配 main cache：`model.make_cache(1, cap, Dtype::Bfloat16)`。
3. 分配 MTP cache：`model.make_mtp_cache(mtp, 1, cap, Dtype::Bfloat16)`。
4. 使用 `forward_text_hidden` 对 prompt 做 chunked prefill，填充 main cache，并得到 last prompt hidden。
5. `model.project_hidden_on(last_prompt_hidden)` 得到 first-token logits。
6. 使用 row PRNG 采样 first token。
7. 更新 `RequestState.generated_tokens`、`real_len`、finish state。
8. 如果未 finished，立即执行一个 speculative window，为后续 step 填充 `pending_tokens`。
9. 返回 first token 的单个 `StepEvent`。

第 8 步与 `MtpTextGenerationStream::next_token` 一致：first event 的返回包含首个 MTP window 的 draft/verify 成本。这让 bench 的 TTFT 语义与现有 MTP stream 保持一致。

### 4.4 Step Flow

`step_mtp_single` 的流程：

1. 如果 request finished，返回空 events。
2. 从 `pending_tokens` 弹出一个 token。
3. 将 token append 到 `RequestState.generated_tokens`，并更新 `real_len`。
4. 检查 stop / length。
5. 如果本次 token 后未结束且 `pending_tokens` 为空，执行下一个 speculative window。
6. 返回这个 token 的单个 `StepEvent`。

`generated_tokens` 只记录已经发出 `StepEvent` 的 token。未来 accepted 但尚未发出的 token 只存在于 `pending_tokens`，避免 scheduler 对外状态提前可见。

### 4.5 Speculative Window Flow

新增 private helper：

```rust
fn fill_mtp_window_single<M>(
    mtp_state: &mut SchedulerMtpState,
    request_state: &mut RequestState,
    main_cache: &mut [LayerCache],
    model: &M,
    mtp: &M::MtpHead,
) -> Result<()>
where
    M: MtpSpeculativeModel;
```

语义与 `MtpTextGenerationStream::fill_window` 对齐：

- `remaining = max_new_tokens - generated_tokens.len()`。
- `draft_budget = cfg.max_draft_tokens.min(remaining)`。
- draft input hidden 从 `mtp_state.last_hidden` 开始。
- draft 阶段使用 `MtpCache::snapshot` / `restore`。
- verify input 为 `[current_token] + draft_tokens`。
- verify start position 为 `prompt_ids.len() + generated_tokens.len() - 1`。
- verify 前对 main cache 做 `LayerCache::snapshot`。
- mismatch 时 restore main cache，再 replay accepted prefix。
- `pending_tokens` 只追加 `resolve_speculative_tokens(...).tokens_to_append` 中尚未 emitted 的 future tokens。

关键不变量：

- `last_hidden` 表示“当前 pending token 之前那个 token”的 hidden。
- accepted draft token 的 K/V 可能已经在 main cache 中，即使尚未对外 emitted。
- corrected / bonus token 不在 main cache 中；它会在 pending drain 后作为下一轮 current token 被 verify 写入。

---

## 5. Bench Integration

当前 `ironmlx-core-bench` 的 `scheduler-text --b-max 1 --mtp-model-dir` 委托到 `MtpTextGenerationStream`。

本阶段改为：

- 构造 `Scheduler::<M>::new(1, effective_cap_max, model.model_meta())`。
- 调用 `prefill_admitted_mtp_single(model, mtp, cfg)` 获取 first event。
- 循环调用 `step_mtp_single(model, mtp)` 直到 finish 或 `max_tokens`。
- 从 `scheduler.mtp_stats()` 写入 `Record.mtp_stats`。

`--b-max > 1` 继续拒绝。

---

## 6. Testing

### 6.1 Unit Tests

新增或扩展 `scheduler.rs::tests`：

- `mtp_prefill_rejects_bmax_gt_one`
  - 构造 `Scheduler` with `b_max=2`。
  - 调用 `prefill_admitted_mtp_single`。
  - 断言错误包含 `b_max 1`。

- `mtp_prefill_rejects_vl_request`
  - request 带 `pixel_values`。
  - 断言错误包含 `text-only`。

- `mtp_step_emits_one_pending_token_per_call`
  - 使用 fake `MtpSpeculativeModel`。
  - 让 first window 产生两个 pending tokens。
  - 断言 `prefill_admitted_mtp_single` 返回 1 个 event。
  - 连续 `step_mtp_single` 每次只返回 1 个 event。

- `mtp_step_updates_stats_after_mismatch`
  - fake model 让 draft token 与 verify token mismatch。
  - 断言 `rollback_count == 1`。
  - 断言 pending corrected token 仍按单 token event 输出。

### 6.2 Bench Tests

扩展 `ironmlx-core-bench` tests：

- `scheduler_text_mtp_uses_scheduler_internal_path`
  - 验证 `run_scheduler_mtp_single_request` 不再委托 `run_mtp_generation_stream`。
  - 若无法直接观察内部调用，则用 fake record/stat helper 测试 `mode == Scheduler` 且 `mtp_stats.is_some()`。

### 6.3 Real Smoke

使用本地模型：

- Base: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- MTP: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485`

命令形态：

```bash
target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file /tmp/ironmlx-core-bench-prompt.txt \
  --mode scheduler-text \
  --b-max 1 \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485 \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 0 \
  --prefill-chunk-size 0 \
  --out /tmp/ironmlx-core-bench-scheduler-mtp-internal-smoke.json
```

Expected JSON：

- `"mode": "scheduler-text"`
- `"mtp_draft_tokens": 1`
- `"mtp_stats"` present
- `"valid": true`

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| `generated_tokens` 提前记录 pending tokens 导致对外状态漂移 | `generated_tokens` 只在 event emitted 时 append；future accepted tokens 只存 `pending_tokens` |
| cache offset 与 scheduler `real_len` 含义混淆 | verify start position 用 `prompt_ids.len() + generated_tokens.len() - 1`，不直接依赖 `real_len` |
| first event TTFT 与现有 MTP stream 不一致 | `prefill_admitted_mtp_single` 在返回 first event 前执行首个 `fill_mtp_window_single` |
| 非 Qwen 模型被泛型边界影响 | `Scheduler<M>` 保持 `M: Model`；只有 MTP 方法加 `M: MtpSpeculativeModel` |
| batch MTP 需求诱导过早泛化 | 本阶段硬拒绝 `b_max != 1`；multi-row 另开设计 |
| actor 接入影响线上服务节奏 | 本阶段不接 `SchedulerActor`；先通过 bench 和 smoke 验证 core scheduler 语义 |

---

## 8. Acceptance Criteria

1. `scheduler-text --b-max 1 --mtp-model-dir` 使用 scheduler-internal MTP，不再委托 `MtpTextGenerationStream`。
2. `Scheduler::step_mtp_single` 每次最多返回一个 `StepEvent`。
3. `generated_tokens` 只包含已 emitted tokens。
4. mismatch rollback 使用 offsets-only snapshots。
5. `--b-max > 1` 仍拒绝 MTP scheduler path。
6. Unit tests 覆盖 gating、pending queue、rollback stats。
7. Qwen3.5-4B real smoke 通过并输出 `mtp_stats`。
8. Rust 检查通过：
   - `cargo fmt`
   - `cargo +nightly fmt --all -- --check`
   - `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings`
   - `MLX_DIR=$HOME/.local/mlx cargo build --release`

---

## 9. Follow-Up Phases

### Phase 2: SchedulerActor Gating

在 core scheduler MTP smoke 稳定后，增加服务层配置与 actor 持有 MTP head 的设计。目标是让 short text Qwen request 可以通过 scheduler actor 使用 MTP。

### Phase 3: Multi-Row MTP

重新设计 per-row pending、ragged verify、MTP cache compact layout、mid-admit 交错策略。这个阶段需要单独 spec，因为它会改变 scheduler hot path 的 batching 组织。
