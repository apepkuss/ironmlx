# B1-p2.3e.2 PRNG Key Centralization — Design (Lightweight Outline)

**Status:** Outline (brainstormed 2026-05-17, post-3e.1a ship)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Predecessor:** [B1-p2.3e.1b vectorize configured sampler](2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md) (必须先 land)
**Branch target:** TBD（3e.1b ship 后定）

> **Lightweight outline rationale**: 3e.2 detailed plan 强依赖 3e.1b 实施后 `sample_batch` 实际签名和 PRNG plumbing 的实际形态。本文档现写定大方向 + interface boundary + acceptance，**plan-grade detail 等 3e.1b land 后再补**。

## 0. Program context

| Sub-spec | Status |
| --- | --- |
| 3e.1a vectorize greedy argmax | ✅ DONE |
| 3e.1b vectorize configured sampler | Spec written ([sibling](2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md)) |
| **3e.2 PRNG key centralization** | **This outline** |

## 1. Goal

3e.1b 完成 batched categorical 后，`sample_batched_categorical` 内 host-side plumbing 仍有：

- **Stage A** — B 次 `Sampler.cell.take()` collect PRNG keys
- **Stage Z** — B 次 `Sampler.cell.set()` distribute new keys 回 Cell
- B 次 Array slice clone 在 `advance_keys[i, :]` 提取

3e.2 把 PRNG state 从 per-row `Sampler.cell` 移到 Scheduler 集中持有的 `[B_max, 2]` 张量，消除以上 plumbing。

## 2. Motivation

### 2.1 Code simplicity > performance

3e.2 的 perf 收益微小：B × 3 µs ≈ 12 µs at B=4 = 0.02% of 64.7ms step。

3e.2 的真正价值在 **代码 simplicity**：

| | Pre-3e.2 | Post-3e.2 |
|---|---|---|
| `Sampler` trait bounds | `!Send` (Cell)、`!Copy` | `Send + Sync + Clone + Copy` |
| Interior mutability | `Cell<Option<Array>>` | 无 |
| sample_batch 内 borrow | `&[&Sampler]` + 手动 take/set | `&[&Sampler]` + 显式 `prng_state: &mut Array` 参数 |
| Cross-thread safety | 需要文档化 single-thread 假设 | 类型系统保证 |

3e.1b 后 `Cell<Option<Array>>` 是 `Sampler` 唯一的 interior mutability source。移除后 `Sampler` 变成 pure config struct，简化所有 borrow / threading concern。

### 2.2 Why not merged into 3e.1b

3e.1b scope 已经 7 ops + batched categorical + history bincount，规模较大。3e.2 独立成 stage：

- 3e.1b 在 batched categorical 接口处保留 `&Sampler` plumbing（不动 Sampler struct），让 3e.1b ship 风险可控
- 3e.2 单独 land 让 review 聚焦 PRNG ownership 重构，不混业务逻辑

## 3. Non-goals

- **NG1.** Reproducibility bit-exact parity 3e.1b ↔ 3e.2 PRNG 输出（init path 不同，相同 `Sampler.seed` 输出 token 序列可能 differ — 接受）
- **NG2.** 新的 sampler op / vectorization 工作（3e.2 仅 state ownership 重构）
- **NG3.** PRNG seed 通过 API endpoint 透传策略变更（保留 `Sampler.seed` 字段）

## 4. Architecture sketch

### 4.1 Sampler struct (post-3e.2)

```rust
#[derive(Debug, Clone, Copy)]   // ← Copy 是新增的 (Cell 移除后)
pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    // REMOVED: pub(crate) prng_cell: Cell<Option<Array>>
}
```

`Sampler` 变成 pure config struct。

### 4.2 Scheduler PRNG state

```rust
pub struct Scheduler {
    // ... existing fields ...
    pub(crate) prng_state: Array,  // [b_max, 2] u32, 持续持有整 batch 的 PRNG keys
}

impl Scheduler {
    pub fn new(b_max: usize, ...) -> Self {
        let prng_state = Array::zeros(&[b_max as i32, 2]);  // 占位，admit 时填充
        // ...
    }

    fn init_row_prng(&mut self, row_idx: usize, seed: Option<u64>) -> Result<()> {
        let key = match seed {
            Some(s) => mlx::random::key(s),
            None => mlx::random::key(default_seed_from_time_xor_idx(row_idx)),
        };
        // Write key into self.prng_state[row_idx, :]
        // (用 mlx scatter 或 slice update)
        ...
    }
}
```

- `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` 入口加 `self.init_row_prng(row_idx, sampler.seed)` call
- `evict_row` / `evict_all` 时可以选择 zero 出 row 对应 prng_state slot（也可不动 — 反正下次 admit 时会覆盖）

### 4.3 sample_batch 签名变更

```rust
pub fn sample_batch(
    samplers: &[&Sampler],     // config only, no PRNG state
    logits: &Array,            // [B, vocab]
    histories: &[&[u32]],
    prng_state: &mut Array,    // [B, 2] u32 — NEW; in-place advance
) -> Result<Vec<u32>>
```

Sample step internal:

```rust
fn sample_batched_categorical_v2(
    probs: &Array,
    prng_state: &mut Array,
) -> Result<Vec<u32>> {
    let split = mlx::random::split(prng_state, 2)?;            // [2, B, 2]
    let advance_keys = split.slice_axis(0, 0, 1)?.squeeze(0)?;  // [B, 2]
    let sample_keys = split.slice_axis(0, 1, 2)?.squeeze(0)?;   // [B, 2]
    let tokens = mlx::random::categorical(probs, &sample_keys, -1, ())?;
    *prng_state = advance_keys;  // in-place advance
    tokens.to_vec::<u32>()
}
```

无 host-side Cell.take/set；in-place advance。

### 4.4 Scheduler::step / prefill_admitted_inner 调用方式

```rust
let tokens = sample_batch(&row_samplers, &logits_bv, &history_refs, &mut self.prng_state)?;
```

Scheduler 持 `prng_state`，每 step 通过 `&mut self.prng_state` 传递 (Stage A collect 消失)。

## 5. Implementation footprint

| File | Change |
|---|---|
| `core/sampler.rs` | 移除 `Cell<Option<Array>>` field；移除 `prng_cell.take()/set()` 内部 helpers；`#[derive(Clone, Copy)]` |
| `core/scheduler.rs` | `Scheduler::new` 加 `prng_state` init；`admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` 加 `init_row_prng` call；`evict_row` / `evict_all` 可选 zero 出 row prng_state |
| `core/sampler.rs::sample_batch` | 签名加 `prng_state: &mut Array`；移除 `Sampler.cell` 操作；改 `sample_batched_categorical_v2` |
| `Scheduler::step` / `prefill_admitted_inner` | sample_batch 调用加 `&mut self.prng_state` 参数 |
| `Scheduler::admit_mid_finalize` | 仍 B=1，但用 single-row PRNG state slice 替代 `state.sampler.cell` |
| Tests | 修：所有用 `Sampler::with_seed(N)` 后比较 token 的 unit/integration tests，可能需 update seed value 或改 statistical check (因 init path 变) |

预估代码改动：~150-250 lines net。

## 6. Acceptance

- ✅ `Sampler` 改为 `#[derive(Clone, Copy)]`，类型系统验证 Send + Sync
- ✅ 36+ scheduler lib tests no regress (token 输出可能 differ — 改 statistical / functional 测试)
- ✅ 10+ sample_batch unit tests no regress（含 mixed batch + identity)
- ✅ 真模型 smoke：3e.1b perf gate 测试在 3e.2 上仍 PASS（per-row median ≤ 250 ms, lockstep ratio ≤ 2×）
- ✅ sweep_full 17 suites PASS
- ✅ Hygiene 全绿
- ✅ 无 backwards-compat 代码（Cell 完全移除，无 Optional wrap）

## 7. Risks

| R# | Risk | Mitigation |
|---|---|---|
| **R1** | 现有 unit tests 依赖 `Sampler::with_seed(N)` → bit-exact 输出 | Update tests 改 statistical (1000 sample freq check) 或 update seed values；记入 plan T-fix |
| **R2** | `Scheduler.prng_state` evict 时不 zero → 下次 admit 同 row 可能用旧 key | 在 init_row_prng 直接 overwrite — 上一行 evict 不需要 clear |
| **R3** | `mlx::ops::slice update` (在 prng_state 上局部 write) mlx-rs API 不齐 | 用整个 prng_state 重新 stack 替代 slice update；plan T0 verify |
| **R4** | `admit_mid_finalize` 单 row PRNG state 与 batch state 不一致 | finalize 把 admitted row 的 PRNG init 作为 finalize 步骤；用 single-row slice from batch state |

## 8. Plan deferral

3e.2 plan 文档 (`docs/superpowers/plans/YYYY-MM-DD-b1-p2-3e-2-prng-key-batching.md`) 等 3e.1b land 后写。原因：

- 3e.1b 实施过程会确定 `sample_batched_categorical` 的实际签名（特别是 PRNG plumbing 的 in/out 形状），3e.2 plan 基于 actual API 写
- 3e.1b 实施中可能发现 mlx::random API 不完全符合预期（R1/R3），导致 3e.2 实现路径需要调整
- 3e.1b 实施过程会自然 surface 哪些 helper 需要 share / 哪些可 inline，影响 3e.2 footprint

Plan-grade detail 包括：task 拆分、step-by-step 代码 diff、unit test list、perf gate、commit 节奏。

## 9. Carry-forward (post-3e.2)

3e.2 后 sampler vectorization series 收尾。下一步：进入 Qwen3.5 MoE 路径。

---

**Document history:**
- 2026-05-17 — Initial outline (brainstormed alongside 3e.1b spec, plan-grade detail deferred to post-3e.1b)
