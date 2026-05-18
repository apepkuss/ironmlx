# B1-p2.3e.2 PRNG Key Centralization — Design (Post-3e.1b)

**Status:** Plan-grade detail (3e.1b shipped 2026-05-18 commit `d146d60`)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Predecessor:** [B1-p2.3e.1b vectorize configured sampler](2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md) (shipped)
**Branch target:** `ironmlx-b1-p2-3e2-prng-centralization` (cut from `ironmlx-b1-p2-3e1b-vectorize-configured` HEAD `d146d60`)

> **Spec rewrite note**: 原 spec (commit 92b5b50) 是 lightweight outline，写于 3e.1b ship 前，assumes 3e.1b would ship full GPU `sample_batched_categorical([B, vocab] + [B, 2] keys)`。实际 3e.1b T4 hot-fix (commit 684ca05) 把 sample step 改为 CPU `sample_row_cpu`，per-row 调 `sampler.ensure_key()`。本文 update §1 / §4 / §5 反映 ship reality。Motivation (Sampler → pure config struct) 不变。

## 0. Program context

| Sub-spec | Status |
| --- | --- |
| 3e.1a vectorize greedy argmax | ✅ DONE (2026-05-17 commit ab4c839) |
| 3e.1b vectorize configured sampler (CPU handoff path) | ✅ DONE (2026-05-18 commit d146d60) |
| **3e.2 PRNG key centralization** | **This spec** |

## 1. Goal

3e.1b ship 后的 PRNG 路径：

```text
sample_row_cpu(probs[vocab], top_p, min_p, sampler: &Sampler) {
    ...                                          // CPU sort / nucleus / min_p / renorm
    let key = sampler.ensure_key()?;             // <- per-row Cell<Option<Array>>.take()
                                                 //    + mlx::random::split + Cell.set
    let u = mlx::random::uniform().shape(1).key(&key).sample()?.item::<f32>()?;
    // CDF walk
}
```

每个 row 的 Sampler 各自持 `key: Cell<Option<Array>>` (interior mutability，line 43 of sampler.rs)，`ensure_key` 内部 split + store。这是 3e.1b ship 状态。

3e.2 把 PRNG state ownership 从 per-row `Sampler.key` 移到 Scheduler 集中持 `[B_max, 2] u32` 张量，并把 `sample_row_cpu` 签名从 `(sampler: &Sampler)` 改为 `(prng_state_row: &mut Array)`。

## 2. Motivation

### 2.1 Code simplicity > performance

3e.2 perf gain 微小：每 step B 次 `Cell.take()` + B 次 `Cell.set()` + B 次小 Array clone = `B × ~5µs ≈ 20µs at B=4` = **0.024% of 82.57ms step** (3e.1b 实测)。

3e.2 真正价值在 **代码 simplicity**：

| | Pre-3e.2 (3e.1b ship) | Post-3e.2 |
| --- | --- | --- |
| `Sampler` trait bounds | `!Send` (Cell), `!Copy`, manual `Clone` impl (resets PRNG) | `Send + Sync + Clone + Copy` (auto-derive) |
| Interior mutability | `Cell<Option<Array>>` | 无 |
| `Sampler::clone()` semantics | Custom impl: 配置 copy + PRNG state 重置 | `#[derive(Clone, Copy)]` |
| `sample_row_cpu` 签名 | `sample_row_cpu(probs, top_p, min_p, sampler: &Sampler)` | `sample_row_cpu(probs, top_p, min_p, prng_state: &mut Array)` |
| PRNG state ownership | Per-row Sampler instance | Centralized: Scheduler.prng_state |
| Cross-thread safety | 文档化 single-thread 假设 (Cell !Sync) | 类型系统保证 |

3e.1b 后 `Cell<Option<Array>>` 是 `Sampler` 唯一的 interior mutability source + 唯一阻止 auto `Clone + Copy + Send + Sync` derive 的 field。移除后 `Sampler` 变成 pure config POD，简化所有 borrow / threading concern。

### 2.2 Why not merged into 3e.1b

3e.1b scope 已经 7 ops + GPU→CPU hot-fix + 10 commits 规模较大；T4 hot-fix 后 review 已经 carry 较多 design adaptation。3e.2 独立 stage:

- 3e.1b 保留 `Sampler.key` Cell 接口 (3e.1a 已有)，3e.1b implementer 不需碰 Sampler struct
- 3e.2 单独 land 让 review 聚焦 PRNG ownership 重构本身，不混业务逻辑改动

### 2.3 Test-impact preview

`Sampler::with_seed(N)` 仍是 public API。但 PRNG init path 从 "Cell mint from seed on first ensure_key" 改为 "Scheduler 在 admit 时从 seed mint 写入 prng_state[row]"。语义上每 row 用同 seed 输出应该一致 (mlx::random::key(seed) 是 deterministic)，但 fresh-seed 与 lazy-init 时机不同，bit-exact 不保证。Tests 用 `Sampler::with_seed(N)` + 比较 token 序列的，**可能需 adjust 期望值** — 或改 statistical (1000 sample freq) 校验。

## 3. Non-goals

- **NG1.** Reproducibility bit-exact parity 3e.1b ↔ 3e.2 PRNG 输出 — init path 时机不同 (admit-time vs lazy)，相同 `Sampler.seed` 输出 token 序列 may differ. 接受 drift; integration tests 改 statistical 校验 (1000 sample freq check) 或 update expected values.
- **NG2.** 新的 sampler op / vectorization (3e.2 pure ownership refactor)
- **NG3.** `Sampler.seed` 字段 API 变更 (保留)
- **NG4.** sample_row_cpu CPU 算法变更 (sort / nucleus / min_p / renorm / CDF 算法都不动)
- **NG5.** GPU pipeline 重新 batched (T4 hot-fix CPU path 保留，CPU 仍是 sample 路径；3e.2 仅改 PRNG state ownership)

## 4. Design

### 4.1 Sampler struct (post-3e.2)

```rust
#[derive(Debug, Clone, Copy)]   // ← Copy 新增 (Cell 移除后)
pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: u64,
    // REMOVED: key: Cell<Option<Array>>
}

impl Sampler {
    pub fn greedy() -> Self { /* unchanged */ }
    pub fn with_temperature(mut self, t: f32) -> Self { /* unchanged */ }
    // ... all with_* builders unchanged ...
    // ensure_key / store_key REMOVED (now Scheduler's responsibility)
}
```

Old manual `Clone` impl (which reset PRNG) deleted — auto-derive `Clone + Copy` 现给出 trivial bit-copy (PRNG state 不在 Sampler 内，无需 reset 逻辑)。

### 4.2 Scheduler PRNG state

```rust
pub struct Scheduler {
    // ... existing fields ...
    /// Per-row PRNG state. Shape `[b_max, 2]` u32. Row `i` 持 row i 的 mlx
    /// random key. Init from `Sampler.seed` 在 admit 时，advance 在每次
    /// configured_pipeline sample step.
    pub(crate) prng_state: Array,
}

impl Scheduler {
    pub fn new(b_max: usize, /* ... */) -> Self {
        let prng_state = mlx::ops::constructors::zeros(
            &[b_max as i32, 2_i32], mlx::Dtype::Uint32, ()
        )?;
        Self { /* ... */, prng_state }
    }

    /// Initialize row `row_idx`'s PRNG state from `seed`. Called by
    /// `admit_inner` / `admit_mid_inner` when a new request occupies the slot.
    fn init_row_prng(&mut self, row_idx: usize, seed: u64) -> Result<()> {
        let key = mlx::random::key(seed)?;  // [2] u32
        // Write key into prng_state[row_idx, :]
        // mlx slice_update API:
        let key_2d = key.reshape(&[1_i32, 2][..])?;  // [1, 2]
        self.prng_state = mlx::ops::indexing::slice_update(
            &self.prng_state,
            &key_2d,
            &[row_idx as i32, 0][..],
            &[row_idx as i32 + 1, 2][..],
        )?;
        Ok(())
    }
}
```

- `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` 入口加 `self.init_row_prng(row_idx, sampler.seed)` call
- `evict_row` / `evict_all` **不动 prng_state** — 下次 admit 会 overwrite (R2)
- `b_max == 0` edge case: prng_state shape `[0, 2]` (mlx 允许 0-sized dimension)

### 4.3 `sample_row_cpu` 签名变更

```rust
fn sample_row_cpu(
    probs: &[f32],
    top_p: f32,
    min_p: f32,
    prng_state_row: &mut Array,  // [2] u32 — NEW; replaces &Sampler
) -> Result<u32> {
    // ... unchanged sort / nucleus / min_p / renorm ...

    // PRNG: split + advance + sample
    let (next_key, sample_key) = mlx::random::split(prng_state_row)?;
    let u_arr = mlx::random::uniform()
        .shape(1_i32)
        .dtype(mlx::Dtype::Float32)
        .key(&sample_key)
        .sample()?;
    let u: f32 = u_arr.item()?;
    *prng_state_row = next_key;  // in-place advance

    // ... unchanged CDF walk ...
}
```

`sample_row_cpu` 不再 import `Sampler` — 仅消费 probs + 2 scalars + per-row PRNG state slice.

### 4.4 `configured_pipeline` 调用方式

```rust
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
    prng_state: &mut Array,  // [B, 2] u32 — passed by Scheduler::step
) -> Result<Vec<u32>> {
    // ... unchanged GPU stage (penalties / temp / top_k / softmax) ...
    let probs_flat: Vec<f32> = probs_gpu.to_vec()?;
    let top_p_host: Vec<f32> = configs.top_p.to_vec()?;
    let min_p_host: Vec<f32> = configs.min_p.to_vec()?;

    let mut tokens = Vec::with_capacity(b);
    for row in 0..b {
        let row_probs = &probs_flat[row * vocab..(row + 1) * vocab];
        // Take a slice view of prng_state[row, :] for in-place advance.
        // Approach: slice → call → write back via slice_update.
        let mut row_key = mlx::ops::indexing::slice(
            prng_state,
            &[row as i32, 0][..],
            &[row as i32 + 1, 2][..],
        )?.reshape(&[2_i32][..])?;
        let token = sample_row_cpu(row_probs, top_p_host[row], min_p_host[row], &mut row_key)?;
        // Write advanced key back into prng_state[row, :]
        let row_key_2d = row_key.reshape(&[1_i32, 2][..])?;
        *prng_state = mlx::ops::indexing::slice_update(
            prng_state,
            &row_key_2d,
            &[row as i32, 0][..],
            &[row as i32 + 1, 2][..],
        )?;
        tokens.push(token);
    }
    Ok(tokens)
}
```

Alternative if `slice_update` overhead 在 hot path 测过太重 — collect updated `row_key` into Vec, stack at end. Plan T1 测决。

### 4.5 `sample_batch` 签名变更

```rust
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
    prng_state: &mut Array,  // NEW
) -> Result<Vec<u32>> {
    // ... validation ...
    if samplers.iter().all(|s| s.is_greedy()) {
        // all-greedy fast path UNCHANGED (no PRNG needed for argmax)
    }
    configured_pipeline(samplers, logits, histories, prng_state)
}
```

`Scheduler::step` / `prefill_admitted_inner` 调用方加 `&mut self.prng_state`。

### 4.6 admit_mid_finalize 处理

`admit_mid_finalize` is B=1 path — 用 single-row PRNG state slice:

```rust
let mut row_key = mlx::ops::indexing::slice(&self.prng_state, /* row_idx */)?.reshape([2])?;
let token = sample_row_cpu(probs, top_p, min_p, &mut row_key)?;
// write back
```

或者 admit_mid_finalize 可以 short-circuit 用 `Sampler::sample` 现有 B=1 path？但 `Sampler::sample` 现也用 self.key Cell — 3e.2 后该 fn 内部也需改为接受 `&mut Array` PRNG state。Plan T2 决定: 改 `Sampler::sample` 签名 OR 把 admit_mid_finalize 切到 sample_row_cpu。

## 5. Implementation footprint

| File | Change |
| --- | --- |
| `ironmlx/src/core/sampler.rs` | 移除 `Sampler.key: Cell<Option<Array>>` field; 移除 manual `Clone` impl (auto-derive `Clone + Copy`); 移除 `ensure_key` + `store_key` fns; `Sampler::sample` 签名加 `prng_state: &mut Array` 参数; `sample_row_cpu` 签名加 `prng_state_row: &mut Array` 替代 `sampler: &Sampler`; `configured_pipeline` 签名加 `prng_state: &mut Array` |
| `ironmlx/src/core/scheduler.rs` | `Scheduler` struct 加 `prng_state: Array` field; `Scheduler::new` init `prng_state = zeros([b_max, 2], u32)`; `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` 加 `self.init_row_prng(row_idx, request.sampler.seed)` call; `Scheduler::step` / `prefill_admitted_inner` 调 `sample_batch(..., &mut self.prng_state)`; `admit_mid_finalize` 调 `Sampler::sample(..., &mut self.prng_state slice)` (or sample_row_cpu) |
| Tests | 修 expected token values in tests using `Sampler::with_seed(N)` (init time差异)，或改 statistical assertions (1000 sample freq within tolerance)。Check: sample_batch_configured_fallback_no_panic_in_range (no exact tokens); sample_batch_no_op_default_configured_pipeline_in_range; perf gate b1_p2_3e_1b_configured_decode_speedup (timing-only, no expected tokens) |

预估代码改动：~150-300 lines net (sampler.rs ~120 lines, scheduler.rs ~80 lines, tests ~50-100 lines).

## 6. Acceptance

- ✅ `Sampler` 改为 `#[derive(Debug, Clone, Copy)]`, types verify Send + Sync via static_assertions
- ✅ 36+ scheduler lib tests no regress (token outputs may differ — change to statistical or update expected)
- ✅ 40+ sampler unit tests no regress (mostly behavior-based, statistical or identity check)
- ✅ Real-model 3e.1b perf gate (`b1_p2_3e_1b_configured_decode_speedup`) post-3e.2 仍 PASS (median ≤ 250 ms, lockstep ratio ≤ 2×)
- ✅ sweep_full 16+1 suites PASS (3e.1b sweep showed 13/16 with environment issues; 3e.2 sweep target: 16/16 in single run if M1 not under cumulative load)
- ✅ Hygiene 全绿
- ✅ 无 backwards-compat 代码 (Cell 完全移除，无 Option wrapper)

## 7. Risks

| R# | Risk | Mitigation |
| --- | --- | --- |
| **R1** | 现有 unit tests 依赖 `Sampler::with_seed(N)` → bit-exact 输出 | Plan T2 update tests: statistical (1000 sample freq) 或 update seed/expected pairs; document drift in test comment |
| **R2** | `Scheduler.prng_state` evict 时不 zero → 下次 admit 同 row 可能用旧 key | `init_row_prng` 在 admit 时 overwrite — 上一行 evict 不需 clear; R2 minor |
| **R3** | `mlx::ops::indexing::slice_update` 在 hot path overhead 重 | Plan T1 bench: 1 sample_row_cpu call 增加 N µs slice_update cost? 若太重则 batch end stack-rewrite alternative |
| **R4** | `admit_mid_finalize` B=1 PRNG state 与 batch state 不一致 | Plan T2 选 path: (a) Sampler::sample 签名也加 prng_state，admit_mid_finalize 传 single-row slice; (b) admit_mid_finalize 切到 sample_row_cpu 减少 code paths |
| **R5** | `mlx::random::key(seed)` mint cost 在 admit-time 阻塞 admission latency | mlx::random::key 是 cheap (creates [2] u32 from seed via hash); admit overhead 可忽略 |
| **R6** | Sampler-derived `Copy` traits broke down-stream code 假设 `Sampler` not Copy | Static analysis: scan ironmlx codebase for `*sampler` or `clone()` patterns; auto-Copy is strict superset |

## 8. Implementation plan decomposition (preview)

Plan 文档 (`docs/superpowers/plans/2026-05-18-b1-p2-3e-2-prng-key-centralization.md`) 会拆约 3-4 tasks:

- **T0 (optional, ~0.5h)** — `mlx::ops::indexing::slice_update` API verify + 1-row write/read probe + perf micro-bench (R3 mitigation pre-check)
- **T1 (~0.5d)** — `Sampler` struct shrink: remove `Cell<Option<Array>>` field + `ensure_key`/`store_key` fns + manual `Clone` impl; add `#[derive(Clone, Copy)]`; static_assertions for Send + Sync
- **T2 (~1d)** — `Scheduler.prng_state` field + `init_row_prng` + `sample_row_cpu` signature change + `configured_pipeline` signature change + `sample_batch` signature change + `Scheduler::step` / `prefill_admitted_inner` / `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` / `admit_mid_finalize` call site updates
- **T3 (~0.5d)** — Test updates: fix expected token values (statistical or value adjust); ensure perf gate compatible
- **T4 (~0.5d)** — perf gate run + sweep_smoke + sweep_full + close-out

预估总: 2-3 days。

## 9. Carry-forward (post-3e.2)

3e.2 后 sampler vectorization series 收尾。后续路径:

- **Qwen3.5 MoE** — main path forward
- **(Optional) GPU sample re-enable** — 等 mlx 暴露 `scatter_along_axis` + 优化 categorical Metal kernel cache (T4 hot-fix carry-forward)
- **(Optional) Sweep_full hygiene** — cooldown / shard for parallel (3e.1b carry-forward)

---

**Document history:**

- 2026-05-17 — Initial outline (commit 92b5b50, brainstormed alongside 3e.1b spec)
- 2026-05-18 — Rewritten to plan-grade detail post-3e.1b ship (T4 GPU→CPU handoff adjustment)
