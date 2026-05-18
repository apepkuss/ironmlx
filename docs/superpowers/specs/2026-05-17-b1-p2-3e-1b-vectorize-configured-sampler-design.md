# B1-p2.3e.1b Vectorize Configured Sampler — Design

**Status:** Draft (brainstormed 2026-05-17, post-3e.1a ship)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Predecessor:** [B1-p2.3e.1 vectorized sampler series](2026-05-17-b1-p2-3e-1-vectorized-sampler-design.md) (3e.1a shipped commits 7719998..ab4c839)
**Successor:** [B1-p2.3e.2 PRNG key batching](2026-05-17-b1-p2-3e-2-prng-key-batching-design.md) (lightweight outline)
**Branch target:** `ironmlx-b1-p2-3e1b-vectorize-configured` (cut from `ironmlx-b1-p2-3e1a-vectorize-greedy` HEAD `ab4c839`)

## 0. Program context

| Sub-spec | Status |
| --- | --- |
| 3e.1a vectorize greedy argmax (all-greedy fast path) | ✅ DONE (2026-05-17) |
| **3e.1b vectorize configured sampler** (7 ops + batched categorical) | **This spec** |
| 3e.2 PRNG key state centralization | Outline ([sibling spec](2026-05-17-b1-p2-3e-2-prng-key-batching-design.md)) |
| Top_k via custom Metal partial-sort kernel | NG (defer to 3f or future) |
| B1-p2.5 production hardening | Future |

## 1. Motivation

3e.1a 完成 `sample_batch` 的 all-greedy fast path：B 个 `.item()` 同步 → 1 个 `argmax([B, vocab])` + 1 个 `to_vec::<u32>()` (1 GPU dispatch + 1 host sync)。Production traffic 多数走此路径，3e.1a 实测 per-row median 64.7 ms 全 4 rows lockstep 1.00x。

但 mixed/configured batch（任一 row 有 temperature / top_p / repetition_penalty / 等非 greedy 配置）仍走 `sample_batch` 内 per-row fallback：B 次 `slice_strided_on` + B 次 `Sampler::sample` (内含 6 op + categorical 单 sample)。fallback path 是 B × (6 GPU op + 1 GPU sync) ≈ B × 4-8 ms = 16-32 ms per step at B=4 — 比 fast path 慢 5-10×。

**3e.1b 目标**：把 fallback path 也改成 batched GPU pipeline。Mixed batch 内所有 rows 走同一条 `[B, vocab]` 流水线（uniform configured pipeline），per-row config 通过 broadcasted no-op default value 控制启用（temp=1.0 / top_p=1.0 / top_k=vocab_size / min_p=0.0 / rep_pen=1.0 / freq_pen=0.0 / pres_pen=0.0 全部 identity）。最后用 `mlx::random::categorical(probs, key=[B, 2])` 一次 batched sample。

收益：mixed batch fallback 从 B 次 dispatch → 7-8 次 batched dispatch + 1 次 sync。Per-step sampler block 从 16-32 ms → 3-6 ms at B=4。

## 2. Goals

- **G1.** 实现 7 个 batched logits/probability ops：`apply_penalties`（rep_pen + freq_pen + pres_pen 合 1 op）/ `apply_temperature` / `apply_top_k` / `softmax` / `apply_top_p` / `apply_min_p` / `renormalize`。
- **G2.** 实现 batched categorical sample step：collect per-row PRNG keys from Sampler.cell → stack `[B, 2]` → split → `mlx::random::categorical(probs, key=sample_keys)` → distribute new keys 回 Cell。
- **G3.** `sample_batch` routing 不变：all-greedy → 3e.1a fast path；mixed → configured pipeline。Sampler struct 不变（保持 3e.2 之前 backward compat）。
- **G4.** Short-circuit `apply_penalties` when no row 需要 history (rep/freq/pres 全 None across batch) — 跳过 CPU bincount + upload + GPU op。
- **G5.** History encoding via CPU bincount → upload `[B, vocab] u32`（vs GPU scatter_add over padded history — 长 history (>= 8K) 浪费严重）。
- **G6.** Per-op unit tests + mixed-batch parity test + 真模型 perf gate 集成测试 + sweep_full 17 suites regression PASS。

## 3. Non-goals

- **NG1.** Top_k partial-sort custom Metal kernel — 用 `mlx::ops::sort` (full sort) 实现，partition 形式作为 verify-time fallback (R2)。Custom kernel defer 到 3f 或后续。
- **NG2.** PRNG state ownership 集中化（Sampler.cell → Scheduler）— 3e.2 范围，3e.1b 仍 collect/distribute 接口。
- **NG3.** Chat completion API 端的 logit_bias / structured output schema — sampler vectorization 不在此层。
- **NG4.** Mixed-config batch 内 server-side request 路由优化 — 假设 admit_inner 内 `RequestState.sampler` 已就绪。
- **NG5.** 3e.1a all-greedy fast path 重构 — 保留 unchanged。
- **NG6.** Reproducibility bit-exact parity vs pre-3e.1b `Sampler::sample` 单 row 路径 — 接受 mlx::random batched 实现的 numerical drift (R4)。测试只校验 statistical / identity property。

## 4. Design

### 4.1 `sample_batch` routing (post-3e.1b)

```rust
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,       // [B, vocab]
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    // ... validation (same as 3e.1a) ...

    // Fast path: all-greedy
    if samplers.iter().all(|s| s.is_greedy()) {
        let ids = reduction::argmax(logits, -1, false)?;
        return Ok(ids.to_vec()?);
    }

    // Configured pipeline (NEW in 3e.1b)
    configured_pipeline(samplers, logits, histories)
}
```

`configured_pipeline` 是新增 free function in `core/sampler.rs`。

### 4.2 `configured_pipeline` 整体结构

```rust
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,        // [B, vocab]
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let b = samplers.len();
    let vocab = logits.shape().as_slice()[1] as usize;

    // §4.3 — collect per-row config tensors with no-op defaults
    let configs = collect_per_row_configs(samplers)?;

    // §4.4 — history bincount (short-circuited when no row needs it)
    let history_count = if configs.need_history {
        Some(build_history_count(histories, vocab)?)
    } else {
        None
    };

    // §4.5 — pipeline
    let logits = apply_penalties(logits, history_count.as_ref(), &configs)?;
    let logits = apply_temperature(&logits, &configs.temp)?;
    let logits = apply_top_k(&logits, &configs.top_k)?;
    let probs = mlx::ops::softmax(&logits, &[-1], None)?;
    let probs = apply_top_p(&probs, &configs.top_p)?;
    let probs = apply_min_p(&probs, &configs.min_p)?;
    let probs = renormalize(&probs)?;

    // §4.6 — batched categorical with PRNG key collect/distribute
    sample_batched_categorical(samplers, &probs)
}
```

### 4.3 Per-row config collection

```rust
struct PerRowConfigs {
    temp: Array,         // [B] f32, None → 1.0
    top_k: Array,        // [B] i32, None → vocab_size (no-op)
    top_p: Array,        // [B] f32, None → 1.0
    min_p: Array,        // [B] f32, None → 0.0
    rep_pen: Array,      // [B] f32, None → 1.0
    freq_pen: Array,     // [B] f32, None → 0.0
    pres_pen: Array,     // [B] f32, None → 0.0
    need_history: bool,  // any of rep/freq/pres has Some
}

fn collect_per_row_configs(samplers: &[&Sampler]) -> Result<PerRowConfigs> {
    let b = samplers.len();
    // For each row, extract config value or fall back to no-op default.
    // Build Array directly from Vec<f32> / Vec<i32>.
    // Set need_history = samplers.iter().any(|s|
    //     s.repetition_penalty.is_some() ||
    //     s.frequency_penalty.is_some() ||
    //     s.presence_penalty.is_some()
    // )
    ...
}
```

### 4.4 History encoding (CPU bincount → upload)

```rust
fn build_history_count(histories: &[&[u32]], vocab: usize) -> Result<Array> {
    let b = histories.len();
    let mut flat = vec![0_u32; b * vocab];
    for (row, hist) in histories.iter().enumerate() {
        let offset = row * vocab;
        for &tok in *hist {
            flat[offset + tok as usize] += 1;
        }
    }
    Ok(Array::from_slice(&flat, &[b as i32, vocab as i32]))
}
```

Cost (Qwen3.5 vocab=151936, B=4, history≈512 tok):

- CPU bincount: 4 × 512 ≈ 2K op，<100µs
- Upload `[B, vocab] u32`: 2.43MB, UMA bandwidth ~30 GB/s → ~30-80µs
- 总 <150µs per step — 0.2% of 64.7ms step budget

**Short-circuit** (G4): `if !configs.need_history → skip build_history_count + skip apply_penalties op`. Production 多数请求只 `temp + top_p` 无 penalty，此短路常生效。

### 4.5 Per-op specifications

#### 4.5.1 `apply_penalties` (rep + freq + pres 合 1 op)

```rust
fn apply_penalties(
    logits: &Array,            // [B, vocab]
    history_count: Option<&Array>,  // [B, vocab] u32, None → skip
    configs: &PerRowConfigs,
) -> Result<Array> {
    let history_count = match history_count {
        None => return Ok(logits.clone()),  // short-circuit
        Some(h) => h,
    };
    let history_mask = greater(history_count, &Array::from(0_u32))?;  // [B, vocab] bool

    // Repetition: where(logit > 0, logit / rep_pen, logit * rep_pen) for seen tokens
    let rep_pen_bv = configs.rep_pen.reshape(&[b as i32, 1][..])?;  // broadcast
    let rep_inv = divide(&Array::from(1.0_f32), &rep_pen_bv)?;
    let positive_logit_mask = greater(logits, &Array::from(0.0_f32))?;
    let rep_factor = where_(&positive_logit_mask, &rep_inv, &rep_pen_bv)?;
    let logits_rep = where_(&history_mask, &multiply(logits, &rep_factor)?, logits)?;

    // Frequency: logit -= freq_pen * count
    let freq_pen_bv = configs.freq_pen.reshape(&[b as i32, 1][..])?;
    let logits_freq = subtract(&logits_rep, &multiply(&freq_pen_bv, &history_count.cast(Dtype::Float32)?)?)?;

    // Presence: logit -= pres_pen * (history_mask as f32)
    let pres_pen_bv = configs.pres_pen.reshape(&[b as i32, 1][..])?;
    let presence_term = multiply(&pres_pen_bv, &history_mask.cast(Dtype::Float32)?)?;
    let logits_pres = subtract(&logits_freq, &presence_term)?;

    Ok(logits_pres)
}
```

合一让 MLX lazy graph fuse 3 个 op，减少 intermediate Array eval。

#### 4.5.2 `apply_temperature`

```rust
fn apply_temperature(logits: &Array, temp: &Array) -> Result<Array> {
    let temp_bv = temp.reshape(&[temp.shape().as_slice()[0], 1][..])?;
    divide(logits, &temp_bv)
}
```

进入 configured pipeline 的 row 至少有一个 non-default config，所以 `temp` 不会全 0。Greedy row 在 batch 中 mixed 时 `temp` 字段是 1.0 (no-op default)，不会触发 div-by-zero。

#### 4.5.3 `apply_top_k` (用 `mlx::ops::sort`，full sort)

```rust
fn apply_top_k(logits: &Array, top_k: &Array) -> Result<Array> {
    // top_k: [B] i32, no-op = vocab_size
    let sorted = mlx::ops::sort(logits, -1)?;                          // [B, vocab] ascending
    // descending equivalent: sorted_desc[i] = sorted[vocab - 1 - i]
    // threshold = k-th largest = sorted[vocab - top_k[i]]
    let vocab = logits.shape().as_slice()[1];
    let topk_idx_from_start = subtract(&Array::from(vocab as i32), top_k)?;  // [B]
    let threshold = gather_along_axis(&sorted, &topk_idx_from_start.reshape(&[..., 1])?, -1)?;
    where_(&greater_equal(logits, &threshold)?, logits, &Array::from(f32::NEG_INFINITY))
}
```

no-op 时 `top_k = vocab_size` → `topk_idx = 0` → threshold = sorted[0] = min logit → mask 全 true → identity.

R2: `mlx::ops::sort` 在 vocab=151K × B=4 在 M1 Pro 上预计 < 2ms。Plan T0 早期 bench；如果太慢则 R2 mitigation 用 `mlx::ops::partition` (partial sort, 如果 mlx-rs expose 此 API)。

#### 4.5.4 `softmax`

```rust
let probs = mlx::ops::softmax(&logits, &[-1], Some(false))?;  // numerically stable, subtract max
```

#### 4.5.5 `apply_top_p` (nucleus sampling, probability space)

```rust
fn apply_top_p(probs: &Array, top_p: &Array) -> Result<Array> {
    // Sort descending: 用 negate + ascending sort 或 mlx::ops::sort 加 `reverse=true` 参数（plan T0 verify mlx-rs 暴露形式）
    let sort_idx_desc = argsort_along_axis_descending(probs, -1)?;  // [B, vocab] i32 indices
    let sorted_probs = gather_along_axis(probs, &sort_idx_desc, -1)?;  // [B, vocab]

    // Cumsum + nucleus mask (保留第一个 cross threshold 的 token)
    let cumsum = mlx::ops::cumsum(&sorted_probs, -1, false, false)?;
    let top_p_bv = top_p.reshape(&[b as i32, 1][..])?;
    let cumsum_excl = subtract(&cumsum, &sorted_probs)?;  // 上一个位置的 cumsum
    let mask_sorted = less(&cumsum_excl, &top_p_bv)?;     // [B, vocab] bool
    let sorted_probs_masked = where_(&mask_sorted, &sorted_probs, &Array::from(0.0_f32))?;

    // Scatter sorted_probs_masked 回 vocab 原顺序:
    //   原顺序 probs_out[i, sort_idx_desc[i, j]] = sorted_probs_masked[i, j]
    // 用 scatter_along_axis：scatter(zeros[B, vocab], sort_idx_desc, sorted_probs_masked, axis=-1)
    let zeros = mlx::ops::zeros_like(probs)?;
    scatter_along_axis(&zeros, &sort_idx_desc, &sorted_probs_masked, -1)
}
```

实现细节（descending sort / scatter_along_axis）走 mlx-rs 实际 API 形式，plan T0 verify (R3)。no-op (`top_p=1.0`) 时 cumsum_excl 总 < 1.0 → mask 全 true → identity。

#### 4.5.6 `apply_min_p`

```rust
fn apply_min_p(probs: &Array, min_p: &Array) -> Result<Array> {
    let max_prob = mlx::ops::max(probs, &[-1], true)?;  // [B, 1]
    let min_p_bv = min_p.reshape(&[..., 1])?;
    let threshold = multiply(&min_p_bv, &max_prob)?;
    where_(&greater_equal(probs, &threshold)?, probs, &Array::from(0.0_f32))
}
```

no-op (`min_p=0.0`) → threshold = 0 → mask 全 true → identity.

#### 4.5.7 `renormalize`

```rust
fn renormalize(probs: &Array) -> Result<Array> {
    let sum = mlx::ops::sum(probs, &[-1], true)?;  // [B, 1]
    divide(probs, &sum)
}
```

Top_p / min_p 会把部分 entries 设 0，renormalize 让 categorical 概率和为 1。

### 4.6 Batched categorical (sample step + PRNG plumbing)

```rust
fn sample_batched_categorical(samplers: &[&Sampler], probs: &Array) -> Result<Vec<u32>> {
    let b = samplers.len();

    // Stage A: collect per-row PRNG keys from Sampler.cell
    let keys_vec: Vec<Array> = samplers.iter().map(|s| {
        s.prng_cell.take().unwrap_or_else(|| make_key_for_sampler(s))
    }).collect();
    let keys_stacked = mlx::ops::stack(&keys_vec, 0)?;          // [B, 2] u32

    // Stage B: split into (advance_keys, sample_keys)
    let split = mlx::random::split(&keys_stacked, 2)?;          // [2, B, 2]
    let advance_keys = split.slice_axis(0, 0, 1)?.squeeze(0)?;  // [B, 2]
    let sample_keys = split.slice_axis(0, 1, 2)?.squeeze(0)?;   // [B, 2]

    // Stage C: batched sample
    let tokens = mlx::random::categorical(probs, &sample_keys, -1, ())?;  // [B] u32

    // Stage Z: distribute new state back into per-row Cells
    for i in 0..b {
        let row_key = advance_keys.slice_axis(0, i as i32, i as i32 + 1)?.squeeze(0)?;
        samplers[i].prng_cell.set(Some(row_key));
    }

    Ok(tokens.to_vec::<u32>()?)
}

fn make_key_for_sampler(s: &Sampler) -> Array {
    // First-use case: derive from s.seed if set, else from a fresh global RNG.
    match s.seed {
        Some(seed) => mlx::random::key(seed),
        None => mlx::random::key(default_seed_from_time_xor_addr(s)),
    }
}
```

R1 — `mlx::random::categorical` 接受 batched key `[B, 2]` 行为需 plan T0 verify。若不支持：fallback per-row categorical (B 次 dispatch) — logits ops 仍 batched (节省 90%+)。

### 4.7 Sampler struct (unchanged in 3e.1b)

3e.1b **不** 改 Sampler struct。`prng_cell: Cell<Option<Array>>` 保留，sample_batch 通过 `take/set` 接口操作。3e.2 才把 PRNG state 从 Sampler 移到 Scheduler。

### 4.8 Mixed-batch semantics

- Batch row config 任意组合 (e.g., row0 greedy / row1 temp+top_p / row2 +rep_pen) 都通过 configured pipeline 处理
- Per-row 不需要的 op 通过 no-op default value 控制（结果 identity）
- All-greedy 仍走 3e.1a fast path（不进 configured pipeline）
- Sentinel sampler (pad rows / inactive slots) — 用 `Sampler::greedy()` (从 3e.1a 继承)，但因为 mixed batch 内有 non-greedy row，所以 sentinel 也走 configured pipeline。Sentinel 的 logits 是 pad row 的 garbage，但 Stage C distribute 会过滤掉非 active row（Scheduler::step 内 `active_at_start` mask）— 所以 sentinel 输出 token 被静默丢弃。

## 5. Risks

| R# | Risk | Severity | Mitigation |
| --- | --- | --- | --- |
| **R1** | `mlx::random::categorical` 不接受 batched key `[B, 2]` | High | Plan T0 verify。Fallback: per-row categorical (B dispatch, logits ops 仍 batched，性能让步 ~90%→~70% 改善) |
| **R2** | `mlx::ops::sort` 在 vocab=151K × B=4 太慢拖 step latency | Medium | T1 早期 bench。Fallback: `mlx::ops::partition` (partial sort), 或保持 sort 但记入 perf budget |
| **R3** | top_p scatter (sorted → vocab order) mlx-rs API 不齐 | Medium | T0 探索 `gather_along_axis` / `scatter_along_axis` 是否 expose；fallback: 用 indexing reshape 等价实现 |
| **R4** | Batched categorical 与 per-row 单 sample numerical drift | Low | 接受 drift。测试只校验 identity property + statistical freq check (1000 sample) |
| **R5** | Production traffic 多数 greedy → 3e.1b perf gate 无 traffic 触发 | Low | All-greedy fast path 兜底；3e.1b 收益靠 HTTP API 端 non-default config request 体现；perf gate 测试构造 mixed batch |
| **R6** | `apply_penalties` 合一 op debug 难 (op fusion 时不分离 intermediate) | Low | cfg(test) hook 拆 3 op verify per-stage；prod 路径合一 |
| **R7** | Long-context history (>= 8K) bincount CPU 退化 | Low | 实测 8K history × 4 row = 32K op, < 300µs，仍可接受。Long-context optimization (incremental update) defer |
| **R8** | `Sampler.cell.take/set` 跨 sample_batch 调用 thread-safety | Low | sample_batch 仅 Scheduler 调用 (`&mut self` 串行)；`&Sampler` 借用 stable 不跨线程 |

## 6. Acceptance criteria + tests

### 6.1 Unit tests (`core/sampler.rs::mod tests`)

7-10 new tests：

- `sample_batch_temperature_scales_logits` — B=2 全 temp=2.0, 校验 sorted logits 顺序保持 (temp 不改 argmax)
- `sample_batch_top_p_truncates_nucleus` — B=2 top_p=0.5 + skewed probs, 校验 sample 出 token 都在 nucleus
- `sample_batch_top_k_keeps_topk` — B=2 top_k=3, 校验 sample 出 token 在 top 3 argsort
- `sample_batch_min_p_filters_low_prob` — B=2 min_p=0.5, 校验低 prob token (< 0.5 × max_prob) 不被 sample
- `sample_batch_repetition_penalty_reduces_seen` — B=1 rep_pen=2.0 + history=[5], 验证 token 5 logit 被 divided by 2
- `sample_batch_frequency_penalty_proportional` — B=1 freq_pen=1.0 + history=[5,5,5], 验证 token 5 logit -= 3
- `sample_batch_presence_penalty_binary` — B=1 pres_pen=1.0 + history=[5,5,5], 验证 token 5 logit -= 1（不是 -3）
- `sample_batch_mixed_some_greedy_some_top_p` — B=4 row0/1 greedy + row2/3 top_p=0.9, verify row0/1 取 argmax + row2/3 在 nucleus
- `sample_batch_no_op_default_identity` — B=1 全 no-op default 值 (temp=1, top_p=1, top_k=vocab, ...) verify output == greedy argmax
- `sample_batch_short_circuit_no_history` — B=4 全 rows 无 rep/freq/pres → verify `apply_penalties` 不调用 (mock observer)

### 6.2 Parity test

- `sample_batch_parity_with_per_row_temperature_top_p` — B=4 同 config + fixed seed (via `Sampler::with_seed`)，verify `sample_batch(...)` output == [`per-row Sampler::sample(slice_i, ...)` for i in 0..B]
  - 接受 1-ulp drift（mlx batched vs per-row 实现可能 floating-point order 差）— 改用 freq check (1000 sample 后 distribution 一致) 作为 statistical parity

### 6.3 Real-model perf gate (`#[ignore]`)

新 `ironmlx/tests/b1_p2_3e_1b_configured_sampler.rs`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3e_1b_configured_decode_speedup() {
    // 4 concurrent admits with: temperature=0.7, top_p=0.9, repetition_penalty=1.1
    //   → 100% configured batch → hits 3e.1b configured pipeline (NOT 3e.1a fast path)
    // Measure per-row median inter-token gap, assert:
    //   - per-row medians within 2× of each other (lockstep proof)
    //   - max median ≤ 250 ms (configured pipeline > fast path; 3e.1a 64.7 ms argmax,
    //     configured adds ~50-100 ms for 7-op + categorical)
}
```

Acceptance threshold 设 250 ms (vs 3e.1a 200 ms) — configured pipeline 比 argmax 重一倍 (8 ops vs 1)。Plan 阶段实测后可能 tighten。

### 6.4 Sweep gates

- `sweep_smoke.sh --suites b1_p2_3b_2_scheduler_actor b1_p2_4_batched_vl::mid_admit_vl_during_text_decode b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup b1_p2_3e_1b_configured_sampler::b1_p2_3e_1b_configured_decode_speedup` PASS
- `sweep_full.sh` 17 suites (16 现有 + 1 新增) PASS

### 6.5 Hygiene gates (every commit)

- `cargo +nightly fmt --all -- --check`
- `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`
- `cargo +stable build --release`

### 6.6 No backwards-compat

- `Sampler::sample` 仅在 `admit_mid_finalize` 保留（B=1 intentional）— 不创建 sample_batch 的 wrapper
- 3e.1a all-greedy fast path 保留不变
- 任何 dead code / commented-out 旧路径必须删除

## 7. Implementation plan decomposition (preview)

Plan 文档 (`docs/superpowers/plans/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler.md`) 会拆成约 5 tasks：

- **T0** — mlx API verification: `mlx::random::categorical` batched key、`mlx::ops::sort` perf、`gather_along_axis`/`scatter_along_axis` expose 状况；mitigate R1/R2/R3
- **T1** — `collect_per_row_configs` + `build_history_count` + `apply_penalties` (1 op for rep+freq+pres) + 3 unit tests
- **T2** — `apply_temperature` + `apply_top_k` + `softmax` + `apply_top_p` + `apply_min_p` + `renormalize` + 5 unit tests
- **T3** — `sample_batched_categorical` + PRNG plumbing + mixed-batch unit + parity test + 整合 `configured_pipeline` 入 `sample_batch`
- **T4** — `Scheduler::step` / `prefill_admitted_inner` call site 兼容性验证（sample_batch 签名不变，应零 change）+ 真模型 perf gate 集成测试 + sweep_smoke + sweep_full + close-out report

Plan 工作量预估：3-4 天 (T0 < 0.5d, T1 ~ 1d, T2 ~ 1d, T3 ~ 0.5-1d, T4 ~ 0.5d)。

## 8. Open questions (deferred to plan T0)

- mlx-rs 暴露的 `mlx::random::categorical` 是否接受 `[B, vocab] probs + [B, 2] key` shape？(R1)
- mlx-rs 是否 expose `mlx::ops::partition` 还是只 `sort`？(R2 mitigation)
- `gather_along_axis` / `scatter_along_axis` 在 mlx-rs 的实际名字与 signature？(R3)
- mlx-rs 是否提供 `reverse_along_axis` 还是要 `slice_strided_on` with negative stride？(R3 mitigation)

T0 task 直接读 `/Volumes/Dev/mlx-rs` source 验证。

## 9. Carry-forward

- **3e.2** — PRNG state ownership 集中化（参见 [sibling spec](2026-05-17-b1-p2-3e-2-prng-key-batching-design.md)）
- **Future** — Top_k partial-sort custom Metal kernel（如 `mlx::ops::sort` 性能不达标）
- **Future** — apply_penalties 用单 fused Metal kernel 替代 3 op pipeline（如 graph fusion 不足）

---

**Document history:**

- 2026-05-17 — Initial draft (post-3e.1a ship, Boss-approved series-level brainstorming)
