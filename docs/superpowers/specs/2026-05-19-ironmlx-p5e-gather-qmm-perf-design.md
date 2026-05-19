# ironmlx P5e — SparseMoeBlock gather_qmm 性能优化设计

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-19 |
| 状态 | Brainstorming approved，准备 writing-plans |
| 范围 | SparseMoeBlock 内三个 `gather_quantized_matmul_on` 调用的优化（gate / up / down projection） |
| 工作分支 | `ironmlx-p5e-perf`（已建，base `ironmlx-p5-moe`） |
| 上游分支 | `ironmlx-p5-moe` → `ironmlx` |
| 硬件 | M5 Max + 128 GB unified |
| 验证模型 | `mlx-community/Qwen3.5-35B-A3B-4bit`（PP=128/512/2048 prefill） |
| 验收 | ironmlx 自身 before/after wall-clock 改进；decode/sweep/数值精度无回归 |
| 显式 out-of-scope | GatedDeltaNet / GatedAttention 优化（留下一 phase）；自写 Metal kernel（不引入）；与 omlx/mlx-vlm 数字对齐 |

## § 1 调研依据与决策摘要

### 1.1 T0 profile 数据（PP=2048）

来源：`reports/p5e-t0-profile.md`（commit `f47d471`），M5 Max 128GB 直 `Model::forward_on`（无 HTTP 路径）。

PP=2048 prefill 总 2223.6 ms（921 tok/s）。Hot path：

| 等级 | Op | Time (ms) | % wall-clock |
|---|---|---|---|
| 1 | `gather_qmm_down` | 642.1 | 28.9% |
| 2 | `gather_qmm_gate` | 408.5 | 18.4% |
| 3 | `gather_qmm_up` | 389.2 | 17.5% |
| — | **3× gather_qmm 合计** | **1439.8** | **64.8%** |
| 4 | GatedDeltaNet (30 linear-attn 层) | 458.3 | 20.6% |
| 5 | GatedAttention (10 full-attn 层) | 145.3 | 6.5% |

非 hot：所有 routing / softmax / SwiGLU activation / shared_expert / embed / norm / lm_head 累加 < 8.1%。

P5e 仅动 #1-3（gather_qmm）。其余留后续 phase。

### 1.2 公平性 caveat

P5d T2 报的 "ironmlx prefill -76% vs omlx" 反映 ironmlx vanilla vs omlx (mlx-vlm + body-replacement patches + paged cache) 的差异，非 ironmlx 实现差。详见 memory [project_omlx_perf_baseline](../../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_omlx_perf_baseline.md) 2026-05-19 update。

**P5e 优化目标基于 T0 profile 找到的 ironmlx 自身 hot path，不是赶上 omlx 数字**（per [feedback_no_spec_from_competitors](../../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_no_spec_from_competitors.md)）。

### 1.3 决策摘要

| 决策项 | 选定方案 | 来源 |
|---|---|---|
| **Q1 范围** | 仅 SparseMoeBlock 内 3 个 gather_qmm | 集中影响面，单 phase 收敛快 |
| **Q2 风险** | 允许算法重构（expert dedup / sorted routing）；**不写 Metal kernel** | 留 Metal kernel 至 P5e+ 或 P8 精度阶段 |
| **Q3 验收** | ironmlx 自身 before/after 改进；decode/sweep/数值无回归；**不定具体 tok/s 数字** | 哲学：不对齐竞品 |
| **Approach** | C = Stage 1 (A: op 重排/compile/parallelism) → Stage 2 (B.1: sorted routing) 串行 | A low-risk quick win 先 ship；B.1 在 A 之上叠加；B.2 grouped matmul 留后续 |

## § 2 架构 / 数据流

### 2.1 总览（2 stage 串行）

```
P5e 起点（current ironmlx-p5e-perf baseline）
   ↓
T0 — baseline 重测（M5 Max 当前实现 PP=128/512/2048 wall-clock 3 runs median）
   ↓
Stage 1 — Approach A（MLX op 重排）
   T1 A.1 — stream parallelism
   T2 A.2 — mlx::compile wrap
   T3 A.3 — shape elimination
   T4 — Stage 1 close-out（选保留组合，full sweep_full + before/after 报告）
   ↓
Stage 2 — Approach B.1（sorted routing）
   T5 — argsort/scatter + sorted_indices=true 实现
   T6 — Stage 2 close-out + P5e final close-out
   ↓
P5e final reviewer + report
```

每 stage 之间 measure + Boss approve 是 explicit gate。Stage 2 依赖 Stage 1 已 ship。

### 2.2 Stage 1 (Approach A) 数据流分项

#### A.1 — Stream parallelism

当前实现：
```rust
let gate_out = mlx::quantization::gather_quantized_matmul_on(&x_in, &gate_w, ..., target)?;
let up_out   = mlx::quantization::gather_quantized_matmul_on(&x_in, &up_w,   ..., target)?;
// gate_out 与 up_out 逻辑独立但 dispatch 串行
```

A.1 改造：把 gate / up 派到不同 stream，down 在 stream-join 后串行：
```rust
let s_gate = mlx::Stream::new(mlx::Device::gpu(0))?;
let s_up   = mlx::Stream::new(mlx::Device::gpu(0))?;
let gate_out = mlx::quantization::gather_quantized_matmul_on(..., s_gate)?;
let up_out   = mlx::quantization::gather_quantized_matmul_on(..., s_up)?;
// down 等到 gate + up 都 eval 完后再调用
```

假设：MLX scheduler / Metal GPU 能并行执行两个 stream 上的 kernels（同 GPU 不同 command queue）。如果硬件 / MLX 不支持，A.1 = 0 增益（accept and ship）。

#### A.2 — `mlx::compile` wrap

把 `SparseMoeBlock::forward_on` 内的 lazy graph 用 `mlx::compile(.., ShapeMode::Shapeless)` 编译。Shapeless 让 prefill (变 PP) 和 decode (PP=1) 共用一份编译图。

可能收益：
- MLX 把 SwiGLU 的 sigmoid + 两个 element-wise mul 融进 gather_qmm 的 epilogue（减少 kernel launch）
- 减少 Python/Rust → Metal 的 dispatch 数

可能不奏效：
- MLX compile 对 quantized matmul + gather op 可能不支持（需 T2 实验验证）
- 实测可能编译开销 > 收益（compile 本身有时间）

#### A.3 — Shape elimination

T0 profile 显示 `expand_dims` 0.03 ms / call（极便宜），但 `squeeze` 在 down_out 后被 4D→3D；如果 weighted_sum 之前才 squeeze，可以省掉 down_out 后的 squeeze。

当前数据流：
```
x_in: [BS, 1, 1, H]
gate_out, up_out: [BS, k, 1, moe_inter]
act = silu(gate) * up: [BS, k, 1, moe_inter]
down_out_4d: [BS, k, 1, H]
down_out = squeeze(down_out_4d, -2): [BS, k, H]    ← 1 个 squeeze kernel
scores_unsq = expand_dims(scores, -1): [BS, k, 1]
weighted = down_out * scores_unsq: [BS, k, H]
routed_y = sum(weighted, -2): [BS, H]
```

A.3 改造：保留 4D 一直到 weighted_sum：
```
down_out_4d: [BS, k, 1, H]
scores_unsq = expand_dims(scores, [-1, -2]): [BS, k, 1, 1]
weighted = down_out_4d * scores_unsq: [BS, k, 1, H]
routed_y = sum(weighted, [-2, -3], keepdim=false): [BS, H]   ← 单次 reduction
```

或者更激进：sum 直接对 [BS, k, 1, H] 在 axis=(-2,-3) 上做（结果 [BS, H]），把 squeeze + sum 合并成一次 reduction。

预期 ~0.5 ms 节省（squeeze + 一次 reshape）。

### 2.3 Stage 2 (Approach B.1) 数据流

现在的 gather_quantized_matmul_on 调用 `sorted_indices: bool = false`。改为 `true` 前需要 token 按 expert id 排好序。

```rust
// 1. flatten indices
let flat_topk = topk_idx.reshape(&[bs * k][..])?;           // [BS*k] uint32

// 2. permutation
let sort_perm = mlx::ops::sort::argsort(&flat_topk, -1)?;    // [BS*k] uint32
let inv_perm  = mlx::ops::sort::argsort(&sort_perm, -1)?;    // [BS*k] uint32

// 3. sort indices & corresponding tokens
let sorted_topk = mlx::ops::indexing::take_along_axis(&flat_topk, &sort_perm, -1)?;
// x_in 是 [BS, 1, 1, H] — broadcast 到 [BS, k, 1, H] 后排序，or 直接对 x_in 重排
let x_in_2d = x_in.reshape(&[bs, h][..])?;                   // [BS, H]
let x_sorted_2d = mlx::ops::indexing::gather_along_axis(&x_in_2d, /* token-from-flat-bs/k */, ..)?;
// 实际 indices to gather: each of [BS*k] positions needs original token's x → repeat x token-index k times then gather by sort_perm

let x_sorted = x_sorted_2d.reshape(&[bs * k, 1, 1, h][..])?; // [BS*k, 1, 1, H]
let sorted_topk_2d = sorted_topk.reshape(&[bs * k, 1][..])?; // [BS*k, 1] (k axis collapsed)

// 4. gather_qmm with sorted_indices=true
let gate_out_sorted = mlx::quantization::gather_quantized_matmul_on(
    &x_sorted, &gate_w, &gate_s, ..., Some(&sorted_topk_2d), 
    true /* transpose */, ..., true /* sorted_indices */, target,
)?;
// gate_out_sorted: [BS*k, 1, 1, moe_inter]
// 同 up, down

// 5. SwiGLU
let act_sorted = (&gate_out_sorted.sigmoid_on(target)? * &gate_out_sorted)? * &up_out_sorted;
let down_out_sorted = gather_quantized_matmul_on(&act_sorted, &down_w, ..., true)?;
// down_out_sorted: [BS*k, 1, 1, H]

// 6. scatter back via inv_perm
let down_out_unsort = mlx::ops::indexing::take_along_axis(
    &down_out_sorted.reshape(&[bs * k, h][..])?,
    &inv_perm.reshape(&[bs * k, 1][..])?,
    0,
)?;                                                          // [BS*k, H] back in original order
let down_out_3d = down_out_unsort.reshape(&[bs, k, h][..])?; // [BS, k, H]

// 7. weighted sum (unchanged downstream)
let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, &[-1_i32][..], target)?;
let weighted = (&down_out_3d * &scores_unsq)?;
let routed_y = mlx::ops::sum_on(&weighted, &[-2_i32][..], false, target)?;
```

实际细节在实施时调整（如 `gather_along_axis` 的 indices 构造、`take_along_axis` 的具体维度等）。

预期增益机制：MLX gather_qmm 的 `sorted_indices=true` 分支按 expert 连续访问权重，cache locality 改善。

## § 3 详细设计

### 3.1 Feature flag（用于 Stage 1 实验阶段）

为了独立 measure 每个 A 实验的收益，在 `ironmlx/Cargo.toml` 加 3 个 feature：

```toml
[features]
p5e-stream-parallel = []   # A.1
p5e-compile = []           # A.2
p5e-shape-elim = []        # A.3
```

每个 feature 在 `sparse_moe.rs::forward_on` 内对应一段 `#[cfg(feature = "...")]` 分支，default off。

每个 stage 1 task：
1. 加 feature + 改动
2. with feature 跑 PP=128/512/2048 wall-clock 3 runs
3. without feature 同样跑（baseline）
4. diff 写入 report
5. 如果增益 > 5% → 把 feature 改成默认 enabled（移除 cfg / 直接成为生产代码）；如果 ≤ 5% → feature 保留为 dev artifact 或删除

Stage 1 close-out (T4)：决定 A.1 / A.2 / A.3 哪些进生产，sweep_full + dense 回归无退化即可。

### 3.2 Stage 2 实现细节

Stage 1 ship 后，sparse_moe.rs 已经包含 stage 1 增益。Stage 2 直接改 `forward_on`：

- 把 routing 后到 weighted_sum 前的整段抽出为一个 inner fn `routed_via_sorted(...)`
- `routed_via_sorted` 内做 argsort / scatter / sorted gather_qmm × 3 / SwiGLU / scatter back
- 若 PP < threshold（如 128），fallback 到 stage 1 path（不 sort）— 避免 sort overhead > 节省

实际 threshold 通过 T5 measure 后定（初始猜 PP=64 或 256）。

### 3.3 baseline 重测策略（T0）

P5e T0 已有 profile data 但**是 with eval-barrier**（用于内部 op breakdown，不是产线 perf）。

P5e 起点 baseline 应是**无 instrumentation 的 wall-clock**：
- 在 `tests/p5e_baseline.rs` 加 `#[ignore]` 测试，跑 `Model::forward_on` PP=128/512/2048 三 length × 3 runs；warmup 1 run
- 取每 length 的 median wall-clock
- 写入 `reports/p5e-baseline.md`

每 Stage 1 实验 task（T1-T3）跑同一测试 with feature on，diff vs baseline median 得到该 feature 的增益。

### 3.4 数值精度验证策略

每 stage close-out 跑：

1. `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel`：argmax=11 仍 PASS（ironmlx 自洽 sentinel）
2. `p5_qwen35_moe_batched`：B=2 vs B=1 per-row argmax bit-identical + LOGITS_TOL=1.0 仍 PASS
3. **新增** P5e top-100 logits 自比较：跑 5 prompt 在 P5e 前/后 ironmlx forward，top-100 logits max_abs_diff 应在 bf16 ULP 范围（threshold 1.0，与 P5d T4 一致）
4. `sweep_full.sh` 19/19 PASS（4B snapshot）

P5e 前 / 后的 logits 自比较脚本：`scripts/p5e_logits_self_diff.py`，类似 P5d T4 `p5d_logits_align.py` 但两边都是 ironmlx（一边 stage N-1 build，一边 stage N build）。Stage 1 close-out 时生成对比；Stage 2 同样。

### 3.5 文件结构

```
ironmlx/src/models/qwen3_5_moe/sparse_moe.rs      # 主战场，stage 1 + stage 2 改动
ironmlx/Cargo.toml                                 # 加 3 个 feature（stage 1 临时）
ironmlx/tests/p5e_baseline.rs                      # 新增，wall-clock baseline
scripts/p5e_logits_self_diff.py                    # 新增，stage 前/后 logits 自比较
reports/p5e-baseline.md                            # T0 wall-clock baseline
reports/p5e-stage1-results.md                      # Stage 1 ship 后报告
reports/p5e-stage2-results.md                      # Stage 2 ship 后报告
reports/p5e-final-results.md                       # close-out 综合报告
```

## § 4 测试策略

### 4.1 单元测试

P5e 不引入新 SparseMoeBlock 类型 / 接口；现有 lib unit tests（`qwen3_5_moe::sparse_moe::tests`）继续覆盖。Stage 1 feature 不影响外部行为，无需新 lib test。

Stage 2 改动 forward 内部路径（sort/scatter），但 lib test 无法直接验证（构造 stub 量化权重 + gather_qmm 需要真实 quantized weight 格式，非 trivial）。Stage 2 数值正确性靠现有 integration test（`p5_qwen35_moe_smoke regression sentinel` + `p5_qwen35_moe_batched` + P5e logits self-diff）覆盖；**不强制 T6 close-out 要求新 lib test**。

### 4.2 集成测试

| 测试 | 验证目标 | 状态 |
|---|---|---|
| `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel` | argmax 不漂 | 已有 |
| `p5_qwen35_moe_smoke::p5b_smoke_forward_shape_and_finite` | shape + finite | 已有 |
| `p5_qwen35_moe_batched` | B=2 等价 | 已有 |
| `p5_qwen35_moe_http_smoke` | HTTP 端到端 | 已有 |
| `p5e_baseline` (新增) | wall-clock measurement | 本 phase 加 |

### 4.3 sweep_full regression

每 stage close-out 必跑（per [feedback_regression_sweep_at_closeout]）：

```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<sha>
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh
```

要求 19/19 PASS。

### 4.4 ironmlx 自身 before/after measurement

Stage 1：T0 baseline wall-clock 与 T4 close-out wall-clock 直接对比。
Stage 2：T4 ship wall-clock 与 T6 close-out wall-clock 对比。
P5e final：T0 baseline 与 T6 close-out wall-clock 端到端对比，写入 `reports/p5e-final-results.md`。

不与任何外部实现的数字做"必须超过 X tok/s"门槛比较。

## § 5 风险

| 风险 | 处理 |
|---|---|
| Stage 1 A.1 stream parallelism 在单 GPU 0 增益 | accept 0 增益，看 A.2 / A.3 |
| Stage 1 A.2 `mlx::compile` 不支持 quantized matmul / gather | T2 实验如发现 mlx::compile 报错或不收益，skip 该 sub-task；P5b T0 memory 已提示 mlx compile 在 quant 路径限制 |
| Stage 1 A.3 shape elimination 改变 reduction 顺序 → 数值漂移 | logits self-diff 在 ULP threshold 1.0 内验证；超阈值回滚 |
| Stage 2 B.1 sorted_indices=true 在 mlx 内部 fallback / 不更快 | T5 measure；如不收益直接 skip 整个 stage 2，P5e 收 stage 1 增益即可结束 |
| Stage 2 argsort+scatter overhead 大于 gather_qmm 节省（短 PP） | 加 PP threshold（如 PP < 128 fallback non-sorted） |
| 数值精度退化 | top-100 logits diff > 1.0 → STOP，回滚到上一 stable commit |
| 跑 sweep_full / measurement 卡 GPU（serial constraint） | 按 [feedback_serial_perf_experiments] 一次只跑一个 |

## § 6 实施任务划分（给 writing-plans）

按 [feedback_task_breakdown_bounded] 5-7 task：

| Task | 目标 | 类型 |
|---|---|---|
| **T0** | Wall-clock baseline 测试 + `reports/p5e-baseline.md` | research/test fixture |
| **T1** | Stage 1 A.1 — stream parallelism 实验 + measurement | feature impl + measure |
| **T2** | Stage 1 A.2 — `mlx::compile` 实验 + measurement | feature impl + measure |
| **T3** | Stage 1 A.3 — shape elimination 实验 + measurement | feature impl + measure |
| **T4** | Stage 1 close-out — 选保留组合 + sweep_full + `reports/p5e-stage1-results.md` | close-out |
| **T5** | Stage 2 B.1 — sorted routing 实现 + measurement | feature impl + measure |
| **T6** | Stage 2 close-out + P5e final close-out（sweep_full + before/after final report） | close-out |

7 task，符合上限。

## § 7 验收标准

### P5e 整体验收

- **`Model::forward_on` PP=128/512/2048 wall-clock 与 P5e 起点 baseline 对比有改进**（具体改进 % 由 measure 后揭晓；零或负退化 → P5e 不闭环 / 评估是否需要回滚）
- decode tok/s 不退化（`p5_qwen35_moe_smoke` 端到端 generate 测试 wall-clock 一致 ± 5%）
- 数值精度：`p5b_first_token_argmax_regression_sentinel` argmax=11 PASS；top-100 logits self-diff < 1.0
- `sweep_full.sh` 19/19 PASS
- 现有 4 个 MoE integration test 全 PASS
- clippy 零 warning，fmt clean，release build PASS

### Stage 1 验收（T4）

- 至少一个 A.x feature 进生产（>5% 增益）；或全部 ≤5% 但合并影响 > 5%
- 数值精度不退化（同上）
- sweep_full 19/19 PASS

### Stage 2 验收（T6）

- B.1 sorted routing 增益 > 5%（含 sort overhead 净增益）；或加 PP threshold 后小 PP 不退化、大 PP 增益 > 10%
- 数值精度不退化
- sweep_full 19/19 PASS

### 显式 out-of-scope

任何下列工作出现于 P5e 实施期间立即拒绝并 surface：

- B.2 grouped matmul (per-expert independent matmul) — 留 P5e+1 / P6+
- GatedDeltaNet 优化 — 不在 P5e 范围（虽然 T0 显示 20.6% 也是 hot path）
- 自写 Metal fused gate+up+SwiGLU+down kernel — 留 P5e+ 或 P8 精度阶段
- 改 attention path（Full / Linear）
- 改 KV cache 结构
- 改 expert routing 算法（router top-k 公式 / norm_topk_prob 等）

### P5e → 后续 phase

- 分支 `ironmlx-p5e-perf` ready for merge consideration into `ironmlx-p5-moe`（或直接 `ironmlx`，pending Boss）
- `reports/p5e-final-results.md` 数据交付
- 如 P5e 增益不足以收尾性能问题：进 P5f（B.2 grouped matmul）或 P5g（GatedDeltaNet 优化）或 P6 VL（性能优化暂告段落）
- 如 P5e 充分：进 P6 VL phase
