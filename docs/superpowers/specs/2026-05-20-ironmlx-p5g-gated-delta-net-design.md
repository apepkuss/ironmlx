# ironmlx P5g — GatedDeltaNet Deep Refactor 性能优化设计

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-20 |
| 状态 | Brainstorming approved，准备 writing-plans |
| 范围 | `ironmlx/src/nn/gated_delta_net.rs` 内部 op-level 优化（profile-driven）。聚焦 GatedDeltaNet (linear-attn, 30/40 layers, T0 profile 20% at PP=2048)。 |
| 工作分支 | `ironmlx-p5g-perf`（新开，base HEAD `d74c405` = P5f close-out polish） |
| 上游分支 | 待定（P5g close-out 时 Boss 决定，可能跟 P5f 合并后再 merge 到 `ironmlx-p5-moe`） |
| 硬件 | M5 Max + 128 GB unified |
| 验证模型 | `mlx-community/Qwen3.5-35B-A3B-4bit`（PP=128/512/2048/4096/8192/16384） |
| 验收 | profile-driven 优化项至少 1 个 ship（>5% per-PP improvement）；全 PP 段无 prefill/decode regression；sentinel + batched + sweep_full 全 PASS。 |
| 显式 out-of-scope | GatedAttention 优化（留 P5h）；long-prompt chunk-size sweep（P5h）；router bypass（P5h 条件性）；multi-request batching（P5h/P6+ per Boss 2026-05-19 directive）；Metal kernel rewrite（优先 op-level，profile 不够再 expand）。 |
| 性能目标 | P5g 单 phase 目标 PP=2048 prefill ≥ 2500 tok/s (+35%)，PP=4096-16384 +28-31%；最低 success bar T1-T3 至少 1 个优化 promote (>5%)。P5f+P5g+P5h 联合追"全 PP 段 omlx+10%" ultimate target，P5g 单 phase 不强求达成。 |

## § 1 调研依据与决策摘要

### 1.1 P5f close-out 数据（baseline，HEAD `d74c405`）

来源：`reports/p5f-final-results.md`（commit `8666798` + polish `d74c405`），M5 Max 128 GB，iron-bench HTTP path。

P5f 已 ship T1 (CLI default `b_max=4 → 1`)：

| PP | P5e baseline | P5f shipped (b_max=1) | omlx | omlx+10% target | P5f vs target |
|---:|---:|---:|---:|---:|---|
| 128 | 390 | 953 (2.44×) | 1088 | 1197 | 79% |
| 512 | 491 | 1577 (3.21×) | 2623 | 2886 | 55% |
| 2048 | 1842 | 1844 | 4227 | 4649 | **40%** |
| 4096 | 1773 | 1827 | 4419 | 4861 | **38%** |
| 8192 | 1725 | 1723 | 4261 | 4687 | **37%** |
| 16384 | 1548 | 1598 | 3669 | 4036 | **40%** |

**关键**：PP=2048-16384 prefill 距 omlx+10% target 仍 60-63% gap；P5f T1 (Scheduler padding 消除) 不影响 GS path (PP=2048+ 走 GS B=1)，gap 全部来自 **model forward 自身**。

PP=16384 decode TG 已超 omlx +10.3% (112.18 vs 101.67)，P5g 不动 decode 主路径（除非 profile 显示 GatedDeltaNet 也是 decode 瓶颈）。

### 1.2 T0 profile data（`reports/p5e-t0-profile.md` commit `f47d471`）

| Component | PP=2048 wall-clock | % |
|---|---:|---:|
| 40 × DecoderLayerMoe | 2220.6 ms | 99.9% |
| Linear attn layers (30/40, GatedDeltaNet) | 458.3 ms 累计 | **~20.6%** |
| Full attn layers (10/40, GatedAttention) | 145.3 ms 累计 | 6.5% |
| 3× gather_qmm (P5e 已优化) | already optimized via sorted routing | — |

GatedDeltaNet **per-layer wall-clock = 15.3 ms** at PP=2048, × 30 layers = 459 ms — P5g 主战场。

### 1.3 GatedDeltaNet 算法步骤（from `ironmlx/src/nn/gated_delta_net.rs:1-1026`）

8 个主要 step（per forward call, B=1 single-request 默认场景）：

```
Step 1: in_proj_qkv → 拆 Q/K/V/a/b
Step 2a: prepend conv_state via concatenate(&[conv_state, qkv], 1)
Step 2b: conv1d + sigmoid + mul（element-wise gating）
Step 2c: update conv_state cache（取 last n_keep tokens）
Step 3: split Q/K/V → per-head shape
Step 4: q/k rms_norm + scale (× inv_scale, × inv_scale²)
Step 5: compute_g = exp(-exp(A_log) * softplus(a + dt_bias))
        elementwise chain: zeros / logaddexp / where / astype(f32) / exp /
        negative / mul / exp
Step 6: beta = sigmoid(b)
Step 7a: build/get gated_delta_step Metal kernel (mask/no-mask 2 variants)
Step 7b: state_in from cache OR fresh zeros
Step 7c: t_arr = ((seq,), ()) 0-dim int32 (每次重建)
Step 7d: kernel dispatch
Step 8: ... (rest of file, 1026 lines)
```

ChatGPT review (2026-05-20, `p5f-code-review-for-claude.md` Boss 本地) 给的 4 个候选 hot points：

| ChatGPT 候选 | 源码位置 | 性质 |
|---|---|---|
| `concatenate(&[conv_state, qkv], 1)` | line 412-423 | 每次 forward 全段 concat |
| `conv1d + sigmoid + mul` | line 425-428 | 独立图节点序列 |
| `compute_g` elementwise chain | line 527-539 | 多 op 链（zeros/logaddexp/where/astype/exp/neg/mul/exp） |
| `t_arr` 0-dim Array 重建 | line 570-571 | 每次新建 |

P5g profile-first 决策 (§ 2) 要求 **measurement 验证** ChatGPT static-code 假设是否真瓶颈。

### 1.4 Boss 决策记录（2026-05-20 brainstorming）

- **Scope（Q1）**：聚焦 GatedDeltaNet only；GatedAttention / chunk sweep / router bypass 留 P5h
- **Process（Q2）**：Profile-first — P5g T0 instrument per-op timing 验证 hot points 真实占比；T1-T3 优化项由 profile 数据决定，不预设
- **Branch（Q3）**：新开 `ironmlx-p5g-perf` 从 `d74c405` 分叉

## § 2 Architecture

```
┌────────────────────────────────────────────────────────────┐
│ P5g process flow                                           │
│                                                            │
│ T0: independent profile                                    │
│   ├─ instrument gated_delta_net.rs 8 steps                 │
│   ├─ mlx::eval barrier per-op                              │
│   ├─ PP=2048 主战场 + PP=4096/8192/16384 趋势               │
│   └─ verify ChatGPT 4 hot points 真实占比                   │
│                                                            │
│         ↓ T0 ranked hot points                             │
│                                                            │
│ T1: highest single-ROI optimization                        │
│ T2: 2nd highest                                            │
│ T3: 3rd highest                                            │
│   each Tn:                                                 │
│   ├─ implement specific op change                          │
│   ├─ sentinel + batched + http_smoke                       │
│   ├─ iron-bench validate (PP=2048+ delta)                  │
│   └─ commit OR revert (per >5% per-PP threshold)           │
│                                                            │
│         ↓                                                  │
│                                                            │
│ T4: close-out                                              │
│   ├─ 4-way bench (ironmlx / mlx-lm / omlx)                 │
│   ├─ sweep_full 19/19                                      │
│   ├─ reports/p5g-final-results.md (self-contained)         │
│   └─ quantify P5h scope drivers (residual gap attribution) │
└────────────────────────────────────────────────────────────┘
```

## § 3 T0 — GatedDeltaNet Independent Profile

### 3.1 触点

| 文件 | 修改 |
|---|---|
| `ironmlx/src/nn/gated_delta_net.rs` | instrument `GatedDeltaNet::forward_on` 内部 per-step `mlx::transforms::eval` barrier + `std::time::Instant` timing。通过 feature flag 或环境变量启用（避免 production 永久 overhead）。 |
| `ironmlx/tests/p5g_t0_gated_delta_profile.rs` | 新增 profile harness — 加载 35B-A3B-4bit, 直接调用 `Model::forward_on(PP=2048/4096/8192/16384)`, 收集 8 step timing aggregated across 30 layers, 输出 per-step total / per-call ms + ChatGPT 4 hot points 验证 |
| `reports/p5g-t0-gated-delta-profile.md` | output 报告 |

### 3.2 Instrument 策略

每个 step 间插入 `mlx::transforms::eval(&[&result])` barrier，记录 `Instant::elapsed()`。与 P5e T0 同样的 instrumented profile 方法（per memory `reports/p5e-t0-profile.md`）。

eval barrier 开销 ~0.1-0.5 ms per call，30 layers × 8 step = 240 barriers per forward；PP=2048 forward 内 barrier overhead ~24-120 ms。**绝对值会被 inflated 但 step-by-step 相对占比可靠**（P5e T0 同性质）。

Feature flag 命名: `cargo --features p5g-profile`（避免污染 release build）。

### 3.3 输出内容

`reports/p5g-t0-gated-delta-profile.md` 含：

1. Methodology
2. Per-step wall-clock 累计（30 layers × 8 steps, ms + %）at PP=2048/4096/8192/16384
3. ChatGPT 4 hot points 验证表：每个 hot point 实测占 GatedDeltaNet wall-clock 多少 %
4. Hot point ranking（按降序 top 3-5）
5. P5g T1-T3 候选优化建议（profile-driven，不抄 omlx.patches）

### 3.4 验证

- profile harness compile + 跑通 PP=2048
- per-step 累计 ≤ GatedDeltaNet 总 wall-clock × 1.2（误差可接受范围）
- 至少 1 个 hot point 占 > 5% 才有 T1-T3 价值（否则 P5g 可能 scope 不足，回 Boss 决策）

## § 4 T1-T3 — Profile-Driven Optimizations

### 4.1 候选优化项（pending T0 ranking）

按 ChatGPT review 候选 + 我自己读代码后的补充，预备清单：

| 候选 | 源码 | 优化思路 |
|---|---|---|
| **C1 compute_g 链** | line 527-539 (Step 5) | (a) token-invariant 常量预计算缓存（`-exp(a_log)` cast to f32 + neg 只算一次跨多 forward）; (b) softplus stabilised 链 fuse（logaddexp / where / mul / exp 合并） |
| **C2 concatenate(conv_state, qkv)** | line 412-423 (Step 2a) | (a) stateful causal depthwise conv（避免每次 concat fresh array）; (b) 缩短 concat 路径（只 concat 实际需要的 n_keep tokens 而不是整段 conv_state） |
| **C3 conv1d + silu fusion** | line 425-428 (Step 2b) | 把 conv1d + sigmoid + mul 三 op 合并成 fused conv-with-silu (如果 mlx 有 fused 算子或可写 fast op) |
| **C4 t_arr 重建消除** | line 570-571 (Step 7c) | 把 0-dim int32 `t_arr = ((seq,), ()).try_into()` 预计算并缓存（按 seq 长度组 keyed lookup），或者改 Metal kernel 接受 const argument |

T0 profile 数据决定哪些进 T1-T3。每个 Tn ≥ 5% per-PP improvement 才 promote；< 5% revert。

### 4.2 单 task 实施模板（每个 Tn 同结构）

```
Step Tn.1: 读源码确认 op 序列（per T0 profile 指向）
Step Tn.2: 实施优化（修改 gated_delta_net.rs）
Step Tn.3: cargo build + fmt + clippy 全 clean
Step Tn.4: p5_qwen35_moe_smoke 跑 (argmax=11 sentinel)
Step Tn.5: p5_qwen35_moe_batched 跑 (B=2 row-equiv)
Step Tn.6: iron-bench PP=2048/4096/8192/16384 quick sweep（runs=3 warmup=1）
Step Tn.7: 决策 promote / revert based on > 5% threshold per PP
Step Tn.8: commit（promote 时含实测 numbers；revert 时含负 ROI 数据 + 归因）
```

### 4.3 数值稳定性 sensitivity

GatedDeltaNet Step 5 含 `exp(-exp(A_log) * softplus(a + dt_bias))`，数值敏感。如果优化重排 op 顺序（如把 softplus 拆分），bf16 算 op 中间可能 ULP 漂移交叉 argmax tie。

每个 Tn step 4 必须验证 argmax=11；若漂移按 P5e T5 sorted routing 经验处理：logit margin 大可接受 record；margin 小 revert。

## § 5 Numerical Safety

Regression sentinel suite (沿用 P5e/P5f)：

- `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel`: argmax = 11
- `p5_qwen35_moe_smoke::p5b_smoke_forward_shape_and_finite`: PASS
- `p5_qwen35_moe_batched::p5b_batched_row_equivalence`: B=2 vs B=1 per-row identical
- `p5_qwen35_moe_http_smoke::p5b_http_chat_smoke`: PASS

P5g 新增（若 T0 / T1-T3 涉及 Metal kernel 改动时）：

- `p5g_gated_delta_kernel_numerical_eq`: 改造前后 fixed-prompt forward 数值一致（or argmax 一致）

每个 task 完成时跑 sentinel suite；close-out 跑 sweep_full 19/19。

## § 6 Validation Gates

| Gate | Command | 必须 PASS |
|---|---|---|
| Build | `MLX_DIR=$HOME/.local/mlx cargo build --release` | Finished, no warning |
| fmt | `cargo +nightly fmt --all -- --check` | clean |
| clippy | `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings` | 0 warnings |
| Smoke | `cargo test --release --test p5_qwen35_moe_smoke -- --ignored` | 2/2 PASS, argmax=11 |
| Batched | `cargo test --release --test p5_qwen35_moe_batched -- --ignored` | 1/1 PASS |
| HTTP smoke | `cargo test --release --test p5_qwen35_moe_http_smoke -- --ignored` | 1/1 PASS |
| sweep_full (close-out) | `./scripts/sweep/sweep_full.sh` | 19/19 PASS |
| 4-way bench (close-out) | iron-bench against ironmlx / mlx-lm / omlx, 6 PP × 5 runs | quantify residual gap |

## § 7 Acceptance Criteria

P5g 落地的 measurable success criteria（按 prefill PP tok/s 主代表 metric）：

| PP | Current P5f shipped | P5g target | Stretch (omlx+10%) | P5g 完后 vs stretch |
|---:|---:|---:|---:|---|
| 128 | 953 | persists (≥950, no regression) | 1197 | 79% (P5h 补) |
| 512 | 1577 | persists (≥1570) | 2886 | 55% (P5h 补) |
| 2048 | 1844 | **≥ 2500 (+35%)** | 4649 | 54% (P5h 补) |
| 4096 | 1827 | ≥ 2400 (+31%) | 4861 | 49% (P5h 补) |
| 8192 | 1723 | ≥ 2200 (+28%) | 4687 | 47% (P5h 补) |
| 16384 | 1598 | ≥ 2050 (+28%) | 4036 | 51% (P5h 补) |

**最低 success bar**: T1-T3 至少 1 个 promote (>5%)，全 PP 无 prefill/decode regression。

**P5g close-out 必须**: quantify P5h scope drivers — 报告里明确"剩余 gap 来自 GatedAttention 长 prompt O(S²) / 其他"。

## § 8 P5h preview / Future phases（out of P5g scope）

P5g close-out 输出会驱动 P5h scope。当前已知候选：

1. **GatedAttention 优化** (full attn, 10/40 layers, long-prompt O(S²))
   - 长 prompt 时占比放大 (PP=16384 可能 30%+)
   - SDPA dispatch tuning / KV layout / memory-access pattern
2. **Long-prompt chunk-size sweep** (PP=4096-16384, NOT bypass chunking)
   - 扫 `prefill_chunk_size = 512/1024/1536/2048/3072/4096` 找曲线
3. **Router bypass single-request idle server** — 条件性 (admit overhead > 50ms 测出)
4. **Multi-request batching (P5h/P6+ deferred)** — per Boss 2026-05-19 directive
   - `--b-max N > 1` 已 functional, 等多用户场景启用
5. **Metal kernel rewrite for GatedDeltaNet** — 若 P5g op-level 优化不足，expand 到 kernel level

## § 9 Out of Scope / Non-Goals

- 不动 GatedAttention（留 P5h）
- 不动 SparseMoeBlock（P5e 已优化, sorted routing shipped）
- 不动 Scheduler / KVCache / forward orchestration（P5f 已 ship）
- 不抄 omlx.patches.gated_delta_advance 实现（per [feedback_no_spec_from_competitors]）
- 不引入 PagedCache 化（[feedback_design_philosophy] 不对齐 omlx）
- 不做 long-prompt chunk-size sweep（P5h scope）
- 不做 router bypass（P5h 条件性）
- 不做 multi-request batching feature changes（保留 `--b-max N > 1` 不变）

## § 10 Task decomposition（writing-plans 阶段细化）

P5g 拟拆为 **5 task**（[feedback_task_breakdown_bounded] 5-7 范围内）：

```
T0: GatedDeltaNet 独立 per-op profile
    - instrument gated_delta_net.rs forward_on per-step
    - feature flag (--features p5g-profile)
    - tests/p5g_t0_gated_delta_profile.rs harness
    - 输出 reports/p5g-t0-gated-delta-profile.md
    - 验证 ChatGPT 4 candidate hot points 真实占比 + ranking
    - commit

T1: highest single-ROI optimization (by T0 ranking)
    - implement specific op change in gated_delta_net.rs
    - sentinel + batched + http_smoke
    - iron-bench PP=2048-16384 validate
    - >5% per-PP promote / <5% revert
    - commit (promote 含 numbers / revert 含 negative ROI doc)

T2: 2nd highest ROI optimization
T3: 3rd highest ROI optimization
    (T2/T3 同 T1 结构)

T4: P5g close-out
    - 跑同 reports/p5f-final-results.md style 4-way bench
    - 写 reports/p5g-final-results.md（self-contained for chatgpt 分析）
    - sweep_full 19/19
    - quantify P5h scope drivers
    - commit
```

## § 11 References

- [reports/p5f-final-results.md](../../../reports/p5f-final-results.md) — P5f close-out (baseline for P5g)
- [reports/p5e-t0-profile.md](../../../reports/p5e-t0-profile.md) — GatedDeltaNet 20% T0 占比依据
- [docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md](2026-05-19-ironmlx-p5f-known-path-perf-design.md) — P5f spec（process precedent）
- `p5f-code-review-for-claude.md`（Boss 本地，不入 repo） — ChatGPT review 给的 4 个 hot points 作 T0 verification 清单
- `memory[no-spec-from-competitors]` — 实现独立
- `memory[design-philosophy]` — 不对齐 omlx
- `memory[task-breakdown-bounded]` — 单 plan 5-7 task
- `memory[honest-answers-no-sycophancy]` — brainstorm 客观判断
- `memory[no-unnecessary-docs]` — 不必要文档不提交
