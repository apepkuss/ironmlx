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
| 性能目标 | **Provisional**, 待 T0a 实测当前 HEAD GatedDeltaNet 真实占比后由 § 7 ceiling 推导锁定。当前推导 (假设 occupancy 仍 ~20% + 优化 cut 50%): PP=2048 prefill 1844 → ~2057 tok/s (+11.5%)。最低 success bar: T1-T3 至少 1 个优化 promote (>5% per-PP improvement) 且全 PP 无 prefill/decode regression。 P5f+P5g+P5h+ 联合追"全 PP 段 omlx+10%" ultimate target，P5g 单 phase 不强求达成。 |

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

主要 step（per forward call, B=1 single-request 默认场景）— op 名跟实际代码 align：

```
Step 1a: in_proj_qkvz → slice [q | k | v | z]
Step 1b: in_proj_ba   → slice [b | a]
Step 2a: prepend conv_state via concatenate(&[conv_state, qkv], 1)
         (conv_state already = [B, kernel_size-1, conv_dim] — trimmed window
          not full history; kernel_size=4 → conv_state holds 3 tokens)
Step 2b: conv1d + sigmoid + mul (silu activation gating)
Step 2c: update conv_state cache (取 last n_keep tokens of conv_input)
Step 3:  split q_per_head / k_per_head / v_per_head per-head shape
Step 4:  q/k rms_norm_on + scale (× inv_scale, × inv_scale²)
Step 5:  compute_g = exp(-exp(A_log) * softplus(a + dt_bias))
         elementwise chain: try_into twenty / zeros_like / logaddexp / greater /
         where / astype(f32) / exp / negative / mul / exp
Step 6:  beta = sigmoid_on(b)
Step 7a: build/get gated_delta_step Metal kernel (mask/no-mask 2 variants
         via OnceLock)
Step 7b: state_in from cache.recurrent_state().clone() OR fresh f32 zeros
Step 7c: t_arr = ((seq,), ()).try_into()  0-dim int32 (每次重建)
Step 7d: kernel dispatch
Step 8+: y output + final rms_norm_gated + out_proj (rest of forward_on)
```

External review (2026-05-20, internal review notes; not committed to repo per
[no-unnecessary-docs]) 给出 4 个候选 hot points 作 T0 verification 清单：

| 候选 | 源码位置 | 性质 |
|---|---|---|
| `concatenate(&[conv_state, qkv], 1)` | line 412-423 (Step 2a) | 每次 forward 全段 concat (conv_state 已是 trimmed 3-token window, 不是 full history) |
| `conv1d + sigmoid + mul` | line 425-428 (Step 2b) | 独立图节点序列 (silu gating) |
| `compute_g` elementwise chain | line 527-539 (Step 5) | 多 op 链（try_into / zeros_like / logaddexp / where / astype / exp / neg / mul / exp） |
| `t_arr` 0-dim Array 重建 | line 570-571 (Step 7c) | 每次新建，可能可缓存 |

P5g profile-first 决策 (§ 2) 要求 **measurement 验证** static-code 假设是否真瓶颈 (P5e T1 stream-parallel + P5f T2 single-shot 两次实测推翻 spec assumption 的教训)。

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
| `ironmlx/src/nn/gated_delta_net.rs` | instrument `GatedDeltaNet::forward_on` 内部 per-step `mlx::transforms::eval` barrier + `std::time::Instant` timing。**Double-gated**: 编译期 feature flag (`p5g-profile`) 只编译 hook 代码；运行期 env var (`IRONMLX_P5G_PROFILE=1`) 才实际启用 barrier。两个 gate 都需要才 profile，否则正常 forward 路径完全无 overhead。这避免 `cargo --all-features` 在 clippy / sweep_full / bench 路径误启 profile mode。 |
| `ironmlx/tests/p5g_t0_gated_delta_profile.rs` | 新增 profile harness — 必须 `#[test] #[ignore]` (heavy test, load 35B model)，必须读 `IRONMLX_MOE_MODEL_DIR` env var 取 model path，必须 `--test-threads=1` (Metal GPU 串行)。普通 `cargo test` / CI / clippy 不应触发模型加载。 |
| `reports/p5g-t0-gated-delta-profile.md` | output 报告 |

### 3.2 Instrument 策略 — 3 层 profile protocol

按外部 review 建议，单纯 per-op eval barrier 会改变 MLX lazy execution 的 op fusion 边界 / 中间 tensor materialization / 调度顺序 / 临时内存生命周期 / kernel launch pattern。per-step 相对占比可能被截断到 fusion-free 量级，不能直接当作优化优先级唯一依据。

3 层 protocol：

**Layer 1 — Baseline mode (no per-op barrier)**
- 仅在 `GatedDeltaNet::forward_on` 入口 + 出口加 eval barrier
- 测当前 HEAD 下 **per-layer GatedDeltaNet 总 wall-clock** + 30-layer 累计 + GatedDeltaNet 占 prefill 总 wall-clock 的 % at PP=2048/4096/8192/16384
- 这一层数字是 P5g target ceiling 计算的依据 (避免重复 P5e T0 旧数字, P5f 已 ship 后 GatedDeltaNet 占比可能已变化)

**Layer 2 — Breakdown mode (per-step barrier)**
- 在 Step 1a / 1b / 2a / 2b / 2c / 3 / 4 / 5 / 6 / 7a-d / 8 各处加 eval barrier
- 测每 step 相对占比 — **作热点定位参考，不是直接优化优先级依据**
- 报告里 explicit acknowledge "barrier 改变 fusion 边界，相对占比 indicative only"

**Layer 3 — Verification mode (ablation / microbench)**
- 对 Layer 2 top-3 ranked 候选热点，单独 microbench 验证：手动 disable / no-op 该 step 看 end-to-end 是否真减 wall time
- 这是 T1-T3 promote 决策的最终依据

报告内容必须同时含：

- non-instrumented GatedDeltaNet 总耗时 (Layer 1)
- instrumented GatedDeltaNet 总耗时 (Layer 2, 含 barrier overhead)
- per-step breakdown (Layer 2)
- instrumented / non-instrumented slowdown ratio
- top-3 候选 ablation 结果 (Layer 3)

### 3.3 输出内容

`reports/p5g-t0-gated-delta-profile.md` 含：

1. Methodology (3-layer protocol)
2. Layer 1: per-layer + 30-layer aggregate GatedDeltaNet wall-clock + % at PP=2048/4096/8192/16384
3. Layer 2: per-step breakdown table (ms + %) + instrumented/non-instrumented slowdown ratio
4. Layer 3: top-3 hot points ablation microbench
5. **更新 P5g performance ceiling 推导** (基于 Layer 1 真实 GatedDeltaNet occupancy)
6. P5g T1-T3 候选优化建议 ranking (profile-driven，不抄 omlx.patches)

### 3.4 验证

- profile harness compile + 跑通 PP=2048
- Layer 1 (entry+exit barrier only) GatedDeltaNet 总耗时 - 非 instrumented PP=2048 总耗时 占比 ≥ 10% 才有 T1-T3 价值 (P5f baseline 1844 tok/s → 1.111s, 10% = 111 ms)
- Layer 2 per-step 累计 ≤ Layer 2 GatedDeltaNet 总 wall-clock × 1.2 (sanity)
- Layer 3 top-3 ablation 中至少 1 个 cut > 5% Layer 1 baseline 才推进 T1-T3 (否则 P5g 可能 scope 不足，回 Boss 决策)

### 3.5 Feature flag 完整启用命令

启动 profile harness:

```bash
IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha>/ \
IRONMLX_P5G_PROFILE=1 \
MLX_DIR=$HOME/.local/mlx \
cargo test -p ironmlx --release --features p5g-profile \
  --test p5g_t0_gated_delta_profile \
  -- --ignored --test-threads=1 --nocapture
```

正常 release build / clippy / sweep_full / bench 不带 `IRONMLX_P5G_PROFILE` env var，即使 `--features p5g-profile` 编译进 hook 也走 normal forward 路径，0 overhead。

## § 4 T1-T3 — Profile-Driven Optimizations

### 4.1 候选优化项（pending T0 ranking）

外部 review 4 个候选 + 我读代码后补充。注意 conv_state 在 ironmlx 已经是 `[B, kernel_size-1, conv_dim]` trimmed window (kernel=4 → 3 tokens, 不是 full history)，所以 C2 优化思路只针对 stateful conv path 重设计，**不是**缩短 concat 长度。

| 候选 | 源码 | 优化思路 |
|---|---|---|
| **C1 compute_g 链** | line 527-539 (Step 5) | (a) token-invariant 常量预计算缓存（`-exp(a_log)` cast to f32 + neg 只算一次跨多 forward）; (b) softplus stabilised 链 fuse（logaddexp / where / mul / exp 合并）；(c) 可能 `mlx::fast` 已有 softplus / silu fused op 可直接复用 |
| **C2 stateful causal depthwise conv** | line 412-423 (Step 2a) + `core/cache/gated_delta.rs:48` | 避免 `concatenate(conv_state, qkv)` 本身。设计 stateful causal depthwise conv path: conv kernel 在 step-stream 上滚动，conv_state 不作显式 concat 输入。需评估 Metal/kernel-level fused conv 可行性。**不**做"只 concat n_keep tokens" — conv_state 已是 trimmed 3-token window |
| **C3 conv1d + silu fusion** | line 425-428 (Step 2b) | 把 conv1d + sigmoid + mul 三 op 合并成 fused conv-with-silu (检查 `mlx::fast` 或可写 fast op) |
| **C4 t_arr 重建消除** | line 570-571 (Step 7c) | 把 0-dim int32 `t_arr = ((seq,), ()).try_into()` 预计算并缓存（按 seq 长度组 keyed lookup），或者改 Metal kernel 接受 const argument |

T0 profile (Layer 3 ablation) 数据决定哪些进 T1-T3。每个 Tn ≥ 5% per-PP improvement (Layer 1 baseline) 才 promote；< 5% revert。

### 4.2 单 task 实施模板（每个 Tn 同结构）

```
Step Tn.1: 读源码确认 op 序列（per T0 profile 指向）
Step Tn.2: 实施优化（修改 gated_delta_net.rs）
Step Tn.3: cargo build + fmt + clippy 全 clean
Step Tn.4: p5_qwen35_moe_smoke 跑 (argmax=11 sentinel)
Step Tn.5: p5_qwen35_moe_batched 跑 (B=2 row-equiv)
Step Tn.6: PP=128 + PP=512 quick smoke (iron-bench runs=3 warmup=1, 验证短 prompt
           prefill 无 regression — 防止仅看长 prompt 优化漏检 short-PP 退化)
Step Tn.7: PP=2048/4096/8192/16384 quick sweep (runs=3 warmup=1)
Step Tn.8: Decode TG smoke (PP=128/2048 max-tokens=32, 验证 decode TG 不 regression)
Step Tn.9: 决策 promote / revert based on > 5% threshold per PP for long prompt
           AND no regression on PP=128/512 AND no decode TG regression
Step Tn.10: commit (promote 时含实测 numbers; revert 时含负 ROI + 归因)
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

### 7.1 Performance target ceiling 推导

P5g 只优化 GatedDeltaNet — 物理上的 end-to-end wall-time 减少上限由 GatedDeltaNet 在 prefill 总 wall-time 的占比 × GatedDeltaNet 内部优化的 cut 比例共同决定。

以 PP=2048 为例 (P5f baseline 1844 tok/s = 1.111s):

| GatedDeltaNet 占比 (待 T0a 实测) | GatedDeltaNet 内部 cut 30% | cut 50% | cut 70% |
|---:|---:|---:|---:|
| 假设 20% (P5e T0 旧数据外推) | 1965 tok/s (+6.6%) | 2057 tok/s (+11.5%) | 2155 tok/s (+16.9%) |
| 假设 25% (cut 比 ~P5f 前略大) | 1996 tok/s (+8.2%) | 2104 tok/s (+14.1%) | 2220 tok/s (+20.4%) |
| 假设 30% | 2027 tok/s (+9.9%) | 2154 tok/s (+16.8%) | 2293 tok/s (+24.3%) |

P5e T0 profile 报 GatedDeltaNet ~20% at PP=2048 (P5e baseline 921 tok/s 时), P5f 已 ship 但只动 Scheduler padding (不影响 GS 路径)，所以 PP=2048 GS 路径 GatedDeltaNet 占比应仍接近 20%。**最 realistic P5g target ≈ +10-15% PP=2048 (1844 → ~2000-2100 tok/s)**, 不是原 +35%。

### 7.2 Provisional target (待 T0a 锁定)

| PP | Current P5f shipped | P5g provisional target | Stretch (omlx+10%) | P5g 完后 vs stretch |
|---:|---:|---:|---:|---|
| 128 | 953 | persists (≥950, no regression) | 1197 | 79% (留 P5h) |
| 512 | 1577 | persists (≥1570, no regression) | 2886 | 55% (留 P5h) |
| 2048 | 1844 | **≥ 2050 (+11%)** | 4649 | 44% (留 P5h) |
| 4096 | 1827 | ≥ 2030 (+11%) | 4861 | 42% (留 P5h) |
| 8192 | 1723 | ≥ 1910 (+11%) | 4687 | 41% (留 P5h) |
| 16384 | 1598 | ≥ 1770 (+11%) | 4036 | 44% (留 P5h) |

**T0a 完成后必须做的事**: 用 Layer 1 实测的 GatedDeltaNet 真实占比 + T0c ablation 估的可达 cut 比例，回写本表锁定 final target。如果 GatedDeltaNet 占比 < 15% (P5e 后续优化使其下降)，则需 Boss 决策是否调整 P5g scope 或合并到 P5h。

### 7.3 最低 success bar

- T1-T3 至少 1 个 optimization promote (>5% per-PP improvement on long-prompt PP=2048-16384)
- 全 PP 段 (含 PP=128/512) 无 prefill regression > 2%
- Decode TG 无 regression > 2% (PP=16384 decode +10.3% over omlx 必须保持)
- sentinel + batched + http_smoke + sweep_full 全 PASS

### 7.4 P5g close-out 必须

quantify P5h scope drivers — 报告里明确 "剩余 gap 来自 GatedAttention 长 prompt O(S²) / chunk-size 调优空间 / Scheduler admission overhead 残余 / 其他"。

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
T0: GatedDeltaNet 独立 per-op profile (3-layer protocol per § 3.2)
    Step T0.a: Layer 1 baseline mode — entry+exit barrier only;
               测当前 HEAD GatedDeltaNet 总耗时 + 占 prefill %
               at PP=2048/4096/8192/16384
    Step T0.b: Layer 2 breakdown mode — per-step barrier;
               输出 per-step ms + %; 报告 slowdown ratio
    Step T0.c: Layer 3 verification mode — top-3 候选 ablation
               microbench; 估每个候选可达 cut 比例
    Step T0.d: 用 T0.a + T0.c 数据回写 § 7.2 锁定 final target
    Files: ironmlx/src/nn/gated_delta_net.rs (instrument hook, double-gated
           by p5g-profile feature + IRONMLX_P5G_PROFILE=1 env var)
           ironmlx/tests/p5g_t0_gated_delta_profile.rs (#[ignore], heavy)
           reports/p5g-t0-gated-delta-profile.md
    Commit

T1: highest single-ROI optimization (by T0.c ranking)
    Steps per § 4.2 implementation template (1-10):
      build / fmt / clippy / smoke / batched / short-PP smoke /
      long-PP quick sweep / decode TG smoke / promote-revert decision /
      commit
    Promote 阈值: >5% on long-prompt PP (2048-16384) AND
                  no >2% regression on PP=128/512 AND
                  no >2% decode TG regression
    < threshold revert (per P5e/P5f precedent: 失败实验也 commit
    negative ROI doc 进 close-out 报告作记录)

T2: 2nd highest ROI optimization (same structure as T1)
T3: 3rd highest ROI optimization (same structure as T1)

T4: P5g close-out
    - 跑同 reports/p5f-final-results.md style 4-way bench
    - 写 reports/p5g-final-results.md (self-contained for offline analysis)
    - sweep_full 19/19 (Qwen3.5-4B-MLX-4bit)
    - quantify P5h scope drivers
    - commit
```

## § 11 References

- [reports/p5f-final-results.md](../../../reports/p5f-final-results.md) — P5f close-out (baseline for P5g)
- [reports/p5e-t0-profile.md](../../../reports/p5e-t0-profile.md) — GatedDeltaNet 20% T0 占比依据
- [docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md](2026-05-19-ironmlx-p5f-known-path-perf-design.md) — P5f spec（process precedent）
- External review notes (not committed to repo per [feedback_no_unnecessary_docs]; substance internalized to § 1.3 candidates + § 4.1 优化思路)
- `memory[no-spec-from-competitors]` — 实现独立
- `memory[design-philosophy]` — 不对齐 omlx
- `memory[task-breakdown-bounded]` — 单 plan 5-7 task
- `memory[honest-answers-no-sycophancy]` — brainstorm 客观判断
- `memory[no-unnecessary-docs]` — 不必要文档不提交
