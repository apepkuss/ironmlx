# ironmlx P5g — GatedDeltaNet Deep Refactor 性能优化设计

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-20 |
| 状态 | Brainstorming approved，准备 writing-plans |
| 范围 | `ironmlx/src/nn/gated_delta_net.rs` 内部 op-level 优化（profile-driven）。聚焦 GatedDeltaNet (linear-attn, 30/40 layers; historical P5e T0 profile showed ~20% at PP=2048 with direct-call instrumented baseline — current HEAD occupancy must be re-measured by T0.a)。 |
| 工作分支 | `ironmlx-p5g-perf`（新开，base HEAD `d74c405` = P5f close-out polish） |
| 上游分支 | 待定（P5g close-out 时 Boss 决定，可能跟 P5f 合并后再 merge 到 `ironmlx-p5-moe`） |
| 硬件 | M5 Max + 128 GB unified |
| 验证模型 | `mlx-community/Qwen3.5-35B-A3B-4bit`（PP=128/512/2048/4096/8192/16384） |
| 验收 | profile-driven 优化项至少 1 个 ship 满足 § 7.3 端到端 ship 指标（**geometric mean prefill > 5% on PP=2048/4096/8192/16384 AND 各档单点 regression < 2%**）；全 PP 段无 prefill/decode regression > 2%；sentinel + batched + http_smoke + sweep_full 全 PASS。 |

## § 1 调研依据与决策摘要

### 1.1 P5f close-out 数据（baseline，HEAD `d74c405`）

来源：`reports/p5f-final-results.md` § 2 (commit `8666798` + polish `d74c405`)，M5 Max 128 GB，iron-bench HTTP path。**数字与 `reports/p5f-final-results.md` 严格同步**（P5g 是 P5f close-out 之上推进，必须用 P5f close-out 重测的 omlx 数字而非 P5e three-way bench 旧数字）。

P5f 已 ship T1 (CLI default `b_max=4 → 1`)：

| PP | P5e baseline | P5f shipped (b_max=1) | omlx | omlx+10% target | P5f vs target |
|---:|---:|---:|---:|---:|---|
| 128 | 390 | 953 (2.44×) | 1078 | 1186 | −19.6% |
| 512 | 491 | 1577 (3.21×) | 2635 | 2898 | −45.6% |
| 2048 | 1842 | 1844 | 4230 | 4653 | **−60.4%** |
| 4096 | 1773 | 1827 | 4413 | 4855 | **−62.4%** |
| 8192 | 1725 | 1723 | 4347 | 4782 | **−64.0%** |
| 16384 | 1548 | 1598 | 3865 | 4252 | **−62.4%** |

**关键**：PP=2048-16384 prefill 距 omlx+10% target 仍 60-64% gap；P5f T1 (Scheduler padding 消除) 不影响 GS path (PP=2048+ 走 GS B=1)，gap 全部来自 **model forward 自身**。

PP=16384 decode TG 已超 omlx +10.3% (`reports/p5f-final-results.md` § 3 报 112.18 vs 101.67)，P5g 不动 decode 主路径（除非 profile 显示 GatedDeltaNet 也是 decode 瓶颈）。

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

- **Process（Q2）**：Profile-first — P5g T0 instrument per-op timing 验证 hot points 估计占比；T1-T3 优化项由 profile 数据决定，不预设
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
│   └─ verify ChatGPT 4 hot points 估计占比                   │
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
│   └─ commit OR revert (per § 7.3 ship metric: geomean >5%   │
│                          + per-PP regression < 2%)          │
│                                                            │
│         ↓                                                  │
│                                                            │
│ T4: close-out                                              │
│   ├─ 4-way bench (ironmlx / mlx-lm / omlx)                 │
│   ├─ sweep_full 19/19                                      │
│   ├─ reports/p5g-final-results.md (self-contained)         │
└────────────────────────────────────────────────────────────┘
```

## § 3 T0 — GatedDeltaNet Independent Profile

### 3.1 触点

| 文件 | 修改 |
|---|---|
| `ironmlx/Cargo.toml` | `[features]` 表加一行 `p5g-profile = []` (空 feature flag, 仅作 cfg gate, 不引入额外依赖)。 |
| `ironmlx/src/nn/gated_delta_net.rs` | (a) `GatedDeltaNet::from_loader(prefix, ...)` **不改 public 签名** — 在函数体内从 `prefix` 解析 layer index (e.g., `prefix = "model.layers.7.linear_attn"` → 用 `.split('.').nth(2)` + `parse::<i32>()` 提取 `7`；解析失败 → `None`)，存入 struct 字段。**避免** cascade 改动到 `nn/decoder_layer.rs` (common path, 被 dense `qwen3_5/text_model.rs` 和 `nn/mtp.rs` 也调用) — 改 public 签名会涉 6 个文件且污染 dense/MTP profile-OOS 路径。(b) struct 增加 `profile_layer_idx: Option<i32>` 字段 (仅 `#[cfg(feature = "p5g-profile")]` 编入)，**`from_loader` AND `from_components` 都必须在 `#[cfg(...)]` 下初始化此字段** (后者设 `None`)，否则 `cargo +nightly clippy --all-features` 会编译失败。(c) instrument `GatedDeltaNet::forward_on` 内部 per-step `mlx::transforms::eval` barrier + `std::time::Instant` timing。**timing 数据通过 `tracing::info!` 在 `timer.stop()` + elapsed 计算完成之后** emit，**严禁 tracing::info!() 出现在 measured timer window 内** (string formatting + subscriber filter + stderr write 会污染测量)。Layer 2 per-step 数据缓存到 forward 末尾批量 emit 单行。**Double-gated**: 编译期 feature flag (`p5g-profile`) 只编译 hook 代码；运行期由 `IRONMLX_P5G_PROFILE_MODE` env var 控制 (见下)，通过 `OnceLock<ProfileMode>` 在 GatedDeltaNet 模块初始化时 cache (避免每次 forward 都 env lookup)。profile-disabled 时无 eval barrier、无 buffer alloc、无 env syscall — 允许一次性 cached flag 检查的不可测算分支成本。 |

**Design pivot note (sixth → seventh round)**: 第六轮 review 推荐 typed layer_idx propagation (text_model → decoder_layer → gated_delta_net)，但第七轮 review 发现 `GatedDeltaNet::from_loader` 还被 common `ironmlx/src/nn/decoder_layer.rs` 调用 (服务 dense Qwen3.5 + MTP)，改 public 签名会 cascade 到 6 个文件并污染 dense/MTP profile-out-of-scope 路径。改用 prefix parse — 只改 gated_delta_net.rs 1 个文件, 内部从 loader prefix 字符串恢复 layer index, profile-only 字段。Trade-off: 依赖 prefix 命名约定 (`model.layers.{N}.linear_attn` 在所有调用点已统一)。
| `ironmlx/tests/p5g_t0_gated_delta_profile.rs` | 新增 profile harness — **HTTP-path profile**: 启动 ironmlx server (with `--features p5g-profile` + 适当 `IRONMLX_P5G_PROFILE_MODE`)，跑 iron-bench HTTP request 触发 forward，server log 由 harness aggregate parse 出 per-layer GatedDeltaNet timing。**与 § 1.1 P5f HTTP baseline (1844 tok/s) 同路径**，profile occupancy 数字可直接对比。必须 `#[test] #[ignore]`，读 `IRONMLX_MOE_MODEL_DIR` env var 取 model path，必须 `--test-threads=1`。普通 `cargo test` / CI / clippy 不触发。**不**用 direct `Model::forward_on` harness 走旁路 (避免跟 HTTP baseline 路径不可比)。 |
| `reports/p5g-t0-gated-delta-profile.md` | output 报告 (parsed log aggregation + 3-layer tables) |

### 3.2 Instrument 策略 — 3 层 profile protocol

按外部 review 建议，单纯 per-op eval barrier 会改变 MLX lazy execution 的 op fusion 边界 / 中间 tensor materialization / 调度顺序 / 临时内存生命周期 / kernel launch pattern。per-step 相对占比可能被截断到 fusion-free 量级，不能直接当作优化优先级唯一依据。

3 层 protocol：

**Layer 1 — Baseline mode (entry/exit barrier only)**

Timing 协议 (timer 边界精确, materialization set 完整覆盖 GatedDeltaNet 所有副作用):

```text
1. eval(GDN_input_set: hidden_input, cache.conv_state, cache.recurrent_state, mask?)
   // drain prior lazy ops + 强制 cache state 物质化, 避免 GDN forward 把
   // 上游某些 lazy compute 错误归到自己头上
2. timer.start()           // ←—— timer 启动在 entry drain 完成后
3. GatedDeltaNet forward   // 包含 8 step + Metal kernel dispatch + cache 更新
4. eval(GDN_output_set: final_output, updated cache.conv_state, updated cache.recurrent_state)
   // materialize **所有** GDN produced lazy ops, 包括 Step 2c new_conv_state
   // slice/take + Step 7e recurrent_state 更新 (若不显式 eval cache 输出,
   // 它们的 lazy graph 可能未触发, 导致 Layer 1 低估 GDN 真实 forward cost,
   // 同时让 Layer 2 Step 2c 跟 Layer 1 总耗不可比)
5. timer.stop()            // ←—— timer 停止在所有 GDN 副作用 materialized 后
```

如果某些 cache array 跟 output 来自同一 kernel dispatch，eval set 仍显式列出 — 保证 profile 语义清晰。

- 测当前 HEAD 下 **per-layer GatedDeltaNet 总 wall-clock** + 30-layer 累计 + GatedDeltaNet 占 prefill 总 wall-clock 的 % at PP=2048/4096/8192/16384
- 这一层数字是 P5g target ceiling 计算的依据 (基于当前 HEAD boundary-isolated occupancy estimate, 不外推 P5e 旧数字)
- Warmup + 测量协议：每个 PP **1 次 warmup run (丢弃) + 至少 3 次 measured run, 报 median**。warmup 覆盖 `OnceLock` kernel build / Metal first dispatch JIT / lazy graph init / buffer allocator init

**Layer 2 — Breakdown mode (per-step barrier)**

- 在 Step 1a / 1b / 2a / 2b / 2c / 3 / 4 / 5 / 6 / 7a-d / 8 各处加 eval barrier
- 测每 step 相对占比 — **作热点定位参考，不是直接优化优先级依据**
- 报告里 explicit acknowledge "barrier 改变 fusion 边界，相对占比 indicative only"
- Warmup + median 协议同 Layer 1

**Layer 3 — Shape-preserving cost ablation**

不是简单 disable/no-op (会破坏 cache state / downstream consumption / lazy graph，且 MLX 可能 deferred 删除后续计算，假 ROI)。

正确做法 — **shape-preserving substitute**:

- 对 Layer 2 top-3 ranked 候选热点，单独 microbench 验证
- 用 **cheap shape-preserving proxy op 替换重 op** (e.g., 复杂 elementwise chain 替成单 `zeros_like(output)` cast / 重 reshape 替成 `identity` style no-effect op)
- 保持 output shape / dtype / downstream consumption 不变
- 测得的 wall-time 减少 = **该 step 在端到端中可减的上限**
- **upper bound 语义 only — 不作为 T1-T3 promote 决策依据**
- Promote 决策见 § 7.3 ship 指标 (端到端 benchmark)

**重要命名澄清**: "Layer 1" 本身就是 instrumentation (entry/exit barrier), 不是真正 non-instrumented。严格说无法在完全无 instrumentation 下测单独 GatedDeltaNet wall time — 后者只能测整层 prefill 总 wall time。报告里区分三个量:

- `whole-prefill baseline` — 整层 forward (无 GatedDeltaNet probe) 的完整 prefill wall time
- `Layer 1 boundary-isolated GatedDeltaNet time` — entry drain + output eval 包夹的 GatedDeltaNet 边界计时 (estimate, 含轻 instrumentation overhead)
- `Layer 2 per-step barrier time` — per-step breakdown (含较重 barrier overhead, 相对占比 indicative only)

报告内容必须同时含：

- `whole-prefill baseline` wall-time (无 GatedDeltaNet probe)
- `Layer 1 boundary-isolated` GatedDeltaNet 总耗时
- `Layer 2 per-step barrier` GatedDeltaNet 总耗时
- Layer 2 per-step breakdown
- Layer 2 / Layer 1 slowdown ratio
- Layer 3 top-3 候选 shape-preserving cost ablation results (标记 **upper bound**)
- 每个 PP 1 warmup + 3 measured runs median
- 显式 annotate: "Layer 1 是 boundary-isolated estimate (含 entry/exit eval barrier 引入的轻 instrumentation overhead), 不等于完全无 instrumentation 下 GatedDeltaNet 的 ground-truth 占比 — 后者在 lazy MLX 下不可直接测；端到端 ROI 仍以 T1-T3 实现后 iron-bench benchmark 为准"

### 3.3 输出内容

`reports/p5g-t0-gated-delta-profile.md` 含：

1. Methodology (3-layer protocol, **HTTP-path profile via instrumented server + iron-bench**)
2. Layer 1: per-layer + 30-layer aggregate **boundary-isolated GatedDeltaNet wall-clock estimate** + 占 prefill wall-clock 的 % at PP=2048/4096/8192/16384
3. Layer 2: per-step breakdown table (ms + %) + **Layer 2 / Layer 1 slowdown ratio**
4. Layer 3: top-3 hot points shape-preserving cost ablation upper bound
5. **更新 P5g performance ceiling 推导** (基于 Layer 1 **boundary-isolated GatedDeltaNet occupancy estimate**, 标注 estimate 而非 true occupancy)
6. P5g T1-T3 候选优化建议 ranking (profile-driven，不抄 omlx.patches)
7. Annotation: "Layer 1 是 boundary-isolated estimate (含 entry/exit eval barrier 引入的轻 instrumentation overhead)，不等于完全无 instrumentation 下 GatedDeltaNet 的 ground-truth 占比 — 后者在 lazy MLX 下不可直接测；端到端 ROI 仍以 T1-T3 实施后 iron-bench benchmark 为准。"

### 3.4 验证

- profile harness compile + 跑通 PP=2048
- Layer 1 boundary-isolated GatedDeltaNet estimate / whole-prefill baseline (无 GDN probe) 占比 ≥ 10% 才有 T1-T3 价值 (P5f baseline 1844 tok/s → 1.111s, 10% = 111 ms)
- Layer 2 per-step 累计 ≤ Layer 2 GatedDeltaNet 总 wall-clock × 1.2 (sanity)
- Layer 3 shape-preserving ablation 中至少 1 个 upper-bound cut > 5% Layer 1 baseline 才推进 T1-T3 (否则 P5g 可能 scope 不足，回 Boss 决策)。**注**：upper bound 仅作 T0/T1 优化值得性诊断，不作 ship 依据。

### 3.5 Profile mode selector + harness multi-phase 流程

**Profile mode env var**: `IRONMLX_P5G_PROFILE_MODE` 取以下值之一:

| Mode | Behavior |
|---|---|
| (unset, or `off`) | profile disabled — no barrier, no log, normal forward path |
| `layer1` | Layer 1 only — entry/exit eval set (incl. cache states) + boundary timer |
| `layer2` | Layer 2 — per-step barrier + per-step timer (Layer 1 boundary also captured) |
| `ablate-compute-g` | Layer 3 — replace Step 5 compute_g with cheap shape-preserving substitute |
| `ablate-conv` | Layer 3 — replace Step 2a-2c stateful conv path with substitute |
| `ablate-t-arr` | Layer 3 — bypass Step 7c t_arr reconstruction with cached const |
| (其他 ablate-* 候选按 T0.b ranking 加) | Layer 3 — corresponding shape-preserving substitute |

Server-side log line schema (single line per GatedDeltaNet forward call):

```text
[p5g-profile] mode=<mode> layer=<i32> batch=<i32> seq=<i32> offset_before=<i32> offset_after=<i32> elapsed_us=<u64> [step_breakdown=...]
```

- `mode`: 当前生效的 `IRONMLX_P5G_PROFILE_MODE`
- `layer`: GatedDeltaNet struct 内 `profile_layer_idx` (`from_loader` 内部从 prefix 字符串解析得到, `Some(i)` 时 log 输出 `layer=<i>`, `None` (prefix 解析失败 或 走 `from_components` 单元测试路径) 时 log 输出 `layer=-1`, harness 把 `-1` 单独报告为 unknown 而不参与 30-layer aggregation)
- `batch / seq`: GDN forward 当前 input shape `[B, S, H]` 的 B/S
- `offset_before / offset_after`: cache offset (累积 KV/recurrent state position)
- `elapsed_us`: Layer 1 boundary timer elapsed microseconds
- `step_breakdown` (仅 mode=layer2): per-step elapsed_us list

**注**: `seq` 在 chunked prefill 下是 chunk length 不是 full prompt PP；PP/run id 由 harness 在请求级别绑定 (按 iron-bench 串行请求边界标定，不在 GDN 内部猜)。

**Harness multi-phase execution**:

```text
Phase A — whole-prefill baseline (no profile):
  spawn `ironmlx serve` WITHOUT IRONMLX_P5G_PROFILE_MODE (即 mode=off)
  run iron-bench PP=2048/4096/8192/16384, 1 warmup + 3 measured
  record whole-prefill wall-time medians
  shutdown server

Phase B — Layer 1 boundary-isolated:
  spawn `ironmlx serve` with IRONMLX_P5G_PROFILE_MODE=layer1
  run iron-bench same sweep
  parse server log for `[p5g-profile] mode=layer1 ...`
  aggregate 30-layer per-PP GDN elapsed
  shutdown server

Phase C — Layer 2 per-step breakdown:
  spawn `ironmlx serve` with IRONMLX_P5G_PROFILE_MODE=layer2
  run iron-bench same sweep
  parse server log `mode=layer2`
  aggregate per-step breakdown
  compute Layer 2 / Layer 1 slowdown ratio
  shutdown server

Phase D — Layer 3 ablation (per top-3 candidate from B/C):
  for each candidate in {compute-g, conv, t-arr, ...}:
    spawn `ironmlx serve` with IRONMLX_P5G_PROFILE_MODE=ablate-<candidate>
    run iron-bench same sweep
    record whole-prefill wall-time delta vs Phase A
    shutdown server
```

**Subprocess binary 来源** (避免误用 PATH 上旧 binary):

- **server**: harness 用 `env!("CARGO_BIN_EXE_ironmlx")` 拿当前 `cargo test -p ironmlx --features p5g-profile` 构建的 binary path 启动；不走 PATH。
- **iron-bench**: 跨 package — `CARGO_BIN_EXE_<name>` 只在 binary 跟 test 同 package 时注入；`iron-bench` 是 workspace 另一个 package, `CARGO_BIN_EXE_iron-bench` 在 ironmlx 的 integration test 中**未定义**, 直接 `env!()` 会编译失败。改用 `std::process::Command::new("cargo")` 子进程: `cargo run -p iron-bench --release --` (cargo 会处理 build + 缓存)。或者 harness 启动时先 `Command::new("cargo").args(["build", "-p", "iron-bench", "--release"]).status()`，然后通过相对 `target/release/iron-bench` 直接启动 subprocess。两种都不依赖 PATH。

Harness top-level 调用命令:

```bash
IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha>/ \
MLX_DIR=$HOME/.local/mlx \
cargo test -p ironmlx --release --features p5g-profile \
  --test p5g_t0_gated_delta_profile \
  -- --ignored --test-threads=1 --nocapture
```

Harness 内部按 Phase A/B/C/D 串行 spawn subprocess，每个 phase 独立 setup/teardown server，profile mode 通过 child env 传入。`MLX_DIR` 通过 harness propagate 给 subprocess。

**Profile-disabled 性能要求**:

- 不插入 eval barrier
- 不分配 profile buffer
- runtime flag (env var 解析) 应在启动时由 `OnceLock<bool>` (或类似一次性 cache 机制) 锁定 — **不要每次 forward 都 env lookup**
- 目标: profile-disabled 时 normal forward 路径无可测量 overhead (per-call CPU branch 跳转可接受，env syscall 不可接受)

正常 release build / clippy / sweep_full / bench 不带 `IRONMLX_P5G_PROFILE_MODE` env var (或设为 `off`)，即使 `--features p5g-profile` 编译进 hook，按以上要求实现也走 normal forward 路径。

## § 4 T1-T3 — Profile-Driven Optimizations

### 4.1 候选优化项（pending T0 ranking）

外部 review 4 个候选 + 我读代码后补充。注意 conv_state 在 ironmlx 已经是 `[B, kernel_size-1, conv_dim]` trimmed window (kernel=4 → 3 tokens, 不是 full history)，所以 C2 优化思路只针对 stateful conv path 重设计，**不是**缩短 concat 长度。

| 候选 | 源码 | 优化思路 (op-level only, 见 scope gate 下) |
|---|---|---|
| **C1 compute_g 链** | line 527-539 (Step 5) | (a) token-invariant 常量预计算缓存（`-exp(a_log)` cast to f32 + neg 只算一次跨多 forward）; (b) softplus stabilised 链 fuse（logaddexp / where / mul / exp 合并）；(c) 可能 `mlx::fast` 已有 softplus / silu fused op 可直接复用 |
| **C2 stateful causal depthwise conv** | line 412-423 (Step 2a) + `core/cache/gated_delta.rs:48` | 避免 `concatenate(conv_state, qkv)` 本身。设计 stateful causal depthwise conv path: conv kernel 在 step-stream 上滚动，conv_state 不作显式 concat 输入。**注**：理想 path 涉及 conv kernel rewrite — 触发 Scope gate (见下) |
| **C3 conv1d + silu fusion** | line 425-428 (Step 2b) | 优先：检查 `mlx::fast` 是否已有 fused conv-with-silu / silu 等 op 直接复用 (op-level)。若需写新 fast op 或新 Metal kernel — 触发 Scope gate |
| **C4 t_arr 重建消除** | line 570-571 (Step 7c) | 把 0-dim int32 `t_arr = ((seq,), ()).try_into()` 预计算并缓存（按 seq 长度组 keyed lookup）— 纯 Rust/op level，不触发 gate。若改 Metal kernel signature 接受 const argument — 触发 Scope gate |

**Scope gate (Boss 决策门)**: P5g 默认只动 Rust/MLX op-level / 常量缓存 / 已有 `mlx::fast` op 复用 / graph shape 调整。**若 T0 / T1 实施过程中证明必须新增或重写 Metal kernel 才能拿到主要 ROI，必须暂停 P5g 实施 + 找 Boss 决策**（是否扩 P5g scope 到 kernel-level，或推到独立 phase）。这防止从 op-level refactor 不知觉扩展到 kernel rewrite。

Promote 决策见 § 7.3 (端到端 benchmark, ship 指标)。Layer 3 upper-bound cut 仅用于决定该候选是否值得实施 T1-T3，不作 ship 依据。

### 4.1a 候选 — Post-T0 v2 实测更新 (2026-05-20)

T0 v2 实测后 (HEAD `52c39bd` + `5e35ab2` tracing→stderr fix; harness 663s wall):

**Phase B 实测 GDN occupancy**: 38.3% (PP=16384) — 45.6% (PP=4096). 远超 § 7.1 假设 15-30% 区间。GDN 实际占 prefill 时间是 spec § 1.3 prior 估算 ~20% 的两倍。

**Phase C 实测 top-3 step ranking** (跨 PP=2048/4096/8192/16384 cross-consistent):

| Rank | Step | % of GDN | 对应 § 4.1 prior 候选? |
|---|---|---:|---|
| **1** | **1a_in_proj_qkvz** | **44-46%** | **未列** (Linear quantized matmul, GDN 内最大开销) |
| 2 | 8_norm_proj (RmsNormGated + reshape + out_proj) | 20-21% | 未列 |
| 3 | 7_kernel (gated_delta_step MetalKernel) | 16-17% | C4 t_arr 是 kernel 辅助输入 |

C1 (compute_g) / C2 (stateful conv) / C3 (conv1d+silu) 都不在 top-3。Step 5 compute_g 实测 ~3-5%；Step 2 conv stack 合计 ~8-10%；Step 7c t_arr 构造 ~1%。

**Phase D 实测 ablation deltas** (vs Phase A pp_tps_median):

| Mode | PP=2048 | PP=4096 | PP=8192 | PP=16384 |
|---|---:|---:|---:|---:|
| ablate-compute-g (~C1) | -8.55% | -8.04% | -6.24% | -2.89% |
| ablate-conv (~C2/C3) | -7.81% | -7.54% | -4.97% | -1.36% |
| ablate-t-arr (~C4) | -10.59% | -7.74% | -7.83% | -4.38% |


**结论**:
- **§ 4.1 C1-C4 prior ranking retired** — 不能基于 Phase D upper-bound 给 C1-C4 排序 / 作 T1 选择依据。C1-C4 实际 attack 面合计约 16-20% of GDN (按 Phase C 实测)，远小于 #1 in_proj_qkvz 的 44-46%。
- **新候选 C5 = Fused Input Projection** 作为 T1 primary。

| 候选 | 源码 | 优化思路 |
|---|---|---|
| **C5 Fused input projection** | `forward_on` Step 1a (in_proj_qkvz) + Step 1b (in_proj_ba) | 合并 `in_proj_qkvz`(hidden→2×key_dim+2×value_dim) + `in_proj_ba`(hidden→b_dim+a_dim) 为单一 Linear (hidden→2×key_dim+2×value_dim+b_dim+a_dim)，forward 后 slice 切回。Op-level (不触发 Scope gate)。理论 saving 源: (a) 一次 input `x` load vs 两次; (b) 一次 4-bit quantized GEMM dispatch vs 两次; (c) 大 matmul GPU occupancy 通常优于两个小 matmul。Profiling schema 从 11 字段 (含独立 `1a_in_proj_qkvz` / `1b_in_proj_ba`) 改为 10 字段 (合并为 `1_input_proj_qkvzba`)。融合权重需沿 output axis 0 拼接 packed `weight` + `scales` + `biases`，slice 顺序固定 `[qkvz \| b \| a]`，eager `mlx::transforms::eval` 防 lazy stream-tagged Array 跨线程进 model fields。 |

**实测收益解读 (chatgpt v1 review):** C5 saving 来源**不是**消除 Step 1a 44% 的 matmul 成本本身 (T1 仍执行 q/k/v/z/b/a 所有量化 matmul 计算)，而是**减少两个原本独立 matmul 之间的 dispatch + input load + GEMM occupancy 浪费**。预期端到端 geomean prefill saving 1-3%，**可能不达 § 7.3 promote threshold +5%** → T1 可能 revert。即便 revert，也是 valuable signal — 锁死 "GDN Linear 已 saturated"，T2/T3 转向 Step 8 (out_proj) 或 Step 7 (kernel，可能触发 Scope gate)。

**T2/T3 候选** (待 T1 outcome):

- 若 T1 promote: T2 探索 Step 8 `8_norm_proj` (其中 out_proj 是另一个 Linear quantized matmul, ~10-15% of GDN; RmsNormGated 已用 mlx::fast)
- 若 T1 revert: T2 重新评估 — 候选 Step 8 out_proj，或 reconsider Phase D 根因 (phase order randomized sanity)
- T3 候选: Step 7 kernel 优化 (可能触发 Scope gate; 待 Boss 决策)

### 4.2 单 task 实施模板（每个 Tn 同结构）

```text
Step Tn.1: 读源码确认 op 序列（per T0 profile 指向）
Step Tn.2: 实施优化（修改 gated_delta_net.rs）
Step Tn.3: cargo build + fmt + clippy 全 clean
Step Tn.4: p5_qwen35_moe_smoke 跑 (argmax=11 sentinel)
Step Tn.5: p5_qwen35_moe_batched 跑 (B=2 row-equiv)
Step Tn.6: p5_qwen35_moe_http_smoke 跑
Step Tn.7: iron-bench PP=128 + PP=512 prefill smoke (runs=3 warmup=1)
Step Tn.8: iron-bench PP=2048/4096/8192/16384 prefill sweep (runs=3 warmup=1)
Step Tn.9: iron-bench Decode TG smoke (PP=128/2048/16384 max-tokens=32)
           ↑ PP=16384 关键: § 7.3 要求 "any PP decode TG regression < 2%"
             并保持 +10.3% over omlx at PP=16384
Step Tn.10: Promote decision per § 7.3 ship metrics (端到端 benchmark only):
            - long-prompt PP=2048/4096/8192/16384 geometric-mean prefill > 5% AND
            - 每个 long PP 单点 regression < 2% AND
            - PP=128/512 prefill regression < 2% AND
            - decode TG regression < 2% on PP=128/2048/16384 AND
            - sentinel + batched + http_smoke ALL PASS (sweep_full at T4 close-out)
            - 否则 revert
Step Tn.11: commit (promote: 含实测 numbers per metric; revert: 含负 ROI + 归因)
```

**Sweep_full per-Tn vs close-out only**: sweep_full 19/19 (~140-160s 每次, Qwen3.5-4B-MLX-4bit) 留 T4 close-out 跑一次，不在每 Tn 跑（per P5e/P5f precedent）。Tn 用 sentinel + batched + http_smoke 三个 MoE 集成测试覆盖正确性 gate。

**指标 namespace 区分** (避免歧义):

- **诊断指标** (用于 T0/T1 优化值得性判断): GatedDeltaNet 内部 wall-time cut 比例 (Layer 1 baseline 对比 / Layer 3 upper bound)。**不**作为 ship 依据。
- **Promote (ship) 指标** (用于 T1-T3 promote / revert 决策): § 7.3 端到端 iron-bench benchmark。**唯一** ship 依据。

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

### 7.1a Amendment — Post-T0 v2 (2026-05-20)

T0 v2 实测后此节的 ablation-based ceiling 公式失效。Phase D 实测全部 negative (见 § 4.1a)，**不能**用 "ablation = cut 比例" 套 § 7.1 表推导 P5g target。修订:

- **Phase B GDN occupancy 实测**: PP=2048 = 40.8%, PP=4096 = 45.6%, PP=8192 = 43.8%, PP=16384 = 38.3%. **远超 § 7.1 假设 15-30% 区间** + § 7.1 sanity gate ≥ 10%。
- **Ceiling 数学结构改用 Phase C 实测 step ranking** 替代 Phase D 上界估算 — T1-T3 端到端 ROI 仍以 § 7.3 ship 指标为准 (this section 不预设具体 cut)。

§ 7.1 prior table (假设性 occupancy × cut) 保留作历史 reference，**T1 决策不再基于此表**。

### 7.1 Performance target ceiling 推导 (假设示例，不作 final target) ⚠️ 已被 § 7.1a Amendment 取代 — 仅作历史

P5g 只优化 GatedDeltaNet — 物理上的 end-to-end wall-time 减少上限由 GatedDeltaNet 在 prefill 总 wall-time 的占比 × GatedDeltaNet 内部优化的 cut 比例共同决定。

以 PP=2048 为例 (P5f baseline 1844 tok/s = 1.111s) 假设性 ceiling 表：

| GatedDeltaNet 占比 (待 T0a 实测) | cut 30% | cut 50% | cut 70% |
|---:|---:|---:|---:|
| 假设 15% | 1934 tok/s (+4.9%) | 1992 tok/s (+8.0%) | 2055 tok/s (+11.4%) |
| 假设 20% | 1965 tok/s (+6.6%) | 2057 tok/s (+11.5%) | 2155 tok/s (+16.9%) |
| 假设 25% | 1996 tok/s (+8.2%) | 2104 tok/s (+14.1%) | 2220 tok/s (+20.4%) |
| 假设 30% | 2027 tok/s (+9.9%) | 2154 tok/s (+16.8%) | 2293 tok/s (+24.3%) |

**重要**: 本表仅作 P5g target 数学结构的示例。**不外推**当前 HEAD GatedDeltaNet 占比 — P5e T0 profile 20% 是 P5e baseline 921 tok/s + per-op instrumented direct call 路径的占比, P5f baseline 1844 tok/s 是 HTTP path no-instrument, 两组数据测量路径 / instrumentation 状态 / 基线均不同, **不能直接外推**。Final target 必须由 T0.a 实测当前 HEAD GatedDeltaNet 占比锁定。

### 7.2 Final target — locked post-T0 v2 + T1 revert (2026-05-20)

P5g 整体 outcome: **no GDN-internal optimization promoted**. T1 (C5 fused input projection) revert per § 7.3 (geomean +0.5% threshold + PP=16384 < 2% regression bound; T1 实测 -0.12% geomean + -2.15% PP=16384). T2/T3 跳过，转 T4 close-out per § 7.3 success bar:


**P5g final target lock**: P5g ship state = P5f baseline (no regression, no promotion). 3-way bench (ironmlx / omlx / mlx-lm) 在 T4 close-out 验证 P5g 仍跟 P5f baseline 一致 (± 2% noise floor)。详细数据见 `reports/p5g-final-results.md`。


1. **Step 7 Metal kernel rewrite** (gated_delta_step kernel; spec § 4.1 Scope gate trigger) — 16-17% of GDN 时间未被任何 op-level 改造覆盖。P5g op-level 已尝试 Step 1 (Linear fuse — saturated) + Step 5 / 2 / 7c (Phase D 反常无 ROI 信号),剩下唯一未触碰的高占比 sub-step。Kernel rewrite 需独立 phase scope。
2. **Step 8 out_proj** (Linear quantized matmul, ~10-15% of GDN) — 与 in_proj_qkvz 同 family (4-bit quant), T1 已证 Linear-family saturated at op-level。若要继续 attack out_proj,只能走 kernel-level (同 driver #1)。


- C5 fused input projection 完整实施 + measured + reverted (audit trail in commit `68545b2`).
- T0 v2 raw data + aggregator output (committed to git as part of T4 close-out report `/reports/p5g-final-results.md`).
- 3-way bench data 在 T4 close-out 显示 P5g state vs omlx vs mlx-lm (验证 ship state == P5f baseline)。

### 7.3 Ship 指标 (T1-T3 promote / revert + P5g 整体 success bar)

Promote (ship) 决策**唯一**指标 — 端到端 iron-bench benchmark。**Promote 比较基线 = 当前 Tn 开始时的 branch HEAD state** (= prior Tn 完成或 revert 之后的 commit)。即严格 task-local ROI，不允许跨 Tn 累积 (e.g. T1+3% / T2 +3% 不能合并算 +6%，每个 Tn 独立 ≥ 5% geomean 才 promote；想累积请合并为单一 Tn patch set 一次性提交)。

**Promote 必须全部满足** (相对当前 Tn 起点 HEAD):

- **长 prompt prefill geomean**: PP=2048/4096/8192/16384 **geometric mean** prefill tok/s 提升 > 5% (单一聚合指标，非"每个 PP 单点 >5%")
- **长 prompt 单点 regression**: 每个 long PP 单点 prefill regression < 2% (容许 geomean 增益不均匀，但不容许任何单点显著退步)
- **短 prompt prefill regression**: PP=128/512 各 < 2%
- **Decode TG regression**: PP=128/2048/16384 promote smoke 各 < 2%；其中 PP=16384 必须保持 +10.3% over omlx (P5f close-out 已 ship 的优势)
- **正确性 gates per Tn**: sentinel argmax=11 + batched B=2 row-equiv + http_smoke ALL PASS

`sweep_full 19/19` 和 `4-way bench` 是**两个独立 gate**，都仅在 T4 close-out 跑：

- `sweep_full.sh` 19/19 PASS (Qwen3.5-4B-MLX-4bit, 全集成测试套)
- 4-way bench (ironmlx / mlx-lm / omlx HTTP iron-bench, 6 PP × 5 runs) — 单独执行，**不**包含在 `sweep_full.sh` 内


T4 close-out 额外补 full-PP decode TG sweep (含 PP=512/4096/8192) 验证 ship 指标 "any PP decode regression < 2%" 在全 PP 范围确认。

### 7.4 P5g close-out 必须




1. **GatedAttention 优化** (full attn, 10/40 layers, long-prompt O(S²))
   - 长 prompt 时占比放大 (PP=16384 可能 30%+)
   - SDPA dispatch tuning / KV layout / memory-access pattern
2. **Long-prompt chunk-size sweep** (PP=4096-16384, NOT bypass chunking)
   - 扫 `prefill_chunk_size = 512/1024/1536/2048/3072/4096` 找曲线
3. **Router bypass single-request idle server** — 条件性 (admit overhead > 50ms 测出)
   - `--b-max N > 1` 已 functional, 等多用户场景启用
5. **Metal kernel rewrite for GatedDeltaNet** — 若 P5g op-level 优化不足，expand 到 kernel level

## § 9 Out of Scope / Non-Goals

- 不动 SparseMoeBlock（P5e 已优化, sorted routing shipped）
- 不动 Scheduler / KVCache / forward orchestration（P5f 已 ship）
- 不抄 omlx.patches.gated_delta_advance 实现（per [feedback_no_spec_from_competitors]）
- 不引入 PagedCache 化（[feedback_design_philosophy] 不对齐 omlx）
- 不做 multi-request batching feature changes（保留 `--b-max N > 1` 不变）

## § 10 Task decomposition（writing-plans 阶段细化）

P5g 拟拆为 **5 task**（[feedback_task_breakdown_bounded] 5-7 范围内）：

```
T0: GatedDeltaNet 独立 per-op profile (3-layer protocol per § 3.2,
    HTTP-path harness with 4-phase execution per § 3.5)
    Step T0.a (Phase A + B): whole-prefill baseline (no profile mode)
               + Layer 1 boundary-isolated GDN estimate
               (IRONMLX_P5G_PROFILE_MODE=layer1)
               at PP=2048/4096/8192/16384, 1 warmup + 3 measured each
    Step T0.b (Phase C): Layer 2 breakdown mode
               (IRONMLX_P5G_PROFILE_MODE=layer2); output per-step ms + %;
               Layer 2 / Layer 1 slowdown ratio
    Step T0.c (Phase D): Layer 3 shape-preserving cost ablation
               (IRONMLX_P5G_PROFILE_MODE=ablate-<candidate> for each
               top-3 from T0.b); estimate upper-bound cut per candidate
    Step T0.d: 用 Phase A baseline + T0.a Layer 1 + T0.c ablation 数据
               回写 § 7.2 锁定 final target
    Files: ironmlx/Cargo.toml (add `p5g-profile = []` feature)
           ironmlx/src/nn/gated_delta_net.rs (PIVOT: 不动 public 签名 —
           `from_loader` 内部从 prefix 解析 layer index, 避免 cascade
           到 common `nn/decoder_layer.rs` 路径污染 dense/MTP scope-OOS;
           struct 加 #[cfg(feature="p5g-profile")] profile_layer_idx
           字段; **`from_components` 也必须在 #[cfg(...)] 下初始化字段**
           为 None (否则 `cargo +nightly clippy --all-features` 编译失败);
           instrument hook, double-gated by p5g-profile feature +
           IRONMLX_P5G_PROFILE_MODE env var with OnceLock<ProfileMode>
           cached; tracing::info! AFTER timer.stop() ONLY, Layer 2
           batched at forward end)
           ironmlx/tests/p5g_t0_gated_delta_profile.rs (#[ignore], heavy;
           HTTP-path harness; server via env!("CARGO_BIN_EXE_ironmlx");
           iron-bench via std::process::Command "cargo run -p iron-bench"
           or pre-built target/release/iron-bench — cross-package
           CARGO_BIN_EXE_iron-bench is NOT injected in ironmlx tests,
           DO NOT use env!() for iron-bench; NO PATH dependency;
           4-phase execution per § 3.5)
           reports/p5g-t0-gated-delta-profile.md
    Commit

T1: highest single-ROI optimization (by T0.c ranking)
    Steps per § 4.2 implementation template (Tn.1-Tn.11):
      build / fmt / clippy / smoke / batched / http_smoke /
      short-PP prefill smoke / long-PP prefill sweep /
      decode TG smoke (PP=128/2048/16384) / promote-revert /
      commit
    Promote 阈值 per § 7.3 ship metrics (单一权威定义，全部满足；
    比较基线 = 当前 Tn 起点 HEAD = prior Tn promote/revert 后 commit):
      - long-PP=2048/4096/8192/16384 geometric mean prefill > 5% AND
      - long-PP 单点 prefill regression < 2% AND
      - short-PP (128/512) prefill regression < 2% AND
      - decode TG on PP=128/2048/16384 regression < 2% (16384 keep
        +10.3% over omlx) AND
      - sentinel + batched + http_smoke ALL PASS
    < threshold revert (per P5e/P5f precedent: 失败实验也 commit
    negative ROI doc 进 close-out 报告作记录)
    注: 单 Tn 不允许累积跨 Tn ROI; 想合并多个微优化请整合为单一 Tn patch set 一次性提交

T2: 2nd highest ROI optimization (same structure as T1)
T3: 3rd highest ROI optimization (same structure as T1)

T4: P5g close-out
    - 跑同 reports/p5f-final-results.md style 4-way bench
    - 跑 full-PP decode TG sweep (含 PP=512/4096/8192 补全 Tn promote
      没覆盖的 PP, 验证 § 7.3 "any PP decode regression < 2%")
    - 写 reports/p5g-final-results.md (self-contained for offline analysis)
    - sweep_full 19/19 (Qwen3.5-4B-MLX-4bit)
    - commit
```

## § 11 References

- [reports/p5f-final-results.md](../../../reports/p5f-final-results.md) — P5f close-out (baseline for P5g)
- [reports/p5e-t0-profile.md](../../../reports/p5e-t0-profile.md) — Historical P5e T0 profile (~20% GatedDeltaNet at PP=2048 on direct-call instrumented 921 tok/s baseline). **Not** used for P5g target extrapolation; T0.a must re-measure current HEAD occupancy on P5f HTTP path 1844 tok/s baseline.
- [docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md](2026-05-19-ironmlx-p5f-known-path-perf-design.md) — P5f spec（process precedent）
- External review notes (not committed to repo per [feedback_no_unnecessary_docs]; substance internalized to § 1.3 candidates + § 4.1 优化思路)
- `memory[no-spec-from-competitors]` — 实现独立
- `memory[design-philosophy]` — 不对齐 omlx
- `memory[task-breakdown-bounded]` — 单 plan 5-7 task
- `memory[honest-answers-no-sycophancy]` — brainstorm 客观判断
- `memory[no-unnecessary-docs]` — 不必要文档不提交
