# ironmlx P5f — MoE text-only known-path 性能优化设计

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-19 |
| 状态 | Brainstorming approved，准备 writing-plans |
| 范围 | Scheduler single-request fast path + GenerationStream single-shot fallback。**已知路径** known-path 工程优化（实测数据已验证 ROI）。 |
| 工作分支 | `ironmlx-p5e-perf`（continuing on the same branch as P5e），base HEAD `a4249af` |
| 上游分支 | 待定（可能 `ironmlx-p5e-perf` 整体 merge 到 `ironmlx-p5-moe` 后再开 P5g） |
| 硬件 | M5 Max + 128 GB unified |
| 验证模型 | `mlx-community/Qwen3.5-35B-A3B-4bit`（PP=128/512/2048/4096/8192/16384） |
| 验收 | ironmlx HTTP-level prefill/decode/e2e 数据 close gap vs omlx 在 sanity-predicted 范围内；sentinel + batched + sweep_full 全 PASS。 |
| 显式 out-of-scope | GatedDeltaNet / GatedAttention 优化（留 P5g）；mlx::compile wrap（API gap 阻断）；PagedCache 化（不对齐 omlx 实现选择）；MoE pad-row skip（ragged batch path 留 P6）；router bypass（如 P5f close-out 显示 admission overhead 显著再 P5g 处理） |
| 性能目标 | **全 PP 段 prefill/decode/e2e 超过 omlx +10%** 是 P5f + P5g 联合目标；P5f 单独**不一定一次性达成**。P5f 期望达到 4-6 个 PP 段中 PP=128/4096/8192/16384 close to target；PP=2048 + PP=512 残余 gap 留 P5g。 |

## § 1 调研依据与决策摘要

### 1.1 P5e 三方 bench 基线（HEAD `a4249af`，M5 Max 128GB）

来源：`reports/p5e-three-way-bench.md`（commit `fc8e6c6`，sanity 数据合并到 `reports/p5e-three-way-bench.md` 待整理）。

**Prefill PP tok/s（median across 5 runs）：**

| PP | ironmlx | mlx-lm | omlx | omlx+10% target | ironmlx/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 390 | 539 | 1088 | 1197 | 0.36× |
| 512 | 491 | 1441 | 2623 | 2886 | 0.19× |
| 2048 | 1842 | 2348 | 4227 | 4649 | 0.44× |
| 4096 | 1773 | 3881 | 4419 | 4861 | 0.40× |
| 8192 | 1725 | 4002 | 4261 | 4687 | 0.40× |
| 16384 | 1548 | 3048 | 3669 | 4036 | 0.42× |

**Decode TG tok/s（median）：**

| PP | ironmlx | mlx-lm | omlx | omlx+10% target |
|---:|---:|---:|---:|---:|
| 128 | 79 | 101 | 129 | 142 |
| 512 | 79 | 102 | 128 | 141 |
| 2048 | 124 | 103 | 124 | 136 |
| 4096 | 121 | 108 | 124 | 136 |
| 8192 | 118 | 105 | 122 | 134 |
| 16384 | 112 | 100 | 103 | 114 |

### 1.2 ChatGPT 协助分析的 hypothesis 已源码核对验证

来源：本次 brainstorming 期间 Boss 协调 ChatGPT 的分析（见对话历史）。核心 hypothesis："PP=128/512 prefill dip + decode regime change at PP=2048 主因是 Scheduler `b_max=4` padding"。

源码核对 5/5 claim 验证为真：

| Claim | 源码位置 | 验证 |
|---|---|---|
| `b_max` default = 4 | [ironmlx/src/cli/serve.rs:34-35](../../../ironmlx/src/cli/serve.rs#L34-L35) | ✅ |
| Scheduler prefill 构造 `[B=b_max, T_max]` padded | [ironmlx/src/core/scheduler.rs:815-860](../../../ironmlx/src/core/scheduler.rs#L815-L860) | ✅ |
| Decode step 也用 `b_max` 构造 `[B,1]` | [ironmlx/src/core/scheduler.rs:1100-1160](../../../ironmlx/src/core/scheduler.rs#L1100-L1160) | ✅ |
| SparseMoeBlock flatten 无 pad mask | [ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:195](../../../ironmlx/src/models/qwen3_5_moe/sparse_moe.rs#L195) | ✅ |
| MoE make_cache 已 `.with_step(cap)` 预分配 | [ironmlx/src/models/qwen3_5_moe/model.rs:168-178](../../../ironmlx/src/models/qwen3_5_moe/model.rs#L168-L178) | ✅ |
| 路由阈值 prefill_chunk_size | [ironmlx/src/core/server/openai.rs:407](../../../ironmlx/src/core/server/openai.rs#L407): `let use_scheduler = state.prefill_chunk_size == 0 \|\| prompt_len <= state.prefill_chunk_size;` | ✅ |

### 1.3 b_max=1 sanity 实测（验证 T1 ROI）

启动 `ironmlx serve --b-max 1`，跑同样 6 PP × 5 runs iron-bench：

| PP | b_max=4 prefill | **b_max=1 prefill** | bmax1/b_max=4 | b_max=4 decode | **b_max=1 decode** |
|---:|---:|---:|---:|---:|---:|
| 128 | 390 | **951** | **2.44×** | 79.6 | **125.6** (1.58×) |
| 512 | 491 | **1577** | **3.21×** | 78.5 | **123.7** (1.58×) |
| 2048 | 1842 | 1843 | 1.00× | 123.7 | 124.9 |
| 4096 | 1773 | 1833 | 1.03× | 121.4 | 122.3 |
| 8192 | 1725 | 1724 | 1.00× | 118.0 | 117.5 |
| 16384 | 1548 | 1606 | 1.04× | 112.0 | 117.0 |

**关键洞察：**

1. **PP=128/512 收益 2.4-3.2× — 来自 Scheduler padding 消除**（active_count==1 时不再按 [b_max=4, T_max] padding 跑 MoE 全 batch）。
2. **PP=2048+ 不受影响 — 早已走 GenerationStream B=1 路径**（prompt + chat_template 长度 > prefill_chunk_size=2048 自然 route 到 GS）。
3. **Decode TG 同步跳变（短 prompt 79 → 125 tok/s）— pad row 进 MoE 计算的副效应**：pad row `per_row_lens=0` 仅跳过 KV write，不跳过 router/top-k/3×gather_qmm/shared_expert，所以 4 个 row 的 MoE 计算量 ≈ 1 row 的 4 倍。
4. **PP=2048+ 长 prompt plateau 来自 chunked prefill**：prompt_len > prefill_chunk_size=2048 时走 GS chunked path，per-chunk eval barrier + 多 chunk setup overhead 累积；这是 T2 改造对象。

### 1.4 Boss 决策记录

- **Phase 达成强度（brainstorming Q1）**：选项 2 — P5f 分批推进，允许 P5g/P5h；P5f 攻 known-path，P5g 攻 deep refactor。
- **T1 实现策略（brainstorming Q2）**：选项 a — Scheduler 内部 fast path（active_count==1 时构造 [1, T_active]），不引入 router bypass（保留架构 evolution 选项）。
- **T2 chunked prefill 范围（brainstorming Q3）**：Single-shot fallback when KV budget allows（不做 mlx::compile graph 复用）。
- **T0 instrumented profile（brainstorming Q4）**：不单独做；现有 P5e three-way bench + b_max=1 sanity + P5e T0 profile 数据足够定位 T1/T2 scope。

## § 2 Architecture

两个 sub-optimization **正交**，互不依赖，可独立 ship：

```
┌──────────────────────────────────────────────────────────────────┐
│ HTTP Request (OpenAI compat /v1/chat/completions)                │
│   render_and_encode → prompt_ids: [N]                            │
│   if N <= prefill_chunk_size (default 2048): route to Scheduler  │
│   else: route to GenerationStream                                │
└──────────────────────────────────────────────────────────────────┘
                │                                       │
                ▼                                       ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ Scheduler path (B=b_max)     │    │ GenerationStream path (B=1)  │
│                              │    │                              │
│ T1: CLI default b_max=1      │    │ T2: when KV budget allows,   │
│  (single-request optimized)  │    │     single-shot forward      │
│  --b-max N>1 opt-in for      │    │     instead of chunked       │
│   future multi-request       │    │                              │
│                              │    │                              │
│ Affects:                     │    │ Affects:                     │
│  PP=128/512 (default route)  │    │  PP=4096/8192/16384          │
│                              │    │                              │
│ Sanity-verified ROI          │    │ Expected ROI:                │
│  (--b-max 1 实测):           │    │  PP=4096 prefill 1.9-2.5×    │
│  PP=128 prefill 2.44×        │    │  PP=8192 prefill 2.0-2.6×    │
│  PP=512 prefill 3.21×        │    │  PP=16384 prefill 1.9-2.5×   │
│  PP=128/512 decode 1.58×     │    │                              │
└──────────────────────────────┘    └──────────────────────────────┘
                │                                       │
                └──────────────────┬────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │ Model::forward_on        │
                    │  (unchanged in P5f)      │
                    │  GatedDeltaNet/Attention │
                    │  → P5g scope             │
                    └──────────────────────────┘
```

## § 3 T1 — CLI Default b_max = 1

### 3.1 设计变更 vs 原方案

原 brainstorming Q2 选项 (a) "Scheduler 内部 runtime fast path (active_count==1 → [1, T])" 在 spec 写 plan 阶段经源码验证不可行：

- [ironmlx/src/core/cache/kv_cache.rs:131-138](../../../ironmlx/src/core/cache/kv_cache.rs#L131-L138) 强制 `per_row_lens.len() == self.batch as usize`，KVCache batch dim 在 `make_cache(batch, ...)` 时锁定，不可 runtime 改。
- "active_count==1 时构造 [1, T] 但 cache 仍 [b_max, ...]" 不可行（KVCache 拒绝 batch mismatch）；要做必须 cache rebuild 跨 admit，是 architectural change。

经 Boss 决策（见对话 2026-05-19）改为 **Option 1: CLI default `b_max = 4 → 1`**。`--b-max 1` sanity 已实测验证 2.44-3.21× 收益，本方案直接把这个 launch-time 配置作为 default。

### 3.2 触点

| 文件 | 修改 |
|---|---|
| `ironmlx/src/cli/serve.rs:34-35` | `#[arg(long, default_value_t = 4)] pub b_max: usize` → `default_value_t = 1` |
| `ironmlx/src/cli/serve.rs` | 启动时 INFO log 一行 `"running with b_max={N}; for concurrent batching, use --b-max N > 1"` |
| README / docs | 一段说明 single-request 默认优化 + multi-request 显式 opt-in 的 rationale |

无 Scheduler / KVCache / forward path 改动。所有现有 b_max>1 code path 保留。

### 3.3 算法

```rust
// ironmlx/src/cli/serve.rs (current):
#[arg(long, default_value_t = 4)]
pub b_max: usize,

// → New (T1):
#[arg(long, default_value_t = 1)]
pub b_max: usize,

// 启动时 (在 build_scheduler / state 初始化路径中):
tracing::info!(
    "ironmlx serve: b_max={} (single-request optimized by default; pass \
     --b-max N > 1 to enable concurrent batching)",
    self.b_max,
);
```

### 3.4 兼容性 / 不破坏点

- **multi-request 用户显式 opt-in**：`ironmlx serve --b-max 4` 完全恢复之前 default 行为。
- **Scheduler / KVCache / forward 路径完全不变**: 所有 b_max>1 invariants 保留。
- **数值正确性**: b_max=1 是已有 code path（admit/step/forward 已支持），跟启动 `--b-max 1` 效果完全一致。
- **multi-request batching feature 不消失**：仅 default 切换；feature 仍存在，需显式启用。

### 3.5 风险与缓解

| 风险 | 缓解 |
|---|---|
| 用户期望 default 是 batch=4 (基于历史习惯) | INFO log + README 说明，breaking change 在 CHANGELOG / release note 显式标注 |
| 集成测试 (sweep_full) 中如有暗自依赖 default b_max=4 行为的用例 | T1 实施时显式 grep `b_max` / `b-max` references，确保 sweep_full / smoke / batched / http_smoke 测试 PASS |
| 未来 P5g/P6 multi-request 启用时遗忘改 default 回 4 | P5g spec / Roadmap 显式记录："multi-request batching 启用后 evaluate 是否调回 default" |

### 3.6 验证

| Test | Pass criterion |
|---|---|
| `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel` | argmax = 11 |
| `p5_qwen35_moe_smoke::p5b_smoke_forward_shape_and_finite` | PASS |
| `p5_qwen35_moe_batched::p5b_batched_row_equivalence` | B=2 vs B=1 per-row identical (测试显式 batched 不受 default 影响) |
| `p5_qwen35_moe_http_smoke::p5b_http_chat_smoke` | PASS (default b_max=1 下 HTTP 路径正确) |
| 显式 `--b-max 4` 启动跑同 sentinel | PASS (multi-request path 仍 functional) |
| iron-bench validate (default 启动) | PP=128 prefill ≥ 950 tok/s, PP=512 ≥ 1500 tok/s (match sanity) |

## § 4 T2 — GenerationStream Single-Shot When KV Budget Allows

### 4.1 触点

| 文件 | 修改 |
|---|---|
| `ironmlx/src/core/generate.rs` | GenerationStream prefill loop：改成 "first try single-shot if budget OK, fallback chunked" |
| `ironmlx/src/core/memory_budget.rs` | 新增 `estimate_prefill_kv_peak_bytes(model_meta, prompt_len, dtype) -> u64` |
| `ironmlx/src/core/memory_budget.rs` | 新增 `available_kv_budget_bytes() -> u64`（unified memory 容量减去 model + activations reserve） |

### 4.2 算法

```rust
// generate.rs::GenerationStream::prefill (current chunked path)
let prefill_chunk_size = request.prefill_chunk_size;
for chunk_start in (0..prompt_len).step_by(prefill_chunk_size) {
    let chunk_end = (chunk_start + prefill_chunk_size).min(prompt_len);
    let chunk_ids = &prompt_ids[chunk_start..chunk_end];
    let chunk_hidden = self.forward_text_hidden(chunk_ids, &cache, /* ... */)?;
    mlx::transforms::eval(&[&chunk_hidden])?;  // per-chunk eval barrier
}
// final chunk runs forward + lm_head

// → New (T2)
let prompt_len = request.prompt_ids.len();
if prompt_len <= request.prefill_chunk_size {
    // 现状: 单 forward + lm_head
    self.forward_full(/*...*/)?;
} else {
    let kv_peak = estimate_prefill_kv_peak_bytes(model_meta, prompt_len, dtype);
    let budget = available_kv_budget_bytes();
    if kv_peak <= budget {
        // 新路径: single-shot 长 prompt（绕开 chunked loop）
        self.forward_full(/*...*/)?;
    } else {
        // 现状: chunked path（保留作 fallback）
        for chunk_start in (0..prompt_len).step_by(request.prefill_chunk_size) {
            /* ... */
        }
    }
}
```

### 4.3 Memory budget 估算

KV 内存峰值（Qwen3.5-MoE-A3B, bf16 dtype）：

```
num_full_attn_layers = 10  (out of 40, full_attention_interval=4)
n_kv_heads = 4
head_dim = 128
dtype_size = 2 bytes (bf16)

kv_per_token_per_layer = 2 (K+V) × n_kv_heads × head_dim × dtype_size
                       = 2 × 4 × 128 × 2 = 2 KB/token/layer

kv_peak_bytes = num_full_attn_layers × kv_per_token_per_layer × prompt_len
              = 10 × 2 KB × prompt_len
              = 20 KB/token total

PP=16384 → 320 MB
PP=131072 → 2.5 GB
```

Available budget：

```
total_unified_memory = 128 GB (M5 Max; query via sysctl hw.memsize or mlx API)
model_resident      = 19.94 GB (4-bit quantized)
activations_reserve = total × (1 - safety_factor)
                    = total × 0.3 (safety_factor=0.7 default)

available_kv_budget = total × safety_factor − model_resident
                    = 128 × 0.7 − 19.94 ≈ 69.7 GB
```

PP=16384 single-shot KV peak = 320 MB << 69.7 GB budget → 单 shot 完全可行。

### 4.4 兼容性 / 不破坏点

- **Chunked path 保留作 fallback**：超长 prompt（> budget allow）仍走原 chunked 行为。
- **Numerical** ：single-shot vs chunked 是 lazy graph composition 差异，数学上等价；可能 ULP 漂移但 argmax 应不变。
- **CLI flag override**：暴露 `--max-single-shot-prompt-len <N>` 允许用户强制 chunked（debug / 内存敏感场景）。

### 4.5 风险与缓解

| 风险 | 缓解 |
|---|---|
| Single-shot 长 prompt 时 MLX 内部 transient buffer 比估算大，OOM / swap | safety_factor 保守 0.7；T2 实施前做 feasibility quick check (PP=16384 单跑确认无 swap) |
| 数值不等价（lazy graph 重排）触发 sentinel argmax 漂移 | sentinel + batched + 新增 sentinel-long (PP=4096 走 single-shot path) 必须 PASS |
| Memory budget 查询失败（macOS API 变动） | 默认 fallback 到保守静态 budget（4 GB 留给 KV，PP=200K 内能用） |

### 4.6 验证

| Test | Pass criterion |
|---|---|
| `p5_qwen35_moe_smoke` (default PP=128 path 走 chunked-free) | argmax = 11，shape OK |
| `p5_qwen35_moe_batched` (B=2 row-equivalence) | PASS |
| `p5_qwen35_moe_http_smoke` | PASS |
| 新增 `p5f_long_prompt_single_shot` test (PP=4096 触发 single-shot path) | 数值 sentinel + 与现有 chunked path argmax 一致 |
| iron-bench validate | PP=4096 prefill > 3500 tok/s, PP=8192 > 3500, PP=16384 > 3000 |

## § 5 Numerical Safety

P5f 改动都不动 forward 算法本身，仅改 dispatch / batching 路径。预期数值等价（bit-level 或 ULP 漂移内）。

**Regression sentinel suite（沿用 P5e）：**

- `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel`: argmax = 11 (`,` token)
- `p5_qwen35_moe_smoke::p5b_smoke_forward_shape_and_finite`: shape + finite check
- `p5_qwen35_moe_batched::p5b_batched_row_equivalence`: B=2 vs B=1 per-row identical
- `p5_qwen35_moe_http_smoke::p5b_http_chat_smoke`: HTTP chat completion 流水正确

**P5f 新增（T2 driven）：**

- `p5f_long_prompt_single_shot`: PP=4096 prompt 通过 GenerationStream，对照 chunked path 同 prompt 输出，argmax 一致。

每个 task 完成时跑完整 sentinel suite；close-out 跑 sweep_full 19/19。

## § 6 Validation Gates

每个 sub-task 完成时：

| Gate | Command | 必须 PASS |
|---|---|---|
| Build | `MLX_DIR=$HOME/.local/mlx cargo build --release` | Finished, 0 warning |
| fmt | `cargo +nightly fmt --all -- --check` | clean |
| clippy | `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings` | 0 warnings |
| Smoke | `cargo test --release --test p5_qwen35_moe_smoke -- --ignored` | 2/2 PASS, argmax=11 |
| Batched | `cargo test --release --test p5_qwen35_moe_batched -- --ignored` | 1/1 PASS |
| HTTP smoke | `cargo test --release --test p5_qwen35_moe_http_smoke -- --ignored` | 1/1 PASS |
| sweep_full (close-out only) | `./scripts/sweep/sweep_full.sh` | 19/19 PASS |
| 4-way bench (close-out only) | iron-bench against ironmlx / mlx-lm / omlx, 6 PP × 5 runs | 量化 P5f vs target gap |

## § 7 Acceptance Criteria

P5f 落地的 measurable success criteria（按 prefill PP tok/s 作主代表 metric；decode/e2e 等比要求）：

| PP | Current (HEAD a4249af, default b_max=4) | T1 expected (default b_max=1) | T2 expected | omlx+10% target | P5f 完后 vs target |
|---:|---:|---:|---:|---:|---|
| 128 | 390 | 950 (sanity 验证) | 950 | 1197 | **79% — 留 P5g 补 26%** |
| 512 | 491 | 1577 (sanity 验证) | 1577 | 2886 | **55% — 留 P5g 补** |
| 2048 | 1842 | 1842 (default 改不影响 GS 路径) | 1842 | 4649 | **40% — 留 P5g 补**（GatedDeltaNet/Attention 主战场） |
| 4096 | 1773 | 1833 | 3500-4500 | 4861 | **72-93% — 接近 target** |
| 8192 | 1725 | 1724 | 3500-4500 | 4687 | **74-96%** |
| 16384 | 1548 | 1606 | 3000-4000 | 4036 | **74-99%** |

**P5f 最低 success bar**: 实现 sanity-predicted ROI（T1: PP=128/512 prefill 2.4-3.2×, decode 1.58×；T2: PP=4096-16384 prefill 1.9-2.5×）。具体能否到 +10% target 由 P5g GatedDeltaNet/Attention 优化补齐。

**P5f close-out 必须量化 P5g scope**: 报告里明确"剩余 gap 来自 GatedDeltaNet ?% / Attention ?% / Scheduler admission ?% / 其他 ?%"，P5g 拿到这数据后再写 spec。

## § 8 P5g preview / Future phases（out of P5f scope）

P5f close-out 输出会驱动 P5g scope。当前已知候选（实施前 deps：P5f close-out 数据）：

1. **GatedDeltaNet 优化** (linear attn, 30/40 layers, T0 profile 20% 占比 at PP=2048)
   - 独立 profile + 重写 recurrent loop / fuse some ops / 可能 Metal kernel 替换
2. **GatedAttention 优化** (full attn, 10/40 layers, T0 profile 6.5% at PP=2048, O(S²))
   - 长 prompt 时占比放大 (PP=16384 可能 30%+)
3. **Router bypass for single-request idle server** (ChatGPT P0-2 候选)
   - 触发条件：P5f close-out 测出 Scheduler admission/queue overhead > 50ms
4. **解耦 prefill_chunk_size 三身任**（scheduler chunk vs GS chunk vs 路由阈值）
   - 跟 (3) 一起做更顺手

### Multi-request batching support (P5h / P6+，必须保留)

Boss 决策（2026-05-19 brainstorming）：T1 选 Option 1 (CLI default `b_max=1`) 把 single-request
做最优作为 default；**multi-request batching feature 不能丢失**，需在未来计划中保留：

- **当前状态**：multi-request batching 已实现于 Scheduler（`--b-max N > 1` 可启用），但 default
  从 4 → 1 后，"开箱即用"路径走 single-request。
- **未来 phase 触发条件**：当 ironmlx 进入 multi-user / agent-fleet 场景，concurrent
  request throughput 成为 primary metric 时，启动以下工作：
  - 评估 default `b_max` 调整（可能根据硬件 / 模型 size 自适应）
  - PagedCache 设计 (block-based KV，跨请求灵活分配；P5f scope 外但 P5g 之后可考虑)
  - Ragged batching (per-token compact MoE dispatch, 跳过 pad row 的 router/topk/gather_qmm)
  - admit_mid 路径效率 (混合长短 prompt batch 的 chunked admission)
  - Scheduler runtime dynamic b_max (cache resize lifecycle)
- **不丢失的承诺**：P5f close-out 报告 + 后续 phase 规划文档中必须显式列出 multi-request
  batching 是 deferred capability，标注启用 path 与 owner，避免被遗忘。

### Brainstorm process 教训（记录避免重蹈）

P5f spec 经历 brainstorm Q2 提的 (a) "Scheduler 内部 runtime fast path" 在 plan 阶段经 KVCache
源码验证不可行：`update_and_fetch_on` 强制 `per_row_lens.len() == self.batch`，导致 cache
runtime batch 切换是 architectural change。Boss 决策回退到 Option 1 (CLI default=1)。

**教训**：未来 brainstorm propose 实现选项前，应先 due-diligence 读关键源码 invariants
（KVCache / forward_on / cache lifecycle 类）再提选项；不要狭义 framing 选项空间
（"runtime conditional logic" vs "launch-time default" 都是合法 dimension）。

## § 9 Out of Scope / Non-Goals

- 不动 Scheduler / KVCache / forward path 架构（T1 仅改 CLI default 一个常量）
- 不做 Scheduler runtime active_count==1 fast path（KVCache batch invariant 阻断；详 § 3.1）
- 不做 router bypass for single-request idle server（条件性留 P5g）
- 不做 PagedCache 化（不对齐 omlx 实现选择，per `feedback_design_philosophy`；multi-request 未来再评估）
- 不做 mlx::compile wrap（P5e T2 已验证 4 个 API gap 阻断；留待"compile-everywhere"专项 task）
- 不做 MoE pad-row skip / ragged batch path（multi-request batching feature 一部分，留 P5h/P6+）
- 不做 sorted-routing 微优化（cache token_idx, put_along_axis — ROI 小 < 2%）
- 不调整 `SORTED_ROUTING_MIN_BS_K` 阈值（512 已对齐 MLX fast-path floor，无 evidence 要改）
- 不做 GatedDeltaNet / GatedAttention 算法优化（P5g 主战场）
- **不删除 multi-request batching 功能**：`--b-max N > 1` 仍支持；本次仅切 default。

## § 10 Task decomposition（writing-plans 阶段细化）

P5f 拟拆为 **4 task**（[feedback_task_breakdown_bounded] 5-7 范围内）：

```
T0: Reference baseline 数据点确认
    - 复用 reports/p5e-three-way-bench.md (HEAD a4249af) baseline
    - 复用 2026-05-19 b_max=1 sanity 实测数据 (PP=128 951 / PP=512 1577 tok/s)
    - 如必要补一次轻量 4-way bench 确认 HEAD 未漂移
    - 输出: reports/p5f-baseline.md (引用现有 + 短 delta note)

T1: CLI default b_max = 1
    - 改 ironmlx/src/cli/serve.rs:34-35 default_value_t = 4 → 1
    - 启动 INFO log 一行 ("running with b_max=N; for batching use --b-max N > 1")
    - 检查 sweep_full / 测试中是否有依赖 b_max=4 default 的隐含假设
      → 如有，给测试显式传 --b-max 4 或测试自己配置 b_max=4
    - sentinel + batched + http_smoke (验证 default b_max=1 不破坏)
    - 显式 --b-max 4 跑一遍 sentinel (验证 multi-request path 仍 functional)
    - iron-bench validate default 启动 (PP=128 ≥ 950, PP=512 ≥ 1500 tok/s)
    - 更新 README / CHANGELOG 说明 default 切换
    - commit

T2: GenerationStream single-shot when KV budget allows
    - 添 memory_budget estimate helper:
        - estimate_prefill_kv_peak_bytes(model_meta, prompt_len, dtype) -> u64
        - available_kv_budget_bytes() -> u64
    - 改 generate.rs::GenerationStream::new_text_only prefill loop dispatch:
        if prompt_len > prefill_chunk_size:
            if kv_peak <= budget: single-shot forward (新)
            else: chunked path (保留)
    - feasibility check (PP=16384 single-shot 内存峰值 quick verify, 无 swap)
    - sentinel + 新增 p5f_long_prompt_single_shot test (PP=4096 触发 single-shot) + batched
    - iron-bench validate (PP=4096 prefill ≥ 3500 tok/s, PP=8192/16384 hit estimate)
    - commit

T3: P5f close-out
    - 跑同 reports/p5e-three-way-bench.md style 4-way bench (ironmlx / mlx-lm / omlx)
    - 写 reports/p5f-final-results.md（self-contained for chatgpt 分析）
        - 含 P5f vs T0 baseline + omlx target gap table
        - 量化 P5g scope (per-PP 残余 gap 归因: GatedDeltaNet / Attn / 其他)
        - 显式记录 multi-request batching 是 deferred capability (P5h/P6+)
    - sweep_full 19/19 PASS
    - commit
```

## § 11 References

- [reports/p5e-three-way-bench.md](../../../reports/p5e-three-way-bench.md) — 4-way HTTP bench baseline (HEAD `a4249af`)
- [reports/p5e-t0-profile.md](../../../reports/p5e-t0-profile.md) — Model::forward_on op-level profile (PP=2048)
- [reports/p5e-final-results.md](../../../reports/p5e-final-results.md) — P5e direct-call measurement (sorted routing 1.92×)
- [docs/superpowers/specs/2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md](2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md) — P5e design spec
- `memory[feedback_no_spec_from_competitors]` — 实现独立，omlx 仅作工程性能参考线
- `memory[feedback_task_breakdown_bounded]` — 单 plan 内 task 数 5-7
- `memory[feedback_design_philosophy]` — ironmlx 不对齐任何竞品
- `memory[reference_current_machine]` — M5 Max 128GB
