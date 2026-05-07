# ironmlx P2: KV cache 设计文档

**日期：** 2026-05-06
**作者：** Claude（与 Boss 协作）
**目标阶段：** ironmlx P2 — full-attention KV cache 基础设施

---

## 1. 范围与决策

P2 在 P1 nn primitives 之上，添加 full-attention 层的 KV cache，让 P3-P7 的 Qwen3.5 推理 + benchmark CLI 能跑 prefill + decode 流程。

### 1.1 已批准的设计决策

| # | 主题 | 决定 | 备注 |
|---|---|---|---|
| Q1 | P2 范围档次 | **A — 精简** | 仅 `KVCache` + Attention 接入 + 测试，~2-3 天工作量 |
| Q2 | KVCache 实现风格 | **C-3** | 必传 `cap`（generate loop 算 prompt+max_new）+ builder `.with_step()` 调增长粒度（默认 256，benchmark 时设为 cap 等同 fixed 一次分配） |
| Q3 | SSM 状态缓存形态 | **C-typed-vec** | 每模型自定义 typed state（在 `models/<m>/` 里），不引入通用 `ArraysCache` |
| Q4 | HTTP server / batched 推理时机 | **(c) — P8 阶段** | 显式列入未来阶段；P2 不预留 batched 抽象 |
| Q5 | 当前 P2 batched 走法 | **A** | 单请求路径；P8 时统一做 batched inference + paged cache + prefix cache + multi-request scheduler |
| Q6 | QuantizedKVCache 时机 | **A — P7.5/P8** | P2 仅做基础 bf16/fp16 KV cache；KV 量化推到 P8 HTTP server（长 context 多请求场景）一起做 |

### 1.2 不在范围内（明确推迟）

- `ArraysCache` / `CacheManager` trait / `LayerCacheConfig` / `ModelCacheConfig` / `Factory`（C-typed-vec 路线下不需要）
- `Qwen35ModelCache` 类型 + `Qwen35LayerCache` enum（属 P3-P4 模型组装阶段；P2 仅提供单层 `KVCache`）
- variable-length batched padding（每条序列长度不同）→ P8
- multi-request paged cache（vLLM 风格 BlockTable / BlockPool）→ P9
- prefix cache（trie / LRU）→ P9
- KV cache quantization（8-bit / 4-bit storage）→ P7.5 或 P8
- Rotating / sliding-window cache（Qwen3.5 不用，推迟）

---

## 2. 架构

P2 在 `ironmlx/src/core/` 下新建 `cache/` 子模块，并修改 P1 已交付的 `nn::Attention::forward` 签名以接受 `Option<&mut KVCache>` 参数。

```
ironmlx/src/
├── core/
│   ├── mod.rs                # +pub mod cache + pub use cache::KVCache
│   └── cache/
│       ├── mod.rs            # NEW
│       └── kv_cache.rs       # NEW
└── nn/
    └── attention.rs          # 修改 forward / forward_on 签名
```

**模块依赖关系**（无环）：
- `core::cache::KVCache` 依赖 `mlx::Array` + 基础 ops（`concatenate` / `slice_strided` / `zeros`，可能 `slice_update`）
- `nn::Attention` 依赖 `core::cache::KVCache`（通过参数传入）

无新外部 crate 依赖。

---

## 3. 详细设计

### 3.1 `core::cache::KVCache`（C-3 实现）

```rust
//! Per-layer KV cache for full-attention layers.
//!
//! Holds keys + values pre-allocated up to `cap` tokens; grows in
//! `step`-size chunks to amortize allocation cost. `update_and_fetch`
//! advances an offset pointer and returns slices of the occupied region.
//!
//! P2 supports single-request usage only (one cache instance per layer
//! per request). Multi-request paged cache is P8/P9 work.

use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

pub struct KVCache {
    /// Allocated K tensor `[batch, n_kv_heads, capacity, head_dim]`.
    /// `None` until first `update_and_fetch` (lazy alloc).
    keys: Option<Array>,
    /// Allocated V tensor `[batch, n_kv_heads, capacity, v_head_dim]`.
    values: Option<Array>,

    /// Current write position (number of tokens already cached).
    offset: i32,
    /// Hard upper bound; `update_and_fetch` errs if exceeded.
    cap: i32,
    /// Grow granularity. Default 256. Set `step >= cap` for one-shot
    /// pre-allocation (benchmark / fixed long context).
    step: i32,

    // Shape metadata captured at construction (immutable).
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}

impl KVCache {
    /// Construct a fresh cache. Keys/values are allocated lazily on first
    /// `update_and_fetch`.
    ///
    /// `cap` is the hard maximum sequence length the cache may hold —
    /// callers compute it as `prompt_tokens + max_new_tokens`.
    pub fn new(
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
    ) -> Self;

    /// Override grow step (default 256). Returns `self` for chaining.
    /// Panics if `step <= 0`.
    pub fn with_step(mut self, step: i32) -> Self;

    pub fn offset(&self) -> i32;
    pub fn cap(&self) -> i32;

    /// Reset offset to 0; retains allocated buffers for reuse on next session.
    pub fn reset(&mut self);

    /// Append `(k, v)` (shape `[batch, n_kv_heads, n_new, head_dim]` for k,
    /// same with `v_head_dim` for v) and return slices covering all cached
    /// tokens (`[..., 0..self.offset, ...]`).
    pub fn update_and_fetch(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)>;

    pub fn update_and_fetch_on(
        &mut self,
        k: &Array, v: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)>;
}
```

**`update_and_fetch` 内部行为**：

1. 检查 `self.offset + n_new > self.cap` —— 超过则返回 `Err("KVCache cap exceeded by N tokens")`
2. 当前已分配容量不够时调 `grow_to(target_capacity)`，target_capacity 按 step 步进 round up，clamped at cap
3. 写 K / V 到 `[..., offset..offset+n_new, ...]` 切片
4. `self.offset += n_new`
5. 返回 K / V 的 `[..., 0..offset, ...]` slice（用 `slice_strided`）

**实施时三选一（T1 实施者评估 cxx-mlx 当前 ops 选）：**

**方案 1：mlx-lm 风格 concatenate**
- First call：直接 assign 新 buffer，写到 `[..n_new]`
- Subsequent：keys = concat([keys[..offset], k_new, zeros[trailing]], axis=2)
- 与 mlx-lm 完全一致；每次 grow 一次大 concat

**方案 2：slice_update 风格（如果 cxx-mlx 已绑定 `slice_update`）**
- First call：alloc zeros buffer + slice_update 写 `[..n_new]`
- Subsequent：slice_update 写 `[offset..offset+n_new]`（in-place）
- grow_to：单次 concat 老 buffer + zeros 新区域，alloc 频率 = grow 频率
- 写入零开销；最优

**方案 3：predetermined-cap 一次分配**
- 永远一次性分配 cap 容量；写入用 slice_update
- 最简单；用户传大 cap 时初始内存大

**T1 决策路径：**
1. T1 实施者第一步 grep `mlx::ops::indexing::slice_update`：
   - 已绑定 → 方案 2
   - 未绑定 → 方案 1（zero-binding-changes path）

公开 API（`new` / `with_step` / `update_and_fetch` / `offset` / `cap` / `reset`）三方案都不变，T1 内部细节可调。

### 3.2 `Attention::forward` 签名扩展

P1 当前签名：

```rust
pub fn forward(
    &self,
    x: &Array, mrope: &Mrope, cos: &Array, sin: &Array,
    mask: Option<&Array>,
) -> Result<Array>

pub fn forward_on(
    &self, x: &Array, mrope: &Mrope, cos: &Array, sin: &Array,
    mask: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>
```

P2 修改为：

```rust
pub fn forward(
    &self,
    x: &Array, mrope: &Mrope, cos: &Array, sin: &Array,
    mask: Option<&Array>,
    cache: Option<&mut crate::core::cache::KVCache>,
) -> Result<Array>

pub fn forward_on(
    &self, x: &Array, mrope: &Mrope, cos: &Array, sin: &Array,
    mask: Option<&Array>,
    cache: Option<&mut crate::core::cache::KVCache>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>
```

**forward 内部增加 cache 路径：**

```text
1. q = q_proj(x) ; k = k_proj(x) ; v = v_proj(x)
2. reshape + transpose 到 [batch, heads, seq, head_dim]
3. q = mrope.apply(q, cos, sin)   # 仍是 P1 stub，P3 实现
   k = mrope.apply(k, cos, sin)
4. q_norm / k_norm 应用（如 cfg.has_qk_norm）
5. IF cache.is_some():
       (k_full, v_full) = cache.update_and_fetch(&k, &v)?
   ELSE:
       (k_full, v_full) = (k, v)
6. SDPA(q, k_full, v_full, scale, mask_mode, mask, sinks)
7. transpose + reshape 回 [batch, seq, hidden]
8. o_proj
```

cache 持有的是已 RoPE-rotated 的 K（参考 mlx-lm `KVCache` 行为；RoPE 在 cache 写入之前）。q_norm/k_norm 也在 cache 之前应用。

### 3.3 KV cache 维度约定

输入 K / V（来自 attention projections + reshape + transpose）：
- `K: [batch, n_kv_heads, seq, head_dim]`
- `V: [batch, n_kv_heads, seq, v_head_dim]`

cache 持有同形态，仅 seq 维度从 0 增长到 cap。

**Qwen3.5 具体值（实际推理时）：**
- `n_kv_heads = 4`（GQA）
- `head_dim = 256`
- `v_head_dim = 256`
- `batch = 1`（P2-P7 范围，单请求）
- `cap` 由 generate loop 算 = `prompt_len + max_new_tokens`

### 3.4 dtype 策略

- 默认 `KVCache` 的 dtype 跟随首次 `update_and_fetch` 输入的 K/V dtype（典型 bf16）
- P2 不提供 dtype upgrade / downgrade（K/V 必须与 cache 一致）
- 不支持 KV cache quantization（推到 P8）

---

## 4. 测试策略

### 4.1 单元测试（`ironmlx/src/core/cache/kv_cache.rs` 内）

- `kv_cache_new_lazy_allocation` — 构造后 keys/values 为 None；offset = 0
- `kv_cache_update_first_call_assigns_buffer` — 第一次 update 后形状正确，offset 推进
- `kv_cache_grows_in_steps` — 连续 update 触发 grow，验证 capacity 按 step 步进
- `kv_cache_cap_exceeded_errors` — 超过 cap 时 `update_and_fetch` 返回 Error
- `kv_cache_reset` — reset 后 offset 归零，下次 update 重用 buffer
- `kv_cache_with_step_overrides_default` — 自定义 step
- `kv_cache_with_step_eq_cap_one_shot` — step == cap 时一次性分配满 cap
- `kv_cache_with_step_panics_on_zero` — assert!(step > 0)
- `kv_cache_returned_slices_have_correct_offset_dim` — fetch 返回的 K/V slice 第三维 == offset

### 4.2 集成测试（`ironmlx/tests/p2_kv_cache.rs`）

- `attention_forward_without_cache_unchanged` — Attention forward 不传 cache 时行为与 P1 一致（回归保护）
- `attention_forward_with_cache_prefill_then_decode` — 构造 Attention，先 prefill seq=8 触发 cache 写入，再 decode seq=1 读 cache 的 K_full/V_full，输出形状正确
- `attention_cache_offset_advances_after_each_call` — 多次 forward 后 cache.offset() 递增

> **注意**：因 `Mrope::cos_sin` / `apply` 仍是 P3 stub，attention 端到端 forward 在 P2 仍返回 Err（来自 mrope.apply 内部）。P2 集成测试关注 cache 接入路径不破坏 P1，不验证 attention 数值正确性。真值校验在 P4 模型组装时做（那时 mrope.apply 已实现）。

### 4.3 回归保证

P1 现有 22 单元测试 + 2 集成测试不变 — Attention forward 签名虽然变化，但默认 `cache=None` 路径与 P1 行为等价；P1 已存在的测试全部不传 cache，自动兼容。

---

## 5. 任务分解

3 个任务：

| # | 任务 | 主要文件 | 依赖 | 估时 |
|---|---|---|---|---|
| T1 | `core::cache::KVCache`（C-3 实现） | `core/cache/mod.rs`、`core/cache/kv_cache.rs` | 无 | 1.5 天 |
| T2 | `Attention::forward / forward_on` 签名扩展 + cache 接入 | `nn/attention.rs` | T1 | 0.5 天 |
| T3 | 集成测试 + P1 回归 | `ironmlx/tests/p2_kv_cache.rs` | T1, T2 | 0.5 天 |

**T1 第一步**：grep `mlx::ops::indexing::slice_update` 验证 cxx-mlx 当前是否绑定。已绑定走方案 2（in-place slice_update）；否则走方案 1（concatenate-based）。

**总计：~2-3 天。**

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| `slice_update` 是否在 cxx-mlx 已绑定？T1 实施策略未定 | T1 第一步 grep 验证 → 已绑定方案 2，未绑定方案 1。两路径公开 API 相同，可后续无破坏切换 |
| Attention forward 签名变更破坏 P1 集成测试 | 默认 `cache=None` 路径与 P1 行为等价；P1 测试不传 cache，自动兼容 |
| P3 模型组装时发现 cache 接口需调整 | KVCache 公开 API 经过 mlx-lm 镜像 + Qwen3.5 实际用法验证；调整概率低 |
| Mrope::apply 在 P1 仍是 stub 导致 P2 集成测试无法端到端跑通 | P2 集成测试只测 cache 接入路径不破坏 P1，不测 attention 数值；真值校验留 P4 |
| cache 持有 RoPE 后还是 RoPE 前的 K | spec § 3.2 明示：cache 持有"已 RoPE-rotated K"，与 mlx-lm `KVCache` 行为一致；q_norm/k_norm 同样在 cache 之前 |
| 长 context 时显存不足 | P2 范围内未优化；用户上层调小 `max_new_tokens` 或减小 prompt；显存优化在 P7.5/P8（QuantizedKVCache + paged）做 |
| `Qwen35ModelCache` enum 何时定义 | P4 模型组装；P2 仅提供 layer-level `KVCache`，不组装 model-level cache |

---

## 7. 与后续阶段的关系

- **P3 Qwen3.5 特殊算子**：`Attention` cache 接口稳定后，gated_attention（Qwen3.5 完整 attention 含 q_norm/k_norm/output_gate）复用此 KVCache；linear_attn (gated delta SSM) 在 `models/qwen3_5/` 自定义 `GatedDeltaCacheState`（C-typed-vec），与 KVCache 并行存在
- **P4 Qwen3.5 Dense 模型组装**：`Qwen35ModelCache { layers: Vec<Qwen35LayerCache> }` enum，每层选 `FullAttn(KVCache)` 或 `GatedDelta(GatedDeltaCacheState)`
- **P5 Qwen3.5 MoE**：复用 P4 的 ModelCache 结构
- **P7 Benchmark CLI**：`KVCache::new(.., cap=prompt+max_new).with_step(cap)` 走 fixed 模式测最优 prefill/decode 性能
- **P7.5 / P8**：`QuantizedKVCache`（int8 / int4 K/V storage），从 KVCache 派生为 enum variant 或独立类型；HTTP server 多请求场景受益最大
- **P8 HTTP server**：在此 KVCache 之上加 `BatchedKVCache`、`PaddedAttentionMask`、`BatchedSampler`；做 streaming SSE + OpenAI/Anthropic 协议
- **P9 Paged cache**：把 single-instance KVCache 抽象为 `BlockTable + BlockPool`，多请求共享 block；vLLM 级 production cache stack

---
