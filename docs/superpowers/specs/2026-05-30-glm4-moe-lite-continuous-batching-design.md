# GLM-4.7-Flash (`glm4_moe_lite`) Continuous-Batching MLA 适配 — 设计

> 状态：设计稿，待 Boss 复审。
> 范围：让 `glm4_moe_lite` 支持 `--b-max > 1`（连续批处理 / 多并发请求），方法是把 `MlaLatentCache` 接进现有的 continuous-batching 行迁移路径。
> Worktree：`ironmlx-glm47-cb-mla`（基于 `ironmlx-glm47-flash` @ `fc1cbae`）。
> 前置：GLM 集成（absorbed-MLA + `MlaLatentCache` + `LayerCache::Mla`）已在 `ironmlx-glm47-flash` 完成；本任务在其上扩展。

## 0. 结论摘要
- continuous-batching 的机制（admission / 行压缩-复用 / mid-chunk admit / decode）在 Qwen/Gemma 上已完整；GLM 唯一缺的是 **Mla cache 的 per-row 迁移**（`adopt_row_from`）。
- 方案：镜像 `KVCache` 的行迁移到 `MlaLatentCache`（双 buffer），补上 2 处 scheduler 的 Mla 臂，移除 `--b-max 1` 拦截，并验证 B>1 decode 在异构 cache 长度下正确。
- 无大架构分叉——行压缩本质就需要 per-row 迁移。
- 验收：正确性（并发逐 token = 串行）+ 连续批处理聚合吞吐**追平/超 omlx**。

---

## 1. 要新增/改的件（均小，file:line 为 `fc1cbae` 现状）

### 1.1 `MlaLatentCache::adopt_row_from`（新增）
文件 `ironmlx/src/models/glm4_moe_lite/mla_cache.rs`（当前公共面：`new/with_step/offsets/cap/dtype/grow_cap/reset/update_and_fetch_on`，**无 `adopt_row_from`**）。
镜像 `KVCache::adopt_row_from`（`ironmlx/src/core/cache/kv_cache.rs`）：
```rust
pub fn adopt_row_from(&mut self, src: &MlaLatentCache, dst_row: usize, src_row: usize) -> Result<()>
```
- 校验 `kv_lora / rope / dtype / batch 边界` 一致（对应 KVCache 的 `n_kv_heads/head_dim/v_head_dim/dtype` 校验）。
- `let src_off = src.offsets[src_row]`；`src_off > self.cap` → err。
- 若 `src_off > 0`：按 `src_off` 增长 self 的**两个 buffer**（`c_kv`、`k_pe`）到 step-rounded 容量；把 `src.c_kv[src_row, .., 0..src_off, ..]` 拷到 `self.c_kv[dst_row, .., 0..src_off, ..]`，`k_pe` 同理（两个 buffer 各按自己的末维宽度 kv_lora / rope）。
- `self.offsets[dst_row] = src_off`。
- 行写入用 `slice_update_on`（与 `update_and_fetch_on` 同款）。

### 1.2 `scheduler::adopt_cache_row_layers`（加 Mla 臂）
`ironmlx/src/core/scheduler.rs:467` 的 match 现有 `Full→Full`、`Linear→Linear`，`_ => Err("cache layer kind mismatch")`。新增：
```rust
(LayerCache::Mla(dst_mla), LayerCache::Mla(src_mla)) => {
    dst_mla.adopt_row_from(src_mla, dst_row, src_row)?;
}
```

### 1.3 mid-admit dtype-finder（加 Mla 臂）
`scheduler.rs:2019` 现 `LayerCache::Full(kv) => Some(kv.dtype()), _ => None`（后接 `.unwrap_or(Bfloat16)`）。加 `LayerCache::Mla(mla) => Some(mla.dtype())`，消除对 bf16 兜底的隐式依赖。

### 1.4 移除 `--b-max>1` 拦截
`ironmlx/src/cli/serve.rs:203-208` 的 `glm4_moe_lite` 臂中 `if args.b_max > 1 { return Err(...) }`——continuous-batching 落地后移除该拦截。

### 已就位（无需改）
- `first_full_layer_offsets`（scheduler.rs:424-426）+ `cache_cap_and_dtype`（:439-441）的 Mla 臂已在（Task 3/6）。
- `model.rs::make_cache`（:219-226）已接受 `batch` 参数 → 构造 B>1 cache 无需改。

---

## 2. B>1 decode 正确性（主要风险——验证为主，非新写）
现有 `MlaAttention` decode 路径**理应已支持 B>1 异构 cache 长度**：
- rope：`rope_with_array_offset_on(offset=[B] i32)`，offset 取自 `caches[0].offsets()`（per-row 位置）。
- cache：`update_and_fetch_on` 接受 `per_row_lens:[B]`。
- decode mask：引擎 `build_per_row_decode_mask` 产 `[B,1,1,Lc]`，GLM 折进 `pe_scores`。
- `run_layers` 的 regime 均匀性断言是对 **`per_row_lens`**（decode 全 1，均匀）而非 cache 长度 → **不阻塞** B>1 decode。

**待验证点（实现期重点）**：
1. per-row decode mask `[B,1,1,Lc]` 与 `pe_scores [B,H,1,Lc]` 的**广播正确**（mask 在 H 维广播）。
2. scheduler 几处 `_ => None`（`:1226 / :1347 / :1355`）在 GLM 路径上**不会静默误处理 Mla**（逐一核实其语义；非 cache-kind 相关则无碍）。
3. mid-chunk admit（`admit_mid_begin/chunk/finalize`）+ 行压缩（`rebuild_cache_layout`，scheduler.rs:898）在 GLM 上端到端正确（靠集成测试覆盖；prefill chunk 走 `forward_on [1,chunk]` 与 decode `[B,1]` 仍分阶段，无 per-row 混合 regime）。

---

## 3. 验收
- **正确性**：GLM `--b-max>1` 下，并发/连续批处理 N 个请求的输出与 `--b-max 1` 串行**逐 token 一致**（greedy bit-identical，temperature 0）；不回归 b-max=1（既有 GLM 测试 + 默认特性套件仍绿）。
- **性能**：iron-bench 实测连续批处理**聚合吞吐 ≥ omlx**（串行跑，避免 GPU/swap 互染）。
  - **前置**：先确认 omlx（mlx-lm）是否支持 glm4_moe_lite 的多并发/连续批处理服务并建基线；若不支持，"追平 omlx"无从定义 → 回退为"聚合吞吐随 b-max 扩展"目标（届时与 Boss 确认）。

## 4. 测试
- **单元**：`MlaLatentCache::adopt_row_from`——双 buffer 行拷贝 + 增长 + offsets 正确（构造 src 多行、adopt 到 dst 不同行，对拍数据 + offsets）。
- **集成**（env-gated，真实 GLM-4.7-Flash-4bit）：
  - b-max>1 并发 N 请求 vs 串行单请求，逐 token 一致；
  - 含 mid-chunk admit 场景（不同长度 prompt 交错入队）；
  - 行压缩/复用（请求完成后槽位被新请求复用）正确。
- **性能**：iron-bench 并发吞吐 vs omlx（串行跑）。

## 5. 风险 / 开放
1. **omlx 连续批处理支持**（§3 前置）——决定 perf 基线是否成立。
2. **mid-admit + 行压缩**在 GLM 上的端到端正确性（§2.3）——靠集成测试覆盖。
3. **B>1 decode mask 广播**（§2.1）——实现期需显式验证。
4. 几处 `_ => None` 的 Mla 语义（§2.2）。

## 6. 参考文件（`fc1cbae` 现状）
- `ironmlx/src/models/glm4_moe_lite/mla_cache.rs`（加 `adopt_row_from`）、`mla_attention.rs`（B>1 decode 路径）、`model.rs`（`run_layers` / `make_cache`）。
- `ironmlx/src/core/cache/kv_cache.rs`（`adopt_row_from` 镜像源）。
- `ironmlx/src/core/scheduler.rs`：`adopt_cache_row_layers`(:451-471)、dtype-finder(:2019)、`first_full_layer_offsets`(:418)、`cache_cap_and_dtype`(:437)、`rebuild_cache_layout`(:898)、`build_per_row_decode_mask`(:1809)、`_ => None`(:1226/1347/1355)。
- `ironmlx/src/cli/serve.rs:203-208`（移除 b-max 拦截）。
