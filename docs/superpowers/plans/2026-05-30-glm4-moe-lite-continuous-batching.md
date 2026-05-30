# GLM-4.7-Flash Continuous-Batching MLA 适配 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `glm4_moe_lite` 支持 `--b-max > 1`（连续批处理 / 多并发请求），方法是把 `MlaLatentCache` 接进现有 continuous-batching 的 per-row 行迁移路径。

**Architecture:** 镜像 `KVCache::adopt_row_from` 到 `MlaLatentCache`（双 buffer：c_kv + k_pe），补 2 处 scheduler 的 `LayerCache::Mla` 臂，移除 `--b-max 1` 拦截，并验证 B>1 decode 在异构 cache 长度下正确（rope array-offset + per-row decode mask + 行压缩/复用）。无新架构——行压缩本质就需 per-row 迁移。

**Tech Stack:** Rust, MLX (`mlx` crate), Apple Silicon。Worktree `ironmlx-glm47-cb-mla`（基于 `ironmlx-glm47-flash` @ `fc1cbae`）。

**Spec:** `docs/superpowers/specs/2026-05-30-glm4-moe-lite-continuous-batching-design.md`。

---

## 环境 & 测试约定
MLX 环境（每条 cargo 命令前 export；见 `docs/mlx-setup.md`）：
```bash
export MLX_ROOT=/tmp/ironmlx-perf-mlx-install-3f6c3113f734
export MLX_DIR="$MLX_ROOT"; export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```
- 本 worktree 基于 `fc1cbae`：测试债已修，`cargo test -p ironmlx --lib <name>` 可直接跑；`cargo test --all-features --workspace` 绿**除** 1 个已知 `mlx --test p3_quantization` 容差项（预存、非本任务，忽略）。
- 每个 Rust commit 前：`cargo +nightly fmt --all`、`fmt --all -- --check`、`cargo +nightly clippy --all-features --workspace -- -D warnings`、`cargo build --release`。
- 真实模型集成测试：`GLM47_MODEL_DIR=$(echo ~/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/*)`。

---

## Task 1: `MlaLatentCache::adopt_row_from`

**Files:**
- Modify: `ironmlx/src/models/glm4_moe_lite/mla_cache.rs`（加方法 + 内联测试）

- [ ] **Step 1: 读镜像源** `ironmlx/src/core/cache/kv_cache.rs::adopt_row_from`（校验 shape/dtype + bounds + `src_off` + `if src_off>0`{grow_to + slice src 行 + slice_update_on 写 dst 行} + `offsets[dst_row]=src_off`）。`MlaLatentCache` 已有 `grow_to` + `slice_strided_on`/`slice_update_on` 导入（`mla_cache.rs:17`）。

- [ ] **Step 2: 写失败测试**（内联 `#[cfg(test)] mod tests`）：
```rust
#[test]
fn adopt_row_copies_src_row_offset_and_data() {
    // src: batch=2, kv_lora=4, rope=2, cap=8. Fill row 1 with 2 tokens.
    let mut src = MlaLatentCache::new(2, 4, 2, Dtype::Float32, 8).with_step(8);
    // row0 lens 0, row1 lens 2: c_kv row1 = 5.0, k_pe row1 = 6.0
    let c_kv: Array = {
        let mut d = vec![0.0_f32; 2 * 1 * 2 * 4]; // [B=2,1,S=2,4]
        for v in d.iter_mut().skip(1 * 2 * 4) { *v = 5.0; } // row1 block
        (&d[..], (2_i32, 1, 2, 4)).try_into().unwrap()
    };
    let k_pe: Array = {
        let mut d = vec![0.0_f32; 2 * 1 * 2 * 2];
        for v in d.iter_mut().skip(1 * 2 * 2) { *v = 6.0; }
        (&d[..], (2_i32, 1, 2, 2)).try_into().unwrap()
    };
    src.update_and_fetch_on(&c_kv, &k_pe, &[0, 2], ()).unwrap();
    assert_eq!(src.offsets(), &[0, 2]);

    // dst: batch=1, same dims. Adopt src row 1 -> dst row 0.
    let mut dst = MlaLatentCache::new(1, 4, 2, Dtype::Float32, 8).with_step(8);
    dst.adopt_row_from(&src, 0, 1).unwrap();
    assert_eq!(dst.offsets(), &[2], "dst row0 offset must == src row1 offset");

    // Read back the adopted data by appending 1 real token and fetching the
    // full [1,1,3,*] history (avoids the all-zero fast path, which returns an
    // empty slice). Tokens 0,1 must be the adopted values; token 2 the new one.
    let new_kv: Array = (&[8.0_f32; 4][..], (1_i32, 1, 1, 4)).try_into().unwrap();
    let new_pe: Array = (&[9.0_f32; 2][..], (1_i32, 1, 1, 2)).try_into().unwrap();
    let (kv_f, pe_f) = dst.update_and_fetch_on(&new_kv, &new_pe, &[1], ()).unwrap();
    assert_eq!(kv_f.shape().as_slice(), &[1, 1, 3, 4]);
    assert_eq!(dst.offsets(), &[3]);
    let kv: Vec<f32> = kv_f.to_vec().unwrap(); // [1,1,3,4] row-major
    for v in kv.iter().take(2 * 4) { assert_eq!(*v, 5.0, "adopted c_kv tokens 0,1 must be 5.0"); }
    for v in kv.iter().take(3 * 4).skip(2 * 4) { assert_eq!(*v, 8.0, "appended c_kv token 2 must be 8.0"); }
    let pe: Vec<f32> = pe_f.to_vec().unwrap(); // [1,1,3,2]
    for v in pe.iter().take(2 * 2) { assert_eq!(*v, 6.0, "adopted k_pe tokens 0,1 must be 6.0"); }
    for v in pe.iter().take(3 * 2).skip(2 * 2) { assert_eq!(*v, 9.0, "appended k_pe token 2 must be 9.0"); }
}

#[test]
fn adopt_row_rejects_dim_mismatch() {
    let src = MlaLatentCache::new(1, 4, 2, Dtype::Float32, 8);
    let mut dst = MlaLatentCache::new(1, 8, 2, Dtype::Float32, 8); // kv_lora differs
    assert!(dst.adopt_row_from(&src, 0, 0).is_err());
}
```
> NOTE: the zero-len fetch path returns `[B,1,max_off,*]` with `max_off = max(offsets) = 2` (the all-zero fast path only triggers when ALL offsets-after are skipped AND nothing cached; here row0 has offset 2 so it returns the cached slice). Verify against `update_and_fetch_on` semantics; if the all-zero path returns empty, instead read back by adopting into a 2-row dst and fetching, or expose a test-only getter. Adjust the readback to whatever cleanly reads dst row0's cached data.

- [ ] **Step 3: 运行 → FAIL** `cargo test -p ironmlx --lib glm4_moe_lite::mla_cache::tests::adopt_row 2>&1 | tail -15`（`adopt_row_from` undefined）。

- [ ] **Step 4: 实现 `adopt_row_from`**（镜像 KVCache，双 buffer）：
```rust
/// Copy src's row `src_row` cached latent (c_kv + k_pe, positions 0..src.offsets[src_row])
/// into self's row `dst_row`, and set self.offsets[dst_row] = src.offsets[src_row].
/// Mirrors KVCache::adopt_row_from for the two differing-width buffers. Used by
/// the scheduler's continuous-batching row compaction (adopt_cache_row_layers).
pub fn adopt_row_from(&mut self, src: &MlaLatentCache, dst_row: usize, src_row: usize) -> Result<()> {
    if self.kv_lora != src.kv_lora || self.rope != src.rope || self.dtype != src.dtype {
        anyhow::bail!(
            "MlaLatentCache::adopt_row_from: shape/dtype mismatch (self={}/{}/{:?}, src={}/{}/{:?})",
            self.kv_lora, self.rope, self.dtype, src.kv_lora, src.rope, src.dtype,
        );
    }
    if dst_row >= self.batch as usize {
        anyhow::bail!("MlaLatentCache::adopt_row_from: dst_row {} >= self.batch {}", dst_row, self.batch);
    }
    if src_row >= src.batch as usize {
        anyhow::bail!("MlaLatentCache::adopt_row_from: src_row {} >= src.batch {}", src_row, src.batch);
    }
    let src_off = src.offsets[src_row];
    if src_off > self.cap {
        anyhow::bail!("MlaLatentCache::adopt_row_from: src.offsets[{}] = {} > self.cap {}", src_row, src_off, self.cap);
    }
    if src_off > 0 {
        let current_capacity = self.c_kv.as_ref().map(|a| a.shape().as_slice()[2]).unwrap_or(0);
        if src_off > current_capacity {
            let target_capacity = ((src_off + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, ().into())?;
        }
        let src_c_kv = src.c_kv.as_ref().ok_or_else(|| anyhow!(
            "MlaLatentCache::adopt_row_from: src has offset {src_off} but c_kv unallocated"))?;
        let src_k_pe = src.k_pe.as_ref().ok_or_else(|| anyhow!(
            "MlaLatentCache::adopt_row_from: src has offset {src_off} but k_pe unallocated"))?;
        let c_kv_slice = slice_strided_on(
            src_c_kv, [src_row as i32, 0, 0, 0],
            [src_row as i32 + 1, 1, src_off, self.kv_lora], [1_i32, 1, 1, 1], (),
        )?;
        let k_pe_slice = slice_strided_on(
            src_k_pe, [src_row as i32, 0, 0, 0],
            [src_row as i32 + 1, 1, src_off, self.rope], [1_i32, 1, 1, 1], (),
        )?;
        let c_kv_full = self.c_kv.as_ref().expect("grow_to allocated c_kv");
        let k_pe_full = self.k_pe.as_ref().expect("grow_to allocated k_pe");
        let new_c_kv = slice_update_on(
            c_kv_full, &c_kv_slice, [dst_row as i32, 0, 0, 0],
            [dst_row as i32 + 1, 1, src_off, self.kv_lora], [1_i32, 1, 1, 1], (),
        )?;
        let new_k_pe = slice_update_on(
            k_pe_full, &k_pe_slice, [dst_row as i32, 0, 0, 0],
            [dst_row as i32 + 1, 1, src_off, self.rope], [1_i32, 1, 1, 1], (),
        )?;
        self.c_kv = Some(new_c_kv);
        self.k_pe = Some(new_k_pe);
    }
    self.offsets[dst_row] = src_off;
    Ok(())
}
```

- [ ] **Step 5: 运行 → PASS** + `cargo build --release -p ironmlx` 干净。

- [ ] **Step 6: Commit** `git commit -m "feat(glm4_moe_lite): MlaLatentCache::adopt_row_from for continuous-batching row migration"`

---

## Task 2: Scheduler wiring（`adopt_cache_row_layers` + dtype-finder Mla 臂；审计 `_ => None`）

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: 加 `(Mla, Mla)` 臂到 `adopt_cache_row_layers`**（`scheduler.rs:467` 的 match，现 `_ => Err("cache layer kind mismatch")`）：
```rust
            (LayerCache::Mla(dst_mla), LayerCache::Mla(src_mla)) => {
                dst_mla.adopt_row_from(src_mla, dst_row, src_row)?;
            }
```

- [ ] **Step 2: 加 Mla 臂到 mid-admit dtype-finder**（`scheduler.rs:2019`，现 `LayerCache::Full(kv) => Some(kv.dtype()), _ => None`）：
```rust
                    LayerCache::Mla(mla) => Some(mla.dtype()),
```

- [ ] **Step 3: 审计 `_ => None`（:1226 / :1347 / :1355）** —— 逐一 `Read` 这三处的上下文，确认它们对 GLM 路径**不会静默误处理 Mla**（若是非 cache-kind 相关的语义，则无碍；若是 offsets/dtype 之类对 GLM decode 必需的，需加 Mla 臂）。在 commit message 或代码注释里记录每处的判定结论。

- [ ] **Step 4: 编译验证** `cargo build --release -p ironmlx` 干净；`cargo +nightly clippy --all-features --workspace -- -D warnings` 干净。（Qwen/Gemma 的 Full/Linear 永不命中新 Mla 臂，行为不变。）

- [ ] **Step 5: Commit** `git commit -m "feat(glm4_moe_lite): wire LayerCache::Mla into scheduler row-adoption + dtype-finder"`

---

## Task 3: 移除 b-max 拦截 + B>1 decode 正确性

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`（移除拦截）；可能 `ironmlx/src/models/glm4_moe_lite/mla_attention.rs`（若 B>1 decode mask 广播需修）
- Test: `ironmlx/tests/glm4_moe_lite_cb.rs`（新，env-gated）

- [ ] **Step 1: 移除 `serve.rs:204-208` 的 `if args.b_max > 1 { return Err(...) }`**（glm4_moe_lite 臂）。`cargo build --release -p ironmlx` 干净。

- [ ] **Step 2: 写 B>1 并发-vs-串行正确性集成测试**（`ironmlx/tests/glm4_moe_lite_cb.rs`，env-gated，跳过若无权重）。逻辑：用调度器以 `b_max=2` 并发跑 2 个不同 prompt（temperature 0，greedy），同时各自以 `b_max=1` 串行跑；断言每个请求的生成 token 序列**逐 token 一致**。
  - 用真实 server/scheduler 路径（grep 既有调度器集成测试如 `b1_p2_*` / `vl_server_smoke` 的驱动方式作模板）。
  - 至少 2 个**不同长度** prompt，触发 per-row 异构 cache 长度的 decode。

- [ ] **Step 3: 运行 → 观察**：`GLM47_MODEL_DIR=… cargo test -p ironmlx --test glm4_moe_lite_cb -- --nocapture 2>&1 | tail -40`。
  - 若一致 → B>1 decode 已正确，进 Step 5。
  - 若不一致 → 用 systematic-debugging 定位（首疑：MlaAttention 把引擎 per-row decode mask `[B,1,1,Lc]` 折进 `pe_scores [B,H,1,Lc]` 时的广播；其次 rope array-offset；其次 §2.2 的 `_ => None`）。Step 4 修复。

- [ ] **Step 4: （若需要）修 B>1 decode** —— 按 Step 3 定位的根因最小修复（例如 mask 广播）；不放宽正确性。重跑 Step 3 直到逐 token 一致。

- [ ] **Step 5: 不回归 b-max=1** `cargo test -p ironmlx --lib glm4_moe_lite 2>&1 | tail` + `GLM47_MODEL_DIR=… cargo test -p ironmlx --test glm4_moe_lite_smoke --test glm4_moe_lite_parity 2>&1 | tail`（均 PASS）。

- [ ] **Step 6: Commit** `git commit -m "feat(glm4_moe_lite): enable --b-max>1; verify B>1 decode correctness"`

---

## Task 4: 连续批处理 e2e（mid-admit / 行复用）+ 性能 vs omlx

**Files:**
- Test: 扩展 `ironmlx/tests/glm4_moe_lite_cb.rs`
- Perf 结果 → 工作报告（reports/，gitignored，不提交）

- [ ] **Step 1: mid-admit / 行复用集成测试** —— 在 `glm4_moe_lite_cb.rs` 加一个场景：交错入队不同长度的请求（一个长 prompt 先入、解码途中再admit一个短 prompt），并让一个请求先完成、其槽位被新请求复用（触发 `adopt_cache_row_layers` + `rebuild_cache_layout` 的 Mla 路径）。断言所有请求输出与各自串行 b-max=1 结果逐 token 一致。运行 → PASS（若失败，systematic-debugging 定位 mid-admit/压缩路径的 Mla 处理）。

- [ ] **Step 2: 确认 omlx 是否支持 glm4_moe_lite 连续批处理** —— 从 `/Users/xin/workspace/iron-rivals/omlx` 查其 server 是否做并发/连续批处理（grep server/batch；`uv run` 起服务发并发请求观察）。
  - 支持 → 作为 perf 基线。
  - 不支持 → 记录；perf 回退为"聚合吞吐随 b-max 扩展"（报告 ironmlx b-max=1 vs b-max=N 的聚合 tok/s），并知会 Boss "追平 omlx" 基线不成立。

- [ ] **Step 3: 性能测量（iron-bench，串行跑，避免互染）** —— 并发负载下 ironmlx 连续批处理聚合吞吐 vs omlx（或 vs 自身 b-max=1）。**验收：ironmlx 聚合吞吐 ≥ omlx**（若 omlx 不支持，则 ≥ b-max=1 且随 b-max 合理扩展）。结果写入 reports/（不提交）。

- [ ] **Step 4: 收口** —— `cargo +nightly fmt --all` + `clippy --all-features --workspace -D warnings` + `cargo build --release` 全过；`cargo test --all-features --workspace` 绿（除已知 quant 项）。Commit 测试代码：`git commit -m "test(glm4_moe_lite): continuous-batching mid-admit + row-reuse correctness"`。perf 数字 + omlx-cb 判定写报告。

---

## Self-Review（plan 作者）
- **Spec 覆盖**：§1.1 adopt_row_from→T1；§1.2/1.3 scheduler 臂→T2；§1.4 b-max 拦截移除→T3.1；§2 B>1 decode 验证→T3.2-4 + T2.3 审计；§3 验收→T3(正确性)+T4(perf/omlx 前置)；§4 测试→T1(单元)/T3(并发=串行)/T4(mid-admit+perf)；§5 风险→T4.2(omlx 前置)/T1+T3(mask/adopt)/T4.1(mid-admit)。
- **占位扫描**：T1/T2 含完整代码；T3.2 测试给出逻辑 + 模板来源（既有调度器集成测试）而非具体 server 驱动样板（因 server 驱动 API 需 implementer 照既有测试照搬）——这是有意"照镜像既有测试"指引，非 vague TODO；T3.4 是条件修复（"若需要"），根因由 T3.3 定位后才知具体改动，故不预写投机代码。T4 perf 步骤是测量/判定，无代码占位。
- **类型一致**：`MlaLatentCache::adopt_row_from(&mut self, src:&MlaLatentCache, dst_row:usize, src_row:usize)->Result<()>`（T1）被 `adopt_cache_row_layers` 的 Mla 臂（T2）调用，签名一致；`dtype()` 已存在（T2.2 用）。
