# P8a Stage 9 — Self-Quant Matmul Metal Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ironmlx 自研 4-bit (MLX affine quantization, group_size=64) Metal matmul kernel，prefill 路径 opt-in 替换 mlx::quantized_matmul。验证假设：通过 device-aware + quant-aware tile selection 在 M1 Pro 上能超 mlx 4-bit baseline (281 tok/s @ PP=2048)，理想超 llama.cpp Q4_K_M。

**Architecture:** opt-in env var (`IRONMLX_USE_SELF_QMM=1`) 路由到新 `nn::self_qmm` 模块；env=0 走 stage 8 的 mlx 默认路径不变。模块三层：`lookup`(device+shape→tile) + `kernel`(PSO cache) + `metal/qmm_t.metal`(MSL 源码 templated by function constants)。3 个 hardcoded tile candidates: (64,64,32) (64,128,32) (128,128,32)。Stage 9 第一个 task 是 fusion barrier 预先验证（echo kernel），通过后才实现 quant kernel。

**Tech Stack:** Rust + cxx + mlx-sys (`mx::fast::metal_kernel` API)、Metal Shading Language、cargo workspace、iron-bench HTTP harness、新 `ironmlx-bench-kernel` crate。

**Spec reference:** [docs/superpowers/specs/2026-05-09-p8a-stage9-quant-kernel-design.md](../specs/2026-05-09-p8a-stage9-quant-kernel-design.md) (commit `8780675`).

---

## File Structure

Files this plan creates / modifies:

**新增**：

```
ironmlx/src/nn/self_qmm/
├── mod.rs              # 入口 + env var 检测 + qmm_t_on() 接口
├── lookup.rs           # (device, M, N, K, bits, group_size) → (BM, BN, BK)
├── kernel.rs           # MetalKernel builder + thread_local PSO cache
└── metal/
    └── qmm_t.metal.in  # MSL 源码（被 build.rs include 进 Rust 字符串常量）

ironmlx/src/nn/echo_kernel.rs   # Task 1 only - fusion barrier 预先验证
ironmlx/tests/p8a_stage9_self_qmm_logits_match.rs  # 数值正确性 integration test

ironmlx-bench-kernel/   # 新 workspace crate
├── Cargo.toml
└── src/
    └── main.rs         # CLI: --M --N --K --BM --BN --BK --bits --group-size --runs
```

**修改**：

```
ironmlx/Cargo.toml        # 加 self_qmm 模块 + echo_kernel 模块
ironmlx/src/nn/mod.rs     # pub mod self_qmm + (Task 1) pub mod echo_kernel
ironmlx/src/nn/linear.rs  # forward_on() 加 env var dispatch 分支
ironmlx/src/nn/gated_delta_net.rs  # forward_on() 加 env var dispatch 分支
Cargo.toml (workspace)    # 加 ironmlx-bench-kernel workspace member
README.md                 # 加 IRONMLX_USE_SELF_QMM env var 说明
```

---

## Task 1: Fusion Barrier 预先验证（Echo Kernel）

**目的：** 在实现真正的 quant kernel 前，验证 `mx::fast::metal_kernel` 注入到 forward path 是否引入 fusion barrier 退化。如果 echo kernel（输入直接输出）让 prefill 退化 > 5%，重选 kernel 注入机制（fork mlx / cxx 桥接 metal）。

**Files:**

- Create: `ironmlx/src/nn/echo_kernel.rs`
- Modify: `ironmlx/src/nn/mod.rs` (加 `pub mod echo_kernel;`)
- Modify: `ironmlx/src/nn/linear.rs` (forward_on 加 echo dispatch 分支)

### Steps

- [ ] **Step 1.1: 创建 echo_kernel.rs 骨架**

```rust
// ironmlx/src/nn/echo_kernel.rs
//! P8a stage 9 task 1 — fusion barrier pre-verification.
//!
//! Inserts an "echo" Metal kernel (input → output, no compute) into the
//! forward path when `IRONMLX_ECHO_KERNEL=1` is set. Used to measure
//! whether `mx::fast::metal_kernel` injection itself introduces fusion
//! barrier overhead, before committing to the real self_qmm kernel.

use std::sync::OnceLock;

use mlx::{Array, MetalKernel};

use crate::Result;

/// Returns true iff `IRONMLX_ECHO_KERNEL=1` env var is set.
pub fn echo_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("IRONMLX_ECHO_KERNEL").as_deref() == Ok("1"))
}

/// Lazy MetalKernel — built once per thread.
fn echo_kernel() -> Result<MetalKernel> {
    let src = r#"
        uint idx = thread_position_in_grid.x;
        if (idx >= total) { return; }
        out[idx] = x[idx];
    "#;
    Ok(MetalKernel::builder("ironmlx_echo")
        .inputs(&["x"])
        .outputs(&["out"])
        .source(src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

/// Pass `x` through an echo Metal kernel. Output is byte-identical to input.
pub fn echo(x: &Array) -> Result<Array> {
    let total = x.size() as i32;
    let kernel = echo_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x])
        .output_shapes(&[x.shape().clone()])
        .output_dtypes(&[x.dtype()])
        .grid(total, 1, 1)
        .threadgroup(256, 1, 1)
        .template_int("total", total)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}
```

- [ ] **Step 1.2: 注册模块**

```rust
// ironmlx/src/nn/mod.rs — 加这一行（插入到现有 pub mod 列表中）
pub mod echo_kernel;
```

- [ ] **Step 1.3: 编译验证 echo kernel**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx
```

期望：编译通过（warnings 容忍），无 error。

- [ ] **Step 1.4: 写 echo 单元测试**

```rust
// 加到 ironmlx/src/nn/echo_kernel.rs 末尾
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn echo_preserves_input() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x: Array = (data.as_slice(), (4_i32, 4)).try_into().unwrap();
        let y = echo(&x).unwrap();
        let yv: Vec<f32> = y.to_vec().unwrap();
        for (i, (a, b)) in data.iter().zip(yv.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
        assert_eq!(y.shape().as_slice(), x.shape().as_slice());
        assert_eq!(y.dtype(), Dtype::Float32);
    }
}
```

- [ ] **Step 1.5: 跑 echo 单元测试**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo test --release -p ironmlx --lib -- nn::echo_kernel:: 2>&1 | tail -10
```

期望：`test result: ok. 1 passed`

- [ ] **Step 1.6: 集成到 Linear::forward_on**

修改 `ironmlx/src/nn/linear.rs` 的 `forward_on` 函数。在 quantized matmul 调用**之后**加 echo passthrough（仅当 env var 启用时），以模拟"插入 metal_kernel 到 forward path"的最小 overhead 场景：

```rust
// 在 ironmlx/src/nn/linear.rs forward_on 现有 quantized_matmul_on 调用结束后，加：
let result = /* existing quantized_matmul_on call */;
if crate::nn::echo_kernel::echo_enabled() {
    crate::nn::echo_kernel::echo(&result)
} else {
    Ok(result)
}
```

具体修改位置：`linear.rs:195` 附近的 `mlx::quantization::quantized_matmul_on()` 调用结束后。把 result 通过 echo passthrough。

- [ ] **Step 1.7: 重新编译 + 跑现有 norm/gdn 测试确认无 regression**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx 2>&1 | tail -5
cargo test --release -p ironmlx --lib -- nn::linear:: 2>&1 | tail -10
```

期望：build 通过，linear unit tests pass。

- [ ] **Step 1.8: iron-bench 跑 echo OFF baseline (5 runs PP=2048)**

确保 ironmlx server 跑 echo OFF 时性能跟 stage 8 commit 811dd36 一致：

```bash
SHA=32f3e8ecf65426fc3306969496342d504bfa13f3
IRONMLX_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$SHA

pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
/Volumes/Dev/cxx-mlx/target/release/ironmlx serve --model "$IRONMLX_MODEL" --port 8080 > /tmp/echo-off.log 2>&1 &
until curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; do sleep 1; done

/Volumes/Dev/cxx-mlx/target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:8080 \
  --model "$IRONMLX_MODEL" --model-dir "$IRONMLX_MODEL" \
  --prompt-len 2048 --max-tokens 64 --runs 5 --warmup 1 --format markdown \
  | tee /tmp/exp-echo-off.md
pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
```

记录: Prefill PP @ PP=2048 = ____ tok/s（预期约 280-281）。

- [ ] **Step 1.9: iron-bench 跑 echo ON 测试**

```bash
SHA=32f3e8ecf65426fc3306969496342d504bfa13f3
IRONMLX_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$SHA

pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
IRONMLX_ECHO_KERNEL=1 /Volumes/Dev/cxx-mlx/target/release/ironmlx serve \
  --model "$IRONMLX_MODEL" --port 8080 > /tmp/echo-on.log 2>&1 &
until curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; do sleep 1; done

/Volumes/Dev/cxx-mlx/target/release/iron-bench \
  --target ironmlx-echo=http://127.0.0.1:8080 \
  --model "$IRONMLX_MODEL" --model-dir "$IRONMLX_MODEL" \
  --prompt-len 2048 --max-tokens 64 --runs 5 --warmup 1 --format markdown \
  | tee /tmp/exp-echo-on.md
pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
```

记录: Prefill PP @ PP=2048 = ____ tok/s。

- [ ] **Step 1.10: 决策门 — fusion barrier 评估**

计算: `regression = (echo_off - echo_on) / echo_off`

- 如果 `regression < 5%`：**通过**，继续 task 2
- 如果 `regression >= 5%`：**暂停 stage 9**，回到 brainstorming 重选 kernel 注入机制（候选：fork mlx / cxx 桥接 metal）

记录数据 + 决策到 commit message。

- [ ] **Step 1.11: Commit task 1**

```bash
git add ironmlx/src/nn/echo_kernel.rs ironmlx/src/nn/mod.rs ironmlx/src/nn/linear.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): echo kernel for fusion barrier pre-verification

Add ironmlx::nn::echo_kernel — minimal mx::fast::metal_kernel that
passes input through unchanged. Linear::forward_on routes through it
when IRONMLX_ECHO_KERNEL=1 is set, otherwise no-op.

Purpose: measure fusion barrier overhead of mx::fast::metal_kernel
injection BEFORE committing to the real self_qmm kernel
implementation.

iron-bench PP=2048 × 5 runs:
  echo OFF: ____ tok/s prefill
  echo ON:  ____ tok/s prefill
  regression: ____%

Decision gate: <5% regression → continue stage 9 task 2.
                >=5% → stop and re-select kernel injection mechanism.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

填入实测数据后 commit。

---

## Task 2: Self-Qmm Metal Kernel — 写死 (64,64,32) + 数值正确性 unit test

**目的：** 实现最小可用的 self_qmm Metal kernel（先固定 BM=64, BN=64, BK=32 写死，不参数化），通过小 shape unit test 验证数值正确性。

**Files:**

- Create: `ironmlx/src/nn/self_qmm/mod.rs`
- Create: `ironmlx/src/nn/self_qmm/kernel.rs`
- Create: `ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in`
- Modify: `ironmlx/src/nn/mod.rs` (加 `pub mod self_qmm;`)

### Steps

- [ ] **Step 2.1: 创建模块目录 + mod.rs 骨架**

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/src/nn/self_qmm/metal
```

```rust
// ironmlx/src/nn/self_qmm/mod.rs
//! Self-implemented quantized matmul Metal kernel for MLX 4-bit affine
//! quantization (group_size=64). Opt-in via IRONMLX_USE_SELF_QMM=1.
//!
//! Stage 9: prefill (qmm_t) only, M1 Pro tuned. See
//! docs/superpowers/specs/2026-05-09-p8a-stage9-quant-kernel-design.md.

mod kernel;

use std::sync::OnceLock;

use mlx::{Array, StreamOrDevice};

use crate::Result;

/// Returns true iff `IRONMLX_USE_SELF_QMM=1` env var is set.
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("IRONMLX_USE_SELF_QMM").as_deref() == Ok("1"))
}

/// 4-bit MLX-affine quantized matmul: `x @ w^T` with per-group scales+biases.
///
/// Inputs:
/// - `x`: bf16 `[B, S, K]` (last dim contiguous)
/// - `w`: packed uint32 `[N, K/8]` (8 4-bit weights per uint32)
/// - `scales`: bf16 `[N, K/group_size]`
/// - `biases`: bf16 `[N, K/group_size]`
/// - `bits`: must be 4 (other values panic)
/// - `group_size`: must be 64 (other values panic)
///
/// Output: bf16 `[B, S, N]`
pub fn qmm_t_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bits: i32,
    group_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    assert_eq!(bits, 4, "self_qmm stage 9 only supports bits=4");
    assert_eq!(group_size, 64, "self_qmm stage 9 only supports group_size=64");
    let _ = target.into(); // accepted but unused at this stage
    kernel::dispatch_qmm_t(x, w, scales, biases)
}
```

- [ ] **Step 2.2: 写 metal source 骨架**

```metal
// ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in
//
// Self-quant matmul kernel (MLX 4-bit affine, group_size=64) — qmm_t form
// (x @ w^T). Stage 9 starting tile: BM=64 BN=64 BK=32 hardcoded.
//
// Inputs (auto-injected shape buffers by mx::fast::metal_kernel):
//   x:       device const half* (bf16) shape [M, K]   (M = B*S after flatten)
//   w:       device const uint32_t*    shape [N, K/8] (4-bit packed)
//   scales:  device const half*        shape [N, K/64]
//   biases:  device const half*        shape [N, K/64]
//
// Output:
//   out:     device half*              shape [M, N]
//
// Constants passed via .template_int():
//   M, N, K (matmul shape)

constant int BM = 64;
constant int BN = 64;
constant int BK = 32;
constant int GROUP_SIZE = 64;

// Stage 9 task 2 starting kernel:
// - one threadgroup processes a [BM, BN] output tile
// - K-loop in steps of BK
// - inline 4-bit dequant: each uint32 packs 8 values, group_size=64 means
//   one (scale, bias) pair covers 64 K-elements (= 2 BK iterations)

kernel void self_qmm_t(
    device const half*    x       [[buffer(0)]],
    device const uint32_t* w      [[buffer(1)]],
    device const half*    scales  [[buffer(2)]],
    device const half*    biases  [[buffer(3)]],
    device       half*    out     [[buffer(4)]],
    constant     int&     M       [[buffer(5)]],
    constant     int&     N       [[buffer(6)]],
    constant     int&     K       [[buffer(7)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    // Tile origin in output:
    const int tile_m = tid.y * BM;
    const int tile_n = tid.x * BN;

    // Allocate threadgroup memory:
    threadgroup half xs[BM * BK];   // [BM=64, BK=32]
    threadgroup half ws[BN * BK];   // [BN=64, BK=32] (post-dequant)

    // Per-thread accumulator: each thread covers a (4 row × 4 col) micro-tile
    // of output. Total threads in TG = (BM/4) * (BN/4) = 16*16 = 256.
    half4x4 acc;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.0h;

    const int row_in_tile = (lid / (BN/4)) * 4;
    const int col_in_tile = (lid % (BN/4)) * 4;

    // K loop:
    for (int k0 = 0; k0 < K; k0 += BK) {

        // 1. Cooperative load Xs from x[tile_m..+BM, k0..+BK]:
        //    256 threads / (BM*BK = 2048 elems) = 8 elems per thread
        for (int t = 0; t < 8; ++t) {
            int idx = lid * 8 + t;
            int r = idx / BK;
            int c = idx % BK;
            int gx = (tile_m + r) * K + (k0 + c);
            xs[r * BK + c] = (tile_m + r < M) ? x[gx] : 0.0h;
        }

        // 2. Cooperative dequant + load Ws from w[tile_n..+BN, k0..+BK]:
        //    Each w element: 4 bits, packed 8 per uint32. group_size=64.
        //    For BK=32, each thread loads 8 K-elements covering 1 uint32 each
        //    (with 4 bytes shifted to extract 4-bit values).
        //    256 threads × (BN*BK = 2048 dequant ops total) = 8 ops per thread
        const int g = k0 / GROUP_SIZE;        // group index in K dim
        for (int t = 0; t < 8; ++t) {
            int idx = lid * 8 + t;
            int n = idx / BK;
            int kk = idx % BK;
            int actual_n = tile_n + n;
            if (actual_n >= N) {
                ws[n * BK + kk] = 0.0h;
                continue;
            }
            // w[n][k] address: w_packed = w[actual_n * (K/8) + (k0+kk)/8]
            int packed_idx = actual_n * (K / 8) + (k0 + kk) / 8;
            uint32_t pack = w[packed_idx];
            int shift = ((k0 + kk) % 8) * 4;
            half nibble = (half)((pack >> shift) & 0xF);
            // Group-aware scale + bias:
            int sb_idx = actual_n * (K / GROUP_SIZE) + g;
            half scale = scales[sb_idx];
            half bias  = biases[sb_idx];
            ws[n * BK + kk] = nibble * scale + bias;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. MMA: each thread computes its 4x4 output micro-tile from
        //    Xs[row..row+4, :BK] @ Ws[col..col+4, :BK]^T  (Ws is row-major
        //    so we access ws[n][kk] then 'transpose' implicitly via index).
        for (int kk = 0; kk < BK; ++kk) {
            half4 xv;
            for (int i = 0; i < 4; ++i) xv[i] = xs[(row_in_tile + i) * BK + kk];
            half4 wv;
            for (int j = 0; j < 4; ++j) wv[j] = ws[(col_in_tile + j) * BK + kk];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += xv[i] * wv[j];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Store result:
    for (int i = 0; i < 4; ++i) {
        int gr = tile_m + row_in_tile + i;
        if (gr >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int gc = tile_n + col_in_tile + j;
            if (gc >= N) continue;
            out[gr * N + gc] = acc[i][j];
        }
    }
}
```

注意：这是 stage 9 起步版本 — 简单实现（thread-level 4×4 micro-tile，无 simdgroup MMA）。性能不会最优，但**正确性优先**。Task 3 替换为 simdgroup MMA 版本时再优化。

- [ ] **Step 2.3: 写 kernel.rs (Rust 端 dispatch)**

```rust
// ironmlx/src/nn/self_qmm/kernel.rs
//! mx::fast::metal_kernel builder + dispatch for self_qmm_t.
//!
//! Stage 9 task 2: hardcoded BM=64 BN=64 BK=32 single kernel variant.
//! Task 3 generalizes via function constants.

use std::sync::OnceLock;

use mlx::{Array, MetalKernel};

use crate::Result;

const QMM_T_SOURCE: &str = include_str!("metal/qmm_t.metal.in");

fn build_kernel() -> Result<MetalKernel> {
    Ok(MetalKernel::builder("ironmlx_self_qmm_t")
        .inputs(&["x", "w", "scales", "biases"])
        .outputs(&["out"])
        .source(QMM_T_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

fn cached_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if CELL.get().is_none() {
        let _ = CELL.set(build_kernel()?);
    }
    Ok(CELL.get().expect("cached_kernel set above"))
}

pub fn dispatch_qmm_t(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
) -> Result<Array> {
    // x shape: [B, S, K] or [M, K]; flatten to [M, K] for kernel.
    let x_dims = x.shape();
    let x_dims_slice = x_dims.as_slice();
    let k = *x_dims_slice.last().expect("x must be at least 1-D");
    let m: i32 = x_dims_slice[..x_dims_slice.len() - 1].iter().product();

    let w_dims = w.shape();
    let n = w_dims.as_slice()[0]; // w shape [N, K/8]

    // Output shape: [..x.shape[:-1], N]
    let mut out_shape: Vec<i32> = x_dims_slice[..x_dims_slice.len() - 1].to_vec();
    out_shape.push(n);

    let kernel = cached_kernel()?;

    // Threadgroup config: 256 threads per TG (16×16 micro-tiles).
    // Grid: (ceil(N/BN), ceil(M/BM), 1)
    const BM: i32 = 64;
    const BN: i32 = 64;
    let grid_x = (n + BN - 1) / BN;
    let grid_y = (m + BM - 1) / BM;

    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, w, scales, biases])
        .output_shapes(&[mlx::Shape::from(out_shape)])
        .output_dtypes(&[x.dtype()])
        .grid(grid_x, grid_y, 1)
        .threadgroup(256, 1, 1)
        .template_int("M", m)
        .template_int("N", n)
        .template_int("K", k)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}
```

- [ ] **Step 2.4: 注册 self_qmm 模块**

```rust
// ironmlx/src/nn/mod.rs — 加这一行
pub mod self_qmm;
```

- [ ] **Step 2.5: 编译验证（可能需要修 metal source 编译错误）**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx 2>&1 | tail -20
```

期望：build 通过。如果 metal source 有语法错误，按 metal 编译 error 信息修。

- [ ] **Step 2.6: 写 small-shape unit test**

```rust
// 加到 ironmlx/src/nn/self_qmm/mod.rs 末尾
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Dtype, ops};

    /// Tiny shape unit test — compares self_qmm against mlx::quantized_matmul.
    #[test]
    fn self_qmm_t_matches_mlx_small_shape() {
        // Shape: M=4, K=64 (one group), N=8 (BN=64 means N=8 is partial tile)
        let m = 4_i32;
        let k = 64_i32;
        let n = 8_i32;
        let group_size = 64_i32;
        let bits = 4_i32;

        // Generate random fp32 weights, quantize via mlx::ops::quantize:
        let raw_w = ops::random::uniform_distribution::<f32>(
            -0.5, 0.5, &[n, k][..], None,
        ).unwrap();
        let raw_w_bf16 = ops::cast::astype(&raw_w, Dtype::Bfloat16).unwrap();
        let (w_packed, w_scales, w_biases) =
            mlx::quantization::quantize(&raw_w_bf16, group_size, bits).unwrap();

        // Generate input x:
        let x_data: Vec<f32> = (0..(m * k) as usize).map(|i| (i as f32) * 0.01).collect();
        let x_f32: Array = (x_data.as_slice(), (m, k)).try_into().unwrap();
        let x = ops::cast::astype(&x_f32, Dtype::Bfloat16).unwrap();

        // Self-qmm output:
        let y_self = qmm_t_on(&x, &w_packed, &w_scales, &w_biases, bits, group_size, ()).unwrap();

        // mlx baseline output:
        let y_mlx = mlx::quantization::quantized_matmul_on(
            &x, &w_packed, &w_scales, Some(&w_biases),
            /* transpose */ true, group_size, bits, "affine", (),
        ).unwrap();

        assert_eq!(y_self.shape().as_slice(), y_mlx.shape().as_slice());

        let y_self_f32 = ops::cast::astype(&y_self, Dtype::Float32).unwrap();
        let y_mlx_f32 = ops::cast::astype(&y_mlx, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_self_f32.to_vec().unwrap();
        let mv: Vec<f32> = y_mlx_f32.to_vec().unwrap();

        assert_eq!(yv.len(), mv.len());
        let max_diff = yv.iter().zip(mv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 0.5,
            "self_qmm vs mlx max abs diff {max_diff} > 0.5 (kernel correctness bug)"
        );
    }
}
```

注意 import path：`mlx::quantization::quantized_matmul_on` 函数签名以 ironmlx 现有调用为准（看 `nn/linear.rs:195` 的实际签名调整）。

- [ ] **Step 2.7: 跑 unit test**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo test --release -p ironmlx --lib -- nn::self_qmm:: 2>&1 | tail -20
```

期望：
- 通过：`test result: ok. 1 passed`
- 不通过：max_diff 输出值 → debug kernel（reduction 顺序、index 边界、dequant 公式）

调试方法：
- 先把 kernel 的 unpack 部分单独验证（写一个 kernel 输出 dequanted weights matrix，跟 mlx::quantization::dequantize 对比）
- 再加 matmul 累加部分

- [ ] **Step 2.8: Commit task 2**

```bash
git add ironmlx/src/nn/self_qmm/ ironmlx/src/nn/mod.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): self_qmm Metal kernel — single-tile (64,64,32) baseline

Initial nn::self_qmm module with hardcoded BM=64, BN=64, BK=32 tile.
4-bit MLX-affine quantization (group_size=64) qmm_t kernel (x @ w^T).

Implementation: thread-level 4×4 micro-tile per thread, 256 threads per
threadgroup, inline 4-bit unpack + dequant. Simple correctness-first
implementation; performance optimization deferred to task 3 (simdgroup
MMA + multi-variant tile).

Unit test (M=4, K=64, N=8) verifies max abs diff < 0.5 vs
mlx::quantization::quantized_matmul_on. End-to-end perf testing
deferred to task 8.

Module not yet integrated into Linear::forward_on — that lands in
task 5 (env var dispatch). This task only validates kernel correctness
in isolation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 推广到 3 Tile Variants — Function Constants 参数化

**目的：** 把 BM/BN/BK 通过 Metal function constants 暴露，编译期生成 3 个 PSO variants。每个 variant 跑同样的 unit test。

**Files:**

- Modify: `ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in` (BM/BN/BK 改成 function constants)
- Modify: `ironmlx/src/nn/self_qmm/kernel.rs` (cache 改 HashMap，按 (BM,BN,BK) 索引)
- Modify: `ironmlx/src/nn/self_qmm/mod.rs` (test 扩展跑 3 variants)

### Steps

- [ ] **Step 3.1: metal source 改 function constants**

```metal
// ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in (顶部 constant 块替换)
constant int BM [[function_constant(0)]];
constant int BN [[function_constant(1)]];
constant int BK [[function_constant(2)]];
constant int GROUP_SIZE = 64;
```

注意：原来的 `acc` 是 `half4x4` (固定 4×4) 假定 BM=BN=64 且 16×16 thread layout。BM/BN 变化时 micro-tile 大小要相应变化。简化方案：**保持 thread-per-output-element 模式**（每个 thread 算 1 个 output），而非 4×4 micro-tile。重写 kernel：

```metal
// 完整重写 kernel 主体（替换 step 2.2 的整个 kernel）
kernel void self_qmm_t(
    device const half*    x       [[buffer(0)]],
    device const uint32_t* w      [[buffer(1)]],
    device const half*    scales  [[buffer(2)]],
    device const half*    biases  [[buffer(3)]],
    device       half*    out     [[buffer(4)]],
    constant     int&     M       [[buffer(5)]],
    constant     int&     N       [[buffer(6)]],
    constant     int&     K       [[buffer(7)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint3 ltid [[thread_position_in_threadgroup]]
) {
    const int tile_m = tid.y * BM;
    const int tile_n = tid.x * BN;

    // Threadgroup memory:
    threadgroup half xs[64 * 32];   // BM_max=64, BK_max=32 — adjust if needed for larger BM/BN
    threadgroup half ws[128 * 32];  // BN_max=128, BK_max=32

    // Each thread computes ONE output element at (tile_m + ltid.y, tile_n + ltid.x).
    // ltid.x in [0, BN), ltid.y in [0, BM). Threads per TG = BM × BN.
    const int row = ltid.y;
    const int col = ltid.x;

    half acc = 0.0h;

    for (int k0 = 0; k0 < K; k0 += BK) {

        // Cooperative load Xs[BM, BK] from x[tile_m..+BM, k0..+BK]:
        // Total elements = BM*BK; threads = BM*BN; each thread loads BM*BK / (BM*BN) = BK/BN elems
        // For (64,64,32) BK/BN = 0.5 (one elem per 2 threads).
        // Simpler: only first BM*BK threads each load 1 elem.
        const int n_load_x = BM * BK;
        if (lid < n_load_x) {
            int r = lid / BK;
            int c = lid % BK;
            int gx = (tile_m + r) * K + (k0 + c);
            xs[r * BK + c] = (tile_m + r < M && (k0 + c) < K) ? x[gx] : 0.0h;
        }

        // Cooperative dequant + load Ws[BN, BK]:
        const int n_load_w = BN * BK;
        if (lid < n_load_w) {
            int n_idx = lid / BK;
            int kk = lid % BK;
            int actual_n = tile_n + n_idx;
            if (actual_n >= N || (k0 + kk) >= K) {
                ws[n_idx * BK + kk] = 0.0h;
            } else {
                int packed_idx = actual_n * (K / 8) + (k0 + kk) / 8;
                uint32_t pack = w[packed_idx];
                int shift = ((k0 + kk) % 8) * 4;
                half nibble = (half)((pack >> shift) & 0xF);
                int g = (k0 + kk) / GROUP_SIZE;
                int sb_idx = actual_n * (K / GROUP_SIZE) + g;
                ws[n_idx * BK + kk] = nibble * scales[sb_idx] + biases[sb_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: this thread computes acc += xs[row, :BK] · ws[col, :BK]
        for (int kk = 0; kk < BK; ++kk) {
            acc += xs[row * BK + kk] * ws[col * BK + kk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store:
    int gr = tile_m + row;
    int gc = tile_n + col;
    if (gr < M && gc < N) {
        out[gr * N + gc] = acc;
    }
}
```

注意：threadgroup memory 大小用静态最大值 `xs[64*32]` `ws[128*32]` 覆盖所有 candidate tiles。后续 task 7 sweep 后如果某 variant 占用大于此，再扩。

- [ ] **Step 3.2: kernel.rs 改 HashMap 缓存**

```rust
// ironmlx/src/nn/self_qmm/kernel.rs (整体替换)
use std::collections::HashMap;
use std::sync::Mutex;

use mlx::{Array, MetalKernel};

use crate::Result;

const QMM_T_SOURCE: &str = include_str!("metal/qmm_t.metal.in");

/// Cache of compiled MetalKernel per (BM, BN, BK) tuple. PSO compile
/// happens on first dispatch with a given tile; subsequent calls reuse.
static KERNEL_CACHE: std::sync::OnceLock<Mutex<HashMap<(i32, i32, i32), MetalKernel>>> =
    std::sync::OnceLock::new();

fn cache() -> &'static Mutex<HashMap<(i32, i32, i32), MetalKernel>> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn build_kernel(bm: i32, bn: i32, bk: i32) -> Result<MetalKernel> {
    let name = format!("ironmlx_self_qmm_t_{bm}x{bn}x{bk}");
    Ok(MetalKernel::builder(&name)
        .inputs(&["x", "w", "scales", "biases"])
        .outputs(&["out"])
        .source(QMM_T_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

fn get_or_build(bm: i32, bn: i32, bk: i32) -> Result<MetalKernel> {
    let mut g = cache().lock().expect("KERNEL_CACHE mutex poisoned");
    if let Some(k) = g.get(&(bm, bn, bk)) {
        return Ok(k.clone());
    }
    let k = build_kernel(bm, bn, bk)?;
    g.insert((bm, bn, bk), k.clone());
    Ok(k)
}

pub fn dispatch_qmm_t(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bm: i32,
    bn: i32,
    bk: i32,
) -> Result<Array> {
    let x_dims = x.shape();
    let x_dims_slice = x_dims.as_slice();
    let k = *x_dims_slice.last().expect("x must be at least 1-D");
    let m: i32 = x_dims_slice[..x_dims_slice.len() - 1].iter().product();

    let w_dims = w.shape();
    let n = w_dims.as_slice()[0];

    let mut out_shape: Vec<i32> = x_dims_slice[..x_dims_slice.len() - 1].to_vec();
    out_shape.push(n);

    let kernel = get_or_build(bm, bn, bk)?;

    let grid_x = (n + bn - 1) / bn;
    let grid_y = (m + bm - 1) / bm;

    // Threads per TG = bm × bn (each thread = one output element)
    let threads_per_tg = bm * bn;
    assert!(
        threads_per_tg <= 1024,
        "TG threads {threads_per_tg} > 1024 Metal limit (bm={bm} bn={bn})"
    );

    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, w, scales, biases])
        .output_shapes(&[mlx::Shape::from(out_shape)])
        .output_dtypes(&[x.dtype()])
        .grid(grid_x, grid_y, 1)
        .threadgroup(bn, bm, 1)
        .template_int("BM", bm)
        .template_int("BN", bn)
        .template_int("BK", bk)
        .template_int("M", m)
        .template_int("N", n)
        .template_int("K", k)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}
```

注意：`threadgroup(bn, bm, 1)` — Metal 中 threadgroup_size = (x, y, z)，第一个维度对应 ltid.x（即 col / N 维度）。

- [ ] **Step 3.3: mod.rs 接口加 tile 参数**

```rust
// ironmlx/src/nn/self_qmm/mod.rs 修改 qmm_t_on 签名（暂时硬编码 (64,64,32)，task 4 引入 lookup）
pub fn qmm_t_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bits: i32,
    group_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    assert_eq!(bits, 4, "self_qmm stage 9 only supports bits=4");
    assert_eq!(group_size, 64, "self_qmm stage 9 only supports group_size=64");
    let _ = target.into();
    // Task 3 hardcoded — task 4 introduces lookup_tile
    kernel::dispatch_qmm_t(x, w, scales, biases, 64, 64, 32)
}
```

- [ ] **Step 3.4: 修改 unit test 跑 3 variants**

```rust
// ironmlx/src/nn/self_qmm/mod.rs tests 模块替换
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Dtype, ops};

    fn run_variant(bm: i32, bn: i32, bk: i32) {
        let m = 4_i32;
        let k = 64_i32;
        let n = 8_i32;
        let group_size = 64_i32;
        let bits = 4_i32;

        let raw_w = ops::random::uniform_distribution::<f32>(
            -0.5, 0.5, &[n, k][..], None,
        ).unwrap();
        let raw_w_bf16 = ops::cast::astype(&raw_w, Dtype::Bfloat16).unwrap();
        let (w_packed, w_scales, w_biases) =
            mlx::quantization::quantize(&raw_w_bf16, group_size, bits).unwrap();

        let x_data: Vec<f32> = (0..(m * k) as usize).map(|i| (i as f32) * 0.01).collect();
        let x_f32: Array = (x_data.as_slice(), (m, k)).try_into().unwrap();
        let x = ops::cast::astype(&x_f32, Dtype::Bfloat16).unwrap();

        let y_self = kernel::dispatch_qmm_t(&x, &w_packed, &w_scales, &w_biases, bm, bn, bk).unwrap();
        let y_mlx = mlx::quantization::quantized_matmul_on(
            &x, &w_packed, &w_scales, Some(&w_biases),
            /* transpose */ true, group_size, bits, "affine", (),
        ).unwrap();

        let y_self_f32 = ops::cast::astype(&y_self, Dtype::Float32).unwrap();
        let y_mlx_f32 = ops::cast::astype(&y_mlx, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_self_f32.to_vec().unwrap();
        let mv: Vec<f32> = y_mlx_f32.to_vec().unwrap();
        let max_diff = yv.iter().zip(mv.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(
            max_diff < 0.5,
            "tile (BM={bm}, BN={bn}, BK={bk}): max abs diff {max_diff} > 0.5"
        );
    }

    #[test]
    fn self_qmm_t_tile_64_64_32() { run_variant(64, 64, 32); }
    #[test]
    fn self_qmm_t_tile_64_128_32() { run_variant(64, 128, 32); }
    #[test]
    fn self_qmm_t_tile_128_128_32() { run_variant(128, 128, 32); }
}
```

- [ ] **Step 3.5: 编译 + 跑 3 个 variant 测试**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo test --release -p ironmlx --lib -- nn::self_qmm:: 2>&1 | tail -20
```

期望：3 tests passed。

不通过情况：
- (128,128,32) 可能因 BM × BN = 16384 > 1024 (Metal threadgroup max) 报错 → 简化实现需要每个 thread 算多个 output（micro-tile），暂时 skip 这个 variant，task 7 sweep 时根据数据决定是否 keep
- 数值误差 > 0.5 → debug kernel

如果 (128,128,32) 因 threadgroup limit 失败：
- 暂时把 panic 改为返回错误，跳过这个 variant 测试
- 在 task 7 时根据 sweep 结果决定是否需要 micro-tile 重写来支持大 tile

- [ ] **Step 3.6: Commit task 3**

```bash
git add ironmlx/src/nn/self_qmm/
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): self_qmm — function constants for 3 tile variants

Generalize self_qmm_t kernel to accept BM/BN/BK via Metal function
constants. Three PSO variants compiled and cached on demand:
(64,64,32), (64,128,32), (128,128,32).

Kernel rewritten to thread-per-output-element model (each thread
computes one output cell). Threadgroup size = BM × BN. (128,128,32)
exceeds Metal's 1024 threads/TG limit when M×N=16384 — variant
support deferred to task 7 sweep where micro-tile rewrite is
considered if data shows it's needed.

Cache: thread-safe Mutex<HashMap<(BM,BN,BK), MetalKernel>>. First
call per tile triggers PSO compile; subsequent calls reuse cached
kernel.

Unit tests (M=4, K=64, N=8) verify max abs diff < 0.5 for all
working variants vs mlx::quantization::quantized_matmul_on.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Device + Shape Lookup Table

**目的：** 加 `lookup.rs`：(device, M, N, K, bits, group_size) → (BM, BN, BK)。M1 Pro 占位行 + 全部 fallback。

**Files:**

- Create: `ironmlx/src/nn/self_qmm/lookup.rs`
- Modify: `ironmlx/src/nn/self_qmm/mod.rs` (使用 lookup 替代 hardcode)

### Steps

- [ ] **Step 4.1: 创建 lookup.rs**

```rust
// ironmlx/src/nn/self_qmm/lookup.rs
//! Tile selection lookup: (device, shape, quant) → (BM, BN, BK).
//!
//! Stage 9: M1 Pro entry placeholder (filled by task 7 sweep) + global
//! fallback. Stage 10 expands to M Max / M Ultra / etc.

use std::sync::OnceLock;

/// Tile dimensions chosen for a quant matmul dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
}

/// Default fallback tile — used when device/shape doesn't match any
/// hardcoded entry. Conservative: small enough to fit any threadgroup
/// memory budget, broad enough to function on any Apple Silicon GPU.
const DEFAULT_TILE: Tile = Tile { bm: 64, bn: 64, bk: 32 };

/// Lookup the optimal tile for the given (device, shape, quant).
///
/// `device_arch`: from `mlx::Device::get_architecture()`. e.g. "apple_g13s"
/// for M1 Pro 16-core GPU.
pub fn lookup_tile(
    device_arch: &str,
    _m: i32,
    _n: i32,
    _k: i32,
    _bits: i32,
    _group_size: i32,
) -> Tile {
    static WARNED: OnceLock<()> = OnceLock::new();
    match device_arch {
        // M1 Pro / M1 Pro Max GPU. Tile populated by task 7 sweep — initial
        // placeholder uses default. (BM=64, BN=128, BK=32) is a pre-sweep
        // educated guess based on llama.cpp's NRA=64/NRB=128 design at
        // similar arch class.
        "apple_g13s" | "apple_g13d" => Tile { bm: 64, bn: 128, bk: 32 },

        // All other devices: warn once and fall back to default.
        _ => {
            let _ = WARNED.set(());
            // Use tracing if available, otherwise eprintln in dev:
            tracing::warn!(
                target = "ironmlx::self_qmm::lookup",
                device = device_arch,
                "no tile entry; using default fallback (BM=64, BN=64, BK=32). \
                 Stage 10 will add explicit entries for additional devices."
            );
            DEFAULT_TILE
        }
    }
}
```

- [ ] **Step 4.2: 修改 mod.rs 使用 lookup**

```rust
// ironmlx/src/nn/self_qmm/mod.rs 修改 — 加 lookup 模块声明 + 在 qmm_t_on 中调用
mod kernel;
mod lookup;

// ... existing imports + enabled() ...

pub fn qmm_t_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bits: i32,
    group_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    assert_eq!(bits, 4, "self_qmm stage 9 only supports bits=4");
    assert_eq!(group_size, 64, "self_qmm stage 9 only supports group_size=64");
    let _ = target.into();

    let x_dims = x.shape();
    let x_dims_slice = x_dims.as_slice();
    let k = *x_dims_slice.last().expect("x must be at least 1-D");
    let m: i32 = x_dims_slice[..x_dims_slice.len() - 1].iter().product();
    let n = w.shape().as_slice()[0];

    // Get device architecture string. mlx::Device::default_device() returns
    // current GPU/CPU device; .architecture() returns identifier like
    // "apple_g13s".
    let device = mlx::Device::default_device();
    let arch = device.get_architecture();

    let tile = lookup::lookup_tile(&arch, m, n, k, bits, group_size);
    kernel::dispatch_qmm_t(x, w, scales, biases, tile.bm, tile.bn, tile.bk)
}
```

注意：`mlx::Device::get_architecture()` 接口 — 看 mlx Rust binding 实际签名。如果 binding 没暴露 `get_architecture`，需要先在 mlx crate 加（或者用 mlx-sys 直接调）。如果 binding 不存在，本 task 暂时返回 "unknown" 让 lookup 走 fallback。

- [ ] **Step 4.3: 编译 + 跑 unit tests 验证 lookup 没破坏正确性**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx 2>&1 | tail -5
cargo test --release -p ironmlx --lib -- nn::self_qmm:: 2>&1 | tail -10
```

期望：3 unit tests pass。

- [ ] **Step 4.4: Commit task 4**

```bash
git add ironmlx/src/nn/self_qmm/
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): self_qmm — device + shape lookup with M1 Pro placeholder

Add nn::self_qmm::lookup module implementing
  (device_arch, M, N, K, bits, group_size) -> (BM, BN, BK).

Initial entries:
  - "apple_g13s" / "apple_g13d" (M1 Pro variants): (64, 128, 32) —
    placeholder; task 7 will replace with sweep-derived optimum
  - any other device: fallback (64, 64, 32) with one-shot warn log

qmm_t_on() now reads device.get_architecture() and dispatches with
the looked-up tile, instead of hardcoding (64,64,32).

Stage 10 will expand to M Max / M Ultra / M3+ entries. For now any
non-M1-Pro chip uses safe-but-suboptimal default tile.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 集成到 Linear / GatedDeltaNet + Default-Path Regression Test

**目的：** ironmlx Linear / GatedDeltaNet `forward_on` 加 env var dispatch 分支：env=1 走 self_qmm，env=0 保持 stage 8 默认路径不变。验证 env=0 回归数据跟 stage 8 commit `811dd36` 完全一致。

**Files:**

- Modify: `ironmlx/src/nn/linear.rs` (forward_on 加 dispatch)
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (in_proj_qkvz / in_proj_ba 调用点同样 dispatch)
- Modify: `ironmlx/src/nn/echo_kernel.rs` (task 1 echo passthrough 移除 — 已完成 verification)

### Steps

- [ ] **Step 5.1: 移除 task 1 echo passthrough**

把 `linear.rs` 里 task 1 加的 `echo_enabled() / echo()` passthrough 删除，恢复到 stage 8 commit 状态 + 加 self_qmm dispatch。

```rust
// ironmlx/src/nn/linear.rs forward_on 修改：
// 把 task 1 加的 echo passthrough 删除；改为 self_qmm dispatch

pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let target = target.into();
    if let Some(qmeta) = &self.quant_meta {
        if crate::nn::self_qmm::enabled() {
            // Stage 9 opt-in path: route through self-quant kernel.
            // Falls back to mlx via panic if any of the assertions in
            // self_qmm::qmm_t_on fail (bits != 4 / group_size != 64).
            crate::nn::self_qmm::qmm_t_on(
                x,
                &self.weight,
                self.scales.as_ref().expect("quantized linear has scales"),
                self.biases.as_ref().expect("quantized linear has biases"),
                qmeta.bits,
                qmeta.group_size,
                target,
            )
        } else {
            // Stage 8 default path — unchanged.
            Ok(mlx::quantization::quantized_matmul_on(
                x,
                &self.weight,
                self.scales.as_ref().expect("quantized linear has scales"),
                self.biases.as_ref(),
                /* transpose */ true,
                qmeta.group_size,
                qmeta.bits,
                "affine",
                target,
            )?)
        }
    } else {
        // Non-quantized linear — unchanged.
        // ... existing code path ...
    }
}
```

注意：以 `linear.rs:195` 现有签名为准。`self.quant_meta` / `self.scales` / `self.biases` 字段名需对照实际 struct 定义。

- [ ] **Step 5.2: 修改 gated_delta_net.rs in_proj_qkvz 调用点**

找到 `ironmlx/src/nn/gated_delta_net.rs` 中 `in_proj_qkvz.forward_on(...)` 和 `in_proj_ba.forward_on(...)` 调用（gated_delta_net.rs:346-347）。这些是 `Linear` 的 forward，已经在 step 5.1 改过 forward_on 内部，所以不需要再改 caller — Linear::forward_on 自动走 dispatch。

✅ Action: 验证 grep 这两个调用点无需修改：

```bash
grep -n "in_proj_qkvz.forward_on\|in_proj_ba.forward_on" /Volumes/Dev/cxx-mlx/ironmlx/src/nn/gated_delta_net.rs
```

期望：调用点存在但保持原状（Linear::forward_on 内部 dispatch）。

- [ ] **Step 5.3: 编译 + 跑现有 unit tests**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx 2>&1 | tail -5
cargo +nightly fmt --all -- --check 2>&1 | tail -3
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -5
cargo test --release -p ironmlx --lib -- nn::self_qmm:: nn::linear:: nn::gated_delta_net:: 2>&1 | tail -20
```

期望：build / fmt / clippy / tests 全部通过。

- [ ] **Step 5.4: 启动 ironmlx server 跑 default-path regression baseline**

```bash
SHA=32f3e8ecf65426fc3306969496342d504bfa13f3
IRONMLX_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$SHA

pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
/Volumes/Dev/cxx-mlx/target/release/ironmlx serve --model "$IRONMLX_MODEL" --port 8080 > /tmp/task5-default.log 2>&1 &
until curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; do sleep 1; done

/Volumes/Dev/cxx-mlx/target/release/iron-bench \
  --target ironmlx-default=http://127.0.0.1:8080 \
  --model "$IRONMLX_MODEL" --model-dir "$IRONMLX_MODEL" \
  --prompt-len 128,512,2048 --max-tokens 64 --runs 5 --warmup 1 --format markdown \
  | tee /tmp/exp-task5-default.md
pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
```

期望：
- Prefill PP=128: ~220 tok/s
- Prefill PP=512: ~268 tok/s
- Prefill PP=2048: ~280 tok/s

跟 stage 8 commit `811dd36` 数据应**无差异**（jitter 内 < 1%）。如果显著退化，说明 forward_on 的 if-branch 引入 overhead，需要 debug。

- [ ] **Step 5.5: 决策门 — default-path regression 评估**

读 `/tmp/exp-task5-default.md`，对比 stage 8 commit `811dd36` 数据：

```
Stage 8 baseline (PP=128/512/2048 prefill tok/s): 220.4 / 268.0 / 280.1
Task 5 default (PP=128/512/2048):                 ____ / ____ / ____
Regression: ____% / ____% / ____%
```

- 任何 PP 上 regression > 1% → debug forward_on dispatch overhead，回 step 5.1 检查
- 全部 < 1%（jitter 内）→ 通过，commit 进 step 5.6

- [ ] **Step 5.6: Commit task 5**

```bash
git add ironmlx/src/nn/linear.rs ironmlx/src/nn/echo_kernel.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): integrate self_qmm into Linear::forward_on

Linear::forward_on() now dispatches to self_qmm::qmm_t_on when
IRONMLX_USE_SELF_QMM=1, otherwise uses mlx::quantized_matmul_on
(stage 8 default path verbatim).

GatedDeltaNet's in_proj_qkvz/in_proj_ba use Linear::forward_on
internally so they automatically pick up the dispatch with no
caller-side changes.

Task 1's echo_kernel passthrough in Linear::forward_on removed
(it served its fusion-barrier verification role; pre-verified
overhead was within jitter so we proceed with mx::fast::metal_kernel).

Default-path regression iron-bench (env=0):
  PP=128 prefill:  ____ tok/s (stage 8: 220.4)
  PP=512 prefill:  ____ tok/s (stage 8: 268.0)
  PP=2048 prefill: ____ tok/s (stage 8: 280.1)
  All within < 1% jitter — default path unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ironmlx-bench-kernel Binary Crate

**目的：** 单独 crate，CLI 调用，测单个 quant matmul kernel 的 wall-clock + GFLOP/s。Stage 9 用于 task 7 M1 Pro tile sweep。

**Files:**

- Create: `ironmlx-bench-kernel/Cargo.toml`
- Create: `ironmlx-bench-kernel/src/main.rs`
- Modify: `Cargo.toml` (workspace 加 member)

### Steps

- [ ] **Step 6.1: 创建 Cargo.toml**

```toml
# ironmlx-bench-kernel/Cargo.toml
[package]
name = "ironmlx-bench-kernel"
version.workspace = true
edition.workspace = true
publish = false

[dependencies]
ironmlx = { path = "../ironmlx" }
mlx = { path = "../mlx" }
anyhow = "1"
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 6.2: 创建 main.rs**

```rust
// ironmlx-bench-kernel/src/main.rs
//! Micro-benchmark for ironmlx self-quant matmul kernel.
//!
//! Measures wall-clock time of a single qmm_t dispatch with given
//! (M, N, K, BM, BN, BK, bits, group_size). Used by stage 9 task 7 to
//! sweep tile candidates on M1 Pro and pick the best (BM, BN, BK).

use anyhow::Result;
use clap::Parser;
use mlx::{ops, Dtype};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "ironmlx self-quant matmul kernel micro-benchmark")]
struct Args {
    /// Matmul rows (typically prompt length × batch)
    #[arg(long, default_value_t = 2048)]
    m: i32,

    /// Matmul output cols (typically intermediate_size)
    #[arg(long, default_value_t = 9216)]
    n: i32,

    /// Matmul depth (typically hidden_size)
    #[arg(long, default_value_t = 2560)]
    k: i32,

    /// Tile BM (rows per threadgroup)
    #[arg(long, default_value_t = 64)]
    bm: i32,

    /// Tile BN (cols per threadgroup)
    #[arg(long, default_value_t = 128)]
    bn: i32,

    /// Tile BK (depth chunk per K iteration)
    #[arg(long, default_value_t = 32)]
    bk: i32,

    /// Quantization bits (only 4 supported in stage 9)
    #[arg(long, default_value_t = 4)]
    bits: i32,

    /// Quantization group size (only 64 supported in stage 9)
    #[arg(long, default_value_t = 64)]
    group_size: i32,

    /// Number of timed runs (median reported)
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Number of warmup runs (excluded)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Run mlx baseline for comparison
    #[arg(long, default_value_t = false)]
    mlx_baseline: bool,
}

fn build_inputs(m: i32, n: i32, k: i32, group_size: i32, bits: i32) -> Result<(mlx::Array, mlx::Array, mlx::Array, mlx::Array)> {
    // x bf16 [M, K]
    let x_f32 = ops::random::uniform_distribution::<f32>(-1.0, 1.0, &[m, k][..], None)?;
    let x = ops::cast::astype(&x_f32, Dtype::Bfloat16)?;

    // raw weights bf16 [N, K]; quantize via mlx::quantization::quantize
    let raw_w = ops::random::uniform_distribution::<f32>(-0.5, 0.5, &[n, k][..], None)?;
    let raw_w_bf16 = ops::cast::astype(&raw_w, Dtype::Bfloat16)?;
    let (w_packed, w_scales, w_biases) =
        mlx::quantization::quantize(&raw_w_bf16, group_size, bits)?;

    Ok((x, w_packed, w_scales, w_biases))
}

fn time_self_qmm(args: &Args, inputs: &(mlx::Array, mlx::Array, mlx::Array, mlx::Array)) -> Result<f64> {
    use ironmlx::nn::self_qmm::kernel;
    let (x, w, s, b) = inputs;

    // Warmup
    for _ in 0..args.warmup {
        let y = kernel::dispatch_qmm_t(x, w, s, b, args.bm, args.bn, args.bk)?;
        mlx::transforms::eval(&[&y])?;
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let t0 = Instant::now();
        let y = kernel::dispatch_qmm_t(x, w, s, b, args.bm, args.bn, args.bk)?;
        mlx::transforms::eval(&[&y])?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn time_mlx_baseline(args: &Args, inputs: &(mlx::Array, mlx::Array, mlx::Array, mlx::Array)) -> Result<f64> {
    let (x, w, s, b) = inputs;

    for _ in 0..args.warmup {
        let y = mlx::quantization::quantized_matmul_on(
            x, w, s, Some(b), true, args.group_size, args.bits, "affine", (),
        )?;
        mlx::transforms::eval(&[&y])?;
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let t0 = Instant::now();
        let y = mlx::quantization::quantized_matmul_on(
            x, w, s, Some(b), true, args.group_size, args.bits, "affine", (),
        )?;
        mlx::transforms::eval(&[&y])?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# ironmlx-bench-kernel");
    println!("M={}, N={}, K={}", args.m, args.n, args.k);
    println!("Tile: BM={}, BN={}, BK={}", args.bm, args.bn, args.bk);
    println!("Quant: bits={}, group_size={}", args.bits, args.group_size);
    println!("Runs: {} measured (after {} warmup)", args.runs, args.warmup);
    println!();

    let inputs = build_inputs(args.m, args.n, args.k, args.group_size, args.bits)?;

    let self_t = time_self_qmm(&args, &inputs)?;
    let flops = 2.0 * (args.m as f64) * (args.n as f64) * (args.k as f64);
    let self_gflops = flops / self_t / 1e9;
    println!("self_qmm:    median {:.3} ms, {:.1} GFLOP/s", self_t * 1000.0, self_gflops);

    if args.mlx_baseline {
        let mlx_t = time_mlx_baseline(&args, &inputs)?;
        let mlx_gflops = flops / mlx_t / 1e9;
        let speedup = mlx_t / self_t;
        println!("mlx affine:  median {:.3} ms, {:.1} GFLOP/s", mlx_t * 1000.0, mlx_gflops);
        println!("self_qmm vs mlx: {:.2}× speedup", speedup);
    }

    Ok(())
}
```

注意：`ironmlx::nn::self_qmm::kernel` 不是 pub — 改 `ironmlx/src/nn/self_qmm/mod.rs` 让 `pub mod kernel;`（暴露给 bench 用）。或者在 mod.rs 加 pub re-export `pub use kernel::dispatch_qmm_t;`。

- [ ] **Step 6.3: 改 self_qmm/mod.rs 暴露 kernel::dispatch_qmm_t**

```rust
// ironmlx/src/nn/self_qmm/mod.rs 头部:
pub mod kernel;  // 改 mod 为 pub mod
mod lookup;
```

- [ ] **Step 6.4: 加 workspace member**

```toml
# /Volumes/Dev/cxx-mlx/Cargo.toml workspace 块加：
[workspace]
members = [
    # ... existing members ...
    "ironmlx-bench-kernel",
]
```

- [ ] **Step 6.5: 编译 binary**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx-bench-kernel 2>&1 | tail -5
```

期望：build 通过，binary 在 `target/release/ironmlx-bench-kernel`。

- [ ] **Step 6.6: 跑 sanity test**

```bash
/Volumes/Dev/cxx-mlx/target/release/ironmlx-bench-kernel \
  --m 2048 --n 9216 --k 2560 --bm 64 --bn 128 --bk 32 \
  --runs 3 --warmup 1 --mlx-baseline
```

期望：
- 输出 `self_qmm:    median X.XXX ms, XX.X GFLOP/s`
- 输出 `mlx affine:  median X.XXX ms, XX.X GFLOP/s`
- 输出 `self_qmm vs mlx: X.XXx speedup`
- 不报错

- [ ] **Step 6.7: Commit task 6**

```bash
git add ironmlx-bench-kernel/ Cargo.toml ironmlx/src/nn/self_qmm/mod.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage9): ironmlx-bench-kernel — micro-benchmark binary

New workspace crate ironmlx-bench-kernel with CLI:
  --M --N --K --BM --BN --BK --bits --group-size --runs --warmup --mlx-baseline

Times a single qmm_t dispatch via ironmlx::nn::self_qmm::kernel and
optionally the mlx::quantization::quantized_matmul_on baseline for
comparison. Median wall-clock + GFLOP/s reported.

Used by task 7 to sweep tile candidates on M1 Pro and populate the
lookup table M1 Pro entry.

Sanity test on Qwen3.5 FFN shape (M=2048, N=9216, K=2560, bits=4,
group_size=64, BM=64, BN=128, BK=32):
  self_qmm: ____ ms, ____ GFLOP/s
  mlx baseline: ____ ms, ____ GFLOP/s
  speedup: ____×

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: M1 Pro Tile Sweep + 填 Lookup 表

**目的：** 用 task 6 的 binary 跑 3 个 tile candidates × Qwen3.5 关键 shape，找 M1 Pro 上的最优 tile，填回 `lookup.rs`。

**Files:**

- Modify: `ironmlx/src/nn/self_qmm/lookup.rs` (M1 Pro entry 替换为 sweep 最优)

### Steps

- [ ] **Step 7.1: 跑 Qwen3.5 FFN up_proj sweep (M=2048, K=2560, N=9216)**

```bash
BENCH=/Volumes/Dev/cxx-mlx/target/release/ironmlx-bench-kernel

echo "=== Qwen3.5 FFN up_proj (M=2048, K=2560, N=9216) ==="
for cfg in "64 64 32" "64 128 32" "128 128 32"; do
  read BM BN BK <<< "$cfg"
  echo "--- BM=$BM BN=$BN BK=$BK ---"
  $BENCH --m 2048 --n 9216 --k 2560 --bm $BM --bn $BN --bk $BK \
    --runs 5 --warmup 1 --mlx-baseline
done
```

记录每 tile 的 self_qmm GFLOP/s + speedup vs mlx：

```
BM=64 BN=64 BK=32:  ____ GFLOP/s, ____× speedup
BM=64 BN=128 BK=32: ____ GFLOP/s, ____× speedup
BM=128 BN=128 BK=32: ____ GFLOP/s, ____× speedup (or N/A if threadgroup limit)
```

- [ ] **Step 7.2: 跑 attention proj sweep (M=2048, K=2560, N=2560)**

```bash
echo "=== Qwen3.5 attention q_proj (M=2048, K=2560, N=2560) ==="
for cfg in "64 64 32" "64 128 32" "128 128 32"; do
  read BM BN BK <<< "$cfg"
  echo "--- BM=$BM BN=$BN BK=$BK ---"
  $BENCH --m 2048 --n 2560 --k 2560 --bm $BM --bn $BN --bk $BK \
    --runs 5 --warmup 1 --mlx-baseline
done
```

记录数据。

- [ ] **Step 7.3: 跑 down_proj sweep (M=2048, K=9216, N=2560)**

```bash
echo "=== Qwen3.5 FFN down_proj (M=2048, K=9216, N=2560) ==="
for cfg in "64 64 32" "64 128 32" "128 128 32"; do
  read BM BN BK <<< "$cfg"
  echo "--- BM=$BM BN=$BN BK=$BK ---"
  $BENCH --m 2048 --n 2560 --k 9216 --bm $BM --bn $BN --bk $BK \
    --runs 5 --warmup 1 --mlx-baseline
done
```

记录数据。

- [ ] **Step 7.4: 决定最优 tile**

汇总 3 个 shape × 3 个 tile = 9 个数据点。在大 N (FFN up/down)、中 N (attention) 两类 shape 上分别评估：

| Shape | Best tile | self_qmm GFLOP/s | speedup vs mlx |
|---|---|---|---|
| FFN up (2048×2560×9216) | ? | ? | ? |
| Attn q (2048×2560×2560) | ? | ? | ? |
| FFN down (2048×9216×2560) | ? | ? | ? |

简化策略（stage 9 时间约束）：选**整体平均 GFLOP/s 最高的 tile** 作为 M1 Pro 单条 lookup 行。如果不同 shape 偏好不同 tile（可能性大），stage 10 再加 shape-conditional lookup。

记录结论：M1 Pro 最优 tile = (BM=___, BN=___, BK=___)

- [ ] **Step 7.5: 更新 lookup.rs M1 Pro entry**

```rust
// ironmlx/src/nn/self_qmm/lookup.rs 修改 match 分支
match device_arch {
    "apple_g13s" | "apple_g13d" => Tile {
        bm: ___,  // 填 step 7.4 选定的 BM
        bn: ___,  // 填 BN
        bk: ___,  // 填 BK
    },
    _ => DEFAULT_TILE,
}
```

- [ ] **Step 7.6: 跑 unit tests 验证 lookup 改动没破坏正确性**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo build --release -p ironmlx 2>&1 | tail -3
cargo test --release -p ironmlx --lib -- nn::self_qmm:: 2>&1 | tail -10
```

期望：3 unit tests pass。

- [ ] **Step 7.7: Commit task 7**

```bash
git add ironmlx/src/nn/self_qmm/lookup.rs
git commit -m "$(cat <<'EOF'
perf(ironmlx-p8a-stage9): tile sweep + populate M1 Pro lookup entry

Ran ironmlx-bench-kernel sweep on M1 Pro 16-core GPU across:
  - Qwen3.5 FFN up_proj  (M=2048, K=2560, N=9216)
  - Qwen3.5 attn q_proj  (M=2048, K=2560, N=2560)
  - Qwen3.5 FFN down_proj (M=2048, K=9216, N=2560)

Tile candidates: (64,64,32) (64,128,32) (128,128,32).

Best tile picked: BM=____, BN=____, BK=____ (highest avg GFLOP/s
across the 3 shapes). speedup vs mlx::quantized_matmul_on: ____×.

Lookup entry "apple_g13s"/"apple_g13d" updated with the sweep result.
Stage 10 will add shape-conditional lookup (different tiles per
matmul size) and entries for M Max / M Ultra / etc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-End iron-bench Validation + 三方对比

**目的：** 启动 ironmlx server with `IRONMLX_USE_SELF_QMM=1`，跑 iron-bench PP=2048。对比 stage 8 baseline + omlx + llama.cpp。

**Files:** 无代码改动（纯实验 + 验证）。

### Steps

- [ ] **Step 8.1: 启动 ironmlx server (env=1) 跑 PP=128/512/2048 iron-bench**

```bash
SHA=32f3e8ecf65426fc3306969496342d504bfa13f3
IRONMLX_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$SHA

pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2

IRONMLX_USE_SELF_QMM=1 /Volumes/Dev/cxx-mlx/target/release/ironmlx serve \
  --model "$IRONMLX_MODEL" --port 8080 > /tmp/task8-self.log 2>&1 &
until curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; do sleep 1; done

echo "=== sanity output ==="
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"x",
  "messages":[{"role":"user","content":"Say hi in 5 words."}],
  "max_tokens":12,
  "temperature":0,
  "chat_template_kwargs":{"enable_thinking":false}
}' | head -c 500; echo

echo "=== iron-bench ==="
/Volumes/Dev/cxx-mlx/target/release/iron-bench \
  --target ironmlx-self=http://127.0.0.1:8080 \
  --model "$IRONMLX_MODEL" --model-dir "$IRONMLX_MODEL" \
  --prompt-len 128,512,2048 --max-tokens 64 --runs 5 --warmup 1 --format markdown \
  | tee /tmp/exp-task8-self.md

pkill -INT -f "ironmlx serve" 2>/dev/null; sleep 2
```

期望:
- Sanity: 输出 "Hello there, friend." (或类似 5-7 词正常回应，无乱码)
- Prefill PP=2048: > 281 tok/s (acceptance criteria)
- 理想: > 332 (llama.cpp Q4_K_M chunk OFF)

记录数据。

- [ ] **Step 8.2: 决策门 — acceptance criteria 评估**

读 `/tmp/exp-task8-self.md` Prefill PP @ PP=2048：

- > 281 tok/s → **PASS**, stage 9 acceptance 达成
- < 281 → **FAIL**, debug：检查 lookup 是否选对 tile，kernel 实现是否有 bug，回 task 2/3/7 修

- [ ] **Step 8.3: 三方对比汇总**

```
Stage 9 final 三方对比 (PP=2048 prefill, tok/s):
  ironmlx (env=1, self_qmm) [task 8]: ____
  ironmlx (env=0, mlx default)        [stage 8]: 281
  omlx 0.3.9.dev1                     [stage 8]: 285
  llama.cpp Q4_K_M chunk OFF          [stage 8]: 332
  llama.cpp Q4_K_M chunk ON  (ub=512) [stage 8]: 390
```

填具体数字，写到 commit message。

- [ ] **Step 8.4: Commit task 8 (端到端验证结果)**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
test(ironmlx-p8a-stage9): end-to-end iron-bench validation

IRONMLX_USE_SELF_QMM=1 iron-bench PP=128/512/2048 × 5 runs serial:
  PP=128 prefill:  ____ tok/s
  PP=512 prefill:  ____ tok/s
  PP=2048 prefill: ____ tok/s

Three-way comparison @ PP=2048:
  ironmlx (env=1) self_qmm:               ____
  ironmlx (env=0) mlx default [stage 8]:  281
  omlx 0.3.9.dev1 [stage 8]:              285
  llama.cpp Q4_K_M chunk OFF [stage 8]:   332
  llama.cpp Q4_K_M chunk ON  [stage 8]:   390

Acceptance: env=1 PP=2048 prefill ____ vs target > 281 [PASS/FAIL].
Stretch goal > 332 [PASS/FAIL]. Extreme goal > 390 [PASS/FAIL].

Sanity: same prompt produces equivalent text vs env=0 path (greedy
output identical for "Say hi in 5 words.").

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: 文档 + Memory 更新

**目的：** README 加 env var 说明，memory 记录 stage 9 完成状态 + 关键发现。

**Files:**

- Modify: `README.md` (加 IRONMLX_USE_SELF_QMM env var 文档)
- Create/Modify: `~/.claude/projects/-Volumes-Dev-cxx-mlx/memory/project_p8a_stage9_findings.md` (新)
- Modify: `~/.claude/projects/-Volumes-Dev-cxx-mlx/memory/MEMORY.md` (索引)

### Steps

- [ ] **Step 9.1: 加 README env var 文档**

在 `/Volumes/Dev/cxx-mlx/README.md` 找合适位置（Configuration / Tuning 段，如不存在则新增）加：

```markdown
### Performance tuning

`ironmlx serve` recognizes the following env vars for kernel-level
opt-in features:

- `IRONMLX_USE_SELF_QMM=1` — Route quantized matmul through ironmlx's
  self-implemented Metal kernel (stage 9). Currently supports MLX 4-bit
  affine quantization (group_size=64) on the prefill (qmm_t) path. M1 Pro
  has a tuned tile selection; other Apple Silicon chips fall back to a
  conservative default tile (subject to broader sweep in stage 10).
  Default: unset (mlx::quantized_matmul, no change from stage 8).

- `IRONMLX_PREFILL_CHUNK_SIZE=N` — Max tokens per prefill forward
  (stage 8). `0` disables chunking. Default: `2048`.
```

- [ ] **Step 9.2: 创建 stage 9 findings memory**

```markdown
<!-- ~/.claude/projects/-Volumes-Dev-cxx-mlx/memory/project_p8a_stage9_findings.md -->
---
name: P8a stage 9 self-quant kernel 发现
description: ironmlx 自研 4-bit quant kernel — fusion barrier 验证 + tile sweep + 三方对比
type: project
---

P8a stage 9 实施 (commits 8780675 spec → ____ task 9 final, 分支 ironmlx-p8a-stage9-quant-kernel) 的关键发现：

**1. Fusion barrier 验证（task 1）**
echo kernel 插入 Linear forward path 测得 fusion barrier overhead 为
____% (PP=2048)。决策门 < 5%，PASS → 继续用 mx::fast::metal_kernel
作为 stage 9 kernel 注入机制。

**2. Self-quant kernel 实现（task 2-3）**
ironmlx::nn::self_qmm 模块实现 MLX 4-bit affine (group_size=64) qmm_t
Metal kernel。Stage 9 起步：thread-per-output-element 模型（每 thread
一个 output cell，threadgroup size = BM × BN）。Function constants 暴
露 BM/BN/BK，3 个 PSO variants：(64,64,32) (64,128,32) (128,128,32)。

数值正确性：3 variants 在 small shape (M=4, K=64, N=8) 上 max abs
diff < 0.5 vs mlx::quantized_matmul_on，全部 PASS。

**3. M1 Pro tile sweep 结论（task 7）**
最优 tile = (BM=___, BN=___, BK=___)，对 Qwen3.5 关键 shape 平均
speedup vs mlx ____×。Lookup 表 "apple_g13s"/"apple_g13d" 行已填。

**4. End-to-end 验证（task 8）**
IRONMLX_USE_SELF_QMM=1 iron-bench PP=2048 prefill ____ tok/s。
- vs ironmlx mlx default (281): ____% (target > 281, [PASS/FAIL])
- vs llama.cpp chunk OFF (332): ____% (stretch [PASS/FAIL])
- vs llama.cpp chunk ON (390): ____% (extreme [PASS/FAIL])

**5. 默认路径回归（task 5）**
IRONMLX_USE_SELF_QMM 未设时，PP=128/512/2048 prefill 跟 stage 8
commit 811dd36 baseline 数据 jitter 内 < 1%。opt-in 完全隔离。

**Stage 9 后续路线（informational）**：
- Stage 10 — 扩 device 分级 (M Max / M Ultra / M3+) + 加 candidates
  + first-run profiling + 启动 PSO warmup
- Stage 11 — 扩 quant scheme (Q4_K_M / Q5_K_M / Q8_0 / bf16) + GGUF
  loader
- Stage 12 — Decode kernel (qmv vector + split-K)
```

- [ ] **Step 9.3: 更新 MEMORY.md 索引**

加一行到 `~/.claude/projects/-Volumes-Dev-cxx-mlx/memory/MEMORY.md`：

```markdown
- [P8a stage 9 self-quant kernel 发现](project_p8a_stage9_findings.md) — fusion barrier OK; self_qmm Metal kernel 3 tile variants; M1 Pro tile sweep 选 ____×____×____; PP=2048 ____ tok/s vs target 281
```

- [ ] **Step 9.4: Commit task 9**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(ironmlx-p8a-stage9): document IRONMLX_USE_SELF_QMM env var

Add Performance tuning section to README documenting the env vars
introduced in stage 8 (IRONMLX_PREFILL_CHUNK_SIZE) and stage 9
(IRONMLX_USE_SELF_QMM).

memory/project_p8a_stage9_findings.md created separately; logs the
stage 9 implementation outcome (sweep results, e2e numbers,
acceptance criteria status).

Stage 9 complete on branch ironmlx-p8a-stage9-quant-kernel:
  Task 1 (fusion barrier verify) → ____
  Task 2 (single-tile kernel)   → ____
  Task 3 (3 tile variants)      → ____
  Task 4 (lookup table)         → ____
  Task 5 (Linear integration)   → ____
  Task 6 (bench-kernel binary)  → ____
  Task 7 (M1 Pro sweep)         → ____
  Task 8 (e2e validation)       → ____ [acceptance PASS/FAIL]
  Task 9 (docs + memory)        → this commit

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

按 writing-plans skill 要求的三项 checklist：

**1. Spec coverage** — spec section 8 的 9 个 step 都有对应 task：

| Spec step | Plan task |
|---|---|
| Fusion barrier 预先验证 | Task 1 |
| Self-qmm Metal source (写死 (64,64,32)) + unit test | Task 2 |
| 3 tile variants (function constants) | Task 3 |
| Device + shape lookup 表 | Task 4 |
| 集成 Linear / GatedDeltaNet + regression test | Task 5 |
| ironmlx-bench-kernel binary | Task 6 |
| M1 Pro tile sweep + 填表 | Task 7 |
| End-to-end iron-bench + 三方对比 | Task 8 |
| 文档 + memory 更新 | Task 9 |

✅ 全部覆盖。

**2. Placeholder scan** — 搜索 "TODO" / "TBD" / "fill in":

数据 placeholder（待 implementer 实测填入）：
- Task 1.10/1.11 fusion barrier % regression
- Task 5.5/5.6 default-path regression %
- Task 6.7 ironmlx-bench-kernel sanity GFLOP/s
- Task 7.4/7.7 sweep 结论
- Task 8.3/8.4 end-to-end 三方对比数据
- Task 9.2/9.4 stage 9 final 数据

这些是**实测数据**待填，不是设计 placeholder（设计已完整），符合 plan 接受范围。

❌ 一个真实问题: Task 4.2 提到"如果 binding 不存在，本 task 暂时返回 'unknown' 让 lookup 走 fallback" — 这是 conditional placeholder。修正：明确 task 4 第 4.2 步先 grep `mlx::Device::get_architecture` 是否在 mlx Rust binding 中暴露；如果不在，先做一个 mlx crate 内的小 patch (additive) 暴露它，而不是 stage 9 plan 内推迟。

**修正 Task 4.2** — 加 step 4.2a：

```markdown
- [ ] **Step 4.2a: 验证 mlx::Device::get_architecture binding 存在**

```bash
grep -n "get_architecture\|fn architecture" /Volumes/Dev/cxx-mlx/mlx/src/ -r 2>&1 | head
```

如果存在 → 直接用，进 step 4.2。
如果不存在 → 先在 mlx crate 加 binding (cxx FFI to mlx_sys::device::ffi::get_architecture)，commit 这一笔 ("feat(mlx): expose Device::get_architecture")，再继续 step 4.2.
```

**3. Type consistency** — 检查跨 task 的接口签名一致性：

- `qmm_t_on()` 签名在 task 2 / 3 / 4 / 5 / 6 都用同一签名 `(x, w, scales, biases, bits, group_size, target)` ✓
- `kernel::dispatch_qmm_t()` 签名 task 2 (无 BM/BN/BK 参数硬编码) → task 3+ (加 BM/BN/BK 参数) — task 3.2 替换全部签名 ✓
- `Tile` struct 在 task 4 引入，task 5+ 都用同一定义 ✓

✅ 一致。

修订上述 self-review 发现的 Task 4.2a，重新写到上面 Task 4 中（已加）。

---

## Acceptance Criteria（plan 完整完成的判定）

✅ Plan 全部 9 task commits 落地  
✅ Task 1 fusion barrier regression < 5%  
✅ Task 2-3 self_qmm 3 tile variants 数值正确（atol < 0.5）  
✅ Task 5 默认路径回归 < 1%  
✅ Task 8 IRONMLX_USE_SELF_QMM=1 PP=2048 prefill > 281 tok/s（必须）  
🎯 Task 8 stretch > 332 tok/s（理想）  
🌟 Task 8 extreme > 390 tok/s（极致）  
✅ Task 9 README + memory 更新

---

## 后续 Stage（informational, not in this plan）

- **Stage 10** — 扩 device 分级 + 加 candidates + first-run profiling + warmup
- **Stage 11** — 扩 quant scheme (Q4_K_M / Q5_K_M / Q8_0 / bf16) + GGUF loader
- **Stage 12** — Decode kernel 自研 (qmv vector + split-K)
