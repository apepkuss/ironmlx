# cxx-mlx P2b · Fast Ops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `mlx::core::fast::*` 命名空间下的 5 个融合算子（rms_norm、layer_norm、rope int offset、rope array offset、scaled_dot_product_attention）提供 Rust 安全绑定。

**Architecture:** 三层结构：shim C++ 适配层（抹平 `std::optional`、重载、`std::string`）→ cxx 桥接层（声明 ABI 边界）→ 安全 Rust API（暴露 `Option<&Array>` / `Result<Array>`）。可选 array 用 `*const MlxArray`（nullptr=None）穿越 FFI；可选 float 用 `(bool, f32)` 双参编码；mask_mode 用 `rust::Str` 借用。

**Tech Stack:** Rust 1.82+（precise capturing `use<>`）、cxx 1.0（含 raw pointer 的 unsafe extern 块）、MLX C++ 共享安装、cargo nightly fmt + clippy + release build。

**Spec reference:** `docs/superpowers/specs/2026-05-04-cxx-mlx-p2b-fast-ops-design.md`

---

## 关键背景信息（实施者必读）

### 项目三层结构与文件入口

- **shim 层**：`mlx-sys/shim/include/cxx_mlx_shim/*.h` + `mlx-sys/shim/src/*.cc` —— 手写 C++ free function，包内部用 STL/重载/异常。已有 array.h/cc, transforms.h/cc, stream.h/cc 作为模式参考。
- **桥接层**：`mlx-sys/src/bridge/*.rs` —— `#[cxx::bridge]` 声明 FFI。`mlx-sys/src/bridge/mod.rs` 列出所有桥接模块；`mlx-sys/src/lib.rs` 顶层 re-export。
- **安全层**：`mlx/src/*.rs` —— `mlx/src/lib.rs` 顶层 re-export 所有公开类型与函数。

### 已有 API 引用点

- `Array` 构造：`Array::from_inner(inner: cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>) -> Self`（[mlx/src/array.rs:11](mlx/src/array.rs#L11)）
- `Array` 取出底层引用：`a.as_inner() -> &mlx_sys::array::ffi::MlxArray`（[mlx/src/array.rs:139](mlx/src/array.rs#L139)）
- 错误转换：`Error::from(cxx::Exception) -> Error`（已映射，由 `?` 自动调）
- 跨桥接共享 `MlxArray`：`type MlxArray = crate::bridge::array::ffi::MlxArray;`（[mlx-sys/src/bridge/transforms.rs:10](mlx-sys/src/bridge/transforms.rs#L10) 是范例）

### 强制检查（CLAUDE.md 规定，每次 commit 前必跑）

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

### MLX 上游 API（来自 `${MLX_DIR}/include/mlx/fast.h`）

```cpp
namespace mlx::core::fast {

array rms_norm(const array& x, const std::optional<array>& weight, float eps,
               StreamOrDevice s = {});

array layer_norm(const array& x, const std::optional<array>& weight,
                 const std::optional<array>& bias, float eps, StreamOrDevice s = {});

array rope(const array& x, int dims, bool traditional,
           std::optional<float> base, float scale, int offset,
           const std::optional<array>& freqs = std::nullopt, StreamOrDevice s = {});

array rope(const array& x, int dims, bool traditional,
           std::optional<float> base, float scale, const array& offset,
           const std::optional<array>& freqs = std::nullopt, StreamOrDevice s = {});

array scaled_dot_product_attention(
    const array& queries, const array& keys, const array& values,
    const float scale, const std::string& mask_mode = "",
    std::optional<array> mask_arr = {}, const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

}
```

`StreamOrDevice s = {}` 即 MLX 默认（caller 线程当前默认 stream），shim 全部不传该参数（保留 `= {}` 默认值）。

---

## 文件清单

### 新建
- `mlx-sys/shim/include/cxx_mlx_shim/fast.h` —— shim 头（5 个 free function 声明）
- `mlx-sys/shim/src/fast.cc` —— shim 实现 + `opt_arr` / `opt_f` 助手
- `mlx-sys/src/bridge/fast.rs` —— cxx::bridge（5 个 unsafe FFI）
- `mlx/src/fast.rs` —— 安全 API（5 个公开函数）
- `mlx/tests/p2b_fast.rs` —— 集成测试

### 修改
- `mlx-sys/build.rs` —— `cxx_build::bridges` 加 `"src/bridge/fast.rs"`，`.file()` 加 `"shim/src/fast.cc"`
- `mlx-sys/src/bridge/mod.rs` —— 加 `pub mod fast;`
- `mlx-sys/src/lib.rs` —— 加 `pub use bridge::fast;`
- `mlx/src/lib.rs` —— 加 `pub mod fast;` + `pub use fast::{...}`

---

## Task 1: 框架搭建 + rms_norm（最小端到端切片）

**目的**：打通 build.rs / mod.rs / lib.rs 的全部接线，并实现首个最简单函数 rms_norm，确保三层链路全部 OK。

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/fast.h`
- Create: `mlx-sys/shim/src/fast.cc`
- Create: `mlx-sys/src/bridge/fast.rs`
- Create: `mlx/src/fast.rs`
- Create: `mlx/tests/p2b_fast.rs`
- Modify: `mlx-sys/build.rs`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/src/lib.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1.1: 写失败的集成测试 `rms_norm`**

将以下完整内容写入 `mlx/tests/p2b_fast.rs`：

```rust
//! Integration tests for mlx::fast — fused MLX kernels for Transformer inference.

use mlx::{fast, Array};

#[test]
fn rms_norm_no_weight_known_values() {
    // x = [[1.0, 2.0, 3.0, 4.0]], shape [1, 4]
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // sqrt(7.5 + 1e-5) ≈ 2.7386140
    // Expected output ≈ x / 2.7386 = [0.36514, 0.73029, 1.09543, 1.46059]
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let out = fast::rms_norm(&x, None, 1e-5).expect("rms_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);

    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.36514_f32, 0.73029, 1.09543, 1.46059];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn rms_norm_with_weight_scales_output() {
    // Same x as above; weight = [2.0, 2.0, 2.0, 2.0] → output = 2 × no-weight result.
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let w = Array::from_slice(&[2.0_f32, 2.0, 2.0, 2.0], &[4]).expect("w");
    let out = fast::rms_norm(&x, Some(&w), 1e-5).expect("rms_norm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.73029_f32, 1.46058, 2.19087, 2.92117];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm_w[{i}] = {got}, want {want}"
        );
    }
}
```

- [ ] **Step 1.2: 运行测试，确认失败（编译错误）**

Run: `cargo test --test p2b_fast --no-run`
Expected: 编译失败，错误信息提到 `mlx::fast` 不存在或 `rms_norm` 未定义。

- [ ] **Step 1.3: 创建 shim 头 `fast.h`**

将以下完整内容写入 `mlx-sys/shim/include/cxx_mlx_shim/fast.h`：

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// rms_norm: weight=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray* weight,
    float eps);

}  // namespace cxx_mlx
```

- [ ] **Step 1.4: 创建 shim 实现 `fast.cc`**

将以下完整内容写入 `mlx-sys/shim/src/fast.cc`：

```cpp
#include "cxx_mlx_shim/fast.h"

#include <optional>
#include <string>

#include "mlx/fast.h"

namespace cxx_mlx {

namespace {

// pointer → optional<array>. mlx::array copy is cheap (refcount on array_desc_).
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<float> opt_f(bool has, float v) {
  return has ? std::optional<float>(v) : std::nullopt;
}

}  // namespace

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x, const MlxArray* weight, float eps) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::rms_norm(x, opt_arr(weight), eps));
}

}  // namespace cxx_mlx
```

注：`opt_f` 现在没人用，但下一个 task（rope）就会用。先放好避免来回改文件。如果 clippy/编译器对未用静态函数报警，无 namespace 限定的 `inline` + 匿名命名空间下 C++ 不会报警。

- [ ] **Step 1.5: 创建桥接 `bridge/fast.rs`**

将以下完整内容写入 `mlx-sys/src/bridge/fast.rs`：

```rust
//! Bridge for MLX fast ops (fused Transformer kernels).
//!
//! Optional `array` arguments use raw `*const MlxArray` — same convention
//! as `async_eval_many` in the stream bridge. nullptr maps to MLX's
//! `std::optional<array>{std::nullopt}`. The Rust-side safe wrapper
//! converts `Option<&Array>` to a raw pointer at call time.
//!
//! Optional `float` arguments (rope's `base`) are encoded as a
//! `bool has_base` + `f32 base` pair to avoid raw float pointers across
//! cxx (which doesn't model `Option<f32>` directly).

#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;

        unsafe fn fast_rms_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            eps: f32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
```

- [ ] **Step 1.6: 在 `mlx-sys/src/bridge/mod.rs` 增加 `pub mod fast;`**

打开 `mlx-sys/src/bridge/mod.rs`，将其末尾改为：

```rust
pub mod array;
pub mod transforms;
pub mod stream;
pub mod fast;
```

（保留前面的注释；只在 `pub mod stream;` 之后增加 `pub mod fast;`）

- [ ] **Step 1.7: 在 `mlx-sys/src/lib.rs` 增加 re-export**

打开 `mlx-sys/src/lib.rs`，把末尾的 re-export 改为：

```rust
pub use bridge::array;
pub use bridge::transforms;
pub use bridge::stream;
pub use bridge::fast;
```

- [ ] **Step 1.8: 在 `mlx-sys/build.rs` 注册 fast 桥接 + shim cc**

打开 `mlx-sys/build.rs`，找到 `cxx_build::bridges([...])` 调用块，把它替换为：

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/fast.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

（仅在桥接列表与 `.file()` 列表的末尾追加 fast 项）

- [ ] **Step 1.9: 创建安全 API `mlx/src/fast.rs`**

将以下完整内容写入 `mlx/src/fast.rs`：

```rust
//! Fused MLX kernels for Transformer inference: rms_norm, layer_norm,
//! rope, scaled_dot_product_attention.
//!
//! These are MLX's `mlx::core::fast::*` ops — single fused Metal kernels,
//! not compositions of primitives. They are the performance-critical
//! primitives for LLM/VLM inference.
//!
//! Like all ops in this crate, fast ops queue work on the caller thread's
//! current default stream. Use [`crate::set_default_stream`] to override.

use crate::{Array, Error, Result};

/// Root-mean-square normalization with optional learned scale.
///
/// `weight=None` skips the scale step (pure normalization).
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: w is null or a borrow of `weight: &Array` valid for this call.
    let inner = unsafe { mlx_sys::fast::ffi::fast_rms_norm(x.as_inner(), w, eps) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 1.10: 在 `mlx/src/lib.rs` 增加 `pub mod fast;` + re-export**

打开 `mlx/src/lib.rs`，在最后一行 `pub use transforms::...` 之后添加：

```rust
pub mod fast;
```

（rms_norm 等具体函数通过 `mlx::fast::rms_norm` 访问；用户希望直接 `mlx::rms_norm` 时走 `pub use fast::*`，但目前先用模块路径，到 Task 6 再 re-export）

- [ ] **Step 1.11: 编译并运行测试**

Run: `cargo test --test p2b_fast`
Expected: PASS — 两个 rms_norm 测试通过（输出与期望值在 1e-3 容差内一致）。

- [ ] **Step 1.12: 跑全套 Rust 检查**

Run sequentially:
```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过，无 warning。

- [ ] **Step 1.13: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/fast.h \
        mlx-sys/shim/src/fast.cc \
        mlx-sys/src/bridge/fast.rs \
        mlx-sys/src/bridge/mod.rs \
        mlx-sys/src/lib.rs \
        mlx-sys/build.rs \
        mlx/src/fast.rs \
        mlx/src/lib.rs \
        mlx/tests/p2b_fast.rs
git commit -m "feat(p2b): scaffold fast module + rms_norm (3 layers, 2 tests)"
```

---

## Task 2: layer_norm

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h`（追加声明）
- Modify: `mlx-sys/shim/src/fast.cc`（追加实现）
- Modify: `mlx-sys/src/bridge/fast.rs`（追加 FFI）
- Modify: `mlx/src/fast.rs`（追加安全函数）
- Modify: `mlx/tests/p2b_fast.rs`（追加测试）

- [ ] **Step 2.1: 写失败的集成测试**

在 `mlx/tests/p2b_fast.rs` 末尾追加：

```rust
#[test]
fn layer_norm_no_weight_no_bias_known_values() {
    // x = [[1.0, 2.0, 3.0, 4.0]], shape [1, 4]
    // mean = 2.5; var = 1.25; sqrt(1.25 + 1e-5) ≈ 1.11803
    // normalized = (x - 2.5) / 1.11803 ≈ [-1.34164, -0.44721, 0.44721, 1.34164]
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let out = fast::layer_norm(&x, None, None, 1e-5).expect("layer_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);

    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [-1.34164_f32, -0.44721, 0.44721, 1.34164];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "layer_norm[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn layer_norm_with_weight_and_bias() {
    // weight=[1,1,1,1], bias=[10,10,10,10] → output = normalized + 10
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let w = Array::from_slice(&[1.0_f32, 1.0, 1.0, 1.0], &[4]).expect("w");
    let b = Array::from_slice(&[10.0_f32, 10.0, 10.0, 10.0], &[4]).expect("b");
    let out = fast::layer_norm(&x, Some(&w), Some(&b), 1e-5).expect("layer_norm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [8.65836_f32, 9.55279, 10.44721, 11.34164];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "layer_norm_wb[{i}] = {got}, want {want}"
        );
    }
}
```

- [ ] **Step 2.2: 运行测试，确认失败**

Run: `cargo test --test p2b_fast --no-run`
Expected: 编译失败，提示 `fast::layer_norm` 不存在。

- [ ] **Step 2.3: shim 头追加 `fast_layer_norm`**

在 `mlx-sys/shim/include/cxx_mlx_shim/fast.h` 的 `fast_rms_norm` 声明之后追加：

```cpp
// layer_norm: weight=nullptr → no weight, bias=nullptr → no bias
std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,
    const MlxArray* bias,
    float eps);
```

- [ ] **Step 2.4: shim cc 追加 `fast_layer_norm`**

在 `mlx-sys/shim/src/fast.cc` 的 `fast_rms_norm` 实现之后追加：

```cpp
std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x, const MlxArray* weight, const MlxArray* bias, float eps) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::layer_norm(x, opt_arr(weight), opt_arr(bias), eps));
}
```

- [ ] **Step 2.5: 桥接追加 `fast_layer_norm`**

在 `mlx-sys/src/bridge/fast.rs` 的 `fast_rms_norm` FFI 之后追加：

```rust
        unsafe fn fast_layer_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            bias: *const MlxArray,
            eps: f32,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 2.6: 安全 API 追加 `layer_norm`**

在 `mlx/src/fast.rs` 的 `rms_norm` 之后追加：

```rust
/// Layer normalization with optional learned scale and bias.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let b = bias.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: w/b each null or borrow of an &Array valid for this call.
    let inner = unsafe { mlx_sys::fast::ffi::fast_layer_norm(x.as_inner(), w, b, eps) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 2.7: 测试通过**

Run: `cargo test --test p2b_fast`
Expected: 4 tests passed（rms_norm 2 + layer_norm 2）。

- [ ] **Step 2.8: Rust 检查**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 2.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/fast.h \
        mlx-sys/shim/src/fast.cc \
        mlx-sys/src/bridge/fast.rs \
        mlx/src/fast.rs \
        mlx/tests/p2b_fast.rs
git commit -m "feat(p2b): layer_norm (3 layers, 2 tests)"
```

---

## Task 3: rope（int offset 形态）

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h`
- Modify: `mlx-sys/shim/src/fast.cc`
- Modify: `mlx-sys/src/bridge/fast.rs`
- Modify: `mlx/src/fast.rs`
- Modify: `mlx/tests/p2b_fast.rs`

- [ ] **Step 3.1: 写失败的集成测试**

在 `mlx/tests/p2b_fast.rs` 末尾追加：

```rust
#[test]
fn rope_basic_shape_finite() {
    // 最简验证：base=Some(10000), traditional=false, offset=0, freqs=None。
    // x: [B=1, H=1, S=4, D=8]，dims=8（旋转全部维度）
    // 主要验证形状不变 + 输出有限 + 与输入显著不同（确实做了旋转）
    let total: usize = 1 * 1 * 4 * 8;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::from_slice(&data, &[1, 1, 4, 8]).expect("x");
    let out = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope");

    assert_eq!(out.shape().as_slice(), &[1, 1, 4, 8]);
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert_eq!(v.len(), total);
    for x in &v {
        assert!(x.is_finite(), "non-finite value in rope output: {x}");
    }
    // 第 0 个位置（pos=0）的旋转应当是单位变换：cos(0)=1, sin(0)=0 → 输出 = 输入
    // 但是实际 MLX 实现里，pos=0 的旋转不是恒等，因为 freq 的角度也跟 dim_idx 走。
    // 这里只验证整体不全等于输入：
    let in_v = x.to_vec::<f32>().expect("x.to_vec");
    let mut differ = 0;
    for (a, b) in v.iter().zip(in_v.iter()) {
        if (a - b).abs() > 1e-6 {
            differ += 1;
        }
    }
    assert!(differ > 0, "rope should rotate at least some elements");
}

#[test]
fn rope_offset_shifts_output() {
    // 同样输入，offset=0 vs offset=4 应当产生不同的输出（实际是把 pos 位置移了 4 步）。
    let total: usize = 1 * 1 * 4 * 8;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::from_slice(&data, &[1, 1, 4, 8]).expect("x");

    let out0 = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope_0");
    let out4 = fast::rope(&x, 8, false, Some(10000.0), 1.0, 4, None).expect("rope_4");

    let v0: Vec<f32> = out0.to_vec().expect("to_vec0");
    let v4: Vec<f32> = out4.to_vec().expect("to_vec4");

    let mut differ = 0;
    for (a, b) in v0.iter().zip(v4.iter()) {
        if (a - b).abs() > 1e-4 {
            differ += 1;
        }
    }
    assert!(
        differ > 0,
        "different offsets should produce different rope outputs"
    );
}

#[test]
fn rope_traditional_differs_from_default() {
    // traditional=true 与 traditional=false 是不同的 rope 排布方式。
    let total: usize = 1 * 1 * 4 * 8;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::from_slice(&data, &[1, 1, 4, 8]).expect("x");

    let out_f = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope_f");
    let out_t = fast::rope(&x, 8, true, Some(10000.0), 1.0, 0, None).expect("rope_t");

    let vf: Vec<f32> = out_f.to_vec().expect("to_vec_f");
    let vt: Vec<f32> = out_t.to_vec().expect("to_vec_t");

    let mut differ = 0;
    for (a, b) in vf.iter().zip(vt.iter()) {
        if (a - b).abs() > 1e-4 {
            differ += 1;
        }
    }
    assert!(differ > 0, "traditional vs non-traditional should differ");
}
```

- [ ] **Step 3.2: 运行测试，确认失败**

Run: `cargo test --test p2b_fast --no-run`
Expected: 编译失败，提示 `fast::rope` 不存在。

- [ ] **Step 3.3: shim 头追加 `fast_rope`**

在 `mlx-sys/shim/include/cxx_mlx_shim/fast.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// rope (int offset)
//   has_base=false → std::nullopt (MLX 内部回落到默认 base 处理逻辑)
//   freqs=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    int32_t offset,
    const MlxArray* freqs);
```

- [ ] **Step 3.4: shim cc 追加 `fast_rope`**

在 `mlx-sys/shim/src/fast.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, int32_t offset,
    const MlxArray* freqs) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs)));
}
```

- [ ] **Step 3.5: 桥接追加 `fast_rope`**

在 `mlx-sys/src/bridge/fast.rs` 的 `fast_layer_norm` 之后追加：

```rust
        unsafe fn fast_rope(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            has_base: bool,
            base: f32,
            scale: f32,
            offset: i32,
            freqs: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 3.6: 安全 API 追加 `rope`**

在 `mlx/src/fast.rs` 的 `layer_norm` 之后追加：

```rust
/// Rotary position embedding with a scalar offset (single-stream decode
/// or fixed-context prefill).
///
/// `base=None` requires `freqs=Some(_)` (precomputed frequencies);
/// `base=Some(_)` typically pairs with `freqs=None`. MLX validates the
/// combination and raises if both are missing.
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
) -> Result<Array> {
    let (has_base, base_val) = base.map_or((false, 0.0), |b| (true, b));
    let f = freqs.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: f is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_rope(
            x.as_inner(), dims, traditional, has_base, base_val, scale, offset, f,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 3.7: 测试通过**

Run: `cargo test --test p2b_fast`
Expected: 7 tests passed（前 4 + rope 3）。

- [ ] **Step 3.8: Rust 检查**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 3.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/fast.h \
        mlx-sys/shim/src/fast.cc \
        mlx-sys/src/bridge/fast.rs \
        mlx/src/fast.rs \
        mlx/tests/p2b_fast.rs
git commit -m "feat(p2b): rope (int offset, 3 tests)"
```

---

## Task 4: rope_with_array_offset（变长批量）

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h`
- Modify: `mlx-sys/shim/src/fast.cc`
- Modify: `mlx-sys/src/bridge/fast.rs`
- Modify: `mlx/src/fast.rs`
- Modify: `mlx/tests/p2b_fast.rs`

- [ ] **Step 4.1: 写失败的集成测试**

在 `mlx/tests/p2b_fast.rs` 末尾追加：

```rust
#[test]
fn rope_with_array_offset_per_batch_offsets() {
    // batch=2，offsets=[0, 4]：行 0 用 offset=0，行 1 用 offset=4。
    // 期望：第 0 行结果 == fast::rope(...offset=0)，第 1 行结果 == fast::rope(...offset=4)
    // 简化验证：用 batch=2 的输入，比较与单一 offset 路径的一致性。

    let per_row: usize = 1 * 4 * 8; // H=1, S=4, D=8
    // batch 0: 全 0.01 增长
    let row0: Vec<f32> = (0..per_row).map(|i| (i as f32) * 0.01).collect();
    // batch 1: 同样的 pattern（让两个 batch 共享数据，方便比对）
    let row1 = row0.clone();

    let mut combined: Vec<f32> = Vec::with_capacity(per_row * 2);
    combined.extend_from_slice(&row0);
    combined.extend_from_slice(&row1);
    let x_batched = Array::from_slice(&combined, &[2, 1, 4, 8]).expect("x_batched");

    let offsets = Array::from_slice(&[0_i32, 4], &[2]).expect("offsets");
    let out = fast::rope_with_array_offset(
        &x_batched, 8, false, Some(10000.0), 1.0, &offsets, None,
    )
    .expect("rope_array");
    assert_eq!(out.shape().as_slice(), &[2, 1, 4, 8]);

    // 单独用 int offset 路径计算两个参考：
    let x_single = Array::from_slice(&row0, &[1, 1, 4, 8]).expect("x_single");
    let ref_0 = fast::rope(&x_single, 8, false, Some(10000.0), 1.0, 0, None).expect("ref0");
    let ref_4 = fast::rope(&x_single, 8, false, Some(10000.0), 1.0, 4, None).expect("ref4");

    let v_out: Vec<f32> = out.to_vec().expect("to_vec");
    let v_ref0: Vec<f32> = ref_0.to_vec().expect("ref0_vec");
    let v_ref4: Vec<f32> = ref_4.to_vec().expect("ref4_vec");

    // 第 0 个 batch 应当与 offset=0 参考一致
    for i in 0..per_row {
        let a = v_out[i];
        let b = v_ref0[i];
        assert!(
            (a - b).abs() < 1e-4,
            "batch0[{i}] = {a}, ref offset=0 = {b}"
        );
    }
    // 第 1 个 batch 应当与 offset=4 参考一致
    for i in 0..per_row {
        let a = v_out[per_row + i];
        let b = v_ref4[i];
        assert!(
            (a - b).abs() < 1e-4,
            "batch1[{i}] = {a}, ref offset=4 = {b}"
        );
    }
}
```

- [ ] **Step 4.2: 运行测试，确认失败**

Run: `cargo test --test p2b_fast --no-run`
Expected: 编译失败，提示 `fast::rope_with_array_offset` 不存在。

- [ ] **Step 4.3: shim 头追加 `fast_rope_with_array_offset`**

在 `mlx-sys/shim/include/cxx_mlx_shim/fast.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// rope (array offset) — 同上 base/freqs 处理；offset 改为引用 array
std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    const MlxArray& offset,
    const MlxArray* freqs);
```

- [ ] **Step 4.4: shim cc 追加 `fast_rope_with_array_offset`**

在 `mlx-sys/shim/src/fast.cc` 末尾 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, const MlxArray& offset,
    const MlxArray* freqs) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs)));
}
```

- [ ] **Step 4.5: 桥接追加 `fast_rope_with_array_offset`**

在 `mlx-sys/src/bridge/fast.rs` 的 `fast_rope` 之后追加：

```rust
        unsafe fn fast_rope_with_array_offset(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            has_base: bool,
            base: f32,
            scale: f32,
            offset: &MlxArray,
            freqs: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 4.6: 安全 API 追加 `rope_with_array_offset`**

在 `mlx/src/fast.rs` 的 `rope` 之后追加：

```rust
/// RoPE with per-batch-row offsets — for variable-length batched inference.
/// `offset` shape: `[batch]`, dtype `i32`.
pub fn rope_with_array_offset(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
) -> Result<Array> {
    let (has_base, base_val) = base.map_or((false, 0.0), |b| (true, b));
    let f = freqs.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: f is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_rope_with_array_offset(
            x.as_inner(), dims, traditional, has_base, base_val, scale,
            offset.as_inner(), f,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4.7: 测试通过**

Run: `cargo test --test p2b_fast`
Expected: 8 tests passed（前 7 + array_offset 1）。

- [ ] **Step 4.8: Rust 检查**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 4.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/fast.h \
        mlx-sys/shim/src/fast.cc \
        mlx-sys/src/bridge/fast.rs \
        mlx/src/fast.rs \
        mlx/tests/p2b_fast.rs
git commit -m "feat(p2b): rope with per-batch array offset (1 test)"
```

---

## Task 5: scaled_dot_product_attention

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h`
- Modify: `mlx-sys/shim/src/fast.cc`
- Modify: `mlx-sys/src/bridge/fast.rs`
- Modify: `mlx/src/fast.rs`
- Modify: `mlx/tests/p2b_fast.rs`

- [ ] **Step 5.1: 写失败的集成测试**

在 `mlx/tests/p2b_fast.rs` 末尾追加：

```rust
#[test]
fn sdpa_no_mask_matches_manual_reference() {
    // Q=K=V=I (4×4 identity)，scale=1，无 mask。
    // softmax(I @ I.T) = softmax(I) → 每行 [exp(1), 1, 1, 1] / (exp(1) + 3)
    // weights @ V (=I) = weights 本身
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::from_slice(&data, &[n as i32, n as i32]).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    let out = fast::scaled_dot_product_attention(&q, &k, &v, 1.0, "", None, None)
        .expect("sdpa");
    assert_eq!(out.shape().as_slice(), &[1, 1, n as i32, n as i32]);

    let result: Vec<f32> = out.to_vec().expect("to_vec");
    let e = std::f32::consts::E;
    let norm = e + 3.0;
    let expected_diag = e / norm;
    let expected_off = 1.0 / norm;

    for i in 0..n {
        for j in 0..n {
            let actual = result[i * n + j];
            let want = if i == j { expected_diag } else { expected_off };
            assert!(
                (actual - want).abs() < 1e-3,
                "sdpa[{i},{j}] = {actual}, want {want}"
            );
        }
    }
}

#[test]
fn sdpa_causal_mode_zeros_future_positions() {
    // mask_mode="causal"：因果掩码。weights @ V=I 时，第 i 行对位置 j>i 的注意力应为 0。
    // 用 Q=K=I, V=I, scale=1.0, causal 模式 → 输出第 i 行的 j>i 位置应为 0。
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::from_slice(&data, &[n as i32, n as i32]).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    let out = fast::scaled_dot_product_attention(&q, &k, &v, 1.0, "causal", None, None)
        .expect("sdpa");
    let result: Vec<f32> = out.to_vec().expect("to_vec");

    for i in 0..n {
        for j in (i + 1)..n {
            let val = result[i * n + j];
            assert!(
                val.abs() < 1e-5,
                "causal sdpa[{i},{j}] should be 0, got {val}"
            );
        }
    }
}

#[test]
fn sdpa_custom_mask_zeros_masked_positions() {
    // 提供自定义 mask（全 -inf 的右上三角等价 causal）。验证传 mask_arr 路径通。
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::from_slice(&data, &[n as i32, n as i32]).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    // additive mask shape [n, n]
    let mut mask_data = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if j > i {
                mask_data[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Array::from_slice(&mask_data, &[n as i32, n as i32]).expect("mask");

    let out = fast::scaled_dot_product_attention(
        &q, &k, &v, 1.0, "", Some(&mask), None,
    )
    .expect("sdpa");
    let result: Vec<f32> = out.to_vec().expect("to_vec");

    for i in 0..n {
        for j in (i + 1)..n {
            let val = result[i * n + j];
            assert!(
                val.abs() < 1e-5,
                "custom-mask sdpa[{i},{j}] should be 0, got {val}"
            );
        }
    }
}
```

- [ ] **Step 5.2: 运行测试，确认失败**

Run: `cargo test --test p2b_fast --no-run`
Expected: 编译失败，提示 `fast::scaled_dot_product_attention` 不存在。

- [ ] **Step 5.3: shim 头追加 `fast_scaled_dot_product_attention`**

在 `mlx-sys/shim/include/cxx_mlx_shim/fast.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// scaled_dot_product_attention
//   mask_mode: rust::Str → std::string  ("" 等价 MLX 默认值)
//   mask_arr=nullptr → std::nullopt
//   sinks=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& queries,
    const MlxArray& keys,
    const MlxArray& values,
    float scale,
    rust::Str mask_mode,
    const MlxArray* mask_arr,
    const MlxArray* sinks);
```

- [ ] **Step 5.4: shim cc 追加 `fast_scaled_dot_product_attention`**

在 `mlx-sys/shim/src/fast.cc` 末尾 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& queries, const MlxArray& keys, const MlxArray& values,
    float scale, rust::Str mask_mode,
    const MlxArray* mask_arr, const MlxArray* sinks) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::scaled_dot_product_attention(
          queries, keys, values, scale,
          std::string(mask_mode),
          opt_arr(mask_arr),
          opt_arr(sinks)));
}
```

- [ ] **Step 5.5: 桥接追加 `fast_scaled_dot_product_attention`**

在 `mlx-sys/src/bridge/fast.rs` 的 `fast_rope_with_array_offset` 之后追加：

```rust
        unsafe fn fast_scaled_dot_product_attention(
            queries: &MlxArray,
            keys: &MlxArray,
            values: &MlxArray,
            scale: f32,
            mask_mode: &str,
            mask_arr: *const MlxArray,
            sinks: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 5.6: 安全 API 追加 `scaled_dot_product_attention`**

在 `mlx/src/fast.rs` 的 `rope_with_array_offset` 之后追加：

```rust
/// Fused scaled dot-product attention: `softmax(Q @ K.T * scale + mask) @ V`.
///
/// `mask_mode`:
/// - `""` — no implicit mask (default if `mask_arr=None`)
/// - `"causal"` — standard causal mask
/// - `"chunked_causal"` — block-causal for chunked prefill
///
/// `mask_arr=Some(_)` supplies a custom additive mask (broadcastable).
/// `sinks=Some(_)` enables attention sinks (StreamingLLM-style).
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    sinks: Option<&Array>,
) -> Result<Array> {
    let m = mask_arr.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let s = sinks.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: m/s each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_scaled_dot_product_attention(
            queries.as_inner(), keys.as_inner(), values.as_inner(),
            scale, mask_mode, m, s,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 5.7: 测试通过**

Run: `cargo test --test p2b_fast`
Expected: 11 tests passed（前 8 + sdpa 3）。

- [ ] **Step 5.8: Rust 检查**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 5.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/fast.h \
        mlx-sys/shim/src/fast.cc \
        mlx-sys/src/bridge/fast.rs \
        mlx/src/fast.rs \
        mlx/tests/p2b_fast.rs
git commit -m "feat(p2b): scaled_dot_product_attention (3 tests)"
```

---

## Task 6: README + lib.rs re-exports + 全套验证

**目的**：把 P2b 5 个函数 re-export 到 `mlx::*`，更新 README 进度，跑完整 workspace 检查。

**Files:**
- Modify: `mlx/src/lib.rs`
- Modify: `README.md`

- [ ] **Step 6.1: 在 `mlx/src/lib.rs` 的 `pub mod fast;` 后追加 re-export**

打开 `mlx/src/lib.rs`，把 `pub mod fast;` 那一行改为：

```rust
pub mod fast;
pub use fast::{
    layer_norm, rms_norm, rope, rope_with_array_offset, scaled_dot_product_attention,
};
```

- [ ] **Step 6.2: 在测试文件追加一条 re-export 验证测试**

在 `mlx/tests/p2b_fast.rs` 末尾追加：

```rust
#[test]
fn top_level_re_exports_work() {
    // 通过 mlx::rms_norm 直接调用（验证 re-export 可达）
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let out = mlx::rms_norm(&x, None, 1e-5).expect("rms_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);
}
```

- [ ] **Step 6.3: 运行所有 fast 测试**

Run: `cargo test --test p2b_fast`
Expected: 12 tests passed。

- [ ] **Step 6.4: 跑完整 workspace 测试**

Run: `cargo test --workspace --all-features`
Expected: 所有现有测试 + 12 个新 P2b 测试全部通过。

- [ ] **Step 6.5: 更新 `README.md`**

在 `README.md` 中找到记录已完成阶段的地方（多半是 "Status" / "进度" / "Roadmap" 章节，参考 P2a 的 commit `50a2d0a`）。把 P2b 状态从未完成改为完成，例如：

- 在已完成模块清单中加一行 `mlx::fast` 或 "P2b: rms_norm / layer_norm / rope / sdpa"
- 在 Roadmap 表里把 P2b 行从 "Pending" 改为 "Done (5 fns, 12 tests)"

打开 `README.md`，定位到上次更新位置，按现有格式插入 P2b 完成条目。如果有"接下来做什么"提示，把 P2b 移走、把 P2c 留下。

提交前用 `git diff README.md` 确认改动符合现有风格。

- [ ] **Step 6.6: 跑全套 Rust 检查**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过，无 warning，所有测试 PASS。

- [ ] **Step 6.7: 提交**

```bash
git add mlx/src/lib.rs mlx/tests/p2b_fast.rs README.md
git commit -m "feat(p2b): re-export fast ops at crate root + README progress"
```

- [ ] **Step 6.8: 最终 git log 与 commit 数核对**

Run: `git log --oneline | head -10`
Expected: 看到 6 个 P2b 相关 commit（Task 1–6 各 1 个），以及之前的 P2a commit。

---

## 自检（plan 作者自检结果）

**Spec 覆盖**：
- ✅ rms_norm（Task 1）
- ✅ layer_norm（Task 2）
- ✅ rope int offset（Task 3）
- ✅ rope_with_array_offset（Task 4）
- ✅ scaled_dot_product_attention（Task 5）
- ✅ shim/bridge/safe 三层结构（每个 task 都覆盖）
- ✅ 所有 spec 中的可选参数处理（`Option<&Array>` → `*const`，`Option<f32>` → `(bool, f32)`，mask_mode 用 `&str`）
- ✅ 错误传播链路（cxx::Exception → Error::from）
- ✅ 测试策略（每个函数 ≥1 数值正确性测试 + 形状/边界验证）
- ✅ 文件清单（shim/bridge/safe + build.rs/mod.rs/lib.rs 接线）
- ✅ README 更新（Task 6）

**类型一致性**：
- 所有任务用 `Array::from_inner(inner)`（不是 `from_unique_ptr`）
- 所有任务用 `a.as_inner()`（带 `as *const _` 转裸指针）
- 所有任务用 `Error::from` + `?`
- bridge 全部 `unsafe fn`（含裸指针）
- 所有 `MlxArray` 跨桥接通过 `type MlxArray = crate::bridge::array::ffi::MlxArray;` 共享
