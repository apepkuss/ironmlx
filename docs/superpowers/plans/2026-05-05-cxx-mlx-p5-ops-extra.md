# cxx-mlx P5 · Ops 补漏 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `mlx::core::ops.h` 中尚未绑定的 8 个 matmul/contraction 家族算子提供 Rust 安全绑定（tensordot×2 + outer + inner_product + addmm + block_masked_mm + gather_mm + segmented_mm）。

**Architecture:** 三层结构（沿用 P0–P4），**无新模块** —— 扩展现有 `mlx-sys/shim/{include,src}/array.{h,cc}` + `mlx-sys/src/bridge/array.rs` + `mlx/src/ops/matmul.rs`。`Option<&Array>` → `*const MlxArray`（含 P3/P4 helpers），`Vec<int>` → `rust::Slice<const int32_t>`，无新 cxx idiom。

**Tech Stack:** Rust 1.82+，cxx 1.0，MLX C++ 共享安装，cargo nightly fmt + clippy + release build。

**Spec reference:** `docs/superpowers/specs/2026-05-05-cxx-mlx-p5-ops-extra-design.md`

---

## 关键背景信息（实施者必读）

### 项目三层结构

- **shim 层**：`mlx-sys/shim/include/cxx_mlx_shim/array.{h,cc}` 已含 P1a/P1b1/P1b2a/P1b2b 段落，P5 追加 `// === P5 ops extensions ===` 段
- **桥接层**：`mlx-sys/src/bridge/array.rs` —— free function 风格（与 P0–P4 一致），P5 追加段落
- **安全层**：`mlx/src/ops/matmul.rs` —— 已有 `pub fn matmul()`，P5 追加 8 个公开函数

### `array.cc` 现状（影响 Task 1）

`mlx-sys/shim/src/array.cc` 顶部有一个匿名 namespace 含 `array_from_typed` 模板和 `dtype_from_u8`（预 P3/P4 helpers 重构时代）。**未** include `shim_helpers.h`。

**P5 Task 1 需要在 array.cc 顶部追加 `#include "cxx_mlx_shim/shim_helpers.h"`**，新增 P5 shim 函数使用 `cxx_mlx::helpers::opt_arr`（避免重复定义 `opt_arr` lambda）。

### cxx 类型映射（全部 P2/P3/P4 已建立）

| MLX C++ | shim 暴露 | bridge 类型 | Rust 端调用 |
|---------|----------|-------------|-------------|
| `std::optional<array>` | `*const MlxArray`（nullptr=None） | `*const MlxArray`（unsafe fn 必须） | `Option<&Array>::map_or(null, |a| a.as_inner() as *const _)` |
| `std::vector<int>` 入参 | `rust::Slice<const int32_t>` | `&[i32]` | `&[i32]` |
| `int axis`/`int block_size` | `int32_t` | `i32` | `i32` |
| `float alpha`/`float beta` | `float` | `f32` | `f32` |
| `bool sorted_indices` | `bool` | `bool` | `bool` |
| 多 overload | 不同函数名（`tensordot_axis` / `tensordot_axes`） | 各自独立 fn | 各自独立 pub fn |

### 已有 API 引用点

- `Array::from_inner(inner: cxx::UniquePtr<...>) -> Self`（[mlx/src/array.rs:11](mlx/src/array.rs#L11)）
- `Array::as_inner(&self) -> &mlx_sys::array::ffi::MlxArray`（[mlx/src/array.rs:139](mlx/src/array.rs#L139)）
- `cxx_mlx::helpers::opt_arr` 在 `mlx-sys/shim/include/cxx_mlx_shim/shim_helpers.h`（P4 Task 1 已抽取）

### 强制检查（CLAUDE.md + 项目约定，每次 commit 前必跑）

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```

### MLX 上游 API（来自 `${MLX_DIR}/include/mlx/ops.h`）

```cpp
namespace mlx::core {

array tensordot(const array& a, const array& b, int axis = 2, ...);
array tensordot(const array& a, const array& b,
                const std::vector<int>& axes_a, const std::vector<int>& axes_b, ...);

array outer(const array& a, const array& b, ...);
array inner(const array& a, const array& b, ...);

array addmm(array c, array a, array b,
            const float& alpha = 1.f, const float& beta = 1.f, ...);

array block_masked_mm(array a, array b, int block_size,
                      std::optional<array> mask_out = std::nullopt,
                      std::optional<array> mask_lhs = std::nullopt,
                      std::optional<array> mask_rhs = std::nullopt, ...);

array gather_mm(array a, array b,
                std::optional<array> lhs_indices = std::nullopt,
                std::optional<array> rhs_indices = std::nullopt,
                bool sorted_indices = false, ...);

array segmented_mm(array a, array b, array segments, ...);

}  // namespace mlx::core
```

`StreamOrDevice s = {}` 全不传（默认 caller 线程 stream）。

---

## 文件清单

### 修改
- `mlx-sys/shim/include/cxx_mlx_shim/array.h`（追加 P5 段，8 声明）
- `mlx-sys/shim/src/array.cc`（顶部加 `#include "cxx_mlx_shim/shim_helpers.h"` + 末尾追加 P5 段，8 实现）
- `mlx-sys/src/bridge/array.rs`（追加 P5 段，8 FFI）
- `mlx/src/ops/matmul.rs`（末尾追加 8 公开函数）
- `mlx/src/ops/mod.rs`（扩展 `pub use matmul::{...}` 列表，在 Task 4）
- `README.md`（在 Task 4）

### 新建
- `mlx/tests/p5_ops_extra.rs`（集成测试）

---

## Task 1: tensordot×2 + outer + inner_product

**目的**：建立 P5 段框架（`#include shim_helpers.h` + 段落注释），实现 4 个简单 contraction 函数。

**Files:**
- Create: `mlx/tests/p5_ops_extra.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx/src/ops/matmul.rs`

- [ ] **Step 1.1: 写失败的集成测试**

将以下完整内容写入 `mlx/tests/p5_ops_extra.rs`：

```rust
//! Integration tests for P5 ops extensions (matmul family).

use mlx::ops::{inner_product, outer, tensordot, tensordot_axes};
use mlx::Array;

#[test]
fn tensordot_axis_matches_matmul_for_2d() {
    // 2D tensordot(a, b, 1) 等价于 matmul(a, b)
    // a: [2, 3], b: [3, 4] → tensordot=matmul=[2, 4]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(
        &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        &[3, 4],
    )
    .expect("b");
    let td = tensordot(&a, &b, 1).expect("tensordot");
    assert_eq!(td.shape().as_slice(), &[2, 4]);
    let mm = a.matmul(&b).expect("matmul");
    let v_td: Vec<f32> = td.to_vec().expect("td vec");
    let v_mm: Vec<f32> = mm.to_vec().expect("mm vec");
    for (t, m) in v_td.iter().zip(&v_mm) {
        assert!((t - m).abs() < 1e-4, "tensordot {t} != matmul {m}");
    }
}

#[test]
fn tensordot_axes_explicit_contraction() {
    // a: [2, 3], b: [3, 4], 收缩 a 的 axis 1 与 b 的 axis 0 → [2, 4]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(
        &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
    )
    .expect("b");
    let td = tensordot_axes(&a, &b, &[1], &[0]).expect("tensordot_axes");
    assert_eq!(td.shape().as_slice(), &[2, 4]);
}

#[test]
fn outer_product_shape_and_values() {
    // outer([a0,a1,a2], [b0,b1]) → [[a0*b0, a0*b1], [a1*b0, a1*b1], [a2*b0, a2*b1]]
    let a = Array::from_slice(&[2.0_f32, 3.0, 5.0], &[3]).expect("a");
    let b = Array::from_slice(&[7.0_f32, 11.0], &[2]).expect("b");
    let o = outer(&a, &b).expect("outer");
    assert_eq!(o.shape().as_slice(), &[3, 2]);
    let v: Vec<f32> = o.to_vec().expect("vec");
    let expected = [
        2.0_f32 * 7.0, 2.0 * 11.0,
        3.0 * 7.0, 3.0 * 11.0,
        5.0 * 7.0, 5.0 * 11.0,
    ];
    for (got, want) in v.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "outer: got {got}, want {want}");
    }
}

#[test]
fn inner_product_dot_scalar() {
    // inner_product([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("a");
    let b = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]).expect("b");
    let dot = inner_product(&a, &b).expect("inner");
    let v: Vec<f32> = dot.to_vec().expect("vec");
    assert_eq!(v.len(), 1);
    assert!((v[0] - 32.0).abs() < 1e-4, "inner_product = {}, want 32", v[0]);
}
```

- [ ] **Step 1.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra --no-run`
Expected: 编译失败，`mlx::ops::tensordot` 等未定义。

- [ ] **Step 1.3: shim 头追加 P5 段（4 个声明）**

打开 `mlx-sys/shim/include/cxx_mlx_shim/array.h`，在最后一个 `}  // namespace cxx_mlx` 之前追加：

```cpp
// === P5 ops extensions: matmul family ===

std::unique_ptr<MlxArray> tensordot_axis(
    const MlxArray& a, const MlxArray& b, int32_t axis);

std::unique_ptr<MlxArray> tensordot_axes(
    const MlxArray& a, const MlxArray& b,
    rust::Slice<const int32_t> axes_a,
    rust::Slice<const int32_t> axes_b);

std::unique_ptr<MlxArray> outer(const MlxArray& a, const MlxArray& b);

std::unique_ptr<MlxArray> inner(const MlxArray& a, const MlxArray& b);
```

- [ ] **Step 1.4: shim cc 追加 include + 4 个实现**

打开 `mlx-sys/shim/src/array.cc`：

**(a)** 在文件顶部已有的 include 区段末尾（`#include "mlx/transforms.h"` 之后或同样的相对位置）追加：

```cpp
#include "cxx_mlx_shim/shim_helpers.h"
```

**(b)** 在最后一个 `}  // namespace cxx_mlx` 之前追加：

```cpp
// === P5 ops extensions ===

std::unique_ptr<MlxArray> tensordot_axis(
    const MlxArray& a, const MlxArray& b, int32_t axis) {
  return std::make_unique<MlxArray>(mlx::core::tensordot(a, b, axis));
}

std::unique_ptr<MlxArray> tensordot_axes(
    const MlxArray& a, const MlxArray& b,
    rust::Slice<const int32_t> axes_a,
    rust::Slice<const int32_t> axes_b) {
  std::vector<int> va(axes_a.begin(), axes_a.end());
  std::vector<int> vb(axes_b.begin(), axes_b.end());
  return std::make_unique<MlxArray>(mlx::core::tensordot(a, b, va, vb));
}

std::unique_ptr<MlxArray> outer(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::outer(a, b));
}

std::unique_ptr<MlxArray> inner(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::inner(a, b));
}
```

- [ ] **Step 1.5: bridge 追加 P5 段（4 个 FFI）**

打开 `mlx-sys/src/bridge/array.rs`，在 `unsafe extern "C++"` 块末尾（最后的 `}` 之前）追加：

```rust
        // === P5 ops extensions ===
        fn tensordot_axis(
            a: &MlxArray, b: &MlxArray, axis: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn tensordot_axes(
            a: &MlxArray, b: &MlxArray,
            axes_a: &[i32], axes_b: &[i32],
        ) -> Result<UniquePtr<MlxArray>>;

        fn outer(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;

        fn inner(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
```

注：4 个函数都不含裸指针参数，**不需 `unsafe fn`**。

- [ ] **Step 1.6: 安全 API 追加 4 个函数**

打开 `mlx/src/ops/matmul.rs`，在文件末尾（现有 `pub fn matmul` 之后）追加：

```rust
// ===== P5 ops extensions =====

/// Tensor contraction over the last `axis` dims of `a` and first `axis` dims of `b`.
///
/// For 2D arrays with `axis=1`, this is equivalent to `a.matmul(b)`.
pub fn tensordot(a: &Array, b: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axis(a.as_inner(), b.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Tensor contraction over arbitrary axes pairs.
///
/// Contracts `a` along `axes_a` and `b` along `axes_b`. The two axes lists
/// must have the same length, and `a.shape[axes_a[i]] == b.shape[axes_b[i]]`
/// for each i.
pub fn tensordot_axes(
    a: &Array,
    b: &Array,
    axes_a: &[i32],
    axes_b: &[i32],
) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axes(
        a.as_inner(),
        b.as_inner(),
        axes_a,
        axes_b,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Outer product of two 1-D vectors. For `a` of shape `[N]` and `b` of shape
/// `[M]`, returns shape `[N, M]` with `out[i, j] = a[i] * b[j]`.
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::outer(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Inner (dot) product of two arrays. Renamed from MLX's `inner` to avoid
/// conflicting with the project's pervasive `as_inner` / `from_inner` naming.
pub fn inner_product(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::inner(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 1.7: 在 `mlx/src/ops/mod.rs` 中扩展 `pub use matmul::{...}`**

打开 `mlx/src/ops/mod.rs`，找到 `pub use matmul::matmul;` 那一行，把它改为（按字母排序）：

```rust
pub use matmul::{inner_product, matmul, outer, tensordot, tensordot_axes};
```

- [ ] **Step 1.8: 编译并运行测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra`
Expected: 4 tests passed。

- [ ] **Step 1.9: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过。

- [ ] **Step 1.10: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/array.h \
        mlx-sys/shim/src/array.cc \
        mlx-sys/src/bridge/array.rs \
        mlx/src/ops/matmul.rs \
        mlx/src/ops/mod.rs \
        mlx/tests/p5_ops_extra.rs
git commit -m "feat(p5): tensordot×2 + outer + inner_product (4 tests)"
```

---

## Task 2: addmm

**目的**：追加 `addmm(c, a, b, alpha, beta)` 函数。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx/src/ops/matmul.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/tests/p5_ops_extra.rs`

- [ ] **Step 2.1: 写失败的集成测试**

在 `mlx/tests/p5_ops_extra.rs` 末尾追加：

```rust
use mlx::ops::addmm;

#[test]
fn addmm_alpha_beta_formula() {
    // D = β*C + α*(A @ B)
    // A: [2, 3], B: [3, 2], C: [2, 2]
    // 设 α=2.0, β=3.0
    // A @ B 的第 [i,j] 元素 = sum_k A[i,k] * B[k,j]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).expect("b");
    let c = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0], &[2, 2]).expect("c");

    let d = addmm(&c, &a, &b, 2.0, 3.0).expect("addmm");
    assert_eq!(d.shape().as_slice(), &[2, 2]);

    // 手算参考:
    // A @ B = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]]
    //       = [[4, 5], [10, 11]]
    // D = 3*C + 2*(A@B)
    //   = [[3*10+2*4, 3*20+2*5], [3*30+2*10, 3*40+2*11]]
    //   = [[38, 70], [110, 142]]
    let v: Vec<f32> = d.to_vec().expect("vec");
    let expected = [38.0_f32, 70.0, 110.0, 142.0];
    for (got, want) in v.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "addmm: got {got}, want {want}");
    }
}
```

- [ ] **Step 2.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra --no-run`
Expected: 编译失败，`addmm` 未定义。

- [ ] **Step 2.3: shim 头追加 `addmm`**

在 `mlx-sys/shim/include/cxx_mlx_shim/array.h` P5 段末尾（最后一个 `}  // namespace cxx_mlx` 之前）追加：

```cpp
std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta);
```

- [ ] **Step 2.4: shim cc 追加 `addmm` 实现**

在 `mlx-sys/shim/src/array.cc` P5 段末尾（最后一个 `}  // namespace cxx_mlx` 之前）追加：

```cpp
std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta) {
  return std::make_unique<MlxArray>(mlx::core::addmm(c, a, b, alpha, beta));
}
```

- [ ] **Step 2.5: bridge 追加 `addmm` FFI**

在 `mlx-sys/src/bridge/array.rs` 的 P5 段末尾（最后的 `}` 之前）追加：

```rust
        fn addmm(
            c: &MlxArray, a: &MlxArray, b: &MlxArray,
            alpha: f32, beta: f32,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 2.6: 安全 API 追加 `addmm`**

在 `mlx/src/ops/matmul.rs` 末尾追加：

```rust
/// Compute `D = beta * C + alpha * (A @ B)` in a single fused kernel.
pub fn addmm(
    c: &Array,
    a: &Array,
    b: &Array,
    alpha: f32,
    beta: f32,
) -> Result<Array> {
    let inner = mlx_sys::array::ffi::addmm(
        c.as_inner(), a.as_inner(), b.as_inner(), alpha, beta,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 2.7: `mlx/src/ops/mod.rs` 扩展 re-export**

打开 `mlx/src/ops/mod.rs`，把现有的 `pub use matmul::{...}` 行改为：

```rust
pub use matmul::{addmm, inner_product, matmul, outer, tensordot, tensordot_axes};
```

- [ ] **Step 2.8: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra`
Expected: 5 tests passed（前 4 + addmm 1）。

- [ ] **Step 2.9: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 2.10: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/array.h \
        mlx-sys/shim/src/array.cc \
        mlx-sys/src/bridge/array.rs \
        mlx/src/ops/matmul.rs \
        mlx/src/ops/mod.rs \
        mlx/tests/p5_ops_extra.rs
git commit -m "feat(p5): addmm (1 test)"
```

---

## Task 3: block_masked_mm + gather_mm + segmented_mm

**目的**：追加 3 个 matmul 变体：含 optional 数组参数（block_masked_mm 3 个 mask、gather_mm 2 个 indices）和 required segments（segmented_mm）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx/src/ops/matmul.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/tests/p5_ops_extra.rs`

- [ ] **Step 3.1: 写失败的集成测试**

在 `mlx/tests/p5_ops_extra.rs` 末尾追加：

```rust
use mlx::ops::{block_masked_mm, gather_mm, segmented_mm};

#[test]
fn block_masked_mm_smoke_no_masks() {
    // 不传 mask 时退化为常规 matmul（块大小不影响结果）
    // A: [4, 4], B: [4, 4], block_size=2
    let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let a = Array::from_slice(&a_data, &[4, 4]).expect("a");
    let b = Array::from_slice(&b_data, &[4, 4]).expect("b");

    let result = block_masked_mm(&a, &b, 2, None, None, None);

    match result {
        Ok(out) => {
            assert_eq!(out.shape().as_slice(), &[4, 4]);
            match out.to_vec::<f32>() {
                Ok(v) => for x in &v { assert!(x.is_finite()); }
                Err(e) => {
                    let msg = format!("{e:?}");
                    assert!(
                        msg.contains("not yet supported") || msg.contains("NYI"),
                        "block_masked_mm eval non-NYI error: {msg}"
                    );
                }
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("not yet supported") || msg.contains("NYI"),
                "block_masked_mm construction non-NYI error: {msg}"
            );
        }
    }
}

#[test]
fn gather_mm_no_indices_smoke() {
    // gather_mm 不传 indices 时退化为常规 batched matmul
    // A: [2, 3, 4], B: [2, 4, 5] → [2, 3, 5]
    let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..40).map(|i| (i as f32) * 0.005).collect();
    let a = Array::from_slice(&a_data, &[2, 3, 4]).expect("a");
    let b = Array::from_slice(&b_data, &[2, 4, 5]).expect("b");

    let out = gather_mm(&a, &b, None, None, false).expect("gather_mm");
    assert_eq!(out.shape().as_slice(), &[2, 3, 5]);
    let v: Vec<f32> = out.to_vec().expect("vec");
    for x in &v {
        assert!(x.is_finite(), "gather_mm: non-finite {x}");
    }
}

#[test]
fn segmented_mm_smoke() {
    // segmented_mm: A: [B, M, K], B: [B, K, N], segments: i32 array
    // 构造最简单的 1-segment 退化情形
    // A: [1, 2, 3], B: [1, 3, 4], segments: [3] (单 segment 覆盖全部 K=3)
    let a_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05).collect();
    let a = Array::from_slice(&a_data, &[1, 2, 3]).expect("a");
    let b = Array::from_slice(&b_data, &[1, 3, 4]).expect("b");
    let segments = Array::from_slice(&[3_i32], &[1]).expect("segments");

    let result = segmented_mm(&a, &b, &segments);

    match result {
        Ok(out) => {
            // shape 通常 [B, M, num_segments, N] = [1, 2, 1, 4] 或类似
            let v: Vec<f32> = match out.to_vec::<f32>() {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("{e:?}");
                    assert!(
                        msg.contains("not yet supported") || msg.contains("NYI"),
                        "segmented_mm eval non-NYI: {msg}"
                    );
                    return;
                }
            };
            for x in &v {
                assert!(x.is_finite(), "segmented_mm: non-finite {x}");
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("not yet supported") || msg.contains("NYI"),
                "segmented_mm construction non-NYI error: {msg}"
            );
        }
    }
}
```

注：3 个测试均采用 NYI-tolerant 模式（与 P3 qqmm/P4 multivariate_normal 一致），避免 Metal 后端某些 op 未实现导致测试失败。

- [ ] **Step 3.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra --no-run`
Expected: 编译失败，3 个新函数未定义。

- [ ] **Step 3.3: shim 头追加 3 个声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/array.h` P5 段末尾追加：

```cpp
std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a, const MlxArray& b, int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs);

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a, const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices);

std::unique_ptr<MlxArray> segmented_mm(
    const MlxArray& a, const MlxArray& b, const MlxArray& segments);
```

- [ ] **Step 3.4: shim cc 追加 3 个实现**

在 `mlx-sys/shim/src/array.cc` P5 段末尾追加：

```cpp
std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a, const MlxArray& b, int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs) {
  return std::make_unique<MlxArray>(mlx::core::block_masked_mm(
      a, b, block_size,
      helpers::opt_arr(mask_out),
      helpers::opt_arr(mask_lhs),
      helpers::opt_arr(mask_rhs)));
}

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a, const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices) {
  return std::make_unique<MlxArray>(mlx::core::gather_mm(
      a, b,
      helpers::opt_arr(lhs_indices),
      helpers::opt_arr(rhs_indices),
      sorted_indices));
}

std::unique_ptr<MlxArray> segmented_mm(
    const MlxArray& a, const MlxArray& b, const MlxArray& segments) {
  return std::make_unique<MlxArray>(mlx::core::segmented_mm(a, b, segments));
}
```

注：使用 `helpers::opt_arr` 来自 `shim_helpers.h`（Task 1 已 include）。

- [ ] **Step 3.5: bridge 追加 3 个 FFI**

在 `mlx-sys/src/bridge/array.rs` 的 P5 段末尾追加：

```rust
        unsafe fn block_masked_mm(
            a: &MlxArray, b: &MlxArray, block_size: i32,
            mask_out: *const MlxArray,
            mask_lhs: *const MlxArray,
            mask_rhs: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn gather_mm(
            a: &MlxArray, b: &MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            sorted_indices: bool,
        ) -> Result<UniquePtr<MlxArray>>;

        fn segmented_mm(
            a: &MlxArray, b: &MlxArray, segments: &MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

注：含 `*const MlxArray` 参数的 fn（block_masked_mm、gather_mm）必须 `unsafe fn`；segmented_mm 全是引用，不需 unsafe。

- [ ] **Step 3.6: 安全 API 追加 3 个函数**

在 `mlx/src/ops/matmul.rs` 末尾追加：

```rust
/// Block-masked matrix product. Each of the 3 masks is optional and applies
/// at block granularity (`block_size`).
pub fn block_masked_mm(
    a: &Array,
    b: &Array,
    block_size: i32,
    mask_out: Option<&Array>,
    mask_lhs: Option<&Array>,
    mask_rhs: Option<&Array>,
) -> Result<Array> {
    let mo = mask_out.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let ml = mask_lhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let mr = mask_rhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    // SAFETY: mo/ml/mr each null or borrow of an &Array valid for this call.
    let inner = unsafe {
        mlx_sys::array::ffi::block_masked_mm(
            a.as_inner(), b.as_inner(), block_size, mo, ml, mr,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Matrix product with row-level gather. **Non-quantized** version.
/// For the quantized counterpart, see `mlx::quantization::gather_qmm` (P3).
pub fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let li = lhs_indices.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let ri = rhs_indices.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    // SAFETY: li/ri each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::array::ffi::gather_mm(
            a.as_inner(), b.as_inner(), li, ri, sorted_indices,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Matrix product with segmented inner dimension. `segments` is an i32
/// array describing how the inner dimension is partitioned across batches.
pub fn segmented_mm(a: &Array, b: &Array, segments: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::segmented_mm(
        a.as_inner(), b.as_inner(), segments.as_inner(),
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 3.7: `mlx/src/ops/mod.rs` 扩展 re-export**

打开 `mlx/src/ops/mod.rs`，把现有的 `pub use matmul::{...}` 行改为：

```rust
pub use matmul::{
    addmm, block_masked_mm, gather_mm, inner_product, matmul, outer,
    segmented_mm, tensordot, tensordot_axes,
};
```

- [ ] **Step 3.8: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra`
Expected: 8 tests passed（前 5 + 新 3）。注意 block_masked_mm/segmented_mm 测试若走 NYI 容错分支，不算失败。

- [ ] **Step 3.9: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 3.10: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/array.h \
        mlx-sys/shim/src/array.cc \
        mlx-sys/src/bridge/array.rs \
        mlx/src/ops/matmul.rs \
        mlx/src/ops/mod.rs \
        mlx/tests/p5_ops_extra.rs
git commit -m "feat(p5): block_masked_mm + gather_mm + segmented_mm (3 tests)"
```

---

## Task 4: README + final verify

**目的**：README 升级到 P5 完成，跑完整 workspace 检查。

**Files:**
- Modify: `mlx/tests/p5_ops_extra.rs`
- Modify: `README.md`

注：顶层 re-export 已在 Tasks 1-3 通过 `mlx/src/ops/mod.rs` 完成（`pub use matmul::*` → 已被 `pub use ops::*` 在 mlx/lib.rs 中传递到顶层）。Task 4 仅需补一个 re-export 验证测试 + README 更新。

- [ ] **Step 4.1: 在测试文件追加 re-export 验证测试**

在 `mlx/tests/p5_ops_extra.rs` 末尾追加：

```rust
#[test]
fn top_level_re_exports_work() {
    // 验证 P5 公开 API 通过 mlx::* 顶层访问
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("a");
    let b = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]).expect("b");
    // mlx::inner_product (再次确认 P1 ops re-export 链路通过到 P5 新增项)
    let dot = mlx::inner_product(&a, &b).expect("inner_product via mlx::*");
    let v: Vec<f32> = dot.to_vec().expect("vec");
    assert!((v[0] - 32.0).abs() < 1e-4);
}
```

- [ ] **Step 4.2: 运行所有 P5 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p5_ops_extra`
Expected: 9 tests passed（8 + re-export 1）。

- [ ] **Step 4.3: 跑完整 workspace 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --workspace --all-features`
Expected: 所有现有测试 + 9 个新 P5 测试全部通过。

- [ ] **Step 4.4: 更新 `README.md`**

`README.md` 的 status banner 与 Roadmap 升级。

**位置 A（Status banner，line 5 附近）**：当前文本类似：
```
**Status:** 🎉 **P4 complete** — `mlx::random` PRNG + 21 distribution functions ...
```

升级为：
```
**Status:** 🎉 **P5 complete** — `mlx::ops` 补漏完成 (tensordot×2, outer, inner_product, addmm, block_masked_mm, gather_mm, segmented_mm). MLX `ops.h` 公开 matmul 家族全覆盖。
```

**位置 B（Roadmap 表格）**：在 P4 行下追加 P5 行：
```
- ✅ **P4** — `random` (...) — 23 integration tests
- ✅ **P5** — `ops` 补漏 (8 matmul family ops) — 9 integration tests
```

- [ ] **Step 4.5: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过。

- [ ] **Step 4.6: 提交**

```bash
git add mlx/tests/p5_ops_extra.rs README.md
git commit -m "feat(p5): re-export verification + README progress"
```

- [ ] **Step 4.7: 最终 git log 与 commit 数核对**

Run: `git log --oneline 3bcc13d..HEAD`
Expected: 4 个 P5 feat commit + spec commit `3bcc13d` 在 base 之前。

---

## 自检（plan 作者自检结果）

**Spec 覆盖**：
- ✅ tensordot × 2 + outer + inner_product → Task 1
- ✅ addmm → Task 2
- ✅ block_masked_mm + gather_mm + segmented_mm → Task 3
- ✅ Re-export + README → Task 4
- ✅ 8 个 MLX 上游函数全覆盖

**类型一致性**：
- 所有 task 用 `Array::from_inner(inner)`
- 所有 task 用 `array.as_inner()` （含 `as *const _` 转裸指针）
- 所有 task 用 `Error::from` + `?`
- bridge 仅含裸指针的 fn 标 `unsafe fn`（block_masked_mm、gather_mm）
- 跨桥接 `MlxArray` 共享类型一致

**已知 placeholder**：
- 无 TBD/TODO/FIXME
- 每个 step 都有完整代码块或具体命令

**命名一致性**：
- shim/bridge 用 `tensordot_axis` / `tensordot_axes`（避 cxx 不支持 overload）
- 安全 API 用 `tensordot` / `tensordot_axes`（Rust 端无 overload 限制，按 spec 命名）
- `inner` shim/bridge 名称保持与 MLX 一致；安全 API 改名 `inner_product` 避 `as_inner`/`from_inner` 冲突
