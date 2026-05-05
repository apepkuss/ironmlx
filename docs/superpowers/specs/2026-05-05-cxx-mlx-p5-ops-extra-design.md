# cxx-mlx P5 · Ops 补漏 设计文档

**日期**: 2026-05-05
**状态**: 已批准，待实施
**前置**: P0 / P1 / P2a / P2b / P2c / P3 / P4 已完成（branch `p4-p6-infra` HEAD = `30a6570`）
**作者**: 通过 brainstorming 与 Boss 协作产出

## 目标

为 `mlx::core::ops.h` 中尚未绑定的 8 个 matmul/contraction 家族算子提供 Rust 安全绑定。完成后 MLX 上游 `ops.h` 公开函数全覆盖。

## 范围（按 MLX 上游 ops.h 公开 API）

| MLX API | Rust API | 复杂度 |
|---------|----------|--------|
| `tensordot(a, b, int axis=2)` | `tensordot(a, b, axis: i32)` | 简单 |
| `tensordot(a, b, vector<int> axes_a, vector<int> axes_b)` | `tensordot_axes(a, b, axes_a: &[i32], axes_b: &[i32])` | 简单 |
| `outer(a, b)` | `outer(a, b)` | 最简单 |
| `inner(a, b)` | `inner(a, b)` | 最简单 |
| `addmm(c, a, b, alpha=1.0, beta=1.0)` | `addmm(c, a, b, alpha: f32, beta: f32)` | 简单 |
| `block_masked_mm(a, b, block_size, mask_out?, mask_lhs?, mask_rhs?)` | `block_masked_mm(...)` | 中（3 optional arrays） |
| `gather_mm(a, b, lhs_indices?, rhs_indices?, sorted_indices)` | `gather_mm(...)` | 简单 |
| `segmented_mm(a, b, segments)` | `segmented_mm(a, b, segments)` | 最简单 |

**总计 8 个 Rust 公开函数**。

### 非目标

- **`matmul`** — P1b2a 已绑
- **`gather_qmm`** — P3 量化版本已绑（注：P5 是 `gather_mm` **非量化**版本）
- **`quantized_matmul` / `qqmm`** — P3 量化版本已绑
- **fast::* 算子** — P2b 已绑

## 设计原则

1. **完整性**：MLX 上游 ops.h 公开 matmul 家族全覆盖
2. **沿用既建模式**：cxx 编码模式直接复用 P2/P3/P4，无新 idiom
3. **不创建新模块**：8 个函数都属于 matmul 家族，扩展现有 `mlx::ops::matmul` 模块；shim 端扩展现有 `mlx-sys/shim/{include,src}/array.{h,cc}` 与 `mlx-sys/src/bridge/array.rs`（matmul 已在那里）

## 架构总览

```mermaid
graph TD
    A["mlx::ops::matmul - 已有 matmul()<br/>+ 8 新函数"] --> B[mlx_sys::array - 已有 array_matmul<br/>+ 8 新 FFI 在 // === P5 段落]
    B --> C[shim/array.{h,cc} - 已有 array_matmul<br/>+ 8 新 shim wrapper]
    C --> D["mlx::core::ops.h 8 个 matmul 家族函数"]
```

### 文件位置决策

不创建新模块。理由：
- 8 个函数都属于 matmul/contraction 家族，与 `matmul` 同源
- 现有 `mlx-sys/shim/{include,src}/array.{h,cc}` 已分段注释（P1a/P1b1/P1b2a/P1b2b）；P5 追加 `// === P5 ops extensions ===` 段保持一致
- 现有 `mlx/src/ops/matmul.rs` 当前仅 ~50 行；追加 ~120 行后总 ~170 行仍可读
- 新增模块会增加 build.rs/mod.rs/lib.rs 接线，但不带来组织收益

### 关键约束

- **不接受 stream 参数**：与 P1b/P2b/P3/P4 一致
- **`Option<&Array>`** → `*const MlxArray`（nullptr=None；P2b/P2c/P3/P4 已建立）
- **`Vec<int>` 入参**（tensordot axes）→ `rust::Slice<const int32_t>`（P4 random shape 模式）
- **scalar f32**（addmm alpha/beta）→ 直接 `f32`
- **bool**（gather_mm sorted_indices）→ 直接 `bool`
- **多 overload**（tensordot × 2）→ shim 端不同函数名（`tensordot_axis` / `tensordot_axes`）

## Shim 层设计

### `cxx_mlx_shim/array.h`（追加 8 个声明）

在 `mlx-sys/shim/include/cxx_mlx_shim/array.h` 末尾追加段落：

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

std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta);

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

### `shim/src/array.cc`（追加 8 个实现）

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

std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta) {
  return std::make_unique<MlxArray>(mlx::core::addmm(c, a, b, alpha, beta));
}

std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a, const MlxArray& b, int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs) {
  // shim_helpers.h 提供 helpers::opt_arr。本文件 array.cc 不直接 include shim_helpers.h
  // 当前；为了与现有 array.cc 风格一致（无 helpers 依赖），内联展开 opt_arr。
  // 如未来 array.cc 已 include shim_helpers.h，可直接用 helpers::opt_arr。
  auto opt = [](const MlxArray* p) -> std::optional<mlx::core::array> {
    return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
  };
  return std::make_unique<MlxArray>(mlx::core::block_masked_mm(
      a, b, block_size, opt(mask_out), opt(mask_lhs), opt(mask_rhs)));
}

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a, const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices) {
  auto opt = [](const MlxArray* p) -> std::optional<mlx::core::array> {
    return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
  };
  return std::make_unique<MlxArray>(mlx::core::gather_mm(
      a, b, opt(lhs_indices), opt(rhs_indices), sorted_indices));
}

std::unique_ptr<MlxArray> segmented_mm(
    const MlxArray& a, const MlxArray& b, const MlxArray& segments) {
  return std::make_unique<MlxArray>(mlx::core::segmented_mm(a, b, segments));
}
```

**实施时优先方案**：在 array.cc 顶部 `#include "cxx_mlx_shim/shim_helpers.h"`，随后用 `helpers::opt_arr` 替代上面的 inline lambda。这避免重复，与 P3/P4 风格统一。**plan 阶段会确认 array.cc 是否已 include shim_helpers.h；若未，在 P5 Task 1 中加上**。

### Shim 层设计要点

| 问题 | 处理 |
|------|------|
| `Option<&array>` cxx 不支持 | `*const MlxArray`，nullptr=None；shim 用 `helpers::opt_arr` 还原（P3/P4 一致） |
| `vector<int>` 入参 | `rust::Slice<const int32_t>` → 用 `std::vector<int>(slice.begin(), slice.end())` |
| `tensordot` 2 overload | shim 用 `tensordot_axis` / `tensordot_axes` 不同名（cxx 不支持 overload） |
| `addmm` 5 个 array 参数 | 全部 `const MlxArray&`（5 args 不超 clippy 7 阈值，无需 allow） |
| `block_masked_mm` 6 个参数（3 optional + int + 2 required） | `clippy::too_many_arguments`：6 个不超阈值；不加 allow |
| `gather_mm` 5 个参数 | 同上，无需 allow |
| MLX 抛异常 | shim 不 catch；cxx Result\<T\> 自动捕获 |

## Bridge 层设计

在 `mlx-sys/src/bridge/array.rs` 的 `unsafe extern "C++"` 块末尾追加（保持 P1a/P1b/... 已有段落分隔风格）：

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

        fn addmm(
            c: &MlxArray, a: &MlxArray, b: &MlxArray,
            alpha: f32, beta: f32,
        ) -> Result<UniquePtr<MlxArray>>;

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

### Bridge 层设计要点

- 含 `*const MlxArray` 的 fn（block_masked_mm、gather_mm）必须 `unsafe fn`（cxx 1.0 要求）
- 不含裸指针的 fn 不需 `unsafe`（tensordot × 2、outer、inner、addmm、segmented_mm）
- `&[i32]` 是 cxx 安全类型，对应 `rust::Slice<const int32_t>`

## 安全 API 层

在 `mlx/src/ops/matmul.rs` 末尾追加（保持现有 matmul() 在文件顶部）：

```rust
// ===== P5 ops extensions =====

/// Tensor contraction over the last `axis` dims of `a` and first `axis` dims of `b`.
pub fn tensordot(a: &Array, b: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axis(a.as_inner(), b.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Tensor contraction over arbitrary axes pairs.
pub fn tensordot_axes(
    a: &Array, b: &Array, axes_a: &[i32], axes_b: &[i32],
) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axes(
        a.as_inner(), b.as_inner(), axes_a, axes_b,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Outer product of two vectors.
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::outer(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Inner (dot) product of two vectors.
pub fn inner_product(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::inner(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute `D = beta * C + alpha * (A @ B)`.
pub fn addmm(
    c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32,
) -> Result<Array> {
    let inner = mlx_sys::array::ffi::addmm(
        c.as_inner(), a.as_inner(), b.as_inner(), alpha, beta,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Block-masked matrix product.
pub fn block_masked_mm(
    a: &Array, b: &Array, block_size: i32,
    mask_out: Option<&Array>,
    mask_lhs: Option<&Array>,
    mask_rhs: Option<&Array>,
) -> Result<Array> {
    let mo = mask_out.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let ml = mask_lhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let mr = mask_rhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    // SAFETY: mo/ml/mr each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::array::ffi::block_masked_mm(
            a.as_inner(), b.as_inner(), block_size, mo, ml, mr,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Matrix product with row-level gather (non-quantized MoE).
pub fn gather_mm(
    a: &Array, b: &Array,
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

/// Matrix product with segmented inner dimension.
pub fn segmented_mm(a: &Array, b: &Array, segments: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::segmented_mm(
        a.as_inner(), b.as_inner(), segments.as_inner(),
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

### 命名说明：`inner` vs `inner_product`

Rust 端不能用 `inner` 作为公开函数名 —— `inner` 是 Rust crate 中常用的内部命名（如 `Array::from_inner` / `as_inner`）。改用 **`inner_product`** 避免歧义。Doc comment 明确说明对应 MLX 的 `inner` op。

### `mlx/src/ops/mod.rs` 改动

```rust
// 当前（P1）：
pub use matmul::matmul;

// 改为：
pub use matmul::{
    addmm, block_masked_mm, gather_mm, inner_product, matmul, outer,
    segmented_mm, tensordot, tensordot_axes,
};
```

由于 `mlx/src/lib.rs` 已 `pub use ops::*`，顶层 re-export 自动通过。

### 安全层设计要点

| 项 | 说明 |
|----|------|
| `Option<&Array>` 用 `map_or(null, |a| a.as_inner() as *const _)` | 5 个函数复用此模式 |
| 单 `unsafe` 块包住 FFI 调用 | block_masked_mm + gather_mm 含裸指针；其他 6 个函数无 unsafe |
| SAFETY 注释 | 仅 unsafe 块需要；纯 array 参数函数不需 |
| `Error::from` + `?` | 所有函数统一 |
| `from_inner` / `as_inner` | 项目惯例 |
| `inner` → `inner_product` | 避开 Rust 内部命名约定 |

## 错误处理

继承 P0–P4 模式：MLX 抛 `runtime_error` → shim 不 catch → cxx `Result<T>` → 安全层 `Error::from`。

不预先做 Rust 端校验。

## 测试策略

集成测试 `mlx/tests/p5_ops_extra.rs`，全部确定性输入。

| 函数 | 测试用例 |
|------|---------|
| `tensordot` | (1) 2D `tensordot(a, b, 1)` 与 `matmul` 一致 |
| `tensordot_axes` | (1) 显式 axes 收缩，shape 与数值正确 |
| `outer` | (1) `outer([N], [M])` shape `[N, M]`，元素 `a[i]*b[j]` |
| `inner_product` | (1) `inner_product([N], [N])` 标量 = 点积 |
| `addmm` | (1) `D = β*C + α*(A@B)` 数值匹配（α=2.0, β=3.0） |
| `block_masked_mm` | (1) shape 正确 + 输出有限（NYI-tolerant 若 Metal 后端 NYI） |
| `gather_mm` | (1) 无 indices 时退化为常规 matmul（NYI-tolerant） |
| `segmented_mm` | (1) shape 正确（NYI-tolerant 若 Metal 后端 NYI） |
| Top-level re-exports | (1) `mlx::ops::tensordot` / `mlx::tensordot` 顶层可达 |

预计 **9 个集成测试**。

## 文件结构

```text
cxx-mlx/
├── mlx-sys/
│   ├── shim/
│   │   ├── include/cxx_mlx_shim/array.h               [改] 追加 P5 段
│   │   └── src/array.cc                               [改] 追加 P5 段 (+ 顶部 include shim_helpers.h)
│   └── src/bridge/array.rs                            [改] 追加 P5 段
└── mlx/
    ├── src/
    │   └── ops/
    │       ├── matmul.rs                              [改] 追加 8 公开函数
    │       └── mod.rs                                 [改] 扩展 pub use 列表
    └── tests/
        └── p5_ops_extra.rs                            [新] 集成测试
```

**没有新建模块**，最小侵入。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| `block_masked_mm` / `segmented_mm` 在 Metal 后端可能 NYI | 测试用 NYI-tolerant 模式（沿用 P3/P4 qqmm/multivariate_normal 模式），仅容忍含 "not yet supported" 的 Err |
| `array.cc` 可能未 include `shim_helpers.h` | Task 1 检查并按需 add include；与现有匿名 lambda 互斥 |
| `inner` 命名冲突 Rust 内部 `as_inner` | 公开 fn 改名为 `inner_product`；doc comment 说明对应 MLX `inner` |
| 8 个函数中无新 cxx idiom | 全部复用 P2/P3/P4 模式，无新桥接难点 |
| `gather_mm` vs P3 `gather_qmm` 命名相近 | doc comment 明确：P5 `gather_mm` 是非量化版本，P3 `gather_qmm` 是量化版本 |

## 与后续阶段关系

- **P5 完成 = ops 补漏闭合**：MLX `ops.h` 公开函数全覆盖（matmul 家族 + 已有所有标量/形状/索引算子）
- **P6（compile）** 紧随：MLX `compile.h` 桥接，cxx 闭包跨 callback 是难点，需要新 idiom 设计（独立 brainstorming）
