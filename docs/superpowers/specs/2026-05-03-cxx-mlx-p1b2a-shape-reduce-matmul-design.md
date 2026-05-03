# cxx-mlx P1b2a — Shape Ops + Reduction + Matmul 设计文档

**日期**: 2026-05-03
**父设计**: [`2026-05-03-cxx-mlx-design.md`](2026-05-03-cxx-mlx-design.md)(P1b 在 Roadmap 中)
**前置**: P1b1 已合入 master(commit `a6b61b8`)
**状态**: 已批准,待实施

## 目标

P1b 拆为 P1b2a + P1b2b。**P1b2a** 交付:

- 6 个 shape ops:`reshape`(支持 `-1` 占位推断)、`transpose`(反转全维)、`transpose_axes`(NumPy permute)、`broadcast_to`、`concatenate`、`stack`、`split_n`、`split_at`
- 5 个 reduction ops:`sum` / `mean` / `max` / `min` / `argmax`,统一通过 `IntoAxes` trait 接收"全轴 / 单轴 / 多轴"三种 axes 输入
- `matmul`(覆盖 batched + broadcast 语义,LLM attention 直接可用)
- `Array::t()` 快捷方法(等价 `transpose`,矩阵语境标准简写)
- `ops.rs` 拆分为 `ops/` 子目录(P1b1 final review 提示)

**验收**:能用 P0 + P1a + P1b1 + P1b2a 写出 `softmax`、`gelu`、`silu` 三个组合算子,数值正确性测试通过。

非目标(留给 P1b2b):

- indexing(`take` / `gather` / `where` / `slice`)
- `scaled_dot_product_attention`(无 fast 优化版)集成测试 —— 需要 `where` 写 mask
- `addmm`(`c + a @ b` 融合)、`einsum` —— 留 P3 优化阶段

## 关键决策

### A1. `IntoAxes` trait + `All` unit struct

5 个 reduction 各 1 个 free fn,axes 参数通过 `IntoAxes` sealed trait 接收:

```rust
// in mlx/src/ops/reduction.rs (or shared axes module)

/// Marker for "all axes" reduction. Use as `sum(&a, All, false)`.
#[derive(Debug, Clone, Copy)]
pub struct All;

mod sealed {
    pub trait Sealed {}
}

pub trait IntoAxes: sealed::Sealed {
    /// Internal: convert to a `&[i32]` (or empty for All) for shim dispatch.
    /// `All` returns `None`; specific axes return `Some(slice)`.
    #[doc(hidden)]
    fn as_axes(&self) -> Option<&[i32]>;
}

impl sealed::Sealed for All {}
impl IntoAxes for All {
    fn as_axes(&self) -> Option<&[i32]> { None }
}

impl sealed::Sealed for i32 {}
impl IntoAxes for i32 {
    fn as_axes(&self) -> Option<&[i32]> { Some(std::slice::from_ref(self)) }
}

impl sealed::Sealed for &[i32] {}
impl IntoAxes for &[i32] {
    fn as_axes(&self) -> Option<&[i32]> { Some(*self) }
}

impl sealed::Sealed for Vec<i32> {}
impl IntoAxes for Vec<i32> {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
}

impl<const N: usize> sealed::Sealed for [i32; N] {}
impl<const N: usize> IntoAxes for [i32; N] {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
}
```

5 reduction × 1 fn = 5 free functions + 5 method wrappers. Calls:

```rust
ops::sum(&a, All, false)?              // all axes, no keepdim
ops::sum(&a, -1, true)?                // last axis, keepdim
ops::sum(&a, &[0, 2], false)?          // axes 0 and 2
ops::sum(&a, vec![0, 2], false)?       // owned form
ops::sum(&a, [0, 2], false)?           // array literal (const generic impl)
```

**Sealed pattern** prevents downstream crates from impl-ing IntoAxes and breaking dispatch.

### A2. `keepdim: bool` 位置参数

5 reduction 都接收 `keepdim: bool` 作为最后位置参数。`sum(&a, axes, keepdim)`。Rust 生态常规做法,与 PyTorch / NumPy 的 `keepdim` 语义一致。

### A3. shim 端 reduction 派发

每个 reduction 生成 3 个 shim(MLX C++ 重载映射):

```cpp
// in mlx-sys/shim/src/array.cc
std::unique_ptr<MlxArray> array_sum_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);
```

5 reduction × 3 = 15 个 shim functions。Rust 端 `ops::sum<A: IntoAxes>` 根据 `axes.as_axes()` 的返回(`None` / `Some(&[a])`(len=1) / `Some(&[a..])`(len>=2))分派到对应 shim。

```rust
pub fn sum<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_sum_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_sum_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => mlx_sys::array::ffi::array_sum_axes(a.as_inner(), axes, keepdim),
    }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

`unary_op!` macro 模式扩展为 `reduction_op!(sum, array_sum_all, array_sum_axis, array_sum_axes)` 一行展开。

### A4. `reshape` 的 `-1` 占位推断

MLX C++ `reshape(array, Shape)` 不支持 `-1`。Rust safe 层在 dispatch 之前推断:

```rust
pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    let total: usize = a.size();
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    let resolved: SmallVec<[i32; 8]> = match neg_count {
        0 => shape.iter().copied().collect(),
        1 => {
            let known: usize = shape.iter().filter(|&&d| d != -1).map(|&d| d as usize).product();
            if known == 0 || total % known != 0 {
                return Err(Error::Mlx(format!(
                    "reshape: cannot infer -1 dim — total {total} not divisible by product {known}"
                )));
            }
            let inferred = (total / known) as i32;
            shape.iter().map(|&d| if d == -1 { inferred } else { d }).collect()
        }
        _ => return Err(Error::Mlx(format!(
            "reshape: at most one -1 placeholder allowed, got {neg_count} in {shape:?}"
        ))),
    };
    let inner = mlx_sys::array::ffi::array_reshape(a.as_inner(), &resolved).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

约 25 行,3 个测试覆盖(单 -1、零 -1、多 -1)。LLM 推理常用 `reshape(&[B, S, -1])` 展平最后两维,无此功能调用方需手算。

### A5. `transpose` API:`transpose` + `transpose_axes` + `Array::t()`

```rust
// 反转所有维度(等价 NumPy `arr.T`)
pub fn transpose(a: &Array) -> Result<Array>;

// 显式 permutation(等价 NumPy `np.transpose(arr, axes)`)
pub fn transpose_axes(a: &Array, axes: &[i32]) -> Result<Array>;

// Array method
impl Array {
    pub fn t(&self) -> Result<Array>;        // 调 ops::transpose
    pub fn transpose(&self) -> Result<Array>; // 同上,长名版
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array>;
}
```

`Array::t()` 是矩阵语境标准简写(`Q @ K.t()` 在 attention 里高频)。`swapaxes` 不暴露(用 `transpose_axes` 手算可达,YAGNI)。

### A6. `broadcast_to` / `concatenate` / `stack` / `split`

```rust
// Broadcast a to target shape (validates compatibility via broadcast_shape).
pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array>;

// Concatenate along axis. arrays must all have same shape except along axis.
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array>;

// Stack along NEW axis (all arrays must have identical shape; result has +1 dim).
pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array>;

// Split into N equal pieces along axis.
pub fn split_n(a: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>>;

// Split at given indices along axis.
pub fn split_at(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>>;
```

`split_n` / `split_at` 返回 `Result<Vec<Array>>`(MLX C++ 返回 `std::vector<array>`)。shim 端用 P1a 已建立的 `eval_many` 套路:不直接桥 vector,而是返回数量 + 提供索引访问。

**实际 shim 模式**(避免 cxx 不直接支持 `Vec<UniquePtr<MlxArray>>`):

```cpp
// shim/src/array.cc
std::unique_ptr<std::vector<MlxArray>> array_split_n(
    const MlxArray& a, int32_t num_splits, int32_t axis);

size_t split_result_len(const std::vector<MlxArray>& v);
std::unique_ptr<MlxArray> split_result_at(const std::vector<MlxArray>& v, size_t i);
```

```rust
// bridge — opaque holder for split result vector
unsafe extern "C++" {
    type MlxArrayVec;
    fn array_split_n(a: &MlxArray, n: i32, axis: i32) -> Result<UniquePtr<MlxArrayVec>>;
    fn split_result_len(v: &MlxArrayVec) -> usize;
    fn split_result_at(v: &MlxArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;
}
```

Rust safe 层 unpack 到 `Vec<Array>`。`split_at` 同模式(共享 `MlxArrayVec` opaque 类型 + `split_result_*` 访问器)。

注:`stack` / `concatenate` 输入 `&[&Array]` 也需要桥接技巧 —— cxx 不接受 `&[&MlxArray]`。Rust 侧提取 raw pointers 传 `&[*const MlxArray]`,shim 重组 `std::vector<MlxArray>`(每个用 copy ctor,refcount 共享)。

### A7. `matmul` API

```rust
pub fn matmul(a: &Array, b: &Array) -> Result<Array>;

impl Array {
    pub fn matmul(&self, rhs: &Array) -> Result<Array>;
}
```

MLX `matmul` 自动处理 batched + broadcast(高维 batch dim 走 NumPy 广播)。Rust 端单 fn 一并覆盖。

LLM attention 用例:`Q.matmul(&k.t())?` 其中 Q=[B, H, S, D]、K=[B, H, S, D] → K.t()=[B, H, D, S] → matmul=[B, H, S, S]。

不实现 `addmm`(P3 优化)、`einsum`(P3 + 复杂)、`tensordot`(YAGNI)。

### A8. shim 端 Result-wrapping 一致

P1b2a 加约 20 个新 shim 函数(15 reduction + 5 split/concat/stack 系列;reshape/transpose/broadcast_to/matmul 各 1)。**全部 `Result<UniquePtr<MlxArray>>` 或 `Result<UniquePtr<MlxArrayVec>>`**(MLX 在 axis 越界、shape 不兼容、矩阵尺寸不匹配等情况会 throw)。`MlxArrayVec` 跨 bridge 类型沿用 P1a 的 `type X = crate::bridge::array::ffi::X;` 模式。

### A9. 文件组织:`ops.rs` 拆为 `ops/` 子目录

P1b1 final review 推荐:`ops.rs` 已 73 行,P1b2a 加 12 个 op 会膨胀到 200+ 行。本期顺手拆分:

```text
mlx/src/ops/
├── mod.rs           (re-export 全部 free fns)
├── binary.rs        (从 ops.rs 搬:add/subtract/multiply/divide/negative)
├── unary.rs         (从 ops.rs 搬:exp/log/sqrt/tanh/sigmoid/square/rsqrt/erf/reciprocal)
├── shape.rs         (新:reshape/transpose/transpose_axes/broadcast_to/concatenate/stack/split_n/split_at)
├── reduction.rs     (新:sum/mean/max/min/argmax + IntoAxes trait + All)
└── matmul.rs        (新:matmul)
```

`ops_impl.rs` 不动(运算符 trait 集中)。`array.rs` 加新 method 包装(每个 1 行调 `crate::ops::*`)。

### A10. softmax/gelu/silu 验收测试

新增 `mlx/tests/p1b2a_compose.rs`,实现并测试三个组合算子(不放进 lib API,仅作集成测试):

```rust
fn softmax(x: &Array, axis: i32) -> Result<Array> {
    let m = ops::max(x, axis, true)?;
    let e = (x - &m)?.exp()?;
    let s = ops::sum(&e, axis, true)?;
    &e / &s
}

fn gelu(x: &Array) -> Result<Array> {
    let sqrt_2 = Array::from_slice(&[2.0_f32.sqrt()], &[])?;
    let half = (x * 0.5_f32)?;
    let inner = (x / &sqrt_2)?.erf()?;
    let one_plus = (&inner + 1.0_f32)?;
    &half * &one_plus
}

fn silu(x: &Array) -> Result<Array> {
    let s = x.sigmoid()?;
    x * &s
}
```

测试断言:
- `softmax([1.0, 2.0, 3.0], -1)` 之和 ≈ 1.0,各项 > 0
- `gelu(0.0) ≈ 0`、`gelu(1.0) ≈ 0.8413`(标准 GELU 表)
- `silu(0.0) ≈ 0`、`silu(1.0) ≈ 0.7311`(= 1 * sigmoid(1))

## 文件改动清单

### 新增

- `mlx/src/ops/` 子目录(6 个文件:`mod.rs`、`binary.rs`、`unary.rs`、`shape.rs`、`reduction.rs`、`matmul.rs`)
- `mlx/src/axes.rs` 或在 `ops/reduction.rs` 内:`All` struct + `IntoAxes` sealed trait + 5 个 impl
- `mlx/tests/p1b2a_shape.rs` — reshape/transpose/broadcast_to/concatenate/stack/split 单元测试
- `mlx/tests/p1b2a_reduction.rs` — 5 reduction × 3 axes 形态 + keepdim 测试
- `mlx/tests/p1b2a_matmul.rs` — 2D 基础 + batched + broadcast 测试
- `mlx/tests/p1b2a_compose.rs` — softmax / gelu / silu 数值正确性

### 修改

- `mlx/src/lib.rs` — `mod ops;` 改为指向新目录,re-export `All`
- `mlx/src/array.rs` — 加新 methods(`reshape`、`transpose`、`t`、`broadcast_to`、`matmul`、5 个 reduction 方法 用 `IntoAxes` 泛型)
- `mlx/src/ops.rs` — 删除(内容拆分到 `ops/` 子目录)
- `mlx/src/ops_impl.rs` — `use crate::ops` 改为 `use crate::ops::binary::*` 等(或保持 `crate::ops::*` 通过 mod.rs re-export)
- `mlx-sys/src/bridge/array.rs` — 加 ~20 个新 FFI 函数 + `MlxArrayVec` opaque type
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — 加新声明
- `mlx-sys/shim/src/array.cc` — 加新实现
- `README.md` — 更新 Status 行 + 加"Reductions / Shape / Matmul"小节短示例

### 已存在,无变化

- `mlx-sys/build.rs`、`bridge/mod.rs`、`bridge/transforms.rs`
- `mlx/src/{dtype,element,error,broadcast}.rs`
- 所有 P0/P1a/P1b1 测试

## 测试策略

### 集成测试(`mlx/tests/p1b2a_*.rs`)

**shape (`p1b2a_shape.rs`)**:
- `reshape(&[2,3,4], &[6, 4])` 数值保留;`reshape(..., &[-1, 4])` 推断为 6;`reshape(&[2,3,4], &[2, -1, -1])` → `Err`
- `transpose([2,3]) → [3,2]`,数值正确(行列对调)
- `transpose_axes([2,3,4], &[2,0,1]) → [4,2,3]`
- `broadcast_to([3], &[2,3]) → [2,3]`,值复制
- `concatenate([&[2,3], &[2,3]], 0) → [4,3]` 和 `axis=1 → [2,6]`
- `stack([&[2,3], &[2,3]], 0) → [2,2,3]`
- `split_n([6,4], 3, 0)` → 3 个 `[2,4]`;`split_at([6,4], &[2, 4], 0)` → 3 个 `[2,4]`/`[2,4]`/`[2,4]`

**reduction (`p1b2a_reduction.rs`)**:
- 5 ops × 全轴(`All`)/ 单轴(`-1`)/ 多轴(`&[0, 1]`、`vec![0, 1]`、`[0, 1]`)= 共 25 个测试矩阵的代表性子集
- keepdim 切换验证:`sum([2,3,4], -1, true) → [2,3,1]`,`keepdim=false → [2,3]`
- `argmax([2,3,4], -1, false) → [2,3]` 类型为 Int32(MLX 自动)

**matmul (`p1b2a_matmul.rs`)**:
- 2D × 2D:`[2,3] @ [3,4] = [2,4]` 数值校验
- 3D batched:`[B,S,D] @ [B,D,M] = [B,S,M]`
- 4D attention:`[B,H,S,D] @ [B,H,D,S] = [B,H,S,S]`
- shape 不匹配 → `Err(Error::Mlx)`

**compose (`p1b2a_compose.rs`)**:
- softmax、gelu、silu 数值正确性(见 A10)

### 单元测试

- `IntoAxes` impl 各类型 → `as_axes()` 返回正确(`All → None`,`-1 → Some(&[-1])`,等)
- reshape `-1` 推断逻辑(零/单/多 -1)

### 回归测试

- P0 + P1a + P1b1 既有测试不变
- `ops::*` 重构后所有测试位置不变(re-export 通过 `ops/mod.rs`)

## 实施分期(P1b2a 内)

约 12 个 TDD 任务:

1. `ops.rs` → `ops/` 子目录拆分(纯重构,P1b1 测试不变)
2. `IntoAxes` trait + `All` struct + 5 impls + 单元测试
3. shim:5 reduction × 3 形态 = 15 个新函数(批量,模式同 P1b1 Task 1)
4. `ops::sum` 用 IntoAxes dispatch + Array::sum method + 测试覆盖三种 axes 形态
5. 其余 4 reduction(`mean/max/min/argmax`),复用 sum 模式
6. shim:`reshape`(单 fn,Rust 侧 -1 推断)+ 测试
7. shim:`transpose` + `transpose_axes` + `broadcast_to`(3 fn)+ 测试 + Array::t() 方法
8. shim:`concatenate` + `stack`(2 fn,处理 `&[*const MlxArray]` 桥接)+ 测试
9. shim:`split_n` + `split_at`(2 fn + `MlxArrayVec` opaque + 访问器)+ 测试
10. shim:`matmul`(单 fn)+ 测试 + Array::matmul 方法
11. compose 集成测试(softmax / gelu / silu)
12. README + 全量 verify + clippy + doc

每步 TDD red→green→commit。

## 决策记录

- **B1**:reduction axis 用 `IntoAxes` sealed trait + `All` unit struct,统一 free fn 名(`sum/mean/max/min/argmax`)
- **B2**:keepdim 是 `bool` 位置参数,与 NumPy/PyTorch 一致
- **B3**:reshape Rust 侧支持 `-1` 占位推断(单个,多个 → Err)
- **B4**:transpose 暴露 3 个名(`transpose` 反转 / `transpose_axes` permute / `Array::t()` 矩阵简写)
- **B5**:matmul 单 fn 覆盖 2D/batched/broadcast(MLX 已统一)
- **B6**:`Vec<Array>` 返回的 split 用 `MlxArrayVec` opaque 跨桥,不直接桥 `Vec<UniquePtr<MlxArray>>`
- **B7**:`&[&Array]` varargs 通过 `&[*const MlxArray]` raw pointer 桥转,shim 重组 vector
- **B8**:`ops.rs` 拆为 `ops/` 子目录(`mod.rs`、`binary.rs`、`unary.rs`、`shape.rs`、`reduction.rs`、`matmul.rs`)
- **B9**:验收靠 softmax / gelu / silu 三个组合算子集成测试,SDPA 留 P1b2b(需要 indexing)
- **B10**:所有 ~20 个新 shim 函数遵守 P1a Result-wrapping 硬规则

## 后续 P1b2b / P1c / P2 接口约束

- `IntoAxes` trait 在 P1b2b/P2 reduction 类操作(reduce_sum 等 fast op)直接复用
- `MlxArrayVec` opaque 类型在 P1b2b indexing(可能返回多 array)和 P2 transforms 复用
- `ops/` 子目录结构稳定,P1b2b 加 `ops/indexing.rs`,P2 加 `ops/fast.rs`、`ops/io.rs`
- `Array::matmul` 签名稳定,P3 优化 `addmm` / `einsum` 是新增不是修改
