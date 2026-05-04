# cxx-mlx P1b2b — Indexing + SDPA 设计文档

**日期**: 2026-05-03
**父设计**: [`2026-05-03-cxx-mlx-design.md`](2026-05-03-cxx-mlx-design.md)(P1b 在 Roadmap 中)
**前置**: P1b2a 已合入 master(commit `4c63794`)
**状态**: 已批准,待实施

## 目标

P1b 的最后一个子阶段。P1b2b 交付:

- **3 个新 Element 类型**:`u16` / `u32` / `u64`(回填 P1b2a 留下的 dtype 缺口;`u32` 是 `argmax` 返回值和 `gather` 索引的必备类型)
- **6 个 indexing op**:`where` / `take` / `take_along_axis` / `slice` / `slice_strided` / `gather`
- **SDPA 集成测试**:完整的 scaled-dot-product-attention(无 fast 优化版),含 causal mask + 数值正确性验证

**验收**:`mlx/tests/p1b2b_sdpa.rs` 中的 SDPA 实现:
- output shape 等于 `[B, H, S, D]`
- 所有元素 finite(无 NaN/Inf)
- causal mask 下未来位置 attention weight ≈ 0
- softmax 行和 = 1
- 与确定性参考输入对比数值正确(容忍 1e-3)

P1b2b 完成后,P1b 全套(P1a + P1b1 + P1b2a + P1b2b)就位,可以开始 P2(`fast` ops + io)。

非目标:
- `slice_update` 系列(写入 slice)—— P3 优化阶段或按需追加
- Boolean fancy indexing(`a[mask]`)—— Python 习惯但 MLX C++ 不直接支持,需要组合 `where` + `take`
- `gather_qmm` / `scatter_*` —— 量化或专用 op,不在 P1b2b 范围

## 关键决策

### A1. Element 扩展:u16 / u32 / u64

P1a 的 `Element` 集合是 10 个类型。P1b2b 加 3 个无符号整数:

| 类型 | Dtype | 用例 |
|---|---|---|
| `u16` | `Dtype::Uint16` | 通用,跟 i16 配对 |
| `u32` | `Dtype::Uint32` | `argmax` 返回值;`take`/`gather` 索引;tokenizer ids |
| `u64` | `Dtype::Uint64` | 大型 index 数组(>2B 元素) |

每个新 dtype 加 3 个 shim function(`array_from_<T>` / `array_item_<T>` / `array_to_vec_<T>`),共 9 个。沿用 P1a 的 `element_impl_simple!` macro 一行展开。

P1a / P1b2a 已经留下的 `argmax` 测试用 `u32` 增强,直接做数值断言。

### A2. `where` API

```rust
pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array>;

impl Array {
    pub fn where_(&self, x: &Array, y: &Array) -> Result<Array>;  // self is condition
}
```

**注意 `where_` 末尾下划线**:`where` 是 Rust 关键字,`r#where` 是常用回避方式但 `where_` 更显眼且与 `Array::where_(...)` 方法形式一致。在 `lib.rs` 文档/README 解释一次。

**Rust 侧主动校验**:连续两次 `broadcast_shape(cond, x)` → 中间 shape;再 `broadcast_shape(中间, y)` → 输出 shape。任一步失败 → `Err(Error::BroadcastMismatch)`。约 15 行 Rust。

`cond` 通常是 bool dtype。Rust 不主动校验 dtype,交给 MLX 决定(MLX 接受任何 numeric 类型,非零视为 true)。

### A3. `take` / `take_along_axis` API

NumPy/PyTorch 风格,直接镜像 MLX:

```rust
/// Take values along `axis` according to a 1-D `indices` array.
/// Output shape: `a.shape` with `axis` dim replaced by `indices.size()`.
pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array>;

/// Take values where `indices` has the same shape as `a` except along `axis`.
/// Output shape: `indices.shape`.  Equivalent to PyTorch's `torch.gather`.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array>;

impl Array {
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array>;
    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array>;
}
```

**indices dtype**:必须是无符号整数类型(`u32` / `u64`)。MLX 自己校验,Rust 不预判。

**索引越界**:行为由 MLX 决定(可能 throw,可能 silent wrap)。Rust 不预判,文档说明"behavior undefined for out-of-range indices, mirrors MLX semantics"。

### A4. `slice` / `slice_strided` API

```rust
/// Slice with strides=1 in every dim. start/stop must each have length == a.ndim().
pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array>;

/// Slice with explicit strides. start/stop/strides must all have length == a.ndim().
pub fn slice_strided(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array>;

impl Array {
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Result<Array>;
    pub fn slice_strided(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array>;
}
```

**Negative indices**:MLX 原生支持(`stop = -3` → 倒数第 3 个),Rust 直接传透过 FFI,不做转换。

**Length mismatch**:Rust 主动校验 `start.len() == stop.len() == a.ndim()`(strided 还要 strides 也等长),不一致 → `Err(Error::ShapeMismatch { expected: vec![ndim, ...], actual: vec![start.len() as i32, ...] })`。约 20 行 Rust。

`slice` 内部直接调 `slice_strided` with `vec![1; ndim]`,共享一个 shim function `array_slice_strided`。`slice` 的轻量级单独 shim 不必要。

### A5. `gather` API

MLX `gather` 的 N-D 多维索引:

```rust
/// N-dimensional gather: pick slices of `a` at the cartesian product of indices
/// along `axes`. `slice_sizes` controls the size of each gathered slice.
///
/// Returns shape `indices_shape ++ slice_sizes` (concatenation).
///
/// See MLX docs for full semantics — this is the most flexible / least
/// intuitive of the indexing ops.
pub fn gather(a: &Array, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array>;

impl Array {
    pub fn gather(&self, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array>;
}
```

`&[&Array]` 跨桥沿用 P1b2a `concatenate`/`stack` 的 raw pointer slice 模式(`&[*const MlxArray]`)。`axes` 和 `slice_sizes` 走普通 `&[i32]`。

`gather` 是 P1b2b 最复杂的 op。设计上为高级用户,文档简短示例 + 链 MLX docs。

### A6. SDPA 集成测试范围(C 完整版)

`mlx/tests/p1b2b_sdpa.rs` 实现 SDPA 算法 + 4 个测试:

```rust
fn scaled_dot_product_attention(
    q: &Array,         // [B, H, S, D]
    k: &Array,         // [B, H, S, D]
    v: &Array,         // [B, H, S, D]
    mask: Option<&Array>,  // [S, S] additive (-inf in masked positions, 0 else)
    scale: f32,
) -> Result<Array> {
    // scores = Q @ K.transpose(-1, -2) * scale
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = q.matmul(&kt)?;
    let scaled = (&scores * scale)?;

    // Apply mask if provided (additive: -inf where masked, 0 else, broadcasts on B/H)
    let masked = match mask {
        Some(m) => (&scaled + m)?,
        None => scaled,
    };

    // Softmax along last axis
    let m = ops::max(&masked, -1, true)?;
    let shifted = (&masked - &m)?;
    let e = shifted.exp()?;
    let s = ops::sum(&e, -1, true)?;
    let weights = (&e / &s)?;

    // out = weights @ V
    weights.matmul(v)
}
```

测试:
1. `sdpa_no_mask_shape_finite` — `[B=1, H=2, S=4, D=8]` 全随机化输入,output shape `[1, 2, 4, 8]`,所有元素 finite
2. `sdpa_causal_mask_zeros_future` — 加 causal mask(下三角 0,上三角 -inf),验证 attention weights 上三角(未来位置)< 1e-6
3. `sdpa_softmax_rows_sum_to_one` — 任何输入下,softmax 输出每行和 ≈ 1.0(容忍 1e-5)
4. `sdpa_numerical_match_reference` — 给固定的 deterministic Q/K/V(简单整数从 from_slice 构造),与手算/参考值对比 output(容忍 1e-3)

SDPA 算法本身放在测试文件内的 `fn`(不进 lib API,P2 fast 会有优化版本)。

### A7. 文件组织

新增:

- `mlx/src/ops/indexing.rs`(6 ops:`where_` / `take` / `take_along_axis` / `slice` / `slice_strided` / `gather`,加助手函数)
- `mlx/tests/p1b2b_indexing.rs`(每个 op 的核心测试)
- `mlx/tests/p1b2b_sdpa.rs`(4 个集成测试)
- `mlx/tests/p1b2b_dtype_extension.rs`(u16/u32/u64 round-trip 测试 + argmax 数值检查)

修改:

- `mlx/src/element.rs` — 加 3 个 `element_impl_simple!` 调用
- `mlx-sys/src/bridge/array.rs` / `shim/include/...array.h` / `shim/src/array.cc` — 加 9 个新 dtype shim + 6 个 indexing shim(共 ~15 个新 shim function)
- `mlx/src/ops/mod.rs` — re-export 新模块
- `mlx/src/array.rs` — 加 6 个新 method(`where_` / `take` / `take_along_axis` / `slice` / `slice_strided` / `gather`)
- `mlx/src/dtype.rs` — 无变化(P1a 已支持 Uint16/Uint32/Uint64)
- `README.md` — 更新 status 行 + 加"Indexing & SDPA"小节

### A8. shim Result-wrapping 一致性

约 15 个新 shim function,**全部 `Result<UniquePtr<MlxArray>>`** 遵守 P1a 硬规则(MLX 在 dtype 不匹配 / shape 不兼容 / 索引越界等情况会 throw)。

`gather` 的 `&[*const MlxArray]` 输入在 bridge 用 `unsafe fn`,与 P1b2a `concatenate`/`stack` 一致;Rust safe 层在 `gather` 内组装 raw pointer vector,unsafe block 仅限于 FFI 调用本身。

### A9. 命名 / re-export

- `where_` 不能叫 `where`(Rust 关键字)。`mlx::ops::where_` 和 `mlx::Array::where_` 都尾下划线。在 `lib.rs` 顶层 doc 解释一次
- `take` 自由函数和 method 同名,无冲突
- `slice` 自由函数和 `Array::slice` method 同名,无冲突(method receiver 是 `&self`)
- 不向 `mlx` 顶层 re-export `where_`(避免误导用户以为它是 Rust 的 `where` 关键字)。`use mlx::ops::where_` 显式调用

## 文件改动清单

### 新增

- `mlx/src/ops/indexing.rs` — 6 个 indexing 自由函数 + Rust 侧校验
- `mlx/tests/p1b2b_indexing.rs` — 每 op 的测试(取一个轴、广播、错误路径)
- `mlx/tests/p1b2b_sdpa.rs` — 4 个 SDPA 集成测试
- `mlx/tests/p1b2b_dtype_extension.rs` — u16/u32/u64 round-trip + argmax 数值

### 修改

- `mlx/src/element.rs` — 加 `u16`/`u32`/`u64` 的 `element_impl_simple!` 调用
- `mlx-sys/src/bridge/array.rs` — 加 ~15 个新 shim 声明
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — 加新声明
- `mlx-sys/shim/src/array.cc` — 加新实现
- `mlx-sys/tests/sys_smoke.rs` — 加几个 link smoke
- `mlx/src/ops/mod.rs` — 加 `pub mod indexing;` + re-export
- `mlx/src/array.rs` — 加 6 个 indexing methods
- `mlx/tests/p1b2a_reduction.rs` — `argmax_basic`/`argmax_all_returns_flat_index` 加 `to_vec::<u32>()` 数值断言(P1b2a 反向修复)
- `README.md` — 更新 status 行 + 加 Indexing & SDPA 示例

### 已存在,无变化

- `mlx-sys/build.rs` / `bridge/mod.rs` / `bridge/transforms.rs`
- `mlx/src/{dtype,error,broadcast,ops/binary,ops/unary,ops/shape,ops/reduction,ops/matmul,ops_impl}.rs`
- 所有 P0/P1a/P1b1/P1b2a 测试(除上面提到的 argmax 增强)

## 实施分期(P1b2b 内)

约 9 个 TDD 任务:

1. shim:9 个新 dtype shim 函数(u16/u32/u64 × from/item/to_vec)+ 15 个 indexing shim 函数(where/take/take_along_axis/slice_strided/gather + 配套)= ~24 个新 shim,批量提交
2. Element u16/u32/u64 + dtype round-trip 测试
3. P1b2a 反向修复:argmax 测试加 u32 数值断言
4. `ops::where_` + Rust 侧广播校验 + 测试
5. `ops::take` + `ops::take_along_axis` + 测试
6. `ops::slice` + `ops::slice_strided` + Rust 侧 length 校验 + 测试
7. `ops::gather`(raw pointer slice 复用 P1b2a 模式)+ 测试
8. SDPA 集成测试(完整 C 范围:shape + finite + causal mask + softmax + 数值)
9. README + final verify + clippy + doc

## 决策记录

- **B1**:Element 扩展 `u16` / `u32` / `u64`,9 个新 shim function 沿用 `element_impl_simple!` macro
- **B2**:`where_`(尾下划线避 Rust 关键字),Rust 侧主动 broadcast 校验
- **B3**:`take` / `take_along_axis` 镜像 NumPy/PyTorch 命名,indices dtype/越界由 MLX 决定
- **B4**:`slice`(strides=1) + `slice_strided`(显式)双形态,共享一个 shim;Rust 主动 length 校验
- **B5**:`gather` 用 P1b2a 的 raw pointer slice 模式跨桥
- **B6**:SDPA 测试范围 C(shape + finite + causal mask + softmax 行和 + 数值)
- **B7**:9 task 分期,所有新 shim 遵守 P1a Result-wrapping 硬规则
- **B8**:`where_` 不向 `mlx` 顶层 re-export,显式 `use mlx::ops::where_`

## 后续 P1c / P2 接口约束

- u16/u32/u64 Element 扩展是稳定 ABI,P2 io 加载 safetensors 时索引/scale 类型可直接用
- `take` / `take_along_axis` / `slice` 在 P2 实战(KV cache 切片、token 选择)直接复用
- `gather` 在 P3 优化(advanced indexing for quantization)留好接口
- SDPA 集成测试是 P2 `fast::scaled_dot_product_attention` 的 baseline 对照(P2 fast 实现要与 P1b2b naive 实现数值一致)
