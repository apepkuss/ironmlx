# cxx-mlx P1b1 — Operators + Element-wise Unary 设计文档

**日期**: 2026-05-03
**父设计**: [`2026-05-03-cxx-mlx-design.md`](2026-05-03-cxx-mlx-design.md)(P1b 在 Roadmap 中)
**前置**: P1a 已合入 master(commit `b22c94a`)
**状态**: 已批准,待实施

## 目标

P1 的"算子主力"阶段 (P1b) 拆为两个子阶段;**P1b1** 交付:

- 5 个二元运算符(`Add` / `Sub` / `Mul` / `Div` / `Neg`)over `Array`,Rust-style 全 4 种引用组合 + 标量 RHS
- 9 个元素 unary op:`exp` / `log` / `sqrt` / `tanh` / `sigmoid` / `square` / `rsqrt` / `erf` / `reciprocal`(自由函数 + Array 方法双形态)
- NumPy-style 广播校验在 Rust 侧前置(`Error::BroadcastMismatch`)
- 可复用的 `broadcast_shape` 推导工具(P1b2 reduction / `broadcast_to` op 复用)

**验收**:能用 P1a + P1b1 的 API 写出可读的 `softmax` 占位实现(因为 reduction 在 P1b2,可以先用 1-element scalar 占位 max/sum,验证语法可行)。

非目标(留给 P1b2):
- shape 操作(reshape/transpose/broadcast_to/concatenate/split/stack)
- reduction(sum/mean/max/min/argmax)
- indexing(take/gather/where/slice)
- matmul

## 关键决策

### A1. 运算符重载:全 4 种引用 + 标量 RHS,无标量 LHS

4 个二元 op(`Add`/`Sub`/`Mul`/`Div`)+ 1 个 unary op(`Neg`)各自 impl 全部 Rust 数值类型常见的引用组合:

```rust
// 4 ref combos for Array op Array
impl Add<Array> for Array { ... }
impl Add<&Array> for Array { ... }
impl Add<Array> for &Array { ... }
impl Add<&Array> for &Array { ... }   // 主路径

// Array op scalar (T: Element)
impl<T: Element> Add<T> for Array { ... }
impl<T: Element> Add<T> for &Array { ... }   // 主路径
```

通过 `forward_ref_binop!` 宏一行展开,实际代码量约 50 行(5 ops × 1 macro 调用)。

**`type Output = Result<Array>`**:运算符返回 `Result<Array>` 而非 `Array`,因为广播校验和 MLX op 都可能失败。用户代码因此长这样:

```rust
let y = (&a + &b)?.matmul(&w)?;       // each op uses ?
let z = (&x * 2.0_f32)?.exp()?;        // scalar mul, then unary
```

不可避免的 ergonomic 代价。比"运算符 panic on mismatch"安全;比"运算符返回 Array 但下次 op 才 throw"信号更早。

**标量 LHS(`1.0 - &a`)不支持**:Rust orphan rule 阻止 `impl Sub<&Array> for f32`(`f32` 不是本地类型)。README 给等价替代:`(-&a)? + 1.0`、`Array::full(...) - &a`(P1b2 加 `full`)、`Array::from_slice(&[1.0], &[])? - a`(P1a 已有)。

### A2. 标量 RHS:全 10 个 Element 类型可用

`impl<T: Element> Add<T> for &Array` 接受所有 10 个 P1a Element 类型(`bool`/`u8`/`i8`/`i16`/`i32`/`i64`/`half::f16`/`half::bf16`/`f32`/`f64`)。Macro 一行带 10 个,代码量与限制 6 个差不多,完整性消除"为什么 i16 不行"的疑问。

dtype 不匹配时(如 `&a + 1.0_f32` where `a: Array` of `Int32`):**不在 Rust 层做提升,完全交给 MLX**。MLX 自带类型提升表;若 MLX throw,cxx 转 `Error::Mlx(String)`。

### A3. Broadcasting:Rust 侧前置校验

二元 op 在 dispatch FFI 之前,Rust 侧调用 `broadcast::broadcast_shape(lhs_shape, rhs_shape) -> Result<SmallVec<[i32; 8]>>`。规则(NumPy):

1. 尾部对齐(右对齐)
2. 缺失维度视为 1
3. 每对维度 `(a, b)` 必须满足 `a == b` ∨ `a == 1` ∨ `b == 1`,结果维度 = `max(a, b)`
4. 任何不满足 → `Err(Error::BroadcastMismatch { lhs: lhs.to_vec(), rhs: rhs.to_vec() })`

**为什么前置校验**:

- `Error::BroadcastMismatch` 变体在 P1a 已加,需要 P1b1 主动产生
- 推理代码里 shape 不匹配是常见编程错误(模型权重 shape 算错、batch dim 弄丢),清晰错误信息直接关系调试体验
- MLX 会 throw 但错误信息是英文字符串,无 Rust 结构化 `lhs` / `rhs` 字段
- 推导出的 `broadcast_shape` 在 P1b2 reduction(`sum_axes` 计算 keepdim shape)和 `broadcast_to` op 直接复用

实现 ~30 行 Rust + 6-8 个测试用例。

### A4. 标量 RHS 实现:Rust 侧构造 1-element Array

不引入 50 个 `array_add_scalar_<T>` shim 函数。Rust 侧统一:

```rust
impl<T: Element> Add<T> for &Array {
    type Output = Result<Array>;
    fn add(self, rhs: T) -> Self::Output {
        let scalar = Array::from_slice(&[rhs], &[])?;  // 1-element scalar Array
        ops::add(self, &scalar)
    }
}
```

代价:每次标量混合多 1 个小 Array 分配 + eval graph 多 1 个 broadcast 节点。MLX 内部对常量 Array 有优化,实际开销可忽略(尤其在推理 hot loop 里 op 的真实成本是 GPU kernel)。

**收益**:shim 只加 5 个二元函数(`array_add` / `array_subtract` / `array_multiply` / `array_divide` / `array_negative`),不是 55 个。

### A5. Unary ops 集合(9 个)+ 双调用风格

| op | C++ 名 | Rust 自由函数 | Array 方法 |
| --- | --- | --- | --- |
| exp | `mlx::core::exp` | `ops::exp(&a)` | `a.exp()` |
| log | `mlx::core::log` | `ops::log(&a)` | `a.log()` |
| sqrt | `mlx::core::sqrt` | `ops::sqrt(&a)` | `a.sqrt()` |
| tanh | `mlx::core::tanh` | `ops::tanh(&a)` | `a.tanh()` |
| sigmoid | `mlx::core::sigmoid` | `ops::sigmoid(&a)` | `a.sigmoid()` |
| square | `mlx::core::square` | `ops::square(&a)` | `a.square()` |
| rsqrt | `mlx::core::rsqrt` | `ops::rsqrt(&a)` | `a.rsqrt()` |
| erf | `mlx::core::erf` | `ops::erf(&a)` | `a.erf()` |
| reciprocal | `mlx::core::reciprocal` | `ops::reciprocal(&a)` | `a.reciprocal()` |

每个 op 1 个 shim + 1 个自由函数 + 1 个 method(method 是 1 行包装)。全部 `Result<Array>` 因为 dtype 不支持(如 `sqrt(int_array)`)MLX 会 throw。

**为什么这 9 个**:

- 5 个 spec 必须(exp/log/sqrt/tanh/sigmoid)
- `square`、`rsqrt`(`1/sqrt(x)`)在 attention scaling 里直接用
- `erf` 是 GELU 精确实现的核心(`gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`)
- `reciprocal` 是常见的 element-wise op,实现成本几乎零

**为什么不在这次加 abs/neg/sin/cos/floor/ceil/sign**:`-&a` 已经通过 `Neg` 提供;其他在 LLM 推理里罕见,YAGNI 原则推到 P2/P3 按需追加。

**Neg 行为**:`impl Neg for &Array`(及 4 个 ref 组合)调 `mlx::core::negative`。在 unsigned/bool 上 MLX 会 throw(没有负数概念),走 `Result` 路径自然得到 `Err(Error::Mlx(...))`,Rust 侧不主动拒绝。

### A6. 文件组织

P1a 留下 `mlx/src/array.rs`(~150 行)和 `element.rs`(~140 行)。P1b1 加约 350-450 行新代码,分布在新建文件中。结构:

```text
mlx/src/
├── lib.rs               (导出 Array / Dtype / Element / Error / Result + 新加 ops 模块)
├── array.rs             (~150 → ~180 行;加 Array 上 9 个 unary 方法,各 1 行)
├── broadcast.rs         (新建,~80 行;广播 shape 推导 + 单元测试)
├── ops.rs               (新建,~150 行;5 个 binary 自由函数 + 9 个 unary 自由函数)
├── ops_impl.rs          (新建,~120 行;4 个二元运算符 trait impl + Neg + forward_ref_binop! 宏 + 标量 dispatch)
├── dtype.rs             (无变化)
├── element.rs           (无变化)
└── error.rs             (无变化)
```

**职责切分**:`ops.rs` 是 source of truth(自由函数);`ops_impl.rs` 把运算符 trait 桥到 `ops::*`;`array.rs` method 也桥到 `ops::*`。三者共享同一份"实际逻辑"。

### A7. shim 端

`mlx-sys/shim/include/cxx_mlx_shim/array.h` + `mlx-sys/shim/src/array.cc` 加 14 个新函数(5 binary + 9 unary):

```cpp
// binary
std::unique_ptr<MlxArray> array_add(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_subtract(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_multiply(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_divide(const MlxArray& a, const MlxArray& b);

// unary
std::unique_ptr<MlxArray> array_negative(const MlxArray& a);
std::unique_ptr<MlxArray> array_exp(const MlxArray& a);
// ... 8 more
```

bridge `mlx-sys/src/bridge/array.rs` 14 行,**全部 `Result<UniquePtr<MlxArray>>`** 遵守 P1a 硬规则(MLX op 在 dtype 不支持等情况下 throw)。

shim 实现各 1 行,直接 `return std::make_unique<MlxArray>(mlx::core::xxx(a, b))`。无新模板助手。

### A8. 测试策略

**单元测试**(`mlx/src/broadcast.rs` 内 `mod tests`):

- `broadcast_shape([2,3], [2,3]) == [2,3]`
- `broadcast_shape([2,3], [3]) == [2,3]`(missing dim → 1)
- `broadcast_shape([2,1,4], [3,4]) == [2,3,4]`
- `broadcast_shape([], [2,3]) == [2,3]`(scalar)
- `broadcast_shape([2,3], [2,4])` → `Err(BroadcastMismatch)`
- `broadcast_shape([3], [2,4])` → `Err(BroadcastMismatch)`

**集成测试**(`mlx/tests/p1b1_ops.rs`):

- 二元基础:`a + b` 数值正确性(用 `from_slice` 构造,`to_vec` 验证)
- 全 4 引用组合可编译:`a + b` / `&a + b` / `a + &b` / `&a + &b` 各一行
- 标量 RHS:`&a + 1.0_f32`、`a * 2_i32`、`&a / half::f16::from_f32(0.5)`
- 广播:`[2,3] + [3]` 数值正确(主动校验通过)
- 广播 mismatch:`[2,3] + [2,4]` → `Err(BroadcastMismatch { lhs: [2,3], rhs: [2,4] })`
- dtype mismatch(MLX 转发):`f32_array + i32_scalar` 是否 work / err 由 MLX 决定,我们只断言"不 panic、不 abort"
- Neg:`-&a` 数值取反;`-Array::from_slice(&[true, false], &[2])` → `Err(Error::Mlx)`
- Unary 数值精度:`exp(0) == 1`、`log(1) == 0`、`sqrt(4) == 2`、`square(3) == 9`、`reciprocal(2) == 0.5`、`erf(0) == 0`(用 `to_vec::<f32>` 抽样,容忍 1e-6 数值误差)
- Method = free fn:`a.exp()` 与 `ops::exp(&a)` 数值一致
- **能写组合算子**:写一个 `pub fn _softmax_simulated_max(x: &Array, max_val: f32) -> Result<Array>`(因为 reduction 还没,用预设 max 标量),验证 `softmax(&[1.0, 2.0, 3.0])` 数值合理(归一化后总和 ≈ 1)

**回归测试**:P0 + P1a 既有测试不变。

## 文件改动清单

### 新增

- `mlx/src/broadcast.rs` — `broadcast_shape` 推导 + 单元测试
- `mlx/src/ops.rs` — 14 个自由函数(5 binary + 9 unary)
- `mlx/src/ops_impl.rs` — 5 个运算符 trait impl + `Neg` + `forward_ref_binop!` 宏 + 标量 RHS dispatch
- `mlx/tests/p1b1_ops.rs` — 集成测试

### 修改

- `mlx/src/lib.rs` — `mod broadcast; mod ops; mod ops_impl;` + 选择性 re-export(`pub use ops;` 让用户写 `mlx::ops::exp(&a)`)
- `mlx/src/array.rs` — 加 9 个 unary methods,每个 1 行调 `ops::xxx(self)`
- `mlx-sys/src/bridge/array.rs` — 加 14 个 `Result<UniquePtr<MlxArray>>` 函数
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — 加 14 个声明
- `mlx-sys/shim/src/array.cc` — 加 14 个实现(各 1 行)
- `README.md` — 更新 status 行 + 在 Quickstart 后加一段"Operators"小节,展示 `&a + &b`、标量、链式 unary 风格

### 已存在,无变化

- `mlx/src/dtype.rs` / `element.rs` / `error.rs`
- `mlx-sys/build.rs` / `bridge/mod.rs` / `bridge/transforms.rs`
- 所有 P0 / P1a 测试

## 实施分期(P1b1 内)

P1b1 不再细分子阶段,但实施 plan 会按 ~10 个 TDD 任务推进:

1. shim + bridge:14 个新函数,无 Rust 上层
2. `broadcast_shape` 函数 + 单元测试(全在 `broadcast.rs` 内)
3. `ops::add` 自由函数(单一 op,验证 broadcasting + shim 调用 link 通)
4. 把 ops::add 接到 Array methods + 二元运算符 impl(只 Add,验证宏)
5. 用 macro 一次性铺其余 4 个二元运算符(Sub/Mul/Div/Neg)
6. 标量 RHS dispatch(用 macro 把 `impl<T: Element> Add<T>` 等批量生成)
7. 9 个 unary 自由函数 + Array methods + 数值测试
8. README + lib.rs re-export + 全量 verify + clippy

每步 TDD red→green→commit,与 P1a 节奏一致。

## 决策记录

- **B1**:运算符重载 4 种引用组合 + 标量 RHS,通过 `forward_ref_binop!` macro 控制代码量;无标量 LHS(orphan rule)
- **B2**:标量 RHS 接受全 10 个 Element 类型(完整性 > YAGNI 在此处)
- **B3**:dtype 提升交给 MLX 决定,Rust 不在二元 op 主动校验 dtype(broadcasting 校验保留)
- **B4**:广播 Rust 侧前置校验,推导出 `broadcast_shape` 工具供 P1b2 复用
- **B5**:9 个 unary(spec 5 + LLM 必备 4:square/rsqrt/erf/reciprocal),free fn + method 双形态,全 `Result<Array>`
- **B6**:标量 RHS 用 Rust 侧 1-element Array 实现,不开 50 个 per-dtype scalar shim
- **B7**:Neg 在 unsigned/bool 上让 MLX throw,Rust 不主动拒绝
- **B8**:文件按职责拆 `ops.rs` / `ops_impl.rs` / `broadcast.rs`,每个 < 200 行
- **B9**:14 个 shim 函数全部 `Result<UniquePtr<MlxArray>>` 遵守 P1a 硬规则

## 后续 P1b2 / P1c / P2 接口约束

P1b1 暴露的接口在后续阶段视为稳定:

- 5 个运算符 trait impl 集合不变(P1b2 加 `Rem` 等不在 plan 中)
- 9 个 unary 函数签名不变;P1b2 加更多 unary(`abs` 等)是新增不是修改
- `broadcast_shape` 函数签名稳定(P1b2 reduction 直接复用)
- `mlx::ops::*` 模块路径稳定(用户代码已开始引用)
