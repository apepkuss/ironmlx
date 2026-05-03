# cxx-mlx P1a — Array Foundation 设计文档

**日期**: 2026-05-03
**父设计**: [`2026-05-03-cxx-mlx-design.md`](2026-05-03-cxx-mlx-design.md)
**状态**: 已批准,待实施
**前置**: P0 已合入 master(commit `1923627`)

## 目标

为 `mlx::Array` 加上算子层(P1b)和 IO 层(P2)所需的全部基础能力,**不**实现任何算子本身。具体包括:

- `Element` trait + 10 个数据类型(覆盖 LLM 推理所需的 f16/bf16,无需等到 P2)
- `from_slice<T>` / `item<T>` / `to_vec<T>` 数据进出方向的 API
- `Clone`(廉价 MLX refcount)、`Debug`(不触发 eval)
- `Send` 标记(`!Sync`,与 `std::shared_ptr` 语义对齐)
- `Error` 变体扩展为 P1b 算子校验做准备
- shape 改用 `SmallVec<[i32; 8]>` 消除算子链中的高频堆分配
- `static_assert` 端点扩展防止 MLX dtype 枚举漂移
- 全 shim 改为强制 `Result` 包装可 throw 函数(回填 P0 的 `array_zeros`)

非目标:
- 任何算子(reshape / transpose / matmul / reduction / 元素 op) → P1b
- random / random key → P1c
- complex64 / u16 / u32 / u64 → 不在 P1 范围(YAGNI)
- async eval / Stream 模型 → P2

## 关键决策

### A1. `Array: Send`,不 `Sync`

**根据**:MLX `array` 内部用 `std::shared_ptr<ArrayDesc>`,refcount 是 atomic,但 `ArrayDesc` 字段(`shape`/`status`/`event`/`is_tracer`/`data`)无 mutable / atomic / mutex。多个 const 方法实际改 state(`set_status` / `attach_event` / `is_available` 触发 `detach_event` + `set_status`)。

**结论**:跨线程 move 安全;两个线程并发持有 `&Array` 同一对象不安全。

**用户编写多线程代码的官方推荐**:`Clone`(廉价,MLX 内部 refcount 共享 storage),不要默认上 `Arc<Mutex<Array>>`。这会在 README 显式说明。

### A2. `Element` trait 与覆盖类型

```rust
mod sealed {
    pub trait Sealed {}
}

pub trait Element: sealed::Sealed + Copy + Send + 'static {
    const DTYPE: Dtype;
}
```

10 个 impl,每个类型同时 `impl sealed::Sealed for T` 和 `impl Element for T`(标准 sealed 模式,阻止下游 crate impl Element):`bool` / `u8` / `i8` / `i16` / `i32` / `i64` / `half::f16` / `half::bf16` / `f32` / `f64`。

`half` crate 是事实标准,`half::f16` / `half::bf16` 内存布局与 MLX 的 `mlx::core::float16_t` / `mlx::core::bfloat16_t` 兼容(都是 16 位 POD,同序),shim 端通过 `reinterpret_cast` 桥接。

**Sealed pattern 必要性**:外部 crate 若能 impl Element,可以构造任意 `T → Dtype` 映射违反 FFI 类型安全(如 `impl Element for String { const DTYPE = Dtype::Float32; }`)。Sealed 阻止这种滥用。

### A3. `shape()` 返回 `SmallVec<[i32; 8]>`

`smallvec = "1"` 加到 `mlx` crate 依赖。`SmallVec<[i32; 8]>` 栈上存 ≤ 8 维(覆盖 99% 推理张量),9+ 维堆分配。算子链中读 shape 实际是零开销。

**Public API 影响**:`Array::shape() -> SmallVec<[i32; 8]>`(P0 的 `Vec<i32>` 是 breaking)。`SmallVec` 实现 `Deref<Target=[T]>`,大部分用法兼容。P0 的两个测试断言改用 `.as_slice() == &[2, 3]`。

`Array::shape_at(dim: i32) -> i32` 顺手补一个,对应 `mlx::core::array::shape(int dim)`(支持负索引)。

### A4. `Error` 变体扩展

```rust
#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX runtime error: {0}")]
    Mlx(String),

    #[error("dtype mismatch: expected {expected:?}, got {actual:?}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },

    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<i32>, actual: Vec<i32> },

    #[error("broadcast mismatch: lhs {lhs:?} vs rhs {rhs:?}")]
    BroadcastMismatch { lhs: Vec<i32>, rhs: Vec<i32> },
}
```

`From<cxx::Exception>` 仍只产生 `Mlx(String)`。P1b 算子在 Rust 侧主动校验 → 用具体变体。`Vec<i32>`(非 `SmallVec`)因为 Error 不在 hot path,且 `Display` 对 `Vec` 输出更直观。

### A5. `static_assert` 端点扩展

`mlx-sys/shim/src/array.cc` 增加两条:

```cpp
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::bool_) == 0,
              "Dtype::Val::bool_ ordinal changed");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::float32) == 10,
              "Dtype::Val::float32 ordinal changed");  // 已有
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::complex64) == 13,
              "Dtype::Val::complex64 ordinal changed");
```

中间任何枚举值被插入,端点至少有一条会偏移触发。

### A6. shim 函数 `Result` wrapping 硬规则

**规则**:任何可能 `throw` 的 shim 函数,Rust bridge 必须声明 `Result<T>` 返回类型。否则 cxx 走 `std::terminate`,程序异常终止而非可恢复错误。

**P1a 应用**:
- `array_zeros` 接受 `dtype: u8` → 可 throw → **改为 `Result<UniquePtr<MlxArray>>`**(P0 遗留,P1a 顺手回填)
- `array_from_slice_<T>` 接受 shape → shape 元素积可能不等于 data.len() → throw → `Result`
- `array_item_<T>` / `array_to_vec_<T>` → MLX 在未 eval / dtype 不匹配等情况可能 throw → `Result`
- `array_clone` → C++ copy ctor 不 throw(noexcept by mlx::core::array contract) → 不用 `Result`
- `array_shape` / `array_dtype` / `array_ndim` / `array_size` / `array_is_available` → 纯 getter → 不用 `Result`

写在 P1a spec 决策记录里 + 在 `mlx-sys/src/bridge/mod.rs` 顶部用注释固化:"凡可 throw,必 Result"。

### A7. `Clone` / `Debug` / Display 语义

- `impl Clone for Array`:走 `array_clone` shim,C++ copy ctor,共享 storage(MLX 内部 refcount++)。**廉价**
- `impl Debug for Array`:输出 `Array { shape: [2, 3], dtype: Float32, evaluated: true }`。**绝对不**触发 eval。`evaluated` 通过新 shim `array_is_available(&MlxArray) -> bool` 读取
- `impl Display for Array`:**不实现**。要看值用 `arr.eval()? + arr.to_vec()?`

### A8. `from_slice` / `item` / `to_vec` API

```rust
impl Array {
    pub fn from_slice<T: Element>(data: &[T], shape: &[i32]) -> Result<Array>;
    pub fn item<T: Element>(&self) -> Result<T>;
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>>;
}
```

**Rust 侧主动校验(优先于让 C++ 抛)**:
- `from_slice`:`data.len() == shape.iter().product::<i32>() as usize`,否则 `Err(ShapeMismatch)`
- `from_slice`:无需校验 dtype,因 `T: Element` 强制类型安全
- `item`:`self.size() == 1`,否则 `Err(Mlx("item() called on non-scalar"))`
- `item` / `to_vec`:`self.dtype() == T::DTYPE`,否则 `Err(DtypeMismatch)`

**`to_vec` 的隐式 eval**:为节省调用方 `arr.eval()?; arr.to_vec()?` 的语法噪音,`to_vec` 和 `item` 内部自动调用 `eval_one`(如果尚未 evaluated)。这是对 spec A4 ("显式 eval")的明确**例外**:`to_vec` / `item` 必须有数据才能拷,语义上隐式 eval 是合理的。`Display` / `Debug` / `shape` / `dtype` / `ndim` / `size` 仍不触发 eval。

### A9. 文件组织

P1a 全部新代码放 `mlx/src/array.rs`(预计完成时 200-300 行)+ 新增 `mlx/src/element.rs` 单独放 `Element` trait 和 10 个 impl(否则 Element 模块会喧宾夺主)。

P1b 加运算符时再考虑拆 `array.rs` → `array.rs` + `array_io.rs` + `array_ops.rs`。

## 文件改动清单

### 新增

- `mlx/src/element.rs` — `Element` trait + sealed pattern + 10 个 impl
- `mlx-sys/shim/src/element_shim.cc.in` 或直接展开 → `mlx-sys/shim/include/cxx_mlx_shim/array.h` / `mlx-sys/shim/src/array.cc` 各加 30 个函数(10 dtype × 3 操作)
- `mlx/tests/p1a_*.rs` — 单元 + 集成测试

### 修改

- `mlx/Cargo.toml` — 加 `half = "2"`、`smallvec = "1"`
- `mlx/src/lib.rs` — 导出 `Element`
- `mlx/src/array.rs` — 加 `Clone` / `Debug` / `from_slice` / `item` / `to_vec` / `shape_at`,`shape()` 改返回 `SmallVec`
- `mlx/src/dtype.rs` — 视情况加 `Dtype::size()` 辅助方法
- `mlx/src/error.rs` — 加 3 个新变体
- `mlx-sys/src/bridge/array.rs` — `array_zeros` 改 `Result<UniquePtr<MlxArray>>`,加 `array_clone` / `array_is_available` / 30 个 element FFI
- `mlx-sys/src/bridge/mod.rs` — 顶部加 "凡可 throw,必 Result" 注释规则
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` / `mlx-sys/shim/src/array.cc` — 加新 shim 函数 + 端点 static_assert
- `mlx/tests/p0_smoke.rs` — `arr.shape() == vec![2,3]` → `arr.shape().as_slice() == &[2,3]`(SmallVec breaking)
- `mlx/src/array.rs` — `Array::zeros` 签名从 `pub fn zeros(...) -> Self` 改为 `pub fn zeros(...) -> Result<Self>`(因新 shim 返回 `Result`,且 `dtype: u8` 校验可 throw)。P0 测试同步更新为 `Array::zeros(...).expect(...)` 或 `?`。Breaking change,但 P1a 阶段可接受

### Element FFI 命名约定

```rust
// mlx-sys/src/bridge/array.rs
fn array_from_f32(data: &[f32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
fn array_from_i32(data: &[i32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
fn array_from_bool(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;  // bool 走 u8 桥
// ... 其余 7 个

fn array_to_vec_f32(a: &MlxArray) -> Result<Vec<f32>>;
// ... 其余 9 个

fn array_item_f32(a: &MlxArray) -> Result<f32>;
// ... 其余 9 个
```

`bool` 在 cxx 桥接 `&[bool]` 不支持(cxx 1.0 限制),改走 `&[u8]`,Rust 侧 `Element for bool` 在传入前做 `bool → u8` 拷贝(P1a 内部用 `Vec<u8>` 中转)。

`half::f16` / `half::bf16` 在 Rust 侧用 `as u16` 转成 `&[u16]` 跨桥;C++ 侧 `reinterpret_cast<mlx::core::float16_t*>(data.data())` 还原。安全前提是两端 16 位类型布局一致(POD,无 padding,小端) —— `half` crate 文档明确保证这点,加 compile-time check `mem::size_of::<half::f16>() == 2`。

## 测试策略

P1a 测试分为三组:

### 单元测试(`mlx/src/element.rs` / `array.rs` 内 `#[cfg(test)] mod tests`)
- `Element::DTYPE` 对每个类型返回正确 `Dtype` 枚举值

### 集成测试(`mlx/tests/p1a_io.rs`)
- `from_slice<T>` 各 dtype 的 round-trip(创建 + `to_vec<T>` 读回 + 数值相等)
- `from_slice` 的 shape mismatch → `Err(ShapeMismatch)`
- `from_slice` `f16` / `bf16` 的位模式正确性(写入特定 bit 模式 → 读回 bit 模式相同)
- `item<T>` 标量 round-trip,且 dtype 不匹配 → `Err(DtypeMismatch)`
- `item<T>` 非标量 → `Err`
- `to_vec` 隐式 eval(创建 lazy zeros,直接 `to_vec` 不需先 `eval()`,数据正确)

### 集成测试(`mlx/tests/p1a_array.rs`)
- `Clone` 与原 `Array` 共享数据(创建 + clone + 各自 `to_vec` 数值相等;clone 后释放原 array,clone 仍可用)
- `Debug` 输出包含 shape / dtype / evaluated 字段且不触发 eval(创建 lazy → format → 仍 lazy)
- `Send` 编译时验证:`fn assert_send<T: Send>() {}` + `assert_send::<Array>();`
- `!Sync` 编译时验证:负向测试 doesn't compile(用 `static_assertions::assert_not_impl_any!`)
- `shape()` 返回 `SmallVec`,栈分配在 ≤ 8 维不触发 heap

### 回归测试(P0 既有)
- `mlx-sys/tests/sys_smoke.rs`:`array_zeros` 改 `Result` 后,测试改 `.expect("zeros should succeed")`
- `mlx/tests/p0_smoke.rs`:`shape()` 返回值改用 `.as_slice()` 比较,`zeros` 改 `Array::zeros(...).expect("zeros should succeed")` 或 `?`
- 静态测试:静态断言 `Dtype::Val` 端点不变(已在 shim 通过 `static_assert`)

## 实施分期(P1a 内)

P1a 不再细分子阶段,但实施 plan 会按 8-10 个 TDD 任务推进:

1. 加依赖(`half`、`smallvec`),拉通构建
2. `Error` 加 3 个变体,Display 测试
3. shim 端点 `static_assert`
4. `Element` trait + 10 个 impl + 单元测试
5. shim `array_zeros` 改 `Result`,Rust `Array::zeros` 跟着改 `Result`,P0 测试同步更新
6. shim `array_clone` + Rust `impl Clone for Array` + 测试
7. `array_is_available` shim + Rust `impl Debug for Array` + 不触发 eval 测试
8. shim `array_from_<T>` 全 10 个 + Rust `from_slice<T>` + shape mismatch 校验 + 测试
9. shim `array_item_<T>` + Rust `item<T>` + dtype/size 校验 + 测试
10. shim `array_to_vec_<T>` + Rust `to_vec<T>` + 隐式 eval + 测试
11. `shape()` 改 `SmallVec`,P0 测试同步更新,`shape_at` 顺手补
12. `Send` 标记 + `Send`/`!Sync` 编译时测试
13. 文档(README "线程安全"小节 + crate 级 doc)

每步 TDD red → green → commit。

## 决策记录

- **B1**:`Array: Send`,不 `Sync`(基于 MLX `array` 内部线程安全调研结论:NEGATIVE)
- **B2**:`Element` 覆盖 10 个类型(bool + 6 个整数 + f16 + bf16 + f32 + f64),`half` crate 依赖
- **B3**:`shape()` 返回 `SmallVec<[i32; 8]>`,`smallvec` 依赖,P0 断言更新
- **B4**:Error 加 `DtypeMismatch` / `ShapeMismatch` / `BroadcastMismatch`(`Vec<i32>`,非 SmallVec)
- **B5**:`static_assert` 加端点(bool_=0、complex64=13)+ 已有的 float32=10
- **B6**:shim Result-wrapping 硬规则,P0 `array_zeros` 回填,P1b 起严格执行
- **B7**:`Clone`(廉价 refcount)、`Debug`(no-eval)、不实现 `Display`
- **B8**:`from_slice` 主动 shape 校验、`item` 主动 size + dtype 校验、`to_vec` 主动 dtype 校验、`to_vec` / `item` 隐式 eval(spec A4 显式 eval 原则的明确例外,理由:必须有数据才能拷)
- **B9**:P1a 文件组织 — `array.rs` + `element.rs` 两个文件,P1b 再细分

## 后续 P1b / P1c 接口约束

P1a 暴露的接口在 P1b / P1c 中视为稳定:
- `Element` trait 不再扩展类型(complex64 仍延后)
- `from_slice` / `item` / `to_vec` 的签名不变
- `Error` 可加新变体(non-exhaustive)但已有变体不变
- `Clone` / `Debug` 行为契约不变
