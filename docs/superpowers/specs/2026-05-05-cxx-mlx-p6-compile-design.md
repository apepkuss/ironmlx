# cxx-mlx P6: compile 设计文档

**日期：** 2026-05-05
**作者：** Claude (与 Boss 协作)
**目标阶段：** P6 — `mlx::core::compile` 完整闭包绑定

---

## 1. 目标与范围

为 cxx-mlx 提供 MLX `compile()` 的完整闭包绑定，使 Rust 用户能够把任意 Rust 闭包传给 MLX 进行图追踪、融合和加速重放。同时绑定全局 compile 控制 API（启用/禁用/模式切换）。

### 1.1 公开 API（Rust 侧）

| API | 说明 |
|---|---|
| `mlx::compile::CompileMode` | enum：`Disabled`/`NoSimplify`/`NoFuse`/`Enabled` |
| `mlx::compile::disable_compile()` | 全局禁用 compile |
| `mlx::compile::enable_compile()` | 全局启用 compile |
| `mlx::compile::set_compile_mode(mode)` | 设置模式 |
| `mlx::compile::compile(f, shapeless) -> Result<CompiledFn>` | 把 Rust 闭包追踪为编译图 |
| `mlx::compile::CompiledFn::invoke(&self, &[&Array]) -> Result<Vec<Array>>` | 执行已编译图 |

### 1.2 类型

- `CompileMode`（`#[repr(u8)]` 枚举）
- `CompiledFn`（opaque RAII 句柄，持有 C++ `std::function`）

### 1.3 不在范围内

- `compile_clear_cache()` 等 MLX 内部 API（暂不需要）
- 函数指针重载（只用 `std::function` 路径，能覆盖闭包和函数）

---

## 2. 架构

### 2.1 三层 FFI

```
Rust 用户闭包  ──>  cxx::bridge   ──>  C++ shim  ──>  mlx::core::compile
   ^                                                          │
   │                                                          ▼
   └──── extern "Rust" CompileCallback::invoke <─── std::function lambda
```

### 2.2 类型映射

| 概念 | C++ | cxx::bridge | Rust |
|---|---|---|---|
| 数组列表 | `std::vector<array>` | `UniquePtr<ArrayVec>` | `&[&Array]` / `Vec<Array>` |
| 闭包 | `std::function<vector<array>(vector<array>)>` | extern "Rust" `CompileCallback` | `Box<dyn Fn>` |
| 编译产物 | `std::function<...>` | `UniquePtr<CompiledFn>` | `CompiledFn` |
| 模式 | `mlx::core::CompileMode` | `u8` | `CompileMode` 枚举 |

### 2.3 关键设计决策

**ArrayVec 双向 opaque：** Rust 不能直接传/收 `Vec<UniquePtr<MlxArray>>`（cxx 1.0 不支持 `Vec<UniquePtr<T>>`）。沿用 P2c LoadResult 的"opaque 容器 + 一次性 take"模式，但 ArrayVec 需要双向：

- C++ → Rust：MLX trace 时把 `vector<array>` 包装成 `UniquePtr<ArrayVec>` 给 Rust 回调
- Rust → C++：Rust 回调返回的 `Vec<Array>` 通过 push 写入新的 `ArrayVec`，再交还给 C++

**CompileCallback 通过 extern "Rust"：** 用 `Box<dyn Fn(&[&Array]) -> Result<Vec<Array>> + Send + Sync>` 包装用户闭包，cxx::bridge 暴露 `invoke` 方法。C++ 把 `rust::Box<CompileCallback>` 装进 `shared_ptr` 以便在 `std::function` lambda 中拷贝捕获（`rust::Box` 自身不可拷贝，只能 move）。

**错误传播：** Rust 闭包返回 `Err` 或 panic 时，cxx 桥会自动转 C++ 异常，由 MLX trace 阶段或 invoke 阶段抛出，再被外层 cxx 转回 Rust `Err`。

---

## 3. 详细设计

### 3.1 ArrayVec opaque 类型

**C++（`cxx_mlx_shim/compile.h`）：**

```cpp
struct ArrayVec {
  std::vector<mlx::core::array> inner;
};

std::unique_ptr<ArrayVec> array_vec_new();
size_t array_vec_count(const ArrayVec& v);
std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i);  // clone
std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i);        // move + erase
void array_vec_push(ArrayVec& v, const MlxArray& a);
```

**cxx::bridge：**

```rust
unsafe extern "C++" {
    type ArrayVec;
    fn array_vec_new() -> UniquePtr<ArrayVec>;
    fn array_vec_count(v: &ArrayVec) -> usize;
    fn array_vec_get_at(v: &ArrayVec, i: usize) -> UniquePtr<MlxArray>;
    fn array_vec_take_at(v: Pin<&mut ArrayVec>, i: usize) -> UniquePtr<MlxArray>;
    fn array_vec_push(v: Pin<&mut ArrayVec>, a: &MlxArray);
}
```

`get_at` 用 `array.copy_shared_buffer()` 共享底层数据；`take_at` 移除元素。

### 3.2 CompileCallback（extern "Rust"）

**Rust 类型：**

```rust
pub struct CompileCallback {
    f: Box<dyn Fn(&[&Array]) -> Result<Vec<Array>, MlxError> + Send + Sync>,
}

impl CompileCallback {
    fn invoke(&self, inputs: &ArrayVec) -> Result<UniquePtr<ArrayVec>, MlxError> {
        let n = array_vec_count(inputs);
        // 把 ArrayVec 转成 Vec<Array> 引用
        let arrays: Vec<Array> = (0..n)
            .map(|i| Array::from_unique_ptr(array_vec_get_at(inputs, i)))
            .collect();
        let refs: Vec<&Array> = arrays.iter().collect();
        let outputs = (self.f)(&refs)?;
        let mut out_vec = array_vec_new();
        for a in &outputs {
            array_vec_push(out_vec.pin_mut(), a.as_inner());
        }
        Ok(out_vec)
    }
}
```

**cxx::bridge：**

```rust
extern "Rust" {
    type CompileCallback;
    fn invoke(self: &CompileCallback, inputs: &ArrayVec)
        -> Result<UniquePtr<ArrayVec>>;
}
```

### 3.3 compile_with_callback（C++ shim）

```cpp
struct CompiledFn {
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      fn;
};

std::unique_ptr<CompiledFn> compile_with_callback(
    rust::Box<CompileCallback> cb, bool shapeless) {
  // shared_ptr 包装：std::function 需要 CopyConstructible，rust::Box 不可拷贝
  auto shared_cb = std::make_shared<rust::Box<CompileCallback>>(std::move(cb));

  auto traced = mlx::core::compile(
      [shared_cb](const std::vector<mlx::core::array>& inputs)
          -> std::vector<mlx::core::array> {
        // 把 inputs 包成 ArrayVec
        auto in_vec = std::make_unique<ArrayVec>();
        in_vec->inner = inputs;
        // 调 Rust 回调（cxx 已生成的方法签名）
        auto out_vec = (*shared_cb)->invoke(*in_vec);
        return std::move(out_vec->inner);
      },
      shapeless);

  auto out = std::make_unique<CompiledFn>();
  out->fn = std::move(traced);
  return out;
}

std::unique_ptr<ArrayVec> compiled_fn_invoke(
    const CompiledFn& cf, const ArrayVec& inputs) {
  auto outputs = cf.fn(inputs.inner);
  auto v = std::make_unique<ArrayVec>();
  v->inner = std::move(outputs);
  return v;
}
```

### 3.4 全局控制

```cpp
void disable_compile();
void enable_compile();
void set_compile_mode(uint8_t mode);  // 0..3
```

Rust enum 通过 `as u8` 传过去；shim 内部 switch 转回 `mlx::core::CompileMode`。

### 3.5 错误传播表

| 错误源 | 传播路径 |
|---|---|
| Rust 闭包返回 `Err(e)` | extern "Rust" `invoke` 返回 `Result::Err` → cxx 抛 C++ 异常 → MLX trace 抛出 → `compile_with_callback` 调用栈中冒出 → cxx 桥转回 Rust `Err` |
| Rust 闭包 panic | cxx `catch_unwind` 转 C++ 异常 → 同上 |
| MLX trace 自身错误（形状不匹配等） | C++ 异常 → cxx 桥转 Rust `Err` |
| invoke 阶段错误 | 同上 |

### 3.6 Rust 公开 API

```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    Disabled = 0,
    NoSimplify = 1,
    NoFuse = 2,
    Enabled = 3,
}

pub fn disable_compile();
pub fn enable_compile();
pub fn set_compile_mode(mode: CompileMode);

pub struct CompiledFn { /* opaque */ }

impl CompiledFn {
    pub fn invoke(&self, inputs: &[&Array]) -> Result<Vec<Array>, MlxError>;
}

pub fn compile<F>(f: F, shapeless: bool) -> Result<CompiledFn, MlxError>
where
    F: Fn(&[&Array]) -> Result<Vec<Array>, MlxError> + Send + Sync + 'static;
```

---

## 4. 集成测试（9 个）

| 名称 | 目的 |
|---|---|
| `compile_mode_setters` | 调 disable/enable/set_mode 不 panic |
| `compile_simple_unary` | 单输入单输出 `x => x + 1`，编译后调用结果正确 |
| `compile_two_input` | 两输入 `(a,b) => a*b + a` |
| `compile_captures_weight` | 闭包捕获外部 `Array`（权重），多次 invoke 结果一致 |
| `compile_shapeless_reuse` | `shapeless=true` 后用不同 shape 调用都成功 |
| `compile_callback_error_propagates` | 闭包返 `Err`，trace 阶段冒出 Rust `Err` |
| `compile_callback_panic_caught` | 闭包 panic，转成 Rust `Err`，进程不崩 |
| `array_vec_round_trip` | 单元测试：push N 个，count==N，take_at 后 count 减 1 |
| `top_level_re_exports_work` | `mlx::compile::compile` 可达 |

---

## 5. 任务分解

1. **Skeleton + 全局控制 + CompileMode**：scaffold shim/bridge/safe API 文件，实现 disable/enable/set_compile_mode 与枚举，1 个测试。
2. **ArrayVec opaque 双向桥**：`array_vec_new/count/get_at/take_at/push` 全套，1 个 round-trip 测试。
3. **CompiledFn + compile() + invoke() 闭包回调**（核心）：CompileCallback extern "Rust"、shim `compile_with_callback`、CompiledFn::invoke、6 个集成测试。
4. **Re-export + README + 最终验证**：在 `mlx/src/lib.rs` 选择性 re-export，更新 README 进度章节，跑完整测试套件。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| `rust::Box<CompileCallback>` 不可拷贝 → `std::function` 要求 CopyConstructible | 用 `shared_ptr<rust::Box<...>>` 包一层 |
| Rust 闭包 panic 跨 FFI | cxx 自动 `catch_unwind`；测试用例显式覆盖 |
| 闭包内部线程安全（MLX 可能多次调用） | trait bound 要求 `Send + Sync` |
| ArrayVec 双向使用易混淆（输入 vs 输出） | 文档明确：每次 trace `in_vec` 由 C++ 创建给 Rust 读，`out_vec` 由 Rust 创建归还给 C++ |
| 多次 invoke 同一 CompiledFn | `std::function` 自身可重入，无额外状态 |
| 环境变量 `MLX_DISABLE_COMPILE` 干扰测试 | 测试不依赖环境变量；显式调 `enable_compile()` 保证状态 |

---

## 7. 与后续阶段的关系

P6 引入的 extern "Rust" 闭包回调模式可被未来阶段复用：

- **P7 linalg**：纯算子绑定，不需要回调，直接走 P5 模式
- **P8 fft**：同 P7
- **P9 fast::metal_kernel**：用户提供 Metal 源码，是字符串而非闭包，但若需要 grad 回调可参考 P6
- **P10 distributed**：进程间通信，可能需要 reduce 回调

P6 的 `ArrayVec` 双向 opaque 类型也可在未来需要 `Vec<Array>` 跨 FFI 的场景复用。

---
