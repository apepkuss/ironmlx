# cxx-mlx 设计文档

**日期**: 2026-05-03
**状态**: 已批准,待实施
**作者**: 通过 brainstorming 与用户协作产出

## 目标

为 Apple 的 [MLX](https://github.com/ml-explore/mlx) C++ 框架提供一套 Rust 绑定,基于 [`cxx`](https://cxx.rs) crate。绑定的目的是支撑**完整的本地推理应用**,覆盖当前及未来主流模型(LLM、VLM、扩散模型等)所需的核心算子与 IO,但不包括训练/微调链路(`grad`/`vjp`/`vmap` 等高阶 transform 不在范围内,留给未来扩展)。

非目标:

- `mlx::nn`(Python-only,Rust 侧的 `Module`/`Linear`/`Attention` 由用户在 ops 之上自行构建)
- 训练 transforms(`grad`、`vjp`、`jvp`、`vmap`)
- 跨平台支持(MLX 仅 Apple Silicon,绑定也仅在 macOS aarch64 上可构建/运行)

## 核心约束

1. **共享 MLX**:用户机器上多个项目共用一份 MLX 安装。绑定**绝不**写入共享 MLX 路径,只读。
2. **静态链接默认**:避免与共享 dylib 的版本耦合;`bundled` feature 也强制静态。
3. **lazy eval 显式触发**:不在 `Display`/`Debug` 等隐式路径触发 eval,避免性能陷阱。
4. **Rust FFI 生态约定**:两 crate 切分(`-sys` 原始 FFI + 安全包装),与 `openssl-sys`/`openssl` 等同构。

## 工作区结构

```text
cxx-mlx/                         (Cargo workspace 根)
├── Cargo.toml                   (workspace = ["mlx-sys", "mlx"])
├── docs/superpowers/specs/      (本设计 + 后续 P0/P1/P2/P3 子设计)
├── mlx-sys/                     (薄 FFI 层)
│   ├── Cargo.toml
│   ├── build.rs                 (find MLX、compile shim、link)
│   ├── src/
│   │   ├── lib.rs               (re-export bridge 模块)
│   │   └── bridge/              (一个文件一个 MLX 子模块)
│   │       ├── array.rs
│   │       ├── ops.rs
│   │       ├── fast.rs
│   │       ├── random.rs
│   │       ├── io.rs
│   │       ├── transforms.rs
│   │       └── stream.rs
│   └── shim/                    (手写 C++ shim,把 MLX 模板/重载平铺为 cxx 友好的 free function)
│       ├── include/cxx_mlx_shim/*.h
│       └── src/*.cc
└── mlx/                         (safe / idiomatic 层)
    ├── Cargo.toml               (depends on mlx-sys)
    └── src/
        ├── lib.rs
        ├── array.rs             (Array 类型,Drop/Clone/Debug/运算符)
        ├── ops.rs               (free functions + Array methods)
        ├── fast.rs
        ├── random.rs
        ├── io.rs
        ├── transforms.rs
        ├── stream.rs
        ├── dtype.rs
        ├── device.rs
        └── error.rs
```

**Crate 命名**:本地工作区采用 `mlx-sys` + `mlx`。注意 crates.io 上已存在第三方的 `mlx-rs`,如果未来发布,需要重新命名(候选:`mlx-cxx-sys` + `mlx-cxx`),但这不影响本地开发。

## 构建与链接策略(`mlx-sys/build.rs`)

### 默认路径(无 feature)

按以下顺序定位 MLX:

1. `MLX_DIR` 环境变量,指向 install prefix(包含 `lib/cmake/MLX/MLXConfig.cmake`)
2. `MLX_INCLUDE_DIR` + `MLX_LIB_DIR`(分别指向头文件和库文件目录)
3. `pkg-config --cflags --libs mlx`(兜底,实际几乎用不到)

定位到 MLX 后:

1. 用 `cxx_build::bridges([...])` 编译所有桥接 + shim `.cc`,把 MLX 头文件路径加入 include
2. 优先链接 `libmlx.a`,找不到才回落 `libmlx.dylib`;同时链接所有 `MLX_DIR/lib/*.a`(`fmt`、`gguflib`、`json` 等 3rdparty 静态依赖)
3. 链接 macOS 框架:`Metal`、`Foundation`、`Accelerate`、`MetalPerformanceShaders`、`MetalPerformanceShadersGraph`
4. 重新构建触发器:`rerun-if-env-changed=MLX_DIR`、`rerun-if-changed=shim/`
5. 从 `MLX_DIR/include/mlx/version.h` 读出 MLX 版本字符串,通过 `env!` 注入,`mlx::version()` 在 runtime 暴露,便于排查 ABI mismatch

### `bundled` feature(opt-in)

- 不读 `MLX_DIR`,改用 [`cmake`](https://crates.io/crates/cmake) crate 把 MLX 源码作为子项目 build,产物放 `OUT_DIR`,完全不污染共享 MLX 安装
- MLX 源码路径通过 `MLX_SOURCE_DIR` 指定,默认 `../../mlx`(相对 workspace 根),找不到则 build.rs 报错
- 强制 `-DBUILD_SHARED_LIBS=OFF`、`-DMLX_BUILD_PYTHON_BINDINGS=OFF`、`-DMLX_BUILD_EXAMPLES=OFF`
- 静态链接产物

### 平台守卫

- `#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]` 在 `mlx-sys/lib.rs` 顶层 `compile_error!`
- `mlx/lib.rs` 同样守卫
- 文档中明确"仅 Apple Silicon macOS"

## `mlx-sys`:cxx 桥接 + shim

### 为什么需要 shim

cxx 的限制:

- 不支持 C++ 模板、函数重载、隐式转换
- `std::vector<T>` 中 `T` 是 opaque 类型时不能跨越
- 不支持 `mlx::core::array(T scalar)` 这类构造模板

MLX 的 C++ API 大量使用模板和重载(`add(array, array)` / `add(array, T)` / `multiply` 同理)。所以**必须**在 C++ 侧手写一层 shim,把模板和重载平铺成命名后缀的 free function。

### shim 模式示例

```cpp
// mlx-sys/shim/include/cxx_mlx_shim/ops.h
#pragma once
#include <memory>
#include <mlx/mlx.h>
#include "rust/cxx.h"

namespace cxx_mlx::ops {
  std::unique_ptr<mlx::core::array> add(
    const mlx::core::array& a, const mlx::core::array& b);
  std::unique_ptr<mlx::core::array> add_scalar_f32(
    const mlx::core::array& a, float b);
  std::unique_ptr<mlx::core::array> matmul(
    const mlx::core::array& a, const mlx::core::array& b);
  std::unique_ptr<mlx::core::array> reshape(
    const mlx::core::array& a, rust::Slice<const int32_t> shape);
  // ...
}
```

### Array 桥接示例

```rust
// mlx-sys/src/bridge/array.rs
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");
        type MlxArray;  // opaque,真身是 mlx::core::array

        fn array_from_f32(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray>;
        fn array_from_i32(data: &[i32], shape: &[i32]) -> UniquePtr<MlxArray>;
        fn array_zeros(shape: &[i32], dtype: u8) -> UniquePtr<MlxArray>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
        fn array_dtype(a: &MlxArray) -> u8;
        fn array_size(a: &MlxArray) -> usize;
        fn array_ndim(a: &MlxArray) -> usize;
        fn array_clone(a: &MlxArray) -> UniquePtr<MlxArray>;  // C++ copy ctor,内部 refcount
        fn array_to_vec_f32(a: &MlxArray) -> Result<Vec<f32>>;
        fn array_item_f32(a: &MlxArray) -> Result<f32>;
    }
}
```

### Vec\<Array\> 边界

`eval(std::vector<array>)` 这种 cxx 不支持。Shim 提供:

- `eval_one(const MlxArray& a)`
- `eval_many(rust::Slice<const MlxArray*>)` —— Rust 侧用 `Vec<&Array>` 收集后转 raw pointer slice 调用

### Dtype 编码

Dtype 用 `u8` enum repr 跨边界传递。Rust 侧 `Dtype` 枚举手动同步 `mlx::core::Dtype::Val` 的数值。在 shim 里加 `static_assert(static_cast<uint8_t>(mlx::core::float32) == 5)` 这类守卫,MLX 升级改了枚举顺序时立刻失败。

### 桥接组织

每个 MLX 子模块对应一个 `bridge/<module>.rs`,内部一个 `cxx::bridge` mod。`mlx-sys/src/lib.rs` 把所有桥接 `pub use ffi`。这样维护时一个 PR 集中改一个子模块,不会撞车。

## `mlx`:safe 层

### Array 类型

```rust
pub struct Array(cxx::UniquePtr<sys::array::MlxArray>);

impl Clone for Array {
    fn clone(&self) -> Self {
        Array(sys::array::array_clone(&self.0))  // C++ 内部共享 storage
    }
}

impl std::ops::Add for &Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array { ops::add(self, rhs) }
}
// 同理 Sub / Mul / Div / Neg

impl Array {
    pub fn from_slice<T: Element>(data: &[T], shape: &[i32]) -> Array;
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Array;
    pub fn ones(shape: &[i32], dtype: Dtype) -> Array;
    pub fn shape(&self) -> Vec<i32>;
    pub fn dtype(&self) -> Dtype;
    pub fn size(&self) -> usize;
    pub fn ndim(&self) -> usize;
    pub fn reshape(&self, shape: &[i32]) -> Array;
    pub fn transpose(&self, axes: &[i32]) -> Array;
    pub fn t(&self) -> Array;  // 最后两维交换
    pub fn matmul(&self, rhs: &Array) -> Array;
    pub fn eval(&self) -> Result<()>;
    pub fn item<T: Element>(&self) -> Result<T>;
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>>;
}
```

### `Element` trait

统一 dtype 校验和泛型分发:

```rust
pub trait Element: sealed::Sealed + Copy {
    const DTYPE: Dtype;
    fn from_array(a: &Array) -> Result<Vec<Self>>;
    fn item_from_array(a: &Array) -> Result<Self>;
}
```

实现集合:`f32`、`f16`(`half::f16`)、`bf16`(`half::bf16`)、`i8`、`i16`、`i32`、`i64`、`u8`、`u16`、`u32`、`u64`、`bool`、`complex64`(后续)。

### lazy eval 显式

- `Debug` 打印 `Array { shape: [B, S, D], dtype: bfloat16, evaluated: false }`,**不**触发 eval
- `Display` 同上,如果想要数值需 `arr.eval()?; arr.to_vec()?` 自己组合
- 提供 `arr.eval_blocking()` 和 `arr.eval_async() -> Future`(后者依赖 `mlx::async_eval` + 一个 `Stream`)

### 模块组织

`ops` / `fast` / `random` / `io` / `transforms` / `stream` / `device` / `dtype` / `error` 各一个文件。`Array` 上的方法是对 `ops::*` 的转发,**链式**风格优先(`a.matmul(&b).reshape(&[B, S, D])`)。

## 错误处理

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MLX runtime error: {0}")]
    Mlx(String),  // 来自 MLX C++ 异常
    #[error("dtype mismatch: expected {expected:?}, got {actual:?}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<i32>, actual: Vec<i32> },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

- MLX C++ 异常通过 cxx 的 `Result<T>` 自动桥转为 `Error::Mlx`
- Rust 侧主动校验(dtype/shape)优先于让 C++ 抛,信息更具体
- 不在 `to_vec`/`item` 这类 hot-path 上分配额外 String 除非真出错

## 测试与 CI

### `mlx-sys`

- Smoke test:能 link、能创建 `array_zeros`、能 `eval`、能读 shape

### `mlx`

- Op 数值正确性:为 `add`/`matmul`/`softmax`/`rms_norm`/`rope`/`sdpa` 等关键 op 写小输入 + 写死期望输出
- IO:用一个 `tests/fixtures/tiny.safetensors`(几十 KB 假权重)和 `tiny.gguf` 验证加载
- `fast::scaled_dot_product_attention` vs naive matmul+softmax 实现的一致性
- 大模型 e2e 不放在测试里,放 `examples/`

### 平台

- macOS Apple Silicon only
- 非 Mac 上 `cargo check` 报清晰的 `compile_error!` 而非 link 错误
- CI 暂不设置(本地开发为主),如未来需要,GitHub Actions `macos-14`(M1)runner 可用

## 实施分期

| 期 | 范围 | 验收 |
| --- | --- | --- |
| **P0** | 脚手架可链:workspace、`build.rs` 找到 MLX、最小 shim(`array_zeros` + `eval` + 读 shape),`cargo test` 跑通"创建 zeros、eval、读 shape" | `cargo test -p mlx-sys -p mlx` 全绿 |
| **P1** | Array + 核心 ops:`Array` safe 包装 / 运算符 / shape 操作(reshape, transpose, broadcast, concatenate, split, stack)/ reduction(sum, mean, max, min, argmax)/ indexing(take, gather, where, slice)/ 广播 +-*/ / matmul / 激活构建块(exp, log, sqrt, tanh, sigmoid)/ `random`(key + uniform/normal/categorical) | 关键 op 数值测试通过;能写 softmax/gelu/silu 等组合算子 |
| **P2** | 推理关键路径:`fast`(rms_norm, layer_norm, rope, sdpa)/ `io::load_safetensors`/`load_gguf` / `transforms::eval`/`async_eval`/`synchronize` / `Stream`/`Device` / 完整 dtype 含 f16/bf16 | 能加载真实 safetensors 文件、跑通一个 transformer block 的 forward |
| **P3** | 量化与 compile:`fast::affine_quantize`/`affine_dequantize` / `compile`,以及 `examples/inference_demo.rs`(Rust 跑一个小开源 LLM 推理)。`fast::metal_kernel` 标注为可延后(自定义 Metal 源码桥接非推理必需) | example 能 generate 出一段连贯文本 |

每期单独有 `docs/superpowers/specs/YYYY-MM-DD-cxx-mlx-pN-design.md` 设计、单独的 implementation plan、单独的 PR。下一步先写 P0 的 implementation plan,P0 跑通后基于实际经验再写 P1/P2/P3 的 plan。

## 决策记录

- **A1(crate 命名)**:本地用 `mlx-sys` + `mlx`,发布时再处理重名
- **A2(链接策略)**:默认静态(`.a` 优先),`bundled` 强制静态
- **A3(共享 MLX)**:`build.rs` 只读 `MLX_DIR`,绝不写入
- **A4(lazy eval)**:显式 `eval()`,无隐式触发
- **A5(fast 模块)**:全部覆盖(rms_norm / layer_norm / rope / sdpa / metal_kernel / affine_quantize / affine_dequantize)
- **A6(io)**:safetensors + gguf 读,无写
- **A7(transforms)**:eval / async_eval / synchronize / compile,不含 grad/vjp/vmap
- **A8(API 分层)**:`-sys` + safe 两 crate(标准 Rust FFI 约定)
- **A9(分期)**:P0 脚手架 → P1 ops → P2 fast+io → P3 量化+compile,每期单独 plan/PR
