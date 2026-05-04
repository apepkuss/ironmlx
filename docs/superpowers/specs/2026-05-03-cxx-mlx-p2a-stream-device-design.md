# cxx-mlx P2a — Stream/Device + async transforms 设计文档

**日期**: 2026-05-03
**父设计**: [`2026-05-03-cxx-mlx-design.md`](2026-05-03-cxx-mlx-design.md)(P2 在 Roadmap 中)
**前置**: P1 完整完成(P1b2b 合入 master,commit `ac81331`)
**状态**: 已批准,待实施

## 目标

P2 的基础设施层。**P2a** 交付:

- **`Device` 类型**:`cpu` / `gpu` 枚举 + index,Rust 端 zero-cost POD struct 与 MLX 二进制兼容
- **`Stream` 类型**:`{ index, device }` POD struct,共享 layout
- **设备查询/设置 API**:`default_device` / `set_default_device` / `is_available` / `device_count`
- **Stream 管理 API**:`default_stream(d)` / `new_stream(d)` / `set_default_stream(s)` / `get_streams()` / `clear_streams()`
- **异步 transforms**:`Array::async_eval` 和 `mlx::async_eval(&[&Array])` 返回 runtime-agnostic `impl Future<Output = Result<()>>`,实现走 `blocking` crate
- **同步 transforms**:`mlx::synchronize()`(默认 stream)和 `mlx::synchronize_stream(Stream)`(指定 stream),阻塞当前线程

P2b 后续:`fast` ops(rms_norm/layer_norm/rope/sdpa)+ f16/bf16 SDPA 测试 + gather 文档加例子。
P2c 后续:`io::load_safetensors` / `io::load_gguf`。

非目标:
- 现有 ops 加 `Stream` 参数 → 不做(API 翻倍 + breaking;LLM 推理 99% 单 stream 够;P3 fast 需要时再加)
- `compile()` → P3(JIT 编译 closure,FFI 复杂)
- `ThreadLocalStream` → P3(高级用法)
- `device_info()` 返回 `unordered_map<string, variant>` → P3(跨桥麻烦)

## 关键决策

### A1. Device / Stream 用 cxx shared POD struct

MLX C++ 端 `Device` 和 `Stream` 都是 POD struct(无 vtable、无智能指针),layout 固定:
- `Device { DeviceType type; int index; }` — 8 字节
- `Stream { int index; Device device; }` — 12 字节(假设无 padding)

cxx 1.0 支持 shared struct(`#[cxx::bridge]` 内 `struct` 声明),两端用同一 layout,跨边界传值零拷贝零分配。

```rust
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(u8)]
    enum DeviceType {
        Cpu = 0,
        Gpu = 1,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Device {
        device_type: DeviceType,
        index: i32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Stream {
        index: i32,
        device: Device,
    }

    extern "Rust" {}

    unsafe extern "C++" {
        include!("cxx_mlx_shim/stream.h");

        // Device queries
        fn default_device() -> Device;
        fn set_default_device(d: Device);
        fn is_available(d: Device) -> bool;
        fn device_count(t: DeviceType) -> i32;

        // Stream lifecycle
        fn default_stream(d: Device) -> Stream;
        fn new_stream(d: Device) -> Result<Stream>;
        fn set_default_stream(s: Stream);
        fn get_streams() -> Vec<Stream>;
        fn clear_streams();

        // Transforms
        unsafe fn async_eval_many(arrays: &[*const MlxArray]) -> Result<()>;
        fn synchronize() -> Result<()>;
        fn synchronize_stream(s: Stream) -> Result<()>;
    }
}
```

**为什么 shared struct 不是 opaque 类型**:Device/Stream 不是 cxx::UniquePtr 管理的资源(没有析构函数副作用)。POD 直接传值,Rust 端可以构造 / 比较 / hash,不需 FFI 函数读字段。Opaque 包装会强制每次访问字段都走 FFI,无意义开销。

**`#[derive(Hash)]`** 在 cxx shared struct 上是可用的(cxx 1.0.140+)。如果 cxx 版本不支持,fallback 到手写 impl。

**Shim 头文件**:`mlx-sys/shim/include/cxx_mlx_shim/stream.h` 提供 cxx 生成的 `Device`/`Stream`/`DeviceType` 与 MLX 原生 `mlx::core::Device`/`Stream` 之间的 trivial 转换函数(memcpy 兼容,但类型名不同)。

### A2. `Device::cpu()` / `Device::gpu(index)` 安全层构造

Safe `mlx` crate 提供 ergonomic 构造和重导出:

```rust
// mlx/src/device.rs
pub use mlx_sys::stream::ffi::{Device, DeviceType};

impl Device {
    pub const fn cpu() -> Self {
        Device { device_type: DeviceType::Cpu, index: 0 }
    }

    pub const fn gpu(index: i32) -> Self {
        Device { device_type: DeviceType::Gpu, index }
    }
}

pub fn default_device() -> Device {
    mlx_sys::stream::ffi::default_device()
}
pub fn set_default_device(d: Device) {
    mlx_sys::stream::ffi::set_default_device(d);
}
pub fn is_available(d: Device) -> bool {
    mlx_sys::stream::ffi::is_available(d)
}
pub fn device_count(t: DeviceType) -> i32 {
    mlx_sys::stream::ffi::device_count(t)
}
```

`mlx::Device` / `mlx::DeviceType` / `mlx::Stream` 顶层 re-export(用户 `use mlx::Device;` 就够了)。

### A3. `Stream` 安全层

```rust
// mlx/src/stream.rs
pub use mlx_sys::stream::ffi::Stream;

pub fn default_stream(d: Device) -> Stream {
    mlx_sys::stream::ffi::default_stream(d)
}
pub fn new_stream(d: Device) -> Result<Stream> {
    mlx_sys::stream::ffi::new_stream(d).map_err(Error::from)
}
pub fn set_default_stream(s: Stream) {
    mlx_sys::stream::ffi::set_default_stream(s);
}
pub fn get_streams() -> Vec<Stream> {
    mlx_sys::stream::ffi::get_streams()
}
pub fn clear_streams() {
    mlx_sys::stream::ffi::clear_streams();
}
```

**线程语义**:`set_default_stream` / `set_default_device` 在 MLX C++ 是 thread-local("Make the stream the default for its device on current thread")。Rust 端不重复包装 thread-local,直接镜像 MLX 行为,文档说明。

### A4. `async_eval` 返回 `impl Future`,实现走 `blocking` crate

```rust
// mlx/src/transforms.rs
use crate::{Array, Result, Stream};

/// Asynchronously evaluate one or more arrays.
///
/// Submits the computation graph to MLX's stream worker (non-blocking) on
/// the **caller's thread's** default stream, then returns a `Future` that
/// resolves when the work completes. The future is runtime-agnostic —
/// `.await` it under tokio, async-std, smol, `futures::executor::block_on`,
/// or any executor.
///
/// **Cancellation**: dropping the returned future without awaiting does
/// **not** cancel the submitted MLX work — MLX has no cancellation
/// primitive. The work runs to completion in the background, consuming
/// GPU time and memory. Any subsequent operation on the same arrays will
/// implicitly synchronize.
///
/// Implementation note: the returned future uses the [`blocking`] crate's
/// global thread pool to wrap MLX's `synchronize_stream()`. The future
/// captures the submission stream at construction time and synchronizes
/// on it explicitly, so the future can be polled on any thread (not just
/// the submitter's thread). Scheduling overhead is ~5µs per call,
/// negligible vs typical MLX kernel times (µs–ms).
pub fn async_eval(arrays: &[&Array]) -> impl Future<Output = Result<()>> + Send + use<> {
    // Capture the submission stream on THIS thread, BEFORE submission.
    // MLX `async_eval` queues work on the caller-thread's default stream;
    // we must wait on that exact stream regardless of which thread polls
    // the returned future. Otherwise `blocking::unblock`'s pool thread
    // would call `synchronize()` against ITS default stream (which has
    // no queued work) and the future would resolve before MLX finishes.
    let device = mlx_sys::stream::ffi::default_device();
    let stream = mlx_sys::stream::ffi::default_stream(device);

    // Step 1 (sync, fast): build raw pointer slice + submit to MLX.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: pointers valid for the duration of this fn (we hold &Array refs).
    // MLX `async_eval` copies arrays internally (refcount-share), so pointers
    // need not outlive THIS function — only the submission.
    let submit_result = unsafe { mlx_sys::stream::ffi::async_eval_many(&raw) };

    // Step 2 (returned future): explicitly synchronize on the captured stream
    // via `blocking`. `Stream` is `Copy` (POD), so it moves cleanly into the
    // closure with no lifetime concerns.
    async move {
        submit_result.map_err(crate::Error::from)?;
        blocking::unblock(move || {
            mlx_sys::stream::ffi::synchronize_stream(stream).map_err(crate::Error::from)
        })
        .await
    }
}

impl Array {
    /// See [`mlx::transforms::async_eval`]. Convenience method for a single array.
    /// The returned future does not borrow `self` (submission consumes the
    /// `&Array` reference; future captures only the owned Stream + submit result).
    pub fn async_eval(&self) -> impl Future<Output = Result<()>> + Send + use<> {
        async_eval(&[self])
    }
}

/// Block the current thread until all queued work on the **current thread's
/// default stream** completes.
pub fn synchronize() -> Result<()> {
    mlx_sys::stream::ffi::synchronize().map_err(crate::Error::from)
}

/// Block the current thread until all queued work on the **given stream**
/// completes (regardless of which thread queued it).
pub fn synchronize_stream(s: Stream) -> Result<()> {
    mlx_sys::stream::ffi::synchronize_stream(s).map_err(crate::Error::from)
}
```

**设计点**:
- **Stream capture at submit time**:这是关键修复。MLX 的 `synchronize()`(无参)阻塞**调用线程的 default stream**(thread-local)。`blocking::unblock` 在 pool 的另一个线程 B 上跑 closure,B 的 default stream ≠ 提交线程 A 的 default stream。如果在 B 上调 `synchronize()`,等的是 B 的空 stream → future 误报完成。**修复:在 submit 时 capture A 的 default stream,future 内部用 `synchronize_stream(captured)`**(Stream-specific 版本不依赖 thread-local default)
- **Submit 在调用线程同步执行**(< 1µs):用户 `let fut = async_eval(...)` 立即返回 Future,但 MLX submission 已经发生
- **Cancellation 语义**:doc 明确说 drop future 不取消 MLX 工作 — MLX 无 cancellation 机制,工作仍跑完。后续 `eval` / `to_vec` 隐式 sync
- **`use<>` / `use<'_>` 捕获语法**(Rust 1.82+ precise capturing):明确 Future 的 lifetime / Send bound
- **Future is `Send`**:`Stream` is `Copy + Send`,closure 只 capture 它(无 `&Array` 引用),所以 future 可跨线程
- **Multi-array form 是 base case**;single-array `Array::async_eval` 是 1-行包装

**Lifetime 分析**:
- `arrays: &[&Array]` 生命周期止于 `async_eval` 函数返回
- `raw` Vec 生命周期同上,在 submit 调用期间有效
- MLX `async_eval(vector<array>)` 在 C++ 侧拷贝 array(refcount-shared),不持有原指针
- 因此 returned Future 不依赖 `arrays` lifetime — Future capture 的是 `Stream`(POD `Copy`)+ `Result<()>`,完全 owned,可跨任意 await 边界 / 线程

### A5. `synchronize` 同步 API 保留

虽然有 `async_eval`,`synchronize` 仍单独暴露。理由:
- 用户在 sync 上下文(无 runtime)需要 block 等待
- 实现 `async_eval` future 时内部就要调它,单独 export 零额外成本
- LLM 推理在 sampling / detokenize 步骤需要拿数据,直接 `arr.eval()?; arr.to_vec()?` 比 `block_on(arr.async_eval()).await` 短

### A6. 现有 P1 ops 不加 `Stream` 参数

确认延续 P1 设计:30+ 个 op 不接 `Stream` 参数,默认走 thread-local default stream。用户切换 stream 的方式:

```rust
let s = mlx::new_stream(mlx::Device::gpu(0))?;
mlx::set_default_stream(s);
// 后续 ops 都在 s 上
let result = a.matmul(&b)?;
```

**为什么不加**:
- 30+ op 加 stream 重载 → API 数量翻倍(60+),代码膨胀
- LLM 推理 99% 单 stream(MLX 默认 GPU stream),多 stream 主要用在 prefill+decode 重叠(P3 fast 优化阶段)
- 现 API 是 `pub fn add(a: &Array, b: &Array) -> Result<Array>`,改成 `pub fn add(a: &Array, b: &Array, s: impl Into<Option<Stream>>) -> Result<Array>` 是 breaking change(虽然小,但跨 28 个签名 + 28 个 method)
- P3 fast SDPA 真正需要 multi-stream 时,设计 `with_stream(s, |s| ...)` scope guard 模式(类似 `set_default_stream` 但 RAII),不污染 op 签名

### A7. shim 端 Result-wrapping

P1 硬规则继续:可 throw 的函数 `Result<T>` 包装。
- `new_stream` → MLX 在无可用 device 时 throw → `Result<Stream>`
- `async_eval_many` / `synchronize` / `synchronize_stream` → MLX 内部错误 throw → `Result<()>`
- `default_device` / `set_default_device` / `is_available` / `device_count` / `default_stream` / `set_default_stream` / `get_streams` / `clear_streams` → 不 throw(query / setter,无失败模式) → 普通返回

### A8. 文件组织

新增:
- `mlx-sys/src/bridge/stream.rs` — cxx::bridge 模块,所有 device/stream/transforms 的 FFI 声明 + shared struct
- `mlx-sys/shim/include/cxx_mlx_shim/stream.h` — shim 头(转换 cxx-generated 类型 ↔ mlx::core 类型)
- `mlx-sys/shim/src/stream.cc` — shim 实现
- `mlx/src/device.rs` — `Device` re-export + ergonomic constructors + 顶层 device 函数
- `mlx/src/stream.rs` — `Stream` re-export + 顶层 stream 函数
- `mlx/src/transforms.rs` — `async_eval` / `synchronize` / `synchronize_stream`(P0 的 `eval` 留在 array.rs 不动)
- `mlx/tests/p2a_device.rs` — Device 基础测试
- `mlx/tests/p2a_stream.rs` — Stream 生命周期测试
- `mlx/tests/p2a_async.rs` — async_eval / synchronize 集成测试

修改:
- `mlx-sys/src/lib.rs` — 加 `pub mod bridge::stream`
- `mlx-sys/build.rs` — `cxx_build::bridges([..., "src/bridge/stream.rs"])` + `.file("shim/src/stream.cc")`
- `mlx/Cargo.toml` — 加 `blocking = "1"`(精确版本号在 plan 里查 crates.io 当前 stable)
- `mlx/src/lib.rs` — `mod device; mod stream; mod transforms;` + re-exports
- `README.md` — 新增"Streams & Devices"小节 + 异步示例

### A9. 测试策略

**Device 基础**(`mlx/tests/p2a_device.rs`):
- `Device::cpu()` / `Device::gpu(0)` 字段值正确
- `default_device()` 在 macOS Apple Silicon 上是 `gpu`
- `is_available(cpu) == true`、`is_available(gpu) == true`
- `device_count(gpu) >= 1`
- `set_default_device(cpu)` 后 `default_device() == cpu`,然后 `set_default_device(gpu)` 复位
- 相等性:`Device::gpu(0) == Device::gpu(0)`、`!= Device::cpu()`

**Stream 基础**(`mlx/tests/p2a_stream.rs`):
- `default_stream(gpu)` 返回有效 stream
- `new_stream(gpu)` 返回新 stream(`index` 与 default 不同)
- `get_streams()` 至少包含 default 和新建的
- `set_default_stream(new)` 后 `default_stream(d) == new`,然后 reset
- `clear_streams()` 不 panic,后续 `default_stream` 仍可用(MLX 重新创建)
- 相等性 + Hash trait 编译通过(确保 `#[derive]` 工作)

**Async 集成**(`mlx/tests/p2a_async.rs`):
- `Array::zeros(&[1024], Float32)?.async_eval().await?` 在 `futures::executor::block_on` 下完成
- 同上但用 `tokio::runtime::Builder::new_current_thread().build()?.block_on(...)` 跑通(verify runtime-agnostic)
- `mlx::async_eval(&[&a, &b]).await?` 多 array
- 提交 → 等 → `to_vec` 数据正确性
- `mlx::synchronize()` 阻塞 OK(创建 lazy zeros,sync,验证 `is_available`)
- 不带 future 的 sync 路径:`a.eval()?; mlx::synchronize()?` 工作

**回归**:P0/P1 全部 140 测试不变,新加约 15-20 个测试,workspace 总数 ~155+。

### A10. 实施分期(P2a 内)

约 7 个 TDD 任务:

1. **shim + bridge for stream.rs**:cxx::bridge with shared structs + 13 FFI 函数 + shim header/cc + 4 个 sys-side smoke tests
2. **`mlx::device` 模块**:Device re-export + cpu/gpu constructors + 4 个顶层函数 + 单元测试 + p2a_device.rs 集成测试(7 tests)
3. **`mlx::stream` 模块**:Stream re-export + 5 个顶层函数 + p2a_stream.rs 集成测试(6 tests)
4. **`mlx::transforms` 模块 + `blocking` 依赖**:`synchronize` / `synchronize_stream` 同步 API,加 `blocking = "1"` 到 mlx/Cargo.toml,验证 build
5. **`async_eval` Future 实现**:`pub fn async_eval(&[&Array]) -> impl Future`,内部 submit + `blocking::unblock` 等待
6. **`Array::async_eval` method + p2a_async 集成测试**:6 测试,包括 tokio + futures::executor 各跑一次
7. **README + final verify + clippy + doc**

每步 TDD red→green→commit,与 P1 节奏一致。

## 决策记录

- **B1**:Device/Stream 用 cxx shared POD struct(zero-overhead value passing,与 MLX layout 二进制兼容)
- **B2**:`Device::cpu()` / `Device::gpu(index)` const fn constructors,top-level re-export
- **B3**:thread-local `set_default_*` 语义直接镜像 MLX,Rust 不额外包装
- **B4**:`async_eval` 返回 `impl Future<Output = Result<()>> + Send`,实现走 `blocking` crate(runtime-agnostic)
- **B5**:`synchronize` / `synchronize_stream` 同步 API 保留(sync context + future 内部都用)
- **B6**:现有 30+ op 不加 `Stream` 参数(P3 fast 阶段如需多 stream 设计 RAII scope guard,不污染签名)
- **B7**:`compile()` / `ThreadLocalStream` / `device_info()` 延后 P3
- **B8**:shim Result-wrapping 硬规则继续 — query/setter 不 throw 走普通返回,可 throw 走 `Result`

## 后续 P2b / P2c / P3 接口约束

- `Stream` / `Device` 的 layout 进入稳定 ABI,P2b fast ops 可以接受 `&Stream` 参数(future-compatible additive)
- `async_eval` 签名 `&[&Array] -> impl Future` 稳定,P2b/P2c 的 io load 完成时也用 future 链,不引入新 async 模式
- `synchronize_stream(Stream)` 签名稳定,P3 fast 多 stream attention 直接调
- `blocking` 是 P2a 引入的第一个 async-related dep,P2b/P2c 可复用,不再额外 dep
