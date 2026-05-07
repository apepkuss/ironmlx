# cxx-mlx P3a: `fast::metal_kernel` binding 设计文档

**日期：** 2026-05-06
**作者：** Claude（与 Boss 协作）
**目标阶段：** P3a — cxx-mlx 绑定 `mlx::core::fast::metal_kernel`，让 ironmlx 能写自定义 Metal kernel

---

## 1. 范围与决策

绑定 MLX C++ `mlx::core::fast::metal_kernel(...)` 到 cxx-mlx，作为 ironmlx P3b（Qwen3.5 特殊算子，特别是 `gated_delta_step` SSM kernel）的前置基础设施。

P3a 是纯 cxx-mlx 工作，不影响 ironmlx 已有代码。

### 1.1 已批准的设计决策

| # | 主题 | 决定 |
|---|---|---|
| Q1 | API 形态 | **A 链式 builder** — 构造期 `MetalKernel::builder(name).inputs(...).outputs(...).source(...).build()`；调用期 `kernel.dispatch_builder().inputs(...).grid(...)....dispatch()` |
| Q2 | 必填参数处理 | **M2 typestate** — 5 个必填字段（inputs / output_shapes / output_dtypes / grid / threadgroup）用 5 个 marker 类型（`Unset`/`Set`），setter 转换状态，`.dispatch()` 仅在所有 marker 为 `Set` 时存在 |
| Q3 | 多输出表示 | **N1 pure** — `dispatch()` 返回 P6 已有的 `ArrayVec`，按 index `take_at(i)` |

**支撑原则**（已记入 user memory `feedback_performance_stability_priority.md`）：
- compile-time 保证 > runtime 检查
- 优化的 hot path > 简单的慢 fallback
- 显式 bounds > 隐式信任

### 1.2 推导出的实施细节

- **TemplateArg 跨 cxx**: 3 个 typed setter (`.template_int` / `.template_bool` / `.template_dtype`)；C++ side 累积到 `vector<pair<string, TemplateArg>>`
- **output_shapes 跨 cxx**: 一次性 `output_shapes(&[shape1, shape2])` 传入；C++ side 用 opaque `ShapesVec` 双向桥（与 ArrayVec 模式对应）
- **MetalKernel 内部 opaque**: C++ side `MetalKernelInner` 包装 `mlx::core::fast::CustomKernelFunction`（`std::function`）；Rust 端持有 `Arc<MetalKernelInner>` 实现 cheap clone + 多线程共享
- **dispatch 返回多输出**: C++ side `CustomKernelFunction(...)` → `std::vector<array>` → `ArrayVec` 返回 → Rust 按 index take
- **Stream 集成**: dispatch builder 加可选 setter `.stream(target: impl Into<StreamOrDevice>)`，默认 `StreamOrDevice::Default`（P5.7 contract）

### 1.3 不在范围内

- `mlx::core::fast::cuda_kernel` / `precompiled_cuda_kernel`（Apple Silicon 不需要）
- `header` Metal source 共享头 — 保留 setter，不深度测试
- `atomic_outputs` + `init_value` — 仅 atomic outputs 用例需要；保留 setter 但不深度测试（P3b gated_delta 不用）

---

## 2. 架构

```
mlx-sys/
├── shim/
│   ├── include/cxx_mlx_shim/fast.h          # +metal_kernel decls + ShapesVec + TemplateArgC
│   └── src/fast.cc                           # +impl
└── src/bridge/fast.rs                        # +bridge entries

mlx/src/fast/
├── mod.rs                                    # 现有 (rms_norm/layer_norm/rope/sdpa) +pub mod metal_kernel
└── metal_kernel/                             # NEW
    ├── mod.rs                                # MetalKernel + MetalKernelBuilder
    └── dispatch.rs                           # DispatchBuilder + typestate markers + TemplateArg
```

`mlx/src/fast.rs` 当前是单文件。P3a 把它改成目录 `mlx/src/fast/`，已有内容拆到 `mlx/src/fast/mod.rs` 顶层（rms_norm / layer_norm / rope / sdpa 仍可用，路径不变）。

---

## 3. 详细设计

### 3.1 公开 API

```rust
// crate root re-export (mlx/src/lib.rs)
pub use crate::fast::metal_kernel::{
    DispatchBuilder, MetalKernel, MetalKernelBuilder, Set, TemplateArg, Unset,
};
```

**`MetalKernel` 结构**

```rust
pub struct MetalKernel {
    inner: Arc<MetalKernelInner>,
}

impl MetalKernel {
    pub fn builder(name: impl Into<String>) -> MetalKernelBuilder;

    pub fn dispatch_builder(&self) -> DispatchBuilder<Unset, Unset, Unset, Unset, Unset>;
}

impl Clone for MetalKernel {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

struct MetalKernelInner {
    handle: cxx::UniquePtr<mlx_sys::fast::ffi::MetalKernelInner>,
    output_count: usize,    // 用于 dispatch 时 sanity check
}
```

`MetalKernel::clone()` 是 refcount inc，多 layer / 多线程共享 kernel 安全。

**`MetalKernelBuilder`**

```rust
pub struct MetalKernelBuilder {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    source: String,
    header: String,                       // 默认 ""
    ensure_row_contiguous: bool,          // 默认 true
    atomic_outputs: bool,                 // 默认 false
}

impl MetalKernelBuilder {
    pub fn inputs(mut self, names: &[&str]) -> Self;
    pub fn outputs(mut self, names: &[&str]) -> Self;
    pub fn source(mut self, src: impl Into<String>) -> Self;
    pub fn header(mut self, hdr: impl Into<String>) -> Self;
    pub fn ensure_row_contiguous(mut self, v: bool) -> Self;
    pub fn atomic_outputs(mut self, v: bool) -> Self;
    pub fn build(self) -> Result<MetalKernel>;
}
```

`build()` 内部：
- 验证 inputs / outputs / source 都非空
- 调 cxx `metal_kernel_build` 函数
- 失败返回 `Err(Error::Mlx(...))`（编译错误等）
- 成功返回 `MetalKernel { inner: Arc::new(...) }`

### 3.2 typestate dispatch builder

**Marker 类型**

```rust
pub struct Unset;
pub struct Set;
```

**字段布局**

```rust
pub struct DispatchBuilder<I, OS, OD, G, TG> {
    kernel: Arc<MetalKernelInner>,

    // 必填字段
    inputs: Option<Vec<*const mlx_sys::array::ffi::MlxArray>>,
    output_shapes: Option<Vec<Shape>>,
    output_dtypes: Option<Vec<Dtype>>,
    grid: Option<(i32, i32, i32)>,
    threadgroup: Option<(i32, i32, i32)>,

    // 可选字段
    template_args: Vec<(String, TemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
    target: StreamOrDevice,

    _markers: PhantomData<(I, OS, OD, G, TG)>,
}
```

**TemplateArg enum**

```rust
#[derive(Debug, Clone)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}
```

**5 个必填 setter**

```rust
impl<OS, OD, G, TG> DispatchBuilder<Unset, OS, OD, G, TG> {
    pub fn inputs(self, arrays: &[&Array]) -> DispatchBuilder<Set, OS, OD, G, TG> {
        let raw: Vec<*const _> = arrays.iter().map(|a| a.as_inner() as *const _).collect();
        DispatchBuilder {
            kernel: self.kernel,
            inputs: Some(raw),
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: self.grid,
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OD, G, TG> DispatchBuilder<I, Unset, OD, G, TG> {
    pub fn output_shapes(self, shapes: &[Shape]) -> DispatchBuilder<I, Set, OD, G, TG>;
}

impl<I, OS, G, TG> DispatchBuilder<I, OS, Unset, G, TG> {
    pub fn output_dtypes(self, dtypes: &[Dtype]) -> DispatchBuilder<I, OS, Set, G, TG>;
}

impl<I, OS, OD, TG> DispatchBuilder<I, OS, OD, Unset, TG> {
    pub fn grid(self, gx: i32, gy: i32, gz: i32) -> DispatchBuilder<I, OS, OD, Set, TG>;
}

impl<I, OS, OD, G> DispatchBuilder<I, OS, OD, G, Unset> {
    pub fn threadgroup(self, tx: i32, ty: i32, tz: i32) -> DispatchBuilder<I, OS, OD, G, Set>;
}
```

**6 个可选 setter（不动 marker）**

```rust
impl<I, OS, OD, G, TG> DispatchBuilder<I, OS, OD, G, TG> {
    pub fn template_int(mut self, name: impl Into<String>, v: i32) -> Self {
        self.template_args.push((name.into(), TemplateArg::Int(v)));
        self
    }
    pub fn template_bool(mut self, name: impl Into<String>, v: bool) -> Self;
    pub fn template_dtype(mut self, name: impl Into<String>, v: Dtype) -> Self;
    pub fn init_value(mut self, v: f32) -> Self;
    pub fn verbose(mut self, v: bool) -> Self;
    pub fn stream(mut self, target: impl Into<StreamOrDevice>) -> Self;
}
```

**dispatch 仅在全 Set 时存在**

```rust
impl DispatchBuilder<Set, Set, Set, Set, Set> {
    pub fn dispatch(self) -> Result<ArrayVec> {
        let inputs = self.inputs.expect("typestate: inputs is Set");
        let output_shapes = self.output_shapes.expect("typestate: output_shapes is Set");
        let output_dtypes = self.output_dtypes.expect("typestate: output_dtypes is Set");
        let grid = self.grid.expect("typestate: grid is Set");
        let threadgroup = self.threadgroup.expect("typestate: threadgroup is Set");

        // Sanity: output_shapes.len() == output_dtypes.len() == kernel.output_count
        if output_shapes.len() != self.kernel.output_count {
            anyhow::bail!(
                "MetalKernel dispatch: output_shapes count {} != declared outputs {}",
                output_shapes.len(),
                self.kernel.output_count,
            );
        }
        // 同理 output_dtypes.len()

        let (has_stream, dev_only, dev_t, idx) = self.target.encode();

        // ShapesVec 跨 cxx
        let mut shapes_vec = mlx_sys::fast::ffi::shapes_vec_new();
        for s in &output_shapes {
            mlx_sys::fast::ffi::shapes_vec_push(shapes_vec.pin_mut(), s.as_slice());
        }

        // ArrayVec for inputs（输入侧）
        let mut input_vec = mlx_sys::compile::ffi::array_vec_new();   // P6 ArrayVec
        for ptr in &inputs {
            // SAFETY: ptr 来自 Array::as_inner，借用有效期为本调用
            unsafe { mlx_sys::compile::ffi::array_vec_push(input_vec.pin_mut(), &**ptr) };
        }

        // template_args 转换
        let template_c: Vec<TemplateArgC> = self.template_args.iter()
            .map(|(name, val)| TemplateArgC::from(name, val))
            .collect();

        // dtype reprs
        let dtype_reprs: Vec<u8> = output_dtypes.iter().map(|d| d.as_u8()).collect();

        // SAFETY: 所有 raw pointer 在调用期内有效
        let result = unsafe {
            mlx_sys::fast::ffi::metal_kernel_dispatch(
                &self.kernel.handle,
                &input_vec,
                &shapes_vec,
                &dtype_reprs,
                grid.0, grid.1, grid.2,
                threadgroup.0, threadgroup.1, threadgroup.2,
                &template_c,
                self.init_value.is_some(), self.init_value.unwrap_or(0.0),
                self.verbose,
                has_stream, dev_only, dev_t, idx,
            )
        }
        .map_err(Error::from)?;

        Ok(ArrayVec::from_inner(result))
    }
}
```

### 3.3 跨 cxx 边界 — C++ shim

```cpp
// mlx-sys/shim/include/cxx_mlx_shim/fast.h

namespace cxx_mlx {

// === Opaque types ===

struct MetalKernelInner {
    mlx::core::fast::CustomKernelFunction fn;
};

struct ShapesVec {
    std::vector<mlx::core::Shape> shapes;
};

// === TemplateArgC (cxx-friendly) ===

struct TemplateArgC {
    rust::String name;
    uint8_t kind;      // 0=int, 1=bool, 2=dtype
    int32_t int_val;   // valid for kind 0 or 2 (Dtype repr in 2)
    bool bool_val;     // valid for kind 1
};

// === ShapesVec API ===
std::unique_ptr<ShapesVec> shapes_vec_new();
void shapes_vec_push(ShapesVec& v, rust::Slice<const int32_t> shape);
size_t shapes_vec_count(const ShapesVec& v);

// === metal_kernel ===

std::unique_ptr<MetalKernelInner> metal_kernel_build(
    rust::Str name,
    rust::Slice<const rust::String> input_names,
    rust::Slice<const rust::String> output_names,
    rust::Str source,
    rust::Str header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

std::unique_ptr<ArrayVec> metal_kernel_dispatch(
    const MetalKernelInner& kernel,
    const ArrayVec& inputs,
    const ShapesVec& output_shapes,
    rust::Slice<const uint8_t> output_dtypes,
    int32_t gx, int32_t gy, int32_t gz,
    int32_t tx, int32_t ty, int32_t tz,
    rust::Slice<const TemplateArgC> template_args,
    bool has_init, float init_value,
    bool verbose,
    bool has_stream, bool dev_only, uint8_t dev_type, int32_t stream_idx);

} // namespace cxx_mlx
```

**`metal_kernel_build` 实现要点**：
- 把 `rust::Slice<const rust::String>` → `std::vector<std::string>`
- 调 `mlx::core::fast::metal_kernel(name, input_names, output_names, source, header, ...)` 返回 `CustomKernelFunction`
- 包装到 `MetalKernelInner` 返回 unique_ptr

**`metal_kernel_dispatch` 实现要点**：
- 把 `inputs` (`ArrayVec`) 转 `std::vector<array>`（refcount share）
- 把 `output_shapes` (`ShapesVec`) 转 `std::vector<Shape>`
- 把 `output_dtypes` (`Slice<u8>`) → `std::vector<Dtype>` (用 `helpers::dtype_from_repr`)
- 把 `template_args` (`Slice<TemplateArgC>`) → `std::vector<pair<string, TemplateArg>>`
- `init_value` → `std::optional<float>`
- stream → `StreamOrDevice` (用 `helpers::decode_stream_or_device`)
- 调 `kernel.fn(inputs, output_shapes, output_dtypes, grid_tuple, threadgroup_tuple, template_args, init_value, verbose, stream)` 返回 `std::vector<array>`
- 包装到 `ArrayVec` 返回

### 3.4 跨 cxx 边界 — Rust bridge

```rust
// mlx-sys/src/bridge/fast.rs

#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    struct TemplateArgC {
        name: String,
        kind: u8,
        int_val: i32,
        bool_val: bool,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MetalKernelInner;
        type ShapesVec;
        type ArrayVec = crate::compile::ffi::ArrayVec;   // 复用 P6 ArrayVec
        type MlxArray = crate::array::ffi::MlxArray;

        // ShapesVec
        fn shapes_vec_new() -> UniquePtr<ShapesVec>;
        fn shapes_vec_push(v: Pin<&mut ShapesVec>, shape: &[i32]);
        fn shapes_vec_count(v: &ShapesVec) -> usize;

        // metal_kernel
        fn metal_kernel_build(
            name: &str,
            input_names: &[String],
            output_names: &[String],
            source: &str,
            header: &str,
            ensure_row_contiguous: bool,
            atomic_outputs: bool,
        ) -> Result<UniquePtr<MetalKernelInner>>;

        unsafe fn metal_kernel_dispatch(
            kernel: &MetalKernelInner,
            inputs: &ArrayVec,
            output_shapes: &ShapesVec,
            output_dtypes: &[u8],
            gx: i32, gy: i32, gz: i32,
            tx: i32, ty: i32, tz: i32,
            template_args: &[TemplateArgC],
            has_init: bool, init_value: f32,
            verbose: bool,
            has_stream: bool, dev_only: bool, dev_type: u8, stream_idx: i32,
        ) -> Result<UniquePtr<ArrayVec>>;
    }
}
```

### 3.5 调用示例（gated_delta_step，P3b 时实际写法）

```rust
const KERNEL_SRC: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    // ... ~150 行 Metal C++ ...
"#;

// 构造期（lazy 全局，类似 mlx-lm 的 lru_cache）
static GATED_DELTA_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
fn get_kernel() -> &'static MetalKernel {
    GATED_DELTA_KERNEL.get_or_init(|| {
        MetalKernel::builder("gated_delta_step")
            .inputs(&["q", "k", "v", "g", "beta", "state", "T"])
            .outputs(&["y", "new_state"])
            .source(KERNEL_SRC)
            .build()
            .expect("kernel compiles")
    })
}

// 调用期（每 layer forward 一次）
let mut outputs = get_kernel().dispatch_builder()
    .inputs(&[&q, &k, &v, &g, &beta, &state, &t_arr])
    .output_shapes(&[
        Shape::from((b, t, hv, dv)),
        state.shape(),
    ])
    .output_dtypes(&[input_type, state_type])
    .grid(32, dv, b * hv)
    .threadgroup(32, 4, 1)
    .template_dtype("InT", input_type)
    .template_dtype("StT", state_type)
    .template_int("Dk", dk)
    .template_int("Dv", dv)
    .template_int("Hk", hk)
    .template_int("Hv", hv)
    .dispatch()?;

let y = outputs.take_at(0)?;
let new_state = outputs.take_at(1)?;
```

漏调任何一个必填 setter（如 `.inputs(...)`），编译期就会 fail：

```
error[E0599]: no method named `dispatch` found for struct
              `DispatchBuilder<Unset, Set, Set, Set, Set>`
```

actionable：用户读 marker 顺序（按 spec § 3.2 字段顺序：I, OS, OD, G, TG）即可定位缺哪个。

---

## 4. 测试策略

### 4.1 单元测试

- `mlx-sys/tests/sys_smoke.rs` 增加：
  - `metal_kernel_build_links`：trivial kernel 构造（Metal 源码 `output[gid] = input[gid] + 1.0`）
  - `metal_kernel_dispatch_links`：trivial kernel dispatch + verify 输出数值

- `mlx/src/fast/metal_kernel/dispatch.rs` 内：
  - `template_arg_int_bool_dtype_setters` — 验证三种 setter 累积正确
  - `dispatch_builder_typestate_traversal` — 走 5 必填 setter 顺序，编译通过

- `mlx/src/fast/metal_kernel/mod.rs` 内：
  - `kernel_clone_is_arc` — clone 后两个 handle 引用同一 inner

### 4.2 集成测试

`mlx-sys/tests/p3a_metal_kernel.rs`:

- `simple_add_kernel` — Metal 源码 `output[gid] = input[gid] + 1.0`，dispatch + 数值验证
- `multi_output_kernel` — Metal 源码同时输出 `y = x * 2` 和 `z = x + 10`，验证 ArrayVec 双输出取出顺序

### 4.3 编译期 typestate 验证

`mlx/tests/trybuild/`:
- `metal_kernel_missing_inputs.rs` — 不调 `.inputs()` → `.dispatch()` 不存在 → expected compile error
- `metal_kernel_missing_grid.rs` — 不调 `.grid()` → 同上
- 等 5 个 trybuild 测试

每个 trybuild test 期望文件（`.stderr`）记录预期编译错误，CI 用 `trybuild` crate 验证。

### 4.4 P3a 不直接测试 gated_delta

P3a 范围是 metal_kernel 通用 binding。`gated_delta_step` kernel 是 P3b 任务。

---

## 5. 任务分解

7 个任务：

| # | 任务 | 主要文件 | 估时 |
|---|---|---|---|
| T1 | shim opaque types + ShapesVec API | `mlx-sys/shim/include/cxx_mlx_shim/fast.h, src/fast.cc` | 0.5 天 |
| T2 | `metal_kernel_build` shim + bridge + sys_smoke | T1 文件 + `mlx-sys/src/bridge/fast.rs, tests/sys_smoke.rs` | 0.5 天 |
| T3 | `metal_kernel_dispatch` shim + bridge + sys_smoke | T2 文件扩展 | 1 天 |
| T4 | Rust safe API: `MetalKernel` + `MetalKernelBuilder` | `mlx/src/fast/mod.rs`, `mlx/src/fast/metal_kernel/mod.rs` | 0.5 天 |
| T5 | Rust safe API: `DispatchBuilder` typestate + 5 markers + setter 全套 + `TemplateArg` enum | `mlx/src/fast/metal_kernel/dispatch.rs` | 1 天 |
| T6 | trybuild typestate compile-fail tests (5 个) | `mlx/tests/trybuild/...` + `mlx/Cargo.toml` 加 trybuild dep | 0.5 天 |
| T7 | 集成测试 (`simple_add_kernel`, `multi_output_kernel`) | `mlx-sys/tests/p3a_metal_kernel.rs` | 0.5 天 |

**总计 ~4 天**。

T1 → T2 → T3 (cxx 层) → T4 → T5 → T6 → T7 串行依赖。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| `CustomKernelFunction` (`std::function`) 跨 cxx 持有 | C++ side 包装到 `MetalKernelInner` 持有 `std::function`，Rust 端通过 opaque type 持有 `unique_ptr`；不直接跨 cxx 暴露 std::function |
| `std::variant<int, bool, Dtype>` 跨 cxx | 展平为 `TemplateArgC { name, kind, int_val, bool_val }`，`kind` 区分类型（0=int, 1=bool, 2=dtype）|
| `std::vector<Shape>` 跨 cxx | `ShapesVec` opaque 双向桥（Rust 端 `shapes_vec_new` + `shapes_vec_push` 累积，再传 dispatch） |
| typestate API 错误消息差 | 每个 setter 加 doc comment 明确 marker 转换；`spec § 3.2` 字段顺序固定让用户读类型推断错误时能定位 |
| Metal 源码字符串编译失败 | `metal_kernel_build` 时 MLX 自身做 syntax check；失败转 cxx exception → Rust `Err(Error::Mlx(...))` actionable |
| dispatch 调用频次 hot path 性能 | M2 typestate 编译期检查 → runtime 路径仅 5 个 `Option::expect`（typestate 保证它们都是 Some，编译器优化掉） |
| 多 stream 并发 dispatch | `MetalKernel` 内部 `Arc<MetalKernelInner>`，read-only 共享；`CustomKernelFunction` 是 immutable 闭包，并发 dispatch 安全 |
| ShapesVec / TemplateArgC 是新 cxx struct | 沿用 P6 ArrayVec 模式（已稳定）；shapes_vec_new / push / count 是 ArrayVec 等价物 |
| trybuild test 增加 dev-dependency | 仅在 `mlx/Cargo.toml` `[dev-dependencies]` 加 `trybuild = "1"`，不影响 release build |

---

## 7. 与后续阶段的关系

- **P3b Qwen3.5 特殊算子**：直接使用 `MetalKernel` 实现 `gated_delta_step` kernel（~150 行 Metal 源码）；其他 Qwen3.5 子组件（gated full attn / MRoPE / RMSNormGated / MTP）走纯 ops 路径，不需要 metal_kernel
- **P5 Qwen3.5 MoE**：MoE 路由可能需要自定义 kernel；如需，复用 P3a 的 `MetalKernel` 接口
- **P7 benchmark**：metal_kernel binding 性能本身不是 P7 范围；但 gated_delta kernel 性能（vs ops fallback）会在 P7 benchmark 中显现
- **P8 HTTP server**：metal_kernel 是 stateless，多请求共享 MetalKernel 实例（`Arc::clone`）；不需要 P8 时改造
- **P9 Paged cache**：与 metal_kernel 独立，不交互

---
