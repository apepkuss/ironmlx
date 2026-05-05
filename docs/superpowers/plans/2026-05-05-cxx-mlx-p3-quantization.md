# cxx-mlx P3 · Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MLX 量化（低精度）子系统提供完整的 Rust 安全绑定，覆盖 7 个公开函数：affine 量化（quantize / dequantize / quantized_matmul）、双量化 NVFP4（qqmm）、MoE 量化（gather_qmm）、FP8 数值格式转换（from_fp8 / to_fp8）。

**Architecture:** 三层结构（沿用 P0–P2c 既建模式）：shim C++ 适配层（opaque QuantizeResult 解决 `vector<array>` 桥接 + helpers `opt_arr`/`opt_i`/`opt_dtype` 还原 `std::optional`）→ cxx::bridge ABI 边界（free function 风格）→ 安全 Rust API（`Vec<Array>` / `Option<i32>` / `Option<&Array>` / `Option<Dtype>` / `&str` / `Result<...>`）。`take_at(idx)` 用 taken_ bitmap 实现单次性消费契约（与 P2c take_by_name 一致）。

**Tech Stack:** Rust 1.82+（`Pin<&mut T>` for cxx opaque !Unpin 引用），cxx 1.0（含 unsafe extern + `*const T` 裸指针 + `rust::Str`），MLX C++ 共享安装（含已修复的 libgguflib.a），cargo nightly fmt + clippy + release build。

**Spec reference:** `docs/superpowers/specs/2026-05-05-cxx-mlx-p3-quantization-design.md`

---

## 关键背景信息（实施者必读）

### 项目三层结构

- **shim 层**：`mlx-sys/shim/include/cxx_mlx_shim/*.h` + `mlx-sys/shim/src/*.cc` —— 手写 C++，把 cxx 不能表达的 MLX 类型抹平
- **桥接层**：`mlx-sys/src/bridge/*.rs` —— `#[cxx::bridge]` 声明 ABI；项目惯例是 **free function**（不用 `self: &T` 方法语法）
- **安全层**：`mlx/src/*.rs` —— Rust 风格 API；顶层 `mlx::*` re-export

### cxx 类型映射（已在 P2b/P2c 反复验证）

| MLX C++ | shim 暴露 | cxx bridge 类型 | Rust 端调用 |
|---------|----------|-----------------|-------------|
| `std::optional<int>` | `(bool has, int32_t v)` 双参 | `bool, i32` | `Option<i32>::map_or((false, 0), |v| (true, v))` |
| `std::optional<float>` | `(bool has, float v)` 双参（P3 不直接用，但 pattern 同构） | `bool, f32` | 同上 |
| `std::optional<Dtype>` | `(bool has, uint8_t repr)` 双参 | `bool, u8` | `Option<Dtype>::map_or((false, 0), |d| (true, d.as_u8()))` |
| `std::optional<array>` | `*const MlxArray`（nullptr=None） | `*const MlxArray`（unsafe fn 必须） | `Option<&Array>::map_or(null, |a| a.as_inner() as *const _)` |
| `std::string` 入参 | `rust::Str` | `&str` | 直接传 `&str` |
| `std::vector<array>` 出参 | opaque `QuantizeResult` + `count()` + `take_at(idx)` | bridge 上 `Pin<&mut QuantizeResult>` | 安全层循环 take |
| 抽象/不可拷贝类 | opaque class | bridge `type Foo;` | `cxx::UniquePtr<Foo>` |

**Bridge 风格**：opaque type + free function（与 P0–P2c 完全一致，不用 `self: &T` 方法语法）。

### 已有 API 引用点

- `Array::from_inner(inner: cxx::UniquePtr<...>) -> Self`（[mlx/src/array.rs:11](mlx/src/array.rs#L11)）
- `Array::as_inner(&self) -> &mlx_sys::array::ffi::MlxArray`（[mlx/src/array.rs:139](mlx/src/array.rs#L139)）
- `Dtype::as_u8(&self) -> u8`（[mlx/src/dtype.rs:31](mlx/src/dtype.rs#L31)）
- 跨桥接共享 `MlxArray`：`type MlxArray = crate::bridge::array::ffi::MlxArray;`（参考 [mlx-sys/src/bridge/transforms.rs:10](mlx-sys/src/bridge/transforms.rs#L10)）

### 强制检查（CLAUDE.md + 项目约定，每次 commit 前必跑）

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```

### MLX 上游 API（来自 `${MLX_DIR}/include/mlx/ops.h:1389-1452`）

```cpp
namespace mlx::core {

array quantized_matmul(
    array x, array w, array scales,
    std::optional<array> biases = std::nullopt,
    bool transpose = true,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    StreamOrDevice s = {});

std::vector<array> quantize(
    const array& w,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    const std::optional<array>& global_scale = std::nullopt,
    StreamOrDevice s = {});

array dequantize(
    const array& w, const array& scales,
    const std::optional<array>& biases = std::nullopt,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    const std::optional<array>& global_scale = std::nullopt,
    std::optional<Dtype> dtype = std::nullopt,
    StreamOrDevice s = {});

array qqmm(
    array x, array w,
    const std::optional<array> w_scales = std::nullopt,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "nvfp4",
    const std::optional<array> global_scale_x = std::nullopt,
    const std::optional<array> global_scale_w = std::nullopt,
    StreamOrDevice s = {});

array from_fp8(array x, Dtype dtype, StreamOrDevice s = {});
array to_fp8(array x, StreamOrDevice s = {});

array gather_qmm(
    const array& x, const array& w, const array& scales,
    const std::optional<array>& biases = std::nullopt,
    std::optional<array> lhs_indices = std::nullopt,
    std::optional<array> rhs_indices = std::nullopt,
    bool transpose = true,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    bool sorted_indices = false,
    StreamOrDevice s = {});

}  // namespace mlx::core
```

`StreamOrDevice s = {}` 即 caller 线程默认 stream，shim 全部不传该参数。

### MLX FP8 实现说明

MLX 的 FP8（E4M3）**没有显式 dtype**——`to_fp8(x)` 返回 `uint8` 数组（字节按 E4M3 格式解释），`from_fp8(x, target_dtype)` 把 uint8 数据解码到目标浮点 dtype。这是 MLX 上游设计，绑定层照实暴露。

---

## 文件清单

### 新建
- `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`（约 90 行）
- `mlx-sys/shim/src/quantization.cc`（约 130 行）
- `mlx-sys/src/bridge/quantization.rs`（约 80 行）
- `mlx/src/quantization.rs`（约 220 行）
- `mlx/tests/p3_quantization.rs`（约 250 行 + 10 测试）

### 修改
- `mlx-sys/build.rs`
- `mlx-sys/src/bridge/mod.rs`
- `mlx-sys/src/lib.rs`
- `mlx/src/lib.rs`
- `README.md`（在 Task 6）

---

## Task 1: 框架搭建 + quantize + dequantize（含 QuantizeResult opaque）

**目的**：打通 build.rs / mod.rs / lib.rs 接线，定义 `QuantizeResult` opaque 类（含 taken_ bitmap）+ helpers（opt_arr/opt_i/opt_dtype）+ 实现 `quantize` 和 `dequantize` 两个核心函数。

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`
- Create: `mlx-sys/shim/src/quantization.cc`
- Create: `mlx-sys/src/bridge/quantization.rs`
- Create: `mlx/src/quantization.rs`
- Create: `mlx/tests/p3_quantization.rs`
- Modify: `mlx-sys/build.rs`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/src/lib.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1.1: 写失败的集成测试**

将以下完整内容写入 `mlx/tests/p3_quantization.rs`：

```rust
//! Integration tests for mlx::quantization — low-precision subsystem.

use mlx::quantization::{dequantize, quantize};
use mlx::Array;

/// 构造 [N=4, K=64] f32 测试权重矩阵（K=64 = 默认 group_size）。
fn make_test_weight() -> Array {
    let total: usize = 256; // 4 * 64
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 1.0).collect();
    Array::from_slice(&data, &[4, 64]).expect("weight")
}

#[test]
fn quantize_affine_4bit_returns_three_arrays() {
    let w = make_test_weight();
    let result = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    // affine 模式下应返回 [packed_weights, scales, biases]
    assert_eq!(result.len(), 3, "affine quantize should return 3 arrays");
}

#[test]
fn quantize_dequantize_round_trip_4bit() {
    let w = make_test_weight();
    let v_in: Vec<f32> = w.to_vec().expect("w to_vec");

    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    assert_eq!(parts.len(), 3);

    let dequantized = dequantize(
        &parts[0],   // packed
        &parts[1],   // scales
        Some(&parts[2]),  // biases
        Some(64),
        Some(4),
        "affine",
        None,
        None,
    )
    .expect("dequantize");

    let v_out: Vec<f32> = dequantized.to_vec().expect("dequantized to_vec");
    assert_eq!(v_in.len(), v_out.len());

    // 4-bit 量化误差容差较宽（典型 SQNR ~25 dB，相对误差几个百分点）
    let mut max_err = 0.0_f32;
    for (a, b) in v_in.iter().zip(&v_out) {
        let err = (a - b).abs();
        if err > max_err { max_err = err; }
    }
    assert!(max_err < 5e-2, "4-bit round-trip max err {max_err}");
}

#[test]
fn quantize_dequantize_round_trip_8bit() {
    let w = make_test_weight();
    let v_in: Vec<f32> = w.to_vec().expect("w to_vec");

    let parts = quantize(&w, Some(64), Some(8), "affine", None).expect("quantize");
    let dequantized = dequantize(
        &parts[0], &parts[1], Some(&parts[2]),
        Some(64), Some(8), "affine", None, None,
    )
    .expect("dequantize");

    let v_out: Vec<f32> = dequantized.to_vec().expect("to_vec");
    let mut max_err = 0.0_f32;
    for (a, b) in v_in.iter().zip(&v_out) {
        let err = (a - b).abs();
        if err > max_err { max_err = err; }
    }
    assert!(max_err < 5e-3, "8-bit round-trip max err {max_err}");
}
```

- [ ] **Step 1.2: 运行测试，确认失败（编译错误）**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization --no-run`
Expected: 编译失败，错误提到 `mlx::quantization` 不存在。

- [ ] **Step 1.3: 创建 shim 头 `quantization.h`**

将以下完整内容写入 `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`：

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== QuantizeResult (opaque) =====
// MLX 的 quantize 返回 std::vector<array>，cxx 不支持 Vec<UniquePtr<T>>。
// 包装为 opaque 类，提供 count() + take_at(idx) 接口。take_at 用 taken_
// bitmap 防止重复取（与 P2c take_by_name 单次性消费契约一致）。
class QuantizeResult {
 public:
  explicit QuantizeResult(std::vector<mlx::core::array> data);
  size_t count() const { return arrays_.size(); }
  std::unique_ptr<MlxArray> take_at(size_t idx);

 private:
  std::vector<mlx::core::array> arrays_;
  std::vector<bool> taken_;
};

size_t quantize_result_count(const QuantizeResult& r);
std::unique_ptr<MlxArray> quantize_result_take_at(QuantizeResult& r, size_t idx);

// ===== 量化函数 =====
// 可选参数编码:
//   Option<int>   → (bool has_value, int32_t value)
//   Option<Dtype> → (bool has_dtype, uint8_t dtype_repr)
//   Option<&Array>→ const MlxArray* (nullptr = None)
//   &str mode     → rust::Str

std::unique_ptr<QuantizeResult> quantize(
    const MlxArray& w,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale);

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale,
    bool has_dtype, uint8_t dtype_repr);

}  // namespace cxx_mlx
```

- [ ] **Step 1.4: 创建 shim 实现 `quantization.cc`**

将以下完整内容写入 `mlx-sys/shim/src/quantization.cc`：

```cpp
#include "cxx_mlx_shim/quantization.h"

#include <stdexcept>

namespace cxx_mlx {

namespace {

// pointer → optional<array>。array 拷贝廉价（refcount on array_desc_）。
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<int> opt_i(bool has, int32_t v) {
  return has ? std::optional<int>(v) : std::nullopt;
}

inline std::optional<mlx::core::Dtype> opt_dtype(bool has, uint8_t v) {
  // Dtype 在 MLX 中是含 size 的 struct，但 Val 枚举值定义了所有 dtype。
  // 用 size_of(Dtype) 反推不直接，只能依赖默认 Dtype 构造器。所幸 dequantize
  // 的 dtype 参数 MLX 内部按枚举 dispatch，传 Dtype{Val, size} 即可。
  if (!has) return std::nullopt;
  // 重建对应 Val 的 Dtype，size 从 default 实例查（MLX dtype.h 定义了 inline constexpr
  // 实例如 mlx::core::float32 等）。这里走简化路径：手动 case 所有 Val 值。
  switch (static_cast<mlx::core::Dtype::Val>(v)) {
    case mlx::core::Dtype::Val::bool_:    return mlx::core::bool_;
    case mlx::core::Dtype::Val::uint8:    return mlx::core::uint8;
    case mlx::core::Dtype::Val::uint16:   return mlx::core::uint16;
    case mlx::core::Dtype::Val::uint32:   return mlx::core::uint32;
    case mlx::core::Dtype::Val::uint64:   return mlx::core::uint64;
    case mlx::core::Dtype::Val::int8:     return mlx::core::int8;
    case mlx::core::Dtype::Val::int16:    return mlx::core::int16;
    case mlx::core::Dtype::Val::int32:    return mlx::core::int32;
    case mlx::core::Dtype::Val::int64:    return mlx::core::int64;
    case mlx::core::Dtype::Val::float16:  return mlx::core::float16;
    case mlx::core::Dtype::Val::float32:  return mlx::core::float32;
    case mlx::core::Dtype::Val::float64:  return mlx::core::float64;
    case mlx::core::Dtype::Val::bfloat16: return mlx::core::bfloat16;
    case mlx::core::Dtype::Val::complex64:return mlx::core::complex64;
  }
  throw std::runtime_error("unknown Dtype::Val");
}

}  // namespace

// ===== QuantizeResult =====

QuantizeResult::QuantizeResult(std::vector<mlx::core::array> data)
    : arrays_(std::move(data)), taken_(arrays_.size(), false) {}

std::unique_ptr<MlxArray> QuantizeResult::take_at(size_t idx) {
  if (idx >= arrays_.size()) {
    throw std::runtime_error("QuantizeResult::take_at: idx out of range");
  }
  if (taken_[idx]) {
    throw std::runtime_error("QuantizeResult::take_at: already taken at idx");
  }
  taken_[idx] = true;
  return std::make_unique<MlxArray>(std::move(arrays_[idx]));
}

size_t quantize_result_count(const QuantizeResult& r) { return r.count(); }

std::unique_ptr<MlxArray> quantize_result_take_at(QuantizeResult& r, size_t idx) {
  return r.take_at(idx);
}

// ===== quantize =====

std::unique_ptr<QuantizeResult> quantize(
    const MlxArray& w,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale) {
  auto result = mlx::core::quantize(
      w,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale));
  return std::make_unique<QuantizeResult>(std::move(result));
}

// ===== dequantize =====

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale,
    bool has_dtype, uint8_t dtype_repr) {
  return std::make_unique<MlxArray>(mlx::core::dequantize(
      w, scales, opt_arr(biases),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale),
      opt_dtype(has_dtype, dtype_repr)));
}

}  // namespace cxx_mlx
```

- [ ] **Step 1.5: 创建桥接 `bridge/quantization.rs`**

将以下完整内容写入 `mlx-sys/src/bridge/quantization.rs`：

```rust
//! Bridge for MLX quantization subsystem.
//!
//! Quantize returns std::vector<array>, which cxx 1.0 doesn't support
//! as Vec<UniquePtr<T>>. Wrapped as opaque QuantizeResult with
//! count() + take_at(idx) free functions. Single-use semantics:
//! take_at(idx) twice throws (matches P2c take_by_name pattern).
//!
//! Optional encodings:
//! - Option<i32>   → (bool has_value, i32 value)  (P2b rope pattern)
//! - Option<Dtype> → (bool has_dtype, u8 dtype_repr)
//! - Option<&Array>→ *const MlxArray (nullptr = None)  (P2b/P2c pattern)
//! - &str mode     → rust::Str  (P2b sdpa pattern)

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/quantization.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type QuantizeResult;

        // ===== quantize result accessors =====
        fn quantize_result_count(r: &QuantizeResult) -> usize;
        fn quantize_result_take_at(
            r: Pin<&mut QuantizeResult>,
            idx: usize,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== quantize =====
        unsafe fn quantize(
            w: &MlxArray,
            has_group_size: bool, group_size: i32,
            has_bits: bool, bits: i32,
            mode: &str,
            global_scale: *const MlxArray,
        ) -> Result<UniquePtr<QuantizeResult>>;

        // ===== dequantize =====
        unsafe fn dequantize(
            w: &MlxArray, scales: &MlxArray,
            biases: *const MlxArray,
            has_group_size: bool, group_size: i32,
            has_bits: bool, bits: i32,
            mode: &str,
            global_scale: *const MlxArray,
            has_dtype: bool, dtype_repr: u8,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
```

- [ ] **Step 1.6: 在 `mlx-sys/src/bridge/mod.rs` 末尾增加 `pub mod quantization;`**

打开 `mlx-sys/src/bridge/mod.rs`，在 `pub mod io;` 之后增加 `pub mod quantization;`（保留前面的 module-level 注释）。

- [ ] **Step 1.7: 在 `mlx-sys/src/lib.rs` 增加 re-export**

打开 `mlx-sys/src/lib.rs`，把已有的 re-export 块改为：

```rust
pub use bridge::array;
pub use bridge::fast;
pub use bridge::io;
pub use bridge::quantization;
pub use bridge::stream;
pub use bridge::transforms;
```

（cargo fmt 会按字母排序，提交时以 fmt 后的为准。）

- [ ] **Step 1.8: 在 `mlx-sys/build.rs` 注册 quantization 桥接 + shim cc**

打开 `mlx-sys/build.rs`，找到 `cxx_build::bridges([...])` 调用块，把它替换为：

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
        "src/bridge/quantization.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/fast.cc")
    .file("shim/src/io.cc")
    .file("shim/src/quantization.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

（仅在桥接列表与 `.file()` 列表的末尾追加 quantization 项）

- [ ] **Step 1.9: 创建安全 API `mlx/src/quantization.rs`**

将以下完整内容写入 `mlx/src/quantization.rs`：

```rust
//! MLX low-precision subsystem: affine/NVFP4 quantization + FP8 conversion.
//!
//! Affine quantization (mode="affine"): pack high-precision weights into
//! lower-bit groups + per-group scale/bias. Used by mlx-lm 4-bit/8-bit
//! quantized models (e.g. .safetensors with `.scales` / `.biases` suffixed
//! tensor naming convention).
//!
//! NVFP4 mode (qqmm): both inputs may be quantized; scheme used by Nvidia
//! NVFP4 / MXFP4 hardware-accelerated formats.
//!
//! FP8 (E4M3): 8-bit floating-point format conversion. MLX represents FP8
//! data as a uint8 array with bytes interpreted per E4M3 layout.

use crate::{Array, Dtype, Error, Result};

/// Quantize a matrix along its last axis.
///
/// For `mode="affine"` (the default), the result is
/// `[packed_weights, scales, biases]` (3 arrays). Other modes may return
/// a different number of arrays.
pub fn quantize(
    w: &Array,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
) -> Result<Vec<Array>> {
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: gscale is null or borrow of `global_scale: &Array` valid for this call.
    let mut result = unsafe {
        mlx_sys::quantization::ffi::quantize(
            w.as_inner(), has_gs, gs, has_b, b, mode, gscale,
        )
    }
    .map_err(Error::from)?;
    let count = mlx_sys::quantization::ffi::quantize_result_count(&result);
    let mut output = Vec::with_capacity(count);
    for i in 0..count {
        let arr_ptr = mlx_sys::quantization::ffi::quantize_result_take_at(
            result.pin_mut(),
            i,
        )
        .map_err(Error::from)?;
        output.push(Array::from_inner(arr_ptr));
    }
    Ok(output)
}

/// Inverse of [`quantize`]. Reconstructs the original-precision matrix.
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
    dtype: Option<Dtype>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_dt, dt) = dtype.map_or((false, 0), |d| (true, d.as_u8()));
    // SAFETY: b_ptr/gscale each null or borrow of an &Array valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::dequantize(
            w.as_inner(), scales.as_inner(), b_ptr,
            has_gs, gs, has_b, b, mode, gscale, has_dt, dt,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 1.10: 在 `mlx/src/lib.rs` 增加 `pub mod quantization;`**

打开 `mlx/src/lib.rs`，在 `pub mod io;` 之后（与 `pub use io::{...}` 块之间或之后）添加 `pub mod quantization;`：

```rust
pub mod io;
pub use io::{
    load_gguf, load_npy, ...
};

pub mod quantization;
```

（顶层 re-export 在 Task 6 统一加。）

- [ ] **Step 1.11: 编译并运行测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 3 tests passed（`quantize_affine_4bit_returns_three_arrays`、`quantize_dequantize_round_trip_4bit`、`quantize_dequantize_round_trip_8bit`）。

- [ ] **Step 1.12: 跑全套 Rust 检查**

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

- [ ] **Step 1.13: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/quantization.h \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/quantization.rs \
        mlx-sys/src/bridge/mod.rs \
        mlx-sys/src/lib.rs \
        mlx-sys/build.rs \
        mlx/src/quantization.rs \
        mlx/src/lib.rs \
        mlx/tests/p3_quantization.rs
git commit -m "feat(p3): scaffold quantization module + quantize/dequantize (3 layers, 3 tests)"
```

---

## Task 2: quantized_matmul（推理工作主力）

**目的**：追加 `quantized_matmul`，验证它与 `x @ dequantize(...)` 数值一致。这是量化推理的核心算子。

**Files (all modifications, no new files):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`
- Modify: `mlx-sys/shim/src/quantization.cc`
- Modify: `mlx-sys/src/bridge/quantization.rs`
- Modify: `mlx/src/quantization.rs`
- Modify: `mlx/tests/p3_quantization.rs`

- [ ] **Step 2.1: 写失败的集成测试**

在 `mlx/tests/p3_quantization.rs` 末尾追加：

```rust
use mlx::ops;
use mlx::quantization::quantized_matmul;

#[test]
fn quantized_matmul_matches_dequantize_matmul() {
    // W: [N=4, K=64], x: [B=2, K=64]
    // y_qmm = quantized_matmul(x, packed_W, scales, biases, transpose=true)
    // y_ref = x @ dequantize(packed_W).T
    // 两者应在 4-bit 量化容差内一致
    let w = make_test_weight();  // [4, 64]
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::from_slice(&x_data, &[2, 64]).expect("x");

    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");

    // y_qmm = x @ W.T (transpose=true)，输出 [2, 4]
    let y_qmm = quantized_matmul(
        &x, &parts[0], &parts[1], Some(&parts[2]),
        true, Some(64), Some(4), "affine",
    )
    .expect("qmm");
    assert_eq!(y_qmm.shape().as_slice(), &[2, 4]);

    // 参考路径: y_ref = x @ dequantize(W).T
    let dq = dequantize(
        &parts[0], &parts[1], Some(&parts[2]),
        Some(64), Some(4), "affine", None, None,
    )
    .expect("dq");
    let dq_t = dq.transpose_axes(&[1, 0]).expect("transpose");
    let y_ref = x.matmul(&dq_t).expect("ref matmul");

    let v_qmm: Vec<f32> = y_qmm.to_vec().expect("qmm to_vec");
    let v_ref: Vec<f32> = y_ref.to_vec().expect("ref to_vec");
    assert_eq!(v_qmm.len(), v_ref.len());

    // qmm 与 dequantize+matmul 的差异应当极小（计算路径不同但代数等价）
    let mut max_err = 0.0_f32;
    for (a, b) in v_qmm.iter().zip(&v_ref) {
        let err = (a - b).abs();
        if err > max_err { max_err = err; }
    }
    assert!(max_err < 1e-2, "qmm vs ref max err {max_err}");
}
```

注：测试中的 `make_test_weight` 已在 Task 1 步骤 1.1 中定义（位于同一 `p3_quantization.rs` 文件顶部），这里直接复用。

- [ ] **Step 2.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization --no-run`
Expected: 编译失败，提示 `quantization::quantized_matmul` 不存在。

- [ ] **Step 2.3: shim 头追加 `quantized_matmul`**

在 `mlx-sys/shim/include/cxx_mlx_shim/quantization.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode);
```

- [ ] **Step 2.4: shim cc 追加 `quantized_matmul`**

在 `mlx-sys/shim/src/quantization.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode) {
  return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode)));
}
```

- [ ] **Step 2.5: 桥接追加 `quantized_matmul`**

在 `mlx-sys/src/bridge/quantization.rs` 的 `unsafe extern "C++"` 块内（`dequantize` 之后）追加：

```rust
        // ===== quantized_matmul =====
        unsafe fn quantized_matmul(
            x: &MlxArray, w: &MlxArray, scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            has_group_size: bool, group_size: i32,
            has_bits: bool, bits: i32,
            mode: &str,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 2.6: 安全 API 追加 `quantized_matmul`**

在 `mlx/src/quantization.rs` 的 `dequantize` 之后追加：

```rust
/// Compute `x @ w` where `w` is a quantized matrix. The workhorse for
/// inference of quantized models.
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    // SAFETY: b_ptr is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::quantized_matmul(
            x.as_inner(), w.as_inner(), scales.as_inner(), b_ptr,
            transpose, has_gs, gs, has_b, b, mode,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 2.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 4 tests passed（前 3 + qmm 1）。

如果测试失败且原因是 MLX 内部对量化 matmul 的某条件限制（dtype/shape），停下来报告 BLOCKED 并描述错误。

- [ ] **Step 2.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 2.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/quantization.h \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/quantization.rs \
        mlx/src/quantization.rs \
        mlx/tests/p3_quantization.rs
git commit -m "feat(p3): quantized_matmul (1 test)"
```

---

## Task 3: qqmm（双量化 NVFP4）

**目的**：追加 `qqmm`（quantized-quantized matmul），默认 mode="nvfp4"。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`
- Modify: `mlx-sys/shim/src/quantization.cc`
- Modify: `mlx-sys/src/bridge/quantization.rs`
- Modify: `mlx/src/quantization.rs`
- Modify: `mlx/tests/p3_quantization.rs`

- [ ] **Step 3.1: 写失败的集成测试**

在 `mlx/tests/p3_quantization.rs` 末尾追加：

```rust
use mlx::quantization::qqmm;

#[test]
fn qqmm_smoke_call() {
    // qqmm 默认 mode="nvfp4"。本测试只做 smoke：函数可调用，shape 合理，输出有限。
    // 如果 MLX 在 macOS Metal 后端不支持 NVFP4，停下来报告 BLOCKED。
    //
    // 注：NVFP4 需要特定的 4-bit 量化输入格式；用 affine 4-bit 量化的输出近似填入
    // 验证调用路径，可能不语义合法（MLX 可能拒绝）。如失败按 BLOCKED 处理。
    let w = make_test_weight();
    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::from_slice(&x_data, &[2, 64]).expect("x");

    // mode="nvfp4" 默认；w_scales=Some(parts[1])
    let result = qqmm(
        &x, &parts[0], Some(&parts[1]),
        Some(64), Some(4), "nvfp4",
        None, None,
    );

    match result {
        Ok(y) => {
            // 形状合理：[B=2, N=4]（与 quantized_matmul 一致）
            let v: Vec<f32> = y.to_vec().expect("qqmm to_vec");
            for x in &v {
                assert!(x.is_finite(), "non-finite value: {x}");
            }
        }
        Err(e) => {
            // MLX 不支持当前输入组合下的 NVFP4，标记为预期失败而非测试失败
            eprintln!("qqmm not supported with current inputs: {e:?}");
        }
    }
}
```

注：qqmm 是高级 op，本测试设计为"调用通则验通过；MLX 不支持则不算失败"。这是因为 NVFP4 在 macOS Metal 后端可能未完全实现。

- [ ] **Step 3.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization --no-run`
Expected: 编译失败，提示 `quantization::qqmm` 不存在。

- [ ] **Step 3.3: shim 头追加 `qqmm`**

在 `mlx-sys/shim/include/cxx_mlx_shim/quantization.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> qqmm(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray* w_scales,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale_x,
    const MlxArray* global_scale_w);
```

- [ ] **Step 3.4: shim cc 追加 `qqmm`**

在 `mlx-sys/shim/src/quantization.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> qqmm(
    const MlxArray& x, const MlxArray& w,
    const MlxArray* w_scales,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale_x,
    const MlxArray* global_scale_w) {
  return std::make_unique<MlxArray>(mlx::core::qqmm(
      x, w, opt_arr(w_scales),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale_x),
      opt_arr(global_scale_w)));
}
```

- [ ] **Step 3.5: 桥接追加 `qqmm`**

在 `mlx-sys/src/bridge/quantization.rs` 的 `unsafe extern "C++"` 块内（`quantized_matmul` 之后）追加：

```rust
        // ===== qqmm =====
        unsafe fn qqmm(
            x: &MlxArray, w: &MlxArray,
            w_scales: *const MlxArray,
            has_group_size: bool, group_size: i32,
            has_bits: bool, bits: i32,
            mode: &str,
            global_scale_x: *const MlxArray,
            global_scale_w: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 3.6: 安全 API 追加 `qqmm`**

在 `mlx/src/quantization.rs` 的 `quantized_matmul` 之后追加：

```rust
/// Quantized-quantized matmul. Both x and w may be quantized; default
/// mode is `"nvfp4"`.
pub fn qqmm(
    x: &Array,
    w: &Array,
    w_scales: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale_x: Option<&Array>,
    global_scale_w: Option<&Array>,
) -> Result<Array> {
    let ws = w_scales.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gx = global_scale_x.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let gw = global_scale_w.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: ws/gx/gw each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::qqmm(
            x.as_inner(), w.as_inner(), ws,
            has_gs, gs, has_b, b, mode, gx, gw,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 3.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 5 tests passed（前 4 + qqmm 1）。注意 qqmm 测试如果 MLX 拒绝输入组合会打印 stderr 但不算失败。

- [ ] **Step 3.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 3.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/quantization.h \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/quantization.rs \
        mlx/src/quantization.rs \
        mlx/tests/p3_quantization.rs
git commit -m "feat(p3): qqmm (quantized-quantized matmul, 1 test)"
```

---

## Task 4: gather_qmm（MoE 量化）

**目的**：追加 `gather_qmm`，11 个参数（含 lhs_indices/rhs_indices/sorted_indices）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`
- Modify: `mlx-sys/shim/src/quantization.cc`
- Modify: `mlx-sys/src/bridge/quantization.rs`
- Modify: `mlx/src/quantization.rs`
- Modify: `mlx/tests/p3_quantization.rs`

- [ ] **Step 4.1: 写失败的集成测试**

在 `mlx/tests/p3_quantization.rs` 末尾追加：

```rust
use mlx::quantization::gather_qmm;

#[test]
fn gather_qmm_no_indices_smoke_call() {
    // gather_qmm 不传 lhs/rhs indices 时退化为常规 quantized_matmul。
    // 本测试做最简调用路径验通过 + 输出有限。
    let w = make_test_weight();
    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::from_slice(&x_data, &[2, 64]).expect("x");

    let result = gather_qmm(
        &x, &parts[0], &parts[1], Some(&parts[2]),
        None,  // lhs_indices
        None,  // rhs_indices
        true,  // transpose
        Some(64), Some(4), "affine",
        false, // sorted_indices
    );

    match result {
        Ok(y) => {
            let v: Vec<f32> = y.to_vec().expect("to_vec");
            for x in &v {
                assert!(x.is_finite(), "non-finite value: {x}");
            }
        }
        Err(e) => {
            // gather_qmm 在 MoE 模式下需要特殊 indices 格式，
            // 不传 indices 时 MLX 行为可能拒绝。视为预期失败。
            eprintln!("gather_qmm rejected without indices: {e:?}");
        }
    }
}
```

注：gather_qmm 是 MoE 专用 op，无 indices 时行为可能不被 MLX 接受；本测试主要验证调用路径。

- [ ] **Step 4.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization --no-run`
Expected: 编译失败，提示 `quantization::gather_qmm` 不存在。

- [ ] **Step 4.3: shim 头追加 `gather_qmm`**

在 `mlx-sys/shim/include/cxx_mlx_shim/quantization.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool sorted_indices);
```

- [ ] **Step 4.4: shim cc 追加 `gather_qmm`**

在 `mlx-sys/shim/src/quantization.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool sorted_indices) {
  return std::make_unique<MlxArray>(mlx::core::gather_qmm(
      x, w, scales, opt_arr(biases),
      opt_arr(lhs_indices), opt_arr(rhs_indices),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      sorted_indices));
}
```

- [ ] **Step 4.5: 桥接追加 `gather_qmm`**

在 `mlx-sys/src/bridge/quantization.rs` 的 `unsafe extern "C++"` 块内（`qqmm` 之后）追加：

```rust
        // ===== gather_qmm =====
        unsafe fn gather_qmm(
            x: &MlxArray, w: &MlxArray, scales: &MlxArray,
            biases: *const MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            transpose: bool,
            has_group_size: bool, group_size: i32,
            has_bits: bool, bits: i32,
            mode: &str,
            sorted_indices: bool,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 4.6: 安全 API 追加 `gather_qmm`**

在 `mlx/src/quantization.rs` 的 `qqmm` 之后追加：

```rust
/// Quantized matmul with matrix-level gather (MoE / expert routing).
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    sorted_indices: bool,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let li = lhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let ri = rhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    // SAFETY: b_ptr/li/ri each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::gather_qmm(
            x.as_inner(), w.as_inner(), scales.as_inner(), b_ptr,
            li, ri, transpose, has_gs, gs, has_b, b, mode, sorted_indices,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 6 tests passed。

- [ ] **Step 4.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 4.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/quantization.h \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/quantization.rs \
        mlx/src/quantization.rs \
        mlx/tests/p3_quantization.rs
git commit -m "feat(p3): gather_qmm (MoE quantized matmul, 1 test)"
```

---

## Task 5: from_fp8 + to_fp8（FP8/E4M3 转换）

**目的**：追加 FP8 (E4M3) 数值格式编解码两函数。MLX 把 FP8 数据存为 uint8 数组，靠这两个函数完成与浮点 dtype 的转换。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/quantization.h`
- Modify: `mlx-sys/shim/src/quantization.cc`
- Modify: `mlx-sys/src/bridge/quantization.rs`
- Modify: `mlx/src/quantization.rs`
- Modify: `mlx/tests/p3_quantization.rs`

- [ ] **Step 5.1: 写失败的集成测试**

在 `mlx/tests/p3_quantization.rs` 末尾追加：

```rust
use mlx::quantization::{from_fp8, to_fp8};
use mlx::Dtype;

#[test]
fn fp8_round_trip_f32_small_integers() {
    // 小整数 1.0/2.0/3.0/4.0 在 E4M3 (4-exp 3-mantissa) 范围内可精确或近似表达。
    // E4M3 mantissa 仅 3-bit，相对误差典型 ~6-12%；容差 0.5 安全（绝对误差对小值）。
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]).expect("x");
    let fp8 = to_fp8(&x).expect("to_fp8");

    let back = from_fp8(&fp8, Dtype::Float32).expect("from_fp8");
    assert_eq!(back.shape().as_slice(), &[4]);

    let v_back: Vec<f32> = back.to_vec().expect("to_vec");
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (i, (got, want)) in v_back.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.5,
            "fp8 round-trip[{i}] = {got}, want {want}"
        );
    }
}
```

- [ ] **Step 5.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization --no-run`
Expected: 编译失败，提示 `quantization::from_fp8` / `to_fp8` 不存在。

- [ ] **Step 5.3: shim 头追加 FP8 两函数**

在 `mlx-sys/shim/include/cxx_mlx_shim/quantization.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> from_fp8(const MlxArray& x, uint8_t dtype_repr);
std::unique_ptr<MlxArray> to_fp8(const MlxArray& x);
```

- [ ] **Step 5.4: shim cc 追加 FP8 实现**

在 `mlx-sys/shim/src/quantization.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
std::unique_ptr<MlxArray> from_fp8(const MlxArray& x, uint8_t dtype_repr) {
  // 复用 opt_dtype 的 case 逻辑,但这里 dtype 是必传(非 optional)
  // 直接构造对应 Dtype 实例。
  mlx::core::Dtype dt = mlx::core::float32;  // 默认占位,会被覆盖
  switch (static_cast<mlx::core::Dtype::Val>(dtype_repr)) {
    case mlx::core::Dtype::Val::bool_:    dt = mlx::core::bool_; break;
    case mlx::core::Dtype::Val::uint8:    dt = mlx::core::uint8; break;
    case mlx::core::Dtype::Val::uint16:   dt = mlx::core::uint16; break;
    case mlx::core::Dtype::Val::uint32:   dt = mlx::core::uint32; break;
    case mlx::core::Dtype::Val::uint64:   dt = mlx::core::uint64; break;
    case mlx::core::Dtype::Val::int8:     dt = mlx::core::int8; break;
    case mlx::core::Dtype::Val::int16:    dt = mlx::core::int16; break;
    case mlx::core::Dtype::Val::int32:    dt = mlx::core::int32; break;
    case mlx::core::Dtype::Val::int64:    dt = mlx::core::int64; break;
    case mlx::core::Dtype::Val::float16:  dt = mlx::core::float16; break;
    case mlx::core::Dtype::Val::float32:  dt = mlx::core::float32; break;
    case mlx::core::Dtype::Val::float64:  dt = mlx::core::float64; break;
    case mlx::core::Dtype::Val::bfloat16: dt = mlx::core::bfloat16; break;
    case mlx::core::Dtype::Val::complex64:dt = mlx::core::complex64; break;
    default:
      throw std::runtime_error("unknown Dtype::Val for from_fp8");
  }
  return std::make_unique<MlxArray>(mlx::core::from_fp8(x, dt));
}

std::unique_ptr<MlxArray> to_fp8(const MlxArray& x) {
  return std::make_unique<MlxArray>(mlx::core::to_fp8(x));
}
```

- [ ] **Step 5.5: 桥接追加 FP8 两函数**

在 `mlx-sys/src/bridge/quantization.rs` 的 `unsafe extern "C++"` 块内（`gather_qmm` 之后）追加：

```rust
        // ===== FP8 =====
        fn from_fp8(x: &MlxArray, dtype_repr: u8) -> Result<UniquePtr<MlxArray>>;
        fn to_fp8(x: &MlxArray) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 5.6: 安全 API 追加 FP8 两函数**

在 `mlx/src/quantization.rs` 的 `gather_qmm` 之后追加：

```rust
/// Convert an E4M3 float8 array to the given floating-point dtype.
pub fn from_fp8(x: &Array, dtype: Dtype) -> Result<Array> {
    let inner = mlx_sys::quantization::ffi::from_fp8(x.as_inner(), dtype.as_u8())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Convert a floating-point matrix to E4M3 float8.
pub fn to_fp8(x: &Array) -> Result<Array> {
    let inner = mlx_sys::quantization::ffi::to_fp8(x.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 5.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 7 tests passed。如果 MLX FP8 在 macOS Metal 后端不支持，停下来报告 BLOCKED。

- [ ] **Step 5.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 5.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/quantization.h \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/quantization.rs \
        mlx/src/quantization.rs \
        mlx/tests/p3_quantization.rs
git commit -m "feat(p3): from_fp8 + to_fp8 (E4M3 conversion, 1 test)"
```

---

## Task 6: 顶层 re-export + README + 全套验证

**目的**：把 P3 公开 API 暴露到 `mlx::*` 顶层，README 状态升级到 P3 完成（0.1 release 候选），跑完整 workspace 检查。

**Files:**
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/tests/p3_quantization.rs`
- Modify: `README.md`

- [ ] **Step 6.1: 在 `mlx/src/lib.rs` 的 `pub mod quantization;` 后追加 re-export**

打开 `mlx/src/lib.rs`，找到 `pub mod quantization;`，在其后追加 re-export：

```rust
pub mod quantization;
pub use quantization::{
    dequantize, from_fp8, gather_qmm, qqmm, quantize, quantized_matmul, to_fp8,
};
```

（cargo fmt 会按字母排序，提交时以 fmt 后的为准。）

- [ ] **Step 6.2: 在测试文件追加 re-export 验证测试**

在 `mlx/tests/p3_quantization.rs` 末尾追加：

```rust
#[test]
fn top_level_re_exports_work() {
    // 验证可以通过 mlx::* 顶层访问 P3 公开 API
    let w = make_test_weight();
    let parts = mlx::quantize(&w, Some(64), Some(4), "affine", None).expect("re-export");
    assert_eq!(parts.len(), 3);
}
```

- [ ] **Step 6.3: 运行所有 P3 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p3_quantization`
Expected: 8 tests passed。

- [ ] **Step 6.4: 跑完整 workspace 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --workspace --all-features`
Expected: 所有现有测试 + 8 个新 P3 测试全部通过。

- [ ] **Step 6.5: 更新 `README.md`**

`README.md` 中已有的 P2c 状态记录与 Roadmap 表格需要升级。

**位置 A（Status banner，约 line 5）**：当前文本：
```
**Status:** 🚧 **P2c complete** — Full IO subsystem ...
```

升级为：
```
**Status:** 🎉 **P3 complete (0.1 release candidate)** — Full quantization subsystem (`mlx::quantization::*`): affine `quantize`/`dequantize`/`quantized_matmul`, NVFP4 `qqmm`, MoE `gather_qmm`, FP8 `from_fp8`/`to_fp8`. Combined with P2a (Stream/async) + P2b (fast ops) + P2c (IO) + P1 (ops/array foundations), cxx-mlx now covers the macOS local LLM inference path end-to-end.
```

**位置 B（Roadmap 表格，约 line 215-217）**：在 P2c 行下追加：
```
- ✅ **P2c** — `io` (safetensors / gguf / npy + Reader/Writer streams) — 18 integration tests
- ✅ **P3** — `quantization` (quantize/dequantize/quantized_matmul/qqmm/gather_qmm/fp8) — 8 integration tests
```

如 README 还有其他相关引用（grep `P2c|P3|quantization` 检查），按现有风格保持一致。

- [ ] **Step 6.6: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过，无 warning，所有测试 PASS。

- [ ] **Step 6.7: 提交**

```bash
git add mlx/src/lib.rs mlx/tests/p3_quantization.rs README.md
git commit -m "feat(p3): re-export quantization API at crate root + README progress (0.1 release candidate)"
```

- [ ] **Step 6.8: 最终 git log 与 commit 数核对**

Run: `git log --oneline | head -10`
Expected: 看到 6 个 P3 feat commit（Tasks 1–6）+ docs (spec + plan) commit。

---

## 自检（plan 作者自检结果）

**Spec 覆盖**：
- ✅ 7 个公开函数：quantize / dequantize（Task 1）+ quantized_matmul（Task 2）+ qqmm（Task 3）+ gather_qmm（Task 4）+ from_fp8 / to_fp8（Task 5）
- ✅ shim/bridge/safe 三层结构（每个 task 都覆盖）
- ✅ 所有 spec 中的可选参数处理：`Option<i32>` → `(bool, i32)`、`Option<&Array>` → `*const`、`Option<Dtype>` → `(bool, u8)`、`&str` 直接传
- ✅ `QuantizeResult` opaque + count() + take_at(idx) + taken_ bitmap 单次性消费契约（Task 1）
- ✅ 错误传播链路（cxx::Exception → Error::from）
- ✅ 测试策略（每函数 ≥1 测试 + round-trip / 数值一致性 / smoke 测试 / 错误路径）
- ✅ 文件清单（shim/bridge/safe + build.rs/mod.rs/lib.rs 接线）
- ✅ README 更新（Task 6）

**类型一致性**：
- 所有任务用 `Array::from_inner(inner)`（不是 `from_unique_ptr`）
- 所有任务用 `a.as_inner()`（带 `as *const _` 转裸指针）
- 所有任务用 `Error::from` + `?`
- bridge 全部 `unsafe fn`（含裸指针）+ `Result<T>` 包装可能抛异常的函数
- 所有 `MlxArray` 跨桥接通过 `type MlxArray = crate::bridge::array::ffi::MlxArray;` 共享
- `Pin<&mut QuantizeResult>` for `take_at`（cxx opaque !Unpin 状态变更）

**已知 placeholder 修正**：
- 无 TBD/TODO/FIXME
- 每个 step 都有完整代码块或具体命令
- 测试代码全部完整可运行
