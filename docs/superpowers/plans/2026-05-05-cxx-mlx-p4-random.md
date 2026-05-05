# cxx-mlx P4 · Random Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MLX 随机数子系统提供完整的 Rust 安全绑定，覆盖 `mlx::core::random` 命名空间下全部公开函数（21 个 Rust 公开 fn）。完成后 LLM 解码 loop 闭环（`categorical(logits)` 在 MLX 计算图内采样 token）。

**Architecture:** 三层结构（沿用 P0–P3）：shim C++ 适配层（含共享 helpers 重构 + KeyPair opaque）→ cxx::bridge ABI 边界（free function 风格 + 全部 unsafe fn 因含裸指针）→ 安全 Rust API。`Option<&Array>` for key 全部用 `*const MlxArray` 编码（P2c/P3 模式）；`Dtype` 用 `u8 dtype_repr` + 共享 `dtype_from_repr` helper；`pair<array, array>` 用 opaque `KeyPair` + `take_first()` / `take_second()` + taken bitmap（P3 单次性消费契约）。

**Tech Stack:** Rust 1.82+（`Pin<&mut T>` for cxx opaque !Unpin 引用），cxx 1.0（含 `unsafe extern "C++"` + `*const T` 裸指针 + `&[i32]` slice），MLX C++ 共享安装（HEAD = main `369ddc1`），cargo nightly fmt + clippy + release build。

**Spec reference:** `docs/superpowers/specs/2026-05-05-cxx-mlx-p4-random-design.md`

---

## 关键背景信息（实施者必读）

### 项目三层结构

- **shim 层**：`mlx-sys/shim/include/cxx_mlx_shim/*.h` + `mlx-sys/shim/src/*.cc` —— 手写 C++，把 cxx 不能表达的 MLX 类型抹平
- **桥接层**：`mlx-sys/src/bridge/*.rs` —— `#[cxx::bridge]` 声明 ABI；项目惯例是 **free function**
- **安全层**：`mlx/src/*.rs` —— Rust 风格 API；顶层 `mlx::*` re-export

### cxx 类型映射（全部 P2b/P2c/P3 已建立）

| MLX C++ | shim 暴露 | bridge 类型 | Rust 端调用 |
|---------|----------|-------------|-------------|
| `std::optional<array>` | `*const MlxArray`（nullptr=None） | `*const MlxArray`（unsafe fn 必须） | `Option<&Array>::map_or(null, |a| a.as_inner() as *const _)` |
| `Dtype` 必传 | `uint8_t dtype_repr` | `u8` | `Dtype::as_u8()` |
| `Shape` (std::vector<int>) | `rust::Slice<const int32_t>` | `&[i32]` | `&[i32]` |
| `pair<array, array>` 出参 | opaque `KeyPair` 类 + `take_first()` / `take_second()` | `Pin<&mut KeyPair>` | 安全层连续调 `take_first` + `take_second` |
| 多 overload | 不同函数名（`bernoulli` / `bernoulli_default`、`categorical` × 3） | 各自独立 fn | 各自独立 pub fn |
| `uint64_t seed` | `uint64_t` | `u64` | `u64` |

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

### MLX 上游 API（来自 `${MLX_DIR}/include/mlx/random.h`）

```cpp
namespace mlx::core::random {

class KeySequence { ... };  // 不直接暴露给 Rust

array key(uint64_t seed);
void seed(uint64_t seed);

array bits(const Shape& shape, int width = 4,
           const std::optional<array>& key = std::nullopt, ...);

std::pair<array, array> split(const array& key, ...);
array split(const array& key, int num, ...);

array uniform(const array& low, const array& high, const Shape& shape,
              Dtype dtype = float32,
              const std::optional<array>& key = std::nullopt, ...);
array uniform(const Shape& shape, Dtype dtype,
              const std::optional<array>& key = std::nullopt, ...);

array normal(const Shape& shape, Dtype dtype,
             const std::optional<array>& loc,
             const std::optional<array>& scale,
             const std::optional<array>& key, ...);

array multivariate_normal(const array& mean, const array& cov,
                          const Shape& shape, Dtype dtype,
                          const std::optional<array>& key = std::nullopt, ...);

array randint(const array& low, const array& high, const Shape& shape,
              Dtype dtype = int32,
              const std::optional<array>& key = std::nullopt, ...);

array bernoulli(const array& p, const Shape& shape,
                const std::optional<array>& key = std::nullopt, ...);
array bernoulli(const array& p, const std::optional<array>& key, ...);

array truncated_normal(const array& lower, const array& upper, const Shape& shape,
                       Dtype dtype = float32,
                       const std::optional<array>& key = std::nullopt, ...);
array truncated_normal(const array& lower, const array& upper,
                       Dtype dtype = float32,
                       const std::optional<array>& key = std::nullopt, ...);

array gumbel(const Shape& shape, Dtype dtype = float32,
             const std::optional<array>& key = std::nullopt, ...);

array categorical(const array& logits, int axis, const Shape& shape,
                  const std::optional<array>& key = std::nullopt, ...);
array categorical(const array& logits, int axis, int num_samples,
                  const std::optional<array>& key = std::nullopt, ...);
array categorical(const array& logits, int axis = -1,
                  const std::optional<array>& key = std::nullopt, ...);

array laplace(const Shape& shape, Dtype dtype, float loc, float scale,
              const std::optional<array>& key = std::nullopt, ...);

array permutation(const array& x, int axis = 0,
                  const std::optional<array>& key = std::nullopt, ...);
array permutation(int x, const std::optional<array>& key = std::nullopt, ...);

}  // namespace mlx::core::random
```

`StreamOrDevice s = {}` 全不传（默认 caller 线程 stream）。

---

## 文件清单

### 新建
- `mlx-sys/shim/include/cxx_mlx_shim/shim_helpers.h`（共享 helpers）
- `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- `mlx-sys/shim/src/random.cc`
- `mlx-sys/src/bridge/random.rs`
- `mlx/src/random.rs`
- `mlx/tests/p4_random.rs`

### 修改
- `mlx-sys/shim/src/quantization.cc`（重构使用 shim_helpers.h）
- `mlx-sys/build.rs`
- `mlx-sys/src/bridge/mod.rs`
- `mlx-sys/src/lib.rs`
- `mlx/src/lib.rs`
- `README.md`（在 Task 6）

---

## Task 1: 框架 + helpers 重构 + state 管理 + KeyPair opaque

**目的**：抽 `shim_helpers.h` + 重构 `quantization.cc` 使用它 + 新建 random 三层接线 + state（key/seed/split/split_n）+ `KeyPair` opaque。

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/shim_helpers.h`
- Create: `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- Create: `mlx-sys/shim/src/random.cc`
- Create: `mlx-sys/src/bridge/random.rs`
- Create: `mlx/src/random.rs`
- Create: `mlx/tests/p4_random.rs`
- Modify: `mlx-sys/shim/src/quantization.cc`
- Modify: `mlx-sys/build.rs`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/src/lib.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1.1: 写失败的集成测试**

将以下完整内容写入 `mlx/tests/p4_random.rs`：

```rust
//! Integration tests for mlx::random — PRNG + distributions.

use mlx::random::{key, seed, split, split_n};
use mlx::Array;

#[test]
fn key_is_deterministic_for_same_seed() {
    let k1 = key(42).expect("key 42");
    let k2 = key(42).expect("key 42 again");
    let v1: Vec<u32> = k1.to_vec().expect("k1 to_vec");
    let v2: Vec<u32> = k2.to_vec().expect("k2 to_vec");
    assert_eq!(v1, v2, "key(42) must be deterministic");
}

#[test]
fn key_differs_for_different_seeds() {
    let k1 = key(42).expect("key 42");
    let k2 = key(43).expect("key 43");
    let v1: Vec<u32> = k1.to_vec().expect("k1");
    let v2: Vec<u32> = k2.to_vec().expect("k2");
    assert_ne!(v1, v2, "different seeds should produce different keys");
}

#[test]
fn split_returns_two_distinct_subkeys() {
    let k = key(42).expect("key");
    let (a, b) = split(&k).expect("split");
    let va: Vec<u32> = a.to_vec().expect("a");
    let vb: Vec<u32> = b.to_vec().expect("b");
    let vk: Vec<u32> = k.to_vec().expect("k");
    assert_ne!(va, vb, "split sub-keys must differ");
    assert_ne!(va, vk, "sub-key 0 must differ from parent");
    assert_ne!(vb, vk, "sub-key 1 must differ from parent");
}

#[test]
fn split_n_returns_n_keys() {
    let k = key(42).expect("key");
    let keys = split_n(&k, 5).expect("split_n");
    assert_eq!(keys.shape().as_slice()[0], 5, "first dim must be num=5");
}

#[test]
fn seed_global_is_callable() {
    // Calling seed() should not error. We don't compare global state across calls
    // because subsequent ops in this test process may also touch the default key.
    seed(123);
    seed(456);
}
```

- [ ] **Step 1.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random --no-run`
Expected: 编译失败，`mlx::random` 不存在。

- [ ] **Step 1.3: 创建共享 helpers `shim_helpers.h`**

将以下完整内容写入 `mlx-sys/shim/include/cxx_mlx_shim/shim_helpers.h`：

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlx/array.h"
#include "mlx/dtype.h"

namespace cxx_mlx::helpers {

// pointer → optional<array>。array 拷贝廉价（refcount on array_desc_）。
inline std::optional<mlx::core::array> opt_arr(
    const mlx::core::array* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<int> opt_i(bool has, int32_t v) {
  return has ? std::optional<int>(v) : std::nullopt;
}

// Dtype::Val → Dtype。MLX 14 个 dtype 全覆盖；未知值抛 runtime_error。
inline mlx::core::Dtype dtype_from_repr(uint8_t v) {
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

inline std::optional<mlx::core::Dtype> opt_dtype(bool has, uint8_t v) {
  if (!has) return std::nullopt;
  return dtype_from_repr(v);
}

}  // namespace cxx_mlx::helpers
```

- [ ] **Step 1.4: 重构 `quantization.cc` 使用共享 helpers**

打开 `mlx-sys/shim/src/quantization.cc`，找到现有的匿名 namespace（含 `opt_arr`、`opt_i`、`opt_dtype`、`dtype_from_repr` 4 个 inline helpers）。

**完整删除**该匿名 namespace，并在文件顶部 `#include "cxx_mlx_shim/quantization.h"` 之后追加：

```cpp
#include "cxx_mlx_shim/shim_helpers.h"
```

然后在 `namespace cxx_mlx {` 之后追加（紧接在打开 namespace 后）：

```cpp
using helpers::dtype_from_repr;
using helpers::opt_arr;
using helpers::opt_dtype;
using helpers::opt_i;
```

这样使用处不需要加前缀，与原匿名 namespace 行为等价。

- [ ] **Step 1.5: 创建 shim 头 `random.h`**

将以下完整内容写入 `mlx-sys/shim/include/cxx_mlx_shim/random.h`：

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "mlx/array.h"
#include "mlx/random.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== KeyPair (opaque) =====
// MLX 的 split(key) 返回 std::pair<array, array>；cxx 不支持 pair。
// 包装为 opaque 类，提供 take_first() + take_second() 接口。
// 各自的 taken_ bool 防重取（与 P3 QuantizeResult 单次性消费契约一致）。
class KeyPair {
 public:
  KeyPair(mlx::core::array first, mlx::core::array second);
  std::unique_ptr<MlxArray> take_first();
  std::unique_ptr<MlxArray> take_second();

 private:
  mlx::core::array first_;
  mlx::core::array second_;
  bool first_taken_ = false;
  bool second_taken_ = false;
};

std::unique_ptr<MlxArray> key_pair_take_first(KeyPair& p);
std::unique_ptr<MlxArray> key_pair_take_second(KeyPair& p);

// ===== State management =====
std::unique_ptr<MlxArray> key(uint64_t seed);
void seed(uint64_t seed);
std::unique_ptr<KeyPair> split(const MlxArray& key);
std::unique_ptr<MlxArray> split_n(const MlxArray& key, int32_t num);

}  // namespace cxx_mlx
```

- [ ] **Step 1.6: 创建 shim 实现 `random.cc`**

将以下完整内容写入 `mlx-sys/shim/src/random.cc`：

```cpp
#include "cxx_mlx_shim/random.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <stdexcept>

namespace cxx_mlx {

// ===== KeyPair =====

KeyPair::KeyPair(mlx::core::array first, mlx::core::array second)
    : first_(std::move(first)), second_(std::move(second)) {}

std::unique_ptr<MlxArray> KeyPair::take_first() {
  if (first_taken_) {
    throw std::runtime_error("KeyPair::take_first: already taken");
  }
  first_taken_ = true;
  return std::make_unique<MlxArray>(std::move(first_));
}

std::unique_ptr<MlxArray> KeyPair::take_second() {
  if (second_taken_) {
    throw std::runtime_error("KeyPair::take_second: already taken");
  }
  second_taken_ = true;
  return std::make_unique<MlxArray>(std::move(second_));
}

std::unique_ptr<MlxArray> key_pair_take_first(KeyPair& p) {
  return p.take_first();
}

std::unique_ptr<MlxArray> key_pair_take_second(KeyPair& p) {
  return p.take_second();
}

// ===== State =====

std::unique_ptr<MlxArray> key(uint64_t seed) {
  return std::make_unique<MlxArray>(mlx::core::random::key(seed));
}

void seed(uint64_t seed) {
  mlx::core::random::seed(seed);
}

std::unique_ptr<KeyPair> split(const MlxArray& key) {
  auto p = mlx::core::random::split(key);
  return std::make_unique<KeyPair>(std::move(p.first), std::move(p.second));
}

std::unique_ptr<MlxArray> split_n(const MlxArray& key, int32_t num) {
  return std::make_unique<MlxArray>(mlx::core::random::split(key, num));
}

}  // namespace cxx_mlx
```

- [ ] **Step 1.7: 创建桥接 `bridge/random.rs`**

将以下完整内容写入 `mlx-sys/src/bridge/random.rs`：

```rust
//! Bridge for MLX random subsystem.
//!
//! KeyPair opaque wraps std::pair<array, array> from split(key). Single-use
//! semantics: take_first / take_second each callable once (taken_ bool flag).
//!
//! Optional encodings:
//! - Option<&Array> → *const MlxArray (nullptr = None)
//! - Dtype → u8 dtype_repr (shim uses dtype_from_repr from shim_helpers.h)

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/random.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type KeyPair;

        // ===== KeyPair accessors =====
        fn key_pair_take_first(p: Pin<&mut KeyPair>) -> Result<UniquePtr<MlxArray>>;
        fn key_pair_take_second(p: Pin<&mut KeyPair>) -> Result<UniquePtr<MlxArray>>;

        // ===== State =====
        fn key(seed: u64) -> Result<UniquePtr<MlxArray>>;
        fn seed(seed: u64);
        fn split(key: &MlxArray) -> Result<UniquePtr<KeyPair>>;
        fn split_n(key: &MlxArray, num: i32) -> Result<UniquePtr<MlxArray>>;
    }
}
```

- [ ] **Step 1.8: 在 `mlx-sys/src/bridge/mod.rs` 末尾增加 `pub mod random;`**

打开 `mlx-sys/src/bridge/mod.rs`，在 `pub mod quantization;` 之后增加 `pub mod random;`。

- [ ] **Step 1.9: 在 `mlx-sys/src/lib.rs` 增加 re-export**

打开 `mlx-sys/src/lib.rs`，把已有的 re-export 块改为（cargo fmt 会按字母排序）：

```rust
pub use bridge::array;
pub use bridge::fast;
pub use bridge::io;
pub use bridge::quantization;
pub use bridge::random;
pub use bridge::stream;
pub use bridge::transforms;
```

- [ ] **Step 1.10: 在 `mlx-sys/build.rs` 注册 random 桥接 + shim cc**

打开 `mlx-sys/build.rs`，找到 `cxx_build::bridges([...])` 调用块，把它替换为：

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
        "src/bridge/quantization.rs",
        "src/bridge/random.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/fast.cc")
    .file("shim/src/io.cc")
    .file("shim/src/quantization.cc")
    .file("shim/src/random.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

- [ ] **Step 1.11: 创建安全 API `mlx/src/random.rs`**

将以下完整内容写入 `mlx/src/random.rs`：

```rust
//! MLX random number generation (PRNG).
//!
//! Functional-style RNG: explicit `key(seed) -> Array` returns a PRNG key,
//! which is split via `split(&key)` (returns 2 sub-keys) or `split_n(&key, n)`
//! to get N sub-keys. Distribution functions (added in subsequent tasks)
//! accept `Option<&Array>` for the key — None uses the global default
//! (set via `seed(seed)`).
//!
//! For LLM token sampling, see `categorical` (P4 Task 3).

use crate::{Array, Error, Result};

/// Get a PRNG key from a u64 seed. The returned array is a uint32 key
/// suitable for passing to distribution functions or to `split` / `split_n`.
pub fn key(seed: u64) -> Result<Array> {
    let inner = mlx_sys::random::ffi::key(seed).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Seed the global default PRNG key sequence. Distribution functions called
/// without an explicit `key` will use this default.
pub fn seed(seed: u64) {
    mlx_sys::random::ffi::seed(seed);
}

/// Split a key into 2 distinct sub-keys. Use to derive independent random
/// streams without correlation.
pub fn split(key: &Array) -> Result<(Array, Array)> {
    let mut pair = mlx_sys::random::ffi::split(key.as_inner()).map_err(Error::from)?;
    let first = mlx_sys::random::ffi::key_pair_take_first(pair.pin_mut())
        .map_err(Error::from)?;
    let second = mlx_sys::random::ffi::key_pair_take_second(pair.pin_mut())
        .map_err(Error::from)?;
    Ok((Array::from_inner(first), Array::from_inner(second)))
}

/// Split a key into `num` distinct sub-keys, returned as a single array
/// with shape `[num, ...]`.
pub fn split_n(key: &Array, num: i32) -> Result<Array> {
    let inner = mlx_sys::random::ffi::split_n(key.as_inner(), num)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 1.12: 在 `mlx/src/lib.rs` 增加 `pub mod random;`**

打开 `mlx/src/lib.rs`，在 `pub mod quantization;` / `pub use quantization::{...}` 块之后添加：

```rust
pub mod random;
```

（顶层 re-export 在 Task 6 统一加。）

- [ ] **Step 1.13: 编译并运行测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 5 tests passed。

- [ ] **Step 1.14: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过。**特别注意 P3 测试**（quantization）必须仍 PASS——helpers 重构正确性的关键验证。

- [ ] **Step 1.15: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/shim_helpers.h \
        mlx-sys/shim/include/cxx_mlx_shim/random.h \
        mlx-sys/shim/src/random.cc \
        mlx-sys/shim/src/quantization.cc \
        mlx-sys/src/bridge/random.rs \
        mlx-sys/src/bridge/mod.rs \
        mlx-sys/src/lib.rs \
        mlx-sys/build.rs \
        mlx/src/random.rs \
        mlx/src/lib.rs \
        mlx/tests/p4_random.rs
git commit -m "feat(p4): scaffold random module + shared helpers + state mgmt (3 layers, 5 tests)"
```

---

## Task 2: 基本分布（bits + uniform × 2 + normal + randint）

**目的**：追加 5 个基本分布函数（bits / uniform / uniform_default / normal / randint）。

**Files (all modifications, no new files):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- Modify: `mlx-sys/shim/src/random.cc`
- Modify: `mlx-sys/src/bridge/random.rs`
- Modify: `mlx/src/random.rs`
- Modify: `mlx/tests/p4_random.rs`

- [ ] **Step 2.1: 写失败的集成测试**

在 `mlx/tests/p4_random.rs` 末尾追加：

```rust
use mlx::random::{bits, normal, randint, uniform, uniform_default};
use mlx::Dtype;

#[test]
fn bits_returns_uint32_with_shape() {
    let k = key(42).expect("key");
    let b = bits(&[10], 4, Some(&k)).expect("bits");
    assert_eq!(b.shape().as_slice(), &[10]);
    let v: Vec<u32> = b.to_vec().expect("to_vec");
    assert_eq!(v.len(), 10);
}

#[test]
fn uniform_default_in_zero_to_one() {
    let k = key(42).expect("key");
    let u = uniform_default(&[100], Dtype::Float32, Some(&k)).expect("uniform");
    assert_eq!(u.shape().as_slice(), &[100]);
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0.0 && *x < 1.0, "uniform value {x} not in [0, 1)");
    }
}

#[test]
fn uniform_with_low_high_in_range() {
    let k = key(42).expect("key");
    let low = Array::from_slice(&[2.0_f32], &[]).expect("low");
    let high = Array::from_slice(&[5.0_f32], &[]).expect("high");
    let u = uniform(&low, &high, &[100], Dtype::Float32, Some(&k)).expect("uniform");
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 2.0 && *x < 5.0, "uniform value {x} not in [2, 5)");
    }
}

#[test]
fn normal_finite_and_centered() {
    let k = key(42).expect("key");
    let n = normal(&[1000], Dtype::Float32, None, None, Some(&k)).expect("normal");
    assert_eq!(n.shape().as_slice(), &[1000]);
    let v: Vec<f32> = n.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite value: {x}");
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    assert!(mean.abs() < 0.2, "normal mean {mean} not near 0 (loose tolerance for N=1000)");
}

#[test]
fn randint_in_range_and_int32() {
    let k = key(42).expect("key");
    let low = Array::from_slice(&[0_i32], &[]).expect("low");
    let high = Array::from_slice(&[10_i32], &[]).expect("high");
    let r = randint(&low, &high, &[100], Dtype::Int32, Some(&k)).expect("randint");
    let v: Vec<i32> = r.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0 && *x < 10, "randint value {x} not in [0, 10)");
    }
}
```

- [ ] **Step 2.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random --no-run`
Expected: 编译失败，函数 `bits` / `uniform` / `uniform_default` / `normal` / `randint` 未定义。

- [ ] **Step 2.3: shim 头追加 5 个声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/random.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Basic distributions =====

std::unique_ptr<MlxArray> bits(
    rust::Slice<const int32_t> shape, int32_t width,
    const MlxArray* key);

std::unique_ptr<MlxArray> uniform(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> uniform_default(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> normal(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* loc, const MlxArray* scale,
    const MlxArray* key);

std::unique_ptr<MlxArray> randint(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);
```

- [ ] **Step 2.4: shim cc 追加 5 个实现**

在 `mlx-sys/shim/src/random.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Basic distributions =====

std::unique_ptr<MlxArray> bits(
    rust::Slice<const int32_t> shape, int32_t width,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::bits(
      shape_vec, width, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> uniform(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> uniform_default(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      shape_vec, helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> normal(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* loc, const MlxArray* scale,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::normal(
      shape_vec, helpers::dtype_from_repr(dtype_repr),
      helpers::opt_arr(loc), helpers::opt_arr(scale),
      helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> randint(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::randint(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}
```

注：`random.cc` 顶部已 include `shim_helpers.h`（Task 1 时建立）；如需要可在文件 namespace `cxx_mlx` 后加 `using helpers::opt_arr;` 等便捷别名，但保持显式 `helpers::` 前缀也清晰。本 plan 用显式前缀风格。

- [ ] **Step 2.5: 桥接追加 5 个 FFI 声明**

在 `mlx-sys/src/bridge/random.rs` 的 `unsafe extern "C++"` 块内（`split_n` 之后）追加：

```rust
        // ===== Basic distributions =====
        unsafe fn bits(
            shape: &[i32], width: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn uniform(
            low: &MlxArray, high: &MlxArray,
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn uniform_default(
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn normal(
            shape: &[i32], dtype_repr: u8,
            loc: *const MlxArray, scale: *const MlxArray,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn randint(
            low: &MlxArray, high: &MlxArray,
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 2.6: 安全 API 追加 5 个函数**

在 `mlx/src/random.rs` 的 `split_n` 之后追加：

```rust
use crate::Dtype;

// ===== Basic distributions =====

/// Generate an array of random uniform 32-bit integers.
pub fn bits(shape: &[i32], width: i32, key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::bits(shape, width, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random numbers in the range `[low, high)`.
pub fn uniform(
    low: &Array,
    high: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::uniform(
            low.as_inner(), high.as_inner(), shape, dtype.as_u8(), k,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random numbers in `[0, 1)` with the given shape and dtype.
pub fn uniform_default(
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::uniform_default(shape, dtype.as_u8(), k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate samples from the normal distribution. `loc` and `scale` default
/// to 0.0 and 1.0 respectively when `None`.
pub fn normal(
    shape: &[i32],
    dtype: Dtype,
    loc: Option<&Array>,
    scale: Option<&Array>,
    key: Option<&Array>,
) -> Result<Array> {
    let l = loc.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let s = scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: l/s/k each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::normal(shape, dtype.as_u8(), l, s, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random integers in `[low, high)`.
pub fn randint(
    low: &Array,
    high: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::randint(
            low.as_inner(), high.as_inner(), shape, dtype.as_u8(), k,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 2.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 10 tests passed（5 旧 + 5 新）。

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
git add mlx-sys/shim/include/cxx_mlx_shim/random.h \
        mlx-sys/shim/src/random.cc \
        mlx-sys/src/bridge/random.rs \
        mlx/src/random.rs \
        mlx/tests/p4_random.rs
git commit -m "feat(p4): basic distributions (bits + uniform×2 + normal + randint, 5 tests)"
```

---

## Task 3: 离散分布（bernoulli × 2 + categorical × 3）

**目的**：追加 5 个离散分布函数。`categorical` 是 LLM token sampling 工作主力（3 个 overload）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- Modify: `mlx-sys/shim/src/random.cc`
- Modify: `mlx-sys/src/bridge/random.rs`
- Modify: `mlx/src/random.rs`
- Modify: `mlx/tests/p4_random.rs`

- [ ] **Step 3.1: 写失败的集成测试**

在 `mlx/tests/p4_random.rs` 末尾追加：

```rust
use mlx::random::{
    bernoulli, bernoulli_default, categorical, categorical_n, categorical_shaped,
};

#[test]
fn bernoulli_only_zero_or_one() {
    let k = key(42).expect("key");
    let p = Array::from_slice(&[0.5_f32], &[]).expect("p");
    let b = bernoulli(&p, &[100], Some(&k)).expect("bernoulli");
    assert_eq!(b.shape().as_slice(), &[100]);
    let v: Vec<bool> = b.to_vec().expect("to_vec");
    assert_eq!(v.len(), 100);
    // bool 元素都是 0/1，由 to_vec::<bool> 类型保证
}

#[test]
fn bernoulli_default_shape_from_p() {
    // p 是标量 → bernoulli 输出标量
    let k = key(42).expect("key");
    let p = Array::from_slice(&[0.7_f32], &[]).expect("p");
    let b = bernoulli_default(&p, Some(&k)).expect("bernoulli_default");
    // 标量输出 shape 是 [] 空形状
    assert_eq!(b.shape().as_slice(), &[] as &[i32]);
}

#[test]
fn categorical_index_in_vocab() {
    // logits shape [batch=4, vocab=8]，axis=-1 沿 vocab 采样
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::from_slice(&logits_data, &[4, 8]).expect("logits");

    let out = categorical(&logits, -1, Some(&k)).expect("categorical");
    // 默认 sample 1 along axis：输出 shape [4]
    assert_eq!(out.shape().as_slice(), &[4]);
    let v: Vec<u32> = out.to_vec().expect("to_vec");
    for idx in &v {
        assert!(*idx < 8, "categorical idx {idx} out of vocab=[0, 8)");
    }
}

#[test]
fn categorical_n_returns_n_samples() {
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::from_slice(&logits_data, &[4, 8]).expect("logits");

    let out = categorical_n(&logits, -1, 3, Some(&k)).expect("categorical_n");
    // 输出 shape [4, 3]：每 batch 3 个采样
    assert_eq!(out.shape().as_slice(), &[4, 3]);
    let v: Vec<u32> = out.to_vec().expect("to_vec");
    for idx in &v {
        assert!(*idx < 8, "categorical_n idx {idx} out of vocab");
    }
}

#[test]
fn categorical_shaped_returns_explicit_shape() {
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::from_slice(&logits_data, &[2, 8]).expect("logits");

    // 显式 shape [2, 5]：每 batch 5 个采样
    let out = categorical_shaped(&logits, -1, &[2, 5], Some(&k)).expect("categorical_shaped");
    assert_eq!(out.shape().as_slice(), &[2, 5]);
}
```

- [ ] **Step 3.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random --no-run`
Expected: 编译失败，5 个新函数未定义。

- [ ] **Step 3.3: shim 头追加 5 个声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/random.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Discrete distributions =====

std::unique_ptr<MlxArray> bernoulli(
    const MlxArray& p, rust::Slice<const int32_t> shape,
    const MlxArray* key);

std::unique_ptr<MlxArray> bernoulli_default(
    const MlxArray& p,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical(
    const MlxArray& logits, int32_t axis,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical_n(
    const MlxArray& logits, int32_t axis, int32_t num_samples,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical_shaped(
    const MlxArray& logits, int32_t axis,
    rust::Slice<const int32_t> shape,
    const MlxArray* key);
```

- [ ] **Step 3.4: shim cc 追加 5 个实现**

在 `mlx-sys/shim/src/random.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Discrete distributions =====

std::unique_ptr<MlxArray> bernoulli(
    const MlxArray& p, rust::Slice<const int32_t> shape,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::bernoulli(
      p, shape_vec, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> bernoulli_default(
    const MlxArray& p,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::bernoulli(
      p, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> categorical(
    const MlxArray& logits, int32_t axis,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> categorical_n(
    const MlxArray& logits, int32_t axis, int32_t num_samples,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, num_samples, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> categorical_shaped(
    const MlxArray& logits, int32_t axis,
    rust::Slice<const int32_t> shape,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, shape_vec, helpers::opt_arr(key)));
}
```

- [ ] **Step 3.5: 桥接追加 5 个 FFI 声明**

在 `mlx-sys/src/bridge/random.rs` 的 `unsafe extern "C++"` 块内（`randint` 之后）追加：

```rust
        // ===== Discrete distributions =====
        unsafe fn bernoulli(
            p: &MlxArray, shape: &[i32],
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn bernoulli_default(
            p: &MlxArray,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical(
            logits: &MlxArray, axis: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical_n(
            logits: &MlxArray, axis: i32, num_samples: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical_shaped(
            logits: &MlxArray, axis: i32, shape: &[i32],
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 3.6: 安全 API 追加 5 个函数**

在 `mlx/src/random.rs` 的 `randint` 之后追加：

```rust
// ===== Discrete distributions =====

/// Sample binary (0/1) values with probability `p`. Output shape must match
/// `p`'s broadcastable shape via the `shape` argument.
pub fn bernoulli(
    p: &Array,
    shape: &[i32],
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::bernoulli(p.as_inner(), shape, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample binary values with probability `p`. Output shape inferred from `p`.
pub fn bernoulli_default(
    p: &Array,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::bernoulli_default(p.as_inner(), k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample 1 index per row from `logits` along `axis`. The canonical token
/// sampling op for LLM decoding.
pub fn categorical(
    logits: &Array,
    axis: i32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::categorical(logits.as_inner(), axis, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample `num_samples` indices per row from `logits` along `axis`.
pub fn categorical_n(
    logits: &Array,
    axis: i32,
    num_samples: i32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::categorical_n(logits.as_inner(), axis, num_samples, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample with explicit output `shape` from `logits` along `axis`.
pub fn categorical_shaped(
    logits: &Array,
    axis: i32,
    shape: &[i32],
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::categorical_shaped(logits.as_inner(), axis, shape, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 3.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 15 tests passed（10 旧 + 5 新）。

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
git add mlx-sys/shim/include/cxx_mlx_shim/random.h \
        mlx-sys/shim/src/random.cc \
        mlx-sys/src/bridge/random.rs \
        mlx/src/random.rs \
        mlx/tests/p4_random.rs
git commit -m "feat(p4): discrete distributions (bernoulli×2 + categorical×3, 5 tests)"
```

---

## Task 4: 特殊分布（truncated_normal × 2 + gumbel + laplace + multivariate_normal）

**目的**：追加 5 个特殊分布函数。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- Modify: `mlx-sys/shim/src/random.cc`
- Modify: `mlx-sys/src/bridge/random.rs`
- Modify: `mlx/src/random.rs`
- Modify: `mlx/tests/p4_random.rs`

- [ ] **Step 4.1: 写失败的集成测试**

在 `mlx/tests/p4_random.rs` 末尾追加：

```rust
use mlx::random::{
    gumbel, laplace, multivariate_normal, truncated_normal, truncated_normal_default,
};

#[test]
fn truncated_normal_in_bounds() {
    let k = key(42).expect("key");
    let lower = Array::from_slice(&[-1.0_f32], &[]).expect("lower");
    let upper = Array::from_slice(&[1.0_f32], &[]).expect("upper");
    let t = truncated_normal(
        &lower, &upper, &[100], Dtype::Float32, Some(&k),
    )
    .expect("truncated_normal");
    let v: Vec<f32> = t.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= -1.0 && *x <= 1.0, "truncated value {x} out of [-1, 1]");
    }
}

#[test]
fn truncated_normal_default_broadcast_shape() {
    let k = key(42).expect("key");
    let lower = Array::from_slice(&[-1.0_f32, -2.0], &[2]).expect("lower");
    let upper = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("upper");
    let t = truncated_normal_default(&lower, &upper, Dtype::Float32, Some(&k))
        .expect("truncated_normal_default");
    // shape 从 broadcast(lower, upper) = [2]
    assert_eq!(t.shape().as_slice(), &[2]);
}

#[test]
fn gumbel_finite() {
    let k = key(42).expect("key");
    let g = gumbel(&[100], Dtype::Float32, Some(&k)).expect("gumbel");
    assert_eq!(g.shape().as_slice(), &[100]);
    let v: Vec<f32> = g.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite gumbel value: {x}");
    }
}

#[test]
fn laplace_finite() {
    let k = key(42).expect("key");
    let l = laplace(&[100], Dtype::Float32, 0.0, 1.0, Some(&k)).expect("laplace");
    assert_eq!(l.shape().as_slice(), &[100]);
    let v: Vec<f32> = l.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite laplace value: {x}");
    }
}

#[test]
fn multivariate_normal_correct_shape() {
    let k = key(42).expect("key");
    // 2-d distribution; mean shape [2], cov shape [2, 2]
    let mean = Array::from_slice(&[0.0_f32, 0.0], &[2]).expect("mean");
    let cov = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[2, 2]).expect("cov");
    let mvn = multivariate_normal(
        &mean, &cov, &[10], Dtype::Float32, Some(&k),
    )
    .expect("multivariate_normal");
    // shape [num_samples, dim] = [10, 2]
    assert_eq!(mvn.shape().as_slice(), &[10, 2]);
}
```

- [ ] **Step 4.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random --no-run`
Expected: 编译失败，5 个新函数未定义。

- [ ] **Step 4.3: shim 头追加 5 个声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/random.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Special distributions =====

std::unique_ptr<MlxArray> truncated_normal(
    const MlxArray& lower, const MlxArray& upper,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> truncated_normal_default(
    const MlxArray& lower, const MlxArray& upper,
    uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> gumbel(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> laplace(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    float loc, float scale,
    const MlxArray* key);

std::unique_ptr<MlxArray> multivariate_normal(
    const MlxArray& mean, const MlxArray& cov,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);
```

- [ ] **Step 4.4: shim cc 追加 5 个实现**

在 `mlx-sys/shim/src/random.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Special distributions =====

std::unique_ptr<MlxArray> truncated_normal(
    const MlxArray& lower, const MlxArray& upper,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
      lower, upper, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> truncated_normal_default(
    const MlxArray& lower, const MlxArray& upper,
    uint8_t dtype_repr,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
      lower, upper,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> gumbel(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::gumbel(
      shape_vec, helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> laplace(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    float loc, float scale,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::laplace(
      shape_vec, helpers::dtype_from_repr(dtype_repr), loc, scale,
      helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> multivariate_normal(
    const MlxArray& mean, const MlxArray& cov,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  std::vector<int> shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::multivariate_normal(
      mean, cov, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}
```

- [ ] **Step 4.5: 桥接追加 5 个 FFI 声明**

在 `mlx-sys/src/bridge/random.rs` 的 `unsafe extern "C++"` 块内（`categorical_shaped` 之后）追加：

```rust
        // ===== Special distributions =====
        unsafe fn truncated_normal(
            lower: &MlxArray, upper: &MlxArray,
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn truncated_normal_default(
            lower: &MlxArray, upper: &MlxArray,
            dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn gumbel(
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn laplace(
            shape: &[i32], dtype_repr: u8,
            loc: f32, scale: f32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn multivariate_normal(
            mean: &MlxArray, cov: &MlxArray,
            shape: &[i32], dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 4.6: 安全 API 追加 5 个函数**

在 `mlx/src/random.rs` 的 `categorical_shaped` 之后追加：

```rust
// ===== Special distributions =====

/// Generate samples from a truncated normal distribution restricted to
/// `[lower, upper]`.
pub fn truncated_normal(
    lower: &Array,
    upper: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::truncated_normal(
            lower.as_inner(), upper.as_inner(), shape, dtype.as_u8(), k,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Truncated normal with output shape inferred from `broadcast(lower, upper)`.
pub fn truncated_normal_default(
    lower: &Array,
    upper: &Array,
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::truncated_normal_default(
            lower.as_inner(), upper.as_inner(), dtype.as_u8(), k,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate samples from the standard Gumbel distribution.
pub fn gumbel(
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::gumbel(shape, dtype.as_u8(), k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate samples from the Laplace distribution with location `loc` and
/// scale `scale`.
pub fn laplace(
    shape: &[i32],
    dtype: Dtype,
    loc: f32,
    scale: f32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::laplace(shape, dtype.as_u8(), loc, scale, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate samples from a multivariate normal distribution with mean `mean`
/// and covariance `cov`. Output shape is `[..., dim]` where `dim` is the
/// last dimension of `mean`.
pub fn multivariate_normal(
    mean: &Array,
    cov: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::multivariate_normal(
            mean.as_inner(), cov.as_inner(), shape, dtype.as_u8(), k,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 20 tests passed（15 旧 + 5 新）。

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
git add mlx-sys/shim/include/cxx_mlx_shim/random.h \
        mlx-sys/shim/src/random.cc \
        mlx-sys/src/bridge/random.rs \
        mlx/src/random.rs \
        mlx/tests/p4_random.rs
git commit -m "feat(p4): special distributions (truncated_normal×2 + gumbel + laplace + multivariate_normal, 5 tests)"
```

---

## Task 5: 置换（permutation × 2）

**目的**：追加 2 个置换函数。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/random.h`
- Modify: `mlx-sys/shim/src/random.cc`
- Modify: `mlx-sys/src/bridge/random.rs`
- Modify: `mlx/src/random.rs`
- Modify: `mlx/tests/p4_random.rs`

- [ ] **Step 5.1: 写失败的集成测试**

在 `mlx/tests/p4_random.rs` 末尾追加：

```rust
use mlx::random::{permutation, permutation_arange};

#[test]
fn permutation_arange_is_valid_perm() {
    let k = key(42).expect("key");
    let p = permutation_arange(10, Some(&k)).expect("permutation_arange");
    assert_eq!(p.shape().as_slice(), &[10]);

    let mut v: Vec<i32> = p.to_vec().expect("to_vec");
    v.sort();
    assert_eq!(v, (0..10).collect::<Vec<i32>>(),
               "permutation must be a re-ordering of 0..n");
}

#[test]
fn permutation_array_preserves_elements() {
    let k = key(42).expect("key");
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], &[5]).expect("x");
    let p = permutation(&x, 0, Some(&k)).expect("permutation");
    assert_eq!(p.shape().as_slice(), &[5]);

    let mut v: Vec<f32> = p.to_vec().expect("to_vec");
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(v, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
               "permutation must preserve the multiset");
}
```

- [ ] **Step 5.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random --no-run`
Expected: 编译失败，2 个新函数未定义。

- [ ] **Step 5.3: shim 头追加 2 个声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/random.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Permutation =====

std::unique_ptr<MlxArray> permutation(
    const MlxArray& x, int32_t axis,
    const MlxArray* key);

std::unique_ptr<MlxArray> permutation_arange(
    int32_t n,
    const MlxArray* key);
```

- [ ] **Step 5.4: shim cc 追加 2 个实现**

在 `mlx-sys/shim/src/random.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== Permutation =====

std::unique_ptr<MlxArray> permutation(
    const MlxArray& x, int32_t axis,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::permutation(
      x, axis, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> permutation_arange(
    int32_t n,
    const MlxArray* key) {
  return std::make_unique<MlxArray>(mlx::core::random::permutation(
      n, helpers::opt_arr(key)));
}
```

- [ ] **Step 5.5: 桥接追加 2 个 FFI 声明**

在 `mlx-sys/src/bridge/random.rs` 的 `unsafe extern "C++"` 块内（`multivariate_normal` 之后）追加：

```rust
        // ===== Permutation =====
        unsafe fn permutation(
            x: &MlxArray, axis: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn permutation_arange(
            n: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 5.6: 安全 API 追加 2 个函数**

在 `mlx/src/random.rs` 的 `multivariate_normal` 之后追加：

```rust
// ===== Permutation =====

/// Randomly permute the elements of `x` along `axis`.
pub fn permutation(
    x: &Array,
    axis: i32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::permutation(x.as_inner(), axis, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Return a random permutation of `arange(n)`.
pub fn permutation_arange(
    n: i32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::permutation_arange(n, k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 5.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 22 tests passed（20 旧 + 2 新）。

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
git add mlx-sys/shim/include/cxx_mlx_shim/random.h \
        mlx-sys/shim/src/random.cc \
        mlx-sys/src/bridge/random.rs \
        mlx/src/random.rs \
        mlx/tests/p4_random.rs
git commit -m "feat(p4): permutation×2 (2 tests)"
```

---

## Task 6: 顶层 re-export + README + 全套验证

**目的**：把 P4 公开 API 暴露到 `mlx::*` 顶层，README 升级到 P4 完成，跑完整 workspace 检查。

**Files:**
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/tests/p4_random.rs`
- Modify: `README.md`

- [ ] **Step 6.1: 在 `mlx/src/lib.rs` 的 `pub mod random;` 后追加 re-export**

打开 `mlx/src/lib.rs`，找到 `pub mod random;`，在其后追加 re-export（cargo fmt 会按字母排序）：

```rust
pub mod random;
pub use random::{
    bernoulli, bernoulli_default, bits, categorical, categorical_n, categorical_shaped,
    gumbel, key, laplace, multivariate_normal, normal, permutation, permutation_arange,
    randint, seed, split, split_n, truncated_normal, truncated_normal_default,
    uniform, uniform_default,
};
```

- [ ] **Step 6.2: 在测试文件追加 re-export 验证测试**

在 `mlx/tests/p4_random.rs` 末尾追加：

```rust
#[test]
fn top_level_re_exports_work() {
    // 验证可以通过 mlx::* 顶层访问 P4 公开 API
    let k = mlx::key(42).expect("key via mlx::*");
    let u = mlx::uniform_default(&[10], mlx::Dtype::Float32, Some(&k))
        .expect("uniform_default via mlx::*");
    assert_eq!(u.shape().as_slice(), &[10]);
}
```

- [ ] **Step 6.3: 运行所有 P4 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p4_random`
Expected: 23 tests passed（22 + re-export 1）。

- [ ] **Step 6.4: 跑完整 workspace 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --workspace --all-features`
Expected: 所有现有测试 + 23 P4 测试全部通过。

- [ ] **Step 6.5: 更新 `README.md`**

`README.md` 的 status banner 与 Roadmap 升级。

**位置 A（Status banner，约 line 5）**：当前文本类似：
```
**Status:** 🎉 **P3 complete (0.1 release candidate)** — Full quantization subsystem ...
```

升级为：
```
**Status:** 🎉 **P4 complete** — `mlx::random` PRNG + 21 distribution functions including `categorical` (token sampling). Combined with P3 (quantization) + P2c (IO) + P2b (fast ops) + P1 (ops/array foundations), the LLM decode loop is now end-to-end inside MLX's compute graph.
```

**位置 B（Roadmap 表格）**：在 P3 行下追加 P4 行：
```
- ✅ **P3** — `quantization` (...) — 8 integration tests
- ✅ **P4** — `random` (key/seed/split + 17 distributions including categorical) — 23 integration tests
```

如有 ⏳ 行如 "compile + LLM inference example"，保留并在描述里把 random 移除（已完成）。

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
Expected: 全部通过。

- [ ] **Step 6.7: 提交**

```bash
git add mlx/src/lib.rs mlx/tests/p4_random.rs README.md
git commit -m "feat(p4): re-export random API at crate root + README progress"
```

- [ ] **Step 6.8: 最终 git log 与 commit 数核对**

Run: `git log --oneline | head -10`
Expected: 看到 6 个 P4 feat commit + docs (spec + plan) commit。

---

## 自检（plan 作者自检结果）

**Spec 覆盖**：
- ✅ State 管理（key/seed/split/split_n + KeyPair opaque）→ Task 1
- ✅ 共享 helpers 重构（shim_helpers.h + quantization.cc 改用）→ Task 1
- ✅ 基本分布（bits + uniform×2 + normal + randint）→ Task 2
- ✅ 离散分布（bernoulli×2 + categorical×3）→ Task 3
- ✅ 特殊分布（truncated_normal×2 + gumbel + laplace + multivariate_normal）→ Task 4
- ✅ 置换（permutation×2）→ Task 5
- ✅ Re-export + README → Task 6
- ✅ 21 个公开 fn 全部覆盖

**类型一致性**：
- 所有 task 用 `Array::from_inner(inner)`
- 所有 task 用 `array.as_inner() as *const _` 转裸指针
- 所有 task 用 `Error::from` + `?`
- bridge 含裸指针的全部 `unsafe fn`
- 跨桥接 `MlxArray` 共享 `type` alias
- `Pin<&mut KeyPair>` for take_first / take_second

**已知 placeholder**：
- 无 TBD/TODO/FIXME
- 每个 step 都有完整代码块或具体命令
