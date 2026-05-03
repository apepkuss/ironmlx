# cxx-mlx P0 (Scaffold) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `cxx-mlx` Cargo workspace (`mlx-sys` + `mlx`) with a minimal cxx FFI bridge and C++ shim covering `array_zeros` / `shape` / `dtype` / `size` / `ndim` / `eval`, and ship a passing smoke test on macOS Apple Silicon.

**Architecture:** Cargo workspace with two crates. `mlx-sys` houses cxx::bridge modules, a hand-written C++ shim that flattens MLX templates/overloads into cxx-friendly free functions, and a `build.rs` that locates MLX via `MLX_DIR` and links the static lib + macOS frameworks. `mlx` re-exports a safe `Array` type with `Drop`/methods built on top.

**Tech Stack:** Rust 1.94+, [cxx](https://cxx.rs) 1.0, MLX C++ 0.32 (existing build at `$MLX_DIR`), C++20, Apple Silicon macOS only.

---

## Prerequisites (do BEFORE starting)

The engineer must have a prebuilt MLX install before P0 can be tested. If not already present:

```bash
cd /Volumes/Dev/mlx
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/mlx
make -j$(sysctl -n hw.ncpu)
make install
export MLX_DIR=$HOME/.local/mlx
```

Sanity check:

```bash
ls $MLX_DIR/include/mlx/array.h && ls $MLX_DIR/lib/libmlx.a
```

Both files must exist or P0 cannot link. `MLX_DIR` must be exported in the shell where `cargo test` runs.

---

## File Structure

**Files this plan creates:**

```text
cxx-mlx/
├── Cargo.toml                                       # workspace root
├── .gitignore
├── README.md                                        # quickstart only
├── mlx-sys/
│   ├── Cargo.toml
│   ├── build.rs                                     # MLX discovery + cxx_build
│   ├── src/
│   │   ├── lib.rs                                   # platform guard + re-exports
│   │   └── bridge/
│   │       ├── mod.rs
│   │       ├── array.rs                             # cxx::bridge for array ops
│   │       └── transforms.rs                        # cxx::bridge for eval
│   ├── shim/
│   │   ├── include/cxx_mlx_shim/
│   │   │   ├── array.h
│   │   │   └── transforms.h
│   │   └── src/
│   │       ├── array.cc
│   │       └── transforms.cc
│   └── tests/
│       └── sys_smoke.rs
└── mlx/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── array.rs                                 # safe Array wrapper
    │   ├── dtype.rs                                 # Dtype enum + conversions
    │   └── error.rs                                 # Error + Result
    └── tests/
        └── p0_smoke.rs
```

**Responsibility split:**

- `mlx-sys/build.rs` — locate MLX, configure cxx_build, link
- `mlx-sys/shim/` — C++ shim (one header + cc per MLX subsystem)
- `mlx-sys/src/bridge/` — cxx::bridge modules (one per shim subsystem)
- `mlx-sys/src/lib.rs` — platform guard, re-export bridge namespaces
- `mlx/src/dtype.rs` — `Dtype` enum mirrored to MLX's `Dtype::Val` (u8 over the wire)
- `mlx/src/array.rs` — `Array` type wrapping `UniquePtr<sys::MlxArray>`, methods
- `mlx/src/error.rs` — `Error` + `Result<T>` aliases

---

## Task 1: Workspace skeleton

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `README.md`
- Create: `mlx-sys/Cargo.toml`, `mlx-sys/src/lib.rs`
- Create: `mlx/Cargo.toml`, `mlx/src/lib.rs`

- [ ] **Step 1: Create workspace root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["mlx-sys", "mlx"]

[workspace.package]
version = "0.0.1"
edition = "2021"
rust-version = "1.94"
license = "MIT OR Apache-2.0"
repository = "https://github.com/wei/cxx-mlx"

[workspace.dependencies]
cxx = "1.0"
cxx-build = "1.0"
thiserror = "2.0"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
/target
**/*.rs.bk
Cargo.lock
.DS_Store
```

(`Cargo.lock` is excluded because this is a library workspace; binary crates added later can override.)

- [ ] **Step 3: Create `README.md`**

````markdown
# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** P0 scaffold (zeros + eval + shape only). See `docs/superpowers/specs/` for the full design.

## Requirements

- macOS, Apple Silicon
- Rust 1.94+
- Prebuilt MLX 0.32+ at `$MLX_DIR` (see `docs/superpowers/plans/2026-05-03-cxx-mlx-p0-scaffold.md` for build instructions)

## Quickstart

```rust
use mlx::{Array, Dtype};

let a = Array::zeros(&[2, 3], Dtype::Float32);
assert_eq!(a.shape(), vec![2, 3]);
a.eval().unwrap();
```
````

- [ ] **Step 4: Create `mlx-sys/Cargo.toml`**

```toml
[package]
name = "mlx-sys"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
links = "mlx"
description = "Raw FFI bindings to MLX C++ via cxx"

[dependencies]
cxx.workspace = true

[build-dependencies]
cxx-build.workspace = true

[features]
default = []
```

- [ ] **Step 5: Create `mlx-sys/src/lib.rs` (placeholder)**

```rust
//! Raw FFI bindings to MLX C++.
//!
//! This crate is the `-sys` half of `cxx-mlx`. For a safe, idiomatic API,
//! depend on the `mlx` crate instead.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");
```

- [ ] **Step 6: Create `mlx/Cargo.toml`**

```toml
[package]
name = "mlx"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Safe, idiomatic Rust bindings to Apple MLX"

[dependencies]
mlx-sys = { path = "../mlx-sys", version = "0.0.1" }
cxx.workspace = true
thiserror.workspace = true
```

- [ ] **Step 7: Create `mlx/src/lib.rs` (placeholder)**

```rust
//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");
```

- [ ] **Step 8: Verify workspace builds**

Run: `cargo check --workspace`
Expected: PASS (both crates compile as empty libs).

If `cargo` complains about `links = "mlx"` without a `build.rs`, that's expected — we add `build.rs` in Task 3. Move on.

- [ ] **Step 9: Commit**

```bash
git add .gitignore README.md Cargo.toml mlx-sys/ mlx/
git commit -m "feat(p0): workspace skeleton for mlx-sys and mlx crates"
```

---

## Task 2: build.rs MLX discovery

**Files:**
- Create: `mlx-sys/build.rs`

- [ ] **Step 1: Create `mlx-sys/build.rs` with MLX_DIR discovery only**

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=src/bridge");

    // P0 only supports the MLX_DIR discovery path. P1 adds MLX_INCLUDE_DIR/MLX_LIB_DIR
    // and pkg-config fallback; P2 adds the `bundled` feature.
    let (include_dir, lib_dir) = locate_mlx();

    // Verify the MLX install looks sane before going further.
    let array_h = include_dir.join("mlx/array.h");
    if !array_h.exists() {
        panic!(
            "MLX install at {} is missing mlx/array.h — is MLX_DIR pointing at the install prefix?",
            include_dir.display()
        );
    }

    // Link search path
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Mandatory link: libmlx (linker picks .a or .dylib; document static preference in README)
    println!("cargo:rustc-link-lib=mlx");

    // Optional transitive static deps that MLX ships in its lib/ when built static
    for dep in ["fmt", "gguflib", "mlxblas"] {
        if lib_dir.join(format!("lib{dep}.a")).exists() {
            println!("cargo:rustc-link-lib=static={dep}");
        }
    }

    // macOS frameworks MLX uses
    for fw in [
        "Metal",
        "Foundation",
        "Accelerate",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }

    // C++ standard library
    println!("cargo:rustc-link-lib=c++");

    // Expose include dir for later cxx_build wiring (Task 4)
    let _ = include_dir;
}

fn locate_mlx() -> (PathBuf, PathBuf) {
    let mlx_dir = env::var_os("MLX_DIR").map(PathBuf::from).unwrap_or_else(|| {
        panic!(
            "MLX_DIR is not set. Build MLX first (see docs/superpowers/plans/2026-05-03-cxx-mlx-p0-scaffold.md) \
             and export MLX_DIR=<install prefix>."
        )
    });
    let include = mlx_dir.join("include");
    let lib = mlx_dir.join("lib");
    if !include.is_dir() || !lib.is_dir() {
        panic!(
            "MLX_DIR={} does not look like an MLX install prefix (missing include/ or lib/)",
            mlx_dir.display()
        );
    }
    (include, lib)
}
```

- [ ] **Step 2: Verify `build.rs` runs and finds MLX**

Run: `MLX_DIR=$HOME/.local/mlx cargo build -p mlx-sys`
Expected: PASS (no link step yet because no objects, but build.rs runs without panicking).

If build.rs panics with `MLX_DIR is not set`, the engineer needs to export it. If it panics with `mlx/array.h missing`, the MLX install is broken — re-run prerequisites.

- [ ] **Step 3: Commit**

```bash
git add mlx-sys/build.rs
git commit -m "feat(p0): mlx-sys build.rs locates MLX via MLX_DIR and configures linking"
```

---

## Task 3: cxx bridge for array_zeros + array_shape (TDD, red phase)

**Files:**
- Create: `mlx-sys/src/bridge/mod.rs`, `mlx-sys/src/bridge/array.rs`
- Create: `mlx-sys/tests/sys_smoke.rs`

- [ ] **Step 1: Write the failing test first**

Create `mlx-sys/tests/sys_smoke.rs`:

```rust
use mlx_sys::array::ffi;

// Mirror of mlx::core::Dtype::Val. Verified against mlx/dtype.h:
// bool_=0, uint8=1, uint16=2, uint32=3, uint64=4, int8=5, int16=6, int32=7,
// int64=8, float16=9, float32=10, float64=11, bfloat16=12, complex64=13.
const FLOAT32: u8 = 10;

#[test]
fn zeros_then_read_shape() {
    let arr = ffi::array_zeros(&[2, 3], FLOAT32);
    let shape = ffi::array_shape(&arr);
    assert_eq!(shape, vec![2, 3]);
}
```

- [ ] **Step 2: Verify the test fails with "module not found"**

Run: `cargo test -p mlx-sys --test sys_smoke`
Expected: FAIL with `unresolved import mlx_sys::array` (the bridge module does not exist yet).

- [ ] **Step 3: Create `mlx-sys/src/bridge/mod.rs`**

```rust
pub mod array;
```

- [ ] **Step 4: Create `mlx-sys/src/bridge/array.rs` (cxx bridge)**

```rust
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");

        /// Opaque holder for `mlx::core::array`. Internally refcounted by MLX.
        type MlxArray;

        fn array_zeros(shape: &[i32], dtype: u8) -> UniquePtr<MlxArray>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
    }
}
```

- [ ] **Step 5: Update `mlx-sys/src/lib.rs` to expose the bridge**

Replace contents with:

```rust
//! Raw FFI bindings to MLX C++.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod bridge;

pub use bridge::array;
```

- [ ] **Step 6: Create the shim header `mlx-sys/shim/include/cxx_mlx_shim/array.h`**

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

// Forward-declare so the cxx-generated header doesn't need to pull in mlx.h.
namespace mlx::core {
class array;
}

namespace cxx_mlx {

using MlxArray = mlx::core::array;

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype);
rust::Vec<int32_t> array_shape(const MlxArray& a);

}  // namespace cxx_mlx
```

- [ ] **Step 7: Create the shim implementation `mlx-sys/shim/src/array.cc`**

```cpp
#include "cxx_mlx_shim/array.h"

#include <stdexcept>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"

namespace cxx_mlx {

namespace {

mlx::core::Dtype dtype_from_u8(uint8_t v) {
  using V = mlx::core::Dtype::Val;
  switch (static_cast<V>(v)) {
    case V::bool_: return mlx::core::bool_;
    case V::uint8: return mlx::core::uint8;
    case V::uint16: return mlx::core::uint16;
    case V::uint32: return mlx::core::uint32;
    case V::uint64: return mlx::core::uint64;
    case V::int8: return mlx::core::int8;
    case V::int16: return mlx::core::int16;
    case V::int32: return mlx::core::int32;
    case V::int64: return mlx::core::int64;
    case V::float16: return mlx::core::float16;
    case V::float32: return mlx::core::float32;
    case V::float64: return mlx::core::float64;
    case V::bfloat16: return mlx::core::bfloat16;
    case V::complex64: return mlx::core::complex64;
  }
  throw std::invalid_argument("cxx_mlx: unknown Dtype::Val value");
}

}  // namespace

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::zeros(s, dtype_from_u8(dtype)));
}

rust::Vec<int32_t> array_shape(const MlxArray& a) {
  rust::Vec<int32_t> out;
  for (auto v : a.shape()) {
    out.push_back(v);
  }
  return out;
}

}  // namespace cxx_mlx
```

- [ ] **Step 8: Wire `cxx_build` into `build.rs`**

Edit `mlx-sys/build.rs`. Add at the very end of `main()` (after the framework links):

```rust
    cxx_build::bridge("src/bridge/array.rs")
        .file("shim/src/array.cc")
        .include("shim/include")
        .include(&include_dir)
        .std("c++20")
        .flag_if_supported("-fvisibility=hidden")
        .compile("cxx_mlx_shim");
```

And remove the `let _ = include_dir;` line (no longer dead code).

- [ ] **Step 9: Verify the test passes**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke -- --nocapture`
Expected: PASS — test prints nothing, but `zeros_then_read_shape` is reported as `ok`.

If link errors mention undefined `fmt::*` or similar 3rdparty symbols, your MLX install was built with `BUILD_SHARED_LIBS=ON` and the static `.a`'s aren't present. Either rebuild MLX with `-DBUILD_SHARED_LIBS=OFF` or document the workaround.

If link errors mention `MetalPerformanceShadersGraph`, your Xcode SDK is older than what MLX needs. Update Xcode CLT.

- [ ] **Step 10: Commit**

```bash
git add mlx-sys/build.rs mlx-sys/src/ mlx-sys/shim/ mlx-sys/tests/
git commit -m "feat(p0): cxx bridge + shim for array_zeros and array_shape"
```

---

## Task 4: Add dtype, size, ndim to the array bridge (TDD)

**Files:**
- Modify: `mlx-sys/tests/sys_smoke.rs`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`

- [ ] **Step 1: Extend the failing test**

Replace contents of `mlx-sys/tests/sys_smoke.rs` with:

```rust
use mlx_sys::array::ffi;

const FLOAT32: u8 = 5;

#[test]
fn zeros_then_read_shape() {
    let arr = ffi::array_zeros(&[2, 3], FLOAT32);
    assert_eq!(ffi::array_shape(&arr), vec![2, 3]);
}

#[test]
fn zeros_metadata() {
    let arr = ffi::array_zeros(&[2, 3, 4], FLOAT32);
    assert_eq!(ffi::array_ndim(&arr), 3);
    assert_eq!(ffi::array_size(&arr), 24);
    assert_eq!(ffi::array_dtype(&arr), FLOAT32);
}
```

- [ ] **Step 2: Verify the new test fails**

Run: `cargo test -p mlx-sys --test sys_smoke zeros_metadata`
Expected: FAIL with `function or associated item array_ndim not found`.

- [ ] **Step 3: Extend the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add three lines inside `extern "C++"`:

```rust
        fn array_ndim(a: &MlxArray) -> usize;
        fn array_size(a: &MlxArray) -> usize;
        fn array_dtype(a: &MlxArray) -> u8;
```

So the full `extern` block reads:

```rust
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");

        type MlxArray;

        fn array_zeros(shape: &[i32], dtype: u8) -> UniquePtr<MlxArray>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
        fn array_ndim(a: &MlxArray) -> usize;
        fn array_size(a: &MlxArray) -> usize;
        fn array_dtype(a: &MlxArray) -> u8;
    }
```

- [ ] **Step 4: Extend the shim header**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add three declarations after `array_shape`:

```cpp
size_t array_ndim(const MlxArray& a);
size_t array_size(const MlxArray& a);
uint8_t array_dtype(const MlxArray& a);
```

- [ ] **Step 5: Extend the shim implementation**

In `mlx-sys/shim/src/array.cc`, add at the end of the `cxx_mlx` namespace (before the closing `}`):

```cpp
size_t array_ndim(const MlxArray& a) {
  return a.ndim();
}

size_t array_size(const MlxArray& a) {
  return a.size();
}

uint8_t array_dtype(const MlxArray& a) {
  return static_cast<uint8_t>(a.dtype().val());
}
```

- [ ] **Step 6: Verify both tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke`
Expected: 2 tests run, 2 passed.

- [ ] **Step 7: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p0): expose ndim/size/dtype on Array via FFI"
```

---

## Task 5: Add eval to FFI (TDD, transforms bridge)

**Files:**
- Create: `mlx-sys/src/bridge/transforms.rs`
- Create: `mlx-sys/shim/include/cxx_mlx_shim/transforms.h`
- Create: `mlx-sys/shim/src/transforms.cc`
- Modify: `mlx-sys/src/lib.rs`, `mlx-sys/src/bridge/mod.rs`, `mlx-sys/build.rs`, `mlx-sys/tests/sys_smoke.rs`

- [ ] **Step 1: Write the failing test**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn zeros_then_eval() {
    let arr = mlx_sys::array::ffi::array_zeros(&[8], FLOAT32);
    mlx_sys::transforms::ffi::eval_one(&arr).expect("eval should succeed");
}
```

- [ ] **Step 2: Verify the new test fails**

Run: `cargo test -p mlx-sys --test sys_smoke zeros_then_eval`
Expected: FAIL with `unresolved import mlx_sys::transforms`.

- [ ] **Step 3: Create the transforms shim header**

Create `mlx-sys/shim/include/cxx_mlx_shim/transforms.h`:

```cpp
#pragma once

#include "rust/cxx.h"

namespace mlx::core {
class array;
}

namespace cxx_mlx {

using MlxArray = mlx::core::array;

void eval_one(const MlxArray& a);

}  // namespace cxx_mlx
```

- [ ] **Step 4: Create the transforms shim implementation**

Create `mlx-sys/shim/src/transforms.cc`:

```cpp
#include "cxx_mlx_shim/transforms.h"

#include "mlx/array.h"
#include "mlx/transforms.h"

namespace cxx_mlx {

void eval_one(const MlxArray& a) {
  // mlx::core::eval takes std::vector<array> by value. The array copy ctor
  // is cheap because internal storage is refcounted.
  mlx::core::eval(std::vector<mlx::core::array>{a});
}

}  // namespace cxx_mlx
```

- [ ] **Step 5: Create the transforms cxx bridge**

Create `mlx-sys/src/bridge/transforms.rs`:

```rust
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/transforms.h");

        // Cross-bridge opaque type alias — both bridges refer to the same
        // C++ type cxx_mlx::MlxArray (= mlx::core::array). cxx 1.0 supports
        // sharing opaque types this way as long as the namespace and
        // underlying C++ type match across both bridges.
        type MlxArray = crate::bridge::array::ffi::MlxArray;

        fn eval_one(a: &MlxArray) -> Result<()>;
    }
}
```

(`Result<()>` tells cxx to wrap any C++ exception as a Rust `Err`.)

- [ ] **Step 6: Register the new bridge module**

In `mlx-sys/src/bridge/mod.rs`:

```rust
pub mod array;
pub mod transforms;
```

In `mlx-sys/src/lib.rs`, add after `pub use bridge::array;`:

```rust
pub use bridge::transforms;
```

- [ ] **Step 7: Wire the transforms shim into `build.rs`**

Replace the existing `cxx_build::bridge(...)` call in `mlx-sys/build.rs` with `bridges` (plural) covering both modules:

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

- [ ] **Step 8: Verify all tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke`
Expected: 3 tests run, 3 passed.

- [ ] **Step 9: Commit**

```bash
git add mlx-sys/build.rs mlx-sys/src/ mlx-sys/shim/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p0): expose eval via FFI transforms bridge"
```

---

## Task 6: Safe `mlx` crate — `Dtype` enum and `Error`

**Files:**
- Create: `mlx/src/dtype.rs`, `mlx/src/error.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1: Create `mlx/src/dtype.rs`**

```rust
//! Dtype mirrors `mlx::core::Dtype::Val` (u8 enum over the FFI boundary).
//!
//! The numeric values must stay in sync with `mlx/dtype.h`. The C++ shim
//! does the round-trip; this enum is the Rust-side mirror.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Bfloat16 = 12,
    Complex64 = 13,
}

impl Dtype {
    pub(crate) fn as_u8(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_u8(v: u8) -> Result<Self, crate::Error> {
        match v {
            0 => Ok(Dtype::Bool),
            1 => Ok(Dtype::Uint8),
            2 => Ok(Dtype::Uint16),
            3 => Ok(Dtype::Uint32),
            4 => Ok(Dtype::Uint64),
            5 => Ok(Dtype::Int8),
            6 => Ok(Dtype::Int16),
            7 => Ok(Dtype::Int32),
            8 => Ok(Dtype::Int64),
            9 => Ok(Dtype::Float16),
            10 => Ok(Dtype::Float32),
            11 => Ok(Dtype::Float64),
            12 => Ok(Dtype::Bfloat16),
            13 => Ok(Dtype::Complex64),
            other => Err(crate::Error::Mlx(format!("unknown Dtype::Val={other}"))),
        }
    }
}
```

NOTE: The numeric values above must match `mlx::core::Dtype::Val` enum order in `mlx/dtype.h`. If the test in Step 5 fails with a `Dtype` mismatch, re-check the order in `/Volumes/Dev/mlx/mlx/dtype.h` lines 14–30 and update both this file and the `FLOAT32` constant in `mlx-sys/tests/sys_smoke.rs`.

- [ ] **Step 2: Create `mlx/src/error.rs`**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX runtime error: {0}")]
    Mlx(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Mlx(e.what().to_owned())
    }
}
```

- [ ] **Step 3: Update `mlx/src/lib.rs` to expose them**

```rust
//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod dtype;
mod error;

pub use dtype::Dtype;
pub use error::{Error, Result};
```

- [ ] **Step 4: Verify the crate still builds**

Run: `MLX_DIR=$HOME/.local/mlx cargo check -p mlx`
Expected: PASS.

- [ ] **Step 5: Verify Dtype numeric values match MLX enum (sync test)**

Append to `mlx-sys/tests/sys_smoke.rs` (the sys-side test is the source of truth for FFI numbers):

```rust
#[test]
fn dtype_float32_constant_matches_mlx() {
    // If this fails, mlx/dtype.h reordered Dtype::Val. Update FLOAT32 here
    // and Dtype enum values in mlx/src/dtype.rs to match.
    let arr = mlx_sys::array::ffi::array_zeros(&[1], FLOAT32);
    assert_eq!(mlx_sys::array::ffi::array_dtype(&arr), FLOAT32);
}
```

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke`
Expected: 4 tests run, 4 passed.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p0): add Dtype enum and Error type to mlx crate"
```

---

## Task 7: Safe `Array` wrapper (TDD)

**Files:**
- Create: `mlx/src/array.rs`
- Create: `mlx/tests/p0_smoke.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1: Write the failing P0 smoke test**

Create `mlx/tests/p0_smoke.rs`:

```rust
use mlx::{Array, Dtype};

#[test]
fn p0_end_to_end() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32);
    assert_eq!(arr.shape(), vec![2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.size(), 6);
    arr.eval().expect("eval should succeed");
}

#[test]
fn empty_shape_is_scalar() {
    let arr = Array::zeros(&[], Dtype::Int32);
    assert_eq!(arr.shape(), Vec::<i32>::new());
    assert_eq!(arr.ndim(), 0);
    assert_eq!(arr.size(), 1);
}
```

- [ ] **Step 2: Verify it fails**

Run: `cargo test -p mlx --test p0_smoke`
Expected: FAIL with `cannot find struct Array` (the type does not exist yet).

- [ ] **Step 3: Create `mlx/src/array.rs`**

```rust
use cxx::UniquePtr;

use crate::{Dtype, Error, Result};

/// An MLX array. Cheap to clone (MLX internally refcounts the storage).
pub struct Array(UniquePtr<mlx_sys::array::ffi::MlxArray>);

impl Array {
    /// Create an array filled with zeros of the given shape and dtype.
    /// The result is lazy — call [`Array::eval`] before reading the data.
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Self {
        Array(mlx_sys::array::ffi::array_zeros(shape, dtype.as_u8()))
    }

    /// The shape of the array. `[]` denotes a scalar.
    pub fn shape(&self) -> Vec<i32> {
        mlx_sys::array::ffi::array_shape(&self.0)
    }

    /// The dtype of the array.
    pub fn dtype(&self) -> Dtype {
        // The shim only ever returns values produced by static_cast<uint8_t>(Dtype::Val),
        // so a missing variant means MLX was upgraded with a new dtype — surface it as a panic
        // (this is a programmer error, not a runtime condition).
        let raw = mlx_sys::array::ffi::array_dtype(&self.0);
        Dtype::from_u8(raw).expect("MLX returned unknown Dtype::Val — mlx-sys/mlx version mismatch")
    }

    pub fn ndim(&self) -> usize {
        mlx_sys::array::ffi::array_ndim(&self.0)
    }

    pub fn size(&self) -> usize {
        mlx_sys::array::ffi::array_size(&self.0)
    }

    /// Force evaluation of the lazy graph backing this array.
    pub fn eval(&self) -> Result<()> {
        mlx_sys::transforms::ffi::eval_one(&self.0).map_err(Error::from)
    }
}
```

NOTE: `Array` does NOT yet implement `Clone`/`Debug` — those land in P1 along with `from_slice`/`item`/`to_vec`. P0 only needs the methods used by the smoke tests.

- [ ] **Step 4: Re-export `Array` from `mlx/src/lib.rs`**

Update to:

```rust
//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod array;
mod dtype;
mod error;

pub use array::Array;
pub use dtype::Dtype;
pub use error::{Error, Result};
```

- [ ] **Step 5: Verify the test passes**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p0_smoke`
Expected: 2 tests run, 2 passed.

If `p0_end_to_end` fails on the `dtype()` assertion, the `Dtype::Float32 = 10` value disagrees with what MLX returned. Re-check `/Volumes/Dev/mlx/mlx/dtype.h` line 14+ and update both `Dtype` and the `FLOAT32` constant in `sys_smoke.rs`.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/array.rs mlx/src/lib.rs mlx/tests/p0_smoke.rs
git commit -m "feat(p0): safe Array wrapper with zeros/shape/dtype/ndim/size/eval"
```

---

## Task 8: Full workspace verification + README polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --workspace`
Expected: All tests pass:
- `mlx-sys`: 4 tests in `sys_smoke`
- `mlx`: 2 tests in `p0_smoke`

- [ ] **Step 2: Run clippy**

Run: `MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

If clippy flags items in `bridge/array.rs` or `bridge/transforms.rs`, those are cxx-generated patterns — silence per-file with `#![allow(clippy::needless_lifetimes)]` (or whichever lint cxx triggers) at the top of those files. Do not silence workspace-wide.

- [ ] **Step 3: Update `README.md` with full P0 quickstart**

Replace the `## Quickstart` section with:

````markdown
## Quickstart

Build MLX once (any prefix you like):

```bash
cd /path/to/mlx
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/mlx
make -j$(sysctl -n hw.ncpu) && make install
export MLX_DIR=$HOME/.local/mlx
```

Then in your project:

```rust
use mlx::{Array, Dtype};

fn main() -> mlx::Result<()> {
    let a = Array::zeros(&[2, 3], Dtype::Float32);
    println!("shape={:?} dtype={:?} size={}", a.shape(), a.dtype(), a.size());
    a.eval()?;
    Ok(())
}
```

## Status

- ✅ P0 — scaffold (zeros + eval + shape)
- ⏳ P1 — Array + core ops
- ⏳ P2 — fast + io + transforms
- ⏳ P3 — quantization + compile + LLM example
````

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(p0): expand README quickstart with build instructions and roadmap"
```

---

## Acceptance Criteria

P0 is done when:

1. `MLX_DIR=$HOME/.local/mlx cargo test --workspace` reports all tests passing (4 in `mlx-sys`, 2 in `mlx`).
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean.
3. The `mlx::Array` type supports `zeros`, `shape`, `dtype`, `ndim`, `size`, `eval`.
4. `MLX_DIR` is the only environment variable needed (no hand-edited paths in any source file).
5. The build fails with a clear `compile_error!` on non-Apple-Silicon-macOS targets.

When all five hold, P0 is complete and ready for code review. P1 planning starts based on what surfaced during P0 implementation.
