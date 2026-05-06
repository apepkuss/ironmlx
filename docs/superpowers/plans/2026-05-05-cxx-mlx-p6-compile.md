# cxx-mlx P6: compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind `mlx::core::compile` so Rust users can JIT-compile arbitrary Rust closures into MLX traced graphs, plus the global `disable/enable/set_compile_mode` controls.

**Architecture:** Three-layer FFI as in P2b–P5: C++ shim adapter → cxx::bridge → Rust safe API. Two new opaque C++ types: `ArrayVec` (bidirectional `std::vector<array>` carrier, since cxx 1.0 disallows `Vec<UniquePtr<T>>`) and `CompiledFn` (RAII handle for `std::function`). Closures cross the boundary via an `extern "Rust"` `CompileCallback` whose `invoke` is called from a C++ lambda capturing `shared_ptr<rust::Box<CompileCallback>>` (so the lambda is `CopyConstructible` for `std::function`).

**Tech Stack:** Rust 2021 + cxx 1.0 + MLX C++ (`include/mlx/compile.h`) + macOS aarch64. Test runner: `cargo test --test p6_compile`. Spec: [docs/superpowers/specs/2026-05-05-cxx-mlx-p6-compile-design.md](../specs/2026-05-05-cxx-mlx-p6-compile-design.md).

---

## File Structure

| File | Responsibility |
|---|---|
| `mlx-sys/shim/include/cxx_mlx_shim/compile.h` (new) | `ArrayVec`, `CompiledFn` declarations + free-function shim |
| `mlx-sys/shim/src/compile.cc` (new) | Implementations: ArrayVec methods, global controls, `compile_with_callback`, `compiled_fn_invoke` |
| `mlx-sys/src/bridge/compile.rs` (new) | cxx::bridge with `extern "Rust" CompileCallback` + extern "C++" surface |
| `mlx-sys/src/bridge/mod.rs` (modify) | `pub mod compile;` |
| `mlx-sys/build.rs` (modify) | Add `src/bridge/compile.rs` to `cxx_build::bridges` and `shim/src/compile.cc` to `.file()` calls |
| `mlx/src/compile.rs` (new) | `CompileMode` enum, `disable_compile`/`enable_compile`/`set_compile_mode`, `CompiledFn` struct, `compile<F>(...)` |
| `mlx/src/lib.rs` (modify) | `pub mod compile;` + selective re-exports |
| `mlx/tests/p6_compile.rs` (new) | 9 integration tests |
| `README.md` (modify) | P6 status line in progress section |

---

## Conventions Recap (don't reinvent)

- **`Option<&Array>` → `*const MlxArray`**: `arr.map_or(std::ptr::null(), |a| a.as_inner() as *const _)` on Rust side; raw pointer + nullptr check on C++ side. Wrap such bridge calls in `unsafe { ... }`.
- **`Array::as_inner()` → `&mlx_sys::array::ffi::MlxArray`** and **`Array::from_inner(UniquePtr<MlxArray>) -> Array`**.
- **All shim functions that throw** must be declared `Result<...>` in the cxx::bridge block (see `mlx-sys/src/bridge/mod.rs` doc comment). Pure getters can stay infallible.
- **Errors**: project uses `mlx::Error` / `mlx::Result`. `cxx::Exception` auto-converts via `impl From<cxx::Exception> for Error` in `mlx/src/error.rs`. Map at the safe API boundary with `.map_err(Error::from)?`.
- **No `self: &T` methods on bridge types**; expose free functions like `array_vec_count(v: &ArrayVec)`.
- **Section comments** in shared files: `// === P6 ... ===`.
- **TDD per step**: write failing test → run (FAIL) → implement → run (PASS) → fmt/clippy/build → commit.
- **After every Rust edit**, the project gate is:
  ```
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```
  Run before each commit. If clippy fires, fix the lint, do not silence it.

---

## Task 1: Skeleton + Global Controls + CompileMode

Set up the new files end-to-end with the simplest possible bridge surface so that Tasks 2 and 3 only add functions rather than wiring infrastructure. Only the global controls + `CompileMode` enum here — no closures, no ArrayVec yet.

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/compile.h`
- Create: `mlx-sys/shim/src/compile.cc`
- Create: `mlx-sys/src/bridge/compile.rs`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/build.rs`
- Create: `mlx/src/compile.rs`
- Modify: `mlx/src/lib.rs`
- Test: `mlx/tests/p6_compile.rs`

- [ ] **Step 1.1: Write the failing test**

Create `mlx/tests/p6_compile.rs`:

```rust
//! Integration tests for mlx::compile — JIT compilation of Rust closures.

use mlx::compile::{disable_compile, enable_compile, set_compile_mode, CompileMode};

#[test]
fn compile_mode_setters() {
    // Round-trip every variant; the calls must not panic and must leave
    // compile in a usable (enabled) state for subsequent tests.
    set_compile_mode(CompileMode::Disabled);
    set_compile_mode(CompileMode::NoSimplify);
    set_compile_mode(CompileMode::NoFuse);
    set_compile_mode(CompileMode::Enabled);
    disable_compile();
    enable_compile();
}
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cargo test --test p6_compile compile_mode_setters`
Expected: FAIL — `unresolved import mlx::compile`.

- [ ] **Step 1.3: Create C++ shim header**

Create `mlx-sys/shim/include/cxx_mlx_shim/compile.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// === P6 global controls ===
void disable_compile();
void enable_compile();
// mode: 0=Disabled, 1=NoSimplify, 2=NoFuse, 3=Enabled.
// Throws std::invalid_argument on out-of-range.
void set_compile_mode(uint8_t mode);

} // namespace cxx_mlx
```

- [ ] **Step 1.4: Create C++ shim implementation**

Create `mlx-sys/shim/src/compile.cc`:

```cpp
#include "cxx_mlx_shim/compile.h"

#include <stdexcept>

namespace cxx_mlx {

// === P6 global controls ===

void disable_compile() {
  mlx::core::disable_compile();
}

void enable_compile() {
  mlx::core::enable_compile();
}

void set_compile_mode(uint8_t mode) {
  using mlx::core::CompileMode;
  switch (mode) {
    case 0:
      mlx::core::set_compile_mode(CompileMode::disabled);
      break;
    case 1:
      mlx::core::set_compile_mode(CompileMode::no_simplify);
      break;
    case 2:
      mlx::core::set_compile_mode(CompileMode::no_fuse);
      break;
    case 3:
      mlx::core::set_compile_mode(CompileMode::enabled);
      break;
    default:
      throw std::invalid_argument(
          "set_compile_mode: invalid CompileMode repr (must be 0..=3)");
  }
}

} // namespace cxx_mlx
```

- [ ] **Step 1.5: Create cxx::bridge module**

Create `mlx-sys/src/bridge/compile.rs`:

```rust
//! Bridge for MLX compile subsystem.
//!
//! P6 Task 1 surface: global controls only. Tasks 2/3 add ArrayVec, the
//! extern "Rust" CompileCallback, CompiledFn, and the compile entry point.

#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        // === P6 global controls ===
        fn disable_compile();
        fn enable_compile();
        fn set_compile_mode(mode: u8) -> Result<()>;
    }
}
```

- [ ] **Step 1.6: Register the new bridge module**

Modify `mlx-sys/src/bridge/mod.rs` — add a single line in the `pub mod` list. Final list (alphabetical-ish, mirroring existing order):

```rust
pub mod array;
pub mod compile;
pub mod fast;
pub mod io;
pub mod quantization;
pub mod random;
pub mod stream;
pub mod transforms;
```

- [ ] **Step 1.7: Wire up build.rs**

Modify `mlx-sys/build.rs` — add the new bridge file and the new `.cc` to the existing `cxx_build::bridges([...])` call. After the change, the `bridges` list and `.file(...)` calls must include `compile`. Final block:

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/compile.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
        "src/bridge/quantization.rs",
        "src/bridge/random.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/compile.cc")
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

- [ ] **Step 1.8: Create the safe Rust API module**

Create `mlx/src/compile.rs`:

```rust
//! MLX `compile()` — JIT-trace Rust closures into MLX graphs for fused
//! execution. Tasks 2/3 add the closure-binding surface; this file currently
//! only exposes the global controls and the `CompileMode` enum.

/// Global compile mode. Mirrors `mlx::core::CompileMode`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Compile is fully disabled; functions run eagerly.
    Disabled = 0,
    /// Compile, but skip the simplify pass.
    NoSimplify = 1,
    /// Compile, but skip kernel fusion.
    NoFuse = 2,
    /// Full compile (default).
    Enabled = 3,
}

/// Globally disable MLX compile. Equivalent to `set_compile_mode(Disabled)`.
pub fn disable_compile() {
    mlx_sys::compile::ffi::disable_compile();
}

/// Globally enable MLX compile. Equivalent to `set_compile_mode(Enabled)`.
pub fn enable_compile() {
    mlx_sys::compile::ffi::enable_compile();
}

/// Set the global compile mode.
pub fn set_compile_mode(mode: CompileMode) {
    // Cast is total: enum is #[repr(u8)] with values 0..=3 → set_compile_mode
    // never returns Err in practice, so unwrap is safe and asserts that.
    mlx_sys::compile::ffi::set_compile_mode(mode as u8)
        .expect("set_compile_mode: bridge accepted in-range CompileMode");
}
```

- [ ] **Step 1.9: Register the module in lib.rs**

Modify `mlx/src/lib.rs` — add `pub mod compile;` near the other `pub mod` entries (after `pub mod ops;`, before `mod ops_impl;` to keep alphabetical-ish public-mod grouping). Final relevant chunk:

```rust
mod broadcast;
pub mod compile;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;
mod stream;
pub mod transforms;
```

(Re-exports are deferred to Task 4.)

- [ ] **Step 1.10: Build & run test to verify it passes**

Run: `cargo build --release && cargo test --test p6_compile compile_mode_setters`
Expected: PASS.

- [ ] **Step 1.11: Format / lint / build**

Run:
```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: all green.

- [ ] **Step 1.12: Commit**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/compile.h \
        mlx-sys/shim/src/compile.cc \
        mlx-sys/src/bridge/compile.rs \
        mlx-sys/src/bridge/mod.rs \
        mlx-sys/build.rs \
        mlx/src/compile.rs \
        mlx/src/lib.rs \
        mlx/tests/p6_compile.rs
git commit -m "feat(p6): scaffold + global compile controls (1 test)"
```

---

## Task 2: ArrayVec Opaque (Bidirectional)

Add the `ArrayVec` opaque type so we can ferry `std::vector<mlx::core::array>` between C++ and Rust in both directions. Element access uses MLX's cheap shared-buffer copy on `get_at`; `take_at` removes the element after moving it out (single-use per slot, like P2c LoadResult).

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/compile.h`
- Modify: `mlx-sys/shim/src/compile.cc`
- Modify: `mlx-sys/src/bridge/compile.rs`
- Modify: `mlx/tests/p6_compile.rs`

- [ ] **Step 2.1: Write the failing test**

Append to `mlx/tests/p6_compile.rs`:

```rust
use mlx::Array;

#[test]
fn array_vec_round_trip() {
    // We don't expose ArrayVec to end users, but we exercise it via the
    // mlx-sys bridge to lock in count/push/get_at/take_at semantics.
    use mlx_sys::compile::ffi::{
        array_vec_count, array_vec_get_at, array_vec_new, array_vec_push, array_vec_take_at,
    };

    let a = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("a");
    let b = Array::from_slice(&[3.0_f32, 4.0, 5.0], &[3]).expect("b");
    let c = Array::from_slice(&[6.0_f32], &[1]).expect("c");

    let mut v = array_vec_new();
    assert_eq!(array_vec_count(&v), 0);

    array_vec_push(v.pin_mut(), a.as_inner());
    array_vec_push(v.pin_mut(), b.as_inner());
    array_vec_push(v.pin_mut(), c.as_inner());
    assert_eq!(array_vec_count(&v), 3);

    // get_at clones (shared buffer). Count is unchanged.
    let got1 = array_vec_get_at(&v, 1).expect("get_at 1");
    let got1 = Array::from_inner(got1);
    let got1_vec: Vec<f32> = got1.to_vec().expect("got1 to_vec");
    assert_eq!(got1_vec, vec![3.0, 4.0, 5.0]);
    assert_eq!(array_vec_count(&v), 3);

    // take_at removes the element. Count drops.
    let taken0 = array_vec_take_at(v.pin_mut(), 0).expect("take 0");
    let taken0 = Array::from_inner(taken0);
    let taken0_vec: Vec<f32> = taken0.to_vec().expect("taken0 to_vec");
    assert_eq!(taken0_vec, vec![1.0, 2.0]);
    assert_eq!(array_vec_count(&v), 2);

    // After taking index 0, the previous index 1 (b) is now at index 0.
    let taken_b = array_vec_take_at(v.pin_mut(), 0).expect("take new 0");
    let taken_b = Array::from_inner(taken_b);
    let taken_b_vec: Vec<f32> = taken_b.to_vec().expect("taken_b");
    assert_eq!(taken_b_vec, vec![3.0, 4.0, 5.0]);
    assert_eq!(array_vec_count(&v), 1);

    // Out-of-range take_at returns Err, not UB.
    assert!(array_vec_take_at(v.pin_mut(), 99).is_err());
    // get_at OOB also returns Err.
    assert!(array_vec_get_at(&v, 99).is_err());
}
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `cargo test --test p6_compile array_vec_round_trip`
Expected: FAIL — `array_vec_new` etc. unresolved.

- [ ] **Step 2.3: Add ArrayVec to the C++ header**

Modify `mlx-sys/shim/include/cxx_mlx_shim/compile.h` — replace the file with:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// === P6 global controls ===
void disable_compile();
void enable_compile();
void set_compile_mode(uint8_t mode);

// === P6 ArrayVec (bidirectional opaque carrier) ===
//
// cxx 1.0 does not support `Vec<UniquePtr<T>>`, so we wrap
// `std::vector<mlx::core::array>` in an opaque struct and expose
// scalar accessors. Used for both C++→Rust (compile callback inputs)
// and Rust→C++ (callback outputs, compiled-fn invoke).
struct ArrayVec {
  std::vector<mlx::core::array> inner;
};

std::unique_ptr<ArrayVec> array_vec_new();
size_t array_vec_count(const ArrayVec& v);

// Returns a clone (shares storage with the element via MLX refcount).
// Throws std::out_of_range if i >= count.
std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i);

// Moves element i out and erases it; subsequent elements shift down.
// Throws std::out_of_range if i >= count.
std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i);

// Appends a copy (cheap MLX refcount).
void array_vec_push(ArrayVec& v, const MlxArray& a);

} // namespace cxx_mlx
```

- [ ] **Step 2.4: Add ArrayVec implementations**

Append to `mlx-sys/shim/src/compile.cc` (after the existing global-controls block):

```cpp
// === P6 ArrayVec ===

std::unique_ptr<ArrayVec> array_vec_new() {
  return std::make_unique<ArrayVec>();
}

size_t array_vec_count(const ArrayVec& v) {
  return v.inner.size();
}

std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i) {
  if (i >= v.inner.size()) {
    throw std::out_of_range("array_vec_get_at: index out of range");
  }
  // Copy ctor of mlx::core::array shares the underlying buffer cheaply.
  return std::make_unique<MlxArray>(v.inner[i]);
}

std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i) {
  if (i >= v.inner.size()) {
    throw std::out_of_range("array_vec_take_at: index out of range");
  }
  auto out = std::make_unique<MlxArray>(std::move(v.inner[i]));
  v.inner.erase(v.inner.begin() + static_cast<std::ptrdiff_t>(i));
  return out;
}

void array_vec_push(ArrayVec& v, const MlxArray& a) {
  v.inner.push_back(a);
}
```

- [ ] **Step 2.5: Add ArrayVec to the cxx::bridge**

Modify `mlx-sys/src/bridge/compile.rs` — replace the `ffi` module with:

```rust
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ArrayVec;

        // === P6 global controls ===
        fn disable_compile();
        fn enable_compile();
        fn set_compile_mode(mode: u8) -> Result<()>;

        // === P6 ArrayVec ===
        fn array_vec_new() -> UniquePtr<ArrayVec>;
        fn array_vec_count(v: &ArrayVec) -> usize;
        fn array_vec_get_at(v: &ArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_take_at(
            v: Pin<&mut ArrayVec>,
            i: usize,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_push(v: Pin<&mut ArrayVec>, a: &MlxArray);
    }
}
```

- [ ] **Step 2.6: Run test to verify it passes**

Run: `cargo build --release && cargo test --test p6_compile array_vec_round_trip`
Expected: PASS.

- [ ] **Step 2.7: Format / lint / build**

Run:
```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: all green.

- [ ] **Step 2.8: Commit**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/compile.h \
        mlx-sys/shim/src/compile.cc \
        mlx-sys/src/bridge/compile.rs \
        mlx/tests/p6_compile.rs
git commit -m "feat(p6): ArrayVec bidirectional opaque (1 test)"
```

---

## Task 3: CompiledFn + compile() + Closure Callback (Core Task)

The hard task. Bind closures across the FFI via `extern "Rust" CompileCallback`, build `compile_with_callback` in the shim using a `shared_ptr<rust::Box<CompileCallback>>` capture (so the std::function lambda is `CopyConstructible`), and expose `CompiledFn::invoke` to replay the traced graph.

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/compile.h`
- Modify: `mlx-sys/shim/src/compile.cc`
- Modify: `mlx-sys/src/bridge/compile.rs`
- Modify: `mlx/src/compile.rs`
- Modify: `mlx/tests/p6_compile.rs`

- [ ] **Step 3.1: Write failing tests for the simple unary path**

Append to `mlx/tests/p6_compile.rs`:

```rust
use mlx::compile::compile;

#[test]
fn compile_simple_unary() {
    enable_compile();
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let x = inputs[0];
            // y = x + 1  (closure must build an MLX graph node, not eval eagerly)
            let one = Array::from_slice(&[1.0_f32], &[1])?;
            let y = x.add(&one)?;
            Ok(vec![y])
        },
        false,
    )
    .expect("compile");

    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("x");
    let outs = f.invoke(&[&x]).expect("invoke");
    assert_eq!(outs.len(), 1);
    let y: Vec<f32> = outs[0].to_vec().expect("to_vec");
    assert_eq!(y, vec![2.0, 3.0, 4.0]);
}

#[test]
fn compile_two_input() {
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            // (a, b) => a * b + a
            let a = inputs[0];
            let b = inputs[1];
            let prod = a.mul(b)?;
            let out = prod.add(a)?;
            Ok(vec![out])
        },
        false,
    )
    .expect("compile");

    let a = Array::from_slice(&[2.0_f32, 3.0], &[2]).expect("a");
    let b = Array::from_slice(&[10.0_f32, 100.0], &[2]).expect("b");
    let outs = f.invoke(&[&a, &b]).expect("invoke");
    let v: Vec<f32> = outs[0].to_vec().expect("v");
    // [2*10 + 2, 3*100 + 3] = [22, 303]
    assert_eq!(v, vec![22.0, 303.0]);
}

#[test]
fn compile_captures_weight() {
    // Closure captures an external Array via Arc-like semantics (Array: Clone).
    let w = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("w");
    let w_for_closure = w.clone();

    let f = compile(
        move |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let x = inputs[0];
            let y = x.mul(&w_for_closure)?;
            Ok(vec![y])
        },
        false,
    )
    .expect("compile");

    let x1 = Array::from_slice(&[1.0_f32, 1.0, 1.0], &[3]).expect("x1");
    let y1: Vec<f32> = f.invoke(&[&x1]).expect("y1")[0].to_vec().expect("v");
    assert_eq!(y1, vec![10.0, 20.0, 30.0]);

    let x2 = Array::from_slice(&[2.0_f32, 2.0, 2.0], &[3]).expect("x2");
    let y2: Vec<f32> = f.invoke(&[&x2]).expect("y2")[0].to_vec().expect("v");
    assert_eq!(y2, vec![20.0, 40.0, 60.0]);
}

#[test]
fn compile_shapeless_reuse() {
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let two = Array::from_slice(&[2.0_f32], &[1])?;
            Ok(vec![inputs[0].mul(&two)?])
        },
        true, // shapeless
    )
    .expect("compile");

    let x1 = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("x1");
    let y1: Vec<f32> = f.invoke(&[&x1]).expect("y1")[0].to_vec().expect("v");
    assert_eq!(y1, vec![2.0, 4.0]);

    // Different shape — shapeless trace should still apply.
    let x2 = Array::from_slice(&[3.0_f32, 4.0, 5.0], &[3]).expect("x2");
    let y2: Vec<f32> = f.invoke(&[&x2]).expect("y2")[0].to_vec().expect("v");
    assert_eq!(y2, vec![6.0, 8.0, 10.0]);
}

#[test]
fn compile_callback_error_propagates() {
    // Closure unconditionally returns Err. The error must surface as a Rust
    // Err from compile() (or from the first invoke, depending on whether MLX
    // traces eagerly) — never panic, never abort.
    let f = compile(
        |_inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            Err(mlx::Error::Mlx("intentional callback failure".into()))
        },
        false,
    );

    let saw_err = match f {
        Err(_) => true,
        Ok(cf) => {
            // If trace was deferred to invoke, the error must surface there.
            let x = Array::from_slice(&[1.0_f32], &[1]).expect("x");
            cf.invoke(&[&x]).is_err()
        }
    };
    assert!(saw_err, "callback Err must propagate as Rust Err");
}

#[test]
fn compile_callback_panic_caught() {
    // A panic in the closure must be caught by cxx and surfaced as Err,
    // not abort the process.
    let f = compile(
        |_inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            panic!("intentional callback panic");
        },
        false,
    );

    let saw_err = match f {
        Err(_) => true,
        Ok(cf) => {
            let x = Array::from_slice(&[1.0_f32], &[1]).expect("x");
            cf.invoke(&[&x]).is_err()
        }
    };
    assert!(saw_err, "callback panic must be caught and surfaced as Err");
}
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `cargo test --test p6_compile`
Expected: tests in 3.1 FAIL — `mlx::compile::compile` unresolved.

- [ ] **Step 3.3: Add `CompiledFn` + closure surface to the C++ header**

Replace the contents of `mlx-sys/shim/include/cxx_mlx_shim/compile.h` with:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// === P6 global controls ===
void disable_compile();
void enable_compile();
void set_compile_mode(uint8_t mode);

// === P6 ArrayVec ===
struct ArrayVec {
  std::vector<mlx::core::array> inner;
};

std::unique_ptr<ArrayVec> array_vec_new();
size_t array_vec_count(const ArrayVec& v);
std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i);
std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i);
void array_vec_push(ArrayVec& v, const MlxArray& a);

// === P6 CompiledFn ===
struct CompiledFn {
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      fn;
};

} // namespace cxx_mlx

// === P6 closure entry points ===
//
// Forward-declared in the cxx::bridge as `extern "Rust" type CompileCallback`,
// so the cxx-generated header provides the actual struct. We just need the
// extern entry points the shim exposes.

namespace cxx_mlx {

// Forward decl: the cxx-generated struct lives in another translation unit.
// We only need a name to refer to it via rust::Box.
struct CompileCallback;

std::unique_ptr<CompiledFn> compile_with_callback(
    rust::Box<CompileCallback> cb, bool shapeless);

std::unique_ptr<ArrayVec> compiled_fn_invoke(
    const CompiledFn& cf, const ArrayVec& inputs);

} // namespace cxx_mlx
```

- [ ] **Step 3.4: Implement compile_with_callback + compiled_fn_invoke**

Append to `mlx-sys/shim/src/compile.cc`:

```cpp
// === P6 closure entry points ===
//
// We must include the cxx-generated header for the bridge module to get the
// declaration of CompileCallback (an extern "Rust" type) and its invoke
// method. cxx_build emits headers under target/cxxbridge, and cxx_build
// arranges include paths so this works.
#include "mlx-sys/src/bridge/compile.rs.h"

namespace cxx_mlx {

std::unique_ptr<CompiledFn> compile_with_callback(
    rust::Box<CompileCallback> cb, bool shapeless) {
  // std::function requires CopyConstructible; rust::Box is move-only.
  // shared_ptr lets the lambda satisfy the requirement.
  auto shared_cb =
      std::make_shared<rust::Box<CompileCallback>>(std::move(cb));

  auto traced = mlx::core::compile(
      [shared_cb](const std::vector<mlx::core::array>& inputs)
          -> std::vector<mlx::core::array> {
        // Wrap inputs into an ArrayVec the Rust callback can read.
        auto in_vec = std::make_unique<ArrayVec>();
        in_vec->inner = inputs;  // copy ctor on each element (cheap refcount).

        // Invoke Rust. cxx generates `invoke` returning UniquePtr<ArrayVec>;
        // a Rust Err / panic surfaces here as a thrown C++ exception, which
        // propagates out of the lambda → MLX trace → compile_with_callback,
        // and ultimately back through cxx as a Rust Err at the boundary.
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

} // namespace cxx_mlx
```

- [ ] **Step 3.5: Add CompileCallback + closure FFI to the bridge**

Replace `mlx-sys/src/bridge/compile.rs` with:

```rust
//! Bridge for MLX compile subsystem.
//!
//! Closure binding via `extern "Rust" CompileCallback`. The shim wraps
//! the Rust callback in a `shared_ptr<rust::Box<CompileCallback>>` so that
//! the lambda passed to `mlx::core::compile` is `CopyConstructible` (a
//! requirement of `std::function`).

use crate::bridge::array::ffi::MlxArray;
use cxx::UniquePtr;

/// Wraps a user-provided Rust closure for use as an MLX trace target.
///
/// The closure runs once per trace (more if `shapeless=false` and the
/// shape changes) and must build an MLX graph from the inputs — every
/// op called on the inputs / capture variables is recorded by MLX.
///
/// Returning `Err` (or panicking) from the closure surfaces as a Rust
/// `Err` from `compile()` or `CompiledFn::invoke()`. cxx auto-translates
/// panics via `catch_unwind`.
pub struct CompileCallback {
    f: Box<
        dyn Fn(&[&mlx_array_ref::Array]) -> mlx_array_ref::Result<Vec<mlx_array_ref::Array>>
            + Send
            + Sync,
    >,
}

// Re-export the safe-API types the closure works with via a tiny wrapper
// module so `mlx-sys` does not depend on the `mlx` crate (which depends on
// `mlx-sys`). See the implementation in `mlx::compile`. To avoid that cycle
// here, we declare a thin trait-friendly facade type alias.
//
// IMPORTANT: This bridge crate does NOT publicly expose CompileCallback.
// The `mlx` crate constructs it via the helper functions defined below.
mod mlx_array_ref {
    // Mirror of the safe-API surface CompileCallback uses. The `mlx` crate
    // re-implements `Array` and `Result` over these primitives; here we use
    // the same low-level `MlxArray` (cxx opaque) plus a duplicated Result
    // alias, to keep the bridge crate self-contained.
    pub use crate::bridge::array::ffi::MlxArray;

    pub type Array = MlxArray;

    #[derive(Debug)]
    pub struct CallbackError(pub String);

    pub type Result<T> = core::result::Result<T, CallbackError>;
}

impl CompileCallback {
    fn invoke(
        &self,
        inputs: &ffi::ArrayVec,
    ) -> Result<UniquePtr<ffi::ArrayVec>, mlx_array_ref::CallbackError> {
        let n = ffi::array_vec_count(inputs);
        // Materialize a Vec<UniquePtr<MlxArray>> via get_at, which clones
        // (shared buffer). Then borrow each as &MlxArray for the closure.
        let owned: Vec<UniquePtr<MlxArray>> = (0..n)
            .map(|i| ffi::array_vec_get_at(inputs, i))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| mlx_array_ref::CallbackError(e.to_string()))?;
        let refs: Vec<&MlxArray> = owned.iter().map(|p| &**p).collect();

        let outputs = (self.f)(&refs)?;

        let mut out_vec = ffi::array_vec_new();
        for a in &outputs {
            ffi::array_vec_push(out_vec.pin_mut(), a);
        }
        Ok(out_vec)
    }
}

/// Construct a `CompileCallback` from a boxed Rust closure. Used by the
/// `mlx` crate to build the callback before passing it across cxx.
pub fn make_callback(
    f: Box<
        dyn Fn(&[&MlxArray]) -> Result<Vec<UniquePtr<MlxArray>>, String> + Send + Sync,
    >,
) -> Box<CompileCallback> {
    // Adapt String→CallbackError, and UniquePtr<MlxArray>→Array (alias).
    Box::new(CompileCallback {
        f: Box::new(move |refs: &[&mlx_array_ref::Array]| -> mlx_array_ref::Result<
            Vec<mlx_array_ref::Array>,
        > {
            let outs = (f)(refs).map_err(mlx_array_ref::CallbackError)?;
            // Move each UniquePtr<MlxArray> into a Vec<MlxArray> by
            // dereferencing — the bridge `array_vec_push(&MlxArray)` will
            // copy via MLX's cheap refcount, so we do not need to give up
            // ownership of the cxx UniquePtr beyond this scope.
            let mut v: Vec<mlx_array_ref::Array> = Vec::with_capacity(outs.len());
            for u in outs.iter() {
                // SAFETY: u is non-null (cxx invariant for UniquePtr returned
                // by Ok-producing C++ code) and we only deref for the
                // duration of this function before Vec is consumed by
                // array_vec_push (which copies).
                v.push((**u).clone());
            }
            Ok(v)
        }),
    })
}

#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    extern "Rust" {
        type CompileCallback;
        fn invoke(
            self: &CompileCallback,
            inputs: &ArrayVec,
        ) -> Result<UniquePtr<ArrayVec>>;
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/compile.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type ArrayVec;
        type CompiledFn;

        // === Global controls ===
        fn disable_compile();
        fn enable_compile();
        fn set_compile_mode(mode: u8) -> Result<()>;

        // === ArrayVec ===
        fn array_vec_new() -> UniquePtr<ArrayVec>;
        fn array_vec_count(v: &ArrayVec) -> usize;
        fn array_vec_get_at(v: &ArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_take_at(
            v: Pin<&mut ArrayVec>,
            i: usize,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_vec_push(v: Pin<&mut ArrayVec>, a: &MlxArray);

        // === CompiledFn ===
        fn compile_with_callback(
            cb: Box<CompileCallback>,
            shapeless: bool,
        ) -> Result<UniquePtr<CompiledFn>>;
        fn compiled_fn_invoke(
            cf: &CompiledFn,
            inputs: &ArrayVec,
        ) -> Result<UniquePtr<ArrayVec>>;
    }
}
```

> **Note on `mlx_array_ref`**: this is a deliberate stand-in to keep `mlx-sys` independent of `mlx`. `Array` here is the bridge-level `MlxArray`, not `mlx::Array`. The `mlx` crate's safe API converts between the two via `Array::from_inner` / `Array::as_inner` / `Array::clone()`.

> **Note on `MlxArray::clone()`**: `mlx::core::array` is copy-constructible (cheap refcount). The cxx-generated `MlxArray` type may not implement `Clone` automatically. If `clippy` or the compiler complains that `MlxArray` is not `Clone`, replace the `(**u).clone()` line with a small extern "C++" helper `mlx_array_clone(&MlxArray) -> UniquePtr<MlxArray>` and `array_vec_push_unique(Pin<&mut ArrayVec>, UniquePtr<MlxArray>)`. The intent is the same: append a refcount-shared copy.

- [ ] **Step 3.6: Wire the safe API**

Append to `mlx/src/compile.rs`:

```rust
use crate::{Array, Error, Result};
use cxx::UniquePtr;

/// A compiled MLX function. Cheap to clone? No — currently single-instance.
/// Drop releases the underlying `std::function`.
pub struct CompiledFn {
    inner: UniquePtr<mlx_sys::compile::ffi::CompiledFn>,
}

impl CompiledFn {
    /// Run the compiled graph on the given inputs.
    pub fn invoke(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        let mut in_vec = mlx_sys::compile::ffi::array_vec_new();
        for a in inputs {
            mlx_sys::compile::ffi::array_vec_push(in_vec.pin_mut(), a.as_inner());
        }

        let mut out_vec = mlx_sys::compile::ffi::compiled_fn_invoke(&self.inner, &in_vec)
            .map_err(Error::from)?;

        let n = mlx_sys::compile::ffi::array_vec_count(&out_vec);
        let mut outs: Vec<Array> = Vec::with_capacity(n);
        // Drain front-to-back: each take_at removes index 0.
        for _ in 0..n {
            let a = mlx_sys::compile::ffi::array_vec_take_at(out_vec.pin_mut(), 0)
                .map_err(Error::from)?;
            outs.push(Array::from_inner(a));
        }
        Ok(outs)
    }
}

/// JIT-compile a Rust closure into an MLX traced graph.
///
/// The closure is invoked once at trace time (and again on shape changes
/// when `shapeless=false`). Every MLX op the closure runs is recorded;
/// subsequent calls to [`CompiledFn::invoke`] replay the optimized graph
/// without re-running the closure.
///
/// The closure is `Send + Sync + 'static` because MLX may trace lazily and
/// from any thread. Returning `Err` from the closure (or panicking) yields
/// a Rust `Err` from `compile()` or `invoke()`.
pub fn compile<F>(f: F, shapeless: bool) -> Result<CompiledFn>
where
    F: Fn(&[&Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
{
    // Wrap the user closure in the bridge-level callback shape:
    // - bridge sees &[&MlxArray] → we map to &[&Array]
    // - user returns Vec<Array>  → we map to Vec<UniquePtr<MlxArray>>
    let bridge_fn: Box<
        dyn Fn(
                &[&mlx_sys::array::ffi::MlxArray],
            ) -> std::result::Result<Vec<UniquePtr<mlx_sys::array::ffi::MlxArray>>, String>
            + Send
            + Sync,
    > = Box::new(move |refs: &[&mlx_sys::array::ffi::MlxArray]| {
        // Borrow each &MlxArray as a temporary Array WITHOUT taking ownership.
        // Array's safe API uses UniquePtr internally; for the closure we
        // construct ephemeral Arrays that share the buffer via clone.
        let temp_arrays: Vec<Array> = refs
            .iter()
            .map(|m| {
                // Clone the MlxArray into a fresh UniquePtr so we can wrap it.
                // mlx::core::array is refcounted, so this is cheap.
                let cloned: UniquePtr<mlx_sys::array::ffi::MlxArray> =
                    mlx_sys::array::ffi::array_clone(m);
                Array::from_inner(cloned)
            })
            .collect();
        let borrows: Vec<&Array> = temp_arrays.iter().collect();

        let outs = f(&borrows).map_err(|e| e.to_string())?;

        // Convert Vec<Array> → Vec<UniquePtr<MlxArray>>. We move ownership
        // out of each Array via Array::into_inner.
        let mut result: Vec<UniquePtr<mlx_sys::array::ffi::MlxArray>> =
            Vec::with_capacity(outs.len());
        for a in outs {
            result.push(a.into_inner());
        }
        Ok(result)
    });

    let cb = mlx_sys::compile::make_callback(bridge_fn);
    let inner = mlx_sys::compile::ffi::compile_with_callback(cb, shapeless)
        .map_err(Error::from)?;
    Ok(CompiledFn { inner })
}
```

- [ ] **Step 3.7: Provide the helper accessors `array_clone` and `Array::into_inner`**

The closure adapter requires:

1. **`mlx_sys::array::ffi::array_clone(&MlxArray) -> UniquePtr<MlxArray>`** — a cheap-clone primitive (MLX refcount).
2. **`Array::into_inner(self) -> UniquePtr<MlxArray>`** — consume an Array and return the inner cxx ptr.

Check whether they already exist:

```bash
grep -n "array_clone\|fn into_inner\|fn from_inner" mlx-sys/src/bridge/array.rs mlx-sys/shim/include/cxx_mlx_shim/array.h mlx-sys/shim/src/array.cc mlx/src/array.rs
```

If `array_clone` is missing, add it:

`mlx-sys/shim/include/cxx_mlx_shim/array.h` — append in the `cxx_mlx` namespace:
```cpp
std::unique_ptr<MlxArray> array_clone(const MlxArray& a);
```

`mlx-sys/shim/src/array.cc` — append:
```cpp
std::unique_ptr<MlxArray> array_clone(const MlxArray& a) {
  return std::make_unique<MlxArray>(a);  // refcount-shared copy.
}
```

`mlx-sys/src/bridge/array.rs` — add inside the existing `unsafe extern "C++"` block:
```rust
fn array_clone(a: &MlxArray) -> UniquePtr<MlxArray>;
```

If `Array::into_inner` is missing, add it to `mlx/src/array.rs`:
```rust
impl Array {
    pub(crate) fn into_inner(self) -> cxx::UniquePtr<mlx_sys::array::ffi::MlxArray> {
        self.0
    }
}
```

- [ ] **Step 3.8: Run the tests**

Run: `cargo build --release && cargo test --test p6_compile`
Expected: all six new tests PASS, plus the two from earlier tasks.

If `compile_callback_panic_caught` aborts the process instead of returning `Err`, the `extern "Rust"` declaration is missing `Result` — re-check the bridge: the `invoke` method must be declared `-> Result<UniquePtr<ArrayVec>>` so cxx wraps panics.

- [ ] **Step 3.9: Format / lint / build**

Run:
```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: all green.

- [ ] **Step 3.10: Commit**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/compile.h \
        mlx-sys/shim/src/compile.cc \
        mlx-sys/src/bridge/compile.rs \
        mlx-sys/shim/include/cxx_mlx_shim/array.h \
        mlx-sys/shim/src/array.cc \
        mlx-sys/src/bridge/array.rs \
        mlx/src/compile.rs \
        mlx/src/array.rs \
        mlx/tests/p6_compile.rs
git commit -m "feat(p6): compile() + CompiledFn::invoke via extern Rust callback (6 tests)"
```

---

## Task 4: Re-export + README + Final Verify

Polish — wire the public re-exports at the crate root, add the P6 line to the README progress section, and run the full test suite.

**Files:**
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/tests/p6_compile.rs`
- Modify: `README.md`

- [ ] **Step 4.1: Write the failing re-export test**

Append to `mlx/tests/p6_compile.rs`:

```rust
#[test]
fn top_level_re_exports_work() {
    // Same as compile_simple_unary, but reaches every symbol via the
    // crate root. If this compiles AND passes, re-exports are correct.
    use mlx::{compile, CompileMode, CompiledFn};

    set_compile_mode(CompileMode::Enabled);

    let f: CompiledFn = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let one = Array::from_slice(&[1.0_f32], &[1])?;
            Ok(vec![inputs[0].add(&one)?])
        },
        false,
    )
    .expect("compile via root");

    let x = Array::from_slice(&[10.0_f32], &[1]).expect("x");
    let v: Vec<f32> = f.invoke(&[&x]).expect("invoke")[0].to_vec().expect("v");
    assert_eq!(v, vec![11.0]);
}
```

(Note: `set_compile_mode` and `Array` are already in scope from previous `use` statements in the file.)

- [ ] **Step 4.2: Run test to verify it fails**

Run: `cargo test --test p6_compile top_level_re_exports_work`
Expected: FAIL — `unresolved import mlx::compile` / `mlx::CompiledFn` / `mlx::CompileMode`.

- [ ] **Step 4.3: Add re-exports**

Modify `mlx/src/lib.rs` — after the `pub mod compile;` declaration and the random re-export block, append:

```rust
pub use compile::{compile, disable_compile, enable_compile, set_compile_mode, CompileMode, CompiledFn};
```

> Note: this re-exports the *function* `compile` at the crate root alongside the *module* `compile`. That is allowed in Rust — they live in different namespaces (the value namespace vs. the type namespace).

- [ ] **Step 4.4: Run test to verify it passes**

Run: `cargo test --test p6_compile top_level_re_exports_work`
Expected: PASS.

- [ ] **Step 4.5: Run the full P6 test suite**

Run: `cargo test --test p6_compile`
Expected: 9 tests PASS (`compile_mode_setters`, `array_vec_round_trip`, `compile_simple_unary`, `compile_two_input`, `compile_captures_weight`, `compile_shapeless_reuse`, `compile_callback_error_propagates`, `compile_callback_panic_caught`, `top_level_re_exports_work`).

- [ ] **Step 4.6: Run the full workspace test suite (regression check)**

Run: `cargo test --release`
Expected: every previously-passing test in the workspace still passes (P0–P5).

- [ ] **Step 4.7: Update README progress section**

Modify `README.md` — change line 5 (the **Status:** line) and append a new bullet at the end of the progress list (after the P5 line, currently line 219).

Replace the **Status:** line with:
```
**Status:** 🎉 **P6 complete** — `mlx::compile` 闭包 JIT 绑定 (`compile()` + `CompiledFn::invoke` + global controls). 用户可把任意 Rust 闭包传给 MLX 进行图追踪 + 融合.
```

Append after the P5 line:
```
- ✅ **P6** — `compile` (closure JIT via extern "Rust" callback + ArrayVec opaque + CompiledFn) — 9 integration tests
```

- [ ] **Step 4.8: Format / lint / build**

Run:
```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```
Expected: all green.

- [ ] **Step 4.9: Commit**

```bash
git add mlx/src/lib.rs mlx/tests/p6_compile.rs README.md
git commit -m "feat(p6): re-export + README progress (1 test)"
```

---

## Verification Checklist

After Task 4 commits, the branch should satisfy:

| Item | Command | Expected |
|---|---|---|
| 9 P6 tests | `cargo test --test p6_compile` | 9 passed |
| All workspace tests | `cargo test --release` | every prior phase still passes |
| Format clean | `cargo +nightly fmt --all -- --check` | no diff |
| Lint clean | `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | no warnings |
| Public surface | `cargo doc --no-deps -p mlx` | `mlx::compile`, `mlx::CompiledFn`, `mlx::CompileMode` documented |
| Spec coverage | grep this plan vs. spec sections 1–7 | every API + test from spec maps to a task |
