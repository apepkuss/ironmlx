# cxx-mlx P1a (Array Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Element` trait + 10 dtypes (incl. `half::f16`/`bf16`), `Clone`/`Debug`/`Send` for `Array`, `from_slice<T>`/`item<T>`/`to_vec<T>` data I/O, expanded `Error` variants, `SmallVec` shape, and the shim `Result`-wrapping rule with P0 backfill.

**Architecture:** Element trait carries per-type FFI dispatch (so `Array::from_slice<T>` is a 1-line delegation). bool bridges through `&[u8]`; `half::f16`/`bf16` bridge through `&[u16]` via `repr(transparent)` reinterpret. shape() returns `SmallVec<[i32; 8]>` to be zero-alloc on ≤ 8 dims (covers 99% of inference tensors). `Array: Send` (matches std::shared_ptr semantics), explicitly `!Sync` because MLX const methods mutate `array_desc_->status` without synchronization.

**Tech Stack:** Rust 1.94+, cxx 1.0, MLX C++ 0.32 (already at `$MLX_DIR`), C++20. New deps: `half = "2"`, `smallvec = "1"`, `static_assertions = "1"` (dev-only).

**Branch:** Work on `p1-ops` (already created off master). MLX install at `$HOME/.local/mlx`; export `MLX_DIR=$HOME/.local/mlx` for every cargo invocation.

---

## File Structure

**New files:**

- `mlx/src/element.rs` — `Element` trait + sealed pattern + 10 type impls (each impl carries the per-dtype FFI dispatch via 3 trait methods)
- `mlx/tests/p1a_io.rs` — `from_slice` / `item` / `to_vec` round-trips, error cases, implicit eval
- `mlx/tests/p1a_array.rs` — `Clone` / `Debug` / `Send` / shape-as-SmallVec
- `mlx/tests/p1a_thread_safety.rs` — `assert_send::<Array>()` + `!Sync` static assertion

**Modified files:**

- `mlx/Cargo.toml` — add `half = "2"`, `smallvec = "1"`, `static_assertions = "1"` (dev)
- `mlx/src/lib.rs` — `mod element; pub use element::Element;`
- `mlx/src/array.rs` — `impl Clone`, `impl Debug`, `Array::shape() -> SmallVec`, `shape_at`, `Array::zeros -> Result`, `Array::from_slice<T>`, `Array::item<T>`, `Array::to_vec<T>`, `unsafe impl Send`
- `mlx/src/dtype.rs` — keep as is (no changes)
- `mlx/src/error.rs` — add `DtypeMismatch`, `ShapeMismatch`, `BroadcastMismatch` variants
- `mlx/tests/p0_smoke.rs` — update `arr.shape() == vec![...]` to `arr.shape().as_slice() == &[...]`; update `Array::zeros` calls to `Array::zeros(...)?` or `.expect(...)`
- `mlx-sys/src/bridge/array.rs` — `array_zeros` returns `Result<UniquePtr<MlxArray>>`; add `array_clone`, `array_is_available`; add 30 `array_from_<T>` / `array_item_<T>` / `array_to_vec_<T>` functions (10 dtypes × 3 ops)
- `mlx-sys/src/bridge/mod.rs` — top-of-file comment fixing the "凡可 throw,必 Result" rule
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — add new function declarations
- `mlx-sys/shim/src/array.cc` — add 2 endpoint `static_assert`s (bool_=0, complex64=13); change `array_zeros` to noexcept-false (cxx will catch); add 33 new shim function impls
- `mlx-sys/tests/sys_smoke.rs` — `array_zeros` calls now need `.expect()` (it returns `Result`)
- `README.md` — add "Threading" section explaining `Array: Send + !Sync`; refresh quickstart code if signatures changed

---

## Task 1: Add new dependencies

**Files:**
- Modify: `mlx/Cargo.toml`

- [ ] **Step 1: Add `half`, `smallvec`, and `static_assertions` (dev) to `mlx/Cargo.toml`**

Replace the `[dependencies]` and `[dev-dependencies]` sections with:

```toml
[dependencies]
mlx-sys = { path = "../mlx-sys", version = "0.0.1" }
cxx.workspace = true
thiserror.workspace = true
half = "2"
smallvec = "1"

[dev-dependencies]
static_assertions = "1"
```

(If there is no existing `[dev-dependencies]` section, add one.)

- [ ] **Step 2: Verify the workspace still builds**

Run: `MLX_DIR=$HOME/.local/mlx cargo check --workspace`
Expected: PASS, possibly with new "unused dependency" warnings (ignore — they go away after Tasks 3-12).

- [ ] **Step 3: Commit**

```bash
git add mlx/Cargo.toml
git commit -m "feat(p1a): add half, smallvec, and static_assertions deps to mlx crate"
```

---

## Task 2: Expand `Error` enum

**Files:**
- Modify: `mlx/src/error.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/src/error.rs` (creating a `#[cfg(test)] mod tests` block at the end if not present):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;

    #[test]
    fn dtype_mismatch_displays() {
        let e = Error::DtypeMismatch { expected: Dtype::Float32, actual: Dtype::Int32 };
        assert_eq!(e.to_string(), "dtype mismatch: expected Float32, got Int32");
    }

    #[test]
    fn shape_mismatch_displays() {
        let e = Error::ShapeMismatch { expected: vec![2, 3], actual: vec![6] };
        assert_eq!(e.to_string(), "shape mismatch: expected [2, 3], got [6]");
    }

    #[test]
    fn broadcast_mismatch_displays() {
        let e = Error::BroadcastMismatch { lhs: vec![3, 1], rhs: vec![2, 4] };
        assert_eq!(e.to_string(), "broadcast mismatch: lhs [3, 1] vs rhs [2, 4]");
    }
}
```

- [ ] **Step 2: Verify tests fail**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx error::tests 2>&1 | tail -10`
Expected: FAIL with "no variant named `DtypeMismatch`" (and similar).

- [ ] **Step 3: Add the three variants**

Replace the `Error` enum in `mlx/src/error.rs` with:

```rust
use thiserror::Error;
use crate::Dtype;

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

pub type Result<T> = std::result::Result<T, Error>;

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Mlx(e.what().to_owned())
    }
}
```

(Note: this adds `use crate::Dtype;` at the top — `Dtype` is now referenced by `DtypeMismatch`.)

- [ ] **Step 4: Verify tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx error::tests 2>&1 | tail -10`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mlx/src/error.rs
git commit -m "feat(p1a): add DtypeMismatch, ShapeMismatch, BroadcastMismatch error variants"
```

---

## Task 3: Add endpoint `static_assert`s in shim

**Files:**
- Modify: `mlx-sys/shim/src/array.cc`

- [ ] **Step 1: Add the two endpoint static_asserts**

In `mlx-sys/shim/src/array.cc`, find the existing block:

```cpp
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::float32) == 10,
              "Dtype::Val::float32 ordinal changed; update FLOAT32 in sys_smoke.rs and Dtype enum");
```

Replace it with three asserts (the existing one plus the two endpoints):

```cpp
// Endpoint static_asserts on mlx::core::Dtype::Val. If MLX inserts a new
// dtype at any position, at least one endpoint shifts and we fail fast at
// the C++ build step before the Rust Dtype mirror has a chance to drift.
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::bool_) == 0,
              "Dtype::Val::bool_ ordinal changed; update Dtype enum in mlx/src/dtype.rs");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::float32) == 10,
              "Dtype::Val::float32 ordinal changed; update FLOAT32 in sys_smoke.rs and Dtype enum");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::complex64) == 13,
              "Dtype::Val::complex64 ordinal changed; update Dtype enum in mlx/src/dtype.rs");
```

- [ ] **Step 2: Verify the C++ shim still compiles**

Run: `MLX_DIR=$HOME/.local/mlx cargo build -p mlx-sys 2>&1 | tail -5`
Expected: PASS (the asserts are true for MLX 0.32).

- [ ] **Step 3: Commit**

```bash
git add mlx-sys/shim/src/array.cc
git commit -m "feat(p1a): add bool_=0 and complex64=13 endpoint static_asserts in shim"
```

---

## Task 4: `Element` trait + 10 type impls

**Files:**
- Create: `mlx/src/element.rs`
- Modify: `mlx/src/lib.rs`

This task only sets up the `Element` trait skeleton — the FFI dispatch methods on `Element` are stubbed to `unimplemented!()` because the underlying shim functions don't exist yet (Tasks 6-11 fill them in).

- [ ] **Step 1: Create `mlx/src/element.rs` with trait + 10 impls (stubs)**

```rust
//! `Element` is the type-class for Rust types that map to MLX dtypes.
//!
//! Sealed (impossible to impl outside this crate) so that downstream code
//! cannot construct an arbitrary `T -> Dtype` mapping that would violate
//! FFI type safety.

use crate::{Array, Dtype, Result};

mod sealed {
    pub trait Sealed {}
}

pub trait Element: sealed::Sealed + Copy + Send + 'static {
    const DTYPE: Dtype;

    /// Construct a new array from a slice of `Self`.
    /// Implementation is per-dtype because cxx FFI is monomorphized on T.
    #[doc(hidden)]
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array>;

    /// Read all elements out as a `Vec<Self>`. Implicitly evals if needed.
    #[doc(hidden)]
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>>;

    /// Read the single scalar element. Caller already verified `arr.size() == 1`.
    #[doc(hidden)]
    fn array_item(arr: &Array) -> Result<Self>;
}

macro_rules! element_stub {
    ($T:ty, $dt:expr) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(_slice: &[Self], _shape: &[i32]) -> Result<Array> {
                unimplemented!("filled in by Task 9")
            }
            fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> {
                unimplemented!("filled in by Task 11")
            }
            fn array_item(_arr: &Array) -> Result<Self> {
                unimplemented!("filled in by Task 10")
            }
        }
    };
}

element_stub!(bool, Dtype::Bool);
element_stub!(u8, Dtype::Uint8);
element_stub!(i8, Dtype::Int8);
element_stub!(i16, Dtype::Int16);
element_stub!(i32, Dtype::Int32);
element_stub!(i64, Dtype::Int64);
element_stub!(half::f16, Dtype::Float16);
element_stub!(half::bf16, Dtype::Bfloat16);
element_stub!(f32, Dtype::Float32);
element_stub!(f64, Dtype::Float64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_const_matches_for_each_element() {
        // Compile-time check that every Element type has the right DTYPE constant.
        assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
        assert_eq!(<u8 as Element>::DTYPE, Dtype::Uint8);
        assert_eq!(<i8 as Element>::DTYPE, Dtype::Int8);
        assert_eq!(<i16 as Element>::DTYPE, Dtype::Int16);
        assert_eq!(<i32 as Element>::DTYPE, Dtype::Int32);
        assert_eq!(<i64 as Element>::DTYPE, Dtype::Int64);
        assert_eq!(<half::f16 as Element>::DTYPE, Dtype::Float16);
        assert_eq!(<half::bf16 as Element>::DTYPE, Dtype::Bfloat16);
        assert_eq!(<f32 as Element>::DTYPE, Dtype::Float32);
        assert_eq!(<f64 as Element>::DTYPE, Dtype::Float64);
    }
}
```

- [ ] **Step 2: Wire up in `mlx/src/lib.rs`**

Update to:

```rust
//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod array;
mod dtype;
mod element;
mod error;

pub use array::Array;
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
```

- [ ] **Step 3: Verify cargo check + element tests**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx element::tests 2>&1 | tail -15`
Expected: 1 test passes (`dtype_const_matches_for_each_element`).

- [ ] **Step 4: Commit**

```bash
git add mlx/src/element.rs mlx/src/lib.rs
git commit -m "feat(p1a): add Element trait with sealed pattern + 10 type stubs"
```

---

## Task 5: Backfill — `array_zeros` returns `Result`

This is a regression to P0 enforcing the new shim Result-wrapping rule. It cascades through the bridge, the safe wrapper, and updates two test files plus the README.

**Files:**
- Modify: `mlx-sys/src/bridge/mod.rs` — top-of-file rule comment
- Modify: `mlx-sys/src/bridge/array.rs` — change return type
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h` — change return type
- Modify: `mlx-sys/shim/src/array.cc` — no signature change in source (cxx auto-wraps), but document
- Modify: `mlx-sys/tests/sys_smoke.rs` — `.expect()` on `array_zeros` calls
- Modify: `mlx/src/array.rs` — `Array::zeros` returns `Result<Self>`
- Modify: `mlx/tests/p0_smoke.rs` — `.expect()` on `Array::zeros` calls
- Modify: `README.md` — quickstart now uses `?`

- [ ] **Step 1: Add the rule comment to `mlx-sys/src/bridge/mod.rs`**

Replace the file with:

```rust
//! Each MLX C++ subsystem gets its own bridge module.
//!
//! **Rule (P1a onward):** Any shim function that can throw a C++ exception
//! MUST be declared `Result<T>` in its `#[cxx::bridge]` block. cxx wraps
//! the throw as `cxx::Exception`, which our `From<cxx::Exception> for Error`
//! impl converts to `Error::Mlx(String)`. Without this, a thrown exception
//! propagates through a non-`Result` cxx function as `std::terminate` —
//! the process aborts instead of yielding a recoverable Rust error.
//!
//! Pure getters (no throw paths) may stay as plain return types.

pub mod array;
pub mod transforms;
```

- [ ] **Step 2: Change `array_zeros` signature in the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, change:

```rust
        fn array_zeros(shape: &[i32], dtype: u8) -> UniquePtr<MlxArray>;
```

to:

```rust
        fn array_zeros(shape: &[i32], dtype: u8) -> Result<UniquePtr<MlxArray>>;
```

(Leave all other functions unchanged.)

- [ ] **Step 3: Update the shim header signature**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, the existing line:

```cpp
std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype);
```

stays the same (cxx auto-wraps thrown exceptions on the C++ side regardless of header signature). No change needed in the shim header.

- [ ] **Step 4: Update sys-side smoke tests to use `.expect`**

In `mlx-sys/tests/sys_smoke.rs`, find every call to `ffi::array_zeros(...)` and change to `ffi::array_zeros(...).expect("zeros should succeed")`. There are 4 such calls (one per test). After the changes, all 4 tests look like:

```rust
#[test]
fn zeros_then_read_shape() {
    let arr = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros should succeed");
    let shape = ffi::array_shape(&arr);
    assert_eq!(shape, vec![2, 3]);
}

#[test]
fn zeros_scalar_has_empty_shape() {
    let arr = ffi::array_zeros(&[], FLOAT32).expect("zeros should succeed");
    assert_eq!(ffi::array_shape(&arr), Vec::<i32>::new());
}

#[test]
fn zeros_metadata() {
    let arr = ffi::array_zeros(&[2, 3, 4], FLOAT32).expect("zeros should succeed");
    assert_eq!(ffi::array_ndim(&arr), 3);
    assert_eq!(ffi::array_size(&arr), 24);
    assert_eq!(ffi::array_dtype(&arr), FLOAT32);
}

#[test]
fn zeros_then_eval() {
    let arr = mlx_sys::array::ffi::array_zeros(&[8], FLOAT32).expect("zeros should succeed");
    mlx_sys::transforms::ffi::eval_one(&arr).expect("eval should succeed");
}
```

- [ ] **Step 5: Verify sys-side tests still pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 6: Update `mlx::Array::zeros` to return `Result`**

In `mlx/src/array.rs`, replace the `zeros` method:

```rust
impl Array {
    /// Create an array filled with zeros of the given shape and dtype.
    /// The result is lazy — call [`Array::eval`] before reading the data.
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Self> {
        let inner = mlx_sys::array::ffi::array_zeros(shape, dtype.as_u8())
            .map_err(Error::from)?;
        Ok(Array(inner))
    }
    // ... other methods unchanged for now ...
}
```

- [ ] **Step 7: Update `mlx/tests/p0_smoke.rs` to use `?` or `.expect`**

In `mlx/tests/p0_smoke.rs`, change to:

```rust
use mlx::{Array, Dtype};

#[test]
fn p0_end_to_end() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros should succeed");
    assert_eq!(arr.shape(), vec![2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.size(), 6);
    arr.eval().expect("eval should succeed");
}

#[test]
fn empty_shape_is_scalar() {
    let arr = Array::zeros(&[], Dtype::Int32).expect("zeros should succeed");
    assert_eq!(arr.shape(), Vec::<i32>::new());
    assert_eq!(arr.ndim(), 0);
    assert_eq!(arr.size(), 1);
}
```

(NB: the `arr.shape() == vec![...]` assertions are STILL using `Vec<i32>` here — that gets fixed in Task 8 when we change shape() to SmallVec. Don't touch them in this task.)

- [ ] **Step 8: Update README quickstart code**

In `README.md`, change the quickstart example to:

```rust
use mlx::{Array, Dtype};

fn main() -> mlx::Result<()> {
    let a = Array::zeros(&[2, 3], Dtype::Float32)?;
    println!("shape={:?} dtype={:?} size={}", a.shape(), a.dtype(), a.size());
    a.eval()?;
    Ok(())
}
```

(Adds `?` after `Array::zeros(...)`.)

- [ ] **Step 9: Verify the workspace passes**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:"`
Expected: 4 sys tests pass + 2 mlx tests pass + 0 doc tests + 1 element test = clean.

- [ ] **Step 10: Commit**

```bash
git add mlx-sys/src/bridge/ mlx-sys/tests/sys_smoke.rs mlx/src/array.rs mlx/tests/p0_smoke.rs README.md
git commit -m "feat(p1a): array_zeros returns Result, applying shim Result-wrapping rule"
```

---

## Task 6: `Clone` for `Array` (with `array_clone` shim)

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1a_array.rs`

- [ ] **Step 1: Write the failing test**

Create `mlx/tests/p1a_array.rs`:

```rust
use mlx::{Array, Dtype};

#[test]
fn clone_shares_storage() {
    let a = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    let b = a.clone();
    // Both arrays should report the same shape — they share underlying storage.
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.size(), b.size());
    assert_eq!(a.dtype(), b.dtype());
}

#[test]
fn original_can_be_dropped_clone_still_usable() {
    let b = {
        let a = Array::zeros(&[5], Dtype::Int32).expect("zeros");
        a.clone()
    };
    // a is dropped; b still works because MLX refcount kept the storage alive.
    assert_eq!(b.size(), 5);
    b.eval().expect("eval after drop should succeed");
}
```

- [ ] **Step 2: Verify it fails**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_array clone_shares_storage 2>&1 | tail -10`
Expected: FAIL with "the trait bound `mlx::Array: Clone` is not satisfied".

- [ ] **Step 3: Add `array_clone` to the shim header**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after `array_dtype`:

```cpp
std::unique_ptr<MlxArray> array_clone(const MlxArray& a);
```

- [ ] **Step 4: Add `array_clone` shim implementation**

In `mlx-sys/shim/src/array.cc`, add after the `array_dtype` definition:

```cpp
std::unique_ptr<MlxArray> array_clone(const MlxArray& a) {
  // mlx::core::array's copy constructor shares the internal shared_ptr<ArrayDesc>;
  // this is cheap (atomic refcount++) and does not copy tensor data.
  return std::make_unique<MlxArray>(a);
}
```

- [ ] **Step 5: Add `array_clone` to the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add inside the `extern "C++"` block (after `array_dtype`):

```rust
        fn array_clone(a: &MlxArray) -> UniquePtr<MlxArray>;
```

(No `Result` wrapping — copy ctor doesn't throw per MLX contract.)

- [ ] **Step 6: Implement `Clone` on `Array`**

In `mlx/src/array.rs`, add at the end of the file (after the existing `impl Array` block):

```rust
impl Clone for Array {
    fn clone(&self) -> Self {
        Array(mlx_sys::array::ffi::array_clone(&self.0))
    }
}
```

- [ ] **Step 7: Verify the tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_array 2>&1 | tail -10`
Expected: 2 tests pass.

- [ ] **Step 8: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx/src/array.rs mlx/tests/p1a_array.rs
git commit -m "feat(p1a): impl Clone for Array via array_clone shim"
```

---

## Task 7: `Debug` for `Array` (with `array_is_available` shim, no eval triggered)

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1a_array.rs`

- [ ] **Step 1: Write the failing test**

Append to `mlx/tests/p1a_array.rs`:

```rust
#[test]
fn debug_does_not_trigger_eval() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    // Force-eval first to compare; then create a fresh lazy and verify Debug doesn't eval.
    let lazy = Array::zeros(&[4, 5], Dtype::Float32).expect("zeros");
    let was_available_before = mlx_sys::array::ffi::array_is_available(&unsafe_inner(&lazy));
    let _ = format!("{:?}", lazy);
    let was_available_after = mlx_sys::array::ffi::array_is_available(&unsafe_inner(&lazy));
    assert_eq!(was_available_before, was_available_after,
               "Debug must not trigger eval");
    // Sanity: after explicit eval, is_available should flip to true.
    lazy.eval().expect("eval");
    let was_available_after_eval = mlx_sys::array::ffi::array_is_available(&unsafe_inner(&lazy));
    assert!(was_available_after_eval, "after eval, is_available should be true");
    let _ = arr;
}

#[test]
fn debug_format_includes_shape_and_dtype() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    let s = format!("{:?}", arr);
    assert!(s.contains("shape"), "Debug output missing 'shape': {}", s);
    assert!(s.contains("Float32"), "Debug output missing 'Float32': {}", s);
    assert!(s.contains("2"), "Debug output missing dim '2': {}", s);
    assert!(s.contains("3"), "Debug output missing dim '3': {}", s);
}

// Helper: the test reaches into the FFI to call array_is_available directly
// because the safe layer doesn't expose this raw bool. Document this is
// for testing only.
fn unsafe_inner(arr: &Array) -> &mlx_sys::array::ffi::MlxArray {
    // SAFETY: This relies on Array being a #[repr(transparent)] newtype around
    // UniquePtr<MlxArray>. We don't enforce repr(transparent), but for testing
    // purposes we use the same access pattern via a public accessor.
    arr.as_inner()
}
```

NOTE: this test references `arr.as_inner()` which doesn't exist yet. Add a `pub(crate) fn as_inner` to Array, then expose it inside the test module via a re-export trick — actually the cleanest path is to make `as_inner` `pub` on Array but `#[doc(hidden)]`. Decide in Step 4.

- [ ] **Step 2: Verify it fails**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_array debug_does_not 2>&1 | tail -10`
Expected: FAIL with `array_is_available not found in ffi` or `as_inner not found`.

- [ ] **Step 3: Add `array_is_available` to the shim**

`mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after `array_clone`:

```cpp
bool array_is_available(const MlxArray& a);
```

`mlx-sys/shim/src/array.cc`, add after `array_clone`:

```cpp
bool array_is_available(const MlxArray& a) {
  // NB: mlx::core::array::is_available() is a const method that internally
  // mutates state via shared_ptr<ArrayDesc> (calls detach_event() and
  // set_status() on the available transition). This is safe under our
  // !Sync contract — only single-thread access to a given Array is allowed.
  return a.is_available();
}
```

- [ ] **Step 4: Add `array_is_available` to the cxx bridge + expose `as_inner` on `Array`**

`mlx-sys/src/bridge/array.rs` extern block, add after `array_clone`:

```rust
        fn array_is_available(a: &MlxArray) -> bool;
```

`mlx/src/array.rs`, add to the `impl Array` block (anywhere after `eval`):

```rust
    /// Hidden raw FFI access for advanced users and internal tests.
    #[doc(hidden)]
    pub fn as_inner(&self) -> &mlx_sys::array::ffi::MlxArray {
        &self.0
    }
```

- [ ] **Step 5: Implement `Debug` on `Array`**

In `mlx/src/array.rs`, add at the end of the file:

```rust
impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // CRITICAL: Debug must NOT trigger eval. Read shape/dtype/availability
        // through the cheap getters that the spec guarantees do not eval.
        let evaluated = mlx_sys::array::ffi::array_is_available(&self.0);
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("evaluated", &evaluated)
            .finish()
    }
}
```

- [ ] **Step 6: Verify all p1a_array tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_array 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx/src/array.rs mlx/tests/p1a_array.rs
git commit -m "feat(p1a): impl Debug for Array (no eval trigger) + as_inner accessor"
```

---

## Task 8: `shape()` returns `SmallVec`, add `shape_at`

**Files:**
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p0_smoke.rs`
- Modify: `mlx/tests/p1a_array.rs`

The cxx bridge `array_shape` continues to return `Vec<i32>` (cxx 1.0 doesn't natively bridge SmallVec). The conversion happens at the safe-layer wrapper. We also add `shape_at(dim: i32) -> i32` since it was specified.

- [ ] **Step 1: Write the failing test**

Append to `mlx/tests/p1a_array.rs`:

```rust
#[test]
fn shape_returns_smallvec_compatible_slice() {
    use smallvec::SmallVec;
    let arr = Array::zeros(&[2, 3, 4], Dtype::Float32).expect("zeros");
    let s = arr.shape();
    // Verify the public type really is a SmallVec; this would not compile if
    // shape() returned Vec<i32>.
    let _: &SmallVec<[i32; 8]> = &s;
    assert_eq!(s.as_slice(), &[2, 3, 4]);
    assert_eq!(s.len(), 3);
}

#[test]
fn shape_at_supports_negative_indexing() {
    let arr = Array::zeros(&[2, 3, 4], Dtype::Float32).expect("zeros");
    assert_eq!(arr.shape_at(0), 2);
    assert_eq!(arr.shape_at(2), 4);
    assert_eq!(arr.shape_at(-1), 4);
    assert_eq!(arr.shape_at(-3), 2);
}
```

- [ ] **Step 2: Verify the new tests fail**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_array shape_ 2>&1 | tail -10`
Expected: FAIL with `expected SmallVec, found Vec` (the existing `shape()` returns `Vec`).

- [ ] **Step 3: Update `shape()` to return `SmallVec` and add `shape_at`**

In `mlx/src/array.rs`, find the existing `shape()` method:

```rust
    pub fn shape(&self) -> Vec<i32> {
        mlx_sys::array::ffi::array_shape(&self.0)
    }
```

Replace with:

```rust
    /// The shape of the array. `[]` denotes a scalar.
    ///
    /// Returns a `SmallVec` with 8 inline slots — zero allocation for
    /// the common case of ≤ 8-dimensional tensors.
    pub fn shape(&self) -> smallvec::SmallVec<[i32; 8]> {
        let raw = mlx_sys::array::ffi::array_shape(&self.0);
        smallvec::SmallVec::from_vec(raw)
    }

    /// The size along the given dimension. Supports negative indexing
    /// (`-1` is the last dim).
    ///
    /// Panics if `dim` is out of range.
    pub fn shape_at(&self, dim: i32) -> i32 {
        let s = self.shape();
        let n = s.len() as i32;
        let idx = if dim < 0 { dim + n } else { dim };
        assert!(idx >= 0 && idx < n, "shape_at({dim}): out of range for ndim={n}");
        s[idx as usize]
    }
```

- [ ] **Step 4: Update `mlx/tests/p0_smoke.rs` for the new shape return type**

Find the two assertions:

```rust
    assert_eq!(arr.shape(), vec![2, 3]);
```

and:

```rust
    assert_eq!(arr.shape(), Vec::<i32>::new());
```

Change to:

```rust
    assert_eq!(arr.shape().as_slice(), &[2, 3]);
```

and:

```rust
    assert_eq!(arr.shape().as_slice(), &[] as &[i32]);
```

- [ ] **Step 5: Verify all tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx 2>&1 | grep "test result:"`
Expected: all test groups pass (p0_smoke 2/2, p1a_array 6/6, element 1/1).

- [ ] **Step 6: Commit**

```bash
git add mlx/src/array.rs mlx/tests/p0_smoke.rs mlx/tests/p1a_array.rs
git commit -m "feat(p1a): Array::shape returns SmallVec<[i32;8]>; add shape_at"
```

---

## Task 9: `from_slice<T>` (10 dtypes via Element + shape validation)

This task adds the FFI dispatch for all 10 element types and the safe `Array::from_slice<T>` method. The shim functions are repetitive but mechanical — bool bridges through `&[u8]` (cxx limitation), `half::f16`/`bf16` through `&[u16]` with `reinterpret_cast`.

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx/src/element.rs`
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1a_io.rs`

- [ ] **Step 1: Write the failing tests**

Create `mlx/tests/p1a_io.rs`:

```rust
use mlx::{Array, Dtype, Error};

#[test]
fn from_slice_f32_round_trip() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_slice(&data, &[2, 3]).expect("from_slice");
    assert_eq!(arr.shape().as_slice(), &[2, 3]);
    assert_eq!(arr.dtype(), Dtype::Float32);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_slice_i32_round_trip() {
    let data = vec![10_i32, 20, 30];
    let arr = Array::from_slice(&data, &[3]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Int32);
    assert_eq!(arr.size(), 3);
}

#[test]
fn from_slice_f16_round_trip() {
    let data = vec![half::f16::from_f32(1.5), half::f16::from_f32(2.5)];
    let arr = Array::from_slice(&data, &[2]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Float16);
    assert_eq!(arr.size(), 2);
}

#[test]
fn from_slice_bool_round_trip() {
    let data = vec![true, false, true, false];
    let arr = Array::from_slice(&data, &[2, 2]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Bool);
    assert_eq!(arr.size(), 4);
}

#[test]
fn from_slice_shape_mismatch_returns_err() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let result = Array::from_slice(&data, &[2, 3]);
    match result {
        Err(Error::ShapeMismatch { expected, actual }) => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn from_slice_empty_shape_is_scalar() {
    let data = vec![42.0_f32];
    let arr = Array::from_slice(&data, &[]).expect("from_slice scalar");
    assert_eq!(arr.size(), 1);
    assert_eq!(arr.ndim(), 0);
}
```

- [ ] **Step 2: Verify the tests fail**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io 2>&1 | tail -10`
Expected: FAIL with "no method named `from_slice` on `Array`".

- [ ] **Step 3: Add 10 `array_from_<T>` shim header declarations**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after `array_is_available`:

```cpp
// from_slice family — one per Element dtype. Slice element type matches
// MLX dtype size; bool bridges through uint8_t (cxx limitation),
// f16/bf16 bridge through uint16_t with reinterpret_cast.

std::unique_ptr<MlxArray> array_from_bool(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u8(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i8(rust::Slice<const int8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i16(rust::Slice<const int16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_bf16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f64(rust::Slice<const double> data, rust::Slice<const int32_t> shape);
```

- [ ] **Step 4: Add 10 `array_from_<T>` shim implementations**

In `mlx-sys/shim/src/array.cc`, add after `array_is_available`. Use a template helper to avoid 10x copy-paste. Add at the top (in the anonymous namespace, alongside `dtype_from_u8`):

```cpp
template <typename CppT>
std::unique_ptr<MlxArray> array_from_typed(
    rust::Slice<const CppT> data,
    rust::Slice<const int32_t> shape,
    mlx::core::Dtype dtype) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(
      mlx::core::array(data.data(), std::move(s), dtype));
}
```

Then add the 10 free functions outside the anonymous namespace (in `cxx_mlx`):

```cpp
std::unique_ptr<MlxArray> array_from_bool(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint8_t>(data, shape, mlx::core::bool_);
}
std::unique_ptr<MlxArray> array_from_u8(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint8_t>(data, shape, mlx::core::uint8);
}
std::unique_ptr<MlxArray> array_from_i8(rust::Slice<const int8_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int8_t>(data, shape, mlx::core::int8);
}
std::unique_ptr<MlxArray> array_from_i16(rust::Slice<const int16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int16_t>(data, shape, mlx::core::int16);
}
std::unique_ptr<MlxArray> array_from_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int32_t>(data, shape, mlx::core::int32);
}
std::unique_ptr<MlxArray> array_from_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int64_t>(data, shape, mlx::core::int64);
}
std::unique_ptr<MlxArray> array_from_f16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  // half::f16 has the same memory layout as mlx::core::float16_t (both 2-byte POD, IEEE 754 binary16).
  return array_from_typed<mlx::core::float16_t>(
      rust::Slice<const mlx::core::float16_t>(
          reinterpret_cast<const mlx::core::float16_t*>(data.data()),
          data.size()),
      shape, mlx::core::float16);
}
std::unique_ptr<MlxArray> array_from_bf16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<mlx::core::bfloat16_t>(
      rust::Slice<const mlx::core::bfloat16_t>(
          reinterpret_cast<const mlx::core::bfloat16_t*>(data.data()),
          data.size()),
      shape, mlx::core::bfloat16);
}
std::unique_ptr<MlxArray> array_from_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<float>(data, shape, mlx::core::float32);
}
std::unique_ptr<MlxArray> array_from_f64(rust::Slice<const double> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<double>(data, shape, mlx::core::float64);
}
```

(NB: if the `mlx::core::array(const T* data, Shape shape, Dtype dtype)` constructor doesn't exist with that exact signature, check `/Volumes/Dev/mlx/mlx/array.h` lines 56-85 for the actual ctor and adapt. Per `array.h:56`, the ctor is `array(const T* data, Shape shape, Dtype dtype = TypeToDtype<T>())` — should work.)

- [ ] **Step 5: Add 10 `array_from_<T>` to the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add inside the `extern "C++"` block (after `array_is_available`):

```rust
        // from_slice family — Result-wrapped per the shim throw rule
        // (MLX may throw on shape×dtype size mismatch).
        fn array_from_bool(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u8(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i8(data: &[i8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i16(data: &[i16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i32(data: &[i32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i64(data: &[i64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_bf16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f32(data: &[f32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f64(data: &[f64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 6: Implement `Element::array_from` for all 10 types**

Replace `mlx/src/element.rs` (just the impl bodies — keep the trait, sealed pattern, tests):

```rust
//! `Element` is the type-class for Rust types that map to MLX dtypes.

use crate::{Array, Dtype, Error, Result};

mod sealed {
    pub trait Sealed {}
}

pub trait Element: sealed::Sealed + Copy + Send + 'static {
    const DTYPE: Dtype;

    #[doc(hidden)]
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array>;

    #[doc(hidden)]
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>>;

    #[doc(hidden)]
    fn array_item(arr: &Array) -> Result<Self>;
}

// === Implementations ===
//
// Pattern: each type's array_from delegates to the corresponding shim
// function, with bool/f16/bf16 doing transparent reinterpret as needed.

impl sealed::Sealed for bool {}
impl Element for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // cxx::Slice doesn't accept &[bool]; convert to &[u8] (each true → 1, false → 0).
        let bytes: Vec<u8> = slice.iter().map(|&b| b as u8).collect();
        let inner = mlx_sys::array::ffi::array_from_bool(&bytes, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("Task 10") }
}

macro_rules! element_impl_simple {
    ($T:ty, $dt:expr, $shim_from:ident) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
                let inner = mlx_sys::array::ffi::$shim_from(slice, shape).map_err(Error::from)?;
                Ok(Array::from_inner(inner))
            }
            fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("Task 11") }
            fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("Task 10") }
        }
    };
}

element_impl_simple!(u8, Dtype::Uint8, array_from_u8);
element_impl_simple!(i8, Dtype::Int8, array_from_i8);
element_impl_simple!(i16, Dtype::Int16, array_from_i16);
element_impl_simple!(i32, Dtype::Int32, array_from_i32);
element_impl_simple!(i64, Dtype::Int64, array_from_i64);
element_impl_simple!(f32, Dtype::Float32, array_from_f32);
element_impl_simple!(f64, Dtype::Float64, array_from_f64);

// f16/bf16 reinterpret through &[u16] (half::f16 is repr(transparent) over u16).
impl sealed::Sealed for half::f16 {}
impl Element for half::f16 {
    const DTYPE: Dtype = Dtype::Float16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // SAFETY: half::f16 is #[repr(transparent)] over u16 (documented invariant of the
        // half crate), and the shim function takes a u16 slice that it reinterprets to
        // mlx::core::float16_t (also a 2-byte POD with identical bit layout).
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_f16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("Task 10") }
}

impl sealed::Sealed for half::bf16 {}
impl Element for half::bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;
    fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
        // SAFETY: half::bf16 is #[repr(transparent)] over u16 (documented invariant of the
        // half crate); shim reinterprets to mlx::core::bfloat16_t (identical 2-byte layout).
        let raw: &[u16] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast::<u16>(), slice.len())
        };
        let inner = mlx_sys::array::ffi::array_from_bf16(raw, shape).map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
    fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("Task 11") }
    fn array_item(_arr: &Array) -> Result<Self> { unimplemented!("Task 10") }
}

// Compile-time guarantee that f16/bf16 are 2 bytes (matching mlx::core::float16_t/bfloat16_t).
const _: () = {
    assert!(std::mem::size_of::<half::f16>() == 2);
    assert!(std::mem::size_of::<half::bf16>() == 2);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_const_matches_for_each_element() {
        assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
        assert_eq!(<u8 as Element>::DTYPE, Dtype::Uint8);
        assert_eq!(<i8 as Element>::DTYPE, Dtype::Int8);
        assert_eq!(<i16 as Element>::DTYPE, Dtype::Int16);
        assert_eq!(<i32 as Element>::DTYPE, Dtype::Int32);
        assert_eq!(<i64 as Element>::DTYPE, Dtype::Int64);
        assert_eq!(<half::f16 as Element>::DTYPE, Dtype::Float16);
        assert_eq!(<half::bf16 as Element>::DTYPE, Dtype::Bfloat16);
        assert_eq!(<f32 as Element>::DTYPE, Dtype::Float32);
        assert_eq!(<f64 as Element>::DTYPE, Dtype::Float64);
    }
}
```

This references `Array::from_inner(inner)` — a `pub(crate)` constructor. Add it next.

- [ ] **Step 7: Add `Array::from_inner` and `Array::from_slice` in `mlx/src/array.rs`**

In `mlx/src/array.rs`, add to the `impl Array` block:

```rust
    /// Construct from a raw cxx UniquePtr. Internal use only — the safe API
    /// is `Array::from_slice<T>` / `Array::zeros` / etc.
    pub(crate) fn from_inner(inner: cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>) -> Self {
        Array(inner)
    }

    /// Construct an array from a slice of `T` and a shape.
    ///
    /// Returns `Err(Error::ShapeMismatch)` if `slice.len()` does not equal
    /// `shape.iter().product()` (or 1 for empty/scalar shapes).
    pub fn from_slice<T: Element>(slice: &[T], shape: &[i32]) -> Result<Array> {
        let expected: usize = shape.iter().map(|&d| d as usize).product();
        let expected = if shape.is_empty() { 1 } else { expected };
        if slice.len() != expected {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![slice.len() as i32],
            });
        }
        T::array_from(slice, shape)
    }
```

(Add `use crate::Element;` at the top of `mlx/src/array.rs` if not already imported.)

- [ ] **Step 8: Verify all p1a_io tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io 2>&1 | tail -15`
Expected: 6 tests pass.

- [ ] **Step 9: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx/src/array.rs mlx/src/element.rs mlx/tests/p1a_io.rs
git commit -m "feat(p1a): from_slice<T> with 10-dtype FFI dispatch and shape validation"
```

---

## Task 10: `item<T>` (10 dtypes + dtype/size validation)

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx/src/element.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1a_io.rs`

- [ ] **Step 1: Write the failing tests**

Append to `mlx/tests/p1a_io.rs`:

```rust
#[test]
fn item_f32_round_trip() {
    let arr = Array::from_slice(&[42.0_f32], &[]).expect("from_slice");
    let v = arr.item::<f32>().expect("item");
    assert_eq!(v, 42.0);
}

#[test]
fn item_dtype_mismatch_returns_err() {
    let arr = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let result: Result<i32, _> = arr.item::<i32>();
    match result {
        Err(Error::DtypeMismatch { expected, actual }) => {
            assert_eq!(expected, Dtype::Int32);
            assert_eq!(actual, Dtype::Float32);
        }
        other => panic!("expected DtypeMismatch, got {other:?}"),
    }
}

#[test]
fn item_non_scalar_returns_err() {
    let arr = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let result = arr.item::<f32>();
    assert!(matches!(result, Err(Error::Mlx(_))));
}
```

(`Result` and `Error` are imported via `use mlx::{Array, Dtype, Error};` already; add `use std::result::Result;` if there's a name clash, or fully qualify.)

- [ ] **Step 2: Verify the tests fail**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io item_ 2>&1 | tail -10`
Expected: FAIL with "no method named `item` on `Array`".

- [ ] **Step 3: Add 10 `array_item_<T>` shim header declarations**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after the `array_from_*` block:

```cpp
// item family — extract the single scalar value. Caller must ensure size()==1
// and dtype matches; the shim does eval implicitly (mlx::array::item triggers it).

bool array_item_bool(const MlxArray& a);
uint8_t array_item_u8(const MlxArray& a);
int8_t array_item_i8(const MlxArray& a);
int16_t array_item_i16(const MlxArray& a);
int32_t array_item_i32(const MlxArray& a);
int64_t array_item_i64(const MlxArray& a);
uint16_t array_item_f16(const MlxArray& a);   // raw bits of half::f16
uint16_t array_item_bf16(const MlxArray& a);  // raw bits of half::bf16
float array_item_f32(const MlxArray& a);
double array_item_f64(const MlxArray& a);
```

- [ ] **Step 4: Add 10 `array_item_<T>` shim implementations**

In `mlx-sys/shim/src/array.cc`, add after the `array_from_*` block:

```cpp
bool array_item_bool(const MlxArray& a) { return a.item<bool>(); }
uint8_t array_item_u8(const MlxArray& a) { return a.item<uint8_t>(); }
int8_t array_item_i8(const MlxArray& a) { return a.item<int8_t>(); }
int16_t array_item_i16(const MlxArray& a) { return a.item<int16_t>(); }
int32_t array_item_i32(const MlxArray& a) { return a.item<int32_t>(); }
int64_t array_item_i64(const MlxArray& a) { return a.item<int64_t>(); }
uint16_t array_item_f16(const MlxArray& a) {
  // Read out as mlx::core::float16_t and reinterpret to raw uint16_t.
  auto v = a.item<mlx::core::float16_t>();
  uint16_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
uint16_t array_item_bf16(const MlxArray& a) {
  auto v = a.item<mlx::core::bfloat16_t>();
  uint16_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
float array_item_f32(const MlxArray& a) { return a.item<float>(); }
double array_item_f64(const MlxArray& a) { return a.item<double>(); }
```

(Add `#include <cstring>` at the top of `array.cc` if not already there — `memcpy` needs it.)

- [ ] **Step 5: Add 10 `array_item_<T>` to the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add inside the `extern "C++"` block (after the `array_from_*` block):

```rust
        // item family — Result-wrapped (MLX item<T>() may throw if dtype mismatches
        // or eval fails for any reason).
        fn array_item_bool(a: &MlxArray) -> Result<bool>;
        fn array_item_u8(a: &MlxArray) -> Result<u8>;
        fn array_item_i8(a: &MlxArray) -> Result<i8>;
        fn array_item_i16(a: &MlxArray) -> Result<i16>;
        fn array_item_i32(a: &MlxArray) -> Result<i32>;
        fn array_item_i64(a: &MlxArray) -> Result<i64>;
        fn array_item_f16(a: &MlxArray) -> Result<u16>;
        fn array_item_bf16(a: &MlxArray) -> Result<u16>;
        fn array_item_f32(a: &MlxArray) -> Result<f32>;
        fn array_item_f64(a: &MlxArray) -> Result<f64>;
```

- [ ] **Step 6: Implement `Element::array_item` for all 10 types in `mlx/src/element.rs`**

Replace the `unimplemented!("Task 10")` lines:

For the `bool` impl:
```rust
    fn array_item(arr: &Array) -> Result<Self> {
        mlx_sys::array::ffi::array_item_bool(arr.as_inner()).map_err(Error::from)
    }
```

Update `element_impl_simple!` macro to also include `$shim_item`:
```rust
macro_rules! element_impl_simple {
    ($T:ty, $dt:expr, $shim_from:ident, $shim_item:ident) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
                let inner = mlx_sys::array::ffi::$shim_from(slice, shape).map_err(Error::from)?;
                Ok(Array::from_inner(inner))
            }
            fn array_to_vec(_arr: &Array) -> Result<Vec<Self>> { unimplemented!("Task 11") }
            fn array_item(arr: &Array) -> Result<Self> {
                mlx_sys::array::ffi::$shim_item(arr.as_inner()).map_err(Error::from)
            }
        }
    };
}

element_impl_simple!(u8, Dtype::Uint8, array_from_u8, array_item_u8);
element_impl_simple!(i8, Dtype::Int8, array_from_i8, array_item_i8);
element_impl_simple!(i16, Dtype::Int16, array_from_i16, array_item_i16);
element_impl_simple!(i32, Dtype::Int32, array_from_i32, array_item_i32);
element_impl_simple!(i64, Dtype::Int64, array_from_i64, array_item_i64);
element_impl_simple!(f32, Dtype::Float32, array_from_f32, array_item_f32);
element_impl_simple!(f64, Dtype::Float64, array_from_f64, array_item_f64);
```

For the `f16` impl, update to:
```rust
    fn array_item(arr: &Array) -> Result<Self> {
        let bits = mlx_sys::array::ffi::array_item_f16(arr.as_inner()).map_err(Error::from)?;
        Ok(half::f16::from_bits(bits))
    }
```

Same for `bf16` (using `half::bf16::from_bits`).

- [ ] **Step 7: Add `Array::item<T>` in `mlx/src/array.rs`**

In the `impl Array` block, add:

```rust
    /// Read this array as a single scalar of type `T`.
    ///
    /// Returns `Err` if the array is not a scalar (size != 1) or if its
    /// dtype does not match `T::DTYPE`. Implicitly evaluates the array.
    pub fn item<T: Element>(&self) -> Result<T> {
        if self.size() != 1 {
            return Err(Error::Mlx(format!(
                "item() called on non-scalar array (size={}, shape={:?})",
                self.size(),
                self.shape().as_slice()
            )));
        }
        if self.dtype() != T::DTYPE {
            return Err(Error::DtypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype(),
            });
        }
        T::array_item(self)
    }
```

- [ ] **Step 8: Verify all p1a_io tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io 2>&1 | grep "test result:"`
Expected: 9 tests pass (6 from Task 9 + 3 from this task).

- [ ] **Step 9: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx/src/array.rs mlx/src/element.rs mlx/tests/p1a_io.rs
git commit -m "feat(p1a): item<T> with 10-dtype FFI + dtype/size validation"
```

---

## Task 11: `to_vec<T>` (10 dtypes + implicit eval + dtype validation)

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx/src/element.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1a_io.rs`

- [ ] **Step 1: Write the failing tests**

Append to `mlx/tests/p1a_io.rs`:

```rust
#[test]
fn to_vec_f32_round_trip() {
    let original = vec![1.0_f32, 2.0, 3.0, 4.0];
    let arr = Array::from_slice(&original, &[2, 2]).expect("from_slice");
    let read_back = arr.to_vec::<f32>().expect("to_vec");
    assert_eq!(read_back, original);
}

#[test]
fn to_vec_implicit_eval() {
    // Lazy zeros — should NOT need explicit eval before to_vec.
    let arr = Array::zeros(&[3], Dtype::Float32).expect("zeros");
    let v = arr.to_vec::<f32>().expect("to_vec triggers eval");
    assert_eq!(v, vec![0.0_f32, 0.0, 0.0]);
}

#[test]
fn to_vec_f16_bit_pattern_preserved() {
    // Specific bit patterns (NaN, denormal, +0/-0) round-trip exactly.
    let original: Vec<half::f16> = vec![
        half::f16::from_f32(1.5),
        half::f16::from_f32(-2.25),
        half::f16::from_bits(0x7C01), // signaling NaN-ish bit pattern
        half::f16::from_bits(0x0001), // denormal
    ];
    let arr = Array::from_slice(&original, &[4]).expect("from_slice");
    let read_back = arr.to_vec::<half::f16>().expect("to_vec");
    for (i, (a, b)) in original.iter().zip(read_back.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bit pattern mismatch at index {i}");
    }
}

#[test]
fn to_vec_dtype_mismatch_returns_err() {
    let arr = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let result: Result<Vec<i32>, _> = arr.to_vec::<i32>();
    match result {
        Err(Error::DtypeMismatch { expected, actual }) => {
            assert_eq!(expected, Dtype::Int32);
            assert_eq!(actual, Dtype::Float32);
        }
        other => panic!("expected DtypeMismatch, got {other:?}"),
    }
}

#[test]
fn to_vec_bool_round_trip() {
    let original = vec![true, false, true];
    let arr = Array::from_slice(&original, &[3]).expect("from_slice");
    let read_back = arr.to_vec::<bool>().expect("to_vec");
    assert_eq!(read_back, original);
}
```

- [ ] **Step 2: Verify the tests fail**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io to_vec 2>&1 | tail -10`
Expected: FAIL with "no method named `to_vec` on `Array`".

- [ ] **Step 3: Add 10 `array_to_vec_<T>` shim header declarations**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after the `array_item_*` block:

```cpp
// to_vec family — copy all elements out as a rust::Vec. Triggers eval.

rust::Vec<uint8_t> array_to_vec_bool(const MlxArray& a);   // 1 byte per bool
rust::Vec<uint8_t> array_to_vec_u8(const MlxArray& a);
rust::Vec<int8_t> array_to_vec_i8(const MlxArray& a);
rust::Vec<int16_t> array_to_vec_i16(const MlxArray& a);
rust::Vec<int32_t> array_to_vec_i32(const MlxArray& a);
rust::Vec<int64_t> array_to_vec_i64(const MlxArray& a);
rust::Vec<uint16_t> array_to_vec_f16(const MlxArray& a);   // raw bits of half::f16
rust::Vec<uint16_t> array_to_vec_bf16(const MlxArray& a);  // raw bits of half::bf16
rust::Vec<float> array_to_vec_f32(const MlxArray& a);
rust::Vec<double> array_to_vec_f64(const MlxArray& a);
```

- [ ] **Step 4: Add 10 `array_to_vec_<T>` shim implementations**

In `mlx-sys/shim/src/array.cc`, add after `array_item_*`. Use a template helper inside the anonymous namespace:

```cpp
template <typename CppT, typename WireT = CppT>
rust::Vec<WireT> array_to_vec_typed(const MlxArray& a) {
  // Triggers eval if needed. mlx::core::array::data<T>() returns a const T*
  // into the array's contiguous storage; .size() is total elements.
  // We require eval to have happened (caller responsibility — typically the
  // safe Rust layer calls Array::eval() before to_vec).
  rust::Vec<WireT> out;
  out.reserve(a.size());
  const CppT* ptr = a.data<CppT>();
  for (size_t i = 0; i < a.size(); ++i) {
    if constexpr (std::is_same_v<CppT, WireT>) {
      out.push_back(ptr[i]);
    } else {
      WireT bits;
      std::memcpy(&bits, &ptr[i], sizeof(bits));
      out.push_back(bits);
    }
  }
  return out;
}

rust::Vec<uint8_t> array_to_vec_bool(const MlxArray& a) {
  // mlx stores bool as 1-byte; reinterpret to uint8_t for the wire.
  return array_to_vec_typed<bool, uint8_t>(a);
}
rust::Vec<uint8_t> array_to_vec_u8(const MlxArray& a)   { return array_to_vec_typed<uint8_t>(a); }
rust::Vec<int8_t> array_to_vec_i8(const MlxArray& a)    { return array_to_vec_typed<int8_t>(a); }
rust::Vec<int16_t> array_to_vec_i16(const MlxArray& a)  { return array_to_vec_typed<int16_t>(a); }
rust::Vec<int32_t> array_to_vec_i32(const MlxArray& a)  { return array_to_vec_typed<int32_t>(a); }
rust::Vec<int64_t> array_to_vec_i64(const MlxArray& a)  { return array_to_vec_typed<int64_t>(a); }
rust::Vec<uint16_t> array_to_vec_f16(const MlxArray& a) {
  return array_to_vec_typed<mlx::core::float16_t, uint16_t>(a);
}
rust::Vec<uint16_t> array_to_vec_bf16(const MlxArray& a) {
  return array_to_vec_typed<mlx::core::bfloat16_t, uint16_t>(a);
}
rust::Vec<float> array_to_vec_f32(const MlxArray& a)    { return array_to_vec_typed<float>(a); }
rust::Vec<double> array_to_vec_f64(const MlxArray& a)   { return array_to_vec_typed<double>(a); }
```

(`#include <cstring>` if not already added in Task 10. `<type_traits>` for `std::is_same_v`.)

- [ ] **Step 5: Add 10 `array_to_vec_<T>` to the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add inside the `extern "C++"` block (after the `array_item_*` block):

```rust
        // to_vec family — Result-wrapped (data() can throw if storage isn't
        // available, e.g. eval hasn't been called).
        fn array_to_vec_bool(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_u8(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_i8(a: &MlxArray) -> Result<Vec<i8>>;
        fn array_to_vec_i16(a: &MlxArray) -> Result<Vec<i16>>;
        fn array_to_vec_i32(a: &MlxArray) -> Result<Vec<i32>>;
        fn array_to_vec_i64(a: &MlxArray) -> Result<Vec<i64>>;
        fn array_to_vec_f16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_bf16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_f32(a: &MlxArray) -> Result<Vec<f32>>;
        fn array_to_vec_f64(a: &MlxArray) -> Result<Vec<f64>>;
```

- [ ] **Step 6: Implement `Element::array_to_vec` for all 10 types**

Replace the `unimplemented!("Task 11")` lines:

For `bool`:
```rust
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
        arr.eval()?;  // implicit eval per spec A8
        let bytes = mlx_sys::array::ffi::array_to_vec_bool(arr.as_inner()).map_err(Error::from)?;
        Ok(bytes.into_iter().map(|b| b != 0).collect())
    }
```

Update the `element_impl_simple!` macro to also accept `$shim_to_vec`:
```rust
macro_rules! element_impl_simple {
    ($T:ty, $dt:expr, $shim_from:ident, $shim_item:ident, $shim_to_vec:ident) => {
        impl sealed::Sealed for $T {}
        impl Element for $T {
            const DTYPE: Dtype = $dt;
            fn array_from(slice: &[Self], shape: &[i32]) -> Result<Array> {
                let inner = mlx_sys::array::ffi::$shim_from(slice, shape).map_err(Error::from)?;
                Ok(Array::from_inner(inner))
            }
            fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
                arr.eval()?;
                let raw = mlx_sys::array::ffi::$shim_to_vec(arr.as_inner()).map_err(Error::from)?;
                Ok(raw.into())
            }
            fn array_item(arr: &Array) -> Result<Self> {
                mlx_sys::array::ffi::$shim_item(arr.as_inner()).map_err(Error::from)
            }
        }
    };
}

element_impl_simple!(u8, Dtype::Uint8, array_from_u8, array_item_u8, array_to_vec_u8);
element_impl_simple!(i8, Dtype::Int8, array_from_i8, array_item_i8, array_to_vec_i8);
element_impl_simple!(i16, Dtype::Int16, array_from_i16, array_item_i16, array_to_vec_i16);
element_impl_simple!(i32, Dtype::Int32, array_from_i32, array_item_i32, array_to_vec_i32);
element_impl_simple!(i64, Dtype::Int64, array_from_i64, array_item_i64, array_to_vec_i64);
element_impl_simple!(f32, Dtype::Float32, array_from_f32, array_item_f32, array_to_vec_f32);
element_impl_simple!(f64, Dtype::Float64, array_from_f64, array_item_f64, array_to_vec_f64);
```

For `f16`:
```rust
    fn array_to_vec(arr: &Array) -> Result<Vec<Self>> {
        arr.eval()?;
        let raw = mlx_sys::array::ffi::array_to_vec_f16(arr.as_inner()).map_err(Error::from)?;
        Ok(raw.into_iter().map(half::f16::from_bits).collect())
    }
```

Same for `bf16` with `half::bf16::from_bits`.

- [ ] **Step 7: Add `Array::to_vec<T>` in `mlx/src/array.rs`**

In the `impl Array` block, add:

```rust
    /// Copy all elements out as a `Vec<T>`. Implicitly evaluates if needed.
    ///
    /// Returns `Err(Error::DtypeMismatch)` if the array's dtype does not
    /// match `T::DTYPE`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DtypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype(),
            });
        }
        T::array_to_vec(self)
    }
```

- [ ] **Step 8: Verify all p1a_io tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_io 2>&1 | grep "test result:"`
Expected: 14 tests pass (6 from Task 9 + 3 from Task 10 + 5 from this task).

- [ ] **Step 9: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx/src/array.rs mlx/src/element.rs mlx/tests/p1a_io.rs
git commit -m "feat(p1a): to_vec<T> with 10-dtype FFI + implicit eval + dtype validation"
```

---

## Task 12: `Send` marker + thread-safety static asserts

**Files:**
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1a_thread_safety.rs`

- [ ] **Step 1: Write the failing test**

Create `mlx/tests/p1a_thread_safety.rs`:

```rust
use mlx::Array;

fn assert_send<T: Send>() {}

#[test]
fn array_is_send() {
    assert_send::<Array>();
}

#[test]
fn array_is_not_sync() {
    use static_assertions::assert_not_impl_any;
    assert_not_impl_any!(Array: Sync);
}
```

- [ ] **Step 2: Verify the Send test fails**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_thread_safety 2>&1 | tail -10`
Expected: FAIL with "the trait `Send` is not implemented for `cxx::UniquePtr<MlxArray>`" or similar (because cxx opaque types are `!Send + !Sync` by default).

- [ ] **Step 3: Add `unsafe impl Send for Array`**

In `mlx/src/array.rs`, add at the end of the file:

```rust
// SAFETY: MLX's `mlx::core::array` is internally backed by
// `std::shared_ptr<ArrayDesc>`. The shared_ptr refcount is atomic, so
// transferring ownership across threads is safe (the destructor in the
// receiving thread can decrement the refcount).
//
// We do NOT impl Sync because MLX's "const" methods (set_status,
// attach_event, is_available's lazy→available transition) mutate the
// underlying ArrayDesc without synchronization. Two threads holding
// `&Array` to the same array would race. To share an Array between
// threads, clone it (cheap MLX refcount) or wrap it in
// `Arc<Mutex<Array>>`. See README "Threading" section.
unsafe impl Send for Array {}
```

- [ ] **Step 4: Verify both tests pass**

Run: `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1a_thread_safety 2>&1 | tail -10`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mlx/src/array.rs mlx/tests/p1a_thread_safety.rs
git commit -m "feat(p1a): Array: Send (not Sync), with safety rationale comment"
```

---

## Task 13: Final workspace verification + README polish

**Files:**
- Modify: `README.md`
- Modify: `mlx/src/lib.rs` — add crate-level threading note
- Verify: full workspace test + clippy + doc build

- [ ] **Step 1: Add a "Threading" section to `README.md`**

Append after the "Quickstart" section:

```markdown
## Threading

`mlx::Array` implements `Send` but **not** `Sync`. Internally, MLX's
`mlx::core::array` is backed by a `std::shared_ptr` whose refcount is atomic,
so transferring ownership across threads is safe. However, MLX's "const"
methods (e.g. `set_status`, `attach_event`, the lazy→available transition
in `is_available`) mutate `ArrayDesc` without synchronization, so two
threads concurrently holding `&Array` to the same instance is a data race.

**To share an array between threads:**

```rust
let a = mlx::Array::zeros(&[2, 3], mlx::Dtype::Float32)?;
let b = a.clone();   // Cheap — MLX refcounts the underlying storage.
std::thread::spawn(move || {
    let _ = b.shape();
});
```

`a.clone()` does an atomic refcount increment on the MLX storage and
allocates a small wrapper. Tensor data is not copied. Avoid wrapping in
`Arc<Mutex<Array>>` unless you genuinely need shared mutable access.
```

- [ ] **Step 2: Add a crate-level doc note to `mlx/src/lib.rs`**

Replace the `//! Safe Rust bindings to Apple MLX.` line with:

```rust
//! Safe Rust bindings to Apple MLX.
//!
//! # Quickstart
//!
//! ```no_run
//! use mlx::{Array, Dtype};
//!
//! # fn main() -> mlx::Result<()> {
//! let a = Array::zeros(&[2, 3], Dtype::Float32)?;
//! let v: Vec<f32> = a.to_vec()?;
//! assert_eq!(v.len(), 6);
//! # Ok(())
//! # }
//! ```
//!
//! # Threading
//!
//! `Array` is `Send` but not `Sync`. To share an array between threads,
//! clone it (cheap MLX refcount). See the README for details.
```

- [ ] **Step 3: Run the full test suite**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:"`
Expected: all groups pass:
- `mlx-sys::sys_smoke`: 4 tests
- `mlx::p0_smoke`: 2 tests
- `mlx::p1a_array`: 6 tests
- `mlx::p1a_io`: 14 tests
- `mlx::p1a_thread_safety`: 2 tests
- `mlx::error::tests`: 3 tests
- `mlx::element::tests`: 1 test
- `mlx` doc tests: 1 test (the quickstart in lib.rs)
- Plus 0 doc tests in mlx-sys

Total: at least 33 tests passing.

- [ ] **Step 4: Run clippy with warnings-as-errors**

Run: `MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -10`
Expected: no Rust-level warnings (some C++ compiler warnings from MLX headers via `cargo:warning=` are noise — they don't fail clippy).

If clippy flags `cast_possible_wrap` or similar in the from_slice product/casts, silence per-file with `#![allow(clippy::cast_possible_wrap)]` at the top of `mlx/src/array.rs`. Do not silence workspace-wide.

- [ ] **Step 5: Build docs to verify the doctest passes**

Run: `MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | tail -5`
Expected: `Finished` with no errors.

- [ ] **Step 6: Commit**

```bash
git add README.md mlx/src/lib.rs
git commit -m "docs(p1a): add Threading section + crate-level doctest"
```

---

## Acceptance Criteria

P1a is complete when:

1. `MLX_DIR=$HOME/.local/mlx cargo test --workspace` reports all tests passing — at least 33 tests across 8 test groups
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean
3. `mlx::Array` supports: `zeros` (returns `Result`), `from_slice<T>`, `item<T>`, `to_vec<T>`, `clone`, `Debug` (no eval trigger), `shape` (returns `SmallVec`), `shape_at`, `dtype`, `ndim`, `size`, `eval`
4. `Element` trait implemented for `bool`, `u8`, `i8`, `i16`, `i32`, `i64`, `half::f16`, `half::bf16`, `f32`, `f64` — no others (sealed)
5. `Array: Send`, verified by compile-time `assert_send`. `Array: !Sync`, verified by `static_assertions::assert_not_impl_any`
6. C++ shim has 3 endpoint `static_assert`s (bool_=0, float32=10, complex64=13)
7. Every shim function that can throw is declared `Result<T>` in its bridge (rule documented in `mlx-sys/src/bridge/mod.rs`)
8. `Array::shape()` returns `SmallVec<[i32; 8]>` (zero-alloc on ≤ 8 dims)
9. README documents the threading model

When all 9 hold, P1a is ready for merge to master and P1b planning starts.
