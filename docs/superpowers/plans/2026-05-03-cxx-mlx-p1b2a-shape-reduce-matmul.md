# cxx-mlx P1b2a (Shape + Reduction + Matmul) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 shape ops (reshape with `-1` inference, transpose/transpose_axes, broadcast_to, concatenate, stack, split_n, split_at), 5 reduction ops (sum/mean/max/min/argmax) unified via the `IntoAxes` sealed trait + `All` marker, and `matmul` (covering 2D/batched/broadcast). Refactor `mlx/src/ops.rs` into an `ops/` subdirectory split by responsibility. Acceptance: numerical-correctness integration tests for `softmax`/`gelu`/`silu` composed atop these ops.

**Architecture:** Pattern continues from P1b1 — three-layer split (free fn in `ops/*.rs` is source of truth, `ops_impl.rs` operator sugar unchanged, `array.rs` thin method wrappers). New cross-bridge `MlxArrayVec` opaque type for `split` returns (cxx 1.0 doesn't bridge `Vec<UniquePtr<T>>` directly). Varargs (`&[&Array]` for `concatenate`/`stack`) cross the FFI boundary as `&[*const MlxArray]` raw pointer slices, reassembled into `std::vector<array>` C++-side via per-element copy ctor (refcount-shared, cheap).

**Tech Stack:** Rust 1.94+, cxx 1.0, MLX C++ 0.32 (already at `$MLX_DIR`), C++20. No new dependencies.

**Branch:** Work on `p1b2-ops` (already created off master). MLX install at `$HOME/.local/mlx`; export `MLX_DIR=$HOME/.local/mlx` for every cargo invocation.

---

## File Structure

**New files:**

- `mlx/src/ops/mod.rs` — module declarations + re-exports (`pub use binary::*; pub use unary::*; pub use shape::*; pub use reduction::*; pub use matmul::*;`)
- `mlx/src/ops/binary.rs` — moved from `ops.rs`: `add`, `subtract`, `multiply`, `divide`, `negative`
- `mlx/src/ops/unary.rs` — moved from `ops.rs`: `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `square`, `rsqrt`, `erf`, `reciprocal`
- `mlx/src/ops/shape.rs` — `reshape` (with `-1` inference), `transpose`, `transpose_axes`, `broadcast_to`, `concatenate`, `stack`, `split_n`, `split_at`
- `mlx/src/ops/reduction.rs` — `IntoAxes` trait + `All` struct + 5 impls + `sum`/`mean`/`max`/`min`/`argmax`
- `mlx/src/ops/matmul.rs` — `matmul`
- `mlx/tests/p1b2a_reduction.rs` — reduction integration tests (3 axis forms × 5 ops + keepdim)
- `mlx/tests/p1b2a_shape.rs` — shape op integration tests
- `mlx/tests/p1b2a_matmul.rs` — matmul integration tests (2D, batched, broadcast)
- `mlx/tests/p1b2a_compose.rs` — softmax/gelu/silu numerical correctness

**Deleted files:**

- `mlx/src/ops.rs` — content moved to `ops/binary.rs` and `ops/unary.rs`

**Modified files:**

- `mlx/src/lib.rs` — `pub mod ops;` continues but now resolves to `ops/mod.rs`; add `pub use ops::reduction::All;` (top-level convenience)
- `mlx/src/array.rs` — add new methods: `reshape`, `transpose`, `t`, `transpose_axes`, `broadcast_to`, `matmul`, `sum`/`mean`/`max`/`min`/`argmax` (with `IntoAxes` generic). Each is a 1-line delegation
- `mlx/src/ops_impl.rs` — `use crate::ops` import paths verified (no functional change)
- `mlx-sys/src/bridge/array.rs` — add ~22 new shim function declarations, `MlxArrayVec` opaque type, accessor functions
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — add new declarations
- `mlx-sys/shim/src/array.cc` — add new implementations
- `README.md` — update Status line + add "Reductions / Shape / Matmul" example section

---

## Task 1: Refactor `ops.rs` → `ops/` subdirectory (no functional change)

**Files:**
- Delete: `mlx/src/ops.rs`
- Create: `mlx/src/ops/mod.rs`
- Create: `mlx/src/ops/binary.rs`
- Create: `mlx/src/ops/unary.rs`

This task is a pure rearrangement — no semantic changes. All 14 P1b1 functions move from `ops.rs` to `ops/binary.rs` (5 functions: `add`, `subtract`, `multiply`, `divide`, `negative`) and `ops/unary.rs` (9 functions: `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `square`, `rsqrt`, `erf`, `reciprocal`). `ops/mod.rs` re-exports everything so `mlx::ops::add` (etc.) keeps working unchanged.

- [ ] **Step 1: Verify the current ops.rs content (read-only orientation)**

```bash
wc -l /Volumes/Dev/cxx-mlx/mlx/src/ops.rs
```
Expected: ~75 lines (14 functions + module-level docstring + macro definitions).

- [ ] **Step 2: Create `mlx/src/ops/binary.rs`**

Extract the 5 binary functions from the existing `ops.rs`. The contents:

```rust
//! Binary element-wise ops with NumPy broadcasting.
//!
//! Each function validates broadcast compatibility before crossing the FFI
//! boundary so we can return `Error::BroadcastMismatch` with structured
//! `lhs`/`rhs` fields, instead of relying on MLX's English exception strings.

use crate::{broadcast, Array, Error, Result};

/// Element-wise addition with NumPy broadcasting.
pub fn add(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise subtraction with NumPy broadcasting.
pub fn subtract(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_subtract(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise multiplication with NumPy broadcasting.
pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_multiply(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise division with NumPy broadcasting.
pub fn divide(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_divide(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise negation.
///
/// On unsigned dtypes (`u8`/`u16`/etc.) MLX wraps two's-complement style
/// (e.g. `1u8 → 255u8`); it does not throw. On `bool` MLX errors at eval
/// time per its own dtype rules.
pub fn negative(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_negative(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 3: Create `mlx/src/ops/unary.rs`**

Extract the 9 unary functions and the `unary_op!` macro from `ops.rs`:

```rust
//! Element-wise unary ops.
//!
//! All return `Result<Array>` because dtype mismatches (e.g. `sqrt` on integer
//! types) raise MLX exceptions that we surface as `Error::Mlx`.

use crate::{Array, Error, Result};

/// Macro to define a unary op delegating to a single shim function.
macro_rules! unary_op {
    ($name:ident, $shim:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(a: &Array) -> Result<Array> {
            let inner = mlx_sys::array::ffi::$shim(a.as_inner()).map_err(Error::from)?;
            Ok(Array::from_inner(inner))
        }
    };
}

unary_op!(exp,        array_exp,        "Element-wise natural exponential.");
unary_op!(log,        array_log,        "Element-wise natural logarithm.");
unary_op!(sqrt,       array_sqrt,       "Element-wise square root.");
unary_op!(tanh,       array_tanh,       "Element-wise hyperbolic tangent.");
unary_op!(sigmoid,    array_sigmoid,    "Element-wise sigmoid (1 / (1 + exp(-x))).");
unary_op!(square,     array_square,     "Element-wise x^2.");
unary_op!(rsqrt,      array_rsqrt,      "Element-wise 1/sqrt(x). Used in attention scaling.");
unary_op!(erf,        array_erf,        "Element-wise error function. Used in GELU.");
unary_op!(reciprocal, array_reciprocal, "Element-wise 1/x.");
```

- [ ] **Step 4: Create `mlx/src/ops/mod.rs`**

```rust
//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.

pub mod binary;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
```

- [ ] **Step 5: Delete `mlx/src/ops.rs`**

```bash
rm /Volumes/Dev/cxx-mlx/mlx/src/ops.rs
```

- [ ] **Step 6: Verify all P0/P1a/P1b1 tests still pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:"
```
Expected: all 57 tests still pass (no functional changes — just file move).

If any test fails, the most likely cause is `mod ops;` resolution. The lib.rs already says `pub mod ops;` which now resolves to `ops/mod.rs` automatically. No lib.rs change needed.

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/ops.rs
git commit -m "refactor(p1b2a): split ops.rs into ops/binary.rs + ops/unary.rs (no functional change)"
```

(`git add mlx/src/ops.rs` is needed to stage the deletion since we deleted but didn't rm via git.)

---

## Task 2: `IntoAxes` sealed trait + `All` unit struct

**Files:**
- Create: `mlx/src/ops/reduction.rs` (initial skeleton — only the trait + struct + tests in this task; reduction functions land in Tasks 4-5)
- Modify: `mlx/src/ops/mod.rs` (declare new module, re-export `All` and `IntoAxes`)
- Modify: `mlx/src/lib.rs` (top-level re-export of `All` for convenience)

- [ ] **Step 1: Write the failing unit tests inline in `mlx/src/ops/reduction.rs`**

Create the file with the trait, struct, impls, and tests:

```rust
//! Reduction ops (`sum`/`mean`/`max`/`min`/`argmax`) over array axes.
//!
//! Axes are passed via the `IntoAxes` sealed trait, accepting:
//!
//! - [`All`] — reduce over every axis (returns a scalar by default; or shape
//!   `[1, 1, ...]` if `keepdim` is true)
//! - `i32` — reduce a single axis (negative supported)
//! - `&[i32]` / `Vec<i32>` / `[i32; N]` — reduce multiple axes
//!
//! Keepdim is a positional `bool` (NumPy/PyTorch convention). When `true`,
//! reduced axes are kept as size-1 to preserve broadcast compatibility.

use crate::{Array, Error, Result};

/// Marker for "reduce over all axes". Use as `sum(&a, All, false)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct All;

mod sealed {
    pub trait Sealed {}
}

/// Sealed trait describing how an argument is interpreted as reduction axes.
///
/// Implemented for [`All`], `i32`, `&[i32]`, `Vec<i32>`, and `[i32; N]`.
/// External crates cannot implement this trait.
pub trait IntoAxes: sealed::Sealed {
    /// Returns `None` for the all-axes case, or `Some(slice)` for specific axes.
    /// Internal — used by the reduction dispatchers to pick the matching shim.
    #[doc(hidden)]
    fn as_axes(&self) -> Option<&[i32]>;
}

impl sealed::Sealed for All {}
impl IntoAxes for All {
    fn as_axes(&self) -> Option<&[i32]> { None }
}

impl sealed::Sealed for i32 {}
impl IntoAxes for i32 {
    fn as_axes(&self) -> Option<&[i32]> { Some(std::slice::from_ref(self)) }
}

impl sealed::Sealed for &[i32] {}
impl IntoAxes for &[i32] {
    fn as_axes(&self) -> Option<&[i32]> { Some(self) }
}

impl sealed::Sealed for Vec<i32> {}
impl IntoAxes for Vec<i32> {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
}

impl<const N: usize> sealed::Sealed for [i32; N] {}
impl<const N: usize> IntoAxes for [i32; N] {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_none() {
        assert_eq!(All.as_axes(), None);
    }

    #[test]
    fn i32_returns_single_element_slice() {
        let axis: i32 = -1;
        assert_eq!(axis.as_axes(), Some(&[-1][..]));
    }

    #[test]
    fn slice_returns_self() {
        let axes: &[i32] = &[0, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }

    #[test]
    fn vec_returns_slice() {
        let axes = vec![0_i32, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }

    #[test]
    fn array_literal_returns_slice() {
        let axes: [i32; 2] = [0, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }
}

// Reduction functions land in Tasks 4 and 5.
#[allow(unused_imports)]
use Array as _;
#[allow(unused_imports)]
use Error as _;
#[allow(unused_imports)]
use Result as _;
```

(The `#[allow(unused_imports)]` block silences warnings until Task 4 actually uses `Array`/`Error`/`Result`. Remove it then.)

- [ ] **Step 2: Wire up the new module in `mlx/src/ops/mod.rs`**

Update `mlx/src/ops/mod.rs` to:

```rust
//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.

pub mod binary;
pub mod reduction;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use reduction::{All, IntoAxes};
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
```

- [ ] **Step 3: Top-level re-export of `All` from `mlx/src/lib.rs`**

In `mlx/src/lib.rs`, add `pub use ops::All;` after the existing re-exports. Full re-export block:

```rust
pub use array::Array;
pub use broadcast::broadcast_shape;
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
```

This lets users write `use mlx::All;` directly without `use mlx::ops::All;`.

- [ ] **Step 4: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx reduction::tests 2>&1 | tail -10
```
Expected: 5 unit tests pass (`all_returns_none`, `i32_returns_single_element_slice`, `slice_returns_self`, `vec_returns_slice`, `array_literal_returns_slice`).

- [ ] **Step 5: Commit**

```bash
git add mlx/src/ops/ mlx/src/lib.rs
git commit -m "feat(p1b2a): add IntoAxes sealed trait + All marker (5 impls, 5 tests)"
```

---

## Task 3: shim + bridge for 15 reduction functions + 1 reshape + 3 transpose/broadcast + 1 matmul + 4 concat/stack/split (batch, ~22 functions)

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/tests/sys_smoke.rs` (add a few link-tests)

This task batches all the ~22 P1b2a shim functions. They share the established 1-line shim pattern from P0/P1a/P1b1. The novel piece is `MlxArrayVec` opaque type for `split` returns, plus the `&[*const MlxArray]` raw pointer slice for `concatenate`/`stack` varargs.

- [ ] **Step 1: Write failing sys-side smoke tests**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn reduction_sum_links() {
    let a = ffi::array_zeros(&[3, 4], FLOAT32).expect("zeros");
    let _s = mlx_sys::array::ffi::array_sum_all(&a, false).expect("sum_all");
    let _s2 = mlx_sys::array::ffi::array_sum_axis(&a, 0, false).expect("sum_axis");
    let axes: Vec<i32> = vec![0, 1];
    let _s3 = mlx_sys::array::ffi::array_sum_axes(&a, &axes, false).expect("sum_axes");
}

#[test]
fn shape_ops_link() {
    let a = ffi::array_zeros(&[6, 4], FLOAT32).expect("zeros");
    let _r = mlx_sys::array::ffi::array_reshape(&a, &[2, 3, 4]).expect("reshape");
    let _t = mlx_sys::array::ffi::array_transpose(&a).expect("transpose");
    let _ta = mlx_sys::array::ffi::array_transpose_axes(&a, &[1, 0]).expect("transpose_axes");
    let _b = mlx_sys::array::ffi::array_broadcast_to(&a, &[2, 6, 4]).expect("broadcast_to");
}

#[test]
fn matmul_links() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[3, 4], FLOAT32).expect("zeros");
    let _c = mlx_sys::array::ffi::array_matmul(&a, &b).expect("matmul");
}

#[test]
fn split_n_links_returns_vec() {
    let a = ffi::array_zeros(&[6, 4], FLOAT32).expect("zeros");
    let v = mlx_sys::array::ffi::array_split_n(&a, 3, 0).expect("split_n");
    assert_eq!(mlx_sys::array::ffi::split_result_len(&v), 3);
    let _first = mlx_sys::array::ffi::split_result_at(&v, 0).expect("split_result_at");
}

#[test]
fn concatenate_links_with_raw_ptr_slice() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    // Raw pointers cross the bridge as &[*const MlxArray] (cxx 1.0 limitation:
    // can't directly bridge &[&MlxArray]).
    let raw_ptrs: Vec<*const _> = vec![&*a as *const _, &*b as *const _];
    let _c = unsafe {
        mlx_sys::array::ffi::array_concatenate(
            std::slice::from_raw_parts(raw_ptrs.as_ptr().cast(), raw_ptrs.len()),
            0
        )
    }.expect("concatenate");
}
```

(The `concatenate` test uses `unsafe` because we're casting raw pointers; the safe wrapper in Task 8 hides this.)

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke reduction_sum_links 2>&1 | tail -10
```
Expected: FAIL with `cannot find function array_sum_all in module ffi`.

- [ ] **Step 3: Add shim header declarations**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, first add the `MlxArrayVec` typedef next to the existing `using MlxArray = ...` line near the top of the `namespace cxx_mlx { ... }` block. The typedef IS the type definition (no forward declaration needed since `mlx::core::array` is already complete from the existing `#include "mlx/array.h"`):

```cpp
using MlxArrayVec = std::vector<mlx::core::array>;
```

Make sure `#include <vector>` is present at the top of the file (it is — pulled in transitively by `mlx/array.h`, but explicit is safer).

Then add the new function declarations at the end of the `namespace cxx_mlx { ... }` block:

```cpp
// === P1b2a reductions (5 ops × 3 forms = 15) ===

std::unique_ptr<MlxArray> array_sum_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_mean_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_mean_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_mean_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_max_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_max_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_max_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_min_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_min_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_min_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

// argmax: only single-axis variant in MLX (no all-axes / multi-axes overload exposed).
// We expose array_argmax_all via flatten-then-argmax for symmetry.
std::unique_ptr<MlxArray> array_argmax_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_argmax_axis(const MlxArray& a, int32_t axis, bool keepdims);

// === P1b2a shape ops ===

std::unique_ptr<MlxArray> array_reshape(const MlxArray& a, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_transpose(const MlxArray& a);
std::unique_ptr<MlxArray> array_transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> array_broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape);

// Concatenate/stack accept raw pointer slices because cxx 1.0 doesn't bridge
// &[&MlxArray] directly. Caller (Rust safe layer) builds the pointer slice.
std::unique_ptr<MlxArray> array_concatenate(rust::Slice<const MlxArray*> arrays, int32_t axis);
std::unique_ptr<MlxArray> array_stack(rust::Slice<const MlxArray*> arrays, int32_t axis);

// Split returns std::vector<array> wrapped in MlxArrayVec opaque holder.
std::unique_ptr<MlxArrayVec> array_split_n(const MlxArray& a, int32_t num_splits, int32_t axis);
std::unique_ptr<MlxArrayVec> array_split_at(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis);

// MlxArrayVec accessors.
size_t split_result_len(const MlxArrayVec& v);
std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i);

// === P1b2a matmul ===

std::unique_ptr<MlxArray> array_matmul(const MlxArray& a, const MlxArray& b);
```

Also at the top of the file (just below the existing `using MlxArray = mlx::core::array;`), add the `MlxArrayVec` definition:

```cpp
using MlxArrayVec = std::vector<mlx::core::array>;
```

(Place it next to the `MlxArray` typedef so the forward declaration above resolves.)

- [ ] **Step 4: Add shim implementations**

In `mlx-sys/shim/src/array.cc`, add at the end of the existing `namespace cxx_mlx { ... }` block:

```cpp
// === P1b2a reductions ===

std::unique_ptr<MlxArray> array_sum_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::sum(a, keepdims));
}
std::unique_ptr<MlxArray> array_sum_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::sum(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_sum_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::sum(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_mean_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::mean(a, keepdims));
}
std::unique_ptr<MlxArray> array_mean_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::mean(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_mean_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::mean(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_max_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::max(a, keepdims));
}
std::unique_ptr<MlxArray> array_max_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::max(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_max_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::max(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_min_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::min(a, keepdims));
}
std::unique_ptr<MlxArray> array_min_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::min(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_min_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::min(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_argmax_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::argmax(a, keepdims));
}
std::unique_ptr<MlxArray> array_argmax_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::argmax(a, axis, keepdims));
}

// === P1b2a shape ops ===

std::unique_ptr<MlxArray> array_reshape(const MlxArray& a, rust::Slice<const int32_t> shape) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::reshape(a, std::move(s)));
}

std::unique_ptr<MlxArray> array_transpose(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::transpose(a));
}

std::unique_ptr<MlxArray> array_transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::transpose(a, std::move(axes_vec)));
}

std::unique_ptr<MlxArray> array_broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::broadcast_to(a, s));
}

std::unique_ptr<MlxArray> array_concatenate(rust::Slice<const MlxArray*> arrays, int32_t axis) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);  // copy ctor — refcount-shared, cheap
  }
  return std::make_unique<MlxArray>(mlx::core::concatenate(std::move(vec), axis));
}

std::unique_ptr<MlxArray> array_stack(rust::Slice<const MlxArray*> arrays, int32_t axis) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);
  }
  return std::make_unique<MlxArray>(mlx::core::stack(vec, axis));
}

std::unique_ptr<MlxArrayVec> array_split_n(const MlxArray& a, int32_t num_splits, int32_t axis) {
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, num_splits, axis));
}

std::unique_ptr<MlxArrayVec> array_split_at(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis) {
  mlx::core::Shape idx(indices.begin(), indices.end());
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, idx, axis));
}

size_t split_result_len(const MlxArrayVec& v) {
  return v.size();
}

std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i) {
  return std::make_unique<MlxArray>(v.at(i));  // copy ctor — refcount-shared
}

// === P1b2a matmul ===

std::unique_ptr<MlxArray> array_matmul(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::matmul(a, b));
}
```

- [ ] **Step 5: Add cxx bridge declarations**

In `mlx-sys/src/bridge/array.rs`, add inside the `unsafe extern "C++"` block (after the existing P1b1 `array_reciprocal` entry):

```rust
        // === P1b2a opaque type for std::vector<array> returns ===
        type MlxArrayVec;

        // === P1b2a reductions (5 ops × {all, axis, axes}) ===
        fn array_sum_all(a: &MlxArray, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_sum_axis(a: &MlxArray, axis: i32, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_sum_axes(a: &MlxArray, axes: &[i32], keepdims: bool) -> Result<UniquePtr<MlxArray>>;

        fn array_mean_all(a: &MlxArray, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_mean_axis(a: &MlxArray, axis: i32, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_mean_axes(a: &MlxArray, axes: &[i32], keepdims: bool) -> Result<UniquePtr<MlxArray>>;

        fn array_max_all(a: &MlxArray, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_max_axis(a: &MlxArray, axis: i32, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_max_axes(a: &MlxArray, axes: &[i32], keepdims: bool) -> Result<UniquePtr<MlxArray>>;

        fn array_min_all(a: &MlxArray, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_min_axis(a: &MlxArray, axis: i32, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_min_axes(a: &MlxArray, axes: &[i32], keepdims: bool) -> Result<UniquePtr<MlxArray>>;

        fn array_argmax_all(a: &MlxArray, keepdims: bool) -> Result<UniquePtr<MlxArray>>;
        fn array_argmax_axis(a: &MlxArray, axis: i32, keepdims: bool) -> Result<UniquePtr<MlxArray>>;

        // === P1b2a shape ops ===
        fn array_reshape(a: &MlxArray, shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_transpose(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_transpose_axes(a: &MlxArray, axes: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_broadcast_to(a: &MlxArray, shape: &[i32]) -> Result<UniquePtr<MlxArray>>;

        unsafe fn array_concatenate(arrays: &[*const MlxArray], axis: i32) -> Result<UniquePtr<MlxArray>>;
        unsafe fn array_stack(arrays: &[*const MlxArray], axis: i32) -> Result<UniquePtr<MlxArray>>;

        fn array_split_n(a: &MlxArray, num_splits: i32, axis: i32) -> Result<UniquePtr<MlxArrayVec>>;
        fn array_split_at(a: &MlxArray, indices: &[i32], axis: i32) -> Result<UniquePtr<MlxArrayVec>>;

        fn split_result_len(v: &MlxArrayVec) -> usize;
        fn split_result_at(v: &MlxArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;

        // === P1b2a matmul ===
        fn array_matmul(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
```

The `unsafe fn` annotation on `array_concatenate`/`array_stack` reflects that the caller must guarantee the raw pointers are valid for the duration of the call. The Rust safe wrapper (Task 8) constructs the pointer slice from `&[&Array]` and the lifetime is bounded by the function call.

- [ ] **Step 6: Verify all sys-side smoke tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke 2>&1 | tail -15
```
Expected: 11 sys tests pass (6 pre-existing + 5 new).

If link errors mention `mlx::core::sum` or related: confirm `mlx/ops.h` has the exact signature you bound. If the C++ shim fails to compile: re-check `mlx::core::Shape s(begin, end);` constructor and `mlx::core::concatenate(vector<array>&&, int)` overload exist.

- [ ] **Step 7: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p1b2a): add ~22 shim functions for reductions, shape, matmul + MlxArrayVec opaque"
```

---

## Task 4: `ops::sum` with IntoAxes dispatch + Array::sum method + integration tests

**Files:**
- Modify: `mlx/src/ops/reduction.rs` (add `sum` function — Tasks 4 + 5 fill in all 5)
- Modify: `mlx/src/array.rs` (add `Array::sum` method)
- Create: `mlx/tests/p1b2a_reduction.rs`

This task adds just `sum` to validate the dispatch pattern. Task 5 adds the other 4 (mean/max/min/argmax) using the same pattern.

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p1b2a_reduction.rs`:

```rust
use mlx::{ops, All, Array, Dtype};

#[test]
fn sum_all_axes_returns_scalar() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let s = ops::sum(&a, All, false).expect("sum_all");
    assert_eq!(s.size(), 1);
    assert_eq!(s.shape().as_slice(), &[] as &[i32]);
    assert!((s.item::<f32>().expect("item") - 10.0).abs() < 1e-6);
}

#[test]
fn sum_single_axis_negative_index() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = ops::sum(&a, -1, false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2]);
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![6.0, 15.0]);
}

#[test]
fn sum_single_axis_keepdim() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = ops::sum(&a, -1, true).expect("sum");
    assert_eq!(s.shape().as_slice(), &[2, 1]);
}

#[test]
fn sum_multi_axis_slice_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, &[0, 2][..], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
    let v = s.to_vec::<f32>().expect("to_vec");
    assert_eq!(v, vec![8.0_f32, 8.0, 8.0]);
}

#[test]
fn sum_multi_axis_vec_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, vec![0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_multi_axis_array_literal_form() {
    let a = Array::from_slice(&[1.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let s = ops::sum(&a, [0, 2], false).expect("sum");
    assert_eq!(s.shape().as_slice(), &[3]);
}

#[test]
fn sum_method_matches_free_fn() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let by_method = a.sum(-1, false).expect("method");
    let by_freefn = ops::sum(&a, -1, false).expect("free fn");
    assert_eq!(
        by_method.to_vec::<f32>().expect("method to_vec"),
        by_freefn.to_vec::<f32>().expect("freefn to_vec")
    );
}

#[test]
fn sum_dtype_preserved_for_integers() {
    let a = Array::from_slice(&[1_i32, 2, 3], &[3]).expect("from_slice");
    let s = ops::sum(&a, All, false).expect("sum");
    assert_eq!(s.dtype(), Dtype::Int32);
    assert_eq!(s.item::<i32>().expect("item"), 6);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_reduction 2>&1 | tail -10
```
Expected: FAIL with `cannot find function sum in module ops` (or similar — `ops::sum` doesn't exist yet).

- [ ] **Step 3: Add `ops::sum` in `mlx/src/ops/reduction.rs`**

Replace the `#[allow(unused_imports)]` block at the bottom of `mlx/src/ops/reduction.rs` with:

```rust
/// Sum over the specified axes.
///
/// Pass [`All`] to reduce over every axis (yielding a scalar by default),
/// `i32` for a single axis (negative indexing supported), or `&[i32]` /
/// `Vec<i32>` / `[i32; N]` for multiple axes. `keepdim = true` retains
/// reduced axes as size-1.
pub fn sum<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_sum_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_sum_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => mlx_sys::array::ffi::array_sum_axes(a.as_inner(), axes, keepdim),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Add `Array::sum` method in `mlx/src/array.rs`**

In the existing `impl Array { ... }` block (after the unary methods from P1b1), add:

```rust
    /// Sum over the specified axes. See [`crate::ops::sum`].
    pub fn sum<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::sum(self, axes, keepdim)
    }
```

(`crate::ops::IntoAxes` is the bound; use the path so we don't need a top-level `use` statement that might pollute the file.)

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_reduction 2>&1 | grep "test result:"
```
Expected: 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops/reduction.rs mlx/src/array.rs mlx/tests/p1b2a_reduction.rs
git commit -m "feat(p1b2a): ops::sum with IntoAxes dispatch + Array::sum method (8 tests)"
```

---

## Task 5: Remaining 4 reductions (mean / max / min / argmax)

**Files:**
- Modify: `mlx/src/ops/reduction.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b2a_reduction.rs`

This task replicates the `sum` pattern for the other 4 reductions. `argmax` only has all/single-axis variants in MLX (no multi-axis); we expose `axis: i32` form for clarity and `All` for "argmax over flattened" (returns a single scalar index).

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2a_reduction.rs`:

```rust
#[test]
fn mean_basic() {
    let a = Array::from_slice(&[2.0_f32, 4.0, 6.0, 8.0], &[2, 2]).expect("from_slice");
    let m = ops::mean(&a, All, false).expect("mean");
    assert!((m.item::<f32>().expect("item") - 5.0).abs() < 1e-6);

    let m2 = ops::mean(&a, -1, false).expect("mean axis");
    assert_eq!(m2.to_vec::<f32>().expect("to_vec"), vec![3.0_f32, 7.0]);
}

#[test]
fn max_basic() {
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0], &[2, 2]).expect("from_slice");
    assert_eq!(ops::max(&a, All, false).expect("max").item::<f32>().expect("item"), 5.0);

    let m = ops::max(&a, -1, false).expect("max axis");
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![5.0_f32, 3.0]);
}

#[test]
fn min_basic() {
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0], &[2, 2]).expect("from_slice");
    assert_eq!(ops::min(&a, All, false).expect("min").item::<f32>().expect("item"), 1.0);

    let m = ops::min(&a, -1, false).expect("min axis");
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![1.0_f32, 2.0]);
}

#[test]
fn argmax_basic() {
    // [[1, 5, 3], [2, 4, 6]] → argmax(-1) = [1, 2]
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).expect("from_slice");
    let am = ops::argmax(&a, -1, false).expect("argmax");
    // argmax returns Int32 in MLX
    assert_eq!(am.dtype(), Dtype::Uint32);  // MLX 0.32 returns Uint32 for argmax
    assert_eq!(am.to_vec::<u32>().expect("to_vec"), vec![1_u32, 2]);
}

#[test]
fn argmax_all_returns_flat_index() {
    // The single max in [1, 5, 3, 2, 4, 6] is at index 5
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).expect("from_slice");
    let am = ops::argmax(&a, All, false).expect("argmax all");
    assert_eq!(am.item::<u32>().expect("item"), 5);
}

#[test]
fn reduction_methods_match_free_fns() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    assert_eq!(
        a.mean(All, false).expect("mean").item::<f32>().expect("item"),
        ops::mean(&a, All, false).expect("mean").item::<f32>().expect("item"),
    );
    assert_eq!(
        a.max(All, false).expect("max").item::<f32>().expect("item"),
        ops::max(&a, All, false).expect("max").item::<f32>().expect("item"),
    );
    assert_eq!(
        a.min(All, false).expect("min").item::<f32>().expect("item"),
        ops::min(&a, All, false).expect("min").item::<f32>().expect("item"),
    );
}
```

NOTE: The `argmax` dtype is checked as `Dtype::Uint32` here based on MLX 0.32's actual return type. If the test fails on dtype, check `mlx/ops.h:argmax` doc — MLX may return `int32` instead. Adjust the assertion and `to_vec::<u32>` → `to_vec::<i32>` as needed (this will be revealed at test run; don't fight the framework, follow MLX's actual return type).

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_reduction mean_basic 2>&1 | tail -10
```
Expected: FAIL with `cannot find function mean in module ops`.

- [ ] **Step 3: Add `mean`, `max`, `min`, `argmax` free functions in `mlx/src/ops/reduction.rs`**

Append to `mlx/src/ops/reduction.rs` (after the existing `sum`):

```rust
/// Mean over the specified axes. See [`sum`] for axes semantics.
pub fn mean<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_mean_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_mean_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => mlx_sys::array::ffi::array_mean_axes(a.as_inner(), axes, keepdim),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Maximum over the specified axes. See [`sum`] for axes semantics.
pub fn max<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_max_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_max_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => mlx_sys::array::ffi::array_max_axes(a.as_inner(), axes, keepdim),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Minimum over the specified axes. See [`sum`] for axes semantics.
pub fn min<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_min_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_min_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => mlx_sys::array::ffi::array_min_axes(a.as_inner(), axes, keepdim),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Indices of the maximum values along the specified axis. Returns `Uint32`
/// (MLX convention). For [`All`], reduces over the flattened array.
///
/// Multi-axis argmax is not supported by MLX; pass a single `i32` axis or [`All`].
pub fn argmax<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::array_argmax_all(a.as_inner(), keepdim),
        Some([axis]) => mlx_sys::array::ffi::array_argmax_axis(a.as_inner(), *axis, keepdim),
        Some(axes) => {
            return Err(Error::Mlx(format!(
                "argmax does not support multi-axis reduction (got axes={axes:?})"
            )));
        }
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Re-export the 4 new functions from `mlx/src/ops/mod.rs`**

Update the `pub use reduction::...` line in `mlx/src/ops/mod.rs` to:

```rust
pub use reduction::{All, IntoAxes, argmax, max, mean, min, sum};
```

- [ ] **Step 5: Add 4 new methods to `Array` in `mlx/src/array.rs`**

In the existing `impl Array { ... }` block (next to the `sum` method from Task 4), add:

```rust
    /// Mean over the specified axes. See [`crate::ops::mean`].
    pub fn mean<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::mean(self, axes, keepdim)
    }

    /// Maximum over the specified axes. See [`crate::ops::max`].
    pub fn max<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::max(self, axes, keepdim)
    }

    /// Minimum over the specified axes. See [`crate::ops::min`].
    pub fn min<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::min(self, axes, keepdim)
    }

    /// Indices of the maximum values along the specified axis. See [`crate::ops::argmax`].
    pub fn argmax<A: crate::ops::IntoAxes>(&self, axes: A, keepdim: bool) -> Result<Array> {
        crate::ops::argmax(self, axes, keepdim)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_reduction 2>&1 | grep "test result:"
```
Expected: 14 tests pass (8 from Task 4 + 6 from this task).

If `argmax_basic` fails on the `Dtype::Uint32` assertion, MLX may have changed the return dtype. Adjust the test (and the doc comment on `argmax`) to match what MLX actually returns.

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2a_reduction.rs
git commit -m "feat(p1b2a): mean/max/min/argmax reductions + matching Array methods (6 tests)"
```

---

## Task 6: `reshape` with `-1` placeholder inference

**Files:**
- Create: `mlx/src/ops/shape.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1b2a_shape.rs`

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p1b2a_shape.rs`:

```rust
use mlx::{Array, Dtype, Error};

#[test]
fn reshape_explicit_shape() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).expect("from_slice");
    let r = a.reshape(&[2, 3]).expect("reshape");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_minus_one_inferred_at_end() {
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let r = a.reshape(&[2, -1]).expect("reshape inferred");
    assert_eq!(r.shape().as_slice(), &[2, 12]);
}

#[test]
fn reshape_minus_one_inferred_in_middle() {
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let r = a.reshape(&[2, -1, 4]).expect("reshape inferred middle");
    assert_eq!(r.shape().as_slice(), &[2, 3, 4]);
}

#[test]
fn reshape_no_minus_one() {
    let a = Array::from_slice(&[0.0_f32; 6], &[6]).expect("from_slice");
    let r = a.reshape(&[2, 3]).expect("reshape");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
}

#[test]
fn reshape_multiple_minus_ones_errors() {
    let a = Array::from_slice(&[0.0_f32; 24], &[24]).expect("from_slice");
    let result = a.reshape(&[-1, -1, 4]);
    match result {
        Err(Error::Mlx(msg)) => assert!(msg.contains("at most one -1"), "msg: {msg}"),
        other => panic!("expected Error::Mlx, got {other:?}"),
    }
}

#[test]
fn reshape_indivisible_minus_one_errors() {
    // 24 elements / 5 = not integer
    let a = Array::from_slice(&[0.0_f32; 24], &[24]).expect("from_slice");
    let result = a.reshape(&[5, -1]);
    match result {
        Err(Error::Mlx(msg)) => assert!(msg.contains("not divisible") || msg.contains("infer"), "msg: {msg}"),
        other => panic!("expected Error::Mlx, got {other:?}"),
    }
}

#[test]
fn reshape_total_size_mismatch_propagates_from_mlx() {
    let a = Array::from_slice(&[0.0_f32; 6], &[6]).expect("from_slice");
    // Asking for 8 elements when we have 6 → MLX rejects
    let result = a.reshape(&[2, 4]);
    assert!(matches!(result, Err(Error::Mlx(_))));
    let _ = Dtype::Float32;  // silence unused import in this test fn
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape reshape_ 2>&1 | tail -10
```
Expected: FAIL with `no method named reshape found for struct Array`.

- [ ] **Step 3: Create `mlx/src/ops/shape.rs` with `reshape` (and stubs for the rest of Tasks 7-9)**

```rust
//! Shape ops: reshape, transpose family, broadcast_to, concatenate, stack, split.

use smallvec::SmallVec;

use crate::{Array, Error, Result};

/// Reshape an array to the given shape. A single `-1` in the shape is replaced
/// by the inferred size; multiple `-1`s or a non-divisible product return
/// `Err(Error::Mlx)`.
pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    let total: usize = a.size();
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    let resolved: SmallVec<[i32; 8]> = match neg_count {
        0 => shape.iter().copied().collect(),
        1 => {
            let known: usize = shape
                .iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .product();
            if known == 0 || total % known != 0 {
                return Err(Error::Mlx(format!(
                    "reshape: cannot infer -1 dim — total {total} not divisible by product {known} of remaining dims {shape:?}"
                )));
            }
            let inferred = (total / known) as i32;
            shape
                .iter()
                .map(|&d| if d == -1 { inferred } else { d })
                .collect()
        }
        _ => {
            return Err(Error::Mlx(format!(
                "reshape: at most one -1 placeholder allowed, got {neg_count} in {shape:?}"
            )))
        }
    };
    let inner =
        mlx_sys::array::ffi::array_reshape(a.as_inner(), &resolved).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Wire `mod shape;` in `mlx/src/ops/mod.rs`**

Update `mlx/src/ops/mod.rs` to add the new module + re-export:

```rust
pub mod binary;
pub mod reduction;
pub mod shape;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use reduction::{All, IntoAxes, argmax, max, mean, min, sum};
pub use shape::reshape;
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
```

- [ ] **Step 5: Add `Array::reshape` method**

In `mlx/src/array.rs` `impl Array { ... }`, add:

```rust
    /// Reshape this array. See [`crate::ops::reshape`].
    pub fn reshape(&self, shape: &[i32]) -> Result<Array> {
        crate::ops::reshape(self, shape)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape 2>&1 | grep "test result:"
```
Expected: 7 tests pass.

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2a_shape.rs
git commit -m "feat(p1b2a): reshape with -1 placeholder inference (7 tests)"
```

---

## Task 7: `transpose`, `transpose_axes`, `broadcast_to` + `Array::t()`

**Files:**
- Modify: `mlx/src/ops/shape.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b2a_shape.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2a_shape.rs`:

```rust
#[test]
fn transpose_2d_swaps_rows_cols() {
    // [[1, 2, 3], [4, 5, 6]] (2x3) transposed → [[1, 4], [2, 5], [3, 6]] (3x2)
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let t = a.transpose().expect("transpose");
    assert_eq!(t.shape().as_slice(), &[3, 2]);
    assert_eq!(t.to_vec::<f32>().expect("to_vec"), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn t_method_alias_for_transpose() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    let t1 = a.t().expect("t");
    let t2 = a.transpose().expect("transpose");
    assert_eq!(
        t1.to_vec::<f32>().expect("to_vec"),
        t2.to_vec::<f32>().expect("to_vec")
    );
}

#[test]
fn transpose_axes_permute() {
    // [2, 3, 4] permuted by [2, 0, 1] → [4, 2, 3]
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let t = a.transpose_axes(&[2, 0, 1]).expect("transpose_axes");
    assert_eq!(t.shape().as_slice(), &[4, 2, 3]);
}

#[test]
fn broadcast_to_expands_singleton_dim() {
    // [3] broadcast to [2, 3] should replicate the row twice
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let b = a.broadcast_to(&[2, 3]).expect("broadcast_to");
    assert_eq!(b.shape().as_slice(), &[2, 3]);
    assert_eq!(b.to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn broadcast_to_incompatible_shape_errors() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let result = a.broadcast_to(&[2, 4]);
    assert!(matches!(result, Err(Error::Mlx(_))));
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape transpose 2>&1 | tail -10
```
Expected: FAIL with `no method named transpose found for struct Array`.

- [ ] **Step 3: Add `transpose`, `transpose_axes`, `broadcast_to` to `mlx/src/ops/shape.rs`**

Append to `mlx/src/ops/shape.rs`:

```rust
/// Reverse all axes (NumPy `arr.T` equivalent).
pub fn transpose(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_transpose(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Permute axes per the given permutation. `axes` must be a permutation of
/// `[0, a.ndim())`; MLX validates and errors otherwise.
pub fn transpose_axes(a: &Array, axes: &[i32]) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_transpose_axes(a.as_inner(), axes).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Broadcast `a` to the given shape, replicating dims of size 1. The target
/// shape must be broadcast-compatible per NumPy rules.
pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_broadcast_to(a.as_inner(), shape).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Update `mlx/src/ops/mod.rs` re-export**

```rust
pub use shape::{broadcast_to, reshape, transpose, transpose_axes};
```

- [ ] **Step 5: Add 4 new methods to `Array` in `mlx/src/array.rs`**

Append to the existing `impl Array { ... }`:

```rust
    /// Reverse all axes. See [`crate::ops::transpose`].
    pub fn transpose(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Shorthand for [`Array::transpose`]. Standard convention in matrix code.
    pub fn t(&self) -> Result<Array> {
        crate::ops::transpose(self)
    }

    /// Permute axes per the given permutation. See [`crate::ops::transpose_axes`].
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        crate::ops::transpose_axes(self, axes)
    }

    /// Broadcast to the given shape. See [`crate::ops::broadcast_to`].
    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        crate::ops::broadcast_to(self, shape)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape 2>&1 | grep "test result:"
```
Expected: 12 tests pass (7 from Task 6 + 5 from this task).

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2a_shape.rs
git commit -m "feat(p1b2a): transpose/transpose_axes/broadcast_to + Array::t() method (5 tests)"
```

---

## Task 8: `concatenate` and `stack` (varargs via `&[*const MlxArray]`)

**Files:**
- Modify: `mlx/src/ops/shape.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/tests/p1b2a_shape.rs`

The unsafe boundary lives in the safe wrapper: it takes `&[&Array]`, extracts raw pointers (lifetime bounded by the call), and delegates to the unsafe shim.

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2a_shape.rs`:

```rust
#[test]
fn concatenate_along_axis_0() {
    // [2,3] + [3,3] along axis 0 → [5, 3]
    let a = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[2.0_f32; 9], &[3, 3]).expect("from_slice");
    let c = mlx::ops::concatenate(&[&a, &b], 0).expect("concatenate");
    assert_eq!(c.shape().as_slice(), &[5, 3]);
}

#[test]
fn concatenate_along_axis_1() {
    // [2,3] + [2,4] along axis 1 → [2, 7]
    let a = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[2.0_f32; 8], &[2, 4]).expect("from_slice");
    let c = mlx::ops::concatenate(&[&a, &b], 1).expect("concatenate");
    assert_eq!(c.shape().as_slice(), &[2, 7]);
}

#[test]
fn stack_creates_new_axis() {
    // Stack two [2,3] along axis 0 → [2, 2, 3]
    let a = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[2.0_f32; 6], &[2, 3]).expect("from_slice");
    let s = mlx::ops::stack(&[&a, &b], 0).expect("stack");
    assert_eq!(s.shape().as_slice(), &[2, 2, 3]);
}

#[test]
fn stack_along_last_axis() {
    let a = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let b = Array::from_slice(&[3.0_f32, 4.0], &[2]).expect("from_slice");
    let s = mlx::ops::stack(&[&a, &b], -1).expect("stack");
    assert_eq!(s.shape().as_slice(), &[2, 2]);
    // Result column-major in the new axis: [[1, 3], [2, 4]]
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![1.0, 3.0, 2.0, 4.0]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape concatenate 2>&1 | tail -10
```
Expected: FAIL with `no function or associated item named concatenate found for module ops`.

- [ ] **Step 3: Add `concatenate` and `stack` to `mlx/src/ops/shape.rs`**

Append to `mlx/src/ops/shape.rs`:

```rust
/// Concatenate arrays along the given axis. All arrays must have identical
/// shape except along the concatenation axis.
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `arrays` (a slice of
    // `&Array`) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `arrays`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner = unsafe {
        mlx_sys::array::ffi::array_concatenate(&raw, axis)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Stack arrays along a new axis. All arrays must have identical shape; the
/// result has rank `arrays[0].ndim() + 1`.
pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: same as `concatenate` — pointers are bounded by call lifetime.
    let inner = unsafe { mlx_sys::array::ffi::array_stack(&raw, axis) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Update `mlx/src/ops/mod.rs` re-export**

```rust
pub use shape::{broadcast_to, concatenate, reshape, stack, transpose, transpose_axes};
```

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape 2>&1 | grep "test result:"
```
Expected: 16 tests pass (12 + 4 from this task).

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops/ mlx/tests/p1b2a_shape.rs
git commit -m "feat(p1b2a): concatenate + stack via raw pointer slice bridge (4 tests)"
```

---

## Task 9: `split_n` and `split_at` (returns `Vec<Array>` via `MlxArrayVec`)

**Files:**
- Modify: `mlx/src/ops/shape.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/tests/p1b2a_shape.rs`

`split_n` / `split_at` cross the bridge as `MlxArrayVec` opaque holder; the safe wrapper unpacks to `Vec<Array>` Rust-side.

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2a_shape.rs`:

```rust
#[test]
fn split_n_equal_pieces() {
    // [6] split into 3 → 3 arrays of shape [2]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).expect("from_slice");
    let parts = mlx::ops::split_n(&a, 3, 0).expect("split_n");
    assert_eq!(parts.len(), 3);
    for part in &parts {
        assert_eq!(part.shape().as_slice(), &[2]);
    }
    assert_eq!(parts[0].to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_vec::<f32>().expect("to_vec"), vec![3.0, 4.0]);
    assert_eq!(parts[2].to_vec::<f32>().expect("to_vec"), vec![5.0, 6.0]);
}

#[test]
fn split_n_along_axis_1() {
    // [2, 6] split into 3 along axis 1 → 3 arrays of shape [2, 2]
    let a = Array::from_slice(&[0.0_f32; 12], &[2, 6]).expect("from_slice");
    let parts = mlx::ops::split_n(&a, 3, 1).expect("split_n axis 1");
    assert_eq!(parts.len(), 3);
    for part in &parts {
        assert_eq!(part.shape().as_slice(), &[2, 2]);
    }
}

#[test]
fn split_at_indices() {
    // [6] split at indices [2, 4] → arrays of shape [2], [2], [2]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).expect("from_slice");
    let parts = mlx::ops::split_at(&a, &[2, 4], 0).expect("split_at");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_vec::<f32>().expect("to_vec"), vec![3.0, 4.0]);
    assert_eq!(parts[2].to_vec::<f32>().expect("to_vec"), vec![5.0, 6.0]);
}

#[test]
fn split_at_uneven_pieces() {
    // [6] split at indices [1, 4] → arrays of shape [1], [3], [2]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).expect("from_slice");
    let parts = mlx::ops::split_at(&a, &[1, 4], 0).expect("split_at uneven");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape().as_slice(), &[1]);
    assert_eq!(parts[1].shape().as_slice(), &[3]);
    assert_eq!(parts[2].shape().as_slice(), &[2]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape split_n 2>&1 | tail -10
```
Expected: FAIL with `cannot find function split_n in module ops`.

- [ ] **Step 3: Add `split_n` and `split_at` to `mlx/src/ops/shape.rs`**

Append to `mlx/src/ops/shape.rs`:

```rust
/// Split `a` into `num_splits` equal-sized pieces along `axis`. Returns a
/// `Vec<Array>` of length `num_splits`. The split axis size must be evenly
/// divisible by `num_splits`; MLX validates and errors otherwise.
pub fn split_n(a: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>> {
    let v = mlx_sys::array::ffi::array_split_n(a.as_inner(), num_splits, axis)
        .map_err(Error::from)?;
    let len = mlx_sys::array::ffi::split_result_len(&v);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let inner = mlx_sys::array::ffi::split_result_at(&v, i).map_err(Error::from)?;
        out.push(Array::from_inner(inner));
    }
    Ok(out)
}

/// Split `a` at the given indices along `axis`. With `indices = [i, j, ...]`
/// and the split axis size `S`, the result has pieces with sizes
/// `[i, j-i, ..., S - last_idx]`.
pub fn split_at(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    let v = mlx_sys::array::ffi::array_split_at(a.as_inner(), indices, axis)
        .map_err(Error::from)?;
    let len = mlx_sys::array::ffi::split_result_len(&v);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let inner = mlx_sys::array::ffi::split_result_at(&v, i).map_err(Error::from)?;
        out.push(Array::from_inner(inner));
    }
    Ok(out)
}
```

- [ ] **Step 4: Update `mlx/src/ops/mod.rs` re-export**

```rust
pub use shape::{
    broadcast_to, concatenate, reshape, split_at, split_n, stack, transpose, transpose_axes,
};
```

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_shape 2>&1 | grep "test result:"
```
Expected: 20 tests pass (16 + 4 from this task).

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops/ mlx/tests/p1b2a_shape.rs
git commit -m "feat(p1b2a): split_n + split_at returning Vec<Array> via MlxArrayVec opaque (4 tests)"
```

---

## Task 10: `matmul` (covers 2D, batched, and broadcast)

**Files:**
- Create: `mlx/src/ops/matmul.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1b2a_matmul.rs`

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p1b2a_matmul.rs`:

```rust
use mlx::{Array, Error};

#[test]
fn matmul_2d() {
    // [2, 3] @ [3, 4] → [2, 4]
    // a = [[1, 2, 3], [4, 5, 6]], b = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    // a @ b = [[1, 2, 3, 0], [4, 5, 6, 0]]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(
        &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
    )
    .expect("from_slice");
    let c = a.matmul(&b).expect("matmul");
    assert_eq!(c.shape().as_slice(), &[2, 4]);
    assert_eq!(
        c.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]
    );
}

#[test]
fn matmul_3d_batched() {
    // [B=2, S=3, D=4] @ [B=2, D=4, M=5] → [B=2, S=3, M=5]
    let a = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4]).expect("from_slice");
    let b = Array::from_slice(&[0.0_f32; 40], &[2, 4, 5]).expect("from_slice");
    let c = a.matmul(&b).expect("matmul");
    assert_eq!(c.shape().as_slice(), &[2, 3, 5]);
}

#[test]
fn matmul_attention_shape() {
    // [B=2, H=4, S=8, D=16] @ [B=2, H=4, D=16, S=8] → [B=2, H=4, S=8, S=8]
    let q = Array::from_slice(&[0.0_f32; 1024], &[2, 4, 8, 16]).expect("from_slice");
    let k = Array::from_slice(&[0.0_f32; 1024], &[2, 4, 16, 8]).expect("from_slice");
    let scores = q.matmul(&k).expect("matmul");
    assert_eq!(scores.shape().as_slice(), &[2, 4, 8, 8]);
}

#[test]
fn matmul_using_t_for_attention() {
    // Q @ K.t() in attention pattern: [S, D] @ [D, S] → [S, S]
    // For 4D: [B, H, S, D] @ [B, H, S, D].t() = [B, H, S, D] @ [B, H, D, S] = [B, H, S, S]
    let q = Array::from_slice(&[0.0_f32; 1024], &[2, 4, 8, 16]).expect("from_slice");
    let k = Array::from_slice(&[0.0_f32; 1024], &[2, 4, 8, 16]).expect("from_slice");
    let kt = k.t().expect("k.t()");
    assert_eq!(kt.shape().as_slice(), &[16, 8, 4, 2]);  // .t() reverses ALL dims
    // For a proper attention pattern we'd need transpose_axes(&[0, 1, 3, 2])
    let kt_proper = k.transpose_axes(&[0, 1, 3, 2]).expect("transpose_axes");
    assert_eq!(kt_proper.shape().as_slice(), &[2, 4, 16, 8]);
    let scores = q.matmul(&kt_proper).expect("matmul");
    assert_eq!(scores.shape().as_slice(), &[2, 4, 8, 8]);
}

#[test]
fn matmul_inner_dim_mismatch_errors() {
    let a = Array::from_slice(&[0.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[0.0_f32; 8], &[4, 2]).expect("from_slice");  // inner dim 3 != 4
    let result = a.matmul(&b);
    assert!(matches!(result, Err(Error::Mlx(_))));
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_matmul 2>&1 | tail -10
```
Expected: FAIL with `no method named matmul found for struct Array`.

- [ ] **Step 3: Create `mlx/src/ops/matmul.rs`**

```rust
//! Matrix multiplication.
//!
//! `matmul(a, b)` covers all NumPy-/MLX-style matmul cases:
//!
//! - 2D × 2D: standard matrix product `[M, K] @ [K, N] → [M, N]`
//! - Batched: `[B..., M, K] @ [B..., K, N] → [B..., M, N]`
//! - Broadcasting on batch dims: `[B, 1, M, K] @ [1, H, K, N] → [B, H, M, N]`
//!
//! MLX handles all dispatch internally; this is a single FFI thin wrapper.

use crate::{Array, Error, Result};

/// Matrix multiplication. See module docs for shape rules.
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_matmul(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Wire `mod matmul;` and re-export**

In `mlx/src/ops/mod.rs`:

```rust
pub mod binary;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use matmul::matmul;
pub use reduction::{All, IntoAxes, argmax, max, mean, min, sum};
pub use shape::{
    broadcast_to, concatenate, reshape, split_at, split_n, stack, transpose, transpose_axes,
};
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
```

- [ ] **Step 5: Add `Array::matmul` method in `mlx/src/array.rs`**

In `impl Array { ... }`:

```rust
    /// Matrix multiplication. See [`crate::ops::matmul`] for shape rules.
    pub fn matmul(&self, rhs: &Array) -> Result<Array> {
        crate::ops::matmul(self, rhs)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_matmul 2>&1 | grep "test result:"
```
Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2a_matmul.rs
git commit -m "feat(p1b2a): matmul (2D + batched + broadcast) with Array::matmul method (5 tests)"
```

---

## Task 11: Compose tests — softmax, gelu, silu

**Files:**
- Create: `mlx/tests/p1b2a_compose.rs`

This task validates that P0 + P1a + P1b1 + P1b2a together can express the standard activation/normalization functions.

- [ ] **Step 1: Write the integration tests**

Create `mlx/tests/p1b2a_compose.rs`:

```rust
use mlx::{ops, Array, Result};

/// Softmax using max-subtraction trick for numerical stability.
fn softmax(x: &Array, axis: i32) -> Result<Array> {
    let m = ops::max(x, axis, true)?;
    let shifted = (x - &m)?;
    let e = shifted.exp()?;
    let s = ops::sum(&e, axis, true)?;
    &e / &s
}

/// Exact GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu(x: &Array) -> Result<Array> {
    let sqrt_2 = std::f32::consts::SQRT_2;
    let half = (x * 0.5_f32)?;
    let inner = (x / sqrt_2)?.erf()?;
    let one_plus = (&inner + 1.0_f32)?;
    &half * &one_plus
}

/// SiLU (a.k.a. Swish): x * sigmoid(x)
fn silu(x: &Array) -> Result<Array> {
    let s = x.sigmoid()?;
    x * &s
}

#[test]
fn softmax_along_last_axis_sums_to_one() {
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let s = softmax(&x, -1).expect("softmax");
    let v = s.to_vec::<f32>().expect("to_vec");
    let total: f32 = v.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "sum should be ~1.0, got {total}");
    // Each value must be positive.
    for val in &v {
        assert!(*val > 0.0, "softmax value should be positive: {val}");
    }
    // Largest input → largest softmax value.
    assert!(v[2] > v[1] && v[1] > v[0]);
}

#[test]
fn softmax_2d_per_row() {
    // [[1, 2, 3], [4, 5, 6]] softmax along axis -1: each row sums to 1
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = softmax(&x, -1).expect("softmax");
    let v = s.to_vec::<f32>().expect("to_vec");
    assert!((v[0] + v[1] + v[2] - 1.0).abs() < 1e-6);
    assert!((v[3] + v[4] + v[5] - 1.0).abs() < 1e-6);
}

#[test]
fn gelu_at_known_points() {
    // gelu(0) ≈ 0
    let zero = Array::from_slice(&[0.0_f32], &[]).expect("from_slice");
    assert!((gelu(&zero).expect("gelu").item::<f32>().expect("item") - 0.0).abs() < 1e-6);

    // gelu(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) ≈ 0.8413
    let one = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let g = gelu(&one).expect("gelu").item::<f32>().expect("item");
    assert!((g - 0.8413).abs() < 1e-3, "gelu(1) ≈ 0.8413, got {g}");

    // gelu(-1) ≈ -0.1587 (symmetric around 0 in a specific way)
    let neg_one = Array::from_slice(&[-1.0_f32], &[]).expect("from_slice");
    let g_neg = gelu(&neg_one).expect("gelu").item::<f32>().expect("item");
    assert!((g_neg - (-0.1587)).abs() < 1e-3, "gelu(-1) ≈ -0.1587, got {g_neg}");
}

#[test]
fn silu_at_known_points() {
    // silu(0) = 0
    let zero = Array::from_slice(&[0.0_f32], &[]).expect("from_slice");
    assert!((silu(&zero).expect("silu").item::<f32>().expect("item") - 0.0).abs() < 1e-6);

    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    let one = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let s = silu(&one).expect("silu").item::<f32>().expect("item");
    assert!((s - 0.7311).abs() < 1e-3, "silu(1) ≈ 0.7311, got {s}");

    // silu(-2) = -2 * sigmoid(-2) ≈ -2 * 0.1192 ≈ -0.2384
    let neg_two = Array::from_slice(&[-2.0_f32], &[]).expect("from_slice");
    let s_neg = silu(&neg_two).expect("silu").item::<f32>().expect("item");
    assert!((s_neg - (-0.2384)).abs() < 1e-3, "silu(-2) ≈ -0.2384, got {s_neg}");
}
```

- [ ] **Step 2: Verify the integration tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_compose 2>&1 | grep "test result:"
```
Expected: 4 tests pass. If any tolerance fails, the spec values may need adjustment to match MLX's actual numerical precision; expand to `1e-2` if needed (and document why).

- [ ] **Step 3: Commit**

```bash
git add mlx/tests/p1b2a_compose.rs
git commit -m "test(p1b2a): integration tests for softmax / gelu / silu (4 tests)"
```

---

## Task 12: README + final workspace verification

**Files:**
- Modify: `README.md`
- Verify: full workspace test + clippy + doc

- [ ] **Step 1: Update README status line**

In `README.md`, change:

```markdown
**Status:** P1b1 — operators (`+ - * / unary -`) + scalar RHS + 9 element-wise unary ops + NumPy broadcasting. Built on P1a Array foundation.
```

to:

```markdown
**Status:** P1b2a — full op surface for inference primitives: 6 shape ops (reshape with `-1` inference, transpose, transpose_axes, broadcast_to, concatenate, stack, split) + 5 reductions (sum/mean/max/min/argmax via `IntoAxes` trait + `All` marker) + matmul. Compose softmax/gelu/silu. Built on P1b1 operators.
```

- [ ] **Step 2: Add a "Reductions / Shape / Matmul" example section**

Append to `README.md` after the "Operators" section, before "Threading":

````markdown
## Reductions, Shape, Matmul

Reductions accept axes via the `IntoAxes` trait — pass `mlx::All` to reduce
all axes, an `i32` for a single axis, or any of `&[i32]` / `Vec<i32>` /
`[i32; N]` for multiple axes:

```rust
use mlx::{Array, Dtype, All, ops};

fn main() -> mlx::Result<()> {
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let total = ops::sum(&x, All, false)?;          // scalar 21.0
    let row_sums = x.sum(-1, false)?;               // [6.0, 15.0]
    let row_sums_kd = x.sum(-1, true)?;             // [[6.0], [15.0]]

    let reshaped = x.reshape(&[3, 2])?;             // [3, 2]
    let auto = x.reshape(&[2, -1])?;                // -1 inferred → [2, 3]
    let t = x.t()?;                                 // [3, 2] (transpose)

    // Matmul covers 2D, batched, and broadcasting on batch dims.
    let q = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4])?;
    let k = Array::from_slice(&[0.0_f32; 24], &[2, 4, 3])?;
    let scores = q.matmul(&k)?;                     // [2, 3, 3]

    Ok(())
}
```

`softmax`, `gelu`, and `silu` compose directly atop these ops — see
[`mlx/tests/p1b2a_compose.rs`](mlx/tests/p1b2a_compose.rs) for the
canonical implementations.
````

- [ ] **Step 3: Update the Roadmap section**

Change:

```markdown
- ✅ **P1b1** — operators + element-wise unary + broadcasting
- ⏳ **P1b2** — shape ops + reduction + indexing + matmul
```

to:

```markdown
- ✅ **P1b1** — operators + element-wise unary + broadcasting
- ✅ **P1b2a** — shape ops + reduction + matmul (compose softmax/gelu/silu)
- ⏳ **P1b2b** — indexing (take/gather/where/slice) + SDPA integration test
```

- [ ] **Step 4: Run the full workspace test suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:" | head -20
```
Expected: all groups pass with ≥ 100 tests total:

- sys_smoke: 11 (6 P0/P1a/P1b1 + 5 added in Task 3)
- p0_smoke: 2
- p1a_array: 6
- p1a_io: 16
- p1a_thread_safety: 2
- p1b1_ops: 13
- p1b2a_reduction: 14 (8 from Task 4 + 6 from Task 5)
- p1b2a_shape: 20 (7 + 5 + 4 + 4 from Tasks 6–9)
- p1b2a_matmul: 5
- p1b2a_compose: 4
- error tests: 3
- element tests: 1
- broadcast tests: 7
- reduction unit tests: 5
- doc tests: 1 passed, 1 ignored

Total: ≥ 110 tests passing.

- [ ] **Step 5: Run clippy**

```bash
MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -v "^warning: mlx-sys@" | tail -10
```
Expected: clean (only upstream MLX header `cargo:warning=` noise filtered out).

- [ ] **Step 6: Build docs**

```bash
MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | tail -5
```
Expected: `Finished` with no errors.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(p1b2a): Reductions/Shape/Matmul section + status/roadmap update"
```

---

## Acceptance Criteria

P1b2a is complete when:

1. `cargo test --workspace` reports ≥ 110 tests passing across 14 test groups
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean
3. `mlx::ops` exposes 12 new free functions: `sum/mean/max/min/argmax`, `reshape`, `transpose/transpose_axes`, `broadcast_to`, `concatenate/stack`, `split_n/split_at`, `matmul` (plus `IntoAxes` and `All` re-exported)
4. `Array` has matching methods: `sum/mean/max/min/argmax`, `reshape`, `transpose`, `t`, `transpose_axes`, `broadcast_to`, `matmul`
5. `softmax`, `gelu`, and `silu` are tested as composable algorithms over the new ops with numerical-correctness assertions
6. `mlx/src/ops.rs` is gone; replaced by `mlx/src/ops/{binary,unary,shape,reduction,matmul}.rs` per spec A9
7. Reductions accept all 5 axis input forms (`All`, `i32`, `&[i32]`, `Vec<i32>`, `[i32; N]`) via the `IntoAxes` sealed trait
8. `reshape` supports a single `-1` placeholder; multiple `-1`s and indivisible products error with `Error::Mlx`
9. `split_n` / `split_at` return `Vec<Array>` via the new `MlxArrayVec` opaque cross-bridge type
10. README documents the new op surface with a runnable `Reductions, Shape, Matmul` example

When all 10 hold, P1b2a is ready for fast-forward to master and P1b2b brainstorm starts.
