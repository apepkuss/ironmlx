# cxx-mlx P1b1 (Operators + Element-wise Unary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 binary operators (`Add`/`Sub`/`Mul`/`Div`) + `Neg` with all 4 reference combinations + scalar RHS for all 10 Element types, 9 element-wise unary ops (`exp`/`log`/`sqrt`/`tanh`/`sigmoid`/`square`/`rsqrt`/`erf`/`reciprocal`) with both free-fn and method styles, NumPy broadcasting validated in Rust producing structured `Error::BroadcastMismatch`, and a reusable `broadcast_shape` helper for P1b2 reductions.

**Architecture:** Three new files split by responsibility. `broadcast.rs` holds the shape-inference algorithm + unit tests. `ops.rs` is the source-of-truth for free functions (5 binary + 9 unary). `ops_impl.rs` adapts free functions to operator traits via `forward_ref_binop!` macro. `Array` gets 9 thin method wrappers in `array.rs`. All operators return `Result<Array>` because broadcast/dtype/MLX errors all surface immediately. Scalar RHS dispatch constructs a 1-element `Array` Rust-side, avoiding 50 per-dtype scalar shim functions.

**Tech Stack:** Rust 1.94+, cxx 1.0, MLX C++ 0.32 (already at `$MLX_DIR`), C++20. No new dependencies.

**Branch:** Work on `p1b-ops` (already created off master). MLX install at `$HOME/.local/mlx`; export `MLX_DIR=$HOME/.local/mlx` for every cargo invocation.

---

## File Structure

**New files:**

- `mlx/src/broadcast.rs` — `broadcast_shape(lhs: &[i32], rhs: &[i32]) -> Result<SmallVec<[i32; 8]>>` + 6 unit tests
- `mlx/src/ops.rs` — 5 binary free functions + 9 unary free functions + module-level docs
- `mlx/src/ops_impl.rs` — `forward_ref_binop!` macro + `Add`/`Sub`/`Mul`/`Div` impl for `Array`/`&Array` × all 4 ref combos + scalar RHS impls for all 10 Element types + `Neg` impl
- `mlx/tests/p1b1_ops.rs` — integration tests (binary, broadcast, scalar RHS, unary, neg, dtype error path, "compose softmax" sanity)

**Modified files:**

- `mlx/src/lib.rs` — `mod broadcast; mod ops; mod ops_impl;` + `pub use ops;` (let users write `mlx::ops::exp(&a)`)
- `mlx/src/array.rs` — add 9 unary methods (each 1 line), keep existing impl block layout
- `mlx-sys/src/bridge/array.rs` — add 14 new function declarations, all `Result<UniquePtr<MlxArray>>`
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — add 14 declarations
- `mlx-sys/shim/src/array.cc` — add 14 one-line implementations
- `README.md` — add an "Operators" subsection demonstrating `&a + &b`, scalar RHS, chained unary

---

## Task 1: shim + bridge for 14 new ops

**Files:**
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/tests/sys_smoke.rs` (add a smoke test that link works)

This task adds the C++ shim + cxx bridge for all 14 ops at once. They follow an identical pattern (all take `const MlxArray&`, return `unique_ptr<MlxArray>`, all `Result`-wrapped per the P1a rule), so batching avoids 14 separate ceremony rounds.

- [ ] **Step 1: Write the failing sys-side smoke test**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn binary_add_links() {
    let a = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    let b = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    let _c = mlx_sys::array::ffi::array_add(&a, &b).expect("add should succeed");
}

#[test]
fn unary_exp_links() {
    let a = ffi::array_zeros(&[3], FLOAT32).expect("zeros");
    let _e = mlx_sys::array::ffi::array_exp(&a).expect("exp should succeed");
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke binary_add_links 2>&1 | tail -10
```
Expected: FAIL with `cannot find function array_add in module ffi`.

- [ ] **Step 3: Add 14 declarations to the shim header**

In `mlx-sys/shim/include/cxx_mlx_shim/array.h`, add after the existing `array_to_vec_*` block (inside `namespace cxx_mlx`):

```cpp
// Binary element-wise ops (broadcasting handled by MLX after Rust-side
// shape validation in mlx::broadcast::broadcast_shape).
std::unique_ptr<MlxArray> array_add(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_subtract(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_multiply(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_divide(const MlxArray& a, const MlxArray& b);

// Unary element-wise ops.
std::unique_ptr<MlxArray> array_negative(const MlxArray& a);
std::unique_ptr<MlxArray> array_exp(const MlxArray& a);
std::unique_ptr<MlxArray> array_log(const MlxArray& a);
std::unique_ptr<MlxArray> array_sqrt(const MlxArray& a);
std::unique_ptr<MlxArray> array_tanh(const MlxArray& a);
std::unique_ptr<MlxArray> array_sigmoid(const MlxArray& a);
std::unique_ptr<MlxArray> array_square(const MlxArray& a);
std::unique_ptr<MlxArray> array_rsqrt(const MlxArray& a);
std::unique_ptr<MlxArray> array_erf(const MlxArray& a);
std::unique_ptr<MlxArray> array_reciprocal(const MlxArray& a);
```

- [ ] **Step 4: Add 14 implementations to the shim source**

In `mlx-sys/shim/src/array.cc`, add after the existing `array_to_vec_*` block (inside `namespace cxx_mlx { ... }`):

```cpp
// === P1b1 binary element-wise ops ===

std::unique_ptr<MlxArray> array_add(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::add(a, b));
}
std::unique_ptr<MlxArray> array_subtract(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::subtract(a, b));
}
std::unique_ptr<MlxArray> array_multiply(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::multiply(a, b));
}
std::unique_ptr<MlxArray> array_divide(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::divide(a, b));
}

// === P1b1 unary element-wise ops ===

std::unique_ptr<MlxArray> array_negative(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::negative(a));
}
std::unique_ptr<MlxArray> array_exp(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::exp(a));
}
std::unique_ptr<MlxArray> array_log(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::log(a));
}
std::unique_ptr<MlxArray> array_sqrt(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::sqrt(a));
}
std::unique_ptr<MlxArray> array_tanh(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::tanh(a));
}
std::unique_ptr<MlxArray> array_sigmoid(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::sigmoid(a));
}
std::unique_ptr<MlxArray> array_square(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::square(a));
}
std::unique_ptr<MlxArray> array_rsqrt(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::rsqrt(a));
}
std::unique_ptr<MlxArray> array_erf(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::erf(a));
}
std::unique_ptr<MlxArray> array_reciprocal(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::reciprocal(a));
}
```

- [ ] **Step 5: Add 14 declarations to the cxx bridge**

In `mlx-sys/src/bridge/array.rs`, add inside the `unsafe extern "C++"` block (after the `array_to_vec_*` block):

```rust
        // Binary ops (P1b1) — Result-wrapped per the shim throw rule.
        fn array_add(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_subtract(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_multiply(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_divide(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;

        // Unary ops (P1b1) — Result-wrapped (MLX may throw on dtype not supported,
        // e.g. sqrt on integer types).
        fn array_negative(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_exp(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_log(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_sqrt(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_tanh(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_sigmoid(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_square(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_rsqrt(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_erf(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_reciprocal(a: &MlxArray) -> Result<UniquePtr<MlxArray>>;
```

- [ ] **Step 6: Verify the smoke tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke 2>&1 | tail -15
```
Expected: 6 sys tests pass (4 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p1b1): add 14 shim functions for binary + unary element-wise ops"
```

---

## Task 2: `broadcast_shape` helper

**Files:**
- Create: `mlx/src/broadcast.rs`
- Modify: `mlx/src/lib.rs`

NumPy-style broadcasting:
1. Right-align the shapes
2. Missing dims on the left are treated as 1
3. For each pair `(a, b)`, output dim is `max(a, b)` if `a == b ∨ a == 1 ∨ b == 1`; otherwise error

Returns `SmallVec<[i32; 8]>` (matching `Array::shape()`'s return type from P1a).

- [ ] **Step 1: Create `mlx/src/broadcast.rs` with the failing tests inline**

```rust
//! NumPy-style broadcasting shape inference.
//!
//! Used by binary operators (`Add`/`Sub`/`Mul`/`Div`) before dispatching to MLX
//! to produce structured `Error::BroadcastMismatch` errors with `lhs`/`rhs`
//! fields, instead of relying on MLX's English exception strings.
//!
//! The same algorithm will be reused by P1b2 reductions (computing keepdim
//! shapes) and `broadcast_to` op.

use smallvec::SmallVec;

use crate::{Error, Result};

/// Compute the broadcast result shape of two operand shapes per NumPy rules.
///
/// Returns `Err(Error::BroadcastMismatch)` if the shapes are incompatible.
pub fn broadcast_shape(lhs: &[i32], rhs: &[i32]) -> Result<SmallVec<[i32; 8]>> {
    let n = lhs.len().max(rhs.len());
    let mut out = SmallVec::<[i32; 8]>::with_capacity(n);
    for i in 0..n {
        // Right-align: treat missing leading dims as 1.
        let a = lhs.get(lhs.len().wrapping_sub(n - i)).copied().unwrap_or(1);
        let b = rhs.get(rhs.len().wrapping_sub(n - i)).copied().unwrap_or(1);
        let dim = match (a, b) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(Error::BroadcastMismatch {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            }
        };
        out.push(dim);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_shapes() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn missing_leading_dim_is_one() {
        // [2, 3] vs [3] → [2, 3] (rhs treated as [1, 3])
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn one_dim_expands_in_middle() {
        // [2, 1, 4] vs [3, 4] → right-align: [2, 1, 4] vs [_, 3, 4] → [2, 3, 4]
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 4]).unwrap().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn scalar_broadcasts_to_anything() {
        // empty shape (scalar) vs [2, 3] → [2, 3]
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap().as_slice(), &[2, 3]);
        assert_eq!(broadcast_shape(&[2, 3], &[]).unwrap().as_slice(), &[2, 3]);
    }

    #[test]
    fn both_scalars() {
        let result = broadcast_shape(&[], &[]).unwrap();
        assert_eq!(result.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn incompatible_dim_errors() {
        // [2, 3] vs [2, 4] → mismatch at axis 1 (neither is 1)
        let err = broadcast_shape(&[2, 3], &[2, 4]).unwrap_err();
        match err {
            Error::BroadcastMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![2, 4]);
            }
            other => panic!("expected BroadcastMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rank_mismatch_with_incompatible_dim() {
        // [3] vs [2, 4] → right-align: [_, 3] vs [2, 4] → mismatch at axis 1
        let err = broadcast_shape(&[3], &[2, 4]).unwrap_err();
        assert!(matches!(err, Error::BroadcastMismatch { .. }));
    }
}
```

- [ ] **Step 2: Wire up in `mlx/src/lib.rs`**

Add `mod broadcast;` after `mod array;` and `pub use broadcast::broadcast_shape;` (so binary op impls in Task 3+ can reach it; also exposes it for P1b2 reuse).

```rust
mod array;
mod broadcast;
mod dtype;
mod element;
mod error;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
```

- [ ] **Step 3: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx broadcast::tests 2>&1 | tail -10
```
Expected: 7 broadcast tests pass.

- [ ] **Step 4: Commit**

```bash
git add mlx/src/broadcast.rs mlx/src/lib.rs
git commit -m "feat(p1b1): add NumPy-style broadcast_shape helper with 7 unit tests"
```

---

## Task 3: `ops::add` — first binary free function

**Files:**
- Create: `mlx/src/ops.rs`
- Modify: `mlx/src/lib.rs`
- Create: `mlx/tests/p1b1_ops.rs`

This task creates `ops.rs` with just `add` (validating the broadcasting + shim wiring works end-to-end). Tasks 5/6/7 add the rest with the same pattern.

- [ ] **Step 1: Write the failing test**

Create `mlx/tests/p1b1_ops.rs`:

```rust
use mlx::{Array, Dtype, Error};

#[test]
fn add_same_shape() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let c = mlx::ops::add(&a, &b).expect("add");
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0]);
}

#[test]
fn add_broadcast_scalar_shape() {
    // [2, 3] + [3] should broadcast to [2, 3]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let c = mlx::ops::add(&a, &b).expect("add");
    assert_eq!(c.shape().as_slice(), &[2, 3]);
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn add_broadcast_mismatch_err() {
    let a = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[1.0_f32; 8], &[2, 4]).expect("from_slice");
    let result = mlx::ops::add(&a, &b);
    match result {
        Err(Error::BroadcastMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![2, 3]);
            assert_eq!(rhs, vec![2, 4]);
        }
        other => panic!("expected BroadcastMismatch, got {other:?}"),
    }
}
```

- [ ] **Step 2: Verify it fails**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops add_ 2>&1 | tail -10
```
Expected: FAIL with `failed to resolve: could not find ops in mlx`.

- [ ] **Step 3: Create `mlx/src/ops.rs` with `add`**

```rust
//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.

use crate::{broadcast, Array, Error, Result};

/// Element-wise addition with NumPy broadcasting.
pub fn add(a: &Array, b: &Array) -> Result<Array> {
    // Validate broadcast compatibility before crossing the FFI boundary so
    // we can return Error::BroadcastMismatch with structured lhs/rhs fields.
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Wire `mod ops;` in `mlx/src/lib.rs`**

Add `mod ops;` and `pub use ops;`:

```rust
mod array;
mod broadcast;
mod dtype;
mod element;
mod error;
pub mod ops;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
```

(Note `pub mod ops;` instead of `mod ops;` so users can write `mlx::ops::add(&a, &b)`.)

- [ ] **Step 5: Verify the tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops 2>&1 | tail -10
```
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops.rs mlx/src/lib.rs mlx/tests/p1b1_ops.rs
git commit -m "feat(p1b1): ops::add free function with broadcast validation"
```

---

## Task 4: `Add` operator trait + `forward_ref_binop!` macro

**Files:**
- Create: `mlx/src/ops_impl.rs`
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/tests/p1b1_ops.rs`

This task validates the macro pattern with just `Add`. Task 5 macro-bulks Sub/Mul/Div/Neg.

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b1_ops.rs`:

```rust
#[test]
fn add_operator_all_ref_combos() {
    let a = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0], &[2]).expect("from_slice");

    // All four reference combinations should compile and produce same result.
    let r1 = (&a + &b).expect("&a + &b");
    let r2 = (a.clone() + &b).expect("a + &b");
    let r3 = (&a + b.clone()).expect("&a + b");
    let r4 = (a.clone() + b.clone()).expect("a + b");

    let expected = vec![11.0_f32, 22.0];
    assert_eq!(r1.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r2.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r3.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r4.to_vec::<f32>().expect("to_vec"), expected);
}
```

- [ ] **Step 2: Verify it fails**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops add_operator 2>&1 | tail -10
```
Expected: FAIL with `cannot add &Array to &Array` (the trait isn't implemented).

- [ ] **Step 3: Create `mlx/src/ops_impl.rs` with the macro and `Add`**

```rust
//! Operator trait impls (`Add`/`Sub`/`Mul`/`Div`/`Neg`) for `Array`.
//!
//! All trait methods return `Result<Array>` (`type Output = Result<Array>`)
//! because broadcasting validation, dtype mismatch, or MLX-side errors all
//! surface immediately. User code threads `?` through expressions:
//!
//!     let y = (&a + &b)?.matmul(&w)?;
//!
//! The `forward_ref_binop!` macro generates the 3 variant impls (`Array op
//! Array`, `Array op &Array`, `&Array op Array`) by delegating to the
//! `&Array op &Array` impl (which holds the actual logic).

use std::ops::Add;

use crate::{ops, Array, Result};

/// Generate the 3 by-value/by-ref variant impls for a binary operator.
///
/// Pattern: write the canonical `impl Trait<&Array> for &Array` body once;
/// the macro forwards `Array` operands to `&Array` via `&self` / `&other`.
macro_rules! forward_ref_binop {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait<Array> for &Array {
            type Output = Result<Array>;
            fn $method(self, other: Array) -> Self::Output {
                std::ops::$trait::$method(self, &other)
            }
        }
        impl std::ops::$trait<&Array> for Array {
            type Output = Result<Array>;
            fn $method(self, other: &Array) -> Self::Output {
                std::ops::$trait::$method(&self, other)
            }
        }
        impl std::ops::$trait<Array> for Array {
            type Output = Result<Array>;
            fn $method(self, other: Array) -> Self::Output {
                std::ops::$trait::$method(&self, &other)
            }
        }
    };
}

// === Add ===

impl Add<&Array> for &Array {
    type Output = Result<Array>;
    fn add(self, other: &Array) -> Self::Output {
        ops::add(self, other)
    }
}
forward_ref_binop!(Add, add);
```

- [ ] **Step 4: Wire `mod ops_impl;` in `mlx/src/lib.rs`**

Add after `pub mod ops;`:

```rust
pub mod ops;
mod ops_impl;
```

(Note `mod ops_impl;` is private — the trait impls become available globally just by being in scope.)

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops 2>&1 | grep "test result:"
```
Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops_impl.rs mlx/src/lib.rs mlx/tests/p1b1_ops.rs
git commit -m "feat(p1b1): impl Add for Array (4 ref combos via forward_ref_binop! macro)"
```

---

## Task 5: Remaining 4 ops via the same pattern (Sub, Mul, Div, Neg)

**Files:**
- Modify: `mlx/src/ops.rs`
- Modify: `mlx/src/ops_impl.rs`
- Modify: `mlx/tests/p1b1_ops.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b1_ops.rs`:

```rust
#[test]
fn sub_mul_div_basic() {
    let a = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let b = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");

    let s = (&a - &b).expect("sub");
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![9.0, 18.0, 27.0]);

    let m = (&a * &b).expect("mul");
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![10.0, 40.0, 90.0]);

    let d = (&a / &b).expect("div");
    assert_eq!(d.to_vec::<f32>().expect("to_vec"), vec![10.0, 10.0, 10.0]);
}

#[test]
fn neg_basic() {
    let a = Array::from_slice(&[1.0_f32, -2.0, 3.0], &[3]).expect("from_slice");
    let n = (-&a).expect("neg &");
    assert_eq!(n.to_vec::<f32>().expect("to_vec"), vec![-1.0, 2.0, -3.0]);
    let n2 = (-a).expect("neg owned");
    assert_eq!(n2.to_vec::<f32>().expect("to_vec"), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn neg_on_unsigned_returns_err() {
    // bool/u8 don't support negation in MLX → routed through Result.
    let a = Array::from_slice(&[1_u8, 2, 3], &[3]).expect("from_slice");
    let result = -&a;
    assert!(matches!(result, Err(Error::Mlx(_))), "got {result:?}");
}
```

- [ ] **Step 2: Verify it fails**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops sub_mul_div_basic 2>&1 | tail -10
```
Expected: FAIL with `cannot subtract &Array from &Array`.

- [ ] **Step 3: Add `subtract`, `multiply`, `divide`, `negative` to `ops.rs`**

Append to `mlx/src/ops.rs`:

```rust
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

/// Element-wise negation. May error on unsigned/bool dtypes.
pub fn negative(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_negative(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Add Sub/Mul/Div/Neg trait impls to `ops_impl.rs`**

Append to `mlx/src/ops_impl.rs`:

```rust
use std::ops::{Div, Mul, Neg, Sub};

// === Sub / Mul / Div ===

impl Sub<&Array> for &Array {
    type Output = Result<Array>;
    fn sub(self, other: &Array) -> Self::Output { ops::subtract(self, other) }
}
forward_ref_binop!(Sub, sub);

impl Mul<&Array> for &Array {
    type Output = Result<Array>;
    fn mul(self, other: &Array) -> Self::Output { ops::multiply(self, other) }
}
forward_ref_binop!(Mul, mul);

impl Div<&Array> for &Array {
    type Output = Result<Array>;
    fn div(self, other: &Array) -> Self::Output { ops::divide(self, other) }
}
forward_ref_binop!(Div, div);

// === Neg ===

impl Neg for &Array {
    type Output = Result<Array>;
    fn neg(self) -> Self::Output { ops::negative(self) }
}

impl Neg for Array {
    type Output = Result<Array>;
    fn neg(self) -> Self::Output { ops::negative(&self) }
}
```

The `Sub/Mul/Div` use cases need their imports added at the top of the file. Update the existing `use std::ops::Add;` line to:

```rust
use std::ops::Add;
use std::ops::{Div, Mul, Neg, Sub};
```

(Or merge into one `use std::ops::{Add, Div, Mul, Neg, Sub};` — either works.)

Also, the test references `Error` in `neg_on_unsigned_returns_err`. Make sure `mlx/tests/p1b1_ops.rs`'s top imports include it:

```rust
use mlx::{Array, Dtype, Error};
```

(This was already present from Task 3's import list.)

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops 2>&1 | grep "test result:"
```
Expected: 7 tests pass (3 from Task 3 + 1 from Task 4 + 3 from this task).

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops.rs mlx/src/ops_impl.rs mlx/tests/p1b1_ops.rs
git commit -m "feat(p1b1): add Sub/Mul/Div/Neg operators (free fn + trait impls)"
```

---

## Task 6: Scalar RHS for all 4 binary operators

**Files:**
- Modify: `mlx/src/ops_impl.rs`
- Modify: `mlx/tests/p1b1_ops.rs`

Scalar RHS dispatches by constructing a 1-element scalar `Array` Rust-side, then calling the existing 2-arg shim. This avoids 50 per-dtype scalar shim functions.

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b1_ops.rs`:

```rust
#[test]
fn scalar_rhs_f32() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let r = (&a + 10.0_f32).expect("scalar add");
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![11.0, 12.0, 13.0]);

    let r2 = (&a * 2.0_f32).expect("scalar mul");
    assert_eq!(r2.to_vec::<f32>().expect("to_vec"), vec![2.0, 4.0, 6.0]);
}

#[test]
fn scalar_rhs_i32_on_owned() {
    let a = Array::from_slice(&[1_i32, 2, 3], &[3]).expect("from_slice");
    let r = (a - 1_i32).expect("scalar sub on owned");
    assert_eq!(r.to_vec::<i32>().expect("to_vec"), vec![0, 1, 2]);
}

#[test]
fn scalar_rhs_half_f16() {
    let a = Array::from_slice(
        &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)], &[2]
    ).expect("from_slice");
    let r = (&a + half::f16::from_f32(0.5)).expect("scalar add f16");
    let v = r.to_vec::<half::f16>().expect("to_vec");
    assert!((v[0].to_f32() - 1.5).abs() < 1e-3);
    assert!((v[1].to_f32() - 2.5).abs() < 1e-3);
}
```

- [ ] **Step 2: Verify it fails**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops scalar_rhs 2>&1 | tail -10
```
Expected: FAIL with `cannot add f32 to &Array` (no `Add<f32>` impl yet).

- [ ] **Step 3: Add scalar-RHS impls via macro**

Append to `mlx/src/ops_impl.rs`:

```rust
use crate::Element;

/// Generate `impl Trait<T: Element> for &Array` and `for Array` by constructing
/// a 1-element scalar `Array` from the RHS scalar and delegating to the
/// `&Array op &Array` impl above.
///
/// Spec A4: this avoids 50 per-dtype scalar shim functions. Cost is a small
/// per-call allocation; broadcasting in MLX makes the actual op cheap.
macro_rules! impl_scalar_rhs {
    ($trait:ident, $method:ident) => {
        impl<T: Element> std::ops::$trait<T> for &Array {
            type Output = Result<Array>;
            fn $method(self, rhs: T) -> Self::Output {
                let scalar = Array::from_slice(&[rhs], &[])?;
                std::ops::$trait::$method(self, &scalar)
            }
        }
        impl<T: Element> std::ops::$trait<T> for Array {
            type Output = Result<Array>;
            fn $method(self, rhs: T) -> Self::Output {
                let scalar = Array::from_slice(&[rhs], &[])?;
                std::ops::$trait::$method(&self, &scalar)
            }
        }
    };
}

impl_scalar_rhs!(Add, add);
impl_scalar_rhs!(Sub, sub);
impl_scalar_rhs!(Mul, mul);
impl_scalar_rhs!(Div, div);
```

- [ ] **Step 4: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops 2>&1 | grep "test result:"
```
Expected: 10 tests pass (7 + 3 from this task).

- [ ] **Step 5: Commit**

```bash
git add mlx/src/ops_impl.rs mlx/tests/p1b1_ops.rs
git commit -m "feat(p1b1): scalar RHS for all 4 binary operators (Element-generic)"
```

---

## Task 7: 9 unary ops (free fn + Array methods)

**Files:**
- Modify: `mlx/src/ops.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b1_ops.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b1_ops.rs`:

```rust
#[test]
fn unary_numerical_correctness() {
    let zero = Array::from_slice(&[0.0_f32], &[]).expect("from_slice");
    assert!((zero.exp().expect("exp").item::<f32>().expect("item") - 1.0).abs() < 1e-6);
    assert!((zero.erf().expect("erf").item::<f32>().expect("item") - 0.0).abs() < 1e-6);

    let one = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    assert!((one.log().expect("log").item::<f32>().expect("item") - 0.0).abs() < 1e-6);
    assert!((one.sqrt().expect("sqrt").item::<f32>().expect("item") - 1.0).abs() < 1e-6);
    assert!((one.tanh().expect("tanh").item::<f32>().expect("item") - 0.7615942).abs() < 1e-6);
    assert!((one.sigmoid().expect("sigmoid").item::<f32>().expect("item") - 0.7310586).abs() < 1e-6);
    assert!((one.reciprocal().expect("reciprocal").item::<f32>().expect("item") - 1.0).abs() < 1e-6);

    let three = Array::from_slice(&[3.0_f32], &[]).expect("from_slice");
    assert!((three.square().expect("square").item::<f32>().expect("item") - 9.0).abs() < 1e-6);

    let four = Array::from_slice(&[4.0_f32], &[]).expect("from_slice");
    assert!((four.rsqrt().expect("rsqrt").item::<f32>().expect("item") - 0.5).abs() < 1e-6);
}

#[test]
fn unary_method_matches_free_fn() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let by_method = a.exp().expect("method");
    let by_freefn = mlx::ops::exp(&a).expect("free fn");
    assert_eq!(
        by_method.to_vec::<f32>().expect("method to_vec"),
        by_freefn.to_vec::<f32>().expect("freefn to_vec")
    );
}

#[test]
fn unary_chain_composes() {
    // Compute (exp(x) - 1) / 2  for x = [0.0, 1.0]; expected ≈ [0.0, 0.859]
    let x = Array::from_slice(&[0.0_f32, 1.0], &[2]).expect("from_slice");
    let r = ((&x.exp().expect("exp") - 1.0_f32).expect("sub") / 2.0_f32).expect("div");
    let v = r.to_vec::<f32>().expect("to_vec");
    assert!((v[0] - 0.0).abs() < 1e-6);
    assert!((v[1] - 0.85914).abs() < 1e-3);
}
```

- [ ] **Step 2: Verify it fails**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops unary_ 2>&1 | tail -10
```
Expected: FAIL with `no method named exp found for struct Array`.

- [ ] **Step 3: Add 9 unary free functions to `ops.rs`**

Append to `mlx/src/ops.rs`:

```rust
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

- [ ] **Step 4: Add 9 unary methods to `Array` in `mlx/src/array.rs`**

Add inside the existing `impl Array { ... }` block (suggested location: after `to_vec`):

```rust
/// Element-wise natural exponential. See [`crate::ops::exp`].
pub fn exp(&self) -> Result<Array> { crate::ops::exp(self) }

/// Element-wise natural logarithm. See [`crate::ops::log`].
pub fn log(&self) -> Result<Array> { crate::ops::log(self) }

/// Element-wise square root. See [`crate::ops::sqrt`].
pub fn sqrt(&self) -> Result<Array> { crate::ops::sqrt(self) }

/// Element-wise hyperbolic tangent. See [`crate::ops::tanh`].
pub fn tanh(&self) -> Result<Array> { crate::ops::tanh(self) }

/// Element-wise sigmoid. See [`crate::ops::sigmoid`].
pub fn sigmoid(&self) -> Result<Array> { crate::ops::sigmoid(self) }

/// Element-wise x^2. See [`crate::ops::square`].
pub fn square(&self) -> Result<Array> { crate::ops::square(self) }

/// Element-wise 1/sqrt(x). See [`crate::ops::rsqrt`].
pub fn rsqrt(&self) -> Result<Array> { crate::ops::rsqrt(self) }

/// Element-wise error function. See [`crate::ops::erf`].
pub fn erf(&self) -> Result<Array> { crate::ops::erf(self) }

/// Element-wise 1/x. See [`crate::ops::reciprocal`].
pub fn reciprocal(&self) -> Result<Array> { crate::ops::reciprocal(self) }
```

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b1_ops 2>&1 | grep "test result:"
```
Expected: 13 tests pass (10 + 3 from this task).

- [ ] **Step 6: Commit**

```bash
git add mlx/src/ops.rs mlx/src/array.rs mlx/tests/p1b1_ops.rs
git commit -m "feat(p1b1): 9 element-wise unary ops (free fn + Array methods)"
```

---

## Task 8: Final verification + README polish

**Files:**
- Modify: `README.md`
- Verify: full workspace test + clippy + doc

- [ ] **Step 1: Add an "Operators" section to `README.md`**

Find the existing "Threading" section, and add the following section right BEFORE it:

````markdown
## Operators

`mlx::Array` supports the standard arithmetic operators with all 4
reference combinations (`a + b`, `&a + b`, `a + &b`, `&a + &b`) and
scalar RHS for any `Element` type. Operators return `Result<Array>`
because broadcasting validation, dtype mismatch, or MLX-side errors
all surface as recoverable Rust errors:

```rust
use mlx::{Array, Dtype};

# fn main() -> mlx::Result<()> {
let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3])?;
let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3])?;

// Binary ops with all reference combos
let r1 = (&a + &b)?;          // most common
let r2 = (&a * 2.0_f32)?;     // scalar RHS

// Chained unary (free fn or method form)
let y = (&a.exp()? - 1.0_f32)?;
let z = mlx::ops::sigmoid(&a)?;

// Negation
let n = (-&a)?;
# Ok(())
# }
```

NumPy-style broadcasting is validated in Rust before the FFI call;
incompatible shapes return `Err(Error::BroadcastMismatch { lhs, rhs })`
with structured fields rather than an opaque MLX exception string.

**No scalar LHS** (`1.0 - &a`): blocked by Rust's orphan rule.
Equivalent expressions: `(-&a)? + 1.0`, or `Array::from_slice(&[1.0], &[])? - a`.

Available unary ops: `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `square`,
`rsqrt`, `erf`, `reciprocal` — sufficient to compose `softmax`, `gelu`
(via `0.5 * x * (1 + erf(x / sqrt(2)))`), and `silu` once P1b2 adds the
needed reductions.
````

Also update the status line and roadmap in the README:

Change:
```markdown
**Status:** P1a — Array foundation ...
```
to:
```markdown
**Status:** P1b1 — operators (`+ - * / unary -`) + scalar RHS + 9 element-wise unary ops + NumPy broadcasting. Built on P1a Array foundation.
```

And update the roadmap section's `P1a` line to add a `✅ P1b1` line:

```markdown
- ✅ **P1a** — Array foundation (Element trait, 10 dtypes, from_slice/item/to_vec, Clone/Debug, Send, SmallVec shape)
- ✅ **P1b1** — operators + element-wise unary + broadcasting
- ⏳ **P1b2** — shape ops + reduction + indexing + matmul
```

- [ ] **Step 2: Run the full test suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:"
```
Expected: all groups pass — total ≥ 57 tests:

- sys_smoke: 6 (4 P0/P1a existing + 2 added in Task 1)
- p0_smoke: 2
- p1a_array: 6
- p1a_io: 16
- p1a_thread_safety: 2
- error: 3
- element: 1
- doctest: 1
- broadcast (new): 7
- p1b1_ops (new): 13

- [ ] **Step 3: Run clippy**

```bash
MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -10
```
Expected: clean (only upstream MLX header `cargo:warning=` noise).

- [ ] **Step 4: Build docs**

```bash
MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | tail -5
```
Expected: `Finished` with no errors.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(p1b1): add Operators section + update status/roadmap"
```

---

## Acceptance Criteria

P1b1 is complete when:

1. `cargo test --workspace` reports all tests passing — at least 48 across 9 test groups
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean
3. `mlx::ops::*` exposes 14 free functions: 5 binary (`add`/`subtract`/`multiply`/`divide`/`negative`) + 9 unary (`exp`/`log`/`sqrt`/`tanh`/`sigmoid`/`square`/`rsqrt`/`erf`/`reciprocal`)
4. `Array` has `Add`/`Sub`/`Mul`/`Div`/`Neg` operator impls covering all 4 reference combinations + scalar RHS for any `T: Element`
5. `Array` has 9 unary methods (`a.exp()`, `a.sigmoid()`, etc.) each delegating to the corresponding free fn
6. `mlx::broadcast::broadcast_shape` is `pub` and produces structured `Error::BroadcastMismatch` on incompatible shapes (no MLX exception strings)
7. README documents the operator API with reference combos, scalar RHS, broadcasting, and the no-scalar-LHS workaround

When all 7 hold, P1b1 is ready for merge to master and P1b2 brainstorm starts.
