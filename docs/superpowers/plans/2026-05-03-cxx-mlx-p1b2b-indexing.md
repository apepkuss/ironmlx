# cxx-mlx P1b2b (Indexing + SDPA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3 new Element types (`u16`/`u32`/`u64` — closing the P1b2a deferred dtype gap), 6 indexing ops (`where_`/`take`/`take_along_axis`/`slice`/`slice_strided`/`gather`), retro-strengthen `argmax` tests with `u32` numerical assertions, and ship a SDPA integration test (causal mask + softmax + numerical correctness) as P1b2's acceptance gate.

**Architecture:** New `mlx/src/ops/indexing.rs` follows the established 3-layer pattern (free fn / operator-impl / Array method). `where_` does Rust-side broadcast validation through two consecutive `broadcast_shape` calls (cond+x then mid+y). `slice`/`slice_strided` validate per-axis length in Rust. `gather` reuses the P1b2a `&[*const MlxArray]` raw pointer slice bridge for its `vector<array>` indices input. SDPA test composes P1b1 ops + P1b2a (matmul/transpose_axes/sum/max/exp) + P1b2b (where_) into the canonical attention algorithm.

**Tech Stack:** Rust 1.94+, cxx 1.0, MLX C++ 0.32 (already at `$MLX_DIR`), C++20. No new dependencies.

**Branch:** Work on `p1b2b-indexing` (already created off master). MLX install at `$HOME/.local/mlx`; export `MLX_DIR=$HOME/.local/mlx` for every cargo invocation.

---

## File Structure

**New files:**

- `mlx/src/ops/indexing.rs` — 6 indexing free functions + Rust-side validation
- `mlx/tests/p1b2b_dtype_extension.rs` — u16/u32/u64 round-trip tests (~6 tests)
- `mlx/tests/p1b2b_indexing.rs` — per-op integration tests (~12 tests)
- `mlx/tests/p1b2b_sdpa.rs` — 4 SDPA integration tests

**Modified files:**

- `mlx/src/element.rs` — add 3 `element_impl_simple!` calls (u16/u32/u64) + corresponding test additions
- `mlx-sys/src/bridge/array.rs` — add ~24 new shim function declarations (9 dtype + 15 indexing)
- `mlx-sys/shim/include/cxx_mlx_shim/array.h` — add new declarations
- `mlx-sys/shim/src/array.cc` — add new implementations
- `mlx-sys/tests/sys_smoke.rs` — add a few link-test smokes
- `mlx/src/ops/mod.rs` — add `pub mod indexing;` + re-exports
- `mlx/src/array.rs` — add 6 new indexing methods
- `mlx/tests/p1b2a_reduction.rs` — strengthen `argmax_basic` and `argmax_all_returns_flat_index` tests with `to_vec::<u32>()` value assertions (P1b2a backfill — Important #1 from P1b2a final review)
- `README.md` — update Status line + add "Indexing & SDPA" example section

---

## Task 1: shim + bridge for ~24 new functions

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/array.h`
- Modify: `mlx-sys/shim/src/array.cc`
- Modify: `mlx-sys/src/bridge/array.rs`
- Modify: `mlx-sys/tests/sys_smoke.rs`

This task batches all the P1b2b shim functions: 9 for dtype expansion (u16/u32/u64 × from/item/to_vec) + 15 for indexing (where + take + take_axis + take_along_axis + slice_strided + gather + variants). All follow established 1-line patterns.

- [ ] **Step 1: Write failing sys-side smoke tests**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
const UINT32: u8 = 3;

#[test]
fn dtype_extension_u32_links() {
    let data: Vec<u32> = vec![1, 2, 3, 4];
    let _arr = mlx_sys::array::ffi::array_from_u32(&data, &[4]).expect("from_u32");
}

#[test]
fn indexing_ops_link() {
    let a = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let cond = ffi::array_zeros(&[2, 3], 0).expect("zeros bool");  // bool dtype
    let b = ffi::array_zeros(&[2, 3], FLOAT32).expect("zeros");
    let _w = mlx_sys::array::ffi::array_where(&cond, &a, &b).expect("where");
    let _s = mlx_sys::array::ffi::array_slice_strided(&a, &[0, 0], &[2, 2], &[1, 1])
        .expect("slice_strided");
    let indices = mlx_sys::array::ffi::array_from_u32(&[0_u32, 2], &[2]).expect("from_u32");
    let _t = mlx_sys::array::ffi::array_take(&a, &indices, 1).expect("take");
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke dtype_extension_u32_links 2>&1 | tail -10
```
Expected: FAIL with `cannot find function array_from_u32 in module ffi`.

- [ ] **Step 3: Add 9 new dtype shim declarations to `mlx-sys/shim/include/cxx_mlx_shim/array.h`**

Add at the end of the existing `namespace cxx_mlx { ... }` block (after the P1b2a additions):

```cpp
// === P1b2b dtype extension: u16/u32/u64 ===

std::unique_ptr<MlxArray> array_from_u16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u64(rust::Slice<const uint64_t> data, rust::Slice<const int32_t> shape);

uint16_t array_item_u16(const MlxArray& a);
uint32_t array_item_u32(const MlxArray& a);
uint64_t array_item_u64(const MlxArray& a);

rust::Vec<uint16_t> array_to_vec_u16(const MlxArray& a);
rust::Vec<uint32_t> array_to_vec_u32(const MlxArray& a);
rust::Vec<uint64_t> array_to_vec_u64(const MlxArray& a);
```

Then the indexing ops:

```cpp
// === P1b2b indexing ops ===

std::unique_ptr<MlxArray> array_where(const MlxArray& cond, const MlxArray& x, const MlxArray& y);

std::unique_ptr<MlxArray> array_take(const MlxArray& a, const MlxArray& indices, int32_t axis);
std::unique_ptr<MlxArray> array_take_along_axis(const MlxArray& a, const MlxArray& indices, int32_t axis);

std::unique_ptr<MlxArray> array_slice_strided(
    const MlxArray& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides);

std::unique_ptr<MlxArray> array_gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes);
```

(Note the `* const` qualifier on `MlxArray*` for the gather raw pointer slice — discovered in P1b2a Task 3.)

- [ ] **Step 4: Add 9 dtype + 5 indexing shim implementations to `mlx-sys/shim/src/array.cc`**

Add at the end of the existing `namespace cxx_mlx { ... }` block. The dtype implementations follow the existing P1a `array_from_typed` / `array_to_vec_typed` template helpers (already in array.cc):

```cpp
// === P1b2b dtype extension implementations ===

std::unique_ptr<MlxArray> array_from_u16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint16_t>(data.data(), shape, mlx::core::uint16);
}
std::unique_ptr<MlxArray> array_from_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint32_t>(data.data(), shape, mlx::core::uint32);
}
std::unique_ptr<MlxArray> array_from_u64(rust::Slice<const uint64_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint64_t>(data.data(), shape, mlx::core::uint64);
}

uint16_t array_item_u16(const MlxArray& a) { return a.item<uint16_t>(); }
uint32_t array_item_u32(const MlxArray& a) { return a.item<uint32_t>(); }
uint64_t array_item_u64(const MlxArray& a) { return a.item<uint64_t>(); }

rust::Vec<uint16_t> array_to_vec_u16(const MlxArray& a) { return array_to_vec_typed<uint16_t>(a); }
rust::Vec<uint32_t> array_to_vec_u32(const MlxArray& a) { return array_to_vec_typed<uint32_t>(a); }
rust::Vec<uint64_t> array_to_vec_u64(const MlxArray& a) { return array_to_vec_typed<uint64_t>(a); }

// === P1b2b indexing implementations ===

std::unique_ptr<MlxArray> array_where(const MlxArray& cond, const MlxArray& x, const MlxArray& y) {
  return std::make_unique<MlxArray>(mlx::core::where(cond, x, y));
}

std::unique_ptr<MlxArray> array_take(const MlxArray& a, const MlxArray& indices, int32_t axis) {
  return std::make_unique<MlxArray>(mlx::core::take(a, indices, axis));
}

std::unique_ptr<MlxArray> array_take_along_axis(const MlxArray& a, const MlxArray& indices, int32_t axis) {
  return std::make_unique<MlxArray>(mlx::core::take_along_axis(a, indices, axis));
}

std::unique_ptr<MlxArray> array_slice_strided(
    const MlxArray& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides) {
  mlx::core::Shape s_start(start.begin(), start.end());
  mlx::core::Shape s_stop(stop.begin(), stop.end());
  mlx::core::Shape s_strides(strides.begin(), strides.end());
  return std::make_unique<MlxArray>(
      mlx::core::slice(a, std::move(s_start), std::move(s_stop), std::move(s_strides)));
}

std::unique_ptr<MlxArray> array_gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes) {
  std::vector<MlxArray> idx_vec;
  idx_vec.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    idx_vec.push_back(*indices[i]);  // copy ctor — refcount-shared, cheap
  }
  std::vector<int> axes_vec(axes.begin(), axes.end());
  mlx::core::Shape ss(slice_sizes.begin(), slice_sizes.end());
  return std::make_unique<MlxArray>(mlx::core::gather(a, idx_vec, axes_vec, ss));
}
```

- [ ] **Step 5: Add cxx bridge declarations**

In `mlx-sys/src/bridge/array.rs`, add inside the `unsafe extern "C++"` block (after the P1b2a `array_matmul` entry):

```rust
        // === P1b2b dtype extension ===
        fn array_from_u16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u32(data: &[u32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u64(data: &[u64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;

        fn array_item_u16(a: &MlxArray) -> Result<u16>;
        fn array_item_u32(a: &MlxArray) -> Result<u32>;
        fn array_item_u64(a: &MlxArray) -> Result<u64>;

        fn array_to_vec_u16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_u32(a: &MlxArray) -> Result<Vec<u32>>;
        fn array_to_vec_u64(a: &MlxArray) -> Result<Vec<u64>>;

        // === P1b2b indexing ops ===
        fn array_where(cond: &MlxArray, x: &MlxArray, y: &MlxArray) -> Result<UniquePtr<MlxArray>>;
        fn array_take(a: &MlxArray, indices: &MlxArray, axis: i32) -> Result<UniquePtr<MlxArray>>;
        fn array_take_along_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> Result<UniquePtr<MlxArray>>;
        fn array_slice_strided(
            a: &MlxArray,
            start: &[i32],
            stop: &[i32],
            strides: &[i32],
        ) -> Result<UniquePtr<MlxArray>>;
        unsafe fn array_gather(
            a: &MlxArray,
            indices: &[*const MlxArray],
            axes: &[i32],
            slice_sizes: &[i32],
        ) -> Result<UniquePtr<MlxArray>>;
```

(`array_gather` is `unsafe fn` because of the raw pointer slice; the Rust safe wrapper in Task 7 satisfies the lifetime contract.)

- [ ] **Step 6: Verify the smoke tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke 2>&1 | tail -15
```
Expected: 13 sys tests pass (11 pre-existing + 2 new from this task).

- [ ] **Step 7: Commit**

```bash
git add mlx-sys/src/bridge/array.rs mlx-sys/shim/ mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p1b2b): add 9 dtype + 6 indexing shim functions (~24 fns total)"
```

---

## Task 2: Element u16 / u32 / u64 + dtype round-trip tests

**Files:**
- Modify: `mlx/src/element.rs`
- Create: `mlx/tests/p1b2b_dtype_extension.rs`

The macro `element_impl_simple!` exists in `element.rs` from P1a — just add 3 invocations.

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p1b2b_dtype_extension.rs`:

```rust
use mlx::{Array, Dtype, Element};

#[test]
fn u16_round_trip() {
    let data: Vec<u16> = vec![1, 100, 65535, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint16);
    let back: Vec<u16> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u32_round_trip() {
    let data: Vec<u32> = vec![1, 1_000_000, u32::MAX, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint32);
    let back: Vec<u32> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u64_round_trip() {
    let data: Vec<u64> = vec![1, 1_000_000_000_000, u64::MAX, 0];
    let arr = Array::from_slice(&data, &[4]).expect("from_slice");
    assert_eq!(arr.dtype(), Dtype::Uint64);
    let back: Vec<u64> = arr.to_vec().expect("to_vec");
    assert_eq!(back, data);
}

#[test]
fn u32_item_scalar() {
    let arr = Array::from_slice(&[42_u32], &[]).expect("from_slice");
    assert_eq!(arr.item::<u32>().expect("item"), 42);
}

#[test]
fn dtype_const_for_new_types() {
    assert_eq!(<u16 as Element>::DTYPE, Dtype::Uint16);
    assert_eq!(<u32 as Element>::DTYPE, Dtype::Uint32);
    assert_eq!(<u64 as Element>::DTYPE, Dtype::Uint64);
}

#[test]
fn shape_validation_for_new_types() {
    // Length mismatch should produce ShapeMismatch (Rust-side check)
    let result = Array::from_slice(&[1_u32, 2, 3], &[5]);
    assert!(matches!(result, Err(mlx::Error::ShapeMismatch { .. })));
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_dtype_extension 2>&1 | tail -10
```
Expected: FAIL with `the trait bound u16: Element is not satisfied` (or similar for u32/u64).

- [ ] **Step 3: Add 3 `element_impl_simple!` calls in `mlx/src/element.rs`**

In `mlx/src/element.rs`, find the existing block of `element_impl_simple!` calls (after the bool special case, before half::f16). Add three new lines:

```rust
element_impl_simple!(u16, Dtype::Uint16, array_from_u16, array_item_u16, array_to_vec_u16);
element_impl_simple!(u32, Dtype::Uint32, array_from_u32, array_item_u32, array_to_vec_u32);
element_impl_simple!(u64, Dtype::Uint64, array_from_u64, array_item_u64, array_to_vec_u64);
```

Place them adjacent to the existing `element_impl_simple!(u8, ...)` line for grouping.

Also update the `dtype_const_matches_for_each_element` test inside `element.rs` (the `#[cfg(test)] mod tests` block) to assert the 3 new types:

```rust
        assert_eq!(<u16 as Element>::DTYPE, Dtype::Uint16);
        assert_eq!(<u32 as Element>::DTYPE, Dtype::Uint32);
        assert_eq!(<u64 as Element>::DTYPE, Dtype::Uint64);
```

(Add inside the existing `dtype_const_matches_for_each_element` fn alongside the other assertions.)

- [ ] **Step 4: Verify all tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_dtype_extension 2>&1 | grep "test result:"
```
Expected: 6 tests pass.

Also run the unit tests:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx element::tests 2>&1 | grep "test result:"
```
Expected: 1 unit test passes (the updated `dtype_const_matches_for_each_element`).

- [ ] **Step 5: Commit**

```bash
git add mlx/src/element.rs mlx/tests/p1b2b_dtype_extension.rs
git commit -m "feat(p1b2b): Element impls for u16/u32/u64 (closes P1b2a deferred dtype gap)"
```

---

## Task 3: P1b2a backfill — strengthen argmax tests with u32 numerical assertions

**Files:**
- Modify: `mlx/tests/p1b2a_reduction.rs`

Now that `u32` is an `Element` (Task 2), the `argmax_basic` and `argmax_all_returns_flat_index` tests can verify actual index values via `to_vec::<u32>()` and `item::<u32>()`. This was P1b2a's Important #1 follow-up.

- [ ] **Step 1: Read the existing test file to find the two argmax tests**

```bash
grep -n "fn argmax_basic\|fn argmax_all_returns_flat_index" /Volumes/Dev/cxx-mlx/mlx/tests/p1b2a_reduction.rs
```

- [ ] **Step 2: Strengthen `argmax_basic`**

Find the existing `argmax_basic` test and replace its body (after the `let am = ops::argmax(&a, -1, false).expect("argmax");` line, the dtype/shape assertions stay; add a value assertion at the end):

The complete strengthened test:

```rust
#[test]
fn argmax_basic() {
    // [[1, 5, 3], [2, 4, 6]] → argmax(-1) = [1, 2]
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).expect("from_slice");
    let am = ops::argmax(&a, -1, false).expect("argmax");
    assert_eq!(am.dtype(), Dtype::Uint32);
    assert_eq!(am.shape().as_slice(), &[2]);
    assert_eq!(am.to_vec::<u32>().expect("to_vec"), vec![1_u32, 2]);
}
```

- [ ] **Step 3: Strengthen `argmax_all_returns_flat_index`**

The complete strengthened test:

```rust
#[test]
fn argmax_all_returns_flat_index() {
    // The single max in [1, 5, 3, 2, 4, 6] is at flat index 5
    let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).expect("from_slice");
    let am = ops::argmax(&a, All, false).expect("argmax all");
    assert_eq!(am.dtype(), Dtype::Uint32);
    assert_eq!(am.size(), 1);
    assert_eq!(am.shape().as_slice(), &[] as &[i32]);
    assert_eq!(am.item::<u32>().expect("item"), 5);
}
```

- [ ] **Step 4: Verify all P1b2a reduction tests still pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2a_reduction 2>&1 | grep "test result:"
```
Expected: 16 tests pass (existing count from P1b2a, now with stronger argmax assertions).

- [ ] **Step 5: Commit**

```bash
git add mlx/tests/p1b2a_reduction.rs
git commit -m "test(p1b2b): backfill argmax tests with u32 value assertions (P1b2a follow-up)"
```

---

## Task 4: `ops::where_` with broadcast validation

**Files:**
- Create: `mlx/src/ops/indexing.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Create: `mlx/tests/p1b2b_indexing.rs`

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p1b2b_indexing.rs`:

```rust
use mlx::{ops, Array, Error};

#[test]
fn where_basic() {
    // cond = [[true, false], [false, true]], x = [[1, 2], [3, 4]], y = [[10, 20], [30, 40]]
    // result = [[1, 20], [30, 4]]
    let cond_data: Vec<u8> = vec![1, 0, 0, 1];
    let cond = Array::from_slice(&cond_data, &[2, 2]).expect("from_slice cond");
    // cond comes in as u8; rely on MLX's nonzero-as-true semantics
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice x");
    let y = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0], &[2, 2]).expect("from_slice y");
    let r = ops::where_(&cond, &x, &y).expect("where_");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn where_with_broadcasting() {
    // cond [2, 1], x [2, 3], y [3] (scalar broadcast)
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2, 1]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let y = Array::from_slice(&[100.0_f32, 200.0, 300.0], &[3]).expect("from_slice");
    let r = ops::where_(&cond, &x, &y).expect("where_");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    // Row 0: cond=1 → x; Row 1: cond=0 → y
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0]
    );
}

#[test]
fn where_broadcast_mismatch_errors() {
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let y = Array::from_slice(&[1.0_f32; 8], &[2, 4]).expect("from_slice");  // mismatch
    let result = ops::where_(&cond, &x, &y);
    assert!(matches!(result, Err(Error::BroadcastMismatch { .. })), "got {result:?}");
}

#[test]
fn where_method_form() {
    // cond.where_(&x, &y) — self is the condition
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let y = Array::from_slice(&[10.0_f32, 20.0], &[2]).expect("from_slice");
    let r = cond.where_(&x, &y).expect("method form");
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 20.0]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing where_ 2>&1 | tail -10
```
Expected: FAIL with `cannot find function where_ in module ops`.

- [ ] **Step 3: Create `mlx/src/ops/indexing.rs`**

```rust
//! Indexing ops: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`.

use crate::{broadcast, Array, Error, Result};

/// Element-wise conditional select: `cond ? x : y`, with NumPy broadcasting
/// across all three operands.
///
/// `cond` is typically a `bool` array but MLX accepts any numeric dtype
/// (non-zero is treated as true).
///
/// Trailing underscore in the name avoids the Rust `where` keyword.
pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    // Validate broadcast compatibility in two steps: cond+x, then result+y.
    // This produces structured Error::BroadcastMismatch instead of opaque MLX strings.
    let cond_x = broadcast::broadcast_shape(&cond.shape(), &x.shape())?;
    broadcast::broadcast_shape(&cond_x, &y.shape())?;
    let inner = mlx_sys::array::ffi::array_where(cond.as_inner(), x.as_inner(), y.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Wire `mod indexing;` in `mlx/src/ops/mod.rs`**

Update `mlx/src/ops/mod.rs` to add the new module + re-export `where_`:

```rust
pub mod binary;
pub mod indexing;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use indexing::where_;
pub use matmul::matmul;
pub use reduction::{All, IntoAxes, argmax, max, mean, min, sum};
pub use shape::{
    broadcast_to, concatenate, reshape, split_at, split_n, stack, transpose, transpose_axes,
};
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
```

(Note: `where_` is NOT re-exported at the crate root in `lib.rs` to avoid mistaking it for the Rust `where` keyword. Users access via `mlx::ops::where_`.)

- [ ] **Step 5: Add `Array::where_` method**

In `mlx/src/array.rs` `impl Array { ... }`, add (suggested location: after the matmul method):

```rust
    /// Use `self` as the condition mask, selecting from `x` where true and `y` where false.
    /// See [`crate::ops::where_`].
    pub fn where_(&self, x: &Array, y: &Array) -> Result<Array> {
        crate::ops::where_(self, x, y)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing 2>&1 | grep "test result:"
```
Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2b_indexing.rs
git commit -m "feat(p1b2b): ops::where_ with two-step broadcast validation (4 tests)"
```

---

## Task 5: `take` and `take_along_axis`

**Files:**
- Modify: `mlx/src/ops/indexing.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b2b_indexing.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2b_indexing.rs`:

```rust
#[test]
fn take_along_axis_0() {
    // a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], indices = [0, 2], axis = 0
    // result = [[1, 2, 3], [7, 8, 9]]
    let a = Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    )
    .expect("from_slice");
    let indices = Array::from_slice(&[0_u32, 2], &[2]).expect("from_slice");
    let r = ops::take(&a, &indices, 0).expect("take");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn take_along_axis_1() {
    // Same a, indices = [0, 2], axis = 1 → pick cols 0 and 2 → [[1, 3], [4, 6], [7, 9]]
    let a = Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    )
    .expect("from_slice");
    let indices = Array::from_slice(&[0_u32, 2], &[2]).expect("from_slice");
    let r = ops::take(&a, &indices, 1).expect("take");
    assert_eq!(r.shape().as_slice(), &[3, 2]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 3.0, 4.0, 6.0, 7.0, 9.0]
    );
}

#[test]
fn take_along_axis_pytorch_gather_semantics() {
    // a = [[10, 20, 30], [40, 50, 60]], indices same shape, axis = 1
    // indices = [[0, 2, 1], [1, 0, 2]] → result = [[10, 30, 20], [50, 40, 60]]
    let a = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).expect("from_slice");
    let indices_data: Vec<u32> = vec![0, 2, 1, 1, 0, 2];
    let indices = Array::from_slice(&indices_data, &[2, 3]).expect("from_slice");
    let r = ops::take_along_axis(&a, &indices, 1).expect("take_along_axis");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![10.0, 30.0, 20.0, 50.0, 40.0, 60.0]
    );
}

#[test]
fn take_method_form() {
    let a = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let indices = Array::from_slice(&[2_u32, 0], &[2]).expect("from_slice");
    let r = a.take(&indices, 0).expect("method take");
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![30.0, 10.0]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing take 2>&1 | tail -10
```
Expected: FAIL with `cannot find function take in module ops`.

- [ ] **Step 3: Add `take` and `take_along_axis` to `mlx/src/ops/indexing.rs`**

Append to `mlx/src/ops/indexing.rs`:

```rust
/// Take values along `axis` according to a 1-D `indices` array.
///
/// Output shape: same as `a` but with the `axis` dim replaced by `indices.size()`.
/// Indices must be an unsigned integer dtype (u32/u64); MLX validates.
pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_take(a.as_inner(), indices.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Take values where `indices` has the same shape as `a` (per-axis pick).
///
/// Equivalent to PyTorch's `torch.gather`. Output shape = `indices.shape`.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_take_along_axis(a.as_inner(), indices.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Update re-exports in `mlx/src/ops/mod.rs`**

```rust
pub use indexing::{take, take_along_axis, where_};
```

- [ ] **Step 5: Add `Array::take` and `Array::take_along_axis` methods**

In `mlx/src/array.rs` `impl Array { ... }`, add:

```rust
    /// Take values along `axis`. See [`crate::ops::take`].
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take(self, indices, axis)
    }

    /// Per-axis gather (PyTorch `torch.gather`). See [`crate::ops::take_along_axis`].
    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
        crate::ops::take_along_axis(self, indices, axis)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing 2>&1 | grep "test result:"
```
Expected: 8 tests pass (4 + 4 from this task).

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2b_indexing.rs
git commit -m "feat(p1b2b): take + take_along_axis (4 tests)"
```

---

## Task 6: `slice` and `slice_strided` with Rust-side length validation

**Files:**
- Modify: `mlx/src/ops/indexing.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b2b_indexing.rs`

- [ ] **Step 1: Write failing tests**

Append to `mlx/tests/p1b2b_indexing.rs`:

```rust
#[test]
fn slice_basic_2d() {
    // a = 3x4 = [[1..4], [5..8], [9..12]]; slice([1, 1], [3, 3]) → [[6, 7], [10, 11]]
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[3, 4]).expect("from_slice");
    let r = ops::slice(&a, &[1, 1], &[3, 3]).expect("slice");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![6.0, 7.0, 10.0, 11.0]);
}

#[test]
fn slice_full_first_dim() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[2, 3]).expect("from_slice");
    let r = ops::slice(&a, &[0, 1], &[2, 3]).expect("slice");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn slice_strided_step_2() {
    // a = [0..6], slice with stride 2 → [0, 2, 4]
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[6]).expect("from_slice");
    let r = ops::slice_strided(&a, &[0], &[6], &[2]).expect("slice_strided");
    assert_eq!(r.shape().as_slice(), &[3]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![0.0, 2.0, 4.0]);
}

#[test]
fn slice_length_mismatch_errors() {
    let a = Array::from_slice(&[0.0_f32; 6], &[2, 3]).expect("from_slice");
    // Pass start with wrong length (1 instead of 2)
    let result = ops::slice(&a, &[0], &[2, 3]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })), "got {result:?}");
}

#[test]
fn slice_method_form() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[3, 4]).expect("from_slice");
    let r = a.slice(&[0, 0], &[2, 2]).expect("method slice");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![0.0, 1.0, 4.0, 5.0]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing slice 2>&1 | tail -10
```
Expected: FAIL with `cannot find function slice in module ops`.

- [ ] **Step 3: Add `slice` and `slice_strided` to `mlx/src/ops/indexing.rs`**

Append to `mlx/src/ops/indexing.rs`:

```rust
/// Slice with stride 1 along every dimension. `start` and `stop` must each have
/// length equal to `a.ndim()`. Negative indices are supported (per MLX rules).
pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides: Vec<i32> = vec![1; a.ndim()];
    slice_strided(a, start, stop, &strides)
}

/// Slice with explicit per-dim strides. `start`, `stop`, `strides` must all
/// have length equal to `a.ndim()`. Negative indices and negative strides are
/// supported per MLX rules.
pub fn slice_strided(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
    let ndim = a.ndim();
    let actual = vec![start.len() as i32, stop.len() as i32, strides.len() as i32];
    let expected = vec![ndim as i32; 3];
    if start.len() != ndim || stop.len() != ndim || strides.len() != ndim {
        return Err(Error::ShapeMismatch { expected, actual });
    }
    let inner = mlx_sys::array::ffi::array_slice_strided(a.as_inner(), start, stop, strides)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Update re-exports in `mlx/src/ops/mod.rs`**

```rust
pub use indexing::{slice, slice_strided, take, take_along_axis, where_};
```

- [ ] **Step 5: Add `Array::slice` and `Array::slice_strided` methods**

In `mlx/src/array.rs`:

```rust
    /// Slice with stride 1. See [`crate::ops::slice`].
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Result<Array> {
        crate::ops::slice(self, start, stop)
    }

    /// Slice with explicit strides. See [`crate::ops::slice_strided`].
    pub fn slice_strided(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
        crate::ops::slice_strided(self, start, stop, strides)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing 2>&1 | grep "test result:"
```
Expected: 13 tests pass (8 + 5 from this task).

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2b_indexing.rs
git commit -m "feat(p1b2b): slice + slice_strided with Rust-side length validation (5 tests)"
```

---

## Task 7: `gather` (raw pointer slice for vector<array> input)

**Files:**
- Modify: `mlx/src/ops/indexing.rs`
- Modify: `mlx/src/ops/mod.rs`
- Modify: `mlx/src/array.rs`
- Modify: `mlx/tests/p1b2b_indexing.rs`

`gather` is the most flexible / least intuitive of the indexing ops. The Rust safe wrapper hides the raw-pointer-slice unsafe boundary.

- [ ] **Step 1: Write a basic gather test**

Append to `mlx/tests/p1b2b_indexing.rs`:

```rust
#[test]
fn gather_basic_1d_index() {
    // Simple case: gather from a [4, 3] along axis 0 with indices [1, 3]
    // and slice_sizes [1, 3]. Result shape: indices_shape (2,) ++ slice_sizes (1, 3)
    // = [2, 1, 3]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[4, 3]).expect("from_slice");
    let idx = Array::from_slice(&[1_u32, 3], &[2]).expect("from_slice idx");
    let r = ops::gather(&a, &[&idx], &[0], &[1, 3]).expect("gather");
    assert_eq!(r.shape().as_slice(), &[2, 1, 3]);
    // Row 1 of original: [3, 4, 5]; Row 3: [9, 10, 11]
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn gather_method_form() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[3, 2]).expect("from_slice");
    let idx = Array::from_slice(&[0_u32, 2], &[2]).expect("from_slice");
    let r = a.gather(&[&idx], &[0], &[1, 2]).expect("method gather");
    assert_eq!(r.shape().as_slice(), &[2, 1, 2]);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing gather 2>&1 | tail -10
```
Expected: FAIL with `cannot find function gather in module ops`.

- [ ] **Step 3: Add `gather` to `mlx/src/ops/indexing.rs`**

Append:

```rust
/// N-dimensional gather. Picks slices of `a` at the cartesian product of
/// `indices` along `axes`, with each gathered slice sized per `slice_sizes`.
///
/// Returns shape `indices_shape ++ slice_sizes` (concatenation). See MLX docs
/// for full semantics — this is the most flexible / least intuitive indexing op.
pub fn gather(
    a: &Array,
    indices: &[&Array],
    axes: &[i32],
    slice_sizes: &[i32],
) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `indices` (a slice of
    // &Array) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        indices.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `indices`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner = unsafe {
        mlx_sys::array::ffi::array_gather(a.as_inner(), &raw, axes, slice_sizes)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

- [ ] **Step 4: Update re-exports in `mlx/src/ops/mod.rs`**

```rust
pub use indexing::{gather, slice, slice_strided, take, take_along_axis, where_};
```

- [ ] **Step 5: Add `Array::gather` method**

In `mlx/src/array.rs`:

```rust
    /// N-dimensional gather. See [`crate::ops::gather`].
    pub fn gather(
        &self,
        indices: &[&Array],
        axes: &[i32],
        slice_sizes: &[i32],
    ) -> Result<Array> {
        crate::ops::gather(self, indices, axes, slice_sizes)
    }
```

- [ ] **Step 6: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_indexing 2>&1 | grep "test result:"
```
Expected: 15 tests pass (13 + 2 from this task).

- [ ] **Step 7: Commit**

```bash
git add mlx/src/ops/ mlx/src/array.rs mlx/tests/p1b2b_indexing.rs
git commit -m "feat(p1b2b): gather with raw pointer slice bridge for vector<array> input (2 tests)"
```

---

## Task 8: SDPA integration test (full C scope)

**Files:**
- Create: `mlx/tests/p1b2b_sdpa.rs`

This is P1b2's acceptance gate. Tests the full attention algorithm using everything from P0/P1a/P1b1/P1b2a/P1b2b.

- [ ] **Step 1: Create `mlx/tests/p1b2b_sdpa.rs`**

```rust
//! SDPA (scaled dot-product attention) integration tests.
//!
//! Implements the canonical attention algorithm using the cxx-mlx ops from
//! P0/P1a/P1b1/P1b2a/P1b2b. P2 will add `fast::scaled_dot_product_attention`
//! which should match these results numerically.

use mlx::{ops, Array, Result};

/// SDPA: out = softmax((Q @ K.T) * scale + mask) @ V
///
/// Q/K/V: [B, H, S, D]
/// mask: [S, S] additive (-inf in masked positions, 0 elsewhere) — broadcasts on B, H
/// Returns: [B, H, S, D]
fn sdpa(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
) -> Result<Array> {
    // K.transpose(-1, -2): [B, H, S, D] → [B, H, D, S]
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = q.matmul(&kt)?;
    let scaled = (&scores * scale)?;
    let masked = match mask {
        Some(m) => (&scaled + m)?,
        None => scaled,
    };
    // Softmax along last axis
    let m = ops::max(&masked, -1, true)?;
    let shifted = (&masked - &m)?;
    let e = shifted.exp()?;
    let s = ops::sum(&e, -1, true)?;
    let weights = (&e / &s)?;
    weights.matmul(v)
}

/// Build a causal mask of shape [S, S]: 0 on/below diagonal, -inf above.
fn causal_mask(s: usize) -> Result<Array> {
    let mut data = Vec::with_capacity(s * s);
    for i in 0..s {
        for j in 0..s {
            data.push(if j <= i { 0.0_f32 } else { f32::NEG_INFINITY });
        }
    }
    Array::from_slice(&data, &[s as i32, s as i32])
}

#[test]
fn sdpa_no_mask_shape_finite() {
    // [B=1, H=2, S=4, D=8]
    let total: usize = 1 * 2 * 4 * 8;
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.02).collect();
    let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.03).collect();
    let q = Array::from_slice(&q_data, &[1, 2, 4, 8]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 2, 4, 8]).expect("k");
    let v = Array::from_slice(&v_data, &[1, 2, 4, 8]).expect("v");

    let scale = 1.0 / (8.0_f32).sqrt();
    let out = sdpa(&q, &k, &v, None, scale).expect("sdpa");
    assert_eq!(out.shape().as_slice(), &[1, 2, 4, 8]);
    let v_out = out.to_vec::<f32>().expect("to_vec");
    for x in &v_out {
        assert!(x.is_finite(), "non-finite value in SDPA output: {x}");
    }
}

#[test]
fn sdpa_softmax_rows_sum_to_one() {
    // Verify the softmax-of-scores property: the attention weight rows sum to 1.
    // We compute weights directly (without the final V matmul) to inspect.
    let total: usize = 1 * 1 * 3 * 4;
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let q = Array::from_slice(&q_data, &[1, 1, 3, 4]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 1, 3, 4]).expect("k");

    let kt = k.transpose_axes(&[0, 1, 3, 2]).expect("kt");
    let scores = q.matmul(&kt).expect("matmul");
    let scaled = (&scores * (1.0 / 2.0_f32)).expect("scale");
    let m = ops::max(&scaled, -1, true).expect("max");
    let shifted = (&scaled - &m).expect("sub");
    let e = shifted.exp().expect("exp");
    let s = ops::sum(&e, -1, true).expect("sum");
    let weights = (&e / &s).expect("div");

    // Row sums of weights: sum over last axis with keepdim=false → [1, 1, 3]
    let row_sums = ops::sum(&weights, -1, false).expect("row_sums");
    let v = row_sums.to_vec::<f32>().expect("to_vec");
    for sum in &v {
        assert!((sum - 1.0).abs() < 1e-5, "row sum should be ~1.0, got {sum}");
    }
}

#[test]
fn sdpa_causal_mask_zeros_future() {
    // With a causal mask, attention weights for j > i (future positions) should be 0.
    // We compute weights manually (without V matmul) to inspect.
    let s = 4;
    let total: usize = 1 * 1 * s * 4;
    let q_data: Vec<f32> = (0..total).map(|i| 0.1 * (i as f32)).collect();
    let k_data: Vec<f32> = (0..total).map(|i| 0.1 * (i as f32)).collect();
    let q = Array::from_slice(&q_data, &[1, 1, s as i32, 4]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 1, s as i32, 4]).expect("k");
    let mask = causal_mask(s).expect("mask");

    let kt = k.transpose_axes(&[0, 1, 3, 2]).expect("kt");
    let scores = q.matmul(&kt).expect("matmul");
    let scaled = (&scores * 0.5_f32).expect("scale");
    let masked = (&scaled + &mask).expect("add mask");
    let m = ops::max(&masked, -1, true).expect("max");
    let shifted = (&masked - &m).expect("sub");
    let e = shifted.exp().expect("exp");
    let sum_e = ops::sum(&e, -1, true).expect("sum");
    let weights = (&e / &sum_e).expect("div");

    // Reshape to [S, S] for inspection
    let w_2d = weights.reshape(&[s as i32, s as i32]).expect("reshape");
    let v = w_2d.to_vec::<f32>().expect("to_vec");
    // For each row i, positions j > i should be ~0
    for i in 0..s {
        for j in 0..s {
            let val = v[i * s + j];
            if j > i {
                assert!(val.abs() < 1e-6, "w[{i},{j}] should be 0 (causal), got {val}");
            }
        }
    }
}

#[test]
fn sdpa_numerical_match_reference() {
    // Deterministic small input. Q=K=V=I (4x4), scale=1, no mask.
    // softmax(Q @ Q.T) = softmax(I) = each row is uniform 1/N? No — Q @ Q.T with Q=I gives I.
    // softmax of identity row [1, 0, 0, 0] = [exp(1), exp(0), exp(0), exp(0)] / sum
    // = [e, 1, 1, 1] / (e + 3) ≈ [0.4754, 0.1749, 0.1749, 0.1749]
    // Then weights @ I = weights.
    //
    // So output[0] = [0.4754, 0.1749, 0.1749, 0.1749]
    // output[1] = [0.1749, 0.4754, 0.1749, 0.1749]  etc.
    let n = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let identity_2d = Array::from_slice(&data, &[n as i32, n as i32]).expect("identity");
    // Reshape to [B=1, H=1, S=4, D=4]
    let q = identity_2d.reshape(&[1, 1, n as i32, n as i32]).expect("reshape");
    let k = q.clone();
    let v = q.clone();

    let out = sdpa(&q, &k, &v, None, 1.0).expect("sdpa");
    let result = out.to_vec::<f32>().expect("to_vec");

    let e = std::f32::consts::E;
    let norm = e + 3.0;
    let expected_diag = e / norm;
    let expected_off = 1.0 / norm;

    // Check diagonal of each row equals expected_diag, off-diagonal equals expected_off
    for i in 0..n {
        for j in 0..n {
            let actual = result[i * n + j];
            let expected = if i == j { expected_diag } else { expected_off };
            assert!(
                (actual - expected).abs() < 1e-3,
                "out[{i},{j}] expected {expected}, got {actual}"
            );
        }
    }
}
```

- [ ] **Step 2: Verify all 4 SDPA tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p1b2b_sdpa 2>&1 | grep "test result:"
```
Expected: 4 tests pass.

If a tolerance fails (1e-5 or 1e-3 etc), document it and relax — MLX numerical implementation can drift across versions.

- [ ] **Step 3: Commit**

```bash
git add mlx/tests/p1b2b_sdpa.rs
git commit -m "test(p1b2b): SDPA integration tests — shape/finite, softmax sum=1, causal mask, numerical reference (4 tests)"
```

---

## Task 9: README + final workspace verification

**Files:**
- Modify: `README.md`
- Verify: full workspace test + clippy + doc

- [ ] **Step 1: Update README status line**

In `README.md`, change:

```markdown
**Status:** P1b2a — full op surface for inference primitives: ...
```

to:

```markdown
**Status:** P1b2b — P1b complete. Adds 3 new dtypes (`u16`/`u32`/`u64`), 6 indexing ops (`where_`/`take`/`take_along_axis`/`slice`/`slice_strided`/`gather`), and SDPA integration tests (causal mask, softmax row-sum, numerical correctness). Built on P1b2a shape/reduction/matmul. Full design in [`docs/superpowers/specs/`](docs/superpowers/specs/).
```

- [ ] **Step 2: Add an "Indexing & SDPA" section to `README.md`**

Add after the "Reductions, Shape, Matmul" section, before the "Threading" section:

````markdown
## Indexing & SDPA

`mlx::ops::where_` (trailing underscore, since `where` is a Rust keyword) selects element-wise from two arrays based on a condition mask. `take` / `take_along_axis` index along an axis (NumPy / PyTorch semantics). `slice` and `slice_strided` extract sub-arrays Python-style:

```rust
use mlx::{ops, Array};

fn main() -> mlx::Result<()> {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let cond = Array::from_slice(&[1_u8, 0, 1, 0, 1, 0], &[2, 3])?;
    let zeros = Array::from_slice(&[0.0_f32; 6], &[2, 3])?;

    let _picked = ops::where_(&cond, &a, &zeros)?;            // element-wise select
    let idx = Array::from_slice(&[0_u32, 2], &[2])?;
    let _cols = a.take(&idx, 1)?;                              // [2, 2] — pick cols 0 and 2
    let _sub = a.slice(&[0, 1], &[2, 3])?;                     // [2, 2] — rows 0..2, cols 1..3
    Ok(())
}
```

A complete SDPA (scaled dot-product attention) implementation composing matmul, transpose, mask-add, softmax, and matmul lives in [`mlx/tests/p1b2b_sdpa.rs`](mlx/tests/p1b2b_sdpa.rs). It's the canonical test that all of P1 (P0 + P1a + P1b1 + P1b2a + P1b2b) integrates correctly. P2's `fast::scaled_dot_product_attention` will match these numerics.
````

- [ ] **Step 3: Update the Roadmap**

Change:

```markdown
- ✅ **P1b2a** — shape ops + reduction + matmul (compose softmax/gelu/silu)
- ⏳ **P1b2b** — indexing (take/gather/where/slice) + SDPA integration test
```

to:

```markdown
- ✅ **P1b2a** — shape ops + reduction + matmul (compose softmax/gelu/silu)
- ✅ **P1b2b** — indexing (take/take_along_axis/where/slice/gather) + u16/u32/u64 dtypes + SDPA integration
- 🎉 **P1 complete** — full inference primitives ready
```

- [ ] **Step 4: Run the full workspace test suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:" | head -20
```
Expected: ≥ 145 tests passing across 17 test groups:

- sys_smoke: 13 (11 pre-existing + 2 from Task 1)
- p0_smoke: 2
- p1a_array: 6
- p1a_io: 16
- p1a_thread_safety: 2
- p1b1_ops: 13
- p1b2a_compose: 4
- p1b2a_matmul: 5
- p1b2a_reduction: 16
- p1b2a_shape: 20
- p1b2b_dtype_extension: 6 (Task 2)
- p1b2b_indexing: 15 (Tasks 4+5+6+7)
- p1b2b_sdpa: 4 (Task 8)
- error tests: 3
- element tests: 1 (with u16/u32/u64 added)
- broadcast tests: 7
- reduction unit tests: 5
- doc tests: 1 passed, 1 ignored

- [ ] **Step 5: Run clippy**

```bash
MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -v "^warning: mlx-sys@" | tail -10
```
Expected: clean (only upstream MLX header noise filtered out).

- [ ] **Step 6: Build docs**

```bash
MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | tail -5
```
Expected: `Finished` with no errors.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(p1b2b): Indexing & SDPA section + status/roadmap (P1 complete)"
```

---

## Acceptance Criteria

P1b2b is complete when:

1. `cargo test --workspace` reports ≥ 145 tests passing across 17 test groups
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean
3. `mlx::Element` is implemented for `u16`, `u32`, `u64` (Element count = 13)
4. `mlx::ops` exposes 6 new indexing functions: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`
5. `Array` has matching methods: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`
6. P1b2a's argmax tests are strengthened with `to_vec::<u32>()` numerical assertions
7. SDPA integration tests pass: shape+finite, softmax row sums to 1, causal mask zeros future, numerical match against deterministic reference
8. README documents the new ops with a runnable Indexing example, and points to the canonical SDPA test

When all 8 hold, P1b2b is ready for fast-forward to master, marking P1 complete. P2 brainstorm starts next (fast ops + io + transforms).
