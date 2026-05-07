# cxx-mlx P3a: `fast::metal_kernel` binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind `mlx::core::fast::metal_kernel` to cxx-mlx with chained-builder construction (`MetalKernel::builder(...).inputs(...).source(...).build()`) + typestate-protected dispatch builder (`kernel.dispatch_builder().inputs(...).grid(...)....dispatch()`). Outputs returned as `ArrayVec` (P6 reuse), taken by index.

**Architecture:** Three-layer FFI as elsewhere in cxx-mlx. C++ shim adds opaque `MetalKernelInner` + `ShapesVec` + a cxx-friendly `TemplateArgC` struct. Rust bridge declares `metal_kernel_build` / `metal_kernel_dispatch` + `shapes_vec_*` API. Rust safe API splits into `mlx/src/fast/metal_kernel/{mod.rs,dispatch.rs}` after restructuring `mlx/src/fast.rs` → `mlx/src/fast/mod.rs`. Typestate enforces 5 mandatory dispatch fields (inputs / output_shapes / output_dtypes / grid / threadgroup) at compile time.

**Tech Stack:** Rust 2021 + cxx 1.0 + MLX C++ (`mlx::core::fast::metal_kernel`). New dev dependency: `trybuild = "1"` for compile-fail tests. Spec: [docs/superpowers/specs/2026-05-06-cxx-mlx-p3a-metal-kernel-design.md](../specs/2026-05-06-cxx-mlx-p3a-metal-kernel-design.md).

---

## Conventions Recap

- **TDD per task**: write failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before commit** (`.claude/CLAUDE.md`):
  ```
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```
  `MLX_DIR=$HOME/.local/mlx` required for any test that exercises MLX FFI (every test in this plan does).
- **MLX source location**: `/Volumes/Dev/mlx` (not the conda install). When verifying `mlx::core::fast::metal_kernel` signatures, read from there.
- **Each task ends green**: workspace `cargo test --release` passes before commit.
- **Commit messages ASCII-safe**.
- **No backwards-compat code** per `.claude/CLAUDE.md`.

---

## File Structure (after P3a)

```
mlx-sys/
├── shim/
│   ├── include/cxx_mlx_shim/fast.h            # +MetalKernelInner, ShapesVec, TemplateArgC, build/dispatch decls
│   └── src/fast.cc                             # +impls
├── src/bridge/fast.rs                          # +metal_kernel_build/dispatch + shapes_vec_*
└── tests/sys_smoke.rs                          # +metal_kernel_build_links + metal_kernel_dispatch_links

mlx/
├── Cargo.toml                                  # +trybuild dev-dep
├── src/
│   ├── lib.rs                                  # +pub use re-exports
│   ├── fast.rs                                 # MOVED → fast/mod.rs
│   └── fast/                                   # NEW (replaces fast.rs)
│       ├── mod.rs                              # existing rms_norm/layer_norm/rope/sdpa + pub mod metal_kernel
│       └── metal_kernel/
│           ├── mod.rs                          # MetalKernel, MetalKernelBuilder, TemplateArg
│           └── dispatch.rs                     # DispatchBuilder + Set/Unset markers + setter impls
└── tests/
    ├── trybuild/                               # NEW — compile-fail tests
    │   ├── ui/
    │   │   ├── metal_kernel_missing_inputs.rs
    │   │   ├── metal_kernel_missing_inputs.stderr
    │   │   ├── metal_kernel_missing_output_shapes.rs
    │   │   ├── metal_kernel_missing_output_shapes.stderr
    │   │   ├── metal_kernel_missing_output_dtypes.rs
    │   │   ├── metal_kernel_missing_output_dtypes.stderr
    │   │   ├── metal_kernel_missing_grid.rs
    │   │   ├── metal_kernel_missing_grid.stderr
    │   │   ├── metal_kernel_missing_threadgroup.rs
    │   │   └── metal_kernel_missing_threadgroup.stderr
    │   └── trybuild.rs                         # cargo-test entry
    └── p3a_metal_kernel.rs                     # integration tests (simple_add, multi_output)
```

---

## Task 1: shim opaque types + `ShapesVec` API

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h` (add `MetalKernelInner`, `ShapesVec`, `TemplateArgC` decls + `shapes_vec_*` API)
- Modify: `mlx-sys/shim/src/fast.cc` (impl `shapes_vec_*`)
- Modify: `mlx-sys/src/bridge/fast.rs` (declare `ShapesVec` opaque type + 3 functions)
- Modify: `mlx-sys/tests/sys_smoke.rs` (smoke link test)

### Goal

Add `MetalKernelInner` and `ShapesVec` opaque structs (forward-declared in header so they can appear as cxx::bridge `type`), plus `TemplateArgC` cxx-shared struct, plus `ShapesVec` push/count API. T1 only adds the types and ShapesVec API — `metal_kernel_build` / `dispatch` come in T2 and T3.

### Steps

- [ ] **Step 1.1: Append opaque type forward decls to `mlx-sys/shim/include/cxx_mlx_shim/fast.h`**

After the existing `fast_*` function declarations (right before the closing `}  // namespace cxx_mlx`), add:

```cpp
// === P3a metal_kernel ===

#include <vector>
#include <string>
#include "mlx/fast.h"

// Opaque types crossing cxx (declared here, defined in fast.cc).
struct MetalKernelInner {
  mlx::core::fast::CustomKernelFunction fn;
};

struct ShapesVec {
  std::vector<mlx::core::Shape> shapes;
};

// === ShapesVec API ===
std::unique_ptr<ShapesVec> shapes_vec_new();
void shapes_vec_push(ShapesVec& v, rust::Slice<const int32_t> shape);
size_t shapes_vec_count(const ShapesVec& v);
```

The `MetalKernelInner` type is referenced now (so cxx::bridge can declare it) but not yet used by any function (T2 adds `metal_kernel_build`).

- [ ] **Step 1.2: Implement `shapes_vec_*` in `mlx-sys/shim/src/fast.cc`**

Append at the end of the file (before `}  // namespace cxx_mlx`):

```cpp
// === P3a ShapesVec ===

std::unique_ptr<ShapesVec> shapes_vec_new() {
  return std::make_unique<ShapesVec>();
}

void shapes_vec_push(ShapesVec& v, rust::Slice<const int32_t> shape) {
  v.shapes.emplace_back(shape.begin(), shape.end());
}

size_t shapes_vec_count(const ShapesVec& v) {
  return v.shapes.size();
}
```

- [ ] **Step 1.3: Add `TemplateArgC` shared struct + `ShapesVec` opaque + `shapes_vec_*` to bridge**

Modify `mlx-sys/src/bridge/fast.rs` — at the top of the `pub mod ffi` block (inside the `#[cxx::bridge(namespace = "cxx_mlx")]` macro), add a shared struct above the existing `unsafe extern "C++"` block:

```rust
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    /// cxx-friendly encoding of `mlx::core::fast::TemplateArg` (variant<int, bool, Dtype>).
    /// `kind`: 0=Int, 1=Bool, 2=Dtype (`int_val` carries the dtype repr).
    struct TemplateArgC {
        name: String,
        kind: u8,
        int_val: i32,
        bool_val: bool,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type MetalKernelInner;
        type ShapesVec;

        // === P3a ShapesVec ===
        fn shapes_vec_new() -> UniquePtr<ShapesVec>;
        fn shapes_vec_push(v: Pin<&mut ShapesVec>, shape: &[i32]);
        fn shapes_vec_count(v: &ShapesVec) -> usize;

        // (existing fast_* fns continue below, unchanged)
        unsafe fn fast_rms_norm(/* ... */) -> Result<UniquePtr<MlxArray>>;
        // ... (other existing fns kept)
    }
}
```

> **Note**: `TemplateArgC` is used at the cxx::bridge level (in T3 dispatch fn) but defined here so it's available now. `MetalKernelInner` opaque is also declared now even though no fn returns/accepts it yet — both types are surfaced for T2/T3 to consume without re-modifying the bridge module.

- [ ] **Step 1.4: Add sys_smoke link tests**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn shapes_vec_links() {
    use mlx_sys::fast::ffi;
    let mut v = ffi::shapes_vec_new();
    assert_eq!(ffi::shapes_vec_count(&v), 0);
    ffi::shapes_vec_push(v.pin_mut(), &[2, 3, 4]);
    ffi::shapes_vec_push(v.pin_mut(), &[8]);
    assert_eq!(ffi::shapes_vec_count(&v), 2);
}
```

- [ ] **Step 1.5: Run gate + commit**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx-sys --test sys_smoke shapes_vec_links

cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release

git add -A
git commit -m "feat(p3a): shim opaque types + ShapesVec API"
```

Expected: 1 new test passes; gate clean.

---

## Task 2: `metal_kernel_build` — shim + bridge + smoke

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h` (add `metal_kernel_build` decl)
- Modify: `mlx-sys/shim/src/fast.cc` (impl)
- Modify: `mlx-sys/src/bridge/fast.rs` (bridge entry)
- Modify: `mlx-sys/tests/sys_smoke.rs` (smoke test with trivial kernel source)

### Goal

Bind `mlx::core::fast::metal_kernel(...)` factory function. Returns `UniquePtr<MetalKernelInner>` holding the `CustomKernelFunction` callable. Verifies a trivial kernel compiles.

### Steps

- [ ] **Step 2.1: Append `metal_kernel_build` decl to `fast.h`**

Right after the `shapes_vec_*` decls:

```cpp
std::unique_ptr<MetalKernelInner> metal_kernel_build(
    rust::Str name,
    rust::Slice<const rust::String> input_names,
    rust::Slice<const rust::String> output_names,
    rust::Str source,
    rust::Str header,
    bool ensure_row_contiguous,
    bool atomic_outputs);
```

- [ ] **Step 2.2: Implement `metal_kernel_build` in `fast.cc`**

Append (before `}  // namespace cxx_mlx`):

```cpp
// === P3a metal_kernel_build ===

std::unique_ptr<MetalKernelInner> metal_kernel_build(
    rust::Str name,
    rust::Slice<const rust::String> input_names,
    rust::Slice<const rust::String> output_names,
    rust::Str source,
    rust::Str header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  std::vector<std::string> in_names;
  in_names.reserve(input_names.size());
  for (const auto& s : input_names) {
    in_names.emplace_back(s);
  }
  std::vector<std::string> out_names;
  out_names.reserve(output_names.size());
  for (const auto& s : output_names) {
    out_names.emplace_back(s);
  }
  auto kernel = mlx::core::fast::metal_kernel(
      std::string(name),
      in_names,
      out_names,
      std::string(source),
      std::string(header),
      ensure_row_contiguous,
      atomic_outputs);
  auto inner = std::make_unique<MetalKernelInner>();
  inner->fn = std::move(kernel);
  return inner;
}
```

- [ ] **Step 2.3: Add bridge entry in `mlx-sys/src/bridge/fast.rs`**

Inside the existing `unsafe extern "C++"` block (after `shapes_vec_count`):

```rust
fn metal_kernel_build(
    name: &str,
    input_names: &[String],
    output_names: &[String],
    source: &str,
    header: &str,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
) -> Result<UniquePtr<MetalKernelInner>>;
```

- [ ] **Step 2.4: Smoke link test**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn metal_kernel_build_links() {
    use mlx_sys::fast::ffi;
    let input_names = vec!["x".to_string()];
    let output_names = vec!["y".to_string()];
    let src = r#"
        uint gid = thread_position_in_grid.x;
        y[gid] = x[gid] + 1.0;
    "#;
    let kernel = ffi::metal_kernel_build(
        "trivial_add_one",
        &input_names,
        &output_names,
        src,
        /* header */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs */ false,
    )
    .expect("kernel compiles");
    // We have a kernel; verify it's not null via Pin/UniquePtr is_null.
    assert!(!kernel.is_null());
}
```

- [ ] **Step 2.5: Run + gate + commit**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx-sys --test sys_smoke metal_kernel_build_links

cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3a): metal_kernel_build shim + bridge"
```

---

## Task 3: `metal_kernel_dispatch` — shim + bridge + smoke

**Files:**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/fast.h` (add `metal_kernel_dispatch` decl)
- Modify: `mlx-sys/shim/src/fast.cc` (impl)
- Modify: `mlx-sys/src/bridge/fast.rs` (bridge entry; reuse `ArrayVec` from compile bridge)
- Modify: `mlx-sys/tests/sys_smoke.rs` (smoke test that runs trivial kernel and checks output)

### Goal

Bind kernel dispatch — calls `kernel.fn(inputs, output_shapes, output_dtypes, grid, threadgroup, template_args, init_value, verbose, stream)` and returns outputs as `ArrayVec`. Reuses P6 `ArrayVec` for inputs and outputs.

### Steps

- [ ] **Step 3.1: Append `metal_kernel_dispatch` decl to `fast.h`**

After `metal_kernel_build` decl, plus the `ArrayVec` forward decl (it's defined in `compile.h`; we need to include or forward-declare):

At the top of `fast.h`, add to the existing includes:

```cpp
#include "cxx_mlx_shim/compile.h"   // for cxx_mlx::ArrayVec
```

(If `compile.h` already exposes `ArrayVec` via header — verify with `grep "struct ArrayVec" mlx-sys/shim/include/cxx_mlx_shim/compile.h`. If it's only forward-declared, define it inline here.)

Then append decl:

```cpp
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
```

> Note `TemplateArgC` is the **cxx-generated header** type. It comes from including the cxxbridge-generated header (added in fast.cc).

- [ ] **Step 3.2: Implement `metal_kernel_dispatch` in `fast.cc`**

Add include at the top of fast.cc (after existing includes):

```cpp
#include "mlx-sys/src/bridge/fast.rs.h"   // cxxbridge-generated; provides TemplateArgC + cxx_mlx namespace
```

(Verify the path matches the project's cxx-build output convention. If different, use the actual generated header path.)

Then append impl:

```cpp
// === P3a metal_kernel_dispatch ===

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
    bool has_stream, bool dev_only, uint8_t dev_type, int32_t stream_idx) {
  // 1. inputs vector copy from ArrayVec (refcount share)
  std::vector<mlx::core::array> ins(inputs.arrays.begin(), inputs.arrays.end());

  // 2. output dtypes
  std::vector<mlx::core::Dtype> out_dtypes;
  out_dtypes.reserve(output_dtypes.size());
  for (auto repr : output_dtypes) {
    out_dtypes.push_back(cxx_mlx::helpers::dtype_from_repr(repr));
  }

  // 3. template args: convert TemplateArgC to mlx variant
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> tmpl;
  tmpl.reserve(template_args.size());
  for (const auto& t : template_args) {
    std::string n(t.name);
    if (t.kind == 0) {
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{static_cast<int>(t.int_val)});
    } else if (t.kind == 1) {
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{t.bool_val});
    } else if (t.kind == 2) {
      auto dt = cxx_mlx::helpers::dtype_from_repr(static_cast<uint8_t>(t.int_val));
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{dt});
    } else {
      throw std::runtime_error("metal_kernel_dispatch: unknown TemplateArgC kind");
    }
  }

  // 4. init_value
  std::optional<float> init = has_init ? std::optional<float>(init_value) : std::nullopt;

  // 5. stream
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_stream, dev_only, dev_type, stream_idx);

  // 6. invoke kernel
  auto outs = kernel.fn(
      ins,
      output_shapes.shapes,
      out_dtypes,
      std::make_tuple(gx, gy, gz),
      std::make_tuple(tx, ty, tz),
      tmpl,
      init,
      verbose,
      target);

  // 7. wrap into ArrayVec
  auto out_vec = std::make_unique<ArrayVec>();
  out_vec->arrays = std::move(outs);
  return out_vec;
}
```

- [ ] **Step 3.3: Add bridge entry**

In `mlx-sys/src/bridge/fast.rs` — inside the `unsafe extern "C++"` block, after `metal_kernel_build`:

```rust
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
```

Inside the same block, reuse `ArrayVec` type from compile bridge:

```rust
type ArrayVec = crate::bridge::compile::ffi::ArrayVec;
```

(Verify `compile.rs` bridge exposes `ArrayVec` as a public `type` — it does per the existing `compile_with_callback` path which returns `UniquePtr<ArrayVec>`.)

- [ ] **Step 3.4: Smoke test — trivial add-one kernel**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn metal_kernel_dispatch_links() {
    use mlx_sys::fast::ffi as fast_ffi;
    use mlx_sys::compile::ffi as compile_ffi;

    let input_names = vec!["x".to_string()];
    let output_names = vec!["y".to_string()];
    let src = r#"
        uint gid = thread_position_in_grid.x;
        y[gid] = x[gid] + 1.0;
    "#;
    let kernel = fast_ffi::metal_kernel_build(
        "trivial_add_one",
        &input_names,
        &output_names,
        src,
        "", true, false,
    )
    .expect("kernel compiles");

    // Build inputs ArrayVec with a single [4] f32 array of zeros.
    let mut inputs = compile_ffi::array_vec_new();
    let zeros: mlx::Array = (&[0.0_f32, 0.0, 0.0, 0.0][..], (4_i32,)).try_into().unwrap();
    compile_ffi::array_vec_push(inputs.pin_mut(), zeros.as_inner());

    // output_shapes ShapesVec with [4]
    let mut shapes = fast_ffi::shapes_vec_new();
    fast_ffi::shapes_vec_push(shapes.pin_mut(), &[4]);

    // Dispatch
    // SAFETY: borrowed handles + slices are valid for the call duration.
    let mut outputs = unsafe {
        fast_ffi::metal_kernel_dispatch(
            &kernel,
            &inputs,
            &shapes,
            &[mlx::Dtype::Float32.as_u8()],
            /* grid */ 4, 1, 1,
            /* threadgroup */ 4, 1, 1,
            /* template_args */ &[],
            /* has_init */ false, 0.0,
            /* verbose */ false,
            /* has_stream */ false, false, 0, 0,
        )
    }
    .expect("dispatch succeeds");

    // Take output and verify y == [1.0; 4]
    let y_inner = compile_ffi::array_vec_take_at(outputs.pin_mut(), 0).expect("take 0");
    let y = mlx::Array::from_inner(y_inner);
    let v: Vec<f32> = y.to_vec().unwrap();
    assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);
}
```

> **Note**: this test uses `mlx_sys` types directly (no safe `MetalKernel` API yet — that's T4-T5). It also requires `mlx::Dtype::as_u8()` and `mlx::Array::from_inner` which already exist (P1+).

- [ ] **Step 3.5: Run + gate + commit**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx-sys --test sys_smoke metal_kernel_dispatch_links

cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3a): metal_kernel_dispatch shim + bridge"
```

---

## Task 4: Rust safe API — `MetalKernel` + `MetalKernelBuilder`

**Files:**
- Move: `mlx/src/fast.rs` → `mlx/src/fast/mod.rs` (preserve content)
- Create: `mlx/src/fast/metal_kernel/mod.rs`
- Modify: `mlx/src/lib.rs` (re-export `MetalKernel`)

### Goal

Restructure `mlx/src/fast.rs` into a directory module so `metal_kernel/` can live as a sub-module. Add `MetalKernel` (Arc-shared opaque handle) and `MetalKernelBuilder` (chained build).

### Steps

- [ ] **Step 4.1: Move `mlx/src/fast.rs` to `mlx/src/fast/mod.rs`**

```bash
mkdir -p mlx/src/fast
git mv mlx/src/fast.rs mlx/src/fast/mod.rs
```

Verify the file content is identical and `cargo build --release -p mlx` still succeeds.

```
cargo build --release -p mlx
```

- [ ] **Step 4.2: Append `pub mod metal_kernel` declaration to `mlx/src/fast/mod.rs`**

At the top of the file (after the existing `//!` doc comment, before any `use`):

```rust
pub mod metal_kernel;

pub use metal_kernel::{
    DispatchBuilder, MetalKernel, MetalKernelBuilder, Set, TemplateArg, Unset,
};
```

(The `DispatchBuilder` / `Set` / `Unset` / `TemplateArg` come in T5 but are forward-listed here; the `pub mod metal_kernel` declaration in this step + the `mod.rs` file in the next step will provide just `MetalKernel` and `MetalKernelBuilder`. Adjust the `pub use` to only re-export those two now and append the rest in T5.)

For T4, use this `pub use`:

```rust
pub use metal_kernel::{MetalKernel, MetalKernelBuilder};
```

- [ ] **Step 4.3: Create `mlx/src/fast/metal_kernel/mod.rs`**

```rust
//! Custom Metal kernel binding for Apple Silicon. Wraps
//! `mlx::core::fast::metal_kernel`.
//!
//! Two-phase API:
//! 1. **Build** — `MetalKernel::builder(name).inputs(...).outputs(...).source(...).build()?`
//!    compiles the Metal source once. Cheap to clone (`Arc` internally).
//! 2. **Dispatch** — `kernel.dispatch_builder().inputs(...).grid(...).threadgroup(...).dispatch()?`
//!    executes the kernel. Mandatory fields enforced at compile time via typestate
//!    (see `dispatch.rs`).
//!
//! See P3a spec § 3 for design rationale.

use std::sync::Arc;

use anyhow::anyhow;

use crate::Result;

mod dispatch;

pub use dispatch::{DispatchBuilder, Set, TemplateArg, Unset};

/// Compiled Metal kernel handle. Cheap to clone (Arc-shared inner).
pub struct MetalKernel {
    inner: Arc<MetalKernelInner>,
}

pub(crate) struct MetalKernelInner {
    pub(crate) handle: cxx::UniquePtr<mlx_sys::fast::ffi::MetalKernelInner>,
    pub(crate) output_count: usize,
}

// SAFETY: cxx::UniquePtr<MetalKernelInner> wraps a C++ object that holds
// `std::function<...>`; immutable after construction. The MLX
// CustomKernelFunction is intended to be called from any thread (the
// kernel itself is stateless; per-dispatch state is in arguments).
// Mark Send+Sync to allow Arc-share across threads.
unsafe impl Send for MetalKernelInner {}
unsafe impl Sync for MetalKernelInner {}

impl Clone for MetalKernel {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MetalKernel {
    /// Start building a kernel with the given name.
    pub fn builder(name: impl Into<String>) -> MetalKernelBuilder {
        MetalKernelBuilder {
            name: name.into(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            source: String::new(),
            header: String::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
    }

    /// Begin a dispatch invocation. Returns a typestate-protected builder
    /// where 5 mandatory fields (inputs / output_shapes / output_dtypes /
    /// grid / threadgroup) must be set before `.dispatch()` is callable.
    pub fn dispatch_builder(&self) -> DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
        DispatchBuilder::new(self.inner.clone())
    }

    /// Access the underlying inner (used by dispatch builder; not part of
    /// the public API).
    pub(crate) fn inner_arc(&self) -> &Arc<MetalKernelInner> {
        &self.inner
    }
}

/// Build-time configuration for a Metal kernel.
pub struct MetalKernelBuilder {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    source: String,
    header: String,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
}

impl MetalKernelBuilder {
    /// Set input parameter names.
    pub fn inputs(mut self, names: &[&str]) -> Self {
        self.input_names = names.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set output parameter names. The number of outputs is fixed at build
    /// time and must match the size of the `output_shapes` / `output_dtypes`
    /// passed at dispatch time (verified at runtime in `dispatch()`).
    pub fn outputs(mut self, names: &[&str]) -> Self {
        self.output_names = names.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set the Metal kernel source code (function body — not a full
    /// `kernel void f(...)` declaration; MLX wraps this).
    pub fn source(mut self, src: impl Into<String>) -> Self {
        self.source = src.into();
        self
    }

    /// Set an optional Metal header included before the kernel source.
    pub fn header(mut self, hdr: impl Into<String>) -> Self {
        self.header = hdr.into();
        self
    }

    /// Whether MLX should ensure inputs are row-contiguous before passing.
    /// Default `true`.
    pub fn ensure_row_contiguous(mut self, v: bool) -> Self {
        self.ensure_row_contiguous = v;
        self
    }

    /// Whether outputs should be initialized for atomic accumulation.
    /// Default `false`.
    pub fn atomic_outputs(mut self, v: bool) -> Self {
        self.atomic_outputs = v;
        self
    }

    /// Compile the kernel.
    pub fn build(self) -> Result<MetalKernel> {
        if self.input_names.is_empty() {
            return Err(anyhow!("MetalKernelBuilder: must call inputs(...) before build()"));
        }
        if self.output_names.is_empty() {
            return Err(anyhow!("MetalKernelBuilder: must call outputs(...) before build()"));
        }
        if self.source.is_empty() {
            return Err(anyhow!("MetalKernelBuilder: must call source(...) before build()"));
        }
        let output_count = self.output_names.len();
        let handle = mlx_sys::fast::ffi::metal_kernel_build(
            &self.name,
            &self.input_names,
            &self.output_names,
            &self.source,
            &self.header,
            self.ensure_row_contiguous,
            self.atomic_outputs,
        )
        .map_err(|e| anyhow!("metal_kernel_build failed: {e}"))?;
        Ok(MetalKernel {
            inner: Arc::new(MetalKernelInner {
                handle,
                output_count,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_rejects_empty_inputs() {
        let r = MetalKernel::builder("k")
            .outputs(&["y"])
            .source("y[0] = 1.0;")
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn builder_rejects_empty_outputs() {
        let r = MetalKernel::builder("k")
            .inputs(&["x"])
            .source("y[0] = 1.0;")
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn builder_rejects_empty_source() {
        let r = MetalKernel::builder("k")
            .inputs(&["x"])
            .outputs(&["y"])
            .build();
        assert!(r.is_err());
    }

    #[test]
    fn build_succeeds_with_valid_inputs() {
        let r = MetalKernel::builder("trivial_add_one")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
            .build();
        assert!(r.is_ok(), "build should succeed: {:?}", r.err());
    }

    #[test]
    fn clone_is_arc_share() {
        let k = MetalKernel::builder("k")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("y[0] = 1.0;")
            .build()
            .unwrap();
        let k2 = k.clone();
        // Both Arcs reference the same inner.
        assert!(Arc::ptr_eq(k.inner_arc(), k2.inner_arc()));
    }
}
```

> **Note**: This file references `dispatch::DispatchBuilder` (and other `dispatch` items) via `mod dispatch;`. The next step creates a stub `dispatch.rs` so this compiles; T5 fills in real impl.

- [ ] **Step 4.4: Create stub `mlx/src/fast/metal_kernel/dispatch.rs`**

```rust
//! Stub for typestate dispatch builder. Full implementation in T5.

use std::sync::Arc;

use crate::Dtype;

pub struct Unset;
pub struct Set;

#[derive(Debug, Clone)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}

pub struct DispatchBuilder<I, OS, OD, G, TG> {
    _kernel: Arc<super::MetalKernelInner>,
    _markers: std::marker::PhantomData<(I, OS, OD, G, TG)>,
}

impl DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
    pub(crate) fn new(kernel: Arc<super::MetalKernelInner>) -> Self {
        Self {
            _kernel: kernel,
            _markers: std::marker::PhantomData,
        }
    }
}
```

- [ ] **Step 4.5: Update `mlx/src/lib.rs` re-exports**

Find the existing `pub use fast::...` block (e.g. `pub use fast::{layer_norm, rms_norm, rope, ...};`) and append `MetalKernel`, `MetalKernelBuilder`:

```rust
pub use fast::{
    layer_norm, rms_norm, rope, rope_with_array_offset, scaled_dot_product_attention,
    MetalKernel, MetalKernelBuilder,
};
```

(Adjust based on actual current re-export shape in `lib.rs`.)

- [ ] **Step 4.6: Run tests + gate + commit**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --lib fast::metal_kernel

cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3a): MetalKernel + MetalKernelBuilder safe API + restructure fast.rs into directory"
```

Expected: 5 unit tests pass (4 builder validation + 1 clone share).

---

## Task 5: `DispatchBuilder` typestate + 5 markers + setter all

**Files:**
- Modify: `mlx/src/fast/metal_kernel/dispatch.rs` (full implementation)
- Modify: `mlx/src/fast/mod.rs` (extend re-exports for `DispatchBuilder` / `Set` / `Unset` / `TemplateArg`)

### Goal

Replace stub `dispatch.rs` with full typestate `DispatchBuilder`: 5 marker-changing setters (inputs / output_shapes / output_dtypes / grid / threadgroup), 6 always-available optional setters (template_int / template_bool / template_dtype / init_value / verbose / stream), and `.dispatch()` only callable when all 5 markers are `Set`.

### Steps

- [ ] **Step 5.1: Replace `mlx/src/fast/metal_kernel/dispatch.rs` with full implementation**

```rust
//! Typestate-protected dispatch builder. Compile-time enforces all 5
//! mandatory fields (inputs, output_shapes, output_dtypes, grid,
//! threadgroup) are set before `.dispatch()` is callable.
//!
//! Marker layout: `DispatchBuilder<I, OS, OD, G, TG>` where each is either
//! `Unset` or `Set`. Setters move the relevant marker from `Unset` to `Set`.

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::anyhow;
use mlx_sys::fast::ffi::TemplateArgC;

use crate::core::ArrayVec;   // Re-export TBD: see step 5.x — might need
                              // to use mlx_sys::compile::ffi::ArrayVec
                              // directly or introduce a safe wrapper.
use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

use super::MetalKernelInner;

/// Marker: builder field has not been set.
pub struct Unset;

/// Marker: builder field has been set.
pub struct Set;

/// cxx-safe template argument. Maps to MLX's
/// `std::variant<int, bool, Dtype>`.
#[derive(Debug, Clone)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}

impl TemplateArg {
    fn to_c(&self, name: &str) -> TemplateArgC {
        match self {
            TemplateArg::Int(v) => TemplateArgC {
                name: name.to_string(),
                kind: 0,
                int_val: *v,
                bool_val: false,
            },
            TemplateArg::Bool(v) => TemplateArgC {
                name: name.to_string(),
                kind: 1,
                int_val: 0,
                bool_val: *v,
            },
            TemplateArg::Dtype(d) => TemplateArgC {
                name: name.to_string(),
                kind: 2,
                int_val: d.as_u8() as i32,
                bool_val: false,
            },
        }
    }
}

/// Typestate-protected dispatch builder.
pub struct DispatchBuilder<I, OS, OD, G, TG> {
    kernel: Arc<MetalKernelInner>,

    inputs: Option<Vec<*const mlx_sys::array::ffi::MlxArray>>,
    output_shapes: Option<Vec<Shape>>,
    output_dtypes: Option<Vec<Dtype>>,
    grid: Option<(i32, i32, i32)>,
    threadgroup: Option<(i32, i32, i32)>,

    template_args: Vec<(String, TemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
    target: StreamOrDevice,

    _markers: PhantomData<(I, OS, OD, G, TG)>,
}

impl DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
    pub(crate) fn new(kernel: Arc<MetalKernelInner>) -> Self {
        Self {
            kernel,
            inputs: None,
            output_shapes: None,
            output_dtypes: None,
            grid: None,
            threadgroup: None,
            template_args: Vec::new(),
            init_value: None,
            verbose: false,
            target: StreamOrDevice::Default,
            _markers: PhantomData,
        }
    }
}

// === 5 mandatory setters (each transitions one marker Unset -> Set) ===

impl<OS, OD, G, TG> DispatchBuilder<Unset, OS, OD, G, TG> {
    /// Set the input arrays. Required.
    pub fn inputs(self, arrays: &[&Array]) -> DispatchBuilder<Set, OS, OD, G, TG> {
        let raw: Vec<*const _> = arrays
            .iter()
            .map(|a| a.as_inner() as *const _)
            .collect();
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
    /// Set the output shapes. Required. Must match `output_dtypes` length and
    /// the kernel's declared output count.
    pub fn output_shapes(self, shapes: &[Shape]) -> DispatchBuilder<I, Set, OD, G, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: Some(shapes.to_vec()),
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

impl<I, OS, G, TG> DispatchBuilder<I, OS, Unset, G, TG> {
    /// Set the output dtypes. Required.
    pub fn output_dtypes(self, dtypes: &[Dtype]) -> DispatchBuilder<I, OS, Set, G, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: Some(dtypes.to_vec()),
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

impl<I, OS, OD, TG> DispatchBuilder<I, OS, OD, Unset, TG> {
    /// Set GPU dispatch grid (x, y, z). Required.
    pub fn grid(self, gx: i32, gy: i32, gz: i32) -> DispatchBuilder<I, OS, OD, Set, TG> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: Some((gx, gy, gz)),
            threadgroup: self.threadgroup,
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

impl<I, OS, OD, G> DispatchBuilder<I, OS, OD, G, Unset> {
    /// Set GPU threadgroup size (x, y, z). Required.
    pub fn threadgroup(self, tx: i32, ty: i32, tz: i32) -> DispatchBuilder<I, OS, OD, G, Set> {
        DispatchBuilder {
            kernel: self.kernel,
            inputs: self.inputs,
            output_shapes: self.output_shapes,
            output_dtypes: self.output_dtypes,
            grid: self.grid,
            threadgroup: Some((tx, ty, tz)),
            template_args: self.template_args,
            init_value: self.init_value,
            verbose: self.verbose,
            target: self.target,
            _markers: PhantomData,
        }
    }
}

// === 6 optional setters (don't change markers) ===

impl<I, OS, OD, G, TG> DispatchBuilder<I, OS, OD, G, TG> {
    /// Add an `int` template argument.
    pub fn template_int(mut self, name: impl Into<String>, v: i32) -> Self {
        self.template_args.push((name.into(), TemplateArg::Int(v)));
        self
    }

    /// Add a `bool` template argument.
    pub fn template_bool(mut self, name: impl Into<String>, v: bool) -> Self {
        self.template_args.push((name.into(), TemplateArg::Bool(v)));
        self
    }

    /// Add a `Dtype` template argument.
    pub fn template_dtype(mut self, name: impl Into<String>, v: Dtype) -> Self {
        self.template_args.push((name.into(), TemplateArg::Dtype(v)));
        self
    }

    /// Set initial value for atomic outputs (only meaningful if kernel was
    /// built with `atomic_outputs(true)`).
    pub fn init_value(mut self, v: f32) -> Self {
        self.init_value = Some(v);
        self
    }

    /// Enable verbose Metal compile logging.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Set target stream/device.
    pub fn stream(mut self, target: impl Into<StreamOrDevice>) -> Self {
        self.target = target.into();
        self
    }
}

// === dispatch() — only callable with all markers Set ===

impl DispatchBuilder<Set, Set, Set, Set, Set> {
    /// Execute the kernel and return outputs as `ArrayVec` (P6 wrapper).
    /// Take individual outputs via `arr_vec.take_at(i)` in the order declared
    /// in `MetalKernelBuilder::outputs(...)`.
    pub fn dispatch(self) -> Result<crate::core::ArrayVec> {
        let inputs = self.inputs.expect("typestate: inputs Set");
        let output_shapes = self.output_shapes.expect("typestate: output_shapes Set");
        let output_dtypes = self.output_dtypes.expect("typestate: output_dtypes Set");
        let grid = self.grid.expect("typestate: grid Set");
        let threadgroup = self.threadgroup.expect("typestate: threadgroup Set");

        // Sanity: counts match
        if output_shapes.len() != self.kernel.output_count {
            return Err(anyhow!(
                "MetalKernel dispatch: output_shapes count {} != declared outputs {}",
                output_shapes.len(),
                self.kernel.output_count,
            ));
        }
        if output_dtypes.len() != self.kernel.output_count {
            return Err(anyhow!(
                "MetalKernel dispatch: output_dtypes count {} != declared outputs {}",
                output_dtypes.len(),
                self.kernel.output_count,
            ));
        }

        // Build inputs ArrayVec (P6 reuse)
        let mut input_vec = mlx_sys::compile::ffi::array_vec_new();
        for ptr in &inputs {
            // SAFETY: ptr came from Array::as_inner; lifetime guaranteed by
            // the &[&Array] borrow held until the dispatch call returns.
            unsafe {
                mlx_sys::compile::ffi::array_vec_push(input_vec.pin_mut(), &**ptr);
            }
        }

        // Build ShapesVec
        let mut shapes_vec = mlx_sys::fast::ffi::shapes_vec_new();
        for s in &output_shapes {
            mlx_sys::fast::ffi::shapes_vec_push(shapes_vec.pin_mut(), s.as_slice());
        }

        // Encode template args
        let template_c: Vec<TemplateArgC> = self
            .template_args
            .iter()
            .map(|(name, val)| val.to_c(name))
            .collect();

        // Encode dtypes
        let dtype_reprs: Vec<u8> = output_dtypes.iter().map(|d| d.as_u8()).collect();

        // Encode stream
        let (has_stream, dev_only, dev_t, idx) = self.target.encode();

        let (init_has, init_v) = match self.init_value {
            Some(v) => (true, v),
            None => (false, 0.0),
        };

        // SAFETY: all borrowed args live for the duration of the call.
        let raw_outputs = unsafe {
            mlx_sys::fast::ffi::metal_kernel_dispatch(
                &self.kernel.handle,
                &input_vec,
                &shapes_vec,
                &dtype_reprs,
                grid.0,
                grid.1,
                grid.2,
                threadgroup.0,
                threadgroup.1,
                threadgroup.2,
                &template_c,
                init_has,
                init_v,
                self.verbose,
                has_stream,
                dev_only,
                dev_t,
                idx,
            )
        }
        .map_err(Error::from)?;

        Ok(crate::core::ArrayVec::from_inner(raw_outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::MetalKernel;

    fn trivial_kernel() -> MetalKernel {
        MetalKernel::builder("trivial_add_one")
            .inputs(&["x"])
            .outputs(&["y"])
            .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
            .build()
            .expect("kernel compiles")
    }

    #[test]
    fn template_arg_int_to_c() {
        let c = TemplateArg::Int(42).to_c("Dk");
        assert_eq!(c.kind, 0);
        assert_eq!(c.int_val, 42);
    }

    #[test]
    fn template_arg_bool_to_c() {
        let c = TemplateArg::Bool(true).to_c("Vec");
        assert_eq!(c.kind, 1);
        assert!(c.bool_val);
    }

    #[test]
    fn template_arg_dtype_to_c() {
        let c = TemplateArg::Dtype(Dtype::Float16).to_c("InT");
        assert_eq!(c.kind, 2);
        assert_eq!(c.int_val, Dtype::Float16.as_u8() as i32);
    }

    #[test]
    fn typestate_setters_traverse_to_dispatchable() {
        let k = trivial_kernel();
        let x: Array = (&[0.0_f32, 0.0, 0.0, 0.0][..], (4_i32,)).try_into().unwrap();
        // Walks all 5 mandatory setters; .dispatch() is callable here.
        let mut outputs = k
            .dispatch_builder()
            .inputs(&[&x])
            .output_shapes(&[Shape::from((4_i32,))])
            .output_dtypes(&[Dtype::Float32])
            .grid(4, 1, 1)
            .threadgroup(4, 1, 1)
            .dispatch()
            .expect("dispatch ok");
        let y_inner = outputs.take_at_inner(0).expect("take");
        let y = Array::from_inner(y_inner);
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }
}
```

> **Note**: This file references `crate::core::ArrayVec` — a thin safe-API wrapper around `mlx_sys::compile::ffi::ArrayVec`. The cxx-mlx repo doesn't currently expose `ArrayVec` as a safe type at the `mlx` crate level; only `mlx_sys` exposes the raw type. The next step adds this thin wrapper.

- [ ] **Step 5.2: Add safe `ArrayVec` wrapper to `mlx/src/lib.rs`**

In `mlx/src/lib.rs`, add a new module `core` (or `compile_helpers`) that exposes a thin `ArrayVec` safe wrapper. To minimize scope, put it in a new file `mlx/src/array_vec.rs`:

```rust
//! Thin safe wrapper around `mlx_sys::compile::ffi::ArrayVec` — used by
//! `MetalKernel::dispatch` and the closure-compile path.
//!
//! `ArrayVec` is C++-side `std::vector<array>` (P6 design, see
//! `docs/superpowers/specs/2026-05-05-cxx-mlx-p6-compile-design.md`).

use crate::{Array, Error, Result};

/// Owning wrapper around a C++ `std::vector<array>`. Outputs of multi-array
/// ops (e.g. `MetalKernel::dispatch`) are returned as `ArrayVec`. Take
/// individual elements via `take_at(i)` in the order they were produced.
pub struct ArrayVec {
    inner: cxx::UniquePtr<mlx_sys::compile::ffi::ArrayVec>,
}

impl ArrayVec {
    /// Construct from a raw cxx UniquePtr. Internal use only.
    pub(crate) fn from_inner(inner: cxx::UniquePtr<mlx_sys::compile::ffi::ArrayVec>) -> Self {
        Self { inner }
    }

    /// Take the inner cxx UniquePtr (consume self). Used by Array::take_at_inner.
    #[doc(hidden)]
    pub(crate) fn take_at_inner(
        &mut self,
        i: usize,
    ) -> Result<cxx::UniquePtr<mlx_sys::array::ffi::MlxArray>> {
        mlx_sys::compile::ffi::array_vec_take_at(self.inner.pin_mut(), i)
            .map_err(Error::from)
    }

    /// Number of arrays.
    pub fn len(&self) -> usize {
        mlx_sys::compile::ffi::array_vec_count(&self.inner)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take the i-th array out of the vec, leaving the slot consumed.
    /// Returns Err if `i >= len()` or already taken.
    pub fn take_at(&mut self, i: usize) -> Result<Array> {
        let inner = self.take_at_inner(i)?;
        Ok(Array::from_inner(inner))
    }
}
```

Add `pub mod array_vec;` to `mlx/src/lib.rs` near the other `mod` declarations. Re-export at crate root:

```rust
pub use array_vec::ArrayVec;
```

Update `mlx/src/fast/metal_kernel/dispatch.rs` import to `use crate::ArrayVec;` (instead of `crate::core::ArrayVec`).

- [ ] **Step 5.3: Update `mlx/src/fast/mod.rs` re-exports**

Replace the T4 limited re-export:

```rust
pub use metal_kernel::{
    DispatchBuilder, MetalKernel, MetalKernelBuilder, Set, TemplateArg, Unset,
};
```

- [ ] **Step 5.4: Update `mlx/src/lib.rs` re-exports**

```rust
pub use fast::{
    layer_norm, rms_norm, rope, rope_with_array_offset, scaled_dot_product_attention,
    DispatchBuilder, MetalKernel, MetalKernelBuilder, Set, TemplateArg, Unset,
};
```

- [ ] **Step 5.5: Run tests + gate + commit**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --lib

cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3a): DispatchBuilder typestate + 5 markers + setters + ArrayVec safe wrapper"
```

Expected: 4 new tests in dispatch.rs (`template_arg_*_to_c` x3 + `typestate_setters_traverse_to_dispatchable`) + existing 5 tests in `mod.rs`. Total ~9 metal_kernel unit tests pass.

---

## Task 6: trybuild typestate compile-fail tests

**Files:**
- Modify: `mlx/Cargo.toml` (add `trybuild` dev-dep)
- Create: `mlx/tests/trybuild.rs` (entry)
- Create: `mlx/tests/trybuild/ui/metal_kernel_missing_inputs.rs` (and 4 more)
- Create: corresponding `.stderr` files

### Goal

Verify at compile time that `.dispatch()` is unavailable when any of the 5 mandatory fields is unset.

### Steps

- [ ] **Step 6.1: Add `trybuild` dev-dep to `mlx/Cargo.toml`**

In the `[dev-dependencies]` section:

```toml
trybuild = "1"
```

- [ ] **Step 6.2: Create `mlx/tests/trybuild.rs` entry**

```rust
//! trybuild compile-fail tests. Each `ui/*.rs` should fail to compile,
//! with the expected error in `ui/*.stderr`.

#[test]
fn metal_kernel_typestate_compile_fails() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/ui/metal_kernel_missing_*.rs");
}
```

- [ ] **Step 6.3: Create `mlx/tests/trybuild/ui/metal_kernel_missing_inputs.rs`**

```rust
//! Compile-fail: missing .inputs(...) on dispatch builder.

use mlx::{Dtype, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();

    // ERROR: .inputs() not called → .dispatch() not callable
    let _ = k
        .dispatch_builder()
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();
}
```

Generate the `.stderr` by running the test once and copying compiler output:

```bash
TRYBUILD=overwrite MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --test trybuild
```

This populates `metal_kernel_missing_inputs.stderr` automatically. Verify the contents include something like:

```
error[E0599]: no method named `dispatch` found for struct
              `DispatchBuilder<Unset, Set, Set, Set, Set>`
```

- [ ] **Step 6.4: Create the other 4 missing-setter tests**

`mlx/tests/trybuild/ui/metal_kernel_missing_output_shapes.rs`:

```rust
use mlx::{Array, Dtype, MetalKernel};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();
    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let _ = k
        .dispatch_builder()
        .inputs(&[&x])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();
}
```

`mlx/tests/trybuild/ui/metal_kernel_missing_output_dtypes.rs`:

```rust
use mlx::{Array, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();
    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let _ = k
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();
}
```

`mlx/tests/trybuild/ui/metal_kernel_missing_grid.rs`:

```rust
use mlx::{Array, Dtype, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();
    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let _ = k
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .threadgroup(4, 1, 1)
        .dispatch();
}
```

`mlx/tests/trybuild/ui/metal_kernel_missing_threadgroup.rs`:

```rust
use mlx::{Array, Dtype, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();
    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let _ = k
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .dispatch();
}
```

- [ ] **Step 6.5: Generate all `.stderr` files**

```bash
TRYBUILD=overwrite MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --test trybuild
```

This generates 5 `.stderr` files in `mlx/tests/trybuild/ui/`. Inspect each to confirm the expected error message includes `no method named 'dispatch' found for struct DispatchBuilder<...Unset...>`.

- [ ] **Step 6.6: Run trybuild test in normal mode (no overwrite) to verify**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --test trybuild
```

Expected: 1 passing test (the entry `metal_kernel_typestate_compile_fails`), which itself runs 5 compile-fail checks internally.

- [ ] **Step 6.7: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "test(p3a): trybuild compile-fail tests for typestate enforcement"
```

---

## Task 7: integration tests + final verify

**Files:**
- Create: `mlx/tests/p3a_metal_kernel.rs`

### Goal

Two end-to-end integration tests using the safe API: trivial add-one kernel and multi-output kernel.

### Steps

- [ ] **Step 7.1: Create `mlx/tests/p3a_metal_kernel.rs`**

```rust
//! Integration tests for P3a — `MetalKernel` end-to-end via safe API.

use mlx::{Array, Dtype, MetalKernel, Shape};

#[test]
fn simple_add_kernel() {
    let kernel = MetalKernel::builder("simple_add")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4_i32,)).try_into().unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .expect("dispatch");

    assert_eq!(outputs.len(), 1);
    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.shape().as_slice(), &[4]);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn multi_output_kernel() {
    // Two outputs: y = x*2, z = x+10
    let kernel = MetalKernel::builder("multi_out")
        .inputs(&["x"])
        .outputs(&["y", "z"])
        .source(
            "uint gid = thread_position_in_grid.x; \
             y[gid] = x[gid] * 2.0; \
             z[gid] = x[gid] + 10.0;",
        )
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4_i32,)).try_into().unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,)), Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32, Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .expect("dispatch");

    assert_eq!(outputs.len(), 2);

    // Take in declared order: y first, then z
    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

    let z = outputs.take_at(1).expect("take 1");
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn template_int_substitution() {
    // Use a template to multiply by a compile-time constant.
    let kernel = MetalKernel::builder("template_mul")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] * static_cast<float>(MUL);")
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0][..], (3_i32,)).try_into().unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((3_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(3, 1, 1)
        .threadgroup(3, 1, 1)
        .template_int("MUL", 7)
        .dispatch()
        .expect("dispatch");

    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![7.0, 14.0, 21.0]);
}

#[test]
fn output_count_mismatch_errors() {
    // Kernel declares 1 output but dispatch passes 2 shapes — should error.
    let kernel = MetalKernel::builder("one_out")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid];")
        .build()
        .expect("compile");

    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();
    let r = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,)), Shape::from((4_i32,))])  // 2 shapes
        .output_dtypes(&[Dtype::Float32, Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();

    assert!(r.is_err(), "expected error from count mismatch");
    let msg = format!("{}", r.unwrap_err());
    assert!(msg.contains("output_shapes count"), "msg: {msg}");
}

#[test]
fn clone_kernel_dispatches_independently() {
    let kernel = MetalKernel::builder("add_two")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 2.0;")
        .build()
        .expect("compile");

    let kernel2 = kernel.clone();
    let x: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let mut o1 = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .unwrap();

    let mut o2 = kernel2
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .unwrap();

    assert_eq!(
        o1.take_at(0).unwrap().to_vec::<f32>().unwrap(),
        o2.take_at(0).unwrap().to_vec::<f32>().unwrap()
    );
}
```

- [ ] **Step 7.2: Run integration tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --test p3a_metal_kernel
```

Expected: 5 tests pass.

- [ ] **Step 7.3: Run full workspace tests for regression**

```
MLX_DIR=$HOME/.local/mlx cargo test --release
```

Expected: all pre-P3a tests still pass + new P3a tests.

- [ ] **Step 7.4: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "test(p3a): MetalKernel integration tests (simple add, multi output, template, count mismatch, clone)"
```

---

## Verification Checklist

After Task 7:

| Item | Command | Expected |
|---|---|---|
| Sys-level smoke | `cargo test --release -p mlx-sys --test sys_smoke` | new shapes_vec_links + metal_kernel_build_links + metal_kernel_dispatch_links pass |
| mlx unit tests | `cargo test --release -p mlx --lib` | builder + dispatch unit tests pass |
| trybuild compile-fail | `cargo test --release -p mlx --test trybuild` | 1 entry test, 5 compile-fail checks |
| Integration tests | `cargo test --release -p mlx --test p3a_metal_kernel` | 5 tests pass |
| Full regression | `cargo test --release` | all earlier tests pass |
| Format | `cargo +nightly fmt --all -- --check` | no diff |
| Clippy | `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | no warnings |
| Build | `cargo build --release` | success |

## Spec Coverage Map

| Spec section | Task |
|---|---|
| § 3.1 公开 API (MetalKernel + MetalKernelBuilder) | T4 |
| § 3.2 typestate DispatchBuilder | T5 |
| § 3.3 跨 cxx 边界 — C++ shim | T1 (opaque types + ShapesVec API) + T2 (build) + T3 (dispatch) |
| § 3.4 跨 cxx 边界 — Rust bridge | T1 + T2 + T3 |
| § 3.5 调用示例 | T7 (integration tests serve as canonical examples) |
| § 4.1 单元测试 | T4 (5 builder tests) + T5 (4 dispatch tests) |
| § 4.2 集成测试 | T7 (5 tests) |
| § 4.3 编译期 typestate 验证 | T6 (5 trybuild compile-fail tests) |

## Risk register (per spec § 6)

- **`std::function` (CustomKernelFunction) 跨 cxx 不直接支持**: handled by T1's `MetalKernelInner` opaque struct holding the function on C++ side; never crosses cxx boundary directly. Verified at T2 (kernel built and reachable).
- **`std::variant` (TemplateArg) 跨 cxx 不直接支持**: handled by T1's `TemplateArgC` cxx-friendly struct (kind + int_val + bool_val). T5 verifies `TemplateArg::to_c` produces correct encoding for all 3 variants.
- **`std::vector<Shape>` 跨 cxx 不直接支持**: handled by T1's `ShapesVec` opaque (mirroring P6 `ArrayVec`). T1 + T7 multi-output test verify round-trip.
- **typestate 错误消息差**: documented in spec § 6; the trybuild `.stderr` files in T6 capture the exact error format, providing a maintained reference for users.
- **dispatch hot-path overhead**: 5 `Option::expect` calls in `dispatch()` are eliminated by the optimizer once typestate guarantees they're `Some`. Per spec § 6: ≤ 5 ns total. Verified at T7 by integration test passing without performance regression vs T3 raw FFI.
- **MLX-side Metal compile failure**: surfaces as `Err(Error::Mlx(...))` from `metal_kernel_build`. T4 builder validation tests confirm error path.
- **`mlx::core::ArrayVec` only in mlx_sys**: T5 adds thin safe wrapper `mlx::ArrayVec` so users don't need to import `mlx_sys` directly.
