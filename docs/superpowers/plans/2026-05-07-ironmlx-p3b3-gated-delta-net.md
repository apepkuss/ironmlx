# ironmlx P3b3 — Gated Delta Net (SSM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Qwen3.5 / Qwen3-Next "linear attention" branch (`GatedDeltaNet`). This is the recursive SSM with delta-rule + scalar gating, alternating with `GatedAttention` per `full_attention_interval`.

**Architecture:** Cross-crate (cxx-mlx + ironmlx). Adds `mlx::ops::conv1d` to cxx-mlx safe layer; adds 4 ironmlx components (`nn::Conv1d`, `nn::RmsNormGated`, `core::cache::GatedDeltaCache`, `nn::GatedDeltaNet`); adds 1 custom Metal kernel (`gated_delta_step`) in 2 variants (scalar gating × {no-mask, masked}). The kernel has a per-token recurrence loop with `simd_sum` reductions over the Dk axis, fp32 state accumulation, and a templated index/dtype set (`Dk, Dv, Hk, Hv, InT, StT`) matching mlx-lm.

**Tech Stack:** Rust 2021 + cxx 1.0 + MLX C++ (`mlx::core::conv1d`, `mlx::core::fast::metal_kernel`) + ironmlx (`anyhow::Result`, P1 nn::Linear, P2 cache scaffolding, P3a MetalKernel typestate, P3b1 Mrope::cos_sin compile-pattern, P6 mlx::compile). **Spec:** [`docs/superpowers/specs/2026-05-07-ironmlx-p3b3-gated-delta-net-design.md`](../specs/2026-05-07-ironmlx-p3b3-gated-delta-net-design.md).

---

## Conventions Recap

- **TDD per task**: failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before each commit** (`.claude/CLAUDE.md`):
  ```
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```
- **`MLX_DIR=$HOME/.local/mlx`** required for tests that exercise MLX FFI / GPU.
- **MLX source location**: `/Volumes/Dev/mlx`.
- **mlx-lm reference**: `/Volumes/Dev/mlx-lm/mlx_lm/models/gated_delta.py` (kernel) + `models/qwen3_5.py:85-205` (GatedDeltaNet class).
- **ironmlx error type**: `anyhow::{Error, Result}` re-exported as `crate::{Error, Result}`. Use `anyhow::anyhow!(...)`.
- **ASCII commit messages**.

---

## File Structure (after P3b3)

```
mlx-sys/                                          # cxx-mlx FFI layer
├── shim/
│   ├── include/cxx_mlx_shim/conv.h               # NEW
│   └── src/conv.cc                               # NEW
├── src/
│   ├── bridge/conv.rs                            # NEW
│   └── lib.rs                                    # MODIFIED — register conv
└── build.rs                                      # MODIFIED — add conv.cc to cxx-build

mlx/                                              # cxx-mlx safe API
└── src/
    ├── ops/conv.rs                               # NEW
    └── ops/mod.rs                                # MODIFIED — pub mod conv + re-exports

ironmlx/                                          # ironmlx
├── src/
│   ├── core/
│   │   └── cache/
│   │       ├── gated_delta.rs                    # NEW — GatedDeltaCache
│   │       └── mod.rs                            # MODIFIED — pub mod gated_delta + re-export
│   └── nn/
│       ├── conv.rs                               # NEW — nn::Conv1d
│       ├── gated_delta_net.rs                    # NEW — GatedDeltaNet + gated_delta_step kernel
│       ├── mod.rs                                # MODIFIED — pub mod conv + gated_delta_net + RmsNormGated re-export
│       └── norm.rs                               # MODIFIED — add RmsNormGated struct
└── tests/
    ├── fixtures/
    │   └── p3b3_gated_delta_net/                 # NEW
    │       ├── README.md
    │       ├── gen_fixture.py
    │       ├── input_x.npy                       # [1, 4, 32] bf16
    │       ├── input_a.npy                       # [1, 4, 4] fp32  (a = in_proj_a(x), Hv=4)
    │       ├── input_b.npy                       # [1, 4, 4] fp32  (b = in_proj_b(x))
    │       ├── (... weight files ...)
    │       └── expected_gated_delta_out.npy      # [1, 4, 32] bf16
    └── p3b3_gated_delta_net.rs                   # NEW
```

---

## Task 1: cxx-mlx `mlx::ops::conv1d` binding (shim + bridge + safe wrapper)

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/conv.h`
- Create: `mlx-sys/shim/src/conv.cc`
- Create: `mlx-sys/src/bridge/conv.rs`
- Modify: `mlx-sys/src/bridge/mod.rs` (add `pub mod conv`)
- Modify: `mlx-sys/src/lib.rs` (add `pub mod conv` re-export)
- Modify: `mlx-sys/build.rs` (add `conv.cc` to cxx-build inputs)
- Create: `mlx/src/ops/conv.rs`
- Modify: `mlx/src/ops/mod.rs` (pub mod conv + re-exports)
- Test: `mlx-sys/tests/sys_smoke.rs` (add `conv1d_links`)

### Goal

Bind `mlx::core::conv1d(input, weight, stride, padding, dilation, groups, target)` through cxx-mlx's three-layer FFI. Used by ironmlx::nn::Conv1d (T2) and is reusable for future P5/P6 conv ops.

### Steps

- [ ] **Step 1.1: Create shim header**

Create `mlx-sys/shim/include/cxx_mlx_shim/conv.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

std::unique_ptr<MlxArray> ops_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

}  // namespace cxx_mlx
```

- [ ] **Step 1.2: Create shim impl**

Create `mlx-sys/shim/src/conv.cc`:

```cpp
#include "cxx_mlx_shim/conv.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include "mlx/ops.h"

namespace cxx_mlx {

std::unique_ptr<MlxArray> ops_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::conv1d(input, weight, stride, padding, dilation, groups, target));
}

}  // namespace cxx_mlx
```

- [ ] **Step 1.3: Create cxx::bridge module**

Create `mlx-sys/src/bridge/conv.rs`:

```rust
//! Bridge for MLX convolution ops (currently conv1d only — conv2d/conv3d on demand).
//!
//! Each fn carries 4 trailing `StreamOrDevice` args (P5.7) — same encoding
//! as the array bridge: `(has_target, is_device_only, device_type, stream_index)`.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/conv.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;

        unsafe fn ops_conv1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
```

- [ ] **Step 1.4: Register in `mlx-sys/src/bridge/mod.rs`**

Find the existing `pub mod ...;` block and add:

```rust
pub mod array;
pub mod compile;
pub mod conv;          // NEW
pub mod fast;
pub mod io;
pub mod quantization;
pub mod random;
pub mod stream;
pub mod transforms;
```

- [ ] **Step 1.5: Register in `mlx-sys/src/lib.rs`**

Find the existing `pub mod ...;` block (should mirror `bridge/mod.rs`) and add:

```rust
pub mod conv {
    pub use crate::bridge::conv::ffi;
}
```

(Adjust the wrapping pattern to match the existing module re-export style — e.g. `pub mod array { pub use crate::bridge::array::ffi; }`.)

- [ ] **Step 1.6: Register in `mlx-sys/build.rs`**

Find the cxx-build inputs list (a `Vec<&str>` or similar listing of `.cc` files). Add `"shim/src/conv.cc"` to the list. The existing pattern:

```rust
.file("shim/src/array.cc")
.file("shim/src/compile.cc")
.file("shim/src/conv.cc")    // NEW
.file("shim/src/fast.cc")
// ...
```

(The exact syntax depends on the existing build.rs; adapt as needed.)

Also add `"src/bridge/conv.rs"` to the cxx::bridge list:

```rust
let bridges = vec![
    "src/bridge/array.rs",
    "src/bridge/compile.rs",
    "src/bridge/conv.rs",    // NEW
    "src/bridge/fast.rs",
    // ...
];
```

- [ ] **Step 1.7: Write the failing sys_smoke link test**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn conv1d_links() {
    use mlx_sys::array::ffi as array_ffi;
    use mlx_sys::conv::ffi as conv_ffi;

    // input: [N=1, L=8, C_in=2] fp32, all zeros
    let input = array_ffi::array_zeros(&[1, 8, 2], FLOAT32).expect("input zeros");
    // weight: [C_out=4, K=3, C_in/groups=2] fp32, all zeros
    let weight = array_ffi::array_zeros(&[4, 3, 2], FLOAT32).expect("weight zeros");

    let out = unsafe {
        conv_ffi::ops_conv1d(
            &input,
            &weight,
            /* stride */ 1,
            /* padding */ 0,
            /* dilation */ 1,
            /* groups */ 1,
            /* has_target */ false,
            /* is_device_only */ false,
            /* device_type */ 0,
            /* stream_index */ 0,
        )
    }
    .expect("conv1d should succeed");
    assert!(!out.is_null());
    // output shape: [N=1, L_out=8-3+1=6, C_out=4]
    assert_eq!(array_ffi::array_shape(&out), vec![1, 6, 4]);
}
```

- [ ] **Step 1.8: Run, verify it fails to compile (no conv module yet)**

```
MLX_DIR=$HOME/.local/mlx cargo build --release -p mlx-sys
```

Expected: build error referring to `mlx_sys::conv::ffi` not found (or build succeeds if Steps 1.1-1.6 already wired it up — in that case the link test should now pass).

- [ ] **Step 1.9: Run the test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx-sys --test sys_smoke conv1d_links
```

Expected: PASS.

- [ ] **Step 1.10: Create safe wrapper `mlx/src/ops/conv.rs`**

```rust
//! 1D convolution: `mlx::core::conv1d`.
//!
//! Input layout `[N, L, C_in]`, weight layout `[C_out, K, C_in / groups]`,
//! output `[N, L_out, C_out]` where `L_out = (L + 2*padding - dilation*(K-1) - 1) / stride + 1`.
//!
//! For depthwise convolution, set `groups = C_in == C_out`.

use crate::{Array, Error, Result, StreamOrDevice};

/// 1D convolution with default stream.
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    conv1d_on(input, weight, stride, padding, dilation, groups, ())
}

/// Stream-targeted 1D convolution.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_on(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input/weight borrows valid for the call duration.
    let inner = unsafe {
        mlx_sys::conv::ffi::ops_conv1d(
            input.as_inner(),
            weight.as_inner(),
            stride,
            padding,
            dilation,
            groups,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ops::constructors, Dtype};

    #[test]
    fn conv1d_shape_basic() {
        // input: [1, 8, 2], weight: [4, 3, 2] (C_out=4, K=3, C_in=2), groups=1
        let input = constructors::zeros((1_i32, 8, 2), Dtype::Float32).unwrap();
        let weight = constructors::zeros((4_i32, 3, 2), Dtype::Float32).unwrap();
        let out = conv1d(&input, &weight, 1, 0, 1, 1).expect("conv1d");
        assert_eq!(out.shape().as_slice(), &[1, 6, 4]);
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    fn conv1d_depthwise_shape() {
        // depthwise: groups = C_in = C_out = 6
        // input: [1, 4, 6], weight: [6, 3, 1] (C_in/groups = 1)
        let input = constructors::zeros((1_i32, 4, 6), Dtype::Float32).unwrap();
        let weight = constructors::zeros((6_i32, 3, 1), Dtype::Float32).unwrap();
        let out = conv1d(&input, &weight, 1, 0, 1, /* groups */ 6).expect("depthwise conv1d");
        // L_out = 4 - 3 + 1 = 2
        assert_eq!(out.shape().as_slice(), &[1, 2, 6]);
    }
}
```

- [ ] **Step 1.11: Re-export from `mlx/src/ops/mod.rs`**

Find the existing `pub mod ...;` block and add:

```rust
pub mod binary;
pub mod cast;
pub mod constructors;
pub mod conv;             // NEW
pub mod cumulative;
pub mod indexing;
pub mod macros;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod sort;
pub mod unary;
```

(Add re-exports near the existing top-level wrappers if conv1d should be at the `mlx::` root; the existing pattern in `mlx/src/lib.rs` will dictate. For Phase 3b3 we just need `mlx::ops::conv1d` / `mlx::ops::conv1d_on` accessible.)

- [ ] **Step 1.12: Run mlx unit tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --lib ops::conv
```

Expected: 2 tests pass (`conv1d_shape_basic`, `conv1d_depthwise_shape`).

- [ ] **Step 1.13: Project gate**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

Expected: clean.

- [ ] **Step 1.14: Commit**

```
git add -A
git commit -m "feat(p3b3): cxx-mlx ops::conv1d binding (shim + bridge + safe wrapper)"
```

---

## Task 2: ironmlx::nn::Conv1d

**Files:**
- Create: `ironmlx/src/nn/conv.rs`
- Modify: `ironmlx/src/nn/mod.rs` (`pub mod conv` + re-exports)

### Goal

Wrap `mlx::ops::conv1d` in a project nn-style layer with `Conv1dConfig`, `from_loader` constructor (auto-loads weight + optional bias), and `forward` / `forward_on` methods.

### Steps

- [ ] **Step 2.1: Write the failing construction test**

Create `ironmlx/src/nn/conv.rs` with the test FIRST (TDD failing test):

```rust
//! 1D convolution layer wrapping `mlx::ops::conv1d`.

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn conv1d_construction_from_components() {
        // Synthetic depthwise conv: in=out=6, kernel=3, groups=6
        let weight = Array::zeros((6_i32, 3, 1), Dtype::Float32).unwrap();
        let cfg = Conv1dConfig {
            in_channels: 6,
            out_channels: 6,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 6,
        };
        let conv = Conv1d::new(weight, None, cfg);
        assert_eq!(conv.config().out_channels, 6);
    }
}
```

- [ ] **Step 2.2: Run, verify it fails (struct doesn't exist)**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::conv
```

Expected: compile error.

- [ ] **Step 2.3: Implement Conv1d**

Replace `ironmlx/src/nn/conv.rs` with:

```rust
//! 1D convolution layer wrapping `mlx::ops::conv1d`.
//!
//! Weight layout (matching MLX C++): `[out_channels, kernel_size, in_channels / groups]`.
//! For depthwise: `groups = in_channels = out_channels`, so weight is
//! `[in_channels, kernel_size, 1]`.

use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Configuration for [`Conv1d`].
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
    pub dilation: i32,
    /// `groups = in_channels = out_channels` for depthwise conv.
    pub groups: i32,
}

/// 1D convolution layer.
pub struct Conv1d {
    weight: Array,
    bias: Option<Array>,
    cfg: Conv1dConfig,
}

impl Conv1d {
    /// Production constructor: load weight from `{prefix}.weight` and optional
    /// bias from `{prefix}.bias`.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: Conv1dConfig) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        let bias = loader
            .tensor_opt(&format!("{prefix}.bias"))
            .cloned();
        Ok(Self { weight, bias, cfg })
    }

    /// Test/composition seam: build from in-memory weight and optional bias.
    #[doc(hidden)]
    pub fn new(weight: Array, bias: Option<Array>, cfg: Conv1dConfig) -> Self {
        Self { weight, bias, cfg }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &Conv1dConfig {
        &self.cfg
    }

    /// Forward pass with default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let mut y = mlx::ops::conv::conv1d_on(
            x,
            &self.weight,
            self.cfg.stride,
            self.cfg.padding,
            self.cfg.dilation,
            self.cfg.groups,
            target,
        )?;
        if let Some(b) = &self.bias {
            // Bias broadcasts over (N, L) on last axis (C_out).
            y = (&y + b)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn small_depthwise_conv() -> Conv1d {
        let weight = Array::zeros((6_i32, 3, 1), Dtype::Float32).unwrap();
        let cfg = Conv1dConfig {
            in_channels: 6,
            out_channels: 6,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 6,
        };
        Conv1d::new(weight, None, cfg)
    }

    #[test]
    fn conv1d_construction_from_components() {
        let conv = small_depthwise_conv();
        assert_eq!(conv.config().out_channels, 6);
        assert_eq!(conv.config().groups, 6);
    }

    #[test]
    fn conv1d_forward_shape_depthwise() {
        let conv = small_depthwise_conv();
        // input: [N=1, L=4, C=6]; output: [1, 4-3+1=2, 6]
        let x = Array::zeros((1_i32, 4, 6), Dtype::Float32).unwrap();
        let y = conv.forward(&x).expect("forward");
        assert_eq!(y.shape().as_slice(), &[1, 2, 6]);
        assert_eq!(y.dtype(), Dtype::Float32);
    }
}
```

- [ ] **Step 2.4: Wire into `nn/mod.rs`**

In `ironmlx/src/nn/mod.rs`, find the `pub mod ...` block:

```rust
pub mod attention;
pub mod conv;                         // NEW
pub mod embedding;
pub mod gated_attention;
pub mod linear;
pub mod mlp;
pub mod mrope;
pub mod norm;
```

And the `pub use ...` block:

```rust
pub use attention::{Attention, AttentionConfig};
pub use conv::{Conv1d, Conv1dConfig};   // NEW
pub use embedding::Embedding;
pub use gated_attention::{GatedAttention, GatedAttentionConfig};
pub use linear::Linear;
pub use mlp::Mlp;
pub use mrope::Mrope;
pub use norm::{LayerNorm, RmsNorm};
```

- [ ] **Step 2.5: Run tests, verify pass**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::conv
```

Expected: 2 tests pass.

- [ ] **Step 2.6: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3b3): nn::Conv1d (depthwise-friendly wrapper)"
```

---

## Task 3: ironmlx::nn::RmsNormGated

**Files:**
- Modify: `ironmlx/src/nn/norm.rs` (append `RmsNormGated`)
- Modify: `ironmlx/src/nn/mod.rs` (`pub use norm::RmsNormGated`)

### Goal

Add `RmsNormGated` layer implementing `silu(z) * rms_norm(y)` with fp32 intermediate (mlx-lm `_precise_swiglu` semantics). Composed entirely from existing cxx-mlx ops (`mlx::fast::rms_norm`, `Array::sigmoid`, element-wise mul, `mlx::ops::cast::astype`).

### Steps

- [ ] **Step 3.1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `ironmlx/src/nn/norm.rs` (or create new tests near the bottom):

```rust
    #[test]
    fn rms_norm_gated_no_gate_eq_rms_norm() {
        // With gate=None, RmsNormGated should equal a plain RmsNorm.
        let weight = mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap();
        let norm = RmsNormGated::new(weight, 1e-6);
        // input: [1, 4] fp32 with non-trivial values
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x: Array = (x_data.as_slice(), (1_i32, 4)).try_into().unwrap();

        let y = norm.forward(&x, None).expect("forward no gate");
        assert_eq!(y.shape().as_slice(), &[1, 4]);
        assert_eq!(y.dtype(), Dtype::Float32);
        // Check finiteness — exact RMSNorm value isn't asserted (relative shapes vs gate path matter).
        let v: Vec<f32> = y.to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
```

- [ ] **Step 3.2: Run, verify it fails (struct doesn't exist)**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::norm::tests::rms_norm_gated_no_gate_eq_rms_norm
```

Expected: compile error.

- [ ] **Step 3.3: Implement RmsNormGated**

Append to `ironmlx/src/nn/norm.rs` (after the existing `RmsNorm` block):

```rust
/// RMSNorm with optional sigmoid-style gate, matching mlx-lm's `Qwen3NextRMSNormGated`.
///
/// `forward(hidden, None) → rms_norm(hidden, weight, eps) cast to hidden dtype`.
/// `forward(hidden, Some(gate)) → cast(silu(gate_fp32) * rms_norm_fp32) to hidden dtype`,
/// matching the precise-SwiGLU pattern: fp32 intermediate, cast back to input dtype.
pub struct RmsNormGated {
    weight: Array,
    eps: f32,
}

impl RmsNormGated {
    /// Production constructor: load `{prefix}.weight`.
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        Ok(Self { weight, eps })
    }

    /// Test/composition seam: build from in-memory weight + eps.
    #[doc(hidden)]
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass with default stream.
    pub fn forward(&self, hidden: &Array, gate: Option<&Array>) -> Result<Array> {
        self.forward_on(hidden, gate, ())
    }

    /// Stream-targeted forward.
    pub fn forward_on(
        &self,
        hidden: &Array,
        gate: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden_dtype = hidden.dtype();

        let normed = mlx::fast::rms_norm_on(hidden, Some(&self.weight), self.eps, target)?;

        match gate {
            Some(g) => {
                // Precise SwiGLU: silu(gate) * normed, computed in fp32, cast back.
                let g_f32 = mlx::ops::cast::astype(g, Dtype::Float32)?;
                // silu(x) = x * sigmoid(x)
                let g_sig = g_f32.sigmoid_on(target)?;
                let g_silu = (&g_f32 * &g_sig)?;
                let normed_f32 = mlx::ops::cast::astype(&normed, Dtype::Float32)?;
                let mul = (&g_silu * &normed_f32)?;
                mlx::ops::cast::astype(&mul, hidden_dtype)
            }
            None => mlx::ops::cast::astype(&normed, hidden_dtype),
        }
    }
}
```

> Add `use mlx::Dtype;` to the top of the file if not already imported (RmsNorm doesn't use it directly so it might be missing).

- [ ] **Step 3.4: Run the test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::norm::tests::rms_norm_gated_no_gate_eq_rms_norm
```

Expected: PASS.

- [ ] **Step 3.5: Add gate-on test**

Append to the test module:

```rust
    #[test]
    fn rms_norm_gated_with_gate_finite() {
        // With gate=Some, dispatch should produce finite output (exact silu * rmsnorm
        // values aren't asserted at unit level — those go in the integration test).
        let weight = mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap();
        let norm = RmsNormGated::new(weight, 1e-6);
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x: Array = (x_data.as_slice(), (1_i32, 4)).try_into().unwrap();
        let g_data: Vec<f32> = vec![0.5, -0.5, 0.0, 1.0];
        let g: Array = (g_data.as_slice(), (1_i32, 4)).try_into().unwrap();

        let y = norm.forward(&x, Some(&g)).expect("forward with gate");
        assert_eq!(y.shape().as_slice(), &[1, 4]);
        assert_eq!(y.dtype(), Dtype::Float32);
        let v: Vec<f32> = y.to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        // gate=0 channel (index 2): silu(0) = 0 * sigmoid(0) = 0, so y[2] = 0
        assert!(v[2].abs() < 1e-6, "gate=0 should yield zero output, got {}", v[2]);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::norm::tests::rms_norm_gated_with_gate_finite
```

Expected: PASS.

- [ ] **Step 3.6: Wire into nn/mod.rs**

In `ironmlx/src/nn/mod.rs`, extend the `norm` re-export:

```rust
pub use norm::{LayerNorm, RmsNorm, RmsNormGated};   // RmsNormGated added
```

- [ ] **Step 3.7: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3b3): nn::RmsNormGated (silu(z) * rms_norm(y) precise SwiGLU)"
```

---

## Task 4: `core::cache::GatedDeltaCache`

**Files:**
- Create: `ironmlx/src/core/cache/gated_delta.rs`
- Modify: `ironmlx/src/core/cache/mod.rs` (`pub mod gated_delta` + re-export)

### Goal

Add the `GatedDeltaCache` struct holding `conv_state` (sliding window) and `recurrent_state` (SSM state). Mirrors P2 KVCache's cap-bounded pattern. Statelessness wrt content; consumer (GatedDeltaNet forward) writes both via `update_conv` / `update_recurrent` and bumps offset via `advance`.

### Steps

- [ ] **Step 4.1: Write the failing construction test**

Create `ironmlx/src/core/cache/gated_delta.rs` with TDD failing test first:

```rust
//! Gated Delta SSM cache: conv_state (sliding window) + recurrent_state (SSM state).

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn cache_initial_zeros() {
        // B=1, kernel=4, conv_dim=8, Hv=4, Dv=8, Dk=8, cap=16
        let cache = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("cache new");
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.cap(), 16);
        assert_eq!(cache.conv_state().shape().as_slice(), &[1, 3, 8]);     // kernel-1=3
        assert_eq!(cache.recurrent_state().shape().as_slice(), &[1, 4, 8, 8]);
        assert_eq!(cache.recurrent_state().dtype(), Dtype::Float32);   // recurrent always fp32
    }
}
```

- [ ] **Step 4.2: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::gated_delta::tests::cache_initial_zeros
```

Expected: compile error.

- [ ] **Step 4.3: Implement GatedDeltaCache**

Replace `ironmlx/src/core/cache/gated_delta.rs`:

```rust
//! Gated Delta SSM cache: conv_state (sliding window) + recurrent_state (SSM state).
//!
//! Used by [`crate::nn::GatedDeltaNet`]. Mirrors P2 [`crate::core::cache::KVCache`]'s
//! cap-bounded design — capacity is fixed at construction; `advance` enforces
//! offset ≤ cap.

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::Result;

/// Per-layer cache for [`crate::nn::GatedDeltaNet`].
pub struct GatedDeltaCache {
    /// Sliding window of last `kernel_size - 1` tokens for conv1d. Shape:
    /// `[B, kernel_size - 1, conv_dim]`. Dtype matches input.
    conv_state: Array,
    /// SSM recurrent state. Shape: `[B, Hv, Dv, Dk]`. Always fp32 to avoid
    /// drift across long sequences.
    recurrent_state: Array,
    /// Number of tokens consumed so far.
    offset: i32,
    /// Maximum tokens this cache will accept (prompt + decode).
    cap: i32,
}

impl GatedDeltaCache {
    /// Allocate a fresh cache.
    ///
    /// `cap` must be ≥ 1. Both states start zero-initialized.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cap(
        b: i32,
        kernel_size: i32,
        conv_dim: i32,
        hv: i32,
        dv: i32,
        dk: i32,
        input_dtype: Dtype,
        cap: i32,
    ) -> Result<Self> {
        if cap < 1 {
            return Err(anyhow!("GatedDeltaCache: cap={cap} must be >= 1"));
        }
        if kernel_size < 1 {
            return Err(anyhow!("GatedDeltaCache: kernel_size={kernel_size} must be >= 1"));
        }
        let conv_state = Array::zeros(
            (b, kernel_size - 1, conv_dim),
            input_dtype,
        )?;
        let recurrent_state = Array::zeros((b, hv, dv, dk), Dtype::Float32)?;
        Ok(Self {
            conv_state,
            recurrent_state,
            offset: 0,
            cap,
        })
    }

    pub fn conv_state(&self) -> &Array {
        &self.conv_state
    }

    pub fn recurrent_state(&self) -> &Array {
        &self.recurrent_state
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Replace the conv_state with a freshly-computed sliding window.
    pub fn update_conv(&mut self, new_conv_state: Array) {
        self.conv_state = new_conv_state;
    }

    /// Replace the recurrent_state with the kernel's `state_out`.
    pub fn update_recurrent(&mut self, new_state: Array) {
        self.recurrent_state = new_state;
    }

    /// Bump offset by `n` tokens. Errors if offset+n > cap.
    pub fn advance(&mut self, n: i32) -> Result<()> {
        let new_off = self.offset + n;
        if new_off > self.cap {
            return Err(anyhow!(
                "GatedDeltaCache: offset {} + {} exceeds cap {}",
                self.offset,
                n,
                self.cap
            ));
        }
        self.offset = new_off;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn cache_initial_zeros() {
        let cache = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("cache new");
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.cap(), 16);
        assert_eq!(cache.conv_state().shape().as_slice(), &[1, 3, 8]);
        assert_eq!(cache.recurrent_state().shape().as_slice(), &[1, 4, 8, 8]);
        assert_eq!(cache.recurrent_state().dtype(), Dtype::Float32);
        assert_eq!(cache.conv_state().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn cache_advance_within_cap() {
        let mut cache = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 8)
            .expect("cache new");
        cache.advance(4).expect("advance 4");
        assert_eq!(cache.offset(), 4);
        cache.advance(4).expect("advance to cap");
        assert_eq!(cache.offset(), 8);
    }

    #[test]
    fn cache_advance_beyond_cap_errors() {
        let mut cache = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 4)
            .expect("cache new");
        cache.advance(2).unwrap();
        let r = cache.advance(3);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("exceeds cap"), "msg: {msg}");
    }

    #[test]
    fn cache_rejects_zero_cap() {
        let r = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 0);
        assert!(r.is_err());
    }
}
```

- [ ] **Step 4.4: Wire into cache/mod.rs**

In `ironmlx/src/core/cache/mod.rs`, find the `pub mod ...;` block and add `gated_delta` (alphabetical order):

```rust
pub mod gated_delta;
pub mod kv;                    // existing — adapt name to match
```

Add re-export:

```rust
pub use gated_delta::GatedDeltaCache;
pub use kv::KVCache;            // existing
```

(Adjust to match the actual existing names; the exact filename for KVCache may be `kv_cache.rs` or similar.)

- [ ] **Step 4.5: Run tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::gated_delta
```

Expected: 4 tests pass.

- [ ] **Step 4.6: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3b3): core::cache::GatedDeltaCache (conv_state + recurrent_state)"
```

---

## Task 5: `gated_delta_step` metal_kernel (2 variants)

**Files:**
- Create: `ironmlx/src/nn/gated_delta_net.rs` (kernel-only at this stage; main module in T6)

### Goal

Implement the recurrent SSM kernel in Metal. Two variants — no-mask and masked — built from a single Rust function via string substitution (mirrors mlx-lm's `_make_gated_delta_kernel(has_mask)` factory). 6 templates: `Dk, Dv, Hk, Hv, InT, StT`. Inputs: `q, k, v, g, beta, state_in, T` (+ `mask` for masked variant). Outputs: `y, state_out`.

### Steps

- [ ] **Step 5.1: Write the failing kernel-builds-and-dispatches test**

Create `ironmlx/src/nn/gated_delta_net.rs` with kernel-only scaffolding + 1 failing test:

```rust
//! Qwen3.5 / Qwen3-Next GatedDeltaNet — recurrent SSM with delta rule.
//!
//! T5 (current): just the `gated_delta_step` metal_kernel (2 variants).
//! T6 will add the GatedDeltaNet main struct around it.

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype, Shape};

    #[test]
    fn gated_delta_step_kernel_links() {
        // Smoke-link: build the no-mask kernel and dispatch it on tiny zeros.
        // B=1, T=1, Hk=1, Hv=1, Dk=8, Dv=8.
        let kernel = build_gated_delta_kernel(false).expect("build kernel");

        let q = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 8), Dtype::Float32).unwrap();
        // T as 0-dim int32 array (so MLX passes it as `device const int32_t& T` reference).
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 8]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1) // (32 simdgroup threads, Dv=8, B*Hv=1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 8)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch");

        let _y = outputs.take_at(0).expect("y");
        let _state = outputs.take_at(0).expect("state");
    }
}
```

- [ ] **Step 5.2: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net::tests::gated_delta_step_kernel_links
```

Expected: compile error (`build_gated_delta_kernel` doesn't exist).

- [ ] **Step 5.3: Implement `build_gated_delta_kernel`**

Replace `ironmlx/src/nn/gated_delta_net.rs` content with:

```rust
//! Qwen3.5 / Qwen3-Next GatedDeltaNet — recurrent SSM with delta rule.
//!
//! T5 (current): the `gated_delta_step` metal_kernel (2 variants — no-mask + masked).
//! Mirrors mlx-lm's `_make_gated_delta_kernel(has_mask)` from
//! `/Volumes/Dev/mlx-lm/mlx_lm/models/gated_delta.py:13-115`.
//!
//! T6 will add the `GatedDeltaNet` main struct around the kernel.
//!
//! Templates: `Dk, Dv, Hk, Hv` (i32), `InT, StT` (Dtype).
//! Grid: `(32, Dv, B * Hv)`; threadgroup: `(32, 4, 1)`.

use mlx::MetalKernel;

use crate::Result;

/// Build the `gated_delta_step` MetalKernel (no-mask or masked variant).
///
/// The shader source is identical between variants except for the per-token
/// guard expression (`mask_clause`). MLX's `metal_kernel` machinery auto-injects
/// `<name>_shape` / `<name>_strides` / `<name>_ndim` for input arrays referenced
/// in the source.
///
/// `T` is passed as a 0-dim int32 array, which MLX treats as `device const
/// int32_t& T` — usable directly as an integer in the shader (e.g.
/// `for (int t = 0; t < T; ++t)`).
pub(crate) fn build_gated_delta_kernel(masked: bool) -> Result<MetalKernel> {
    let mask_clause = if masked { "mask[b_idx * T + t]" } else { "true" };
    let src = format!(
        r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_clause}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
        "#
    );

    let name = if masked {
        "ironmlx_gated_delta_masked"
    } else {
        "ironmlx_gated_delta"
    };

    let inputs: &[&str] = if masked {
        &["q", "k", "v", "g", "beta", "state_in", "T", "mask"]
    } else {
        &["q", "k", "v", "g", "beta", "state_in", "T"]
    };

    Ok(MetalKernel::builder(name)
        .inputs(inputs)
        .outputs(&["y", "state_out"])
        .source(&src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype, Shape};

    #[test]
    fn gated_delta_step_kernel_links() {
        let kernel = build_gated_delta_kernel(false).expect("build kernel");

        let q = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 8), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 8]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 8)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch");

        let _y = outputs.take_at(0).expect("y");
        let _state = outputs.take_at(0).expect("state");
    }

    #[test]
    fn gated_delta_step_masked_zero_path() {
        // mask=0 everywhere: output should be 0, state unchanged.
        // Use non-zero state_in to verify state isn't accidentally modified.
        let kernel = build_gated_delta_kernel(true).expect("build masked kernel");

        // Initial state has values [1.0; 64] (8*8 = 64 elements).
        let init_state_data: Vec<f32> = (0..64).map(|_| 1.0_f32).collect();
        let state_in: Array = (init_state_data.as_slice(), (1_i32, 1, 8, 8))
            .try_into()
            .unwrap();

        let q = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();
        // mask: [B*T] = [1*1 = 1] all-zero (masked out)
        let mask = Array::zeros((1_i32,), Dtype::Bool).unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr, &mask])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 8]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 8)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch masked");

        let y = outputs.take_at(0).expect("y");
        let state_out = outputs.take_at(0).expect("state_out");

        // y must be all-zero (else branch sets `y[dv_idx] = 0`).
        let y_f32 = mlx::ops::cast::astype(&y, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_f32.to_vec().unwrap();
        assert!(yv.iter().all(|x| x.abs() < 1e-6), "masked output not zero: {:?}", yv);

        // state_out must equal state_in (no update under mask=0 — kernel writes
        // back the unchanged register-cached state at the end).
        let sv: Vec<f32> = state_out.to_vec().unwrap();
        assert!(sv.iter().all(|x| (x - 1.0).abs() < 1e-6), "state changed under mask=0: {:?}", sv);
    }
}
```

> **Note on the masked-path zero test**: under `mask=0`, the kernel:
> 1. Skips the inner kv_mem / delta / out computation block.
> 2. Writes `y[dv_idx] = 0`.
> 3. Continues to the next `t` (no state update).
> 4. After the time loop, writes the unchanged register-cached `state[i]` back to `o_state`.
>
> So state_out should equal state_in. The test verifies both invariants.

- [ ] **Step 5.4: Wire into nn/mod.rs (provisional)**

In `ironmlx/src/nn/mod.rs`, add the new module declaration BUT do not yet add a re-export (T6 will add the `GatedDeltaNet` re-export):

```rust
pub mod attention;
pub mod conv;
pub mod embedding;
pub mod gated_attention;
pub mod gated_delta_net;          // NEW (T5; main struct lands T6)
pub mod linear;
pub mod mlp;
pub mod mrope;
pub mod norm;
```

- [ ] **Step 5.5: Run tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net
```

Expected: 2 tests pass (`gated_delta_step_kernel_links`, `gated_delta_step_masked_zero_path`).

> **If `gated_delta_step_masked_zero_path` fails because Mask dtype isn't supported as kernel input**: try `Dtype::Uint8` (0/1 byte) instead of `Dtype::Bool`. mlx-lm passes mask as bool; if MLX C++ rejects bool buffers, fall back to u8 + `if (mask[idx] != 0)` shader-side.

- [ ] **Step 5.6: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3b3): gated_delta_step metal_kernel (no-mask + masked variants)"
```

---

## Task 6: `nn::GatedDeltaNet` main module

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (add main struct above the kernel function)
- Modify: `ironmlx/src/nn/mod.rs` (add `GatedDeltaNet` + `GatedDeltaNetConfig` re-exports)

### Goal

Wire all components into the main `GatedDeltaNet` module. Implements the 7-step forward path: project (qkv + z + a + b), conv1d + silu, split + reshape, q/k rms_norm, compute_g via mlx::compile, sigmoid beta, dispatch gated_delta_step kernel, RmsNormGated + out_proj.

### Steps

- [ ] **Step 6.1: Write the failing main-module construction test**

Append to the `tests` mod in `ironmlx/src/nn/gated_delta_net.rs`:

```rust
    fn small_gdn_components() -> GatedDeltaNet {
        // Synthetic small model:
        // hidden=32, num_v_heads=4, num_k_heads=2, head_k_dim=8, head_v_dim=8,
        // conv_kernel=4, eps=1e-6
        let cfg = GatedDeltaNetConfig {
            hidden_size: 32,
            num_v_heads: 4,
            num_k_heads: 2,
            head_k_dim: 8,
            head_v_dim: 8,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        };
        // key_dim = 2*8 = 16; value_dim = 4*8 = 32
        // qkv proj output = key_dim*2 + value_dim = 16+16+32 = 64
        // conv_dim = key_dim*2 + value_dim = 64
        // out_proj input = value_dim = 32

        let qkv_w = Array::zeros((64_i32, 32), Dtype::Float32).unwrap();
        let z_w = Array::zeros((32_i32, 32), Dtype::Float32).unwrap();
        let b_w = Array::zeros((4_i32, 32), Dtype::Float32).unwrap();
        let a_w = Array::zeros((4_i32, 32), Dtype::Float32).unwrap();
        let conv_w = Array::zeros((64_i32, 4, 1), Dtype::Float32).unwrap();
        let norm_w = mlx::ops::constructors::ones((8_i32,), Dtype::Float32).unwrap();
        let out_w = Array::zeros((32_i32, 32), Dtype::Float32).unwrap();
        let a_log = Array::zeros((4_i32,), Dtype::Float32).unwrap();
        let dt_bias = mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap();

        GatedDeltaNet::from_components(
            crate::nn::Linear::new_fp(qkv_w, None),
            crate::nn::Linear::new_fp(z_w, None),
            crate::nn::Linear::new_fp(b_w, None),
            crate::nn::Linear::new_fp(a_w, None),
            crate::nn::Conv1d::new(
                conv_w,
                None,
                crate::nn::Conv1dConfig {
                    in_channels: 64,
                    out_channels: 64,
                    kernel_size: 4,
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: 64, // depthwise
                },
            ),
            crate::nn::RmsNormGated::new(norm_w, cfg.rms_norm_eps),
            crate::nn::Linear::new_fp(out_w, None),
            a_log,
            dt_bias,
            cfg,
        )
    }

    #[test]
    fn gdn_construction_carries_config() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        assert_eq!(cfg.num_v_heads, 4);
        assert_eq!(cfg.num_k_heads, 2);
        assert_eq!(cfg.conv_kernel_size, 4);
    }
```

- [ ] **Step 6.2: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net::tests::gdn_construction_carries_config
```

Expected: compile error.

- [ ] **Step 6.3: Implement GatedDeltaNet struct + config + from_loader/from_components + stubbed forward**

Add to the top of `ironmlx/src/nn/gated_delta_net.rs` (above the `build_gated_delta_kernel` fn, but after the module doc):

```rust
use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::compile::{CompiledFn, ShapeMode};
use mlx::ops::shape::{concatenate, split_n_on};
use mlx::{Array, Dtype, MetalKernel, StreamOrDevice};

use crate::core::cache::GatedDeltaCache;
use crate::core::Loader;
use crate::nn::{Conv1d, Conv1dConfig, Linear, RmsNormGated};
use crate::Result;

/// Configuration for [`GatedDeltaNet`].
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNetConfig {
    pub fn key_dim(&self) -> i32 {
        self.num_k_heads * self.head_k_dim
    }
    pub fn value_dim(&self) -> i32 {
        self.num_v_heads * self.head_v_dim
    }
    pub fn conv_dim(&self) -> i32 {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Qwen3.5 / Qwen3-Next "linear attention" branch — recurrent SSM with
/// delta rule and scalar gating.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,         // [num_v_heads]
    dt_bias: Array,       // [num_v_heads]
    cfg: GatedDeltaNetConfig,
    compute_g_compiled: OnceLock<CompiledFn>,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
}

impl GatedDeltaNet {
    /// Production constructor: load all 7 weight tensors + a_log + dt_bias.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self> {
        let in_proj_qkv = Linear::from_loader(loader, &format!("{prefix}.in_proj_qkv"))?;
        let in_proj_z = Linear::from_loader(loader, &format!("{prefix}.in_proj_z"))?;
        let in_proj_b = Linear::from_loader(loader, &format!("{prefix}.in_proj_b"))?;
        let in_proj_a = Linear::from_loader(loader, &format!("{prefix}.in_proj_a"))?;
        let conv1d_cfg = Conv1dConfig {
            in_channels: cfg.conv_dim(),
            out_channels: cfg.conv_dim(),
            kernel_size: cfg.conv_kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: cfg.conv_dim(), // depthwise
        };
        let conv1d = Conv1d::from_loader(loader, &format!("{prefix}.conv1d"), conv1d_cfg)?;
        let norm = RmsNormGated::from_loader(loader, &format!("{prefix}.norm"), cfg.rms_norm_eps)?;
        let out_proj = Linear::from_loader(loader, &format!("{prefix}.out_proj"))?;
        let a_log = loader.tensor(&format!("{prefix}.A_log"))?.clone();
        let dt_bias = loader.tensor(&format!("{prefix}.dt_bias"))?.clone();

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            compute_g_compiled: OnceLock::new(),
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
        })
    }

    /// Test/composition seam.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
        conv1d: Conv1d,
        norm: RmsNormGated,
        out_proj: Linear,
        a_log: Array,
        dt_bias: Array,
        cfg: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            compute_g_compiled: OnceLock::new(),
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
        }
    }

    pub fn config(&self) -> &GatedDeltaNetConfig {
        &self.cfg
    }

    /// Forward pass with default stream.
    ///
    /// **Stub at T6.1**: returns `Err`. Real body lands in T6.4.
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> Result<Array> {
        self.forward_on(x, mask, cache, ())
    }

    /// Stream-targeted forward — currently stubbed.
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let _ = (x, mask, cache, target);
        Err(anyhow!(
            "GatedDeltaNet::forward not implemented at T6.1 — body lands in T6.4"
        ))
    }
}
```

- [ ] **Step 6.4: Run the construction test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net::tests::gdn_construction_carries_config
```

Expected: PASS.

- [ ] **Step 6.5: Implement compute_g pipeline (mlx::compile cell)**

Add a private method on `GatedDeltaNet`. Find the `impl GatedDeltaNet { ... }` block and append before the closing `}`:

```rust
    /// Build the compute_g pipeline:
    ///   `g = exp(-exp(A_log) * softplus(a + dt_bias))`
    ///
    /// where `softplus(x) = where(x > 20, x, log(1 + exp(x)))` (numerically stable).
    fn build_compute_g_pipeline() -> Result<CompiledFn> {
        let pipeline = move |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let a_log = inputs[0];      // [num_v_heads]
            let a = inputs[1];          // [B, T, num_v_heads]
            let dt_bias = inputs[2];    // [num_v_heads]

            // softplus(a + dt_bias) — numerically stable
            let x = (&(a) + dt_bias)?;
            let twenty: Array = (&[20.0_f32][..], ()).try_into()?;
            let zeros = a.zeros_like()?;
            // log(1 + exp(x)) via logaddexp(0, x)
            let safe = zeros.logaddexp(&x)?;
            let cond = x.greater(&twenty)?;
            let sp = cond.where_(&x, &safe)?;

            // exp(A_log) cast to fp32, multiply broadcast to [B, T, num_v_heads]
            let a_log_f32 = mlx::ops::cast::astype(a_log, Dtype::Float32)?;
            let exp_alog = a_log_f32.exp()?;
            let neg_exp_alog = exp_alog.negative()?;
            // g = exp(neg_exp_alog * sp)
            let inner = (&neg_exp_alog * &sp)?;
            let g = inner.exp()?;
            Ok(vec![g])
        };

        mlx::compile::compile(pipeline, ShapeMode::Shapeless).map_err(anyhow::Error::from)
    }
```

> **API note**: `Array::greater`, `Array::negative`, `Array::zeros_like`, `Array::logaddexp` — verify each exists in `mlx/src/array.rs`. If `negative` doesn't exist, use `(&zeros - &x)?` or multiply by `-1.0` array. `greater` may be `gt`. Adjust at implementation time.
>
> If the pipeline closure can't return `mlx::Result`, switch to closure returning `crate::Result` and adjust `.map_err`. The pattern matches T1's cos_sin pipeline in P3b1.

- [ ] **Step 6.6: Implement forward body (replace stub)**

Find the existing `forward_on` stub:

```rust
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let _ = (x, mask, cache, target);
        Err(anyhow!(
            "GatedDeltaNet::forward not implemented at T6.1 — body lands in T6.4"
        ))
    }
```

Replace with the full 7-step body (long; full code below):

```rust
    /// Stream-targeted forward — Qwen3-Next gated delta net algorithm.
    ///
    /// 7 steps:
    ///   1. project qkv, z, a, b
    ///   2. conv1d + silu (with conv_state from cache prepended)
    ///   3. split + reshape per-head
    ///   4. q/k rms_norm (no weight)
    ///   5. compute_g via mlx::compile
    ///   6. beta = sigmoid(b)
    ///   7. dispatch gated_delta_step kernel; update cache; norm + out_proj
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let dims = x.shape();
        let dims = dims.as_slice();
        let batch = dims[0];
        let seq = dims[1];

        // Step 1: projections
        let qkv = self.in_proj_qkv.forward_on(x, target)?;     // [B, S, conv_dim]
        let z = self.in_proj_z.forward_on(x, target)?;         // [B, S, value_dim]
        let a = self.in_proj_a.forward_on(x, target)?;         // [B, S, num_v_heads]
        let b = self.in_proj_b.forward_on(x, target)?;         // [B, S, num_v_heads]

        // Step 2a: prepend conv_state
        let conv_input = match cache.as_deref_mut() {
            Some(c) => concatenate(&[c.conv_state(), &qkv], 1)?,
            None => {
                // Synthesize a fresh zero conv_state of shape [B, kernel_size-1, conv_dim].
                let zeros = Array::zeros(
                    (batch, self.cfg.conv_kernel_size - 1, self.cfg.conv_dim()),
                    qkv.dtype(),
                )?;
                concatenate(&[&zeros, &qkv], 1)?
            }
        };

        // Step 2b: conv1d + silu
        let conv_out = self.conv1d.forward_on(&conv_input, target)?;
        // silu(x) = x * sigmoid(x)
        let conv_out_sig = conv_out.sigmoid_on(target)?;
        let conv_out = (&conv_out * &conv_out_sig)?;

        // Step 2c: update conv_state cache (last kernel_size-1 tokens of conv_input)
        if let Some(c) = cache.as_deref_mut() {
            let n_keep = self.cfg.conv_kernel_size - 1;
            // slice last n_keep tokens along axis=1
            let conv_input_dims = conv_input.shape();
            let total_len = conv_input_dims.as_slice()[1];
            let new_conv_state = mlx::ops::indexing::slice(
                &conv_input,
                vec![0_i32, total_len - n_keep, 0].as_slice(),
                vec![batch, total_len, self.cfg.conv_dim()].as_slice(),
            )?;
            c.update_conv(new_conv_state);
        }

        // Step 3: split + reshape per-head
        // conv_out shape: [B, S, conv_dim] = [B, S, key_dim*2 + value_dim]
        // Split at [key_dim, 2*key_dim] → 3 segments [B, S, key_dim], [B, S, key_dim], [B, S, value_dim]
        let split_at = vec![self.cfg.key_dim(), 2 * self.cfg.key_dim()];
        let parts = mlx::ops::shape::split_at_on(&conv_out, &split_at, -1, target)?;
        let q_flat = &parts[0]; // [B, S, num_k_heads * head_k_dim]
        let k_flat = &parts[1]; // [B, S, num_k_heads * head_k_dim]
        let v_flat = &parts[2]; // [B, S, num_v_heads * head_v_dim]

        let q_per_head = q_flat.reshape_on(
            (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
            target,
        )?;
        let k_per_head = k_flat.reshape_on(
            (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
            target,
        )?;
        let v_per_head = v_flat.reshape_on(
            (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
            target,
        )?;

        // Step 4: q/k rms_norm (no weight)
        let inv_scale = 1.0 / (self.cfg.head_k_dim as f32).sqrt();
        let q_normed = mlx::fast::rms_norm_on(&q_per_head, None, 1e-6, target)?;
        let q_scaled = (&q_normed * (inv_scale * inv_scale))?;
        let k_normed = mlx::fast::rms_norm_on(&k_per_head, None, 1e-6, target)?;
        let k_scaled = (&k_normed * inv_scale)?;

        // Step 5: compute_g via compile cell
        let cg = self.compute_g_compiled.get_or_init(|| {
            Self::build_compute_g_pipeline()
                .expect("build_compute_g_pipeline cannot fail at first call")
        });
        let g_outs = cg.invoke(&[&self.a_log, &a, &self.dt_bias])?;
        let g = g_outs.into_iter().next().expect("compute_g returns 1");

        // Step 6: beta = sigmoid(b)
        let beta = b.sigmoid_on(target)?;

        // Step 7a: build/get the appropriate kernel
        let kernel = if mask.is_some() {
            self.kernel_masked.get_or_init(|| {
                build_gated_delta_kernel(true).expect("build masked kernel")
            })
        } else {
            self.kernel_no_mask.get_or_init(|| {
                build_gated_delta_kernel(false).expect("build no-mask kernel")
            })
        };

        // Step 7b: get state_in from cache (or fresh zeros)
        let state_in = match cache.as_deref() {
            Some(c) => c.recurrent_state().clone(),
            None => Array::zeros(
                (batch, self.cfg.num_v_heads, self.cfg.head_v_dim, self.cfg.head_k_dim),
                Dtype::Float32,
            )?,
        };

        // Step 7c: T as 0-dim int32 array
        let t_arr: Array = (&[seq][..], ()).try_into()?;

        let in_dtype = x.dtype();
        let st_dtype = Dtype::Float32;
        let y_shape = mlx::Shape::from(vec![
            batch,
            seq,
            self.cfg.num_v_heads,
            self.cfg.head_v_dim,
        ]);
        let state_shape = mlx::Shape::from(vec![
            batch,
            self.cfg.num_v_heads,
            self.cfg.head_v_dim,
            self.cfg.head_k_dim,
        ]);

        // Step 7d: dispatch
        let mut kernel_inputs: Vec<&Array> =
            vec![&q_scaled, &k_scaled, &v_per_head, &g, &beta, &state_in, &t_arr];
        if let Some(m) = mask {
            kernel_inputs.push(m);
        }

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&kernel_inputs)
            .output_shapes(&[y_shape, state_shape])
            .output_dtypes(&[in_dtype, st_dtype])
            .grid(32, self.cfg.head_v_dim, batch * self.cfg.num_v_heads)
            .threadgroup(32, 4, 1)
            .template_int("Dk", self.cfg.head_k_dim)
            .template_int("Dv", self.cfg.head_v_dim)
            .template_int("Hk", self.cfg.num_k_heads)
            .template_int("Hv", self.cfg.num_v_heads)
            .template_dtype("InT", in_dtype)
            .template_dtype("StT", st_dtype)
            .dispatch()?;

        let y = outputs.take_at(0)?;            // [B, S, Hv, Dv]
        let new_state = outputs.take_at(0)?;    // [B, Hv, Dv, Dk]

        // Step 7e: update cache recurrent_state, advance offset
        if let Some(c) = cache.as_deref_mut() {
            c.update_recurrent(new_state);
            c.advance(seq)?;
        }

        // Step 8: RmsNormGated(y, z) + reshape + out_proj
        let z_per_head = z.reshape_on(
            (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
            target,
        )?;
        let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
        let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
        self.out_proj.forward_on(&normed_flat, target)
    }
```

> **API verification at impl time** (these may need adjustment):
> - `mlx::ops::shape::split_at_on(arr, &[indices], axis, target)` — verify against `mlx/src/ops/shape.rs:208`. The signature was `pub fn split_at_on(...)`.
> - `Array::greater(rhs)` vs `Array::gt(rhs)` — implementer to find the actual name.
> - `mlx::Shape::from(Vec<i32>)` — verified above.

- [ ] **Step 6.7: Add forward shape/dtype tests**

Append to the tests mod:

```rust
    #[test]
    fn gdn_forward_shape_dtype_no_cache() {
        let gdn = small_gdn_components();
        // x: [B=1, S=4, hidden=32] — note: small zeros so the SSM dispatch
        // succeeds even with our trivial weights.
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let out = gdn
            .forward(&x, None, None)
            .expect("forward no cache");
        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    fn gdn_forward_with_cache_advances_offset() {
        let gdn = small_gdn_components();
        let mut cache = GatedDeltaCache::new_with_cap(
            1, // B
            4, // kernel_size
            64, // conv_dim
            4, // Hv
            8, // Dv
            8, // Dk
            Dtype::Float32,
            16, // cap
        )
        .expect("cache");
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let _out = gdn
            .forward(&x, None, Some(&mut cache))
            .expect("forward with cache");
        assert_eq!(cache.offset(), 4);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net
```

Expected: 5 tests pass total (`gated_delta_step_kernel_links`, `gated_delta_step_masked_zero_path`, `gdn_construction_carries_config`, `gdn_forward_shape_dtype_no_cache`, `gdn_forward_with_cache_advances_offset`).

- [ ] **Step 6.8: Wire main module re-exports into nn/mod.rs**

In `ironmlx/src/nn/mod.rs`, add to the `pub use ...` block:

```rust
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
```

- [ ] **Step 6.9: Project gate + commit**

```
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release

git add -A
git commit -m "feat(p3b3): GatedDeltaNet main module (7-step forward + compute_g compile cell)"
```

---

## Task 7: Python fixture + numerical correctness integration test

**Files:**
- Create: `ironmlx/tests/fixtures/p3b3_gated_delta_net/README.md`
- Create: `ironmlx/tests/fixtures/p3b3_gated_delta_net/gen_fixture.py`
- Create: `ironmlx/tests/fixtures/p3b3_gated_delta_net/*.npy` (~14 files)
- Create: `ironmlx/tests/p3b3_gated_delta_net.rs`

### Goal

Generate a small-scale (~5KB) fixture via an independent Python re-implementation of the gated delta algorithm (using `gated_delta_ops`-style ops, NOT the Metal kernel — to avoid circular-validation against ourselves). Add a Rust integration test that loads the fixture and verifies forward output matches at bf16/fp32 atol=1e-3.

### Steps

- [ ] **Step 7.1: Create README**

```bash
mkdir -p ironmlx/tests/fixtures/p3b3_gated_delta_net
```

Create `ironmlx/tests/fixtures/p3b3_gated_delta_net/README.md`:

````markdown
# P3b3 GatedDeltaNet fixtures

Reference data for `nn::GatedDeltaNet` numerical-correctness tests.

The reference is an **independent re-implementation** of the gated delta
algorithm using `mlx.core` primitives (the `gated_delta_ops`-style sequential
loop, NOT mlx-lm's Metal kernel). This avoids circular validation if our
Metal kernel and mlx-lm's Metal kernel share the same bug.

Small-scale synthetic config: B=1, S=4, num_v_heads=4, num_k_heads=2,
head_k_dim=8, head_v_dim=8, hidden=32, conv_kernel=4, eps=1e-6.

## Regenerate

Requires the `mlx` Python version pinned in `gen_fixture.py`. Re-run after
any algorithmic change to the reference.

```bash
cd ironmlx/tests/fixtures/p3b3_gated_delta_net
python gen_fixture.py
```

Generated `.npy` files (committed to git, ~5 KB total):

| File | Shape | Dtype |
|---|---|---|
| `input_x.npy` | `[1, 4, 32]` | bf16 |
| `qkv_proj_weight.npy` | `[64, 32]` | bf16 |
| `z_proj_weight.npy` | `[32, 32]` | bf16 |
| `a_proj_weight.npy` | `[4, 32]` | bf16 |
| `b_proj_weight.npy` | `[4, 32]` | bf16 |
| `conv1d_weight.npy` | `[64, 4, 1]` | bf16 |
| `norm_weight.npy` | `[8]` | fp32 |
| `out_proj_weight.npy` | `[32, 32]` | bf16 |
| `A_log.npy` | `[4]` | fp32 |
| `dt_bias.npy` | `[4]` | fp32 |
| `expected_output.npy` | `[1, 4, 32]` | (varies; see Note below) |

> **Note on output dtype**: `mx.fast.rms_norm(bf16, fp32_weight)` promotes to
> fp32, which propagates through the rest of the chain. The integration test
> asserts the dtype matches whatever the fixture produces.
````

- [ ] **Step 7.2: Create gen_fixture.py**

Create `ironmlx/tests/fixtures/p3b3_gated_delta_net/gen_fixture.py`:

```python
"""Generate P3b3 GatedDeltaNet fixtures.

Independent re-implementation using `mlx.core` primitives, NOT mlx-lm's
Metal kernel (avoids circular validation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as mxnn

EXPECTED_MLX_VERSION = "0.31.1"
_mlx_version = mx.__version__
if _mlx_version != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {_mlx_version}, expected "
        f"{EXPECTED_MLX_VERSION}. Bump and regenerate the .npy fixtures."
    )

OUT_DIR = Path(__file__).parent

# ---- Small synthetic config ----
B, S = 1, 4
HV = 4
HK = 2
DK = 8
DV = 8
HIDDEN = HV * DV  # 32
KEY_DIM = HK * DK  # 16
VALUE_DIM = HV * DV  # 32
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 16+16+32 = 64
CONV_KERNEL = 4
EPS = 1e-6


def _ref_gated_delta_step_ops(q, k, v, g, beta, state):
    """Sequential ops-based step (per mlx-lm gated_delta._gated_delta_step_ops)."""
    # q, k: [B, H, Dk]; v: [B, H, Dv]; g: [B, H]; beta: [B, H]; state: [B, H, Dv, Dk]
    decay = g[..., None, None]
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    return y.astype(q.dtype), state


def _ref_softplus(x):
    return mx.where(x > 20, x, mx.logaddexp(mx.zeros_like(x), x))


def _ref_compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * _ref_softplus(a + dt_bias))


def _ref_rms_norm(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


def _ref_silu(x):
    return x * mx.sigmoid(x)


def _ref_gated_delta_net(
    x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias
):
    """Independent ref impl of GatedDeltaNet (no cache, no mask)."""
    # x: [B, S, hidden]
    qkv = x @ qkv_w.T   # [B, S, conv_dim]
    z = x @ z_w.T       # [B, S, value_dim]
    a = x @ a_w.T       # [B, S, num_v_heads]
    b = x @ b_w.T       # [B, S, num_v_heads]

    # conv1d depthwise. mlx.fast.conv1d with manual prepend of zero conv_state.
    conv_state = mx.zeros((B, CONV_KERNEL - 1, CONV_DIM), dtype=qkv.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)  # [B, S+K-1, conv_dim]
    # mlx.core.conv1d weight: [C_out, K, C_in/groups]; here groups=conv_dim → C_in/groups=1
    conv_out = mx.conv1d(conv_input, conv_w, stride=1, padding=0, groups=CONV_DIM)
    # output shape: [B, S, conv_dim]
    conv_out = _ref_silu(conv_out)

    # Split last axis: [key_dim, key_dim, value_dim]
    q_flat = conv_out[..., :KEY_DIM]
    k_flat = conv_out[..., KEY_DIM:2*KEY_DIM]
    v_flat = conv_out[..., 2*KEY_DIM:]

    q = q_flat.reshape(B, S, HK, DK)
    k = k_flat.reshape(B, S, HK, DK)
    v = v_flat.reshape(B, S, HV, DV)

    inv_scale = DK ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, EPS)
    k = inv_scale * mx.fast.rms_norm(k, None, EPS)

    g = _ref_compute_g(A_log, a, dt_bias)  # [B, S, HV]
    beta = mx.sigmoid(b)                   # [B, S, HV]

    # GQA repeat: q, k from HK heads -> HV heads
    repeat = HV // HK
    q_rep = mx.repeat(q, repeat, axis=-2)  # [B, S, HV, DK]
    k_rep = mx.repeat(k, repeat, axis=-2)

    # Sequential SSM loop
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    ys = []
    for t in range(S):
        y_t, state = _ref_gated_delta_step_ops(
            q_rep[:, t], k_rep[:, t], v[:, t], g[:, t], beta[:, t], state,
        )
        ys.append(y_t)
    y = mx.stack(ys, axis=1)  # [B, S, HV, DV]

    # RmsNormGated: silu(z) * rms_norm(y) in fp32, cast back
    z_per_head = z.reshape(B, S, HV, DV)
    y_normed = mx.fast.rms_norm(y, norm_w, EPS)
    z_silu = _ref_silu(z_per_head.astype(mx.float32))
    y_normed_f32 = y_normed.astype(mx.float32)
    out_per_head = (z_silu * y_normed_f32).astype(y.dtype)

    out_flat = out_per_head.reshape(B, S, HIDDEN)
    out = out_flat @ out_w.T
    return out


def main():
    np.random.seed(46)

    def randn(shape, dtype=mx.bfloat16, scale=0.1):
        a = np.random.randn(*shape).astype(np.float32) * scale
        return mx.array(a).astype(dtype)

    x = randn((B, S, HIDDEN))
    qkv_w = randn((CONV_DIM, HIDDEN))
    z_w = randn((VALUE_DIM, HIDDEN))
    a_w = randn((HV, HIDDEN))
    b_w = randn((HV, HIDDEN))
    conv_w = randn((CONV_DIM, CONV_KERNEL, 1))  # depthwise, in_per_group=1
    norm_w = randn((DV,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    out_w = randn((HIDDEN, VALUE_DIM))
    A_log = randn((HV,), dtype=mx.float32, scale=1.0)  # log of A in [0, 16]
    dt_bias = randn((HV,), dtype=mx.float32, scale=0.5) + mx.array([1.0])

    out = _ref_gated_delta_net(
        x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias,
    )
    mx.eval(x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias, out)

    def save(name, arr):
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_x", x)
    save("qkv_proj_weight", qkv_w)
    save("z_proj_weight", z_w)
    save("a_proj_weight", a_w)
    save("b_proj_weight", b_w)
    save("conv1d_weight", conv_w)
    save("norm_weight", norm_w)
    save("out_proj_weight", out_w)
    save("A_log", A_log)
    save("dt_bias", dt_bias)
    save("expected_output", out)


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd ironmlx/tests/fixtures/p3b3_gated_delta_net
python gen_fixture.py
```

Expected: 11 .npy files written.

- [ ] **Step 7.3: Create the integration test**

Create `ironmlx/tests/p3b3_gated_delta_net.rs`:

```rust
//! P3b3 GatedDeltaNet numerical-correctness integration test.

use mlx::{Array, Dtype};

use ironmlx::nn::{Conv1d, Conv1dConfig, GatedDeltaNet, GatedDeltaNetConfig, Linear, RmsNormGated};

const FIXTURE_DIR: &str = "tests/fixtures/p3b3_gated_delta_net";

const HV: i32 = 4;
const HK: i32 = 2;
const DK: i32 = 8;
const DV: i32 = 8;
const HIDDEN: i32 = HV * DV;       // 32
const CONV_DIM: i32 = HK * DK * 2 + HV * DV;  // 16+16+32 = 64
const CONV_KERNEL: i32 = 4;

fn load(name: &str) -> Array {
    let path = format!("{FIXTURE_DIR}/{name}.npy");
    mlx::io::load_npy(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).unwrap();
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).unwrap();
    let av: Vec<f32> = a32.to_vec().unwrap();
    let bv: Vec<f32> = b32.to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn gated_delta_net_matches_python_fixture() {
    let qkv_w = load("qkv_proj_weight");
    let z_w = load("z_proj_weight");
    let a_w = load("a_proj_weight");
    let b_w = load("b_proj_weight");
    let conv_w = load("conv1d_weight");
    let norm_w = load("norm_weight");
    let out_w = load("out_proj_weight");
    let a_log = load("A_log");
    let dt_bias = load("dt_bias");

    let cfg = GatedDeltaNetConfig {
        hidden_size: HIDDEN,
        num_v_heads: HV,
        num_k_heads: HK,
        head_k_dim: DK,
        head_v_dim: DV,
        conv_kernel_size: CONV_KERNEL,
        rms_norm_eps: 1e-6,
    };

    let conv1d = Conv1d::new(
        conv_w,
        None,
        Conv1dConfig {
            in_channels: CONV_DIM,
            out_channels: CONV_DIM,
            kernel_size: CONV_KERNEL,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: CONV_DIM,
        },
    );

    let gdn = GatedDeltaNet::from_components(
        Linear::new_fp(qkv_w, None),
        Linear::new_fp(z_w, None),
        Linear::new_fp(b_w, None),
        Linear::new_fp(a_w, None),
        conv1d,
        RmsNormGated::new(norm_w, cfg.rms_norm_eps),
        Linear::new_fp(out_w, None),
        a_log,
        dt_bias,
        cfg,
    );

    let x = load("input_x");
    let expected = load("expected_output");

    let out = gdn.forward(&x, None, None).expect("forward");

    assert_eq!(out.shape().as_slice(), expected.shape().as_slice());
    let err = max_abs_diff(&out, &expected);
    assert!(err < 1e-3, "GatedDeltaNet output max abs diff = {err} > 1e-3");
}
```

- [ ] **Step 7.4: Run the integration test**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b3_gated_delta_net
```

Expected: 1 test passes.

If it fails:
- Numerical mismatch >1e-3: most likely a bug in the kernel's recurrent state update or in the ops-level forward steps (RmsNormGated, GQA repeat, conv1d). Check the per-step intermediate values by printing.
- Compile error about visibility: `Linear::new_fp` and `Conv1d::new` and `GatedDeltaNet::from_components` all need to be visible to integration tests. Verify they're `pub` (with `#[doc(hidden)]`) like P3b2.

- [ ] **Step 7.5: Workspace regression check**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx
```

Expected: all earlier tests + new GatedDeltaNet tests + new integration test pass.

- [ ] **Step 7.6: Project gate**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

- [ ] **Step 7.7: Commit**

```
git add -A
git commit -m "test(p3b3): GatedDeltaNet Python fixture + numerical correctness"
```

---

## Verification Checklist

After Task 7:

| Item | Command | Expected |
|---|---|---|
| cxx-mlx conv smoke | `MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx-sys --test sys_smoke conv1d_links` | passes |
| cxx-mlx conv unit | `MLX_DIR=$HOME/.local/mlx cargo test --release -p mlx --lib ops::conv` | 2 tests pass |
| nn::Conv1d unit | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::conv` | 2 tests pass |
| nn::RmsNormGated unit | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::norm::tests::rms_norm_gated` | 2 tests pass |
| GatedDeltaCache unit | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::gated_delta` | 4 tests pass |
| Kernel + GatedDeltaNet unit | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net` | 5 tests pass |
| Integration test | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b3_gated_delta_net` | 1 test passes |
| Workspace regression | `MLX_DIR=$HOME/.local/mlx cargo test --release` | all tests pass (incl. P3b1+P3b2 fixtures) |
| Format | `cargo +nightly fmt --all -- --check` | no diff |
| Clippy | `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| Build | `cargo build --release` | success |

## Spec Coverage Map

| Spec section | Task |
|---|---|
| § 3.1 cxx-mlx conv1d binding | T1 |
| § 3.2 nn::Conv1d | T2 |
| § 3.3 nn::RmsNormGated | T3 |
| § 3.4 GatedDeltaCache | T4 |
| § 3.5 gated_delta_step kernel (2 variants) | T5 |
| § 3.6 GatedDeltaNet main module | T6 |
| § 4.1 unit tests (13 across components) | T1.12, T2.5, T3.4-3.5, T4.5, T5.5, T6.7 |
| § 4.2 integration test | T7.4 |
| § 4.3 fixture | T7.1-T7.3 |
| § 5 risks (state drift, conv state advance, simd_sum, mask zero, GQA repeat, T scalar) | T5.5 (kernel link + masked zero) + T6.7 (advance) + T7 (numerical) |

## Risk Register

- **Recurrent state numerical drift** → fp32 state buffer; 4-token integration test catches accumulation errors.
- **Conv state edge slicing** → `update_conv` slices last `kernel_size-1` tokens; integration test 4-token sequence catches off-by-one.
- **simd_sum reduction error** → `gated_delta_step_kernel_links` smoke + integration test verifies reduction correctness.
- **Mask=0 must leave state unchanged** → `gated_delta_step_masked_zero_path` covers this directly.
- **GQA repeat (Hv > Hk)** → shader uses `hk_idx = hv_idx / (Hv/Hk)`; Python ref uses `mx.repeat`; integration test Hv=4, Hk=2 validates.
- **T scalar input handling** → MLX docs say 0-dim arrays are passed as `T&` references; Step 5.1 verifies kernel compiles + dispatches with `(&[seq][..], ()).try_into()` 0-dim Array.

## Self-Review

After writing the complete plan:

1. **Spec coverage**: every § 3 / § 4 entry has a task. § 5 risks are addressed by tests in T5/T6/T7. § 6 acceptance is the verification checklist above.
2. **Placeholder scan**: no "TBD" / "TODO". Three "API verification at impl time" notes exist (`Array::greater`, `split_at_on` signature, etc.) — these are deliberate, marked as resolution points rather than placeholders. The plan tells the implementer where to look (`mlx/src/array.rs`, `mlx/src/ops/shape.rs:208`).
3. **Type consistency**: `GatedDeltaNetConfig` field names consistent (`num_v_heads`, `num_k_heads`, `head_k_dim`, `head_v_dim`, `conv_kernel_size`, `rms_norm_eps`, `hidden_size`). `from_components` parameter order matches across T6.1, T6.6 (forward), T7.3 (test). Cache `update_conv` / `update_recurrent` / `advance(n)` consistent. Kernel template names (`Dk`, `Dv`, `Hk`, `Hv`, `InT`, `StT`) consistent across T5 + T6.
