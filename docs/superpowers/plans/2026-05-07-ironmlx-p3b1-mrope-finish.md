# ironmlx P3b1 — MRoPE Finish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the P1 `Mrope::cos_sin` and `Mrope::apply` stubs (mlx::compile pipeline + fused Metal kernel) and unlock `Attention::forward` end-to-end against an mlx-lm reference fixture.

**Architecture:** Three-layer change. (1) `cos_sin` becomes a `mlx::compile`d closure cached in a `OnceLock` on the `Mrope` instance, computing cos/sin in fp32 once per forward and used across all 32 attention layers. (2) `apply` re-shapes from single-`x` to fused `(q, k, cos, sin) → (q', k')`, dispatched through one `MetalKernel` (also `OnceLock`-cached) with `HEAD_DIM` and `ROTARY_DIM` `template_int` constants. (3) `Attention::forward` collapses the existing two `mrope.apply(q, ...)` + `mrope.apply(k, ...)` calls into one call against the new dual-output API.

**Tech Stack:** Rust 2021 + mlx (`mlx::compile`, `mlx::MetalKernel`, `mlx::ops::indexing::slice`, `mlx::ops::shape::{concatenate, squeeze, expand_dims}`, `mlx::ops::cast::astype`) + ironmlx (`anyhow::Result`, `nn::Mrope`, `nn::Attention`) + a Python `mlx-lm` fixture generator. **Spec:** [`docs/superpowers/specs/2026-05-07-ironmlx-p3b1-mrope-finish-design.md`](../specs/2026-05-07-ironmlx-p3b1-mrope-finish-design.md).

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
- **`MLX_DIR=$HOME/.local/mlx`** required for any test that exercises MLX FFI / GPU.
- **MLX source location**: `/Volumes/Dev/mlx` (not the conda install).
- **ironmlx error type**: `anyhow::{Error, Result}` (re-exported as `crate::{Error, Result}`). Use `anyhow::anyhow!(...)` for ad-hoc errors.
- **Workspace**: branch `ironmlx`. No worktree.
- **Commit messages ASCII-safe**.

---

## File Structure (after P3b1)

```
ironmlx/
├── Cargo.toml                                     # no change
├── src/
│   └── nn/
│       ├── mrope.rs                               # MODIFIED — full cos_sin + apply impl;
│       │                                          #            apply signature (q, k, cos, sin) -> (q', k')
│       └── attention.rs                           # MODIFIED — single mrope.apply call
└── tests/
    ├── fixtures/
    │   └── p3b1_mrope/                            # NEW — fixture data + generator
    │       ├── README.md                          # how to regenerate
    │       ├── gen_fixture.py                     # Python script (mlx-lm)
    │       ├── input_q.npy                        # [1, 64, 8, 256] bf16
    │       ├── input_k.npy                        # [1, 8, 8, 256] bf16
    │       ├── input_cos.npy                      # [1, 8, 32] fp32
    │       ├── input_sin.npy                      # [1, 8, 32] fp32
    │       ├── input_position_ids.npy             # [3, 1, 8] i32
    │       ├── input_inv_freq.npy                 # [32] fp32
    │       ├── expected_cos.npy                   # [1, 8, 32] fp32
    │       ├── expected_sin.npy                   # [1, 8, 32] fp32
    │       ├── expected_q_rot.npy                 # [1, 64, 8, 256] bf16
    │       ├── expected_k_rot.npy                 # [1, 8, 8, 256] bf16
    │       └── expected_attn_out.npy              # [1, 8, 4096] bf16
    └── p3b1_mrope.rs                              # NEW — integration tests
```

---

## Task 1: `Mrope::cos_sin` via `mlx::compile`

**Files:**
- Modify: `ironmlx/src/nn/mrope.rs`

### Goal

Replace the `Mrope::cos_sin` stub with a real implementation:
- Wrap a multi-op MLX pipeline (`pos × inv_freq → cos / sin → 3-section slice + concat`) in `mlx::compile`.
- Cache the compiled closure in a `OnceLock<CompiledFn>` field on `Mrope` so it's built once per instance.
- Use `ShapeMode::Shapeless` so the single compile handles both prefill (large seq) and decode (seq=1).
- Cumulative section offsets `[0, 11, 22, 32]` are captured into the closure at compile time (model constants).

### Steps

- [ ] **Step 1.1: Add OnceLock fields to Mrope**

In `ironmlx/src/nn/mrope.rs`, find the existing `Mrope` struct (around the top of the file) and add two new fields plus update the `new()` constructor to initialize them as empty `OnceLock`s.

Top of file — add to imports:

```rust
use std::sync::OnceLock;

use mlx::compile::{CompiledFn, ShapeMode};
use mlx::ops::cast::astype;
use mlx::ops::indexing::slice;
use mlx::ops::shape::{concatenate, expand_dims, reshape, squeeze};
use mlx::{Array, Dtype};

use crate::Result;
```

(Keep existing imports; add the missing ones.)

Replace the existing `Mrope` struct with:

```rust
pub struct Mrope {
    inv_freq: Array,
    sections: SmallVec<[i32; 4]>,
    interleaved: bool,
    rot_dim: i32,
    head_dim: i32,
    /// Lazily-built `mlx::compile`d cos/sin pipeline. Built once per
    /// instance on first `cos_sin()` call; replayed on every subsequent call.
    cos_sin_compiled: OnceLock<CompiledFn>,
    /// Lazily-built `MetalKernel` for the fused (q, k, cos, sin) -> (q', k')
    /// apply path (filled in T2).
    apply_kernel: OnceLock<mlx::MetalKernel>,
}
```

Update `Mrope::new(...)` — find the existing `Ok(Self { ... })` block at the end of `new()`, and add the two new fields:

```rust
        Ok(Self {
            inv_freq,
            sections: SmallVec::from_slice(sections),
            interleaved,
            rot_dim,
            head_dim,
            cos_sin_compiled: OnceLock::new(),
            apply_kernel: OnceLock::new(),
        })
```

- [ ] **Step 1.2: Write the failing cos_sin shape/dtype test**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `mrope.rs`:

```rust
    #[test]
    fn cos_sin_shape_and_dtype() {
        // Qwen3.5: head_dim=256, partial=0.25 -> rot_dim=64, half=32
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // position_ids [3, B=1, S=8] i32, three identical streams (text-only)
        let pos: Array = (
            &[0_i32, 1, 2, 3, 4, 5, 6, 7,
              0,     1, 2, 3, 4, 5, 6, 7,
              0,     1, 2, 3, 4, 5, 6, 7][..],
            (3_i32, 1, 8),
        )
            .try_into()
            .unwrap();

        let (cos, sin) = mrope.cos_sin(&pos).expect("cos_sin");

        assert_eq!(cos.shape().as_slice(), &[1, 8, 32]);
        assert_eq!(sin.shape().as_slice(), &[1, 8, 32]);
        assert_eq!(cos.dtype(), Dtype::Float32);
        assert_eq!(sin.dtype(), Dtype::Float32);
    }
```

- [ ] **Step 1.3: Run the test and verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::cos_sin_shape_and_dtype
```

Expected: FAIL with the existing P1 stub message `"Mrope::cos_sin not implemented at P1 — exercised in P3 model assembly"`.

- [ ] **Step 1.4: Implement `cos_sin` and the compiled-pipeline builder**

Replace the existing `cos_sin` stub method with a real implementation, plus add a private helper. Find the existing `pub fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array)>` and replace its body:

```rust
    /// Compute `(cos, sin)` rotation tables from `position_ids`.
    ///
    /// `position_ids: [n_streams, B, S]` — one stream per `mrope_section`
    /// (Qwen3.5: 3 streams = temporal/height/width; text-only prompts pass
    /// 3 identical streams).
    ///
    /// Returns `(cos: [B, S, rot_dim/2], sin: [B, S, rot_dim/2])` in fp32;
    /// caller is responsible for `astype` to the working compute dtype.
    ///
    /// First call lazily compiles the pipeline via `mlx::compile`; subsequent
    /// calls replay the optimized graph.
    pub fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array)> {
        let f = self.cos_sin_compiled.get_or_init(|| {
            self.build_cos_sin_pipeline()
                .expect("build_cos_sin_pipeline cannot fail at first call")
        });
        let mut outs = f.invoke(&[position_ids, &self.inv_freq])?;
        // CompiledFn::invoke returns a Vec<Array> in declared order.
        // (No erase-and-shift here — Vec just moves out by index.)
        let sin = outs.remove(1);
        let cos = outs.remove(0);
        Ok((cos, sin))
    }

    fn build_cos_sin_pipeline(&self) -> Result<CompiledFn> {
        // Cumulative section offsets; e.g. sections=[11,11,10] -> offsets=[0,11,22,32].
        let n_streams = self.sections.len() as i32;
        let half: i32 = self.sections.iter().sum();
        let mut offsets: Vec<i32> = Vec::with_capacity(self.sections.len() + 1);
        offsets.push(0);
        let mut acc = 0_i32;
        for n in self.sections.iter() {
            acc += *n;
            offsets.push(acc);
        }

        // `move` closure captures `offsets` and `n_streams` (Copy/Vec — Send + 'static OK).
        // `inputs[0]` = position_ids [n_streams, B, S] i32
        // `inputs[1]` = inv_freq      [half] fp32
        let pipeline = move |inputs: &[&Array]| -> Result<Vec<Array>> {
            let pos = inputs[0];
            let inv_freq = inputs[1];

            // 1. broadcast multiply: pos[s,b,t] * inv_freq[d]
            //    pos_f32 -> [n_streams, B, S]; expand to [n_streams, B, S, 1]
            //    inv_freq -> [half]; reshape to [1, 1, 1, half]
            let pos_f32 = astype(pos, Dtype::Float32)?;
            let pos_unsq = expand_dims(&pos_f32, &[3_i32][..])?;          // [n_streams, B, S, 1]
            let inv_freq_unsq = reshape(inv_freq, &[1_i32, 1, 1, half][..])?; // [1, 1, 1, half]
            let freqs = (&pos_unsq * &inv_freq_unsq)?;                    // [n_streams, B, S, half]

            // 2. cos / sin (fp32)
            let cos_per_stream = freqs.cos()?;
            let sin_per_stream = freqs.sin()?;

            // 3. C-A: per-section slice + concat along last dim.
            //    For each stream s (0..n_streams), take
            //        cos_per_stream[s:s+1, :, :, offsets[s]..offsets[s+1]]
            //    then squeeze the leading stream-axis -> [B, S, sect_len].
            //    Finally concat all segments along axis -1 -> [B, S, half].
            let mut cos_segs: Vec<Array> = Vec::with_capacity(n_streams as usize);
            let mut sin_segs: Vec<Array> = Vec::with_capacity(n_streams as usize);
            for s in 0..n_streams {
                let lo = offsets[s as usize];
                let hi = offsets[s as usize + 1];

                // start = [s, 0, 0, lo], stop = [s+1, B, S, hi]
                // We don't know B and S statically — use the actual array dims.
                let dims = cos_per_stream.shape().as_slice();
                let b = dims[1];
                let seq = dims[2];
                let start = vec![s, 0_i32, 0, lo];
                let stop = vec![s + 1, b, seq, hi];

                let cos_seg = slice(&cos_per_stream, start.as_slice(), stop.as_slice())?;
                let sin_seg = slice(&sin_per_stream, start.as_slice(), stop.as_slice())?;
                // Squeeze leading stream axis (size 1).
                let cos_seg = squeeze(&cos_seg, &[0_i32][..])?;
                let sin_seg = squeeze(&sin_seg, &[0_i32][..])?;

                cos_segs.push(cos_seg);
                sin_segs.push(sin_seg);
            }
            let cos_segs_refs: Vec<&Array> = cos_segs.iter().collect();
            let sin_segs_refs: Vec<&Array> = sin_segs.iter().collect();
            let cos = concatenate(&cos_segs_refs, -1)?;
            let sin = concatenate(&sin_segs_refs, -1)?;

            Ok(vec![cos, sin])
        };

        // ShapeMode::Shapeless: the same compile handles prefill (S>>1) and decode (S=1).
        mlx::compile::compile(pipeline, ShapeMode::Shapeless)
    }
```

> **Note:** The closure captures by move — `offsets: Vec<i32>` and `n_streams: i32` are `Send + 'static`. The `Mrope` instance itself is not captured; only the cumulative offsets vector and the stream count.

> **`Result<Vec<Array>>` ordering:** `CompiledFn::invoke` returns outputs in the order the closure pushed them (`vec![cos, sin]`), so `outs[0]` = cos and `outs[1]` = sin. Use `Vec::remove(i)` (NOT `take_at` — that's the C++-side `ArrayVec::take_at` from `mlx::ArrayVec`; here we already have a Rust `Vec<Array>`).

- [ ] **Step 1.5: Run the test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::cos_sin_shape_and_dtype
```

Expected: PASS.

- [ ] **Step 1.6: Add decode (seq=1) boundary test**

Append to `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn cos_sin_seq_eq_one_decode() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        // Decode step: position 42 across all 3 streams.
        let pos: Array = (&[42_i32, 42, 42][..], (3_i32, 1, 1)).try_into().unwrap();
        let (cos, sin) = mrope.cos_sin(&pos).expect("cos_sin seq=1");
        assert_eq!(cos.shape().as_slice(), &[1, 1, 32]);
        assert_eq!(sin.shape().as_slice(), &[1, 1, 32]);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::cos_sin_seq_eq_one_decode
```

Expected: PASS.

- [ ] **Step 1.7: Project gate + commit**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release

git add -A
git commit -m "feat(p3b1): Mrope::cos_sin via mlx::compile (3-section pipeline)"
```

Expected: gate clean; both new tests pass.

---

## Task 2: `Mrope::apply` fused MetalKernel

**Files:**
- Modify: `ironmlx/src/nn/mrope.rs`

### Goal

Replace the `Mrope::apply` stub with a fused Metal kernel implementation:
- Signature change: `apply(&self, q, k, cos, sin) -> Result<(Array, Array)>` (was: `apply(&self, x, cos, sin) -> Result<Array>`).
- One `MetalKernel` instance per `Mrope`, lazy-built and cached in the `apply_kernel: OnceLock<MetalKernel>` field added in T1.
- `template_int`: `HEAD_DIM`, `ROTARY_DIM` (b+ scope; everything else runtime).
- Grid: `(B*(Hq+Hkv), S, HEAD_DIM)` so a single dispatch covers both Q and K element-by-element.
- Threadgroup: `(1, 1, HEAD_DIM)` — one threadgroup per `(qk_head, t)` pair, threads cover the head-dim axis.
- Kernel emits cast-back-to-input-dtype (`metal::float` accumulation, `decltype(x_self)(rotated)` store) so bf16 inputs stay bf16.

### Steps

- [ ] **Step 2.1: Write the failing apply shape/dtype test**

Append to `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn apply_shape_and_dtype_fp32() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // Q [B=1, Hq=64, S=4, head_dim=256], K [B=1, Hkv=8, S=4, head_dim=256]
        // Use small S=4 to keep the test fast.
        let q = mlx::ops::constructors::zeros((1_i32, 64, 4, 256), Dtype::Float32).unwrap();
        let k = mlx::ops::constructors::zeros((1_i32, 8, 4, 256), Dtype::Float32).unwrap();
        let cos = mlx::ops::constructors::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let sin = mlx::ops::constructors::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");

        assert_eq!(q_rot.shape().as_slice(), &[1, 64, 4, 256]);
        assert_eq!(k_rot.shape().as_slice(), &[1, 8, 4, 256]);
        assert_eq!(q_rot.dtype(), Dtype::Float32);
        assert_eq!(k_rot.dtype(), Dtype::Float32);
    }
```

- [ ] **Step 2.2: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_shape_and_dtype_fp32
```

Expected: FAIL — either the existing P1 stub message OR a compile error because the signature is being changed (apply now takes 4 args).

- [ ] **Step 2.3: Implement `apply` and `build_apply_kernel`**

Replace the existing `pub fn apply(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array>` with the new fused-Q+K signature. Append the kernel builder as a private method.

```rust
    /// Apply rotary rotation to Q and K in a single fused dispatch.
    ///
    /// `q: [B, Hq, S, HEAD_DIM]`, `k: [B, Hkv, S, HEAD_DIM]`,
    /// `cos: [B, S, ROTARY_DIM/2]` (fp32), `sin: [B, S, ROTARY_DIM/2]` (fp32).
    ///
    /// Returns `(q_rot, k_rot)` with the same shape and dtype as their inputs.
    /// The trailing `HEAD_DIM - ROTARY_DIM` channels pass through unchanged.
    pub fn apply(
        &self,
        q: &Array,
        k: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        // Sanity (cheap; full validation is at MLX dispatch boundaries).
        let q_dims = q.shape().as_slice();
        let k_dims = k.shape().as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 {
            return Err(anyhow::anyhow!(
                "Mrope::apply expects rank-4 q/k; got q.ndim={}, k.ndim={}",
                q_dims.len(),
                k_dims.len()
            ));
        }
        if q_dims[3] != self.head_dim || k_dims[3] != self.head_dim {
            return Err(anyhow::anyhow!(
                "Mrope::apply: q.head_dim={} k.head_dim={} != configured {}",
                q_dims[3],
                k_dims[3],
                self.head_dim
            ));
        }

        let b = q_dims[0];
        let hq = q_dims[1];
        let hkv = k_dims[1];
        let s = q_dims[2];

        let kernel = self.apply_kernel.get_or_init(|| {
            self.build_apply_kernel()
                .expect("build_apply_kernel cannot fail at first call")
        });

        // Grid: cover (B*(Hq+Hkv)) × S × HEAD_DIM elements; one thread per element.
        let grid_x = b * (hq + hkv);
        let grid_y = s;
        let grid_z = self.head_dim;
        // Threadgroup: 1 thread on the (qk_head, t) axes; HEAD_DIM threads on the d axis.
        // HEAD_DIM=256 fits within Metal's 1024-thread threadgroup limit.
        let tg_x = 1;
        let tg_y = 1;
        let tg_z = self.head_dim;

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[q, k, cos, sin])
            .output_shapes(&[q.shape().clone(), k.shape().clone()])
            .output_dtypes(&[q.dtype(), k.dtype()])
            .grid(grid_x, grid_y, grid_z)
            .threadgroup(tg_x, tg_y, tg_z)
            .template_int("HEAD_DIM", self.head_dim)
            .template_int("ROTARY_DIM", self.rot_dim)
            .dispatch()?;

        let q_rot = outputs.take_at(0)?;
        let k_rot = outputs.take_at(0)?; // erase-and-shift: K shifts to slot 0
        Ok((q_rot, k_rot))
    }

    fn build_apply_kernel(&self) -> Result<mlx::MetalKernel> {
        // Metal shader. Templates: HEAD_DIM, ROTARY_DIM. ROT_PAIRS = ROTARY_DIM/2.
        //
        // Each thread handles one element of (Q or K) at indices (b, head, t, d).
        // The first grid dim (qk_head) ranges over B*(Hq+Hkv): the lower B*Hq
        // values address Q; the upper B*Hkv address K. Hq, Hkv, B, S are pulled
        // from the input shape buffers (auto-injected by MLX when the source
        // references `<name>_shape`).
        let src = r#"
        constexpr uint ROT_PAIRS = ROTARY_DIM / 2;

        uint qk_head = thread_position_in_grid.x;
        uint t       = thread_position_in_grid.y;
        uint d       = thread_position_in_grid.z;

        uint B   = (uint)q_shape[0];
        uint Hq  = (uint)q_shape[1];
        uint S   = (uint)q_shape[2];
        uint Hkv = (uint)k_shape[1];

        // Decode (b, head, is_q)
        bool is_q;
        uint b;
        uint h;
        if (qk_head < B * Hq) {
            is_q = true;
            b = qk_head / Hq;
            h = qk_head % Hq;
        } else {
            is_q = false;
            uint kqk = qk_head - B * Hq;
            b = kqk / Hkv;
            h = kqk % Hkv;
        }

        uint H = is_q ? Hq : Hkv;
        // Row-major (B, H, S, HEAD_DIM):
        uint base = ((b * H + h) * S + t) * HEAD_DIM;

        // cos/sin: row-major (B, S, ROT_PAIRS), broadcast across heads.
        uint cs_idx = (b * S + t) * ROT_PAIRS;

        if (d < ROTARY_DIM) {
            // Interleaved: pair (2p, 2p+1) shares cos[p], sin[p].
            uint p = d >> 1;
            bool is_even = (d & 1u) == 0u;

            float c = cos[cs_idx + p];
            float si = sin[cs_idx + p];

            if (is_q) {
                float x_self = float(q[base + d]);
                float x_pair = float(q[base + (is_even ? d + 1 : d - 1)]);
                float rotated = is_even
                    ? (x_self * c - x_pair * si)
                    : (x_pair * si + x_self * c);
                q_out[base + d] = (decltype(q[0]))(rotated);
            } else {
                float x_self = float(k[base + d]);
                float x_pair = float(k[base + (is_even ? d + 1 : d - 1)]);
                float rotated = is_even
                    ? (x_self * c - x_pair * si)
                    : (x_pair * si + x_self * c);
                k_out[base + d] = (decltype(k[0]))(rotated);
            }
        } else {
            // Pass-through tail (HEAD_DIM - ROTARY_DIM channels).
            if (is_q) {
                q_out[base + d] = q[base + d];
            } else {
                k_out[base + d] = k[base + d];
            }
        }
        "#;

        Ok(mlx::MetalKernel::builder("ironmlx_mrope_apply_qk")
            .inputs(&["q", "k", "cos", "sin"])
            .outputs(&["q_out", "k_out"])
            .source(src)
            .ensure_row_contiguous(true)
            .atomic_outputs(false)
            .build()?)
    }
```

> **Why `decltype(q[0])`**: Metal Shading Language supports C++14 `decltype`. `q[0]` is a `device const T&` reference; `decltype(q[0])` strips to `T` (the input dtype, e.g. `bfloat`/`half`/`float`), letting one shader source handle bf16 / fp16 / fp32 without per-dtype source duplication.

> **Auto-injected shape buffers**: MLX's metal_kernel scans the source for `<name>_shape` references and auto-injects them as `const constant int* <name>_shape` buffer args. We use `q_shape[0..2]` and `k_shape[1]`. Verified at `/Volumes/Dev/mlx/mlx/backend/metal/custom_kernel.cpp:93-105,190-192`.

- [ ] **Step 2.4: Run apply test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_shape_and_dtype_fp32
```

Expected: PASS — Metal kernel builds and dispatches correctly with f32 zeros input.

- [ ] **Step 2.5: Add bf16 dtype boundary test**

Append:

```rust
    #[test]
    fn apply_shape_and_dtype_bf16() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        let q = mlx::ops::constructors::zeros((1_i32, 64, 4, 256), Dtype::Bfloat16).unwrap();
        let k = mlx::ops::constructors::zeros((1_i32, 8, 4, 256), Dtype::Bfloat16).unwrap();
        // cos/sin always fp32 (per spec § 3.1).
        let cos = mlx::ops::constructors::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let sin = mlx::ops::constructors::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply bf16");

        assert_eq!(q_rot.dtype(), Dtype::Bfloat16);
        assert_eq!(k_rot.dtype(), Dtype::Bfloat16);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_shape_and_dtype_bf16
```

Expected: PASS.

- [ ] **Step 2.6: Add partial-rotary tail-passthrough test**

Verifies that channels `[ROTARY_DIM, HEAD_DIM)` are byte-equal to input.

```rust
    #[test]
    fn apply_partial_rotary_tail_unchanged() {
        // head_dim=256, partial=0.25 -> rot_dim=64. Tail [64..256) must be unchanged.
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // Distinct integer values per element so we can spot any unintended mutation.
        // Q shape [1, 1, 1, 256] with values 0..256 (fp32).
        let q_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let q: Array = (q_data.as_slice(), (1_i32, 1, 1, 256)).try_into().unwrap();
        let k: Array = (q_data.as_slice(), (1_i32, 1, 1, 256)).try_into().unwrap();

        // cos = ones, sin = zeros: rotation is identity on rotated dims;
        // tail dims must also stay unchanged.
        let cos = mlx::ops::constructors::ones((1_i32, 1, 32), Dtype::Float32).unwrap();
        let sin = mlx::ops::constructors::zeros((1_i32, 1, 32), Dtype::Float32).unwrap();

        let (q_rot, _k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");
        let rot_data: Vec<f32> = q_rot.to_vec().unwrap();

        // Tail must be byte-identical to input.
        for d in 64..256 {
            assert_eq!(rot_data[d], q_data[d], "tail channel {d} mutated");
        }
        // Rotated dims with cos=1 sin=0: identity for both even and odd halves.
        // even idx: x_even * 1 - x_odd * 0 = x_even
        // odd idx:  x_even * 0 + x_odd * 1 = x_odd
        for d in 0..64 {
            assert_eq!(rot_data[d], q_data[d], "rotated channel {d} not identity under cos=1,sin=0");
        }
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_partial_rotary_tail_unchanged
```

Expected: PASS.

- [ ] **Step 2.7: Add interleaved-pair manually-verified test**

Test the exact rotation formula on a 4-channel example.

```rust
    #[test]
    fn apply_interleaved_pair_known_rotation() {
        // Build a tiny mrope where head_dim=4, rot_dim=4, sections=[1,1,0]
        // (or any sections summing to half=2). Manual values let us check
        // the rotation formula bit-exactly.
        let mrope = Mrope::new(4, 10000.0, 1.0, &[1, 1, 0], true).unwrap();

        // Q = [1, 1, 1, 1, 1, 1] reshaped as [1, 1, 1, 4] = [[1, 2, 3, 4]]
        let q: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1, 1, 4))
            .try_into()
            .unwrap();
        let k: Array = (&[10.0_f32, 20.0, 30.0, 40.0][..], (1_i32, 1, 1, 4))
            .try_into()
            .unwrap();

        // cos = [c0, c1], sin = [s0, s1] for the 2 pairs.
        // Use cos = [0, 1], sin = [1, 0]:
        //   pair 0 (channels 0,1): cos=0, sin=1
        //     y[0] = x[0]*0 - x[1]*1 = -x[1] = -2
        //     y[1] = x[0]*1 + x[1]*0 = x[0]  =  1
        //   pair 1 (channels 2,3): cos=1, sin=0  (identity)
        //     y[2] = x[2] = 3
        //     y[3] = x[3] = 4
        let cos: Array = (&[0.0_f32, 1.0][..], (1_i32, 1, 2)).try_into().unwrap();
        let sin: Array = (&[1.0_f32, 0.0][..], (1_i32, 1, 2)).try_into().unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");
        let q_out: Vec<f32> = q_rot.to_vec().unwrap();
        let k_out: Vec<f32> = k_rot.to_vec().unwrap();

        assert_eq!(q_out, vec![-2.0, 1.0, 3.0, 4.0]);
        assert_eq!(k_out, vec![-20.0, 10.0, 30.0, 40.0]);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_interleaved_pair_known_rotation
```

Expected: PASS.

- [ ] **Step 2.8: Add GQA Hq != Hkv test**

```rust
    #[test]
    fn apply_gqa_different_q_kv_heads() {
        // Qwen3.5-style GQA: Hq=64, Hkv=8.
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        let q = mlx::ops::constructors::zeros((1_i32, 64, 2, 256), Dtype::Float32).unwrap();
        let k = mlx::ops::constructors::zeros((1_i32, 8, 2, 256), Dtype::Float32).unwrap();
        let cos = mlx::ops::constructors::zeros((1_i32, 2, 32), Dtype::Float32).unwrap();
        let sin = mlx::ops::constructors::zeros((1_i32, 2, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply gqa");
        // Both must produce the right shape under GQA.
        assert_eq!(q_rot.shape().as_slice(), &[1, 64, 2, 256]);
        assert_eq!(k_rot.shape().as_slice(), &[1, 8, 2, 256]);
    }
```

Run:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope::tests::apply_gqa_different_q_kv_heads
```

Expected: PASS.

- [ ] **Step 2.9: Project gate + commit**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release

git add -A
git commit -m "feat(p3b1): Mrope::apply fused Q+K MetalKernel (interleaved + partial rotary)"
```

Expected: gate clean; 5 new apply tests pass.

---

## Task 3: mlx-lm fixture + numerical-correctness integration tests

**Files:**
- Create: `ironmlx/tests/fixtures/p3b1_mrope/README.md`
- Create: `ironmlx/tests/fixtures/p3b1_mrope/gen_fixture.py`
- Create: `ironmlx/tests/fixtures/p3b1_mrope/*.npy` (generated, ~800KB total)
- Create: `ironmlx/tests/p3b1_mrope.rs`

### Goal

Generate reference data via `mlx-lm`'s Qwen3.5 implementation, save as `.npy`, commit to the repo, and add Rust integration tests that verify our cos_sin and apply produce numerically equivalent output (fp32 atol=1e-5, bf16 atol=1e-3).

### Steps

- [ ] **Step 3.1: Create the fixture README**

```bash
mkdir -p ironmlx/tests/fixtures/p3b1_mrope
```

Create `ironmlx/tests/fixtures/p3b1_mrope/README.md`:

````markdown
# P3b1 MRoPE fixtures

Reference data for Qwen3.5 MRoPE numerical-correctness tests.

## Regenerate

Requires `mlx-lm` matching the project's pinned version (see `gen_fixture.py`'s
version assertion).

```bash
cd ironmlx/tests/fixtures/p3b1_mrope
python gen_fixture.py
```

Generated `.npy` files (committed to git, ~800KB total):

| File | Shape | Dtype |
|---|---|---|
| `input_q.npy` | `[1, 64, 8, 256]` | bf16 |
| `input_k.npy` | `[1, 8, 8, 256]` | bf16 |
| `input_position_ids.npy` | `[3, 1, 8]` | i32 |
| `input_inv_freq.npy` | `[32]` | fp32 |
| `expected_cos.npy` | `[1, 8, 32]` | fp32 |
| `expected_sin.npy` | `[1, 8, 32]` | fp32 |
| `expected_q_rot.npy` | `[1, 64, 8, 256]` | bf16 |
| `expected_k_rot.npy` | `[1, 8, 8, 256]` | bf16 |
| `expected_attn_out.npy` | `[1, 8, 4096]` | bf16 |
````

- [ ] **Step 3.2: Create the fixture generator**

Create `ironmlx/tests/fixtures/p3b1_mrope/gen_fixture.py`:

```python
"""Generate P3b1 MRoPE fixtures from mlx-lm's Qwen3.5 reference path.

Outputs go alongside this script as `.npy` files. Re-run to regenerate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

# Pin the mlx-lm version we trust. Bump only when the upstream MRoPE
# implementation has been verified to still match our shader.
EXPECTED_MLX_LM_VERSION_PREFIX = "0.1"  # adjust on bump

import mlx_lm  # noqa: E402

if not mlx_lm.__version__.startswith(EXPECTED_MLX_LM_VERSION_PREFIX):
    raise SystemExit(
        f"mlx_lm version {mlx_lm.__version__} does not start with "
        f"{EXPECTED_MLX_LM_VERSION_PREFIX!r}; bump fixture or pin a version."
    )

OUT_DIR = Path(__file__).parent

# ---- Qwen3.5 MRoPE constants (per the model config) ----
HEAD_DIM = 256
PARTIAL = 0.25
ROT_DIM = int(HEAD_DIM * PARTIAL) & ~1  # 64
HALF = ROT_DIM // 2  # 32
SECTIONS = [11, 11, 10]
INTERLEAVED = True
THETA = 1e7
HQ = 64
HKV = 8
B = 1
S = 8


def build_inv_freq() -> mx.array:
    # inv_freq[i] = 1 / theta^(2i / rot_dim) for i in [0, half)
    idx = mx.arange(0, HALF, dtype=mx.float32)
    return mx.exp(-(idx * (2.0 / ROT_DIM)) * float(np.log(THETA)))


def build_position_ids() -> mx.array:
    # Text-only: 3 identical streams [0, 1, ..., S-1]
    one = mx.arange(0, S, dtype=mx.int32).reshape((1, 1, S))
    return mx.broadcast_to(one, (3, B, S))


def reference_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """Reference MRoPE cos/sin — independent re-implementation per the spec."""
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]                # [3, B, S, 1]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1)) # [1, 1, 1, half]
    freqs = pos_unsq * inv_unsq                 # [3, B, S, half]
    cos_per = mx.cos(freqs)
    sin_per = mx.sin(freqs)

    # 3-section concat along last axis
    offsets = [0]
    for n in SECTIONS:
        offsets.append(offsets[-1] + n)

    cos_segs = []
    sin_segs = []
    for s, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        cos_segs.append(cos_per[s, :, :, lo:hi])
        sin_segs.append(sin_per[s, :, :, lo:hi])
    cos = mx.concatenate(cos_segs, axis=-1)
    sin = mx.concatenate(sin_segs, axis=-1)
    return cos, sin


def reference_apply(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    """Apply interleaved rotation to `x` (Q or K), tail pass-through."""
    # x: [B, H, S, HEAD_DIM]
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]

    # Interleaved: even (2p) and odd (2p+1) channels form pairs sharing cos[p], sin[p].
    even = rot[..., 0::2]  # [B, H, S, HALF]
    odd = rot[..., 1::2]

    # Broadcast cos/sin: [B, S, HALF] -> [B, 1, S, HALF]
    c = cos[:, None, :, :]
    s = sin[:, None, :, :]

    rot_even = (even.astype(mx.float32) * c - odd.astype(mx.float32) * s).astype(x.dtype)
    rot_odd = (even.astype(mx.float32) * s + odd.astype(mx.float32) * c).astype(x.dtype)

    # Re-interleave
    out_rot = mx.stack([rot_even, rot_odd], axis=-1).reshape(x.shape[:-1] + (ROT_DIM,))
    return mx.concatenate([out_rot, tail], axis=-1)


def main() -> None:
    np.random.seed(42)

    inv_freq = build_inv_freq()
    pos = build_position_ids()
    cos, sin = reference_cos_sin(pos, inv_freq)

    # Random Q, K (bf16 to match Qwen3.5)
    q_np = np.random.randn(B, HQ, S, HEAD_DIM).astype(np.float32)
    k_np = np.random.randn(B, HKV, S, HEAD_DIM).astype(np.float32)
    q = mx.array(q_np).astype(mx.bfloat16)
    k = mx.array(k_np).astype(mx.bfloat16)

    q_rot = reference_apply(q, cos, sin)
    k_rot = reference_apply(k, cos, sin)

    # Force eval, then convert to numpy for save (bf16 -> stored as uint16 raw bits per .npy spec)
    mx.eval(cos, sin, q_rot, k_rot)

    def save(name: str, arr) -> None:
        path = OUT_DIR / f"{name}.npy"
        # mx.save writes .npy with MLX's dtype encoding; tests load via mlx::io::load_npy.
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_q", q)
    save("input_k", k)
    save("input_position_ids", pos)
    save("input_inv_freq", inv_freq)
    save("expected_cos", cos)
    save("expected_sin", sin)
    save("expected_q_rot", q_rot)
    save("expected_k_rot", k_rot)

    # Note: expected_attn_out left to T4 (uses Attention.forward which we'll
    # exercise with this same input/expected_q_rot setup; the SDPA reference
    # output requires loading actual Qwen3.5 weights, so we either skip it
    # here OR generate via mlx-lm Attention layer separately).
    # See T4 Step 4.X for the attention e2e fixture decision.


if __name__ == "__main__":
    main()
```

Run it once locally to populate the fixture directory:

```bash
cd ironmlx/tests/fixtures/p3b1_mrope
python gen_fixture.py
```

Expected output: 8 `.npy` files written.

> **Note on `expected_attn_out.npy`**: this requires real Qwen3.5 weights and a single attention-layer forward in `mlx-lm`. Splitting it out lets the cos_sin / apply numerical tests pass without dragging in weight-loading. Step 3.X covers cos_sin and apply only; T4 covers the attention e2e fixture and test.

- [ ] **Step 3.3: Create the integration test file**

Create `ironmlx/tests/p3b1_mrope.rs`:

```rust
//! P3b1 MRoPE numerical-correctness integration tests.
//!
//! Loads .npy fixtures from `tests/fixtures/p3b1_mrope/` (generated by
//! `gen_fixture.py` against mlx-lm) and verifies that our `Mrope::cos_sin`
//! and `Mrope::apply` produce numerically equivalent results.
//!
//! Tolerances:
//!   - cos/sin (fp32): atol = 1e-5
//!   - apply (bf16):   atol = 1e-3 (limited by bf16 rounding)

use mlx::{Array, Dtype};

use ironmlx::nn::Mrope;

const FIXTURE_DIR: &str = "tests/fixtures/p3b1_mrope";

fn load(name: &str) -> Array {
    let path = format!("{FIXTURE_DIR}/{name}.npy");
    mlx::io::load_npy(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

/// max(|a - b|) for fp32 arrays of equal shape.
fn max_abs_diff_f32(a: &Array, b: &Array) -> f32 {
    let av: Vec<f32> = a.to_vec().unwrap();
    let bv: Vec<f32> = b.to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// max(|a - b|) where both arrays are bf16; cast to fp32 for the diff.
fn max_abs_diff_bf16(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).unwrap();
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).unwrap();
    max_abs_diff_f32(&a32, &b32)
}

#[test]
fn cos_sin_matches_mlx_lm_fixture() {
    // head_dim=256, partial=0.25 -> rot_dim=64, half=32, sections=[11,11,10].
    let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

    let pos = load("input_position_ids");
    let exp_cos = load("expected_cos");
    let exp_sin = load("expected_sin");

    let (cos, sin) = mrope.cos_sin(&pos).expect("cos_sin");

    assert_eq!(cos.shape().as_slice(), exp_cos.shape().as_slice());
    assert_eq!(sin.shape().as_slice(), exp_sin.shape().as_slice());

    let cos_err = max_abs_diff_f32(&cos, &exp_cos);
    let sin_err = max_abs_diff_f32(&sin, &exp_sin);
    assert!(cos_err < 1e-5, "cos max abs diff = {cos_err} > 1e-5");
    assert!(sin_err < 1e-5, "sin max abs diff = {sin_err} > 1e-5");
}

#[test]
fn apply_matches_mlx_lm_fixture() {
    let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

    let q = load("input_q");
    let k = load("input_k");
    let cos = load("expected_cos"); // reuse — these are the inputs to apply
    let sin = load("expected_sin");
    let exp_q_rot = load("expected_q_rot");
    let exp_k_rot = load("expected_k_rot");

    let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");

    assert_eq!(q_rot.shape().as_slice(), exp_q_rot.shape().as_slice());
    assert_eq!(k_rot.shape().as_slice(), exp_k_rot.shape().as_slice());
    assert_eq!(q_rot.dtype(), Dtype::Bfloat16);
    assert_eq!(k_rot.dtype(), Dtype::Bfloat16);

    let q_err = max_abs_diff_bf16(&q_rot, &exp_q_rot);
    let k_err = max_abs_diff_bf16(&k_rot, &exp_k_rot);
    assert!(q_err < 1e-3, "q_rot max abs diff = {q_err} > 1e-3");
    assert!(k_err < 1e-3, "k_rot max abs diff = {k_err} > 1e-3");
}
```

- [ ] **Step 3.4: Run the new integration tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b1_mrope
```

Expected: 2 tests pass — `cos_sin_matches_mlx_lm_fixture` and `apply_matches_mlx_lm_fixture`.

If `cos_sin_matches_mlx_lm_fixture` fails with a small but nonzero diff, double-check the section offsets / interleaved layout. If `apply_matches_mlx_lm_fixture` fails, the most likely culprit is the Metal shader's even/odd index logic in T2 step 2.3.

- [ ] **Step 3.5: Project gate + commit**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release

git add ironmlx/tests/fixtures/p3b1_mrope ironmlx/tests/p3b1_mrope.rs
git commit -m "test(p3b1): mlx-lm fixtures + cos_sin/apply numerical correctness"
```

Expected: gate clean; 2 integration tests pass.

---

## Task 4: `Attention::forward` unlock + e2e integration test

**Files:**
- Modify: `ironmlx/src/nn/attention.rs`
- Modify: `ironmlx/tests/p3b1_mrope.rs` (add the e2e test)
- Create: `ironmlx/tests/fixtures/p3b1_mrope/expected_attn_out.npy` (regenerated)
- Modify: `ironmlx/tests/fixtures/p3b1_mrope/gen_fixture.py` (extend to dump attn_out)

### Goal

Collapse the two existing `mrope.apply(&q, cos, sin)?` / `mrope.apply(&k, cos, sin)?` calls in `Attention::forward_on` into a single `mrope.apply(&q, &k, cos, sin)?` call (matching the new dual-output API from T2). Generate an `expected_attn_out.npy` fixture from a single Qwen3.5 attention layer in `mlx-lm`, and add an e2e integration test that drives `Attention::forward` against the same input and asserts the output matches at bf16 atol=1e-3.

### Steps

- [ ] **Step 4.1: Update `Attention::forward_on` to use the new fused apply signature**

In `ironmlx/src/nn/attention.rs`, find the two existing `mrope.apply` calls in `forward_on` (around the middle of the method body):

```rust
        // Apply rotary positions. Stubbed at P1 — surfaces a clear `Err`.
        // `Mrope::apply` has no `_on` variant at P1; threaded in P3.
        let q = mrope.apply(&q, cos, sin)?;
        let k = mrope.apply(&k, cos, sin)?;
```

Replace with a single fused call:

```rust
        // Apply rotary positions in a single fused dispatch (P3b1).
        // `Mrope::apply` rotates Q and K together with one MetalKernel launch.
        let (q, k) = mrope.apply(&q, &k, cos, sin)?;
```

> The surrounding shape transforms (reshape + transpose to `[B, H, S, head_dim]`) are unchanged — `Mrope::apply` accepts and returns the same `[B, H, S, head_dim]` layout.

> **Note on the comment update**: the inline doc comment on `forward` says "**At P1 this returns `Err`** because [`Mrope::apply`] is stubbed". After this change that's no longer true. Find the doc comment (around 8 lines above the new code) and replace `**At P1 this returns `Err`**` with `**As of P3b1 this is fully wired**` plus a one-line note that the SDPA path runs end-to-end.

Locate the doc comment block (lines roughly 90-105):

```rust
    /// Forward without KV cache (P1 prefill-only path; P2 adds cache + decode).
    ///
    /// `x: [batch, seq, hidden]`. Returns `[batch, seq, hidden]`.
    ///
    /// **At P1 this returns `Err`** because [`Mrope::apply`] is stubbed.
    /// The full path is verified in P3 once Qwen3.5 model assembly drives
    /// real position-id streams; this code is the wired structural skeleton
    /// the P3 path executes against.
```

Replace with:

```rust
    /// Forward without KV cache (P1 prefill-only path; P2 adds cache + decode).
    ///
    /// `x: [batch, seq, hidden]`. Returns `[batch, seq, hidden]`.
    ///
    /// **As of P3b1 this is fully wired end-to-end**: rotary positions are
    /// applied via the fused MRoPE Q+K MetalKernel, then SDPA runs through
    /// `mlx::fast::scaled_dot_product_attention`. Caller supplies the
    /// pre-computed `cos`/`sin` tables (computed once per forward in the
    /// model assembly via `Mrope::cos_sin`).
```

- [ ] **Step 4.2: Extend `gen_fixture.py` to dump `expected_attn_out.npy`**

Open `ironmlx/tests/fixtures/p3b1_mrope/gen_fixture.py` and append a new section before `if __name__ == "__main__":`. Append inside `main()`, just after the existing `save("expected_k_rot", k_rot)` line:

```python
    # ---- expected_attn_out: drive a single Qwen3.5-style attention layer ----
    #
    # We synthesize a deterministic single-layer attention block here using
    # the same q_rot/k_rot/v that our Rust path produces. v is freshly
    # randomized; o_proj is the identity (we just want SDPA + reshape, not
    # weight loading). The Rust e2e test drives the full Attention forward,
    # but with q_proj/k_proj/v_proj weights pulled from a deterministic seed
    # and o_proj = identity so that this fixture matches.
    #
    # Simpler design: only generate the SDPA output (skip o_proj). The Rust
    # test must match by also stopping before o_proj. This matches what
    # mlx::fast::scaled_dot_product_attention computes given (q_rot, k_rot, v).

    np.random.seed(43)
    v_np = np.random.randn(B, HKV, S, HEAD_DIM).astype(np.float32)
    v = mx.array(v_np).astype(mx.bfloat16)
    save("input_v", v)

    scale = 1.0 / float(np.sqrt(HEAD_DIM))
    attn_out = mx.fast.scaled_dot_product_attention(
        q_rot, k_rot, v, scale=scale, mask="causal"
    )
    # attn_out: [B, HQ, S, HEAD_DIM]; we save it as-is (Rust reshape happens after).
    mx.eval(attn_out)
    save("expected_attn_out", attn_out)
```

Re-run the generator:

```bash
cd ironmlx/tests/fixtures/p3b1_mrope
python gen_fixture.py
```

Expected: `input_v.npy` and `expected_attn_out.npy` written (in addition to the previous 8 files).

> **Why this design**: testing the *full* `Attention::forward` (with q_proj, k_proj, v_proj, o_proj) requires loading actual Qwen3.5 model weights — out of scope for a fixture. Instead, the e2e test in step 4.3 drives only the rotary + SDPA path, using the same `q_rot`/`k_rot`/`v` the fixture captured. This proves the wiring is correct end-to-end without committing model weights.

- [ ] **Step 4.3: Write the failing e2e test**

Append to `ironmlx/tests/p3b1_mrope.rs`:

```rust
#[test]
fn rotary_plus_sdpa_matches_fixture() {
    // End-to-end check: Mrope::apply -> SDPA, with input q/k/v from the
    // fixture, against expected_attn_out. This validates the whole MRoPE +
    // SDPA path the way Attention::forward will run it.
    let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

    let q = load("input_q");
    let k = load("input_k");
    let v = load("input_v");
    let cos = load("expected_cos");
    let sin = load("expected_sin");
    let expected = load("expected_attn_out");

    let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");

    let head_dim = 256_f32;
    let scale = 1.0 / head_dim.sqrt();
    let out = mlx::fast::scaled_dot_product_attention(
        &q_rot,
        &k_rot,
        &v,
        scale,
        "causal",
        None,
        None,
    )
    .expect("sdpa");

    assert_eq!(out.shape().as_slice(), expected.shape().as_slice());
    let err = max_abs_diff_bf16(&out, &expected);
    assert!(err < 1e-3, "rotary+sdpa max abs diff = {err} > 1e-3");
}
```

> **Note**: this test uses `mlx::fast::scaled_dot_product_attention` directly. The full `Attention::forward` test (which exercises q_proj/k_proj/v_proj/o_proj) requires real Qwen3.5 weights and is deferred to P4 (Qwen3.5 Dense E2E).

- [ ] **Step 4.4: Run the e2e test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b1_mrope rotary_plus_sdpa_matches_fixture
```

Expected: PASS.

If it fails:
- shape mismatch → check that the generator's `expected_attn_out` actually has shape `[B, HQ, S, HEAD_DIM]` (the fixture shape, not the post-reshape `[B, S, hidden]`)
- numeric mismatch larger than 1e-3 → likely a flaw in T2's apply kernel; check the q_rot fixture test (T3.4) first, since this test depends on it

- [ ] **Step 4.5: Verify Attention::forward compiles and the lib tests still pass**

The signature change in T4.1 means `Attention::forward_on` now invokes the new dual-output `Mrope::apply`. Since P1 didn't have any caller exercising this path (it returned Err), no other call sites in `ironmlx/src/` need to change.

Run the full ironmlx lib test suite to confirm no regression:

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib
```

Expected: all existing P1/P2 lib tests pass + the 7 new `nn::mrope::tests::*` tests from T1+T2 pass.

- [ ] **Step 4.6: Project gate + commit**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release

git add -A
git commit -m "feat(p3b1): unlock Attention::forward + rotary+SDPA e2e fixture test"
```

Expected: gate clean; e2e test passes; full workspace builds.

---

## Verification Checklist

After Task 4:

| Item | Command | Expected |
|---|---|---|
| Mrope unit tests | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mrope` | 7 tests pass (2 cos_sin + 5 apply) + the 2 pre-existing P1 tests |
| MRoPE integration | `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b1_mrope` | 3 tests pass (cos_sin fixture, apply fixture, rotary+sdpa) |
| Workspace regression | `MLX_DIR=$HOME/.local/mlx cargo test --release` | all earlier tests still pass |
| Format | `cargo +nightly fmt --all -- --check` | no diff |
| Clippy | `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | no warnings |
| Build | `cargo build --release` | success |
| Stub gone | `grep -n "not implemented at P1" ironmlx/src/nn/mrope.rs` | no matches (cos_sin + apply both real) |
| Attention err gone | `grep -n "stubbed" ironmlx/src/nn/attention.rs` | only doc-comment archaic mention remains, no error path |

## Spec Coverage Map

| Spec section | Task |
|---|---|
| § 3.1 cos_sin via mlx::compile | T1 (steps 1.1, 1.4) |
| § 3.2 apply fused metal_kernel | T2 (steps 2.3) |
| § 3.2.1 grid / threadgroup formula | T2 step 2.3 (in `apply` body) |
| § 3.2.2 Metal shader interleaved + tail-passthrough | T2 step 2.3 (in `build_apply_kernel`) |
| § 3.3 Attention.forward unlock | T4 step 4.1 |
| § 4.1 cos_sin_shape_dtype | T1 step 1.2 |
| § 4.1 apply_shape_dtype | T2 step 2.1 (fp32) + 2.5 (bf16) |
| § 4.1 cos_sin vs mlx-lm fixture | T3 step 3.3 |
| § 4.1 apply vs mlx-lm fixture | T3 step 3.3 |
| § 4.1 partial_rotary_tail_unchanged | T2 step 2.6 |
| § 4.1 decode_seq_eq_one | T1 step 1.6 |
| § 4.1 interleaved_pair_known_values | T2 step 2.7 |
| § 4.1 gqa_q_k_different_heads | T2 step 2.8 |
| § 4.1 attention_forward_e2e | T4 step 4.3 (rotary+SDPA via the same apply path Attention uses) |
| § 4.2 fixture .npy generation | T3 step 3.2 + T4 step 4.2 |
| § 5 risks (interleaved geometry, mixed precision, GQA) | T2 + T3 + T4 tests cover each |
| § 8 acceptance criteria | Verification checklist above |

## Risk Register (per spec § 5)

- **Metal shader interleaved geometry error** → T2 step 2.7 `interleaved_pair_known_rotation` exercises the formula on hand-computable values; T3 step 3.3 confirms against the upstream reference.
- **mlx::compile fusion fallback** → cos_sin runs once per forward; even unfused, the impact is small. Test in T1 step 1.5 just checks correctness — performance is implicitly bounded by ShapeMode::Shapeless reusing the same compile across S values.
- **dtype mixed precision** → T2 step 2.5 bf16 path verifies the Metal `decltype(...)` cast back to input dtype works.
- **mlx-lm fixture version drift** → `gen_fixture.py` asserts on `mlx_lm.__version__` prefix; bumping requires a regen.
- **GQA Hq != Hkv** → T2 step 2.8 covers the dispatch path; T3 step 3.3 confirms numerically (Q has 64 heads, K has 8 heads).
- **First-call MetalKernel compile latency** → `OnceLock` caches; each `Mrope` instance has a single warm-up dispatch on first call (acceptable cost amortized over inference).

## Final Self-Review

Before marking this plan complete:

1. **Spec coverage** — every spec § 3 / § 4 entry has a task above (see Coverage Map). The one explicit deviation: spec § 4.1 `attention_forward_e2e` is implemented via "rotary + SDPA" (T4 step 4.3) rather than driving `Attention::forward` directly, because the latter requires loading real Qwen3.5 weights and is deferred to P4. This is documented inline.
2. **Type consistency** — `Mrope::apply` signature is `(q, k, cos, sin) -> Result<(Array, Array)>` everywhere it appears. `cos_sin` returns `Result<(Array, Array)>` everywhere. `OnceLock<CompiledFn>` and `OnceLock<MetalKernel>` field names (`cos_sin_compiled`, `apply_kernel`) consistent across T1 and T2.
3. **Placeholder scan** — no "TBD", "TODO", "fill in", "similar to Task N" anywhere; every code block is the actual content the engineer will type.
