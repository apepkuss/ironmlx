# P8a-stage2 Kernel Fuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift ironmlx single-request decode TG from ~30 tok/s (post-P8a) to ≥40 tok/s on Qwen3.5-4B-MLX-4bit by eliminating ~6.4-8.6ms/step of un-fused Metal kernel dispatch overhead via three structural fixes (RmsNormGated SwiGLU compile-fuse, GatedDeltaNet 4→2 input projection concat, conv1d-output silu compile-fuse).

**Architecture:** All three fixes are confined to `ironmlx/src/nn/{norm.rs, gated_delta_net.rs, linear.rs}`. SwiGLU and silu are fused via module-level `OnceLock<CompiledFn>` initialized lazily on first call (matches mlx-lm's `@partial(mx.compile, shapeless=True)` decorator pattern; coexists with the existing instance-level `compute_g_compiled` OnceLock in `GatedDeltaNet`). Input projection fusion concatenates 4 source quantized tensors into 2 fused `Linear` instances at load time, with output slicing in `forward_on`.

**Tech Stack:** Rust 2021, mlx 0.0.1 safe wrapper (already exposes `mlx::compile::compile` + `mlx::ops::concat::concat`), tokenizers 0.20.4 (no change), anyhow 1, std::sync::OnceLock.

**Spec:** [`docs/superpowers/specs/2026-05-08-p8a-stage2-kernel-fuse-design.md`](../specs/2026-05-08-p8a-stage2-kernel-fuse-design.md)

---

## Conventions Recap

- **TDD per task** (where tests apply): failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before each commit** (per `.claude/CLAUDE.md`):

  ```text
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  MLX_DIR=/Users/sam/.local/mlx cargo build --release
  ```

- `MLX_DIR=/Users/sam/.local/mlx` is the install prefix for full workspace builds (mlx-sys links headers from there). Always include it on `cargo build --release` / `cargo +nightly clippy ...`. `cargo test -p ironmlx --lib` may need it too if any unit test triggers MLX-linked code.
- ASCII-only commit messages.
- `tests/p4_qwen35_logits_match.rs` is the integration regression and requires `--include-ignored` plus `QWEN35_MODEL` env var pointing at the Qwen3.5-4B-MLX-4bit snapshot.
- Server lifecycle for iron-bench: respect that the menubar oMLX.app on `:8001` is Boss's running service — never touch it. Use `:8080` for ironmlx, `:8081` for the source-launched omlx (from `/Volumes/Dev/omlx` via `uv run`).

---

## File Structure (after P8a-stage2)

```text
ironmlx/src/nn/
├── linear.rs          # MODIFIED: + Linear::new_quant constructor (+1 test)
├── norm.rs            # MODIFIED: SwiGLU branch goes via module-level swiglu_fused() (+1 test)
└── gated_delta_net.rs # MODIFIED:
                       #   - struct field rename: in_proj_qkv/_z/_a/_b → in_proj_qkvz/_ba (4→2)
                       #   - from_loader: load 4 tensors via raw API, concat into 2 fused Linears
                       #   - from_components: signature update (4→2 Linear params)
                       #   - forward_on Step 1: 2 matmuls + slice (was 4 matmuls)
                       #   - conv1d-output silu via module-level silu_fused() (+1 test)
iron-bench/README.md   # MODIFIED: append "Measured numbers — post-stage2" table
```

No new files. Three modules touched. The module-level `swiglu_fused()` lives in `norm.rs`; the module-level `silu_fused()` lives in `gated_delta_net.rs`.

---

## Task 1: `Linear::new_quant` constructor + 1 unit test

**Files:**
- Modify: `ironmlx/src/nn/linear.rs`

### Goal

Add a public `new_quant` constructor that composes a `Linear` from already-loaded quantized weight Arrays. This is the load-time seam that Task 3's GDN projection fusion uses to construct the `in_proj_qkvz` / `in_proj_ba` fused projections from concatenated source tensors. No existing path is touched; the existing `from_loader` and `new_fp` continue to work unchanged.

### Steps

- [ ] **Step 1.1: Write the failing test**

`linear.rs` already has a `#[cfg(test)] mod tests` module — verify by reading lines around the bottom of the file (search for `mod tests`). If a test module exists, append. If not, create one. Append this test:

```rust
    #[test]
    fn new_quant_round_trips_via_from_loader_shape() {
        // We cannot construct a real quantized weight from thin air without a
        // tokenizer / safetensors fixture. Instead verify the structural
        // contract: new_quant accepts the 6 fields exactly and stores them in
        // LinearImpl::Quant. Cross-check by inspecting in_features /
        // out_features which compute from the stored shapes.

        // Build a fake quantized weight matching MLX's packed layout for
        // 4-bit, group_size=64: weight shape [out, in/8] u32, scales shape
        // [out, in/64] f32, biases (zero-points) shape [out, in/64] f32.
        let out = 32_i32;
        let in_dim = 64_i32; // single q-group along input axis
        let weight_packed_dim = in_dim / 8; // 4 bits per weight, 8 weights per u32
        let weight_data = vec![0u32; (out * weight_packed_dim) as usize];
        let scales_data = vec![0.01_f32; (out * 1) as usize]; // in/group_size=1
        let weight: Array = (
            weight_data.as_slice(),
            &[out, weight_packed_dim][..],
        )
            .try_into()
            .unwrap();
        let scales: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();
        let biases: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();

        let lin = Linear::new_quant(weight, scales, Some(biases), None, 64, 4);

        assert_eq!(lin.in_features(), in_dim as usize);
        assert_eq!(lin.out_features(), out as usize);
    }
```

This test verifies the constructor accepts the documented args and that `in_features` / `out_features` accessors return correct dimensions. It does NOT exercise the matmul kernel — the round-trip-correctness test against `from_loader` lives in Task 3 (where the actual `in_proj_qkvz` concat path is exercised end-to-end via P4 fixture).

- [ ] **Step 1.2: Run test to verify it fails**

```sh
cd /Volumes/Dev/cxx-mlx
cargo test --release -p ironmlx --lib nn::linear::tests::new_quant -- --nocapture
```

Expected: compile error — `Linear::new_quant` not defined.

- [ ] **Step 1.3: Implement `new_quant`**

Edit `ironmlx/src/nn/linear.rs`. Find the `new_fp` method (currently around line 96 — look for `pub fn new_fp(weight: Array, bias: Option<Array>) -> Self`). Add the new `new_quant` method directly after it, inside the same `impl Linear` block:

```rust
    /// Compose a quantized [`Linear`] from already-loaded Arrays. Used by
    /// callers that fuse multiple weight tensors at load time (e.g.
    /// [`GatedDeltaNet`](crate::nn::GatedDeltaNet)'s concatenated input
    /// projections). Production code that loads a single weight from a
    /// safetensors checkpoint should use [`Linear::from_loader`].
    ///
    /// `weight` is the packed quantized weight matrix; `scales` is per-group
    /// scales; `biases` is per-group zero-points (Some for affine
    /// quantization, None for symmetric); `bias` is the additive linear bias
    /// term separate from `biases` (typically None for Qwen3.5).
    /// `group_size` and `bits` are the quantization metadata (typically
    /// 64 / 4 for Qwen3.5 4-bit checkpoints).
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can
    /// use it. Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new_quant(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            inner: LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            },
        }
    }
```

- [ ] **Step 1.4: Run test to verify it passes**

```sh
cargo test --release -p ironmlx --lib nn::linear::tests::new_quant -- --nocapture
```

Expected: 1 passed.

- [ ] **Step 1.5: Project gate**

```sh
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: clean. Pre-existing mlx-sys upstream C++ `-Wdeprecated-copy` warnings are fine (not Rust-side).

- [ ] **Step 1.6: Commit**

```sh
git add ironmlx/src/nn/linear.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage2): Linear::new_quant constructor

Public composition seam mirroring new_fp but for the Quant variant.
Lets callers that fuse multiple weight tensors at load time (next task:
GatedDeltaNet's qkvz / ba projections) construct a Linear from
already-loaded packed weight + scales + (optional) biases + bias
without re-routing through the Loader API.

1 unit test verifies the structural contract (in_features /
out_features round-trip) using a synthetic 32-out / 64-in
group_size=64 quantized shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: H-K4b SwiGLU + bonus conv1d silu compile-fuse + tests

**Files:**
- Modify: `ironmlx/src/nn/norm.rs` (SwiGLU fuse + 1 test)
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (conv1d silu fuse + 1 test)

### Goal

Replace the un-fused 6-op SwiGLU chain in `RmsNormGated::forward_on` with a `mlx::compile`-traced graph held in a module-level `OnceLock<CompiledFn>`. Same pattern (smaller closure) for the 2-op silu after `conv1d` in `GatedDeltaNet::forward_on`. Both compile cells are stateless free closures (no captured Arrays) and shareable across all calls. Eliminates ~144 dispatches/step (120 from SwiGLU × 24 GDN layers + 24 from conv1d silu × 24 layers).

### Steps

#### 2A — RmsNormGated SwiGLU

- [ ] **Step 2.1: Write the failing test**

Edit `ironmlx/src/nn/norm.rs`. Find the existing `#[cfg(test)] mod tests` block (currently around line 156). Append this test inside it:

```rust
    #[test]
    fn swiglu_fused_matches_reference_path() {
        // Build small [4, 4] gate + normed Arrays, run through the
        // module-level swiglu_fused() compile cell and through a hand-rolled
        // reference (sigmoid → mul → mul). Assert close in fp32.
        let g_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let normed_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let shape = &[4_i32, 4][..];
        let g: Array = (g_data.as_slice(), shape).try_into().unwrap();
        let normed: Array = (normed_data.as_slice(), shape).try_into().unwrap();

        // Fused path
        let fused_outs = swiglu_fused().invoke(&[&g, &normed]).unwrap();
        let fused = fused_outs.into_iter().next().unwrap();
        let fused_vec: Vec<f32> = fused.to_vec().unwrap();

        // Reference unfused path: silu(g) * normed, all in fp32 (inputs already fp32).
        let g_sig = g.sigmoid().unwrap();
        let g_silu = &g * &g_sig;
        let ref_arr = &g_silu * &normed;
        let ref_vec: Vec<f32> = ref_arr.to_vec().unwrap();

        assert_eq!(fused_vec.len(), ref_vec.len());
        for (i, (a, b)) in fused_vec.iter().zip(ref_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at index {i}: fused={a}, ref={b}",
            );
        }
    }
```

The test imports needed (top of `mod tests`) — verify that `use super::*;` is already present (it is, per existing test module). The test calls `swiglu_fused()` directly which Step 2.3 will define as a module-level function.

- [ ] **Step 2.2: Run test to verify it fails**

```sh
cd /Volumes/Dev/cxx-mlx
cargo test --release -p ironmlx --lib nn::norm::tests::swiglu_fused -- --nocapture
```

Expected: compile error — `swiglu_fused` not defined.

- [ ] **Step 2.3: Implement `swiglu_fused()` and switch `forward_on` to use it**

Edit `ironmlx/src/nn/norm.rs`. At the top of the file, add to imports (currently lines 11-14):

```rust
use std::sync::OnceLock;

use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::Result;
```

(Added `use std::sync::OnceLock;` and added `mlx::compile::{compile, CompiledFn, ShapeMode}` to the mlx import line.)

Above the `#[cfg(test)] mod tests` block (at the end of the impl section, after `RmsNormGated`'s impl block closes), add the module-level fused function:

```rust
/// Module-level lazy-initialized SwiGLU graph for [`RmsNormGated`]'s gated
/// path. Mirrors mlx-lm's `@partial(mx.compile, shapeless=True)` decorator
/// pattern from `qwen3_next.py:58-62`. Single `OnceLock` shared across all
/// `RmsNormGated` instances — only one trace per process lifetime.
///
/// Inputs (in order): `g` (gate, any dtype), `normed` (rms-normed hidden,
/// any dtype). Output: f32 Array equal to `silu(g_f32) * normed_f32` —
/// caller is responsible for casting back to the input dtype.
static SWIGLU_FUSED: OnceLock<CompiledFn> = OnceLock::new();

fn swiglu_fused() -> &'static CompiledFn {
    SWIGLU_FUSED.get_or_init(|| {
        compile(
            |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
                let g = inputs[0];
                let normed = inputs[1];
                let g_f32 = mlx::ops::cast::astype(g, Dtype::Float32)?;
                let g_sig = g_f32.sigmoid()?;
                let g_silu = &g_f32 * &g_sig;
                let normed_f32 = mlx::ops::cast::astype(normed, Dtype::Float32)?;
                let mul_f32 = &g_silu * &normed_f32;
                Ok(vec![mul_f32])
            },
            ShapeMode::Shapeless,
        )
        .expect("compile swiglu_fused")
    })
}
```

Replace the body of `RmsNormGated::forward_on` (currently lines 129-153). The current body matches:

```rust
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
                let g_silu = &g_f32 * &g_sig;
                let normed_f32 = mlx::ops::cast::astype(&normed, Dtype::Float32)?;
                let mul = &g_silu * &normed_f32;
                Ok(mlx::ops::cast::astype(&mul, hidden_dtype)?)
            }
            None => Ok(mlx::ops::cast::astype(&normed, hidden_dtype)?),
        }
    }
```

Change ONLY the `Some(g)` arm:

```rust
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
                // Precise SwiGLU via module-level mlx::compile cell — fuses
                // 6 elementwise ops (astype, sigmoid, mul, astype, mul) into
                // a single Metal dispatch. Output is fp32; cast back to
                // hidden_dtype outside the compiled graph (per-call data).
                let outs = swiglu_fused().invoke(&[g, &normed])?;
                let mul_f32 = outs
                    .into_iter()
                    .next()
                    .expect("swiglu_fused returns one output");
                Ok(mlx::ops::cast::astype(&mul_f32, hidden_dtype)?)
            }
            None => Ok(mlx::ops::cast::astype(&normed, hidden_dtype)?),
        }
    }
```

The `target: StreamOrDevice` is no longer used inside the gate arm (the compiled graph runs on the default stream). Compiler will flag `target` as unused for that arm — silence with `let _ = target;` if needed, but better: keep the `target` binding — it's still used by `mlx::fast::rms_norm_on` at line just above.

- [ ] **Step 2.4: Run the test to verify pass**

```sh
cargo test --release -p ironmlx --lib nn::norm::tests::swiglu_fused -- --nocapture
```

Expected: 1 passed.

#### 2B — conv1d silu fuse

- [ ] **Step 2.5: Write the failing test**

Edit `ironmlx/src/nn/gated_delta_net.rs`. Find the existing `#[cfg(test)] mod tests` block (look for `mod tests`; if it exists at the bottom of the file). Append this test:

```rust
    #[test]
    fn silu_fused_matches_reference_path() {
        let x_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let shape = &[2_i32, 16][..];
        let x: Array = (x_data.as_slice(), shape).try_into().unwrap();

        // Fused path
        let outs = silu_fused().invoke(&[&x]).unwrap();
        let fused = outs.into_iter().next().unwrap();
        let fused_vec: Vec<f32> = fused.to_vec().unwrap();

        // Reference unfused path
        let x_sig = x.sigmoid().unwrap();
        let ref_arr = &x * &x_sig;
        let ref_vec: Vec<f32> = ref_arr.to_vec().unwrap();

        assert_eq!(fused_vec.len(), ref_vec.len());
        for (i, (a, b)) in fused_vec.iter().zip(ref_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at index {i}: fused={a}, ref={b}",
            );
        }
    }
```

If `mod tests` does NOT exist at the bottom of `gated_delta_net.rs`, create it:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_fused_matches_reference_path() {
        // ... body as above
    }
}
```

- [ ] **Step 2.6: Run the test to verify it fails**

```sh
cargo test --release -p ironmlx --lib nn::gated_delta_net::tests::silu_fused -- --nocapture
```

Expected: compile error — `silu_fused` not defined.

- [ ] **Step 2.7: Implement `silu_fused()` and switch the conv1d-output silu to use it**

Edit `ironmlx/src/nn/gated_delta_net.rs`. The file already imports `OnceLock`, `CompiledFn`, `ShapeMode` (verify lines 12-15). No new imports needed.

After the `GatedDeltaNet` impl block closes and BEFORE the `#[cfg(test)] mod tests` block (or before EOF if no test block exists), add the module-level fused silu:

```rust
/// Module-level lazy-initialized silu graph for the conv1d-output gating in
/// [`GatedDeltaNet::forward_on`]. Single `OnceLock` shared across all 24 GDN
/// layers — one trace per process lifetime.
///
/// Input: `x` (any dtype). Output: `x * sigmoid(x)` (silu) preserving dtype.
static SILU_FUSED: OnceLock<CompiledFn> = OnceLock::new();

fn silu_fused() -> &'static CompiledFn {
    SILU_FUSED.get_or_init(|| {
        mlx::compile::compile(
            |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
                let x = inputs[0];
                Ok(vec![x * &x.sigmoid()?])
            },
            ShapeMode::Shapeless,
        )
        .expect("compile silu_fused")
    })
}
```

Locate the conv1d silu lines in `forward_on` (currently lines 290-294 — the comment `// Step 2b: conv1d + silu` followed by `let conv_out_sig = conv_out.sigmoid_on(target)?;` and `let conv_out = &conv_out * &conv_out_sig;`). Replace those three lines with:

```rust
        // Step 2b: conv1d + silu (silu fused via module-level compile cell)
        let conv_out = self.conv1d.forward_on(&conv_input, target)?;
        let outs = silu_fused().invoke(&[&conv_out])?;
        let conv_out = outs
            .into_iter()
            .next()
            .expect("silu_fused returns one output");
```

So the diff is: keep the `conv1d.forward_on` call, replace the explicit sigmoid+mul with one `silu_fused()` invoke.

- [ ] **Step 2.8: Run both tests to verify pass**

```sh
cargo test --release -p ironmlx --lib nn::norm::tests::swiglu_fused nn::gated_delta_net::tests::silu_fused -- --nocapture
```

Expected: 2 passed.

#### 2C — Project gate + commit

- [ ] **Step 2.9: Project gate**

```sh
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: clean. If clippy flags `target` as unused in the SwiGLU gate arm, that's because the compiled graph runs on the default stream and the target argument is not threaded into it. Either accept the warning (we use `target` elsewhere in the function), or rename the binding to `_target` if clippy enforces the lint at error level. Current workspace lints permit `unused_variables` at warning level only, so this should pass.

- [ ] **Step 2.10: Commit**

```sh
git add ironmlx/src/nn/norm.rs ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage2): fuse RmsNormGated SwiGLU + conv1d silu via mlx::compile

H-K4b — RmsNormGated::forward_on's Some(gate) arm previously dispatched
6 elementwise ops per call (astype, sigmoid, mul, astype, mul, plus the
trailing astype). Replace the inner 5 with a module-level OnceLock
swiglu_fused() compile cell traced once on first call. Across 24 GDN
layers per decode step that's -120 Metal kernel dispatches.

Bonus — gated_delta_net's conv1d-output silu (sigmoid + mul) replaced
by silu_fused() compile cell. Across 24 GDN layers that's -24
dispatches/step.

Both compile cells are stateless free closures (no Array captures),
shareable across all instances. Mirrors mlx-lm's
@partial(mx.compile, shapeless=True) decorator pattern from
qwen3_next.py:58-62.

2 unit tests: each fused path matches the unfused reference within
1e-5 fp32 tolerance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: H-K7 GatedDeltaNet projection concat fusion

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (struct fields, from_loader, from_components, forward_on)

### Goal

Concatenate the 4 source quantized tensors `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` (axis 0 = output dim) into 2 fused `Linear` instances `in_proj_qkvz` and `in_proj_ba` at load time. `forward_on` fires 2 quantized matmuls instead of 4, then `slice_strided` recovers per-projection outputs as lazy views (no buffer copy). Eliminates 48 quantized matmul dispatches per step (2 × 24 GDN layers).

mlx-lm's fused order: `in_proj_qkvz` is `qkv ⊕ z` (qkv first, z second along axis 0), `in_proj_ba` is `b ⊕ a` (b first, a second). Match this ordering so the slice offsets parallel mlx-lm's `qwen3_next.py:200-203`.

### Steps

- [ ] **Step 3.1: Write the failing test**

Edit `ironmlx/src/nn/gated_delta_net.rs`. Append to the `#[cfg(test)] mod tests` block (created in Task 2 if it didn't already exist):

```rust
    #[test]
    fn qkvz_concat_load_matches_separate_matmuls() {
        // Use Linear::new_fp (fp32, no quant) so we can exercise the concat
        // logic without needing a real quantized fixture. The concat math is
        // identical for fp and quantized weights along axis 0.
        let hidden_size = 4_i32;
        let qkv_out = 6_i32;
        let z_out = 4_i32;

        // Random-ish but reproducible input.
        let x_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x: Array = (x_data.as_slice(), &[1_i32, 1, hidden_size][..])
            .try_into()
            .unwrap();

        // Construct two separate fp Linears.
        let w_qkv_data: Vec<f32> = (0..qkv_out * hidden_size)
            .map(|i| (i as f32) * 0.01 - 0.05)
            .collect();
        let w_z_data: Vec<f32> = (0..z_out * hidden_size)
            .map(|i| (i as f32) * 0.02 + 0.03)
            .collect();
        let w_qkv: Array = (w_qkv_data.as_slice(), &[qkv_out, hidden_size][..])
            .try_into()
            .unwrap();
        let w_z: Array = (w_z_data.as_slice(), &[z_out, hidden_size][..])
            .try_into()
            .unwrap();

        let lin_qkv = Linear::new_fp(w_qkv.clone(), None);
        let lin_z = Linear::new_fp(w_z.clone(), None);
        let out_qkv: Vec<f32> = lin_qkv.forward(&x).unwrap().to_vec().unwrap();
        let out_z: Vec<f32> = lin_z.forward(&x).unwrap().to_vec().unwrap();

        // Construct fused Linear with weights concatenated along axis 0.
        let w_fused = mlx::ops::shape::concatenate(&[&w_qkv, &w_z], 0).unwrap();
        let lin_fused = Linear::new_fp(w_fused, None);
        let out_fused = lin_fused.forward(&x).unwrap();

        // Slice the fused output: [..., :qkv_out] and [..., qkv_out:qkv_out+z_out]
        let fused_qkv = mlx::ops::indexing::slice_strided(
            &out_fused,
            &[0_i32, 0, 0][..],
            &[1_i32, 1, qkv_out][..],
            &[1_i32, 1, 1][..],
        )
        .unwrap();
        let fused_z = mlx::ops::indexing::slice_strided(
            &out_fused,
            &[0_i32, 0, qkv_out][..],
            &[1_i32, 1, qkv_out + z_out][..],
            &[1_i32, 1, 1][..],
        )
        .unwrap();
        let fused_qkv_vec: Vec<f32> = fused_qkv.to_vec().unwrap();
        let fused_z_vec: Vec<f32> = fused_z.to_vec().unwrap();

        // Assert byte-identical (same ops, same order — should be exact).
        assert_eq!(fused_qkv_vec, out_qkv);
        assert_eq!(fused_z_vec, out_z);
    }
```

- [ ] **Step 3.2: Run the test to verify it passes**

```sh
cd /Volumes/Dev/cxx-mlx
cargo test --release -p ironmlx --lib nn::gated_delta_net::tests::qkvz_concat -- --nocapture
```

Expected: 1 passed. (The test only exercises stable mlx ops + Linear::new_fp, not the GDN struct itself. It documents the contract that Step 3.3-3.5's load-time concat must satisfy.)

- [ ] **Step 3.3: Update `GatedDeltaNet` struct fields**

Edit `ironmlx/src/nn/gated_delta_net.rs`. Find the `GatedDeltaNet` struct (currently lines 71-85). Replace the 4 separate `in_proj_*` fields with 2 fused fields:

```rust
pub struct GatedDeltaNet {
    /// Fused (qkv, z) input projection — concatenated along axis 0 at load
    /// time. Output `[B, S, conv_dim + value_dim]`; sliced in `forward_on`.
    in_proj_qkvz: Linear,
    /// Fused (b, a) input projection — concatenated along axis 0 at load
    /// time. Output `[B, S, num_v_heads * 2]`; sliced in `forward_on`.
    in_proj_ba: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,   // [num_v_heads]
    dt_bias: Array, // [num_v_heads]
    cfg: GatedDeltaNetConfig,
    compute_g_compiled: OnceLock<CompiledFn>,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
}
```

Also update the doc comment block above the struct (currently lines 57-70) to reflect the fused projections:

```rust
/// Qwen3.5 / Qwen3-Next "linear attention" branch — recurrent SSM with
/// delta rule and scalar gating.
///
/// Mirrors mlx-lm's `Qwen3NextGatedDeltaNet`
/// (`/Volumes/Dev/mlx-lm/mlx_lm/models/qwen3_5.py:85-205`). Components:
///
/// - `in_proj_qkvz` — fused matmul (Q+K+V outputs concat'd with the gate
///   `z`); equivalent to mlx-lm's `in_proj_qkvz`. Sliced in `forward_on`.
/// - `in_proj_ba` — fused matmul (forget signal `b` + decay signal `a`);
///   equivalent to mlx-lm's `in_proj_ba`. Sliced in `forward_on`.
/// - `conv1d` — depthwise temporal mixing across the Q/K/V channels (then
///   silu via module-level fused compile cell)
/// - `norm` — `RmsNormGated`: `silu(z) * rms_norm(y)` final mixing
/// - `out_proj` — back to `hidden_size`
/// - `a_log` / `dt_bias` — per-head learned parameters for compute_g
```

- [ ] **Step 3.4: Update `from_loader` to concat at load time**

Replace the body of `pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self>` (currently lines 89-124) with:

```rust
    /// Production constructor: load all weight tensors + a_log + dt_bias.
    /// Source checkpoint stores `in_proj_qkv` / `_z` / `_b` / `_a` as 4
    /// separate tensors (HF / mlx-community convention). At load time we
    /// concat (qkv, z) and (b, a) along the output axis to produce 2 fused
    /// quantized Linears, mirroring mlx-lm's `in_proj_qkvz` / `in_proj_ba`
    /// pre-merged form. Eliminates 2 quantized matmul dispatches per step.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self> {
        let qmeta = loader.quant_meta().ok_or_else(|| {
            anyhow!("{prefix}: GatedDeltaNet input projections require quantized loader")
        })?;

        // Fuse in_proj_qkv + in_proj_z → in_proj_qkvz (output axis 0).
        let qkv_w = loader.tensor(&format!("{prefix}.in_proj_qkv.weight"))?.clone();
        let qkv_s = loader.tensor(&format!("{prefix}.in_proj_qkv.scales"))?.clone();
        let qkv_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_qkv.biases"))
            .cloned();
        let z_w = loader.tensor(&format!("{prefix}.in_proj_z.weight"))?.clone();
        let z_s = loader.tensor(&format!("{prefix}.in_proj_z.scales"))?.clone();
        let z_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_z.biases"))
            .cloned();

        let qkvz_weight = mlx::ops::shape::concatenate(&[&qkv_w, &z_w], 0)?;
        let qkvz_scales = mlx::ops::shape::concatenate(&[&qkv_s, &z_s], 0)?;
        let qkvz_biases = match (qkv_b_opt, z_b_opt) {
            (Some(a), Some(b)) => Some(mlx::ops::shape::concatenate(&[&a, &b], 0)?),
            (None, None) => None,
            _ => {
                return Err(anyhow!(
                    "{prefix}: in_proj_qkv.biases and in_proj_z.biases must agree on Some/None"
                ));
            }
        };
        let in_proj_qkvz = Linear::new_quant(
            qkvz_weight,
            qkvz_scales,
            qkvz_biases,
            None,
            qmeta.group_size,
            qmeta.bits,
        );

        // Fuse in_proj_b + in_proj_a → in_proj_ba (b first, a second; matches
        // mlx-lm's qwen3_next.py:201 ordering).
        let b_w = loader.tensor(&format!("{prefix}.in_proj_b.weight"))?.clone();
        let b_s = loader.tensor(&format!("{prefix}.in_proj_b.scales"))?.clone();
        let b_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_b.biases"))
            .cloned();
        let a_w = loader.tensor(&format!("{prefix}.in_proj_a.weight"))?.clone();
        let a_s = loader.tensor(&format!("{prefix}.in_proj_a.scales"))?.clone();
        let a_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_a.biases"))
            .cloned();

        let ba_weight = mlx::ops::shape::concatenate(&[&b_w, &a_w], 0)?;
        let ba_scales = mlx::ops::shape::concatenate(&[&b_s, &a_s], 0)?;
        let ba_biases = match (b_b_opt, a_b_opt) {
            (Some(p), Some(q)) => Some(mlx::ops::shape::concatenate(&[&p, &q], 0)?),
            (None, None) => None,
            _ => {
                return Err(anyhow!(
                    "{prefix}: in_proj_b.biases and in_proj_a.biases must agree on Some/None"
                ));
            }
        };
        let in_proj_ba = Linear::new_quant(
            ba_weight,
            ba_scales,
            ba_biases,
            None,
            qmeta.group_size,
            qmeta.bits,
        );

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
            in_proj_qkvz,
            in_proj_ba,
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
```

- [ ] **Step 3.5: Update `from_components` signature + body**

Replace the existing `from_components` (currently lines 130-159). The new signature takes 2 fused Linears:

```rust
    /// Test/composition seam: build from pre-built nn building blocks.
    ///
    /// `in_proj_qkvz` and `in_proj_ba` must already be the fused forms
    /// (output dim concatenated along axis 0). For tests that build
    /// separate qkv/z/a/b Linears, concat the underlying weights via
    /// `mlx::ops::shape::concatenate` first then pass a single fused
    /// Linear here.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        in_proj_qkvz: Linear,
        in_proj_ba: Linear,
        conv1d: Conv1d,
        norm: RmsNormGated,
        out_proj: Linear,
        a_log: Array,
        dt_bias: Array,
        cfg: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            in_proj_qkvz,
            in_proj_ba,
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
```

If any callers in `ironmlx/tests/` or elsewhere use `from_components` with the old 4-Linear signature, those callers must be updated to concat first. Search:

```sh
grep -rn "from_components" /Volumes/Dev/cxx-mlx/ironmlx/ 2>/dev/null
```

If callers exist outside the file, update each to pass the fused Linears (or add a temporary helper in the test file that concats fp Linear weights using `mlx::ops::shape::concatenate` + `Linear::new_fp`). Defer that work to within the same commit if it's mechanical; if it's substantial, escalate as DONE_WITH_CONCERNS.

- [ ] **Step 3.6: Update `forward_on` Step 1 (projections) and remove split helpers**

Locate the Step 1 projections in `forward_on` (currently lines 271-275):

```rust
        // Step 1: projections
        let qkv = self.in_proj_qkv.forward_on(x, target)?; // [B, S, conv_dim]
        let z = self.in_proj_z.forward_on(x, target)?; // [B, S, value_dim]
        let a = self.in_proj_a.forward_on(x, target)?; // [B, S, num_v_heads]
        let b = self.in_proj_b.forward_on(x, target)?; // [B, S, num_v_heads]
```

Replace with:

```rust
        // Step 1: fused projections + slice (was 4 quantized matmuls; now 2).
        let qkvz = self.in_proj_qkvz.forward_on(x, target)?; // [B, S, conv_dim + value_dim]
        let ba = self.in_proj_ba.forward_on(x, target)?;     // [B, S, num_v_heads * 2]

        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let num_v_heads = self.cfg.num_v_heads;

        // Slice qkvz → qkv (axis=2, [0, conv_dim)) and z (axis=2, [conv_dim, conv_dim + value_dim))
        let qkv = mlx::ops::indexing::slice_strided(
            &qkvz,
            &[0_i32, 0, 0][..],
            &[batch, seq, conv_dim][..],
            &[1_i32, 1, 1][..],
        )?;
        let z = mlx::ops::indexing::slice_strided(
            &qkvz,
            &[0_i32, 0, conv_dim][..],
            &[batch, seq, conv_dim + value_dim][..],
            &[1_i32, 1, 1][..],
        )?;

        // Slice ba → b (first num_v_heads outputs) + a (next num_v_heads outputs)
        let b = mlx::ops::indexing::slice_strided(
            &ba,
            &[0_i32, 0, 0][..],
            &[batch, seq, num_v_heads][..],
            &[1_i32, 1, 1][..],
        )?;
        let a = mlx::ops::indexing::slice_strided(
            &ba,
            &[0_i32, 0, num_v_heads][..],
            &[batch, seq, num_v_heads + num_v_heads][..],
            &[1_i32, 1, 1][..],
        )?;
```

The variables `qkv`, `z`, `a`, `b` retain the same names and shapes as the pre-fuse path, so all downstream code (Step 2 conv1d, Step 3 split-per-head, Step 4 q/k rms_norm, Step 5 compute_g, Step 6 beta sigmoid) is unchanged.

`batch` and `seq` are already in scope from the pre-flight validation block (`let batch = dims[0]; let seq = dims[1];` at lines 268-269).

- [ ] **Step 3.7: Run all unit tests in this module**

```sh
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx cargo test --release -p ironmlx --lib nn::gated_delta_net:: -- --nocapture --test-threads=1
```

Expected: pre-existing tests still pass, plus the 2 new tests (`silu_fused_matches_reference_path` from Task 2, `qkvz_concat_load_matches_separate_matmuls` from this task).

If any pre-existing test fails because it constructed the GDN via `from_components` with 4 Linears, fix that test by concatenating the weights first (use the same `mlx::ops::shape::concatenate` pattern as the new test).

- [ ] **Step 3.8: Project gate**

```sh
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: clean. clippy may flag `from_components` with `clippy::too_many_arguments` if the new arg count drops below clippy's threshold — the existing `#[allow(clippy::too_many_arguments)]` attribute on the function should be retained (8 args still triggers the lint at default 7-threshold).

- [ ] **Step 3.9: Commit**

```sh
git add ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a-stage2): fuse GatedDeltaNet input projections (qkv+z, b+a)

Concat the 4 source quantized weights (in_proj_qkv / _z / _b / _a) into
2 fused quantized Linears (in_proj_qkvz / in_proj_ba) at load time via
mlx::ops::shape::concatenate along axis 0 (output dim). Per-group
quantization metadata (q-block=64 along input axis) is unaffected by
output-axis concat — bit-for-bit equivalent. Forward path fires 2
quantized matmuls instead of 4, then slice_strided recovers per-projection
outputs as lazy views. Across 24 GDN layers per decode step that's -48
quantized matmul dispatches.

Struct field rename: 4 fields collapse to 2. from_components signature
updated accordingly.

1 unit test verifies the concat + slice contract (using fp Linear
surrogates; quantized correctness is verified end-to-end by the P4
fixture in the next task).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: P4 fixture regression check (no commit unless regressions)

**Files:**
- Read-only verification: `tests/p4_qwen35_logits_match.rs`

### Goal

Run the existing P4 logits-match fixture against a real Qwen3.5-4B-MLX-4bit checkpoint to verify all three structural fixes (SwiGLU fuse, silu fuse, projection concat) preserve byte-identical token sequences vs the mlx-lm reference. Greedy is deterministic; the new fused paths are mathematically equivalent transformations of the unfused paths.

### Steps

- [ ] **Step 4.1: Run the P4 logits-match fixture**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
QWEN35_MODEL="$SNAP" MLX_DIR=/Users/sam/.local/mlx \
  cargo test --release -p ironmlx --test p4_qwen35_logits_match -- --nocapture --test-threads=1 --include-ignored
```

Expected: PASS, byte-identical token sequence to mlx-lm reference. Output should end with `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;`.

- [ ] **Step 4.2: If FAIL, root-cause before proceeding**

If the fixture fails post-stage2, the fused paths have diverged from the unfused reference. Common root causes (in order of likelihood):

1. **SwiGLU compile cell `target` stream mismatch** — the compiled graph runs on the default stream; if the rest of `forward_on` was on a different stream, results may differ slightly. Less likely (cast back to dtype is on `target`); more likely the compile is producing slightly different fp32 intermediate. Compare `swiglu_fused()` output dtype: should be f32. The `astype(&mul_f32, hidden_dtype)` outside the compile cell handles the cast back.

2. **Slice offset miscount in Step 3.6** — the slice `[conv_dim, conv_dim + value_dim]` for `z` should EXCLUSIVE upper bound matches `slice_strided`'s end index. Re-derive: qkvz output dim = `conv_dim + value_dim`; z is the second segment, indices `[conv_dim, conv_dim + value_dim)`. Per `slice_strided(start, stop, stride)` semantics in mlx, `stop` is the exclusive upper bound — so `stop = conv_dim + value_dim` is correct. Same logic for b/a slices over `ba`: b is `[0, num_v_heads)`, a is `[num_v_heads, 2 * num_v_heads)`.

3. **Concat order swapped** — verify `concatenate(&[&qkv_w, &z_w], 0)` produces a weight whose output rows are qkv first, then z (same as the slice order). For ba: `concatenate(&[&b_w, &a_w], 0)` — b first, a second. If you swapped, the slice extracts the wrong projection.

4. **Quantization scales / biases concat axis mismatch** — for 4-bit Qwen3.5 with q-block=64 along input axis, scales shape is `[out, in/group_size]`. Concat along axis 0 (output) preserves layout. If you accidentally concat'd along axis 1, the per-group scale alignment would break. Verify the axis arg in `concatenate(...)` is `0`.

If the issue is non-obvious, dispatch a small diagnostic subagent to instrument the fused path with `eprintln!` at each step and compare intermediate Array values against the unfused reference. Do not commit until the fixture passes.

- [ ] **Step 4.3: If PASS, no commit**

The fixture passing in isolation is a regression check, not a deliverable. No commit. Proceed to Task 5.

---

## Task 5: iron-bench rerun + acceptance + README update

**Files:**
- Modify: `iron-bench/README.md` (append a "Measured numbers — post-stage2" subsection)

### Goal

Run the full iron-bench protocol against the post-stage2 build, verify the spec acceptance criteria are met (decode TG ≥ 40 tok/s; gap to omlx < 30%), and capture the actual numbers in the README for future reference.

### Steps

- [ ] **Step 5.1: Verify ports + servers state**

```sh
for p in 8001 8080 8081; do
  pid=$(lsof -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null | tail -n +2 | awk '{print $1, $2}')
  if [ -n "$pid" ]; then echo "  :$p occupied: $pid"; else echo "  :$p free"; fi
done
```

Expected: `:8001` may show `python3 <pid>` (the menubar oMLX.app — never touch). `:8080` and `:8081` should be free.

- [ ] **Step 5.2: Start ironmlx server (terminal 1, or background)**

```sh
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
MLX_DIR=/Users/sam/.local/mlx cargo run --release -p ironmlx -- serve --model "$SNAP" --port 8080
```

Wait for `ironmlx server listening on http://127.0.0.1:8080`.

- [ ] **Step 5.3: Start omlx server from /Volumes/Dev/omlx (terminal 2, or background)**

```sh
cd /Volumes/Dev/omlx
uv run python -m omlx.cli serve \
  --model-dir /Users/sam/.omlx/models \
  --port 8081 \
  --no-cache \
  --max-concurrent-requests 4 \
  --log-level info
```

Wait until `curl -sf http://127.0.0.1:8081/v1/models` returns the model list (~30s for cold load).

- [ ] **Step 5.4: Run iron-bench (terminal 3)**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir "$SNAP" \
  --model Qwen3.5-4B-MLX-4bit \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 3 --warmup 1 \
  --format markdown
```

Expected: ~3-5 minutes total. Markdown report on stdout. Capture the report for Step 5.6.

**Acceptance criteria** (all must hold):
- ironmlx Decode TG (tok/s) median ≥ **40** at PP=128, 512, 2048 (post-P8a was 28-32).
- ironmlx vs omlx Decode TG gap < **30%** at all three PP cells.
- ironmlx TTFT and Prefill PP medians within ±5% of post-P8a numbers (no regression on prefill).
- `cached_tokens > 0 detected for: (none)` warning unchanged.

If TG hits ≥40 but gap to omlx remains >30%, stage2 is **accepted** — note as a follow-up under "P8a-stage3" in a separate spec.

If TG fails to hit 40, do not commit "acceptance". Investigate (see spec §6.3 for diagnostic paths). Potential pitfalls:
- `mlx::compile` may not be fusing as expected on this MLX version. Profile via Instruments.app Time Profiler attached to the running ironmlx process.
- `Linear::new_quant` + concat path may produce numerically equivalent but performance-different output. The P4 fixture passing means *correctness* is fine; if perf is unchanged, the concat itself isn't the bottleneck (defer to P8a-stage3).

- [ ] **Step 5.5: Tear down servers**

```sh
kill $(pgrep -f "ironmlx.*serve.*--port 8080") 2>/dev/null
kill $(pgrep -f "omlx.cli serve.*--port 8081") 2>/dev/null
sleep 1
for p in 8080 8081; do
  o=$(lsof -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null | tail -n +2 | head -1)
  if [ -n "$o" ]; then echo "  :$p still occupied"; else echo "  :$p free"; fi
done
```

Verify both ports are free. Do NOT touch port 8001 (the menubar oMLX.app — Boss's running service).

- [ ] **Step 5.6: Update `iron-bench/README.md`**

Edit `/Volumes/Dev/cxx-mlx/iron-bench/README.md`. Find the existing "## Measured numbers — Qwen3.5-4B-MLX-4bit, M-series Apple Silicon" section (added by P8a; ends with the "TTFT / Prefill" paragraph). Append a new subsection AFTER it with the actual measured numbers from Step 5.4:

```markdown
### Post-stage2 numbers (after kernel-fuse)

After P8a-stage2 (RmsNormGated SwiGLU compile-fuse + GDN projection 4→2 concat
+ conv1d silu compile-fuse), the same protocol re-run yields:

| Target  | Decode TG (tok/s) median | TTFT PP=128 (ms) | TTFT PP=2048 (ms) | Prefill PP=2048 (tok/s) |
|---------|--------------------------|------------------|-------------------|-------------------------|
| ironmlx | <fill from runs>         | <fill>           | <fill>            | <fill>                  |
| omlx    | <fill from runs>         | <fill>           | <fill>            | <fill>                  |

Stage2 closed the decode gap from ~1.7-1.9× to <X×. The remaining gap (if any)
is documented as P8a-stage3 follow-up.
```

Replace each `<fill ...>` placeholder with the actual median value from Step 5.4's iron-bench output. Also replace `<X×>` with the actual ratio (e.g. `1.2×`).

- [ ] **Step 5.7: Project gate (sanity for the README change)**

```sh
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

Expected: clean (the README change doesn't touch Rust).

- [ ] **Step 5.8: Commit**

```sh
git add iron-bench/README.md
git commit -m "$(cat <<'EOF'
docs(iron-bench): record P8a-stage2 measured numbers

Captures iron-bench rerun results after P8a-stage2 kernel fuse
(SwiGLU + projection concat + silu). Decode TG closed from <X1>x to
<X2>x of omlx; numbers populated from the rerun.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<X1>` and `<X2>` with the actual pre/post stage2 ratios.

---

## Self-Review Notes

Verified before saving the plan:

**Spec coverage**:
- §2 Architecture (the 3 fixes), §3.1 H-K4b SwiGLU, §3.3 H-K7-bonus conv1d silu → Task 2
- §3.2 H-K7 GDN projection concat (Linear::new_quant + from_loader fuse + forward slice) → Tasks 1 + 3
- §5 Error handling (qmeta missing, biases inconsistent, expect on compile failure) → exact code in Tasks 1, 2, 3
- §6 Testing (3 unit tests + P4 fixture + iron-bench acceptance) → Tasks 1, 2, 3 (one unit test each); Tasks 4 + 5 (regression + perf)
- §7 Risks → addressed in Step 4.2 root-cause guide
- §8 Out of scope → no tasks
- §9 Acceptance gate → Step 5.4 acceptance bullets

**No placeholders**: All code blocks are complete; the only `<fill>` markers in Step 5.6 are intended literal placeholders for the measured-numbers table (the implementer fills them in with iron-bench output).

**Type consistency**: `is_pipelinable` (Sampler), `swiglu_fused()` and `silu_fused()` (module-level fns), `Linear::new_quant`, `in_proj_qkvz` / `in_proj_ba` (struct fields, used consistently across struct def, from_loader, from_components, forward_on), `mlx::ops::shape::concatenate` (the actual mlx wrapper API for concat — not `mlx::ops::concat::concat` which doesn't exist in this codebase per the existing use in `gated_delta_net.rs:16`, `mlx::ops::shape::concatenate`).

**Bite-sized**: Each task has 5-10 explicit checkbox steps. TDD honored for Tasks 1-3 (failing test → run → impl → run pass → commit). Tasks 4-5 are verification/acceptance (no implementation, just run + record).
