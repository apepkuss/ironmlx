# MXFP4 and MXFP8 Production Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and serve native MLX MXFP4 and MXFP8 checkpoints with independent production correctness and performance gates.

**Architecture:** Extend `QuantMode` with exact MLX floating-point quantization modes and centralize each mode's metadata and tensor-storage contract. Propagate the mode through generic Linear/Embedding paths and every model-owned direct qmm path while keeping affine-only custom kernels gated out. Validate pinned Qwen3.5-4B checkpoints through Rust parity, external HTTP, long-context, concurrency, stability, and affine-relative performance matrices.

**Tech Stack:** Rust, cxx-backed MLX 938006e4, Metal/NAX, safetensors, Hugging Face CLI, Python standard-library validation tooling, iron-bench.

## Global Constraints

- Work only in `/Users/xin/workspace/ironmlx-backend-feat-mxfp4-mxfp8` on `feat/mxfp4-mxfp8`.
- Use `/Users/xin/.local/mlx` with `MACOSX_DEPLOYMENT_TARGET=26.2`.
- Accept exact lowercase `mxfp4` and `mxfp8`; add no aliases or compatibility branches.
- MXFP4 requires bits 4, group size 32, uint8 scales, and no quant biases.
- MXFP8 requires bits 8, group size 32, uint8 scales, and no quant biases.
- Never route MXFP storage through affine-only self-written kernels.
- Keep MXFP4 and MXFP8 correctness and performance results independent.
- Do not create a pull request.
- If Rust changes, run `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.

---

### Task 1: Quantization Mode and Metadata Contract

**Files:**
- Modify: `ironmlx/src/core/loader.rs`

**Interfaces:**
- Produces: `QuantMode::{Mxfp4,Mxfp8}`, `QuantMode::mlx_mode`, `QuantMode::uses_affine_storage`, `QuantMode::output_dtype`, and `QuantMeta::validate_storage`.
- Consumes: `mlx::{Array, Dtype}` and existing config metadata.

- [ ] **Step 1: Add failing parser and storage tests**

Add tests that parse exact global and override metadata, reject mismatched bits/group sizes, and validate uint8 scales with no biases. The positive assertions are:

```rust
assert_eq!(parse_quant_meta(&json!({
    "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
})).unwrap().unwrap().mode, QuantMode::Mxfp4);
assert_eq!(parse_quant_meta(&json!({
    "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"}
})).unwrap().unwrap().mode, QuantMode::Mxfp8);
```

Storage tests use `Array::zeros((2, 2), Dtype::Uint8)` and assert that an
unexpected biases array or float scale returns a contextual error.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx core::loader::tests::parse_quant_meta_mxfp -- --nocapture
```

Expected: compile failure because the variants and contract methods do not exist.

- [ ] **Step 3: Implement exact mode and storage validation**

Add the enum variants and use this mode mapping:

```rust
match self {
    Self::Affine | Self::OptiQ => "affine",
    Self::Mxfp4 => "mxfp4",
    Self::Mxfp8 => "mxfp8",
}
```

Parse the two exact strings, validate `(4, 32)` and `(8, 32)`, return bf16
from `output_dtype` for MXFP modes, and require uint8 scales plus absent
quantization biases in `validate_storage`.

- [ ] **Step 4: Run loader tests and commit**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx core::loader::tests -- --nocapture
git add ironmlx/src/core/loader.rs
git commit -m "feat: add MXFP quantization metadata"
```

Expected: loader tests pass.

### Task 2: Generic Linear and Embedding Runtime

**Files:**
- Modify: `ironmlx/src/nn/linear.rs`
- Modify: `ironmlx/src/nn/embedding.rs`

**Interfaces:**
- Consumes: `QuantMeta::validate_storage` and `QuantMode::output_dtype`.
- Produces: checkpoint-construction validation plus native MLX mode dispatch.

- [ ] **Step 1: Add failing synthetic MXFP Linear tests**

For each mode, quantize a bf16 `[3, 32]` matrix with MLX, construct
`Linear::new_quant_with_mode(q[0], q[1], None, None, 32, bits, mode)`, and
compare `forward` with `mlx::quantization::quantized_matmul` using the same
mode. Assert max absolute error below `1e-5` after fp32 casting.

- [ ] **Step 2: Add failing Embedding tests**

Construct test-only MXFP embeddings from quantized arrays and assert:

```rust
assert_eq!(embedding.output_dtype(), Dtype::Bfloat16);
assert_abs_diff_eq!(lookup_value, reference_value, epsilon = 1e-3);
assert_abs_diff_eq!(output_value, reference_output, epsilon = 1e-3);
```

Also extend the self-qmm predicate test to assert both MXFP modes return false.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx nn::linear::tests::mxfp -- --nocapture
cargo test -p ironmlx nn::embedding::tests::mxfp -- --nocapture
```

Expected: failures until the new variants and output dtype behavior are wired.

- [ ] **Step 4: Implement construction and output dtype wiring**

In both `from_loader` methods, call:

```rust
qmeta.validate_storage(prefix, &scales, biases.as_ref())?;
```

In `Embedding::output_dtype`, delegate to the mode so MXFP lookup output is
bf16. Keep `qembedding_decode_on` affine-only; its existing guard must return
`None` for MXFP and use gather-then-native-dequantize.

- [ ] **Step 5: Run focused and nn tests, then commit**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx nn::linear::tests -- --nocapture
cargo test -p ironmlx nn::embedding::tests -- --nocapture
git add ironmlx/src/nn/linear.rs ironmlx/src/nn/embedding.rs
git commit -m "feat: run MXFP Linear and Embedding"
```

Expected: all focused tests pass.

### Task 3: Fused and Direct Quantized Model Paths

**Files:**
- Modify: `ironmlx/src/models/gemma4/quant_fusion.rs`
- Modify: `ironmlx/src/models/gemma4/attention.rs`
- Modify: `ironmlx/src/models/gemma4/mlp.rs`
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Modify: `ironmlx/src/models/diffusion_gemma/moe.rs`
- Modify: `ironmlx/src/models/glm4_moe_lite/mla_attention.rs`

**Interfaces:**
- Consumes: `QuantMode::mlx_mode` and `QuantMeta::validate_storage`.
- Produces: mode-preserving fusion, gather-qmm, per-head qmm, and dequantization.

- [ ] **Step 1: Add failing mode-propagation tests**

Add pure constructor/metadata tests proving matching MXFP metadata is fusible,
mixed modes are not fusible, routed expert objects retain their mode, and
`PerHeadQuantLinear` stores the requested mode.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx models::gemma4::quant_fusion::tests -- --nocapture
cargo test -p ironmlx models::qwen3_5_moe::sparse_moe::tests -- --nocapture
cargo test -p ironmlx models::diffusion_gemma::moe::tests -- --nocapture
cargo test -p ironmlx models::glm4_moe_lite::mla_attention::tests -- --nocapture
```

Expected: compile failures for missing mode fields or assertions.

- [ ] **Step 3: Propagate mode and validate grouped storage**

Store `QuantMode` beside `bits` and `group_size` in direct qmm owners. Replace
production `"affine"` arguments with `self.mode.mlx_mode()`. Validate each
gate/up/down or per-head tensor group before constructing fused state. Preserve
literal affine mode only in synthetic affine tests and diagnostic binaries.

- [ ] **Step 4: Run model tests and commit**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx models::gemma4 --lib -- --nocapture
cargo test -p ironmlx models::qwen3_5_moe --lib -- --nocapture
cargo test -p ironmlx models::diffusion_gemma --lib -- --nocapture
cargo test -p ironmlx models::glm4_moe_lite --lib -- --nocapture
git add ironmlx/src/models
git commit -m "feat: propagate MXFP through model quantization paths"
```

Expected: all selected model tests pass.

### Task 4: Pinned Real-Checkpoint Correctness

**Files:**
- Create: `ironmlx/tests/qwen35_mxfp_real_model.rs`
- Create: `ironmlx/tests/fixtures/qwen35_mxfp/README.md`
- Create: `ironmlx/tests/fixtures/qwen35_mxfp/gen_logits.py`
- Create: generated `.npy` fixtures under `ironmlx/tests/fixtures/qwen35_mxfp/`

**Interfaces:**
- Consumes: pinned Hub snapshots and `Qwen35Model`.
- Produces: load, prefill/decode, blocking-thread, argmax, and max-abs parity evidence per mode.

- [ ] **Step 1: Download and verify pinned snapshots**

Run:

```bash
hf download mlx-community/Qwen3.5-4B-mxfp4 --revision 8e9cb97ec8ee0f6a04021220b7a6b5845353df56 --cache-dir /Users/xin/.ironmlx/models
hf cache verify mlx-community/Qwen3.5-4B-mxfp4 --revision 8e9cb97ec8ee0f6a04021220b7a6b5845353df56 --cache-dir /Users/xin/.ironmlx/models
hf download mlx-community/Qwen3.5-4B-mxfp8 --revision a34dd69c7f165c0db75d71061e1bd8f4aeb9eead --cache-dir /Users/xin/.ironmlx/models
hf cache verify mlx-community/Qwen3.5-4B-mxfp8 --revision a34dd69c7f165c0db75d71061e1bd8f4aeb9eead --cache-dir /Users/xin/.ironmlx/models
```

Expected: checksum verification succeeds for both pinned revisions.

- [ ] **Step 2: Add and run real-checkpoint tests**

The test table contains `(label, env_var, QuantMode, bits, revision)` for both
formats. Each case asserts mode/group/bits, model construction, finite logits,
exact greedy argmax, max absolute error below `0.5`, and blocking-thread
forward success.

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
IRONMLX_TEST_REAL_MXFP=1 cargo test --release -p ironmlx --test qwen35_mxfp_real_model -- --test-threads=1 --nocapture
```

Expected: both real checkpoint cases pass.

- [ ] **Step 3: Commit real-model tests and fixtures**

```bash
git add ironmlx/tests/qwen35_mxfp_real_model.rs ironmlx/tests/fixtures/qwen35_mxfp
git commit -m "test: validate real MXFP checkpoints"
```

### Task 5: HTTP Production Matrix

**Files:**
- Modify: `scripts/quant_validation_matrix.py`
- Modify: `scripts/test_quant_validation_matrix.py`
- Create: `reports/mxfp-validation/<timestamp>/...`

**Interfaces:**
- Consumes: release `ironmlx`, `iron-bench`, pinned snapshots, and affine baseline snapshots.
- Produces: raw JSON/logs, `manifest.json`, `summary.csv`, and `summary.md`.

- [ ] **Step 1: Add failing matrix identity tests**

Extend the Python tests so the manifest records checkpoint revision,
quantization mode, expected bits/group size, request completion counts, and
health-before/health-after status for every model.

- [ ] **Step 2: Implement manifest identity checks and run tests**

Run:

```bash
python3 scripts/test_quant_validation_matrix.py
python3 -m py_compile scripts/quant_validation_matrix.py
```

Expected: all Python tests pass.

- [ ] **Step 3: Build release binaries and run MXFP matrix**

Run sequential, multi-turn, strict decode, 8K/32K long context, concurrency 1
and 8, target lengths 128 and 512, and repeated-request stability. Use
`--request-timeout 1800`, `--startup-timeout 900`, and
`--serve-max-cache-cap 65536`.

Expected: zero request/server failures, all health checks pass, and raw results
exist for every requested cell.

- [ ] **Step 4: Commit tooling and evidence**

```bash
git add scripts/quant_validation_matrix.py scripts/test_quant_validation_matrix.py
git commit -m "test: add MXFP production validation matrix"
```

### Task 6: Performance Gate and Optimization Loop

**Files:**
- Create: `reports/mxfp-performance/<timestamp>/...`
- Modify: runtime files only when profiling identifies a format-specific root cause.

**Interfaces:**
- Consumes: MXFP4/MXFP8 and matching affine Qwen3.5-4B snapshots.
- Produces: decode TPOT, concurrent ITL p95, 8K/32K E2E p95, throughput, and memory comparisons.

- [ ] **Step 1: Run matched performance cells**

Compare MXFP4 with `Qwen3.5-4B-MLX-4bit` and MXFP8 with
`Qwen3.5-4B-MLX-8bit` under identical service parameters.

- [ ] **Step 2: Evaluate the 25% gate**

For each mode, calculate `candidate / affine` for sequential TPOT, c=8 ITL
p95, and 8K/32K E2E p95. Every ratio must be `<= 1.25`.

- [ ] **Step 3: Profile and optimize any failing cell**

Use focused qmm/embedding/model-stage timing to locate the failing path. Add a
failing regression test or benchmark assertion, implement only the confirmed
root-cause fix, and rerun the affected matrix before the full matrix.

- [ ] **Step 4: Commit accepted performance work and evidence**

Use a `perf:` commit for runtime optimization and a `docs:` commit for the
curated validation summary. Raw benchmark evidence remains under `reports/`
and is not committed. Do not commit rejected experimental changes.

### Task 7: Final Quality and Production Readiness Gate

**Files:**
- Verify all branch changes.
- Update: `docs/superpowers/specs/2026-07-10-mxfp4-mxfp8-production-support-design.md` status.

**Interfaces:**
- Consumes: all implementation, tests, real-model and performance evidence.
- Produces: production-ready conclusion for each mode.

- [ ] **Step 1: Run required Rust checks**

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands exit 0.

- [ ] **Step 2: Run regression and evidence checks**

Run all focused MXFP tests, Python matrix tests, real-checkpoint tests, and
`git diff --check`. Record the unrelated deterministic MTP baseline failure
without modifying it in this feature branch.

- [ ] **Step 3: Mark completion and commit**

Change the spec status to complete only if both independent correctness and
performance gates pass. Commit final documentation with:

```bash
git add docs
git commit -m "docs: record MXFP production readiness"
```

Expected: clean worktree and no pull request.
