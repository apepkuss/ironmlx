# Affine 5-bit and 6-bit Production Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade native MLX affine 5-bit and 6-bit checkpoint support while preserving one shared affine 2/4/5/6/8-bit runtime implementation.

**Architecture:** Extend only `QuantMode::Affine`, introduce checked packed-width validation, and keep every model path on the existing MLX-native affine dispatch. Existing 4-bit-only kernels retain exact guards. Gemma4 E2B-it and Qwen3.5-2B provide independent real-model and performance gates.

**Tech Stack:** Rust, MLX C++/Metal/NAX at `938006e4`, safetensors, Python HTTP benchmark runners, OpenAI-compatible HTTP API.

## Global Constraints

- Branch: `feat/affine-5bit-6bit` based on `dev@a28d595`.
- Worktree: `/Users/xin/workspace/ironmlx-backend-feat-affine-5bit-6bit`.
- MLX environment: `source /Users/xin/.local/mlx/mlx-env.sh`.
- Affine valid bits become exactly `2, 4, 5, 6, 8`; 3-bit and 7-bit remain unsupported by ironmlx.
- OptiQ's bit-width contract must not change.
- Existing 4-bit custom kernels remain 4-bit only unless a failed clean gate proves a targeted extension is required.
- Raw benchmark output belongs under gitignored `reports/`.
- The eight-hour soak is outside this task.

---

### Task 1: Checked Affine Metadata and Packed Width

**Files:**
- Modify: `ironmlx/src/core/loader.rs`
- Modify: `ironmlx/src/nn/linear.rs`
- Test: inline unit tests in both files

**Interfaces:**
- Produces: `logical_width_from_packed(packed_columns: i32, bits: i32) -> Result<i32>`.
- Produces: affine metadata validation for bits `2 | 4 | 5 | 6 | 8`.
- Preserves: OptiQ metadata acceptance exactly as it exists on `dev`.

- [x] **Step 1: Write failing loader and width tests**

Add tests that parse affine 5/6-bit global and per-prefix metadata, reject
OptiQ 5/6-bit, and prove these mappings:

```rust
assert_eq!(logical_width_from_packed(320, 4).unwrap(), 2560);
assert_eq!(logical_width_from_packed(400, 5).unwrap(), 2560);
assert_eq!(logical_width_from_packed(480, 6).unwrap(), 2560);
assert_eq!(logical_width_from_packed(640, 8).unwrap(), 2560);
assert!(logical_width_from_packed(401, 5).is_err());
```

- [x] **Step 2: Run RED tests**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p ironmlx core::loader::tests --lib
cargo test -p ironmlx nn::linear::tests --lib
```

Expected: FAIL because affine 5/6-bit metadata is rejected and the checked
width helper does not exist.

- [x] **Step 3: Implement minimal shared contract**

Split the current combined affine/OptiQ match so only affine gains 5/6-bit:

```rust
match mode {
    QuantMode::Affine if matches!(bits, 2 | 4 | 5 | 6 | 8) => Ok(()),
    QuantMode::OptiQ if matches!(bits, 2 | 4 | 8) => Ok(()),
    // Existing mode-specific errors remain contextual.
}
```

Implement checked logical width using multiplication before division and reject
non-positive or non-integral results. Replace `packed_columns * (32 / bits)` in
`Linear::in_features()` with this helper.

- [x] **Step 4: Run GREEN tests and commit**

Run the two focused test commands and `cargo fmt`. Expected: PASS.

Commit:

```bash
git add ironmlx/src/core/loader.rs ironmlx/src/nn/linear.rs
git commit -m "feat: accept affine 5-bit and 6-bit metadata"
```

---

### Task 2: Eager Affine Storage Validation

**Files:**
- Modify: `ironmlx/src/core/loader.rs`
- Test: inline loader tests

**Interfaces:**
- Consumes: `logical_width_from_packed` from Task 1.
- Produces: `QuantMeta::validate_storage` rejection before first inference.

- [x] **Step 1: Write failing storage-contract tests**

Construct synthetic rank-2 and rank-3 affine metadata/tensor combinations and
assert rejection for:

```text
weight dtype != uint32
missing affine biases
unsupported group_size
packed width not exactly divisible by bits
scales shape != biases shape
scales trailing width != logical_width / group_size
leading dimensions inconsistent with weight
```

Include valid 5-bit `[10240, 400]` plus `[10240, 40]` scale/bias shapes and
valid 6-bit `[10240, 480]` plus `[10240, 40]` scale/bias shapes.

- [x] **Step 2: Run RED test**

Run `cargo test -p ironmlx core::loader::tests --lib`.
Expected: FAIL because affine storage currently receives no eager validation.

- [x] **Step 3: Implement rank-generic affine validation**

Validate the shared leading dimensions and derive only the trailing logical
width. Keep MXFP and OptiQ behavior mode-specific; do not reinterpret either as
the new affine contract.

- [x] **Step 4: Run GREEN test and commit**

Run the focused loader tests and `cargo fmt`. Expected: PASS.

Commit:

```bash
git add ironmlx/src/core/loader.rs
git commit -m "feat: validate affine packed storage eagerly"
```

---

### Task 3: Native MLX 5/6-bit Component Paths

**Files:**
- Modify: `mlx/tests/p3_quantization.rs`
- Modify: `ironmlx/src/nn/linear.rs`
- Modify: `ironmlx/src/nn/embedding.rs`
- Modify: `ironmlx/src/models/gemma4/quant_fusion.rs`
- Test: inline tests in the same files

**Interfaces:**
- Consumes: shared affine metadata and packed width.
- Produces: tested 5/6-bit QMM, embedding, tied output, and fusion behavior.

- [x] **Step 1: Write failing 5/6-bit component tests**

Parameterize tests over `[5, 6]` and cover:

```rust
for bits in [5, 6] {
    assert_affine_qmm_matches_dequantized_reference(bits, 1);
    assert_affine_qmm_matches_dequantized_reference(bits, 64);
    assert_quantized_tokens_match_dequantize(bits, QuantMode::Affine);
    assert_tied_output_matches_native_qmm(bits, QuantMode::Affine);
}
```

Add dispatch tests proving `should_dispatch_self_qmm`, quantized Embedding
decode, and Gemma4 GeGLU decode all return false/None for 5/6-bit.

- [x] **Step 2: Run RED tests**

Run:

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo test -p mlx --test p3_quantization affine_non_power_of_two
cargo test -p ironmlx nn::linear::tests --lib
cargo test -p ironmlx nn::embedding::tests --lib
cargo test -p ironmlx models::gemma4::quant_fusion::tests --lib
```

Expected: at least the ironmlx 5/6-bit construction tests fail under the old
metadata contract.

- [x] **Step 3: Make only required component changes**

Use the existing `quantized_matmul_on`, `dequantize_on`, and fusion paths. Do
not add a 5/6-bit unpacker or duplicate Linear/Embedding implementations.

- [x] **Step 4: Run GREEN tests and commit**

Run all commands above. Expected: PASS with finite outputs and reference error
within the existing BF16 QMM tolerance.

Commit:

```bash
git add mlx/tests/p3_quantization.rs ironmlx/src/nn/linear.rs ironmlx/src/nn/embedding.rs ironmlx/src/models/gemma4/quant_fusion.rs
git commit -m "test: cover native affine 5-bit and 6-bit paths"
```

---

### Task 4: Model-Specific Gather and MTP Contracts

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Modify: `ironmlx/src/models/diffusion_gemma/moe.rs`
- Modify: `ironmlx/src/models/glm4_moe_lite/mla_attention.rs`
- Modify: `ironmlx/src/models/qwen3_5_moe/mtp.rs`
- Modify: `ironmlx/src/nn/mtp.rs`
- Test: inline tests and existing MTP shape tests

**Interfaces:**
- Consumes: checked logical width and unchanged `QuantMode::Affine` dispatch.
- Produces: proof that direct quantized call sites preserve bits 5/6.

- [x] **Step 1: Write failing propagation and shape tests**

Add synthetic 5/6-bit tests for gather QMM and MTP `fc` dimensions. The MTP
test must use packed columns 400/480 for logical width 2560 and prove both are
accepted as the same input width.

- [x] **Step 2: Run RED tests**

Run focused model and MTP tests. Expected: FAIL on the old power-of-two width
assumption or missing 5/6-bit fixture construction.

- [x] **Step 3: Reuse the shared helper and generic MLX dispatch**

Update comments and any local shape arithmetic that still encodes `/8`; retain
row-wise concatenation along non-packed axes.

- [x] **Step 4: Run GREEN tests and commit**

Run focused tests and `cargo fmt`. Expected: PASS.

Commit:

```bash
git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs ironmlx/src/models/diffusion_gemma/moe.rs ironmlx/src/models/glm4_moe_lite/mla_attention.rs ironmlx/src/models/qwen3_5_moe/mtp.rs ironmlx/src/nn/mtp.rs
git commit -m "test: validate affine 5-bit and 6-bit model paths"
```

---

### Task 5: Real Checkpoint Correctness Harness

**Files:**
- Create: `ironmlx/tests/affine56_real_models.rs`
- Create: `ironmlx/tests/fixtures/affine56/README.md`
- Create: `ironmlx/tests/fixtures/affine56/gen_reference.py`
- Create: generated compact JSON/NPY reference fixtures for four candidates
- Modify: `scripts/quant_validation_matrix.py`
- Modify: `scripts/test_quant_validation_matrix.py`

**Interfaces:**
- Produces: revision-pinned loader/logit/generation parity for Gemma4 and Qwen3.5.
- Produces: HTTP manifests that record affine bit-width and complete scheduler configuration.

- [x] **Step 1: Stage and verify six checkpoints**

Use `/Users/xin/.ironmlx/models` and exact revisions from the design spec:

```bash
hf download mlx-community/gemma-4-e2b-it-5bit --revision dc565aea8c49afb542497310a2d86bf1fd91391f --cache-dir /Users/xin/.ironmlx/models
hf download mlx-community/gemma-4-e2b-it-6bit --revision ebd7756d4e55627e11ae043af9cad8ed6465a2e2 --cache-dir /Users/xin/.ironmlx/models
hf download mlx-community/Qwen3.5-2B-5bit --revision 0934527791eb8008cd84b66550b8ab3eefd15b85 --cache-dir /Users/xin/.ironmlx/models
hf download mlx-community/Qwen3.5-2B-6bit --revision ba2bcf03dd5b502646de7e32b003cf538f2ca4d6 --cache-dir /Users/xin/.ironmlx/models
```

Stage the two 4-bit baselines if absent and run `hf cache verify` for all six.

- [x] **Step 2: Write failing real-model tests**

Require environment variables for each candidate and assert config identity,
complete model construction, finite prefill/decode logits, exact greedy token,
raw max-absolute logit error `< 1.0`, centered max-absolute error `< 0.55`,
centered RMSE `< 0.10`, centered p99 absolute error `< 0.25`, top-64 overlap
`>= 60`, deterministic generation, and blocking-thread execution.

- [x] **Step 3: Run RED tests before implementation completion**

Run each candidate test with its snapshot environment variable. Expected:
FAIL until all loader and packed-shape behavior is complete.

- [x] **Step 4: Generate pinned references and run GREEN tests**

Generate fixtures using the matching MLX checkpoint, record repository revision
and prompt hash, then run all four candidate tests. Expected: PASS.

- [x] **Step 5: Commit harness and fixtures**

```bash
git add ironmlx/tests/affine56_real_models.rs ironmlx/tests/fixtures/affine56 scripts/quant_validation_matrix.py scripts/test_quant_validation_matrix.py
git commit -m "test: validate real affine 5-bit and 6-bit checkpoints"
```

---

### Task 6: Clean Performance Gate

**Files:**
- Create: `scripts/affine56_performance_gate.py`
- Create: `scripts/test_affine56_performance_gate.py`
- Create: `scripts/affine56_prefill.py`
- Create: `scripts/test_affine56_prefill.py`
- Reuse: `scripts/mxfp_strict_decode.py`
- Reuse: `scripts/quant_validation_matrix.py`

**Interfaces:**
- Consumes: scheduler-pinned matrix, strict-decode, and two reverse-order
  strict-prefill manifests.
- Produces: per-architecture, per-bit release status and ratios.

- [x] **Step 1: Write failing performance-gate tests**

Synthetic fixtures must prove:

```text
5-bit threshold = 1.375x matching 4-bit baseline
6-bit threshold = 1.650x matching 4-bit baseline
5-bit must not exceed 1.10x matching 6-bit latency
all manifests must have identical complete scheduler_config
strict prefill requires two rounds with exact reverse model order
checkpoint identity and request configuration must match between rounds
each prefill cell pools ten valid raw TTFT samples before taking the median
missing cells, failed requests, premature strict decode, or missing memory fail the gate
Gemma4 and Qwen3.5 statuses remain independent
```

- [x] **Step 2: Run RED test**

Run `python3 scripts/test_affine56_performance_gate.py`.
Expected: FAIL because the gate does not exist.

- [x] **Step 3: Implement the gate**

Read structured CSV/JSON manifests, reject legacy evidence without scheduler
configuration, reject single-round or non-counterbalanced prefill evidence,
emit `gate.json` and `summary.md`, and keep raw output under
`reports/affine56-performance/`.

- [x] **Step 4: Run GREEN tests and commit**

Run the new test plus all existing quant validation script tests and
`python3 -m py_compile` for changed scripts. Expected: PASS.

Commit:

```bash
git add scripts/affine56_performance_gate.py scripts/test_affine56_performance_gate.py scripts/quant_validation_matrix.py scripts/test_quant_validation_matrix.py
git commit -m "test: add affine 5-bit and 6-bit performance gate"
```

---

### Task 7: HTTP, Long-Context, Concurrency, and Performance Matrix

**Files:**
- Runtime output only: `reports/affine56-validation/`
- Runtime output only: `reports/affine56-strict-decode/`
- Runtime output only: `reports/affine56-prefill/`
- Runtime output only: `reports/affine56-performance/`

**Interfaces:**
- Consumes: six pinned checkpoints and release runners.
- Produces: final clean production evidence excluding the eight-hour soak.

- [x] **Step 1: Build release binaries under NAX MLX**

Run `cargo build --release -p ironmlx -p iron-bench` after sourcing
`mlx-env.sh`.

- [x] **Step 2: Run both architecture matrices**

For Gemma4 and Qwen3.5, run 4/5/6-bit with identical scheduler values and
cover TG=128/TG=512, PP=8K/32K, c=1/c=8, sequential, multi-turn, and stability.

- [x] **Step 3: Run strict full-length decode**

Run TG=512 at c=1/c=8 for all six models. Require every request to return 512
tokens with `finish_reason=length` and zero failures.

- [x] **Step 4: Generate clean gate**

Run `scripts/affine56_performance_gate.py` against only the scheduler-pinned
new runs, including two strict-prefill rounds whose model orders are exact
reverses. Expected: all four candidate mode/architecture statuses PASS.

- [x] **Step 5: Profile and optimize any failed cell**

If a cell fails, attribute it by prefill/decode, QMM/QMV, model layer, and
concurrency before changing code. Add a failing regression benchmark, implement
the smallest reliable optimization, and rerun the complete affected matrix.

Result: the initial single-round Gemma4 PP32K `5/6=1.132` failure was caused by
cross-model order and thermal-state drift, not a 5-bit runtime path regression.
The same checkpoint varied by about 9% across orders. The gate now requires two
exact reverse-order rounds and pools ten raw samples per cell; the original
`1.10` threshold passes without runtime or kernel changes.

---

### Task 8: Final Documentation and Quality Gate

**Files:**
- Modify: `docs/superpowers/specs/2026-07-10-affine-5bit-6bit-production-support-design.md`
- Modify: `docs/superpowers/plans/2026-07-10-affine-5bit-6bit-production-support.md`

**Interfaces:**
- Produces: curated production-readiness record and clean branch.

- [x] **Step 1: Record final evidence**

Document pinned revisions, request counts, correctness errors, worst latency
ratios, memory ratios, scheduler configuration, report paths, and the explicit
exclusion of the eight-hour soak.

- [x] **Step 2: Run all quality commands**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --workspace --all-features
```

Also run all Python script tests, `py_compile`, and `git diff --check`. Record
the pre-existing `p8c_mtp_speculative` replay assertion separately if it remains
the only full-suite failure.

- [x] **Step 3: Verify branch scope and commit**

Confirm raw `reports/` artifacts are ignored, inspect `git diff dev...HEAD`,
and commit the curated result:

```bash
git add docs/superpowers/specs/2026-07-10-affine-5bit-6bit-production-support-design.md docs/superpowers/plans/2026-07-10-affine-5bit-6bit-production-support.md
git commit -m "docs: record affine 5-bit and 6-bit production readiness"
```

Task completion requires both architectures and both new bit-widths to pass all
correctness, HTTP, stability, and clean performance gates. Do not mark the task
complete for a partial or correctness-only result.
