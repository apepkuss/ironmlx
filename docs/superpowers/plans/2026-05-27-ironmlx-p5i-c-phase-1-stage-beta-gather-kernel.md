# P5i.c Phase 1 Stage Beta Gather Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MLX `gather_quantized_matmul_on` at the `gather_qmm_gate_up` call sites with a custom Metal gather kernel, while preserving correctness and proving L1 substep >= 30% reduction plus L2 e2e pp_tps >= 5% gain at PP=128 and PP=512.

**Architecture:** Add a focused `ironmlx::nn::gather_qmm` module that mirrors the current `self_qmm` `MetalKernel::builder + include_str!(metal/*.metal.in) + template_int` dispatch pattern. The kernel consumes existing routing products only: sorted branch uses already-sorted `x` plus one expert id per output slot; default branch uses token `x` plus `[BS, topk]` expert ids and maps each output slot back to its source token. Rust keeps `expand_dims_on` and `slice_on`; the kernel returns fused `[..., 2I]` output for gate/up.

**Tech Stack:** Rust, MLX Rust wrapper, MLX `MetalKernel`, Metal shader source under `ironmlx/src/nn/gather_qmm/metal/`, `ironmlx-bench-kernel` for bounded kernel-only measurement, `tools/p5h_2b_protocol_experiment.py` plus `tools/p5i_c_pp_tps_envelope.py` for production measurement.

---

## Review Corrections Applied

This plan replaces the earlier draft because that draft contained non-executable pseudo-code, invalid `iron-bench` CLI usage, and close-out placeholders. These are now fixed:

- Use the real measurement harness: `tools/p5h_2b_protocol_experiment.py`, not direct `iron-bench --mlx-dir/--logging-mode/--output-dir` commands.
- Use the real Metal dispatch precedent: `ironmlx/src/nn/self_qmm/kernel.rs` uses `mlx::MetalKernel`, `include_str!`, and `template_int`; there is no `build.rs` shader compilation step and no need to touch `mlx-sys`.
- Benchmark actual routing shapes. Default branch is not a sorted-like proxy: output slots are `BS * topk`, with `x_row = slot / topk` and `expert_id = rhs_indices[slot]`.
- Split EG-2 operationally: EG-2a kernel oracle must pass before production wiring; EG-2b 35B regression must pass immediately after wiring and before any L1/L2 acceptance measurement.
- Remove all future-value placeholders from close-out. T6 writes close-out from actual JSON artifacts produced by T2/T3/T5.

## Gates And Non-Negotiables

- Stage Beta implementation execution starts only after Boss approves this plan.
- Baseline for L1/L2 is Stage alpha prep commit `a9c2beb`.
- Stage alpha child-span infra is already shipped; do not re-instrument and do not re-run the skipped Stage alpha sweep.
- No dispatch-time MLX fallback. Unsupported layout, quantization, rank, or shape must return an error before enabling the new path.
- Do not run production e2e sweeps for tile search. Tile search is bench-kernel only.
- Do not start the long L2 production measurement unless EG-1, EG-2a, production wiring smoke, and EG-2b have passed.
- Do not repeat any long measurement just to improve optics. If an accepted gate fails, stop and report the evidence for Boss/controller decision.
- Rust-changing tasks must run:
  ```bash
  export MLX_DIR=$HOME/.local/mlx
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace -- -D warnings
  cargo build --release
  ```
- Final close-out additionally runs `cargo build --release --features p5h-profile`, the oracle tests, and `PYTHONPATH=. uv run pytest tools/p5h_aggregator/tests/ -q`.
- The reviewed plan document may be committed after Boss approval and before execution. No code or measurement-evidence commits before T6. T6 is the single Stage Beta implementation close-out commit.

## File Map

| Path | Status | Task | Responsibility |
|---|---|---|---|
| `ironmlx/src/nn/gather_qmm/mod.rs` | Create | T0-T1 | Public API, shape contracts, quant/layout validation |
| `ironmlx/src/nn/gather_qmm/kernel.rs` | Create | T0-T1 | `MetalKernel` dispatch and output-shape construction |
| `ironmlx/src/nn/gather_qmm/lookup.rs` | Create | T0-T2 | M5 Max prefill tile lookup for sorted/default routing shapes |
| `ironmlx/src/nn/gather_qmm/metal/qmm_gather.metal.in` | Create | T1 | Q4 affine group_size=64 gather kernel |
| `ironmlx/src/nn/mod.rs` | Modify | T0 | Export `gather_qmm` module |
| `ironmlx-bench-kernel/src/gather_qmm.rs` | Create | T0-T2 | Deterministic routing harness, MLX baseline compare, bounded tile sweep |
| `ironmlx-bench-kernel/src/main.rs` | Modify | T0-T2 | Add gather bench mode without breaking current self_qmm mode |
| `ironmlx/tests/gather_qmm_oracle.rs` | Create | T3 | EG-2a synthetic oracle |
| `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` | Modify | T4 | Replace exactly four gate_up MLX call sites; do not touch down path |
| `tools/p5i_c_generation_regression.py` | Create | T4 | EG-2b baseline-vs-candidate 35B generation regression via local OpenAI-compatible servers |
| `docs/p5i-c-phase-1-stage-beta-close-out.md` | Create | T6 | Final evidence and shipped verdict |
| `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` | Modify | T6 | Mark G2/G3/Phase 1 status after evidence exists |

## Shape Contract

The implementation must normalize these four call-site shapes without hardcoding model dimensions:

| Path | `x` shape | `rhs_indices` shape | Output shape |
|---|---|---|---|
| profile sorted | `[BS*k, 1, 1, H]` | `[BS*k, 1]` | `[BS*k, 1, 1, 2I]` |
| profile default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, 2I]` |
| production sorted | `[BS*k, 1, H]` | `[BS*k]` | `[BS*k, 1, 2I]` |
| production default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, 2I]` |

Sorted mapping:
- output slot count = number of rhs expert ids
- source x row = output slot
- expert id = `rhs_indices[slot]`

Default mapping:
- `BS = x.shape()[0]`
- `topk = rhs_indices.shape()[1]`
- output slot count = `BS * topk`
- source x row = `slot / topk`
- expert id = `rhs_indices[slot]`

The kernel must derive `H`, `I`, `E`, and `topk` from tensor shapes and config-owned values at the call site. The verified Qwen3.5-35B-A3B-4bit values are sanity checks only: `H=2048`, `I=512`, `E=256`, `topk=8`, quantization `bits=4`, `group_size=64`, `mode=affine`.

---

## Task 0: Scaffold Module And Bench Harness

**Files:**
- Create: `ironmlx/src/nn/gather_qmm/mod.rs`
- Create: `ironmlx/src/nn/gather_qmm/kernel.rs`
- Create: `ironmlx/src/nn/gather_qmm/lookup.rs`
- Create: `ironmlx/src/nn/gather_qmm/metal/qmm_gather.metal.in`
- Create: `ironmlx-bench-kernel/src/gather_qmm.rs`
- Modify: `ironmlx/src/nn/mod.rs`
- Modify: `ironmlx-bench-kernel/src/main.rs`

- [ ] **Step 0.1: Confirm current dispatch precedent and CLI shape**

Run:
```bash
sed -n '1,160p' ironmlx/src/nn/self_qmm/kernel.rs
sed -n '1,180p' ironmlx-bench-kernel/src/main.rs
rg -n "pub fn gather_quantized_matmul_on|pub fn quantize|MetalKernel::builder" mlx/src ironmlx/src -g '*.rs'
```

Expected:
- `self_qmm/kernel.rs` uses `MetalKernel::builder`, `include_str!`, `dispatch_builder`, `output_shapes`, `output_dtypes`, `grid`, `threadgroup`, and `template_int`.
- `ironmlx-bench-kernel/src/main.rs` currently has one self_qmm mode. Preserve that path while adding gather mode.

- [ ] **Step 0.2: Create `gather_qmm` module skeleton**

Implement:
- `WeightLayout::{GateUpFused, Down}`
- `OutputShapeConstraint::{FusedGateUp2I, SingleDownH}`
- `GatherQmmInputs::{Sorted { x, rhs_indices }, Default { x, rhs_indices }}`
- `QuantParams { bits, group_size, mode }`
- `gather_qmm_on(...) -> crate::Result<Array>`

Validation rules:
- `Down` and `SingleDownH` return an error in Stage Beta v1.
- Only `bits=4`, `group_size=64`, `mode="affine"` is accepted.
- The T0 dispatch body may return a clear "not implemented until T1" error, but the public API and module must compile.

- [ ] **Step 0.3: Add lookup skeleton**

Create `lookup.rs` with:
- `Tile { bm, bn, bk }`
- `InputShapeFamily::{RoutingGatheredSorted, RoutingGatheredDefault}`
- `Phase::{Prefill, Decode}`
- `select_tile(...)`

T0 returns the fixed safe starting tile `(BM=32, BN=64, BK=32)` for both prefill shape families. Decode returns an error at the dispatch layer; decode tile selection is out of scope.

- [ ] **Step 0.4: Add Metal source shell**

Create `qmm_gather.metal.in` with the final kernel name `ironmlx_gather_qmm`. It must compile as Metal source when loaded by `MetalKernel::builder`. T0 can have a minimal body because T1 replaces it before any benchmark gate.

- [ ] **Step 0.5: Add bench gather mode**

Add `ironmlx-bench-kernel/src/gather_qmm.rs` with argument parsing for:
- `--gather-qmm`
- `--shape sorted|default`
- `--m`, `--n-per-expert`, `--k`, `--num-experts`, `--topk`
- `--bm`, `--bn`, `--bk`
- `--runs`, `--warmup`
- `--mlx-baseline`
- `--sweep`

Preserve current self_qmm behavior when `--gather-qmm` is not supplied.

- [ ] **Step 0.6: Verify scaffold**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo build --release -p ironmlx-bench-kernel
target/release/ironmlx-bench-kernel --help
target/release/ironmlx-bench-kernel --gather-qmm --shape sorted --m 64 --n-per-expert 512 --k 2048 --num-experts 256 --topk 8 --mlx-baseline
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected:
- Help lists gather options.
- Gather run exits with the explicit T0 not-implemented message.
- Cargo gates pass.

- [ ] **Step 0.7: Stop and report**

Report files touched and command outputs. Do not commit.

---

## Task 1: Implement Kernel And Deterministic Bench Compare

**Files:**
- Modify: `ironmlx/src/nn/gather_qmm/mod.rs`
- Modify: `ironmlx/src/nn/gather_qmm/kernel.rs`
- Modify: `ironmlx/src/nn/gather_qmm/metal/qmm_gather.metal.in`
- Modify: `ironmlx-bench-kernel/src/gather_qmm.rs`

- [ ] **Step 1.1: Implement Rust dispatch using `MetalKernel` only**

Model `kernel.rs` after `self_qmm/kernel.rs`:
- `const GATHER_QMM_SOURCE: &str = include_str!("metal/qmm_gather.metal.in");`
- one `OnceLock<MetalKernel>`
- `MetalKernel::builder("ironmlx_gather_qmm")`
- inputs: `x`, `w`, `scales`, `biases`, `rhs_indices`
- output: `out`
- `ensure_row_contiguous(true)`
- template ints: `M_OUT`, `N2`, `K`, `TOPK`, `BM`, `BN`, `BK`, `SORTED`, `HAS_BIAS`

Do not touch `mlx-sys`.

- [ ] **Step 1.2: Implement shape normalization**

In Rust before dispatch:
- validate `x` rank is 3 or 4 for production/profile paths.
- validate `rhs_indices` rank is 1 or 2.
- derive `K` from `x.shape().last()`.
- derive `N2 = weight.shape()[1]` and require `N2 % 2 == 0`.
- sorted: derive `M_OUT` from flattened rhs id count and preserve the non-H leading dimensions from `x`.
- default: derive `BS = x.shape()[0]`, `TOPK = rhs_indices.shape()[1]`, `M_OUT = BS * TOPK`, and output shape `[BS, TOPK, 1, N2]`.
- require expert ids are `u32` or cast at call site before invoking this API.

- [ ] **Step 1.3: Implement Metal kernel**

Implement the Q4 affine group_size=64 gather matmul:
- per output slot, map to `x_row` and `expert_id` using sorted/default rules.
- weight base = `expert_id * per_expert_stride`.
- dequantize Q4 affine using the same scale/bias semantics as MLX affine quantization and `self_qmm`.
- accumulate over `K`.
- write fused channels `0..2I` into `out`.
- keep bounds checks for partial `BM` and `BN` tiles.

The first complete implementation may use the single safe tile `(32,64,32)`. Additional tile candidates are enabled only after T1 deterministic correctness passes.

- [ ] **Step 1.4: Build deterministic bench inputs**

In `ironmlx-bench-kernel/src/gather_qmm.rs`:
- build raw `x` and raw per-expert weights from deterministic `Vec<f32>` data.
- quantize raw weights with `mlx::quantization::quantize(..., Some(64), Some(4), "affine", None)`.
- never synthesize random packed weights directly.
- for sorted shape, create `x` with `[M * topk, 1, K]` or `[M * topk, 1, 1, K]` and rhs ids with `[M * topk]`.
- for default shape, create `x` with `[M, 1, 1, K]` and rhs ids with `[M, topk]`.
- when `--mlx-baseline` is set, compare against `mlx::quantization::gather_quantized_matmul_on` with matching `sorted_indices`.
- print CSV with exact header:
  ```text
  shape,bm,bn,bk,candidate_us,mlx_us,ratio
  ```

- [ ] **Step 1.5: Smoke both shape families**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo build --release -p ironmlx-bench-kernel
target/release/ironmlx-bench-kernel --gather-qmm --shape sorted --m 64 --n-per-expert 512 --k 2048 --num-experts 256 --topk 8 --bm 32 --bn 64 --bk 32 --runs 5 --warmup 2 --mlx-baseline
target/release/ironmlx-bench-kernel --gather-qmm --shape default --m 128 --n-per-expert 512 --k 2048 --num-experts 256 --topk 8 --bm 32 --bn 64 --bk 32 --runs 5 --warmup 2 --mlx-baseline
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected:
- Both bench commands print one CSV data row.
- Candidate output is numerically checked inside the bench before timing is trusted.
- Cargo gates pass.

- [ ] **Step 1.6: Stop and report**

Report sorted/default CSV rows and any numerical drift summary. Do not commit.

---

## Task 2: Bounded Tile Search And EG-1

**Files:**
- Modify: `ironmlx/src/nn/gather_qmm/lookup.rs`
- Modify: `ironmlx-bench-kernel/src/gather_qmm.rs`

- [ ] **Step 2.1: Enable only bounded candidate set**

Use this candidate set unless T1 evidence shows a correctness issue for a specific tile:

```text
(32,64,32)
(32,128,32)
(64,64,32)
(64,128,32)
(32,64,64)
(32,128,64)
```

Do not expand the set before Boss/controller approval.

- [ ] **Step 2.2: Run sorted and default sweeps**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo build --release -p ironmlx-bench-kernel
target/release/ironmlx-bench-kernel --gather-qmm --shape sorted --m 64 --n-per-expert 512 --k 2048 --num-experts 256 --topk 8 --sweep --runs 9 --warmup 3 --mlx-baseline > /tmp/p5i-c-stage-beta-eg1-sorted.csv
target/release/ironmlx-bench-kernel --gather-qmm --shape default --m 128 --n-per-expert 512 --k 2048 --num-experts 256 --topk 8 --sweep --runs 9 --warmup 3 --mlx-baseline > /tmp/p5i-c-stage-beta-eg1-default.csv
```

- [ ] **Step 2.3: Compute EG-1 verdict**

Run:
```bash
python3 - <<'PY'
import csv, json
from pathlib import Path

def best(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    for r in rows:
        r["ratio"] = float(r["ratio"])
    return min(rows, key=lambda r: r["ratio"])

sorted_best = best("/tmp/p5i-c-stage-beta-eg1-sorted.csv")
default_best = best("/tmp/p5i-c-stage-beta-eg1-default.csv")
verdict = {
    "sorted": sorted_best,
    "default": default_best,
    "pass": sorted_best["ratio"] <= 0.70 and default_best["ratio"] <= 0.70,
}
Path("/tmp/p5i-c-stage-beta-eg1.json").write_text(json.dumps(verdict, indent=2))
print(json.dumps(verdict, indent=2))
if not verdict["pass"]:
    raise SystemExit("EG-1 FAIL: do not proceed to Task 3")
PY
```

Expected: both sorted and default ratios are `<= 0.70`.

- [ ] **Step 2.4: Populate lookup only after PASS**

Update `select_tile()` with the winning sorted and default prefill tiles from `/tmp/p5i-c-stage-beta-eg1.json`. Decode remains out of scope and must error before dispatch.

- [ ] **Step 2.5: Cargo gates**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

- [ ] **Step 2.6: Stop and report**

Report top tile per shape, ratios, and `/tmp/p5i-c-stage-beta-eg1.json`. Do not commit.

---

## Task 3: Correctness Oracle And EG-2a

**Files:**
- Create: `ironmlx/tests/gather_qmm_oracle.rs`
- May modify: `ironmlx/src/nn/gather_qmm/mod.rs`

- [ ] **Step 3.1: Add deterministic oracle tests**

Create tests using deterministic arrays and `mlx::quantization::quantize`. Do not use random packed weights.

Required non-ignored tests:
- default branch equivalence vs MLX baseline: `[BS,1,1,H] + [BS,topk] -> [BS,topk,1,2I]`
- profile sorted equivalence vs MLX baseline: `[BS*k,1,1,H] + [BS*k,1] -> [BS*k,1,1,2I]`
- production sorted rank equivalence: `[BS*k,1,H] + [BS*k] -> [BS*k,1,2I]`
- gate/up slice equivalence: slicing `[...,0..I]` and `[...,I..2I]` matches MLX fused output slices.
- top-k order invariance: after canonicalizing or reversing the top-k axis back, values match.
- unsupported contracts reject: `WeightLayout::Down`, non-affine mode, non-64 group size, and decode phase.

Numeric threshold:
- Use `max_abs_diff < 0.5` for Q4/bf16 equivalence, matching the existing `self_qmm` test scale.

- [ ] **Step 3.2: Keep 35B regression out of the oracle unit test**

Do not add an ignored Rust generation test here. EG-2b is implemented in T4 as `tools/p5i_c_generation_regression.py`, where baseline and candidate servers can be compared through the real OpenAI-compatible path after production wiring exists.

- [ ] **Step 3.3: Run EG-2a**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo test --release --test gather_qmm_oracle -- --nocapture
python3 - <<'PY'
import json
from pathlib import Path
Path("/tmp/p5i-c-stage-beta-eg2a.json").write_text(json.dumps({"pass": True}, indent=2))
PY
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all non-ignored oracle tests pass. If any test fails, stop and return to T1.

- [ ] **Step 3.4: Stop and report**

Report test names and `/tmp/p5i-c-stage-beta-eg2a.json`. Do not commit.

---

## Task 4: Production Wiring And EG-2b

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Create: `tools/p5i_c_generation_regression.py`

- [ ] **Step 4.1: Replace exactly four gate_up call sites**

Modify only:
- profile sorted gate_up call inside `gate_up_gather_qmm_call` child span;
- profile default gate_up call inside `gate_up_gather_qmm_call` child span;
- production sorted gate_up call;
- production default gate_up call.

Do not modify the two `gather_qmm_down` MLX calls.

- [ ] **Step 4.2: Verify call-site count**

Run:
```bash
rg -n "gather_quantized_matmul_on\\(" ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
rg -n "gather_qmm_on\\(" ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
```

Expected:
- exactly 2 `gather_quantized_matmul_on(` code call sites remain, both for down path;
- exactly 4 `gather_qmm_on(` call sites exist, all for gate_up.

- [ ] **Step 4.3: Build both profile and production**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo build --release --features p5h-profile
cargo test --release --test gather_qmm_oracle -- --nocapture
```

- [ ] **Step 4.4: Run short production harness smoke**

Run the real harness, not direct `iron-bench`:
```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
uv run python tools/p5h_2b_protocol_experiment.py \
  --phase t4 --exp-id p5i_c_beta_smoke \
  --server-lifecycle same_spawn_per_pp \
  --pp-order 128 \
  --logging-mode default_profile \
  --mode production \
  --repeats 1 --pps 128 \
  --runs-per-pp '128:3' \
  --preheat-seconds 0 --preheat-runs 1 \
  --preheat-pp-list '{pp}' \
  --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
  --out-base /tmp/p5i-c-stage-beta-smoke \
  --skip-envelope
```

Expected: one cell directory with `bench.csv`, `server.log`, `meta.json`, and no server ERROR lines.

- [ ] **Step 4.5: Create EG-2b generation regression tool**

Create `tools/p5i_c_generation_regression.py` with these exact behaviors:
- arguments: `--baseline-bin`, `--candidate-bin`, `--model-dir`, `--baseline-port`, `--candidate-port`, `--out-json`;
- start both servers with `serve --model <model-dir> --host 127.0.0.1 --port <port>`;
- wait for both `/v1/chat/completions` endpoints to accept requests;
- send 5 fixed prompts:
  - raw short: `"Explain matrix multiplication in one sentence."`
  - raw medium: `"Write three concise bullets about Apple Silicon unified memory."`
  - raw code: `"Return a Rust function signature for adding two f32 values."`
  - chat short: system `"You are concise."`, user `"What is a MoE router?"`
  - chat medium: system `"You answer technically."`, user `"Summarize why gather matmul is expensive in MoE prefill."`
- request body for each prompt: `model="qwen3.5-moe"`, `max_tokens=16`, `temperature=0`, `seed=42`, `stream=false`;
- compare `choices[0].message.content` exactly between baseline and candidate;
- write `/tmp/p5i-c-stage-beta-eg2b.json` with `pass`, per-prompt baseline/candidate text, and mismatch list;
- terminate both servers on success or failure.

Use only Python standard library modules (`argparse`, `json`, `subprocess`, `time`, `urllib.request`, `urllib.error`, `signal`, `contextlib`, `pathlib`) so no new dependency is introduced.

- [ ] **Step 4.6: Build/reuse baseline binary for EG-2b**

Run:
```bash
if [ ! -x /tmp/p5i-c-stage-beta-baseline-tree/target/release/ironmlx ]; then
  rm -rf /tmp/p5i-c-stage-beta-baseline-tree
  git worktree add /tmp/p5i-c-stage-beta-baseline-tree a9c2beb
  ( cd /tmp/p5i-c-stage-beta-baseline-tree && export MLX_DIR=$HOME/.local/mlx && cargo build --release )
fi
```

- [ ] **Step 4.7: Run EG-2b 35B regression**

Run:
```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
python3 tools/p5i_c_generation_regression.py \
  --baseline-bin /tmp/p5i-c-stage-beta-baseline-tree/target/release/ironmlx \
  --candidate-bin ./target/release/ironmlx \
  --model-dir "$SNAP" \
  --baseline-port 18101 \
  --candidate-port 18102 \
  --out-json /tmp/p5i-c-stage-beta-eg2b.json
test -f /tmp/p5i-c-stage-beta-eg2b.json
cat /tmp/p5i-c-stage-beta-eg2b.json
```

Expected: JSON contains `"pass": true`. If it fails, stop and return to T1/T4. Do not run L1/L2.

- [ ] **Step 4.8: Stop and report**

Report call-site count, smoke output dir, and EG-2b JSON. Do not commit.

---

## Task 5: L1 And L2 Acceptance Measurement

**Files:** no source changes.

- [ ] **Step 5.1: Create or reuse baseline worktree**

Run:
```bash
if [ ! -d /tmp/p5i-c-stage-beta-baseline-tree ]; then
  git worktree add /tmp/p5i-c-stage-beta-baseline-tree a9c2beb
fi
( cd /tmp/p5i-c-stage-beta-baseline-tree && export MLX_DIR=$HOME/.local/mlx && cargo build --release && cargo build --release --features p5h-profile )
```

- [ ] **Step 5.2: Build candidate**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo build --release
cargo build --release --features p5h-profile
```

- [ ] **Step 5.3: L1 same-cohort diagnostic capture**

Use `default_profile` and child spans. Run baseline:
```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
( cd /tmp/p5i-c-stage-beta-baseline-tree && \
  IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1 uv run python tools/p5h_2b_protocol_experiment.py \
    --phase t4 --exp-id baseline_l1 \
    --server-lifecycle same_spawn_per_pp \
    --pp-order 128,512 \
    --logging-mode default_profile \
    --mode production \
    --repeats 3 --pps 128,512 \
    --runs-per-pp '128:15,512:15' \
    --preheat-seconds 300 --preheat-runs 550 \
    --preheat-pp-list '512,{pp}' \
    --inter-run-cooldown-secs 120 \
    --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
    --out-base /tmp/p5i-c-stage-beta-l1 \
    --skip-envelope )
```

Run candidate:
```bash
IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1 uv run python tools/p5h_2b_protocol_experiment.py \
  --phase t4 --exp-id candidate_l1 \
  --server-lifecycle same_spawn_per_pp \
  --pp-order 128,512 \
  --logging-mode default_profile \
  --mode production \
  --repeats 3 --pps 128,512 \
  --runs-per-pp '128:15,512:15' \
  --preheat-seconds 300 --preheat-runs 550 \
  --preheat-pp-list '512,{pp}' \
  --inter-run-cooldown-secs 120 \
  --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
  --out-base /tmp/p5i-c-stage-beta-l1 \
  --skip-envelope
```

- [ ] **Step 5.4: Compute L1 verdict**

Run:
```bash
PYTHONPATH=. python3 - <<'PY'
import json
from pathlib import Path
from statistics import median
from tools.p5h_aggregator.multi_repeat import load_spans_for_child_attribution

def med(exp_id, pp):
    vals = []
    for r in (1, 2, 3):
        log = Path(f"/tmp/p5i-c-stage-beta-l1-{exp_id}-r{r}-pp{pp}/server.log")
        spans = load_spans_for_child_attribution(log)
        vals.extend(s.inclusive_us for s in spans if s.span_name == "gate_up_gather_qmm_call")
    if not vals:
        raise SystemExit(f"no gate_up_gather_qmm_call spans for {exp_id} PP={pp}")
    return median(vals), len(vals)

out = {}
for pp in (128, 512):
    b, bn = med("baseline_l1", pp)
    c, cn = med("candidate_l1", pp)
    reduction = (b - c) / b * 100.0
    out[str(pp)] = {"baseline_median_us": b, "candidate_median_us": c, "reduction_pct": reduction, "baseline_n": bn, "candidate_n": cn, "pass": reduction >= 30.0}
out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
Path("/tmp/p5i-c-stage-beta-l1.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
if not out["pass"]:
    raise SystemExit("L1 FAIL: do not proceed to L2 without Boss/controller decision")
PY
```

- [ ] **Step 5.5: L2 quiet_acceptance production measurement**

Only run if L1 passed. Baseline:
```bash
( cd /tmp/p5i-c-stage-beta-baseline-tree && \
  uv run python tools/p5h_2b_protocol_experiment.py \
    --phase t4 --exp-id baseline_l2 \
    --server-lifecycle same_spawn_per_pp \
    --pp-order 128,512 \
    --logging-mode quiet_acceptance \
    --mode production \
    --repeats 3 --pps 128,512 \
    --runs-per-pp '128:15,512:15' \
    --preheat-seconds 300 --preheat-runs 550 \
    --preheat-pp-list '512,{pp}' \
    --inter-run-cooldown-secs 120 \
    --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
    --out-base /tmp/p5i-c-stage-beta-l2 )
```

Candidate:
```bash
uv run python tools/p5h_2b_protocol_experiment.py \
  --phase t4 --exp-id candidate_l2 \
  --server-lifecycle same_spawn_per_pp \
  --pp-order 128,512 \
  --logging-mode quiet_acceptance \
  --mode production \
  --repeats 3 --pps 128,512 \
  --runs-per-pp '128:15,512:15' \
  --preheat-seconds 300 --preheat-runs 550 \
  --preheat-pp-list '512,{pp}' \
  --inter-run-cooldown-secs 120 \
  --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
  --out-base /tmp/p5i-c-stage-beta-l2
```

- [ ] **Step 5.6: Compute L2 verdict**

Run:
```bash
python3 - <<'PY'
import json
from pathlib import Path

out = {}
for pp in (128, 512):
    b = json.load(open(f"/tmp/p5i-c-stage-beta-l2-baseline_l2-pp{pp}-envelope.json"))
    c = json.load(open(f"/tmp/p5i-c-stage-beta-l2-candidate_l2-pp{pp}-envelope.json"))
    b_tps = b["mean_median"]
    c_tps = c["mean_median"]
    gain = (c_tps - b_tps) / b_tps * 100.0
    out[str(pp)] = {
        "baseline_pp_tps": b_tps,
        "candidate_pp_tps": c_tps,
        "gain_pct": gain,
        "baseline_envelope": b["final_uncertainty_envelope_pct"],
        "candidate_envelope": c["final_uncertainty_envelope_pct"],
        "baseline_verdict": b["verdict"],
        "candidate_verdict": c["verdict"],
        "pass": gain >= 5.0 and b["verdict"] == "PASS" and c["verdict"] == "PASS",
    }
out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
Path("/tmp/p5i-c-stage-beta-l2.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
if not out["pass"]:
    raise SystemExit("L2 FAIL: stop and report; do not rerun without decision")
PY
```

- [ ] **Step 5.7: PP=2048 smoke, not gate**

Run:
```bash
uv run python tools/p5h_2b_protocol_experiment.py \
  --phase t4 --exp-id candidate_pp2048_smoke \
  --server-lifecycle same_spawn_per_pp \
  --pp-order 2048 \
  --logging-mode quiet_acceptance \
  --mode production \
  --repeats 1 --pps 2048 \
  --runs-per-pp '2048:3' \
  --preheat-seconds 0 --preheat-runs 1 \
  --preheat-pp-list '{pp}' \
  --model-dir "$SNAP" --mlx-dir "$HOME/.local/mlx" \
  --out-base /tmp/p5i-c-stage-beta-l2 \
  --skip-envelope
```

- [ ] **Step 5.8: Cleanup baseline worktree**

Run:
```bash
git worktree remove /tmp/p5i-c-stage-beta-baseline-tree
```

- [ ] **Step 5.9: Stop and report**

Report `/tmp/p5i-c-stage-beta-l1.json`, `/tmp/p5i-c-stage-beta-l2.json`, and PP=2048 smoke output. Do not commit.

---

## Task 6: Close-Out, Spec Status, And Single Commit

**Files:**
- Create: `docs/p5i-c-phase-1-stage-beta-close-out.md`
- Modify: `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md`
- Stage all source/test/bench files changed by T0-T5.

- [ ] **Step 6.1: Verify evidence files exist**

Run:
```bash
test -f /tmp/p5i-c-stage-beta-eg1.json
test -f /tmp/p5i-c-stage-beta-eg2a.json
test -f /tmp/p5i-c-stage-beta-eg2b.json
test -f /tmp/p5i-c-stage-beta-l1.json
test -f /tmp/p5i-c-stage-beta-l2.json
python3 - <<'PY'
import json
for path in [
    "/tmp/p5i-c-stage-beta-eg1.json",
    "/tmp/p5i-c-stage-beta-eg2a.json",
    "/tmp/p5i-c-stage-beta-eg2b.json",
    "/tmp/p5i-c-stage-beta-l1.json",
    "/tmp/p5i-c-stage-beta-l2.json",
]:
    data = json.load(open(path))
    if data.get("pass") is not True:
        raise SystemExit(f"{path} is not PASS")
print("all evidence PASS")
PY
```

- [ ] **Step 6.2: Write close-out doc from evidence**

Create `docs/p5i-c-phase-1-stage-beta-close-out.md` with:
- status date from `date +%F`;
- commit lineage: `a9c2beb` Stage alpha baseline, current Stage Beta working tree;
- EG-1 sorted/default ratios and tiles from `/tmp/p5i-c-stage-beta-eg1.json`;
- EG-2a and EG-2b verdicts;
- L1 table from `/tmp/p5i-c-stage-beta-l1.json`;
- L2 table from `/tmp/p5i-c-stage-beta-l2.json`;
- PP=2048 smoke output directory;
- statement that Phase 2 `gather_qmm_down` remains deferred.

After writing, run:
```bash
rg -n "TBD|TODO|FIXME|<|>|2026-05-XX|PLACEHOLDER" docs/p5i-c-phase-1-stage-beta-close-out.md
```

Expected: no matches. If using `set -e`, run it as `! rg -n "TBD|TODO|FIXME|<|>|2026-05-XX|PLACEHOLDER" docs/p5i-c-phase-1-stage-beta-close-out.md`.

- [ ] **Step 6.3: Update spec status**

Update `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md`:
- top status: Phase 1 shipped via Stage Beta close-out commit, with close-out doc path;
- § 6 G2 and G3: satisfied after Boss-approved plan and close-out commit;
- § 10: Stage Beta close-out produced and Phase 1 shipped.

- [ ] **Step 6.4: Final verification**

Run:
```bash
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo build --release --features p5h-profile
cargo test --release --test gather_qmm_oracle -- --nocapture
PYTHONPATH=. uv run pytest tools/p5h_aggregator/tests/ -q
git diff --check
```

- [ ] **Step 6.5: Stage and commit**

Run:
```bash
git add \
  ironmlx/src/nn/gather_qmm/mod.rs \
  ironmlx/src/nn/gather_qmm/kernel.rs \
  ironmlx/src/nn/gather_qmm/lookup.rs \
  ironmlx/src/nn/gather_qmm/metal/qmm_gather.metal.in \
  ironmlx/src/nn/mod.rs \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs \
  ironmlx/tests/gather_qmm_oracle.rs \
  ironmlx-bench-kernel/src/main.rs \
  ironmlx-bench-kernel/src/gather_qmm.rs \
  tools/p5i_c_generation_regression.py \
  docs/p5i-c-phase-1-stage-beta-close-out.md \
  docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md
git status --short
git commit -m "feat(p5i-c-stage-beta): ship custom Metal gather-qmm gate-up kernel"
git log --oneline -3
```

Expected: one Stage Beta close-out commit.

---

## Self-Review Checklist

- [ ] No production wiring before EG-1 and EG-2a.
- [ ] No L1/L2 measurement before EG-2b.
- [ ] No production e2e used for tile search.
- [ ] No dispatch-time MLX fallback.
- [ ] No hardcoded model dimensions in kernel or call sites.
- [ ] Exactly four gate_up call sites wired; down remains untouched.
- [ ] L1 uses `default_profile` and child spans, not `quiet_acceptance`.
- [ ] L2 uses `quiet_acceptance` and P5h+2.e protocol.
- [ ] Long measurements are run once per gate; failures stop for decision.
- [ ] Close-out has actual measured values and no placeholders.
