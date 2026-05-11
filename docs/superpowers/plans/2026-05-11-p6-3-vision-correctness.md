# P6.3 Vision Functional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive ironmlx's real-inference vision quality to functional parity with mlx-vlm on four test images by closing four quantitative gates (preprocess byte diff, vision encoder max_diff, end-to-end logits diff, semantic correctness on Item 3 images).

**Architecture:** Three sequenced sub-phases — P6.3a (preprocess alignment, Gate 1), P6.3b (op-level vision encoder alignment, Gate 2), P6.3c (semantic + production-path verification, Gates 3 & 4). Each sub-phase is a diagnose-fix-verify loop driven by the P6.1 diff pipeline.

**Tech Stack:** Rust + ironmlx + mlx-rs + Python (mlx-vlm venv) + safetensors + matplotlib + bash. Builds on P6.1/P6.2 infrastructure (vision-dump cargo feature, run_p6_1_diff.sh, diff_pipeline.py).

**Spec:** `docs/superpowers/specs/2026-05-11-p6-3-functional-correctness-design.md` (commit `49133ef`)

**Branch base:** `ironmlx-p6-1-vision-diff` (last commit `49133ef`)
**Branch target:** `ironmlx-p6-3-vision-correctness`

---

## File Structure

New files:
- `ironmlx/tests/p6_3a_preprocess_dump.rs` — feature-gated Rust integration test, runs ironmlx `image_processor::preprocess` and dumps result as `00_ironmlx_pv.safetensors` (note: distinct from `00_pixel_values.safetensors`).
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess.py` — Python helper that compares the two preprocess outputs and emits a focused report.
- `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh` — driver for P6.3a stage.
- `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh` — driver for P6.3b stage with op-level dumps.
- `ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.py` — Python driver that starts ironmlx server, hits it with 4 images, runs Gate 4 sub-checks.
- `ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/02_op_level_hooks.patch` — archive of mlx-vlm fork's op-level hook additions.
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md` — final P6.3 close-out report with the acceptance table.

Modified files (across the plan):
- `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` — add intra-block hooks (Task 9).
- `ironmlx/src/models/qwen3_5/vision/block.rs` — add `forward_with_name_prefix` for intra-block dumps + likely fix candidates (Tasks 9, 11-13).
- `ironmlx/src/models/qwen3_5/vision/mod.rs` — switch dump path to use name-prefix variant (Task 9).
- `ironmlx/src/models/qwen3_5/image_processor.rs` — likely fix candidates (Tasks 3-4).

---

## Task 1: Branch setup + P6.3a stage scaffolding

**Files:**
- No new files yet.

- [ ] **Step 1.1: Create branch from `ironmlx-p6-1-vision-diff`**

```bash
cd /Volumes/Dev/cxx-mlx
git checkout ironmlx-p6-1-vision-diff
git checkout -b ironmlx-p6-3-vision-correctness
git log -1 --oneline   # expect: 49133ef docs(p6.3): ...
```

- [ ] **Step 1.2: Verify P6.1/P6.2 baseline still reproducible**

Sanity-check the existing diff pipeline still works before extending it:

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh 2>&1 | tail -5
```

Expected last line: `=== Done. Report: <path>/report.md ===`

Inspect the new report's `01_patch_embed_out` row — must still be `0.0000` (P6.2 baseline) and `29_merger_out` must still be `~0.90`.

- [ ] **Step 1.3: No commit yet — Task 1 is environment verification**

---

## Task 2: P6.3a — Build ironmlx preprocess dumper

**Files:**
- Create: `ironmlx/tests/p6_3a_preprocess_dump.rs`

This task adds the diagnostic capability to dump ironmlx's own `image_processor::preprocess` output for byte-level comparison against mlx-vlm's. We dump in two related forms:

- `00_ironmlx_pv_native.safetensors` — raw output `[N, T=2, C=3, 16, 16]` (what ironmlx PatchEmbed consumes)
- `00_ironmlx_pv_vlmlayout.safetensors` — same data reshaped+transposed to `[N, 1536]` C-major (what mlx-vlm's processor produces; allows direct byte-diff)

- [ ] **Step 2.1: Write the integration test**

Create `ironmlx/tests/p6_3a_preprocess_dump.rs`:

```rust
//! P6.3a: dump ironmlx's `image_processor::preprocess` output for byte-level
//! comparison against mlx-vlm's processor output (Gate 1 diagnostic).
//!
//! Driven by:
//!   IMAGE_PATH=/path/to/coco_sample.jpg
//!   IRONMLX_PREPROCESS_DUMP_DIR=/tmp/p6_diff/ironmlx_pre
//!   MLX_DIR=$HOME/.local/mlx
//!   cargo test -p ironmlx --features vision-dump --test p6_3a_preprocess_dump --release -- --ignored

#![cfg(feature = "vision-dump")]

use std::path::Path;

use mlx::Dtype;

use ironmlx::models::qwen3_5::image_processor::preprocess;

#[test]
#[ignore]
fn p6_3a_preprocess_dump() {
    let img_path = std::env::var("IMAGE_PATH").expect("IMAGE_PATH env required");
    let out_dir = std::env::var("IRONMLX_PREPROCESS_DUMP_DIR")
        .expect("IRONMLX_PREPROCESS_DUMP_DIR env required");

    let bytes = std::fs::read(Path::new(&img_path)).expect("read image");
    let (pv_native, grid_h, grid_w) = preprocess(&bytes).expect("preprocess");

    eprintln!(
        "[p6_3a_preprocess_dump] grid_thw=[1,{grid_h},{grid_w}] native_shape={:?}",
        pv_native.shape().as_slice()
    );

    // Dump native [N, T=2, C=3, 16, 16] layout
    {
        let path = format!("{out_dir}/00_ironmlx_pv_native.safetensors");
        let mut map = std::collections::HashMap::new();
        let pv_bf16 = mlx::ops::cast::astype(&pv_native, Dtype::Bfloat16).expect("cast");
        mlx::transforms::eval(&[&pv_bf16]).expect("eval native");
        map.insert("tensor".to_string(), pv_bf16);
        let metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        mlx::io::save_safetensors(&path, &map, &metadata).expect("save native");
    }

    // Re-shape to mlx-vlm's [N, 1536] C-major layout for direct byte-diff:
    // ironmlx is [N, T, C, H, W]; transpose [0, 2, 1, 3, 4] -> [N, C, T, H, W]
    // then reshape [N, C*T*H*W] = [N, 1536].
    let n = pv_native.shape().as_slice()[0];
    let pv_c_first =
        mlx::ops::shape::transpose_axes(&pv_native, &[0_i32, 2, 1, 3, 4][..]).expect("transpose");
    let pv_flat = pv_c_first.reshape(&[n, 1536_i32][..]).expect("reshape");
    let pv_flat_bf16 = mlx::ops::cast::astype(&pv_flat, Dtype::Bfloat16).expect("cast flat");
    mlx::transforms::eval(&[&pv_flat_bf16]).expect("eval flat");
    {
        let path = format!("{out_dir}/00_ironmlx_pv_vlmlayout.safetensors");
        let mut map = std::collections::HashMap::new();
        map.insert("tensor".to_string(), pv_flat_bf16);
        let metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        mlx::io::save_safetensors(&path, &map, &metadata).expect("save flat");
    }

    eprintln!("[p6_3a_preprocess_dump] dumped native + vlmlayout to {out_dir}");
}
```

- [ ] **Step 2.2: Smoke-run the new test**

```bash
mkdir -p /tmp/p6_diff/ironmlx_pre
MLX_DIR=/Users/sam/.local/mlx \
IMAGE_PATH=/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg \
IRONMLX_PREPROCESS_DUMP_DIR=/tmp/p6_diff/ironmlx_pre \
cargo test -p ironmlx --features vision-dump --test p6_3a_preprocess_dump --release -- --ignored 2>&1 | tail -10
```

Expected: test passes; `ls /tmp/p6_diff/ironmlx_pre/` shows the two `.safetensors`.

- [ ] **Step 2.3: CI gauntlet**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --features vision-dump --test p6_3a_preprocess_dump -- -D warnings 2>&1 | tail -3
```

Both clean.

- [ ] **Step 2.4: Commit**

```bash
git add ironmlx/tests/p6_3a_preprocess_dump.rs
git commit -m "feat(p6.3a): p6_3a_preprocess_dump integration test"
```

---

## Task 3: P6.3a — Preprocess diff tool + Gate 1 measurement

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess.py`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh`

- [ ] **Step 3.1: Write the focused diff tool**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess.py`:

```python
"""P6.3a: focused byte-level diff between ironmlx and mlx-vlm preprocess outputs.

Unlike diff_pipeline.py (which does full 29-tensor vision-tower diff), this
tool only compares the two `00_*pv*` files and emits a Gate-1 verdict line
that the close-out report can ingest.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_preprocess.py \
        --vlm /tmp/p6_diff/python/00_pixel_values.safetensors \
        --iron /tmp/p6_diff/ironmlx_pre/00_ironmlx_pv_vlmlayout.safetensors \
        --out /path/to/p6_3a_preprocess_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_tensor(path: Path) -> np.ndarray:
    arr = mx.load(str(path))
    t = arr["tensor"] if isinstance(arr, dict) else arr
    t = t.astype(mx.float32)
    mx.eval(t)
    return np.array(t)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, type=Path)
    parser.add_argument("--iron", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate", type=float, default=0.05, help="Gate 1 threshold")
    args = parser.parse_args()

    a = load_tensor(args.vlm)
    b = load_tensor(args.iron)
    if a.shape != b.shape:
        msg = f"shape mismatch: vlm {a.shape} vs iron {b.shape}"
        print(f"ERROR: {msg}", file=sys.stderr)
        args.out.write_text(f"# P6.3a Preprocess Diff\n\n**ERROR**: {msg}\n")
        return 2

    d = np.abs(a - b)
    stats = {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "count_above_1e-3": int((d > 1e-3).sum()),
        "count_above_1e-2": int((d > 1e-2).sum()),
        "total": int(d.size),
    }

    # Top 5 outliers
    flat_diff = d.flatten()
    k = min(5, len(flat_diff))
    idxs = np.argpartition(flat_diff, -k)[-k:]
    idxs = idxs[np.argsort(flat_diff[idxs])[::-1]]
    outliers = [
        {"idx": int(i),
         "vlm": float(a.flatten()[i]),
         "iron": float(b.flatten()[i]),
         "diff": float(flat_diff[i])}
        for i in idxs
    ]

    pass_gate = stats["max"] < args.gate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P6.3a Preprocess Diff (Gate 1)",
        "",
        f"- Tensor shape: {list(a.shape)}",
        f"- Gate 1 threshold: < {args.gate}",
        f"- Observed max_diff: **{stats['max']:.4f}**",
        f"- Gate 1 verdict: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
        "## Stats",
        "",
        f"- max: {stats['max']:.6f}",
        f"- mean: {stats['mean']:.6f}",
        f"- p99: {stats['p99']:.6f}",
        f"- count > 1e-3: {stats['count_above_1e-3']} / {stats['total']}",
        f"- count > 1e-2: {stats['count_above_1e-2']} / {stats['total']}",
        "",
        "## Top 5 outliers",
        "",
        "| flat_idx | vlm | iron | abs_diff |",
        "| --- | --- | --- | --- |",
    ]
    for o in outliers:
        lines.append(f"| {o['idx']} | {o['vlm']:.4f} | {o['iron']:.4f} | {o['diff']:.4f} |")
    args.out.write_text("\n".join(lines) + "\n")

    print(f"[diff_preprocess] gate 1 max_diff = {stats['max']:.4f} ({'PASS' if pass_gate else 'FAIL'})")
    print(f"[diff_preprocess] report → {args.out}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3.2: Write the P6.3a driver script**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh`:

```bash
#!/usr/bin/env bash
# P6.3a: drive Gate 1 measurement (preprocess byte diff).
# Required env: MLX_DIR, QWEN35_MODEL (for mlx-vlm side dump only).
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
PY_DIR="${PY_DIR:-/tmp/p6_diff/python}"
IRON_PRE_DIR="${IRON_PRE_DIR:-/tmp/p6_diff/ironmlx_pre}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_3a-$STAMP"

mkdir -p "$PY_DIR" "$IRON_PRE_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$IRON_PRE_DIR"/*.safetensors

echo "=== Step 1: mlx-vlm pixel_values dump ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"

echo "=== Step 2: ironmlx preprocess dump ==="
cd "$REPO_ROOT"
MLX_DIR="$MLX_DIR" \
    IMAGE_PATH="$FIXTURE_DIR/coco_sample.jpg" \
    IRONMLX_PREPROCESS_DUMP_DIR="$IRON_PRE_DIR" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test p6_3a_preprocess_dump \
        -- --ignored 2>&1 | tail -5

echo "=== Step 3: preprocess byte diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_preprocess.py" \
    --vlm "$PY_DIR/00_pixel_values.safetensors" \
    --iron "$IRON_PRE_DIR/00_ironmlx_pv_vlmlayout.safetensors" \
    --out "$REPORT_DIR/p6_3a_preprocess_report.md" || true

echo "=== Report: $REPORT_DIR/p6_3a_preprocess_report.md ==="
```

- [ ] **Step 3.3: Make it executable and run**

```bash
chmod +x /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh 2>&1 | tail -8
```

Expected: report written. The Gate-1 verdict line will say PASS or FAIL.

- [ ] **Step 3.4: Inspect Gate 1 baseline**

```bash
REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3a-* | head -1)
head -15 "$REPORT/p6_3a_preprocess_report.md"
```

Record the **baseline `max_diff`**. This is the "Before P6.3a" number in the close-out table.

- [ ] **Step 3.5: Commit driver + tool**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess.py \
        ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh
git add -f "$REPORT/p6_3a_preprocess_report.md"
git commit -m "feat(p6.3a): preprocess diff tool + Gate 1 baseline report"
```

---

## Task 4: P6.3a — Diagnose + fix preprocess divergence

This is an **iterative diagnose-fix-verify** task. If Gate 1 already passed in Task 3 (unlikely — ironmlx uses different image decoder + Lanczos3 from PIL), skip to Task 5.

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/image_processor.rs` (likely fix candidates)

- [ ] **Step 4.1: Identify divergence layer**

Open the report from Task 3.4 and look at the top-5 outliers + the overall max_diff distribution.

Run a stratified probe (one-time helper, save to `/tmp/p6_3a_probe.py`):

```python
#!/usr/bin/env python
"""Stratify the preprocess diff: where does the divergence come from?"""
import numpy as np
import mlx.core as mx
from PIL import Image
import sys
sys.path.insert(0, "/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl")
from diff_preprocess import load_tensor

# Step A: confirm smart_resize output dims match (mlx-vlm hint via h2,w2 = h*16, w*16 from grid)
img = Image.open("/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg").convert("RGB")
print("orig (W,H):", img.size)

# Step B: hand-run mlx-vlm's smart_resize + Lanczos3 + normalize, save raw normalized pixels
# (replicate run_python_dump.py's path but with intermediate dumps)
# ... see comments in run_python_dump.py for the exact preprocessing path ...

# Step C: load ironmlx's preprocess output ([N, 1536] C-major) and decode patch[0]
iron = load_tensor("/tmp/p6_diff/ironmlx_pre/00_ironmlx_pv_vlmlayout.safetensors")
vlm = load_tensor("/tmp/p6_diff/python/00_pixel_values.safetensors")

print(f"iron[0,:8]: {iron[0, :8]}")
print(f"vlm [0,:8]: {vlm [0, :8]}")
print(f"diff at  0: {abs(iron[0, 0] - vlm[0, 0]):.6f}")
print(f"diff @ patch_0 mean: {abs(iron[0] - vlm[0]).mean():.6f}")
```

Run with `~/.venvs/mlxvlm-ref/bin/python /tmp/p6_3a_probe.py`.

Decide which layer of `preprocess` is the source by reading the diff pattern:

| Symptom in report | Likely root cause | Fix candidate |
| --- | --- | --- |
| max_diff > 0.5, large fraction (>50%) of values differ | `Lanczos3` resample implementation diverges from PIL.Image.LANCZOS | `image_processor.rs::preprocess` resize step |
| max_diff 0.02–0.1, scattered outliers | image decoder (jpeg-decoder vs libjpeg) DCT differences | accept (decoder layer; deferred) OR call `image` crate's libjpeg-turbo backend |
| max_diff > 1 only at patch boundaries | `patchify` byte-layout bug | `image_processor.rs::patchify` |
| max_diff in (0, 0.005) but always positive bias | normalize formula off-by-one | `image_processor.rs::normalize_pixel` |
| max_diff = 0 (or < 1e-5) | Gate 1 already green; commit + Task 5 | n/a |

- [ ] **Step 4.2: Apply the targeted fix**

Based on the table above, edit `ironmlx/src/models/qwen3_5/image_processor.rs`. Read the existing function first:

```bash
grep -n "fn preprocess\|fn normalize_pixel\|fn patchify\|fn smart_resize\|Lanczos3" /Volumes/Dev/cxx-mlx/ironmlx/src/models/qwen3_5/image_processor.rs | head -10
```

Apply the smallest possible change that addresses the root cause. Each fix candidate has its own shape:

- **Lanczos3 alignment**: the `image` crate's `Lanczos3` filter has a different windowing constant from PIL's. The most reliable byte-level match is to call PIL via a Python subprocess for the resize step — but that's heavy. The lighter alternative: switch the test path to feed pre-resized pixels from mlx-vlm's side and skip ironmlx's resize. Decide on the cost/benefit at this point — if Gate 1 only needs to be `< 0.05` and the Lanczos3 diff is ~0.02-0.04, that's already green.
- **Decoder difference**: similar — the `jpeg-decoder` crate's DCT is IDCT-accurate but rounds differently from libjpeg. Cap the impact by always re-loading the same JPEG via the same pipeline.
- **Normalize off-by-one**: `normalize_pixel` is at [ironmlx/src/models/qwen3_5/image_processor.rs](ironmlx/src/models/qwen3_5/image_processor.rs); the formula is `(px / 255 - 0.5) / 0.5`. mlx-vlm uses the same. Confirm by reading both — if they match, this branch is a dead lead.
- **Patchify**: layout is `[N, T, C, H, W]` post-Task 21 fix. Inspect: does each patch get the correct pixel coordinates from the resized image? Verify by hand for one patch.

- [ ] **Step 4.3: Re-run Task 3 driver and check the Gate**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3a_diff.sh 2>&1 | tail -5
```

Read the new report. If `max_diff < 0.05` → Gate 1 PASS, proceed to step 4.5. Otherwise, go back to 4.1 and iterate (consider the next candidate from the table).

- [ ] **Step 4.4: If Gate 1 stuck after 2 iterations, escalate**

If two cycles of fix attempts cannot close Gate 1 below 0.05:

- Document the residual divergence source in the report.
- Decide whether to relax the gate (write a new spec entry justifying e.g. `< 0.1`) OR escalate to Boss.

Do NOT silently widen the gate.

- [ ] **Step 4.5: Run full unit + integration test regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: 152 passed (P6.2 baseline + 0 new since image_processor unit tests are the same set).

Also run the P6.2 Task 21 logits-match to confirm production path is not regressed:

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --test p6_qwen35_vl_logits_match --release -- --ignored 2>&1 | tail -5
```

Expected: 1 passed.

- [ ] **Step 4.6: Commit Gate 1 green**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3a-* | head -1)
git add ironmlx/src/models/qwen3_5/image_processor.rs
git add -f "$REPORT_DIR/p6_3a_preprocess_report.md"
git commit -m "fix(p6.3a): preprocess byte diff — Gate 1 green"
```

---

## Task 5: P6.3a — Re-run full P6.1 pipeline to confirm Gate 2 trend

Even though Gate 2 is targeted by P6.3b, re-running the full pipeline after Gate 1 changes lets us confirm preprocess fixes did not regress vision-encoder numerics (the P6.1 pipeline uses mlx-vlm's pixel_values as the source of truth for both sides, so it's a sanity check on the encoder, not the preprocess).

- [ ] **Step 5.1: Re-run P6.1 pipeline**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh 2>&1 | tail -5
```

- [ ] **Step 5.2: Confirm `01_patch_embed_out` still bit-identical and `29_merger_out` is still ~0.90**

```bash
REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/2026-* | head -1)
grep -E "01_patch_embed_out|29_merger_out" "$REPORT/report.md"
```

Expected: `01_patch_embed_out` = `0.0000`, `29_merger_out` ≈ `0.90`.

If `01_patch_embed_out` regressed away from 0, the preprocess fix in Task 4 broke the C-major byte layout. Diagnose immediately before proceeding.

- [ ] **Step 5.3: No commit (verification only)**

---

## Task 6: P6.3b — Add op-level dump infrastructure (mlx-vlm side)

**Files:**
- Modify: `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py`
- Create/Modify: `ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/02_op_level_hooks.patch`

This adds 4 sub-op dump points inside each `Qwen3VLMoEVisionBlock.__call__` (so `4 × 24 = 96` new dumps per run, plus the existing 30, = 126 total). Hooks are no-op when env is unset.

- [ ] **Step 6.1: Read the existing block class**

```bash
grep -n "class Qwen3VLMoEVisionBlock\|def __call__" /Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py | head -10
sed -n '175,200p' /Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py
```

The class is around line 175. Its `__call__` is short (typically: norm1 → attn → residual → norm2 → mlp → residual). Note the exact variable names.

- [ ] **Step 6.2: Add 4 dump hooks inside the block**

The block-level dump in P6.1 (Task 5) wraps the block call site in `VisionModel.__call__` with `_maybe_dump(f"{5+i:02d}_block_{i:02d}_out", ...)`. For intra-block, we need to know the block index inside the block class — pass it via a thread_local or via a kwarg.

Cleanest: add an optional `name_prefix: str | None = None` kwarg to `Qwen3VLMoEVisionBlock.__call__`. Caller (VisionModel) passes `f"{5+i:02d}_block_{i:02d}"`; block uses it as the prefix for 4 intra-block dumps.

Modify `Qwen3VLMoEVisionBlock.__call__`:

```python
def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb, name_prefix=None) -> mx.array:
    normed1 = self.norm1(hidden_states)
    if name_prefix is not None:
        _maybe_dump(f"{name_prefix}_a_norm1_out", normed1)
    attn_out = self.attn(normed1, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    if name_prefix is not None:
        _maybe_dump(f"{name_prefix}_b_attn_residual", hidden_states + attn_out)
    h = hidden_states + attn_out
    normed2 = self.norm2(h)
    if name_prefix is not None:
        _maybe_dump(f"{name_prefix}_c_norm2_out", normed2)
    mlp_out = self.mlp(normed2)
    if name_prefix is not None:
        _maybe_dump(f"{name_prefix}_d_mlp_residual", h + mlp_out)
    return h + mlp_out
```

**Important**: preserve the existing block's exact return type and behavior when `name_prefix=None`. Verify by reading the existing block body first — match its variable names (it may call them `hidden_states`, `attn_out`, etc.).

- [ ] **Step 6.3: Update `VisionModel.__call__` to pass the prefix**

In `VisionModel.__call__`'s block loop (around line 405), update the call site:

```python
for layer_num, blk in enumerate(self.blocks):
    block_prefix = f"{5+layer_num:02d}_block_{layer_num:02d}"
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        name_prefix=block_prefix,
    )
    _maybe_dump(f"{block_prefix}_out", hidden_states)
    if layer_num in self.deepstack_visual_indexes:
        # ... existing deepstack code preserved ...
```

(Keep the existing `_maybe_dump(f"{block_prefix}_out", hidden_states)` from P6.1 — that's the block-output, P6.1 site.)

- [ ] **Step 6.4: Verify import still works with env unset**

```bash
unset MLXVLM_VISION_DUMP_DIR
~/.venvs/mlxvlm-ref/bin/python -c "from mlx_vlm.models.qwen3_vl.vision import VisionModel; print('import ok')"
```

- [ ] **Step 6.5: Verify gen_fixture.py still produces first_token=760**

```bash
unset MLXVLM_VISION_DUMP_DIR
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/gen_fixture.py 2>&1 | tail -3
```

Expected: `first_token_id = 760`. (Unchanged behavior when env is unset.)

- [ ] **Step 6.6: Smoke-run with env set to confirm all 126 files write**

```bash
mkdir -p /tmp/p6_diff/python_oplevel
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py \
    --image /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg \
    --out-dir /tmp/p6_diff/python_oplevel
ls /tmp/p6_diff/python_oplevel/*.safetensors | wc -l   # expect: 126
```

`run_python_dump.py` does NOT need changes — its expected-file check just lists 30 names. The script doesn't validate that NO MORE files exist; only that the expected 30 are present. So 126 total (30 P6.1 + 96 P6.3b) is fine.

- [ ] **Step 6.7: Archive the new patch in cxx-mlx**

```bash
cd /Volumes/Dev/mlx-vlm
git diff mlx_vlm/models/qwen3_vl/vision.py > /tmp/mlx_vlm_op_level_hooks.patch
cp /tmp/mlx_vlm_op_level_hooks.patch /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/02_op_level_hooks.patch
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/02_op_level_hooks.patch
git commit -m "chore(p6.3b): archive mlx-vlm op-level hook patch"
```

(The `02_op_level_hooks.patch` is CUMULATIVE — it includes both Task 5 of P6.1's hooks and this task's intra-block hooks. To regenerate the diff against pristine upstream, the engineer should reset mlx-vlm to its upstream commit before generating both patches sequentially. If that workflow gets noisy, accept the cumulative patch as-is for archival.)

---

## Task 7: P6.3b — Add op-level dump infrastructure (ironmlx side)

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/vision/block.rs`
- Modify: `ironmlx/src/models/qwen3_5/vision/mod.rs`

Mirror Task 6's mlx-vlm changes on the Rust side.

- [ ] **Step 7.1: Read the existing block forward**

```bash
grep -n "pub fn forward\b\|let normed1\|let attn_out\|let h\|let normed2\|let mlp_out" /Volumes/Dev/cxx-mlx/ironmlx/src/models/qwen3_5/vision/block.rs | head -15
```

Note the exact body of `VitBlock::forward`.

- [ ] **Step 7.2: Add `forward_with_name_prefix`**

In `ironmlx/src/models/qwen3_5/vision/block.rs`, add a new method on `VitBlock` next to `forward`:

```rust
    /// Like [`forward`] but emits 4 intra-block dumps (post-norm1, attn-residual,
    /// post-norm2, mlp-residual) under the given name prefix. The non-feature
    /// build erases the dump calls; in feature builds the dumps fire only if
    /// `IRONMLX_VISION_DUMP_DIR` is set. Used by the P6.3b op-level diff path.
    pub fn forward_with_name_prefix(
        &self,
        x: &mlx::Array,
        rotary_pos_emb: &mlx::Array,
        cu_seqlens: &[i32],
        name_prefix: &str,
    ) -> anyhow::Result<mlx::Array> {
        use crate::models::qwen3_5::vision::dump::dump_tensor;
        let normed1 = self.norm1.forward(x)?;
        dump_tensor(&format!("{name_prefix}_a_norm1_out"), &normed1);
        let attn_out = self.attn.forward(&normed1, rotary_pos_emb, cu_seqlens)?;
        let h = x + &attn_out;
        dump_tensor(&format!("{name_prefix}_b_attn_residual"), &h);
        let normed2 = self.norm2.forward(&h)?;
        dump_tensor(&format!("{name_prefix}_c_norm2_out"), &normed2);
        let mlp_out = self.mlp.forward(&normed2)?;
        let out = &h + &mlp_out;
        dump_tensor(&format!("{name_prefix}_d_mlp_residual"), &out);
        Ok(out)
    }
```

(Confirm the field/method names match the existing struct: `self.norm1`, `self.attn`, `self.norm2`, `self.mlp` should all exist per P6 Task 10. If a field is named differently, adapt.)

- [ ] **Step 7.3: Update `VisionTower::forward` to use the new method**

In `ironmlx/src/models/qwen3_5/vision/mod.rs`, find the block loop in `forward`:

```rust
for (i, blk) in self.blocks.iter().enumerate() {
    x = blk.forward(&x, &rotary, &cu_seqlens)?;
    dump_tensor(&format!("{:02}_block_{:02}_out", 5 + i, i), &x);
}
```

Replace with:

```rust
for (i, blk) in self.blocks.iter().enumerate() {
    let prefix = format!("{:02}_block_{:02}", 5 + i, i);
    x = blk.forward_with_name_prefix(&x, &rotary, &cu_seqlens, &prefix)?;
    dump_tensor(&format!("{prefix}_out"), &x);
}
```

- [ ] **Step 7.4: Verify default build (no dump) + feature build both compile**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release --features vision-dump 2>&1 | tail -3
```

Both must show `Finished`.

- [ ] **Step 7.5: Run vision unit tests to confirm no regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release vision -- --test-threads=1 2>&1 | tail -10
```

Expected: all passing.

- [ ] **Step 7.6: CI gauntlet**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --features vision-dump -- -D warnings 2>&1 | tail -3
```

All clean.

- [ ] **Step 7.7: Commit**

```bash
git add ironmlx/src/models/qwen3_5/vision/block.rs \
        ironmlx/src/models/qwen3_5/vision/mod.rs
git commit -m "feat(p6.3b): VitBlock::forward_with_name_prefix for op-level dumps"
```

---

## Task 8: P6.3b — Op-level driver + Gate 2 baseline measurement

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh`

The existing `run_p6_1_diff.sh` already drives the dump pipeline and feeds `diff_pipeline.py`. `diff_pipeline.py` pairs files by basename — it doesn't care if there are 29 or 125 of them. So this task is mostly a wrapper that calls `run_p6_1_diff.sh` and copies the report to a P6.3b-named directory for clarity.

- [ ] **Step 8.1: Write the P6.3b driver**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh`:

```bash
#!/usr/bin/env bash
# P6.3b: drive the full op-level diff (30 module-level + 96 intra-block = 126 tensors).
# Same pipeline as P6.1's run_p6_1_diff.sh, but report dir uses a p6_3b- prefix.
set -euo pipefail
if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
PY_DIR="${PY_DIR:-/tmp/p6_diff/python}"
RUST_DIR="${RUST_DIR:-/tmp/p6_diff/rust}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_3b-$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$RUST_DIR"/*.safetensors

echo "=== P6.3b: 126-tensor op-level diff ==="
echo "=== Step 1: mlx-vlm op-level dump (30 + 96 = 126 tensors) ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"
echo "  Files in $PY_DIR: $(ls "$PY_DIR"/*.safetensors | wc -l)"

echo "=== Step 2: ironmlx op-level dump (29 + 96 = 125 tensors) ==="
cd "$REPO_ROOT"
QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
    PIXEL_VALUES_PATH="$PY_DIR/00_pixel_values.safetensors" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test p6_vision_dump \
        -- --ignored 2>&1 | tail -5
echo "  Files in $RUST_DIR: $(ls "$RUST_DIR"/*.safetensors | wc -l)"

echo "=== Step 3: diff + report ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline.py" \
    --py "$PY_DIR" \
    --rust "$RUST_DIR" \
    --out "$REPORT_DIR"

echo "=== Done. Report: $REPORT_DIR/report.md ==="
```

```bash
chmod +x /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh
```

- [ ] **Step 8.2: Run it**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh 2>&1 | tail -10
```

Expected: ends with `=== Done. Report: <path>/report.md ===`.

- [ ] **Step 8.3: Inspect the report — locate the first intra-block rupture**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3b-* | head -1)
# Show the per-tensor table for blocks 14-17 specifically (the band where P6.2 saw the first jump)
grep -E "block_1[4-7]" "$REPORT_DIR/report.md"
```

Read the `max` column for each `block_NN_{a_norm1_out, b_attn_residual, c_norm2_out, d_mlp_residual, out}`. The first sub-op whose max_diff is >> 5× its predecessor is the **rupture point**.

- [ ] **Step 8.4: Commit driver + Gate 2 baseline**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3b-* | head -1)
git add ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh
git add -f "$REPORT_DIR/report.md" "$REPORT_DIR/max_diff_curve.png" "$REPORT_DIR/outliers.json"
git commit -m "feat(p6.3b): op-level driver run_p6_3b_diff.sh + Gate 2 baseline"
```

Record the baseline `29_merger_out max_diff` (should be ~0.90, same as P6.2) and the **identified rupture sub-op** (e.g. `block_15_b_attn_residual` if SDPA is the source). This goes in the close-out report.

---

## Task 9: P6.3b — Diagnose + fix the rupture sub-op

**Files:**
- Modify: one of `ironmlx/src/models/qwen3_5/vision/block.rs` or related (depends on which sub-op ruptures)

This is an iterative diagnose-fix-verify task driven by Task 8's findings. The fix file depends on **which sub-op** is the source.

- [ ] **Step 9.1: Match rupture to fix candidate**

| Rupture sub-op | Hypothesis | Fix target file | First thing to read |
| --- | --- | --- | --- |
| `block_NN_a_norm1_out` or `c_norm2_out` first jumps | LayerNorm accumulator precision | `ironmlx/src/nn/norm.rs` `LayerNorm::forward_on` | Check whether `mlx::fast::layer_norm_on` is called with same `eps` and `weight`/`bias` types as mlx-vlm `nn.LayerNorm` |
| `block_NN_b_attn_residual` first jumps | SDPA or rotary | `ironmlx/src/models/qwen3_5/vision/block.rs` `VitAttention::forward` | Look at rank-4 expand_dims/squeeze wrapping; compare to mlx-vlm `vision.py:135-160` Attention block |
| `block_NN_d_mlp_residual` first jumps | GELU implementation | `ironmlx/src/models/qwen3_5/vision/block.rs` `VitMLP::forward` | Compare hand-rolled GELU tanh approx against mlx-vlm `nn.GELU(approximate="tanh")` |
| Multiple sites jump together | float reduction order in matmul | escalate: probably needs lower-level mlx kernel investigation | n/a |

- [ ] **Step 9.2: Inspect the matched implementation against mlx-vlm reference**

Open `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` and the matched ironmlx file side-by-side. The check pattern depends on the rupture:

- **For attention**: walk through `VitAttention::forward` op-by-op against mlx-vlm's `Attention.__call__`. Pay attention to ordering: does rotary apply to (Q, K) before or after head split? Does the SDPA take rank-3 or rank-4? Are the transposes the same?
- **For LayerNorm**: ironmlx uses `mlx::fast::layer_norm_on` — that's the same kernel mlx-vlm uses. The likely difference is in `weight`/`bias` dtype passed in. Check if `LayerNorm::weight` is bf16 (matching disk) or upcast.
- **For GELU**: ironmlx uses a hand-rolled tanh approximation (Task 8 of P6); mlx-vlm uses `nn.GELU(approximate="tanh")`. Compare the formulas. The mlx-vlm path internally also computes `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))` — verify the constants and the multiplication order in ironmlx match.

- [ ] **Step 9.3: Apply the targeted fix**

Make the smallest change that addresses the difference identified in 9.2. Always preserve `cargo fmt` + `cargo clippy` cleanliness as you edit.

- [ ] **Step 9.4: Verify build + run regression unit tests**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: build OK, 152 passed.

- [ ] **Step 9.5: Re-run P6.3b pipeline + check Gate 2**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_3b_diff.sh 2>&1 | tail -5
REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3b-* | head -1)
grep "29_merger_out" "$REPORT/report.md"
```

If `29_merger_out max_diff < 0.1` → Gate 2 PASS. Proceed to Task 10. Otherwise, re-run 9.1 with the new rupture point (the divergence may have moved to a downstream block).

- [ ] **Step 9.6: If Gate 2 stuck after 3 iterations, escalate**

Document residual rupture point and what was tried. Decide whether to relax the gate (spec amendment + re-brainstorm) or accept the partial improvement and proceed to Task 10.

Do NOT silently widen the gate.

- [ ] **Step 9.7: Commit each fix immediately**

After each gate-improving fix in 9.3 + verifying 9.5, commit:

```bash
git add <changed files>
git commit -m "fix(p6.3b): align <sub-op name> with mlx-vlm reference"
```

So the close-out report can list the exact set of fixes applied.

- [ ] **Step 9.8: Final commit of Gate 2 green report**

When Gate 2 passes:

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3b-* | head -1)
git add -f "$REPORT_DIR/report.md" "$REPORT_DIR/max_diff_curve.png" "$REPORT_DIR/outliers.json"
git commit -m "docs(p6.3b): Gate 2 green — op-level vision encoder diff < 0.1"
```

---

## Task 10: P6.3b — Verify Task 21 still passes (Gate 3A regression check)

After Gate 2 fixes touched `vision/*.rs` or `nn/norm.rs`, the LM-only path is unchanged but the vision encoder is now closer to mlx-vlm. Task 21's `p6_qwen35_vl_logits_match` test should still pass; in fact it should improve.

- [ ] **Step 10.1: Run the Task 21 test**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --test p6_qwen35_vl_logits_match --release -- --ignored 2>&1 | tail -10
```

Expected: PASS. Read the eprintln output line — the printed `max_diff` should be lower than the P6.2 baseline of `0.5039`. If it's actually closer to or below 0.3, Gate 3A is well within bounds. If it's `> 0.5`, Gate 3A is in danger and needs investigation.

- [ ] **Step 10.2: Record the new Gate 3 max_diff**

This number goes in the close-out table. No commit needed (test is run from the working tree).

---

## Task 11: P6.3c — Semantic verification driver

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.py`
- Modify: `ironmlx/tests/fixtures/p6_qwen35_vl/.gitignore` (add `item3_check_outputs/`)

This script automates Gate 4: starts ironmlx server, hits it with 4 images at `temperature=0` + `enable_thinking=false`, parses responses, applies per-image pass criteria.

- [ ] **Step 11.1: Verify the 4 test images are present**

```bash
ls -lh /tmp/p6vl_test_imgs/{scene,counting,text}.jpg 2>&1
ls -lh /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg
```

If `/tmp/p6vl_test_imgs/*.jpg` are missing, re-download:

```bash
mkdir -p /tmp/p6vl_test_imgs
cd /tmp/p6vl_test_imgs
curl -kLso scene.jpg "https://images.cocodataset.org/val2017/000000000139.jpg"
curl -kLso counting.jpg "https://images.cocodataset.org/val2017/000000001000.jpg"
curl -kLso text.jpg "https://images.cocodataset.org/val2017/000000000724.jpg"
file *.jpg
```

- [ ] **Step 11.2: Write the semantic check script**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.py`:

```python
#!/usr/bin/env python
"""P6.3c Gate 4: semantic functional correctness on 4 test images.

Starts an ironmlx HTTP server, queries it with each image at temperature=0
and enable_thinking=false, applies per-image pass criteria from the P6.3 spec.

Usage:
    MLX_DIR=$HOME/.local/mlx \
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
    ~/.venvs/mlxvlm-ref/bin/python item3_semantic_check.py \
        --out /path/to/p6_3c_semantic_report.md
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "ironmlx/tests/fixtures/p6_qwen35_vl"
PROMPT = "Describe this image in detail. If there are multiple people or objects, count them."

IMAGES = [
    {
        "name": "coco_cats",
        "path": FIXTURE_DIR / "coco_sample.jpg",
        "criteria": {
            "type": "key_facts",
            # Each key fact is a list of acceptable synonyms (case-insensitive substring match)
            "facts": [
                ["two cats", "2 cats", "two tabby", "two kittens"],
                ["green collar", "collar"],
                ["remote", "remotes"],
            ],
            "min_pass": 2,
        },
    },
    {
        "name": "scene_room",
        "path": Path("/tmp/p6vl_test_imgs/scene.jpg"),
        "criteria": {
            "type": "forbid_keywords",
            "forbid": ["side-by-side", "side by side", "stereoscopic", "composite",
                       "duplicated", "duplicate", "mirrored", "mirror image", "stitched"],
        },
    },
    {
        "name": "counting_kids",
        "path": Path("/tmp/p6vl_test_imgs/counting.jpg"),
        "criteria": {
            "type": "count_in_range",
            # Extract integers from response; pass if any integer is in [10, 16]
            "range": [10, 16],
        },
    },
    {
        "name": "text_stop",
        "path": Path("/tmp/p6vl_test_imgs/text.jpg"),
        "criteria": {
            "type": "inversion_keyword",
            "keywords": ["upside down", "upside-down", "rotated 180",
                         "rotated by 180", "POTS", "flipped"],
        },
    },
]


def wait_for_port(port: int, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def query(port: int, image_path: Path) -> tuple[str, str]:
    """Returns (response_text, finish_reason). Raises on HTTP error."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": "qwen3_5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 400,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], body["choices"][0]["finish_reason"]


def evaluate(text: str, criteria: dict) -> tuple[bool, str]:
    """Returns (passed, note)."""
    t = text.lower()
    if criteria["type"] == "key_facts":
        hits = 0
        details = []
        for fact_synonyms in criteria["facts"]:
            matched = next((s for s in fact_synonyms if s.lower() in t), None)
            if matched is not None:
                hits += 1
                details.append(f"✓ {matched}")
            else:
                details.append(f"✗ {fact_synonyms[0]}")
        return hits >= criteria["min_pass"], f"{hits}/{len(criteria['facts'])} ({'; '.join(details)})"
    if criteria["type"] == "forbid_keywords":
        for fk in criteria["forbid"]:
            if fk.lower() in t:
                return False, f"contains forbidden term '{fk}'"
        return True, "no forbidden keywords"
    if criteria["type"] == "count_in_range":
        nums = [int(n) for n in re.findall(r"\b(\d+)\b", text)]
        lo, hi = criteria["range"]
        ok_nums = [n for n in nums if lo <= n <= hi]
        return bool(ok_nums), f"numbers in response: {nums}; in [{lo},{hi}]: {ok_nums}"
    if criteria["type"] == "inversion_keyword":
        for kw in criteria["keywords"]:
            if kw.lower() in t:
                return True, f"matched '{kw}'"
        return False, "no inversion keyword"
    raise ValueError(f"unknown criteria type {criteria['type']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_dir = os.environ.get("QWEN35_MODEL")
    mlx_dir = os.environ.get("MLX_DIR")
    if not model_dir or not mlx_dir:
        print("ERROR: set MLX_DIR and QWEN35_MODEL", file=sys.stderr)
        return 1

    # Kill any leftover server first
    subprocess.run(["pkill", "-KILL", "-f", "ironmlx serve"], check=False)
    time.sleep(2)

    # Start ironmlx server
    server_log = open("/tmp/p6_3c_server.log", "w")
    env = dict(os.environ)
    env["MLX_DIR"] = mlx_dir
    server = subprocess.Popen(
        [str(REPO_ROOT / "target/release/ironmlx"), "serve",
         "--model", model_dir,
         "--host", "127.0.0.1",
         "--port", str(args.port)],
        env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_port(args.port, timeout_s=120):
            print("ERROR: server failed to start; see /tmp/p6_3c_server.log", file=sys.stderr)
            return 2

        results = []
        for spec in IMAGES:
            print(f"[item3] querying {spec['name']}...")
            text, finish = query(args.port, spec["path"])
            passed, note = evaluate(text, spec["criteria"])
            results.append({
                "name": spec["name"],
                "criteria_type": spec["criteria"]["type"],
                "passed": passed,
                "note": note,
                "finish_reason": finish,
                "response": text,
            })
            print(f"  → {'PASS' if passed else 'FAIL'}: {note}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()

    # Write report
    n_pass = sum(1 for r in results if r["passed"])
    lines = [
        "# P6.3c Semantic Verification (Gate 4)",
        "",
        f"- Images tested: {len(results)}",
        f"- Passed: {n_pass} / {len(results)}",
        f"- Gate 4 threshold: ≥ 3 / 4 → **{'PASS' if n_pass >= 3 else 'FAIL'}**",
        "",
        "## Per-image",
        "",
    ]
    for r in results:
        lines.append(f"### {r['name']} — {'✅ PASS' if r['passed'] else '❌ FAIL'}")
        lines.append("")
        lines.append(f"- criterion: `{r['criteria_type']}`")
        lines.append(f"- finish_reason: `{r['finish_reason']}`")
        lines.append(f"- note: {r['note']}")
        lines.append("")
        lines.append("Response:")
        lines.append("```")
        lines.append(r["response"])
        lines.append("```")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"[item3] report → {args.out}")
    return 0 if n_pass >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 11.3: Update .gitignore**

Add to `ironmlx/tests/fixtures/p6_qwen35_vl/.gitignore`:

```
# P6.3c output artifacts (server log, transient)
/tmp/p6_3c_server.log
```

(The Python script writes to `/tmp` so we don't actually need a gitignore here; this is a documentation marker. If you prefer, skip the .gitignore change.)

- [ ] **Step 11.4: Verify `requests` is installed in the venv**

```bash
~/.venvs/mlxvlm-ref/bin/python -c "import requests; print(requests.__version__)"
```

If missing: `~/.venvs/mlxvlm-ref/bin/pip install requests`.

- [ ] **Step 11.5: Commit the script**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.py
git commit -m "feat(p6.3c): item3_semantic_check.py — Gate 4 driver"
```

---

## Task 12: P6.3c — Run Gate 4 + record results

- [ ] **Step 12.1: Build a fresh release binary**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
```

Expected: `Finished`.

- [ ] **Step 12.2: Run the semantic check**

```bash
STAMP=$(date +%Y-%m-%d-%H%M)
REPORT_DIR=/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3c-$STAMP
mkdir -p "$REPORT_DIR"
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.py \
    --out "$REPORT_DIR/p6_3c_semantic_report.md" 2>&1 | tail -20
```

Expected last lines: `PASS` for ≥ 3 out of 4 images. Final line: `[item3] report → <path>`.

- [ ] **Step 12.3: Inspect report**

```bash
head -50 "$REPORT_DIR/p6_3c_semantic_report.md"
```

Confirm the verdict line.

- [ ] **Step 12.4: Commit Gate 4 report**

```bash
git add -f "$REPORT_DIR/p6_3c_semantic_report.md"
git commit -m "docs(p6.3c): Gate 4 semantic verification report"
```

---

## Task 13: P6.3 — Close-out report + final acceptance table

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md`

- [ ] **Step 13.1: Gather final numbers**

Collect into one place:

- Gate 1 (preprocess max_diff): from the latest `p6_3a-*` report
- Gate 2 (29_merger_out max_diff): from the latest `p6_3b-*` report
- Gate 3A (E2E logits max_diff): from Task 10 step 10.2's eprintln output
- Gate 3B (greedy first-token): "PASS (760)" if Task 10 step 10.1 passed
- Gate 4 (semantic pass count): from the latest `p6_3c-*` report

- [ ] **Step 13.2: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md`:

```markdown
# P6.3 Vision Functional Correctness — Close-out

**Branch:** `ironmlx-p6-3-vision-correctness`
**Date:** <YYYY-MM-DD>
**Spec:** `docs/superpowers/specs/2026-05-11-p6-3-functional-correctness-design.md`

## Acceptance table

| Gate | Target | Before P6.3 | After P6.3 | Status |
| --- | --- | --- | --- | --- |
| 1. Preprocess max_diff | < 0.05 | <baseline> | <observed> | <PASS/FAIL> |
| 2. Vision encoder max_diff | < 0.1 | 0.9023 | <observed> | <PASS/FAIL> |
| 3A. E2E logits max_diff | < 0.5 | 0.5039 | <observed> | <PASS/FAIL> |
| 3B. Greedy first-token | bit-identical | 760 ✅ | <observed> | <PASS/FAIL> |
| 4a. COCO key facts | ≥ 2/3 | 1/3 (a cat) | <observed> | <PASS/FAIL> |
| 4b. scene non-double | yes | no | <observed> | <PASS/FAIL> |
| 4c. counting ±2 | in [10, 16] | 20-25 | <observed> | <PASS/FAIL> |
| 4d. STOP inversion ≥1 keyword | yes | no | <observed> | <PASS/FAIL> |

## Fixes applied (chronological)

- <commit SHA> <message> (e.g. fix(p6.3a): preprocess byte diff — Gate 1 green)
- <commit SHA> <message> (e.g. fix(p6.3b): align <sub-op> with mlx-vlm)
- ...

## Linked reports

- Preprocess diff: `diff_reports/p6_3a-<stamp>/p6_3a_preprocess_report.md`
- Op-level vision encoder diff: `diff_reports/p6_3b-<stamp>/report.md`
- Semantic verification: `diff_reports/p6_3c-<stamp>/p6_3c_semantic_report.md`

## Notes

<Free-form notes: what was hardest, any residual concerns, P6.4 candidate items.>
```

Fill in all `<...>` placeholders with actual values from Task 13.1's gathered data.

- [ ] **Step 13.3: Commit close-out**

```bash
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md
git commit -m "docs(p6.3): close-out report — acceptance table + fixes summary"
```

- [ ] **Step 13.4: Final regression check**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: build OK, 152 passed (no regression).

```bash
git log --oneline ironmlx-p6-1-vision-diff..HEAD
```

Should list all commits made during P6.3 in chronological order.

---

## Self-Review

**Spec coverage check** (each requirement in P6.3 spec mapped to a task):

- Spec §2 Gate 1 (preprocess) → Tasks 2-4 (build dumper + diff tool + iterate fix)
- Spec §2 Gate 2 (vision encoder) → Tasks 6-9 (mlx-vlm hooks + ironmlx hooks + driver + iterate fix)
- Spec §2 Gate 3A/B (E2E logits + first-token) → Task 10 (re-run Task 21 logits-match)
- Spec §2 Gate 4 (semantic) → Tasks 11-12 (driver + run)
- Spec §3 Acceptance Report → Task 13 (close-out)
- Spec §4 Approach sequencing → Tasks 1-13 are linearly ordered: 3a (Tasks 2-5) → 3b (Tasks 6-10) → 3c (Tasks 11-13)
- Spec §7 Rollback → Each fix is its own commit (Task 9.7 + Task 4.6 + Task 13's "Fixes applied" section), so any individual fix can be reverted.
- Spec §9 Files anticipated → All listed in "File Structure" header and each touched by an explicit task.

**Placeholder scan**: 

- No "TBD" / "TODO" / "fill in later" left in steps.
- Tasks 4 and 9 have decision-tree-style guidance because the fix code depends on what the diagnostic reveals — that is intrinsic to a diagnose-fix loop, not a placeholder. Each branch of the decision tree references concrete files/functions.
- Task 13.2 has `<...>` placeholders in the close-out template — those are intended to be filled by the engineer with actual measured numbers. That's the artifact of the report, not a plan placeholder.

**Type/path consistency**:

- `dump_tensor(name: &str, t: &Array)` (Tasks 7) matches the signature from P6.1 Task 2.
- Dump filename convention `NN_block_NN_X_subop` (Tasks 6-7) is consistent across mlx-vlm and ironmlx sides.
- `forward_with_name_prefix(&self, x, rotary_pos_emb, cu_seqlens, name_prefix: &str) -> Result<Array>` (Task 7.2) referenced by VisionTower in Task 7.3 — same signature.
- `Loader::open_multimodal(Path)` and `QWEN35_MODEL` env are consistent with P6.1/P6.2.
- The semantic check script's port `8082` does not collide with the P6 Task 22 default of `8081`, but that's incidental — neither is currently bound at task-execution time.

No type drift or path inconsistencies found.

---

## Plan complete and saved to `docs/superpowers/plans/2026-05-11-p6-3-vision-correctness.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task + two-stage review. Best fit because Tasks 2, 3, 6, 7, 11 are largely independent scaffolding tasks (one file, clear spec). Tasks 4 and 9 are iterative diagnose-fix loops that benefit from clean per-iteration context.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Better if you want to inspect each diff report as it's produced and steer the fix decisions in real time.

Which approach?
