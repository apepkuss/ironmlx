# P6.6 Multi-Image-Per-Request Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify ironmlx VL inference matches mlx-vlm reference on a 1-chat-request-with-2-images workload, and fix anything that surfaces — closing 4 acceptance gates (preprocess byte-diff per image, vision encoder concat-output diff, end-to-end logits diff + greedy first-token, 2-image semantic correctness).

**Architecture:** Three sub-phases — P6.6a (build a 6-tool diagnostic pipeline mirroring P6.1/P6.3 layout but for N=2 images, run baseline, set thresholds), P6.6b (per-finding diagnose-fix loop), P6.6c (semantic verification + close-out). All P6.6 tools are NEW files; P6.1/P6.3 single-image tools stay untouched.

**Tech Stack:** Rust + ironmlx + Python (mlx-vlm venv) + safetensors + matplotlib + bash. Builds on P6.5 cleanup branch (`spatial_merge_size` + `image_token_id` already plumbed; `image_grid_thw: Vec` and `pixel_values: concat`'d structures in place).

**Spec:** `docs/superpowers/specs/2026-05-11-p6-6-multi-image-design.md` (commit `ee3aef7`)

**Branch base:** `ironmlx-p6-4-cleanup` (last commit `fcde351`)
**Branch target:** `ironmlx-p6-6-multi-image` (already cut, head `ee3aef7`)

---

## File Structure

Created by this plan (all NEW):

```
ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/
├── image_0.jpg                    # COCO val2017 download
├── image_1.jpg                    # COCO val2017 download
└── .gitignore                     # ignore expected_*.npy/safetensors
ironmlx/tests/fixtures/p6_qwen35_vl/
├── run_p6_6_dump.py               # mlx-vlm 2-image dump driver
├── run_p6_6_diff.sh               # top-level orchestrator
├── diff_preprocess_multi.py       # per-image preprocess diff (Gate 1)
├── diff_pipeline_multi.py         # multi-image op-level diff (Gate 2)
└── p6_6_semantic_check.py         # Gate 4 driver
ironmlx/tests/
├── p6_6_multi_image_dump.rs       # ironmlx-side dump (feature-gated)
└── p6_6_logits_match.rs           # e2e Gate 3 integration test
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
└── p6_6_closeout/report.md        # final acceptance table
```

May be modified if Gates surface bugs:
- `ironmlx/src/models/qwen3_5/vision/mod.rs` (multi-grid path)
- `ironmlx/src/models/qwen3_5/cross_modal.rs` (multi-span scatter)

mlx-vlm fork — **no new changes**; the 30-point + 96-intra-block hooks from P6.1 + P6.3b are reused.

---

## Task 1: Branch sanity + image download

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/.gitignore`

- [ ] **Step 1.1: Confirm branch state**

```bash
cd /Volumes/Dev/cxx-mlx
git branch --show-current  # expect: ironmlx-p6-6-multi-image
git log -1 --oneline       # expect: ee3aef7 docs(p6.6): ...
```

- [ ] **Step 1.2: Pick + download 2 COCO val2017 images with deliberately different content**

Two semantically distinct test images so the 2-image semantic gate has unambiguous keys:

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image
cd /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image
# image_0: dog on couch — COCO val2017 ID 000000397133
curl -kLso image_0.jpg "https://images.cocodataset.org/val2017/000000397133.jpg"
# image_1: tennis player on court — COCO val2017 ID 000000252219
curl -kLso image_1.jpg "https://images.cocodataset.org/val2017/000000252219.jpg"
file image_0.jpg image_1.jpg
```

Expected: both `JPEG image data, JFIF standard 1.01, ... baseline, precision 8, ...`. If either is HTML/XML (404 page), pick a different val2017 id and retry. Both must be ≥ 50 KB.

- [ ] **Step 1.3: Sanity-check images visually**

These will be the ground truth for the semantic check, so verify they show the topics:

```bash
~/.venvs/mlxvlm-ref/bin/python -c "
from PIL import Image
for p in ['image_0.jpg', 'image_1.jpg']:
    im = Image.open(p)
    print(p, im.size, im.mode)
"
```

Open both in the IDE if visual confirmation needed. Topic must be clearly identifiable (e.g. image_0 is a dog scene, image_1 is a tennis scene). If a chosen image is ambiguous, replace with a different val2017 id.

- [ ] **Step 1.4: Write `.gitignore`**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/.gitignore`:

```
# P6.6 generated artifacts — regenerate via run_p6_6_dump.py
expected_*.npy
expected_*.safetensors
image_*_pv*.safetensors
vision_embeds.safetensors
expected_first_token.txt
```

- [ ] **Step 1.5: Commit images + .gitignore**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/
git commit -m "feat(p6.6): add 2-image fixture (COCO val2017 dog + tennis)"
```

---

## Task 2: mlx-vlm side multi-image dump driver

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_dump.py`

This script is a multi-image extension of `run_python_dump.py` (P6.1 Task 6). It:
1. Loads model + processor via `mlx_vlm.load`.
2. Builds a 1-chat-request-with-2-images prompt + image inputs via `prepare_inputs`.
3. Saves `input_ids`, per-image `pixel_values` (`image_0_pv.safetensors`, `image_1_pv.safetensors`), concatenated `pixel_values`, `image_grid_thw` (shape `[2, 3]`), full vision-tower output (`vision_embeds.safetensors`), last-position logits, expected first token.
4. Optionally activates the P6.1 + P6.3b vision dump hooks if `MLXVLM_VISION_DUMP_DIR` env is set (no source change needed — hooks already in fork from P6.1 Task 5 + P6.3b Task 6).

- [ ] **Step 2.1: Verify the mlx-vlm fork still has P6.1+P6.3b hooks**

```bash
unset MLXVLM_VISION_DUMP_DIR
~/.venvs/mlxvlm-ref/bin/python -c "
from mlx_vlm.models.qwen3_vl.vision import VisionModel, _maybe_dump
print('hooks present')
"
```

Expected: `hooks present`. If `ImportError`, the mlx-vlm fork has been reset — re-apply patches from `ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/`.

- [ ] **Step 2.2: Write `run_p6_6_dump.py`**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_dump.py`:

```python
#!/usr/bin/env python
"""P6.6 mlx-vlm-side multi-image dump driver.

Loads model + processor, builds a 1-chat-request prompt that includes
TWO image_url parts, runs preprocess to get per-image pixel_values +
image_grid_thw, runs the vision tower once to capture the concatenated
vision_embeds, then runs the full LM forward to capture last-position
logits. All outputs are saved as safetensors / .npy under --out-dir.

Optionally, when MLXVLM_VISION_DUMP_DIR is set externally, the existing
P6.1+P6.3b vision hooks (30 + 96 = 126 intermediate tensors) fire and
write into that dir for op-level Gate 2 diff.

Usage:
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
    ~/.venvs/mlxvlm-ref/bin/python run_p6_6_dump.py \
        --image-0 /path/to/image_0.jpg \
        --image-1 /path/to/image_1.jpg \
        --out-dir /tmp/p6_diff_multi/python
"""
import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.utils import prepare_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-0", required=True, type=Path)
    parser.add_argument("--image-1", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="Describe both images in detail. Mention key objects you see in each.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = os.environ.get("QWEN35_MODEL")
    if not model_dir:
        print("ERROR: QWEN35_MODEL env required", file=sys.stderr)
        return 1
    for p in (args.image_0, args.image_1):
        if not p.exists():
            print(f"ERROR: image not found: {p}", file=sys.stderr)
            return 1

    model, processor = load(model_dir)
    config = model.config
    image_token_id = config.image_token_id

    # mlx-vlm prepare_inputs supports multi-image: pass list of [paths]
    inputs = prepare_inputs(
        processor=processor,
        prompts=[args.prompt],
        images=[[str(args.image_0), str(args.image_1)]],
        videos=None,
        image_token=processor.tokenizer.decode([image_token_id]),
        video_token=None,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    # Sanity: should have 2 entries in grid_thw
    assert grid_thw.shape[0] == 2, f"expected 2 images, grid_thw shape={grid_thw.shape}"

    # Save input_ids, image_grid_thw
    np.save(args.out_dir / "expected_input_ids.npy", np.array(input_ids.astype(mx.int32)))
    np.save(args.out_dir / "expected_image_grid_thw.npy", np.array(grid_thw.astype(mx.int32)))

    # Save concatenated pixel_values (what ironmlx's GenerateRequest also stores)
    mx.eval(pixel_values)
    mx.save_safetensors(
        str(args.out_dir / "expected_pixel_values.safetensors"),
        {"tensor": pixel_values.astype(mx.bfloat16)},
    )

    # Save per-image pixel_values slices for Gate 1 (per-image preprocess diff)
    # pixel_values is [N_total, 1536] flat; split by N_i = grid_h_i * grid_w_i (no
    # spatial_merge_size factor — pixel_values is the pre-merger patch grid)
    grid_np = np.array(grid_thw.astype(mx.int32))
    n0 = int(grid_np[0, 1] * grid_np[0, 2])
    n1 = int(grid_np[1, 1] * grid_np[1, 2])
    assert n0 + n1 == pixel_values.shape[0], \
        f"split mismatch: n0={n0}, n1={n1}, total={pixel_values.shape[0]}"
    pv_0 = pixel_values[:n0]
    pv_1 = pixel_values[n0:]
    mx.save_safetensors(
        str(args.out_dir / "image_0_pv.safetensors"),
        {"tensor": pv_0.astype(mx.bfloat16)},
    )
    mx.save_safetensors(
        str(args.out_dir / "image_1_pv.safetensors"),
        {"tensor": pv_1.astype(mx.bfloat16)},
    )

    # Forward vision tower (with optional op-level dumps if env is set externally)
    pv_bf16 = pixel_values.astype(mx.bfloat16)
    embeds, _deepstack = model.vision_tower(pv_bf16, grid_thw)
    mx.eval(embeds)
    mx.save_safetensors(
        str(args.out_dir / "vision_embeds.safetensors"),
        {"tensor": embeds.astype(mx.bfloat16)},
    )

    # Full LM forward to get last-position logits + first token
    # Use model.language_model with the pre-computed embeddings substitution
    # via cross-modal token replacement (mlx-vlm internal helper).
    # Simplest path: call model.__call__ which orchestrates the whole thing.
    logits = model(input_ids, pixel_values=pv_bf16, image_grid_thw=grid_thw)
    mx.eval(logits)
    last = logits[:, -1, :]
    np.save(args.out_dir / "expected_last_logits.npy",
            np.array(last.astype(mx.float32)))
    first_token = int(mx.argmax(last, axis=-1).item())
    (args.out_dir / "expected_first_token.txt").write_text(str(first_token) + "\n")

    print(f"[run_p6_6_dump] input_ids shape: {input_ids.shape}")
    print(f"[run_p6_6_dump] pixel_values shape: {pixel_values.shape}")
    print(f"[run_p6_6_dump] grid_thw: {grid_np.tolist()}")
    print(f"[run_p6_6_dump] vision_embeds shape: {embeds.shape}")
    print(f"[run_p6_6_dump] last_logits shape: {last.shape}")
    print(f"[run_p6_6_dump] first_token: {first_token}")
    print(f"[run_p6_6_dump] all artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.3: Run it end-to-end to confirm working**

```bash
mkdir -p /tmp/p6_diff_multi/python
rm -f /tmp/p6_diff_multi/python/*
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_dump.py \
    --image-0 /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg \
    --image-1 /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg \
    --out-dir /tmp/p6_diff_multi/python 2>&1 | tail -10
ls /tmp/p6_diff_multi/python/
```

Expected: 6 artifact files (`expected_input_ids.npy`, `expected_image_grid_thw.npy`, `expected_pixel_values.safetensors`, `image_0_pv.safetensors`, `image_1_pv.safetensors`, `vision_embeds.safetensors`, `expected_last_logits.npy`, `expected_first_token.txt`) and stdout shows the printed shapes + a `first_token` integer.

If `prepare_inputs` errors on `images=[[a, b]]`, check the actual mlx-vlm 0.5.0 API — likely needs flat `images=[a, b]` and `prompts` adjusts. Adapt the call signature accordingly.

- [ ] **Step 2.4: Commit driver**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_dump.py
git commit -m "feat(p6.6): mlx-vlm 2-image dump driver run_p6_6_dump.py"
```

---

## Task 3: ironmlx-side multi-image dump integration test

**Files:**
- Create: `ironmlx/tests/p6_6_multi_image_dump.rs`

Mirrors `p6_vision_dump.rs` (P6.1 Task 4) but for 2 images. Reads `expected_pixel_values.safetensors` + `expected_image_grid_thw.npy` from the P6.6a mlx-vlm dump, drives `VisionTower::forward`, dumps the resulting `vision_embeds` as a safetensors plus relies on the existing `dump_tensor` hooks (29 module-level + 96 intra-block) for op-level diff.

- [ ] **Step 3.1: Write the integration test**

Create `ironmlx/tests/p6_6_multi_image_dump.rs`:

```rust
//! P6.6 multi-image vision dump integration test.
//!
//! Driven by `IRONMLX_VISION_DUMP_DIR`, `QWEN35_MODEL`, `PIXEL_VALUES_PATH`,
//! `IMAGE_GRID_THW_PATH` env vars set by `run_p6_6_diff.sh`. Reads the
//! mlx-vlm-prepared `expected_pixel_values.safetensors` (concatenated 2-image
//! patches in C-major [N_total, 1536] layout) and `expected_image_grid_thw.npy`
//! (shape `[2, 3]`), drives ONE `VisionTower::forward` over the concatenated
//! input, and dumps the output as `vision_embeds.safetensors` in the dump dir.

#![cfg(feature = "vision-dump")]

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::vision::VisionTower;
use ironmlx::models::qwen3_5::Qwen35Config;

#[test]
#[ignore] // requires QWEN35_MODEL + PIXEL_VALUES_PATH + IMAGE_GRID_THW_PATH + IRONMLX_VISION_DUMP_DIR
fn p6_6_multi_image_dump() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let pv_path = std::env::var("PIXEL_VALUES_PATH").expect("PIXEL_VALUES_PATH env required");
    let grid_path = std::env::var("IMAGE_GRID_THW_PATH").expect("IMAGE_GRID_THW_PATH env required");
    let dump_dir = std::env::var("IRONMLX_VISION_DUMP_DIR")
        .expect("IRONMLX_VISION_DUMP_DIR env required");

    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let vc = cfg.vision_config.expect("vision_config");
    let tower = VisionTower::from_loader(&loader, &vc).expect("tower");

    // Load mlx-vlm's [N_total, 1536] pixel_values (C-major) and reshape to
    // [N_total, T, C, H, W] (same conversion as Task 21 / P6.2 logits_match).
    let (mut loaded, _meta) = mlx::io::load_safetensors(&pv_path).expect("load pv");
    let pv_flat: Array = loaded.remove("tensor").expect("tensor key in pv");
    let n_total = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat
        .reshape(&[n_total, 3, 2, 16, 16][..])
        .expect("reshape pv");
    let pv = mlx::ops::shape::transpose_axes(&pv_5d, &[0, 2, 1, 3, 4][..]).expect("transpose pv");

    // Load grid_thw from .npy. Shape [2, 3].
    // mlx.load supports .npy directly.
    let grid_arr = mlx::io::load(&grid_path).expect("load grid_thw npy");
    let grid_i32 = mlx::ops::cast::astype(&grid_arr, Dtype::Int32).expect("cast grid to i32");
    let grids_flat: Vec<i32> = grid_i32.to_vec::<i32>().expect("grid to_vec");
    let grids: Vec<(i32, i32, i32)> = grids_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    assert_eq!(grids.len(), 2, "P6.6 expects exactly 2 images");

    eprintln!(
        "[p6_6_multi_image_dump] pv.shape={:?} grids={:?}",
        pv.shape().as_slice(),
        grids
    );

    // Run vision tower forward. Op-level dumps fire via the dump_tensor hooks
    // inserted at 29 module + 96 intra-block sites (P6.1 + P6.3b).
    let embeds = tower.forward(&pv, &grids).expect("vision forward");
    mlx::transforms::eval(&[&embeds]).expect("eval embeds");

    // Save final vision_embeds for Gate 2 (concatenated 2-image merger output)
    let path = format!("{dump_dir}/vision_embeds.safetensors");
    let mut map = std::collections::HashMap::new();
    let embeds_bf16 = mlx::ops::cast::astype(&embeds, Dtype::Bfloat16).expect("cast embeds");
    map.insert("tensor".to_string(), embeds_bf16);
    let metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    mlx::io::save_safetensors(&path, &map, &metadata).expect("save embeds");

    eprintln!(
        "[p6_6_multi_image_dump] vision_embeds.shape={:?} saved to {}",
        embeds.shape().as_slice(),
        dump_dir
    );
}
```

- [ ] **Step 3.2: Smoke-test compile (no run yet)**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --tests --features vision-dump 2>&1 | tail -5
```

Expected: `Finished` line; no compile errors. Adapt imports if `mlx::io::load` is named differently — verify with `grep "pub fn load" /Volumes/Dev/cxx-mlx/mlx/src/io.rs | head -5`.

- [ ] **Step 3.3: CI gauntlet for the new test file only**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --features vision-dump --test p6_6_multi_image_dump -- -D warnings 2>&1 | tail -3
```

Both clean.

- [ ] **Step 3.4: Commit**

```bash
git add ironmlx/tests/p6_6_multi_image_dump.rs
git commit -m "feat(p6.6): p6_6_multi_image_dump ironmlx integration test"
```

---

## Task 4: Multi-image diff tools

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess_multi.py`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline_multi.py`

`diff_preprocess_multi.py` runs the single-image `diff_preprocess.py` twice (once per image) and emits two Gate 1 verdicts. `diff_pipeline_multi.py` mirrors `diff_pipeline.py` but:
1. Verifies `vision_embeds.safetensors` exists on both sides and computes a Gate-2 max_diff over the concatenated merger output.
2. If both sides also have the 30 (or 126) op-level dumps from the existing hooks, runs the standard per-tensor diff for an op-level rupture report.

- [ ] **Step 4.1: Write `diff_preprocess_multi.py`**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess_multi.py`:

```python
"""P6.6 per-image preprocess diff (Gate 1).

Runs the diff_preprocess routine twice — once per image — and emits a
combined report with two verdict lines (1A for image_0, 1B for image_1).

Each per-image diff treats the ironmlx-side preprocess output (vlmlayout
[N_i, 1536] C-major) against mlx-vlm's pre-split slice
(image_{i}_pv.safetensors).

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_preprocess_multi.py \\
        --py /tmp/p6_diff_multi/python \\
        --iron /tmp/p6_diff_multi/ironmlx_pre \\
        --out /path/to/p6_6_preprocess_report.md \\
        --gate 0.05
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


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "count_above_1e-3": int((d > 1e-3).sum()),
        "count_above_1e-2": int((d > 1e-2).sum()),
        "total": int(d.size),
    }


def diff_one(vlm: Path, iron: Path) -> dict:
    a = load_tensor(vlm)
    b = load_tensor(iron)
    if a.shape != b.shape:
        return {"error": f"shape mismatch: vlm {a.shape} vs iron {b.shape}"}
    return diff_stats(a, b)


def render(image_id: int, stats: dict, gate: float) -> list[str]:
    if "error" in stats:
        return [f"## image_{image_id}", "", f"**ERROR**: {stats['error']}", ""]
    pass_gate = stats["max"] < gate
    return [
        f"## image_{image_id}",
        "",
        f"- max: {stats['max']:.6f}",
        f"- mean: {stats['mean']:.6f}",
        f"- p99: {stats['p99']:.6f}",
        f"- count > 1e-3: {stats['count_above_1e-3']} / {stats['total']}",
        f"- Gate 1{'A' if image_id == 0 else 'B'} verdict: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path,
                        help="dir with image_0_pv.safetensors + image_1_pv.safetensors")
    parser.add_argument("--iron", required=True, type=Path,
                        help="dir with image_0_pv_vlmlayout.safetensors + image_1_pv_vlmlayout.safetensors")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate", type=float, default=0.05)
    args = parser.parse_args()

    lines = ["# P6.6 Multi-Image Preprocess Diff (Gate 1)", "",
             f"- Gate 1 threshold: < {args.gate}", ""]
    overall_pass = True
    for i in (0, 1):
        vlm = args.py / f"image_{i}_pv.safetensors"
        iron = args.iron / f"image_{i}_pv_vlmlayout.safetensors"
        if not vlm.exists() or not iron.exists():
            lines.append(f"## image_{i} — missing input")
            lines.append(f"- vlm exists: {vlm.exists()}")
            lines.append(f"- iron exists: {iron.exists()}")
            lines.append("")
            overall_pass = False
            continue
        stats = diff_one(vlm, iron)
        lines.extend(render(i, stats, args.gate))
        if "error" in stats or stats.get("max", 99.0) >= args.gate:
            overall_pass = False

    lines.append(f"## Overall Gate 1: **{'PASS' if overall_pass else 'FAIL'}**")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[diff_preprocess_multi] overall {'PASS' if overall_pass else 'FAIL'}; report → {args.out}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.2: Write `diff_pipeline_multi.py`**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline_multi.py`:

```python
"""P6.6 multi-image vision-encoder diff (Gate 2 + op-level rupture).

Compares mlx-vlm and ironmlx vision-tower outputs for a 2-image input.
Reuses the per-tensor pairing pattern from diff_pipeline.py (P6.1) but
focuses on the final vision_embeds tensor for Gate 2; if op-level
intermediate tensors are also present in both dirs (29 module + 96
intra-block sites from P6.1+P6.3b hooks), include them in a per-tensor
table.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_pipeline_multi.py \\
        --py /tmp/p6_diff_multi/python \\
        --rust /tmp/p6_diff_multi/rust \\
        --out /path/to/p6_6_vision_report
"""
from __future__ import annotations

import argparse
import json
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


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "count_above_1e-3": int((d > 1e-3).sum()),
        "count_above_1e-2": int((d > 1e-2).sum()),
        "count_above_1e-1": int((d > 1e-1).sum()),
        "total": int(d.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate2", type=float, default=0.1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Gate 2: final vision_embeds
    vlm_emb = args.py / "vision_embeds.safetensors"
    iron_emb = args.rust / "vision_embeds.safetensors"
    if not vlm_emb.exists() or not iron_emb.exists():
        msg = f"missing vision_embeds — vlm={vlm_emb.exists()}, iron={iron_emb.exists()}"
        print(f"ERROR: {msg}", file=sys.stderr)
        (args.out / "report.md").write_text(f"# P6.6 Vision Diff\n\n**ERROR**: {msg}\n")
        return 2

    a = load_tensor(vlm_emb)
    b = load_tensor(iron_emb)
    if a.shape != b.shape:
        msg = f"vision_embeds shape mismatch: vlm {a.shape} vs iron {b.shape}"
        print(f"ERROR: {msg}", file=sys.stderr)
        (args.out / "report.md").write_text(f"# P6.6 Vision Diff\n\n**ERROR**: {msg}\n")
        return 2

    final_stats = diff_stats(a, b)
    gate2_pass = final_stats["max"] < args.gate2

    # Op-level (optional) — only if the existing P6.1+P6.3b hook outputs are
    # also in py_dir AND rust_dir. Reuse the basename-pair pattern.
    op_rows = []
    py_files = {p.stem: p for p in sorted(args.py.glob("*.safetensors"))}
    rust_files = {p.stem: p for p in sorted(args.rust.glob("*.safetensors"))}
    common = sorted(set(py_files) & set(rust_files))
    common = [c for c in common if c != "vision_embeds"]  # already reported
    for name in common:
        try:
            a2 = load_tensor(py_files[name])
            b2 = load_tensor(rust_files[name])
            if a2.shape != b2.shape:
                op_rows.append({"name": name, "error": f"shape mismatch {a2.shape} vs {b2.shape}"})
                continue
            s = diff_stats(a2, b2)
            s["name"] = name
            s["shape"] = list(a2.shape)
            op_rows.append(s)
        except Exception as e:
            op_rows.append({"name": name, "error": str(e)})

    lines = [
        "# P6.6 Multi-Image Vision Diff (Gate 2)",
        "",
        f"- Gate 2 threshold: < {args.gate2}",
        f"- vision_embeds shape: {list(a.shape)}",
        f"- vision_embeds max_diff: **{final_stats['max']:.4f}**",
        f"- Gate 2 verdict: **{'PASS' if gate2_pass else 'FAIL'}**",
        "",
        "## Final vision_embeds stats",
        "",
        f"- max: {final_stats['max']:.6f}",
        f"- mean: {final_stats['mean']:.6f}",
        f"- p99: {final_stats['p99']:.6f}",
        f"- count > 1e-2: {final_stats['count_above_1e-2']} / {final_stats['total']}",
        f"- count > 1e-1: {final_stats['count_above_1e-1']} / {final_stats['total']}",
        "",
    ]
    if op_rows:
        lines.append("## Op-level intermediate tensors")
        lines.append("")
        lines.append("| tensor | shape | max | mean | >1e-2 | >1e-1 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in op_rows:
            if "error" in r:
                lines.append(f"| `{r['name']}` | — | error: {r['error']} | | | |")
                continue
            lines.append(
                f"| `{r['name']}` | {r['shape']} | {r['max']:.4f} | {r['mean']:.6f} | "
                f"{r['count_above_1e-2']}/{r['total']} | {r['count_above_1e-1']}/{r['total']} |"
            )
        lines.append("")

    (args.out / "report.md").write_text("\n".join(lines))
    (args.out / "summary.json").write_text(json.dumps({
        "gate2_max_diff": final_stats["max"],
        "gate2_pass": gate2_pass,
        "vision_embeds_shape": list(a.shape),
    }, indent=2))
    print(f"[diff_pipeline_multi] Gate 2 {'PASS' if gate2_pass else 'FAIL'}; report → {args.out}/report.md")
    return 0 if gate2_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.3: Commit both tools**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/diff_preprocess_multi.py \
        ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline_multi.py
git commit -m "feat(p6.6): multi-image diff_preprocess + diff_pipeline tools"
```

---

## Task 5: ironmlx preprocess re-use for Gate 1 (per-image)

The ironmlx-side single-image preprocess dumper from P6.3a (`p6_3a_preprocess_dump.rs`) already does exactly what we need — but for one image at a time. We'll just invoke it twice from the driver (Task 7) and rename outputs to `image_{i}_pv_*.safetensors`. No new Rust file needed.

- [ ] **Step 5.1: Confirm `p6_3a_preprocess_dump.rs` exists and works**

```bash
ls /Volumes/Dev/cxx-mlx/ironmlx/tests/p6_3a_preprocess_dump.rs
grep "IMAGE_PATH\|IRONMLX_PREPROCESS_DUMP_DIR\|fn p6_3a_preprocess_dump" /Volumes/Dev/cxx-mlx/ironmlx/tests/p6_3a_preprocess_dump.rs | head -5
```

Expected: file exists, references the two env vars + the test function name. If not present (P6.3a wasn't merged into the chain), see `ironmlx-p6-3-vision-correctness` git history.

- [ ] **Step 5.2: Smoke-run it on one image to confirm artifacts**

```bash
mkdir -p /tmp/p6_diff_multi/ironmlx_pre_test
MLX_DIR=/Users/sam/.local/mlx \
IMAGE_PATH=/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg \
IRONMLX_PREPROCESS_DUMP_DIR=/tmp/p6_diff_multi/ironmlx_pre_test \
cargo test -p ironmlx --features vision-dump --test p6_3a_preprocess_dump --release -- --ignored 2>&1 | tail -5
ls /tmp/p6_diff_multi/ironmlx_pre_test/
```

Expected: PASS, and `00_ironmlx_pv_native.safetensors` + `00_ironmlx_pv_vlmlayout.safetensors` present.

The driver in Task 7 will:
1. Run this test once per image into a dedicated subdir, then mv the outputs from `00_…safetensors` to `image_{i}_pv_*.safetensors`.

- [ ] **Step 5.3: No commit (verification only)**

---

## Task 6: ironmlx-side e2e logits-match (Gate 3)

**Files:**
- Create: `ironmlx/tests/p6_6_logits_match.rs`

Mirrors `p6_qwen35_vl_logits_match.rs` (P6 Task 21) but the fixture has 2 images. The forward path uses `model.forward_vl` with `pixel_values` = the concatenated [N_total, 1536] from mlx-vlm + the multi-image `grid_thw`.

- [ ] **Step 6.1: Write the e2e test**

Create `ironmlx/tests/p6_6_logits_match.rs`:

```rust
//! P6.6 multi-image e2e logits-match (Gate 3).
//!
//! Drives Qwen35Model::forward_vl on a 2-image input + the mlx-vlm-generated
//! input_ids, compares last-position logits + greedy first token against
//! the mlx-vlm reference fixture written by run_p6_6_dump.py.
//!
//! Run with:
//!   MLX_DIR=$HOME/.local/mlx \
//!   QWEN35_MODEL=/path/to/model \
//!   cargo test -p ironmlx --test p6_6_logits_match --release -- --ignored

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::generate::{build_position_ids_vl, IMAGE_TOKEN_ID};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Config;
use ironmlx::models::qwen3_5::Qwen35Model;

const FIXTURE_DIR: &str = "tests/fixtures/p6_qwen35_vl/multi_image";

fn load_npy_int32(path: &str) -> Array {
    let arr = mlx::io::load(path).expect("load_npy");
    mlx::ops::cast::astype(&arr, Dtype::Int32).expect("cast")
}

fn load_npy_f32(path: &str) -> Array {
    let arr = mlx::io::load(path).expect("load_npy");
    mlx::ops::cast::astype(&arr, Dtype::Float32).expect("cast")
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).expect("a32");
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).expect("b32");
    let av: Vec<f32> = a32.to_vec::<f32>().expect("av");
    let bv: Vec<f32> = b32.to_vec::<f32>().expect("bv");
    av.iter()
        .zip(&bv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore]
fn p6_6_logits_match() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    // Load mlx-vlm fixture artifacts.
    let input_ids = load_npy_int32(&format!("{FIXTURE_DIR}/expected_input_ids.npy"));
    let grid_arr = load_npy_int32(&format!("{FIXTURE_DIR}/expected_image_grid_thw.npy"));
    let (mut loaded, _meta) =
        mlx::io::load_safetensors(&format!("{FIXTURE_DIR}/expected_pixel_values.safetensors"))
            .expect("load pv");
    let pv_flat: Array = loaded.remove("tensor").expect("tensor key");
    let expected_logits =
        load_npy_f32(&format!("{FIXTURE_DIR}/expected_last_logits.npy"));
    let expected_first_token: i32 = std::fs::read_to_string(
        format!("{FIXTURE_DIR}/expected_first_token.txt"),
    )
    .expect("read first_token")
    .trim()
    .parse()
    .expect("parse first_token");

    // Convert mlx-vlm [N_total, 1536] C-major to ironmlx [N_total, T, C, H, W].
    let n_total = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat
        .reshape(&[n_total, 3, 2, 16, 16][..])
        .expect("reshape pv");
    let pv = mlx::ops::shape::transpose_axes(&pv_5d, &[0, 2, 1, 3, 4][..]).expect("transpose pv");

    // grid_thw: [2, 3] -> Vec<(t, h, w)>
    let grids_flat: Vec<i32> = grid_arr.to_vec::<i32>().expect("grid to_vec");
    let grids: Vec<(i32, i32, i32)> = grids_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    assert_eq!(grids.len(), 2);

    // Build MRoPE VL position ids using ironmlx's own builder (matches mlx-vlm).
    let spatial_merge_size = cfg
        .vision_config
        .as_ref()
        .map(|vc| vc.spatial_merge_size)
        .unwrap_or(2);
    let ids_i32: Vec<i32> = input_ids.to_vec::<i32>().expect("ids to_vec");
    let pos_ids = build_position_ids_vl(&ids_i32, &grids, IMAGE_TOKEN_ID, spatial_merge_size)
        .expect("position ids");

    // Forward.
    let logits = model
        .forward_vl(
            &input_ids,
            &pos_ids,
            None,
            Some(&pv),
            Some(&grids),
            IMAGE_TOKEN_ID,
            (),
        )
        .expect("forward_vl");

    // Last-position diff.
    let dim = logits.shape().as_slice();
    let vocab = *dim.last().expect("logits has at least 1 dim");
    let our_flat = logits
        .reshape(&[vocab][..])
        .expect("reshape our logits to [vocab]");
    let expected_flat = expected_logits
        .reshape(&[vocab][..])
        .expect("reshape expected_logits");

    let max_diff = max_abs_diff(&our_flat, &expected_flat);
    eprintln!("[p6_6_logits_match] max_abs_diff = {max_diff:.4}");

    let argmax = mlx::ops::reduce::argmax(&our_flat, 0, false).expect("argmax");
    let our_first: i32 = mlx::ops::cast::astype(&argmax, Dtype::Int32)
        .expect("cast first")
        .item::<i32>()
        .expect("item first");
    eprintln!(
        "[p6_6_logits_match] our_first_token={our_first} expected={expected_first_token}"
    );

    assert_eq!(
        our_first, expected_first_token,
        "Gate 3B (greedy first token) failed"
    );
    // Gate 3A threshold is set after Task 7 baseline run.
    // For now, this test only checks the bit-identical first token.
}
```

- [ ] **Step 6.2: Compile-check**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --tests 2>&1 | tail -5
```

Expected: `Finished`. Adapt any wrong import paths.

- [ ] **Step 6.3: CI gauntlet**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --test p6_6_logits_match -- -D warnings 2>&1 | tail -3
```

- [ ] **Step 6.4: Commit**

```bash
git add ironmlx/tests/p6_6_logits_match.rs
git commit -m "feat(p6.6): p6_6_logits_match e2e integration test"
```

---

## Task 7: Top-level driver + baseline run + threshold-set

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh`

- [ ] **Step 7.1: Write the orchestrator**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh`:

```bash
#!/usr/bin/env bash
# P6.6 multi-image diff orchestrator.
#
# Stage 1: mlx-vlm dump (multi-image; optionally op-level if MLXVLM_VISION_DUMP_DIR=$PY_DIR is set inside this script)
# Stage 2: ironmlx preprocess dump (twice — once per image)
# Stage 3: ironmlx vision dump on concatenated pixel_values (op-level via existing hooks)
# Stage 4: diff_preprocess_multi (Gate 1)
# Stage 5: diff_pipeline_multi (Gate 2 + op-level)
# Stage 6: p6_6_logits_match (Gate 3, integration test)
#
# Required env: MLX_DIR, QWEN35_MODEL
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
MULTI_DIR="$FIXTURE_DIR/multi_image"
PY_DIR="${PY_DIR:-/tmp/p6_diff_multi/python}"
RUST_DIR="${RUST_DIR:-/tmp/p6_diff_multi/rust}"
IRON_PRE_DIR="${IRON_PRE_DIR:-/tmp/p6_diff_multi/ironmlx_pre}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_6-$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$IRON_PRE_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$PY_DIR"/*.npy "$PY_DIR"/*.txt
rm -f "$RUST_DIR"/*.safetensors
rm -rf "$IRON_PRE_DIR"
mkdir -p "$IRON_PRE_DIR"

echo "=== Stage 1: mlx-vlm 2-image dump (with op-level hooks if hooks active) ==="
# Setting MLXVLM_VISION_DUMP_DIR=$PY_DIR activates op-level hooks too.
MLXVLM_VISION_DUMP_DIR="$PY_DIR" \
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_p6_6_dump.py" \
        --image-0 "$MULTI_DIR/image_0.jpg" \
        --image-1 "$MULTI_DIR/image_1.jpg" \
        --out-dir "$PY_DIR" 2>&1 | tail -10
echo "  PY_DIR files: $(ls "$PY_DIR" | wc -l)"

echo "=== Stage 2: ironmlx preprocess dump (per image) ==="
cd "$REPO_ROOT"
for i in 0 1; do
    SUBDIR="$IRON_PRE_DIR/image_${i}"
    mkdir -p "$SUBDIR"
    MLX_DIR="$MLX_DIR" \
        IMAGE_PATH="$MULTI_DIR/image_${i}.jpg" \
        IRONMLX_PREPROCESS_DUMP_DIR="$SUBDIR" \
        cargo test -p ironmlx --features vision-dump --release \
            --test p6_3a_preprocess_dump -- --ignored 2>&1 | tail -3
    mv "$SUBDIR/00_ironmlx_pv_native.safetensors" "$IRON_PRE_DIR/image_${i}_pv_native.safetensors"
    mv "$SUBDIR/00_ironmlx_pv_vlmlayout.safetensors" "$IRON_PRE_DIR/image_${i}_pv_vlmlayout.safetensors"
    rmdir "$SUBDIR"
done
echo "  IRON_PRE files: $(ls "$IRON_PRE_DIR" | wc -l)"

echo "=== Stage 3: ironmlx vision dump on concatenated input (op-level via existing hooks) ==="
MLX_DIR="$MLX_DIR" \
QWEN35_MODEL="$QWEN35_MODEL" \
IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
PIXEL_VALUES_PATH="$PY_DIR/expected_pixel_values.safetensors" \
IMAGE_GRID_THW_PATH="$PY_DIR/expected_image_grid_thw.npy" \
    cargo test -p ironmlx --features vision-dump --release \
        --test p6_6_multi_image_dump -- --ignored 2>&1 | tail -5
echo "  RUST_DIR files: $(ls "$RUST_DIR" | wc -l)"

echo "=== Stage 4: Gate 1 — per-image preprocess diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_preprocess_multi.py" \
    --py "$PY_DIR" --iron "$IRON_PRE_DIR" \
    --out "$REPORT_DIR/p6_6_preprocess_report.md" \
    --gate 0.05 || true

echo "=== Stage 5: Gate 2 — vision encoder diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline_multi.py" \
    --py "$PY_DIR" --rust "$RUST_DIR" \
    --out "$REPORT_DIR/vision" \
    --gate2 0.1 || true

echo "=== Stage 6: Gate 3 — e2e logits-match integration test ==="
# Symlink the fixture into the canonical location the test reads from
ln -sf "$PY_DIR/expected_input_ids.npy" "$MULTI_DIR/expected_input_ids.npy" || true
ln -sf "$PY_DIR/expected_image_grid_thw.npy" "$MULTI_DIR/expected_image_grid_thw.npy" || true
ln -sf "$PY_DIR/expected_pixel_values.safetensors" "$MULTI_DIR/expected_pixel_values.safetensors" || true
ln -sf "$PY_DIR/expected_last_logits.npy" "$MULTI_DIR/expected_last_logits.npy" || true
ln -sf "$PY_DIR/expected_first_token.txt" "$MULTI_DIR/expected_first_token.txt" || true

QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 \
    | tee "$REPORT_DIR/p6_6_logits_match.log" | tail -15

echo "=== Done. Reports in: $REPORT_DIR ==="
```

- [ ] **Step 7.2: Make executable + run**

```bash
chmod +x /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh 2>&1 | tail -25
```

Expected: 6 stages complete, last line `=== Done. Reports in: <path> ===`. Each Gate's verdict (PASS or FAIL) is in its corresponding report file. The logits-match test may either PASS (Gate 3B) or fail with a specific assertion line in the .log.

- [ ] **Step 7.3: Read baselines + set 4 thresholds**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-* | head -1)
cat "$REPORT_DIR/p6_6_preprocess_report.md"
echo "---"
cat "$REPORT_DIR/vision/report.md" | head -30
echo "---"
grep "max_abs_diff\|first_token\|panicked" "$REPORT_DIR/p6_6_logits_match.log" | head -10
```

Record into a temporary text file:
- Gate 1A baseline (image_0 max_diff)
- Gate 1B baseline (image_1 max_diff)
- Gate 2 baseline (vision_embeds max_diff)
- Gate 3A baseline (e2e logits max_diff)
- Gate 3B baseline (first_token equal/not)

**Threshold-set rule** (per spec §2):
- If baseline ≤ 1.5× the P6.3 single-image number → use the P6.3 number as threshold
- If baseline > 1.5× → set threshold to `ceil(baseline / 0.05) * 0.05` (round up to next 0.05)
- If baseline ≥ 5× the P6.3 number → that's a bug; do NOT widen threshold; proceed to Task 8 fix loop

Concrete P6.3 anchor numbers (from `tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md`):
- Gate 1: 0.0254  → P6.6 threshold likely 0.05
- Gate 2: 0.0330  → P6.6 threshold likely 0.1
- Gate 3A: 0.3906 → P6.6 threshold likely 0.5
- Gate 3B: bit-identical (hard gate)

- [ ] **Step 7.4: Commit driver + baseline reports**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-* | head -1)
git add ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh
git add -f "$REPORT_DIR/p6_6_preprocess_report.md" "$REPORT_DIR/vision/report.md" \
       "$REPORT_DIR/vision/summary.json" "$REPORT_DIR/p6_6_logits_match.log"
git commit -m "feat(p6.6): orchestrator run_p6_6_diff.sh + multi-image baseline report"
```

---

## Task 8: P6.6b — Diagnose-fix loop

**Files:**
- Modify (only if Gate fails): `ironmlx/src/models/qwen3_5/vision/mod.rs`,
  `ironmlx/src/models/qwen3_5/cross_modal.rs`, or wherever rupture localizes

This task is an iterative fix loop. Skip entirely if Task 7's run shows all 5 gates PASS.

- [ ] **Step 8.1: Identify the first failing gate**

The order of failure-priority matches the data flow:
1. Gate 1A or 1B fails → preprocess produces different bytes per image (unlikely — single-image P6.3a passed; double-check the per-image split in `run_p6_6_dump.py` matches what `image_processor::preprocess` produces for the same image)
2. Gate 2 fails → vision encoder multi-grid path (rupture in `add_learned_pos_embed`, `compute_rotary_pos_emb`, `cu_seqlens` build, or per-block forward)
3. Gate 3A fails (Gate 2 passed) → `cross_modal::replace_image_tokens` multi-span scatter routing
4. Gate 3B fails (token mismatch) → likely Gate 2 or Gate 3A logic; check op-level report

- [ ] **Step 8.2: Map the rupture to a likely fix file**

Decision matrix:

| Symptom | Hypothesis | Inspect file |
| --- | --- | --- |
| Gate 1A or 1B max_diff > 0.06 | Per-image patchify byte layout differs from mlx-vlm | `ironmlx/src/models/qwen3_5/image_processor.rs::patchify` (search for hardcoded N=1 assumptions) |
| Gate 2 max_diff >> 0.1, Op-level report shows rupture at `02_pos_embed_contrib` | `add_learned_pos_embed` multi-grid iteration error (e.g., flat indices not offset per-image) | `ironmlx/src/models/qwen3_5/vision/mod.rs::add_learned_pos_embed` |
| Op-level rupture at `04_rotary_freqs` | `compute_rotary_pos_emb` multi-image build | same file, `compute_rotary_pos_emb` |
| Op-level rupture at first `block_NN_b_attn_residual` | `cu_seqlens` boundary wrong (attention bleeds across images) | same file, the loop building `cu_seqlens` in `forward` |
| Gate 3A fails, Gate 2 passes | `cross_modal::replace_image_tokens` row-order assumption: vision_embeds[k] should map to the k-th image-token in input_ids regardless of image | `ironmlx/src/models/qwen3_5/cross_modal.rs` |
| Gate 3B fails AND all other gates pass | likely numerical drift; if Gate 3A max_diff < threshold, this is unexpected. Investigate which logit position differs and whether it's near argmax of either side. | check `expected_first_token.txt` actual value + ironmlx printed value |

- [ ] **Step 8.3: Apply ONE fix per hypothesis test**

Read the relevant function. Walk through what it does for N=1 (the P6.3 verified path) and what changes for N=2. The most common multi-image bug: **a flat index built from the first grid being used unmodified for the second grid**.

Example mock-fix sketch for `cu_seqlens` (in `vision/mod.rs::forward`):

```rust
// Wrong: cu_seqlens [0, n0+n1] — attention sees both images as one segment
// Right:  cu_seqlens [0, n0, n0+n1] — attention isolates each image's tokens
let cu_seqlens: Vec<i32> = {
    let mut v = vec![0_i32];
    let mut total = 0_i32;
    for &(t, h, w) in grid_thw {
        total += t * h * w;
        v.push(total);
    }
    v
};
```

(This particular form is the existing code, so it's actually right. Use it as a model for what other multi-image paths must look like.)

Make the minimum change. Commit each verified fix individually.

- [ ] **Step 8.4: Re-run the orchestrator**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh 2>&1 | tail -15
```

Check the new report for the targeted gate. If max_diff drops below threshold, that fix is good — commit. Otherwise, revert and try the next hypothesis.

- [ ] **Step 8.5: Cap to 3 hypothesis tests per gate**

If after 3 iterations a gate is still failing:

1. Document the residual rupture and what was tried (commit message body).
2. Decide: relax the threshold (with explicit Boss approval — do NOT silently widen) OR escalate via separate brainstorming for a deeper investigation.

- [ ] **Step 8.6: Run full Rust regression after each commit**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: 153 passed (the P6.4 baseline). If a fix regressed single-image, that's a problem — revert.

- [ ] **Step 8.7: Run P6.3 Task 21 single-image logits-match to confirm no single-image regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --test p6_qwen35_vl_logits_match --release -- --ignored 2>&1 | tail -5
```

Expected: PASS, max_diff still ≤ 0.5039 (P6.3 single-image baseline).

- [ ] **Step 8.8: Final commit of Gate-green report**

When all of Gates 1A, 1B, 2, 3A, 3B PASS:

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-* | head -1)
git add -f "$REPORT_DIR/"
git commit -m "docs(p6.6): Gates 1+2+3 green (multi-image numerical)"
```

---

## Task 9: Semantic check (Gate 4)

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/p6_6_semantic_check.py`

Mirrors `item3_semantic_check.py` (P6.3c) but builds ONE request with BOTH images and checks the response for per-image key facts.

- [ ] **Step 9.1: Pick per-image key facts**

Inspect both images and document 3 key facts each:

```bash
# Open the two images and write down what's in them.
# Example for the recommended val2017 ids 397133 and 252219:
#   image_0 (dog on couch):
#     facts: ["dog", "couch / sofa / sofa", "person"]
#   image_1 (tennis player):
#     facts: ["tennis", "racket", "court"]
```

The actual key-fact list goes into the script.

- [ ] **Step 9.2: Write the script**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/p6_6_semantic_check.py`:

```python
#!/usr/bin/env python
"""P6.6 Gate 4: 2-image semantic correctness.

Starts the ironmlx HTTP server, queries it with ONE chat request that
includes BOTH images as image_url parts, and verifies the response
text contains key facts from each image.

Per-image criteria mirror item3_semantic_check.py — ≥ 2 / 3 keys must
match per image. Overall Gate 4 passes if both image criteria pass.

Usage:
    MLX_DIR=$HOME/.local/mlx \\
    QWEN35_MODEL=/path/to/model \\
    ~/.venvs/mlxvlm-ref/bin/python p6_6_semantic_check.py \\
        --out /path/to/p6_6_semantic_report.md
"""
from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "ironmlx/tests/fixtures/p6_qwen35_vl"
MULTI_DIR = FIXTURE_DIR / "multi_image"
PROMPT = (
    "There are two images. Describe each one separately. "
    "For each image, mention the key objects and what is happening."
)

# Per-image keys (TODO at run-time: confirm by visually inspecting both images
# and editing this list).  Each fact is a list of acceptable synonyms.
KEYS_PER_IMAGE = {
    0: [
        ["dog", "puppy", "canine"],
        ["couch", "sofa", "cushion"],
        ["person", "people", "owner", "human", "man", "woman"],
    ],
    1: [
        ["tennis", "racket", "racquet"],
        ["court", "tennis court"],
        ["player", "athlete", "person", "people"],
    ],
}
MIN_KEYS_PER_IMAGE = 2  # ≥ 2 / 3 per image


def wait_for_port(port: int, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def evaluate_per_image(text: str) -> dict:
    t = text.lower()
    per_image_results = {}
    for i, key_groups in KEYS_PER_IMAGE.items():
        hits = []
        for synonyms in key_groups:
            matched = next((s for s in synonyms if s.lower() in t), None)
            if matched is not None:
                hits.append(matched)
            else:
                hits.append(None)
        n_hit = sum(1 for h in hits if h is not None)
        per_image_results[i] = {
            "n_hit": n_hit,
            "n_total": len(key_groups),
            "hits": hits,
            "passed": n_hit >= MIN_KEYS_PER_IMAGE,
        }
    return per_image_results


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

    subprocess.run(["pkill", "-KILL", "-f", "ironmlx serve"], check=False)
    time.sleep(2)

    log_path = Path("/tmp/p6_6_server.log")
    server_log = log_path.open("w")
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
        if not wait_for_port(args.port, timeout_s=180):
            print(f"ERROR: server failed to start; see {log_path}", file=sys.stderr)
            return 2

        b0 = base64.b64encode((MULTI_DIR / "image_0.jpg").read_bytes()).decode("ascii")
        b1 = base64.b64encode((MULTI_DIR / "image_1.jpg").read_bytes()).decode("ascii")
        payload = {
            "model": "qwen3_5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b0}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b1}"}},
                ],
            }],
            "max_tokens": 600,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        r = requests.post(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                          json=payload, timeout=900)
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        finish = body["choices"][0]["finish_reason"]

        per_image = evaluate_per_image(text)
        passed = all(v["passed"] for v in per_image.values())

        lines = ["# P6.6 Multi-Image Semantic Verification (Gate 4)", "",
                 f"- Finish reason: `{finish}`",
                 f"- Overall verdict: **{'PASS' if passed else 'FAIL'}**",
                 ""]
        for i, res in per_image.items():
            lines.append(f"## image_{i} — {'✅ PASS' if res['passed'] else '❌ FAIL'}")
            lines.append("")
            lines.append(f"- {res['n_hit']} / {res['n_total']} keys found")
            for synonyms, hit in zip(KEYS_PER_IMAGE[i], res["hits"]):
                lines.append(f"  - {'✓' if hit else '✗'} `{synonyms[0]}` "
                             f"({'matched ' + hit if hit else 'missing'})")
            lines.append("")
        lines.append("## Response")
        lines.append("")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        args.out.write_text("\n".join(lines))
        print(f"[p6_6_semantic_check] {'PASS' if passed else 'FAIL'}; report → {args.out}")
        return 0 if passed else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9.3: Verify the model binary is fresh**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
```

- [ ] **Step 9.4: Run Gate 4**

```bash
STAMP=$(date +%Y-%m-%d-%H%M)
REPORT_DIR=/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6c-$STAMP
mkdir -p "$REPORT_DIR"
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/p6_6_semantic_check.py \
    --out "$REPORT_DIR/p6_6_semantic_report.md" 2>&1 | tail -10
```

Expected: `[p6_6_semantic_check] PASS; report → ...`. If FAIL, inspect the response to decide:
- Did the model only describe one image? → genuine multi-image-routing bug; go back to Task 8
- Did the model describe both but used different vocab? → adjust the synonym lists in `KEYS_PER_IMAGE`
- Did the model ramble without specifics? → not a code bug, but Gate 4 stays FAIL

- [ ] **Step 9.5: Commit semantic driver + Gate 4 report**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6c-* | head -1)
git add ironmlx/tests/fixtures/p6_qwen35_vl/p6_6_semantic_check.py
git add -f "$REPORT_DIR/p6_6_semantic_report.md"
git commit -m "feat(p6.6c): Gate 4 semantic verification + multi-image report"
```

---

## Task 10: Close-out report

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6_closeout/report.md`

- [ ] **Step 10.1: Gather final numbers**

```bash
# From the latest p6_6-* report dir
PREP_REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-*/p6_6_preprocess_report.md | head -1)
VISION_REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-*/vision/report.md | head -1)
LOG=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6-*/p6_6_logits_match.log | head -1)
SEM_REPORT=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6c-*/p6_6_semantic_report.md | head -1)
echo "Gate 1: $PREP_REPORT"
echo "Gate 2: $VISION_REPORT"
echo "Gate 3 log: $LOG"
echo "Gate 4: $SEM_REPORT"
```

Read each and extract the final numbers.

- [ ] **Step 10.2: Write `p6_6_closeout/report.md`**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6_closeout/report.md`:

```markdown
# P6.6 Multi-Image-Per-Request — Close-out

**Branch:** `ironmlx-p6-6-multi-image`
**Date:** <fill: date>
**Spec:** `docs/superpowers/specs/2026-05-11-p6-6-multi-image-design.md` (commit `ee3aef7`)

## Acceptance Table

| Gate | Target | Baseline (diagnose) | Final | Status |
| --- | --- | --- | --- | --- |
| 1A. image_0 preprocess max_diff | < <fill: threshold> | <fill: baseline> | <fill: final> | <fill: ✅/❌> |
| 1B. image_1 preprocess max_diff | < <fill: threshold> | <fill: baseline> | <fill: final> | <fill: ✅/❌> |
| 2. Vision encoder concat max_diff | < <fill: threshold> | <fill: baseline> | <fill: final> | <fill: ✅/❌> |
| 3A. E2E logits max_diff | < <fill: threshold> | <fill: baseline> | <fill: final> | <fill: ✅/❌> |
| 3B. Greedy first-token | bit-identical | <fill: yes/no> | <fill: yes/no> | <fill: ✅/❌> |
| 4a. image_0 key facts | ≥ 2/3 | <fill: baseline> | <fill: final> | <fill: ✅/❌> |
| 4b. image_1 key facts | ≥ 2/3 | <fill: baseline> | <fill: final> | <fill: ✅/❌> |

## Fixes Applied

<fill: list of commits with one-line description each>

## Regression Status

- `cargo test -p ironmlx --lib --release -- --test-threads=1`: <fill: N passed / 0 failed>
- P6.3 single-image logits-match (Task 21): <fill: max_diff value, PASS/FAIL>
- P6.4 cleanup tests: <fill: still green>

## Notes

<fill: anything surprising, residual concerns, P6.7+ candidates>

## Linked Reports

- Preprocess (Gate 1): <fill: path>
- Vision encoder (Gate 2): <fill: path>
- E2E logits (Gate 3): <fill: path>
- Semantic (Gate 4): <fill: path>
```

Replace every `<fill: …>` with the actual recorded number / verdict / commit list.

- [ ] **Step 10.3: Run a final full regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx -- -D warnings 2>&1 | tail -3
```

All clean.

- [ ] **Step 10.4: Commit close-out**

```bash
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_6_closeout/report.md
git commit -m "docs(p6.6): close-out — multi-image-per-request all gates green"
```

- [ ] **Step 10.5: Final summary log**

```bash
git log --oneline ironmlx-p6-4-cleanup..HEAD
```

Should list all P6.6 commits from Tasks 1–10 in order.

---

## Self-Review

**Spec coverage check:**

- Spec §2 Gate 1 (preprocess per image) → Tasks 2 (mlx-vlm), 5 (ironmlx re-use), 4 (diff_preprocess_multi), 7 (driver)
- Spec §2 Gate 2 (vision encoder concat) → Tasks 2 (mlx-vlm dump), 3 (ironmlx dump), 4 (diff_pipeline_multi)
- Spec §2 Gate 3A (e2e logits) + 3B (first-token) → Task 6 + Task 7 stage 6
- Spec §2 Gate 4 (semantic) → Tasks 9
- Spec §4 P6.6a (fixture + diagnose) → Tasks 1–7
- Spec §4 P6.6b (diagnose-fix) → Task 8
- Spec §4 P6.6c (semantic + close-out) → Tasks 9–10
- Spec §5 File structure: all 6 new tools listed → covered
- Spec §7 Risk + rollback: per-fix commits, 3-iteration cap → Task 8 step 8.5

No gaps.

**Placeholder scan:**

- Task 10's close-out report template has `<fill: …>` — these are intentional template slots for the engineer to fill at close-out time with actual measured values (mirrors P6.3 close-out template). Not plan placeholders.
- Task 8's decision matrix references "P6.3 baseline numbers" — concretely captured at the end of Task 7.3.
- No "TBD" / "implement later" / "appropriate error handling" patterns.

**Type/path consistency:**

- `image_grid_thw` shape `[2, 3]` (Tasks 2, 3, 6) consistent.
- `pixel_values` shape `[N_total, 1536]` C-major (Tasks 2, 3, 6) — matches P6.2 reshape contract.
- `grids: Vec<(i32, i32, i32)>` (Task 3 and Task 6) — same type.
- `IRONMLX_VISION_DUMP_DIR`, `IRONMLX_PREPROCESS_DUMP_DIR`, `PIXEL_VALUES_PATH`, `IMAGE_GRID_THW_PATH` env var names consistent across Tasks 3, 5, 7.
- File names `image_{i}_pv.safetensors`, `image_{i}_pv_native.safetensors`, `image_{i}_pv_vlmlayout.safetensors`, `vision_embeds.safetensors`, `expected_*.npy` consistent across Tasks 2, 3, 4, 5, 7.

No drift found.

---

## Plan complete and saved to `docs/superpowers/plans/2026-05-11-p6-6-multi-image.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task + two-stage review. Best fit: Tasks 1–7 are scaffolding (per-task isolated work). Task 8 is iterative diagnose-fix where fresh-context per iteration is helpful. Tasks 9–10 are scripted.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Better if you want to inspect each baseline number as it lands and steer the threshold-set + fix-decision in real time.

Which approach?
