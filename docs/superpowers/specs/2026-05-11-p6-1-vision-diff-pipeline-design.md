# P6.1 Vision Encoder Diff Pipeline — Design

**Status:** Draft (brainstormed 2026-05-11)
**Owner:** ironmlx
**Parent:** P6 VL (committed on `ironmlx-p6-vl`, max_diff = 0.5039 vs mlx-vlm)
**Branch target:** new `ironmlx-p6-1-vision-diff` branched from `ironmlx-p6-vl`

## 1. Motivation

P6 VL acceptance was reached (greedy first-token bit-identical with mlx-vlm; max_abs_diff = 0.5039 on the 248320-vocab logit vector). However, end-to-end functional testing on five distinct test images (Item 3 verification) revealed a **systematic long-decoding-chain divergence** from mlx-vlm:

- 5/5 images misclassified as "side-by-side composite / stereoscopic 3D pair"
- Counting tasks off by 2× (network court group: 20–25 reported vs ~12 actual; mlx-vlm answered 13+1)
- Special-attribute recognition fails: an upside-down STOP sign was missed entirely by ironmlx but recognized correctly by mlx-vlm ("rotated 180 degrees... reads POTS")

The Task 21 reviewer's own diagnostics had already established that:

- VisionTower output mean diff vs mlx-vlm = 0.000005 (zero systematic bias)
- 226/768000 VisionTower output values differ by up to 0.85 (outlier rounding)
- These 226 outliers, propagated through 28 quantized LM layers, produce a +0.042 systematic logit bias and push 1/248320 logits over the 0.50 threshold by 0.0039
- LM-only path with Python reference embeds = 0.3828 max_diff — well within bounds

**Conclusion**: the LM path is fine. The vision encoder produces a numerically-close-but-not-bit-identical output to mlx-vlm, and this small per-token difference causes early divergence in long decoding chains.

A separate investigation during brainstorming established that **dtype is not the issue** — both ironmlx and mlx-vlm use bf16 weights with F32 accumulators (mlx Steel kernel `typedef float accum_type;`; llama.cpp `GGML_PREC_F32` is the analog). The divergence is at the **op-level implementation path**, not the precision strategy.

P6.1 is the diagnostic precursor to that op-level fix. P6.1 produces a report that identifies exactly **which layer / which op** is the divergence source. The fix itself is deferred to a follow-up spec ("P6.2") driven by P6.1's findings.

## 2. Goals

- Produce a reproducible, automated diff pipeline that compares ironmlx and mlx-vlm vision encoder intermediate tensors layer-by-layer.
- Generate a markdown report and a `max_diff` curve PNG that pinpoints the first significant divergence.
- Stage the granularity: first 30-tensor coarse-grained sweep across module boundaries; later, if needed, a follow-up sub-spec drops into op-level granularity within whatever 1–2 blocks the coarse report flags.
- **Zero impact on production binary**: ironmlx-side dump hooks are gated behind a Cargo feature; mlx-vlm-side hooks live on a local fork.

## 3. Non-Goals

- No fixes to vision tower implementation in this spec (`vision/*.rs` untouched except for compile-gated dump statements).
- No dtype changes anywhere.
- No upstream contribution back to mlx-vlm (fork stays local).
- No batched / multi-image diff (P6 is single-image only).
- No video-modality dumps (P6 is image-only).

## 4. Architecture

```text
┌─────────────────────────────┐    ┌────────────────────────────────┐
│  /Volumes/Dev/mlx-vlm       │    │  ironmlx (vision-dump feature) │
│  vision.py + hook (forked)  │    │  vision/*.rs + cfg(feature)    │
│       ↓ writes ↓            │    │       ↓ writes ↓               │
│  /tmp/p6_diff/python/       │    │  /tmp/p6_diff/rust/            │
│      <name>.safetensors     │    │      <name>.safetensors        │
└─────────────────────────────┘    └────────────────────────────────┘
                ↓                                 ↓
                └────────► diff_pipeline.py ◄─────┘
                                ↓ outputs ↓
            tests/fixtures/p6_qwen35_vl/diff_reports/<date>/
                report.md + max_diff_curve.png + outliers.json
```

Three independent units, each owns one file family:

- **mlx-vlm fork**: `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` (modified inline). Hook activated when env `MLXVLM_VISION_DUMP_DIR` is set.
- **ironmlx vision-dump**: `cargo feature = "vision-dump"`, code gated on `#[cfg(feature = "vision-dump")]`. Hook activated when env `IRONMLX_VISION_DUMP_DIR` is set.
- **diff_pipeline.py**: lives in `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py`; reads the two dump dirs, writes the report dir.

## 5. Dump Protocol

### 5.1 File Format

Each dump tensor is a separate `.safetensors` file containing a single key named `tensor` plus a small metadata dict. Choice rationale: mlx-vlm and ironmlx both already have safetensors I/O; mlx-vlm's `mx.save_safetensors` works on `mx.array` directly; ironmlx's `mlx::io::save_safetensors` works on `Array` directly. No format conversion needed.

```text
# Each file
shape: list[int]   # tensor shape
dtype: str         # source dtype string; diff pipeline casts to f32 before comparing
tensor: array
```

### 5.2 Dump Point Names (canonical, identical on both sides)

| # | filename | shape | semantic stage |
| --- | --- | --- | --- |
| 0 | `00_pixel_values.safetensors` | `[N, 2, 3, 16, 16]` | preprocessed input |
| 1 | `01_patch_embed_out.safetensors` | `[N, 1024]` | after PatchEmbed |
| 2 | `02_pos_embed_contrib.safetensors` | `[N, 1024]` | bilinear interp output (additive) |
| 3 | `03_after_pos_embed.safetensors` | `[N, 1024]` | patch_embed + pos_embed (block input) |
| 4 | `04_rotary_freqs.safetensors` | `[N, 64]` | rotary frequency table |
| 5–28 | `05_block_00_out.safetensors` ... `28_block_23_out.safetensors` | `[N, 1024]` | output of each ViT block |
| 29 | `29_merger_out.safetensors` | `[N/4, 2560]` | final vision_embeds (model output) |

Both sides MUST emit identical filenames so diff_pipeline can pair them by name. Order index is encoded in filename prefix so listing-sorted iteration matches semantic order.

### 5.3 Input Determinism

To guarantee byte-identical inputs on both sides, **pixel_values is generated by Step 1 of the driver (the mlx-vlm side)** using the same `processor` + `prepare_inputs` path that Task 20's `gen_fixture.py` already uses, and saved as `00_pixel_values.safetensors`. ironmlx's dump path **reads this file** instead of running its own preprocess. This isolates the diff to the vision encoder forward path.

The image fixture is `tests/fixtures/p6_qwen35_vl/coco_sample.jpg` (already committed). Grid is `[1, 30, 40]` → 1200 patches → 300 merged tokens.

## 6. mlx-vlm Side (fork)

Modify `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py`:

```python
import os
_DUMP_DIR = os.environ.get("MLXVLM_VISION_DUMP_DIR")

def _maybe_dump(name: str, tensor: mx.array):
    if _DUMP_DIR:
        mx.eval(tensor)
        mx.save_safetensors(
            os.path.join(_DUMP_DIR, f"{name}.safetensors"),
            {"tensor": tensor}
        )

# Insert _maybe_dump calls at the 30 dump points inside __call__ /
# fast_pos_embed_interpolate / rot_pos_emb / VitBlock.__call__ / PatchMerger.__call__
```

The hook is no-op when `MLXVLM_VISION_DUMP_DIR` is unset, so the fork is safe to keep enabled.

A small driver script `tests/fixtures/p6_qwen35_vl/run_python_dump.py` calls the forked path with the env set, taking `pixel_values.safetensors` as input (NOT the raw image) to ensure determinism:

```python
#!/usr/bin/env python
# run_python_dump.py --pixel-values <path> --out-dir <path>
import argparse, os, mlx.core as mx
from mlx_vlm import load
...
os.environ["MLXVLM_VISION_DUMP_DIR"] = args.out_dir
pixel_values = mx.load(args.pixel_values)["tensor"]
model, _ = load(MODEL_DIR)
# Save input
mx.save_safetensors(f"{args.out_dir}/00_pixel_values.safetensors",
                    {"tensor": pixel_values})
# Forward vision tower
embeds = model.visual(pixel_values, grid_thw=mx.array([[1, 30, 40]]))
```

## 7. ironmlx Side (feature flag)

Add to `ironmlx/Cargo.toml`:

```toml
[features]
vision-dump = []
```

Add to `ironmlx/src/models/qwen3_5/vision/mod.rs` and friends:

```rust
#[cfg(feature = "vision-dump")]
fn dump_tensor(name: &str, t: &Array) {
    use std::env;
    let Ok(dir) = env::var("IRONMLX_VISION_DUMP_DIR") else { return };
    mlx::transforms::eval(&[t]).expect("dump eval");
    let path = format!("{dir}/{name}.safetensors");
    let mut map = std::collections::HashMap::new();
    map.insert("tensor".to_string(), t.clone());
    mlx::io::save_safetensors(&path, &map).expect("dump save");
}

#[cfg(not(feature = "vision-dump"))]
fn dump_tensor(_: &str, _: &Array) {}
```

Then insert `dump_tensor("XX_name", &tensor);` at the 30 sites. The non-feature build compiles the function to a no-op call which the compiler trivially inlines away.

**Implementation note**: at planning time, verify `mlx::io::save_safetensors` is exposed by the ironmlx-bundled `mlx` Rust crate. Task 9 verified `load_safetensors` exists; the saver path may need to be added to the crate's `mlx::io` module if missing. If the saver is absent, the fallback is to dump as raw little-endian `.bin` files plus a sibling `.json` manifest with shape+dtype, and update `diff_pipeline.py` to read both formats — but verify the saver first.

A new integration test `ironmlx/tests/p6_vision_dump.rs` (also `#[cfg(feature = "vision-dump")]` gated) drives a single forward pass:

```rust
#[test]
#[cfg(feature = "vision-dump")]
#[ignore]
fn p6_vision_dump() {
    use std::env;
    let pv_path = env::var("PIXEL_VALUES_PATH").expect("set PIXEL_VALUES_PATH");
    let model_dir = env::var("QWEN35_MODEL").expect("set QWEN35_MODEL");
    let dump_dir = env::var("IRONMLX_VISION_DUMP_DIR").expect("set IRONMLX_VISION_DUMP_DIR");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).unwrap();
    let cfg = Qwen35Config::from_loader(&loader).unwrap();
    let vc = cfg.vision_config.expect("vision_config");
    let tower = VisionTower::from_loader(&loader, &vc).unwrap();
    // Load pixel_values from disk (mlx-vlm preprocessed)
    let pv = mlx::io::load_safetensors(&pv_path).unwrap().remove("tensor").unwrap();
    // ... reshape from [N, 1536] to [N, 2, 3, 16, 16] (matches Task 21 convention)
    let _ = tower.forward(&pv_reshaped, &[(1, 30, 40)]).unwrap();
}
```

Run: `IRONMLX_VISION_DUMP_DIR=/tmp/p6_diff/rust/ QWEN35_MODEL=... PIXEL_VALUES_PATH=/tmp/p6_diff/python/00_pixel_values.safetensors MLX_DIR=... cargo test -p ironmlx --features vision-dump --test p6_vision_dump --release -- --ignored`

## 8. diff_pipeline.py

`tests/fixtures/p6_qwen35_vl/diff_pipeline.py`:

```python
# Usage: diff_pipeline.py --py <dir> --rust <dir> --out <dir>
import argparse, os, json, glob
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

def load_tensor(path):
    return np.array(mx.load(path)["tensor"].astype(mx.float32))

def diff_stats(a, b):
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

# 1. Pair files by basename
# 2. For each pair, compute diff_stats
# 3. Emit report.md (table + first-rupture summary)
# 4. Plot max_diff curve (log y) → max_diff_curve.png
# 5. Emit outliers.json with top-5 outliers (flat index + values) per tensor
```

Output report sections:

- **Summary**: number of tensors compared, first rupture point (where max_diff jumps > 5×), final merger_out max_diff
- **Per-tensor table**: all 30 rows with diff stats
- **Top-5 outliers in merger_out**: flat indices, ironmlx value, mlx-vlm value, abs diff
- **Curve**: `max_diff_curve.png` (log scale on y-axis)

## 9. Driver Script

Single shell script `tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT=$(date +%Y-%m-%d)
PY_DIR=/tmp/p6_diff/python
RUST_DIR=/tmp/p6_diff/rust
REPORT_DIR=tests/fixtures/p6_qwen35_vl/diff_reports/${OUT}

mkdir -p $PY_DIR $RUST_DIR $REPORT_DIR

# Step 1: mlx-vlm dump (includes 00_pixel_values.safetensors)
MLXVLM_VISION_DUMP_DIR=$PY_DIR ~/.venvs/mlxvlm-ref/bin/python \
  tests/fixtures/p6_qwen35_vl/run_python_dump.py \
  --image tests/fixtures/p6_qwen35_vl/coco_sample.jpg \
  --out-dir $PY_DIR

# Step 2: ironmlx dump (consumes 00_pixel_values.safetensors)
QWEN35_MODEL=/Users/sam/.ironmlx/models/.../snapshots/.../ \
MLX_DIR=$HOME/.local/mlx \
IRONMLX_VISION_DUMP_DIR=$RUST_DIR \
PIXEL_VALUES_PATH=$PY_DIR/00_pixel_values.safetensors \
cargo test -p ironmlx --features vision-dump --release --test p6_vision_dump -- --ignored

# Step 3: diff + report
~/.venvs/mlxvlm-ref/bin/python tests/fixtures/p6_qwen35_vl/diff_pipeline.py \
  --py $PY_DIR --rust $RUST_DIR --out $REPORT_DIR

echo "Report written to $REPORT_DIR/report.md"
```

## 10. Error Handling

- **mlx-vlm side hook**: `_maybe_dump` is `try/except`-wrapped; a save failure logs to stderr and continues (we don't want a dump bug to crash the reference forward).
- **ironmlx side**: `dump_tensor` returns on any error path silently (env var unset, dir missing, save fail) — same rationale. Production build has zero impact (compiled out).
- **diff_pipeline**:
  - Missing pair (file in one dir but not other): warn, skip, list at end of report
  - Shape mismatch: error, abort with diagnosis line ("ironmlx X has shape A; mlx-vlm X has shape B — alignment broken before tensor X")
  - Empty dir: error, abort early

## 11. Testing

- **diff_pipeline unit tests** (Python, in `tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py`):
  - identical tensors → all diff = 0
  - synthetic offset tensor → max_diff matches the offset
  - shape mismatch → raises with clear message
- **Integration**: run `run_p6_1_diff.sh` once; verify `report.md` and `max_diff_curve.png` exist; verify the report contains all 30 tensors and a "first rupture point" line.

## 12. Acceptance

P6.1 is **complete** when:

1. `run_p6_1_diff.sh` runs end-to-end on a fresh checkout with no manual intervention beyond setting `MLX_DIR` and `QWEN35_MODEL`.
2. `report.md` lists per-tensor stats for all 30 dump points.
3. Report identifies a specific "first rupture" tensor (the layer where max_diff jumps by > 5×).
4. `max_diff_curve.png` is generated and shows the divergence visually.
5. The follow-up P6.2 spec can be authored using only the contents of this report (no further re-runs needed for diagnosis).

## 13. Out of Scope (deferred to P6.2 or later)

- Any fix to `vision/*.rs` implementation
- Op-level (sub-block) dump granularity — only if P6.1's coarse report cannot localize the rupture
- Multi-image diff
- LM-side dump (Task 21 evidence shows LM is fine)
- Performance regression test
- Upstream contribution of dump hooks to mlx-vlm

## 14. Estimated Effort

| Task | Estimate |
| --- | --- |
| mlx-vlm fork + hook injection at 30 points | 1.5h |
| run_python_dump.py | 0.5h |
| ironmlx Cargo feature + dump_tensor helper | 1h |
| Insert dump_tensor calls at 30 sites | 1.5h |
| p6_vision_dump integration test | 0.5h |
| diff_pipeline.py | 2h |
| diff_pipeline unit tests | 0.5h |
| run_p6_1_diff.sh + end-to-end validation | 1h |
| Report formatting polish | 0.5h |
| **Total** | **~9h** |

One working day, single implementer.
