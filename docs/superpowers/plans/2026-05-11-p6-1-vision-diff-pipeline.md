# P6.1 Vision Encoder Diff Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an automated, reproducible diff pipeline that pinpoints the layer/op where ironmlx's vision encoder diverges from mlx-vlm's reference path. Produces a markdown report + max_diff curve. No fix to vision encoder in this plan — fix is deferred to P6.2 spec.

**Architecture:** Three independent units exporting via filesystem: (1) mlx-vlm fork with hooks gated on env `MLXVLM_VISION_DUMP_DIR`; (2) ironmlx `vision-dump` Cargo feature gated on env `IRONMLX_VISION_DUMP_DIR`; (3) Python `diff_pipeline.py` pairs the two dumps and emits the report. Each unit's contract is the filename → tensor mapping in spec §5.2.

**Tech Stack:** Rust (ironmlx) + Python 3.11 (mlx-vlm venv) + mlx-rs / mlx Python + safetensors + matplotlib.

**Spec:** `docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md` (commit `e2b08ce`)

**Branch base:** `ironmlx-p6-vl` (last commit `e2b08ce`)
**Branch target:** `ironmlx-p6-1-vision-diff`

---

## File Structure

Created or modified by this plan:

- `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` (modify; local fork)
- `ironmlx/Cargo.toml` (modify; add feature)
- `ironmlx/src/models/qwen3_5/vision/dump.rs` (create; dump helper)
- `ironmlx/src/models/qwen3_5/vision/mod.rs` (modify; insert dump calls)
- `ironmlx/tests/p6_vision_dump.rs` (create; integration test)
- `ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py` (create; mlx-vlm driver)
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py` (create; Python diff)
- `ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py` (create; Python unit tests)
- `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh` (create; top-level driver)
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/` (create dir; gitignored except `.gitkeep` and final example report)

Test files added to `.gitignore`:
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/*/` (per-run artifacts are local; final report optional commit)

---

## Task 1: Branch setup + sanity checks

**Files:**
- No new files.

- [ ] **Step 1.1: Create branch from `ironmlx-p6-vl`**

```bash
cd /Volumes/Dev/cxx-mlx
git checkout ironmlx-p6-vl
git checkout -b ironmlx-p6-1-vision-diff
git log -1 --oneline   # expect: e2b08ce docs(p6.1): ...
```

- [ ] **Step 1.2: Verify `mlx::io::save_safetensors` exists in the bundled mlx Rust crate**

```bash
grep -n "save_safetensors\|fn save_" /Volumes/Dev/cxx-mlx/mlx/src/io/mod.rs /Volumes/Dev/cxx-mlx/mlx/src/io.rs 2>/dev/null | head -5
```

Expected: at least one match. If zero matches, run a wider search:

```bash
grep -rn "pub fn save_safetensors\|pub fn save_" /Volumes/Dev/cxx-mlx/mlx/src/ 2>/dev/null | head -10
```

If still nothing, dump module uses `.bin` + `.json` manifest fallback (per spec §7 implementation note). Record the choice in `dump.rs` module docstring.

- [ ] **Step 1.3: Verify `mlx-vlm` venv runs the existing fixture script**

```bash
ls -lh /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/gen_fixture.py
~/.venvs/mlxvlm-ref/bin/python -c "import mlx_vlm; from mlx_vlm import load; print(load.__module__)"
```

Expected: `mlx_vlm.utils`.

- [ ] **Step 1.4: No commit yet**

This task verifies preconditions; no source changes.

---

## Task 2: ironmlx — Cargo feature + dump helper module

**Files:**
- Modify: `ironmlx/Cargo.toml`
- Create: `ironmlx/src/models/qwen3_5/vision/dump.rs`
- Modify: `ironmlx/src/models/qwen3_5/vision/mod.rs` (declare submodule only)

- [ ] **Step 2.1: Add `vision-dump` feature to Cargo.toml**

Open `ironmlx/Cargo.toml`. Find the `[features]` section (or create it just before `[dependencies]`). Add:

```toml
[features]
default = []
vision-dump = []
```

If `[features]` already exists, append `vision-dump = []` and ensure `default` is preserved.

- [ ] **Step 2.2: Create `dump.rs` with feature-gated `dump_tensor` helper**

Create `ironmlx/src/models/qwen3_5/vision/dump.rs` with:

```rust
//! Compile-gated tensor dump for the P6.1 diff pipeline.
//!
//! When the `vision-dump` cargo feature is OFF (default / production builds),
//! [`dump_tensor`] is a `#[inline] fn _: () {}` no-op that the compiler erases.
//! When the feature is ON, the function reads `IRONMLX_VISION_DUMP_DIR` and,
//! if set, eagerly evaluates and saves the tensor as
//! `<dir>/<name>.safetensors`. See spec
//! `docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md`.

use mlx::Array;

#[cfg(feature = "vision-dump")]
pub fn dump_tensor(name: &str, t: &Array) {
    use std::env;
    let Ok(dir) = env::var("IRONMLX_VISION_DUMP_DIR") else {
        return;
    };
    if let Err(e) = mlx::transforms::eval(&[t]) {
        eprintln!("[vision-dump] eval {name} failed: {e}");
        return;
    }
    let path = format!("{dir}/{name}.safetensors");
    let mut map = std::collections::HashMap::new();
    map.insert("tensor".to_string(), t.clone());
    if let Err(e) = mlx::io::save_safetensors(&path, &map) {
        eprintln!("[vision-dump] save {name} failed: {e}");
    }
}

#[cfg(not(feature = "vision-dump"))]
#[inline(always)]
pub fn dump_tensor(_: &str, _: &Array) {}
```

**Implementation note from Task 1.2:** if `mlx::io::save_safetensors` does NOT exist, replace the `save_safetensors` body with:

```rust
    // Fallback: raw little-endian bytes + sibling .json manifest
    let bytes: Vec<u8> = match t.to_vec::<u8>() { /* see fallback in Task 1.2 notes */ };
    std::fs::write(&path, &bytes).ok();
    let manifest = serde_json::json!({"shape": t.shape().as_slice(), "dtype": format!("{:?}", t.dtype())});
    std::fs::write(format!("{dir}/{name}.json"), manifest.to_string()).ok();
```

(Concrete fallback implementation deferred to actual encounter — only inserted if Task 1.2 found no saver.)

- [ ] **Step 2.3: Wire submodule into `vision/mod.rs`**

Open `ironmlx/src/models/qwen3_5/vision/mod.rs`. Near the top (after the file's doc comment), add (inserting alphabetically with existing `pub mod` lines):

```rust
pub mod block;
pub mod dump;
pub mod merger;
pub mod patch_embed;
```

(The exact existing `pub mod` lines are `block`, `merger`, `patch_embed` — verify and merge.)

- [ ] **Step 2.4: Verify default build compiles (no dump calls yet)**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
```

Expected: `Finished \`release\` profile [optimized] target(s) in <N>s`.

- [ ] **Step 2.5: Verify vision-dump feature compiles**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release --features vision-dump 2>&1 | tail -3
```

Expected: same `Finished` line.

- [ ] **Step 2.6: Run CI gauntlet on both build modes**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --all-features -- -D warnings 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx -- -D warnings 2>&1 | tail -5
```

Expected: all three pass with no errors.

- [ ] **Step 2.7: Commit**

```bash
git add ironmlx/Cargo.toml ironmlx/src/models/qwen3_5/vision/dump.rs ironmlx/src/models/qwen3_5/vision/mod.rs
git commit -m "feat(p6.1): vision-dump cargo feature + dump_tensor helper"
```

---

## Task 3: ironmlx — insert `dump_tensor` calls at 29 sites

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/vision/mod.rs` (`VisionTower::forward` + `add_learned_pos_embed`)

**Note:** ironmlx does not dump `00_pixel_values` (per spec §5.3 mlx-vlm is source of truth). 29 dumps: `01..29`.

- [ ] **Step 3.1: Read current `vision/mod.rs::forward` impl**

```bash
sed -n '/pub fn forward/,/^    fn /p' /Volumes/Dev/cxx-mlx/ironmlx/src/models/qwen3_5/vision/mod.rs | head -40
```

Note the current call sequence: `patch_embed.forward → add_learned_pos_embed → compute_rotary_pos_emb → 24× blocks → merger.forward`.

- [ ] **Step 3.2: Insert dumps in `forward` for points 01, 03, 04, 5..28, 29**

Edit `VisionTower::forward` to insert dumps. The final form should be:

```rust
pub fn forward(&self, pixel_values: &Array, grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
    let mut x = self.patch_embed.forward(pixel_values)?;
    super::vision::dump::dump_tensor("01_patch_embed_out", &x);

    x = self.add_learned_pos_embed(&x, grid_thw)?;
    super::vision::dump::dump_tensor("03_after_pos_embed", &x);

    let rotary = self.compute_rotary_pos_emb(grid_thw)?;
    super::vision::dump::dump_tensor("04_rotary_freqs", &rotary);

    let cu_seqlens: Vec<i32> = {
        let mut v = vec![0_i32];
        let mut total = 0_i32;
        for (t, h, w) in grid_thw {
            total += t * h * w;
            v.push(total);
        }
        v
    };

    for (i, blk) in self.blocks.iter().enumerate() {
        x = blk.forward(&x, &rotary, &cu_seqlens)?;
        super::vision::dump::dump_tensor(&format!("{:02}_block_{:02}_out", 5 + i, i), &x);
    }

    let out = self.merger.forward(&x, grid_thw)?;
    super::vision::dump::dump_tensor("29_merger_out", &out);
    Ok(out)
}
```

Notes:
- `super::vision::dump::dump_tensor` resolves because `forward` is in `vision/mod.rs` and `dump` is a submodule. If that path is wrong inside the same file, use `dump::dump_tensor` directly.
- Use `super::dump::dump_tensor` if the function lives at the crate-root level; use the simplest path that compiles. Settle the import path at the top of `mod.rs`: `use self::dump::dump_tensor;` then call sites become just `dump_tensor(...)`.

Cleanup pass: at top of `vision/mod.rs`, add `use self::dump::dump_tensor;` so call sites are just `dump_tensor("01_patch_embed_out", &x);`.

- [ ] **Step 3.3: Insert dump in `add_learned_pos_embed` for point 02**

Find `add_learned_pos_embed`. Locate the line that produces the final `pos_embed_all` (the bilinear-interp result, before adding to `x`). Insert immediately after that line, before the final addition:

```rust
    // ... existing code that builds pos_embed_all ...
    let pos_embed_all = ops::concatenate(&refs, 0)?;
    dump_tensor("02_pos_embed_contrib", &pos_embed_all);

    let result = &x_cast + &pos_embed_all;
    Ok(result)
```

(Variable names should match the existing code; the key is `dump_tensor("02_pos_embed_contrib", &<the_final_pos_embed_tensor>);` placed before the `+` add.)

- [ ] **Step 3.4: Verify default-build still compiles (dump is no-op)**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
```

Expected: `Finished` line.

- [ ] **Step 3.5: Verify feature-build compiles**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release --features vision-dump 2>&1 | tail -3
```

Expected: `Finished` line.

- [ ] **Step 3.6: Run all vision unit tests on default build (regression)**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release vision 2>&1 | tail -10
```

Expected: all vision tests pass (no regression — dumps are no-op).

- [ ] **Step 3.7: Run all unit tests with the feature on (still no regression)**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release --features vision-dump 2>&1 | tail -5
```

Expected: 151 passed; 0 failed (or whatever the current baseline is).

- [ ] **Step 3.8: CI gauntlet on both modes**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx -- -D warnings 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --features vision-dump -- -D warnings 2>&1 | tail -5
```

All pass.

- [ ] **Step 3.9: Commit**

```bash
git add ironmlx/src/models/qwen3_5/vision/mod.rs
git commit -m "feat(p6.1): insert dump_tensor calls at 29 vision-encoder sites"
```

---

## Task 4: ironmlx — integration test driver (`p6_vision_dump.rs`)

**Files:**
- Create: `ironmlx/tests/p6_vision_dump.rs`

- [ ] **Step 4.1: Write the integration test**

Create `ironmlx/tests/p6_vision_dump.rs`:

```rust
//! P6.1 vision dump integration test.
//!
//! Driven by `IRONMLX_VISION_DUMP_DIR`, `QWEN35_MODEL`, and `PIXEL_VALUES_PATH`
//! env vars set by `tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh`. Reads the
//! mlx-vlm-prepared `00_pixel_values.safetensors`, drives one forward pass
//! through `VisionTower`, and as a side effect causes the 29 `dump_tensor`
//! sites in `vision/mod.rs` to write their tensors into the dump dir.

#![cfg(feature = "vision-dump")]

use std::path::Path;

use mlx::Array;

use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::config::Qwen35Config;
use ironmlx::models::qwen3_5::vision::VisionTower;

#[test]
#[ignore] // requires QWEN35_MODEL + PIXEL_VALUES_PATH + IRONMLX_VISION_DUMP_DIR
fn p6_vision_dump() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let pv_path = std::env::var("PIXEL_VALUES_PATH").expect("PIXEL_VALUES_PATH env required");
    let _dump_dir =
        std::env::var("IRONMLX_VISION_DUMP_DIR").expect("IRONMLX_VISION_DUMP_DIR env required");

    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let vc = cfg.vision_config.expect("vision_config");
    let tower = VisionTower::from_loader(&loader, &vc).expect("tower");

    // mlx-vlm preprocesses to [N, 1536] (flattened) — reshape to [N, 2, 16, 16, 3]
    // then transpose to [N, 2, 3, 16, 16] to match VisionTower input convention.
    let pv_flat: Array = mlx::io::load_safetensors(&pv_path)
        .expect("load pixel_values")
        .remove("tensor")
        .expect("tensor key");
    let n = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat.reshape(&[n, 2, 16, 16, 3][..]).expect("reshape pv");
    let pv = mlx::ops::shape::transpose_axes(&pv_5d, &[0, 1, 4, 2, 3]).expect("transpose pv");

    // Grid for the COCO sample, see Task 20 gen_fixture output: image_grid_thw = [[1, 30, 40]]
    let grids: Vec<(i32, i32, i32)> = vec![(1, 30, 40)];

    let _embeds = tower.forward(&pv, &grids).expect("vision forward");
    // Force eval so all 29 dump calls complete before the test exits.
    mlx::transforms::eval(&[&_embeds]).expect("eval embeds");

    eprintln!("[p6_vision_dump] forward complete; dumps should be in IRONMLX_VISION_DUMP_DIR");
}
```

- [ ] **Step 4.2: Smoke-run the test with explicit env to verify it loads**

```bash
mkdir -p /tmp/p6_diff/rust
# Use the existing expected_pixel_values.npy → temporarily convert; in real run
# this comes from Task 6's run_python_dump.py. For now, smoke test only verifies
# the test compiles and finds the model.
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
PIXEL_VALUES_PATH=/dev/null \
IRONMLX_VISION_DUMP_DIR=/tmp/p6_diff/rust \
cargo test -p ironmlx --features vision-dump --test p6_vision_dump --release -- --ignored 2>&1 | tail -20
```

Expected: test compiles. It WILL fail at `load_safetensors(/dev/null)` because there's no real pixel_values yet — that's expected. We're just confirming the test wiring is good. Look for "compiles" + clean error about load failure (not a compile error).

- [ ] **Step 4.3: CI gauntlet**

```bash
cargo fmt -p ironmlx -- --check
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy -p ironmlx --features vision-dump --tests -- -D warnings 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 4.4: Commit**

```bash
git add ironmlx/tests/p6_vision_dump.rs
git commit -m "feat(p6.1): p6_vision_dump integration test driver"
```

---

## Task 5: mlx-vlm fork — vision.py hook injection

**Files:**
- Modify: `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py`

This is a local fork. **Do not push to upstream.** Hook is no-op when env var is unset, so safe to leave in fork.

- [ ] **Step 5.1: Add `_maybe_dump` helper at top of vision.py**

Open `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py`. After the imports (around line 10–20, just below `import mlx.core as mx`), insert:

```python
import os as _os

_VISION_DUMP_DIR = _os.environ.get("MLXVLM_VISION_DUMP_DIR")

def _maybe_dump(name: str, tensor):
    """P6.1 diff hook. No-op unless MLXVLM_VISION_DUMP_DIR is set.

    See /Volumes/Dev/cxx-mlx/docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md
    """
    if not _VISION_DUMP_DIR:
        return
    try:
        mx.eval(tensor)
        mx.save_safetensors(
            _os.path.join(_VISION_DUMP_DIR, f"{name}.safetensors"),
            {"tensor": tensor},
        )
    except Exception as e:
        import sys
        print(f"[mlxvlm-dump] {name} failed: {e}", file=sys.stderr)
```

(`import sys` inside the except is local to keep the no-op fast path clean.)

- [ ] **Step 5.2: Insert hooks in the vision tower forward path**

mlx-vlm's `Qwen3_VLVisionModel.__call__` (or whatever the top-level class is named in this version — likely `Qwen3VLVisionModel.__call__` or `Qwen2_5_VLVisionModel.__call__`) is the integration point. Skim the file:

```bash
grep -n "class.*[Vv]ision.*:\|def __call__\|def forward" /Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py | head -30
```

Identify the top-level `__call__` that wraps patch_embed → fast_pos_embed_interpolate → rot_pos_emb → blocks → merger.

Hook insertion points (insert immediately AFTER the producing line):

| Site | Insert after | Dump call |
| --- | --- | --- |
| 01 | `hidden_states = self.patch_embed(hidden_states)` | `_maybe_dump("01_patch_embed_out", hidden_states)` |
| 02 | inside `fast_pos_embed_interpolate`, after the `mx.concatenate(patch_pos_embeds_permute)` final line, BEFORE `return` | `_maybe_dump("02_pos_embed_contrib", patch_pos_embeds)` then `return patch_pos_embeds` |
| 03 | `hidden_states = hidden_states + pos_embeds` | `_maybe_dump("03_after_pos_embed", hidden_states)` |
| 04 | `rotary_pos_emb = self.rot_pos_emb(grid_thw)` | `_maybe_dump("04_rotary_freqs", rotary_pos_emb)` |
| 05..28 | inside the block loop `for layer_num, blk in enumerate(self.blocks):` after `hidden_states = blk(...)` | `_maybe_dump(f"{5+layer_num:02d}_block_{layer_num:02d}_out", hidden_states)` |
| 29 | after the final `merger` call (likely `hidden_states = self.merger(hidden_states)`) | `_maybe_dump("29_merger_out", hidden_states)` |

Edit each site. Example for site 01:

```python
hidden_states = self.patch_embed(hidden_states)
_maybe_dump("01_patch_embed_out", hidden_states)
```

For site 02 (inside `fast_pos_embed_interpolate`), find the `return` at the end and split:

```python
patch_pos_embeds = mx.concatenate(patch_pos_embeds_permute)
_maybe_dump("02_pos_embed_contrib", patch_pos_embeds)
return patch_pos_embeds
```

For site 05..28 (inside the block loop):

```python
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
    )
    _maybe_dump(f"{5+layer_num:02d}_block_{layer_num:02d}_out", hidden_states)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature_lists.append(...)  # existing code
```

(Exact existing block-loop body should be preserved; only the `_maybe_dump` line is added.)

For site 29 (final return):

```python
hidden_states = self.merger(hidden_states)
_maybe_dump("29_merger_out", hidden_states)
return hidden_states, deepstack_feature_lists  # or whatever the existing return is
```

(Preserve exact existing return signature.)

- [ ] **Step 5.3: Verify the fork still imports cleanly with env unset**

```bash
unset MLXVLM_VISION_DUMP_DIR
~/.venvs/mlxvlm-ref/bin/python -c "
from mlx_vlm.models.qwen3_vl.vision import Qwen3VLMoEVisionBlock
print('import ok')
"
```

(Adjust class name if it differs.)

Expected: `import ok`.

- [ ] **Step 5.4: Verify hook is no-op when env is unset**

```bash
unset MLXVLM_VISION_DUMP_DIR
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/gen_fixture.py 2>&1 | tail -5
```

Expected: script completes as before (Task 20 produced first_token=760). No regression from added hooks.

- [ ] **Step 5.5: No git commit in cxx-mlx**

The mlx-vlm fork lives outside the cxx-mlx repo. To bookmark the change, capture the diff:

```bash
cd /Volumes/Dev/mlx-vlm
git diff mlx_vlm/models/qwen3_vl/vision.py > /tmp/mlx_vlm_dump_hooks.patch
ls -lh /tmp/mlx_vlm_dump_hooks.patch
```

Move/copy that patch file into the cxx-mlx repo for archival:

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches
cp /tmp/mlx_vlm_dump_hooks.patch /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/01_dump_hooks.patch
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/01_dump_hooks.patch
git commit -m "chore(p6.1): archive mlx-vlm dump-hook patch (local fork)"
```

The patch can be re-applied via `git apply` if the mlx-vlm fork is ever reset.

---

## Task 6: mlx-vlm driver — `run_python_dump.py`

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py`

- [ ] **Step 6.1: Write the driver**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py`:

```python
#!/usr/bin/env python
"""P6.1 mlx-vlm-side vision dump driver.

Reads an image, runs mlx-vlm's preprocess to produce pixel_values, saves
pixel_values as `00_pixel_values.safetensors`, then runs the vision tower
once with MLXVLM_VISION_DUMP_DIR set so hooks in vision.py write tensors
`01..29` into the same dir.

Requires the mlx-vlm fork at /Volumes/Dev/mlx-vlm with the dump hooks
(see Task 5).

Usage:
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
    ~/.venvs/mlxvlm-ref/bin/python run_python_dump.py \
        --image coco_sample.jpg \
        --out-dir /tmp/p6_diff/python
"""
import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import prepare_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Prompt only used to drive prepare_inputs; not relevant to dumps.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = os.environ.get("QWEN35_MODEL")
    if not model_dir:
        print("ERROR: QWEN35_MODEL env required", file=sys.stderr)
        return 1

    # Load model + processor
    model, processor = load(model_dir)
    config = model.config
    image_token_id = config.image_token_id

    # Build inputs (input_ids + pixel_values + image_grid_thw + attention_mask)
    inputs = prepare_inputs(
        processor=processor,
        prompts=[args.prompt],
        images=[[str(args.image)]],
        videos=None,
        image_token=processor.tokenizer.decode([image_token_id]),
        video_token=None,
    )
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    # Save pixel_values as the 00 dump file (source of truth, ironmlx consumes it)
    mx.eval(pixel_values)
    mx.save_safetensors(
        str(args.out_dir / "00_pixel_values.safetensors"),
        {"tensor": pixel_values.astype(mx.bfloat16)},
    )
    print(f"[run_python_dump] saved 00_pixel_values shape={pixel_values.shape}")

    # Set dump env so the forked vision.py hooks write 01..29
    os.environ["MLXVLM_VISION_DUMP_DIR"] = str(args.out_dir)

    # Forward through the vision tower
    embeds = model.visual(pixel_values, grid_thw=grid_thw)
    mx.eval(embeds)
    print(f"[run_python_dump] forward complete; embeds shape={embeds.shape}")

    # Verify all 30 files now exist
    expected = (
        ["00_pixel_values", "01_patch_embed_out", "02_pos_embed_contrib",
         "03_after_pos_embed", "04_rotary_freqs"]
        + [f"{5+i:02d}_block_{i:02d}_out" for i in range(24)]
        + ["29_merger_out"]
    )
    missing = [n for n in expected if not (args.out_dir / f"{n}.safetensors").exists()]
    if missing:
        print(f"ERROR: missing dump files: {missing}", file=sys.stderr)
        return 2
    print(f"[run_python_dump] all 30 dump files present in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.2: Run it end-to-end (this exercises Task 5 hooks)**

```bash
mkdir -p /tmp/p6_diff/python
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py \
    --image /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg \
    --out-dir /tmp/p6_diff/python 2>&1 | tail -10
```

Expected: `[run_python_dump] all 30 dump files present in /tmp/p6_diff/python`

- [ ] **Step 6.3: Verify file count and sizes**

```bash
ls /tmp/p6_diff/python/*.safetensors | wc -l   # expect: 30
ls -lh /tmp/p6_diff/python/01_patch_embed_out.safetensors   # expect: ~2.5MB (1200×1024×2 bytes bf16)
```

- [ ] **Step 6.4: Commit driver**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py
git commit -m "feat(p6.1): mlx-vlm dump driver run_python_dump.py"
```

---

## Task 7: diff_pipeline.py — core stats + unit tests

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py`

TDD: write tests first, then the implementation that makes them pass. Use the mlx-vlm venv since it already has numpy + matplotlib + mlx.

- [ ] **Step 7.1: Write the unit test file**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py`:

```python
"""Unit tests for diff_pipeline.py.

Run from within the cxx-mlx repo:
    ~/.venvs/mlxvlm-ref/bin/python -m pytest \
        ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py -v
"""
import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))
import diff_pipeline


def _save(tmp: Path, name: str, arr) -> None:
    if isinstance(arr, np.ndarray):
        arr = mx.array(arr)
    mx.eval(arr)
    mx.save_safetensors(str(tmp / f"{name}.safetensors"), {"tensor": arr})


def test_diff_stats_identical():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == 0.0
    assert s["mean"] == 0.0
    assert s["count_above_1e-3"] == 0
    assert s["total"] == 16


def test_diff_stats_offset():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 0.5, dtype=np.float32)
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == pytest.approx(0.5)
    assert s["mean"] == pytest.approx(0.5)
    assert s["count_above_1e-3"] == 16
    assert s["count_above_1e-2"] == 16
    assert s["count_above_1e-1"] == 16


def test_diff_stats_single_outlier():
    a = np.zeros((10,), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)
    b[3] = 0.85
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == pytest.approx(0.85)
    assert s["count_above_1e-1"] == 1
    assert s["count_above_1e-2"] == 1


def test_pair_files_skips_unpaired_with_warning(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        py_dir = Path(tmp) / "py"
        rust_dir = Path(tmp) / "rust"
        py_dir.mkdir()
        rust_dir.mkdir()
        _save(py_dir, "00_pixel_values", np.zeros((2, 2), dtype=np.float32))
        _save(py_dir, "01_patch_embed_out", np.zeros((2, 2), dtype=np.float32))
        _save(rust_dir, "01_patch_embed_out", np.zeros((2, 2), dtype=np.float32))
        # No matching 00 in rust → should be skipped with warning
        pairs, unpaired = diff_pipeline.pair_files(py_dir, rust_dir)
        assert len(pairs) == 1
        assert pairs[0][0] == "01_patch_embed_out"
        assert "00_pixel_values" in unpaired["py_only"]


def test_pair_files_shape_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        py_dir = Path(tmp) / "py"
        rust_dir = Path(tmp) / "rust"
        py_dir.mkdir()
        rust_dir.mkdir()
        _save(py_dir, "01_foo", np.zeros((4, 4), dtype=np.float32))
        _save(rust_dir, "01_foo", np.zeros((4, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="shape mismatch"):
            diff_pipeline.diff_pair(py_dir / "01_foo.safetensors",
                                    rust_dir / "01_foo.safetensors")


def test_top_outliers_returns_top_n():
    a = np.zeros((10,), dtype=np.float32)
    b = np.array([0.0, 0.1, 0.3, 0.5, 0.0, 0.2, 0.4, 0.0, 0.6, 0.0],
                 dtype=np.float32)
    out = diff_pipeline.top_outliers(a, b, n=3)
    assert len(out) == 3
    # Sorted descending by abs diff
    assert out[0]["idx"] == 8 and out[0]["diff"] == pytest.approx(0.6)
    assert out[1]["idx"] == 3 and out[1]["diff"] == pytest.approx(0.5)
    assert out[2]["idx"] == 6 and out[2]["diff"] == pytest.approx(0.4)


def test_render_report_contains_required_sections(tmp_path):
    rows = [
        {"name": "01_patch_embed_out", "shape": [1200, 1024],
         "max": 0.001, "mean": 0.0001, "p99": 0.001,
         "count_above_1e-3": 12, "count_above_1e-2": 0, "count_above_1e-1": 0,
         "total": 1228800},
        {"name": "10_block_05_out", "shape": [1200, 1024],
         "max": 0.156, "mean": 0.01, "p99": 0.1,
         "count_above_1e-3": 50000, "count_above_1e-2": 100, "count_above_1e-1": 5,
         "total": 1228800},
    ]
    text = diff_pipeline.render_report(rows, rupture="10_block_05_out", top_outliers=[])
    assert "## Summary" in text
    assert "## Per-tensor table" in text
    assert "10_block_05_out" in text
```

- [ ] **Step 7.2: Run the tests — verify they fail (diff_pipeline.py does not exist yet)**

```bash
~/.venvs/mlxvlm-ref/bin/python -m pytest /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'diff_pipeline'` or `ImportError`.

- [ ] **Step 7.3: Write the diff_pipeline.py implementation**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py`:

```python
"""P6.1 vision encoder diff pipeline.

Reads ironmlx + mlx-vlm side-by-side `.safetensors` dumps, pairs them by
basename, computes per-tensor diff stats, identifies the first significant
divergence ("rupture"), and emits a markdown report + max_diff curve PNG.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_pipeline.py \
        --py /tmp/p6_diff/python \
        --rust /tmp/p6_diff/rust \
        --out tests/fixtures/p6_qwen35_vl/diff_reports/2026-05-11

See spec docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np


def load_tensor(path: Path) -> np.ndarray:
    """Load a .safetensors file produced by either mlx-vlm or ironmlx and
    return it as a numpy float32 array (cast from whatever dtype was on disk).
    """
    arr = mx.load(str(path))
    if isinstance(arr, dict):
        t = arr["tensor"]
    else:
        t = arr
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


def top_outliers(a: np.ndarray, b: np.ndarray, n: int = 5) -> list[dict]:
    d = np.abs(a.flatten() - b.flatten())
    if len(d) == 0:
        return []
    k = min(n, len(d))
    idxs = np.argpartition(d, -k)[-k:]
    idxs = idxs[np.argsort(d[idxs])[::-1]]
    return [
        {
            "idx": int(i),
            "a_val": float(a.flatten()[i]),
            "b_val": float(b.flatten()[i]),
            "diff": float(d[i]),
        }
        for i in idxs
    ]


def pair_files(py_dir: Path, rust_dir: Path) -> tuple[list[tuple[str, Path, Path]], dict]:
    """Pair .safetensors files by basename. Returns (paired_list, unpaired_dict)."""
    py_files = {p.stem: p for p in sorted(py_dir.glob("*.safetensors"))}
    rust_files = {p.stem: p for p in sorted(rust_dir.glob("*.safetensors"))}
    common = sorted(set(py_files) & set(rust_files))
    pairs = [(name, py_files[name], rust_files[name]) for name in common]
    unpaired = {
        "py_only": sorted(set(py_files) - set(rust_files)),
        "rust_only": sorted(set(rust_files) - set(py_files)),
    }
    if unpaired["py_only"]:
        print(f"[diff_pipeline] py-only (skipped): {unpaired['py_only']}", file=sys.stderr)
    if unpaired["rust_only"]:
        print(f"[diff_pipeline] rust-only (skipped): {unpaired['rust_only']}", file=sys.stderr)
    return pairs, unpaired


def diff_pair(py_path: Path, rust_path: Path) -> dict:
    a = load_tensor(py_path)
    b = load_tensor(rust_path)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch for {py_path.stem}: python {a.shape} vs rust {b.shape}"
        )
    stats = diff_stats(a, b)
    stats["name"] = py_path.stem
    stats["shape"] = list(a.shape)
    stats["_a"] = a  # carried only when caller asks for outliers
    stats["_b"] = b
    return stats


def find_rupture(rows: list[dict], factor: float = 5.0) -> str | None:
    """Return the name of the first tensor whose max_diff jumps `factor`× over
    the previous max_diff. If no such jump, return the tensor with the largest
    max_diff overall."""
    prev = None
    for r in rows:
        if prev is not None and r["max"] > factor * max(prev, 1e-6):
            return r["name"]
        prev = r["max"]
    # Fallback: max
    return max(rows, key=lambda r: r["max"])["name"] if rows else None


def render_report(rows: list[dict], rupture: str | None, top_outliers: list[dict]) -> str:
    lines = ["# P6 VL Vision Encoder Diff Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total tensors compared: {len(rows)}")
    if rupture:
        lines.append(f"- First rupture point: **{rupture}**")
    if rows:
        last = rows[-1]
        lines.append(f"- Final tensor `{last['name']}`: max_diff = {last['max']:.4f}, mean = {last['mean']:.6f}")
    lines.append("")
    lines.append("## Per-tensor table")
    lines.append("")
    lines.append("| # | tensor | shape | max | mean | p99 | >1e-3 | >1e-2 | >1e-1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows):
        lines.append(
            f"| {i} | `{r['name']}` | {r['shape']} | {r['max']:.4f} | {r['mean']:.6f} | {r['p99']:.4f} | "
            f"{r['count_above_1e-3']}/{r['total']} | {r['count_above_1e-2']}/{r['total']} | {r['count_above_1e-1']}/{r['total']} |"
        )
    lines.append("")
    if top_outliers:
        lines.append("## Top outliers in final tensor")
        lines.append("")
        lines.append("| flat_idx | mlx-vlm | ironmlx | abs_diff |")
        lines.append("| --- | --- | --- | --- |")
        for o in top_outliers:
            lines.append(
                f"| {o['idx']} | {o['a_val']:.4f} | {o['b_val']:.4f} | {o['diff']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_curve(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(rows)))
    ys = [max(r["max"], 1e-9) for r in rows]
    labels = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("max_abs_diff (log)")
    ax.set_title("P6 VL Vision Encoder: ironmlx vs mlx-vlm")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs, unpaired = pair_files(args.py, args.rust)
    if not pairs:
        print("ERROR: no paired tensors found", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for name, py_p, rust_p in pairs:
        try:
            r = diff_pair(py_p, rust_p)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        rows.append(r)

    rupture = find_rupture(rows)

    # Top outliers in the final tensor (merger output)
    final = rows[-1]
    final_top = top_outliers(final["_a"], final["_b"], n=5)

    # Strip the carried arrays before serializing
    rows_clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    report = render_report(rows_clean, rupture, final_top)
    (args.out / "report.md").write_text(report)
    plot_curve(rows_clean, args.out / "max_diff_curve.png")
    (args.out / "outliers.json").write_text(
        json.dumps({"final_tensor": final["name"], "top": final_top, "unpaired": unpaired}, indent=2)
    )

    print(f"[diff_pipeline] report → {args.out}/report.md")
    print(f"[diff_pipeline] curve  → {args.out}/max_diff_curve.png")
    print(f"[diff_pipeline] rupture: {rupture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.4: Run the unit tests — verify they all pass**

```bash
~/.venvs/mlxvlm-ref/bin/python -m pytest /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 7.5: Commit**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py
git commit -m "feat(p6.1): diff_pipeline.py + unit tests"
```

---

## Task 8: Driver script — `run_p6_1_diff.sh`

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh`
- Modify: `ironmlx/tests/fixtures/p6_qwen35_vl/.gitignore` (add `diff_reports/*/`)

- [ ] **Step 8.1: Write the driver script**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh`:

```bash
#!/usr/bin/env bash
# P6.1 end-to-end diff pipeline orchestrator.
#
# Required env:
#   MLX_DIR        — mlx C++ install (e.g. $HOME/.local/mlx)
#   QWEN35_MODEL   — local snapshot path of Qwen3.5-4B-MLX-4bit
#
# Optional env:
#   PY_DIR=/tmp/p6_diff/python   — where mlx-vlm dumps land
#   RUST_DIR=/tmp/p6_diff/rust   — where ironmlx dumps land
#
# Produces:
#   ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/<YYYY-MM-DD-HHMM>/
#       report.md + max_diff_curve.png + outliers.json
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
PY_DIR="${PY_DIR:-/tmp/p6_diff/python}"
RUST_DIR="${RUST_DIR:-/tmp/p6_diff/rust}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$REPORT_DIR"

# Clean stale dumps from prior runs
rm -f "$PY_DIR"/*.safetensors "$RUST_DIR"/*.safetensors

echo "=== Step 1: mlx-vlm dump ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"

echo "=== Step 2: ironmlx dump ==="
cd "$REPO_ROOT"
QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
    PIXEL_VALUES_PATH="$PY_DIR/00_pixel_values.safetensors" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test p6_vision_dump \
        -- --ignored 2>&1 | tail -10

echo "=== Step 3: diff + report ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline.py" \
    --py "$PY_DIR" \
    --rust "$RUST_DIR" \
    --out "$REPORT_DIR"

echo "=== Done. Report: $REPORT_DIR/report.md ==="
```

- [ ] **Step 8.2: Make it executable**

```bash
chmod +x /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh
```

- [ ] **Step 8.3: Update .gitignore**

Open `ironmlx/tests/fixtures/p6_qwen35_vl/.gitignore`. Append:

```
# P6.1 diff outputs (regenerate via run_p6_1_diff.sh)
diff_reports/*/
!diff_reports/.gitkeep
```

Create `diff_reports/.gitkeep`:

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports
touch /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/.gitkeep
```

- [ ] **Step 8.4: Commit driver + gitignore**

```bash
git add ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh \
        ironmlx/tests/fixtures/p6_qwen35_vl/.gitignore \
        ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/.gitkeep
git commit -m "feat(p6.1): driver script run_p6_1_diff.sh + gitignore diff_reports"
```

---

## Task 9: End-to-end validation + final report commit

**Files:**
- (Generated) `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/<date>/report.md`
- May commit the first report as a baseline reference (optional)

- [ ] **Step 9.1: Run the full pipeline**

```bash
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
/Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh
```

Expected end of output: `=== Done. Report: <path>/report.md ===`.

- [ ] **Step 9.2: Verify all artifacts present**

```bash
REPORT_DIR=$(ls -dt /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/*/ | head -1)
ls -lh "$REPORT_DIR"
```

Expected: `report.md`, `max_diff_curve.png`, `outliers.json`.

- [ ] **Step 9.3: Sanity-check report contents**

```bash
head -40 "$REPORT_DIR/report.md"
```

Verify:
- Has `## Summary` section
- Has `## Per-tensor table` with 29 rows (00_pixel_values is py-only, skipped per spec §5.3)
- Reports a "First rupture point" with a specific tensor name
- Final tensor row is `29_merger_out` with the merger max_diff

- [ ] **Step 9.4: Sanity-check curve PNG**

```bash
file "$REPORT_DIR/max_diff_curve.png"   # expect: PNG image data
```

- [ ] **Step 9.5: Commit a reference report**

The actual diff_reports dir is gitignored per Task 8.3, but we want a sample report committed for documentation. Override the gitignore for the first run:

```bash
STAMP=$(basename "$REPORT_DIR")
git add -f "$REPORT_DIR/report.md" "$REPORT_DIR/max_diff_curve.png" "$REPORT_DIR/outliers.json"
git commit -m "docs(p6.1): baseline diff report — $STAMP"
```

- [ ] **Step 9.6: Confirm `cargo build` (default) is still clean — no production regression**

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx cargo build -p ironmlx --release 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: build succeeds; same number of passing tests as the pre-P6.1 baseline (151 + new p6_vision_dump = 152, with the new one in `0 ignored` when feature is off because `#![cfg(feature = "vision-dump")]` excludes the whole test).

- [ ] **Step 9.7: Push branch**

```bash
git log --oneline ironmlx-p6-vl..HEAD
# Should list 7-8 commits from Tasks 2-9
git push -u origin ironmlx-p6-1-vision-diff   # if Boss authorized push; otherwise local only
```

(Push is optional; do not push without explicit Boss approval per CLAUDE.md / Executing actions guidance.)

---

## Self-Review (run after writing this plan)

**Spec coverage check:** every section in `2026-05-11-p6-1-vision-diff-pipeline-design.md` is implemented somewhere in this plan:

- §4 Architecture → Tasks 2, 3, 5, 6 build the three units and the integration test in Task 4
- §5.1 File Format (.safetensors per tensor) → Tasks 5, 6 (Python), Task 2 (Rust)
- §5.2 Dump point names → Tasks 3 (Rust 29 sites), 5 (Python 30 sites)
- §5.3 Input Determinism (pixel_values from mlx-vlm) → Task 6 step 6.1 saves pixel_values; Task 4 reads it via `PIXEL_VALUES_PATH`
- §6 mlx-vlm fork → Task 5
- §7 ironmlx feature flag → Tasks 2, 3
- §8 diff_pipeline.py → Task 7
- §9 Driver script → Task 8
- §10 Error handling → Tasks 2 step 2.2 (try/except in dump_tensor), 5 step 5.1 (try/except in _maybe_dump), 7 step 7.3 (shape mismatch ValueError, unpaired warnings)
- §11 Testing → Task 7 step 7.1 (unit tests), Task 9 (integration)
- §12 Acceptance gates → Task 9 steps 9.1–9.4 verify all five gates

**Placeholder scan:** no "TBD", no "TODO", no "see somewhere else", no "similar to". Every code block is concrete. One exception: Task 2.2 references a `.bin + .json` fallback that's only inserted if Task 1.2 finds no `save_safetensors` — the conditional is explicit and the fallback structure is sketched concretely enough that an engineer hitting that branch can complete it.

**Type/path consistency:**
- `dump_tensor(name: &str, t: &Array)` is the same signature in Tasks 2 and 3
- Dump filenames follow the `NN_<stage>.safetensors` convention across Tasks 3, 5, 6, 7
- Path to `dump` module: `ironmlx::models::qwen3_5::vision::dump::dump_tensor` (qualified) or `dump_tensor` (after `use self::dump::dump_tensor;` at top of `vision/mod.rs`) — Task 3 step 3.2 settles this with a `use` statement at the top
- Cargo feature name `vision-dump` is consistent across Tasks 2, 3, 4, 8

No gaps found.

---

## Plan complete and saved to `docs/superpowers/plans/2026-05-11-p6-1-vision-diff-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review between tasks, fast iteration. Best fit for this plan because Tasks 2–5 are largely independent (mlx-vlm fork has no Rust deps; ironmlx feature scaffolding has no Python deps).

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Better for tight feedback if you want to watch Boss-style.

Which approach?
