# MiniCPM-V-4.6 VLM — P3 Multi-Slice + Multi-Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support MiniCPM-V-4.6's LLaVA-UHD adaptive image slicing (one high-res image → a source image + a grid of slices) and multiple images per prompt, end-to-end, matching mlx-vlm.

**Architecture:** Extend the single-image preprocessor (P2a) to the slicing pipeline (`get_sliced_grid` grid search → `slice_image` source+refine→split → per-slice pack, yielding a Vec of slices per image); build the sliced prompt-id convention (`<image>`source`</image>` + a grid of `<slice>`…`</slice>` with row newlines); wire CLI/serve to emit the multi-slice pixel-value list + grids + sliced prompt; verify end-to-end vs mlx-vlm. The model-level VL forward (per-slice vision via `compute_vision_embeds` loop + scatter via `replace_image_tokens` + sequential positions) already handles N image-pad runs from P2a/P2b — P3 mainly extends preprocessing + prompt construction + CLI/serve plumbing.

**Tech Stack:** Rust, cxx-mlx (`mlx`), MLX Metal, `image` crate. Reference (observation only): `/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/processing_minicpmv4_6.py`.

**Scope:** P3 of: P1 (vision stack ✓) → P2a (single-image model forward ✓) → P2b (single-image CLI/serve e2e ✓, commit 3492361) → **P3 (multi-slice + multi-image)**. This is the final phase + the biggest correctness risk (slicing grid search + per-slice pack + sliced-prompt convention). Boss: no further sub-splitting.

**Environment:** `source ~/.local/mlx/mlx-env.sh` first. Canonical clippy gate: `cargo +nightly clippy --all-features --workspace -- -D warnings` (NOT `--all-targets`). Integration tests `#[ignore]` + env-gated `MINICPMV46_MODEL`.

## Authoritative slicing algorithm (from processing_minicpmv4_6.py — observation, verify against the file)
- `scale_resolution = 448`, `patch_size = 14`, `max_slice_nums = 9`, `merge_factor = patch*4 = 56`. mean=std=[0.5,0.5,0.5].
- `_ensure_divide(len, d) = max(round(len/d)*d, d)`.
- `_find_best_resize((w,h), scale_res, patch, allow_upscale)`: if `w*h > scale_res²` or allow_upscale → `ratio=w/h; h=int(scale_res/sqrt(ratio)); w=int(h*ratio)`; then `best_w=_ensure_divide(w, 56)`, `best_h=_ensure_divide(h, 56)`. Returns `(best_w, best_h)`. (Note: width-first `(w,h)` tuple ordering, PIL convention.)
- `get_sliced_grid((w,h), max_slice_nums=9)`: `ratio=w*h/scale_res²`; `multiple=min(ceil(ratio), max_slice_nums)`; if `multiple<=1` → `None` (no slicing). Else candidates = {multiple-1, multiple, multiple+1} filtered to `1<gn<=max`; for each gn enumerate `(factor, gn/factor)` for all factors dividing gn; pick grid minimizing `|log(w/h) - log(grid_x/grid_y)|`. Returns `(grid_x, grid_y)`.
- `slice_image`: if grid None → `source=resize(_find_best_resize(allow_upscale=True))`, no patches. Else: `source=resize(_find_best_resize(allow_upscale=False))`; `refine_size=_get_refine_size(orig, grid, scale, patch, allow_upscale=True)` (= `_find_best_resize((refine_w/grid_x, refine_h/grid_y))*grid` where `refine_w=_ensure_divide(w, grid_x)` etc.); `refine_image=resize(refine_size)`; `patches=_split_to_patches(refine_image, grid)` (crop into grid_y rows × grid_x cols, each `(w/grid_x, h/grid_y)`, row-major).
- Per image the ordered slice list = `[source] + patches_flattened_row_major`. Each slice → `_preprocess_resized`: normalize `(px/255 - 0.5)/0.5`, CHW, `_reshape_by_patch` → packed `[3, 14, n*14]` (the model later transposes to `[14, n*14, 3]` HWC + expand → `[1,14,n*14,3]`); `tgt_size = [h//14, w//14]` (= SigLIP patch grid (gh,gw)).
- All resizes use **PIL BICUBIC** (P2a already has a bit-exact PIL-BICUBIC port — reuse it).
- **Sliced-prompt id convention** (`_build_placeholder_ids_for_image`): per image, token_count[i] = `prod(tgt_size[i]) // token_divisor` where `token_divisor = 16` (total 16× vision downsample; = (gh/4)*(gw/4) since grids are multiples of 4). Build: `<image>` + image_token×token_count[0] (source) + `</image>`, then for `row in 0..grid_y`: for `col in 0..grid_x`: `<slice>` + image_token×per_slice_tokens (=token_count[1]) + `</slice>`; after each row except the last, a `"\n"`. (No image_id wrapper when use_image_id=False — P2a used use_image_id=False, keep it.) Token ids: `<image>`=248078, `</image>`=248079, `<slice>`/`</slice>` = their tokenizer ids (look up), image_token=`<|image_pad|>`=248056.

## File Structure
- Create `ironmlx/tests/common/minicpmv46_parity.rs` — shared parity-test helpers (Task 1).
- Modify `ironmlx/src/models/minicpmv4_6/image_processor.rs` — add the slicing pipeline (`preprocess_sliced` → `Vec<(Array, i32, i32)>` + the grid-search/slice/refine/split helpers), reusing the P2a PIL-BICUBIC resize + normalize + pack.
- Modify `ironmlx/src/models/minicpmv4_6/mod.rs` — extend `image_placeholder_string` (or add `sliced_image_placeholder_string`) for the source+slice-grid convention.
- Modify `ironmlx/src/cli/generate.rs` + `ironmlx/src/core/server/openai.rs` — emit the multi-slice pixel-value list + grids + sliced prompt; handle multiple images.
- Create `ironmlx/tests/minicpmv46_multislice_parity.rs` + `ironmlx/tests/fixtures/minicpmv46_vl/gen_multislice.py` — multi-slice (+ multi-image) parity vs mlx-vlm.

---

### Task 1: Extract shared parity-test helpers to `tests/common/`
**Why:** 5 parity test files (text/vision/single_image/preprocess/generate_e2e) duplicate ~50 lines of helpers (`load_npy`, `to_f32_vec`, `greedy_argmax`, `top_k`, `checkpoint_dir`, pixel-shape asserts). Both the P2a holistic + T4 code-quality reviews flagged extracting before P3 adds more. Do this FIRST so P3's new parity test (Task 5) uses the shared module.

**Files:** Create `ironmlx/tests/common/minicpmv46_parity.rs`; modify the 5 existing parity test files to `mod common;`/`use` the shared helpers (the integration-test `tests/common/` pattern — `tests/common/mod.rs` already exists for `clean_state`; add a sibling module or extend).

- [ ] Step 1: Read the 5 test files (`minicpmv46_text_logits_match.rs`, `minicpmv46_vision_parity.rs`, `minicpmv46_single_image_parity.rs`, `minicpmv46_preprocess_parity.rs`, `minicpmv46_single_image_generate_e2e.rs`) + the existing `tests/common/mod.rs`. Identify the exact duplicated helpers (their signatures must be unified).
- [ ] Step 2: Create `tests/common/minicpmv46_parity.rs` with the canonical helpers: `pub fn checkpoint_dir() -> PathBuf` (env MINICPMV46_MODEL), `pub fn fixture_dir() -> &str`, `pub fn load_npy(name) -> Array`, `pub fn to_f32_vec(&Array) -> Vec<f32>`, `pub fn greedy_argmax(&Array) -> usize`, `pub fn top_k(&Array, k) -> Vec<usize>`, `pub fn max_abs_diff(&Array,&Array) -> f32`. Register it in `tests/common/mod.rs` (`pub mod minicpmv46_parity;`).
- [ ] Step 3: In each of the 5 test files, replace the local helper definitions with `mod common;` + `use common::minicpmv46_parity::*;` (or the path that works — integration tests each declare their own `mod common;`). Delete the now-duplicate local fns. Keep test bodies identical.
- [ ] Step 4: Run `source ~/.local/mlx/mlx-env.sh && cargo build --release -p ironmlx` (tests compile) + `cargo test --release -p ironmlx --lib` (no lib regression). The #[ignore] tests must still COMPILE (`cargo test --release -p ironmlx --tests --no-run`). Canonical clippy (`--all-targets` will flag pre-existing debt elsewhere — use the standard `--workspace` gate; additionally `cargo +nightly clippy -p ironmlx --tests` should be clean for these files) + fmt clean.
- [ ] Step 5: Commit: `git add ironmlx/tests/common/ ironmlx/tests/minicpmv46_*.rs && git commit -m "test(minicpmv4_6): extract shared parity helpers to tests/common"`

(If sharing across separate `--test` targets via `tests/common/` proves awkward — each integration test is its own crate — the standard Rust idiom is a `tests/common/mod.rs` included via `mod common;` in each test file; confirm the existing `clean_state` usage pattern and follow it. If genuinely blocked, report — but this is a well-trodden pattern.)

---

### Task 2: Multi-slice preprocessing
**Files:** `ironmlx/src/models/minicpmv4_6/image_processor.rs` (extend). Test: parity vs mlx-vlm per-slice pixel + grids.

**Deliverable:** `pub fn preprocess_sliced(img_bytes: &[u8], max_slice_nums: i32) -> Result<Vec<(Array, i32, i32)>>` returning the ordered slice list (source first, then patch slices row-major), each `(pixel_values [1,14,n*14,3], gh, gw)`. Reuse the P2a PIL-BICUBIC resize (`pil_bicubic_resize` or whatever it's named), normalize, and `pack_patches` already in this file. Implement the helpers per the Authoritative algorithm above:
- `fn ensure_divide(len, d) -> i32`, `fn find_best_resize((w,h), scale, patch, allow_upscale) -> (i32,i32)`, `fn get_sliced_grid((w,h), max_slice) -> Option<(i32,i32)>`, `fn get_refine_size(...)`, `fn split_to_patches(image, grid) -> Vec<Vec<SubImage>>` (crop the decoded/resized image into grid cells), `fn slice_image(...) -> (source, patches, Option<grid>)`.
- `preprocess_sliced`: decode → `get_sliced_grid` → if None: single resized image (= existing `preprocess` path, source only) → vec of 1; else: source (find_best_resize allow_upscale=false) + refine→split patches → each normalized+packed → vec.

- [ ] Step 1: Write pure-Rust unit tests for the grid math (no image needed):
```rust
#[test]
fn get_sliced_grid_matches_reference() {
    // coco_sample 640x480, scale_res 448: ratio=640*480/448²≈1.53 → multiple=ceil(1.53)=2.
    // candidates {1(skip),2,3}; grids for 2:{(1,2),(2,1)}, for 3:{(1,3),(3,1)};
    // log(640/480)=log(1.333)=0.2877; pick grid minimizing |0.2877 - log(gx/gy)|.
    // (1,2):log0.5=-0.69 err0.98; (2,1):log2=0.69 err0.40; (1,3):-1.10 err1.38; (3,1):1.10 err0.81.
    // → best (2,1).
    assert_eq!(get_sliced_grid((640, 480), 9), Some((2, 1)));
    // tiny image → None (no slicing)
    assert_eq!(get_sliced_grid((200, 150), 9), None);
}
#[test]
fn ensure_divide_and_find_best_resize() {
    assert_eq!(ensure_divide(100, 56), 112); // round(100/56)=2 → 112
    // find_best_resize for a large image divides by 56
    let (w, h) = find_best_resize((1280, 960), 448, 14, false);
    assert_eq!(w % 56, 0); assert_eq!(h % 56, 0);
}
```
(Compute the EXACT reference values yourself from the algorithm — verify against a quick Python repro of get_sliced_grid if unsure. The (2,1) for 640×480 above is illustrative — RECOMPUTE precisely and assert the true value.)
- [ ] Step 2: Run → FAIL (fns undefined).
- [ ] Step 3: Implement the slicing helpers + `preprocess_sliced` (reuse P2a resize/normalize/pack). `split_to_patches` crops the refine-resized RGB buffer into grid cells (`grid_x = refine_w/grid[0]`, `grid_y_px = refine_h/grid[1]`, row-major crops) then each cell → normalize + pack.
- [ ] Step 4: Run unit tests → PASS. Then the parity test (Step from Task 5's fixture, OR a dedicated one here): see Task 5 — the per-slice pixel+grid parity is verified there against mlx-vlm `preprocess(...)["pixel_values"]`/`["tgt_sizes"]`. For Task 2, a `#[ignore]` parity test `minicpmv46_multislice_preprocess_parity` that loads mlx-vlm's per-slice fixture (Task 5 gen script dumps it) and asserts each slice's pixel max_abs < 0.05 + grid match + slice COUNT match.
- [ ] Step 5: Commit: `git add ironmlx/src/models/minicpmv4_6/image_processor.rs ironmlx/tests/minicpmv46_multislice_preprocess_parity.rs && git commit -m "feat(minicpmv4_6): LLaVA-UHD multi-slice image preprocessing"`

---

### Task 3: Sliced-prompt id construction
**Files:** `ironmlx/src/models/minicpmv4_6/mod.rs` (extend the placeholder helper). Test: assert the produced ids match the convention.

**Deliverable:** a fn building the sliced-image placeholder, e.g. `pub fn sliced_image_placeholder_string(source_grid: (i32,i32), slice_grids: &[(i32,i32)], best_grid: (i32,i32)) -> String` OR work in token-count terms. Per the Authoritative convention: `<image>` + `<|image_pad|>`×N0 (N0 = (source_gh/4)*(source_gw/4)) + `</image>`, then for each of best_grid's (grid_y rows × grid_x cols): `<slice>` + `<|image_pad|>`×Ns (Ns = (slice_gh/4)*(slice_gw/4)) + `</slice>`, with `"\n"` between rows (not after the last). When best_grid is None (no slicing) → just the `<image>`…`</image>` block (= P2a `image_placeholder_string`). Keep `use_image_id=false` (no image-id wrapper).

- [ ] Step 1: Write a unit test asserting the string for a known grid:
```rust
#[test]
fn sliced_placeholder_structure() {
    // best_grid (2,1): 1 source + 2 slices in 1 row (grid_y=1 row, grid_x=2 cols), no inter-row newline.
    // source N0=4, per-slice Ns=2 (illustrative): "<image>"+pad*4+"</image>" + "<slice>"+pad*2+"</slice>"*2
    let s = sliced_image_placeholder_string(4, 2, (2, 1));
    assert_eq!(s.matches("<image>").count(), 1);
    assert_eq!(s.matches("<slice>").count(), 2);          // grid_x*grid_y = 2*1
    assert_eq!(s.matches("<|image_pad|>").count(), 4 + 2*2); // source + 2 slices
    assert!(!s.contains("\n"));                            // single row → no newline
}
```
(Adapt the fn signature to whatever is cleanest — pass source token-count + per-slice token-count + best_grid; OR pass the per-slice grids. Confirm the newline placement: newline AFTER each row except the last, per `_build_placeholder_ids_for_image` lines 1060-1064. For a multi-row grid like (2,2) assert exactly 1 newline.)
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement per the convention. The token counts come from the per-slice grids ((gh/4)*(gw/4)); the source is slice 0, the patches all share the same slice grid (uniform refine).
- [ ] Step 4: Run → PASS. Confirm the no-slice case still equals `image_placeholder_string`.
- [ ] Step 5: Commit: `git add ironmlx/src/models/minicpmv4_6/mod.rs && git commit -m "feat(minicpmv4_6): sliced-image prompt placeholder convention"`

---

### Task 4: CLI + serve multi-slice + multi-image wiring
**Files:** `ironmlx/src/cli/generate.rs` (`prepare_images` minicpmv4_6 branch → use `preprocess_sliced` + the sliced placeholder; loop multiple `--image`), `ironmlx/src/core/server/openai.rs` (the MiniCpmV46 extract_images arm → same). Test: unit-level (token-count/placeholder) + e2e in Task 5.

- [ ] Step 1: Write/extend a unit test on the CLI multi-slice path's grid→placeholder mapping (no real model: feed a synthetic slice list, assert the placeholder + the per-slice grid_thw pushed). 
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement. In the minicpmv4_6 CLI branch (from P2b): for each `--image`, call `preprocess_sliced(bytes, max_slice_nums=9)` → push EACH slice's pixel_values + its (1,gh,gw) grid into `all_pixel_values`/`grids` (in order); build the prompt placeholder via the Task 3 sliced helper (using the per-slice grids). For MULTIPLE images: loop, concatenating per-image placeholder blocks (the existing `for path in &args.images` loop already iterates; extend it to push multiple slices per image + the right placeholder). Mirror in openai.rs's MiniCpmV46 arm. `max_slice_nums`: default 9 (config) — read from config or hardcode 9 with a comment.
   - IMPORTANT: the order of pixel_values in `all_pixel_values` must match the order of image_token runs in the prompt (source then slices, row-major; image 0 before image 1) — `replace_image_tokens` scatters in row-major order of image_token occurrences, and `compute_vision_embeds` concatenates per-image-slice in the same order. Keep them aligned.
- [ ] Step 4: Run `cargo build --release -p ironmlx`; `cargo test --release -p ironmlx --lib -- --nocapture`; clippy/fmt clean. (E2E in Task 5.)
- [ ] Step 5: Commit: `git add ironmlx/src/cli/generate.rs ironmlx/src/core/server/openai.rs && git commit -m "feat(minicpmv4_6): CLI/serve multi-slice + multi-image wiring"`

---

### Task 5: End-to-end multi-slice (+ multi-image) parity vs mlx-vlm
**Files:** Create `ironmlx/tests/fixtures/minicpmv46_vl/gen_multislice.py` + `ironmlx/tests/minicpmv46_multislice_parity.rs` (`#[ignore]`). Uses the `tests/common` helpers (Task 1).

- [ ] Step 1: Fixture generator — run mlx-vlm's processor with `slice_mode=True, max_slice_nums=9` on a fixed HIGH-RES image (use coco_sample or a larger fixture that actually slices — verify it produces multiple → if coco_sample doesn't slice at 9, pick/resize one that does, or set the image so get_sliced_grid returns a real grid). Dump: per-slice `input_pixel_values_sliced.npy` (stacked or per-slice), `input_grids_sliced.npy` (per-slice (gh,gw)), `expected_input_ids_sliced.npy` (the full sliced-prompt ids from mlx-vlm's prompt builder), and `expected_sliced_logits.npy` (full-model last-token logits). ALSO a 2-image fixture for multi-image (dump the same for a 2-image prompt). Assert slice count == 1 + grid_x*grid_y.
- [ ] Step 2: Parity tests (use `common::minicpmv46_parity`):
   - (a) **preprocess parity** (Task 2's gate): ironmlx `preprocess_sliced(image, 9)` per-slice pixel max_abs < 0.05 + grids match + count matches the fixture.
   - (b) **prompt-ids parity** (Task 3's gate): ironmlx sliced placeholder → encode → equals `expected_input_ids_sliced.npy` image-region ids.
   - (c) **e2e logits parity**: feed the fixture ids + all slice pixel_values + grids into `MiniCpmV46Model` (compute_vision_embeds over all slices → forward_vl_chunk with sequential positions) → last-token logits argmax + top-5 + max_abs < 1.0 vs `expected_sliced_logits.npy`.
   - (d) **multi-image**: the 2-image variant of (c).
- [ ] Step 3: Run (regenerate fixtures first). Debug divergences stage-by-stage (preprocess slice mismatch → fix Task 2; prompt ids mismatch → fix Task 3; logits mismatch with correct ids+pixels → scatter/order bug). The per-slice vision is bit-exact (P1) so logits divergence localizes to slice ordering / scatter / prompt-id alignment.
- [ ] Step 4: Commit: `git add ironmlx/tests/fixtures/minicpmv46_vl/gen_multislice.py ironmlx/tests/minicpmv46_multislice_parity.rs && git commit -m "test(minicpmv4_6): P3 multi-slice + multi-image parity vs mlx-vlm"`

---

## Final Gate (P3 done = full VLM done)
`cargo +nightly fmt --all -- --check` ✓; canonical clippy ✓; `cargo build --release` ✓; `cargo test --release -p ironmlx --lib` ✓; all P1/P2a/P2b regressions still ✓; multi-slice preprocess + prompt-ids + e2e logits + multi-image parity ✓. Optionally a manual `ironmlx generate --image <high-res>.jpg` produces coherent text exercising real slicing.

On green: the full MiniCPM-V-4.6 VLM (text + single-image + multi-slice + multi-image) is complete → finishing-a-development-branch (merge/PR decision with Boss).

## Self-Review notes
- Spec coverage: P3 covers spec §7 (full LLaVA-UHD slicing) + §9 P3 (multi-slice + multi-image). The deferred test-helper extraction (P2a/P2b reviews) is Task 1.
- Reuse: P2a PIL-BICUBIC resize + normalize + pack (Task 2 reuses, doesn't reimplement); `image_placeholder_string` (Task 3 extends for the no-slice case); the P2a/P2b model VL forward (per-slice compute_vision_embeds loop + scatter + sequential positions) already supports N image-pad runs — P3 feeds it more slices, no model change expected.
- Dominant risks: (Task 2) the BICUBIC resize of the REFINE image + crop boundaries (per-slice pixel parity, like P2a's 0.0235 floor — the JPEG-decode residual will recur per slice); (Task 3) the exact sliced-prompt convention (source `<image>` + grid `<slice>` rows with newlines — assert ids against the mlx-vlm fixture); (Task 4) pixel_values ordering ↔ image_token-run ordering alignment.
- token_divisor = 16 (= (gh/4)*(gw/4); grids are multiples of 4 via merge_factor=56). Confirm against the `_encode_with_image_placeholders` call site.
- No model-forward change expected — if Task 5 logits diverge with correct ids+pixels, it's a slice-ordering/scatter bug in the wiring, not the model.
