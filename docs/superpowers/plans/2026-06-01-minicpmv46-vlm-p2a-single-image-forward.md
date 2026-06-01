# MiniCPM-V-4.6 VLM — P2a Single-Image Model Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MiniCpmV46Model` run a correct single-image VL forward — SigLIP vision embeds scattered into the Qwen3.5-text backbone — verified by last-token logits parity vs mlx-vlm.

**Architecture:** New `MiniCpmV46Model { text: Qwen35TextModel, lm_head: Option<Linear>, vision: Option<MiniCpmV46Vision>, image_token_id }` (spec §8.1; Boss-confirmed over the wrap-`Qwen35Model` alternative to avoid inheriting Qwen3-VL's NaViT vision methods, which are wrong for MiniCPM-V). It reuses `Qwen35TextModel` (embed / transformer / final norm) + `cross_modal::replace_image_tokens` (scatter) + `MiniCpmV46Vision` (P1 SigLIP), and reimplements the thin, config-driven orchestration (`make_cache` hybrid cache partition, last-token slice+project, `forward_vl_chunk` embed→scatter→text). Plus a new single-image (no-slice) image preprocessor. MiniCPM-V uses flat sequential LM positions (`build_position_ids`), so this plan verifies at the model level with sequential positions directly.

**Tech Stack:** Rust, cxx-mlx (`mlx`), MLX Metal. Reference (observation only): `/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/{minicpmv4_6,processing_minicpmv4_6}.py`.

**Scope:** This is **P2a of**: P1 (vision stack ✓, commit bf1c537) → **P2a (single-image model forward + parity)** → P2b (GenerationStream flat-1D VL positions + CLI `--image`/serve + e2e generate) → P3 (LLaVA-UHD multi-slice + multi-image). P2 was split because the CLI/serve e2e path requires modifying the shared `GenerationStream`, which currently hardcodes `build_position_ids_vl` (spatial MRoPE) at `core/generate.rs:1156` — MiniCPM-V needs sequential positions, a shared-core change that warrants its own plan after the model forward is proven. **P2a does NOT touch `GenerationStream` or the CLI `--image` flow.**

**Environment (every cargo/test command):** `source ~/.local/mlx/mlx-env.sh` first. Canonical clippy gate: `cargo +nightly clippy --all-features --workspace -- -D warnings` (NOT `--all-targets`). Integration tests are `#[ignore]` + env-gated (`MINICPMV46_MODEL`).

**Authoritative facts** (spec §1): vision bf16; image_token_id=248056; total vision downsample 16× = VitMerger(2×2)×Merger(2×2) → image-token count for a SigLIP grid `(gh,gw)` is `(gh/4)*(gw/4)`. LM hidden 1024. MiniCPM-V LM positions are flat sequential (mlx-vlm `_set_position_state` = `arange` broadcast to 3 streams).

---

## File Structure

- Create `ironmlx/src/models/minicpmv4_6/image_processor.rs` — single-image (no-slice) preprocess → `(pixel_values [1,14,n*14,3], grid_h, grid_w)`.
- Create `ironmlx/src/models/minicpmv4_6/model.rs` — `MiniCpmV46Model` (Model + DenseVlMethods).
- Modify `ironmlx/src/models/minicpmv4_6/mod.rs` — `pub mod image_processor; pub mod model; pub use model::MiniCpmV46Model;`; change `model_from_loader` to return `MiniCpmV46Model` (subsume facade) — see Task 2.
- Modify dispatch sites `ironmlx/src/cli/{generate,serve}.rs` + `ironmlx/src/bin/ironmlx-core-bench.rs` — construct `MiniCpmV46Model` (text-only path unchanged; no `--image` wiring yet — that's P2b).
- Modify `ironmlx/tests/minicpmv46_text_logits_match.rs` — point at `MiniCpmV46Model`'s text path (keep the regression green).
- Create `ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_logits.py` + `ironmlx/tests/minicpmv46_single_image_parity.rs` (`#[ignore]`).

---

### Task 1: Single-image (no-slice) image preprocessor

**Files:** Create `ironmlx/src/models/minicpmv4_6/image_processor.rs`; add `pub mod image_processor;` to `minicpmv4_6/mod.rs`. Test: a `#[ignore]` parity test reusing P1's mlx-vlm `input_pixel_values.npy`/`input_grid.npy` fixture (already produced by `tests/fixtures/minicpmv46_vl/gen_vision_embeds.py` with slice_mode off).

**Goal:** `pub fn preprocess(img_bytes: &[u8]) -> Result<(Array, i32, i32)>` returning `(pixel_values, grid_h, grid_w)` where `pixel_values` is patch-packed `[1, 14, n*14, 3]` (n = grid_h*grid_w) matching what mlx-vlm's processor produces for a SINGLE slice (slice_mode off). The Rust side must match mlx-vlm's `MiniCPMVImageProcessor` single-slice path.

**Reference (observation):** Read `/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/processing_minicpmv4_6.py` — for slice_mode off / single source image: `_find_best_resize(original_size, scale_resolution, patch_size)` + `_ensure_divide`, then resize (the interpolation filter — confirm: PIL BICUBIC or similar from the processor), normalize with `self.mean`/`self.std` (read their values from the processor / `preprocessor_config.json` in the model dir), then `reshape_by_patch`/patch-packing into `[3, 14, n*14]` (CHW) → the model transposes to `[14, n*14, 3]` (HWC) + expand_dims. The Rust `preprocess` should produce the HWC `[1,14,n*14,3]` directly (the layout `SiglipEmbeddings::forward_on` consumes). Pattern to follow for structure (decode/normalize/patchify + return signature): `ironmlx/src/models/qwen3_5/image_processor.rs` (`smart_resize`/`normalize_pixel`/`patchify`/`preprocess`), but the resize math + normalization constants + pack layout are MiniCPM-V's, NOT Qwen's.

- [ ] **Step 1: Write the failing parity test**

`ironmlx/tests/minicpmv46_preprocess_parity.rs`:
```rust
//! P2a: MiniCPM-V-4.6 single-image (no-slice) preprocess parity vs mlx-vlm.
use mlx::{Array, Dtype};
const DIR: &str = "tests/fixtures/minicpmv46_vl";
fn npy(n: &str) -> Array { mlx::io::load_npy(&format!("{DIR}/{n}")).expect(n) }

#[test]
#[ignore = "requires the mlx-vlm fixture (gen_vision_embeds.py) + the coco_sample image"]
fn minicpmv46_preprocess_matches_mlxvlm() {
    // Reference pixel_values + grid captured from mlx-vlm's processor (slice_mode off).
    let exp_pix = npy("input_pixel_values.npy"); // [1,14,n*14,3] f32
    let grid: Vec<i32> = npy("input_grid.npy").to_vec().expect("grid");
    let (gh_ref, gw_ref) = (grid[0], grid[1]);

    let bytes = std::fs::read("tests/fixtures/p6_qwen35_vl/coco_sample.jpg").expect("image");
    let (pix, gh, gw) = ironmlx::models::minicpmv4_6::image_processor::preprocess(&bytes).expect("preprocess");
    assert_eq!((gh, gw), (gh_ref, gw_ref), "grid mismatch");

    let a: Vec<f32> = mlx::ops::cast::astype(&pix, Dtype::Float32).unwrap().to_vec().unwrap();
    let b: Vec<f32> = exp_pix.to_vec().unwrap();
    assert_eq!(a.len(), b.len(), "pixel count mismatch");
    let max_abs = a.iter().zip(&b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
    println!("preprocess max_abs={max_abs:.5}");
    // Resize/resample introduces small interpolation differences vs PIL; set the
    // bound from the observed value (document it). Pixel values are normalized
    // (~[-2,2]); a faithful resize should give max_abs well under 0.05.
    assert!(max_abs < 0.05, "preprocess max_abs {max_abs} too high — resize/normalize mismatch");
}
```

- [ ] **Step 2: Run to verify it fails** — `source ~/.local/mlx/mlx-env.sh && MINICPMV46_MODEL=/Users/xin/.ironmlx/models/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19 cargo test --release -p ironmlx --test minicpmv46_preprocess_parity -- --ignored --nocapture` → FAIL (preprocess not defined). Ensure the P1 fixture exists first (regenerate via `gen_vision_embeds.py` if needed).

- [ ] **Step 3: Implement `preprocess`** — Translate mlx-vlm's single-slice path. Structure:
  1. Decode JPEG/PNG (use the same image-decode crate `qwen3_5/image_processor.rs` uses — read it; likely `image` crate).
  2. `(rh, rw) = find_best_resize(orig_h, orig_w, scale_resolution, patch=14)` per mlx-vlm `_find_best_resize` + `_ensure_divide(len,14)=max(round(len/14)*14, 14)`. Read the EXACT scale_resolution (likely 448 from config / `image_size`) from `preprocessor_config.json`.
  3. Resize with the SAME resampling filter mlx-vlm uses (confirm BICUBIC; the `image` crate's `imageops::resize` with `FilterType::CatmullRom`≈bicubic — match as closely as possible; resize is the dominant parity risk).
  4. Normalize: `(pixel/255 - mean)/std` with mean/std from the processor (read `preprocessor_config.json`).
  5. Patch-pack into `[1,14,n*14,3]` HWC where n=(rh/14)*(rw/14)=gh*gw, grid_h=rh/14, grid_w=rw/14. Match mlx-vlm's `reshape_by_patch` ordering exactly (row-major over the (gh,gw) patch grid).
  6. Return `(pixel_values: Array bf16-or-f32, grid_h, grid_w)`.

  (Full code authored against the compiler + the reference; the fixture parity test in Step 4 is the acceptance gate. Use `qwen3_5/image_processor.rs` for the decode/normalize/Array-construction idioms.)

- [ ] **Step 4: Run parity** — same command as Step 2 → PASS (`max_abs < 0.05`; lock the bound to the observed value with a comment, like P1). If grid mismatches, fix the resize math; if pixel max_abs is high, fix the resample filter / normalization / pack ordering (compare against a per-step dump from mlx-vlm if needed).

- [ ] **Step 5: Commit** — `git add ironmlx/src/models/minicpmv4_6/image_processor.rs ironmlx/src/models/minicpmv4_6/mod.rs ironmlx/tests/minicpmv46_preprocess_parity.rs && git commit -m "feat(minicpmv4_6): single-image no-slice image preprocessor"`

---

### Task 2: `MiniCpmV46Model` struct + `Model` trait + subsume facade

**Files:** Create `ironmlx/src/models/minicpmv4_6/model.rs`; modify `minicpmv4_6/mod.rs` (`pub mod model; pub use model::MiniCpmV46Model;`, change `model_from_loader`); modify dispatch `cli/generate.rs`, `cli/serve.rs`, `bin/ironmlx-core-bench.rs`; modify `tests/minicpmv46_text_logits_match.rs`.

**Reference:** `ironmlx/src/models/qwen3_5/model.rs` (`Qwen35Model` — READ its `make_cache` hybrid Full/Linear partition, its `slice_last_and_project` + `per_row_slice_last` last-token projection, its `forward_on`/`batched_prefill`/`forward_vl_chunk`/`forward_vl_hidden`/`batched_prefill_vl` bodies, and its `Model` + `DenseVlMethods` impls — these are the bodies you reimplement on `MiniCpmV46Model` over a `Qwen35TextModel` instead of over a full `Qwen35Model`). `ironmlx/src/models/qwen3_5/text_model.rs` (`Qwen35TextModel::{from_loader, embed_on, forward_on, forward_post_embedding_on, as_output_on, config}`). `ironmlx/src/models/qwen3_5/cross_modal.rs::replace_image_tokens`.

**Struct (spec §8.1, Boss-confirmed — own the text core + lm_head directly; do NOT wrap `Qwen35Model`, to avoid inheriting its NaViT `compute_vision_embeds`/`batched_prefill_vl` which are wrong for MiniCPM-V):**
```rust
pub struct MiniCpmV46Model {
    /// Qwen3.5-text backbone (embed + hybrid decoder layers + final norm).
    text: crate::models::Qwen35TextModel,
    /// `Some` when `!tie_word_embeddings`; `None` reuses `text` embedding matrix
    /// for the output projection (MiniCPM-V-4.6 ties, so this is `None`).
    lm_head: Option<crate::nn::Linear>,
    /// SigLIP vision stack (P1). `Some` only when loaded via open_multimodal with
    /// vision_tower weights present.
    vision: Option<crate::models::minicpmv4_6::vision::MiniCpmV46Vision>,
    image_token_id: i32,
}
```

**Shared helper to reimplement (mirror `Qwen35Model`):** a private `slice_last_and_project(&self, hidden, last_positions: Option<&[i32]>, target) -> Result<Array>` that slices the last (or per-row last) position from `[B,S,H]` and projects via `self.lm_head` (if `Some`) or `self.text.as_output_on` (tied) → `[B,1,vocab]`. Copy the body + the `per_row_slice_last` free fn from `qwen3_5/model.rs` (they are self-contained). This is the ~40-line reused-but-reimplemented piece; it is config/shape-driven and stable.

- [ ] **Step 1: Write the failing test** (text-only construction + forward, in model.rs `#[cfg(test)]` or as a lib test):
```rust
#[test]
fn minicpmv46_model_text_only_constructs_and_forwards() {
    // Uses from_cfg_for_test-style stub if available, else gated on real model.
    // Minimal: assert MiniCpmV46Model implements Model + DenseVlMethods via a
    // bound-check fn, and num_hidden_layers delegates to core.
    fn assert_surface<M: crate::core::Model + crate::core::scheduler::DenseVlMethods>(_: &M) {}
    let _ = assert_surface::<MiniCpmV46Model>; // compile-time trait check
}
```
(If a no-weights stub is impractical, make this a compile-time trait-bound assertion + defer behavioral checks to Task 3/4. Do NOT fabricate weights.)

- [ ] **Step 2: Run to verify it fails** — `cargo test --release -p ironmlx --lib minicpmv4_6::model -- --nocapture` → FAIL (type undefined).

- [ ] **Step 3: Implement `MiniCpmV46Model` + `Model` impl + from_loader**
```rust
impl MiniCpmV46Model {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = crate::models::minicpmv4_6::config::text_config_from_loader(loader)?; // Qwen35Config, vision_config=None
        let tie = cfg.tie_word_embeddings;
        let text = crate::models::Qwen35TextModel::from_loader(loader, cfg)?;
        let lm_head = if tie { None } else { Some(crate::nn::Linear::from_loader(loader, "lm_head")?) };
        let vcfg = crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig::from_loader(loader).ok();
        let vision = match (&vcfg, loader.contains("vision_tower.embeddings.patch_embedding.weight")) {
            (Some(vc), true) => Some(crate::models::minicpmv4_6::vision::MiniCpmV46Vision::from_loader(loader, vc)?),
            _ => None,
        };
        let image_token_id = vcfg.map(|v| v.image_token_id).unwrap_or(crate::core::generate::IMAGE_TOKEN_ID);
        Ok(Self { text, lm_head, vision, image_token_id })
    }
}
```
(`cfg` is consumed by `Qwen35TextModel::from_loader`; read `tie_word_embeddings` before moving it. Confirm `loader.contains(...)` exists — it is used by `Qwen35Model::from_loader_with_config`; if the name differs, use the real one.)

Implement `impl Model for MiniCpmV46Model` by REIMPLEMENTING (mirroring `Qwen35Model`'s bodies, but over `self.text` + the local `slice_last_and_project`):
- `make_cache(batch, cap, dtype)` — copy `Qwen35Model::make_cache`'s hybrid partition verbatim (it reads `self.text.config()`: for each layer, `Full` → `KVCache::new(...).with_step(cap)`, `Linear` → `GatedDeltaCache::new_with_cap(...)` by `cfg.layer_kind(i)`). ~40 lines, config-driven.
- `forward_on(input_ids, position_ids, per_row_lens, decode_mask, cache, target)` → `self.text.forward_on(...)` (returns hidden `[B,S,H]`) then `self.slice_last_and_project(&hidden, None, target)`.
- `batched_prefill(...)` → `self.text.embed_on` + `self.text.forward_post_embedding_on(..., attention_mask, linear_attention_mask, per_row_lens, ...)` then `slice_last_and_project(&hidden, Some(&per_row_lens-1), target)` (mirror `Qwen35Model::batched_prefill`).
- `forward_text_hidden(...)` → `self.text.forward_on(...)` (hidden, no project).
- `model_meta()` → build from `self.text.config()` (mirror `Qwen35Model::model_meta`; `spatial_merge_size` for MiniCPM = `merge_group` product = 4, or reuse the config's value — for P2a model_meta's spatial_merge_size is only consumed by the CLI image-token-count which is P2b, so a correct constant of 4 is fine; document it).
- `num_hidden_layers()` → `self.text.config().num_hidden_layers as usize`.

Confirm the exact `Model` trait method set from `crate::core::model::Model`. The `slice_last_and_project` + `per_row_slice_last` helpers are copied from `qwen3_5/model.rs` (self-contained).

- [ ] **Step 4: Subsume facade + wire dispatch**
  - In `minicpmv4_6/mod.rs`: change `pub fn model_from_loader(loader) -> Result<Qwen35Model>` to `-> Result<MiniCpmV46Model>` returning `MiniCpmV46Model::from_loader(loader)`. (Keep the function name so dispatch call sites need minimal change; OR rename and update sites — choose the smaller diff.) Update the doc comment.
  - `cli/generate.rs`, `cli/serve.rs`, `bin/ironmlx-core-bench.rs`: the `MiniCpmV46 =>` arms now bind a `MiniCpmV46Model`. For generate/bench (which require `M: Model + DenseVlMethods`) and serve (`serve_with_model`), `MiniCpmV46Model` must satisfy those bounds — it will after Task 3 adds `DenseVlMethods`. To keep the build green BETWEEN tasks, implement a minimal `DenseVlMethods` now (Task 3 fills the real bodies) OR do Task 2+3 as one commit. RECOMMENDED: defer the dispatch-site edit to the end of Task 3 (when DenseVlMethods is complete) so the build is never broken. For Task 2, just add the struct + Model impl + a `#![allow(dead_code)]`-free path by NOT yet wiring dispatch. (If `model_from_loader` return type changes, the dispatch arms break compile — so either keep `model_from_loader -> Qwen35Model` until Task 3, or change return type + add DenseVlMethods in the same task. Pick the approach that keeps `cargo build` green at each commit; document which.)
  - Update `tests/minicpmv46_text_logits_match.rs`: replace `minicpmv4_6::model_from_loader` (→Qwen35Model) usage with `MiniCpmV46Model::from_loader` + its `forward_on` (text path). The 4-prompt logits regression must still PASS (text path numerics unchanged — MiniCpmV46Model's core IS the same Qwen35Model).

- [ ] **Step 5: Run + commit** — `cargo build --release -p ironmlx` + `cargo test --release -p ironmlx --lib minicpmv4_6 -- --nocapture` + the canonical clippy/fmt. Re-run the text logits regression (`MINICPMV46_MODEL=... cargo test --release -p ironmlx --test minicpmv46_text_logits_match -- --ignored --nocapture`) → still PASS. Commit: `git commit -m "feat(minicpmv4_6): MiniCpmV46Model wrapping Qwen35 core; subsume text-only facade"`

---

### Task 3: VL forward — `DenseVlMethods` (SigLIP vision + reused scatter)

**Files:** `ironmlx/src/models/minicpmv4_6/model.rs` (extend); finalize dispatch-site wiring deferred from Task 2. Test: text-only-equivalence unit test (mirror `qwen3_6_moe/model.rs`'s `text_only_vl_chunk_delegates_to_core_forward`).

**Implement `impl DenseVlMethods for MiniCpmV46Model`** (trait at `crate::core::scheduler::DenseVlMethods`). All vision via `self.vision` (SigLIP); all scatter via `cross_modal::replace_image_tokens`; all projection via the local `slice_last_and_project` (Task 2). NO delegation to a Qwen35Model:
```rust
fn compute_vision_embeds(&self, pixel_values: &[Array], grid_thw: &[(i32,i32,i32)], target: StreamOrDevice) -> Result<Array> {
    let vision = self.vision.as_ref().ok_or_else(|| anyhow!("MiniCpmV46Model has no vision tower; use Loader::open_multimodal"))?;
    if pixel_values.len() != grid_thw.len() { return Err(anyhow!("compute_vision_embeds: len mismatch")); }
    let mut embeds = Vec::with_capacity(pixel_values.len());
    for (pix, &(_t, h, w)) in pixel_values.iter().zip(grid_thw.iter()) {
        embeds.push(vision.compute_vision_embeds(pix, h, w, target)?); // SigLIP → [N_i, 1024]
    }
    if embeds.len() == 1 { Ok(embeds.pop().unwrap()) }
    else { let refs: Vec<&Array> = embeds.iter().collect(); Ok(mlx::ops::concatenate(&refs, 0)?) }
}
fn forward_vl_chunk(&self, input_ids, position_ids, per_row_lens, decode_mask, cache, vision_embeds_slice, image_token_id, target) -> Result<Array> {
    // embed → (optional) scatter pre-computed vision embeds → text transformer → last-token project.
    // Mirror Qwen35Model::forward_vl_chunk, but over self.text + local slice_last_and_project.
    let mut hidden = self.text.embed_on(input_ids, target)?;
    if let Some(ve) = vision_embeds_slice {
        hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(&hidden, input_ids, ve, image_token_id)?;
    }
    let hidden = self.text.forward_post_embedding_on(&hidden, position_ids, cache, decode_mask, None, per_row_lens, target)?;
    self.slice_last_and_project(&hidden, None, target)
}
fn forward_vl_hidden(&self, input_ids, position_ids, per_row_lens, decode_mask, cache, vision_embeds_slice, image_token_id, target) -> Result<Array> {
    // Same as forward_vl_chunk but return hidden (no projection) — for prefix-prefill chunks.
    let mut hidden = self.text.embed_on(input_ids, target)?;
    if let Some(ve) = vision_embeds_slice {
        hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(&hidden, input_ids, ve, image_token_id)?;
    }
    self.text.forward_post_embedding_on(&hidden, position_ids, cache, decode_mask, None, per_row_lens, target)
}
fn batched_prefill_vl(&self, input_ids, position_ids, attention_mask, linear_attention_mask, per_row_lens, per_row_pixel_values, per_row_grid_thw, image_token_id, cache, target) -> Result<Array> {
    // Mirror Qwen35Model::batched_prefill_vl: per-row SigLIP vision via self.compute_vision_embeds,
    // concat in row order, scatter into the batched embeds, text batched forward, per-row last project.
    // (P2a's single-image parity uses forward_vl_chunk at B=1; batched_prefill_vl must compile + be
    //  correct for B≥1 — reimplement faithfully from Qwen35Model::batched_prefill_vl, swapping its
    //  per-row compute_vision_embeds(NaViT) for self.vision SigLIP via self.compute_vision_embeds.)
    // ... full body mirrored from qwen3_5/model.rs::batched_prefill_vl ...
}
```
`cross_modal::replace_image_tokens` is `pub` (confirm; if it's `pub(crate)` or module-private, either it's reachable from this crate path or add a re-export). Then complete the dispatch-site wiring (generate/serve/bench arms construct `MiniCpmV46Model`; `model_from_loader -> MiniCpmV46Model`) now that `DenseVlMethods` is satisfied, and confirm `cargo build` green.

- [ ] **Step 1: Write the failing test** (text-only forward_vl_chunk == forward_on, real-model `#[ignore]` OR a stubbed core if feasible). Mirror `qwen3_6_moe/model.rs::text_only_vl_chunk_delegates_to_core_forward`:
```rust
#[test]
#[ignore = "requires MINICPMV46_MODEL"]
fn minicpmv46_text_only_vl_chunk_matches_forward_on() {
    // load MiniCpmV46Model; forward_on(ids) vs forward_vl_chunk(ids, vision_embeds_slice=None) → byte-equal.
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo build` fails until DenseVlMethods is implemented (the dispatch arms / test reference it).

- [ ] **Step 3: Implement** the 4 DenseVlMethods + finalize dispatch wiring per above.

- [ ] **Step 4: Run** — `cargo build --release`, `cargo test --release -p ironmlx --lib minicpmv4_6 -- --nocapture`, canonical clippy/fmt clean. The text-only-equivalence ignored test (with MINICPMV46_MODEL) → PASS. Text logits regression still PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(minicpmv4_6): DenseVlMethods (SigLIP vision + reused cross-modal scatter)"`

---

### Task 4: Single-image VL logits parity vs mlx-vlm (model level)

**Files:** Create `ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_logits.py` + `ironmlx/tests/minicpmv46_single_image_parity.rs` (`#[ignore]`). Reuse the `.gitignore` (`expected_*.npy`/`input_*.npy`) already in `tests/fixtures/minicpmv46_vl/`.

**Goal:** With a fixed prompt + the coco_sample image, verify ironmlx's `MiniCpmV46Model` single-image forward (vision → scatter → text → last-token logits, with SEQUENTIAL `build_position_ids`) matches mlx-vlm's full-model single-image last-token logits (argmax + top-5 set + structural max_abs), at B=1, no GenerationStream.

- [ ] **Step 1: Fixture generator** `gen_single_image_logits.py` (mlx-vlm): load model+processor; build a fixed prompt with the image placeholder (use the processor's `image_placeholder`/`<image>` convention for a single slice, slice_mode off); run the FULL model forward on `(input_ids, single image)` to get last-token logits `[vocab]`; dump `expected_input_ids_img.npy` (the exact token ids incl. the N=image_token placeholders), `input_pixel_values.npy` (reuse/confirm the same single-slice pixel tensor `[1,14,n*14,3]`), `input_grid.npy`, and `expected_single_image_logits.npy` (f32 `[vocab]`). Print the image-token count N and assert it equals `(gh/4)*(gw/4)`. Run from the mlx-vlm checkout; iterate until clean.

- [ ] **Step 2: Parity test** `minicpmv46_single_image_parity.rs`:
```rust
#[test]
#[ignore = "requires MINICPMV46_MODEL + fixtures"]
fn minicpmv46_single_image_logits_match_mlxvlm() {
    let dir = std::env::var("MINICPMV46_MODEL").expect("MINICPMV46_MODEL");
    let loader = Loader::open_multimodal(Path::new(&dir)).expect("open_multimodal");
    let model = MiniCpmV46Model::from_loader(&loader).expect("model");
    let ids: Vec<i32> = npy("expected_input_ids_img.npy").to_vec().unwrap();
    let pix = astype(npy("input_pixel_values.npy"), Bfloat16); // [1,14,n*14,3]
    let grid: Vec<i32> = npy("input_grid.npy").to_vec().unwrap(); // [gh,gw]
    let s = ids.len() as i32;
    let input_ids = Array::from((ids, [1, s]));
    let position_ids = build_position_ids(0, s)?; // SEQUENTIAL (flat 1D) — MiniCPM-V contract
    let mut cache = model.make_cache(1, s+1, Bfloat16)?;
    // forward_vl: compute vision embeds + scatter + text + last logits.
    let logits = model.forward_vl(&input_ids, &position_ids, Some(&[s]), None, Some(&mut cache),
        Some(&[pix]), Some(&[(1, grid[0], grid[1])]), model_image_token_id, ())?; // see note
    // compare last-position logits argmax + top-5 set + max_abs to expected.
}
```
NOTE: `MiniCpmV46Model` may not have a single-shot `forward_vl` (that computes vision + scatter + project). If not, the test does it in two steps: `let ve = model.compute_vision_embeds(&[pix], &[(1,gh,gw)], ())?;` then `model.forward_vl_chunk(&input_ids, &position_ids, Some(&[s]), None, Some(&mut cache), Some(&ve), image_token_id, ())?` → `[1,1,vocab]`. Use whichever the surface provides (forward_vl_chunk + compute_vision_embeds definitely exist after Task 3). Assert argmax matches + top-5 set equal + `max_abs < <bound>` (set from observed; vision is bf16 dense → expect small, but the LM is 4-bit so expect ~0.1-0.5 like the text-only test; document the bound and reasoning, don't loosen post-hoc).

- [ ] **Step 3: Run** — `source ~/.local/mlx/mlx-env.sh && MINICPMV46_MODEL=<snap> cargo test --release -p ironmlx --test minicpmv46_single_image_parity -- --ignored --nocapture`. If argmax/top-5 match → DONE; lock max_abs bound. If diverges → debug: the vision-embeds are already bit-exact (P1), so a divergence is in scatter (image_token placement / count), positions (must be sequential), or the LM path. Localize (compare the scattered inputs_embeds, check N image tokens == vision rows). Fix root cause.

- [ ] **Step 4: Commit** — `git add ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_logits.py ironmlx/tests/minicpmv46_single_image_parity.rs && git commit -m "test(minicpmv4_6): P2a single-image VL logits parity vs mlx-vlm"`

---

## Final Gate (P2a done)
`source ~/.local/mlx/mlx-env.sh` then: `cargo +nightly fmt --all -- --check` ✓; `cargo +nightly clippy --all-features --workspace -- -D warnings` ✓; `cargo build --release` ✓; `cargo test --release -p ironmlx --lib` ✓ (no regression); preprocess parity ✓; text-logits regression ✓; single-image logits parity (argmax+top-5) ✓.

On green: P2a lands. Author P2b (GenerationStream flat-1D VL positions + CLI `--image` + serve VisionInputConfig + e2e generate parity), then P3 (LLaVA-UHD multi-slice + multi-image).

## Self-Review notes
- Spec coverage: P2a covers spec §7 (single-image no-slice preprocess), §8.1-8.2 (MiniCpmV46Model + cross-modal via reused scatter + sequential positions), §8.3 dispatch (construct the model; `--image` CLI flow + serve VisionInputConfig DEFERRED to P2b), §8 facade subsume. §8.4 (loader resampler retention) done in P1. §9 P2's "generate parity" → P2b.
- Reuse-not-replicate: `Qwen35TextModel` (embed/transformer/norm), `cross_modal::replace_image_tokens` (scatter), `MiniCpmV46Vision` (P1), `qwen3_5/image_processor.rs` idioms. Reimplemented-but-stable (copied from `Qwen35Model`): `make_cache` hybrid partition + `slice_last_and_project`/`per_row_slice_last` + the forward_vl orchestration (~60 lines, config/shape-driven).
- Open risks flagged: preprocess resize-filter parity (T1, the dominant risk — bicubic match); `batched_prefill_vl` faithful mirror (T3, minor for single-image scope); the sequential-vs-spatial position contract (verified: MiniCPM uses flat `build_position_ids`).
- Struct = spec §8.1 (own `Qwen35TextModel`+`lm_head`+`Option<MiniCpmV46Vision>`), Boss-confirmed over wrapping `Qwen35Model` — avoids inheriting NaViT vision methods (wrong for MiniCPM-V), explicit override boundary, decoupled from Qwen35Model's VL API; small reimplementation cost accepted.
