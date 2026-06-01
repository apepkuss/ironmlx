# MiniCPM-V-4.6 VLM — P2b CLI/Serve End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MiniCPM-V-4.6 single-image inference through the user-facing CLI (`ironmlx generate --image`) and HTTP serve paths, end-to-end, matching mlx-vlm.

**Architecture:** Three integration seams: (1) make the shared `GenerationStream` emit flat **sequential** VL positions for MiniCPM-V (it currently hardcodes spatial MRoPE) via a new default-false `Model` method; (2) add a `minicpmv4_6` branch to the CLI `prepare_images`; (3) add a `VisionInputConfig::MiniCpmV46` variant + `openai.rs` preprocessing branches + `serve.rs` wiring. Verified by an end-to-end single-image generate parity vs mlx-vlm.

**Tech Stack:** Rust, cxx-mlx (`mlx`), MLX Metal, axum (serve). Reference (observation): `/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/`.

**Scope:** P2b of: P1 (vision stack ✓) → P2a (single-image model forward ✓, commit c79f08c) → **P2b (CLI/serve e2e)** → P3 (LLaVA-UHD multi-slice + multi-image). Single-image only (no slicing — P3). Boss confirmed: no further sub-splitting.

**Environment (every cargo/test command):** `source ~/.local/mlx/mlx-env.sh` first. Canonical clippy gate: `cargo +nightly clippy --all-features --workspace -- -D warnings` (NOT `--all-targets`). Integration tests `#[ignore]` + env-gated `MINICPMV46_MODEL`.

**Authoritative facts (from P1/P2a):** image_token_id=248056; MiniCPM-V vision total downsample 16× → effective `spatial_merge_size = 4` for image-token-count `(gh/4)*(gw/4)` (already returned by `MiniCpmV46Model::model_meta().spatial_merge_size`); LM uses flat sequential positions even with images (mlx-vlm `_set_position_state` = arange broadcast); `image_processor::preprocess(bytes) -> (Array[1,14,n*14,3], gh, gw)` (P2a); `MiniCpmV46Model::image_token_id()` getter exists.

---

## File Structure

- Modify `ironmlx/src/core/model.rs` — add `fn vl_positions_sequential(&self) -> bool { false }` to the `Model` trait (default = spatial MRoPE; existing models unchanged).
- Modify `ironmlx/src/models/minicpmv4_6/model.rs` — override `vl_positions_sequential() -> true`.
- Modify `ironmlx/src/core/generate.rs` — in `GenerationStream::new`, the VL `pos_full` branch chooses `build_position_ids` (sequential) vs `build_position_ids_vl` (spatial) by `model.vl_positions_sequential()`.
- Modify `ironmlx/src/cli/generate.rs` — `prepare_images` gains a `minicpmv4_6` branch (minicpm preprocess + image_token_id + token count + placeholder).
- Modify `ironmlx/src/core/server/mod.rs` — add `VisionInputConfig::MiniCpmV46 { ... }` variant.
- Modify `ironmlx/src/core/server/openai.rs` — `extract_images` + token-count/placeholder branches for the new variant.
- Modify `ironmlx/src/cli/serve.rs` — set `vision_input = VisionInputConfig::MiniCpmV46` for the minicpmv4_6 architecture.
- Create `ironmlx/tests/minicpmv46_single_image_generate_e2e.rs` (`#[ignore]`) — CLI-path e2e single-image generate parity.

---

### Task 1: `GenerationStream` flat-1D sequential VL positions

**Files:** `ironmlx/src/core/model.rs` (trait method), `ironmlx/src/models/minicpmv4_6/model.rs` (override), `ironmlx/src/core/generate.rs` (the branch). Test: a unit test on the position selection + Qwen/Gemma regression (no behavior change for them).

**Why:** `GenerationStream::new` (generate.rs:1156) builds VL prefill positions via `build_position_ids_vl` (spatial 3-stream MRoPE) whenever `model.requires_position_ids()` is true. MiniCPM-V requires positions BUT must use flat sequential positions (all 3 MRoPE streams = `0..S-1`), per mlx-vlm `_set_position_state`. A spatial-MRoPE position grid would feed wrong rotary phases → garbage.

- [ ] **Step 1: Write the failing test** — In `model.rs` (`#[cfg(test)]`) assert the default + a sequential override compile/return correctly is trivial; the meaningful test is the GenerationStream branch. Add a focused unit test in `generate.rs` `#[cfg(test)]` that does NOT need a real model: verify `build_position_ids(0, s)` produces 3 identical sequential streams (the MiniCPM contract) — `build_position_ids` already exists; assert `[3,1,s]` shape + stream 0 == stream 1 == stream 2 == `0..s`. (This documents the sequential-position contract that the new branch relies on.)
```rust
#[test]
fn build_position_ids_is_flat_sequential_three_streams() {
    let p = build_position_ids(0, 4).unwrap();
    assert_eq!(p.shape().as_slice(), &[3, 1, 4]);
    let v: Vec<i32> = p.to_vec().unwrap(); // [3*1*4]
    // all three streams equal 0,1,2,3
    assert_eq!(&v[0..4], &[0, 1, 2, 3]);
    assert_eq!(&v[4..8], &[0, 1, 2, 3]);
    assert_eq!(&v[8..12], &[0, 1, 2, 3]);
}
```

- [ ] **Step 2: Run** — `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib core::generate -- --nocapture` (or the module path; build_position_ids is in generate.rs). Expected: PASS already if build_position_ids is flat-sequential (this test just locks the contract); if `build_position_ids` is NOT 3 identical streams, STOP and report — the whole MiniCPM VL approach depends on it. (It is — verified: `build_position_ids` broadcasts arange to [3,1,S].)

- [ ] **Step 3: Implement the trait method + override + branch**
  - `ironmlx/src/core/model.rs`, add to the `Model` trait (after `requires_position_ids`):
```rust
    /// When this model takes a VL prefill (images present), should the
    /// position ids be flat **sequential** (`build_position_ids`, all three
    /// MRoPE streams identical) rather than spatial 2-D MRoPE
    /// (`build_position_ids_vl`)? MiniCPM-V-4.6 uses sequential positions even
    /// with images (mlx-vlm `_set_position_state` = arange broadcast). Default
    /// `false` preserves the spatial-MRoPE behavior used by Qwen3.5-VL / Gemma.
    fn vl_positions_sequential(&self) -> bool {
        false
    }
```
  - `ironmlx/src/models/minicpmv4_6/model.rs`, in `impl Model for MiniCpmV46Model`, add:
```rust
    fn vl_positions_sequential(&self) -> bool {
        true
    }
```
  - `ironmlx/src/core/generate.rs`, in `GenerationStream::new`, the `pos_full` construction (currently lines ~1152-1162): replace the `else { build_position_ids_vl(...) }` so the spatial-vs-sequential choice is gated:
```rust
            let pos_full = if dummy_position_ids.is_some() {
                None
            } else if model.vl_positions_sequential() {
                // MiniCPM-V: flat sequential positions over the whole prompt
                // (image tokens included), all three MRoPE streams identical.
                Some(build_position_ids(0, prompt_len_i32)?)
            } else {
                let full_ids_i32: Vec<i32> = request.prompt_ids.iter().map(|&u| u as i32).collect();
                Some(build_position_ids_vl(
                    &full_ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?)
            };
```
  (`prompt_len_i32` is defined a few lines below today — hoist its definition above this block, or use `prompt_len as i32` inline. Confirm `build_position_ids` is in scope in generate.rs — it is, defined in the same file. The per-chunk slicing `slice_pos_ids_axis2(pos_full, pos, pos+n)` and the decode-step `build_position_ids(pos, 1)` already work unchanged for sequential positions.)

- [ ] **Step 4: Run** — `cargo test --release -p ironmlx --lib -- --nocapture` (all pass incl. the new contract test); `cargo build --release -p ironmlx`; canonical clippy + fmt clean. Confirm Qwen/Gemma VL behavior is unchanged (default false → same `build_position_ids_vl` path; no existing test should change).

- [ ] **Step 5: Commit** — `git add ironmlx/src/core/model.rs ironmlx/src/models/minicpmv4_6/model.rs ironmlx/src/core/generate.rs && git commit -m "feat(minicpmv4_6): GenerationStream sequential VL positions for MiniCPM-V"`

---

### Task 2: CLI `prepare_images` minicpmv4_6 branch

**Files:** `ironmlx/src/cli/generate.rs` (`prepare_images` + the model_type branch). Test: a unit test on the minicpm token-count path; e2e covered by Task 4.

**Why:** `prepare_images` (generate.rs:142-219) currently branches `if model_type == "gemma4" { gemma path } else { qwen3_5 path }`. The `else` uses `qwen3_5::image_processor::preprocess` + `<|image_pad|>` + Qwen token-count — WRONG for MiniCPM-V. Add an explicit `minicpmv4_6` branch.

- [ ] **Step 1: Write the failing test** — In `generate.rs` `#[cfg(test)]`, assert the minicpm image-token count uses the 16× downsample:
```rust
#[test]
fn minicpmv46_image_token_count_uses_4x_downsample() {
    // MiniCPM-V grid (28,36) → vision tokens (28/4)*(36/4) = 63.
    assert_eq!(image_token_count_for_grid((1, 28, 36), 4).unwrap(), 63);
}
```
(`image_token_count_for_grid` exists; spatial_merge_size=4 is what `MiniCpmV46Model::model_meta` returns, so the existing helper already yields the right count — this test documents it.)

- [ ] **Step 2: Run** → PASS (documents the contract; `image_token_count_for_grid((1,28,36),4)` = 63 already). If the helper rejects t!=1 or non-divisible, adjust per its existing semantics.

- [ ] **Step 3: Implement the `minicpmv4_6` branch in `prepare_images`** — change the `if model_type == "gemma4" {...} else {...}` into a 3-way match/if on `model_type`. The minicpm branch mirrors the qwen `else` branch structurally but uses the minicpm preprocessor + image_token_id + placeholder. Read the existing qwen `else` branch (generate.rs:191-210) and replicate with these substitutions:
```rust
    // inside prepare_images, replacing the 2-way branch:
    let (spatial_merge_size, image_token_id) = if model_type == "gemma4" {
        // ... existing gemma4 path unchanged ...
    } else if model_type == "minicpmv4_6" {
        let image_token_id = tokenizer
            .token_to_id("<image_pad>") // confirm the MiniCPM-V image-pad token string from tokenizer_config; fallback to 248056
            .map(|id| id as i32)
            .unwrap_or(248056);
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let (pixel_values, gh, gw) = crate::models::minicpmv4_6::image_processor::preprocess(&bytes)
                .with_context(|| format!("preprocessing --image {}", path.display()))?;
            let grid = (1, gh, gw);
            let token_count = image_token_count_for_grid(grid, default_spatial_merge_size)?; // default_spatial_merge_size = model_meta().spatial_merge_size = 4
            all_pixel_values.push(pixel_values);
            grids.push(grid);
            placeholders.push(image_placeholder_string(image_token_id, token_count)); // see note
        }
        (default_spatial_merge_size, image_token_id)
    } else {
        // ... existing qwen3_5 path unchanged ...
    };
```
  NOTES (resolve against the real code):
  - `default_spatial_merge_size` is passed into `prepare_images` (= `model.model_meta().spatial_merge_size` = 4 for MiniCpmV46Model). Use it for the token count.
  - The placeholder string: read how the qwen `else` branch builds `placeholders` (it pushes a string of `image_token_id`-repeated or an `<image>`-marker sequence that `inject_image_placeholders` expands). MATCH mlx-vlm's MiniCPM-V prompt convention: the LM just needs `token_count` occurrences of `image_token_id` at the image position. The simplest correct approach that matches the P2a fixture: the prompt's image marker expands to exactly `token_count` `image_token_id` tokens (plus any `<image>`/`</image>` wrapper tokens mlx-vlm uses — confirm from the P2a `gen_single_image_logits.py` which built the exact ids; reuse that convention so the CLI ids match what the parity test validated).
  - The image-pad token string: P2a used image_token_id=248056 directly; confirm whether the tokenizer maps a string token to it (read tokenizer_config.json / the P2a gen script) — prefer the direct id 248056 (the model config's `image_token_id`) over a string lookup if the string is ambiguous.

- [ ] **Step 4: Run** — `cargo test --release -p ironmlx --lib cli::generate -- --nocapture` (unit pass) + `cargo build --release` + clippy/fmt clean. (Full e2e in Task 4.)

- [ ] **Step 5: Commit** — `git add ironmlx/src/cli/generate.rs && git commit -m "feat(minicpmv4_6): CLI --image preprocessing branch"`

---

### Task 3: Serve `VisionInputConfig::MiniCpmV46` + openai.rs + serve.rs

**Files:** `ironmlx/src/core/server/mod.rs` (variant), `ironmlx/src/core/server/openai.rs` (extract_images + count/placeholder branches), `ironmlx/src/cli/serve.rs` (wiring). Test: unit test on the openai image-token-count branch for the new variant; e2e/HTTP smoke optional (Task 4 covers model-level e2e).

**Why:** The HTTP server's `extract_images` (openai.rs:201-282) branches on `VisionInputConfig {Qwen, Gemma4}` to preprocess. MiniCPM-V needs its own variant + branch so `ironmlx serve` handles MiniCPM-V images.

- [ ] **Step 1: Write the failing test** — In `openai.rs` `#[cfg(test)]`, assert the count/merge for the new variant (mirror the existing Qwen/Gemma count tests at openai.rs:~402):
```rust
#[test]
fn minicpmv46_vision_input_merge_size_is_4() {
    let cfg = VisionInputConfig::MiniCpmV46 { spatial_merge_size: 4 };
    // the merge-size accessor (line ~215) must return 4 for token counting
    assert_eq!(merge_size_for(&cfg), 4); // use the actual accessor name
}
```

- [ ] **Step 2: Run** → FAIL (variant undefined).

- [ ] **Step 3: Implement**
  - `server/mod.rs`: add to `enum VisionInputConfig`:
```rust
    MiniCpmV46 {
        /// Effective downsample for image-token counting = 4 (VitMerger 2×2 × Merger 2×2).
        spatial_merge_size: i32,
    },
```
  - `openai.rs`: add `VisionInputConfig::MiniCpmV46 { spatial_merge_size } => *spatial_merge_size` to the merge-size accessor (line ~215) and the `(spatial_merge_size, ...)` tuple match (line ~402); in `extract_images` (line ~235) add the preprocess arm:
```rust
                        VisionInputConfig::MiniCpmV46 { .. } => {
                            let (pv, gh, gw) = crate::models::minicpmv4_6::image_processor::preprocess(&img_bytes)?;
                            grid_thw.push((1, gh, gw));
                            all_pixel_values.push(pv);
                        }
```
  Match the placeholder/token-count + image_token_id handling to the Qwen arm's structure but with image_token_id=248056 + the (gh/4,gw/4) count (via spatial_merge_size=4). Read the full Qwen arm to replicate the placeholder injection consistently with Task 2's CLI convention (the prompt-side image-token expansion must be identical between CLI and serve so both match mlx-vlm).
  - `serve.rs`: extend the `vision_input` selection (currently only sets Gemma4) so minicpmv4_6 → `Some(VisionInputConfig::MiniCpmV46 { spatial_merge_size: 4 })`:
```rust
    let vision_input = match architecture {
        crate::models::ModelArchitecture::Gemma4 => { /* existing */ }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            Some(server::VisionInputConfig::MiniCpmV46 { spatial_merge_size: 4 })
        }
        _ => None,
    };
```
  (Confirm `server::serve`/`scheduler_actor` thread the request's pixel_values + image_grid_thw + image_token_id through to `forward_vl_chunk` / `batched_prefill_vl` the same way for all variants — they do; the only per-variant logic is preprocessing + token-count, both in openai.rs. The GenerationStream sequential-position fix from Task 1 applies to the server path too since the server drives the same GenerationStream / scheduler VL forward.)

- [ ] **Step 4: Run** — `cargo test --release -p ironmlx --lib server -- --nocapture` (+ openai unit) pass; `cargo build --release`; clippy/fmt clean. Confirm Qwen/Gemma serve VL branches unchanged.

- [ ] **Step 5: Commit** — `git add ironmlx/src/core/server/mod.rs ironmlx/src/core/server/openai.rs ironmlx/src/cli/serve.rs && git commit -m "feat(minicpmv4_6): serve VisionInputConfig + HTTP image preprocessing"`

---

### Task 4: End-to-end single-image generate parity vs mlx-vlm

**Files:** Create `ironmlx/tests/minicpmv46_single_image_generate_e2e.rs` (`#[ignore]`). Reuses the `tests/fixtures/minicpmv46_vl/` `.gitignore`.

**Why:** Verify the FULL CLI path (preprocess → prepare_images → GenerationStream with sequential positions → forward_vl_chunk → sampled tokens) produces the same first token(s) as mlx-vlm for a single image, exercising Task 1+2 together (not just the model-level forward P2a already proved).

- [ ] **Step 1: Fixture generator** — Extend or add to `tests/fixtures/minicpmv46_vl/`: a script `gen_single_image_generate.py` that runs mlx-vlm's GENERATE (greedy, max_tokens=1 or a few) on a fixed prompt + coco_sample image and dumps the generated token id(s) `expected_gen_tokens.npy` (int32) + the rendered prompt string (so the Rust side builds identical input). (If P2a's `expected_single_image_logits.npy` argmax already equals the first greedy token, you may reuse it — but a generate-path fixture is cleaner for the e2e claim.) Run from the mlx-vlm checkout.

- [ ] **Step 2: e2e test** — Build a `GenerateRequest` exactly as the CLI does for a minicpmv4_6 image request: preprocess coco_sample via `minicpmv4_6::image_processor::preprocess`, set pixel_values/image_grid_thw/image_spatial_merge_size=4/image_token_id=248056, build prompt_ids via the same placeholder convention as Task 2 (`token_count = (gh/4)*(gw/4)` image_token_id occurrences at the image position), then drive `GenerationStream::new(&MiniCpmV46Model, &tokenizer, request)` and pull `next_token()` greedily. Assert the first generated token == mlx-vlm's (`expected_gen_tokens.npy[0]`), and ideally the first K tokens match. This exercises the sequential-position GenerationStream path (Task 1) end-to-end.
```rust
#[test]
#[ignore = "requires MINICPMV46_MODEL + fixtures"]
fn minicpmv46_single_image_generate_matches_mlxvlm() {
    // load MiniCpmV46Model (open_multimodal) + tokenizer
    // preprocess coco_sample → (pix, gh, gw); token_count = (gh/4)*(gw/4)
    // build prompt_ids = <prompt prefix> + image_token_id*token_count + <suffix> (match Task 2 / mlx-vlm convention)
    // GenerateRequest { prompt_ids, max_new_tokens: K, sampler: greedy, pixel_values: Some(vec![pix]),
    //   image_grid_thw: Some(vec![(1,gh,gw)]), image_spatial_merge_size: 4, image_token_id: 248056, ... }
    // let mut stream = GenerationStream::new(&model, &tokenizer, request)?;
    // collect first K token ids; assert == expected_gen_tokens.npy
}
```

- [ ] **Step 3: Run** — `source ~/.local/mlx/mlx-env.sh && MINICPMV46_MODEL=<snap> cargo test --release -p ironmlx --test minicpmv46_single_image_generate_e2e -- --ignored --nocapture`. If the first token matches mlx-vlm greedy → PASS (P2a already proved the logits match, so this should follow once positions+preprocess+prompt are wired correctly). If it diverges: the model-level parity (P2a) passed, so the bug is in the e2e wiring — check (a) GenerationStream used sequential positions (Task 1 fired — `vl_positions_sequential()` true), (b) prompt_ids image-token count == vision rows, (c) preprocess output matches the P2a-captured pixel (it does — same preprocessor). Localize + fix.

- [ ] **Step 4: Commit** — `git add ironmlx/tests/minicpmv46_single_image_generate_e2e.rs ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_generate.py && git commit -m "test(minicpmv4_6): P2b end-to-end single-image generate parity vs mlx-vlm"`

---

## Final Gate (P2b done)
`source ~/.local/mlx/mlx-env.sh` then: `cargo +nightly fmt --all -- --check` ✓; `cargo +nightly clippy --all-features --workspace -- -D warnings` ✓; `cargo build --release` ✓; `cargo test --release -p ironmlx --lib` ✓ (no regression — Qwen/Gemma VL unchanged); P2a regressions (vision/text/single-image parity) still ✓; e2e single-image generate parity ✓. Optionally: a manual `ironmlx generate --model <minicpmv46> --image coco_sample.jpg --prompt "Describe this image."` produces coherent text.

On green: P2b lands. Author P3 (LLaVA-UHD adaptive multi-slice + multi-image): the biggest remaining correctness risk (slice grid search + per-slice preprocessing + slice markers + multi-image).

## Self-Review notes
- Spec coverage: P2b covers spec §9 P2's "dispatch/CLI/serve image 路径 → 端到端单图 generate". §8.4 (loader) done P1; §7 preprocess done P2a; §8.1-8.2 model+scatter done P2a.
- Reuse: existing `prepare_images`/`extract_images` Qwen/Gemma branch structure (replicate, don't restructure); `image_token_count_for_grid`; `image_processor::preprocess` (P2a); `GenerationStream` (only the one pos_full branch changes).
- Key risk: Task 1 (shared GenerationStream change) — mitigated by default-false trait method (Qwen/Gemma untouched) + the sequential-position contract test + P2a already proving the model forward is correct under sequential positions.
- Open item flagged for the implementer: the exact image-placeholder/prompt convention (how many wrapper tokens around the `image_token_id*N` run) must MATCH what P2a's `gen_single_image_logits.py` captured + what mlx-vlm renders, so CLI/serve ids == the validated ids. Resolve by reading that gen script + the mlx-vlm prompt builder.
- Decode positions: unchanged (`build_position_ids(pos,1)` per step) — correct for MiniCPM sequential; the Task-1 change only affects VL PREFILL pos_full.
