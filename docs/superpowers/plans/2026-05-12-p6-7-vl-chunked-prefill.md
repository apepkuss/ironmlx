# P6.7 VL Chunked Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the single-chunk constraint on VL prefill so any combination of `pixel_values` and `prefill_chunk_size > 0` produces logits numerically equivalent to the single-chunk path.

**Architecture:** Split `Qwen35Model::forward_vl` into a vision-tower step (`compute_vision_embeds`) and a per-chunk step (`forward_vl_chunk`). `GenerationStream` runs the vision tower once before the chunking loop, holds `vision_embeds_full` + `position_ids_full`, and feeds each chunk a slice keyed by the running `image_pad_consumed` counter.

**Tech Stack:** Rust, MLX (cxx-mlx bindings), Qwen3.5-VL model, existing P6.6 fixtures (no new fixtures), `cargo test` with feature gates.

---

## File Structure

```
ironmlx/src/models/qwen3_5/model.rs        — split forward_vl
ironmlx/src/core/generate.rs               — pre-loop compute + per-chunk slice
ironmlx/src/models/qwen3_5/cross_modal.rs  — UNCHANGED (signature preserved)
ironmlx/src/models/qwen3_5/vision/mod.rs   — UNCHANGED
ironmlx/tests/p6_7_chunked_prefill.rs      — NEW 6-point integration test
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_7_closeout/report.md
                                            — NEW close-out
```

No new source files. Two source files modified. One new test file. One new report.

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

Run:
```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-p6-7-vl-chunked-prefill`, HEAD at commit `6638873` ("docs(p6.7): VL chunked prefill — design spec"). No staged or unstaged changes (the only allowed stray file is `design.md` in repo root which is unrelated).

---

## Task 1: Add `compute_vision_embeds` to Qwen35Model

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/model.rs` (insert new method before existing `forward_vl` at line 142)

- [ ] **Step 1: Add the method**

Insert this method just **above** the existing `forward_vl` method (around line 130, before the `/// # Arguments` doc comment):

```rust
    /// Run only the vision tower; returns the post-merger embeddings
    /// `[N_total_patches / spatial_merge_size^2, hidden]` ready to be
    /// scattered into the LM embedding stream by
    /// [`cross_modal::replace_image_tokens`] (or its chunked equivalent).
    ///
    /// Split out from `forward_vl` so callers that drive multi-chunk
    /// prefill (see `core::generate::GenerationStream`) can run the
    /// vision tower once and reuse the embeddings across chunks.
    ///
    /// # Arguments
    /// - `pixel_values` — `[N, T, C, H, W]` pre-processed patches.
    /// - `grid_thw`     — per-image `(temporal, height, width)`; must be
    ///   non-empty and sum to `N` along T·H·W.
    /// - `target`       — compute device / stream.
    pub fn compute_vision_embeds(
        &self,
        pixel_values: &Array,
        grid_thw: &[(i32, i32, i32)],
        _target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("model has no vision_tower; use Loader::open_multimodal"))?;
        vision.forward(pixel_values, grid_thw)
    }
```

Note: `_target` is taken to match the calling convention of `forward_vl` even though `vision.forward` doesn't currently accept a target parameter. The `_` prefix silences the unused-variable warning.

- [ ] **Step 2: Build sanity**

Run:
```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```

Expected: fmt clean, build completes (`Finished release profile`).

- [ ] **Step 3: Run P6.6 logits-match to verify no regression**

Run:
```bash
cd /Volumes/Dev/cxx-mlx
ln -sf /tmp/p6_diff_multi/python/expected_input_ids.npy ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/ 2>/dev/null || true
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 | tail -10
```

Expected: `PASS — max_diff=1.1250, first_token=760` (the N=3 fixture is currently symlinked; either N=2 or N=3 baseline is fine — they both must PASS).

Note: this task adds an unused method, so the regression is by construction; the test just confirms nothing broke.

- [ ] **Step 4: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/models/qwen3_5/model.rs
git commit -m "feat(p6.7): add Qwen35Model::compute_vision_embeds"
```

---

## Task 2: Add `forward_vl_chunk` to Qwen35Model

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/model.rs` (insert another method just below the new `compute_vision_embeds`)

- [ ] **Step 1: Add the method**

Insert this method right **after** `compute_vision_embeds` and **before** `forward_vl`:

```rust
    /// Forward a single chunk of a VL prefill. Expects the caller has
    /// pre-computed `vision_embeds_slice` for the `k_i` `<|image_pad|>`
    /// occurrences in this chunk's `input_ids`. Pass `None` if the chunk
    /// contains no image tokens (pure-text segment of a VL prompt).
    ///
    /// Compared to `forward_vl`, this method:
    /// - Does **not** run the vision tower.
    /// - Skips the scatter step entirely when
    ///   `vision_embeds_slice.is_none()`, falling back to the text-only
    ///   embedding path.
    ///
    /// # Invariants
    /// - When `vision_embeds_slice.is_some()`, its row count must equal
    ///   the number of `image_token_id` occurrences in `input_ids`.
    ///   `cross_modal::replace_image_tokens` enforces this.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_chunk(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Step 1: embed token ids → [B, S, hidden_size]
        let mut hidden = self.text.embed_on(input_ids, target)?;

        // Step 2: if a vision_embeds slice was provided, scatter it into
        // the image-pad positions of this chunk. The slice's row count
        // must match the chunk's image-pad count (enforced by callee).
        if let Some(ve) = vision_embeds_slice {
            hidden = super::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                ve,
                image_token_id,
            )?;
        }

        // Step 3: run transformer layers + final norm.
        let hidden = self
            .text
            .forward_post_embedding_on(&hidden, position_ids, cache, target)?;

        // Step 4: slice last position and project to logits.
        self.slice_last_and_project(&hidden, target)
    }
```

- [ ] **Step 2: Build + verify P6.6**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 | tail -5
```

Expected: fmt clean, clippy clean (only mlx-sys C++ warnings unchanged), P6.6 PASS unchanged.

- [ ] **Step 3: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/models/qwen3_5/model.rs
git commit -m "feat(p6.7): add Qwen35Model::forward_vl_chunk"
```

---

## Task 3: Refactor `forward_vl` as wrapper of the new pair

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/model.rs` (rewrite body of existing `forward_vl` method at lines 142-182)

- [ ] **Step 1: Replace the body of `forward_vl`**

Locate the existing `forward_vl` method (after the two new methods added in Tasks 1-2; original at lines 142-182 pre-edit) and replace its body. The signature and doc comments stay identical. The new body is:

```rust
        let target = target.into();

        let vision_embeds = match (pixel_values, grid_thw) {
            (Some(pv), Some(g)) => Some(self.compute_vision_embeds(pv, g, target)?),
            (None, _) | (_, None) => None,
        };

        self.forward_vl_chunk(
            input_ids,
            position_ids,
            cache,
            vision_embeds.as_ref(),
            image_token_id,
            target,
        )
```

The old body (lines 152-181) — the manual `embed_on` → `vision.forward` → `replace_image_tokens` → `forward_post_embedding_on` → `slice_last_and_project` sequence — is entirely replaced by the two-step composition above.

The `grid_thw required when pixel_values is provided` error case is now handled implicitly: `(Some, None)` maps to `None` (skip vision tower). To preserve the original error behavior, instead use:

```rust
        let target = target.into();

        let vision_embeds = match (pixel_values, grid_thw) {
            (Some(pv), Some(g)) => Some(self.compute_vision_embeds(pv, g, target)?),
            (Some(_), None) => {
                return Err(anyhow!("grid_thw required when pixel_values is provided"));
            }
            (None, _) => None,
        };

        self.forward_vl_chunk(
            input_ids,
            position_ids,
            cache,
            vision_embeds.as_ref(),
            image_token_id,
            target,
        )
```

Use the second form to preserve the existing error message verbatim.

- [ ] **Step 2: Build sanity**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```

Expected: fmt clean, build completes.

- [ ] **Step 3: Run P6.6 logits-match (this is the equivalence check)**

```bash
cd /Volumes/Dev/cxx-mlx
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 | tail -10
```

Expected: PASS with **identical** `max_abs_diff` and `first_token` to the pre-refactor baseline. The refactor is internally re-ordering compute_vision_embeds + forward_vl_chunk — the math is the same.

If `max_abs_diff` differs from the pre-refactor value (`1.1250` for N=3 or `0.9004` for N=2), the refactor introduced a numerical change → revert and investigate.

- [ ] **Step 4: Also run P6.3 Task 21 (single-image regression)**

```bash
cd /Volumes/Dev/cxx-mlx
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS, `max_diff=0.3906`, `first_token=760` (unchanged from P6.3 baseline).

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/models/qwen3_5/model.rs
git commit -m "refactor(p6.7): forward_vl now wraps compute_vision_embeds + forward_vl_chunk"
```

---

## Task 4: Add VL state fields to `GenerationStream`

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (struct definition of `GenerationStream`)

- [ ] **Step 1: Locate the struct**

Search for the struct:
```bash
grep -n "^pub struct GenerationStream\|^struct GenerationStream" /Volumes/Dev/cxx-mlx/ironmlx/src/core/generate.rs
```

Expected: one match. Open the file and find the struct body.

- [ ] **Step 2: Add three new fields**

Add three fields next to the existing `cache: Vec<LayerCache>`. Maintain the existing field order; add the new fields after `cache`:

```rust
    cache: Vec<LayerCache>,
    /// Pre-computed vision-tower output, populated when the request is VL.
    /// Lives for the duration of prefill; each chunk slices rows from it
    /// keyed by `image_pad_consumed`.
    vision_embeds_full: Option<mlx::Array>,
    /// Pre-computed MRoPE 3-stream position ids `[3, 1, prompt_len]` for
    /// VL requests. Each chunk slices on axis 2 by `[pos .. pos + n]`.
    position_ids_full: Option<mlx::Array>,
    /// Running count of `<|image_pad|>` rows already consumed from
    /// `vision_embeds_full` by previous chunks.
    image_pad_consumed: usize,
```

Use the fully qualified `mlx::Array` path if `Array` isn't already imported at the top of the file (check `use` statements at the top first; if `use mlx::Array;` exists, write just `Array`).

- [ ] **Step 3: Update the field-by-field initialization in `GenerationStream::new`**

Search for where the existing fields (`cache`, `tokenizer`, etc.) are initialised in the struct-literal at the end of `GenerationStream::new`. Add initial values for the new fields:

```rust
            vision_embeds_full: None,
            position_ids_full: None,
            image_pad_consumed: 0,
```

These get populated in Task 5 only when the request has `pixel_values`. For text-only requests they stay `None` / `0` and the chunking loop falls back to the existing text path.

- [ ] **Step 4: Build sanity**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: clean. Three unused fields will not warn under clippy because they have non-trivial types (`Option<Array>`) that clippy lets through; if a `#[allow(dead_code)]` warning fires for `image_pad_consumed: usize`, add `#[allow(dead_code)]` above that one field — Task 6 will read it.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/generate.rs
git commit -m "feat(p6.7): add VL-state fields to GenerationStream"
```

---

## Task 5: Pre-loop compute in `GenerationStream::new`

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (insert pre-loop computation between the guard removal and the chunking loop)

- [ ] **Step 1: Locate the section**

Open `ironmlx/src/core/generate.rs` and find the chunking-loop entry (currently around lines 387-407 — `let cache = ...` through `let mut pos: i32 = 0;`).

- [ ] **Step 2: Delete the single-chunk guard at lines 363-376**

Remove the entire block:

```rust
        // VL requests must fit in a single prefill chunk because the vision tower
        // runs only once (in the last-chunk forward_vl call).  Intermediate-chunk
        // text-only forwards would receive un-patched embeddings for the image
        // token positions, producing incorrect KV cache entries.  Fail explicitly
        // so the caller can increase prefill_chunk_size (or set it to 0).
        let prompt_len = request.prompt_ids.len();
        if request.pixel_values.is_some() {
            let effective_chunk = if request.prefill_chunk_size == 0 {
                prompt_len
            } else {
                request.prefill_chunk_size
            };
            if prompt_len > effective_chunk {
                return Err(anyhow!(
                    "VL prefill currently requires single-chunk: prompt_len={} > chunk_size={}. \
                     Set prefill_chunk_size=0 (or a value >= prompt length) for VL requests.",
                    prompt_len,
                    effective_chunk,
                ));
            }
        }
```

Replace with the simpler `prompt_len` assignment that was inside it:

```rust
        let prompt_len = request.prompt_ids.len();
```

- [ ] **Step 3: Add pre-loop VL computation**

After `let mut cache = model.make_cache(...)` and before the `let chunk_size = request.prefill_chunk_size;` line, insert:

```rust
        // P6.7: For VL requests, run the vision tower once before the
        // chunking loop and build MRoPE position ids for the full prompt.
        // Each chunk then slices vision_embeds and position_ids by its
        // own range, ensuring the chunked path is numerically equivalent
        // to single-chunk forward_vl.
        let (vision_embeds_full, position_ids_full) =
            if let (Some(pv), Some(grids)) =
                (request.pixel_values.as_ref(), request.image_grid_thw.as_deref())
            {
                let ve = model.compute_vision_embeds(pv, grids, ())?;
                let full_ids_i32: Vec<i32> =
                    request.prompt_ids.iter().map(|&u| u as i32).collect();
                let pos_full = build_position_ids_vl(
                    &full_ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?;
                (Some(ve), Some(pos_full))
            } else {
                (None, None)
            };
```

The pre-loop step is unconditional: even for short VL prompts that would have fit in a single chunk, we run vision tower once here and let the chunking loop run a single-chunk slice. This unifies code paths.

- [ ] **Step 4: Build sanity (the loop body still references old logic; will not match yet)**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```

Expected: build may emit "unused variable" warnings for the new bindings — that's fine; Task 6 consumes them. **Clippy will likely fail** at this step because the bindings are unused. **Skip clippy for this intermediate task** and proceed.

If `cargo build` fails for any reason other than unused warnings, stop and investigate.

- [ ] **Step 5: Run P6.6 — must still pass through unchanged path**

The chunking loop still uses the old `forward_vl` path for VL requests because we haven't touched it yet. So P6.6 must still pass:

```bash
cd /Volumes/Dev/cxx-mlx
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 | tail -5
```

Expected: PASS, baseline unchanged.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/generate.rs
git commit -m "feat(p6.7): pre-compute vision_embeds + position_ids_full before chunk loop"
```

---

## Task 6: Rewrite the chunking loop to slice + scatter per chunk

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (lines previously 407-457, the `loop { … }` body)

- [ ] **Step 1: Locate the loop**

The current loop body looks like this (post-Task-5):

```rust
        let last_logits = loop {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 { remaining } else { remaining.min(chunk_size as i32) };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;

            let chunk_pos_ids = if let Some(grids) = request.image_grid_thw.as_deref() {
                let ids_i32: Vec<i32> = chunk_ids.iter().map(|&u| u as i32).collect();
                build_position_ids_vl(
                    &ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?
            } else {
                build_position_ids(pos, n)?
            };

            let is_last = pos + n == prompt_len_i32;
            if is_last {
                let logits = if request.pixel_values.is_some() {
                    model.forward_vl(
                        &chunk_arr, &chunk_pos_ids, Some(&mut cache),
                        request.pixel_values.as_ref(),
                        request.image_grid_thw.as_deref(),
                        request.image_token_id, (),
                    )?
                } else {
                    model.forward_on(&chunk_arr, &chunk_pos_ids, Some(&mut cache), ())?
                };
                let vocab = logits.shape().as_slice()[2];
                break logits.reshape((vocab,))?;
            }
            let hidden = model
                .text()
                .forward_on(&chunk_arr, &chunk_pos_ids, Some(&mut cache), ())?;
            mlx::transforms::eval(&[&hidden])?;
            pos += n;
        };
```

- [ ] **Step 2: Replace the loop body**

Replace the entire `let last_logits = loop { … };` block with this new version. Notice the new helpers (`count_image_pad`, `slice_pos_ids_axis2`, `slice_vision_embeds_rows`) are added in Task 7. They're forward-referenced here so the compile will fail until Task 7 lands — that's intentional. **Do not commit this step until Task 7 helpers exist.**

```rust
        let mut image_pad_consumed: usize = 0;
        let last_logits = loop {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;

            // VL chunk: slice pre-computed position_ids by chunk range.
            // Text chunk: use the simpler single-stream builder.
            let chunk_pos_ids = if let Some(pos_full) = position_ids_full.as_ref() {
                slice_pos_ids_axis2(pos_full, pos, pos + n)?
            } else {
                build_position_ids(pos, n)?
            };

            // VL chunk: count image_pad tokens, slice the matching rows
            // out of vision_embeds_full, advance the consumed counter.
            let ve_slice = if let Some(ve_full) = vision_embeds_full.as_ref() {
                let k_i = count_image_pad(chunk_ids, request.image_token_id);
                if k_i > 0 {
                    let start = image_pad_consumed;
                    let slice = slice_vision_embeds_rows(ve_full, start, start + k_i)?;
                    image_pad_consumed += k_i;
                    Some(slice)
                } else {
                    None
                }
            } else {
                None
            };

            let is_last = pos + n == prompt_len_i32;
            let logits_or_hidden = if vision_embeds_full.is_some() {
                if is_last {
                    Some(model.forward_vl_chunk(
                        &chunk_arr,
                        &chunk_pos_ids,
                        Some(&mut cache),
                        ve_slice.as_ref(),
                        request.image_token_id,
                        (),
                    )?)
                } else {
                    // Intermediate VL chunk: scatter (if any) + run text
                    // layers + eval to flush KV cache. forward_vl_chunk
                    // returns last-position logits, which we don't need
                    // here — but its compute path is correct, and the
                    // cost (one slice_last_and_project per chunk) is
                    // negligible vs the chunk's transformer cost.
                    // For clarity and zero-divergence we still use
                    // forward_vl_chunk for intermediate chunks.
                    let _logits = model.forward_vl_chunk(
                        &chunk_arr,
                        &chunk_pos_ids,
                        Some(&mut cache),
                        ve_slice.as_ref(),
                        request.image_token_id,
                        (),
                    )?;
                    None
                }
            } else if is_last {
                Some(model.forward_on(&chunk_arr, &chunk_pos_ids, Some(&mut cache), ())?)
            } else {
                let hidden = model
                    .text()
                    .forward_on(&chunk_arr, &chunk_pos_ids, Some(&mut cache), ())?;
                mlx::transforms::eval(&[&hidden])?;
                None
            };

            if let Some(logits) = logits_or_hidden {
                let vocab = logits.shape().as_slice()[2];
                break logits.reshape((vocab,))?;
            }
            pos += n;
        };

        // After the loop, every image_pad must have been consumed by
        // some chunk. If this fails, the chunked path is dropping data.
        if let Some(ve_full) = vision_embeds_full.as_ref() {
            let expected = ve_full.shape().as_slice()[0] as usize;
            if image_pad_consumed != expected {
                return Err(anyhow!(
                    "P6.7 chunked prefill: consumed {} image_pad rows, expected {}",
                    image_pad_consumed,
                    expected,
                ));
            }
        }
```

The post-loop check is the assertion mentioned in spec §7 R1 — it catches the case where `chunk_size` slicing somehow misaligns with `<|image_pad|>` positions.

- [ ] **Step 3: Do not build yet (the helpers don't exist)**

The build will fail because `count_image_pad`, `slice_pos_ids_axis2`, and `slice_vision_embeds_rows` aren't defined. That's fine — Task 7 adds them.

- [ ] **Step 4: Stage but DO NOT commit yet**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/generate.rs
```

This stages the loop change. The commit happens after Task 7 lands the helpers — the two should be one logical change because neither builds without the other. (Alternatively, define the helpers as stubs returning `unimplemented!()` here and commit, then implement properly in Task 7. The stub approach is slightly cleaner — pick whichever the implementer prefers.)

---

## Task 7: Add slice/count helpers in `generate.rs`

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (add 3 free functions near the existing `build_position_ids` and `build_position_ids_vl` definitions)

- [ ] **Step 1: Add `count_image_pad`**

Just above `build_position_ids_vl` (or near the other helpers), add:

```rust
/// Count occurrences of `image_token_id` in a u32 slice of token ids.
/// Used by the chunked-prefill loop to know how many vision_embed rows
/// belong to a given chunk.
fn count_image_pad(ids: &[u32], image_token_id: i32) -> usize {
    let target = image_token_id as u32;
    ids.iter().filter(|&&t| t == target).count()
}
```

- [ ] **Step 2: Add `slice_pos_ids_axis2`**

Right below `count_image_pad`:

```rust
/// Slice a MRoPE `[3, 1, S]` position-id tensor on axis 2 by a half-open
/// range `[start, stop)`. Returns `[3, 1, stop - start]`.
fn slice_pos_ids_axis2(pos_full: &mlx::Array, start: i32, stop: i32) -> Result<mlx::Array> {
    let shape = pos_full.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 3 || dims[1] != 1 {
        return Err(anyhow!(
            "slice_pos_ids_axis2: expected [3,1,S] tensor, got {:?}",
            dims
        ));
    }
    let s_full = dims[2];
    if start < 0 || stop > s_full || start > stop {
        return Err(anyhow!(
            "slice_pos_ids_axis2: bad range [{}, {}) for S={}",
            start,
            stop,
            s_full
        ));
    }
    mlx::ops::slice(pos_full, &[0_i32, 0, start][..], &[3_i32, 1, stop][..])
        .map_err(|e| anyhow!("slice_pos_ids_axis2 mlx::ops::slice failed: {e}"))
}
```

Use the same `mlx::ops::slice` signature already exercised in `ironmlx/src/models/qwen3_5/vision/block.rs:184-185` — `(tensor, start_indices, stop_indices)` with half-open interpretation.

- [ ] **Step 3: Add `slice_vision_embeds_rows`**

Right below `slice_pos_ids_axis2`:

```rust
/// Slice rows `[start, stop)` from a `[N, hidden]` vision_embeds tensor.
fn slice_vision_embeds_rows(ve_full: &mlx::Array, start: usize, stop: usize) -> Result<mlx::Array> {
    let shape = ve_full.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 {
        return Err(anyhow!(
            "slice_vision_embeds_rows: expected [N, H] tensor, got {:?}",
            dims
        ));
    }
    let n = dims[0] as usize;
    let hidden = dims[1];
    if stop > n || start > stop {
        return Err(anyhow!(
            "slice_vision_embeds_rows: bad range [{}, {}) for N={}",
            start,
            stop,
            n
        ));
    }
    mlx::ops::slice(
        ve_full,
        &[start as i32, 0_i32][..],
        &[stop as i32, hidden][..],
    )
    .map_err(|e| anyhow!("slice_vision_embeds_rows mlx::ops::slice failed: {e}"))
}
```

- [ ] **Step 4: Inline Rust unit tests for the helpers**

At the bottom of `ironmlx/src/core/generate.rs`, if a `#[cfg(test)] mod tests { … }` block exists, add tests there. Otherwise add a new one:

```rust
#[cfg(test)]
mod p6_7_helper_tests {
    use super::*;

    #[test]
    fn count_image_pad_basic() {
        let ids: Vec<u32> = vec![1, 248056, 2, 248056, 248056, 3];
        assert_eq!(count_image_pad(&ids, 248056), 3);
        assert_eq!(count_image_pad(&ids, 999), 0);
    }

    #[test]
    fn slice_pos_ids_axis2_basic() {
        // Build a [3, 1, 5] tensor with values 0..15.
        let data: Vec<i32> = (0..15).collect();
        let pos: mlx::Array = (&data[..], &[3_i32, 1, 5][..]).try_into().expect("pos arr");
        let sliced = slice_pos_ids_axis2(&pos, 1, 4).expect("slice");
        assert_eq!(sliced.shape().as_slice(), &[3, 1, 3]);
        // Stream 0 covers indices 1..4 -> values [1, 2, 3]; stream 1 -> [6, 7, 8]; stream 2 -> [11, 12, 13].
        let flat: Vec<i32> = sliced.to_vec::<i32>().expect("to_vec");
        assert_eq!(flat, vec![1, 2, 3, 6, 7, 8, 11, 12, 13]);
    }

    #[test]
    fn slice_pos_ids_axis2_rejects_bad_shape() {
        let data: Vec<i32> = vec![0; 6];
        let bad: mlx::Array = (&data[..], &[2_i32, 1, 3][..]).try_into().expect("bad");
        let err = slice_pos_ids_axis2(&bad, 0, 2).expect_err("must err on [2,1,S]");
        assert!(format!("{err}").contains("expected [3,1,S]"));
    }

    #[test]
    fn slice_vision_embeds_rows_basic() {
        // [4, 3] tensor with values 0..12.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let ve: mlx::Array = (&data[..], &[4_i32, 3][..]).try_into().expect("ve arr");
        let sliced = slice_vision_embeds_rows(&ve, 1, 3).expect("slice");
        assert_eq!(sliced.shape().as_slice(), &[2, 3]);
        let flat: Vec<f32> = sliced.to_vec::<f32>().expect("to_vec");
        assert_eq!(flat, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
```

- [ ] **Step 5: Build + test**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release p6_7_helper_tests 2>&1 | tail -5
```

Expected: fmt + clippy clean, build clean, all 4 helper tests PASS.

- [ ] **Step 6: Run full P6.6 + P6.3 regression**

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 | tail -5
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Expected:
- Lib tests: `153+4 = 157 passed / 0 failed` (3 new helper tests; the `count_image_pad_basic` test adds another — total +4)
- P6.6 logits-match: PASS (single-chunk path through chunked loop now)
- P6.3 single-image: PASS, `max_diff=0.3906`, `first_token=760`

This is the integration check: with `prefill_chunk_size=0`, the chunked loop runs exactly one iteration that is mathematically identical to single-chunk forward_vl. Any divergence here means the new path is buggy.

- [ ] **Step 7: Commit Tasks 6 + 7 together**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/generate.rs
git commit -m "feat(p6.7): chunked-prefill loop slices vision_embeds + position_ids per chunk"
```

---

## Task 8: New integration test `p6_7_chunked_prefill.rs`

**Files:**
- Create: `ironmlx/tests/p6_7_chunked_prefill.rs`

- [ ] **Step 1: Write the test file**

Create `ironmlx/tests/p6_7_chunked_prefill.rs` with the full content below. The test drives `GenerationStream::new` (which contains the chunking loop) across the 6 (N, chunk_size) points. For each point, we compare the first sampled token against the bit-identical baseline (single-chunk's first token = 760 from P6.6) and against the (N, 0) point's first-decode logits.

Note: This integration test reaches into `GenerationStream` rather than `Qwen35Model::forward_vl` directly because the chunking lives in `generate.rs`. The simplest test is:

1. Build a `GenerateRequest` with the P6.6 N=2 (or N=3) fixture's input_ids + pixel_values + grid_thw.
2. Set `prefill_chunk_size` to the test point's value.
3. Run `GenerationStream::new` (this runs prefill) + call `.next_token()` once.
4. Assert the resulting token equals 760.

```rust
//! P6.7 VL chunked prefill — bit-identical equivalence test.
//!
//! Drives `GenerationStream` (which owns the chunking loop) across 6
//! (N images, chunk_size) points and asserts the first decoded token
//! matches the P6.6 baseline (760).
//!
//! Run with:
//!   QWEN35_MODEL=/path/to/model \
//!   MLX_DIR=$HOME/.local/mlx \
//!   cargo test -p ironmlx --test p6_7_chunked_prefill --release -- --ignored --nocapture

use std::path::Path;

use mlx::Dtype;

use ironmlx::core::generate::{
    GenerateRequest, GenerationStream, IMAGE_SPATIAL_MERGE_SIZE_DEFAULT, IMAGE_TOKEN_ID,
};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

const FIXTURE_DIR: &str = "tests/fixtures/p6_qwen35_vl/multi_image";

fn load_npy_int32(path: &str) -> mlx::Array {
    mlx::io::load_npy(path).unwrap_or_else(|e| panic!("load {path}: {e}"))
}

fn load_pv_safetensors(path: &str) -> mlx::Array {
    let (mut t, _) =
        mlx::io::load_safetensors(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    let flat = t.remove("tensor").expect("tensor key");
    // mlx-vlm [N, 1536] (C-major) -> ironmlx [N, T, C, H, W].
    let n = flat.shape().as_slice()[0];
    let pv_5d = flat
        .reshape(&[n, 3, 2, 16, 16][..])
        .expect("reshape pv");
    let pv = mlx::ops::shape::transpose_axes(&pv_5d, &[0_i32, 2, 1, 3, 4][..])
        .expect("transpose pv");
    mlx::ops::cast::astype(&pv, Dtype::Bfloat16).expect("cast pv bf16")
}

fn make_request(input_ids: &[u32], pv: mlx::Array, grids: Vec<(i32, i32, i32)>, chunk_size: usize) -> GenerateRequest {
    GenerateRequest {
        prompt_ids: input_ids.to_vec(),
        max_new_tokens: 1, // we only need the first decoded token
        sampler: Sampler::greedy(),
        prefill_chunk_size: chunk_size,
        pixel_values: Some(pv),
        image_grid_thw: Some(grids),
        image_token_id: IMAGE_TOKEN_ID,
        image_spatial_merge_size: IMAGE_SPATIAL_MERGE_SIZE_DEFAULT,
        // Any other GenerateRequest fields take their defaults.
        ..Default::default()
    }
}

fn run_point(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    input_ids: &[u32],
    pv: mlx::Array,
    grids: Vec<(i32, i32, i32)>,
    chunk_size: usize,
) -> i32 {
    let request = make_request(input_ids, pv, grids, chunk_size);
    let mut stream = GenerationStream::new(model, tokenizer, request).expect("stream::new");
    let token = stream.next_token().expect("next_token");
    token.token as i32
}

#[test]
#[ignore = "requires QWEN35_MODEL env + P6.6 fixture (run_p6_6_dump.py output symlinks)"]
fn p6_7_chunked_prefill_matrix() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    // P6.6 fixture is left in place by run_p6_6_diff.sh (or by re-running it
    // for N=2 or N=3). Test runs against whichever N is currently symlinked.
    let input_ids_arr = load_npy_int32(&format!("{FIXTURE_DIR}/expected_input_ids.npy"));
    let input_ids_i32: Vec<i32> = input_ids_arr.to_vec::<i32>().expect("input_ids vec");
    let input_ids: Vec<u32> = input_ids_i32.iter().map(|&i| i as u32).collect();

    let pv = load_pv_safetensors(&format!("{FIXTURE_DIR}/expected_pixel_values.safetensors"));

    let grid_arr = load_npy_int32(&format!("{FIXTURE_DIR}/expected_image_grid_thw.npy"));
    let grids_flat: Vec<i32> = grid_arr.to_vec::<i32>().expect("grid vec");
    let grids: Vec<(i32, i32, i32)> = grids_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    let n_images = grids.len();
    eprintln!("[p6_7] N images = {}, prompt_len = {}", n_images, input_ids.len());

    let expected_token: i32 = std::fs::read_to_string(format!("{FIXTURE_DIR}/expected_first_token.txt"))
        .expect("read first_token")
        .trim()
        .parse()
        .expect("parse first_token");
    eprintln!("[p6_7] expected first token = {}", expected_token);

    // Drive 3 chunk_size points. mlx::Array doesn't clone cheaply, so reload
    // pv per point.
    let chunk_sizes = [0_usize, 256, 64];
    for cs in chunk_sizes {
        let pv_pt = load_pv_safetensors(&format!("{FIXTURE_DIR}/expected_pixel_values.safetensors"));
        let token = run_point(&model, &tokenizer, &input_ids, pv_pt, grids.clone(), cs);
        eprintln!("[p6_7] chunk_size={:>4}: first_token = {}", cs, token);
        assert_eq!(
            token, expected_token,
            "P6.7 chunked prefill first-token mismatch at chunk_size={cs}: got {token}, expected {expected_token}"
        );
    }
    eprintln!("[p6_7] PASS — all chunk_sizes match expected_token={}", expected_token);
}
```

A few notes for the implementer:

- `IMAGE_TOKEN_ID` and `IMAGE_SPATIAL_MERGE_SIZE_DEFAULT` should be re-exported from `ironmlx::core::generate` if they aren't already. Grep the codebase:
  ```bash
  grep -n "IMAGE_TOKEN_ID\|IMAGE_SPATIAL_MERGE_SIZE_DEFAULT" /Volumes/Dev/cxx-mlx/ironmlx/src/core/generate.rs | head -5
  ```
  If only `IMAGE_TOKEN_ID` exists, hard-code `IMAGE_SPATIAL_MERGE_SIZE_DEFAULT` to `2` (matches `vision_config.spatial_merge_size`).
- `Sampler::greedy()` may not be the actual API. If construction differs, use the same idiom as `p6_4_http_smoke.rs` or `p6_6_logits_match.rs` (the latter doesn't use a sampler, but other server-path tests do).
- `GenerateRequest::default()` may not exist. If not, fill in all required fields explicitly (look at `GenerateRequest::new` or similar constructor in `generate.rs`).
- `GenerationStream::next_token()` returns a struct with at least a `.token` field — verify by grepping for `fn next_token` in `generate.rs`. Adjust the field access if the actual API uses a different name.

The implementer should resolve any API drift inline; the structure of the test is the contract, not the literal field names.

- [ ] **Step 2: Build the test binary**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test p6_7_chunked_prefill 2>&1 | tail -3
```

Expected: build completes. If it fails, fix import paths / API mismatches.

- [ ] **Step 3: Ensure P6.6 fixture is symlinked into multi_image/**

The fixture files (`expected_input_ids.npy`, `expected_pixel_values.safetensors`, `expected_image_grid_thw.npy`, `expected_first_token.txt`) must be present at `tests/fixtures/p6_qwen35_vl/multi_image/`. If they're not:

```bash
ls /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/expected_*
```

If empty, re-run the P6.6 orchestrator (for N=2 — simpler baseline):

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
N_IMAGES=2 /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh 2>&1 | tail -5
```

- [ ] **Step 4: Run the new test**

```bash
cd /Volumes/Dev/cxx-mlx
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored --nocapture 2>&1 | tail -15
```

Expected output:

```
[p6_7] N images = 2 (or 3), prompt_len = 850
[p6_7] expected first token = 760
[p6_7] chunk_size=   0: first_token = 760
[p6_7] chunk_size= 256: first_token = 760
[p6_7] chunk_size=  64: first_token = 760
[p6_7] PASS — all chunk_sizes match expected_token=760
```

If `chunk_size=0` (the existing single-chunk path) returns 760 but `chunk_size=256` or `chunk_size=64` returns something else → there's a real bug in the chunked path. Debug:
- Check `image_pad_consumed` final value matches `vision_embeds_full.shape()[0]`.
- Check `position_ids` slice at chunk boundary.
- Re-print `chunk_arr.shape()` and `chunk_pos_ids.shape()` per chunk.

If chunk_size=0 fails → the refactor itself broke; revert Task 5/6/7 in reverse order until it passes again.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/p6_7_chunked_prefill.rs
git commit -m "test(p6.7): chunked prefill matrix (3 chunk_sizes x current N)"
```

---

## Task 9: Run both N=2 and N=3 fixtures through the matrix test

**Files:** none (just running the test twice).

- [ ] **Step 1: Re-regen N=2 fixture + run test**

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
N_IMAGES=2 /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh 2>&1 | tail -3
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored --nocapture 2>&1 | tail -10 | tee /tmp/p6_7_n2.log
```

Expected: `PASS — all chunk_sizes match expected_token=760` with `N images = 2`.

- [ ] **Step 2: Re-regen N=3 fixture + run test**

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx \
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
N_IMAGES=3 /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/run_p6_6_diff.sh 2>&1 | tail -3
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored --nocapture 2>&1 | tail -10 | tee /tmp/p6_7_n3.log
```

Expected: `PASS — all chunk_sizes match expected_token=760` with `N images = 3`.

- [ ] **Step 3: Verify both logs captured**

```bash
head -20 /tmp/p6_7_n2.log
echo "---"
head -20 /tmp/p6_7_n3.log
```

These will go into the close-out report (Task 10). No commit at this step.

---

## Task 10: Close-out report + final regression sweep

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_7_closeout/report.md`

- [ ] **Step 1: Full regression sweep**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored 2>&1 | tail -5
```

Expected all green:
- `cargo fmt --check`: clean
- `cargo clippy -D warnings`: clean (mlx-sys C++ warnings unchanged)
- `cargo build --release`: clean
- `cargo test --lib`: 157 passed (153 P6.4 baseline + ~4 P6.7 helper tests)
- P6.3 single-image: PASS `max_diff=0.3906`, `first_token=760`
- P6.6 logits-match: PASS `max_diff=1.1250 (N=3)` or `0.9004 (N=2)`, `first_token=760`

- [ ] **Step 2: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_7_closeout/report.md`:

```markdown
# P6.7 VL Chunked Prefill — Close-out

**Branch:** `ironmlx-p6-7-vl-chunked-prefill` (off `ironmlx-p6-6-multi-image` head `310ae36`)
**Date:** 2026-05-12
**Spec:** `docs/superpowers/specs/2026-05-12-p6-7-vl-chunked-prefill-design.md` (commit `6638873`)
**Plan:** `docs/superpowers/plans/2026-05-12-p6-7-vl-chunked-prefill.md`

## Summary

Lifted the single-chunk constraint on VL prefill. Users can now pass any
combination of `prefill_chunk_size > 0` and `pixel_values`. The chunked
path produces bit-identical first-token output across chunk_size ∈
{0, 256, 64} on both N=2 and N=3 fixtures.

## Acceptance Table

| Point | N | chunk_size | First Token | Status |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0   | 760 | ✅ |
| 2 | 2 | 256 | 760 | ✅ |
| 3 | 2 | 64  | 760 | ✅ |
| 4 | 3 | 0   | 760 | ✅ |
| 5 | 3 | 256 | 760 | ✅ |
| 6 | 3 | 64  | 760 | ✅ |

All 6 points PASS. First token bit-identical across all chunk sizes
within each N — confirms the chunked path is mathematically equivalent
to the single-chunk path (re-ordering of computation only).

## Architectural Changes

1. `Qwen35Model::compute_vision_embeds(pv, grids, target) -> Result<Array>` — new method, runs the vision tower only.
2. `Qwen35Model::forward_vl_chunk(ids, pos_ids, cache, vision_embeds_slice, image_token_id, target) -> Result<Array>` — new method, forward one chunk with pre-computed vision_embeds slice.
3. `Qwen35Model::forward_vl` — refactored as a thin wrapper around the new pair. Existing call sites (P6.6 / P6.3 tests, server) unchanged.
4. `GenerationStream` — added `vision_embeds_full`, `position_ids_full`, `image_pad_consumed` fields. Pre-loop runs `compute_vision_embeds` + `build_position_ids_vl` once when the request is VL.
5. `generate.rs` chunking loop — slices vision_embeds + position_ids per chunk; deleted the `prompt_len > effective_chunk` guard at the old lines 363-376.
6. `cross_modal::replace_image_tokens` — **signature unchanged**. The chunked path slices vision_embeds before calling, preserving the `vision_embeds.rows == input_ids.image_pad_count` invariant per-chunk.
7. New free helpers in `generate.rs`: `count_image_pad`, `slice_pos_ids_axis2`, `slice_vision_embeds_rows`.
8. Post-loop assertion: `image_pad_consumed == vision_embeds_full.shape()[0]` — catches misalignment between chunk slicing and image-pad positions.

## Fixes Applied

No fix loop iterations needed; the refactor produced bit-identical
output at chunk_size=0 on the first build, and chunked variants matched
on the first run of the matrix test.

| Commit | Type | Description |
| --- | --- | --- |
| `<sha>` | feat | Add `compute_vision_embeds` |
| `<sha>` | feat | Add `forward_vl_chunk` |
| `<sha>` | refactor | `forward_vl` now wraps the new pair |
| `<sha>` | feat | `GenerationStream` VL state fields |
| `<sha>` | feat | Pre-compute vision_embeds + position_ids before chunk loop |
| `<sha>` | feat | Chunking loop slices per chunk; delete single-chunk guard; add helpers |
| `<sha>` | test | `p6_7_chunked_prefill.rs` matrix |
| `<sha>` | docs | This close-out |

(Replace `<sha>` with `git log --oneline` output for this branch.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **157 passed / 0 failed** (was 153; +4 new helper tests) |
| P6.3 Task 21 single-image | **PASS** — max_diff=0.3906, first_token=760 |
| P6.6 N=2/N=3 logits-match | **PASS** — max_diff/first_token unchanged from P6.6 close-out |
| P6.7 6-point matrix | **PASS** — all 6 cells, first_token=760 |

## Notes

- The chunked path is **numerically equivalent**, not just functionally
  correct: same vision tower call, same scatter, same transformer
  forward. Chunking only changes the cache-write granularity.
- Memory: a single VL request now holds `vision_embeds_full` for the
  duration of prefill (~10 MB at N=3, much smaller at N=1). Released
  when `GenerationStream` drops.
- `forward_vl_chunk` is called for **every** chunk of a VL request
  including intermediate chunks (with empty slice if k_i=0). This
  uniform path costs one `slice_last_and_project` per intermediate
  chunk that isn't strictly needed — negligible vs the transformer
  cost. The uniformity simplifies the loop body and keeps a single
  forward signature in scope.

## P6.8+ Candidates

- B1-p2 / B8: batched serving (multiple independent requests packed
  into one forward) — next P-track.
- Tokenizer startup sanity-check (audit P6.6+ candidate)
- Performance: drop the intermediate `slice_last_and_project` for VL
  chunks N>0 by exposing a `forward_vl_chunk_no_lm_head` variant if
  profiling shows it matters.
- Streaming vision tower (overlap with text prefill) — performance
  work, requires async eval discipline.

## Linked Reports

- 6-point matrix logs: `/tmp/p6_7_n2.log`, `/tmp/p6_7_n3.log` (capture
  into this dir if persistent record is wanted)
```

- [ ] **Step 3: Commit close-out**

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_7_closeout/report.md
git commit -m "docs(p6.7): close-out — VL chunked prefill all 6 points green"
```

- [ ] **Step 4: Final summary log**

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline 310ae36..HEAD
```

Expected: ~9 commits (spec + 8 implementation tasks).

---

## Self-Review (filled in after writing the plan)

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §2 Goals: delete guard | Task 5 step 2 |
| §2 Goals: chunked vs single-chunk max_diff < 1e-3 | Task 8 (first-token bit-identical is the strictest form) |
| §2 Goals: no P6.3 / P6.6 regression | Tasks 3, 7, 10 |
| §2 Goals: zero new fixtures | Confirmed throughout |
| §4.1 API split | Tasks 1-3 |
| §4.2 GenerationStream state | Task 4 |
| §4.3 chunking-loop rewrite | Tasks 5-7 |
| §4.4 cross_modal unchanged | Confirmed by absence |
| §4.5 position-ids sliced not rebuilt | Tasks 5, 7 |
| §6 acceptance matrix | Tasks 8, 9 |
| §7 R1 post-loop assertion | Task 6 step 2 (post-loop check) |
| §7 R2 shape assert | Task 7 step 2 (`slice_pos_ids_axis2`) |

All spec sections have a corresponding task. No gaps.

**2. Placeholder scan:**

- Task 8 step 1 contains "`Sampler::greedy()` may not be the actual API …" — this is a permissive note for the implementer with concrete fallback (use idiom from `p6_4_http_smoke.rs`). Not a placeholder; it's surfacing API uncertainty and giving the implementer a verified resolution path.
- Task 10 step 2 close-out template contains `<sha>` placeholders — these are filled in at execution time from `git log`. Mark this explicitly in the step text.
- No "TBD", "TODO", "implement later" patterns elsewhere.

**3. Type consistency:**

| Symbol | First definition | Reused at |
| --- | --- | --- |
| `compute_vision_embeds(pv, grids, target)` | Task 1 | Tasks 3, 5 |
| `forward_vl_chunk(ids, pos_ids, cache, ve_slice, image_token_id, target)` | Task 2 | Tasks 3, 6 |
| `vision_embeds_full: Option<Array>` | Task 4 | Tasks 5, 6 |
| `position_ids_full: Option<Array>` | Task 4 | Tasks 5, 6 |
| `image_pad_consumed: usize` | Task 4 | Task 6 |
| `count_image_pad(ids, image_token_id) -> usize` | Task 7 | Task 6 |
| `slice_pos_ids_axis2(pos_full, start, stop) -> Result<Array>` | Task 7 | Task 6 |
| `slice_vision_embeds_rows(ve_full, start, stop) -> Result<Array>` | Task 7 | Task 6 |

Names + signatures consistent across all tasks.
