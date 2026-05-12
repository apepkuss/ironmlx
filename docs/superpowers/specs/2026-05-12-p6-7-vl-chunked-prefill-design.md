# P6.7 VL Chunked Prefill — Design

**Status:** Approved (brainstormed 2026-05-12)
**Owner:** ironmlx
**Parent:** P6.6 multi-image close-out (commit `310ae36` on branch `ironmlx-p6-6-multi-image`)
**Branch target:** `ironmlx-p6-7-vl-chunked-prefill` (cut from `ironmlx-p6-6-multi-image` head)

## 1. Motivation

`ironmlx/src/core/generate.rs:363-376` contains an explicit guard that
rejects requests combining `prefill_chunk_size > 0` with `pixel_values`:

```rust
if request.pixel_values.is_some() {
    let effective_chunk = if request.prefill_chunk_size == 0 {
        prompt_len
    } else { request.prefill_chunk_size };
    if prompt_len > effective_chunk {
        return Err(anyhow!(
            "VL prefill currently requires single-chunk: prompt_len={} > chunk_size={}. ...",
            prompt_len, effective_chunk,
        ));
    }
}
```

Real users with long prompts plus an image hit this `Err`. The audit
(P6.4 close-out, item B7) flagged this for follow-up: "VL prefill
single-chunk guard — Medium: needs vision-tower pre-cache so chunk N>0
reads pre-patched embeddings."

The current single-chunk constraint exists because `forward_vl` runs the
vision tower and `cross_modal::replace_image_tokens` in the same call,
and the scatter step requires
`vision_embeds.rows == input_ids.image_pad_count` — which holds only
when the full prompt (containing all image tokens) is the input. Chunked
prefill slices the prompt; each chunk's `input_ids` has its own (often
zero) `image_pad` count.

P6.7 lifts that constraint while preserving numerical equivalence with
the single-chunk path.

## 2. Goals

- Delete the `single-chunk` guard at `generate.rs:363-376`.
- Allow any `prefill_chunk_size` value with `pixel_values` present.
- Numerical equivalence:
  - `max_abs_diff` of last-position logits, chunked vs single-chunk, < 1e-3
  - Greedy first-token bit-identical across all chunk sizes
- No regression on the single-chunk path:
  - P6.3 Task 21 single-image logits-match: `max_diff = 0.3906`, first_token = 760
  - P6.6 N=2 and N=3 all gates green at their P6.6 thresholds
  - 153 lib tests pass, cargo fmt + clippy clean
- Zero new fixtures — reuse P6.6 N=2 and N=3.

## 3. Non-goals

- **Batched serving (B>1)**: separate P-track (B1-p2 / B8). P6.7 stays at B=1.
- **Video** (temporal_patch_size > 2): audit B2 + B3.
- **Anthropic multi-image** (P6.5 B6: currently 400-rejected).
- **Adaptive chunk sizing**: user passes `prefill_chunk_size` explicitly; no
  server-side auto-tuning.
- **Performance optimization**: correctness first. Memory cost of holding
  `vision_embeds_full` for the duration of prefill is acceptable.
- **Streaming vision tower**: vision tower runs once per request, before
  the chunking loop. No incremental vision-tower forward across chunks.

## 4. Architecture

### 4.1 Split `forward_vl` into two callables

`Qwen35Model` gains two new public methods alongside the existing
`forward_vl`:

```rust
/// Run only the vision tower; returns vision_embeds [N_total_patches/4, hidden].
pub fn compute_vision_embeds(
    &self,
    pixel_values: &Array,                  // [N_patches, T, C, H, W]
    grid_thw: &[(i32, i32, i32)],
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;

/// Forward one chunk; expects vision_embeds already computed and sliced
/// to this chunk's k_i image_pad rows.
pub fn forward_vl_chunk(
    &self,
    input_ids: &Array,                     // [1, n_chunk]
    position_ids: &Array,                  // [3, 1, n_chunk] — sliced from full
    cache: Option<&mut [LayerCache]>,
    vision_embeds_slice: Option<&Array>,   // [k_i, hidden]; None if k_i == 0
    image_token_id: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

The existing `forward_vl` is preserved as a thin convenience wrapper:

```rust
pub fn forward_vl(
    &self,
    input_ids: &Array, position_ids: &Array, cache: Option<&mut [LayerCache]>,
    pixel_values: Option<&Array>, grid_thw: Option<&[(i32, i32, i32)]>,
    image_token_id: i32, target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    match (pixel_values, grid_thw) {
        (Some(pv), Some(g)) => {
            let ve = self.compute_vision_embeds(pv, g, target.into())?;
            self.forward_vl_chunk(input_ids, position_ids, cache,
                                  Some(&ve), image_token_id, target.into())
        }
        _ => self.forward_vl_chunk(input_ids, position_ids, cache,
                                   None, image_token_id, target.into()),
    }
}
```

This keeps `p6_6_logits_match` and the existing integration tests untouched.

### 4.2 `GenerationStream` holds VL state across chunks

`ironmlx/src/core/generate.rs` `GenerationStream` adds three fields:

```rust
struct GenerationStream {
    // existing fields …
    cache: Vec<LayerCache>,

    // new for P6.7:
    vision_embeds_full: Option<Array>,   // computed once before chunk loop
    position_ids_full: Option<Array>,    // MRoPE [3, 1, S], computed once
    image_pad_consumed: usize,           // running count of consumed image_pad
}
```

### 4.3 Chunking loop changes

Delete the single-chunk guard at `generate.rs:363-376`.

Before entering the chunking loop:

```text
if request.pixel_values && request.image_grid_thw:
    vision_embeds_full = model.compute_vision_embeds(pv, grids, target)
    position_ids_full  = build_position_ids_vl(full_prompt_ids, grids,
                                                image_token_id, spatial_merge_size)
    image_pad_consumed = 0
```

Inside the chunking loop, for each chunk `[pos .. pos+n]`:

```text
chunk_ids = prompt_ids.slice([pos..pos+n])
if vision_embeds_full.is_some():
    chunk_pos = position_ids_full.slice(axis=2, [pos..pos+n])
    k_i = count(chunk_ids == image_token_id)
    if k_i > 0:
        slice_start = image_pad_consumed
        ve_slice = vision_embeds_full.slice(axis=0, [slice_start..slice_start+k_i])
        image_pad_consumed += k_i
    else:
        ve_slice = None
    model.forward_vl_chunk(chunk_ids, chunk_pos, &mut cache, ve_slice, image_token_id, target)
else:
    // pure-text path (current code path)
    model.text().forward_on(chunk_ids, …)
```

### 4.4 `cross_modal::replace_image_tokens` signature unchanged

The function continues to require
`vision_embeds.rows == input_ids.image_pad_count`. The chunking layer is
responsible for slicing `vision_embeds_full` to the correct k_i rows
before calling. Semantically the function operates on whatever subset of
the prompt is in scope — the contract becomes "scatter as many rows as
there are image tokens." No new API.

### 4.5 Position IDs are sliced, not rebuilt

`build_position_ids_vl` is called once with the full prompt before the
chunking loop. Each chunk reads
`position_ids_full.slice(axis=2, chunk_start..chunk_end)`. This avoids
recomputing MRoPE 3-stream positions per chunk and guarantees that the
position ids seen by the LM are bit-identical to the single-chunk path
(simple slicing, no rounding).

## 5. File changes

| File | Change |
| --- | --- |
| `ironmlx/src/models/qwen3_5/model.rs` | Add `compute_vision_embeds`, `forward_vl_chunk`; refactor `forward_vl` as wrapper |
| `ironmlx/src/core/generate.rs` | Add 3 fields to `GenerationStream`; delete `single-chunk` guard at 363-376; rewrite chunking loop body |
| `ironmlx/tests/p6_7_chunked_prefill.rs` | New integration test driving 6 (N, chunk_size) points |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_7_closeout/report.md` | Acceptance table |

No changes to: `cross_modal.rs`, `vision/`, `image_processor.rs`, fixtures.

## 6. Acceptance

Reuse P6.6 N=2 fixture (`ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/`)
and N=3 stress fixture. Drive 6 test points in a single integration
test (`p6_7_chunked_prefill.rs`):

| (N images, prefill_chunk_size) | Expected |
| --- | --- |
| (2, 0) | baseline = P6.6 N=2 single-chunk logits; max_diff vs mlx-vlm ≤ 0.95, first_token == 760 |
| (2, 256) | max_abs_diff vs (2, 0) < 1e-3; first_token == 760 (bit-identical) |
| (2, 64) | max_abs_diff vs (2, 0) < 1e-3; first_token == 760 |
| (3, 0) | baseline = P6.6 N=3 single-chunk logits; max_diff vs mlx-vlm ≤ 1.20, first_token == 760 |
| (3, 256) | max_abs_diff vs (3, 0) < 1e-3; first_token == 760 |
| (3, 64) | max_abs_diff vs (3, 0) < 1e-3; first_token == 760 |

Internal comparison (chunked vs single-chunk on the **same** ironmlx
build) sidesteps the bf16 cumulative drift studied in P6.6: chunking
should not change *what* is computed, only the order. Any drift larger
than 1e-3 indicates a logic bug.

Regression gates:

- `cargo test -p ironmlx --lib --release -- --test-threads=1`: 153 passed
- `cargo +nightly fmt --all -- --check`: clean
- `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`: clean
- P6.3 Task 21 single-image: max_diff=0.3906, first_token=760
- P6.6 N=2 / N=3 close-out tables unchanged

## 7. Risks + Rollback

**R1 — Chunk boundary cutting through an image-token span.** With
`chunk_size = 64` and the P6.6 fixtures (each image = 1040 image_pad
tokens after merge), the boundary falls in the middle of an image. The
chunked path must consume vision_embeds rows strictly in the global
image_pad encounter order. The N=3 fixture forces this rigorously
because each chunk_size=64 step covers only ~6% of one image's pad span.
**Mitigation**: dedicated test point at chunk_size=64; explicit assert
that `image_pad_consumed == total_image_pad_count` at end of prefill.

**R2 — `position_ids_full` slicing.** MRoPE 3-stream `[3, 1, S]`
slicing must preserve the leading 3 + B axes. **Mitigation**: shape
assertion immediately after the slice; one of the 6 test points
implicitly verifies this through bit-identical first-token.

**R3 — KV cache divergence across chunks.** Chunks N>0 with no
image_pad must produce the same KV-cache writes as if the full prompt
were processed in one chunk. Since `forward_vl_chunk` with
`vision_embeds_slice = None` is exactly the text-only path on
text-only-ids, this falls back to existing chunked-text behavior.
**Mitigation**: max_diff < 1e-3 gate catches any divergence.

**R4 — Memory hold for vision_embeds_full.** A 3-image request holds
~ 2080 × 2560 × 2 bytes ≈ 10.4 MB for the duration of prefill. Acceptable.

**Rollback strategy**: each piece is independent — `compute_vision_embeds`
+ `forward_vl_chunk` is purely additive; `forward_vl` is unchanged in
behavior. If the chunked path proves unreliable, revert only the
chunking-loop changes and re-add the `single-chunk` guard.

## 8. Estimated Effort

| Phase | Work | Estimate |
| --- | --- | --- |
| P6.7a | API split (`compute_vision_embeds` + `forward_vl_chunk` + wrapper); unit tests | 0.5–1d |
| P6.7b | `GenerationStream` fields + chunking-loop rewrite + guard removal | 0.5–1d |
| P6.7c | Position-ids slicing + boundary tests | 0.5d |
| P6.7d | `p6_7_chunked_prefill.rs` 6-point test + diagnostic | 1d |
| P6.7e | Close-out report + regression sweep | 0.5d |
| **Total** | | **~3–4 working days** |

## 9. Out of Scope / Deferred

- Batched serving (B>1) — own P-track (B1-p2)
- Video — own P-track (B2 + B3)
- Anthropic multi-image — currently 400-rejected (P6.5 B6)
- Vision-tower streaming / overlap with text prefill — performance work
- chunk_size autotune — user-controlled for now
