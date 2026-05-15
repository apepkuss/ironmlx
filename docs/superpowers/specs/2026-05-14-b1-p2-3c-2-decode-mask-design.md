# B1-p2.3c-2 — Per-row decode mask activation in Scheduler::step

**Date:** 2026-05-14
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3c-1 head `902dffe`)
**Predecessor sub-phase:** B1-p2.3c-1 — Per-row KV cache offset (cache + model API merged). 3c-1 shipped `KVCache.offsets: Vec<i32>`, `GatedDeltaCache.offsets: Vec<i32>`, `per_row_lens` threading through the model API, the right-padding migration, and the `build_per_row_decode_mask` helper. The helper was shipped without a production caller; 3c-2 wires it in.
**Sibling sub-phases (rest of 3c series, separate specs):**
- **3c-3** — `SchedulerActor::driver_loop` mid-batch admit/evict + admission window during active Decoding
- **3c+** — chunked batched prefill (removes long-prompt GS fallback)
**Successor sub-phases:** 3d (admission queue + preemption + `ADMISSION_DEADLINE` config exposure), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Fix a numerical-correctness gap in `Scheduler::step` when cache offsets diverge across rows. Currently `model.forward_on` is called without an attention mask; finished rows have frozen `cache.offsets()` while active rows continue advancing, so the SDPA `[B, n_kv_heads, max_off, head_dim]` returned K/V slice contains stale buffer-init zero K/V at positions `[offsets[i]..max_off]` for any row `i` whose offset trails `max_off`. Active rows' Q at decode-time attends to those stale positions, contaminating the softmax denominator (~25–30% weight reallocation in observed cases; output direction preserved because stale V is zero, but magnitude is attenuated).

2. Wire `build_per_row_decode_mask` (shipped in 3c-1) as the first production caller. `Scheduler::step` constructs `per_row_real_lens[i] = cache.offsets()[i] + per_row_lens[i]` (post-write offsets), builds an additive bf16 `[B, 1, 1, max_real_len]` mask, and passes it to `Qwen35Model::forward_on` via a new `decode_mask: Option<&Array>` parameter.

3. Extend `Qwen35Model::forward_on` and `Qwen35TextModel::forward_on` with `decode_mask: Option<&Array>`. The mask is forwarded into the existing `attention_mask` parameter of `forward_post_embedding_on`. **No changes to `attention.rs` / `gated_attention.rs` / `decoder_layer.rs` / `gated_delta_net.rs` signatures** — mlx fast SDPA already consumes additive masks in this shape via `mask_mode = ""` + `mask_arr`; `[B, 1, 1, K]` is the decode-time degenerate of the prefill `[B, 1, T, T]` mask layout.

4. Fold in 3c-1's three carry-over minor items (final reviewer recommendation):
   - Add 2 new `KVCache` lib unit tests (`multi_step_accumulation` + `per_row_data_isolation`) — closes the plan I-2/I-3 gap that was preempted by the right-padding scope expansion in 3c-1 Task 4.
   - Update `build_per_row_decode_mask` doc-comment to name its first production caller (`Scheduler::step`).
   - Remove a stale `// TEMP(b1-p2.3c-1 Task 1)` comment in `mtp_cache.rs` test code.

## §2 Non-goals

- **Mid-batch evict.** Finished rows still occupy their slot. `slots[i] = Some(state) where state.finished = true` remains the representation. 3c-3 lifts this.
- **Mid-batch admit during active Decoding.** Same — 3c-3.
- **`Phase` enum changes.** State machine is unchanged: `Idle / Admitting / Decoding / Finished`. The "all rows finish together" constraint becomes inconsequential at the numerics layer (mask corrects attention) but the phase transition is still "active_count == 0 ⇒ Finished".
- **Server / HTTP layer changes.** `core/server/scheduler_actor.rs`, `openai.rs`, `anthropic.rs` are not touched.
- **VL or Linear-attention path changes.** Decoder layer routes `full_attn_mask` to `Attention` / `GatedAttention`; `GatedDeltaNet` continues to receive only `linear_attn_mask`. Decode-mask leakage into the linear path is structurally prevented by the existing dispatch (verified in §4).
- **Chunked prefill.** Long-prompt GS fallback remains in place; 3c+ handles it.
- **Lockstep-cost optimization.** Finished rows still pad the forward at full B. 3c-3 evict reduces this.

## §3 Background

### 3.1 The numerical gap (concrete trace)

`Scheduler::step` ([`scheduler.rs:512-665`](../../ironmlx/src/core/scheduler.rs#L512)) builds `per_row_lens = [1 if active else 0; B]` and calls `model.forward_on(input_ids, position_ids, Some(&per_row_lens), Some(cache_ref), ())` — no mask.

Inside `Qwen35Model::forward_on` → `text_model::forward_on` → `forward_post_embedding_on(.., attention_mask=None, ..)` → each `DecoderLayer::forward_on(.., full_attn_mask=None, ..)` → `Attention::forward_on(.., mask=None, ..)`. The cache layer writes K/V at per-row offsets, then returns slices truncated to `max(post_write_offsets)`. SDPA runs in `mask_mode = "causal"` over those slices.

Concrete scenario at b_max=2, prompt_lens=[4, 4], max_new_tokens=[3, 8]:

```
step 3: row 0 emits its 3rd token and transitions to finished='length'.
        cache.offsets() ends at [7, 7].

step 4: per_row_lens = [0, 1]
        pre-write offsets = [7, 7]
        post-write offsets = [7, 8]
        max_off = 8
        Returned K/V slice shape [2, n_kv_heads, 8, head_dim]
        row 0's positions [7..8] are stale (zero K/V from buffer init)
        row 1's positions [0..8] are all real
        SDPA causal mode: row 1 attends to its [0..8] — correct
                          row 0 attends to its [0..8], including stale [7]
                          row 0's logit is discarded by Scheduler — but
                          inside SDPA, softmax(row 0 scores over [0..8])
                          has 1 stale-zero score in the denominator, so
                          real-position weights are deflated by exp(0) /
                          sum ≈ 1/8 ≈ 12.5%.
        row 0's V at stale position is zero → output unchanged direction,
        attenuated magnitude. Argmax likely preserved at this size.

step 8: per_row_lens = [0, 1]
        pre-write offsets = [7, 11]
        post-write offsets = [7, 12]
        max_off = 12
        row 0's positions [7..12] are stale (5 stale zeros)
        row 0 attends 1+5 = 6 valid + 5 stale positions; softmax dilution
        is now 5/12 ≈ 42% on the denominator (relative deflation)
        row 0 output is still discarded.
```

The bug is silent in the current `b1_p2_3b_1` suite because:
- `b2_happy` / `b4_happy` use equal-length prompts → all rows advance lockstep → no offset divergence at any step.
- `mixed_finish` uses equal `max_new_tokens` and same-length prompts → equal stop times → no divergence.

A test with **mixed `max_new_tokens` + same-length prompts** (per Q5=α) produces divergence at the exact step the smaller cap hits. The active row's attention output is the production-visible artifact — see §3.2.

### 3.2 Why active-row outputs are corrupted (not just finished-row outputs)

Per-batch independence: SDPA's `Q @ K^T` is computed independently per batch row. Row 1's logit does NOT depend on row 0's stale K/V because the matmul indices over `[b=1, ..., :, :]` only.

**But:** the SDPA op operates on the full `[B, n_kv_heads, K_max, head_dim]` slab as a single tensor — every row's K_max columns are visible to that row's Q. For row 1, K_max = `max(offsets) = 8` (step 4) or `12` (step 8). Row 1's offset is also 8 / 12, so its `[0..K_max]` view is entirely real K — no stale positions to attend to. **Active rows are correct without a mask** when their own offset equals max_off.

**The corruption hits finished rows (row 0 above):** their offsets are < max_off, so their `[0..K_max]` slice contains stale positions. Row 0's output is discarded by `Scheduler::step` (no event emitted), but the forward still ran. **The bug is wasted compute + silent numerical noise on discarded rows, not active-row correctness.**

This nuance changes the urgency: 3c-2's mask is not a correctness fix for any currently-observable output (current tests pass bit-id 1.0000). It is correctness for **future** scenarios where finished-row outputs might be inspected (debugging, telemetry), and a hard prerequisite for 3c-3's evict/admit where slot reuse means the same B-row position transitions through multiple `(active → finished → vacated → new admit → active)` cycles and stale K/V in the buffer becomes a real corruption source.

### 3.3 Why "fix it now" rather than "defer to 3c-3"

1. **Cheap to fix:** ~40 lines of Scheduler code + a 1-line `forward_on` signature change. No new mlx ops, no new helpers, no new cache mutations. The infrastructure (`build_per_row_decode_mask`) is already shipped.

2. **Future-proofs 3c-3:** when 3c-3 introduces mid-batch admit, a new request's slot reuses a position whose cache previously held a finished row's K/V at `[0..old_offset]`. Without a mask, the new row's Q sees the old row's K/V as "real" data. 3c-3 has to ship the mask anyway; doing it here keeps the change small.

3. **Closes the helper-ships-without-caller gap:** Final reviewer for 3c-1 flagged `build_per_row_decode_mask` as a "ships unused; first caller lands in 3c-2" risk — code that's dead-on-arrival invites deletion or refactoring without realizing the dependency.

### 3.4 Industry reference (informs the masking surface)

- **vLLM:** PagedAttention with per-request page maps. No global K/V slab — each request has owned pages. No mask needed for cross-request isolation; attention kernel reads per-request page list directly.
- **SGLang:** Radix tree + per-request mask similar to vLLM.
- **HF transformers (legacy batched decode):** dense K/V cache + per-row attention_mask matching the cached length. The mask path is exactly what 3c-2 ships.
- **llama.cpp server:** dense K/V + per-row slot tracking with the same per-row real-len mask.

**ironmlx 3c-2 positioning:** dense K/V + `[B, 1, 1, max_K]` additive mask, computed by the Scheduler before each `forward_on`. Same pattern HF uses; same pattern 3c-3 will keep. Consistent with the right-pad migration from 3c-1.

## §4 Architecture

### 4.1 `Qwen35Model::forward_on` signature change

Before (3c-1):
```rust
pub fn forward_on(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    per_row_lens: Option<&[i32]>,
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

After (3c-2):
```rust
pub fn forward_on(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    per_row_lens: Option<&[i32]>,
    decode_mask: Option<&Array>,        // NEW
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

`decode_mask` semantics:
- `None` — single-stream caller (GenerationStream, B1-p2.1/2 per-stream baselines, etc.). SDPA runs in `mask_mode = "causal"`. Bit-identical to pre-3c-2 behavior for B=1.
- `Some(m)` where `m.shape() == [B, 1, 1, K]` — batched-decode caller (Scheduler::step). SDPA runs in `mask_mode = ""` + `mask_arr = Some(m)`, additive mask over the K dimension.

Body change: forward `decode_mask` to `text_model::forward_on` (added param), which forwards to `forward_post_embedding_on(attention_mask = decode_mask, ...)`. The existing `attention_mask` parameter is reused — it already routes to `Attention` / `GatedAttention` via `DecoderLayer::forward_on(.., full_attn_mask=attention_mask, ..)` and reaches mlx fast SDPA's `mask_arr` slot when `Some`.

### 4.2 `Qwen35Model::forward_vl_chunk` / `forward_vl` mirror change

Both gain `decode_mask: Option<&Array>` for signature symmetry. Current callers are all single-stream B=1 (P6 VL test suite + future B1-p2.4); they pass `None`. The parameter forwards to `text_model::forward_post_embedding_on(attention_mask = decode_mask, ...)`.

`forward_from_embeds` is single-stream; passes `None`.

`batched_prefill` is **not changed** — its `attention_mask: &Array` parameter is already passed to `forward_post_embedding_on(attention_mask = Some(attention_mask), ...)`. Prefill's mask and decode's mask never coexist (different call sites).

### 4.3 `Qwen35TextModel::forward_on` signature change

Before:
```rust
pub fn forward_on(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    per_row_lens: Option<&[i32]>,
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

After:
```rust
pub fn forward_on(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    per_row_lens: Option<&[i32]>,
    decode_mask: Option<&Array>,        // NEW
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

Body:
```rust
let hidden = self.embed_on(input_ids, target)?;
self.forward_post_embedding_on(
    &hidden, position_ids, cache,
    decode_mask,      // attention_mask = decode_mask
    None,             // linear_attention_mask
    per_row_lens, target,
)
```

`forward_post_embedding_on` itself is **not changed** — it already takes `attention_mask: Option<&Array>` (used by batched_prefill). Decode-mask plugs into the same slot.

### 4.4 Why `attention.rs` / `gated_attention.rs` / `decoder_layer.rs` / `gated_delta_net.rs` are NOT touched

- `DecoderLayer::forward_on` already routes `full_attn_mask: Option<&Array>` to `Attention::forward_on` and `GatedAttention::forward_on` in the Full-attention dispatch. `linear_attn_mask` routes to `GatedDeltaNet` (linear path) separately. 3c-2's decode mask goes through `attention_mask` → `full_attn_mask` → SDPA via the existing path.
- `Attention::forward_on` and `GatedAttention::forward_on` accept `mask: Option<&Array>`. When `Some`, they invoke `mlx::fast::scaled_dot_product_attention_on(.., scale, "", Some(m), None, target)`. mlx fast SDPA accepts `[B, N, T_q, T_kv]`-broadcastable additive masks. `[B, 1, 1, K]` broadcasts cleanly against `[B, n_heads, 1, K]` — no shape change.
- `GatedDeltaNet::forward_on` receives only `linear_attn_mask`. Decode mask is `attention_mask` (separate parameter slot in `forward_post_embedding_on`). Cannot leak into linear path.

### 4.5 `Scheduler::step` mask construction

Add a private helper:

```rust
// In ironmlx/src/core/scheduler.rs:
fn first_full_layer_offsets(cache: &[LayerCache]) -> Result<&[i32]> {
    cache.iter()
        .find_map(|c| match c {
            LayerCache::Full(kv) => Some(kv.offsets()),
            _ => None,
        })
        .ok_or_else(|| anyhow!(
            "Scheduler::step: no Full-attention layer in cache; per-row offsets unavailable"
        ))
}
```

In `step_inner` body, between building `per_row_lens` and calling `forward_on`:

```rust
// Pre-write offsets, read from the first Full-attention layer's KVCache.
// Lockstep advance across layers means any Full layer's offsets() works;
// Linear (GatedDelta) layers advance in sync — verified by the per-row
// offset infrastructure 3c-1 shipped.
let pre_offsets: Vec<i32> = first_full_layer_offsets(cache_ref)?.to_vec();

// Post-write real lens: active rows advance by 1, finished rows stay.
// Active: per_row_real_lens[i] = pre_offsets[i] + 1
// Finished or empty: per_row_real_lens[i] = pre_offsets[i] (frozen)
//
// Helper rejects l == 0 (zero-row contract from 3c-1 §4.7). Finished
// rows have pre_offsets[i] > 0 because they ran through prefill +
// at least one decode step, so this is always >= 1.
let per_row_real_lens: Vec<i32> = pre_offsets
    .iter()
    .zip(per_row_lens.iter())
    .map(|(off, n)| off + n)
    .collect();

let max_real_len = per_row_real_lens
    .iter()
    .copied()
    .max()
    .expect("b_max >= 1");

let decode_mask = build_per_row_decode_mask(
    &per_row_real_lens,
    max_real_len,
    Dtype::Bfloat16,
)?;

let logits = model.forward_on(
    &input_ids,
    &position_ids,
    Some(&per_row_lens),
    Some(&decode_mask),
    Some(cache_ref),
    (),
)?;
```

**Cache borrow timing:** `first_full_layer_offsets(cache_ref)` takes `&[LayerCache]`; `to_vec()` clones into an owned `Vec<i32>` so the immutable borrow ends before `model.forward_on(.., Some(cache_ref), ..)` re-borrows mutably. Verified by the borrow checker.

### 4.6 Edge case — every slot empty / all finished pre-step

If `active_count() == 0` at step entry, the existing phase guard catches it (step is illegal outside `Decoding`, and Decoding is exited when `all_done`). No additional handling needed.

If `pre_offsets[i] == 0` for any row (shouldn't happen after a successful `prefill_admitted` — every admitted row writes its prompt to cache), `per_row_real_lens[i]` could be 0 for an empty slot with `per_row_lens[i] == 0`. The `build_per_row_decode_mask` helper would Err. Mitigation: empty slots get `per_row_lens[i] = 0` and `pre_offsets[i] = 0` after prefill_admitted's synthetic prompt-length=1 fallback (which writes 1 token's worth at offset 0 → offset becomes 1). So `per_row_real_lens[i] == 1 > 0` even for empty slots. **No special-casing needed.**

(If a future change makes synthetic empty-slot prefill write 0 tokens, the helper Err surfaces it loudly — that's correct behavior.)

### 4.7 Invariants

1. `decode_mask.shape() == [B, 1, 1, max_real_len]` where `B == cache.b_max`.
2. `decode_mask.dtype() == Bfloat16` (matches SDPA promoted type used elsewhere in the model).
3. `per_row_real_lens.len() == B`, all entries `>= 1` (enforced by `build_per_row_decode_mask` contract from 3c-1).
4. `mask_value[i, 0, 0, k] == 0.0` if `k < per_row_real_lens[i]`, else `NEG_INFINITY`.
5. The mask is constructed inside `Scheduler::step` once per decode step; not cached across steps because `per_row_real_lens` changes every step.

### 4.8 Module surface summary

```text
ironmlx/src/core/cache/kv_cache.rs            — MODIFY (add 2 unit tests, ~30 lines)
  + kvcache_multi_step_accumulation
  + kvcache_per_row_data_isolation

ironmlx/src/core/cache/mtp_cache.rs           — MODIFY (delete 1 line)
  - stale TEMP(b1-p2.3c-1 Task 1) comment in tests module

ironmlx/src/core/generate.rs                  — MODIFY (~3 lines doc only)
  + build_per_row_decode_mask doc: "Production callers: Scheduler::step (3c-2)"

ironmlx/src/models/qwen3_5/model.rs           — MODIFY (~10 lines)
  + forward_on signature gains decode_mask: Option<&Array>
  + forward_vl_chunk / forward_vl mirror change
  + forward_from_embeds passes None
  + body forwards decode_mask to text_model

ironmlx/src/models/qwen3_5/text_model.rs      — MODIFY (~5 lines)
  + forward_on signature gains decode_mask: Option<&Array>
  + body forwards to forward_post_embedding_on(attention_mask = decode_mask, ...)
  (forward_post_embedding_on itself unchanged)

ironmlx/src/core/scheduler.rs                 — MODIFY (~30 lines)
  + first_full_layer_offsets private helper
  + step_inner builds per_row_real_lens + decode_mask
  + forward_on callsite passes Some(&decode_mask)

ironmlx/src/nn/attention.rs                   — no change
ironmlx/src/nn/gated_attention.rs             — no change
ironmlx/src/nn/decoder_layer.rs               — no change
ironmlx/src/nn/gated_delta_net.rs             — no change
ironmlx/src/core/server/scheduler_actor.rs    — no change
ironmlx/src/core/server/openai.rs             — no change
ironmlx/src/core/server/anthropic.rs          — no change

ironmlx/tests/b1_p2_3c_2_scheduler_decode_mask.rs  — NEW (~250 lines)
  + scheduler_per_row_finish_different_steps integration test

ironmlx/tests/b1_p2_1_batched_prefill.rs      — MODIFY (callsite updates)
ironmlx/tests/b1_p2_2_batched_decode.rs       — MODIFY (callsite updates)
ironmlx/tests/p4_qwen35_logits_match.rs       — MODIFY (callsite updates)
ironmlx/tests/p6_6_logits_match.rs            — MODIFY (callsite updates)
ironmlx/tests/p6_qwen35_vl_logits_match.rs    — MODIFY (callsite updates)
  + Each .forward_on(...) callsite gains None for decode_mask
```

## §5 Tests

### 5.1 Lib unit tests (new — from 3c-1 carry-over)

In `kv_cache.rs::tests`:

1. **`kvcache_multi_step_accumulation`** — B=2 cache, two successive `update_and_fetch(&k, &v, &[4, 4])` calls. Verify:
   - `c.offsets() == &[4, 4]` after first call
   - `c.offsets() == &[8, 8]` after second call
   - Returned slice from second call has shape `[2, n_kv_heads, 8, head_dim]`
   - Returned K values at `[i, :, 0..4, :]` match the first call's K values
   - Returned K values at `[i, :, 4..8, :]` match the second call's K values

2. **`kvcache_per_row_data_isolation`** — B=2 cache, single `update_and_fetch(&k, &v, &[4, 4])` with row-distinct K data (row 0 K filled with value `1.0`, row 1 K with value `2.0`). Verify the returned K slice has the correct per-row pattern at positions `[0..4]`:
   - Row 0 columns 0..4 = 1.0
   - Row 1 columns 0..4 = 2.0
   - No cross-row contamination

### 5.2 Existing test file callsite updates

All existing `model.forward_on(input_ids, pos_ids, per_row_lens, cache, target)` callers must gain `None` for the new `decode_mask` parameter:

- `tests/b1_p2_1_batched_prefill.rs` per-stream reference (line ~91)
- `tests/b1_p2_2_batched_decode.rs` per-stream reference (line ~121); per-stream decode (line ~142); batched decode (line ~284)
- `tests/p4_qwen35_logits_match.rs` (any direct callsite)
- `tests/p6_6_logits_match.rs` (any direct callsite)
- `tests/p6_qwen35_vl_logits_match.rs` (any direct callsite)
- `core/generate.rs::GenerationStream` internal callers (verify all hit None for decode_mask)

`Scheduler::step` is updated to pass `Some(&decode_mask)`. `Scheduler::prefill_admitted` does NOT change (`batched_prefill` is the prefill path, not decode).

### 5.3 New integration test `tests/b1_p2_3c_2_scheduler_decode_mask.rs`

Single scenario `scheduler_per_row_finish_different_steps` (Q5=α):

```rust
//! B=2 with same prompt but different max_new_tokens. Verifies:
//!   - row 0 transitions to finished='length' at step 3
//!   - row 1 continues independently to step 8
//!   - per-row tokens match B=1 GenerationStream baseline with the same
//!     prompt + same max_new_tokens at bit-id ≥ 0.95
//!   - cache offsets diverge: [L+3, L+3] after step 3, [L+3, L+8] after step 8

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn scheduler_per_row_finish_different_steps() {
    // load_fixture, tokenize_prompt, make_request, run_b1_baseline,
    // argmax_bit_id_ratio — mirror b1_p2_3b_3_admission_window helpers.

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop = tokenizer.eos_token_ids().to_vec();

    // B=1 baselines: same prompt, different max_new_tokens.
    let baseline_a = spawn_blocking(run_b1_baseline(prompt_ids.clone(), 3, ...));
    let baseline_b = spawn_blocking(run_b1_baseline(prompt_ids.clone(), 8, ...));

    // Drive Scheduler directly (not via Actor) to inspect step events
    // and cache offsets between steps.
    spawn_blocking(move || {
        let mut sched = Scheduler::new(2);
        sched.admit(make_request(prompt_ids.clone(), 3, stop.clone())).unwrap();
        sched.admit(make_request(prompt_ids.clone(), 8, stop)).unwrap();
        let _prefill_events = sched.prefill_admitted(&model).unwrap();

        let mut tokens_a = Vec::new();
        let mut tokens_b = Vec::new();
        // capture first prefill events
        ...

        // Decode loop: step until phase == Finished.
        loop {
            let events = sched.step(&model).unwrap();
            // collect per-row tokens
            for ev in &events { ... }

            // After step 3, row 0 should be finished with 'length'.
            // After step 3+, only row 1 should appear in events.

            if sched.phase() == Phase::Finished { break; }
        }

        // Compare to baselines.
        let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
        let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
        assert!(ratio_a >= 0.95, "row 0 bit-id {} < 0.95", ratio_a);
        assert!(ratio_b >= 0.95, "row 1 bit-id {} < 0.95", ratio_b);

        // Inspect cache offsets divergence.
        let cache = sched.cache_ref().expect("cache");  // exposed via test seam
        let off = first_full_layer_offsets(cache).unwrap();
        assert_eq!(off[1] - off[0], 5, "row 1 advanced 5 more steps than row 0");
    })
    .await.unwrap();
}
```

Note: `sched.cache_ref()` is a test-only accessor (gated `#[cfg(test)]` or `#[doc(hidden)]` `pub`). If exposing it requires invasive surgery, the assertion can instead read offsets indirectly via the scheduler's `active()` method or a step-event side channel.

### 5.4 Regression sweep

All 10 existing integration suites must PASS unchanged:

- P6.3 / P6.6 / P6.7 (VL + chunked-prefill — single-stream callers; verify decode_mask=None doesn't regress)
- B1-p2.1 batched prefill (prefill path; unchanged)
- B1-p2.2 batched decode (per-stream reference uses forward_on with None; batched path also uses forward_on with None — neither uses Scheduler::step so no mask is involved)
- B1-p2.3b-1 scheduler scenarios (`b2_happy` / `b4_happy` / `mixed_finish`) — now go through the mask path. Must keep bit-id 1.0000 on all three.
- B1-p2.3b-2 scheduler_actor scenarios (also through mask path)
- B1-p2.3b-3 admission_window scenarios (also through mask path; 2-row + 4-row concurrent admits)
- B1-p2.3b-4 anthropic_actor scenarios (also through mask path)
- B1-p2.3c-1 per_row_offset scenarios (Scenario 1 explicitly uses forward_on directly without Scheduler — verify decode_mask=None for that callsite)

## §6 Acceptance gates

- All 2 new KVCache lib unit tests + 1 new integration scenario PASS
- All 10 existing regression suites PASS unchanged (timings within ±10% of 3c-1 close-out, token output bit-identical to pre-3c-2 for lockstep paths)
- `cargo +nightly fmt --check`, `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`, `cargo build --release -p ironmlx`: clean
- Lib test count: 202 (3c-1) + 2 (KVCache unit tests) = **204 lib tests**
- `b1_p2_3b_1::mixed_finish` bit-id remains 1.0000 (mask should NOT regress; ideally is mathematically more correct under the same numerics)

## §7 Estimate

**3 working days** (smaller than 3c-1 — no scope-expansion risk, no nn/* layer changes):

- Day 1 — Task 1: 3c-1 carry-over cleanup (3 items, all small)
- Day 2 — Task 2: `Qwen35Model::forward_on` + `text_model::forward_on` signature change + all callsite updates + Task 3 start: `Scheduler::step` mask construction
- Day 3 — Task 4: new integration scenario + full regression sweep + close-out

## §8 Compat sunset notes

3c-2 inherits all 4 sunset markers from 3b series + 3c-1:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-2 introduces no new compat. The `forward_on` signature change is a hard break (no shim, consistent with 3c-1's `per_row_lens: Option<&[i32]>` precedent).

## §9 Risk register

| Risk | Mitigation |
| --- | --- |
| mlx fast SDPA doesn't accept `[B, 1, 1, K]` mask at decode (T_q=1) — shape mismatch with `[B, n_heads, 1, K]` expected by kernel | Task 2 starts with a small lib test directly calling `Attention::forward_on` with a `[1, 1, 1, K]` mask + cache; if it fails, broadcast the mask to `[B, 1, 1, K]` explicitly in `build_per_row_decode_mask` (it already does) — SDPA broadcast rules handle `[B, 1, 1, K] × [B, n_heads, 1, K]` by replicating along axis 1. Escalate if the kernel surface differs |
| `first_full_layer_offsets` borrows cache immutably while `forward_on` needs mutable borrow — borrow checker conflict | Clone offsets into `Vec<i32>` (already specified in §4.5: `.to_vec()`). Immutable borrow ends before mutable. |
| `b1_p2_3b_1::mixed_finish` bit-id regresses below 1.0000 because the mask changes the numerics from the "lockstep without mask" baseline | Investigate per-row delta: if it's < 1e-3 max_abs_diff (purely numerical refinement, no argmax flip) then accept; if argmax flips investigate which row + which token + cross-reference against B=1 baseline. Mask numerics are MORE correct so any regression flagged is likely a test-baseline staleness, not a real bug. Boss-flag if argmax flips ≥ 5%. |
| New integration test depends on `sched.cache_ref()` accessor that doesn't exist | Add `#[cfg(test)] pub(crate) fn cache_ref(&self) -> Option<&[LayerCache]>` private accessor; or use `step` return values + verified offsets via mock route; if invasive, drop the offset divergence assertion and keep bit-id assertion only |
| Performance regression from mask build per step (CPU `vec![]` + `try_into Array` + `astype` runs every step) | At b_max=4, max_K=2048: 32KB f32 alloc + bf16 cast per step = sub-millisecond CPU; SDPA dominates GPU time. Expected overhead < 1% of step time; measure in regression sweep. Escalate if any suite slows > 10%. |
| GenerationStream's internal `forward_on` callers regress when adding `decode_mask = None` (silent change of behavior?) | None is bit-identical to pre-3c-2 (SDPA falls through to `mask_mode = "causal"`). Verified by P6.3 + B1-p2.1 + B1-p2.2 (per-stream) suites passing unchanged. |

## §10 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Mask reuse strategy | Pass `decode_mask` through existing `attention_mask` param in `forward_post_embedding_on` (no nn/* changes) | New `decode_mask` parameter through DecoderLayer / Attention / GatedAttention signatures (would require attention.rs / gated_attention.rs / decoder_layer.rs changes; large blast radius for no semantic gain) |
| `forward_on` signature | Extend with `decode_mask: Option<&Array>` parameter | New `forward_on_with_mask` method (API surface bloat; Scheduler dispatch becomes "which forward to call" instead of "what to pass") |
| Finished-row mask semantics | per_row_real_lens[i] = cache.offsets()[i] (frozen at finish) | Mask row all-zero (silent semantic violation; would let finished row's attention output contaminate downstream layers if accidentally inspected) |
| `Phase` enum | Unchanged (Idle / Admitting / Decoding / Finished) | Add `PartiallyFinished` (no behavior change relative to Decoding; cosmetic only) |
| Cache borrow approach | Clone offsets into `Vec<i32>` to release immutable borrow before `forward_on` | Hold immutable borrow across multiple computations + use RefCell internally (over-engineered for a 4-i32 vec) |
| Carry-over cleanup placement | Fold into Task 1 of 3c-2 plan | Standalone 3c-1.x commit (clean but adds inter-sub-phase commit) |
| Integration test design | Mixed `max_new_tokens` (Q5=α) | Same `max_new_tokens` + different prompt lengths (overlaps b1_p2_3b_1_mixed_finish; doesn't exercise step-asynchronous finish); EOS-driven finish (brittle vs model changes) |

## §11 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-14-b1-p2-3c-1-per-row-offset-design.md`](2026-05-14-b1-p2-3c-1-per-row-offset-design.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_1_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_1_closeout/report.md)
- 3c-1 helper to wire: [`ironmlx/src/core/generate.rs::build_per_row_decode_mask`](../../ironmlx/src/core/generate.rs)
- `Scheduler::step` current implementation (target of refactor): [`ironmlx/src/core/scheduler.rs:512`](../../ironmlx/src/core/scheduler.rs#L512)
- `Qwen35Model::forward_on` current signature (target of extension): [`ironmlx/src/models/qwen3_5/model.rs:93`](../../ironmlx/src/models/qwen3_5/model.rs#L93)
- `Qwen35TextModel::forward_on` current signature (target of extension): [`ironmlx/src/models/qwen3_5/text_model.rs:181`](../../ironmlx/src/models/qwen3_5/text_model.rs#L181)
- `KVCache` offsets accessor (used by `first_full_layer_offsets`): [`ironmlx/src/core/cache/kv_cache.rs`](../../ironmlx/src/core/cache/kv_cache.rs)
