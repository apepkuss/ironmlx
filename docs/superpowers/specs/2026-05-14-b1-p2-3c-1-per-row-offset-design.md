# B1-p2.3c-1 — Per-row KV cache offset (cache + model API merged)

**Date:** 2026-05-14
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-4 head `170c50b`)
**Predecessor sub-phase:** B1-p2.3b-4 — Anthropic handler refactor (closed at `170c50b`); the 3b series is complete and the SchedulerActor + admission window infrastructure are live for both OpenAI and Anthropic routes.
**Sibling sub-phases (rest of 3c series, separate specs):**
- **3c-2** — `Scheduler` lockstep state-machine relaxation (mid-batch admit/evict)
- **3c-3** — `SchedulerActor::driver_loop` continuous-batching activation (admission during active batch)
- **3c+** — chunked batched prefill (removes long-prompt GS fallback)
**Successor sub-phases:** 3d (admission queue + preemption + `ADMISSION_DEADLINE` config exposure), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Replace `KVCache.offset: i32` and `GatedDeltaCache.offset: i32` (single shared offset across all B rows) with `offsets: Vec<i32>` (per-row tracking).
2. Update `KVCache::update_and_fetch_on` and `GatedDeltaCache::advance` to accept `per_row_lens: &[i32]` so callers can specify how many tokens row `i` is actually writing in this call. This breaks the lockstep K/V-write assumption that has held since 3a.
3. Thread `per_row_lens: &[i32]` through `Qwen35Model::{batched_prefill, forward_on}` and the internal `Attention` / `GatedAttention` / `GatedDeltaNet` layers so the model side honors per-row write semantics.
4. Add `build_per_row_decode_mask(per_row_real_lens, max_len, dtype) -> Array` helper in `core/generate.rs`. Decode is now ragged: row `i` attends to `[0..per_row_real_lens[i]]` while other positions are `-inf` masked.
5. Ship as a single commit (merged 3c-1 + 3c-2-of-original-plan) so the lib build never sees a broken cache+model state. Boss preference: no `#[deprecated]` compat shims, no broken-build commit windows.

## §2 Non-goals

- **Scheduler lockstep relaxation.** Defer to 3c-2 (renumbered from original 3c-3). 3c-1 ships the cache+model machinery; `Scheduler::step` / `prefill_admitted` continue to pass uniform `per_row_lens` (lockstep-equivalent) until 3c-2.
- **Mid-batch admit/evict.** Defer to 3c-2.
- **Continuous-batching admission window.** Defer to 3c-3 (renumbered from original 3c-4) — needs the driver_loop refactor.
- **Long-prompt chunked prefill.** Defer to 3c+ — needs the per-row offset machinery 3c-1 ships, plus the model-API chunking that the chunked-prefill phase introduces.
- **PagedAttention / paged KV cache.** Defer to B1-p2.5 production hardening or later. ironmlx 3c sticks with the dense `[B, n_kv_heads, cap, head_dim]` layout and adds `Vec<i32>` offsets.
- **Decoupling `KVCache.batch` from `Scheduler.b_max`.** Cache `batch` is still hardcoded at construction; per-row offset doesn't change cache allocation shape.

## §3 Background

### 3.1 The 3b series completed the SchedulerActor; 3c lifts lockstep

3a–3b ship a lockstep scheduler: all rows must prefill together (`batched_prefill` writes K/V for every row at the same offset), all rows must decode together (each `step` advances cache.offset by 1 for every row), and `evict_all` resets the whole cache to start a new batch. This is sufficient for "static B>1 with admission window" (3b-3) but cannot do continuous batching — a row that finishes early holds its slot until every other row finishes too.

3c lifts the cache-level lockstep by giving each row its own offset. The 3c series unfolds in 4 sub-phases:

| Sub-phase | Scope | Status |
| --- | --- | --- |
| **3c-1** (this) | KVCache + GatedDeltaCache per-row offsets; model API accepts `per_row_lens`; per-row decode mask helper | merged 3c-1+3c-2 of original 5-phase plan to avoid broken-build commit |
| **3c-2** | `Scheduler` state machine relaxation: mid-batch admit/evict; `Phase` enum redesign; per-row `RequestState` decoupling from `max_len` | separate spec, depends on 3c-1 |
| **3c-3** | `SchedulerActor::driver_loop` continuous-batching activation: admission window can run during active Decoding | separate spec, depends on 3c-2 |
| **3c+** | Chunked batched prefill; removes long-prompt fallback to GS path | separate spec, depends on 3c-1 |

### 3.2 The lockstep cache constraint (from 3a spec §3.2)

`KVCache.offset: i32` is mutated by `update_and_fetch_on` ([`kv_cache.rs:127`](../../ironmlx/src/core/cache/kv_cache.rs#L127)): `self.offset = new_offset`. Internally, `write_at_offset` ([`kv_cache.rs:220-245`](../../ironmlx/src/core/cache/kv_cache.rs#L220)) writes K/V via:

```rust
slice_update_on(
    keys_full,
    k,
    [0_i32, 0, self.offset, 0],
    [self.batch, self.n_kv_heads, end, self.head_dim],
    [1_i32, 1, 1, 1],
    target,
)?;
```

The start tuple `[0, 0, self.offset, 0]` and end tuple `[batch, n_kv_heads, end, head_dim]` write **all B rows simultaneously to the same sequence slab** `[offset..offset+n_new)`. This is the lockstep K/V write assumption 3c-1 breaks.

External code never reads or writes `offset` directly — only `update_and_fetch_on` (internal) and `reset` (called by `Scheduler::evict_all`) mutate it. The `offset()` accessor is currently used only by tests ([`tests/p2_kv_cache.rs`](../../ironmlx/tests/p2_kv_cache.rs), `nn/gated_delta_net.rs::tests`). This makes the API replacement (delete `offset()` / add `offsets()`) low-risk.

### 3.3 Industry references (informs design choices)

- **vLLM**: PagedAttention. KV cache is a page pool (e.g., 16 tokens/page), each request tracks page pointers. Cache shape is `[num_pages, page_size, num_heads, head_dim]`. Per-row offset is implicit via the request's owned pages. Enables true continuous batching and prefix sharing.
- **SGLang**: Radix attention. Pages organized as a radix tree for prefix sharing across requests.
- **llama.cpp server**: Static slot allocation. Cache is `[seq, batch, heads, dim]` with each request fixed to a slot at construction. No true continuous batching — requests serialize per slot.

**ironmlx 3c-1 design positioning:**

| System | Data structure | Per-row offset | Cache shape | Mid-batch admission |
| --- | --- | --- | --- | --- |
| vLLM | Page pool + pointers | Implicit (pages) | `[pages, page_size, heads, dim]` | Yes (easy) |
| SGLang | Radix tree + leaves | Radix node + local offset | Tree of pages | Yes |
| llama.cpp | Dense buffer + map | Per-request length (no true offset) | `[seq, batch, heads, dim]` | No |
| **ironmlx 3c-1** | Dense buffer + `Vec<i32>` offsets | Explicit `Vec<i32>` | `[batch, heads, seq, dim]` | Enabled (driven by 3c-2/3c-3) |

3c-1 sits between llama.cpp (simpler than vLLM) and vLLM (more flexible than llama.cpp). It avoids the page-pool complexity but introduces real per-row offset tracking. This is the minimal viable design for continuous batching given ironmlx's current cache shape — a paged-attention rewrite is deferred.

### 3.4 Why merge cache + model API into a single sub-phase

The original brainstorm proposed 5 sub-phases: 3c-1 (cache only) → 3c-2 (model API) → 3c-3 (scheduler) → 3c-4 (driver) → 3c+ (chunked prefill). Boss preference rejects "broken build commits" — shipping 3c-1 alone (cache layer with new API) would leave `Qwen35Model::batched_prefill` calling the old signature, breaking lib compile until 3c-2 lands. Three options were considered:

| Option | Description | Verdict |
| --- | --- | --- |
| A. Merge 3c-1 + 3c-2 | Cache + model API in one commit | **Selected** — keeps lib green |
| B. `#[deprecated]` shim on cache | Old API wraps new; phase out in 3c-2 | Rejected — violates "no compat code" |
| C. Broken-build commit | Allow temporary failure between 3c-1 and 3c-2 | Rejected — not production-grade |

The merged 3c-1 spans ~440 lines across 8 files. It is the largest 3c-series sub-phase but produces a working build at every commit.

## §4 Architecture

### 4.1 `KVCache` per-row offset internals

**Struct definition** (after 3c-1):

```rust
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offsets: Vec<i32>,     // NEW — length == batch; per-row tracking
    cap: i32,
    step: i32,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}
```

`offsets` is initialized to `vec![0; batch as usize]` in `new()`. `reset()` sets all entries to `0` (preserves Array allocations as before).

**Public API (final)**:

```rust
pub fn new(batch: i32, n_kv_heads: i32, head_dim: i32, v_head_dim: i32, dtype: Dtype, cap: i32) -> Self;
pub fn with_step(mut self, step: i32) -> Self;
pub fn offsets(&self) -> &[i32];     // NEW — per-row read access
pub fn cap(&self) -> i32;
pub fn reset(&mut self);              // unchanged behavior; clears all offsets to 0
pub fn update_and_fetch(&mut self, k: &Array, v: &Array, per_row_lens: &[i32]) -> Result<(Array, Array)>;
pub fn update_and_fetch_on(
    &mut self,
    k: &Array,
    v: &Array,
    per_row_lens: &[i32],
    target: impl Into<StreamOrDevice>,
) -> Result<(Array, Array)>;
```

**Removed**: `pub fn offset(&self) -> i32` — replaced by `offsets()`.

**`update_and_fetch_on` semantics**:
- Input `k: [batch, n_kv_heads, S_max, head_dim]` and `v: [batch, n_kv_heads, S_max, v_head_dim]`.
- `per_row_lens.len() == batch`, all entries `>= 0`, all entries `<= S_max`.
- For each row `i`, writes `k[i, :, 0..per_row_lens[i], :]` to `cache.keys[i, :, offsets[i]..offsets[i]+per_row_lens[i], :]`; similarly for `values`.
- `per_row_lens[i] == 0` skips row `i` entirely (offset unchanged).
- After write: `offsets[i] += per_row_lens[i]` for each row.
- Returns the post-write fetched slices: `(keys_fetched, values_fetched)` shaped `[batch, n_kv_heads, max(offsets), head_dim]` (i.e., truncated to the maximum across all rows; positions `>= offsets[i]` in row `i` are stale — caller must mask them out).
- Error conditions:
  - `per_row_lens.len() != batch` → `Err`
  - `per_row_lens[i] < 0` → `Err`
  - `per_row_lens[i] > k.shape()[2]` → `Err`
  - `offsets[i] + per_row_lens[i] > cap` → `Err`

### 4.2 `GatedDeltaCache` per-row offset internals

**Struct definition** (after 3c-1):

```rust
pub struct GatedDeltaCache {
    conv_state: Array,         // [B, kernel_size-1, conv_dim]
    recurrent_state: Array,    // [B, Hv, Dv, Dk]
    offsets: Vec<i32>,         // NEW — length == B
    cap: i32,
}
```

`offsets` initialized to `vec![0; B]` in `new_with_cap`; cleared to all-zero in `reset`.

**Public API (final)**:

```rust
pub fn new_with_cap(b, kernel_size, conv_dim, hv, dv, dk, input_dtype, cap) -> Result<Self>;
pub fn conv_state(&self) -> &Array;
pub fn recurrent_state(&self) -> &Array;
pub fn offsets(&self) -> &[i32];           // NEW
pub fn cap(&self) -> i32;
pub fn update_conv(&mut self, new_conv_state: Array);
pub fn update_recurrent(&mut self, new_state: Array);
pub fn advance(&mut self, per_row_n: &[i32]) -> Result<()>;  // CHANGED — per-row n
pub fn reset(&mut self) -> Result<()>;     // unchanged behavior; clears all offsets to 0
```

**Removed**: `pub fn offset(&self) -> i32`.

Linear attention recurrent state is per-row by nature (`recurrent_state[B, ...]` is per-row), so per-row offset tracking is the only structural change. `update_conv` and `update_recurrent` continue to accept full-batch tensors — per-row semantics are encoded in the `per_row_n` argument to `advance`.

`advance(per_row_n)` rules:
- `per_row_n.len() == B`
- `per_row_n[i] >= 0`
- `offsets[i] + per_row_n[i] <= cap`
- Sets `offsets[i] += per_row_n[i]` for each row.

### 4.3 Model API: `per_row_lens` threading

```rust
// model.rs
pub fn batched_prefill(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    attention_mask: &Array,
    linear_attention_mask: &Array,
    per_row_lens: &[i32],          // NEW — actual prompt token count per row (no padding)
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;

pub fn forward_on(
    &self,
    input_ids: &Array,
    position_ids: &Array,
    per_row_lens: &[i32],          // NEW — usually [1; B] for decode
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

**Semantics**:
- `per_row_lens.len() == batch`. Caller specifies how many tokens row `i` writes.
- For prefill: `per_row_lens[i] == prompt_lens[i]` (the actual prompt length, NOT `max_len`). Cache writes only the real K/V, skipping left-pad positions.
- For decode: `per_row_lens = vec![1; B]` typically; pad rows (None slots, finished rows) can pass 0 to skip writing.
- `text_model.rs::forward_post_embedding_on` and `forward_on` propagate `per_row_lens` to each decoder layer; each layer's `Attention` / `GatedAttention` / `GatedDeltaNet` passes it to its cache's `update_and_fetch_on` / `advance`.

### 4.4 Attention layer internal forwarding

[`nn/attention.rs`](../../ironmlx/src/nn/attention.rs), [`nn/gated_attention.rs`](../../ironmlx/src/nn/gated_attention.rs), [`nn/gated_delta_net.rs`](../../ironmlx/src/nn/gated_delta_net.rs) all currently accept `cache: Option<&mut KVCache>` (or `&mut GatedDeltaCache`) and internally call `cache.update_and_fetch_on(&k, &v, target)`. After 3c-1:

```rust
pub fn forward_on(
    &self,
    ...,
    per_row_lens: &[i32],          // NEW
    cache: Option<&mut KVCache>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    ...
    let (k_full, v_full) = if let Some(cache) = cache {
        cache.update_and_fetch_on(&k, &v, per_row_lens, target)?
    } else {
        (k, v)
    };
    ...
}
```

`GatedDeltaNet` similarly propagates `per_row_lens` to `cache.advance(per_row_lens)`.

### 4.5 Per-row decode mask helper

```rust
// core/generate.rs (NEW)
pub fn build_per_row_decode_mask(
    per_row_real_lens: &[i32],     // each row's valid K/V length after this step
    max_len: i32,                   // = max(per_row_real_lens); the fetched cache slice's K dimension
    dtype: Dtype,
) -> Result<Array>;
```

**Returns** `[B, 1, 1, max_len]` mask:
- `mask[b, 0, 0, k] = 0.0` if `k < per_row_real_lens[b]` (real K/V — attention allowed)
- `mask[b, 0, 0, k] = f32::NEG_INFINITY` if `k >= per_row_real_lens[b]` (pad / stale — attention blocked)

Used during decode (`Scheduler::step` will pass this in 3c-2). Prefill continues to use the existing `build_batch_attention_mask` (`[B, 1, T_max, T_max]` causal+pad-aware) since prefill is still uniformly `T_max` columns wide regardless of per-row lengths — per-row offsets only affect cache K/V positions, not the mask shape for prefill itself.

**Why a new helper**: existing `build_batch_attention_mask` is shaped `[B, 1, T_q, T_kv]` with `T_q = T_kv = max_len` — it covers full causal attention with padding. Decode is shaped `[B, 1, 1, K_total]` (Q has only 1 row), so the existing builder doesn't apply. The new helper is decode-specific.

### 4.6 Cache write implementation strategies (plan-writing decides)

Spec lists three candidate strategies for the per-row K/V write in `KVCache::update_and_fetch_on`. Plan-writing chooses based on mlx-rs API availability + benchmark:

**Strategy A — Loop B times**:

```rust
for i in 0..(self.batch as usize) {
    let len_i = per_row_lens[i];
    if len_i == 0 { continue; }
    let start_i = self.offsets[i];
    let end_i = start_i + len_i;
    slice_update_on(
        keys_full, k_per_row_i,
        [i, 0, start_i, 0],
        [i + 1, n_kv_heads, end_i, head_dim],
        [1, 1, 1, 1],
        target,
    )?;
}
```

- ✅ Simple, no new mlx ops required
- ❌ O(B) GPU op count (B = 4 is fine; B = 64 might slow)

**Strategy B — `scatter_nd` (or similar gather/scatter kernel)**:

- ✅ Single op
- ❌ Requires mlx-rs to expose `scatter_nd` with the right axis support; may not be available

**Strategy C — Dense single write + per-row mask**:

- Keep current `slice_update_on(..., [0, 0, max_offset, 0], ...)` but zero out K/V at positions `[offsets[i]+per_row_lens[i] .. max_offset+per_row_lens_max]` for row `i` before the write
- ✅ Minimal change to current code
- ❌ Wastes GPU compute on writes to positions that get masked out anyway

Spec recommends **Strategy A** (simple, B small in current targets) as the default; plan-writing benchmarks against B and `prompt_len` and revisits if performance is unacceptable.

### 4.7 Invariants (compile-time + runtime)

1. `KVCache.offsets.len() == self.batch as usize` (constructor + all mutations preserve)
2. `KVCache.offsets[i] >= 0` AND `KVCache.offsets[i] <= cap` for all `i`
3. `per_row_lens.len() == cache.batch` (caller responsibility; checked in `update_and_fetch_on`)
4. `per_row_lens[i] >= 0` AND `per_row_lens[i] <= k.shape()[2]` (input K/V tensor must be long enough)
5. `offsets[i] + per_row_lens[i] <= cap` (cap check after write)
6. `GatedDeltaCache.offsets.len() == B` (same as KVCache)
7. `build_per_row_decode_mask`: `per_row_real_lens.len() == B`, `max_len >= max(per_row_real_lens)`, returns `[B, 1, 1, max_len]`

### 4.8 Module surface summary

```text
ironmlx/src/core/cache/kv_cache.rs           — MODIFY (~120 lines)
  + offsets: Vec<i32> field
  + offsets() accessor
  + update_and_fetch_on takes per_row_lens: &[i32]
  - offset() accessor REMOVED
  + Strategy-A per-row loop in write_at_offset

ironmlx/src/core/cache/gated_delta.rs        — MODIFY (~80 lines)
  + offsets: Vec<i32> field
  + offsets() accessor
  + advance(per_row_n: &[i32])
  - offset() accessor REMOVED

ironmlx/src/core/cache/mod.rs                — no change
  (LayerCache::reset dispatcher unchanged)

ironmlx/src/core/generate.rs                 — MODIFY (~60 lines added)
  + build_per_row_decode_mask helper

ironmlx/src/models/qwen3_5/model.rs          — MODIFY (~30 lines)
  + batched_prefill takes per_row_lens
  + forward_on takes per_row_lens

ironmlx/src/models/qwen3_5/text_model.rs     — MODIFY (~20 lines)
  + Thread per_row_lens through forward_on / forward_post_embedding_on

ironmlx/src/nn/attention.rs                  — MODIFY (~30 lines)
  + forward_on takes per_row_lens, passes to cache

ironmlx/src/nn/gated_attention.rs            — MODIFY (~30 lines)
  + forward_on takes per_row_lens, passes to cache

ironmlx/src/nn/gated_delta_net.rs            — MODIFY (~20 lines)
  + forward_on takes per_row_lens, passes to cache.advance

ironmlx/src/core/scheduler.rs                — MODIFY (~10 lines)
  + prefill_admitted constructs per_row_lens = prompt_lens (lockstep-equivalent input)
  + step constructs per_row_lens = vec![1; b_max]
  (No state-machine relaxation — 3c-2's job)

ironmlx/src/core/server/scheduler_actor.rs   — no change

ironmlx/src/core/server/openai.rs            — no change
ironmlx/src/core/server/anthropic.rs         — no change

ironmlx/tests/p2_kv_cache.rs                 — MODIFY (~10 lines)
  + Update cache.offset() callsites → cache.offsets()[0]
  + Update cache.advance(n) → cache.advance(&vec![n; B])

ironmlx/tests/b1_p2_1_batched_prefill.rs     — MODIFY
  + Pass per_row_lens to batched_prefill (= prompt_lens)
  + cache.offset() → cache.offsets()[0]

ironmlx/tests/b1_p2_2_batched_decode.rs      — MODIFY
  + Pass per_row_lens to batched_prefill / forward_on

ironmlx/tests/b1_p2_3b_1_scheduler_step.rs   — MODIFY (if test calls model directly)

ironmlx/tests/b1_p2_3c_1_per_row_offset.rs   — NEW (5 integration scenarios)
```

## §5 Tests

### 5.1 Cache unit tests (new)

In `kv_cache.rs::tests`:
1. `kvcache_per_row_offsets_initial_zero`
2. `kvcache_per_row_write_uniform_lens`
3. `kvcache_per_row_write_mixed_lens`
4. `kvcache_per_row_zero_len_skips_row`
5. `kvcache_reset_clears_all_offsets`
6. `kvcache_per_row_lens_len_mismatch_returns_err`
7. `kvcache_per_row_lens_negative_returns_err`
8. `kvcache_per_row_lens_exceeds_k_returns_err`
9. `kvcache_per_row_cap_exceeded_returns_err`

In `gated_delta.rs::tests`:
10. `gdcache_per_row_offsets_initial_zero`
11. `gdcache_advance_uniform`
12. `gdcache_advance_mixed`
13. `gdcache_advance_invalid_returns_err`
14. `gdcache_reset_clears_all_offsets`

Net: +14 lib tests.

### 5.2 Existing test file updates

- `tests/p2_kv_cache.rs`: 6 assertions touch `cache.offset()` → change to `cache.offsets()[0]` (B=1 in those tests). `cache.advance(n)` (in 1 gated test) → `cache.advance(&vec![n; B])`.
- `tests/b1_p2_1_batched_prefill.rs`: model.batched_prefill callsite passes `per_row_lens = &prompt_lens` (existing variable).
- `tests/b1_p2_2_batched_decode.rs`: prefill passes `per_row_lens = &prompt_lens`; decode forward_on calls pass `per_row_lens = &vec![1i32; B]`.
- `tests/b1_p2_3b_1_scheduler_step.rs`: any direct model call (likely indirect via Scheduler) — Scheduler itself updates so test code unaffected.
- `tests/b1_p2_3b_2_scheduler_actor.rs`, `b1_p2_3b_3_admission_window.rs`, `b1_p2_3b_4_anthropic_actor.rs`: indirect via SchedulerActor/Scheduler — no callsite update needed.

### 5.3 New integration test `tests/b1_p2_3c_1_per_row_offset.rs`

5 `#[ignore]` `#[tokio::test(...)]` scenarios:

**Scenario 1 — `per_row_offset_uniform_lens_matches_lockstep_baseline`**
1. Load model + tokenizer.
2. Build B=2 prompts of equal length, prompt_lens = `[16, 16]`.
3. Run **lockstep baseline**: `batched_prefill(..., per_row_lens = &[16, 16], ...)` — by spec, this should produce identical cache state to the pre-3c-1 lockstep code.
4. Read `cache.offsets()` → should be `[16, 16]`.
5. Run decode loop for 8 steps with `per_row_lens = &[1, 1]` each step.
6. After 8 steps: `cache.offsets() == [24, 24]`.
7. Assert tokens match B=1 GenerationStream baseline at bit-id ≥ 0.95 per row.

**Scenario 2 — `per_row_offset_ragged_lens_offsets_diverge`**
1. B=2, ragged prompts: prompt_lens = `[8, 16]`.
2. `batched_prefill(..., per_row_lens = &[8, 16], ...)` — row 0 writes only 8 K/V; row 1 writes 16.
3. Assert `cache.offsets() == [8, 16]`.
4. Slice cache directly (read accessor) and check row 0's `[8..16]` slab is still zero (stale region — not written).

**Scenario 3 — `per_row_offset_zero_len_skips_row`**
1. B=2, `per_row_lens = [0, 12]`.
2. `batched_prefill` with row 0's prompt_ids = pad zeros (it's inactive).
3. Assert `cache.offsets() == [0, 12]` after prefill (row 0 unchanged).
4. Row 1 cache `[0..12]` populated normally.

**Scenario 4 — `per_row_offset_decode_with_ragged_offsets`**
1. From Scenario 2 state (cache.offsets = `[8, 16]`).
2. Decode step with `per_row_lens = [1, 1]` and `build_per_row_decode_mask(&[9, 17], 17, Bfloat16)`.
3. Row 0 attention should only see K/V at positions `0..8` (mask `-inf` at 8..17).
4. Row 1 attention sees K/V at `0..17`.
5. Sample row 0's logit + sample row 1's logit.
6. Cross-check vs B=1 GenerationStream baseline for each row's prompt — bit-id ≥ 0.95 per row.

**Scenario 5 — `per_row_offset_invalid_args_return_err`**
1. `per_row_lens.len() != B` → `Err`.
2. `per_row_lens[i] = -1` → `Err`.
3. `per_row_lens[i] = k.shape()[2] + 1` → `Err`.
4. `offsets[i] + per_row_lens[i] > cap` → `Err`.

### 5.4 Regression sweep updates

All 9 existing integration suites (P6.3 / P6.6 / P6.7 / B1-p2.1 / B1-p2.2 / B1-p2.3b-1 / 3b-2 / 3b-3 / 3b-4) must pass unchanged after callsite updates. Bit-id and token output should be identical (`per_row_lens = prompt_lens` and `[1; B]` produce lockstep-equivalent behavior to current code).

## §6 Acceptance gates

- All 14 new cache unit tests + 5 new integration scenarios PASS
- All 9 existing regression suites PASS unchanged (token output and bit-ids must match pre-3c-1 baselines for uniform-`per_row_lens` calls)
- `cargo +nightly fmt --check`, `clippy -D warnings`, `cargo build --release -p ironmlx`: clean
- Lib test count: 188 (3b-4) + 14 cache unit tests = **~202 lib tests**, integration test count + 5 scenarios

## §7 Estimate

**5–7 working days** (largest 3c-series sub-phase):
- Day 1 — `KVCache` + `GatedDeltaCache` internal refactor + unit tests
- Day 2 — `core/generate.rs` `build_per_row_decode_mask` helper + unit tests
- Day 3 — `Qwen35Model::batched_prefill` + `forward_on` API change + `text_model` + `nn/attention.rs` + `nn/gated_attention.rs` + `nn/gated_delta_net.rs` threading
- Day 4 — `Scheduler::prefill_admitted` + `step` callsite updates (passes lockstep-equivalent `per_row_lens`)
- Day 5 — All existing test callsite updates + first 3 new integration scenarios (uniform / ragged / zero_len)
- Day 6 — Remaining 2 integration scenarios (decode-with-ragged-offsets, invalid args) + full regression sweep + close-out
- Day 7 (buffer)

## §8 Compat sunset notes

3c-1 inherits all 4 sunset markers from 3b series:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-1 introduces no new compat. The `offset()` → `offsets()` API change is a hard break (no shim), consistent with Boss's no-compat-code preference.

## §9 Risk register

| Risk | Mitigation |
| --- | --- |
| mlx-rs `slice_update_on` doesn't accept dynamic per-row indices → Strategy A loop is the only option | Plan-writing benchmarks Strategy A at B=4 to confirm GPU op overhead is acceptable; if not, escalate to Strategy B/C |
| Per-row decode mask numerics drift vs current implicit-causal decode | Scenario 1 (uniform-lens) is the equivalence test — bit-id must match pre-3c-1 baseline at ≥ 95%. Drift > 0.19 max_diff requires root-cause investigation |
| Callsite update breaks pre-3b tests (P6.3, etc.) | All callsites identified in §4.8 module-surface table; regression sweep catches any miss |
| `GatedDeltaCache::advance` per-row may interact poorly with conv_state shape (single-batch shared) | conv_state is `[B, kernel_size-1, conv_dim]` — already per-batch. Per-row advance only changes when each row "consumes" tokens from the conv window. No structural conflict |
| Some existing test uses `cache.offset()` reflectively or via macro | Grep `offset(` exhaustively; only direct callsites should exist |
| 5–7d estimate slips because of mlx-rs op surprises | Buffer Day 7; if more time needed, prefer correctness over schedule (Boss preference) |

## §10 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Cache layout | Dense `[B, n_kv_heads, cap, head_dim]` + `Vec<i32>` offsets | PagedAttention (too large, deferred); Per-row separate KVCache instances (loses batched attention vectorization) |
| Per-row write strategy | Loop B times (Strategy A) — plan-writing benchmarks | `scatter_nd` (mlx availability unknown); Dense-write-then-mask (wasted compute) |
| API shape (legacy `offset()`) | Hard delete | `#[deprecated]` shim (Boss preference forbids); Keep + add `offsets()` (API ambiguity) |
| Scope merger | 3c-1 (cache) + 3c-2-original (model API) merged | Sequential commits with broken-build window (Boss preference forbids); `#[deprecated]` shim |
| Decode mask helper | New `build_per_row_decode_mask` | Generalize existing `build_batch_attention_mask` (API ambiguity — different Q/K shapes) |
| Mask output shape | `[B, 1, 1, max_len]` (cap-truncated) | `[B, 1, 1, cap]` (wastes compute on always-masked positions) |
| Tests for per-row write Strategy A vs C | Functional tests only; benchmark in plan-writing | Performance regression tests inline (premature optimization) |

## §11 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-13-b1-p2-3b-4-anthropic-handler-design.md`](2026-05-13-b1-p2-3b-4-anthropic-handler-design.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md)
- KVCache current (target of refactor): [`ironmlx/src/core/cache/kv_cache.rs`](../../ironmlx/src/core/cache/kv_cache.rs)
- GatedDeltaCache current: [`ironmlx/src/core/cache/gated_delta.rs`](../../ironmlx/src/core/cache/gated_delta.rs)
- Model API targets: [`ironmlx/src/models/qwen3_5/model.rs`](../../ironmlx/src/models/qwen3_5/model.rs)
- Existing decode mask construction (reference): [`ironmlx/src/core/generate.rs:323-344`](../../ironmlx/src/core/generate.rs#L323)
- Scheduler `prefill_admitted` / `step` (callsite updates): [`ironmlx/src/core/scheduler.rs`](../../ironmlx/src/core/scheduler.rs)
