# B1-p2.2 Static Batched Decode — Design

**Status:** Approved (brainstormed 2026-05-12)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (5-phase decomposition; B1-p2.1 shipped on `ironmlx-b1-p2-batched-serving`)
**Branch target:** `ironmlx-b1-p2-2-batched-decode` (cut from `ironmlx-b1-p2-batched-serving` head `b24aae8`)

## 0. Program context

B1-p2.1 ✅ shipped: `Qwen35Model::batched_prefill` produces `[B, 1, vocab]` last-position logits from B prompts packed left-padded into one forward. KV cache (batch=B) is populated at offset = `S_max` after that call. Verified 4 points (B ∈ {2, 4} × {same-length, mixed-length}) with `max_abs_diff = 0.000977` (= bf16 ULP).

B1-p2.2 (this) extends to **decode** — advancing B streams one token at a time using the KV cache that batched_prefill populated. Phase 2 stays at model-level; HTTP server, scheduler, per-row stop, sampler scaffold are deferred to B1-p2.3.

## 1. Motivation

B1-p2.1 verified `KVCache::write_at_offset` correctness for a single write at offset=0 (prefill). The decode path exercises a different code path: `update_and_fetch` increments `offset` by 1 each step. **This increment path has never been verified at B>1.** Single-stream P6 / P6.3 / P6.7 tests cover B=1 decode; B1-p2.1 covers B>1 prefill. The intersection — B>1 decode — is the untested region.

Phase 2 closes that gap with a 4-point × 4-step matrix.

## 2. Goals

- Verify `Qwen35Model::forward_on([B, 1], [3, B, 1], cache(batch=B), ())` produces logits numerically equivalent to per-stream `forward_on` across N decode steps.
- Add helper `build_decode_position_ids(per_row_pos: &[i32]) -> Result<Array>` returning `[3, B, 1]`.
- Numerical contract per step per batch row: `max_abs_diff(batched[i], per_stream[i]) < 1e-3` AND greedy argmax bit-identical.
- KV cache contents implicitly verified: any per-row K/V cell divergence would propagate to logits above the 1e-3 threshold (same proxy argument as B1-p2.1).
- No regression on single-stream paths (P6.3, P6.6, P6.7) or B1-p2.1.

## 3. Non-goals

- HTTP server / OpenAI handler changes → B1-p2.3
- `GenerationStream` refactor for B>1 (per-row histories, per-row finished flags) → B1-p2.3
- Per-row stop logic, sampler scaffold → B1-p2.3
- Continuous batching, dynamic admit/evict → B1-p2.3
- VL B>1 → B1-p2.4
- Per-row early-stop / different-offset KV cache rows → B1-p2.3 (continuous batching prereq)
- Throughput benchmarking → B1-p2.5
- Adding a `batched_decode` wrapper on `Qwen35Model` — `forward_on` already accepts `[B, 1]` transparently (attention.rs uses `batch = dims[0]`); a wrapper would be a non-functional alias. Per Boss preference "do not write compat code unless needed", reuse `forward_on`.

## 4. Architecture

### 4.1 Decode forward — reuse `forward_on`

```rust
// At the integration-test caller level (or future scheduler):
let logits = model.forward_on(
    &input_ids,        // [B, 1] uint32 — newly-sampled tokens, one per row
    &position_ids,     // [3, B, 1] int32 — from build_decode_position_ids
    Some(&mut cache),  // batch=B, populated by batched_prefill, offset = S_max + step
    (),
)?;
// logits: [B, 1, vocab]
```

Inside, the call chain is:
1. `text_model::forward_on` calls `embed_on([B, 1])` → `[B, 1, hidden]`
2. `forward_post_embedding_on(hidden, pos_ids, cache, None /* attention_mask */, target)`
3. Each `DecoderLayer::forward_on` calls `attention::forward_on(.., mask=None, ..)` (single-stream path)
4. `attention::forward_on(mask=None)` calls SDPA with `mask_mode="causal"` (post Task 3 of B1-p2.1)
5. SDPA "causal" lower-right alignment: at T_q=1, T_kv=cache_len, the single query attends to all keys ≤ itself per batch row. **Per-row causality is automatic.**

This path is **already exercised at B=1 in every single-stream decode test** (P6.3, P6.6, P6.7). Phase 2 verifies it at B>1.

### 4.2 Decode position_ids — new helper

B1-p2.1's `build_position_ids_batched(prompt_lens, max_len)` is prefill-semantics: each row's trailing `L_i` positions fill `0..L_i-1`. **It does not produce the decode-step shape we need.**

Decode step k needs `[3, B, 1]` where row i holds `current_pos[i]` (a single scalar). All three MRoPE streams hold the same value (text-only convention).

```rust
/// Build MRoPE position ids for one batched decode step.
/// Returns `[3, B, 1]` int32. Each batch row i holds the position id
/// `per_row_pos[i]` for its new token; all three MRoPE streams hold
/// the same value (text-only convention; VL B>1 in B1-p2.4 will need
/// a multi-stream variant).
pub fn build_decode_position_ids(per_row_pos: &[i32]) -> Result<mlx::Array>;
```

Body: ~15 lines. Validate `per_row_pos` non-empty, validate each `>= 0`. Build flat `[3*B]` int32 by tiling `per_row_pos` three times, reshape to `[3, B, 1]`.

### 4.3 Per-row position tracking — uniform within phase 2

After batched_prefill, all B streams share the same offset (= `S_max`, because B1-p2.1 left-padded all prompts to `S_max`). Decode step k advances offset by 1 for every row uniformly:

```text
step 0 (= prefill output):  offset = S_max
step 1:                      offset = S_max + 1, current_pos[i] = S_max for all i
step 2:                      offset = S_max + 2, current_pos[i] = S_max + 1
...
step k:                      offset = S_max + k, current_pos[i] = S_max + k - 1
```

Per-row early-stop (rows with different offsets) is **deferred to B1-p2.3**. Phase 2 always advances all rows together.

### 4.4 KV cache hand-off — already supported

`KVCache::new(batch=B, ...)` and `update_and_fetch` already support B>1; only the single shared `offset` exists, which matches phase 2's uniform-advance assumption. No KV cache code changes.

## 5. File changes

| File | Change |
| --- | --- |
| `ironmlx/src/core/generate.rs` | NEW `build_decode_position_ids` + 2 inline unit tests |
| `ironmlx/tests/b1_p2_2_batched_decode.rs` | NEW 4-point × 4-step integration test |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_2_closeout/report.md` | NEW close-out |

No changes to: `Qwen35Model`, `text_model.rs`, `attention.rs`, `KVCache`, `cross_modal.rs`, or any other source file.

## 6. Acceptance

### 6.1 Helper unit tests (inline in `core/generate.rs`)

- `build_decode_position_ids_basic` — `per_row_pos = [10, 20]` → shape `[3, 2, 1]`, values match
- `build_decode_position_ids_rejects_empty` — empty slice → Err

### 6.2 Integration test `b1_p2_2_batched_decode.rs`

4 points × 4 decode steps = 16 step-level checks. Same fixture configuration as B1-p2.1 to reuse synthetic-prompt LCG:

| Point | B | prompt_lens | decode steps |
| --- | --- | --- | --- |
| 1 | 2 | [128, 128] | 4 |
| 2 | 2 | [128, 96] | 4 |
| 3 | 4 | [128, 128, 128, 128] | 4 |
| 4 | 4 | [128, 96, 64, 128] | 4 |

Per point:
1. Generate synthetic prompts (deterministic LCG, same as B1-p2.1)
2. **Per-stream reference**: for each prompt i, run `forward_on` prefill + 4 decode steps with a fresh batch=1 cache. Record each step's last_logits and sampled (greedy) token.
3. **Batched path**:
   a. Build left-padded `[B, S_max]` input_ids, `[3, B, S_max]` position_ids (via `build_position_ids_batched`), `[B, 1, S_max, S_max]` mask (via `build_batch_attention_mask`), `make_cache(batch=B, cap=S_max+5, ...)`.
   b. `batched_prefill(...)` → `[B, 1, vocab]` last_logits.
   c. Greedy-sample B tokens from last_logits → `[B, 1]` next_input.
   d. **Decode loop (k = 1..=4)**:
      - `pos = vec![S_max as i32 + k - 1; B]`
      - `pos_ids = build_decode_position_ids(&pos)` → `[3, B, 1]`
      - `logits = forward_on(&next_input, &pos_ids, Some(&mut cache), ())` → `[B, 1, vocab]`
      - For each row i:
        - `max_abs_diff(logits[i], per_stream[i].step_k_logits) < 1e-3`
        - `argmax(logits[i]) == argmax(per_stream[i].step_k_logits)`
      - Sample B tokens → next_input for step k+1

If any check fails, panic with point + step + row index + observed diff.

### 6.3 Regression gates

- `cargo +nightly fmt --all -- --check`: clean
- `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`: clean
- `cargo build --release -p ironmlx`: clean
- `cargo test -p ironmlx --lib --release -- --test-threads=1`: ≥ 162 passed (B1-p2.1 baseline 160 + 2 new helper tests)
- P6.3 Task 21 single-image logits-match: PASS, max_diff = 0.3906, first_token = 760
- P6.6 logits-match: PASS, baseline unchanged
- P6.7 chunked-prefill 6-point matrix: PASS, all chunk_sizes → 760
- B1-p2.1 4-point batched prefill matrix: PASS, all 4 points, max_abs_diff ≈ 0.000977

## 7. Risks + rollback

**R1 — SDPA "causal" at T_q=1, T_kv>1.** mlx fast SDPA documents `"causal"` as lower-right alignment: the last query aligns with the last key. At T_q=1 + T_kv=cache_len, this gives the single query attention over all cache history including itself — correct per-row causal behavior. **Indirect verification**: P6.3 / P6.6 / P6.7 decode all run T_q=1 at B=1 and produce correct outputs. **Direct verification**: phase 2's logits-equivalence at B>1 is the new check.

**R2 — KV cache `update_and_fetch` increment at B>1.** Untested at this branch point. Phase 2's N=4 steps exercise `offset` advancing through (S_max, S_max+1, S_max+2, S_max+3). If a B>1 increment write goes to the wrong cache slot, the next step's attention will read stale K/V → logits divergence beyond 1e-3. The test will catch this immediately at step 2 of the first point.

**R3 — Mixed-length prefill leaving stale pad K/V in cache rows.** In points 2 and 4, prefill writes pad-token K/V into the leading positions of short-row cache slices (because attention computed K/V for pad tokens even though the mask zeroed their attention weights). When decode reads cache history, these stale K/V cells are in the query's attention window. **However**: the attention mask used for prefill zeroed pad-position attention weights, so the pad-position **outputs** (the hidden states post-attention) were correct; therefore the K/V cells subsequently written at pad-positions are based on correct upstream hidden states; therefore the decode-step attention over those K/V cells should produce results equivalent to per-stream (which also has the same pad-position K/V cells from its left-padded prefill). **Verification**: points 2 and 4 mixed-length directly probe this. If decode-step logits diverge in points 2/4 but not in 1/3 (same-length), the pad-K/V hypothesis is confirmed and we escalate (rollback to "phase 2 same-length only" + deeper investigation).

**R4 — Sampler determinism.** Phase 2 uses pure greedy (argmax). No temperature, no top-k, no penalties. Argmax in the test is implemented as a stable max-by, identical between batched and per-stream paths.

**Rollback**: each commit is independent. If R3 surfaces, phase 2 can ship with same-length-only points (1, 3) and defer mixed-length to a B1-p2.2.5 investigation. The new helper (build_decode_position_ids) is purely additive — removing the test alone reverts phase 2 cleanly.

## 8. Estimated effort

| Phase | Work | Estimate |
| --- | --- | --- |
| B1-p2.2a | `build_decode_position_ids` + 2 unit tests | 0.5 d |
| B1-p2.2b | 4-point × 4-step integration test | 1 d |
| B1-p2.2c | Debug + close-out + regression sweep | 0.5 d |
| **Total** | | **~2 working days** |

## 9. Out of scope (deferred to later B1-p2.x sub-specs)

- HTTP server / OpenAI handler → B1-p2.3
- `GenerationStream` B>1 refactor → B1-p2.3
- Per-row stop logic, sampler scaffold → B1-p2.3
- Continuous batching → B1-p2.3
- Per-row early stop, dynamic shrink batch → B1-p2.3
- VL B>1 → B1-p2.4
- Throughput benchmarking → B1-p2.5
- Non-greedy sampling at B>1 → B1-p2.3 onwards
