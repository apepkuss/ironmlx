# B1-p2.1 Static Batched Prefill (Model-Level) — Design

**Status:** Approved (brainstormed 2026-05-12)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (5-phase decomposition)
**Branch target:** `ironmlx-b1-p2-batched-serving` (cut from `ironmlx-p6-7-vl-chunked-prefill` head `343f173`)

## 0. Program context — B1-p2 5-phase decomposition

P6.4 audit item B1 ("Multi-image batch (B>1)") was deferred as a multi-week P-track. Phase 1 (multi-image-per-request) shipped as P6.6. **Phase 2** — multiple independent chat requests packed into one forward — is large enough to need decomposition itself. The accepted plan:

| Sub-spec | Scope | Estimate |
| --- | --- | --- |
| **B1-p2.1** (this doc) | Static batched prefill, model-level API only | ~3–4 d |
| B1-p2.2 | Static batched decode + KV cache hand-off | ~1 w |
| B1-p2.3 | Continuous batching (scheduler + admit/evict + token-level loop) | ~2–3 w |
| B1-p2.4 | VL B>1 (multi-stream with per-stream images) | ~1 w |
| B1-p2.5 | Production hardening (admission control, OOM safety, fairness) | ~1 w |

Each sub-spec ships working software on its own. B1-p2.1 ships a **correctness foundation** — no throughput change, but proves the model forward is numerically correct at B>1 so the later phases can build on it without ambiguity.

## 1. Motivation

ironmlx currently assumes a single in-flight request per `GenerationStream` instance. Each request gets its own `Vec<LayerCache>` and its own forward call. The model-level forward path (`Qwen35Model::forward_on`, `Qwen35Model::forward_vl`) is **implicitly B=1**: `build_position_ids` returns `[3, 1, S]`, `mlx::fast::scaled_dot_product_attention_on` is invoked with `mask_mode="causal"` (no per-sequence segmentation).

To serve multiple requests in one forward (the prerequisite for any throughput improvement), the model needs a `batched_prefill` entry point that:

1. Accepts B prompts packed into a single `[B, S_max]` tensor with left-padding;
2. Uses per-batch-row position ids `[3, B, S_max]`;
3. Applies an attention mask that is both causal AND respects the left-padding boundary per row;
4. Writes one `KVCache` instance allocated with `batch=B` so all subsequent batched-decode work has the right state.

P6.6 already proved the multi-image scatter path is N-extensible at B=1. B1-p2.1 proves the transformer forward is B-extensible. Together they unblock the later phases.

## 2. Goals

- Add public method `Qwen35Model::batched_prefill` with the signature in §4.1.
- Add public helpers in `core/generate.rs`: `build_position_ids_batched`, `build_batch_attention_mask`.
- Numerical equivalence: for any (B, prompt set), the last-position logits from `batched_prefill` for batch row i must match `forward_on(prompt_i)` to within max_abs_diff < 1e-3, and the greedy argmax must be bit-identical.
- KV cache contents: after `batched_prefill`, each batch row's KV cache slice (`[i, :, :, :]`) must be numerically equivalent to what `forward_on(prompt_i)` would have written to a `batch=1` cache.
- Acceptance: 4 test points (B ∈ {2, 4} × {same-length, mixed-length}) PASS the above two checks.
- No regression on the single-request path:
  - P6.3 Task 21 single-image: max_diff = 0.3906, first_token = 760
  - P6.6 N=2 / N=3: max_diff / first_token unchanged
  - P6.7 6-point matrix: all chunk_sizes still produce first_token = 760
  - `cargo test -p ironmlx --lib --release`: ≥ 156 passed (P6.7 baseline) plus the new helper unit tests

## 3. Non-goals

- HTTP server / OpenAI handler changes — deferred to B1-p2.2 (server orchestration)
- Batched decode (`next_token` at B>1) — deferred to B1-p2.2
- Continuous batching, dynamic admit/evict — deferred to B1-p2.3
- VL B>1 (one of the B streams carries images) — deferred to B1-p2.4; **`batched_prefill` does NOT accept `pixel_values`**
- Throughput benchmarking — deferred to B1-p2.5
- TP / PP / multi-process — out of P-track entirely

## 4. Architecture

### 4.1 New public API on `Qwen35Model`

Insert after the existing `forward_vl_chunk` / `forward_vl` cluster in `ironmlx/src/models/qwen3_5/model.rs`:

```rust
#[allow(clippy::too_many_arguments)]
pub fn batched_prefill(
    &self,
    input_ids: &Array,            // [B, S_max] int32; left-padded
    position_ids: &Array,         // [3, B, S_max] int32 (MRoPE 3-stream per-row)
    attention_mask: &Array,       // [B, 1, S_max, S_max] additive or boolean
    cache: Option<&mut [LayerCache]>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;               // [B, vocab] — last position per row, projected
```

`batched_prefill` runs `text.embed_on` → `text.forward_post_embedding_on` (with `attention_mask` threaded through) → `slice_last_and_project` on the last `S_max` position. Pure text — no vision tower.

The existing single-stream `forward_on` and `forward_vl_chunk` stay unchanged; they pass `attention_mask=None` to the new threaded parameter (see §4.5) which routes attention.rs back to its current `mask_mode="causal"` behavior. **The single-stream path is byte-identical to today**.

### 4.2 Attention forward refactor

`ironmlx/src/nn/attention.rs::forward_on` currently calls:

```rust
mlx::fast::scaled_dot_product_attention_on(&q, &k_full, &v_full, self.scale, "causal", None, None, target)
```

Refactor: add a new optional argument `attention_mask: Option<&Array>` to the public `forward_on` signature. Inside:

- If `attention_mask.is_none()`: keep the exact current call (`mask_mode="causal"`, `mask_arr=None`). Numerical equivalence with the old code is by construction.
- If `attention_mask.is_some()`: switch to `mask_mode=""` and pass the array. mlx fast SDPA accepts arrays at most 4-dim broadcastable to `[B, N, T_q, T_kv]` (verified in `mlx/python/src/fast.cpp:215-225`).

### 4.3 Position ids for B>1

`core/generate.rs` currently has `build_position_ids(start: i32, n: i32) -> Array` returning `[3, 1, n]` with all three MRoPE streams identical (`0..n`).

Add helper:

```rust
/// Build MRoPE position ids for a batched, left-padded prefill.
/// Returns `[3, B, max_len]`. For batch row i with actual length L_i,
/// the trailing L_i positions hold `0..L_i-1`; the leading
/// `max_len - L_i` positions hold 0 (will be masked out by attention).
pub fn build_position_ids_batched(
    prompt_lens: &[i32],
    max_len: i32,
) -> Result<Array>;
```

Rationale for pad-position = 0: HuggingFace and vLLM both use this convention. The attention mask (§4.4) zeros out the pad columns and rows, so any RoPE applied at pad positions has no effect on real-position logits.

### 4.4 Attention mask construction

Add helper:

```rust
/// Build an additive attention mask `[B, 1, max_len, max_len]` (bfloat16)
/// for a left-padded batch. For batch row i with actual length L_i,
/// position `pad_start_i = max_len - L_i`:
///   mask[i, 0, q, k] = 0.0    iff (q >= pad_start_i) AND (k >= pad_start_i) AND (k <= q)
///                   = -inf   otherwise
pub fn build_batch_attention_mask(
    prompt_lens: &[i32],
    max_len: i32,
    dtype: Dtype,
) -> Result<Array>;
```

The mask combines three constraints: (a) causal (`k <= q`), (b) left-padding boundary on rows (`q >= pad_start_i`), (c) left-padding boundary on columns (`k >= pad_start_i`).

Boolean mask alternative is supported by mlx fast SDPA, but additive (bf16 0 / −inf) is closer to existing call-site conventions and avoids dtype mismatch when summed with attention scores.

### 4.5 Threading `attention_mask` through the model

The mask must reach `attention.rs::forward_on`. Touched files:

- `ironmlx/src/models/qwen3_5/text_model.rs` — `forward_post_embedding_on` gains `attention_mask: Option<&Array>`, passed to each layer.
- `ironmlx/src/models/qwen3_5/text/layer.rs` — layer `forward_on` gains `attention_mask: Option<&Array>`, passed to `self_attn.forward_on`.
- `ironmlx/src/nn/attention.rs` — already covered in §4.2.

Existing call sites (`Qwen35Model::forward_on`, `Qwen35Model::forward_vl_chunk`) pass `None`. `batched_prefill` passes `Some(&mask)`.

### 4.6 KV cache batched allocation

No code change to `KVCache` or `make_cache`; the existing `make_cache(batch, cap, dtype)` already accepts arbitrary batch. `batched_prefill` callers pass `batch=B`. Per the comment at `core/cache/kv_cache.rs:21`, single-request-per-cache is by design; this is the first call site that actually exercises `batch>1`.

## 5. File structure

| File | Change |
| --- | --- |
| `ironmlx/src/models/qwen3_5/model.rs` | NEW `batched_prefill` method |
| `ironmlx/src/models/qwen3_5/text_model.rs` | `forward_post_embedding_on` gains `attention_mask: Option<&Array>` |
| `ironmlx/src/models/qwen3_5/text/layer.rs` | layer `forward_on` gains `attention_mask: Option<&Array>` |
| `ironmlx/src/nn/attention.rs` | `forward_on` gains `attention_mask: Option<&Array>`; routes to `mask_mode="" + array` or `"causal" + None` |
| `ironmlx/src/core/generate.rs` | NEW `build_position_ids_batched`, NEW `build_batch_attention_mask`, 4 inline unit tests for helpers |
| `ironmlx/tests/b1_p2_1_batched_prefill.rs` | NEW 4-point matrix integration test |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_1_closeout/report.md` | NEW close-out report |

No new fixtures — synthetic input_ids drawn from a fixed seed are enough for correctness.

## 6. Acceptance

### 6.1 Helper unit tests (inline in `core/generate.rs`)

- `build_position_ids_batched_basic` — B=2 with same length, verify shape `[3, 2, n]` and per-row content
- `build_position_ids_batched_padded` — B=2 with `prompt_lens = [3, 5]`, max_len = 5, verify pad-position-0 + real `0..L_i-1`
- `build_batch_attention_mask_causal` — B=1 same-length, verify it matches mlx fast SDPA's `"causal"` mode (allowed cells = lower triangle, additive 0)
- `build_batch_attention_mask_padded` — B=2 mixed-length, verify pad rows and columns are -inf

### 6.2 Integration test `b1_p2_1_batched_prefill.rs`

4 points, all run within one `#[test]`:

| Point | B | prompt_lens | Pass criterion |
| --- | --- | --- | --- |
| 1 | 2 | [128, 128] | for i in 0..2: max_abs(batched_prefill_last_logits[i] − forward_on(prompt_i).last_logits) < 1e-3 AND argmax(batched_prefill[i]) == argmax(forward_on(prompt_i)) |
| 2 | 2 | [128, 96] | same |
| 3 | 4 | [128, 128, 128, 128] | same |
| 4 | 4 | [128, 96, 64, 128] | same |

Synthetic prompts: generate using a fixed RNG seed within the vocab range; no real tokenizer needed. The reference is `forward_on` on each prompt individually with a fresh `batch=1` cache.

KV cache equivalence: after each batched call, slice `cache[layer].keys[i, :, :, :]` and compare to the cache state from the per-stream reference. Tolerance: max_abs < 1e-3.

### 6.3 Regression gates

- `cargo +nightly fmt --all -- --check`: clean
- `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`: clean
- `cargo build --release -p ironmlx`: clean
- `cargo test -p ironmlx --lib --release -- --test-threads=1`: ≥ 156 passed + 4 new helper tests = 160 passed
- P6.3 Task 21 single-image logits-match: PASS, max_diff = 0.3906, first_token = 760
- P6.6 N=2 / N=3 logits-match: PASS, baselines unchanged
- P6.7 6-point matrix (`p6_7_chunked_prefill`): PASS, all chunk_sizes → 760

## 7. Risks + rollback

**R1 — mlx fast SDPA array mask in bf16.** Verified API (`mlx/mlx/fast.h:47`, `mlx/python/src/fast.cpp:215-225`): mask can be at most 4-dim, broadcast-compatible with `[B, N, T_q, T_kv]`, boolean or additive (additive must promote to q/k/v dtype). Actual perf and numerics under bf16 array mask have not been benchmarked here; the integration test's < 1e-3 gate catches any SDPA-kernel surprise. **Mitigation**: Point 1 (B=2, same length) is the simplest verification; if it fails the regression bisects to attention.rs.

**R2 — Single-stream regression.** The threading change touches `forward_post_embedding_on`, `layer.rs::forward_on`, `attention.rs::forward_on`. Existing call sites must pass `attention_mask=None` and hit the unchanged `mask_mode="causal"` path. **Mitigation**: P6.3 / P6.6 / P6.7 regression gates re-run after each commit; the first commit of the threading change is the canary.

**R3 — Pad-position RoPE leak.** Pad cells get position 0 in MRoPE, then RoPE rotates them. If the attention mask doesn't fully zero out pad-row and pad-column attention scores, the rotated pad-cell K/V could leak into real-row output. **Mitigation**: Mask zeros both pad rows AND pad columns (additive -inf). Mixed-length test points (2 and 4) catch any leak: a single-token mask error shifts the last-position logits visibly above 1e-3.

**R4 — KV cache batch-write correctness.** This is the first non-test code path that calls `KVCache` with `batch>1`. The cache code claims to support it (per the constructor signature and `grow_to` logic), but no caller has exercised it. **Mitigation**: §6.2's KV-equivalence check directly probes the cache contents per batch row.

**Rollback strategy.** Each commit is independent — adding the helper (4.3, 4.4), refactoring attention.rs (4.2), threading through model layers (4.5), adding `batched_prefill` (4.1), and the test (6.2) each commit separately. If R2 surfaces, revert the threading commits; if R3/R4 surface, the helper or attention-mask construction is the suspect.

## 8. Estimated effort

| Phase | Work | Estimate |
| --- | --- | --- |
| B1-p2.1a | Helpers (`build_position_ids_batched`, `build_batch_attention_mask`) + 4 unit tests | 0.5 d |
| B1-p2.1b | `attention.rs::forward_on` gains `attention_mask` parameter; thread through layer.rs / text_model.rs | 0.5 d |
| B1-p2.1c | `Qwen35Model::batched_prefill` method | 0.5 d |
| B1-p2.1d | 4-point integration test + KV-equivalence check | 1 d |
| B1-p2.1e | Debug + close-out + regression sweep | 1–2 d |
| **Total** | | **~3–4 working days** |

## 9. Out of scope (deferred to later B1-p2.x sub-specs)

- HTTP server / OpenAI handler → B1-p2.2
- Batched decode (`next_token` at B>1) → B1-p2.2
- Continuous batching, scheduler, admit/evict → B1-p2.3
- VL B>1 (one stream carries images) → B1-p2.4
- Throughput benchmarking → B1-p2.5
- TP / PP / multi-process → out of program
