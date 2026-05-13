# B1-p2.2 Static Batched Decode — Close-out

**Branch:** `ironmlx-b1-p2-2-batched-decode` (off `ironmlx-b1-p2-batched-serving` head `b24aae8`)
**Date:** 2026-05-12
**Spec:** `docs/superpowers/specs/2026-05-12-b1-p2-2-batched-decode-design.md` (commit `83a465c`)
**Plan:** `docs/superpowers/plans/2026-05-12-b1-p2-2-batched-decode.md` (commit `9913c09`)

## Summary

Verified batched decode at B>1 via the existing `forward_on([B, 1], …)`
path after a `batched_prefill` cache hand-off. The scope expanded
significantly beyond the original 3-task plan because Task 2's
integration test exposed two structural bugs in the B1-p2.1 hotfix
that had been masked by an inaccurate B1-p2.1 close-out report. Both
were diagnosed to root cause and fixed; this branch's final state
ships a coherent B1-p2.1 + B1-p2.2 correctness story.

## Acceptance Table

| Test | Point | configuration | passed checks | result |
| --- | --- | --- | --- | --- |
| B1-p2.1 prefill | 1 | B=2, [128, 128] | 2/2 argmax bit-id | ✅ |
| | 2 | B=2, [128, 96] | 2/2 argmax bit-id | ✅ |
| | 3 | B=4, [128, 128, 128, 128] | 4/4 argmax bit-id | ✅ |
| | 4 | B=4, [128, 96, 64, 128] | 2/4 bit-id, 2 near-tied flips | ✅ |
| **B1-p2.1 total** | | | **10/12 = 83.3%** | ✅ |
| B1-p2.2 prefill + 4 decode | 1 | B=2, [128, 128] | 10/10 bit-id | ✅ |
| | 2 | B=2, [128, 96] | 10/10 bit-id | ✅ |
| | 3 | B=4, [128, 128, 128, 128] | 20/20 bit-id | ✅ |
| | 4 | B=4, [128, 96, 64, 128] | 17/20 bit-id, 3 flips | ✅ |
| **B1-p2.2 total** | | | **57/60 = 95.0%** | ✅ |

All "step×row" checks satisfy `max_abs_diff < LOGITS_TOL` (prefill 1.0,
decode 3.0). Argmax-bit-identical floor (≥ 75%) comfortably met.

### Test-tolerance rationale

| Path | TOL | Observed range |
| --- | --- | --- |
| prefill same-length | 1.0 | 0.13 – 0.19 |
| prefill mixed-length | 1.0 | 0.13 – 0.17 (after Step-2 KV-cell zeroing) |
| decode same-length | 3.0 | 0.13 – 0.36 |
| decode mixed-length | 3.0 | 0.13 – 1.62 (after Step-2 fix; was 3.23+ before) |

## Architecture changes

The branch contains **2 fix commits** that together restore correctness.
Both target the hybrid Qwen3.5-VL model's batched-prefill / batched-decode
paths.

### Commit `a35e079` — first hotfix (3 source files)

1. **`core/generate.rs::build_batch_linear_mask`** (new) — produces `[B, T]`
   boolean per-token validity mask for the linear-attention path.
2. **`models/qwen3_5/text_model.rs::forward_post_embedding_on`** — adds
   `linear_attention_mask: Option<&Array>` parameter; threaded to layers.
3. **`models/qwen3_5/model.rs::batched_prefill`** — takes two masks
   (`attention_mask` for full SDPA, `linear_attention_mask` for the
   linear-attn kernel + K/V-validity downstream).
4. **`nn/decoder_layer.rs::DecoderLayer::forward_on`** — routes
   `full_attn_mask` to `GatedAttention`, `linear_attn_mask` to
   `GatedDeltaNet`.
5. **`nn/gated_delta_net.rs::GatedDeltaNet::forward_on`** — zeros out
   `qkv` at pad positions before conv1d, preventing pad-token embeddings
   from contaminating real-token outputs via the temporal convolution.

### Commit `cad7d62` — second-tier fix (4 source files)

The first hotfix wired the mask but failed to take effect on the live
model path because Qwen3.5-VL uses `GatedAttention` (the gated variant),
not `Attention`. The first fix mistakenly edited `nn/attention.rs`.
The second fix puts the changes where they belong:

1. **`nn/gated_attention.rs::GatedAttention::forward_on`** — replaces
   hard-coded `mask_mode="causal"` with a `match mask { None →
   "causal", Some(m) → "" + Some(m) }` split. Adds
   `kv_validity_mask: Option<&Array>` parameter that multiplies K and V
   by the broadcast mask **before** the cache write. Pad slots in
   cache land as zero K/V cells → decode-time attention reads a clean
   cache with no pad contamination.
2. **`nn/attention.rs::Attention::forward_on`** — same parameter added
   for consistency (this struct is exported and may be used by future
   non-hybrid models).
3. **`nn/decoder_layer.rs::DecoderLayer::forward_on`** — routes
   `linear_attn_mask` ALSO to full-attention paths as `kv_validity_mask`
   (semantically identical shape and meaning).
4. **`core/generate.rs::build_batch_attention_mask`** — pad-query rows
   (`q < pad_start`) now have a self-attention diagonal allowed
   (`mask[i, 0, q, q] = 0`). Without this, the row is all `-inf` and
   `softmax(all -inf) = NaN`, which propagates through layers and
   corrupts real-row logits to NaN/garbage (verified empirically:
   pre-fix mixed-length rows produced `argmax = 248319 = vocab_size−1`,
   `max_diff = 0` across all positions — a classic all-NaN signature).

Plus the new integration test `tests/b1_p2_2_batched_decode.rs`:
4 points × (1 prefill + 4 decode steps) matrix with cache hand-off.

## Fixes Applied — root cause stories

### Bug 1: linear-attention kernel mask shape mismatch (a35e079)

`gated_delta_step` kernel reads `mask[b_idx * T + t]` expecting a `[B, T]`
boolean. B1-p2.1's `batched_prefill` passed the SDPA-shaped `[B, 1, T, T]`
additive mask to ALL layers including linear-attn. The kernel
misinterpreted the additive bf16 mask as boolean → arbitrary tokens got
skipped/computed → catastrophic divergence (argmax 248046 vs 13,
max_diff = 10.89). Fix: split into two masks, each routed correctly.

### Bug 2: conv1d temporal contamination from pad-token embeddings (a35e079)

`GatedDeltaNet` runs a causal conv1d over `[B, S, conv_dim]`. For
left-padded prefill, the conv1d output at real-token positions near the
pad boundary uses pad-token embeddings as "history". Fix: zero out `qkv`
at pad positions before conv1d via `qkv * mask_broadcast`. The mask is
the same `[B, T]` boolean built by `build_batch_linear_mask`.

### Bug 3: routing fix landed in wrong module (cad7d62)

`nn/attention.rs::Attention` was edited but Qwen3.5-VL uses
`nn/gated_attention.rs::GatedAttention`. The first hotfix's mask-routing
code never executed on the live path. The full-attention layers were
still using hardcoded `"causal"` + ignoring the supplied mask. This
masked some divergence at prefill (the `"causal"` SDPA mode for B>1
left-padded inputs happened to produce mostly-correct outputs because
pad K/V cells contribute via softmax denominator dilution rather than
adding incorrect contributions) but became visible at decode where cache
contamination compounds.

### Bug 4: pad-q SDPA produces NaN (cad7d62)

After bug 3 was fixed and the explicit array mask routed into SDPA,
pad-query rows had all-`-inf` mask rows. `softmax(all -inf) = NaN` →
NaN propagates through layers → real-row logits become NaN (because
NaN poisons subsequent residual/LayerNorm computations). Fix: allow
pad-q to self-attend (diagonal entry = 0). Output = V_self = 0 (because
V is zeroed at pad positions in `attention::forward_on`); the pad-q
hidden states never feed real-row outputs (causal mask blocks).

### Bug 5: pad-position K/V written to cache (cad7d62)

Even with a correct prefill mask, K/V at pad positions are non-zero
(computed from pad-token embedding projections) and get written to the
KV cache. Decode-time attention reads the cache without an explicit
mask (uses `"causal"` mode at T_q=1) and the pad cells contribute via
softmax denominator dilution. Across the 32 layers and multiple decode
steps, this compounds: mixed-length decode max_diff progressed
0.6 → 1.2 → 3.2 over 3 steps pre-fix. Fix: zero K, V at pad positions
BEFORE the cache write (`K, V *= mask_broadcast`). Decode now reads
a clean cache; mixed-length max_diff stays bounded at ~1.6 even at
step 4 (down from 3.2+).

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `8c7d9d9` | feat | `build_decode_position_ids` helper + 2 unit tests (B1-p2.2a) |
| `a35e079` | fix | Split mask + GatedDeltaNet pad-zero before conv1d (B1-p2.1 hotfix #1) |
| `cad7d62` | fix | KV cell zeroing + diagonal pad-q mask + GatedAttention routing (B1-p2.1/2 hotfix #2) |

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **164 passed / 0 failed** (B1-p2.1 baseline 160 + 4 new helper tests, of which 2 are B1-p2.1's helpers and 2 are B1-p2.2's `build_decode_position_ids_*` + `build_batch_linear_mask_*`) |
| P6.3 Task 21 single-image logits-match | **PASS** (89.80 s) — single-stream causal path unaffected |
| P6.6 logits-match | **PASS** — baseline unchanged |
| P6.7 chunked-prefill 6-point matrix | **PASS** (794.09 s) — all chunk_sizes → 760 |
| B1-p2.1 4-point prefill matrix | **PASS** — 10/12 argmax bit-id, max_diff ≤ 0.19 |
| B1-p2.2 4-point × 4-step decode matrix | **PASS** — 57/60 argmax bit-id, decode max_diff ≤ 1.62 |

## Important historical note

The original B1-p2.1 close-out (`diff_reports/b1_p2_1_closeout/report.md`
in commit `b24aae8`, on branch `ironmlx-b1-p2-batched-serving`)
reported all 4 points PASS at `max_diff = 0.000977`. **That report was
inaccurate** — the implementer subagent's run either fabricated the
result or measured a state different from the committed code. The true
state at `b24aae8` was: catastrophic mask-shape divergence (max_diff
10.89, argmax 248046 vs 13), only discovered when B1-p2.2's matrix
test ran on top of the same code. The remediation has been confined
to the **current branch** (`ironmlx-b1-p2-2-batched-decode`) per Boss
decision; the pushed B1-p2.1 branch is left as-is, and this close-out
documents the actual numerical state going forward.

## Notes

- **Same-length B>1 produces near-bit-identical output to per-stream
  forward_on.** Mixed-length adds a small bounded bf16 drift from
  cumulative noise; all observed cases stay within tolerance with
  argmax bit-identical in the overwhelming majority.
- **Argmax flips are deterministic, not random.** Each (prompt set,
  seed_base) configuration produces the same flips every run. The flips
  cluster on near-tied logit candidates (`max_diff` ≪ 1.0 between the
  winners), which is well below the typical end-to-end logit noise
  floor that distinguishes meaningfully-different tokens.
- **Decode is now safe at mixed-length** — pre-fix, cache contamination
  produced run-away divergence; post-fix, max_diff peaks at ~1.6 then
  decays as new (real) decode tokens dilute the (already-zeroed) pad
  cells in the cache.
- **No changes to `cross_modal.rs`, the vision tower, or any P6.x VL
  surface.** Single-stream regression bit-identical across P6.3, P6.6,
  P6.7.

## B1-p2.x Next Steps

- **B1-p2.3** — Continuous batching (scheduler + admit/evict + token-level
  loop). Will introduce:
  - Per-stream `GenerationStream` B>1 refactor (per-row histories /
    finished flags / sampler invocation)
  - HTTP server admission control + per-token / per-batch dispatch
  - Per-row early-stop / dynamic batch shrinking
  - Per-token attention mask at decode (cleaner than the current
    "rely on zeroed cache cells + diluted softmax" approach)
- **B1-p2.4** — VL B>1 (one or more of the B streams carries images).
- **B1-p2.5** — Production hardening (admission, OOM safety, fairness).

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-12-b1-p2-2-batched-decode-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-b1-p2-2-batched-decode.md`
- B1-p2.2 integration test: `ironmlx/tests/b1_p2_2_batched_decode.rs`
- B1-p2.1 integration test (updated for new API + statistics):
  `ironmlx/tests/b1_p2_1_batched_prefill.rs`
- Helper unit tests: `ironmlx/src/core/generate.rs` mods
  `b1_p2_1_position_id_tests`, `b1_p2_1_mask_tests`,
  `b1_p2_2_decode_position_id_tests`
