# P5i.a T2 — fused gate+up gather_qmm equivalence evidence

**Scope.** Documents the bit-exact equivalence between the LANDED fused
gate+up `gather_quantized_matmul` path (`5f9a269`) and the prior two-call
legacy path (gate_proj `gather_qmm` then up_proj `gather_qmm`). This file
exists to satisfy the Codex P5i.a P2 finding #3 — the production code
intentionally contains ONLY the fused path (per spec § 4.3 review), so the
PoC's dual-path comparison helper was removed before the LAND commit; this
doc preserves the methodology + measured data + mathematical argument that
underpinned the correctness gate.

## 1. PoC methodology (PRE-LAND, env-gated)

Pre-LAND we ran the fused path AND the legacy two-call path side-by-side
under an env gate `IRONMLX_P5I_A_T2_DUAL_PATH=1`. For each forward call,
each MoE layer:

1. Computed fused output via single `gather_qmm` on the concatenated
   `[E, 2*I, K_packed]` weight then sliced along axis=-1 into
   `(gate_out_fused, up_out_fused)`.
2. Computed legacy outputs via two separate `gather_qmm` calls on the
   original `gate_proj` / `up_proj` weights, producing
   `(gate_out_legacy, up_out_legacy)` of identical shape.
3. Asserted `max_abs(gate_out_fused - gate_out_legacy) ≤ 1e-2` and
   `max_rel ≤ 5e-2`, same for `up_out`.

Helper + env gate were both removed in `5f9a269` per spec § 4.3
Codex revision ("production code must contain ONLY the selected fused
path"). They are not present on `main` or on `ironmlx-p5i-a-gather-qmm-feasibility`.

## 2. Measured data (PoC, recorded `reports/p5i-a-bench-log.md`)

Model: `mlx-community/Qwen3.5-35B-A3B-4bit` (4-bit affine, group_size=64,
E=128 experts, I=512 moe_intermediate, H=4096 hidden).

| branch | shape (BS, k=8) | max_abs | max_rel | run coverage |
|---|---|---|---|---|
| default (BS=4, BS*k=32) | gate_out `[4, 8, 1, 512]`, up_out `[4, 8, 1, 512]` | **0.000e0** | **0.000e0** | 64 layers × smoke prompt |
| sorted (BS=64, BS*k=512) | gate_out `[512, 1, 512]`, up_out `[512, 1, 512]` | **0.000e0** | **0.000e0** | 64 layers × long-prompt probe |

Both branches: bit-exact. Every position of every tensor matches between
fused and legacy paths.

## 3. Mathematical argument — why bit-exact is expected

4-bit affine quantization per-row scale/bias is stored as:

- `weight[e, r, k_packed]`: packed 4-bit codes
- `scales[e, r, g]`, `biases[e, r, g]`: one (scale, bias) per group of
  `group_size=64` along K.

Dequantization is purely row-local: `dequant[e, r, k] = code[e, r, k] *
scales[e, r, k / group_size] + biases[e, r, k / group_size]`. The result
for output row `r` depends ONLY on that row's `(weight, scales, biases)`
slice — no cross-row state.

Concatenating gate_proj and up_proj along the intermediate axis
(`axis=1`, the row axis) is therefore a permutation-free row-wise
rearrangement. The fused weight `W_fused[e, r, k]` for `r ∈ [0, I)`
equals `W_gate[e, r, k]`, and for `r ∈ [I, 2I)` equals `W_up[e, r-I, k]`.
Same for scales and biases. The K-axis groups are unchanged.

Inside `gather_qmm`, the per-row dot-product `out[e, r, n] = sum_k
dequant(W[e, r, k]) * x[..., n, k]` is computed independently per output
row. No accumulator is shared across the I→2I axis. Therefore fused
slice `out_fused[..., 0:I]` is bit-identical to legacy `gate_out`, and
`out_fused[..., I:2I]` is bit-identical to legacy `up_out` — INCLUDING
the rounding pattern, because the same FMA sequence on the same input
data executes on the same Metal kernel.

This is independent of `sorted_indices` because that flag only changes
how `rhs_indices` is consumed for the gather; the per-row matmul math is
unchanged.

## 4. Why the LAND commit removed the dual-path helper

Per spec § 4.3 Codex revision (cited in `reports/p5i-a-results-codex-review.md`
§ 2 T2): "production code must contain ONLY the selected fused path".
Reasons:

- Dual-path doubles weight memory at runtime (~16 GB on 35B model).
- Two `gather_qmm` calls per layer per forward defeats the perf point.
- Env-gated dead code in the hot path is a maintenance hazard.

The compromise: bit-exact evidence is preserved here (this file) plus the
T2 commit message body in `5f9a269` plus the PoC log
`reports/p5i-a-bench-log.md` (gitignored per `[feedback_no_reports_commit]`
but locally reproducible).

## 5. Re-running this comparison

If a future reviewer wants to re-verify equivalence without the env-gated
helper, the cleanest path is:

1. Branch off `ironmlx-p5i-a-gather-qmm-feasibility` HEAD.
2. Restore `RoutedExperts::from_loader` to keep `gate_weight` /
   `up_weight` / `gate_scales` / `up_scales` (+ optional biases) as
   separate fields alongside the fused tensors (do NOT call
   `drop(source)` after the fused build).
3. In `SparseMoeBlock::forward_on` substep 3 (gather_qmm_gate_up),
   compute both fused-and-sliced AND legacy-two-call outputs in parallel
   on the same `target` stream.
4. Assert `max_abs == 0.0` and `max_rel == 0.0` element-wise.
5. Discard the branch — DO NOT merge.

## 6. References

- Commit `5f9a269` — perf(p5i-a-t2): fuse gate_proj + up_proj gather_qmm into single call
- `docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md` — spec § 4.3 (Level (c) fusion + dual-path helper removal directive)
- `docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md` — plan T2 Step 3.3 (correctness gate)
- `reports/p5i-a-bench-log.md` (gitignored) — PoC dual-path data + bench measurements
- `reports/p5i-a-results-codex-review.md` (gitignored) — § 2 T2 close-out + Codex P2 finding #3
