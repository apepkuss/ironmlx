# P5e Final — SparseMoeBlock gather_qmm Perf Optimization Close-Out

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Branch | ironmlx-p5e-perf |
| Spec | docs/superpowers/specs/2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md |
| Plan | docs/superpowers/plans/2026-05-19-ironmlx-p5e-gather-qmm-perf.md |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit (Qwen3.5-MoE-A3B, 128 experts, top-4) |
| Final HEAD | filled by close-out commit (see `git log`) |

## P5e wall-clock summary (M5 Max 128GB, Model::forward_on direct, 3-run median)

| PP | T0 baseline (ms) | Stage 1 final (ms) | Stage 2 final (ms) | Δ vs T0 | T0 tok/s → final tok/s |
|---|---|---|---|---|---|
| 128 | 127.66 | 128.16 | 116.99 | -8.36% | 1002.7 → 1094.2 |
| 512 | 488.30 | 488.61 | 307.37 | -37.06% | 1048.5 → 1665.7 |
| 2048 | 2067.45 | 2061.35 | 1076.99 | -47.91% | 990.6 → 1901.6 |

## Promotions

- **Stage 1 (Approach A: MLX op rearrangement):** ALL DISCARDED
  - A.1 stream parallelism: -6% to -17% regression
  - A.2 mlx::compile wrap: blocked by 4 safe-wrapper API gaps; no-op
  - A.3 shape elimination: ±0.5% noise
- **Stage 2 (Approach B.1: sorted routing):** SHIPPED as default code path
  - Threshold `SORTED_ROUTING_MIN_BS_K = 512` aligned to MLX `gather_qmm_rhs` fast-path floor (`B>=16 && B/E>=4` → `bs_k >= 4*128 = 512` for 128-expert model).

## Stage 2 outcome detail

Mechanism: sort tokens by expert id before the 3 `gather_quantized_matmul_on`
calls (gate / up / down), pass `sorted_indices=true`, and use inverse permutation
to scatter results back to original token order before weighted sum + shared
expert.

MLX fast path verified at `mlx/backend/metal/quantized.cpp:1484`: when
`right_sorted_ = sorted_indices && !rhs_indices_` is true and the shape condition
is met, MLX dispatches `gather_qmm_rhs` (specialized kernel) instead of the
generic `gather_qmm`. (T5 commit `b8e3f26` body referenced `lhs_indices_` in the
narrative — that was a typo; the formula in source at `mlx/ops.cpp:5359` is
`right_sorted_ = sorted_indices && !rhs_indices_`. T5 implementation passes
`rhs_indices = sorted_topk` and `lhs_indices = None`, so the predicate evaluates
correctly. Documenting here per CLAUDE.md "no-amend" — commit narrative typo
does not affect runtime behavior.)

Measured impact at PP=2048: 990.6 tok/s → 1901.6 tok/s (1.92× throughput).

## Validation gates (post-Stage-2)

- `p5_qwen35_moe_smoke` regression sentinel argmax=11: PASS
- `p5_qwen35_moe_batched` (B=2 vs B=1 per-row equivalence): PASS
- `p5_qwen35_moe_http_smoke` chat completion: PASS
- `sweep_full`: 19/19 in 157 seconds (Qwen3.5-4B-MLX-4bit, M5 Max)
- `cargo +nightly clippy --all-features --workspace --release -- -D warnings`: 0 warnings
- `cargo +nightly fmt --all -- --check`: clean
- `cargo build --release`: PASS

## Comparison to T0 profile expectations

T0 profile (`reports/p5e-t0-profile.md`) identified:
- 3× `gather_quantized_matmul` = 64.8% of PP=2048 prefill wall-clock
- Per-call: down (28.9%) + gate (18.4%) + up (17.5%)
- GatedDeltaNet 20.6% (second-largest hot path)

P5e changes targeted gather_qmm directly. Observed wall-clock change at PP=2048
of -47.91% suggests roughly 74% of the 64.8% gather_qmm budget was eliminated by
the sorted-path Metal kernel. (T0 forecast was that Stage 2 alone might cut the
gather_qmm portion in half; actual outcome substantially exceeded this.)

## Known debt / follow-ups (not P5e scope)

- **`forward_on` function size**: `SparseMoeBlock::forward_on` grew from ~150 LoC
  body to ~290 LoC body after T5. Per code reviewer recommendation, future
  cleanup could extract `forward_on_sorted_gather` and `scatter_back_to_order`
  helpers to restore readability without changing semantics. Not done in T5/T6
  per CLAUDE.md "don't refactor beyond what the task requires" and to keep T5
  reviewable.
- **A.2 mlx::compile follow-up**: 4 safe-wrapper API gaps remain (closure
  `'static`, private `LinearImpl/MlpImpl`, runtime M-aware dispatch in
  `Linear::forward_on`, integer reshape literals). A future "compile-everywhere"
  task could address these but is out of P5e scope.
- **A.1 stream parallelism**: per-call `new_stream` overhead dominates kernel
  overlap benefit. A future task could investigate a stream-pool pattern at
  model construction time if MLX safe wrapper grows the API surface.
- **B.2 grouped matmul**: if further perf is needed, group tokens by expert and
  replace `gather_qmm` with per-expert `quantized_matmul` (would need MLX
  grouped matmul API exposure check).
- **GatedDeltaNet (20.6% of PP=2048)**: second-largest hot path, unchanged by
  P5e. Future linear-attention optimization phase candidate.
- **GatedAttention (6.5%, 10 layers, O(S²))**: smaller share but super-linear
  in context length; matters at long context.
- **Threshold band**: `SORTED_ROUTING_MIN_BS_K=512` was tuned against the
  Qwen3.5-MoE-A3B 128-expert config. Other MoE models (different `E`) may need
  a different floor — currently a single-model constant. If P6+ adds more MoE
  models, the threshold should become a config-driven value.

## Cross-reference: omlx (observation only)

Per memory[project_omlx_perf_baseline] and memory[no_spec_from_competitors]:
omlx serve on the same Qwen3.5-35B-A3B-4bit snapshot achieves PP=2048 prefill
≈ 4214 tok/s (per `reports/p5d-perf-comparison.md`, M1 Pro 32GB), but with
default-on optimizations (body-replacement patches + paged cache, no opt-out)
that are not directly comparable to ironmlx's design. omlx is recorded here
as observation, NOT as an alignment target — ironmlx independently chose the
sorted-routing path based on its own architecture and the MLX fast-path
condition surfaced by T0 profile.

P5e self-improvement (1.92× PP=2048 throughput on the same hardware) is the
acceptance metric; comparison to any competitor is informational only.
