# Qwen3.8-27B DFlash2 performance recipe

This recipe is the qualification record for the frozen P2 DFlash2 execution
path. It separates supported regimes, correctness evidence, performance
evidence, and rejected ideas so later tuning does not repeat failed work.

## Qualified artifacts and runtime

| Item | Qualification value |
| --- | --- |
| Frozen P2 source | `ff2fcea50a49ed5c9213c70ddcdbc1decc605655` |
| MLX fork | `apepkuss/mlx`, commit `73ad5df20cb30be4192e5c4d0ae8130674773427` |
| MLX upstream base | `ml-explore/mlx`, revision `8a81722b1d71cac9b7dde47e56a438c4b529129b` |
| Target | `mlx-community/Qwen3.8-27B-4bit`, snapshot `3e6447f...` |
| Draft | `JonasLoos/Qwen3.8-27B-DFlash2-b32`, snapshot `fc65843...` |
| Draft checkpoint width | 32 |
| Runtime draft quantization | affine 4-bit |
| Execution lane | B1, linear proposal, affine-4 exact verify |
| Sampling | greedy; position-keyed exact sampled |
| Candidate widths | Q8 and Q16 |
| Contexts | 2K, 8K, 32K |
| Output length | 128 tokens |
| Pairing | alternating Q8→Q16 / Q16→Q8, one warmup pair then seven measured pairs |

The benchmark accepts only existing directories below
`~/.ironmlx/models/huggingface`; it has no Hub resolution or download path.
The Q8 and Q16 member of every pair use the same prompt, mode, seed, target,
draft, and output length. Any token mismatch fails the run before statistics
are written.

## Measurement contract

- `runtime_generation`: the synchronized DFlash2 generation interval exposed
  by the runtime, including draft, verify, projection, sampling, host sync, and
  rollback work. It is not a Metal hardware-counter measurement.
- `wall_total`: construction, uncached prefill, and generation observed by the
  caller. The warmup pair removes first-shape compilation from measured pairs;
  no prefix KV is reused between requests.
- CI95: deterministic paired bootstrap of the geometric mean Q16/Q8 TPS ratio.
  Per-cell resampling is paired; aggregate resampling is stratified by mode and
  context so every matrix cell keeps equal weight. Ratios above 1.0 favor Q16.
- Q16 promotion requires exact tokens, at least five pairs per cell, aggregate
  runtime CI95 lower bound above 1.0, aggregate total-wall lower bound at least
  0.98, and no cell runtime lower bound below 0.95.

The raw runner is `dflash2-q8-q16-qualification`; the analyzer is
`scripts/analyze_dflash2_q8_q16_qualification.py`. Raw outputs under `reports/`
are intentionally untracked. The reviewed summary below is tracked.

## Result

Qualification completed on 2026-09-28 on an Apple M5 Max MacBook Pro with
128 GB unified memory and macOS 26.4 (25E246), using the IronMLX-pinned MLX
fork described above. The runtime was installed with `scripts/setup-mlx.sh`.
The 42 measured pairs all produced exact Q8/Q16 token sequences.

| Mode | Context | Pairs | Runtime Q8 TPS | Runtime Q16 TPS | Runtime Q16/Q8 CI95 | Total-wall Q16/Q8 CI95 | Acceptance Q8/Q16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | 2K | 7 | 47.838 | 45.924 | 0.9591 [0.9515, 0.9663] | 0.9801 [0.9664, 0.9919] | 0.892/0.795 |
| greedy | 8K | 7 | 39.079 | 37.892 | 0.9683 [0.9607, 0.9765] | 0.9977 [0.9868, 1.0086] | 0.892/0.795 |
| greedy | 32K | 7 | 23.666 | 20.810 | 0.8437 [0.6490, 1.1222] | 1.0834 [0.9764, 1.2761] | 0.930/0.267 |
| sampled | 2K | 7 | 32.614 | 31.564 | 0.9831 [0.9682, 1.0068] | 0.9864 [0.9783, 0.9934] | 0.658/0.593 |
| sampled | 8K | 7 | 30.819 | 29.938 | 0.9667 [0.9620, 0.9711] | 0.9934 [0.9877, 0.9988] | 0.694/0.625 |
| sampled | 32K | 7 | 17.927 | 16.958 | 0.8551 [0.7190, 1.0183] | 0.9898 [0.9606, 1.0203] | 0.214/0.067 |

The cell-balanced aggregate runtime ratio is **0.9275** (CI95
**0.8794–0.9805**), so Q16 is about 7.25% slower in the synchronized generation
interval. The aggregate total-wall ratio is **1.0045** (CI95
**0.9856–1.0331**), which is statistically neutral under the 0.98 guardrail.
Q16 fails both the aggregate runtime promotion gate and the 32K per-cell
regression guard. Therefore **Q8 remains the default and Q16 remains opt-in**.

The ignored evidence artifacts are
`reports/dflash2/q8-q16-raw-20260928.json`,
`reports/dflash2/q8-q16-ci95-20260928.json`, and
`reports/dflash2/q8-q16-ci95-20260928.md`.

## Supported and deferred regimes

| Regime | Status |
| --- | --- |
| B1/Q8 affine-4 linear | Stable |
| B1/Q16 affine-4 linear with b32 draft | Correctness qualified; CI95 promotion gate failed, opt-in only |
| B>1/Q16 | Limited B2/B4 feasibility gate failed; do not productize |
| Multi-leaf tree/Q16 | Closed by the failed linear B>1/Q16 continuation gate |
| Q32 | Closed; no measured basis for a wider lane |

## B2/B4 Q16 feasibility gate

The bounded P4 continuation gate ran greedy 2K batches with 128 generated
tokens per row. Each cell used five measured Q16 batches bracketed by Q8 runs
before and after the candidate so monotonic thermal drift would not create a
false Q16 regression. Every run entered real tensor-batched windows, and all
Q8/Q16 row token sequences were exact.

| Batch | Q8 bracket TPS | Q16 TPS | Generation Q16/Q8 CI95 | Full-wall Q16/Q8 CI95 |
| ---: | ---: | ---: | ---: | ---: |
| B2 | 21.251 | 19.842 | 0.9338 [0.9239, 0.9412] | 0.9216 [0.9097, 0.9337] |
| B4 | 17.250 | 16.357 | 0.9475 [0.9410, 0.9542] | 0.9480 [0.9412, 0.9548] |

The equal-cell aggregate generation ratio was **0.9406** (CI95
**0.9332–0.9473**) and the full-wall ratio was **0.9347** (CI95
**0.9237–0.9451**). The continuation gate required at least a 1.10 point
estimate with a generation lower bound above 1.0. Q16 missed that threshold
decisively, so the planned sampled and 8K Q16 expansion was stopped. An 8K Q8
baseline was collected before the early stop but is not used to infer an 8K
Q16 ratio.

The ignored gate evidence is
`reports/dflash2/batched-q8-greedy-gate-20260928.json`,
`reports/dflash2/batched-q16-greedy-gate-2k-20260928.json`,
`reports/dflash2/batched-q8-greedy-gate-2k-after-20260928.json`,
`reports/dflash2/batched-q8-q16-gate-20260928.json`, and
`reports/dflash2/batched-q8-q16-gate-20260928.md`.

## Useful design choices

- Verify width is a first-class lane with explicit capability checks; it is
  never inferred from a draft filename.
- Exact sampled verification uses position-keyed draws during cross-width
  qualification so Q8/Q16 output equality is testable.
- Build/schedule/sync/rollback telemetry remains separate, allowing a wider
  lane's lower launch count to be distinguished from acceptance regressions.
- The runtime can fall back to an ordinary Q1 window when the adaptive draft
  policy predicts negative value.

## Failure and rejection archive

- A block-size-8 draft cannot produce a real Q16 proposal. Target-only Q16
  verify tests do not remove that limitation; the b32 draft is required.
- Q16 correctness does not establish a throughput win. It stays opt-in until
  the paired CI95 gate passes.
- B1 results do not qualify B>1, tree/Q16, or Q32 layouts.
- Missing `mlx.metallib` is an environment failure, not a numerical or
  performance result. Qualification must source the configured MLX runtime and
  make its metallib visible to the built executable.
