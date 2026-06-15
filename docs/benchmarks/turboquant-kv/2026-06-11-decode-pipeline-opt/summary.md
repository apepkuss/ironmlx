# TurboQuant Decode Pipeline Optimization

Date: 2026-06-11

Branch: `codex/turboquant-decode-pipeline-opt`

Base: `ab5d948 perf: tune turboquant v chunk dim grouping`

## Goal

Continue optimization from the retained `V_CHUNK_DIMS_PER_THREADGROUP = 8` path and inspect the remaining decode pipeline hotspots. The longer profile confirmed that the warm measured path is dominated by `q_rotate`, `weighted_v_chunk`, and `qk`.

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 64 --runs 1 --warmup-runs 1`
- Core long run: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Core stability run: `--max-tokens 32 --runs 5 --warmup-runs 1`

Profile statistics below use only the measured half of the profile output, excluding the warmup half. The warmup half includes one cold long-tail event per new decode `seq_len`, matching the earlier QK-kernel optimization notes.

## Result

Keep the QK norm-hoist optimization.

The retained kernel moves the per-K vector norm multiply out of the inner `dim += 32` loop:

- Before: each lane multiplies `q_rot * k_codebook * k_norm` for every scanned dimension.
- After: each lane accumulates `q_rot * k_codebook`; the simd-reduced dot product is multiplied by `k_norm` once.

This preserves the same packed K layout and kernel dispatch shape. The dense-materialized TurboQuant reference test passed.

## End-to-End Decode

| Run | Max tokens | Runs | TPS mean | Decode mean (ms) | TPS CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline vdim8 | 64 | 3 | 60.498656 | 1041.346417 | 0.118% |
| qknorm | 64 | 3 | 61.442344 | 1025.369611 | 0.514% |
| previous vdim8 | 32 | 5 | 60.597760 | 511.859666 | 2.647% |
| qknorm | 32 | 5 | 60.944250 | 508.860825 | 2.210% |

Compared with the baseline vdim8 3x64 run:

- Generation TPS: `60.498656` -> `61.442344` (+1.56%)
- Decode time: `1041.346417 ms` -> `1025.369611 ms` (-1.53%)

Compared with the previous vdim8 5x32 stability run:

- Generation TPS: `60.597760` -> `60.944250` (+0.57%)
- Decode time: `511.859666 ms` -> `508.860825 ms` (-0.59%)

## Profile

Measured half of 1x64 profile run.

| Run | Stage | Total (ms) | Mean (us) | Median (us) | p95 (us) | p99 (us) | Max (us) | >=10 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline vdim8 | q_rotate | 457.527 | 907.8 | 927 | 965 | 979 | 1127 | 0 |
| baseline vdim8 | weighted_v_chunk | 405.183 | 803.9 | 795 | 850 | 872 | 1791 | 0 |
| baseline vdim8 | qk | 400.663 | 795.0 | 754 | 819 | 843 | 16444 | 1 |
| qknorm | q_rotate | 457.939 | 908.6 | 929 | 967 | 983 | 1018 | 0 |
| qknorm | weighted_v_chunk | 407.166 | 807.9 | 799 | 857 | 888 | 1013 | 0 |
| qknorm | qk | 378.255 | 750.5 | 741 | 807 | 826 | 874 | 0 |

QK stage total improved by 5.59% in the measured profile sample.

## Rejected Experiment

`simd-qrotate-qknorm` replaced the first five Hadamard stages in `q_rotate` with simd shuffle operations, keeping threadgroup memory only for cross-simdgroup stages. It passed the dense reference test, but measured q_rotate time did not improve:

- baseline q_rotate: `457.527 ms`, `907.8 us` mean
- simd q_rotate: `457.842 ms`, `908.4 us` mean

This suggests q_rotate is currently dominated by dispatch/synchronization or surrounding pipeline cost rather than the intra-simd Hadamard barriers. The experiment was reverted.

## Notes

- The generated-token tails still show the same known tie-sensitive variation seen in previous long-context K3V4 runs.
- `q_rotate` remains the largest warm measured attention stage. A meaningful next step likely requires reducing the standalone q_rotate dispatch or fusing it with a broader decode operation, not micro-optimizing the Hadamard body.

## Artifacts

- Core bench files: `ctx-32k/core-bench/`
- Profile files: `ctx-32k/profile/`
