# MoE Sorted Weighted-Sum Kernel Benchmark

Date: 2026-07-16

Base: `dev@4b96b1b0c74ec6ab511688e83f8e17f01a323fd2`

Candidate: `perf/moe-sorted-weighted-sum`

## Scope

The candidate combines route restoration, router-score multiplication, and
top-k reduction in one Metal kernel. It consumes expert outputs in sorted-route
order and writes `[tokens, hidden]` directly, avoiding the generic
`[tokens, top_k, hidden]` intermediate. Production dispatch covers Qwen SwiGLU,
Gemma4 GeGLU, and GLM cast-output MoE paths with top-k 4, 6, or 8, FP16/BF16
expert output, and at least 1024 tokens.

## Layer microbenchmark

The table compares the candidate production path with the generic local full
pipeline in the same process. Values are p50 latency for one Qwen MoE layer.

| Model | Tokens | Generic (ms) | Kernel (ms) | Change |
|---|---:|---:|---:|---:|
| Qwen3.5-35B-A3B-4bit | 1024 | 3.5908 | 3.0061 | -16.28% |
| Qwen3.5-35B-A3B-4bit | 2048 | 4.7830 | 3.5100 | -26.62% |
| Qwen3.6-35B-A3B-4bit | 1024 | 3.6171 | 3.0105 | -16.77% |
| Qwen3.6-35B-A3B-4bit | 2048 | 4.8193 | 3.5207 | -26.95% |

At 512 tokens the kernel is gated off. Qwen3.5 measured 2.9852 ms versus
2.9929 ms for the generic path; Qwen3.6 measured 2.9968 ms versus 2.9909 ms.
This supports using 1024 tokens as the production crossover.

Parameters: layer 1, hidden size 2048, MoE intermediate size 512, 256 experts,
top-k 8, 4-bit group size 64, five warmups, and 20 measured runs. Raw data is
under `reports/moe-sorted-weighted-sum/20260716/` in the worktree.

## Full-model prefill

Qwen3.5-35B-A3B-4bit was run through `ironmlx-core-bench` with a 4735-token
prompt, 2048-token prefill chunks, one generated token, one warmup, and three
measured runs.

| Metric | dev | Candidate | Change |
|---|---:|---:|---:|
| TTFT mean | 1239.24 ms | 1193.92 ms | -3.66% |
| TTFT p50 | 1239.58 ms | 1193.89 ms | -3.69% |

All measured runs were valid. Baseline and candidate produced token ID `33371`
(`CHECK`) in every measured run.

## Additional architecture validation

The same production kernel was subsequently enabled for Gemma4 GeGLU and GLM
cast-output combines. Both reuse `RoutedExperts`; activation happens before the
kernel, while the GLM path requests that the FP32 weighted sum be cast back to
the expert dtype.

| Model and workload | Generic | Kernel | Change |
|---|---:|---:|---:|
| GLM-4.7 layer 1, 1024 tokens, p50 | 2.4761 ms | 2.3132 ms | -6.58% |
| GLM-4.7 layer 1, 2048 tokens, p50 | 6.4952 ms | 4.1791 ms | -35.66% |
| GLM-4.7 full model, 3850-token TTFT mean | 1480.36 ms | 1420.21 ms | -4.06% |
| Gemma4 full model, 5618-token TTFT mean | 2309.72 ms | 2289.79 ms | -0.86% |
| Gemma4 full model, 11162-token TTFT mean | 5628.53 ms | 5579.52 ms | -0.87% |
| Gemma4 full model, 11162-token TTFT p50 | 5647.17 ms | 5564.41 ms | -1.47% |

The full-model baseline and candidate emitted identical generated token IDs in
every measured GLM and Gemma4 run. A reverse-order seven-run repeat remained
positive for both models, but its absolute latency was affected by machine
temperature; the table therefore reports the more conservative first-round
results rather than the thermally amplified repeat.
