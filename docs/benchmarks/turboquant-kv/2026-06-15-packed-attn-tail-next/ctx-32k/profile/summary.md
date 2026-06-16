# TurboQuant Packed Attention Tail Next

Date: 2026-06-15

Branch: `codex/turboquant-packed-attn-tail-next`

Base: `bdc938c3b16837139c478502b2f2896dbb72ba63` (`codex/scheduler-autotune-v2`)

## Goal

Continue from the post-qk / weighted-v mainline attribution and test one focused packed-attention
tail candidate before returning to broader decode-layer work.

The selected candidate is:

- `QK_POSITIONS_PER_SIMDGROUP`: `2 -> 4`

The intent is to let each qk SIMD group cover four adjacent sequence positions, amortizing the
query value load across a slightly larger position block while keeping the prior conservative
`QK_SIMDGROUPS_PER_THREADGROUP = 4` setting.

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Core benchmark shape: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Profile shape: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 64 --runs 1 --warmup-runs 0`

## Core Benchmark

Baseline is the current mainline confirmation run from
`docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-qhead-main-confirmation/ctx-32k/core-bench/main-k3v4-3x64.json`.

| variant | valid | decode p50 ms | decode mean ms | generation p50 tps | generation mean tps |
| --- | ---: | ---: | ---: | ---: | ---: |
| main confirmation | 3/3 | 865.687 | 868.605 | 72.775 | 72.536 |
| qkpos4 | 3/3 | 813.897 | 812.814 | 77.405 | 77.529 |

Delta:

| metric | qkpos4 delta |
| --- | ---: |
| decode p50 | -5.98% |
| decode mean | -6.42% |
| generation p50 tps | +6.36% |
| generation mean tps | +6.88% |

## Profile Rollup

The current profile run is `qkpos4-k3v4-profile-1x64.stderr.txt`.

Packed attention stages:

| stage | count | total ms | mean us | p50 us | p95 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| qk | 504 | 269.186 | 534.1 | 527 | 715 |
| softmax | 504 | 90.815 | 180.2 | 181 | 223 |
| weighted_v_chunk | 504 | 276.122 | 547.9 | 524 | 671 |
| weighted_v_reduce | 504 | 93.581 | 185.7 | 188 | 223 |

For comparison, the preceding mainline attribution combined two 1x64 profile runs and reported:

| stage | count | total ms | mean us | p50 us | p95 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| qk | 1008 | 903.267 | 896.1 | 787 | 1729 |
| weighted_v_chunk | 1008 | 754.373 | 748.4 | 692 | 1572 |
| softmax | 1008 | 193.278 | 191.7 | 187 | 253 |
| weighted_v_reduce | 1008 | 187.890 | 186.4 | 184 | 236 |

Decoder-layer profile context from the qkpos4 run:

| event | stage | count | total ms | mean us | p50 us | p95 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| decoder layer | decode_mlp_path | 2016 | 510.313 | 253.1 | 253 | 296 |
| decoder layer | decode_attention_path | 2016 | 501.637 | 248.8 | 261 | 310 |
| decoder layer | decode_layer_output | 2016 | 323.675 | 160.6 | 158 | 196 |
| decoder layer | decode_input_norm | 2016 | 320.674 | 159.1 | 156 | 191 |
| decoder layer | decode_post_attention_norm | 2016 | 319.343 | 158.4 | 152 | 193 |
| decoder layer | decode_attention_residual | 2016 | 313.035 | 155.3 | 149 | 191 |
| gated attention | decode_qkv_proj | 504 | 93.778 | 186.1 | 184 | 216 |
| gated attention | decode_q_split_norm_reshape | 504 | 79.552 | 157.8 | 155 | 187 |
| mrope | decode_query_turbo_rotation | 504 | 81.677 | 162.1 | 160 | 192 |

## Decision

Retain `QK_POSITIONS_PER_SIMDGROUP = 4`.

The core 3x64 benchmark clears the retention gate with a clear decode-time reduction and no mean
regression. Profile attribution also supports the decision: the qk bucket drops materially versus
the prior mainline attribution, which matches the only production-code change in this branch.

The next bottleneck is no longer a single obvious qk spike. The largest remaining buckets are now
split between packed attention (`weighted_v_chunk`, `qk`) and decode layer work (`decode_mlp_path`,
`decode_attention_path`, normalization/residual tail kernels). Further work should start from fresh
attribution rather than retuning qk blindly.

## Artifacts

- `docs/benchmarks/turboquant-kv/2026-06-15-packed-attn-tail-next/ctx-32k/core-bench/qkpos4-k3v4-3x64.json`
- `docs/benchmarks/turboquant-kv/2026-06-15-packed-attn-tail-next/ctx-32k/profile/qkpos4-k3v4-profile-1x64.json`
- `docs/benchmarks/turboquant-kv/2026-06-15-packed-attn-tail-next/ctx-32k/profile/qkpos4-k3v4-profile-1x64.stderr.txt`

## Verification

- `MLX_DIR=$HOME/.local/mlx cargo test -p mlx qk_positions_per_simdgroup_tuning_constant_matches_candidate -- --nocapture`
- `MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel -- --nocapture --test-threads=1`
- `MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_sdpa_decode_parallel_matches_dense_materialized_reference -- --nocapture --test-threads=1`
- `cargo fmt`
- `MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check`
- `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings`
- `MLX_DIR=$HOME/.local/mlx cargo build --release`
- `MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench`
