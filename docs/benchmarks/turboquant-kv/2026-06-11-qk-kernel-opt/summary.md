# TurboQuant QK Kernel Optimization

- branch: `codex/turboquant-qk-kernel-opt`
- base: `a5d82d2` (`codex/turboquant-packed-attn-profile`)
- model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- prompt: `ctx-32k.txt` (`37383` prompt tokens)
- kv quant: `k3v4`
- max tokens: `32` for end-to-end decode checks, `8` for stage profiling

## Conclusion

This phase keeps a QK decode-kernel optimization for TurboQuant packed attention.

The retained kernel replaces the per-score `HEAD_DIM`-thread threadgroup reduction with a
simdgroup reduction:

- 32 lanes cooperatively scan `HEAD_DIM` with `dim += 32`
- `simd_sum` reduces the per-lane partial dot products
- each threadgroup carries `4` independent simdgroups, so one dispatch group computes four QK scores
- the dense-materialized TurboQuant reference test still passes

Warm steady-state `k3v4` decode improved from `22.31` TPS to `31.40` TPS on the
32k prompt. The profiled QK stage dropped from `139.984` ms to `44.264` ms for the
8-token stage sample.

Cold numbers include Metal template/JIT effects and should not be mixed with warm
steady-state kernel timing. The simdgroup QK cold profile showed one slow QK event per
new decode `seq_len`; the warm rerun removed those long-tail events.

## End-to-End Decode

| run | decode ms | TPS | generated tail | note |
| --- | ---: | ---: | --- | --- |
| baseline cold | 2314.33 | 13.39 | `validate long-context` | first run in this worktree |
| baseline warm | 1389.34 | 22.31 | `validate the KV` | valid baseline for steady state |
| qk-cache cold | 1994.15 | 15.55 | `validate long-context` | rejected |
| qk-simd1 cold | 1483.36 | 20.90 | `validate long-context` | superseded by retained simd4 dispatch |
| qk-simd4 cold | 1494.65 | 20.74 | `validate long-context` | includes JIT long tail |
| qk-simd4 warm | 987.15 | 31.40 | `validate long-context` | retained |

The retained warm run shares all `32/32` generated token ids with the baseline cold run.
Against the baseline warm run, the first `30/32` token ids match; the same final-token
boundary also differs between baseline cold and baseline warm, so it is not treated as a
new quality regression. The dense reference test is the primary correctness check.

## Stage Profile

The profiler forces `eval` after each stage, so absolute profiled runtime is not the
same as default-path TPS. It is used here to locate relative stage cost.

| run | stage | events | total ms | p50 us | p95 us | max us | events >= 10ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `qk` | 56 | 139.984 | 2481 | 2593 | 2791 | 0 |
| baseline | `weighted_v_chunk` | 56 | 153.625 | 2721 | 2880 | 3041 | 0 |
| qk-cache | `qk` | 56 | 336.010 | 2683 | 24762 | 56362 | 7 |
| qk-simd1 cold | `qk` | 56 | 214.237 | 770 | 21017 | 50240 | 7 |
| qk-simd4 cold | `qk` | 56 | 218.804 | 768 | 21625 | 50467 | 7 |
| qk-simd4 warm | `qk` | 56 | 44.264 | 790 | 880 | 942 | 0 |
| qk-simd4 warm | `weighted_v_chunk` | 56 | 152.825 | 2710 | 2815 | 3016 | 0 |

After the retained QK optimization, `weighted_v_chunk` is again the largest remaining
attention stage in the warm profile.

## Rejected Experiments

| experiment | result |
| --- | --- |
| qk-cache packed K words and norm in threadgroup memory | Rejected. QK total rose to `336.010` ms in the 8-token profile and decode fell to `15.55` TPS. The extra barrier/threadgroup memory pressure dominated. |
| one-simdgroup QK dispatch | Useful proof of the `simd_sum` reduction shape, but not retained directly. It still showed cold long-tail events and was superseded by the four-simdgroup-per-threadgroup dispatch. |

## Artifacts

- baseline: `ctx-32k/core-bench/baseline-k3v4.json`, `ctx-32k/core-bench/baseline-k3v4-rerun.json`
- retained run: `ctx-32k/core-bench/qk-simd4-k3v4-rerun.json`
- retained profile: `ctx-32k/profile/qk-simd4-k3v4-8tok-rerun.stderr.txt`
- cold retained artifacts: `ctx-32k/core-bench/qk-simd4-k3v4.json`, `ctx-32k/profile/qk-simd4-k3v4-8tok.stderr.txt`
- rejected artifacts: `ctx-32k/core-bench/qk-cache-k3v4.json`, `ctx-32k/profile/qk-cache-k3v4-8tok.stderr.txt`, `ctx-32k/core-bench/qk-simd-k3v4.json`, `ctx-32k/profile/qk-simd-k3v4-8tok.stderr.txt`
