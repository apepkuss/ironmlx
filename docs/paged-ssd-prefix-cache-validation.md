# Paged SSD Prefix Cache Validation

## Scope

This document records the acceptance status for the paged SSD prefix cache work
on branch `codex/paged-ssd-prefix-cache`.

The implemented feature covers:

- Paged SSD prefix cache store schema v2.
- Full-KV page export/restore.
- Linear cache export/restore.
- MLA latent cache export/restore.
- VL image/grid fingerprint separation.
- MTP prefix cache export/restore, including MTP K/V layers and
  `mtp_last_hidden`.
- B>1 batch prefix cache restore through the scheduler.
- Full-KV decode through the paged attention kernel.

The implementation intentionally does not add schema v1 compatibility code.

## Runtime Surface

The feature is enabled by starting the server with:

```text
ironmlx serve \
  --model <checkpoint> \
  --paged-prefix-cache-dir [cache-dir] \
  --paged-prefix-cache-block-size 16 \
  --paged-prefix-cache-max-pages <pages>
```

If `--paged-prefix-cache-dir` is passed without a value, the server uses:

```text
~/.ironmlx/cache/paged_prefix_cache
```

MTP can be enabled at the same time for Qwen dense/MoE models:

```text
ironmlx serve \
  --model <qwen-checkpoint> \
  --mtp-model-dir <qwen-mtp-checkpoint> \
  --mtp-draft-tokens 1 \
  --paged-prefix-cache-dir \
  --b-max 2
```

`--paged-prefix-cache-dir` remains mutually exclusive with TurboQuant KV cache.

## Observability

When a paged SSD prefix cache entry is saved or restored, the scheduler emits
structured log fields in the message body:

- `key`: stable cache key for the entry.
- `row` / `main_row` / `mtp_row`: scheduler row involved in restore/save.
- `tokens`: cached prefix length.
- `restored`: restored token count on hit.
- `load_us` / `save_us`: SSD load/save wall time in microseconds.
- `payload_bytes`: tensor payload bytes represented by the entry.
- `tensors`: number of tensors in the entry.
- `main_layers`, `full_layers`, `linear_layers`, `mla_layers`, `mtp_layers`:
  layer composition.
- `full_pages`: physical Full-KV pages represented by the entry.

Miss attempts are logged at `trace` level with `status` values such as
`MissingEntry`, `MetadataMismatch`, `PayloadReadFailed`, `PayloadInvalid`, or
`EntryInvalid`.

The default server log filter includes `info`, so hit/save timing is visible
without changing `RUST_LOG`. Use `RUST_LOG=ironmlx=trace` when diagnosing miss
reasons.

## Benchmark Probe

Normal `iron-bench` synthetic prompts include a per-run nonce to avoid accidental
prefix-cache hits. To measure paged SSD prefix cache behavior, pass
`--prefix-cache-probe`; this reuses the same synthetic prompt within each cell.

Sequential cold/write plus warm-hit probe:

```text
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --model-dir <checkpoint> \
  --prompt-len 2048 \
  --max-tokens 16 \
  --runs 3 \
  --warmup 0 \
  --prefix-cache-probe \
  --format csv
```

With `--warmup 0`, CSV/JSON mark run 0 as `cold_or_miss_candidate` and later
runs as `warm_hit_candidate`. With `--warmup > 0`, all measured runs are marked
as `warm_hit_candidate` because warmup requests can populate the cache first.

B>1 shared-prefix probe:

```text
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --model-dir <checkpoint> \
  --prompt-len 2048 \
  --max-tokens 16 \
  --concurrent 2 \
  --duration 30 \
  --warmup-duration 5 \
  --prefix-cache-probe \
  --format json
```

In concurrent mode, all workers reuse the same synthetic prompt within each
cell, so TTFT reflects shared-prefix warm-hit behavior and any contention in
the server cache path.

## Acceptance Matrix

| Area | Check | Evidence |
| --- | --- | --- |
| Full-KV | Paged K/V pages save and restore | Unit tests plus Qwen real E2E metadata kind `full_paged` |
| Linear | Conv/recurrent state save and restore | Unit tests plus Qwen real E2E metadata kind `linear` |
| MLA | Latent `c_kv/k_pe` save and restore | Unit tests plus GLM real E2E metadata kind `mla` |
| VL | Image-identical prompts hit; image-changed prompts miss | Qwen VL real E2E with flipped same-shape image |
| MTP | Main cache, MTP cache, and `mtp_last_hidden` save and restore | Qwen VL+MTP real E2E cache metadata and hit logs |
| B>1 | Concurrent exact-hit requests restore prefixes per row | Matrix E2E and VL+MTP E2E with `--b-max 2` |
| Kernel | Full-KV decode uses paged attention kernel | `mlx/tests/paged_attention.rs` B=1 and B=2 ragged reference tests |

## Real Checkpoint Validation

The following commands were run with local snapshots under
`/Users/xin/.ironmlx/models` and `MLX_DIR=$HOME/.local/mlx` on 2026-06-19.

```text
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  -- --ignored --test-threads=1 --nocapture
```

Result: 7 passed, 0 failed.

Covered real checkpoint paths:

- Qwen3.5-4B text/linear: B>1 exact-hit restore; metadata includes
  `full_paged` and `linear`.
- GLM-4.7-Flash MLA: B>1 exact-hit restore; metadata includes `mla`.
- Qwen3.5-4B VL without MTP: exact-hit restore and flipped-image miss;
  metadata includes a non-null `fingerprint_hash`.
- Qwen3.5-4B text restart persistence: exact-hit restore after restarting
  `ironmlx serve` with the same SSD prefix cache directory.
- MiniCPM-V-4.6 VL: exact-hit restore and same-shape flipped-image miss.
- Gemma4 VL: exact-hit restore and same-shape flipped-image miss.
- Gemma4 VL split-prefill regression: paged KV and dense KV produce the same
  next-token argmax (`This`) for the same real image prompt.

```text
cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
  -- --ignored --test-threads=1 --nocapture
```

Result: 1 passed, 0 failed.

Covered real checkpoint path:

- Qwen3.5-4B VL + Qwen3.5-4B MTP: warm save, B>1 exact-hit restore,
  MTP prefix hit, MTP batch prefill, flipped-image miss, and cache metadata
  containing `main_layers`, `mtp_layers`, and `mtp_last_hidden`.

```text
cargo test -p mlx --test paged_attention
```

Result: 2 passed, 0 failed.

Covered kernel paths:

- B=1 paged decode vs reference.
- B=2 ragged paged decode vs reference.

## Long Text Route-Fix Performance Regression

The following real-checkpoint performance checks were run on 2026-06-20 after
commit `4943594 fix: route paged prefix long prompts through scheduler`.

Checkpoint:

```text
/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
```

Common server parameters:

```text
MLX_DIR=$HOME/.local/mlx target/release/ironmlx serve \
  --model <checkpoint> \
  --host 127.0.0.1 \
  --port 18080 \
  --b-max 4 \
  --max-cache-cap 16384 \
  --prefill-chunk-size 2048
```

For cache-on runs, the server also used:

```text
--paged-prefix-cache-dir /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-qwen35-ab-routefix-20260620-1421 \
--paged-prefix-cache-block-size 16 \
--paged-prefix-cache-max-pages 8192
```

Sequential A/B command:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18080 \
  --model-dir <checkpoint> \
  --prompt-len 2048,4096,8192 \
  --max-tokens 16 \
  --runs 3 \
  --warmup 0 \
  --prefix-cache-probe \
  --format csv \
  --timeout 900
```

Sequential TTFT results:

| PP | Cache | Cold TTFT ms | Warm TTFT ms, avg runs 2-3 | Warm TTFT change vs no-cache |
| --- | --- | ---: | ---: | ---: |
| 2048 | off | 492.437 | 455.537 | baseline |
| 2048 | on | 657.540 | 121.645 | 3.7x lower TTFT |
| 4096 | off | 897.062 | 896.549 | baseline |
| 4096 | on | 1327.274 | 247.916 | 3.6x lower TTFT |
| 8192 | off | 1836.521 | 1896.655 | baseline |
| 8192 | on | 3068.425 | 486.633 | 3.9x lower TTFT |

Interpretation:

- Warm repeated-prefix TTFT improves substantially once long text-only requests
  route through `SchedulerActor` and can restore paged SSD prefix cache entries.
- Cold cache-on requests are slower than no-cache because they include miss
  probing, SchedulerActor routing, and SSD entry writes.
- End-to-end generation throughput regresses on the cache-on path for
  `max_tokens=16`. For example, PP=8192 warm e2e was about 5.94s cache-on vs
  about 2.00s no-cache. The logs show each warm-hit request still writes a
  prompt-plus-one-token entry (`tokens=8204`) with `save_us` roughly
  0.38-0.46s, and paged-prefix decode ITL is also much higher. This is a
  follow-up performance issue, not a correctness failure.

Observed cache-on hit/save evidence:

| PP | Restored tokens | Hit load_us | Payload bytes | Full pages |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 2059 | 256-283 | 119144448 | 1032 |
| 4096 | 4107 | 297-315 | 186253312 | 2056 |
| 8192 | 8203 | 389-442 | 320471040 | 4104 |

Cache directory size after the sequential cache-on run:

```text
1.2G  /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-qwen35-ab-routefix-20260620-1421
```

Concurrent large-prefix stress, cache-on:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18080 \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --concurrent 4 \
  --warmup-duration 30 \
  --duration 300 \
  --prefix-cache-probe \
  --format json \
  --timeout 900
```

Result: 84 requests completed in 300s, `finish_reason_summary=length=84`,
`cached_tokens_warning=false`, worker distribution `[21,21,21,21]`,
`req_per_sec=0.28`, `tokens_per_sec=4.48`, TTFT p50/p95/p99 =
1119.625/1263.406/1946.906 ms, ITL p50/p95 =
876.424/925.528 ms.

Post-stress `/healthz` reported `status=healthy`, `b_active=0`, `b_queued=0`,
`admission_queue_full_count=0`, `memory_budget_exceeded_count=0`, and
`kv_cache_active_bytes=0`. Server logs during the run continued to show
PP=8192 cache hits for `restored=8203` with load times around 0.38-0.46ms.

Concurrent large-prefix no-cache comparison:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18080 \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --concurrent 4 \
  --warmup-duration 20 \
  --duration 120 \
  --prefix-cache-probe \
  --format json \
  --timeout 900
```

Result: 53 requests completed in 120s, `finish_reason_summary=length=53`,
`cached_tokens_warning=false`, worker distribution `[13,14,13,13]`,
`req_per_sec=0.4417`, `tokens_per_sec=7.0667`, TTFT p50/p95/p99 =
9148.432/10662.159/10712.936 ms, ITL p50/p95 = 7.467/7.738 ms.

The comparison shows the intended trade-off after the route fix:

- Cache-on is much better for first-token latency on repeated long prefixes.
- No-cache remains better for decode throughput in the measured text-only
  `TG=16` workload.
- Next performance work should focus on reducing paged-prefix warm-hit decode
  cost and avoiding or deferring redundant prompt-plus-one SSD writes.

## Warm-Hit Decode/Save Follow-Up

The following follow-up checks were run on 2026-06-20 after keeping B>1 ragged
decode on the paged-attention path and skipping already-present SSD entries
before exporting prefix payloads.

Cache-on server parameters matched the long text route-fix run, with the cache
directory changed to:

```text
/Users/xin/.ironmlx/cache/paged_prefix_cache/bench-qwen35-postfix-20260620
```

Sequential warm-hit probe:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18080 \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --runs 3 \
  --warmup 1 \
  --prefix-cache-probe \
  --format markdown \
  --timeout 300
```

Result: TTFT median/p95 = 435.8/441.2 ms, decode TG = 3.0 tok/s, E2E =
5.853s, TPOT = 361.13 ms/tok. This confirms B=1 warm-hit TTFT remains good,
but B=1 decode throughput is still limited by the current paged-prefix decode
path.

Concurrent B=4 warm-hit stress:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18080 \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --concurrent 4 \
  --warmup-duration 20 \
  --duration 90 \
  --prefix-cache-probe \
  --format markdown \
  --timeout 300
```

Result: 44 requests completed, `req_per_sec=0.49`, `tokens_per_sec=7.8`,
TTFT p50/p95/p99 = 1310.4/1351.4/3065.1 ms, ITL p50/p95/p99 =
485.69/513.62/515.68 ms, per-worker completion count `[11,11,11,11]`.

Compared with the earlier cache-on B=4 stress result (`tokens_per_sec=4.48`,
ITL p50/p95 = 876.424/925.528 ms), the follow-up reduces median ITL by about
45% and raises aggregate token throughput by about 74% for this workload.

Server logs showed only the initial cold saves for `tokens=8203` and
`tokens=8204`. Subsequent warm-hit requests restored `tokens=8203` with
`load_us` roughly 350-425us and did not emit repeated save logs, confirming the
redundant SSD rewrite path was removed from steady state.

Additional scheduler/MTP real-checkpoint checks:

```text
cargo run --release --bin ironmlx-core-bench -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282 \
  --prompt-file docs/benchmarks/mtp-phase4-policy/2026-06-07-160427/fixed_prompt.txt \
  --mode scheduler-text \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-27B-MTP-4bit/snapshots/83795d546e9d328160e593fb0bf10b2bf2fe637e \
  --mtp-draft-tokens 1 \
  --max-tokens 2 \
  --runs 1 \
  --warmup-runs 0 \
  --b-max 2 \
  --out /tmp/ironmlx-qwen36-27b-mtp-scheduler.json
```

Result: exit 0. The output record was valid, generated 2 tokens, and reported
`mtp_stats.windows=1`, `drafted_tokens=1`, `accepted_draft_tokens=1`,
`mtp_cache_reuse_count=1`, and `mtp_cache_reused_tokens=1`.

```text
cargo run --release --bin ironmlx-core-bench -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --prompt-file docs/benchmarks/mtp-phase4-policy/2026-06-07-160427/fixed_prompt.txt \
  --mode scheduler-text \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit/snapshots/0295b81421bf4d0fccca9a7c0fcfb1418dda3516 \
  --mtp-draft-tokens 1 \
  --max-tokens 2 \
  --runs 1 \
  --warmup-runs 0 \
  --b-max 2 \
  --out /tmp/ironmlx-qwen36-35b-a3b-mtp-scheduler.json
```

Result: exit 0. The output record was valid, generated 2 tokens, and reported
`mtp_stats.windows=1`, `drafted_tokens=1`, `accepted_draft_tokens=1`,
`mtp_cache_reuse_count=1`, and `mtp_cache_reused_tokens=1`.

## Latest Performance Acceptance

The following performance and stress checks were run on 2026-06-21 after commit
`f0ad256 fix: batch VL paged prefix cold misses`.

Artifacts:

```text
/tmp/ironmlx-paged-prefix-perf-20260621-230540
```

Cache directory:

```text
/Users/xin/.ironmlx/cache/paged_prefix_cache/bench-perf-20260621-230540
```

Text checkpoint:

```text
/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
```

Common text server parameters:

```text
MLX_DIR=$HOME/.local/mlx target/release/ironmlx serve \
  --model <checkpoint> \
  --host 127.0.0.1 \
  --b-max 4 \
  --admission-deadline-ms 200 \
  --prefill-chunk-size 2048 \
  --max-cache-cap 16384
```

For cache-on runs, the server also used:

```text
--paged-prefix-cache-dir /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-perf-20260621-230540/text \
--paged-prefix-cache-block-size 16 \
--paged-prefix-cache-max-pages 16384
```

Sequential probe command:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:<port> \
  --model-dir <checkpoint> \
  --prompt-len 2048,8192 \
  --max-tokens 16 \
  --runs 3 \
  --warmup 0 \
  --prefix-cache-probe \
  --format csv \
  --timeout 900
```

Sequential results:

| PP | Cache | Cold TTFT ms | Warm TTFT ms, avg runs 2-3 | Warm E2E s, avg runs 2-3 | Warm decode tok/s, avg runs 2-3 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2048 | off | 487.744 | 453.711 | 0.553 | 161.1 |
| 2048 | on | 795.095 | 226.488 | 0.349 | 130.9 |
| 8192 | off | 1839.003 | 1838.478 | 1.943 | 152.8 |
| 8192 | on | 2425.943 | 236.804 | 0.375 | 116.1 |

Interpretation:

- Warm repeated-prefix TTFT improves by about 2.0x at PP=2048 and 7.8x at
  PP=8192 in this run.
- Warm end-to-end latency also improves for `max_tokens=16` despite lower
  per-token decode throughput, because the restored prefix removes most of the
  long prefill cost.
- Cold cache-on requests remain slower than no-cache requests because they
  include miss probing and SSD cache writes.

B=4 concurrent PP=8192 probe command:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:<port> \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --concurrent 4 \
  --warmup-duration 10 \
  --duration 60 \
  --prefix-cache-probe \
  --format json \
  --timeout 900
```

B=4 concurrent comparison:

| Cache | Duration s | Requests | Req/s | Tokens/s | TTFT p50/p95/p99 ms | ITL p50/p95/p99 ms | Worker counts |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| off | 60 | 30 | 0.500 | 8.000 | 8945.7 / 9404.4 / 9498.6 | 7.09 / 7.64 / 7.87 | `[7,8,8,7]` |
| on | 60 | 412 | 6.867 | 109.867 | 243.1 / 276.3 / 282.5 | 15.69 / 20.58 / 21.19 | `[103,103,103,103]` |

The cache-on path improves B=4 repeated-prefix throughput by about 13.7x in
this PP=8192 workload. The no-cache path has lower ITL once decode starts, but
spends most wall time in repeated prefill.

B=4 cache-on 180s stress:

```text
target/release/iron-bench \
  --target ironmlx=http://127.0.0.1:18085 \
  --model-dir <checkpoint> \
  --prompt-len 8192 \
  --max-tokens 16 \
  --concurrent 4 \
  --warmup-duration 10 \
  --duration 180 \
  --prefix-cache-probe \
  --format json \
  --timeout 900
```

Result: 1212 requests completed in 180s, `finish_reason_summary=length=1212`,
`cached_tokens_warning=false`, worker distribution `[303,303,303,303]`,
`req_per_sec=6.733`, `tokens_per_sec=107.733`, TTFT p50/p95/p99 =
247.818/280.564/287.297 ms, ITL p50/p95/p99 =
15.765/20.870/21.487 ms.

Post-stress `/healthz` reported `status=healthy`, `b_active=0`, `b_queued=0`,
`admission_queue_full_count=0`, `memory_budget_exceeded_count=0`, and
`kv_cache_active_bytes=0`. The stress server log contained only the existing
scheduler profile coverage warning; no runtime errors, queue-full events, or
memory-budget failures were observed. Hit logs continued to show PP=8192
restores with `tokens=8203`, `restored=8203`, and row 0-3 hits per batch.

Cache directory size after the run:

```text
918M  /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-perf-20260621-230540
839M  /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-perf-20260621-230540/text
 79M  /Users/xin/.ironmlx/cache/paged_prefix_cache/bench-perf-20260621-230540/vl
```

MiniCPM-V checkpoint:

```text
/Users/xin/.ironmlx/models/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19
```

VL B=2 cold/write validation used `--admission-deadline-ms 1000` to force the
two image requests to coalesce:

- Cold red/blue B=2 request returned `Red` and `Blue`.
- Logs showed `paged SSD prefix cache saved batched VL prefix` for row 0 and
  row 1, then `paged SSD prefix cache saved batched VL prompt` for row 0 and
  row 1.

VL B=2 warm-hit latency validation reused the same SSD cache directory with
`--admission-deadline-ms 200`:

| Round | Requests | Elapsed ms | Outputs | Hit evidence |
| --- | ---: | ---: | --- | --- |
| 0 | 2 | 253.9 | `Red`, `Red` | `batch_restore_install rows=2`; row 0/1 `restored=88` |
| 1 | 2 | 242.8 | `Red`, `Red` | `batch_restore_install rows=2`; row 0/1 `restored=88` |

The warm-hit logs reported `load_us=239-261` for the loaded row and `load_us=0`
for the shared row, confirming B=2 shared-prefix restore for the real VL
checkpoint.

## Final Coverage Audit

This table summarizes the final functional coverage after the 2026-06-21
performance and VL validation pass.

| Surface | B=1 cold/miss | B=1 exact/warm | B>1 cold/miss | B>1 exact/warm | Evidence |
| --- | --- | --- | --- | --- | --- |
| Full-KV text | Covered | Covered | Covered | Covered | Scheduler unit tests; Qwen3.5 real E2E; 2026-06-21 PP=2048/8192 text benchmark |
| Linear cache | Covered | Covered | Covered | Covered | Linear export/restore unit tests; Qwen3.5 text/linear real E2E; MiniCPM5 B=2 scheduler smoke |
| MLA cache | Covered | Covered | Covered | Covered | MLA export/restore unit tests; GLM-4.7-Flash B>1 real E2E metadata kind `mla`; GLM B=2 scheduler smoke |
| VL without MTP | Covered | Covered | Covered | Covered | VL fingerprint unit tests; Qwen/Gemma/MiniCPM-V real E2E; 2026-06-21 MiniCPM-V B=2 cold batched save and warm exact-hit restore |
| VL with MTP | Covered | Covered | Covered | Covered | `vl_mtp_paged_prefix_e2e`; Qwen3.5 VL+MTP B>1 exact-hit restore and MTP metadata |
| MTP text | Covered | Covered | Covered | Covered | MTP scheduler/unit coverage; Qwen3.5 and Qwen3.6 MTP real scheduler checks with `b_max=2` |
| Restart persistence | N/A | Covered | N/A | Covered | Qwen3.5 text restart persistence real E2E with same SSD prefix cache directory |
| VL fingerprint miss | Covered | N/A | Covered | N/A | Qwen/MiniCPM-V/Gemma4 flipped same-shape image miss checks |
| Paged attention kernel | Covered | Covered | Covered | Covered | `cargo test -p mlx --test paged_attention`, including B=1 and B=2 ragged reference tests |

Acceptance status:

- Complete for the requested paged SSD prefix cache scope: Full-KV, Linear,
  MLA, VL, MTP, B=1, B>1, SSD exact-hit restore, miss/cold save, and paged
  attention kernel coverage.
- Cold cache-on latency is expected to be worse than no-cache because it pays
  miss probing plus SSD write cost.
- The remaining scheduler profile warning in benchmark logs is unrelated to
  paged SSD prefix cache correctness; it indicates the local autotune profile
  lacks long-prompt/concurrent calibration coverage.

## Coverage Notes

Gemma4 requires fp32 KV cache tensors. The implementation now exposes
`Model::cache_dtype()` and overrides it for Gemma4, so scheduler/generation/MTP
allocation no longer hard-code bf16 for model-owned K/V storage. The default
remains bf16 for the existing Qwen, GLM, MiniCPM, and Llama-style paths.

## Verification Commands

The Rust-required verification set was run after the feature and E2E additions:

```text
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
git diff --check
```

Result: all commands exited successfully. MLX C++ header warnings from
`mlx-sys` were present but did not fail the Rust checks.

The 2026-06-21 documentation and final coverage update changed only this
Markdown file. It was verified with:

```text
git diff --check
```
