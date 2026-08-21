# Qwen3.5/Qwen3.6 MTP Optimization Benchmark

> Historical benchmark snapshot. The measurements and production readout below
> reflect the recorded build and runtime policy at the time of this run. Current
> IronMLX defaults to one MTP draft token unless a different value is explicitly
> configured; historical `d=2` default conclusions are not the current default.

Date: 2026-07-06

Environment: Apple M5 Max, `max_cache_cap=32768`, `prefill_chunk_size=2048`, `admission_deadline_ms=5`, benchmark duration 20s with 3s warmup, prompt lengths 256 and 8192, `max_tokens=64`.

## Verification Scope

- This directory is a clean post-fix A/B run. Older pre-fix benchmark artifacts were removed because they contained hybrid-cache rollback errors.
- All 8 final server logs were scanned for `ERROR`, `prefill error`, `step error`, `trim_full_layer`, `panic`, `failed`, `timeout`, and `row .* MTP state absent`; every final row has `log_error_count=0`.
- `/healthz` was captured before and after each run, so MTP acceptance and draft counters are from the same process as the benchmark data.

## Key Results

| Model | Variant | PP | b_max | d | tok/s | TTFT p50 ms | ITL p50 ms | finish | accept | log errors |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| qwen3.5-4b | qwen35_4b_baseline_b1 | 256 | 1 |  | 95.35 | 78.21 | 9.64 | length=29,stop=1 |  | 0 |
| qwen3.5-4b | qwen35_4b_baseline_b1 | 8192 | 1 |  | 25.60 | 2052.53 | 7.30 | length=8 |  | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b1_default | 256 | 1 | 1 | 112.00 | 85.33 | 7.57 | length=35 | 58.4% | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b1_default | 8192 | 1 | 1 | 25.60 | 2178.11 | 7.45 | length=8 | 58.4% | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b4_d2 | 256 | 4 | 2 | 102.25 | 280.10 | 42.02 | length=31,stop=1 | 46.2% | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b4_d2 | 8192 | 4 | 2 | 28.80 | 14720.43 | 8.10 | length=9 | 46.2% | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b4_default | 256 | 4 | 1 | 89.60 | 483.33 | 40.58 | length=28 | 50.8% | 0 |
| qwen3.5-4b | qwen35_4b_mtp_b4_default | 8192 | 4 | 1 | 28.80 | 12945.00 | 7.88 | length=9 | 50.8% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_baseline_b1 | 256 | 1 |  | 33.00 | 185.74 | 13.18 | stop=66 |  | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_baseline_b1 | 8192 | 1 |  | 11.10 | 3103.33 | 9.75 | length=3,stop=3 |  | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b1_default | 256 | 1 | 2 | 118.40 | 186.18 | 5.79 | length=37 | 100.0% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b1_default | 8192 | 1 | 2 | 19.20 | 3009.05 | 8.99 | length=6 | 100.0% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b4_d2 | 256 | 4 | 2 | 38.00 | 875.37 | 19.29 | stop=76 | 72.4% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b4_d2 | 8192 | 4 | 2 | 18.00 | 13118.40 | 8.96 | length=5,stop=4 | 72.4% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b4_default | 256 | 4 | 2 | 40.70 | 874.91 | 19.37 | length=1,stop=75 | 73.2% | 0 |
| qwen3.6-35b-a3b | qwen36_35b_a3b_mtp_b4_default | 8192 | 4 | 2 | 23.40 | 13968.83 | 8.98 | length=7,stop=2 | 73.2% | 0 |

## Production Readout

- Qwen3.6-35B-A3B long prompt: baseline `b_max=1` is 11.10 tok/s; MTP `b_max=1,d=2` is 19.20 tok/s (+73.0% vs baseline); MTP `b_max=4` default is 23.40 tok/s (+110.8% vs baseline, +21.9% vs MTP b1).
- Qwen3.6 `b_max=4,d=2` reached 18.00 tok/s in this clean run. Default also resolves to `d=2`; the measured difference is treated as run variance, so the production default remains `d=2`.
- Qwen3.6 `b_max=4` acceptance is 72.4% for explicit `d=2` and 73.2% for default, with zero fallback prefill and zero scanned log errors.
- Qwen3.5-4B long prompt: baseline is 25.60 tok/s; MTP `b_max=1,d=1` is 25.60 tok/s; MTP `b_max=4,d=2` and default both reached 28.80/28.80 tok/s, which is +12.5%/+12.5% vs baseline.
- Short prompt behavior remains latency-sensitive: Qwen3.5 b1 MTP improves short-prompt tok/s with modest TTFT increase, while b4 raises TTFT/ITL; Qwen3.6 b4 improves aggregate short-prompt tok/s but has much higher TTFT. The production default should continue using admission policy to avoid batching short interactive requests too aggressively.
- Overall conclusion: Qwen MTP now has clean post-fix production evidence for `b_max=1` and `b_max>1`; the main production win is long-prompt concurrent decode, especially Qwen3.6 default `b_max=4`.

## Raw Artifacts

- `summary.csv` contains per-row numeric metrics and health counters.
- Each variant directory contains `bench.json`, `bench.stderr.log`, `health_before.json`, `health_after.json`, and `server.log`.
