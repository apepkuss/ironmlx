iron-bench v2 (concurrent): 1 target(s), prompt_len=[128, 512], max_tokens=64, concurrent=4, duration=20s, warmup_duration=5s
[ironmlx] PP=128 TG=64 concurrent=4: warmup 5s ...
[ironmlx] PP=128 TG=64 concurrent=4: timed 20s ...
[ironmlx] PP=128 TG=64 concurrent=4: 16 requests completed
[ironmlx] PP=512 TG=64 concurrent=4: warmup 5s ...
[ironmlx] PP=512 TG=64 concurrent=4: timed 20s ...
[ironmlx] PP=512 TG=64 concurrent=4: 8 requests completed
# iron-bench v2 (concurrent) results

- concurrent workers per cell: **4**
- timed duration: **20s**
- warmup duration: **5s**

Targets:
- `ironmlx` → `http://127.0.0.1:8080`

## Per-cell aggregate metrics

| target | PP | TG | N req | p50 TTFT (ms) | p95 TTFT (ms) | p99 TTFT (ms) | p50 ITL (ms) | p95 ITL (ms) | p99 ITL (ms) | tokens/s | req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ironmlx | 128 | 64 | 16 | 2021.4 | 2029.2 | 2029.3 | 68.24 | 68.27 | 68.27 | 51.2 | 0.80 |
| ironmlx | 512 | 64 | 8 | 7358.9 | 7362.8 | 7362.8 | 69.04 | 69.04 | 69.04 | 25.6 | 0.40 |

## Per-worker breakdown

### ironmlx | PP=128 TG=64 | 4 workers

| worker | req count | tokens/s |
| --- | --- | --- |
| 0 | 4 | 12.8 |
| 1 | 4 | 12.8 |
| 2 | 4 | 12.8 |
| 3 | 4 | 12.8 |

### ironmlx | PP=512 TG=64 | 4 workers

| worker | req count | tokens/s |
| --- | --- | --- |
| 0 | 2 | 6.4 |
| 1 | 2 | 6.4 |
| 2 | 2 | 6.4 |
| 3 | 2 | 6.4 |

## Notes

- `ironmlx` PP=128 TG=64: finish_reasons=length=16
- `ironmlx` PP=512 TG=64: finish_reasons=length=8

real 67.08
user 0.70
sys 0.10
