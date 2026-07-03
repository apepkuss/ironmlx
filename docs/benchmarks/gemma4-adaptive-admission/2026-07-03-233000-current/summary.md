# Gemma4 E4B Adaptive Admission A/B Current Tree

- Result dir: `/Users/xin/workspace/ironmlx-backend-gemma4-adaptive-admission/docs/benchmarks/gemma4-adaptive-admission/2026-07-03-233000-current`
- Model: gemma-4-e4b-it-4bit + gemma-4-E4B-it-qat-assistant-4bit
- Shape: concurrent=4, max_tokens=64, duration=90s, warmup=0, max_cache_cap=65536
- Configs: baseline_b1 and fixed_b4 use dev binary; adaptive_default uses current worktree binary.
- Note: iron-bench concurrent JSON reports TTFT/ITL/throughput, not per-request E2E aggregate.

| Prompt | Config | Runs | Requests median | Tok/s median | TTFT p50 median ms | TTFT p95 median ms | ITL p50 median ms | ITL p95 median ms | Early ITL p50 median ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24576 | baseline_b1 | 3 | 9 | 6.40 | 70046.4 | 70639.8 | 21.34 | 21.79 | 24.19 |
| 24576 | fixed_b4 | 3 | 8 | 5.69 | 68023.6 | 71310.2 | 65.34 | 95.34 | 72.81 |
| 24576 | adaptive_default | 3 | 11 | 7.82 | 43236.6 | 49375.5 | 17.38 | 23.91 | 32.25 |
| 32768 | baseline_b1 | 3 | 7 | 4.98 | 103353.0 | 104407.8 | 20.88 | 24.12 | 30.85 |
| 32768 | fixed_b4 | 3 | 5 | 3.56 | 101028.6 | 102518.8 | 94.09 | 94.09 | 120.70 |
| 32768 | adaptive_default | 3 | 8 | 5.69 | 62305.4 | 72309.5 | 25.00 | 26.91 | 40.20 |

## Prompt 24576

- Adaptive vs baseline: tok/s +22.2%, TTFT p50 -38.3%, ITL p50 -18.6%.
- Fixed b4 vs baseline: tok/s -11.1%, TTFT p50 -2.9%, ITL p50 +206.2%.
- Adaptive vs fixed b4: tok/s +37.5%, TTFT p50 -36.4%, ITL p50 -73.4%.

## Prompt 32768

- Adaptive vs baseline: tok/s +14.3%, TTFT p50 -39.7%, ITL p50 +19.7%.
- Fixed b4 vs baseline: tok/s -28.6%, TTFT p50 -2.2%, ITL p50 +350.6%.
- Adaptive vs fixed b4: tok/s +60.0%, TTFT p50 -38.3%, ITL p50 -73.4%.
