# Strict Decode Stress Matrix

- Run directory: `docs/benchmarks/bf16-decode-high-concurrency/2026-07-09-223455-strict-decode`
- Fixed prompt: `/tmp/ironmlx-decode-stress-agent-prompt.txt`
- Prompt local length: `11266` tokens from iron-bench output.
- All measured requests in this strict matrix reached `finish_reason=length`; concurrent raw runs report `completion_min/median/max = 512/512/512`.
- Concurrent `tokens_per_sec` uses the benchmark requested duration as denominator; use ITL/E2E/elapsed for wall-time comparisons.

| model | cell | finish | completion min/med/max | TTFT p95 ms | E2E p95 s | decode metric | elapsed s |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B-MLX-4bit | seq_tg512_c1 | length | 512 | 2700.622 | 6.538 | TPOT 7.510 ms | 6.8 |
| Qwen3.5-4B-MLX-4bit | conc_tg512_c8 | {"length": 9} | 512/512/512 | 50582.349 | 54.472 | ITL p50 7.610 ms | 115.1 |
| Qwen3.5-4B-MLX-8bit | seq_tg512_c1 | length | 512 | 2956.787 | 8.805 | TPOT 11.445 ms | 9.0 |
| Qwen3.5-4B-MLX-8bit | conc_tg512_c8 | {"length": 9} | 512/512/512 | 74199.819 | 80.265 | ITL p50 11.666 ms | 165.7 |
| Qwen3.5-4B-MLX-bf16 | seq_tg512_c1 | length | 512 | 2987.499 | 13.031 | TPOT 19.655 ms | 13.3 |
| Qwen3.5-4B-MLX-bf16 | conc_tg512_c8 | {"length": 8} | 512/512/512 | 100305.901 | 110.307 | ITL p50 19.555 ms | 217.9 |

## Interpretation

- In strict 512-token decode, bf16 remains the slowest decode path: sequential TPOT `19.655 ms`, compared with 4bit `7.510 ms` and 8bit `11.445 ms`.
- In `C=8`, bf16 ITL p50 is `19.555 ms`, compared with 4bit `7.610 ms` and 8bit `11.666 ms`.
- The strict prompt removes the earlier early-stop ambiguity. The bf16 decode gap is therefore real and should be profiled in dense Linear decode rather than attributed to prompt/output variance.
