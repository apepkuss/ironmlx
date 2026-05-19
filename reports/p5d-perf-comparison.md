# P5d MoE Perf Baseline (Qwen3.5-35B-A3B-4bit, M1 Pro 32GB)

Measurement: 3 runs / 1 warmup per cell, serial (single server up at a time per
[feedback_serial_perf_experiments]).

Model: `mlx-community/Qwen3.5-35B-A3B-4bit`  
Snapshot: `~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec`  
Date: 2026-05-19  
ironmlx branch: `ironmlx-p5-moe`  
omlx: `/Users/xin/workspace/iron-rivals/omlx` (editable install via `uv run --with-editable`)

## Note on pp_tps measurement methodology

ironmlx does not return `prompt_tokens` in the response body, so `pp_tps` is computed
from `local_token_count / ttft_s`. omlx returns server-side `prompt_tokens` (which
includes chat-template tokens, adding ~12 tokens per request), so its `pp_tps` in the
raw JSON uses server-reported counts. To ensure apples-to-apples comparison, the
prefill tok/s values below are all computed as `local_prompt_len / ttft_s`.

## Prefill Performance (tok/s, local-token-count-normalized)

| Prompt Length | ironmlx (tok/s) | omlx (tok/s) | delta (ironmlx-omlx)/omlx |
|---|---|---|---|
| PP=128  | 865.4  | 996.8  | -13.2% |
| PP=512  | 986.0  | 2589.2 | **-61.9%** |
| PP=2048 | 995.9  | 4214.2 | **-76.4%** |

## Decode Performance (tok/s, steady-state)

| Prompt Length | ironmlx (tok/s) | omlx (tok/s) | delta |
|---|---|---|---|
| PP=128  | 122.5 | 132.8 | -7.7% |
| PP=512  | 126.7 | 127.7 | -0.8% |
| PP=2048 | 128.0 | 125.7 | +1.9% |

## Time to First Token (TTFT)

| Prompt Length | ironmlx TTFT (ms) | omlx TTFT (ms) | delta |
|---|---|---|---|
| PP=128  | 147.9 | 128.4 | +15.2% |
| PP=512  | 519.3 | 197.7 | +162.7% |
| PP=2048 | 2056.5 | 486.0 | +323.1% |

## Per-call median latencies (from iron-bench JSON stats)

### ironmlx

| PP | pp_tps_median | tg_tps_median | tg_tps_p95 | ttft_ms_median |
|---|---|---|---|---|
| 128  | 865.4  | 122.5 | 127.5 | 147.9 |
| 512  | 986.0  | 126.7 | 127.0 | 519.3 |
| 2048 | 995.9  | 128.0 | 128.6 | 2056.5 |

### omlx

| PP | pp_tps_server | tg_tps_median | tg_tps_p95 | ttft_ms_median |
|---|---|---|---|---|
| 128  | 1090.2 | 132.8 | 133.0 | 128.4 |
| 512  | 2649.8 | 127.7 | 131.4 | 197.7 |
| 2048 | 4238.9 | 125.7 | 128.3 | 486.0 |

Note: omlx `pp_tps_server` uses server-reported token counts (128+12=140, 512+12=524, 2048+12=2060).

## Peak memory

Not exposed via iron-bench output. From ironmlx healthz at load time:
- `kv_cache_soft_limit_bytes`: 285,212,672 (~272 MB active KV budget with `--max-cache-cap 4096`)
- Model size (omlx log): 19.94 GB (4-bit quantized)

## Gate verdict

Per spec §4.3 + plan §2.3: acceptable = ironmlx vs omlx all metrics within ±20%.

| Metric | Delta | Gate |
|---|---|---|
| Prefill PP=128  | -13.2% | PASS (<20%) |
| Prefill PP=512  | -61.9% | **FAIL** (>20%) |
| Prefill PP=2048 | -76.4% | **FAIL** (>20%) |
| Decode PP=128   | -7.7%  | PASS (<20%) |
| Decode PP=512   | -0.8%  | PASS (<20%) |
| Decode PP=2048  | +1.9%  | PASS (<20%) |

**Observation**: ironmlx and the observed competitor (omlx) reach close decode
throughput (all within ±8%); ironmlx prefill at PP≥512 is observed to be 2.6×~4.2×
slower than omlx. omlx's internal prefill implementation is not directly observable
from ironmlx's side — possible explanations include MoE-aware chunking, expert
deduplication across batch tokens, or other optimizations on the competitor side.
ironmlx is an independent implementation and does not aim to mirror any
competitor's prefill design; this is recorded as an observation, not as evidence
that ironmlx must adopt the same strategy. P5e perf phase will independently
analyze ironmlx's prefill from its own architecture and choose the best design.

## Observation: ironmlx prefill gap widens with prompt length

- omlx TTFT at PP=512: 197ms → throughput 2589 tok/s
- ironmlx TTFT at PP=512: 519ms → throughput 986 tok/s (2.6× slower)
- omlx TTFT at PP=2048: 486ms → throughput 4214 tok/s
- ironmlx TTFT at PP=2048: 2057ms → throughput 996 tok/s (4.2× slower)

The ironmlx prefill duration scales roughly linearly with prompt length while the
competitor's scales sub-linearly. This observation suggests ironmlx is computing
something at full prompt scope that the competitor reduces or parallelizes; the
ironmlx-side root cause is what P5e perf phase will investigate (independent
of how any competitor solved it).

Verification of no prefix-cache skew: all three measured runs per cell show stable
TTFT (omlx PP=2048: 486.4/486.0/485.6ms — within 0.2%), confirming no inter-run
KV cache reuse inflating omlx numbers.
