# MTP Phase 5.1 Server API Validation

- Host: Apple M5 Max / 128GB
- Branch: codex/mtp-phase5.1-server-api-validation
- Base: codex/mtp-phase5-server-api
- Protocol: each model started with `ironmlx serve --b-max 1 --mtp-model-dir <mtp>`
- API checks: `/healthz`, OpenAI non-streaming greedy, OpenAI non-streaming non-greedy, Anthropic non-streaming greedy, OpenAI streaming benchmark via iron-bench, Anthropic streaming SSE smoke

## Summary

| model | draft | initial | OpenAI greedy counters | non-greedy unchanged | Anthropic non-stream counters | Anthropic stream events | Anthropic stream counters | OpenAI stream TG tok/s | final counters |
|---|---:|---|---|---|---|---|---|---:|---|
| qwen35_4b | 1 | enabled=true, 0/0 | 1/7 | yes | 2/14 | 1/1/8/1/1/1 | 1/7 | 113.132 | 4/76 |
| qwen36_27b | 2 | enabled=true, 0/0 | 1/7 | yes | 2/14 | 1/1/8/1/1/1 | 1/7 | 65.017 | 4/34 |
| qwen36_35b_a3b | 2 | enabled=true, 0/0 | 1/7 | yes | 2/14 | 1/1/8/1/1/1 | 1/7 | 131.858 | 4/76 |

## Interpretation

- `/healthz.mtp.enabled` and model-aware default `draft_tokens` are correct for all three models.
- Greedy OpenAI and Anthropic non-streaming requests increment MTP counters, proving both API handlers reach scheduler MTP execution.
- Anthropic `stream: true` returns the expected SSE event sequence and increments MTP counters for all three models.
- OpenAI `temperature=0.7` requests complete successfully without changing MTP counters, proving request-level non-greedy fallback stays on the regular scheduler path while server-level MTP remains enabled.
- The OpenAI streaming benchmark remains a smoke/perf sample (`prompt_len=128`, `max_tokens=32`, `runs=1`, `warmup=1`), not a statistically robust performance baseline.

Anthropic stream event tuple format: `message_start/content_block_start/content_block_delta/content_block_stop/message_delta/message_stop`.
