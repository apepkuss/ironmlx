# Scheduler Profile v5

[简体中文](zh-CN/scheduler-profile-v5.md)

Scheduler Profile v5 is explicit offline calibration. It compares candidate scheduler configurations for a complete runtime context; it does not tune the request hot path automatically.

## Usage entry point

For CLI users and maintainers performing calibration. Run `ironmlx scheduler-autotune calibrate --help` for current options, then calibrate with the actual model and runtime settings. The App profile-generation entry uses the same runtime parameters. Calibration requires real models and available GPU capacity; it is not required to start serving.

## Runtime boundaries

Calibration builds a runtime context, compares candidates, checks health, scores results and produces a profile. Loading checks the exact context again.

- Schema version is 5. The store is `~/.ironmlx/scheduler-profiles`, using `index-v5.json` separately from the existing `index.json`.
- Filenames are `{model}--{hardware}--{selection-profile}--{model-path-hash}--{runtime-context-hash}.json`.
- Long prompts account for chat-template tokens and round-trip headroom so prompt, template and output fit within `max_cache_cap`.
- Automatic loading requires matching normalized model path, hardware label, schema and runtime fingerprint. It does not fall back by model name, reuse old-schema defaults or share measurements across contexts.
- Explicit CLI scheduler settings can override profile values after loading.

## Runtime context

Changes to these fingerprint inputs require recalibration:

| Category | Inputs |
| --- | --- |
| Execution | Scheduler execution model |
| Model | Architecture and model-content fingerprint |
| Weights | Quantization mode and weight fingerprint |
| Speculation | Disabled/Qwen MTP/Gemma4 drafter, draft fingerprint, draft tokens |
| KV | Quantization and logical token cap |
| Prefix cache | Enabled, block size, max pages, LRU/SSD budgets |
| Active KV | Enabled and resident token cap |
| Memory | Total/model memory limits |

Model fingerprints use local configuration, weight metadata and sampled contents. Drafts have separate fingerprints to prevent reuse based only on directory names.

## Default candidate matrix

| Parameter | Default candidates |
| --- | --- |
| Ordinary decode / Gemma4 drafter `b_max` | 1, 2, 4 |
| Qwen MTP `b_max` | 1, 2 |
| `prefill_chunk_size` | 1024, 2048 |
| `decode_cadence_mid_chunk_cap` | 128, 256 |
| `admission_deadline_ms` / `admission_queue_max` | 5 / 32 |
| Prompt length | About 1K, 8K and a long context within the logical KV cap |
| Concurrency | 1, 2, 4, 8 |
| Cache state | Cold only if prefix cache is off; cold and warm otherwise |

All candidates use the runtime context's `max_cache_cap`. Runs are grouped by concurrency with alternating candidate order to reduce thermal/order bias. Higher speculative batch widths are excluded from the default Qwen MTP search.

## Measurements and rejection rules

`iron-bench --format autotune-json` reads health before and after a run and reports differences. Reject the entire candidate if:

- benchmark requests do not complete;
- `memory_budget_ok=false` or runtime health is not healthy;
- queue-full or memory-budget-exceeded counters increase;
- Active KV degrades or swap-error counters increase;
- cached tokens appear in a cold run;
- speculative context is enabled but no measurement contains draft tokens;
- the candidate lacks scenarios covered by other candidates.

MTP draft/accept counts, phase timings and cache commit/restore measurements support diagnostics and path-validity checks; they do not directly alter scoring weights.

## Scoring and output

Within each `(prompt_len, max_new_tokens, concurrency, cache_state)` scenario, normalize TTFT p95, ITL p95, E2E p95 and throughput against that scenario's best result, then apply selection-profile weights. Do not merge results from different runtime contexts.

Outputs include `runtime-context.json`, `run-order.json`, per-candidate/concurrency/cache-state JSON and logs, `calibration.json`, `selection.json`, `selection.txt` and `scheduler-profile.json`.
The serve and App entry points pass MTP, KV quantization, prefix cache, Active KV, memory and model context-cap settings to calibration so the calibration context matches serving.
