# Gemma4 Drafter Active KV Regression

Opt-in heavy regression runner for Gemma4 target models with Gemma4 assistant drafter, paged prefix cache, and Active KV offload.

This is intentionally not part of default CI because it needs local Gemma4 checkpoints and enough Apple Silicon memory/GPU headroom.

## Local Model Defaults

The runner auto-detects the latest snapshot under these Hugging Face cache roots:

- `~/.ironmlx/models/models--mlx-community--gemma-4-E4B-it-qat-4bit/snapshots/`
- `~/.ironmlx/models/models--mlx-community--gemma-4-E4B-it-4bit/snapshots/`
- `~/.ironmlx/models/models--mlx-community--gemma-4-E4B-it-qat-assistant-4bit/snapshots/`
- `~/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/`
- `~/.ironmlx/models/models--mlx-community--gemma-4-12B-it-assistant-4bit/snapshots/`

Override paths with:

```bash
export IRONMLX_GEMMA4_E4B_MODEL_DIR=/path/to/e4b
export IRONMLX_GEMMA4_E4B_DRAFTER_DIR=/path/to/e4b-assistant
export IRONMLX_GEMMA4_12B_MODEL_DIR=/path/to/12b
export IRONMLX_GEMMA4_12B_DRAFTER_DIR=/path/to/12b-assistant
```

## Dry Run

```bash
python3 scripts/gemma4_drafter_active_kv_regression.py \
  --dry-run \
  --variant 12b_b2 \
  --out-root docs/benchmarks/gemma4-drafter-active-kv-regression/dry-run
```

Dry run writes `run_commands.sh`, `metadata.json`, and empty planned summaries without checking local model files.

## Real Run

```bash
python3 scripts/gemma4_drafter_active_kv_regression.py \
  --build \
  --variant 12b_b2 \
  --out-root docs/benchmarks/gemma4-drafter-active-kv-regression/manual-12b
```

Useful shorter smoke:

```bash
python3 scripts/gemma4_drafter_active_kv_regression.py \
  --build \
  --variant e4b_b2 \
  --prompt-lens 2048 \
  --duration 10 \
  --warmup-duration 0 \
  --out-root docs/benchmarks/gemma4-drafter-active-kv-regression/smoke-e4b
```

## Assertions

Each benchmark cell fails if any of these occur:

- model load fails;
- `memory.kv_cache_budget_policy` is not `active_kv_offload`;
- logical cap is not `min(262144, model.max_position_embeddings)`;
- resident cap is not lower than logical cap;
- `scheduler.memory_budget_exceeded_count` increases;
- Active KV reports degraded state or swap errors;
- MTP is disabled or `draft_tokens` is not `2`;
- MTP counters do not increase after greedy traffic;
- `iron-bench` returns zero completed requests.

## Artifacts

Each run writes:

- `run_commands.sh`: replayable serve/load/bench commands;
- `metadata.json`: matrix metadata;
- `<variant>/server.log`: server stdout/stderr;
- `<variant>/load-response.json`: App daemon load response;
- `<variant>/healthz-before.json` and `<variant>/healthz-pp*.json`: health snapshots;
- `<variant>/bench-pp*.json`: raw `iron-bench` JSON;
- `summary.json`, `summary.csv`, `summary.md`: compact result summaries.
