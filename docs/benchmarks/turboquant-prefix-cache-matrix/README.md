# TurboQuant x Prefix Cache Benchmark Matrix

Use `scripts/benchmark_turboquant_prefix_cache_matrix.py` to compare the four
runtime/cache combinations that matter for the packed Prefix Cache path:

| Variant | TurboQuant | Prefix Cache |
| --- | --- | --- |
| `baseline_dense` | no | no |
| `turboquant_only` | yes | no |
| `prefix_cache_only` | no | yes |
| `turboquant_prefix_cache` | yes | yes |

The runner starts one `ironmlx serve` process per variant, runs `iron-bench`
with `--prefix-cache-probe`, then stops the server before moving to the next
variant. Prefix Cache variants get a fresh cache directory per run so cold/write
and warm/hit behavior can be read from one cell without cross-variant residue.

```bash
python3 scripts/benchmark_turboquant_prefix_cache_matrix.py \
  --model-dir ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<snapshot> \
  --prompt-len 2048,8192 \
  --max-tokens 16 \
  --runs 3 \
  --kv-quant k3v4
```

Useful development smoke:

```bash
python3 scripts/benchmark_turboquant_prefix_cache_matrix.py \
  --prompt-len 128 \
  --max-tokens 2 \
  --runs 2 \
  --dry-run
```

Each run writes to `docs/benchmarks/turboquant-prefix-cache-matrix/<timestamp>/`
by default:

- `metadata.json`: model, prompt lengths, and variant configuration.
- `run_commands.sh`: the exact serve and benchmark commands.
- `summary.json`, `summary.csv`, `summary.md`: cold/warm TTFT, decode TPS,
  actual token counts, cache directory size, and `/healthz` MLX peak memory.
- `<variant>/server.log`, `<variant>/bench-pp*.json`, and `<variant>/healthz-pp*.json`:
  per-variant raw evidence.
