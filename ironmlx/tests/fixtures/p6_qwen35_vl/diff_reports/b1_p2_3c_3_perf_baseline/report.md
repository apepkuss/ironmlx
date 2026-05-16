# B1-p2.3c-3 perf baseline — iron-bench v2 concurrent

**Date:** 2026-05-15
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (head `be93eaa` — post 3c-3 close-out)
**Machine:** MacBook Pro 18,3 (M1 Pro, 32GB RAM, macOS 25.3.0)
**Model:** `mlx-community/Qwen3.5-4B-MLX-4bit` (snapshot `32f3e8e…`)
**Server:** `ironmlx serve --port 8080` (b_max=4 hardcoded at [core/server/mod.rs:54](../../../../src/core/server/mod.rs#L54))
**Bench tool:** `iron-bench` v2 concurrent (this branch)

## Summary

3c-3 continuous batching delivers **linear throughput scaling up to b_max=4**:

| concurrent workers | aggregate tok/s | speedup vs c=1 |
| --- | --- | --- |
| 1 | 12.8 | 1.00× (baseline) |
| 2 | 25.6 | 2.00× |
| 4 | 51.2 | 4.00× |
| 8 | — | rejected (b_max=4 cap; 3d task scope) |

Per-worker ITL stays flat (~67ms) across c ∈ {1, 2, 4} — rolling decode loop fills b_max slots without per-worker degradation. This validates the 3c-3 design goal (mid-batch admit + rolling decode loop replacing the per-batch boundary stall).

## Test matrix

PP = prompt tokens, TG = max generated tokens per request, c = concurrent workers per cell.

| cell | PP | TG | c | n req | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | tok/s | req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 128 | 64 | 1 | 3  | 1980 | 1984 | 67 | 67 | 12.8 | 0.20 |
| B | 128 | 64 | 2 | 6  | 1995 | 2034 | 67 | 67 | 25.6 | 0.40 |
| C | 128 | 64 | 4 | 12 | 2024 | 2024 | 68 | 68 | 51.2 | 0.80 |
| D | 512 | 64 | 4 | 8  | 9161 | 9168 | 79 | 79 | 34.1 | 0.53 |
| E | 128 | 64 | 8 | —  | — | — | — | — | — | rejected at admit |

Notes:
- Cells A/B/C used `--duration 15 --warmup-duration 3`.
- Cell D used `--duration 15 --warmup-duration 3 --prompt-len 512` (same matrix, longer prompt).
- Cell E (c=8) fails fast: `HTTP 400 Bad Request — admit failed: scheduler full: no row available (b_max=4)`. No retry in iron-bench v2 (deliberate — surfacing the 3d gap).

Source data archived under `results/`:
- [`concurrent_sweep.txt`](results/concurrent_sweep.txt) — cells A/B/C (concurrent levels)
- [`c4_pp_sweep.md`](results/c4_pp_sweep.md) — cells C/D (PP levels)
- [`c4_full.json`](results/c4_full.json) — JSON archive of cell C and D for downstream tooling

## Single-request control

Direct `ironmlx generate --prompt "hello" --max-tokens 10`:

- Cold (process start, file-cache hot): **0.97s** wall, ~10ms/token decode after prefill
- Server B=1 via curl (warm): **0.86s** wall for the same 11-token prompt + 10-token generation

The server adds ~−0.1s (within noise) over direct CLI — confirming the per-request HTTP/JSON path does **not** add measurable overhead. The b_max=4 lockstep cost (every decode step computes B=4 forward even with 1 active row) raises ITL from ~10ms (direct CLI pipelined path) to ~67ms (server sync path). This is **expected** and matches the 3c-3 design's known limitations.

## Why server ITL is ~67ms vs direct CLI ~10ms

Two factors compound:

1. **b_max=4 lockstep**: `Scheduler::step_inner` runs a B=4 forward every decode step regardless of how many rows are active. With 1 active row, 75% of GPU work is wasted on pad rows. Lockstep cost is documented in spec §7 of [3c-3 design](../../../../../docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md).
2. **Sync sample path**: `Scheduler::step_inner` calls `sampler.sample` with `.item::<u32>()` on the argmax — a synchronous GPU barrier per active row per step. The direct CLI path uses `next_token_pipelined` ([core/generate.rs:969](../../../../src/core/generate.rs#L969)) with `sample_async_greedy` + `async_eval` to overlap step N+1's CPU graph build with step N's GPU materialisation.

With c=4 fully utilising b_max=4, the lockstep penalty disappears (every row is active) and aggregate throughput scales linearly. ITL stays flat because the GPU is doing the same B=4 work; only request count varies.

## Limitations exposed (out of 3c-3 scope)

The benchmark surfaces three concrete follow-up tasks already in the [3c-3 close-out](../b1_p2_3c_3_closeout/report.md):

1. **3c+ chunked prefill** — Cell D (PP=512 c=4) TTFT p50 = 9.2s. `Scheduler::prefill_admitted` runs `model.batched_prefill` single-shot over the full [B, max_len] padded prompt; longer prompts compound the b_max lockstep on prefill. Chunked prefill (mirroring `GenerationStream`'s [chunk loop](../../../../src/core/generate.rs#L780)) would bound prefill latency by chunk_size and overlap chunks with active decode rows.
2. **3d admission queue** — Cell E (c=8) rejects on `admit_mid` reaching b_max=4. A small bounded FIFO queue ahead of the scheduler would absorb burst load and let 3c-3's rolling decode loop drain it as slots free.
3. **3e per-row sampler tuning / async path** — ITL floor of 67ms is dominated by sync `.item()`. A future scheduler-side pipelined path (mirror `next_token_pipelined`'s async_eval) would cut ITL to GPU compute time alone.

None of these block 3c-3's correctness — all 3 integration scenarios (mid_decode_admit / full_reject / drains_to_empty) and the 12-suite regression sweep PASS at bit-id 1.0000. The numbers above are the **honest 3c-3 baseline** before those follow-ups land.

## Reproducer

```sh
MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve --model "$MODEL" --port 8080 &
sleep 8

# Concurrent sweep (cells A/B/C)
for c in 1 2 4; do
  ./target/release/iron-bench \
    --target "ironmlx=http://127.0.0.1:8080" \
    --model-dir "$MODEL" \
    --concurrent $c --duration 15 --warmup-duration 3 \
    --prompt-len 128 --max-tokens 64
done

# Long-prompt cell (D)
./target/release/iron-bench \
  --target "ironmlx=http://127.0.0.1:8080" \
  --model-dir "$MODEL" \
  --concurrent 4 --duration 15 --warmup-duration 3 \
  --prompt-len 512 --max-tokens 64
```

## Linked artifacts

- [3c-3 close-out report](../b1_p2_3c_3_closeout/report.md)
- [3c-3 design spec](../../../../../docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md)
- [iron-bench v2 design spec](../../../../../docs/superpowers/specs/2026-05-15-iron-bench-v2-concurrent-design.md)
