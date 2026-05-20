# `ironmlx serve` CLI defaults

## `--b-max` (default `1` since P5f)

`ironmlx serve` defaults to `--b-max 1` (single-request optimized). Pass
`--b-max N > 1` to enable concurrent multi-request batching.

The single-request default delivers up to **2.44× prefill speedup** on
short prompts (PP=128: 390 → 951 tok/s), **3.21× prefill speedup** at
PP=512 (491 → 1577 tok/s), and **1.58× decode TG** (79 → 124 tok/s) vs.
the prior `--b-max 4` default. The speedup comes from eliminating
`[B=4, T_max]`-padded MoE compute when only one slot is in use — the
common case for single-user chat / agent serve workloads.

Multi-request throughput is unaffected when batching is explicitly
enabled. Scheduler / KVCache / forward-path semantics are unchanged.

### Boot log

At startup `ironmlx serve` emits an INFO log surfacing the active value:

```
INFO ironmlx::cli::serve: ironmlx serve: b_max=1 (single-request
optimized by default; pass --b-max N > 1 to enable concurrent
multi-request batching)
```
