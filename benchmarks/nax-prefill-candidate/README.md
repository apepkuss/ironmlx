# NAX prefill attention candidate (N1) — source baseline

Experimental candidate, default off. Not formally accepted: in the N1 formal
acceptance the Decode 0.99 gate and the short-input gate failed (32K latency,
increment over the previous candidate, memory, outputs and identity passed).
No production default changes.

## What this commit contains

The exact uncommitted N1 source that was frozen for the n1-v1c screen and
used for every N1 measurement, on top of e8349c93:

- `mlx-sys/shim/src/prefill_d256_nax.cc`, `mlx-sys/shim/include/cxx_mlx_shim/prefill_d256_nax.h`,
  hook in `mlx-sys/shim/src/fast.cc`, `mlx-sys/build.rs`: fused D256 NAX
  prefill attention, active only with
  `IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_METALLIB=<metallib>`.
- `ironmlx-lm/src/nn/gated_attention.rs`: diagnostic-only attention-output
  dither (`IRONMLX_DIAGNOSTIC_ATTN_OUTPUT_DITHER_SEED`), compiled into the
  serving binary, default off; used for the NLL noise-floor calibration.
- `ironmlx-runtime/src/bin/prefill-nll.rs` and its `[[bin]]` entry
  (`tools` feature): the teacher-forced NLL quality tool.
- `mlx/tests/prefill_d256_nax.rs`: operator test.

These eight files are byte-identical to the screen archive
`results/n1-screen-frozen/source.tar.gz` (sha256 1eee2b25158087d8990c59c8f3b0db929b9b640901532fb601c36564d4032dce;
tracked diff sha256 0be6f8fb6a300b52…) under
`/Users/xin/workspace/b1-api-performance/ironmlx-backend/benchmarks/b1-memory-ttft-v1/`.

## External frozen dependencies (kept in place, not committed)

| item | path | sha256 |
| --- | --- | --- |
| MLX static library 0.32.2 | /Users/xin/.local/mlx/lib/libmlx.a | 27f1e5d1bb1d5772d79ea527025666d3a1a7d53f9df9756e8ab656b0718f3400 |
| MLX metallib | /Users/xin/.local/mlx/lib/mlx.metallib | adc6967e4a81b1e33777f81f2742db8deef7e817a958ad7637aeefca1cf73a06 |
| NAX safe-math shader | /Users/xin/workspace/b1-api-performance/ironmlx-backend/benchmarks/b1-api-32k-optimization/results/tree-group6-model-parity-v1/candidate.metallib (source archive `source.tar.gz` alongside, sha256 8653b2e5ed70426b36ff15e3b86556002db41f3a99f7188b4d58a5b20ba4a638) | 7d9281498c7fcc4047e47fc347c4ee64f8af62786476d0ebb4290fb1db9567c9 |
| QMM-down shader | /Users/xin/workspace/b1-api-performance/ironmlx-backend/benchmarks/b1-api-32k-optimization/results/qmm-down-fixed-history-screen-v1/qmm-mtile.metallib | 961833aaa3cc02ab8c41d25aad79cc5ab0c6bbe12da2851f5bc21511ede2b633 |
| target model | mlx-community/Qwen3.8-27B-4bit, revision 3e6447f082e89cc7f0bc6e5441afd38dfce760ff, /Users/xin/.ironmlx/models/huggingface/mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff | per-file hashes in b1-memory-ttft-v1/evidence/n1-formal-identity-pre.json (f399fa55647688c1360586b5ef66f5d5c9ae4152e1e34227d64a01be3b6aea81) |
| draft model | z-lab/Qwen3.8-27B-DFlash2, revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4, /Users/xin/.ironmlx/models/huggingface/z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4 | same record |
| frozen N1 serving binary | b1-memory-ttft-v1/results/n1-screen-frozen/ironmlx | 6dc903995c8c74e4aab59af6b20527eace6fb4bb8c1075f16494c57bda910d32 |
| NLL tool binary | b1-memory-ttft-v1/results/nll-noise-v1/prefill-nll | 61532c42c4d1dc9e007a4084d6070d7acdfbcdaf2bf21b6da2a3e03df80e2ae6 |

Build: `source /Users/xin/.local/mlx/mlx-env.sh && cargo build --release --locked -p ironmlx --bin ironmlx --target-dir target/bench-release`
(rustc 1.94.0). Rebuilding this commit in another worktree gives a binary of
the same size whose only differences from 6dc90399 are the Mach-O LC_UUID
(16 bytes) and the ad-hoc code signature (31 bytes); strings are identical
and no worktree path is embedded.

## N1 serving flags

The c2c3c6bq flags (`IRONMLX_EXPERIMENTAL_M5_SHARED_WEIGHT_LAYOUT=1`,
`IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX=1`, `IRONMLX_EXPERIMENTAL_MLX_CACHE_MAX_MIB=4096`,
`IRONMLX_EXPERIMENTAL_PREFILL_CHUNK_CACHE_RESET=1`, QMM shader) plus the NAX
shader, with the M5 DFlash2 lane flags; exact profiles are recorded in
b1-memory-ttft-v1/results/formal-n1-v1/{32k,short}/formal.json (`profiles.n1`).

## Evidence index (b1-memory-ttft-v1/, b1-short-four-app-v1/ under the b1-api-performance checkout)

- Natural-text NLL gate: docs/nll-natural-v1-results.md
- Screen n1-v1c: docs/n1-screen-v1-results.md
- Formal acceptance (not passed): docs/n1-formal-v1-results.md, evidence/formal-n1-v1-statistics.json
- Short four-application comparison: b1-short-four-app-v1/docs/results.md
