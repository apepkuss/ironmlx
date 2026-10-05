# Quantized embedding decode without token-count specialization

Experimental, default off (`IRONMLX_EXPERIMENTAL_QEMBEDDING_RUNTIME_COUNT=1`).
An experimental candidate on top of the N1 baseline and the start-up
pre-compilation; not formally accepted. Short prompts only.

## Problem

`ironmlx_qembedding_decode_{4,8}bit_gs64` took `TOKEN_COUNT` as a template
constant, used only in its bound check, so every new prompt length compiled a
new Metal library on first use (about 25–90 ms each in a cold session).

## Change

New kernels `..._rc` with the identical per-element body; the bound
`TOKEN_COUNT * DIM` is replaced by `threads_per_grid.x` (the dispatched grid
is exactly token_count x DIM). No extra input or host work. The switch selects
them; the baseline kernels remain the default.

## Validation

- `runtime_count_kernel_is_bitwise_identical`: 4- and 8-bit, bf16 and f32,
  token counts 1, 2, 15, 26, 46, 129, 1000 — identical bits. All 14
  embedding tests pass.
- `runtime_count_single_token_dispatch_cost` (ignored, timing only; 2000
  single-token dispatch+eval each, alternating): baseline 173.9 / 163.3 µs,
  runtime count 166.7 / 167.3 µs — no measurable difference.

External evidence (root
`/Users/xin/workspace/b1-api-performance/ironmlx-backend/benchmarks/b1-short-four-app-v1/`):

| evidence | path | sha256 |
| --- | --- | --- |
| combined binary (this commit builds it byte-identically) | fix-validation/binaries/ironmlx-combined | 4247071e707977ceafd7d751941dba8f7ff76994206365761e624502f15c1c48 |
| combined + first-use diagnostic | fix-validation/binaries/ironmlx-combined-diag | 107325602241df5d4ed47f2a443b3fe14fe05e0e28e664721a265434015531ef |
| local check (cold) | fix-validation/qembedding-local-v1/validation.json | 39ada8fd7d77b52af56550cb4290363e1e36ff13a3a78441acaadc7b0301e57d |
| combined API summary | fix-validation/combined-api-v1-summary.json | 6ee98cdc5c78f24accd940e053e2df3651a51820c8b308911145556c86bc0a3b |

Local check (diagnostic on, cold): one qembedding compile per process
(about 112 ms first execution) instead of one per new prompt length.

API check, diagnostic off, warm compile cache:

| session | switches | pre-compile function (ms) | start->ready (s) | warmup-1 TTFT | first 33–48-token TTFT | steady TTFT median | Decode median | E2E median | lifecycle peak (GB) | outputs = scored N1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | runtime count | — | 2.55 | 0.208 | 0.119 | 0.128 | 79.3 | 11.14 | 22.39 | yes |
| 1 | off | — | 2.04 | 0.263 | 0.141 | 0.150 | 79.0 | 11.17 | 22.41 | yes |
| 2 | both | 55 | 2.05 | 0.160 | 0.146 | 0.122 | 78.2 | 11.35 | 22.72 | yes |
| 3 | both | 49 | 2.06 | 0.157 | 0.103 | 0.115 | 77.0 | 11.40 | 22.41 | yes |
| 4 | off | — | 2.06 | 0.214 | 0.110 | 0.142 | 76.8 | 11.48 | 22.42 | yes |
| 5 | runtime count | — | 2.06 | 0.215 | 0.152 | 0.138 | 76.5 | 11.54 | 22.74 | yes |

## Reading

- Outputs are byte-identical to the scored N1 outputs in every session.
- The change removes compile work (one library per process instead of one per
  new length) rather than moving it.
- Decode medians fell with session order (79.3 to 76.5); adjacent
  runtime-count/off pairs are 79.3/79.0 and 76.5/76.8. Six sessions cannot
  attribute a Decode difference to either switch; together with the
  single-token microbenchmark, no single-token decode cost was found.
- Lifecycle peaks vary 22.39–22.74 GB across all configurations.

## Remaining before any acceptance

Formal acceptance under the original gates has not been run for this change.
