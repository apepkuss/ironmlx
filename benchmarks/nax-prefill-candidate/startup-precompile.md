# M5 affine4 start-up pre-compilation

Experimental, default off (`IRONMLX_EXPERIMENTAL_M5_PRECOMPILE=1`). An
experimental candidate on top of the N1 baseline; not formally accepted.
Validated on short prompts only (no 32K run, no rival run).

## Problem

MLX compiles a custom kernel's Metal library and pipeline inside its first
GPU evaluation in a process (`CustomKernel::eval_gpu` -> `Device::get_library`
/ `get_kernel`). The M5 affine4 projection kernels are keyed by
(N, K, tmr, edge); the first prompt of 33–48 tokens in a fresh server needed
five new edge variants. A default-off diagnostic (first execution minus second
execution) estimated about 414 ms for them in a cold session and about
5.7 ms in the following session.

## Change

At the end of `Qwen35Model::from_loader_dflash2`, before the server listens,
each projection that can take the M5 route is run once through its normal
`forward_on` on a throwaway constant activation inside the M5 route scope;
the outputs are evaluated and dropped and the MLX cache is cleared.
`kernel_variant(m)` is shared with the dispatch. Reachable keys only:
decoder projections 1..=128 rows -> (tmr1, edge0), (tmr2, edge0),
(tmr2, edge1) via 1, 17 and 33 rows; lm_head at most 16 rows -> (tmr1, edge0).
For the Qwen3.8-27B lane: 16 affine4 kernels and 3 xsum variants. No request
cache is created; no extra weight copy is kept (a shared store switches to the
tiled layout as the first M5 request would). Unit test:
`precompile_rows_cover_every_reachable_variant`.

## Validation (external evidence, kept in place)

Root: `/Users/xin/workspace/b1-api-performance/ironmlx-backend/benchmarks/b1-short-four-app-v1/`.

| evidence | path | sha256 |
| --- | --- | --- |
| validated binary (this change before a line-wrap-only `cargo fmt`; this commit builds 8b04c1971c7da98b19fda55aed2154bf0e5f79e3f7c58317961c0db4abd6de32) | fix-validation/binaries/ironmlx-precompile | 0b7713f1e32f3ee37c779f124657c0e992a9e232dc299ca906252620a319d785 |
| same change + first-use diagnostic | fix-validation/binaries/ironmlx-precompile-diag | 7da6ed3f564967ff12449df43631930d000ed85269b04d6eb2df30f95a023aa0 |
| local check (cold) | fix-validation/precompile-local-v2/validation.json | 5638c179d120d20ffaadcab25a4fc74d6b9227f2329368ef023e7c37a2dd2333 |
| API check summary | fix-validation/precompile-api-v1-summary.json | e7af027eb0ef48142d5422f3e84cccfa875e766a950710c423157ea2c2c10b89 |
| runner / summary scripts | scripts/validate_fix.py, scripts/summarize_validation.py | 25cb2da4…, 7c832b54… |
| failed first attempt (no server started) | fix-validation/precompile-local-v1/ | kept |

Local check, diagnostic on, cold cache: all 16 affine4 kernels and the 3 xsum
variants were compiled before the "pre-compilation finished" line; no affine4
kernel was compiled during requests after ready. The pre-compilation function
took 1501 ms; startup to ready was 4.54 s. Sessions without the switch took
2.02–2.06 s to ready, but those were warm and without the diagnostic, so the
cold increase of about 2.5 s is not a like-for-like measurement and the part
beyond the function's 1.5 s is not attributed. knowledge-1 TTFT 0.135 s and
warmup-1 0.735 s (diagnostic on), versus about 0.55 s and 1.3–1.4 s in cold
sessions without the fix.

API check, diagnostic off, all sessions with a warm compile cache:

| session | switch | pre-compile function (ms) | start->ready (s) | warmup-1 TTFT | first 33–48-token TTFT | steady TTFT median | Decode median | E2E median | lifecycle peak (GB) | outputs = scored N1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | on | 42 | 3.03 | 0.158 | 0.122 | 0.120 | 86.7 | 10.64 | 22.65 | yes |
| 1 | off | — | 2.02 | 0.211 | 0.110 | 0.125 | 83.0 | 10.65 | 22.66 | yes |
| 2 | off | — | 2.03 | 0.208 | 0.118 | 0.138 | 81.1 | 10.88 | 22.66 | yes |
| 3 | on | 50 | 2.04 | 0.158 | 0.098 | 0.106 | 79.3 | 11.11 | 22.65 | yes |

Readiness is polled every 0.5 s, so start->ready has about 0.5 s resolution;
session 0's extra second is larger than its 42 ms pre-compilation and is not
attributed.

## Reading

- Outputs are byte-identical to the scored N1 outputs.
- The change moves the affine4 compile cost before ready; it does not remove
  it. Warm: the function costs about 50 ms and warmup-1 TTFT is about 50 ms
  lower. Cold: the first 33–48-token request no longer pays about 0.4 s; the
  cold startup increase without the diagnostic was not measured.
- Decode medians fell monotonically with session order (86.7, 83.0, 81.1,
  79.3) regardless of the switch; four sessions cannot attribute any Decode
  difference to it in either direction. Lifecycle peaks are unchanged within
  0.01 GB.
- The cross-process compile cache could not be controlled; the cold benefit
  rests on one diagnostic session.

## Remaining before any acceptance

Formal acceptance under the original gates (32K five rounds, short three
rounds, matched controls, Decode non-inferiority, memory, quality review)
has not been run for this change.
