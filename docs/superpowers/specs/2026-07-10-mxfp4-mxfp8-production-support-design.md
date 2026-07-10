# MXFP4 and MXFP8 Production Support Design

**Date:** 2026-07-10
**Branch:** `feat/mxfp4-mxfp8`
**Status:** Correctness and clean performance gates passed; eight-hour rollout soak pending

## Goal

Add production-grade, checkpoint-native MXFP4 and MXFP8 inference support to
ironmlx. Both formats share one implementation task and one branch, but each
format has an independent correctness, HTTP, stability, and performance gate.

The implementation must load real MLX safetensors checkpoints without format
conversion and must dispatch the checkpoint's quantization mode to MLX rather
than treating the tensors as affine storage.

## Format Contracts

| Mode | Config spelling | Bits | Group size | Packed weight | Scales | Quant biases |
| --- | --- | ---: | ---: | --- | --- | --- |
| MXFP4 | `mxfp4` | 4 | 32 | `uint32` | `uint8` | absent |
| MXFP8 | `mxfp8` | 8 | 32 | `uint32` | `uint8` | absent |

ironmlx will accept the exact lowercase MLX mode spellings. It will reject
invalid bits, group sizes, scale dtypes, or affine bias tensors for these
modes with a contextual load error. No aliases or compatibility spellings are
part of this task.

## Architecture

### Quantization metadata

`QuantMode` gains distinct `Mxfp4` and `Mxfp8` variants. Each variant owns its
MLX mode string and storage rules. `QuantMeta` remains the single metadata
object propagated from the model config to modules.

The loader parses both top-level `quantization` and `quantization_config`, plus
the existing per-prefix overrides. Validation is mode-specific:

- affine and OptiQ retain their current supported bit widths and storage
  behavior;
- MXFP4 requires `(bits=4, group_size=32)`;
- MXFP8 requires `(bits=8, group_size=32)`.

Unknown modes remain hard errors.

### Module loading and dispatch

`Linear` and `Embedding` load the packed weight and scale tensors using the
mode-specific storage contract. Their forward paths pass `mxfp4` or `mxfp8`
unchanged to `mlx::quantization::quantized_matmul_on`.

All model-specific direct quantized matmul call sites that can consume global
or per-prefix metadata must propagate `QuantMode` instead of hard-coding
`affine`. This includes routed MoE and special projection paths. Existing
affine-only fused or self-written kernels must explicitly reject MXFP modes and
fall back to MLX's native quantized matmul path.

Gemma4 quantized fusion may combine tensors only when all participating
metadata, including mode, matches. MXFP fused tensors carry scales but no
quantization biases.

### MLX boundary

No `mlx-sys` ABI change is expected. The current Rust and C++ bridge already
passes the mode as a string to MLX, and MLX commit
`938006e4aee7d9e6c3ac9af3b6f343835a5438e2` supports `Mxfp4` and `Mxfp8` in
quantize, dequantize, quantized matmul, and gather quantized matmul.

Implementation and validation use the NAX-enabled installation at
`/Users/xin/.local/mlx` with deployment target 26.2.

## Error Handling

Checkpoint errors must be raised during model construction, before the first
request, whenever the violation can be determined from config and loaded
tensors. Error messages include the tensor prefix and expected mode contract.

The implementation must not silently:

- reinterpret MXFP tensors as affine;
- synthesize quantization biases;
- change group size or bit width;
- route MXFP tensors through `self_qmm` or another affine-layout kernel;
- dequantize the entire checkpoint as a fallback.

## Real Checkpoints

The primary validation pair uses the same Qwen3.5-4B base architecture so that
format behavior is isolated from model architecture differences:

| Mode | Repository | Revision |
| --- | --- | --- |
| MXFP4 | `mlx-community/Qwen3.5-4B-mxfp4` | `8e9cb97ec8ee0f6a04021220b7a6b5845353df56` |
| MXFP8 | `mlx-community/Qwen3.5-4B-mxfp8` | `a34dd69c7f165c0db75d71061e1bd8f4aeb9eead` |

Both configs declare group size 32 and their respective MLX mode. Their
checkpoint indexes contain packed `.weight` tensors and `.scales` tensors,
without `.biases` tensors for quantized modules.

The snapshots are stored under `/Users/xin/.ironmlx/models`. Revisions are
pinned in tests and validation manifests so a moving Hub branch cannot change
the evidence.

## Test Strategy

### Unit and component tests

- Parse valid MXFP4 and MXFP8 global metadata and per-prefix overrides.
- Reject wrong mode spelling, bit width, and group size.
- Reject MXFP scales with a non-`uint8` dtype and unexpected quant biases.
- Exercise synthetic MXFP4 and MXFP8 `Linear` forward against MLX
  dequantize-plus-matmul references.
- Exercise quantized `Embedding` lookup and tied output projection.
- Prove MXFP modes never dispatch to `self_qmm`.
- Prove compatible fused metadata preserves the MXFP mode and incompatible
  metadata disables fusion.
- Cover direct quantized matmul and gather quantized matmul mode propagation.

### Real-checkpoint correctness

Each pinned checkpoint must independently pass:

- loader metadata and tensor-contract checks;
- complete `Qwen35Model::from_loader` construction;
- finite prefill and decode logits;
- exact greedy first-token agreement with an MLX Python reference on the
  existing fixed prompt;
- last-position logits maximum absolute error below `0.5`, matching the
  existing Qwen3.5 structural parity gate;
- deterministic multi-token generation and blocking-thread execution.

### HTTP and stability matrix

Validation uses the external OpenAI-compatible HTTP boundary and records raw
JSON plus a manifest and summary. Each format independently covers:

- sequential smoke and multi-turn requests;
- strict decode at concurrency 1 and 8;
- 8K and 32K prompts at concurrency 1 and 8;
- target lengths 128 and 512 for long-prompt agent scenarios;
- repeated-request stability with zero request, server, or non-finite-output
  failures;
- health checks before and after each matrix segment.

An eight-hour soak remains a separate extended validation run, consistent with
the existing quantization validation policy. It is recorded before final
production rollout but does not block source implementation review.

## Performance Gates

Performance is measured after correctness passes, using the NAX MLX build and
the same server configuration for candidates and baselines.

- MXFP4 is compared with `Qwen3.5-4B-MLX-4bit`.
- MXFP8 is compared with `Qwen3.5-4B-MLX-8bit`.
- Sequential decode TPOT and concurrent ITL p95 must not exceed the matching
  affine baseline by more than 25%.
- 8K/32K HTTP E2E p95 at concurrency 1 and 8 must not exceed the matching
  affine baseline by more than 25%.
- Any larger regression blocks the corresponding mode from being marked
  production-ready and triggers profiling and optimization in this branch.

MXFP4 and MXFP8 receive separate results. One mode passing cannot mask a
failure in the other mode.

## Validation Results and Clean Performance Gate

Functional validation passed on the NAX-enabled MLX build at commit
`938006e4aee7d9e6c3ac9af3b6f343835a5438e2`.

- Pinned real-checkpoint parity passed for both formats. The full-vocabulary
  maximum absolute logit differences were `0.109375` for MXFP4 and `0.125000`
  for MXFP8, with exact greedy first-token agreement.
- Clean TG=128 and TG=512 HTTP matrices completed 565 requests with zero
  failures across sequential, multi-turn, stability, 8K/32K, c=1, and c=8
  cells. Both matrices pinned the complete scheduler configuration:
  `b_max=8`, `prefill_chunk_size=2048`, `admission_deadline_ms=5`,
  `admission_queue_max=32`, `max_cache_cap=65536`, and
  `decode_cadence_mid_chunk_cap=256`.
- The clean fixed-prompt strict-decode matrix completed 127 additional
  requests. Every c=1 and c=8 request generated exactly 512 tokens and ended
  with `finish_reason=length`.
- The clean affine-relative release gate evaluated 42 ratios and passed every
  one against the `1.25x` threshold. MXFP4's worst ratio was `1.218x` for
  32K/c=8/TG=128 long-prompt ITL p95, leaving limited but positive headroom.
  MXFP8's worst ratio was `1.138x` for 8K/c=1/TG=128 sequential TPOT. The
  strict-decode c=8 ratios were `0.974x` for MXFP4 and `1.090x` for MXFP8.
- The clean MLX active-memory ratios were `0.957x` for MXFP4 versus affine
  4-bit and `0.974x` for MXFP8 versus affine 8-bit.
- Earlier raw reports are not release-gate inputs because their manifests do
  not record the complete effective scheduler configuration and could therefore
  have auto-loaded different scheduler profiles.

The raw evidence is local, gitignored run output stored under:

- `reports/mxfp-validation/scheduler-fixed/tg128/2026-07-10-155931`
- `reports/mxfp-validation/scheduler-fixed/tg512/2026-07-10-163717`
- `reports/mxfp-strict-decode/scheduler-fixed/2026-07-10-173559`
- `reports/mxfp-performance/scheduler-fixed/2026-07-10-174518`

The repository keeps the validation runners, their regression tests, and this
curated summary; it does not version raw request payloads, process logs, or
machine-local artifact paths.

The synthetic 128-token prompt c=8 cell is not used for the short-decode ITL
gate because model-specific early EOS produced unequal completion counts and
load. The fixed-prompt full-length matrix replaces that cell for the decode
comparison. The eight-hour soak remains the previously agreed separate rollout
validation and is not represented as completed here.

## Baseline Exception

The clean `dev` baseline at `92affcf` has one deterministic unrelated failure:
`mtp_stream_rolls_back_and_replays_accepted_prefix_after_partial_reject`.
Commit `9078f31` made an empty fake cache satisfy the full-cache trim predicate,
so the test's expected replay call no longer occurs. This task records the
failure and does not alter MTP behavior or expectations.

## Non-Goals

- NVFP4 support.
- GGUF or compressed-tensors checkpoint ingestion.
- A checkpoint conversion or quantization tool.
- Activation quantization or MXFP8 `qqmm` input quantization.
- Custom MXFP Metal, TensorOps, or `self_qmm` kernels.
- Changes to existing affine, OptiQ, or bf16 checkpoint semantics.
- Repairing the unrelated MTP baseline test.

## Delivery Structure

The branch has one shared implementation followed by two independent gates:

1. shared metadata, storage validation, module dispatch, and regression tests;
2. MXFP4 real-checkpoint and production matrix gate;
3. MXFP8 real-checkpoint and production matrix gate;
4. combined documentation, benchmark evidence, and final Rust quality gate.

Both mode performance gates now pass under pinned, identical scheduler
configuration. The remaining release-rollout condition is the separate
eight-hour soak; until it passes, this result is a clean source and performance
gate, not final production-rollout approval.
