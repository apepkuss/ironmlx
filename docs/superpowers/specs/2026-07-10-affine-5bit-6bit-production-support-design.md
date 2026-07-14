# Affine 5-bit and 6-bit Production Support Design

**Date:** 2026-07-10
**Branch:** `feat/affine-5bit-6bit`
**Status:** Production support complete; eight-hour soak remains a separate task

## Goal

Extend ironmlx's existing affine quantization implementation to load and serve
native MLX 5-bit and 6-bit checkpoints at production quality. The implementation
remains shared across affine 2/4/5/6/8-bit weights; 5-bit and 6-bit receive
independent correctness and performance gates.

An eight-hour soak is explicitly outside this task. It will be run later as a
separate rollout task.

## Checkpoint Contract

Affine checkpoints use the existing MLX storage contract:

- `mode = "affine"`;
- `group_size` is 32, 64, or 128;
- `bits` is 2, 4, 5, 6, or 8;
- packed weights are `uint32` tensors;
- scales and quantization biases are floating tensors with identical shapes;
- logical width is recovered with `packed_columns * 32 / bits` and must divide
  exactly;
- the scale and quantization-bias trailing dimension is
  `logical_width / group_size`.

The non-power-of-two packing is owned by MLX. A 5-bit block packs eight values
into five bytes, and a 6-bit block packs four values into three bytes. ironmlx
must not unpack or reinterpret this storage in model code.

OptiQ retains its existing independent bit-width contract. Adding affine 5/6-bit
must not make OptiQ 5/6-bit metadata valid.

## Architecture

### Metadata and validation

`QuantMode::Affine` remains the only mode for 5/6-bit checkpoints. Loader
validation accepts affine 2/4/5/6/8-bit and rejects 3-bit, 7-bit, unsupported
group sizes, missing quantization biases, invalid dtypes, non-integral packed
widths, and incompatible scale/bias shapes before the first request.

The packed-width calculation becomes a checked shared helper. `Linear`, MTP
shape validation, and future callers use the same calculation so 5/6-bit
weights cannot be interpreted with the power-of-two-only `32 / bits` formula.

### Runtime dispatch

Affine 2/4/5/6/8-bit share the existing MLX-native paths:

- `quantized_matmul_on` for Linear and tied output projection;
- `dequantize_on` for gathered embedding rows and dense-weight consumers;
- `gather_quantized_matmul_on` for routed MoE paths;
- existing metadata-preserving Gemma4 fusion.

Existing 4-bit-only optimizations remain guarded by `bits == 4` and are not
silently used for 5/6/8-bit:

- `self_qmm`;
- quantized Embedding decode;
- Gemma4 fused GeGLU decode.

5/6-bit use MLX's Metal/NAX kernels, matching the 8-bit strategy. The final
counterbalanced performance gate passed without a new ironmlx custom kernel.

No `mlx-sys` ABI change is required. Validation uses the NAX-enabled MLX build
at commit `938006e4aee7d9e6c3ac9af3b6f343835a5438e2` through
`/Users/xin/.local/mlx/mlx-env.sh`.

## Real Checkpoints

All repositories are pinned by revision in validation manifests.

| Architecture | Bits | Repository | Revision |
| --- | ---: | --- | --- |
| Gemma4 E2B-it | 4 | `mlx-community/gemma-4-e2b-it-4bit` | `238767527555cb75a05732a84dff5d6ba0dd6809` |
| Gemma4 E2B-it | 5 | `mlx-community/gemma-4-e2b-it-5bit` | `dc565aea8c49afb542497310a2d86bf1fd91391f` |
| Gemma4 E2B-it | 6 | `mlx-community/gemma-4-e2b-it-6bit` | `ebd7756d4e55627e11ae043af9cad8ed6465a2e2` |
| Qwen3.5 2B | 4 | `mlx-community/Qwen3.5-2B-4bit` | `674aaa7240b91e8012fcad5d791b7dfe5ba90207` |
| Qwen3.5 2B | 5 | `mlx-community/Qwen3.5-2B-5bit` | `0934527791eb8008cd84b66550b8ab3eefd15b85` |
| Qwen3.5 2B | 6 | `mlx-community/Qwen3.5-2B-6bit` | `ba2bcf03dd5b502646de7e32b003cf538f2ca4d6` |

The 4-bit checkpoints are same-architecture performance baselines, not
correctness references for 5/6-bit logits.

## Correctness Gates

Unit and component coverage must prove:

- valid global and per-prefix affine 5/6-bit metadata parsing;
- invalid mode, bit-width, group size, dtype, packed width, and scale/bias shape
  rejection;
- exact logical width recovery for 2/4/5/6/8-bit packed tensors;
- 5/6-bit Linear decode and prefill agreement with MLX
  dequantize-plus-dense-matmul references;
- quantized Embedding lookup and tied output projection agreement;
- Gemma4 fused projection metadata preservation;
- routed/gather QMM mode and bit-width propagation;
- guaranteed bypass of all 4-bit-only custom kernels.

Each 5/6-bit real checkpoint independently passes:

- complete model construction and eager storage validation;
- finite prefill and decode logits;
- exact greedy first-token agreement with an MLX Python reference;
- exact first-token argmax and four-token greedy sequence agreement;
- raw full-vocabulary and top-logit maximum absolute error below `1.0`;
- centered maximum absolute error below `0.55`, RMSE below `0.10`, and p99
  absolute error below `0.25`;
- top-64 overlap of at least 60 tokens;
- deterministic multi-token generation and blocking-thread execution.

## HTTP and Stability Gates

The external OpenAI-compatible HTTP boundary runs with a pinned complete
scheduler configuration shared by candidates and baselines. Each architecture
and bit-width covers:

- sequential smoke, multi-turn, and repeated-request stability;
- strict full-length 512-token decode at c=1 and c=8;
- 8K and 32K prompts at c=1 and c=8;
- target lengths 128 and 512;
- health checks before and after each matrix segment;
- zero request, server, non-finite-output, or premature-EOS failures in strict
  decode.

Raw request logs and benchmark artifacts are stored under gitignored
`reports/`; only runners, tests, and this curated result summary are committed.

## Performance Gates

5-bit and 6-bit are compared with the matching architecture's affine 4-bit
checkpoint. Larger bit-widths move more packed weight data, so the hard gate is
normalized by the theoretical packed-weight ratio rather than requiring equal
latency:

- 5-bit latency ratio must not exceed `1.375x` the 4-bit baseline
  (`5/4 * 1.10`);
- 6-bit latency ratio must not exceed `1.650x` the 4-bit baseline
  (`6/4 * 1.10`);
- the 5-bit result must not be more than `1.10x` slower than the 6-bit result in
  a matching cell;
- active memory must remain between the matching 4-bit and dense/bf16
  checkpoints and must not exceed its expected packed-weight growth by more
  than 10%;
- strict prefill uses two complete rounds with exact reverse model order. Each
  round uses two warmups and five measured requests per PP=2K/8K/32K cell, a
  fixed nonce, and one-second inter-run cooldown. The gate pools all ten raw
  TTFT samples per model/cell before taking the median;
- ordinary TG=128/512 matrices prove HTTP, long-context, concurrency,
  multi-turn, stability, and memory behavior, while strict full-length decode
  and counterbalanced prefill provide the release latency metrics;
- every strict-prefill median, strict-decode ITL p95, and memory comparison
  must pass independently for Gemma4 and Qwen3.5.

A failure triggers profiling and optimization in this branch. One architecture
or bit-width passing cannot mask another failure.

## Final Production Evidence

The release runs used one scheduler configuration: `b_max=8`, prefill chunk
`2048`, admission deadline `5ms`, admission queue `32`, cache cap `65536`, and
mid-chunk decode cadence cap `256`.

- TG=128 HTTP matrix: `reports/affine56-validation/tg128/2026-07-10-215443`,
  269 measured requests, zero failures;
- TG=512 HTTP matrix: `reports/affine56-validation/tg512/2026-07-10-222849`,
  251 measured requests, zero failures;
- strict full-length decode:
  `reports/affine56-strict-decode/2026-07-10-231838`, 230 measured 512-token
  requests, zero failures or early stops;
- counterbalanced prefill rounds:
  `reports/affine56-prefill/2026-07-10-234230` and
  `reports/affine56-prefill/2026-07-10-235555`, 180 measured requests, zero
  failures or early stops;
- final gate: `reports/affine56-performance/2026-07-11-001650`, with Gemma4
  5-bit, Gemma4 6-bit, Qwen3.5 5-bit, and Qwen3.5 6-bit all `passed`.

At PP=32768, the final pooled prefill ratios were Gemma4
`5/4=0.993`, `6/4=0.928`, `5/6=1.070`, and Qwen3.5
`5/4=1.071`, `6/4=1.101`, `5/6=0.973`. The worst strict-decode ratios against
4-bit were `1.126` for 5-bit and `1.210` for 6-bit. Active-memory ratios stayed
within the expected packed growth, with maxima of `1.201` for 5-bit and `1.403`
for 6-bit.

The release real-checkpoint test loaded all six pinned models, ran prefill plus
four-token greedy decode twice, and repeated each case from a blocking thread.
For the four 5/6-bit candidates, the worst raw max-absolute logit error was
`0.718492`, centered max-absolute error `0.494750`, centered RMSE `0.076849`,
centered p99 absolute error `0.196221`, and top-64 overlap `63`. All first-token
argmax and four-token greedy sequences matched their pinned MLX references.

The initially failing single-round Gemma4 PP32K `5/6=1.132` result reversed
direction when model order changed. A second exact reverse-order round showed
up to about 9% absolute TTFT drift for the same checkpoint. Pooling the two
rounds removed this order bias without changing runtime code or relaxing the
`1.10` threshold.

Final verification passed `cargo fmt`, nightly fmt check, workspace/all-features
clippy with warnings denied, release build, all 59 Python runner/gate tests,
Python bytecode compilation, and the six-model release correctness test. The
full workspace/all-features Rust run retained one pre-existing baseline failure:
`p8c_mtp_speculative::mtp_stream_rolls_back_and_replays_accepted_prefix_after_partial_reject`.
There are no branch changes relative to `dev@a28d595` in that test,
`core/speculative.rs`, or `core/generate.rs`.

## Quality and Delivery Gates

Before completion, the branch must pass:

- `cargo fmt`;
- `cargo +nightly fmt --all -- --check`;
- `cargo +nightly clippy --all-features --workspace -- -D warnings`;
- `cargo build --release`;
- focused and full regression tests, with any pre-existing baseline failure
  recorded separately;
- all real-checkpoint correctness, HTTP, stability, and clean performance gates.

Production completion means both 5-bit and 6-bit pass every gate above. The
separate eight-hour soak is not represented as completed or required in this
branch.

## Non-Goals

- Affine 3-bit support.
- NVFP4 support.
- OptiQ 5/6-bit support.
- New checkpoint conversion tooling.
- Extending a custom kernel without a measured production-gate need.
- The separate eight-hour soak.
