# B1 API optimization — development evidence

Status: **accepted for the fixed B1 API qualification**, 2026-09-30, under the
user-approved relative-quality criterion with model defects disclosed.
The eight-session four-application comparison
passes the performance gate and complete natural-stop serial/tree equality has
passed. A further balanced cycle and the combined twelve-session analysis also
support that performance result. This does not certify error-free knowledge
answers or untested workloads. See the [report](b1-api-performance-report.md).
Protocol: [frozen protocol](b1-api-performance-protocol.md). Base: `a2aec98d887`.
Performance observations below are local development screens, not confidence-backed wins.
Raw reports/logs are retained locally under
`benchmarks/b1-api-comparison/results/local-archive-2026-09-30/`. They are
ignored by Git because some logs and local settings may contain credentials.

## Fresh initial screens

Six measured natural-stop requests per application, plus two unrelated warmups.
These pooled medians are descriptive only; the final gate uses paired
per-prompt ratios, category guards and block-bootstrap confidence intervals.

| Application | TTFT s | Decode tok/s | E2E s |
|---|---:|---:|---:|
| IronMLX base, effective block 8 | 0.2273 | 47.93 | 19.46 |
| oMLX 0.7.0rc1, aggressive burst | 0.2599 | 78.27 | 12.32 |
| Splash 1.1.0 | 0.2440 | 68.26 | 12.57 |
| TensorFold 0.3.6.2 | 0.1226 | 75.71 | 12.31 |

Files: `baseline-ironmlx-valid.json`, `baseline-{omlx,splash,tensorfold}-s0.json`.
The first IronMLX CLI launch without an explicit metallib failed; a request
using unsupported `reasoning_effort=none` also failed. Both logs were retained.
The corrected IronMLX requests use `enable_thinking=false` without that field.

## Tensor-unit route

The M5 affine4/group64 BF16 prototype adapts TensorFold's MIT-licensed tiled
QMM, retaining a separate packed copy. Its reduction order differs from
ordinary MLX; it is not claimed to be an ordinary-MLX bitwise replacement.
Only an explicit experimental environment variable activates it. The current
scope is the Qwen3.8-27B DFlash2 target on compatible M5+ architecture strings.
M1–M4 and incompatible quantization profiles retain existing routing.
Future hardware has not been physically tested.

Initial QMM row-width tests passed, but full-model serial/verify parity failed
on two knowledge prompts. Layer tracing localized the divergence to layer 0.
The Q1 route projected GDN's 48-channel b/a tensors separately (unsupported by
the 32-wide tiled kernel), while verify fused them. Making projection fusion
identical for Q1, prefill and verify removed the observed divergence. All six
128-token diagnostics then matched; all 64 traced layers matched as well.

| Natural-stop API candidate | TTFT s | Decode tok/s | E2E s |
|---|---:|---:|---:|
| Fused M5 route + fixed d=7 | 0.2129 | 74.52 | 13.50 |
| Additionally grouped verify attention | 0.2066 | 58.94 | 16.30 |
| Fused M5 route + single prefill, no grouping | 0.1657 | 73.13 | 12.52 |

Files: `m5-fused-d7-ironmlx-s0.json`, `m5-grouped-d7-ironmlx-s0.json`,
`m5-single-prefill-d7-ironmlx-s0.json`. Each has six naturally stopped answers.
Output lengths differ across prefill arithmetic variants; E2E alone therefore
does not establish an optimization win. Grouped attention passed boundary and
model parity checks but regressed in full API requests, and is not selected.
64-wide cooperative tiles plus shape-based K splitting passed limited numerical
checks but did not produce a stable development improvement; they remain off.

## Tree and draft work

Existing leaf-path batching computes shared prefixes repeatedly. Its 15-node
tree retained serial token equality in six 128-token diagnostics, but was much
slower than linear Q8. `m5-tree15-serial-d7-128.json` retains that result.

A separate flat-tree experiment computes each unique node once, uses depth-based
positions, parent-indexed GDN recurrence and transactional accepted-path commits.
The initial output dtype mismatch was retained as a failed experiment and fixed.
`m5-flat-v2-serial-d7-128.json` passed six prefix token checks; its layer traces
include branching layouts and showed no divergence. Full natural-stop and
diverse-branch regression qualification are still required.

TensorFold source inspection additionally found its greedy tree profile uses
up to 16 draft positions, four children, pairwise weight 0.6 and score temperature
1.5, despite the checkpoint's native block size 8. This is a proposal-only
experiment, not permission to silently change production block-size defaults.
`IRONMLX_EXPERIMENTAL_DFLASH2_TREE_PROFILE=tf-v1` explicitly selects that candidate;
the ordinary checkpoint-aware Q8 default remains unchanged. The experimental
flat-tree route currently requires a fixed budget and greedy sampling.

Grouping tree attention queries by equal path depth preserves the one-query
kernel/key-length regime while sharing dispatches. Six short diagnostic runs
and branching layer traces passed with the experimental batch-attention switch.

Local MLX 0.32.2 source (`metal/sort.cpp`, `ArgPartition::eval_gpu`) confirms
argpartition delegates to a full merge sort. A BF16 radix top-k candidate keeps
the full vocabulary and resolves all ties by token ID without a bounded tie
buffer. CPU-reference tests cover 257, 8193 and 248320 columns, multiple rows,
K=1/16/64, and all-equal logits. Performance and full-model qualification remain
separate gates. TensorFold's additional draft-vocabulary restriction has **not**
been adopted; target verification always uses the full vocabulary.

## Profiling limits

Additional full API screens (six prompts, not acceptance):

| Candidate | TTFT s | Decode tok/s | E2E s |
|---|---:|---:|---:|
| Linear Q8 + single prefill + exact radix top-k | 0.1455 | 77.82 | 11.99 |
| Flat tree + TF proposal profile + grouped tree attention + radix | 0.1471 | 66.69 | 13.25 |

Files: `m5-linear-radix-d7-ironmlx-s0.json`,
`m5-flat-tree-radix-ironmlx-s0.json`. Outputs match the preceding
single-prefill candidate exactly; the tree route is not currently selected.
These pooled medians do not establish the frozen per-prompt/category/CI gates.

A separate M5 causal attention experiment uses fixed 64-key tiles and 512-key
partial reductions, adapting TensorFold's MIT-licensed direct register variant.
BF16 queries/keys, FP16 softmax products and FP32 accumulators are intentional
arithmetic differences from ordinary MLX. Six boundary fixtures (prefixes
0, 63, 511, 512, 1024, 2048) passed bitwise Q8-versus-Q1 checks; maximum absolute
error versus ordinary MLX was 0.001953125 on those synthetic inputs. All six
128-token model comparisons and 64-layer traces subsequently passed. The
natural-stop linear API screen (`m5-lane-attention-radix-ironmlx-s0.json`) was
0.1626 s / 69.74 tok/s / 12.57 s; this was not an improvement over the earlier
linear/radix route, and changed outputs must be audited separately.

A further tree-specific attention experiment shares full committed-prefix
tiles across query nodes and gathers only each path's small tail inside Metal.
It adapts TensorFold `stream_attention` to B1, preserving physical KV strides
and the serial route's logical key order. This replaces full-prefix gathers
and multiple stock SDPA calls; benefits remain a hypothesis until measured.

The native tree-attention screen completed at 0.1391 s / 74.00 tok/s / 11.13 s.
The next combined candidate batches selector host-transfer casts before one
GPU evaluation boundary and uses four SIMD-group tiles per attention group.
Its full API screen (`m5-tree-group4-ironmlx-s0.json`) completed at 0.1393 s /
83.51 tok/s / 10.38 s, with identical answer hashes and lengths to the native
tree-attention candidate. These two changes are not separately ablated, so
their individual contributions are not claimed. The fixed grouping passed
nine focused kernel tests, including serial/branching boundary equality.
The three generated programs passed the code audit harness: 106 interval
cases, LRU access/update/eviction/object-key cases, and async concurrency/order/
exception checks. Factual knowledge review remains separate.

Library regression attempt 1 failed because test executables could not locate
`mlx.metallib` (118 failures, retained log). Current MLX source uses an explicit
override or colocated library, not the old environment variable at runtime.
A symlink beside the test executables to the verified local metallib corrected
the test environment. Attempt 2 passed 613 LM tests and 537 runtime tests;
14 and 7 tests, respectively, remained ignored. This does not count ignored
real-model tests as passing.

`candidate-natural-serial-tree-v2.json` now passes all six complete responses:
serial and 15-node tree token sequences are exactly equal through natural EOS
(686, 1394, 448, 1420, 1623, 448 tokens, including EOS). This validates the
selected route against its own serial arithmetic, **not** bitwise equivalence
to ordinary MLX. The first natural-stop attempt omitted `--tree-nodes 15`;
it was terminated as misconfigured and its log retained, not counted as a pass.

`m5-cpu-profile.sample.txt` is a five-second CPU call-stack sample. It shows
substantial waits inside MLX evaluation and on GPU completion. The runtime's
`verify_schedule_us` includes waiting and must not be labelled CPU compute time
or isolated kernel GPU time. Its associated diagnostic was sampled and capped,
so it is excluded from API performance acceptance.

All prototype controls are opt-in. No production default, merge or release has
been authorized by these experiments. Failed/inconclusive candidates are retained
and do not count toward the user's competitive performance gate.
