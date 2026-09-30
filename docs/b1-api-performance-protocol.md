# B1 API performance acceptance protocol

Frozen before implementation: 2026-09-29; base `a2aec98d887f405e4d3faf5827321441f9cf7987`.

## Workload and service contract

Use the six original quick-comparison prompts, verbatim, in
`scripts/fixtures/b1-api-prompts.json` (three code, three knowledge). Do not
replace difficult prompts after seeing results. Two separate warmup prompts
precede each server session. Requests use temperature 0, top_p 1, thinking
off, one user message, streaming, max_tokens 4096 as a safety ceiling only.
Every measured response must finish naturally (`stop`); truncation, errors,
reasoning leakage or invalid output invalidate acceptance, not just the row.

Same M5 Max, AC power, same target snapshot `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
and DFlash2 snapshot `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` across applications.
Draft affine4/group64 where supported; record any preparation differences.
No concurrent inference, compilation or GPU tests during measured sessions.
Use normal application HTTP serving paths and Release IronMLX builds.
Versions: oMLX 0.7.0rc1, Splash 1.1.0, TensorFold 0.3.6.2. Record artifact/source
identity, dependencies and effective configuration, not just package labels.

Measure warm-loaded, uncached independent requests. Disable persistent prompt
reuse and in-memory prompt caches where exposed. If a service cannot disable
its cache, each prompt appears once per fresh process/session; record any
shared-prefix hits. No measured prompt is a warmup. Restart every session,
with startup excluded. Retain prepared weights and compiled kernels on disk.
Wait 1 second between requests and 10 seconds between application sessions.
Record thermal warnings before/after sessions; warn and repeat affected blocks
with their originals preserved. Prompt order rotates by session for every app.

## Measurement and statistics

TTFT: request start to first nonempty visible content SSE event. This is a
client-visible first-content latency, not a GPU-only timing. Decode TPS:
`(server completion_tokens - 1) / (last content time - first content time)`.
Retain all content-event timestamps/sizes, usage, common-tokenizer token count,
first-chunk token count, output/hash and stop reason to audit batching, hidden
tokens and endpoint counting differences. Report common-tokenizer TPS alongside
server TPS; disagreement that changes ranking prevents acceptance until resolved.
E2E: request start to completed HTTP response stream, including final usage.

Initial baseline and development screens: one six-prompt session per app or
candidate, explicitly not acceptance. Final acceptance: eight sessions per app,
two balanced four-session cycles in application order:

1. IronMLX, oMLX, Splash, TensorFold
2. oMLX, TensorFold, IronMLX, Splash
3. Splash, IronMLX, TensorFold, oMLX
4. TensorFold, Splash, oMLX, IronMLX

Thus each app has eight observations per prompt (48 measured requests).
Report each prompt's median and both category summaries. For a metric, compute
the equally weighted geometric mean of six per-prompt IronMLX/rival median
ratios. Bootstrap paired session blocks (10,000 resamples, seed 20260929) for
95% percentile intervals. Preserve prompt pairing and all six prompts together.
Lower is better for latency; higher is better for TPS.

"Better": the entire interval is on the favorable side of 1. "No worse": the
point estimate is on the favorable side of 1 and the interval excludes a
regression greater than 3% (latency upper <=1.03; TPS lower >=1/1.03). Report
this explicitly as a 3% noninferiority margin, never as proven exact equality.
No category may have a point-estimate regression >3% on a claimed passing
metric. All three metrics must beat oMLX; at least two must be no worse than
TensorFold. Splash has no mandatory superiority gate. Borderline results are
inconclusive, not a pass; add a complete balanced cycle for all apps if needed.

## Correctness and evidence

Review knowledge outputs for requested coverage and factual errors; execute
extracted code in isolated test fixtures for interval merging, LRU eviction and
async concurrency/order/error handling. Record natural output lengths; E2E
advantages from shorter answers require equivalent task coverage and cannot
substitute for decode gains. Cross-kernel bitwise identity is not assumed:
test new math against ordinary MLX, row-width invariance, and speculative versus
serial generation using the same new execution path. Numerical or task-quality
regressions block acceptance. Keep experimental routes opt-in until qualified.

Keep raw unsuccessful experiments, commands, source changes and a final report.
DSH E2E and the deferred stability qualification are outside this stage; focused
stability/correctness regression checks are still required. No merge or release.

## Hardware research

Apple documents direct tensor-operation acceleration for Apple GPU family 10+
in [Metal 4 inline ML](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4).
Investigate weight reuse, arithmetic intensity, reduction order, dispatch and
host synchronization with measurements. TensorFold's complete lane package is
an initial hypothesis supported by exploratory ablations, not a proof that one
kernel alone delivers the goal. Its MIT license must accompany adapted code.

## Development experiments (not production defaults)

`IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM=1` opts the Qwen3.8-27B DFlash2
target into a tensor-unit affine4/group64 BF16 route on `applegpu_gN`, N>=17.
Other formats retain existing routing. The experimental reduction order differs
from ordinary MLX; it must not be described as bitwise equivalent to that
reference. It retains extra packed weights during this prototype. The shader
is adapted from TensorFold v0.3.6.2 under its MIT license, retained beside it.

`IRONMLX_EXPERIMENTAL_DFLASH2_FIXED_BUDGET=0..7` freezes the request's draft
budget for diagnosis; 0 is the serial target control. It is validated against
the checkpoint and verify capabilities. With neither environment variable set,
the existing defaults and policy remain in force. `dflash2-lane-diagnostic`
compares exact token IDs across budgets and reports stage timings; capped runs
are correctness diagnostics, never natural-stop performance acceptance.

The local runner `scripts/run_b1_api_sessions.py` records commands and runs one
owned application process at a time. It uses the preserved release artifacts
under `/Users/xin/workspace/b1-rival-benchmark/artifacts`; Splash's prepared
manifest confirms the same target revision and local drafter. oMLX's isolated
settings select aggressive burst decode, B1 and disabled DFlash prefix reuse.
TensorFold uses lane kernels, B1 and prompt-cache disabled. Splash defaults to
INT8 KV and no disk cache; logs must confirm zero cached tokens for this workload.
IronMLX uses BF16 KV and no cross-request prefix cache. These are recorded
application-specific configurations, not an assertion of bitwise output identity.

`IRONMLX_EXPERIMENTAL_GROUPED_VERIFY_ATTN=1` additionally groups compatible
B1 BF16 attention queries under the experimental M5 route. It is restricted
to the audited GQA=6/head-dimension=256 shape and excludes quantized KV and
explicit masks. The local MLX 0.32.2 vector-kernel/key-partition boundaries
must retain serial arithmetic; boundary tests and full-model parity are
required. Other runtimes are not qualified by these results.

## Candidate freeze for the final comparison

The selected candidate keeps normal Release API serving and uses the following
explicit experimental configuration (not a new global default):

```sh
IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM=1 \
IRONMLX_EXPERIMENTAL_M5_LANE_ATTN=1 \
IRONMLX_EXPERIMENTAL_M5_ATTN_GROUP_TILES=4 \
IRONMLX_EXPERIMENTAL_DFLASH2_FIXED_BUDGET=7 \
IRONMLX_EXPERIMENTAL_DFLASH2_SINGLE_PREFILL=1 \
IRONMLX_EXPERIMENTAL_DFLASH2_RADIX_TOPK=1 \
IRONMLX_EXPERIMENTAL_DFLASH2_FLAT_TREE=1 \
IRONMLX_EXPERIMENTAL_DFLASH2_TREE_PROFILE=tf-v1 \
/Users/xin/workspace/b1-rival-benchmark/artifacts/tensorfold/venv/bin/python \
scripts/run_b1_api_sessions.py --label final-candidate-v1 --sessions 8 \
  --ironmlx-tree-nodes 15
```

This uses checkpoint block 8 for ordinary linear proposals and a separately
qualified, explicit 16-position/15-node tree proposal experiment. No vocabulary
restriction, answer rewriting, forced short output, or changed sampling is used.
The group size changes launch scheduling without changing output bits. Selector
casts are evaluated together to avoid repeated small GPU/host synchronization.
The candidate-source manifest freezes dirty source files, base commit, binary,
MLX archive/metallib, rival source digests and model metadata before final runs.
The measured machine has 40 GPU cores and reports Metal 4 support. Other device
generations, model formats, longer prompts and concurrency remain unqualified.
