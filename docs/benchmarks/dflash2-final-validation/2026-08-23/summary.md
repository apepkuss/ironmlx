# DFlash2 P3 Final Validation

Date: 2026-08-23

Status: passed. P3, P3.5, P3.5.1, P3.5.2, P3.6 and the pre-P4
performance/regression gate are complete.

## Baseline

| Component | Revision or configuration |
|---|---|
| IronMLX | `feat/dflash2-execution-path@a565a6871e084b224dd9e7fc40c14a03c064ba90` |
| MLX | `0.32.2`, local NAX/QMM source baseline `fix/nax-qmm-product-shape-main@73ad5df20` |
| macOS | `26.4` (`25E246`) |
| Hardware | MacBook Pro `Mac17,6`, Apple M5 Max, 18 CPU cores, 128 GB unified memory |
| Target | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| MTP | `mlx-community/Qwen3.8-27B-MTP-4bit@b643c01b6d3b094e325edb6ebd832e16c486c575` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

DFlash2 used block size 4 and 4-bit runtime draft quantization. Greedy B1
comparison used `max-sequences=1`. Sampled repeated runs had one active request
on a server configured with `max-sequences=2`; concurrency and queue tests used
`max-sequences=2` and `admission-queue-max=2`.

## Measurement Contract

- Greedy comparison: three fixed math, Chinese and Rust/UTF-8 prompts,
  `max_tokens=128`, `temperature=0`, `ignore_eos=true`.
- `Wall tok/s` is `completion_tokens / HTTP request wall time`; it includes
  request preparation, prefill, generation and response assembly.
- `Generation tok/s` is the DFlash2 actor's generation-only health metric.
- Each greedy cell is one paired acceptance run on the same machine. These rows
  are a regression gate and directional comparison, not a statistically stable
  cross-machine benchmark.
- Sampled runs use `temperature=0.7`, `seed=20260823`, the checkpoint default
  `top_k=20`, and 128 forced output tokens.

## Greedy Correctness and Performance

All three DFlash2 and MTP responses matched ordinary decoding byte for byte; the
SHA-256 of decoded response text was identical within each prompt row.

| Prompt | Ordinary wall tok/s | MTP wall tok/s | DFlash2 wall tok/s | vs ordinary | vs MTP | DFlash2 generation tok/s | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math | 27.843 | 27.572 | 49.207 | +76.7% | +78.5% | 54.957 | 68.80% |
| Chinese | 30.053 | 34.924 | 48.123 | +60.1% | +37.8% | 52.534 | 56.74% |
| Rust/UTF-8 | 28.485 | 28.469 | 49.018 | +72.1% | +72.2% | 53.099 | 56.03% |

The fixed math reference reached 54.96 generation tok/s and passed the accepted
54-57 tok/s Greedy gate. Prompt-dependent acceptance explains the lower
generation TPS in the Chinese and Rust rows; all three retain a substantial
same-condition wall-throughput advantage over ordinary decode and MTP.

## Exact SampledVerify

Three repeated sampled runs produced the same response SHA-256,
`a41d89508ebbe01609b9156d4b8544927c55061f9b6663e0b4d163294f25a201`.

| Run | Generation tok/s | Acceptance | Residual corrections |
|---:|---:|---:|---:|
| 1 | 40.964 | 56.74% | 27 |
| 2 | 41.031 | 56.74% | 27 |
| 3 | 40.840 | 56.74% | 27 |

The exact deterministic-coupling distribution test passed. Non-zero residual
corrections confirm that the rejection path was exercised rather than only the
accept-all path.

## B>1 Runtime Acceptance

One greedy and one sampled request, each requesting 384 tokens, ran
concurrently:

| Request | Completion tokens | Wall time | Result |
|---|---:|---:|---|
| Greedy | 384 | 18.682 s | Completed |
| Sampled | 384 | 18.682 s | Completed |

The live health snapshot reported `b_active=2`, `b_queued=0`. Aggregate output
throughput was 41.079 tok/s and both requests completed at effectively the same
time. This validates isolated actor-level concurrency and fairness; it is not a
tensor-batching throughput claim.

Queue saturation with five concurrent 384-token requests reported
`b_active=2`, `b_queued=2`, and `admission_queue_full_count=1`. Four requests
completed with HTTP 200; one returned HTTP 503 with
`scheduler_queue_full`. After completion, both active and queued counts returned
to zero.

A 1024-token SSE request was disconnected after one second. The client observed
a timeout/disconnect and the server returned to `b_active=0`, `b_queued=0` after
the current safe forward boundary.

## Protocol Acceptance

Release HTTP smoke passed for synchronous and SSE forms of:

- OpenAI Chat Completions;
- OpenAI Responses typed lifecycle;
- Anthropic Messages lifecycle.

Responses with `reasoning.effort=none` emitted 48 text deltas and one terminal
event. Chat emitted content deltas plus `[DONE]`; Messages emitted text deltas
and `message_stop`. No protocol used a fallback execution path.

## Source and Test Gates

- `cargo fmt`: passed;
- `cargo +nightly fmt --all -- --check`: passed;
- `cargo +nightly clippy --all-features --workspace -- -D warnings`: passed;
- `cargo build --release`: passed;
- `cargo test --all-features --workspace -- --test-threads=1`: passed;
- DFlash2 focused tests: 20 passed, 1 real-checkpoint test ignored by default;
- the ignored Qwen3.8 ordinary-Q1/DFlash2 logits exactness test was enabled
  separately with the real target checkpoint and passed.

## Evidence Boundaries

- Results apply to the pinned machine, revisions, checkpoints and parameters
  above. They do not define a universal TPS promise.
- Greedy numbers are single paired runs; temperature sampling uses three
  repeated runs only. Re-run a counterbalanced multi-run benchmark before using
  these values for a public performance claim.
- DFlash2 B>1 is request-level actor concurrency. True MLX tensor batching is a
  separate future performance task.
- No local absolute paths, raw prompts or generated response bodies are part of
  this committed record.

Detailed numeric rows are available in [`summary.csv`](summary.csv).
