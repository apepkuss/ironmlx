# DFlash2 Pairwise/B2 Tensor Batching Historical Archive

Date: 2026-08-24

Status: passed and superseded. The B2 kernel gate established strict per-row
equivalence with a modest device-level speedup. The later Pairwise actor path
improved concurrent Greedy and Sampled aggregate throughput while retaining B1
output records. The subsequent [B=N qualification](../2026-08-24/summary.md)
supersedes this record for current width selection.

## Revisions and Environment

| Component | Revision or configuration |
|---|---|
| Request-rotation baseline | `bf8abe3940c58f489f2b9822a0018ca7935b7dba` plus the uncommitted closed-batch runner |
| B2 kernel qualification | `331b0d05cc745204a079d6c5f39b36b5f4db9822` |
| Pairwise tensor batching | `fc28af6a6568ae7bccb60874550ad47c84aa9045` |
| MLX | local NAX/QMM baseline `73ad5df20cb30be4192e5c4d0ae8130674773427` |
| macOS | `26.4` (`25E246`) |
| Hardware | MacBook Pro `Mac17,6`, Apple M5 Max, 18 CPU cores, 128 GB unified memory |
| Target | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

The HTTP servers used DFlash2 block size 4, draft 4-bit,
`max-sequences=4`, and a 32,768-token logical KV cap. Prefix cache was disabled
for the retained throughput reports.

This archive extends the earlier
[P3 final validation](../../dflash2-final-validation/2026-08-23/summary.md).
It preserves the historical step between request-level actor rotation and the
later arbitrary-width tensor implementation.

## Measurement Contracts

The dedicated runner measures complete closed batches. Aggregate TPS is total
completion tokens divided by the wall time until every request in the batch
finishes. This avoids the tail-request inflation found in the initial
duration-window screening run.

Two prompt contracts were used:

| Prompt ID | Contract | SHA-256 |
|---|---|---|
| `fixture_readme_519b` | Repository fixture, 519 bytes | `14e61403e99c7c570d30cb5ee617f51f9706c50bef74d43a7be836d3e3e1d402` |
| `unicode_rust_short_129b` | Fixed short Rust/Unicode task, 129 bytes | `26d0963398b4ae6f58d541dd2ed10a50cb307757f79da0e4a2132eb5a7883bd8` |

Common request settings were 128 forced output tokens, EOS ignored, thinking
disabled, base seed `20260824`, and per-worker seed `base_seed + worker`.
Greedy used `temperature=0` and `top_p=1`; Sampled used `temperature=0.7`,
`top_p=1`, and the checkpoint default `top_k=20`.

The rotation baseline used one warmup plus five measured batches per cell. The
Pairwise acceptance used one warmup plus three measured batches per cell.

## B2 Kernel Qualification

The microbenchmark compares two serial B1 executions with one B2 execution.
Each result is the median of five measured samples after one warmup. All draft
tokens and target logits matched the corresponding B1 rows exactly.

| Operation | Serial B1 pair | B2 | Speedup |
|---|---:|---:|---:|
| Draft proposal | 19,541 us | 18,102 us | 1.080x |
| Target GreedyVerify | 89,931 us | 84,327 us | 1.066x |
| Target SampledVerify | 111,083 us | 99,292 us | 1.119x |

Strict equivalence required the batch-isolated QMM path. Product-stable QMM
alone still changed one draft proposal because some projection shapes did not
meet its aligned fast-path conditions.

## Request-Rotation Baseline

These measurements predate the production Pairwise actor path; concurrency was
served by rotating independent B1 requests.

| Mode | C1 | C2 | C4 |
|---|---:|---:|---:|
| Greedy aggregate TPS | 47.84 | 39.29 | 33.85 |
| Sampled aggregate TPS | 27.37 | 26.86 | 25.30 |

The Greedy cells completed successfully, but their JSON report was not emitted
because the subsequent Sampled request attempted an unsupported explicit
`top_k` DTO field. Their medians and wall times survive in the execution log.
Sampled was then rerun using the model-default `top_k=20`; its complete JSON
report survives and is represented in `summary.csv`.

## Pairwise Acceptance

### Greedy

| Concurrency | Aggregate TPS median | P05-P95 | Batch wall median |
|---:|---:|---:|---:|
| C1 | 53.14 | 50.02-53.46 | 2.409 s |
| C2 | 47.95 | 47.61-48.44 | 5.339 s |
| C4 | 44.54 | 44.42-45.36 | 11.495 s |

All Greedy workers and all three concurrency cells produced the same 128-token
output SHA-256 record.

### Sampled replay bottleneck and correction

The first Pairwise Sampled run was rejected as a performance acceptance result:
C1/C2/C4 measured 29.68, 18.28, and 20.21 tok/s. Different worker seeds usually
produced different accepted lengths, and the initial implementation preserved
per-seed output by replaying divergent rows with token-at-a-time Q=1 work.

Direct per-row cache restoration removed that replay bottleneck without changing
the fixed-seed output hashes:

| Concurrency | Aggregate TPS median | P05-P95 | Batch wall median |
|---:|---:|---:|---:|
| C1 | 37.64 | 36.52-38.88 | 3.401 s |
| C2 | 36.84 | 35.56-37.01 | 6.949 s |
| C4 | 30.88 | 24.84-33.57 | 16.581 s |

For each worker index, the accepted report and the rejected intermediate report
have identical output hashes. C1, C2, and C4 therefore preserve the same
base-seed-plus-worker mapping after the optimization.

At stage close-out, the Pairwise results were summarized as approximately
`+22% / +32%` for Greedy C2/C4 and `+37% / +22%` for Sampled C2/C4 relative to
the rotation baseline. These are directional historical comparisons: the
baseline used `fixture_readme_519b`, while final Pairwise acceptance used
`unicode_rust_short_129b`. They are not a strict same-prompt A/B claim.

## Runtime Evidence

- Pairwise health reported a 20,914,906,208-byte peak, equal to 19.48 GiB or
  20.91 decimal GB.
- The accepted Sampled report recorded 96 tensor windows, 11 groups created,
  and 84 divergent per-row restores over the full server run.
- No admission-queue-full or memory-budget-exceeded event was reported in the
  retained acceptance snapshots.
- Separate HTTP gates at this revision covered cancellation survival, uneven
  rows, JSON Schema constraints, forced tool calls, and tool-result replay.
  Those functional responses are not represented as TPS rows in this archive.

## Evidence Provenance

Raw reports remain external to this archive because they contain absolute local
paths and complete health snapshots. The sanitized source identities are:

| Source role | SHA-256 |
|---|---|
| Rotation Sampled baseline | `181e3b7c3cb019ff5fe5185d004910e76a3332a811be9d5666d79a2068e6864b` |
| Pairwise Greedy and rejected Sampled run | `361bd7341ac4d9ae35720a3ef8e9933880898d42fc5087c20daf2a85069b84a8` |
| Pairwise accepted Sampled direct-restore run | `418abbbff68370922155a01f9c3b0e4c8f2b0e6c573ec674ec9255e52ded7d5b` |

## Evidence Boundaries

- This is a historical stage record, not a current B=N performance promise.
- Results apply only to the pinned revisions, hardware, models, prompts, and
  request settings above.
- Greedy baseline P05/P95 values and per-worker hashes are unavailable because
  the combined report terminated after those cells and before JSON emission.
- The rejected Sampled rows are retained only to document the replay bottleneck;
  they must not be presented as final Pairwise performance.
- No raw prompt text, generated response body, absolute local path, host serial,
  or device identifier is included in this candidate archive.

Detailed sanitized rows are available in [`summary.csv`](summary.csv).
