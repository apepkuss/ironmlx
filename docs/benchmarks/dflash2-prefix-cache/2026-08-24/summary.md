# DFlash2 Cross-Request Prefix Cache Performance Archive

Date: 2026-08-24

Status: passed. The DFlash2-specific in-memory prefix artifact materially reduces
repeated long-prefix latency without changing the generated output contract.

## Baseline

| Component | Revision or configuration |
|---|---|
| IronMLX archive tree | `feat/dflash2-execution-path@9bbee41061afe611a102deedd6ec388d4990d302` |
| Prefix-cache feature | `09639c4e30c35066667520966953005876155393` |
| MLX | local NAX/QMM baseline `73ad5df20cb30be4192e5c4d0ae8130674773427` |
| macOS | `26.4` (`25E246`) |
| Hardware | MacBook Pro `Mac17,6`, Apple M5 Max, 18 CPU cores, 128 GB unified memory |
| Target | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| DFlash2 runtime | block size 4, draft 4-bit |
| Prefix cache | DFlash2 in-memory LRU, 8 GiB configured maximum in the App run |

The feature commit and archive-tree commit delimit the implementation range. The
measurements were collected during that feature-branch interval and were not
re-run after the later tensor-batching changes. They remain a prefix-cache A/B
record, not a performance claim for unrelated later scheduling changes.

This archive extends the earlier [P3 final validation](../../dflash2-final-validation/2026-08-23/summary.md).
It does not rewrite or replace that commit-pinned baseline.

## Measurement Contracts

Two complementary contracts are retained because generation TPS alone does not
measure the main prefix-cache benefit.

### Product agent session

- OMP repeatedly sent a large agent request containing a common system prompt
  and tool-schema prefix.
- `Total wall time` includes request preparation, prefill, generation and
  response assembly.
- `Generation tok/s` is the DFlash2 generation-only health metric and therefore
  intentionally excludes most prefix-cache savings.
- The archive preserves only the accepted session summary. Raw prompts,
  response bodies and absolute local paths are excluded.

| Scenario | Total wall time median | Generation tok/s | Interpretation |
|---|---:|---:|---|
| Cache disabled | 29.52 s | 27.31 | Repeated full prefill control |
| Cache enabled, cold | 27.39 s | 26.69 | Artifact construction and publication |
| Cache enabled, hit | 1.83 s | 28.33 | Target KV and retained target-layer state restored |

The no-cache/cache-hit wall-time ratio is 16.13x. The similar generation TPS
confirms that the gain is prefill avoidance rather than faster token generation.

### Exact long-prefix control timing

- OpenAI Responses request with 21,674 input tokens and one forced output token.
- Cache-disabled control: three independent runs.
- Cache-enabled run: one cold request followed by two exact hits.
- The one-token output intentionally isolates prefix preparation and restore;
  generation TPS is not meaningful for this contract.

| Scenario | Samples | Wall median | Observed range |
|---|---:|---:|---:|
| Cache disabled | 3 | 33.91 s | 30.39-35.34 s |
| Cache enabled, cold | 1 | 31.11 s | 31.11 s |
| Cache enabled, hit | 2 | 2.15 s | 2.14-2.16 s |

This control independently shows a 15.77x median hit/no-cache wall-time ratio.

### Short-prompt generation smoke

A separate cache-enabled short-prompt `iron-bench` run reported 58.63 tok/s,
with a 58.24-58.81 tok/s observed range. Its exact cache-hit state was not
retained in the source artifact, so it is informational only and must not be
compared directly with the OMP long-prefix rows.

## Reproduction Shape

The product A/B used the same target, draft and request payload while toggling
only the DFlash2 prefix LRU. The cache-enabled server used an 8 GiB maximum.
The control repeated one sanitized Responses payload three times and sampled
health after each completion. Model paths, the prompt body and response bodies
are deliberately omitted.

## Correctness and Runtime Boundaries

- Prefix artifacts are DFlash2-specific and include target KV, retained
  target-layer hidden state, position and a runtime fingerprint.
- Cache lookup is exact and fail-closed; a fingerprint or prefix mismatch does
  not reuse an artifact.
- The cache is in-memory only. This record does not qualify paged-KV, SSD
  persistence or Active KV offload for DFlash2.
- Prefix hits improve TTFT and total latency. They do not promise higher
  generation TPS, and a cold-cache run can be slightly slower because it must
  construct and publish the artifact.
- Results apply only to the pinned machine, checkpoints and feature-branch
  interval above.

Detailed sanitized rows are available in [`summary.csv`](summary.csv).
