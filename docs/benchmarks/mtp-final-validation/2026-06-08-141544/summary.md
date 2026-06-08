# MTP Final Branch and Validation Record

- Date: 2026-06-08
- Host: Apple M5 Max / 128GB, macOS 26.4, arm64
- Worktree: `/Users/xin/workspace/ironmlx-backend-mtp-phase5.1-server-api-validation`
- Current branch: `codex/mtp-phase5.1-server-api-validation`
- Tracking branch: `origin/codex/mtp-phase5.1-server-api-validation`
- Head before this record: `41b7724d7f517c287a4b959671b6689cdc2e449a`
- Mainline base: `codex/scheduler-autotune-v2` at `f394ea950ad562756275de9d614b5343685ca2dc`
- Scope: branch organization and final validation record only; no Pull Request was created.

## Branch Organization

`git fetch --prune origin` completed with no stale MTP remote refs reported. All local `codex/mtp*` branches listed below have matching remote heads after the prune/fetch pass.

| Branch | Head | Role | Integration status |
|---|---|---|---|
| `codex/mtp-rollback-phase1` | `072d010` | Phase 1 rollback primitives and MTP foundation | Ancestor of final integration path |
| `codex/mtp-phase2-actor-gating` | `e98cb9c` | Scheduler actor MTP wiring and gating | Ancestor of final integration path |
| `codex/mtp-phase2-performance-baseline` | `482a6d2` | Phase 2 benchmark artifact branch | Preserved as benchmark evidence |
| `codex/mtp-phase2-llamacpp-parity` | `0c99b6b` | llama.cpp parity cache-commit semantics | Ancestor of final integration path |
| `codex/mtp-phase3-performance` | `82982a6` | MTP speculative performance optimization | Ancestor of final integration path |
| `codex/mtp-phase4-policy` | `8120223` | Model-aware draft token policy | Ancestor of final integration path |
| `codex/mtp-phase4_10-omlx-local` | `3c2a81d` | Local omlx comparison benchmark | Preserved as benchmark evidence |
| `codex/mtp-phase5-server-api` | `c8ebe43` | Server `/healthz` MTP diagnostics | Ancestor of final validation branch |
| `codex/mtp-phase5.1-server-api-validation` | `41b7724` | Server/API validation artifacts | Final aggregation and validation branch |

No local or remote MTP branches were deleted in this pass. Benchmark-only branches remain available because they carry measurement artifacts that are useful for later review, but they are not part of the recommended merge path.

Recommended merge candidate, if integration is desired later: `codex/mtp-phase5.1-server-api-validation`.

## Current Support Boundary

The final MTP branch stack supports startup-level MTP configuration for CLI/server flows:

- CLI/server startup accepts MTP model configuration through the existing MTP path.
- Server `/healthz` exposes MTP enabled state, configured draft token count, and live scheduler MTP counters.
- OpenAI-compatible non-streaming and streaming requests were validated against server-level MTP.
- Anthropic-compatible non-streaming and streaming requests were validated against server-level MTP.
- Non-greedy request fallback remains on the regular scheduler path while server-level MTP stays enabled.

The API does not expose per-request OpenAI or Anthropic MTP parameters. That omission is intentional for this branch stack.

## Runtime Validation References

The server/API validation artifacts are preserved at:

- `docs/benchmarks/mtp-phase5.1-server-api-validation/2026-06-08-134532/summary.md`
- `docs/benchmarks/mtp-phase5.1-server-api-validation/2026-06-08-134532/summary.csv`
- `docs/benchmarks/mtp-phase5.1-server-api-validation/2026-06-08-134532/anthropic_stream_summary.csv`

The Phase 5.1 API validation covered:

- `/healthz`
- OpenAI non-streaming greedy request
- OpenAI non-streaming non-greedy fallback request
- OpenAI streaming smoke/perf sample
- Anthropic non-streaming greedy request
- Anthropic streaming SSE smoke

## Final Verification

Commands were run from `/Users/xin/workspace/ironmlx-backend-mtp-phase5.1-server-api-validation` with `MLX_DIR=$HOME/.local/mlx` where applicable.

| Command | Result |
|---|---|
| `cargo fmt` | exit 0 |
| `cargo +nightly fmt --all -- --check` | exit 0 |
| `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings` | exit 0, finished dev profile |
| `MLX_DIR=$HOME/.local/mlx cargo build --release` | exit 0, finished release profile |
| `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx snapshot_mtp -- --nocapture` | exit 0, 2 passed |
| `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx health_collector_mtp -- --nocapture` | exit 0, 2 passed |

The MLX C++ shim emitted third-party/header warnings during build and test compilation. They did not fail the Rust `clippy -D warnings` gate or the release build.

## Final Handling

- No PR was created.
- No branch was deleted.
- The final validation record belongs to `origin/codex/mtp-phase5.1-server-api-validation`.
