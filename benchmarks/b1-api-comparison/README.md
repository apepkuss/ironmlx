# B1 API application comparison

This directory is the self-contained, versioned harness and documentation for
the B1 API comparison of IronMLX, oMLX, Splash, and TensorFold.

## Contents

- `docs/`: frozen protocol, performance report, output-quality review, and
  experiment history.
- `fixtures/`: the fixed six-prompt test set and sanitized oMLX settings.
- `scripts/`: session runner, streaming HTTP client, analysis and audit tools.
- `results/`: ignored local run outputs and raw evidence. Never publish this
  directory without reviewing it; runtime settings and logs can contain local
  paths or credentials.

The completed B1 API result and its limits are in
[`docs/b1-api-performance-report.md`](docs/b1-api-performance-report.md).
DSH Desktop E2E and the deferred stability qualification are not part of that
acceptance and remain future work.

## Reproduce on the original test setup

The reported run used an M5 Max Mac, Release IronMLX, and locally installed
artifacts for the four applications, MLX, Qwen3.8-27B-4bit, and its DFlash2
drafter. Each path may be set explicitly; defaults retain the original host's
layout for convenience:

```sh
export B1_ARTIFACTS_ROOT=/path/to/b1-rival-benchmark/artifacts
export B1_TARGET_MODEL=/path/to/Qwen3.8-27B-4bit/snapshot
export B1_DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2/snapshot
export B1_MLX_LIB=/path/to/mlx/lib
export B1_OMLX_MODEL_DIR=/path/to/omlx/models
export B1_SPLASH_PREPARED='/path/to/Splash/models/.resolved/<prepared-id>'
```

The artifact root must contain the recorded application layout (`tensorfold`,
`omlx`, and `splash`); TensorFold's environment supplies the Python runtime used
by the runner and tokenizer client. Build the IronMLX Release server with the
same MLX installation identified by `B1_MLX_LIB` before measuring. Do not build,
compile, or run other inference workloads during measured sessions.

From the repository root, first check the runner options and confirm port 18480
is free. Then run a fresh label; measured and failed outputs are never
overwritten:

```sh
"$B1_ARTIFACTS_ROOT/tensorfold/venv/bin/python" \
  benchmarks/b1-api-comparison/scripts/run_b1_api_sessions.py \
  --label my-reproduction --sessions 8
```

Outputs go to `benchmarks/b1-api-comparison/results/my-reproduction/`. Analyze
them without replacing the raw session files:

```sh
"$B1_ARTIFACTS_ROOT/tensorfold/venv/bin/python" \
  benchmarks/b1-api-comparison/scripts/analyze_b1_api.py \
  benchmarks/b1-api-comparison/results/my-reproduction \
  --label my-reproduction \
  --output benchmarks/b1-api-comparison/results/my-reproduction/analysis.json
```

The historical final candidate used explicit experimental IronMLX settings and
15 tree nodes. For an exact protocol reproduction, use the candidate command
in [`docs/b1-api-performance-protocol.md`](docs/b1-api-performance-protocol.md)
and `--ironmlx-tree-nodes 15`. Review output quality and run the offline evidence
audits before making acceptance claims. See the protocol for workload,
warm-up, ordering, statistics, and validity rules.

The tracked report, scripts, and fixtures are intended to make the procedure
auditable. The ignored local archive at
`results/local-archive-2026-09-30/` preserves the original experiment logs and
run records on the test machine; it is intentionally not part of commits due
to size and sensitive runtime details. The report contains the accepted
aggregate results and explicitly documents this evidence boundary.
