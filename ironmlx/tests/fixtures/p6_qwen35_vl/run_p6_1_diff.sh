#!/usr/bin/env bash
# P6.1 end-to-end diff pipeline orchestrator.
#
# Required env:
#   MLX_DIR        — mlx C++ install (e.g. $HOME/.local/mlx)
#   QWEN35_MODEL   — local snapshot path of Qwen3.5-4B-MLX-4bit
#
# Optional env:
#   PY_DIR=/tmp/p6_diff/python   — where mlx-vlm dumps land
#   RUST_DIR=/tmp/p6_diff/rust   — where ironmlx dumps land
#
# Produces:
#   ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/<YYYY-MM-DD-HHMM>/
#       report.md + max_diff_curve.png + outliers.json
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
PY_DIR="${PY_DIR:-/tmp/p6_diff/python}"
RUST_DIR="${RUST_DIR:-/tmp/p6_diff/rust}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$REPORT_DIR"

# Clean stale dumps from prior runs
rm -f "$PY_DIR"/*.safetensors "$RUST_DIR"/*.safetensors

echo "=== Step 1: mlx-vlm dump ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"

echo "=== Step 2: ironmlx dump ==="
cd "$REPO_ROOT"
QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
    PIXEL_VALUES_PATH="$PY_DIR/00_pixel_values.safetensors" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test p6_vision_dump \
        -- --ignored 2>&1 | tail -10

echo "=== Step 3: diff + report ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline.py" \
    --py "$PY_DIR" \
    --rust "$RUST_DIR" \
    --out "$REPORT_DIR"

echo "=== Done. Report: $REPORT_DIR/report.md ==="
