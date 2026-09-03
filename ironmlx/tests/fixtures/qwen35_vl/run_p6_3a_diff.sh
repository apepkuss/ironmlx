#!/usr/bin/env bash
# P6.3a: drive Gate 1 measurement (preprocess byte diff).
# Required env: MLX_DIR, QWEN35_MODEL (for mlx-vlm side dump only).
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/qwen35_vl"
PY_DIR="${PY_DIR:-/tmp/p6_diff/python}"
IRON_PRE_DIR="${IRON_PRE_DIR:-/tmp/p6_diff/ironmlx_pre}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_3a-$STAMP"

mkdir -p "$PY_DIR" "$IRON_PRE_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$IRON_PRE_DIR"/*.safetensors

echo "=== Step 1: mlx-vlm pixel_values dump ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"

echo "=== Step 2: ironmlx preprocess dump ==="
cd "$REPO_ROOT"
MLX_DIR="$MLX_DIR" \
    IMAGE_PATH="$FIXTURE_DIR/coco_sample.jpg" \
    IRONMLX_PREPROCESS_DUMP_DIR="$IRON_PRE_DIR" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test preprocess_dump \
        -- --ignored 2>&1 | tail -5

echo "=== Step 3: preprocess byte diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_preprocess.py" \
    --vlm "$PY_DIR/00_pixel_values.safetensors" \
    --iron "$IRON_PRE_DIR/00_ironmlx_pv_vlmlayout.safetensors" \
    --out "$REPORT_DIR/p6_3a_preprocess_report.md" || true

echo "=== Report: $REPORT_DIR/p6_3a_preprocess_report.md ==="
