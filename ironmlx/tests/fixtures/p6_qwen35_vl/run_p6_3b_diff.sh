#!/usr/bin/env bash
# P6.3b: drive the full op-level diff (30 module-level + 96 intra-block = 126 tensors).
# Same pipeline as P6.1's run_p6_1_diff.sh, but report dir uses a p6_3b- prefix.
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
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_3b-$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$RUST_DIR"/*.safetensors

echo "=== P6.3b: 126-tensor op-level diff ==="
echo "=== Step 1: mlx-vlm op-level dump (30 + 96 = 126 tensors) ==="
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_python_dump.py" \
    --image "$FIXTURE_DIR/coco_sample.jpg" \
    --out-dir "$PY_DIR"
echo "  Files in $PY_DIR: $(ls "$PY_DIR"/*.safetensors | wc -l)"

echo "=== Step 2: ironmlx op-level dump (29 + 96 = 125 tensors) ==="
cd "$REPO_ROOT"
QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
    PIXEL_VALUES_PATH="$PY_DIR/00_pixel_values.safetensors" \
    cargo test -p ironmlx \
        --features vision-dump \
        --release \
        --test p6_vision_dump \
        -- --ignored 2>&1 | tail -5
echo "  Files in $RUST_DIR: $(ls "$RUST_DIR"/*.safetensors | wc -l)"

echo "=== Step 3: diff + report ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline.py" \
    --py "$PY_DIR" \
    --rust "$RUST_DIR" \
    --out "$REPORT_DIR"

echo "=== Done. Report: $REPORT_DIR/report.md ==="
