#!/usr/bin/env bash
# P6.6 multi-image diff orchestrator.
#
# Stage 1: mlx-vlm dump (multi-image; op-level hooks active via MLXVLM_VISION_DUMP_DIR)
# Stage 2: ironmlx preprocess dump (twice — once per image)
# Stage 3: ironmlx vision dump on concatenated pixel_values (op-level via existing hooks)
# Stage 4: diff_preprocess_multi (Gate 1)
# Stage 5: diff_pipeline_multi (Gate 2 + op-level)
# Stage 6: p6_6_logits_match (Gate 3, integration test)
#
# Required env: MLX_DIR, QWEN35_MODEL
set -euo pipefail

if [[ -z "${MLX_DIR:-}" || -z "${QWEN35_MODEL:-}" ]]; then
    echo "ERROR: set MLX_DIR and QWEN35_MODEL env vars" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
MULTI_DIR="$FIXTURE_DIR/multi_image"
PY_DIR="${PY_DIR:-/tmp/p6_diff_multi/python}"
RUST_DIR="${RUST_DIR:-/tmp/p6_diff_multi/rust}"
IRON_PRE_DIR="${IRON_PRE_DIR:-/tmp/p6_diff_multi/ironmlx_pre}"
STAMP="$(date +%Y-%m-%d-%H%M)"
REPORT_DIR="$FIXTURE_DIR/diff_reports/p6_6-$STAMP"

mkdir -p "$PY_DIR" "$RUST_DIR" "$IRON_PRE_DIR" "$REPORT_DIR"
rm -f "$PY_DIR"/*.safetensors "$PY_DIR"/*.npy "$PY_DIR"/*.txt
rm -f "$RUST_DIR"/*.safetensors
rm -rf "$IRON_PRE_DIR"
mkdir -p "$IRON_PRE_DIR"

echo "=== Stage 1: mlx-vlm 2-image dump (with op-level hooks) ==="
MLXVLM_VISION_DUMP_DIR="$PY_DIR" \
QWEN35_MODEL="$QWEN35_MODEL" \
    ~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/run_p6_6_dump.py" \
        --image-0 "$MULTI_DIR/image_0.jpg" \
        --image-1 "$MULTI_DIR/image_1.jpg" \
        --out-dir "$PY_DIR" 2>&1 | tail -10
echo "  PY_DIR files: $(ls "$PY_DIR" | wc -l)"

echo "=== Stage 2: ironmlx preprocess dump (per image) ==="
cd "$REPO_ROOT"
for i in 0 1; do
    SUBDIR="$IRON_PRE_DIR/image_${i}"
    mkdir -p "$SUBDIR"
    MLX_DIR="$MLX_DIR" \
        IMAGE_PATH="$MULTI_DIR/image_${i}.jpg" \
        IRONMLX_PREPROCESS_DUMP_DIR="$SUBDIR" \
        cargo test -p ironmlx --features vision-dump --release \
            --test p6_3a_preprocess_dump -- --ignored 2>&1 | tail -3
    mv "$SUBDIR/00_ironmlx_pv_native.safetensors" "$IRON_PRE_DIR/image_${i}_pv_native.safetensors"
    mv "$SUBDIR/00_ironmlx_pv_vlmlayout.safetensors" "$IRON_PRE_DIR/image_${i}_pv_vlmlayout.safetensors"
    rmdir "$SUBDIR"
done
echo "  IRON_PRE files: $(ls "$IRON_PRE_DIR" | wc -l)"

echo "=== Stage 3: ironmlx vision dump on concatenated input ==="
MLX_DIR="$MLX_DIR" \
QWEN35_MODEL="$QWEN35_MODEL" \
IRONMLX_VISION_DUMP_DIR="$RUST_DIR" \
PIXEL_VALUES_PATH="$PY_DIR/expected_pixel_values.safetensors" \
IMAGE_GRID_THW_PATH="$PY_DIR/expected_image_grid_thw.npy" \
    cargo test -p ironmlx --features vision-dump --release \
        --test p6_6_multi_image_dump -- --ignored 2>&1 | tail -5
echo "  RUST_DIR files: $(ls "$RUST_DIR" | wc -l)"

echo "=== Stage 4: Gate 1 — per-image preprocess diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_preprocess_multi.py" \
    --py "$PY_DIR" --iron "$IRON_PRE_DIR" \
    --out "$REPORT_DIR/p6_6_preprocess_report.md" \
    --gate 0.05 || true

echo "=== Stage 5: Gate 2 — vision encoder diff ==="
~/.venvs/mlxvlm-ref/bin/python "$FIXTURE_DIR/diff_pipeline_multi.py" \
    --py "$PY_DIR" --rust "$RUST_DIR" \
    --out "$REPORT_DIR/vision" \
    --gate2 0.1 || true

echo "=== Stage 6: Gate 3 — e2e logits-match integration test ==="
ln -sf "$PY_DIR/expected_input_ids.npy" "$MULTI_DIR/expected_input_ids.npy" || true
ln -sf "$PY_DIR/expected_image_grid_thw.npy" "$MULTI_DIR/expected_image_grid_thw.npy" || true
ln -sf "$PY_DIR/expected_pixel_values.safetensors" "$MULTI_DIR/expected_pixel_values.safetensors" || true
ln -sf "$PY_DIR/expected_last_logits.npy" "$MULTI_DIR/expected_last_logits.npy" || true
ln -sf "$PY_DIR/expected_first_token.txt" "$MULTI_DIR/expected_first_token.txt" || true

QWEN35_MODEL="$QWEN35_MODEL" \
    MLX_DIR="$MLX_DIR" \
    cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --nocapture 2>&1 \
    | tee "$REPORT_DIR/p6_6_logits_match.log" | tail -15

echo "=== Done. Reports in: $REPORT_DIR ==="
