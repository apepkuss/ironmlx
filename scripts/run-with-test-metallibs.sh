#!/usr/bin/env bash
# Run a command with $MLX_DIR/lib/mlx.metallib copied next to every crate whose
# tests load MLX kernels, and remove the copies afterwards.
#
# Usage: scripts/run-with-test-metallibs.sh <command> [args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
: "${MLX_DIR:?MLX_DIR must point at the MLX install prefix}"
[ "$#" -gt 0 ] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }
test -f "$MLX_DIR/lib/mlx.metallib"

metallib_paths=()
for crate in mlx-sys ironmlx ironmlx-core ironmlx-audio ironmlx-image ironmlx-lm \
  ironmlx-runtime mlx ironmlx-bench-kernel turboquant; do
  metallib_paths+=("$REPO_ROOT/$crate/mlx.metallib")
done
cleanup_metallib() {
  for path in "${metallib_paths[@]}"; do
    rm -f "$path"
  done
}
trap cleanup_metallib EXIT
for path in "${metallib_paths[@]}"; do
  cp "$MLX_DIR/lib/mlx.metallib" "$path"
done

"$@"
