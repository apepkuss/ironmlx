#!/usr/bin/env bash
# Create a clean detached MLX checkout at the release-pinned immutable commit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

destination="${1:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -n "$destination" ] || fail "usage: scripts/checkout-release-mlx.sh <empty-destination>"
[ ! -e "$destination" ] || fail "destination already exists: $destination"
command -v git >/dev/null || fail "required tool is missing: git"

mkdir -p "$destination"
git -C "$destination" init --quiet
git -C "$destination" remote add origin "$IRONMLX_MLX_REPOSITORY"
git -C "$destination" fetch --quiet --depth=1 origin "$IRONMLX_MLX_COMMIT"
git -C "$destination" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD

actual_commit="$(git -C "$destination" rev-parse HEAD)"
[ "$actual_commit" = "$IRONMLX_MLX_COMMIT" ] || \
  fail "MLX checkout mismatch: expected $IRONMLX_MLX_COMMIT, found $actual_commit"
[ -z "$(git -C "$destination" status --porcelain=v1 --untracked-files=normal)" ] || \
  fail "MLX checkout is not clean: $destination"

echo "MLX checkout ready: $actual_commit"
