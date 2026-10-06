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
command -v python3 >/dev/null || fail "required tool is missing: python3"

mkdir -p "$destination"
git -C "$destination" init --quiet
git -C "$destination" remote add origin "$IRONMLX_MLX_REPOSITORY"
git -C "$destination" fetch --quiet --depth=1 origin "$IRONMLX_MLX_COMMIT"
git -C "$destination" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD

# Vendored headers can come from a different upstream revision than the build.
# Fetch only the declared verification commits; keep HEAD pinned to the build.
python3 - "$SCRIPT_DIR/../compliance/native-dependencies.json" "$destination" <<'PY'
import json
import re
import subprocess
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    dependencies = json.load(handle)["dependencies"]
git = ["git", "-C", sys.argv[2]]
for dependency in dependencies:
    verification = dependency.get("source_verification", {})
    if verification.get("type") != "git-files" or verification.get("repository") != "mlx:.":
        continue
    commit = verification["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"verification commit must be an immutable SHA: {commit}")
    available = subprocess.run(
        [*git, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True
    )
    if available.returncode != 0:
        print(f"Fetching MLX verification commit: {commit}", flush=True)
        subprocess.run(
            [*git, "fetch", "--quiet", "--depth=1", "--no-tags", dependency["repository"], commit],
            check=True,
        )
    subprocess.run([*git, "cat-file", "-e", f"{commit}^{{commit}}"], check=True)
PY

actual_commit="$(git -C "$destination" rev-parse HEAD)"
[ "$actual_commit" = "$IRONMLX_MLX_COMMIT" ] || \
  fail "MLX checkout mismatch: expected $IRONMLX_MLX_COMMIT, found $actual_commit"
[ -z "$(git -C "$destination" status --porcelain=v1 --untracked-files=normal)" ] || \
  fail "MLX checkout is not clean: $destination"

echo "MLX checkout ready: $actual_commit"
