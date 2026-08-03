#!/usr/bin/env bash
# Intentionally refresh tracked third-party materials after dependency review.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

command -v rsync >/dev/null || {
  echo "error: required update tool is missing: rsync" >&2
  exit 1
}

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-third-party-update.XXXXXX")"
trap 'rm -rf "$temp_root"' EXIT

"$SCRIPT_DIR/generate-third-party-materials.sh" "$temp_root"
cp "$temp_root/THIRD_PARTY_NOTICES.md" "$REPO_ROOT/THIRD_PARTY_NOTICES.md"
cp "$temp_root/third-party-inventory.json" "$REPO_ROOT/third-party-inventory.json"
mkdir -p "$REPO_ROOT/THIRD_PARTY_LICENSES"
rsync --archive --delete \
  "$temp_root/THIRD_PARTY_LICENSES/" \
  "$REPO_ROOT/THIRD_PARTY_LICENSES/"

echo "Updated tracked third-party materials; review the complete Git diff before committing"
