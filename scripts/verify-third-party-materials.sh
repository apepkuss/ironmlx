#!/usr/bin/env bash
# Regenerate and compare tracked third-party materials to detect dependency drift.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() {
  echo "error: $*" >&2
  exit 1
}

for tracked in THIRD_PARTY_NOTICES.md third-party-inventory.json THIRD_PARTY_LICENSES; do
  [ -e "$REPO_ROOT/$tracked" ] || fail "tracked third-party material is missing: $tracked"
done

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-third-party-verify.XXXXXX")"
trap 'rm -rf "$temp_root"' EXIT

"$SCRIPT_DIR/generate-third-party-materials.sh" "$temp_root"

for tracked in THIRD_PARTY_NOTICES.md third-party-inventory.json THIRD_PARTY_LICENSES; do
  diff -ru "$REPO_ROOT/$tracked" "$temp_root/$tracked" || fail \
    "third-party materials have drifted; regenerate them after reviewing the dependency change"
done

echo "Third-party dependency materials are complete and reproducible"
