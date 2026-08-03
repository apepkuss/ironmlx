#!/usr/bin/env bash
# Verify every product version surface and crates.io publication boundary.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly VERSION_FILE="$REPO_ROOT/VERSION"
readonly INFO_PLIST="$REPO_ROOT/ironmlx-app/Packaging/Info.plist"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -f "$VERSION_FILE" ] || fail "canonical VERSION file is missing"
product_version="$(tr -d '[:space:]' < "$VERSION_FILE")"
[[ "$product_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
  fail "VERSION must be a stable semantic version (X.Y.Z): $product_version"

for tool in cargo plutil python3; do
  command -v "$tool" >/dev/null || fail "required tool is missing: $tool"
done

metadata_file="$(mktemp "${TMPDIR:-/tmp}/ironmlx-version-metadata.XXXXXX")"
trap 'rm -f "$metadata_file"' EXIT
cargo metadata \
  --manifest-path "$REPO_ROOT/Cargo.toml" \
  --format-version 1 \
  --no-deps \
  --locked > "$metadata_file"

python3 - "$metadata_file" "$REPO_ROOT/Cargo.lock" "$product_version" <<'PY'
import json
import re
import sys

metadata_path, lock_path, expected = sys.argv[1:]
with open(metadata_path, encoding="utf-8") as handle:
    metadata = json.load(handle)

errors = []
workspace_names = set()
for package in metadata["packages"]:
    workspace_names.add(package["name"])
    if package["version"] != expected:
        errors.append(
            f"workspace package {package['name']} has version "
            f"{package['version']}, expected {expected}"
        )
    if package.get("publish") != []:
        errors.append(
            f"workspace package {package['name']} must declare publish = false"
        )

lock_versions = {}
current = {}
with open(lock_path, encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if line == "[[package]]":
            if current.get("name") in workspace_names and "source" not in current:
                lock_versions[current["name"]] = current.get("version")
            current = {}
            continue
        match = re.fullmatch(r'(name|version|source) = "([^"]+)"', line)
        if match:
            current[match.group(1)] = match.group(2)
if current.get("name") in workspace_names and "source" not in current:
    lock_versions[current["name"]] = current.get("version")
for package_name in sorted(workspace_names):
    if lock_versions.get(package_name) != expected:
        errors.append(
            f"Cargo.lock package {package_name} has version "
            f"{lock_versions.get(package_name, 'missing')}, expected {expected}"
        )

if errors:
    print("\n".join(f"error: {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
PY

grep -Eq "mlx-sys = \{ path = \"\.\./mlx-sys\", version = \"$product_version\" \}" \
  "$REPO_ROOT/mlx/Cargo.toml" || \
  fail "mlx/Cargo.toml mlx-sys dependency version does not match VERSION"

app_version="$(plutil -extract CFBundleShortVersionString raw "$INFO_PLIST")"
[ "$app_version" = "$product_version" ] || \
  fail "App version is $app_version, expected $product_version"
app_build="$(plutil -extract CFBundleVersion raw "$INFO_PLIST")"
[[ "$app_build" =~ ^[1-9][0-9]*$ ]] || fail "CFBundleVersion must be a positive integer"

if [ "${GITHUB_REF_TYPE:-}" = "tag" ]; then
  [ "${GITHUB_REF_NAME:-}" = "v$product_version" ] || \
    fail "release tag must be v$product_version, found ${GITHUB_REF_NAME:-missing}"
fi

echo "IronMLX version consistency passed: $product_version (build $app_build)"
