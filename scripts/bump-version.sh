#!/usr/bin/env bash
# Update all authoritative IronMLX version surfaces, then verify consistency.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly VERSION_FILE="$REPO_ROOT/VERSION"
readonly INFO_PLIST="$REPO_ROOT/ironmlx-app/Packaging/Info.plist"

new_version="${1:-}"
requested_build="${2:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "usage: scripts/bump-version.sh X.Y.Z [positive-build-number]" >&2
  fail "version must be a stable semantic version"
}
if [ -n "$requested_build" ]; then
  [[ "$requested_build" =~ ^[1-9][0-9]*$ ]] || fail "build number must be a positive integer"
fi

for tool in cargo perl plutil; do
  command -v "$tool" >/dev/null || fail "required tool is missing: $tool"
done

current_version="$(tr -d '[:space:]' < "$VERSION_FILE")"
current_build="$(plutil -extract CFBundleVersion raw "$INFO_PLIST")"
[[ "$current_build" =~ ^[1-9][0-9]*$ ]] || fail "current App build number is invalid"

if [ -n "$requested_build" ]; then
  new_build="$requested_build"
elif [ "$new_version" = "$current_version" ]; then
  new_build="$current_build"
else
  new_build="$((current_build + 1))"
fi

printf '%s\n' "$new_version" > "$VERSION_FILE"
IRONMLX_NEW_VERSION="$new_version" perl -0pi -e \
  's/(\[workspace\.package\]\s+version = ")[^"]+("\s+)/$1$ENV{IRONMLX_NEW_VERSION}$2/' \
  "$REPO_ROOT/Cargo.toml"
IRONMLX_NEW_VERSION="$new_version" perl -0pi -e \
  's/(mlx-sys = \{ path = "\.\.\/mlx-sys", version = ")[^"]+(" \})/$1$ENV{IRONMLX_NEW_VERSION}$2/' \
  "$REPO_ROOT/mlx/Cargo.toml"
IRONMLX_NEW_VERSION="$new_version" perl -0pi -e \
  's/(<key>CFBundleShortVersionString<\/key>\s*<string>)[^<]+(<\/string>)/$1$ENV{IRONMLX_NEW_VERSION}$2/' \
  "$INFO_PLIST"
IRONMLX_NEW_BUILD="$new_build" perl -0pi -e \
  's/(<key>CFBundleVersion<\/key>\s*<string>)[^<]+(<\/string>)/$1$ENV{IRONMLX_NEW_BUILD}$2/' \
  "$INFO_PLIST"

cargo update \
  --manifest-path "$REPO_ROOT/Cargo.toml" \
  -p ironmlx \
  --precise "$new_version"
"$SCRIPT_DIR/verify-version-consistency.sh"

echo "IronMLX version updated: $current_version ($current_build) -> $new_version ($new_build)"
