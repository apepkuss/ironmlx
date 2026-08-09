#!/usr/bin/env bash
# Generate third-party materials for the macOS arm64 Release product.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly MLX_SOURCE="${MLX_SRC:-$REPO_ROOT/../iron-rivals/mlx}"
readonly MLX_BUILD="${MLX_BUILD_DIR:-$REPO_ROOT/.build/app-bundle/mlx-build}"
readonly OUTPUT_ROOT="${1:-$REPO_ROOT}"
readonly CARGO_ABOUT="${CARGO_ABOUT:-cargo-about}"

fail() {
  echo "error: $*" >&2
  exit 1
}

for tool in cargo git python3 swift; do
  command -v "$tool" >/dev/null || fail "required inventory tool is missing: $tool"
done
command -v "$CARGO_ABOUT" >/dev/null || fail \
  "cargo-about 0.9.1 is required; install it with: cargo install --locked --features cli --version 0.9.1 cargo-about"
[ -d "$MLX_SOURCE" ] || fail "MLX source directory is missing: $MLX_SOURCE"
[ -d "$MLX_BUILD" ] || fail "MLX build directory is missing: $MLX_BUILD"
[ ! -e "$OUTPUT_ROOT/THIRD_PARTY_LICENSES" ] || fail \
  "output license directory already exists: $OUTPUT_ROOT/THIRD_PARTY_LICENSES"

about_version="$($CARGO_ABOUT --version)"
[ "$about_version" = "cargo-about 0.9.1" ] || fail \
  "cargo-about version must be 0.9.1, found: $about_version"

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-third-party-generate.XXXXXX")"
trap 'rm -rf "$temp_root"' EXIT

for package in ironmlx iron-bench; do
  # cargo-about loads the complete locked dependency graph before applying the
  # target filter from about.toml, so its offline metadata pass needs every
  # locked package source available locally.
  cargo fetch \
    --locked \
    --manifest-path "$REPO_ROOT/$package/Cargo.toml"
  "$CARGO_ABOUT" generate \
    --format json \
    --locked \
    --offline \
    --config "$REPO_ROOT/about.toml" \
    --manifest-path "$REPO_ROOT/$package/Cargo.toml" \
    --output-file "$temp_root/cargo-about-$package.json"
done

swift package --package-path "$REPO_ROOT/ironmlx-app" dump-package \
  > "$temp_root/swift-package.json"

python3 "$SCRIPT_DIR/generate-third-party-materials.py" \
  --cargo-about-json "$temp_root/cargo-about-ironmlx.json" \
  --cargo-about-json "$temp_root/cargo-about-iron-bench.json" \
  --native-manifest "$REPO_ROOT/compliance/native-dependencies.json" \
  --mlx-source "$MLX_SOURCE" \
  --mlx-build "$MLX_BUILD" \
  --swift-manifest "$REPO_ROOT/ironmlx-app/Package.swift" \
  --swift-package-json "$temp_root/swift-package.json" \
  --output-root "$OUTPUT_ROOT"

echo "Generated third-party materials in: $OUTPUT_ROOT"
