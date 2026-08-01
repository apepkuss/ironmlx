#!/usr/bin/env bash
# Verify that all executable v0.1 release artifacts share the declared
# Apple Silicon + macOS 26.2 platform boundary.

set -euo pipefail

readonly EXPECTED_ARCHITECTURE="arm64"
readonly EXPECTED_MACOS_VERSION="26.2"

HELPER_BINARY="${1:-target/release/ironmlx}"
APP_BINARY="${2:-ironmlx-app/.build/release/ironmlx-app}"
MIGRATOR_BINARY="${APP_BINARY%/*}/ironmlx-model-migrate"
METALLIB="${3:-${MLX_DIR:+$MLX_DIR/lib/mlx.metallib}}"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || fail "required release artifact is missing: $1"
}

verify_macho() {
  local label="$1"
  local binary="$2"
  local architectures
  local minimum_version

  require_file "$binary"
  architectures="$(lipo -archs "$binary")"
  [ "$architectures" = "$EXPECTED_ARCHITECTURE" ] || \
    fail "$label must contain only $EXPECTED_ARCHITECTURE, found: $architectures"

  minimum_version="$(otool -l "$binary" | awk '
    /LC_BUILD_VERSION/ { in_build_version = 1; next }
    in_build_version && $1 == "minos" { print $2; exit }
  ')"
  [ "$minimum_version" = "$EXPECTED_MACOS_VERSION" ] || \
    fail "$label minimum macOS must be $EXPECTED_MACOS_VERSION, found: ${minimum_version:-missing}"

  echo "ok: $label is $EXPECTED_ARCHITECTURE with macOS $EXPECTED_MACOS_VERSION minimum"
}

grep -Fq '.macOS("26.2")' ironmlx-app/Package.swift || \
  fail "Swift package does not declare macOS 26.2"
[ -n "$METALLIB" ] || \
  fail "pass the metallib path as argument 3 or set MLX_DIR"

verify_macho "IronMLX backend helper" "$HELPER_BINARY"
verify_macho "IronMLX App executable" "$APP_BINARY"
verify_macho "IronMLX model migrator" "$MIGRATOR_BINARY"

require_file "$METALLIB"
METALLIB_TYPE="$(file "$METALLIB")"
[[ "$METALLIB_TYPE" == *"MetalLib executable (MacOS)"* ]] || \
  fail "MLX metallib is not a macOS MetalLib executable: $METALLIB"
LC_ALL=C grep -aEq 'air64_v[0-9]+-apple-macosx26\.2\.0' "$METALLIB" || \
  fail "MLX metallib does not contain a macOS 26.2 AIR target"
echo "ok: MLX metallib targets macOS $EXPECTED_MACOS_VERSION"

echo "IronMLX v0.1 release platform verification passed"
