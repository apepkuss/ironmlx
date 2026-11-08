#!/usr/bin/env bash
# Build and install the release-pinned MLX static library used by IronMLX.
#
# Usage: scripts/build-mlx.sh <mlx-source> <build-root>
#
# Produces <build-root>/mlx-build and <build-root>/mlx-install. A build is
# always made from scratch unless IRONMLX_REUSE_MLX_BUILD=1 and the stamp of an
# existing build matches every input below (MLX commit, CMake configuration and
# the compiler, Metal and SDK versions). CI uses reuse with a restored cache;
# release builds never set it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ "$#" -eq 2 ] || fail "usage: scripts/build-mlx.sh <mlx-source> <build-root>"
MLX_SOURCE="$1"
BUILD_ROOT="$2"
readonly BUILDER_HOME="${HOME:?HOME must be set}"
readonly C_PATH_REMAP_FLAGS="-ffile-prefix-map=$BUILDER_HOME=. -fdebug-prefix-map=$BUILDER_HOME=."
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

for tool in cmake git xcrun; do
  command -v "$tool" >/dev/null || fail "required build tool is missing: $tool"
done
[ -d "$MLX_SOURCE/.git" ] || git -C "$MLX_SOURCE" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "MLX source is not a Git checkout: $MLX_SOURCE"
mlx_commit="$(git -C "$MLX_SOURCE" rev-parse HEAD)"
[ "$mlx_commit" = "$IRONMLX_MLX_COMMIT" ] || \
  fail "MLX checkout must be pinned to $IRONMLX_MLX_COMMIT, found $mlx_commit"
[ -z "$(git -C "$MLX_SOURCE" status --porcelain=v1 --untracked-files=normal)" ] || \
  fail "MLX checkout must be clean: $MLX_SOURCE"

mkdir -p "$BUILD_ROOT"
BUILD_ROOT="$(cd "$BUILD_ROOT" && pwd)"
readonly MLX_BUILD="$BUILD_ROOT/mlx-build"
readonly MLX_INSTALL="$BUILD_ROOT/mlx-install"
readonly STAMP_FILE="$BUILD_ROOT/mlx-build.stamp"

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DMLX_BUILD_METAL=ON
  -DMLX_METAL_JIT=OFF
  -DMLX_BUILD_TESTS=OFF
  -DMLX_BUILD_EXAMPLES=OFF
  -DMLX_BUILD_BENCHMARKS=OFF
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
  -DMLX_METAL_PATH=.
  "-DCMAKE_C_FLAGS=$C_PATH_REMAP_FLAGS"
  "-DCMAKE_CXX_FLAGS=$C_PATH_REMAP_FLAGS"
  "-DCMAKE_OSX_ARCHITECTURES=$IRONMLX_ARCHITECTURE"
  "-DCMAKE_OSX_DEPLOYMENT_TARGET=$IRONMLX_MACOS_DEPLOYMENT_TARGET"
  "-DCMAKE_INSTALL_PREFIX=$MLX_INSTALL"
)

# Every input that can change the produced library or metallib. A mismatch on
# any line forces a clean build.
expected_stamp="$(
  printf 'mlx_commit=%s\n' "$mlx_commit"
  printf 'mlx_source=%s\n' "$MLX_SOURCE"
  printf 'cmake_arg=%s\n' "${cmake_args[@]}"
  printf 'cmake=%s\n' "$(cmake --version | head -n 1)"
  printf 'clang=%s\n' "$(xcrun clang --version | head -n 1)"
  printf 'metal=%s\n' "$(xcrun -sdk macosx metal --version 2>&1 | head -n 1)"
  printf 'sdk=%s\n' "$(xcrun --sdk macosx --show-sdk-build-version)"
)"

reused=false
if [ "${IRONMLX_REUSE_MLX_BUILD:-0}" = "1" ] && [ -f "$STAMP_FILE" ] && \
  [ "$(cat "$STAMP_FILE")" = "$expected_stamp" ]; then
  echo "==> Reuse MLX $mlx_commit build with a matching stamp"
  reused=true
else
  rm -rf "$MLX_BUILD" "$MLX_INSTALL" "$STAMP_FILE"
  mkdir -p "$MLX_BUILD" "$MLX_INSTALL"

  echo "==> Build MLX $mlx_commit (Release, $IRONMLX_ARCHITECTURE, macOS $IRONMLX_MACOS_DEPLOYMENT_TARGET)"
  cmake -S "$MLX_SOURCE" -B "$MLX_BUILD" "${cmake_args[@]}"
  cmake --build "$MLX_BUILD" --parallel "$JOBS"

  # MLX's metallib install rule interprets a relative MLX_METAL_PATH against the
  # source directory even though the custom build command emits it in the binary
  # directory. Install all other CMake components normally, then stage the exact
  # metallib produced by this isolated build. Keeping MLX_METAL_PATH relative is
  # what prevents a builder-specific absolute path from entering libmlx.a.
  cmake --install "$MLX_BUILD" --component Unspecified
  cmake --install "$MLX_BUILD" --component headers
  cmake --install "$MLX_BUILD" --component metal_cpp_source

  gguf_library="$MLX_BUILD/mlx/io/libgguflib.a"
  [ -f "$gguf_library" ] || fail "MLX build did not produce libgguflib.a"
  cp "$gguf_library" "$MLX_INSTALL/lib/libgguflib.a"
  metallib_build="$MLX_BUILD/mlx/backend/metal/kernels/mlx.metallib"
  [ -f "$metallib_build" ] || fail "MLX build did not produce mlx.metallib"
  cp "$metallib_build" "$MLX_INSTALL/lib/mlx.metallib"
fi

for artifact in include/mlx/array.h lib/libmlx.a lib/libgguflib.a lib/mlx.metallib; do
  [ -f "$MLX_INSTALL/$artifact" ] || fail "MLX install is incomplete: $artifact"
done
mlx_flags_file="$MLX_BUILD/CMakeFiles/mlx.dir/flags.make"
[ -f "$mlx_flags_file" ] || fail "MLX CMake flags file is missing: $mlx_flags_file"
if grep -Fq 'MLX_METAL_NO_NAX' "$mlx_flags_file"; then
  fail "MLX was compiled without required NAX Metal kernels"
fi

if [ "$reused" = false ]; then
  printf '%s\n' "$expected_stamp" > "$STAMP_FILE"
fi
echo "MLX install ready: $MLX_INSTALL"
