#!/usr/bin/env bash
# Build one self-contained, ad-hoc-signed IronMLX.app for the v0.1 platform:
# Apple Silicon arm64 on macOS 26.2 or newer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"
readonly APP_SOURCE_DIR="$REPO_ROOT/ironmlx-app"
readonly PACKAGING_DIR="$APP_SOURCE_DIR/Packaging"
readonly BUILD_ROOT="$REPO_ROOT/.build/p0-1-app-bundle"
readonly DIST_DIR="$REPO_ROOT/dist"
readonly APP_BUNDLE="$DIST_DIR/IronMLX.app"
readonly DEPLOYMENT_TARGET="26.2"
readonly ARCHITECTURE="arm64"
readonly BUILDER_HOME="${HOME:?HOME must be set}"
readonly C_PATH_REMAP_FLAGS="-ffile-prefix-map=$BUILDER_HOME=. -fdebug-prefix-map=$BUILDER_HOME=."
readonly RUST_PATH_REMAP_FLAG="--remap-path-prefix=$BUILDER_HOME=."

MLX_SOURCE="${MLX_SRC:-$REPO_ROOT/../iron-rivals/mlx}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

fail() {
  echo "error: $*" >&2
  exit 1
}

for tool in cargo cmake codesign git iconutil lipo plutil sips swift xcrun; do
  command -v "$tool" >/dev/null || fail "required build tool is missing: $tool"
done

[ "$(uname -s)" = "Darwin" ] || fail "IronMLX.app can only be built on macOS"
[ "$(uname -m)" = "$ARCHITECTURE" ] || fail "IronMLX.app requires an Apple Silicon build host"
[ -d "$MLX_SOURCE/.git" ] || git -C "$MLX_SOURCE" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "MLX_SRC is not a Git checkout: $MLX_SOURCE"

mlx_commit="$(git -C "$MLX_SOURCE" rev-parse HEAD)"
[ "$mlx_commit" = "$IRONMLX_MLX_COMMIT" ] || \
  fail "MLX checkout must be pinned to $IRONMLX_MLX_COMMIT, found $mlx_commit"
[ -z "$(git -C "$MLX_SOURCE" status --porcelain=v1 --untracked-files=normal)" ] || \
  fail "MLX checkout must be clean: $MLX_SOURCE"
mlx_branch="$(git -C "$MLX_SOURCE" branch --show-current)"
case "$mlx_branch" in
  diagnose/m1-metal-stall|diagnose/m1-metal-stall-mlx)
    fail "diagnostic MLX branches are excluded from P0-1 builds: $mlx_branch"
    ;;
esac

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT/mlx-build" "$BUILD_ROOT/mlx-install" "$DIST_DIR"

echo "==> Build MLX $IRONMLX_MLX_COMMIT (Release, arm64, macOS $DEPLOYMENT_TARGET)"
cmake -S "$MLX_SOURCE" -B "$BUILD_ROOT/mlx-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_METAL_JIT=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_METAL_PATH=. \
  -DCMAKE_C_FLAGS="$C_PATH_REMAP_FLAGS" \
  -DCMAKE_CXX_FLAGS="$C_PATH_REMAP_FLAGS" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCHITECTURE" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/mlx-install"
cmake --build "$BUILD_ROOT/mlx-build" --parallel "$JOBS"

# MLX's metallib install rule interprets a relative MLX_METAL_PATH against the
# source directory even though the custom build command emits it in the binary
# directory. Install all other CMake components normally, then stage the exact
# metallib produced by this isolated build. Keeping MLX_METAL_PATH relative is
# what prevents a builder-specific absolute path from entering libmlx.a.
cmake --install "$BUILD_ROOT/mlx-build" --component Unspecified
cmake --install "$BUILD_ROOT/mlx-build" --component headers
cmake --install "$BUILD_ROOT/mlx-build" --component metal_cpp_source

gguf_library="$BUILD_ROOT/mlx-build/mlx/io/libgguflib.a"
[ -f "$gguf_library" ] || fail "MLX build did not produce libgguflib.a"
cp "$gguf_library" "$BUILD_ROOT/mlx-install/lib/libgguflib.a"
metallib_build="$BUILD_ROOT/mlx-build/mlx/backend/metal/kernels/mlx.metallib"
[ -f "$metallib_build" ] || fail "MLX build did not produce mlx.metallib"
cp "$metallib_build" "$BUILD_ROOT/mlx-install/lib/mlx.metallib"
for artifact in include/mlx/array.h lib/libmlx.a lib/libgguflib.a lib/mlx.metallib; do
  [ -f "$BUILD_ROOT/mlx-install/$artifact" ] || fail "MLX install is incomplete: $artifact"
done
mlx_flags_file="$BUILD_ROOT/mlx-build/CMakeFiles/mlx.dir/flags.make"
[ -f "$mlx_flags_file" ] || fail "MLX CMake flags file is missing: $mlx_flags_file"
if grep -Fq 'MLX_METAL_NO_NAX' "$mlx_flags_file"; then
  fail "MLX was compiled without required NAX Metal kernels"
fi

echo "==> Build Rust helpers (Release, isolated target directory)"
env \
  MLX_DIR="$BUILD_ROOT/mlx-install" \
  MACOSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  CARGO_TARGET_DIR="$BUILD_ROOT/cargo-target" \
  CFLAGS="$C_PATH_REMAP_FLAGS" \
  CXXFLAGS="$C_PATH_REMAP_FLAGS" \
  RUSTFLAGS="$RUST_PATH_REMAP_FLAG" \
  cargo build --manifest-path "$REPO_ROOT/Cargo.toml" --locked --release --bin ironmlx --bin iron-bench

echo "==> Build Swift App executable (Release, App-Bundle resource mode)"
env MACOSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  swift build \
    --package-path "$APP_SOURCE_DIR" \
    --configuration release \
    --scratch-path "$BUILD_ROOT/swift-build" \
    --product ironmlx-app \
    -Xswiftc=-DIRONMLX_APP_BUNDLE \
    -Xswiftc=-gnone

echo "==> Assemble IronMLX.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Helpers" "$APP_BUNDLE/Contents/Resources"
cp "$PACKAGING_DIR/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
cp "$BUILD_ROOT/swift-build/release/ironmlx-app" "$APP_BUNDLE/Contents/MacOS/IronMLX"
cp "$BUILD_ROOT/cargo-target/release/ironmlx" "$APP_BUNDLE/Contents/Helpers/ironmlx"
cp "$BUILD_ROOT/cargo-target/release/iron-bench" "$APP_BUNDLE/Contents/Helpers/iron-bench"
cp "$BUILD_ROOT/mlx-install/lib/mlx.metallib" "$APP_BUNDLE/Contents/Resources/mlx.metallib"
for resource in dashboard2.html logo.png menubar-icon.png menubar-icon@2x.png sidebar-logo@2x.png; do
  cp "$APP_SOURCE_DIR/Sources/IronMLXAppCore/Resources/$resource" "$APP_BUNDLE/Contents/Resources/$resource"
done

iconset="$BUILD_ROOT/AppIcon.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
  double_size="$((size * 2))"
  sips -z "$size" "$size" "$PACKAGING_DIR/AppIcon-1024.png" \
    --out "$iconset/icon_${size}x${size}.png" >/dev/null
  sips -z "$double_size" "$double_size" "$PACKAGING_DIR/AppIcon-1024.png" \
    --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"

chmod 0755 \
  "$APP_BUNDLE/Contents/MacOS/IronMLX" \
  "$APP_BUNDLE/Contents/Helpers/ironmlx" \
  "$APP_BUNDLE/Contents/Helpers/iron-bench"

echo "==> Ad-hoc sign assembled bundle"
codesign --force --deep --sign - "$APP_BUNDLE"

"$SCRIPT_DIR/verify-app-bundle.sh" "$APP_BUNDLE"
echo "Built: $APP_BUNDLE"
