#!/usr/bin/env bash
# Build one self-contained, ad-hoc-signed IronMLX.app for the v0.1 platform:
# Apple Silicon arm64 on macOS 26.4 or newer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"
readonly APP_SOURCE_DIR="$REPO_ROOT/ironmlx-app"
readonly PACKAGING_DIR="$APP_SOURCE_DIR/Packaging"
readonly BUILD_ROOT="$REPO_ROOT/.build/app-bundle"
readonly DIST_DIR="$REPO_ROOT/dist"
readonly APP_BUNDLE="$DIST_DIR/IronMLX.app"
readonly DEPLOYMENT_TARGET="$IRONMLX_MACOS_DEPLOYMENT_TARGET"
readonly ARCHITECTURE="$IRONMLX_ARCHITECTURE"
readonly BUILDER_HOME="${HOME:?HOME must be set}"
readonly C_PATH_REMAP_FLAGS="-ffile-prefix-map=$BUILDER_HOME=. -fdebug-prefix-map=$BUILDER_HOME=."
readonly RUST_PATH_REMAP_FLAG="--remap-path-prefix=$BUILDER_HOME=."

MLX_SOURCE="${MLX_SRC:-$REPO_ROOT/../iron-rivals/mlx}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
UPDATE_CHANNEL="${IRONMLX_UPDATE_CHANNEL:-disabled}"
UPDATE_FEED_URL="${IRONMLX_UPDATE_FEED_URL:-}"
UPDATE_PUBLIC_ED_KEY="${IRONMLX_UPDATE_PUBLIC_ED_KEY:-}"
APP_BUILD_NUMBER="${IRONMLX_APP_BUILD_NUMBER:-}"
DISTRIBUTION_CHANNEL="${IRONMLX_DISTRIBUTION_CHANNEL:-local-release}"

fail() {
  echo "error: $*" >&2
  exit 1
}

for tool in base64 cargo cmake codesign ditto git iconutil lipo plutil realpath sips swift xcrun; do
  command -v "$tool" >/dev/null || fail "required build tool is missing: $tool"
done

python3 "$SCRIPT_DIR/configure-app-updates.py" "$UPDATE_CHANNEL" "$UPDATE_FEED_URL" "$UPDATE_PUBLIC_ED_KEY"
if [ -n "$APP_BUILD_NUMBER" ]; then
  [[ "$APP_BUILD_NUMBER" =~ ^[1-9][0-9]*$ ]] || \
    fail "IRONMLX_APP_BUILD_NUMBER must be a positive integer"
fi

"$SCRIPT_DIR/verify-version-consistency.sh"

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

source_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
source_tree_state="clean"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal)" ]; then
  source_tree_state="dirty"
fi

# Start from an empty build root. Only the MLX build may survive, and only when
# build-mlx.sh confirms its stamp under IRONMLX_REUSE_MLX_BUILD=1.
if [ -d "$BUILD_ROOT" ]; then
  find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 \
    ! -name mlx-build ! -name mlx-install ! -name mlx-build.stamp \
    -exec rm -rf {} +
fi
mkdir -p "$BUILD_ROOT" "$DIST_DIR"

"$SCRIPT_DIR/build-mlx.sh" "$MLX_SOURCE" "$BUILD_ROOT"

echo "==> Verify tracked third-party dependency materials"
MLX_SRC="$MLX_SOURCE" \
  MLX_BUILD_DIR="$BUILD_ROOT/mlx-build" \
  "$SCRIPT_DIR/verify-third-party-materials.sh"
"$SCRIPT_DIR/verify-sbom.sh"

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
    -Xlinker -rpath \
    -Xlinker @executable_path/../Frameworks \
    -Xswiftc=-DIRONMLX_APP_BUNDLE \
    -Xswiftc=-gnone

echo "==> Assemble IronMLX.app"
rm -rf "$APP_BUNDLE"
mkdir -p \
  "$APP_BUNDLE/Contents/MacOS" \
  "$APP_BUNDLE/Contents/Helpers" \
  "$APP_BUNDLE/Contents/Frameworks" \
  "$APP_BUNDLE/Contents/Resources/Legal"
cp "$PACKAGING_DIR/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXSourceCommit -string "$source_commit" "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXSourceTreeState -string "$source_tree_state" "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXMLXCommit -string "$mlx_commit" "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXDistributionChannel -string "$DISTRIBUTION_CHANNEL" "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXDeveloperIDSigned -string unsigned "$APP_BUNDLE/Contents/Info.plist"
plutil -replace IronMLXNotarizationStatus -string not_notarized "$APP_BUNDLE/Contents/Info.plist"
if [ -n "$APP_BUILD_NUMBER" ]; then
  plutil -replace CFBundleVersion -string "$APP_BUILD_NUMBER" "$APP_BUNDLE/Contents/Info.plist"
fi
python3 "$SCRIPT_DIR/configure-app-updates.py" "$UPDATE_CHANNEL" "$UPDATE_FEED_URL" "$UPDATE_PUBLIC_ED_KEY" \
  --plist "$APP_BUNDLE/Contents/Info.plist"
cp "$BUILD_ROOT/swift-build/release/ironmlx-app" "$APP_BUNDLE/Contents/MacOS/IronMLX"
sparkle_framework="$BUILD_ROOT/swift-build/release/Sparkle.framework"
[ -d "$sparkle_framework" ] || fail "Swift build did not produce Sparkle.framework"
ditto "$sparkle_framework" "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework"
cp "$BUILD_ROOT/cargo-target/release/ironmlx" "$APP_BUNDLE/Contents/Helpers/ironmlx"
cp "$BUILD_ROOT/cargo-target/release/iron-bench" "$APP_BUNDLE/Contents/Helpers/iron-bench"
cp "$BUILD_ROOT/mlx-install/lib/mlx.metallib" "$APP_BUNDLE/Contents/Resources/mlx.metallib"
for resource in \
  indextts25-preparation.json \
  dashboard2.html \
  supported-models.json \
  deepseek-harness-logo-dark.svg \
  deepseek-harness-logo.svg \
  dsh-desktop-logo.png \
  hermes-agent-logo.svg \
  logo.png \
  menubar-icon.png \
  menubar-icon@2x.png \
  oh-my-pi-logo.svg \
  opencode-logo.svg \
  sidebar-logo@2x.png; do
  cp "$APP_SOURCE_DIR/Sources/IronMLXAppCore/Resources/$resource" "$APP_BUNDLE/Contents/Resources/$resource"
done
cp "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$APP_BUNDLE/Contents/Resources/Legal/"
cp "$REPO_ROOT/LICENSE" "$APP_BUNDLE/Contents/Resources/Legal/"
cp "$REPO_ROOT/NOTICE" "$APP_BUNDLE/Contents/Resources/Legal/"
cp "$REPO_ROOT/SBOM.cdx.json" "$APP_BUNDLE/Contents/Resources/Legal/"
cp "$REPO_ROOT/third-party-inventory.json" "$APP_BUNDLE/Contents/Resources/Legal/"
cp -R "$REPO_ROOT/THIRD_PARTY_LICENSES" "$APP_BUNDLE/Contents/Resources/Legal/"
cp "$REPO_ROOT/docs/model-license-boundary.md" "$APP_BUNDLE/Contents/Resources/Legal/"

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

echo "==> Thin Sparkle to the supported arm64 product architecture"
while IFS= read -r -d '' bundled_file; do
  if file "$bundled_file" | grep -q "Mach-O"; then
    architectures="$(lipo -archs "$bundled_file")"
    if [ "$architectures" != "$ARCHITECTURE" ]; then
      thin_file="$bundled_file.arm64"
      original_mode="$(stat -f '%Lp' "$bundled_file")"
      lipo "$bundled_file" -thin "$ARCHITECTURE" -output "$thin_file"
      chmod "$original_mode" "$thin_file"
      mv "$thin_file" "$bundled_file"
    fi
  fi
done < <(find "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework" -type f -print0)

echo "==> Ad-hoc sign nested code from the inside out"
while IFS= read -r -d '' bundled_file; do
  if file "$bundled_file" | grep -q "Mach-O"; then
    codesign --force --sign - "$bundled_file"
  fi
done < <(find "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework" -type f -print0)
while IFS= read -r nested_bundle; do
  codesign --force --sign - "$nested_bundle"
done < <(
  find "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework" -type d \
    \( -name '*.app' -o -name '*.xpc' \) -print | awk '{ print length, $0 }' | \
    sort -rn | cut -d ' ' -f 2-
)
codesign --force --sign - "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework"
codesign --force --sign - "$APP_BUNDLE/Contents/Helpers/ironmlx"
codesign --force --sign - "$APP_BUNDLE/Contents/Helpers/iron-bench"
codesign --force --sign - "$APP_BUNDLE/Contents/MacOS/IronMLX"
codesign --force --sign - "$APP_BUNDLE"

"$SCRIPT_DIR/verify-app-bundle.sh" "$APP_BUNDLE"
"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$APP_BUNDLE"
echo "Built: $APP_BUNDLE"
