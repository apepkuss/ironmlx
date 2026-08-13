#!/usr/bin/env bash
# Static release gate for an assembled IronMLX.app.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"
readonly APP_BUNDLE="${1:-$REPO_ROOT/dist/IronMLX.app}"
readonly EXPECTED_ARCHITECTURE="arm64"
readonly EXPECTED_MACOS_VERSION="26.2"
readonly EXPECTED_PRODUCT_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || fail "required bundled file is missing: $1"
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

  while IFS= read -r dependency; do
    case "$dependency" in
      /System/Library/*|/usr/lib/*) ;;
      @rpath/Sparkle.framework/Versions/B/Sparkle) ;;
      *) fail "$label has a non-system dynamic dependency: $dependency" ;;
    esac
  done < <(otool -L "$binary" | awk 'NR > 1 { print $1 }')
  echo "ok: $label is arm64, minos 26.2, with system-only dynamic dependencies"
}

[ -d "$APP_BUNDLE/Contents" ] || fail "not an App Bundle: $APP_BUNDLE"
require_file "$APP_BUNDLE/Contents/Info.plist"
for file in \
  Contents/MacOS/IronMLX \
  Contents/Helpers/ironmlx \
  Contents/Helpers/iron-bench \
  Contents/Frameworks/Sparkle.framework/Versions/B/Sparkle \
  Contents/Resources/mlx.metallib \
  Contents/Resources/dashboard2.html \
  Contents/Resources/hermes-agent-logo.svg \
  Contents/Resources/oh-my-pi-logo.svg \
  Contents/Resources/AppIcon.icns \
  Contents/Resources/menubar-icon.png \
  Contents/Resources/menubar-icon@2x.png \
  Contents/Resources/logo.png \
  Contents/Resources/sidebar-logo@2x.png \
  Contents/Resources/Legal/THIRD_PARTY_NOTICES.md \
  Contents/Resources/Legal/third-party-inventory.json; do
  require_file "$APP_BUNDLE/$file"
done
[ -d "$APP_BUNDLE/Contents/Resources/Legal/THIRD_PARTY_LICENSES" ] || \
  fail "bundled third-party license directory is missing"
diff -q \
  "$REPO_ROOT/THIRD_PARTY_NOTICES.md" \
  "$APP_BUNDLE/Contents/Resources/Legal/THIRD_PARTY_NOTICES.md" >/dev/null || \
  fail "bundled third-party notices differ from the verified source material"
diff -q \
  "$REPO_ROOT/third-party-inventory.json" \
  "$APP_BUNDLE/Contents/Resources/Legal/third-party-inventory.json" >/dev/null || \
  fail "bundled third-party inventory differs from the verified source material"
diff -qr \
  "$REPO_ROOT/THIRD_PARTY_LICENSES" \
  "$APP_BUNDLE/Contents/Resources/Legal/THIRD_PARTY_LICENSES" >/dev/null || \
  fail "bundled third-party license texts differ from the verified source material"

[ "$(plutil -extract CFBundleIdentifier raw "$APP_BUNDLE/Contents/Info.plist")" = "com.ironmlx.app" ] || \
  fail "unexpected CFBundleIdentifier"
[ "$(plutil -extract LSMinimumSystemVersion raw "$APP_BUNDLE/Contents/Info.plist")" = "$EXPECTED_MACOS_VERSION" ] || \
  fail "Info.plist minimum macOS must be $EXPECTED_MACOS_VERSION"
[ "$(plutil -extract CFBundleShortVersionString raw "$APP_BUNDLE/Contents/Info.plist")" = \
  "$EXPECTED_PRODUCT_VERSION" ] || fail "Info.plist product version must be $EXPECTED_PRODUCT_VERSION"
source_commit="$(plutil -extract IronMLXSourceCommit raw "$APP_BUNDLE/Contents/Info.plist")"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "IronMLXSourceCommit must be a full lowercase SHA"
source_tree_state="$(plutil -extract IronMLXSourceTreeState raw "$APP_BUNDLE/Contents/Info.plist")"
[[ "$source_tree_state" =~ ^(clean|dirty)$ ]] || fail "invalid IronMLXSourceTreeState: $source_tree_state"
[ "$(plutil -extract IronMLXMLXCommit raw "$APP_BUNDLE/Contents/Info.plist")" = "$IRONMLX_MLX_COMMIT" ] || \
  fail "IronMLXMLXCommit does not match the pinned MLX commit"
[ -n "$(plutil -extract IronMLXDistributionChannel raw "$APP_BUNDLE/Contents/Info.plist")" ] || \
  fail "IronMLXDistributionChannel is missing"
developer_id_status="$(plutil -extract IronMLXDeveloperIDSigned raw "$APP_BUNDLE/Contents/Info.plist")"
[[ "$developer_id_status" =~ ^(unsigned|developer_id|unavailable)$ ]] || \
  fail "invalid IronMLXDeveloperIDSigned: $developer_id_status"
notarization_status="$(plutil -extract IronMLXNotarizationStatus raw "$APP_BUNDLE/Contents/Info.plist")"
[[ "$notarization_status" =~ ^(not_notarized|stapled|unavailable)$ ]] || \
  fail "invalid IronMLXNotarizationStatus: $notarization_status"

"$APP_BUNDLE/Contents/Helpers/ironmlx" --version | \
  grep -Fxq "ironmlx $EXPECTED_PRODUCT_VERSION" || fail "ironmlx helper version mismatch"
"$APP_BUNDLE/Contents/Helpers/iron-bench" --version | \
  grep -Fxq "iron-bench $EXPECTED_PRODUCT_VERSION" || fail "iron-bench helper version mismatch"

verify_macho "IronMLX App executable" "$APP_BUNDLE/Contents/MacOS/IronMLX"
verify_macho "IronMLX backend helper" "$APP_BUNDLE/Contents/Helpers/ironmlx"
verify_macho "iron-bench helper" "$APP_BUNDLE/Contents/Helpers/iron-bench"

app_rpaths="$(otool -l "$APP_BUNDLE/Contents/MacOS/IronMLX" | awk '
  /LC_RPATH/ { in_rpath = 1; next }
  in_rpath && $1 == "path" { print $2; in_rpath = 0 }
')"
grep -Fxq '@executable_path/../Frameworks' <<< "$app_rpaths" || \
  fail "IronMLX App executable is missing the Bundle-local Frameworks rpath"

while IFS= read -r -d '' bundled_file; do
  if file "$bundled_file" | grep -q "Mach-O"; then
    architectures="$(lipo -archs "$bundled_file")"
    [ "$architectures" = "$EXPECTED_ARCHITECTURE" ] || \
      fail "bundled Sparkle code must contain only arm64: $bundled_file ($architectures)"
    while IFS= read -r dependency; do
      case "$dependency" in
        /System/Library/*|/usr/lib/*|@rpath/Sparkle.framework/Versions/B/Sparkle) ;;
        *) fail "bundled Sparkle code has an external dependency: $bundled_file -> $dependency" ;;
      esac
    done < <(otool -L "$bundled_file" | awk 'NR > 1 { print $1 }')
  fi
done < <(find "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework" -type f -print0)

metallib="$APP_BUNDLE/Contents/Resources/mlx.metallib"
file "$metallib" | grep -Fq "MetalLib executable (MacOS)" || fail "mlx.metallib is not a macOS metallib"
LC_ALL=C grep -aEq 'air64_v[0-9]+-apple-macosx26\.2\.0' "$metallib" || \
  fail "mlx.metallib does not target macOS 26.2"

sparkle_root="$(realpath "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework")"
while IFS= read -r -d '' bundle_link; do
  case "$bundle_link" in
    "$APP_BUNDLE/Contents/Frameworks/Sparkle.framework"/*)
      resolved_link="$(realpath "$bundle_link")"
      case "$resolved_link" in
        "$sparkle_root"/*) ;;
        *) fail "Sparkle framework symlink escapes its Bundle root: $bundle_link" ;;
      esac
      ;;
    *) fail "unexpected App Bundle symbolic link: $bundle_link" ;;
  esac
done < <(find "$APP_BUNDLE" -type l -print0)
while IFS= read -r -d '' bundled_file; do
  if developer_paths="$(strings -a "$bundled_file" | LC_ALL=C grep -E '/Users/|target/(debug|release)')"; then
    fail "App Bundle contains a developer or Cargo fallback path in $bundled_file: $developer_paths"
  fi
done < <(find "$APP_BUNDLE" -type f -print0)

codesign --verify --deep --strict "$APP_BUNDLE"
echo "IronMLX.app static bundle verification passed"
