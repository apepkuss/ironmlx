#!/usr/bin/env bash
# Verify preview labels, ad-hoc signature, archives, checksums, and Bundle gates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

asset_dir="${1:-}"
preview_tag="${2:-}"
source_commit="${3:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -d "$asset_dir" ] || fail "asset directory not found: $asset_dir"
[[ "$preview_tag" =~ ^preview-[0-9]{8}-[0-9a-f]{7}$ ]] || fail "invalid preview tag"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source commit"

package_name="IronMLX-$preview_tag-ADHOC-NOT-NOTARIZED"
zip_path="$asset_dir/$package_name.zip"
dmg_path="$asset_dir/$package_name.dmg"
for required in \
  "$zip_path" \
  "$dmg_path" \
  "$asset_dir/DEVELOPMENT-PREVIEW-NOTICE.txt" \
  "$asset_dir/PREVIEW-BUILD-METADATA.json" \
  "$asset_dir/RELEASE-NOTES.md" \
  "$asset_dir/SHA256SUMS"; do
  [ -f "$required" ] || fail "required preview asset is missing: $required"
done

grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" "$asset_dir/DEVELOPMENT-PREVIEW-NOTICE.txt" || \
  fail "preview notice does not contain the required warning"
grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" "$asset_dir/RELEASE-NOTES.md" || \
  fail "release notes do not contain the required warning"

[ "$(plutil -extract preview_tag raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "$preview_tag" ] || \
  fail "metadata preview tag mismatch"
[ "$(plutil -extract ironmlx_commit raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "$source_commit" ] || \
  fail "metadata source commit mismatch"
[ "$(plutil -extract developer_id_signed raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "false" ] || \
  fail "metadata must declare Developer ID signing disabled"
[ "$(plutil -extract apple_notarized raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "false" ] || \
  fail "metadata must declare Apple notarization disabled"

(
  cd "$asset_dir"
  shasum -a 256 -c SHA256SUMS
)

if zipinfo -1 "$zip_path" | grep -E '(^/|(^|/)\.\.(/|$)|/Users/)' >/dev/null; then
  fail "ZIP contains an unsafe or developer-specific path"
fi

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-preview-verify.XXXXXX")"
zip_extract="$temp_root/zip"
dmg_mount="$temp_root/dmg"
mounted=0

cleanup() {
  if [ "$mounted" -eq 1 ]; then
    hdiutil detach "$dmg_mount" -quiet || true
  fi
  rm -rf "$temp_root"
}
trap cleanup EXIT

verify_preview_app() {
  local preview_app="$1"
  local signature_details

  [ -d "$preview_app/Contents" ] || fail "preview App is missing: $preview_app"
  "$SCRIPT_DIR/verify-app-bundle.sh" "$preview_app"
  [ "$(plutil -extract CFBundleDisplayName raw "$preview_app/Contents/Info.plist")" = \
    "IronMLX Development Preview" ] || fail "preview display name is not explicit"
  [ "$(plutil -extract IronMLXDistributionChannel raw "$preview_app/Contents/Info.plist")" = \
    "development-preview" ] || fail "preview distribution channel is missing"
  [ "$(plutil -extract IronMLXDeveloperIDSigned raw "$preview_app/Contents/Info.plist")" = "false" ] || \
    fail "preview App must declare Developer ID signing disabled"
  [ "$(plutil -extract IronMLXAppleNotarized raw "$preview_app/Contents/Info.plist")" = "false" ] || \
    fail "preview App must declare Apple notarization disabled"
  grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" \
    "$preview_app/Contents/Resources/DEVELOPMENT-PREVIEW-NOTICE.txt" || \
    fail "preview App does not contain the required warning"

  signature_details="$(codesign -dvvv "$preview_app" 2>&1)"
  grep -Fq "Signature=adhoc" <<<"$signature_details" || fail "preview App is not ad-hoc signed"
  grep -Fq "TeamIdentifier=not set" <<<"$signature_details" || \
    fail "preview App unexpectedly has a signing TeamIdentifier"
}

mkdir -p "$zip_extract" "$dmg_mount"
ditto -x -k "$zip_path" "$zip_extract"
verify_preview_app "$zip_extract/$package_name/IronMLX Development Preview.app"

hdiutil attach "$dmg_path" -readonly -nobrowse -mountpoint "$dmg_mount" -quiet
mounted=1
verify_preview_app "$dmg_mount/IronMLX Development Preview.app"
hdiutil detach "$dmg_mount" -quiet
mounted=0

echo "IronMLX development preview verification passed"
