#!/usr/bin/env bash
# Package an explicitly labeled ad-hoc, non-notarized development preview.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

readonly SOURCE_APP="$REPO_ROOT/dist/IronMLX.app"
readonly BUILD_ROOT="$REPO_ROOT/.build/development-preview-release"
readonly ASSET_DIR="$BUILD_ROOT/assets"

preview_tag="${1:-}"
source_commit="${2:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ "$preview_tag" =~ ^preview-[0-9]{8}-[0-9a-f]{7}$ ]] || \
  fail "preview tag must match preview-YYYYMMDD-shortSHA: $preview_tag"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "source commit must be a full lowercase SHA"
[ "${preview_tag##*-}" = "${source_commit:0:7}" ] || \
  fail "preview tag short SHA does not match source commit"
[ -d "$SOURCE_APP/Contents" ] || fail "build the App Bundle first: $SOURCE_APP"

for tool in codesign ditto hdiutil plutil shasum; do
  command -v "$tool" >/dev/null || fail "required packaging tool is missing: $tool"
done

package_name="IronMLX-$preview_tag-ADHOC-NOT-NOTARIZED"
package_root="$BUILD_ROOT/$package_name"
preview_app="$package_root/IronMLX Development Preview.app"
notice_file="$package_root/DEVELOPMENT-PREVIEW-NOTICE.txt"
metadata_file="$package_root/PREVIEW-BUILD-METADATA.json"

rm -rf "$BUILD_ROOT"
mkdir -p "$package_root" "$ASSET_DIR"
ditto "$SOURCE_APP" "$preview_app"

cat > "$notice_file" <<EOF
IronMLX Development Preview / IronMLX 开发预览

警告：${IRONMLX_PREVIEW_WARNING_ZH}。
WARNING: ${IRONMLX_PREVIEW_WARNING_EN}.

This artifact is not a stable release. macOS Gatekeeper is expected to block
normal installation because the App uses only an ad-hoc signature and carries
no Apple notarization ticket.

Preview tag: $preview_tag
IronMLX source commit: $source_commit
MLX source commit: $IRONMLX_MLX_COMMIT
EOF

created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$metadata_file" <<EOF
{
  "apple_notarized": false,
  "created_at_utc": "$created_at",
  "developer_id_signed": false,
  "distribution_channel": "development-preview",
  "ironmlx_commit": "$source_commit",
  "mlx_commit": "$IRONMLX_MLX_COMMIT",
  "preview_tag": "$preview_tag",
  "signature_type": "ad-hoc",
  "warning": "${IRONMLX_PREVIEW_WARNING_ZH}"
}
EOF

cp "$notice_file" "$preview_app/Contents/Resources/DEVELOPMENT-PREVIEW-NOTICE.txt"
cp "$metadata_file" "$preview_app/Contents/Resources/PREVIEW-BUILD-METADATA.json"
plutil -replace CFBundleDisplayName -string "IronMLX Development Preview" \
  "$preview_app/Contents/Info.plist"
plutil -insert IronMLXDistributionChannel -string "development-preview" \
  "$preview_app/Contents/Info.plist"
plutil -insert IronMLXDeveloperIDSigned -bool NO "$preview_app/Contents/Info.plist"
plutil -insert IronMLXAppleNotarized -bool NO "$preview_app/Contents/Info.plist"
plutil -insert IronMLXPreviewTag -string "$preview_tag" "$preview_app/Contents/Info.plist"
plutil -insert IronMLXSourceCommit -string "$source_commit" "$preview_app/Contents/Info.plist"

codesign --force --deep --sign - "$preview_app"
"$SCRIPT_DIR/verify-app-bundle.sh" "$preview_app"

signature_details="$(codesign -dvvv "$preview_app" 2>&1)"
grep -Fq "Signature=adhoc" <<<"$signature_details" || fail "preview App is not ad-hoc signed"
grep -Fq "TeamIdentifier=not set" <<<"$signature_details" || \
  fail "preview App unexpectedly has a signing TeamIdentifier"

zip_path="$ASSET_DIR/$package_name.zip"
dmg_path="$ASSET_DIR/$package_name.dmg"
ditto -c -k --sequesterRsrc --keepParent "$package_root" "$zip_path"
hdiutil create \
  -volname "IronMLX Dev Preview" \
  -srcfolder "$package_root" \
  -format UDZO \
  -ov \
  "$dmg_path" >/dev/null

cp "$notice_file" "$ASSET_DIR/DEVELOPMENT-PREVIEW-NOTICE.txt"
cp "$metadata_file" "$ASSET_DIR/PREVIEW-BUILD-METADATA.json"

cat > "$ASSET_DIR/RELEASE-NOTES.md" <<EOF
# ⚠️ IronMLX 开发预览

> **${IRONMLX_PREVIEW_WARNING_ZH}。**

This prerelease is **${IRONMLX_PREVIEW_WARNING_EN}**.

- Channel: GitHub Actions development preview
- Preview tag: \`$preview_tag\`
- IronMLX immutable commit: \`$source_commit\`
- MLX immutable commit: \`$IRONMLX_MLX_COMMIT\`
- Platform: Apple Silicon arm64, macOS 26.2+
- Signature: ad-hoc only; no Developer ID identity or Team ID
- Apple notarization/stapling: not performed

Gatekeeper is expected to block normal installation. This build must not be
described or redistributed as a stable release. Developer ID signing,
notarization, stapling, formal target-machine inference acceptance, and stable
release publication remain outside this preview stage.
EOF

(
  cd "$ASSET_DIR"
  shasum -a 256 \
    "$(basename "$dmg_path")" \
    "$(basename "$zip_path")" \
    DEVELOPMENT-PREVIEW-NOTICE.txt \
    PREVIEW-BUILD-METADATA.json \
    RELEASE-NOTES.md > SHA256SUMS
)

"$SCRIPT_DIR/verify-development-preview.sh" "$ASSET_DIR" "$preview_tag" "$source_commit"
echo "Development preview assets: $ASSET_DIR"
