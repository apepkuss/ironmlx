#!/usr/bin/env bash
# Validate RC archives or package a Developer ID signed, notarized RC.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
rc_tag="${1:?usage: package-release-candidate.sh vX.Y.Z-rc.N [validate|publish]}"
mode="${2:-validate}"
case "$mode" in validate|publish) ;; *) echo 'error: expected validate or publish' >&2; exit 1 ;; esac
app="$REPO_ROOT/dist/IronMLX.app"
python3 "$SCRIPT_DIR/verify-release-identity.py" --candidate "$rc_tag" "$app"
"$SCRIPT_DIR/verify-app-bundle.sh" "$app"
if [ "$mode" = publish ]; then
  "$SCRIPT_DIR/release-legal-gate.sh"
  test "$(plutil -extract IronMLXDistributionChannel raw "$app/Contents/Info.plist")" = release-candidate
  test "$(plutil -extract IronMLXUpdateChannel raw "$app/Contents/Info.plist")" = release-candidate
  test "$(plutil -extract IronMLXDeveloperIDSigned raw "$app/Contents/Info.plist")" = developer_id
  codesign --verify --deep --strict "$app"
  xcrun stapler validate "$app"
  spctl --assess --type execute "$app"
fi
python3 "$SCRIPT_DIR/release-archives.py" assemble "$app" "$REPO_ROOT/.build/stable-release"
