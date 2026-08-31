#!/usr/bin/env bash
# Build and exercise a development-only Sparkle update between two ad-hoc App Bundles.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly APPCAST_PORT="${IRONMLX_DEVELOPMENT_APPCAST_PORT:-18443}"
readonly FEED_URL="https://127.0.0.1:$APPCAST_PORT/appcast.xml"
readonly WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-sparkle-validation.XXXXXX")"
readonly PRIVATE_KEY="$WORK_ROOT/development-eddsa-private-key"
readonly TLS_KEY="$WORK_ROOT/localhost-tls-key.pem"
readonly TLS_CERT="$WORK_ROOT/localhost-tls-cert.pem"
readonly FEED_ROOT="$WORK_ROOT/feed"
readonly INSTALL_ROOT="$WORK_ROOT/install"
readonly INSTALL_APP="$INSTALL_ROOT/IronMLX.app"
readonly READY_MARKER="$WORK_ROOT/update-ready"
readonly OFFLINE_MARKER="$WORK_ROOT/offline-update"
readonly SPARKLE_BIN="$REPO_ROOT/ironmlx-app/.build/artifacts/sparkle/Sparkle/bin"
readonly LOCAL_CARGO_ABOUT="$REPO_ROOT/.build/release-tools/bin/cargo-about"
server_pid=""
trusted_certificate_sha=""
install_app_real=""

fail() {
  echo "error: $*" >&2
  exit 1
}

cleanup() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
  if [ -n "$install_app_real" ]; then
    for app_root in "$INSTALL_APP" "$install_app_real"; do
      while IFS= read -r process_id; do
        [ -n "$process_id" ] && kill "$process_id" 2>/dev/null || true
      done < <(pgrep -f "$app_root/Contents/" || true)
    done
  fi
  if [ -n "$trusted_certificate_sha" ]; then
    security delete-certificate -Z "$trusted_certificate_sha" "$HOME/Library/Keychains/login.keychain-db" \
      >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK_ROOT"
}
trap cleanup EXIT

for tool in codesign curl ditto openssl plutil python3 security swift; do
  command -v "$tool" >/dev/null || fail "required validation tool is missing: $tool"
done

mkdir -p "$FEED_ROOT" "$INSTALL_ROOT"
public_key="$($SCRIPT_DIR/generate-development-update-key.swift "$PRIVATE_KEY")"
[[ "$public_key" =~ ^[A-Za-z0-9+/]{43}=$ ]] || fail "development public EdDSA key is invalid"

openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 1 \
  -keyout "$TLS_KEY" \
  -out "$TLS_CERT" \
  -subj '/CN=127.0.0.1' \
  -addext 'subjectAltName=IP:127.0.0.1,DNS:localhost' \
  -addext 'keyUsage=digitalSignature,keyEncipherment' \
  -addext 'extendedKeyUsage=serverAuth' \
  >/dev/null 2>&1
trusted_certificate_sha="$(openssl x509 -in "$TLS_CERT" -noout -fingerprint -sha1 | cut -d= -f2 | tr -d :)"

echo "==> Build development update source (build 1)"
env \
  CARGO_ABOUT="${CARGO_ABOUT:-$LOCAL_CARGO_ABOUT}" \
  IRONMLX_UPDATE_CHANNEL=development \
  IRONMLX_UPDATE_FEED_URL="$FEED_URL" \
  IRONMLX_UPDATE_PUBLIC_ED_KEY="$public_key" \
  IRONMLX_APP_BUILD_NUMBER=1 \
  "$SCRIPT_DIR/build-app-bundle.sh"
ditto "$REPO_ROOT/dist/IronMLX.app" "$INSTALL_APP"
install_app_real="$(realpath "$INSTALL_APP")"

echo "==> Build development update target (build 2)"
env \
  CARGO_ABOUT="${CARGO_ABOUT:-$LOCAL_CARGO_ABOUT}" \
  IRONMLX_UPDATE_CHANNEL=development \
  IRONMLX_UPDATE_FEED_URL="$FEED_URL" \
  IRONMLX_UPDATE_PUBLIC_ED_KEY="$public_key" \
  IRONMLX_APP_BUILD_NUMBER=2 \
  "$SCRIPT_DIR/build-app-bundle.sh"
ditto -c -k --sequesterRsrc --keepParent \
  "$REPO_ROOT/dist/IronMLX.app" "$FEED_ROOT/IronMLX-0.1.0-2.zip"

[ -x "$SPARKLE_BIN/generate_appcast" ] || fail "Sparkle generate_appcast tool is missing"
[ -x "$SPARKLE_BIN/sign_update" ] || fail "Sparkle sign_update tool is missing"
"$SPARKLE_BIN/generate_appcast" \
  --ed-key-file "$PRIVATE_KEY" \
  --download-url-prefix "https://127.0.0.1:$APPCAST_PORT/" \
  --maximum-deltas 0 \
  "$FEED_ROOT"
grep -Fq 'sparkle:edSignature=' "$FEED_ROOT/appcast.xml" || fail "appcast feed is not signed"
"$SPARKLE_BIN/sign_update" --verify --ed-key-file "$PRIVATE_KEY" \
  "$FEED_ROOT/appcast.xml" >/dev/null || fail "signed appcast feed failed verification"

echo "==> Verify archive and feed tampering are rejected"
archive_signature="$(
  "$SPARKLE_BIN/sign_update" --ed-key-file "$PRIVATE_KEY" \
    "$FEED_ROOT/IronMLX-0.1.0-2.zip" | sed -n 's/.*sparkle:edSignature="\([^"]*\)".*/\1/p'
)"
[ -n "$archive_signature" ] || fail "could not obtain archive signature"
cp "$FEED_ROOT/IronMLX-0.1.0-2.zip" "$WORK_ROOT/tampered.zip"
printf 'tampered' >> "$WORK_ROOT/tampered.zip"
if "$SPARKLE_BIN/sign_update" --verify --ed-key-file "$PRIVATE_KEY" \
  "$WORK_ROOT/tampered.zip" "$archive_signature" >/dev/null 2>&1; then
  fail "tampered update archive passed EdDSA verification"
fi
cp "$FEED_ROOT/appcast.xml" "$WORK_ROOT/tampered-appcast.xml"
sed -i '' 's/<title>/<title>TAMPERED /' "$WORK_ROOT/tampered-appcast.xml"
if "$SPARKLE_BIN/sign_update" --verify --ed-key-file "$PRIVATE_KEY" \
  "$WORK_ROOT/tampered-appcast.xml" >/dev/null 2>&1; then
  fail "tampered signed feed passed EdDSA verification"
fi

echo "==> Serve signed appcast over trusted loopback HTTPS"
echo "macOS may request permission to trust the temporary 127.0.0.1 certificate"
security add-trusted-cert -r trustRoot -p ssl -s 127.0.0.1 \
  -k "$HOME/Library/Keychains/login.keychain-db" "$TLS_CERT"
python3 "$SCRIPT_DIR/serve-development-appcast.py" \
  --root "$FEED_ROOT" \
  --certificate "$TLS_CERT" \
  --private-key "$TLS_KEY" \
  --port "$APPCAST_PORT" \
  >"$WORK_ROOT/https-server.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 50); do
  if curl --silent --fail --cacert "$TLS_CERT" "$FEED_URL" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl --silent --fail --cacert "$TLS_CERT" "$FEED_URL" >/dev/null || fail "loopback appcast server did not start"

echo "==> Update ad-hoc build 1 to build 2 through Sparkle"
open -n "$INSTALL_APP" --args \
  --ironmlx-development-update-test-marker "$READY_MARKER"
for _ in $(seq 1 600); do
  [ -f "$READY_MARKER.error" ] && fail "Sparkle update failed: $(cat "$READY_MARKER.error")"
  current_build="$(plutil -extract CFBundleVersion raw "$INSTALL_APP/Contents/Info.plist")"
  [ "$current_build" = 2 ] && break
  sleep 0.25
done
[ "$(plutil -extract CFBundleVersion raw "$INSTALL_APP/Contents/Info.plist")" = 2 ] || \
  fail "Sparkle did not replace build 1 with build 2"
codesign --verify --deep --strict "$INSTALL_APP"

echo "==> Verify offline update failure leaves the installed app intact"
while IFS= read -r process_id; do
  [ -n "$process_id" ] && kill "$process_id" 2>/dev/null || true
done < <(pgrep -f "$install_app_real/Contents/MacOS/IronMLX" || true)
plutil -replace SUFeedURL -string 'https://127.0.0.1:1/appcast.xml' \
  "$INSTALL_APP/Contents/Info.plist"
codesign --force --sign - "$INSTALL_APP"
open -n "$INSTALL_APP" --args \
  --ironmlx-development-update-test-marker "$OFFLINE_MARKER"
for _ in $(seq 1 200); do
  [ -f "$OFFLINE_MARKER.error" ] && break
  sleep 0.25
done
[ -f "$OFFLINE_MARKER.error" ] || fail "offline update check did not report a bounded failure"
[ "$(plutil -extract CFBundleVersion raw "$INSTALL_APP/Contents/Info.plist")" = 2 ] || \
  fail "offline update check changed the installed app"

echo "Development Sparkle validation passed: signed build 1 -> build 2, tamper rejection, signed-feed rejection, and offline recovery"
