#!/usr/bin/env bash
# Verify that a distributable App/archive does not bundle model weights.

set -euo pipefail

artifact="${1:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -n "$artifact" ] || fail "usage: $0 <App bundle, ZIP, or DMG>"
[ -e "$artifact" ] || fail "artifact not found: $artifact"

is_model_weight_path() {
  local path="${1##*/}"
  case "$path" in
    *.safetensors|*.safetensors.index.json|*.gguf|*.ggml|*.bin|*.pt|*.pth|*.onnx|*.mlx|*.npz)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

verify_directory() {
  local root="$1"
  local violation=""
  local path
  while IFS= read -r -d '' path; do
    if is_model_weight_path "$path"; then
      echo "model-weight file found: $path" >&2
      violation=1
    fi
  done < <(find -P "$root" -type f -print0)
  [ -z "$violation" ] || fail "artifact bundles model weights: $root"
}

verify_zip() {
  local archive="$1"
  local violation=""
  local path
  command -v zipinfo >/dev/null || fail "required verification tool is missing: zipinfo"
  zipinfo -t "$archive" >/dev/null || fail "ZIP archive is invalid: $archive"
  while IFS= read -r path; do
    if is_model_weight_path "$path"; then
      echo "model-weight file found in ZIP: $path" >&2
      violation=1
    fi
  done < <(zipinfo -1 "$archive")
  [ -z "$violation" ] || fail "ZIP bundles model weights: $archive"
}

case "$artifact" in
  *.zip|*.ZIP)
    verify_zip "$artifact"
    ;;
  *.dmg|*.DMG)
    command -v hdiutil >/dev/null || fail "required verification tool is missing: hdiutil"
    mount_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-model-boundary.XXXXXX")"
    mounted=0
    cleanup() {
      if [ "$mounted" -eq 1 ]; then
        hdiutil detach "$mount_root" -quiet || true
      fi
      rmdir "$mount_root" 2>/dev/null || true
    }
    trap cleanup EXIT
    hdiutil attach "$artifact" -readonly -nobrowse -mountpoint "$mount_root" -quiet
    mounted=1
    verify_directory "$mount_root"
    ;;
  *)
    [ -d "$artifact" ] || fail "unsupported artifact type (expected App bundle, ZIP, or DMG): $artifact"
    verify_directory "$artifact"
    ;;
esac

echo "Model distribution boundary passed: $artifact"
