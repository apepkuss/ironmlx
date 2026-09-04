#!/usr/bin/env bash
# Verify the tracked CycloneDX SBOM is deterministic and covers current inputs.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly SBOM="$REPO_ROOT/SBOM.cdx.json"

fail() {
  echo "error: $*" >&2
  exit 1
}

command -v python3 >/dev/null || fail "python3 is required"
[ -s "$SBOM" ] || fail "tracked SBOM is missing or empty: $SBOM"

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-sbom-verify.XXXXXX")"
trap 'rm -rf "$temp_root"' EXIT

python3 "$SCRIPT_DIR/generate-sbom.py" --output "$temp_root/SBOM.cdx.json"
diff -u "$SBOM" "$temp_root/SBOM.cdx.json" || fail \
  "SBOM is stale; regenerate it after reviewing dependency changes"

python3 - "$SBOM" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    bom = json.load(handle)

errors = []
if bom.get("bomFormat") != "CycloneDX":
    errors.append("bomFormat must be CycloneDX")
if bom.get("specVersion") != "1.6":
    errors.append("specVersion must be 1.6")
if not isinstance(bom.get("serialNumber"), str) or not bom["serialNumber"].startswith("urn:uuid:"):
    errors.append("serialNumber must be a UUID URN")
if not bom.get("metadata", {}).get("component", {}).get("name") == "IronMLX":
    errors.append("metadata component must be IronMLX")
components = bom.get("components")
if not isinstance(components, list) or not components:
    errors.append("components must be a non-empty list")
else:
    refs = [item.get("bom-ref") for item in components]
    if any(not ref for ref in refs):
        errors.append("every component must have a bom-ref")
    if len(refs) != len(set(refs)):
        errors.append("component bom-refs must be unique")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
print(f"CycloneDX SBOM verified: {len(components)} components")
PY
