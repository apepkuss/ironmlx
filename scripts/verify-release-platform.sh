#!/usr/bin/env bash
# Release platform entry point for the current App Bundle release gate.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/verify-app-bundle.sh" "${1:-$SCRIPT_DIR/../dist/IronMLX.app}"
