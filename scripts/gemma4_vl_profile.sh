#!/usr/bin/env bash
# Gemma4 vision profile baseline runner.
#
# Captures comparable profile logs for single-image, repeated multi-image, and
# distinct multi-image prompts across several prefill chunk sizes. Reports are
# written under reports/gemma4-vl-profile/<timestamp>/.
#
# Usage:
#   ./scripts/gemma4_vl_profile.sh
#   ./scripts/gemma4_vl_profile.sh --case multi-repeat
#   ./scripts/gemma4_vl_profile.sh --build
#
# Env:
#   MLX_DIR=$HOME/.local/mlx
#   IRONMLX_BIN=<repo>/target/release/ironmlx
#   CARGO_TARGET_DIR=<target-dir>      # used to derive IRONMLX_BIN
#   GEMMA4_MODEL=<snapshot-dir>
#   GEMMA4_IMAGE=<image-path>
#   GEMMA4_DISTINCT_IMAGE_1=<image-path>
#   GEMMA4_DISTINCT_IMAGE_2=<image-path>
#   CHUNK_SIZES="0 256 64"
#   MAX_TOKENS=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE_CASE="all"
BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            PROFILE_CASE="${2:-}"
            shift 2
            ;;
        --build)
            BUILD=1
            shift
            ;;
        -h|--help)
            sed -n '1,25p' "$0"
            exit 0
            ;;
        *)
            echo "[gemma4-vl-profile] unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

case "$PROFILE_CASE" in
    all|single|multi-repeat|multi-distinct)
        ;;
    *)
        echo "[gemma4-vl-profile] --case must be one of: all, single, multi-repeat, multi-distinct" >&2
        exit 2
        ;;
esac

export MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"

if [[ -z "${IRONMLX_BIN:-}" ]]; then
    if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
        IRONMLX_BIN="$CARGO_TARGET_DIR/release/ironmlx"
    else
        IRONMLX_BIN="$REPO_ROOT/target/release/ironmlx"
    fi
fi

if [[ "$BUILD" -eq 1 ]]; then
    (cd "$REPO_ROOT" && MLX_DIR="$MLX_DIR" cargo build --release)
fi

if [[ ! -x "$IRONMLX_BIN" ]]; then
    echo "[gemma4-vl-profile] ERROR: ironmlx binary not executable: $IRONMLX_BIN" >&2
    echo "[gemma4-vl-profile]        run with --build or set IRONMLX_BIN/CARGO_TARGET_DIR" >&2
    exit 2
fi

GEMMA4_MODEL="${GEMMA4_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/* 2>/dev/null | head -1 || true)}"
if [[ -z "$GEMMA4_MODEL" || ! -d "$GEMMA4_MODEL" ]]; then
    echo "[gemma4-vl-profile] ERROR: GEMMA4_MODEL not found: $GEMMA4_MODEL" >&2
    exit 2
fi

FIXTURE_IMAGE="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg"
FIXTURE_DISTINCT_1="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg"
FIXTURE_DISTINCT_2="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg"

GEMMA4_IMAGE="${GEMMA4_IMAGE:-$FIXTURE_IMAGE}"
GEMMA4_DISTINCT_IMAGE_1="${GEMMA4_DISTINCT_IMAGE_1:-$FIXTURE_DISTINCT_1}"
GEMMA4_DISTINCT_IMAGE_2="${GEMMA4_DISTINCT_IMAGE_2:-$FIXTURE_DISTINCT_2}"
CHUNK_SIZES="${CHUNK_SIZES:-0 256 64}"
MAX_TOKENS="${MAX_TOKENS:-2}"
COOLDOWN_SECS="${COOLDOWN_SECS:-1}"
PROMPT="${PROMPT:-Describe this image in one short sentence.}"
MULTI_PROMPT="${MULTI_PROMPT:-Describe the images in one short sentence.}"

STAMP="$(date +%Y-%m-%d-%H%M%S)"
REPORT_DIR="$REPO_ROOT/reports/gemma4-vl-profile/$STAMP"
mkdir -p "$REPORT_DIR"

METRICS_TSV="$REPORT_DIR/metrics.tsv"
SUMMARY_TSV="$REPORT_DIR/summary.tsv"
{
    printf 'case\tchunk_size\tmetric\tvalue_ms\tlog_file\n'
} > "$METRICS_TSV"
{
    printf 'case\tchunk_size\toutput_sha256\toutput_bytes\thidden_chunks\tslice_projects\tdedup_ms\tdedup_images\tdedup_unique\tdedup_duplicates\tstdout_file\tstderr_file\n'
} > "$SUMMARY_TSV"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[gemma4-vl-profile] ERROR: file not found: $1" >&2
        exit 2
    fi
}

append_metrics() {
    local name="$1"
    local chunk_size="$2"
    local log_file="$3"

    awk -v case_name="$name" -v chunk="$chunk_size" -v log_file="$log_file" '
        /\[gemma4-vl-profile\]/ {
            line = $0
            while (match(line, /[[:alnum:]_]+_ms=[0-9]+(\.[0-9]+)?/)) {
                kv = substr(line, RSTART, RLENGTH)
                split(kv, parts, "=")
                metric = parts[1]
                sub(/_ms$/, "", metric)
                printf "%s\t%s\t%s\t%s\t%s\n", case_name, chunk, metric, parts[2], log_file
                line = substr(line, RSTART + RLENGTH)
            }
        }
    ' "$log_file" >> "$METRICS_TSV"
}

dedup_field() {
    local line="$1"
    local field="$2"
    printf '%s\n' "$line" | sed -n "s/.*${field}=\\([0-9.][0-9.]*\\).*/\\1/p"
}

run_case() {
    local name="$1"
    local prompt="$2"
    shift 2
    local images=("$@")

    for image in "${images[@]}"; do
        require_file "$image"
    done

    for chunk_size in $CHUNK_SIZES; do
        local base="$REPORT_DIR/${name}-chunk${chunk_size}"
        local stdout_file="${base}.stdout.txt"
        local stderr_file="${base}.stderr.log"
        local command_file="${base}.command.txt"

        local image_args=()
        for image in "${images[@]}"; do
            image_args+=(--image "$image")
        done

        {
            printf 'MLX_DIR=%q RUST_LOG=info IRONMLX_GEMMA4_VL_PROFILE=1 %q generate --model %q ' "$MLX_DIR" "$IRONMLX_BIN" "$GEMMA4_MODEL"
            printf '%q ' "${image_args[@]}"
            printf -- '--prompt %q --max-tokens %q --prefill-chunk-size %q\n' "$prompt" "$MAX_TOKENS" "$chunk_size"
        } > "$command_file"

        echo "=== $name chunk_size=$chunk_size ==="
        echo "  images: ${images[*]}"
        echo "  stdout: $stdout_file"
        echo "  stderr: $stderr_file"

        MLX_DIR="$MLX_DIR" \
            RUST_LOG=info \
            IRONMLX_GEMMA4_VL_PROFILE=1 \
            "$IRONMLX_BIN" generate \
            --model "$GEMMA4_MODEL" \
            "${image_args[@]}" \
            --prompt "$prompt" \
            --max-tokens "$MAX_TOKENS" \
            --prefill-chunk-size "$chunk_size" \
            > "$stdout_file" 2> "$stderr_file"

        append_metrics "$name" "$chunk_size" "$stderr_file"

        local output_sha output_bytes hidden_chunks slice_projects dedup_line
        output_sha="$(shasum -a 256 "$stdout_file" | awk '{print $1}')"
        output_bytes="$(wc -c < "$stdout_file" | tr -d ' ')"
        hidden_chunks="$(grep -c 'forward_vl_hidden_chunk_total_ms' "$stderr_file" || true)"
        slice_projects="$(grep -c 'forward_vl_slice_project_ms' "$stderr_file" || true)"
        dedup_line="$(grep 'compute_vision_exact_dedup_ms' "$stderr_file" | tail -1 || true)"

        local dedup_ms dedup_images dedup_unique dedup_duplicates
        dedup_ms="$(dedup_field "$dedup_line" 'compute_vision_exact_dedup_ms')"
        dedup_images="$(dedup_field "$dedup_line" 'images')"
        dedup_unique="$(dedup_field "$dedup_line" 'unique')"
        dedup_duplicates="$(dedup_field "$dedup_line" 'duplicates')"
        dedup_ms="${dedup_ms:--}"
        dedup_images="${dedup_images:--}"
        dedup_unique="${dedup_unique:--}"
        dedup_duplicates="${dedup_duplicates:--}"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$chunk_size" "$output_sha" "$output_bytes" "$hidden_chunks" "$slice_projects" \
            "$dedup_ms" "$dedup_images" "$dedup_unique" "$dedup_duplicates" \
            "$stdout_file" "$stderr_file" >> "$SUMMARY_TSV"

        local preview
        preview="$(tr '\n' ' ' < "$stdout_file" | cut -c 1-100)"
        echo "  output: $preview"
        echo "  hidden_chunks=$hidden_chunks slice_projects=$slice_projects dedup=${dedup_unique:-?}/${dedup_images:-?}"
        sleep "$COOLDOWN_SECS"
    done
}

echo "=== Gemma4 VL profile — $(date) ==="
echo "report: $REPORT_DIR"
echo "binary: $IRONMLX_BIN"
echo "model:  $GEMMA4_MODEL"
echo "chunks: $CHUNK_SIZES"
echo ""

if [[ "$PROFILE_CASE" == "all" || "$PROFILE_CASE" == "single" ]]; then
    run_case "single" "$PROMPT" "$GEMMA4_IMAGE"
    echo ""
fi

if [[ "$PROFILE_CASE" == "all" || "$PROFILE_CASE" == "multi-repeat" ]]; then
    run_case "multi-repeat" "$MULTI_PROMPT" "$GEMMA4_IMAGE" "$GEMMA4_IMAGE"
    echo ""
fi

if [[ "$PROFILE_CASE" == "all" || "$PROFILE_CASE" == "multi-distinct" ]]; then
    run_case "multi-distinct" "$MULTI_PROMPT" "$GEMMA4_DISTINCT_IMAGE_1" "$GEMMA4_DISTINCT_IMAGE_2"
    echo ""
fi

echo "=== Gemma4 VL profile PASS ==="
echo "summary: $SUMMARY_TSV"
echo "metrics: $METRICS_TSV"
