#!/usr/bin/env bash
# Gemma4 vision semantic consistency smoke.
#
# Compares chunked prefill outputs against a non-chunked baseline for
# single-image, repeated multi-image, and distinct multi-image prompts.
# Repeated-image runs also assert exact image embedding dedup is active.
#
# Usage:
#   ./scripts/gemma4_vl_semantic_check.sh
#   ./scripts/gemma4_vl_semantic_check.sh --case single
#   ./scripts/gemma4_vl_semantic_check.sh --build
#
# Env:
#   MLX_DIR=$HOME/.local/mlx
#   IRONMLX_BIN=<repo>/target/release/ironmlx
#   CARGO_TARGET_DIR=<target-dir>      # used to derive IRONMLX_BIN
#   GEMMA4_MODEL=<snapshot-dir>
#   COMPARE_CHUNK_SIZES="256 64"
#   MAX_TOKENS=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECK_CASE="all"
BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CHECK_CASE="${2:-}"
            shift 2
            ;;
        --build)
            BUILD=1
            shift
            ;;
        -h|--help)
            sed -n '1,24p' "$0"
            exit 0
            ;;
        *)
            echo "[gemma4-vl-semantic] unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

case "$CHECK_CASE" in
    all|single|multi-repeat|multi-distinct)
        ;;
    *)
        echo "[gemma4-vl-semantic] --case must be one of: all, single, multi-repeat, multi-distinct" >&2
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
    echo "[gemma4-vl-semantic] ERROR: ironmlx binary not executable: $IRONMLX_BIN" >&2
    echo "[gemma4-vl-semantic]        run with --build or set IRONMLX_BIN/CARGO_TARGET_DIR" >&2
    exit 2
fi

GEMMA4_MODEL="${GEMMA4_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/* 2>/dev/null | head -1 || true)}"
if [[ -z "$GEMMA4_MODEL" || ! -d "$GEMMA4_MODEL" ]]; then
    echo "[gemma4-vl-semantic] ERROR: GEMMA4_MODEL not found: $GEMMA4_MODEL" >&2
    exit 2
fi

FIXTURE_IMAGE="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/coco_sample.jpg"
FIXTURE_DISTINCT_1="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg"
FIXTURE_DISTINCT_2="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg"

GEMMA4_IMAGE="${GEMMA4_IMAGE:-$FIXTURE_IMAGE}"
GEMMA4_DISTINCT_IMAGE_1="${GEMMA4_DISTINCT_IMAGE_1:-$FIXTURE_DISTINCT_1}"
GEMMA4_DISTINCT_IMAGE_2="${GEMMA4_DISTINCT_IMAGE_2:-$FIXTURE_DISTINCT_2}"
COMPARE_CHUNK_SIZES="${COMPARE_CHUNK_SIZES:-256 64}"
MAX_TOKENS="${MAX_TOKENS:-2}"
COOLDOWN_SECS="${COOLDOWN_SECS:-1}"
PROMPT="${PROMPT:-Describe this image in one short sentence.}"
MULTI_PROMPT="${MULTI_PROMPT:-Describe the images in one short sentence.}"

STAMP="$(date +%Y-%m-%d-%H%M%S)"
REPORT_DIR="$REPO_ROOT/reports/gemma4-vl-semantic/$STAMP"
mkdir -p "$REPORT_DIR"

SUMMARY_TSV="$REPORT_DIR/summary.tsv"
{
    printf 'case\tchunk_size\tbaseline_sha256\tcandidate_sha256\thidden_chunks\tslice_projects\tdedup_images\tdedup_unique\tdedup_duplicates\tverdict\tstdout_file\tstderr_file\n'
} > "$SUMMARY_TSV"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[gemma4-vl-semantic] ERROR: file not found: $1" >&2
        exit 2
    fi
}

dedup_field() {
    local line="$1"
    local field="$2"
    printf '%s\n' "$line" | sed -n "s/.*${field}=\\([0-9.][0-9.]*\\).*/\\1/p"
}

run_generate() {
    local name="$1"
    local chunk_size="$2"
    local prompt="$3"
    shift 3
    local images=("$@")

    local base="$REPORT_DIR/${name}-chunk${chunk_size}"
    local stdout_file="${base}.stdout.txt"
    local stderr_file="${base}.stderr.log"
    local command_file="${base}.command.txt"

    local image_args=()
    for image in "${images[@]}"; do
        require_file "$image"
        image_args+=(--image "$image")
    done

    {
        printf 'MLX_DIR=%q RUST_LOG=info IRONMLX_GEMMA4_VL_PROFILE=1 %q generate --model %q ' "$MLX_DIR" "$IRONMLX_BIN" "$GEMMA4_MODEL"
        printf '%q ' "${image_args[@]}"
        printf -- '--prompt %q --max-tokens %q --prefill-chunk-size %q\n' "$prompt" "$MAX_TOKENS" "$chunk_size"
    } > "$command_file"

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

    printf '%s\n' "$stdout_file"
}

assert_chunk_invariants() {
    local name="$1"
    local chunk_size="$2"
    local stderr_file="$3"

    local hidden_chunks slice_projects
    hidden_chunks="$(grep -c 'forward_vl_hidden_chunk_total_ms' "$stderr_file" || true)"
    slice_projects="$(grep -c 'forward_vl_slice_project_ms' "$stderr_file" || true)"

    if [[ "$chunk_size" != "0" ]]; then
        if [[ "$hidden_chunks" -lt 1 ]]; then
            echo "[gemma4-vl-semantic] ERROR: $name chunk_size=$chunk_size did not emit hidden chunk profile" >&2
            exit 1
        fi
        if [[ "$slice_projects" -ne 1 ]]; then
            echo "[gemma4-vl-semantic] ERROR: $name chunk_size=$chunk_size expected exactly one final slice project, got $slice_projects" >&2
            exit 1
        fi
    fi
}

assert_repeat_dedup() {
    local chunk_size="$1"
    local stderr_file="$2"

    local dedup_line dedup_images dedup_unique dedup_duplicates
    dedup_line="$(grep 'compute_vision_exact_dedup_ms' "$stderr_file" | tail -1 || true)"
    dedup_images="$(dedup_field "$dedup_line" 'images')"
    dedup_unique="$(dedup_field "$dedup_line" 'unique')"
    dedup_duplicates="$(dedup_field "$dedup_line" 'duplicates')"

    if [[ "$dedup_images" != "2" || "$dedup_unique" != "1" || "$dedup_duplicates" != "1" ]]; then
        echo "[gemma4-vl-semantic] ERROR: multi-repeat chunk_size=$chunk_size expected dedup images=2 unique=1 duplicates=1" >&2
        echo "[gemma4-vl-semantic]        got: ${dedup_line:-<missing>}" >&2
        exit 1
    fi
}

append_summary() {
    local name="$1"
    local chunk_size="$2"
    local baseline_file="$3"
    local candidate_file="$4"
    local stderr_file="$5"
    local verdict="$6"

    local baseline_sha candidate_sha hidden_chunks slice_projects dedup_line
    baseline_sha="$(shasum -a 256 "$baseline_file" | awk '{print $1}')"
    candidate_sha="$(shasum -a 256 "$candidate_file" | awk '{print $1}')"
    hidden_chunks="$(grep -c 'forward_vl_hidden_chunk_total_ms' "$stderr_file" || true)"
    slice_projects="$(grep -c 'forward_vl_slice_project_ms' "$stderr_file" || true)"
    dedup_line="$(grep 'compute_vision_exact_dedup_ms' "$stderr_file" | tail -1 || true)"

    local dedup_images dedup_unique dedup_duplicates
    dedup_images="$(dedup_field "$dedup_line" 'images')"
    dedup_unique="$(dedup_field "$dedup_line" 'unique')"
    dedup_duplicates="$(dedup_field "$dedup_line" 'duplicates')"
    dedup_images="${dedup_images:--}"
    dedup_unique="${dedup_unique:--}"
    dedup_duplicates="${dedup_duplicates:--}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$chunk_size" "$baseline_sha" "$candidate_sha" "$hidden_chunks" "$slice_projects" \
        "$dedup_images" "$dedup_unique" "$dedup_duplicates" "$verdict" "$candidate_file" "$stderr_file" \
        >> "$SUMMARY_TSV"
}

run_case() {
    local name="$1"
    local prompt="$2"
    shift 2
    local images=("$@")

    echo "=== $name ==="
    echo "  images: ${images[*]}"

    local baseline_stdout baseline_stderr baseline_preview
    baseline_stdout="$(run_generate "$name" 0 "$prompt" "${images[@]}")"
    baseline_stderr="${baseline_stdout%.stdout.txt}.stderr.log"
    baseline_preview="$(tr '\n' ' ' < "$baseline_stdout" | cut -c 1-100)"
    echo "  baseline chunk_size=0 output: $baseline_preview"
    append_summary "$name" 0 "$baseline_stdout" "$baseline_stdout" "$baseline_stderr" "baseline"
    if [[ "$name" == "multi-repeat" ]]; then
        assert_repeat_dedup 0 "$baseline_stderr"
    fi

    for chunk_size in $COMPARE_CHUNK_SIZES; do
        local candidate_stdout candidate_stderr diff_file preview
        candidate_stdout="$(run_generate "$name" "$chunk_size" "$prompt" "${images[@]}")"
        candidate_stderr="${candidate_stdout%.stdout.txt}.stderr.log"
        diff_file="$REPORT_DIR/${name}-chunk${chunk_size}.diff"
        preview="$(tr '\n' ' ' < "$candidate_stdout" | cut -c 1-100)"

        assert_chunk_invariants "$name" "$chunk_size" "$candidate_stderr"
        if [[ "$name" == "multi-repeat" ]]; then
            assert_repeat_dedup "$chunk_size" "$candidate_stderr"
        fi

        if cmp -s "$baseline_stdout" "$candidate_stdout"; then
            append_summary "$name" "$chunk_size" "$baseline_stdout" "$candidate_stdout" "$candidate_stderr" "match"
            echo "  chunk_size=$chunk_size MATCH output: $preview"
        else
            diff -u "$baseline_stdout" "$candidate_stdout" > "$diff_file" || true
            append_summary "$name" "$chunk_size" "$baseline_stdout" "$candidate_stdout" "$candidate_stderr" "mismatch"
            echo "[gemma4-vl-semantic] ERROR: $name chunk_size=$chunk_size output mismatch; diff: $diff_file" >&2
            exit 1
        fi

        sleep "$COOLDOWN_SECS"
    done
}

echo "=== Gemma4 VL semantic check — $(date) ==="
echo "report: $REPORT_DIR"
echo "binary: $IRONMLX_BIN"
echo "model:  $GEMMA4_MODEL"
echo "compare chunk sizes: $COMPARE_CHUNK_SIZES"
echo ""

if [[ "$CHECK_CASE" == "all" || "$CHECK_CASE" == "single" ]]; then
    run_case "single" "$PROMPT" "$GEMMA4_IMAGE"
    echo ""
fi

if [[ "$CHECK_CASE" == "all" || "$CHECK_CASE" == "multi-repeat" ]]; then
    run_case "multi-repeat" "$MULTI_PROMPT" "$GEMMA4_IMAGE" "$GEMMA4_IMAGE"
    echo ""
fi

if [[ "$CHECK_CASE" == "all" || "$CHECK_CASE" == "multi-distinct" ]]; then
    run_case "multi-distinct" "$MULTI_PROMPT" "$GEMMA4_DISTINCT_IMAGE_1" "$GEMMA4_DISTINCT_IMAGE_2"
    echo ""
fi

echo "=== Gemma4 VL semantic check PASS ==="
echo "summary: $SUMMARY_TSV"
