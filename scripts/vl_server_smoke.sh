#!/usr/bin/env bash
# OpenAI server VL smoke for Gemma4 and Qwen3.5.
#
# Defaults discover local checkpoints under ~/.ironmlx/models and use the
# checked-in P6 image fixture. Override paths and ports with env vars.
#
# Usage:
#   ./scripts/vl_server_smoke.sh
#   ./scripts/vl_server_smoke.sh --case gemma4
#   ./scripts/vl_server_smoke.sh --case gemma4-multi
#   ./scripts/vl_server_smoke.sh --case qwen
#   ./scripts/vl_server_smoke.sh --build
#
# Env:
#   MLX_DIR=$HOME/.local/mlx
#   IRONMLX_BIN=<repo>/target/release/ironmlx
#   CARGO_TARGET_DIR=<target-dir>      # used to derive IRONMLX_BIN
#   GEMMA4_MODEL=<snapshot-dir>
#   QWEN35_MODEL=<snapshot-dir>
#   GEMMA4_IMAGE=<image-path>
#   GEMMA4_IMAGE_2=<image-path>
#   QWEN35_IMAGE=<image-path>
#   GEMMA4_PORT=18178
#   QWEN35_PORT=18179
#   MAX_TOKENS=8
#   MIN_PROMPT_TOKENS=64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SMOKE_CASE="all"
BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            SMOKE_CASE="${2:-}"
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
            echo "[vl-smoke] unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

case "$SMOKE_CASE" in
    all|gemma4|gemma4-multi|qwen)
        ;;
    *)
        echo "[vl-smoke] --case must be one of: all, gemma4, gemma4-multi, qwen" >&2
        exit 2
        ;;
esac

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[vl-smoke] ERROR: required command not found: $1" >&2
        exit 2
    fi
}

require_cmd base64
require_cmd curl
require_cmd jq
require_cmd lsof

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
    echo "[vl-smoke] ERROR: ironmlx binary not executable: $IRONMLX_BIN" >&2
    echo "[vl-smoke]        run with --build or set IRONMLX_BIN/CARGO_TARGET_DIR" >&2
    exit 2
fi

GEMMA4_MODEL="${GEMMA4_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/* 2>/dev/null | head -1 || true)}"
QWEN35_MODEL="${QWEN35_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/* 2>/dev/null | head -1 || true)}"

FIXTURE_IMAGE="$REPO_ROOT/ironmlx/tests/fixtures/qwen35_vl/coco_sample.jpg"
GEMMA4_IMAGE="${GEMMA4_IMAGE:-$FIXTURE_IMAGE}"
GEMMA4_IMAGE_2="${GEMMA4_IMAGE_2:-$GEMMA4_IMAGE}"
QWEN35_IMAGE="${QWEN35_IMAGE:-$FIXTURE_IMAGE}"

GEMMA4_PORT="${GEMMA4_PORT:-18178}"
QWEN35_PORT="${QWEN35_PORT:-18179}"
MAX_TOKENS="${MAX_TOKENS:-8}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-64}"
MAX_CACHE_CAP="${MAX_CACHE_CAP:-4096}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-180}"
COOLDOWN_SECS="${COOLDOWN_SECS:-3}"
PROMPT="${PROMPT:-Describe this image in one short sentence.}"
MULTI_PROMPT="${MULTI_PROMPT:-Describe the two images in one short sentence.}"

STAMP="$(date +%Y-%m-%d-%H%M%S)"
REPORT_DIR="$REPO_ROOT/reports/vl-server-smoke/$STAMP"
mkdir -p "$REPORT_DIR"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

log() {
    echo "$@"
}

wait_ready() {
    local name="$1"
    local port="$2"
    local log_file="$3"

    for second in $(seq 1 "$READY_TIMEOUT_SECS"); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            log "  ready in ${second}s"
            return 0
        fi
        sleep 1
    done

    log "  ERROR: $name did not become ready within ${READY_TIMEOUT_SECS}s"
    tail -40 "$log_file" | sed 's/^/    /' || true
    return 1
}

run_smoke() {
    local name="$1"
    local model_dir="$2"
    local image="$3"
    local port="$4"
    local image2="${5:-}"

    if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
        echo "[vl-smoke] ERROR: $name model dir not found: $model_dir" >&2
        exit 2
    fi
    if [[ ! -f "$image" ]]; then
        echo "[vl-smoke] ERROR: $name image not found: $image" >&2
        exit 2
    fi
    if [[ -n "$image2" && ! -f "$image2" ]]; then
        echo "[vl-smoke] ERROR: $name second image not found: $image2" >&2
        exit 2
    fi
    if lsof -ti "tcp:$port" >/dev/null 2>&1; then
        echo "[vl-smoke] ERROR: port $port is already in use for $name" >&2
        exit 2
    fi

    local server_log="$REPORT_DIR/${name}-server.log"
    local response_json="$REPORT_DIR/${name}-response.json"
    local healthz_json="$REPORT_DIR/${name}-healthz.json"

    log "=== $name ==="
    log "  model: $model_dir"
    log "  image: $image"
    if [[ -n "$image2" ]]; then
        log "  image2: $image2"
    fi
    log "  port:  $port"

    MLX_DIR="$MLX_DIR" "$IRONMLX_BIN" serve \
        --model "$model_dir" \
        --host 127.0.0.1 \
        --port "$port" \
        --prefill-chunk-size 0 \
        --max-cache-cap "$MAX_CACHE_CAP" \
        > "$server_log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    log "  pid:   $pid"

    wait_ready "$name" "$port" "$server_log"

    local b64
    b64="$(base64 -i "$image" | tr -d '\n')"

    local body
    if [[ -n "$image2" ]]; then
        local b64_2
        b64_2="$(base64 -i "$image2" | tr -d '\n')"
        body="$(jq -nc \
            --arg model "$name-smoke" \
            --arg prompt "$MULTI_PROMPT" \
            --arg image_url "data:image/jpeg;base64,$b64" \
            --arg image_url_2 "data:image/jpeg;base64,$b64_2" \
            --argjson max_tokens "$MAX_TOKENS" \
            '{
                model:$model,
                messages:[{
                    role:"user",
                    content:[
                        {type:"text", text:$prompt},
                        {type:"image_url", image_url:{url:$image_url}},
                        {type:"text", text:" Second image:"},
                        {type:"image_url", image_url:{url:$image_url_2}}
                    ]
                }],
                max_tokens:$max_tokens,
                stream:false
            }')"
    else
        body="$(jq -nc \
            --arg model "$name-smoke" \
            --arg prompt "$PROMPT" \
            --arg image_url "data:image/jpeg;base64,$b64" \
            --argjson max_tokens "$MAX_TOKENS" \
            '{
                model:$model,
                messages:[{
                    role:"user",
                    content:[
                        {type:"text", text:$prompt},
                        {type:"image_url", image_url:{url:$image_url}}
                    ]
                }],
                max_tokens:$max_tokens,
                stream:false
            }')"
    fi

    curl -fsS \
        -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" \
        > "$response_json"

    curl -sS "http://127.0.0.1:$port/healthz" > "$healthz_json" || true

    local content prompt_tokens completion_tokens finish_reason
    content="$(jq -r '.choices[0].message.content // ""' "$response_json")"
    prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$response_json")"
    completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$response_json")"
    finish_reason="$(jq -r '.choices[0].finish_reason // ""' "$response_json")"

    if [[ -z "$content" ]]; then
        log "  ERROR: empty assistant content"
        tail -40 "$server_log" | sed 's/^/    /' || true
        exit 1
    fi
    if [[ "$prompt_tokens" -lt "$MIN_PROMPT_TOKENS" ]]; then
        log "  ERROR: prompt_tokens=$prompt_tokens below MIN_PROMPT_TOKENS=$MIN_PROMPT_TOKENS"
        exit 1
    fi

    log "  PASS: prompt_tokens=$prompt_tokens completion_tokens=$completion_tokens finish_reason=$finish_reason"
    log "  output: $(printf '%s' "$content" | tr '\n' ' ' | cut -c 1-120)"

    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
    sleep "$COOLDOWN_SECS"
}

log "=== VL server smoke — $(date) ==="
log "report: $REPORT_DIR"
log "binary: $IRONMLX_BIN"
log ""

if [[ "$SMOKE_CASE" == "all" || "$SMOKE_CASE" == "gemma4" ]]; then
    run_smoke "gemma4" "$GEMMA4_MODEL" "$GEMMA4_IMAGE" "$GEMMA4_PORT"
    log ""
fi

if [[ "$SMOKE_CASE" == "all" || "$SMOKE_CASE" == "gemma4-multi" ]]; then
    run_smoke "gemma4-multi" "$GEMMA4_MODEL" "$GEMMA4_IMAGE" "$GEMMA4_PORT" "$GEMMA4_IMAGE_2"
    log ""
fi

if [[ "$SMOKE_CASE" == "all" || "$SMOKE_CASE" == "qwen" ]]; then
    run_smoke "qwen" "$QWEN35_MODEL" "$QWEN35_IMAGE" "$QWEN35_PORT"
    log ""
fi

log "=== VL server smoke PASS ==="
log "report: $REPORT_DIR"
