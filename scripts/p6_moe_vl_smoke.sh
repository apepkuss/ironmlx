#!/bin/bash
# Qwen3.5 MoE VL smoke matrix.
#
# Covers:
#   - single-image unary OpenAI chat request
#   - single-image SSE request
#   - GS chunked SSE path via --prefill-chunk-size 256
#   - --b-max 2 mixed concurrency: VL + VL + text-only
#   - existing 2-image and 3-image semantic checks
#
# Env:
#   IRONMLX_MOE_VL_MODEL_DIR  Local MoE VL snapshot path.
#   MLX_DIR                   MLX install prefix. Defaults to ~/.local/mlx.
#   IRONMLX_BIN               ironmlx binary. Defaults to target/release/ironmlx.
#   BASE_PORT                 First local port. Defaults to 18190.
#   OUT_DIR                   Report/log dir. Defaults to /tmp/ironmlx-p6-moe-vl-smoke.
#   SKIP_BUILD=1              Reuse existing binary instead of cargo build --release -p ironmlx.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${IRONMLX_MOE_VL_MODEL_DIR:-$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec}"
MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
IRONMLX_BIN="${IRONMLX_BIN:-$REPO_ROOT/target/release/ironmlx}"
BASE_PORT="${BASE_PORT:-18190}"
OUT_DIR="${OUT_DIR:-/tmp/ironmlx-p6-moe-vl-smoke}"
SKIP_BUILD="${SKIP_BUILD:-0}"

FIXTURE_DIR="$REPO_ROOT/ironmlx/tests/fixtures/p6_qwen35_vl"
PY_HELPER="$OUT_DIR/p6_moe_vl_smoke_client.py"

SERVER_PID=""

log() {
  echo "[p6-moe-vl] $*"
}

fail() {
  echo "[p6-moe-vl] ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_health() {
  local port="$1"
  local pid="$2"
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 2
    fi
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

port_in_use() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

start_server() {
  local port="$1"
  shift
  local log_path="$OUT_DIR/server-$port.log"
  cleanup
  SERVER_PID=""
  if port_in_use "$port"; then
    fail "port $port is already in use; choose a free BASE_PORT"
  fi
  log "starting server on port $port: $*"
  MLX_DIR="$MLX_DIR" "$IRONMLX_BIN" serve \
    --model "$MODEL_DIR" \
    --host 127.0.0.1 \
    --port "$port" \
    "$@" >"$log_path" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_health "$port" "$SERVER_PID"; then
    tail -80 "$log_path" >&2 || true
    fail "server on port $port did not become healthy"
  fi
}

write_python_helper() {
  mkdir -p "$OUT_DIR"
  cat >"$PY_HELPER" <<'PY'
import base64
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def image_part(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def post_json(port, payload, stream=False):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        if not stream:
            return resp.status, json.loads(resp.read().decode("utf-8"))
        chunks = []
        finish = None
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            choice = obj.get("choices", [{}])[0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if "content" in delta:
                chunks.append(delta["content"])
        return resp.status, {"text": "".join(chunks), "chunks": len(chunks), "finish": finish}


def common_payload(content, max_tokens=32, stream=False):
    return {
        "model": "qwen3_5_moe",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
    }


def assert_text(name, status, text):
    if status != 200 or not text.strip():
        raise RuntimeError(f"{name}: status={status}, text_len={len(text)}")


def run_unary(port, fixture_dir):
    content = [
        {"type": "text", "text": "Describe this image in one short sentence."},
        image_part(str(Path(fixture_dir) / "coco_sample.jpg")),
    ]
    status, body = post_json(port, common_payload(content, max_tokens=32), stream=False)
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    assert_text("single_image_unary", status, text)
    print(json.dumps({
        "case": "single_image_unary",
        "status": status,
        "finish": body.get("choices", [{}])[0].get("finish_reason"),
        "usage": body.get("usage"),
        "content_len": len(text),
        "preview": text[:160],
    }, ensure_ascii=False))


def run_stream(port, fixture_dir, case):
    content = [
        {"type": "text", "text": "Describe this image briefly."},
        image_part(str(Path(fixture_dir) / "coco_sample.jpg")),
    ]
    status, body = post_json(port, common_payload(content, max_tokens=24, stream=True), stream=True)
    text = body["text"]
    assert_text(case, status, text)
    print(json.dumps({
        "case": case,
        "status": status,
        "finish": body["finish"],
        "chunks": body["chunks"],
        "content_len": len(text),
        "preview": text[:160],
    }, ensure_ascii=False))


def run_multi_image(port, fixture_dir):
    multi_dir = Path(fixture_dir) / "multi_image"
    content = [{"type": "text", "text": "Describe both images briefly, one sentence each."}]
    content.append(image_part(str(multi_dir / "image_0.jpg")))
    content.append(image_part(str(multi_dir / "image_1.jpg")))
    status, body = post_json(port, common_payload(content, max_tokens=48), stream=False)
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    assert_text("multi_image_unary", status, text)
    print(json.dumps({
        "case": "multi_image_unary",
        "status": status,
        "finish": body.get("choices", [{}])[0].get("finish_reason"),
        "usage": body.get("usage"),
        "content_len": len(text),
        "preview": text[:160],
    }, ensure_ascii=False))


def run_concurrency(port, fixture_dir):
    barrier = threading.Barrier(3)
    fixture_dir = Path(fixture_dir)
    cases = [
        ("vl_cats", common_payload([
            {"type": "text", "text": "Describe this image in one short sentence."},
            image_part(str(fixture_dir / "coco_sample.jpg")),
        ], max_tokens=32)),
        ("vl_kitchen", common_payload([
            {"type": "text", "text": "Describe this image in one short sentence."},
            image_part(str(fixture_dir / "multi_image" / "image_0.jpg")),
        ], max_tokens=32)),
        ("text_only", common_payload("Reply with exactly five words about reliable systems.", max_tokens=32)),
    ]

    def run_case(name, payload):
        barrier.wait(timeout=10)
        start = time.time()
        status, body = post_json(port, payload, stream=False)
        elapsed = time.time() - start
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        assert_text(name, status, text)
        return {
            "case": f"concurrency_{name}",
            "status": status,
            "finish": body.get("choices", [{}])[0].get("finish_reason"),
            "usage": body.get("usage"),
            "elapsed_s": round(elapsed, 3),
            "content_len": len(text),
            "preview": text[:160].replace("\n", " "),
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_case, name, payload) for name, payload in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False))


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: helper.py <case> <port> <fixture_dir>")
    case = sys.argv[1]
    port = int(sys.argv[2])
    fixture_dir = sys.argv[3]
    if case == "unary":
        run_unary(port, fixture_dir)
    elif case == "stream":
        run_stream(port, fixture_dir, "single_image_sse")
    elif case == "gs_stream":
        run_stream(port, fixture_dir, "gs_chunked_sse")
    elif case == "multi":
        run_multi_image(port, fixture_dir)
    elif case == "concurrency":
        run_concurrency(port, fixture_dir)
    else:
        raise SystemExit(f"unknown case: {case}")


if __name__ == "__main__":
    main()
PY
}

[ -d "$MODEL_DIR" ] || fail "model dir not found: $MODEL_DIR"
[ -d "$MLX_DIR" ] || fail "MLX_DIR not found: $MLX_DIR"

mkdir -p "$OUT_DIR"
write_python_helper

if [ "$SKIP_BUILD" != "1" ]; then
  log "building release binary"
  (cd "$REPO_ROOT" && MLX_DIR="$MLX_DIR" cargo build --release -p ironmlx)
fi

[ -x "$IRONMLX_BIN" ] || fail "ironmlx binary not executable: $IRONMLX_BIN"

PORT_A="$BASE_PORT"
PORT_B=$((BASE_PORT + 1))
PORT_C=$((BASE_PORT + 2))
PORT_D=$((BASE_PORT + 3))
PORT_E=$((BASE_PORT + 4))

SUMMARY="$OUT_DIR/summary.jsonl"
: >"$SUMMARY"

log "model: $MODEL_DIR"
log "output: $OUT_DIR"

start_server "$PORT_A"
python3 "$PY_HELPER" unary "$PORT_A" "$FIXTURE_DIR" | tee -a "$SUMMARY"
python3 "$PY_HELPER" stream "$PORT_A" "$FIXTURE_DIR" | tee -a "$SUMMARY"
python3 "$PY_HELPER" multi "$PORT_A" "$FIXTURE_DIR" | tee -a "$SUMMARY"
cleanup
SERVER_PID=""

start_server "$PORT_B" --prefill-chunk-size 256
python3 "$PY_HELPER" gs_stream "$PORT_B" "$FIXTURE_DIR" | tee -a "$SUMMARY"
cleanup
SERVER_PID=""

start_server "$PORT_C" --b-max 2 --admission-deadline-ms 50
python3 "$PY_HELPER" concurrency "$PORT_C" "$FIXTURE_DIR" | tee -a "$SUMMARY"
cleanup
SERVER_PID=""

log "running semantic check N=2"
QWEN35_MODEL="$MODEL_DIR" MLX_DIR="$MLX_DIR" \
  uv run --with requests python "$FIXTURE_DIR/p6_6_semantic_check.py" \
  --out "$OUT_DIR/p6_6_semantic_n2.md" \
  --port "$PORT_D" \
  --model-name qwen3_5_moe \
  --n-images 2

log "running semantic check N=3"
QWEN35_MODEL="$MODEL_DIR" MLX_DIR="$MLX_DIR" \
  uv run --with requests python "$FIXTURE_DIR/p6_6_semantic_check.py" \
  --out "$OUT_DIR/p6_6_semantic_n3.md" \
  --port "$PORT_E" \
  --model-name qwen3_5_moe \
  --n-images 3

log "PASS. Summary: $SUMMARY"
