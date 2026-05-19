#!/bin/bash
# P5d T3: cross-prompt greedy alignment vs omlx HTTP baseline.
# Serial: only one server up at a time. Both run on 35B-A3B-4bit snapshot.
# Both sides use /v1/chat/completions with identical user prompts.
# enable_thinking=false: bypass thinking mode to avoid logit near-ties in
# the long thinking path that cause legitimate divergence between independent
# inference implementations (same model weights, different numerical paths).
# Chat template + tokenizer + decoder pipeline is still fully exercised.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR=${IRONMLX_MOE_MODEL_DIR:-$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec}
PROMPTS="$SCRIPT_DIR/p5d_prompts.txt"
OUT_DIR="$REPO_ROOT/reports/p5d-argmax"
IRONMLX_BIN="$REPO_ROOT/target/release/ironmlx"
OMLX_DIR="/Users/xin/workspace/iron-rivals/omlx"

IRONMLX_PORT=8092
OMLX_PORT=8093

mkdir -p "$OUT_DIR"
echo "[P5d-T3] Model: $MODEL_DIR"
echo "[P5d-T3] Prompts: $PROMPTS"
echo "[P5d-T3] Output: $OUT_DIR"
echo ""

# ========== Phase A: ironmlx HTTP serve + chat/completions ==========
echo "[P5d-T3] Phase A: launching ironmlx serve on port $IRONMLX_PORT"
MLX_DIR=$HOME/.local/mlx "$IRONMLX_BIN" serve \
    --model "$MODEL_DIR" \
    --port "$IRONMLX_PORT" \
    > "$OUT_DIR/ironmlx-serve.log" 2>&1 &
IRONMLX_PID=$!
echo "  ironmlx pid: $IRONMLX_PID"

# Wait for ironmlx healthy (up to 120s for model load)
READY=0
for s in $(seq 1 120); do
    if curl -fsS "http://localhost:$IRONMLX_PORT/health" >/dev/null 2>&1; then
        echo "  ironmlx ready in ${s}s"
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: ironmlx did not become ready within 120s"
    kill "$IRONMLX_PID" 2>/dev/null || true
    exit 1
fi

# Query ironmlx for each prompt
> "$OUT_DIR/ironmlx.jsonl"
i=0
while IFS= read -r raw_prompt; do
    # Unescape \n literals in the prompt text
    prompt=$(printf '%b' "$raw_prompt")
    echo "  ironmlx prompt $i: ${raw_prompt:0:60}..."
    body=$(jq -nc --arg p "$prompt" \
        '{model:"ironmlx",messages:[{role:"user",content:$p}],max_tokens:200,temperature:0.0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    resp=$(curl -fsS -X POST "http://localhost:$IRONMLX_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null)
    out=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // "<ERROR>"' 2>/dev/null || echo "<ERROR>")
    printf '{"idx":%d,"prompt":%s,"output":%s}\n' \
        "$i" \
        "$(printf '%s' "$prompt" | jq -Rs .)" \
        "$(printf '%s' "$out" | jq -Rs .)" \
        >> "$OUT_DIR/ironmlx.jsonl"
    i=$((i+1))
done < "$PROMPTS"

# Stop ironmlx
echo "  ironmlx: stopping (pid $IRONMLX_PID)"
kill "$IRONMLX_PID" 2>/dev/null || true
wait "$IRONMLX_PID" 2>/dev/null || true
echo "[P5d-T3] Phase A done ($i prompts)"
echo ""

# Short cooldown before starting omlx
sleep 3

# ========== Phase B: omlx serve + chat/completions ==========
echo "[P5d-T3] Phase B: launching omlx serve on port $OMLX_PORT"

# Create symlink so omlx can serve the model by directory name
OMLX_MODEL_DIR="/tmp/ironmlx-p5d-omlx-models"
mkdir -p "$OMLX_MODEL_DIR"
ln -sfn "$MODEL_DIR" "$OMLX_MODEL_DIR/qwen3_5_moe"

cd "$OMLX_DIR"
uv run --with-editable . omlx serve \
    --model-dir "$OMLX_MODEL_DIR" \
    --port "$OMLX_PORT" \
    > "$OUT_DIR/omlx-serve.log" 2>&1 &
OMLX_PID=$!
echo "  omlx pid: $OMLX_PID"
cd "$REPO_ROOT"

# Wait for omlx healthy (up to 180s for cold load)
READY=0
for s in $(seq 1 180); do
    if curl -fsS "http://localhost:$OMLX_PORT/v1/models" >/dev/null 2>&1; then
        echo "  omlx ready in ${s}s"
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: omlx did not become ready within 180s"
    kill "$OMLX_PID" 2>/dev/null || true
    exit 1
fi

# Query omlx for each prompt
> "$OUT_DIR/omlx.jsonl"
i=0
while IFS= read -r raw_prompt; do
    prompt=$(printf '%b' "$raw_prompt")
    echo "  omlx prompt $i: ${raw_prompt:0:60}..."
    body=$(jq -nc --arg p "$prompt" \
        '{model:"qwen3_5_moe",messages:[{role:"user",content:$p}],max_tokens:200,temperature:0.0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    resp=$(curl -fsS -X POST "http://localhost:$OMLX_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null)
    out=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // "<ERROR>"' 2>/dev/null || echo "<ERROR>")
    printf '{"idx":%d,"prompt":%s,"output":%s}\n' \
        "$i" \
        "$(printf '%s' "$prompt" | jq -Rs .)" \
        "$(printf '%s' "$out" | jq -Rs .)" \
        >> "$OUT_DIR/omlx.jsonl"
    i=$((i+1))
done < "$PROMPTS"

# Stop omlx
echo "  omlx: stopping (pid $OMLX_PID)"
kill "$OMLX_PID" 2>/dev/null || true
wait "$OMLX_PID" 2>/dev/null || true
echo "[P5d-T3] Phase B done ($i prompts)"
echo ""
echo "[P5d-T3] Harness complete. Run: python3 scripts/p5d_compare_argmax.py"
