#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MODEL_ROOT="$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots"

if [[ -z "${MODEL:-}" ]]; then
  if [[ ! -d "$DEFAULT_MODEL_ROOT" ]]; then
    echo "MODEL is not set and default model root does not exist: $DEFAULT_MODEL_ROOT" >&2
    exit 1
  fi
  MODEL="$(find "$DEFAULT_MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "$MODEL" || ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 1
fi

STAMP="${STAMP:-$(date +%Y-%m-%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/docs/benchmarks/turboquant-kv/$STAMP}"
MAX_TOKENS="${MAX_TOKENS:-16}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-32}"
RUNS="${RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
export MLX_DIR

IFS=',' read -r -a KV_QUANTS <<< "${KV_QUANTS:-none,turbo3,turbo4,k3v4}"
KV_JOINED="$(IFS=,; echo "${KV_QUANTS[*]}")"

mkdir -p "$OUT_DIR/prompts" "$OUT_DIR/core-bench"

if [[ -n "${PROMPT_FILE:-}" ]]; then
  PROMPT="$PROMPT_FILE"
else
  PROMPT="$OUT_DIR/prompts/technical_summary.txt"
  cat > "$PROMPT" <<'PROMPT_TEXT'
You are validating a KV cache quantization implementation. Write a compact technical summary with three numbered points about how to compare logits drift, generation quality, latency, and memory usage.
PROMPT_TEXT
fi

cargo build --release \
  -p ironmlx \
  --bin ironmlx-turboquant-kv-validate \
  --bin ironmlx-core-bench

"$ROOT/target/release/ironmlx-turboquant-kv-validate" \
  --model "$MODEL" \
  --prompt-file "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --kv-quant "$KV_JOINED" \
  --out "$OUT_DIR/logits.json"

for kv in "${KV_QUANTS[@]}"; do
  /usr/bin/time -l "$ROOT/target/release/ironmlx-core-bench" \
    --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --mode gs-text \
    --max-tokens "$BENCH_MAX_TOKENS" \
    --runs "$RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --kv-quant "$kv" \
    --out "$OUT_DIR/core-bench/$kv.json" \
    > "$OUT_DIR/core-bench/$kv.stdout" \
    2> "$OUT_DIR/core-bench/$kv.time.txt"
done

python3 - "$OUT_DIR" "${KV_QUANTS[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
order = sys.argv[2:]
logits = json.loads((out_dir / "logits.json").read_text())

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}f}"

def one_line(text, limit=160):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def mean(values):
    return sum(values) / len(values) if values else None

def max_rss_gib(time_path):
    if not time_path.exists():
        return None
    rss = None
    for line in time_path.read_text(errors="replace").splitlines():
        if "maximum resident set size" in line:
            parts = line.strip().split()
            if parts:
                try:
                    rss = int(parts[0])
                except ValueError:
                    pass
    if rss is None:
        return None
    return rss / (1024 ** 3)

bench = {}
for kv in order:
    path = out_dir / "core-bench" / f"{kv}.json"
    if path.exists():
        bench[kv] = json.loads(path.read_text())

baseline_tokens = None
if "none" in bench and bench["none"].get("records"):
    baseline_tokens = bench["none"]["records"][0].get("generated_token_ids", [])

lines = []
lines.append("# TurboQuant KV Validation")
lines.append("")
lines.append(f"- model: `{logits['meta']['model_dir']}`")
lines.append(f"- prompt: `{logits['meta']['prompt_file']}`")
lines.append(f"- prompt_tokens: `{logits['meta']['prompt_tokens']}`")
lines.append(f"- logits_max_tokens: `{logits['meta']['max_tokens']}`")
lines.append("")

lines.append("## Logits Replay")
lines.append("")
lines.append("| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |")
lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for cfg in logits["configs"]:
    steps = cfg["steps"]
    label = cfg["label"]
    argmax_matches = sum(1 for step in steps if step["argmax_matches"])
    first_mismatch = cfg["first_token_mismatch_step"]
    lines.append(
        "| {label} | {exact} | {mismatch} | {argmax}/{total} | {max_abs} | {mean_abs} | {rms} | {cosine} | {top5} |".format(
            label=label,
            exact=cfg["exact_token_match_count"],
            mismatch="none" if first_mismatch is None else first_mismatch,
            argmax=argmax_matches,
            total=len(steps),
            max_abs=fmt(mean([step["max_abs_diff"] for step in steps]), 6),
            mean_abs=fmt(mean([step["mean_abs_diff"] for step in steps]), 6),
            rms=fmt(mean([step["rms_diff"] for step in steps]), 6),
            cosine=fmt(min(step["cosine_similarity"] for step in steps), 9),
            top5=fmt(mean([step["top5_overlap"] for step in steps]), 2),
        )
    )
lines.append("")

lines.append("## Core Generation")
lines.append("")
lines.append("| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | exact-prefix vs none |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for kv in order:
    data = bench.get(kv)
    if not data:
        continue
    records = data.get("records", [])
    tokens = records[0].get("generated_token_ids", []) if records else []
    if baseline_tokens is None:
        exact = "n/a"
    else:
        exact_count = 0
        for left, right in zip(tokens, baseline_tokens):
            if left != right:
                break
            exact_count += 1
        exact = str(exact_count)
    summary = data["summary"]
    lines.append(
        "| {kv} | {tokens} | {ttft} | {e2e} | {tps} | {rss} | {exact} |".format(
            kv=kv,
            tokens=records[0].get("generated_tokens", 0) if records else 0,
            ttft=fmt(summary["ttft_ms"]["p50"], 2),
            e2e=fmt(summary["e2e_ms"]["p50"], 2),
            tps=fmt(summary["generation_tps"]["p50"], 2),
            rss=fmt(max_rss_gib(out_dir / "core-bench" / f"{kv}.time.txt"), 3),
            exact=exact,
        )
    )
lines.append("")

lines.append("## Generated Text")
lines.append("")
for kv in order:
    data = bench.get(kv)
    if not data or not data.get("records"):
        continue
    lines.append(f"- `{kv}`: {one_line(data['records'][0].get('generated_text', ''))}")
lines.append("")

lines.append("## Raw Artifacts")
lines.append("")
lines.append("- logits replay: `logits.json`")
lines.append("- benchmark JSON: `core-bench/<kv>.json`")
lines.append("- benchmark RSS/time: `core-bench/<kv>.time.txt`")

(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print(out_dir / "summary.md")
PY
