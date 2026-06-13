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

STAMP="${STAMP:-$(date +%Y-%m-%d-%H%M%S)-long-context}"
OUT_ROOT="${OUT_ROOT:-$ROOT/docs/benchmarks/turboquant-kv/$STAMP}"
MAX_TOKENS="${MAX_TOKENS:-32}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-32}"
RUNS="${RUNS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-0}"
MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
export MLX_DIR

IFS=',' read -r -a KV_QUANTS <<< "${KV_QUANTS:-none,turbo4,k3v4}"
KV_JOINED="$(IFS=,; echo "${KV_QUANTS[*]}")"

# label:record-count pairs. The actual tokenizer length is recorded in logits.json.
IFS=',' read -r -a CONTEXTS <<< "${CONTEXTS:-ctx-4k:88,ctx-8k:176,ctx-16k:352,ctx-32k:704}"
CONTEXTS_FOR_SUMMARY="$(IFS=,; echo "${CONTEXTS[*]}")"
export CONTEXTS_FOR_SUMMARY

mkdir -p "$OUT_ROOT/prompts"

cargo build --release \
  -p ironmlx \
  --bin ironmlx-turboquant-kv-validate \
  --bin ironmlx-core-bench

for context in "${CONTEXTS[@]}"; do
  label="${context%%:*}"
  records="${context##*:}"
  prompt="$OUT_ROOT/prompts/$label.txt"
  out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir/core-bench"

  python3 - "$prompt" "$records" <<'PY'
import sys
from pathlib import Path

prompt = Path(sys.argv[1])
records = int(sys.argv[2])

lines = [
    "You are validating long-context KV cache stability for TurboQuant.\n",
    "Read every record. The final answer must use the last Record line only.\n",
    "Return one line in this exact schema: CHECKSUM record=<id> alpha=<alpha> beta=<beta> gamma=<gamma>\n",
    "\n",
]
for i in range(1, records + 1):
    alpha = i % 17
    beta = i % 29
    gamma = i % 43
    lines.append(
        f"Record {i:05d}: alpha={alpha:02d}; beta={beta:02d}; gamma={gamma:02d}; "
        f"payload=TurboQuant-KV-long-context-validation-{i:05d}; "
        "instruction=preserve-order-and-answer-from-the-final-record-only.\n"
    )
lines.extend(
    [
        "\n",
        "Question: What is the checksum for the final record?\n",
        "Answer:\n",
    ]
)
prompt.write_text("".join(lines))
PY

  "$ROOT/target/release/ironmlx-turboquant-kv-validate" \
    --model "$MODEL" \
    --prompt-file "$prompt" \
    --max-tokens "$MAX_TOKENS" \
    --kv-quant "$KV_JOINED" \
    --out "$out_dir/logits.json"

  for kv in "${KV_QUANTS[@]}"; do
    /usr/bin/time -l "$ROOT/target/release/ironmlx-core-bench" \
      --model "$MODEL" \
      --prompt-file "$prompt" \
      --mode gs-text \
      --max-tokens "$BENCH_MAX_TOKENS" \
      --runs "$RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --kv-quant "$kv" \
      --out "$out_dir/core-bench/$kv.json" \
      > "$out_dir/core-bench/$kv.stdout" \
      2> "$out_dir/core-bench/$kv.time.txt"
    if [[ ! -s "$out_dir/core-bench/$kv.stdout" ]]; then
      rm -f "$out_dir/core-bench/$kv.stdout"
    fi
  done
done

python3 - "$OUT_ROOT" "${KV_QUANTS[@]}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
order = sys.argv[2:]
context_order = [
    item.split(":", 1)[0]
    for item in os.environ.get("CONTEXTS_FOR_SUMMARY", "").split(",")
    if item
]

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}f}"

def one_line(text, limit=180):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def mean(values):
    return sum(values) / len(values) if values else None

def max_rss_gib(time_path):
    return memory_metric_gib(time_path, "maximum resident set size")

def peak_footprint_gib(time_path):
    return memory_metric_gib(time_path, "peak memory footprint")

def memory_metric_gib(time_path, metric):
    if not time_path.exists():
        return None
    value = None
    for line in time_path.read_text(errors="replace").splitlines():
        if metric in line:
            parts = line.strip().split()
            if parts:
                try:
                    value = int(parts[0])
                except ValueError:
                    pass
    if value is None:
        return None
    return value / (1024 ** 3)

def expected_answer(prompt_path):
    record = None
    alpha = None
    beta = None
    gamma = None
    for line in Path(prompt_path).read_text(errors="replace").splitlines():
        if not line.startswith("Record "):
            continue
        head, rest = line.split(":", 1)
        record = head.split()[1]
        parts = {}
        for item in rest.split(";"):
            item = item.strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()
        alpha = parts.get("alpha")
        beta = parts.get("beta")
        gamma = parts.get("gamma")
    if record is None:
        return "n/a"
    return f"CHECKSUM record={record} alpha={alpha} beta={beta} gamma={gamma}"

def exact_prefix(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count

def load_contexts():
    contexts = []
    seen = set()
    ordered_paths = []
    for label in context_order:
        path = out_root / label
        if path.exists():
            ordered_paths.append(path)
            seen.add(path.name)
    ordered_paths.extend(path for path in sorted(out_root.glob("ctx-*")) if path.name not in seen)
    for path in ordered_paths:
        logits_path = path / "logits.json"
        if logits_path.exists():
            contexts.append((path.name, path, json.loads(logits_path.read_text())))
    return contexts

def write_context_summary(label, path, logits):
    bench = {}
    for kv in order:
        bench_path = path / "core-bench" / f"{kv}.json"
        if bench_path.exists():
            bench[kv] = json.loads(bench_path.read_text())

    baseline_tokens = []
    if "none" in bench and bench["none"].get("records"):
        baseline_tokens = bench["none"]["records"][0].get("generated_token_ids", [])

    lines = []
    lines.append(f"# TurboQuant KV Long Context: {label}")
    lines.append("")
    lines.append(f"- model: `{logits['meta']['model_dir']}`")
    lines.append(f"- prompt: `{logits['meta']['prompt_file']}`")
    lines.append(f"- prompt_tokens: `{logits['meta']['prompt_tokens']}`")
    lines.append(f"- logits_max_tokens: `{logits['meta']['max_tokens']}`")
    lines.append(f"- expected: `{expected_answer(logits['meta']['prompt_file'])}`")
    lines.append("")
    lines.append("## Logits Replay")
    lines.append("")
    lines.append("| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cfg in logits["configs"]:
        steps = cfg["steps"]
        argmax_matches = sum(1 for step in steps if step["argmax_matches"])
        mismatch = cfg["first_token_mismatch_step"]
        lines.append(
            "| {label} | {exact} | {mismatch} | {argmax}/{total} | {max_abs} | {mean_abs} | {rms} | {cosine} | {top5} |".format(
                label=cfg["label"],
                exact=cfg["exact_token_match_count"],
                mismatch="none" if mismatch is None else mismatch,
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
    lines.append("| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for kv in order:
        data = bench.get(kv)
        if not data:
            continue
        records = data.get("records", [])
        tokens = records[0].get("generated_token_ids", []) if records else []
        exact = "n/a" if not baseline_tokens else str(exact_prefix(tokens, baseline_tokens))
        summary = data["summary"]
        lines.append(
                "| {kv} | {tokens} | {ttft} | {e2e} | {tps} | {rss} | {peak} | {exact} |".format(
                kv=kv,
                tokens=records[0].get("generated_tokens", 0) if records else 0,
                ttft=fmt(summary["ttft_ms"]["p50"], 2),
                e2e=fmt(summary["e2e_ms"]["p50"], 2),
                tps=fmt(summary["generation_tps"]["p50"], 2),
                rss=fmt(max_rss_gib(path / "core-bench" / f"{kv}.time.txt"), 3),
                peak=fmt(peak_footprint_gib(path / "core-bench" / f"{kv}.time.txt"), 3),
                exact=exact,
            )
        )
    lines.append("")
    lines.append("## Generated Text")
    lines.append("")
    for kv in order:
        data = bench.get(kv)
        if data and data.get("records"):
            lines.append(f"- `{kv}`: {one_line(data['records'][0].get('generated_text', ''))}")
    (path / "summary.md").write_text("\n".join(lines) + "\n")

contexts = load_contexts()
for label, path, logits in contexts:
    write_context_summary(label, path, logits)

lines = []
lines.append("# TurboQuant KV Long Context Validation")
lines.append("")
lines.append(f"- root: `{out_root}`")
lines.append(f"- kv matrix: `{','.join(order)}`")
lines.append("")
lines.append("## Logits Replay")
lines.append("")
lines.append("| context | prompt tokens | kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine |")
lines.append("| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
for label, _path, logits in contexts:
    prompt_tokens = logits["meta"]["prompt_tokens"]
    for cfg in logits["configs"]:
        steps = cfg["steps"]
        argmax_matches = sum(1 for step in steps if step["argmax_matches"])
        mismatch = cfg["first_token_mismatch_step"]
        lines.append(
            "| {context} | {prompt_tokens} | {kv} | {exact} | {mismatch} | {argmax}/{total} | {max_abs} | {mean_abs} | {rms} | {cosine} |".format(
                context=label,
                prompt_tokens=prompt_tokens,
                kv=cfg["label"],
                exact=cfg["exact_token_match_count"],
                mismatch="none" if mismatch is None else mismatch,
                argmax=argmax_matches,
                total=len(steps),
                max_abs=fmt(mean([step["max_abs_diff"] for step in steps]), 6),
                mean_abs=fmt(mean([step["mean_abs_diff"] for step in steps]), 6),
                rms=fmt(mean([step["rms_diff"] for step in steps]), 6),
                cosine=fmt(min(step["cosine_similarity"] for step in steps), 9),
            )
        )
lines.append("")
lines.append("## Core Generation")
lines.append("")
lines.append("| context | prompt tokens | kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none | generated text |")
lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for label, path, logits in contexts:
    prompt_tokens = logits["meta"]["prompt_tokens"]
    baseline_tokens = []
    none_path = path / "core-bench" / "none.json"
    if none_path.exists():
        none_data = json.loads(none_path.read_text())
        if none_data.get("records"):
            baseline_tokens = none_data["records"][0].get("generated_token_ids", [])
    for kv in order:
        bench_path = path / "core-bench" / f"{kv}.json"
        if not bench_path.exists():
            continue
        data = json.loads(bench_path.read_text())
        records = data.get("records", [])
        record = records[0] if records else {}
        tokens = record.get("generated_token_ids", [])
        exact = "n/a" if not baseline_tokens else str(exact_prefix(tokens, baseline_tokens))
        summary = data["summary"]
        lines.append(
            "| {context} | {prompt_tokens} | {kv} | {tokens} | {ttft} | {e2e} | {tps} | {rss} | {peak} | {exact} | {text} |".format(
                context=label,
                prompt_tokens=prompt_tokens,
                kv=kv,
                tokens=record.get("generated_tokens", 0),
                ttft=fmt(summary["ttft_ms"]["p50"], 2),
                e2e=fmt(summary["e2e_ms"]["p50"], 2),
                tps=fmt(summary["generation_tps"]["p50"], 2),
                rss=fmt(max_rss_gib(path / "core-bench" / f"{kv}.time.txt"), 3),
                peak=fmt(peak_footprint_gib(path / "core-bench" / f"{kv}.time.txt"), 3),
                exact=exact,
                text=one_line(record.get("generated_text", ""), 120).replace("|", "\\|"),
            )
        )
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- `prompt tokens` are measured by the ironmlx tokenizer at runtime.")
lines.append("- `exact-prefix vs none` compares the first measured core-bench record for each KV setting against `none`.")
lines.append("- `peak footprint GiB` comes from macOS `/usr/bin/time -l` and is more sensitive to MLX memory pressure than `maximum resident set size` on this workload.")
lines.append("- The benchmark uses one measured run by default; use higher `RUNS` for stable latency percentiles.")

(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print(out_root / "summary.md")
PY
