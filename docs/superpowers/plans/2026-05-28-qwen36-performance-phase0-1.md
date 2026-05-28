# Qwen3.6 Performance Phase 0/1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible Qwen3.6 MoE performance baseline and black-box compare ironmlx against omlx on the same local model.

**Architecture:** Keep the feature branch clean and use `/tmp` for heavy artifacts, logs, and CSV outputs. Run ironmlx and omlx one at a time to avoid double-loading the 35B 4-bit model and distorting memory pressure. Use fixed prompt material for concurrent tests because `iron-bench --concurrent` intentionally uses time-based nonces and Qwen synthetic prompts can occasionally stop early.

**Tech Stack:** Rust `cargo`, `ironmlx serve`, `iron-bench`, omlx CLI from `/Users/xin/workspace/iron-rivals/omlx/.venv/bin/omlx`, Python `httpx` and `tokenizers` for fixed-prompt concurrent streaming probes.

---

## Common Inputs

- Worktree: `/Users/xin/workspace/ironmlx-qwen36-perf`
- Model snapshot: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46`
- Artifact root pattern: `/tmp/ironmlx-qwen36-perf-phase01-YYYYMMDD-HHMMSS`
- ironmlx port: `18140`
- omlx port: `18141`
- Benchmark matrix:
  - `c=1 pp512 tg1`
  - `c=1 pp512 tg16`
  - `c=2 pp512 tg16`
  - `c=4 pp512 tg16`

## Task 0: Workspace and Baseline Verification

**Files:**
- Read: `AGENTS.md`
- Read: `Cargo.toml`
- Create artifact directory under `/tmp`

- [ ] Verify worktree status:

```bash
git -C /Users/xin/workspace/ironmlx-qwen36-perf status --short --branch
```

Expected: clean branch `ironmlx-qwen36-perf`.

- [ ] Run required Rust baseline checks:

```bash
cd /Users/xin/workspace/ironmlx-qwen36-perf
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib -- --test-threads=1
```

Expected: all commands exit 0. MLX C++ header warnings are acceptable only when the cargo command exits 0.

## Task 1: Artifact Layout

**Files:**
- Create under `/tmp`, not in git:
  - `meta.env`
  - `ironmlx/`
  - `omlx/`
  - `reports/`
  - `omlx-model-root/qwen36` symlink

- [ ] Create artifact layout:

```bash
OUT=/tmp/ironmlx-qwen36-perf-phase01-$(date +%Y%m%d-%H%M%S)
MODEL=$HOME/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
mkdir -p "$OUT"/{ironmlx,omlx,reports,omlx-model-root}
ln -sfn "$MODEL" "$OUT/omlx-model-root/qwen36"
cat > "$OUT/meta.env" <<EOF
OUT=$OUT
MODEL=$MODEL
IRONMLX_PORT=18140
OMLX_PORT=18141
IRONMLX_BRANCH=$(git -C /Users/xin/workspace/ironmlx-qwen36-perf branch --show-current)
IRONMLX_HEAD=$(git -C /Users/xin/workspace/ironmlx-qwen36-perf rev-parse HEAD)
OMLX_HEAD=$(git -C /Users/xin/workspace/iron-rivals/omlx rev-parse HEAD)
EOF
ln -sfn "$OUT" /tmp/ironmlx-qwen36-perf-phase01-latest
```

Expected: `/tmp/ironmlx-qwen36-perf-phase01-latest` points at the current run directory.

## Task 2: ironmlx Black-Box Runs

**Files:**
- Read: `iron-bench/src/main.rs`
- Generate under artifact root:
  - `ironmlx/server.log`
  - `ironmlx/seq_tg1.csv`
  - `ironmlx/seq_tg16.csv`
  - `ironmlx/fixed_c1_tg16.json`
  - `ironmlx/fixed_c2_tg16.json`
  - `ironmlx/fixed_c4_tg16.json`

- [ ] Start ironmlx:

```bash
cd /Users/xin/workspace/ironmlx-qwen36-perf
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve \
  --model "$MODEL" \
  --port "$IRONMLX_PORT" \
  --host 127.0.0.1 \
  --b-max 4 \
  --prefill-chunk-size 2048 \
  >"$OUT/ironmlx/server.log" 2>&1
```

- [ ] Run sequential c=1 probes with fixed nonce:

```bash
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
cd /Users/xin/workspace/ironmlx-qwen36-perf
./target/release/iron-bench --target ironmlx=http://127.0.0.1:$IRONMLX_PORT \
  --model-dir "$MODEL" --model qwen36 --prompt-len 512 --max-tokens 1 \
  --runs 5 --warmup 1 --nonce-seed 42 --format csv > "$OUT/ironmlx/seq_tg1.csv"
./target/release/iron-bench --target ironmlx=http://127.0.0.1:$IRONMLX_PORT \
  --model-dir "$MODEL" --model qwen36 --prompt-len 512 --max-tokens 16 \
  --runs 5 --warmup 1 --nonce-seed 42 --format csv > "$OUT/ironmlx/seq_tg16.csv"
```

Expected: record measured finish reasons. If either target stops before the
requested token count, keep the CSV as background only and use the fixed-prompt
probe as the authoritative fair comparison.

- [ ] Run concurrent fixed-prompt probes:

```bash
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
for c in 1 2 4; do
  /Users/xin/workspace/iron-rivals/omlx/.venv/bin/python "$OUT/reports/fixed_prompt_concurrent.py" \
    --url http://127.0.0.1:$IRONMLX_PORT \
    --model qwen36 \
    --model-dir "$MODEL" \
    --prompt-len 512 \
    --max-tokens 16 \
    --concurrency "$c" \
    --duration 15 \
    --out "$OUT/ironmlx/fixed_c${c}_tg16.json"
done
```

Expected: `valid_requests == requests` and all completions produce 16 tokens
with `finish_reason=length`.

## Task 3: omlx Black-Box Runs

**Files:**
- Generate under artifact root:
  - `omlx/server.log`
  - `omlx/seq_tg1.csv`
  - `omlx/seq_tg16.csv`
  - `omlx/fixed_c1_tg16.json`
  - `omlx/fixed_c2_tg16.json`
  - `omlx/fixed_c4_tg16.json`

- [ ] Start omlx with cache disabled:

```bash
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
cd /Users/xin/workspace/iron-rivals/omlx
/Users/xin/workspace/iron-rivals/omlx/.venv/bin/omlx serve \
  --model-dir "$OUT/omlx-model-root" \
  --host 127.0.0.1 \
  --port "$OMLX_PORT" \
  --max-concurrent-requests 4 \
  --no-cache \
  --base-path "$OUT/omlx/base" \
  >"$OUT/omlx/server.log" 2>&1
```

- [ ] Run the same sequential and concurrent probes against omlx:

```bash
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
cd /Users/xin/workspace/ironmlx-qwen36-perf
./target/release/iron-bench --target omlx=http://127.0.0.1:$OMLX_PORT \
  --model-dir "$MODEL" --model qwen36 --prompt-len 512 --max-tokens 1 \
  --runs 5 --warmup 1 --nonce-seed 42 --format csv > "$OUT/omlx/seq_tg1.csv"
./target/release/iron-bench --target omlx=http://127.0.0.1:$OMLX_PORT \
  --model-dir "$MODEL" --model qwen36 --prompt-len 512 --max-tokens 16 \
  --runs 5 --warmup 1 --nonce-seed 42 --format csv > "$OUT/omlx/seq_tg16.csv"
for c in 1 2 4; do
  /Users/xin/workspace/iron-rivals/omlx/.venv/bin/python "$OUT/reports/fixed_prompt_concurrent.py" \
    --url http://127.0.0.1:$OMLX_PORT --model qwen36 --model-dir "$MODEL" \
    --prompt-len 512 --max-tokens 16 --concurrency "$c" --duration 15 \
    --out "$OUT/omlx/fixed_c${c}_tg16.json"
done
```

Expected: same validity gates as ironmlx.

## Task 4: Report

**Files:**
- Generate under artifact root:
  - `reports/summary.md`
  - `reports/summary.json`

- [ ] Summarize metrics:

```bash
source /tmp/ironmlx-qwen36-perf-phase01-latest/meta.env
/Users/xin/workspace/iron-rivals/omlx/.venv/bin/python "$OUT/reports/summarize_phase01.py"
```

Expected: summary includes ironmlx/omlx ratios for c=1 tg1, c=1 tg16, c=2 tg16, and c=4 tg16.

## Decision Gate

- If ironmlx is faster or within 5 percent of omlx in all valid cells: move to white-box omlx design analysis and selective kernel-level optimization.
- If ironmlx is slower by more than 5 percent in any valid cell: inspect the losing cell first, then decide whether the gap is scheduler, prefill, decode, or MoE gather-qmm dominated.
- If any cell is invalid due to early stop, server error, or partial generation: fix the benchmark protocol before drawing performance conclusions.

## Execution Notes: 2026-05-28

Artifact root:
`/tmp/ironmlx-qwen36-perf-phase01-20260528-175900`

Phase 0 verification completed on branch `ironmlx-qwen36-perf` at
`65d269bc6e8d3ab0d0d639816293b9e271791052`:

- `cargo fmt`
- `MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check`
- `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings`
- `MLX_DIR=$HOME/.local/mlx cargo build --release`
- `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib -- --test-threads=1`

The initial synthetic `iron-bench` `pp512 tg16` cell exposed an important
measurement issue: omlx stopped after 10 tokens for that prompt, while ironmlx
hit the 16-token length cap. That cell is retained as background data, but not
used as the primary fair comparison. The main comparison uses a fixed 512-token
prompt ending with a numeric-continuation instruction; both targets generated
16 tokens and ended by `length` for all valid rows.

Primary fixed-prompt results:

| Cell | ironmlx TTFT p50 ms | omlx TTFT p50 ms | iron/omlx TTFT | ironmlx E2E p50 ms | omlx E2E p50 ms | iron/omlx E2E | ironmlx tok/s | omlx tok/s | iron/omlx tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c=1 pp512 tg16` | 334.69 | 221.95 | 1.508 | 531.02 | 350.99 | 1.513 | 30.10 | 45.26 | 0.665 |
| `c=2 pp512 tg16` | 581.34 | 252.70 | 2.300 | 778.53 | 628.37 | 1.239 | 41.06 | 50.96 | 0.806 |
| `c=4 pp512 tg16` | 1098.45 | 732.21 | 1.500 | 1290.84 | 1174.87 | 1.099 | 49.28 | 55.00 | 0.896 |

Decision gate result: ironmlx is slower than omlx by more than 5 percent in all
primary fixed-prompt cells. The largest gap is c=1 TTFT/E2E, which points first
to model execution and prefill path efficiency rather than only queueing. The
c=4 throughput gap narrows to about 10 percent, so scheduler batching is not the
only suspected cause. Next phase should combine white-box omlx design analysis
with targeted ironmlx attribution around Qwen3.6 MoE prefill/decode execution
cost.
