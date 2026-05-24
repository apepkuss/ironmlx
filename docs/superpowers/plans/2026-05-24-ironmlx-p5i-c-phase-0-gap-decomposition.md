# P5i.c Phase 0 — Gap Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-rank short-PP candidates at PP=128 and PP=512 on current HEAD `6a593c4`, using the P5h+2.a measurement protocol, so P5i.c Phase 1 can choose candidates from measured attribution, CI95, tied tiers, and a clean vs-omlx baseline.

**Architecture:** Phase 0 adds a dedicated P5i.c capture harness rather than mutating the validated P5h T5 harness. Capture runs are serial: ironmlx T1 first, omlx T2 after T1, then CPU-only aggregation. Python wrappers compute multi-repeat substep CI, production-root denominators, pp_tps envelopes, tied tiers, 4-category coverage, and Phase 1 rule output.

**Tech Stack:** Rust `cargo test --features p5h-profile`, `iron-bench`, Python 3, pytest, ruff, existing `tools/p5h_aggregator` parser/schema/ranking code, P5h+2.a monolithic preheat protocol.

**Spec ref:** `docs/superpowers/specs/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition-design.md` at branch HEAD `6a593c4`.

---

## File Structure

**Create:**
- `ironmlx/tests/p5i_c_phase_0_capture.rs` — env-driven, dual-mode capture harness for one repeat and one mode.
- `tools/p5h_aggregator/multi_repeat.py` — probe-mode per-substep multi-repeat CI plus production-root extraction from server logs.
- `tools/p5i_c_pp_tps_envelope.py` — multi-repeat pp_tps envelope and ironmlx-vs-omlx delta.
- `tools/p5i_c_phase0_compose.py` — final ranking JSON composer.
- `tools/p5h_aggregator/tests/test_multi_repeat.py`
- `tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py`
- `tools/p5h_aggregator/tests/test_roi_ranking_phase_0.py`
- `reports/p5i-c-phase-0-audit.md` and `reports/p5i-c-phase-0-audit.json` — gitignored audit outputs.
- `reports/p5i-c-phase-0-bench-log.md` — gitignored run log.
- `docs/p5i-c-phase-0-ranking-snapshot.md`
- `docs/p5i-c-phase-0-close-out.md`
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5i_c_phase_0_findings.md`

**Modify:**
- `tools/p5h_aggregator/roi_ranking.py` — add tied-tier, category coverage, Phase 1 rule, dense trigger, and production-root-aware output helpers.
- Conditional T0.5 only: Rust source file owning a missing scheduler/KV span; `tools/p5h_aggregator/schema_validator.py::LANE_A_REQUIRED_TREE` only when the added span is mandatory on every non-aborted Lane A target request.
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md`

**Do not modify:**
- `ironmlx/tests/p5h_t5_attribution_capture.rs`
- `ironmlx/tests/p5h_common/mod.rs`
- `tools/p5i_a_baseline_aggregate.py`
- `ironmlx/src/core/p5h.rs::LANE_B_ALLOWED_TRY_SPAN_NAMES`

---

## Task 1: T0 Audit And Conditional T0.5 Spans

**Files:**
- Create: `reports/p5i-c-phase-0-audit.md`
- Create: `reports/p5i-c-phase-0-audit.json`
- Conditional modify: Rust span call site; `tools/p5h_aggregator/schema_validator.py`; `tools/p5h_aggregator/roi_ranking.py`

- [ ] **Step 1.1: Inspect current instrumentation**

```bash
cd /Users/xin/workspace/ironmlx-backend
HEAD=$(git rev-parse --short HEAD)
test "$HEAD" = "6a593c4"

rg -n "try_with_p5h_span_from_current_trace|with_p5h_span_from_current_trace|open_p5h_span_at|close_p5h_span" \
  ironmlx/src/core/scheduler.rs \
  ironmlx/src/core \
  ironmlx/src/models \
  ironmlx/src/nn

rg -n "LANE_A_REQUIRED_TREE|LANE_B_REQUIRED_TREE|LANE_B_ALLOWED_TREE" tools/p5h_aggregator/schema_validator.py
rg -n "KERNEL_BOUND_SPANS|fused_sdpa|gather_qmm|cache_state_update|scheduler" tools/p5h_aggregator/roi_ranking.py
```

Expected: enough evidence to classify scheduler/chunking, KV cache layout, attention/fused_sdpa, and MoE gather_qmm as `measured`, `unmeasured`, or `proxy-only`.

- [ ] **Step 1.2: Write machine-readable audit JSON**

The JSON must use only these status values: `measured`, `unmeasured`, `proxy-only`.

```bash
mkdir -p reports
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
audit = {
    "head": head,
    "categories": {
        "scheduler": {
            "status": "unmeasured",
            "spans": [],
            "t0_5_action": "add_one_representative_lane_a_span_if_a_single_span_can_cover_scheduler_dispatch",
        },
        "kv_cache": {
            "status": "proxy-only",
            "spans": ["cache_state_update"],
            "t0_5_action": "upgrade_to_measured_only_if_one_lane_a_span_can_cover_layout_specific_work",
        },
        "attention": {
            "status": "measured",
            "spans": ["fused_sdpa", "kv_mask_update", "gda_step_1a_in_proj_qkvz", "gda_step_7_kernel_dispatch_and_materialize", "gda_step_8_norm_proj"],
            "t0_5_action": "none",
        },
        "moe": {
            "status": "measured",
            "spans": ["gather_qmm_gate_up", "gather_qmm_down", "routing_sort_pack", "routing_unsort_weighted_reduce"],
            "t0_5_action": "none",
        },
    },
}
Path("reports/p5i-c-phase-0-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
PY
```

The initial JSON above is a conservative starting point. Replace any status/span list with the actual Step 1.1 evidence before continuing.

- [ ] **Step 1.3: Write human-readable audit report**

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("reports/p5i-c-phase-0-audit.json").read_text())
lines = [
    "# P5i.c Phase 0 — Instrumentation Audit",
    "",
    f"**HEAD:** `{data['head']}`",
    "",
    "| Category | Status | Existing spans | T0.5 action |",
    "|---|---|---|---|",
]
for category, row in data["categories"].items():
    spans = ", ".join(row["spans"]) if row["spans"] else "none"
    lines.append(f"| {category} | `{row['status']}` | {spans} | {row['t0_5_action']} |")
Path("reports/p5i-c-phase-0-audit.md").write_text("\n".join(lines) + "\n")
PY
```

- [ ] **Step 1.4: Decide whether T0.5 runs**

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("reports/p5i-c-phase-0-audit.json").read_text())
needs = [
    category
    for category, row in data["categories"].items()
    if row["status"] == "unmeasured" or (
        row["status"] == "proxy-only"
        and "one_lane_a_span" in row["t0_5_action"]
    )
]
print("T0.5 required for:", ", ".join(needs) if needs else "none")
PY
```

If T0.5 is required, continue with Steps 1.5-1.9. If no category can be upgraded with one representative span, record that limitation in both audit files and skip to Task 2.

- [ ] **Step 1.5: Conditional T0.5 Rust span insertion**

Use `try_with_p5h_span_from_current_trace` for internal scheduler/model call sites. This emits on Lane A, is gated by the Lane B allow-list on Lane B, and no-ops safely when no OpenAI trace is active.

```rust
let result = crate::core::p5h::try_with_p5h_span_from_current_trace(
    "scheduler_admit_dispatch",
    crate::core::p5h::SpanFields::default,
    || {
        // Existing scheduler or KV work moves here unchanged.
        existing_work()
    },
);
```

If the target code does not return a value, use:

```rust
crate::core::p5h::try_with_p5h_span_from_current_trace(
    "kv_cache_layout",
    crate::core::p5h::SpanFields::default,
    || {
        existing_work();
    },
);
```

Do not add the new names to `LANE_B_ALLOWED_TRY_SPAN_NAMES`.

- [ ] **Step 1.6: Conditional schema/ranking updates**

```bash
cd /Users/xin/workspace/ironmlx-backend
rg -n "LANE_A_REQUIRED_TREE|KERNEL_BOUND_SPANS" tools/p5h_aggregator/schema_validator.py tools/p5h_aggregator/roi_ranking.py
```

Add a new span to `LANE_A_REQUIRED_TREE` only if it is mandatory on every non-aborted PP=128 and PP=512 request. Add it to `KERNEL_BOUND_SPANS` only if it represents kernel-bound work rather than scheduler/control-flow work.

- [ ] **Step 1.7: Conditional T0.5 verification**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v -k "schema or lane or allow"
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
IRONMLX_MOE_MODEL_DIR="$IRONMLX_MOE_MODEL_DIR" cargo test --release -p ironmlx --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --nocapture
```

Expected: all commands pass. Smoke pp_tps must remain within the existing ±2% feature-off vs feature-on flag-OFF parity bound.

- [ ] **Step 1.8: Conditional T0.5 commit**

```bash
git add ironmlx/src tools/p5h_aggregator/schema_validator.py tools/p5h_aggregator/roi_ranking.py tools/p5h_aggregator/tests
git commit -m "feat(p5i-c-t0): add minimal Lane A spans for Phase 0 audit"
```

Skip the commit if T0.5 did not modify tracked files.

---

## Task 2: Build P5i.c Capture Harness

**Files:**
- Create: `ironmlx/tests/p5i_c_phase_0_capture.rs`

- [ ] **Step 2.1: Implement harness contract**

The harness must implement these exact behaviors:

- Env vars:
  - `P5I_C_PP_LIST`, default `128,512`
  - `P5I_C_RUNS_PER_PP`, default `128:7,512:15`
  - `P5I_C_PREHEAT_SECONDS`, default `300`
  - `P5I_C_PREHEAT_RUNS`, default `1100`
  - `P5I_C_REPEAT_INDEX`, required
  - `P5I_C_MODE`, required and one of `probe`, `production`
  - `P5I_C_MODEL`, default `qwen3.5-moe`
  - `P5I_C_MODEL_DIR`, default from `IRONMLX_MOE_MODEL_DIR`
- Output dir per cell: `/tmp/p5i-c-phase-0-r${repeat}-pp${pp}-${mode}/`
- Files per cell: `server.log`, `bench.csv`, `meta.json`
- Probe mode:
  - server has `--p5h-measurement-eval-probes`
  - iron-bench uses `--capture-server-request-id`
  - `--warmup 0`
- Production mode:
  - server does not have `--p5h-measurement-eval-probes`
  - iron-bench does not use `--capture-server-request-id`
  - `--warmup 1`
  - server log is still captured because root spans provide `production_root_us`
- Preheat:
  - one monolithic preheat per repeat/mode before the PP cells
  - create `/tmp/p5i-c-phase-0-r${repeat}-preheat-${mode}/` before spawning the preheat server
  - record `preheat_wall_s` in every cell `meta.json`
- Safety:
  - check port 18099 is free before each server spawn
  - health check validates HTTP 200, not response body text
  - always kill and wait on server in success and failure paths

- [ ] **Step 2.2: Use existing ironmlx serve CLI correctly**

`ironmlx serve --model` expects the local model directory path. Do not pass `P5I_C_MODEL` to `ironmlx serve`; use it only for iron-bench request payloads.

```bash
cd /Users/xin/workspace/ironmlx-backend
rg -n "pub struct ServeArgs|pub model: String" ironmlx/src/cli/serve.rs
```

- [ ] **Step 2.3: Verify compile and full Rust gate**

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands pass.

- [ ] **Step 2.4: Commit harness**

```bash
git add ironmlx/tests/p5i_c_phase_0_capture.rs
git commit -m "feat(p5i-c-t1): add dual-mode Phase 0 capture harness"
```

---

## Task 3: Run Ironmlx T1 Capture Sweep

**Files:**
- Outputs: `/tmp/p5i-c-phase-0-r{1,2,3}-pp{128,512}-{probe,production}/`
- Append: `reports/p5i-c-phase-0-bench-log.md`

- [ ] **Step 3.1: Prepare env file**

```bash
cd /Users/xin/workspace/ironmlx-backend
SNAP=$(find "$HOME/.ironmlx/models" -path '*Qwen3*MoE*4bit*/snapshots/*' -type d -maxdepth 7 | head -1)
test -n "$SNAP"
test -f "$SNAP/tokenizer.json"
cat > /tmp/p5i-c-env.sh <<EOF
SNAP=$SNAP
MLX_DIR=$HOME/.local/mlx
EOF
cat /tmp/p5i-c-env.sh
```

- [ ] **Step 3.2: Run all 12 ironmlx cells serially**

```bash
cd /Users/xin/workspace/ironmlx-backend
source /tmp/p5i-c-env.sh
for r in 1 2 3; do
  for mode in probe production; do
    P5I_C_REPEAT_INDEX=$r \
    P5I_C_MODE=$mode \
    P5I_C_MODEL_DIR=$SNAP \
    MLX_DIR=$MLX_DIR \
      cargo test --release -p ironmlx --features p5h-profile \
      --test p5i_c_phase_0_capture -- --ignored --test-threads=1 --nocapture \
      2>&1 | tee /tmp/p5i-c-r${r}-${mode}.log
  done
done
```

- [ ] **Step 3.3: Verify all T1 cell artifacts**

```bash
cd /Users/xin/workspace/ironmlx-backend
python3 - <<'PY'
import csv
import json
from pathlib import Path

expected = {128: 7, 512: 15}
failures = []
for repeat in (1, 2, 3):
    for pp, rows_expected in expected.items():
        for mode in ("probe", "production"):
            d = Path(f"/tmp/p5i-c-phase-0-r{repeat}-pp{pp}-{mode}")
            bench = d / "bench.csv"
            log = d / "server.log"
            meta = d / "meta.json"
            if not bench.exists() or not log.exists() or not meta.exists():
                failures.append(f"{d}: missing bench/server/meta")
                continue
            rows = list(csv.DictReader(bench.open()))
            if len(rows) != rows_expected:
                failures.append(f"{d}: expected {rows_expected} bench rows, got {len(rows)}")
            if mode == "probe" and "request_id" not in rows[0]:
                failures.append(f"{d}: probe bench.csv missing request_id column")
            if mode == "production" and "request_id" in rows[0]:
                failures.append(f"{d}: production bench.csv unexpectedly has request_id column")
            m = json.loads(meta.read_text())
            if m["preheat_wall_s"] < 300:
                failures.append(f"{d}: preheat_wall_s={m['preheat_wall_s']} < 300")
if failures:
    raise SystemExit("\n".join(failures))
print("OK: all 12 ironmlx cells verified")
PY
```

- [ ] **Step 3.4: Append T1 bench log**

```bash
mkdir -p reports
python3 - <<'PY' >> reports/p5i-c-phase-0-bench-log.md
import csv
import json
from pathlib import Path

print("# P5i.c Phase 0 T1 — Ironmlx Capture")
print()
print("| repeat | mode | pp | rows | preheat_wall_s |")
print("|---|---|---|---|---|")
for repeat in (1, 2, 3):
    for mode in ("probe", "production"):
        for pp in (128, 512):
            d = Path(f"/tmp/p5i-c-phase-0-r{repeat}-pp{pp}-{mode}")
            rows = len(list(csv.DictReader((d / "bench.csv").open())))
            meta = json.loads((d / "meta.json").read_text())
            print(f"| {repeat} | {mode} | {pp} | {rows} | {meta['preheat_wall_s']} |")
print()
PY
```

Task 3 has no commit because artifacts are gitignored.

---

## Task 4: Run Omlx T2 Remeasure

**Files:**
- Outputs: `/tmp/p5i-c-phase-0-omlx-r{1,2,3}-pp{128,512}/bench.csv`
- Append: `reports/p5i-c-phase-0-bench-log.md`

- [ ] **Step 4.1: Resolve omlx invocation**

```bash
cd /Users/xin/workspace/ironmlx-backend
ls /Users/xin/workspace/iron-rivals/omlx
rg -n "serve|healthz|OpenAI" /Users/xin/workspace/iron-rivals/omlx -g '*.py' | head -40
```

Use the source checkout under `/Users/xin/workspace/iron-rivals/omlx`. Do not use a pip-installed mlx_lm fallback.

- [ ] **Step 4.2: Run 3 repeats serially**

```bash
cd /Users/xin/workspace/ironmlx-backend
source /tmp/p5i-c-env.sh
OMLX_PORT=18100
for r in 1 2 3; do
  cd /Users/xin/workspace/iron-rivals/omlx
  uv run --with-editable . python -m omlx serve --model "$SNAP" --port "$OMLX_PORT" --host 127.0.0.1 > /tmp/p5i-c-omlx-r${r}-serve.log 2>&1 &
  OMLX_PID=$!
  cd /Users/xin/workspace/ironmlx-backend
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${OMLX_PORT}/healthz" || true)
    test "$code" = "200" && break
    sleep 5
  done
  cargo run --release -p iron-bench -- \
    --target omlx_preheat=http://127.0.0.1:${OMLX_PORT} \
    --model qwen3.5-moe --model-dir "$SNAP" \
    --prompt-len 512 --max-tokens 1 --runs 1100 --warmup 0 --format csv > /tmp/p5i-c-omlx-r${r}-preheat.csv
  for pp in 128 512; do
    runs=$([ "$pp" = "128" ] && echo 7 || echo 15)
    mkdir -p /tmp/p5i-c-phase-0-omlx-r${r}-pp${pp}
    cargo run --release -p iron-bench -- \
      --target omlx=http://127.0.0.1:${OMLX_PORT} \
      --model qwen3.5-moe --model-dir "$SNAP" \
      --prompt-len "$pp" --max-tokens 1 --runs "$runs" --warmup 1 --format csv \
      > /tmp/p5i-c-phase-0-omlx-r${r}-pp${pp}/bench.csv
  done
  kill "$OMLX_PID"
  wait "$OMLX_PID" 2>/dev/null || true
  sleep 5
done
```

- [ ] **Step 4.3: Verify all 6 omlx cells**

```bash
python3 - <<'PY'
import csv
from pathlib import Path

expected = {128: 7, 512: 15}
for repeat in (1, 2, 3):
    for pp, n in expected.items():
        path = Path(f"/tmp/p5i-c-phase-0-omlx-r{repeat}-pp{pp}/bench.csv")
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == n, f"{path}: expected {n}, got {len(rows)}"
print("OK: all 6 omlx cells verified")
PY
```

Task 4 has no commit because artifacts are gitignored.

---

## Task 5: Build And Run Aggregation

**Files:**
- Create: `tools/p5h_aggregator/multi_repeat.py`
- Create: `tools/p5i_c_pp_tps_envelope.py`
- Create: `tools/p5i_c_phase0_compose.py`
- Create tests listed in File Structure
- Modify: `tools/p5h_aggregator/roi_ranking.py`

- [ ] **Step 5.1: Implement `tools/p5i_c_pp_tps_envelope.py`**

Required functions:

```python
EXPECTED_RUNS = {128: 7, 512: 15}

def load_pp_tps(csv_path: Path, pp: int) -> list[float]:
    """Hard-fail if pp_target is not pp, rows are missing, pp_tps is invalid, or row count != EXPECTED_RUNS[pp]."""

def compute_pp_tps_envelope(repeat_csvs: list[Path], pp: int) -> dict:
    """Return per-repeat medians, within_sweep_ci95_max_pct, between_sweep_half_range_pct, final_uncertainty_envelope_pct."""

def compute_vs_omlx_delta(ironmlx_repeat_csvs: list[Path], omlx_repeat_csvs: list[Path], pp: int) -> dict:
    """Return delta_pct_median plus conservative CI bounds using both sides' final envelopes."""
```

CLI:

```bash
uv run python tools/p5i_c_pp_tps_envelope.py \
  --pp 128 \
  --repeat-csv /tmp/p5i-c-phase-0-r1-pp128-production/bench.csv \
  --repeat-csv /tmp/p5i-c-phase-0-r2-pp128-production/bench.csv \
  --repeat-csv /tmp/p5i-c-phase-0-r3-pp128-production/bench.csv \
  --out-json /tmp/p5i-c-phase-0-pp128-ironmlx-envelope.json
```

The CLI must also support `--compare-repeat-csv` for omlx inputs and emit delta fields when provided.

- [ ] **Step 5.2: Implement `tools/p5h_aggregator/multi_repeat.py`**

Required functions:

```python
def run_aggregator_one_probe_cell(repeat_dir: Path, tmp_dir: Path) -> tuple[Path, Path]:
    """Run p5h_aggregator.aggregator on probe-mode server.log + bench.csv; production mode is not accepted here."""

def parse_attribution_csv(path: Path) -> dict[int, dict[str, float]]:
    """Root row is the tree row with parent_span_id == ''. Exclude the root itself from candidate shares."""

def aggregate_multi_repeat(repeat_dirs: list[Path], pp: int) -> dict:
    """Collect per-repeat probe shares; emit median_pct, ci95_low_pct, ci95_high_pct, ci95_half_width_pct, between_sweep_half_range_pct for each substep."""

def extract_production_root_us(repeat_dirs: list[Path], pp: int) -> dict:
    """Parse production server.log lines with schema_validator.parse_line; select root span parent_span_id is None; median root inclusive_us per repeat and across repeats."""
```

`between_sweep_half_range_pct` for substep shares is a percentage-point half range: `(max(shares_pct) - min(shares_pct)) / 2`. Do not divide by the mean again.

- [ ] **Step 5.3: Extend `tools/p5h_aggregator/roi_ranking.py`**

Add functions:

```python
def identify_tied_tiers(
    ranking: list[tuple[str, float]],
    ci95_by_name: dict[str, tuple[float, float]],
) -> list[list[str]]:
    """Adjacent-overlap chain: if prev low <= curr high, curr joins current tier."""

def emit_category_coverage(
    audit_result: dict[str, str],
    measured_spans: set[str],
) -> dict[str, str]:
    """Return scheduler, kv_cache, attention, moe statuses; preserve proxy-only as a limitation."""

def emit_phase_1_default_rule(
    ranking_per_pp: dict[int, list[tuple[str, float]]],
    tiers_per_pp: dict[int, list[list[str]]],
    coverage: dict[str, str],
) -> dict:
    """Return R1, R2, R3, or data_insufficient plus suggested candidates and rationale."""

def evaluate_dense_diagnostic_trigger(
    tiers_per_pp: dict[int, list[list[str]]],
    per_substep_medians: dict[int, dict[str, float]],
) -> dict:
    """Implement trigger-A and trigger-B from the design spec."""
```

Add production-root awareness without changing existing CSV output semantics unexpectedly. If the existing `Candidate` dataclass is used, add `production_share_pct: float | None = None` with a default. If any target PP lacks production root data, the Phase 0 composer must emit `data_insufficient_for_production_share` and stop before close-out.

- [ ] **Step 5.4: Implement `tools/p5i_c_phase0_compose.py`**

Inputs:
- `--audit-json reports/p5i-c-phase-0-audit.json`
- `--out-json /tmp/p5i-c-phase-0-ranking.json`
- `--summary-md reports/p5i-c-phase-0-ranking-summary.md`

Required behavior:
- load probe multi-repeat JSON for PP=128 and PP=512
- load production root medians for PP=128 and PP=512
- rank candidates by `production_share_pct = probe_exclusive_us_median / production_root_us_median * 100`
- keep probe share separately as `probe_share_pct`
- compute tied tiers from CI95
- compute category coverage from audit JSON and measured spans
- compute Phase 1 default rule
- compute dense diagnostic trigger
- load ironmlx and omlx pp_tps envelopes and delta data
- write `/tmp/p5i-c-phase-0-ranking.json`
- write `reports/p5i-c-phase-0-ranking-summary.md`

- [ ] **Step 5.5: Add tests**

Required tests:
- pp_tps envelope passes for 3 stable repeats
- pp_tps envelope rejects fewer than 3 repeats
- pp_tps envelope rejects wrong `pp_target` and wrong row count
- vs-omlx delta is emitted when compare CSVs are provided
- `parse_attribution_csv` recognizes root via empty `parent_span_id`
- production root extraction parses root spans without request-id join
- tied-tier separate, adjacent merge, and chain merge
- category coverage preserves `proxy-only`
- R1/R2/R3 rule detection
- dense trigger-A and trigger-B

- [ ] **Step 5.6: Python hygiene and tests**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check \
  tools/p5h_aggregator/multi_repeat.py \
  tools/p5i_c_pp_tps_envelope.py \
  tools/p5i_c_phase0_compose.py \
  tools/p5h_aggregator/roi_ranking.py \
  tools/p5h_aggregator/tests/test_multi_repeat.py \
  tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py \
  tools/p5h_aggregator/tests/test_roi_ranking_phase_0.py
uv run --with ruff ruff format --check \
  tools/p5h_aggregator/multi_repeat.py \
  tools/p5i_c_pp_tps_envelope.py \
  tools/p5i_c_phase0_compose.py \
  tools/p5h_aggregator/roi_ranking.py \
  tools/p5h_aggregator/tests
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v
```

Expected: all commands pass.

- [ ] **Step 5.7: Run aggregation**

```bash
cd /Users/xin/workspace/ironmlx-backend
for pp in 128 512; do
  uv run python tools/p5i_c_pp_tps_envelope.py \
    --pp "$pp" \
    --repeat-csv /tmp/p5i-c-phase-0-r1-pp${pp}-production/bench.csv \
    --repeat-csv /tmp/p5i-c-phase-0-r2-pp${pp}-production/bench.csv \
    --repeat-csv /tmp/p5i-c-phase-0-r3-pp${pp}-production/bench.csv \
    --compare-repeat-csv /tmp/p5i-c-phase-0-omlx-r1-pp${pp}/bench.csv \
    --compare-repeat-csv /tmp/p5i-c-phase-0-omlx-r2-pp${pp}/bench.csv \
    --compare-repeat-csv /tmp/p5i-c-phase-0-omlx-r3-pp${pp}/bench.csv \
    --out-json /tmp/p5i-c-phase-0-pp${pp}-ironmlx-vs-omlx-envelope.json

  uv run python tools/p5h_aggregator/multi_repeat.py \
    --pp "$pp" \
    --repeat-dir /tmp/p5i-c-phase-0-r1-pp${pp}-probe \
    --repeat-dir /tmp/p5i-c-phase-0-r2-pp${pp}-probe \
    --repeat-dir /tmp/p5i-c-phase-0-r3-pp${pp}-probe \
    --production-repeat-dir /tmp/p5i-c-phase-0-r1-pp${pp}-production \
    --production-repeat-dir /tmp/p5i-c-phase-0-r2-pp${pp}-production \
    --production-repeat-dir /tmp/p5i-c-phase-0-r3-pp${pp}-production \
    --out-json /tmp/p5i-c-phase-0-pp${pp}-multirepeat.json
done

uv run python tools/p5i_c_phase0_compose.py \
  --audit-json reports/p5i-c-phase-0-audit.json \
  --out-json /tmp/p5i-c-phase-0-ranking.json \
  --summary-md reports/p5i-c-phase-0-ranking-summary.md
```

- [ ] **Step 5.8: Acceptance gate**

```bash
python3 - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("/tmp/p5i-c-phase-0-ranking.json").read_text())
assert set(d["ranking_per_pp"].keys()) == {"128", "512"}
assert set(d["category_coverage"].keys()) == {"scheduler", "kv_cache", "attention", "moe"}
assert d["phase_1_default_rule"]["triggered_rule"] in {"R1", "R2", "R3"}
for pp, env in d["envelopes_per_pp"].items():
    assert env["ironmlx"]["final_uncertainty_envelope_pct"] <= 2.0, pp
for pp, ranking in d["ranking_per_pp"].items():
    top5 = [row["span_name"] for row in ranking[:5]]
    assert "first_token_sampling_materialize_and_sample" not in top5, pp
for pp, tiers in d["tiers_per_pp"].items():
    assert isinstance(tiers, list) and all(isinstance(t, list) for t in tiers), pp
print("OK: Phase 0 acceptance gate passed")
PY
```

If omlx PP=512 exceeds ±2%, do not fail close-out; the composer must record it as `external_baseline_caveat`.

- [ ] **Step 5.9: Commit aggregation code**

```bash
git add \
  tools/p5h_aggregator/multi_repeat.py \
  tools/p5i_c_pp_tps_envelope.py \
  tools/p5i_c_phase0_compose.py \
  tools/p5h_aggregator/roi_ranking.py \
  tools/p5h_aggregator/tests/test_multi_repeat.py \
  tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py \
  tools/p5h_aggregator/tests/test_roi_ranking_phase_0.py
git commit -m "feat(p5i-c-t3): add Phase 0 multi-repeat ranking pipeline"
```

---

## Task 6: Conditional Dense Diagnostic

**Files:**
- Read: `/tmp/p5i-c-phase-0-ranking.json`
- Conditional outputs: `/tmp/p5i-c-phase-0-dense-r{1,2,3}-pp{128,512}-{probe,production}/`

- [ ] **Step 6.1: Check trigger**

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("/tmp/p5i-c-phase-0-ranking.json").read_text())
print(json.dumps({
    "dense_diagnostic_triggered": d["dense_diagnostic_triggered"],
    "dense_diagnostic_reason": d["dense_diagnostic_reason"],
}, indent=2))
PY
```

If `dense_diagnostic_triggered` is false, skip the rest of Task 6.

- [ ] **Step 6.2: Locate Dense model without downloading**

```bash
DENSE_SNAP=$(find "$HOME/.ironmlx/models" -path '*Qwen3*4bit*/snapshots/*' -type d -maxdepth 7 | grep -vi moe | head -1)
test -n "$DENSE_SNAP"
test -f "$DENSE_SNAP/tokenizer.json"
echo "$DENSE_SNAP"
```

If no local Dense snapshot exists, stop and ask Boss before downloading.

- [ ] **Step 6.3: Run Dense capture with model overrides**

```bash
cd /Users/xin/workspace/ironmlx-backend
source /tmp/p5i-c-env.sh
for r in 1 2 3; do
  for mode in probe production; do
    P5I_C_REPEAT_INDEX=$r \
    P5I_C_MODE=$mode \
    P5I_C_MODEL=qwen3.5-dense \
    P5I_C_MODEL_DIR=$DENSE_SNAP \
    MLX_DIR=$MLX_DIR \
      cargo test --release -p ironmlx --features p5h-profile \
      --test p5i_c_phase_0_capture -- --ignored --test-threads=1 --nocapture
    for pp in 128 512; do
      src=/tmp/p5i-c-phase-0-r${r}-pp${pp}-${mode}
      dst=/tmp/p5i-c-phase-0-dense-r${r}-pp${pp}-${mode}
      rm -rf "$dst"
      mv "$src" "$dst"
    done
  done
done
```

- [ ] **Step 6.4: Aggregate Dense and re-compose ranking JSON**

```bash
cd /Users/xin/workspace/ironmlx-backend
for pp in 128 512; do
  uv run python tools/p5h_aggregator/multi_repeat.py \
    --pp "$pp" \
    --repeat-dir /tmp/p5i-c-phase-0-dense-r1-pp${pp}-probe \
    --repeat-dir /tmp/p5i-c-phase-0-dense-r2-pp${pp}-probe \
    --repeat-dir /tmp/p5i-c-phase-0-dense-r3-pp${pp}-probe \
    --production-repeat-dir /tmp/p5i-c-phase-0-dense-r1-pp${pp}-production \
    --production-repeat-dir /tmp/p5i-c-phase-0-dense-r2-pp${pp}-production \
    --production-repeat-dir /tmp/p5i-c-phase-0-dense-r3-pp${pp}-production \
    --out-json /tmp/p5i-c-phase-0-dense-pp${pp}-multirepeat.json
done

uv run python tools/p5i_c_phase0_compose.py \
  --audit-json reports/p5i-c-phase-0-audit.json \
  --out-json /tmp/p5i-c-phase-0-ranking.json \
  --summary-md reports/p5i-c-phase-0-ranking-summary.md \
  --dense-pp128-json /tmp/p5i-c-phase-0-dense-pp128-multirepeat.json \
  --dense-pp512-json /tmp/p5i-c-phase-0-dense-pp512-multirepeat.json
```

Task 6 has no commit unless code changes were required.

---

## Task 7: Close-Out Docs, Memory, And Commit

**Files:**
- Create: `docs/p5i-c-phase-0-ranking-snapshot.md`
- Create: `docs/p5i-c-phase-0-close-out.md`
- Create: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5i_c_phase_0_findings.md`
- Modify: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md`

- [ ] **Step 7.1: Generate committed docs from ranking JSON**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run python tools/p5i_c_phase0_compose.py \
  --audit-json reports/p5i-c-phase-0-audit.json \
  --out-json /tmp/p5i-c-phase-0-ranking.json \
  --summary-md docs/p5i-c-phase-0-ranking-snapshot.md \
  --close-out-md docs/p5i-c-phase-0-close-out.md
```

The generated docs must include:
- per-PP top-N with CI95 and tier labels
- 4-category coverage table
- Phase 1 default rule and suggested candidates
- tied-tier callouts
- vs-omlx delta and caveats
- Dense diagnostic section only when triggered
- all 8 acceptance criteria statuses

- [ ] **Step 7.2: Generate memory file**

```bash
mkdir -p /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory
uv run python tools/p5i_c_phase0_compose.py \
  --audit-json reports/p5i-c-phase-0-audit.json \
  --out-json /tmp/p5i-c-phase-0-ranking.json \
  --memory-md /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5i_c_phase_0_findings.md
```

Then edit `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md` to add one link to `project_p5i_c_phase_0_findings.md`.

- [ ] **Step 7.3: Final verification**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_aggregator/multi_repeat.py tools/p5i_c_pp_tps_envelope.py tools/p5i_c_phase0_compose.py tools/p5h_aggregator/roi_ranking.py tools/p5h_aggregator/tests
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("/tmp/p5i-c-phase-0-ranking.json").read_text())
assert d["phase_1_default_rule"]["triggered_rule"] in {"R1", "R2", "R3"}
assert Path("docs/p5i-c-phase-0-ranking-snapshot.md").exists()
assert Path("docs/p5i-c-phase-0-close-out.md").exists()
print("OK: close-out verification passed")
PY
```

- [ ] **Step 7.4: Commit close-out docs**

```bash
git add docs/p5i-c-phase-0-ranking-snapshot.md docs/p5i-c-phase-0-close-out.md
git commit -m "docs(p5i-c-t5): close Phase 0 gap decomposition"
```

After commit, update the memory file with the final commit SHA:

```bash
T5_SHA=$(git rev-parse --short HEAD)
python3 - <<PY
from pathlib import Path
p = Path("/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5i_c_phase_0_findings.md")
text = p.read_text()
text = text.replace("commit pending", "$T5_SHA")
p.write_text(text)
PY
```

---

## Self-Review Checklist

Run before handing off implementation:

1. Spec coverage:
   - PP scope is PP=128 and PP=512 only.
   - T0/T0.5 handles scheduler, KV cache, attention, and MoE categories.
   - No Lane B allow-list modification is planned.
   - P5i.c capture harness is new and does not mutate P5h T5 harness.
   - Production root denominator is extracted from flag-OFF server logs.
   - pp_tps envelope uses `MAX(within bootstrap CI95, between-sweep half-range)`.
   - Substep uncertainty is surfaced via CI and tied tiers, not hard-gated at ±2%.
   - vs-omlx delta is computed from three omlx repeats.
   - Dense diagnostic uses model/model-dir overrides and never downloads silently.

2. Placeholder scan:

```bash
python3 - <<'PY'
import re
from pathlib import Path

path = Path("docs/superpowers/plans/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition.md")
body = path.read_text().split("## Self-Review Checklist")[0]
patterns = [
    r"TBD",
    r"TODO",
    r"implement later",
    r"fill in",
    r"<[A-Za-z][^>]*>",
    r"fb2d1c0",
    r"parallel omlx",
    r"per substep top-5",
    r"LANE_B_ALLOWED_TRY_SPAN_NAMES.*add",
    r"p5h_t5_attribution_capture\.rs.*mutate",
]
hits = []
for lineno, line in enumerate(body.splitlines(), start=1):
    for pat in patterns:
        if re.search(pat, line):
            hits.append(f"{lineno}: {line}")
if hits:
    raise SystemExit("\n".join(hits))
print("OK: no placeholder/stale-pattern hits")
PY
```

Expected: no output.

3. Fence and Mermaid scan:

```bash
python3 - <<'PY'
from pathlib import Path
text = Path("docs/superpowers/plans/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition.md").read_text()
fence_lines = [line for line in text.splitlines() if line.startswith("```")]
assert len(fence_lines) % 2 == 0
print("OK: fenced code blocks closed")
PY
```

Expected: `OK: fenced code blocks closed`.

Plan is ready for implementation once this checklist passes.
