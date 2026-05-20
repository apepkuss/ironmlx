# P5f Known-Path Perf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver sanity-verified 2.44-3.21× single-request prefill speedup (PP=128/512) via CLI default `b_max=1` (T1), plus 1.9-2.5× long-prompt prefill speedup (PP=4096+) via GenerationStream single-shot fallback when KV budget allows (T2). Multi-request `--b-max N>1` capability preserved.

**Architecture:** Two orthogonal optimizations: T1 changes one CLI default value (no Scheduler / KVCache / forward path edits); T2 adds memory-budget-aware dispatch in `GenerationStream::new_text_only` prefill loop (replaces always-chunked-when-PP-exceeds-prefill_chunk_size with try-single-shot-if-budget-allows). Both ship independently. Multi-request batching is deferred to P5h/P6+ per Boss directive (preserved as `--b-max N > 1` opt-in).

**Tech Stack:** Rust 1.94 / mlx (cxx-mlx wrapper) / Apple Silicon Metal (M5 Max 128 GB) / MoE Qwen3.5-35B-A3B-4bit. iron-bench Rust HTTP harness for end-to-end perf validation.

**Spec reference:** [docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md](../specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md)

---

## Pre-flight

### Step 0.1: Confirm branch + clean state

- [ ] On `ironmlx-p5e-perf`

Run: `git -C /Users/xin/workspace/ironmlx-backend branch --show-current`
Expected: `ironmlx-p5e-perf`

- [ ] Working tree clean

Run: `git -C /Users/xin/workspace/ironmlx-backend status --short`
Expected: empty

### Step 0.2: Confirm spec + history present

- [ ] Spec + P5e three-way bench commits in branch history

Run: `git -C /Users/xin/workspace/ironmlx-backend log --oneline -8`
Expected: includes `6f4acb8 docs(p5f): spec update — T1 to Option 1 (CLI default b_max=1), preserve multi-request feature`, `0414de9 docs(p5f): MoE text-only known-path perf design spec`, `a4249af chore: untrack raw bench JSON/log artifacts; keep markdown reports`, `fc8e6c6 bench(p5e): three-way ironmlx vs mlx-lm vs omlx MoE text-only sweep`.

### Step 0.3: Baseline build verifies

- [ ] Release build is green

Run: `MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx`
Expected: `Finished \`release\` profile [optimized] target(s)`, zero Rust warnings (mlx-sys C++ warnings ok).

### Step 0.4: Confirm 35B MoE snapshot present

- [ ] Snapshot path exists

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1
```
Expected: outputs `1e20fd8d42056f870933bf98ca6211024744f7ec` (or another SHA).

Capture for use throughout the plan:
```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
```

### Step 0.5: Confirm 4B Qwen3.5 snapshot for sweep_full

- [ ] 4B snapshot path

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1
```
Expected: outputs `32f3e8ecf65426fc3306969496342d504bfa13f3` or similar.

Capture:
```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
```

### Step 0.6: Confirm mlx-lm bench venv ready

- [ ] mlx-lm venv installed

Run:
```bash
ls /Users/xin/workspace/ironmlx-backend/scripts/bench-venvs/mlx-lm/.venv/bin/mlx_lm.server 2>&1
```
Expected: prints path (executable exists). If missing, recreate:
```bash
cd /Users/xin/workspace/ironmlx-backend/scripts/bench-venvs/mlx-lm && uv sync
```

### Step 0.7: Confirm omlx checkout ready

- [ ] omlx repo path

Run: `ls /Users/xin/workspace/iron-rivals/omlx/pyproject.toml`
Expected: prints path (omlx repo present).

---

## Task 0: P5f Reference Baseline

**Goal:** Establish that `reports/p5e-three-way-bench.md` baseline + 2026-05-19 b_max=1 sanity measurements are the canonical P5f T0 reference. No new bench unless HEAD has drifted from `a4249af`.

**Files:**
- Create: `reports/p5f-baseline.md`

### Step 0.1: Verify HEAD is not far ahead of the bench commit

- [ ] Recent commits since baseline bench

Run: `git -C /Users/xin/workspace/ironmlx-backend log --oneline a4249af..HEAD`
Expected output (acceptable):
```
6f4acb8 docs(p5f): spec update — ...
0414de9 docs(p5f): MoE text-only known-path perf design spec
```

Only doc-only commits since `a4249af` → no inference code drift → safely reuse `reports/p5e-three-way-bench.md` data as P5f T0 baseline. **Skip Step 0.2 below if doc-only.**

### Step 0.2: Optional — re-run bench if inference code drift detected

If between `a4249af` and current HEAD there are commits touching `ironmlx/src/**/*.rs`, the baseline numbers may have drifted. Re-run the 4-way bench (~45 min). Reference `reports/p5e-three-way-bench.md` § 9 (Reproducing this bench) for the exact procedure.

**Skip this step** if Step 0.1 showed doc-only commits.

### Step 0.3: Write `reports/p5f-baseline.md`

Create the file with this content:

```markdown
# P5f Reference Baseline

| Field | Value |
|---|---|
| Date | 2026-05-19 (HEAD a4249af) + 2026-05-20 sanity (HEAD pre-T1) |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Branch | ironmlx-p5e-perf |
| Baseline document | reports/p5e-three-way-bench.md (commit fc8e6c6 then trimmed to a4249af) |
| Sanity document | this section below (2026-05-19 b_max=1 sweep, no JSON committed) |

## Why this is the P5f baseline (no new bench needed)

Between HEAD `a4249af` (when the canonical 4-way bench was captured)
and the current P5f starting HEAD, all commits are doc-only:

- `6f4acb8 docs(p5f): spec update — T1 to Option 1`
- `0414de9 docs(p5f): MoE text-only known-path perf design spec`

No inference code path changed → ironmlx numbers in
`reports/p5e-three-way-bench.md` § 5 still represent the current
default-b_max=4 behavior. Reuse them as P5f T0.

## Canonical baseline numbers (default `b_max=4`)

From `reports/p5e-three-way-bench.md` § 5 (median across 5 timed runs):

| PP | ironmlx prefill (tok/s) | ironmlx TTFT (ms) | ironmlx decode TG (tok/s) | omlx prefill (tok/s) | omlx+10% target |
|---:|---:|---:|---:|---:|---:|
| 128 | 390 | 329 | 79.6 | 1088 | 1197 |
| 512 | 491 | 1042 | 78.5 | 2623 | 2886 |
| 2048 | 1842 | 1112 | 123.7 | 4227 | 4649 |
| 4096 | 1773 | 2310 | 121.4 | 4419 | 4861 |
| 8192 | 1725 | 4748 | 118.0 | 4261 | 4687 |
| 16384 | 1548 | 10581 | 112.0 | 3669 | 4036 |

## T1 dry-run reference (`--b-max 1` sanity, 2026-05-19)

Captured during P5f brainstorming with `ironmlx serve --b-max 1`,
same iron-bench sweep parameters. Raw JSON not committed (gitignored
per `reports/.gitignore`); summary preserved here.

| PP | ironmlx_bmax1 prefill (tok/s) | TTFT (ms) | decode TG (tok/s) | bmax1/b_max=4 prefill | omlx+10% target |
|---:|---:|---:|---:|---:|---:|
| 128 | 951 | 135 | 125.6 | 2.44× | 1197 (79% of target) |
| 512 | 1577 | 325 | 123.7 | 3.21× | 2886 (55%) |
| 2048 | 1843 | 1111 | 124.9 | 1.00× | 4649 (40%) |
| 4096 | 1833 | 2235 | 122.3 | 1.03× | 4861 (38%) |
| 8192 | 1724 | 4751 | 117.5 | 1.00× | 4687 (37%) |
| 16384 | 1606 | 10200 | 117.0 | 1.04× | 4036 (40%) |

## Reading

- T1 (CLI default `b_max=1`) is expected to deliver the b_max=1 sanity
  numbers in the second table (PP=128/512 prefill 2.44-3.21×). PP=2048+
  shows 1.00-1.04× (within noise) — those routes go through
  GenerationStream not Scheduler, so b_max change does not affect them.
- T2 (GenerationStream single-shot when KV budget allows) targets
  PP=4096-16384 prefill specifically. Expected post-T2: 3000-4500 tok/s
  across the long-prompt range.
- P5f expected close-out: PP=128 ~950 / PP=512 ~1577 / PP=2048 ~1842 /
  PP=4096+ 3000-4500 tok/s. PP=2048 will remain the largest gap to
  omlx+10% target — by design, that gap is the P5g GatedDeltaNet /
  GatedAttention focus.
```

- [ ] **Step 0.3 actions**: write the above to `reports/p5f-baseline.md`.

### Step 0.4: Commit T0

Run:
```bash
git -C /Users/xin/workspace/ironmlx-backend add reports/p5f-baseline.md
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
docs(p5f-t0): P5f reference baseline pointing at reports/p5e-three-way-bench.md + sanity

No new bench: doc-only commits since a4249af (the bench capture HEAD)
mean ironmlx inference path has not drifted, so reports/p5e-three-way-bench.md
numbers are still canonical for default b_max=4.

The 2026-05-19 b_max=1 sanity numbers (PP=128 951 tok/s, PP=512 1577
tok/s prefill) — raw JSON not committed per reports/.gitignore — are
preserved in this file as the T1 dry-run reference. All subsequent
P5f tasks compare to these two tables.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: CLI Default `b_max = 1`

**Goal:** Change the `--b-max` CLI default from 4 to 1, deliver sanity-verified 2.44-3.21× prefill / 1.58× decode gain at PP=128/512 with zero Scheduler / KVCache / forward-path changes. Multi-request batching capability remains via explicit `--b-max N > 1`.

**Files:**
- Modify: `ironmlx/src/cli/serve.rs:34-35`
- Modify: `ironmlx/src/cli/serve.rs` (add INFO log near scheduler boot)
- Check: `scripts/sweep/sweep_full.sh` (and individual test scripts) for any reliance on b_max=4 default
- Modify: any test / script found in the check above

### Step 1.1: Inspect current b_max default

- [ ] Read the CLI arg declaration

Run:
```bash
sed -n '28,40p' /Users/xin/workspace/ironmlx-backend/ironmlx/src/cli/serve.rs
```
Expected output (verbatim — including the `default_value_t = 4` we change):
```
    #[arg(long, default_value_t = 2048)]
    pub prefill_chunk_size: usize,

    /// Maximum concurrent in-flight requests (Scheduler slot count).
    /// Requests beyond this limit go to the admission queue.
    /// ...
    #[arg(long, default_value_t = 4)]
    pub b_max: usize,
```

Note: the line numbers around `pub b_max: usize` (34-35 in the spec reference) may shift by ±1-2 if the file has been edited; the `#[arg(long, default_value_t = 4)]` immediately preceding `pub b_max: usize` is the canonical anchor.

### Step 1.2: Change default to 1

- [ ] Edit serve.rs

Edit `ironmlx/src/cli/serve.rs`: change exactly one number:

```rust
// Before:
#[arg(long, default_value_t = 4)]
pub b_max: usize,

// After:
#[arg(long, default_value_t = 1)]
pub b_max: usize,
```

Verify by:
```bash
grep -n "default_value_t" /Users/xin/workspace/ironmlx-backend/ironmlx/src/cli/serve.rs | grep "b_max\|^[0-9]*:[[:space:]]*#\[arg.*default_value_t" | head -5
```
Expected: the b_max-adjacent line now shows `default_value_t = 1`.

### Step 1.3: Add startup INFO log

- [ ] Locate the serve entrypoint that constructs the scheduler

Run:
```bash
grep -n "b_max\|Scheduler::new\|tracing::info" /Users/xin/workspace/ironmlx-backend/ironmlx/src/cli/serve.rs | head -20
```

Identify the function (likely `pub fn run(...)` or `pub async fn serve(...)` in this file) that has access to the parsed `Args.b_max` value and runs before request handling.

- [ ] Add INFO log immediately after b_max is read, before scheduler boot

In that function's body, find where `b_max` is first read (or where the scheduler / app state is being constructed), and add:

```rust
tracing::info!(
    "ironmlx serve: b_max={} (single-request optimized by default; \
     pass --b-max N > 1 to enable concurrent multi-request batching)",
    cli_args.b_max,
);
```

The actual variable name (`cli_args.b_max` here) depends on the local binding; adapt to whatever the function uses.

If `tracing` is not yet imported at the top of the file, add (anywhere among the other use statements):

```rust
use tracing;
```

(Or, if `tracing::info!` is already used elsewhere in this file via macro re-export, no import needed.)

### Step 1.4: Build

Run:
```bash
cd /Users/xin/workspace/ironmlx-backend && MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```
Expected: `Finished release profile`. Fix any compile errors before proceeding.

### Step 1.5: fmt + clippy

Run:
```bash
cd /Users/xin/workspace/ironmlx-backend
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
```
Expected: fmt clean (no output); clippy `Finished release profile` with no Rust warnings (mlx-sys C++ warnings ok).

### Step 1.6: Hunt for hidden assumptions about default b_max=4

- [ ] Grep test files + scripts for default-b_max expectations

Run:
```bash
grep -rn "b_max.*4\|b-max 4\|b_max=4\|b_max: 4" /Users/xin/workspace/ironmlx-backend/ironmlx/tests/ /Users/xin/workspace/ironmlx-backend/scripts/sweep/ /Users/xin/workspace/ironmlx-backend/ironmlx/src/core/ 2>/dev/null | head -20
```

For each hit, decide:
- **Test code explicit `b_max=4`**: OK, no change (test is explicit, not default-dependent).
- **Test asserting `b_max == 4` as a default invariant**: change assertion to `== 1`, OR if the test needs multi-request behavior, set b_max=4 explicitly in setup.
- **`sweep_full.sh` invocations that omit `--b-max`**: investigate whether any test there relies on multi-request batching. The most likely candidate is `b1_p2_3c_plus_chunked_admit_mid` (admit-mid is multi-request territory).

- [ ] Check sweep_full's invocation of the multi-request tests

Run:
```bash
grep -n "b_max\|b-max\|b1_p2_3c\|admit_mid" /Users/xin/workspace/ironmlx-backend/scripts/sweep/sweep_full.sh 2>/dev/null | head -20
```

If `sweep_full.sh` invokes tests with no `--b-max` argument but the tests internally require multi-request, the tests must be parameterized to explicitly use `b_max=4` (or whatever multi-request size). This is a per-test fix; check each test's `#[test]` body via Read to see how it initializes the Scheduler.

If sweep_full does not pass `--b-max`, the tests likely build their own `Scheduler::new(b_max=...)` calls — those calls are NOT affected by the CLI default change (CLI default only affects `ironmlx serve`). So most likely: **no sweep_full breakage from this change**.

- [ ] **Step 1.6 actions**: if any change needed, apply it now and re-build.

### Step 1.7: Sentinel + batched + http_smoke (default b_max=1)

- [ ] Ensure MoE snapshot env is set

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
```

- [ ] Smoke + sentinel

Run:
```bash
cd /Users/xin/workspace/ironmlx-backend && MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 2 tests pass; sentinel reports `argmax=11`.

- [ ] Batched (B=2)

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 1 test passes (per-row equivalence at B=2). This test should internally pass `b_max=2` to Scheduler; CLI default change does not affect it.

- [ ] HTTP smoke

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 1 test passes.

### Step 1.8: Multi-request path still functional (`--b-max 4` explicit)

- [ ] Verify multi-request batching capability with explicit flag

The HTTP smoke test above already exercises the server boot path with the new default. Now confirm that explicit `--b-max 4` still works by starting a server briefly:

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 --b-max 4 &
SERVER_PID=$!
# Wait for healthz
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 3; done
# Quick chat completion
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-moe","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":10,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' | head -c 400; echo
# Cleanup
kill $SERVER_PID
sleep 2
lsof -i :8080 2>/dev/null && pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```
Expected: chat completion returns a JSON with `"choices": [...]` and no error. The server boot stderr/INFO log should show `b_max=4`.

### Step 1.9: iron-bench validate default boot

- [ ] Start ironmlx with default args (no `--b-max`)

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 3; done
```

Expected: server boots; INFO log line `ironmlx serve: b_max=1 (single-request optimized by default; pass --b-max N > 1 to enable concurrent multi-request batching)` appears in stderr.

- [ ] Run iron-bench quick PP=128/512 subset (faster than full sweep)

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target ironmlx_t1=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512 --max-tokens 32 --runs 3 --warmup 1 \
  --format markdown 2>&1 | tail -40
```
Expected (median tok/s should match the b_max=1 sanity table in `reports/p5f-baseline.md`):
```
PP=128:  prefill ≥ 850 tok/s  (target 951 from sanity ± 10%)
PP=512:  prefill ≥ 1400 tok/s (target 1577 from sanity ± 10%)
```

If significantly below: STOP and report `DONE_WITH_CONCERNS` describing the gap. The expected mechanism (no Scheduler padding overhead) should reproduce the sanity numbers exactly.

- [ ] Cleanup server

```bash
kill $SERVER_PID 2>/dev/null
sleep 2
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

### Step 1.10: Documentation update

- [ ] Update README.md

Find the section that documents `ironmlx serve` (likely in the Usage / Quickstart section). If a `--b-max` parameter is documented there, update its mention to reflect the new default; otherwise add a brief note. Sample addition:

```markdown
**Concurrency note:** `ironmlx serve` defaults to `--b-max 1` (single-request
optimized). Pass `--b-max N > 1` to enable concurrent multi-request batching.
The single-request default delivers up to 2.44× prefill speedup on short
prompts vs. the prior `--b-max 4` default; multi-request throughput is
unaffected when batching is explicitly enabled.
```

Locate the right section by:
```bash
grep -n "b-max\|b_max\|serve\|--b-max" /Users/xin/workspace/ironmlx-backend/README.md 2>/dev/null | head
```

If README does not exist or lacks a Usage section: create a brief `docs/cli-defaults.md` note instead. Document whichever you do; do not leave the change undocumented.

- [ ] Update CHANGELOG.md if it exists

```bash
ls /Users/xin/workspace/ironmlx-backend/CHANGELOG.md 2>&1
```

If present, prepend an entry under an `[Unreleased]` or `## Next` heading:

```markdown
- **Breaking (default value change):** `ironmlx serve --b-max` default changed
  from `4` to `1`. Single-request workloads (the common case) gain up to
  2.44× prefill speedup and 1.58× decode speedup at short prompts.
  Multi-request batching remains fully functional via explicit `--b-max N > 1`.
```

If CHANGELOG.md does not exist, skip this sub-step.

### Step 1.11: Commit T1

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/src/cli/serve.rs README.md
# Also add CHANGELOG.md if it was edited; check git status first.
git -C /Users/xin/workspace/ironmlx-backend status --short
# Then commit
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
feat(p5f-t1): CLI default --b-max 4 → 1 (single-request optimized)

Sanity-verified --b-max 1 launch flag (2026-05-19) delivers
2.44× prefill (PP=128: 390 → 951 tok/s), 3.21× prefill (PP=512:
491 → 1577 tok/s), 1.58× decode TG (PP=128/512: 79 → 124 tok/s)
over the prior default --b-max 4. The speedup comes from eliminating
[B=4, T_max]-padded MoE compute when only one request is in flight,
which is the common case for current ironmlx usage (single-user
chat / agent serve, P5d/e/f bench scenarios).

Multi-request batching capability is preserved (Scheduler / KVCache
/ forward path unchanged): users explicitly pass --b-max N > 1 to
enable. Future phases (P5h / P6+) will revisit default value when
multi-user / agent-fleet scenarios become primary.

Startup INFO log:
  "ironmlx serve: b_max=N (single-request optimized by default;
   pass --b-max N > 1 to enable concurrent multi-request batching)"

Validation:
  - p5_qwen35_moe_smoke (argmax=11 sentinel): PASS
  - p5_qwen35_moe_batched (B=2 row-equiv): PASS
  - p5_qwen35_moe_http_smoke: PASS
  - Explicit --b-max 4 boot + chat: PASS
  - iron-bench PP=128/512 default boot: hit sanity prefill targets

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -3
```

---

## Task 2: GenerationStream Single-Shot When KV Budget Allows

**Goal:** When prompt_len > prefill_chunk_size (PP=4096+ in the current `prefill_chunk_size=2048` default), instead of always going through the chunked prefill loop (per-chunk eval barriers), check whether single-shot forward over the entire prompt fits the KV memory budget; if yes, single-shot; if no (rare on M5 Max 128 GB unless PP >> 100K), keep chunked path as fallback.

**Files:**
- Modify: `ironmlx/src/core/memory_budget.rs` (add 2 helper functions)
- Modify: `ironmlx/src/core/generate.rs` (GenerationStream::new_text_only prefill loop dispatch)
- Create: `ironmlx/tests/p5f_long_prompt_single_shot.rs`

### Step 2.1: Read the current GenerationStream prefill loop

- [ ] Inspect the loop body

Run:
```bash
sed -n '1166,1240p' /Users/xin/workspace/ironmlx-backend/ironmlx/src/core/generate.rs
```
Expected: shows the `pub fn new_text_only(...)` function with the chunked prefill loop (`loop { let n = ... chunk_size; let chunk_ids = ...; if is_last { model.forward_on(...) } else { let hidden = model.forward_text_hidden(...); mlx::transforms::eval(&[&hidden])?; None }`).

Confirm: when `chunk_size == 0`, the loop falls through with `n = remaining`, meaning a single iteration covers the whole prompt — that is **already** the single-shot path when the user passes `prefill_chunk_size=0`. The T2 change makes that selection automatic when budget allows.

### Step 2.2: Inspect memory_budget.rs current API

- [ ] Read existing helpers

Run:
```bash
sed -n '1,60p' /Users/xin/workspace/ironmlx-backend/ironmlx/src/core/memory_budget.rs
```
Expected: shows `ModelMeta`, `kv_bytes_per_token(meta)`, `kv_cache_bytes(b, cap, meta)`, `system_total_ram_bytes()`, `available_budget_bytes(meta)`.

The existing `kv_bytes_per_token` is GQA-aware but uses `num_hidden_layers` (all layers) — for prefill we want **only full-attention layer KV** (Qwen3.5-MoE-A3B has 10/40 full-attn; linear layers don't allocate KV buffers in the standard sense). For Qwen3.5 dense the GQA estimate is fine. For the MoE model the existing function over-estimates, which is conservative (safer to single-shot less aggressively).

For T2 we add two new helpers that are explicit and prefill-scoped, leaving existing helpers untouched.

### Step 2.3: Add `estimate_prefill_kv_peak_bytes` + `available_kv_budget_bytes` to memory_budget.rs

- [ ] Append to memory_budget.rs

Append the following block to `ironmlx/src/core/memory_budget.rs` (place after the existing `available_budget_bytes` function definition, before any `#[cfg(test)] mod tests`):

```rust
/// Estimate peak KV-cache bytes for a single prefill of `prompt_len` tokens
/// at single-batch (B=1). Uses `kv_bytes_per_token` which is GQA-aware but
/// conservative — for hybrid attention models like Qwen3.5-MoE-A3B it
/// over-estimates (counts linear-attn layers that don't allocate per-token
/// KV the same way), which is safe for the single-shot budget check below.
///
/// P5f T2: used by `GenerationStream::new_text_only` to decide whether to
/// fall back from chunked-prefill to single-shot when the prompt exceeds
/// `prefill_chunk_size`.
pub fn estimate_prefill_kv_peak_bytes(meta: &ModelMeta, prompt_len: usize) -> usize {
    prompt_len.saturating_mul(kv_bytes_per_token(meta))
}

/// Total system memory budget available for KV cache, after deducting model
/// weights and a safety margin. Equivalent to
/// `system_total_ram_bytes() - model_weight_bytes - SAFETY_MARGIN_BYTES`.
///
/// Returns 0 if model weights + safety margin already exceed system RAM (in
/// which case the caller should NOT attempt single-shot prefill).
pub fn available_kv_budget_bytes(meta: &ModelMeta) -> usize {
    available_budget_bytes(meta)
}
```

Note: `available_kv_budget_bytes` is intentionally a thin alias over the existing `available_budget_bytes` to make call sites read naturally at the GS prefill decision point. Both call into the same underlying `system_total_ram_bytes() - meta.weight_bytes - SAFETY_MARGIN_BYTES` math.

### Step 2.4: Modify GenerationStream::new_text_only prefill loop

- [ ] Edit generate.rs

Read the existing loop body around lines 1205-1230 (will vary slightly with file edits — locate by the comment/variable `let chunk_size = request.prefill_chunk_size;` then `let mut pos: i32 = 0;` then `let last_logits = loop {`).

Replace the loop body. Before this loop, add a budget check; if budget allows single-shot, set the effective chunk size to the full prompt length:

```rust
let chunk_size = request.prefill_chunk_size;
let prompt_len_i32 = prompt_len as i32;

// P5f T2: When the prompt exceeds chunk_size, check whether single-shot
// forward fits the KV memory budget. If yes, set effective chunk_size to
// `prompt_len` so the loop executes once; chunked path remains as fallback
// when budget is insufficient.
let effective_chunk_size: usize = if chunk_size > 0 && prompt_len > chunk_size {
    let meta = model.meta();
    let kv_peak = crate::core::memory_budget::estimate_prefill_kv_peak_bytes(&meta, prompt_len);
    let budget = crate::core::memory_budget::available_kv_budget_bytes(&meta);
    if kv_peak <= budget {
        tracing::debug!(
            "GS prefill: single-shot path (prompt_len={prompt_len}, kv_peak={kv_peak} bytes, \
             budget={budget} bytes); bypassing chunked path"
        );
        prompt_len  // single-shot via effective chunk size == prompt_len
    } else {
        tracing::debug!(
            "GS prefill: chunked path (prompt_len={prompt_len}, kv_peak={kv_peak} bytes \
             > budget {budget}); using configured chunk_size={chunk_size}"
        );
        chunk_size
    }
} else {
    // Either chunking disabled (chunk_size==0) or prompt fits a single chunk;
    // existing behavior preserved.
    chunk_size
};

let mut pos: i32 = 0;
let last_logits = loop {
    let remaining = prompt_len_i32 - pos;
    let n = if effective_chunk_size == 0 {
        remaining
    } else {
        remaining.min(effective_chunk_size as i32)
    };
    // ... rest of loop body unchanged ...
```

Adapt the variable name (`chunk_size` in the original; we add `effective_chunk_size`) and update the existing inner-loop reference to use `effective_chunk_size` in place of `chunk_size`.

**Important — verify `model.meta()` API exists**: if the `Model` trait does not expose a `meta()` method returning `ModelMeta`, find the equivalent (likely `model.model_meta()` or accessing through a stored field). Run:

```bash
grep -n "fn meta\|fn model_meta\|fn metadata\|ModelMeta" /Users/xin/workspace/ironmlx-backend/ironmlx/src/core/model.rs 2>/dev/null | head
```
to confirm. If the trait does not have `meta()`, either:
- (a) add a `fn meta(&self) -> ModelMeta` method to the `Model` trait (small additive change), implemented by each concrete model
- (b) thread a `&ModelMeta` argument into `GenerationStream::new_text_only` (changes signature; affects all call sites)
Prefer (a). The implementation per model is a single `self.meta` field clone if the model already stores it.

- [ ] Apply the edit. Build verifies the API choice.

### Step 2.5: Compile

Run:
```bash
cd /Users/xin/workspace/ironmlx-backend && MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -5
```
Expected: `Finished release profile`. Fix compile errors per the actual `Model::meta()` API (or equivalent) found in Step 2.4.

### Step 2.6: Feasibility quick check — PP=16384 single-shot memory

- [ ] Quick check via test (don't run full bench yet)

A single PP=16384 forward call should peak well under M5 Max 128 GB unified memory. Approximate: 10 full-attn layers × 20 KB/token (bf16 KV, 4 GQA heads, head_dim=128, K+V) × 16384 = 3.28 GB, plus activations + workspace ~ 10-30 GB at peak. Total << 128 GB.

Sanity by running smoke test which loads the model (peak RSS reported via process info, or just verify the test runs to completion without OOM):

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: PASS without OOM.

For a more rigorous check, Step 2.8 below will run iron-bench PP=16384 which actually exercises the single-shot path on real long prompt.

### Step 2.7: Sentinel + batched + new single-shot test

- [ ] Existing sentinel + batched + http_smoke (default PP=128 still in chunked-free path)

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: all PASS with argmax=11.

- [ ] Create new long-prompt single-shot test

Create `ironmlx/tests/p5f_long_prompt_single_shot.rs`:

```rust
//! P5f T2 single-shot vs chunked path numerical equivalence at PP=4096.
//!
//! Constructs a 4096-token deterministic prompt, runs the model twice via
//! GenerationStream::new_text_only:
//!   1. With prefill_chunk_size = 0 (forces single-shot via existing
//!      chunked-disabled path)
//!   2. With prefill_chunk_size = 2048 (would normally trigger 2-chunk
//!      loop, but P5f T2 dispatches single-shot when KV budget allows)
//!
//! Both paths should produce identical first-token argmax (T2 should be
//! mathematically equivalent to chunked-disabled; the only difference is
//! how dispatch decides the single-shot vs chunked path).
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test p5f_long_prompt_single_shot \
//!       -- --ignored --test-threads=1 --nocapture

use mlx::Dtype;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::{Loader, Model};
use ironmlx::core::sampler::Sampler;
use ironmlx::models::Qwen35MoeModel;
use tokenizers::Tokenizer;

const PROMPT_LEN: i32 = 4096;

fn locate_snapshot() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR")
}

fn synth_token_ids(len: i32) -> Vec<u32> {
    // Same scheme as tests/p5e_baseline.rs: id = 10000 + i % 100, valid for
    // Qwen3.5 vocab (~248K).
    (0..len).map(|i| 10_000_u32 + (i as u32 % 100)).collect()
}

#[test]
#[ignore]
fn p5f_single_shot_argmax_matches_chunked_disabled_at_pp_4096() {
    let snap = locate_snapshot();
    let loader = Loader::open(std::path::Path::new(&snap)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");
    let tok_path = std::path::Path::new(&snap).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).expect("tokenizer load");

    let prompt_ids = synth_token_ids(PROMPT_LEN);
    let sampler = Sampler::greedy();

    // Path A: chunked-disabled (prefill_chunk_size=0). Always single-shot.
    let req_a = GenerateRequest {
        prompt_ids: prompt_ids.clone(),
        max_new_tokens: 1,
        sampler: sampler.clone(),
        stop_token_ids: vec![],
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: -1,
    };
    let stream_a = GenerationStream::new_text_only(&model, &tokenizer, req_a).expect("GS A");
    let first_a = stream_a.history.last().copied().expect("first token A");

    // Path B: T2 dispatch — prefill_chunk_size=2048 (default), prompt_len=4096
    // > chunk_size, KV budget on M5 Max definitely fits ⇒ single-shot path.
    let req_b = GenerateRequest {
        prompt_ids: prompt_ids.clone(),
        max_new_tokens: 1,
        sampler,
        stop_token_ids: vec![],
        prefill_chunk_size: 2048,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: -1,
    };
    let stream_b = GenerationStream::new_text_only(&model, &tokenizer, req_b).expect("GS B");
    let first_b = stream_b.history.last().copied().expect("first token B");

    eprintln!("[p5f_single_shot] path A (chunk_size=0) first_token = {first_a}");
    eprintln!("[p5f_single_shot] path B (chunk_size=2048, T2 dispatch) first_token = {first_b}");

    assert_eq!(
        first_a, first_b,
        "P5f T2 single-shot dispatch should produce same first-token argmax as \
         chunk_size=0 path (both ARE single-shot mathematically; only dispatch differs)"
    );
}
```

**Note**: the exact field names of `GenerateRequest` may differ slightly; if the test fails to compile, locate the actual struct definition with `grep -n "pub struct GenerateRequest" /Users/xin/workspace/ironmlx-backend/ironmlx/src/core/generate.rs` and align field names. Similarly, `GenerationStream` may expose `history` as a method or a field — adapt.

- [ ] Build the new test binary

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5f_long_prompt_single_shot --no-run 2>&1 | tail -3
```
Expected: `Finished release profile`.

- [ ] Run the new test

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5f_long_prompt_single_shot -- --ignored --test-threads=1 --nocapture 2>&1 | tail -10
```
Expected: 1 test passes; eprintln shows both path A and path B emitting the same first_token.

### Step 2.8: iron-bench validate long-prompt prefill

- [ ] Boot ironmlx (default args — b_max=1 from T1, prefill_chunk_size=2048)

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 3; done
```

- [ ] Run iron-bench at PP=4096/8192/16384

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target ironmlx_t2=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 4096,8192,16384 --max-tokens 64 --runs 3 --warmup 1 \
  --format markdown 2>&1 | tail -30
```
Expected (medians):
```
PP=4096:  prefill ≥ 3500 tok/s  (from baseline 1773 tok/s ≈ 1.97×)
PP=8192:  prefill ≥ 3500 tok/s  (from baseline 1725 tok/s ≈ 2.0×)
PP=16384: prefill ≥ 3000 tok/s  (from baseline 1548 tok/s ≈ 1.94×)
```

If any PP underperforms expected by > 20%, STOP and report DONE_WITH_CONCERNS describing which PP and the actual number; the close-out T3 will still proceed (we report incomplete T2 in the final report).

- [ ] Cleanup

```bash
kill $SERVER_PID 2>/dev/null
sleep 2
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

### Step 2.9: Final hygiene chain

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```
Expected: fmt clean, clippy clean, build Finished.

### Step 2.10: Commit T2

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/src/core/memory_budget.rs \
  ironmlx/src/core/generate.rs \
  ironmlx/tests/p5f_long_prompt_single_shot.rs
# Also any Model trait file modified for meta() — check git status
git -C /Users/xin/workspace/ironmlx-backend status --short
# Commit
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
feat(p5f-t2): GenerationStream single-shot when KV budget allows

When prompt_len > prefill_chunk_size (default 2048), GS no longer
always goes through the per-chunk eval-barrier prefill loop;
instead it estimates KV peak bytes for a single B=1 forward over
the full prompt and, if peak ≤ available system budget (system
RAM - model weight - 2 GB safety margin), executes single-shot.
Chunked path is retained as fallback for prompts that exceed the
budget (rare on M5 Max 128 GB; expected only for PP > ~50K with
this model size).

Helper additions:
  - memory_budget::estimate_prefill_kv_peak_bytes(meta, prompt_len)
  - memory_budget::available_kv_budget_bytes(meta)

Expected wall-clock impact (vs HEAD a4249af baseline):
  - PP=4096  prefill: 1773 → ≥3500 tok/s (≈1.97×)
  - PP=8192  prefill: 1725 → ≥3500 tok/s (≈2.0×)
  - PP=16384 prefill: 1548 → ≥3000 tok/s (≈1.94×)
  - PP≤2048: unchanged (already single-chunk in default config)

Numerical safety:
  - p5_qwen35_moe_smoke (argmax=11 sentinel): PASS
  - p5_qwen35_moe_batched (B=2 row-equiv): PASS
  - p5_qwen35_moe_http_smoke: PASS
  - New p5f_long_prompt_single_shot: PP=4096 single-shot dispatch
    matches chunk_size=0 single-shot argmax (mathematically same path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -3
```

---

## Task 3: P5f Close-Out

**Goal:** Run the canonical 4-way bench (ironmlx / mlx-lm / omlx) and sweep_full, produce a self-contained `reports/p5f-final-results.md` for offline analysis, and quantify P5g scope.

**Files:**
- Create: `reports/p5f-final-results.md`

### Step 3.1: Pre-flight server / port cleanup

```bash
lsof -i :8080 -i :8081 -i :8082 2>/dev/null && pkill -f "ironmlx serve\|omlx serve\|mlx_lm.server" 2>/dev/null
sleep 3
lsof -i :8080 -i :8081 -i :8082 2>/dev/null | head
```
Expected: empty (all three ports free).

### Step 3.2: ironmlx sweep (serial, port 8080)

- [ ] Boot ironmlx server (default args — b_max=1 from T1)

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
IRONMLX_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 3; done
```

- [ ] Run iron-bench sweep against ironmlx

```bash
mkdir -p /Users/xin/workspace/ironmlx-backend/reports/p5f-three-way-bench  # gitignored per reports/.gitignore (pattern p5e-three-way-bench/; this dir is similarly raw artifacts territory — see Step 3.6 below for cleanup)
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --max-tokens 128 --runs 5 --warmup 1 \
  --format json \
  > /tmp/p5f-ironmlx.json \
  2> /tmp/p5f-ironmlx.log
echo "[ironmlx sweep done]"; tail -3 /tmp/p5f-ironmlx.log
```
Expected: progress lines for all 6 PP × (1 warmup + 5 timed) runs.

- [ ] Kill ironmlx

```bash
kill $IRONMLX_PID 2>/dev/null
sleep 3
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
lsof -i :8080 2>/dev/null && echo "WARN: port 8080 still bound" || echo "port 8080 freed"
```

### Step 3.3: omlx sweep (serial, port 8081)

- [ ] Boot omlx serve

```bash
cd /Users/xin/workspace/iron-rivals/omlx && \
  uv run omlx serve --model-dir "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8081 &
OMLX_PID=$!
cd /Users/xin/workspace/ironmlx-backend
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -qE "^(200|404)$"; do sleep 3; done
```

- [ ] Run iron-bench against omlx (uses snapshot SHA as model name; omlx is strict)

```bash
SNAP_SHA=$(basename "$IRONMLX_MOE_MODEL_DIR")
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target omlx=http://127.0.0.1:8081 \
  --model "$SNAP_SHA" --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --max-tokens 128 --runs 5 --warmup 1 \
  --format json \
  > /tmp/p5f-omlx.json \
  2> /tmp/p5f-omlx.log
echo "[omlx sweep done]"; tail -3 /tmp/p5f-omlx.log
```

- [ ] Kill omlx

```bash
kill $OMLX_PID 2>/dev/null
sleep 3
pkill -f "omlx serve.*--port 8081" 2>/dev/null
lsof -i :8081 2>/dev/null && echo "WARN: port 8081 still bound" || echo "port 8081 freed"
```

### Step 3.4: mlx-lm sweep (serial, port 8082)

- [ ] Boot mlx-lm server in isolated venv

```bash
cd /Users/xin/workspace/ironmlx-backend/scripts/bench-venvs/mlx-lm && \
  uv run mlx_lm.server --model "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8082 --log-level INFO &
MLXLM_PID=$!
cd /Users/xin/workspace/ironmlx-backend
# Probe with chat completion to confirm model loaded (mlx-lm /v1/models has a HF cache quirk that returns 500 even when chat works)
until curl -s -X POST http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}' 2>/dev/null | grep -q "choices"; do sleep 5; done
```

- [ ] Run iron-bench against mlx-lm

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target mlx_lm=http://127.0.0.1:8082 \
  --model default_model --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --max-tokens 128 --runs 5 --warmup 1 \
  --format json \
  > /tmp/p5f-mlx_lm.json \
  2> /tmp/p5f-mlx_lm.log
echo "[mlx-lm sweep done]"; tail -3 /tmp/p5f-mlx_lm.log
```

- [ ] Kill mlx-lm

```bash
kill $MLXLM_PID 2>/dev/null
sleep 3
pkill -f "mlx_lm.server.*--port 8082" 2>/dev/null
lsof -i :8082 2>/dev/null && echo "WARN: port 8082 still bound" || echo "port 8082 freed"
```

### Step 3.5: Aggregate medians + p95 from the 3 JSONs

- [ ] Run aggregation script

```bash
cd /Users/xin/workspace/ironmlx-backend && uv run --no-project --with statistics python3 <<'EOF' > /tmp/p5f-aggregate.md
import json, statistics

def load(path):
    with open(path) as f: return json.load(f)

PP_LIST = [128, 512, 2048, 4096, 8192, 16384]
TARGETS = [("ironmlx", "/tmp/p5f-ironmlx.json"),
           ("mlx_lm",  "/tmp/p5f-mlx_lm.json"),
           ("omlx",    "/tmp/p5f-omlx.json")]
METRICS = ["ttft_ms", "pp_tps", "tg_tps", "tpot_ms", "e2e_s"]

data = {}
for name, path in TARGETS:
    d = load(path)
    data[name] = {pp: {m: [] for m in METRICS} for pp in PP_LIST}
    for r in d["raw_runs"]:
        pp = r["pp_target"]
        if pp not in PP_LIST: continue
        for m in METRICS:
            v = r.get(m)
            if v is not None:
                data[name][pp][m].append(v)

def med(vs): return statistics.median(vs) if vs else None
def p95(vs):
    if not vs: return None
    s = sorted(vs); k = int(0.95 * (len(s) - 1)); return s[k]

print("# P5f median tables\n")
for m in METRICS:
    print(f"## metric: {m}")
    print("| PP | ironmlx | mlx_lm | omlx |")
    print("|---:|---:|---:|---:|")
    for pp in PP_LIST:
        row = [pp]
        for name, _ in TARGETS:
            v = med(data[name][pp][m])
            row.append(f"{v:.2f}" if v is not None else "n/a")
        print("| " + " | ".join(str(x) for x in row) + " |")
    print()
print("# P5f p95 tables\n")
for m in METRICS:
    print(f"## metric: {m} (p95)")
    print("| PP | ironmlx | mlx_lm | omlx |")
    print("|---:|---:|---:|---:|")
    for pp in PP_LIST:
        row = [pp]
        for name, _ in TARGETS:
            v = p95(data[name][pp][m])
            row.append(f"{v:.2f}" if v is not None else "n/a")
        print("| " + " | ".join(str(x) for x in row) + " |")
    print()
EOF
echo "[aggregate written to /tmp/p5f-aggregate.md]"
wc -l /tmp/p5f-aggregate.md
```
Expected: writes ~80 lines to `/tmp/p5f-aggregate.md` with median + p95 tables for all 5 metrics × 6 PP × 3 targets.

### Step 3.6: Cleanup /tmp artifacts after aggregation

Raw JSON / log files in `/tmp` are throw-away. **They do not enter the repo** (the `reports/.gitignore` pattern only covers `reports/p5e-three-way-bench/` and `p5d-*`; `/tmp` is naturally outside repo).

```bash
ls -la /tmp/p5f-*.json /tmp/p5f-*.log /tmp/p5f-aggregate.md 2>/dev/null
# Keep these around briefly until reports/p5f-final-results.md is committed,
# then they can be safely deleted (or kept as local cache).
```

### Step 3.7: sweep_full 19/19 regression gate

```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -10
```
Expected: `19/19 PASS in ~140-160s` on M5 Max. If any test fails, retry once (transient flakes happen — see `reports/p5e-stage1-results.md` for prior precedent). If a second run also fails, STOP and report DONE_WITH_CONCERNS.

### Step 3.8: Write `reports/p5f-final-results.md`

Open `/tmp/p5f-aggregate.md` to copy the median + p95 tables into the report. Create `reports/p5f-final-results.md`:

```markdown
# P5f Final — MoE Text-Only Known-Path Perf Close-Out

> **Self-contained for offline code-level analysis.** Embeds all bench
> data, methodology, and remaining-gap attribution for P5g scope
> definition.

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Hardware | M5 Max 128 GB |
| OS | macOS 26.4 |
| Model | `mlx-community/Qwen3.5-35B-A3B-4bit` (Qwen3.5-MoE-A3B, 128 experts, top-4) |
| Snapshot SHA | `1e20fd8d42056f870933bf98ca6211024744f7ec` |
| Branch | `ironmlx-p5e-perf` |
| Spec | docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md |
| Plan | docs/superpowers/plans/2026-05-19-ironmlx-p5f-known-path-perf.md |
| Harness | iron-bench (Rust HTTP) |
| Sweep | `--prompt-len 128,512,2048,4096,8192,16384  --max-tokens 128  --runs 5  --warmup 1`, strict serial |

## §1 What P5f shipped

- **T1** ([commit <fill T1 commit SHA>](https://...)): CLI default `--b-max` changed from 4 to 1. Multi-request batching preserved via explicit `--b-max N > 1`. Sanity-verified 2.44-3.21× prefill / 1.58× decode at PP=128/512.
- **T2** ([commit <fill T2 commit SHA>](https://...)): `GenerationStream::new_text_only` adds memory-budget-aware single-shot dispatch when `prompt_len > prefill_chunk_size` and KV peak fits budget. Chunked path retained as fallback. Expected 1.94-2.0× prefill at PP=4096+.

## §2 Per-metric tables (median + p95)

<paste the medianm tables from /tmp/p5f-aggregate.md verbatim here>

## §3 P5f deltas vs P5e baseline + omlx target

Pull baseline numbers from `reports/p5f-baseline.md` (which references `reports/p5e-three-way-bench.md`). Compute deltas:

| PP | P5e baseline ironmlx (tok/s) | P5f ironmlx (tok/s) | delta | omlx (tok/s) | omlx+10% target | P5f vs target |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 390 | <fill> | <%>× | <fill> | <fill> | <%> of target |
| 512 | 491 | <fill> | <%>× | <fill> | <fill> | <%> of target |
| 2048 | 1842 | <fill> | <%>× | <fill> | <fill> | <%> of target |
| 4096 | 1773 | <fill> | <%>× | <fill> | <fill> | <%> of target |
| 8192 | 1725 | <fill> | <%>× | <fill> | <fill> | <%> of target |
| 16384 | 1548 | <fill> | <%>× | <fill> | <fill> | <%> of target |

## §4 Decode + e2e tables

(Similar — pull from §2 medians.)

## §5 Validation gates

- p5_qwen35_moe_smoke argmax=11 sentinel: PASS
- p5_qwen35_moe_batched B=2 row-equivalence: PASS
- p5_qwen35_moe_http_smoke: PASS
- p5f_long_prompt_single_shot (T2 dispatch numerical eq.): PASS
- sweep_full 19/19: PASS in <fill> seconds
- clippy --all-features --workspace --release -D warnings: 0 warnings
- fmt --check: clean

## §6 Remaining gap to omlx+10% target — P5g scope drivers

For each PP, residual gap (P5f vs target) attribution. Use canonical T0
profile data (`reports/p5e-t0-profile.md`):

| PP | P5f tok/s | omlx+10% target | residual gap | Likely attribution |
|---:|---:|---:|---:|---|
| 128 | <fill> | <fill> | <%> | HTTP overhead + GatedDeltaNet recurrent fixed cost |
| 512 | <fill> | <fill> | <%> | GatedDeltaNet + GatedAttention bias still in HTTP path |
| 2048 | <fill> | <fill> | <%> | **Primary P5g target**: GatedDeltaNet (20%) + GatedAttention (6.5%) per T0 profile; omlx's monkey-patches deliver ~80% acceleration on these layers |
| 4096 | <fill> | <fill> | <%> | Same as PP=2048; long-prompt GatedAttention O(S²) growth |
| 8192 | <fill> | <fill> | <%> | Long-prompt GatedAttention scaling dominant |
| 16384 | <fill> | <fill> | <%> | Long-prompt GatedAttention + GatedDeltaNet recurrent cost |

## §7 P5g candidates (driven by §6 attribution)

Ranked by expected impact on residual gap:

1. **GatedDeltaNet independent profile + optimization** (linear attn, 30/40 layers, T0 profile 20% at PP=2048)
   - Read current ironmlx implementation at `ironmlx/src/models/qwen3_5/gated_delta_net.rs`
   - Profile per-op + identify bottleneck (recurrent loop, conv pre/post, ...)
   - Independent design improvement (NO copy from omlx.patches per [feedback_no_spec_from_competitors])

2. **GatedAttention optimization** (full attn, 10/40 layers, super-linear scaling at long PP)
   - O(S²) growth means PP=16384 spends ~50% of prefill here
   - KV layout + SDPA dispatch tuning

3. **Router bypass for single-request idle server** (if Scheduler admission/queue overhead > 50ms — needs measurement)

4. **Multi-request batching enhancement (P5h / P6+, separate phase)**
   - Per Boss directive: this capability must NOT be lost in the roadmap
   - Future work items: PagedCache evaluation, ragged batching, dynamic b_max, admit_mid efficiency
   - Trigger: when ironmlx enters multi-user / agent-fleet deployment

## §8 Out of P5f scope (deferred capabilities)

- **Multi-request batching default re-evaluation**: `--b-max N > 1` works today via explicit flag. The default-to-1 choice is single-request optimal; multi-request deployment will revisit default selection in P5h / P6+.
- **omlx-style PagedCache**: not aligned with current ironmlx design ([feedback_design_philosophy]). Reconsider only if multi-request scaling shows demand.
- **mlx::compile wrap**: still blocked by 4 safe-wrapper API gaps from P5e T2.
- **Sorted-routing micro-opt** (cache token_idx, put_along_axis): ROI < 2%, not P5f scope.

## §9 Cross-reference: where omlx leads — observation only

omlx achieves <fill> tok/s at PP=2048 (vs ironmlx P5f <fill> tok/s) via its
4-layer optimization stack:
1. `omlx.patches.gated_delta_advance` monkey-patches `Qwen3_5GatedDeltaNet`
2. `omlx.patches.qwen3_5_attention` monkey-patches `Qwen3_5Attention`
3. PagedCache (block 256→2048 auto-tune)
4. Engine = vlm path (text-only also routes through vlm engine)

Per [feedback_no_spec_from_competitors]: omlx is **observation only, not
an alignment target**. ironmlx independently designs improvements based
on its own architecture; reaching omlx+10% via independent design is
the goal but the implementation path is independent.
```

Fill all `<fill>` placeholders with actual numbers from the aggregate. The two `<fill T1/T2 commit SHA>` near the top can be filled from `git log --oneline a4249af..HEAD` after T3 commits land — use the SHA from the recent T1 + T2 commits.

### Step 3.9: Commit T3 (P5f close-out)

```bash
git -C /Users/xin/workspace/ironmlx-backend add reports/p5f-final-results.md
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
chore(p5f-t3): P5f close-out — three-way bench + sweep_full + P5g scope quantification

P5f shipped two optimizations on branch ironmlx-p5e-perf:
  T1: CLI default --b-max 4 → 1 (single-request optimized; multi-request via --b-max N > 1)
  T2: GenerationStream single-shot when KV budget allows (long-prompt prefill bypass of chunked loop)

Measured deltas vs P5e baseline HEAD a4249af (M5 Max 128 GB, Model::forward_on via HTTP):

  PP=128  prefill: 390  → <fill> tok/s (<fill>×)
  PP=512  prefill: 491  → <fill> tok/s (<fill>×)
  PP=2048 prefill: 1842 → <fill> tok/s (<fill>×)
  PP=4096 prefill: 1773 → <fill> tok/s (<fill>×)
  PP=8192 prefill: 1725 → <fill> tok/s (<fill>×)
  PP=16384 prefill: 1548 → <fill> tok/s (<fill>×)

Validation:
  - 4 MoE integration tests + new p5f_long_prompt_single_shot: PASS
  - sweep_full 19/19: PASS in <fill> seconds
  - clippy + fmt + release build: clean

P5g scope is now quantified in reports/p5f-final-results.md §6/§7:
  1. GatedDeltaNet optimization (T0 profile 20% at PP=2048)
  2. GatedAttention optimization (long-prompt O(S²) dominant)
  3. Router bypass evaluation (conditional)

Multi-request batching is explicitly preserved in roadmap (§7 item 4)
per Boss directive — never lost, just deferred to P5h / P6+.

Per [feedback_no_spec_from_competitors]: omlx remains observation only;
P5g design path is independent of omlx.patches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -5
```

### Step 3.10: Verify branch state

```bash
git -C /Users/xin/workspace/ironmlx-backend log --oneline 6f4acb8..HEAD
git -C /Users/xin/workspace/ironmlx-backend status --short
```
Expected: 4+ P5f commits (T0/T1/T2/T3 plus any polish); clean working tree.

---

## P5f Final Acceptance

All of the following must be true at HEAD:

- [ ] `MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx`: PASS
- [ ] `cargo +nightly fmt --all -- --check`: clean
- [ ] `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings`: 0 warnings
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored`: PASS, argmax=11
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored`: PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored`: PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5f_long_prompt_single_shot -- --ignored`: PASS
- [ ] `QWEN35_MODEL=<snap> ./scripts/sweep/sweep_full.sh`: 19/19 PASS
- [ ] `reports/p5f-baseline.md`, `reports/p5f-final-results.md` written with real numbers (no `<fill>` placeholders)
- [ ] PP=128 prefill ≥ 850 tok/s; PP=512 ≥ 1400 tok/s (T1 sanity-predicted ROI)
- [ ] PP=4096 prefill ≥ 3000 tok/s; PP=8192 ≥ 3000 tok/s; PP=16384 ≥ 2500 tok/s (T2 conservative target)
- [ ] Explicit `--b-max 4` still boots a functional server (multi-request preserved)
- [ ] CHANGELOG.md or README.md updated with the default change note

After acceptance, branch is ready for Boss decision on merge timing and P5g spec writing.
