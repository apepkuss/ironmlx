# Qwen3.6 Performance Phase 2 Root Cause Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the root cause of the Qwen3.6 MoE performance gap between ironmlx and omlx before attempting optimizations.

**Architecture:** Treat the Phase 1 gap as a debugging problem: preserve black-box benchmark artifacts, map both serving stacks, then collect targeted attribution evidence from ironmlx. Do not implement performance fixes until the evidence distinguishes model execution, prefill/mask construction, MoE routing/gather-qmm, scheduler admission, and API streaming overhead.

**Tech Stack:** Rust `cargo`, `ironmlx serve`, `iron-bench`, Python helper scripts under `/tmp`, omlx Python source under `/Users/xin/workspace/iron-rivals/omlx`.


---

## Common Inputs

- Worktree: `/Users/xin/workspace/ironmlx-qwen36-perf`
- omlx repo: `/Users/xin/workspace/iron-rivals/omlx`
- Model snapshot: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46`
- Phase 1 artifact root: `/tmp/ironmlx-qwen36-perf-phase01-20260528-175900`
- Phase 2 artifact root pattern: `/tmp/ironmlx-qwen36-perf-phase2-YYYYMMDD-HHMMSS`
- Primary losing cells from Phase 1:
  - `c=1 pp512 tg16`: ironmlx E2E p50 531.02 ms vs omlx 350.99 ms
  - `c=2 pp512 tg16`: ironmlx E2E p50 778.53 ms vs omlx 628.37 ms
  - `c=4 pp512 tg16`: ironmlx E2E p50 1290.84 ms vs omlx 1174.87 ms

## Task 0: Phase 2 Artifact Setup

**Files:**
- Create under `/tmp`, not in git:
  - `meta.env`
  - `reports/`
  - `logs/`
  - `captures/`

- [x] **Step 0.1: Create artifact layout**

Run:

```bash
cd /Users/xin/workspace/ironmlx-qwen36-perf
OUT=/tmp/ironmlx-qwen36-perf-phase2-$(date +%Y%m%d-%H%M%S)
MODEL=$HOME/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
mkdir -p "$OUT"/reports "$OUT"/logs "$OUT"/captures
{
  printf 'OUT=%s\n' "$OUT"
  printf 'MODEL=%s\n' "$MODEL"
  printf 'PHASE1_OUT=%s\n' /tmp/ironmlx-qwen36-perf-phase01-20260528-175900
  printf 'IRONMLX_PORT=18140\n'
  printf 'OMLX_PORT=18141\n'
  printf 'IRONMLX_BRANCH=%s\n' "$(git branch --show-current)"
  printf 'IRONMLX_HEAD=%s\n' "$(git rev-parse HEAD)"
  printf 'OMLX_HEAD=%s\n' "$(git -C /Users/xin/workspace/iron-rivals/omlx rev-parse HEAD)"
  printf 'CREATED_AT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/meta.env"
ln -sfn "$OUT" /tmp/ironmlx-qwen36-perf-phase2-latest
```

Expected: `/tmp/ironmlx-qwen36-perf-phase2-latest/meta.env` exists and points to the current Phase 2 run.

## Task 1: omlx White-Box Map

**Files:**
- Read:
  - `/Users/xin/workspace/iron-rivals/omlx/README.md`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/server.py`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/engine/batched.py`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/scheduler.py`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/request.py`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/output_collector.py`
  - `/Users/xin/workspace/iron-rivals/omlx/omlx/cache/*.py`
- Generate:
  - `$OUT/reports/omlx-whitebox.md`

- [x] **Step 1.1: Locate omlx execution boundaries**

Run:

```bash
cd /Users/xin/workspace/iron-rivals/omlx
source /tmp/ironmlx-qwen36-perf-phase2-latest/meta.env
rg -n "BatchGenerator|generate_step|prefill|KV|cache|schedule|enqueue|stream|chat_completions|mlx_lm|load_model|make_prompt_cache" \
  omlx/server.py omlx/engine/batched.py omlx/scheduler.py omlx/request.py omlx/output_collector.py omlx/cache \
  > "$OUT/reports/omlx-rg-map.txt"
```

Expected: map includes the FastAPI route, batched engine, scheduler, request object, output collector, and cache hooks.

- [x] **Step 1.2: Read the core files and write the white-box report**

Create `$OUT/reports/omlx-whitebox.md` with:

```markdown
# omlx White-Box Map

## Request Path

Describe the route from `/v1/chat/completions` to the batched engine and back to SSE streaming.

## Execution Engine

Record whether omlx delegates core text generation to mlx-lm `BatchGenerator` or another engine, and which object owns prefill/decode stepping.

## Batching and Queueing

Record the request admission policy, maximum concurrent request behavior, and whether decode requests are continuously batched.

## Cache Behavior

Record which caches remain active with `--no-cache` and which are disabled, including prompt cache, paged SSD cache, and hot KV cache.

## Candidate Performance Advantages

List concrete design features that can explain the Phase 1 gaps. Each item must cite file names and line ranges.
```

Expected: the report contains file/line-backed observations only; no optimization proposals yet.

## Task 2: ironmlx White-Box Map

**Files:**
- Read:
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/core/server/openai.rs`
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/core/server/scheduler_actor.rs`
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/core/scheduler.rs`
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/models/qwen3_6_moe/model.rs`
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/models/qwen3_5_moe/model.rs`
  - `/Users/xin/workspace/ironmlx-qwen36-perf/ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Generate:
  - `$OUT/reports/ironmlx-whitebox.md`

- [x] **Step 2.1: Locate ironmlx execution boundaries**

Run:

```bash
cd /Users/xin/workspace/ironmlx-qwen36-perf
source /tmp/ironmlx-qwen36-perf-phase2-latest/meta.env
rg -n "handle_chat|SchedulerActor|Admit|admit_mid|prefill_admitted|step_inner|batched_prefill|forward_on|SparseMoeBlock|gather_quantized" \
  ironmlx/src/core/server/openai.rs \
  ironmlx/src/core/server/scheduler_actor.rs \
  ironmlx/src/core/scheduler.rs \
  ironmlx/src/models/qwen3_6_moe/model.rs \
  ironmlx/src/models/qwen3_5_moe/model.rs \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs \
  > "$OUT/reports/ironmlx-rg-map.txt"
```

Expected: map includes OpenAI route, scheduler actor, scheduler admission/prefill/decode, Qwen3.6 facade, shared Qwen3.5 MoE execution, and MoE gather-qmm path.

- [x] **Step 2.2: Write the ironmlx white-box report**

Create `$OUT/reports/ironmlx-whitebox.md` with:

```markdown
# ironmlx White-Box Map

## Request Path

Describe the route from `/v1/chat/completions` to `SchedulerActor` and back to SSE streaming.

## Qwen3.6 Execution

Record that Qwen3.6 dispatches through the explicit Qwen3.6 facade and delegates numeric execution to the shared Qwen3.5 MoE-VL kernel.

## Batching and Queueing

Record the exact prefill admission, decode step, and mid-admission behavior relevant to `c=1`, `c=2`, and `c=4`.

## MoE Forward Path

Record routed expert steps, sorted-routing threshold, fused gate/up behavior, and shared expert behavior.

## Candidate Bottlenecks

List concrete bottleneck candidates that can explain Phase 1 gaps. Each item must cite file names and line ranges.
```

Expected: the report separates observed behavior from hypotheses.

## Task 3: Root-Cause Synthesis

**Files:**
- Read:
  - `$PHASE1_OUT/reports/summary.md`
  - `$OUT/reports/omlx-whitebox.md`
  - `$OUT/reports/ironmlx-whitebox.md`
- Generate:
  - `$OUT/reports/root-cause-hypotheses.md`
  - update this plan's execution notes

- [x] **Step 4.1: Write root-cause hypotheses**

Create `$OUT/reports/root-cause-hypotheses.md` with:

```markdown
# Qwen3.6 Phase 2 Root-Cause Hypotheses

## Evidence Summary

Summarize Phase 1 measurements and Phase 2 white-box observations.

## Hypotheses

For each hypothesis, include:
- supporting evidence
- contradicting evidence
- minimal next measurement
- expected win ceiling if confirmed

## Ranking

Rank hypotheses by expected impact and evidence strength, not by implementation ease.

## Next Action

Choose exactly one next measurement or instrumentation change.
```

Expected: the next action is evidence-gathering or a targeted optimization only after a root-cause hypothesis is testable.

## Execution Notes

Artifact root:

- `/tmp/ironmlx-qwen36-perf-phase2-20260528-182259`
- Symlink: `/tmp/ironmlx-qwen36-perf-phase2-latest`

Generated reports:

- `reports/omlx-whitebox.md`
- `reports/ironmlx-whitebox.md`
- `reports/root-cause-hypotheses.md`

Key findings:

1. omlx delegates Qwen3.6 MoE numeric execution to `mlx-lm` + `BatchGenerator`; Phase 1 used `--no-cache`, so persistent omlx prefix/SSD cache is not the explanation.
2. ironmlx Qwen3.6 is a productized facade over the shared Qwen3.5 MoE/VL kernel. The OpenAI `pp512` path routes through `SchedulerActor`.
3. Therefore the primary `c=1` gap is model execution/materialization, not scheduler admission, tokenization, or SSE formatting.
4. `IRONMLX_EXPERT_OCCUPANCY_LOG=1` was verified but is timing-perturbing because it materializes routing indices in `routing_sort_pack`; use it only for routing-distribution diagnostics.

Recommended next action:

Build a model-level parity bench against mlx-lm/omlx external-prefill shape, outside HTTP and scheduler, then separately instrument production-build `c=2/c=4` admission-path classification.

- [ ] **Step 4.2: Commit documentation-only results**

Run:

```bash
cd /Users/xin/workspace/ironmlx-qwen36-perf
git add docs/superpowers/plans/2026-05-28-qwen36-performance-phase2-root-cause.md
git commit -m "docs(perf): plan qwen36 phase 2 root cause analysis"
```

Expected: branch contains a documentation commit. Do not push unless Boss asks.
