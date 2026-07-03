# Gemma4 Adaptive Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemma4 drafter-specific adaptive admission policy that exposes b_max=4 capacity without making fixed b_max=4 fresh batching the default behavior.

**Architecture:** Add a small `adaptive_admission` module under `ironmlx/src/core/server`. CLI/App/engine config resolution raises physical capacity for Gemma4 drafter only when the user has not explicitly configured max sequences or an explicit scheduler profile. Scheduler actor receives an `AdaptiveAdmissionPolicy` and uses it for fresh-window and rolling queue decisions.

**Tech Stack:** Rust, Tokio scheduler actor, existing scheduler profile/runtime config, existing Gemma4 drafter AppState.

---

### Task 1: Add Adaptive Policy Unit

**Files:**
- Create: `ironmlx/src/core/server/adaptive_admission.rs`
- Modify: `ironmlx/src/core/server/mod.rs`

- [x] Write tests for disabled behavior, Gemma4 short fresh limit, Gemma4 long chunked fresh limit, Gemma4 long chunked mid-admit, and non-chunked latency cap.
- [x] Run `cargo test -p ironmlx --lib adaptive_admission --release` and verify tests fail because the module does not exist.
- [x] Implement `AdaptiveAdmissionPolicy`, `AdmissionRequestShape`, and constants.
- [x] Run the same tests and verify they pass.

### Task 2: Wire Policy Into Scheduler Actor

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`

- [x] Add failing tests around `drain_admission_queue` and fresh-limit helper calls using the new policy.
- [x] Pass disabled policy through normal/Qwen MTP spawn functions and Gemma4 policy through Gemma4 drafter spawn functions.
- [x] Replace direct calls to model fresh-prefill limits and rolling mid-admit checks with adaptive-aware helpers.
- [x] Run focused scheduler actor tests and verify pass.

### Task 3: Apply Gemma4 Drafter Physical Defaults

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`
- Modify: `ironmlx/src/core/server/model_manager.rs`

- [x] Add failing tests showing Gemma4 drafter default raises physical b_max to 4, explicit `--max-sequences` is preserved, explicit profiles are preserved, and App dynamic model loading uses the same helper.
- [x] Implement the shared helper in `serve.rs`.
- [x] Call it from CLI single-model, engine-pool, and App dynamic loading after MTP/Gemma4 architecture is known.
- [x] Run focused CLI/model-manager tests and verify pass.

### Task 4: Verification

**Files:**
- No production files expected beyond Tasks 1-3.

- [x] Run `cargo fmt`.
- [x] Run `cargo +nightly fmt --all -- --check`.
- [x] Run `cargo +nightly clippy --all-features --workspace -- -D warnings`.
- [x] Run `cargo build --release`.
- [x] Run `python3 scripts/test_gemma4_drafter_active_kv_regression.py`.
- [x] Run at least a short Gemma4 adaptive regression smoke and compare against fixed b2/b4 when time and hardware allow.

Smoke notes:

- E4B + drafter, PP=8192, TG=32, concurrent=4, duration=20s. Adaptive default completed 9 requests at 14.4 tok/s, TTFT p50 13.16s, ITL p50 11.31ms. Explicit b_max=1 completed 9 requests at 14.4 tok/s, TTFT p50 13.61s, ITL p50 11.78ms.
- E4B + drafter, PP=24576, TG=64, concurrent=4, duration=90s. Adaptive default completed 11 requests at 7.82 tok/s, TTFT p50 49.84s, ITL p50 13.99ms. Explicit b_max=1 completed 10 requests at 7.11 tok/s, TTFT p50 53.10s, ITL p50 12.95ms. Treat these as same-machine smoke comparisons, not statistically stable performance claims.

### Task 5: Commit

**Files:**
- Stage all implementation and documentation files.

- [x] Inspect `git diff --stat`.
- [x] Commit with `feat(gemma4): add adaptive drafter admission`.
