# Gemma4 Drafter Rolling Mid-Admit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Gemma4 drafter adaptive admission to actually use physical `b_max=4` during long-prompt concurrent decode by supporting drafter-safe rolling mid-admit.

**Architecture:** Keep the existing adaptive policy unchanged. Move mid-admit operations behind `SchedulerActorMtpMode` so Gemma4 drafter can use a Gemma4-specific mid-admit handle that preserves main KV, `last_hidden`, shared KV, prefix-cache state, and row drafter state. Generic no-MTP and Qwen MTP keep the existing `AdmitMidHandle` path.

**Tech Stack:** Rust, Tokio scheduler actor, MLX scheduler/cache primitives, existing `iron-bench`/Python regression harness.

---

### Task 1: Regression Coverage

**Files:**
- Modify: `scripts/gemma4_drafter_active_kv_regression.py`
- Modify: `scripts/test_gemma4_drafter_active_kv_regression.py`

- [ ] **Step 1: Add a rolling mid-admit profile check**

Add a helper that reads a server log and asserts:

```python
def assert_rolling_mid_admit_profile(log_text: str) -> None:
    if "event=mid_begin" not in log_text:
        raise AssertionError("Gemma4 drafter adaptive run did not start rolling mid-admit")
    if "event=mid_finalize" not in log_text:
        raise AssertionError("Gemma4 drafter adaptive run did not finalize rolling mid-admit")
    if not re.search(r"active_(?:before|after)=[2-9]\\d*", log_text):
        raise AssertionError("Gemma4 drafter adaptive run never exceeded active_count=1")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("gemma4_regression", "scripts/gemma4_drafter_active_kv_regression.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
log = Path("docs/benchmarks/gemma4-adaptive-admission/diagnostics/adaptive-profile-server.log").read_text()
try:
    module.assert_rolling_mid_admit_profile(log)
except AssertionError as exc:
    print(f"EXPECTED_RED: {exc}")
else:
    raise SystemExit("expected helper to reject pre-fix profile log")
PY
```

Expected: unit tests pass, and the old profile log is rejected with “did not start rolling mid-admit”.

### Task 2: Scheduler Gemma4 Drafter Mid-Admit

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Add a Gemma4-specific mid-admit handle**

Create `Gemma4DrafterAdmitMidHandle` with the same chunk-routing fields as `AdmitMidHandle`, plus `last_prompt_hidden`, `last_shared_kv`, and `stats`.

- [ ] **Step 2: Implement `admit_mid_begin_gemma4_drafter`**

Reserve a row, allocate a B=1 temp cache, restore Gemma4 drafter prefix entries through `Gemma4DrafterPrefixCache::try_restore`, build reusable position/vision data, and return the handle.

- [ ] **Step 3: Implement `admit_mid_chunk_gemma4_drafter`**

Run one text or VL hidden forward with shared KV, save Gemma4 drafter prefix entries through `Gemma4DrafterPrefixCache::try_save`, update `last_prompt_hidden`, `last_shared_kv`, and return whether the prompt is complete.

- [ ] **Step 4: Implement `admit_mid_finalize_gemma4_drafter`**

Install the temp cache row, project `last_prompt_hidden`, sample the first token, update the slot, insert `SchedulerGemma4DrafterRowState`, merge stats, and leave pending tokens empty so the next batched drafter step fills the new row together with existing rows.

### Task 3: Actor Integration

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`

- [ ] **Step 1: Add an associated mid-admit handle to `SchedulerActorMtpMode`**

Give the trait `type MidAdmitHandle` and methods for begin/chunk/finalize plus accessors needed by rolling profile logging.

- [ ] **Step 2: Update rolling helpers**

Make `in_flight_mid_admit`, `begin_mid_admit`, `advance_mid_admit_one_chunk`, and `drain_admission_queue` generic over the mode’s handle.

- [ ] **Step 3: Enable Gemma4 drafter rolling mid-admit**

Return `true` from Gemma4 drafter `allow_rolling_mid_admit` and route begin/chunk/finalize through the new scheduler methods.

### Task 4: Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Rust verification**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

- [ ] **Step 2: Regression tests**

Run:

```bash
cargo test -p ironmlx --lib --release
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

- [ ] **Step 3: Runtime profile verification**

Run the E4B + drafter adaptive profile smoke with `IRONMLX_CHUNKED_ROLLING_PROFILE=1`, `prompt_len=8192`, `concurrent=4`, and `max_tokens=32`. Expected log: `mid_begin > 0`, `mid_finalize > 0`, and at least one `active_before` or `active_after` value greater than 1.

- [ ] **Step 4: Performance sanity**

Run a short A/B for `baseline_b1` and `adaptive_default` on a long prompt. Expected result: adaptive profile must no longer be indistinguishable from b1 in scheduler profile logs, and must not regress request success or produce drafter state errors.
