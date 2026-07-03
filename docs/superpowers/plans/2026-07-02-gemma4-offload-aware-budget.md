# Gemma4 Offload-Aware Memory Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Allow Gemma4 long-context models, including Gemma4 + drafter, to load and run with Active KV offload without treating the full `max_cache_cap` as always-hot KV memory.

**Architecture:** Keep `max_cache_cap` as the logical request cap and introduce a separate hot-resident KV budget when Active KV offload is enabled. Startup and runtime admission charge only the hot-resident window while paged hot/cold tiering keeps older KV pages offloadable. App defaults become memory-aware so the UI no longer sends the full 256K model limit unless the configured runtime can safely support it.

**Tech Stack:** Rust scheduler/model manager, MLX KV cache, Swift Package macOS App, Swift Testing, Cargo tests.

---

### Task 1: Add Offload-Aware Budget Policy

**Files:**
- Modify: `ironmlx/src/core/memory_budget.rs`
- Test: `ironmlx/src/core/memory_budget.rs`

- [x] **Step 1: Add failing tests for hot-resident startup budget**

Add tests that prove a 256K logical cap fails without offload but succeeds when only an 8K hot window is charged:

```rust
#[test]
fn startup_budget_without_offload_rejects_large_logical_cap() {
    let meta = test_meta_gemma4_12b();
    with_total_ram_bytes("137438953472", || {
        let error = validate_startup_budget(1, 262_144, &meta)
            .expect_err("full-resident 256K cache should exceed budget");
        assert_eq!(error.cap, 262_144);
    });
}

#[test]
fn startup_budget_with_offload_charges_hot_resident_cap() {
    let meta = test_meta_gemma4_12b();
    let policy = KvBudgetPolicy::active_kv_offload(8_192);
    with_total_ram_bytes("137438953472", || {
        let state = validate_startup_budget_with_policy(1, 262_144, &meta, policy)
            .expect("offload hot window should fit");
        assert_eq!(state.logical_cap(), 262_144);
        assert_eq!(state.resident_cap(), 8_192);
    });
}
```

- [x] **Step 2: Implement `KvBudgetPolicy` and resident cap accounting**

Add a small policy object:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBudgetPolicy {
    FullResident,
    ActiveKvOffload { resident_cap: usize },
}
```

Add `validate_startup_budget_with_policy`, store logical/resident caps on `BudgetState`, and keep `validate_startup_budget` as a `FullResident` wrapper.

- [x] **Step 3: Run focused memory budget tests**

Run:

```bash
cargo test -p ironmlx memory_budget --lib
```

Expected: all memory budget tests pass.

### Task 2: Wire Policy Through Scheduler Actor

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`
- Modify: `ironmlx/src/core/scheduler.rs`
- Test: `ironmlx/src/core/server/scheduler_actor.rs`
- Test: `ironmlx/src/core/scheduler.rs`

- [x] **Step 1: Add actor test for 256K logical cap with Active KV**

Create a test that spawns the actor with `effective_cap_max=262144`, `ActiveKvOffloadConfig::enabled(...)`, and Gemma4-like metadata. It should fail before policy wiring and pass after.

- [x] **Step 2: Build policy before scheduler construction**

When Active KV is enabled, derive the resident cap from the paged prefix block size and configured/default hot-window page count. Pass `KvBudgetPolicy::active_kv_offload(resident_cap)` into startup validation.

- [x] **Step 3: Use resident charge for runtime admission**

Keep request size validation against logical `effective_cap_max`, but charge `min(row_cap, budget_state.resident_cap())` for `try_admit`. Store the charged bytes in `RequestState.kv_bytes_admitted` so release/park/restore remain balanced.

- [x] **Step 4: Run focused scheduler tests**

Run:

```bash
cargo test -p ironmlx active_kv --lib
cargo test -p ironmlx memory_budget --lib
```

Expected: Active KV and memory budget tests pass.

### Task 3: Improve Error Semantics and Health

**Files:**
- Modify: `ironmlx/src/core/memory_budget.rs`
- Modify: `ironmlx/src/core/server/model_manager.rs`
- Modify: `ironmlx/src/core/server/health.rs`
- Test: `ironmlx/src/core/server/model_manager.rs`
- Test: `ironmlx/src/core/server/health.rs`

- [x] **Step 1: Add test coverage for actionable offload error messages**

Test that startup budget errors distinguish full-resident budget pressure from offload-disabled long-context pressure. The API should no longer collapse every memory budget error into generic GPU memory wording when the remedy is enabling Active KV offload or lowering MAX TOKENS.

- [x] **Step 2: Add budget fields to health**

Expose logical cap, resident cap, and policy name in health snapshots where available. Preserve existing fields for compatibility only if already present; do not add compatibility shims.

- [x] **Step 3: Run model manager and health tests**

Run:

```bash
cargo test -p ironmlx model_manager --lib
cargo test -p ironmlx health --lib
```

Expected: tests pass with clearer memory diagnostics.

### Task 4: App Safe MAX TOKENS Defaults

**Files:**
- Modify: `ironmlx-app/Sources/IronMLXAppCore/ModelParameterStore.swift`
- Modify: `ironmlx-app/Sources/IronMLXAppCore/LocalModelScanner.swift`
- Modify: `ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html`
- Test: `ironmlx-app/Tests/IronMLXAppCoreTests/ModelParameterStoreTests.swift`
- Test: `ironmlx-app/Tests/IronMLXAppCoreTests/LocalModelScannerTests.swift`
- Test: `ironmlx-app/Tests/IronMLXAppCoreTests/DashboardBridgeSettingsTests.swift`

- [x] **Step 1: Add failing tests for Gemma4 12B default cap**

For a 256K Gemma4 model without a saved user `max_tokens`, assert the App chooses a safe default cap rather than `max_position_embeddings`. For Active KV enabled, assert it may offer the larger logical cap while backend startup arguments include `--active-kv-offload`.

- [x] **Step 2: Implement memory-aware default cap**

Keep user-saved `max_tokens` authoritative. Otherwise choose a conservative default based on model architecture, model size class, active KV setting, and device memory. Gemma4 12B on a 128GB Apple Silicon machine should default to 32K without Active KV and allow higher logical caps only when Active KV is enabled.

- [x] **Step 3: Update Dashboard copy**

Clarify that MAX TOKENS is the logical per-request cap and that Active KV offload enables long logical caps by limiting hot residency.

- [x] **Step 4: Run Swift tests**

Run:

```bash
cd ironmlx-app
swift test
```

Expected: all App tests pass.

### Task 5: End-to-End Validation

**Files:**
- Modify: `docs/gemma4-drafter-attribution.md` or add a validation note under `docs/`

- [x] **Step 1: Verify direct backend behavior**

Run App daemon from the new worktree with active KV:

```bash
RUST_LOG=info,ironmlx=debug ./target/release/ironmlx serve \
  --host 127.0.0.1 \
  --port 19086 \
  --max-sequences 1 \
  --paged-prefix-cache-dir /tmp/ironmlx-gemma4-offload-prefix \
  --active-kv-offload \
  --kv-quant k3v4
```

Load `gemma-4-12B-it-4bit + gemma-4-12B-it-assistant-4bit` with `max_cache_cap=262144`. Expected: load succeeds when Active KV is enabled.

- [x] **Step 2: Verify disabled-offload behavior remains safe**

Load the same model with `max_cache_cap=262144` and Active KV disabled. Expected: request fails with an actionable memory budget error.

- [x] **Step 3: Verify text and VL routes**

Run one OpenAI text request and one VL request through the loaded Gemma4 model. Expected: both routes return successfully or fail only for input/model capability reasons unrelated to memory budgeting.

- [x] **Step 4: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all pass.

### Task 6: Commit

**Files:**
- Stage all files changed by this plan.

- [x] **Step 1: Review diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: no whitespace errors; diff is scoped to memory budgeting, scheduler wiring, App defaults, tests, and docs.

- [x] **Step 2: Commit**

Run:

```bash
git add ironmlx ironmlx-app docs
git commit -m "feat(gemma4): support offload-aware long context budget"
```

Expected: commit succeeds on `feat/gemma4-offload-aware-budget`.

---

### Validation Results

- `cargo fmt`: passed.
- `cargo +nightly fmt --all -- --check`: passed.
- `cargo +nightly clippy --all-features --workspace -- -D warnings`: passed.
- `cargo test -p ironmlx --lib`: passed, 772 passed, 17 ignored.
- `cargo build --release`: passed.
- `cd ironmlx-app && swift test`: passed, 153 tests.
- Runtime positive test: `gemma-4-12B-it-4bit + gemma-4-12B-it-assistant-4bit`, `max_cache_cap=262144`, paged prefix cache, Active KV offload, HTTP 200 load.
- Runtime health test: `/healthz` reported `kv_cache_logical_cap_tokens=262144`, `kv_cache_resident_cap_tokens=1024`, `kv_cache_budget_policy=active_kv_offload`, `mtp.enabled=true`, `mtp.draft_tokens=2`.
- Runtime text test: OpenAI text chat completion returned HTTP 200.
- Runtime VL test: OpenAI image_url chat completion returned HTTP 200.
- Runtime negative test: same 262144 cap without Active KV offload returned `code=kv_memory_budget_exceeded`.
