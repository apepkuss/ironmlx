# Gemma4 Drafter Batched Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade Gemma4 assistant drafter serving for `b_max > 1`, with `b_max=2` as the first acceptance target and higher `b_max` supported through the existing scheduler queue/admission model.

**Architecture:** Keep the existing Gemma4 `b_max=1` direct stream semantics for CLI and benchmark paths, but route serving requests through `SchedulerActor` when Gemma4 drafter is enabled. Add a Gemma4-specific scheduler MTP mode that reuses scheduler admission, per-row cache adoption, paged prefix cache, prefix LRU, and Active KV offload, while maintaining Gemma4 drafter row state (`last_hidden`, `shared_kv`, pending tokens, adaptive draft budget).

**Tech Stack:** Rust, MLX arrays/KV caches, Axum HTTP server, existing `Scheduler`, `SchedulerActor`, Gemma4 dense/VL model implementations, Swift app tests for launch/load parameter plumbing.

## Global Constraints

- Branch/worktree: implement in `/Users/xin/workspace/ironmlx-backend-gemma4-drafter-batched-serving` on `feat/gemma4-drafter-batched-serving`.
- No compatibility shims unless explicitly requested.
- Preserve Gemma4 text and Gemma4 VL request paths.
- Preserve paged prefix cache, prefix LRU, and Active KV offload semantics.
- Use `b_max=2` as the first runtime acceptance target; do not hard-code a maximum of 2.
- Rust validation before completion: `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, `cargo build --release`.
- App validation before completion: `swift test` under `ironmlx-app`.

---

### Task 1: Remove CLI-level Gemma4 drafter `b_max > 1` rejection

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`

**Interfaces:**
- Consumes: `resolve_serve_mtp_config(args, architecture, raw_config, scheduler_config) -> Result<Option<ServeMtpConfig>>`
- Produces: Gemma4 + MTP accepts `SchedulerServeConfig { b_max: 2, .. }`

- [ ] **Step 1: Write the failing test**

Change `serve_mtp_config_rejects_gemma4_batched_scheduler` into an acceptance test named `serve_mtp_config_accepts_gemma4_batched_scheduler`:

```rust
#[test]
fn serve_mtp_config_accepts_gemma4_batched_scheduler() {
    let temp_dir = unique_temp_dir("serve-mtp-gemma4-batched");
    std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
    let mut args = base_args();
    args.mtp_model_dir = Some(temp_dir.clone());

    let cfg = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Gemma4,
        &serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}),
        SchedulerServeConfig {
            b_max: 2,
            ..SchedulerServeConfig::default()
        },
    )
    .expect("resolve")
    .expect("enabled");

    assert_eq!(cfg.model_dir, temp_dir);
    std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
}
```

- [ ] **Step 2: Verify RED**

Run: `cargo test -p ironmlx serve_mtp_config_accepts_gemma4_batched_scheduler --lib`

Expected: fail because `resolve_serve_mtp_config` still rejects Gemma4 MTP unless `b_max == 1`.

- [ ] **Step 3: Implement minimal production change**

Remove only this guard:

```rust
if architecture == crate::models::ModelArchitecture::Gemma4 && scheduler_config.b_max != 1 {
    bail!("ironmlx serve Gemma4 --mtp-model-dir currently requires --max-sequences 1");
}
```

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p ironmlx serve_mtp_config_accepts_gemma4_batched_scheduler --lib`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/cli/serve.rs
git commit -m "test(gemma4): accept batched drafter serve config"
```

### Task 2: Add Gemma4 drafter scheduler mode

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`
- Modify: `ironmlx/src/core/server/mod.rs`

**Interfaces:**
- Consumes: `SchedulerActorMtpMode<M>`
- Produces: `Gemma4DrafterSchedulerActor` used by `build_gemma4_drafter_app_state`

- [ ] **Step 1: Write the failing actor/state tests**

Add tests in `scheduler_actor.rs` proving that a Gemma4 drafter mode calls scheduler-side Gemma4 drafter prefill/step counters rather than the no-MTP path. Add a server-state test in `mod.rs` proving `build_gemma4_drafter_app_state(..., b_max=2, ...)` no longer rejects.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: fail because the Gemma4 scheduler mode and builder path do not exist.

- [ ] **Step 3: Implement mode wiring**

Add a mode struct:

```rust
struct SchedulerActorGemma4Drafter {
    drafter: crate::models::gemma4::Gemma4AssistantModel,
    cfg: MtpSpeculativeConfig,
}
```

Implement `SchedulerActorMtpMode<crate::models::Gemma4Model>` by delegating to new scheduler methods:

```rust
sched.prefill_admitted_gemma4_drafter_batch(model, &self.drafter, self.cfg)
sched.step_gemma4_drafter_batch(model, &self.drafter)
```

Add `spawn_scheduler_actor_with_gemma4_drafter` and active-KV variant mirroring `spawn_scheduler_actor_with_mtp`.

- [ ] **Step 4: Replace Gemma4 direct serving state wiring**

In `build_gemma4_drafter_app_state`, remove the `b_max != 1` rejection and build `base` via `build_app_state` with a Gemma4 drafter spawner. Keep `Gemma4DrafterAppState` fields temporarily so handlers can be migrated in Task 4.

- [ ] **Step 5: Verify GREEN**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/server/scheduler_actor.rs ironmlx/src/core/server/mod.rs
git commit -m "feat(gemma4): wire drafter scheduler actor"
```

### Task 3: Implement Gemma4 drafter row state in Scheduler

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`
- Modify: `ironmlx/src/models/gemma4/drafter.rs`

**Interfaces:**
- Consumes: `Gemma4AssistantModel::forward_on`, `Gemma4Model::forward_text_hidden_with_shared_kv_on`, `Gemma4Model::forward_vl_hidden_with_shared_kv_on`
- Produces:
  - `Scheduler::prefill_admitted_gemma4_drafter_batch`
  - `Scheduler::step_gemma4_drafter_batch`
  - `Scheduler::gemma4_drafter_stats`

- [ ] **Step 1: Write RED scheduler tests**

Add tests using the existing fake model style to prove:

- `b_max=2` prefill stores two Gemma4 drafter row states.
- `step_gemma4_drafter_batch` emits at most one event per unfinished row.
- a finished row keeps its pending tokens/state isolated from another active row.
- paged prefix cache exact-hit path can restore/save Gemma4 drafter last hidden per row.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: fail because the scheduler methods do not exist.

- [ ] **Step 3: Extract reusable Gemma4 direct stream helpers**

Move private logic from `Gemma4DrafterGenerationStream` into helper functions that can be called by scheduler code without owning a detokenizer:

- `gemma4_drafter_prefill_single_request`
- `gemma4_drafter_fill_window_for_row`
- `gemma4_shared_kv_from_cache_on` remains available inside the module or is exposed `pub(crate)`.

The helpers must return typed state and counters, not text.

- [ ] **Step 4: Implement batched scheduler by temp-row reuse**

Follow the Qwen MTP pattern: build a temporary `Scheduler<Gemma4Model>` per active row, run Gemma4 drafter single-row prefill/step, adopt the temp cache row back into the main scheduler cache, and store per-row Gemma4 drafter state in a scheduler-owned `HashMap<usize, Gemma4DrafterSchedulerRowState>`.

- [ ] **Step 5: Preserve text + VL behavior**

Use the same prefill helper for text and VL. For decode verification, keep text-token verification because generated continuation tokens after multimodal prefill are text tokens; preserve the existing direct stream behavior.

- [ ] **Step 6: Preserve prefix/active-KV behavior**

Use existing scheduler paged prefix cache and prefix LRU configs for temp schedulers. Ensure Active KV hot/cold tiering is enabled through `make_model_cache`, not by Gemma4 direct stream local cache setup.

- [ ] **Step 7: Verify GREEN**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add ironmlx/src/core/scheduler.rs ironmlx/src/models/gemma4/drafter.rs
git commit -m "feat(gemma4): add drafter batched scheduler state"
```

### Task 4: Route Gemma4 drafter OpenAI and Anthropic requests through scheduler

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs`
- Modify: `ironmlx/src/core/server/anthropic.rs`
- Modify: `ironmlx/src/core/server/mod.rs`

**Interfaces:**
- Consumes: existing `serve_via_scheduler_stream` and `serve_via_scheduler_unary`
- Produces: Gemma4 drafter text/VL requests use scheduler actor when serving, including stream and unary modes.

- [ ] **Step 1: Write RED route tests**

Add tests proving Gemma4 drafter route selection returns scheduler stream/unary routes. The tests should not construct real models; test route enum selection and handler-visible state flags.

- [ ] **Step 2: Verify RED**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: fail because Gemma4 drafter handlers still construct `Gemma4DrafterGenerationStream` directly.

- [ ] **Step 3: Implement route migration**

For both OpenAI and Anthropic Gemma4 drafter handlers, build the same `GenerateRequest` as before, but call scheduler-serving helpers using `state.base`. Remove direct blocking `Gemma4DrafterGenerationStream` use from server request paths.

- [ ] **Step 4: Preserve health counters**

Use scheduler actor MTP counters in `/healthz`. Keep direct stream counters only if direct stream remains for CLI/benchmark paths; server health should report scheduler MTP counters.

- [ ] **Step 5: Verify GREEN**

Run: `cargo test -p ironmlx gemma4_drafter --lib`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/server/openai.rs ironmlx/src/core/server/anthropic.rs ironmlx/src/core/server/mod.rs
git commit -m "feat(gemma4): route drafter serving through scheduler"
```

### Task 5: End-to-end validation and performance smoke tests

**Files:**
- Modify tests only if validation exposes a real issue.

**Interfaces:**
- Produces: evidence that `b_max=2` Gemma4 drafter works for text and VL with prefix cache/Active KV combinations.

- [ ] **Step 1: Run required Rust checks**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

- [ ] **Step 2: Run app tests**

```bash
cd ironmlx-app
swift test
```

- [ ] **Step 3: Run focused runtime smoke tests**

Use local Gemma4 E4B/12B checkpoints when available:

- `gemma-4-e4b-it-4bit + drafter`, `b_max=2`, text prompt.
- `gemma-4-e4b-it-4bit + drafter`, `b_max=2`, VL prompt when image assets are available.
- `gemma-4-12b-it-4bit + drafter`, `b_max=2`, text prompt, with Active KV offload on for long context.

- [ ] **Step 4: Record results**

Summarize:

- load success/failure
- prefill/decode/e2e if benchmark data is available
- MTP health counters (`prefill_count`, `step_count`, `drafted_tokens`, `accepted_draft_tokens`)
- any remaining limits

- [ ] **Step 5: Final commit**

```bash
git status --short
git commit --allow-empty -m "test(gemma4): validate batched drafter serving"
```

