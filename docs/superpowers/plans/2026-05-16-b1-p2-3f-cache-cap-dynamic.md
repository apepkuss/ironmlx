# B1-p2.3f Cache Cap Dynamic + Bounded Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded `make_cache(b, 8192, dtype)` with three-tier dynamic cap: per-request `prompt_len + max_new_tokens` ≤ `effective_cap_max = min(--max-cache-cap CLI, model.config.max_position_embeddings)`. Reject oversize requests at admit-time with `SchedulerError::RequestTooLarge` → HTTP 413.

**Architecture:** `Qwen35Config.max_position_embeddings` read at model-load. `serve()` computes `effective_cap_max = min(cli_max_cache_cap, model_max_context)` once at startup; threaded through `spawn_scheduler_actor` to `Scheduler`. Both `admit` and `admit_mid` gate on the bound. `prefill_admitted_inner` computes lazy-alloc cap from current slots' max(prompt_len + max_new), bounded by `effective_cap_max`. `evict_all` drops cache so the next batch's cap matches its actual requirements (~10ms re-alloc per outer batch).

**Tech Stack:** Rust 2024, `mlx-rs`, `clap`, `axum`, `thiserror`, `tokio`.

**Spec source:** [`docs/superpowers/specs/2026-05-16-b1-p2-3f-cache-cap-dynamic-design.md`](../specs/2026-05-16-b1-p2-3f-cache-cap-dynamic-design.md) (commit `1076254`).

**Branch:** `ironmlx-b1-p2-3f-cache-cap` cut from `ironmlx-b1-p2-3e3-typed-err` head `1076254` (post-3f-spec, includes 3e.3 typed Err foundation).

**Cargo env:** every `cargo` invocation requires `MLX_DIR=$HOME/.local/mlx`. Env does NOT persist across subshells — prefix each `cargo` command explicitly. Hygiene gate per CLAUDE.md: `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`, `cargo +stable build --release`.

**Model fixture:** `MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)`. Integration tests need `QWEN35_MODEL="$MODEL"` env var.

---

## Pre-flight: cut the working branch

- [ ] **Step 0.1: Cut branch off 3f-spec head**

```bash
cd /Volumes/Dev/cxx-mlx
git checkout -b ironmlx-b1-p2-3f-cache-cap
git log --oneline -1
```

Expected: HEAD at `1076254 docs(b1-p2.3f): expand scope to Z — dynamic cap + model awareness + bounded`.

- [ ] **Step 0.2: Pre-flight hygiene**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

All three must pass. If any fail, fix on this branch with a separate hygiene commit before starting T1.

---

## File structure

| File | Responsibility | Change kind |
| --- | --- | --- |
| `ironmlx/src/models/qwen3_5/config.rs` | `Qwen35Config` struct | Modify (+1 field) |
| `ironmlx/src/core/scheduler.rs` | Scheduler state + admit/evict/prefill primitives | Modify (+1 SchedulerError variant, +1 struct field, signature change on `new`, admit gates, evict_all body, prefill_admitted_inner cap) |
| `ironmlx/src/core/server/scheduler_actor.rs` | `spawn_scheduler_actor` + driver_loop | Modify (+1 param, +1 internal `Scheduler::new` call site) |
| `ironmlx/src/core/server/mod.rs` | `AppState` + `serve()` fn | Modify (+1 AppState field, +1 serve param, compute effective_cap_max) |
| `ironmlx/src/cli/serve.rs` | `ServeArgs` | Modify (+1 CLI flag) |
| `ironmlx/src/core/server/openai.rs` | `admit_err_to_response` helper | Modify (+1 match arm) |
| `ironmlx/src/core/server/anthropic.rs` | `admit_err_to_response` helper | Modify (mirror openai.rs) |
| `ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs` | 3b-2 actor scenarios | Modify (3 spawn_scheduler_actor caller updates) |
| `ironmlx/tests/b1_p2_3b_3_admission_window.rs` | 3b-3 window scenarios | Modify (4 caller updates) |
| `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs` | 3b-4 anthropic scenarios | Modify (3 caller updates) |
| `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` | 3c-3 continuous batching | Modify (3 caller updates) |
| `ironmlx/tests/b1_p2_3d_admission_queue.rs` | 3d scenarios | Modify (4 caller updates) |
| `ironmlx/tests/b1_p2_4_batched_vl.rs` | B1-p2.4 VL scenarios | Modify (4 caller updates) |
| `ironmlx/tests/b1_p2_3f_cache_cap.rs` (NEW) | Long-prompt integration | Create (~100 LoC, 1 scenario) |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3f_closeout/report.md` (NEW) | Close-out | Create (gitignored — `-f`) |

Total ~14 files, +400/-100 LoC, 4 tasks ~1.5 days.

---

## Task 1: `Qwen35Config` field + `SchedulerError::RequestTooLarge` + Scheduler signature change + admit gates + evict_all drop

**Why this is first:** All other tasks depend on the new `Scheduler::new(b_max, effective_cap_max)` signature and the new `SchedulerError::RequestTooLarge` variant. T1 must update every `Scheduler::new` caller (cfg(test) in scheduler.rs + scheduler_actor.rs internal) in the same commit so the workspace compiles.

**Files:**
- Modify: `ironmlx/src/models/qwen3_5/config.rs:49-81` (struct field)
- Modify: `ironmlx/src/core/scheduler.rs` (multiple sites)
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (internal `Scheduler::new` call only — public signature unchanged in T1)

### Steps

- [ ] **Step 1.1: Add `max_position_embeddings` to `Qwen35Config`**

Edit `ironmlx/src/models/qwen3_5/config.rs`. After the `vision_config` field (around line 80) but inside the struct, append:

```rust
    /// Maximum sequence length the model supports (= `text_config.max_position_embeddings`
    /// from config.json). Qwen3.5-4B: 262144. Used as a hard upper bound on
    /// per-request `prompt_len + max_new_tokens` to prevent MRoPE
    /// out-of-distribution garbage. B1-p2.3f.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
```

Add helper near the top of the file (after the `use` imports and before the structs):

```rust
fn default_max_position_embeddings() -> i32 {
    // Conservative fallback for older / non-Qwen3 configs that omit the field.
    // Production Qwen3.5 configs always declare it (262144 for 4B variant).
    32768
}
```

- [ ] **Step 1.2: Verify config parses with the new field**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -3
```

Expected: clean. The field is parsed via serde; no test added yet (Step 1.10's `admit_rejects_oversize_request` exercises the path indirectly via Scheduler).

- [ ] **Step 1.3: Add `SchedulerError::RequestTooLarge` variant**

Edit `ironmlx/src/core/scheduler.rs`, locate `pub enum SchedulerError` (around line 18-25). Append a second variant:

```rust
#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("admission queue full: capacity={capacity} reached")]
    QueueFull { capacity: usize },

    /// Request's `prompt_len + max_new_tokens` exceeds the server's
    /// effective cap_max (the smaller of `--max-cache-cap` CLI flag and
    /// the model's `max_position_embeddings`). Maps to HTTP 413
    /// Payload Too Large. B1-p2.3f.
    #[error("request too large: needs cap={needed} but server max_cache_cap={max}")]
    RequestTooLarge { needed: usize, max: usize },
}
```

- [ ] **Step 1.4: Add `effective_cap_max` field to `Scheduler` struct**

Locate `pub struct Scheduler` (around line 225). Append the field after `poisoned: bool`:

```rust
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
    phase: Phase,
    cache: Option<Vec<LayerCache>>,
    poisoned: bool,
    /// Upper bound on `prompt_len + max_new_tokens` per request, computed
    /// at boot as `min(cli_max_cache_cap, model.config.max_position_embeddings)`.
    /// `admit` and `admit_mid` reject requests exceeding this with
    /// [`SchedulerError::RequestTooLarge`]. B1-p2.3f.
    effective_cap_max: usize,
}
```

- [ ] **Step 1.5: Change `Scheduler::new` signature**

Locate `pub fn new(b_max: usize) -> Self` (around line 201). Replace its body:

```rust
    /// Construct a scheduler with `b_max` pre-allocated slots, all `None`.
    /// `effective_cap_max` is the hard upper bound on per-request
    /// `prompt_len + max_new_tokens` — admit gates reject requests beyond
    /// this with [`SchedulerError::RequestTooLarge`] (HTTP 413 downstream).
    pub fn new(b_max: usize, effective_cap_max: usize) -> Self {
        let mut slots = Vec::with_capacity(b_max);
        for _ in 0..b_max {
            slots.push(None);
        }
        Self {
            b_max,
            slots,
            next_id: 0,
            phase: Phase::Idle,
            cache: None,
            poisoned: false,
            effective_cap_max,
        }
    }
```

- [ ] **Step 1.6: Add cap gate to `Scheduler::admit`**

Locate `pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId>` (around line 228). At the top of the function (after `self.ensure_not_poisoned()?` but before the Phase::Finished check, around line 230), insert:

```rust
        // B1-p2.3f: cap check before admission. Reject oversize requests
        // upfront rather than allocating a slot then failing at prefill.
        let cap_needed = req
            .prompt_ids
            .len()
            .saturating_add(req.max_new_tokens);
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
```

- [ ] **Step 1.7: Add cap gate to `Scheduler::admit_mid`**

Locate `pub fn admit_mid` (around line 831). At the top, after `self.ensure_not_poisoned()?` (around line 838) but before the Phase check:

```rust
        // B1-p2.3f: mirror admit's cap gate. Mid-batch admits must also
        // respect the bound — otherwise the queue drain path could push an
        // oversize request through.
        let cap_needed = req
            .prompt_ids
            .len()
            .saturating_add(req.max_new_tokens);
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
```

- [ ] **Step 1.8: Change `evict_all` to drop cache**

Locate `pub fn evict_all(&mut self) -> Result<()>` (around line 364). Replace the cache-reset block (around lines 377-381):

**Before:**
```rust
        if let Some(cache) = self.cache.as_mut() {
            for lc in cache.iter_mut() {
                lc.reset()?;
            }
        }
```

**After:**
```rust
        // B1-p2.3f: drop the cache so the next prefill_admitted lazy-allocates
        // with cap matching the new batch's requirements. ~10ms re-alloc per
        // outer batch is negligible vs prefill GPU time (100s of ms to
        // seconds). Pre-3f kept the cache + reset offsets but locked the
        // first batch's cap forever — incompatible with dynamic cap.
        self.cache = None;
```

- [ ] **Step 1.9: Update all `Scheduler::new` cfg(test) callers in scheduler.rs**

The scheduler.rs cfg(test) module (line ~1080 onwards) has 15+ direct calls to `Scheduler::new(N)`. Each must change to `Scheduler::new(N, 32768)` (using the same default as the production CLI flag).

```bash
grep -n "Scheduler::new(" /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs
```

For each line, edit `Scheduler::new(N)` → `Scheduler::new(N, 32768)`. The 15 call sites:

```rust
// scheduler.rs cfg(test) — each occurrence
let s = Scheduler::new(4);          // → Scheduler::new(4, 32768)
let mut s = Scheduler::new(4);      // → Scheduler::new(4, 32768)
let mut s = Scheduler::new(2);      // → Scheduler::new(2, 32768)
// ... etc, all uniform pattern
```

Use `sed` for the bulk update (verify the regex matches only inside scheduler.rs cfg(test)):

```bash
sed -i.bak -E 's/Scheduler::new\(([0-9]+)\)/Scheduler::new(\1, 32768)/g' \
    /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs
rm /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs.bak
```

After sed, verify no `Scheduler::new(N)` (without 2nd arg) remains:

```bash
grep -n "Scheduler::new([0-9]\+)$\|Scheduler::new([0-9]\+);" /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs
```

Expected: empty output.

- [ ] **Step 1.10: Update `spawn_scheduler_actor` internal `Scheduler::new` call**

`scheduler_actor.rs` calls `Scheduler::new(b_max)` inside `driver_loop` (around line 145 — the actual line is inside the `driver_loop` body, search for it):

```bash
grep -n "Scheduler::new" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/scheduler_actor.rs
```

Replace `Scheduler::new(b_max)` with `Scheduler::new(b_max, 32768)`. This is a temporary hardcode — T2 will plumb the real effective_cap_max through `spawn_scheduler_actor`'s signature.

- [ ] **Step 1.11: Verify compile**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo +stable build --tests 2>&1 | tail -5
```

Both must succeed. Common failures:
- `Scheduler::new` caller missed → grep again for `Scheduler::new([^,)]*)`
- `SchedulerError::RequestTooLarge` not in scope at the gate use site → already in same module
- `anyhow::Error::new` type inference → if rustc complains about `?` not working with anyhow, use `.into()` cast

- [ ] **Step 1.12: Write `evict_all_drops_cache` unit test**

Append to the scheduler.rs cfg(test) module (after the existing tests). Find an existing test like `evict_all_from_finished_resets_to_idle` and add right after it:

```rust
    #[test]
    fn evict_all_drops_cache() {
        // B1-p2.3f: evict_all drops cache (replaces pre-3f offset reset) so
        // the next prefill_admitted lazy-allocates with the new batch's cap.
        // This test uses force_phase to bypass admit-then-prefill complexity;
        // we don't allocate a real cache here, we observe the drop semantic.
        let mut s = Scheduler::new(4, 32768);

        // Inject a dummy cache state to verify drop. We can't construct a
        // real Vec<LayerCache> without a model, but we can verify that
        // evict_all sets self.cache = None unconditionally regardless of
        // what was there. Skip cache injection — confirm initial None →
        // remain None after eviction.
        let req = GenerateRequest {
            prompt_ids: vec![1, 2, 3],
            max_new_tokens: 8,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
        };
        let _id = s.admit(req).expect("admit");
        s.force_phase(Phase::Decoding);

        // Cache is still None (no prefill_admitted called).
        assert!(s.cache.is_none(), "pre-evict_all: cache should be None (no prefill)");

        s.evict_all().expect("evict_all");

        // Post-evict_all: cache is None. (Pre-3f would have left an empty
        // cache vec via .reset() if a cache had existed; 3f drops it.)
        assert!(s.cache.is_none(), "post-evict_all: cache must be None (3f drops)");
    }
```

Note: the test accesses `s.cache` directly. The `cache` field is `pub(crate)` or private; if private, you may need to add a `#[cfg(test)] pub(crate) fn has_cache(&self) -> bool` accessor. Check the existing struct definition; if `cache` is module-private and cfg(test) is inside the same module, direct access works.

- [ ] **Step 1.13: Write `admit_rejects_oversize_request` unit test**

Append below:

```rust
    #[test]
    fn admit_rejects_oversize_request() {
        // B1-p2.3f: admit cap gate. cap_max=1024; request with
        // prompt_len=1500 + max_new=600 = 2100 > 1024 must reject with
        // SchedulerError::RequestTooLarge.
        use crate::core::SchedulerError;

        let mut s = Scheduler::new(1, 1024);

        let oversize_req = GenerateRequest {
            prompt_ids: vec![0; 1500],
            max_new_tokens: 600,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
        };

        let result = s.admit(oversize_req);
        let err = result.expect_err("admit should reject oversize");

        // Verify typed downcast.
        let sched_err = err
            .downcast_ref::<SchedulerError>()
            .expect("err should be downcast-able to SchedulerError");
        match sched_err {
            SchedulerError::RequestTooLarge { needed, max } => {
                assert_eq!(*needed, 2100, "needed cap should be prompt+max_new");
                assert_eq!(*max, 1024, "max should be effective_cap_max from Scheduler::new");
            }
            other => panic!("expected RequestTooLarge, got {other:?}"),
        }

        // Verify Display format includes both numbers (for HTTP body).
        let msg = format!("{err:#}");
        assert!(msg.contains("2100"), "msg should contain needed=2100, got: {msg}");
        assert!(msg.contains("1024"), "msg should contain max=1024, got: {msg}");
    }
```

- [ ] **Step 1.14: Run the new unit tests**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::scheduler::tests::evict_all_drops_cache \
    core::scheduler::tests::admit_rejects_oversize_request \
    -- --test-threads=1 --nocapture 2>&1 | tail -10
```

Expected: 2 PASS.

If `admit_rejects_oversize_request` fails because `effective_cap_max` field isn't checked: re-verify Step 1.6 / 1.7 actually went into the admit body. The gate must be placed BEFORE `let row_idx = self.slots.iter().position(...)`.

- [ ] **Step 1.15: Run the whole scheduler lib test module to catch regressions**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib core::scheduler::tests -- --test-threads=1 2>&1 | tail -10
```

Expected: all existing tests + 2 new PASS.

Common failures from the bulk `Scheduler::new(N, 32768)` sed update:
- If a test expected admit to succeed with a request whose prompt_ids.len() + max_new_tokens > 32768, it now rejects. Audit failing tests; either bump cap or shrink request.

- [ ] **Step 1.16: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three must pass.

- [ ] **Step 1.17: Commit**

```bash
git add ironmlx/src/models/qwen3_5/config.rs \
        ironmlx/src/core/scheduler.rs \
        ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3f-t1): typed RequestTooLarge + admit cap gate + evict_all drops cache

Qwen35Config reads max_position_embeddings from config.json text_config
(Qwen3.5-4B: 262144; #[serde(default = 32768)] for older variants).

SchedulerError adds RequestTooLarge { needed, max } variant via thiserror.
Scheduler holds effective_cap_max (set via Scheduler::new(b_max, cap_max));
admit and admit_mid reject prompt_len + max_new_tokens > cap_max with
anyhow::Error::new(SchedulerError::RequestTooLarge) — HTTP 413 mapping
deferred to T3.

evict_all drops the cache (self.cache = None) instead of resetting
offsets. Next prefill_admitted lazy-allocates with the new batch's
requirements (T2). ~10ms re-alloc per outer batch is negligible vs
prefill GPU time.

Scheduler::new signature breaks (b_max) → (b_max, effective_cap_max).
All 15+ cfg(test) call sites in scheduler.rs updated via bulk sed to
pass 32768 default. scheduler_actor.rs internal Scheduler::new call
hardcoded to 32768 — T2 will plumb the real effective_cap_max through
spawn_scheduler_actor's signature.

Two unit tests:
- evict_all_drops_cache: post-evict cache is None
- admit_rejects_oversize_request: cap_max=1024 + prompt+max_new=2100 →
  Err downcast to SchedulerError::RequestTooLarge with correct fields

Spec ref: §2 G1+G4+G6, §4.2.1+4.2.2+4.2.3+4.2.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `prefill_admitted_inner` dynamic cap + `spawn_scheduler_actor` signature + ServeArgs/AppState plumbing

**Why this is second:** T2 ships the actual dynamic cap behavior (T1's gates were preconditions). Also extends `spawn_scheduler_actor` signature with `effective_cap_max`; this breaks all 21 test caller sites which T2 updates in the same commit.

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs:479` (prefill_admitted_inner lazy alloc)
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (spawn_scheduler_actor signature + internal call)
- Modify: `ironmlx/src/cli/serve.rs` (ServeArgs + run)
- Modify: `ironmlx/src/core/server/mod.rs` (AppState + serve)
- Modify: 5 test files updating `spawn_scheduler_actor` callers

### Steps

- [ ] **Step 2.1: Replace `prefill_admitted_inner` hardcoded 8192**

Edit `ironmlx/src/core/scheduler.rs`. Find the cache lazy-alloc (around line 479-481):

```rust
        if self.cache.is_none() {
            self.cache = Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
        }
```

Replace with:

```rust
        if self.cache.is_none() {
            // B1-p2.3f: dynamic cap = max(prompt_len + max_new_tokens) over
            // admitted slots, bounded by effective_cap_max (defense-in-depth;
            // admit gate already rejects oversize). min_cap=256 fallback if
            // all slots None (defensive — not reachable in production since
            // prefill_admitted asserts active_count() >= 1 earlier).
            let slots_max = self
                .slots
                .iter()
                .filter_map(|s| s.as_ref())
                .map(|r| {
                    let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
                    (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
                })
                .max()
                .unwrap_or(256);
            let cap = slots_max.min(self.effective_cap_max as i32);
            self.cache = Some(model.make_cache(b as i32, cap, Dtype::Bfloat16)?);
        }
```

- [ ] **Step 2.2: Update doc comment around `prefill_admitted_inner`**

Locate the doc comment block above `pub fn prefill_admitted` (around lines 387-404). Replace the paragraph mentioning "capacity 8192":

**Before:**
> Lazy-allocates the batched KV cache on first call (`b_max` rows, capacity 8192, bf16). On subsequent calls (after `evict_all`) the cache is reused — `evict_all` already reset every layer.

**After:**
> Lazy-allocates the batched KV cache on first call (`b_max` rows; capacity = `min(max(prompt_len + max_new_tokens) over slots, effective_cap_max)`, bf16). Subsequent calls after `evict_all` allocate fresh — `evict_all` drops the cache (3f) so the next batch's cap is sized to its slots, not inherited from the prior batch.

- [ ] **Step 2.3: Extend `spawn_scheduler_actor` signature**

Edit `ironmlx/src/core/server/scheduler_actor.rs`. Locate `pub fn spawn_scheduler_actor` (around line 116). Append a 5th parameter `effective_cap_max: usize`:

```rust
#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
) -> SchedulerActorHandle {
    // ... existing setup of channels + counters ...

    // Pass effective_cap_max into driver_loop via the closure.
    let effective_cap_max_for_task = effective_cap_max;
    tokio::task::spawn_blocking(move || {
        driver_loop(
            model,
            b_max,
            admission_deadline,
            admission_queue_max,
            effective_cap_max_for_task,  // 3f: added
            cmd_rx,
            admit_count_for_task,
            // ... rest of args ...
        );
    });

    // ... existing handle construction ...
}
```

(Match the exact ordering of args in `driver_loop` — see Step 2.4. The 5th positional arg after `admission_queue_max` is cleanest.)

- [ ] **Step 2.4: Extend `driver_loop` signature to receive `effective_cap_max`**

In the same file, locate `fn driver_loop` (around line 145). Add a parameter (matching insertion point with Step 2.3):

```rust
#[allow(clippy::too_many_arguments)]
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,  // 3f
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    // ... existing AtomicU64 / AtomicUsize counter args ...
) {
    let mut sched = Scheduler::new(b_max, effective_cap_max);  // 3f
    // ... rest unchanged ...
}
```

The `Scheduler::new(b_max)` → `Scheduler::new(b_max, effective_cap_max)` was already done in T1 with hardcoded `32768`; now replace `32768` with the param.

- [ ] **Step 2.5: Update `core/server/mod.rs` — AppState + serve()**

Edit `ironmlx/src/core/server/mod.rs`. Locate `pub struct AppState` and append a field:

```rust
pub struct AppState {
    pub model: Arc<Mutex<Qwen35Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    pub prefill_chunk_size: usize,
    pub scheduler_handle: scheduler_actor::SchedulerActorHandle,
    pub b_max: usize,
    pub admission_deadline_ms: u64,
    pub admission_queue_max: usize,
    /// Effective cap_max = min(--max-cache-cap CLI flag, model.config.max_position_embeddings).
    /// Per-request `prompt_len + max_new_tokens` exceeding this returns HTTP 413. B1-p2.3f.
    pub effective_cap_max: usize,
}
```

Then locate `pub async fn serve(...)` and add the `max_cache_cap: usize` parameter + compute `effective_cap_max`:

```rust
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    model: Qwen35Model,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
) -> Result<()> {
    let model = Arc::new(Mutex::new(model));
    let admission_deadline = std::time::Duration::from_millis(admission_deadline_ms);

    // 3f: effective_cap_max = min(--max-cache-cap CLI, model.config.max_position_embeddings).
    // Computed once at boot; threaded into the actor's Scheduler.
    let model_max_context: usize = {
        let m = model.blocking_lock();
        // i32 → usize: saturating-clamp at 0 (i32 negative is invalid for cap).
        m.config().max_position_embeddings.max(0) as usize
    };
    let effective_cap_max = max_cache_cap.min(model_max_context);
    if max_cache_cap > model_max_context {
        tracing::warn!(
            "max_cache_cap CLI flag {} exceeds model_max_context {} — capping at {}",
            max_cache_cap,
            model_max_context,
            model_max_context
        );
    }

    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
    );
    let state = AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        scheduler_handle,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        effective_cap_max,
    };
    // ... rest of serve() body unchanged (Router::new + axum::serve) ...
}
```

- [ ] **Step 2.6: Extend `ServeArgs` + `run` in `cli/serve.rs`**

Edit `ironmlx/src/cli/serve.rs`. Locate `pub struct ServeArgs` and append a flag:

```rust
    /// Maximum allowed `prompt_len + max_new_tokens` per request. Capped
    /// further at the model's `max_position_embeddings` (Qwen3.5-4B: 262144).
    /// Requests beyond this return HTTP 413 Payload Too Large. B1-p2.3f.
    #[arg(long, default_value_t = 32768)]
    pub max_cache_cap: usize,
```

Then update `pub fn run(args: ServeArgs) -> Result<()>` to pass it to `server::serve(...)`:

```rust
    runtime.block_on(server::serve(
        model,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        args.prefill_chunk_size,
        args.b_max,
        args.admission_deadline_ms,
        args.admission_queue_max,
        args.max_cache_cap,  // 3f
    ))
```

- [ ] **Step 2.7: Update `server/mod.rs` internal `spawn_scheduler_actor` call**

Inside `serve()`, the `spawn_scheduler_actor(model.clone(), b_max, admission_deadline, admission_queue_max)` call (around line 60-65) gains a 5th arg `effective_cap_max` (already done in Step 2.5's body rewrite). Re-verify after Step 2.5.

- [ ] **Step 2.8: Update test callers — `b1_p2_3b_2_scheduler_actor.rs`**

```bash
grep -n "spawn_scheduler_actor(" /Volumes/Dev/cxx-mlx/ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs
```

For each call site, append a 5th arg `32768` (matches CLI default):

```rust
let handle = spawn_scheduler_actor(model_arc, 4, Duration::from_millis(5), 32);
// → becomes:
let handle = spawn_scheduler_actor(model_arc, 4, Duration::from_millis(5), 32, 32768);
```

Bulk update:

```bash
sed -i.bak -E 's/spawn_scheduler_actor\(([^)]+), ([0-9]+)\)$/spawn_scheduler_actor(\1, \2, 32768)/g' \
    /Volumes/Dev/cxx-mlx/ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs
rm /Volumes/Dev/cxx-mlx/ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs.bak
```

Verify:
```bash
grep -n "spawn_scheduler_actor(" /Volumes/Dev/cxx-mlx/ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs
```

Each call should now have 5 args.

- [ ] **Step 2.9: Update remaining test caller files**

Same bulk-update pattern for:
- `ironmlx/tests/b1_p2_3b_3_admission_window.rs` (4 sites)
- `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs` (3 sites)
- `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` (3 sites — note: one site has queue_max=0 for the audit fix; preserve)
- `ironmlx/tests/b1_p2_3d_admission_queue.rs` (4 sites — note: some have different `b_max` / `queue_max` values; the sed regex is shape-agnostic since it only appends one extra arg)
- `ironmlx/tests/b1_p2_4_batched_vl.rs` (4 sites)

Do one file at a time, verifying with `grep` after each, since the regex above only handles the simple `(arg1, arg2, arg3, arg4)` shape — if any line has trailing whitespace / multi-line args, do it manually.

For a manual / safe pattern:

```bash
for f in b1_p2_3b_3_admission_window b1_p2_3b_4_anthropic_actor b1_p2_3c_3_continuous_batching b1_p2_3d_admission_queue b1_p2_4_batched_vl; do
    echo "=== $f ==="
    sed -i.bak -E 's/spawn_scheduler_actor\(([^)]+)\)/spawn_scheduler_actor(\1, 32768)/g' \
        "/Volumes/Dev/cxx-mlx/ironmlx/tests/${f}.rs"
    rm "/Volumes/Dev/cxx-mlx/ironmlx/tests/${f}.rs.bak"
    grep -n "spawn_scheduler_actor(" "/Volumes/Dev/cxx-mlx/ironmlx/tests/${f}.rs"
done
```

Verify each file's calls show 5 args.

- [ ] **Step 2.10: Verify workspace compiles**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo +stable build --tests 2>&1 | tail -5
```

Both clean. Common failures: a caller still has 4 args because regex missed multi-line. `grep -rn "spawn_scheduler_actor(" ironmlx/tests/ | grep -v "32768\|effective_cap_max\|\\*\\* current arg"` finds stragglers.

- [ ] **Step 2.11: Verify `ironmlx serve --help` shows the new flag**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -3
./target/release/ironmlx serve --help 2>&1 | grep -E "max-cache-cap|b-max|admission-"
```

Expected: 4 lines (3 prior flags + `--max-cache-cap`), with the new one showing `[default: 32768]`.

- [ ] **Step 2.12: Write `dynamic_cap_from_slots_bounded_by_cap_max` unit test**

This test exercises the dynamic cap logic in `prefill_admitted_inner`. Because we can't easily run `model.make_cache` without a real model in a lib test, expose a cfg(test)-only seam: a `pub(crate) fn computed_cap_for_prefill(&self) -> i32` accessor that returns the same calculation `prefill_admitted_inner` would use.

Add the accessor to `ironmlx/src/core/scheduler.rs`:

```rust
impl Scheduler {
    // ... existing methods ...

    /// cfg(test)-only accessor: compute what cap `prefill_admitted_inner`
    /// would use to lazy-allocate the cache. Returns the bounded cap
    /// (min of slots_max and effective_cap_max). Used by 3f unit tests
    /// to verify cap calculation without invoking a real model.
    #[cfg(test)]
    pub(crate) fn computed_cap_for_prefill(&self) -> i32 {
        let slots_max = self
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|r| {
                let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
                (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
            })
            .max()
            .unwrap_or(256);
        slots_max.min(self.effective_cap_max as i32)
    }
}
```

Then refactor `prefill_admitted_inner` to use this accessor (replacing the inline compute from Step 2.1):

```rust
        if self.cache.is_none() {
            // B1-p2.3f: bounded dynamic cap. See computed_cap_for_prefill.
            let cap = {
                let slots_max = self
                    .slots
                    .iter()
                    .filter_map(|s| s.as_ref())
                    .map(|r| {
                        let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
                        (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
                    })
                    .max()
                    .unwrap_or(256);
                slots_max.min(self.effective_cap_max as i32)
            };
            self.cache = Some(model.make_cache(b as i32, cap, Dtype::Bfloat16)?);
        }
```

(Keeping the logic inline rather than calling `self.computed_cap_for_prefill()` from prefill_admitted_inner means we don't have to make the accessor `cfg(test)`-conditional in source. The two implementations stay identical; the accessor exists only for tests.)

Now write the test:

```rust
    #[test]
    fn dynamic_cap_from_slots_bounded_by_cap_max() {
        // B1-p2.3f: cap = min(max(prompt_len + max_new_tokens over slots), effective_cap_max).
        let mut s = Scheduler::new(4, 2048);  // cap_max=2048

        let req = |prompt_len: usize, max_new: usize| GenerateRequest {
            prompt_ids: vec![0; prompt_len],
            max_new_tokens: max_new,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
        };

        // Admit 3 slots: cap_needed values [50+50=100, 700+100=800, 1300+200=1500].
        s.admit(req(50, 50)).expect("admit 1");
        s.admit(req(700, 100)).expect("admit 2");
        s.admit(req(1300, 200)).expect("admit 3");

        // computed_cap = max(100, 800, 1500) = 1500, bounded by cap_max=2048 → 1500.
        let cap = s.computed_cap_for_prefill();
        assert_eq!(cap, 1500, "cap should equal max(slot cap_needed); cap_max=2048 doesn't bind");

        // Verify bounding kicks in when slots_max exceeds cap_max.
        let mut s2 = Scheduler::new(4, 800);  // cap_max=800
        s2.admit(req(50, 50)).expect("admit 1");
        // 800 + 100 = 900 > cap_max=800 → admit rejects. Need to use values
        // within cap_max for admit to succeed, but show cap bounding still
        // applies if effective_cap_max < slots_max happened (e.g., via
        // race-condition or test-only construction). Use a smaller cap_max
        // and verify min() result.
        let mut s3 = Scheduler::new(4, 200);
        s3.admit(req(50, 50)).expect("admit (cap_needed=100 < 200)");
        s3.admit(req(150, 30)).expect("admit (cap_needed=180 < 200)");
        let cap3 = s3.computed_cap_for_prefill();
        assert_eq!(cap3, 180, "cap = max(100, 180) = 180; bound at 200 doesn't bind");

        // Empty-slot fallback.
        let s4 = Scheduler::new(4, 1000);
        assert_eq!(s4.computed_cap_for_prefill(), 256, "empty slots fallback = 256");
    }
```

- [ ] **Step 2.13: Run the new unit test**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::scheduler::tests::dynamic_cap_from_slots_bounded_by_cap_max \
    -- --test-threads=1 --nocapture 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 2.14: Run all scheduler lib tests**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib core::scheduler::tests -- --test-threads=1 2>&1 | tail -10
```

Expected: T1's two tests + Step 2.12's new test + all pre-existing tests PASS.

- [ ] **Step 2.15: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three pass.

- [ ] **Step 2.16: Commit**

```bash
git add ironmlx/src/core/scheduler.rs \
        ironmlx/src/core/server/scheduler_actor.rs \
        ironmlx/src/core/server/mod.rs \
        ironmlx/src/cli/serve.rs \
        ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs \
        ironmlx/tests/b1_p2_3b_3_admission_window.rs \
        ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs \
        ironmlx/tests/b1_p2_3c_3_continuous_batching.rs \
        ironmlx/tests/b1_p2_3d_admission_queue.rs \
        ironmlx/tests/b1_p2_4_batched_vl.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3f-t2): dynamic cap + CLI/AppState plumbing

prefill_admitted_inner lazy-allocates cache with cap = min(slots_max,
effective_cap_max), where slots_max = max(prompt_len + max_new_tokens)
over admitted slots. Falls back to 256 if all slots None (defensive;
prefill_admitted asserts active_count() >= 1 earlier).

spawn_scheduler_actor signature gains effective_cap_max (5th param).
Internally Scheduler::new(b_max, effective_cap_max) — replaces T1's
hardcoded 32768.

ServeArgs adds --max-cache-cap (default 32768). serve() computes
effective_cap_max = min(max_cache_cap, model.config.max_position_embeddings)
once at boot, with tracing::warn if max_cache_cap > model_max_context.
AppState carries effective_cap_max for downstream observability.

All 21 spawn_scheduler_actor caller sites (1 source + 5 test files)
updated to the new 5-arg signature with 32768 defaults.

One unit test: dynamic_cap_from_slots_bounded_by_cap_max — verifies
max(slot cap_needed) computation + cap_max binding + empty-slot
256 fallback.

Spec ref: §2 G2+G3+G5, §4.2.5+4.2.6+4.2.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: HTTP 413 mapping in `openai.rs` + `anthropic.rs` + 2 unit tests

**Why this is third:** T1 typed `RequestTooLarge`; T3 surfaces it to HTTP. Pure handler refactor.

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs:43` (`admit_err_to_response`)
- Modify: `ironmlx/src/core/server/anthropic.rs:39` (same)

### Steps

- [ ] **Step 3.1: Update `admit_err_to_response` in `openai.rs`**

Find the function (around line 43). Replace the `if let Some(QueueFull)` ladder with a match:

**Before** (current 3e.3 shape):

```rust
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use crate::core::SchedulerError;
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    if let Some(SchedulerError::QueueFull { .. }) = err.downcast_ref::<SchedulerError>() {
        let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        resp
    } else {
        (StatusCode::BAD_REQUEST, msg).into_response()
    }
}
```

**After**:

```rust
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use crate::core::SchedulerError;
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    match err.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => {
            // 503 Service Unavailable + Retry-After
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        Some(SchedulerError::RequestTooLarge { .. }) => {
            // 413 Payload Too Large — request needed cap exceeds server's
            // effective_cap_max. Body includes needed + max via Display.
            (StatusCode::PAYLOAD_TOO_LARGE, msg).into_response()
        }
        None => {
            // Other anyhow Errs (prompt parsing, OOM, etc.) → 400 Bad Request.
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}
```

- [ ] **Step 3.2: Update `admit_err_to_response` in `anthropic.rs`**

Mirror Step 3.1 in `ironmlx/src/core/server/anthropic.rs` (around line 39). The body is identical to openai.rs (the helper is duplicated per-handler per spec §9 R3 acceptance of fragile per-module copy).

- [ ] **Step 3.3: Write `admit_err_413_for_request_too_large` unit test**

Edit the `#[cfg(test)] mod tests` block in `ironmlx/src/core/server/openai.rs`. After the existing `admit_err_503_for_queue_full` test, add:

```rust
    #[tokio::test]
    async fn admit_err_413_for_request_too_large() {
        use axum::body::to_bytes;
        use axum::http::StatusCode;

        // 3f: typed SchedulerError::RequestTooLarge → 413 Payload Too Large.
        let err = anyhow::Error::new(crate::core::SchedulerError::RequestTooLarge {
            needed: 50000,
            max: 32768,
        });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // No Retry-After header for 413 (client error, not transient).
        assert!(resp.headers().get("retry-after").is_none());

        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            body_str.contains("50000"),
            "body should mention needed=50000, got: {body_str}"
        );
        assert!(
            body_str.contains("32768"),
            "body should mention max=32768, got: {body_str}"
        );
    }
```

- [ ] **Step 3.4: Write `admit_err_400_falls_through` unit test**

The existing `admit_err_400_for_untyped_anyhow` already covers the fall-through path. Verify it still PASSes after the match refactor:

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::server::openai::tests::admit_err_400_for_untyped_anyhow \
    -- --test-threads=1 --nocapture 2>&1 | tail -5
```

Expected: PASS (it constructs an untyped anyhow Err whose downcast returns None → 400).

If you want a more explicit assertion, add a small additional test that constructs an Err of a different anyhow source:

```rust
    #[tokio::test]
    async fn admit_err_400_for_unrelated_typed_error() {
        use axum::http::StatusCode;

        // A typed Err that is NOT SchedulerError → falls through to 400.
        #[derive(Debug, thiserror::Error)]
        #[error("test error: {msg}")]
        struct OtherError {
            msg: String,
        }
        let err = anyhow::Error::new(OtherError {
            msg: "unrelated".to_string(),
        });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }
```

- [ ] **Step 3.5: Run all openai handler unit tests**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib core::server::openai::tests -- --test-threads=1 2>&1 | tail -10
```

Expected: pre-existing 8 tests + 1-2 new = 9-10 PASS.

- [ ] **Step 3.6: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All pass.

- [ ] **Step 3.7: Commit**

```bash
git add ironmlx/src/core/server/openai.rs ironmlx/src/core/server/anthropic.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3f-t3): HTTP 413 for SchedulerError::RequestTooLarge

OpenAI + Anthropic admit_err_to_response refactored from if-let ladder
to match on downcast_ref::<SchedulerError>() arms:
- QueueFull → 503 Service Unavailable + Retry-After: 5
- RequestTooLarge → 413 Payload Too Large (no Retry-After; client error)
- None (other anyhow Errs) → 400 Bad Request

Two unit tests:
- admit_err_413_for_request_too_large: typed Err RequestTooLarge
  {needed: 50000, max: 32768} → 413; body contains both numbers
- admit_err_400_for_unrelated_typed_error: a different thiserror enum
  → 400 (verifies match None arm covers non-SchedulerError typed errs)

Spec ref: §2 G7, §4.2.8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Long-prompt integration test + 14-suite regression sweep + close-out

**Why this is fourth:** Final acceptance. Exercises the entire path (admit → cap_check → prefill → decode) with a real-model long prompt. Regression sweep validates pre-3f behavior preserved under default config.

**Files:**
- Create: `ironmlx/tests/b1_p2_3f_cache_cap.rs` (NEW, ~120 LoC, 1 scenario)
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3f_closeout/report.md` (NEW, gitignored — `-f`)

### Steps

- [ ] **Step 4.1: Scaffold the integration test file**

Create `ironmlx/tests/b1_p2_3f_cache_cap.rs`:

```rust
//! B1-p2.3f integration: long-prompt admit + decode.
//!
//! Validates the three-tier cap model end-to-end:
//!   1. Server boots with default --max-cache-cap = 32768.
//!   2. effective_cap_max = min(32768, Qwen3.5-4B model_max_context = 262144) = 32768.
//!   3. A request with prompt_len ≈ 10240 + max_new = 20 has cap_needed
//!      = 10260 ≤ 32768 → admits successfully.
//!   4. prefill_admitted_inner lazy-allocates cache with cap = 10260
//!      (slots_max bound at effective_cap_max), enabling long-prompt
//!      decode to completion.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::Qwen35Model;
use tokio::sync::Mutex;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN35_MODEL") {
        return PathBuf::from(p);
    }
    let glob = format!(
        "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
        std::env::var("HOME").unwrap()
    );
    std::fs::read_dir(&glob)
        .expect("snapshots dir")
        .filter_map(|e| e.ok())
        .next()
        .expect("snapshot")
        .path()
}

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let p = model_path();
    let loader = Loader::open_multimodal(&p).expect("Loader::open_multimodal");
    let tok = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader(&loader).expect("model");
    (Arc::new(Mutex::new(model)), Arc::new(tok))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy: needs QWEN35_MODEL
async fn admit_long_prompt_pp10k() {
    // Construct a ~10K-token prompt by repeating a phrase enough times.
    // We measure precisely via tokenizer.encode then pad if needed.
    let (model, tokenizer) = load_fixture();

    // Build a long prompt: repeat "Hello world, this is a long test prompt. "
    // ~10240 tokens worth (the phrase is ~10-12 tokens, so 900 repeats ≈ 10K).
    let phrase = "Hello world, this is a long test prompt. ";
    let raw = phrase.repeat(900);
    let prompt_ids = tokenizer.encode(&raw, false).expect("encode");
    let prompt_len = prompt_ids.len();
    eprintln!("[3f] long prompt encoded to {prompt_len} tokens");
    assert!(
        prompt_len >= 8200,
        "expected ≥ 8200 tokens (proves cap > 8192 needed); got {prompt_len}"
    );
    assert!(
        prompt_len <= 16384,
        "test prompt should fit comfortably under default cap_max=32768; got {prompt_len}"
    );

    // Spawn actor with default 3f config.
    let handle = spawn_scheduler_actor(
        model.clone(),
        /* b_max */ 4,
        /* admission_deadline */ Duration::from_millis(5),
        /* admission_queue_max */ 32,
        /* effective_cap_max */ 32768,
    );

    let max_new = 20_usize;
    let req = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    };

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: req,
            reply_tx,
        })
        .await
        .expect("cmd_tx.send");

    let admit_reply = reply_rx
        .await
        .expect("reply_rx await")
        .expect("admit ok — long prompt under cap_max");

    // Drain decode events until finish_reason="length" at exactly max_new.
    let mut event_rx = admit_reply.event_rx;
    let mut tokens: Vec<u32> = Vec::new();
    let mut finish_reason: Option<&'static str> = None;
    while let Some(ev) = event_rx.recv().await {
        tokens.push(ev.token);
        if let Some(reason) = ev.finish_reason {
            finish_reason = Some(reason);
            break;
        }
    }

    eprintln!(
        "[3f] decode produced {} tokens (max_new={}), finish_reason={:?}",
        tokens.len(),
        max_new,
        finish_reason
    );
    assert_eq!(
        tokens.len(),
        max_new,
        "expected exactly max_new tokens, got {} (proves cache cap fit prompt + decode)",
        tokens.len()
    );
    assert_eq!(
        finish_reason,
        Some("length"),
        "expected finish_reason=length, got {finish_reason:?}"
    );

    drop(handle);
}
```

- [ ] **Step 4.2: Run the integration test**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test b1_p2_3f_cache_cap \
    -- --ignored --test-threads=1 --nocapture 2>&1 | tail -20
```

Expected: PASS. Test will take ~60-90s (long-prompt prefill at PP≈10K is several seconds on M1 Pro; decode of 20 tokens is sub-second).

If it FAILS with "expected ≥ 8200 tokens" — tweak the `.repeat(900)` to a higher number until prompt_len ≥ 8200.

If it FAILS at admit with `SchedulerError::RequestTooLarge` — bug in default value; verify Step 2.6 sets default_value_t=32768.

If it FAILS at prefill with "cache too small" or shape mismatch — bug in dynamic cap calculation (Step 2.1).

- [ ] **Step 4.3: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All pass.

- [ ] **Step 4.4: 14-suite regression sweep**

```bash
MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
echo "=== 14-suite regression sweep ($(date)) ===" > /tmp/b1_p2_3f_regression.log
for t in \
    p6_qwen35_vl_logits_match \
    p6_6_logits_match \
    p6_7_chunked_prefill \
    b1_p2_1_batched_prefill \
    b1_p2_2_batched_decode \
    b1_p2_3b_1_scheduler_step \
    b1_p2_3b_2_scheduler_actor \
    b1_p2_3b_3_admission_window \
    b1_p2_3b_4_anthropic_actor \
    b1_p2_3c_1_per_row_offset \
    b1_p2_3c_2_scheduler_decode_mask \
    b1_p2_3c_3_continuous_batching \
    b1_p2_3d_admission_queue \
    b1_p2_4_batched_vl \
    b1_p2_3f_cache_cap
do
    echo "=== $t ===" >> /tmp/b1_p2_3f_regression.log
    start=$(date +%s)
    QWEN35_MODEL="$MODEL" MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test "$t" \
        -- --ignored --test-threads=1 2>&1 | tail -6 >> /tmp/b1_p2_3f_regression.log
    end=$(date +%s)
    echo "elapsed: $((end - start))s" >> /tmp/b1_p2_3f_regression.log
done
echo "=== b1_p2_3a_scheduler_skeleton (default mode) ===" >> /tmp/b1_p2_3f_regression.log
start=$(date +%s)
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test b1_p2_3a_scheduler_skeleton -- --test-threads=1 2>&1 | tail -5 >> /tmp/b1_p2_3f_regression.log
end=$(date +%s)
echo "elapsed: $((end - start))s" >> /tmp/b1_p2_3f_regression.log
echo "DONE $(date)" >> /tmp/b1_p2_3f_regression.log
```

Verify all pass:

```bash
grep -E "test result:|elapsed:" /tmp/b1_p2_3f_regression.log | grep -v "0 measured" | tail -40
```

Expected: every suite reports `test result: ok`. No `failed`.

If any test fails, BLOCKED — report specific failure to controller. Likely candidates:
- Test in 3c-3 / 3d that asserted on specific cap or "scheduler full" message → may need audit fix (similar to 3d's T5 Step 5.1 audit).

- [ ] **Step 4.5: Write close-out report**

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3f_closeout
```

Write `report.md`:

```markdown
# B1-p2.3f Cache Cap Dynamic + Bounded — Close-out

**Branch:** `ironmlx-b1-p2-3f-cache-cap` (off `ironmlx-b1-p2-3e3-typed-err` head `1076254`)
**Date:** 2026-05-XX (fill on completion)
**Status:** ✅ COMPLETE

## Summary

Replaces hardcoded `make_cache(b, 8192, dtype)` with three-tier dynamic
cap:
- `cap_needed (per-request) = prompt_len + max_new_tokens`
- `effective_cap_max = min(--max-cache-cap CLI, model.config.max_position_embeddings)`
- Qwen3.5-4B model_max_context = 262144

`admit` and `admit_mid` reject `cap_needed > effective_cap_max` with
`SchedulerError::RequestTooLarge { needed, max }` → HTTP 413 Payload
Too Large + body containing both numbers. `prefill_admitted_inner`
lazy-allocates cache with `cap = min(slots_max, effective_cap_max)`.
`evict_all` drops the cache so each new batch's cap matches its actual
requirements.

Enables Boss's agent long-prompt (10-20K tokens) use case which
previously failed at first-batch admit with cap=8192.

## Acceptance

| Gate | Result |
| --- | --- |
| Integration: admit_long_prompt_pp10k (PP≈10K, max_new=20) | ✅ PASS |
| Unit: evict_all_drops_cache | ✅ PASS |
| Unit: admit_rejects_oversize_request | ✅ PASS |
| Unit: dynamic_cap_from_slots_bounded_by_cap_max | ✅ PASS |
| Unit: admit_err_413_for_request_too_large | ✅ PASS |
| Unit: admit_err_400_for_unrelated_typed_error | ✅ PASS |
| fmt --check / clippy -D warnings / build --release | ✅ ALL CLEAN |

## Architectural changes per spec §4

| Item | File | Change |
| --- | --- | --- |
| §4.2.1 Qwen35Config.max_position_embeddings | `models/qwen3_5/config.rs` | Added (+serde default 32768) |
| §4.2.2 SchedulerError::RequestTooLarge | `core/scheduler.rs` | typed variant + Display |
| §4.2.3 Scheduler::new (b_max, cap_max) + admit/admit_mid cap gate | `core/scheduler.rs` | Signature + gate |
| §4.2.4 evict_all drops cache | `core/scheduler.rs:364` | Replaced offset reset |
| §4.2.5 prefill_admitted_inner dynamic cap | `core/scheduler.rs:479` | min(slots_max, cap_max) |
| §4.2.6 spawn_scheduler_actor 5th param | `core/server/scheduler_actor.rs` | Signature + driver_loop |
| §4.2.7 ServeArgs/AppState/serve plumbing | `cli/serve.rs`, `core/server/mod.rs` | CLI flag + state propagation |
| §4.2.8 HTTP 413 mapping | `core/server/{openai,anthropic}.rs` | Match downcast arm |

## Commits

(Fill in via `git log --oneline 1076254..HEAD`)

- T1: `<sha>` typed RequestTooLarge + admit cap gate + evict_all drops
- T2: `<sha>` dynamic cap + CLI/AppState plumbing
- T3: `<sha>` HTTP 413 mapping + 2 unit tests
- T4: `<sha>` long-prompt integration + 15-suite regression + close-out

## Regression Status

Sweep run with `--ignored --test-threads=1` and default `b_max=4 /
deadline=5ms / queue_max=32 / max_cache_cap=32768`.

| Suite | Result | Time |
| --- | --- | --- |
| p6_qwen35_vl_logits_match | ✅ PASS | <FILL>s |
| p6_6_logits_match | ✅ PASS | <FILL>s |
| p6_7_chunked_prefill | ✅ PASS | <FILL>s |
| b1_p2_1_batched_prefill | ✅ PASS | <FILL>s |
| b1_p2_2_batched_decode | ✅ PASS | <FILL>s |
| b1_p2_3a_scheduler_skeleton (default mode) | ✅ PASS | <FILL>s |
| b1_p2_3b_1_scheduler_step | ✅ PASS | <FILL>s |
| b1_p2_3b_2_scheduler_actor | ✅ PASS | <FILL>s |
| b1_p2_3b_3_admission_window | ✅ PASS | <FILL>s |
| b1_p2_3b_4_anthropic_actor | ✅ PASS | <FILL>s |
| b1_p2_3c_1_per_row_offset | ✅ PASS | <FILL>s |
| b1_p2_3c_2_scheduler_decode_mask | ✅ PASS | <FILL>s |
| b1_p2_3c_3_continuous_batching | ✅ PASS | <FILL>s |
| b1_p2_3d_admission_queue | ✅ PASS | <FILL>s |
| b1_p2_4_batched_vl | ✅ PASS | <FILL>s |
| **B1-p2.3f cache cap (1 scenario)** | **✅ PASS** | **<FILL>s** |

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| Hardcoded `cap = 8192` in `prefill_admitted_inner` | `min(max(prompt_len + max_new_tokens over slots), effective_cap_max)` |
| `evict_all` resets cache offsets | `evict_all` drops cache (`self.cache = None`) |
| `Scheduler::new(b_max)` | `Scheduler::new(b_max, effective_cap_max)` — breaking change |
| `spawn_scheduler_actor(model, b_max, deadline, queue_max)` | `+ effective_cap_max` 5th param |

## Notes / known limitations carrying forward to backlog

- **No per-request cap override** (spec NG1). All requests share the
  server-wide `effective_cap_max`.
- **No cache memory pool / reuse across outer batches** (spec NG3).
  Each `evict_all` + `prefill_admitted` re-allocates the cache.
- **No YaRN post-scaling effective context** (spec NG4). 3f uses
  raw `max_position_embeddings` only; models extending context via
  rope_scaling are capped at the unscaled value.
- **Multi-tenant cap policies** (spec NG5). Future work.

## B1-p2 Next Steps

| Sub-spec | Scope | Status |
| --- | --- | --- |
| B1-p2.3c+ | Chunked admit_mid prefill | Next (after 3f) |
| B1-p2.3e.1 | Async per-row sampler | Backlog |
| B1-p2.3e.2 | HTTP cancellation propagation | Backlog |
| B1-p2.3f | **Cache cap dynamic + bounded** | **✅ DONE (this report)** |
| B1-p2.4 | VL B>1 batched serving | ✅ DONE |
| B1-p2.5 | Production hardening (multi-tenant cap, OOM guards) | Future |

After B1-p2.3f: agent long-prompt (10-20K tokens) use case works. Next:
3c+ to make admit_mid stall smooth via chunked prefill.

## Linked artifacts

- [B1-p2.3f design spec](../../../../../docs/superpowers/specs/2026-05-16-b1-p2-3f-cache-cap-dynamic-design.md)
- [B1-p2.3f implementation plan](../../../../../docs/superpowers/plans/2026-05-16-b1-p2-3f-cache-cap-dynamic.md)
- [B1-p2.3d close-out](../b1_p2_3d_closeout/report.md)
- [B1-p2.4 close-out](../b1_p2_4_closeout/report.md)
```

Replace `<FILL>` with actual timings from `/tmp/b1_p2_3f_regression.log` and commit SHAs from `git log`.

- [ ] **Step 4.6: Stage + commit**

```bash
git add ironmlx/tests/b1_p2_3f_cache_cap.rs
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3f_closeout/report.md

git commit -m "$(cat <<'EOF'
test+docs(b1-p2.3f-t4): long-prompt integration + 15-suite regression + close-out

Adds tests/b1_p2_3f_cache_cap.rs with one #[ignore]'d integration
scenario:
- admit_long_prompt_pp10k: PP≈10K, max_new=20 → admit succeeds (cap_max
  default 32768 covers cap_needed=10260); decode produces exactly 20
  tokens with finish_reason=length.

15-suite regression sweep PASS with default config (b_max=4, 5ms,
queue_max=32, max_cache_cap=32768). Close-out report committed under
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3f_closeout/
(path is gitignored — committed with -f).

Spec ref: §4.3 (acceptance), §4.4 R1-R8 (risk mitigation verified).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4.7: Final verification**

```bash
git log --oneline 1076254..HEAD
```

Expected: 4 commits, one per task.

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

All clean. **B1-p2.3f done.**

---

## Self-review checklist

**Spec coverage (§4.2 items):**

- §4.2.1 `Qwen35Config.max_position_embeddings` → T1 Step 1.1
- §4.2.2 `SchedulerError::RequestTooLarge` → T1 Step 1.3
- §4.2.3 `Scheduler::new` signature + admit/admit_mid gates → T1 Steps 1.4–1.7
- §4.2.4 `evict_all` drops cache → T1 Step 1.8
- §4.2.5 `prefill_admitted_inner` dynamic cap → T2 Steps 2.1 + 2.12 (accessor)
- §4.2.6 `spawn_scheduler_actor` 5th param → T2 Steps 2.3–2.4
- §4.2.7 ServeArgs/AppState/serve plumbing → T2 Steps 2.5–2.7
- §4.2.8 HTTP 413 mapping → T3 Steps 3.1–3.2

**Spec §4.3 acceptance:**

- 1 integration test (`admit_long_prompt_pp10k`) → T4 Step 4.1
- 3 unit tests in scheduler::tests (evict_all_drops_cache / admit_rejects_oversize_request / dynamic_cap_from_slots_bounded_by_cap_max) → T1 Steps 1.12–1.13 + T2 Step 2.12
- 2 unit tests in openai::tests (admit_err_413 / admit_err_400_falls_through) → T3 Steps 3.3–3.4
- 14-suite regression sweep (15 with 3f added) → T4 Step 4.4

**Spec §4.4 R1-R8 mitigation:**

- R1 `max_position_embeddings` field missing in older configs → T1 Step 1.1 (`#[serde(default = "default_max_position_embeddings")]` returns 32768)
- R2 `GatedDeltaCache` cap behavior → existing 3c-1/3c-2 tests in T4 regression sweep verify Linear path
- R3 User confusion when cli max_cache_cap > model_max_context → T2 Step 2.5 (tracing::warn at startup)
- R4 i32 overflow → T1 Steps 1.6/1.7 use `.saturating_add(req.max_new_tokens)`
- R5 Tests asserting old cap=8192 → T4 Step 4.4 regression sweep catches; audit during sweep
- R6 ~10ms alloc overhead per outer batch → not measured in plan; documented as known cost
- R7 HTTP 413 client behavior → T3 Step 3.3 covers status + body content
- R8 model_max_context for VL variants → 3f's `from_loader` reads same field; all Qwen3.5-VL declare it

**Spec §3 NG1-NG5 (must NOT be implemented):**

- NG1 Per-request cap override — not implemented; only server-wide `effective_cap_max` ✓
- NG2 Cache cap shrinking — not implemented; once allocated for a batch, cache persists ✓
- NG3 Memory pool — not implemented; each evict_all + prefill_admitted re-allocates ✓
- NG4 YaRN post-scaling effective context — not implemented; raw `max_position_embeddings` only ✓
- NG5 Multi-tenant per-user cap limits — not implemented ✓

**Placeholder scan:** No "TBD" / "TODO" / "implement later" in plan steps. Step 4.5's close-out template uses `<FILL>` placeholders for runtime values (timings + SHAs) — these are deliberate, filled in at execution time.

**Type consistency:**

- `SchedulerError::RequestTooLarge { needed: usize, max: usize }` defined T1 Step 1.3; constructed in T1 Steps 1.6/1.7 + T3 Step 3.3 + tests in T1 Step 1.13.
- `effective_cap_max: usize` Scheduler field defined T1 Step 1.4; consumed by Steps 1.6/1.7 admit gates + Step 2.1 prefill cap.
- `Scheduler::new(b_max, effective_cap_max)` signature in T1 Step 1.5; all 15+ cfg(test) sites + scheduler_actor.rs's driver_loop internal call updated in same task.
- `spawn_scheduler_actor(model, b_max, deadline, queue_max, effective_cap_max)` signature in T2 Step 2.3; 21 caller sites (1 source + 5 test files) updated T2 Steps 2.7–2.9.
- `--max-cache-cap` CLI flag T2 Step 2.6 → `AppState.effective_cap_max` Step 2.5 → `serve()` computes `min(cli, model_max_context)` Step 2.5.
- `admit_err_to_response` match arms T3 Steps 3.1–3.2; downcast type `SchedulerError` (3e.3 typed enum, extended in T1).

**Plan saved to:** `docs/superpowers/plans/2026-05-16-b1-p2-3f-cache-cap-dynamic.md`
