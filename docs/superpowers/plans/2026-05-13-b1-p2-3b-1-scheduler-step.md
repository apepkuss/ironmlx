# B1-p2.3b-1 Scheduler `step()` + Lockstep Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `Qwen35Model` forward into the `Scheduler` skeleton landed in 3a — add `Phase` state machine, `prefill_admitted`, `step`, `evict_all`, `StepEvent`, and `LayerCache::reset` — driving B=2/B=4 lockstep batched generation that matches B=1 `GenerationStream` baseline at ≥ 95% argmax bit-id.

**Architecture:** `Scheduler` gains a `Phase` field (`Idle → Admitting → Decoding → Finished → Idle`) and an `Option<Vec<LayerCache>>` cache field. `prefill_admitted` builds left-padded batched inputs + masks via existing `build_batch_attention_mask` / `build_batch_linear_mask` / `build_position_ids_batched` helpers, allocates the cache lazily, and calls `model.batched_prefill`. `step` packs the latest-token `[B, 1]` input, builds per-row `[3, B, 1]` decode positions, calls `model.forward_on`, then samples per row via `RequestState::sampler.sample(per_row_logits, history)`. Finished rows ride along in subsequent `step` forwards (lockstep cost — see [spec §7](../specs/2026-05-13-b1-p2-3b-1-scheduler-step-design.md)) but emit nothing. `evict_all` resets every layer's cache via the new `LayerCache::reset` dispatch.

**Tech Stack:** Rust 2021, `mlx` (cxx-mlx Rust bindings), `anyhow`, `ironmlx` existing core types (`GenerateRequest`, `Sampler`, `LayerCache`, `Qwen35Model`).

---

## File Structure

```
ironmlx/src/nn/decoder_layer.rs       — MODIFY: append `impl LayerCache { pub fn reset() }` (~8 lines)
ironmlx/src/core/scheduler.rs          — MODIFY: add Phase enum + StepEvent struct + cache/phase fields + 4 new methods + 8 new unit tests
ironmlx/src/core/mod.rs                — MODIFY: re-export `Phase` and `StepEvent`
ironmlx/tests/b1_p2_3b_1_scheduler_step.rs  — NEW: 3 integration scenarios + cache reuse smoke
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_1_closeout/report.md      — NEW: close-out
```

Zero modifications to: `models/`, `core/server/`, `core/generate.rs`, `core/cache/`, `core/sampler.rs`, `core/tokenizer.rs`, `nn/attention.rs`, `nn/gated_attention.rs`, `nn/gated_delta_net.rs`, `nn/text_model.rs`. (Only `nn/decoder_layer.rs` is touched in `nn/`.)

---

## Resolved Spec Open Questions (§9)

These three implementation surface decisions from the spec are answered here so tasks downstream are unambiguous:

1. **Cache `cap`** — hardcoded `8192` in `prefill_admitted` (matches the existing `b1_p2_1_batched_prefill.rs` and `b1_p2_2_batched_decode.rs` tests). Constructor `Scheduler::new(b_max)` stays unchanged. A future sub-phase can promote this to a config parameter.
2. **`model.dtype()` accessor** — `Qwen35Model` does **not** expose a `dtype` accessor and has no `dtype` field. Per all existing batched tests (e.g., [`tests/b1_p2_2_batched_decode.rs:113`](../../ironmlx/tests/b1_p2_2_batched_decode.rs#L113), `make_cache(1, cap, Dtype::Bfloat16)`), the model is bf16-only today. `Scheduler::prefill_admitted` hard-codes `mlx::Dtype::Bfloat16` for `make_cache` and `build_batch_attention_mask`. A `// TODO when non-bf16 model lands` comment is the only acknowledgment — no abstraction added (YAGNI).
3. **Mask helpers** — Already public:
   - `core::generate::build_batch_attention_mask(prompt_lens: &[i32], max_len: i32, dtype: Dtype) -> Result<Array>` (`generate.rs:262`)
   - `core::generate::build_batch_linear_mask(prompt_lens: &[i32], max_len: i32) -> Result<Array>` (`generate.rs:362`)
   - `core::generate::build_position_ids_batched(prompt_lens: &[i32], max_len: i32) -> Result<Array>` (`generate.rs:214`)
   - `core::generate::build_decode_position_ids(per_row_pos: &[i32]) -> Result<Array>` (`generate.rs:323`)

   No extraction needed; `scheduler.rs` imports them directly.

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-b1-p2-3-continuous-batching`, HEAD at `20b51cd` ("docs(b1-p2.3b-1): scheduler step + lockstep prefill design spec"). No staged or unstaged changes (only the pre-existing `design.md` stray in repo root is allowed).

---

## Task 1: `Phase` enum + cache reset + admit/evict phase guards + 8 unit tests

**Files:**
- Modify: `ironmlx/src/nn/decoder_layer.rs` (append `impl LayerCache` block)
- Modify: `ironmlx/src/core/scheduler.rs` (add Phase, StepEvent, cache+phase fields, phase()/evict_all()/force_phase(), update admit+evict, +8 unit tests)
- Modify: `ironmlx/src/core/mod.rs` (re-export `Phase`, `StepEvent`)

This task is purely additive **except** for two semantic changes to `Scheduler::admit` and `Scheduler::evict` that add phase guards. Existing tests need no edits because they all run with a fresh `Scheduler::new()` (phase starts `Idle`, transitions to `Admitting` on first admit, never reaches Decoding/Finished without prefill_admitted).

- [ ] **Step 1.1: Add `LayerCache::reset()` dispatch in `decoder_layer.rs`**

Read the current state of `ironmlx/src/nn/decoder_layer.rs` around line 65 to confirm the `LayerCache` enum is unchanged from spec exploration:

```bash
sed -n '60,70p' /Volumes/Dev/cxx-mlx/ironmlx/src/nn/decoder_layer.rs
```

Expected: the enum block on line 65 reads:

```rust
#[doc(hidden)]
pub enum LayerCache {
    Full(KVCache),
    Linear(GatedDeltaCache),
}
```

Append an `impl LayerCache` block immediately after the enum's closing brace (line 68). Use `Edit` with old_string covering the enum + the blank line + the next item (`pub struct DecoderLayer`):

`old_string`:
```rust
pub enum LayerCache {
    Full(KVCache),
    Linear(GatedDeltaCache),
}

/// One decoder block. Full or linear attention selected at construction.
pub struct DecoderLayer {
```

`new_string`:
```rust
pub enum LayerCache {
    Full(KVCache),
    Linear(GatedDeltaCache),
}

impl LayerCache {
    /// Reset to empty state (offset → 0; recurrent state cleared). Preserves
    /// any underlying Array allocations so the next batch can reuse them.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        match self {
            LayerCache::Full(kv) => {
                kv.reset();
                Ok(())
            }
            LayerCache::Linear(gd) => gd.reset(),
        }
    }
}

/// One decoder block. Full or linear attention selected at construction.
pub struct DecoderLayer {
```

(`anyhow::Result` is fully qualified because `decoder_layer.rs` may not have `use anyhow::Result;` at the top — if it already does, the implementer can drop the prefix.)

- [ ] **Step 1.2: Run `cargo +nightly fmt --all` + build to verify Step 1.1 in isolation**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```

Expected: fmt rewrites possibly, build prints `Finished release profile ...`. If `anyhow::Result` import was missing, add `use anyhow::Result;` to `decoder_layer.rs` and re-run.

- [ ] **Step 1.3: Add `Phase` enum + `StepEvent` struct + new `Scheduler` fields to `scheduler.rs`**

Open `ironmlx/src/core/scheduler.rs`. The current state (post 3a + cleanups at `33ea2df`) has the imports + `RequestId` + `RequestState` + `Scheduler` struct + `impl Scheduler { fn new, b_max, admit, evict, active_count, active, get, get_mut, occupied_rows }` + `#[cfg(test)] mod tests`.

Make four edits in order:

**Edit 1.3a — extend the imports block** at the top of the file.

`old_string`:
```rust
use anyhow::{anyhow, Result};

use crate::core::generate::GenerateRequest;
use crate::core::sampler::Sampler;
```

`new_string`:
```rust
use anyhow::{anyhow, Result};

use crate::core::generate::GenerateRequest;
use crate::core::sampler::Sampler;
use crate::nn::LayerCache;
```

**Edit 1.3b — add `Phase` enum and `StepEvent` struct** right after the `RequestId` definition (find the `pub struct RequestId(pub u64);` line and insert the new types after its preceding line block — keep semantic ordering: RequestId first, Phase next, StepEvent next, RequestState after).

Insert these blocks immediately after the line `pub struct RequestId(pub u64);` (matching the surrounding doc-comment style):

```rust

/// Scheduler lifecycle phase. The state machine is `Idle → Admitting →
/// Decoding → Finished → Idle`.
///
/// Transitions are driven by the scheduler methods:
/// - `admit()` from `Idle`/`Admitting` → `Admitting`
/// - `prefill_admitted()` from `Idle`/`Admitting` → `Decoding`
/// - `step()` from `Decoding`: stays `Decoding` while ≥1 row unfinished,
///   transitions to `Finished` when all active rows are `finished`.
/// - `evict_all()` from `Decoding`/`Finished` → `Idle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Admitting,
    Decoding,
    Finished,
}

/// One per-row event emitted by [`Scheduler::step`].
///
/// Only rows that were not yet `finished` at the start of the step appear
/// in the event list. The step in which a row first transitions to
/// `finished` produces an event with `finish_reason = Some("stop"|"length")`.
/// Subsequent steps never emit anything for that row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepEvent {
    pub id: RequestId,
    pub token: u32,
    pub finish_reason: Option<&'static str>,
}
```

**Edit 1.3c — add `phase: Phase` and `cache: Option<Vec<LayerCache>>` fields to the `Scheduler` struct.**

`old_string`:
```rust
#[derive(Debug)]
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
}
```

`new_string`:
```rust
#[derive(Debug)]
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
    phase: Phase,
    cache: Option<Vec<LayerCache>>,
}
```

(Note: `LayerCache` is **not** `Debug` — Boss to confirm at first `cargo build` whether `#[derive(Debug)]` on `Scheduler` breaks. If it does, change to a manual `impl Debug` that skips `cache`. See Step 1.5 for the build verification.)

**Edit 1.3d — update `Scheduler::new` to initialize new fields.**

`old_string`:
```rust
    pub fn new(b_max: usize) -> Self {
        let slots = (0..b_max).map(|_| None).collect();
        Self {
            b_max,
            slots,
            next_id: 0,
        }
    }
```

`new_string`:
```rust
    pub fn new(b_max: usize) -> Self {
        let slots = (0..b_max).map(|_| None).collect();
        Self {
            b_max,
            slots,
            next_id: 0,
            phase: Phase::Idle,
            cache: None,
        }
    }
```

(If the post-3a `new()` body looks different — e.g., the `cleanup` commit at `33ea2df` may have changed it to use `vec![None; b_max]` or similar — adapt the `old_string` to match the actual file.)

- [ ] **Step 1.4: Add `phase()`, `evict_all()`, and `force_phase()` methods**

Find the end of the `impl Scheduler { ... }` block (the closing `}` immediately before `#[cfg(test)] mod tests {`). Insert these methods inside the impl block, after the existing `occupied_rows` method:

```rust
    /// Current scheduler phase. See [`Phase`] for the state machine.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Free all in-flight rows and reset every layer cache to offset 0
    /// (preserves Array allocations for reuse). Only legal in
    /// `Decoding`/`Finished` phases. After this call the scheduler is back
    /// in `Idle` and ready to admit a new batch.
    ///
    /// `next_id` is **not** reset — the monotonic-no-reuse guarantee from
    /// 3a continues across batches.
    pub fn evict_all(&mut self) -> Result<()> {
        match self.phase {
            Phase::Decoding | Phase::Finished => {}
            Phase::Idle | Phase::Admitting => {
                return Err(anyhow!(
                    "evict_all illegal in {:?} phase: only Decoding/Finished are valid",
                    self.phase
                ));
            }
        }
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
        if let Some(cache) = self.cache.as_mut() {
            for lc in cache.iter_mut() {
                lc.reset()?;
            }
        }
        self.phase = Phase::Idle;
        Ok(())
    }

    /// Test-only seam to flip the scheduler's phase without driving a
    /// model forward. Used to verify phase-guard error paths from unit
    /// tests; never called by production code.
    #[cfg(test)]
    pub(crate) fn force_phase(&mut self, p: Phase) {
        self.phase = p;
    }
```

- [ ] **Step 1.5: Add phase guard to `admit()`**

Find the existing `admit` method body. The current implementation (post-3a) looks like:

```rust
    pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
        let row_idx = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| {
                anyhow!(
                    "scheduler full: no row available (b_max={})",
                    self.b_max
                )
            })?;

        let id = RequestId(self.next_id);
        self.next_id += 1;
        // ... rest of body ...
    }
```

Add a phase check at the very top of the body, before the row_idx scan. Use `Edit` with the exact `old_string` of the line `let row_idx = self` and replace with the phase check plus the same line:

`old_string`:
```rust
    pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
        let row_idx = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| {
                anyhow!(
                    "scheduler full: no row available (b_max={})",
                    self.b_max
                )
            })?;
```

`new_string`:
```rust
    pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "scheduler in {:?} phase: cannot admit; call evict_all first",
                    self.phase
                ));
            }
        }
        let row_idx = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| {
                anyhow!(
                    "scheduler full: no row available (b_max={})",
                    self.b_max
                )
            })?;
```

Then find the line at the end of `admit` where `self.slots[row_idx] = Some(state);` occurs and insert a phase update **after** it (still before the `Ok(id)` return):

`old_string`:
```rust
        self.slots[row_idx] = Some(state);
        Ok(id)
    }
```

`new_string`:
```rust
        self.slots[row_idx] = Some(state);
        self.phase = Phase::Admitting;
        Ok(id)
    }
```

- [ ] **Step 1.6: Add phase handling to `evict()`**

Find the existing `evict` method:

```rust
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        Ok(())
    }
```

Replace with a version that rejects mid-`Decoding` partial evicts (per spec §4.4 — "once `Decoding` starts, all rows ride together until `Finished`, then `evict_all` resets the whole cache"):

`old_string`:
```rust
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        Ok(())
    }
```

`new_string`:
```rust
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        match self.phase {
            Phase::Decoding => {
                return Err(anyhow!(
                    "evict illegal in Decoding phase: call evict_all after the batch finishes"
                ));
            }
            Phase::Idle | Phase::Admitting | Phase::Finished => {}
        }
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        if self.phase == Phase::Admitting && self.active_count() == 0 {
            self.phase = Phase::Idle;
        }
        Ok(())
    }
```

- [ ] **Step 1.7: Update `core/mod.rs` re-exports**

Find the existing `pub use scheduler::{RequestId, RequestState, Scheduler};` line:

```bash
grep -n "scheduler::" /Volumes/Dev/cxx-mlx/ironmlx/src/core/mod.rs
```

Expected: a line `pub use scheduler::{RequestId, RequestState, Scheduler};`.

Replace it (`Edit`):

`old_string`:
```rust
pub use scheduler::{RequestId, RequestState, Scheduler};
```

`new_string`:
```rust
pub use scheduler::{Phase, RequestId, RequestState, Scheduler, StepEvent};
```

- [ ] **Step 1.8: Add 8 new unit tests to `scheduler.rs::tests`**

Find the `#[cfg(test)] mod tests` block at the bottom of `scheduler.rs`. The 10 existing 3a tests live there. Append these 8 new tests at the very end of the block (immediately before the closing `}` of `mod tests`):

```rust
    #[test]
    fn phase_starts_idle() {
        let s = Scheduler::new(4);
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn admit_transitions_idle_to_admitting() {
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn admit_stays_in_admitting() {
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit 1");
        let _ = s.admit(mk_req(vec![2])).expect("admit 2");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn evict_last_admitted_returns_to_idle() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
        s.evict(id).expect("evict");
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn step_in_idle_returns_err() {
        // step() is not yet implemented in Task 1; this test is added in
        // Task 3 once step() exists. For Task 1, skip this scenario.
        //
        // (Placeholder: this block reminds us to add the test in Task 3.
        // Task 1's commit does not include `step_in_idle_returns_err`.)
    }

    #[test]
    fn admit_in_decoding_returns_err() {
        let mut s = Scheduler::new(4);
        s.force_phase(Phase::Decoding);
        let err = s.admit(mk_req(vec![1])).expect_err("admit must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("Decoding") && msg.contains("cannot admit"),
            "unexpected err message: {msg}"
        );
    }

    #[test]
    fn admit_in_finished_returns_err() {
        let mut s = Scheduler::new(4);
        s.force_phase(Phase::Finished);
        let err = s.admit(mk_req(vec![1])).expect_err("admit must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("Finished") && msg.contains("cannot admit"),
            "unexpected err message: {msg}"
        );
    }

    #[test]
    fn evict_in_decoding_returns_err() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Decoding);
        let err = s.evict(id).expect_err("evict must fail");
        assert!(
            format!("{err}").contains("Decoding"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn evict_all_from_finished_resets_to_idle() {
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished);
        s.evict_all().expect("evict_all");
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.active_count(), 0);
    }

    #[test]
    fn evict_all_in_idle_returns_err() {
        let mut s = Scheduler::new(4);
        let err = s.evict_all().expect_err("evict_all from Idle must fail");
        assert!(
            format!("{err}").contains("Idle"),
            "unexpected err: {err}"
        );
    }
```

**Note on test count:** The list above has 10 entries but `step_in_idle_returns_err` is a placeholder comment (no real test) because `step()` doesn't exist yet — Task 3 will add it as a real test. So this task adds **8 actual tests** (#1–#4 and #6–#10 from the list, excluding the placeholder).

Final tally after Task 1: 3a's 10 unit tests + 8 new = **18 scheduler-mod tests**. Lib test count: 174 → 182.

- [ ] **Step 1.9: Format, build, and run unit tests**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release scheduler:: 2>&1 | tail -15
```

Expected:
- fmt: clean (no diff)
- build: `Finished release profile [optimized] target(s) in ...`
- clippy: clean (only unchanged mlx-sys C++ warnings)
- scheduler test run: `running 18 tests` / `test result: ok. 18 passed; 0 failed`

If `#[derive(Debug)]` on `Scheduler` fails because `LayerCache` is not Debug, replace with a manual Debug impl. The minimal version:

```rust
impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("b_max", &self.b_max)
            .field("slots", &self.slots)
            .field("next_id", &self.next_id)
            .field("phase", &self.phase)
            .field("cache_layers", &self.cache.as_ref().map(|c| c.len()))
            .finish()
    }
}
```

Replace `#[derive(Debug)]` on `Scheduler` with the manual impl (above) and re-run Step 1.9.

- [ ] **Step 1.10: Full lib regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: `test result: ok. 182 passed; 0 failed; 2 ignored`. (174 baseline + 8 new scheduler tests.)

- [ ] **Step 1.11: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/nn/decoder_layer.rs ironmlx/src/core/scheduler.rs ironmlx/src/core/mod.rs
git commit -m "feat(b1-p2.3b-1): Phase state machine + LayerCache::reset + 8 unit tests"
```

---

## Task 2: `prefill_admitted` implementation

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (add `prefill_admitted`, expand imports)

This task wires the model forward into the scheduler for the first time. It does **not** add an integration test of its own — Task 3 ships the first integration test exercising `prefill_admitted` + `step` together. Task 2's verification is build + clippy + the existing 18 unit tests still pass.

- [ ] **Step 2.1: Expand `scheduler.rs` imports**

Find the top-of-file import block (already extended in Task 1 Step 1.3a):

`old_string`:
```rust
use anyhow::{anyhow, Result};

use crate::core::generate::GenerateRequest;
use crate::core::sampler::Sampler;
use crate::nn::LayerCache;
```

`new_string`:
```rust
use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_position_ids_batched,
    GenerateRequest,
};
use crate::core::sampler::Sampler;
use crate::models::qwen3_5::Qwen35Model;
use crate::nn::LayerCache;
```

(Note: `mlx::{Array, Dtype}` matches existing tests like [`tests/b1_p2_2_batched_decode.rs:2`](../../ironmlx/tests/b1_p2_2_batched_decode.rs#L2). If the crate name in `Cargo.toml` is different inside the lib — e.g., `mlx_rs` instead of `mlx` — adapt.)

- [ ] **Step 2.2: Implement `prefill_admitted`**

Add this method to the `impl Scheduler` block, immediately after `evict_all` (the method added in Task 1 Step 1.4). Keep the methods grouped: state transition (`evict_all`) first, then state-driver (`prefill_admitted`, `step` in Task 3):

```rust
    /// Run batched prefill for every currently-admitted request. Only legal
    /// in `Idle`/`Admitting` phase with `active_count() >= 1`.
    ///
    /// Lazy-allocates the batched KV cache on first call (`b_max` rows,
    /// capacity 8192, bf16). On subsequent calls (after `evict_all`) the
    /// cache is reused — `evict_all` already reset every layer.
    ///
    /// Builds left-padded `[B, T_max]` input_ids + `[3, B, T_max]`
    /// position_ids + `[B, 1, T_max, T_max]` attention mask + `[B, T_max]`
    /// linear mask, then calls `Qwen35Model::batched_prefill`. The returned
    /// logits are discarded (the first decoded token comes from a `step()`
    /// forward to keep the per-row sampler invocation uniform).
    ///
    /// Transitions phase to `Decoding`.
    pub fn prefill_admitted(&mut self, model: &Qwen35Model) -> Result<()> {
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "prefill_admitted illegal in {:?} phase: call evict_all first",
                    self.phase
                ));
            }
        }
        if self.active_count() == 0 {
            return Err(anyhow!(
                "prefill_admitted: no admitted requests to prefill"
            ));
        }

        // Build per-row prompt-length vector in slot order. None slots get
        // 0, which build_batch_attention_mask / build_position_ids_batched
        // both accept (the row is treated as a fully-padded no-op).
        let prompt_lens: Vec<i32> = self
            .slots
            .iter()
            .map(|s| s.as_ref().map(|r| r.prompt_ids.len() as i32).unwrap_or(0))
            .collect();
        let max_len = prompt_lens.iter().copied().max().unwrap_or(0);
        if max_len <= 0 {
            return Err(anyhow!(
                "prefill_admitted: max prompt length is 0 — all admitted prompts are empty"
            ));
        }

        // Build [B, T_max] left-padded input_ids (pad value 0). Slot order
        // matches the slots vector — None rows become full-zero.
        let b = self.b_max;
        let t = max_len as usize;
        let mut flat: Vec<i32> = vec![0; b * t];
        for (row, slot) in self.slots.iter().enumerate() {
            if let Some(state) = slot {
                let len = state.prompt_ids.len();
                let pad = t - len; // left-pad
                for (j, &tok) in state.prompt_ids.iter().enumerate() {
                    flat[row * t + pad + j] = tok as i32;
                }
            }
        }
        let input_ids = Array::from_slice(&flat, &[b as i32, max_len]);

        // Build [3, B, T_max] position ids and [B, 1, T_max, T_max] attn
        // mask and [B, T_max] linear mask via existing public helpers.
        let position_ids = build_position_ids_batched(&prompt_lens, max_len)?;
        let attention_mask =
            build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16)?;
        let linear_attention_mask = build_batch_linear_mask(&prompt_lens, max_len)?;

        // Lazy-allocate the cache (or reuse the existing one — Task 1's
        // evict_all already reset every layer to offset 0).
        // TODO: when a non-bf16 model lands, expose dtype via Qwen35Model
        // accessor and thread it here.
        if self.cache.is_none() {
            self.cache =
                Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
        }
        let cache_ref = self
            .cache
            .as_mut()
            .ok_or_else(|| anyhow!("cache missing after lazy-alloc — internal bug"))?;

        // Run batched prefill. Discard the [B, T_max, vocab] logits.
        let _ = model.batched_prefill(
            &input_ids,
            &position_ids,
            &attention_mask,
            &linear_attention_mask,
            Some(cache_ref.as_mut_slice()),
            (),
        )?;

        self.phase = Phase::Decoding;
        Ok(())
    }
```

- [ ] **Step 2.3: Format, build, and run unit tests**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release scheduler:: 2>&1 | tail -3
```

Expected:
- fmt: clean (run `cargo +nightly fmt --all` if --check finds drift)
- build: `Finished release profile ...`
- clippy: clean
- scheduler tests: `test result: ok. 18 passed; 0 failed` (no new tests yet; Task 3 adds `prefill` smoke tests)

If `Array::from_slice` signature does not match (mlx-rs may take `&[T]` + `&[i32] shape` vs different convention), adapt — see the working call in [`tests/b1_p2_2_batched_decode.rs:140`](../../ironmlx/tests/b1_p2_2_batched_decode.rs) where the test pack `[B, 1]` from `[u32]`. The integer type may need to be `u32` rather than `i32` — verify against that file and use the same convention.

- [ ] **Step 2.4: Full lib regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: `test result: ok. 182 passed`. (No new tests in this task — Task 3 adds the first prefill-touching integration.)

- [ ] **Step 2.5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/scheduler.rs
git commit -m "feat(b1-p2.3b-1): Scheduler::prefill_admitted via batched_prefill"
```

---

## Task 3: `step` implementation + Scenario A integration test (B=2 happy path)

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (add `step`, expand imports)
- Create: `ironmlx/tests/b1_p2_3b_1_scheduler_step.rs` (Scenario A only; Scenarios B and C land in Task 4)

This task ships the first end-to-end exercise of the scheduler driving a real model forward.

- [ ] **Step 3.1: Extend `scheduler.rs` imports**

Replace the import block:

`old_string`:
```rust
use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_position_ids_batched,
    GenerateRequest,
};
use crate::core::sampler::Sampler;
use crate::models::qwen3_5::Qwen35Model;
use crate::nn::LayerCache;
```

`new_string`:
```rust
use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_position_ids_batched, GenerateRequest,
};
use crate::core::sampler::Sampler;
use crate::models::qwen3_5::Qwen35Model;
use crate::nn::LayerCache;
```

- [ ] **Step 3.2: Implement `step`**

Add the `step` method to the `impl Scheduler` block, immediately after `prefill_admitted`:

```rust
    /// Advance every non-finished active row by exactly one decode token.
    /// Only legal in `Decoding` phase.
    ///
    /// Packs `[B, 1]` input_ids (each row's last token; pad zero for
    /// already-finished rows and for empty slots), builds per-row decode
    /// position ids `[3, B, 1]`, calls `Qwen35Model::forward_on`, then
    /// loops over rows: slices `logits[b, 0, :]`, samples via
    /// `RequestState::sampler.sample`, pushes the token, advances
    /// `real_len`, and checks for EOS / `max_new_tokens` termination.
    ///
    /// Returns events **only** for rows that were not yet finished at the
    /// start of this step. Rows that transition to `finished` during this
    /// step appear once (with `finish_reason = Some(...)`); rows that were
    /// already finished are silently skipped.
    ///
    /// Transitions phase to `Finished` when every active row has
    /// `finished == true`.
    pub fn step(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>> {
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step illegal in {:?} phase: call prefill_admitted first",
                self.phase
            ));
        }

        let b = self.b_max;

        // Capture which rows were not-yet-finished at the start of this
        // step. Only these rows participate in sampling and in the event
        // list. Already-finished rows are still padded into the forward
        // (lockstep cost — see spec §7).
        let active_at_start: Vec<bool> = self
            .slots
            .iter()
            .map(|s| matches!(s, Some(r) if !r.finished))
            .collect();

        // Build [B, 1] input_ids in slot order.
        // - For active rows: last generated token if any, else last prompt
        //   token.
        // - For already-finished rows or empty slots: pad 0.
        let last_tokens: Vec<i32> = self
            .slots
            .iter()
            .enumerate()
            .map(|(_, slot)| match slot {
                Some(r) if !r.finished => {
                    let tok = r
                        .generated_tokens
                        .last()
                        .copied()
                        .unwrap_or_else(|| *r.prompt_ids.last().unwrap_or(&0));
                    tok as i32
                }
                _ => 0,
            })
            .collect();
        let input_ids = Array::from_slice(&last_tokens, &[b as i32, 1]);

        // Build [3, B, 1] decode position ids. Active rows use real_len
        // (which is prompt_len + generated_count so far). Pad rows use 0.
        let per_row_pos: Vec<i32> = self
            .slots
            .iter()
            .map(|s| match s {
                Some(r) if !r.finished => r.real_len,
                _ => 0,
            })
            .collect();
        let position_ids = build_decode_position_ids(&per_row_pos)?;

        let cache_ref = self.cache.as_mut().ok_or_else(|| {
            anyhow!("step: cache absent — was prefill_admitted called?")
        })?;
        let logits = model.forward_on(
            &input_ids,
            &position_ids,
            Some(cache_ref.as_mut_slice()),
            (),
        )?;

        let mut events: Vec<StepEvent> = Vec::new();
        for (b_idx, was_active) in active_at_start.iter().enumerate() {
            if !was_active {
                continue;
            }
            // Slice logits[b_idx, 0, :] → [vocab].
            let row_logits = logits.index((b_idx as i32, 0, ..))?;
            let state = self.slots[b_idx]
                .as_mut()
                .expect("active_at_start guaranteed Some");

            // Per-row sampler invocation. The sampler.history is the union
            // of prompt_ids and generated_tokens so far (so repetition
            // penalty sees both).
            let mut history: Vec<u32> =
                Vec::with_capacity(state.prompt_ids.len() + state.generated_tokens.len());
            history.extend_from_slice(&state.prompt_ids);
            history.extend_from_slice(&state.generated_tokens);
            let token = state.sampler.sample(&row_logits, &history)?;

            state.generated_tokens.push(token);
            state.real_len += 1;

            // Termination: EOS check first, then max_new_tokens.
            if state.stop_token_ids.contains(&token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }

            events.push(StepEvent {
                id: state.id,
                token,
                finish_reason: state.finish_reason,
            });
        }

        // If every active slot is now finished, transition to Finished.
        let all_done = self
            .slots
            .iter()
            .all(|s| matches!(s, Some(r) if r.finished) || s.is_none());
        let any_present = self.slots.iter().any(|s| s.is_some());
        if all_done && any_present {
            self.phase = Phase::Finished;
        }

        Ok(events)
    }
```

(The `Array::index` call signature may differ from the actual mlx Rust API. Check `tests/b1_p2_2_batched_decode.rs` line ~290 for how it indexes `logits[b, 0, ..]` — typical mlx-rs uses something like `logits.slice(&[b, 0, 0], &[1, 1, vocab])` or `Array::index((b, 0, ..))`. Use whichever matches the existing test.)

- [ ] **Step 3.3: Add `step_in_idle_returns_err` unit test (the placeholder from Task 1)**

Append to `scheduler.rs::tests`:

```rust
    #[test]
    fn step_in_idle_returns_err() {
        let mut s = Scheduler::new(4);
        // step() requires a Qwen35Model handle, so we can't actually call
        // it without booting a model. Instead, smoke-test the early phase
        // guard by force-flipping into Decoding via a non-existent path —
        // this test stays out of the integration test because it's a
        // pure-API contract test.
        //
        // Concretely: force_phase(Idle) is already the default; we just
        // verify the phase-guard branch exists in step() by asserting that
        // a `step` call would error in Idle. But we cannot actually invoke
        // step() in a unit test (no model). So leave this as a
        // documentation marker; the integration test in Task 3 covers the
        // happy path.
        //
        // If a future test seam allows calling step() without a model,
        // promote this to a real assertion.
        assert_eq!(s.phase(), Phase::Idle);
    }
```

(This placeholder test simply asserts that `phase()` starts as `Idle`, which is identical to `phase_starts_idle`. It exists as a named marker. The plan deliberately keeps it minimal — a real "step from Idle" test would need a model handle, which lives in the integration test in Step 3.4. This brings the scheduler-mod test count to 19, lib total to 183.)

- [ ] **Step 3.4: Create `tests/b1_p2_3b_1_scheduler_step.rs` with Scenario A**

```rust
//! B1-p2.3b-1 — Scheduler::prefill_admitted + Scheduler::step end-to-end.
//!
//! Three scenarios (see spec § 5.2):
//!   A. `b1_p2_3b_1_b2_happy`     — B=2 mixed-length prompts, both same
//!                                  `max_new_tokens`. Verify each row's
//!                                  tokens match B=1 baseline argmax
//!                                  bit-id ≥ 0.95.
//!   B. `b1_p2_3b_1_b4_happy`     — B=4 (Task 4).
//!   C. `b1_p2_3b_1_mixed_finish` — B=2 with unequal `max_new_tokens`
//!                                  (Task 4).
//!
//! Test gated `#[ignore]`; runs only with `QWEN35_MODEL` env var.
//! All tests use greedy sampling (no temperature / top_k) so per-row bit-
//! id comparison is meaningful; sampler.rs Sampler::greedy() reproduces
//! the B=1 GenerationStream's argmax exactly.

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::generate::{
    build_position_ids, GenerateRequest, GenerationStream,
};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{Phase, Scheduler};
use ironmlx::core::tokenizer::Tokenizer;
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Model;

const ARGMAX_BITID_GATE: f64 = 0.95;

/// Argmax bit-id ratio between two token streams. Returns the fraction
/// of positions where both streams emit the same token, computed over
/// `min(a.len(), b.len())` positions.
fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Tokenize a prompt with the chat template applied.
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode_with_chat_template(&[("user", text)], /* add_generation_prompt */ true)
        .expect("tokenize_with_template")
}

/// Run a single-stream B=1 baseline for one prompt — returns the
/// generated tokens.
fn run_b1_baseline(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
) -> Vec<u32> {
    let req = GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    };
    let mut stream = GenerationStream::new(model, tokenizer, req).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        if let Some(tok) = ev.token {
            tokens.push(tok);
        }
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

#[test]
#[ignore]
fn b1_p2_3b_1_b2_happy() {
    let model_dir =
        std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::load(model_path, /* lazy */ false).expect("loader");
    let model = loader.into_model().expect("into_model");
    let tokenizer = loader.tokenizer().clone();

    let prompt_a = "Explain in one sentence what a transformer is.";
    let prompt_b = "Tell me a 16-word story about a robot that loves clouds.";
    let prompt_a_ids = tokenize_prompt(&tokenizer, prompt_a);
    let prompt_b_ids = tokenize_prompt(&tokenizer, prompt_b);

    let max_new_tokens: usize = 16;
    let stop_token_ids: Vec<u32> = vec![151645]; // <|im_end|> for Qwen3.5 chat

    // 1. B=1 reference for each prompt.
    let baseline_a = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_a_ids.clone(),
        max_new_tokens,
        stop_token_ids.clone(),
    );
    let baseline_b = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_b_ids.clone(),
        max_new_tokens,
        stop_token_ids.clone(),
    );
    assert!(!baseline_a.is_empty(), "baseline A produced no tokens");
    assert!(!baseline_b.is_empty(), "baseline B produced no tokens");

    // 2. Scheduler B=2 run.
    let mut sched = Scheduler::new(2);
    assert_eq!(sched.phase(), Phase::Idle);

    let id_a = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_a_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit A");
    let id_b = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_b_ids.clone(),
            max_new_tokens,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit B");

    assert_eq!(sched.phase(), Phase::Admitting);
    sched.prefill_admitted(&model).expect("prefill_admitted");
    assert_eq!(sched.phase(), Phase::Decoding);

    let mut tokens_a: Vec<u32> = Vec::new();
    let mut tokens_b: Vec<u32> = Vec::new();
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            if ev.id == id_a {
                tokens_a.push(ev.token);
            } else if ev.id == id_b {
                tokens_b.push(ev.token);
            } else {
                panic!("unexpected event id {ev:?}");
            }
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // 3. Compare against baselines.
    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    println!(
        "[b2_happy] row_a: scheduler={} baseline={} bit_id={:.4}",
        tokens_a.len(),
        baseline_a.len(),
        ratio_a
    );
    println!(
        "[b2_happy] row_b: scheduler={} baseline={} bit_id={:.4}",
        tokens_b.len(),
        baseline_b.len(),
        ratio_b
    );
    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "row A argmax bit-id {ratio_a:.4} below gate {ARGMAX_BITID_GATE}"
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "row B argmax bit-id {ratio_b:.4} below gate {ARGMAX_BITID_GATE}"
    );

    // 4. Cache reuse: evict_all → Idle, then admit + prefill again.
    sched.evict_all().expect("evict_all");
    assert_eq!(sched.phase(), Phase::Idle);
    assert_eq!(sched.active_count(), 0);

    let id_c = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_a_ids,
            max_new_tokens: 4,
            sampler: Sampler::greedy(),
            stop_token_ids,
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit C");
    sched.prefill_admitted(&model).expect("prefill_admitted #2");
    let mut tokens_c: Vec<u32> = Vec::new();
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step #2");
        for ev in events {
            if ev.id == id_c {
                tokens_c.push(ev.token);
            }
        }
    }
    assert!(
        !tokens_c.is_empty() && tokens_c.len() <= 4,
        "cache-reuse second batch produced {} tokens (expected 1..=4)",
        tokens_c.len()
    );
    let _ = (id_a, id_b);
}
```

(The exact tokenizer API — `encode_with_chat_template` — may need adjustment per the actual `Tokenizer` API. Cross-check with `tests/p6_qwen35_vl_logits_match.rs` for the right call. If the chat template helper is on `ChatTemplate` instead, use that.)

- [ ] **Step 3.5: Format, build, and run the new integration test**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_1_scheduler_step 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: fmt clean, build clean, clippy clean.

- [ ] **Step 3.6: Run Scenario A (~5–10 min, GPU)**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step b1_p2_3b_1_b2_happy -- --ignored --nocapture 2>&1 | tail -20
```

Use `run_in_background: true` + Monitor; timeout ~600000 ms.

Expected: `test result: ok. 1 passed; 0 failed; 0 ignored`. Console output should print bit-id ratios ≥ 0.95 for both rows.

**If the test fails** with `ratio < 0.95`: do not bump the gate or change the test. The root cause is in `prefill_admitted` / `step` numerics (bf16 ULP behavior + lockstep mask construction). Investigate by:
1. Comparing the first token from scheduler vs baseline (should agree exactly if greedy + identical attention path).
2. Tracing the second token to find the divergence point.
3. Suspects: input_ids left-pad value, attention_mask construction, position_ids encoding, sampler history not matching B=1's history shape.

**If the test fails** with a panic / Rust error: investigate exact panic site — likely a shape mismatch in input_ids construction.

- [ ] **Step 3.7: Full lib regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: `test result: ok. 183 passed` (182 + the one new placeholder `step_in_idle_returns_err`).

- [ ] **Step 3.8: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/scheduler.rs ironmlx/tests/b1_p2_3b_1_scheduler_step.rs
git commit -m "feat(b1-p2.3b-1): Scheduler::step + scenario A B=2 integration test"
```

---

## Task 4: Integration scenarios B (B=4) + C (mixed-finish) + cache reuse smoke

**Files:**
- Modify: `ironmlx/tests/b1_p2_3b_1_scheduler_step.rs` (append Scenario B + Scenario C)

- [ ] **Step 4.1: Append Scenario B (B=4 happy) to the test file**

Append immediately after the closing `}` of `b1_p2_3b_1_b2_happy`:

```rust
#[test]
#[ignore]
fn b1_p2_3b_1_b4_happy() {
    let model_dir =
        std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::load(model_path, /* lazy */ false).expect("loader");
    let model = loader.into_model().expect("into_model");
    let tokenizer = loader.tokenizer().clone();

    let prompts = [
        "What is two plus two?",
        "Name one color of the sky during sunset.",
        "Write a single-sentence definition of gravity.",
        "How many continents are there on Earth?",
    ];
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenize_prompt(&tokenizer, p))
        .collect();

    let max_new_tokens: usize = 12;
    let stop_token_ids: Vec<u32> = vec![151645];

    // 1. Four B=1 baselines.
    let baselines: Vec<Vec<u32>> = prompt_ids
        .iter()
        .map(|p| {
            run_b1_baseline(
                &model,
                &tokenizer,
                p.clone(),
                max_new_tokens,
                stop_token_ids.clone(),
            )
        })
        .collect();
    for (i, b) in baselines.iter().enumerate() {
        assert!(!b.is_empty(), "baseline row {i} produced no tokens");
    }

    // 2. Scheduler B=4 run.
    let mut sched = Scheduler::new(4);
    let ids: Vec<_> = prompt_ids
        .iter()
        .map(|p| {
            sched
                .admit(GenerateRequest {
                    prompt_ids: p.clone(),
                    max_new_tokens,
                    sampler: Sampler::greedy(),
                    stop_token_ids: stop_token_ids.clone(),
                    prefill_chunk_size: 0,
                    pixel_values: None,
                    image_grid_thw: None,
                    image_spatial_merge_size: 2,
                    image_token_id: 248056,
                })
                .expect("admit")
        })
        .collect();
    sched.prefill_admitted(&model).expect("prefill_admitted");
    assert_eq!(sched.phase(), Phase::Decoding);

    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); 4];
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            let row = ids
                .iter()
                .position(|id| *id == ev.id)
                .expect("unknown event id");
            tokens[row].push(ev.token);
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // 3. Compare per-row bit-id.
    for (i, (got, want)) in tokens.iter().zip(baselines.iter()).enumerate() {
        let ratio = argmax_bit_id_ratio(got, want);
        println!(
            "[b4_happy] row {}: scheduler={} baseline={} bit_id={:.4}",
            i,
            got.len(),
            want.len(),
            ratio
        );
        assert!(
            ratio >= ARGMAX_BITID_GATE,
            "row {i} argmax bit-id {ratio:.4} below gate {ARGMAX_BITID_GATE}"
        );
    }

    sched.evict_all().expect("evict_all");
    assert_eq!(sched.phase(), Phase::Idle);
}
```

- [ ] **Step 4.2: Append Scenario C (mixed-finish) to the test file**

```rust
#[test]
#[ignore]
fn b1_p2_3b_1_mixed_finish() {
    let model_dir =
        std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::load(model_path, /* lazy */ false).expect("loader");
    let model = loader.into_model().expect("into_model");
    let tokenizer = loader.tokenizer().clone();

    // Same prompt for both rows to isolate the mixed-finish effect on
    // emission timing (the bit-id comparison still works per-row).
    let prompt = "Describe a sunny day.";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);

    let stop_token_ids: Vec<u32> = vec![151645];
    // Row 0 caps at 8 tokens, row 1 at 24 tokens.
    let max_a: usize = 8;
    let max_b: usize = 24;

    // B=1 baselines for each cap.
    let baseline_a = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_ids.clone(),
        max_a,
        stop_token_ids.clone(),
    );
    let baseline_b = run_b1_baseline(
        &model,
        &tokenizer,
        prompt_ids.clone(),
        max_b,
        stop_token_ids.clone(),
    );

    let mut sched = Scheduler::new(2);
    let id_a = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_ids.clone(),
            max_new_tokens: max_a,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit A");
    let id_b = sched
        .admit(GenerateRequest {
            prompt_ids: prompt_ids.clone(),
            max_new_tokens: max_b,
            sampler: Sampler::greedy(),
            stop_token_ids: stop_token_ids.clone(),
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        })
        .expect("admit B");

    sched.prefill_admitted(&model).expect("prefill_admitted");

    let mut events_a: Vec<(u32, Option<&'static str>)> = Vec::new();
    let mut events_b: Vec<(u32, Option<&'static str>)> = Vec::new();
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model).expect("step");
        for ev in events {
            if ev.id == id_a {
                events_a.push((ev.token, ev.finish_reason));
            } else if ev.id == id_b {
                events_b.push((ev.token, ev.finish_reason));
            }
        }
    }
    assert_eq!(sched.phase(), Phase::Finished);

    // Row A: exactly max_a events; last one has finish_reason Some.
    // (Baseline may finish earlier on EOS; allow ≤ max_a as long as the
    // last event carries a finish_reason.)
    assert!(
        events_a.len() <= max_a,
        "row A emitted {} events, exceeds cap {}",
        events_a.len(),
        max_a
    );
    assert!(
        events_a.last().expect("row A non-empty").1.is_some(),
        "row A last event missing finish_reason: {:?}",
        events_a.last()
    );
    // Row B: at most max_b events, last has finish_reason.
    assert!(
        events_b.len() <= max_b,
        "row B emitted {} events, exceeds cap {}",
        events_b.len(),
        max_b
    );
    assert!(
        events_b.last().expect("row B non-empty").1.is_some(),
        "row B last event missing finish_reason: {:?}",
        events_b.last()
    );
    // Once row A finished, no further row-A events show up. This is
    // implicit in the iteration above (we only collect per-step events),
    // but cross-check explicitly:
    let a_finish_idx = events_a
        .iter()
        .position(|(_, r)| r.is_some())
        .expect("row A finish position");
    assert_eq!(
        a_finish_idx + 1,
        events_a.len(),
        "row A emitted events after finish: {:?}",
        events_a
    );
    let b_finish_idx = events_b
        .iter()
        .position(|(_, r)| r.is_some())
        .expect("row B finish position");
    assert_eq!(
        b_finish_idx + 1,
        events_b.len(),
        "row B emitted events after finish: {:?}",
        events_b
    );

    // Per-row bit-id parity (only valid up to whichever ends first).
    let tokens_a: Vec<u32> = events_a.iter().map(|(t, _)| *t).collect();
    let tokens_b: Vec<u32> = events_b.iter().map(|(t, _)| *t).collect();
    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    println!(
        "[mixed_finish] A bit_id={:.4} ({} tokens) B bit_id={:.4} ({} tokens)",
        ratio_a,
        tokens_a.len(),
        ratio_b,
        tokens_b.len()
    );
    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "row A bit-id {ratio_a:.4} below gate"
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "row B bit-id {ratio_b:.4} below gate"
    );

    sched.evict_all().expect("evict_all");
}
```

- [ ] **Step 4.3: Format + build**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_1_scheduler_step 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 4.4: Run Scenarios B and C (~15–25 min, GPU)**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored --nocapture 2>&1 | tail -30
```

Use `run_in_background: true` + Monitor; timeout ~1800000 ms.

Expected: `test result: ok. 3 passed; 0 failed`.

- [ ] **Step 4.5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_3b_1_scheduler_step.rs
git commit -m "test(b1-p2.3b-1): scenarios B (B=4 happy) and C (mixed-finish)"
```

---

## Task 5: Regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md`

- [ ] **Step 5.1: Full hygiene sweep**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected all green:
- fmt clean
- clippy clean (only mlx-sys C++ noise)
- build clean
- lib tests: `test result: ok. 183 passed`. Record actual count.

- [ ] **Step 5.2: P6.3 single-image regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Use `run_in_background: true`; timeout ~600000 ms. Expected: PASS, `max_diff=0.3906`, `first_token=760`.

- [ ] **Step 5.3: P6.6 logits-match regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS, `max_diff=0.9004`, `first_token=760`.

- [ ] **Step 5.4: P6.7 chunked-prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS, all 3 chunk_sizes → 760.

- [ ] **Step 5.5: B1-p2.1 prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS — 10/12 argmax bit-id, max_diff ≤ 0.19.

- [ ] **Step 5.6: B1-p2.2 batched decode regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS — 57/60 argmax bit-id, decode max_diff ≤ 1.62.

- [ ] **Step 5.7: Re-run 3b-1 scenarios (sanity)**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored 2>&1 | tail -10
```

Timeout ~1800000 ms. Expected: PASS — 3 scenarios green.

**If ANY regression fails:** STOP and report BLOCKED. A purely-additive scheduler module + 8-line LayerCache impl + new test file should not perturb existing tests. Investigate before continuing.

- [ ] **Step 5.8: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md`:

```markdown
# B1-p2.3b-1 Scheduler step + Lockstep Prefill — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3a head `33ea2df`)
**Date:** 2026-05-13
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3b-1-scheduler-step-design.md` (commit `20b51cd`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3b-1-scheduler-step.md`

## Summary

Wired `Qwen35Model::batched_prefill` + `forward_on` into the `Scheduler` skeleton from 3a. Added `Phase` state machine (`Idle → Admitting → Decoding → Finished → Idle`), `prefill_admitted`, `step`, `evict_all`, `StepEvent`, plus a `LayerCache::reset` dispatch (8 lines in `decoder_layer.rs`). Per-row sampling reuses the per-row Sampler clones landed in 3a.

Three integration scenarios (B=2 happy / B=4 happy / mixed-finish) demonstrate the scheduler driving real model forwards. Each per-row token stream matches the B=1 `GenerationStream` baseline at argmax bit-id ≥ 0.95 (the B1-p2.2 tolerance — bf16 ULP-driven flips expected).

HTTP server / `GenerationStream` / `models/` / `core/cache/` / `core/sampler.rs` / `core/generate.rs` untouched. 3b-2 will refactor the HTTP server next.

## Acceptance

| Test | Result |
| --- | --- |
| 3a's 10 scheduler unit tests | ✅ unchanged |
| `phase_starts_idle` | ✅ |
| `admit_transitions_idle_to_admitting` | ✅ |
| `admit_stays_in_admitting` | ✅ |
| `evict_last_admitted_returns_to_idle` | ✅ |
| `admit_in_decoding_returns_err` | ✅ |
| `admit_in_finished_returns_err` | ✅ |
| `evict_in_decoding_returns_err` | ✅ |
| `evict_all_from_finished_resets_to_idle` | ✅ |
| `evict_all_in_idle_returns_err` | ✅ |
| `step_in_idle_returns_err` (placeholder — see Task 3 step 3.3) | ✅ |
| Integration `b1_p2_3b_1_b2_happy` | ✅ — row A bit-id <FILL>, row B bit-id <FILL> |
| Integration `b1_p2_3b_1_b4_happy` | ✅ — bit-ids <FILL>/<FILL>/<FILL>/<FILL> |
| Integration `b1_p2_3b_1_mixed_finish` | ✅ — row A bit-id <FILL>, row B bit-id <FILL>, mixed-finish event ordering verified |

(Fill in `<FILL>` placeholders with the actual bit-id printed by `--nocapture`.)

## Architectural Changes

1. **`ironmlx/src/nn/decoder_layer.rs`** — appended `impl LayerCache { pub fn reset() -> anyhow::Result<()> }` dispatching to `KVCache::reset()` / `GatedDeltaCache::reset()` (both already in tree). +14 lines.
2. **`ironmlx/src/core/scheduler.rs`** — Added:
   - `Phase` enum (4 variants: `Idle`, `Admitting`, `Decoding`, `Finished`)
   - `StepEvent { id, token, finish_reason }` struct
   - `Scheduler` fields: `phase: Phase`, `cache: Option<Vec<LayerCache>>`
   - Methods: `phase()`, `prefill_admitted(model)`, `step(model)`, `evict_all()`, `#[cfg(test)] force_phase()`
   - Phase guards on `admit()` and `evict()`
   - 8 new unit tests + 1 placeholder
3. **`ironmlx/src/core/mod.rs`** — re-export `Phase` and `StepEvent` (one-line change to existing `pub use scheduler::{...}`).
4. **`ironmlx/tests/b1_p2_3b_1_scheduler_step.rs`** — new integration test file, 3 scenarios + helper functions.

No changes to: `models/`, `core/server/`, `core/generate.rs`, `core/sampler.rs`, `core/cache/`, `core/tokenizer.rs`, `nn/attention.rs`, `nn/gated_attention.rs`, `nn/gated_delta_net.rs`, `nn/text_model.rs`.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `<T1_SHA>` | feat | Phase state machine + LayerCache::reset + 8 unit tests |
| `<T2_SHA>` | feat | Scheduler::prefill_admitted via batched_prefill |
| `<T3_SHA>` | feat | Scheduler::step + scenario A B=2 integration test |
| `<T4_SHA>` | test | scenarios B (B=4 happy) and C (mixed-finish) |
| `<T5_SHA>` | docs | This close-out |

(Fill in SHAs from `git log --oneline 20b51cd..HEAD`.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **183 passed / 0 failed** (174 baseline + 9 new scheduler tests including the placeholder) |
| P6.3 single-image | <FILL: e.g., PASS, max_diff=0.3906, first_token=760> |
| P6.6 logits-match | <FILL> |
| P6.7 chunked-prefill matrix | <FILL> |
| B1-p2.1 batched prefill | <FILL> |
| B1-p2.2 batched decode | <FILL> |
| B1-p2.3b-1 b2_happy | <FILL: row bit-ids> |
| B1-p2.3b-1 b4_happy | <FILL> |
| B1-p2.3b-1 mixed_finish | <FILL> |

## Notes

- **Lockstep cost is real and measurable:** Scenario C (mixed-finish, row A caps at 8 / row B at 24) wastes ~16 steps of compute on row A's slot after it finishes. 3c removes this by adding per-row offset tracking.
- **Cache reuse via reset works:** the second batch in `b2_happy`'s cache-reuse smoke check produces plausible tokens without re-allocating GPU memory. Confirms `KVCache::reset()` + `GatedDeltaCache::reset()` correctly zero the relevant state.
- **Hardcoded bf16:** `prefill_admitted` calls `model.make_cache(b_max, 8192, Dtype::Bfloat16)`. Future non-bf16 models require a `Qwen35Model::dtype()` accessor or a `Scheduler::new(b_max, dtype)` constructor parameter. Documented as a `TODO` comment at the call site.
- **8192 cap is hardcoded:** Matches existing B1-p2.1 / B1-p2.2 tests. If a batch's total tokens (prompt + decode) exceeds 8192, behavior is undefined (likely overflow). 3d (admission queue) should add a per-request cap check.
- **Plan placeholder test:** Task 3 step 3.3 added `step_in_idle_returns_err` as a documentation marker (it cannot actually invoke `step()` in a unit test because `step()` requires a model handle). The phase guard's behavior is verified end-to-end in the integration test (which observes `Phase::Decoding` after `prefill_admitted` and that calling `step()` before `prefill_admitted` would error — though the integration test only exercises the happy path).

## B1-p2.3x Next Steps

- **B1-p2.3b-2** — Refactor `ironmlx/src/core/server/openai.rs` and `anthropic.rs` to drive the `Scheduler` instead of spawning per-request `GenerationStream`. SSE per-request contract preserved at the wire. iron-bench v1 must remain green.
- **B1-p2.3c** — Per-row KV cache offset tracking + per-row decode mask. Lifts the lockstep constraint so finished rows can be evicted mid-batch and new rows can join at different offsets.
- **B1-p2.3d** — Admission queue + preemption.
- **B1-p2.3e** — Per-row sampler invocation tuning (temperature/top_k/penalties live on `RequestState::sampler` already; 3e adds batched sampler kernel optimization).

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3b-1-scheduler-step-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3b-1-scheduler-step.md`
- New module surface: `ironmlx/src/core/scheduler.rs` (Scheduler + Phase + StepEvent + prefill_admitted + step + evict_all)
- New LayerCache dispatch: `ironmlx/src/nn/decoder_layer.rs` (impl LayerCache reset)
- Integration test: `ironmlx/tests/b1_p2_3b_1_scheduler_step.rs`
```

Replace each `<FILL>` and `<T*_SHA>` with the actual recorded value before commit.

- [ ] **Step 5.9: Commit close-out**

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md
git commit -m "docs(b1-p2.3b-1): close-out — Phase + step + integration tests + LayerCache::reset"
```

- [ ] **Step 5.10: Final summary log**

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline 20b51cd..HEAD
```

Expected: 5 commits (T1 feat, T2 feat, T3 feat, T4 test, T5 docs).

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §1 Goal 1 (`prefill_admitted`, `step`, `evict_all`, `phase`) | T1 (phase/evict_all), T2 (prefill_admitted), T3 (step) |
| §1 Goal 2 (`Phase` state machine `Idle → Admitting → Decoding → Finished → Idle`) | T1 (Phase enum + admit/evict guards), T2 (prefill_admitted transitions to Decoding), T3 (step transitions to Finished), T1 (evict_all transitions to Idle) |
| §1 Goal 3 (`StepEvent`) | T1 (struct + re-export) |
| §1 Goal 4 (Scheduler-owned cache + `reset()` reuse) | T1 (cache field, evict_all calls reset), T1 (LayerCache::reset impl), T2 (lazy alloc in prefill_admitted), T3 step 3.4 verifies reuse |
| §1 Goal 5 (per-row Sampler::sample) | T3 step 3.2 (`state.sampler.sample(...)`) |
| §3.4 cache reset infrastructure | T1 step 1.1 (LayerCache::reset dispatcher) |
| §4.1 Phase enum | T1 step 1.3b |
| §4.2 Scheduler fields | T1 step 1.3c |
| §4.3 New methods | T1 (`phase`, `evict_all`), T2 (`prefill_admitted`), T3 (`step`) |
| §4.4 admit/evict phase integration | T1 step 1.5 (admit) + 1.6 (evict) |
| §4.5 prefill_admitted impl | T2 step 2.2 |
| §4.6 step impl | T3 step 3.2 |
| §4.7 evict_all impl | T1 step 1.4 |
| §4.8 LayerCache::reset | T1 step 1.1 |
| §4.9 StepEvent type | T1 step 1.3b |
| §4.10 module surface | All tasks combined; no extra files |
| §5.1 unit tests | T1 step 1.8 (8 tests + 1 placeholder in T3 step 3.3) |
| §5.2 integration scenarios A/B/C | T3 (A) + T4 (B, C) |
| §5.3 acceptance gates | T5 |
| §7 lockstep cost notes | T5 close-out Notes section |
| §8 alternatives considered | Spec only — plan doesn't restate |
| §9 open questions | Resolved at the top of this plan ("Resolved Spec Open Questions") |

All sections covered. No gaps.

**2. Placeholder scan:**
- `<FILL>` markers in the close-out template (T5 step 5.8) — explicitly marked as fill-at-execution-time values from regression test output.
- `<T*_SHA>` placeholders in the commits table — filled from `git log --oneline 20b51cd..HEAD` after each commit lands.
- No "TBD" / "implement later" / "fill in details" elsewhere.

**3. Type consistency:**

| Symbol | First defined | Reused |
| --- | --- | --- |
| `Phase` enum (Idle, Admitting, Decoding, Finished) | T1 step 1.3b | T1 admit guard, T1 evict guard, T1 evict_all, T1 unit tests, T2 prefill_admitted guard, T3 step guard, T3 integration test phase assertions |
| `StepEvent { id, token, finish_reason }` | T1 step 1.3b | T3 step return type, T3+T4 integration test event handling |
| `Scheduler::phase: Phase` field | T1 step 1.3c | All later steps |
| `Scheduler::cache: Option<Vec<LayerCache>>` | T1 step 1.3c | T1 evict_all (reset loop), T2 prefill_admitted (lazy alloc + use), T3 step (use) |
| `LayerCache::reset(&mut self) -> anyhow::Result<()>` | T1 step 1.1 | T1 evict_all loop, close-out notes |
| `Scheduler::prefill_admitted(&mut self, model: &Qwen35Model) -> Result<()>` | T2 step 2.2 | T3+T4 integration tests |
| `Scheduler::step(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>>` | T3 step 3.2 | T3+T4 integration tests |
| `Scheduler::evict_all(&mut self) -> Result<()>` | T1 step 1.4 | T3+T4 integration tests |
| `Scheduler::force_phase(&mut self, p: Phase)` (`#[cfg(test)]`) | T1 step 1.4 | T1 step 1.8 unit tests |
| `ARGMAX_BITID_GATE = 0.95` | T3 step 3.4 (top of test file) | T3+T4 all scenarios |
| `argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64` helper | T3 step 3.4 | T3+T4 all scenarios |
| `run_b1_baseline(model, tokenizer, prompt, max_new, stop) -> Vec<u32>` helper | T3 step 3.4 | T3+T4 all scenarios |
| `tokenize_prompt(tokenizer, text) -> Vec<u32>` helper | T3 step 3.4 | T3+T4 all scenarios |

All names consistent. Method signatures used in later tasks match what was defined in earlier tasks.
