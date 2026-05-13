# B1-p2.3a Scheduler Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the data foundation for continuous batching — `Scheduler`, `RequestState`, `RequestId` structs with `admit`/`evict` API — covered by 10 unit tests and one integration test. No forward, no KV cache integration, no HTTP server changes.

**Architecture:** New module `ironmlx/src/core/scheduler.rs` owns a pre-allocated `Vec<Option<RequestState>>` of length `b_max`. `admit` walks the vec for the first `None` slot, fills it with a freshly-constructed `RequestState` (cloning the sampler, recording `row_idx` and `real_len`), and returns a monotonically increasing `RequestId(u64)`. `evict` walks for the matching id and replaces with `None`. ID values are never reused after eviction.

**Tech Stack:** Rust, ironmlx existing core types (`GenerateRequest`, `Sampler`). No new external deps.

---

## File Structure

```
ironmlx/src/core/scheduler.rs                       — NEW (Scheduler / RequestState / RequestId + 10 inline unit tests)
ironmlx/src/core/mod.rs                              — ADD `pub mod scheduler;` + re-exports
ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs         — NEW integration test
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3a_closeout/report.md                      — NEW close-out
```

Zero modifications to: `models/`, `core/server/`, `core/generate.rs`, `core/cache/`, `core/sampler.rs`, `core/tokenizer.rs`, `nn/`, `tests/p6_*`, `tests/b1_p2_*` (the new test gets its own file).

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-b1-p2-3-continuous-batching`, HEAD at `3db4b7e` ("docs(b1-p2.3a): scheduler skeleton design spec"). No staged or unstaged changes (only `design.md` in repo root is an allowed stray).

---

## Task 1: Create `core/scheduler.rs` with `Scheduler` + `RequestState` + `RequestId` + 10 unit tests

**Files:**
- Create: `ironmlx/src/core/scheduler.rs`
- Modify: `ironmlx/src/core/mod.rs` (add `pub mod scheduler;` + re-exports)

- [ ] **Step 1.1: Create the new module file**

Create `ironmlx/src/core/scheduler.rs` with the following content. This single file holds: the `RequestId` newtype, the `RequestState` struct, the `Scheduler` struct, and 10 inline unit tests.

```rust
//! B1-p2.3a scheduler skeleton — per-request state + fixed-capacity admit/evict.
//!
//! Subsequent sub-phases extend this module:
//! - B1-p2.3b adds `Scheduler::step()` driving `model.forward_on([B, 1], ...)`
//!   and the HTTP server refactor.
//! - B1-p2.3c adds per-row KV cache offset tracking + per-row decode mask.
//! - B1-p2.3d adds an admission queue + preemption when `b_max` is full.
//! - B1-p2.3e adds per-row sampler invocation (temperature/top_k per row).
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md`.

use anyhow::{anyhow, Result};

use crate::core::generate::GenerateRequest;
use crate::core::sampler::Sampler;

/// Opaque, monotonically-increasing identifier for an admitted request.
///
/// Never reused after the request is evicted — admitting another request into
/// the same `row_idx` produces a new `RequestId` value. This eliminates
/// stale-id bugs at the cost of a 64-bit counter (~10^19 IDs before overflow;
/// practically infinite).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// All per-request state the scheduler tracks. Pre-allocated at admit time
/// and held until eviction.
///
/// Fields are chosen to cover B1-p2.3b–3e needs without a later refactor.
/// VL fields (`pixel_values`, `image_grid_thw`, etc.) are intentionally
/// omitted from 3a — they get added in B1-p2.4 when VL B>1 lands.
#[derive(Debug)]
pub struct RequestState {
    /// Opaque token returned by [`Scheduler::admit`].
    pub id: RequestId,
    /// Position in the scheduler's slot vector (0..b_max). Fixed for the
    /// lifetime of this request — subsequent admits never relocate it.
    /// Used by 3b to index into the batched KV cache and per-row mask
    /// tensors.
    pub row_idx: usize,
    /// Original prompt token ids; copied from `GenerateRequest::prompt_ids`.
    pub prompt_ids: Vec<u32>,
    /// Decode-time tokens produced so far. Empty at admit. 3b pushes one
    /// token per `Scheduler::step()` per row.
    pub generated_tokens: Vec<u32>,
    /// Hard cap on tokens generated beyond the prompt.
    pub max_new_tokens: usize,
    /// Token ids that terminate the stream when produced; copied from
    /// `GenerateRequest::stop_token_ids`.
    pub stop_token_ids: Vec<u32>,
    /// Per-row sampler — cloned from the request's sampler at admit time so
    /// each row owns independent sampler state (the `Cell` inside `Sampler`
    /// requires per-row independence — see `core/sampler.rs:43`).
    pub sampler: Sampler,
    /// Effective KV-cache length for this row: starts at `prompt_ids.len()`
    /// and is incremented by 1 per decode step (3b). Used by 3c to build
    /// the per-row decode mask.
    pub real_len: i32,
    /// `false` at admit; 3b sets `true` on EOS / `max_new_tokens` reached.
    pub finished: bool,
    /// `"stop"` or `"length"` when `finished` is `true`; otherwise `None`.
    pub finish_reason: Option<&'static str>,
}

/// Fixed-capacity scheduler holding up to `b_max` in-flight requests.
///
/// 3a is single-threaded only — no `Send + Sync` impls. A later sub-phase
/// will decide whether to run the scheduler on the main runtime thread or
/// in `tokio::spawn_blocking`.
#[derive(Debug)]
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
}

impl Scheduler {
    /// Construct a scheduler with `b_max` pre-allocated slots, all `None`.
    pub fn new(b_max: usize) -> Self {
        let mut slots = Vec::with_capacity(b_max);
        for _ in 0..b_max {
            slots.push(None);
        }
        Self {
            b_max,
            slots,
            next_id: 0,
        }
    }

    /// Maximum concurrent in-flight requests this scheduler can hold.
    pub fn b_max(&self) -> usize {
        self.b_max
    }

    /// Admit a new request. Walks `slots` for the first `None`, fills it
    /// with a freshly-constructed `RequestState`, and returns a new
    /// monotonically-increasing [`RequestId`]. Returns `Err` if the
    /// scheduler is full (B1-p2.3d will replace this with queueing).
    ///
    /// The request's sampler is **cloned** so each row has its own
    /// independent sampler state.
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

        let real_len = req.prompt_ids.len() as i32;
        let state = RequestState {
            id,
            row_idx,
            prompt_ids: req.prompt_ids,
            generated_tokens: Vec::new(),
            max_new_tokens: req.max_new_tokens,
            stop_token_ids: req.stop_token_ids,
            sampler: req.sampler,
            real_len,
            finished: false,
            finish_reason: None,
        };
        self.slots[row_idx] = Some(state);
        Ok(id)
    }

    /// Evict an in-flight request, freeing its slot for reuse. The slot
    /// index is freed but the [`RequestId`] is **never** reissued (the
    /// counter keeps incrementing).
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        Ok(())
    }

    /// Number of occupied slots.
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Borrow every occupied slot's `RequestState`, in slot order.
    pub fn active(&self) -> Vec<&RequestState> {
        self.slots.iter().filter_map(|s| s.as_ref()).collect()
    }

    /// Look up by id. `None` if the id was never admitted or has been evicted.
    pub fn get(&self, id: RequestId) -> Option<&RequestState> {
        self.slots
            .iter()
            .find_map(|s| s.as_ref().filter(|r| r.id == id))
    }

    /// Mutable lookup by id.
    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut RequestState> {
        self.slots
            .iter_mut()
            .find_map(|s| s.as_mut().filter(|r| r.id == id))
    }

    /// `row_idx` of every occupied slot, in slot order. Used by 3b to
    /// build batched inputs.
    pub fn occupied_rows(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| s.as_ref().map(|_| idx))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal `GenerateRequest` for tests. Uses
    /// `Sampler::greedy()` and an arbitrary 4-token prompt unless overridden.
    fn mk_req(prompt_ids: Vec<u32>) -> GenerateRequest {
        GenerateRequest {
            prompt_ids,
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
        }
    }

    #[test]
    fn scheduler_new_empty() {
        let s = Scheduler::new(4);
        assert_eq!(s.b_max(), 4);
        assert_eq!(s.active_count(), 0);
        assert!(s.active().is_empty());
        assert!(s.occupied_rows().is_empty());
    }

    #[test]
    fn admit_happy_path() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1, 2, 3, 4])).expect("admit");
        assert_eq!(id, RequestId(0));
        assert_eq!(s.active_count(), 1);
        let state = s.get(id).expect("get");
        assert_eq!(state.row_idx, 0);
        assert_eq!(state.real_len, 4);
        assert_eq!(state.prompt_ids, vec![1, 2, 3, 4]);
        assert!(state.generated_tokens.is_empty());
        assert!(!state.finished);
        assert!(state.finish_reason.is_none());
    }

    #[test]
    fn admit_assigns_distinct_rows() {
        let mut s = Scheduler::new(4);
        let ids: Vec<_> = (0..4)
            .map(|i| s.admit(mk_req(vec![i as u32])).expect("admit"))
            .collect();
        let rows: Vec<usize> = ids.iter().map(|id| s.get(*id).unwrap().row_idx).collect();
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(s.active_count(), 4);
    }

    #[test]
    fn evict_releases_row() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.active_count(), 1);
        s.evict(id).expect("evict");
        assert_eq!(s.active_count(), 0);
        assert!(s.get(id).is_none());
    }

    #[test]
    fn admit_after_evict_reuses_row() {
        let mut s = Scheduler::new(4);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        assert_eq!(s.get(id_a).unwrap().row_idx, 0);
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert_eq!(s.get(id_b).unwrap().row_idx, 0); // same slot
        assert_ne!(id_a, id_b); // distinct id
    }

    #[test]
    fn admit_full_returns_err() {
        let mut s = Scheduler::new(2);
        s.admit(mk_req(vec![1])).expect("admit 0");
        s.admit(mk_req(vec![2])).expect("admit 1");
        let err = s.admit(mk_req(vec![3])).expect_err("admit full");
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler full"),
            "unexpected err: {msg}"
        );
        assert!(msg.contains("b_max=2"), "missing b_max in err: {msg}");
    }

    #[test]
    fn evict_unknown_id_returns_err() {
        let mut s = Scheduler::new(2);
        let err = s.evict(RequestId(42)).expect_err("evict unknown");
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn id_monotonic_after_evict() {
        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert!(id_b.0 > id_a.0, "next id should be > previous: {:?} vs {:?}", id_b, id_a);
    }

    #[test]
    fn sampler_cloned_per_request() {
        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");

        // The two `RequestState`s must hold distinct Sampler instances
        // (separately addressable in memory). Probe via pointer identity
        // of references — if Sampler shared interior state via Arc it
        // would still produce different & references, but for ironmlx's
        // Sampler the clone is value-copy of the configuration plus a
        // fresh `Cell<Option<Array>>`, so this is the right invariant.
        let p_a: *const Sampler = &s.get(id_a).unwrap().sampler;
        let p_b: *const Sampler = &s.get(id_b).unwrap().sampler;
        assert_ne!(p_a, p_b);
    }

    #[test]
    fn occupied_rows_reflects_state() {
        let mut s = Scheduler::new(4);
        let id_0 = s.admit(mk_req(vec![1])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![2])).expect("admit 1");
        let id_2 = s.admit(mk_req(vec![3])).expect("admit 2");
        assert_eq!(s.occupied_rows(), vec![0, 1, 2]);
        s.evict(id_1).expect("evict 1");
        assert_eq!(s.occupied_rows(), vec![0, 2]);
        // Silence unused id warnings.
        let _ = (id_0, id_2);
    }
}
```

- [ ] **Step 1.2: Register the module in `core/mod.rs`**

```bash
grep -n "^pub mod\|^pub use" /Volumes/Dev/cxx-mlx/ironmlx/src/core/mod.rs | head -10
```

Expected: lines 3-9 declare `pub mod cache; chat_template; generate; loader; sampler; server; tokenizer;` and lines 11-16 re-export.

Add `scheduler` to both blocks. The final state of those line ranges should read:

```rust
pub mod cache;
pub mod chat_template;
pub mod generate;
pub mod loader;
pub mod sampler;
pub mod scheduler;
pub mod server;
pub mod tokenizer;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use generate::{build_position_ids, GenerateEvent, GenerateRequest, GenerationStream};
pub use loader::{EosTokenId, Loader, QuantMeta, QuantMode, TokenizerConfig};
pub use sampler::Sampler;
pub use scheduler::{RequestId, RequestState, Scheduler};
pub use tokenizer::Tokenizer;
```

Use `Edit` with `replace_all=false` and an `old_string`/`new_string` pair that includes enough surrounding context (2-3 adjacent declaration lines) to uniquely locate the insertion point. The two added lines are `pub mod scheduler;` (between `sampler;` and `server;` to keep alphabetical order) and `pub use scheduler::{RequestId, RequestState, Scheduler};` (between `sampler::Sampler;` and `tokenizer::Tokenizer;`).

- [ ] **Step 1.3: Build + fmt + clippy**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected:
- fmt: clean (no output / no diff)
- build: `Finished release profile ...`
- clippy: clean (only unchanged mlx-sys C++ warnings)

If fmt forces a reformat, accept the rewrite — no semantic change.

- [ ] **Step 1.4: Run the 10 unit tests**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release scheduler:: 2>&1 | tail -15
```

Expected: **10 passed / 0 failed**. The tests are:
1. `scheduler_new_empty`
2. `admit_happy_path`
3. `admit_assigns_distinct_rows`
4. `evict_releases_row`
5. `admit_after_evict_reuses_row`
6. `admit_full_returns_err`
7. `evict_unknown_id_returns_err`
8. `id_monotonic_after_evict`
9. `sampler_cloned_per_request`
10. `occupied_rows_reflects_state`

- [ ] **Step 1.5: Full lib regression**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: **174 passed** (B1-p2.2 baseline 164 + 10 new scheduler tests).

- [ ] **Step 1.6: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/scheduler.rs ironmlx/src/core/mod.rs
git commit -m "feat(b1-p2.3a): add scheduler skeleton + 10 unit tests"
```

---

## Task 2: Integration test `b1_p2_3a_scheduler_skeleton.rs`

**Files:**
- Create: `ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs`

- [ ] **Step 2.1: Write the integration test**

Create `ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs`:

```rust
//! B1-p2.3a scheduler skeleton — end-to-end admit/evict integration test.
//!
//! No model load, no GPU work. Exercises the scheduler API across a
//! realistic admit / evict / re-admit sequence to verify the skeleton
//! is internally consistent before B1-p2.3b layers the forward pass on
//! top of it.
//!
//! Run with:
//!   cargo test -p ironmlx --release --test b1_p2_3a_scheduler_skeleton

use ironmlx::core::generate::GenerateRequest;
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{RequestId, Scheduler};

/// Build a minimal `GenerateRequest` for a synthetic prompt of length `n`.
fn mk_req(seed: u32, n: usize) -> GenerateRequest {
    let prompt: Vec<u32> = (0..n as u32).map(|i| seed.wrapping_add(i)).collect();
    GenerateRequest {
        prompt_ids: prompt,
        max_new_tokens: 32,
        sampler: Sampler::greedy(),
        stop_token_ids: vec![2],
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

#[test]
fn b1_p2_3a_admit_evict_sequence() {
    let mut s = Scheduler::new(4);

    // 1. Admit 4 mock requests; verify monotonic ids 0..3 and row_idx 0..3.
    let mut ids: Vec<RequestId> = Vec::new();
    for i in 0..4 {
        let id = s
            .admit(mk_req(100 * (i + 1) as u32, 8 + i))
            .expect("admit");
        let state = s.get(id).expect("get");
        assert_eq!(state.row_idx, i as usize);
        assert_eq!(state.real_len, (8 + i) as i32);
        assert_eq!(state.prompt_ids.len(), 8 + i);
        ids.push(id);
    }
    assert_eq!(s.active_count(), 4);
    assert_eq!(s.occupied_rows(), vec![0, 1, 2, 3]);

    // 2. Evict id=1 and id=3.
    s.evict(ids[1]).expect("evict 1");
    s.evict(ids[3]).expect("evict 3");
    assert_eq!(s.active_count(), 2);
    assert_eq!(s.occupied_rows(), vec![0, 2]);
    let actives = s.active();
    assert_eq!(actives.len(), 2);
    let rows: Vec<usize> = actives.iter().map(|r| r.row_idx).collect();
    assert_eq!(rows, vec![0, 2]);

    // 3. Admit a fifth request; verify it reuses row 1.
    let id_5 = s.admit(mk_req(500, 12)).expect("admit 5");
    assert_eq!(s.get(id_5).unwrap().row_idx, 1);
    assert!(id_5.0 > ids[3].0); // monotonic across evict

    // 4. Admit a sixth; verify it reuses row 3.
    let id_6 = s.admit(mk_req(600, 14)).expect("admit 6");
    assert_eq!(s.get(id_6).unwrap().row_idx, 3);
    assert_eq!(s.active_count(), 4);

    // 5. Admit a seventh; verify Err (b_max full).
    let err = s.admit(mk_req(700, 16)).expect_err("admit 7 must fail");
    let msg = format!("{err}");
    assert!(msg.contains("scheduler full"), "unexpected err: {msg}");
    assert!(msg.contains("b_max=4"), "missing b_max in err: {msg}");

    // 6. Evicting an already-evicted id returns Err.
    let err = s.evict(ids[1]).expect_err("evict id=1 again must fail");
    assert!(format!("{err}").contains("not found"));

    // 7. Final state: 4 active rows, distinct ids.
    assert_eq!(s.active_count(), 4);
    let final_ids: Vec<u64> = s.active().iter().map(|r| r.id.0).collect();
    // ids[0]=0, ids[2]=2, id_5=4, id_6=5 (counter is monotonic; 1 and 3 were skipped at evict
    // but next_id keeps advancing on admit, not at evict — so after admits 0..4 next_id=4,
    // then admit 5 → id=4, admit 6 → id=5).
    assert_eq!(final_ids, vec![0, 2, 4, 5]);
}
```

- [ ] **Step 2.2: Build the integration test**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3a_scheduler_skeleton 2>&1 | tail -3
```

Expected: fmt clean, build clean.

- [ ] **Step 2.3: Run the integration test**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --release --test b1_p2_3a_scheduler_skeleton 2>&1 | tail -5
```

Expected: **1 passed / 0 failed** (`b1_p2_3a_admit_evict_sequence`). Runtime well under 1 second — no model load, no GPU.

- [ ] **Step 2.4: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs
git commit -m "test(b1-p2.3a): integration test for admit/evict sequence"
```

---

## Task 3: Regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3a_closeout/report.md`

- [ ] **Step 3.1: Full hygiene sweep**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected all green:
- fmt: clean
- clippy: clean (only unchanged mlx-sys C++ warnings)
- build: `Finished release profile ...`
- lib tests: **174 passed / 0 failed**

- [ ] **Step 3.2: P6.3 single-image regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Use `run_in_background: true` then Monitor on the test PID exit; timeout ~600000 ms. Expected: PASS, `max_diff=0.3906`, `first_token=760`.

- [ ] **Step 3.3: P6.6 logits-match regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS unchanged (first_token=760).

- [ ] **Step 3.4: P6.7 chunked-prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored 2>&1 | tail -5
```

Use timeout ~1500000 ms. Expected: PASS, all 3 chunk_sizes → 760.

- [ ] **Step 3.5: B1-p2.1 prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored 2>&1 | tail -5
```

Use timeout ~1500000 ms. Expected: PASS — 10/12 argmax bit-id, max_diff ≤ 0.19.

- [ ] **Step 3.6: B1-p2.2 batched decode regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored 2>&1 | tail -5
```

Use timeout ~1500000 ms. Expected: PASS — 57/60 argmax bit-id, decode max_diff ≤ 1.62.

- [ ] **Step 3.7: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3a_closeout/report.md`:

```markdown
# B1-p2.3a Scheduler Skeleton — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off `ironmlx-b1-p2-2-batched-decode` head `1ed51dc`)
**Date:** 2026-05-13
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md` (commit `3db4b7e`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md`

## Summary

Added the scheduler data foundation for B1-p2.3 continuous batching.
New `core::scheduler` module with `Scheduler`, `RequestState`, and
`RequestId` types plus `admit`/`evict`/`active`/`get`/`get_mut`/
`occupied_rows` API, covered by 10 unit tests + 1 integration test.
Zero touches to model/server/cache/generate — purely additive.

Subsequent sub-phases extend this skeleton: 3b adds `Scheduler::step()`
+ HTTP refactor; 3c per-row KV cache offsets; 3d admission queue /
preemption; 3e per-row sampler invocation.

## Acceptance

| Test | Result |
| --- | --- |
| `scheduler::tests::scheduler_new_empty` | ✅ |
| `scheduler::tests::admit_happy_path` | ✅ |
| `scheduler::tests::admit_assigns_distinct_rows` | ✅ |
| `scheduler::tests::evict_releases_row` | ✅ |
| `scheduler::tests::admit_after_evict_reuses_row` | ✅ |
| `scheduler::tests::admit_full_returns_err` | ✅ |
| `scheduler::tests::evict_unknown_id_returns_err` | ✅ |
| `scheduler::tests::id_monotonic_after_evict` | ✅ |
| `scheduler::tests::sampler_cloned_per_request` | ✅ |
| `scheduler::tests::occupied_rows_reflects_state` | ✅ |
| `b1_p2_3a_admit_evict_sequence` (integration) | ✅ |

## Architectural Changes

1. **New module `ironmlx/src/core/scheduler.rs`** (~280 lines including tests)
   - `RequestId(u64)` — opaque, monotonic, never reused.
   - `RequestState` — per-request state owned by the scheduler.
     Fields: id, row_idx, prompt_ids, generated_tokens, max_new_tokens,
     stop_token_ids, sampler (cloned), real_len, finished, finish_reason.
     VL fields deferred to B1-p2.4.
   - `Scheduler` — fixed-capacity `Vec<Option<RequestState>>` of length
     `b_max`. `admit` linear-scans for first `None`; `evict` linear-scans
     by id. ID counter advances on admit only.
2. **`ironmlx/src/core/mod.rs`** — added `pub mod scheduler;` and
   re-exports of `RequestId`, `RequestState`, `Scheduler`.

No changes to: `models/`, `core/server/`, `core/generate.rs`,
`core/cache/`, `core/sampler.rs`, `core/tokenizer.rs`, `nn/`.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `<sha>` | feat | Scheduler skeleton + 10 unit tests |
| `<sha>` | test | Integration test for admit/evict sequence |
| `<sha>` | docs | This close-out |

(Fill in `<sha>` from `git log --oneline 3db4b7e..HEAD`.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **174 passed / 0 failed** (B1-p2.2 baseline 164 + 10 new) |
| P6.3 single-image | PASS, `max_diff=0.3906`, `first_token=760` |
| P6.6 logits-match | PASS, first_token=760 |
| P6.7 chunked-prefill matrix | PASS |
| B1-p2.1 batched prefill | PASS, 10/12 argmax bit-id, max_diff ≤ 0.19 |
| B1-p2.2 batched decode | PASS, 57/60 argmax bit-id, decode max_diff ≤ 1.62 |

## Notes

- **Pre-allocation choice**: `Vec<Option<RequestState>>` with fixed length `b_max` avoids reallocation churn and lets `row_idx` stay stable across the request's lifetime. Subsequent sub-phases rely on the row_idx being a fixed slot index into a batched KV cache.
- **Monotonic ID, no reuse**: prevents a class of stale-id bugs that would otherwise emerge when a slot is reused immediately after eviction. The cost is a 64-bit counter (practically infinite headroom).
- **Single-threaded**: Scheduler is not `Send + Sync`. 3b will choose between running it on the main runtime thread or in `spawn_blocking`. Either way it stays single-owner.
- **Sampler cloning**: Sampler holds a `Cell<Option<Array>>` for the pre-dispatched async-greedy state (sampler.rs:43). Per-row independence requires per-row clones — verified by the `sampler_cloned_per_request` test.

## B1-p2.3x Next Steps

- **B1-p2.3b** — Add `Scheduler::step(model: &Qwen35Model, cache: &mut [LayerCache], target) -> Result<Vec<StepEvent>>` that packs all `active()` rows' input tokens into a `[B, 1]` tensor, calls `model.forward_on`, samples per row, updates each `RequestState`, and returns per-row events. HTTP server (OpenAI handler) refactored to drive the scheduler instead of `GenerationStream`.
- **B1-p2.3c** — Per-row KV cache offset tracking + per-row decode mask.
- **B1-p2.3d** — Admission queue + preemption when `b_max` is full.
- **B1-p2.3e** — Per-row sampler invocation (temperature, top_k, penalties per row).

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md`
- New module: `ironmlx/src/core/scheduler.rs`
- Integration test: `ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs`
```

Replace `<sha>` placeholders with actual commit SHAs from `git log --oneline 3db4b7e..HEAD` after the prior two commits land.

- [ ] **Step 3.8: Commit close-out**

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3a_closeout/report.md
git commit -m "docs(b1-p2.3a): close-out — scheduler skeleton + 10 unit tests + integration test"
```

- [ ] **Step 3.9: Final summary log**

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline 3db4b7e..HEAD
```

Expected: 3 commits (spec was at `3db4b7e`, then 1 implementation + 1 test + 1 close-out).

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §2 Goal: new module `core/scheduler.rs` | Task 1 Step 1.1 |
| §2 Goal: `Scheduler`, `RequestState`, `RequestId` types | Task 1 Step 1.1 |
| §2 Goal: `admit`/`evict`/`active`/`active_count`/`get`/`get_mut`/`occupied_rows` API | Task 1 Step 1.1 |
| §2 Goal: 10 unit tests | Task 1 Step 1.1 (inline mod tests, listed at Step 1.4) |
| §2 Goal: integration test | Task 2 |
| §2 Goal: zero src regressions | Task 3 Steps 3.1–3.6 |
| §4.1 RequestState field list | Task 1 Step 1.1 RequestState struct (matches exactly: id, row_idx, prompt_ids, generated_tokens, max_new_tokens, stop_token_ids, sampler, real_len, finished, finish_reason) |
| §4.2 Scheduler internals (b_max, slots Vec<Option<…>> length=b_max, next_id) | Task 1 Step 1.1 Scheduler struct |
| §4.3 admit/evict semantics (Err on full, Err on unknown id, no ID reuse) | Task 1 Step 1.1 admit + evict bodies |
| §4.4 File layout | Task 1 (scheduler.rs + mod.rs), Task 2 (integration test), Task 3 (close-out) |
| §5.1 10 unit tests by name | Task 1 Step 1.4 (names match: scheduler_new_empty, admit_happy_path, admit_assigns_distinct_rows, evict_releases_row, admit_after_evict_reuses_row, admit_full_returns_err, evict_unknown_id_returns_err, id_monotonic_after_evict, sampler_cloned_per_request, occupied_rows_reflects_state) |
| §5.2 Integration test 7-step scenario | Task 2 Step 2.1 (7-step body covers admit 4 / evict 1+3 / re-admit / overflow / double-evict-err) |
| §5.3 Regression gates | Task 3 Steps 3.1–3.6 |

All spec sections have a corresponding task. No gaps.

**2. Placeholder scan:**

- Task 3 Step 3.7 close-out template contains `<sha>` placeholders (filled at execution time from `git log`) — marked explicitly.
- No "TBD", "implement later", "fill in details" elsewhere.

**3. Type consistency:**

| Symbol | First defined | Reused |
| --- | --- | --- |
| `RequestId(u64)` newtype, `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` | Task 1 Step 1.1 | Task 1 unit tests, Task 2 integration test |
| `RequestState { id, row_idx, prompt_ids, generated_tokens, max_new_tokens, stop_token_ids, sampler, real_len, finished, finish_reason }` | Task 1 Step 1.1 | Task 1 unit tests (`s.get(id).unwrap().row_idx`, `real_len`), Task 2 integration test |
| `Scheduler::{new, b_max, admit, evict, active_count, active, get, get_mut, occupied_rows}` | Task 1 Step 1.1 | Task 1 unit tests, Task 2 integration test |
| `mk_req(prompt_ids: Vec<u32>) -> GenerateRequest` helper (inline tests) | Task 1 Step 1.1 unit tests | Internal to unit tests |
| `mk_req(seed: u32, n: usize) -> GenerateRequest` helper (integration test) | Task 2 Step 2.1 | Internal to integration test |

All names consistent across tasks. The two `mk_req` helpers have different signatures intentionally — the unit tests' takes an explicit prompt, the integration test's generates a synthetic one from a seed (so each row has a distinct prompt). They are in separate scopes so there is no collision.
