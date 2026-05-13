# B1-p2.3a Scheduler Skeleton — Design

**Status:** Approved (brainstormed 2026-05-13)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving / phase 3 continuous batching
**Branch target:** `ironmlx-b1-p2-3-continuous-batching` (cut from `ironmlx-b1-p2-2-batched-decode` head `1ed51dc`)

## 0. Program context — B1-p2.3 sub-phase decomposition

B1-p2.3 (continuous batching) is a multi-subsystem effort decomposed into
five sub-phases by Boss decision:

| Sub-phase | Scope | This doc |
| --- | --- | --- |
| **B1-p2.3a** | Scheduler skeleton + per-request state struct; admit/evict API; unit tests only — **no forward** | ✅ |
| B1-p2.3b | Token-loop unification: `Scheduler::step()` packs B histories → batched forward → per-row sample → update state. HTTP server refactored. |  |
| B1-p2.3c | Per-row KV cache offset tracking + per-row decode mask. |  |
| B1-p2.3d | Admission control + eviction queue / preemption. |  |
| B1-p2.3e | Per-row sampler scaffolding (temperature, top_k, penalties per row). |  |

Each sub-phase ships working software (3a: passing unit tests of an
isolated data structure; 3b: end-to-end batched serving through the new
scheduler; etc.). This doc covers **3a only**.

## 1. Motivation

ironmlx currently has zero scheduler infrastructure (recon at branch
cut: no `pub struct Scheduler`, no `pub mod scheduler`, no `pub fn
admit`/`evict`). The HTTP server today spawns one tokio task per
request and acquires a `Mutex<Qwen35Model>` for the duration of that
request's token loop (`server/openai.rs:372-430`); concurrent requests
serialise on the mutex. To unlock batched serving, the codebase needs
a single-threaded scheduler that owns per-request state for up to
`B_max` in-flight requests and demultiplexes one model forward per
decode step.

B1-p2.3a's purpose is to build the **data foundation** — `Scheduler`,
`RequestState`, `RequestId`, the admit/evict API — without yet touching
the forward path or HTTP server. Subsequent sub-phases extend it.

## 2. Goals

- New module `ironmlx/src/core/scheduler.rs` containing:
  - `pub struct Scheduler { b_max, slots, next_id }` with fixed-capacity
    pre-allocated `Vec<Option<RequestState>>` of length `b_max`.
  - `pub struct RequestState` carrying all per-request state needed by
    later sub-phases (3b token-loop, 3c cache offsets, 3e sampler).
  - `pub struct RequestId(u64)` — opaque newtype, monotonically
    increasing across admit calls (never reused after evict).
  - `Scheduler::new(b_max: usize) -> Self`
  - `Scheduler::b_max(&self) -> usize`
  - `Scheduler::admit(&mut self, req: GenerateRequest) -> Result<RequestId>`
  - `Scheduler::evict(&mut self, id: RequestId) -> Result<()>`
  - `Scheduler::active_count(&self) -> usize`
  - `Scheduler::active(&self) -> Vec<&RequestState>`
  - `Scheduler::get(&self, id: RequestId) -> Option<&RequestState>`
  - `Scheduler::get_mut(&mut self, id: RequestId) -> Option<&mut RequestState>`
  - `Scheduler::occupied_rows(&self) -> Vec<usize>` — returns the
    `row_idx` of every occupied slot (used by 3b to build batched inputs).
- 8+ unit tests covering admit/evict happy paths + edge cases.
- One integration test driving an admit/evict sequence (no forward).
- Zero regressions on existing src — module is purely additive.

## 3. Non-goals

- Token-loop dispatch (`Scheduler::step()`) — B1-p2.3b
- Model forward integration — B1-p2.3b
- KV cache hand-off / per-row offset tracking — B1-p2.3c
- Admission queue / preemption when full — B1-p2.3d (3a returns Err)
- Per-row sampler invocation (temperature, top_k, penalties) — B1-p2.3e
- HTTP server refactor — B1-p2.3b starts the integration
- VL B>1 (pixel_values per request) — B1-p2.4
- Throughput benchmarking — B1-p2.5

## 4. Architecture

### 4.1 `RequestState`

```rust
pub struct RequestState {
    pub id: RequestId,                       // opaque token returned by admit
    pub row_idx: usize,                      // 0..b_max; fixed for the lifetime of this request
    pub prompt_ids: Vec<u32>,                // original prompt; copied from GenerateRequest
    pub generated_tokens: Vec<u32>,          // empty at admit; pushed by 3b as decode runs
    pub max_new_tokens: usize,               // copied from GenerateRequest
    pub stop_token_ids: Vec<u32>,            // copied from GenerateRequest
    pub sampler: Sampler,                    // cloned at admit (per-row independent sampler state)
    pub real_len: i32,                       // = prompt_ids.len() at admit; 3b advances by 1 per decode step
    pub finished: bool,                      // false at admit; 3b sets true on EOS / max_new_tokens
    pub finish_reason: Option<&'static str>, // "stop" or "length" when finished
}
```

VL fields (`pixel_values`, `image_grid_thw`, `image_spatial_merge_size`,
`image_token_id`) are **intentionally omitted** in 3a — they get added
in B1-p2.4 when the VL B>1 path is built. The `GenerateRequest` source
struct will continue to carry them; the scheduler simply ignores them
in 3a.

### 4.2 `Scheduler`

```rust
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>, // length == b_max, pre-allocated
    next_id: u64,
}
```

Construction pre-allocates `b_max` `None` slots so that `admit` never
needs to grow the vector and `row_idx` is stable for the request's
lifetime.

`admit` walks `slots` for the first `None`, fills it, and bumps
`next_id`. Returns `Err("scheduler full: no row available (b_max={...})")` if all
slots are `Some`.

`evict` walks `slots` for the matching `id` and replaces it with
`None`. Returns `Err("request id {n} not found")` otherwise.

### 4.3 `RequestId`

```rust
pub struct RequestId(u64);
```

Newtype, monotonically increasing, **never reused** after eviction.
Eliminates a class of bugs where a stale ID could refer to a different
request after slot reuse.

### 4.4 File layout

```
ironmlx/src/core/scheduler.rs                       — NEW (Scheduler/RequestState/RequestId + inline tests)
ironmlx/src/core/mod.rs                              — add `pub mod scheduler;` + re-exports
ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs         — NEW integration test
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3a_closeout/report.md                      — NEW close-out
```

No changes to: `models/`, `core/server/`, `core/generate.rs`,
`core/cache/`, `core/sampler.rs`, `core/tokenizer.rs`, or anywhere else.

## 5. Acceptance

### 5.1 Unit tests (inline `#[cfg(test)] mod` in `core/scheduler.rs`)

| Test | Verifies |
| --- | --- |
| `scheduler_new_empty` | `b_max=4` → `active_count == 0`, `b_max() == 4` |
| `admit_happy_path` | Single admit → `Ok(id_0)`, `active_count == 1`, `row_idx == 0`, `real_len == prompt_ids.len()` |
| `admit_assigns_distinct_rows` | 4 admits → row_idx 0, 1, 2, 3 (in admit order); active_count == 4 |
| `evict_releases_row` | admit + evict → `active_count == 0`; same slot reset to `None` |
| `admit_after_evict_reuses_row` | admit, evict row 0, admit again → new request's row_idx == 0 |
| `admit_full_returns_err` | `b_max=2`, three admits → third returns `Err(... no row available ...)` |
| `evict_unknown_id_returns_err` | evict on fresh `RequestId(42)` → `Err(... not found ...)` |
| `id_monotonic_after_evict` | admit (id=0), evict id=0, admit again → new id != 0 |
| `sampler_cloned_per_request` | admit twice with the same `Sampler` template → each `RequestState` owns its own clone (mutate one, the other unchanged) |
| `occupied_rows_reflects_state` | admit 0,1,2; evict 1; → `occupied_rows() == [0, 2]` |

### 5.2 Integration test (`tests/b1_p2_3a_scheduler_skeleton.rs`)

One test driving an end-to-end admit/evict sequence:
1. Create `Scheduler::new(4)`.
2. Admit 4 mock requests (no forward); verify ids increase, row_idx 0–3.
3. Evict id=1 and id=3.
4. Verify `active()` returns 2 RequestStates with row_idx ∈ {0, 2}.
5. Admit a fifth request; verify it reuses row 1.
6. Admit a sixth; verify it reuses row 3.
7. Admit a seventh; verify `Err`.

This integration test exercises slot reuse + ID monotonicity end-to-end.

### 5.3 Regression gates

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release -- --test-threads=1` | **≥ 174 passed** (B1-p2.2 baseline 164 + 10 new scheduler tests) |
| P6.3 Task 21 single-image logits-match | PASS unchanged |
| P6.6 logits-match | PASS unchanged |
| P6.7 chunked-prefill matrix | PASS unchanged |
| B1-p2.1 prefill matrix | PASS unchanged (10/12 argmax bit-id, max_diff ≤ 0.19) |
| B1-p2.2 decode matrix | PASS unchanged (57/60 argmax bit-id, decode max_diff ≤ 1.62) |
| B1-p2.3a integration test | PASS |

3a adds a new module; nothing existing is touched. All existing regressions are bit-identical by construction.

## 6. Risks

- **R1 — feature creep**: It is tempting to put 3b-3e functionality
  into `RequestState` upfront (e.g., to "save a future refactor"). The
  spec deliberately defines exactly the fields needed up to and
  including 3e — VL fields (3.4 / B1-p2.4) are the only deferred
  addition. Mitigation: review every PR commit against §4.1's field
  list.

- **R2 — `RequestId` collision under wraparound**: `u64` ID space is
  practically infinite (~10^19); even at 10^9 requests/sec, wraparound
  takes >500 years. No mitigation needed.

- **R3 — Scheduler thread safety**: 3a deliberately does not impl
  `Send + Sync`. Single-threaded use only. Subsequent sub-phases will
  decide whether the scheduler runs on the main runtime thread or in
  spawn_blocking. Mitigation: don't add Send/Sync bounds in 3a.

- **R4 — `Sampler` clone semantics**: `Sampler` has `Clone` via
  `core/sampler.rs:61-77`; the `Cell<Option<Array>>` field is correctly
  cloned (Cell::clone clones the inner value). Per-row independent
  sampler state is preserved. Mitigation: `sampler_cloned_per_request`
  unit test directly probes this invariant.

## 7. Estimate

| Phase | Work | Estimate |
| --- | --- | --- |
| 3a-impl | `core/scheduler.rs` Scheduler/RequestState/RequestId + 10 unit tests | 0.5–1 d |
| 3a-test | `tests/b1_p2_3a_scheduler_skeleton.rs` integration | 0.5 d |
| 3a-closeout | regression sweep + close-out doc | 0.5 d |
| **Total** | | **~1.5–2 working days** |

## 8. Out of scope (deferred)

| Item | Phase |
| --- | --- |
| `Scheduler::step()` driving `model.forward_on([B, 1], ...)` | B1-p2.3b |
| Per-row stop logic / EOS detection / max_new_tokens enforcement | B1-p2.3b |
| KV cache hand-off / per-row offset tracking | B1-p2.3c |
| Per-row decode attention mask (cleaner than B1-p2.2's "zeroed cache + softmax dilution") | B1-p2.3c |
| Admission queue when `b_max` full / preemption policy / fairness | B1-p2.3d |
| Per-row sampler invocation (temperature, top_k, penalties per row) | B1-p2.3e |
| HTTP server / OpenAI handler refactor to drive Scheduler | B1-p2.3b onward |
| VL B>1 (`pixel_values` / `image_grid_thw` per request) | B1-p2.4 |
| Throughput benchmarking, iron-bench v2 contract validation | B1-p2.5 |
