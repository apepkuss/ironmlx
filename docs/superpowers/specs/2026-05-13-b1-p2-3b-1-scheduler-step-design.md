# B1-p2.3b-1 — Scheduler `step()` + lockstep prefill (design)

**Date:** 2026-05-13
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3a head `33ea2df`)
**Predecessor sub-phase:** B1-p2.3a Scheduler Skeleton (closed at commit `33ea2df`)
**Sibling sub-phase:** B1-p2.3b-2 — HTTP server refactor (separate spec; depends on this)
**Successor sub-phases:** 3c (per-row offset / per-row decode mask), 3d (admission queue + preemption), 3e (per-row sampler invocation)

---

## §1 Goals

1. Add `prefill_admitted()`, `step()`, `evict_all()`, `phase()` to `Scheduler` so it can actually drive batched generation against `Qwen35Model`.
2. Define a `Phase` state machine (`Idle → Admitting → Decoding → Finished → Idle`) so the API is impossible to misuse and 3b-2 has clear contract to integrate against.
3. Define a minimal `StepEvent` carried out of `step()` for 3b-2 HTTP SSE to consume.
4. Scheduler persistently holds the batched `Vec<LayerCache>` and reuses it across batches via `reset()`, avoiding GPU memory churn.
5. Per-row sampling uses each `RequestState::sampler` (the per-row clone landed in 3a), preserving correctness vs the single-stream `GenerationStream` baseline.

## §2 Non-goals

- **Per-row KV-cache offset.** Defer to 3c. Lockstep semantics (single global offset per layer) is an accepted constraint for 3b.
- **Admission queue / preemption.** Defer to 3d. `admit()` continues to return `Err` when full.
- **HTTP server changes.** Defer to 3b-2. `openai.rs` / `anthropic.rs` / `generate.rs::GenerationStream` are untouched.
- **Per-row sampler config / batched sampler kernel.** Defer to 3e. Sampling stays a per-row loop calling `Sampler::sample` on a `[vocab]` slice — already supported by `core/sampler.rs`.
- **VL (image) requests.** `RequestState` still has no VL fields (3a punted them to 3b — but it was actually punted to **3b-1 if needed by HTTP test**, otherwise to 3c/3d). For 3b-1 the integration test is text-only; VL request admission stays out.
- **Concurrent / multi-threaded scheduler.** Single-threaded, `!Send`, identical to 3a.

## §3 Background

### 3.1 Where 3a left off
3a (commit `33ea2df`) shipped `Scheduler` + `RequestState` + `RequestId` + 11 tests (10 unit + 1 integration). API surface (verified by exploration):

| Method | Signature |
| --- | --- |
| `new` | `pub fn new(b_max: usize) -> Self` |
| `b_max` | `pub fn b_max(&self) -> usize` |
| `admit` | `pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId>` |
| `evict` | `pub fn evict(&mut self, id: RequestId) -> Result<()>` |
| `active_count` | `pub fn active_count(&self) -> usize` |
| `active` | `pub fn active(&self) -> Vec<&RequestState>` |
| `get` | `pub fn get(&self, id: RequestId) -> Option<&RequestState>` |
| `get_mut` | `pub fn get_mut(&mut self, id: RequestId) -> Option<&mut RequestState>` |
| `occupied_rows` | `pub fn occupied_rows(&self) -> Vec<usize>` |

`RequestState` carries: `id`, `row_idx`, `prompt_ids`, `generated_tokens`, `max_new_tokens`, `stop_token_ids`, `sampler`, `real_len: i32`, `finished`, `finish_reason`.

### 3.2 The lockstep cache constraint
`KVCache` allocation is `[batch, n_kv_heads, T_max, head_dim]` ([`kv_cache.rs:160`](../../ironmlx/src/core/cache/kv_cache.rs)) but `offset: i32` is a single global counter per layer ([`kv_cache.rs:30`](../../ironmlx/src/core/cache/kv_cache.rs#L30)). `update_and_fetch` writes all B rows at the same `offset..offset+new` slab and advances offset by `new`.

Consequence: **every row in a batched cache must share the same `offset` at all times.** This forces 3b into "lockstep" semantics — admit a fixed set, prefill all together, decode in lockstep, then evict all and reset before starting the next batch. Continuous batching (rows joining / leaving at different per-row offsets) is exactly what 3c will lift by adding per-row offset tracking.

### 3.3 Why this sub-phase is scoped tight
B1-p2.3 originally collected five concerns: scheduler API (3a), `step()` wiring (3b), per-row offset (3c), admission queue (3d), per-row sampler config (3e). 3b was further split into 3b-1 (this spec — model-layer wiring + lockstep step) and 3b-2 (HTTP server refactor) so each sub-phase produces a reviewable change in 3–5 working days. 3b-1 ships a fully testable `Scheduler::step()` with an integration test, but does not touch the HTTP server, `GenerationStream`, or any model code.

### 3.4 Cache reset infrastructure (already in tree)
Exploration confirmed:
- [`KVCache::reset(&mut self)`](../../ironmlx/src/core/cache/kv_cache.rs#L80) — sets `offset` to 0, preserves Array alloc. Already implemented.
- [`GatedDeltaCache::reset(&mut self) -> Result<()>`](../../ironmlx/src/core/cache/gated_delta.rs#L114) — same idea, but fallible because it re-zeros recurrent state. Already implemented.
- [`LayerCache` enum](../../ironmlx/src/nn/decoder_layer.rs#L65): `Full(KVCache) | Linear(GatedDeltaCache)`. Has **no `impl` block today** — 3b-1 adds an 8-line `impl LayerCache { pub fn reset(&mut self) -> Result<()> }` that dispatches to the underlying type.

The cache reset story is essentially "wire up a dispatcher and call it from `Scheduler::evict_all`."

### 3.5 Existing batched forward path
The B1-p2.1 / B1-p2.2 work already shipped:
- [`Qwen35Model::batched_prefill(input_ids, position_ids, attention_mask, linear_attention_mask, cache, target) -> Result<Array>`](../../ironmlx/src/models/qwen3_5/model.rs#L290) — accepts `[B, T_max]` input, left-padded; per-row prompt lengths encoded in the masks.
- [`Qwen35Model::forward_on(input_ids, position_ids, cache, target) -> Result<Array>`](../../ironmlx/src/models/qwen3_5/model.rs#L93) — accepts `[B, 1]` for decode, no explicit mask (causal is implicit at S=1).
- Position-id builders: [`build_position_ids_batched`](../../ironmlx/src/core/generate.rs#L214) for prefill, [`build_decode_position_ids`](../../ironmlx/src/core/generate.rs#L323) for decode.

3b-1 reuses these — no new model-side code. The work is entirely on the scheduler side.

## §4 Architecture

### 4.1 Phase state machine

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,        // No rows; cache empty (or absent); evict_all just ran or scheduler just constructed.
    Admitting,   // ≥1 row admitted via admit(); cache empty; prefill_admitted not yet called.
    Decoding,    // prefill_admitted ran; cache holds batched prefill state; step() in progress.
    Finished,    // All active rows have RequestState::finished == true; waiting for evict_all.
}
```

Transitions:

```mermaid
stateDiagram-v2
    [*] --> Idle: Scheduler::new(b_max)
    Idle --> Admitting: admit(req)
    Admitting --> Admitting: admit(req)
    Admitting --> Decoding: prefill_admitted(model)
    Decoding --> Decoding: step(model) — at least 1 row still unfinished
    Decoding --> Finished: step(model) — all rows finished
    Finished --> Idle: evict_all()
    Admitting --> Idle: evict(id) drops last admitted row
```

All transitions outside this diagram return `Err(anyhow!(...))` with a clear message.

### 4.2 `Scheduler` fields (3a baseline + new fields)

```rust
pub struct Scheduler {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
    phase: Phase,                         // NEW — drives state machine
    cache: Option<Vec<LayerCache>>,       // NEW — lazy-allocated on first prefill_admitted, reused via reset
}
```

`cache` is `Option` because:
- `Scheduler::new()` doesn't yet know the model (and hence the layer partition / dtype / capacity).
- First `prefill_admitted()` call materializes via `model.make_cache(b_max as i32, cap, dtype)`. `cap` is hardcoded to **8192** for 3b-1 (matches existing tests; 3c can make it configurable per-row via offset tracking).
- After `evict_all()` the cache stays `Some` but is `reset()` in place.

### 4.3 New methods

```rust
impl Scheduler {
    pub fn phase(&self) -> Phase;
    pub fn prefill_admitted(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>>;
    pub fn step(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>>;
    pub fn evict_all(&mut self) -> Result<()>;
}
```

> **Revision note (2026-05-13, during implementation of Task 3):** `prefill_admitted` originally returned `Result<()>`. While implementing Scenario A (B=2 happy), bit-id parity with `GenerationStream` failed because GS runs in **pipelined mode** by default for greedy sampling — its first `next_token` call returns the prefill argmax and pre-fires `forward([token_0])` for the next call. That puts GS's cache trajectory one step ahead of any scheduler that feeds `last_prompt_token` to step 1. To match GS pipelined trajectory exactly (and let 3b-2 swap GS cleanly), `prefill_admitted` now samples the first token per row from prefill logits and emits a `StepEvent` per row, just like `step()` does. See revised §4.5 + §4.6 below.

### 4.4 `admit` / `evict` phase integration

`admit()` (3a behavior):
- Phase ∈ {`Idle`, `Admitting`}: walks `slots` for first `None`, fills it, returns new `RequestId`, sets `phase = Admitting`.
- Phase ∈ {`Decoding`, `Finished`}: returns `Err(anyhow!("scheduler in {phase:?} phase: cannot admit; call evict_all first"))`.
- Phase == `Idle` AND full: returns 3a's existing `"scheduler full"` error (unreachable when empty since `b_max ≥ 1`, but defensive).

`evict()` (3a behavior + phase handling):
- Phase ∈ {`Admitting`, `Decoding`, `Finished`}: walks for id, sets slot to `None`. If `active_count() == 0` afterward, transitions back to `Idle` (only valid when `Admitting`) — phase rules:
  - `Admitting` → drop last admitted → `Idle` (no cache work; cache is `None` or already empty)
  - `Decoding` → keeps `phase = Decoding` even if active_count drops to 0? **No** — 3b-1 forbids partial evict during Decoding. Returns `Err` unless `phase == Finished` or `Admitting`.
  - `Finished` → fine (caller is cleaning up one-by-one before optional `evict_all`)
- Phase == `Idle`: returns `Err(anyhow!("request id {n} not found"))` (unchanged from 3a — no rows to find).

Why forbid partial evict during `Decoding`? Per §3.2 lockstep, evicting one row mid-decode leaves stale KV in that cache slot at the active global offset. Without per-row offset (3c), there is no way to re-use that slot without invalidating the rest of the batch. The cleanest invariant for 3b-1: **once `Decoding` starts, all rows ride together until they `Finished`, then `evict_all` resets the whole cache.** Future 3d sub-phase will introduce mid-decode preemption with its own cache surgery.

### 4.5 `prefill_admitted(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>>`

Preconditions:
- `phase ∈ {Idle, Admitting}` (if `Idle`: `active_count() == 0`, returns `Err("no admitted requests to prefill")`).
- All admitted rows in `slots` (the ones with `Some(state)`).

Steps:
1. Verify `phase != Decoding && phase != Finished`. Else `Err`.
2. Verify `active_count() >= 1`. Else `Err("no admitted requests to prefill")`.
3. Determine `max_len = max(state.prompt_ids.len() for state in active())`.
4. Build batched `input_ids: [B, max_len]` int32 array. For each row in slot order:
   - If `Some(state)`: left-pad `state.prompt_ids` to `max_len` (pad value `0` matches B1-p2.1 path; pad lives at the start).
   - If `None`: row is filled with pad zeros (the row exists in the cache shape because `b_max` is fixed; this is fine — its KV slot becomes effective-empty and never participates in attention).
5. Build `position_ids: [3, B, max_len]` via `build_position_ids_batched(prompt_lens: &[i32], max_len)` where `prompt_lens[b] = state.prompt_ids.len() as i32` for occupied rows, `0` for None.
6. Build `attention_mask: [B, 1, max_len, max_len]` and `linear_attention_mask: [B, max_len]` exactly as B1-p2.1's batched test does — `core::generate::build_batch_attention_mask` and `core::generate::build_batch_linear_mask` (both already `pub`).
7. Allocate or reuse cache:
   - If `cache.is_none()`: `self.cache = Some(model.make_cache(b_max as i32, 8192, Dtype::Bfloat16))`. (`Dtype::Bfloat16` is hardcoded for 3b-1; see §9 Open Questions #2.)
   - Else: reuse the existing cache. After `evict_all` every layer is already at offset 0 — caller invariant.
8. Call `let logits = model.batched_prefill(&input_ids, &position_ids, &attention_mask, &linear_attention_mask, Some(<cache>), ())?`. Returns `[B, 1, vocab]` — batched_prefill internally collapses each row to its per-row last-prompt-position prediction (see `tests/b1_p2_1_batched_prefill.rs:173` for the empirical assertion). It does **not** return logits for every prefill position.
9. **Sample first token per occupied row from the prefill logits.** For each `b` where `slots[b].is_some()`:
   - Slice `row_logits = logits[b, 0, :]` → `[vocab]`. (Middle dim is length 1; the per-row last-prompt-position selection is already done by `batched_prefill`.)
   - Sample: `let token = state.sampler.sample(&row_logits, &history)?` where `history = &state.prompt_ids` (no generated tokens yet).
   - Push `token` to `state.generated_tokens`.
   - `state.real_len += 1`.
   - Termination check (same order as `step`): EOS first (`state.stop_token_ids.contains(&token)` → `finished = true; finish_reason = Some("stop")`); else `max_new_tokens` (`state.generated_tokens.len() >= state.max_new_tokens` → `finished = true; finish_reason = Some("length")`).
   - Build `StepEvent { id: state.id, token, finish_reason: state.finish_reason }` and append to result.
10. Set `phase`:
    - If every occupied row finished after step 9 (rare — only when every prompt's first token is EOS or `max_new_tokens == 1`): `phase = Finished`.
    - Else: `phase = Decoding`.
11. Return `Ok(events)`.

**Why this matches `GenerationStream` pipelined trajectory:** GS pipelined returns the prefill argmax via its first `next_token` call and pre-fires `forward([token_0])` to populate `pending_token_arr` for next call. After step 9 + 10 here, scheduler's cache and `RequestState` reflect exactly the same state GS reaches after its first `next_token` call: cache offset = max_len (unchanged from batched_prefill), `generated_tokens[0] = token_0`, `real_len = prompt_len + 1`. The next `step()` call feeds `token_0` to `forward_on`, identical to what GS's pipelined pre-fire would do. Cache trajectories agree from this point onward.

### 4.6 `step(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>>`

Preconditions:
- `phase == Decoding`. Else `Err("scheduler not in Decoding phase: call prefill_admitted first")`.

Steps:
1. Collect per-row inputs in slot order, length `b_max`:
   - `last_tokens[b]`: if `Some(state)` and `!state.finished`: `*state.generated_tokens.last().expect("prefill_admitted always pushes ≥ 1 token before step")`. Else (None slot or finished row): `0`.
   - `per_row_pos[b]`: if `Some(state)` and not finished: `state.real_len`. Else: `0` (None slot or finished row; both contribute pad).
2. Build `input_ids: [B, 1]` from `last_tokens` (int32).
3. Build `position_ids: [3, B, 1]` via `build_decode_position_ids(&per_row_pos)`.
4. Call `model.forward_on(&input_ids, &position_ids, Some(&mut cache.as_mut().unwrap()), ())`. Returns `logits: [B, 1, vocab]`.
5. For each row `b` in slot order:
   - Skip if `slots[b].is_none()` or `slots[b].as_ref().unwrap().finished`.
   - Slice `logits[b, 0, :] → [vocab]`.
   - Sample: `let token = state.sampler.sample(&row_logits)?` (per-row Sampler, see [`core/sampler.rs::Sampler::sample`](../../ironmlx/src/core/sampler.rs)).
   - Push to `state.generated_tokens`.
   - `state.real_len += 1`.
   - Termination check (in this exact order; first match wins):
     - If `state.stop_token_ids.contains(&token)`: set `state.finished = true; state.finish_reason = Some("stop")`.
     - Else if `state.generated_tokens.len() >= state.max_new_tokens`: set `state.finished = true; state.finish_reason = Some("length")`.
   - Push `StepEvent { id: state.id, token, finish_reason: state.finish_reason }` onto the result.
6. After all rows processed: if every occupied slot has `finished == true`, set `phase = Finished`. Else stay `Decoding`.
7. Return `Ok(events)`.

The events list contains **only** rows that were not-yet-finished at the start of this step. A row that finished during this step appears in the events with `finish_reason = Some(...)` — exactly once. Subsequent steps never emit anything for that row.

### 4.7 `evict_all(&mut self) -> Result<()>`

Preconditions:
- `phase ∈ {Decoding, Finished}`. If `Idle` or `Admitting`: returns `Err`.

Steps:
1. Set every `slots[i]` to `None`.
2. If `cache.is_some()`: iterate `cache.as_mut().unwrap()` and call `lc.reset()?` on each `LayerCache`.
3. Set `phase = Idle`.
4. `next_id` is **not** reset (the monotonic-no-reuse guarantee from 3a continues across batches).
5. Return `Ok(())`.

### 4.8 `LayerCache::reset(&mut self) -> Result<()>`

New `impl` block in [`ironmlx/src/nn/decoder_layer.rs`](../../ironmlx/src/nn/decoder_layer.rs) just after the enum:

```rust
impl LayerCache {
    /// Reset to empty state (offset → 0; recurrent state cleared). Preserves
    /// any underlying Array allocations so the next batch can reuse them.
    pub fn reset(&mut self) -> Result<()> {
        match self {
            LayerCache::Full(kv) => {
                kv.reset();
                Ok(())
            }
            LayerCache::Linear(gd) => gd.reset(),
        }
    }
}
```

(8 lines including the doc comment; pure dispatch.)

### 4.9 `StepEvent` data type

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepEvent {
    pub id: RequestId,
    pub token: u32,
    pub finish_reason: Option<&'static str>,
}
```

Minimal contract: caller looks up its request by `id`, decodes `token` via tokenizer, terminates the stream when `finish_reason.is_some()`. `row_idx` is intentionally absent — `Scheduler::get(id)` provides it if needed (3b-2 won't).

`PartialEq + Eq + Copy` enable easy assertion in tests.

### 4.10 Module surface summary

```text
ironmlx/src/core/scheduler.rs       (modified)
  + Phase enum
  + StepEvent struct
  + Scheduler::phase / prefill_admitted / step / evict_all
  + Scheduler.cache field
  + Scheduler.phase field
  + admit / evict phase integration

ironmlx/src/core/mod.rs              (modified)
  + pub use scheduler::{Phase, StepEvent};

ironmlx/src/nn/decoder_layer.rs      (modified)
  + impl LayerCache { pub fn reset(...) }

ironmlx/tests/b1_p2_3b_1_scheduler_step.rs  (new)
  + 3 integration scenarios (B=2 happy, B=4 happy, B=2 mixed-finish)
```

Zero changes to: `core/server/`, `core/generate.rs`, `core/sampler.rs`, `core/cache/`, `models/`, all `nn/` except `decoder_layer.rs`'s new 8 lines.

## §5 Tests

### 5.1 Unit tests (extend `scheduler.rs::tests` from 3a)

3a has 10 unit tests. 3b-1 adds 8 more, focused on phase transitions and `Err` paths — model-touching code is exercised in §5.2 integration tests.

| # | Test name | What it asserts |
| --- | --- | --- |
| 1 | `phase_starts_idle` | `Scheduler::new(4).phase() == Phase::Idle` |
| 2 | `admit_transitions_idle_to_admitting` | `admit()` in `Idle` → returns `Ok`; `phase() == Admitting` |
| 3 | `admit_stays_in_admitting` | Two consecutive `admit()` → both `Ok`; `phase() == Admitting` after each |
| 4 | `prefill_in_idle_returns_err` | `prefill_admitted` with no rows → `Err` containing "no admitted requests" |
| 5 | `step_in_admitting_returns_err` | `step` before `prefill_admitted` → `Err` containing "not in Decoding phase" |
| 6 | `step_in_idle_returns_err` | `step` in fresh scheduler → `Err` containing "not in Decoding phase" |
| 7 | `admit_in_decoding_returns_err` | Test must artificially flip the scheduler into `Decoding` via a helper, since we can't run real prefill in a unit test. Solution: gate the constructor with `#[cfg(test)] fn force_phase(&mut self, p: Phase)`. `admit` after force-flip → `Err` containing "cannot admit". |
| 8 | `evict_all_from_finished_resets_to_idle` | `force_phase(Finished)` then `evict_all` → `phase() == Idle`, `active_count() == 0` |

The `force_phase` helper is `#[cfg(test)]`-only and not part of the public API. It lets unit tests exercise phase-transition guards without booting a model.

### 5.2 Integration test `tests/b1_p2_3b_1_scheduler_step.rs`

Three scenarios, all using the existing P6 Qwen3.5-VL fixture model (loaded once per file via `tokio::sync::OnceCell` or a test-local lazy static — follow the pattern in `tests/b1_p2_2_batched_decode.rs`).

**Scenario A — `b1_p2_3b_1_b2_happy`** (B = 2 equal max_new_tokens):
1. Admit 2 requests with text prompts of unequal length (e.g., 16 and 24 tokens after templating). Both with `max_new_tokens = 16` and greedy sampler.
2. Call `prefill_admitted`. Assert `phase() == Decoding`. **Collect the returned `Vec<StepEvent>` — this is the first token per row** (revised §4.5).
3. Loop `step()` until `phase() != Decoding`. Collect events per request id, appending to the prefill events.
4. **In parallel reference:** run a single `GenerationStream` for each prompt with the same sampler and `max_new_tokens`, collect their token sequences (GS's default pipelined mode).
5. Compare: for each row, `argmax_bit_id_ratio(scheduler_tokens, baseline_tokens) >= 0.95` (the B1-p2.2 tolerance — bf16 ULP-driven flips expected). Token sequences should match closely because `prefill_admitted` now matches GS pipelined cache trajectory by sampling the first token from prefill logits.
6. Assert `phase() == Finished` at the end.
7. Call `evict_all`. Assert `phase() == Idle`, `active_count() == 0`.
8. **Re-use check:** admit 2 more rows + prefill + step a few times — confirm `cache.reset()` is honored (no offset overflow, no NaN, second batch yields plausible logits).

**Scenario B — `b1_p2_3b_1_b4_happy`** (B = 4 mixed prompt lengths):
Same template as A but with 4 prompts of lengths spanning, e.g., 12 / 16 / 20 / 24 tokens; `max_new_tokens = 12` for all; greedy. Each row's tokens must match its B=1 baseline at ≥ 95% bit-id.

**Scenario C — `b1_p2_3b_1_mixed_finish`** (B = 2 unequal `max_new_tokens`):
1. Admit 2 requests: row0 has `max_new_tokens = 8`, row1 has `max_new_tokens = 24`. Both greedy, both same prompt length.
2. Prefill + step until `phase() == Finished` (worst case 24 steps).
3. Collect events. Assert:
   - row0 events emitted on steps 1..=8; the step-8 event has `finish_reason = Some("length")`.
   - row1 events emitted on steps 1..=N where N ≤ 24 (and `finish_reason = Some("length")` on the last one OR `Some("stop")` if model emits an EOS).
   - No row0 events appear after step 8.
   - The total event count equals `8 + N`.
4. Verify each row's tokens vs B=1 baseline ≥ 95% bit-id.
5. `evict_all` and re-admit smoke check (1 row, 4 steps, no crash).

Tests are `#[ignore]`-gated like the other GPU tests (require `QWEN35_MODEL` env var + `MLX_DIR`).

### 5.3 Acceptance gates (referenced by Task 3 of the implementation plan)

- All 8 new unit tests + 18 prior scheduler tests pass.
- All 3 integration scenarios pass with argmax bit-id ≥ 95%.
- `cargo +nightly fmt --check` / `clippy -D warnings` / `cargo build --release -p ironmlx` clean.
- Lib test count rises from 174 → 182 (164 baseline + 10 scheduler 3a + 8 new 3b-1).
- P6.3, P6.6, P6.7, B1-p2.1, B1-p2.2 regression tests all pass with the same numerics they had at 3a close-out.

## §6 Estimate

**3–5 working days** broken down:
- Day 1 — `Phase` enum + scheduler field plumbing + `admit`/`evict` phase guards + 8 unit tests + `LayerCache::reset` dispatcher.
- Day 2 — `prefill_admitted` implementation; first integration scenario green.
- Day 3 — `step` implementation; per-row sampling loop; scenarios A + B green.
- Day 4 — Scenario C (mixed-finish) + re-use check; tighten error messages; full regression sweep.
- Day 5 (buffer) — close-out doc + any review-loop fixes.

This is comparable to 3a's 1.5d but model-touching adds debug cost (B1-p2.1 / 2.2 each took ~3d on the same surface).

## §7 Lockstep cost (must be documented in close-out)

- **Compute waste:** once a row finishes, it still occupies a slot in `step()`'s forward — the model spends `1 / (B - finished_count)` extra compute per step on padded rows. Worst case: B=4 with one row at 4 tokens and one at 1024 tokens — 1023 steps with 3 padded rows. 3c removes this.
- **Latency tail:** the whole batch's wall time is dominated by the longest `max_new_tokens`. 3d (preemption) will let new requests jump in once short ones finish.
- **No mid-batch admit:** caller cannot add a request to a running batch. Must wait for `Finished → Idle` transition. 3d (admission queue) addresses this.
- **First-token-from-prefill not used:** §4.5 step 9 discards the prefill logits. Future 3e optimization can sample the first new token directly from the prefill output, saving one full forward pass.

These costs are intentional for 3b-1 — they keep the implementation tight enough to land in a week. 3c–3e lift them in sequence.

## §8 Alternatives considered

The brainstorming session locked in six choices. Recording them here so future readers see why each was selected:

| Decision | Selected | Rejected alternatives (1-line summary) |
| --- | --- | --- |
| Scope | 3b-1 + 3b-2 split | Single 3b spec (2-week blast radius); 3b = step()-only (HTTP rot left dangling); 3b = HTTP-first (model-side untested) |
| API shape | Explicit `prefill_admitted` + `step` | Auto-prefill inside `step()` (hidden state machine); `run_until_finished` helper (breaks SSE streaming); mode-flag on `step()` (ugly) |
| Finished-row semantics | Skip sample, no event emit, slot retained | Sample anyway + filter (wasted compute + inconsistent semantics); auto-evict (slot can't be reused under lockstep anyway) |
| Cache lifecycle | Scheduler-owned + `reset()` reuse | Rebuild per batch (GPU alloc churn); caller-owned (two state machines to coordinate) |
| Sampling | Per-row `Sampler::sample` loop | argmax-only (no temperature/top_k); batched argmax kernel (premature, no sampler config support yet) |
| Acceptance gate | argmax bit-id ≥ 95% + 3 scenarios | 100% match (bf16 ULP makes this impossible); functional smoke only (would mask regressions) |

## §9 Open questions (intentional — to be answered during plan-writing or implementation)

1. **`cap` for `model.make_cache`** — `8192` is the hardcoded literal in 3b-1. Should this be a constructor argument on `Scheduler::new(b_max, cap)`? Decision deferred to plan-writing. The integration tests need a known value; in practice this affects how long a single batch can run before cache exhaustion.
2. **`model.dtype()` accessor** — does `Qwen35Model` expose its dtype publicly? If not, 3b-1 must add a one-line getter or `make_cache` must derive it. Inspection during plan-writing will confirm.
3. **`build_attention_mask_batched` helper location** — B1-p2.1 / B1-p2.2 build masks in test code. 3b-1's `prefill_admitted` needs this functionality. The plan task will either expose B1-p2.1's helper as `pub` or inline the equivalent ~15 lines into `scheduler.rs`. Either is acceptable — to be decided after reading B1-p2.1's actual implementation.

These three are implementation surface decisions, not design decisions. They don't change the architecture; the plan-writing step decides which is cleanest.

## §10 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md`](2026-05-13-b1-p2-3a-scheduler-skeleton-design.md)
- Predecessor plan: [`docs/superpowers/plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md`](../plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3a_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3a_closeout/report.md)
- Driving pattern reference: [`ironmlx/tests/b1_p2_2_batched_decode.rs`](../../ironmlx/tests/b1_p2_2_batched_decode.rs)
- Cache types: [`ironmlx/src/core/cache/`](../../ironmlx/src/core/cache/), [`ironmlx/src/nn/decoder_layer.rs`](../../ironmlx/src/nn/decoder_layer.rs)
- Model batched entry points: [`ironmlx/src/models/qwen3_5/model.rs`](../../ironmlx/src/models/qwen3_5/model.rs)
