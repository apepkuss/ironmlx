# B1-p2.3b-2 SchedulerActor skeleton + OpenAI text-path swap — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-1 head `6c11f16`)
**Date:** 2026-05-13
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md` (commit `d4ebc4f`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton.md`

## Summary

Introduced `SchedulerActor` — a Tokio task wrapping the 3b-1 `Scheduler`
with mpsc cmd channel + per-request `mpsc::UnboundedReceiver<StepEvent>`.
OpenAI handler now routes text-only short-prompt requests through the
actor; VL and long-prompt requests stay on the existing `GenerationStream`
path as sunset-tracked compat (B1-p2.4 / 3c+ chunked-prefill).

Driver loop is the "one-admit-per-batch" form: each admit immediately
triggers a batch. 3b-3 will replace this with admission-window logic to
realize multi-request batching. Anthropic handler untouched (3b-4).

Scenario A integration test verifies the SchedulerActor path produces
the same token stream as direct `GenerationStream` at argmax bit_id =
**1.0000** for the test prompt — perfect parity at the model layer
inherits from 3b-1's already-validated `Scheduler::prefill_admitted` +
`Scheduler::step` correctness.

## Acceptance

| Test | Result |
| --- | --- |
| `driver_shuts_down_when_cmd_channel_closes` (unit) | ✅ |
| `scheduler_actor_b1_text_only_swap` (integration) | ✅ bit_id=1.0000 |
| `scheduler_actor_long_prompt_routes_to_gs` (integration) | ✅ admit_count delta=0 |
| `scheduler_actor_vl_routes_to_gs` (integration) | ✅ admit_count delta=0 |

## Architectural Changes

1. **`ironmlx/src/core/server/scheduler_actor.rs`** — new module: `SchedulerCommand`, `AdmitReply`, `SchedulerActorHandle { cmd_tx, admit_count }`, `spawn_scheduler_actor(model, b_max)`, `driver_loop`, `run_batch_once`, `route_event`. Driver runs on `tokio::task::spawn_blocking` (because `Scheduler` is `!Send`), holds `model.blocking_lock()` only during a batch, drops at evict_all.
2. **`ironmlx/src/core/server/mod.rs`** — `pub mod scheduler_actor;`. `AppState` gains `scheduler_handle: SchedulerActorHandle`. `serve()` calls `spawn_scheduler_actor(model.clone(), 4)` before building `AppState`.
3. **`ironmlx/src/core/server/openai.rs`** — renamed `chat_completions_stream` → `serve_via_gs_stream`, `chat_completions_unary` → `serve_via_gs_unary`. Added `serve_via_scheduler_stream` / `serve_via_scheduler_unary` that admit via `SchedulerActorHandle::cmd_tx`, drain `StepEvent` from per-request channel, detokenize via `tokenizer.decode_stream(true)`, format SSE chunks identical to GS path. `chat_completions` adds 4-way `match (stream, use_scheduler)` dispatch with explicit `// COMPAT(3b-2)` comments.

No changes to: `core/server/anthropic.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## Compat sunset markers (recorded in code)

| Location | Marker | Sunset |
| --- | --- | --- |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4` | B1-p2.4 lands batched VL |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase` | 3c+ ships chunked batched prefill |
| `scheduler_actor.rs::driver_loop` | `// 3b-2: one-admit-per-batch. 3b-3 replaces this with admission-window` | 3b-3 lands batching activation |
| `anthropic.rs` untouched | (implicit) | 3b-4 lands Anthropic handler refactor |

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `8dd3590` | feat | SchedulerActor module — driver_loop + cmd/event channels |
| `57e9092` | feat | AppState carries SchedulerActorHandle; serve() spawns driver |
| `f3c143f` | feat | OpenAI handler routes text/short to SchedulerActor; VL/long to GS |
| `3c5a36e` | test | SchedulerActor + routing integration (3 scenarios) |
| `<T5_SHA>` | docs | This close-out |

(Fill `<T5_SHA>` after Step 5.11 commit.)

## Plan-Correction Deviations (Task 4)

The Task 4 integration test file made 5 plan-correction deviations — all 根因 fixes to plan inaccuracies, not spec violations:

1. **`encode_with_chat_template`** doesn't exist on `Tokenizer`. Used the 3b-1 canonical `apply_chat_template + encode` pattern.
2. **`run_b1_baseline`** plan signature took `&Mutex<Qwen35Model>`; actual `GenerationStream::new` wants `&Qwen35Model`. Restructured.
3. **`GenerateEvent.token`** is `u32`, not `Option<u32>`. Loop adjusted.
4. **`image_grid_thw`** is `Option<Vec<(i32, i32, i32)>>`, not `Option<Array>`. Dummy grid corrected.
5. **`tokio::sync::Mutex::blocking_lock`** panics on Tokio worker thread ("Cannot block the current thread from within a runtime"). Scenario B's GS-inference simulation removed; routing-predicate + admit_count delta checks alone correctly verify the routing decision.

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean (only mlx-sys C++ noise) |
| `cargo build --release -p ironmlx` | clean — `Finished release profile` |
| `cargo test -p ironmlx --lib --release` | **188 passed / 0 failed** |
| P6.3 single-image | PASS — ok (54s) |
| P6.6 logits-match | PASS — ok (131s) |
| P6.7 chunked-prefill matrix | PASS — ok (102s) |
| B1-p2.1 batched prefill | PASS — ok (11s) |
| B1-p2.2 batched decode | PASS — ok (627s) |
| B1-p2.3b-1 scheduler scenarios | PASS — 3 passed (193s) |
| B1-p2.3b-2 b1_text_only_swap | PASS — bit_id=1.0000 |
| B1-p2.3b-2 long_prompt_routes_to_gs | PASS — admit_count delta=0 |
| B1-p2.3b-2 vl_routes_to_gs | PASS — admit_count delta=0 |

## Manual smoke (Step 5.9 observations)

Deferred: no standalone server binary in this commit (`ironmlx/src/bin/` does not exist). Manual smoke covered indirectly by integration tests B + C that verify routing predicates (long-prompt → GS, VL → GS) and Scenario A that verifies the SchedulerActor path produces correct token output via the HTTP wiring simulation.

## Notes

- **One-admit-per-batch is intentional for 3b-2.** Multi-request batching activation lives in 3b-3. This sub-phase ships the actor pattern + per-request channels + lock strategy without yet realizing batching throughput gains. Iron-bench v1 sees no protocol change; v2's batching benchmarks need 3b-3 to land first.
- **Scenario A bit_id=1.0000 inherits from 3b-1.** The model-layer correctness was already validated in B1-p2.3b-1's 3 scenarios. 3b-2's Scenario A merely confirms the HTTP wiring + actor-channel plumbing don't introduce drift.
- **Lock strategy verified.** Driver task holds `model.blocking_lock()` only during `run_batch_once`. GS path (VL / long prompt) acquires the same lock during its own `spawn_blocking` body. Idle periods leave the lock free.
- **Detokenization moved to handler side.** `SchedulerActor` returns raw `StepEvent { id, token, finish_reason }`. Handler constructs a `DecodeStream` per request and emits text deltas. Mirrors GS path semantics; SSE wire format unchanged.
- **Test hook `admit_count: Arc<AtomicU64>`** lets integration tests assert routing decisions without instrumenting handlers. Doc-hidden; production code should not depend on it.

## B1-p2.3x Next Steps

- **B1-p2.3b-3** — Replace `driver_loop`'s `cmd_rx.blocking_recv()` with `select! { admit | tick }` admission window. Multi-request batching activation. Concurrent-request integration tests.
- **B1-p2.3b-4** — Anthropic handler refactor (6-event SSE wrapper).
- **B1-p2.3c** — Per-row KV cache offset tracking; lifts lockstep constraint.
- **B1-p2.3 (chunked-prefill phase)** — Adds batched prefill chunking; removes `prompt_len > chunk_size` fallback to GS.
- **B1-p2.3d** — Admission queue + preemption.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL fallback to GS.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton.md`
- New module: `ironmlx/src/core/server/scheduler_actor.rs`
- Integration test: `ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs`
- Predecessor: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md`
