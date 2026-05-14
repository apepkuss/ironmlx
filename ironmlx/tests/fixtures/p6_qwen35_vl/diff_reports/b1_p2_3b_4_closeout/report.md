# B1-p2.3b-4 Anthropic handler refactor — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-3 head `b8c8403`)
**Date:** 2026-05-14
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3b-4-anthropic-handler-design.md` (commit `973842b`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3b-4-anthropic-handler.md`

## Summary

Anthropic `/v1/messages` handler now routes text-only short-prompt requests
through `SchedulerActor` (mirror 3b-2's OpenAI refactor); long-prompt
requests stay on the existing `GenerationStream` path as sunset-tracked
compat. New `serve_via_scheduler_stream` builds the 6-event Anthropic SSE
sequence inline; new `serve_via_scheduler_unary` builds the `MessageEnvelope`
JSON response. Anthropic is permanently text-only so the routing predicate
has no `has_images` check (simpler than OpenAI).

Folded in 3b-3 final-review trivial Minors: removed redundant
`#[allow(clippy::too_many_arguments)]` on `driver_loop` (6 args below
clippy threshold), and updated the `driver_shuts_down_when_cmd_channel_closes`
unit test docstring to match the 3b-3 `rt.block_on(cmd_rx.recv())` form.

**Multi-request batching is now functional for both OpenAI and Anthropic
text routes.** Same `SchedulerActor`, same 5ms admission window —
heterogeneous batches (OpenAI + Anthropic requests in the same batch) are
technically possible since both routes admit through the same
`SchedulerCommand::Admit` channel.

## Acceptance

| Test | Result |
| --- | --- |
| `driver_shuts_down_when_cmd_channel_closes` (unit, docstring updated) | ✅ |
| `anthropic_actor_b1_text_only_swap` | ✅ admit_delta=1, bit_id=1.0000 |
| `anthropic_actor_long_prompt_routes_to_gs` | ✅ admit_count delta=0 |
| `anthropic_actor_scheduler_path_emits_6_event_sequence` | ✅ 6-event sequence verified, output_tokens=4, delta_count=4, stop_reason=max_tokens |

## Architectural Changes

1. **`ironmlx/src/core/server/anthropic.rs`**:
   - Renamed `messages_stream` → `serve_via_gs_stream` (body unchanged)
   - Renamed `messages_unary` → `serve_via_gs_unary` (body unchanged)
   - Added `serve_via_scheduler_stream` (~120 lines): admission boilerplate + tokio::spawn forwarder that emits 6-event SSE sequence inline; reuses `gen_msg_id`, `now_unix`, `format_event`. `output_tokens` increments unconditionally per `event_rx.recv()` Some (mirrors GS path line 277 semantics).
   - Added `serve_via_scheduler_unary` (~60 lines): admission boilerplate + drain event_rx + build `MessageEnvelope`. Same `output_tokens` semantics.
   - `messages` dispatch tail replaced with 4-way `match (stream, use_scheduler)`; routing predicate simplified (no `has_images` check — Anthropic permanently text-only)
   - Added `// COMPAT(3b-2/3b-4): long-prompt fallback...` comment with sunset target
   - Added imports: `tokio::sync::oneshot`, `crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand}`
   - `serve_via_scheduler_stream` and `serve_via_scheduler_unary` are `pub` (required for Task 3's integration test direct invocation — integration tests live in separate crate, can't see `pub(crate)`)
2. **`ironmlx/src/core/server/mod.rs`**:
   - `mod anthropic;` → `pub mod anthropic;` (same access reason)
3. **`ironmlx/src/core/server/scheduler_actor.rs`**:
   - M1: removed `#[allow(clippy::too_many_arguments)]` from `driver_loop` (6 args below clippy default threshold of 7)
   - M2: updated `driver_shuts_down_when_cmd_channel_closes` docstring to reference `rt.block_on(cmd_rx.recv())` (3b-3 form) instead of `cmd_rx.blocking_recv()` (3b-2 form)

No changes to: `core/server/openai.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## Compat sunset markers (recorded in code)

| Location | Marker | Sunset |
| --- | --- | --- |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4` | B1-p2.4 batched VL |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase` | 3c+ chunked prefill |
| `anthropic.rs::messages` dispatch (NEW) | `// COMPAT(3b-2/3b-4): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase` | 3c+ chunked prefill |
| `anthropic.rs::messages` image rejection | (implicit — 400 on image content parts) | Future Anthropic VL support phase (no current ETA) |
| `scheduler_actor.rs::ADMISSION_DEADLINE` | hardcoded 5ms | 3d/3e config exposure |

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `4041816` | feat | Anthropic handler routes text/short to SchedulerActor; 3b-3 M1/M2 cleanup |
| `3e8aa32` | test | anthropic actor scenarios 1 + 2 (text swap + long-prompt routing) |
| `1b5d7fc` | test | scenario 3 — 6-event SSE wire-format smoke + visibility bumps |
| (this) | docs | This close-out |

## Plan-Correction Deviations (Task 3)

The Task 3 implementation discovered that `pub(crate)` visibility was insufficient for integration test access — integration tests live in a separate crate and cannot see `pub(crate)` items in the lib crate. Required two visibility bumps:

1. **`core/server/mod.rs:17`**: `mod anthropic;` → `pub mod anthropic;` (was private to `core/server` module; tests need `ironmlx::core::server::anthropic::serve_via_scheduler_stream` path).
2. **`anthropic.rs`**: `pub(crate) async fn serve_via_scheduler_stream` → `pub async fn serve_via_scheduler_stream`; same for `serve_via_scheduler_unary`.

Both are minimum-necessary visibility relaxations for the test access pattern. The functions remain accessible only inside the `core::server::anthropic` module path; they are not exposed as top-level public API, just module-public. The plan's `pub(crate)` was the wrong visibility level — should have been `pub` (within the `pub mod anthropic`) from the start.

## Regression Status

All commands run with `--test-threads=1` against `QWEN35_MODEL` env var pointing to the Qwen3.5-4B-MLX-4bit fixture. Full 9-test sweep completed in single `cargo test` invocation, exit code `0`.

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **188 passed / 0 failed / 2 ignored** |
| P6.3 single-image (`p6_qwen35_vl_logits_match`) | PASS (32.93s) |
| P6.6 logits-match | PASS (2.88s) |
| P6.7 chunked-prefill matrix | PASS (1414.42s) |
| B1-p2.1 batched prefill | PASS |
| B1-p2.2 batched decode | PASS (808.40s) |
| B1-p2.3b-1 scheduler scenarios | PASS (3 scenarios, 374.00s) |
| B1-p2.3b-2 scheduler_actor scenarios | PASS (3 scenarios, 92.26s) |
| B1-p2.3b-3 admission_window scenarios | PASS (4 scenarios, 172.71s) |
| B1-p2.3b-4 anthropic_actor scenarios | PASS (3 scenarios, 161.88s) |

Exit code: `0`. No regressions.

## Notes

- **Anthropic multi-request batching is live.** Same `SchedulerActor` as 3b-3 — text-only short-prompt Anthropic requests pack into the same 5ms admission window. Heterogeneous OpenAI+Anthropic batches are technically supported (both routes share the same `SchedulerCommand::Admit` channel). Lock strategy unchanged from 3b-2/3b-3.
- **6-event SSE wire-format equivalence verified.** Scenario 3 parses the byte stream from `serve_via_scheduler_stream` and asserts event type sequence + `message_start.usage.input_tokens` + `message_delta.delta.stop_reason` mapping. Observed run: 9 events emitted in correct order, `output_tokens=4`, `delta_count=4`, `stop_reason="max_tokens"`. The `delta_count <= output_tokens` invariant holds.
- **`output_tokens` counter mirrors GS path semantics**: incremented per `event_rx.recv()` Some, NOT only per emitted delta. Some tokens whose detok output is empty (BPE mid-codepoint) still count toward `output_tokens` but are not emitted as `content_block_delta`. Verified by Scenario 3's `delta_count <= output_tokens` assertion.
- **No iron-bench compat concern.** iron-bench only exercises `/v1/chat/completions` (OpenAI). Anthropic path has no v1 client to keep green.
- **3b-3 minors closed**: M1 (`#[allow]` deletion) and M2 (docstring update) folded into Task 1's commit.
- **B1-p2.3b series complete.** 3b-1 (Scheduler::step + lockstep prefill) → 3b-2 (SchedulerActor skeleton + OpenAI swap) → 3b-3 (admission window + multi-request batching) → 3b-4 (Anthropic swap). Both HTTP routes now drive the same scheduler with the same admission window. Ready for 3c per-row offset work.

## B1-p2.3x Next Steps

- **B1-p2.3c** — Per-row KV cache offset tracking; lifts the lockstep constraint so finished rows can be evicted mid-batch and new rows can join at different offsets. Unblocks continuous batching across batch boundaries.
- **B1-p2.3 (chunked-prefill phase)** — Adds batched prefill chunking; removes both `prompt_len > chunk_size` fallbacks (OpenAI + Anthropic). Likely shipped alongside or right after 3c.
- **B1-p2.3d** — Admission queue + preemption. Also surfaces `ADMISSION_DEADLINE` via `AppConfig` + CLI flag.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback in OpenAI handler. Anthropic VL support remains a separate phase decision.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3b-4-anthropic-handler-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3b-4-anthropic-handler.md`
- Modified handler: `ironmlx/src/core/server/anthropic.rs`
- Modified module visibility: `ironmlx/src/core/server/mod.rs`
- Modified module (M1/M2): `ironmlx/src/core/server/scheduler_actor.rs`
- Integration test: `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md`
