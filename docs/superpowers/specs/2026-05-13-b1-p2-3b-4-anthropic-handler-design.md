# B1-p2.3b-4 — Anthropic handler refactor (design)

**Date:** 2026-05-14
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-3 head `b8c8403`)
**Predecessor sub-phase:** B1-p2.3b-3 — Admission window + multi-request batching (closed at `b8c8403`)
**Sibling sub-phase:** None — 3b-4 is the last 3b-family sub-phase before 3c
**Successor sub-phases:** 3c (per-row offset), 3c+ (chunked batched prefill), 3d (admission queue + preemption + config exposure), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Route Anthropic `/v1/messages` text-only short-prompt requests through `SchedulerActor` (mirror 3b-2's OpenAI refactor); long-prompt requests stay on `GenerationStream` as sunset-tracked compat (inherited from 3b-2).
2. Implement a 6-event SSE wrapper (`message_start` → `content_block_start` → N × `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop`) on top of `SchedulerActor`'s `StepEvent` stream — produces wire output indistinguishable from the existing GS-path 6-event sequence.
3. Mirror 3b-2's `serve_via_gs_*` / `serve_via_scheduler_*` helper split. Existing internal helpers `messages_stream` / `messages_unary` get renamed; new SchedulerActor-driven variants are added.
4. Fold in 3b-3 final-review trivial Minors: remove redundant `#[allow(clippy::too_many_arguments)]` on `driver_loop` (6 args don't trigger clippy's default threshold of 7), and update the `driver_shuts_down_when_cmd_channel_closes` unit test docstring to match the 3b-3 `rt.block_on(cmd_rx.recv())` form.

## §2 Non-goals

- **Anthropic VL support.** Anthropic handler permanently rejects `ContentPart::ImageUrl` with HTTP 400 (see `anthropic.rs:123-137`). Routing predicate has no `has_images` branch.
- **`tool_use` / `system` / `thinking` content types.** Out of scope; the existing handler doesn't support them either.
- **Configurable `b_max` or `ADMISSION_DEADLINE` exposure.** Still hardcoded — 3d/3e responsibility.
- **HTTP/axum end-to-end concurrent tests.** Spec §5 integration tests run at the actor level (skip axum), matching the 3b-2/3b-3 pattern.
- **SSE parser robustness tests.** Scenario 3 verifies event sequence shape; full malformed-message robustness is outside the wire-format smoke test scope.
- **Anthropic-specific scheduler stats (e.g., per-handler admit/batch counters).** Existing `admit_count` / `batch_count` are global to the actor — both handlers contribute. Splitting per-handler is a 3d/3e concern.

## §3 Background

### 3.1 Where 3b-3 left off

3b-3 (commits `b3ec1f9` → `b8c8403`) shipped:

- Admission window in `SchedulerActor::driver_loop`: hybrid 5ms-deadline + b_max-saturate. Multi-request batching live for the OpenAI text-only short-prompt path.
- `SchedulerActorHandle` gained `batch_count` + `saturate_triggered` test hooks.
- 3b-2 final-review minors M1 (evict_all warn) + M2 (chunk_size==0 routing comment) + M3 (Scenario 4 concurrent-with-GS no-deadlock) all closed.
- 4 integration scenarios for OpenAI path verified.

3b-4 leaves the actor untouched — just adds a second caller path (Anthropic) using the same `SchedulerCommand::Admit` API.

### 3.2 Current Anthropic handler architecture

[`ironmlx/src/core/server/anthropic.rs:110`](../../ironmlx/src/core/server/anthropic.rs#L110) `messages` handler:

```text
pub async fn messages(state, json_req) -> Response
  ├─ Image content rejection (lines 123-137) — return HTTP 400 if any ContentPart::ImageUrl
  ├─ Message flattening (lines 140-163) — collapse multi-part content to plain text
  ├─ Chat template + tokenize (line 167) — render_and_encode → prompt_ids
  ├─ Build GenerateRequest (lines 179-191) — pixel_values: None (text-only)
  ├─ Compute input_tokens = prompt_ids.len() as u32
  └─ Dispatch on req.stream:
     ├─ stream=true → messages_stream(state, request, model_id, input_tokens)  (line 200)
     │   ├─ tokio::task::spawn_blocking task acquires model.blocking_lock()
     │   ├─ GenerationStream::new(...)
     │   ├─ Emit message_start (line 213) + content_block_start (line 233)
     │   ├─ Loop next_token() emitting content_block_delta (line 265) + tracking output_tokens
     │   ├─ Map ev.finish_reason → stop_reason (line 278-284): "stop"→"end_turn", "length"→"max_tokens"
     │   ├─ Emit content_block_stop + message_delta(stop_reason, output_tokens) + message_stop
     │   └─ Response::builder() with SSE Body
     └─ stream=false → messages_unary(state, request, model_id, input_tokens)  (line 324)
         ├─ spawn_blocking task — buffer all text
         └─ Construct MessageEnvelope JSON
```

Key reusable helpers (stay in place):
- `gen_msg_id() -> String` (line 80-82): `format!("msg_{}", now_unix())`
- `now_unix() -> u64` (line 73-78)
- `format_event(event_type, payload) -> Bytes` (line 99-108): `event: <type>\ndata: <json>\n\n` framing

Event payloads constructed inline via `serde_json::json!` (not pre-defined structs).

### 3.3 OpenAI handler reference (3b-2 pattern to mirror)

[`ironmlx/src/core/server/openai.rs:355-381`](../../ironmlx/src/core/server/openai.rs#L355) is the routing pattern 3b-4 mirrors. Key admission boilerplate at `openai.rs:472-498`:

```rust
let (reply_tx, reply_rx) = oneshot::channel();
if state.scheduler_handle.cmd_tx
    .send(SchedulerCommand::Admit { request, reply_tx })
    .await.is_err()
{
    return (StatusCode::SERVICE_UNAVAILABLE, "scheduler actor unavailable").into_response();
}
let AdmitReply { request_id: _, mut event_rx } = match reply_rx.await {
    Ok(Ok(r)) => r,
    Ok(Err(e)) => return (StatusCode::BAD_REQUEST, format!("admit failed: {e}")).into_response(),
    Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response(),
};
```

3b-4 reuses this verbatim — only the post-admission wrapper differs (6-event SSE vs OpenAI flat chunks).

### 3.4 iron-bench impact (none)

iron-bench client only hits `/v1/chat/completions` (OpenAI). The Anthropic `/v1/messages` endpoint is not exercised by iron-bench v1 or v2. 3b-4 has no iron-bench compatibility concern.

## §4 Architecture

### 4.1 Routing decision tree (simpler than OpenAI — text-only by design)

```text
messages
  ├─ Image content parts → 400 (unchanged from current line 123-137)
  ├─ Render + tokenize (handler thread, unchanged)
  ├─ Build GenerateRequest with pixel_values=None
  ├─ input_tokens = prompt_ids.len() as u32
  └─ routing:
     ├─ prompt_len > chunk_size (and chunk_size != 0)  → GS path (sunset 3c+)
     └─ otherwise                                       → SchedulerActor path (new)
```

`use_scheduler` predicate (simpler than OpenAI — no `has_images` check needed):

```rust
let prompt_len = request.prompt_ids.len();
let use_scheduler =
    state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size;
```

### 4.2 `anthropic.rs` renames + new helpers

Existing helpers renamed (signatures unchanged):
- `messages_stream(state, request, model_id, input_tokens)` → `serve_via_gs_stream(state, request, model_id, input_tokens)` — body unchanged
- `messages_unary(state, request, model_id, input_tokens)` → `serve_via_gs_unary(state, request, model_id, input_tokens)` — body unchanged

New helpers:

```rust
/// Text-only short-prompt streaming path via SchedulerActor.
async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response;

/// Text-only short-prompt unary path via SchedulerActor.
async fn serve_via_scheduler_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response;
```

`messages` entry dispatch (replaces current line 194-198):

```rust
// COMPAT(3b-2/3b-4): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase.
let prompt_len = request.prompt_ids.len();
let use_scheduler =
    state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size;

match (req.stream, use_scheduler) {
    (true, true) => serve_via_scheduler_stream(state, request, model_label, input_tokens).await,
    (true, false) => serve_via_gs_stream(state, request, model_label, input_tokens).await,
    (false, true) => serve_via_scheduler_unary(state, request, model_label, input_tokens).await,
    (false, false) => serve_via_gs_unary(state, request, model_label, input_tokens).await,
}
```

### 4.3 6-event SSE wrapper (the meaty part)

`serve_via_scheduler_stream` body skeleton:

```rust
async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let msg_id = gen_msg_id();

    // 1. Admit via SchedulerActor — boilerplate identical to OpenAI.
    let (reply_tx, reply_rx) = oneshot::channel();
    if state.scheduler_handle.cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await.is_err()
    {
        return (StatusCode::SERVICE_UNAVAILABLE, "scheduler actor unavailable").into_response();
    }
    let AdmitReply { request_id: _, mut event_rx } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return (StatusCode::BAD_REQUEST, format!("admit failed: {e}")).into_response(),
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response(),
    };

    // 2. Spawn an async forwarder that emits the 6-event sequence as it
    // detokenizes StepEvents.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let tokenizer = state.tokenizer.clone();
    let msg_id_for_task = msg_id.clone();
    let model_id_for_task = model_id.clone();

    tokio::spawn(async move {
        // Event 1: message_start
        let payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id_for_task,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_id_for_task,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        });
        if tx.send(Ok(format_event("message_start", &payload))).await.is_err() {
            return;
        }

        // Event 2: content_block_start
        let payload = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        if tx.send(Ok(format_event("content_block_start", &payload))).await.is_err() {
            return;
        }

        // Events 3..N+2: content_block_delta per non-empty detok output.
        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        let mut output_tokens: u32 = 0;
        let mut stop_reason: &'static str = "end_turn";
        while let Some(ev) = event_rx.recv().await {
            // Detok current token.
            let text = match detok.step(ev.token) {
                Ok(Some(s)) => s,
                Ok(None) => String::new(), // BPE mid-codepoint — skip emission
                Err(_) => String::new(),   // surface as empty delta — acceptable for smoke
            };
            if !text.is_empty() {
                let payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                });
                if tx.send(Ok(format_event("content_block_delta", &payload))).await.is_err() {
                    return;
                }
                output_tokens += 1;
            }
            if let Some(reason) = ev.finish_reason {
                stop_reason = match reason {
                    "stop" => "end_turn",
                    "length" => "max_tokens",
                    other => other,
                };
                break;
            }
        }

        // Event N+3: content_block_stop
        let payload = serde_json::json!({"type": "content_block_stop", "index": 0});
        if tx.send(Ok(format_event("content_block_stop", &payload))).await.is_err() {
            return;
        }

        // Event N+4: message_delta
        let payload = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        if tx.send(Ok(format_event("message_delta", &payload))).await.is_err() {
            return;
        }

        // Event N+5: message_stop
        let payload = serde_json::json!({"type": "message_stop"});
        let _ = tx.send(Ok(format_event("message_stop", &payload))).await;
    });

    let stream = ReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}
```

Wire output is byte-for-byte identical to the GS path's 6-event sequence (same `format_event` framing, same JSON payload structure, same `stop_reason` mapping).

### 4.4 `serve_via_scheduler_unary`

```rust
async fn serve_via_scheduler_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let msg_id = gen_msg_id();

    // Admit (same boilerplate).
    let (reply_tx, reply_rx) = oneshot::channel();
    if state.scheduler_handle.cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await.is_err()
    {
        return (StatusCode::SERVICE_UNAVAILABLE, "scheduler actor unavailable").into_response();
    }
    let AdmitReply { request_id: _, mut event_rx } = match reply_rx.await { /* same match as stream */ };

    // Drain events; build response body.
    let mut detok = state.tokenizer.decode_stream(true);
    let mut text = String::new();
    let mut output_tokens: u32 = 0;
    let mut stop_reason: &'static str = "end_turn";
    while let Some(ev) = event_rx.recv().await {
        match detok.step(ev.token) {
            Ok(Some(s)) => { text.push_str(&s); output_tokens += 1; }
            Ok(None) => {}  // mid-codepoint; next token resolves
            Err(_) => {}    // best-effort
        }
        if let Some(reason) = ev.finish_reason {
            stop_reason = match reason {
                "stop" => "end_turn",
                "length" => "max_tokens",
                other => other,
            };
            break;
        }
    }

    let resp = MessageEnvelope {
        kind: "message",
        id: msg_id,
        role: "assistant",
        content: vec![ContentBlockText { kind: "text", text }],
        model: model_id,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: AnthropicUsage { input_tokens, output_tokens },
    };
    Json(resp).into_response()
}
```

(Field names follow existing `MessageEnvelope` / `ContentBlockText` / `AnthropicUsage` types in `anthropic.rs` lines 54-71. If those types are private to the module, the new helpers are also private, so no visibility change needed.)

### 4.5 `stop_reason` mapping reuse

The `match reason { "stop" → "end_turn", "length" → "max_tokens", other → other }` mapping appears in both `serve_via_gs_*` (existing) and `serve_via_scheduler_*` (new). Currently duplicated; not factored to a helper because (a) it's 4 lines, (b) only two call sites, (c) factoring would force a `pub(super)` exposure that obscures the inline reading. Spec §9 alternatives table documents the rejection.

### 4.6 3b-3 minor cleanups (in `scheduler_actor.rs`)

**M1**: Remove `#[allow(clippy::too_many_arguments)]` at `scheduler_actor.rs:111`. The `driver_loop` has 6 parameters, below clippy's default threshold of 7 — the `#[allow]` was unnecessary defensive scaffolding.

**M2**: Update `driver_shuts_down_when_cmd_channel_closes` test docstring (lines ~283-286) to reference the 3b-3 form `rt.block_on(cmd_rx.recv())` instead of the 3b-2 `cmd_rx.blocking_recv()` form. The test body itself doesn't change.

Both fixes are part of Task 1's commit — single-commit fold-in.

### 4.7 Module surface summary

```text
ironmlx/src/core/server/anthropic.rs           — MODIFY
  + Rename messages_stream → serve_via_gs_stream
  + Rename messages_unary  → serve_via_gs_unary
  + Add serve_via_scheduler_stream / serve_via_scheduler_unary
  + Replace messages dispatch with match (stream, use_scheduler)
  + Add // COMPAT(3b-4): long-prompt fallback comment

ironmlx/src/core/server/scheduler_actor.rs     — MODIFY
  + Remove #[allow(clippy::too_many_arguments)] (M1)
  + Update unit test docstring (M2)

ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs    — NEW
  + 3 integration scenarios + helpers

ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_4_closeout/report.md               — NEW close-out
```

Zero modifications to: `core/server/openai.rs`, `core/server/mod.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## §5 Tests

### 5.1 Unit tests

No new unit tests in `scheduler_actor.rs` (M1/M2 don't change behavior — M1 deletes an unused attribute, M2 updates a comment). Existing `driver_shuts_down_when_cmd_channel_closes` test passes unchanged.

No new unit tests in `anthropic.rs` either — wire-format coverage comes from Scenario 3 (integration).

### 5.2 Integration test scenarios (`tests/b1_p2_3b_4_anthropic_actor.rs`)

All `#[ignore]` `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]`:

**Scenario 1 — `anthropic_actor_b1_text_only_swap`**
1. Load model + tokenizer
2. Run B=1 GS baseline (via `GenerationStream` direct call, wrapped in `spawn_blocking`)
3. Spawn `SchedulerActor` via `spawn_scheduler_actor(model.clone(), 4)`
4. Admit single text request via `cmd_tx.send(SchedulerCommand::Admit { ... })`
5. Drain `event_rx`, collect tokens
6. Assert: `admit_count delta == 1`, `batch_count delta == 1`, per-row argmax bit-id ≥ 0.95 vs baseline (note: bit_id can flip due to bf16 ULP at single-row B=1 the path is numerically identical, but the assertion uses ≥0.95 as a safety margin matching 3b-2 pattern).

**Scenario 2 — `anthropic_actor_long_prompt_routes_to_gs`**
1. Build synthetic long prompt (> `prefill_chunk_size = 64`)
2. Verify routing predicate selects GS path (`use_scheduler == false`)
3. Spawn actor; record `admit_count` before
4. Call `serve_via_gs_*` simulation (no actor interaction)
5. Assert `admit_count delta == 0`

**Scenario 3 — `anthropic_actor_scheduler_path_emits_6_event_sequence`** (NEW unique to 3b-4)
1. Load model; spawn actor
2. Build text request with known small `max_new_tokens` (e.g., 4)
3. Directly invoke `serve_via_scheduler_stream(state, request, model_id, input_tokens)` from test
4. Collect SSE Response body bytes via `axum::body::to_bytes(response.into_body(), usize::MAX).await`
5. Parse SSE chunks: split on `\n\n`, extract `event: <type>` lines
6. Assert event sequence:
   - First event: `message_start`
   - Second event: `content_block_start`
   - Events 3..N+2: at least 1 × `content_block_delta` (count ≥ 1, typically = output_tokens)
   - Event N+3: `content_block_stop`
   - Event N+4: `message_delta`
   - Event N+5: `message_stop`
7. Assert `message_start` JSON payload contains `usage.input_tokens == input_tokens`, `output_tokens == 0`, `stop_reason: null`
8. Assert `message_delta` JSON payload contains `stop_reason == "end_turn"` (or `"max_tokens"`) AND `usage.output_tokens == count(content_block_delta events)`

Scenario 3's assertions match the contract observed by Anthropic SDK clients. If a future refactor breaks the event sequence, this test catches it.

### 5.3 Acceptance gates

- 3 integration scenarios PASS
- Existing regression sweep (P6.3 / P6.6 / P6.7 / B1-p2.1 / B1-p2.2 / B1-p2.3b-1 / B1-p2.3b-2 / B1-p2.3b-3): green
- `cargo +nightly fmt --check`, `clippy -D warnings`, `cargo build --release -p ironmlx`: clean
- Lib test count: 188 (unchanged — 3b-4 adds no lib tests, only integration tests)

## §6 Estimate

**3-5 working days:**
- D1 — `anthropic.rs` refactor (renames + new helpers + routing branch) + 3b-3 M1/M2 fixes in `scheduler_actor.rs`
- D2 — Scenarios 1 + 2 (B=1 swap + long-prompt routing)
- D3 — Scenario 3 (6-event wire-format smoke)
- D4 — Full regression sweep + close-out
- D5 (buffer)

## §7 Compat sunset notes

3b-4 inherits all 4 sunset markers from 3b-2/3b-3; no new compat introduced:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI long-prompt → GS | 3c+ chunked-prefill phase |
| Anthropic long-prompt → GS (NEW in 3b-4) | 3c+ chunked-prefill phase (same phase as OpenAI) |
| Anthropic image-content → 400 | B1-p2.4 (when batched VL lands AND Anthropic gets VL support — may be later than B1-p2.4) |
| `scheduler_actor.rs::ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config exposure |

Anthropic 400-on-images is permanent until Anthropic VL support is added, which is a separate decision after B1-p2.4.

## §8 Risk register

| Risk | Mitigation |
| --- | --- |
| `output_tokens` counter mismatch between Scenario 3's `message_delta` and observed `content_block_delta` count due to `decode_stream` skipping mid-codepoint tokens | Spec §4.3 only increments `output_tokens` when `text.is_empty() == false` — same condition as the `content_block_delta` emit. Counter invariant: every increment corresponds to one emitted delta. |
| 6-event order assertion in Scenario 3 fragile if `format_event` reorders bytes | Test parses the byte stream linearly; `format_event` writes deterministically. No nondeterminism. |
| Anthropic `unused_variable` warning on `_request_id` or shadowed bindings | Use `request_id: _` destructure pattern (3b-2 OpenAI uses this; lints clean). |
| Two `match reason { "stop" → ... }` mappings (GS path and scheduler path) drift over time | Documented in §9 alternatives. If drift becomes a real concern (3rd path added), factor to `map_finish_reason()`. |
| `decode_stream` lifetime gymnastics in spawned async task | Spec §4.3 clones `Arc<Tokenizer>` into the task and constructs `decode_stream` inside the task — same pattern as 3b-2 OpenAI scheduler stream path (which works). |

## §9 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Routing predicate | No `has_images` check (Anthropic permanently text-only) | Mirror OpenAI predicate verbatim (unnecessary defensive check) |
| 6-event SSE wrapper organization | Inline in `serve_via_scheduler_stream` (~80 lines) | Helper struct buffering state (premature abstraction for 1 caller); pre-defined per-event structs (current code uses ad-hoc `serde_json::json!` macros — consistent) |
| Wire-format test coverage | 1 sequence-shape smoke (Scenario 3) | Wire-format on all 3 scenarios (overkill); no wire-format test at all (3b-2 OpenAI didn't have one — but Anthropic's 6-event state machine is more complex and worth a smoke); axum end-to-end Server (heavy setup) |
| `stop_reason` mapping factoring | Inline match block (duplicated between GS path and scheduler path) | Extract `map_finish_reason(reason: &str) -> &'static str` helper (4 lines, 2 callers — abstraction cost > duplication cost) |
| 3b-3 trivial Minors | Fold into 3b-4 Task 1 | Punt to 3c (cruft accumulation); separate cleanup commit (overhead for 5-line changes) |
| Per-handler scheduler stats (admit/batch counts per handler) | Skip — accept that `admit_count` is global to actor | Add `anthropic_admit_count` / `openai_admit_count` (handler-attributed) — 3d/3e responsibility |

## §10 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md`](2026-05-13-b1-p2-3b-3-admission-window-design.md)
- Predecessor plan: [`docs/superpowers/plans/2026-05-13-b1-p2-3b-3-admission-window.md`](../plans/2026-05-13-b1-p2-3b-3-admission-window.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md)
- Current Anthropic handler: [`ironmlx/src/core/server/anthropic.rs`](../../ironmlx/src/core/server/anthropic.rs)
- OpenAI 3b-2 scheduler swap reference: [`ironmlx/src/core/server/openai.rs:459-end`](../../ironmlx/src/core/server/openai.rs)
- Scheduler actor API (unchanged in 3b-4): [`ironmlx/src/core/server/scheduler_actor.rs`](../../ironmlx/src/core/server/scheduler_actor.rs)
- 3b-2 OpenAI integration test pattern: [`ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs`](../../ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs)
