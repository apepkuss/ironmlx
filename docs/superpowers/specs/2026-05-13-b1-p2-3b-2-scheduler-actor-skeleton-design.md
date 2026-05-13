# B1-p2.3b-2 — SchedulerActor skeleton + OpenAI text-path swap (design)

**Date:** 2026-05-13
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-1 head `6c11f16`)
**Predecessor sub-phase:** B1-p2.3b-1 — Scheduler step + lockstep prefill (closed at commit `7e587a5` + `6c11f16`)
**Sibling sub-phases:**
- 3b-3 — Admission window + multi-request batching activation (separate spec; depends on this)
- 3b-4 — Anthropic handler refactor (separate spec; depends on this)
**Successor sub-phases:** 3c (per-row offset), 3c+ (chunked batched prefill), 3d (admission queue + preemption), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Introduce a `SchedulerActor` — a dedicated Tokio task that owns a `Scheduler` instance and accepts `SchedulerCommand::Admit` via mpsc channel, returning per-request `mpsc::UnboundedReceiver<StepEvent>` so HTTP handlers can stream events back as SSE.
2. Refactor `core/server/openai.rs` `chat_completions_stream` / `_unary` to route **text-only short-prompt** requests to the SchedulerActor path; **VL** and **long-prompt** requests remain on the existing `GenerationStream` path (temporary compat — see §8 sunset table).
3. Land the actor-pattern plumbing (cmd channel, per-request event channel, lock strategy, driver state machine) **without** activating multi-request batching — 3b-3 will switch the driver from "1-admit-per-batch" to "admission-window-multi-admit".
4. Maintain bit-id parity with `GenerationStream` baseline at ≥ 0.95 for text-only short-prompt requests routed via SchedulerActor (3b-1 already achieved 1.0000 at the scheduler layer; this sub-phase verifies the HTTP wiring doesn't regress).

## §2 Non-goals

- **Multi-request batching activation.** Defer to 3b-3. 3b-2's driver accepts one admit at a time and runs the batch immediately — equivalent to B=1 per batch.
- **Anthropic handler refactor.** Defer to 3b-4. `core/server/anthropic.rs` is untouched.
- **Batched-prefill chunking.** Defer to a phase after 3c (when per-row offset lifts the lockstep cache constraint). Long prompts route to GS path.
- **VL request handling in SchedulerActor.** Defer to B1-p2.4. VL requests route to GS path.
- **`Model` trait abstraction.** Known debt (3b-1 M-3); deferred until a second concrete model lands (P5 `qwen3_5_moe` or similar). `SchedulerActor` continues to accept `Arc<Mutex<Qwen35Model>>` concretely.
- **Concurrent-request stress testing.** 3b-3 ships the multi-request integration test suite. 3b-2 verifies single-request swap-in.

## §3 Background

### 3.1 Where 3b-1 left off

3b-1 (commits `f1f609b` → `6c11f16`) shipped:

- `Scheduler` API: `new`, `b_max`, `admit`, `evict`, `evict_all`, `active`, `active_count`, `get`, `get_mut`, `occupied_rows`, `phase`, `prefill_admitted(model) -> Result<Vec<StepEvent>>`, `step(model) -> Result<Vec<StepEvent>>`, `#[cfg(test)] force_phase`
- `Phase` state machine (`Idle → Admitting → Decoding → Finished → Idle`)
- `StepEvent { id, token, finish_reason }`
- `LayerCache::reset()` dispatcher
- Poison flag — any `Err` from `prefill_admitted` or `step` poisons the scheduler; `evict_all` clears
- Integration scenarios at bit_id=1.0000 vs GenerationStream baseline (B=1 / B=2 / B=4 / mixed-finish)

3b-2 builds on this API surface unchanged. No new `Scheduler` methods, no `RequestState` field changes.

### 3.2 Current HTTP path (to be swapped in 3b-2 for text/short)

[`ironmlx/src/core/server/openai.rs:261`](../../ironmlx/src/core/server/openai.rs#L261) `chat_completions` handler today:

```text
async chat_completions(state, json_req)
  ├─ image preprocessing (async)
  ├─ render_and_encode → prompt_ids (async)
  └─ branch on stream:
     ├─ stream=true → chat_completions_stream (line 362)
     │   ├─ spawn_blocking task (line 372)
     │   │   ├─ state.model.blocking_lock() (line 373)
     │   │   ├─ GenerationStream::new(&*model, &tokenizer, request) (line 375)
     │   │   └─ loop: stream.next_token() → mpsc::Sender::blocking_send(SSE chunk)
     │   └─ Body::from_stream(ReceiverStream::new(rx)) → SSE response
     └─ stream=false → chat_completions_unary (line 443)
         ├─ spawn_blocking task
         └─ loop: stream.next_token() → buffer → CompletionResponse JSON
```

Key facts:

- Per-request `spawn_blocking` + per-request `state.model.blocking_lock()` ⇒ concurrent requests **serialize** (test at [`mod.rs:75-102`](../../ironmlx/src/core/server/mod.rs#L75) confirms).
- `GenerationStream` owns its cache (allocated in `new` via `model.make_cache(1, cap, dtype)`) — single-row cache.
- Detokenization happens inside `GenerationStream::next_token` (pipelined `DecodeStream` or sync full-history diff).
- Termination (`EOS` / `max_new_tokens` / `finish_reason`) all inside `GenerationStream`.

### 3.3 Multi-request batching gap

Today's HTTP architecture cannot batch concurrent requests because each request constructs its own `GenerationStream` with its own cache, and the model is locked once per-request for the full duration. Even if two HTTP requests arrive in the same tokio tick, the first to acquire the lock runs the entire generation before releasing.

3b-2 introduces the actor-pattern plumbing that **could** batch concurrent requests, but **does not yet activate** the admission window. Behavior in 3b-2 remains effectively serial: handlers send admit commands to the driver; the driver immediately runs one batch per admit. 3b-3 changes the driver loop to wait briefly for additional admits before starting a batch.

This split lets 3b-2 ship a smaller, reviewable refactor without the additional complexity of admission-window timing (which has its own design considerations — sleep duration, fairness, fail-open semantics).

### 3.4 iron-bench compatibility

[`iron-bench/src/client.rs:158`](../../iron-bench/src/client.rs#L158) hits only `/v1/chat/completions` with `stream: true`. Parses `choices[0].delta.content`, `finish_reason`, optional `usage`. No concurrent-request tests in tree. 3b-2 must keep the OpenAI handler's SSE chunk format unchanged so iron-bench v1 continues to work without modification.

## §4 Architecture

### 4.1 Request routing decision tree

OpenAI handler entry decides per-request whether to use SchedulerActor or fall back to GS path:

```text
chat_completions
  ├─ image preprocess + tokenize (async, handler thread, unchanged)
  ├─ build GenerateRequest
  └─ routing:
     ├─ has_images (request.pixel_values.is_some())  → GS path (compat)
     ├─ prompt_ids.len() > state.prefill_chunk_size  → GS path (compat)
     └─ otherwise                                    → SchedulerActor path (new)
```

Both paths produce identical SSE output (same wire contract). iron-bench v1 sees no protocol change.

### 4.2 New module `ironmlx/src/core/server/scheduler_actor.rs`

Top-level structure:

```rust
//! SchedulerActor — Tokio task wrapper around `Scheduler` for serving
//! HTTP requests via mpsc channels. See spec §4 for protocol.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::generate::GenerateRequest;
use crate::core::scheduler::{Phase, RequestId, Scheduler, StepEvent};
use crate::models::qwen3_5::Qwen35Model;

/// Commands accepted by [`SchedulerActorHandle`].
pub enum SchedulerCommand {
    /// Submit a request for batched generation. On success, replies with the
    /// admitted `RequestId` and an mpsc receiver that streams `StepEvent`s
    /// (prefill-first-token, then one per decode step, until `finish_reason`
    /// is `Some(_)` on the last event for this row).
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
}

pub struct AdmitReply {
    pub request_id: RequestId,
    pub event_rx: mpsc::UnboundedReceiver<StepEvent>,
}

/// Handle held by `AppState`. Cheap to clone (`mpsc::Sender` is `Clone`).
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
}

/// Spawn the driver task. Returns a handle to send commands.
///
/// `b_max` is the scheduler's fixed slot count. 3b-2 ships with `b_max = 4`
/// hardcoded (matches B1-p2.3b-1 integration tests). 3b-3 may make this
/// configurable via env var or AppConfig.
///
/// The driver task is spawned via `tokio::task::spawn_blocking` because
/// `Scheduler` and `Qwen35Model::blocking_lock()` are sync. A long-running
/// blocking task is the correct vehicle.
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
) -> SchedulerActorHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    tokio::task::spawn_blocking(move || driver_loop(model, b_max, cmd_rx));
    SchedulerActorHandle { cmd_tx }
}
```

### 4.3 Driver loop (3b-2 form: one-admit-per-batch)

```rust
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> =
        HashMap::new();

    while let Some(cmd) = cmd_rx.blocking_recv() {
        match cmd {
            SchedulerCommand::Admit { request, reply_tx } => {
                let (event_tx, event_rx) = mpsc::unbounded_channel();
                match sched.admit(request) {
                    Ok(id) => {
                        event_txs.insert(id, event_tx);
                        if reply_tx
                            .send(Ok(AdmitReply { request_id: id, event_rx }))
                            .is_err()
                        {
                            // Caller dropped reply_rx before getting the id.
                            // Evict the orphan slot and continue.
                            let _ = sched.evict(id);
                            event_txs.remove(&id);
                            continue;
                        }
                        // 3b-2: run the batch immediately. 3b-3 will replace
                        // this with admission-window logic that drains more
                        // SchedulerCommand::Admit messages before batching.
                        if let Err(e) =
                            run_batch_once(&mut sched, &model, &mut event_txs)
                        {
                            // Driver-internal error: best-effort evict +
                            // log. Caller's event_rx will hit EOF (channel
                            // dropped), surfacing as truncated SSE stream.
                            eprintln!("[SchedulerActor] batch error: {e:?}");
                            let _ = sched.evict_all();
                            event_txs.clear();
                        }
                    }
                    Err(e) => {
                        let _ = reply_tx.send(Err(e));
                    }
                }
            }
        }
    }
    // cmd_rx closed → all senders dropped → shutting down. Drop sched +
    // event_txs naturally.
}

/// Acquire the model lock, run the full batch (prefill + step loop +
/// evict_all), and route events to per-request channels. Releases the
/// lock when the function returns.
fn run_batch_once(
    sched: &mut Scheduler,
    model: &Arc<Mutex<Qwen35Model>>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) -> Result<()> {
    let model = model.blocking_lock();

    let prefill_events = sched.prefill_admitted(&*model)?;
    for ev in prefill_events {
        route_event(ev, event_txs);
    }

    while sched.phase() == Phase::Decoding {
        let events = sched.step(&*model)?;
        for ev in events {
            route_event(ev, event_txs);
        }
    }

    sched.evict_all()?;
    event_txs.clear(); // closes all per-request channels → handlers see EOF
    Ok(())
}

fn route_event(
    ev: StepEvent,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // unbounded_channel — send is sync + infallible unless the receiver
        // was dropped (handler abandoned the request). On Err we don't
        // remove the entry; the next route attempt will see Err again and
        // the entry naturally clears at evict_all → event_txs.clear().
        let _ = tx.send(ev);
    }
}
```

**Why `tokio::task::spawn_blocking` for the driver?**
- `Scheduler` is `!Send` (because `Sampler` contains `Cell<Option<Array>>` — see 3a spec §4.2).
- `Qwen35Model::blocking_lock` is sync.
- `tokio::task::spawn_blocking` runs the closure on a dedicated thread pool, allowing arbitrary blocking work without starving the runtime.
- The closure owns `Scheduler` and never moves it across threads, satisfying `!Send`.

### 4.4 `AppState` integration

[`ironmlx/src/core/server/mod.rs`](../../ironmlx/src/core/server/mod.rs) `AppState` extends:

```rust
pub struct AppState {
    pub model: Arc<Mutex<Qwen35Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    pub prefill_chunk_size: usize,
    pub scheduler_handle: SchedulerActorHandle,  // NEW
}
```

Server startup function spawns the actor before constructing `AppState`:

```rust
pub fn serve(addr: SocketAddr, model_dir: &Path) -> Result<()> {
    // ... existing model + tokenizer load ...
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        4,  // b_max — hardcoded for 3b-2; future phase makes configurable
    );
    let state = AppState {
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        scheduler_handle,
    };
    // ... existing axum router setup with state ...
}
```

### 4.5 OpenAI handler routing

`chat_completions_stream` / `_unary` body before `spawn_blocking` already builds `GenerateRequest`. Add the routing branch:

```rust
async fn chat_completions_stream(state: AppState, req: GenerateRequest) -> Response {
    let use_scheduler =
        req.pixel_values.is_none()
        && req.prompt_ids.len() <= state.prefill_chunk_size;

    if use_scheduler {
        serve_via_scheduler_stream(state, req).await
    } else {
        serve_via_gs_stream(state, req).await  // existing path, renamed
    }
}
```

`serve_via_scheduler_stream` (new helper, in `openai.rs`):

```rust
async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
) -> Response {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "scheduler actor unavailable");
    }
    let AdmitReply { request_id: _, mut event_rx } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return error_response(StatusCode::BAD_REQUEST, &format!("admit failed: {e:?}")),
        Err(_) => return error_response(StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost"),
    };

    // Driver sent us the receiver; now stream events as SSE.
    let tokenizer = state.tokenizer.clone();
    let model_id = state.model_id.clone();
    let (sse_tx, sse_rx) = mpsc::channel(64);

    tokio::spawn(async move {
        let mut detok = DecodeStream::new(&tokenizer);
        // Initial role chunk (matches GS path output) — see openai.rs:398.
        let role_chunk = format_openai_role_chunk(&model_id);
        if sse_tx.send(Ok(role_chunk.into())).await.is_err() {
            return;
        }
        while let Some(ev) = event_rx.recv().await {
            let text = detok.step(ev.token).unwrap_or_default();
            let chunk = format_openai_delta_chunk(&model_id, &text, ev.finish_reason);
            if sse_tx.send(Ok(chunk.into())).await.is_err() {
                break;
            }
            if ev.finish_reason.is_some() {
                break;
            }
        }
        let _ = sse_tx.send(Ok("data: [DONE]\n\n".into())).await;
    });

    let stream = ReceiverStream::new(sse_rx);
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}
```

`serve_via_scheduler_unary` is analogous but buffers all events and returns a single `CompletionResponse` JSON.

`format_openai_role_chunk` / `format_openai_delta_chunk` are extracted helpers from the existing `chat_completions_stream` body (these helpers already exist as inline closures today; 3b-2 promotes them to module-level `fn`s so both paths use them).

### 4.6 Detokenization responsibility

- **SchedulerActor: no detok.** Returns raw `StepEvent { id, token, finish_reason }`. Keeps the actor independent of tokenizer (driver only depends on model). 3b-3 / 3b-4 reuse this contract.
- **Handler-side detok.** `serve_via_scheduler_stream` constructs a fresh `DecodeStream` and calls `step(token)` per event. Matches `GenerationStream::next_token` semantics (pipelined-mode path in `generate.rs:920-925`).
- **`finish_reason` handling.** The `StepEvent.finish_reason: Option<&'static str>` field carries `"stop"` / `"length"` per 3b-1 §4.6. Handler maps to OpenAI's `finish_reason` field on the last delta chunk (matches current GS path mapping).

### 4.7 Lock strategy

Per the brainstorm decision: **driver task acquires `model.blocking_lock()` per batch, releases on batch completion**.

- **Idle state (`Scheduler::phase() == Idle`)**: driver is parked on `cmd_rx.blocking_recv()`. Lock is **not held**. GS path handlers can acquire freely.
- **Active batch state**: driver holds the lock for the entire `run_batch_once` call (prefill_admitted + N×step + evict_all). GS path handlers block on the lock until the batch completes.
- **GS path handlers**: continue calling `state.model.blocking_lock()` directly (unchanged from today).

Implication: while the SchedulerActor is running a batch, VL or long-prompt requests serialize behind the batch. For 3b-2 single-request-per-batch behavior, this is identical to today's all-serial behavior; no worse. 3b-3's multi-request batching genuinely improves text-request throughput at the cost of mildly longer VL queueing (acceptable per Boss preference: text batching is the iron-bench v2 driver).

### 4.8 Error paths

| Failure | Behavior |
| --- | --- |
| `cmd_tx.send` fails (channel closed — driver crashed) | Return 503 to client |
| `reply_rx.await` fails (driver dropped reply_tx) | Return 503 |
| `reply_rx.await` returns `Err` (admit rejected — full / phase wrong / poisoned) | Return 400 with the underlying anyhow message |
| `prefill_admitted` / `step` returns `Err` mid-batch | Driver poisons the scheduler internally (3b-1 mechanism), calls `evict_all` (which clears poison), drops `event_txs`. Handler's `event_rx.recv()` returns `None` → SSE stream truncates after the last successful event. Client sees a partial response with `[DONE]` but no `finish_reason` chunk — best-effort. |
| Driver task panics | `cmd_tx.send` starts failing; future requests get 503. Future hardening: restart-on-panic. Not in 3b-2 scope. |

### 4.9 Module surface summary

```text
ironmlx/src/core/server/scheduler_actor.rs      — NEW (~200 lines)
  + SchedulerCommand enum
  + SchedulerActorHandle (cmd_tx wrapper)
  + AdmitReply struct
  + spawn_scheduler_actor()
  + driver_loop()
  + run_batch_once()
  + route_event()

ironmlx/src/core/server/mod.rs                   — MODIFY (~30 lines)
  + AppState.scheduler_handle field
  + serve() spawns the actor before building AppState
  + pub mod scheduler_actor;

ironmlx/src/core/server/openai.rs                — MODIFY (~150 lines)
  + routing branch in chat_completions_stream / _unary
  + serve_via_scheduler_stream / _unary helpers
  + format_openai_role_chunk / format_openai_delta_chunk helpers
    (extracted from inline closures)
  + serve_via_gs_stream / _unary helpers (renamed from current body)

ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs      — NEW (~300 lines)
  + 3 integration scenarios + helpers

ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_2_closeout/report.md                — NEW close-out
```

Zero changes to: `core/server/anthropic.rs`, `core/server/chat_format.rs`, `core/generate.rs`, `core/scheduler.rs`, `core/sampler.rs`, `core/cache/`, `core/tokenizer.rs`, `models/`, `nn/`.

## §5 Tests

### 5.1 Unit tests

No new unit tests in `scheduler.rs` (its API is unchanged from 3b-1). New unit tests in `scheduler_actor.rs` are limited — Tokio actor patterns are hard to unit-test without a real model. The integration tests in §5.2 cover the actor end-to-end.

A small unit test ensures the driver shuts down cleanly when the cmd channel is dropped:

```rust
#[tokio::test]
async fn scheduler_actor_shuts_down_on_channel_close() {
    // Spawn a no-op driver with a dummy model placeholder. Drop the
    // SchedulerActorHandle; verify the task completes.
    // (This requires a #[cfg(test)] no-op model stub; if too invasive,
    // skip this test and rely on integration coverage.)
}
```

If the no-op model stub adds too much surface, drop this unit test and rely on integration tests + manual smoke. Plan-writing phase decides.

### 5.2 Integration test scenarios

`ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs`, all `#[ignore]`-gated:

**Scenario A — `scheduler_actor_b1_text_only_swap`** (B=1 text via SchedulerActor):
1. Spawn a real server (or use a `axum::body::to_bytes` test harness — both work).
2. Send a text-only chat completion request (no images) with prompt under `prefill_chunk_size`.
3. Verify the SSE stream parses cleanly: `role` chunk + N × `delta.content` chunks + final `finish_reason` chunk + `[DONE]`.
4. Compare the decoded text against running the same prompt through `GenerationStream` directly (B=1 baseline).
5. Assert per-token argmax bit-id ≥ 0.95 (expect 1.0000 per 3b-1).

**Scenario B — `scheduler_actor_long_prompt_routes_to_gs`** (long prompt falls back to GS):
1. Send a request with prompt_len = `chunk_size + 1` (exactly one over threshold).
2. Verify the response is generated (no 4xx error).
3. Side-channel check: instrument the driver to expose `admit_count` (a test-only `AtomicU64`); assert it did NOT increment for this request.
4. Assert the token sequence matches the GS-only baseline exactly (it must — same code path).

**Scenario C — `scheduler_actor_vl_routes_to_gs`** (VL falls back to GS):
1. Send a single-image OpenAI chat completion (use a small fixture image).
2. Verify the response is generated without error.
3. `admit_count` did not increment.
4. Compare the first decoded token against the P6.3 single-image baseline (`first_token=760`, `max_diff=0.3906`).

### 5.3 Manual smoke checks (recorded in close-out)

- Start the server (`cargo run --bin ironmlx-server`), send 2 concurrent text requests via `curl`, verify both complete with correct content. (Will be serial per 3b-2's one-batch-per-admit design; 3b-3 makes them concurrent.)
- Send a VL request via `curl`, verify works.
- Send a long-prompt request, verify works.
- Verify SSE chunk format with `curl -N` matches what iron-bench v1 expects.

## §6 Acceptance gates

- All 3 new integration tests pass
- Existing regression suite passes unchanged:
  - P6.3 single-image: PASS, `max_diff=0.3906`, `first_token=760`
  - P6.6 logits-match: PASS, `first_token=760`
  - P6.7 chunked-prefill: PASS all chunk_sizes → 760
  - B1-p2.1 batched prefill: PASS, 10/12 argmax bit-id, max_diff ≤ 0.19
  - B1-p2.2 batched decode: PASS, 57/60 argmax bit-id, decode max_diff ≤ 1.62
  - B1-p2.3b-1 scheduler scenarios: PASS, all bit_id=1.0000
- `cargo +nightly fmt --all -- --check`, `clippy -D warnings`, `cargo build --release -p ironmlx`: all clean
- Lib test count: 187 (3b-1 baseline) + 0..1 (depending on unit test inclusion in §5.1)
- Manual smoke: 2 concurrent text requests succeed (serial behavior is fine for 3b-2)

## §7 Estimate

**3–5 working days** (matches 3a's pacing):

- Day 1 — `scheduler_actor.rs` module (struct + cmd enum + driver_loop + spawn helper) + manual smoke that the actor compiles + driver can be spawned with a model handle
- Day 2 — `AppState` integration + `serve_via_scheduler_stream` / `_unary` helpers + chunk formatter extraction + handler routing branch
- Day 3 — Scenario A + B integration tests; debug any SSE format drift
- Day 4 — Scenario C VL routing test + full regression sweep + manual smoke
- Day 5 (buffer) — close-out + review fixes

## §8 Compat sunset table

3b-2 ships three temporary `GS path` fallbacks. Each has an explicit sunset trigger:

| Compat | Reason | Sunset trigger |
| --- | --- | --- |
| GS path for VL requests | `Scheduler::RequestState` has no VL fields; B1-p2.4 adds them | B1-p2.4 lands batched VL — handler routing removes the `has_images` branch |
| GS path for long prompts (prompt_len > chunk_size) | `Scheduler::prefill_admitted` calls `batched_prefill` in one shot — no chunking | "3c+ chunked-prefill phase" (TBD phase number, after 3c per-row offset) ships chunked batched prefill — handler routing removes the length branch |
| Anthropic handler stays on GS | Different SSE wire format (6-event sequence) needs its own refactor | 3b-4 lands — anthropic.rs gets a similar routing branch |
| One-admit-per-batch in driver | Admission window logic deferred | 3b-3 lands — driver loop switches to `select! { admit | tick }` |

Each item is annotated with `// COMPAT(3b-2): sunset in <phase>` in the code at the corresponding decision site.

## §9 Alternatives considered

The brainstorming session locked in 6 choices. Recording the rejected paths:

| Decision | Selected | Rejected alternatives |
| --- | --- | --- |
| Architecture | Actor pattern with dedicated driver task | `Mutex<Scheduler>` in AppState with delayed-prefill (race-prone state machine, locks not reentrant); Plain `Mutex<Scheduler>` + `Mutex<Model>` (no batching, defeats B1-p2.3 goal); Defer to 3b-3 (no actor pattern shipped, can't validate architecture incrementally) |
| Scope | 3b-2 = actor + B=1 swap; 3b-3 = batching; 3b-4 = Anthropic | Single big 3b-2 (high blast radius); Even finer split 3b-2-1/2/3/4 (too granular, ceremony cost > benefit) |
| VL routing | Stay on GS path (sunset @ B1-p2.4) | Return 501 (breaks current users); 503+Retry-After (also breaks); Add VL to scheduler now (out of scope) |
| Long-prompt routing | Stay on GS path (sunset @ 3c+ chunked-prefill phase) | Add chunking to scheduler now (200-300 lines + 2-3d debug — derails 3b-2 skeleton goal); Block long prompts with 413 (breaks users); No threshold check (OOM risk on 4K+ prompts) |
| Trait abstraction | Skip — keep concrete `Qwen35Model` (3b-1 M-3 stays as documented debt) | Sealed trait now (premature — single concrete model); Generic `<M: Model>` (compile-time blow-up); Wait for P5/P6 to land (matches our decision) |
| Lock strategy | Idle: no lock held. Batch: full batch holds lock | Per-step lock acquire-release (GS hogs lock); Two paths fully serialized via single Mutex (defeats batching when 3b-3 lands); Condvar/signal between paths (over-engineered) |

## §10 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-13-b1-p2-3b-1-scheduler-step-design.md`](2026-05-13-b1-p2-3b-1-scheduler-step-design.md)
- Predecessor plan: [`docs/superpowers/plans/2026-05-13-b1-p2-3b-1-scheduler-step.md`](../plans/2026-05-13-b1-p2-3b-1-scheduler-step.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_1_closeout/report.md)
- Current OpenAI handler (to refactor): [`ironmlx/src/core/server/openai.rs`](../../ironmlx/src/core/server/openai.rs)
- Current AppState: [`ironmlx/src/core/server/mod.rs`](../../ironmlx/src/core/server/mod.rs)
- Scheduler API surface: [`ironmlx/src/core/scheduler.rs`](../../ironmlx/src/core/scheduler.rs)
- GS path reference: [`ironmlx/src/core/generate.rs`](../../ironmlx/src/core/generate.rs)
- iron-bench client expectations: [`iron-bench/src/client.rs`](../../iron-bench/src/client.rs)
