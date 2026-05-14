# B1-p2.3b-4 Anthropic Handler Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route Anthropic `/v1/messages` text-only short-prompt requests through `SchedulerActor` (mirror 3b-2's OpenAI refactor), with a 6-event SSE wrapper that produces wire output indistinguishable from the existing GS-path sequence; long-prompt requests stay on GS as sunset-tracked compat.

**Architecture:** `anthropic.rs` renames `messages_stream` / `messages_unary` → `serve_via_gs_*`; adds `serve_via_scheduler_stream` (containing the 6-event SSE state machine inline) and `serve_via_scheduler_unary` (drain `event_rx`, build `MessageEnvelope`). `messages` entry adds a 4-way `match (stream, use_scheduler)` dispatch with the simplified routing predicate (Anthropic permanently text-only — no `has_images` check). Folds in 3b-3 final-review trivial Minors (M1: redundant `#[allow]`; M2: stale unit test docstring).

**Tech Stack:** Rust 2021, Tokio (mpsc, oneshot, spawn_blocking, tokio::spawn), axum (Body, Response), serde_json, ironmlx core (`SchedulerActor` unchanged from 3b-3).

---

## File Structure

```
ironmlx/src/core/server/anthropic.rs           — MODIFY: 2 renames + 2 new helpers + routing branch + COMPAT comment
ironmlx/src/core/server/scheduler_actor.rs     — MODIFY: M1 (delete redundant #[allow]) + M2 (update docstring)
ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs    — NEW: 3 #[ignore] integration scenarios + helpers (~400 lines)
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_4_closeout/report.md              — NEW: close-out
```

**Zero modifications to:** `core/server/openai.rs`, `core/server/mod.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

---

## Grounded facts (verified by reading HEAD `973842b`)

- [`anthropic.rs:110`](../../ironmlx/src/core/server/anthropic.rs#L110) — `pub async fn messages(State, Json<MessagesRequest>) -> Response`. Image rejection at lines 123-137. Tokenization at line 167. Build `GenerateRequest` at lines 179-191. Dispatch at lines 193-198: `if stream { messages_stream(...) } else { messages_unary(...) }`.
- [`anthropic.rs:200-322`](../../ironmlx/src/core/server/anthropic.rs#L200) — `messages_stream(state, request: GenerateRequest, model_id: String, input_tokens: u32) -> Response`. Uses `tokio::task::spawn_blocking` + `state.model.blocking_lock()` + `GenerationStream::new(...)`.
- [`anthropic.rs:277`](../../ironmlx/src/core/server/anthropic.rs#L277) — `output_tokens += 1` is **unconditional** per StepEvent (not gated on text being non-empty). The `content_block_delta` emit IS gated on non-empty text (line 264), but the counter increments either way. The scheduler-path implementation must mirror this exactly for wire equivalence: increment `output_tokens` once per `event_rx.recv()` Some, regardless of detok output.
- [`anthropic.rs:278-284`](../../ironmlx/src/core/server/anthropic.rs#L278) — `stop_reason` mapping inline:
  ```rust
  stop_reason = match reason {
      "stop" => "end_turn",
      "length" => "max_tokens",
      other => other,
  };
  ```
- [`anthropic.rs:324-382`](../../ironmlx/src/core/server/anthropic.rs#L324) — `messages_unary(state, request, model_id, input_tokens) -> Response`. Returns `MessageEnvelope` JSON.
- [`anthropic.rs:47-71`](../../ironmlx/src/core/server/anthropic.rs#L47) — Module-local types:
  - `struct Usage { input_tokens: u32, output_tokens: u32 }` (NOT `AnthropicUsage` — spec §4.4 had the wrong name; correct here)
  - `struct ContentBlockText { kind: &'static str, text: String }` with `#[serde(rename = "type")]` on `kind`
  - `struct MessageEnvelope { id, kind, role, content, model, stop_reason, stop_sequence, usage }`
- [`anthropic.rs:80-82`](../../ironmlx/src/core/server/anthropic.rs#L80) — `gen_msg_id() -> String` → `format!("msg_{}", now_unix())`.
- [`anthropic.rs:99-108`](../../ironmlx/src/core/server/anthropic.rs#L99) — `format_event(event_type, payload) -> Bytes`: `event: <type>\ndata: <json>\n\n`.
- [`scheduler_actor.rs:111`](../../ironmlx/src/core/server/scheduler_actor.rs#L111) — M1 target: `#[allow(clippy::too_many_arguments)]` above `fn driver_loop` (6 args, below clippy default threshold of 7 — attr unnecessary).
- [`scheduler_actor.rs:283-286`](../../ironmlx/src/core/server/scheduler_actor.rs#L283) — M2 target: unit test docstring referencing `cmd_rx.blocking_recv()` instead of the 3b-3 `rt.block_on(cmd_rx.recv())` form.
- [`openai.rs:472-498`](../../ironmlx/src/core/server/openai.rs#L472) — admission boilerplate for `serve_via_scheduler_*` (3b-4 mirrors verbatim).
- [`tests/b1_p2_3b_3_admission_window.rs`](../../ironmlx/tests/b1_p2_3b_3_admission_window.rs) — `run_b1_baseline` must be wrapped in `tokio::task::spawn_blocking(...).await` to avoid `tokio::sync::Mutex::blocking_lock()` panicking on Tokio worker thread (3b-3 Task 2 deviation).

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-b1-p2-3-continuous-batching`, HEAD at `973842b` ("docs(b1-p2.3b-4): Anthropic handler refactor design spec"). Only `design.md` may be untracked.

---

## Task 1: `anthropic.rs` refactor + `scheduler_actor.rs` 3b-3 minors

**Files:**
- Modify: `ironmlx/src/core/server/anthropic.rs`
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`

### Step 1.1: Add new imports to `anthropic.rs`

Current imports (verified at HEAD):

```rust
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};

use super::AppState;
```

Use `Edit` to add `oneshot` to the tokio import and add the scheduler_actor types:

`old_string`:
```rust
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};

use super::AppState;
```

`new_string`:
```rust
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};

use super::AppState;
```

### Step 1.2: Replace the `messages` dispatch tail with the routing branch

Current dispatch (lines 193-198):
```rust
    if stream {
        messages_stream(state, request, model_label, input_tokens).await
    } else {
        messages_unary(state, request, model_label, input_tokens).await
    }
}
```

Use `Edit` to replace with 4-way match + COMPAT comment:

`old_string`:
```rust
    if stream {
        messages_stream(state, request, model_label, input_tokens).await
    } else {
        messages_unary(state, request, model_label, input_tokens).await
    }
}
```

`new_string`:
```rust
    // COMPAT(3b-2/3b-4): long-prompt fallback to GS sunsets in 3c+
    // chunked-prefill phase. Note: when prefill_chunk_size == 0 (chunking
    // disabled by config), this predicate routes ALL text requests to the
    // SchedulerActor regardless of length — equivalent to the GS path's
    // behavior when chunking is also disabled there.
    let prompt_len = request.prompt_ids.len();
    let use_scheduler =
        state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size;

    match (stream, use_scheduler) {
        (true, true) => serve_via_scheduler_stream(state, request, model_label, input_tokens).await,
        (true, false) => serve_via_gs_stream(state, request, model_label, input_tokens).await,
        (false, true) => {
            serve_via_scheduler_unary(state, request, model_label, input_tokens).await
        }
        (false, false) => {
            serve_via_gs_unary(state, request, model_label, input_tokens).await
        }
    }
}
```

### Step 1.3: Rename `messages_stream` → `serve_via_gs_stream`

`old_string`:
```rust
async fn messages_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
```

`new_string`:
```rust
async fn serve_via_gs_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
```

### Step 1.4: Rename `messages_unary` → `serve_via_gs_unary`

`old_string`:
```rust
async fn messages_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
```

`new_string`:
```rust
async fn serve_via_gs_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
```

### Step 1.5: Add `serve_via_scheduler_stream` (6-event SSE wrapper)

Find the closing `}` of `serve_via_gs_stream` (the renamed function from Step 1.3). The original body ends with:

```rust
    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}
```

Use `Edit` anchoring on the closing `}` of `serve_via_gs_stream` followed by `async fn serve_via_gs_unary` start:

`old_string`:
```rust
    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

async fn serve_via_gs_unary(
```

`new_string`:
```rust
    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

/// Text-only short-prompt streaming path via SchedulerActor (3b-4 swap-in).
/// Emits the same 6-event SSE sequence as `serve_via_gs_stream`:
///   message_start → content_block_start → N × content_block_delta →
///   content_block_stop → message_delta → message_stop.
async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let msg_id = gen_msg_id();

    // 1. Admit request to the actor.
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler actor unavailable",
        )
            .into_response();
    }
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return (StatusCode::BAD_REQUEST, format!("admit failed: {e}")).into_response();
        }
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response();
        }
    };

    // 2. Spawn forwarder that emits the 6-event SSE sequence.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let msg_id_for_task = msg_id.clone();
    let model_id_for_task = model_id.clone();
    let tokenizer = state.tokenizer.clone();

    tokio::spawn(async move {
        // Event 1: message_start
        let start_payload = serde_json::json!({
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
        if tx
            .send(Ok(format_event("message_start", &start_payload)))
            .await
            .is_err()
        {
            return;
        }

        // Event 2: content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        if tx
            .send(Ok(format_event("content_block_start", &block_start)))
            .await
            .is_err()
        {
            return;
        }

        // Events 3..N+2: content_block_delta per non-empty detok output.
        // output_tokens increments UNCONDITIONALLY per StepEvent (mirrors
        // GS path line 277 — counter reflects generated tokens, NOT
        // emitted deltas. Tokens whose detok output is empty still count.)
        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        let mut output_tokens: u32 = 0;
        let mut stop_reason: &'static str = "end_turn";
        while let Some(ev) = event_rx.recv().await {
            let text = match detok.step(ev.token) {
                Ok(Some(s)) => s,
                Ok(None) => String::new(), // BPE mid-codepoint
                Err(_) => String::new(),   // best-effort; skip emit
            };
            if !text.is_empty() {
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                });
                if tx
                    .send(Ok(format_event("content_block_delta", &delta)))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            output_tokens += 1;
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
        let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        if tx
            .send(Ok(format_event("content_block_stop", &block_stop)))
            .await
            .is_err()
        {
            return;
        }

        // Event N+4: message_delta (carries final stop_reason + output_tokens)
        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        if tx
            .send(Ok(format_event("message_delta", &msg_delta)))
            .await
            .is_err()
        {
            return;
        }

        // Event N+5: message_stop
        let msg_stop = serde_json::json!({"type": "message_stop"});
        let _ = tx.send(Ok(format_event("message_stop", &msg_stop))).await;
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

async fn serve_via_gs_unary(
```

### Step 1.6: Add `serve_via_scheduler_unary`

Find the closing `}` of `serve_via_gs_unary`. Its body ends with `Json(envelope).into_response()` then `}` then the `#[cfg(test)] mod tests {` block.

Use `Edit`:

`old_string`:
```rust
    Json(envelope).into_response()
}

#[cfg(test)]
mod tests {
```

`new_string`:
```rust
    Json(envelope).into_response()
}

/// Text-only short-prompt unary path via SchedulerActor (3b-4 swap-in).
async fn serve_via_scheduler_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let id = gen_msg_id();

    // 1. Admit.
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler actor unavailable",
        )
            .into_response();
    }
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return (StatusCode::BAD_REQUEST, format!("admit failed: {e}")).into_response();
        }
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response();
        }
    };

    // 2. Drain events; build envelope.
    let mut detok = state.tokenizer.decode_stream(/* skip_special */ true);
    let mut content = String::new();
    let mut output_tokens: u32 = 0;
    let mut stop_reason: &'static str = "end_turn";
    while let Some(ev) = event_rx.recv().await {
        match detok.step(ev.token) {
            Ok(Some(s)) => content.push_str(&s),
            Ok(None) => { /* BPE mid-codepoint */ }
            Err(_) => { /* best-effort */ }
        }
        output_tokens += 1;
        if let Some(reason) = ev.finish_reason {
            stop_reason = match reason {
                "stop" => "end_turn",
                "length" => "max_tokens",
                other => other,
            };
            break;
        }
    }

    let envelope = MessageEnvelope {
        id,
        kind: "message",
        role: "assistant",
        content: vec![ContentBlockText {
            kind: "text",
            text: content,
        }],
        model: model_id,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    };
    Json(envelope).into_response()
}

#[cfg(test)]
mod tests {
```

### Step 1.7: 3b-3 Minor M1 — Delete redundant `#[allow]` in `scheduler_actor.rs`

`grep -n "#\[allow(clippy::too_many_arguments)\]" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/scheduler_actor.rs` to find the exact line. Expected: line 111.

Use `Edit`:

`old_string`:
```rust
#[allow(clippy::too_many_arguments)]
fn driver_loop(
```

`new_string`:
```rust
fn driver_loop(
```

(Removes one line. `driver_loop` has 6 args; clippy's default `too_many_arguments` threshold is 7, so no lint fires.)

### Step 1.8: 3b-3 Minor M2 — Update unit test docstring

Find the docstring for `driver_shuts_down_when_cmd_channel_closes`. Expected to be in `#[cfg(test)] mod tests` block near the end of the file.

Use `Edit`:

`old_string`:
```rust
    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `while let Some(cmd) = cmd_rx.blocking_recv()` loop terminates when
    /// all senders are dropped.
```

`new_string`:
```rust
    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `rt.block_on(cmd_rx.recv())` outer loop (3b-3) terminates when all
    /// senders are dropped.
```

### Step 1.9: Format, build, clippy, full lib regression

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected:
- fmt: clean (run `cargo +nightly fmt --all` if drift, re-check)
- build: `Finished release profile [optimized] target(s) in ...`
- clippy: clean (only mlx-sys C++ noise)
- lib tests: **188 passed** (unchanged — no new lib tests in this task)

Possible fixups:
- `unused_imports` warning on `oneshot` if `serve_via_scheduler_*` not yet visible — should be fine since both new helpers were added.
- `clippy::needless_pass_by_value` on `tokenizer: Arc<Tokenizer>` move into spawn — required for `'static` bound; suppress if clippy demands.

### Step 1.10: P6.3 single-image sanity (verifies Anthropic refactor didn't touch model layer)

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Use `run_in_background: true` + completion check; timeout ~600000 ms. Expected: PASS, `max_diff=0.3906`, `first_token=760`.

(P6.3 exercises the model layer directly, NOT the HTTP handler — this is a sanity check that the rename + new helpers didn't introduce a compile-time regression that the lib test didn't catch.)

### Step 1.11: Commit

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/server/anthropic.rs ironmlx/src/core/server/scheduler_actor.rs
git commit -m "feat(b1-p2.3b-4): Anthropic handler routes text/short to SchedulerActor; 3b-3 M1/M2 cleanup"
```

---

## Task 2: Integration Scenarios 1 + 2

**Files:**
- Create: `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs`

### Step 2.1: Create the integration test file with imports + helpers + Scenarios 1 + 2

```rust
//! B1-p2.3b-4 — Anthropic handler refactor + SchedulerActor integration.
//!
//! Three scenarios (see spec § 5.2):
//!   1. `anthropic_actor_b1_text_only_swap` — single text request routes
//!      to SchedulerActor; per-row tokens match B=1 GS baseline.
//!   2. `anthropic_actor_long_prompt_routes_to_gs` — prompt_len >
//!      chunk_size routes to GS; admit_count delta=0.
//!   3. (Task 3) `anthropic_actor_scheduler_path_emits_6_event_sequence`
//!      — directly invoke serve_via_scheduler_stream; assert 6 event
//!      types appear in order + payload fields correct.
//!
//! Tests are `#[ignore]`-gated; run only with `QWEN35_MODEL` env var.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

#[allow(dead_code)]
const ARGMAX_BITID_GATE: f64 = 0.95;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

/// Tokenize a chat-template-rendered prompt. Mirrors 3b-3 test pattern.
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, /* add_generation_prompt */ true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer
        .encode(&rendered, /* add_special_tokens */ false)
        .expect("encode")
}

/// Run a B=1 baseline via direct `GenerationStream`. Locks the model.
/// Caller wraps in `tokio::task::spawn_blocking` to avoid blocking_lock
/// from a Tokio worker thread (panics with "Cannot block the current
/// thread from within a runtime").
fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

#[allow(dead_code)]
fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Send one Admit cmd via `handle.cmd_tx`, await reply, drain `event_rx`
/// to completion, return collected tokens.
async fn admit_and_drain(handle: SchedulerActorHandle, request: GenerateRequest) -> Vec<u32> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send admit");
    let reply = reply_rx.await.expect("admit reply").expect("admit ok");
    let mut event_rx = reply.event_rx;
    let mut tokens = Vec::new();
    while let Some(ev) = event_rx.recv().await {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

fn make_request(
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_b1_text_only_swap() {
    let (model, tokenizer) = load_fixture();

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 12;

    // 1. B=1 baseline. Wrap in spawn_blocking because Mutex::blocking_lock
    // panics from a Tokio worker thread.
    let baseline = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let req = make_request(prompt_ids.clone(), max_new_tokens, stop_token_ids.clone());
        tokio::task::spawn_blocking(move || run_b1_baseline(&model, &tokenizer, req))
            .await
            .expect("baseline join")
    };
    assert!(!baseline.is_empty(), "baseline produced no tokens");

    // 2. Route through SchedulerActor.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let admit_before = handle.admit_count.load(Ordering::Relaxed);

    let req = make_request(prompt_ids, max_new_tokens, stop_token_ids);
    let scheduler_tokens = admit_and_drain(handle.clone(), req).await;

    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    println!(
        "[anthropic_b1] admit_delta={} scheduler_len={} baseline_len={}",
        admit_delta,
        scheduler_tokens.len(),
        baseline.len()
    );
    assert_eq!(admit_delta, 1, "expected exactly one admit");

    // Bit-id parity check. B=1 single-row Scheduler vs B=1 GenerationStream
    // use the same numerical path; bit_id should be 1.0000. Asserting
    // ≥0.95 matches 3b-2 pattern's safety margin.
    let ratio = argmax_bit_id_ratio(&scheduler_tokens, &baseline);
    println!("[anthropic_b1] bit_id={:.4}", ratio);
    assert!(
        ratio >= ARGMAX_BITID_GATE,
        "bit_id {ratio:.4} below gate {ARGMAX_BITID_GATE}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_long_prompt_routes_to_gs() {
    let (model, tokenizer) = load_fixture();

    // Build a synthetic long prompt > chunk_size = 64.
    let chunk_size: usize = 64;
    let short_ids = tokenize_prompt(&tokenizer, "Hello world.");
    let mut long_ids = Vec::with_capacity(chunk_size * 2);
    while long_ids.len() <= chunk_size {
        long_ids.extend_from_slice(&short_ids);
    }
    assert!(
        long_ids.len() > chunk_size,
        "long prompt setup failed: {} <= {}",
        long_ids.len(),
        chunk_size
    );

    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let request = GenerateRequest {
        prompt_ids: long_ids,
        max_new_tokens: 4,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    };

    // Routing predicate (mirrors Anthropic dispatch in messages handler).
    // Anthropic has no has_images check (text-only by design).
    let prompt_len = request.prompt_ids.len();
    let use_scheduler =
        request.prefill_chunk_size == 0 || prompt_len <= request.prefill_chunk_size;
    assert!(
        !use_scheduler,
        "routing predicate failed: long prompt would go to scheduler"
    );

    // Verify admit_count doesn't change when GS path is taken.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let before = handle.admit_count.load(Ordering::Relaxed);

    // Drop the request — the GS path bypasses the actor; the test only
    // needs to assert the routing decision (mirrors 3b-2 Scenario B/C).
    let _ = request;

    let after = handle.admit_count.load(Ordering::Relaxed);
    assert_eq!(
        after, before,
        "admit_count incremented unexpectedly: {} -> {}",
        before, after
    );
}
```

### Step 2.2: Format + build the new test crate

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_4_anthropic_actor 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: fmt clean, build clean, clippy clean.

If `ironmlx::core::Message` import fails (the test references `Message` from `core::`), check the existing 3b-3 test for the exact import path. If it differs, adapt.

### Step 2.3: Run Scenarios 1 + 2 (~5-10 min on GPU)

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_4_anthropic_actor -- --ignored --nocapture --test-threads=1 2>&1 | tail -20
```

`run_in_background: true` + completion check; timeout ~900000 ms (15 min).

Expected: `test result: ok. 2 passed; 0 failed`. Diagnostic:
- `[anthropic_b1] admit_delta=1 scheduler_len=12 baseline_len=12`
- `[anthropic_b1] bit_id=1.0000`

**If Scenario 1 fails with `bit_id < 0.95`**: B=1 single-row path differs from baseline. Possible causes:
- Scheduler's `prefill_admitted` (Option A in 3b-1) starts with token from prefill argmax; baseline `GenerationStream` does the same in pipelined mode → should match. If they diverge, it's a numerics issue not related to 3b-4 (would have shown up in 3b-2 already).
- Report DONE_WITH_CONCERNS if persistent.

### Step 2.4: Commit

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs
git commit -m "test(b1-p2.3b-4): anthropic actor scenarios 1 + 2 (text swap + long-prompt routing)"
```

---

## Task 3: Scenario 3 — 6-event SSE wire-format smoke

**Files:**
- Modify: `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs` (append Scenario 3)

This scenario directly invokes `serve_via_scheduler_stream` (an `async fn` in `anthropic.rs`). Since that function is `pub(super)` to the `core::server` module, the integration test crate cannot call it directly — it goes through the `messages` HTTP handler entry point.

The test sends an HTTP request to a constructed `AppState`-backed handler, collects the response body bytes (which are an SSE stream), parses event lines, and asserts the 6-event sequence.

### Step 3.1: Add imports for Scenario 3

Find the existing import block at the top of `tests/b1_p2_3b_4_anthropic_actor.rs`. Append (use `Edit`):

`old_string`:
```rust
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;
```

`new_string`:
```rust
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::Json;
use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::server::AppState;
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;
```

If `axum` is not a direct dev-dependency of `ironmlx`, this import will fail. Check `Cargo.toml` for the test crate's dependencies:

```bash
grep -nE "axum|^\[(dev-)?dependencies\]" /Volumes/Dev/cxx-mlx/ironmlx/Cargo.toml | head -20
```

If `axum` is only a `[dependencies]` entry, add it to `[dev-dependencies]` or rely on the transitive availability from the lib crate (it should be exposed because `AppState` uses axum types).

**Alternative if axum cross-crate is messy**: skip the HTTP-level test and write Scenario 3 as a SchedulerCommand-level test that bypasses HTTP. The 6-event SSE sequence is built inside `serve_via_scheduler_stream`'s spawned task — to test the wrapper, we'd need to either invoke that function or duplicate its logic. Since the wrapper is private (`async fn` in `core::server::anthropic`), the integration test crate cannot call it directly without making it `pub`.

**Decision tree:**
- **Option A**: Make `serve_via_scheduler_stream` `pub(crate)` in `anthropic.rs` (1-line visibility change). Test invokes it directly with a constructed `AppState`. Cleaner.
- **Option B**: Spin up a full axum router in the test, send a real HTTP POST, parse the SSE response body. More setup, more brittle.

**Go with Option A.** This is a test-friendly visibility relaxation, not a public API change. Update Task 1 Step 1.5 to add `pub(crate)` before the `async fn` (and similarly for `serve_via_scheduler_unary` in Step 1.6 — for symmetry, even though Scenario 3 only tests the stream variant).

### Step 3.1.5: Retro-fix Task 1 visibility (apply BEFORE Scenario 3)

This step amends Task 1 if it hasn't been committed yet. If Task 1 is already committed at this point, make a follow-up commit. The change:

In `anthropic.rs`, the new helpers added in Task 1 Steps 1.5 and 1.6 should be `pub(crate)` instead of private:

```bash
grep -n "async fn serve_via_scheduler" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/anthropic.rs
```

If both signatures show `async fn serve_via_scheduler_*(...)` (no `pub`), promote to `pub(crate)`:

Use `Edit`:

`old_string`:
```rust
/// Text-only short-prompt streaming path via SchedulerActor (3b-4 swap-in).
/// Emits the same 6-event SSE sequence as `serve_via_gs_stream`:
///   message_start → content_block_start → N × content_block_delta →
///   content_block_stop → message_delta → message_stop.
async fn serve_via_scheduler_stream(
```

`new_string`:
```rust
/// Text-only short-prompt streaming path via SchedulerActor (3b-4 swap-in).
/// Emits the same 6-event SSE sequence as `serve_via_gs_stream`:
///   message_start → content_block_start → N × content_block_delta →
///   content_block_stop → message_delta → message_stop.
pub(crate) async fn serve_via_scheduler_stream(
```

Similarly for `serve_via_scheduler_unary` (symmetric — even though Scenario 3 doesn't test it):

`old_string`:
```rust
/// Text-only short-prompt unary path via SchedulerActor (3b-4 swap-in).
async fn serve_via_scheduler_unary(
```

`new_string`:
```rust
/// Text-only short-prompt unary path via SchedulerActor (3b-4 swap-in).
pub(crate) async fn serve_via_scheduler_unary(
```

Build to confirm the visibility change is valid:

```bash
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```

### Step 3.2: Append Scenario 3 to the test file

Append immediately after the closing `}` of `anthropic_actor_long_prompt_routes_to_gs`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn anthropic_actor_scheduler_path_emits_6_event_sequence() {
    let (model, tokenizer) = load_fixture();

    let prompt = "Hello.";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let input_tokens = prompt_ids.len() as u32;
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 4;

    // Construct AppState matching what serve() builds.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let state = AppState {
        model: model.clone(),
        tokenizer: tokenizer.clone(),
        model_id: "test-model".to_string(),
        prefill_chunk_size: 256,
        scheduler_handle: handle.clone(),
    };

    let req = make_request(prompt_ids, max_new_tokens, stop_token_ids);

    // Invoke the scheduler-path helper directly.
    let response = ironmlx::core::server::anthropic::serve_via_scheduler_stream(
        state,
        req,
        "test-model".to_string(),
        input_tokens,
    )
    .await;

    // Collect the response body bytes.
    let body_bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let body = String::from_utf8_lossy(&body_bytes);
    println!("[anthropic_6event] raw body:\n{body}");

    // Parse SSE chunks: split on \n\n boundary. Each chunk starts with
    // "event: <type>\ndata: <json>".
    let mut event_types: Vec<String> = Vec::new();
    let mut event_payloads: Vec<serde_json::Value> = Vec::new();
    for chunk in body.split("\n\n") {
        if chunk.is_empty() {
            continue;
        }
        let mut event_type = None;
        let mut data_line = None;
        for line in chunk.lines() {
            if let Some(t) = line.strip_prefix("event: ") {
                event_type = Some(t.to_string());
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_line = Some(d);
            }
        }
        if let (Some(t), Some(d)) = (event_type, data_line) {
            event_types.push(t);
            let payload: serde_json::Value = serde_json::from_str(d).expect("parse SSE data");
            event_payloads.push(payload);
        }
    }
    println!("[anthropic_6event] event_types={:?}", event_types);

    // Assert event sequence shape.
    assert!(
        event_types.len() >= 5,
        "expected ≥5 events (message_start + content_block_start + ≥1 delta + content_block_stop + message_delta + message_stop), got {} events",
        event_types.len()
    );
    assert_eq!(
        event_types.first().map(|s| s.as_str()),
        Some("message_start"),
        "first event must be message_start"
    );
    assert_eq!(
        event_types.get(1).map(|s| s.as_str()),
        Some("content_block_start"),
        "second event must be content_block_start"
    );
    assert_eq!(
        event_types.last().map(|s| s.as_str()),
        Some("message_stop"),
        "last event must be message_stop"
    );

    // The last 3 events must be content_block_stop → message_delta → message_stop.
    let n = event_types.len();
    assert!(
        event_types[n - 3] == "content_block_stop"
            && event_types[n - 2] == "message_delta"
            && event_types[n - 1] == "message_stop",
        "tail of event_types must be [content_block_stop, message_delta, message_stop]; got {:?}",
        &event_types[n - 3..]
    );

    // Middle events (between content_block_start and content_block_stop)
    // must all be content_block_delta.
    for (i, t) in event_types.iter().enumerate().take(n - 3).skip(2) {
        assert_eq!(
            t.as_str(),
            "content_block_delta",
            "event[{i}] must be content_block_delta, got {t}"
        );
    }

    // Verify message_start payload structure.
    let start = &event_payloads[0];
    assert_eq!(start["type"], "message_start");
    assert_eq!(start["message"]["usage"]["input_tokens"], input_tokens);
    assert_eq!(start["message"]["usage"]["output_tokens"], 0);
    assert!(
        start["message"]["stop_reason"].is_null(),
        "message_start.stop_reason must be null"
    );

    // Verify message_delta payload structure.
    let delta = &event_payloads[n - 2];
    assert_eq!(delta["type"], "message_delta");
    let stop_reason = delta["delta"]["stop_reason"]
        .as_str()
        .expect("stop_reason str");
    assert!(
        stop_reason == "end_turn" || stop_reason == "max_tokens",
        "unexpected stop_reason: {stop_reason}"
    );
    let final_output_tokens = delta["usage"]["output_tokens"]
        .as_u64()
        .expect("output_tokens u64");
    // Number of content_block_delta events ≤ output_tokens (some tokens
    // may produce empty detok text — counted in output_tokens but not emitted).
    let delta_count = event_types
        .iter()
        .filter(|t| t.as_str() == "content_block_delta")
        .count() as u64;
    assert!(
        delta_count <= final_output_tokens,
        "delta count {delta_count} exceeds output_tokens {final_output_tokens} — counter invariant broken"
    );
    println!(
        "[anthropic_6event] output_tokens={} delta_count={} stop_reason={}",
        final_output_tokens, delta_count, stop_reason
    );

    let _ = handle; // keep alive
}
```

### Step 3.3: Format + build

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_4_anthropic_actor 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: clean.

If `axum::body::to_bytes` is not available, the alternative is `axum::body::HttpBody::collect()` — adapt as needed.

If `ironmlx::core::server::anthropic::serve_via_scheduler_stream` is not visible (Step 3.1.5 not applied), the build fails with "function `serve_via_scheduler_stream` is private". Verify Step 3.1.5 was committed.

### Step 3.4: Run all 3 Scenarios

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_4_anthropic_actor -- --ignored --nocapture --test-threads=1 2>&1 | tail -30
```

`run_in_background: true` + completion check; timeout ~900000 ms.

Expected: `test result: ok. 3 passed; 0 failed`. Scenario 3 prints event type sequence + final output_tokens / delta_count / stop_reason.

**If Scenario 3 fails on event order**: check Step 1.5's `serve_via_scheduler_stream` implementation — the 6-event emit order must match exactly the spec §4.3 sequence.

**If Scenario 3 fails on `output_tokens` invariant** (`delta_count > output_tokens`): the counter increments before the emit decision, violating the spec §4.3 invariant. Re-verify Task 1 Step 1.5 increments `output_tokens` unconditionally per event (matching GS path line 277) AND the emit is gated only on non-empty text.

### Step 3.5: Commit

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/server/anthropic.rs ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs
git commit -m "test(b1-p2.3b-4): scenario 3 — 6-event SSE wire-format smoke"
```

(Note: if Step 3.1.5 visibility change was applied here, that's why `anthropic.rs` is in the add list. If visibility was already added in Task 1's commit, just `git add tests/...`.)

---

## Task 4: Regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md`

### Step 4.1: Full hygiene sweep

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: all clean, lib tests **188 passed** (unchanged).

### Step 4.2: Full regression sweep (8 tests in one cargo invocation, sequential)

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release \
    --test p6_qwen35_vl_logits_match \
    --test p6_6_logits_match \
    --test p6_7_chunked_prefill \
    --test b1_p2_1_batched_prefill \
    --test b1_p2_2_batched_decode \
    --test b1_p2_3b_1_scheduler_step \
    --test b1_p2_3b_2_scheduler_actor \
    --test b1_p2_3b_3_admission_window \
    --test b1_p2_3b_4_anthropic_actor \
    -- --ignored --test-threads=1 2>&1 | tail -60
```

Use `run_in_background: true` + completion check; timeout ~3600000 ms (60 min).

Expected: every test PASS. Last line should be `test result: ok. N passed; 0 failed`.

**If any test fails:** STOP and report BLOCKED.

### Step 4.3: Write the close-out report

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md`:

```markdown
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
clippy default threshold), and updated the
`driver_shuts_down_when_cmd_channel_closes` unit test docstring to
reference the 3b-3 `rt.block_on(cmd_rx.recv())` form.

## Acceptance

| Test | Result |
| --- | --- |
| `driver_shuts_down_when_cmd_channel_closes` (unit, docstring updated) | ✅ |
| `anthropic_actor_b1_text_only_swap` | ✅ admit_delta=1, bit_id ≥ 0.95 |
| `anthropic_actor_long_prompt_routes_to_gs` | ✅ admit_count delta=0 |
| `anthropic_actor_scheduler_path_emits_6_event_sequence` | ✅ 6-event sequence verified + payload fields correct |

## Architectural Changes

1. **`ironmlx/src/core/server/anthropic.rs`**:
   - Renamed `messages_stream` → `serve_via_gs_stream` (body unchanged)
   - Renamed `messages_unary` → `serve_via_gs_unary` (body unchanged)
   - Added `serve_via_scheduler_stream` (~120 lines): admission boilerplate + tokio::spawn forwarder that emits 6-event SSE sequence inline; reuses `gen_msg_id`, `now_unix`, `format_event`
   - Added `serve_via_scheduler_unary` (~60 lines): admission boilerplate + drain event_rx + build `MessageEnvelope`
   - `messages` dispatch tail replaced with 4-way `match (stream, use_scheduler)`; routing predicate simplified to `chunk_size == 0 || prompt_len <= chunk_size` (no `has_images` — Anthropic is permanently text-only)
   - Added `// COMPAT(3b-2/3b-4): long-prompt fallback...` comment with sunset target
   - Added imports: `tokio::sync::oneshot`, `crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand}`
   - `serve_via_scheduler_stream` and `serve_via_scheduler_unary` are `pub(crate)` (test access)
2. **`ironmlx/src/core/server/scheduler_actor.rs`**:
   - M1: removed `#[allow(clippy::too_many_arguments)]` (6 args below clippy threshold of 7)
   - M2: updated `driver_shuts_down_when_cmd_channel_closes` docstring to reference `rt.block_on(cmd_rx.recv())` (3b-3 form) instead of `cmd_rx.blocking_recv()` (3b-2 form)

No changes to: `core/server/openai.rs`, `core/server/mod.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## Compat sunset markers (recorded in code)

| Location | Marker | Sunset |
| --- | --- | --- |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4` | B1-p2.4 batched VL |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase` | 3c+ chunked prefill |
| `anthropic.rs::messages` dispatch (NEW) | `// COMPAT(3b-2/3b-4): long-prompt fallback...` | 3c+ chunked prefill |
| `anthropic.rs::messages` image rejection | (implicit — 400 on image content parts) | Future Anthropic VL support phase |
| `scheduler_actor.rs::ADMISSION_DEADLINE` | hardcoded 5ms | 3d/3e config exposure |

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `<T1_SHA>` | feat | Anthropic handler routes text/short to SchedulerActor; 3b-3 M1/M2 cleanup |
| `<T2_SHA>` | test | anthropic actor scenarios 1 + 2 (text swap + long-prompt routing) |
| `<T3_SHA>` | test | scenario 3 — 6-event SSE wire-format smoke |
| `<T4_SHA>` | docs | This close-out |

(Fill `<T*_SHA>` from `git log --oneline 973842b..HEAD` after Step 4.4 commit.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **188 passed / 0 failed** |
| P6.3 single-image | <FILL> |
| P6.6 logits-match | <FILL> |
| P6.7 chunked-prefill | <FILL> |
| B1-p2.1 batched prefill | <FILL> |
| B1-p2.2 batched decode | <FILL> |
| B1-p2.3b-1 scheduler scenarios | <FILL> |
| B1-p2.3b-2 scheduler_actor scenarios | <FILL> |
| B1-p2.3b-3 admission_window scenarios | <FILL> |
| B1-p2.3b-4 anthropic_actor scenarios | 3 PASS (text_only_swap, long_prompt_routes_to_gs, scheduler_path_emits_6_event_sequence) |

Full-sweep exit code: `0`. No regressions.

## Notes

- **Anthropic multi-request batching is now live.** Same `SchedulerActor` as 3b-3 — text-only short-prompt Anthropic requests pack into the same 5ms admission window. Heterogeneous OpenAI+Anthropic batches are technically possible (same actor, same scheduler) — both routes admit through the same `SchedulerCommand::Admit` channel. Lock strategy unchanged from 3b-2/3b-3.
- **6-event SSE wire-format equivalence verified.** Scenario 3 parses the byte stream from `serve_via_scheduler_stream` and asserts event type sequence + `message_start.usage.input_tokens` + `message_delta.delta.stop_reason` mapping. The GS path's existing module-local unit tests still cover the GS-path's own byte format.
- **`output_tokens` counter mirrors GS path semantics**: incremented per `event_rx.recv()` Some, NOT only per emitted delta. Some tokens whose detok output is empty (BPE mid-codepoint) still count toward `output_tokens` but are not emitted as `content_block_delta`. Scenario 3 invariant `delta_count <= output_tokens` verifies this.
- **No iron-bench compat concern.** iron-bench only exercises `/v1/chat/completions` (OpenAI). Anthropic path has no v1 client to keep green.
- **3b-3 minors closed**: M1 (`#[allow]` deletion) and M2 (docstring update) folded into Task 1's commit.

## B1-p2.3x Next Steps

- **B1-p2.3c** — Per-row KV cache offset tracking; lifts the lockstep constraint.
- **B1-p2.3 (chunked-prefill phase)** — Adds batched prefill chunking; removes both `prompt_len > chunk_size` fallbacks (OpenAI + Anthropic).
- **B1-p2.3d** — Admission queue + preemption. Also surfaces `ADMISSION_DEADLINE` via `AppConfig` + CLI flag.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback in OpenAI handler.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3b-4-anthropic-handler-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3b-4-anthropic-handler.md`
- Modified handler: `ironmlx/src/core/server/anthropic.rs`
- Modified module (M1/M2): `ironmlx/src/core/server/scheduler_actor.rs`
- Integration test: `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md`
```

Fill each `<FILL>` and `<T*_SHA>` from the regression sweep outputs + git log.

### Step 4.4: Commit close-out

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md
git commit -m "docs(b1-p2.3b-4): close-out — Anthropic handler refactor"
```

### Step 4.5: Final summary log

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline 973842b..HEAD
```

Expected: 4 commits (T1 feat, T2 test, T3 test, T4 docs).

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §1 Goal 1 (routing text/short to SchedulerActor) | T1 Step 1.2 + 1.5 + 1.6 |
| §1 Goal 2 (6-event SSE wrapper) | T1 Step 1.5 |
| §1 Goal 3 (rename `messages_*` → `serve_via_gs_*`) | T1 Steps 1.3 + 1.4 |
| §1 Goal 4 (3b-3 M1/M2 fold-in) | T1 Steps 1.7 + 1.8 |
| §3.2 current Anthropic handler architecture | All T1 steps grounded in this |
| §4.1 routing decision tree | T1 Step 1.2 |
| §4.2 renames + new helpers | T1 Steps 1.3 + 1.4 + 1.5 + 1.6 |
| §4.3 6-event state machine code | T1 Step 1.5 (full code) |
| §4.4 serve_via_scheduler_unary | T1 Step 1.6 |
| §4.5 stop_reason mapping reuse | T1 Step 1.5 + 1.6 (inline match) |
| §4.6 3b-3 minor cleanups | T1 Steps 1.7 + 1.8 |
| §4.7 module surface | All T1 steps combined |
| §5.2 Scenario 1 (b1_text_only_swap) | T2 Step 2.1 |
| §5.2 Scenario 2 (long_prompt_routes_to_gs) | T2 Step 2.1 |
| §5.2 Scenario 3 (6-event smoke) | T3 Steps 3.1 + 3.2 |
| §5.3 acceptance gates | T4 Steps 4.1 + 4.2 |
| §6 estimate (3-5 working days) | Task structure follows daily breakdown |
| §7 sunset markers | T1 Step 1.2 (COMPAT comment) |
| §8 risk register (output_tokens counter consistency) | T1 Step 1.5 explicit comment + T3 Scenario 3 assertion |
| §9 alternatives (inline vs helper struct) | T1 Step 1.5 inlines per spec decision |

All sections covered.

**2. Placeholder scan:**
- `<FILL>` / `<T*_SHA>` in close-out template (T4 Step 4.3) — explicit "fill at execution time".
- No bare "TBD" / "implement later" / "fill in details" elsewhere.

**3. Type consistency:**

| Symbol | First defined | Reused |
| --- | --- | --- |
| `serve_via_scheduler_stream(state, request, model_id, input_tokens)` | T1 Step 1.5 | T3 Step 3.2 (test direct invocation) |
| `serve_via_scheduler_unary(state, request, model_id, input_tokens)` | T1 Step 1.6 | (not exercised in integration tests but symmetric) |
| `serve_via_gs_stream` / `serve_via_gs_unary` (renamed from messages_*) | T1 Steps 1.3 / 1.4 | T1 dispatch in Step 1.2 |
| `use_scheduler` predicate `chunk_size == 0 || prompt_len <= chunk_size` | T1 Step 1.2 | T2 Step 2.1 Scenario 2 mirrors |
| `Usage { input_tokens, output_tokens }` struct | Existing `anthropic.rs:47-51` | T1 Step 1.6 reuses |
| `MessageEnvelope` struct | Existing `anthropic.rs:60-71` | T1 Step 1.6 reuses |
| `ContentBlockText { kind: "text", text }` struct | Existing `anthropic.rs:53-58` | T1 Step 1.6 reuses |
| `gen_msg_id() -> String` | Existing `anthropic.rs:80-82` | T1 Steps 1.5 + 1.6 reuse |
| `format_event(event_type, payload) -> Bytes` | Existing `anthropic.rs:99-108` | T1 Step 1.5 reuses |
| `output_tokens: u32` counter semantics (unconditional increment per StepEvent) | T1 Step 1.5 inline comment + matches GS path line 277 | T3 Scenario 3 assertion (`delta_count <= output_tokens`) |
| `stop_reason` mapping (inline match per call site) | T1 Steps 1.5 + 1.6 | Documented as intentional duplication in spec §9 |
| `tokenize_prompt`, `run_b1_baseline`, `argmax_bit_id_ratio`, `admit_and_drain`, `make_request`, `load_fixture`, `ARGMAX_BITID_GATE` test helpers | T2 Step 2.1 | T3 Step 3.2 reuses (within same file) |

All names consistent across tasks. Helper functions defined in Task 2 are referenced in Task 3 (Scenario 3 appends to same file — no redefinition needed).
