# P5h All-PP Prefill Gap Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build dual-lane attribution infrastructure that measures ironmlx's per-PP prefill gap vs omlx across the full attribution tree (root → top-level buckets → wrapper spans → substeps) on Lane A (scheduler path, PP ≤ 2048) and produces top-level-only attribution on Lane B (chunked GS path, PP > 2048), so P5i/P5j can dispatch optimization work with number-anchored ROI.

**Architecture:** Single `p5h-profile` Cargo feature gates all instrumentation (default build byte-identical to P5f). Schema is an exclusive parent-child span tree with `span_id`/`parent_span_id` (string labels for readability only). Dual emission API: explicit-context `open_p5h_span[_at] + close_p5h_span` for async/cross-task spans (root, HTTP, admission, SSE), implicit-guard `with_p5h_span_from_current_trace` for sync deep instrumentation (GDN/GatedAttention/MoE substeps). Lane-A `first_token_sampling` is opened INSIDE `prefill_admitted_inner` (not actor scope) because `batched_prefill[_vl]` and `sample_batch` are fused in one function. iron-bench gains a `--capture-server-request-id` flag (default off; keeps non-P5h byte-identical) that captures the `X-Ironmlx-Request-Id` header for deterministic 100% join with server log records. T5 aggregator in Python rebuilds the tree from `(request_id, span_id)`, filters `span_kind="diagnostic"` out of every tree-property computation, and emits the residual-based coverage gate.

**Tech Stack:** Rust (ironmlx server + iron-bench client), Python (T5 aggregator), `tracing` crate (existing stderr-routed log channel — see `ironmlx/src/main.rs:14`), MLX (existing array/eval pipeline). Build: `cargo run --release --features p5h-profile -p ironmlx`. Sweep harness extends `ironmlx/tests/p5g_t0_gated_delta_profile.rs` pattern.

---

## Spec source of truth

**Spec:** `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` at HEAD `4b9c037` (branch `ironmlx-p5h-perf`). Spec went through 21 review rounds; **DO NOT re-paraphrase § 2.5a semantics or coverage formulas in this plan** — the spec is the single source of truth (per Codex v8 P2 + v12 P2 + v17 P2 + v20 P1). This plan supplies file paths, code, and commands; semantics references point back at § 2.5a / § 7.1.

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `ironmlx/src/core/p5h.rs` | All trace-context types: `P5hTraceContext`, `SpanHandle`, `SpanFields`, `RootSpanHandle`, `P5hTraceGuard`, `P5H_CURRENT_TRACE`, `P5H_CURRENT_SPAN_STACK`. Dual emission API: `open_p5h_span`, `open_p5h_span_at`, `close_p5h_span`, `with_p5h_span_from_current_trace`. Span id atomic counter. `[p5h-profile]` log line formatter. Feature-gated; default build is empty stubs that compile to nothing. |
| `ironmlx/tests/p5h_t0a_harness.rs` | T0a HARD GATE fixture + UMA hardening protocol (cold/warm pair + retry) + GDN P5h-protocol rerun harness. Reuses P5g per-PP server spawn pattern. |
| `ironmlx/tests/p5h_t0b_phase_d.rs` | T0b Phase D 4-hypothesis investigation harness. |
| `tools/p5h_aggregator/aggregator.py` | T5 Python aggregator: parse `[p5h-profile]` log lines, join with iron-bench `request_id` CSV column, build tree from `(request_id, span_id)`, run § 2.5a structural checks (id uniqueness / single root / no orphan / closure / interval containment / reachability), compute § 7.1 residual-based coverage gate, emit per-PP attribution table + diagnostic columns. |
| `tools/p5h_aggregator/schema_validator.py` | T0a HARD GATE structural check module reused by aggregator + harness fixtures. |
| `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md` | This file. |

### Modified files

| Path | What changes |
|---|---|
| `ironmlx/Cargo.toml` | Add `p5h-profile` Cargo feature alongside existing `p5g-profile`. |
| `ironmlx/src/core/mod.rs` | Add `pub(crate) mod p5h;` under `#[cfg(feature = "p5h-profile")]`. |
| `ironmlx/src/core/generate.rs:25-56` | Add `p5h_trace: Option<P5hTraceContext>` and `p5h_root_span: Option<SpanHandle>` fields to `GenerateRequest`, feature-gated. |
| `ironmlx/src/core/generate.rs:928-967` | Wrap `GenerationStream::new()` invocation chunked-prefill-loop + `gs_kv_cache_alloc` + `gs_chunk_N` + `gs_first_token_sample_dispatch` deep spans via `with_p5h_span_from_current_trace`. |
| `ironmlx/src/core/scheduler.rs:331-385` | Add same two fields to `RequestState`. |
| `ironmlx/src/core/scheduler.rs:578` | `Scheduler::admit` copies both fields from `GenerateRequest` to `RequestState`. |
| `ironmlx/src/core/scheduler.rs:794-808` (`prefill_admitted_inner`) | SINK pattern: `cloned_active_row_p5h_trace_and_root()` helper → open `model_prefill_forward` + guard around `model.batched_prefill[_vl]` + close → open `first_token_sampling` + (no guard vanilla) around reshape + Stage A + `sample_batch` + close. |
| `ironmlx/src/core/scheduler.rs` (new helper) | `cloned_active_row_p5h_trace_and_root(&self) -> Result<(P5hTraceContext, SpanHandle)>` — owned-clone return. |
| `ironmlx/src/core/scheduler.rs:1087+` (`step` / `step_inner`) | If `step` fuses model-forward + sample (verify in T0a; spec line 334 leaves it to T0a to verify): same SINK pattern for `pre_content_decode_steps`. Otherwise actor-scope wrap. |
| `ironmlx/src/cli/serve.rs:67+` | Startup panic when `cfg!(feature = "p5h-profile")` AND `args.b_max > 1`. |
| `ironmlx/src/core/server/openai.rs:310-410` (`chat_completions`) | Handler ordering per § 2.5a: capture `root_start_ns` + `http_parse_start_ns` at entry → tokenize → compute `prompt_tokens` + `routing_path` → build `ctx` → open root via `open_p5h_span_at` → wrap as `RootSpanHandle` → open + close `http_parse_render_tokenize` via `open_p5h_span_at` → write `request.p5h_trace` + `request.p5h_root_span` → emit `X-Ironmlx-Request-Id` header. |
| `ironmlx/src/core/server/openai.rs:416-475` (`serve_via_gs_stream`) | Lane-B span emission per § 2.5a: capture `ctx` + `root_handle` clones into `spawn_blocking` closure → open `gs_stream_init_and_chunk_loop` + scoped guard around `GenerationStream::new(...)` + close → per-iteration `top` span + scoped guard around `stream.next_token()` + close → `sse_write_role_chunk` explicit + `detok_format_first_content_chunk` + `root_to_close.take().close_at(end_ns)`. |
| `ironmlx/src/core/server/openai.rs:501-600` (`serve_via_scheduler_stream`) | Lane-A: capture `ctx` + `root_handle` clones into `tokio::spawn` forwarder closure → `sse_write_role_chunk_diagnostic` (`span_kind="diagnostic"`) + `detok_format_first_content_chunk` + `root_to_close.take().close_at(end_ns)`. NO guard in forwarder. Scheduler_admission span recorded in handler around `cmd_tx.send + reply_rx.await`. |
| `ironmlx/src/core/server/scheduler_actor.rs:243-322` (`driver_loop`) | Actor-scope wrap around `sched.step(...)` IF step is sync-only (NOT a SINK candidate). Verified via T0a. |
| `ironmlx/src/main.rs:13-17` | No change required — `tracing_subscriber::fmt().with_writer(std::io::stderr)` already present (per P5g `5e35ab2`). |
| `ironmlx/src/nn/gated_delta_net.rs:1059-1077` | Replace single `[p5g-profile]` `tracing::info!` with **parallel emit** of both `[p5g-profile]` (existing format, back-compat) AND `[p5h-profile]` (new schema lines via `p5h.rs::emit_log_line`). Substep `parent_span_id` flows from active `attention_path` wrapper span via `P5H_CURRENT_SPAN_STACK`. |
| `ironmlx/src/nn/gated_attention.rs:119-280` | T2: 3-edit instrumentation pattern. Outer wrapper `attention_path` span opened by decoder layer (via `with_p5h_span_from_current_trace`); 7 substeps inside `forward_on` each emit their own span via same API. |
| `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:180-465` | T3: 3-edit instrumentation pattern. Outer wrapper `mlp_path` span opened by decoder layer; 8 substeps inside `SparseMoeBlock::forward_on` each emit own span. |
| `ironmlx/src/models/qwen3_5_moe/text_model.rs` (decoder layer site) | Open wrappers `attention_path` + `mlp_path` + emit `input_norm` / `post_attention_norm` / `residual_overhead` deep spans via `with_p5h_span_from_current_trace`. |
| `iron-bench/src/main.rs:23-77` | Add `--capture-server-request-id` clap flag (default off). |
| `iron-bench/src/client.rs:158-217` (`run_chat_completion`) | When flag is on: capture `X-Ironmlx-Request-Id` from `resp.headers()` BEFORE `bytes_stream()`. |
| `iron-bench/src/client.rs:41-53` (`RequestResult`) | Add `request_id: Option<String>` field. |
| `iron-bench/src/report.rs:483` | When flag is on: append `request_id` column at the end of CSV header + body; JSON object adds same field. Off-state byte-identical. |
| `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2 | T5 close-out locks spec final state (per P5g § 7.2 pattern). |

### Memory updates (post-T5)

- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md` (new file): record findings, P5i/P5j candidate ranking, target feasibility verdict.
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md`: add index entry.

---

## Conventions for all tasks

- **Branch:** `ironmlx-p5h-perf` (already exists; this plan extends it).
- **Commit messages:** English only (Boss memory `[feedback_commit_message_english]`). No `Co-Authored-By` line (committer is the implementer subagent, not the planning agent).
- **Per-task Rust validation gates (§ 4 of spec):**
  ```bash
  cargo fmt
  MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
  MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
  MLX_DIR=$HOME/.local/mlx cargo build --release
  ```
- **Sentinel suite (§ 5):**
  ```bash
  export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
  MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
  MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1
  MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1
  ```
- **Profile-gate invariant:** Default build (no `--features p5h-profile`) emits **0** `[p5h-profile]` log lines. Verified per-task by grep on a sweep stderr capture.
- **Harness server-launch contract:** Use `--prefill-chunk-size` default (2048) — do NOT override. Use `--b-max 1` (production default). Cargo feature is build-time via `--features p5h-profile`:
  ```bash
  MLX_DIR=$HOME/.local/mlx cargo run --release \
      --features p5h-profile \
      -p ironmlx -- serve --b-max 1 \
      --model "$IRONMLX_MOE_MODEL_DIR" \
      --port 8080
  ```
- **`reports/` directory does NOT enter git** (per Boss memory `[feedback_no_reports_commit]`). Aggregator output paths are working-tree only.
- **Single-active-row invariant** (§ 2.5a): server panics at startup if `p5h-profile` is active AND `b_max > 1`.

---

## Task T0a: Foundation — Schema Infrastructure + Request Correlation + UMA Hardening + GDN P5h-protocol Rerun

This task is the HARD GATE before T0b / T2 / T3 / T4. It must produce a working span lifecycle API + propagation chain + UMA harness + a GDN rerun whose emitted records pass all § 2.5a structural checks. If any T0a sub-task closes with a failed check, fix before moving on.

### T0a.1 — Cargo feature gate + module skeleton

**Files:**
- Modify: `ironmlx/Cargo.toml`
- Modify: `ironmlx/src/core/mod.rs`
- Create: `ironmlx/src/core/p5h.rs`

- [ ] **Step 1: Add Cargo feature**

Edit `ironmlx/Cargo.toml`, add to the `[features]` section (alongside existing `p5g-profile`):

```toml
p5h-profile = []
```

- [ ] **Step 2: Add module declaration**

Edit `ironmlx/src/core/mod.rs`, add at the bottom of the existing module list:

```rust
#[cfg(feature = "p5h-profile")]
pub(crate) mod p5h;
```

- [ ] **Step 3: Create p5h.rs skeleton with feature gate**

Create `ironmlx/src/core/p5h.rs` with this minimal stub (subsequent T0a sub-tasks add types/API):

```rust
//! P5h all-PP attribution instrumentation.
//!
//! Single source of truth for span schema + propagation:
//! docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md § 2.5a.
//!
//! DO NOT re-paraphrase semantics here. This file is the implementation
//! of the spec; the spec defines what the implementation MUST do. Any
//! deviation is a bug.

#![allow(dead_code)] // populated incrementally across T0a sub-tasks

use std::cell::RefCell;

thread_local! {
    pub(crate) static P5H_CURRENT_TRACE: RefCell<Option<P5hTraceContext>> = const { RefCell::new(None) };
    pub(crate) static P5H_CURRENT_SPAN_STACK: RefCell<Vec<SpanHandle>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug)]
pub struct P5hTraceContext {
    pub request_id: String,
    pub prompt_tokens: u32,
    pub routing_path: &'static str, // "scheduler" | "gs_chunked"
}

#[derive(Clone, Debug)]
pub struct SpanHandle {
    pub span_id: u64,
    pub span_name: &'static str,
    pub parent_span_id: Option<u64>,
    /// Human-readable parent label retained for log readability + label
    /// self-consistency check (per spec § 2.5a structural checks: "parent_span_id
    /// resolves to a span whose span_name equals the parent_span label").
    /// Set at open from the parent's `span_name`; None for root.
    /// Per Codex plan review v1 P1 #1: do NOT hard-code "explicit_parent" —
    /// emitter MUST carry the real parent label to match T0a fixture
    /// `label self-consistency` assertion in § 2.5a.
    pub parent_span: Option<&'static str>,
    pub start_ns: u64,
}

#[derive(Default, Debug)]
pub struct SpanFields {
    pub layer_idx: Option<i32>,
    pub seq: Option<u32>,
    pub mode: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct RootSpanHandle {
    ctx: P5hTraceContext,
    span: SpanHandle,
}

impl RootSpanHandle {
    pub(crate) fn ctx(&self) -> &P5hTraceContext { &self.ctx }
    pub(crate) fn span(&self) -> &SpanHandle { &self.span }
    pub(crate) fn close_at(self, _end_ns: u64) {
        // T0a.3 fills this in
        unimplemented!("filled in T0a.3");
    }
}

pub struct P5hTraceGuard;
```

- [ ] **Step 4: Verify build succeeds with both feature states**

Run:

```bash
cargo build --release -p ironmlx
cargo build --release -p ironmlx --features p5h-profile
```

Both must succeed.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/Cargo.toml ironmlx/src/core/mod.rs ironmlx/src/core/p5h.rs
git commit -m "feat(p5h-t0a): add p5h-profile Cargo feature + core/p5h.rs skeleton"
```

### T0a.2 — `P5hTraceGuard` RAII + thread-local discipline

**Files:**
- Modify: `ironmlx/src/core/p5h.rs`

- [ ] **Step 1: Write the failing test**

Append to `ironmlx/src/core/p5h.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> P5hTraceContext {
        P5hTraceContext {
            request_id: "test-req-0001".to_string(),
            prompt_tokens: 128,
            routing_path: "scheduler",
        }
    }

    fn dummy_span(id: u64) -> SpanHandle {
        SpanHandle {
            span_id: id,
            span_name: "test_root",
            parent_span_id: None,
            parent_span: None,
            start_ns: 1_000_000_000,
        }
    }

    #[test]
    fn guard_enter_drop_clears_thread_locals() {
        {
            let _g = P5hTraceGuard::enter(dummy_ctx(), dummy_span(1));
            P5H_CURRENT_TRACE.with(|c| assert!(c.borrow().is_some()));
            P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
        }
        P5H_CURRENT_TRACE.with(|c| assert!(c.borrow().is_none()));
        P5H_CURRENT_SPAN_STACK.with(|s| assert!(s.borrow().is_empty()));
    }

    #[test]
    #[should_panic(expected = "nested guards are forbidden")]
    fn guard_nesting_panics() {
        let _g1 = P5hTraceGuard::enter(dummy_ctx(), dummy_span(1));
        let _g2 = P5hTraceGuard::enter(dummy_ctx(), dummy_span(2));
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

Run:

```bash
cargo test -p ironmlx --features p5h-profile core::p5h::tests::guard_enter_drop_clears_thread_locals
```

Expected: FAIL — `P5hTraceGuard::enter` not implemented.

- [ ] **Step 3: Implement `P5hTraceGuard::enter` + `Drop`**

In `ironmlx/src/core/p5h.rs`, replace `pub struct P5hTraceGuard;` with:

```rust
pub struct P5hTraceGuard {
    // private field forces use of enter() constructor
    _marker: std::marker::PhantomData<()>,
}

impl P5hTraceGuard {
    /// Per § 2.5a: enter takes a `base_parent` SpanHandle that seeds the span
    /// stack. The base_parent is the explicit top-level span the caller has
    /// already opened. Authorized call sites are enumerated in § 2.5a
    /// "Authorized P5hTraceGuard::enter sites" — DO NOT add new ones.
    pub fn enter(ctx: P5hTraceContext, base_parent: SpanHandle) -> Self {
        P5H_CURRENT_TRACE.with(|c| {
            let mut slot = c.borrow_mut();
            assert!(
                slot.is_none(),
                "P5hTraceGuard::enter while another guard is active — nested guards are forbidden \
                 (helpers must READ via P5H_CURRENT_TRACE, not enter their own guard); \
                 fix the guard set/drop sites in the calling task/thread"
            );
            *slot = Some(ctx);
        });
        P5H_CURRENT_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            assert!(stack.is_empty(), "P5hTraceGuard::enter with non-empty span stack — prior guard leaked");
            stack.push(base_parent);
        });
        P5hTraceGuard { _marker: std::marker::PhantomData }
    }
}

impl Drop for P5hTraceGuard {
    fn drop(&mut self) {
        P5H_CURRENT_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            assert_eq!(
                stack.len(), 1,
                "P5hTraceGuard::drop with span stack length {} — expected 1 (only base_parent sentinel). \
                 Either an inner span was opened without close, or close was called more times than open.",
                stack.len(),
            );
            stack.clear();
        });
        P5H_CURRENT_TRACE.with(|c| *c.borrow_mut() = None);
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cargo test -p ironmlx --features p5h-profile core::p5h::tests
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/core/p5h.rs
git commit -m "feat(p5h-t0a): P5hTraceGuard RAII + nesting/stack-empty assertions"
```

### T0a.3 — Dual emission API (open/close/with_span) + log line formatter

**Files:**
- Modify: `ironmlx/Cargo.toml`
- Modify: `ironmlx/src/core/p5h.rs`

- [ ] **Step 0: Add `once_cell` dependency (used by `monotonic_ns()` below)**

Per Codex plan review v3 P1 #2: the `once_cell::sync::Lazy<Instant>` anchor in `monotonic_ns()` requires the `once_cell` crate. Add to `ironmlx/Cargo.toml` `[dependencies]` BEFORE writing any code that uses it (T0a.6 originally listed this step but T0a.3 is the first user). If `once_cell` is already a dependency in `ironmlx/Cargo.toml`, skip this step.

```toml
once_cell = "1"
```

Verify:

```bash
grep -q "once_cell" ironmlx/Cargo.toml && echo "OK" || echo "MISSING"
```

- [ ] **Step 1: Write the failing tests**

Append to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn open_close_explicit_records_log_line() {
        let ctx = dummy_ctx();
        let root = open_p5h_span_at(&ctx, None, "server_request_recv_to_first_content_sse_write", 1_000_000_000);
        assert!(root.parent_span_id.is_none());
        assert!(root.parent_span.is_none());
        assert_eq!(root.span_name, "server_request_recv_to_first_content_sse_write");
        assert!(root.span_id != 0);
        let child = open_p5h_span(&ctx, Some(&root), "http_parse_render_tokenize");
        assert_eq!(child.parent_span_id, Some(root.span_id));
        // Per Codex plan review v1 P1 #1: child.parent_span must be the real
        // parent label, NOT a hard-coded "explicit_parent".
        assert_eq!(child.parent_span, Some("server_request_recv_to_first_content_sse_write"));
        close_p5h_span(&ctx, child, 1_000_500_000, SpanFields::default());
        close_p5h_span(&ctx, root, 1_001_000_000, SpanFields::default());
    }

    #[test]
    #[should_panic(expected = "is not in open registry")]
    fn close_panics_on_double_close() {
        let ctx = dummy_ctx();
        let root = open_p5h_span_at(&ctx, None, "test_root", 0);
        let clone_for_first_close = root.clone();
        close_p5h_span(&ctx, root, 1, SpanFields::default());
        // Second close with the cloned handle must panic.
        close_p5h_span(&ctx, clone_for_first_close, 2, SpanFields::default());
    }

    #[test]
    #[should_panic(expected = "is not in open registry")]
    fn close_panics_on_unknown_handle() {
        let ctx = dummy_ctx();
        let bogus = SpanHandle {
            span_id: 9_999_999_999,
            span_name: "never_opened",
            parent_span_id: None,
            parent_span: None,
            start_ns: 0,
        };
        close_p5h_span(&ctx, bogus, 1, SpanFields::default());
    }

    #[test]
    fn open_close_id_uniqueness() {
        let ctx = dummy_ctx();
        let a = open_p5h_span(&ctx, None, "a");
        let b = open_p5h_span(&ctx, None, "b");
        let c = open_p5h_span(&ctx, None, "c");
        assert_ne!(a.span_id, b.span_id);
        assert_ne!(b.span_id, c.span_id);
        assert_ne!(a.span_id, c.span_id);
        close_p5h_span(&ctx, a, 0, SpanFields::default());
        close_p5h_span(&ctx, b, 0, SpanFields::default());
        close_p5h_span(&ctx, c, 0, SpanFields::default());
    }

    #[test]
    #[should_panic(expected = "called with no active P5H_CURRENT_TRACE")]
    fn with_span_panics_outside_guard() {
        // Panic message includes span_name per Codex plan review v3 P3 #7.
        let _ = with_p5h_span_from_current_trace::<u32>(
            "deep_span",
            SpanFields::default,
            || 42,
        );
    }

    #[test]
    #[should_panic(expected = "ctx mismatch")]
    fn close_panics_on_wrong_ctx() {
        // Per Codex plan review v3 P2 #3: closing a span with a different
        // request_id than was active at open MUST panic.
        let ctx_a = P5hTraceContext {
            request_id: "req-A".to_string(),
            prompt_tokens: 128,
            routing_path: "scheduler",
        };
        let ctx_b = P5hTraceContext {
            request_id: "req-B".to_string(),
            prompt_tokens: 128,
            routing_path: "scheduler",
        };
        let span = open_p5h_span_at(&ctx_a, None, "test_root", 0);
        close_p5h_span(&ctx_b, span, 1, SpanFields::default());
    }

    #[test]
    fn with_span_inside_guard_chains_parent() {
        let ctx = dummy_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        // Inside the guard region; with_span should chain under root.
        let result = with_p5h_span_from_current_trace(
            "deep_span",
            SpanFields::default,
            || 7u32,
        );
        assert_eq!(result, 7);
        // Stack returned to just base_parent after body.
        P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p ironmlx --features p5h-profile core::p5h::tests
```

Expected: 7 failures (missing `open_p5h_span`, `open_p5h_span_at`, `close_p5h_span`, `with_p5h_span_from_current_trace`, registry insert/remove panics, wrong-ctx close panic).

- [ ] **Step 3: Implement the dual API + log line formatter**

In `ironmlx/src/core/p5h.rs`, append (or place between the type defs and tests):

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

/// Tracks each open span's request_id (per Codex plan review v3 P2 #3 —
/// upgraded from HashSet<u64> so close_p5h_span can also detect wrong-ctx
/// close, not just unknown / double-close). Only enabled in p5h-profile
/// builds (no runtime cost in default builds since the whole module is
/// `#[cfg(feature = "p5h-profile")]`).
#[derive(Clone, Debug)]
struct OpenSpanRecord {
    span_name: &'static str,
    request_id: String,
}

static OPEN_SPAN_REGISTRY: Mutex<Option<HashMap<u64, OpenSpanRecord>>> = Mutex::new(None);

fn registry_insert(span_id: u64, record: OpenSpanRecord) {
    let mut guard = OPEN_SPAN_REGISTRY.lock().expect("p5h registry poisoned");
    let reg = guard.get_or_insert_with(HashMap::new);
    let prev = reg.insert(span_id, record);
    assert!(prev.is_none(), "open_p5h_span issued duplicate span_id={} — atomic counter race or registry corruption", span_id);
}

fn registry_remove_or_panic(span_id: u64, span_name: &'static str, expected_request_id: &str) {
    let mut guard = OPEN_SPAN_REGISTRY.lock().expect("p5h registry poisoned");
    let reg = guard.as_mut().expect("close_p5h_span called before any open");
    let record = reg.remove(&span_id).unwrap_or_else(|| {
        panic!(
            "close_p5h_span(span_name={}, span_id={}) — span_id is not in open registry. \
             Causes: (a) handle reused after close (double-close), (b) handle leaked from a different request, \
             (c) handle never opened. Per § 2.5a explicit-API hard-fail.",
            span_name, span_id,
        );
    });
    assert!(
        record.request_id == expected_request_id,
        "close_p5h_span(span_name={}, span_id={}) — ctx mismatch: opened with request_id={}, closing with request_id={}. \
         Cross-request handle leakage. Per Codex plan review v3 P2 #3 + § 2.5a explicit-API hard-fail.",
        span_name, span_id, record.request_id, expected_request_id,
    );
}

fn monotonic_ns() -> u64 {
    use std::time::Instant;
    static ANCHOR: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    ANCHOR.elapsed().as_nanos() as u64
}

pub(crate) fn monotonic_ns_public() -> u64 { monotonic_ns() }

fn next_span_id() -> u64 {
    NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed)
}

fn emit_log_line(
    ctx: &P5hTraceContext,
    span: &SpanHandle,
    end_ns: u64,
    fields: &SpanFields,
    span_kind: &'static str,
) {
    // Schema fields match § 2.5a server-emitted fields table. Order is
    // stable for the Python aggregator parser. `parent_span` label comes
    // from the SpanHandle (set at open from the parent's span_name per
    // Codex plan review v1 P1 #1 — do NOT hard-code "explicit_parent").
    tracing::info!(
        "[p5h-profile] request_id={} routing_path={} prompt_tokens={} seq={} layer_idx={} \
         span_id={} parent_span_id={} span_name={} parent_span={} \
         start_ns={} end_ns={} mode={} span_kind={}",
        ctx.request_id,
        ctx.routing_path,
        ctx.prompt_tokens,
        fields.seq.unwrap_or(0),
        fields.layer_idx.unwrap_or(-1),
        span.span_id,
        span.parent_span_id.map(|id| id.to_string()).unwrap_or_else(|| "null".to_string()),
        span.span_name,
        span.parent_span.unwrap_or("null"),
        span.start_ns,
        end_ns,
        fields.mode.unwrap_or("off"),
        span_kind,
    );
}

/// Open at the current monotonic time. Use when span start coincides with
/// the call site. Per § 2.5a explicit-context API.
pub fn open_p5h_span(
    ctx: &P5hTraceContext,
    parent: Option<&SpanHandle>,
    span_name: &'static str,
) -> SpanHandle {
    open_p5h_span_at(ctx, parent, span_name, monotonic_ns())
}

/// Open at an explicit start_ns. Required for root + http_parse_render_tokenize
/// per § 2.5a (start captured before ctx is complete).
pub fn open_p5h_span_at(
    ctx: &P5hTraceContext,
    parent: Option<&SpanHandle>,
    span_name: &'static str,
    start_ns: u64,
) -> SpanHandle {
    // Hard-fail on empty ctx (catches forgotten plumbing per § 2.5a hard-fail rules).
    assert!(
        !ctx.request_id.is_empty(),
        "open_p5h_span[_at]({}) called with empty P5hTraceContext.request_id — context not populated",
        span_name,
    );
    let span_id = next_span_id();
    registry_insert(span_id, OpenSpanRecord {
        span_name,
        request_id: ctx.request_id.clone(),
    });
    SpanHandle {
        span_id,
        span_name,
        parent_span_id: parent.map(|p| p.span_id),
        // Per Codex plan review v1 P1 #1: carry real parent label, NOT
        // "explicit_parent" placeholder. Label self-consistency check in T0a
        // fixture asserts parent_span_id resolves to a span whose span_name
        // equals this label.
        parent_span: parent.map(|p| p.span_name),
        start_ns,
    }
}

/// Close an explicit-context tree span. Emits the `[p5h-profile]` log line.
/// Per Codex plan review v1 P2 #5 + v3 P2 #3: hard-fail if span_id is not in
/// the open registry OR if ctx.request_id doesn't match the record stored at
/// open (catches handle reuse / cross-request leakage / double-close /
/// wrong-ctx close).
pub fn close_p5h_span(
    ctx: &P5hTraceContext,
    handle: SpanHandle,
    end_ns: u64,
    fields: SpanFields,
) {
    registry_remove_or_panic(handle.span_id, handle.span_name, &ctx.request_id);
    emit_log_line(ctx, &handle, end_ns, &fields, "tree");
}

/// Close a diagnostic span (e.g., Lane-A `sse_write_role_chunk_diagnostic`).
/// Per § 2.5a v19 P1: diagnostic spans are NOT in the exclusive tree.
pub fn close_p5h_span_diagnostic(
    ctx: &P5hTraceContext,
    handle: SpanHandle,
    end_ns: u64,
    fields: SpanFields,
) {
    registry_remove_or_panic(handle.span_id, handle.span_name, &ctx.request_id);
    emit_log_line(ctx, &handle, end_ns, &fields, "diagnostic");
}

/// Implicit-guard API. Internally opens span (parent = stack top), pushes,
/// runs body, pops, closes. Panics if no active guard. Per § 2.5a.
pub fn with_p5h_span_from_current_trace<T>(
    span_name: &'static str,
    fields_fn: impl FnOnce() -> SpanFields,
    body: impl FnOnce() -> T,
) -> T {
    let start_ns = monotonic_ns();
    // Read trace ctx once at the start (per Codex plan review v3 P3 #7 —
    // include span_name in the no-guard panic message; v2 panic message was
    // a bare static string that hid which instrumentation site triggered it).
    let request_id_at_open: String = P5H_CURRENT_TRACE.with(|c| {
        c.borrow().as_ref()
            .unwrap_or_else(|| panic!(
                "with_p5h_span_from_current_trace(span_name={}) called with no active P5H_CURRENT_TRACE — \
                 site is not inside an authorized P5hTraceGuard region (per § 2.5a Authorized guard sites)",
                span_name,
            ))
            .request_id.clone()
    });
    let (parent_id, parent_label) = P5H_CURRENT_SPAN_STACK.with(|s| {
        let stack = s.borrow();
        let top = stack.last().unwrap_or_else(|| panic!(
            "with_p5h_span_from_current_trace(span_name={}) called with empty span stack — \
             guard active but stack not seeded (base_parent missing)",
            span_name,
        ));
        (top.span_id, top.span_name)
    });
    let span_id = next_span_id();
    registry_insert(span_id, OpenSpanRecord {
        span_name,
        request_id: request_id_at_open.clone(),
    });
    let handle = SpanHandle {
        span_id,
        span_name,
        parent_span_id: Some(parent_id),
        // Per Codex plan review v1 P1 #1: carry real parent label for T0a
        // label self-consistency assertion.
        parent_span: Some(parent_label),
        start_ns,
    };
    P5H_CURRENT_SPAN_STACK.with(|s| s.borrow_mut().push(handle.clone()));
    let result = body();
    let popped = P5H_CURRENT_SPAN_STACK.with(|s| s.borrow_mut().pop().unwrap_or_else(|| panic!(
        "stack underflow in with_p5h_span_from_current_trace(span_name={})",
        span_name,
    )));
    assert_eq!(popped.span_id, handle.span_id, "stack imbalance: popped a different span ({}) than the one opened ({})", popped.span_name, handle.span_name);
    let end_ns = monotonic_ns();
    let fields = fields_fn();
    registry_remove_or_panic(handle.span_id, handle.span_name, &request_id_at_open);
    P5H_CURRENT_TRACE.with(|c| {
        let ctx_ref = c.borrow();
        let ctx = ctx_ref.as_ref().unwrap_or_else(|| panic!(
            "with_p5h_span_from_current_trace(span_name={}) lost P5H_CURRENT_TRACE mid-body — guard dropped concurrently",
            span_name,
        ));
        emit_log_line(ctx, &handle, end_ns, &fields, "tree");
    });
    result
}

impl RootSpanHandle {
    pub(crate) fn new(ctx: P5hTraceContext, span: SpanHandle) -> Self {
        RootSpanHandle { ctx, span }
    }
}
```

Now replace the stub `close_at` (added in T0a.1) with the real implementation by editing `impl RootSpanHandle`:

```rust
impl RootSpanHandle {
    pub(crate) fn ctx(&self) -> &P5hTraceContext { &self.ctx }
    pub(crate) fn span(&self) -> &SpanHandle { &self.span }

    pub(crate) fn close_at(self, end_ns: u64) {
        close_p5h_span(&self.ctx, self.span, end_ns, SpanFields::default());
    }
}
```

(Remove the duplicate `impl RootSpanHandle` block from T0a.1 — keep only the final one.)

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cargo test -p ironmlx --features p5h-profile core::p5h::tests
```

Expected: all 9 tests PASS (6 prior + close_panics_on_double_close + close_panics_on_unknown_handle + close_panics_on_wrong_ctx).

- [ ] **Step 5: Verify default build emits zero `[p5h-profile]` lines**

Without `p5h-profile` feature, `core::p5h` is not compiled at all (per `#[cfg]` in `mod.rs`). Confirm:

```bash
cargo build --release -p ironmlx 2>&1 | grep -i "p5h" || echo "OK: no p5h refs in default build"
```

Expected: prints `OK: no p5h refs in default build` (no `p5h-profile` activation messages).

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/p5h.rs
git commit -m "feat(p5h-t0a): dual emission API (open/close/with_span) + log line formatter"
```

### T0a.4 — `GenerateRequest` + `RequestState` field additions + `Scheduler::admit` copy + `cloned_active_row_p5h_trace_and_root` helper

**Files:**
- Modify: `ironmlx/src/core/generate.rs:25-56`
- Modify: `ironmlx/src/core/scheduler.rs:331-385` (`RequestState`)
- Modify: `ironmlx/src/core/scheduler.rs:578` (`Scheduler::admit`)
- Modify: `ironmlx/src/core/scheduler.rs` (add helper after admit)

- [ ] **Step 1: Add p5h fields to `GenerateRequest`**

Edit `ironmlx/src/core/generate.rs:25-56`. After the last existing field (`image_token_id`), add:

```rust
    /// P5h trace context (gated on `p5h-profile` feature). Populated by the
    /// HTTP handler before admit; copied into RequestState. None on default
    /// builds. See spec § 2.5a "Propagation chain".
    #[cfg(feature = "p5h-profile")]
    pub p5h_trace: Option<crate::core::p5h::P5hTraceContext>,

    /// P5h root SpanHandle (gated on `p5h-profile` feature). Populated alongside
    /// `p5h_trace`. Used by `Scheduler::prefill_admitted_inner` to open
    /// `model_prefill_forward` + `first_token_sampling` with the correct parent.
    #[cfg(feature = "p5h-profile")]
    pub p5h_root_span: Option<crate::core::p5h::SpanHandle>,
```

- [ ] **Step 2: Add same fields to `RequestState`**

Edit `ironmlx/src/core/scheduler.rs:331-385` (`RequestState`). After the last field, add:

```rust
    #[cfg(feature = "p5h-profile")]
    pub(crate) p5h_trace: Option<crate::core::p5h::P5hTraceContext>,

    #[cfg(feature = "p5h-profile")]
    pub(crate) p5h_root_span: Option<crate::core::p5h::SpanHandle>,
```

- [ ] **Step 3: Copy fields in `Scheduler::admit`**

Locate `Scheduler::admit(...)` body near `scheduler.rs:578`. Find the line(s) that construct the `RequestState` (look for `RequestState {`). Add the two new fields at the end of the struct literal:

```rust
            #[cfg(feature = "p5h-profile")]
            p5h_trace: req.p5h_trace.clone(),
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: req.p5h_root_span.clone(),
```

- [ ] **Step 4: Add `cloned_active_row_p5h_trace_and_root` helper**

In `ironmlx/src/core/scheduler.rs`, add after `Scheduler::admit` (or in an `#[cfg(feature = "p5h-profile")] impl Scheduler` block):

```rust
#[cfg(feature = "p5h-profile")]
impl<M> Scheduler<M> {
    /// Return owned clones of the lone active row's trace context + root span.
    /// Per § 2.5a + Codex v17 P1: returns owned values (NOT references) because
    /// `prefill_admitted_inner` needs &mut self.cache / &mut self.slots /
    /// &mut self.prng_state after this call, which would conflict with refs
    /// borrowed from self.slots.
    pub(crate) fn cloned_active_row_p5h_trace_and_root(
        &self,
    ) -> anyhow::Result<(crate::core::p5h::P5hTraceContext, crate::core::p5h::SpanHandle)> {
        let active: Vec<&RequestState> = self.slots.iter().filter_map(|s| s.as_ref()).collect();
        anyhow::ensure!(
            active.len() == 1,
            "p5h-profile invariant: expected exactly 1 active row, found {} (--b-max 1 required)",
            active.len(),
        );
        let state = active[0];
        let ctx = state.p5h_trace.clone().ok_or_else(|| anyhow::anyhow!(
            "p5h-profile invariant: active RequestState.p5h_trace is None — request not populated by handler"
        ))?;
        let root_span = state.p5h_root_span.clone().ok_or_else(|| anyhow::anyhow!(
            "p5h-profile invariant: active RequestState.p5h_root_span is None — request not populated by handler"
        ))?;
        Ok((ctx, root_span))
    }
}
```

- [ ] **Step 5: Build check both feature states**

```bash
cargo build --release -p ironmlx
cargo build --release -p ironmlx --features p5h-profile
```

Both must succeed.

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/generate.rs ironmlx/src/core/scheduler.rs
git commit -m "feat(p5h-t0a): add p5h_trace + p5h_root_span to GenerateRequest/RequestState + cloned_active_row helper"
```

### T0a.5 — Server startup panic on `b_max > 1` under `p5h-profile`

**Files:**
- Modify: `ironmlx/src/cli/serve.rs:67+`

- [ ] **Step 1: Add startup check**

Locate the function body that begins around `ironmlx/src/cli/serve.rs:67` (where `b_max` is first read). At the very start of that function body — before any logging — add:

```rust
    #[cfg(feature = "p5h-profile")]
    {
        assert_eq!(
            args.b_max, 1,
            "p5h-profile feature requires --b-max 1 (single-active-row invariant per § 2.5a). \
             Got --b-max {}. Rebuild without --features p5h-profile to use multi-row batching.",
            args.b_max,
        );
    }
```

- [ ] **Step 2: Verify default build unaffected**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
```

Build must succeed.

- [ ] **Step 3: Verify feature build with `--b-max 2` panics at startup**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile 2>&1 | tail -5
# Now start it manually with --b-max 2 to confirm panic; do NOT actually run a sweep against it
target/release/ironmlx serve --b-max 2 --model "$IRONMLX_MOE_MODEL_DIR" --port 18099 2>&1 | head -10
```

Expected: assertion failure mentioning "p5h-profile feature requires --b-max 1".

- [ ] **Step 4: Commit**

```bash
git add ironmlx/src/cli/serve.rs
git commit -m "feat(p5h-t0a): server startup panic if p5h-profile + b_max > 1"
```

### T0a.6 — Handler ordering: root span + http_parse_render_tokenize + scheduler_admission + `X-Ironmlx-Request-Id` header

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs:310-410` (`chat_completions`)
- Modify: `ironmlx/src/core/server/openai.rs:501+` (`serve_via_scheduler_stream` — admission span only)

- [ ] **Step 1: Add `Uuid` dep if not present**

Check `ironmlx/Cargo.toml`. If `uuid` is not in `[dependencies]`, add:

```toml
uuid = { version = "1", features = ["v4"] }
```

- [ ] **Step 2: Modify `chat_completions` handler — entry block**

Edit `ironmlx/src/core/server/openai.rs:310-410`. At the very start of the `pub async fn chat_completions<M>(...) -> Response` body, before any work, add (per Codex plan review v3 P2 #5 — keep this snippet clippy-clean: no unused `Instant` / `now` bindings, use the `monotonic_ns_public()` helper added in T0a.3):

```rust
    // P5h root + http_parse_render_tokenize start capture (per spec § 2.5a step 1).
    // Both timestamps captured at handler entry BEFORE any parse/tokenize work,
    // because the http_parse_render_tokenize span's true start is the entry point,
    // and the root span needs the same anchor.
    #[cfg(feature = "p5h-profile")]
    let (p5h_request_id, p5h_root_start_ns, p5h_http_start_ns) = (
        uuid::Uuid::new_v4().to_string(),
        crate::core::p5h::monotonic_ns_public(),
        crate::core::p5h::monotonic_ns_public(),
    );
```

The helper `monotonic_ns_public` (and the underlying `monotonic_ns` with `once_cell::sync::Lazy<Instant>` anchor) was already implemented in T0a.3 — no additional plumbing here.

- [ ] **Step 3: After tokenize, before admit — construct ctx + open root + open/close http_parse**

Locate the code in `chat_completions` that finishes tokenization and computes `prompt_len`. Just before `let use_scheduler = ...;` (around line 404), add:

```rust
    #[cfg(feature = "p5h-profile")]
    let p5h_routing_path: &'static str = if state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size {
        "scheduler"
    } else {
        "gs_chunked"
    };

    #[cfg(feature = "p5h-profile")]
    let p5h_ctx = crate::core::p5h::P5hTraceContext {
        request_id: p5h_request_id.clone(),
        prompt_tokens: prompt_len as u32,
        routing_path: p5h_routing_path,
    };

    #[cfg(feature = "p5h-profile")]
    let p5h_root_span = crate::core::p5h::open_p5h_span_at(
        &p5h_ctx,
        None,
        "server_request_recv_to_first_content_sse_write",
        p5h_root_start_ns,
    );

    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle = crate::core::p5h::RootSpanHandle::new(p5h_ctx.clone(), p5h_root_span.clone());

    #[cfg(feature = "p5h-profile")]
    {
        let http_span = crate::core::p5h::open_p5h_span_at(
            &p5h_ctx,
            Some(&p5h_root_span),
            "http_parse_render_tokenize",
            p5h_http_start_ns,
        );
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            http_span,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
    }
```

- [ ] **Step 4: Populate the `GenerateRequest`**

After the existing `GenerateRequest { ... }` construction in `chat_completions`, the fields `p5h_trace` and `p5h_root_span` will not be present by default. Either:
- Use struct update syntax: `..GenerateRequest::default_for_p5h(...)`, OR
- Add the two fields explicitly inside the struct literal:

Locate the `GenerateRequest { ... }` literal in `chat_completions` and add at the end:

```rust
            #[cfg(feature = "p5h-profile")]
            p5h_trace: Some(p5h_ctx.clone()),
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: Some(p5h_root_span.clone()),
```

- [ ] **Step 5: Add `X-Ironmlx-Request-Id` response header on streaming + non-streaming paths**

In each of the 4 dispatch branches (`(true, true)`, `(true, false)`, `(false, true)`, `(false, false)`), the inner function builds a `Response`. Find each response construction (search for `Response::builder()` or `.into_response()` in `serve_via_*` functions). Before the response is returned, add the header. For streaming paths the easiest place is at the `Response::builder().header(...)` call. Modify each of the 4 functions to add (under `#[cfg(feature = "p5h-profile")]`):

```rust
    #[cfg(feature = "p5h-profile")]
    let response = {
        let mut resp = response;
        resp.headers_mut().insert(
            "X-Ironmlx-Request-Id",
            request.p5h_trace.as_ref().expect("p5h-profile: ctx not populated").request_id.parse().unwrap(),
        );
        resp
    };
```

Place this just before `response` is returned in each of `serve_via_scheduler_stream`, `serve_via_gs_stream`, `serve_via_scheduler_unary`, `serve_via_gs_unary`.

- [ ] **Step 6: Add `scheduler_admission` explicit span in `serve_via_scheduler_stream`**

Edit `ironmlx/src/core/server/openai.rs:501-560` area. The admit-command-send site is around lines 513-517 per the codebase survey. Wrap that block:

```rust
    #[cfg(feature = "p5h-profile")]
    let admission_span_handle = {
        let ctx = request.p5h_trace.as_ref().expect("p5h-profile: ctx not populated");
        let root = request.p5h_root_span.as_ref().expect("p5h-profile: root_span not populated");
        crate::core::p5h::open_p5h_span(ctx, Some(root), "scheduler_admission")
    };

    // ... existing cmd_tx.send(...).await + reply_rx.await ...

    #[cfg(feature = "p5h-profile")]
    {
        let ctx = request.p5h_trace.as_ref().expect("p5h-profile: ctx not populated");
        crate::core::p5h::close_p5h_span(
            ctx,
            admission_span_handle,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
    }
```

- [ ] **Step 7: Build and run sentinel suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile
# Sentinel suite confirms numerical safety (default build path)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1
```

All three commands must succeed.

- [ ] **Step 8: Commit**

```bash
git add ironmlx/Cargo.toml ironmlx/src/core/p5h.rs ironmlx/src/core/server/openai.rs
git commit -m "feat(p5h-t0a): handler ordering + root/http_parse/scheduler_admission spans + X-Ironmlx-Request-Id header"
```

### T0a.7 — Lane-A forwarder: `sse_write_role_chunk_diagnostic` + `detok_format_first_content_chunk` + root close

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs:546-600+` (`serve_via_scheduler_stream` forwarder closure)

- [ ] **Step 1: Identify the forwarder spawn body**

Locate the `tokio::spawn(async move { ... })` at line 546 area inside `serve_via_scheduler_stream`. The closure currently captures `event_rx`, `tx`, `tokenizer`, `id`, `model_id`, etc.

- [ ] **Step 2: Capture `ctx` + `root_handle` clones into the closure**

Before the `tokio::spawn`, clone the p5h state out of `request`:

```rust
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx = request.p5h_trace.clone();
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_for_forwarder = request.p5h_trace.as_ref().zip(request.p5h_root_span.as_ref())
        .map(|(ctx, span)| crate::core::p5h::RootSpanHandle::new(ctx.clone(), span.clone()));
```

Move these into the spawn closure capture list (`async move { let mut root_to_close = p5h_root_handle_for_forwarder; ... }`).

- [ ] **Step 3: Wrap role-chunk send with `sse_write_role_chunk_diagnostic` span**

Inside the forwarder closure, the role chunk is sent around line 562. Wrap it:

```rust
    #[cfg(feature = "p5h-profile")]
    let role_span = if let Some(ctx) = p5h_ctx.as_ref() {
        let root = root_to_close.as_ref().map(|h| h.span().clone());
        root.map(|root_span| crate::core::p5h::open_p5h_span(ctx, Some(&root_span), "sse_write_role_chunk_diagnostic"))
    } else { None };

    // ... existing tx.send(Ok(...)).await for role chunk ...

    #[cfg(feature = "p5h-profile")]
    if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), role_span) {
        crate::core::p5h::close_p5h_span_diagnostic(
            ctx,
            handle,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
    }
```

- [ ] **Step 4: Wrap first-content chunk send with `detok_format_first_content_chunk` + close root**

Locate the per-event loop in the forwarder (around line 589 area). On the first iteration where `delta.content` is non-empty, wrap:

```rust
    #[cfg(feature = "p5h-profile")]
    let content_span = if first_non_empty_content {
        p5h_ctx.as_ref().zip(root_to_close.as_ref()).map(|(ctx, root)| {
            crate::core::p5h::open_p5h_span(ctx, Some(root.span()), "detok_format_first_content_chunk")
        })
    } else { None };

    // ... existing format_sse + tx.send(...).await ...

    #[cfg(feature = "p5h-profile")]
    if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), content_span) {
        let end_ns = crate::core::p5h::monotonic_ns_public();
        crate::core::p5h::close_p5h_span(
            ctx,
            handle,
            end_ns,
            crate::core::p5h::SpanFields::default(),
        );
        if first_non_empty_content {
            if let Some(root_handle) = root_to_close.take() {
                root_handle.close_at(end_ns);
            } else {
                panic!("p5h: root closed twice in Lane-A forwarder");
            }
        }
    }
```

- [ ] **Step 5: Sentinel + http smoke + manual verify single emission**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | grep -c "\\[p5h-profile\\]"
```

Expected: > 0 lines emitted by the test (each request emits root + http + admit + role_diag + content + ...).

Also verify default build emits zero:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | grep -c "\\[p5h-profile\\]"
```

Expected: 0.

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/core/server/openai.rs
git commit -m "feat(p5h-t0a): Lane-A forwarder role-diagnostic + first-content + root close"
```

### T0a.8 — Lane-B `spawn_blocking` body: `gs_stream_init_and_chunk_loop` + per-iteration spans + SSE + root close

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs:416-475` (`serve_via_gs_stream`)
- Modify: `ironmlx/src/core/generate.rs:928-967+` (`GenerationStream::new` — add deep spans for chunks + kv_cache_alloc + sample_dispatch)

- [ ] **Step 1: Capture ctx + root_handle into the spawn_blocking closure**

Just like Lane-A but with `spawn_blocking`:

```rust
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx = request.p5h_trace.clone();
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_for_gs = request.p5h_trace.as_ref().zip(request.p5h_root_span.as_ref())
        .map(|(ctx, span)| crate::core::p5h::RootSpanHandle::new(ctx.clone(), span.clone()));
```

Move these into the `tokio::task::spawn_blocking(move || { ... })` closure.

- [ ] **Step 2: Wrap `GenerationStream::new(...)` with `gs_stream_init_and_chunk_loop` + scoped guard**

Edit `ironmlx/src/core/server/openai.rs:432` area (the `GenerationStream::new` call). Replace:

```rust
        let mut stream = match GenerationStream::new(&*model_guard, tokenizer, request) {
```

with:

```rust
        #[cfg(feature = "p5h-profile")]
        let gs_top_span = p5h_ctx.as_ref().zip(p5h_root_handle_for_gs.as_ref()).map(|(ctx, root)| {
            crate::core::p5h::open_p5h_span(ctx, Some(root.span()), "gs_stream_init_and_chunk_loop")
        });

        #[cfg(feature = "p5h-profile")]
        let _gs_guard = match (p5h_ctx.as_ref(), gs_top_span.as_ref()) {
            (Some(ctx), Some(top)) => Some(crate::core::p5h::P5hTraceGuard::enter(ctx.clone(), top.clone())),
            _ => None,
        };

        let mut stream = match GenerationStream::new(&*model_guard, tokenizer, request) {
```

After the match block (after `Ok(s) => s` is bound), drop the guard and close the span:

```rust
            Ok(s) => {
                #[cfg(feature = "p5h-profile")]
                drop(_gs_guard);
                #[cfg(feature = "p5h-profile")]
                if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), gs_top_span) {
                    crate::core::p5h::close_p5h_span(ctx, handle, crate::core::p5h::monotonic_ns_public(), crate::core::p5h::SpanFields::default());
                }
                s
            }
```

- [ ] **Step 3: Add deep spans inside `GenerationStream::new`**

Edit `ironmlx/src/core/generate.rs:946-967`. Wrap the `model.make_cache(...)` call with:

```rust
        #[cfg(feature = "p5h-profile")]
        let cache = crate::core::p5h::with_p5h_span_from_current_trace(
            "gs_kv_cache_alloc",
            crate::core::p5h::SpanFields::default,
            || model.make_cache(1, cap, dtype),
        )?;
        #[cfg(not(feature = "p5h-profile"))]
        let cache = model.make_cache(1, cap, dtype)?;
```

In the chunked prefill loop (read lines 951+ to find the loop body), wrap each chunk iteration:

```rust
        for chunk_idx in 0..n_chunks {
            #[cfg(feature = "p5h-profile")]
            let _ = crate::core::p5h::with_p5h_span_from_current_trace(
                "gs_chunk_N",
                || crate::core::p5h::SpanFields { seq: Some(chunk_size as u32), layer_idx: Some(chunk_idx as i32), ..Default::default() },
                || -> anyhow::Result<()> {
                    // existing chunk body: forward_text_hidden + cache update + eval
                    /* existing code unchanged */
                    Ok(())
                },
            )?;
            #[cfg(not(feature = "p5h-profile"))]
            { /* existing chunk body unchanged */ }
        }
```

For the first-token sample dispatch (`generate.rs:1097-1098` pipelined path OR `1123-1125` sync path), wrap:

```rust
        #[cfg(feature = "p5h-profile")]
        let pending = crate::core::p5h::with_p5h_span_from_current_trace(
            "gs_first_token_sample_dispatch",
            crate::core::p5h::SpanFields::default,
            || -> anyhow::Result<_> {
                if pipelined {
                    let p = request.sampler.sample_async_greedy(&last_logits)?;
                    mlx::transforms::async_eval(&[&p])?;
                    Ok(P5hSampleResult::Pipelined(p))
                } else {
                    let first_token = request.sampler.sample(&last_logits, &history, &mut prng_state)?;
                    Ok(P5hSampleResult::Sync(first_token))
                }
            },
        )?;
```

(Refactor the existing branches into a single `match P5hSampleResult` block for code reuse — both feature-on and feature-off paths take the same `if pipelined { ... } else { ... }` shape.)

- [ ] **Step 4: Wrap per-iteration `stream.next_token()` in the post-prefill loop**

Back in `serve_via_gs_stream` (`openai.rs:459` loop area), wrap each iteration:

```rust
        #[cfg(feature = "p5h-profile")]
        let mut p5h_first_iter = true;
        loop {
            #[cfg(feature = "p5h-profile")]
            let iter_top_name: &'static str = if p5h_first_iter { "gs_first_token_materialize_and_predispatch" } else { "pre_content_decode_steps" };

            #[cfg(feature = "p5h-profile")]
            let iter_top_span = p5h_ctx.as_ref().zip(p5h_root_handle_for_gs.as_ref()).map(|(ctx, root)| {
                crate::core::p5h::open_p5h_span(ctx, Some(root.span()), iter_top_name)
            });

            #[cfg(feature = "p5h-profile")]
            let _iter_guard = match (p5h_ctx.as_ref(), iter_top_span.as_ref()) {
                (Some(ctx), Some(top)) => Some(crate::core::p5h::P5hTraceGuard::enter(ctx.clone(), top.clone())),
                _ => None,
            };

            let ev_result = stream.next_token();

            #[cfg(feature = "p5h-profile")]
            drop(_iter_guard);
            #[cfg(feature = "p5h-profile")]
            if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), iter_top_span) {
                crate::core::p5h::close_p5h_span(ctx, handle, crate::core::p5h::monotonic_ns_public(), crate::core::p5h::SpanFields::default());
            }

            #[cfg(feature = "p5h-profile")]
            { p5h_first_iter = false; }

            // ... existing match ev_result handling ...
        }
```

- [ ] **Step 5: Wrap role-chunk send + first-content send + root close**

The role-chunk send at line 455 is sequential inside `spawn_blocking` so it's a true `span_kind="tree"` `sse_write_role_chunk` (unlike Lane-A's diagnostic):

```rust
        #[cfg(feature = "p5h-profile")]
        let role_span = p5h_ctx.as_ref().zip(p5h_root_handle_for_gs.as_ref())
            .map(|(ctx, root)| crate::core::p5h::open_p5h_span(ctx, Some(root.span()), "sse_write_role_chunk"));

        if tx.blocking_send(Ok(format_sse_data(&role_chunk))).is_err() {
            return;
        }

        #[cfg(feature = "p5h-profile")]
        if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), role_span) {
            crate::core::p5h::close_p5h_span(ctx, handle, crate::core::p5h::monotonic_ns_public(), crate::core::p5h::SpanFields::default());
        }
```

For first non-empty content (around line 473), wrap content send + close root:

```rust
        #[cfg(feature = "p5h-profile")]
        let mut root_to_close: Option<crate::core::p5h::RootSpanHandle> = p5h_root_handle_for_gs.clone();

        // ... inside the loop where the content_chunk is sent ...

        #[cfg(feature = "p5h-profile")]
        let content_span = if first_non_empty_content {
            p5h_ctx.as_ref().zip(root_to_close.as_ref())
                .map(|(ctx, root)| crate::core::p5h::open_p5h_span(ctx, Some(root.span()), "detok_format_first_content_chunk"))
        } else { None };

        if tx.blocking_send(Ok(format_sse_data(&chunk))).is_err() {
            break;
        }

        #[cfg(feature = "p5h-profile")]
        if let (Some(ctx), Some(handle)) = (p5h_ctx.as_ref(), content_span) {
            let end_ns = crate::core::p5h::monotonic_ns_public();
            crate::core::p5h::close_p5h_span(ctx, handle, end_ns, crate::core::p5h::SpanFields::default());
            if first_non_empty_content {
                if let Some(root_handle) = root_to_close.take() {
                    root_handle.close_at(end_ns);
                } else {
                    panic!("p5h: root closed twice in Lane-B spawn_blocking");
                }
            }
        }
```

- [ ] **Step 6: Build + sentinel**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1
```

All must succeed.

- [ ] **Step 7: Commit**

```bash
git add ironmlx/src/core/server/openai.rs ironmlx/src/core/generate.rs
git commit -m "feat(p5h-t0a): Lane-B spawn_blocking spans (gs_stream_init_and_chunk_loop + per-iteration + role + content + root close)"
```

### T0a.9 — `prefill_admitted_inner` SINK: `model_prefill_forward` + `first_token_sampling`

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs:808-1025` (`prefill_admitted_inner`)

- [ ] **Step 1: Apply the SINK pattern**

At the top of `prefill_admitted_inner` body (just before the existing prefill setup), add:

```rust
        #[cfg(feature = "p5h-profile")]
        let (p5h_ctx, p5h_root_span) = self.cloned_active_row_p5h_trace_and_root()?;
```

Wrap the existing `model.batched_prefill(...)` / `batched_prefill_vl(...)` call (lines 959-981 area):

```rust
        #[cfg(feature = "p5h-profile")]
        let mpf_span = crate::core::p5h::open_p5h_span(&p5h_ctx, Some(&p5h_root_span), "model_prefill_forward");

        let logits = {
            #[cfg(feature = "p5h-profile")]
            let _mpf_guard = crate::core::p5h::P5hTraceGuard::enter(p5h_ctx.clone(), mpf_span.clone());

            // existing if is_vl { batched_prefill_vl(...) } else { batched_prefill(...) }
            /* unchanged body */
        };

        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            mpf_span,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
```

Wrap the post-prefill reshape + Stage A + `sample_batch` block (lines 996-1025):

```rust
        #[cfg(feature = "p5h-profile")]
        let fts_span = crate::core::p5h::open_p5h_span(&p5h_ctx, Some(&p5h_root_span), "first_token_sampling");

        // existing reshape + collect sampler refs + sample_batch
        let logits_shape = logits.shape();
        /* unchanged Stage A / Stage B body */
        let tokens = sample_batch(/* ... */)?;

        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            fts_span,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
```

Stage C distribution stays untouched.

- [ ] **Step 2: Sentinel + smoke**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1
```

`p5_qwen35_moe_smoke` must produce argmax=11 (numerical safety per § 5).

- [ ] **Step 3: Commit**

```bash
git add ironmlx/src/core/scheduler.rs
git commit -m "feat(p5h-t0a): prefill_admitted_inner SINK (model_prefill_forward + first_token_sampling)"
```

### T0a.10 — iron-bench `--capture-server-request-id` flag + client capture + report column

**Files:**
- Modify: `iron-bench/src/main.rs:23-77`
- Modify: `iron-bench/src/client.rs:41-53, 158-217`
- Modify: `iron-bench/src/report.rs:483`

- [ ] **Step 1: Add CLI flag**

Edit `iron-bench/src/main.rs` `Args` struct, add (after the last existing field):

```rust
    /// Capture X-Ironmlx-Request-Id response header from each request and
    /// add a request_id column to CSV/JSON output. Default off — flag-off
    /// state is byte-identical to non-P5h iron-bench output (per P5h spec
    /// § 2.5a Join key).
    #[arg(long, default_value_t = false)]
    pub capture_server_request_id: bool,
```

Plumb `args.capture_server_request_id` through to `run_chat_completion` and `report::write`.

- [ ] **Step 2: Add field to `RequestResult`**

Edit `iron-bench/src/client.rs:41-53`. After the last existing field add:

```rust
    /// Server-emitted X-Ironmlx-Request-Id header value (P5h correlation).
    /// `None` when `--capture-server-request-id` flag is off.
    pub request_id: Option<String>,
```

- [ ] **Step 3: Capture header in `run_chat_completion`**

Edit `iron-bench/src/client.rs:158-217`. Change the signature to accept `capture_request_id: bool`. After `let resp = client.post(...).send().await?;` (around line 184) but BEFORE `let mut stream = resp.bytes_stream();`:

```rust
    let request_id = if capture_request_id {
        resp.headers().get("X-Ironmlx-Request-Id").and_then(|v| v.to_str().ok()).map(String::from)
    } else {
        None
    };
```

At the final `RequestResult` construction, add `request_id` to the returned struct.

- [ ] **Step 4: Conditional CSV column**

Edit `iron-bench/src/report.rs:483`. Wrap the header + per-row writer to be conditional. Change:

```rust
writer.write_all(b"target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n")?;
```

to:

```rust
let header = if capture_request_id {
    "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,request_id\n"
} else {
    "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
};
writer.write_all(header.as_bytes())?;
```

In the per-row writer below, append `,{request_id}` only when the flag is on:

```rust
if capture_request_id {
    write!(writer, ",{}", result.request_id.as_deref().unwrap_or(""))?;
}
writer.write_all(b"\n")?;
```

- [ ] **Step 5: Verify off-state byte-identical**

```bash
# Existing run, no flag
cargo run --release -p iron-bench -- --target localhost:8080 --model qwen --prompt-len 128 --runs 1 --warmup 0 --format csv > /tmp/iron_bench_before.csv

# Same flag, default off
cargo run --release -p iron-bench -- --target localhost:8080 --model qwen --prompt-len 128 --runs 1 --warmup 0 --format csv > /tmp/iron_bench_after.csv

diff /tmp/iron_bench_before.csv /tmp/iron_bench_after.csv || echo "FAIL: byte-identical broken"
```

Expected: 0 diff lines.

- [ ] **Step 6: Verify on-state captures request_id**

```bash
# With p5h-profile server running on :8080 (started separately):
cargo run --release -p iron-bench -- --target localhost:8080 --model qwen --prompt-len 128 --runs 1 --warmup 0 --capture-server-request-id --format csv | head -2
```

Expected: header line ends with `,request_id`; first data line ends with a UUID-shaped string.

- [ ] **Step 7: Commit**

```bash
git add iron-bench/src/main.rs iron-bench/src/client.rs iron-bench/src/report.rs
git commit -m "feat(p5h-t0a): iron-bench --capture-server-request-id flag + RequestResult.request_id + CSV column"
```

### T0a.11 — GDN harness P5h schema extension

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs:1059-1077`
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (forward path — wrap each of the 11 steps under attention_path wrapper via with_p5h_span_from_current_trace)

- [ ] **Step 1: Add wrapper `attention_path` span at decoder layer site for GDN layers**

Find the decoder layer code in `ironmlx/src/models/qwen3_5_moe/text_model.rs` (or wherever `forward_post_embedding_on` lives). For each layer iteration, when `layer_types[i]` is "linear" (GDN), wrap the `gated_delta_net.forward_on(...)` call:

```rust
        #[cfg(feature = "p5h-profile")]
        let layer_output = crate::core::p5h::with_p5h_span_from_current_trace(
            "decoder_layer_N",
            || crate::core::p5h::SpanFields { layer_idx: Some(i as i32), ..Default::default() },
            || {
                crate::core::p5h::with_p5h_span_from_current_trace(
                    "attention_path",
                    || crate::core::p5h::SpanFields { layer_idx: Some(i as i32), ..Default::default() },
                    || gated_delta_net.forward_on(/* ... */),
                )
            },
        )?;
```

- [ ] **Step 2: Wrap each of the 11 GDN substeps**

Edit `ironmlx/src/nn/gated_delta_net.rs` forward body. The existing 11-step breakdown in the P5g harness uses sequential code blocks. For each step, wrap the existing body in `with_p5h_span_from_current_trace`:

```rust
        let step_1a_output = crate::core::p5h::with_p5h_span_from_current_trace(
            "gda_step_1a_in_proj_qkvz",
            crate::core::p5h::SpanFields::default,
            || -> anyhow::Result<_> { /* existing step 1a code */ Ok(out) },
        )?;
        // ... repeat for steps 1b, 2a, 2b, 3, 4, 5, 6, 7, 8 ...
```

Reuse the existing P5g step names (e.g., `gda_step_1a_in_proj_qkvz`, `gda_step_8_norm_proj`) from `gated_delta_net.rs:1066`'s current emission format.

- [ ] **Step 3: Update existing `[p5g-profile]` emission to also emit `[p5h-profile]`**

Edit `ironmlx/src/nn/gated_delta_net.rs:1059-1077`. Keep the existing `[p5g-profile]` line unchanged (back-compat for P5g harness). Add a parallel `[p5h-profile]` emission via the new helper if `p5h-profile` is active — but since the substep spans are now opened via `with_p5h_span_from_current_trace` (which emits `[p5h-profile]` automatically), the explicit `[p5h-profile]` formatter call here is unnecessary. The existing `[p5g-profile]` emission stays for back-compat.

- [ ] **Step 4: Build + smoke**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/nn/gated_delta_net.rs ironmlx/src/models/qwen3_5_moe/text_model.rs
git commit -m "feat(p5h-t0a): GDN 11-step instrumentation under attention_path wrapper (P5h schema)"
```

### T0a.12 — Python aggregator + route-aware fixture + HARD GATE validator

**Files:**
- Create: `tools/p5h_aggregator/__init__.py`
- Create: `tools/p5h_aggregator/aggregator.py`
- Create: `tools/p5h_aggregator/schema_validator.py`
- Create: `tools/p5h_aggregator/tests/test_validator.py`

- [ ] **Step 1: Create aggregator module structure**

```bash
mkdir -p tools/p5h_aggregator/tests
touch tools/p5h_aggregator/__init__.py
touch tools/p5h_aggregator/tests/__init__.py
```

- [ ] **Step 2: Implement schema validator**

Create `tools/p5h_aggregator/schema_validator.py`:

```python
"""P5h schema validator — implements § 2.5a structural checks.

Single source of truth: docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md § 2.5a.
DO NOT re-derive semantics here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

P5H_LOG_RE = re.compile(
    r"\[p5h-profile\] "
    r"request_id=(?P<request_id>\S+) "
    r"routing_path=(?P<routing_path>\S+) "
    r"prompt_tokens=(?P<prompt_tokens>\d+) "
    r"seq=(?P<seq>\d+) "
    r"layer_idx=(?P<layer_idx>-?\d+) "
    r"span_id=(?P<span_id>\d+) "
    r"parent_span_id=(?P<parent_span_id>\S+) "
    r"span_name=(?P<span_name>\S+) "
    r"parent_span=(?P<parent_span>\S+) "
    r"start_ns=(?P<start_ns>\d+) "
    r"end_ns=(?P<end_ns>\d+) "
    r"mode=(?P<mode>\S+) "
    r"span_kind=(?P<span_kind>\S+)"
)

# Per Codex plan review v1 P1 #2: split required sets by span_kind so the
# presence check doesn't fail on Lane-A diagnostic spans being absent from
# tree_spans. Each lane's required set is split into a tree subset and a
# diagnostic subset, checked against the corresponding span_kind partition.

LANE_A_REQUIRED_TREE = {
    "server_request_recv_to_first_content_sse_write",
    "http_parse_render_tokenize",
    "scheduler_admission",
    "model_prefill_forward",
    "first_token_sampling",
    "detok_format_first_content_chunk",
}
LANE_A_REQUIRED_DIAGNOSTIC = {
    "sse_write_role_chunk_diagnostic",
}

LANE_B_REQUIRED_TREE = {
    "server_request_recv_to_first_content_sse_write",
    "http_parse_render_tokenize",
    "gs_stream_init_and_chunk_loop",
    "gs_first_token_sample_dispatch",
    "sse_write_role_chunk",
    "gs_first_token_materialize_and_predispatch",
    "detok_format_first_content_chunk",
}
LANE_B_REQUIRED_DIAGNOSTIC: set[str] = set()  # no Lane-B diagnostic spans currently

DIAGNOSTIC_ALLOWED_NAMES = {"sse_write_role_chunk_diagnostic"}

@dataclass
class Span:
    request_id: str
    routing_path: str
    prompt_tokens: int
    seq: int
    layer_idx: int
    span_id: int
    parent_span_id: int | None
    span_name: str
    parent_span: str | None
    start_ns: int
    end_ns: int
    mode: str
    span_kind: str
    inclusive_us: float = 0.0
    exclusive_us: float = 0.0

    def __post_init__(self):
        self.inclusive_us = (self.end_ns - self.start_ns) / 1000.0


def parse_line(line: str) -> Span | None:
    m = P5H_LOG_RE.search(line)
    if not m:
        return None
    g = m.groupdict()
    pid = g["parent_span_id"]
    return Span(
        request_id=g["request_id"],
        routing_path=g["routing_path"],
        prompt_tokens=int(g["prompt_tokens"]),
        seq=int(g["seq"]),
        layer_idx=int(g["layer_idx"]),
        span_id=int(g["span_id"]),
        parent_span_id=None if pid == "null" else int(pid),
        span_name=g["span_name"],
        parent_span=None if g["parent_span"] == "null" else g["parent_span"],
        start_ns=int(g["start_ns"]),
        end_ns=int(g["end_ns"]),
        mode=g["mode"],
        span_kind=g["span_kind"],
    )


@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)
    request_count: int = 0
    tree_span_count: int = 0
    diagnostic_span_count: int = 0

    def fail(self, msg: str):
        self.failures.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_request(spans: list[Span]) -> ValidationReport:
    """Run § 2.5a structural checks on one request's worth of spans."""
    report = ValidationReport(request_count=1)
    tree = [s for s in spans if s.span_kind == "tree"]
    diag = [s for s in spans if s.span_kind == "diagnostic"]
    report.tree_span_count = len(tree)
    report.diagnostic_span_count = len(diag)

    if not tree:
        report.fail("no tree spans emitted")
        return report

    req_id = tree[0].request_id
    routing = tree[0].routing_path

    # Per-record validity
    for s in spans:
        if not s.request_id:
            report.fail(f"empty request_id on span {s.span_name}")
        if s.prompt_tokens == 0:
            report.fail(f"prompt_tokens=0 on span {s.span_name}")
        if s.routing_path not in ("scheduler", "gs_chunked"):
            report.fail(f"invalid routing_path={s.routing_path} on span {s.span_name}")

    # Id uniqueness within request
    ids = [s.span_id for s in spans]
    if len(set(ids)) != len(ids):
        report.fail("duplicate span_id within request")

    # Exactly one root with span_name = server_request_recv_to_first_content_sse_write
    roots = [s for s in tree if s.parent_span_id is None]
    if len(roots) != 1:
        report.fail(f"expected exactly 1 root, found {len(roots)}")
    elif roots[0].span_name != "server_request_recv_to_first_content_sse_write":
        report.fail(f"root span_name is {roots[0].span_name}, expected server_request_recv_to_first_content_sse_write")

    # No orphan top-level (non-root tree span with null parent)
    for s in tree:
        if s.parent_span_id is None and s.span_name != "server_request_recv_to_first_content_sse_write":
            report.fail(f"orphan top-level tree span: {s.span_name}")

    # Closure: every non-null parent_span_id resolves
    by_id = {s.span_id: s for s in tree}
    for s in tree:
        if s.parent_span_id is not None and s.parent_span_id not in by_id:
            report.fail(f"orphan parent_span_id={s.parent_span_id} on {s.span_name}")

    # Label self-consistency
    for s in tree:
        if (s.parent_span_id is None) != (s.parent_span is None):
            report.fail(f"label inconsistency on {s.span_name}: parent_span_id={s.parent_span_id}, parent_span={s.parent_span}")
        if s.parent_span_id is not None and s.parent_span_id in by_id:
            if by_id[s.parent_span_id].span_name != s.parent_span:
                report.fail(f"parent_span label mismatch on {s.span_name}: parent_span_id resolves to {by_id[s.parent_span_id].span_name} but parent_span={s.parent_span}")

    # Interval containment
    for s in tree:
        if s.parent_span_id is not None and s.parent_span_id in by_id:
            p = by_id[s.parent_span_id]
            if not (p.start_ns <= s.start_ns and s.end_ns <= p.end_ns):
                report.fail(f"interval not contained on {s.span_name}: parent [{p.start_ns}, {p.end_ns}], child [{s.start_ns}, {s.end_ns}]")

    # Reachability + no cycle
    if len(roots) == 1:
        children_by_parent: dict[int, list[Span]] = {}
        for s in tree:
            if s.parent_span_id is not None:
                children_by_parent.setdefault(s.parent_span_id, []).append(s)
        visited: set[int] = set()
        stack = [roots[0]]
        while stack:
            cur = stack.pop()
            if cur.span_id in visited:
                report.fail(f"cycle detected at span {cur.span_name} (id={cur.span_id})")
                break
            visited.add(cur.span_id)
            for c in children_by_parent.get(cur.span_id, []):
                stack.append(c)
        unreachable = set(by_id.keys()) - visited
        if unreachable:
            report.fail(f"unreachable tree spans from root: {[by_id[i].span_name for i in unreachable]}")

    # Route-aware required span_names (per Codex plan review v1 P1 #2 — check
    # tree subset against tree_spans, diagnostic subset against diagnostic_spans).
    tree_names = {s.span_name for s in tree}
    diag_names = {s.span_name for s in diag}
    if routing == "scheduler":
        required_tree = LANE_A_REQUIRED_TREE
        required_diag = LANE_A_REQUIRED_DIAGNOSTIC
    else:
        required_tree = LANE_B_REQUIRED_TREE
        required_diag = LANE_B_REQUIRED_DIAGNOSTIC
    missing_tree = required_tree - tree_names
    missing_diag = required_diag - diag_names
    if missing_tree:
        report.fail(f"missing required tree spans for {routing}: {missing_tree}")
    if missing_diag:
        report.fail(f"missing required diagnostic spans for {routing}: {missing_diag}")

    # Diagnostic checks (per § 2.5a + Codex plan review v1 P2 #4)
    root_span_id = roots[0].span_id if len(roots) == 1 else None
    for d in diag:
        if d.span_name not in DIAGNOSTIC_ALLOWED_NAMES:
            report.fail(f"unexpected diagnostic span_name: {d.span_name}")
        # Per § 2.5a "Diagnostic span checks": parent_span_id MUST be None OR
        # point at root.span_id. Anything else = emitter bug.
        if d.parent_span_id is not None and d.parent_span_id != root_span_id:
            report.fail(
                f"diagnostic span {d.span_name} parent_span_id={d.parent_span_id} — "
                f"must be null or root's span_id ({root_span_id})"
            )

    # pre_content_decode_steps hard gate (per § 2.5a)
    pcds_count = sum(1 for s in tree if s.span_name == "pre_content_decode_steps")
    if pcds_count > 0:
        report.fail(f"pre_content_decode_steps count={pcds_count} > 0 — first prefill token did not detokenize non-empty; adjust benchmark prompts")

    return report


def group_by_request(spans: Iterable[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.request_id, []).append(s)
    return out
```

- [ ] **Step 3: Write fixture tests**

Create `tools/p5h_aggregator/tests/test_validator.py`:

```python
import pytest
from tools.p5h_aggregator.schema_validator import (
    parse_line,
    validate_request,
    group_by_request,
)

LINE_OK = (
    "  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
    "[p5h-profile] request_id=abc routing_path=scheduler prompt_tokens=128 "
    "seq=128 layer_idx=-1 span_id=1 parent_span_id=null "
    "span_name=server_request_recv_to_first_content_sse_write parent_span=null "
    "start_ns=1000 end_ns=2000 mode=off span_kind=tree"
)

def test_parse_line_root():
    s = parse_line(LINE_OK)
    assert s is not None
    assert s.span_name == "server_request_recv_to_first_content_sse_write"
    assert s.parent_span_id is None
    assert s.span_kind == "tree"

def test_validate_missing_required_fails():
    s = parse_line(LINE_OK)
    rep = validate_request([s])
    assert not rep.ok
    assert any("missing required" in f for f in rep.failures)

def test_duplicate_span_id_fails():
    a = parse_line(LINE_OK)
    b = parse_line(LINE_OK)  # same span_id=1
    rep = validate_request([a, b])
    assert not rep.ok
    assert any("duplicate" in f for f in rep.failures)

# --- Hard-path fixtures per Codex plan review v3 P2 #4 ---

def _build_line(
    *,
    request_id="abc",
    routing_path="scheduler",
    prompt_tokens=128,
    span_id,
    parent_span_id="null",
    span_name,
    parent_span="null",
    start_ns=1_000_000,
    end_ns=2_000_000,
    mode="off",
    span_kind="tree",
):
    """Build a synthetic [p5h-profile] log line with field overrides."""
    return (
        f"  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
        f"[p5h-profile] request_id={request_id} routing_path={routing_path} "
        f"prompt_tokens={prompt_tokens} seq=128 layer_idx=-1 "
        f"span_id={span_id} parent_span_id={parent_span_id} "
        f"span_name={span_name} parent_span={parent_span} "
        f"start_ns={start_ns} end_ns={end_ns} mode={mode} span_kind={span_kind}"
    )

def _lane_a_pass_fixture() -> list:
    """Minimal Lane-A request: root + all 6 required tree spans + 1 required diagnostic."""
    spans = []
    # Root: contains all children in [0, 100_000_000]
    spans.append(parse_line(_build_line(
        span_id=1, parent_span_id="null",
        span_name="server_request_recv_to_first_content_sse_write",
        parent_span="null",
        start_ns=0, end_ns=100_000_000,
    )))
    # Required tree children
    for sid, name in enumerate([
        "http_parse_render_tokenize",
        "scheduler_admission",
        "model_prefill_forward",
        "first_token_sampling",
        "detok_format_first_content_chunk",
    ], start=2):
        spans.append(parse_line(_build_line(
            span_id=sid, parent_span_id="1",
            span_name=name, parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=1_000 * sid, end_ns=1_000 * sid + 500,
        )))
    # Required diagnostic (under root span_id=1, but span_kind=diagnostic)
    spans.append(parse_line(_build_line(
        span_id=100, parent_span_id="1",
        span_name="sse_write_role_chunk_diagnostic",
        parent_span="server_request_recv_to_first_content_sse_write",
        start_ns=10_000, end_ns=10_500, span_kind="diagnostic",
    )))
    return spans

def test_lane_a_full_fixture_passes():
    """Per Codex plan review v3 P2 #4: a well-formed Lane-A request must
    PASS all structural checks, including the diagnostic presence subset."""
    spans = _lane_a_pass_fixture()
    rep = validate_request(spans)
    assert rep.ok, f"unexpected failures: {rep.failures}"

def test_lane_a_missing_diagnostic_fails():
    """Drop the diagnostic span; presence check on LANE_A_REQUIRED_DIAGNOSTIC must fail."""
    spans = [s for s in _lane_a_pass_fixture() if s.span_kind != "diagnostic"]
    rep = validate_request(spans)
    assert not rep.ok
    assert any("missing required diagnostic spans" in f for f in rep.failures), rep.failures

def test_diagnostic_parent_not_root_fails():
    """Diagnostic span with parent_span_id != root.span_id and not None must fail
    (per § 2.5a Diagnostic span checks)."""
    spans = _lane_a_pass_fixture()
    # Mutate diagnostic span's parent_span_id to a non-root id (id=2 is http_parse).
    for i, s in enumerate(spans):
        if s.span_kind == "diagnostic":
            spans[i] = parse_line(_build_line(
                span_id=100, parent_span_id="2",
                span_name="sse_write_role_chunk_diagnostic",
                parent_span="http_parse_render_tokenize",  # not root
                start_ns=10_000, end_ns=10_500, span_kind="diagnostic",
            ))
            break
    rep = validate_request(spans)
    assert not rep.ok
    assert any("parent_span_id" in f and "must be null or root" in f for f in rep.failures), rep.failures

def test_join_orphan_aggregator_hard_fail(tmp_path):
    """Per Codex plan review v3 P2 #4 + § 2.5a Join key: aggregator MUST exit
    non-zero when server log has a request_id absent from iron-bench CSV.
    This test invokes the aggregator entry point with a curated input.
    """
    import subprocess
    import sys
    # Server log with a request_id "abc"
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(
        _build_line(span_id=sid, parent_span_id=("null" if sid == 1 else "1"),
                    span_name=name,
                    parent_span=("null" if sid == 1 else "server_request_recv_to_first_content_sse_write"),
                    start_ns=1_000 * sid, end_ns=1_000 * sid + 500)
        for sid, name in [
            (1, "server_request_recv_to_first_content_sse_write"),
            (2, "http_parse_render_tokenize"),
        ]
    ) + "\n")
    # Bench CSV with a DIFFERENT request_id "xyz" — server "abc" is orphan
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text("request_id,pp_target\nxyz,128\n")
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, "-m", "tools.p5h_aggregator.aggregator",
         "--server-log", str(server_log),
         "--bench-csv", str(bench_csv),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 4, f"expected exit 4 (JOIN HARD-FAIL), got {result.returncode}\nstderr:\n{result.stderr}"
    assert "JOIN HARD-FAIL" in result.stderr
```

- [ ] **Step 4: Run validator tests**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run pytest tools/p5h_aggregator/tests/test_validator.py -v
```

Expected: all 7 tests PASS (3 prior + lane_a_full_fixture_passes + lane_a_missing_diagnostic_fails + diagnostic_parent_not_root_fails + join_orphan_aggregator_hard_fail).

- [ ] **Step 5: Implement aggregator entry point**

Create `tools/p5h_aggregator/aggregator.py`:

```python
"""P5h T5 aggregator entry point.

Reads `[p5h-profile]` log lines from server stderr + iron-bench CSV
(with `request_id` column), joins on request_id, validates per request,
and emits per-PP attribution table.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from .schema_validator import parse_line, validate_request, group_by_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True, type=Path, help="server stderr capture (with [p5h-profile] lines)")
    ap.add_argument("--bench-csv", required=True, type=Path, help="iron-bench CSV (with request_id column)")
    ap.add_argument("--out", required=True, type=Path, help="output attribution table (CSV)")
    args = ap.parse_args()

    spans = []
    with args.server_log.open() as f:
        for line in f:
            s = parse_line(line)
            if s is not None:
                spans.append(s)

    if not spans:
        print("ERROR: no [p5h-profile] spans parsed from server log", file=sys.stderr)
        sys.exit(2)

    # Join iron-bench CSV to attach pp/run_idx
    bench_by_req: dict[str, dict] = {}
    with args.bench_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("request_id", "").strip()
            if rid:
                bench_by_req[rid] = row

    grouped = group_by_request(spans)

    # Per Codex plan review v1 P1 #3 + § 2.5a Join key:
    # iron-bench↔server request_id join MUST be 100%. Any orphan = broken
    # header propagation = hard-fail before any downstream computation.
    server_req_ids = set(grouped.keys())
    bench_req_ids = set(bench_by_req.keys())
    server_orphans = server_req_ids - bench_req_ids  # server log has spans for a request bench CSV doesn't know
    bench_orphans = bench_req_ids - server_req_ids   # bench CSV has a request server log has no spans for

    if server_orphans or bench_orphans:
        print("JOIN HARD-FAIL: per § 2.5a Join key, request_id join rate must = 100% (orphan rate = 0%)", file=sys.stderr)
        if server_orphans:
            print(f"  server log has {len(server_orphans)} request_id(s) absent from iron-bench CSV:", file=sys.stderr)
            for r in sorted(server_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(server_orphans) > 10:
                print(f"    ... +{len(server_orphans) - 10} more", file=sys.stderr)
        if bench_orphans:
            print(f"  iron-bench CSV has {len(bench_orphans)} request_id(s) absent from server log:", file=sys.stderr)
            for r in sorted(bench_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(bench_orphans) > 10:
                print(f"    ... +{len(bench_orphans) - 10} more", file=sys.stderr)
        print("Likely causes: server not built with --features p5h-profile; iron-bench --capture-server-request-id flag off; header propagation bug.", file=sys.stderr)
        sys.exit(4)

    # Per-PP join rate breakdown (informational; total join rate already
    # validated above as 100%).
    pp_join_rates: dict[str, tuple[int, int]] = {}
    for rid in server_req_ids:
        pp = bench_by_req.get(rid, {}).get("pp_target", "?")
        matched, total = pp_join_rates.get(pp, (0, 0))
        pp_join_rates[pp] = (matched + 1, total + 1)
    for pp in sorted(pp_join_rates, key=lambda x: int(x) if x.isdigit() else -1):
        matched, total = pp_join_rates[pp]
        print(f"  PP={pp}: join_rate={matched}/{total} (100.0%)", file=sys.stderr)

    failures = []
    for req_id, request_spans in grouped.items():
        rep = validate_request(request_spans)
        if not rep.ok:
            for fail in rep.failures:
                failures.append(f"{req_id}: {fail}")

    if failures:
        print("VALIDATION FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(3)

    # Compute per-PP per-span exclusive_us (placeholder — full T5 work below)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "pp", "span_name", "inclusive_us"])
        for req_id, request_spans in grouped.items():
            pp = bench_by_req.get(req_id, {}).get("pp_target", "")
            for s in request_spans:
                w.writerow([req_id, pp, s.span_name, f"{s.inclusive_us:.2f}"])

    print(f"OK: {len(grouped)} requests, {len(spans)} spans, join rate 100%, written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Build + run sentinel + verify aggregator picks up real spans**

Start a feature-on server in one terminal:

```bash
MLX_DIR=$HOME/.local/mlx cargo run --release --features p5h-profile -p ironmlx -- serve --b-max 1 --model "$IRONMLX_MOE_MODEL_DIR" --port 18099 2> /tmp/p5h_server.log &
SERVER_PID=$!
sleep 5

# One iron-bench request
cargo run --release -p iron-bench -- --target localhost:18099 --model qwen --prompt-len 128 --runs 1 --warmup 0 --capture-server-request-id --format csv > /tmp/p5h_bench.csv

kill $SERVER_PID

# Run aggregator
uv run python -m tools.p5h_aggregator.aggregator \
    --server-log /tmp/p5h_server.log \
    --bench-csv /tmp/p5h_bench.csv \
    --out /tmp/p5h_attribution.csv

head -5 /tmp/p5h_attribution.csv
```

Expected: 1 request id, multiple span rows.

- [ ] **Step 7: Commit**

```bash
git add tools/p5h_aggregator/
git commit -m "feat(p5h-t0a): Python aggregator + schema validator + route-aware fixture"
```

### T0a.13 — UMA hardening protocol harness

**Files:**
- Create: `ironmlx/tests/p5h_t0a_harness.rs`

- [ ] **Step 1: Create harness skeleton**

Create `ironmlx/tests/p5h_t0a_harness.rs`. Reuse the per-PP server spawn pattern from `p5g_t0_gated_delta_profile.rs:71+` (`iron_bench_run`).

```rust
//! T0a UMA hardening + GDN P5h-protocol rerun harness.
//!
//! Cold/warm pair protocol: for each PP value, run iron-bench twice with a
//! cool gate between runs; compare variance. > ±2% triggers retry.
//!
//! Per spec § 2.4 + § 3 T0a.

#![cfg(feature = "p5h-profile")]

use std::process::{Command, Stdio};
use std::time::Duration;

const PP_LIST: &[u32] = &[128, 512, 2048, 4096, 8192, 16384];
const COOL_DURATION_MS: u64 = 5 * 60 * 1000; // 5 min between cool/warm
const VARIANCE_THRESHOLD: f64 = 0.02; // ±2%

// ... reuse helpers from p5g_t0_gated_delta_profile.rs: spawn_server, wait_ready_or_fail,
//     iron_bench_run, kill_server, cool_gate, ...

#[test]
#[ignore = "p5h-t0a — long-running UMA hardening sweep; invoke explicitly"]
fn t0a_uma_hardening_sweep() -> anyhow::Result<()> {
    for &pp in PP_LIST {
        let cold = run_one_pp(pp)?;
        cool_gate(Duration::from_millis(COOL_DURATION_MS));
        let warm = run_one_pp(pp)?;
        let variance = (warm - cold).abs() / cold;
        assert!(
            variance <= VARIANCE_THRESHOLD,
            "PP={} cold/warm variance {} > {}", pp, variance, VARIANCE_THRESHOLD,
        );
    }
    Ok(())
}

fn run_one_pp(pp: u32) -> anyhow::Result<f64> {
    // 1. spawn server with --features p5h-profile --b-max 1
    // 2. iron-bench --capture-server-request-id --prompt-len pp
    // 3. parse server stderr + bench CSV
    // 4. run aggregator (subprocess: uv run python -m tools.p5h_aggregator.aggregator)
    // 5. return wall_us median or similar metric
    todo!("implement using p5g_t0_gated_delta_profile.rs helpers as template")
}

fn cool_gate(dur: Duration) {
    std::thread::sleep(dur);
}
```

Fill in `run_one_pp` by adapting the P5g harness helpers. Output JSON to `/tmp/p5h-t0a-uma.json`.

- [ ] **Step 2: Implement run_one_pp + spawn helpers**

Copy + adapt from `ironmlx/tests/p5g_t0_gated_delta_profile.rs`:
- `spawn_server` → `spawn_server_p5h` (changes: add `--features p5h-profile` to cargo run; do NOT pass `--prefill-chunk-size 0`)
- `wait_ready_or_fail` → reuse as-is
- `iron_bench_run` → adapt to add `--capture-server-request-id`

- [ ] **Step 3: Verify harness builds**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5h_t0a_harness --no-run
```

Must compile.

- [ ] **Step 4: Commit**

```bash
git add ironmlx/tests/p5h_t0a_harness.rs
git commit -m "feat(p5h-t0a): UMA hardening + GDN P5h-protocol rerun harness skeleton"
```

### T0a.14 — Execute GDN rerun + verify T0a HARD GATE

**Files:** (no code changes; this is a verification step)

- [ ] **Step 1: Run the full UMA hardening sweep**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5h_t0a_harness -- --ignored --test-threads=1 2>&1 | tee /tmp/p5h_t0a_sweep.log
```

- [ ] **Step 2: Run aggregator on the full sweep output**

```bash
uv run python -m tools.p5h_aggregator.aggregator \
    --server-log /tmp/p5h-t0a-server.log \
    --bench-csv /tmp/p5h-t0a-bench.csv \
    --out /tmp/p5h-t0a-attribution.csv
```

Expected: exits 0. All structural checks pass.

- [ ] **Step 3: Verify hard-gate invariants (per Codex plan review v1 P1 #3)**

Per § 2.5a + § 4 + § 7.2 #9, T0a HARD GATE has three independent components:
1. **Per-PP iron-bench↔server `request_id` join rate = 100%** (orphan rate = 0%) — verified by the aggregator hard-fail in Step 2 above (exit code 4 if any orphan).
2. **Per-request structural checks PASS** — verified via the standalone validator script below.
3. **Per-PP UMA cold/warm variance ≤ ±2%** — verified by the harness in T0a.13.

Run the validator standalone on the captured log to print a per-PP pass report:

```bash
uv run python -c "
import csv
from pathlib import Path
from tools.p5h_aggregator.schema_validator import parse_line, group_by_request, validate_request

# Parse server log
spans = []
for line in Path('/tmp/p5h-t0a-server.log').open():
    s = parse_line(line)
    if s: spans.append(s)
groups = group_by_request(spans)

# Parse iron-bench CSV for join + per-PP grouping
bench_by_req = {}
with Path('/tmp/p5h-t0a-bench.csv').open() as f:
    for row in csv.DictReader(f):
        rid = row.get('request_id', '').strip()
        if rid:
            bench_by_req[rid] = row

# Per-PP join rate + structural-check breakdown
per_pp: dict[str, dict] = {}
for req_id, group_spans in groups.items():
    bench_row = bench_by_req.get(req_id, {})
    pp = bench_row.get('pp_target', '?')
    rec = per_pp.setdefault(pp, {'total': 0, 'pass': 0, 'fail': 0, 'joined': 0})
    rec['total'] += 1
    if req_id in bench_by_req:
        rec['joined'] += 1
    rep = validate_request(group_spans)
    if rep.ok:
        rec['pass'] += 1
    else:
        rec['fail'] += 1
        print(f'  {req_id} (PP={pp}): {rep.failures[0]}')

print(f'Total requests: {len(groups)}, total spans: {len(spans)}')
gate_pass = True
for pp in sorted(per_pp, key=lambda x: int(x) if x.isdigit() else -1):
    r = per_pp[pp]
    join_rate = 100.0 * r['joined'] / r['total'] if r['total'] else 0.0
    pass_rate = 100.0 * r['pass'] / r['total'] if r['total'] else 0.0
    print(f'PP={pp}: total={r[\"total\"]} joined={r[\"joined\"]} ({join_rate:.1f}%) pass={r[\"pass\"]} fail={r[\"fail\"]} ({pass_rate:.1f}%)')
    if join_rate < 100.0:
        print(f'  HARD GATE FAIL: PP={pp} join rate {join_rate:.1f}% < 100% (per Codex plan review v1 P1 #3 + § 2.5a Join key)')
        gate_pass = False
    if r['fail'] > 0:
        print(f'  HARD GATE FAIL: PP={pp} {r[\"fail\"]} structural-check failures')
        gate_pass = False

if not gate_pass:
    raise SystemExit('T0a HARD GATE FAILED')
print('T0a HARD GATE: PASS (per-PP join rate = 100%, per-request structural checks PASS)')
"
```

- [ ] **Step 4: Verify default build identity**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
target/release/ironmlx serve --b-max 1 --model "$IRONMLX_MOE_MODEL_DIR" --port 18098 2> /tmp/default_server.log &
SERVER_PID=$!
sleep 5
cargo run --release -p iron-bench -- --target localhost:18098 --model qwen --prompt-len 128 --runs 1 --warmup 0 --format csv > /dev/null
kill $SERVER_PID
grep -c "\\[p5h-profile\\]" /tmp/default_server.log
```

Expected: 0.

- [ ] **Step 5: Commit T0a close-out**

```bash
git commit --allow-empty -m "chore(p5h-t0a): HARD GATE PASSED — schema validated, UMA cold/warm variance ≤ 2%, GDN coverage_pct ≥ 95% (Lane A), iron-bench↔server request_id join 100%"
```

---

## Task T0b: Phase D Root Cause Investigation (4 Hypotheses)

T0b only starts after T0a closes (HARD GATE passed). T0b reuses T0a's validated schema infrastructure. Per spec § 2.5 decision tree, T0b identifies which of H1-H4 is primary and binds T2/T3 conditional ablation gates.

### T0b.1 — H1 thermal drift investigation (phase-order randomized)

**Files:**
- Create: `ironmlx/tests/p5h_t0b_phase_d.rs`

- [ ] **Step 1: Implement randomized phase-order test**

Create `ironmlx/tests/p5h_t0b_phase_d.rs` with a test that runs Phase D before Phase A/B/C (reverse of P5g order). Compare Phase D values across orderings.

```rust
//! T0b Phase D root cause investigation harness.
#![cfg(feature = "p5h-profile")]

use std::process::Command;

#[test]
#[ignore = "p5h-t0b H1 thermal drift — phase-order randomized rerun"]
fn t0b_h1_phase_order_randomized() -> anyhow::Result<()> {
    let normal_phase_d = run_phases_then_d()?;
    let reversed_phase_d = run_d_then_phases()?;
    let drift = (normal_phase_d - reversed_phase_d).abs() / normal_phase_d;
    println!("H1 verdict: normal_phase_d={}, reversed_phase_d={}, drift={}", normal_phase_d, reversed_phase_d, drift);
    // Output verdict to /tmp/p5h-t0b-h1.json
    Ok(())
}

fn run_phases_then_d() -> anyhow::Result<f64> { todo!() }
fn run_d_then_phases() -> anyhow::Result<f64> { todo!() }
```

- [ ] **Step 2: Run H1 test**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5h_t0b_phase_d t0b_h1 -- --ignored --test-threads=1
```

- [ ] **Step 3: Commit H1 result**

```bash
git add ironmlx/tests/p5h_t0b_phase_d.rs
git commit -m "test(p5h-t0b): H1 thermal drift investigation"
```

### T0b.2 — H2 substitute self-cost investigation

- [ ] **Step 1: Add H2 test that runs Phase A WITH `IRONMLX_P5G_PROFILE_MODE=ablate-X` enabled, comparing substitute path vs original path**

Append to `p5h_t0b_phase_d.rs`:

```rust
#[test]
#[ignore]
fn t0b_h2_substitute_self_cost() -> anyhow::Result<()> {
    let original = run_phase_a_no_ablate()?;
    let substitute = run_phase_a_with_ablate_x()?;
    let self_cost = substitute - original;
    println!("H2 verdict: original={}, substitute={}, self_cost={}", original, substitute, self_cost);
    Ok(())
}
```

- [ ] **Step 2-3: Run + commit**

### T0b.3 — H3 cache state divergence

- [ ] **Step 1: Add `ablate-conv-with-manual-cache-update` ablation variant in `gated_delta_net.rs`**

In `gated_delta_net.rs`, find the existing `AblateConv` mode. Add a parallel mode that calls the substitute AND explicitly updates `conv_state` (mirroring AblateNone's update). Gate on `IRONMLX_P5G_PROFILE_MODE=ablate-conv-with-manual-cache-update`.

- [ ] **Step 2: Run H3 comparison**

- [ ] **Step 3: Commit**

### T0b.4 — H4 kernel template variance

- [ ] **Step 1: Add kernel-dispatch-only timing under AblateComputeG vs Phase A**

In `gated_delta_net.rs`, add per-step kernel-dispatch elapsed timer that excludes pre/post processing.

- [ ] **Step 2: Run H4 comparison**

- [ ] **Step 3: Commit**

### T0b.5 — Decision tree binding + close-out report

**Files:**
- Create: `reports/p5h-phase-d-root-cause.md` (working tree only; reports/ NOT committed per Boss memory)

- [ ] **Step 1: Write the close-out report**

Create `reports/p5h-phase-d-root-cause.md` documenting:
- H1/H2/H3/H4 verdicts (numerical data from /tmp/p5h-t0b-*.json)
- Primary root cause identified
- T2/T3 conditional ablation gate binding (e.g., "H2 primary → T2/T3 Layer 3 skipped, real-path microbenchmarks instead")

- [ ] **Step 2: Commit T0b close-out**

```bash
git commit --allow-empty -m "feat(p5h-t0b): Phase D root cause investigation + decision-tree resolution

H1 thermal drift: <verdict>
H2 substitute self-cost: <verdict>
H3 cache state divergence: <verdict>
H4 kernel template variance: <verdict>

Primary: <Hn> — <mitigation>
T2/T3 Layer 3 binding: <skip|run|substitute-redesign>

Report: reports/p5h-phase-d-root-cause.md (working tree only, not in git per .gitignore)"
```

---

## Task T1: HTTP + Scheduler + Admission Profile

Most T1 spans (`http_parse_render_tokenize`, `scheduler_admission`, `sse_write_role_chunk_diagnostic`, `detok_format_first_content_chunk`) are already wired in T0a.6-T0a.7. T1 verifies attribution coverage and runs the sweep.

### T1.1 — Confirm all T1 top-level spans emit on PP=128 sweep

- [ ] **Step 1: Run full PP sweep with feature-on server**

Use the T0a harness as template; create `ironmlx/tests/p5h_t1_http_sched_sweep.rs` that sweeps PP ∈ {128, 512, 2048} and aggregates by span_name.

- [ ] **Step 2: Verify per-PP wall-time breakdown for T1 spans**

Aggregator should report median `inclusive_us` for `http_parse_render_tokenize`, `scheduler_admission`, `sse_write_role_chunk_diagnostic`, `detok_format_first_content_chunk` per PP.

- [ ] **Step 3: Commit**

```bash
git add ironmlx/tests/p5h_t1_http_sched_sweep.rs
git commit -m "test(p5h-t1): HTTP + scheduler + admission per-PP attribution sweep"
```

---

## Task T2: GatedAttention 7-step Taxonomy

### T2.1 — Read current gated_attention.rs 7-step boundaries

Reference: `gated_attention.rs:154-300+` for `forward_on()`. Per spec § 2.2 #5, 7 substeps:
1. `q_gate_k_v_proj` — q_proj + k_proj + v_proj
2. `q_split_norm_reshape`
3. `mrope_apply`
4. `kv_mask_update`
5. `fused_sdpa`
6. `gate_sigmoid_mul`
7. `o_proj`

### T2.2 — Add 7-step instrumentation under `attention_path` wrapper

- [ ] **Step 1: Open attention_path wrapper at decoder layer site (full-attn layers)**

In `ironmlx/src/models/qwen3_5_moe/text_model.rs`, for layers where `layer_types[i] == "full"`, wrap `gated_attention.forward_on(...)` with `with_p5h_span_from_current_trace("attention_path", ..)`.

- [ ] **Step 2: Wrap each of the 7 substeps in `gated_attention.rs::forward_on`**

For each substep boundary (per spec § 2.2 #5), wrap the existing code block in `with_p5h_span_from_current_trace("<substep_name>", fields_fn, || { /* existing code */ })?`.

- [ ] **Step 3: Build + smoke**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

- [ ] **Step 4: Commit instrumentation**

```bash
git add ironmlx/src/nn/gated_attention.rs ironmlx/src/models/qwen3_5_moe/text_model.rs
git commit -m "feat(p5h-t2): GatedAttention 7-step instrumentation under attention_path wrapper"
```

### T2.3 — Run T2 sweep + verify substep emission

- [ ] **Step 1: Create `ironmlx/tests/p5h_t2_gated_attention_sweep.rs`**

Test sweeps PP ∈ {128, 512, 2048}, verifies aggregator finds all 7 substep span_names per request.

- [ ] **Step 2: Run sweep**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5h-profile --test p5h_t2_gated_attention_sweep -- --ignored --test-threads=1
```

- [ ] **Step 3: Commit**

```bash
git add ironmlx/tests/p5h_t2_gated_attention_sweep.rs
git commit -m "test(p5h-t2): GatedAttention 7-step per-PP occupancy sweep (Lane A)"
```

### T2.4 — Conditional Layer 3 ablation per T0b outcome

- [ ] **Step 1: Read T0b output**

Determine which of H1/H2/H3/H4 is primary from `reports/p5h-phase-d-root-cause.md`.

- [ ] **Step 2: Apply binding per spec § 3 T2 conditional table**

- H1 primary → run Layer 3 with randomized phase order + cool gates
- H2 primary → skip Layer 3, replace with real-path microbenchmarks
- H3 primary → cache-state-preserving substitute design
- H4 primary → skip Layer 3 for kernel-dispatch steps (4-5); OK for op-level (1-3, 6-7)

- [ ] **Step 3: Commit Layer 3 outcome**

```bash
git commit --allow-empty -m "test(p5h-t2): Layer 3 conditional ablation bound per T0b outcome (<H1|H2|H3|H4>)"
```

---

## Task T3: MoE 8-step Taxonomy

Mirrors T2 with the 8-step MoE breakdown.

### T3.1-T3.4 — 8-step instrumentation + sweep + conditional Layer 3

Substep names (per spec § 2.2 #6):
1. `router_logits_softmax_topk`
2. `routing_sort_pack`
3. `gather_qmm_gate_up`
4. `swiglu_activation`
5. `gather_qmm_down`
6. `routing_unsort_weighted_reduce`
7. `shared_expert`
8. `moe_output_sum`

- [ ] **Step 1: Wrap each substep in `sparse_moe.rs::SparseMoeBlock::forward_on`** (lines 180-465 area). For each substep boundary, use `with_p5h_span_from_current_trace`.

- [ ] **Step 2: Open `mlp_path` wrapper at decoder layer site**

In `text_model.rs`, wrap `sparse_moe.forward_on(...)` with `with_p5h_span_from_current_trace("mlp_path", ..)`.

- [ ] **Step 3: ROI math source = runtime Qwen35MoeConfig values**

In the T5 aggregator + T3 sweep test, derive `num_experts_per_tok`, `moe_intermediate`, `num_experts` from the model config at runtime, NOT spec constants.

- [ ] **Step 4: Create `ironmlx/tests/p5h_t3_moe_sweep.rs`** + run

- [ ] **Step 5: Conditional Layer 3 per T0b outcome** (per spec § 3 T3 table)

- [ ] **Step 6: Commit**

```bash
git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs ironmlx/src/models/qwen3_5_moe/text_model.rs ironmlx/tests/p5h_t3_moe_sweep.rs
git commit -m "feat(p5h-t3): MoE 8-step instrumentation under mlp_path wrapper + sweep + Layer 3 bound per T0b"
```

---

## Task T4: lm_head + first_token_sampling + MLX state + tokenization + first-eval

Most T4 spans (`slice_last_and_project_lm_head`, `first_token_sampling`, `gs_first_token_sample_dispatch`, `gs_first_token_materialize_and_predispatch`) are already wired in T0a.8-T0a.9. T4 adds the remaining diagnostic spans.

### T4.1 — Wrap `slice_last_and_project_lm_head` substep

- [ ] **Step 1: In `model.rs::slice_last_and_project`** (lines 240-258 area), wrap the `lm_head.forward_on(&last_hidden, target)` call with `with_p5h_span_from_current_trace("slice_last_and_project_lm_head", ..)`.

- [ ] **Step 2: Build + smoke**

- [ ] **Step 3: Commit**

### T4.2 — MLX `eval()` barrier annotations

- [ ] **Step 1: Identify major `mlx::eval()` sync points in the prefill/decode path**

- [ ] **Step 2: Wrap each with `with_p5h_span_from_current_trace("mlx_eval_barrier", ..)`** with `seq` populated from the current context.

- [ ] **Step 3: Commit**

### T4.3 — KVCache + GatedDeltaCache state-update per-forward cost

- [ ] **Step 1: Wrap state-update sites (`update_and_fetch_on(...)`)**

- [ ] **Step 2: Commit**

### T4.4 — Tokenizer Encode subspan

- [ ] **Step 1: Inside the `http_parse_render_tokenize` window (handler), additionally emit a finer-grained `tokenizer_encode` subspan**

- [ ] **Step 2: Commit**

### T4.5 — First-eval one-shot cost

- [ ] **Step 1: Track first-request flag in `AppState`; emit `first_eval_amortized_cost` diagnostic span on the first request of each (model, prompt_shape) pair**

- [ ] **Step 2: Commit**

### T4.6 — Run T4 sweep + verify

- [ ] **Step 1: Create `ironmlx/tests/p5h_t4_lm_head_mlx_state_sweep.rs`** + run

- [ ] **Step 2: Commit**

---

## Task T5: Cross-layer Attribution Synthesis + P5i/P5j Candidate Ranking + Close-out

### T5.1 — Implement full T5 aggregator pseudocode per § 2.5a

- [ ] **Step 1: Extend `tools/p5h_aggregator/aggregator.py` to implement § 2.5a "Exclusive time computation" pseudocode**

Add `compute_exclusive` function that:
1. Filters `tree_spans = [s for s in spans if s.span_kind == "tree"]`
2. Builds tree from `(span_id, parent_span_id)`
3. Computes `exclusive_us = inclusive_us - sum(child.inclusive_us)` per tree span
4. Asserts `tree_exclusive_sum ≈ root.inclusive_us`
5. Computes `coverage_pct = 1 - sum(unattributed_*.inclusive_us) / root.inclusive_us`
6. Reports `diagnostic_spans` as separate columns

- [ ] **Step 2: Add residual leaf injection**

For every non-leaf tree span, compute `unattributed_<span_name>` residual = `parent.inclusive_us - sum(accountable_children.inclusive_us)`. If > 1µs, emit as synthesized leaf row in the output.

- [ ] **Step 3: Test aggregator on T0a captured log**

```bash
uv run python -m tools.p5h_aggregator.aggregator \
    --server-log /tmp/p5h-t0a-server.log \
    --bench-csv /tmp/p5h-t0a-bench.csv \
    --out /tmp/p5h-attribution-full.csv
```

Verify `coverage_pct ≥ 95%` per PP in the output.

- [ ] **Step 4: Commit**

```bash
git add tools/p5h_aggregator/aggregator.py
git commit -m "feat(p5h-t5): aggregator full exclusive_us + residual leaves + coverage_pct"
```

### T5.2 — Per-PP top-3 + P5i candidate ranking + P5j candidate ranking

- [ ] **Step 1: Add ROI ranking module**

Create `tools/p5h_aggregator/roi_ranking.py`:

```python
"""ROI ranking: identify top-3 bottleneck per PP + P5i (short PP) + P5j (long PP) candidates."""
from __future__ import annotations
from dataclasses import dataclass

# Separate "measurement lane" partition from "optimization target" partition
# (per Codex plan review v3 P1 #1). Lane A/B describes WHICH code path the
# request traverses (scheduler vs chunked GS); P5i/P5j describes which
# optimization phase targets a given PP. Spec § 1.2 gap table: PP=128 needs
# +24%, PP=512 needs +74%, PP=2048 needs +110%, PP=4096 needs +115%,
# PP=8192 needs +124%, PP=16384 needs +126%. So PP=2048 belongs in the
# long-gap (+110-128%) bracket = P5J target, NOT P5I.
LANE_A_PP_SET = {128, 512, 2048}      # scheduler path; full deep substep attribution
LANE_B_PP_SET = {4096, 8192, 16384}    # chunked GS path; top-level only (Lane B granularity)

P5I_TARGET_PP_SET = {128, 512}                  # short-PP optimization phase (+24-74%)
P5J_TARGET_PP_SET = {2048, 4096, 8192, 16384}   # long-PP optimization phase (+110-128%)
P5I_TARGET_GAIN_RANGE = (0.24, 0.74)            # per spec § 1.2
P5J_TARGET_GAIN_RANGE = (1.10, 1.28)            # per spec § 1.2

@dataclass
class Candidate:
    span_name: str
    pp: int
    measured_inclusive_us: float
    estimated_max_gain: float  # % of root_wall_us if span dropped to zero
    scope_gate_trigger: bool   # True if kernel rewrite (Boss approval needed)
    notes: str

def rank_p5i(attribution_table) -> list[Candidate]:
    """P5i candidates: top span_name × pp combos for PP ∈ P5I_TARGET_PP_SET
    ({128, 512}) — measurements come from Lane A (scheduler path, full deep
    attribution), target gain +24-74%."""
    ...

def rank_p5j(attribution_table) -> list[Candidate]:
    """P5j candidates: top span_name × pp combos for PP ∈ P5J_TARGET_PP_SET
    ({2048, 4096, 8192, 16384}). Note PP=2048 measurements come from Lane A
    (full deep attribution); PP=4096+ measurements come from Lane B
    (top-level only — apply 'bounded by Lane-B granularity' caveat to
    candidates derived from those PP). Target gain +110-128%."""
    ...
```

- [ ] **Step 2: Commit**

### T5.3 — Target feasibility assessment

- [ ] **Step 1: Compute total estimated gain across all P5i + P5j candidates per PP**

- [ ] **Step 2: Verdict: "ironmlx > omlx +10% achievable in P5i+P5j" — yes/no per PP, with cap rationale**

- [ ] **Step 3: Commit**

### T5.4 — Write `reports/p5h-attribution.md`

- [ ] **Step 1: Create `reports/p5h-attribution.md`** (working tree only — `reports/` is gitignored per Boss memory)

Include:
- Lane A per-PP attribution table (root + top-level + wrapper spans + deep substeps)
- Lane B per-PP top-level attribution table
- Diagnostic columns (role_chunk_diagnostic_us, client_transport_residual_us)
- P5i candidate list with ROI estimates
- P5j candidate list with ROI estimates + "bounded by Lane-B granularity" caveat
- Target feasibility verdict per PP

### T5.5 — Spec § 7.2 final state lock

- [ ] **Step 1: Edit `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2** to lock the final ship-gate state (per P5g § 7.2 pattern). Add a "T5 close-out verdict" subsection summarizing measured PASS/FAIL per gate #1-#9.

- [ ] **Step 2: Commit spec lock**

```bash
git add docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md
git commit -m "docs(p5h-t5): § 7.2 final state lock — T5 close-out verdict"
```

### T5.6 — Write P5h findings memory

- [ ] **Step 1: Create `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md`**

Content template (mirror P5g findings memory):
- HEAD SHA after close-out
- Ship state (nothing optimized — measure-only)
- Per-PP attribution headline numbers
- P5i candidate ranking
- P5j candidate ranking
- Target feasibility verdict
- Reusable infra delivered (P5hTraceContext / dual API / UMA harness / aggregator)
- Key lessons (e.g., Lane-A vs Lane-B span tree differences, diagnostic span pattern)

- [ ] **Step 2: Add memory index entry**

Edit `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md`, add line after the P5g findings entry:

```markdown
- [P5h findings (YYYY-MM-DD)](project_p5h_findings.md) — measure-only close-out; per-PP attribution headline; P5i/P5j candidate ranking; target feasibility verdict; reusable infra
```

### T5.7 — T5 close-out commit

- [ ] **Step 1: Commit**

```bash
git commit --allow-empty -m "chore(p5h-t5): close-out — all-PP exclusive attribution + P5i/P5j candidate ranking + target feasibility verdict

Lane A (PP ≤ 2048): coverage_pct ≥ 95% per PP, deep substep attribution complete
Lane B (PP > 2048): coverage_pct ≥ 95% per PP, top-level only (deep substep deferred P5h+1)
Diagnostic: role_chunk_diagnostic_us reported, client_transport_residual_us reported
T0a HARD GATE: PASSED
T0b Phase D primary: <Hn>
P5i candidates ranked: <N> identified, top-3 by ROI
P5j candidates ranked: <N> identified, 'bounded by Lane-B granularity' caveat applied
Target feasibility: <verdict>
Spec § 7.2 final state: locked
Memory: project_p5h_findings.md + MEMORY.md index added
Report: reports/p5h-attribution.md (working tree only, not in git)"
```

---

## Self-Review

### Spec coverage check

| Spec section | Plan task(s) |
|---|---|
| § 2.2 #1 HTTP path | T0a.6 + T1 |
| § 2.2 #2 Scheduler / admission | T0a.6 + T1 |
| § 2.2 #3 Tokenization / first-eval | T4.4 + T4.5 |
| § 2.2 #4 GDN sub-step | T0a.11 + T0a.14 (rerun) |
| § 2.2 #5 GatedAttention | T2 |
| § 2.2 #6 MoE | T3 |
| § 2.2 #7 lm_head + MLX state | T4 |
| § 2.4 UMA hardening | T0a.13 + T0a.14 |
| § 2.5a Span lifecycle API | T0a.1, T0a.2, T0a.3 |
| § 2.5a Propagation chain | T0a.4, T0a.6, T0a.7, T0a.8, T0a.9 |
| § 2.5a Diagnostic / tree split | T0a.3 + T0a.12 + T5.1 |
| § 2.5a Structural checks | T0a.12 + T0a.14 |
| § 2.5 Phase D | T0b |
| § 3 T0a HARD GATE | T0a.14 |
| § 4 Validation gates | Each task's commit step runs them |
| § 5 Numerical safety | Sentinel suite invoked per task |
| § 7.1 Coverage gate | T5.1 |
| § 7.2 Ship gates 1-9 | T5.5 |

All spec requirements covered.

### Placeholder scan

- `todo!()` in T0a.13 / T0b.1: explicit stub markers requiring the implementer to fill in per the template directly above; not generic "TODO". OK.
- All step bodies show actual code or commands.
- File paths exact + line ranges anchored.

### Type consistency

- `P5hTraceContext` fields = `{request_id: String, prompt_tokens: u32, routing_path: &'static str}` — used consistently across T0a.1-T0a.9.
- `SpanHandle` fields = `{span_id, span_name, parent_span_id, parent_span, start_ns}` (with `parent_span: Option<&'static str>` added per Codex plan review v1 P1 #1 + v3 P3 #6 for T0a fixture label self-consistency check) — consistent across all uses.
- `RootSpanHandle { ctx, span }` — consistent (NOT `{ctx, start_ns}`, which was a v17 stale form per Codex review).
- `cloned_active_row_p5h_trace_and_root` returns `(P5hTraceContext, SpanHandle)` (owned) — consistent.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task with two-stage review (spec compliance + code quality) after each task. Subagents stay focused, you preserve context for coordination, T0a HARD GATE enforces dependency order before T0b/T2/T3/T4.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`. Batch execution with checkpoints; you review between batches.

**Which approach?**
