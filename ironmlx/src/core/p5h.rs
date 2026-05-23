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
    /// P5h+1 T2 Lane-B chunk context stack. Entered via the
    /// `P5hChunkContextGuard` RAII (`enter_chunk_context`); read by
    /// `emit_log_line_*` to populate the `chunk_idx` field on every span
    /// emitted while a `gs_chunk_N` ancestor is active. The stack supports
    /// nesting in case a future site nests chunks (current Lane-B
    /// `GenerationStream::new` loop opens one chunk at a time, so depth is
    /// effectively 1, but the stack semantics keep the helper symmetric with
    /// `P5H_CURRENT_SPAN_STACK`).
    pub(crate) static P5H_CURRENT_CHUNK_STACK: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard that owns the top of `P5H_CURRENT_CHUNK_STACK`. Created by
/// `enter_chunk_context`; the caller MUST bind it to a local
/// (`let _chunk_guard = enter_chunk_context(idx)`) so the guard lives for the
/// chunk body's scope and Drop fires on every exit path (including `?`
/// early-returns from the closure body).
///
/// Manual push/pop is intentionally NOT exposed: an early `?` return between
/// a push and a manual pop would leak the chunk_idx on the stack and
/// contaminate the next chunk's span emission with a stale ancestor id.
pub(crate) struct P5hChunkContextGuard {
    chunk_idx: u32,
    active: bool,
}

pub(crate) fn enter_chunk_context(chunk_idx: u32) -> P5hChunkContextGuard {
    P5H_CURRENT_CHUNK_STACK.with(|s| s.borrow_mut().push(chunk_idx));
    P5hChunkContextGuard {
        chunk_idx,
        active: true,
    }
}

fn current_chunk_idx() -> Option<u32> {
    P5H_CURRENT_CHUNK_STACK.with(|s| s.borrow().last().copied())
}

impl Drop for P5hChunkContextGuard {
    fn drop(&mut self) {
        if self.active {
            P5H_CURRENT_CHUNK_STACK.with(|s| {
                let popped = s.borrow_mut().pop();
                assert_eq!(
                    popped,
                    Some(self.chunk_idx),
                    "P5hChunkContextGuard dropped out of order"
                );
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct P5hTraceContext {
    pub request_id: String,
    pub prompt_tokens: u32,
    pub routing_path: &'static str, // "scheduler" | "gs_chunked"
}

/// Span handle returned by `open_p5h_span[_at]` and pushed onto the
/// per-thread `P5H_CURRENT_SPAN_STACK` by `with_p5h_span_from_current_trace`.
///
/// Fields are **truly private** (per Codex plan review v6 P2 — `pub(crate)`
/// in v4 only restricted construction outside the crate, but ALL ironmlx
/// modules including `openai.rs` / `scheduler.rs` could still mutate or
/// construct ad-hoc instances). Construction is gated through
/// `open_p5h_span[_at]` + `with_p5h_span_from_current_trace`; cross-module
/// reads use the `pub(crate)` accessors below; mutation is impossible from
/// outside `core::p5h` because no `pub(crate)` mutating method exists.
///
/// Sub-modules of `core::p5h` (notably `#[cfg(test)] mod tests`) DO see the
/// private fields per Rust's child-module visibility rule — this is required
/// so the tamper-detection test can synthesize a "mutated clone" scenario
/// that production code outside the module physically cannot reach. The
/// close-side `handle_snapshot` check inside `registry_remove_or_panic`
/// remains as defense in depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanHandle {
    span_id: u64,
    span_name: &'static str,
    parent_span_id: Option<u64>,
    /// Human-readable parent label retained for log readability + label
    /// self-consistency check (per spec § 2.5a structural checks: "parent_span_id
    /// resolves to a span whose span_name equals the parent_span label").
    /// Set at open from the parent's `span_name`; None for root.
    /// Per Codex plan review v1 P1 #1: do NOT hard-code "explicit_parent" —
    /// emitter MUST carry the real parent label to match T0a fixture
    /// `label self-consistency` assertion in § 2.5a.
    parent_span: Option<&'static str>,
    start_ns: u64,
}

impl SpanHandle {
    pub(crate) fn span_id(&self) -> u64 {
        self.span_id
    }
    pub(crate) fn span_name(&self) -> &'static str {
        self.span_name
    }
    pub(crate) fn parent_span_id(&self) -> Option<u64> {
        self.parent_span_id
    }
    pub(crate) fn parent_span(&self) -> Option<&'static str> {
        self.parent_span
    }
    pub(crate) fn start_ns(&self) -> u64 {
        self.start_ns
    }
}

#[derive(Default, Debug)]
pub struct SpanFields {
    pub layer_idx: Option<i32>,
    /// Lane-B chunk index (zero-based). Explicit value set by `gs_chunk_N`
    /// open; inherited automatically by all descendant spans via the
    /// `P5H_CURRENT_CHUNK_STACK` thread-local fallback inside
    /// `emit_log_line_*`. Stays `None` (rendered as `null` on the wire) for
    /// all spans emitted outside a Lane-B chunk body (Lane-A entirely; Lane-B
    /// pre-loop sites such as `gs_kv_cache_alloc`; Lane-B post-loop sites
    /// such as `gs_first_token_sample_dispatch`).
    pub chunk_idx: Option<u32>,
    pub seq: Option<u32>,
    pub mode: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct RootSpanHandle {
    ctx: P5hTraceContext,
    span: SpanHandle,
}

impl RootSpanHandle {
    pub(crate) fn new(ctx: P5hTraceContext, span: SpanHandle) -> Self {
        RootSpanHandle { ctx, span }
    }

    pub(crate) fn ctx(&self) -> &P5hTraceContext {
        &self.ctx
    }
    pub(crate) fn span(&self) -> &SpanHandle {
        &self.span
    }

    pub(crate) fn close_at(self, end_ns: u64) {
        close_p5h_span(&self.ctx, self.span, end_ns, SpanFields::default());
    }

    /// Per Codex plan review v12 P2 #6 — close the root span on a
    /// pre-first-content abort path (Lane-A role-send failure, Lane-B
    /// `GenerationStream::new` Err, Lane-B role-send failure).
    ///
    /// Same registry + log-line emission as `close_at`, but writes
    /// `mode = "aborted"` via `SpanFields { mode: Some("aborted"), .. }` so
    /// the T5 aggregator can exclude this request from coverage gates (first
    /// content was never sent, so the tree intentionally lacks
    /// `detok_format_first_content_chunk` and may lack later spans). The
    /// `mode` field already exists on `SpanFields` from T0a.1; the aggregator
    /// + validator just need to recognize the new "aborted" value.
    pub(crate) fn close_at_aborted(self, end_ns: u64) {
        let fields = SpanFields {
            mode: Some("aborted"),
            ..SpanFields::default()
        };
        close_p5h_span(&self.ctx, self.span, end_ns, fields);
    }
}

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
            assert!(
                stack.is_empty(),
                "P5hTraceGuard::enter with non-empty span stack — prior guard leaked"
            );
            stack.push(base_parent);
        });
        P5hTraceGuard {
            _marker: std::marker::PhantomData,
        }
    }
}

impl Drop for P5hTraceGuard {
    fn drop(&mut self) {
        P5H_CURRENT_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            assert_eq!(
                stack.len(),
                1,
                "P5hTraceGuard::drop with span stack length {} — expected 1 (only base_parent sentinel). \
                 Either an inner span was opened without close, or close was called more times than open.",
                stack.len(),
            );
            stack.clear();
        });
        P5H_CURRENT_TRACE.with(|c| *c.borrow_mut() = None);
    }
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

/// P5h+1 T1 measurement-probe global. When `true`, selected ROI substep
/// closures call `mlx::transforms::eval` on their returned `Array` value(s)
/// before the span closes so each substep accrues the incremental MLX
/// materialization cost it caused (closes the wrapper-dominance gap reported
/// by P5h T5: lazy MLX graph defers ~96-99% of root_inclusive_us to the
/// outermost `.to_vec()` materialization site).
///
/// Production default OFF preserves lazy-graph semantics — only flipped on by
/// the `--p5h-measurement-eval-probes` CLI flag (which is itself gated by the
/// `p5h-profile` feature). Feature-off builds never set this flag and read it
/// via the inline always-`false` fallback in `is_measurement_eval_probes_active`.
#[cfg(feature = "p5h-profile")]
static MEASUREMENT_EVAL_PROBES_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Set the global measurement-eval-probes flag. Called once at server boot
/// from `server::serve` based on the `--p5h-measurement-eval-probes` CLI flag.
#[cfg(feature = "p5h-profile")]
pub fn set_measurement_eval_probes_active(active: bool) {
    MEASUREMENT_EVAL_PROBES_ACTIVE.store(active, Ordering::Relaxed);
}

/// Read the global measurement-eval-probes flag. Always available; returns
/// `false` in feature-off builds via the inline branch (no atomic load).
#[inline]
pub fn is_measurement_eval_probes_active() -> bool {
    #[cfg(feature = "p5h-profile")]
    {
        MEASUREMENT_EVAL_PROBES_ACTIVE.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "p5h-profile"))]
    {
        false
    }
}

/// Records the full state captured at open. close_* verifies the incoming
/// SpanHandle matches this record on every field (per Codex plan review v5
/// P2 + v6 P2 — `SpanHandle` fields are truly private + Clone, so production
/// callers outside `core::p5h` cannot construct or mutate handles directly,
/// but defense-in-depth catches any in-module mistake or future regression.
/// A sub-module test in this file mutates `span_name` / `parent_span_id` /
/// `parent_span` / `start_ns` on a
/// clone and emit an inconsistent log line; close-side tamper detection
/// catches it). request_id is the ctx-mismatch check (v3 P2 #3).
#[derive(Clone, Debug)]
struct OpenSpanRecord {
    request_id: String,
    /// Snapshot of every SpanHandle field at open time; close compares
    /// against the incoming handle. Any mutation between open and close
    /// triggers tamper-detection panic.
    handle_snapshot: SpanHandle,
}

/// Per Codex plan review v6 P1 + v7 P1: registry is a single eagerly-constructed
/// HashMap state — no `Option<HashMap<...>>` (v5 design) and no two-branch
/// "NeverOpened/Unknown" failure mode. `Mutex::new(HashMap::new())` is NOT
/// usable as a `static` initializer because `HashMap::new` is not `const fn`
/// (E0015 on rustc 1.95.0), so we wrap in `once_cell::sync::Lazy` which is
/// already a dependency (added in T0a.3 Step 0 for `monotonic_ns()`).
static OPEN_SPAN_REGISTRY: once_cell::sync::Lazy<Mutex<HashMap<u64, OpenSpanRecord>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Insert a fresh open record. Returns Err if the span_id is already
/// present (atomic-counter race or registry corruption; never expected
/// under correct usage). Caller panics outside the lock — per Codex plan
/// review v5 P1, panicking while holding the Mutex guard poisons it and
/// breaks subsequent #[should_panic] tests.
fn registry_try_insert(span_id: u64, record: OpenSpanRecord) -> Result<(), u64> {
    let mut reg = OPEN_SPAN_REGISTRY.lock().expect("p5h registry poisoned");
    if reg.contains_key(&span_id) {
        return Err(span_id);
    }
    reg.insert(span_id, record);
    Ok(())
}

fn registry_insert(span_id: u64, record: OpenSpanRecord) {
    if let Err(duplicate_id) = registry_try_insert(span_id, record) {
        panic!(
            "open_p5h_span issued duplicate span_id={} — atomic counter race or registry corruption",
            duplicate_id,
        );
    }
}

/// Remove the open record for `span_id`. Returns None if the id is absent.
/// Single error condition: "is not in open registry" (covers both
/// double-close and never-opened — distinguishing them was dead-code
/// branching per Codex v6 P1).
fn registry_try_remove(span_id: u64) -> Option<OpenSpanRecord> {
    let mut reg = OPEN_SPAN_REGISTRY.lock().expect("p5h registry poisoned");
    reg.remove(&span_id)
}

/// Verify `handle` matches the record stored at open; panic if mismatched.
/// Panic happens AFTER the Mutex guard is released (per Codex plan review
/// v5 P1) — registry_try_remove returns the owned record, we inspect it
/// outside any lock.
fn registry_remove_or_panic(handle: &SpanHandle, expected_request_id: &str) {
    let record = registry_try_remove(handle.span_id).unwrap_or_else(|| panic!(
        "close_p5h_span(span_name={}, span_id={}) — span_id is not in open registry. \
         Causes: (a) handle reused after close (double-close), (b) handle leaked from a different request, \
         (c) handle never opened. Per § 2.5a explicit-API hard-fail.",
        handle.span_name, handle.span_id,
    ));
    // ctx mismatch (per Codex v3 P2 #3)
    if record.request_id != expected_request_id {
        panic!(
            "close_p5h_span(span_name={}, span_id={}) — ctx mismatch: opened with request_id={}, closing with request_id={}. \
             Cross-request handle leakage. Per Codex plan review v3 P2 #3 + § 2.5a explicit-API hard-fail.",
            handle.span_name, handle.span_id, record.request_id, expected_request_id,
        );
    }
    // Tamper detection (per Codex v5 P2): every SpanHandle field MUST equal
    // the snapshot recorded at open. Mutating clone before close is a bug.
    let snap = &record.handle_snapshot;
    if handle.span_name != snap.span_name
        || handle.parent_span_id != snap.parent_span_id
        || handle.parent_span != snap.parent_span
        || handle.start_ns != snap.start_ns
    {
        panic!(
            "close_p5h_span(span_id={}) — handle field tamper detected.\n  \
              opened: span_name={}, parent_span_id={:?}, parent_span={:?}, start_ns={}\n  \
              closing: span_name={}, parent_span_id={:?}, parent_span={:?}, start_ns={}\n\
             Per Codex plan review v5 P2 + § 2.5a explicit-API hard-fail.",
            handle.span_id,
            snap.span_name,
            snap.parent_span_id,
            snap.parent_span,
            snap.start_ns,
            handle.span_name,
            handle.parent_span_id,
            handle.parent_span,
            handle.start_ns,
        );
    }
}

fn monotonic_ns() -> u64 {
    use std::time::Instant;
    static ANCHOR: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    ANCHOR.elapsed().as_nanos() as u64
}

pub(crate) fn monotonic_ns_public() -> u64 {
    monotonic_ns()
}

fn next_span_id() -> u64 {
    NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed)
}

fn emit_log_line_with_end_ns(
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
    //
    // P5h+1 T2: `chunk_idx` is inserted between `layer_idx` and `span_id`.
    // Explicit `fields.chunk_idx` wins; otherwise inherit from
    // `P5H_CURRENT_CHUNK_STACK` top (set by the RAII guard at chunk-body
    // entry). `None` renders as `null` per other Option fields.
    let chunk_idx = fields.chunk_idx.or_else(current_chunk_idx);
    let chunk_idx_str = chunk_idx
        .map(|c| c.to_string())
        .unwrap_or_else(|| "null".to_string());
    tracing::info!(
        "[p5h-profile] request_id={} routing_path={} prompt_tokens={} seq={} layer_idx={} \
         chunk_idx={} span_id={} parent_span_id={} span_name={} parent_span={} \
         start_ns={} end_ns={} mode={} span_kind={}",
        ctx.request_id,
        ctx.routing_path,
        ctx.prompt_tokens,
        fields.seq.unwrap_or(0),
        fields.layer_idx.unwrap_or(-1),
        chunk_idx_str,
        span.span_id,
        span.parent_span_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "null".to_string()),
        span.span_name,
        span.parent_span.unwrap_or("null"),
        span.start_ns,
        end_ns,
        fields.mode.unwrap_or("off"),
        span_kind,
    );
}

/// Per Option E (T0a coverage gap fix): emit a log line where `end_ns` is
/// captured AS LATE AS POSSIBLE — immediately before `tracing::info!` fires.
/// This pulls the cost of the caller's post-body infrastructure
/// (`fields_fn` evaluation + `registry_remove_or_panic` + format pre-work)
/// INTO `inclusive_us = end_ns - start_ns`, instead of leaving it as
/// invisible gap between sibling substeps. Without this, ~3-5us per span of
/// hidden cost capped GDN coverage at ~50%. Used by
/// `with_p5h_span_from_current_trace`; explicit-context closers
/// (`close_p5h_span` / `close_p5h_span_diagnostic`) still take a
/// caller-provided `end_ns` via `emit_log_line_with_end_ns` because the
/// caller controls the semantic moment of close (e.g., root span at
/// handler exit, Lane-A/B forwarder boundaries).
fn emit_log_line_capture_end_ns(
    ctx: &P5hTraceContext,
    span: &SpanHandle,
    fields: &SpanFields,
    span_kind: &'static str,
) {
    let end_ns = monotonic_ns();
    emit_log_line_with_end_ns(ctx, span, end_ns, fields, span_kind);
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
    let handle = SpanHandle {
        span_id,
        span_name,
        parent_span_id: parent.map(|p| p.span_id),
        // Per Codex plan review v1 P1 #1: carry real parent label, NOT
        // "explicit_parent" placeholder. Label self-consistency check in T0a
        // fixture asserts parent_span_id resolves to a span whose span_name
        // equals this label.
        parent_span: parent.map(|p| p.span_name),
        start_ns,
    };
    // Snapshot the full handle so close-side tamper detection can catch any
    // mid-flight mutation (per Codex plan review v5 P2).
    registry_insert(
        span_id,
        OpenSpanRecord {
            request_id: ctx.request_id.clone(),
            handle_snapshot: handle.clone(),
        },
    );
    handle
}

/// Close an explicit-context tree span. Emits the `[p5h-profile]` log line.
/// Per Codex plan review v1 P2 #5 + v3 P2 #3 + v5 P2: hard-fail if span_id
/// is not in the open registry, ctx.request_id doesn't match, OR any
/// SpanHandle field was mutated since open (catches handle reuse /
/// cross-request leakage / double-close / wrong-ctx close / field tamper).
pub fn close_p5h_span(ctx: &P5hTraceContext, handle: SpanHandle, end_ns: u64, fields: SpanFields) {
    registry_remove_or_panic(&handle, &ctx.request_id);
    emit_log_line_with_end_ns(ctx, &handle, end_ns, &fields, "tree");
}

/// Close a diagnostic span (e.g., Lane-A `sse_write_role_chunk_diagnostic`).
/// Per § 2.5a v19 P1: diagnostic spans are NOT in the exclusive tree.
pub fn close_p5h_span_diagnostic(
    ctx: &P5hTraceContext,
    handle: SpanHandle,
    end_ns: u64,
    fields: SpanFields,
) {
    registry_remove_or_panic(&handle, &ctx.request_id);
    emit_log_line_with_end_ns(ctx, &handle, end_ns, &fields, "diagnostic");
}

/// Implicit-guard API. Internally opens span (parent = stack top), pushes,
/// runs body, pops, closes. Panics if no active guard. Per § 2.5a.
///
/// **Panic-during-body semantics (per spec § 2.5a Hard-fail under p5h-profile + `--b-max 1`):**
///
/// If `body()` panics, this function does NOT execute the post-body pop +
/// `registry_remove_or_panic` + `emit_log_line` block. On unwind:
///   1. `OPEN_SPAN_REGISTRY` retains the open record (logical leak)
///   2. `P5H_CURRENT_SPAN_STACK` retains the pushed handle (length 2)
///   3. When the enclosing `P5hTraceGuard` drops, its `assert_eq!(stack.len(), 1, ...)`
///      fires a SECOND panic during stack unwind → Rust aborts the process.
///
/// This is **intentional** under the P5h fail-fast contract: P5h instrumentation
/// runs under `--b-max 1` in single-request serial sweeps; a panic inside
/// instrumented model code is a real bug that MUST stop the process so the
/// operator notices. Silent recovery would mask the bug AND leave inconsistent
/// state for any subsequent request. The "registry leak" only materializes if a
/// caller wraps the panic in `catch_unwind` and continues running — which P5h
/// does not do anywhere in the authorized guard sites (§ 2.5a). If a future
/// authorized site needs catch-unwind semantics, that site must add its own
/// scope-exit cleanup; do NOT add panic-safe pop/remove inside this helper —
/// it would hide bugs the spec wants to surface.
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
        let top = stack.last().unwrap_or_else(|| {
            panic!(
                "with_p5h_span_from_current_trace(span_name={}) called with empty span stack — \
             guard active but stack not seeded (base_parent missing)",
                span_name,
            )
        });
        (top.span_id, top.span_name)
    });
    let span_id = next_span_id();
    let handle = SpanHandle {
        span_id,
        span_name,
        parent_span_id: Some(parent_id),
        // Per Codex plan review v1 P1 #1: carry real parent label for T0a
        // label self-consistency assertion.
        parent_span: Some(parent_label),
        start_ns,
    };
    registry_insert(
        span_id,
        OpenSpanRecord {
            request_id: request_id_at_open.clone(),
            handle_snapshot: handle.clone(),
        },
    );
    P5H_CURRENT_SPAN_STACK.with(|s| s.borrow_mut().push(handle.clone()));
    let result = body();
    let popped = P5H_CURRENT_SPAN_STACK.with(|s| {
        s.borrow_mut().pop().unwrap_or_else(|| {
            panic!(
                "stack underflow in with_p5h_span_from_current_trace(span_name={})",
                span_name,
            )
        })
    });
    assert_eq!(
        popped.span_id, handle.span_id,
        "stack imbalance: popped a different span ({}) than the one opened ({})",
        popped.span_name, handle.span_name
    );
    // Per Option E (T0a coverage gap fix): do NOT capture `end_ns` here. The
    // post-body cost of `fields_fn()` + `registry_remove_or_panic()` +
    // `tracing::info!` format pre-work (~3-5us per span) must be INCLUDED in
    // `inclusive_us = end_ns - start_ns`, otherwise it surfaces as invisible
    // gap between sibling substeps and caps GDN coverage at ~50%. The actual
    // `let end_ns = monotonic_ns()` lives in `emit_log_line_capture_end_ns`
    // immediately before `tracing::info!` fires.
    let fields = fields_fn();
    registry_remove_or_panic(&handle, &request_id_at_open);
    P5H_CURRENT_TRACE.with(|c| {
        let ctx_ref = c.borrow();
        let ctx = ctx_ref.as_ref().unwrap_or_else(|| panic!(
            "with_p5h_span_from_current_trace(span_name={}) lost P5H_CURRENT_TRACE mid-body — guard dropped concurrently",
            span_name,
        ));
        emit_log_line_capture_end_ns(ctx, &handle, &fields, "tree");
    });
    result
}

/// Per Codex plan review v12 P1 #1 — non-OpenAI entry paths (anthropic.rs / CLI / tests
/// / scheduler_actor internals) keep `GenerateRequest.p5h_trace` + `.p5h_root_span` at
/// `None`, so `cloned_active_row_p5h_trace_and_root()` returns `Ok(None)` and the SINK
/// in `prefill_admitted_inner` skips opening `model_prefill_forward` (no guard entered,
/// `P5H_CURRENT_TRACE` stays None). But `model.batched_prefill[_vl](...)` still runs and
/// flows through `DecoderLayerMoe::forward_on` → GDN/full-attn substeps; if those deep
/// sites unconditionally called `with_p5h_span_from_current_trace(...)`, the strict
/// helper above would panic on missing `P5H_CURRENT_TRACE` and break every non-OpenAI
/// caller under `--features p5h-profile`.
///
/// Codex recommended fix (option A): keep the strict helper's fail-fast contract intact;
/// deep instrumentation sites use this `try_` variant that no-ops when no active trace.
/// All deep callers (T0a.11 decoder layer + GDN substeps; T2 GatedAttention substeps;
/// T3 MoE substeps; T4 lm_head) MUST use `try_with_p5h_span_from_current_trace`. The
/// strict variant is reserved for sites where ctx is provably populated — currently
/// none in P5h Phase 1, but we keep it for future fail-fast diagnostics (e.g., a span
/// that should only be opened inside a known-OpenAI guarded region).
///
/// Per Codex plan review v20 P1 #1 + P5h+1 T2 — Lane-B (`routing_path ==
/// "gs_chunked"`) emission allow-list. Original P5h scope was TOP-LEVEL-ONLY
/// because (a) deep emission under chunked GS would multiply records per
/// request by N chunks without a `chunk_idx` schema field to disambiguate
/// and (b) the T0a.14 hard gate would have enforced GDN coverage on a lane
/// that lacked deep instrumentation. P5h+1 T2 lifts both blockers:
///
///   * Schema: `SpanFields.chunk_idx: Option<u32>` is now emitted on every
///     span; `P5H_CURRENT_CHUNK_STACK` populates the field automatically
///     for any span emitted under a `gs_chunk_N` ancestor (via the RAII
///     `P5hChunkContextGuard` in `core::generate::GenerationStream::new`).
///   * Coverage gates: P5h+1 closes the Lane-B `gs_chunk_N` wrapper
///     dominance (97-99% of root_inclusive) by opening the full decoder
///     hierarchy here so T1's per-substep eval probes light up on Lane B.
///
/// When the active ctx is Lane-B, this helper checks `span_name` against
/// this allow-list; names NOT on it no-op even though `P5H_CURRENT_TRACE`
/// is Some (defense in depth against accidental Lane-A names leaking into
/// Lane-B emission). The list now spans the full decoder hierarchy:
///   * Lane-B top-level: `gs_kv_cache_alloc`, `gs_chunk_N`,
///     `gs_first_token_sample_dispatch`.
///   * Decoder wrappers + per-step substeps: `decoder_layer_N`,
///     `input_norm`, `attention_path`, `residual_overhead`,
///     `post_attention_norm`, `mlp_path`.
///   * T2 GatedAttention substeps under `attention_path`.
///   * T3 MoE substeps under `mlp_path` (incl. shared expert + routing).
///   * GDN 11 substeps under `attention_path` for hybrid models.
///   * Cache + lm_head: `cache_state_update`,
///     `slice_last_and_project_lm_head`.
///
/// Names retain Lane-A emission semantics — they are accepted on Lane-A
/// unconditionally by the strict `"scheduler"` branch. Only Lane-B is
/// allow-list-gated.
const LANE_B_ALLOWED_TRY_SPAN_NAMES: &[&str] = &[
    "gs_kv_cache_alloc",
    "gs_chunk_N",
    "gs_first_token_sample_dispatch",
    "decoder_layer_N",
    "input_norm",
    "attention_path",
    "residual_overhead",
    "post_attention_norm",
    "mlp_path",
    "q_gate_k_v_proj",
    "q_split_norm_reshape",
    "mrope_apply",
    "kv_mask_update",
    "fused_sdpa",
    "gate_sigmoid_mul",
    "o_proj",
    "router_logits_softmax_topk",
    "routing_sort_pack",
    "gather_qmm_gate_up",
    "swiglu_activation",
    "gather_qmm_down",
    "routing_unsort_weighted_reduce",
    "shared_expert",
    "moe_output_sum",
    "cache_state_update",
    "slice_last_and_project_lm_head",
    "gda_step_1a_in_proj_qkvz",
    "gda_step_1b_in_proj_ba",
    "gda_step_2a_prepend_conv_state",
    "gda_step_2b_conv1d_silu",
    "gda_step_2c_update_conv_state",
    "gda_step_3_split_reshape_per_head",
    "gda_step_4_qk_rmsnorm",
    "gda_step_5_compute_g",
    "gda_step_6_sigmoid_beta",
    "gda_step_7_kernel_and_cache_update",
    "gda_step_8_norm_proj",
];

pub fn try_with_p5h_span_from_current_trace<T>(
    span_name: &'static str,
    fields_fn: impl FnOnce() -> SpanFields,
    body: impl FnOnce() -> T,
) -> T {
    let routing: Option<&'static str> =
        P5H_CURRENT_TRACE.with(|c| c.borrow().as_ref().map(|ctx| ctx.routing_path));
    match routing {
        // Non-OpenAI path under --features p5h-profile: no active trace, run body
        // directly with no span open/close. Symmetric to default-build behavior
        // (cfg!(feature = "p5h-profile") = false → wrapper compiled out entirely).
        None => body(),
        // Lane-A: full deep emission via the strict helper.
        Some("scheduler") => with_p5h_span_from_current_trace(span_name, fields_fn, body),
        // Lane-B: top-level-only per Codex v20 P1 #1. Only allow-listed span
        // names emit; everything else no-ops.
        Some("gs_chunked") => {
            if LANE_B_ALLOWED_TRY_SPAN_NAMES.contains(&span_name) {
                with_p5h_span_from_current_trace(span_name, fields_fn, body)
            } else {
                body()
            }
        }
        // Unknown routing_path: emitter bug. Panic on the canonical fail-fast
        // discipline of this module (consistent with `with_p5h_span_*` panics
        // on missing ctx). The only legal values are the two enumerated in
        // `P5hTraceContext.routing_path` and validated in § 2.5a.
        Some(other) => panic!(
            "try_with_p5h_span_from_current_trace: unknown routing_path={:?} \
             on active P5H_CURRENT_TRACE (only 'scheduler' or 'gs_chunked' \
             permitted per spec § 2.5a) — span_name={}",
            other, span_name,
        ),
    }
}

/// Per Codex plan review v14 P1 #1 — RAII guard that OWNS the optional
/// `RootSpanHandle` and exposes accessor + once-close methods. Earlier v13
/// design held `&'a mut Option<RootSpanHandle>` borrowed from an outer
/// variable, which made the outer variable unusable for the rest of the
/// scope (borrow checker rejected `root_to_close.as_ref()` / `.take()` at
/// every subsequent callsite — happy-path content_chunk parent lookup AND
/// root close BOTH inaccessible). The owning-Option pattern lets callers
/// hold `let mut root_guard = P5hRootCloseGuard::new(root)` for the entire
/// closure body and call `.span()` + `.close_success(end_ns)` without
/// fighting borrow rules; Drop still runs `close_at_aborted` on any
/// pre-first-content terminal path (panic, early return, async cancel).
pub struct P5hRootCloseGuard {
    root: Option<RootSpanHandle>,
}

impl P5hRootCloseGuard {
    /// Wrap the root for once-close + RAII abort cleanup.
    pub(crate) fn new(root: RootSpanHandle) -> Self {
        Self { root: Some(root) }
    }

    /// True iff the root is still open (no success close + no abort close yet).
    /// Use this in per-iteration loops to gate "open another child span under
    /// root" — once `close_success` has fired, root is closed and later
    /// children would orphan / panic on a closed parent.
    pub(crate) fn is_open(&self) -> bool {
        self.root.is_some()
    }

    /// Borrow root's `SpanHandle` to set as `parent` on child spans. Panics
    /// if root has already been closed (caller bug: gating on `is_open()` was
    /// missed). Borrow is short-lived (just to read parent metadata at open
    /// time), so no borrow checker conflict with subsequent `close_success`.
    pub(crate) fn span(&self) -> &SpanHandle {
        self.root
            .as_ref()
            .expect("P5hRootCloseGuard::span called after root closed")
            .span()
    }

    /// Happy-path close: success first-content sent at `end_ns`. Takes the
    /// root out of the guard so Drop becomes a no-op. Panics if called twice
    /// — that means the caller advanced `first_non_empty_content` state
    /// twice without resetting.
    pub(crate) fn close_success(&mut self, end_ns: u64) {
        let root = self
            .root
            .take()
            .expect("P5hRootCloseGuard::close_success called twice");
        root.close_at(end_ns);
    }
}

impl Drop for P5hRootCloseGuard {
    fn drop(&mut self) {
        // Pre-first-content terminal path: root still open at drop time
        // (no close_success call ever fired). Emit close_at_aborted with
        // current monotonic_ns — covers ALL Lane-A early returns (role-send
        // err, detok err, event_rx end, async cancel) AND all Lane-B
        // closure exits (GS init err, role-send err, stream.next_token err,
        // Ok(None) before content, empty content + finish_reason break,
        // panic in spawn_blocking).
        if let Some(root) = self.root.take() {
            root.close_at_aborted(monotonic_ns());
        }
    }
}

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

    #[test]
    fn open_close_explicit_records_log_line() {
        let ctx = dummy_ctx();
        let root = open_p5h_span_at(
            &ctx,
            None,
            "server_request_recv_to_first_content_sse_write",
            1_000_000_000,
        );
        assert!(root.parent_span_id.is_none());
        assert!(root.parent_span.is_none());
        assert_eq!(
            root.span_name,
            "server_request_recv_to_first_content_sse_write"
        );
        assert!(root.span_id != 0);
        let child = open_p5h_span(&ctx, Some(&root), "http_parse_render_tokenize");
        assert_eq!(child.parent_span_id, Some(root.span_id));
        // Per Codex plan review v1 P1 #1: child.parent_span must be the real
        // parent label, NOT a hard-coded "explicit_parent".
        assert_eq!(
            child.parent_span,
            Some("server_request_recv_to_first_content_sse_write")
        );
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
        let _ = with_p5h_span_from_current_trace::<u32>("deep_span", SpanFields::default, || 42);
    }

    #[test]
    #[should_panic(expected = "handle field tamper detected")]
    fn close_panics_on_tampered_handle_field() {
        // Per Codex plan review v5 P2 + v6 P2: SpanHandle fields are TRULY
        // PRIVATE (not pub(crate)). This test sits in a child module of
        // `core::p5h` and per Rust's child-module visibility rule can still
        // see the parent module's private fields — that's the only way to
        // synthesize the "mutated clone" scenario. Production callers in
        // other modules (openai.rs / scheduler.rs / generate.rs) cannot
        // reach this path: they have neither field access nor a mutating
        // method. The close-side snapshot check serves as defense in depth
        // even for accidental in-module mutation.
        let ctx = dummy_ctx();
        let mut handle = open_p5h_span_at(&ctx, None, "test_root", 0);
        handle.span_name = "tampered_name"; // mutate the clone (only possible from same-module path)
        close_p5h_span(&ctx, handle, 1, SpanFields::default());
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
        let result = with_p5h_span_from_current_trace("deep_span", SpanFields::default, || 7u32);
        assert_eq!(result, 7);
        // Stack returned to just base_parent after body.
        P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
    }

    #[test]
    fn try_with_span_outside_guard_no_ops_returns_body_value() {
        // Per Codex plan review v12 P1 #1: non-OpenAI entry paths leave
        // P5H_CURRENT_TRACE = None; the try_ variant must run body directly
        // (no span open/close, no panic) and return its value.
        // Confirm we are NOT inside a guard before the call.
        P5H_CURRENT_TRACE.with(|c| assert!(c.borrow().is_none()));
        let result =
            try_with_p5h_span_from_current_trace("deep_span_no_trace", SpanFields::default, || {
                13u32
            });
        assert_eq!(result, 13);
        // No span emitted: stack still empty.
        P5H_CURRENT_SPAN_STACK.with(|s| assert!(s.borrow().is_empty()));
    }

    #[test]
    fn try_with_span_inside_guard_emits_span_same_as_strict() {
        // Per Codex plan review v12 P1 #1: when ctx IS active (Lane-A in
        // particular — `routing_path = "scheduler"`), try_ forwards to strict
        // — emits a span exactly like with_p5h_span_from_current_trace.
        // dummy_ctx() returns routing_path = "scheduler" so this covers the
        // Lane-A allow-everything path.
        let ctx = dummy_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        let result = try_with_p5h_span_from_current_trace(
            "deep_span_with_trace",
            SpanFields::default,
            || 11u32,
        );
        assert_eq!(result, 11);
        P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
    }

    fn lane_b_ctx() -> P5hTraceContext {
        // Helper: Lane-B ctx (routing_path = "gs_chunked").
        P5hTraceContext {
            request_id: "test-lane-b".to_string(),
            prompt_tokens: 4096,
            routing_path: "gs_chunked",
        }
    }

    #[test]
    fn try_with_span_lane_b_allowlist_emits_for_top_level_names() {
        // Per Codex plan review v20 P1 #1 + v21 P1: under Lane-B
        // (routing_path = "gs_chunked"), try_ must emit ONLY for the
        // top-level allow-listed names. `gs_chunk_N` is on the allow-list.
        // v21 P1 strengthens this test: prove a span ACTUALLY emitted (not
        // just that stack length returned to base) by capturing the
        // next-span-id BEFORE the call and verifying NEXT_SPAN_ID advanced
        // by exactly 1 — which only happens when `with_p5h_span_from_current_trace`
        // ran the open path. (Stack length restored is necessary but not
        // sufficient: a Lane-B SUPPRESSED call also leaves the stack
        // unchanged, so the prior assertion accepted both correct and
        // incorrect behavior.)
        let ctx = lane_b_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        let id_before = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
        let result =
            try_with_p5h_span_from_current_trace("gs_chunk_N", SpanFields::default, || 13u32);
        let id_after = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(result, 13);
        // Span DID emit → atomic counter advanced by exactly 1.
        assert_eq!(
            id_after, id_before + 1,
            "Lane-B allow-listed span_name MUST emit (advance NEXT_SPAN_ID by 1); got id_before={id_before}, id_after={id_after}",
        );
        // And stack returned to base_parent after body.
        P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
    }

    #[test]
    fn try_with_span_lane_b_other_allowlisted_names_emit_too() {
        // Per Codex v21 P1: the OTHER two allow-listed names must also emit.
        // Without this test, an implementation regression that allow-lists
        // only one name would still pass `try_with_span_lane_b_allowlist_emits_for_top_level_names`.
        let ctx = lane_b_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        for name in ["gs_kv_cache_alloc", "gs_first_token_sample_dispatch"] {
            let id_before = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            let _ = try_with_p5h_span_from_current_trace(name, SpanFields::default, || ());
            let id_after = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                id_after,
                id_before + 1,
                "Lane-B allow-listed span_name `{name}` MUST emit",
            );
        }
    }

    #[test]
    fn try_with_span_lane_b_emits_deep_decoder_names_in_p5h_plus_1() {
        // P5h+1 T2 reverses P5h's top-level-only Lane-B policy. The deep
        // decoder hierarchy (decoder_layer_N, gda_step_*, attention substeps,
        // MoE substeps, slice_last_and_project_lm_head, ...) is now on
        // `LANE_B_ALLOWED_TRY_SPAN_NAMES` and MUST emit on Lane-B so T1's
        // per-substep eval probes can close the gs_chunk_N wrapper-dominance
        // gap. (Original P5h v20 P1 #1 test asserted suppression; v22 P1
        // strengthened it to also check NEXT_SPAN_ID; both assertions are
        // inverted here because the semantic policy itself changed.)
        let ctx = lane_b_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        for name in [
            "decoder_layer_N",
            "gda_step_1a_in_proj_qkvz",
            "slice_last_and_project_lm_head",
            "q_gate_k_v_proj",
            "router_logits_softmax_topk",
            "cache_state_update",
        ] {
            let id_before = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            let result = try_with_p5h_span_from_current_trace(name, SpanFields::default, || 17u32);
            let id_after = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(result, 17);
            assert_eq!(
                id_after,
                id_before + 1,
                "P5h+1 T2 Lane-B deep span_name `{name}` MUST emit (advance NEXT_SPAN_ID by 1); got id_before={id_before}, id_after={id_after}",
            );
            // Stack returned to base_parent after body.
            P5H_CURRENT_SPAN_STACK.with(|s| assert_eq!(s.borrow().len(), 1));
        }
    }

    #[test]
    fn try_with_span_lane_b_suppresses_unknown_names() {
        // Defense in depth (per LANE_B_ALLOWED_TRY_SPAN_NAMES doc comment):
        // even after the P5h+1 T2 allow-list expansion, names NOT on the
        // list must still no-op on Lane-B so a typo or accidental Lane-A
        // name leaking into deep instrumentation does not silently emit.
        let ctx = lane_b_ctx();
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        for name in ["totally_unknown_span", "made_up_op_name"] {
            let before_len = P5H_CURRENT_SPAN_STACK.with(|s| s.borrow().len());
            let id_before = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            let result = try_with_p5h_span_from_current_trace(name, SpanFields::default, || 17u32);
            let id_after = NEXT_SPAN_ID.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(result, 17);
            assert_eq!(
                id_after, id_before,
                "Lane-B unknown span_name `{name}` MUST NOT emit; NEXT_SPAN_ID changed from {id_before} to {id_after}",
            );
            let after_len = P5H_CURRENT_SPAN_STACK.with(|s| s.borrow().len());
            assert_eq!(
                after_len, before_len,
                "Lane-B suppression must NOT touch the span stack"
            );
        }
    }

    #[test]
    #[should_panic(expected = "unknown routing_path")]
    fn try_with_span_unknown_routing_path_panics() {
        // Per Codex plan review v20 P1 #1: routing_path values outside
        // {"scheduler", "gs_chunked"} are emitter bugs and must panic at
        // the helper call site so the bad value never silently emits or
        // silently suppresses.
        let ctx = P5hTraceContext {
            request_id: "bad-routing".to_string(),
            prompt_tokens: 128,
            routing_path: "totally_made_up",
        };
        let root = dummy_span(99);
        let _g = P5hTraceGuard::enter(ctx.clone(), root.clone());
        let _ = try_with_p5h_span_from_current_trace("any_span_name", SpanFields::default, || 0u32);
    }
}
