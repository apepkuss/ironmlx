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
    pub(crate) fn span_id(&self) -> u64 { self.span_id }
    pub(crate) fn span_name(&self) -> &'static str { self.span_name }
    pub(crate) fn parent_span_id(&self) -> Option<u64> { self.parent_span_id }
    pub(crate) fn parent_span(&self) -> Option<&'static str> { self.parent_span }
    pub(crate) fn start_ns(&self) -> u64 { self.start_ns }
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
