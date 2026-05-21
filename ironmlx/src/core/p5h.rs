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
    pub seq: Option<u32>,
    pub mode: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct RootSpanHandle {
    ctx: P5hTraceContext,
    span: SpanHandle,
}

impl RootSpanHandle {
    pub(crate) fn ctx(&self) -> &P5hTraceContext {
        &self.ctx
    }
    pub(crate) fn span(&self) -> &SpanHandle {
        &self.span
    }
    pub(crate) fn close_at(self, _end_ns: u64) {
        // T0a.3 fills this in
        unimplemented!("filled in T0a.3");
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
