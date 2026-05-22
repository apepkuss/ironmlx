//! P5h T3 — MoE 8-substep emission sweep (Lane A) + Lane B opportunistic
//! top-level smoke.
//!
//! Verifies the T3.2 instrumentation (commit `67e131a` —
//! `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs::SparseMoeBlock::forward_on`)
//! emits one `[p5h-profile]` record per substep per MoE layer per request,
//! and reports the trimmed_median(inclusive_us) per (PP, span_name) cell.
//!
//! ## Span set (9 spans total)
//!
//!   * `mlp_path` — wrapper opened by
//!     `ironmlx/src/models/qwen3_5_moe/decoder_layer.rs::DecoderLayerMoe::forward_on`
//!     (T0a.11 step 1, line 249). Fires on EVERY decoder layer (all 40); the
//!     MoE substeps below nest under this wrapper.
//!
//!   * MoE 8 substeps (T3.2, per spec § 3 T3). Fire on every decoder layer
//!     under `serve_via_scheduler_stream` (Lane A). Lane B
//!     (`serve_via_gs_stream`) suppresses deep substep attribution per spec
//!     § 3 T0a line 963 (deferred to P5h+1).
//!     1. `router_logits_softmax_topk` — router logits + softmax + topk
//!     2. `routing_sort_pack` — token-to-expert routing / sort / pack (sorted
//!        path only at `bs_k >= 512`)
//!     3. `gather_qmm_gate_up` — fused gather + quantized gate_up matmul
//!     4. `swiglu_activation` — SwiGLU(gate, up)
//!     5. `gather_qmm_down` — fused gather + quantized down_proj matmul
//!     6. `routing_unsort_weighted_reduce` — unsort + weight + reduce
//!     7. `shared_expert` — shared MoE expert branch
//!     8. `moe_output_sum` — routed + shared expert sum
//!
//! ## Routing-path note for `routing_sort_pack`
//!
//! 35B model sorted-routing threshold is `bs_k >= 512` where
//! `bs_k = batch * seq_len * k` (k=8 top-k experts). Inside a single MoE
//! forward call:
//!   * Prefill PP >= 64 → `bs_k = 1 * PP * 8 >= 512` → sorted path → real
//!     `inclusive_us > 0`.
//!   * Decode (T_q=1) → `bs_k = 8 < 512` → non-sorted path → zero-op
//!     closure → `inclusive_us ≈ 0`.
//!
//! With `--max-tokens 1` the request produces 0 or 1 decode forwards; the
//! records are dominated by prefill chunks (sorted path with real timing).
//! Median over the prefill-dominated records is > 0 → PASS criterion holds.
//!
//! ## Lane-aware cell schema (per Codex T2 review feedback)
//!
//! `T3_PP_LIST = [128, 512, 1024, 4096]` per Boss decision (3 Lane A + 1
//! Lane B opportunistic). Lane derivation is runtime-by-`routing_path` per
//! T1 lesson (memory `project_p5h_t1_findings`), NOT nominal-PP-based —
//! the Qwen3 ChatML wrapper adds ~30 tokens, so `--prompt-len N` reaches
//! `prompt_len ≈ N+30` at the routing predicate.
//!
//! Each cell carries:
//!   * `lane` — span classification. All 9 T3 spans are `"lane_a"`-only
//!     (none fire purely on Lane B, none on both).
//!   * `actual_lane_observed` — what the PP actually routed (per-request
//!     `routing_path` aggregation): `"lane_a"` | `"lane_b"` | `"mixed"`.
//!   * `expected_to_emit` — `true` iff `actual_lane_observed` intersects
//!     the span's lane. Lane B PPs (PP=4096 at production-default
//!     `prefill_chunk_size=2048`) get `expected_to_emit = false` so the
//!     verdict does NOT penalize the intentional Lane B suppression.
//!
//! PASS criterion: every cell with `expected_to_emit == true` has
//! `record_count > 0` AND `median_inclusive_us > 0`. Lane B PP cells are
//! reported in the JSON (with their observed `record_count`) for diagnostics
//! but are exempt from PASS.
//!
//! Per Codex T2 review feedback + spec § 3 T0a line 963: Lane-B deep substep
//! attribution is suppressed/rejected at this milestone and deferred to
//! P5h+1. T3 keeps the Lane B PP in the sweep purely as an opportunistic
//! top-level smoke (it exercises the routing predicate + Lane B handler) and
//! to keep the JSON record-count diagnostics visible if Lane B unexpectedly
//! starts emitting some/all MoE substeps.
//!
//! Server gate: `--features p5h-profile` (verifying the [p5h-profile]
//! emission schema itself, not ProfileMode ablation). 5min preheat at entry
//! per T0b binding (memory `project_p5h_t0b_findings`). Per-PP spawn-kill
//! per T0a/T0b/T1 precedent — each PP starts from server cold-restart.
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5h-profile \
//!     --test p5h_t3_moe_sweep -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

mod p5h_common;
use p5h_common::*;

use std::collections::{BTreeMap, BTreeSet};

/// PP set per Boss decision: 3 Lane A + 1 Lane B opportunistic. Lane
/// derivation is runtime-by-`routing_path` per T1 lesson (NOT nominal-PP).
///
/// At production-default `prefill_chunk_size=2048`:
///   * `1024 + chat_template_overhead ~30 ≈ 1054 < 2048` → 3 safely-Lane-A PPs.
///   * `4096 + ~30 ≈ 4126 > 2048` → 1 safely-Lane-B PP.
const T3_PP_LIST: [i32; 4] = [128, 512, 1024, 4096];

/// 8 MoE substeps per spec § 3 T3 + T3.2 instrumentation (commit `67e131a`,
/// `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs::SparseMoeBlock::forward_on`).
/// Emitted only on Lane A (under `serve_via_scheduler_stream`). Lane B path
/// (`serve_via_gs_stream`) suppresses deep substep attribution per spec § 3
/// T0a line 963 — deferred to P5h+1.
const T3_MOE_SUBSTEPS: [&str; 8] = [
    "router_logits_softmax_topk",
    "routing_sort_pack",
    "gather_qmm_gate_up",
    "swiglu_activation",
    "gather_qmm_down",
    "routing_unsort_weighted_reduce",
    "shared_expert",
    "moe_output_sum",
];

/// Wrapper span opened by
/// `ironmlx/src/models/qwen3_5_moe/decoder_layer.rs::DecoderLayerMoe::forward_on`
/// (T0a.11 step 1, line 249) for every decoder layer. The 8 MoE substeps are
/// nested inside this wrapper for every decoder layer's MoE forward path. T3
/// treats it as the 9th expected span for Lane A coverage (paired with T2's
/// `attention_path` wrapper which sits in the same layer).
const T3_MLP_PATH_WRAPPER: &str = "mlp_path";

/// Combined T3 span set (wrapper + 8 substeps) for cell generation. Order is
/// deterministic: wrapper first, then substeps in spec order.
fn t3_all_span_names() -> Vec<&'static str> {
    let mut v = vec![T3_MLP_PATH_WRAPPER];
    v.extend_from_slice(&T3_MOE_SUBSTEPS);
    v
}

const T3_OUTPUT_PATH: &str = "/tmp/p5h-t3.json";

/// Lane identifiers as they appear in `[p5h-profile]` records' `routing_path`
/// field (per `ironmlx/src/core/p5h.rs:23`). Used to derive the per-request
/// `actual_lane` from server-emitted records.
const ROUTING_PATH_LANE_A: &str = "scheduler";
const ROUTING_PATH_LANE_B: &str = "gs_chunked";

const PREHEAT_PROTOCOL_DESC: &str =
    "5min preheat per T0b binding: T3_PP_LIST × PREHEAT_RUNS=3 throwaway Phase A \
     iron-bench runs with spawn-kill per PP (using T3_PP_LIST=[128,512,1024,4096]); \
     results discarded; runs BEFORE first measurement spawn to drive GPU into \
     thermal saturation";

const INITIAL_COOL_PROTOCOL_DESC: &str =
    "INTER_PP_COOLDOWN=3s; per-PP spawn-kill so each PP starts from server cold-restart \
     (T0a/T0b/T1/T2 precedent — no inter-phase cool gate, preheat handles thermal saturation)";

/// Parsed `[p5h-profile]` record. Carries the lane-derivation inputs
/// (`request_id` + `routing_path` + `prompt_tokens`) in addition to the span
/// identity / timing fields used by the verdict. T3 also extracts
/// `parent_span` for diagnostic visibility (MoE substeps should report
/// `parent_span = "mlp_path"`); the value is recorded but NOT enforced in
/// the verdict (deferred to T5 structural tree check).
#[derive(Debug, serde::Serialize, Clone)]
struct P5hProfileRecord {
    request_id: String,
    /// Server-emitted lane: `"scheduler"` (Lane A) | `"gs_chunked"` (Lane B).
    routing_path: String,
    /// Server-side tokenized prompt length AFTER chat-template render.
    prompt_tokens: i64,
    /// Parent span label from the emitter (for T3 substeps this should be
    /// `"mlp_path"`). Recorded for downstream diagnostic / T5 structural
    /// check; not enforced in the T3 verdict.
    parent_span: String,
    span_name: String,
    start_ns: u64,
    end_ns: u64,
    /// Either `"tree"` or `"diagnostic"` per `[p5h-profile]` `span_kind` field.
    span_kind: String,
}

/// Parse `[p5h-profile]` records from server stderr. Tolerates the tracing
/// default formatter prefix (`<ts>  INFO <module>: <message>`) by using
/// `line.find("[p5h-profile]")` to locate the structured field portion.
///
/// Field set (per `ironmlx/src/core/p5h.rs:313` emission contract):
///   request_id routing_path prompt_tokens seq layer_idx span_id parent_span_id
///   span_name parent_span start_ns end_ns mode span_kind
///
/// T3 consumes `request_id`, `routing_path`, `prompt_tokens`, `parent_span`,
/// `span_name`, `start_ns`, `end_ns`, `span_kind`. The other 5 fields
/// (seq, layer_idx, span_id, parent_span_id, mode) are intentionally
/// ignored — split_once('=') skips them.
///
/// Panics on a tagged line missing any of the 8 required fields or on a
/// malformed numeric value (failure-fast surfaces emission-side bugs).
fn parse_p5h_records(stderr_bytes: &[u8]) -> Vec<P5hProfileRecord> {
    let s = String::from_utf8_lossy(stderr_bytes);
    let mut records = Vec::new();
    for line in s.lines() {
        let Some(start) = line.find("[p5h-profile]") else {
            continue;
        };
        let tail = &line[start + "[p5h-profile]".len()..];
        let mut request_id: Option<String> = None;
        let mut routing_path: Option<String> = None;
        let mut prompt_tokens: Option<i64> = None;
        let mut parent_span: Option<String> = None;
        let mut span_name: Option<String> = None;
        let mut start_ns: Option<u64> = None;
        let mut end_ns: Option<u64> = None;
        let mut span_kind: Option<String> = None;
        for tok in tail.split_whitespace() {
            let Some((k, v)) = tok.split_once('=') else {
                continue;
            };
            match k {
                "request_id" => request_id = Some(v.to_string()),
                "routing_path" => routing_path = Some(v.to_string()),
                "prompt_tokens" => {
                    prompt_tokens = Some(v.parse::<i64>().unwrap_or_else(|e| {
                        panic!("[p5h-profile] bad prompt_tokens={v}: {e}; line={line}")
                    }))
                }
                "parent_span" => parent_span = Some(v.to_string()),
                "span_name" => span_name = Some(v.to_string()),
                "start_ns" => {
                    start_ns = Some(v.parse::<u64>().unwrap_or_else(|e| {
                        panic!("[p5h-profile] bad start_ns={v}: {e}; line={line}")
                    }))
                }
                "end_ns" => {
                    end_ns = Some(v.parse::<u64>().unwrap_or_else(|e| {
                        panic!("[p5h-profile] bad end_ns={v}: {e}; line={line}")
                    }))
                }
                "span_kind" => span_kind = Some(v.to_string()),
                _ => {}
            }
        }
        let request_id =
            request_id.unwrap_or_else(|| panic!("[p5h-profile] line missing request_id: {line}"));
        let routing_path = routing_path
            .unwrap_or_else(|| panic!("[p5h-profile] line missing routing_path: {line}"));
        let prompt_tokens = prompt_tokens
            .unwrap_or_else(|| panic!("[p5h-profile] line missing prompt_tokens: {line}"));
        let parent_span =
            parent_span.unwrap_or_else(|| panic!("[p5h-profile] line missing parent_span: {line}"));
        let span_name =
            span_name.unwrap_or_else(|| panic!("[p5h-profile] line missing span_name: {line}"));
        let start_ns =
            start_ns.unwrap_or_else(|| panic!("[p5h-profile] line missing start_ns: {line}"));
        let end_ns = end_ns.unwrap_or_else(|| panic!("[p5h-profile] line missing end_ns: {line}"));
        let span_kind =
            span_kind.unwrap_or_else(|| panic!("[p5h-profile] line missing span_kind: {line}"));
        records.push(P5hProfileRecord {
            request_id,
            routing_path,
            prompt_tokens,
            parent_span,
            span_name,
            start_ns,
            end_ns,
            span_kind,
        });
    }
    records
}

#[derive(Debug, serde::Serialize)]
struct T3Cell {
    pp: i32,
    span_name: String,
    /// Lane classification for this SPAN. Always `"lane_a"` for T3: all 9
    /// spans (mlp_path wrapper + 8 MoE substeps) are Lane-A-only by design
    /// — Lane B suppresses deep substep attribution per spec § 3 T0a line
    /// 963.
    lane: &'static str,
    /// What the server actually routed for requests at this PP, derived
    /// per request from `routing_path` aggregation. `"lane_a"` |
    /// `"lane_b"` | `"mixed"`. `None` when the PP produced 0 requests.
    actual_lane_observed: Option<&'static str>,
    /// True iff this cell's span is expected to emit at this PP given the
    /// PP's `actual_lane_observed`. For T3 (all spans Lane-A-only):
    /// `expected_to_emit = matches!(actual_lane_observed, Some("lane_a") |
    /// Some("mixed"))`. Lane B PPs get `expected_to_emit = false` →
    /// exempt from PASS criterion per spec § 3 T0a Lane B suppression
    /// carve-out.
    expected_to_emit: bool,
    /// Number of `[p5h-profile]` records matching `(span_name, pp)` with
    /// `end_ns > start_ns`.
    record_count: usize,
    /// Trimmed median of `(end_ns - start_ns) / 1000` across all matching
    /// records (None when record_count == 0).
    median_inclusive_us: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct T3Verdict {
    verdict: String,
    rationale: String,
    cells: Vec<T3Cell>,
    /// (PP, lane, span_name) tuples whose cell has `expected_to_emit=true`
    /// but produced 0 records or only non-positive inclusive_us samples.
    /// Empty in the happy path. Lane B PP cells are NEVER added here
    /// (their `expected_to_emit=false` excludes them by construction).
    missing_or_invalid: Vec<String>,
    preheat_protocol: String,
    initial_cool_protocol: String,
}

/// Per-request lane classification derived from `routing_path` aggregation.
/// `"lane_a"` | `"lane_b"`. A single `request_id` carrying mixed
/// `routing_path` values panics with diagnostic (catches emitter regression).
fn request_actual_lane(request_id: &str, recs: &[&P5hProfileRecord]) -> &'static str {
    assert!(
        !recs.is_empty(),
        "request_id={request_id} produced 0 [p5h-profile] records — \
         impossible if parser succeeded; emitter regression"
    );
    let mut paths: BTreeSet<&str> = BTreeSet::new();
    for r in recs {
        paths.insert(r.routing_path.as_str());
    }
    if paths.len() != 1 {
        panic!(
            "request_id={request_id} carries mixed routing_path values {:?} — \
             a single request cannot route both lanes; emitter regression",
            paths
        );
    }
    let path = *paths.iter().next().expect("BTreeSet len==1");
    match path {
        ROUTING_PATH_LANE_A => "lane_a",
        ROUTING_PATH_LANE_B => "lane_b",
        other => panic!(
            "request_id={request_id} carries unknown routing_path={other:?} — \
             expected {ROUTING_PATH_LANE_A:?} or {ROUTING_PATH_LANE_B:?}"
        ),
    }
}

/// Group `recs` by `request_id` preserving insertion order semantics via
/// `BTreeMap`. Each value is the per-request slice of refs to records.
fn group_by_request_id(recs: &[P5hProfileRecord]) -> BTreeMap<String, Vec<&P5hProfileRecord>> {
    let mut by_req: BTreeMap<String, Vec<&P5hProfileRecord>> = BTreeMap::new();
    for r in recs {
        by_req.entry(r.request_id.clone()).or_default().push(r);
    }
    by_req
}

/// Run iron-bench once per PP in `T3_PP_LIST`, capture server stderr, parse
/// `[p5h-profile]` records, and group them per (PP).
///
/// Per-PP spawn-kill matches T0a/T0b/T1/T2 precedent: every PP starts from a
/// fresh server. Failure paths go through `shutdown_and_join` to avoid
/// leaking Child processes.
fn run_t3_collect_records(
    model_dir: &str,
    port: u16,
) -> anyhow::Result<BTreeMap<i32, Vec<P5hProfileRecord>>> {
    let mut per_pp: BTreeMap<i32, Vec<P5hProfileRecord>> = BTreeMap::new();
    for &pp in &T3_PP_LIST {
        assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;
        // T3 does NOT set IRONMLX_P5G_PROFILE_MODE — verifying the
        // [p5h-profile] schema, not a ProfileMode ablation.
        // No --prefill-chunk-size override: production default 2048
        // exercises the real routing predicate (T1 lesson).
        let mut server = spawn_server(None, model_dir, port);
        let (stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

        if let Err(e) = wait_for_ready(port, 300) {
            shutdown_and_join(server, drainer);
            anyhow::bail!("PP={pp} T3: server not ready: {e}");
        }
        match server.try_wait() {
            Ok(Some(status)) => {
                let _ = drainer.join();
                anyhow::bail!("PP={pp} T3: ironmlx serve exited before bench: {status}");
            }
            Ok(None) => {}
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T3: try_wait failed: {e}");
            }
        }

        let out = match iron_bench_run(port, model_dir, pp) {
            Ok(o) => o,
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T3: iron-bench spawn failed: {e}");
            }
        };

        // Shutdown FIRST so drainer EOF + join completes before we drain.
        let _ = server.kill();
        let _ = server.wait();
        let _ = drainer.join();

        if !out.status.success() {
            anyhow::bail!(
                "PP={pp} T3: iron-bench non-success: stdout={}, stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let captured = drain_stderr_into_buf(&stderr_buf);
        let records = parse_p5h_records(&captured);
        eprintln!(
            "[p5h-t3] PP={pp}: captured {} [p5h-profile] records ({} stderr bytes)",
            records.len(),
            captured.len()
        );
        per_pp.insert(pp, records);
        std::thread::sleep(INTER_PP_COOLDOWN);
    }
    Ok(per_pp)
}

/// Aggregate per-PP records into (PP, span_name) cells with lane-aware
/// `expected_to_emit` gating.
///
/// For each PP × each of the 9 T3 spans:
///   * Compute `actual_lane_observed` from per-request `routing_path`
///     aggregation (`"lane_a"` | `"lane_b"` | `"mixed"` | None on empty).
///   * `expected_to_emit = matches!(actual_lane_observed, Some("lane_a") |
///     Some("mixed"))`.
///   * Count matching records with `end_ns > start_ns` + median.
///   * Only `expected_to_emit=true` cells with 0 valid records / median
///     non-positive contribute to `missing_or_invalid`.
///
/// PASS iff `missing_or_invalid.is_empty()`. Lane B PP cells are reported in
/// the JSON (with `record_count` for diagnostic visibility) but exempted
/// from the verdict per spec § 3 T0a line 963 — Lane-B deep substep
/// attribution is suppressed/rejected at this milestone and deferred to
/// P5h+1.
fn compute_t3_verdict(per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>) -> T3Verdict {
    let mut cells: Vec<T3Cell> = Vec::new();
    let mut missing_or_invalid: Vec<String> = Vec::new();
    let mut lane_a_pps: BTreeSet<i32> = BTreeSet::new();
    let mut lane_b_pps: BTreeSet<i32> = BTreeSet::new();
    let mut mixed_pps: BTreeSet<i32> = BTreeSet::new();

    for (&pp, recs) in per_pp {
        let by_req = group_by_request_id(recs);
        let actual_lane_observed: Option<&'static str> = if by_req.is_empty() {
            None
        } else {
            let mut pp_request_lanes: BTreeSet<&'static str> = BTreeSet::new();
            for (req_id, req_recs) in &by_req {
                pp_request_lanes.insert(request_actual_lane(req_id, req_recs));
            }
            let pp_has_lane_a = pp_request_lanes.contains("lane_a");
            let pp_has_lane_b = pp_request_lanes.contains("lane_b");
            match (pp_has_lane_a, pp_has_lane_b) {
                (true, false) => Some("lane_a"),
                (false, true) => Some("lane_b"),
                (true, true) => Some("mixed"),
                (false, false) => None, // unreachable when by_req non-empty
            }
        };

        match actual_lane_observed {
            Some("lane_a") => {
                lane_a_pps.insert(pp);
            }
            Some("lane_b") => {
                lane_b_pps.insert(pp);
            }
            Some("mixed") => {
                mixed_pps.insert(pp);
                lane_a_pps.insert(pp);
                lane_b_pps.insert(pp);
            }
            _ => {}
        }

        // expected_to_emit policy: all 9 T3 spans are Lane-A-only by design.
        // Lane B PP exempt from PASS criterion (spec § 3 T0a Lane B suppression
        // carve-out — deferred to P5h+1). Mixed PPs count as Lane A coverage
        // (≥1 Lane A request must produce the substeps).
        let expected_to_emit = matches!(actual_lane_observed, Some("lane_a") | Some("mixed"));

        for span in t3_all_span_names() {
            let filtered: Vec<&P5hProfileRecord> = recs
                .iter()
                .filter(|r| r.span_name == span && r.end_ns > r.start_ns)
                .collect();
            let count = filtered.len();
            let median = if count == 0 {
                None
            } else {
                let us: Vec<f64> = filtered
                    .iter()
                    .map(|r| (r.end_ns - r.start_ns) as f64 / 1000.0)
                    .collect();
                trimmed_median(us)
            };
            let valid = count > 0 && matches!(median, Some(m) if m > 0.0);
            if expected_to_emit && !valid {
                missing_or_invalid.push(format!(
                    "PP={pp} lane=lane_a span={span} actual_lane_observed={actual_lane_observed:?}: \
                     0 valid records (record_count={count}, median_inclusive_us={median:?})"
                ));
            }
            cells.push(T3Cell {
                pp,
                span_name: span.to_string(),
                lane: "lane_a",
                actual_lane_observed,
                expected_to_emit,
                record_count: count,
                median_inclusive_us: median,
            });
        }
    }

    let lane_a_pps_vec: Vec<i32> = lane_a_pps.iter().copied().collect();
    let lane_b_pps_vec: Vec<i32> = lane_b_pps.iter().copied().collect();
    let total_records: usize = per_pp.values().map(|v| v.len()).sum();
    let expected_cell_count = cells.iter().filter(|c| c.expected_to_emit).count();

    let verdict = if total_records == 0 {
        "no_data"
    } else if missing_or_invalid.is_empty() {
        "pass"
    } else {
        "missing_spans"
    };

    let rationale = if total_records == 0 {
        format!(
            "T3 no_data: total_records=0 across all PPs {:?}. Verify --features p5h-profile was \
             set and that decoder_layer.rs::DecoderLayerMoe::forward_on opens the mlp_path \
             wrapper.",
            T3_PP_LIST
        )
    } else if missing_or_invalid.is_empty() {
        format!(
            "T3 pass: every Lane-A-expected (PP, span_name) cell has >=1 valid record. \
             total_records={total_records}, Lane-A PPs (observed)={lane_a_pps_vec:?}, \
             Lane-B PPs (observed)={lane_b_pps_vec:?} (exempted from PASS per spec § 3 T0a); \
             expected cells={expected_cell_count} (all emit). Total cells={}.",
            cells.len(),
        )
    } else {
        format!(
            "T3 missing_spans: {} cells failed PASS criterion (>=1 record with end_ns>start_ns). \
             total_records={total_records}, Lane-A PPs (observed)={lane_a_pps_vec:?}, \
             Lane-B PPs (observed)={lane_b_pps_vec:?} (exempted from PASS per spec § 3 T0a); \
             expected cells={expected_cell_count}. Failed cells: {}",
            missing_or_invalid.len(),
            missing_or_invalid.join("; "),
        )
    };

    T3Verdict {
        verdict: verdict.to_string(),
        rationale,
        cells,
        missing_or_invalid,
        preheat_protocol: PREHEAT_PROTOCOL_DESC.to_string(),
        initial_cool_protocol: INITIAL_COOL_PROTOCOL_DESC.to_string(),
    }
}

/// Per-PP, per-request observed `prompt_tokens` (the server-side tokenized
/// count post chat-template render). Mirrors T1/T2 — grounds the ChatML
/// overhead lesson in the JSON payload for future debugging.
fn prompt_tokens_observed_per_pp(
    per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>,
) -> BTreeMap<i32, Vec<i64>> {
    let mut out: BTreeMap<i32, Vec<i64>> = BTreeMap::new();
    for (&pp, recs) in per_pp {
        let by_req = group_by_request_id(recs);
        let mut tokens: Vec<i64> = by_req
            .values()
            .map(|req_recs| {
                req_recs
                    .first()
                    .map(|r| r.prompt_tokens)
                    .expect("group_by_request_id never produces empty Vec")
            })
            .collect();
        tokens.sort();
        out.insert(pp, tokens);
    }
    out
}

/// Per-PP, per-request observed `actual_lane` derived from `routing_path`.
/// One entry per request at the PP (so RUNS=7 produces 7 entries). All-same
/// is the typical case at production default; divergent values surface
/// directly in the JSON.
fn actual_lane_per_pp(
    per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>,
) -> BTreeMap<i32, Vec<&'static str>> {
    let mut out: BTreeMap<i32, Vec<&'static str>> = BTreeMap::new();
    for (&pp, recs) in per_pp {
        let by_req = group_by_request_id(recs);
        let mut lanes: Vec<&'static str> = by_req
            .iter()
            .map(|(req_id, req_recs)| request_actual_lane(req_id, req_recs))
            .collect();
        lanes.sort();
        out.insert(pp, lanes);
    }
    out
}

#[test]
#[ignore = "p5h-t3 — MoE 8-substep emission sweep (Lane A) + Lane B opportunistic top-level smoke (~10-15min GPU + 5min preheat)"]
fn t3_moe_sweep() -> anyhow::Result<()> {
    let model_dir = snapshot_dir();
    let _mlx_dir = std::env::var("MLX_DIR")
        .expect("set MLX_DIR env var pointing to MLX install prefix (e.g. $HOME/.local/mlx)");
    eprintln!("[p5h-t3] starting; model={model_dir}");

    eprintln!("[p5h-t3] preheat phase");
    preheat_to_saturation(&model_dir, PROFILE_PORT, &T3_PP_LIST)?;

    eprintln!("[p5h-t3] measurement phase (PP ∈ {T3_PP_LIST:?})");
    let per_pp = run_t3_collect_records(&model_dir, PROFILE_PORT)?;

    let verdict = compute_t3_verdict(&per_pp);
    eprintln!("[p5h-t3] {}", verdict.rationale);

    // Per-PP record counts (across all span_names) for diagnostic visibility.
    let per_pp_counts: BTreeMap<i32, usize> = per_pp.iter().map(|(k, v)| (*k, v.len())).collect();
    let total_records: usize = per_pp.values().map(|v| v.len()).sum();
    let prompt_tokens_observed = prompt_tokens_observed_per_pp(&per_pp);
    let actual_lane_per_pp_obs = actual_lane_per_pp(&per_pp);

    let out_json = serde_json::json!({
        "pp_list": T3_PP_LIST,
        "span_names": t3_all_span_names(),
        "runs": RUNS,
        "warmup": WARMUP,
        "inter_pp_cooldown_secs": INTER_PP_COOLDOWN.as_secs(),
        "per_pp_record_counts": per_pp_counts,
        "total_records": total_records,
        "prompt_tokens_observed": prompt_tokens_observed,
        "actual_lane_per_pp": actual_lane_per_pp_obs,
        "cells": verdict.cells,
        "verdict": verdict.verdict,
        "rationale": verdict.rationale,
        "missing_or_invalid": verdict.missing_or_invalid,
        "preheat_protocol": verdict.preheat_protocol,
        "initial_cool_protocol": verdict.initial_cool_protocol,
    });
    let json_str = serde_json::to_string_pretty(&out_json)?;
    // Dump full payload to stderr BEFORE the file write so that if /tmp write
    // fails (disk full / permissions / fs read-only) the data is still
    // recoverable from --nocapture scrollback after the long GPU sweep.
    eprintln!("[p5h-t3] JSON payload (preserved in case file-write fails):\n{json_str}");
    std::fs::write(T3_OUTPUT_PATH, &json_str)?;
    eprintln!(
        "[p5h-t3] wrote {} bytes to {T3_OUTPUT_PATH}",
        json_str.len()
    );

    // Per T0b harness convention: return Ok regardless of verdict — verdict
    // string in JSON is the consumed signal.
    Ok(())
}

// ===== Unit tests: parser + lane-derivation + verdict shape =====
//
// All unit tests run without GPU under `cargo test --features p5h-profile`.

#[cfg(test)]
mod parser_tests {
    use super::*;

    /// Build a single-request stderr blob with the given
    /// (span_name, parent_span, span_kind, start_ns, end_ns) tuples, all
    /// sharing the supplied request_id + routing_path + prompt_tokens.
    fn build_stderr(
        request_id: &str,
        routing_path: &str,
        prompt_tokens: i64,
        spans: &[(&str, &str, &str, u64, u64)],
    ) -> Vec<u8> {
        let mut out = String::new();
        for (i, (span_name, parent_span, span_kind, start, end)) in spans.iter().enumerate() {
            out.push_str(&format!(
                "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
                 [p5h-profile] request_id={request_id} routing_path={routing_path} \
                 prompt_tokens={prompt_tokens} seq=0 layer_idx=-1 span_id={span_id} \
                 parent_span_id=null span_name={span_name} parent_span={parent_span} \
                 start_ns={start} end_ns={end} mode=off span_kind={span_kind}\n",
                span_id = 1000 + i,
            ));
        }
        out.into_bytes()
    }

    /// Helper: build a full Lane-A 9-span emission (mlp_path wrapper + 8 MoE
    /// substeps) for one request_id at one PP. Each substep emits once; the
    /// wrapper emits once (in real execution the wrapper emits 40× per
    /// request but the verdict is `count > 0`, so 1× is enough for the unit
    /// test).
    fn build_full_lane_a_emission(request_id: &str, prompt_tokens: i64) -> Vec<u8> {
        build_stderr(
            request_id,
            ROUTING_PATH_LANE_A,
            prompt_tokens,
            &[
                ("mlp_path", "null", "tree", 100, 300),
                ("router_logits_softmax_topk", "mlp_path", "tree", 110, 120),
                ("routing_sort_pack", "mlp_path", "tree", 121, 130),
                ("gather_qmm_gate_up", "mlp_path", "tree", 131, 160),
                ("swiglu_activation", "mlp_path", "tree", 161, 170),
                ("gather_qmm_down", "mlp_path", "tree", 171, 200),
                (
                    "routing_unsort_weighted_reduce",
                    "mlp_path",
                    "tree",
                    201,
                    210,
                ),
                ("shared_expert", "mlp_path", "tree", 211, 240),
                ("moe_output_sum", "mlp_path", "tree", 241, 250),
            ],
        )
    }

    #[test]
    fn p5h_record_parser_extracts_all_fields_for_moe_substep() {
        // Verify all 8 fields (including parent_span) parse correctly for a
        // T3 substep emission. parent_span="mlp_path" exercises the
        // T3-specific field beyond T1's parse_p5h_records.
        let stderr = build_stderr(
            "req-1",
            "scheduler",
            128,
            &[
                ("mlp_path", "null", "tree", 100, 300),
                ("router_logits_softmax_topk", "mlp_path", "tree", 110, 120),
                ("routing_sort_pack", "mlp_path", "tree", 121, 130),
                ("gather_qmm_gate_up", "mlp_path", "tree", 131, 160),
                ("swiglu_activation", "mlp_path", "tree", 161, 170),
                ("gather_qmm_down", "mlp_path", "tree", 171, 200),
                (
                    "routing_unsort_weighted_reduce",
                    "mlp_path",
                    "tree",
                    201,
                    210,
                ),
                ("shared_expert", "mlp_path", "tree", 211, 240),
                ("moe_output_sum", "mlp_path", "tree", 241, 250),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        assert_eq!(recs.len(), 9, "expected 9 P5h records, got {}", recs.len());

        for r in &recs {
            assert_eq!(r.request_id, "req-1");
            assert_eq!(r.routing_path, "scheduler");
            assert_eq!(r.prompt_tokens, 128);
        }

        assert_eq!(recs[0].span_name, "mlp_path");
        assert_eq!(recs[0].parent_span, "null");
        assert_eq!(recs[0].start_ns, 100);
        assert_eq!(recs[0].end_ns, 300);
        assert_eq!(recs[0].span_kind, "tree");

        assert_eq!(recs[1].span_name, "router_logits_softmax_topk");
        assert_eq!(recs[1].parent_span, "mlp_path");

        assert_eq!(recs[2].span_name, "routing_sort_pack");
        assert_eq!(recs[2].parent_span, "mlp_path");

        assert_eq!(recs[5].span_name, "gather_qmm_down");
        assert_eq!(recs[5].parent_span, "mlp_path");
        assert_eq!(recs[5].start_ns, 171);
        assert_eq!(recs[5].end_ns, 200);

        assert_eq!(recs[8].span_name, "moe_output_sum");
        assert_eq!(recs[8].parent_span, "mlp_path");
    }

    #[test]
    fn p5h_record_parser_skips_non_p5h_lines() {
        // Lines without the [p5h-profile] tag must be silently ignored.
        let mut stderr = String::new();
        stderr.push_str(
            "2026-05-22T12:34:56.789012Z  INFO some::other::module: unrelated INFO line\n",
        );
        stderr.push_str("2026-05-22T12:34:56.890123Z DEBUG ironmlx::other: [other-tag] foo=bar\n");
        stderr.push_str(
            "2026-05-22T12:34:56.901234Z  INFO ironmlx::core::p5h: [p5h-profile] \
             request_id=r routing_path=scheduler prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 \
             parent_span_id=null span_name=gather_qmm_gate_up parent_span=mlp_path \
             start_ns=10 end_ns=20 mode=off span_kind=tree\n",
        );
        let recs = parse_p5h_records(stderr.as_bytes());
        assert_eq!(recs.len(), 1, "only the [p5h-profile] line should match");
        assert_eq!(recs[0].request_id, "r");
        assert_eq!(recs[0].routing_path, "scheduler");
        assert_eq!(recs[0].prompt_tokens, 2);
        assert_eq!(recs[0].parent_span, "mlp_path");
        assert_eq!(recs[0].span_name, "gather_qmm_gate_up");
        assert_eq!(recs[0].start_ns, 10);
        assert_eq!(recs[0].end_ns, 20);
    }

    #[test]
    #[should_panic(expected = "line missing routing_path")]
    fn p5h_record_parser_fails_on_missing_routing_path() {
        // Strip routing_path from an otherwise valid line — parser must panic.
        let line = b"2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
            [p5h-profile] request_id=r prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 \
            parent_span_id=null span_name=gather_qmm_gate_up parent_span=mlp_path \
            start_ns=10 end_ns=20 mode=off span_kind=tree\n";
        let _ = parse_p5h_records(line);
    }

    #[test]
    fn lane_derivation_pure_lane_a() {
        // Single request routed Lane A with full 9-span emission → PASS.
        // All cells have actual_lane_observed=Some("lane_a") and
        // expected_to_emit=true.
        let stderr = build_full_lane_a_emission("req-A", 128);
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t3_verdict(&per_pp);
        assert_eq!(verdict.verdict, "pass", "rationale={}", verdict.rationale);

        // 1 PP × 9 spans = 9 cells.
        assert_eq!(verdict.cells.len(), 9);
        for c in &verdict.cells {
            assert_eq!(c.lane, "lane_a");
            assert_eq!(c.actual_lane_observed, Some("lane_a"));
            assert!(
                c.expected_to_emit,
                "Lane A span {} should be expected",
                c.span_name
            );
            assert!(
                c.record_count > 0,
                "span {} expected >=1 record",
                c.span_name
            );
            assert!(
                matches!(c.median_inclusive_us, Some(m) if m > 0.0),
                "span {} expected positive median",
                c.span_name
            );
        }
        assert!(verdict.missing_or_invalid.is_empty());
    }

    #[test]
    fn lane_derivation_pure_lane_b_exempts_t3_spans() {
        // Lane B PP with NO MoE substeps emitted (the realistic Lane B
        // reality — deep substep attribution suppressed per spec § 3 T0a).
        // Verdict MUST pass: all 9 cells have expected_to_emit=false, so
        // missing records do not fail PASS. Build a Lane B request with no
        // MoE spans at all (mimics serve_via_gs_stream which never invokes
        // SparseMoeBlock::forward_on through the wrapped path).
        let stderr = build_stderr(
            "req-B",
            ROUTING_PATH_LANE_B,
            4096,
            &[
                // A non-T3 span is fine — we just need >=1 record so by_req
                // is non-empty and actual_lane_observed=Some("lane_b").
                ("gs_first_token_sample_dispatch", "null", "tree", 100, 200),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(4096, recs);
        let verdict = compute_t3_verdict(&per_pp);
        assert_eq!(
            verdict.verdict, "pass",
            "Lane B PP missing T3 spans must NOT fail PASS (spec § 3 T0a carve-out); \
             rationale={}",
            verdict.rationale
        );

        assert_eq!(verdict.cells.len(), 9);
        for c in &verdict.cells {
            assert_eq!(c.lane, "lane_a"); // span classification is unchanged
            assert_eq!(c.actual_lane_observed, Some("lane_b"));
            assert!(
                !c.expected_to_emit,
                "Lane B PP cell {} must have expected_to_emit=false",
                c.span_name
            );
            // record_count==0 is expected (no T3 spans on Lane B emission).
            assert_eq!(c.record_count, 0);
            assert!(c.median_inclusive_us.is_none());
        }
        assert!(
            verdict.missing_or_invalid.is_empty(),
            "Lane B PP must not contribute to missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    fn verdict_flags_missing_substep_on_lane_a() {
        // Lane-A request missing gather_qmm_down entirely → missing_spans
        // verdict + the missing cell appears in missing_or_invalid.
        let stderr = build_stderr(
            "req-A",
            ROUTING_PATH_LANE_A,
            128,
            &[
                ("mlp_path", "null", "tree", 100, 300),
                ("router_logits_softmax_topk", "mlp_path", "tree", 110, 120),
                ("routing_sort_pack", "mlp_path", "tree", 121, 130),
                ("gather_qmm_gate_up", "mlp_path", "tree", 131, 160),
                ("swiglu_activation", "mlp_path", "tree", 161, 170),
                // gather_qmm_down absent
                (
                    "routing_unsort_weighted_reduce",
                    "mlp_path",
                    "tree",
                    201,
                    210,
                ),
                ("shared_expert", "mlp_path", "tree", 211, 240),
                ("moe_output_sum", "mlp_path", "tree", 241, 250),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t3_verdict(&per_pp);
        assert_eq!(verdict.verdict, "missing_spans");
        assert!(
            verdict
                .missing_or_invalid
                .iter()
                .any(|m| m.contains("gather_qmm_down")),
            "expected gather_qmm_down in missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
        // The other 8 Lane-A spans should still pass.
        assert_eq!(
            verdict.missing_or_invalid.len(),
            1,
            "exactly one missing cell expected; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    fn verdict_passes_when_lane_b_substeps_missing() {
        // Two-PP scenario: PP=128 is Lane A with full emission, PP=4096 is
        // Lane B with no T3 spans. Verdict MUST be pass — Lane B missing
        // substeps are intentionally exempted.
        let lane_a = build_full_lane_a_emission("req-A", 158);
        let lane_b = build_stderr(
            "req-B",
            ROUTING_PATH_LANE_B,
            4126,
            &[("gs_first_token_sample_dispatch", "null", "tree", 100, 200)],
        );
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, parse_p5h_records(&lane_a));
        per_pp.insert(4096, parse_p5h_records(&lane_b));
        let verdict = compute_t3_verdict(&per_pp);
        assert_eq!(
            verdict.verdict, "pass",
            "Lane A complete + Lane B empty must PASS; rationale={}",
            verdict.rationale
        );
        assert!(verdict.missing_or_invalid.is_empty());
        assert_eq!(verdict.cells.len(), 18); // 2 PPs × 9 spans
        let lane_b_cells: Vec<&T3Cell> = verdict.cells.iter().filter(|c| c.pp == 4096).collect();
        assert_eq!(lane_b_cells.len(), 9);
        for c in &lane_b_cells {
            assert!(!c.expected_to_emit);
            assert_eq!(c.record_count, 0);
        }
    }

    #[test]
    #[should_panic(expected = "mixed routing_path values")]
    fn request_with_mixed_routing_path_panics() {
        // Single request carrying both scheduler + gs_chunked records —
        // emitter regression; verdict computation must panic.
        let line1 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=1 parent_span_id=null span_name=gather_qmm_gate_up parent_span=mlp_path \
             start_ns=100 end_ns=200 mode=off span_kind=tree\n",
        );
        let line2 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=gs_chunked prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=2 parent_span_id=null span_name=moe_output_sum parent_span=mlp_path \
             start_ns=300 end_ns=400 mode=off span_kind=tree\n",
        );
        let stderr = format!("{line1}{line2}").into_bytes();
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let _ = compute_t3_verdict(&per_pp);
    }

    #[test]
    #[should_panic(expected = "produced 0 [p5h-profile] records")]
    fn request_actual_lane_panics_on_empty_records() {
        // Direct test of request_actual_lane — an empty Vec is impossible
        // in practice but the function must fail-loud rather than silently
        // pick a default lane.
        let empty: Vec<&P5hProfileRecord> = Vec::new();
        let _ = request_actual_lane("req-empty", &empty);
    }
}
