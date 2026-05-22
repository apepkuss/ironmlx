//! P5h T1 — HTTP + Scheduler + Admission per-PP attribution sweep.
//!
//! Verifies that the per-lane top-level spans emit on every measured request
//! and reports trimmed_median(inclusive_us) per (span_name, PP):
//!
//!   Shared (BOTH lanes — fire on every PP):
//!     - http_parse_render_tokenize             (handler entry, before routing)
//!     - detok_format_first_content_chunk       (both Lane A + Lane B detok path)
//!
//!   Lane A only (`serve_via_scheduler_stream`):
//!     - scheduler_admission
//!     - sse_write_role_chunk_diagnostic
//!
//!   Lane B only (`serve_via_gs_stream`):
//!     - gs_first_token_sample_dispatch
//!     - gs_first_token_materialize_and_predispatch
//!
//! Output: /tmp/p5h-t1.json
//!
//! ## Lane attribution — derived from server-emitted `routing_path`, NOT nominal PP
//!
//! Two prior T1 sweep attempts (HEADs `fa1f6f6` and `8a10a64`) returned
//! verdict=`missing_spans` because nominal `pp < prefill_chunk_size(=2048)`
//! is **not** a reliable Lane A predictor:
//!
//!   * Routing predicate (`ironmlx/src/core/server/openai.rs:413`):
//!     `let use_scheduler = state.prefill_chunk_size == 0
//!                          || prompt_len <= state.prefill_chunk_size;`
//!   * `iron-bench --prompt-len N` does NOT request exactly N input tokens.
//!     The Qwen3 ChatML wrapper (`<|im_start|>system ...<|im_end|>\n`
//!     `<|im_start|>user ...<|im_end|>\n<|im_start|>assistant\n`) adds ~30
//!     tokens, so `--prompt-len 2048` reaches `prompt_len≈2078` at the
//!     predicate and routes to Lane B. Even `--prompt-len 2047` crossed the
//!     boundary in practice — both prior T1 attempts saw 0 records on the
//!     Lane-A spans for the PP that was supposed to land just under 2048.
//!
//! Per Codex review decision, T1 now derives the lane from the server's
//! emitted `routing_path` field (`"scheduler"` = Lane A, `"gs_chunked"` =
//! Lane B; see `ironmlx/src/core/p5h.rs:23` doc + emission line 313). The
//! expected per-PP cell set is computed from the actual `routing_path`
//! observed in `[p5h-profile]` records, not from any nominal-PP guess.
//!
//! `T1_PP_LIST = [128, 512, 1024, 4096]` chosen so:
//!   * `1024 + ~30 ≈ 1054 < 2048` → 3 safely-Lane-A PPs (128 / 512 / 1024).
//!   * `4096 + ~30 ≈ 4126 > 2048` → 1 safely-Lane-B PP.
//!
//! Production default `prefill_chunk_size=2048` preserved — `spawn_server`
//! is called with no `--prefill-chunk-size` override so the test reflects
//! the routing decisions a real client would trigger.
//!
//! ## Boundary tracking is OUT of T1 scope
//!
//! Boss + Codex decision: T1 = pure per-lane top-level span emission
//! coverage. Boundary-edge probing (i.e., does PP exactly at
//! `prefill_chunk_size` route as predicate documents) moves to T5 or a
//! separate follow-up so T1 keeps a single concern.
//!
//! ## Verdict cell selection
//!
//! Per request, the cell set required to PASS is:
//!   * `actual_lane == "lane_a"`: BOTH spans + Lane-A spans = 4
//!   * `actual_lane == "lane_b"`: BOTH spans + Lane-B spans = 4
//!
//! Aggregated at the (PP, span_name) cell level — a cell is expected to
//! emit ≥1 record iff at least one request at that PP routed through the
//! span's lane (with BOTH-lane spans always expected if any request at the
//! PP exists). Mismatched `routing_path` within a single `request_id`
//! panics with diagnostic (catches emitter regression).
//!
//! Server gate: --features p5h-profile (verifying [p5h-profile] schema itself,
//! not ProfileMode ablation). 5min preheat at entry per T0b binding.
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5h-profile \
//!     --test p5h_t1_http_sched_sweep -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

mod p5h_common;
use p5h_common::*;

use std::collections::{BTreeMap, BTreeSet};

/// PP set per Codex T1 binding: 3 PPs safely under Lane A boundary
/// (1024 + chat_template_overhead ~30 ≈ 1054 < 2048 = default prefill_chunk_size)
/// + 1 PP safely over (4096 + ~30 = 4126 > 2048).
///
/// Codex rationale: "do not bump --prefill-chunk-size to keep T1 testing
/// production default". Boundary tracking moved to T5 or follow-up.
const T1_PP_LIST: [i32; 4] = [128, 512, 1024, 4096];
const T1_OUTPUT_PATH: &str = "/tmp/p5h-t1.json";

/// Spans that emit ONLY on Lane A (`serve_via_scheduler_stream`).
/// Verified per PP whose actual_lane resolves to "lane_a" from per-request
/// `routing_path` aggregation.
/// Names MUST match exactly the `span_name` values emitted in
/// `[p5h-profile]` log lines (see `ironmlx/src/core/server/openai.rs` Lane-A
/// path + `ironmlx/src/core/p5h.rs` span schema § 2.5a).
const T1_LANE_A_SPAN_NAMES: [&str; 2] = ["scheduler_admission", "sse_write_role_chunk_diagnostic"];

/// Spans that emit ONLY on Lane B (`serve_via_gs_stream`).
/// Verified per PP whose actual_lane resolves to "lane_b".
const T1_LANE_B_SPAN_NAMES: [&str; 2] = [
    "gs_first_token_sample_dispatch",
    "gs_first_token_materialize_and_predispatch",
];

/// Spans that fire on BOTH lanes. Verdict requires ≥1 record at every PP
/// (regardless of actual_lane) because the handler entry / first-chunk-detok
/// path runs unconditionally for every request.
const T1_BOTH_LANE_SPAN_NAMES: [&str; 2] = [
    "http_parse_render_tokenize",
    "detok_format_first_content_chunk",
];

/// Lane identifiers as they appear in `[p5h-profile]` records' `routing_path`
/// field (per `ironmlx/src/core/p5h.rs:23`). Used to derive the per-request
/// `actual_lane` from server-emitted records.
const ROUTING_PATH_LANE_A: &str = "scheduler";
const ROUTING_PATH_LANE_B: &str = "gs_chunked";

const PREHEAT_PROTOCOL_DESC: &str =
    "5min preheat per T0b binding: T1_PP_LIST × PREHEAT_RUNS=3 throwaway Phase A \
     iron-bench runs with spawn-kill per PP (using T1_PP_LIST=[128,512,1024,4096]); \
     results discarded; runs BEFORE first measurement spawn to drive GPU into \
     thermal saturation";

const INITIAL_COOL_PROTOCOL_DESC: &str =
    "INTER_PP_COOLDOWN=3s; per-PP spawn-kill so each PP starts from server cold-restart \
     (T0a/T0b precedent — no inter-phase cool gate, preheat handles thermal saturation)";

/// Parsed `[p5h-profile]` record. Carries the lane-derivation inputs
/// (`request_id` + `routing_path` + `prompt_tokens`) in addition to the
/// span identity / timing fields used by the verdict.
#[derive(Debug, serde::Serialize, Clone)]
struct P5hProfileRecord {
    request_id: String,
    /// Server-emitted lane: `"scheduler"` (Lane A) | `"gs_chunked"` (Lane B).
    routing_path: String,
    /// Server-side tokenized prompt length AFTER chat-template render. Used
    /// to expose chat-template overhead per PP in `prompt_tokens_observed`.
    prompt_tokens: i64,
    span_name: String,
    start_ns: u64,
    end_ns: u64,
    /// Either "tree" or "diagnostic" per `[p5h-profile]` `span_kind` field.
    /// T1 reports but does not filter on this — caller can filter downstream.
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
/// T1 consumes `request_id`, `routing_path`, `prompt_tokens`, `span_name`,
/// `start_ns`, `end_ns`, `span_kind`. The other 6 fields
/// (seq, layer_idx, span_id, parent_span_id, parent_span, mode) are
/// intentionally ignored — split_once('=') skips them.
///
/// Panics on a tagged line missing any of the 7 required fields or on a
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
            span_name,
            start_ns,
            end_ns,
            span_kind,
        });
    }
    records
}

#[derive(Debug, serde::Serialize)]
struct T1Cell {
    pp: i32,
    span_name: String,
    /// Lane classification for this SPAN (not for the PP):
    ///   "lane_a" — span only emits on `serve_via_scheduler_stream`
    ///   "lane_b" — span only emits on `serve_via_gs_stream`
    ///   "both"   — span fires on both lanes (handler entry / shared detok)
    lane: &'static str,
    /// What the server actually routed for requests at this PP, derived per
    /// request from `routing_path` aggregation. Typically all RUNS=7 share a
    /// single value at the production default; if they diverge the value is
    /// `"mixed"` so the verdict / debugger can see it. None when the PP
    /// produced 0 requests (parser failure path — should be impossible).
    actual_lane_observed: Option<&'static str>,
    /// Number of `[p5h-profile]` records matching `(span_name, pp)` with
    /// `end_ns > start_ns`.
    record_count: usize,
    /// Trimmed median of `(end_ns - start_ns) / 1000` across all matching
    /// records (None when record_count == 0).
    median_inclusive_us: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct T1Verdict {
    verdict: String,
    rationale: String,
    cells: Vec<T1Cell>,
    /// (PP, lane, span_name) tuples that produced 0 records or only
    /// non-positive inclusive_us samples. Empty in the happy path.
    missing_or_invalid: Vec<String>,
    preheat_protocol: String,
    initial_cool_protocol: String,
}

/// Run iron-bench once per PP in `T1_PP_LIST`, capture server stderr, parse
/// `[p5h-profile]` records, and group them per (PP).
///
/// Per-PP spawn-kill matches T0a/T0b precedent: every PP starts from a fresh
/// server so cross-PP state (KV cache, MLX allocator, OS port reuse) cannot
/// confound the per-span timing. Failure paths go through `shutdown_and_join`
/// to avoid leaking Child processes.
fn run_t1_collect_records(
    model_dir: &str,
    port: u16,
) -> anyhow::Result<BTreeMap<i32, Vec<P5hProfileRecord>>> {
    let mut per_pp: BTreeMap<i32, Vec<P5hProfileRecord>> = BTreeMap::new();
    for &pp in &T1_PP_LIST {
        assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;
        // T1 does not set IRONMLX_P5G_PROFILE_MODE — Lane-A + Lane-B + shared
        // spans all emit unconditionally under --features p5h-profile.
        // No --prefill-chunk-size override: keep the production default so the
        // routing decisions seen by the test mirror real client behavior.
        let mut server = spawn_server(None, model_dir, port);
        let (stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

        if let Err(e) = wait_for_ready(port, 300) {
            shutdown_and_join(server, drainer);
            anyhow::bail!("PP={pp} T1: server not ready: {e}");
        }
        match server.try_wait() {
            Ok(Some(status)) => {
                let _ = drainer.join();
                anyhow::bail!("PP={pp} T1: ironmlx serve exited before bench: {status}");
            }
            Ok(None) => {}
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T1: try_wait failed: {e}");
            }
        }

        let out = match iron_bench_run(port, model_dir, pp) {
            Ok(o) => o,
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T1: iron-bench spawn failed: {e}");
            }
        };

        // Shutdown FIRST so drainer EOF + join completes before we drain.
        let _ = server.kill();
        let _ = server.wait();
        let _ = drainer.join();

        if !out.status.success() {
            anyhow::bail!(
                "PP={pp} T1: iron-bench non-success: stdout={}, stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let captured = drain_stderr_into_buf(&stderr_buf);
        let records = parse_p5h_records(&captured);
        eprintln!(
            "[p5h-t1] PP={pp}: captured {} [p5h-profile] records ({} stderr bytes)",
            records.len(),
            captured.len()
        );
        per_pp.insert(pp, records);
        std::thread::sleep(INTER_PP_COOLDOWN);
    }
    Ok(per_pp)
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

/// Aggregate per-PP records into (PP, span_name, lane) cells, computing
/// trimmed_median(inclusive_us) per cell.
///
/// Expected-cell derivation: for each PP, walk per-request `actual_lane`
/// values (from `routing_path`). The span set required to emit at that PP is:
///   * `T1_BOTH_LANE_SPAN_NAMES` — always, if any request exists at the PP.
///   * `T1_LANE_A_SPAN_NAMES` — iff ≥1 request at the PP routed Lane A.
///   * `T1_LANE_B_SPAN_NAMES` — iff ≥1 request at the PP routed Lane B.
///
/// PASS criterion: every expected cell has ≥1 record with `end_ns > start_ns`.
/// Lane intersection is enforced by the expected-cell derivation so we never
/// require a Lane-B span on a PP where every request routed Lane A.
fn compute_t1_verdict(per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>) -> T1Verdict {
    let mut cells: Vec<T1Cell> = Vec::new();
    let mut missing_or_invalid: Vec<String> = Vec::new();
    let mut lane_a_pps: BTreeSet<i32> = BTreeSet::new();
    let mut lane_b_pps: BTreeSet<i32> = BTreeSet::new();

    for (&pp, recs) in per_pp {
        // Group records by request_id, then derive per-request actual_lane.
        let by_req = group_by_request_id(recs);
        if by_req.is_empty() {
            // Parser succeeded but no records — likely server crashed or
            // emitter regression. Mark every BOTH-lane span as missing so the
            // verdict captures the failure visibly.
            for &span in &T1_BOTH_LANE_SPAN_NAMES {
                missing_or_invalid.push(format!(
                    "PP={pp} lane=both span={span}: PP produced 0 [p5h-profile] records"
                ));
                cells.push(T1Cell {
                    pp,
                    span_name: span.to_string(),
                    lane: "both",
                    actual_lane_observed: None,
                    record_count: 0,
                    median_inclusive_us: None,
                });
            }
            continue;
        }

        // Per-request actual_lane aggregation.
        let mut pp_request_lanes: BTreeSet<&'static str> = BTreeSet::new();
        for (req_id, req_recs) in &by_req {
            let lane = request_actual_lane(req_id, req_recs);
            pp_request_lanes.insert(lane);
        }
        let pp_has_lane_a = pp_request_lanes.contains("lane_a");
        let pp_has_lane_b = pp_request_lanes.contains("lane_b");

        // Track which PPs exercised which lane (for rationale + JSON).
        if pp_has_lane_a {
            lane_a_pps.insert(pp);
        }
        if pp_has_lane_b {
            lane_b_pps.insert(pp);
        }

        // Aggregated actual_lane_observed for this PP — "lane_a" / "lane_b"
        // when all requests agree, "mixed" when both lanes are represented.
        let actual_lane_observed: Option<&'static str> = match (pp_has_lane_a, pp_has_lane_b) {
            (true, false) => Some("lane_a"),
            (false, true) => Some("lane_b"),
            (true, true) => Some("mixed"),
            (false, false) => None, // unreachable: by_req non-empty guarantees ≥1 lane
        };

        // Expected span set: BOTH always; Lane-A iff pp_has_lane_a; Lane-B iff pp_has_lane_b.
        // For each cell, count records matching (span_name, valid inclusive_us).
        let mut emit_cell = |span: &'static str, lane: &'static str| {
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
            if count == 0 {
                missing_or_invalid.push(format!(
                    "PP={pp} lane={lane} span={span} actual_lane_observed={:?}: 0 valid records",
                    actual_lane_observed,
                ));
            }
            cells.push(T1Cell {
                pp,
                span_name: span.to_string(),
                lane,
                actual_lane_observed,
                record_count: count,
                median_inclusive_us: median,
            });
        };

        for &span in &T1_BOTH_LANE_SPAN_NAMES {
            emit_cell(span, "both");
        }
        if pp_has_lane_a {
            for &span in &T1_LANE_A_SPAN_NAMES {
                emit_cell(span, "lane_a");
            }
        }
        if pp_has_lane_b {
            for &span in &T1_LANE_B_SPAN_NAMES {
                emit_cell(span, "lane_b");
            }
        }
    }

    let lane_a_pps_vec: Vec<i32> = lane_a_pps.iter().copied().collect();
    let lane_b_pps_vec: Vec<i32> = lane_b_pps.iter().copied().collect();
    let both_cells = T1_BOTH_LANE_SPAN_NAMES.len() * per_pp.len();
    let lane_a_cells = T1_LANE_A_SPAN_NAMES.len() * lane_a_pps_vec.len();
    let lane_b_cells = T1_LANE_B_SPAN_NAMES.len() * lane_b_pps_vec.len();

    let verdict = if missing_or_invalid.is_empty() {
        "pass"
    } else {
        "missing_spans"
    };
    let rationale = if missing_or_invalid.is_empty() {
        format!(
            "T1 pass: every (PP, lane, span_name) cell has >=1 valid record. \
             Lane-A PPs (observed)={lane_a_pps_vec:?}, Lane-B PPs (observed)={lane_b_pps_vec:?}; \
             cells={} (both={} + lane_a={} + lane_b={})",
            cells.len(),
            both_cells,
            lane_a_cells,
            lane_b_cells,
        )
    } else {
        format!(
            "T1 missing_spans: {} cells failed PASS criterion (>=1 record with end_ns>start_ns). \
             Lane-A PPs (observed)={lane_a_pps_vec:?}, Lane-B PPs (observed)={lane_b_pps_vec:?}; \
             expected cells={} (both={} + lane_a={} + lane_b={}). Failed cells: {}",
            missing_or_invalid.len(),
            cells.len(),
            both_cells,
            lane_a_cells,
            lane_b_cells,
            missing_or_invalid.join("; "),
        )
    };
    T1Verdict {
        verdict: verdict.to_string(),
        rationale,
        cells,
        missing_or_invalid,
        preheat_protocol: PREHEAT_PROTOCOL_DESC.to_string(),
        initial_cool_protocol: INITIAL_COOL_PROTOCOL_DESC.to_string(),
    }
}

/// Per-PP, per-request observed `prompt_tokens` (the server-side tokenized
/// count post chat-template render). Used in the JSON payload so future
/// debugging is grounded: e.g., shows that `--prompt-len 2048` lands at
/// `prompt_tokens ≈ 2078` and therefore routes Lane B at default
/// `prefill_chunk_size=2048`.
fn prompt_tokens_observed_per_pp(
    per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>,
) -> BTreeMap<i32, Vec<i64>> {
    let mut out: BTreeMap<i32, Vec<i64>> = BTreeMap::new();
    for (&pp, recs) in per_pp {
        let by_req = group_by_request_id(recs);
        let mut tokens: Vec<i64> = by_req
            .values()
            .map(|req_recs| {
                // All records for a given request_id should carry the same
                // prompt_tokens (it's a property of the request, not the span).
                // Take the first record's value; report per-request rather than
                // per-record to avoid RUNS×spans inflation.
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
/// is the typical case at production default; divergent values are visible
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
#[ignore = "p5h-t1 HTTP+scheduler+admission per-PP attribution sweep (~10-15min GPU + 5min preheat)"]
fn t1_http_sched_admission_sweep() -> anyhow::Result<()> {
    let model_dir = snapshot_dir();
    let _mlx_dir = std::env::var("MLX_DIR")
        .expect("set MLX_DIR env var pointing to MLX install prefix (e.g. $HOME/.local/mlx)");
    eprintln!("[p5h-t1] starting; model={model_dir}");

    eprintln!("[p5h-t1] preheat phase");
    preheat_to_saturation(&model_dir, PROFILE_PORT, &T1_PP_LIST)?;

    eprintln!("[p5h-t1] measurement phase (PP ∈ {T1_PP_LIST:?})");
    let per_pp = run_t1_collect_records(&model_dir, PROFILE_PORT)?;

    let verdict = compute_t1_verdict(&per_pp);
    eprintln!("[p5h-t1] {}", verdict.rationale);

    // Per-PP record counts (across all span_names) for diagnostic visibility.
    let per_pp_counts: BTreeMap<i32, usize> = per_pp.iter().map(|(k, v)| (*k, v.len())).collect();

    // Lane-derivation observation series — keys the ChatML-overhead lesson
    // into the JSON output (per Codex review decision: grounding for future
    // debugging, replaces the obsolete `pp_list_by_lane` field).
    let prompt_tokens_observed = prompt_tokens_observed_per_pp(&per_pp);
    let actual_lane_per_pp_obs = actual_lane_per_pp(&per_pp);

    // Flat union of all expected span names (any lane) — back-compat field for
    // downstream consumers that grep `span_names` without lane awareness.
    let mut span_names_union: Vec<&'static str> = Vec::new();
    span_names_union.extend(T1_BOTH_LANE_SPAN_NAMES.iter().copied());
    span_names_union.extend(T1_LANE_A_SPAN_NAMES.iter().copied());
    span_names_union.extend(T1_LANE_B_SPAN_NAMES.iter().copied());

    let out_json = serde_json::json!({
        "pp_list": T1_PP_LIST,
        "span_names": span_names_union,
        "runs": RUNS,
        "warmup": WARMUP,
        "inter_pp_cooldown_secs": INTER_PP_COOLDOWN.as_secs(),
        "per_pp_record_counts": per_pp_counts,
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
    eprintln!("[p5h-t1] JSON payload (preserved in case file-write fails):\n{json_str}");
    std::fs::write(T1_OUTPUT_PATH, &json_str)?;
    eprintln!(
        "[p5h-t1] wrote {} bytes to {T1_OUTPUT_PATH}",
        json_str.len()
    );

    // Per T0b harness convention: return Ok regardless of verdict — verdict
    // string in JSON is the consumed signal. PASS / missing_spans both produce
    // a valid JSON report that downstream tooling reads.
    Ok(())
}

// ===== Unit tests: parser + lane-derivation + verdict shape =====
//
// All unit tests run without GPU under `cargo test --features p5h-profile`.

#[cfg(test)]
mod parser_tests {
    use super::*;

    /// Build a single-request stderr blob with the given (span_name, span_kind)
    /// pairs, all sharing the supplied request_id + routing_path + prompt_tokens.
    fn build_stderr(
        request_id: &str,
        routing_path: &str,
        prompt_tokens: i64,
        spans: &[(&str, &str, u64, u64)],
    ) -> Vec<u8> {
        let mut out = String::new();
        for (i, (span_name, span_kind, start, end)) in spans.iter().enumerate() {
            out.push_str(&format!(
                "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
                 [p5h-profile] request_id={request_id} routing_path={routing_path} \
                 prompt_tokens={prompt_tokens} seq=0 layer_idx=-1 span_id={span_id} \
                 parent_span_id=null span_name={span_name} parent_span=null \
                 start_ns={start} end_ns={end} mode=off span_kind={span_kind}\n",
                span_id = 1000 + i,
            ));
        }
        out.into_bytes()
    }

    #[test]
    fn p5h_record_parser_extracts_all_fields() {
        // tracing's default formatter prefixes each line with
        // "<ts>  INFO <module>: <message>" — verify the parser handles it.
        // Field order matches `ironmlx/src/core/p5h.rs:313` emission contract:
        //   request_id routing_path prompt_tokens seq layer_idx span_id
        //   parent_span_id span_name parent_span start_ns end_ns mode span_kind
        let stderr = build_stderr(
            "req-1",
            "scheduler",
            128,
            &[
                ("http_parse_render_tokenize", "tree", 1_000_000, 1_500_000),
                ("scheduler_admission", "tree", 1_600_000, 1_700_000),
                (
                    "sse_write_role_chunk_diagnostic",
                    "diagnostic",
                    1_800_000,
                    1_850_000,
                ),
                (
                    "detok_format_first_content_chunk",
                    "tree",
                    1_900_000,
                    2_000_000,
                ),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        assert_eq!(recs.len(), 4, "expected 4 P5h records, got {}", recs.len());

        for r in &recs {
            assert_eq!(r.request_id, "req-1");
            assert_eq!(r.routing_path, "scheduler");
            assert_eq!(r.prompt_tokens, 128);
        }

        assert_eq!(recs[0].span_name, "http_parse_render_tokenize");
        assert_eq!(recs[0].start_ns, 1_000_000);
        assert_eq!(recs[0].end_ns, 1_500_000);
        assert_eq!(recs[0].span_kind, "tree");

        assert_eq!(recs[1].span_name, "scheduler_admission");
        assert_eq!(recs[1].span_kind, "tree");

        assert_eq!(recs[2].span_name, "sse_write_role_chunk_diagnostic");
        assert_eq!(recs[2].span_kind, "diagnostic");

        assert_eq!(recs[3].span_name, "detok_format_first_content_chunk");
        assert_eq!(recs[3].start_ns, 1_900_000);
        assert_eq!(recs[3].end_ns, 2_000_000);
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
             parent_span_id=null span_name=scheduler_admission parent_span=null \
             start_ns=10 end_ns=20 mode=off span_kind=tree\n",
        );
        let recs = parse_p5h_records(stderr.as_bytes());
        assert_eq!(recs.len(), 1, "only the [p5h-profile] line should match");
        assert_eq!(recs[0].request_id, "r");
        assert_eq!(recs[0].routing_path, "scheduler");
        assert_eq!(recs[0].prompt_tokens, 2);
        assert_eq!(recs[0].span_name, "scheduler_admission");
        assert_eq!(recs[0].start_ns, 10);
        assert_eq!(recs[0].end_ns, 20);
    }

    #[test]
    #[should_panic(expected = "line missing routing_path")]
    fn p5h_record_parser_fails_on_missing_routing_path() {
        // Strip routing_path from an otherwise valid line — parser must panic.
        let line = b"2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
            [p5h-profile] request_id=r prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 \
            parent_span_id=null span_name=scheduler_admission parent_span=null \
            start_ns=10 end_ns=20 mode=off span_kind=tree\n";
        let _ = parse_p5h_records(line);
    }

    #[test]
    fn lane_derivation_pure_lane_a() {
        // Single request, all records routing_path=scheduler → lane_a.
        let stderr = build_stderr(
            "req-A",
            "scheduler",
            128,
            &[
                ("http_parse_render_tokenize", "tree", 100, 200),
                ("scheduler_admission", "tree", 300, 400),
                ("sse_write_role_chunk_diagnostic", "diagnostic", 500, 600),
                ("detok_format_first_content_chunk", "tree", 700, 800),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t1_verdict(&per_pp);
        assert_eq!(verdict.verdict, "pass", "rationale={}", verdict.rationale);

        // Cells: 2 both + 2 lane_a = 4 (no lane_b since no request routed Lane B).
        assert_eq!(verdict.cells.len(), 4);
        let lane_a_cells: Vec<&T1Cell> = verdict
            .cells
            .iter()
            .filter(|c| c.lane == "lane_a")
            .collect();
        assert_eq!(lane_a_cells.len(), 2);
        let both_cells: Vec<&T1Cell> = verdict.cells.iter().filter(|c| c.lane == "both").collect();
        assert_eq!(both_cells.len(), 2);
        let lane_b_cells: Vec<&T1Cell> = verdict
            .cells
            .iter()
            .filter(|c| c.lane == "lane_b")
            .collect();
        assert!(
            lane_b_cells.is_empty(),
            "no lane_b cells expected when no request routed Lane B"
        );

        // actual_lane_observed must read lane_a everywhere.
        for c in &verdict.cells {
            assert_eq!(c.actual_lane_observed, Some("lane_a"));
        }
    }

    #[test]
    fn lane_derivation_pure_lane_b() {
        // Single request, routing_path=gs_chunked → lane_b.
        let stderr = build_stderr(
            "req-B",
            "gs_chunked",
            4096,
            &[
                ("http_parse_render_tokenize", "tree", 100, 200),
                ("gs_first_token_sample_dispatch", "tree", 300, 400),
                (
                    "gs_first_token_materialize_and_predispatch",
                    "tree",
                    500,
                    600,
                ),
                ("detok_format_first_content_chunk", "tree", 700, 800),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(4096, recs);
        let verdict = compute_t1_verdict(&per_pp);
        assert_eq!(verdict.verdict, "pass", "rationale={}", verdict.rationale);
        assert_eq!(verdict.cells.len(), 4);
        for c in &verdict.cells {
            assert_eq!(c.actual_lane_observed, Some("lane_b"));
        }
        let lane_a_cells: Vec<&T1Cell> = verdict
            .cells
            .iter()
            .filter(|c| c.lane == "lane_a")
            .collect();
        assert!(lane_a_cells.is_empty());
    }

    #[test]
    fn verdict_flags_missing_lane_a_span() {
        // Lane-A request missing the scheduler_admission span entirely →
        // missing_spans verdict + reference cell in missing_or_invalid.
        let stderr = build_stderr(
            "req-A",
            "scheduler",
            128,
            &[
                ("http_parse_render_tokenize", "tree", 100, 200),
                // scheduler_admission absent
                ("sse_write_role_chunk_diagnostic", "diagnostic", 500, 600),
                ("detok_format_first_content_chunk", "tree", 700, 800),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t1_verdict(&per_pp);
        assert_eq!(verdict.verdict, "missing_spans");
        assert!(
            verdict
                .missing_or_invalid
                .iter()
                .any(|m| m.contains("scheduler_admission")),
            "expected scheduler_admission in missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    #[should_panic(expected = "mixed routing_path values")]
    fn request_with_mixed_routing_path_panics() {
        // Single request carrying both scheduler + gs_chunked records —
        // emitter regression; verdict computation must panic.
        let line1 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=1 parent_span_id=null span_name=http_parse_render_tokenize parent_span=null \
             start_ns=100 end_ns=200 mode=off span_kind=tree\n",
        );
        let line2 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=gs_chunked prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=2 parent_span_id=null span_name=detok_format_first_content_chunk parent_span=null \
             start_ns=300 end_ns=400 mode=off span_kind=tree\n",
        );
        let stderr = format!("{line1}{line2}").into_bytes();
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let _ = compute_t1_verdict(&per_pp);
    }

    #[test]
    #[should_panic(expected = "produced 0 [p5h-profile] records")]
    fn request_actual_lane_panics_on_empty_records() {
        // Direct test of request_actual_lane — an empty Vec is impossible in
        // practice but the function must fail-loud rather than silently pick
        // a default lane.
        let empty: Vec<&P5hProfileRecord> = Vec::new();
        let _ = request_actual_lane("req-empty", &empty);
    }

    #[test]
    fn prompt_tokens_observed_aggregates_per_request() {
        // Two requests at PP=128, one with prompt_tokens=158 (scheduler) and
        // one with prompt_tokens=162 (scheduler) — both should be present in
        // the per-PP Vec.
        let s1 = build_stderr(
            "req-1",
            "scheduler",
            158,
            &[("http_parse_render_tokenize", "tree", 100, 200)],
        );
        let s2 = build_stderr(
            "req-2",
            "scheduler",
            162,
            &[("http_parse_render_tokenize", "tree", 300, 400)],
        );
        let mut combined = s1;
        combined.extend_from_slice(&s2);
        let recs = parse_p5h_records(&combined);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let obs = prompt_tokens_observed_per_pp(&per_pp);
        assert_eq!(
            obs.get(&128).map(|v| v.as_slice()),
            Some([158, 162].as_slice())
        );
    }
}
