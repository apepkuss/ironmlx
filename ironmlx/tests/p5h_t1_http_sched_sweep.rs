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
//! Per spec § 3 T1 + plan T1.1 (PP=2048 boundary correction + Lane B addition).
//!
//! ## Why PP=2047 (not 2048) and why PP=4096 was added
//!
//! `openai.rs:413`:
//!   `let use_scheduler = state.prefill_chunk_size == 0
//!                        || prompt_len <= state.prefill_chunk_size;`
//!
//! With the default `prefill_chunk_size=2048` the boundary is
//! `prompt_len ≤ 2048 → Lane A` and `prompt_len > 2048 → Lane B`.
//! `iron-bench --prompt-len N` does NOT request exactly N input tokens; the
//! chat-template render adds ~1 token of overhead, so `--prompt-len 2048`
//! reaches `prompt_len=2049` at the predicate and routes to Lane B
//! (the original T1 sweep at HEAD `fa1f6f6` therefore saw 0 records on the
//! 4 Lane-A spans for PP=2048 and verdict was `missing_spans`).
//!
//! Fix: PP=2047 keeps the Lane-A boundary measurement intact under
//! chat-template overhead, and PP=4096 is added to also exercise Lane B
//! (`serve_via_gs_stream`) so the Lane-B top-level spans get coverage in
//! the same harness.
//!
//! ## Verdict cell selection (lane intersection)
//!
//! For each PP, the verdict requires records for:
//!   * BOTH-lane spans (always — they fire on every request regardless of lane)
//!   * Lane-A spans  iff `lane_for_pp(pp) == "lane_a"`
//!   * Lane-B spans  iff `lane_for_pp(pp) == "lane_b"`
//!
//! With `T1_PP_LIST = [128, 512, 2047, 4096]` this yields 16 cells:
//!   - PP=128  (lane_a): 2 both + 2 lane_a = 4
//!   - PP=512  (lane_a): 2 both + 2 lane_a = 4
//!   - PP=2047 (lane_a): 2 both + 2 lane_a = 4
//!   - PP=4096 (lane_b): 2 both + 2 lane_b = 4
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

use std::collections::BTreeMap;

const T1_PP_LIST: [i32; 4] = [128, 512, 2047, 4096];
const T1_OUTPUT_PATH: &str = "/tmp/p5h-t1.json";

/// Spans that emit ONLY on Lane A (`serve_via_scheduler_stream`).
/// Verified per PP whose lane resolves to "lane_a".
/// Names MUST match exactly the `span_name` values emitted in
/// `[p5h-profile]` log lines (see `ironmlx/src/core/server/openai.rs` Lane-A
/// path + `ironmlx/src/core/p5h.rs` span schema § 2.5a).
const T1_LANE_A_SPAN_NAMES: [&str; 2] = ["scheduler_admission", "sse_write_role_chunk_diagnostic"];

/// Spans that emit ONLY on Lane B (`serve_via_gs_stream`).
/// Verified per PP whose lane resolves to "lane_b".
const T1_LANE_B_SPAN_NAMES: [&str; 2] = [
    "gs_first_token_sample_dispatch",
    "gs_first_token_materialize_and_predispatch",
];

/// Spans that fire on BOTH lanes. Verdict requires ≥1 record at every PP
/// (regardless of lane) because the handler entry / first-chunk-detok path
/// runs unconditionally for every request.
const T1_BOTH_LANE_SPAN_NAMES: [&str; 2] = [
    "http_parse_render_tokenize",
    "detok_format_first_content_chunk",
];

/// Default `prefill_chunk_size` in `ironmlx serve` — the threshold used by
/// the openai.rs:413 lane-routing predicate (`prompt_len <= prefill_chunk_size`
/// → Lane A). The chat-template render adds ~1 token of overhead in practice,
/// so the effective boundary in iron-bench `--prompt-len` terms is
/// `PP < 2048` for Lane A (PP=2047 fits, PP=2048 does not).
const PREFILL_CHUNK_SIZE_DEFAULT: i32 = 2048;

const PREHEAT_PROTOCOL_DESC: &str =
    "5min preheat per T0b binding: T1_PP_LIST × PREHEAT_RUNS=3 throwaway Phase A \
     iron-bench runs with spawn-kill per PP (using T1_PP_LIST=[128,512,2047,4096]); \
     results discarded; runs BEFORE first measurement spawn to drive GPU into \
     thermal saturation";

const INITIAL_COOL_PROTOCOL_DESC: &str =
    "INTER_PP_COOLDOWN=3s; per-PP spawn-kill so each PP starts from server cold-restart \
     (T0a/T0b precedent — no inter-phase cool gate, preheat handles thermal saturation)";

/// Resolve a request's lane based on PP. Mirrors the openai.rs:413 predicate,
/// adjusted for chat-template overhead (~1 token), so the effective boundary
/// is `PP < PREFILL_CHUNK_SIZE_DEFAULT`, not `<=`.
fn lane_for_pp(pp: i32) -> &'static str {
    if pp < PREFILL_CHUNK_SIZE_DEFAULT {
        "lane_a"
    } else {
        "lane_b"
    }
}

#[derive(Debug, serde::Serialize, Clone)]
struct P5hRecord {
    span_name: String,
    start_ns: u64,
    end_ns: u64,
    /// Either "tree" or "diagnostic" per `[p5h-profile]` `span_kind` field.
    /// T1 reports both but does not filter — caller can filter downstream.
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
/// T1 only consumes `span_name`, `start_ns`, `end_ns`, `span_kind`. The other
/// 9 fields are intentionally ignored — split_once('=') skips them.
///
/// Panics on a tagged line missing any of the 4 required fields or on a
/// malformed numeric value (failure-fast surfaces emission-side bugs).
fn parse_p5h_records(stderr_bytes: &[u8]) -> Vec<P5hRecord> {
    let s = String::from_utf8_lossy(stderr_bytes);
    let mut records = Vec::new();
    for line in s.lines() {
        let Some(start) = line.find("[p5h-profile]") else {
            continue;
        };
        let tail = &line[start + "[p5h-profile]".len()..];
        let mut span_name: Option<String> = None;
        let mut start_ns: Option<u64> = None;
        let mut end_ns: Option<u64> = None;
        let mut span_kind: Option<String> = None;
        for tok in tail.split_whitespace() {
            let Some((k, v)) = tok.split_once('=') else {
                continue;
            };
            match k {
                "span_name" => span_name = Some(v.to_string()),
                "start_ns" => {
                    start_ns = Some(
                        v.parse::<u64>()
                            .unwrap_or_else(|e| panic!("[p5h-profile] bad start_ns={v}: {e}")),
                    )
                }
                "end_ns" => {
                    end_ns = Some(
                        v.parse::<u64>()
                            .unwrap_or_else(|e| panic!("[p5h-profile] bad end_ns={v}: {e}")),
                    )
                }
                "span_kind" => span_kind = Some(v.to_string()),
                _ => {}
            }
        }
        let span_name =
            span_name.unwrap_or_else(|| panic!("[p5h-profile] line missing span_name: {line}"));
        let start_ns =
            start_ns.unwrap_or_else(|| panic!("[p5h-profile] line missing start_ns: {line}"));
        let end_ns = end_ns.unwrap_or_else(|| panic!("[p5h-profile] line missing end_ns: {line}"));
        let span_kind =
            span_kind.unwrap_or_else(|| panic!("[p5h-profile] line missing span_kind: {line}"));
        records.push(P5hRecord {
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
    /// Lane attribution for this cell:
    ///   "lane_a" — span only emits on `serve_via_scheduler_stream`
    ///   "lane_b" — span only emits on `serve_via_gs_stream`
    ///   "both"   — span fires on both lanes (handler entry / shared detok)
    lane: &'static str,
    /// Number of `[p5h-profile]` records matching `(span_name, pp)`.
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
) -> anyhow::Result<BTreeMap<i32, Vec<P5hRecord>>> {
    let mut per_pp: BTreeMap<i32, Vec<P5hRecord>> = BTreeMap::new();
    for &pp in &T1_PP_LIST {
        assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;
        // T1 does not set IRONMLX_P5G_PROFILE_MODE — Lane-A + Lane-B + shared
        // spans all emit unconditionally under --features p5h-profile.
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
            "[p5h-t1] PP={pp} lane={}: captured {} [p5h-profile] records ({} stderr bytes)",
            lane_for_pp(pp),
            records.len(),
            captured.len()
        );
        per_pp.insert(pp, records);
        std::thread::sleep(INTER_PP_COOLDOWN);
    }
    Ok(per_pp)
}

/// Build the (pp, span_name, lane) tuple list that the verdict will check.
/// Filters Lane-A / Lane-B spans by the PP's resolved lane while always
/// including BOTH-lane spans for every PP.
fn t1_expected_cells() -> Vec<(i32, &'static str, &'static str)> {
    let mut out: Vec<(i32, &'static str, &'static str)> = Vec::new();
    for &pp in &T1_PP_LIST {
        let lane = lane_for_pp(pp);
        for &span in &T1_BOTH_LANE_SPAN_NAMES {
            out.push((pp, span, "both"));
        }
        match lane {
            "lane_a" => {
                for &span in &T1_LANE_A_SPAN_NAMES {
                    out.push((pp, span, "lane_a"));
                }
            }
            "lane_b" => {
                for &span in &T1_LANE_B_SPAN_NAMES {
                    out.push((pp, span, "lane_b"));
                }
            }
            other => panic!("unexpected lane attribution {other:?} for PP={pp}"),
        }
    }
    out
}

/// Aggregate per-PP records into (PP, span_name, lane) cells, computing
/// trimmed_median(inclusive_us) per cell.
///
/// PASS criterion: every expected cell (per `t1_expected_cells`) must have
/// ≥1 record with `end_ns > start_ns`. Lane intersection is enforced by the
/// expected-cell generator so we never check Lane-B spans on a Lane-A PP
/// (those records are guaranteed-zero by construction, not a bug).
fn compute_t1_verdict(per_pp: &BTreeMap<i32, Vec<P5hRecord>>) -> T1Verdict {
    let mut cells: Vec<T1Cell> = Vec::new();
    let mut missing_or_invalid: Vec<String> = Vec::new();

    for (pp, span_name, lane) in t1_expected_cells() {
        let recs = per_pp.get(&pp);
        let filtered: Vec<&P5hRecord> = recs
            .map(|v| {
                v.iter()
                    .filter(|r| r.span_name == span_name && r.end_ns > r.start_ns)
                    .collect()
            })
            .unwrap_or_default();
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
                "PP={pp} lane={lane} span={span_name}: 0 valid records"
            ));
        }
        cells.push(T1Cell {
            pp,
            span_name: span_name.to_string(),
            lane,
            record_count: count,
            median_inclusive_us: median,
        });
    }

    let lane_a_pps: Vec<i32> = T1_PP_LIST
        .iter()
        .copied()
        .filter(|&p| lane_for_pp(p) == "lane_a")
        .collect();
    let lane_b_pps: Vec<i32> = T1_PP_LIST
        .iter()
        .copied()
        .filter(|&p| lane_for_pp(p) == "lane_b")
        .collect();
    let both_cells = T1_BOTH_LANE_SPAN_NAMES.len() * T1_PP_LIST.len();
    let lane_a_cells = T1_LANE_A_SPAN_NAMES.len() * lane_a_pps.len();
    let lane_b_cells = T1_LANE_B_SPAN_NAMES.len() * lane_b_pps.len();

    let verdict = if missing_or_invalid.is_empty() {
        "pass"
    } else {
        "missing_spans"
    };
    let rationale = if missing_or_invalid.is_empty() {
        format!(
            "T1 pass: every (PP, lane, span_name) cell has >=1 valid record. \
             PP_LIST={:?} lane_a + {:?} lane_b; cells={} (both={} + lane_a={} + lane_b={})",
            lane_a_pps,
            lane_b_pps,
            cells.len(),
            both_cells,
            lane_a_cells,
            lane_b_cells,
        )
    } else {
        format!(
            "T1 missing_spans: {} cells failed PASS criterion (>=1 record with end_ns>start_ns). \
             PP_LIST={:?} lane_a + {:?} lane_b; expected cells={} (both={} + lane_a={} + lane_b={}). \
             Failed cells: {}",
            missing_or_invalid.len(),
            lane_a_pps,
            lane_b_pps,
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

    // Record per-PP record counts (across all span_names) for diagnostic visibility.
    let per_pp_counts: BTreeMap<i32, usize> = per_pp.iter().map(|(k, v)| (*k, v.len())).collect();

    // Lane-attributed PP + span listings for the JSON consumer. The old flat
    // `pp_list` + `span_names` fields are retained below for back-compat with
    // downstream tooling that still greps for them.
    let mut pp_list_by_lane: BTreeMap<&'static str, Vec<i32>> = BTreeMap::new();
    for &pp in &T1_PP_LIST {
        pp_list_by_lane.entry(lane_for_pp(pp)).or_default().push(pp);
    }
    let mut span_names_by_lane: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
    span_names_by_lane.insert("lane_a", T1_LANE_A_SPAN_NAMES.to_vec());
    span_names_by_lane.insert("lane_b", T1_LANE_B_SPAN_NAMES.to_vec());
    span_names_by_lane.insert("both", T1_BOTH_LANE_SPAN_NAMES.to_vec());

    // Flat union of all expected span names (any lane) — back-compat field for
    // downstream consumers that grep `span_names` without lane awareness.
    let mut span_names_union: Vec<&'static str> = Vec::new();
    span_names_union.extend(T1_BOTH_LANE_SPAN_NAMES.iter().copied());
    span_names_union.extend(T1_LANE_A_SPAN_NAMES.iter().copied());
    span_names_union.extend(T1_LANE_B_SPAN_NAMES.iter().copied());

    let out_json = serde_json::json!({
        "pp_list": T1_PP_LIST,
        "span_names": span_names_union,
        "pp_list_by_lane": pp_list_by_lane,
        "span_names_by_lane": span_names_by_lane,
        "runs": RUNS,
        "warmup": WARMUP,
        "inter_pp_cooldown_secs": INTER_PP_COOLDOWN.as_secs(),
        "per_pp_record_counts": per_pp_counts,
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

// ===== Parser self-test: hand-crafted stderr sample =====
//
// Verify parse_p5h_records handles tracing's default formatter prefix and
// extracts all 4 required fields (span_name, start_ns, end_ns, span_kind).
// Runs without GPU under `cargo test --features p5h-profile`.

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn p5h_record_parser_extracts_fields() {
        // tracing's default formatter prefixes each line with
        // "<ts>  INFO <module>: <message>" — verify the parser handles it.
        // Field order matches `ironmlx/src/core/p5h.rs:313` emission contract:
        //   request_id routing_path prompt_tokens seq layer_idx span_id
        //   parent_span_id span_name parent_span start_ns end_ns mode span_kind
        let stderr = b"\
2026-05-22T12:34:56.789012Z  INFO ironmlx::core::p5h: [p5h-profile] request_id=req-1 routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 span_id=42 parent_span_id=null span_name=http_parse_render_tokenize parent_span=root start_ns=1000000 end_ns=1500000 mode=off span_kind=tree\n\
some unrelated line that should be ignored\n\
2026-05-22T12:34:56.890123Z  INFO ironmlx::core::p5h: [p5h-profile] request_id=req-1 routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 span_id=43 parent_span_id=42 span_name=scheduler_admission parent_span=root start_ns=1600000 end_ns=1700000 mode=off span_kind=tree\n\
2026-05-22T12:34:56.901234Z  INFO ironmlx::core::p5h: [p5h-profile] request_id=req-1 routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 span_id=44 parent_span_id=null span_name=sse_write_role_chunk_diagnostic parent_span=null start_ns=1800000 end_ns=1850000 mode=off span_kind=diagnostic\n\
2026-05-22T12:34:56.912345Z  INFO ironmlx::core::p5h: [p5h-profile] request_id=req-1 routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 span_id=45 parent_span_id=42 span_name=detok_format_first_content_chunk parent_span=root start_ns=1900000 end_ns=2000000 mode=off span_kind=tree\n\
";
        let recs = parse_p5h_records(stderr);
        assert_eq!(recs.len(), 4, "expected 4 P5h records, got {}", recs.len());

        assert_eq!(recs[0].span_name, "http_parse_render_tokenize");
        assert_eq!(recs[0].start_ns, 1_000_000);
        assert_eq!(recs[0].end_ns, 1_500_000);
        assert_eq!(recs[0].span_kind, "tree");

        assert_eq!(recs[1].span_name, "scheduler_admission");
        assert_eq!(recs[1].start_ns, 1_600_000);
        assert_eq!(recs[1].end_ns, 1_700_000);
        assert_eq!(recs[1].span_kind, "tree");

        assert_eq!(recs[2].span_name, "sse_write_role_chunk_diagnostic");
        assert_eq!(recs[2].start_ns, 1_800_000);
        assert_eq!(recs[2].end_ns, 1_850_000);
        assert_eq!(recs[2].span_kind, "diagnostic");

        assert_eq!(recs[3].span_name, "detok_format_first_content_chunk");
        assert_eq!(recs[3].start_ns, 1_900_000);
        assert_eq!(recs[3].end_ns, 2_000_000);
        assert_eq!(recs[3].span_kind, "tree");
    }

    #[test]
    fn p5h_record_parser_skips_non_p5h_lines() {
        // Lines without the [p5h-profile] tag must be silently ignored.
        let stderr = b"\
2026-05-22T12:34:56.789012Z  INFO some::other::module: unrelated INFO line\n\
2026-05-22T12:34:56.890123Z DEBUG ironmlx::other: [other-tag] foo=bar\n\
2026-05-22T12:34:56.901234Z  INFO ironmlx::core::p5h: [p5h-profile] request_id=r routing_path=scheduler prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 parent_span_id=null span_name=scheduler_admission parent_span=null start_ns=10 end_ns=20 mode=off span_kind=tree\n\
";
        let recs = parse_p5h_records(stderr);
        assert_eq!(recs.len(), 1, "only the [p5h-profile] line should match");
        assert_eq!(recs[0].span_name, "scheduler_admission");
        assert_eq!(recs[0].start_ns, 10);
        assert_eq!(recs[0].end_ns, 20);
    }

    #[test]
    fn lane_for_pp_matches_openai_predicate() {
        // Boundary cases: PP=2047 in Lane A, PP=2048 in Lane B (chat-template
        // overhead shifts the effective boundary by 1).
        assert_eq!(lane_for_pp(128), "lane_a");
        assert_eq!(lane_for_pp(512), "lane_a");
        assert_eq!(lane_for_pp(2047), "lane_a");
        assert_eq!(lane_for_pp(2048), "lane_b");
        assert_eq!(lane_for_pp(4096), "lane_b");
    }

    #[test]
    fn t1_expected_cells_lane_intersection() {
        let cells = t1_expected_cells();
        // 16 cells: 4 PPs × (2 both + 2 lane-specific) = 16.
        assert_eq!(cells.len(), 16, "expected 16 cells, got {}", cells.len());

        // Lane-A PPs must not produce any lane_b cells, and vice versa.
        for &(pp, span, lane) in &cells {
            match lane {
                "both" => {
                    assert!(
                        T1_BOTH_LANE_SPAN_NAMES.contains(&span),
                        "PP={pp} span={span} marked 'both' but not in T1_BOTH_LANE_SPAN_NAMES"
                    );
                }
                "lane_a" => {
                    assert_eq!(
                        lane_for_pp(pp),
                        "lane_a",
                        "lane_a cell at PP={pp} which resolves to {}",
                        lane_for_pp(pp),
                    );
                    assert!(
                        T1_LANE_A_SPAN_NAMES.contains(&span),
                        "PP={pp} span={span} marked 'lane_a' but not in T1_LANE_A_SPAN_NAMES"
                    );
                }
                "lane_b" => {
                    assert_eq!(
                        lane_for_pp(pp),
                        "lane_b",
                        "lane_b cell at PP={pp} which resolves to {}",
                        lane_for_pp(pp),
                    );
                    assert!(
                        T1_LANE_B_SPAN_NAMES.contains(&span),
                        "PP={pp} span={span} marked 'lane_b' but not in T1_LANE_B_SPAN_NAMES"
                    );
                }
                other => panic!("unexpected lane tag {other:?} at PP={pp} span={span}"),
            }
        }

        // Each PP contributes exactly 4 cells (2 both + 2 lane-specific).
        for &pp in &T1_PP_LIST {
            let n = cells.iter().filter(|(p, _, _)| *p == pp).count();
            assert_eq!(n, 4, "PP={pp} produced {n} cells, expected 4");
        }
    }
}
