//! P5h T1 — HTTP + Scheduler + Admission per-PP attribution sweep.
//!
//! Verifies that the 4 Lane-A top-level spans emit on every measured request
//! and reports trimmed_median(inclusive_us) per (span_name, PP):
//!   - http_parse_render_tokenize
//!   - scheduler_admission
//!   - sse_write_role_chunk_diagnostic
//!   - detok_format_first_content_chunk
//!
//! Output: /tmp/p5h-t1.json
//!
//! Per spec § 3 T1 + plan T1.1.
//!
//! Sweep PP ∈ {128, 512, 2048} (Lane A: PP ≤ default prefill_chunk_size).
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

const T1_PP_LIST: [i32; 3] = [128, 512, 2048];
const T1_OUTPUT_PATH: &str = "/tmp/p5h-t1.json";
/// Lane-A top-level spans verified by this sweep. Names MUST match exactly
/// the `span_name` values emitted in `[p5h-profile]` log lines from
/// `ironmlx/src/core/p5h.rs:313` (see § 2.5a span schema).
const T1_SPAN_NAMES: [&str; 4] = [
    "http_parse_render_tokenize",
    "scheduler_admission",
    "sse_write_role_chunk_diagnostic",
    "detok_format_first_content_chunk",
];

const PREHEAT_PROTOCOL_DESC: &str =
    "5min preheat per T0b binding: T1_PP_LIST × PREHEAT_RUNS=3 throwaway Phase A \
     iron-bench runs with spawn-kill per PP (using T1_PP_LIST=[128,512,2048]); \
     results discarded; runs BEFORE first measurement spawn to drive GPU into \
     thermal saturation";

const INITIAL_COOL_PROTOCOL_DESC: &str =
    "INTER_PP_COOLDOWN=3s; per-PP spawn-kill so each PP starts from server cold-restart \
     (T0a/T0b precedent — no inter-phase cool gate, preheat handles thermal saturation)";

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
    /// (PP, span_name) pairs that produced 0 records or only non-positive
    /// inclusive_us samples. Empty in the happy path.
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
        // T1 does not set IRONMLX_P5G_PROFILE_MODE — Lane-A spans emit
        // unconditionally under --features p5h-profile.
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

/// Aggregate per-PP records into (PP, span_name) cells, computing
/// trimmed_median(inclusive_us) per cell.
///
/// PASS criterion (per Boss decision 2): every (PP, span_name) cell must
/// have ≥1 record with `end_ns > start_ns`. Cells failing that produce a
/// "missing_spans" verdict with a diagnostic message listing which cells failed.
fn compute_t1_verdict(per_pp: &BTreeMap<i32, Vec<P5hRecord>>) -> T1Verdict {
    let mut cells: Vec<T1Cell> = Vec::new();
    let mut missing_or_invalid: Vec<String> = Vec::new();

    for &pp in &T1_PP_LIST {
        let recs = per_pp.get(&pp);
        for &span_name in &T1_SPAN_NAMES {
            // Collect inclusive_us samples for this (pp, span_name) cell.
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
                missing_or_invalid.push(format!("PP={pp} span={span_name}: 0 valid records"));
            }
            cells.push(T1Cell {
                pp,
                span_name: span_name.to_string(),
                record_count: count,
                median_inclusive_us: median,
            });
        }
    }

    let verdict = if missing_or_invalid.is_empty() {
        "pass"
    } else {
        "missing_spans"
    };
    let rationale = if missing_or_invalid.is_empty() {
        format!(
            "T1 pass: every (PP, span_name) cell has >=1 valid record (4 spans x {} PPs = {} cells)",
            T1_PP_LIST.len(),
            cells.len(),
        )
    } else {
        format!(
            "T1 missing_spans: {} cells failed PASS criterion (>=1 record with end_ns>start_ns). \
             Failed cells: {}",
            missing_or_invalid.len(),
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

    let out_json = serde_json::json!({
        "pp_list": T1_PP_LIST,
        "span_names": T1_SPAN_NAMES,
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
}
