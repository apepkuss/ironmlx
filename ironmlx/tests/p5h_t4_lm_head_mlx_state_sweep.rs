//! P5h T4 — lm_head + MLX state + tokenization + first-eval emission sweep
//! (Lane A primary) + Lane B opportunistic top-level smoke.
//!
//! Verifies the T4.1-T4.5 instrumentation (commits `3e31537` → `085fba9`)
//! emits one `[p5h-profile]` record per expected span per request, and
//! reports the trimmed_median(inclusive_us) per (PP, span_name) cell.
//!
//! ## Span set (5 spans, heterogeneous lane/verdict semantics)
//!
//!   * `slice_last_and_project_lm_head` — T4.1 (commit `3e31537`).
//!     Wraps `Qwen35MoeModel::slice_last_and_project` lm_head projection in
//!     `ironmlx/src/models/qwen3_5_moe/model.rs`. Lane A only — called from
//!     `model_prefill_forward` under `prefill_admitted`. Expected ≥1 record
//!     per Lane A request with `inclusive_us > 0` (one lm_head call per
//!     prefill; `--max-tokens 1` ⇒ 1 prefill + 0 decode).
//!
//!   * `mlx_eval_barrier` — T4.2 (commit `be2491b`). INERT today: T4.2
//!     wraps `mlx::transforms::eval(...)` in `admit_mid_chunk` VL + text
//!     branches at `ironmlx/src/core/scheduler.rs:1698,1721`, but the
//!     `handle_admit_mid_chunked` path does NOT push a `P5hTraceGuard`, so
//!     the wrapper no-ops on the primary streaming path. T4.6 reports
//!     `record_count` for diagnostic visibility but DOES NOT verdict-gate
//!     on it (cell.expected_to_emit=false on both lanes; deferred plumbing
//!     tracked as P5h+1 follow-up).
//!
//!   * `cache_state_update` — T4.3 (commit `08b876a`). Wraps cache update
//!     at 3 caller sites (KVCache::update_and_fetch_on inside
//!     `kv_mask_update`; GatedDeltaCache update_conv inside `gda_step_2c`;
//!     GatedDeltaCache update_recurrent + advance inside `gda_step_7`).
//!     Lane A only — gated_attention / gated_delta_net only run inside
//!     Lane A `prefill_admitted_inner`. Expected ≥1 record per Lane A
//!     request. SPECIAL: accept `inclusive_us ≥ 0` (NOT `> 0` like other
//!     spans). The GDN cache ops are CPU-only Arc-share / offset-increment
//!     and legitimately report `inclusive_us == 0`; only
//!     `KVCache::update_and_fetch_on` does meaningful work. The
//!     mixed-population median may be small but at least one of the 3
//!     callers will produce non-zero — we accept the cell so long as
//!     `record_count > 0`.
//!
//!   * `tokenizer_encode` — T4.4 (commit `883b5b9`). Retroactive subspan
//!     of `http_parse_render_tokenize` opened in the OpenAI HTTP handler
//!     BEFORE lane routing. Lane-agnostic: fires on BOTH lanes (1 record
//!     per request). Expected ≥1 record per PP with `inclusive_us > 0`.
//!
//!   * `first_eval_amortized_cost` — T4.5 (commit `085fba9`). Diagnostic
//!     span (`span_kind="diagnostic"`) emitted at most once per process via
//!     static `OnceLock`. Wraps the first `prefill_admitted_inner`
//!     `model_prefill_forward` body. Lane A only. Per-PP spawn-kill =
//!     fresh process per PP, so expected ≥1 record per Lane A PP (fires on
//!     the FIRST of RUNS=7 requests at that PP; subsequent 6 see
//!     `OnceLock` already set and no-op).
//!
//! ## Lane-aware cell schema (per Codex Q-T4-4 binding)
//!
//! `T4_PP_LIST = [128, 512, 1024, 4096]` mirrors T1/T2/T3 (3 Lane A + 1
//! Lane B opportunistic). Lane derivation is runtime-by-`routing_path`
//! per T1 lesson — the Qwen3 ChatML wrapper adds ~30 tokens so
//! `--prompt-len N` reaches `prompt_len ≈ N+30` at the routing predicate.
//!
//! Each cell carries:
//!   * `lane` — span classification: `"lane_a"` | `"both"` | `"inert"`.
//!   * `actual_lane_observed` — what the PP actually routed
//!     (`"lane_a"` | `"lane_b"` | `"mixed"` | None).
//!   * `expected_to_emit` — derived from `(meta.lane, actual_lane_observed)`:
//!       - `"inert"` ⇒ `false` (T4.2 INERT today, both lanes)
//!       - `"lane_a"`, observed `"lane_a"` or `"mixed"` ⇒ `true`
//!       - `"lane_a"`, observed `"lane_b"` ⇒ `false` (Lane B exempt)
//!       - `"both"`, observed `Some(_)` ⇒ `true` (fires on either lane)
//!   * `span_kind` — `"tree"` | `"diagnostic"` (derived from meta).
//!
//! ## Verdict
//!
//! PASS iff every cell with `expected_to_emit == true` has
//! `record_count > 0`. For non-`cache_state_update` spans the cell also
//! needs `median_inclusive_us > 0` (matches T2/T3 semantics). For
//! `cache_state_update`, accept any `record_count > 0` regardless of
//! median (mixed CPU/GPU population may have a 0us median legitimately).
//!
//! `mlx_eval_barrier` records are reported but never participate in
//! verdict (always `expected_to_emit=false` per INERT classification).
//! Lane B PP cells for Lane-A-only spans are reported with
//! `record_count` for diagnostic visibility but exempt from PASS per spec
//! § 3 T0a Lane-B suppression carve-out.
//!
//! Server gate: `--features p5h-profile` (verifying the [p5h-profile]
//! emission schema itself, not ProfileMode ablation). 5min preheat at
//! entry per T0b binding. Per-PP spawn-kill per T0a/T0b/T1/T2/T3
//! precedent.
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5h-profile \
//!     --test p5h_t4_lm_head_mlx_state_sweep -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

mod p5h_common;
use p5h_common::*;

use std::collections::{BTreeMap, BTreeSet};

/// PP set per Codex Q-T4-4 binding: 3 Lane A + 1 Lane B opportunistic.
/// Mirrors T1/T2/T3. Lane derivation is runtime-by-`routing_path` per T1
/// lesson — NOT nominal-PP-based.
const T4_PP_LIST: [i32; 4] = [128, 512, 1024, 4096];

/// T4 span set per T4.1-T4.5 commits (`3e31537` → `085fba9`). Each span
/// has different lane attribution + verdict semantics — see `t4_span_meta`.
const T4_SPAN_NAMES: [&str; 5] = [
    "slice_last_and_project_lm_head",
    "mlx_eval_barrier",
    "cache_state_update",
    "tokenizer_encode",
    "first_eval_amortized_cost",
];

const T4_OUTPUT_PATH: &str = "/tmp/p5h-t4.json";

/// Lane identifiers as they appear in `[p5h-profile]` records' `routing_path`
/// field (per `ironmlx/src/core/p5h.rs:23`). Used to derive the per-request
/// `actual_lane` from server-emitted records.
const ROUTING_PATH_LANE_A: &str = "scheduler";
const ROUTING_PATH_LANE_B: &str = "gs_chunked";

const PREHEAT_PROTOCOL_DESC: &str =
    "5min preheat per T0b binding: T4_PP_LIST × PREHEAT_RUNS=3 throwaway Phase A \
     iron-bench runs with spawn-kill per PP (using T4_PP_LIST=[128,512,1024,4096]); \
     results discarded; runs BEFORE first measurement spawn to drive GPU into \
     thermal saturation";

const INITIAL_COOL_PROTOCOL_DESC: &str =
    "INTER_PP_COOLDOWN=3s; per-PP spawn-kill so each PP starts from server cold-restart \
     (T0a/T0b/T1/T2/T3 precedent — no inter-phase cool gate, preheat handles thermal \
     saturation)";

/// Per-span metadata: lane classification + verdict treatment. The
/// `lane`/`accept_zero_inclusive_us`/`is_diagnostic` triple captures the
/// heterogeneity of the T4 span set (vs. T3's uniform Lane-A tree spans).
struct T4SpanMeta {
    #[allow(dead_code)]
    span_name: &'static str,
    /// `"lane_a"` | `"both"` | `"inert"` — what lane(s) this span is
    /// designed to fire on.
    lane: &'static str,
    /// `true` for `cache_state_update`: accept records with
    /// `inclusive_us == 0` because the 3 callers include CPU-only Arc-
    /// share / offset-increment sites (GDN cache ops) that legitimately
    /// report 0us. Verdict only requires `record_count > 0`.
    accept_zero_inclusive_us: bool,
    /// `true` for `first_eval_amortized_cost` (T4.5) — emitted via
    /// `close_p5h_span_diagnostic` ⇒ `span_kind="diagnostic"`.
    is_diagnostic: bool,
}

fn t4_span_meta(span_name: &str) -> T4SpanMeta {
    match span_name {
        "slice_last_and_project_lm_head" => T4SpanMeta {
            span_name: "slice_last_and_project_lm_head",
            lane: "lane_a",
            accept_zero_inclusive_us: false,
            is_diagnostic: false,
        },
        "mlx_eval_barrier" => T4SpanMeta {
            span_name: "mlx_eval_barrier",
            lane: "inert",
            accept_zero_inclusive_us: true,
            is_diagnostic: false,
        },
        "cache_state_update" => T4SpanMeta {
            span_name: "cache_state_update",
            lane: "lane_a",
            accept_zero_inclusive_us: true,
            is_diagnostic: false,
        },
        "tokenizer_encode" => T4SpanMeta {
            span_name: "tokenizer_encode",
            lane: "both",
            accept_zero_inclusive_us: false,
            is_diagnostic: false,
        },
        "first_eval_amortized_cost" => T4SpanMeta {
            span_name: "first_eval_amortized_cost",
            lane: "lane_a",
            accept_zero_inclusive_us: false,
            is_diagnostic: true,
        },
        other => panic!("t4_span_meta: unknown T4 span_name {other}"),
    }
}

/// Parsed `[p5h-profile]` record. Mirrors T2.3/T3.3 — carries the
/// lane-derivation inputs (`request_id` + `routing_path` + `prompt_tokens`)
/// in addition to span identity / timing fields used by the verdict.
/// `span_kind` is REQUIRED here (T4 needs to verify the `"diagnostic"`
/// classification on `first_eval_amortized_cost`).
#[derive(Debug, serde::Serialize, Clone)]
struct P5hProfileRecord {
    request_id: String,
    /// Server-emitted lane: `"scheduler"` (Lane A) | `"gs_chunked"` (Lane B).
    routing_path: String,
    /// Server-side tokenized prompt length AFTER chat-template render.
    prompt_tokens: i64,
    /// Parent span label from the emitter. Recorded for downstream
    /// diagnostic / T5 structural check; not enforced in T4 verdict.
    parent_span: String,
    span_name: String,
    start_ns: u64,
    end_ns: u64,
    /// `"tree"` or `"diagnostic"` per `[p5h-profile]` emission contract.
    span_kind: String,
}

/// Parse `[p5h-profile]` records from server stderr. Mirrors T2.3/T3.3
/// parser exactly — fails loud on a tagged line missing any of the 8
/// required fields or on a malformed numeric value.
///
/// Field set (per `ironmlx/src/core/p5h.rs:313` emission contract):
///   request_id routing_path prompt_tokens seq layer_idx span_id parent_span_id
///   span_name parent_span start_ns end_ns mode span_kind
///
/// T4 consumes `request_id`, `routing_path`, `prompt_tokens`, `parent_span`,
/// `span_name`, `start_ns`, `end_ns`, `span_kind`. The other 5 fields
/// (seq, layer_idx, span_id, parent_span_id, mode) are intentionally
/// ignored — split_once('=') skips them.
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
struct T4Cell {
    pp: i32,
    span_name: String,
    /// Span lane classification: `"lane_a"` | `"both"` | `"inert"`.
    lane: &'static str,
    /// What the server actually routed for requests at this PP, derived
    /// per request from `routing_path` aggregation. `"lane_a"` |
    /// `"lane_b"` | `"mixed"`. `None` when the PP produced 0 requests.
    actual_lane_observed: Option<&'static str>,
    /// True iff this cell's span is expected to emit at this PP given the
    /// PP's `actual_lane_observed` and the span's lane classification.
    expected_to_emit: bool,
    /// Number of `[p5h-profile]` records matching `(span_name, pp)`. For
    /// `cache_state_update` (accept_zero_inclusive_us=true) all records
    /// count; for other spans only records with `end_ns > start_ns` count.
    record_count: usize,
    /// Trimmed median of `(end_ns - start_ns) / 1000` across all
    /// counted records (None when record_count == 0).
    median_inclusive_us: Option<f64>,
    /// `"tree"` | `"diagnostic"` per span meta.
    span_kind: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct T4Verdict {
    verdict: String,
    rationale: String,
    cells: Vec<T4Cell>,
    /// (PP, lane, span_name) tuples whose cell has `expected_to_emit=true`
    /// but produced 0 valid records. Empty in the happy path.
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

/// Run iron-bench once per PP in `T4_PP_LIST`, capture server stderr, parse
/// `[p5h-profile]` records, and group them per (PP).
///
/// Per-PP spawn-kill matches T0a/T0b/T1/T2/T3 precedent: every PP starts
/// from a fresh server. Failure paths go through `shutdown_and_join` to
/// avoid leaking Child processes.
fn run_t4_collect_records(
    model_dir: &str,
    port: u16,
) -> anyhow::Result<BTreeMap<i32, Vec<P5hProfileRecord>>> {
    let mut per_pp: BTreeMap<i32, Vec<P5hProfileRecord>> = BTreeMap::new();
    for &pp in &T4_PP_LIST {
        assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;
        // T4 does NOT set IRONMLX_P5G_PROFILE_MODE — verifying the
        // [p5h-profile] schema, not a ProfileMode ablation.
        // No --prefill-chunk-size override: production default 2048
        // exercises the real routing predicate (T1 lesson).
        let mut server = spawn_server(None, model_dir, port);
        let (stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

        if let Err(e) = wait_for_ready(port, 300) {
            shutdown_and_join(server, drainer);
            anyhow::bail!("PP={pp} T4: server not ready: {e}");
        }
        match server.try_wait() {
            Ok(Some(status)) => {
                let _ = drainer.join();
                anyhow::bail!("PP={pp} T4: ironmlx serve exited before bench: {status}");
            }
            Ok(None) => {}
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T4: try_wait failed: {e}");
            }
        }

        let out = match iron_bench_run(port, model_dir, pp) {
            Ok(o) => o,
            Err(e) => {
                shutdown_and_join(server, drainer);
                anyhow::bail!("PP={pp} T4: iron-bench spawn failed: {e}");
            }
        };

        // Shutdown FIRST so drainer EOF + join completes before we drain.
        let _ = server.kill();
        let _ = server.wait();
        let _ = drainer.join();

        if !out.status.success() {
            anyhow::bail!(
                "PP={pp} T4: iron-bench non-success: stdout={}, stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let captured = drain_stderr_into_buf(&stderr_buf);
        let records = parse_p5h_records(&captured);
        eprintln!(
            "[p5h-t4] PP={pp}: captured {} [p5h-profile] records ({} stderr bytes)",
            records.len(),
            captured.len()
        );
        per_pp.insert(pp, records);
        std::thread::sleep(INTER_PP_COOLDOWN);
    }
    Ok(per_pp)
}

/// Resolve `(span_lane, actual_lane_observed)` → `expected_to_emit`.
///
///   * `lane="inert"` ⇒ `false` always (T4.2 INERT today, both lanes).
///   * `lane="lane_a"`, observed `"lane_a"` | `"mixed"` ⇒ `true`.
///   * `lane="lane_a"`, observed `"lane_b"` or None ⇒ `false` (Lane B exempt).
///   * `lane="both"`, observed `Some(_)` ⇒ `true` (fires on either lane).
///   * `lane="both"`, observed None ⇒ `false` (no requests at this PP).
fn expected_to_emit(span_lane: &str, actual_lane_observed: Option<&'static str>) -> bool {
    match (span_lane, actual_lane_observed) {
        ("inert", _) => false,
        ("lane_a", Some("lane_a") | Some("mixed")) => true,
        ("lane_a", _) => false,
        ("both", Some(_)) => true,
        ("both", None) => false,
        (other, _) => panic!("expected_to_emit: unknown span lane {other:?}"),
    }
}

/// Aggregate per-PP records into (PP × T4 span) cells with lane-aware +
/// per-span `expected_to_emit` gating + per-span verdict semantics.
///
/// For each PP × each of the 5 T4 spans:
///   * Compute `actual_lane_observed` from per-request `routing_path`
///     aggregation (`"lane_a"` | `"lane_b"` | `"mixed"` | None on empty).
///   * `expected_to_emit` via `expected_to_emit(meta.lane, observed)`.
///   * Count matching records (for `cache_state_update` accept all
///     records; for others only `end_ns > start_ns`) + median.
///   * Only `expected_to_emit=true` cells with 0 records contribute to
///     `missing_or_invalid`. For non-`cache_state_update` spans the cell
///     also needs `median_inclusive_us > 0`.
///
/// PASS iff `missing_or_invalid.is_empty()`. `mlx_eval_barrier` cells are
/// always `expected_to_emit=false` (INERT) so they NEVER fail the verdict.
fn compute_t4_verdict(per_pp: &BTreeMap<i32, Vec<P5hProfileRecord>>) -> T4Verdict {
    let mut cells: Vec<T4Cell> = Vec::new();
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

        for span in T4_SPAN_NAMES {
            let meta = t4_span_meta(span);
            let cell_expected = expected_to_emit(meta.lane, actual_lane_observed);

            // Record filter: cache_state_update accepts inclusive_us == 0
            // (GDN cache ops are CPU-only Arc-share ~0us); other spans
            // require end_ns > start_ns to count.
            let filtered: Vec<&P5hProfileRecord> = recs
                .iter()
                .filter(|r| r.span_name == span)
                .filter(|r| {
                    if meta.accept_zero_inclusive_us {
                        r.end_ns >= r.start_ns
                    } else {
                        r.end_ns > r.start_ns
                    }
                })
                .collect();
            let count = filtered.len();
            let median = if count == 0 {
                None
            } else {
                let us: Vec<f64> = filtered
                    .iter()
                    .map(|r| (r.end_ns.saturating_sub(r.start_ns)) as f64 / 1000.0)
                    .collect();
                trimmed_median(us)
            };

            // Verdict: expected_to_emit ⇒ require record_count > 0; for
            // non-accept_zero spans additionally require median > 0.
            let valid = if !cell_expected {
                true
            } else if meta.accept_zero_inclusive_us {
                count > 0
            } else {
                count > 0 && matches!(median, Some(m) if m > 0.0)
            };
            if cell_expected && !valid {
                missing_or_invalid.push(format!(
                    "PP={pp} lane={lane} span={span} actual_lane_observed={actual_lane_observed:?}: \
                     0 valid records (record_count={count}, median_inclusive_us={median:?})",
                    lane = meta.lane,
                ));
            }

            cells.push(T4Cell {
                pp,
                span_name: span.to_string(),
                lane: meta.lane,
                actual_lane_observed,
                expected_to_emit: cell_expected,
                record_count: count,
                median_inclusive_us: median,
                span_kind: if meta.is_diagnostic {
                    "diagnostic"
                } else {
                    "tree"
                },
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
            "T4 no_data: total_records=0 across all PPs {:?}. Verify --features p5h-profile was \
             set and that the T4.1-T4.5 wrappers fire (slice_last_and_project_lm_head, \
             cache_state_update, tokenizer_encode, first_eval_amortized_cost; mlx_eval_barrier \
             is INERT today per T4.2 admit_mid_chunk path).",
            T4_PP_LIST
        )
    } else if missing_or_invalid.is_empty() {
        format!(
            "T4 pass: every expected (PP, span_name) cell has >=1 valid record. \
             total_records={total_records}, Lane-A PPs (observed)={lane_a_pps_vec:?}, \
             Lane-B PPs (observed)={lane_b_pps_vec:?} (Lane-A-only spans exempted per spec § 3 \
             T0a); expected cells={expected_cell_count}. mlx_eval_barrier exempted (INERT \
             today). Total cells={}.",
            cells.len(),
        )
    } else {
        format!(
            "T4 missing_spans: {} cells failed PASS criterion. total_records={total_records}, \
             Lane-A PPs (observed)={lane_a_pps_vec:?}, Lane-B PPs (observed)={lane_b_pps_vec:?} \
             (Lane-A-only spans exempted per spec § 3 T0a); expected cells={expected_cell_count}. \
             mlx_eval_barrier exempted (INERT today). Failed cells: {}",
            missing_or_invalid.len(),
            missing_or_invalid.join("; "),
        )
    };

    T4Verdict {
        verdict: verdict.to_string(),
        rationale,
        cells,
        missing_or_invalid,
        preheat_protocol: PREHEAT_PROTOCOL_DESC.to_string(),
        initial_cool_protocol: INITIAL_COOL_PROTOCOL_DESC.to_string(),
    }
}

/// Per-PP, per-request observed `prompt_tokens` (the server-side tokenized
/// count post chat-template render). Mirrors T1/T2/T3 — grounds the ChatML
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
#[ignore = "p5h-t4 — lm_head + MLX state + tokenization + first-eval emission sweep (Lane A) + Lane B opportunistic top-level smoke (~10-15min GPU)"]
fn t4_lm_head_mlx_state_sweep() -> anyhow::Result<()> {
    let model_dir = snapshot_dir();
    let _mlx_dir = std::env::var("MLX_DIR")
        .expect("set MLX_DIR env var pointing to MLX install prefix (e.g. $HOME/.local/mlx)");
    eprintln!("[p5h-t4] starting; model={model_dir}");

    eprintln!("[p5h-t4] preheat phase");
    preheat_to_saturation(&model_dir, PROFILE_PORT, &T4_PP_LIST)?;

    eprintln!("[p5h-t4] measurement phase (PP ∈ {T4_PP_LIST:?})");
    let per_pp = run_t4_collect_records(&model_dir, PROFILE_PORT)?;

    let verdict = compute_t4_verdict(&per_pp);
    eprintln!("[p5h-t4] {}", verdict.rationale);

    // Per-PP record counts (across all span_names) for diagnostic visibility.
    let per_pp_counts: BTreeMap<i32, usize> = per_pp.iter().map(|(k, v)| (*k, v.len())).collect();
    let total_records: usize = per_pp.values().map(|v| v.len()).sum();
    let prompt_tokens_observed = prompt_tokens_observed_per_pp(&per_pp);
    let actual_lane_per_pp_obs = actual_lane_per_pp(&per_pp);

    let out_json = serde_json::json!({
        "pp_list": T4_PP_LIST,
        "span_names": T4_SPAN_NAMES,
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
    eprintln!("[p5h-t4] JSON payload (preserved in case file-write fails):\n{json_str}");
    std::fs::write(T4_OUTPUT_PATH, &json_str)?;
    eprintln!(
        "[p5h-t4] wrote {} bytes to {T4_OUTPUT_PATH}",
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

    /// Helper: build a full Lane-A T4 emission. Includes one record for
    /// every Lane-A-expected span (slice_last + cache_state_update +
    /// tokenizer_encode + first_eval_amortized_cost). `mlx_eval_barrier` is
    /// intentionally absent (INERT today). For `cache_state_update` we
    /// emit two records — one with inclusive_us > 0 (KVCache work) and one
    /// with inclusive_us == 0 (GDN CPU-only path) — to exercise the
    /// accept_zero_inclusive_us treatment.
    fn build_full_lane_a_emission(request_id: &str, prompt_tokens: i64) -> Vec<u8> {
        build_stderr(
            request_id,
            ROUTING_PATH_LANE_A,
            prompt_tokens,
            &[
                (
                    "tokenizer_encode",
                    "http_parse_render_tokenize",
                    "tree",
                    10,
                    30,
                ),
                (
                    "slice_last_and_project_lm_head",
                    "model_prefill_forward",
                    "tree",
                    100,
                    200,
                ),
                ("cache_state_update", "kv_mask_update", "tree", 300, 350),
                ("cache_state_update", "gda_step_2c", "tree", 360, 360),
                (
                    "first_eval_amortized_cost",
                    "prefill_admitted",
                    "diagnostic",
                    400,
                    900,
                ),
            ],
        )
    }

    #[test]
    fn p5h_record_parser_extracts_all_fields_for_t4_span() {
        // Verify happy-path parse for a T4 span set including a diagnostic
        // record.
        let stderr = build_full_lane_a_emission("req-A", 128);
        let recs = parse_p5h_records(&stderr);
        assert_eq!(recs.len(), 5, "expected 5 P5h records, got {}", recs.len());

        for r in &recs {
            assert_eq!(r.request_id, "req-A");
            assert_eq!(r.routing_path, "scheduler");
            assert_eq!(r.prompt_tokens, 128);
        }

        assert_eq!(recs[0].span_name, "tokenizer_encode");
        assert_eq!(recs[0].parent_span, "http_parse_render_tokenize");
        assert_eq!(recs[0].span_kind, "tree");

        assert_eq!(recs[1].span_name, "slice_last_and_project_lm_head");
        assert_eq!(recs[1].parent_span, "model_prefill_forward");
        assert_eq!(recs[1].span_kind, "tree");
        assert_eq!(recs[1].start_ns, 100);
        assert_eq!(recs[1].end_ns, 200);

        assert_eq!(recs[2].span_name, "cache_state_update");
        assert_eq!(recs[2].parent_span, "kv_mask_update");
        assert_eq!(recs[3].span_name, "cache_state_update");
        assert_eq!(recs[3].start_ns, 360);
        assert_eq!(recs[3].end_ns, 360);

        assert_eq!(recs[4].span_name, "first_eval_amortized_cost");
        assert_eq!(recs[4].span_kind, "diagnostic");
        assert_eq!(recs[4].parent_span, "prefill_admitted");
    }

    #[test]
    fn p5h_record_parser_skips_non_p5h_lines() {
        let mut stderr = String::new();
        stderr.push_str(
            "2026-05-22T12:34:56.789012Z  INFO some::other::module: unrelated INFO line\n",
        );
        stderr.push_str("2026-05-22T12:34:56.890123Z DEBUG ironmlx::other: [other-tag] foo=bar\n");
        stderr.push_str(
            "2026-05-22T12:34:56.901234Z  INFO ironmlx::core::p5h: [p5h-profile] \
             request_id=r routing_path=scheduler prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 \
             parent_span_id=null span_name=tokenizer_encode parent_span=http_parse_render_tokenize \
             start_ns=10 end_ns=20 mode=off span_kind=tree\n",
        );
        let recs = parse_p5h_records(stderr.as_bytes());
        assert_eq!(recs.len(), 1, "only the [p5h-profile] line should match");
        assert_eq!(recs[0].request_id, "r");
        assert_eq!(recs[0].routing_path, "scheduler");
        assert_eq!(recs[0].span_name, "tokenizer_encode");
        assert_eq!(recs[0].parent_span, "http_parse_render_tokenize");
        assert_eq!(recs[0].start_ns, 10);
        assert_eq!(recs[0].end_ns, 20);
    }

    #[test]
    #[should_panic(expected = "line missing routing_path")]
    fn p5h_record_parser_fails_on_missing_routing_path() {
        let line = b"2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
            [p5h-profile] request_id=r prompt_tokens=2 seq=0 layer_idx=-1 span_id=1 \
            parent_span_id=null span_name=tokenizer_encode parent_span=http_parse_render_tokenize \
            start_ns=10 end_ns=20 mode=off span_kind=tree\n";
        let _ = parse_p5h_records(line);
    }

    #[test]
    fn lane_derivation_pure_lane_a_for_t4() {
        // Single request routed Lane A with full Lane-A T4 emission → PASS.
        // 5 cells: 4 Lane-A-expected spans + mlx_eval_barrier (INERT). The
        // 4 Lane-A-expected cells must all be expected_to_emit=true.
        let stderr = build_full_lane_a_emission("req-A", 128);
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(verdict.verdict, "pass", "rationale={}", verdict.rationale);

        // 1 PP × 5 spans = 5 cells.
        assert_eq!(verdict.cells.len(), 5);

        for c in &verdict.cells {
            assert_eq!(c.actual_lane_observed, Some("lane_a"));
            match c.span_name.as_str() {
                "mlx_eval_barrier" => {
                    assert_eq!(c.lane, "inert");
                    assert!(!c.expected_to_emit, "mlx_eval_barrier must be INERT");
                    assert_eq!(c.record_count, 0);
                    assert_eq!(c.span_kind, "tree");
                }
                "tokenizer_encode" => {
                    assert_eq!(c.lane, "both");
                    assert!(c.expected_to_emit);
                    assert!(c.record_count > 0);
                    assert!(matches!(c.median_inclusive_us, Some(m) if m > 0.0));
                    assert_eq!(c.span_kind, "tree");
                }
                "first_eval_amortized_cost" => {
                    assert_eq!(c.lane, "lane_a");
                    assert!(c.expected_to_emit);
                    assert!(c.record_count > 0);
                    assert_eq!(c.span_kind, "diagnostic");
                }
                "cache_state_update" => {
                    assert_eq!(c.lane, "lane_a");
                    assert!(c.expected_to_emit);
                    // Two records — one inclusive_us > 0, one == 0
                    assert_eq!(c.record_count, 2);
                    assert_eq!(c.span_kind, "tree");
                }
                "slice_last_and_project_lm_head" => {
                    assert_eq!(c.lane, "lane_a");
                    assert!(c.expected_to_emit);
                    assert!(c.record_count > 0);
                    assert!(matches!(c.median_inclusive_us, Some(m) if m > 0.0));
                    assert_eq!(c.span_kind, "tree");
                }
                other => panic!("unexpected span_name {other}"),
            }
        }
        assert!(verdict.missing_or_invalid.is_empty());
    }

    #[test]
    fn lane_derivation_pure_lane_b_exempts_t4_deep_spans() {
        // Lane B PP — only tokenizer_encode should fire (it's lane="both",
        // emitted in the HTTP handler pre-routing). Lane-A-only spans
        // (slice_last + cache + first_eval) must be expected_to_emit=false.
        // mlx_eval_barrier always expected_to_emit=false (INERT).
        let stderr = build_stderr(
            "req-B",
            ROUTING_PATH_LANE_B,
            4126,
            &[(
                "tokenizer_encode",
                "http_parse_render_tokenize",
                "tree",
                10,
                30,
            )],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(4096, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(
            verdict.verdict, "pass",
            "Lane B PP with tokenizer_encode only must PASS; rationale={}",
            verdict.rationale
        );

        assert_eq!(verdict.cells.len(), 5);
        for c in &verdict.cells {
            assert_eq!(c.actual_lane_observed, Some("lane_b"));
            match c.span_name.as_str() {
                "tokenizer_encode" => {
                    // lane="both" + observed lane_b ⇒ expected_to_emit=true
                    assert!(c.expected_to_emit);
                    assert_eq!(c.record_count, 1);
                }
                "mlx_eval_barrier" => {
                    assert!(!c.expected_to_emit);
                    assert_eq!(c.record_count, 0);
                }
                "slice_last_and_project_lm_head"
                | "cache_state_update"
                | "first_eval_amortized_cost" => {
                    // Lane-A-only spans, Lane B PP ⇒ exempt
                    assert!(
                        !c.expected_to_emit,
                        "Lane-A-only span {} must be exempt on Lane B PP",
                        c.span_name
                    );
                    assert_eq!(c.record_count, 0);
                }
                other => panic!("unexpected span_name {other}"),
            }
        }
        assert!(
            verdict.missing_or_invalid.is_empty(),
            "Lane B PP must not contribute to missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    fn mlx_eval_barrier_never_in_verdict_failure() {
        // Even on Lane A with 0 mlx_eval_barrier records (INERT today),
        // the verdict must PASS as long as the other 4 Lane-A spans are
        // present. Inert spans never contribute to missing_or_invalid.
        let stderr = build_full_lane_a_emission("req-A", 128);
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(verdict.verdict, "pass");

        // Verify the mlx_eval_barrier cell exists with record_count=0 and
        // expected_to_emit=false — and was NOT added to missing_or_invalid.
        let mlx_cell = verdict
            .cells
            .iter()
            .find(|c| c.span_name == "mlx_eval_barrier")
            .expect("mlx_eval_barrier cell must exist");
        assert_eq!(mlx_cell.record_count, 0);
        assert!(!mlx_cell.expected_to_emit);
        assert!(
            !verdict
                .missing_or_invalid
                .iter()
                .any(|m| m.contains("mlx_eval_barrier")),
            "mlx_eval_barrier MUST NOT appear in missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    fn cache_state_update_accepts_zero_inclusive_us() {
        // A Lane-A request whose ONLY cache_state_update record has
        // inclusive_us == 0 (e.g., GDN CPU-only Arc-share + offset bump)
        // must still produce a valid cell — record_count > 0 is enough.
        // Other Lane-A spans included to satisfy the verdict.
        let stderr = build_stderr(
            "req-A",
            ROUTING_PATH_LANE_A,
            128,
            &[
                (
                    "tokenizer_encode",
                    "http_parse_render_tokenize",
                    "tree",
                    10,
                    30,
                ),
                (
                    "slice_last_and_project_lm_head",
                    "model_prefill_forward",
                    "tree",
                    100,
                    200,
                ),
                ("cache_state_update", "gda_step_2c", "tree", 360, 360),
                ("cache_state_update", "gda_step_7", "tree", 370, 370),
                (
                    "first_eval_amortized_cost",
                    "prefill_admitted",
                    "diagnostic",
                    400,
                    900,
                ),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(
            verdict.verdict, "pass",
            "cache_state_update with inclusive_us==0 must still PASS; rationale={}",
            verdict.rationale
        );

        let cache_cell = verdict
            .cells
            .iter()
            .find(|c| c.span_name == "cache_state_update")
            .expect("cache_state_update cell must exist");
        assert_eq!(cache_cell.record_count, 2);
        // Both records have start==end so median should be Some(0.0).
        assert_eq!(cache_cell.median_inclusive_us, Some(0.0));
        assert!(cache_cell.expected_to_emit);
        assert!(verdict.missing_or_invalid.is_empty());
    }

    #[test]
    fn first_eval_amortized_cost_diagnostic_kind() {
        // Verify the cell for first_eval_amortized_cost reports
        // span_kind="diagnostic" (vs. tree for the other 4 spans).
        let stderr = build_full_lane_a_emission("req-A", 128);
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t4_verdict(&per_pp);
        let diag_cell = verdict
            .cells
            .iter()
            .find(|c| c.span_name == "first_eval_amortized_cost")
            .expect("first_eval_amortized_cost cell must exist");
        assert_eq!(diag_cell.span_kind, "diagnostic");

        for other_span in [
            "slice_last_and_project_lm_head",
            "cache_state_update",
            "tokenizer_encode",
            "mlx_eval_barrier",
        ] {
            let cell = verdict
                .cells
                .iter()
                .find(|c| c.span_name == other_span)
                .unwrap_or_else(|| panic!("{other_span} cell must exist"));
            assert_eq!(cell.span_kind, "tree", "{other_span} should be tree");
        }
    }

    #[test]
    fn verdict_flags_missing_lm_head_on_lane_a() {
        // Lane-A request missing slice_last_and_project_lm_head →
        // missing_spans verdict + the missing cell appears in
        // missing_or_invalid. mlx_eval_barrier remains exempt.
        let stderr = build_stderr(
            "req-A",
            ROUTING_PATH_LANE_A,
            128,
            &[
                (
                    "tokenizer_encode",
                    "http_parse_render_tokenize",
                    "tree",
                    10,
                    30,
                ),
                // slice_last_and_project_lm_head absent
                ("cache_state_update", "kv_mask_update", "tree", 300, 350),
                (
                    "first_eval_amortized_cost",
                    "prefill_admitted",
                    "diagnostic",
                    400,
                    900,
                ),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(verdict.verdict, "missing_spans");
        assert!(
            verdict
                .missing_or_invalid
                .iter()
                .any(|m| m.contains("slice_last_and_project_lm_head")),
            "expected slice_last_and_project_lm_head in missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
        // Only the missing span should fail — the other 3 expected spans
        // (tokenizer_encode + cache_state_update + first_eval_amortized_cost)
        // were present and must pass; mlx_eval_barrier is INERT and exempt.
        assert_eq!(
            verdict.missing_or_invalid.len(),
            1,
            "exactly one missing cell expected; got {:?}",
            verdict.missing_or_invalid
        );
    }

    #[test]
    fn lane_b_pp_missing_tokenizer_encode_fails_verdict() {
        // Lane B PP with NO records — actual_lane_observed=None, so
        // tokenizer_encode (lane="both") ends up expected_to_emit=false
        // (no observed lane to fire on). This matches the realistic
        // "no requests at this PP" case — should not fail PASS.
        //
        // To exercise the actual failure path (Lane B PP DID route a
        // request but the handler dropped tokenizer_encode), include a
        // non-tokenizer Lane B record so actual_lane_observed=Some("lane_b")
        // and tokenizer_encode becomes expected_to_emit=true but has 0
        // records.
        let stderr = build_stderr(
            "req-B",
            ROUTING_PATH_LANE_B,
            4126,
            &[
                // A non-tokenizer span on Lane B so by_req is non-empty.
                ("gs_first_token_sample_dispatch", "null", "tree", 100, 200),
            ],
        );
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(4096, recs);
        let verdict = compute_t4_verdict(&per_pp);
        assert_eq!(
            verdict.verdict, "missing_spans",
            "Lane B PP with routing but no tokenizer_encode must FAIL; rationale={}",
            verdict.rationale
        );
        assert!(
            verdict
                .missing_or_invalid
                .iter()
                .any(|m| m.contains("tokenizer_encode")),
            "expected tokenizer_encode in missing_or_invalid; got {:?}",
            verdict.missing_or_invalid
        );
        // Other Lane-A-only spans on Lane B are exempt, mlx_eval_barrier
        // is INERT — only tokenizer_encode should fail.
        assert_eq!(verdict.missing_or_invalid.len(), 1);
    }

    #[test]
    #[should_panic(expected = "mixed routing_path values")]
    fn request_with_mixed_routing_path_panics() {
        // Single request carrying both scheduler + gs_chunked records —
        // emitter regression; verdict computation must panic.
        let line1 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=scheduler prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=1 parent_span_id=null span_name=tokenizer_encode parent_span=http_parse_render_tokenize \
             start_ns=10 end_ns=30 mode=off span_kind=tree\n",
        );
        let line2 = String::from(
            "2026-05-22T12:34:56.000000Z  INFO ironmlx::core::p5h: \
             [p5h-profile] request_id=req-X routing_path=gs_chunked prompt_tokens=128 seq=0 layer_idx=-1 \
             span_id=2 parent_span_id=null span_name=tokenizer_encode parent_span=http_parse_render_tokenize \
             start_ns=300 end_ns=320 mode=off span_kind=tree\n",
        );
        let stderr = format!("{line1}{line2}").into_bytes();
        let recs = parse_p5h_records(&stderr);
        let mut per_pp = BTreeMap::new();
        per_pp.insert(128, recs);
        let _ = compute_t4_verdict(&per_pp);
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
