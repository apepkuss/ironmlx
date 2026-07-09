//! Stats reduction (median + p95) + Markdown / CSV / JSON output formatters.

use crate::runner::{CellResult, RunOutcome};

const EARLY_ITL_WINDOW: usize = 8;
const AUTOTUNE_SCHEMA_VERSION: u32 = 4;

/// Aggregated per-cell statistics across N timed runs.
#[derive(Debug, Clone)]
pub struct CellStats {
    pub target_name: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub n_runs: usize,
    pub ttft_ms_median: f64,
    pub ttft_ms_p95: f64,
    pub tg_tps_median: f64,
    pub tg_tps_p95: f64,
    pub tpot_ms_median: f64,
    pub tpot_ms_p95: f64,
    pub early_itl_window: usize,
    pub early_itl_sample_count: usize,
    pub early_itl_ms_median: f64,
    pub early_itl_ms_p95: f64,
    pub pp_tps_median: f64,
    pub e2e_s_median: f64,
    pub e2e_s_p95: f64,
    pub finish_reason_summary: String,
    pub cached_tokens_warning: bool,
}

/// Reduce one cell's runs to a `CellStats`. Median + p95 over N runs.
pub fn reduce_cell(c: &CellResult) -> CellStats {
    let mut ttft_ms: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut tg_tps: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut tpot_ms: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut early_itl_ms: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut pp_tps: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut e2e_s: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut finish_reasons: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut cached_warning = false;

    for outcome in &c.runs {
        let r = &outcome.result;
        let ttft = r.timings.ttft();
        let gen = r.timings.gen_duration();
        let e2e = r.timings.e2e();

        let ttft_seconds = ttft.as_secs_f64().max(1e-9);
        let gen_seconds = gen.as_secs_f64().max(1e-9);

        let prompt_tokens = r
            .server_prompt_tokens
            .map(|n| n as f64)
            .unwrap_or(outcome.prompt_tokens_local as f64);
        let completion_tokens = r
            .server_completion_tokens
            .map(|n| n as f64)
            .unwrap_or(r.chunk_count as f64);

        ttft_ms.push(ttft_seconds * 1000.0);
        tg_tps.push(completion_tokens / gen_seconds);
        // TPOT = gen / (N-1) inter-token gaps; floor divisor to 1 when ct <= 1
        // (only prefill output, no inter-token gap exists — gen_ms acts as a sentinel).
        let tpot_div = (completion_tokens - 1.0).max(1.0);
        tpot_ms.push((gen_seconds / tpot_div) * 1000.0);
        if let Some(value) = mean_early_itl_ms(&r.timings, EARLY_ITL_WINDOW) {
            early_itl_ms.push(value);
        }
        pp_tps.push(prompt_tokens / ttft_seconds);
        e2e_s.push(e2e.as_secs_f64());

        *finish_reasons.entry(r.finish_reason.clone()).or_insert(0) += 1;
        if r.server_cached_tokens.is_some_and(|n| n > 0) {
            cached_warning = true;
        }
    }

    let finish_reason_summary = finish_reasons
        .iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(", ");

    CellStats {
        target_name: c.target_name.clone(),
        pp_target: c.pp_target,
        tg_target: c.tg_target,
        n_runs: c.runs.len(),
        ttft_ms_median: median(&mut ttft_ms.clone()),
        ttft_ms_p95: p95(&mut ttft_ms),
        tg_tps_median: median(&mut tg_tps.clone()),
        tg_tps_p95: p95(&mut tg_tps),
        tpot_ms_median: median(&mut tpot_ms.clone()),
        tpot_ms_p95: p95(&mut tpot_ms),
        early_itl_window: EARLY_ITL_WINDOW,
        early_itl_sample_count: early_itl_ms.len(),
        early_itl_ms_median: median(&mut early_itl_ms.clone()),
        early_itl_ms_p95: p95(&mut early_itl_ms),
        pp_tps_median: median(&mut pp_tps),
        e2e_s_median: median(&mut e2e_s.clone()),
        e2e_s_p95: p95(&mut e2e_s),
        finish_reason_summary,
        cached_tokens_warning: cached_warning,
    }
}

// === v2 (concurrent mode) reductions + percentile helpers ===

/// Per-cell aggregated stats for v2 concurrent mode.
#[derive(Debug, Clone)]
pub struct ConcurrentCellStats {
    pub target_name: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub concurrent: usize,
    pub wall_duration_s: f64,
    pub n_requests: usize,
    // TTFT (ms) distribution
    pub ttft_ms_p50: f64,
    pub ttft_ms_p95: f64,
    pub ttft_ms_p99: f64,
    // ITL (ms / inter-token) distribution. Per-request mean ITL = gen_duration / (completion_tokens - 1).
    pub itl_ms_p50: f64,
    pub itl_ms_p95: f64,
    pub itl_ms_p99: f64,
    pub early_itl_window: usize,
    pub early_itl_sample_count: usize,
    pub early_itl_ms_p50: f64,
    pub early_itl_ms_p95: f64,
    pub early_itl_ms_p99: f64,
    pub e2e_s_p95: f64,
    // Aggregate throughput
    pub agg_tokens_per_sec: f64,
    pub agg_req_per_sec: f64,
    // Per-worker breakdown (N entries, sorted by worker_id)
    pub per_worker_req_count: Vec<usize>,
    pub per_worker_tokens_per_sec: Vec<f64>,
    pub finish_reason_summary: String,
    pub cached_tokens_warning: bool,
}

/// Compute p (0-100) percentile by sort-and-index. Empty input returns 0.0.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Reduce one concurrent cell's outcomes to aggregated stats.
pub fn reduce_concurrent_cell(c: &crate::runner::ConcurrentCellResult) -> ConcurrentCellStats {
    let wall_duration_s = (c.cell_end - c.cell_start).as_secs_f64().max(1e-9);
    let n = c.outcomes.len();

    let mut ttft_ms: Vec<f64> = Vec::with_capacity(n);
    let mut itl_ms: Vec<f64> = Vec::with_capacity(n);
    let mut early_itl_ms: Vec<f64> = Vec::with_capacity(n);
    let mut e2e_s: Vec<f64> = Vec::with_capacity(n);
    let mut total_tokens: u64 = 0;
    let mut finish_reasons: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut cached_warning = false;
    let mut per_worker_req_count: Vec<usize> = vec![0; c.concurrent];
    let mut per_worker_tokens: Vec<u64> = vec![0; c.concurrent];

    for outcome in &c.outcomes {
        let r = &outcome.result;
        let ttft = r.timings.ttft();
        let gen = r.timings.gen_duration();
        let e2e = r.timings.e2e();

        let completion_tokens = r
            .server_completion_tokens
            .map(|n| n as f64)
            .unwrap_or(r.chunk_count as f64);

        let ttft_seconds = ttft.as_secs_f64().max(1e-9);
        let gen_seconds = gen.as_secs_f64().max(1e-9);

        ttft_ms.push(ttft_seconds * 1000.0);
        // ITL: average inter-token-latency for this request = gen_seconds / (completion - 1).
        // Floor divisor to 1.0 when completion <= 1 (matches reduce_cell TPOT semantics).
        let itl_div = (completion_tokens - 1.0).max(1.0);
        itl_ms.push((gen_seconds / itl_div) * 1000.0);
        if let Some(value) = mean_early_itl_ms(&r.timings, EARLY_ITL_WINDOW) {
            early_itl_ms.push(value);
        }
        e2e_s.push(e2e.as_secs_f64());

        total_tokens = total_tokens.saturating_add(completion_tokens as u64);

        if outcome.worker_id < c.concurrent {
            per_worker_req_count[outcome.worker_id] += 1;
            per_worker_tokens[outcome.worker_id] =
                per_worker_tokens[outcome.worker_id].saturating_add(completion_tokens as u64);
        }

        *finish_reasons.entry(r.finish_reason.clone()).or_insert(0) += 1;
        if r.server_cached_tokens.map(|n| n > 0).unwrap_or(false) {
            cached_warning = true;
        }
    }

    ttft_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    itl_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    early_itl_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    e2e_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let agg_tokens_per_sec = (total_tokens as f64) / wall_duration_s;
    let agg_req_per_sec = (n as f64) / wall_duration_s;

    let per_worker_tokens_per_sec: Vec<f64> = per_worker_tokens
        .iter()
        .map(|&t| (t as f64) / wall_duration_s)
        .collect();

    let finish_reason_summary = if finish_reasons.is_empty() {
        "(none)".to_string()
    } else {
        finish_reasons
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    ConcurrentCellStats {
        target_name: c.target_name.clone(),
        pp_target: c.pp_target,
        tg_target: c.tg_target,
        concurrent: c.concurrent,
        wall_duration_s,
        n_requests: n,
        ttft_ms_p50: percentile(&ttft_ms, 50.0),
        ttft_ms_p95: percentile(&ttft_ms, 95.0),
        ttft_ms_p99: percentile(&ttft_ms, 99.0),
        itl_ms_p50: percentile(&itl_ms, 50.0),
        itl_ms_p95: percentile(&itl_ms, 95.0),
        itl_ms_p99: percentile(&itl_ms, 99.0),
        early_itl_window: EARLY_ITL_WINDOW,
        early_itl_sample_count: early_itl_ms.len(),
        early_itl_ms_p50: percentile(&early_itl_ms, 50.0),
        early_itl_ms_p95: percentile(&early_itl_ms, 95.0),
        early_itl_ms_p99: percentile(&early_itl_ms, 99.0),
        e2e_s_p95: percentile(&e2e_s, 95.0),
        agg_tokens_per_sec,
        agg_req_per_sec,
        per_worker_req_count,
        per_worker_tokens_per_sec,
        finish_reason_summary,
        cached_tokens_warning: cached_warning,
    }
}

fn mean_early_itl_ms(timings: &crate::client::RequestTimings, max_intervals: usize) -> Option<f64> {
    let mut sum_ms = 0.0;
    let mut count = 0_usize;
    for pair in timings.token_times.windows(2).take(max_intervals) {
        sum_ms += pair[1].duration_since(pair[0]).as_secs_f64() * 1000.0;
        count += 1;
    }
    (count > 0).then_some(sum_ms / count as f64)
}

/// Median of a slice of f64. Mutates input (sorts in place). Empty input yields 0.0.
fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// 95th percentile (linear interpolation).
fn p95(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if xs.len() == 1 {
        return xs[0];
    }
    let rank = 0.95 * (xs.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(xs.len() - 1);
    let frac = rank - lo as f64;
    xs[lo] + frac * (xs[hi] - xs[lo])
}

/// Render Markdown tables, one per metric (TTFT, TG, E2E, PP, TPOT). Each table has
/// rows = targets, columns = (PP, TG) cells.
pub fn render_markdown(
    cells: &[CellResult],
    targets: &[(String, String)],
    warmup: usize,
) -> String {
    let stats: Vec<CellStats> = cells.iter().map(reduce_cell).collect();
    if stats.is_empty() {
        return String::from("(no cells run)\n");
    }

    let target_names: Vec<&str> = targets.iter().map(|(n, _)| n.as_str()).collect();
    // Distinct (pp, tg) cell columns, in the order they appear in `cells`.
    let mut cell_cols: Vec<(usize, usize)> = Vec::new();
    for s in &stats {
        let key = (s.pp_target, s.tg_target);
        if !cell_cols.contains(&key) {
            cell_cols.push(key);
        }
    }
    let n_runs = stats.first().map(|s| s.n_runs).unwrap_or(0);

    let mut out = String::new();
    out.push_str("# iron-bench results\n\n");
    out.push_str(&format!(
        "- Targets: {}\n",
        targets
            .iter()
            .map(|(n, u)| format!("{n}={u}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("- Sampler: temperature=0, top_p=1 (greedy)\n");
    out.push_str(&format!(
        "- Runs: {n_runs} measured (after {warmup} warmup), median + p95\n\n"
    ));

    // Build each table inline, using an inner function to avoid closure borrow issues.
    fn build_table(
        title: &str,
        target_names: &[&str],
        cell_cols: &[(usize, usize)],
        stats: &[CellStats],
        value_for: fn(&CellStats) -> String,
    ) -> String {
        let mut t = String::new();
        t.push_str(&format!("## {title}\n\n"));
        t.push_str("| target |");
        for (pp, tg) in cell_cols {
            t.push_str(&format!(" PP={pp} TG={tg} |"));
        }
        t.push('\n');
        t.push_str("|---|");
        for _ in cell_cols {
            t.push_str("---|");
        }
        t.push('\n');
        for name in target_names {
            t.push_str(&format!("| {name} |"));
            for (pp, tg) in cell_cols {
                let cell = stats
                    .iter()
                    .find(|s| s.target_name == *name && s.pp_target == *pp && s.tg_target == *tg);
                let s = cell.map(value_for).unwrap_or_else(|| "\u{2014}".into());
                t.push_str(&format!(" {s} |"));
            }
            t.push('\n');
        }
        t.push('\n');
        t
    }

    out.push_str(&build_table(
        "TTFT (ms)",
        &target_names,
        &cell_cols,
        &stats,
        |s| format!("{:.1} (p95 {:.1})", s.ttft_ms_median, s.ttft_ms_p95),
    ));
    out.push_str(&build_table(
        "Decode TG (tok/s)",
        &target_names,
        &cell_cols,
        &stats,
        |s| format!("{:.1} (p95 {:.1})", s.tg_tps_median, s.tg_tps_p95),
    ));
    out.push_str(&build_table(
        "E2E (s)",
        &target_names,
        &cell_cols,
        &stats,
        |s| format!("{:.3} (p95 {:.3})", s.e2e_s_median, s.e2e_s_p95),
    ));
    out.push_str(&build_table(
        "Prefill PP (tok/s, derived)",
        &target_names,
        &cell_cols,
        &stats,
        |s| format!("{:.1}", s.pp_tps_median),
    ));
    out.push_str(&build_table(
        "TPOT (ms/tok)",
        &target_names,
        &cell_cols,
        &stats,
        |s| format!("{:.2}", s.tpot_ms_median),
    ));

    let warned: Vec<String> = stats
        .iter()
        .filter(|s| s.cached_tokens_warning)
        .map(|s| format!("{} PP={} TG={}", s.target_name, s.pp_target, s.tg_target))
        .collect();
    if warned.is_empty() {
        out.push_str("\u{26a0} cached_tokens > 0 detected for: (none)\n");
    } else {
        out.push_str(&format!(
            "\u{26a0} cached_tokens > 0 detected for: {}\n",
            warned.join(", ")
        ));
    }
    out
}

pub fn render_markdown_with_prefix_cache_probe(
    cells: &[CellResult],
    targets: &[(String, String)],
    warmup: usize,
) -> String {
    let mut out = render_markdown(cells, targets, warmup);
    out.push_str("\n## Prefix Cache Probe\n\n");
    out.push_str(
        "Enabled: synthetic prompts are reused within each cell. With warmup=0, run 0 is the cold/miss candidate and later runs are warm-hit candidates. With warmup>0, measured runs are warm-hit candidates.\n",
    );
    out
}

pub fn render_markdown_concurrent(
    cells: &[crate::runner::ConcurrentCellResult],
    targets: &[(String, String)],
    concurrent: usize,
    duration: u64,
    warmup_duration: u64,
) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    writeln!(out, "# iron-bench v2 (concurrent) results\n").unwrap();
    writeln!(
        out,
        "- concurrent workers per cell: **{concurrent}**\n- timed duration: **{duration}s**\n- warmup duration: **{warmup_duration}s**\n",
    )
    .unwrap();
    writeln!(out, "Targets:").unwrap();
    for (name, url) in targets {
        writeln!(out, "- `{name}` → `{url}`").unwrap();
    }
    writeln!(out).unwrap();

    let stats: Vec<ConcurrentCellStats> = cells.iter().map(reduce_concurrent_cell).collect();
    if stats.is_empty() {
        writeln!(out, "_(no cells)_").unwrap();
        return out;
    }

    // Aggregate table: one row per cell.
    writeln!(out, "## Per-cell aggregate metrics\n").unwrap();
    writeln!(
        out,
        "| target | PP | TG | N req | p50 TTFT (ms) | p95 TTFT (ms) | p99 TTFT (ms) | p50 ITL (ms) | p95 ITL (ms) | p99 ITL (ms) | p95 first-{} ITL (ms) | tokens/s | req/s |",
        EARLY_ITL_WINDOW
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for s in &stats {
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.2} |",
            s.target_name,
            s.pp_target,
            s.tg_target,
            s.n_requests,
            s.ttft_ms_p50,
            s.ttft_ms_p95,
            s.ttft_ms_p99,
            s.itl_ms_p50,
            s.itl_ms_p95,
            s.itl_ms_p99,
            s.early_itl_ms_p95,
            s.agg_tokens_per_sec,
            s.agg_req_per_sec,
        )
        .unwrap();
    }

    // Per-worker breakdown.
    writeln!(out, "\n## Per-worker breakdown\n").unwrap();
    for s in &stats {
        writeln!(
            out,
            "### {} | PP={} TG={} | {} workers\n",
            s.target_name, s.pp_target, s.tg_target, s.concurrent
        )
        .unwrap();
        writeln!(out, "| worker | req count | tokens/s |").unwrap();
        writeln!(out, "| --- | --- | --- |").unwrap();
        for w in 0..s.concurrent {
            writeln!(
                out,
                "| {} | {} | {:.1} |",
                w, s.per_worker_req_count[w], s.per_worker_tokens_per_sec[w]
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    // Notes.
    writeln!(out, "## Notes\n").unwrap();
    for s in &stats {
        writeln!(
            out,
            "- `{}` PP={} TG={}: finish_reasons={}{}",
            s.target_name,
            s.pp_target,
            s.tg_target,
            s.finish_reason_summary,
            if s.cached_tokens_warning {
                " \u{26a0} cached_tokens > 0 (PP measurement may be unreliable)"
            } else {
                ""
            }
        )
        .unwrap();
    }

    out
}

pub fn render_markdown_concurrent_with_prefix_cache_probe(
    cells: &[crate::runner::ConcurrentCellResult],
    targets: &[(String, String)],
    concurrent: usize,
    duration: u64,
    warmup_duration: u64,
) -> String {
    let mut out = render_markdown_concurrent(cells, targets, concurrent, duration, warmup_duration);
    out.push_str("\n## Prefix Cache Probe\n\n");
    out.push_str(
        "Enabled: all workers reuse the same synthetic prompt within each cell, so concurrent TTFT reflects shared-prefix contention and warm-hit behavior when the server prefix cache is enabled.\n",
    );
    out
}

/// CSV output: one row per timed run. Stable column order.
///
/// When `capture_request_id` is true, an extra `request_id` column is appended
/// to both the header and each row (empty string if the per-row
/// `RequestResult.request_id` is `None`).
///
/// When `capture_run_timestamps` is true, `run_start_unix_ns` and
/// `run_end_unix_ns` columns are appended (empty string when None). Both
/// flags compose: when both are on, `request_id` precedes the timestamp
/// columns. Downstream parsers MUST use header names (csv.DictReader), not
/// fixed positions. Flag-off-both output is byte-identical to the base schema.
pub fn render_csv(
    cells: &[CellResult],
    capture_request_id: bool,
    capture_run_timestamps: bool,
) -> String {
    let mut out = String::new();
    let mut header = String::from(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason",
    );
    if capture_request_id {
        header.push_str(",request_id");
    }
    if capture_run_timestamps {
        header.push_str(",run_start_unix_ns,run_end_unix_ns");
    }
    header.push('\n');
    out.push_str(&header);
    for c in cells {
        for outcome in &c.runs {
            out.push_str(&csv_row(c, outcome));
            if capture_request_id {
                out.push(',');
                out.push_str(outcome.result.request_id.as_deref().unwrap_or(""));
            }
            if capture_run_timestamps {
                out.push(',');
                if let Some(v) = outcome.run_start_unix_ns {
                    out.push_str(&v.to_string());
                }
                out.push(',');
                if let Some(v) = outcome.run_end_unix_ns {
                    out.push_str(&v.to_string());
                }
            }
            out.push('\n');
        }
    }
    out
}

pub fn render_csv_with_prefix_cache_probe(
    cells: &[CellResult],
    capture_request_id: bool,
    capture_run_timestamps: bool,
    warmup: usize,
) -> String {
    let base = render_csv(cells, capture_request_id, capture_run_timestamps);
    let mut out = String::new();
    for (line_idx, line) in base.lines().enumerate() {
        if line_idx == 0 {
            out.push_str(line);
            out.push_str(",prefix_cache_probe,prefix_cache_probe_phase\n");
            continue;
        }
        let run_idx = line
            .split(',')
            .nth(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        out.push_str(line);
        out.push_str(",true,");
        out.push_str(prefix_cache_probe_phase(warmup, run_idx));
        out.push('\n');
    }
    out
}

pub fn render_csv_concurrent(cells: &[crate::runner::ConcurrentCellResult]) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    // Header: one row per request (worker_id-level detail). Pandas-friendly.
    writeln!(
        out,
        "target,pp,tg,concurrent,worker_id,request_idx_in_worker,ttft_ms,gen_secs,early_itl_ms,e2e_s,completion_tokens,prompt_tokens,finish_reason"
    )
    .unwrap();
    for c in cells {
        // Track per-worker request index for CSV row ordering.
        let mut per_worker_idx: Vec<usize> = vec![0; c.concurrent];
        for outcome in &c.outcomes {
            let r = &outcome.result;
            let ttft_ms = r.timings.ttft().as_secs_f64() * 1000.0;
            let gen_s = r.timings.gen_duration().as_secs_f64();
            let early_itl_ms = mean_early_itl_ms(&r.timings, EARLY_ITL_WINDOW);
            let e2e_s = r.timings.e2e().as_secs_f64();
            let completion_tokens = r
                .server_completion_tokens
                .map(|n| n as f64)
                .unwrap_or(r.chunk_count as f64);
            let prompt_tokens = r
                .server_prompt_tokens
                .map(|n| n as f64)
                .unwrap_or(outcome.prompt_tokens_local as f64);
            let req_idx = if outcome.worker_id < c.concurrent {
                let i = per_worker_idx[outcome.worker_id];
                per_worker_idx[outcome.worker_id] += 1;
                i
            } else {
                0
            };
            writeln!(
                out,
                "{},{},{},{},{},{},{:.3},{:.6},{},{:.6},{:.0},{:.0},{}",
                c.target_name,
                c.pp_target,
                c.tg_target,
                c.concurrent,
                outcome.worker_id,
                req_idx,
                ttft_ms,
                gen_s,
                early_itl_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_default(),
                e2e_s,
                completion_tokens,
                prompt_tokens,
                r.finish_reason,
            )
            .unwrap();
        }
    }
    out
}

pub fn render_csv_concurrent_with_prefix_cache_probe(
    cells: &[crate::runner::ConcurrentCellResult],
) -> String {
    let base = render_csv_concurrent(cells);
    let mut out = String::new();
    for (line_idx, line) in base.lines().enumerate() {
        out.push_str(line);
        if line_idx == 0 {
            out.push_str(",prefix_cache_probe");
        } else {
            out.push_str(",true");
        }
        out.push('\n');
    }
    out
}

fn csv_row(c: &CellResult, o: &RunOutcome) -> String {
    let r = &o.result;
    let ttft = r.timings.ttft();
    let gen = r.timings.gen_duration();
    let ttft_s = ttft.as_secs_f64().max(1e-9);
    let gen_s = gen.as_secs_f64().max(1e-9);
    let prompt_tokens = r
        .server_prompt_tokens
        .map(|n| n as f64)
        .unwrap_or(o.prompt_tokens_local as f64);
    let completion_tokens = r
        .server_completion_tokens
        .map(|n| n as f64)
        .unwrap_or(r.chunk_count as f64);
    let tpot_div = (completion_tokens - 1.0).max(1.0);

    format!(
        "{name},{pp},{tg},{idx},{ttft_ms:.3},{tg_tps:.3},{tpot_ms:.3},{pp_tps:.3},{e2e_s:.6},{p_local},{p_server},{c_server},{cached},{finish}",
        name = c.target_name,
        pp = c.pp_target,
        tg = c.tg_target,
        idx = o.run_idx,
        ttft_ms = ttft_s * 1000.0,
        tg_tps = completion_tokens / gen_s,
        tpot_ms = (gen_s / tpot_div) * 1000.0,
        pp_tps = prompt_tokens / ttft_s,
        e2e_s = r.timings.e2e().as_secs_f64(),
        p_local = o.prompt_tokens_local,
        p_server = r
            .server_prompt_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        c_server = r
            .server_completion_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        cached = r
            .server_cached_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        finish = r.finish_reason,
    )
}

/// JSON output: nested object with `metadata`, `stats`, and `raw_runs`.
pub fn render_json(cells: &[CellResult], targets: &[(String, String)], warmup: usize) -> String {
    let stats: Vec<CellStats> = cells.iter().map(reduce_cell).collect();
    let mut metadata = serde_json::Map::new();
    metadata.insert("warmup".into(), serde_json::Value::from(warmup));
    metadata.insert(
        "runs_measured".into(),
        serde_json::Value::from(stats.first().map(|s| s.n_runs).unwrap_or(0)),
    );
    let mut sampler = serde_json::Map::new();
    sampler.insert("temperature".into(), serde_json::Value::from(0.0_f64));
    sampler.insert("top_p".into(), serde_json::Value::from(1.0_f64));
    metadata.insert("sampler".into(), serde_json::Value::Object(sampler));
    metadata.insert(
        "targets".into(),
        serde_json::Value::Array(
            targets
                .iter()
                .map(|(n, u)| serde_json::json!({"name": n, "url": u}))
                .collect(),
        ),
    );

    let stats_json: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "target": s.target_name,
                "pp_target": s.pp_target,
                "tg_target": s.tg_target,
                "n_runs": s.n_runs,
                "ttft_ms_median": s.ttft_ms_median,
                "ttft_ms_p95": s.ttft_ms_p95,
                "tg_tps_median": s.tg_tps_median,
                "tg_tps_p95": s.tg_tps_p95,
                "tpot_ms_median": s.tpot_ms_median,
                "early_itl_ms": {
                    "first_n": s.early_itl_window,
                    "sample_count": s.early_itl_sample_count,
                    "median": s.early_itl_ms_median,
                    "p95": s.early_itl_ms_p95,
                },
                "pp_tps_median": s.pp_tps_median,
                "e2e_s_median": s.e2e_s_median,
                "e2e_s_p95": s.e2e_s_p95,
                "finish_reason_summary": s.finish_reason_summary,
                "cached_tokens_warning": s.cached_tokens_warning,
            })
        })
        .collect();

    let raw_runs: Vec<serde_json::Value> = cells
        .iter()
        .flat_map(|c| {
            c.runs.iter().map(move |o| {
                let r = &o.result;
                let ttft_s = r.timings.ttft().as_secs_f64().max(1e-9);
                let gen_s = r.timings.gen_duration().as_secs_f64().max(1e-9);
                let prompt_tokens = r
                    .server_prompt_tokens
                    .map(|n| n as f64)
                    .unwrap_or(o.prompt_tokens_local as f64);
                let completion_tokens = r
                    .server_completion_tokens
                    .map(|n| n as f64)
                    .unwrap_or(r.chunk_count as f64);
                let tpot_div = (completion_tokens - 1.0).max(1.0);
                let early_itl_ms = mean_early_itl_ms(&r.timings, EARLY_ITL_WINDOW);
                serde_json::json!({
                    "target": c.target_name,
                    "pp_target": c.pp_target,
                    "tg_target": c.tg_target,
                    "run_idx": o.run_idx,
                    "ttft_ms": ttft_s * 1000.0,
                    "tg_tps": completion_tokens / gen_s,
                    "tpot_ms": (gen_s / tpot_div) * 1000.0,
                    "early_itl_ms": early_itl_ms,
                    "pp_tps": prompt_tokens / ttft_s,
                    "e2e_s": r.timings.e2e().as_secs_f64(),
                    "prompt_tokens_local": o.prompt_tokens_local,
                    "prompt_tokens_server": r.server_prompt_tokens,
                    "completion_tokens_server": r.server_completion_tokens,
                    "cached_tokens": r.server_cached_tokens,
                    "finish_reason": r.finish_reason,
                })
            })
        })
        .collect();

    let root = serde_json::json!({
        "metadata": metadata,
        "stats": stats_json,
        "raw_runs": raw_runs,
    });
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".into())
}

pub fn render_json_with_prefix_cache_probe(
    cells: &[CellResult],
    targets: &[(String, String)],
    warmup: usize,
) -> String {
    let raw = render_json(cells, targets, warmup);
    let mut root: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(metadata) = root
        .get_mut("metadata")
        .and_then(serde_json::Value::as_object_mut)
    {
        metadata.insert("prefix_cache_probe".into(), serde_json::Value::from(true));
    }
    if let Some(raw_runs) = root
        .get_mut("raw_runs")
        .and_then(serde_json::Value::as_array_mut)
    {
        for row in raw_runs {
            let run_idx = row
                .get("run_idx")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            row["prefix_cache_probe_phase"] =
                serde_json::Value::from(prefix_cache_probe_phase(warmup, run_idx));
        }
    }
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".into())
}

pub fn render_json_concurrent(
    cells: &[crate::runner::ConcurrentCellResult],
    targets: &[(String, String)],
    concurrent: usize,
    duration: u64,
    warmup_duration: u64,
) -> String {
    let stats: Vec<ConcurrentCellStats> = cells.iter().map(reduce_concurrent_cell).collect();

    let stats_json: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "target_name": s.target_name,
                "pp_target": s.pp_target,
                "tg_target": s.tg_target,
                "concurrent": s.concurrent,
                "wall_duration_s": s.wall_duration_s,
                "n_requests": s.n_requests,
                "ttft_ms": {
                    "p50": s.ttft_ms_p50,
                    "p95": s.ttft_ms_p95,
                    "p99": s.ttft_ms_p99,
                },
                "itl_ms": {
                    "p50": s.itl_ms_p50,
                    "p95": s.itl_ms_p95,
                    "p99": s.itl_ms_p99,
                },
                "early_itl_ms": {
                    "first_n": s.early_itl_window,
                    "sample_count": s.early_itl_sample_count,
                    "p50": s.early_itl_ms_p50,
                    "p95": s.early_itl_ms_p95,
                    "p99": s.early_itl_ms_p99,
                },
                "aggregate": {
                    "tokens_per_sec": s.agg_tokens_per_sec,
                    "req_per_sec": s.agg_req_per_sec,
                },
                "e2e_s_p95": s.e2e_s_p95,
                "per_worker": {
                    "req_count": s.per_worker_req_count,
                    "tokens_per_sec": s.per_worker_tokens_per_sec,
                },
                "finish_reason_summary": s.finish_reason_summary,
                "cached_tokens_warning": s.cached_tokens_warning,
            })
        })
        .collect();
    let raw_runs = concurrent_raw_runs_json(cells);

    let payload = serde_json::json!({
        "mode": "concurrent",
        "concurrent": concurrent,
        "duration_s": duration,
        "warmup_duration_s": warmup_duration,
        "targets": targets.iter().map(|(n, u)| serde_json::json!({"name": n, "url": u})).collect::<Vec<_>>(),
        "cells": stats_json,
        "raw_runs": raw_runs,
    });

    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}

fn concurrent_raw_runs_json(
    cells: &[crate::runner::ConcurrentCellResult],
) -> Vec<serde_json::Value> {
    cells
        .iter()
        .flat_map(|c| {
            let mut per_worker_idx: Vec<usize> = vec![0; c.concurrent];
            c.outcomes.iter().map(move |outcome| {
                let r = &outcome.result;
                let ttft_s = r.timings.ttft().as_secs_f64().max(1e-9);
                let gen_s = r.timings.gen_duration().as_secs_f64().max(1e-9);
                let e2e_s = r.timings.e2e().as_secs_f64();
                let completion_tokens = r.server_completion_tokens.unwrap_or(r.chunk_count);
                let prompt_tokens = r
                    .server_prompt_tokens
                    .map(|n| n as usize)
                    .unwrap_or(outcome.prompt_tokens_local);
                let completion_tokens_f = completion_tokens as f64;
                let itl_div = (completion_tokens_f - 1.0).max(1.0);
                let request_idx_in_worker = if outcome.worker_id < c.concurrent {
                    let idx = per_worker_idx[outcome.worker_id];
                    per_worker_idx[outcome.worker_id] += 1;
                    idx
                } else {
                    0
                };

                serde_json::json!({
                    "target": c.target_name,
                    "pp_target": c.pp_target,
                    "tg_target": c.tg_target,
                    "concurrent": c.concurrent,
                    "worker_id": outcome.worker_id,
                    "request_idx_in_worker": request_idx_in_worker,
                    "ttft_ms": ttft_s * 1000.0,
                    "gen_secs": gen_s,
                    "e2e_s": e2e_s,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "prompt_tokens_local": outcome.prompt_tokens_local,
                    "prompt_tokens_server": r.server_prompt_tokens,
                    "completion_tokens_server": r.server_completion_tokens,
                    "cached_tokens": r.server_cached_tokens,
                    "chunk_count": r.chunk_count,
                    "finish_reason": r.finish_reason,
                    "request_id": r.request_id,
                    "itl_ms": (gen_s / itl_div) * 1000.0,
                    "early_itl_ms": mean_early_itl_ms(&r.timings, EARLY_ITL_WINDOW),
                })
            })
        })
        .collect()
}

pub fn render_json_concurrent_with_prefix_cache_probe(
    cells: &[crate::runner::ConcurrentCellResult],
    targets: &[(String, String)],
    concurrent: usize,
    duration: u64,
    warmup_duration: u64,
) -> String {
    let raw = render_json_concurrent(cells, targets, concurrent, duration, warmup_duration);
    let mut root: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}));
    root["prefix_cache_probe"] = serde_json::Value::from(true);
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".to_string())
}

fn prefix_cache_probe_phase(warmup: usize, run_idx: usize) -> &'static str {
    if warmup > 0 || run_idx > 0 {
        "warm_hit_candidate"
    } else {
        "cold_or_miss_candidate"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutotuneProfileConfig {
    pub b_max: usize,
    pub prefill_chunk_size: usize,
    pub admission_deadline_ms: u64,
    pub admission_queue_max: usize,
    pub max_cache_cap: usize,
    pub decode_cadence_mid_chunk_cap: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutotuneExportOptions {
    pub model_name: String,
    pub hardware_label: String,
    pub config: AutotuneProfileConfig,
    pub memory_budget_ok: bool,
}

pub fn render_autotune_json_sequential(
    cells: &[CellResult],
    options: &AutotuneExportOptions,
) -> String {
    let measurements: Vec<serde_json::Value> = cells
        .iter()
        .map(|cell| {
            let stats = reduce_cell(cell);
            autotune_measurement_json(
                options,
                stats.pp_target,
                stats.tg_target,
                1,
                stats.ttft_ms_p95,
                stats.tpot_ms_p95,
                stats.early_itl_ms_p95,
                stats.e2e_s_p95,
                stats.tg_tps_median,
                stats.cached_tokens_warning,
            )
        })
        .collect();
    render_autotune_root(options, measurements)
}

pub fn render_autotune_json_concurrent(
    cells: &[crate::runner::ConcurrentCellResult],
    options: &AutotuneExportOptions,
) -> String {
    let measurements: Vec<serde_json::Value> = cells
        .iter()
        .map(|cell| {
            let stats = reduce_concurrent_cell(cell);
            autotune_measurement_json(
                options,
                stats.pp_target,
                stats.tg_target,
                stats.concurrent,
                stats.ttft_ms_p95,
                stats.itl_ms_p95,
                stats.early_itl_ms_p95,
                stats.e2e_s_p95,
                stats.agg_tokens_per_sec,
                stats.cached_tokens_warning,
            )
        })
        .collect();
    render_autotune_root(options, measurements)
}

#[allow(clippy::too_many_arguments)]
fn autotune_measurement_json(
    options: &AutotuneExportOptions,
    prompt_len: usize,
    max_new_tokens: usize,
    concurrency: usize,
    ttft_ms_p95: f64,
    itl_ms_p95: f64,
    early_itl_ms_p95: f64,
    e2e_s_p95: f64,
    tokens_per_sec: f64,
    cached_tokens_warning: bool,
) -> serde_json::Value {
    serde_json::json!({
        "config": autotune_config_json(options.config),
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "concurrency": concurrency,
        "ttft_ms_p95": ttft_ms_p95,
        "itl_ms_p95": itl_ms_p95,
        "early_itl_ms_p95": early_itl_ms_p95,
        "e2e_s_p95": e2e_s_p95,
        "tokens_per_sec": tokens_per_sec,
        "memory_budget_ok": options.memory_budget_ok,
        "cached_tokens_warning": cached_tokens_warning,
    })
}

fn render_autotune_root(
    options: &AutotuneExportOptions,
    measurements: Vec<serde_json::Value>,
) -> String {
    let payload = serde_json::json!({
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "model_name": options.model_name,
        "hardware_label": options.hardware_label,
        "measurements": measurements,
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}

fn autotune_config_json(config: AutotuneProfileConfig) -> serde_json::Value {
    serde_json::json!({
        "b_max": config.b_max,
        "prefill_chunk_size": config.prefill_chunk_size,
        "admission_deadline_ms": config.admission_deadline_ms,
        "admission_queue_max": config.admission_queue_max,
        "max_cache_cap": config.max_cache_cap,
        "decode_cadence_mid_chunk_cap": config.decode_cadence_mid_chunk_cap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{RequestResult, RequestTimings};
    use std::time::{Duration, Instant};

    fn fake_outcome(
        run_idx: usize,
        ttft_ms: f64,
        gen_ms: f64,
        completion_tokens: u32,
    ) -> RunOutcome {
        let start = Instant::now();
        let first_token = start + Duration::from_millis(ttft_ms as u64);
        let end = first_token + Duration::from_millis(gen_ms as u64);
        RunOutcome {
            run_idx,
            prompt_tokens_local: 128,
            result: RequestResult {
                timings: RequestTimings {
                    start,
                    first_token: Some(first_token),
                    end,
                    token_times: Vec::new(),
                },
                server_prompt_tokens: Some(128),
                server_completion_tokens: Some(completion_tokens),
                server_cached_tokens: Some(0),
                chunk_count: completion_tokens,
                finish_reason: "stop".into(),
                content_chars: completion_tokens as usize * 4,
                request_id: None, // default-off mirrors production default
            },
            run_start_unix_ns: None,
            run_end_unix_ns: None,
        }
    }

    #[test]
    fn stats_median_and_p95_with_outlier() {
        // 5 runs: TTFT = 40, 42, 45, 50, 200 ms. Median = 45, p95 = ~170 (interpolated).
        let cell = CellResult {
            target_name: "t".into(),
            target_url: "u".into(),
            pp_target: 128,
            tg_target: 128,
            runs: vec![
                fake_outcome(0, 40.0, 800.0, 100),
                fake_outcome(1, 42.0, 800.0, 100),
                fake_outcome(2, 45.0, 800.0, 100),
                fake_outcome(3, 50.0, 800.0, 100),
                fake_outcome(4, 200.0, 800.0, 100),
            ],
        };
        let s = reduce_cell(&cell);
        assert_eq!(s.n_runs, 5);
        assert!(
            (s.ttft_ms_median - 45.0).abs() < 0.5,
            "median expected ~45, got {}",
            s.ttft_ms_median
        );
        // p95 of [40,42,45,50,200] with linear interp at rank 0.95*4=3.8:
        //   xs[3] + 0.8*(xs[4]-xs[3]) = 50 + 0.8*(200-50) = 50 + 120 = 170
        assert!(
            (s.ttft_ms_p95 - 170.0).abs() < 0.5,
            "p95 expected ~170, got {}",
            s.ttft_ms_p95
        );
    }

    #[test]
    fn csv_columns_stable() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        let csv = render_csv(&[cell], false, false);
        let header = csv.lines().next().expect("header line");
        assert_eq!(
            header,
            "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason"
        );
        let body = csv.lines().nth(1).expect("data line");
        assert!(
            body.starts_with("ironmlx,128,64,0,"),
            "unexpected row: {body}"
        );
        assert!(
            body.ends_with(",stop"),
            "expected to end with finish_reason=stop, got: {body}"
        );
    }

    #[test]
    fn csv_columns_stable_default_off() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        let csv = render_csv(&[cell], false, false);

        // GOLDEN: deterministic full-string match. fake_outcome uses fixed
        // Instant deltas so every numeric column is reproducible.
        let expected = "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\nironmlx,128,64,0,50.000,128.000,7.937,2560.000,0.550000,128,128,64,0,stop\n";
        assert_eq!(
            csv, expected,
            "default-off CSV must be byte-identical to the pre-flag golden \
             — drift in any column/value/order fails this gate"
        );
    }

    #[test]
    fn csv_prefix_cache_probe_appends_phase_columns() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![
                fake_outcome(0, 50.0, 500.0, 64),
                fake_outcome(1, 10.0, 500.0, 64),
            ],
        };
        let csv = render_csv_with_prefix_cache_probe(&[cell], false, false, 0);
        let header = csv.lines().next().expect("header line");
        assert!(
            header.ends_with(",prefix_cache_probe,prefix_cache_probe_phase"),
            "header should append prefix-cache probe columns, got: {header}"
        );
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert!(
            rows[0].ends_with(",true,cold_or_miss_candidate"),
            "run 0 without warmup should be labeled cold/miss candidate, got: {}",
            rows[0]
        );
        assert!(
            rows[1].ends_with(",true,warm_hit_candidate"),
            "run 1 should be labeled warm-hit candidate, got: {}",
            rows[1]
        );
    }

    #[test]
    fn json_prefix_cache_probe_records_metadata_and_run_phase() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        let raw = render_json_with_prefix_cache_probe(
            &[cell],
            &[("ironmlx".to_string(), "http://localhost:8080".to_string())],
            1,
        );
        let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(json["metadata"]["prefix_cache_probe"], true);
        assert_eq!(json["metadata"]["warmup"], 1);
        assert_eq!(
            json["raw_runs"][0]["prefix_cache_probe_phase"],
            "warm_hit_candidate"
        );
    }

    #[test]
    fn csv_columns_stable_capture_on() {
        let mut cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        cell.runs[0].result.request_id = Some("deadbeef-1234".into());

        let csv = render_csv(&[cell], true, false);

        let expected = "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,request_id\nironmlx,128,64,0,50.000,128.000,7.937,2560.000,0.550000,128,128,64,0,stop,deadbeef-1234\n";
        assert_eq!(csv, expected, "capture-on CSV byte-identity check");
    }

    #[test]
    fn csv_capture_on_with_none_request_id() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)], // request_id = None
        };
        let csv = render_csv(&[cell], true, false);
        let body = csv.lines().nth(1).expect("data line");
        assert!(
            body.ends_with(",stop,"),
            "capture-on row with None request_id must end with `,stop,` (empty trailing field), got: {body}"
        );
    }

    #[test]
    fn csv_includes_run_timestamps_when_enabled() {
        let mut cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        cell.runs[0].run_start_unix_ns = Some(1_000);
        cell.runs[0].run_end_unix_ns = Some(2_000);
        let csv = render_csv(&[cell], false, true);
        let header = csv.lines().next().expect("header line");
        assert!(
            header.ends_with(",run_start_unix_ns,run_end_unix_ns"),
            "header should end with timestamp columns, got: {header}"
        );
        assert!(
            csv.contains(",1000,2000\n"),
            "row should embed timestamp values, got: {csv}"
        );
    }

    #[test]
    fn csv_includes_both_when_request_id_and_timestamps_enabled() {
        let mut cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        cell.runs[0].result.request_id = Some("deadbeef".into());
        cell.runs[0].run_start_unix_ns = Some(1_000);
        cell.runs[0].run_end_unix_ns = Some(2_000);
        let csv = render_csv(&[cell], true, true);
        let header = csv.lines().next().expect("header line");
        assert!(
            header.ends_with(",request_id,run_start_unix_ns,run_end_unix_ns"),
            "header should append request_id before timestamp columns, got: {header}"
        );
        let row = csv.lines().nth(1).expect("data line");
        assert!(
            row.ends_with(",deadbeef,1000,2000"),
            "row should end with request_id then timestamps, got: {row}"
        );
    }

    #[test]
    fn csv_timestamps_empty_when_none() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)], // both timestamps = None
        };
        let csv = render_csv(&[cell], false, true);
        let row = csv.lines().nth(1).expect("data line");
        assert!(
            row.ends_with(",stop,,"),
            "row with None timestamps should end `,stop,,` (two empty trailing fields), got: {row}"
        );
    }

    #[test]
    fn percentile_basic() {
        // v = [1.0, 2.0, ..., 10.0], len = 10.
        // p50: idx = round(0.50 * 9) = round(4.5) = 5 → sorted[5] = 6.0
        // p95: idx = round(0.95 * 9) = round(8.55) = 9 → sorted[9] = 10.0
        // p99: idx = round(0.99 * 9) = round(8.91) = 9 → sorted[9] = 10.0
        // p0:  idx = round(0.00 * 9) = 0 → sorted[0] = 1.0
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 50.0), 6.0);
        assert_eq!(percentile(&v, 95.0), 10.0);
        assert_eq!(percentile(&v, 99.0), 10.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
    }

    #[test]
    fn percentile_edge_cases() {
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 99.0), 42.0);
        let same = vec![7.0_f64; 100];
        assert_eq!(percentile(&same, 50.0), 7.0);
        assert_eq!(percentile(&same, 99.0), 7.0);
    }

    #[test]
    fn render_markdown_concurrent_smoke() {
        // Synthetic ConcurrentCellResult with 0 outcomes — verifies the
        // formatter doesn't crash on empty input and emits the expected
        // section headers.
        let now = std::time::Instant::now();
        let cell = crate::runner::ConcurrentCellResult {
            target_name: "mock".into(),
            target_url: "http://localhost:0".into(),
            pp_target: 128,
            tg_target: 64,
            concurrent: 2,
            cell_start: now,
            cell_end: now + std::time::Duration::from_secs(1),
            outcomes: Vec::new(),
        };
        let targets = vec![("mock".into(), "http://localhost:0".into())];
        let md = render_markdown_concurrent(&[cell], &targets, 2, 1, 0);
        assert!(md.contains("iron-bench v2 (concurrent)"));
        assert!(md.contains("Per-cell aggregate metrics"));
        assert!(md.contains("Per-worker breakdown"));
        assert!(md.contains("p50 TTFT"));
        assert!(md.contains("tokens/s"));
        assert!(md.contains("## Notes"), "Notes section should be present");
    }

    #[test]
    fn render_json_concurrent_exports_e2e_tail_latency() {
        use crate::runner::{ConcurrentCellResult, RequestOutcome};

        let cell_start = Instant::now();
        let request_start = Instant::now();
        let first_token = request_start + Duration::from_millis(10);
        let end = first_token + Duration::from_millis(90);
        let outcome = RequestOutcome {
            worker_id: 0,
            prompt_tokens_local: 128,
            result: RequestResult {
                timings: RequestTimings {
                    start: request_start,
                    first_token: Some(first_token),
                    end,
                    token_times: vec![first_token, end],
                },
                server_prompt_tokens: Some(128),
                server_completion_tokens: Some(2),
                server_cached_tokens: Some(0),
                chunk_count: 2,
                finish_reason: "stop".to_string(),
                content_chars: 8,
                request_id: None,
            },
        };
        let cell = ConcurrentCellResult {
            target_name: "mock".into(),
            target_url: "http://localhost:0".into(),
            pp_target: 128,
            tg_target: 2,
            concurrent: 1,
            cell_start,
            cell_end: cell_start + Duration::from_secs(1),
            outcomes: vec![outcome],
        };
        let targets = vec![("mock".into(), "http://localhost:0".into())];

        let raw = render_json_concurrent(&[cell], &targets, 1, 1, 0);
        let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let value = json["cells"][0]["e2e_s_p95"]
            .as_f64()
            .expect("cell should export e2e_s_p95");

        assert!((value - 0.1).abs() < 1e-6);
    }

    #[test]
    fn render_json_concurrent_exports_raw_request_runs() {
        use crate::runner::{ConcurrentCellResult, RequestOutcome};

        let cell_start = Instant::now();
        let req0_start = Instant::now();
        let req0_first = req0_start + Duration::from_millis(10);
        let req0_end = req0_first + Duration::from_millis(90);
        let req1_start = Instant::now();
        let req1_first = req1_start + Duration::from_millis(20);
        let req1_end = req1_first + Duration::from_millis(180);

        let cell = ConcurrentCellResult {
            target_name: "mock".into(),
            target_url: "http://localhost:0".into(),
            pp_target: 2048,
            tg_target: 512,
            concurrent: 2,
            cell_start,
            cell_end: cell_start + Duration::from_secs(1),
            outcomes: vec![
                RequestOutcome {
                    worker_id: 0,
                    prompt_tokens_local: 2048,
                    result: RequestResult {
                        timings: RequestTimings {
                            start: req0_start,
                            first_token: Some(req0_first),
                            end: req0_end,
                            token_times: vec![
                                req0_first,
                                req0_first + Duration::from_millis(30),
                                req0_end,
                            ],
                        },
                        server_prompt_tokens: Some(2048),
                        server_completion_tokens: Some(3),
                        server_cached_tokens: Some(0),
                        chunk_count: 3,
                        finish_reason: "stop".to_string(),
                        content_chars: 12,
                        request_id: Some("req-0".to_string()),
                    },
                },
                RequestOutcome {
                    worker_id: 1,
                    prompt_tokens_local: 2049,
                    result: RequestResult {
                        timings: RequestTimings {
                            start: req1_start,
                            first_token: Some(req1_first),
                            end: req1_end,
                            token_times: vec![req1_first, req1_end],
                        },
                        server_prompt_tokens: Some(2050),
                        server_completion_tokens: Some(2),
                        server_cached_tokens: Some(7),
                        chunk_count: 2,
                        finish_reason: "length".to_string(),
                        content_chars: 8,
                        request_id: Some("req-1".to_string()),
                    },
                },
            ],
        };
        let targets = vec![("mock".into(), "http://localhost:0".into())];

        let raw = render_json_concurrent(&[cell], &targets, 2, 1, 0);
        let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let raw_runs = json["raw_runs"]
            .as_array()
            .expect("concurrent JSON should export raw_runs");

        assert_eq!(raw_runs.len(), 2);
        assert_eq!(raw_runs[0]["target"], "mock");
        assert_eq!(raw_runs[0]["pp_target"], 2048);
        assert_eq!(raw_runs[0]["tg_target"], 512);
        assert_eq!(raw_runs[0]["concurrent"], 2);
        assert_eq!(raw_runs[0]["worker_id"], 0);
        assert_eq!(raw_runs[0]["request_idx_in_worker"], 0);
        assert_eq!(raw_runs[0]["prompt_tokens_local"], 2048);
        assert_eq!(raw_runs[0]["prompt_tokens_server"], 2048);
        assert_eq!(raw_runs[0]["completion_tokens_server"], 3);
        assert_eq!(raw_runs[0]["completion_tokens"], 3);
        assert_eq!(raw_runs[0]["cached_tokens"], 0);
        assert_eq!(raw_runs[0]["chunk_count"], 3);
        assert_eq!(raw_runs[0]["finish_reason"], "stop");
        assert_eq!(raw_runs[0]["request_id"], "req-0");
        assert!((raw_runs[0]["ttft_ms"].as_f64().unwrap() - 10.0).abs() < 1e-6);
        assert!((raw_runs[0]["e2e_s"].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!(raw_runs[0]["early_itl_ms"].as_f64().is_some());

        assert_eq!(raw_runs[1]["worker_id"], 1);
        assert_eq!(raw_runs[1]["request_idx_in_worker"], 0);
        assert_eq!(raw_runs[1]["prompt_tokens_local"], 2049);
        assert_eq!(raw_runs[1]["prompt_tokens_server"], 2050);
        assert_eq!(raw_runs[1]["completion_tokens_server"], 2);
        assert_eq!(raw_runs[1]["cached_tokens"], 7);
        assert_eq!(raw_runs[1]["finish_reason"], "length");
        assert_eq!(raw_runs[1]["request_id"], "req-1");
    }

    #[test]
    fn reduce_concurrent_cell_aggregates_correctly() {
        use crate::runner::{ConcurrentCellResult, RequestOutcome};

        // Synthetic ConcurrentCellResult: 2 workers, 3 requests each (6 total).
        // Each request:
        //   - ttft = 10ms
        //   - gen_duration = 90ms (5 completion tokens => ITL = 90/(5-1) = 22.5ms)
        //   - completion_tokens = 5
        //   - prompt_tokens = 16
        // Wall duration of cell = 1 second exactly (cell_end - cell_start).

        let cell_start = Instant::now();
        let cell_end = cell_start + Duration::from_secs(1);

        let mut outcomes: Vec<RequestOutcome> = Vec::new();
        for worker_id in 0..2_usize {
            for _ in 0..3_usize {
                let start = Instant::now();
                let first_token = start + Duration::from_millis(10);
                let end = first_token + Duration::from_millis(90);
                let timings = RequestTimings {
                    start,
                    first_token: Some(first_token),
                    end,
                    token_times: vec![
                        first_token,
                        first_token + Duration::from_millis(20),
                        first_token + Duration::from_millis(40),
                        first_token + Duration::from_millis(65),
                        first_token + Duration::from_millis(90),
                    ],
                };
                let result = RequestResult {
                    timings,
                    server_prompt_tokens: Some(16),
                    server_completion_tokens: Some(5),
                    server_cached_tokens: None,
                    chunk_count: 5,
                    finish_reason: "stop".to_string(),
                    content_chars: 0,
                    request_id: None,
                };
                outcomes.push(RequestOutcome {
                    worker_id,
                    prompt_tokens_local: 16,
                    result,
                });
            }
        }

        let cell = ConcurrentCellResult {
            target_name: "synthetic".into(),
            target_url: "http://0".into(),
            pp_target: 16,
            tg_target: 5,
            concurrent: 2,
            cell_start,
            cell_end,
            outcomes,
        };

        let stats = reduce_concurrent_cell(&cell);

        // Totals.
        assert_eq!(stats.n_requests, 6, "6 total requests across 2 workers");
        assert_eq!(stats.per_worker_req_count, vec![3, 3]);

        // Aggregate throughput: 6 reqs × 5 tokens = 30 tokens / 1 sec = 30 tok/s.
        assert!(
            (stats.agg_tokens_per_sec - 30.0).abs() < 1e-6,
            "agg_tokens_per_sec should be 30.0; got {}",
            stats.agg_tokens_per_sec,
        );
        assert!(
            (stats.agg_req_per_sec - 6.0).abs() < 1e-6,
            "agg_req_per_sec should be 6.0; got {}",
            stats.agg_req_per_sec,
        );

        // Per-worker tokens/s: 3 reqs × 5 tokens = 15 tokens / 1 sec = 15 tok/s.
        assert!(
            (stats.per_worker_tokens_per_sec[0] - 15.0).abs() < 1e-6
                && (stats.per_worker_tokens_per_sec[1] - 15.0).abs() < 1e-6,
            "per-worker tokens/s should be 15.0 each; got {:?}",
            stats.per_worker_tokens_per_sec,
        );

        // TTFT and ITL: all requests are identical (10ms / 22.5ms), so all
        // percentiles should be those values.
        assert!(
            (stats.ttft_ms_p50 - 10.0).abs() < 1e-3,
            "ttft p50 should be ~10ms; got {}",
            stats.ttft_ms_p50,
        );
        assert!(
            (stats.itl_ms_p99 - 22.5).abs() < 1e-3,
            "itl p99 should be ~22.5ms; got {}",
            stats.itl_ms_p99,
        );
        assert_eq!(stats.early_itl_window, 8);
        assert_eq!(stats.early_itl_sample_count, 6);
        assert!(
            (stats.early_itl_ms_p95 - 22.5).abs() < 1e-3,
            "early_itl p95 should be request-level first-N mean ~22.5ms; got {}",
            stats.early_itl_ms_p95,
        );

        // finish_reasons should be {"stop": 6}.
        assert!(
            stats.finish_reason_summary.contains("stop=6"),
            "finish_reason_summary should contain 'stop=6'; got: {}",
            stats.finish_reason_summary,
        );

        assert!(!stats.cached_tokens_warning);
    }

    #[test]
    fn autotune_json_sequential_exports_scheduler_calibration_schema() {
        let mut cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 2048,
            tg_target: 128,
            runs: vec![fake_outcome(0, 120.0, 1280.0, 128)],
        };
        cell.runs[0].result.server_cached_tokens = Some(7);

        let options = AutotuneExportOptions {
            model_name: "GLM-4.7-flash-4bit".to_string(),
            hardware_label: "m3-max".to_string(),
            config: AutotuneProfileConfig {
                b_max: 2,
                prefill_chunk_size: 1024,
                admission_deadline_ms: 5,
                admission_queue_max: 32,
                max_cache_cap: 32768,
                decode_cadence_mid_chunk_cap: 256,
            },
            memory_budget_ok: true,
        };

        let raw = render_autotune_json_sequential(&[cell], &options);
        let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(json["schema_version"], 4);
        assert_eq!(json["model_name"], "GLM-4.7-flash-4bit");
        assert_eq!(json["hardware_label"], "m3-max");

        let measurements = json["measurements"].as_array().expect("measurements array");
        assert_eq!(measurements.len(), 1);
        let row = &measurements[0];
        assert_eq!(row["config"]["b_max"], 2);
        assert_eq!(row["config"]["prefill_chunk_size"], 1024);
        assert_eq!(row["config"]["admission_deadline_ms"], 5);
        assert_eq!(row["config"]["admission_queue_max"], 32);
        assert_eq!(row["config"]["max_cache_cap"], 32768);
        assert_eq!(row["config"]["decode_cadence_mid_chunk_cap"], 256);
        assert_eq!(row["prompt_len"], 2048);
        assert_eq!(row["max_new_tokens"], 128);
        assert_eq!(row["concurrency"], 1);
        assert_eq!(row["memory_budget_ok"], true);
        assert_eq!(row["cached_tokens_warning"], true);
        assert!(row["ttft_ms_p95"].as_f64().unwrap() > 0.0);
        assert!(row["itl_ms_p95"].as_f64().unwrap() > 0.0);
        assert!(row["early_itl_ms_p95"].as_f64().is_some());
        assert!(row["e2e_s_p95"].as_f64().unwrap() > 0.0);
        assert!(row["tokens_per_sec"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn autotune_json_concurrent_exports_scheduler_calibration_schema() {
        use crate::runner::{ConcurrentCellResult, RequestOutcome};

        let cell_start = Instant::now();
        let cell_end = cell_start + Duration::from_secs(1);
        let mut outcomes = Vec::new();
        for worker_id in 0..2_usize {
            for _ in 0..3_usize {
                let start = Instant::now();
                let first_token = start + Duration::from_millis(10);
                let end = first_token + Duration::from_millis(90);
                outcomes.push(RequestOutcome {
                    worker_id,
                    prompt_tokens_local: 2048,
                    result: RequestResult {
                        timings: RequestTimings {
                            start,
                            first_token: Some(first_token),
                            end,
                            token_times: vec![
                                first_token,
                                first_token + Duration::from_millis(20),
                                first_token + Duration::from_millis(40),
                                first_token + Duration::from_millis(65),
                                first_token + Duration::from_millis(90),
                            ],
                        },
                        server_prompt_tokens: Some(2048),
                        server_completion_tokens: Some(5),
                        server_cached_tokens: Some(0),
                        chunk_count: 5,
                        finish_reason: "stop".to_string(),
                        content_chars: 20,
                        request_id: None,
                    },
                });
            }
        }

        let cell = ConcurrentCellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 2048,
            tg_target: 128,
            concurrent: 2,
            cell_start,
            cell_end,
            outcomes,
        };

        let options = AutotuneExportOptions {
            model_name: "GLM-4.7-flash-4bit".to_string(),
            hardware_label: "m3-max".to_string(),
            config: AutotuneProfileConfig {
                b_max: 2,
                prefill_chunk_size: 1024,
                admission_deadline_ms: 5,
                admission_queue_max: 32,
                max_cache_cap: 32768,
                decode_cadence_mid_chunk_cap: 256,
            },
            memory_budget_ok: false,
        };

        let raw = render_autotune_json_concurrent(&[cell], &options);
        let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let row = &json["measurements"][0];

        assert_eq!(row["prompt_len"], 2048);
        assert_eq!(row["max_new_tokens"], 128);
        assert_eq!(row["concurrency"], 2);
        assert_eq!(row["memory_budget_ok"], false);
        assert_eq!(row["cached_tokens_warning"], false);
        assert_eq!(row["ttft_ms_p95"].as_f64().unwrap(), 10.0);
        assert_eq!(row["itl_ms_p95"].as_f64().unwrap(), 22.5);
        assert_eq!(row["early_itl_ms_p95"].as_f64().unwrap(), 22.5);
        assert!((row["e2e_s_p95"].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert_eq!(row["tokens_per_sec"].as_f64().unwrap(), 30.0);
    }
}
