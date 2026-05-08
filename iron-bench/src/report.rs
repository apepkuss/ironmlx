//! Stats reduction (median + p95) + Markdown / CSV / JSON output formatters.

use crate::runner::{CellResult, RunOutcome};

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
        // TPOT excludes the first token (which is the prefill output): divide gen by (N-1) tokens.
        let tpot_div = (completion_tokens - 1.0).max(1.0);
        tpot_ms.push((gen_seconds / tpot_div) * 1000.0);
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
        tpot_ms_median: median(&mut tpot_ms),
        pp_tps_median: median(&mut pp_tps),
        e2e_s_median: median(&mut e2e_s.clone()),
        e2e_s_p95: p95(&mut e2e_s),
        finish_reason_summary,
        cached_tokens_warning: cached_warning,
    }
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

/// CSV output: one row per timed run. Stable column order.
pub fn render_csv(cells: &[CellResult]) -> String {
    let mut out = String::new();
    out.push_str(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n",
    );
    for c in cells {
        for outcome in &c.runs {
            out.push_str(&csv_row(c, outcome));
            out.push('\n');
        }
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
                serde_json::json!({
                    "target": c.target_name,
                    "pp_target": c.pp_target,
                    "tg_target": c.tg_target,
                    "run_idx": o.run_idx,
                    "ttft_ms": ttft_s * 1000.0,
                    "tg_tps": completion_tokens / gen_s,
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
                },
                server_prompt_tokens: Some(128),
                server_completion_tokens: Some(completion_tokens),
                server_cached_tokens: Some(0),
                chunk_count: completion_tokens,
                finish_reason: "stop".into(),
                content_chars: completion_tokens as usize * 4,
            },
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
        let csv = render_csv(&[cell]);
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
}
