# iron-bench v2 — Concurrent benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add closed-loop concurrent benchmarking (`--concurrent N --duration S`) to iron-bench while preserving v1 sequential mode (`--runs N --warmup W`) as the default. Unblocks ironmlx 3c-3 multi-concurrent-request aggregate throughput comparison.

**Architecture:** N tokio worker tasks per (target, prompt_len) cell, each fires HTTP request → awaits response → repeats until wall-clock deadline. Per-request outcomes aggregated to sorted percentile data (p50/p95/p99 TTFT + ITL + aggregate tokens/s + per-worker breakdown). v1 path unchanged; v2 dispatched in `main.rs` on `--concurrent` flag presence.

**Tech Stack:** Rust + tokio multi-thread runtime + `Arc<reqwest::Client>` shared connection pool + `Arc<Tokenizer>` shared. No new crates beyond what v1 uses (futures, serde, clap, anyhow already present).

---

## Standing Per-Task Hygiene Gate

After each task's implementation step but BEFORE the commit step, run from `/Volumes/Dev/cxx-mlx`:

```bash
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release -p iron-bench
```

All three must be clean. If `fmt --check` fails, run `cargo +nightly fmt --all` to format and re-check. If clippy emits a warning you don't know how to fix, **STOP and ask Boss** — don't paper over with `#[allow]`.

Note: iron-bench does NOT need `MLX_DIR=$HOME/.local/mlx` (it has no mlx-sys dependency). Cargo invocations for iron-bench can omit MLX_DIR.

Each task ends with a single git commit. Commit subject prefix: `feat(iron-bench):` / `test(iron-bench):` / `docs(iron-bench):` / `fix(iron-bench):`.

The `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer is established repo convention — every prior commit on this branch uses it. Include verbatim in every commit body.

---

## File Structure

| File | Task | Role |
| --- | --- | --- |
| `iron-bench/src/main.rs` | 1 | CLI flags: `--concurrent N`, `--duration S`, `--warmup-duration S`; mode dispatch (sequential vs concurrent) |
| `iron-bench/src/runner.rs` | 2 | `run_cell_concurrent` + `ConcurrentCellResult` + `RequestOutcome` structs |
| `iron-bench/src/report.rs` | 3 | percentile helpers + `render_markdown_concurrent` / `render_csv_concurrent` / `render_json_concurrent`; main.rs dispatch on cell type |
| `iron-bench/tests/concurrent_smoke.rs` | 4 | NEW — integration smoke test against an in-process mock SSE server |
| `iron-bench/README.md` | 4 | Update "Single-request only" line + add v2 concurrent mode section |

---

### Task 1: New CLI flags + mode dispatch in main.rs

**Files:**
- Modify: `iron-bench/src/main.rs` (~80 line delta)

- [ ] **Step 1: Add new CLI flags to the `Args` struct**

Open `iron-bench/src/main.rs`. Find the `#[derive(Parser, Debug)] struct Args` block (around line 18). Add 3 new fields. The full updated `Args` struct should look like (PRESERVE all existing fields; only ADD the 3 marked `// NEW`):

```rust
#[derive(Parser, Debug)]
#[command(
    name = "iron-bench",
    about = "Head-to-head HTTP benchmark for OpenAI-compatible LLM endpoints",
    version
)]
struct Args {
    /// Target endpoints. Repeat for multiple targets.
    /// Format: `name=URL` (e.g., `--target ironmlx=http://localhost:8080`).
    #[arg(long, value_parser = parse_target, required = true, num_args = 1..)]
    target: Vec<(String, String)>,

    /// Path to model dir containing `tokenizer.json` (used for prompt synthesis only).
    #[arg(long)]
    model_dir: PathBuf,

    /// Model name to send in the `model` field of each JSON request.
    #[arg(long, default_value = "qwen3.5-4b")]
    model: String,

    /// Prompt token lengths to test (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![128_usize, 512, 2048])]
    prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// (v1 sequential mode) Timed runs per cell. Mutually exclusive with `--concurrent`.
    #[arg(long, default_value_t = 5, conflicts_with = "concurrent")]
    runs: usize,

    /// (v1 sequential mode) Warmup runs per cell (excluded from stats).
    /// Mutually exclusive with `--concurrent`.
    #[arg(long, default_value_t = 1, conflicts_with = "concurrent")]
    warmup: usize,

    /// (v2 concurrent mode) Number of concurrent workers per cell. Each worker
    /// fires request -> awaits response -> repeats until `--duration` deadline.
    /// When absent, runs in v1 sequential mode.
    #[arg(long)]                                          // NEW
    concurrent: Option<usize>,                            // NEW

    /// (v2 concurrent mode) Wall-clock duration per cell (seconds).
    /// Only meaningful when `--concurrent` is set; ignored otherwise.
    #[arg(long, default_value_t = 30)]                    // NEW
    duration: u64,                                        // NEW

    /// (v2 concurrent mode) Wall-clock warmup duration per cell (seconds).
    /// Only meaningful when `--concurrent` is set; ignored otherwise.
    #[arg(long, default_value_t = 5)]                     // NEW
    warmup_duration: u64,                                 // NEW

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,

    /// HTTP request timeout (seconds).
    #[arg(long, default_value_t = 300)]
    timeout: u64,
}
```

The `conflicts_with = "concurrent"` annotations on `runs` and `warmup` make clap reject `--runs 5 --concurrent 4` at parse time.

- [ ] **Step 2: Update startup banner**

Find the `eprintln!("iron-bench: ...")` line (~line 84 of current main.rs). Branch the banner text based on mode:

```rust
    match args.concurrent {
        None => eprintln!(
            "iron-bench v1 (sequential): {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
            args.target.len(),
            args.prompt_len,
            args.max_tokens,
            args.runs,
            args.warmup,
        ),
        Some(n) => eprintln!(
            "iron-bench v2 (concurrent): {} target(s), prompt_len={:?}, max_tokens={}, concurrent={}, duration={}s, warmup_duration={}s",
            args.target.len(),
            args.prompt_len,
            args.max_tokens,
            n,
            args.duration,
            args.warmup_duration,
        ),
    }
```

- [ ] **Step 3: Replace the cell-iteration loop with mode dispatch**

Find the current cell-iteration loop (~lines 105-120 of current main.rs):

```rust
    let mut cells: Vec<runner::CellResult> = Vec::new();
    for pp in &args.prompt_len {
        for (target_name, target_url) in &args.target {
            let cell = runner::run_cell(
                &client,
                target_name,
                target_url,
                &args.model,
                *pp,
                args.max_tokens,
                args.warmup,
                args.runs,
                &tokenizer,
            )
            .await?;
            cells.push(cell);
        }
    }
```

Replace with mode-aware dispatch using an enum:

```rust
    // Cells are heterogeneous between v1 (Sequential) and v2 (Concurrent) modes.
    // Use the unified enum so the existing `for cell in cells { render }` loop
    // in main.rs stays clean.
    enum AnyCell {
        Sequential(runner::CellResult),
        Concurrent(runner::ConcurrentCellResult),
    }

    let mut cells: Vec<AnyCell> = Vec::new();

    match args.concurrent {
        None => {
            // v1 sequential path
            for pp in &args.prompt_len {
                for (target_name, target_url) in &args.target {
                    let cell = runner::run_cell(
                        &client,
                        target_name,
                        target_url,
                        &args.model,
                        *pp,
                        args.max_tokens,
                        args.warmup,
                        args.runs,
                        &tokenizer,
                    )
                    .await?;
                    cells.push(AnyCell::Sequential(cell));
                }
            }
        }
        Some(concurrent) => {
            // v2 concurrent path: share Client + Tokenizer via Arc.
            let client_arc = std::sync::Arc::new(client);
            let tokenizer_arc = std::sync::Arc::new(tokenizer);
            for pp in &args.prompt_len {
                for (target_name, target_url) in &args.target {
                    let cell = runner::run_cell_concurrent(
                        client_arc.clone(),
                        target_name,
                        target_url,
                        &args.model,
                        *pp,
                        args.max_tokens,
                        std::time::Duration::from_secs(args.warmup_duration),
                        std::time::Duration::from_secs(args.duration),
                        concurrent,
                        tokenizer_arc.clone(),
                    )
                    .await?;
                    cells.push(AnyCell::Concurrent(cell));
                }
            }
        }
    }
```

(`runner::run_cell_concurrent` and `runner::ConcurrentCellResult` don't exist yet — Task 2 adds them. This file won't compile at end of Task 1's edits; it builds at end of Task 2.)

- [ ] **Step 4: Replace the output rendering call with mode-aware dispatch**

Find the current rendering call (~lines 122-127):

```rust
    let out = match args.format {
        OutputFormat::Markdown => report::render_markdown(&cells, &args.target, args.warmup),
        OutputFormat::Csv => report::render_csv(&cells),
        OutputFormat::Json => report::render_json(&cells, &args.target, args.warmup),
    };
    println!("{out}");
```

Replace with:

```rust
    // Split cells back into sequential vs concurrent slices for the existing
    // (v1) renderers + the new (v2) renderers. Per-cell mode mixing is
    // impossible (CLI dispatches uniformly), so all cells share one mode.
    let out = match args.concurrent {
        None => {
            let seq_cells: Vec<runner::CellResult> = cells
                .into_iter()
                .filter_map(|c| match c {
                    AnyCell::Sequential(s) => Some(s),
                    AnyCell::Concurrent(_) => None,
                })
                .collect();
            match args.format {
                OutputFormat::Markdown => report::render_markdown(&seq_cells, &args.target, args.warmup),
                OutputFormat::Csv => report::render_csv(&seq_cells),
                OutputFormat::Json => report::render_json(&seq_cells, &args.target, args.warmup),
            }
        }
        Some(concurrent) => {
            let conc_cells: Vec<runner::ConcurrentCellResult> = cells
                .into_iter()
                .filter_map(|c| match c {
                    AnyCell::Sequential(_) => None,
                    AnyCell::Concurrent(c) => Some(c),
                })
                .collect();
            match args.format {
                OutputFormat::Markdown => report::render_markdown_concurrent(&conc_cells, &args.target, concurrent, args.duration, args.warmup_duration),
                OutputFormat::Csv => report::render_csv_concurrent(&conc_cells),
                OutputFormat::Json => report::render_json_concurrent(&conc_cells, &args.target, concurrent, args.duration, args.warmup_duration),
            }
        }
    };
    println!("{out}");
```

(The `render_markdown_concurrent` / `render_csv_concurrent` / `render_json_concurrent` functions don't exist yet — Task 3 adds them. main.rs still won't compile.)

- [ ] **Step 5: Build to verify clap parser changes only (will fail on runner/report symbols)**

```bash
cargo build --release -p iron-bench 2>&1 | head -30
```

Expected: COMPILE FAIL with errors referencing `run_cell_concurrent`, `ConcurrentCellResult`, `render_markdown_concurrent`, etc. — these are added in Tasks 2 + 3. This confirms the clap derive parses correctly.

If errors mention parsing/derive issues with the new flag annotations, fix those immediately (e.g., `conflicts_with` syntax error). If only undefined-symbol errors remain → ready for Task 2.

- [ ] **Step 6: Defer the commit until Task 2 lands**

Since main.rs alone doesn't compile, do NOT commit Task 1 in isolation. Task 1 and Task 2 commit together at the end of Task 2 (so the working tree has a passing build at every commit). Save unstaged changes (no `git add` yet); proceed to Task 2.

If the changes are too much to hold uncommitted, an alternative is to land Task 1 with the new functions stubbed in runner.rs / report.rs (each function returns `unimplemented!()`). The build would pass but lib tests would fail. Pick whichever is cleaner for the implementer; the simpler path is "uncommitted until Task 2".

---

### Task 2: `run_cell_concurrent` + new structs in runner.rs

**Files:**
- Modify: `iron-bench/src/runner.rs` (~100 line addition)

- [ ] **Step 1: Add new structs `ConcurrentCellResult` + `RequestOutcome`**

Open `iron-bench/src/runner.rs`. After the existing `CellResult` + `RunOutcome` structs (~line 30), add:

```rust
/// (v2 concurrent mode) One worker iteration's outcome.
#[derive(Debug)]
pub struct RequestOutcome {
    pub worker_id: usize,
    pub prompt_tokens_local: usize,
    pub result: RequestResult,
}

/// (v2 concurrent mode) Per-cell result: N workers ran for `duration` seconds,
/// produced `outcomes` requests in aggregate.
#[derive(Debug)]
pub struct ConcurrentCellResult {
    pub target_name: String,
    #[allow(dead_code)]
    pub target_url: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub concurrent: usize,
    /// Wall-clock start of the timed phase (after warmup). Used to compute
    /// aggregate tokens/s and req/s precisely.
    pub cell_start: std::time::Instant,
    /// Wall-clock end of the timed phase (after all workers joined).
    pub cell_end: std::time::Instant,
    pub outcomes: Vec<RequestOutcome>,
}
```

- [ ] **Step 2: Add `run_cell_concurrent` function**

Below the existing `run_cell` function in `runner.rs`, add:

```rust
/// (v2 concurrent mode) Drive a single cell with `concurrent` workers for
/// `warmup_duration` (discarded) then `duration` (timed) wall-clock seconds.
///
/// Each worker independently fires `run_chat_completion` -> awaits response ->
/// repeats with a fresh nonce, until the deadline. Outcomes from all workers
/// are flattened into `ConcurrentCellResult.outcomes` for the reporter.
///
/// `client` and `tokenizer` are shared via `Arc` to avoid per-worker resource
/// duplication (HTTP connection pool reuse + tokenizer load amortization).
#[allow(clippy::too_many_arguments)]
pub async fn run_cell_concurrent(
    client: std::sync::Arc<reqwest::Client>,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup_duration: std::time::Duration,
    duration: std::time::Duration,
    concurrent: usize,
    tokenizer: std::sync::Arc<Tokenizer>,
) -> Result<ConcurrentCellResult> {
    use std::time::Instant;

    eprintln!(
        "[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: warmup {warmup_duration:?} ..."
    );

    // === 1. Warmup phase: N workers run for warmup_duration, discard outcomes. ===
    if !warmup_duration.is_zero() {
        let warmup_deadline = Instant::now() + warmup_duration;
        let mut warmup_handles = Vec::with_capacity(concurrent);
        for worker_id in 0..concurrent {
            let client_w = client.clone();
            let tokenizer_w = tokenizer.clone();
            let url = target_url.to_string();
            let model_w = model.to_string();
            warmup_handles.push(tokio::spawn(async move {
                let mut nonce = nonce_seed() ^ (worker_id as u64);
                while Instant::now() < warmup_deadline {
                    let (prompt, _) = crate::prompt::synthesize_prompt(&tokenizer_w, pp, nonce)?;
                    let _ = crate::client::run_chat_completion(&client_w, &url, &model_w, &prompt, tg).await?;
                    nonce = nonce.wrapping_add(1);
                }
                Ok::<(), anyhow::Error>(())
            }));
        }
        for h in warmup_handles {
            h.await??;
        }
    }

    eprintln!(
        "[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: timed {duration:?} ..."
    );

    // === 2. Timed phase: N workers, duration, collect outcomes. ===
    let cell_start = Instant::now();
    let timed_deadline = cell_start + duration;
    let mut timed_handles = Vec::with_capacity(concurrent);
    for worker_id in 0..concurrent {
        let client_w = client.clone();
        let tokenizer_w = tokenizer.clone();
        let url = target_url.to_string();
        let model_w = model.to_string();
        timed_handles.push(tokio::spawn(async move {
            let mut outcomes: Vec<RequestOutcome> = Vec::new();
            // Distinct nonce space per worker: high 16 bits = worker_id,
            // low 48 bits = wrapping counter. No collisions across workers
            // until each worker has fired 2^48 requests (effectively never).
            let mut nonce = nonce_seed() ^ ((worker_id as u64) << 48);
            while Instant::now() < timed_deadline {
                let (prompt, prompt_local) =
                    crate::prompt::synthesize_prompt(&tokenizer_w, pp, nonce)?;
                let result =
                    crate::client::run_chat_completion(&client_w, &url, &model_w, &prompt, tg).await?;
                outcomes.push(RequestOutcome {
                    worker_id,
                    prompt_tokens_local: prompt_local,
                    result,
                });
                nonce = nonce.wrapping_add(1);
            }
            Ok::<Vec<RequestOutcome>, anyhow::Error>(outcomes)
        }));
    }

    let mut all_outcomes: Vec<RequestOutcome> = Vec::new();
    for h in timed_handles {
        all_outcomes.extend(h.await??);
    }
    let cell_end = Instant::now();

    eprintln!(
        "[{target_name}] PP={pp} TG={tg} concurrent={concurrent}: {} requests completed",
        all_outcomes.len()
    );

    Ok(ConcurrentCellResult {
        target_name: target_name.into(),
        target_url: target_url.into(),
        pp_target: pp,
        tg_target: tg,
        concurrent,
        cell_start,
        cell_end,
        outcomes: all_outcomes,
    })
}
```

`nonce_seed()` is the existing private helper at the bottom of `runner.rs` (returns `u64` from `SystemTime::now().duration_since(UNIX_EPOCH).as_nanos() as u64`). Keep it as-is.

- [ ] **Step 3: Build (still expecting failure due to report.rs symbols)**

```bash
cargo build --release -p iron-bench 2>&1 | tail -20
```

Expected: COMPILE FAIL with errors referencing `render_markdown_concurrent`, `render_csv_concurrent`, `render_json_concurrent`. main.rs + runner.rs changes are now compatible; report.rs is the only remaining gap.

- [ ] **Step 4: Defer commit to Task 3 (paired build green)**

Same rationale as Task 1: don't commit runner.rs alone since main.rs references symbols that need Task 3. Tasks 1 + 2 + 3 all land together at the end of Task 3.

If preferring earlier commits, stub the 3 new `render_*_concurrent` functions in `report.rs` to return placeholder strings — that lets Tasks 1 + 2 + early-3 land independently. Choose whichever is cleaner.

---

### Task 3: Percentile helpers + 3 concurrent formatters in report.rs

**Files:**
- Modify: `iron-bench/src/report.rs` (~200 line addition)

- [ ] **Step 1: Add percentile helpers + per-cell stats struct**

Open `iron-bench/src/report.rs`. After the existing `CellStats` struct + `reduce_cell` function, add a new section for v2 stats:

```rust
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
        agg_tokens_per_sec,
        agg_req_per_sec,
        per_worker_req_count,
        per_worker_tokens_per_sec,
        finish_reason_summary,
        cached_tokens_warning: cached_warning,
    }
}
```

- [ ] **Step 2: Add `render_markdown_concurrent`**

After the existing `render_markdown` function, add:

```rust
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
        "| target | PP | TG | N req | p50 TTFT (ms) | p95 TTFT (ms) | p99 TTFT (ms) | p50 ITL (ms) | p95 ITL (ms) | p99 ITL (ms) | tokens/s | req/s |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for s in &stats {
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} | {:.2} | {:.1} | {:.2} |",
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
                " ⚠ cached_tokens > 0 (PP measurement may be unreliable)"
            } else {
                ""
            }
        )
        .unwrap();
    }

    out
}
```

- [ ] **Step 3: Add `render_csv_concurrent`**

After `render_csv`:

```rust
pub fn render_csv_concurrent(cells: &[crate::runner::ConcurrentCellResult]) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    // Header: one row per request (worker_id-level detail). Pandas-friendly.
    writeln!(
        out,
        "target,pp,tg,concurrent,worker_id,request_idx_in_worker,ttft_ms,gen_secs,e2e_s,completion_tokens,prompt_tokens,finish_reason"
    )
    .unwrap();
    for c in cells {
        // Track per-worker request index for CSV row ordering.
        let mut per_worker_idx: Vec<usize> = vec![0; c.concurrent];
        for outcome in &c.outcomes {
            let r = &outcome.result;
            let ttft_ms = r.timings.ttft().as_secs_f64() * 1000.0;
            let gen_s = r.timings.gen_duration().as_secs_f64();
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
                "{},{},{},{},{},{},{:.3},{:.6},{:.6},{:.0},{:.0},{}",
                c.target_name,
                c.pp_target,
                c.tg_target,
                c.concurrent,
                outcome.worker_id,
                req_idx,
                ttft_ms,
                gen_s,
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
```

- [ ] **Step 4: Add `render_json_concurrent`**

After `render_json`:

```rust
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
                "aggregate": {
                    "tokens_per_sec": s.agg_tokens_per_sec,
                    "req_per_sec": s.agg_req_per_sec,
                },
                "per_worker": {
                    "req_count": s.per_worker_req_count,
                    "tokens_per_sec": s.per_worker_tokens_per_sec,
                },
                "finish_reason_summary": s.finish_reason_summary,
                "cached_tokens_warning": s.cached_tokens_warning,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "mode": "concurrent",
        "concurrent": concurrent,
        "duration_s": duration,
        "warmup_duration_s": warmup_duration,
        "targets": targets.iter().map(|(n, u)| serde_json::json!({"name": n, "url": u})).collect::<Vec<_>>(),
        "cells": stats_json,
    });

    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}
```

- [ ] **Step 5: Add lib unit tests for percentile helpers + reducer**

Append to the `#[cfg(test)] mod tests` block at the bottom of `report.rs` (or add one if not present):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 50.0), 5.0); // index 4-5 round
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
    }
}
```

If a `mod tests` block already exists from v1, append these 3 tests inside it (don't create a duplicate `mod tests`). Read the existing test module first.

- [ ] **Step 6: Build + lib tests**

```bash
cargo build --release -p iron-bench
cargo test -p iron-bench --release --lib
```

Expected: BOTH PASS. main.rs + runner.rs + report.rs now all consistent. Lib tests include the 3 new percentile/reducer/markdown tests + any existing v1 tests.

- [ ] **Step 7: Hygiene gate**

Run the Standing Per-Task Hygiene Gate (top of doc). All clean.

- [ ] **Step 8: Commit Tasks 1+2+3 together**

```bash
git add iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/src/report.rs
git commit -m "$(cat <<'EOF'
feat(iron-bench): v2 closed-loop concurrent mode — CLI + runner + report

iron-bench v2 ships closed-loop concurrent benchmarking. v1 sequential
mode preserved as default; v2 opt-in via --concurrent N --duration S
--warmup-duration S.

CLI (main.rs):
  - New flags: --concurrent N (Option<usize>; opt-in), --duration S
    (default 30s), --warmup-duration S (default 5s).
  - --runs / --warmup gated mutually-exclusive with --concurrent via
    clap conflicts_with.
  - main.rs dispatches to v1 or v2 runner based on --concurrent presence,
    then renders via mode-specific formatter.

Runner (runner.rs):
  - New: RequestOutcome { worker_id, prompt_tokens_local, result }
  - New: ConcurrentCellResult { target_name, pp_target, tg_target,
    concurrent, cell_start, cell_end, outcomes }
  - New: run_cell_concurrent (Arc<Client> + Arc<Tokenizer> shared; N
    tokio::spawn workers run warmup phase then timed phase; each worker
    fires request -> await -> repeat until wall-clock deadline; per-worker
    nonce space = (worker_id<<48) ^ counter to guarantee unique prompts).

Report (report.rs):
  - New: ConcurrentCellStats { p50/p95/p99 TTFT + ITL + aggregate
    tokens/s + req/s + per-worker breakdown + finish_reason_summary +
    cached_tokens_warning }
  - New: percentile(sorted, p) helper (sort-and-index; empty -> 0.0)
  - New: reduce_concurrent_cell, render_markdown_concurrent,
    render_csv_concurrent (per-request rows, pandas-friendly),
    render_json_concurrent (mode=concurrent + per-cell aggregates +
    per-worker + targets)
  - New: 3 lib unit tests (percentile_basic, percentile_edge_cases,
    render_markdown_concurrent_smoke)

LOC delta: ~+400 across the 3 source files. v1 path unchanged
(reduce_cell + render_markdown + render_csv + render_json untouched).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Integration smoke test + README update

**Files:**
- Create: `iron-bench/tests/concurrent_smoke.rs`
- Modify: `iron-bench/README.md`
- Modify: `iron-bench/Cargo.toml` (add `[dev-dependencies] axum` for mock SSE server)

- [ ] **Step 1: Add axum + http-body-util dev-dependency**

Open `iron-bench/Cargo.toml`. Add a `[dev-dependencies]` section (or extend if present):

```toml
[dev-dependencies]
axum = "0.7"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

If `[dev-dependencies]` already exists, just append the `axum = "0.7"` line. Don't duplicate the `tokio` line if it's already there.

- [ ] **Step 2: Create the smoke test file**

Create `iron-bench/tests/concurrent_smoke.rs`:

```rust
//! Integration smoke test for iron-bench v2 concurrent mode.
//!
//! Launches an in-process mock SSE server that emits 5 OpenAI-compatible
//! `data: {...}` chunks per request after a 20ms delay, then `data: [DONE]`.
//! Invokes the iron-bench binary against the mock with --concurrent 2
//! --duration 1 and verifies the JSON output contains the expected
//! concurrent-mode fields.

use std::process::Command;
use std::time::Duration;

use axum::{
    response::sse::{Event, Sse},
    routing::post,
    Router,
};
use tokio::net::TcpListener;

async fn mock_sse_endpoint() -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>
{
    use async_stream::stream;
    Sse::new(stream! {
        // Emit 5 content chunks at 4ms intervals.
        for i in 0..5 {
            tokio::time::sleep(Duration::from_millis(4)).await;
            let chunk = serde_json::json!({
                "id": "mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "delta": {"content": format!("tok{i}")},
                    "finish_reason": null,
                }],
            });
            yield Ok(Event::default().data(chunk.to_string()));
        }
        // Emit usage block (in stream_options.include_usage = true contract).
        let usage = serde_json::json!({
            "id": "mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 16, "completion_tokens": 5, "cached_tokens": 0},
        });
        yield Ok(Event::default().data(usage.to_string()));
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_smoke_against_mock_server() {
    // 1. Bind mock server to a random port.
    let app = Router::new().route("/v1/chat/completions", post(mock_sse_endpoint));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });

    // 2. Resolve the bench binary path (cargo sets CARGO_BIN_EXE_iron-bench at test build time).
    let bench_bin = env!("CARGO_BIN_EXE_iron-bench");

    // 3. Need a tokenizer.json. Use the project root's iron-bench/tests/fixtures/tokenizer.json
    //    if it exists; otherwise skip with a warning (CI may not stage fixtures).
    let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "[smoke] tokenizer fixture not found at {} — skipping",
            tokenizer_path.display()
        );
        return;
    }

    let url = format!("http://{addr}");
    let output = Command::new(bench_bin)
        .args([
            "--target", &format!("mock={url}"),
            "--model-dir", tokenizer_path.parent().unwrap().to_str().unwrap(),
            "--model", "mock",
            "--prompt-len", "16",
            "--max-tokens", "5",
            "--concurrent", "2",
            "--duration", "1",
            "--warmup-duration", "0",
            "--format", "json",
        ])
        .output()
        .expect("spawn iron-bench");

    assert!(
        output.status.success(),
        "iron-bench v2 failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("output should be JSON when --format json");

    assert_eq!(json["mode"], "concurrent");
    assert_eq!(json["concurrent"], 2);
    let cells = json["cells"].as_array().expect("cells array");
    assert!(!cells.is_empty(), "should have at least one cell");
    let cell = &cells[0];
    assert_eq!(cell["concurrent"], 2);
    assert!(
        cell["n_requests"].as_u64().expect("n_requests u64") > 0,
        "workers should have run at least once"
    );
    assert!(cell["ttft_ms"]["p50"].is_number());
    assert!(cell["itl_ms"]["p99"].is_number());
    assert!(cell["aggregate"]["tokens_per_sec"].is_number());
    assert!(cell["per_worker"]["req_count"].is_array());
}
```

This requires the `async-stream` crate (used by axum::sse). Add it to dev-dependencies:

```toml
[dev-dependencies]
axum = "0.7"
async-stream = "0.3"
futures = { workspace = true }
serde_json = "1"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

(Check `workspace.dependencies` for the canonical versions used in cxx-mlx; `futures` is already in iron-bench's `[dependencies]` per `Cargo.toml` line ~21.)

If `axum 0.7`'s SSE API doesn't match the sketch above (the example uses post() returning Sse<Stream>), adapt — the goal is "any HTTP server that emits 5 SSE chunks + DONE upon POST". Alternative: use `hyper` directly or a simpler `tokio::net::TcpListener` + hand-rolled HTTP/SSE writes. If axum integration proves invasive, **STOP and ask** about replacing with a simpler approach.

- [ ] **Step 3: Tokenizer fixture**

The smoke test needs a `tokenizer.json` at `iron-bench/tests/fixtures/tokenizer.json`. Without it the test self-skips with a warning. For CI / local runs, you can stage a minimal tokenizer.json from e.g. a small Qwen tokenizer:

```bash
mkdir -p iron-bench/tests/fixtures
# Manually copy from $HOME/.ironmlx/models/.../snapshots/.../tokenizer.json:
cp "$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/)tokenizer.json" \
   iron-bench/tests/fixtures/tokenizer.json
```

Decide whether to commit this fixture into git (it's typically ~5MB; check size first). If too large, gitignore it and document in README that CI / local setup must stage the fixture before running smoke tests. Either way, the test handles the missing-fixture case gracefully.

- [ ] **Step 4: Update README**

Open `iron-bench/README.md`. Find the section that currently says (around line 86-90):

```
## Limitations

- **Single-request only**. Multi-request concurrency comes in v2 once ironmlx P8b ships
  the batched scheduler.
```

Replace with:

```
## Concurrency modes

iron-bench supports two modes:

### v1 sequential (default)

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1
```

One request at a time per (target, prompt_len) cell. Reports median + p95 over
the `--runs` timed iterations. Good for **single-request latency** comparison.

### v2 concurrent (multi-worker)

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --concurrent 4 --duration 30 --warmup-duration 5
```

`N` concurrent workers per cell run for `--duration` seconds (after
`--warmup-duration` discarded warmup). Reports **p50/p95/p99 TTFT + ITL +
aggregate tokens/s + per-worker breakdown**. Good for **multi-request
throughput** comparison.

Server requirements for v2:
- `ironmlx`: needs B1-p2.3c-3 (continuous batching, mid-batch admit). Set `b_max ≥ N` to avoid 'scheduler full' errors during the cell.
- `omlx`, `mlx-lm-server`: native multi-request support already.
- `vllm-mlx`, `llama.cpp`: configure server-side `--max-num-seqs ≥ N`.

## Limitations

- **Closed-loop only.** Each worker awaits its response before firing the next.
  Open-loop (Poisson arrival rate) ships in **v3** when fairness metrics become
  meaningful (ironmlx 3d admission queue).
- **No fairness metrics** (Jain's index, per-tenant quotas). Defers to v3.
- **No distributed load generation.** v2 runs from one machine. For higher load,
  scale `--concurrent` up (constrained by OS fd limits — `ulimit -n 65536` for
  N > 256).
```

- [ ] **Step 5: Run the smoke test**

```bash
cargo test -p iron-bench --release --test concurrent_smoke 2>&1 | tail -20
```

Expected: PASS. The test self-skips if `tests/fixtures/tokenizer.json` is missing (acceptable on CI without fixture staged).

- [ ] **Step 6: Manual end-to-end smoke against ironmlx**

If an ironmlx server is convenient to start, run a 5-second concurrent bench as a final check:

```bash
# Terminal 1: start ironmlx server (in another shell)
# (or skip this step if not convenient — the unit + integration tests are sufficient)
# MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx-app -- serve --port 8080 ...

# Terminal 2: run iron-bench v2
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --model-dir $(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  --model qwen3.5-4b \
  --prompt-len 128 \
  --max-tokens 32 \
  --concurrent 2 --duration 5 --warmup-duration 1 \
  --format markdown
```

Expected: prints a v2 markdown report with p50/p95/p99 TTFT/ITL + aggregate tokens/s + per-worker breakdown. No crashes. Errors during the bench (e.g., `Err("scheduler full")` if N > ironmlx's b_max) should be reported and propagate — confirms error path works.

This step is optional if ironmlx isn't running locally. The lib + integration tests are the gate.

- [ ] **Step 7: Hygiene gate**

```bash
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release -p iron-bench
```

All clean.

- [ ] **Step 8: Commit**

```bash
git add iron-bench/Cargo.toml iron-bench/tests/concurrent_smoke.rs iron-bench/README.md
# If you decided to stage tests/fixtures/tokenizer.json:
# git add iron-bench/tests/fixtures/tokenizer.json
git commit -m "$(cat <<'EOF'
test+docs(iron-bench): v2 concurrent smoke + README

Adds an integration smoke test that launches an in-process axum SSE mock
server, invokes the iron-bench binary against it with --concurrent 2
--duration 1, and verifies the JSON output contains the expected v2
concurrent-mode fields (mode=concurrent, concurrent=2, p50/p95/p99 TTFT
+ ITL, aggregate tokens/s, per-worker req_count).

Test self-skips if iron-bench/tests/fixtures/tokenizer.json isn't
present (CI / local setup stages the fixture; smoke test is opt-in
without it).

dev-dependencies add: axum 0.7, async-stream 0.3, serde_json 1. tokio is
already in dependencies; reused for the smoke test runtime.

README updated:
- "Single-request only" limitation note removed.
- New "Concurrency modes" section with v1 sequential + v2 concurrent
  invocation examples.
- Server requirements per backend documented (ironmlx needs 3c-3 b_max,
  vllm/llama.cpp need --max-num-seqs, omlx/mlx-lm-server native).
- New "Limitations" section: closed-loop only, no fairness metrics, no
  distributed load gen. v3 backlog noted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Plan Self-Review

**1. Spec coverage** (spec §4 architecture + §5 tests + §9 risks):

- ✅ Spec §4.1 CLI surface (`--concurrent`, `--duration`, `--warmup-duration`, mode dispatch) — Task 1.
- ✅ Spec §4.2 `run_cell_concurrent` — Task 2.
- ✅ Spec §4.3 module surface (main.rs, runner.rs, report.rs, README) — Tasks 1+2+3+4.
- ✅ Spec §4.4 percentile via sort-and-index — Task 3 `percentile` helper.
- ✅ Spec §4.5 worker startup skew (not compensated; sub-ms) — implicit in `tokio::spawn` choice; no explicit handling needed.
- ✅ Spec §4.6 HTTP client pool sizing — `Arc<reqwest::Client>` in Task 2.
- ✅ Spec §4.7 tokenizer concurrency — `Arc<Tokenizer>` in Task 2.
- ✅ Spec §4.8 invariants (nonce uniqueness, Arc sharing, worker outcome agg, cell isolation, timing reference) — Task 2 implementation.
- ✅ Spec §5.1 unit tests for percentile + render_markdown_concurrent — Task 3.
- ✅ Spec §5.2 integration smoke test — Task 4.
- ✅ Spec §5.3 no v1 regression — preserved by leaving `reduce_cell` + `render_markdown` etc. untouched; verified by manual v1 invocation test.
- ✅ Spec §6 acceptance gates (hygiene + LOC budget + new metrics fields present) — Task 4 + final hygiene.
- ✅ Spec §9 R1 (runtime starvation): documented in README — Task 4.
- ✅ Spec §9 R2 (server rate limits): existing `--timeout` works; documented.
- ✅ Spec §9 R3 (clock drift): single Instant origin — Task 2.
- ✅ Spec §9 R4 (small M percentile): edge-case tests — Task 3.
- ✅ Spec §9 R5 (Vec memory at large N): documented; <10MB worst case.
- ✅ Spec §9 R6 (ironmlx 3c-3 `scheduler full`): documented in README server-requirements + happens at request layer (caller sees Err) — runner propagates.
- ✅ Spec §9 R7 (connection pool fd exhaustion): README note about `ulimit -n`.

**2. Placeholder scan:** No "TBD", "implement later", or "Similar to Task N" without code. Every step contains exact commands or full code blocks.

**3. Type consistency:**

- `ConcurrentCellResult { target_name, target_url, pp_target, tg_target, concurrent, cell_start, cell_end, outcomes }` — consistent across Task 2 (definition) and Task 3 (consumed in `reduce_concurrent_cell`).
- `RequestOutcome { worker_id, prompt_tokens_local, result }` — consistent.
- `ConcurrentCellStats` field names — consistent between Task 3 step 1 (definition) and step 2 (markdown formatter consumption) and step 4 (json formatter consumption).
- `percentile(sorted: &[f64], p: f64) -> f64` — consistent across uses.
- CLI flag names (`--concurrent`, `--duration`, `--warmup-duration`) — consistent across Tasks 1+4 (README) + integration smoke test invocation.

Plan looks clean.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-iron-bench-v2-concurrent.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + spec compliance + code quality review between tasks. Pattern used for 3c-1/2/3.

**2. Inline Execution** — Run tasks in this session with checkpoint commits.

**Which approach?**
