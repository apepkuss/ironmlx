//! P5h T0b — Phase D root cause investigation harness.
//!
//! T0b.1 H1 thermal drift: phase-order comparison.
//!   - normal order: A -> B -> C -> D[3 modes]
//!   - reversed order: D[3 modes] -> C -> B -> A
//!   - identical cooldown policy in both (P5g-compatible minimal cooling)
//!   - if H1 verifies, optional mitigation-confirmation pass with 5min pre-phase
//!     cool gate (run on Boss directive; not part of the order-vs-order verdict)
//!
//! Output: /tmp/p5h-t0b-h1.json with normal/reversed phase_d, drift_pct_per_cell,
//! verdict.
//!
//! Per spec § 2.5 H1 + plan T0b.1 + reports/p5h-t0b-phase-d-design.md § 3 T0b.1.
//!
//! Server feature gate: p5g-profile (NOT p5h-profile per § 2.5 + D-decision):
//! T0b measurements use the lower-overhead P5g profiling path so per-span
//! tracing::info! overhead does not contaminate substitute/kernel timing.
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5g-profile \
//!     --test p5h_t0b_phase_d t0b_h1_phase_order_comparison \
//!     -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5g-profile")]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const PP_LIST: [i32; 4] = [2048, 4096, 8192, 16384];
const ABLATION_MODES: [&str; 3] = ["ablate-compute-g", "ablate-conv", "ablate-t-arr"];
// Distinct from P5g (18080) and P5h T0a (18099) to avoid clash if any stale
// server lingers across consecutive test invocations.
const PROFILE_PORT: u16 = 18081;
// RUNS=7 + trimmed median (drop min + max) gives a 5-sample stable estimator;
// T0a.13-fix lesson — single-run PP=16384 outliers contaminated the verdict.
const RUNS: usize = 7;
// P5h convention: WARMUP=0 keeps first-spawn cold thermal state inside the
// measurement window, matching design memo "test must reproduce 'D runs late'
// mechanism" (P5g used WARMUP=1 which masks the very effect H1 is checking).
const WARMUP: usize = 0;
// P5g-compatible minimal cooling: 3s OS port-release + GPU-model-release gap.
// NO additional inter-phase cool gate (per design memo § 3 T0b.1: the H1
// experiment must reproduce P5g contamination conditions, not mitigate them).
const INTER_PP_COOLDOWN: Duration = Duration::from_secs(3);

const H1_OUTPUT_PATH: &str = "/tmp/p5h-t0b-h1.json";

// ===== Helpers inline-copied from p5g_t0_gated_delta_profile.rs =====
// Option A per design memo § 4: zero coupling with the P5g harness so this
// test can evolve independently and P5g back-compat is preserved.

fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

/// Spawn `cargo run -p iron-bench` against the running server. T0b uses
/// WARMUP=0 + RUNS=7 (vs. P5g's WARMUP=1 + RUNS=3) so the trimmed-median
/// reducer sees enough samples to absorb single-run environmental outliers
/// while keeping the first-spawn cold state in the measurement window.
fn iron_bench_run(
    port: u16,
    model_dir: &str,
    prompt_len: i32,
) -> std::io::Result<std::process::Output> {
    Command::new("cargo")
        .args([
            "run",
            "-p",
            "iron-bench",
            "--release",
            "--",
            "--target",
            &format!("p5g_profile=http://127.0.0.1:{port}"),
            "--model",
            "qwen3.5-moe",
            "--model-dir",
            model_dir,
            "--prompt-len",
            &prompt_len.to_string(),
            "--max-tokens",
            "1",
            "--runs",
            &RUNS.to_string(),
            "--warmup",
            &WARMUP.to_string(),
            "--format",
            "json",
        ])
        .output()
}

/// Parse iron-bench `--format json` stdout — extract `raw_runs[].pp_tps` values
/// (one per measured run). Panics with full stdout context on parse failure.
fn parse_pp_tps_from_bench(stdout_bytes: &[u8]) -> Vec<f64> {
    let s = String::from_utf8_lossy(stdout_bytes);
    let v: serde_json::Value = serde_json::from_str(&s).unwrap_or_else(|e| {
        let preview: String = s.chars().take(400).collect();
        panic!("iron-bench JSON parse failed: {e}; raw stdout (first 400): {preview}")
    });
    let mut tps = Vec::new();
    if let Some(arr) = v.get("raw_runs").and_then(|x| x.as_array()) {
        for r in arr {
            if let Some(p) = r.get("pp_tps").and_then(|x| x.as_f64()) {
                tps.push(p);
            }
        }
    }
    tps
}

/// Spawn `ironmlx serve` with optional `IRONMLX_P5G_PROFILE_MODE` env. Always
/// clear the env var first — defensive against stray exports in the caller's
/// shell that would silently flip Phase A / normal-order baseline into a
/// profile-mode run.
fn spawn_server(profile_mode: Option<&str>, model_dir: &str, port: u16) -> Child {
    let bin = env!("CARGO_BIN_EXE_ironmlx");
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model",
        model_dir,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    cmd.env_remove("IRONMLX_P5G_PROFILE_MODE");
    if let Some(mode) = profile_mode {
        cmd.env("IRONMLX_P5G_PROFILE_MODE", mode);
    }
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    cmd.stderr(Stdio::piped());
    cmd.spawn().expect("ironmlx serve spawn")
}

/// Hard port-free check via TCP bind — refuse to auto-kill any stale server,
/// refuse to silently re-use its healthz. Bind-test surfaces the same OS-level
/// constraint `ironmlx serve` will see, unlike `lsof`-style introspection.
fn assert_port_free(port: u16) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    drop(listener);
    Ok(())
}

/// Healthz poll. Returns Err so the caller can run shutdown + drainer join on
/// failure instead of leaking the Child on panic.
fn wait_for_ready(port: u16, max_seconds: u64) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/healthz");
    let deadline = std::time::Instant::now() + Duration::from_secs(max_seconds);
    loop {
        if let Ok(out) = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
        {
            if String::from_utf8_lossy(&out.stdout).trim() == "200" {
                return Ok(());
            }
        }
        if std::time::Instant::now() > deadline {
            return Err(format!(
                "ironmlx serve did not become ready within {max_seconds}s at {url}"
            ));
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

/// Best-effort shutdown helper for the failure paths: kill, wait, join drainer.
fn shutdown_and_join(mut server: Child, drainer: JoinHandle<()>) {
    let _ = server.kill();
    let _ = server.wait();
    let _ = drainer.join();
}

/// Spawn a line-by-line stderr drainer thread to keep the server's stderr pipe
/// from filling (>64KB on startup blocks the server before healthz comes up).
/// The shared buffer is kept around but NOT parsed for H1 — T0b.1 only needs
/// the iron-bench JSON output's `raw_runs[].pp_tps` per the design's stated
/// metric (P5g log records are not part of the H1 verdict).
fn spawn_stderr_drainer(server: &mut Child) -> (Arc<Mutex<Vec<u8>>>, JoinHandle<()>) {
    let stderr_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let handle = server.stderr.take().expect("server stderr");
    let buf_clone = Arc::clone(&stderr_buf);
    let drainer = std::thread::spawn(move || {
        let mut rdr = BufReader::new(handle);
        let mut line = String::new();
        loop {
            line.clear();
            match rdr.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => buf_clone.lock().unwrap().extend_from_slice(line.as_bytes()),
                Err(_) => break,
            }
        }
    });
    (stderr_buf, drainer)
}

// ===== T0b.1-specific helpers =====

/// Trimmed median: drop min + max, return median of the remaining middle
/// values. With RUNS=7 the trimmed set has 5 values. Falls back to plain
/// median when `vals.len() < 3` (insufficient samples to trim). Returns None
/// on empty input. Panics on NaN — iron-bench shouldn't emit NaN; failure-fast
/// surfaces upstream measurement bugs.
fn trimmed_median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("pp_tps contained NaN"));
    if v.len() < 3 {
        let n = v.len();
        return Some(if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        });
    }
    let trimmed = &v[1..v.len() - 1];
    let n = trimmed.len();
    Some(if n % 2 == 1 {
        trimmed[n / 2]
    } else {
        (trimmed[n / 2 - 1] + trimmed[n / 2]) / 2.0
    })
}

/// Spawn server with the given profile mode, run iron-bench RUNS=7 times at
/// `pp` prompt length, kill server, and return the trimmed-median pp_tps.
/// Server stderr captured via drainer but not parsed for H1 — only the
/// iron-bench JSON output's `raw_runs[].pp_tps` matters per design memo § 3.
fn run_one_pp_one_mode(
    mode: Option<&str>,
    model_dir: &str,
    port: u16,
    pp: i32,
) -> anyhow::Result<f64> {
    assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;

    let mut server = spawn_server(mode, model_dir, port);

    // Drainer BEFORE wait_for_ready — server startup can fill the 64KB stderr
    // pipe and block before healthz goes up otherwise.
    let (_stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

    if let Err(e) = wait_for_ready(port, 300) {
        shutdown_and_join(server, drainer);
        anyhow::bail!("PP={pp} mode={mode:?}: server not ready: {e}");
    }

    // Detect server that exited before healthz came up (spawn failed, model
    // missing, MLX init error, etc.). Without this check, a dead-server
    // failure would look like a startup hang on the next iteration.
    match server.try_wait() {
        Ok(Some(status)) => {
            let _ = drainer.join();
            anyhow::bail!("PP={pp} mode={mode:?}: ironmlx serve exited before bench with {status}");
        }
        Ok(None) => {}
        Err(e) => {
            shutdown_and_join(server, drainer);
            anyhow::bail!("PP={pp} mode={mode:?}: try_wait failed: {e}");
        }
    }

    let out = match iron_bench_run(port, model_dir, pp) {
        Ok(o) => o,
        Err(e) => {
            shutdown_and_join(server, drainer);
            anyhow::bail!("PP={pp} mode={mode:?}: iron-bench spawn failed: {e}");
        }
    };

    // Shutdown FIRST so drainer EOF + join completes before we touch buffers.
    let _ = server.kill();
    let _ = server.wait();
    let _ = drainer.join();

    if !out.status.success() {
        anyhow::bail!(
            "PP={pp} mode={mode:?}: iron-bench non-success: stdout={}, stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    let raw_runs = parse_pp_tps_from_bench(&out.stdout);
    if raw_runs.len() != RUNS {
        anyhow::bail!(
            "PP={pp} mode={mode:?}: expected {} pp_tps samples, got {} — iron-bench stdout \
             truncated or runs missing pp_tps. stdout={}",
            RUNS,
            raw_runs.len(),
            String::from_utf8_lossy(&out.stdout),
        );
    }
    // Defensive: trimmed_median can only return None on empty input, which the
    // length check above already rejects. Keep the fallback to avoid an
    // unreachable!() panic path that future RUNS=0 misconfigurations would hit.
    let med = trimmed_median(raw_runs)
        .ok_or_else(|| anyhow::anyhow!("PP={pp} mode={mode:?}: no pp_tps from iron-bench"))?;

    eprintln!("[p5h-t0b-h1] mode={mode:?} PP={pp}: pp_tps_trimmed_median={med:.2}");
    std::thread::sleep(INTER_PP_COOLDOWN);
    Ok(med)
}

/// Run phases in normal P5g order: A -> B -> C -> D[ablate-compute-g,
/// ablate-conv, ablate-t-arr]. Returns only the Phase D pp_tps_median per
/// (ablate_mode, PP); Phase A/B/C measurements run for thermal accumulation
/// realism but are discarded.
fn run_normal_order_capture_d(
    model_dir: &str,
    port: u16,
) -> anyhow::Result<BTreeMap<String, BTreeMap<i32, f64>>> {
    // Phase A: NO profile mode, 4 PPs — establishes the same thermal
    // accumulator P5g built up before D ran.
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(None, model_dir, port, pp)?;
    }
    // Phase B: layer1 boundary-isolated.
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(Some("layer1"), model_dir, port, pp)?;
    }
    // Phase C: layer2 per-step breakdown.
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(Some("layer2"), model_dir, port, pp)?;
    }
    // Phase D — capture.
    let mut phase_d: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
    for &mode in &ABLATION_MODES {
        let mut by_pp = BTreeMap::new();
        for &pp in &PP_LIST {
            let pp_tps = run_one_pp_one_mode(Some(mode), model_dir, port, pp)?;
            by_pp.insert(pp, pp_tps);
        }
        phase_d.insert(mode.to_string(), by_pp);
    }
    Ok(phase_d)
}

/// Run phases in reversed order: D[3 modes] -> C -> B -> A. Returns only the
/// Phase D pp_tps_median per (ablate_mode, PP); Phase A/B/C measurements run
/// AFTER Phase D for symmetric thermal contamination structure but are
/// discarded.
fn run_reversed_order_capture_d(
    model_dir: &str,
    port: u16,
) -> anyhow::Result<BTreeMap<String, BTreeMap<i32, f64>>> {
    // Phase D — capture first.
    let mut phase_d: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
    for &mode in &ABLATION_MODES {
        let mut by_pp = BTreeMap::new();
        for &pp in &PP_LIST {
            let pp_tps = run_one_pp_one_mode(Some(mode), model_dir, port, pp)?;
            by_pp.insert(pp, pp_tps);
        }
        phase_d.insert(mode.to_string(), by_pp);
    }
    // Then C -> B -> A (discarded for H1; symmetric spawn-count only).
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(Some("layer2"), model_dir, port, pp)?;
    }
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(Some("layer1"), model_dir, port, pp)?;
    }
    for &pp in &PP_LIST {
        let _ = run_one_pp_one_mode(None, model_dir, port, pp)?;
    }
    Ok(phase_d)
}

#[derive(Debug, serde::Serialize)]
struct DriftCell {
    ablate_mode: String,
    pp: i32,
    normal_pp_tps: f64,
    reversed_pp_tps: f64,
    drift_pct: f64,
}

#[derive(Debug, serde::Serialize)]
struct H1Verdict {
    /// One of "verified" | "rejected" | "inconclusive" — machine-readable
    /// signal for T0b.5 decision tree binding.
    verdict: String,
    /// Human-readable rationale citing median + max drift_pct + thresholds.
    rationale: String,
    cells: Vec<DriftCell>,
    median_drift_pct: f64,
    max_drift_pct: f64,
    /// Records the exact initial cool / cooldown protocol the H1 sweep used,
    /// per design memo D8 "T0b starts from a clean/cool baseline and records
    /// the exact initial cool/restart protocol".
    initial_cool_protocol: String,
    /// Cells skipped because normal/reversed BTreeMap keys disagreed. Empty in
    /// the happy path. Non-empty entries indicate partial data — downstream
    /// T0b.5 reads `cells.len()` + `skipped.len()` to assess data quality.
    skipped: Vec<String>,
}

/// Compute drift_pct per (mode, PP) cell + final verdict per design memo § 3
/// T0b.1:
///   drift_pct = |reversed_D - normal_D| / normal_D
///   VERIFIED:    median > 5% OR max > 10%
///   REJECTED:    max < 2%
///   INCONCLUSIVE: otherwise
fn compute_h1_verdict(
    normal: &BTreeMap<String, BTreeMap<i32, f64>>,
    reversed: &BTreeMap<String, BTreeMap<i32, f64>>,
) -> H1Verdict {
    let mut cells: Vec<DriftCell> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for mode in normal.keys() {
        let normal_per_pp = &normal[mode];
        let Some(reversed_per_pp) = reversed.get(mode) else {
            skipped.push(format!("reversed missing mode={mode}"));
            continue;
        };
        for (&pp, &normal_tps) in normal_per_pp {
            let Some(&reversed_tps) = reversed_per_pp.get(&pp) else {
                skipped.push(format!("reversed missing mode={mode} PP={pp}"));
                continue;
            };
            let drift_pct = if normal_tps > 0.0 {
                (reversed_tps - normal_tps).abs() / normal_tps
            } else {
                0.0
            };
            cells.push(DriftCell {
                ablate_mode: mode.clone(),
                pp,
                normal_pp_tps: normal_tps,
                reversed_pp_tps: reversed_tps,
                drift_pct,
            });
        }
    }
    let mut drifts: Vec<f64> = cells.iter().map(|c| c.drift_pct).collect();
    drifts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_drift_pct = if drifts.is_empty() {
        0.0
    } else {
        drifts[drifts.len() / 2]
    };
    let max_drift_pct = drifts.last().copied().unwrap_or(0.0);
    let verdict = if median_drift_pct > 0.05 || max_drift_pct > 0.10 {
        "verified"
    } else if max_drift_pct < 0.02 {
        "rejected"
    } else {
        "inconclusive"
    };
    let rationale = if skipped.is_empty() {
        format!(
            "H1 {verdict}: median_drift_pct={median_drift_pct:.4}, \
             max_drift_pct={max_drift_pct:.4} (thresholds: verified if median>0.05 OR \
             max>0.10; rejected if max<0.02)"
        )
    } else {
        format!(
            "H1 {verdict}: median_drift_pct={median_drift_pct:.4}, \
             max_drift_pct={max_drift_pct:.4} (thresholds: verified if median>0.05 OR \
             max>0.10; rejected if max<0.02) — WARNING: {} cells skipped: {}",
            skipped.len(),
            skipped.join("; ")
        )
    };
    H1Verdict {
        verdict: verdict.to_string(),
        rationale,
        cells,
        median_drift_pct,
        max_drift_pct,
        initial_cool_protocol: "no explicit cool restart; assumes serial test invocation; \
                                INTER_PP_COOLDOWN=3s; no inter-phase cool (P5g-compatible \
                                minimal cooling per design memo § 3 T0b.1)"
            .to_string(),
        skipped,
    }
}

#[test]
#[ignore = "p5h-t0b H1 thermal drift — phase-order comparison (~45-60min GPU)"]
fn t0b_h1_phase_order_comparison() -> anyhow::Result<()> {
    let model_dir = snapshot_dir();
    // Fail-fast guard: missing MLX_DIR surfaces here in 0 seconds instead of a
    // confusing "ironmlx serve exited before bench" 30s into spawn_server. The
    // env-var read at spawn_server line 136 stays — this just adds an early
    // operator-friendly error.
    let _mlx_dir = std::env::var("MLX_DIR")
        .expect("set MLX_DIR env var pointing to MLX install prefix (e.g. $HOME/.local/mlx)");
    eprintln!("[p5h-t0b-h1] starting; model={model_dir}");

    eprintln!("[p5h-t0b-h1] phase 1: normal order (A -> B -> C -> D[3 modes])");
    let normal_phase_d = run_normal_order_capture_d(&model_dir, PROFILE_PORT)?;

    eprintln!("[p5h-t0b-h1] phase 2: reversed order (D[3 modes] -> C -> B -> A)");
    let reversed_phase_d = run_reversed_order_capture_d(&model_dir, PROFILE_PORT)?;

    let verdict = compute_h1_verdict(&normal_phase_d, &reversed_phase_d);
    eprintln!("[p5h-t0b-h1] {}", verdict.rationale);

    let out_json = serde_json::json!({
        "pp_list": PP_LIST,
        "ablation_modes": ABLATION_MODES,
        "runs": RUNS,
        "warmup": WARMUP,
        "inter_pp_cooldown_secs": INTER_PP_COOLDOWN.as_secs(),
        "normal_order_phase_d": normal_phase_d,
        "reversed_order_phase_d": reversed_phase_d,
        "drift_pct_per_cell": verdict.cells,
        "median_drift_pct": verdict.median_drift_pct,
        "max_drift_pct": verdict.max_drift_pct,
        "verdict": verdict.verdict,
        "rationale": verdict.rationale,
        "initial_cool_protocol": verdict.initial_cool_protocol,
        "skipped_cells": verdict.skipped,
    });
    let json_str = serde_json::to_string_pretty(&out_json)?;
    // Dump the full payload to stderr BEFORE the file write so that if the
    // /tmp write fails (disk full / permissions / fs read-only) the data is
    // still recoverable from --nocapture scrollback after 45-60min of GPU
    // work.
    eprintln!("[p5h-t0b-h1] JSON payload (preserved in case file-write fails):\n{json_str}");
    std::fs::write(H1_OUTPUT_PATH, &json_str)?;
    eprintln!(
        "[p5h-t0b-h1] wrote {} bytes to {H1_OUTPUT_PATH}",
        json_str.len()
    );

    // Per design memo § 3 T0b.1: do NOT fail the test on inconclusive/rejected
    // — H1 records the verdict and downstream T0b.5 reads it. Verified vs
    // rejected are both valid scientific outcomes; harness writes JSON and
    // exits cleanly. The verdict string in JSON is the consumed signal.
    Ok(())
}
