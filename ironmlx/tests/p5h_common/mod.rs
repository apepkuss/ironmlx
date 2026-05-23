//! P5h shared test helpers — server spawn / stderr drain / iron-bench wrappers
//! / preheat / aggregation utilities used by T1+ harnesses.
//!
//! Extracted 2026-05-22 from `tests/p5h_t0b_phase_d.rs`'s inline copies
//! (T0b kept its inline copies untouched per Option A; this module exists for
//! T1+ to import without coupling to T0b's commit history). T0a's separate
//! inline helpers in `tests/p5h_t0a_harness.rs` also stay untouched.
//!
//! NOT gated by any feature: callers `cargo test --features p5h-profile`
//! (T1+) or `--features p5g-profile` (T0b's inline copy path) both compile
//! against this module; the test target binary `CARGO_BIN_EXE_ironmlx` is
//! rebuilt per feature flavor by cargo so the helpers themselves are
//! feature-agnostic.
//!
//! Conventions preserved from T0b inline source (load-bearing — DO NOT relax):
//!   * RUNS=7 + trimmed median (drop min + max) for 5-sample stable estimator
//!     after the T0a.13-fix lesson — single-run PP=16384 outliers contaminated
//!     verdicts otherwise.
//!   * WARMUP=0 keeps first-spawn cold thermal state inside the measurement
//!     window (P5g WARMUP=1 masks the very effect H1 was checking).
//!   * INTER_PP_COOLDOWN=3s is the P5g-compatible minimal cooling gap (OS
//!     port-release + GPU-model-release); per-PP spawn-kill is the cool
//!     mechanism, not a long sleep.
//!   * Stderr drainer started BEFORE wait_for_ready: server startup can fill
//!     the 64KB stderr pipe and block before healthz comes up otherwise.
//!   * Failure paths route through `shutdown_and_join` so the Child + drainer
//!     are not leaked on panic.

#![allow(dead_code)] // each test file uses a subset; full set kept for future T1+ sweeps

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// RUNS=7 + trimmed median (drop min + max) gives a 5-sample stable estimator;
/// T0a.13-fix lesson — single-run PP=16384 outliers contaminated the verdict.
pub const RUNS: usize = 7;

/// P5h convention: WARMUP=0 keeps first-spawn cold thermal state inside the
/// measurement window.
pub const WARMUP: usize = 0;

/// Preheat throwaway run count per PP (Boss option C decision after T0b.1 H1).
pub const PREHEAT_RUNS: usize = 3;

/// P5g-compatible minimal cooling gap: 3s OS port-release + GPU-model-release.
pub const INTER_PP_COOLDOWN: Duration = Duration::from_secs(3);

/// Shared port for all `--features p5h-profile` sweeps. Aligned with T0a's
/// `PORT = 18099` so the p5h-profile flavor uses a single consistent port
/// (T0b uses 18081 because it runs `p5g-profile` feature; that file keeps
/// its own const).
pub const PROFILE_PORT: u16 = 18099;

/// Snapshot directory env-var reader. Panics on missing `IRONMLX_MOE_MODEL_DIR`.
pub fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

/// Spawn `cargo run -p iron-bench` against the running server. Uses
/// `RUNS` + `WARMUP` consts from this module.
pub fn iron_bench_run(
    port: u16,
    model_dir: &str,
    prompt_len: i32,
) -> std::io::Result<std::process::Output> {
    iron_bench_run_with_runs(port, model_dir, prompt_len, RUNS, WARMUP)
}

/// iron-bench variant accepting explicit `runs` + `warmup` counts. Used by the
/// preheat helper (RUNS=PREHEAT_RUNS, WARMUP=0) and by harness code that wants
/// a non-default sample shape.
pub fn iron_bench_run_with_runs(
    port: u16,
    model_dir: &str,
    prompt_len: i32,
    runs: usize,
    warmup: usize,
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
            &runs.to_string(),
            "--warmup",
            &warmup.to_string(),
            "--format",
            "json",
        ])
        .output()
}

/// Parse iron-bench `--format json` stdout — extract `raw_runs[].pp_tps`
/// values (one per measured run). Panics with full stdout context on parse
/// failure.
pub fn parse_pp_tps_from_bench(stdout_bytes: &[u8]) -> Vec<f64> {
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
/// shell that would silently flip a baseline run into a profile-mode run.
/// T1 callers pass `None` (no profile mode); T0b-style callers pass
/// `Some("h2-measure")` etc.
pub fn spawn_server(profile_mode: Option<&str>, model_dir: &str, port: u16) -> Child {
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
pub fn assert_port_free(port: u16) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    drop(listener);
    Ok(())
}

/// Healthz poll. Returns Err so the caller can run shutdown + drainer join on
/// failure instead of leaking the Child on panic.
pub fn wait_for_ready(port: u16, max_seconds: u64) -> Result<(), String> {
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

/// Best-effort shutdown helper for failure paths: kill, wait, join drainer.
pub fn shutdown_and_join(mut server: Child, drainer: JoinHandle<()>) {
    let _ = server.kill();
    let _ = server.wait();
    let _ = drainer.join();
}

/// Spawn a line-by-line stderr drainer thread to keep the server's stderr pipe
/// from filling (>64KB on startup blocks the server before healthz comes up).
/// Returns `(shared_buffer, drainer_handle)`. Caller drains via
/// `drain_stderr_into_buf` after `server.kill() + wait() + drainer.join()`.
pub fn spawn_stderr_drainer(server: &mut Child) -> (Arc<Mutex<Vec<u8>>>, JoinHandle<()>) {
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

/// Atomically drain the stderr buffer into an owned `Vec<u8>` so the caller
/// can attribute the captured records to a specific iron-bench PP iteration.
/// Used by harnesses which must associate server emissions with the PP that
/// produced them (server's stderr does NOT include the PP).
pub fn drain_stderr_into_buf(stderr_buf: &Arc<Mutex<Vec<u8>>>) -> Vec<u8> {
    let mut g = stderr_buf.lock().unwrap();
    std::mem::take(&mut *g)
}

/// Trimmed median: drop min + max, return median of the remaining middle
/// values. With RUNS=7 the trimmed set has 5 values. Falls back to plain
/// median when `v.len() < 3`. Returns None on empty input. Panics on NaN —
/// failure-fast surfaces upstream measurement bugs.
pub fn trimmed_median(mut v: Vec<f64>) -> Option<f64> {
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

/// Median of an f64 vec (panics on NaN). Returns None on empty input. Used
/// for per-PP per-step ratio aggregation across many records when the sample
/// count is unknown ahead of time (vs. `trimmed_median` for fixed-N iron-bench
/// pp_tps samples).
pub fn plain_median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("median input contained NaN"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Spawn server with the given profile mode, run iron-bench RUNS=7 times at
/// `pp` prompt length, kill server, and return the trimmed-median pp_tps.
/// Server stderr captured via drainer but not parsed.
pub fn run_one_pp_one_mode(
    mode: Option<&str>,
    model_dir: &str,
    port: u16,
    pp: i32,
) -> anyhow::Result<f64> {
    run_one_pp_one_mode_with_runs(mode, model_dir, port, pp, RUNS)
}

/// Variant of `run_one_pp_one_mode` that accepts a `runs` parameter so the
/// preheat helper can use RUNS=PREHEAT_RUNS while the measurement loop uses
/// RUNS=7.
pub fn run_one_pp_one_mode_with_runs(
    mode: Option<&str>,
    model_dir: &str,
    port: u16,
    pp: i32,
    runs: usize,
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
    // missing, MLX init error, etc.).
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

    let out = match iron_bench_run_with_runs(port, model_dir, pp, runs, WARMUP) {
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
    if raw_runs.len() != runs {
        anyhow::bail!(
            "PP={pp} mode={mode:?}: expected {runs} pp_tps samples, got {} — iron-bench stdout \
             truncated or runs missing pp_tps. stdout={}",
            raw_runs.len(),
            String::from_utf8_lossy(&out.stdout),
        );
    }
    let med = trimmed_median(raw_runs)
        .ok_or_else(|| anyhow::anyhow!("PP={pp} mode={mode:?}: no pp_tps from iron-bench"))?;

    eprintln!("[p5h-common runs={runs}] mode={mode:?} PP={pp}: pp_tps_trimmed_median={med:.2}");
    std::thread::sleep(INTER_PP_COOLDOWN);
    Ok(med)
}

/// Preheat: drive GPU into thermal saturation. Runs `pp_list` × PREHEAT_RUNS
/// throwaway Phase A iron-bench iterations. Results discarded — pure thermal
/// conditioning per Boss option C decision after T0b.1 H1 inconclusive verdict.
///
/// Signature change vs. T0b inline copy: takes `pp_list: &[i32]` so each
/// sweep passes its own PP range (T1: `[128, 512, 2048]`; T0b's inline copy
/// stays on `[2048, 4096, 8192, 16384]`; T2-T4 may pick others).
pub fn preheat_to_saturation(model_dir: &str, port: u16, pp_list: &[i32]) -> anyhow::Result<()> {
    eprintln!(
        "[preheat] starting throwaway workload ({} PPs × RUNS={}) for thermal saturation",
        pp_list.len(),
        PREHEAT_RUNS
    );
    for &pp in pp_list {
        let _ = run_one_pp_one_mode_with_runs(None, model_dir, port, pp, PREHEAT_RUNS)?;
    }
    eprintln!("[preheat] thermal saturation done");
    Ok(())
}
