//! T0a UMA hardening + GDN P5h-protocol rerun harness.
//!
//! Cold/warm pair protocol: for each PP value, run iron-bench once after a
//! cold spawn, cool 5 minutes, then run again warm; compare variance.
//! > ±2% triggers a failure.
//!
//! Per spec § 2.4 + § 3 T0a.13 + Codex plan review v20 P1 #2
//! (--warmup 0 + --capture-server-request-id).
//!
//! Invocation:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5h-profile \
//!     --test p5h_t0a_harness -- --ignored --test-threads=1

#![cfg(feature = "p5h-profile")]

use std::fs::File;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const PP_LIST: &[u32] = &[128, 512, 2048, 4096, 8192, 16384];
const COOL_DURATION_MS: u64 = 5 * 60 * 1000;
const VARIANCE_THRESHOLD: f64 = 0.02;
const RUNS: usize = 7;
// P5h sweeps timed-only — per Codex plan review v20 P1 #2:
//  `--capture-server-request-id` requires `--warmup 0` because warmup
//  RequestResults are discarded by `iron-bench/src/runner.rs:72-75`
//  while the server still emits `[p5h-profile]` records +
//  `X-Ironmlx-Request-Id` headers for warmup requests, causing the
//  T0a.12 aggregator's 100% join gate to hard-fail on otherwise-correct
//  header propagation.
const WARMUP: usize = 0;

// Dedicated P5h harness port (avoid clash with prod default 8080 and
// P5g harness's PROFILE_PORT 18080).
const PORT: u16 = 18099;

const SERVER_LOG_PATH: &str = "/tmp/p5h-t0a-server.log";
const BENCH_CSV_PATH: &str = "/tmp/p5h-t0a-bench.csv";

fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

fn assert_port_free(port: u16) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    drop(listener);
    Ok(())
}

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

/// Spawn ironmlx serve with `--b-max 1`. Test binary is already built with
/// `--features p5h-profile` (the `#![cfg]` attribute at top of file ensures
/// this harness only compiles + runs under that feature), so the
/// `env!("CARGO_BIN_EXE_ironmlx")` binary is also feature-gated.
fn spawn_server_p5h(model_dir: &str, port: u16) -> Child {
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
        "--b-max",
        "1",
    ]);
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    cmd.stderr(Stdio::piped());
    cmd.spawn().expect("ironmlx serve spawn")
}

fn spawn_stderr_drainer(server: &mut Child) -> (Arc<Mutex<Vec<u8>>>, JoinHandle<()>) {
    let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let buf_clone = Arc::clone(&stderr_buf);
    let stderr = server.stderr.take().expect("server stderr should be piped");
    let handle = std::thread::spawn(move || {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(stderr);
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Ok(mut g) = buf_clone.lock() {
                        g.extend_from_slice(&buf[..n]);
                    }
                }
                Err(_) => break,
            }
        }
    });
    (stderr_buf, handle)
}

/// Run iron-bench against the spawned server in CSV mode with
/// --capture-server-request-id. Per Codex plan review v20 P1 #2: --warmup 0
/// (timed-only) is required for join-gate compliance.
fn iron_bench_run_p5h(
    port: u16,
    model_dir: &str,
    prompt_len: u32,
) -> std::io::Result<std::process::Output> {
    Command::new("cargo")
        .args([
            "run",
            "-p",
            "iron-bench",
            "--release",
            "--",
            "--target",
            &format!("p5h_profile=http://127.0.0.1:{port}"),
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
            "--capture-server-request-id",
            "--format",
            "csv",
        ])
        .output()
}

/// Parse the iron-bench CSV stdout and extract the median pp_tps value across
/// all rows (one row per timed run). Returns the median as the per-PP wall-cost
/// metric used for cold/warm variance comparison.
fn parse_pp_tps_median_from_csv(stdout_bytes: &[u8]) -> Option<f64> {
    let csv = String::from_utf8_lossy(stdout_bytes);
    let mut lines = csv.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').collect();
    let pp_tps_idx = cols.iter().position(|&c| c == "pp_tps")?;
    let mut vals: Vec<f64> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if let Some(v) = parts.get(pp_tps_idx).and_then(|s| s.parse::<f64>().ok()) {
            vals.push(v);
        }
    }
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Per T0a.14 rerun #1 diagnosis: single environmental outliers (GC pause, MLX
    // kernel recompile, scheduler glitch) can hit one of 5 runs and pull the
    // median 12%+. Drop the lowest and highest before taking median, leaving the
    // middle (RUNS - 2) values. With RUNS=7 this gives a 5-sample trimmed median.
    if vals.len() < 3 {
        // Fewer than 3 — can't trim; just return median.
        let n = vals.len();
        return Some(if n % 2 == 1 {
            vals[n / 2]
        } else {
            (vals[n / 2 - 1] + vals[n / 2]) / 2.0
        });
    }
    let trimmed = &vals[1..vals.len() - 1];
    let n = trimmed.len();
    Some(if n % 2 == 1 {
        trimmed[n / 2]
    } else {
        (trimmed[n / 2 - 1] + trimmed[n / 2]) / 2.0
    })
}

/// Append captured server stderr + bench CSV to the per-sweep accumulator files
/// at /tmp/p5h-t0a-server.log and /tmp/p5h-t0a-bench.csv. T0a.14 reads these.
fn append_to_sweep_files(
    server_stderr: &[u8],
    bench_csv_stdout: &[u8],
    is_first_call: bool,
) -> std::io::Result<()> {
    // server log — always append (one server spawn per cold/warm cycle).
    let mut server_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(SERVER_LOG_PATH)?;
    server_log.write_all(server_stderr)?;

    // bench CSV — write header on first call, append data rows on subsequent
    // calls. Each iron-bench invocation writes its own header + rows; for the
    // cross-PP sweep we want ONE header + all rows concatenated.
    let bench_csv = String::from_utf8_lossy(bench_csv_stdout);
    let mut lines = bench_csv.lines();
    let header_line = lines.next();
    if is_first_call {
        let mut f = File::create(BENCH_CSV_PATH)?;
        if let Some(h) = header_line {
            writeln!(f, "{h}")?;
        }
        for l in lines {
            writeln!(f, "{l}")?;
        }
    } else {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(BENCH_CSV_PATH)?;
        for l in lines {
            writeln!(f, "{l}")?;
        }
    }
    Ok(())
}

/// One cold spawn + iron-bench timed runs + record outputs. Returns median pp_tps.
fn run_one_pp(pp: u32, is_first_overall: bool) -> anyhow::Result<f64> {
    let model_dir = snapshot_dir();
    assert_port_free(PORT).map_err(|e| anyhow::anyhow!("port {} not free: {}", PORT, e))?;

    let mut server = spawn_server_p5h(&model_dir, PORT);
    let (stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

    if let Err(e) = wait_for_ready(PORT, 90) {
        let _ = server.kill();
        let _ = server.wait();
        let _ = drainer.join();
        anyhow::bail!("PP={pp}: server did not become ready: {e}");
    }

    let out = iron_bench_run_p5h(PORT, &model_dir, pp);

    // Stop server before parsing — give the stderr drainer a moment to catch
    // any post-shutdown emission.
    let _ = server.kill();
    let _ = server.wait();
    let _ = drainer.join();

    let bench_out = out.map_err(|e| anyhow::anyhow!("iron-bench spawn failed: {e}"))?;
    if !bench_out.status.success() {
        anyhow::bail!(
            "PP={pp}: iron-bench failed: stdout={}\nstderr={}",
            String::from_utf8_lossy(&bench_out.stdout),
            String::from_utf8_lossy(&bench_out.stderr),
        );
    }

    let captured_stderr = stderr_buf.lock().map(|g| g.clone()).unwrap_or_default();

    append_to_sweep_files(&captured_stderr, &bench_out.stdout, is_first_overall)
        .map_err(|e| anyhow::anyhow!("failed to append to sweep files: {e}"))?;

    let median = parse_pp_tps_median_from_csv(&bench_out.stdout)
        .ok_or_else(|| anyhow::anyhow!("PP={pp}: could not parse pp_tps from iron-bench CSV"))?;
    Ok(median)
}

fn cool_gate(dur: Duration) {
    eprintln!("[p5h-t0a] cool gate: {} ms", dur.as_millis());
    std::thread::sleep(dur);
}

#[test]
#[ignore = "p5h-t0a — long-running UMA hardening sweep; invoke explicitly"]
fn t0a_uma_hardening_sweep() -> anyhow::Result<()> {
    // Truncate per-sweep accumulator files. First call to run_one_pp re-writes
    // headers; subsequent calls append data rows only.
    let _ = std::fs::remove_file(SERVER_LOG_PATH);
    let _ = std::fs::remove_file(BENCH_CSV_PATH);

    let mut overall_first = true;
    for (i, &pp) in PP_LIST.iter().enumerate() {
        // Per T0a.14 first-sweep diagnosis + T0a.13 reviewer I1: insert an
        // inter-PP cool gate before each PP except the first, so the GPU is
        // genuinely cooled between PPs. Without this, large-PP "cold"
        // measurements are contaminated by GPU heat accumulated from prior
        // PPs (the first sweep saw PP=16384 cold/warm variance 2.6% > 2%
        // threshold because warm was faster than cold — GPU was hotter on
        // "cold" than after the intra-PP cool gate).
        if i > 0 {
            eprintln!("[p5h-t0a] inter-PP cool gate before PP={pp}");
            cool_gate(Duration::from_millis(COOL_DURATION_MS));
        }

        eprintln!("[p5h-t0a] PP={pp}: cold run");
        let cold = run_one_pp(pp, overall_first)?;
        overall_first = false;

        eprintln!("[p5h-t0a] PP={pp}: intra-PP cool gate (cold → warm)");
        cool_gate(Duration::from_millis(COOL_DURATION_MS));

        eprintln!("[p5h-t0a] PP={pp}: warm run");
        let warm = run_one_pp(pp, false)?;

        let variance = (warm - cold).abs() / cold;
        eprintln!("[p5h-t0a] PP={pp}: cold={cold:.2} warm={warm:.2} variance={variance:.3}");
        if variance > VARIANCE_THRESHOLD {
            anyhow::bail!(
                "PP={pp}: cold/warm variance {} > {} threshold (per § 2.4 UMA hardening)",
                variance,
                VARIANCE_THRESHOLD,
            );
        }
    }
    eprintln!("[p5h-t0a] All PPs passed UMA hardening (variance ≤ {VARIANCE_THRESHOLD}).");
    Ok(())
}
