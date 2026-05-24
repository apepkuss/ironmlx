//! P5i.c Phase 0 raw-capture sweep — env-var driven, dual-mode (probe +
//! production), multi-PP, multi-repeat. Output contract: per-cell directory
//! `/tmp/p5i-c-phase-0-r${REPEAT}-pp${PP}-${MODE}/{server.log,bench.csv,meta.json}`
//! consumed downstream by `tools/p5h_aggregator/multi_repeat.py` (probe substep
//! CI) + `tools/p5i_c_pp_tps_envelope.py` (pp_tps envelope).
//!
//! Per spec § 4.2.1: this harness exists INSTEAD OF mutating
//! `p5h_t5_attribution_capture.rs` (the P5h T5 validated baseline).
//!
//! P5h+2.b extensions (per spec § 4.2 + § 6): adds three env vars
//! (`P5I_C_SERVER_LIFECYCLE`, `P5I_C_PP_ORDER`, `P5I_C_LOGGING_MODE`) to vary
//! the server lifecycle topology, PP iteration order, and logging mode for the
//! T1/T2 protocol-state matrix sweep. `meta.json` records seven Unix-ns
//! timestamps + warmup_count + server_lifecycle + pp_order + logging_mode so
//! the downstream thermal-overlay + outlier-source analyzers can decompose
//! each cell's wall time deterministically.
//!
//! Env vars (all optional unless marked required):
//!   * `P5I_C_PP_ORDER` — comma-separated PPs, default `"128,512"` (replaces
//!     pre-P5h+2.b `P5I_C_PP_LIST`). Order matters for `same_spawn_cross_pp`.
//!   * `P5I_C_RUNS_PER_PP` — `"PP1:N1,PP2:N2"`, default `"128:7,512:15"`
//!   * `P5I_C_PREHEAT_SECONDS` — wall target, default `"300"`
//!   * `P5I_C_PREHEAT_RUNS` — iron-bench --runs N for preheat, default `"1100"`
//!     (M5 Max: 1100 ≈ 395s; calibrate per hardware)
//!   * `P5I_C_REPEAT_INDEX` — REQUIRED, `"1"|"2"|"3"`
//!   * `P5I_C_MODE` — REQUIRED, `"probe"|"production"`
//!   * `P5I_C_MODEL` — iron-bench --model token (not passed to ironmlx serve;
//!     see Plan Step 2.2), default `"qwen3.5-moe"`
//!   * `P5I_C_MODEL_DIR` — model snapshot dir, default = `IRONMLX_MOE_MODEL_DIR`
//!   * `P5I_C_SERVER_LIFECYCLE` — `"phase0_current"|"same_spawn_cross_pp"|
//!     "same_spawn_per_pp"`, default `"phase0_current"`
//!   * `P5I_C_LOGGING_MODE` — `"default_profile"|"quiet_acceptance"|
//!     "buffered_profile"`, default `"default_profile"`
//!
//! Run example (one cell, legacy behavior preserved with defaults):
//!   P5I_C_REPEAT_INDEX=1 P5I_C_MODE=probe \
//!     IRONMLX_MOE_MODEL_DIR=$SNAP MLX_DIR=$HOME/.local/mlx \
//!     cargo test --release -p ironmlx --features p5h-profile \
//!     --test p5i_c_phase_0_capture -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

use anyhow::Context as _;
use std::collections::HashMap;
use std::fs::{copy as fs_copy, create_dir_all, remove_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::Path as StdPath;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_PP_ORDER: &str = "128,512";
const DEFAULT_RUNS_PER_PP: &str = "128:7,512:15";
const DEFAULT_PREHEAT_SECONDS: &str = "300";
const DEFAULT_PREHEAT_RUNS: &str = "1100";
const DEFAULT_MODEL: &str = "qwen3.5-moe";
const DEFAULT_SERVER_LIFECYCLE: &str = "phase0_current";
const DEFAULT_LOGGING_MODE: &str = "default_profile";
const PORT: u16 = 18099;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_pp_order() -> Vec<i32> {
    let out: Vec<i32> = env_or("P5I_C_PP_ORDER", DEFAULT_PP_ORDER)
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<i32>()
                .expect("P5I_C_PP_ORDER entries must be i32")
        })
        .collect();
    if out.is_empty() {
        panic!("P5I_C_PP_ORDER must contain at least one PP");
    }
    out
}

fn parse_runs_per_pp() -> HashMap<i32, usize> {
    let s = env_or("P5I_C_RUNS_PER_PP", DEFAULT_RUNS_PER_PP);
    s.split(',')
        .filter_map(|pair| {
            let mut it = pair.split(':');
            let pp = it.next()?.trim().parse::<i32>().ok()?;
            let runs = it.next()?.trim().parse::<usize>().ok()?;
            Some((pp, runs))
        })
        .collect()
}

fn parse_repeat_index() -> u32 {
    std::env::var("P5I_C_REPEAT_INDEX")
        .expect("P5I_C_REPEAT_INDEX env var required (e.g. 1, 2, 3)")
        .parse::<u32>()
        .expect("P5I_C_REPEAT_INDEX must be u32")
}

fn parse_mode() -> &'static str {
    match std::env::var("P5I_C_MODE")
        .expect("P5I_C_MODE env var required (probe|production)")
        .as_str()
    {
        "probe" => "probe",
        "production" => "production",
        other => panic!("P5I_C_MODE must be 'probe' or 'production', got {other:?}"),
    }
}

fn iron_bench_model_token() -> String {
    env_or("P5I_C_MODEL", DEFAULT_MODEL)
}

fn ironmlx_model_dir() -> String {
    std::env::var("P5I_C_MODEL_DIR")
        .or_else(|_| std::env::var("IRONMLX_MOE_MODEL_DIR"))
        .expect("set P5I_C_MODEL_DIR or IRONMLX_MOE_MODEL_DIR")
}

fn cell_out_dir(repeat: u32, pp: i32, mode: &str) -> String {
    format!("/tmp/p5i-c-phase-0-r{repeat}-pp{pp}-{mode}")
}

fn preheat_out_dir(repeat: u32, mode: &str) -> String {
    format!("/tmp/p5i-c-phase-0-r{repeat}-preheat-{mode}")
}

fn shared_log_path(repeat: u32, mode: &str) -> String {
    format!("/tmp/p5i-c-phase-0-r{repeat}-shared-{mode}.log")
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ServerLifecycle {
    /// Dedicated preheat spawn + fresh measurement spawn per PP (legacy
    /// P5i.c Phase 0 behavior).
    Phase0Current,
    /// Single server spawn for the entire repeat; preheat once, then measure
    /// every PP in `pp_order` against the same server.
    SameSpawnCrossPp,
    /// One spawn per PP for the entire repeat; preheat inside the same spawn
    /// that measures that PP; kill and respawn between PPs.
    SameSpawnPerPp,
}

fn parse_server_lifecycle() -> ServerLifecycle {
    match env_or("P5I_C_SERVER_LIFECYCLE", DEFAULT_SERVER_LIFECYCLE).as_str() {
        "phase0_current" => ServerLifecycle::Phase0Current,
        "same_spawn_cross_pp" => ServerLifecycle::SameSpawnCrossPp,
        "same_spawn_per_pp" => ServerLifecycle::SameSpawnPerPp,
        other => panic!("P5I_C_SERVER_LIFECYCLE invalid: {other:?}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LoggingMode {
    /// `RUST_LOG` default; server emits full `[p5h-profile]` info lines
    /// directly to the per-cell `server.log` file.
    DefaultProfile,
    /// `RUST_LOG=error`; no `[p5h-profile]` decomposition possible. Probes
    /// HTTP overhead by silencing the info-level write path.
    QuietAcceptance,
    /// `RUST_LOG` default but server stderr piped through a harness-owned
    /// drainer thread that buffers writes to `server.log`. Decoupling
    /// per-line direct file writes from the server process probes whether
    /// the synchronous log emission contributes to wall-time variance.
    BufferedProfile,
}

fn parse_logging_mode() -> LoggingMode {
    match env_or("P5I_C_LOGGING_MODE", DEFAULT_LOGGING_MODE).as_str() {
        "default_profile" => LoggingMode::DefaultProfile,
        "quiet_acceptance" => LoggingMode::QuietAcceptance,
        "buffered_profile" => LoggingMode::BufferedProfile,
        other => panic!("P5I_C_LOGGING_MODE invalid: {other:?}"),
    }
}

fn lifecycle_str(l: ServerLifecycle) -> &'static str {
    match l {
        ServerLifecycle::Phase0Current => "phase0_current",
        ServerLifecycle::SameSpawnCrossPp => "same_spawn_cross_pp",
        ServerLifecycle::SameSpawnPerPp => "same_spawn_per_pp",
    }
}

fn logging_str(l: LoggingMode) -> &'static str {
    match l {
        LoggingMode::DefaultProfile => "default_profile",
        LoggingMode::QuietAcceptance => "quiet_acceptance",
        LoggingMode::BufferedProfile => "buffered_profile",
    }
}

fn now_unix_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn assert_port_free(port: u16) -> std::io::Result<()> {
    use std::net::TcpListener;
    let _ = TcpListener::bind(("127.0.0.1", port))?;
    Ok(())
}

/// Owned server child + optional stderr-drainer JoinHandle (present only when
/// `LoggingMode::BufferedProfile` pipes stderr through the harness).
struct ServerProcess {
    child: Child,
    stderr_drainer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

fn spawn_server_to_log(
    model_dir: &str,
    mode: &str,
    log_path: &str,
    logging_mode: LoggingMode,
) -> std::io::Result<ServerProcess> {
    let bin = env!("CARGO_BIN_EXE_ironmlx");
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model",
        model_dir,
        "--port",
        &PORT.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    if mode == "probe" {
        cmd.arg("--p5h-measurement-eval-probes");
    }
    cmd.env_remove("IRONMLX_P5G_PROFILE_MODE");
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    match logging_mode {
        LoggingMode::QuietAcceptance => {
            cmd.env("RUST_LOG", "error");
            let log_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)?;
            cmd.stderr(Stdio::from(log_file));
        }
        LoggingMode::BufferedProfile => {
            // Drainer thread receives server stderr via pipe and buffers
            // writes to log_path. Direct file writes are removed from the
            // server's hot path under this mode.
            cmd.stderr(Stdio::piped());
        }
        LoggingMode::DefaultProfile => {
            let log_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)?;
            cmd.stderr(Stdio::from(log_file));
        }
    }
    let mut child = cmd.spawn()?;
    let stderr_drainer = if logging_mode == LoggingMode::BufferedProfile {
        let stderr = child
            .stderr
            .take()
            .expect("buffered_profile must produce a stderr pipe");
        let path = log_path.to_string();
        Some(std::thread::spawn(move || {
            use std::io::{BufReader, BufWriter, Write as _};
            let mut reader = BufReader::new(stderr);
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            let mut writer = BufWriter::new(file);
            std::io::copy(&mut reader, &mut writer)?;
            writer.flush()?;
            Ok(())
        }))
    } else {
        None
    };
    Ok(ServerProcess {
        child,
        stderr_drainer,
    })
}

fn wait_for_healthz(timeout_s: u64) -> std::io::Result<()> {
    let start = Instant::now();
    loop {
        let out = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                &format!("http://127.0.0.1:{PORT}/healthz"),
            ])
            .output();
        if let Ok(o) = out {
            if String::from_utf8_lossy(&o.stdout).trim() == "200" {
                return Ok(());
            }
        }
        if start.elapsed().as_secs() >= timeout_s {
            return Err(std::io::Error::other(format!(
                "healthz timeout after {timeout_s}s on port {PORT}"
            )));
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn kill_and_wait(mut server: ServerProcess) {
    let _ = server.child.kill();
    let _ = server.child.wait();
    if let Some(handle) = server.stderr_drainer.take() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("[p5h+2-b WARN] buffered stderr drainer failed: {e}"),
            Err(_) => eprintln!("[p5h+2-b WARN] buffered stderr drainer panicked"),
        }
    }
}

fn monolithic_preheat(
    model_dir: &str,
    preheat_seconds: u64,
    preheat_runs: usize,
) -> std::io::Result<u64> {
    let start = Instant::now();
    let out = Command::new("cargo")
        .args([
            "run",
            "--release",
            "-p",
            "iron-bench",
            "--",
            "--target",
            &format!("preheat=http://127.0.0.1:{PORT}"),
            "--model",
            &iron_bench_model_token(),
            "--model-dir",
            model_dir,
            "--prompt-len",
            "512",
            "--max-tokens",
            "1",
            "--runs",
            &preheat_runs.to_string(),
            "--warmup",
            "0",
            "--format",
            "csv",
        ])
        .output()?;
    let wall_s = start.elapsed().as_secs();
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "preheat non-success: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    if wall_s < preheat_seconds {
        eprintln!(
            "[p5i-c WARN] preheat wall {wall_s}s < target {preheat_seconds}s; \
             consider bumping P5I_C_PREHEAT_RUNS (current {preheat_runs})"
        );
    }
    Ok(wall_s)
}

/// Per-cell timestamp bundle assembled across `run_*` lifecycles before being
/// flushed by `write_cell_meta`. Each field is a Unix-ns instant captured at
/// the corresponding lifecycle event; `now_unix_ns` is fail-soft (0 on clock
/// failure) per `[feedback_performance_stability_priority]`.
#[derive(Debug, Clone, Copy)]
struct CellTimestamps {
    server_spawn_unix_ns: u64,
    server_healthy_unix_ns: u64,
    preheat_start_unix_ns: u64,
    preheat_end_unix_ns: u64,
    measurement_start_unix_ns: u64,
    measurement_end_unix_ns: u64,
    server_kill_unix_ns: u64,
}

#[allow(clippy::too_many_arguments)]
fn write_cell_meta(
    meta_path: &str,
    repeat: u32,
    pp: i32,
    runs: usize,
    mode: &str,
    warmup_count: usize,
    preheat_wall_s: u64,
    server_lifecycle: ServerLifecycle,
    pp_order: &[i32],
    logging_mode: LoggingMode,
    ts: CellTimestamps,
) -> std::io::Result<()> {
    let lifecycle_s = lifecycle_str(server_lifecycle);
    let logging_s = logging_str(logging_mode);
    let pp_order_s = pp_order
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\n  \"repeat\": {repeat},\n  \"pp\": {pp},\n  \"runs\": {runs},\n  \
         \"mode\": \"{mode}\",\n  \"warmup_count\": {warmup_count},\n  \
         \"preheat_wall_s\": {preheat_wall_s},\n  \
         \"server_lifecycle\": \"{lifecycle_s}\",\n  \
         \"pp_order\": \"{pp_order_s}\",\n  \
         \"logging_mode\": \"{logging_s}\",\n  \
         \"server_spawn_unix_ns\": {server_spawn},\n  \
         \"server_healthy_unix_ns\": {server_healthy},\n  \
         \"preheat_start_unix_ns\": {preheat_start},\n  \
         \"preheat_end_unix_ns\": {preheat_end},\n  \
         \"measurement_start_unix_ns\": {measurement_start},\n  \
         \"measurement_end_unix_ns\": {measurement_end},\n  \
         \"server_kill_unix_ns\": {server_kill},\n  \
         \"port\": {PORT}\n}}\n",
        server_spawn = ts.server_spawn_unix_ns,
        server_healthy = ts.server_healthy_unix_ns,
        preheat_start = ts.preheat_start_unix_ns,
        preheat_end = ts.preheat_end_unix_ns,
        measurement_start = ts.measurement_start_unix_ns,
        measurement_end = ts.measurement_end_unix_ns,
        server_kill = ts.server_kill_unix_ns,
    );
    let mut f = File::create(meta_path)?;
    f.write_all(json.as_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// Build the iron-bench CLI args for one cell. Always passes
/// `--capture-run-timestamps` (per spec § 6) so downstream thermal-overlay
/// joins have run_start_unix_ns/run_end_unix_ns columns. `--capture-server-
/// request-id` is added only for probe mode (production-mode CSV would orphan
/// request_ids per Codex review v20 P1 #2).
fn build_iron_args(model_dir: &str, pp: i32, runs: usize, mode: &str) -> Vec<String> {
    let warmup = if mode == "probe" { "0" } else { "1" };
    let mut iron_args: Vec<String> = vec![
        "run".into(),
        "--release".into(),
        "-p".into(),
        "iron-bench".into(),
        "--".into(),
        "--target".into(),
        format!("p5i_c=http://127.0.0.1:{PORT}"),
        "--model".into(),
        iron_bench_model_token(),
        "--model-dir".into(),
        model_dir.into(),
        "--prompt-len".into(),
        pp.to_string(),
        "--max-tokens".into(),
        "1".into(),
        "--runs".into(),
        runs.to_string(),
        "--warmup".into(),
        warmup.into(),
        "--format".into(),
        "csv".into(),
        "--capture-run-timestamps".into(),
    ];
    if mode == "probe" {
        iron_args.push("--capture-server-request-id".into());
    }
    iron_args
}

/// Erase and recreate a directory so append-mode `server.log` never mixes
/// stale records from a previous failed run.
fn reset_dir(path: &str) -> std::io::Result<()> {
    if StdPath::new(path).exists() {
        remove_dir_all(path)?;
    }
    create_dir_all(path)?;
    Ok(())
}

/// Run iron-bench against the (already running) server and persist the CSV.
/// Returns (measurement_start_unix_ns, measurement_end_unix_ns).
fn measure_cell(
    model_dir: &str,
    pp: i32,
    runs: usize,
    mode: &str,
    bench_csv: &str,
) -> anyhow::Result<(u64, u64)> {
    let iron_args = build_iron_args(model_dir, pp, runs, mode);
    let measurement_start = now_unix_ns();
    let bench_out = Command::new("cargo")
        .args(&iron_args)
        .output()
        .map_err(|e| anyhow::anyhow!("PP={pp} mode={mode}: iron-bench spawn: {e}"))?;
    let measurement_end = now_unix_ns();
    if !bench_out.status.success() {
        anyhow::bail!(
            "PP={pp} mode={mode}: iron-bench non-success: stderr={}",
            String::from_utf8_lossy(&bench_out.stderr)
        );
    }
    let mut f = File::create(bench_csv)?;
    f.write_all(&bench_out.stdout)?;
    f.sync_all()?;
    Ok((measurement_start, measurement_end))
}

#[allow(clippy::too_many_arguments)]
fn run_phase0_current(
    repeat: u32,
    mode: &str,
    model_dir: &str,
    pp_order: &[i32],
    runs_map: &HashMap<i32, usize>,
    preheat_seconds: u64,
    preheat_runs: usize,
    lifecycle: ServerLifecycle,
    logging_mode: LoggingMode,
    warmup: usize,
) -> anyhow::Result<()> {
    // Step 1: dedicated preheat spawn (shared across cells in this repeat).
    let preheat_dir = preheat_out_dir(repeat, mode);
    reset_dir(&preheat_dir)?;
    let preheat_log = format!("{preheat_dir}/server.log");

    assert_port_free(PORT).map_err(|e| anyhow::anyhow!("preheat: port {PORT} not free: {e}"))?;

    let preheat_spawn_ns = now_unix_ns();
    let preheat_server = spawn_server_to_log(model_dir, mode, &preheat_log, logging_mode)
        .map_err(|e| anyhow::anyhow!("preheat: spawn failed: {e}"))?;
    if let Err(e) = wait_for_healthz(300) {
        kill_and_wait(preheat_server);
        anyhow::bail!("preheat: healthz: {e}");
    }
    let _preheat_healthy_ns = now_unix_ns();

    let preheat_start_ns = now_unix_ns();
    let preheat_wall = match monolithic_preheat(model_dir, preheat_seconds, preheat_runs) {
        Ok(w) => w,
        Err(e) => {
            kill_and_wait(preheat_server);
            anyhow::bail!("preheat: {e}");
        }
    };
    let preheat_end_ns = now_unix_ns();
    eprintln!("[p5i-c] preheat_wall={preheat_wall}s (target ≥ {preheat_seconds}s)");
    kill_and_wait(preheat_server);
    let _preheat_kill_ns = now_unix_ns();
    let _ = preheat_spawn_ns;

    // Step 2: per-cell fresh spawn per (PP, mode).
    for &pp in pp_order {
        let runs = *runs_map
            .get(&pp)
            .ok_or_else(|| anyhow::anyhow!("no runs configured for PP={pp}"))?;
        let out_dir = cell_out_dir(repeat, pp, mode);
        reset_dir(&out_dir)?;
        let server_log = format!("{out_dir}/server.log");
        let bench_csv = format!("{out_dir}/bench.csv");
        let meta_json = format!("{out_dir}/meta.json");

        assert_port_free(PORT)
            .map_err(|e| anyhow::anyhow!("PP={pp}: port {PORT} not free: {e}"))?;

        let spawn_ns = now_unix_ns();
        let server = spawn_server_to_log(model_dir, mode, &server_log, logging_mode)
            .map_err(|e| anyhow::anyhow!("PP={pp} mode={mode}: spawn failed: {e}"))?;
        if let Err(e) = wait_for_healthz(300) {
            kill_and_wait(server);
            anyhow::bail!("PP={pp} mode={mode}: healthz: {e}");
        }
        let healthy_ns = now_unix_ns();

        let measure_result = measure_cell(model_dir, pp, runs, mode, &bench_csv);

        // Shutdown server FIRST so stderr file flushes cleanly.
        let kill_ns = now_unix_ns();
        kill_and_wait(server);

        let (m_start_ns, m_end_ns) = measure_result?;
        let ts = CellTimestamps {
            server_spawn_unix_ns: spawn_ns,
            server_healthy_unix_ns: healthy_ns,
            preheat_start_unix_ns: preheat_start_ns,
            preheat_end_unix_ns: preheat_end_ns,
            measurement_start_unix_ns: m_start_ns,
            measurement_end_unix_ns: m_end_ns,
            server_kill_unix_ns: kill_ns,
        };
        write_cell_meta(
            &meta_json,
            repeat,
            pp,
            runs,
            mode,
            warmup,
            preheat_wall,
            lifecycle,
            pp_order,
            logging_mode,
            ts,
        )?;
        eprintln!("[p5i-c] PP={pp} mode={mode} repeat={repeat} → {bench_csv}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_same_spawn_cross_pp(
    repeat: u32,
    mode: &str,
    model_dir: &str,
    pp_order: &[i32],
    runs_map: &HashMap<i32, usize>,
    preheat_seconds: u64,
    preheat_runs: usize,
    lifecycle: ServerLifecycle,
    logging_mode: LoggingMode,
    warmup: usize,
) -> anyhow::Result<()> {
    // One server spawn for the entire repeat. Preheat once, then iterate all
    // PPs against the same server. Use a shared log file while alive; copy
    // into each per-cell directory after kill.
    let shared_log = shared_log_path(repeat, mode);
    if StdPath::new(&shared_log).exists() {
        let _ = std::fs::remove_file(&shared_log);
    }
    // Pre-create cell dirs so callers depending on the contract see directories
    // even if a later PP fails; bench.csv + server.log + meta.json are written
    // per-PP below.
    for &pp in pp_order {
        reset_dir(&cell_out_dir(repeat, pp, mode))?;
    }

    assert_port_free(PORT).map_err(|e| anyhow::anyhow!("port {PORT} not free: {e}"))?;
    let spawn_ns = now_unix_ns();
    let server = spawn_server_to_log(model_dir, mode, &shared_log, logging_mode)
        .map_err(|e| anyhow::anyhow!("spawn failed: {e}"))?;
    if let Err(e) = wait_for_healthz(300) {
        kill_and_wait(server);
        anyhow::bail!("healthz: {e}");
    }
    let healthy_ns = now_unix_ns();

    let preheat_start_ns = now_unix_ns();
    let preheat_wall = match monolithic_preheat(model_dir, preheat_seconds, preheat_runs) {
        Ok(w) => w,
        Err(e) => {
            kill_and_wait(server);
            anyhow::bail!("preheat: {e}");
        }
    };
    let preheat_end_ns = now_unix_ns();
    eprintln!("[p5i-c] preheat_wall={preheat_wall}s (target ≥ {preheat_seconds}s)");

    // Iterate PPs against the same server. Collect (pp, runs, m_start, m_end,
    // bench_csv, meta_json) tuples and finalize meta + log copy after kill.
    let mut per_pp_results: Vec<(i32, usize, u64, u64, String, String)> =
        Vec::with_capacity(pp_order.len());
    let mut iter_err: Option<anyhow::Error> = None;
    for &pp in pp_order {
        let runs = match runs_map.get(&pp) {
            Some(r) => *r,
            None => {
                iter_err = Some(anyhow::anyhow!("no runs configured for PP={pp}"));
                break;
            }
        };
        let out_dir = cell_out_dir(repeat, pp, mode);
        let bench_csv = format!("{out_dir}/bench.csv");
        let meta_json = format!("{out_dir}/meta.json");
        match measure_cell(model_dir, pp, runs, mode, &bench_csv) {
            Ok((m_start_ns, m_end_ns)) => {
                per_pp_results.push((pp, runs, m_start_ns, m_end_ns, bench_csv, meta_json));
            }
            Err(e) => {
                iter_err = Some(e);
                break;
            }
        }
    }

    // Shutdown server so the shared log flushes cleanly.
    let kill_ns = now_unix_ns();
    kill_and_wait(server);

    if let Some(e) = iter_err {
        return Err(e);
    }

    // Finalize each cell: copy shared log + write meta.json with shared
    // spawn/healthy/preheat/kill timestamps (identical across PPs) plus
    // per-PP measurement timestamps.
    for (pp, runs, m_start_ns, m_end_ns, bench_csv, meta_json) in per_pp_results {
        let out_dir = cell_out_dir(repeat, pp, mode);
        let dst_log = format!("{out_dir}/server.log");
        fs_copy(&shared_log, &dst_log).with_context(|| {
            format!("PP={pp} same_spawn_cross_pp: copy shared server.log to {dst_log}")
        })?;
        let ts = CellTimestamps {
            server_spawn_unix_ns: spawn_ns,
            server_healthy_unix_ns: healthy_ns,
            preheat_start_unix_ns: preheat_start_ns,
            preheat_end_unix_ns: preheat_end_ns,
            measurement_start_unix_ns: m_start_ns,
            measurement_end_unix_ns: m_end_ns,
            server_kill_unix_ns: kill_ns,
        };
        write_cell_meta(
            &meta_json,
            repeat,
            pp,
            runs,
            mode,
            warmup,
            preheat_wall,
            lifecycle,
            pp_order,
            logging_mode,
            ts,
        )?;
        eprintln!("[p5i-c] PP={pp} mode={mode} repeat={repeat} → {bench_csv} (shared spawn)");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_same_spawn_per_pp(
    repeat: u32,
    mode: &str,
    model_dir: &str,
    pp_order: &[i32],
    runs_map: &HashMap<i32, usize>,
    preheat_seconds: u64,
    preheat_runs: usize,
    lifecycle: ServerLifecycle,
    logging_mode: LoggingMode,
    warmup: usize,
) -> anyhow::Result<()> {
    // Per PP: dedicated spawn + preheat + measure + kill, all inside the same
    // server process. Each PP has its own preheat timestamps.
    for &pp in pp_order {
        let runs = *runs_map
            .get(&pp)
            .ok_or_else(|| anyhow::anyhow!("no runs configured for PP={pp}"))?;
        let out_dir = cell_out_dir(repeat, pp, mode);
        reset_dir(&out_dir)?;
        let server_log = format!("{out_dir}/server.log");
        let bench_csv = format!("{out_dir}/bench.csv");
        let meta_json = format!("{out_dir}/meta.json");

        assert_port_free(PORT)
            .map_err(|e| anyhow::anyhow!("PP={pp}: port {PORT} not free: {e}"))?;

        let spawn_ns = now_unix_ns();
        let server = spawn_server_to_log(model_dir, mode, &server_log, logging_mode)
            .map_err(|e| anyhow::anyhow!("PP={pp} mode={mode}: spawn failed: {e}"))?;
        if let Err(e) = wait_for_healthz(300) {
            kill_and_wait(server);
            anyhow::bail!("PP={pp} mode={mode}: healthz: {e}");
        }
        let healthy_ns = now_unix_ns();

        let preheat_start_ns = now_unix_ns();
        let preheat_wall = match monolithic_preheat(model_dir, preheat_seconds, preheat_runs) {
            Ok(w) => w,
            Err(e) => {
                kill_and_wait(server);
                anyhow::bail!("PP={pp} preheat: {e}");
            }
        };
        let preheat_end_ns = now_unix_ns();
        eprintln!("[p5i-c] PP={pp} preheat_wall={preheat_wall}s (target ≥ {preheat_seconds}s)");

        let measure_result = measure_cell(model_dir, pp, runs, mode, &bench_csv);

        let kill_ns = now_unix_ns();
        kill_and_wait(server);

        let (m_start_ns, m_end_ns) = measure_result?;
        let ts = CellTimestamps {
            server_spawn_unix_ns: spawn_ns,
            server_healthy_unix_ns: healthy_ns,
            preheat_start_unix_ns: preheat_start_ns,
            preheat_end_unix_ns: preheat_end_ns,
            measurement_start_unix_ns: m_start_ns,
            measurement_end_unix_ns: m_end_ns,
            server_kill_unix_ns: kill_ns,
        };
        write_cell_meta(
            &meta_json,
            repeat,
            pp,
            runs,
            mode,
            warmup,
            preheat_wall,
            lifecycle,
            pp_order,
            logging_mode,
            ts,
        )?;
        eprintln!("[p5i-c] PP={pp} mode={mode} repeat={repeat} → {bench_csv} (per-PP spawn)");
    }
    Ok(())
}

#[test]
#[ignore = "p5h+2-b — single-repeat capture with configurable server lifecycle (~6-15 min GPU); invoke explicitly per env vars"]
fn p5i_c_phase_0_capture_one_repeat() -> anyhow::Result<()> {
    let repeat = parse_repeat_index();
    let mode = parse_mode();
    let model_dir = ironmlx_model_dir();
    let pp_order = parse_pp_order();
    let runs_map = parse_runs_per_pp();
    let preheat_seconds: u64 = env_or("P5I_C_PREHEAT_SECONDS", DEFAULT_PREHEAT_SECONDS)
        .parse()
        .expect("P5I_C_PREHEAT_SECONDS must be u64");
    let preheat_runs: usize = env_or("P5I_C_PREHEAT_RUNS", DEFAULT_PREHEAT_RUNS)
        .parse()
        .expect("P5I_C_PREHEAT_RUNS must be usize");
    let lifecycle = parse_server_lifecycle();
    let logging_mode = parse_logging_mode();
    let warmup = if mode == "probe" { 0_usize } else { 1_usize };

    eprintln!(
        "[p5h+2-b] repeat={repeat} mode={mode} lifecycle={lifecycle:?} \
         pp_order={pp_order:?} runs={runs_map:?} logging={logging_mode:?} \
         preheat_target_s={preheat_seconds}"
    );

    match lifecycle {
        ServerLifecycle::Phase0Current => run_phase0_current(
            repeat,
            mode,
            &model_dir,
            &pp_order,
            &runs_map,
            preheat_seconds,
            preheat_runs,
            lifecycle,
            logging_mode,
            warmup,
        ),
        ServerLifecycle::SameSpawnCrossPp => run_same_spawn_cross_pp(
            repeat,
            mode,
            &model_dir,
            &pp_order,
            &runs_map,
            preheat_seconds,
            preheat_runs,
            lifecycle,
            logging_mode,
            warmup,
        ),
        ServerLifecycle::SameSpawnPerPp => run_same_spawn_per_pp(
            repeat,
            mode,
            &model_dir,
            &pp_order,
            &runs_map,
            preheat_seconds,
            preheat_runs,
            lifecycle,
            logging_mode,
            warmup,
        ),
    }
}
