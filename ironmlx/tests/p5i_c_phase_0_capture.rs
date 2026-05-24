//! P5i.c Phase 0 raw-capture sweep — env-var driven, dual-mode (probe +
//! production), multi-PP, multi-repeat. Output contract: per-cell directory
//! `/tmp/p5i-c-phase-0-r${REPEAT}-pp${PP}-${MODE}/{server.log,bench.csv,meta.json}`
//! consumed downstream by `tools/p5h_aggregator/multi_repeat.py` (probe substep
//! CI) + `tools/p5i_c_pp_tps_envelope.py` (pp_tps envelope).
//!
//! Per spec § 4.2.1: this harness exists INSTEAD OF mutating
//! `p5h_t5_attribution_capture.rs` (the P5h T5 validated baseline).
//!
//! Env vars (all optional unless marked required):
//!   * `P5I_C_PP_LIST` — comma-separated PPs, default `"128,512"`
//!   * `P5I_C_RUNS_PER_PP` — `"PP1:N1,PP2:N2"`, default `"128:7,512:15"`
//!   * `P5I_C_PREHEAT_SECONDS` — wall target, default `"300"`
//!   * `P5I_C_PREHEAT_RUNS` — iron-bench --runs N for preheat, default `"1100"`
//!     (M5 Max: 1100 ≈ 395s; calibrate per hardware)
//!   * `P5I_C_REPEAT_INDEX` — REQUIRED, `"1"|"2"|"3"`
//!   * `P5I_C_MODE` — REQUIRED, `"probe"|"production"`
//!   * `P5I_C_MODEL` — iron-bench --model token (not passed to ironmlx serve;
//!     see Plan Step 2.2), default `"qwen3.5-moe"`
//!   * `P5I_C_MODEL_DIR` — model snapshot dir, default = `IRONMLX_MOE_MODEL_DIR`
//!
//! Run example (one cell):
//!   P5I_C_REPEAT_INDEX=1 P5I_C_MODE=probe \
//!     IRONMLX_MOE_MODEL_DIR=$SNAP MLX_DIR=$HOME/.local/mlx \
//!     cargo test --release -p ironmlx --features p5h-profile \
//!     --test p5i_c_phase_0_capture -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

use std::collections::HashMap;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_PP_LIST: &str = "128,512";
const DEFAULT_RUNS_PER_PP: &str = "128:7,512:15";
const DEFAULT_PREHEAT_SECONDS: &str = "300";
const DEFAULT_PREHEAT_RUNS: &str = "1100";
const DEFAULT_MODEL: &str = "qwen3.5-moe";
const PORT: u16 = 18099;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_pp_list() -> Vec<i32> {
    env_or("P5I_C_PP_LIST", DEFAULT_PP_LIST)
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .collect()
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

fn assert_port_free(port: u16) -> std::io::Result<()> {
    use std::net::TcpListener;
    let _ = TcpListener::bind(("127.0.0.1", port))?;
    Ok(())
}

fn spawn_server_to_log(model_dir: &str, mode: &str, log_path: &str) -> std::io::Result<Child> {
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
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
    cmd.stderr(Stdio::from(log_file));
    cmd.spawn()
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

fn kill_and_wait(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn monolithic_preheat(model_dir: &str, preheat_seconds: u64) -> std::io::Result<u64> {
    let preheat_runs = env_or("P5I_C_PREHEAT_RUNS", DEFAULT_PREHEAT_RUNS)
        .parse::<usize>()
        .expect("P5I_C_PREHEAT_RUNS must be usize");

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

fn capture_one_cell(
    model_dir: &str,
    pp: i32,
    runs: usize,
    mode: &str,
    repeat: u32,
    preheat_wall_s: u64,
) -> anyhow::Result<()> {
    let out_dir = cell_out_dir(repeat, pp, mode);
    create_dir_all(&out_dir)?;

    let server_log = format!("{out_dir}/server.log");
    let bench_csv = format!("{out_dir}/bench.csv");
    let meta_json = format!("{out_dir}/meta.json");

    assert_port_free(PORT).map_err(|e| anyhow::anyhow!("port {PORT} not free: {e}"))?;

    let server = spawn_server_to_log(model_dir, mode, &server_log)
        .map_err(|e| anyhow::anyhow!("PP={pp} mode={mode}: spawn failed: {e}"))?;

    if let Err(e) = wait_for_healthz(300) {
        kill_and_wait(server);
        anyhow::bail!("PP={pp} mode={mode}: healthz: {e}");
    }

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
    ];
    if mode == "probe" {
        iron_args.push("--capture-server-request-id".into());
    }

    let bench_result = Command::new("cargo").args(&iron_args).output();

    // Shutdown server FIRST so stderr file flushes cleanly.
    kill_and_wait(server);

    let bench_out =
        bench_result.map_err(|e| anyhow::anyhow!("PP={pp} mode={mode}: iron-bench spawn: {e}"))?;
    if !bench_out.status.success() {
        anyhow::bail!(
            "PP={pp} mode={mode}: iron-bench non-success: stderr={}",
            String::from_utf8_lossy(&bench_out.stderr)
        );
    }

    let mut f = File::create(&bench_csv)?;
    f.write_all(&bench_out.stdout)?;
    f.sync_all()?;

    // meta.json: cell metadata for downstream Python verification (per plan
    // Step 3.3 acceptance script + close-out reporting).
    let meta = format!(
        "{{\n  \"repeat\": {repeat},\n  \"pp\": {pp},\n  \"runs\": {runs},\n  \"mode\": \"{mode}\",\n  \"preheat_wall_s\": {preheat_wall_s},\n  \"port\": {PORT}\n}}\n"
    );
    let mut mf = File::create(&meta_json)?;
    mf.write_all(meta.as_bytes())?;
    mf.sync_all()?;

    eprintln!("[p5i-c] PP={pp} mode={mode} repeat={repeat} → {bench_csv}");
    Ok(())
}

#[test]
#[ignore = "p5i-c Phase 0 — single-repeat capture (~6-8 min GPU); invoke explicitly per env vars"]
fn p5i_c_phase_0_capture_one_repeat() -> anyhow::Result<()> {
    let repeat = parse_repeat_index();
    let mode = parse_mode();
    let model_dir = ironmlx_model_dir();
    let pp_list = parse_pp_list();
    let runs_map = parse_runs_per_pp();
    let preheat_seconds: u64 = env_or("P5I_C_PREHEAT_SECONDS", DEFAULT_PREHEAT_SECONDS)
        .parse()
        .expect("P5I_C_PREHEAT_SECONDS must be u64");

    eprintln!(
        "[p5i-c] repeat={repeat} mode={mode} pp_list={pp_list:?} runs={runs_map:?} \
         preheat_target_s={preheat_seconds}"
    );

    // Monolithic preheat: dedicated spawn that runs iron-bench long enough to
    // reach thermal saturation per `[project_p5h_2a_findings]` (M5 Max: 300s
    // wall requires --runs 1100). Plan Step 2.1 requires creating the preheat
    // out dir before the preheat server spawn.
    let preheat_dir = preheat_out_dir(repeat, mode);
    create_dir_all(&preheat_dir)?;
    let preheat_log = format!("{preheat_dir}/server.log");

    assert_port_free(PORT).map_err(|e| anyhow::anyhow!("preheat: port {PORT} not free: {e}"))?;
    let preheat_server = spawn_server_to_log(&model_dir, mode, &preheat_log)
        .map_err(|e| anyhow::anyhow!("preheat: spawn failed: {e}"))?;

    if let Err(e) = wait_for_healthz(300) {
        kill_and_wait(preheat_server);
        anyhow::bail!("preheat: healthz: {e}");
    }
    let preheat_wall = match monolithic_preheat(&model_dir, preheat_seconds) {
        Ok(w) => w,
        Err(e) => {
            kill_and_wait(preheat_server);
            anyhow::bail!("preheat: {e}");
        }
    };
    eprintln!("[p5i-c] preheat_wall={preheat_wall}s (target ≥ {preheat_seconds}s)");
    kill_and_wait(preheat_server);

    // Per-cell capture: fresh spawn per (PP, mode) cell — each cell records
    // preheat_wall_s in meta.json so downstream verification can fail on
    // insufficient preheat.
    for &pp in &pp_list {
        let runs = *runs_map
            .get(&pp)
            .ok_or_else(|| anyhow::anyhow!("no runs configured for PP={pp}"))?;
        capture_one_cell(&model_dir, pp, runs, mode, repeat, preheat_wall)?;
    }

    Ok(())
}
