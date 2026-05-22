//! P5h T5 raw-capture sweep — persists [p5h-profile] server stderr + iron-bench
//! CSV (with request_id) to disk for downstream T5.1 aggregator consumption.
//!
//! Per Codex T5 Q-T5-1 A+ direction: T1-T4 harnesses drain stderr in-memory
//! then drop; T5 needs raw spans on disk so the T5.1 aggregator can join
//! `[p5h-profile]` records against iron-bench RequestResults by
//! `request_id` <-> `X-Ironmlx-Request-Id` across the full PP sweep.
//!
//! Output contract (T5.1 aggregator reads both):
//!   * `/tmp/p5h-t5-server.log` — server stderr for the full sweep, appended
//!     per-PP measurement spawn. Preheat stderr is NOT written here (preheat
//!     uses the in-memory drainer in `p5h_common::preheat_to_saturation`).
//!   * `/tmp/p5h-t5-bench.csv` — iron-bench CSV stdout for the full sweep.
//!     Header on first PP only; subsequent PPs append data rows only.
//!
//! Both files are truncated at test start so prior runs don't pollute the
//! aggregator input.
//!
//! Sample shape per PP: single-shot RUNS=7 (T1-T4 pattern). Variance comes
//! from iron-bench's RUNS=7 distribution, not a cold/warm pair — T0a's
//! cold/warm protocol was for UMA hardening at the gate level; T5 is a
//! measurement-data producer and doesn't need it. 6 spawn-kill cycles vs
//! 12 saves ~15min wall.
//!
//! Per-PP spawn-kill topology mirrors T0a/T1/T2/T3/T4. Server stderr goes
//! directly to disk via `Stdio::from(File)` (page-cache-buffered — won't
//! block the way the 64KB pipe does, so no drainer thread needed).
//!
//! No in-harness verdict logic — test returns `Ok(())` on successful capture.
//! Verdict + ROI ranking happens in the T5.1 Python aggregator + T5.2
//! ranking module that consume these two files.
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5h-profile \
//!     --test p5h_t5_attribution_capture -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "p5h-profile")]

mod p5h_common;
use p5h_common::*;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::process::{Child, Command, Stdio};

/// Full PP sweep for T5 attribution capture (per Codex T5 binding).
const T5_PP_LIST: [i32; 6] = [128, 512, 2048, 4096, 8192, 16384];

/// Persisted server stderr — consumed by the T5.1 Python aggregator. One
/// file across the full sweep (per-PP measurement spawns append).
const T5_SERVER_LOG_PATH: &str = "/tmp/p5h-t5-server.log";

/// Persisted iron-bench CSV — consumed by the T5.1 Python aggregator. One
/// file across the full sweep (header on first PP only; data rows append).
const T5_BENCH_CSV_PATH: &str = "/tmp/p5h-t5-bench.csv";

/// Spawn `ironmlx serve` with stderr redirected straight to the per-sweep
/// server log file (append mode). The file handle is opened fresh per
/// spawn — the kernel page cache buffers writes so the server can't block
/// on stderr the way it would on the 64KB pipe (which is why no drainer
/// thread is needed here, unlike `p5h_common::spawn_server`).
fn spawn_server_to_log(model_dir: &str, port: u16) -> std::io::Result<Child> {
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(T5_SERVER_LOG_PATH)?;
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
    // Defensive: never inherit a stray profile mode from caller shell.
    cmd.env_remove("IRONMLX_P5G_PROFILE_MODE");
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    cmd.stderr(Stdio::from(log_file));
    cmd.spawn()
}

/// Run iron-bench with `--capture-server-request-id --format csv --warmup 0`
/// against the spawned server. The capture flag is a switch (no value) per
/// `iron-bench/src/main.rs:82-83` (`default_value_t = false`), and requires
/// `--warmup 0` per the runtime guard at `iron-bench/src/main.rs:117-126`.
fn iron_bench_run_csv(
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
            &format!("p5h_t5=http://127.0.0.1:{port}"),
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
            "csv",
            "--capture-server-request-id",
        ])
        .output()
}

/// Append iron-bench CSV stdout to the per-sweep accumulator file. On the
/// first PP we write header + data rows; subsequent PPs strip the header
/// line and append data rows only (avoids duplicate-header rows that would
/// break the T5.1 aggregator's `pandas.read_csv`).
fn append_bench_csv(stdout_bytes: &[u8], is_first_pp: bool) -> std::io::Result<usize> {
    let csv_text = String::from_utf8_lossy(stdout_bytes);
    let mut lines = csv_text.lines();
    let header = lines.next();
    let mut buf = String::new();
    if is_first_pp {
        if let Some(h) = header {
            buf.push_str(h);
            buf.push('\n');
        }
    }
    for l in lines {
        buf.push_str(l);
        buf.push('\n');
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(T5_BENCH_CSV_PATH)?;
    f.write_all(buf.as_bytes())?;
    f.sync_all()?;
    Ok(buf.len())
}

/// One PP: spawn server (stderr → log file), run iron-bench, kill server,
/// append CSV. Failure paths kill+wait on the server unconditionally.
fn capture_one_pp(model_dir: &str, port: u16, pp: i32, is_first_pp: bool) -> anyhow::Result<()> {
    assert_port_free(port).map_err(|e| anyhow::anyhow!("port {port} not free: {e}"))?;

    // Mark this PP's section in the server log so the aggregator can
    // optionally segment by PP (request_id is the join key, but a PP marker
    // helps diagnose missing-records bugs).
    {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(T5_SERVER_LOG_PATH)?;
        writeln!(f, "[p5h-t5-marker] PP={pp} measurement-start")?;
    }

    let mut server = spawn_server_to_log(model_dir, port)
        .map_err(|e| anyhow::anyhow!("PP={pp}: ironmlx serve spawn failed: {e}"))?;

    if let Err(e) = wait_for_ready(port, 300) {
        let _ = server.kill();
        let _ = server.wait();
        anyhow::bail!("PP={pp}: server not ready: {e}");
    }

    // Detect server that exited before healthz came up (mirrors common helper).
    match server.try_wait() {
        Ok(Some(status)) => {
            anyhow::bail!("PP={pp}: ironmlx serve exited before bench with {status}");
        }
        Ok(None) => {}
        Err(e) => {
            let _ = server.kill();
            let _ = server.wait();
            anyhow::bail!("PP={pp}: try_wait failed: {e}");
        }
    }

    let bench_result = iron_bench_run_csv(port, model_dir, pp);

    // Shutdown server first so stderr file finishes flushing through the OS
    // close path before we move on to the next PP.
    let _ = server.kill();
    let _ = server.wait();

    let bench_out =
        bench_result.map_err(|e| anyhow::anyhow!("PP={pp}: iron-bench spawn failed: {e}"))?;
    if !bench_out.status.success() {
        anyhow::bail!(
            "PP={pp}: iron-bench non-success: stdout={}, stderr={}",
            String::from_utf8_lossy(&bench_out.stdout),
            String::from_utf8_lossy(&bench_out.stderr),
        );
    }

    let appended = append_bench_csv(&bench_out.stdout, is_first_pp)
        .map_err(|e| anyhow::anyhow!("PP={pp}: append_bench_csv failed: {e}"))?;

    eprintln!(
        "[p5h-t5] PP={pp}: server stderr appended to {T5_SERVER_LOG_PATH}; \
         {appended} bytes appended to {T5_BENCH_CSV_PATH}"
    );
    Ok(())
}

#[test]
#[ignore = "p5h-t5 — raw-capture sweep for T5.1 aggregator (~15-20min GPU wall); invoke explicitly"]
fn t5_attribution_capture() -> anyhow::Result<()> {
    let model_dir = snapshot_dir();
    // MLX_DIR is required by `cmd.env("MLX_DIR", ...)` in spawn helpers —
    // surface a clear error early rather than letting an empty value silently
    // make the server fail at MLX init.
    let _mlx_dir = std::env::var("MLX_DIR")
        .map_err(|_| anyhow::anyhow!("set MLX_DIR env var (e.g. $HOME/.local/mlx)"))?;

    // Truncate per-sweep accumulator files BEFORE preheat. Preheat itself
    // uses `p5h_common::preheat_to_saturation` which internally calls
    // `spawn_server` (with `Stdio::piped()` + in-memory drainer) — preheat
    // stderr is captured to a `Vec<u8>` then dropped when the function
    // returns, so preheat does NOT touch either of these files.
    File::create(T5_SERVER_LOG_PATH)
        .map_err(|e| anyhow::anyhow!("truncate {T5_SERVER_LOG_PATH}: {e}"))?;
    File::create(T5_BENCH_CSV_PATH)
        .map_err(|e| anyhow::anyhow!("truncate {T5_BENCH_CSV_PATH}: {e}"))?;
    eprintln!("[p5h-t5] truncated {T5_SERVER_LOG_PATH} + {T5_BENCH_CSV_PATH}");

    // Preheat — drives GPU into thermal saturation. Output discarded by
    // p5h_common; nothing reaches the T5 disk files.
    eprintln!("[p5h-t5] preheat phase (output discarded; T5 disk files untouched)");
    preheat_to_saturation(&model_dir, PROFILE_PORT, &T5_PP_LIST)?;

    // Measurement sweep — one server spawn + one iron-bench batch per PP.
    eprintln!("[p5h-t5] measurement phase (PP ∈ {T5_PP_LIST:?}, RUNS={RUNS}, WARMUP={WARMUP})");
    for (idx, &pp) in T5_PP_LIST.iter().enumerate() {
        let is_first_pp = idx == 0;
        capture_one_pp(&model_dir, PROFILE_PORT, pp, is_first_pp)?;
        std::thread::sleep(INTER_PP_COOLDOWN);
    }

    eprintln!(
        "[p5h-t5] capture complete; T5.1 aggregator consumes {T5_SERVER_LOG_PATH} + \
         {T5_BENCH_CSV_PATH}"
    );
    Ok(())
}
