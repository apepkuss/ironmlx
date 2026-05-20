//! P5g T0 — GatedDeltaNet 4-phase HTTP-path profile harness.
//!
//! Phase A: whole-prefill baseline (server NO profile mode, iron-bench sweep, median pp_tps)
//! Phase B: Layer 1 boundary-isolated (server mode=layer1, iron-bench sweep, parse [p5g-profile] log)
//! Phase C: Layer 2 per-step breakdown (server mode=layer2, iron-bench sweep, parse step_breakdown)
//! Phase D: Layer 3 ablation across 3 pre-defined modes from Step 0.12
//!          (ablate-compute-g, ablate-conv, ablate-t-arr); per-mode pp_tps median
//!          + delta vs Phase A
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5g-profile \
//!       --test p5g_t0_gated_delta_profile \
//!       -- --ignored --test-threads=1 --nocapture
//!
//! Output:
//!   /tmp/p5g-t0-phases.json — full parsed phase data for Step 0.18 report writing.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};

const PP_LIST: [i32; 4] = [2048, 4096, 8192, 16384];
const WARMUP: usize = 1;
const RUNS: usize = 3;
const PROFILE_PORT: u16 = 18080;

const ABLATION_MODES: [&str; 3] = ["ablate-compute-g", "ablate-conv", "ablate-t-arr"];

/// Sleep between per-PP server spawns to give the OS time to release the bound
/// port + free GPU memory used by the prior server's model load.
const INTER_PP_COOLDOWN: Duration = Duration::from_secs(3);

fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

fn output_path() -> PathBuf {
    PathBuf::from("/tmp/p5g-t0-phases.json")
}

/// Median of f64. Returns None on empty input. Panics if NaN present (iron-bench
/// shouldn't emit NaN; failure-fast surfaces upstream measurement bugs).
fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("pp_tps contained NaN"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Spawn `cargo run -p iron-bench` (cross-package, can't use env!("CARGO_BIN_EXE_iron-bench")).
///
/// `max-tokens=1` minimizes decode work per request. Whether the server still
/// dispatches one decode forward (seq=1) depends on `GenerationStream` internal
/// behavior under `max_new_tokens=1`; the aggregator defensively filters any
/// `seq==1` records via `seq > 1` so the outcome is the same either way. T0
/// target is prefill profile only; ship-validation T1/T2/T3 (Steps 1.6-1.8
/// etc.) continue using max-tokens=32 for end-to-end ship metrics.
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
    let v: Value = serde_json::from_str(&s).unwrap_or_else(|e| {
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
    // ALWAYS clear before set — defensive against stray exports in the
    // caller's shell. Phase A (mode=None) must guarantee NO profile, and
    // env inheritance is silently fatal for that invariant.
    cmd.env_remove("IRONMLX_P5G_PROFILE_MODE");
    if let Some(mode) = profile_mode {
        cmd.env("IRONMLX_P5G_PROFILE_MODE", mode);
    }
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    cmd.stderr(Stdio::piped());
    cmd.spawn().expect("ironmlx serve spawn")
}

/// Hard port-free check. Returns Err if the port is already bound — refuses
/// to auto-kill, refuses to silently re-use a stale server's healthz. Done as
/// a TCP bind (not just `lsof`) to test the OS-visible bind constraint that
/// `cargo run ... serve` will see.
fn assert_port_free(port: u16) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    drop(listener); // release immediately so the server can bind next
    Ok(())
}

/// Healthz poll with Result return so the caller can run shutdown + drainer
/// join on failure instead of leaking the Child on panic. Caller must invoke
/// shutdown_and_join on Err.
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
/// Ignores errors — used right before a panic so we don't leak the Child.
fn shutdown_and_join(mut server: Child, drainer: JoinHandle<()>) {
    let _ = server.kill();
    let _ = server.wait();
    let _ = drainer.join();
}

/// Parse `[p5g-profile]` log lines from a stderr byte slice.
/// Each line example (Layer 1):
///   [p5g-profile] mode=layer1 layer=12 batch=1 seq=2048 offset_before=4096 offset_after=6144 elapsed_us=15301
/// Layer 2 additionally has: step_breakdown=us1,us2,us3,...  (single-line append — spec § Step 0.11).
/// Returns one record (k=v map) per matched line. `mode` value is the env-name
/// (`layer1` / `layer2`), NOT the Debug form — `ProfileMode::as_str()` guarantees.
fn parse_profile_log(stderr_bytes: &[u8]) -> Vec<BTreeMap<String, String>> {
    let mut records = Vec::new();
    for line in BufReader::new(stderr_bytes).lines().filter_map(|l| l.ok()) {
        if let Some(rest) = line.split_once("[p5g-profile] ").map(|(_, r)| r) {
            let mut rec: BTreeMap<String, String> = BTreeMap::new();
            for kv in rest.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    rec.insert(k.to_string(), v.to_string());
                }
            }
            records.push(rec);
        }
    }
    records
}

/// Per-PP profile result: median pp_tps + records emitted during that PP's bench window only.
#[derive(Default, serde::Serialize)]
struct PpProfile {
    pp_tps_median: f64,
    records: Vec<BTreeMap<String, String>>,
}

/// Spawn a line-by-line stderr drainer thread. Returns the shared buffer + the
/// thread handle. Drainer terminates only on stderr EOF (server.kill / exit).
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
                Ok(0) => break, // EOF (server exited)
                Ok(_) => buf_clone.lock().unwrap().extend_from_slice(line.as_bytes()),
                Err(_) => break,
            }
        }
    });
    (stderr_buf, drainer)
}

/// Run a phase: for EACH PP, spawn a fresh server, run iron-bench (WARMUP +
/// RUNS requests on that one PP), shutdown server + join drainer, then parse
/// the complete stderr for ALL records in that PP. Per-PP server isolation
/// eliminates per-PP-attribution races (no shared buffer across PPs, no fixed
/// `sleep(N ms)` drain grace) at the cost of (N_PP × model_load) extra time.
///
/// Phase A (mode=None) returns empty records — server emits no `[p5g-profile]`
/// lines without `IRONMLX_P5G_PROFILE_MODE` set.
fn run_phase(mode: Option<&str>, model_dir: &str, port: u16) -> BTreeMap<i32, PpProfile> {
    let mut per_pp: BTreeMap<i32, PpProfile> = BTreeMap::new();

    for &pp in &PP_LIST {
        eprintln!("[p5g-t0] spawning fresh server for mode={mode:?} PP={pp}");

        // Port pre-check — refuse to run against a stale server that might
        // still be bound from a prior failed iteration. Auto-killing would
        // be wrong (might be Boss's manual session); fail loud instead.
        assert_port_free(port).unwrap_or_else(|e| {
            panic!("port {port} not free before spawn for PP={pp}: {e}");
        });

        let mut server = spawn_server(mode, model_dir, port);

        // CRITICAL: start the stderr drainer BEFORE wait_for_ready. Server
        // startup (model load + MLX init + cargo metadata) can emit enough
        // log output to fill the default 64KB pipe buffer; without an active
        // drainer the server blocks on write, healthz never goes up, and
        // wait_for_ready times out. Order: spawn → drainer → wait → bench.
        let (stderr_buf, drainer) = spawn_stderr_drainer(&mut server);

        // wait_for_ready may time out (Err). Caller path: cleanup + panic.
        if let Err(e) = wait_for_ready(port, 300) {
            shutdown_and_join(server, drainer);
            panic!("[p5g-t0] PP={pp}: {e}");
        }

        // Detect server that exited before healthz came up (spawn failed, model
        // not found, etc.). Without this check, wait_for_ready could time out
        // on a dead server and the failure would look like a startup hang.
        match server.try_wait() {
            Ok(Some(status)) => {
                let _ = drainer.join();
                panic!("[p5g-t0] PP={pp}: ironmlx serve exited before bench with {status}");
            }
            Ok(None) => {} // still running — normal
            Err(e) => {
                shutdown_and_join(server, drainer);
                panic!("[p5g-t0] PP={pp}: try_wait failed: {e}");
            }
        }

        let out = match iron_bench_run(port, model_dir, pp) {
            Ok(o) => o,
            Err(e) => {
                // server + drainer already spawned; clean them up before panic.
                shutdown_and_join(server, drainer);
                panic!("[p5g-t0] PP={pp}: iron-bench spawn failed: {e}");
            }
        };
        let bench_ok = out.status.success();
        if !bench_ok {
            eprintln!("[p5g-t0] iron-bench failed at PP={pp}: exit={}", out.status);
            eprintln!("stderr: {}", String::from_utf8_lossy(&out.stderr));
        }

        // Shutdown FIRST — drainer ends on EOF + join completes, then the
        // shared buffer has ALL records from this PP (no race, no drain grace).
        let _ = server.kill();
        let _ = server.wait();
        drainer.join().expect("stderr drainer join");

        if !bench_ok {
            panic!("iron-bench failed at PP={pp}");
        }

        let tps_list = parse_pp_tps_from_bench(&out.stdout);
        let med = median(tps_list).expect("no pp_tps in iron-bench output");

        let stderr_bytes = stderr_buf.lock().unwrap().clone();
        let records = parse_profile_log(&stderr_bytes);
        eprintln!(
            "[p5g-t0] PP={pp} mode={:?}: pp_tps_median={:.2} records={} (post-shutdown parse)",
            mode,
            med,
            records.len()
        );
        per_pp.insert(
            pp,
            PpProfile {
                pp_tps_median: med,
                records,
            },
        );

        // Cooldown — let OS release port + GPU release the prior model's KV
        // before next spawn.
        std::thread::sleep(INTER_PP_COOLDOWN);
    }

    per_pp
}

#[test]
#[ignore]
fn p5g_t0_gated_delta_profile_4phase() {
    let model_dir = snapshot_dir();
    eprintln!("[p5g-t0] starting 4-phase harness; model={model_dir}");

    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    out.insert("pp_list".into(), json!(PP_LIST));
    out.insert("warmup".into(), json!(WARMUP));
    out.insert("runs".into(), json!(RUNS));
    out.insert("model_dir".into(), json!(model_dir));

    // ===== Phase A =====
    eprintln!("[p5g-t0] Phase A: ironmlx serve (NO profile mode) — whole-prefill baseline");
    let phase_a = run_phase(None, &model_dir, PROFILE_PORT);
    out.insert(
        "phase_a_by_pp".into(),
        json!(phase_a
            .iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>()),
    );

    // ===== Phase B =====
    eprintln!("[p5g-t0] Phase B: IRONMLX_P5G_PROFILE_MODE=layer1 — boundary-isolated GDN");
    let phase_b = run_phase(Some("layer1"), &model_dir, PROFILE_PORT);
    out.insert(
        "phase_b_by_pp".into(),
        json!(phase_b
            .iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>()),
    );

    // ===== Phase C =====
    eprintln!("[p5g-t0] Phase C: IRONMLX_P5G_PROFILE_MODE=layer2 — per-step breakdown");
    let phase_c = run_phase(Some("layer2"), &model_dir, PROFILE_PORT);
    out.insert(
        "phase_c_by_pp".into(),
        json!(phase_c
            .iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>()),
    );

    // ===== Phase D =====
    let mut phase_d: BTreeMap<String, BTreeMap<i32, PpProfile>> = BTreeMap::new();
    for &abl_mode in &ABLATION_MODES {
        eprintln!("[p5g-t0] Phase D[{abl_mode}]: IRONMLX_P5G_PROFILE_MODE={abl_mode}");
        let per_pp = run_phase(Some(abl_mode), &model_dir, PROFILE_PORT);
        phase_d.insert(abl_mode.to_string(), per_pp);
    }
    out.insert(
        "phase_d_by_pp".into(),
        json!(phase_d
            .iter()
            .map(|(mode, per_pp)| (
                mode.clone(),
                per_pp
                    .iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect::<BTreeMap<_, _>>()
            ))
            .collect::<BTreeMap<_, _>>()),
    );

    // ===== Write output =====
    let v = serde_json::to_value(&out).expect("serialize phases");
    let json_str = serde_json::to_string_pretty(&v).expect("pretty print");
    let mut f = std::fs::File::create(output_path())
        .unwrap_or_else(|e| panic!("create {}: {e}", output_path().display()));
    f.write_all(json_str.as_bytes()).unwrap();

    eprintln!(
        "[p5g-t0] complete. {} bytes written to {}",
        json_str.len(),
        output_path().display()
    );
    let summarize = |label: &str, m: &BTreeMap<i32, PpProfile>| {
        let tps: BTreeMap<i32, f64> = m.iter().map(|(k, v)| (*k, v.pp_tps_median)).collect();
        let rec_counts: BTreeMap<i32, usize> =
            m.iter().map(|(k, v)| (*k, v.records.len())).collect();
        eprintln!("[p5g-t0] {label} pp_tps_median: {tps:?}");
        eprintln!("[p5g-t0] {label} records_per_pp: {rec_counts:?}");
    };
    summarize("Phase A", &phase_a);
    summarize("Phase B", &phase_b);
    summarize("Phase C", &phase_c);
    for (mode, per_pp) in &phase_d {
        summarize(&format!("Phase D[{mode}]"), per_pp);
    }
}
