# P5h+2.d Thermal / Residual-Variance Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Investigate residual within-sweep variance mechanism (H1 thermal family) exposed after P5h+2.c scheduler fix; produce evidence + recommendation deciding whether Phase 0 § 7 #4 production envelope ≤ ±2% backfills PASS.

**Architecture:** Two-gate framework. T0+T1 ship new iron-bench cooldown CLI flag + aggregator diagnostic fields. T2 runs Stage 1 ironmlx cooldown matrix `{0,60,120}s × {PP=128,PP=512} × 3 repeats = 18 cells` against the Mechanism gate; T3 (gated on T2 strong/weak yes) runs Stage 2 sudo powermetrics overlay; T4 (gated on T2 strong/weak yes) runs omlx BEST+WORST control. T5 packages all infra + tests + docs + Phase 0 PASS backfill or deferred/failed-attempt note into single commit per spec § 9.4.

**Tech Stack:** Rust (iron-bench Args + tokio sleep), Python (aggregator + orchestrator), pytest, cargo test, `sudo powermetrics`, `iron-bench --target omlx-baseline`. Reuses `tools/p5h_2b_protocol_experiment.py` (driver) + `tools/p5h_2b_thermal_overlay.py` (Stage 2; existing pytests extended for plist) + `tools/p5i_c_pp_tps_envelope.py` (envelope; extended).

**Spec:** `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md` (commit `e9dd6ad`).

**Predecessor close-outs:** `docs/p5h+2-b-close-out.md` § 9-10, `docs/p5h+2-c-close-out.md`, `docs/p5i-c-phase-0-close-out.md` § 1 #4.

**Branch:** Implementation fork to `ironmlx-p5h+2-d-thermal-investigation` off current HEAD `e9dd6ad` BEFORE T0 begins. Boss pushes current `ironmlx-p5h+2-c-scheduler-finished-fix` branch first.

**Single-commit discipline (spec § 9.4):** T0-T4 produce WIP only. T5 makes ONE commit attaching all changes + close-out doc. Each non-T5 task ends with "Stop and report DONE; DO NOT commit".

---

## File Structure

| Path | Role | Touched in task |
|---|---|---|
| `iron-bench/src/main.rs` | Args struct + v1/v2 dispatch + validation | T0 (modify) |
| `iron-bench/src/runner.rs` | Sequential measured-run loop (insert cooldown sleep) | T0 (modify) |
| `iron-bench/tests/inter_run_cooldown_secs.rs` (NEW) | Async integration timing test | T0 (create) |
| `ironmlx/tests/p5i_c_phase_0_capture.rs` | Capture harness `build_iron_args` + env-var reader | T0 (modify) |
| `tools/p5h_2b_protocol_experiment.py` | Existing driver — add `--inter-run-cooldown-secs` arg + env pass-through, Rule D server-log scan, and `--expected-runs` envelope pass-through | T0/T1 (modify) |
| `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py` (NEW) | Driver smoke pytest verifying env-var passthrough + server-log scan behavior | T0 (create) |
| `tools/p5i_c_pp_tps_envelope.py` | Add diagnostic fields + flexible `EXPECTED_RUNS` | T1 (modify) |
| `tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py` | Extend with 3 new diagnostic-field tests | T1 (modify) |
| `tools/p5h_2d_thermal_experiment.py` (NEW) | Thin orchestrator — cooldown loop wrapping `p5h_2b_protocol_experiment.py` + Mechanism gate analyzer | T2 (create) |
| `tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py` (NEW) | Mechanism gate analyzer unit tests | T2 (create) |
| `tools/p5h_2b_thermal_overlay.py` | Existing thermal overlay joiner — extend with plist parsing for local powermetrics output | T3 (modify) |
| `docs/p5h+2-d-close-out.md` (NEW) | T5 close-out with Mechanism + Acceptance gate verdicts + Phase 0 backfill | T5 (create) |
| `docs/p5i-c-phase-0-close-out.md` | Phase 0 backfill per § 1.3 outcome matrix | T5 (modify) |
| `docs/p5i-c-phase-0-ranking-snapshot.md` | Preamble status update | T5 (modify) |

---

## Predeclared exclusion rules (lock BEFORE T2 begins; spec § 7)

- **Rule B** (drop first 1-2 cold-start runs): kept for envelope NUMBER trim only; pattern analyzer (trailing_slowdown_pct / fast_start_drop_pct) operates on RAW per-run series.
- **Rule C** (conditional drop last 2): REMOVED.
- **Rule D**: any server.log ERROR line → cell FAILS hard-stop (driver-level); WARN → check allow-list; non-allow-listed WARN → cell marked for human review (NOT auto-drop).

Non-scheduler WARN allow-list (spec § 7.1):
- `[tracing]` initialization warnings on first run
- KVCache buffer-resize warnings under PP=128
- `mlx::eval` lazy-materialization info-level spam (already filtered by `quiet_acceptance`)

---

## Task 0: iron-bench `--inter-run-cooldown-secs` flag + harness/driver pass-through

**Files:**
- Modify: `iron-bench/src/main.rs` Args struct (after `capture_run_timestamps` field; ~line 80-95) + v1 dispatch (~line 200-220) + v2 validation block (~line 145-155)
- Modify: `iron-bench/src/runner.rs` `run_cell()` signature + measured loop (~line 71-130)
- Create: `iron-bench/tests/inter_run_cooldown_secs.rs`
- Modify: `ironmlx/tests/p5i_c_phase_0_capture.rs` (`env_or` reader + `build_iron_args` push)
- Modify: `tools/p5h_2b_protocol_experiment.py` argparse + `run_one_repeat` env block
- Create: `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py`

### T0.A — iron-bench Rust flag (TDD)

- [ ] **Step A1: Write failing iron-bench integration timing test**

Create `iron-bench/tests/inter_run_cooldown_secs.rs`:

```rust
//! Verifies `--inter-run-cooldown-secs N` adds inter-run sleep only between
//! measured sequential runs, and is rejected for concurrent mode.
//!
//! Mirrors `concurrent_smoke.rs`: launches an in-process OpenAI-compatible
//! SSE mock server and invokes the built iron-bench binary.

use std::{convert::Infallible, process::Command, time::{Duration, Instant}};

use axum::{response::sse::Event, routing::post, Router};
use tokio::net::TcpListener;
use tokio_stream::wrappers::ReceiverStream;

async fn mock_sse_handler() -> axum::response::sse::Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let body = serde_json::json!({
            "id": "mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": { "content": "tok" },
                "finish_reason": null,
            }],
        });
        let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
        let usage_body = serde_json::json!({
            "id": "mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 1,
                "cached_tokens": 0,
            },
        });
        let _ = tx.send(Ok(Event::default().data(usage_body.to_string()))).await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    axum::response::sse::Sse::new(ReceiverStream::new(rx))
}

fn tokenizer_fixture_dir() -> Option<std::path::PathBuf> {
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    if fixture_dir.join("tokenizer.json").exists() {
        Some(fixture_dir)
    } else {
        eprintln!(
            "[cooldown] tokenizer fixture missing at {}; skipping timing assertion",
            fixture_dir.join("tokenizer.json").display()
        );
        None
    }
}

fn run_iron_bench(bench_bin: &str, url: &str, model_dir: &str, cooldown_secs: &str) -> Duration {
    let start = Instant::now();
    let output = Command::new(bench_bin)
        .args([
            "--target",
            &format!("mock={url}"),
            "--model-dir",
            model_dir,
            "--model",
            "mock",
            "--prompt-len",
            "16",
            "--max-tokens",
            "1",
            "--runs",
            "2",
            "--warmup",
            "0",
            "--inter-run-cooldown-secs",
            cooldown_secs,
            "--format",
            "csv",
        ])
        .output()
        .expect("spawn iron-bench");
    let elapsed = start.elapsed();
    assert!(
        output.status.success(),
        "iron-bench exited non-zero:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    elapsed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cooldown_inserts_sleep_between_measured_runs_only() {
    let Some(fixture_dir) = tokenizer_fixture_dir() else {
        return;
    };
    let app = Router::new().route("/v1/chat/completions", post(mock_sse_handler));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });

    let bench_bin = env!("CARGO_BIN_EXE_iron-bench");
    let url = format!("http://{addr}");
    let model_dir = fixture_dir.to_str().expect("utf-8 fixture_dir");

    let no_cooldown = run_iron_bench(bench_bin, &url, model_dir, "0");
    let one_second = run_iron_bench(bench_bin, &url, model_dir, "1");
    let delta = one_second.saturating_sub(no_cooldown);
    assert!(
        delta >= Duration::from_millis(800),
        "expected cooldown=1 to add roughly one inter-run sleep; \
         no_cooldown={no_cooldown:?} cooldown_1s={one_second:?} delta={delta:?}",
    );
}

#[test]
fn cooldown_rejects_concurrent_mode_when_nonzero() {
    let bin = env!("CARGO_BIN_EXE_iron-bench");
    let out = Command::new(bin)
        .args([
            "--target",
            "bogus=http://127.0.0.1:1",
            "--model",
            "x",
            "--model-dir",
            "/tmp/nonexistent",
            "--prompt-len",
            "16",
            "--max-tokens",
            "1",
            "--concurrent",
            "2",
            "--duration",
            "1",
            "--warmup-duration",
            "0",
            "--inter-run-cooldown-secs",
            "1",
            "--format",
            "csv",
        ])
        .output()
        .expect("iron-bench spawn");
    assert_ne!(out.status.code(), Some(0), "expected non-zero exit");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("inter-run-cooldown-secs")
            && (combined.contains("concurrent") || combined.contains("v2")),
        "expected validation error mentioning inter-run-cooldown-secs + concurrent/v2; got: {combined}"
    );
}
```

- [ ] **Step A2: Run test to confirm failure**

```bash
cargo test --release -p iron-bench --test inter_run_cooldown_secs
```

Expected: `cooldown_rejects_concurrent_mode_when_nonzero` FAILS with clap unknown-flag output. `cooldown_inserts_sleep_between_measured_runs_only` also FAILS with clap unknown-flag output when the tokenizer fixture is present; otherwise it self-skips.

- [ ] **Step A3: Add Args field**

In `iron-bench/src/main.rs`, after the `capture_run_timestamps` field (before `format` field), append:

```rust
    /// Sleep N seconds between measured runs in sequential (v1) mode.
    /// Does NOT sleep during preheat or warmup; does NOT sleep after the
    /// final measured run. Default 0 (no behavior change).
    /// Per P5h+2.d spec § 3.1 — production-grade flag for any sweep where
    /// thermal isolation between consecutive measured runs matters.
    #[arg(long, default_value_t = 0u64)]
    pub inter_run_cooldown_secs: u64,
```

- [ ] **Step A4: Add v2-concurrent-mode rejection**

In `iron-bench/src/main.rs`, find the existing `args.concurrent.is_some() && args.warmup_duration != 0` validation block (~line 147) and add an analogous block immediately after it:

```rust
        if args.concurrent.is_some() && args.inter_run_cooldown_secs != 0 {
            anyhow::bail!(
                "--inter-run-cooldown-secs is incompatible with concurrent (v2) mode \
                 (per P5h+2.d spec § 3.1): cooldown semantics are defined only for \
                 the sequential measured-run loop. Set --inter-run-cooldown-secs 0 \
                 when using --concurrent."
            );
        }
```

- [ ] **Step A5: Plumb arg through `run_cell` signature**

In `iron-bench/src/runner.rs`, locate the `pub async fn run_cell(...)` signature (~line 65). Add `inter_run_cooldown_secs: u64,` after `capture_run_timestamps: bool,`:

```rust
pub async fn run_cell(
    client: &reqwest::Client,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup: usize,
    runs: usize,
    capture_request_id: bool,
    capture_run_timestamps: bool,
    inter_run_cooldown_secs: u64,
    tokenizer: &Tokenizer,
) -> Result<CellResult> {
```

- [ ] **Step A6: Insert cooldown sleep in measured loop**

In the SAME `runner.rs` `run_cell`, inside the `for i in 0..runs` loop, AFTER the run is recorded but BEFORE the `}` of the loop body, insert:

```rust
        if inter_run_cooldown_secs > 0 && i + 1 < runs {
            tokio::time::sleep(std::time::Duration::from_secs(inter_run_cooldown_secs)).await;
        }
```

The `i + 1 < runs` guard skips the sleep after the FINAL measured run (spec § 3.1).

- [ ] **Step A7: Update v1 dispatch call site**

In `iron-bench/src/main.rs` v1 sequential dispatch block (~line 200-220), inside the `for pp in &args.prompt_len { for (target_name, target_url) in &args.target { runner::run_cell( ... ).await? }` call, add `args.inter_run_cooldown_secs,` after `args.capture_run_timestamps,`:

```rust
                    let cell = runner::run_cell(
                        &client,
                        target_name,
                        target_url,
                        &args.model,
                        *pp,
                        args.max_tokens,
                        args.warmup,
                        args.runs,
                        args.capture_server_request_id,
                        args.capture_run_timestamps,
                        args.inter_run_cooldown_secs,
                        &tokenizer,
                    )
                    .await?;
```

- [ ] **Step A8: Run tests to confirm pass**

```bash
cargo test --release -p iron-bench --test inter_run_cooldown_secs
```

Expected: BOTH tests PASS.

- [ ] **Step A9: Run full iron-bench test suite for regression**

```bash
cargo test --release -p iron-bench
```

Expected: ALL tests PASS (no regression from new arg / signature change).

- [ ] **Step A10: Run cargo fmt + clippy gates**

```bash
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace -- -D warnings && cargo build --release
```

Expected: zero warnings, zero errors.

### T0.B — Capture harness env-var pass-through

- [ ] **Step B1: Add env-var reader + arg push to `build_iron_args`**

In `ironmlx/tests/p5i_c_phase_0_capture.rs`, locate `fn build_iron_args(model_dir: &str, pp: i32, runs: usize, mode: &str) -> Vec<String>` (~line 437). Inside the function, AFTER the `iron_args.push("--capture-run-timestamps".into());` line (the existing final push at the end of the vec init), add a conditional push for cooldown:

```rust
    let cooldown = env_or("P5I_C_INTER_RUN_COOLDOWN_SECS", "0");
    if cooldown != "0" {
        iron_args.push("--inter-run-cooldown-secs".into());
        iron_args.push(cooldown);
    }
```

Note: existing `env_or` helper at line 60 returns `String`; "0" check avoids passing the flag when caller didn't set the env var (preserves preheat byte-identity).

- [ ] **Step B2: Update file doc-comment header**

In `ironmlx/tests/p5i_c_phase_0_capture.rs` at the top doc comment block (~line 22-35 where `P5I_C_PREHEAT_RUNS` and `P5I_C_MODEL` are listed), add:

```rust
//!   * `P5I_C_INTER_RUN_COOLDOWN_SECS` — iron-bench `--inter-run-cooldown-secs N`
//!     for measured cells only (NOT applied to preheat), default `"0"`.
```

- [ ] **Step B3: Verify preheat is NOT affected (defensive read)**

Read `ironmlx/tests/p5i_c_phase_0_capture.rs` `fn monolithic_preheat(...)` (~line 322). Confirm it builds its iron-bench command inline (does NOT call `build_iron_args`). Cooldown env-var addition therefore does NOT touch preheat. No code change required; this is a verification step.

- [ ] **Step B4: cargo gates**

```bash
cargo fmt && cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace -- -D warnings && cargo build --release && cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture -- --list
```

Expected: zero warnings; release build succeeds; P5h profile capture test binary compiles and lists tests.

### T0.C — Python driver pass-through + smoke pytest (TDD)

- [ ] **Step C1: Write failing smoke pytest**

Create `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py`:

```python
"""Smoke pytest for tools/p5h_2b_protocol_experiment.py P5h+2.d cooldown
pass-through. Verifies --inter-run-cooldown-secs is propagated to the
P5I_C_INTER_RUN_COOLDOWN_SECS env var the capture harness reads, and Rule D
server-log scan hard-fails ERROR lines while surfacing non-allow-listed WARNs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

import p5h_2b_protocol_experiment as drv  # noqa: E402


def _base_argv() -> list[str]:
    return [
        "p5h_2b_protocol_experiment.py",
        "--phase", "t1",
        "--exp-id", "smoke",
        "--server-lifecycle", "same_spawn_per_pp",
        "--pp-order", "128",
        "--logging-mode", "quiet_acceptance",
        "--mode", "production",
        "--repeats", "1",
        "--pps", "128",
        "--runs-per-pp", "128:15",
        "--model-dir", "/tmp/model",
        "--mlx-dir", "/tmp/mlx",
        "--out-base", "/tmp/p5h-2b-smoke",
        "--skip-envelope",
    ]


def test_cooldown_env_var_propagated_when_flag_set():
    """If --inter-run-cooldown-secs is set, the env passed to cargo test
    must include P5I_C_INTER_RUN_COOLDOWN_SECS=<N>."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1  # force run_one_repeat to raise before relocate
        return result

    argv = _base_argv() + ["--inter-run-cooldown-secs", "60"]
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert captured_env.get("P5I_C_INTER_RUN_COOLDOWN_SECS") == "60", (
        f"expected env P5I_C_INTER_RUN_COOLDOWN_SECS=60; "
        f"got {captured_env.get('P5I_C_INTER_RUN_COOLDOWN_SECS')!r}"
    )


def test_cooldown_env_var_absent_when_flag_not_set():
    """Default 0 cooldown: env must NOT have P5I_C_INTER_RUN_COOLDOWN_SECS
    (preserves harness byte-identity for legacy invocations)."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1
        return result

    argv = _base_argv()
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert "P5I_C_INTER_RUN_COOLDOWN_SECS" not in captured_env, (
        f"expected no P5I_C_INTER_RUN_COOLDOWN_SECS env when flag absent; "
        f"got {captured_env.get('P5I_C_INTER_RUN_COOLDOWN_SECS')!r}"
    )


def test_scan_server_log_hard_fails_any_error(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "2026-05-25T00:00:00Z INFO ok\n"
        "2026-05-25T00:00:01Z ERROR non-scheduler failure\n"
    )
    with pytest.raises(SystemExit) as exc:
        drv.scan_server_log(log, allow_server_errors=False)
    assert "ERROR lines detected" in str(exc.value)


def test_scan_server_log_marks_non_allowlisted_warn(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "2026-05-25T00:00:00Z WARN unexpected thermal warning\n"
        "2026-05-25T00:00:01Z WARN [tracing] initialization warning\n"
    )
    scan = drv.scan_server_log(log, allow_server_errors=False)
    assert scan["error_count"] == 0
    assert scan["allowlisted_warn_count"] == 1
    assert scan["non_allowlisted_warn_count"] == 1
    assert "unexpected thermal warning" in scan["non_allowlisted_warn_lines"][0]
```

- [ ] **Step C2: Run pytest to confirm failure**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py -v
```

Expected: cooldown env-var test with the flag FAILS with "unrecognized arguments: --inter-run-cooldown-secs"; server-log scan tests FAIL with `AttributeError: module ... has no attribute 'scan_server_log'`. The default-cooldown env-var test may already pass.

- [ ] **Step C3: Add argparse arg + env propagation + Rule D scan**

In `tools/p5h_2b_protocol_experiment.py`:

(a) Add `import json` near the top with the other imports:

```python
import json
```

(b) Replace `check_no_scheduler_errors(...)` with the broader Rule D scanner:

```python
ALLOWLISTED_WARN_SUBSTRINGS = (
    "[tracing]",
    "KVCache",
    "buffer-resize",
    "mlx::eval",
)


def scan_server_log(server_log_path: Path, allow_server_errors: bool) -> dict:
    """P5h+2.d Rule D server-log scan.

    Any ERROR line hard-fails by default. WARN lines are split into
    allow-listed vs non-allow-listed; non-allow-listed WARNs are surfaced for
    human review but do not auto-drop the cell.
    """
    scan = {
        "path": str(server_log_path),
        "error_count": 0,
        "error_lines": [],
        "allowlisted_warn_count": 0,
        "non_allowlisted_warn_count": 0,
        "non_allowlisted_warn_lines": [],
    }
    if not server_log_path.exists():
        scan["missing"] = True
        return scan
    with server_log_path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if "ERROR" in line:
                scan["error_count"] += 1
                scan["error_lines"].append(f"{line_no}: {line}")
            elif "WARN" in line:
                if any(token in line for token in ALLOWLISTED_WARN_SUBSTRINGS):
                    scan["allowlisted_warn_count"] += 1
                else:
                    scan["non_allowlisted_warn_count"] += 1
                    scan["non_allowlisted_warn_lines"].append(f"{line_no}: {line}")
    if scan["error_count"] > 0 and not allow_server_errors:
        preview = "\n".join(scan["error_lines"][:5])
        raise SystemExit(
            f"{server_log_path}: {scan['error_count']} ERROR lines detected. "
            f"Rule D hard-fail. First lines:\n{preview}"
        )
    return scan
```

(c) In `def run_one_repeat(args, repeat)`, after each `shutil.move(...)`, replace the old scheduler-only check with:

```python
        scan = scan_server_log(dst / "server.log", args.allow_server_errors)
        (dst / "server_log_scan.json").write_text(json.dumps(scan, indent=2))
        if scan["non_allowlisted_warn_count"] > 0:
            print(
                f"  WARN review required for {dst}: "
                f"{scan['non_allowlisted_warn_count']} non-allow-listed WARN lines"
            )
```

(d) In `def main()`, after the `--allow-server-errors` argparse `add_argument` call (~line 195), add:

```python
    p.add_argument(
        "--inter-run-cooldown-secs",
        type=int,
        default=0,
        help="iron-bench --inter-run-cooldown-secs N for measured cells only "
        "(NOT applied to preheat). Default 0. Per P5h+2.d spec § 3.1.",
    )
```

(e) In `def run_one_repeat(args, repeat)`, inside the `env.update({...})` block (~line 79-88), append a conditional set:

```python
    if args.inter_run_cooldown_secs > 0:
        env["P5I_C_INTER_RUN_COOLDOWN_SECS"] = str(args.inter_run_cooldown_secs)
```

Place AFTER the existing `env.update({...})` closing brace, before the `cmd = [...]` block. Conditional (NOT unconditional) preserves byte-identity for legacy invocations.

- [ ] **Step C4: Run pytest to confirm pass**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py -v
```

Expected: BOTH tests PASS.

- [ ] **Step C5: Stop and report DONE — do NOT commit**

Per spec § 9.4 single-commit policy. Report status DONE to controller; controller will dispatch T1.

---

## Task 1: aggregator diagnostic-field extension

**Files:**
- Modify: `tools/p5i_c_pp_tps_envelope.py` (relax `EXPECTED_RUNS` + add diagnostic fields to `compute_pp_tps_envelope` per-repeat output)
- Modify: `tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py` (add 3 new pytests)

### T1.A — Tests first (TDD)

- [ ] **Step A1: Append 3 new pytests to existing test file**

In `tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py`, append at end of file:

```python


def _make_cell_csv(path: Path, pp_tps_values: list[float], pp: int) -> None:
    """Helper: write iron-bench CSV with given pp_tps series; pp_target fixed."""
    rows = []
    for idx, v in enumerate(pp_tps_values):
        rows.append({
            "target": "p5i_c",
            "pp_target": str(pp),
            "tg_target": "1",
            "run_idx": str(idx),
            "ttft_ms": "100",
            "tg_tps": "1.0",
            "tpot_ms": "1.0",
            "pp_tps": str(v),
            "e2e_s": "1.0",
            "prompt_tokens_local": str(pp),
            "prompt_tokens_server": str(pp),
            "completion_tokens_server": "1",
            "cached_tokens": "0",
            "finish_reason": "length",
        })
    write_iron_bench_csv(rows, path)


def test_trailing_slowdown_pct_emitted_per_repeat():
    """Positive trailing-slowdown case: last-3 median < first-3 median.

    Series: [1000]*12 + [800,800,800]. first3=1000, last3=800.
    trailing_slowdown_pct = 800/1000 - 1 = -0.20 = -20.0%
    """
    pp = 128
    series = [1000.0] * 12 + [800.0, 800.0, 800.0]
    assert len(series) == 15
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", str(len(series)),
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert "trailing_slowdown_pct" in per_rep
            assert abs(per_rep["trailing_slowdown_pct"] - (-20.0)) < 0.01
            assert "first_3_runs_median_pp_tps" in per_rep
            assert "last_3_runs_median_pp_tps" in per_rep
            assert abs(per_rep["first_3_runs_median_pp_tps"] - 1000.0) < 0.01
            assert abs(per_rep["last_3_runs_median_pp_tps"] - 800.0) < 0.01


def test_fast_start_drop_pct_emitted_per_repeat():
    """Positive fast-start-drop case: max(first-3) > median(last-3).

    Series: [1500,1500,1500] + [1200]*12. first3_max=1500, last3_med=1200.
    fast_start_drop_pct = 1500/1200 - 1 = +0.25 = +25.0%
    """
    pp = 512
    series = [1500.0, 1500.0, 1500.0] + [1200.0] * 12
    assert len(series) == 15
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", str(len(series)),
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert abs(per_rep["fast_start_drop_pct"] - 25.0) < 0.01


def test_diagnostic_fields_present_for_short_series_degenerate():
    """N < 3 runs: trailing_slowdown_pct / fast_start_drop_pct fields MUST
    still be present in JSON (null), so downstream tooling does not KeyError.
    Override --expected-runs to 2 to bypass the per-PP row-count guard.
    """
    pp = 128
    series = [1000.0, 1000.0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", "2",
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert "trailing_slowdown_pct" in per_rep
            assert "fast_start_drop_pct" in per_rep
            assert per_rep["trailing_slowdown_pct"] is None
            assert per_rep["fast_start_drop_pct"] is None
```

- [ ] **Step A2: Run pytests to confirm failure**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py -v -k "trailing_slowdown or fast_start or diagnostic_fields_present"
```

Expected: 3 tests FAIL — either with "unrecognized argument: --expected-runs" or "KeyError: trailing_slowdown_pct".

### T1.B — Implementation

- [ ] **Step B1: Refactor EXPECTED_RUNS to per-call override**

In `tools/p5i_c_pp_tps_envelope.py`:

(a) Replace the module-level `EXPECTED_RUNS: dict[int, int] = {128: 7, 512: 15}` with:

```python
# Legacy default; can be overridden via --expected-runs at call sites
# (P5h+2.d uses 15 for both PPs).
DEFAULT_EXPECTED_RUNS: dict[int, int] = {128: 7, 512: 15}
```

(b) Change `def load_pp_tps(csv_path: Path, pp: int) -> list[float]:` to accept `expected_runs: int`:

```python
def load_pp_tps(csv_path: Path, pp: int, expected_runs: int) -> list[float]:
```

Inside, remove the `if pp not in EXPECTED_RUNS:` guard and the `expected = EXPECTED_RUNS[pp]` line; replace the count check with:

```python
    if len(rows) != expected_runs:
        raise SystemExit(
            f"{csv_path}: expected {expected_runs} rows for PP={pp}, got {len(rows)}"
        )
```

(c) Change `def compute_pp_tps_envelope(repeat_csvs: list[Path], pp: int)` to accept `expected_runs: int`:

```python
def compute_pp_tps_envelope(
    repeat_csvs: list[Path], pp: int, expected_runs: int
) -> dict:
```

Pass `expected_runs` through to `load_pp_tps(path, pp, expected_runs)`.

- [ ] **Step B2: Add diagnostic-field computation inside `compute_pp_tps_envelope`**

Inside the existing per-repeat loop in `compute_pp_tps_envelope`, AFTER `ci = bootstrap_median_ci(...)`:

```python
        # P5h+2.d spec § 4.1: per-repeat diagnostic fields (no gate logic).
        if len(tps) >= 3:
            first_3 = tps[:3]
            last_3 = tps[-3:]
            first_3_med = median(first_3)
            last_3_med = median(last_3)
            trailing_slowdown_pct = (last_3_med / first_3_med - 1) * 100
            fast_start_drop_pct = (max(first_3) / last_3_med - 1) * 100
        else:
            first_3_med = None
            last_3_med = None
            trailing_slowdown_pct = None
            fast_start_drop_pct = None
        per_repeat.append(
            {
                "path": str(path),
                "n": len(tps),
                "median": med,
                "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
                "ci95_half_width_pct": ci["ci95_half_width_pct"],
                "first_3_runs_median_pp_tps": first_3_med,
                "last_3_runs_median_pp_tps": last_3_med,
                "trailing_slowdown_pct": trailing_slowdown_pct,
                "fast_start_drop_pct": fast_start_drop_pct,
            }
        )
```

Delete the OLD `per_repeat.append({...})` block (the one without diagnostic fields).

- [ ] **Step B3: Add `--expected-runs` CLI arg + plumb through**

In `def main()`:

```python
    p.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Per-PP expected row count (overrides DEFAULT_EXPECTED_RUNS). "
        "Required when PP not in default map (e.g., P5h+2.d uses 15 for both).",
    )
```

Replace the `compute_pp_tps_envelope(args.repeat_csv, args.pp)` calls with:

```python
    expected = (
        args.expected_runs
        if args.expected_runs is not None
        else DEFAULT_EXPECTED_RUNS.get(args.pp)
    )
    if expected is None:
        raise SystemExit(
            f"--expected-runs required for pp={args.pp} (no default registered)"
        )
    ironmlx = compute_pp_tps_envelope(args.repeat_csv, args.pp, expected)
    result: dict = {"ironmlx": ironmlx}
    if args.compare_repeat_csv:
        comparator = compute_pp_tps_envelope(
            args.compare_repeat_csv, args.pp, expected
        )
```

- [ ] **Step B4: Run pytests to confirm pass**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py -v
```

Expected: ALL tests in file PASS (existing tests + 3 new). If pre-existing tests fail because they relied on hardcoded EXPECTED_RUNS{128:7}, fix them by passing `--expected-runs 7` in their subprocess invocations.

- [ ] **Step B5: Driver pass-through for `--expected-runs`**

In `tools/p5h_2b_protocol_experiment.py` `def run_envelope(args, cell_map_per_repeat)`, locate the `cmd = [...]` block constructing the envelope sub-call. Add `--expected-runs` derived from `--runs-per-pp` parsed map:

```python
    # Parse runs-per-pp string ("128:15,512:15" -> {128:15, 512:15})
    runs_map: dict[int, int] = {}
    for entry in args.runs_per_pp.split(","):
        pp_s, n_s = entry.split(":")
        runs_map[int(pp_s.strip())] = int(n_s.strip())

    for pp in args.pps.split(","):
        ...
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp", pp,
            "--out-json", str(out_json),
            "--expected-runs", str(runs_map[int(pp)]),
        ]
```

(Replace just the `cmd = [...]` block; rest of `run_envelope` body unchanged.)

- [ ] **Step B6: Run full pytest suite (regression)**

```bash
uv run pytest tools/p5h_aggregator/tests/ -v
```

Expected: ALL tests PASS (no regression).

- [ ] **Step B7: Stop and report DONE — do NOT commit**

Per spec § 9.4. Report DONE to controller.

---

## Task 2: Stage 1 orchestrator + sweep + Mechanism gate analysis

**Files:**
- Create: `tools/p5h_2d_thermal_experiment.py`
- Create: `tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py`
- Output (host): `/tmp/p5h+2-d-stage1-r${R}-pp${PP}-cd${N}s/{bench.csv,server.log,meta.json,server_log_scan.json}` + `/tmp/p5h+2-d-stage1-cd${N}s-pp${PP}-envelope.json`
- Output (close-out artifact): `/tmp/p5h+2-d-stage1-mechanism-gate.json`

### T2.A — Orchestrator + analyzer code (TDD on analyzer logic)

- [ ] **Step A1: Write failing pytest for Mechanism gate analyzer**

Create `tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py`:

```python
"""Pytest for tools/p5h_2d_thermal_experiment.py Mechanism gate analyzer
(spec § 2.4)."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

import p5h_2d_thermal_experiment as drv  # noqa: E402


def _envelope_with_diagnostic(
    medians_pp128_per_repeat: list[dict], medians_pp512_per_repeat: list[dict]
) -> dict:
    """Build a synthetic combined envelope-like dict the analyzer consumes.

    Each per_repeat entry needs: trailing_slowdown_pct, fast_start_drop_pct.
    """
    return {
        "pp_envelopes": {
            "0s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
            "60s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
            "120s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
        }
    }


def test_strong_yes_when_both_pps_show_50pct_reduction_with_residual_under_10pct():
    """0s has 20% trailing on PP=128; 60s has 5% trailing -> 75% reduction.
    0s has 30% fast-start-drop on PP=512; 60s has 8% -> 73% reduction.
    Both PP-specific BEST residuals <= 10%.
    """
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -3.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 8.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 6.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "strong_yes", verdict
    assert verdict["best_cooldown_per_pp"]["128"] in ("60s", "120s")
    assert verdict["best_cooldown_per_pp"]["512"] in ("60s", "120s")


def test_weak_yes_when_only_one_pp_reduction():
    """PP=128 reduces 75%; PP=512 unchanged."""
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 28.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 27.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "weak_yes", verdict


def test_weak_yes_when_both_reduce_but_one_residual_above_10pct():
    """Both PPs reduce by >=50%, but PP=128 BEST residual remains >10%.
    Spec § 2.4 classifies this as weak_yes, not no.
    """
    pp128_0s = [{"trailing_slowdown_pct": -40.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -12.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -11.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 8.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 7.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "weak_yes", verdict
    assert verdict["details"]["128"]["reduced_by_50pct"] is True
    assert verdict["details"]["128"]["residual_le_10pct"] is False


def test_no_when_neither_pp_reduces_50pct():
    """Both PPs show < 50% reduction across all cooldowns."""
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -18.0, "fast_start_drop_pct": 4.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -17.0, "fast_start_drop_pct": 4.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 28.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 27.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "no", verdict


def test_no_when_0s_baseline_already_clean():
    """0s baseline residual <= 10% for both PPs -> mechanism not demonstrated.
    Spec § 2.4 last clause: classify as no/inconclusive."""
    pp128_0s = [{"trailing_slowdown_pct": -3.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 1.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 5.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 2.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 2.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "no", verdict
    assert "baseline_already_clean" in verdict.get("reason", ""), verdict
```

- [ ] **Step A2: Run pytests to confirm failure**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py -v
```

Expected: 5 tests FAIL with `ModuleNotFoundError: No module named 'p5h_2d_thermal_experiment'`.

- [ ] **Step A3: Create orchestrator + Mechanism gate analyzer**

Create `tools/p5h_2d_thermal_experiment.py`:

```python
"""P5h+2.d Stage 1 thermal cooldown orchestrator + Mechanism gate analyzer.

Per spec § 1.1 + § 2 + § 2.4.

CLI:
    python tools/p5h_2d_thermal_experiment.py sweep \\
        --cooldown-levels 0,60,120 \\
        --pps 128,512 \\
        --repeats 3 \\
        --runs-per-pp '128:15,512:15' \\
        --model-dir $SNAP \\
        --mlx-dir $HOME/.local/mlx \\
        --out-base /tmp/p5h+2-d-stage1

    python tools/p5h_2d_thermal_experiment.py gate \\
        --cooldown-levels 0,60,120 \\
        --pps 128,512 \\
        --envelope-glob '/tmp/p5h+2-d-stage1-cd{cooldown}-pp{pp}-envelope.json' \\
        --out-json /tmp/p5h+2-d-stage1-mechanism-gate.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
PROTOCOL_DRIVER = TOOLS_DIR / "p5h_2b_protocol_experiment.py"

REDUCTION_THRESHOLD_PCT = 50.0
RESIDUAL_MAX_PCT = 10.0
BASELINE_CLEAN_THRESHOLD_PCT = 10.0


def _abs_trailing(per_repeat: list[dict]) -> float:
    """Median across repeats of |trailing_slowdown_pct| clipped to non-negative."""
    vals = [
        max(0.0, -r["trailing_slowdown_pct"])
        for r in per_repeat
        if r["trailing_slowdown_pct"] is not None
    ]
    return median(vals) if vals else 0.0


def _pos_fast_start(per_repeat: list[dict]) -> float:
    """Median across repeats of fast_start_drop_pct clipped to non-negative."""
    vals = [
        max(0.0, r["fast_start_drop_pct"])
        for r in per_repeat
        if r["fast_start_drop_pct"] is not None
    ]
    return median(vals) if vals else 0.0


def _pp_residual(per_repeat: list[dict], pp: str) -> float:
    """Per-PP dominant residual: PP=128 -> trailing; PP=512 -> fast-start."""
    if pp == "128":
        return _abs_trailing(per_repeat)
    if pp == "512":
        return _pos_fast_start(per_repeat)
    raise ValueError(f"unknown PP={pp}")


def _pick_best_cooldown(matrix: dict, pp: str) -> tuple[str, float]:
    """Return (best_cooldown_str, residual_at_best) among non-baseline cooldowns.

    `0s` is WORST/baseline for control comparisons, not a valid BEST
    mechanism candidate.
    """
    candidates = []
    for cooldown in matrix:
        if cooldown == "0s":
            continue
        residual = _pp_residual(matrix[cooldown][pp], pp)
        candidates.append((cooldown, residual))
    if not candidates:
        raise SystemExit("matrix has no non-baseline cooldown candidates")
    # Sort by residual ASC, then cooldown integer ASC (tie-breaker).
    candidates.sort(key=lambda x: (x[1], int(x[0].rstrip("s"))))
    return candidates[0]


def compute_mechanism_gate(matrix: dict) -> dict:
    """Mechanism gate logic per spec § 2.4.

    matrix shape: {cooldown_str: {pp_str: per_repeat_list_of_diagnostic_dicts}}
    Returns: {"verdict": "strong_yes"|"weak_yes"|"no", "best_cooldown_per_pp": {...},
              "reason": str, "details": {...}}
    """
    pps = ["128", "512"]
    if not all("0s" in matrix and pp in matrix["0s"] for pp in pps):
        raise SystemExit("matrix missing 0s baseline for one or both PPs")
    baseline_residuals = {pp: _pp_residual(matrix["0s"][pp], pp) for pp in pps}

    # Spec § 2.4 last clause: if 0s baseline already clean -> mechanism not demonstrated.
    if all(baseline_residuals[pp] <= BASELINE_CLEAN_THRESHOLD_PCT for pp in pps):
        return {
            "verdict": "no",
            "best_cooldown_per_pp": {pp: "0s" for pp in pps},
            "reason": "baseline_already_clean: 0s residual <= 10% for both PPs",
            "details": {"baseline_residuals": baseline_residuals},
        }

    # Per PP: BEST cooldown + reduction vs baseline.
    best_per_pp: dict[str, str] = {}
    pp_full_pass: dict[str, bool] = {}
    pp_reduced_by_50: dict[str, bool] = {}
    pp_details: dict[str, dict] = {}
    for pp in pps:
        best_cd, best_res = _pick_best_cooldown(matrix, pp)
        best_per_pp[pp] = best_cd
        base = baseline_residuals[pp]
        reduction_pct = (
            (base - best_res) / base * 100.0 if base > 0 else 0.0
        )
        reduced_by_50 = reduction_pct >= REDUCTION_THRESHOLD_PCT
        residual_ok = best_res <= RESIDUAL_MAX_PCT
        pp_reduced_by_50[pp] = reduced_by_50
        pp_full_pass[pp] = reduced_by_50 and residual_ok
        pp_details[pp] = {
            "baseline_residual_pct": base,
            "best_cooldown": best_cd,
            "best_residual_pct": best_res,
            "reduction_pct": reduction_pct,
            "reduced_by_50pct": reduced_by_50,
            "residual_le_10pct": residual_ok,
            "passes_50pct_and_residual_le_10pct": pp_full_pass[pp],
        }

    n_pass = sum(pp_full_pass.values())
    if n_pass == 2:
        verdict = "strong_yes"
        reason = "both PPs >=50% reduction AND BEST residual <=10%"
    elif n_pass == 1 or all(pp_reduced_by_50.values()):
        verdict = "weak_yes"
        if n_pass == 1:
            reason = (
                f"only PP={'128' if pp_full_pass['128'] else '512'} passes full gate; "
                "other PP not demonstrated"
            )
        else:
            reason = "both PPs reduce >=50%, but at least one BEST residual remains >10%"
    else:
        verdict = "no"
        reason = "neither PP shows >=50% reduction with BEST residual <=10%"

    return {
        "verdict": verdict,
        "best_cooldown_per_pp": best_per_pp,
        "reason": reason,
        "details": pp_details,
    }


def _cooldown_label(n: int) -> str:
    return f"{n}s"


def _parse_runs_map(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in s.split(","):
        pp_s, n_s = entry.split(":")
        out[pp_s.strip()] = int(n_s.strip())
    return out


def _run_envelope_for_cooldown(args: argparse.Namespace, label: str) -> None:
    """Run envelope once per PP from exact P5h+2.d Stage 1 cell dirs."""
    pps = [pp.strip() for pp in args.pps.split(",")]
    runs_map = _parse_runs_map(args.runs_per_pp)
    for pp in pps:
        out_json = Path(f"{args.out_base}-cd{label}-pp{pp}-envelope.json")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp", pp,
            "--out-json", str(out_json),
            "--expected-runs", str(runs_map[pp]),
        ]
        for repeat in range(1, args.repeats + 1):
            cell_dir = Path(f"{args.out_base}-r{repeat}-pp{pp}-cd{label}")
            cmd.extend(["--repeat-csv", str(cell_dir / "bench.csv")])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"envelope failed cooldown={label} PP={pp}: {result.stderr}")
        print(f"  cooldown={label} PP={pp} envelope -> {out_json}")


def run_sweep(args: argparse.Namespace) -> None:
    """Run cooldown matrix: for each cooldown level, invoke p5h_2b driver."""
    cooldown_levels = [int(s) for s in args.cooldown_levels.split(",")]
    pps = [pp.strip() for pp in args.pps.split(",")]
    for cooldown in cooldown_levels:
        label = _cooldown_label(cooldown)
        exp_id = f"stage1-cd{label}"
        driver_out_base = f"{args.out_base}-driver-cd{label}"
        cmd = [
            sys.executable,
            str(PROTOCOL_DRIVER),
            "--phase", "t4",  # reuse t4 phase label for production sweeps
            "--exp-id", exp_id,
            "--server-lifecycle", "same_spawn_per_pp",
            "--pp-order", args.pps,
            "--logging-mode", "quiet_acceptance",
            "--mode", "production",
            "--repeats", str(args.repeats),
            "--pps", args.pps,
            "--runs-per-pp", args.runs_per_pp,
            "--preheat-seconds", str(args.preheat_seconds),
            "--preheat-runs", str(args.preheat_runs),
            "--model-dir", args.model_dir,
            "--mlx-dir", args.mlx_dir,
            "--out-base", driver_out_base,
            "--inter-run-cooldown-secs", str(cooldown),
            "--skip-envelope",
        ]
        print(f"=== Stage 1 cooldown={label} ===")
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            raise SystemExit(f"Stage 1 cooldown={label} sweep failed (exit {r.returncode})")
        for repeat in range(1, args.repeats + 1):
            for pp in pps:
                src = Path(f"{driver_out_base}-{exp_id}-r{repeat}-pp{pp}")
                dst = Path(f"{args.out_base}-r{repeat}-pp{pp}-cd{label}")
                if not src.exists():
                    raise SystemExit(f"expected cell dir missing: {src}")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
        _run_envelope_for_cooldown(args, label)


def run_gate(args: argparse.Namespace) -> None:
    """Read per-cooldown per-PP envelope JSONs; compute Mechanism gate; write
    one consolidated JSON."""
    cooldown_levels = [int(s) for s in args.cooldown_levels.split(",")]
    pps = args.pps.split(",")
    matrix: dict = {}
    for cooldown in cooldown_levels:
        label = _cooldown_label(cooldown)
        matrix[label] = {}
        for pp in pps:
            env_path = Path(args.envelope_glob.format(cooldown=label, pp=pp))
            if not env_path.exists():
                raise SystemExit(f"missing envelope JSON: {env_path}")
            env_json = json.loads(env_path.read_text())
            matrix[label][pp] = env_json["ironmlx"]["per_repeat"]
    verdict = compute_mechanism_gate(matrix)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep", help="Run Stage 1 cooldown matrix sweep")
    sweep.add_argument("--cooldown-levels", default="0,60,120")
    sweep.add_argument("--pps", default="128,512")
    sweep.add_argument("--repeats", type=int, default=3)
    sweep.add_argument("--runs-per-pp", default="128:15,512:15")
    sweep.add_argument("--preheat-seconds", type=int, default=300)
    sweep.add_argument("--preheat-runs", type=int, default=1100)
    sweep.add_argument("--model-dir", required=True)
    sweep.add_argument("--mlx-dir", required=True)
    sweep.add_argument("--out-base", default="/tmp/p5h+2-d-stage1")

    gate = sub.add_parser("gate", help="Compute Mechanism gate from envelope JSONs")
    gate.add_argument("--cooldown-levels", default="0,60,120")
    gate.add_argument("--pps", default="128,512")
    gate.add_argument(
        "--envelope-glob",
        default="/tmp/p5h+2-d-stage1-cd{cooldown}-pp{pp}-envelope.json",
        help="Format string with {cooldown} (e.g. '60s') and {pp} (e.g. '128')",
    )
    gate.add_argument(
        "--out-json", default="/tmp/p5h+2-d-stage1-mechanism-gate.json"
    )

    args = p.parse_args()
    if args.cmd == "sweep":
        run_sweep(args)
    elif args.cmd == "gate":
        run_gate(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step A4: Run pytests to confirm pass**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py -v
```

Expected: 5 tests PASS.

### T2.B — Pre-flight + sweep execution

- [ ] **Step B1: Boss-side prerequisite check (record in log)**

Verify in terminal:

```bash
ls /Users/xin/.ironmlx/models/models--Qwen--Qwen3.5-4B-MLX-4bit/snapshots/*/  # or current MoE target
echo "MLX_DIR=$HOME/.local/mlx"
which iron-bench  # should be cargo-built
```

If MoE model path differs from default, set `IRONMLX_MOE_MODEL_DIR` env var matching predecessor sweeps.

- [ ] **Step B2: Compute Boss-facing wall estimate + confirm OK before launch**

Per spec § 9.3: Stage 1 worst-case wall =
- 0s cell: ~7-8 min × 6 cells = ~45 min
- 60s cooldown sleep: 14 inter-run gaps × 60s × 6 cells = 84 min cooldown alone, plus benchmark/preheat overhead
- 120s cooldown sleep: 14 inter-run gaps × 120s × 6 cells = 168 min cooldown alone, plus benchmark/preheat overhead
- Plus 1100-run preheat per repeat × 3 repeats × 3 cooldowns = ~9 spawns @ ~5 min = 45 min

Total estimate: ~6 hr GPU wall for Stage 1 after benchmark/preheat overhead.

Print estimate; controller asks Boss to confirm before invoking sweep:

```
Stage 1 wall estimate: ~6 hr (cooldown sleep dominant at 120s level).
Spec § 9.3 budget: 12 hr GPU + 4 hr docs.
Proceed?  [y/N]
```

- [ ] **Step B3: Launch Stage 1 sweep (background)**

```bash
SNAP=$(ls -d /Users/xin/.ironmlx/models/models--Qwen--Qwen3.5-4B-MLX-4bit/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx
export SNAP MLX_DIR
rm -f /tmp/p5h+2-d-stage1-sweep.status
nohup bash -lc 'python tools/p5h_2d_thermal_experiment.py sweep \
    --cooldown-levels 0,60,120 \
    --pps 128,512 \
    --repeats 3 \
    --runs-per-pp "128:15,512:15" \
    --model-dir "$SNAP" \
    --mlx-dir "$MLX_DIR" \
    --out-base /tmp/p5h+2-d-stage1 \
    > /tmp/p5h+2-d-stage1-sweep.log 2>&1; echo $? > /tmp/p5h+2-d-stage1-sweep.status' &
echo $! > /tmp/p5h+2-d-stage1-sweep.pid
```

- [ ] **Step B4: Wait for sweep completion (block)**

```bash
PID=$(cat /tmp/p5h+2-d-stage1-sweep.pid)
while kill -0 "$PID" 2>/dev/null; do
  echo "Stage 1 still running; tail follows"
  tail -n 20 /tmp/p5h+2-d-stage1-sweep.log
  sleep 60
done
status=$(cat /tmp/p5h+2-d-stage1-sweep.status 2>/dev/null || echo 999)
test "$status" = "0" && echo "Stage 1 sweep exit OK" || (echo "Stage 1 sweep failed: status=$status"; exit "$status")
```

Expected: exit 0. If non-zero, inspect `/tmp/p5h+2-d-stage1-sweep.log` + per-cell `/tmp/p5h+2-d-stage1-r*-pp*-cd*s/server.log` per Rule D; on first ERROR find, stop and report BLOCKED to controller.

- [ ] **Step B5: Sanity-check Rule D scan outputs per cell**

```bash
for d in /tmp/p5h+2-d-stage1-r*-pp*-cd*s/; do
    test -f "$d/server_log_scan.json" || echo "VIOLATION: missing scan $d"
    python - "$d/server_log_scan.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
if d.get("error_count", 0) != 0:
    print(f"VIOLATION: {p} has {d['error_count']} ERROR lines")
if d.get("non_allowlisted_warn_count", 0) != 0:
    print(f"WARN_REVIEW: {p} has {d['non_allowlisted_warn_count']} non-allow-listed WARN lines")
PY
done
echo "Rule D scan review done"
```

Expected: zero VIOLATION lines. Any WARN_REVIEW lines must be reviewed and recorded before running the Mechanism gate verdict.

### T2.C — Mechanism gate analysis

- [ ] **Step C1: Run Mechanism gate analyzer**

```bash
python tools/p5h_2d_thermal_experiment.py gate \
    --cooldown-levels 0,60,120 \
    --pps 128,512 \
    --envelope-glob '/tmp/p5h+2-d-stage1-cd{cooldown}-pp{pp}-envelope.json' \
    --out-json /tmp/p5h+2-d-stage1-mechanism-gate.json
cat /tmp/p5h+2-d-stage1-mechanism-gate.json
```

Expected: JSON with `verdict` ∈ {`strong_yes`, `weak_yes`, `no`}, `best_cooldown_per_pp`, `reason`, `details`.

- [ ] **Step C2: Record verdict in a working-state file (NOT committed)**

```bash
verdict=$(python -c "import json; print(json.load(open('/tmp/p5h+2-d-stage1-mechanism-gate.json'))['verdict'])")
echo "P5h+2.d Stage 1 Mechanism gate verdict: $verdict" > /tmp/p5h+2-d-stage1-verdict.txt
cat /tmp/p5h+2-d-stage1-verdict.txt
```

- [ ] **Step C3: Stop and report DONE + VERDICT — do NOT commit**

Report to controller (and Boss): Stage 1 verdict = {strong_yes|weak_yes|no}; BEST cooldown per PP; key residual deltas. Controller decides whether to dispatch T3 + T4 (gated on strong_yes/weak_yes) or jump to T5 FAIL escalation (gated on no).

---

## Task 3: Stage 2 sudo powermetrics overlay (GATED on T2 verdict ∈ {strong_yes, weak_yes})

**Gating reminder:** If T2 verdict = `no`, SKIP this task. Proceed directly to T5 FAIL escalation. Do NOT execute any step below.

**Files:**
- Reuses or modifies: `tools/p5h_2b_thermal_overlay.py` (Stage 2 overlay parser)
- Output (host): `/tmp/p5h+2-d-stage2-pm-r${R}-pp${PP}.plist` + `/tmp/p5h+2-d-stage2-overlay-r${R}-pp${PP}.json`

### T3.A — powermetrics preflight

- [ ] **Step A1: Verify local powermetrics format + sampler support**

Local `/usr/bin/powermetrics --help` on this machine advertises `--format text|plist`, not JSON, and sampler names include `gpu_power` + `thermal` but not `smc`. Lock Stage 2 to plist output:

```bash
/usr/bin/powermetrics --help 2>&1 | sed -n '/output formats/,/samplers are supported/p'
```

Expected: output lists `plist`, `gpu_power`, and `thermal`. If `smc`/fan fields are unavailable in the resulting plist, H1.b fan-hysteresis evidence is recorded as `indeterminate`, not as failure.

- [ ] **Step A2: Verify Boss has applied sudoers rule (manual check)**

Confirm with Boss:

```bash
sudo cat /etc/sudoers.d/ironmlx-powermetrics 2>/dev/null
```

Expected output:

```
# /etc/sudoers.d/ironmlx-powermetrics
xin ALL=(root) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power,thermal --format plist -i 500 -o /tmp/p5h+2-d-*
```

If not present, BLOCK and report to controller.

- [ ] **Step A3: Confirm passwordless sudo invocation starts and can be stopped**

```bash
sudo -n /usr/bin/powermetrics --samplers gpu_power,thermal --format plist -i 500 -o /tmp/p5h+2-d-stage2-smoke.plist &
PM_PID=$!
sleep 2
kill -INT "$PM_PID" 2>/dev/null || kill -TERM "$PM_PID" 2>/dev/null
wait "$PM_PID" 2>/dev/null || true
ls -lh /tmp/p5h+2-d-stage2-smoke.plist
head -c 80 /tmp/p5h+2-d-stage2-smoke.plist
```

Expected: file exists and non-zero size. If sudo prompts for password, the sudoers rule is wrong — BLOCK. If `kill -INT`/`kill -TERM` cannot stop the captured PID, BLOCK and do not launch Stage 2 until the stop mechanism is corrected.

- [ ] **Step A4: Add plist parser support to existing thermal overlay**

Append this pytest to `tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py`:

```python
def test_parse_powermetrics_plist_nul_stream(tmp_path):
    import plistlib

    p = tmp_path / "thermal.plist"
    first = plistlib.dumps({"samples": [{"timestamp_ms": 1000, "gpu_die_temp_c": 60.0}]})
    second = plistlib.dumps({"samples": [{"timestamp_ms": 1500, "gpu_die_temp_c": 62.0}]})
    p.write_bytes(first + b"\0" + second)
    samples = parse_powermetrics_samples(p)
    assert len(samples) == 2
    assert _infer_timestamp_field(samples) == "timestamp_ms"
```

Then update `tools/p5h_2b_thermal_overlay.py`:

```python
import plistlib
```

Add helpers above `parse_powermetrics_samples`:

```python
def _collect_sample_dicts(obj: object) -> list[dict]:
    out: list[dict] = []

    def walk(x: object) -> None:
        if isinstance(x, dict):
            if any(k in x for k in ("timestamp", "sample_time_ms", "timestamp_ms", "time_ms")):
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return out


def _parse_plist_samples(raw: bytes) -> list[dict]:
    samples: list[dict] = []
    for part in raw.split(b"\0"):
        part = part.strip()
        if not part:
            continue
        try:
            samples.extend(_collect_sample_dicts(plistlib.loads(part)))
        except Exception:
            continue
    return samples
```

At the start of `parse_powermetrics_samples(json_path: Path)`, before `text = ...`, insert:

```python
    raw = json_path.read_bytes()
    stripped = raw.lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"bplist"):
        samples = _parse_plist_samples(raw)
        if samples:
            return samples
```

Run:

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py -v
```

Expected: all thermal overlay pytests PASS. If the real smoke plist contains no timestamp-like sample dicts after this change, BLOCK Stage 2 and inspect the first plist schema before running the 6 production cells.

### T3.B — Stage 2 sweep execution (2 PPs × 3 repeats = 6 cells at BEST cooldown)

- [ ] **Step B1: Read BEST cooldown per PP from Mechanism gate JSON**

```bash
BEST_128=$(python -c "import json; print(json.load(open('/tmp/p5h+2-d-stage1-mechanism-gate.json'))['best_cooldown_per_pp']['128'])")
BEST_512=$(python -c "import json; print(json.load(open('/tmp/p5h+2-d-stage1-mechanism-gate.json'))['best_cooldown_per_pp']['512'])")
echo "BEST cooldowns: PP=128 $BEST_128 / PP=512 $BEST_512"
```

- [ ] **Step B2: Run Stage 2 sweep (loop manually per cell)**

For each repeat r ∈ {1, 2, 3} and PP ∈ {128, 512}:

```bash
set -euo pipefail
SNAP=$(ls -d /Users/xin/.ironmlx/models/models--Qwen--Qwen3.5-4B-MLX-4bit/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx
for r in 1 2 3; do
  for pp_cooldown in "128:${BEST_128}" "512:${BEST_512}"; do
    pp=${pp_cooldown%:*}
    cooldown=${pp_cooldown#*:}
    cooldown_int=${cooldown%s}
    pm_out=/tmp/p5h+2-d-stage2-pm-r${r}-pp${pp}.plist
    rm -f "$pm_out"
    sudo -n /usr/bin/powermetrics --samplers gpu_power,thermal --format plist -i 500 -o "$pm_out" &
    pm_pid=$!
    echo "[r=$r pp=$pp cooldown=$cooldown] powermetrics pid=$pm_pid"
    sleep 1

    # One-PP sweep with cooldown via existing P5h+2.b driver
    set +e
    python tools/p5h_2b_protocol_experiment.py \
      --phase t4 --exp-id "stage2-r${r}-pp${pp}" \
      --server-lifecycle same_spawn_per_pp \
      --pp-order "$pp" \
      --logging-mode quiet_acceptance \
      --mode production \
      --repeats 1 --pps "$pp" \
      --runs-per-pp "${pp}:15" \
      --preheat-seconds 300 --preheat-runs 1100 \
      --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
      --out-base /tmp/p5h+2-d-stage2 \
      --inter-run-cooldown-secs "$cooldown_int" \
      --skip-envelope
    driver_rc=$?
    kill -INT "$pm_pid" 2>/dev/null || kill -TERM "$pm_pid" 2>/dev/null
    wait "$pm_pid" 2>/dev/null || true
    set -e
    test -s "$pm_out" || (echo "powermetrics output missing/empty: $pm_out"; exit 1)
    test "$driver_rc" = "0" || exit "$driver_rc"

  done
done
```

Expected: completes without ERROR; per cell `bench.csv` + `server.log` + non-empty powermetrics plist written.

### T3.C — Overlay analysis + H1 sub-hypothesis identification

- [ ] **Step C1: Run thermal overlay joiner per cell**

```bash
for r in 1 2 3; do
  for pp in 128 512; do
    cell_dir=/tmp/p5h+2-d-stage2-stage2-r${r}-pp${pp}-r1-pp${pp}
    pm=/tmp/p5h+2-d-stage2-pm-r${r}-pp${pp}.plist
    overlay_out=/tmp/p5h+2-d-stage2-overlay-r${r}-pp${pp}.json
    python tools/p5h_2b_thermal_overlay.py \
      --powermetrics-json "$pm" \
      --cell-dir "$cell_dir" \
      --out-json "$overlay_out"
  done
done
```

Expected: 6 overlay JSONs written with per-run pp_tps plus thermal-summary alignment. Fan/rpm fields are used only if present in the local powermetrics plist schema.

- [ ] **Step C2: Manually inspect overlays for H1.a / H1.b / H1.c signal**

```bash
for f in /tmp/p5h+2-d-stage2-overlay-r*-pp*.json; do
  echo "=== $f ==="
  python -c "
import json, sys
d = json.load(open('$f'))
overlay = d.get('overlay', [])
if not overlay: print('empty overlay'); sys.exit()
for row in overlay:
    ts = row.get('thermal_summary') or {}
    print(f\"run_idx={row.get('run_idx')} pp_tps={row.get('pp_tps')} max_gpu_die_c={ts.get('max_gpu_die_c')} n_temp_samples={ts.get('n_temp_samples')}\")
"
done
```

Document findings in `/tmp/p5h+2-d-stage2-h1-analysis.md` (working-state, not committed):

```
H1.a thermal soak evidence: <yes/no — gpu_temp monotonic rise + correlation>
H1.b fan hysteresis evidence: <yes/no/indeterminate — fan/rpm field lag if present; indeterminate if local powermetrics plist lacks fan fields>
H1.c preheat topology evidence: <yes/no — PP-asymmetry>
Verdict: <H1.a | H1.b | H1.c | indeterminate>
```

- [ ] **Step C3: Stop and report DONE + H1 SUB-HYPOTHESIS — do NOT commit**

Report verdict to controller.

---

## Task 4: omlx control sweep (GATED on T2 verdict ∈ {strong_yes, weak_yes})

**Gating reminder:** Same gate as T3. If T2 verdict = `no`, SKIP.

**Files:**
- No new code; reuses `iron-bench --target omlx-baseline` + driver
- Output (host): `/tmp/p5h+2-d-omlx-{best,worst}-r${R}-pp${PP}/{bench.csv,server.log}`

### T4.A — omlx per-cell server lifecycle helpers

- [ ] **Step A1: Verify omlx baseline can start (per `[reference-iron-rivals-baselines]`)**

```bash
cd /Users/xin/workspace/iron-rivals/omlx
nohup uv run --with-editable . mlx-omni-server --host 0.0.0.0 --port 8090 \
    > /tmp/p5h+2-d-omlx-smoke-server.log 2>&1 &
echo $! > /tmp/p5h+2-d-omlx-server.pid
sleep 8
curl -s http://127.0.0.1:8090/v1/models | head -c 500
kill "$(cat /tmp/p5h+2-d-omlx-server.pid)" 2>/dev/null
```

Expected: omlx returns models list and stops cleanly. If not, BLOCK.

### T4.B — omlx sweep at BEST and WORST cooldown

- [ ] **Step B1: Loop omlx control cells**

WORST cooldown is `0s` (Stage 1 baseline). BEST cooldown read from Mechanism gate JSON per § T3.B.B1.

```bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OMLX_DIR=/Users/xin/workspace/iron-rivals/omlx
OMLX_MODEL_DIR=/Users/xin/workspace/iron-rivals/omlx/.models/Qwen3.5-4B-MLX-4bit
BEST_128=$(python -c "import json; print(json.load(open('/tmp/p5h+2-d-stage1-mechanism-gate.json'))['best_cooldown_per_pp']['128'])")
BEST_512=$(python -c "import json; print(json.load(open('/tmp/p5h+2-d-stage1-mechanism-gate.json'))['best_cooldown_per_pp']['512'])")

start_omlx() {
  local log=$1
  cd "$OMLX_DIR"
  nohup uv run --with-editable . mlx-omni-server --host 0.0.0.0 --port 8090 > "$log" 2>&1 &
  echo $!
}

wait_omlx() {
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:8090/v1/models >/dev/null && return 0
    sleep 2
  done
  return 1
}

stop_omlx() {
  local pid=$1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

for r in 1 2 3; do
  for pp in 128 512; do
    for tag in best worst; do
      if [ "$tag" = "best" ]; then
        cd_label=$([ "$pp" = "128" ] && echo "$BEST_128" || echo "$BEST_512")
      else
        cd_label=0s
      fi
      cd_int=${cd_label%s}
      cell_dir=/tmp/p5h+2-d-omlx-${tag}-r${r}-pp${pp}
      mkdir -p "$cell_dir"
      server_log="$cell_dir/server.log"
      server_pid=$(start_omlx "$server_log")
      trap 'stop_omlx "$server_pid"' EXIT
      wait_omlx || (echo "omlx failed health check; see $server_log"; exit 1)

      cd "$REPO"
      cargo run --release -p iron-bench -- \
        --target "omlx-baseline=http://127.0.0.1:8090" \
        --model qwen3.5-4b \
        --model-dir "$OMLX_MODEL_DIR" \
        --prompt-len 512 \
        --max-tokens 1 \
        --runs 1100 \
        --warmup 0 \
        --format csv > "$cell_dir/preheat.csv"

      cargo run --release -p iron-bench -- \
        --target "omlx-baseline=http://127.0.0.1:8090" \
        --model qwen3.5-4b \
        --model-dir "$OMLX_MODEL_DIR" \
        --prompt-len "$pp" \
        --max-tokens 1 \
        --runs 15 \
        --warmup 1 \
        --inter-run-cooldown-secs "$cd_int" \
        --format csv > "$cell_dir/bench.csv"

      stop_omlx "$server_pid"
      trap - EXIT
      echo "[omlx $tag r=$r pp=$pp cooldown=$cd_label] done"
    done
  done
done
```

Adjust `OMLX_MODEL_DIR` if Boss's omlx baseline path differs; document at run time. This preserves the design requirement: same monolithic 1100-run preheat and fresh server process per control cell.

### T4.C — omlx interpretation

- [ ] **Step C1: Build omlx diagnostic envelopes + per-cell median table**

```bash
for tag in best worst; do
  for pp in 128 512; do
    cmd=(python tools/p5i_c_pp_tps_envelope.py --pp "$pp" --expected-runs 15 \
      --out-json "/tmp/p5h+2-d-omlx-${tag}-pp${pp}-envelope.json")
    for r in 1 2 3; do
      cmd+=(--repeat-csv "/tmp/p5h+2-d-omlx-${tag}-r${r}-pp${pp}/bench.csv")
    done
    "${cmd[@]}"
  done
done

echo "tag,r,pp,cooldown,median_pp_tps" > /tmp/p5h+2-d-omlx-medians.csv
for tag in best worst; do
  for r in 1 2 3; do
    for pp in 128 512; do
      f=/tmp/p5h+2-d-omlx-${tag}-r${r}-pp${pp}/bench.csv
      if [ ! -f "$f" ]; then continue; fi
      med=$(python -c "
import csv
from statistics import median
rows = [float(r['pp_tps']) for r in csv.DictReader(open('$f'))]
print(f'{median(rows):.2f}')
")
      cd=$( [ "$tag" = "worst" ] && echo "0s" || ([ "$pp" = "128" ] && echo "$BEST_128" || echo "$BEST_512") )
      echo "$tag,$r,$pp,$cd,$med" >> /tmp/p5h+2-d-omlx-medians.csv
    done
  done
done
cat /tmp/p5h+2-d-omlx-medians.csv
```

- [ ] **Step C2: Classify (proportional / flat / mixed)**

Manually inspect `/tmp/p5h+2-d-omlx-medians.csv` and the four `/tmp/p5h+2-d-omlx-{best,worst}-pp{128,512}-envelope.json` diagnostic outputs. Classification rule per spec § 6.3:

- **Proportional shift**: omlx WORST (`0s`) shows the same dominant residual direction as ironmlx and BEST reduces that residual by ≥ 50% for at least one PP.
- **Flat**: omlx WORST and BEST dominant residuals differ by < 2 percentage points for both PPs.
- **Mixed**: inconsistent

Write verdict + interpretation in `/tmp/p5h+2-d-omlx-verdict.md` (working state, not committed).

- [ ] **Step C3: Stop and report DONE + omlx CLASSIFICATION — do NOT commit**

---

## Task 5: Close-out — single commit packaging all infra + tests + docs + Phase 0 PASS backfill or deferred note

**Files:**
- Create: `docs/p5h+2-d-close-out.md`
- Modify: `docs/p5i-c-phase-0-close-out.md` (§ 1 #4 + § 6 + § 9 backfill per § 1.3 outcome matrix)
- Modify: `docs/p5i-c-phase-0-ranking-snapshot.md` (preamble status)
- All WIP from T0-T4: iron-bench Args + runner; harness env-var; protocol_experiment driver; envelope tool; new orchestrator + tests; thermal overlay plist parser update

### T5.A — Backfill docs based on T2 verdict

- [ ] **Step A1: Read all verdict files**

```bash
cat /tmp/p5h+2-d-stage1-verdict.txt 2>/dev/null
cat /tmp/p5h+2-d-stage1-mechanism-gate.json 2>/dev/null
cat /tmp/p5h+2-d-stage2-h1-analysis.md 2>/dev/null
cat /tmp/p5h+2-d-omlx-verdict.md 2>/dev/null
```

- [ ] **Step A2: Write `docs/p5h+2-d-close-out.md`**

Required sections (per outcome — strong_yes / weak_yes / no — pick matching template branch from § 1.3 outcome matrix):

```markdown
# P5h+2.d — Thermal / Residual-Variance Investigation: Close-out

**Status:** {Strong PASS | Weak evidence | Mechanism-only | FAIL/DEFERRED} per spec § 1.3 outcome matrix.
**Date:** 2026-05-25 (or actual close date).
**Branch:** `ironmlx-p5h+2-d-thermal-investigation` HEAD `<this commit SHA>`.

## § 1 Mechanism gate verdict (Stage 1)

Stage 1 protocol: spec § 2.1. 18 cells executed; all server.log ERROR == 0 and all non-allow-listed WARNs reviewed.

- Verdict: `<strong_yes | weak_yes | no>` (per `/tmp/p5h+2-d-stage1-mechanism-gate.json`)
- BEST cooldown PP=128: `<value>` (residual `<pct>`)
- BEST cooldown PP=512: `<value>` (residual `<pct>`)
- Reason: `<analyzer reason field>`
- 0s baseline residuals: PP=128 `<pct>` / PP=512 `<pct>`

## § 2 Acceptance gate verdict (Stage 2 + omlx control; if gate triggered)

- Acceptance gate envelope per PP at BEST cooldown: PP=128 `<pct>` / PP=512 `<pct>`
- A1 verdict: `<PASS | FAIL>`
- A2 Stage 2 H1 sub-hypothesis: `<H1.a | H1.b | H1.c | indeterminate>` per `/tmp/p5h+2-d-stage2-h1-analysis.md`
- A3 omlx control: `<proportional | flat | mixed>` per `/tmp/p5h+2-d-omlx-verdict.md`
- A4 Phase 0 § 7 #4 backfill action chosen: `<PASS | caveat | additive note | failed-attempt note>` per § 1.3 outcome matrix

## § 3 What landed (reusable infra)

- iron-bench `--inter-run-cooldown-secs` production CLI flag + tests
- Aggregator diagnostic fields (`trailing_slowdown_pct`, `fast_start_drop_pct`, `first/last_3_runs_median_pp_tps`) + 3 pytests
- `tools/p5h_2d_thermal_experiment.py` orchestrator + Mechanism gate analyzer + 4 pytests
- Stage 2 sudo powermetrics invocation recipe + overlay parser support for the actual local powermetrics format

## § 4 Wall summary

| Bucket | Cap | Actual |
|---|---|---|
| GPU wall | 12 hr | `<H>` hr |
| Docs/analysis wall | 4 hr | `<H>` hr |

## § 5 References

- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md` (commit `e9dd6ad`)
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-d-thermal-investigation.md` (this commit predecessor)
- Predecessor: `docs/p5h+2-b-close-out.md` § 9-10, `docs/p5h+2-c-close-out.md`, `docs/p5i-c-phase-0-close-out.md` § 1 #4
- Memory: `[project-p5h-2b-findings]`, `[project-p5h-2c-findings]`, `[project-p5h-2a-findings]`; new `[project-p5h-2d-findings]` written by close-out
- Codex review chain: `reports/p5h+2-d-brainstorm-codex-questions.md` (Codex brainstorm consultation), `reports/p5h+2-b-rerun-codex-review.md` (Codex round-4)
```

- [ ] **Step A3: Update `docs/p5i-c-phase-0-close-out.md` § 1 row 4**

Locate the existing § 1 row 4 cell ending with "...see `docs/p5h+2-b-close-out.md` § 9-10)". Append a new sentence matching the chosen § 1.3 outcome:

- If strong PASS: replace the "STILL FAIL/DEFERRED" wording with "PASS — restored via P5h+2.d thermal protocol fix (BEST cooldown PP=128 `<X>s` / PP=512 `<Y>s`); see `docs/p5h+2-d-close-out.md`."
- If weak evidence: append "P5h+2.d closed with weak evidence (PP=128 `<X>%` / PP=512 `<Y>%`); no automatic PASS backfill per Boss + Codex; criterion #4 remains FAIL/DEFERRED pending explicit decision."
- If mechanism-only or FAIL: append "P5h+2.d closed FAIL/DEFERRED (mechanism gate verdict `<verdict>`); criterion #4 STILL FAIL/DEFERRED; P5h+2 chain closed."

Also update § 6 status pointer + § 9 next-phase ordering accordingly.

- [ ] **Step A4: Update `docs/p5i-c-phase-0-ranking-snapshot.md` preamble**

Append a `**2026-05-25 update:**` sentence summarizing P5h+2.d outcome + Phase 0 § 7 #4 status change (if any).

- [ ] **Step A5: Write memory entry**

Create `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2d_findings.md`:

```markdown
---
name: project-p5h-2d-findings
description: P5h+2.d thermal/residual-variance investigation — <outcome>; Stage 1 Mechanism gate <verdict>; BEST cooldown <values>; Acceptance gate <if triggered>; Phase 0 § 7 #4 <PASS | STILL FAIL/DEFERRED>
metadata:
  type: project
---

P5h+2.d closed <DATE> as <outcome>.

**Stage 1 Mechanism gate**: <verdict>. BEST cooldown PP=128 <X>s (residual <Y>%); PP=512 <X>s (residual <Y>%). 0s baseline residuals: <details>.

**Stage 2 H1 sub-hypothesis**: <H1.a | H1.b | H1.c | indeterminate>.

**omlx control**: <proportional | flat | mixed>.

**Phase 0 § 7 #4 outcome**: <PASS backfill | weak-evidence caveat | STILL FAIL/DEFERRED>.

**Reusable infra**: iron-bench --inter-run-cooldown-secs; aggregator diagnostic fields; tools/p5h_2d_thermal_experiment.py; Stage 2 powermetrics overlay parser support.

Links: [[project-p5h-2b-findings]] (re-attempt that spawned this), [[project-p5h-2c-findings]] (scheduler fix), [[project-p5h-2a-findings]] (preheat protocol).
```

Append a one-line entry to `MEMORY.md`:

```
- [P5h+2.d thermal/residual-variance investigation](project_p5h_2d_findings.md) — <outcome one-line summary>
```

### T5.B — cargo + pytest regression gates

- [ ] **Step B1: Full cargo gates**

```bash
cargo fmt && cargo +nightly fmt --all -- --check && \
  cargo +nightly clippy --all-features --workspace -- -D warnings && \
  cargo build --release && \
  cargo test --release -p iron-bench
```

Expected: zero warnings; all PASS.

- [ ] **Step B2: Full pytest gate**

```bash
uv run pytest tools/p5h_aggregator/tests/ -v
```

Expected: all PASS (existing + new from T0/T1/T2).

### T5.C — Single commit

- [ ] **Step C1: Stage all changes**

```bash
git add iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/tests/inter_run_cooldown_secs.rs \
  ironmlx/tests/p5i_c_phase_0_capture.rs \
  tools/p5h_2b_protocol_experiment.py \
  tools/p5h_2b_thermal_overlay.py \
  tools/p5i_c_pp_tps_envelope.py \
  tools/p5h_2d_thermal_experiment.py \
  tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py \
  tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py \
  tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py \
  tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py \
  docs/p5h+2-d-close-out.md \
  docs/p5i-c-phase-0-close-out.md \
  docs/p5i-c-phase-0-ranking-snapshot.md
git status
```

Expected: clean staging matching list above. NO other files staged.

- [ ] **Step C2: Create single commit**

```bash
git commit -m "$(cat <<'EOF'
feat(p5h+2-d): thermal/residual-variance investigation — <outcome>; Phase 0 § 7 #4 <action>

Phase outcome per spec § 1.3: <Strong PASS | Weak evidence | Mechanism-only |
FAIL/DEFERRED>.

Stage 1 Mechanism gate verdict: <strong_yes | weak_yes | no>. BEST cooldown
PP=128 <X>s (residual <Y>%); PP=512 <X>s (residual <Y>%). All 18 cells
executed; server.log ERROR == 0 and non-allow-listed WARN review completed at production scale.

<if gated tasks ran:>
Stage 2 sudo powermetrics overlay: H1 sub-hypothesis = <H1.a thermal-soak |
H1.b fan-hysteresis | H1.c preheat-topology | indeterminate>. 6 cells; PID-
managed powermetrics capture with partial-output preservation worked across all runs.

omlx control: <proportional | flat | mixed> shift. 12 cells run via iron-bench
--target omlx-baseline at BEST + WORST=0s cooldown.

Phase 0 § 7 #4 backfill action: <PASS | weak-evidence caveat | additive note |
failed-attempt note>.

Reusable infra shipped:
- iron-bench --inter-run-cooldown-secs production CLI flag (sequential mode
  only; rejects concurrent mode; preserves byte-identity when default 0); 2
  integration tests
- Capture harness env-var P5I_C_INTER_RUN_COOLDOWN_SECS pass-through (build_iron_args)
- tools/p5h_2b_protocol_experiment.py --inter-run-cooldown-secs pass-through
  + Rule D server-log scan + smoke pytests
- tools/p5i_c_pp_tps_envelope.py per-repeat diagnostic fields (trailing_slowdown_pct,
  fast_start_drop_pct, first/last_3_runs_median_pp_tps); flexible
  --expected-runs override; 3 new pytests
- tools/p5h_2d_thermal_experiment.py orchestrator + Mechanism gate analyzer
  with strong_yes / weak_yes / no verdicts; baseline-already-clean edge case
  guarded; 5 unit tests
- Stage 2 powermetrics PID-managed shell capture with SIGINT/SIGTERM shutdown
  + partial plist preservation

Predeclared exclusion rules locked before T2: Rule B kept (envelope trim only);
Rule C removed (cannot exclude the degradation being studied); Rule D revised
(ANY server.log ERROR -> cell FAILS hard-stop; non-allow-listed WARN -> human
review not auto-drop).

Wall: GPU <H>hr (cap 12); docs/analysis <H>hr (cap 4); total <H>hr (cap 16
per spec § 9.3).

Spec: docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md
Plan: docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-d-thermal-investigation.md
Close-out: docs/p5h+2-d-close-out.md
Codex review chain: reports/p5h+2-d-* + reports/p5h+2-b-rerun-codex-review.md (gitignored)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
git log --oneline -3
```

Expected: clean commit; working tree clean; new commit at HEAD.

- [ ] **Step C3: Report DONE — Boss handles git push manually**

Per Boss's prior pattern: do NOT push. Report to controller; Boss decides when to push to remote.

---

## Self-review check (controller before T5 dispatch)

1. **Spec coverage**: § 1.1 sequencing → T2/T3/T4 + gating; § 2 cooldown matrix → T2 sweep; § 3 iron-bench flag → T0; § 4 aggregator → T1; § 5 sudo powermetrics → T3; § 6 omlx control → T4; § 7 rules → predeclared in plan top + driver hard-stop in T0.C/T2.B; § 8 acceptance → analyzer in T2 + close-out narrative in T5; § 9 budget split tracked in T5.A § 4; § 10 sudoers verified in T3.A; § 11 Phase 1 brainstorm boundary noted but out-of-scope of this plan; § 12 references cited in close-out template.
2. **Placeholder scan**: No "TBD" / "TODO" / "fill in later"; all code blocks verbatim; angle-bracket `<value>` placeholders ONLY in close-out templates (intentional — implementer fills based on actual T2-T4 verdicts).
3. **Type consistency**: `inter_run_cooldown_secs` (Rust u64) ↔ `--inter-run-cooldown-secs` (clap) ↔ `P5I_C_INTER_RUN_COOLDOWN_SECS` (env) ↔ `args.inter_run_cooldown_secs` (Python int) — names + types consistent. `trailing_slowdown_pct` / `fast_start_drop_pct` consistent between aggregator output (T1) + analyzer consumer (T2 analyzer). `best_cooldown_per_pp` dict key shape `"128"` / `"512"` consistent T2 → T3.B → T4.B.

No placeholder issues found.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-d-thermal-investigation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — controller dispatches fresh subagent per task with full task text + context; two-stage review (spec compliance then code quality) between tasks; T5 commits all WIP.
2. **Inline Execution** — controller executes inline via superpowers:executing-plans skill; checkpoint reviews per task.

Boss chooses.
