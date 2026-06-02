# Scheduler Autotune Calibrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ironmlx scheduler-autotune calibrate`, an opt-in CLI runner that launches local `ironmlx serve` candidates, drives `iron-bench --format autotune-json`, merges results, selects a scheduler profile, and optionally writes a runtime profile.

**Architecture:** Keep the hot path unchanged. Add a focused CLI orchestration module beside the existing scheduler-autotune CLI; pure candidate parsing and command construction are unit-tested, while the real runner uses subprocesses and existing `core::scheduler_autotune` merge/select/profile APIs.

**Tech Stack:** Rust, clap, serde_json, reqwest, tokio, std::process, existing `core::scheduler_autotune`, external `iron-bench` binary.

---

## File Structure

- Modify `ironmlx/src/cli/mod.rs` — add the calibrate parser test and expose the sibling calibrate module.
- Modify `ironmlx/src/cli/scheduler_autotune.rs` — add a `Calibrate` subcommand variant and dispatch to the new module.
- Create `ironmlx/src/cli/scheduler_autotune_calibrate.rs` — own calibrate args, candidate parsing, artifact path generation, command construction, health wait, subprocess runner, and final merge/select/profile output.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — document the one-command calibration workflow.
- Modify `docs/superpowers/specs/2026-06-02-scheduler-autotune-calibrate-design.md` — update status after implementation.

### Task 1: Parser And Candidate Config

**Files:**
- Modify: `ironmlx/src/cli/mod.rs`
- Modify: `ironmlx/src/cli/scheduler_autotune.rs`
- Create: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [ ] **Step 1.1: Write failing parser and candidate tests**

Add a parser test in `ironmlx/src/cli/mod.rs`:

```rust
#[test]
fn scheduler_autotune_calibrate_subcommand_parses_required_matrix() {
    let cli = Cli::parse_from([
        "ironmlx",
        "scheduler-autotune",
        "calibrate",
        "--model",
        "/tmp/model",
        "--model-name",
        "GLM-4.7-flash-4bit",
        "--hardware-label",
        "m5-max-128g",
        "--iron-bench-bin",
        "target/release/iron-bench",
        "--output-dir",
        "/tmp/autotune",
        "--candidate",
        "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768",
        "--prompt-len",
        "1024,2048",
        "--max-tokens",
        "128",
        "--concurrency",
        "1,2",
        "--write-profile",
        "/tmp/scheduler-profile.json",
    ]);

    match cli.command {
        Command::SchedulerAutotune(args) => match args.action {
            Some(super::scheduler_autotune::SchedulerAutotuneAction::Calibrate(calibrate)) => {
                assert_eq!(calibrate.model.to_string_lossy(), "/tmp/model");
                assert_eq!(calibrate.model_name, "GLM-4.7-flash-4bit");
                assert_eq!(calibrate.hardware_label, "m5-max-128g");
                assert_eq!(calibrate.iron_bench_bin.to_string_lossy(), "target/release/iron-bench");
                assert_eq!(calibrate.output_dir.to_string_lossy(), "/tmp/autotune");
                assert_eq!(calibrate.candidates.len(), 1);
                assert_eq!(calibrate.candidates[0].b_max, 2);
                assert_eq!(calibrate.candidates[0].prefill_chunk_size, 1024);
                assert_eq!(calibrate.prompt_len, vec![1024, 2048]);
                assert_eq!(calibrate.max_tokens, 128);
                assert_eq!(calibrate.concurrency, vec![1, 2]);
                assert_eq!(
                    calibrate
                        .write_profile
                        .as_ref()
                        .expect("expected write profile")
                        .to_string_lossy(),
                    "/tmp/scheduler-profile.json"
                );
            }
            other => panic!("expected Calibrate action, got {other:?}"),
        },
        other => panic!("expected SchedulerAutotune command, got {other:?}"),
    }
}
```

Add unit tests in `ironmlx/src/cli/scheduler_autotune_calibrate.rs`:

```rust
#[test]
fn parse_candidate_config_accepts_all_required_fields() {
    let config = parse_candidate_config(
        "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768",
    )
    .expect("candidate should parse");

    assert_eq!(config.b_max, 2);
    assert_eq!(config.prefill_chunk_size, 1024);
    assert_eq!(config.admission_deadline_ms, 5);
    assert_eq!(config.admission_queue_max, 32);
    assert_eq!(config.max_cache_cap, 32768);
}

#[test]
fn parse_candidate_config_rejects_missing_field() {
    let err = parse_candidate_config(
        "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32",
    )
    .expect_err("missing max_cache_cap should fail");

    assert!(err.contains("max_cache_cap"), "unexpected error: {err}");
}
```

- [ ] **Step 1.2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx scheduler_autotune_calibrate -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_calibrate_subcommand_parses_required_matrix -- --nocapture
```

Expected: compile failure because the calibrate module/action does not exist.

- [ ] **Step 1.3: Implement calibrate args and parser**

Create `ironmlx/src/cli/scheduler_autotune_calibrate.rs` with:

```rust
use std::path::PathBuf;

use clap::Args;

use crate::core::scheduler_autotune::SchedulerAutotuneProfileConfig;
use crate::Result;

const DEFAULT_PORT: u16 = 18080;
const DEFAULT_STARTUP_TIMEOUT_SEC: u64 = 300;

#[derive(Args, Debug)]
pub struct SchedulerAutotuneCalibrateArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub model_name: String,
    #[arg(long)]
    pub hardware_label: String,
    #[arg(long)]
    pub iron_bench_bin: PathBuf,
    #[arg(long)]
    pub output_dir: PathBuf,
    #[arg(long = "candidate", required = true, value_parser = parse_candidate_config)]
    pub candidates: Vec<SchedulerAutotuneProfileConfig>,
    #[arg(long, value_delimiter = ',', required = true)]
    pub prompt_len: Vec<usize>,
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,
    #[arg(long, value_delimiter = ',', required = true)]
    pub concurrency: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    pub runs: usize,
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    #[arg(long, default_value_t = 30)]
    pub duration: u64,
    #[arg(long, default_value_t = 5)]
    pub warmup_duration: u64,
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    #[arg(long, default_value_t = DEFAULT_STARTUP_TIMEOUT_SEC)]
    pub startup_timeout_sec: u64,
    #[arg(long)]
    pub write_profile: Option<PathBuf>,
}

pub fn parse_candidate_config(
    raw: &str,
) -> std::result::Result<SchedulerAutotuneProfileConfig, String> {
    let mut b_max = None;
    let mut prefill_chunk_size = None;
    let mut admission_deadline_ms = None;
    let mut admission_queue_max = None;
    let mut max_cache_cap = None;

    for part in raw.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("candidate item must be key=value: {part}"))?;
        match key {
            "b_max" => b_max = Some(value.parse::<usize>().map_err(|e| e.to_string())?),
            "prefill_chunk_size" => {
                prefill_chunk_size = Some(value.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "admission_deadline_ms" => {
                admission_deadline_ms = Some(value.parse::<u64>().map_err(|e| e.to_string())?)
            }
            "admission_queue_max" => {
                admission_queue_max = Some(value.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "max_cache_cap" => {
                max_cache_cap = Some(value.parse::<usize>().map_err(|e| e.to_string())?)
            }
            other => return Err(format!("unknown candidate key: {other}")),
        }
    }

    Ok(SchedulerAutotuneProfileConfig {
        b_max: b_max.ok_or_else(|| "missing b_max".to_string())?,
        prefill_chunk_size: prefill_chunk_size
            .ok_or_else(|| "missing prefill_chunk_size".to_string())?,
        admission_deadline_ms: admission_deadline_ms
            .ok_or_else(|| "missing admission_deadline_ms".to_string())?,
        admission_queue_max: admission_queue_max
            .ok_or_else(|| "missing admission_queue_max".to_string())?,
        max_cache_cap: max_cache_cap.ok_or_else(|| "missing max_cache_cap".to_string())?,
    })
}

pub fn run(_args: SchedulerAutotuneCalibrateArgs) -> Result<()> {
    anyhow::bail!("scheduler-autotune calibrate runner is added in Task 4")
}
```

In `ironmlx/src/cli/mod.rs`, add:

```rust
mod scheduler_autotune_calibrate;
```

In `ironmlx/src/cli/scheduler_autotune.rs`, add:

```rust
Calibrate(super::scheduler_autotune_calibrate::SchedulerAutotuneCalibrateArgs),
```

and dispatch:

```rust
Some(SchedulerAutotuneAction::Calibrate(calibrate)) => {
    super::scheduler_autotune_calibrate::run(calibrate)
}
```

- [ ] **Step 1.4: Verify GREEN**

Run the same two test commands. Expected: parser and candidate tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add ironmlx/src/cli/mod.rs ironmlx/src/cli/scheduler_autotune.rs ironmlx/src/cli/scheduler_autotune_calibrate.rs
git commit -m "feat: parse scheduler autotune calibrate"
```

### Task 2: Artifact Paths And Command Builders

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [ ] **Step 2.1: Write failing pure command tests**

Add tests:

```rust
#[test]
fn candidate_artifact_path_includes_candidate_and_concurrency() {
    let path = candidate_artifact_path(Path::new("/tmp/out"), 3, 2);
    assert_eq!(path.to_string_lossy(), "/tmp/out/candidate-003-c2.json");
}

#[test]
fn build_serve_invocation_includes_scheduler_config() {
    let args = sample_args();
    let command = build_serve_invocation(Path::new("/tmp/ironmlx"), &args, profile_config(), 19000);

    assert_eq!(command.program.to_string_lossy(), "/tmp/ironmlx");
    assert!(command.args.contains(&"serve".to_string()));
    assert!(command.args.contains(&"--b-max".to_string()));
    assert!(command.args.contains(&"2".to_string()));
    assert!(command.args.contains(&"--prefill-chunk-size".to_string()));
    assert!(command.args.contains(&"1024".to_string()));
    assert!(command.args.contains(&"--port".to_string()));
    assert!(command.args.contains(&"19000".to_string()));
}

#[test]
fn build_iron_bench_invocation_uses_sequential_mode_for_concurrency_one() {
    let args = sample_args();
    let command = build_iron_bench_invocation(
        &args,
        profile_config(),
        1,
        "http://127.0.0.1:18080",
    );

    assert!(!command.args.contains(&"--concurrent".to_string()));
    assert!(command.args.contains(&"--runs".to_string()));
    assert!(command.args.contains(&"5".to_string()));
    assert!(command.args.contains(&"--warmup".to_string()));
    assert!(command.args.contains(&"1".to_string()));
}

#[test]
fn build_iron_bench_invocation_uses_concurrent_mode_for_concurrency_above_one() {
    let args = sample_args();
    let command = build_iron_bench_invocation(
        &args,
        profile_config(),
        2,
        "http://127.0.0.1:18080",
    );

    assert!(command.args.contains(&"--concurrent".to_string()));
    assert!(command.args.contains(&"2".to_string()));
    assert!(command.args.contains(&"--duration".to_string()));
    assert!(command.args.contains(&"30".to_string()));
    assert!(command.args.contains(&"--warmup-duration".to_string()));
    assert!(command.args.contains(&"5".to_string()));
}
```

- [ ] **Step 2.2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::scheduler_autotune_calibrate:: -- --nocapture
```

Expected: compile failure because command builder helpers do not exist.

- [ ] **Step 2.3: Implement pure helpers**

Add:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessInvocation {
    program: PathBuf,
    args: Vec<String>,
}
```

Add helper functions:

```rust
fn candidate_artifact_path(output_dir: &Path, candidate_idx: usize, concurrency: usize) -> PathBuf;
fn serve_log_path(output_dir: &Path, candidate_idx: usize) -> PathBuf;
fn build_serve_invocation(
    ironmlx_bin: &Path,
    args: &SchedulerAutotuneCalibrateArgs,
    config: SchedulerAutotuneProfileConfig,
    port: u16,
) -> ProcessInvocation;
fn build_iron_bench_invocation(
    args: &SchedulerAutotuneCalibrateArgs,
    config: SchedulerAutotuneProfileConfig,
    concurrency: usize,
    target_url: &str,
) -> ProcessInvocation;
```

The `iron-bench` invocation must include `--format autotune-json` and all required `--autotune-*` config flags.

- [ ] **Step 2.4: Verify GREEN**

Run the same module test command. Expected: command builder tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add ironmlx/src/cli/scheduler_autotune_calibrate.rs
git commit -m "feat: build scheduler autotune calibrate commands"
```

### Task 3: Subprocess Runner

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [ ] **Step 3.1: Write failing non-live orchestration tests**

Add tests for final output path names and health URL:

```rust
#[test]
fn health_url_uses_localhost_and_selected_port() {
    assert_eq!(health_url(19000), "http://127.0.0.1:19000/health");
}

#[test]
fn final_artifact_paths_are_stable() {
    let paths = FinalArtifactPaths::new(Path::new("/tmp/out"), Some(PathBuf::from("/tmp/profile.json")));

    assert_eq!(paths.calibration.to_string_lossy(), "/tmp/out/calibration.json");
    assert_eq!(paths.selection_json.to_string_lossy(), "/tmp/out/selection.json");
    assert_eq!(paths.selection_text.to_string_lossy(), "/tmp/out/selection.txt");
    assert_eq!(
        paths.runtime_profile.as_ref().expect("profile").to_string_lossy(),
        "/tmp/profile.json"
    );
}
```

- [ ] **Step 3.2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::scheduler_autotune_calibrate:: -- --nocapture
```

Expected: compile failure until the helper types exist.

- [ ] **Step 3.3: Implement subprocess runner**

Implement:

```rust
struct FinalArtifactPaths {
    calibration: PathBuf,
    selection_json: PathBuf,
    selection_text: PathBuf,
    runtime_profile: Option<PathBuf>,
}
```

Implement:

```rust
fn health_url(port: u16) -> String;
fn wait_for_health(url: &str, timeout: Duration) -> Result<()>;
fn spawn_serve(invocation: &ProcessInvocation, log_path: &Path) -> Result<ServeChild>;
fn run_iron_bench(invocation: &ProcessInvocation, output_json: &Path, stderr_log: &Path) -> Result<()>;
```

`ServeChild` owns `std::process::Child` and kills/waits in `Drop` if the child is still running.

`wait_for_health` uses a small Tokio runtime and `reqwest::Client` to poll every 500 ms until HTTP success or timeout.

`run_iron_bench` writes stdout to the candidate JSON path and stderr to a `.stderr.log` file. On non-zero exit, return an error containing the exit status and stderr log path.

- [ ] **Step 3.4: Verify GREEN**

Run the same module test command. Expected: non-live orchestration tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add ironmlx/src/cli/scheduler_autotune_calibrate.rs
git commit -m "feat: run scheduler autotune calibrate subprocesses"
```

### Task 4: Merge, Select, And Profile Output

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [ ] **Step 4.1: Write failing output tests**

Add a pure test that builds two small calibration inputs, merges them through a helper, and verifies output strings:

```rust
#[test]
fn write_final_outputs_writes_calibration_selection_and_profile() {
    let dir = std::env::temp_dir().join(format!(
        "ironmlx-autotune-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp output dir");

    let profile_path = dir.join("scheduler-profile.json");
    let artifacts = FinalArtifactPaths::new(&dir, Some(profile_path.clone()));
    let inputs = vec![sample_calibration(profile_config())];

    write_final_outputs(inputs, &artifacts).expect("write outputs");

    assert!(artifacts.calibration.exists());
    assert!(artifacts.selection_json.exists());
    assert!(artifacts.selection_text.exists());
    assert!(profile_path.exists());

    let profile_raw = std::fs::read_to_string(profile_path).expect("read profile");
    assert!(profile_raw.contains("\"schema_version\": 1"));

    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 4.2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::scheduler_autotune_calibrate::write_final_outputs_writes_calibration_selection_and_profile -- --nocapture
```

Expected: compile failure until `write_final_outputs` exists.

- [ ] **Step 4.3: Implement final output path**

After all candidate JSON files are produced, read them as `SchedulerAutotuneCalibrationInput`, call:

```rust
merge_scheduler_autotune_calibrations(inputs, SchedulerAutotuneMergeOptions::default())?;
select_scheduler_autotune_profile(merged.clone());
build_scheduler_autotune_runtime_profile(&selection)?;
```

Write:

- pretty `calibration.json` plus newline;
- pretty `selection.json` plus newline;
- `selection.render_text()` to `selection.txt`;
- pretty runtime profile plus newline when `runtime_profile` is present.

- [ ] **Step 4.4: Wire `run(args)` end to end**

`run(args)` should:

1. create `output_dir`;
2. get `std::env::current_exe()` for the `ironmlx serve` subprocess;
3. for each candidate, spawn serve and wait for health;
4. run all requested concurrency scenarios;
5. drop the serve child before the next candidate;
6. write final outputs.

- [ ] **Step 4.5: Verify GREEN**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::scheduler_autotune_calibrate:: -- --nocapture
```

Expected: all calibrate unit tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add ironmlx/src/cli/scheduler_autotune_calibrate.rs
git commit -m "feat: write scheduler autotune calibrate outputs"
```

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md`
- Modify: `docs/superpowers/specs/2026-06-02-scheduler-autotune-calibrate-design.md`

- [ ] **Step 5.1: Update Chinese docs**

Document the calibrate workflow in the existing scheduler/autotune research doc. Include:

```bash
cargo build --release -p ironmlx -p iron-bench
cargo run --release -p ironmlx -- \
  scheduler-autotune calibrate \
  --model /path/to/model \
  --model-name GLM-4.7-flash-4bit \
  --hardware-label m5-max-128g \
  --iron-bench-bin target/release/iron-bench \
  --output-dir reports/scheduler-autotune/glm47-m5max \
  --candidate b_max=1,prefill_chunk_size=2048,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768 \
  --candidate b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768 \
  --prompt-len 1024,2048,4096 \
  --max-tokens 128 \
  --concurrency 1,2 \
  --write-profile reports/scheduler-autotune/glm47-m5max/scheduler-profile.json
```

- [ ] **Step 5.2: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p ironmlx scheduler_autotune_calibrate -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_merge -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_calibrate_subcommand_parses_required_matrix -- --nocapture
git diff --check
```

- [ ] **Step 5.3: Commit**

```bash
git add ironmlx/src/cli/mod.rs ironmlx/src/cli/scheduler_autotune.rs ironmlx/src/cli/scheduler_autotune_calibrate.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/specs/2026-06-02-scheduler-autotune-calibrate-design.md
git commit -m "feat: add scheduler autotune calibrate runner"
```
