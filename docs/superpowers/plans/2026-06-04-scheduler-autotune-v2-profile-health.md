# Scheduler Autotune V2 Profile Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scheduler autotune profile metadata and health checks so stored profiles can be automatically applied with clear trust/staleness diagnostics.

**Architecture:** Extend `core::scheduler_autotune` with v2 runtime profile metadata and a pure health evaluator. Reuse the existing profile store for automatic lookup, add a `profile doctor` CLI path, and surface health warnings during `serve` startup without adding runtime auto-tuning.

**Tech Stack:** Rust, serde, serde_json, clap, existing `SchedulerProfileStore`, existing scheduler/autotune selector structs.

---

## File Structure

- Modify `ironmlx/src/core/scheduler_autotune.rs` — bump profile schema, add metadata, scenario list on selection, runtime profile builder with deterministic timestamp helper, health evaluator, and text rendering.
- Modify `ironmlx/tests/scheduler_autotune_profile.rs` — add TDD tests for metadata, scenario coverage, and health statuses.
- Modify `ironmlx/src/cli/scheduler_autotune.rs` — add `scheduler-autotune profile doctor --model`.
- Modify `ironmlx/src/cli/mod.rs` — add parser test for the doctor command.
- Modify `ironmlx/src/cli/serve.rs` — evaluate and log profile health when a profile is explicitly or automatically loaded.
- Modify `ironmlx/src/cli/scheduler_autotune_calibrate.rs` tests if runtime profile construction expectations need schema/metadata updates.
- Modify `ironmlx/src/cli/scheduler_profile_store.rs` tests if runtime profile fixtures need metadata.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — correct outdated runtime apply wording.
- Create `docs/superpowers/specs/2026-06-04-scheduler-autotune-v2-profile-health.md` — Chinese v2 design.

### Task 1: Runtime Profile Metadata

**Files:**
- Modify: `ironmlx/src/core/scheduler_autotune.rs`
- Modify: `ironmlx/tests/scheduler_autotune_profile.rs`

- [x] **Step 1: Write failing tests**

Add tests:

```rust
#[test]
fn profile_selection_records_scenario_coverage_for_runtime_metadata() {
    let selected_config = config(1, 2048, 5);
    let selection = select_scheduler_autotune_profile(input(vec![
        measurement(selected_config, 1024, 128, 1, 100.0, 10.0, 2.0, 90.0),
        measurement(selected_config, 4096, 128, 2, 200.0, 11.0, 4.0, 80.0),
    ]));

    assert_eq!(selection.scenarios.len(), 2);
    assert!(selection.scenarios.iter().any(|scenario| {
        scenario.prompt_len == 4096 && scenario.max_new_tokens == 128 && scenario.concurrency == 2
    }));
}

#[test]
fn runtime_profile_metadata_captures_selection_context() {
    let selected_config = config(1, 2048, 5);
    let selection = select_scheduler_autotune_profile(input(vec![
        measurement(selected_config, 1024, 128, 1, 100.0, 10.0, 2.0, 90.0),
        measurement(selected_config, 4096, 128, 2, 200.0, 11.0, 4.0, 80.0),
    ]));

    let profile = build_scheduler_autotune_runtime_profile_at(&selection, 1811606400000)
        .expect("expected runtime profile");

    assert_eq!(profile.schema_version, SCHEDULER_AUTOTUNE_SCHEMA_VERSION);
    assert_eq!(profile.metadata.created_at_unix_ms, 1811606400000);
    assert_eq!(profile.metadata.selection_profile, SchedulerAutotuneSelectionProfile::AgentLongPrompt);
    assert_eq!(profile.metadata.scenario_coverage.len(), 2);
    assert_eq!(profile.metadata.candidate_count, 1);
    assert_eq!(profile.metadata.rejected_count, 0);
    assert!(profile.metadata.selected_score.is_finite());
}
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_profile profile_selection_records_scenario_coverage_for_runtime_metadata -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_profile runtime_profile_metadata_captures_selection_context -- --nocapture
```

Expected: compile failure because `selection.scenarios`, `metadata`, and `build_scheduler_autotune_runtime_profile_at` do not exist.

- [x] **Step 3: Implement metadata**

Implement:

```rust
pub const SCHEDULER_AUTOTUNE_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeProfileMetadata {
    pub created_at_unix_ms: u64,
    pub ironmlx_version: String,
    pub selection_profile: SchedulerAutotuneSelectionProfile,
    pub objective: SchedulerAutotuneObjective,
    pub scenario_coverage: Vec<SchedulerAutotuneScenario>,
    pub selected_score: f64,
    pub candidate_count: usize,
    pub rejected_count: usize,
    pub selection_warnings: Vec<SchedulerAutotuneSelectionNote>,
}
```

Add `scenarios: Vec<SchedulerAutotuneScenario>` to `SchedulerAutotuneProfileSelection`, populate it from `required_scenarios`, and add `metadata` to `SchedulerAutotuneRuntimeProfile`.

- [x] **Step 4: Run tests to verify GREEN**

Run the same two tests. Expected: both pass.

### Task 2: Profile Health Evaluator

**Files:**
- Modify: `ironmlx/src/core/scheduler_autotune.rs`
- Modify: `ironmlx/tests/scheduler_autotune_profile.rs`

- [x] **Step 1: Write failing tests**

Add tests:

```rust
#[test]
fn profile_health_reports_healthy_for_matching_fresh_agent_coverage() {
    let profile = runtime_profile_with_metadata(1811606400000, vec![
        SchedulerAutotuneScenario { prompt_len: 1024, max_new_tokens: 128, concurrency: 1 },
        SchedulerAutotuneScenario { prompt_len: 4096, max_new_tokens: 128, concurrency: 2 },
    ]);

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "test-model",
        expected_hardware_label: "test-host",
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000 + 1000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Healthy);
    assert!(report.notes.iter().all(|note| note.level == SchedulerAutotuneProfileHealthLevel::Info));
}

#[test]
fn profile_health_warns_for_stale_version_and_missing_concurrency_coverage() {
    let mut profile = runtime_profile_with_metadata(1811606400000, vec![
        SchedulerAutotuneScenario { prompt_len: 1024, max_new_tokens: 128, concurrency: 1 },
    ]);
    profile.metadata.ironmlx_version = "0.0.0-test".to_string();

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "other-model-name",
        expected_hardware_label: "test-host",
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000 + 31 * 24 * 60 * 60 * 1000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Warning);
    assert!(report.notes.iter().any(|note| note.code == "profile_stale"));
    assert!(report.notes.iter().any(|note| note.code == "ironmlx_version_changed"));
    assert!(report.notes.iter().any(|note| note.code == "model_name_mismatch"));
    assert!(report.notes.iter().any(|note| note.code == "no_concurrent_coverage"));
}

#[test]
fn profile_health_invalidates_schema_and_hardware_mismatch() {
    let mut profile = runtime_profile_with_metadata(1811606400000, vec![
        SchedulerAutotuneScenario { prompt_len: 4096, max_new_tokens: 128, concurrency: 2 },
    ]);
    profile.schema_version = SCHEDULER_AUTOTUNE_SCHEMA_VERSION + 1;
    profile.hardware_label = "other-host".to_string();

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "test-model",
        expected_hardware_label: "test-host",
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Invalid);
    assert!(report.notes.iter().any(|note| note.code == "schema_version_mismatch"));
    assert!(report.notes.iter().any(|note| note.code == "hardware_label_mismatch"));
}
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_profile profile_health_ -- --nocapture
```

Expected: compile failure because health types and evaluator do not exist.

- [x] **Step 3: Implement health evaluator**

Add `SchedulerAutotuneProfileHealthInput`, `SchedulerAutotuneProfileHealthStatus`, `SchedulerAutotuneProfileHealthLevel`, `SchedulerAutotuneProfileHealthNote`, `SchedulerAutotuneProfileHealthReport`, `evaluate_scheduler_autotune_profile_health`, and `render_text`.

- [x] **Step 4: Run tests to verify GREEN**

Run the same `profile_health_` tests. Expected: all pass.

### Task 3: Profile Doctor CLI

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune.rs`
- Modify: `ironmlx/src/cli/mod.rs`

- [x] **Step 1: Write failing parser test**

Add parser coverage:

```rust
#[test]
fn scheduler_autotune_profile_doctor_subcommand_parses_model() {
    let cli = Cli::try_parse_from([
        "ironmlx",
        "scheduler-autotune",
        "profile",
        "doctor",
        "--model",
        "/tmp/model",
    ])
    .expect("parse cli");

    match cli.command {
        Command::SchedulerAutotune(args) => match args.action {
            Some(super::scheduler_autotune::SchedulerAutotuneAction::Profile(profile)) => {
                match profile.action {
                    super::scheduler_autotune::SchedulerAutotuneProfileAction::Doctor(doctor) => {
                        assert_eq!(doctor.model.to_string_lossy(), "/tmp/model");
                        assert_eq!(doctor.max_age_days, 30);
                    }
                    other => panic!("expected Doctor action, got {other:?}"),
                }
            }
            other => panic!("expected Profile action, got {other:?}"),
        },
        other => panic!("expected SchedulerAutotune command, got {other:?}"),
    }
}
```

- [x] **Step 2: Run parser test to verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::tests::scheduler_autotune_profile_doctor_subcommand_parses_model -- --nocapture
```

Expected: compile failure because `Doctor` does not exist.

- [x] **Step 3: Implement doctor command**

Add `Doctor(SchedulerAutotuneProfileDoctorArgs)` under `SchedulerAutotuneProfileAction`, with `--model`, `--format text|json`, and `--max-age-days` defaulting to `30`. Use `SchedulerProfileStore::find_profile`, read the selected profile JSON, run the health evaluator, and print either text or JSON.

- [x] **Step 4: Run parser test to verify GREEN**

Run the same parser test. Expected: pass.

### Task 4: Serve Startup Health Logging

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`

- [x] **Step 1: Write failing resolver/logging helper tests**

Add tests around a pure helper:

```rust
#[test]
fn scheduler_profile_health_warning_does_not_prevent_profile_resolution() {
    let profile = runtime_profile_with_metadata(1811606400000, vec![
        SchedulerAutotuneScenario { prompt_len: 1024, max_new_tokens: 128, concurrency: 1 },
    ]);
    let args = ServeArgs::parse_from(["serve", "--model", "/tmp/model"]);

    let checked = check_loaded_scheduler_profile_health(
        &profile,
        "different-model-name",
        "test-host",
        1811606400000 + 31 * 24 * 60 * 60 * 1000,
    )
    .expect("warning health should not fail");

    assert_eq!(checked.status, SchedulerAutotuneProfileHealthStatus::Warning);
    assert!(resolve_scheduler_runtime_profile(&args, Some(&profile)).is_ok());
}
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::serve::scheduler_profile_tests::scheduler_profile_health_warning_does_not_prevent_profile_resolution -- --nocapture
```

Expected: compile failure because the helper is missing.

- [x] **Step 3: Implement serve health logging**

Before applying a loaded profile, call the health evaluator with detected hardware label, current version, and current timestamp. Log `info` for healthy, `warn` for warning, and return an error for invalid explicit profiles. Automatic store profiles should not normally be invalid because store lookup filters schema and hardware; if invalid is encountered, skip it and fall back to CLI/default config.

- [x] **Step 4: Run test to verify GREEN**

Run the same serve test. Expected: pass.

### Task 5: Docs And Fixture Updates

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md`
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`
- Modify: `ironmlx/src/cli/scheduler_profile_store.rs`
- Modify: tests that construct `SchedulerAutotuneRuntimeProfile`

- [x] **Step 1: Update stale docs**

Replace outdated statements that say profile application is only explicit. Document that stored local profiles are auto-loaded by `serve`, and that v2 adds health diagnostics.

- [x] **Step 2: Update Rust fixtures**

Every `SchedulerAutotuneRuntimeProfile` literal must include `metadata`, or use a local helper that builds a valid profile at a fixed timestamp.

- [x] **Step 3: Run targeted tests**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_profile_doctor_subcommand_parses_model -- --nocapture
cargo test --release -p ironmlx cli::serve::scheduler_profile_tests::scheduler_profile -- --nocapture
```

Expected: all targeted tests pass.

### Task 6: Required Rust Verification And Commit

**Files:**
- All files touched in previous tasks.

- [x] **Step 1: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
git diff --check
```

Expected: all commands exit 0.

- [x] **Step 2: Commit**

Run:

```bash
git add ironmlx/src/core/scheduler_autotune.rs ironmlx/tests/scheduler_autotune_profile.rs ironmlx/src/cli/scheduler_autotune.rs ironmlx/src/cli/mod.rs ironmlx/src/cli/serve.rs ironmlx/src/cli/scheduler_autotune_calibrate.rs ironmlx/src/cli/scheduler_profile_store.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/specs/2026-06-04-scheduler-autotune-v2-profile-health.md docs/superpowers/plans/2026-06-04-scheduler-autotune-v2-profile-health.md
git commit -m "feat: add scheduler autotune profile health"
```
