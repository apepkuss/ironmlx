use ironmlx::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, select_scheduler_autotune_profile,
    SchedulerAutotuneCalibrationInput, SchedulerAutotuneMeasurement, SchedulerAutotuneObjective,
    SchedulerAutotuneProfileConfig,
};

fn config(
    b_max: usize,
    prefill_chunk_size: usize,
    admission_deadline_ms: u64,
) -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        b_max,
        prefill_chunk_size,
        admission_deadline_ms,
        admission_queue_max: 32,
        max_cache_cap: 32768,
    }
}

#[allow(clippy::too_many_arguments)]
fn measurement(
    config: SchedulerAutotuneProfileConfig,
    prompt_len: usize,
    max_new_tokens: usize,
    concurrency: usize,
    ttft_ms_p95: f64,
    itl_ms_p95: f64,
    e2e_s_p95: f64,
    tokens_per_sec: f64,
) -> SchedulerAutotuneMeasurement {
    SchedulerAutotuneMeasurement {
        config,
        prompt_len,
        max_new_tokens,
        concurrency,
        ttft_ms_p95,
        itl_ms_p95,
        e2e_s_p95,
        tokens_per_sec,
        memory_budget_ok: true,
        cached_tokens_warning: false,
    }
}

fn input(measurements: Vec<SchedulerAutotuneMeasurement>) -> SchedulerAutotuneCalibrationInput {
    SchedulerAutotuneCalibrationInput {
        schema_version: 1,
        model_name: "test-model".to_string(),
        hardware_label: "test-host".to_string(),
        objective: SchedulerAutotuneObjective::agent_default(),
        measurements,
    }
}

#[test]
fn profile_selection_prefers_agent_balanced_config_over_ttft_only_winner() {
    let ttft_only = config(1, 2048, 2);
    let balanced = config(2, 1024, 5);
    let selection = select_scheduler_autotune_profile(input(vec![
        measurement(ttft_only, 2048, 128, 1, 90.0, 18.0, 2.8, 72.0),
        measurement(ttft_only, 2048, 128, 2, 140.0, 28.0, 5.8, 62.0),
        measurement(balanced, 2048, 128, 1, 110.0, 12.0, 2.4, 86.0),
        measurement(balanced, 2048, 128, 2, 125.0, 14.0, 3.1, 112.0),
    ]));

    let selected = selection
        .selected
        .as_ref()
        .expect("expected selected profile");
    assert_eq!(selected.config, balanced);
    assert!(selection.diagnose_only);
    assert!(selection.render_text().contains("diagnose-only"));
}

#[test]
fn profile_selection_rejects_candidates_missing_scenario_coverage() {
    let complete = config(2, 1024, 5);
    let incomplete = config(4, 1024, 5);
    let selection = select_scheduler_autotune_profile(input(vec![
        measurement(complete, 2048, 128, 1, 120.0, 12.0, 2.5, 90.0),
        measurement(complete, 2048, 128, 2, 135.0, 15.0, 3.4, 120.0),
        measurement(incomplete, 2048, 128, 1, 95.0, 11.0, 2.3, 100.0),
    ]));

    let selected = selection.selected.expect("expected selected profile");
    assert_eq!(selected.config, complete);
    assert!(selection
        .rejected
        .iter()
        .any(|item| item.config == incomplete && item.code == "missing_scenario_coverage"));
}

#[test]
fn profile_selection_rejects_memory_unsafe_candidates() {
    let safe = config(1, 2048, 2);
    let unsafe_config = config(4, 2048, 2);
    let mut unsafe_row = measurement(unsafe_config, 4096, 128, 1, 80.0, 10.0, 2.0, 130.0);
    unsafe_row.memory_budget_ok = false;

    let selection = select_scheduler_autotune_profile(input(vec![
        measurement(safe, 4096, 128, 1, 130.0, 15.0, 3.2, 90.0),
        unsafe_row,
    ]));

    let selected = selection.selected.expect("expected selected profile");
    assert_eq!(selected.config, safe);
    assert!(selection
        .rejected
        .iter()
        .any(|item| item.config == unsafe_config && item.code == "memory_budget_unsafe"));
}

#[test]
fn profile_selection_warns_when_agent_long_prompt_or_concurrency_coverage_is_absent() {
    let only_short_single = config(1, 2048, 2);
    let selection = select_scheduler_autotune_profile(input(vec![measurement(
        only_short_single,
        512,
        128,
        1,
        60.0,
        10.0,
        1.8,
        100.0,
    )]));

    assert!(selection
        .warnings
        .iter()
        .any(|item| item.code == "no_long_prompt_coverage"));
    assert!(selection
        .warnings
        .iter()
        .any(|item| item.code == "no_concurrent_coverage"));
}

#[test]
fn runtime_profile_uses_selected_config_and_metadata() {
    let selected_config = config(2, 1024, 5);
    let selection = select_scheduler_autotune_profile(input(vec![measurement(
        selected_config,
        2048,
        128,
        1,
        90.0,
        12.0,
        2.4,
        100.0,
    )]));

    let profile =
        build_scheduler_autotune_runtime_profile(&selection).expect("expected runtime profile");

    assert_eq!(profile.schema_version, 1);
    assert_eq!(profile.model_name, "test-model");
    assert_eq!(profile.hardware_label, "test-host");
    assert_eq!(profile.config, selected_config);
}

#[test]
fn runtime_profile_requires_selected_candidate() {
    let unsafe_config = config(4, 2048, 5);
    let mut unsafe_row = measurement(unsafe_config, 2048, 128, 1, 90.0, 12.0, 2.4, 100.0);
    unsafe_row.memory_budget_ok = false;
    let selection = select_scheduler_autotune_profile(input(vec![unsafe_row]));

    let error = build_scheduler_autotune_runtime_profile(&selection)
        .expect_err("profile export should require a selected candidate");

    assert!(
        error.to_string().contains("selected"),
        "unexpected error: {error}"
    );
}
