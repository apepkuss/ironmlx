use ironmlx::core::scheduler_autotune::{
    merge_scheduler_autotune_calibrations, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneMeasurement, SchedulerAutotuneMergeOptions, SchedulerAutotuneObjective,
    SchedulerAutotuneProfileConfig,
};

fn config(b_max: usize, chunk: usize) -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        b_max,
        prefill_chunk_size: chunk,
        admission_deadline_ms: 5,
        admission_queue_max: 32,
        max_cache_cap: 32768,
    }
}

fn measurement(
    config: SchedulerAutotuneProfileConfig,
    prompt_len: usize,
    max_new_tokens: usize,
    concurrency: usize,
) -> SchedulerAutotuneMeasurement {
    SchedulerAutotuneMeasurement {
        config,
        prompt_len,
        max_new_tokens,
        concurrency,
        ttft_ms_p95: 100.0,
        itl_ms_p95: 12.0,
        e2e_s_p95: 2.4,
        tokens_per_sec: 80.0,
        memory_budget_ok: true,
        cached_tokens_warning: false,
    }
}

fn input(
    model_name: &str,
    hardware_label: &str,
    measurements: Vec<SchedulerAutotuneMeasurement>,
) -> SchedulerAutotuneCalibrationInput {
    SchedulerAutotuneCalibrationInput {
        schema_version: 1,
        model_name: model_name.to_string(),
        hardware_label: hardware_label.to_string(),
        objective: SchedulerAutotuneObjective::agent_default(),
        measurements,
    }
}

#[test]
fn merge_calibrations_preserves_metadata_and_concatenates_measurements() {
    let first_config = config(1, 2048);
    let second_config = config(2, 1024);
    let first = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![
            measurement(first_config, 2048, 128, 1),
            measurement(first_config, 2048, 128, 2),
        ],
    );
    let second = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![
            measurement(second_config, 2048, 128, 1),
            measurement(second_config, 2048, 128, 2),
        ],
    );

    let merged = merge_scheduler_autotune_calibrations(
        vec![first, second],
        SchedulerAutotuneMergeOptions::default(),
    )
    .expect("merge should succeed");

    assert_eq!(merged.schema_version, 1);
    assert_eq!(merged.model_name, "GLM-4.7-flash-4bit");
    assert_eq!(merged.hardware_label, "m3-max");
    assert_eq!(
        merged.objective,
        SchedulerAutotuneObjective::agent_default()
    );
    assert_eq!(merged.measurements.len(), 4);
}

#[test]
fn merge_calibrations_rejects_model_name_mismatch() {
    let first = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![measurement(config(1, 2048), 2048, 128, 1)],
    );
    let second = input(
        "Qwen3.5",
        "m3-max",
        vec![measurement(config(2, 1024), 2048, 128, 1)],
    );

    let error = merge_scheduler_autotune_calibrations(
        vec![first, second],
        SchedulerAutotuneMergeOptions::default(),
    )
    .expect_err("merge should reject model mismatch");

    assert!(
        error.to_string().contains("model_name"),
        "unexpected error: {error}"
    );
}

#[test]
fn merge_calibrations_rejects_incomplete_scenario_coverage() {
    let first_config = config(1, 2048);
    let second_config = config(2, 1024);
    let first = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![
            measurement(first_config, 2048, 128, 1),
            measurement(first_config, 2048, 128, 2),
        ],
    );
    let second = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![measurement(second_config, 2048, 128, 1)],
    );

    let error = merge_scheduler_autotune_calibrations(
        vec![first, second],
        SchedulerAutotuneMergeOptions::default(),
    )
    .expect_err("merge should reject incomplete coverage");

    assert!(
        error.to_string().contains("scenario coverage"),
        "unexpected error: {error}"
    );
}

#[test]
fn merge_calibrations_allows_incomplete_coverage_when_requested() {
    let first_config = config(1, 2048);
    let second_config = config(2, 1024);
    let first = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![
            measurement(first_config, 2048, 128, 1),
            measurement(first_config, 2048, 128, 2),
        ],
    );
    let second = input(
        "GLM-4.7-flash-4bit",
        "m3-max",
        vec![measurement(second_config, 2048, 128, 1)],
    );

    let merged = merge_scheduler_autotune_calibrations(
        vec![first, second],
        SchedulerAutotuneMergeOptions {
            require_complete_coverage: false,
        },
    )
    .expect("merge should allow incomplete coverage when requested");

    assert_eq!(merged.measurements.len(), 3);
}
