use ironmlx::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, select_scheduler_autotune_profile,
    select_scheduler_autotune_profile_with_options, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneCandidateScore, SchedulerAutotuneMeasurement, SchedulerAutotuneObjective,
    SchedulerAutotuneProfileConfig, SchedulerAutotuneProfileSelection,
    SchedulerAutotuneRuntimeRequest, SchedulerAutotuneScenario, SchedulerAutotuneScenarioOverride,
    SchedulerAutotuneSelectionOptions, SchedulerAutotuneSelectionProfile,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
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
        decode_cadence_mid_chunk_cap: 256,
    }
}

fn config_with_cadence(
    b_max: usize,
    prefill_chunk_size: usize,
    admission_deadline_ms: u64,
    decode_cadence_mid_chunk_cap: usize,
) -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        decode_cadence_mid_chunk_cap,
        ..config(b_max, prefill_chunk_size, admission_deadline_ms)
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
        early_itl_ms_p95: itl_ms_p95,
        memory_budget_ok: true,
        cached_tokens_warning: false,
    }
}

fn input(measurements: Vec<SchedulerAutotuneMeasurement>) -> SchedulerAutotuneCalibrationInput {
    SchedulerAutotuneCalibrationInput {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "test-model".to_string(),
        hardware_label: "test-host".to_string(),
        objective: SchedulerAutotuneObjective::agent_default(),
        measurements,
    }
}

fn candidate_score(config: SchedulerAutotuneProfileConfig) -> SchedulerAutotuneCandidateScore {
    SchedulerAutotuneCandidateScore {
        config,
        score: 1.0,
        scenario_count: 1,
        mean_ttft_norm: 1.0,
        mean_itl_norm: 1.0,
        mean_early_itl_norm: 1.0,
        mean_e2e_norm: 1.0,
        mean_throughput_norm: 1.0,
    }
}

fn scenario_override(
    prompt_len: usize,
    max_new_tokens: usize,
    concurrency: usize,
    config: SchedulerAutotuneProfileConfig,
) -> SchedulerAutotuneScenarioOverride {
    SchedulerAutotuneScenarioOverride {
        scenario: SchedulerAutotuneScenario {
            prompt_len,
            max_new_tokens,
            concurrency,
        },
        config,
        score: 1.0,
        baseline_score: 1.1,
    }
}

fn selection_with_overrides(
    selected_config: SchedulerAutotuneProfileConfig,
    scenario_overrides: Vec<SchedulerAutotuneScenarioOverride>,
) -> SchedulerAutotuneProfileSelection {
    SchedulerAutotuneProfileSelection {
        diagnose_only: true,
        model_name: "test-model".to_string(),
        hardware_label: "test-host".to_string(),
        selection_profile: SchedulerAutotuneSelectionProfile::AgentLongPrompt,
        objective: SchedulerAutotuneObjective::agent_default(),
        selected: Some(candidate_score(selected_config)),
        candidates: vec![candidate_score(selected_config)],
        scenario_overrides,
        rejected: Vec::new(),
        warnings: Vec::new(),
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
fn agent_long_prompt_profile_prefers_long_prompt_stability_over_short_concurrency_gain() {
    let long_stable = config(1, 2048, 5);
    let short_concurrency_winner = config(2, 2048, 5);
    let calibration = input(vec![
        measurement(long_stable, 1024, 128, 1, 345.7, 10.74, 1.70, 94.7),
        measurement(long_stable, 4096, 128, 1, 1650.4, 12.39, 3.21, 81.6),
        measurement(long_stable, 1024, 128, 2, 2052.6, 10.92, 3.42, 80.9),
        measurement(long_stable, 4096, 128, 2, 5104.2, 12.72, 6.68, 46.8),
        measurement(
            short_concurrency_winner,
            1024,
            128,
            1,
            365.7,
            11.05,
            1.76,
            92.3,
        ),
        measurement(
            short_concurrency_winner,
            4096,
            128,
            1,
            2036.2,
            13.03,
            3.67,
            78.3,
        ),
        measurement(
            short_concurrency_winner,
            1024,
            128,
            2,
            815.9,
            15.76,
            2.80,
            93.8,
        ),
        measurement(
            short_concurrency_winner,
            4096,
            128,
            2,
            5770.1,
            12.97,
            7.38,
            42.6,
        ),
    ]);

    let balanced = select_scheduler_autotune_profile_with_options(
        calibration.clone(),
        SchedulerAutotuneSelectionOptions {
            profile: SchedulerAutotuneSelectionProfile::Balanced,
        },
    );
    let agent_long_prompt = select_scheduler_autotune_profile_with_options(
        calibration,
        SchedulerAutotuneSelectionOptions {
            profile: SchedulerAutotuneSelectionProfile::AgentLongPrompt,
        },
    );

    assert_eq!(
        balanced.selected.expect("balanced selected").config,
        short_concurrency_winner
    );
    assert_eq!(
        agent_long_prompt
            .selected
            .as_ref()
            .expect("agent-long-prompt selected")
            .config,
        long_stable
    );
    assert_eq!(
        agent_long_prompt.selection_profile,
        SchedulerAutotuneSelectionProfile::AgentLongPrompt
    );
    assert!(agent_long_prompt
        .render_text()
        .contains("selection_profile: agent-long-prompt"));
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

    assert_eq!(profile.schema_version, SCHEDULER_AUTOTUNE_SCHEMA_VERSION);
    assert_eq!(profile.model_name, "test-model");
    assert_eq!(profile.hardware_label, "test-host");
    assert_eq!(profile.config, selected_config);
    assert_eq!(profile.config.decode_cadence_mid_chunk_cap, 256);
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

#[test]
fn runtime_profile_exports_and_applies_long_tg_concurrent_scenario_override() {
    let global = config_with_cadence(1, 1024, 5, 128);
    let pressure_point = config_with_cadence(1, 2048, 5, 512);
    let calibration = input(vec![
        measurement(global, 4096, 128, 1, 100.0, 10.0, 2.0, 100.0),
        measurement(pressure_point, 4096, 128, 1, 140.0, 13.0, 2.8, 90.0),
        measurement(global, 4096, 128, 2, 120.0, 10.5, 2.4, 95.0),
        measurement(pressure_point, 4096, 128, 2, 160.0, 13.5, 3.1, 88.0),
        measurement(global, 8192, 128, 2, 220.0, 11.0, 4.0, 80.0),
        measurement(pressure_point, 8192, 128, 2, 260.0, 14.0, 4.8, 76.0),
        measurement(global, 8192, 512, 2, 320.0, 13.0, 9.0, 68.0),
        measurement(pressure_point, 8192, 512, 2, 290.0, 13.5, 8.2, 69.0),
    ]);

    let selection = select_scheduler_autotune_profile(calibration);

    assert_eq!(
        selection.selected.as_ref().expect("selected").config,
        global
    );
    assert_eq!(selection.scenario_overrides.len(), 1);
    let override_rule = &selection.scenario_overrides[0];
    assert_eq!(override_rule.scenario.prompt_len, 8192);
    assert_eq!(override_rule.scenario.max_new_tokens, 512);
    assert_eq!(override_rule.scenario.concurrency, 2);
    assert_eq!(override_rule.config, pressure_point);

    let profile =
        build_scheduler_autotune_runtime_profile(&selection).expect("expected runtime profile");

    assert_eq!(profile.config, global);
    assert_eq!(profile.rules.len(), 1);
    assert_eq!(profile.rules[0].config, pressure_point);
    assert_eq!(profile.rules[0].when.prompt_len_gte, 8192);
    assert_eq!(profile.rules[0].when.max_new_tokens_gte, 512);
    assert_eq!(profile.rules[0].when.effective_concurrency_gte, 2);

    let selected = profile.select_config(SchedulerAutotuneRuntimeRequest {
        prompt_len: 8192,
        max_new_tokens: 512,
        effective_concurrency: 2,
    });
    assert_eq!(selected, pressure_point);

    let fallback = profile.select_config(SchedulerAutotuneRuntimeRequest {
        prompt_len: 8192,
        max_new_tokens: 128,
        effective_concurrency: 2,
    });
    assert_eq!(fallback, global);
}

#[test]
fn runtime_profile_compresses_equivalent_tg_rules() {
    let global = config_with_cadence(1, 1024, 5, 128);
    let pp4096_concurrent = config_with_cadence(1, 2048, 5, 128);
    let pp8192_concurrent = config_with_cadence(1, 2048, 5, 256);
    let selection = selection_with_overrides(
        global,
        vec![
            scenario_override(4096, 128, 2, pp4096_concurrent),
            scenario_override(4096, 512, 2, pp4096_concurrent),
            scenario_override(8192, 128, 2, pp8192_concurrent),
            scenario_override(8192, 512, 2, pp8192_concurrent),
        ],
    );

    let profile =
        build_scheduler_autotune_runtime_profile(&selection).expect("expected runtime profile");

    assert_eq!(profile.rules.len(), 2);
    assert_eq!(profile.rules[0].when.prompt_len_gte, 8192);
    assert_eq!(profile.rules[0].when.max_new_tokens_gte, 128);
    assert_eq!(profile.rules[0].config, pp8192_concurrent);
    assert_eq!(profile.rules[1].when.prompt_len_gte, 4096);
    assert_eq!(profile.rules[1].when.max_new_tokens_gte, 128);
    assert_eq!(profile.rules[1].config, pp4096_concurrent);

    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 128,
            effective_concurrency: 2,
        }),
        pp4096_concurrent
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 512,
            effective_concurrency: 2,
        }),
        pp4096_concurrent
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 8192,
            max_new_tokens: 128,
            effective_concurrency: 2,
        }),
        pp8192_concurrent
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 8192,
            max_new_tokens: 512,
            effective_concurrency: 2,
        }),
        pp8192_concurrent
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 512,
            effective_concurrency: 1,
        }),
        global
    );
}

#[test]
fn runtime_profile_keeps_specific_rule_when_intermediate_rule_changes_selection() {
    let global = config_with_cadence(1, 1024, 5, 128);
    let low_and_high = config_with_cadence(1, 2048, 5, 128);
    let intermediate = config_with_cadence(1, 2048, 5, 256);
    let selection = selection_with_overrides(
        global,
        vec![
            scenario_override(4096, 128, 2, low_and_high),
            scenario_override(4096, 256, 2, intermediate),
            scenario_override(4096, 512, 2, low_and_high),
        ],
    );

    let profile =
        build_scheduler_autotune_runtime_profile(&selection).expect("expected runtime profile");

    assert_eq!(profile.rules.len(), 3);
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 128,
            effective_concurrency: 2,
        }),
        low_and_high
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 256,
            effective_concurrency: 2,
        }),
        intermediate
    );
    assert_eq!(
        profile.select_config(SchedulerAutotuneRuntimeRequest {
            prompt_len: 4096,
            max_new_tokens: 512,
            effective_concurrency: 2,
        }),
        low_and_high
    );
}
