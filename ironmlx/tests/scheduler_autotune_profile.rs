use ironmlx::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, build_scheduler_autotune_runtime_profile_at,
    evaluate_scheduler_autotune_profile_health, select_scheduler_autotune_profile,
    select_scheduler_autotune_profile_with_options, SchedulerAutotuneCacheState,
    SchedulerAutotuneCalibrationInput, SchedulerAutotuneCandidateScore,
    SchedulerAutotuneMeasurement, SchedulerAutotuneObjective, SchedulerAutotuneProfileConfig,
    SchedulerAutotuneProfileHealthInput, SchedulerAutotuneProfileHealthLevel,
    SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneProfileSelection,
    SchedulerAutotuneRuntimeContext, SchedulerAutotuneRuntimeHealth,
    SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeProfileMetadata,
    SchedulerAutotuneRuntimeRequest, SchedulerAutotuneScenario, SchedulerAutotuneScenarioOverride,
    SchedulerAutotuneSelectionOptions, SchedulerAutotuneSelectionProfile, SchedulerSpeculativeMode,
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
        cache_state: SchedulerAutotuneCacheState::Cold,
        ttft_ms_p95,
        itl_ms_p95,
        e2e_s_p95,
        tokens_per_sec,
        early_itl_ms_p95: itl_ms_p95,
        memory_budget_ok: true,
        cached_tokens_warning: false,
        runtime_health: healthy_runtime(),
    }
}

fn input(measurements: Vec<SchedulerAutotuneMeasurement>) -> SchedulerAutotuneCalibrationInput {
    SchedulerAutotuneCalibrationInput {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "test-model".to_string(),
        hardware_label: "test-host".to_string(),
        runtime_context: runtime_context(),
        objective: SchedulerAutotuneObjective::agent_default(),
        measurements,
    }
}

fn runtime_profile_with_metadata(
    created_at_unix_ms: u64,
    scenario_coverage: Vec<SchedulerAutotuneScenario>,
) -> SchedulerAutotuneRuntimeProfile {
    let mut metadata = SchedulerAutotuneRuntimeProfileMetadata::synthetic(created_at_unix_ms);
    metadata.scenario_coverage = scenario_coverage;
    metadata.candidate_count = 1;

    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "test-model".to_string(),
        hardware_label: "test-host".to_string(),
        runtime_context: runtime_context(),
        config: config(1, 2048, 5),
        rules: Vec::new(),
        metadata,
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
            cache_state: SchedulerAutotuneCacheState::Cold,
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
        runtime_context: runtime_context(),
        selection_profile: SchedulerAutotuneSelectionProfile::AgentLongPrompt,
        objective: SchedulerAutotuneObjective::agent_default(),
        scenarios: scenario_overrides
            .iter()
            .map(|item| item.scenario.clone())
            .collect(),
        selected: Some(candidate_score(selected_config)),
        candidates: vec![candidate_score(selected_config)],
        scenario_overrides,
        rejected: Vec::new(),
        warnings: Vec::new(),
    }
}

fn runtime_context() -> SchedulerAutotuneRuntimeContext {
    SchedulerAutotuneRuntimeContext::local_default(32768)
}

fn healthy_runtime() -> SchedulerAutotuneRuntimeHealth {
    SchedulerAutotuneRuntimeHealth {
        healthy: true,
        status: "healthy".to_string(),
        request_completion_ok: true,
        admission_queue_full_count_delta: 0,
        memory_budget_exceeded_count_delta: 0,
        active_kv_degraded: false,
        active_kv_swap_error_count_delta: 0,
        logical_kv_cap_tokens: 32768,
        resident_kv_cap_tokens: 32768,
        mtp: None,
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
fn profile_selection_rejects_unhealthy_runtime_measurement() {
    let selected_config = config(1, 2048, 5);
    let mut row = measurement(selected_config, 2048, 128, 1, 100.0, 12.0, 2.0, 80.0);
    row.runtime_health.healthy = false;
    row.runtime_health.status = "admission-queue-full".to_string();
    row.runtime_health.admission_queue_full_count_delta = 1;

    let selection = select_scheduler_autotune_profile(input(vec![row]));

    assert!(selection.selected.is_none());
    assert!(selection
        .rejected
        .iter()
        .any(|item| item.code == "runtime_health_unsafe"));
}

#[test]
fn profile_selection_rejects_inactive_speculative_context() {
    let selected_config = config(1, 2048, 5);
    let mut calibration = input(vec![measurement(
        selected_config,
        2048,
        128,
        1,
        100.0,
        12.0,
        2.0,
        80.0,
    )]);
    calibration.runtime_context.speculative.mode = SchedulerSpeculativeMode::QwenMtp;
    calibration
        .runtime_context
        .speculative
        .draft_model_fingerprint = Some("draft-model".to_string());
    calibration.runtime_context.speculative.draft_tokens = Some(3);

    let selection = select_scheduler_autotune_profile(calibration);

    assert!(selection.selected.is_none());
    assert!(selection
        .rejected
        .iter()
        .any(|item| item.code == "speculative_path_inactive"));
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
    assert_eq!(
        profile.metadata.selection_profile,
        SchedulerAutotuneSelectionProfile::AgentLongPrompt
    );
    assert_eq!(profile.metadata.scenario_coverage.len(), 2);
    assert_eq!(profile.metadata.candidate_count, 1);
    assert_eq!(profile.metadata.rejected_count, 0);
    assert!(profile.metadata.selected_score.is_finite());
}

#[test]
fn profile_health_reports_healthy_for_matching_fresh_agent_coverage() {
    let profile = runtime_profile_with_metadata(
        1811606400000,
        vec![
            SchedulerAutotuneScenario {
                prompt_len: 1024,
                max_new_tokens: 128,
                concurrency: 1,
                cache_state: SchedulerAutotuneCacheState::Cold,
            },
            SchedulerAutotuneScenario {
                prompt_len: 4096,
                max_new_tokens: 128,
                concurrency: 2,
                cache_state: SchedulerAutotuneCacheState::Cold,
            },
        ],
    );

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "test-model",
        expected_hardware_label: "test-host",
        expected_runtime_context: &profile.runtime_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000 + 1000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Healthy);
    assert!(report
        .notes
        .iter()
        .all(|note| note.level == SchedulerAutotuneProfileHealthLevel::Info));
}

#[test]
fn profile_health_warns_for_stale_version_and_missing_concurrency_coverage() {
    let mut profile = runtime_profile_with_metadata(
        1811606400000,
        vec![SchedulerAutotuneScenario {
            prompt_len: 1024,
            max_new_tokens: 128,
            concurrency: 1,
            cache_state: SchedulerAutotuneCacheState::Cold,
        }],
    );
    profile.metadata.ironmlx_version = "0.0.0-test".to_string();

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "other-model-name",
        expected_hardware_label: "test-host",
        expected_runtime_context: &profile.runtime_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000 + 31 * 24 * 60 * 60 * 1000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Warning);
    assert!(report.notes.iter().any(|note| note.code == "profile_stale"));
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "ironmlx_version_changed"));
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "model_name_mismatch"));
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "no_concurrent_coverage"));
}

#[test]
fn profile_health_invalidates_schema_and_hardware_mismatch() {
    let mut profile = runtime_profile_with_metadata(
        1811606400000,
        vec![SchedulerAutotuneScenario {
            prompt_len: 4096,
            max_new_tokens: 128,
            concurrency: 2,
            cache_state: SchedulerAutotuneCacheState::Cold,
        }],
    );
    profile.schema_version = SCHEDULER_AUTOTUNE_SCHEMA_VERSION + 1;
    profile.hardware_label = "other-host".to_string();

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "test-model",
        expected_hardware_label: "test-host",
        expected_runtime_context: &profile.runtime_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Invalid);
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "schema_version_mismatch"));
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "hardware_label_mismatch"));
}

#[test]
fn profile_health_invalidates_runtime_context_mismatch() {
    let profile = runtime_profile_with_metadata(
        1811606400000,
        vec![SchedulerAutotuneScenario {
            prompt_len: 4096,
            max_new_tokens: 128,
            concurrency: 2,
            cache_state: SchedulerAutotuneCacheState::Cold,
        }],
    );
    let mut expected_context = profile.runtime_context.clone();
    expected_context.logical_kv_cap_tokens += 1;

    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: "test-model",
        expected_hardware_label: "test-host",
        expected_runtime_context: &expected_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: 1811606400000,
        max_age_days: 30,
    });

    assert_eq!(report.status, SchedulerAutotuneProfileHealthStatus::Invalid);
    assert!(report
        .notes
        .iter()
        .any(|note| note.code == "runtime_context_mismatch"));
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
