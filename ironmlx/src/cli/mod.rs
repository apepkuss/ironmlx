//! Command-line interface.
//!
//! Subcommands are dispatched here. Each subcommand lives in its own
//! file under `src/cli/`.

mod generate;
mod info;
mod kv_quant;
mod mtp_draft_cap;
mod scheduler_autotune;
mod scheduler_autotune_calibrate;
mod scheduler_profile_context;
pub(crate) mod scheduler_profile_store;
pub(crate) mod serve;

pub(crate) use kv_quant::KvQuantArg;

use clap::{Parser, Subcommand};

use crate::Result;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx",
    about = "Local LLM inference on Apple Silicon",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print discovered model and runtime info, then exit.
    Info(info::InfoArgs),
    /// Generate text from a prompt (prefill + decode).
    Generate(generate::GenerateArgs),
    /// Select a scheduler/autotune profile from offline calibration results.
    SchedulerAutotune(scheduler_autotune::SchedulerAutotuneArgs),
    /// Recommend Gemma4 drafter caps from offline benchmark observations.
    MtpDraftCap(mtp_draft_cap::MtpDraftCapArgs),
    /// Boot an OpenAI/Anthropic-compatible HTTP server (single-stream).
    Serve(Box<serve::ServeArgs>),
}

impl Cli {
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Info(args) => info::run(args),
            Command::Generate(args) => generate::run(args),
            Command::SchedulerAutotune(args) => scheduler_autotune::run(args),
            Command::MtpDraftCap(args) => mtp_draft_cap::run(args),
            Command::Serve(args) => serve::run(*args),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, KvQuantArg};

    #[test]
    fn mtp_draft_cap_subcommand_parses_inputs_and_threshold() {
        let cli = Cli::parse_from([
            "ironmlx",
            "mtp-draft-cap",
            "--input",
            "cap1.json",
            "--input",
            "cap2.json",
            "--min-windows",
            "64",
            "--min-records",
            "5",
            "--min-improvement-percent",
            "4.5",
            "--output",
            "recommendation.json",
        ]);

        match cli.command {
            Command::MtpDraftCap(args) => {
                assert_eq!(args.input.len(), 2);
                assert_eq!(args.min_windows, 64);
                assert_eq!(args.min_records, 5);
                assert_eq!(args.min_improvement_percent, 4.5);
                assert_eq!(
                    args.output.expect("output").to_string_lossy(),
                    "recommendation.json"
                );
            }
            other => panic!("expected MtpDraftCap command, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_autotune_subcommand_parses_input_and_json_format() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "--input",
            "calibration.json",
            "--format",
            "json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => {
                assert_eq!(
                    args.input.expect("expected legacy input").to_string_lossy(),
                    "calibration.json"
                );
                assert_eq!(
                    args.format,
                    super::scheduler_autotune::SchedulerAutotuneOutputFormat::Json
                );
                assert_eq!(
                    args.selection_profile,
                    super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg::AgentLongPrompt
                );
            }
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_autotune_select_subcommand_parses_write_profile() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "select",
            "--input",
            "calibration.json",
            "--format",
            "json",
            "--selection-profile",
            "balanced",
            "--write-profile",
            "scheduler-profile.json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Select(select)) => {
                    assert_eq!(select.input.to_string_lossy(), "calibration.json");
                    assert_eq!(
                        select.format,
                        super::scheduler_autotune::SchedulerAutotuneOutputFormat::Json
                    );
                    assert_eq!(
                        select.selection_profile,
                        super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg::Balanced
                    );
                    assert_eq!(
                        select
                            .write_profile
                            .as_ref()
                            .expect("expected write profile")
                            .to_string_lossy(),
                        "scheduler-profile.json"
                    );
                }
                other => panic!("expected Select action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_autotune_merge_subcommand_parses_inputs_and_output() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "merge",
            "--input",
            "candidate-a.json",
            "--input",
            "candidate-b.json",
            "--output",
            "calibration.json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Merge(merge)) => {
                    assert_eq!(merge.input.len(), 2);
                    assert_eq!(merge.input[0].to_string_lossy(), "candidate-a.json");
                    assert_eq!(merge.input[1].to_string_lossy(), "candidate-b.json");
                    assert_eq!(
                        merge
                            .output
                            .as_ref()
                            .expect("expected output")
                            .to_string_lossy(),
                        "calibration.json"
                    );
                    assert!(!merge.allow_incomplete_coverage);
                }
                other => panic!("expected Merge action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

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
            "--iron-bench-bin",
            "target/release/iron-bench",
            "--output-dir",
            "/tmp/autotune",
            "--candidate",
            "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768,decode_cadence_mid_chunk_cap=256",
            "--prompt-len",
            "1024,2048",
            "--max-tokens",
            "128",
            "--concurrency",
            "1,2",
            "--selection-profile",
            "balanced",
            "--write-profile",
            "/tmp/scheduler-profile.json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Calibrate(calibrate)) => {
                    assert_eq!(calibrate.model.to_string_lossy(), "/tmp/model");
                    assert_eq!(
                        calibrate.model_name.as_ref().expect("expected model name"),
                        "GLM-4.7-flash-4bit"
                    );
                    assert_eq!(
                        calibrate
                            .iron_bench_bin
                            .as_ref()
                            .expect("expected iron-bench bin")
                            .to_string_lossy(),
                        "target/release/iron-bench"
                    );
                    assert_eq!(
                        calibrate
                            .output_dir
                            .as_ref()
                            .expect("expected output dir")
                            .to_string_lossy(),
                        "/tmp/autotune"
                    );
                    assert_eq!(calibrate.candidates.len(), 1);
                    assert_eq!(calibrate.candidates[0].b_max, 2);
                    assert_eq!(calibrate.candidates[0].prefill_chunk_size, 1024);
                    assert_eq!(calibrate.candidates[0].decode_cadence_mid_chunk_cap, 256);
                    assert_eq!(calibrate.prompt_len, vec![1024, 2048]);
                    assert_eq!(calibrate.max_tokens, 128);
                    assert_eq!(calibrate.concurrency, vec![1, 2]);
                    assert_eq!(
                        calibrate.selection_profile,
                        super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg::Balanced
                    );
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

    #[test]
    fn scheduler_autotune_calibrate_subcommand_accepts_full_auto_defaults() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "calibrate",
            "--model",
            "/tmp/GLM-4.7-flash-4bit",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Calibrate(calibrate)) => {
                    assert_eq!(calibrate.model.to_string_lossy(), "/tmp/GLM-4.7-flash-4bit");
                    assert!(calibrate.model_name.is_none());
                    assert!(calibrate.iron_bench_bin.is_none());
                    assert!(calibrate.output_dir.is_none());
                    assert!(calibrate.candidates.is_empty());
                    assert!(calibrate.prompt_len.is_empty());
                    assert!(calibrate.concurrency.is_empty());
                    assert_eq!(
                        calibrate.selection_profile,
                        super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg::AgentLongPrompt
                    );
                    assert!(calibrate.write_profile.is_none());
                }
                other => panic!("expected Calibrate action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_autotune_profile_subcommands_parse_profile_id() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "profile",
            "show",
            "test-profile-id",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Profile(profile)) => {
                    match profile.action {
                        super::scheduler_autotune::SchedulerAutotuneProfileAction::Show(show) => {
                            assert_eq!(show.id, "test-profile-id");
                        }
                        other => panic!("expected profile show action, got {other:?}"),
                    }
                }
                other => panic!("expected Profile action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }

        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "profile",
            "import",
            "--model",
            "/tmp/model",
            "--profile",
            "/tmp/scheduler-profile.json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Profile(profile)) => {
                    match profile.action {
                        super::scheduler_autotune::SchedulerAutotuneProfileAction::Import(
                            import,
                        ) => {
                            assert_eq!(import.model.to_string_lossy(), "/tmp/model");
                            assert_eq!(
                                import.profile.to_string_lossy(),
                                "/tmp/scheduler-profile.json"
                            );
                        }
                        other => panic!("expected profile import action, got {other:?}"),
                    }
                }
                other => panic!("expected Profile action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }

        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "profile",
            "remove",
            "test-profile-id",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Profile(profile)) => {
                    match profile.action {
                        super::scheduler_autotune::SchedulerAutotuneProfileAction::Remove(
                            remove,
                        ) => {
                            assert_eq!(remove.id, "test-profile-id");
                        }
                        other => panic!("expected profile remove action, got {other:?}"),
                    }
                }
                other => panic!("expected Profile action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_autotune_profile_doctor_subcommand_parses_model() {
        let cli = Cli::parse_from([
            "ironmlx",
            "scheduler-autotune",
            "profile",
            "doctor",
            "--model",
            "/tmp/model",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Profile(profile)) => {
                    match profile.action {
                        super::scheduler_autotune::SchedulerAutotuneProfileAction::Doctor(
                            doctor,
                        ) => {
                            assert_eq!(doctor.model.to_string_lossy(), "/tmp/model");
                            assert_eq!(doctor.max_age_days, 30);
                        }
                        other => panic!("expected profile doctor action, got {other:?}"),
                    }
                }
                other => panic!("expected Profile action, got {other:?}"),
            },
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_scheduler_profile() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--scheduler-profile",
            "scheduler-profile.json",
            "--kv-quant",
            "turbo4",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.model.as_deref(), Some("/tmp/model"));
                assert_eq!(
                    args.scheduler_profile
                        .as_ref()
                        .expect("expected scheduler profile")
                        .to_string_lossy(),
                    "scheduler-profile.json"
                );
                assert_eq!(args.kv_quant, KvQuantArg::Turbo4);
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_k3v4_kv_quant() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--kv-quant",
            "k3v4",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.kv_quant, KvQuantArg::K3V4);
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_paged_prefix_cache() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--paged-prefix-cache-block-size",
            "32",
            "--paged-prefix-cache-max-pages",
            "4096",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(
                    args.paged_prefix_cache_dir
                        .as_ref()
                        .expect("prefix cache dir")
                        .to_string_lossy(),
                    "/tmp/prefix-cache"
                );
                assert_eq!(args.paged_prefix_cache_block_size, 32);
                assert_eq!(args.paged_prefix_cache_max_pages, Some(4096));
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_ssd_prefix_cache_max_gb() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--ssd-prefix-cache-max-gb",
            "10",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.ssd_prefix_cache_max_gb, Some(10));
                assert_eq!(args.paged_prefix_cache_max_pages, None);
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_active_kv_offload() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--active-kv-offload",
            "--active-kv-offload-dir",
            "/tmp/active-kv",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert!(args.active_kv_offload);
                assert_eq!(
                    args.active_kv_offload_dir
                        .as_ref()
                        .expect("active kv offload dir")
                        .to_string_lossy(),
                    "/tmp/active-kv"
                );
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_parses_memory_limits() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--memory-limit-total-gb",
            "64",
            "--memory-limit-model-gb",
            "40",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.memory_limit_total_gb, Some(64));
                assert_eq!(args.memory_limit_model_gb, Some(40));
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_accepts_default_paged_prefix_cache_dir() {
        let cli = Cli::try_parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--paged-prefix-cache-dir",
        ])
        .expect("parse serve with default prefix cache dir");

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(
                    args.paged_prefix_cache_dir
                        .as_ref()
                        .expect("prefix cache dir")
                        .to_string_lossy(),
                    "~/.ironmlx/cache/paged_prefix_cache"
                );
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_subcommand_allows_app_daemon_without_model() {
        let cli = Cli::parse_from(["ironmlx", "serve", "--port", "9068"]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.model, None);
                assert_eq!(args.port, 9068);
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }

    #[test]
    fn generate_subcommand_parses_k3v4_kv_quant() {
        let cli = Cli::parse_from([
            "ironmlx",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--kv-quant",
            "k3v4",
        ]);

        match cli.command {
            Command::Generate(args) => {
                assert_eq!(args.kv_quant, KvQuantArg::K3V4);
            }
            other => panic!("expected Generate command, got {other:?}"),
        }
    }
}
