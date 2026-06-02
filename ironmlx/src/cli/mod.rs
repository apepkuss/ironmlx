//! Command-line interface.
//!
//! Subcommands are dispatched here. Each subcommand lives in its own
//! file under `src/cli/`.

mod generate;
mod info;
mod scheduler_autotune;
mod scheduler_autotune_calibrate;
mod serve;

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
    /// Boot an OpenAI/Anthropic-compatible HTTP server (single-stream).
    Serve(serve::ServeArgs),
}

impl Cli {
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Info(args) => info::run(args),
            Command::Generate(args) => generate::run(args),
            Command::SchedulerAutotune(args) => scheduler_autotune::run(args),
            Command::Serve(args) => serve::run(args),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

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
            "--write-profile",
            "/tmp/scheduler-profile.json",
        ]);

        match cli.command {
            Command::SchedulerAutotune(args) => match args.action {
                Some(super::scheduler_autotune::SchedulerAutotuneAction::Calibrate(calibrate)) => {
                    assert_eq!(calibrate.model.to_string_lossy(), "/tmp/model");
                    assert_eq!(calibrate.model_name, "GLM-4.7-flash-4bit");
                    assert_eq!(
                        calibrate.iron_bench_bin.to_string_lossy(),
                        "target/release/iron-bench"
                    );
                    assert_eq!(calibrate.output_dir.to_string_lossy(), "/tmp/autotune");
                    assert_eq!(calibrate.candidates.len(), 1);
                    assert_eq!(calibrate.candidates[0].b_max, 2);
                    assert_eq!(calibrate.candidates[0].prefill_chunk_size, 1024);
                    assert_eq!(calibrate.candidates[0].decode_cadence_mid_chunk_cap, 256);
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

    #[test]
    fn serve_subcommand_parses_scheduler_profile() {
        let cli = Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--scheduler-profile",
            "scheduler-profile.json",
        ]);

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.model, "/tmp/model");
                assert_eq!(
                    args.scheduler_profile
                        .as_ref()
                        .expect("expected scheduler profile")
                        .to_string_lossy(),
                    "scheduler-profile.json"
                );
            }
            other => panic!("expected Serve command, got {other:?}"),
        }
    }
}
