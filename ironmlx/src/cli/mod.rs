//! Command-line interface.
//!
//! Subcommands are dispatched here. Each subcommand lives in its own
//! file under `src/cli/`.

mod generate;
mod info;
mod scheduler_autotune;
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
}
