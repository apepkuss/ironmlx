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
                assert_eq!(args.input.to_string_lossy(), "calibration.json");
                assert_eq!(
                    args.format,
                    super::scheduler_autotune::SchedulerAutotuneOutputFormat::Json
                );
            }
            other => panic!("expected SchedulerAutotune command, got {other:?}"),
        }
    }
}
