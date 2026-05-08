//! Command-line interface.
//!
//! Subcommands are dispatched here. Each subcommand lives in its own
//! file under `src/cli/`.

mod generate;
mod info;
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
    /// Boot an OpenAI/Anthropic-compatible HTTP server (single-stream).
    Serve(serve::ServeArgs),
}

impl Cli {
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Info(args) => info::run(args),
            Command::Generate(args) => generate::run(args),
            Command::Serve(args) => serve::run(args),
        }
    }
}
