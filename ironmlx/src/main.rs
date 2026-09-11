//! ironmlx CLI entry point.

use clap::Parser;
use ironmlx::cli::Cli;

fn main() -> ironmlx::Result<()> {
    ironmlx::logging::init();

    let cli = Cli::parse();
    cli.run()
}
