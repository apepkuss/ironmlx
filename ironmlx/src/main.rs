//! ironmlx CLI entry point.

use clap::Parser;
use ironmlx::cli::Cli;

fn main() -> ironmlx::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("ironmlx=info,warn")),
        )
        .init();

    let cli = Cli::parse();
    cli.run()
}
