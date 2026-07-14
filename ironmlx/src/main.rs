//! ironmlx CLI entry point.

use clap::Parser;
use ironmlx::cli::Cli;

fn main() -> ironmlx::Result<()> {
    // Route diagnostic logs to stderr (Unix convention; stdout reserved for
    // OpenAI/Anthropic SSE response bodies and similar data streams).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("ironmlx=info,warn")),
        )
        .init();

    let cli = Cli::parse();
    cli.run()
}
