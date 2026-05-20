//! ironmlx CLI entry point.

use clap::Parser;
use ironmlx::cli::Cli;

fn main() -> ironmlx::Result<()> {
    // Route diagnostic logs to stderr (Unix convention; stdout reserved for
    // OpenAI/Anthropic SSE response bodies and similar data streams).
    // P5g T0 harness depends on this: its stderr drainer captures
    // `[p5g-profile]` records emitted by GatedDeltaNet::forward_on; if tracing
    // wrote to stdout, the harness would silently produce empty Phase B/C
    // records arrays (latent bug found 2026-05-20).
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
