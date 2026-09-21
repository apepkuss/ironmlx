//! ironmlx CLI entry point.

use clap::Parser;
use ironmlx::cli::Cli;

fn main() -> ironmlx::Result<()> {
    // Process-wide precision must be chosen before MLX initializes and before
    // worker threads start. Audio loading rejects an explicit incompatible value.
    if std::env::var_os("MLX_ENABLE_TF32").is_none() {
        std::env::set_var("MLX_ENABLE_TF32", "0");
    }
    ironmlx::logging::init();

    let cli = Cli::parse();
    cli.run()
}
