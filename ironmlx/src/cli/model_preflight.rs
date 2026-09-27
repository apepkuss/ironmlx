use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::Result;

#[derive(Args, Debug)]
pub struct ModelPreflightArgs {
    /// Directory containing the immutable commit's metadata files.
    #[arg(long)]
    metadata_dir: PathBuf,
}

pub fn run(args: ModelPreflightArgs) -> Result<()> {
    let json = if args.metadata_dir.join("model_index.json").exists() {
        serde_json::to_string(&ironmlx_image::preflight_model_metadata(
            &args.metadata_dir,
        )?)
    } else {
        serde_json::to_string(&ironmlx_lm::core::preflight_model_metadata(
            &args.metadata_dir,
        )?)
    }
    .context("serializing model preflight result")?;
    println!("{json}");
    Ok(())
}
