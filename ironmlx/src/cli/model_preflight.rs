use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::Result;
use ironmlx_lm::core::loader::preflight_model_metadata;

#[derive(Args, Debug)]
pub struct ModelPreflightArgs {
    /// Directory containing the immutable commit's metadata files.
    #[arg(long)]
    metadata_dir: PathBuf,
}

pub fn run(args: ModelPreflightArgs) -> Result<()> {
    let result = preflight_model_metadata(&args.metadata_dir)?;
    let json = serde_json::to_string(&result).context("serializing model preflight result")?;
    println!("{json}");
    Ok(())
}
