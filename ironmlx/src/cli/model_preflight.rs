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
    let config: serde_json::Value = std::fs::read(args.metadata_dir.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let json = if config["model_type"] == "qwen3_asr" {
        serde_json::to_string(&ironmlx_audio::qwen3_asr::preflight_model_metadata(
            &args.metadata_dir,
        )?)
    } else if args.metadata_dir.join("model_index.json").exists() {
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
