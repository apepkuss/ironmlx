use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use ironmlx_decision::{DecisionRequest, Laya};

use crate::Result;

#[derive(Args, Debug)]
pub struct DecideArgs {
    /// Directory containing the pinned Laya multilingual MLX checkpoint.
    #[arg(long)]
    model_dir: PathBuf,
    /// JSON file containing model, state, and typed questions.
    #[arg(long)]
    request: PathBuf,
}

pub fn run(args: DecideArgs) -> Result<()> {
    let request: DecisionRequest = serde_json::from_slice(&std::fs::read(&args.request)?)
        .context("parsing decision request")?;
    let laya = Laya::load(&args.model_dir).context("loading Laya multilingual model")?;
    let response = laya.predict(&request)?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
