//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::core::{server, Loader, Tokenizer};
use crate::models::Qwen35Model;
use crate::Result;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    /// HF repo-id resolution is deferred to a future phase; pass a local path for now.
    #[arg(long)]
    pub model: String,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            args.model
        ));
    }

    let loader = Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
    let model_id = args.model.clone();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio::Runtime::new")?;
    runtime.block_on(server::serve(
        model, tokenizer, model_id, &args.host, args.port,
    ))
}
