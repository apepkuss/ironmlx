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

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head.
    #[arg(long, default_value_t = 2048)]
    pub prefill_chunk_size: usize,

    /// Maximum concurrent in-flight requests (Scheduler slot count).
    /// Requests beyond this limit go to the admission queue.
    #[arg(long, default_value_t = 4)]
    pub b_max: usize,

    /// Admission-window deadline in milliseconds. After the first
    /// admit in a batch arrives, additional admits are absorbed until
    /// this deadline expires or the batch saturates at b_max.
    #[arg(long, default_value_t = 5)]
    pub admission_deadline_ms: u64,

    /// Capacity of the FIFO admission queue. Requests received while
    /// the scheduler is saturated are parked here. `0` disables queueing
    /// (immediate Err on saturation — mirrors pre-3d behavior).
    #[arg(long, default_value_t = 32)]
    pub admission_queue_max: usize,

    /// Maximum allowed `prompt_len + max_new_tokens` per request. Capped
    /// further at the model's `max_position_embeddings` (Qwen3.5-4B: 262144).
    /// Requests beyond this return HTTP 413 Payload Too Large. B1-p2.3f.
    #[arg(long, default_value_t = 32768)]
    pub max_cache_cap: usize,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            args.model
        ));
    }

    // open_multimodal so VL checkpoints retain vision_tower.* keys; for
    // text-only checkpoints the loader simply finds no vision keys and
    // Qwen35Model::from_loader sets vision = None.
    let loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
    let model_id = args.model.clone();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio::Runtime::new")?;
    runtime.block_on(server::serve(
        model,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        args.prefill_chunk_size,
        args.b_max,
        args.admission_deadline_ms,
        args.admission_queue_max,
        args.max_cache_cap, // 3f
    ))
}
