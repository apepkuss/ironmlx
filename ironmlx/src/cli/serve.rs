//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use anyhow::Context;
use clap::Args;

use crate::core::scheduler::DenseVlMethods;
use crate::core::{server, Loader, Model, Tokenizer};
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
    /// Requests beyond this limit go to the admission queue. Default `1`
    /// optimizes single-request prefill / decode by avoiding [B,T_max]-padded
    /// MoE compute when only one slot is occupied; pass `--b-max N > 1` to
    /// enable concurrent multi-request batching. `0` rejected at startup
    /// because Scheduler with zero slots cannot admit any request.
    #[arg(long, default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
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

/// Generic serve helper — shared by all model types that satisfy the
/// `SchedulerActor<M>` / `AppState<M>` bounds.
///
/// The `DenseVlMethods` bound is required by `server::serve<M>` /
/// `SchedulerActor<M>`. For text-only models (e.g. `Qwen35MoeModel`) a
/// panic-on-call stub impl satisfies the bound at compile time; those code
/// paths are never reachable because VL endpoints are dense-only (P5c §3.10).
fn serve_with_model<M>(model: M, tokenizer: Tokenizer, args: &ServeArgs) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    #[cfg(feature = "p5h-profile")]
    {
        assert_eq!(
            args.b_max, 1,
            "p5h-profile feature requires --b-max 1 (single-active-row invariant per § 2.5a). \
             Got --b-max {}. Rebuild without --features p5h-profile to use multi-row batching.",
            args.b_max,
        );
    }

    // Surface b_max at boot so operators can confirm whether single-request
    // optimized mode (default) or multi-request batching is active without
    // having to inspect process args.
    if args.b_max == 1 {
        tracing::info!(
            "ironmlx serve: b_max=1 (single-request optimized mode; \
             pass --b-max N > 1 to enable concurrent multi-request batching)"
        );
    } else {
        tracing::info!(
            "ironmlx serve: b_max={} (multi-request batching enabled; \
             pass --b-max 1 to switch to single-request optimized mode)",
            args.b_max,
        );
    }

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
        args.max_cache_cap,
    ))
}

pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = std::path::PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            args.model
        ));
    }

    // open_multimodal so VL checkpoints retain vision_tower.* keys; for
    // text-only checkpoints the loader simply finds no vision keys.
    let loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;

    let model_type = loader
        .config_raw_value()
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?
        .to_owned();

    match model_type.as_str() {
        "qwen3_5" => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            serve_with_model(model, tokenizer, &args)
        }
        "qwen3_5_moe" => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            serve_with_model(model, tokenizer, &args)
        }
        other => Err(anyhow::anyhow!("unsupported model_type: {other}")),
    }
}
