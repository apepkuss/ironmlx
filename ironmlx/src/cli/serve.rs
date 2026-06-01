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

    /// P5h+1 T1 measurement probe: force selected span bodies (Lane A
    /// `first_token_sampling_materialize_and_sample` + the ROI substep
    /// closures under GatedAttention / GatedDeltaNet / SparseMoeBlock +
    /// `slice_last_and_project_lm_head` + `cache_state_update`) to call
    /// `mlx::transforms::eval` on returned `Array` value(s) before the
    /// span closes. Measurement-only: defaults OFF so production lazy-graph
    /// semantics are preserved. Use ONLY for P5h+1 attribution sweeps.
    #[cfg(feature = "p5h-profile")]
    #[arg(long, default_value_t = false)]
    pub p5h_measurement_eval_probes: bool,
}

/// Generic serve helper — shared by all model types that satisfy the
/// `SchedulerActor<M>` / `AppState<M>` bounds.
///
/// The `DenseVlMethods` bound is required by `server::serve<M>` /
/// `SchedulerActor<M>`. The trait name is historical; both dense and MoE
/// Qwen3.5 variants implement it so the same OpenAI VL route can serve either
/// checkpoint family.
fn serve_with_model<M>(
    model: M,
    tokenizer: Tokenizer,
    args: &ServeArgs,
    vision_input: Option<server::VisionInputConfig>,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    #[cfg(feature = "p5h-profile")]
    {
        assert!(
            args.b_max == 1 || crate::core::p5h::scheduler_decode_allow_multi_row(),
            "p5h-profile feature requires --b-max 1 for legacy request-root attribution \
             (single-active-row invariant per § 2.5a). Got --b-max {}. \
             Set IRONMLX_P5H_SCHEDULER_DECODE_ALLOW_MULTI_ROW=1 only for experimental \
             unary scheduler decode attribution, or rebuild without --features p5h-profile \
             to use ordinary multi-row batching.",
            args.b_max,
        );
        if args.b_max != 1 {
            tracing::warn!(
                "p5h-profile multi-row escape hatch enabled for scheduler decode attribution; \
                 use unary/non-streaming clients so legacy streaming request-root p5h trees do \
                 not enter their single-active-row prefill path"
            );
        }
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
    // P5h+1 T1: derive measurement-eval-probes flag (feature-gated CLI arg);
    // feature-off builds always pass `false` so the receiver-side `set_*`
    // call site can remain unconditional in signature.
    #[cfg(feature = "p5h-profile")]
    let p5h_measurement_eval_probes = args.p5h_measurement_eval_probes;
    #[cfg(not(feature = "p5h-profile"))]
    let p5h_measurement_eval_probes = false;
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
        p5h_measurement_eval_probes,
        vision_input,
    ))
}

fn read_model_type(model_dir: &std::path::Path) -> Result<String> {
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))
}

pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = std::path::PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            args.model
        ));
    }

    let model_type = read_model_type(&model_dir)?;
    let architecture = crate::models::ModelArchitecture::from_model_type(&model_type)?;
    // open_multimodal so Qwen VL checkpoints retain vision_tower.* keys.
    let loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let vision_input = if architecture == crate::models::ModelArchitecture::Gemma4 {
        let cfg = crate::models::gemma4::Gemma4Config::from_loader(&loader)
            .context("Gemma4Config::from_loader")?;
        cfg.vision_config
            .map(|vision_config| server::VisionInputConfig::Gemma4 { vision_config })
    } else {
        None
    };

    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            serve_with_model(model, tokenizer, &args, vision_input)
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, vision_input)
        }
        crate::models::ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            serve_with_model(model, tokenizer, &args, vision_input)
        }
        crate::models::ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, None)
        }
        crate::models::ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, None)
        }
    }
}

#[cfg(test)]
#[cfg(feature = "p5h-profile")]
mod tests {
    #[test]
    fn p5h_scheduler_decode_multi_row_escape_hatch_is_owned_by_p5h_config() {
        assert!(crate::core::p5h::scheduler_decode_allow_multi_row_from_env_value(Some("1")));
    }
}
