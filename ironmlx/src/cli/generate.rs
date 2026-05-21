//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::core::generate::{GenerateRequest, GenerationStream, IMAGE_TOKEN_ID};
use crate::core::sampler::Sampler;
use crate::core::{Loader, Message, Model, Tokenizer};
use crate::Result;

#[derive(Args, Debug)]
pub struct GenerateArgs {
    #[arg(long)]
    pub model: String,

    #[arg(long)]
    pub prompt: String,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// If set, apply the chat template; otherwise tokenize the raw prompt.
    #[arg(long, default_value_t = true)]
    pub chat: bool,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head.
    #[arg(long, default_value_t = 2048)]
    pub prefill_chunk_size: usize,
}

fn run_generation_with_model<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    args: &GenerateArgs,
) -> Result<()> {
    let prompt = if args.chat && tokenizer.has_chat_template() {
        let messages = vec![Message {
            role: "user".into(),
            content: args.prompt.clone(),
        }];
        tokenizer.apply_chat_template(&messages, true, /* extra_kwargs = */ None)?
    } else {
        args.prompt.clone()
    };
    let prompt_ids = tokenizer.encode(&prompt, /* add_special_tokens = */ false)?;

    let mut sampler = Sampler::greedy();
    if args.temperature > 0.0 {
        sampler = sampler.with_temperature(args.temperature);
    }
    if args.top_p < 1.0 {
        sampler = sampler.with_top_p(args.top_p);
    }
    if args.seed != 0 {
        sampler = sampler.with_seed(args.seed);
    }

    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: args.max_tokens,
        sampler,
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: args.prefill_chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        // CLI is text-only; both values unused when image_grid_thw is None.
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    };

    let mut stream = GenerationStream::new_text_only(model, tokenizer, request)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while let Some(ev) = stream.next_token()? {
        if !ev.text.is_empty() {
            out.write_all(ev.text.as_bytes())?;
            out.flush()?;
        }
        if ev.finish_reason.is_some() {
            break;
        }
    }
    writeln!(out)?;
    Ok(())
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}')",
            args.model
        ));
    }
    let loader = Loader::open(&model_dir).context("Loader::open")?;
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
            run_generation_with_model(&model, &tokenizer, &args)
        }
        "qwen3_5_moe" => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &args)
        }
        other => Err(anyhow::anyhow!(
            "unsupported model_type: {other} (expected 'qwen3_5' or 'qwen3_5_moe')"
        )),
    }
}
