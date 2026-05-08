//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::{Loader, Message, Tokenizer};
use crate::models::Qwen35Model;
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
    let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;

    let prompt = if args.chat && tokenizer.has_chat_template() {
        let messages = vec![Message {
            role: "user".into(),
            content: args.prompt.clone(),
        }];
        tokenizer.apply_chat_template(&messages, true)?
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
    };

    let mut stream = GenerationStream::new(&model, &tokenizer, request)?;
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
