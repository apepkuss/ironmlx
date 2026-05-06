//! `ironmlx generate` — generate text from a prompt.

use clap::Args;

use crate::Result;

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Model directory or HuggingFace repo id (e.g. `mlx-community/Qwen3.5-4B-MLX-4bit`).
    #[arg(long)]
    pub model: String,

    /// User prompt.
    #[arg(long)]
    pub prompt: String,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = greedy / argmax).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-p nucleus sampling (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    /// PRNG seed for reproducible sampling.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn run(_args: GenerateArgs) -> Result<()> {
    Err(anyhow::anyhow!(
        "generate not yet wired — model loading / forward / decode loop pending P3-P4"
    ))
}
