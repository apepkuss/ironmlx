//! `ironmlx info` — print runtime + model info.

use clap::Args;

use crate::Result;

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Model directory or HuggingFace repo id.
    #[arg(long)]
    pub model: Option<String>,
}

pub fn run(_args: InfoArgs) -> Result<()> {
    println!("ironmlx — backed by Apple MLX");
    println!("MLX device: {:?}", mlx::default_device());
    Ok(())
}
