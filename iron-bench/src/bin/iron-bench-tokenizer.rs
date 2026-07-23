//! Persistent tokenizer sidecar for benchmark orchestration.
//!
//! The process loads one tokenizer and accepts newline-delimited JSON requests
//! on stdin. Keeping it alive avoids repeatedly parsing a large tokenizer.json
//! while a corpus is calibrated to several context lengths.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "iron-bench-tokenizer")]
struct Args {
    /// Model directory containing tokenizer.json.
    #[arg(long)]
    model_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    text: String,
    #[serde(default)]
    include_ids: bool,
}

#[derive(Debug, Serialize)]
struct TokenizeResponse<'a> {
    token_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids: Option<&'a [u32]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load tokenizer at {}: {error}",
            tokenizer_path.display()
        )
    })?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.context("reading tokenizer request")?;
        if line.trim().is_empty() {
            continue;
        }
        let request: TokenizeRequest =
            serde_json::from_str(&line).context("parsing tokenizer request JSON")?;
        let encoding = tokenizer
            .encode(request.text, false)
            .map_err(|error| anyhow::anyhow!("tokenizer.encode: {error}"))?;
        let ids = encoding.get_ids();
        let response = TokenizeResponse {
            token_count: ids.len(),
            token_ids: request.include_ids.then_some(ids),
        };
        serde_json::to_writer(&mut stdout, &response).context("writing tokenizer response")?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
