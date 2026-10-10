//! Numeric-regression dump for the Qwen3.6 MoE target: full next-token
//! logits along fixed token sequences, so builds of different source
//! versions can be compared offline on identical inputs.
//!
//! - `generate`: chat-template each prompt, greedily extend it with
//!   ordinary decoding and write the token sequences (`--sequences`).
//! - `decode`: prefill each sequence up to `decode_from`, then feed the
//!   remaining tokens one at a time (ordinary decoding, teacher-forced).
//! - `prefill`: one prefill over the whole sequence.
//!
//! Both dump modes write, per sequence, the BF16 logits `[P, V]` predicting
//! tokens `decode_from..len` (`<id>.<mode>.bf16`) and a JSON index line.
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ironmlx_lm::core::chat_template::Message;
use ironmlx_lm::core::model_input::build_position_ids;
use ironmlx_lm::models::dflash2::{DFlash2Target, DFlash2TargetForwardMode as Mode};
use ironmlx_lm::models::Qwen35MoeModel;
use ironmlx_lm::{Loader, Model, Tokenizer};
use mlx::{Array, Dtype};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    target: PathBuf,
    /// `generate`, `decode` or `prefill`.
    #[arg(long)]
    mode: String,
    /// generate: JSON list of {"id", "prompt", "new_tokens"} (prompt file
    /// paths are read with `@path`). dump modes: unused.
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// generate: output; dump modes: input. JSON list of
    /// {"id", "tokens", "decode_from"}.
    #[arg(long)]
    sequences: PathBuf,
    /// Dump modes: directory for the logits files and `index.jsonl`.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long)]
    m5_profile: bool,
}

#[derive(serde::Deserialize)]
struct PromptCase {
    id: String,
    prompt: String,
    new_tokens: usize,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct Sequence {
    id: String,
    tokens: Vec<u32>,
    decode_from: usize,
}

fn ids(tokens: &[u32]) -> Result<Array> {
    Ok((tokens, &[1_i32, tokens.len() as i32][..]).try_into()?)
}

fn rows(hidden: &Array, start: i32, end: i32) -> Result<Array> {
    let dims = hidden.shape();
    Ok(mlx::ops::indexing::slice_strided(
        hidden,
        &[0_i32, start, 0][..],
        &[1_i32, end, dims.as_slice()[2]][..],
        &[1_i32, 1, 1][..],
    )?)
}

fn bf16_bytes(logits: &Array) -> Result<Vec<u8>> {
    let logits = mlx::ops::cast::astype(logits, Dtype::Float32)?;
    Ok(logits
        .to_vec::<f32>()?
        .into_iter()
        // BF16 logits are exact in F32, so the round-to-nearest-even is the identity.
        .map(|v| {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        })
        .flat_map(u16::to_le_bytes)
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ironmlx_core::m5_profile::install_for_target(
        if args.m5_profile {
            ironmlx_core::m5_profile::M5ProfileMode::Auto
        } else {
            ironmlx_core::m5_profile::M5ProfileMode::Off
        },
        true,
        ironmlx_core::m5_profile::M5ProfileTarget::Qwen36Moe,
    );
    let loader = Loader::open(&args.target).context("open target")?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35MoeModel::from_loader(&loader).context("load MoE target")?;
    drop(loader);
    let taps = [1_usize];
    let forward = |tokens: &[u32], start: usize, cache: &mut Vec<_>, mode: Mode| {
        target.dflash2_forward_target_on(
            &ids(tokens)?,
            &build_position_ids(start as i32, tokens.len() as i32)?,
            Some(cache),
            &taps,
            mode,
            ().into(),
        )
    };
    let argmax = |logits: &Array| -> Result<u32> {
        Ok(mlx::ops::reduction::argmax(logits, -1, false)?
            .reshape((-1,))?
            .to_vec::<u32>()?[0])
    };

    if args.mode == "generate" {
        let path = args.prompts.as_ref().context("--prompts")?;
        let prompts: Vec<PromptCase> = serde_json::from_reader(std::fs::File::open(path)?)?;
        let mut sequences = Vec::new();
        for case in prompts {
            let prompt = match case.prompt.strip_prefix('@') {
                Some(file) => std::fs::read_to_string(file)?,
                None => case.prompt.clone(),
            };
            let text = tokenizer.apply_chat_template(
                &[Message {
                    role: "user".into(),
                    content: prompt,
                }],
                true,
                Some(&json!({"enable_thinking": false})),
            )?;
            let mut tokens = tokenizer.encode(&text, false)?;
            let decode_from = tokens.len();
            let cap = (decode_from + case.new_tokens + 8) as i32;
            let mut cache = target.make_cache(1, cap, target.cache_dtype())?;
            let out = forward(&tokens, 0, &mut cache, Mode::Prefill)?;
            let len = out.hidden.shape().as_slice()[1];
            let mut logits =
                target.dflash2_project_hidden_on(&rows(&out.hidden, len - 1, len)?, ().into())?;
            for _ in 0..case.new_tokens {
                let token = argmax(&logits)?;
                let position = tokens.len();
                tokens.push(token);
                let out = forward(&[token], position, &mut cache, Mode::OrdinaryDecode)?;
                logits = target.dflash2_project_hidden_on(&out.hidden, ().into())?;
            }
            eprintln!(
                "generated {}: prompt {decode_from} total {}",
                case.id,
                tokens.len()
            );
            sequences.push(Sequence {
                id: case.id,
                tokens,
                decode_from,
            });
        }
        serde_json::to_writer(std::fs::File::create(&args.sequences)?, &sequences)?;
        return Ok(());
    }

    let out_dir = args.output_dir.as_ref().context("--output-dir")?;
    std::fs::create_dir_all(out_dir)?;
    let sequences: Vec<Sequence> = serde_json::from_reader(std::fs::File::open(&args.sequences)?)?;
    let mut index = std::fs::File::create(out_dir.join(format!("index.{}.jsonl", args.mode)))?;
    for seq in &sequences {
        let n = seq.tokens.len();
        let from = seq.decode_from;
        let mut cache = target.make_cache(1, (n + 8) as i32, target.cache_dtype())?;
        let mut file =
            std::fs::File::create(out_dir.join(format!("{}.{}.bf16", seq.id, args.mode)))?;
        let mut vocab = 0;
        // Logits predicting tokens from..n, i.e. at input positions from-1..n-1.
        match args.mode.as_str() {
            "decode" => {
                let out = forward(&seq.tokens[..from], 0, &mut cache, Mode::Prefill)?;
                let len = out.hidden.shape().as_slice()[1];
                let logits = target
                    .dflash2_project_hidden_on(&rows(&out.hidden, len - 1, len)?, ().into())?;
                vocab = logits.shape().as_slice()[2];
                file.write_all(&bf16_bytes(&logits)?)?;
                for position in from..n - 1 {
                    let out = forward(
                        &seq.tokens[position..position + 1],
                        position,
                        &mut cache,
                        Mode::OrdinaryDecode,
                    )?;
                    let logits = target.dflash2_project_hidden_on(&out.hidden, ().into())?;
                    file.write_all(&bf16_bytes(&logits)?)?;
                }
            }
            "prefill" => {
                let out = forward(&seq.tokens[..n - 1], 0, &mut cache, Mode::Prefill)?;
                for start in ((from - 1)..(n - 1)).step_by(16) {
                    let end = (start + 16).min(n - 1);
                    let logits = target.dflash2_project_hidden_on(
                        &rows(&out.hidden, start as i32, end as i32)?,
                        ().into(),
                    )?;
                    vocab = logits.shape().as_slice()[2];
                    file.write_all(&bf16_bytes(&logits)?)?;
                }
            }
            other => anyhow::bail!("unknown mode {other}"),
        }
        writeln!(
            index,
            "{}",
            json!({"id": seq.id, "mode": args.mode, "positions": n - from, "vocab": vocab,
                   "decode_from": from, "len": n})
        )?;
        eprintln!("dumped {} {} positions={}", seq.id, args.mode, n - from);
    }
    Ok(())
}
