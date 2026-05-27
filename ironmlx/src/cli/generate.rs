//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::Args;
use mlx::ops::shape::concatenate;
use mlx::Array;

use crate::core::generate::{GenerateRequest, GenerationStream, IMAGE_TOKEN_ID};
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::{Loader, Message, Model, Tokenizer};
use crate::models::qwen3_5::image_processor;
use crate::Result;

#[derive(Args, Debug)]
pub struct GenerateArgs {
    #[arg(long)]
    pub model: String,

    #[arg(long)]
    pub prompt: String,

    /// Local image path. Repeat to provide multiple images. If the prompt
    /// contains <image> markers, they are replaced in argument order;
    /// otherwise image placeholders are prepended before the prompt.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Apply the chat template; set to false to tokenize the raw prompt.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub chat: bool,

    /// Enable thinking-mode chat templates. Defaults off so CLI generation
    /// returns the requested answer directly unless the caller opts in.
    #[arg(long, default_value_t = false)]
    pub enable_thinking: bool,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head.
    #[arg(long, default_value_t = 2048)]
    pub prefill_chunk_size: usize,
}

struct PreparedImages {
    pixel_values: Option<Array>,
    image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    placeholders: Vec<String>,
}

fn image_token_count_for_grid(grid: (i32, i32, i32), spatial_merge_size: i32) -> Result<usize> {
    let (_t, gh, gw) = grid;
    if spatial_merge_size <= 0 {
        return Err(anyhow!(
            "image_spatial_merge_size must be > 0, got {spatial_merge_size}"
        ));
    }
    if gh % spatial_merge_size != 0 || gw % spatial_merge_size != 0 {
        return Err(anyhow!(
            "image grid {gh}x{gw} is not divisible by spatial_merge_size={spatial_merge_size}"
        ));
    }
    Ok(((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize)
}

fn image_placeholder_string(token_count: usize) -> String {
    let mut out = String::with_capacity(
        "<|vision_start|>".len() + token_count * "<|image_pad|>".len() + "<|vision_end|>".len(),
    );
    out.push_str("<|vision_start|>");
    for _ in 0..token_count {
        out.push_str("<|image_pad|>");
    }
    out.push_str("<|vision_end|>");
    out
}

fn inject_image_placeholders(prompt: &str, placeholders: &[String]) -> Result<String> {
    if placeholders.is_empty() {
        return Ok(prompt.to_owned());
    }

    let marker = "<image>";
    let marker_count = prompt.match_indices(marker).count();
    if marker_count == 0 {
        let mut out = String::new();
        for placeholder in placeholders {
            out.push_str(placeholder);
        }
        out.push_str(prompt);
        return Ok(out);
    }

    if marker_count != placeholders.len() {
        return Err(anyhow!(
            "prompt contains {marker_count} <image> markers but {} --image arguments were provided",
            placeholders.len()
        ));
    }

    let mut out =
        String::with_capacity(prompt.len() + placeholders.iter().map(String::len).sum::<usize>());
    let mut rest = prompt;
    for placeholder in placeholders {
        let Some(idx) = rest.find(marker) else {
            break;
        };
        out.push_str(&rest[..idx]);
        out.push_str(placeholder);
        rest = &rest[idx + marker.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

fn prepare_images(args: &GenerateArgs, spatial_merge_size: i32) -> Result<PreparedImages> {
    if args.images.is_empty() {
        return Ok(PreparedImages {
            pixel_values: None,
            image_grid_thw: None,
            placeholders: Vec::new(),
        });
    }

    let mut all_pixel_values = Vec::with_capacity(args.images.len());
    let mut grids = Vec::with_capacity(args.images.len());
    let mut placeholders = Vec::with_capacity(args.images.len());

    for path in &args.images {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading --image {}", path.display()))?;
        let (pixel_values, gh, gw) = image_processor::preprocess(&bytes)
            .with_context(|| format!("preprocessing --image {}", path.display()))?;
        let grid = (1, gh, gw);
        let token_count = image_token_count_for_grid(grid, spatial_merge_size)?;
        all_pixel_values.push(pixel_values);
        grids.push(grid);
        placeholders.push(image_placeholder_string(token_count));
    }

    let refs: Vec<&Array> = all_pixel_values.iter().collect();
    let pixel_values = concatenate(&refs, 0).context("concatenating CLI image pixel_values")?;
    mlx::transforms::eval(&[&pixel_values]).context("evaluating CLI image pixel_values")?;

    Ok(PreparedImages {
        pixel_values: Some(pixel_values),
        image_grid_thw: Some(grids),
        placeholders,
    })
}

fn run_generation_with_model<M: Model + DenseVlMethods>(
    model: &M,
    tokenizer: &Tokenizer,
    args: &GenerateArgs,
) -> Result<()> {
    let spatial_merge_size = model.model_meta().spatial_merge_size;
    let prepared_images = prepare_images(args, spatial_merge_size)?;
    let prompt_content = inject_image_placeholders(&args.prompt, &prepared_images.placeholders)?;
    let prompt = if args.chat && tokenizer.has_chat_template() {
        let messages = vec![Message {
            role: "user".into(),
            content: prompt_content,
        }];
        let extra_kwargs = serde_json::json!({"enable_thinking": args.enable_thinking});
        tokenizer.apply_chat_template(&messages, true, Some(&extra_kwargs))?
    } else {
        prompt_content
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
        pixel_values: prepared_images.pixel_values,
        image_grid_thw: prepared_images.image_grid_thw,
        image_spatial_merge_size: spatial_merge_size,
        image_token_id: tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(IMAGE_TOKEN_ID),
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    };

    let has_images = request.pixel_values.is_some();
    let mut stream = if has_images {
        GenerationStream::new(model, tokenizer, request)?
    } else {
        GenerationStream::new_text_only(model, tokenizer, request)?
    };
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
    let loader = if args.images.is_empty() {
        Loader::open(&model_dir).context("Loader::open")?
    } else {
        Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?
    };
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
            if crate::models::is_qwen36_moe_config(loader.config_raw_value()) {
                let model = crate::models::Qwen36MoeModel::from_loader(&loader)
                    .context("Qwen36MoeModel::from_loader")?;
                run_generation_with_model(&model, &tokenizer, &args)
            } else {
                let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                    .context("Qwen35MoeModel::from_loader")?;
                run_generation_with_model(&model, &tokenizer, &args)
            }
        }
        "gemma4" => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &args)
        }
        other => Err(anyhow::anyhow!(
            "unsupported model_type: {other} (expected 'qwen3_5', 'qwen3_5_moe', or 'gemma4')"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct GenerateTestCli {
        #[command(flatten)]
        args: GenerateArgs,
    }

    #[test]
    fn image_token_count_uses_spatial_merge_size() {
        assert_eq!(image_token_count_for_grid((1, 4, 6), 2).unwrap(), 6);
    }

    #[test]
    fn enable_thinking_defaults_off_and_can_be_enabled() {
        let default_cli =
            GenerateTestCli::parse_from(["test", "--model", "/tmp/model", "--prompt", "hello"]);
        assert!(!default_cli.args.enable_thinking);

        let enabled_cli = GenerateTestCli::parse_from([
            "test",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--enable-thinking",
        ]);
        assert!(enabled_cli.args.enable_thinking);
    }

    #[test]
    fn inject_image_placeholders_replaces_markers_in_order() {
        let out = inject_image_placeholders(
            "A <image> then B <image>",
            &["[img0]".to_owned(), "[img1]".to_owned()],
        )
        .unwrap();
        assert_eq!(out, "A [img0] then B [img1]");
    }

    #[test]
    fn inject_image_placeholders_prepends_when_prompt_has_no_markers() {
        let out = inject_image_placeholders(
            "Describe this.",
            &["[img0]".to_owned(), "[img1]".to_owned()],
        )
        .unwrap();
        assert_eq!(out, "[img0][img1]Describe this.");
    }

    #[test]
    fn inject_image_placeholders_rejects_marker_count_mismatch() {
        let err =
            inject_image_placeholders("A <image>", &["[img0]".to_owned(), "[img1]".to_owned()])
                .expect_err("marker mismatch");
        assert!(err.to_string().contains("markers"));
    }
}
