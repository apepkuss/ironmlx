//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::Args;
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
    pixel_values: Option<Vec<Array>>,
    image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    placeholders: Vec<String>,
    image_spatial_merge_size: i32,
    image_token_id: i32,
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

fn qwen_image_placeholder_string(token_count: usize) -> String {
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

/// Build the MiniCPM-V-4.6 image placeholder string:
/// `<image>` + `<|image_pad|>` × token_count + `</image>`.
///
/// When tokenised (all three are registered special tokens), this produces the
/// id sequence `[248078] + [248056]*token_count + [248079]`, which exactly
/// matches what the P2a gen-script (`gen_single_image_logits.py`,
/// `use_image_id=False`, `slice_mode=False`) dumps into
/// `expected_input_ids_img.npy`.
fn minicpmv46_image_placeholder_string(token_count: usize) -> String {
    let mut out = String::with_capacity(
        "<image>".len() + token_count * "<|image_pad|>".len() + "</image>".len(),
    );
    out.push_str("<image>");
    for _ in 0..token_count {
        out.push_str("<|image_pad|>");
    }
    out.push_str("</image>");
    out
}

fn gemma4_placeholder(token_count: usize) -> String {
    let mut out = String::from("<|image>");
    for _ in 0..token_count {
        out.push_str("<|image|>");
    }
    out.push_str("<image|>");
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

fn prepare_images(
    args: &GenerateArgs,
    loader: &Loader,
    tokenizer: &Tokenizer,
    model_type: &str,
    default_spatial_merge_size: i32,
) -> Result<PreparedImages> {
    if args.images.is_empty() {
        return Ok(PreparedImages {
            pixel_values: None,
            image_grid_thw: None,
            placeholders: Vec::new(),
            image_spatial_merge_size: default_spatial_merge_size,
            image_token_id: tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(IMAGE_TOKEN_ID),
        });
    }

    let mut all_pixel_values = Vec::with_capacity(args.images.len());
    let mut grids = Vec::with_capacity(args.images.len());
    let mut placeholders = Vec::with_capacity(args.images.len());

    let (spatial_merge_size, image_token_id) = if model_type == "gemma4" {
        let cfg = crate::models::gemma4::Gemma4Config::from_loader(loader)
            .context("Gemma4Config::from_loader")?;
        let vision_config = cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4 config has no vision_config"))?;
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let processed =
                crate::models::gemma4::image_processor::preprocess(&bytes, vision_config)
                    .with_context(|| format!("preprocessing --image {}", path.display()))?;
            all_pixel_values.push(processed.pixel_values);
            grids.push((1, processed.grid_h, processed.grid_w));
            placeholders.push(gemma4_placeholder(processed.soft_tokens));
        }
        (
            vision_config.pooling_kernel_size,
            tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .or(cfg.image_token_id)
                .unwrap_or(258_880),
        )
    } else if model_type == "minicpmv4_6" {
        // MiniCPM-V-4.6: use model-config image_token_id (248056 = <|image_pad|>);
        // spatial_merge_size = 4 (2×2 Merger, "16x" downsample mode).
        // Placeholder: <image> + <|image_pad|>×N + </image>  (use_image_id=False,
        // slice_mode=False), matching P2a gen_single_image_logits.py convention:
        // ids = [248078] + [248056]*N + [248079], N = (gh//4)*(gw//4).
        let vcfg = crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig::from_loader(loader)
            .context("MiniCpmV46VisionConfig::from_loader")?;
        // image_token_id from config (248056); fallback to tokenizer lookup, then literal.
        let image_tok_id = tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(vcfg.image_token_id);
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let (pixel_values, gh, gw) =
                crate::models::minicpmv4_6::image_processor::preprocess(&bytes)
                    .with_context(|| format!("preprocessing --image {}", path.display()))?;
            let grid = (1, gh, gw);
            let token_count = image_token_count_for_grid(grid, default_spatial_merge_size)?;
            all_pixel_values.push(pixel_values);
            grids.push(grid);
            placeholders.push(minicpmv46_image_placeholder_string(token_count));
        }
        (default_spatial_merge_size, image_tok_id)
    } else {
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let (pixel_values, gh, gw) = image_processor::preprocess(&bytes)
                .with_context(|| format!("preprocessing --image {}", path.display()))?;
            let grid = (1, gh, gw);
            let token_count = image_token_count_for_grid(grid, default_spatial_merge_size)?;
            all_pixel_values.push(pixel_values);
            grids.push(grid);
            placeholders.push(qwen_image_placeholder_string(token_count));
        }
        (
            default_spatial_merge_size,
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(IMAGE_TOKEN_ID),
        )
    };

    Ok(PreparedImages {
        pixel_values: Some(all_pixel_values),
        image_grid_thw: Some(grids),
        placeholders,
        image_spatial_merge_size: spatial_merge_size,
        image_token_id,
    })
}

fn run_generation_with_model<M: Model + DenseVlMethods>(
    model: &M,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()> {
    let prepared_images = prepare_images(
        args,
        loader,
        tokenizer,
        model_type,
        model.model_meta().spatial_merge_size,
    )?;
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
        image_spatial_merge_size: prepared_images.image_spatial_merge_size,
        image_token_id: prepared_images.image_token_id,
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

    let architecture =
        crate::models::ModelArchitecture::from_config_value(loader.config_raw_value())?;
    let model_type = architecture.model_type();

    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        crate::models::ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        crate::models::ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        crate::models::ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            let model = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
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
    fn minicpmv46_image_token_count_uses_4x_downsample() {
        // MiniCPM-V grid (28,36) → vision tokens (28/4)*(36/4) = 63.
        assert_eq!(image_token_count_for_grid((1, 28, 36), 4).unwrap(), 63);
    }

    #[test]
    fn minicpmv46_placeholder_wraps_correct_tokens() {
        // Verify the helper builds the correct <image>...<|image_pad|>...</image> string.
        let s = minicpmv46_image_placeholder_string(3);
        assert_eq!(s, "<image><|image_pad|><|image_pad|><|image_pad|></image>");
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
