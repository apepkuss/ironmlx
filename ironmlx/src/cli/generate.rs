//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::Args;
use mlx::Array;

use crate::core::generate::{GenerateEvent, GenerateRequest, GenerationStream, IMAGE_TOKEN_ID};
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::speculative::{
    resolve_mtp_draft_tokens, MtpDraftTokensArg, MtpSpeculativeConfig, MtpSpeculativeModel,
    MtpTextGenerationStream,
};
use crate::core::{Loader, Message, Model, Tokenizer};
use crate::models::qwen3_5::image_processor;
use crate::Result;

use super::KvQuantArg;

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

    /// MTP model directory. When set, generation uses the Qwen MTP head for
    /// text-only greedy speculative decoding.
    #[arg(long = "mtp-model-dir")]
    pub mtp_model_dir: Option<PathBuf>,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long)]
    pub mtp_draft_tokens: Option<usize>,

    /// KV cache quantization used by attention reads: none, turbo3, turbo4, or k3v4.
    #[arg(long = "kv-quant", value_enum, default_value = "none")]
    pub kv_quant: KvQuantArg,
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

fn gemma4_placeholder(token_count: usize) -> String {
    let mut out = String::from("<|image>");
    for _ in 0..token_count {
        out.push_str("<|image|>");
    }
    out.push_str("<image|>");
    out
}

fn diffusion_gemma_placeholder(token_count: usize) -> String {
    gemma4_placeholder(token_count)
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
    } else if model_type == "diffusion_gemma" {
        let cfg = crate::models::DiffusionGemmaConfig::from_loader(loader)
            .context("DiffusionGemmaConfig::from_loader")?;
        let vision_config = cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| anyhow!("DiffusionGemma config has no vision_config"))?;
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let processed =
                crate::models::gemma4::image_processor::preprocess(&bytes, vision_config)
                    .with_context(|| format!("preprocessing --image {}", path.display()))?;
            all_pixel_values.push(processed.pixel_values);
            grids.push((1, processed.grid_h, processed.grid_w));
            placeholders.push(diffusion_gemma_placeholder(processed.soft_tokens));
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
        // Multi-slice (LLaVA-UHD): source slice first, then refine patches row-major.
        // preprocess_sliced_to_parts is the single source of truth for the
        // divisibility guard and placeholder construction (CLI + serve share it).
        let vcfg = crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig::from_loader(loader)
            .context("MiniCpmV46VisionConfig::from_loader")?;
        // image_token_id: tokenizer lookup (<|image_pad|> → 248056) first; fallback to config image_token_id.
        let image_tok_id = tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(vcfg.image_token_id);
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading --image {}", path.display()))?;
            let parts = crate::models::minicpmv4_6::preprocess_sliced_to_parts(
                &bytes,
                default_spatial_merge_size,
            )
            .with_context(|| format!("preprocessing --image {}", path.display()))?;
            all_pixel_values.extend(parts.pixel_values);
            grids.extend(parts.grid_thw);
            placeholders.push(parts.placeholder);
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

fn ensure_mtp_generation_supported(
    architecture: crate::models::ModelArchitecture,
    has_images: bool,
    args: &GenerateArgs,
) -> Result<()> {
    if args.mtp_model_dir.is_none() {
        return Ok(());
    }
    if has_images {
        return Err(anyhow!(
            "--mtp-model-dir currently supports text-only generation; remove --image"
        ));
    }
    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense
        | crate::models::ModelArchitecture::Qwen35Moe => Ok(()),
        _ => Err(anyhow!(
            "--mtp-model-dir currently supports Qwen dense/MoE text models only"
        )),
    }
}

fn build_sampler(args: &GenerateArgs) -> Sampler {
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
    sampler
}

fn build_generate_request<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<GenerateRequest> {
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

    Ok(GenerateRequest {
        prompt_ids,
        max_new_tokens: args.max_tokens,
        sampler: build_sampler(args),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: args.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
        pixel_values: prepared_images.pixel_values,
        image_grid_thw: prepared_images.image_grid_thw,
        image_spatial_merge_size: prepared_images.image_spatial_merge_size,
        image_token_id: prepared_images.image_token_id,
    })
}

fn write_generation_events(
    mut next_token: impl FnMut() -> Result<Option<GenerateEvent>>,
) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while let Some(ev) = next_token()? {
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

fn run_generation_with_model<M: Model + DenseVlMethods>(
    model: &M,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()> {
    let request = build_generate_request(model, tokenizer, loader, model_type, args)?;

    let has_images = request.pixel_values.is_some();
    let mut stream = if has_images {
        GenerationStream::new(model, tokenizer, request)?
    } else {
        GenerationStream::new_text_only(model, tokenizer, request)?
    };
    write_generation_events(|| stream.next_token())
}

fn run_generation_with_mtp_model<M: MtpSpeculativeModel>(
    model: &M,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()> {
    let request = build_generate_request(model, tokenizer, loader, model_type, args)?;
    if request.pixel_values.is_some() {
        return Err(anyhow!(
            "--mtp-model-dir currently supports text-only generation; remove --image"
        ));
    }
    let mtp_dir = args
        .mtp_model_dir
        .as_ref()
        .ok_or_else(|| anyhow!("run_generation_with_mtp_model called without --mtp-model-dir"))?;
    if !mtp_dir.exists() {
        return Err(anyhow!(
            "--mtp-model-dir must point to a local directory (got '{}')",
            mtp_dir.display()
        ));
    }
    let mtp_loader = Loader::open_mtp(mtp_dir).context("Loader::open_mtp")?;
    let mtp = model
        .load_mtp_head(&mtp_loader)
        .context("loading MTP draft head")?;
    let draft_tokens = resolve_mtp_draft_tokens(
        loader.config_raw_value(),
        args.mtp_draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    let cfg = MtpSpeculativeConfig::new(draft_tokens, request.sampler)?;
    let mut stream = MtpTextGenerationStream::new_text_only(model, &mtp, tokenizer, request, cfg)?;
    write_generation_events(|| stream.next_token())
}

fn run_diffusion_gemma_generation(
    model: &crate::models::DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    loader: &Loader,
    args: &GenerateArgs,
) -> Result<()> {
    if args.mtp_model_dir.is_some() {
        return Err(anyhow!(
            "--mtp-model-dir is not supported for DiffusionGemma block diffusion"
        ));
    }
    let default_spatial_merge_size = model
        .config
        .vision_config
        .as_ref()
        .map(|vc| vc.pooling_kernel_size)
        .unwrap_or(3);
    let prepared_images = prepare_images(
        args,
        loader,
        tokenizer,
        "diffusion_gemma",
        default_spatial_merge_size,
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
    let generation_config = crate::models::DiffusionGemmaGenerationConfig::from_loader(loader)
        .context("DiffusionGemmaGenerationConfig::from_loader")?;
    let events = match (
        prepared_images.pixel_values.as_deref(),
        prepared_images.image_grid_thw.as_deref(),
    ) {
        (Some(pixel_values), Some(image_grid_thw)) => {
            crate::models::diffusion_gemma::generate_image_text(
                model,
                tokenizer,
                &prompt_ids,
                pixel_values,
                image_grid_thw,
                prepared_images.image_token_id,
                &generation_config,
                args.max_tokens,
                args.temperature,
                Some(args.seed),
            )?
        }
        _ => crate::models::diffusion_gemma::generate_text(
            model,
            tokenizer,
            &prompt_ids,
            &generation_config,
            args.max_tokens,
            args.temperature,
            Some(args.seed),
        )?,
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for ev in events {
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
    ensure_mtp_generation_supported(architecture, !args.images.is_empty(), &args)?;

    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            if args.mtp_model_dir.is_some() {
                run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
            } else {
                run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
            }
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            if args.mtp_model_dir.is_some()
                && crate::models::is_qwen36_moe_config(loader.config_raw_value())
            {
                let model = crate::models::Qwen36MoeModel::from_loader(&loader)
                    .context("Qwen36MoeModel::from_loader")?;
                run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
            } else {
                let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                    .context("Qwen35MoeModel::from_loader")?;
                if args.mtp_model_dir.is_some() {
                    run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
                } else {
                    run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
                }
            }
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
        crate::models::ModelArchitecture::DiffusionGemma => {
            let model = crate::models::DiffusionGemmaModel::from_loader(&loader)
                .context("DiffusionGemmaModel::from_loader")?;
            run_diffusion_gemma_generation(&model, &tokenizer, &loader, &args)
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
        // Verify the canonical fn builds the correct <image>...<|image_pad|>...</image> string.
        let s = crate::models::minicpmv4_6::image_placeholder_string(3);
        assert_eq!(s, "<image><|image_pad|><|image_pad|><|image_pad|></image>");
    }

    #[test]
    fn diffusion_gemma_placeholder_wraps_image_soft_tokens() {
        assert_eq!(
            diffusion_gemma_placeholder(2),
            "<|image><|image|><|image|><image|>"
        );
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
    fn mtp_args_default_off_and_parse_explicit_model_dir() {
        let default_cli =
            GenerateTestCli::parse_from(["test", "--model", "/tmp/model", "--prompt", "hello"]);
        assert!(default_cli.args.mtp_model_dir.is_none());
        assert_eq!(default_cli.args.mtp_draft_tokens, None);

        let enabled_cli = GenerateTestCli::parse_from([
            "test",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--mtp-model-dir",
            "/tmp/mtp",
            "--mtp-draft-tokens",
            "6",
        ]);
        assert_eq!(
            enabled_cli.args.mtp_model_dir.as_deref(),
            Some(std::path::Path::new("/tmp/mtp"))
        );
        assert_eq!(enabled_cli.args.mtp_draft_tokens, Some(6));
    }

    #[test]
    fn mtp_support_policy_allows_text_qwen_and_rejects_other_modes() {
        let mut args =
            GenerateTestCli::parse_from(["test", "--model", "/tmp/model", "--prompt", "hello"])
                .args;

        assert!(ensure_mtp_generation_supported(
            crate::models::ModelArchitecture::Gemma4,
            false,
            &args
        )
        .is_ok());

        args.mtp_model_dir = Some(PathBuf::from("/tmp/mtp"));
        assert!(ensure_mtp_generation_supported(
            crate::models::ModelArchitecture::Qwen35Dense,
            false,
            &args
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            crate::models::ModelArchitecture::Qwen35Moe,
            false,
            &args
        )
        .is_ok());

        let image_err = ensure_mtp_generation_supported(
            crate::models::ModelArchitecture::Qwen35Dense,
            true,
            &args,
        )
        .unwrap_err();
        assert!(image_err.to_string().contains("text-only"));

        let arch_err =
            ensure_mtp_generation_supported(crate::models::ModelArchitecture::Gemma4, false, &args)
                .unwrap_err();
        assert!(arch_err.to_string().contains("Qwen"));
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

    /// Verify the slice→token-count + placeholder wiring without a real model or
    /// image decode. Synthetic slice list: source grid (28,36), two refine patches
    /// (40,28) each — matching a 640×480 coco-sample-like image with best_grid (2,1).
    ///
    /// source_tokens = (28/4)*(36/4) = 7*9 = 63
    /// slice_tokens  = (40/4)*(28/4) = 10*7 = 70
    /// grid (2,1) → 1 row × 2 cols → 2 <slice> blocks, 0 inter-row newlines
    #[test]
    fn minicpmv46_multislice_token_count_and_placeholder_wiring() {
        let spatial_merge_size = 4_i32;
        let best_grid: Option<(i32, i32)> = Some((2, 1));

        // Synthetic per-slice (gh, gw) pairs: [source, patch0, patch1].
        let slice_grids: Vec<(i32, i32)> = vec![(28, 36), (40, 28), (40, 28)];

        // Source tokens: slice[0].
        let (src_gh, src_gw) = slice_grids[0];
        let source_tokens =
            ((src_gh / spatial_merge_size) * (src_gw / spatial_merge_size)) as usize;
        assert_eq!(source_tokens, 63, "source_tokens = (28/4)*(36/4) = 63");

        // Slice tokens: slice[1] (first patch; all patches have the same grid).
        let slice_tokens = if slice_grids.len() > 1 {
            let (sl_gh, sl_gw) = slice_grids[1];
            ((sl_gh / spatial_merge_size) * (sl_gw / spatial_merge_size)) as usize
        } else {
            0
        };
        assert_eq!(slice_tokens, 70, "slice_tokens = (40/4)*(28/4) = 70");

        // Build the placeholder using the canonical function.
        let grid = best_grid.unwrap_or((0, 0));
        let placeholder = crate::models::minicpmv4_6::sliced_image_placeholder_string(
            source_tokens,
            slice_tokens,
            grid,
        );

        // Structural checks: 1 <image>, 2 <slice>, no inter-row newlines.
        assert_eq!(
            placeholder.matches("<image>").count(),
            1,
            "exactly one <image> block"
        );
        assert_eq!(
            placeholder.matches("</image>").count(),
            1,
            "exactly one </image>"
        );
        assert_eq!(
            placeholder.matches("<slice>").count(),
            2,
            "grid (2,1) → 2 <slice> blocks"
        );
        assert_eq!(
            placeholder.matches("</slice>").count(),
            2,
            "grid (2,1) → 2 </slice>"
        );
        assert_eq!(
            placeholder.matches('\n').count(),
            0,
            "single row → 0 inter-row newlines"
        );

        // Token counts embedded in placeholder.
        let pad_count = placeholder.matches("<|image_pad|>").count();
        assert_eq!(
            pad_count,
            source_tokens + 2 * slice_tokens,
            "total pads = 63 + 2*70 = 203"
        );
    }
}
