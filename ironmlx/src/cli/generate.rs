//! `ironmlx generate` — single-prompt CLI generation backed by core::generate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::Args;

use crate::Result;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::speculative_model::MtpSpeculativeModel;
use ironmlx_lm::core::vision::DenseVlMethods;
use ironmlx_runtime::core::generate::GenerationStream;
use {
    ironmlx_lm::core::chat_template::Message, ironmlx_lm::core::loader::Loader,
    ironmlx_lm::core::model::Model, ironmlx_lm::core::tokenizer::Tokenizer,
    ironmlx_runtime::core::dflash2::DFlash2TextGenerationStream,
};
use {
    ironmlx_runtime::core::generation_types::GenerateEvent,
    ironmlx_runtime::core::generation_types::GenerateRequest,
};
use {
    ironmlx_runtime::core::speculative::resolve_mtp_draft_tokens,
    ironmlx_runtime::core::speculative::MtpDraftTokensArg,
    ironmlx_runtime::core::speculative::MtpSpeculativeConfig,
    ironmlx_runtime::core::speculative::MtpTextGenerationStream,
};

use super::KvQuantArg;

use ironmlx_lm::core::prompt_images::{inject_image_placeholders, NamedImage, PreparedImages};

fn prepare_images(
    args: &GenerateArgs,
    loader: &Loader,
    tokenizer: &Tokenizer,
    model_type: &str,
    default_spatial_merge_size: i32,
) -> Result<PreparedImages> {
    let images = args.images.iter().map(|path| {
        std::fs::read(path)
            .with_context(|| format!("reading --image {}", path.display()))
            .map(|bytes| NamedImage {
                label: format!("--image {}", path.display()),
                bytes,
            })
    });
    ironmlx_lm::core::prompt_images::prepare_images(
        images,
        loader,
        tokenizer,
        model_type,
        default_spatial_merge_size,
    )
}

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

    /// MTP/drafter model directory. When set, generation uses Qwen MTP heads or
    /// Gemma4 assistant drafter weights for greedy or exact sampled speculative decoding.
    #[arg(long = "mtp-model-dir")]
    pub mtp_model_dir: Option<PathBuf>,

    /// Official DFlash2 draft checkpoint directory. This selects the isolated,
    /// text-only DFlash2 path with greedy or exact sampled decoding.
    #[arg(long = "dflash2-model-dir")]
    pub dflash2_model_dir: Option<PathBuf>,

    /// DFlash2 proposal block width. Current MLX quantized target kernels are
    /// fastest at width 4 for the official Qwen3.8 checkpoint.
    #[arg(long, default_value_t = 4)]
    pub dflash2_block_size: usize,

    /// Runtime affine quantization for the official BF16 DFlash2 draft. Zero
    /// keeps BF16; 4 and 8 select the supported quantized variants.
    #[arg(long, default_value_t = 4)]
    pub dflash2_draft_bits: i32,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long)]
    pub mtp_draft_tokens: Option<usize>,

    /// KV cache quantization used by attention reads: none, turbo3, turbo4, or k3v4.
    #[arg(long = "kv-quant", value_enum, default_value = "none")]
    pub(crate) kv_quant: KvQuantArg,
}

fn ensure_mtp_generation_supported(
    architecture: ironmlx_lm::models::ModelArchitecture,
    _has_images: bool,
    args: &GenerateArgs,
) -> Result<()> {
    if args.mtp_model_dir.is_none() {
        return Ok(());
    }
    match architecture {
        ironmlx_lm::models::ModelArchitecture::Qwen35Dense
        | ironmlx_lm::models::ModelArchitecture::Qwen35Moe
        | ironmlx_lm::models::ModelArchitecture::Gemma4 => Ok(()),
        _ => Err(anyhow!(
            "--mtp-model-dir currently supports Qwen/Gemma4 generation only"
        )),
    }
}

fn ensure_dflash2_generation_supported(
    architecture: ironmlx_lm::models::ModelArchitecture,
    args: &GenerateArgs,
) -> Result<()> {
    if args.dflash2_model_dir.is_none() {
        return Ok(());
    }
    if args.mtp_model_dir.is_some() || args.mtp_draft_tokens.is_some() {
        return Err(anyhow!(
            "--dflash2-model-dir cannot be combined with MTP arguments"
        ));
    }
    if architecture != ironmlx_lm::models::ModelArchitecture::Qwen35Dense {
        return Err(anyhow!(
            "--dflash2-model-dir currently supports dense Qwen3.5 targets only"
        ));
    }
    if !args.images.is_empty() {
        return Err(anyhow!(
            "--dflash2-model-dir is text-only and cannot be combined with --image"
        ));
    }
    if args.kv_quant.turboquant_bits().is_some() {
        return Err(anyhow!(
            "--dflash2-model-dir P0-P2 has not qualified --kv-quant"
        ));
    }
    if !(2..=8).contains(&args.dflash2_block_size) {
        return Err(anyhow!(
            "--dflash2-block-size must be in [2, 8] for the official Qwen3.8 draft"
        ));
    }
    if !matches!(args.dflash2_draft_bits, 0 | 4 | 8) {
        return Err(anyhow!("--dflash2-draft-bits must be one of 0, 4, or 8"));
    }
    Ok(())
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
        constraint: None,
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

fn run_generation_with_dflash2_model(
    model: &ironmlx_lm::models::Qwen35Model,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()> {
    let request = build_generate_request(model, tokenizer, loader, model_type, args)?;
    let draft_dir = args.dflash2_model_dir.as_ref().ok_or_else(|| {
        anyhow!("run_generation_with_dflash2_model called without --dflash2-model-dir")
    })?;
    if !draft_dir.is_dir() {
        return Err(anyhow!(
            "--dflash2-model-dir must point to a local directory (got '{}')",
            draft_dir.display()
        ));
    }
    let draft_loader = Loader::open_dflash2(draft_dir).context("Loader::open_dflash2")?;
    let draft_bits = (args.dflash2_draft_bits != 0).then_some(args.dflash2_draft_bits);
    let draft = ironmlx_lm::models::DFlash2DraftModel::from_loader(
        &draft_loader,
        model.config(),
        draft_bits,
    )
    .context("DFlash2DraftModel::from_loader")?;
    drop(draft_loader);
    mlx::clear_cache();
    let mut stream = DFlash2TextGenerationStream::new_text_only(
        model,
        &draft,
        tokenizer,
        request,
        args.dflash2_block_size,
    )?;
    write_generation_events(|| stream.next_token())?;
    eprintln!(
        "ironmlx generate: dflash2_metrics {}",
        serde_json::to_string(&stream.metrics())?
    );
    Ok(())
}

fn run_generation_with_mtp_model<M>(
    model: &M,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    let request = build_generate_request(model, tokenizer, loader, model_type, args)?;
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
    if request.pixel_values.is_some() {
        let mut stream = ironmlx_runtime::core::single_request::MtpSchedulerGenerationStream::new(
            model, &mtp, tokenizer, request, cfg,
        )?;
        write_generation_events(|| stream.next_token())
    } else {
        let mut stream =
            MtpTextGenerationStream::new_text_only(model, &mtp, tokenizer, request, cfg)?;
        write_generation_events(|| stream.next_token())
    }
}

fn run_generation_with_gemma4_drafter_model(
    model: &ironmlx_lm::models::Gemma4Model,
    tokenizer: &Tokenizer,
    loader: &Loader,
    model_type: &str,
    args: &GenerateArgs,
) -> Result<()> {
    let request = build_generate_request(model, tokenizer, loader, model_type, args)?;
    let mtp_dir = args.mtp_model_dir.as_ref().ok_or_else(|| {
        anyhow!("run_generation_with_gemma4_drafter_model called without --mtp-model-dir")
    })?;
    if !mtp_dir.exists() {
        return Err(anyhow!(
            "--mtp-model-dir must point to a local directory (got '{}')",
            mtp_dir.display()
        ));
    }
    let drafter_loader =
        Loader::open_gemma4_drafter(mtp_dir).context("Loader::open_gemma4_drafter")?;
    let drafter = ironmlx_lm::models::gemma4::Gemma4AssistantModel::from_loader(&drafter_loader)
        .context("Gemma4AssistantModel::from_loader")?;
    let draft_tokens = resolve_mtp_draft_tokens(
        loader.config_raw_value(),
        args.mtp_draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    let cfg = MtpSpeculativeConfig::new(draft_tokens, request.sampler)?;
    let mut stream = ironmlx_runtime::core::gemma4_generation::Gemma4DrafterGenerationStream::new(
        model, &drafter, tokenizer, request, cfg,
    )?;
    write_generation_events(|| stream.next_token())
}

fn run_diffusion_gemma_generation(
    model: &ironmlx_lm::models::DiffusionGemmaModel,
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
    let generation_config = ironmlx_lm::models::DiffusionGemmaGenerationConfig::from_loader(loader)
        .context("DiffusionGemmaGenerationConfig::from_loader")?;
    let events = match (
        prepared_images.pixel_values.as_deref(),
        prepared_images.image_grid_thw.as_deref(),
    ) {
        (Some(pixel_values), Some(image_grid_thw)) => {
            ironmlx_lm::models::diffusion_gemma::generate_image_text(
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
        _ => ironmlx_lm::models::diffusion_gemma::generate_text(
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
    let mut loader = if args.images.is_empty() {
        Loader::open(&model_dir).context("Loader::open")?
    } else {
        Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?
    };
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;

    let architecture =
        ironmlx_lm::models::ModelArchitecture::from_config_value(loader.config_raw_value())?;
    let model_type = architecture.model_type();
    ensure_mtp_generation_supported(architecture, !args.images.is_empty(), &args)?;
    ensure_dflash2_generation_supported(architecture, &args)?;

    match architecture {
        ironmlx_lm::models::ModelArchitecture::Qwen35Dense => {
            let model = if args.dflash2_model_dir.is_some() {
                ironmlx_lm::models::Qwen35Model::from_loader_dflash2(&mut loader)
                    .context("Qwen35Model::from_loader_dflash2")?
            } else {
                ironmlx_lm::models::Qwen35Model::from_loader(&loader)
                    .context("Qwen35Model::from_loader")?
            };
            if args.dflash2_model_dir.is_some() {
                run_generation_with_dflash2_model(&model, &tokenizer, &loader, model_type, &args)
            } else if args.mtp_model_dir.is_some() {
                run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
            } else {
                run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
            }
        }
        ironmlx_lm::models::ModelArchitecture::Qwen35Moe => {
            if args.mtp_model_dir.is_some()
                && ironmlx_lm::models::is_qwen36_moe_config(loader.config_raw_value())
            {
                let model = ironmlx_lm::models::Qwen36MoeModel::from_loader(&loader)
                    .context("Qwen36MoeModel::from_loader")?;
                run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
            } else {
                let model = ironmlx_lm::models::Qwen35MoeModel::from_loader(&loader)
                    .context("Qwen35MoeModel::from_loader")?;
                if args.mtp_model_dir.is_some() {
                    run_generation_with_mtp_model(&model, &tokenizer, &loader, model_type, &args)
                } else {
                    run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
                }
            }
        }
        ironmlx_lm::models::ModelArchitecture::Gemma4 => {
            let model = ironmlx_lm::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            if args.mtp_model_dir.is_some() {
                run_generation_with_gemma4_drafter_model(
                    &model, &tokenizer, &loader, model_type, &args,
                )
            } else {
                run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
            }
        }
        ironmlx_lm::models::ModelArchitecture::Glm4MoeLite => {
            let model = ironmlx_lm::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        ironmlx_lm::models::ModelArchitecture::Llama => {
            let model = ironmlx_lm::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        ironmlx_lm::models::ModelArchitecture::MiniCpmV46 => {
            let model = ironmlx_lm::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            run_generation_with_model(&model, &tokenizer, &loader, model_type, &args)
        }
        ironmlx_lm::models::ModelArchitecture::DiffusionGemma => {
            let model = ironmlx_lm::models::DiffusionGemmaModel::from_loader(&loader)
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
        assert!(default_cli.args.dflash2_model_dir.is_none());
        assert_eq!(default_cli.args.mtp_draft_tokens, None);
        assert_eq!(default_cli.args.dflash2_block_size, 4);
        assert_eq!(default_cli.args.dflash2_draft_bits, 4);

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
    fn dflash2_arg_parses_and_policy_is_strictly_isolated() {
        let mut args = GenerateTestCli::parse_from([
            "test",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--dflash2-model-dir",
            "/tmp/dflash2",
        ])
        .args;
        assert_eq!(
            args.dflash2_model_dir.as_deref(),
            Some(std::path::Path::new("/tmp/dflash2"))
        );
        assert!(ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &args
        )
        .is_ok());

        let architecture_error = ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Moe,
            &args,
        )
        .expect_err("reject non-dense target");
        assert!(architecture_error.to_string().contains("dense Qwen3.5"));

        args.temperature = 0.8;
        assert!(ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &args,
        )
        .is_ok());

        args.temperature = 0.0;
        args.mtp_model_dir = Some(PathBuf::from("/tmp/mtp"));
        let isolation_error = ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &args,
        )
        .expect_err("reject MTP combination");
        assert!(isolation_error.to_string().contains("cannot be combined"));

        args.mtp_model_dir = None;
        args.dflash2_block_size = 1;
        let block_error = ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &args,
        )
        .expect_err("reject invalid block size");
        assert!(block_error.to_string().contains("must be in [2, 8]"));

        args.dflash2_block_size = 5;
        args.dflash2_draft_bits = 6;
        let draft_bits_error = ensure_dflash2_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &args,
        )
        .expect_err("reject unsupported draft quantization");
        assert!(draft_bits_error.to_string().contains("0, 4, or 8"));
    }

    #[test]
    fn mtp_support_policy_allows_qwen_and_gemma4_text_and_vl() {
        let mut args =
            GenerateTestCli::parse_from(["test", "--model", "/tmp/model", "--prompt", "hello"])
                .args;

        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Gemma4,
            false,
            &args
        )
        .is_ok());

        args.mtp_model_dir = Some(PathBuf::from("/tmp/mtp"));
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            false,
            &args
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Moe,
            false,
            &args
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            true,
            &args,
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Qwen35Moe,
            true,
            &args
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Gemma4,
            false,
            &args
        )
        .is_ok());
        assert!(ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Gemma4,
            true,
            &args
        )
        .is_ok());

        let arch_err = ensure_mtp_generation_supported(
            ironmlx_lm::models::ModelArchitecture::Llama,
            false,
            &args,
        )
        .unwrap_err();
        assert!(arch_err.to_string().contains("Qwen/Gemma4"));
    }
}
