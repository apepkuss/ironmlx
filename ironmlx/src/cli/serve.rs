//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use clap::Args;

use crate::core::scheduler::DenseVlMethods;
use crate::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
};
use crate::core::{server, Loader, Model, Tokenizer};
use crate::Result;

const DEFAULT_PREFILL_CHUNK_SIZE: usize = 2048;
const DEFAULT_B_MAX: usize = 1;
const DEFAULT_ADMISSION_DEADLINE_MS: u64 = 5;
const DEFAULT_ADMISSION_QUEUE_MAX: usize = 32;
const DEFAULT_MAX_CACHE_CAP: usize = 32768;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    /// HF repo-id resolution is deferred to a future phase; pass a local path for now.
    #[arg(long)]
    pub model: String,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head. Defaults to `2048` unless supplied by
    /// `--scheduler-profile`.
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,

    /// Maximum concurrent in-flight requests (Scheduler slot count).
    /// Requests beyond this limit go to the admission queue. Default `1`
    /// optimizes single-request prefill / decode by avoiding [B,T_max]-padded
    /// MoE compute when only one slot is occupied; pass `--b-max N > 1` to
    /// enable concurrent multi-request batching. `0` rejected at startup
    /// because Scheduler with zero slots cannot admit any request.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub b_max: Option<usize>,

    /// Admission-window deadline in milliseconds. After the first
    /// admit in a batch arrives, additional admits are absorbed until
    /// this deadline expires or the batch saturates at b_max.
    /// Defaults to `5` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub admission_deadline_ms: Option<u64>,

    /// Capacity of the FIFO admission queue. Requests received while
    /// the scheduler is saturated are parked here. `0` disables queueing
    /// (immediate Err on saturation — mirrors pre-3d behavior).
    /// Defaults to `32` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub admission_queue_max: Option<usize>,

    /// Maximum allowed `prompt_len + max_new_tokens` per request. Capped
    /// further at the model's `max_position_embeddings` (Qwen3.5-4B: 262144).
    /// Requests beyond this return HTTP 413 Payload Too Large. B1-p2.3f.
    /// Defaults to `32768` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub max_cache_cap: Option<usize>,

    /// Runtime scheduler profile exported by `scheduler-autotune select --write-profile`.
    #[arg(long)]
    pub scheduler_profile: Option<PathBuf>,

    /// Print scheduler/autotune diagnostics and recommendations at startup.
    /// Diagnose-only: this does not change any runtime parameter.
    #[arg(long, default_value_t = false)]
    pub scheduler_autotune_report: bool,

    /// P5h+1 T1 measurement probe: force selected span bodies (Lane A
    /// `first_token_sampling_materialize_and_sample` + the ROI substep
    /// closures under GatedAttention / GatedDeltaNet / SparseMoeBlock +
    /// `slice_last_and_project_lm_head` + `cache_state_update`) to call
    /// `mlx::transforms::eval` on returned `Array` value(s) before the
    /// span closes. Measurement-only: defaults OFF so production lazy-graph
    /// semantics are preserved. Use ONLY for P5h+1 attribution sweeps.
    #[cfg(feature = "p5h-profile")]
    #[arg(long, default_value_t = false)]
    pub p5h_measurement_eval_probes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SchedulerServeConfig {
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
}

impl Default for SchedulerServeConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            b_max: DEFAULT_B_MAX,
            admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
            admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
            max_cache_cap: DEFAULT_MAX_CACHE_CAP,
        }
    }
}

fn default_scheduler_profile_config() -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        b_max: DEFAULT_B_MAX,
        prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
        admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
        max_cache_cap: DEFAULT_MAX_CACHE_CAP,
    }
}

fn read_scheduler_runtime_profile(path: &Path) -> Result<SchedulerAutotuneRuntimeProfile> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn resolve_scheduler_serve_config(
    args: &ServeArgs,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
) -> Result<SchedulerServeConfig> {
    if let Some(profile) = profile {
        if profile.schema_version != 1 {
            bail!(
                "scheduler profile schema_version mismatch: expected 1, got {}",
                profile.schema_version
            );
        }
    }

    let base = profile
        .map(|profile| profile.config)
        .unwrap_or_else(default_scheduler_profile_config);
    let config = SchedulerServeConfig {
        prefill_chunk_size: args.prefill_chunk_size.unwrap_or(base.prefill_chunk_size),
        b_max: args.b_max.unwrap_or(base.b_max),
        admission_deadline_ms: args
            .admission_deadline_ms
            .unwrap_or(base.admission_deadline_ms),
        admission_queue_max: args.admission_queue_max.unwrap_or(base.admission_queue_max),
        max_cache_cap: args.max_cache_cap.unwrap_or(base.max_cache_cap),
    };

    if config.b_max == 0 {
        bail!("scheduler b_max must be >= 1");
    }

    Ok(config)
}

/// Generic serve helper — shared by all model types that satisfy the
/// `SchedulerActor<M>` / `AppState<M>` bounds.
///
/// The `DenseVlMethods` bound is required by `server::serve<M>` /
/// `SchedulerActor<M>`. The trait name is historical; both dense and MoE
/// Qwen3.5 variants implement it so the same OpenAI VL route can serve either
/// checkpoint family.
fn serve_with_model<M>(
    model: M,
    tokenizer: Tokenizer,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    vision_input: Option<server::VisionInputConfig>,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    #[cfg(feature = "p5h-profile")]
    {
        assert!(
            scheduler_config.b_max == 1 || crate::core::p5h::scheduler_decode_allow_multi_row(),
            "p5h-profile feature requires --b-max 1 for legacy request-root attribution \
             (single-active-row invariant per § 2.5a). Got --b-max {}. \
             Set IRONMLX_P5H_SCHEDULER_DECODE_ALLOW_MULTI_ROW=1 only for experimental \
             unary scheduler decode attribution, or rebuild without --features p5h-profile \
             to use ordinary multi-row batching.",
            scheduler_config.b_max,
        );
        if scheduler_config.b_max != 1 {
            tracing::warn!(
                "p5h-profile multi-row escape hatch enabled for scheduler decode attribution; \
                 use unary/non-streaming clients so legacy streaming request-root p5h trees do \
                 not enter their single-active-row prefill path"
            );
        }
    }

    // Surface b_max at boot so operators can confirm whether single-request
    // optimized mode (default) or multi-request batching is active without
    // having to inspect process args.
    if scheduler_config.b_max == 1 {
        tracing::info!(
            "ironmlx serve: b_max=1 (single-request optimized mode; \
             pass --b-max N > 1 to enable concurrent multi-request batching)"
        );
    } else {
        tracing::info!(
            "ironmlx serve: b_max={} (multi-request batching enabled; \
             pass --b-max 1 to switch to single-request optimized mode)",
            scheduler_config.b_max,
        );
    }

    let model_id = args.model.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio::Runtime::new")?;
    // P5h+1 T1: derive measurement-eval-probes flag (feature-gated CLI arg);
    // feature-off builds always pass `false` so the receiver-side `set_*`
    // call site can remain unconditional in signature.
    #[cfg(feature = "p5h-profile")]
    let p5h_measurement_eval_probes = args.p5h_measurement_eval_probes;
    #[cfg(not(feature = "p5h-profile"))]
    let p5h_measurement_eval_probes = false;
    runtime.block_on(server::serve(
        model,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        scheduler_config.prefill_chunk_size,
        scheduler_config.b_max,
        scheduler_config.admission_deadline_ms,
        scheduler_config.admission_queue_max,
        scheduler_config.max_cache_cap,
        args.scheduler_autotune_report,
        p5h_measurement_eval_probes,
        vision_input,
    ))
}

fn read_model_type(model_dir: &std::path::Path) -> Result<String> {
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))
}

pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            args.model
        ));
    }

    let scheduler_profile = args
        .scheduler_profile
        .as_deref()
        .map(read_scheduler_runtime_profile)
        .transpose()?;
    let scheduler_config = resolve_scheduler_serve_config(&args, scheduler_profile.as_ref())?;
    if let Some(profile) = &scheduler_profile {
        tracing::info!(
            "ironmlx serve: scheduler profile applied model_name={} hardware_label={}",
            profile.model_name,
            profile.hardware_label
        );
    }

    let model_type = read_model_type(&model_dir)?;
    let architecture = crate::models::ModelArchitecture::from_model_type(&model_type)?;
    // open_multimodal so Qwen VL checkpoints retain vision_tower.* keys.
    let loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let vision_input = match architecture {
        crate::models::ModelArchitecture::Gemma4 => {
            let cfg = crate::models::gemma4::Gemma4Config::from_loader(&loader)
                .context("Gemma4Config::from_loader")?;
            cfg.vision_config
                .map(|vision_config| server::VisionInputConfig::Gemma4 { vision_config })
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            Some(server::VisionInputConfig::MiniCpmV46 {
                spatial_merge_size: 4,
            })
        }
        _ => None,
    };

    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, vision_input)
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, vision_input)
        }
        crate::models::ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, vision_input)
        }
        crate::models::ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, None)
        }
        crate::models::ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, None)
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            // MiniCpmV46Model serves text + single-image VL (vision_input set above).
            let model = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            serve_with_model(model, tokenizer, &args, scheduler_config, vision_input)
        }
    }
}

#[cfg(test)]
mod scheduler_profile_tests {
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    };

    use super::{resolve_scheduler_serve_config, SchedulerServeConfig, ServeArgs};

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 7,
            admission_queue_max: 16,
            max_cache_cap: 8192,
        }
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: 1,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            config: profile_config(),
        }
    }

    fn base_args() -> ServeArgs {
        ServeArgs {
            model: "/tmp/model".to_string(),
            port: 8080,
            host: "127.0.0.1".to_string(),
            prefill_chunk_size: None,
            b_max: None,
            admission_deadline_ms: None,
            admission_queue_max: None,
            max_cache_cap: None,
            scheduler_profile: None,
            scheduler_autotune_report: false,
            #[cfg(feature = "p5h-profile")]
            p5h_measurement_eval_probes: false,
        }
    }

    #[test]
    fn scheduler_profile_supplies_missing_scheduler_values() {
        let args = base_args();

        let config =
            resolve_scheduler_serve_config(&args, Some(&runtime_profile())).expect("resolved");

        assert_eq!(
            config,
            SchedulerServeConfig {
                prefill_chunk_size: 1024,
                b_max: 2,
                admission_deadline_ms: 7,
                admission_queue_max: 16,
                max_cache_cap: 8192,
            }
        );
    }

    #[test]
    fn scheduler_profile_cli_values_override_profile_values() {
        let args = ServeArgs {
            prefill_chunk_size: Some(256),
            b_max: Some(1),
            admission_deadline_ms: Some(9),
            admission_queue_max: Some(7),
            max_cache_cap: Some(4096),
            ..base_args()
        };

        let config =
            resolve_scheduler_serve_config(&args, Some(&runtime_profile())).expect("resolved");

        assert_eq!(
            config,
            SchedulerServeConfig {
                prefill_chunk_size: 256,
                b_max: 1,
                admission_deadline_ms: 9,
                admission_queue_max: 7,
                max_cache_cap: 4096,
            }
        );
    }
}

#[cfg(test)]
#[cfg(feature = "p5h-profile")]
mod tests {
    #[test]
    fn p5h_scheduler_decode_multi_row_escape_hatch_is_owned_by_p5h_config() {
        assert!(crate::core::p5h::scheduler_decode_allow_multi_row_from_env_value(Some("1")));
    }
}
