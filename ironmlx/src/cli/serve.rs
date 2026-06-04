//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use clap::Args;

use super::scheduler_profile_store::{
    detect_scheduler_profile_hardware_label, SchedulerProfileStore,
};
use crate::core::scheduler::DenseVlMethods;
use crate::core::scheduler_autotune::{
    evaluate_scheduler_autotune_profile_health, SchedulerAutotuneProfileConfig,
    SchedulerAutotuneProfileHealthInput, SchedulerAutotuneProfileHealthReport,
    SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneRuntimeProfile,
    SchedulerAutotuneRuntimeProfileMetadata, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use crate::core::{server, Loader, Model, Tokenizer};
use crate::Result;

const DEFAULT_PREFILL_CHUNK_SIZE: usize = 2048;
const DEFAULT_B_MAX: usize = 1;
const DEFAULT_ADMISSION_DEADLINE_MS: u64 = 5;
const DEFAULT_ADMISSION_QUEUE_MAX: usize = 32;
const DEFAULT_MAX_CACHE_CAP: usize = 32768;
const DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP: usize = 256;

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

    /// Maximum chunk size used by rolling mid-admit while decode rows are active.
    /// Smaller values protect decode cadence under concurrent long-prompt admission;
    /// larger values can reduce queued-request TTFT at the cost of longer decode gaps.
    /// Defaults to `256` unless supplied by `--scheduler-profile`.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub decode_cadence_mid_chunk_cap: Option<usize>,

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
    decode_cadence_mid_chunk_cap: usize,
}

impl Default for SchedulerServeConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            b_max: DEFAULT_B_MAX,
            admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
            admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
            max_cache_cap: DEFAULT_MAX_CACHE_CAP,
            decode_cadence_mid_chunk_cap: DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP,
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
        decode_cadence_mid_chunk_cap: DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP,
    }
}

fn default_scheduler_runtime_profile() -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "default".to_string(),
        hardware_label: "local".to_string(),
        config: default_scheduler_profile_config(),
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
    }
}

fn read_scheduler_runtime_profile(path: &Path) -> Result<SchedulerAutotuneRuntimeProfile> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[derive(Debug)]
struct SchedulerProfileLoad {
    path: PathBuf,
    profile: SchedulerAutotuneRuntimeProfile,
    auto_loaded: bool,
}

fn load_scheduler_profile_for_model(
    args: &ServeArgs,
    model_dir: &Path,
    store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
) -> Result<Option<SchedulerProfileLoad>> {
    if let Some(path) = args.scheduler_profile.as_deref() {
        return Ok(Some(SchedulerProfileLoad {
            path: path.to_path_buf(),
            profile: read_scheduler_runtime_profile(path)?,
            auto_loaded: false,
        }));
    }

    let Some(store) = store else {
        return Ok(None);
    };
    let model_name = scheduler_profile_model_name(model_dir)?;
    let Some(path) = (match store.find_profile(model_dir, &model_name, hardware_label) {
        Ok(path) => path,
        Err(error) => {
            tracing::warn!(
                "ironmlx serve: scheduler profile store unavailable path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                store.root().display(),
                model_name,
                hardware_label,
                error
            );
            None
        }
    }) else {
        return Ok(None);
    };

    let profile = match read_scheduler_runtime_profile(&path) {
        Ok(profile) => profile,
        Err(error) => {
            tracing::warn!(
                "ironmlx serve: scheduler profile ignored path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                path.display(),
                model_name,
                hardware_label,
                error
            );
            return Ok(None);
        }
    };

    Ok(Some(SchedulerProfileLoad {
        profile,
        path,
        auto_loaded: true,
    }))
}

fn check_loaded_scheduler_profile_health(
    profile: &SchedulerAutotuneRuntimeProfile,
    expected_model_name: &str,
    expected_hardware_label: &str,
    now_unix_ms: u64,
) -> Result<SchedulerAutotuneProfileHealthReport> {
    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile,
        expected_model_name,
        expected_hardware_label,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms,
        max_age_days: 30,
    });
    if report.status == SchedulerAutotuneProfileHealthStatus::Invalid {
        bail!("invalid scheduler profile:\n{}", report.render_text());
    }
    Ok(report)
}

fn log_scheduler_profile_health(
    profile_path: &Path,
    report: &SchedulerAutotuneProfileHealthReport,
) {
    let note_codes = report
        .notes
        .iter()
        .map(|note| note.code.as_str())
        .collect::<Vec<_>>()
        .join(",");
    match report.status {
        SchedulerAutotuneProfileHealthStatus::Healthy => {
            tracing::info!(
                "ironmlx serve: scheduler profile health status={} path={} notes={}",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
        SchedulerAutotuneProfileHealthStatus::Warning => {
            tracing::warn!(
                "ironmlx serve: scheduler profile health status={} path={} notes={} recommendation=\"rerun scheduler-autotune calibrate for this model\"",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
        SchedulerAutotuneProfileHealthStatus::Invalid => {
            tracing::warn!(
                "ironmlx serve: scheduler profile health status={} path={} notes={}",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
    }
}

fn scheduler_profile_model_name(model_dir: &Path) -> Result<String> {
    model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("--model has no directory name for scheduler profile lookup")
        })
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

fn apply_scheduler_cli_overrides(
    args: &ServeArgs,
    base: SchedulerAutotuneProfileConfig,
) -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        prefill_chunk_size: args.prefill_chunk_size.unwrap_or(base.prefill_chunk_size),
        b_max: args.b_max.unwrap_or(base.b_max),
        admission_deadline_ms: args
            .admission_deadline_ms
            .unwrap_or(base.admission_deadline_ms),
        admission_queue_max: args.admission_queue_max.unwrap_or(base.admission_queue_max),
        max_cache_cap: args.max_cache_cap.unwrap_or(base.max_cache_cap),
        decode_cadence_mid_chunk_cap: args
            .decode_cadence_mid_chunk_cap
            .unwrap_or(base.decode_cadence_mid_chunk_cap),
    }
}

fn validate_scheduler_serve_config(config: SchedulerAutotuneProfileConfig) -> Result<()> {
    if config.b_max == 0 {
        bail!("scheduler b_max must be >= 1");
    }
    if config.decode_cadence_mid_chunk_cap == 0 {
        bail!("scheduler decode_cadence_mid_chunk_cap must be >= 1");
    }
    Ok(())
}

fn validate_dynamic_rules(profile: &SchedulerAutotuneRuntimeProfile) -> Result<()> {
    for rule in &profile.rules {
        if rule.config.b_max != profile.config.b_max
            || rule.config.admission_deadline_ms != profile.config.admission_deadline_ms
            || rule.config.admission_queue_max != profile.config.admission_queue_max
            || rule.config.max_cache_cap != profile.config.max_cache_cap
        {
            bail!(
                "scheduler profile dynamic rules may only vary prefill_chunk_size and decode_cadence_mid_chunk_cap"
            );
        }
        validate_scheduler_serve_config(rule.config)?;
    }
    Ok(())
}

fn resolve_scheduler_runtime_profile(
    args: &ServeArgs,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    if let Some(profile) = profile {
        if profile.schema_version != SCHEDULER_AUTOTUNE_SCHEMA_VERSION {
            bail!(
                "scheduler profile schema_version mismatch: expected {}, got {}",
                SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
                profile.schema_version
            );
        }
    }

    let mut resolved = profile
        .cloned()
        .unwrap_or_else(default_scheduler_runtime_profile);
    resolved.config = apply_scheduler_cli_overrides(args, resolved.config);
    for rule in &mut resolved.rules {
        rule.config = apply_scheduler_cli_overrides(args, rule.config);
    }
    validate_scheduler_serve_config(resolved.config)?;
    validate_dynamic_rules(&resolved)?;
    Ok(resolved)
}

#[cfg(test)]
fn resolve_scheduler_serve_config(
    args: &ServeArgs,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
) -> Result<SchedulerServeConfig> {
    let profile = resolve_scheduler_runtime_profile(args, profile)?;
    Ok(SchedulerServeConfig {
        prefill_chunk_size: profile.config.prefill_chunk_size,
        b_max: profile.config.b_max,
        admission_deadline_ms: profile.config.admission_deadline_ms,
        admission_queue_max: profile.config.admission_queue_max,
        max_cache_cap: profile.config.max_cache_cap,
        decode_cadence_mid_chunk_cap: profile.config.decode_cadence_mid_chunk_cap,
    })
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
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
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
        scheduler_config.decode_cadence_mid_chunk_cap,
        scheduler_runtime_profile,
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

    let scheduler_profile_store = if args.scheduler_profile.is_none() {
        match SchedulerProfileStore::default() {
            Ok(store) => Some(store),
            Err(error) => {
                tracing::warn!(
                    "ironmlx serve: scheduler profile store disabled error={:#}; using CLI/default scheduler config",
                    error
                );
                None
            }
        }
    } else {
        None
    };
    let scheduler_profile_hardware_label = detect_scheduler_profile_hardware_label();
    let mut scheduler_profile_load = load_scheduler_profile_for_model(
        &args,
        &model_dir,
        scheduler_profile_store.as_ref(),
        &scheduler_profile_hardware_label,
    )?;
    let scheduler_profile_model_name = scheduler_profile_model_name(&model_dir)?;
    let mut discard_auto_profile = false;
    if let Some(load) = scheduler_profile_load.as_ref() {
        match check_loaded_scheduler_profile_health(
            &load.profile,
            &scheduler_profile_model_name,
            &scheduler_profile_hardware_label,
            unix_time_ms(),
        ) {
            Ok(report) => log_scheduler_profile_health(&load.path, &report),
            Err(error) if load.auto_loaded => {
                tracing::warn!(
                    "ironmlx serve: scheduler profile ignored path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                    load.path.display(),
                    scheduler_profile_model_name,
                    scheduler_profile_hardware_label,
                    error
                );
                discard_auto_profile = true;
            }
            Err(error) => return Err(error),
        }
    }
    if discard_auto_profile {
        scheduler_profile_load = None;
    }
    let scheduler_runtime_profile = resolve_scheduler_runtime_profile(
        &args,
        scheduler_profile_load.as_ref().map(|load| &load.profile),
    )?;
    if scheduler_profile_load.is_none() && args.scheduler_profile.is_none() {
        match scheduler_profile_store.as_ref() {
            Some(store) => tracing::info!(
                "ironmlx serve: no matching scheduler profile found store={} model={} hardware_label={}; using CLI/default scheduler config",
                store.root().display(),
                model_dir.display(),
                scheduler_profile_hardware_label
            ),
            None => tracing::info!(
                "ironmlx serve: no scheduler profile store available model={} hardware_label={}; using CLI/default scheduler config",
                model_dir.display(),
                scheduler_profile_hardware_label
            ),
        }
    }
    let scheduler_config = SchedulerServeConfig {
        prefill_chunk_size: scheduler_runtime_profile.config.prefill_chunk_size,
        b_max: scheduler_runtime_profile.config.b_max,
        admission_deadline_ms: scheduler_runtime_profile.config.admission_deadline_ms,
        admission_queue_max: scheduler_runtime_profile.config.admission_queue_max,
        max_cache_cap: scheduler_runtime_profile.config.max_cache_cap,
        decode_cadence_mid_chunk_cap: scheduler_runtime_profile
            .config
            .decode_cadence_mid_chunk_cap,
    };
    if let Some(load) = &scheduler_profile_load {
        let source = if load.auto_loaded {
            "store"
        } else {
            "explicit"
        };
        tracing::info!(
            "ironmlx serve: scheduler profile applied source={} path={} model_name={} hardware_label={} rules={}",
            source,
            load.path.display(),
            load.profile.model_name,
            load.profile.hardware_label,
            scheduler_runtime_profile.rules.len()
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
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                vision_input,
            )
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                vision_input,
            )
        }
        crate::models::ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                vision_input,
            )
        }
        crate::models::ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                None,
            )
        }
        crate::models::ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                None,
            )
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            // MiniCpmV46Model serves text + single-image VL (vision_input set above).
            let model = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                vision_input,
            )
        }
    }
}

#[cfg(test)]
mod scheduler_profile_tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneProfileHealthStatus,
        SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeRule,
        SchedulerAutotuneRuntimeRuleCondition, SchedulerAutotuneScenario,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    use super::{
        check_loaded_scheduler_profile_health, load_scheduler_profile_for_model,
        resolve_scheduler_runtime_profile, resolve_scheduler_serve_config, SchedulerServeConfig,
        ServeArgs,
    };

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 7,
            admission_queue_max: 16,
            max_cache_cap: 8192,
            decode_cadence_mid_chunk_cap: 384,
        }
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            config: profile_config(),
            rules: vec![SchedulerAutotuneRuntimeRule {
                when: SchedulerAutotuneRuntimeRuleCondition {
                    prompt_len_gte: 8192,
                    max_new_tokens_gte: 512,
                    effective_concurrency_gte: 2,
                },
                config: SchedulerAutotuneProfileConfig {
                    prefill_chunk_size: 2048,
                    decode_cadence_mid_chunk_cap: 512,
                    ..profile_config()
                },
            }],
            metadata:
                crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                    1811606400000,
                ),
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
            decode_cadence_mid_chunk_cap: None,
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
                decode_cadence_mid_chunk_cap: 384,
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
            decode_cadence_mid_chunk_cap: Some(512),
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
                decode_cadence_mid_chunk_cap: 512,
            }
        );
    }

    #[test]
    fn scheduler_profile_cli_values_override_dynamic_rule_values() {
        let args = ServeArgs {
            prefill_chunk_size: Some(256),
            decode_cadence_mid_chunk_cap: Some(64),
            ..base_args()
        };

        let profile =
            resolve_scheduler_runtime_profile(&args, Some(&runtime_profile())).expect("resolved");

        assert_eq!(profile.config.prefill_chunk_size, 256);
        assert_eq!(profile.config.decode_cadence_mid_chunk_cap, 64);
        assert_eq!(profile.rules.len(), 1);
        assert_eq!(profile.rules[0].config.prefill_chunk_size, 256);
        assert_eq!(profile.rules[0].config.decode_cadence_mid_chunk_cap, 64);
    }

    #[test]
    fn scheduler_profile_health_warning_does_not_prevent_profile_resolution() {
        let mut profile = runtime_profile();
        profile.metadata.created_at_unix_ms = 1811606400000;
        profile.metadata.scenario_coverage = vec![SchedulerAutotuneScenario {
            prompt_len: 1024,
            max_new_tokens: 128,
            concurrency: 1,
        }];
        let args = base_args();

        let checked = check_loaded_scheduler_profile_health(
            &profile,
            "different-model-name",
            "test-host",
            1811606400000 + 31 * 24 * 60 * 60 * 1000,
        )
        .expect("warning health should not fail");

        assert_eq!(
            checked.status,
            SchedulerAutotuneProfileHealthStatus::Warning
        );
        assert!(resolve_scheduler_runtime_profile(&args, Some(&profile)).is_ok());
    }

    #[test]
    fn scheduler_profile_health_invalid_returns_error() {
        let mut profile = runtime_profile();
        profile.hardware_label = "other-host".to_string();

        let error = check_loaded_scheduler_profile_health(
            &profile,
            "test-model",
            "test-host",
            1811606400000,
        )
        .expect_err("invalid health should fail");

        assert!(format!("{error:#}").contains("invalid scheduler profile"));
    }

    #[test]
    fn serve_auto_loads_matching_profile_from_store_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-store-serve");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");
        let args = ServeArgs {
            model: model_dir.to_string_lossy().into_owned(),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(&args, &model_dir, Some(&store), "test-host")
            .expect("load profile")
            .expect("stored profile should match");

        assert_eq!(loaded.profile.config, profile_config());
        assert_eq!(
            loaded.path,
            store.profile_path("test-model", "test-host", &model_dir)
        );

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn explicit_scheduler_profile_overrides_profile_store() {
        let temp_dir = unique_temp_dir("scheduler-profile-explicit");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist stored profile");
        let explicit_profile = SchedulerAutotuneRuntimeProfile {
            model_name: "explicit-model".to_string(),
            config: SchedulerAutotuneProfileConfig {
                prefill_chunk_size: 4096,
                ..profile_config()
            },
            ..runtime_profile()
        };
        let explicit_path = temp_dir.join("explicit-profile.json");
        let output = serde_json::to_string_pretty(&explicit_profile).expect("serialize profile");
        std::fs::write(&explicit_path, format!("{output}\n")).expect("write explicit profile");
        let args = ServeArgs {
            model: model_dir.to_string_lossy().into_owned(),
            scheduler_profile: Some(explicit_path.clone()),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(&args, &model_dir, Some(&store), "test-host")
            .expect("load profile")
            .expect("explicit profile should load");

        assert_eq!(loaded.profile.model_name, "explicit-model");
        assert_eq!(loaded.profile.config.prefill_chunk_size, 4096);
        assert_eq!(loaded.path, explicit_path);

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn serve_ignores_corrupt_profile_store_index_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-corrupt-index");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store_root = temp_dir.join("store");
        std::fs::create_dir_all(&store_root).expect("create store dir");
        std::fs::write(store_root.join("index.json"), "not json").expect("write corrupt index");
        let store = SchedulerProfileStore::from_root(store_root);
        let args = ServeArgs {
            model: model_dir.to_string_lossy().into_owned(),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(&args, &model_dir, Some(&store), "test-host")
            .expect("corrupt store should fall back");

        assert!(loaded.is_none());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn serve_ignores_corrupt_auto_loaded_profile_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-corrupt-profile");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        let stored_path = store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");
        std::fs::write(&stored_path, "not json").expect("corrupt stored profile");
        let args = ServeArgs {
            model: model_dir.to_string_lossy().into_owned(),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(&args, &model_dir, Some(&store), "test-host")
            .expect("corrupt auto profile should fall back");

        assert!(loaded.is_none());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
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
