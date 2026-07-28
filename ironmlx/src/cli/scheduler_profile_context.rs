use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use clap::Args;

use super::KvQuantArg;
use crate::core::cache::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE;
use crate::core::scheduler_autotune::{
    SchedulerActiveKvContext, SchedulerAutotuneRuntimeContext, SchedulerExecutionModel,
    SchedulerKvQuantization, SchedulerPrefixCacheContext, SchedulerSpeculativeContext,
    SchedulerSpeculativeMode, SchedulerWeightQuantizationContext,
};
use crate::models::ModelArchitecture;
use crate::Result;

const BYTES_PER_GIB: usize = 1024 * 1024 * 1024;
const FINGERPRINT_SAMPLE_BYTES: usize = 4096;

#[derive(Args, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SchedulerProfileRuntimeArgs {
    /// Optional local MTP/drafter model directory used during calibration.
    #[arg(long = "mtp-model-dir")]
    pub(crate) mtp_model_dir: Option<PathBuf>,

    /// Explicit MTP draft token count. The model-aware default is used when omitted.
    #[arg(long = "mtp-draft-tokens")]
    pub(crate) mtp_draft_tokens: Option<usize>,

    /// Enable request-local greedy PromptLookup during calibration.
    #[arg(long = "prompt-lookup", default_value_t = false)]
    pub(crate) prompt_lookup: bool,

    #[arg(long = "prompt-lookup-cross-request", default_value_t = false)]
    pub(crate) prompt_lookup_cross_request: bool,

    #[arg(long = "prompt-lookup-min-ngram")]
    pub(crate) prompt_lookup_min_ngram: Option<usize>,

    #[arg(long = "prompt-lookup-max-ngram")]
    pub(crate) prompt_lookup_max_ngram: Option<usize>,

    #[arg(long = "prompt-lookup-max-draft-tokens")]
    pub(crate) prompt_lookup_max_draft_tokens: Option<usize>,

    #[arg(long = "prompt-lookup-history-window-tokens")]
    pub(crate) prompt_lookup_history_window_tokens: Option<usize>,

    #[arg(long = "prompt-lookup-max-index-entries")]
    pub(crate) prompt_lookup_max_index_entries: Option<usize>,

    /// KV cache quantization used by the calibrated server.
    #[arg(long = "kv-quant", value_enum, default_value = "none")]
    pub(crate) kv_quant: KvQuantArg,

    /// Enable paged prefix cache during calibration.
    #[arg(long = "paged-prefix-cache-dir")]
    pub(crate) paged_prefix_cache_dir: Option<PathBuf>,

    /// Tokens per paged prefix cache block.
    #[arg(long = "paged-prefix-cache-block-size", default_value_t = DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE)]
    pub(crate) paged_prefix_cache_block_size: i32,

    /// Maximum paged prefix cache pages.
    #[arg(long = "paged-prefix-cache-max-pages")]
    pub(crate) paged_prefix_cache_max_pages: Option<i32>,

    /// In-process prefix LRU byte limit.
    #[arg(long = "prefix-lru-cache-max-bytes")]
    pub(crate) prefix_lru_cache_max_bytes: Option<usize>,

    /// SSD prefix cache size limit in GiB.
    #[arg(long = "ssd-prefix-cache-max-gb")]
    pub(crate) ssd_prefix_cache_max_gb: Option<usize>,

    /// Enable Active KV offload during calibration.
    #[arg(long = "active-kv-offload", default_value_t = false)]
    pub(crate) active_kv_offload: bool,

    /// Total engine-pool memory limit in GiB.
    #[arg(long = "memory-limit-total-gb")]
    pub(crate) memory_limit_total_gb: Option<usize>,

    /// Per-model engine-pool memory limit in GiB.
    #[arg(long = "memory-limit-model-gb")]
    pub(crate) memory_limit_model_gb: Option<usize>,

    /// Logical KV capacity policy used for every candidate.
    #[arg(long = "max-cache-cap", default_value_t = 32768)]
    pub(crate) max_cache_cap: usize,
}

impl Default for SchedulerProfileRuntimeArgs {
    fn default() -> Self {
        Self {
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: false,
            prompt_lookup_cross_request: false,
            prompt_lookup_min_ngram: None,
            prompt_lookup_max_ngram: None,
            prompt_lookup_max_draft_tokens: None,
            prompt_lookup_history_window_tokens: None,
            prompt_lookup_max_index_entries: None,
            kv_quant: KvQuantArg::None,
            paged_prefix_cache_dir: None,
            paged_prefix_cache_block_size: DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
            paged_prefix_cache_max_pages: None,
            prefix_lru_cache_max_bytes: None,
            ssd_prefix_cache_max_gb: None,
            active_kv_offload: false,
            memory_limit_total_gb: None,
            memory_limit_model_gb: None,
            max_cache_cap: 32768,
        }
    }
}

pub(crate) struct SchedulerProfileContextOptions<'a> {
    pub(crate) mtp_model_dir: Option<&'a Path>,
    pub(crate) mtp_draft_tokens: Option<usize>,
    pub(crate) prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    pub(crate) kv_quantization: SchedulerKvQuantization,
    pub(crate) paged_prefix_cache_enabled: bool,
    pub(crate) paged_prefix_cache_block_size: i32,
    pub(crate) paged_prefix_cache_max_pages: Option<i32>,
    pub(crate) prefix_lru_cache_max_bytes: Option<usize>,
    pub(crate) ssd_prefix_cache_max_bytes: Option<usize>,
    pub(crate) active_kv_offload: bool,
    pub(crate) logical_kv_cap_tokens: usize,
    pub(crate) memory_limit_total_bytes: Option<usize>,
    pub(crate) memory_limit_model_bytes: Option<usize>,
}

impl SchedulerProfileRuntimeArgs {
    pub(crate) fn context_for_model(
        &self,
        model_dir: &Path,
    ) -> Result<SchedulerAutotuneRuntimeContext> {
        validate_positive_i32(
            self.paged_prefix_cache_block_size,
            "--paged-prefix-cache-block-size",
        )?;
        if let Some(max_pages) = self.paged_prefix_cache_max_pages {
            validate_positive_i32(max_pages, "--paged-prefix-cache-max-pages")?;
        }
        if self.mtp_draft_tokens.is_some() && self.mtp_model_dir.is_none() {
            bail!("--mtp-draft-tokens requires --mtp-model-dir");
        }
        if self.mtp_draft_tokens == Some(0) {
            bail!("--mtp-draft-tokens must be > 0");
        }
        let has_prompt_lookup_params = self.prompt_lookup_min_ngram.is_some()
            || self.prompt_lookup_max_ngram.is_some()
            || self.prompt_lookup_max_draft_tokens.is_some()
            || self.prompt_lookup_history_window_tokens.is_some()
            || self.prompt_lookup_max_index_entries.is_some()
            || self.prompt_lookup_cross_request;
        if !self.prompt_lookup && has_prompt_lookup_params {
            bail!("prompt lookup source parameters require --prompt-lookup");
        }
        let prompt_lookup = if self.prompt_lookup {
            let defaults = crate::core::prompt_lookup::PromptLookupConfig::default();
            Some(
                crate::core::prompt_lookup::PromptLookupConfig {
                    min_ngram: self.prompt_lookup_min_ngram.unwrap_or(defaults.min_ngram),
                    max_ngram: self.prompt_lookup_max_ngram.unwrap_or(defaults.max_ngram),
                    max_draft_tokens: self
                        .prompt_lookup_max_draft_tokens
                        .unwrap_or(defaults.max_draft_tokens),
                    history_window_tokens: self
                        .prompt_lookup_history_window_tokens
                        .unwrap_or(defaults.history_window_tokens),
                    max_index_entries: self
                        .prompt_lookup_max_index_entries
                        .unwrap_or(defaults.max_index_entries),
                    cross_request: self.prompt_lookup_cross_request,
                }
                .validate()?,
            )
        } else {
            None
        };
        if self.prefix_lru_cache_max_bytes.is_some() && self.paged_prefix_cache_dir.is_none() {
            bail!("--prefix-lru-cache-max-bytes requires --paged-prefix-cache-dir");
        }
        if self.prefix_lru_cache_max_bytes == Some(0) {
            bail!("--prefix-lru-cache-max-bytes must be > 0");
        }
        if self.ssd_prefix_cache_max_gb == Some(0) {
            bail!("--ssd-prefix-cache-max-gb must be > 0");
        }
        build_scheduler_runtime_context(
            model_dir,
            SchedulerProfileContextOptions {
                mtp_model_dir: self.mtp_model_dir.as_deref(),
                mtp_draft_tokens: self.mtp_draft_tokens,
                prompt_lookup,
                kv_quantization: self.kv_quant.profile_context(),
                paged_prefix_cache_enabled: self.paged_prefix_cache_dir.is_some(),
                paged_prefix_cache_block_size: self.paged_prefix_cache_block_size,
                paged_prefix_cache_max_pages: self.paged_prefix_cache_max_pages,
                prefix_lru_cache_max_bytes: self.prefix_lru_cache_max_bytes,
                ssd_prefix_cache_max_bytes: gib_to_bytes(self.ssd_prefix_cache_max_gb)?,
                active_kv_offload: self.active_kv_offload,
                logical_kv_cap_tokens: self.max_cache_cap,
                memory_limit_total_bytes: gib_to_bytes(self.memory_limit_total_gb)?,
                memory_limit_model_bytes: gib_to_bytes(self.memory_limit_model_gb)?,
            },
        )
    }
}

pub(crate) fn build_scheduler_runtime_context(
    model_dir: &Path,
    options: SchedulerProfileContextOptions<'_>,
) -> Result<SchedulerAutotuneRuntimeContext> {
    let config_path = model_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    let architecture = ModelArchitecture::from_config_value(&config)?;
    let model_revision_fingerprint = model_fingerprint(model_dir, &config_bytes)?;
    let weight_quantization = weight_quantization_context(model_dir, &config)?;

    let speculative = match (options.mtp_model_dir, options.prompt_lookup) {
        (Some(mtp_model_dir), prompt_lookup) => {
            if !mtp_model_dir.is_dir() {
                bail!(
                    "MTP model path must point to a local directory: {}",
                    mtp_model_dir.display()
                );
            }
            let mode = match (architecture, prompt_lookup.is_some()) {
                (ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe, false) => {
                    SchedulerSpeculativeMode::QwenMtp
                }
                (ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe, true) => {
                    SchedulerSpeculativeMode::QwenMtpPromptLookup
                }
                (ModelArchitecture::Gemma4, false) => SchedulerSpeculativeMode::Gemma4Drafter,
                (ModelArchitecture::Gemma4, true) => {
                    SchedulerSpeculativeMode::Gemma4DrafterPromptLookup
                }
                _ => bail!("scheduler profile MTP context supports Qwen and Gemma4 only"),
            };
            let draft_config =
                std::fs::read(mtp_model_dir.join("config.json")).with_context(|| {
                    format!("reading {}", mtp_model_dir.join("config.json").display())
                })?;
            let neural_fingerprint = model_fingerprint(mtp_model_dir, &draft_config)?;
            let source_fingerprint = prompt_lookup.map_or(neural_fingerprint.clone(), |config| {
                format!(
                    "neural={neural_fingerprint};lookup=min={};max={};draft={};history={};entries={};cross_request={}",
                    config.min_ngram,
                    config.max_ngram,
                    config.max_draft_tokens,
                    config.history_window_tokens,
                    config.max_index_entries,
                    config.cross_request
                )
            });
            let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
                &config,
                options
                    .mtp_draft_tokens
                    .map(crate::core::speculative::MtpDraftTokensArg::Explicit)
                    .unwrap_or(crate::core::speculative::MtpDraftTokensArg::Omitted),
            );
            SchedulerSpeculativeContext {
                mode,
                source_fingerprint: Some(source_fingerprint),
                draft_tokens: Some(draft_tokens),
            }
        }
        (None, Some(config)) => SchedulerSpeculativeContext {
            mode: SchedulerSpeculativeMode::PromptLookup,
            source_fingerprint: Some(format!(
                "min={};max={};draft={};history={};entries={};cross_request={}",
                config.min_ngram,
                config.max_ngram,
                config.max_draft_tokens,
                config.history_window_tokens,
                config.max_index_entries,
                config.cross_request
            )),
            draft_tokens: Some(config.max_draft_tokens),
        },
        (None, None) => SchedulerSpeculativeContext {
            mode: SchedulerSpeculativeMode::Disabled,
            source_fingerprint: None,
            draft_tokens: None,
        },
    };

    let block_size = usize::try_from(options.paged_prefix_cache_block_size)
        .context("paged prefix cache block size must be positive")?;
    let resident_cap_tokens = if options.active_kv_offload && options.paged_prefix_cache_enabled {
        let hot_window_pages =
            usize::try_from(crate::core::scheduler::default_active_kv_hot_window_pages(
                options.paged_prefix_cache_block_size,
            ))
            .context("active KV hot window pages must be positive")?;
        Some(
            hot_window_pages
                .saturating_mul(block_size)
                .min(options.logical_kv_cap_tokens.max(1)),
        )
    } else {
        None
    };

    let prefix_cache = if options.paged_prefix_cache_enabled {
        SchedulerPrefixCacheContext {
            enabled: true,
            block_size: Some(block_size),
            max_pages: options
                .paged_prefix_cache_max_pages
                .map(usize::try_from)
                .transpose()
                .context("paged prefix cache max pages must be positive")?,
            lru_max_bytes: options.prefix_lru_cache_max_bytes,
            ssd_max_bytes: options.ssd_prefix_cache_max_bytes,
        }
    } else {
        SchedulerPrefixCacheContext {
            enabled: false,
            block_size: None,
            max_pages: None,
            lru_max_bytes: None,
            ssd_max_bytes: None,
        }
    };

    Ok(SchedulerAutotuneRuntimeContext {
        execution_model: SchedulerExecutionModel::RollingV1,
        model_architecture: architecture.model_type().to_string(),
        model_fingerprint: model_revision_fingerprint,
        weight_quantization,
        speculative,
        kv_quantization: options.kv_quantization,
        prefix_cache,
        active_kv: SchedulerActiveKvContext {
            enabled: options.active_kv_offload,
            resident_cap_tokens,
        },
        logical_kv_cap_tokens: options.logical_kv_cap_tokens,
        memory_limit_total_bytes: options.memory_limit_total_bytes,
        memory_limit_model_bytes: options.memory_limit_model_bytes,
    })
}

fn weight_quantization_context(
    model_dir: &Path,
    config: &serde_json::Value,
) -> Result<SchedulerWeightQuantizationContext> {
    let quantization = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"));
    let optiq_path = model_dir.join("optiq_metadata.json");
    let optiq_bytes = optiq_path
        .exists()
        .then(|| std::fs::read(&optiq_path))
        .transpose()
        .with_context(|| format!("reading {}", optiq_path.display()))?;
    let mode = if optiq_bytes.is_some() {
        "optiq".to_string()
    } else if let Some(quantization) = quantization {
        quantization
            .get("mode")
            .or_else(|| quantization.get("method"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("affine")
            .to_string()
    } else {
        "bf16".to_string()
    };
    let mut fingerprint = StableHasher::new();
    fingerprint.update(mode.as_bytes());
    if let Some(quantization) = quantization {
        fingerprint.update(&serde_json::to_vec(quantization)?);
    }
    if let Some(optiq_bytes) = optiq_bytes {
        fingerprint.update(&optiq_bytes);
    }
    Ok(SchedulerWeightQuantizationContext {
        mode,
        fingerprint: fingerprint.finish(),
    })
}

fn model_fingerprint(model_dir: &Path, config_bytes: &[u8]) -> Result<String> {
    let mut hasher = StableHasher::new();
    hasher.update(config_bytes);
    let mut weight_files = std::fs::read_dir(model_dir)
        .with_context(|| format!("reading model directory {}", model_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    weight_files.sort();
    if weight_files.is_empty() {
        hasher.update(b"no-safetensors");
    }
    for path in weight_files {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .context("model weight file name must be UTF-8")?;
        let mut file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("reading metadata for {}", path.display()))?
            .len();
        hasher.update(name.as_bytes());
        hasher.update(&len.to_le_bytes());
        let mut sample = vec![0_u8; FINGERPRINT_SAMPLE_BYTES.min(len as usize)];
        if !sample.is_empty() {
            file.read_exact(&mut sample)
                .with_context(|| format!("reading fingerprint head from {}", path.display()))?;
            hasher.update(&sample);
        }
        if len > FINGERPRINT_SAMPLE_BYTES as u64 {
            file.seek(SeekFrom::End(-(FINGERPRINT_SAMPLE_BYTES as i64)))
                .with_context(|| format!("seeking fingerprint tail in {}", path.display()))?;
            let mut tail = vec![0_u8; FINGERPRINT_SAMPLE_BYTES];
            file.read_exact(&mut tail)
                .with_context(|| format!("reading fingerprint tail from {}", path.display()))?;
            hasher.update(&tail);
        }
    }
    Ok(hasher.finish())
}

fn validate_positive_i32(value: i32, flag: &str) -> Result<()> {
    if value <= 0 {
        bail!("{flag} must be > 0");
    }
    Ok(())
}

fn gib_to_bytes(value: Option<usize>) -> Result<Option<usize>> {
    value
        .map(|gib| {
            gib.checked_mul(BYTES_PER_GIB)
                .context("GiB value exceeds usize bytes")
        })
        .transpose()
}

struct StableHasher(u64);

impl StableHasher {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> String {
        format!("{:016x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        build_scheduler_runtime_context, SchedulerProfileContextOptions,
        SchedulerProfileRuntimeArgs, StableHasher,
    };
    use crate::core::scheduler_autotune::{SchedulerKvQuantization, SchedulerSpeculativeMode};

    #[test]
    fn stable_hasher_is_deterministic() {
        let mut left = StableHasher::new();
        left.update(b"model");
        let mut right = StableHasher::new();
        right.update(b"model");
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    fn runtime_context_captures_model_and_serving_factors() {
        let temp_dir = unique_temp_dir("scheduler-profile-context");
        let model_dir = temp_dir.join("model");
        let mtp_dir = temp_dir.join("mtp");
        create_model(
            &model_dir,
            r#"{"model_type":"qwen3_5","quantization":{"mode":"affine","bits":4}}"#,
            b"main-weights-v1",
        );
        create_model(&mtp_dir, r#"{"model_type":"qwen3_5_mtp"}"#, b"mtp-weights");

        let context = build_scheduler_runtime_context(
            &model_dir,
            SchedulerProfileContextOptions {
                mtp_model_dir: Some(&mtp_dir),
                mtp_draft_tokens: Some(3),
                prompt_lookup: None,
                kv_quantization: SchedulerKvQuantization::K3V4,
                paged_prefix_cache_enabled: true,
                paged_prefix_cache_block_size: 256,
                paged_prefix_cache_max_pages: Some(512),
                prefix_lru_cache_max_bytes: Some(1_048_576),
                ssd_prefix_cache_max_bytes: Some(8 * 1024 * 1024 * 1024),
                active_kv_offload: true,
                logical_kv_cap_tokens: 65_536,
                memory_limit_total_bytes: Some(96 * 1024 * 1024 * 1024),
                memory_limit_model_bytes: Some(64 * 1024 * 1024 * 1024),
            },
        )
        .expect("build runtime context");

        assert_eq!(context.model_architecture, "qwen3_5");
        assert_eq!(context.weight_quantization.mode, "affine");
        assert_eq!(context.speculative.mode, SchedulerSpeculativeMode::QwenMtp);
        assert_eq!(context.speculative.draft_tokens, Some(3));
        assert!(context.speculative.source_fingerprint.is_some());
        assert_eq!(context.kv_quantization, SchedulerKvQuantization::K3V4);
        assert_eq!(context.prefix_cache.block_size, Some(256));
        assert_eq!(context.prefix_cache.max_pages, Some(512));
        assert!(context.active_kv.enabled);
        assert!(context.active_kv.resident_cap_tokens.is_some());
        assert_eq!(context.logical_kv_cap_tokens, 65_536);
        assert_eq!(
            context.memory_limit_total_bytes,
            Some(96 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            context.memory_limit_model_bytes,
            Some(64 * 1024 * 1024 * 1024)
        );

        std::fs::write(model_dir.join("model.safetensors"), b"main-weights-v2")
            .expect("replace model weights");
        let changed = build_scheduler_runtime_context(
            &model_dir,
            SchedulerProfileContextOptions {
                mtp_model_dir: Some(&mtp_dir),
                mtp_draft_tokens: Some(3),
                prompt_lookup: None,
                kv_quantization: SchedulerKvQuantization::K3V4,
                paged_prefix_cache_enabled: true,
                paged_prefix_cache_block_size: 256,
                paged_prefix_cache_max_pages: Some(512),
                prefix_lru_cache_max_bytes: Some(1_048_576),
                ssd_prefix_cache_max_bytes: Some(8 * 1024 * 1024 * 1024),
                active_kv_offload: true,
                logical_kv_cap_tokens: 65_536,
                memory_limit_total_bytes: Some(96 * 1024 * 1024 * 1024),
                memory_limit_model_bytes: Some(64 * 1024 * 1024 * 1024),
            },
        )
        .expect("rebuild runtime context");
        assert_ne!(context.model_fingerprint, changed.model_fingerprint);

        let prefix_disabled = build_scheduler_runtime_context(
            &model_dir,
            SchedulerProfileContextOptions {
                mtp_model_dir: None,
                mtp_draft_tokens: None,
                prompt_lookup: None,
                kv_quantization: SchedulerKvQuantization::None,
                paged_prefix_cache_enabled: false,
                paged_prefix_cache_block_size: 256,
                paged_prefix_cache_max_pages: Some(512),
                prefix_lru_cache_max_bytes: None,
                ssd_prefix_cache_max_bytes: Some(8 * 1024 * 1024 * 1024),
                active_kv_offload: false,
                logical_kv_cap_tokens: 32_768,
                memory_limit_total_bytes: None,
                memory_limit_model_bytes: None,
            },
        )
        .expect("build prefix-disabled context");
        assert_eq!(
            prefix_disabled.prefix_cache,
            crate::core::scheduler_autotune::SchedulerPrefixCacheContext {
                enabled: false,
                block_size: None,
                max_pages: None,
                lru_max_bytes: None,
                ssd_max_bytes: None,
            }
        );

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn runtime_context_fingerprints_all_prompt_lookup_parameters() {
        let temp_dir = unique_temp_dir("scheduler-profile-prompt-lookup");
        let model_dir = temp_dir.join("model");
        create_model(
            &model_dir,
            r#"{"model_type":"qwen3_5"}"#,
            b"main-weights-v1",
        );
        let base = crate::core::prompt_lookup::PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 4,
            max_draft_tokens: 4,
            history_window_tokens: 4096,
            max_index_entries: 8192,
            cross_request: false,
        };
        let build = |prompt_lookup| {
            build_scheduler_runtime_context(
                &model_dir,
                SchedulerProfileContextOptions {
                    mtp_model_dir: None,
                    mtp_draft_tokens: None,
                    prompt_lookup: Some(prompt_lookup),
                    kv_quantization: SchedulerKvQuantization::None,
                    paged_prefix_cache_enabled: false,
                    paged_prefix_cache_block_size: 256,
                    paged_prefix_cache_max_pages: None,
                    prefix_lru_cache_max_bytes: None,
                    ssd_prefix_cache_max_bytes: None,
                    active_kv_offload: false,
                    logical_kv_cap_tokens: 32768,
                    memory_limit_total_bytes: None,
                    memory_limit_model_bytes: None,
                },
            )
            .expect("build prompt lookup context")
        };
        let context = build(base);
        assert_eq!(
            context.speculative.mode,
            SchedulerSpeculativeMode::PromptLookup
        );
        assert_eq!(context.speculative.draft_tokens, Some(4));
        let baseline_fingerprint = context
            .speculative
            .source_fingerprint
            .expect("source fingerprint");

        for changed in [
            crate::core::prompt_lookup::PromptLookupConfig {
                min_ngram: 3,
                ..base
            },
            crate::core::prompt_lookup::PromptLookupConfig {
                max_ngram: 5,
                ..base
            },
            crate::core::prompt_lookup::PromptLookupConfig {
                max_draft_tokens: 5,
                ..base
            },
            crate::core::prompt_lookup::PromptLookupConfig {
                history_window_tokens: 8192,
                ..base
            },
            crate::core::prompt_lookup::PromptLookupConfig {
                max_index_entries: 16384,
                ..base
            },
            crate::core::prompt_lookup::PromptLookupConfig {
                cross_request: true,
                ..base
            },
        ] {
            assert_ne!(
                build(changed).speculative.source_fingerprint.as_deref(),
                Some(baseline_fingerprint.as_str())
            );
        }

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn runtime_context_distinguishes_qwen_hybrid_from_neural_only() {
        let temp_dir = unique_temp_dir("scheduler-profile-hybrid");
        let model_dir = temp_dir.join("model");
        let mtp_dir = temp_dir.join("mtp");
        create_model(
            &model_dir,
            r#"{"model_type":"qwen3_5"}"#,
            b"main-weights-v1",
        );
        create_model(&mtp_dir, r#"{"model_type":"qwen3_5_mtp"}"#, b"mtp-weights");
        let build = |prompt_lookup| {
            build_scheduler_runtime_context(
                &model_dir,
                SchedulerProfileContextOptions {
                    mtp_model_dir: Some(&mtp_dir),
                    mtp_draft_tokens: Some(3),
                    prompt_lookup,
                    kv_quantization: SchedulerKvQuantization::None,
                    paged_prefix_cache_enabled: false,
                    paged_prefix_cache_block_size: 256,
                    paged_prefix_cache_max_pages: None,
                    prefix_lru_cache_max_bytes: None,
                    ssd_prefix_cache_max_bytes: None,
                    active_kv_offload: false,
                    logical_kv_cap_tokens: 32768,
                    memory_limit_total_bytes: None,
                    memory_limit_model_bytes: None,
                },
            )
            .expect("build speculative context")
        };
        let neural = build(None);
        let hybrid = build(Some(
            crate::core::prompt_lookup::PromptLookupConfig::default(),
        ));

        assert_eq!(neural.speculative.mode, SchedulerSpeculativeMode::QwenMtp);
        assert_eq!(
            hybrid.speculative.mode,
            SchedulerSpeculativeMode::QwenMtpPromptLookup
        );
        assert_ne!(neural.fingerprint(), hybrid.fingerprint());
        assert!(hybrid
            .speculative
            .source_fingerprint
            .as_deref()
            .is_some_and(|fingerprint| fingerprint.contains(";lookup=")));

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn runtime_args_reject_draft_tokens_without_mtp_model() {
        let args = SchedulerProfileRuntimeArgs {
            mtp_draft_tokens: Some(3),
            ..SchedulerProfileRuntimeArgs::default()
        };

        let error = args
            .context_for_model(Path::new("/missing-model"))
            .expect_err("draft tokens without an MTP model must fail");

        assert!(error.to_string().contains("requires --mtp-model-dir"));
    }

    fn create_model(path: &Path, config: &str, weights: &[u8]) {
        std::fs::create_dir_all(path).expect("create model dir");
        std::fs::write(path.join("config.json"), config).expect("write model config");
        std::fs::write(path.join("model.safetensors"), weights).expect("write model weights");
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }
}
