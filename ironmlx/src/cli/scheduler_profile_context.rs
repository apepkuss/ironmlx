use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use clap::Args;

use super::KvQuantArg;
use crate::Result;
use ironmlx_runtime::core::cache::prefix_store::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE;
use ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext;

const BYTES_PER_GIB: usize = 1024 * 1024 * 1024;

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
            let defaults = ironmlx_runtime::core::prompt_lookup::PromptLookupConfig::default();
            Some(
                ironmlx_runtime::core::prompt_lookup::PromptLookupConfig {
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

pub(crate) use {
    ironmlx_runtime::core::scheduler_profile_context::build_scheduler_runtime_context,
    ironmlx_runtime::core::scheduler_profile_context::SchedulerProfileContextOptions,
};

#[cfg(test)]
mod tests {
    use super::*;
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
}
