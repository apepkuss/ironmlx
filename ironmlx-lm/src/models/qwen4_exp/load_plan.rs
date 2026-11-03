//! Metadata-only admission. The 30 GiB PLE table is always read on demand;
//! routed experts are either resident or served by a separately bounded cache.
use super::{
    config::FlashTextConfig,
    text_model::{ExpertPagingOptions, FlashTextModel},
};
use crate::core::{
    residency::{ExpertResidency, ResidencyPlan, ResidencyRequest},
    tensor_file::ModelTensorFiles,
    Loader,
};
use anyhow::{ensure, Context, Result};
use std::{path::Path, sync::Arc};
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[derive(Clone, Debug)]
pub struct FlashLoadRequest {
    pub usable_bytes: u64,
    pub max_sequences: usize,
    pub max_cache_cap: usize,
    pub prefill_chunk_size: usize,
    pub draft_bytes: u64,
    pub expert_residency: ExpertResidency,
    pub expert_cache_limit_bytes: u64,
}
impl FlashLoadRequest {
    pub fn new(
        usable_bytes: u64,
        max_sequences: usize,
        max_cache_cap: usize,
        prefill_chunk_size: usize,
        draft_bytes: u64,
    ) -> Self {
        Self {
            usable_bytes,
            max_sequences,
            max_cache_cap,
            prefill_chunk_size,
            draft_bytes,
            expert_residency: ExpertResidency::Auto,
            expert_cache_limit_bytes: if usable_bytes <= 24 * GIB {
                4 * GIB
            } else if usable_bytes <= 56 * GIB {
                8 * GIB
            } else {
                16 * GIB
            },
        }
    }
    /// Optional explicit process configuration, shared by App and CLI. Invalid
    /// settings are errors, never a silent return to full-weight loading.
    pub fn with_environment(mut self) -> Result<Self> {
        if let Ok(value) = std::env::var("IRONMLX_FLASH_EXPERT_RESIDENCY") {
            self.expert_residency = match value.as_str() {
                "auto" => ExpertResidency::Auto,
                "full" => ExpertResidency::Full,
                "ssd" => ExpertResidency::Ssd,
                _ => anyhow::bail!("IRONMLX_FLASH_EXPERT_RESIDENCY must be auto, full or ssd"),
            };
        }
        if let Ok(value) = std::env::var("IRONMLX_FLASH_EXPERT_CACHE_MIB") {
            self.expert_cache_limit_bytes = value
                .parse::<u64>()
                .context("invalid Flash expert cache MiB")?
                .checked_mul(MIB)
                .context("Flash cache size overflow")?;
        }
        Ok(self)
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct FlashLoadPlan {
    pub residency: ResidencyPlan,
    pub fixed_weight_bytes: u64,
    pub vision_weight_bytes: u64,
    pub ple_cache_bytes: usize,
    pub ple_staging_bytes: usize,
    pub qsa_scratch_bytes: usize,
    pub paging: Option<ExpertPagingOptions>,
}
impl FlashLoadPlan {
    pub fn inspect(path: &Path, request: &FlashLoadRequest) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(path.join("config.json"))?)?;
        let cfg = FlashTextConfig::from_root(&root)?;
        // Initial serving scope is the qualified Flash-Next recipe. Broader
        // qwen4_exp geometries need independent kernels and memory qualification.
        ensure!(
            cfg.hidden_size == 2560
                && cfg.num_hidden_layers == 48
                && cfg.hc_count == 4
                && cfg.num_attention_heads == 24
                && cfg.num_key_value_heads == 2
                && cfg.head_dim == 256
                && cfg.linear_num_value_heads == 48
                && cfg.linear_key_head_dim == 128
                && cfg.linear_value_head_dim == 128
                && cfg.linear_num_key_heads == 16
                && cfg.linear_conv_kernel_dim == 4
                && cfg.num_experts == 512
                && cfg.num_experts_per_tok == 10
                && cfg.moe_intermediate_size == 640,
            "Flash serving currently requires the qualified Qwen3.8-Flash-Next geometry"
        );
        if let Some(value) = root.get("vision_config").filter(|v| !v.is_null()) {
            let vision: crate::models::qwen3_5::VisionConfig =
                serde_json::from_value(value.clone())?;
            ensure!(
                vision.depth == 27
                    && vision.hidden_size == 1152
                    && vision.num_heads == 16
                    && vision.intermediate_size == 4304
                    && vision.out_hidden_size == 2560
                    && vision.patch_size == 16
                    && vision.temporal_patch_size == 2
                    && vision.spatial_merge_size == 2
                    && vision.in_channels == 3
                    && vision.num_position_embeddings == 2304
                    && vision.deepstack_visual_indexes.is_empty(),
                "Flash serving requires the qualified Qwen3.8 vision geometry"
            );
        }
        let quant = &root["quantization"];
        ensure!(
            quant["bits"] == 4
                && quant["group_size"] == 32
                && quant
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("affine")
                    == "affine",
            "Flash serving requires affine 4-bit group-32 weights"
        );
        ensure!(
            request.max_sequences > 0
                && request.max_cache_cap > 0
                && request.max_cache_cap <= cfg.max_position_embeddings as usize,
            "invalid Flash serving capacity"
        );
        let files = ModelTensorFiles::open(path)?;
        let mut fixed = 0u64;
        let mut experts = 0u64;
        let mut vision = 0u64;
        for (name, info) in files.tensors() {
            if name.starts_with("vision_tower.") || name.starts_with("visual.") {
                vision = vision
                    .checked_add(info.bytes())
                    .context("Flash vision bytes overflow")?;
                continue;
            }
            if name.contains(".ngram_embedding.") || name.contains(".mtp.") {
                continue;
            }
            if name.contains(".mlp.switch_mlp.") {
                experts = experts
                    .checked_add(info.bytes())
                    .context("Flash expert bytes overflow")?;
            } else {
                fixed = fixed
                    .checked_add(info.bytes())
                    .context("Flash resident bytes overflow")?;
            }
        }
        ensure!(
            fixed > 0 && experts > 0,
            "Flash tensor inventory is incomplete"
        );
        let ple_cache = 64 * MIB;
        let chunk = if request.prefill_chunk_size == 0 {
            request.max_cache_cap
        } else {
            request.prefill_chunk_size.min(request.max_cache_cap)
        };
        let scratch = (chunk as u64)
            .checked_mul(request.max_cache_cap as u64)
            .and_then(|n| n.checked_mul(4))
            .context("Flash QSA scratch overflow")?
            .max(64 * MIB);
        // Include immutable old/new KV handles and up to eight recurrent
        // verification snapshots. This conservative envelope is used by both
        // initial load planning and per-request runtime admission.
        let cache_bytes = (request.max_sequences as u64)
            .checked_mul(
                (request.max_cache_cap as u64)
                    .checked_mul(FLASH_CACHE_BYTES_PER_TOKEN)
                    .and_then(|n| n.checked_add(FLASH_FIXED_BYTES_PER_SEQUENCE))
                    .context("Flash cache envelope overflow")?,
            )
            .context("Flash batch memory overflow")?;
        let temporary = scratch
            .checked_add(4 * GIB + 32 * MIB)
            .context("Flash temporary bytes overflow")?;
        let residency = ResidencyRequest {
            mode: request.expert_residency,
            usable_bytes: request.usable_bytes,
            fixed_weight_bytes: fixed
                .checked_add(vision)
                .context("Flash fixed bytes overflow")?,
            expert_weight_bytes: experts,
            minimum_expert_bytes: experts / (48 * 512) * 10 + 10 * 2048,
            expert_cache_limit_bytes: Some(request.expert_cache_limit_bytes),
            ple_cache_bytes: ple_cache * cfg.ple_layer_ids.len() as u64,
            draft_bytes: request.draft_bytes,
            kv_and_recurrent_bytes: cache_bytes,
            temporary_bytes: temporary,
        }
        .plan().context("Flash memory plan rejected: use auto/ssd expert residency or reduce Max Sequences/context capacity; fixed weights, request state, vision/draft and temporary work must still fit")?;
        let paging = (residency.mode == ExpertResidency::Ssd).then_some(ExpertPagingOptions {
            cache_bytes: usize::try_from(residency.resident_expert_bytes)?,
            staging_bytes: 16 * MIB as usize,
            // Already covered by the fixed temporary envelope. Artificially
            // shrinking this cap on 32 GiB would repeatedly repack the same
            // layer without releasing any admission reservation.
            pack_bytes: 2 * GIB as usize,
        });
        Ok(Self {
            residency,
            fixed_weight_bytes: fixed,
            vision_weight_bytes: vision,
            ple_cache_bytes: ple_cache as usize,
            ple_staging_bytes: 16 * MIB as usize,
            qsa_scratch_bytes: usize::try_from(scratch)?,
            paging,
        })
    }
    pub fn open_loader(&self, path: &Path) -> Result<Loader> {
        if self.paging.is_some() {
            Loader::open_flash_text_paged(path, self.fixed_weight_bytes)
        } else {
            Loader::open_flash_text(
                path,
                self.fixed_weight_bytes + self.residency.resident_expert_bytes,
            )
        }
    }
    pub fn build_model(&self, loader: &Loader) -> Result<FlashTextModel> {
        let mut model = FlashTextModel::from_loader_with_paging(
            loader,
            Arc::new(ModelTensorFiles::open(loader.model_dir())?),
            self.ple_cache_bytes,
            self.ple_staging_bytes,
            self.qsa_scratch_bytes,
            self.paging,
        )?;
        model.resident_bytes = model
            .resident_bytes
            .checked_add(usize::try_from(self.vision_weight_bytes)?)
            .and_then(|n| n.checked_add(self.qsa_scratch_bytes))
            .and_then(|n| n.checked_add((4 * GIB + 32 * MIB) as usize))
            .context("Flash runtime reservation overflow")?;
        tracing::info!(mode=?self.residency.mode,expert_cache_bytes=self.residency.resident_expert_bytes,peak_bytes=self.residency.peak_bytes,usable_bytes=self.residency.usable_bytes,"Flash memory plan admitted");
        Ok(model)
    }
}
// Deliberately conservative vs measured persistent cost (28,416 B/token,
// about 116 MB recurrent state). Covers KV copy-on-write and Q8 snapshots.
pub const FLASH_CACHE_BYTES_PER_TOKEN: u64 = 196_608;
pub const FLASH_FIXED_BYTES_PER_SEQUENCE: u64 = 2 * GIB;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn low_memory_defaults_bound_cache_without_relaxing_state_cost() {
        let low = FlashLoadRequest::new(24 * GIB, 1, 4096, 512, 0);
        assert_eq!(low.expert_cache_limit_bytes, 4 * GIB);
        assert_eq!(
            FlashLoadRequest::new(56 * GIB, 2, 8192, 512, 0).expert_cache_limit_bytes,
            8 * GIB
        );
        assert_eq!(
            FlashLoadRequest::new(100 * GIB, 4, 32768, 2048, 0).expert_cache_limit_bytes,
            16 * GIB
        );
        assert_eq!(FLASH_CACHE_BYTES_PER_TOKEN, 196_608);
    }
}
