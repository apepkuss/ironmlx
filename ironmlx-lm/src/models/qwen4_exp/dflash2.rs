//! Author-qualified HC capture contract and transactional Q=1 verification.
//! The initial path makes no acceleration claim; measured routing may choose
//! ordinary decode until a wider verification kernel has a demonstrated win.
use super::text_model::FlashTextModel;
use crate::{
    core::{
        cache::layer::{LayerCache, LayerCacheSnapshot},
        Model,
    },
    models::dflash2::*,
};
use anyhow::{ensure, Result};
use mlx::{Array, StreamOrDevice};

impl DFlash2Target for FlashTextModel {
    fn dflash2_sequential_verify_cost(&self) -> bool {
        true
    }

    fn dflash2_window_cost_budget_policy(&self) -> bool {
        true
    }

    fn dflash2_validate_draft_config(&self, draft: &DFlash2Config) -> Result<()> {
        draft.ensure_flash_next_compatible(&self.config)
    }
    fn dflash2_supports_images(&self) -> bool {
        self.vision.available()
    }
    fn dflash2_load_vision(&self) -> Result<bool> {
        self.vision.load_deferred()
    }
    fn dflash2_vision_embeds_on(
        &self,
        pixels: &[Array],
        grids: &[(i32, i32, i32)],
        target: StreamOrDevice,
    ) -> Result<Array> {
        crate::core::vision::DenseVlMethods::compute_vision_embeds(self, pixels, grids, target)
    }
    fn dflash2_forward_target_vl_on(
        &self,
        ids: &Array,
        embeds: &Array,
        image_id: i32,
        positions: &Array,
        cache: Option<&mut [LayerCache]>,
        captures: &[usize],
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let embeddings = self.image_inputs_on(ids, Some(embeds), image_id, target)?;
        self.runtime_inputs_on(
            ids,
            Some(positions),
            Some(&embeddings),
            None,
            None,
            cache,
            captures,
            DFlash2TargetForwardMode::Prefill,
            target,
        )
    }
    fn dflash2_target_spec(&self) -> DFlash2TargetSpec {
        (&self.config).into()
    }
    fn dflash2_target_cache_cost(&self) -> DFlash2TargetCacheCost {
        DFlash2TargetCacheCost {
            bytes_per_token: super::load_plan::FLASH_CACHE_BYTES_PER_TOKEN as usize,
            fixed_bytes_per_sequence: super::load_plan::FLASH_FIXED_BYTES_PER_SEQUENCE as usize,
        }
    }
    fn dflash2_verify_capabilities(&self) -> DFlash2VerifyCapabilities {
        DFlash2VerifyCapabilities {
            profile: if self.layer_major_verify {
                "qwen38-flash-next-vl-layer-major-q1-v5"
            } else {
                "qwen38-flash-next-vl-q1-chain-v5"
            }
            .into(),
            row_bit_exact_qmm: true,
            row_bit_exact_attention: true,
            transactional_state_restore: true,
            supported_shapes: [1, 2, 4]
                .into_iter()
                .flat_map(|batch_width| {
                    (2..=8).map(move |verify_width| DFlash2VerifyShape {
                        batch_width,
                        verify_width,
                    })
                })
                .collect(),
            lane_kernel_pack: None,
            flat_tree_max_nodes: 0,
            ragged_rows_only: true,
        }
    }
    fn dflash2_execution_fingerprint(&self) -> String {
        let quant = self.quant.map_or_else(
            || "full".into(),
            |q| format!("{}{}g{}", q.mode.mlx_backend_mode(), q.bits, q.group_size),
        );
        format!("qwen38-flash-next-vl-v5;mlx=a8f66b590;rope=reference-power;quant={quant};capture=completed-hc-mean;ple=bounded-ssd;experts={};dispatch=source-e512;prefill=scheduler-image-tail-v1;verify={};paging-batch={}",
            if self.expert_paging_enabled(){"bounded-ssd"}else{"resident"},if self.layer_major_verify{"layer-major-q1"}else{"token-major-q1"},if self.native_paged_batching && self.expert_paging_enabled(){"native"}else{"row"})
    }
    fn dflash2_instance(&self) -> Option<&DFlash2Instance> {
        Some(&self.dflash_instance)
    }
    fn dflash2_embed_on(&self, ids: &Array, target: StreamOrDevice) -> Result<Array> {
        self.embed_on(ids, target)
    }
    fn dflash2_forward_target_on(
        &self,
        ids: &Array,
        positions: &Array,
        cache: Option<&mut [LayerCache]>,
        captures: &[usize],
        mode: DFlash2TargetForwardMode,
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        self.runtime_inputs_on(
            ids,
            Some(positions),
            None,
            None,
            None,
            cache,
            captures,
            mode,
            target,
        )
    }
    fn dflash2_restore_target_prefix_on(
        &self,
        cache: &mut [LayerCache],
        base: &[LayerCacheSnapshot],
        accepted: usize,
        target: StreamOrDevice,
    ) -> Result<()> {
        let rows = match cache.first() {
            Some(LayerCache::Flash(layer)) => layer.offsets().len(),
            _ => anyhow::bail!("expected Flash cache"),
        };
        self.dflash2_restore_target_prefix_rows_on(cache, base, &vec![accepted; rows], target)
    }
    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        base: &[LayerCacheSnapshot],
        accepted: &[usize],
        target: StreamOrDevice,
    ) -> Result<()> {
        ensure!(
            cache.len() == base.len() && cache.len() == self.config.num_hidden_layers,
            "Flash commit layer count mismatch"
        );
        for (cache, base) in cache.iter().zip(base) {
            match (cache, base) {
                (LayerCache::Flash(cache), LayerCacheSnapshot::Flash(base)) => {
                    cache.validate_prefix_rows(base, accepted)?
                }
                _ => anyhow::bail!("Flash commit received foreign cache/snapshot"),
            }
        }
        for (cache, base) in cache.iter_mut().zip(base) {
            if let (LayerCache::Flash(cache), LayerCacheSnapshot::Flash(base)) = (cache, base) {
                cache.commit_prefix_rows(base, accepted, target)?;
            }
        }
        Ok(())
    }
    fn dflash2_project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        let _stable = crate::nn::position_stable_qmm::scope();
        self.project_hidden_on(hidden, target)
    }
}
