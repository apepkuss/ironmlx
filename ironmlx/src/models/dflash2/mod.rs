//! DFlash2 auxiliary draft model.
//!
//! This module is intentionally independent from the existing MTP and
//! scheduler execution paths. It implements only the official
//! `DFlash2DraftModel` checkpoint contract used by the standalone DFlash2 CLI
//! engine.

mod attention;
mod config;
mod conv;
mod layer;
mod model;
mod selector;

pub use config::DFlash2Config;
pub use model::{DFlash2DraftCache, DFlash2DraftModel};

use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{LayerCache, Linear};
use crate::Result;

const DFLASH2_DRAFT_QUANT_GROUP_SIZE: i32 = 64;

fn load_linear(loader: &Loader, prefix: &str, draft_bits: Option<i32>) -> Result<Linear> {
    let Some(bits) = draft_bits else {
        return Linear::from_loader(loader, prefix);
    };
    if !matches!(bits, 4 | 8) {
        anyhow::bail!("DFlash2 runtime draft quantization supports only 4 or 8 bits");
    }
    let weight = loader.tensor(&format!("{prefix}.weight"))?;
    let bias = loader.tensor_opt(&format!("{prefix}.bias")).cloned();
    let quantized = mlx::quantization::quantize(
        weight,
        Some(DFLASH2_DRAFT_QUANT_GROUP_SIZE),
        Some(bits),
        "affine",
        None,
    )?;
    if quantized.len() != 3 {
        anyhow::bail!(
            "DFlash2 affine quantization for {prefix} returned {} tensors, expected 3",
            quantized.len()
        );
    }
    mlx::transforms::eval(&[&quantized[0], &quantized[1], &quantized[2]])?;
    Ok(Linear::new_quant(
        quantized[0].clone(),
        quantized[1].clone(),
        Some(quantized[2].clone()),
        bias,
        DFLASH2_DRAFT_QUANT_GROUP_SIZE,
        bits,
    ))
}

/// Target-model output required by one DFlash2 draft/verify cycle.
pub(crate) struct DFlash2TargetOutput {
    pub(crate) hidden: Array,
    pub(crate) context_hidden: Array,
}

/// Resident target-cache cost charged for one request-local DFlash2 stream.
/// Hybrid targets must count only token-growing cache layers and report
/// recurrent/convolution state separately as fixed per-sequence storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DFlash2TargetCacheCost {
    pub(crate) bytes_per_token: usize,
    pub(crate) fixed_bytes_per_sequence: usize,
}

impl DFlash2TargetCacheCost {
    pub(crate) fn request_bytes(self, token_cap: usize) -> usize {
        token_cap
            .saturating_mul(self.bytes_per_token)
            .saturating_add(self.fixed_bytes_per_sequence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DFlash2TargetForwardMode {
    /// Prompt ingestion; no speculative verify routing is active.
    Prefill,
    /// Qualified batched greedy verify used by the P3 execution path.
    GreedyVerify,
    /// Position-stable target logits required by exact speculative sampling.
    SampledVerify,
}

impl DFlash2TargetForwardMode {
    pub(crate) fn is_verify(self) -> bool {
        self != Self::Prefill
    }

    pub(crate) fn requires_position_stability(self) -> bool {
        self.is_verify()
    }
}

/// Narrow target capability required by DFlash2.
///
/// The trait is deliberately separate from `MtpSpeculativeModel`: DFlash2
/// captures several target layers and owns a different draft cache and
/// proposal distribution.
pub(crate) trait DFlash2Target: crate::core::Model {
    fn dflash2_target_cache_cost(&self) -> DFlash2TargetCacheCost;

    fn dflash2_embed_on(&self, input_ids: &Array, target: StreamOrDevice) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn dflash2_forward_target_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target_layer_ids: &[usize],
        mode: DFlash2TargetForwardMode,
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput>;

    fn dflash2_restore_target_prefix_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_len: usize,
        target: StreamOrDevice,
    ) -> Result<()>;

    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: StreamOrDevice,
    ) -> Result<()>;

    fn dflash2_project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array>;
}

#[cfg(test)]
mod tests {
    use super::DFlash2TargetForwardMode;

    #[test]
    fn every_verify_mode_requires_position_stability() {
        assert!(!DFlash2TargetForwardMode::Prefill.is_verify());
        assert!(!DFlash2TargetForwardMode::Prefill.requires_position_stability());

        for mode in [
            DFlash2TargetForwardMode::GreedyVerify,
            DFlash2TargetForwardMode::SampledVerify,
        ] {
            assert!(mode.is_verify());
            assert!(mode.requires_position_stability());
        }
    }
}
