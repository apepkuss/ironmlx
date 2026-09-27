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

use crate::core::cache::layer::LayerCache;
use crate::core::Loader;
use crate::nn::Linear;
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
pub struct DFlash2TargetOutput {
    pub hidden: Array,
    pub context_hidden: Array,
}

/// Resident target-cache cost charged for one request-local DFlash2 stream.
/// Hybrid targets must count only token-growing cache layers and report
/// recurrent/convolution state separately as fixed per-sequence storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2TargetCacheCost {
    pub bytes_per_token: usize,
    pub fixed_bytes_per_sequence: usize,
}

/// One target verification shape whose row-wise logits and state transitions
/// are certified against ordinary Q=1 decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DFlash2VerifyShape {
    pub batch_width: usize,
    pub verify_width: usize,
}

/// A model-family-specific execution pack for wide DFlash2 verification.
///
/// The pack is deliberately descriptive rather than a global feature flag:
/// callers can only select it through a target's certified capability
/// matrix.  This keeps the ordinary Q=1 path independent while making the
/// prepared quantized layout, attention route, transactional cache contract,
/// and graph-submission cadence part of the execution fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlash2LaneKernelPack {
    pub family: String,
    pub revision: u32,
    pub quant_bits: i32,
    pub quant_group_size: i32,
    pub max_lanes: usize,
    pub prepared_layout: String,
    pub attention_layout: String,
    pub state_layout: String,
    pub layer_submit_interval: usize,
}

impl DFlash2LaneKernelPack {
    pub fn supports(&self, batch_width: usize, verify_width: usize) -> bool {
        batch_width > 0
            && verify_width > 1
            && batch_width
                .checked_mul(verify_width)
                .is_some_and(|lanes| lanes <= self.max_lanes)
    }

    pub fn stable_fingerprint(&self) -> String {
        format!(
            "family={};revision={};quant=affine{}g{};max-lanes={};weights={};attention={};state={};submit={}",
            self.family,
            self.revision,
            self.quant_bits,
            self.quant_group_size,
            self.max_lanes,
            self.prepared_layout,
            self.attention_layout,
            self.state_layout,
            self.layer_submit_interval,
        )
    }
}

/// Target-side contract consumed by the DFlash2 scheduler. Keeping the
/// qualification matrix next to the target implementation prevents the actor
/// from inferring numerical safety from a configured block size alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlash2VerifyCapabilities {
    pub profile: String,
    pub row_bit_exact_qmm: bool,
    pub row_bit_exact_attention: bool,
    pub transactional_state_restore: bool,
    pub supported_shapes: Vec<DFlash2VerifyShape>,
    pub lane_kernel_pack: Option<DFlash2LaneKernelPack>,
}

impl DFlash2VerifyCapabilities {
    pub fn supports(&self, batch_width: usize, verify_width: usize) -> bool {
        verify_width == 1
            || self.supported_shapes.contains(&DFlash2VerifyShape {
                batch_width,
                verify_width,
            })
    }

    pub fn stable_fingerprint(&self) -> String {
        let mut supported_shapes = self.supported_shapes.clone();
        supported_shapes.sort_by_key(|shape| (shape.batch_width, shape.verify_width));
        let shapes = supported_shapes
            .iter()
            .map(|shape| format!("b{}q{}", shape.batch_width, shape.verify_width))
            .collect::<Vec<_>>()
            .join(",");
        let lane_pack = self
            .lane_kernel_pack
            .as_ref()
            .map(DFlash2LaneKernelPack::stable_fingerprint)
            .unwrap_or_else(|| "none".to_owned());
        format!(
            "profile={};qmm={};attention={};state={};shapes={shapes};lane-pack={lane_pack}",
            self.profile,
            self.row_bit_exact_qmm,
            self.row_bit_exact_attention,
            self.transactional_state_restore
        )
    }

    pub fn max_draft_tokens(&self, batch_width: usize) -> Option<usize> {
        self.supported_shapes
            .iter()
            .filter(|shape| shape.batch_width == batch_width)
            .map(|shape| shape.verify_width.saturating_sub(1))
            .max()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlash2VerifyExecution {
    OrdinaryDecode,
    Speculative,
}

/// Explicit execution plan for one DFlash2 target window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2VerifyPlan {
    pub execution: DFlash2VerifyExecution,
    pub batch_width: usize,
    pub draft_tokens: usize,
    pub verify_width: usize,
}

impl DFlash2VerifyPlan {
    pub fn build(
        capabilities: &DFlash2VerifyCapabilities,
        batch_width: usize,
        draft_tokens: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            batch_width > 0,
            "DFlash2 verify batch width must be positive"
        );
        let verify_width = draft_tokens.saturating_add(1);
        if draft_tokens > 0 {
            anyhow::ensure!(
                capabilities.row_bit_exact_qmm
                    && capabilities.row_bit_exact_attention
                    && capabilities.transactional_state_restore,
                "DFlash2 verify profile {} does not provide the full row-exact execution contract",
                capabilities.profile
            );
            if let Some(pack) = capabilities.lane_kernel_pack.as_ref() {
                anyhow::ensure!(
                    pack.supports(batch_width, verify_width),
                    "DFlash2 lane pack {} revision {} cannot execute B{batch_width}/Q{verify_width}",
                    pack.family,
                    pack.revision,
                );
            }
        }
        anyhow::ensure!(
            capabilities.supports(batch_width, verify_width),
            "DFlash2 verify profile {} does not certify B{batch_width}/Q{verify_width}",
            capabilities.profile
        );
        Ok(Self {
            execution: if draft_tokens == 0 {
                DFlash2VerifyExecution::OrdinaryDecode
            } else {
                DFlash2VerifyExecution::Speculative
            },
            batch_width,
            draft_tokens,
            verify_width,
        })
    }
}

impl DFlash2TargetCacheCost {
    pub fn request_bytes(self, token_cap: usize) -> usize {
        token_cap
            .saturating_mul(self.bytes_per_token)
            .saturating_add(self.fixed_bytes_per_sequence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlash2TargetForwardMode {
    /// Prompt ingestion; no speculative verify routing is active.
    Prefill,
    /// One-token target-only decode used as the measured control path when
    /// speculative drafting is not currently profitable.
    OrdinaryDecode,
    /// Qualified batched greedy verify used by the P3 execution path.
    GreedyVerify,
    /// Position-stable target logits required by exact speculative sampling.
    SampledVerify,
}

impl DFlash2TargetForwardMode {
    pub(crate) fn is_verify(self) -> bool {
        matches!(self, Self::GreedyVerify | Self::SampledVerify)
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
pub trait DFlash2Target: crate::core::Model {
    fn dflash2_target_cache_cost(&self) -> DFlash2TargetCacheCost;

    fn dflash2_verify_capabilities(&self) -> DFlash2VerifyCapabilities;

    fn dflash2_execution_fingerprint(&self) -> String;

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
        snapshots: &[crate::core::cache::layer::LayerCacheSnapshot],
        accepted_len: usize,
        target: StreamOrDevice,
    ) -> Result<()>;

    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::core::cache::layer::LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: StreamOrDevice,
    ) -> Result<()>;

    fn dflash2_project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array>;
}

pub use config::{DFlash2Parameters, DFlash2RopeParameters};

#[cfg(test)]
mod tests {
    use super::{
        DFlash2TargetForwardMode, DFlash2VerifyCapabilities, DFlash2VerifyExecution,
        DFlash2VerifyPlan, DFlash2VerifyShape,
    };

    #[test]
    fn every_verify_mode_requires_position_stability() {
        assert!(!DFlash2TargetForwardMode::Prefill.is_verify());
        assert!(!DFlash2TargetForwardMode::Prefill.requires_position_stability());
        assert!(!DFlash2TargetForwardMode::OrdinaryDecode.is_verify());
        assert!(!DFlash2TargetForwardMode::OrdinaryDecode.requires_position_stability());

        for mode in [
            DFlash2TargetForwardMode::GreedyVerify,
            DFlash2TargetForwardMode::SampledVerify,
        ] {
            assert!(mode.is_verify());
            assert!(mode.requires_position_stability());
        }
    }

    #[test]
    fn verify_plan_requires_a_certified_shape_but_always_allows_q1() {
        let capabilities = DFlash2VerifyCapabilities {
            profile: "test".into(),
            row_bit_exact_qmm: true,
            row_bit_exact_attention: true,
            transactional_state_restore: true,
            supported_shapes: vec![DFlash2VerifyShape {
                batch_width: 2,
                verify_width: 4,
            }],
            lane_kernel_pack: None,
        };
        let ordinary = DFlash2VerifyPlan::build(&capabilities, 7, 0).unwrap();
        assert_eq!(ordinary.execution, DFlash2VerifyExecution::OrdinaryDecode);
        assert_eq!(ordinary.verify_width, 1);
        assert!(DFlash2VerifyPlan::build(&capabilities, 2, 3).is_ok());
        assert!(DFlash2VerifyPlan::build(&capabilities, 2, 4).is_err());
        assert_eq!(
            capabilities.stable_fingerprint(),
            "profile=test;qmm=true;attention=true;state=true;shapes=b2q4;lane-pack=none"
        );

        let mut incomplete = capabilities;
        incomplete.transactional_state_restore = false;
        assert!(DFlash2VerifyPlan::build(&incomplete, 2, 3).is_err());
        assert!(DFlash2VerifyPlan::build(&incomplete, 7, 0).is_ok());
        assert_eq!(incomplete.max_draft_tokens(2), Some(3));
        assert_eq!(incomplete.max_draft_tokens(1), None);
    }
}
