//! DFlash2 target admission shared by `generate` and `serve`.
//!
//! The draft loop is target-agnostic; what a target may execute is not. This
//! module turns a target checkpoint into the execution scope it is qualified
//! for, so both entry points enforce the same limits.

use std::path::Path;

use anyhow::{anyhow, Result};
use ironmlx_core::m5_profile::M5ProfileTarget;
use ironmlx_lm::models::dflash2::DFlash2Target;
use ironmlx_lm::models::ModelArchitecture;
use ironmlx_runtime::core::dflash2::DFlash2BlockSizeResolution;

/// Default runtime quantization of the BF16 draft for dense targets.
const DENSE_DEFAULT_DRAFT_BITS: i32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DFlash2TargetFamily {
    /// Dense Qwen3.5-family targets (the Qwen3.8 lane and its qualification).
    Qwen35Dense,
    /// Qwen3.6-35B-A3B MoE with an exactly qualified affine recipe. The
    /// execution scope is B1, linear proposals and the original BF16 draft.
    Qwen36Moe { bits: i32 },
}

impl DFlash2TargetFamily {
    pub(crate) fn detect(
        architecture: ModelArchitecture,
        config_raw: &serde_json::Value,
    ) -> Result<Self> {
        match architecture {
            ModelArchitecture::Qwen35Dense => Ok(Self::Qwen35Dense),
            ModelArchitecture::Qwen35Moe => {
                ironmlx_lm::models::qwen36_moe_dflash2_target_bits(config_raw)
                    .map(|bits| Self::Qwen36Moe { bits })
                    .map_err(|error| {
                        anyhow!(
                            "--dflash2-model-dir supports MoE targets only for qualified Qwen3.6-35B-A3B affine 4/5/6/8-bit checkpoints: {error:#}"
                        )
                    })
            }
            _ => Err(anyhow!(
                "--dflash2-model-dir supports dense Qwen3.5 targets and qualified Qwen3.6-35B-A3B MoE targets only"
            )),
        }
    }

    /// M5 DFlash2 profile family of this target. Each family installs only
    /// its own qualified settings.
    pub(crate) fn m5_profile_target(self) -> M5ProfileTarget {
        match self {
            Self::Qwen35Dense => M5ProfileTarget::Qwen35Dense,
            Self::Qwen36Moe { .. } => M5ProfileTarget::Qwen36Moe,
        }
    }

    /// Default flat-tree size when the option is omitted.
    pub(crate) fn default_tree_max_nodes(self, profile_active: bool) -> usize {
        match (self, profile_active) {
            (Self::Qwen35Dense, true) => ironmlx_core::m5_profile::DFLASH2_TREE_MAX_NODES,
            (Self::Qwen36Moe { .. }, true) => ironmlx_core::m5_profile::MOE_DFLASH2_TREE_MAX_NODES,
            (_, false) => 0,
        }
    }

    /// Resolve the draft quantization. Dense targets keep their runtime
    /// 4-bit default; MoE targets are qualified only with the BF16 draft.
    pub(crate) fn resolve_draft_bits(self, requested: Option<i32>) -> Result<i32> {
        if let Some(bits) = requested {
            anyhow::ensure!(
                matches!(bits, 0 | 4 | 8),
                "--dflash2-draft-bits must be one of 0, 4, or 8"
            );
        }
        match self {
            Self::Qwen35Dense => Ok(requested.unwrap_or(DENSE_DEFAULT_DRAFT_BITS)),
            Self::Qwen36Moe { .. } => match requested.unwrap_or(0) {
                0 => Ok(0),
                bits => Err(anyhow!(
                    "Qwen3.6 MoE DFlash2 is qualified only with the original BF16 draft; --dflash2-draft-bits {bits} is not supported (use 0)"
                )),
            },
        }
    }

    /// Reject execution shapes outside the target's qualification.
    pub(crate) fn ensure_execution_scope(
        self,
        tree_max_nodes: Option<usize>,
        position_keyed_sampling: bool,
    ) -> Result<()> {
        if let Self::Qwen36Moe { .. } = self {
            anyhow::ensure!(
                tree_max_nodes.unwrap_or(0)
                    <= ironmlx_lm::models::dflash2::DFlash2DraftTree::MAX_NODES,
                "--dflash2-tree-max-nodes must be at most {}",
                ironmlx_lm::models::dflash2::DFlash2DraftTree::MAX_NODES
            );
            anyhow::ensure!(
                !position_keyed_sampling,
                "Qwen3.6 MoE DFlash2 has not qualified --dflash2-position-keyed-sampling"
            );
        }
        Ok(())
    }

    /// Check a requested flat-tree size against the target's own tree
    /// certification. Dense targets keep their lane-pack rules.
    pub(crate) fn ensure_tree_supported<M: DFlash2Target>(
        self,
        model: &M,
        tree_max_nodes: usize,
    ) -> Result<()> {
        if let Self::Qwen36Moe { bits } = self {
            let certified = model.dflash2_verify_capabilities().flat_tree_max_nodes;
            anyhow::ensure!(
                tree_max_nodes <= certified,
                "--dflash2-tree-max-nodes {tree_max_nodes} exceeds the {certified}-node flat tree certified for this Qwen3.6 MoE affine{bits} target"
            );
        }
        Ok(())
    }

    /// Bound the proposal block by the widest verify the target certifies.
    /// The draft checkpoint width and the target verify width are separate
    /// limits: an automatic block is narrowed to the target, while an
    /// explicit wider block is rejected for qualified-scope targets.
    pub(crate) fn resolve_block_size<M: DFlash2Target>(
        self,
        model: &M,
        requested: Option<usize>,
        checkpoint_block_size: usize,
    ) -> Result<DFlash2BlockSizeResolution> {
        let mut resolution = ironmlx_runtime::core::dflash2::resolve_dflash2_block_size(
            requested,
            checkpoint_block_size,
        )?;
        if let Self::Qwen36Moe { bits } = self {
            let capabilities = model.dflash2_verify_capabilities();
            let max_verify_width = capabilities
                .max_draft_tokens(1)
                .map(|drafts| drafts + 1)
                .unwrap_or(0);
            anyhow::ensure!(
                max_verify_width >= 2,
                "Qwen3.6 MoE affine{bits} target certifies no DFlash2 verify width"
            );
            if resolution.explicit {
                anyhow::ensure!(
                    resolution.block_size <= max_verify_width,
                    "--dflash2-block-size {} exceeds the Q{max_verify_width} verify width certified for this Qwen3.6 MoE affine{bits} target ({})",
                    resolution.block_size,
                    capabilities.profile
                );
            } else {
                resolution.block_size = resolution.block_size.min(max_verify_width);
            }
        }
        Ok(resolution)
    }
}

/// DFlash2 target family of `model_dir`, read before any model is loaded
/// because the M5 profile must be installed first. Unreadable or unsupported
/// targets return `None`; their admission error is reported once the model
/// directory is opened.
pub(crate) fn model_dir_dflash2_family(model_dir: &Path) -> Option<DFlash2TargetFamily> {
    std::fs::File::open(model_dir.join("config.json"))
        .ok()
        .and_then(|file| serde_json::from_reader::<_, serde_json::Value>(file).ok())
        .and_then(|raw| {
            let architecture = ModelArchitecture::from_config_value(&raw).ok()?;
            DFlash2TargetFamily::detect(architecture, &raw).ok()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_scope_requires_bf16_draft_linear_and_stateful_sampling() {
        let moe = DFlash2TargetFamily::Qwen36Moe { bits: 6 };
        assert_eq!(moe.resolve_draft_bits(None).unwrap(), 0);
        assert_eq!(moe.resolve_draft_bits(Some(0)).unwrap(), 0);
        assert!(moe.resolve_draft_bits(Some(4)).is_err());
        assert!(moe.resolve_draft_bits(Some(8)).is_err());
        assert!(moe.ensure_execution_scope(None, false).is_ok());
        assert!(moe.ensure_execution_scope(Some(0), false).is_ok());
        assert!(moe.ensure_execution_scope(Some(15), false).is_ok());
        assert!(moe.ensure_execution_scope(Some(16), false).is_err());
        assert!(moe.ensure_execution_scope(None, true).is_err());
        assert_eq!(moe.m5_profile_target(), M5ProfileTarget::Qwen36Moe);
        assert_eq!(moe.default_tree_max_nodes(false), 0);

        let dense = DFlash2TargetFamily::Qwen35Dense;
        assert_eq!(dense.resolve_draft_bits(None).unwrap(), 4);
        assert_eq!(dense.resolve_draft_bits(Some(0)).unwrap(), 0);
        assert!(dense.resolve_draft_bits(Some(5)).is_err());
        assert!(dense.ensure_execution_scope(Some(15), true).is_ok());
        assert_eq!(dense.m5_profile_target(), M5ProfileTarget::Qwen35Dense);
        assert_eq!(dense.default_tree_max_nodes(true), 15);
    }
}
