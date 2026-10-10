use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::models::{Qwen35Config, Qwen35MoeConfig};
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2Parameters {
    pub block_size: i32,
    pub conv_group_size: i32,
    pub conv_kernel_size: i32,
    pub mask_token_id: u32,
    pub selector_rank: i32,
    pub selector_top_k: i32,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2RopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

/// Exact configuration contract for the official DFlash2 draft checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2Config {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub dtype: String,
    pub hidden_act: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub is_causal: bool,
    pub head_dim: i32,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: i32,
    pub model_type: String,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub num_target_layers: i32,
    pub rms_norm_eps: f32,
    pub rope_parameters: DFlash2RopeParameters,
    pub sliding_window: i32,
    pub vocab_size: i32,
    pub dflash_config: DFlash2Parameters,
}

/// Target fields a DFlash2 draft depends on, independent of whether the
/// target is a dense or a mixture-of-experts model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DFlash2TargetSpec {
    pub hidden_size: i32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub num_hidden_layers: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl From<&Qwen35Config> for DFlash2TargetSpec {
    fn from(cfg: &Qwen35Config) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            max_position_embeddings: cfg.max_position_embeddings,
            num_hidden_layers: cfg.num_hidden_layers,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_parameters.rope_theta,
        }
    }
}

impl From<&Qwen35MoeConfig> for DFlash2TargetSpec {
    fn from(cfg: &Qwen35MoeConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            max_position_embeddings: cfg.max_position_embeddings,
            num_hidden_layers: cfg.num_hidden_layers,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_parameters.rope_theta,
        }
    }
}

impl DFlash2Config {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let config: Self = serde_json::from_value(loader.config_raw_value().clone())
            .context("deserializing DFlash2 config.json")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.architectures.as_slice() != ["DFlash2DraftModel"] {
            return Err(anyhow!(
                "DFlash2 architectures must be exactly [DFlash2DraftModel], got {:?}",
                self.architectures
            ));
        }
        if self.model_type != "qwen3" {
            return Err(anyhow!(
                "DFlash2 model_type must be qwen3, got {}",
                self.model_type
            ));
        }
        if self.dtype != "bfloat16" {
            return Err(anyhow!(
                "DFlash2 draft dtype must be bfloat16, got {}",
                self.dtype
            ));
        }
        if self.hidden_act != "silu" {
            return Err(anyhow!(
                "DFlash2 hidden_act must be silu, got {}",
                self.hidden_act
            ));
        }
        if self.attention_bias {
            return Err(anyhow!("DFlash2 attention_bias=true is unsupported"));
        }
        if self.is_causal {
            return Err(anyhow!(
                "DFlash2 first execution path requires non-causal block attention"
            ));
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("head_dim", self.head_dim),
            ("num_attention_heads", self.num_attention_heads),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_key_value_heads", self.num_key_value_heads),
            ("num_target_layers", self.num_target_layers),
            ("max_position_embeddings", self.max_position_embeddings),
            ("sliding_window", self.sliding_window),
            ("vocab_size", self.vocab_size),
            ("block_size", self.dflash_config.block_size),
            ("conv_group_size", self.dflash_config.conv_group_size),
            ("conv_kernel_size", self.dflash_config.conv_kernel_size),
            ("selector_rank", self.dflash_config.selector_rank),
            ("selector_top_k", self.dflash_config.selector_top_k),
        ] {
            if value <= 0 {
                return Err(anyhow!("DFlash2 {name} must be positive, got {value}"));
            }
        }
        if self.num_attention_heads * self.head_dim <= 0
            || self.num_key_value_heads > self.num_attention_heads
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(anyhow!(
                "DFlash2 invalid GQA heads: heads={} kv_heads={} head_dim={}",
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim
            ));
        }
        if self.hidden_size % self.dflash_config.conv_group_size != 0 {
            return Err(anyhow!(
                "DFlash2 conv_group_size {} must divide hidden_size {}",
                self.dflash_config.conv_group_size,
                self.hidden_size
            ));
        }
        if self.dflash_config.block_size < 2 || self.dflash_config.block_size > self.sliding_window
        {
            return Err(anyhow!(
                "DFlash2 block_size {} must be in [2, sliding_window={}]",
                self.dflash_config.block_size,
                self.sliding_window
            ));
        }
        if self.dflash_config.selector_top_k > self.vocab_size {
            return Err(anyhow!(
                "DFlash2 selector_top_k {} exceeds vocab_size {}",
                self.dflash_config.selector_top_k,
                self.vocab_size
            ));
        }
        if self.dflash_config.mask_token_id >= self.vocab_size as u32 {
            return Err(anyhow!(
                "DFlash2 mask_token_id {} exceeds vocab_size {}",
                self.dflash_config.mask_token_id,
                self.vocab_size
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers as usize
            || self
                .layer_types
                .iter()
                .any(|layer_type| layer_type != "sliding_attention")
        {
            return Err(anyhow!(
                "DFlash2 first execution path requires one sliding_attention entry per draft layer"
            ));
        }
        // The context projection concatenates every tapped target layer, so
        // the tap count is independent of the draft depth (for example eight
        // taps feeding six draft layers).
        if self.dflash_config.target_layer_ids.is_empty() {
            return Err(anyhow!("DFlash2 target_layer_ids must not be empty"));
        }
        let mut previous = None;
        for &layer in &self.dflash_config.target_layer_ids {
            if layer >= self.num_target_layers as usize {
                return Err(anyhow!(
                    "DFlash2 target layer {layer} is outside target layer count {}",
                    self.num_target_layers
                ));
            }
            if previous.is_some_and(|prior| layer <= prior) {
                return Err(anyhow!(
                    "DFlash2 target_layer_ids must be strictly increasing"
                ));
            }
            previous = Some(layer);
        }
        if self.rope_parameters.rope_type != "default"
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
        {
            return Err(anyhow!(
                "DFlash2 requires finite positive default RoPE parameters"
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(anyhow!("DFlash2 rms_norm_eps must be finite and positive"));
        }
        Ok(())
    }

    /// Width of the `fc` context projection input: one hidden vector per
    /// tapped target layer.
    pub fn context_width(&self) -> Result<i32> {
        i32::try_from(self.dflash_config.target_layer_ids.len())
            .ok()
            .and_then(|taps| taps.checked_mul(self.hidden_size))
            .ok_or_else(|| anyhow!("DFlash2 context width overflows i32"))
    }

    /// Compare only the fields the draft algorithm consumes from the target:
    /// the shared hidden space (embedding and LM head), vocabulary, position
    /// range, tapped layer range, final-norm epsilon and RoPE base. Target FFN
    /// widths are deliberately not compared: a dense draft MLP has no relation
    /// to a target's dense or expert FFN width.
    pub fn ensure_target_compatible(&self, target: &DFlash2TargetSpec) -> Result<()> {
        macro_rules! check_eq {
            ($field:ident) => {
                if self.$field != target.$field {
                    return Err(anyhow!(
                        "DFlash2 target {} mismatch: draft={} target={}",
                        stringify!($field),
                        self.$field,
                        target.$field
                    ));
                }
            };
        }
        check_eq!(hidden_size);
        check_eq!(vocab_size);
        check_eq!(max_position_embeddings);
        if self.num_target_layers != target.num_hidden_layers {
            return Err(anyhow!(
                "DFlash2 target layer count mismatch: draft={} target={}",
                self.num_target_layers,
                target.num_hidden_layers
            ));
        }
        if (self.rms_norm_eps - target.rms_norm_eps).abs() > f32::EPSILON {
            return Err(anyhow!(
                "DFlash2 target rms_norm_eps mismatch: draft={} target={}",
                self.rms_norm_eps,
                target.rms_norm_eps
            ));
        }
        if (self.rope_parameters.rope_theta - target.rope_theta).abs() > f32::EPSILON {
            return Err(anyhow!(
                "DFlash2 target rope_theta mismatch: draft={} target={}",
                self.rope_parameters.rope_theta,
                target.rope_theta
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_config() -> DFlash2Config {
        serde_json::from_value(serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "attention_bias": false,
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "is_causal": false,
            "head_dim": 128,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "max_position_embeddings": 262144,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 64,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "sliding_window": 2048,
            "vocab_size": 248320,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]
            }
        }))
        .expect("parse official config")
    }

    #[test]
    fn official_qwen38_dflash2_contract_is_accepted() {
        official_config()
            .validate()
            .expect("validate official config");
    }

    fn qwen36_moe_config() -> DFlash2Config {
        let mut cfg = official_config();
        cfg.hidden_size = 2048;
        cfg.intermediate_size = 6144;
        cfg.num_hidden_layers = 6;
        cfg.num_target_layers = 40;
        cfg.layer_types = vec!["sliding_attention".to_owned(); 6];
        cfg.dflash_config.mask_token_id = 248077;
        cfg.dflash_config.target_layer_ids = vec![1, 6, 11, 16, 22, 27, 32, 37];
        cfg
    }

    #[test]
    fn tap_count_is_independent_of_draft_depth() {
        let cfg = qwen36_moe_config();
        cfg.validate().expect("eight taps feeding six draft layers");
        assert_eq!(cfg.context_width().unwrap(), 8 * 2048);

        let mut empty = qwen36_moe_config();
        empty.dflash_config.target_layer_ids.clear();
        assert!(empty.validate().is_err());
        let mut out_of_range = qwen36_moe_config();
        out_of_range.dflash_config.target_layer_ids[7] = 40;
        assert!(out_of_range.validate().is_err());
        let mut unordered = qwen36_moe_config();
        unordered.dflash_config.target_layer_ids.swap(0, 1);
        assert!(unordered.validate().is_err());
    }

    #[test]
    fn target_compatibility_ignores_ffn_widths() {
        let cfg = qwen36_moe_config();
        let target = DFlash2TargetSpec {
            hidden_size: 2048,
            vocab_size: 248320,
            max_position_embeddings: 262144,
            num_hidden_layers: 40,
            rms_norm_eps: 0.000001,
            rope_theta: 10_000_000.0,
        };
        cfg.ensure_target_compatible(&target)
            .expect("MoE target without a dense intermediate size");
        for mismatch in [
            DFlash2TargetSpec {
                hidden_size: 4096,
                ..target
            },
            DFlash2TargetSpec {
                vocab_size: 151_936,
                ..target
            },
            DFlash2TargetSpec {
                num_hidden_layers: 48,
                ..target
            },
            DFlash2TargetSpec {
                max_position_embeddings: 131_072,
                ..target
            },
            DFlash2TargetSpec {
                rms_norm_eps: 0.00001,
                ..target
            },
            DFlash2TargetSpec {
                rope_theta: 1_000_000.0,
                ..target
            },
        ] {
            assert!(cfg.ensure_target_compatible(&mismatch).is_err());
        }
    }

    #[test]
    fn legacy_or_causal_draft_is_rejected() {
        let mut cfg = official_config();
        cfg.architectures = vec!["DFlashDraftModel".to_owned()];
        assert!(cfg.validate().is_err());

        let mut cfg = official_config();
        cfg.is_causal = true;
        assert!(cfg.validate().is_err());
    }
}
