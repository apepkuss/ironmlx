use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::models::Qwen35Config;
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
        if self.dflash_config.target_layer_ids.len() != self.num_hidden_layers as usize {
            return Err(anyhow!(
                "DFlash2 target_layer_ids count {} must equal draft layer count {}",
                self.dflash_config.target_layer_ids.len(),
                self.num_hidden_layers
            ));
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

    pub fn ensure_target_compatible(&self, target: &Qwen35Config) -> Result<()> {
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
        check_eq!(intermediate_size);
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
        if (self.rope_parameters.rope_theta - target.rope_parameters.rope_theta).abs()
            > f32::EPSILON
        {
            return Err(anyhow!(
                "DFlash2 target rope_theta mismatch: draft={} target={}",
                self.rope_parameters.rope_theta,
                target.rope_parameters.rope_theta
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
