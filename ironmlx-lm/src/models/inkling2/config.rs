//! Inkling 2's looped MoE and vision configuration.
use anyhow::{ensure, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2Config {
    pub model_type: String,
    pub text_config: TextConfig,
    pub vision_config: Option<VisionConfig>,
    #[serde(default = "image_token")]
    pub image_token_id: i32,
}
fn image_token() -> i32 {
    200_054
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: i32,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub vocab_size: i32,
    pub unpadded_vocab_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub swa_num_attention_heads: i32,
    pub swa_num_key_value_heads: i32,
    pub swa_head_dim: i32,
    pub sliding_window_size: i32,
    pub local_layer_ids: Vec<usize>,
    pub d_rel: i32,
    pub rel_extent: i32,
    pub log_scaling_n_floor: i32,
    pub log_scaling_alpha: f32,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub sconv_kernel_size: i32,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: i32,
    pub num_experts_per_tok: i32,
    pub intermediate_size: i32,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub final_logit_softcapping: f32,
    pub loop_config: Option<LoopConfig>,
    #[serde(default = "context_length")]
    pub model_max_length: i32,
}
fn context_length() -> i32 {
    1_048_576
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub native_patch_size: i32,
    pub adapter_merge_size: i32,
    pub posemb_grid_size: i32,
    pub projection_hidden_dim: i32,
    pub mlp_ratio: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
}
impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 23,
            num_attention_heads: 16,
            native_patch_size: 20,
            adapter_merge_size: 2,
            posemb_grid_size: 32,
            projection_hidden_dim: 4096,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}
impl Inkling2Config {
    pub fn from_value(value: &serde_json::Value) -> Result<Self> {
        let config: Self = serde_json::from_value(value.clone())?;
        ensure!(
            matches!(config.model_type.as_str(), "inkling2" | "inkling2_loop"),
            "expected Inkling 2 configuration"
        );
        config.text_config.validate()?;
        if let Some(v) = &config.vision_config {
            ensure!(
                v.hidden_size > 0
                    && v.num_attention_heads > 0
                    && v.hidden_size % v.num_attention_heads == 0
                    && (v.hidden_size / v.num_attention_heads) % 4 == 0
                    && v.num_hidden_layers > 0
                    && v.native_patch_size > 0
                    && v.adapter_merge_size > 0
                    && v.posemb_grid_size > 0
                    && v.projection_hidden_dim > 0
                    && v.mlp_ratio > 0.0
                    && v.layer_norm_eps > 0.0
                    && v.rope_theta > 0.0,
                "invalid Inkling 2 vision configuration"
            );
        }
        ensure!(
            (0..config.text_config.vocab_size).contains(&config.image_token_id),
            "Inkling 2 image token is outside vocabulary"
        );
        Ok(config)
    }
    pub fn from_loader(loader: &crate::core::Loader) -> Result<Self> {
        Self::from_value(loader.config_raw_value())
    }
}
impl TextConfig {
    pub fn is_local(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }
    pub fn attention_dims(&self, layer: usize) -> (i32, i32, i32) {
        if self.is_local(layer) {
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        } else {
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
        }
    }
    pub fn visits(&self) -> Vec<usize> {
        match &self.loop_config {
            Some(c) => (0..c.core_start)
                .chain((0..c.num_passes).flat_map(|_| c.core_start..c.core_end))
                .chain(c.core_end..self.num_hidden_layers)
                .collect(),
            None => (0..self.num_hidden_layers).collect(),
        }
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size > 0
                && self.num_hidden_layers > 0
                && self.vocab_size > 0
                && self.unpadded_vocab_size > 0
                && self.unpadded_vocab_size <= self.vocab_size
                && self.intermediate_size > 0
                && self.model_max_length > 0,
            "invalid Inkling 2 text dimensions"
        );
        for (h, kv, d) in [
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            ),
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            ),
        ] {
            ensure!(
                h > 0 && kv > 0 && d > 0 && h % kv == 0,
                "invalid Inkling 2 attention dimensions"
            );
        }
        ensure!(
            self.sliding_window_size > 0
                && self.rel_extent > 0
                && self.d_rel > 0
                && self.log_scaling_n_floor > 0
                && self.sconv_kernel_size > 1
                && self.n_layers_per_attn_res_block > 0
                && self.n_routed_experts > 0
                && self.num_experts_per_tok > 0
                && self.num_experts_per_tok <= self.n_routed_experts,
            "invalid Inkling 2 attention, convolution or expert configuration"
        );
        ensure!(
            [
                self.rms_norm_eps,
                self.qk_l2_clip_threshold,
                self.rel_l2_clip_threshold,
                self.mlp_gate_softcap,
                self.mlp_up_softcap,
                self.route_scale,
                self.final_logit_softcapping
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
                && self.log_scaling_alpha.is_finite(),
            "invalid Inkling 2 numerical parameters"
        );
        ensure!(
            self.local_layer_ids
                .iter()
                .all(|i| *i < self.num_hidden_layers),
            "invalid Inkling 2 local layer index"
        );
        if let Some(c) = &self.loop_config {
            ensure!(
                c.core_start > 0
                    && c.core_start < c.core_end
                    && c.core_end <= self.num_hidden_layers
                    && c.num_passes > 0
                    && c.gate_rank > 0
                    && !c.is_varloop
                    && c.core_start % self.n_layers_per_attn_res_block == 0
                    && c.core_end % self.n_layers_per_attn_res_block == 0,
                "Inkling 2 requires fixed loops aligned to attention residual blocks"
            );
        }
        Ok(())
    }
}
