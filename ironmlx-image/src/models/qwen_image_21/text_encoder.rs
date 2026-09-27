use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use ironmlx_lm::core::model_input::build_position_ids_vl;
use ironmlx_lm::models::qwen3_5::cross_modal::replace_image_tokens;
use ironmlx_lm::models::vision::VisionTower;
use ironmlx_lm::nn::{Attention, AttentionConfig, Embedding, Mlp, Mrope, RmsNorm};

use crate::loader::ComponentLoader;
use crate::Result;

use super::config::{vision_config_from_loader, QwenImage21TextConfig};

struct TextDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl TextDecoderLayer {
    fn from_loader(
        loader: &ComponentLoader,
        layer_index: i32,
        config: &QwenImage21TextConfig,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_index}");
        Ok(Self {
            input_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.input_layernorm"),
                config.rms_norm_eps,
            )?,
            self_attn: Attention::from_loader_qwen3_vl(
                loader,
                &format!("{prefix}.self_attn"),
                AttentionConfig {
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    rms_norm_eps: config.rms_norm_eps,
                    has_qk_norm: true,
                },
            )?,
            post_attention_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                config.rms_norm_eps,
            )?,
            mlp: Mlp::from_loader(loader, &format!("{prefix}.mlp"))?,
        })
    }

    fn forward_on(
        &self,
        hidden: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward_qwen3_vl_on(hidden, target)?;
        let attended = self
            .self_attn
            .forward_on(&normed, mrope, cos, sin, None, None, None, None, target)?;
        let residual = mlx::ops::binary::add_on(hidden, &attended, target)?;
        let normed = self
            .post_attention_layernorm
            .forward_qwen3_vl_on(&residual, target)?;
        let mlp = self.mlp.forward_on(&normed, target)?;
        Ok(mlx::ops::binary::add_on(&residual, &mlp, target)?)
    }
}

/// Qwen3-VL encoder used for text and image-conditioned Qwen Image 2.1 prompts.
pub struct QwenImage21TextEncoder {
    embeddings: Embedding,
    layers: Vec<TextDecoderLayer>,
    mrope: Mrope,
    vision: VisionTower,
    config: QwenImage21TextConfig,
}

impl QwenImage21TextEncoder {
    pub const MAX_SEQUENCE_LENGTH: usize = 2048;

    pub fn from_loader(loader: &ComponentLoader) -> Result<Self> {
        let config = QwenImage21TextConfig::from_loader(loader)?;
        let vision_config = vision_config_from_loader(loader)?;
        let embeddings = Embedding::from_loader(loader, "language_model.model.embed_tokens")
            .context("loading Qwen Image 2.1 token embeddings")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(
                TextDecoderLayer::from_loader(loader, layer_index, &config)
                    .with_context(|| format!("loading Qwen Image 2.1 text layer {layer_index}"))?,
            );
        }
        let mrope = Mrope::new(
            config.head_dim,
            config.rope_theta,
            1.0,
            &config.rope_scaling.mrope_section,
            config.rope_scaling.mrope_interleaved,
        )?;
        let vision = VisionTower::from_loader(loader, &vision_config)
            .context("loading Qwen Image 2.1 vision tower")?;
        if vision.deepstack_feature_count() != vision_config.deepstack_visual_indexes.len() {
            return Err(anyhow!(
                "Qwen Image 2.1 vision tower is missing DeepStack merger weights"
            ));
        }
        Ok(Self {
            embeddings,
            layers,
            mrope,
            vision,
            config,
        })
    }

    pub fn config(&self) -> &QwenImage21TextConfig {
        &self.config
    }

    pub fn forward(&self, token_ids: &[u32]) -> Result<Array> {
        self.forward_on(token_ids, ())
    }

    pub fn forward_on(
        &self,
        token_ids: &[u32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        if token_ids.is_empty() || token_ids.len() > Self::MAX_SEQUENCE_LENGTH {
            return Err(anyhow!(
                "Qwen Image 2.1 prompt token count must be in 1..={}, got {}",
                Self::MAX_SEQUENCE_LENGTH,
                token_ids.len()
            ));
        }
        let target = target.into();
        let sequence = i32::try_from(token_ids.len())?;
        let positions = mlx::ops::constructors::arange(0.0, sequence as f64, 1.0, Dtype::Int32)?
            .reshape((1_i32, 1_i32, sequence))?;
        let position_ids = mlx::ops::shape::concatenate(&[&positions, &positions, &positions], 0)?;
        self.forward_embedded_on(token_ids, &position_ids, None, target)
    }

    pub fn forward_conditioned(
        &self,
        token_ids: &[u32],
        pixel_values: &Array,
        grid: (i32, i32, i32),
        image_token_id: i32,
    ) -> Result<Array> {
        if token_ids.is_empty() || token_ids.len() > Self::MAX_SEQUENCE_LENGTH {
            return Err(anyhow!(
                "Qwen Image 2.1 prompt token count must be in 1..={}, got {}",
                Self::MAX_SEQUENCE_LENGTH,
                token_ids.len()
            ));
        }
        let position_token_ids = token_ids
            .iter()
            .map(|token| i32::try_from(*token))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let position_ids = build_position_ids_vl(&position_token_ids, &[grid], image_token_id, 2)?;
        let vision_output = self.vision.forward_with_deepstack(pixel_values, &[grid])?;
        self.forward_embedded_on(
            token_ids,
            &position_ids,
            Some((
                &vision_output.pooled,
                vision_output.deepstack.as_slice(),
                image_token_id,
            )),
            (),
        )
    }

    fn forward_embedded_on(
        &self,
        token_ids: &[u32],
        position_ids: &Array,
        vision: Option<(&Array, &[Array], i32)>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let sequence = i32::try_from(token_ids.len())?;
        let token_array: Array = (token_ids, (1_i32, sequence)).try_into()?;
        let (cos, sin) = self.mrope.cos_sin(position_ids)?;

        let mut hidden = self.embeddings.forward_on(&token_array, target)?;
        if let Some((vision_embeds, _, image_token_id)) = vision {
            hidden = replace_image_tokens(&hidden, &token_array, vision_embeds, image_token_id)?;
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_on(&hidden, &self.mrope, &cos, &sin, target)?;
            if let Some((_, deepstack, image_token_id)) = vision {
                if let Some(visual_embeds) = deepstack.get(layer_index) {
                    let zeros = Array::zeros(hidden.shape(), hidden.dtype())?;
                    let visual_delta =
                        replace_image_tokens(&zeros, &token_array, visual_embeds, image_token_id)?;
                    hidden = mlx::ops::binary::add_on(&hidden, &visual_delta, target)?;
                }
            }
        }
        // Qwen Image 2.1 conditions the diffusion transformer on the last
        // decoder block output before Qwen3-VL's final RMSNorm.
        Ok(hidden)
    }
}
