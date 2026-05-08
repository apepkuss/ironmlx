//! Top-level Qwen3.5 model: text model + (tied or explicit) lm_head + heterogeneous cache.

use anyhow::Context;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::Loader;
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::Qwen35Config;
use super::text_model::Qwen35TextModel;

pub struct Qwen35Model {
    text: Qwen35TextModel,
    /// `Some` when `!tie_word_embeddings`. `None` reuses `text.embed_tokens` for output projection.
    lm_head: Option<Linear>,
}

impl Qwen35Model {
    /// Production constructor. Calls [`Qwen35Config::from_loader`] then
    /// [`Qwen35TextModel::from_loader`]; loads `lm_head` only when not tied.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Qwen35Config::from_loader(loader)
            .context("parsing Qwen35Config from loader.config_raw_value")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35Config) -> Result<Self> {
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(Linear::from_loader(loader, "lm_head")?)
        };
        let text = Qwen35TextModel::from_loader(loader, cfg)?;
        Ok(Self { text, lm_head })
    }

    /// Test seam.
    #[doc(hidden)]
    pub fn from_components(text: Qwen35TextModel, lm_head: Option<Linear>) -> Self {
        Self { text, lm_head }
    }

    pub fn config(&self) -> &Qwen35Config {
        self.text.config()
    }

    pub fn text(&self) -> &Qwen35TextModel {
        &self.text
    }

    /// Forward to logits `[B, S, vocab_size]`.
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self
            .text
            .forward_on(input_ids, position_ids, cache, target)?;
        match &self.lm_head {
            Some(head) => head.forward_on(&hidden, target),
            None => self.text.as_output_on(&hidden, target),
        }
    }

    /// Construct a per-layer cache list matching this model's hybrid topology.
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let head_dim = cfg.effective_head_dim();
        let mut out = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_kind(i) {
                AttnKind::Full => {
                    out.push(LayerCache::Full(KVCache::new(
                        batch,
                        cfg.num_key_value_heads,
                        head_dim,
                        head_dim,
                        dtype,
                        cap,
                    )));
                }
                AttnKind::Linear => {
                    let conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                        + cfg.linear_value_head_dim * cfg.linear_num_value_heads;
                    out.push(LayerCache::Linear(GatedDeltaCache::new_with_cap(
                        batch,
                        cfg.linear_conv_kernel_dim,
                        conv_dim,
                        cfg.linear_num_value_heads,
                        cfg.linear_value_head_dim,
                        cfg.linear_key_head_dim,
                        dtype,
                        cap,
                    )?));
                }
            }
        }
        Ok(out)
    }

    /// Test-only stub: constructs a Qwen35Model whose `text` field is unsuitable
    /// for forward (the layers vec is empty, embeddings are stubs) but whose
    /// `make_cache` is fully driven by `cfg`. Used only by tests in this
    /// module to verify cache-partition behavior without synthesizing weights.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn from_cfg_for_test(cfg: Qwen35Config) -> Self {
        let mrope = crate::nn::Mrope::new(
            cfg.effective_head_dim(),
            cfg.rope_parameters.rope_theta,
            cfg.rope_parameters.partial_rotary_factor,
            &cfg.rope_parameters.mrope_section,
            true,
        )
        .expect("Mrope::new with valid cfg");
        let h = cfg.hidden_size;
        let stub_embed = crate::nn::Embedding::from_components_fp_for_test(
            mlx::Array::zeros((cfg.vocab_size, h), mlx::Dtype::Bfloat16).unwrap(),
        );
        let stub_norm = crate::nn::RmsNorm::new(
            mlx::ops::constructors::ones((h,), mlx::Dtype::Float32).unwrap(),
            cfg.rms_norm_eps,
        );
        let text = Qwen35TextModel::from_components(stub_embed, Vec::new(), stub_norm, mrope, cfg);
        Self {
            text,
            lm_head: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::AttnKind;
    use mlx::Dtype;

    fn make_cfg() -> Qwen35Config {
        // 4 layers, full_attention_interval=2 → layers {1, 3} are Full.
        Qwen35Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_parameters: super::super::config::RopeParams {
                partial_rotary_factor: 1.0,
                rope_theta: 1e7,
                mrope_section: vec![2, 1, 1],
            },
        }
    }

    #[test]
    fn make_cache_layer_kinds_match_partition() {
        let cfg = make_cfg();
        // Verify partition logic on the config alone first.
        assert_eq!(cfg.layer_kind(0), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(1), AttnKind::Full);
        assert_eq!(cfg.layer_kind(2), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(3), AttnKind::Full);

        let model = Qwen35Model::from_cfg_for_test(cfg);
        let cache = model
            .make_cache(/* batch */ 1, /* cap */ 16, Dtype::Bfloat16)
            .unwrap();
        assert_eq!(cache.len(), 4);
        assert!(
            matches!(cache[0], LayerCache::Linear(_)),
            "layer 0 should be Linear"
        );
        assert!(
            matches!(cache[1], LayerCache::Full(_)),
            "layer 1 should be Full"
        );
        assert!(
            matches!(cache[2], LayerCache::Linear(_)),
            "layer 2 should be Linear"
        );
        assert!(
            matches!(cache[3], LayerCache::Full(_)),
            "layer 3 should be Full"
        );
    }
}
