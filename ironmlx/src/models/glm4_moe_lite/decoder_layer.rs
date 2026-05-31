//! GLM-4.7-Flash (`glm4_moe_lite`) decoder layer.
//!
//! Mirrors `glm4_moe_lite.py:293-317` (`Glm4MoeLiteDecoderLayer`):
//! `input_layernorm → MlaAttention → residual → post_attention_layernorm →
//! FFN → residual`. The FFN is the dense [`Mlp`] for layer 0 (`first_k_dense_replace`)
//! and the noaux_tc [`Glm4MoeBlock`] for layers `>= first_k_dense_replace`.
//!
//! p5h profiling spans mirror `qwen3_5_moe/decoder_layer.rs` under the
//! `p5h-profile` feature only; default builds keep the plain hot path.

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Mlp, RmsNorm};

use super::config::Glm4MoeLiteConfig;
use super::mla_attention::MlaAttention;
use super::mla_cache::MlaLatentCache;
use super::moe::{Glm4MoeBlock, GlmMoeBlockMode};
use super::rope::RopeOffset;

#[cfg(feature = "p5h-profile")]
fn p5h_layer_fields(layer_idx: i32) -> crate::core::p5h::SpanFields {
    crate::core::p5h::SpanFields {
        layer_idx: Some(layer_idx),
        ..Default::default()
    }
}

/// Feed-forward sub-block: dense SwiGLU MLP (layer 0) or the MoE router block.
///
/// `Glm4MoeBlock` is large (stacked expert weights), so it is boxed to keep the
/// enum (and the `Vec<Glm4DecoderLayer>` element stride) small.
enum Ffn {
    Dense(Mlp),
    Moe(Box<Glm4MoeBlock>),
}

/// One GLM-4.7-Flash transformer decoder layer.
pub struct Glm4DecoderLayer {
    input_layernorm: RmsNorm,
    attn: MlaAttention,
    post_attention_layernorm: RmsNorm,
    ffn: Ffn,
}

/// Diagnostic block execution mode used by full-forward attribution benches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderBlockMode {
    Full,
    SkipAttention,
    SkipAttentionRoutedOnly,
    SkipAttentionRoutedFixedOnly,
    SkipAttentionSharedOnly,
    SkipFfn,
}

impl Glm4DecoderLayer {
    /// Load all submodules at `model.layers.{layer_idx}.{name}`.
    pub fn from_loader(loader: &Loader, layer_idx: i32, cfg: &Glm4MoeLiteConfig) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}");
        let input_layernorm =
            RmsNorm::from_loader(loader, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?;
        let attn = MlaAttention::from_loader(loader, &format!("{p}.self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{p}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let ffn = if cfg.is_moe_layer(layer_idx) {
            Ffn::Moe(Box::new(Glm4MoeBlock::from_loader(
                loader,
                &format!("{p}.mlp"),
                cfg,
            )?))
        } else {
            Ffn::Dense(Mlp::from_loader(loader, &format!("{p}.mlp"))?)
        };
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            ffn,
        })
    }

    /// Pre-norm transformer block: `h = x + attn(norm(x)); out = h + ffn(norm(h))`.
    ///
    /// `offset` is the per-row `[B]` i32 RoPE start position; `cache` is this
    /// layer's latent MLA cache; `per_row_lens` is the number of new tokens
    /// written this step; `mask` is the engine's additive float attention mask
    /// (`None` for decode, or for prefill where the model derives the causal
    /// mask itself).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        offset: &Array,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        self.forward_with_rope_offset(
            x,
            RopeOffset::PerRow(offset),
            cache,
            per_row_lens,
            mask,
            target,
            layer_idx,
            DecoderBlockMode::Full,
        )
    }

    /// Diagnostic variant of [`Self::forward_on`] with sub-block skipping.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on_with_block_mode(
        &self,
        x: &Array,
        offset: &Array,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
        block_mode: DecoderBlockMode,
    ) -> Result<Array> {
        self.forward_with_rope_offset(
            x,
            RopeOffset::PerRow(offset),
            cache,
            per_row_lens,
            mask,
            target,
            layer_idx,
            block_mode,
        )
    }

    /// B=1 fast path using scalar RoPE offset, matching mlx-lm's `cache.offset`
    /// call shape while preserving the per-row array path for batched rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on_scalar_offset(
        &self,
        x: &Array,
        offset: i32,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        self.forward_with_rope_offset(
            x,
            RopeOffset::Scalar(offset),
            cache,
            per_row_lens,
            mask,
            target,
            layer_idx,
            DecoderBlockMode::Full,
        )
    }

    /// Diagnostic variant of [`Self::forward_on_scalar_offset`] with sub-block skipping.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on_scalar_offset_with_block_mode(
        &self,
        x: &Array,
        offset: i32,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
        block_mode: DecoderBlockMode,
    ) -> Result<Array> {
        self.forward_with_rope_offset(
            x,
            RopeOffset::Scalar(offset),
            cache,
            per_row_lens,
            mask,
            target,
            layer_idx,
            block_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_rope_offset(
        &self,
        x: &Array,
        offset: RopeOffset<'_>,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
        block_mode: DecoderBlockMode,
    ) -> Result<Array> {
        let target = target.into();
        let skip_attention_moe_mode = match block_mode {
            DecoderBlockMode::SkipAttentionRoutedOnly => Some(GlmMoeBlockMode::RoutedOnly),
            DecoderBlockMode::SkipAttentionRoutedFixedOnly => {
                Some(GlmMoeBlockMode::RoutedFixedOnly)
            }
            DecoderBlockMode::SkipAttentionSharedOnly => Some(GlmMoeBlockMode::SharedOnly),
            DecoderBlockMode::SkipAttention => Some(GlmMoeBlockMode::Full),
            _ => None,
        };
        #[cfg(feature = "p5h-profile")]
        {
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "decoder_layer_N",
                || p5h_layer_fields(layer_idx),
                || -> Result<Array> {
                    if let Some(moe_mode) = skip_attention_moe_mode {
                        let normed_post = crate::core::p5h::try_with_p5h_span_from_current_trace(
                            "post_attention_norm",
                            || p5h_layer_fields(layer_idx),
                            || self.post_attention_layernorm.forward_on(x, target),
                        )?;
                        let ffn_out = crate::core::p5h::try_with_p5h_span_from_current_trace(
                            "mlp_path",
                            || p5h_layer_fields(layer_idx),
                            || match &self.ffn {
                                Ffn::Dense(m) => m.forward_on(&normed_post, target),
                                Ffn::Moe(b) => b.forward_on_with_mode(
                                    &normed_post,
                                    target,
                                    layer_idx,
                                    moe_mode,
                                ),
                            },
                        )?;
                        return crate::core::p5h::try_with_p5h_span_from_current_trace(
                            "residual_overhead",
                            || p5h_layer_fields(layer_idx),
                            || -> Result<Array> {
                                Ok(mlx::ops::binary::add_on(x, &ffn_out, target)?)
                            },
                        );
                    }

                    let normed_in = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "input_norm",
                        || p5h_layer_fields(layer_idx),
                        || self.input_layernorm.forward_on(x, target),
                    )?;
                    let attn = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "attention_path",
                        || p5h_layer_fields(layer_idx),
                        || {
                            self.attn.forward_with_rope_offset(
                                &normed_in,
                                offset,
                                cache,
                                per_row_lens,
                                mask,
                                target,
                                layer_idx,
                            )
                        },
                    )?;
                    let h = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "residual_overhead",
                        || p5h_layer_fields(layer_idx),
                        || mlx::ops::binary::add_on(x, &attn, target),
                    )?;

                    if block_mode == DecoderBlockMode::SkipFfn {
                        return Ok(h);
                    }

                    let normed_post = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "post_attention_norm",
                        || p5h_layer_fields(layer_idx),
                        || self.post_attention_layernorm.forward_on(&h, target),
                    )?;
                    let ffn_out = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "mlp_path",
                        || p5h_layer_fields(layer_idx),
                        || match &self.ffn {
                            Ffn::Dense(m) => m.forward_on(&normed_post, target),
                            Ffn::Moe(b) => b.forward_on_with_mode(
                                &normed_post,
                                target,
                                layer_idx,
                                GlmMoeBlockMode::Full,
                            ),
                        },
                    )?;
                    crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "residual_overhead",
                        || p5h_layer_fields(layer_idx),
                        || -> Result<Array> { Ok(mlx::ops::binary::add_on(&h, &ffn_out, target)?) },
                    )
                },
            )
        }

        #[cfg(not(feature = "p5h-profile"))]
        {
            if let Some(moe_mode) = skip_attention_moe_mode {
                let normed_post = self.post_attention_layernorm.forward_on(x, target)?;
                let ffn_out = match &self.ffn {
                    Ffn::Dense(m) => m.forward_on(&normed_post, target)?,
                    Ffn::Moe(b) => {
                        b.forward_on_with_mode(&normed_post, target, layer_idx, moe_mode)?
                    }
                };
                return Ok(mlx::ops::binary::add_on(x, &ffn_out, target)?);
            }

            let normed_in = self.input_layernorm.forward_on(x, target)?;
            let attn = self.attn.forward_with_rope_offset(
                &normed_in,
                offset,
                cache,
                per_row_lens,
                mask,
                target,
                layer_idx,
            )?;
            let h = mlx::ops::binary::add_on(x, &attn, target)?;

            if block_mode == DecoderBlockMode::SkipFfn {
                return Ok(h);
            }

            let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
            let ffn_out = match &self.ffn {
                Ffn::Dense(m) => m.forward_on(&normed_post, target)?,
                Ffn::Moe(b) => {
                    b.forward_on_with_mode(&normed_post, target, layer_idx, GlmMoeBlockMode::Full)?
                }
            };
            Ok(mlx::ops::binary::add_on(&h, &ffn_out, target)?)
        }
    }
}
