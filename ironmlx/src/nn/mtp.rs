//! MTP (Multi-Token Prediction) head — the speculative-decoding draft head.
//!
//! Mirrors vllm-mlx's `_MTPModule` (`/Volumes/Dev/vllm-mlx/vllm_mlx/patches/qwen3_5_mtp.py:204-216`):
//!
//! ```text
//! e = pre_fc_norm_embedding(next_token_embeds)
//! h = pre_fc_norm_hidden(hidden_states)
//! x = fc(concat([e, h], axis=-1))           # 2H -> H, no bias
//! for layer in layers:                      # N DecoderLayers, fa-only
//!     x = layer(x, mask=causal, cache=mtp_cache[i])
//! x = norm(x)
//! return x  # caller does tied lm_head: embed_tokens.as_linear(x)
//! ```
//!
//! Caller responsibilities (kept out of this module to preserve isolation):
//! - Embed `next_token_ids` to `next_token_embeds` via the main model's `Embedding`.
//! - Project the returned post-norm hidden state to logits via `Embedding::as_linear`
//!   (when `tie_word_embeddings = true`) or a separate `lm_head` Linear.
//! - Run the speculative-decoding loop (draft / verify / accept / KV rollback).
//!   That layer lands in P8c, not here.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::MtpCache;
use crate::core::Loader;
use crate::nn::{DecoderLayer, DecoderLayerConfig, Linear, Mrope, RmsNorm};
use crate::Result;

/// Configuration for [`Mtp`].
#[derive(Debug, Clone, Copy)]
pub struct MtpConfig {
    pub hidden_size: i32,
    /// Number of MTP DecoderLayers. Qwen3.5 checkpoints ship with `1`.
    pub num_mtp_layers: i32,
    /// Per-layer config (forwarded verbatim to each `DecoderLayer::from_loader`).
    pub layer: DecoderLayerConfig,
}

/// Multi-Token Prediction head.
pub struct Mtp {
    pre_fc_norm_hidden: RmsNorm,
    pre_fc_norm_embedding: RmsNorm,
    fc: Linear, // [2H] -> [H], no bias
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    cfg: MtpConfig,
}

impl Mtp {
    /// Test/composition seam: build an `Mtp` from pre-built sub-modules.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(
        pre_fc_norm_hidden: RmsNorm,
        pre_fc_norm_embedding: RmsNorm,
        fc: Linear,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        cfg: MtpConfig,
    ) -> Self {
        Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        }
    }

    /// Read-only view of the head config.
    pub fn config(&self) -> &MtpConfig {
        &self.cfg
    }

    /// Number of DecoderLayers in this MTP head.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Default-stream forward. See [`forward_on`](Self::forward_on).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
    ) -> Result<Array> {
        self.forward_on(
            hidden_states,
            next_token_embeds,
            mrope,
            cos,
            sin,
            mask,
            mtp_cache,
            (),
        )
    }

    /// Stream-targeted forward.
    ///
    /// Inputs:
    /// - `hidden_states`: post-norm hidden state from the main model, `[B, S, hidden_size]`.
    ///   Caller MUST pass `inner.norm(...)`-applied hidden state, matching mlx-lm's
    ///   `qwen3_5_mtp.py:366` (`return out, normed`).
    /// - `next_token_embeds`: caller-pre-computed embedding of the next-token ids,
    ///   `[B, S, hidden_size]` (typically `embed_tokens(next_token_ids)`).
    /// - `cos`/`sin`: precomputed by [`Mrope::cos_sin`].
    /// - `mask`: forwarded to each [`DecoderLayer`] (currently always-causal in
    ///   [`crate::nn::GatedAttention`]).
    /// - `mtp_cache`: optional KV caches for the `N` MTP layers; if `Some`, must
    ///   satisfy `mtp_cache.num_layers() == self.num_layers()`.
    ///
    /// Output: `[B, S, hidden_size]` — the post-`mtp.norm` hidden state. Caller
    /// projects to logits via tied `Embedding::as_linear` to obtain `[B, S, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mut mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight validation (production-grade stability).
        self.validate_inputs(hidden_states, next_token_embeds, mtp_cache.as_deref())?;

        // Step 1: pre-FC norms.
        let h = self.pre_fc_norm_hidden.forward_on(hidden_states, target)?;
        let e = self
            .pre_fc_norm_embedding
            .forward_on(next_token_embeds, target)?;

        // Step 2: concat([e, h], axis=-1)  →  [B, S, 2H]
        // Order is [e, h] — not [h, e] — to match mlx-lm `qwen3_5_mtp.py:380`.
        let concat = mlx::ops::shape::concatenate_on(&[&e, &h], -1, target)?;

        // Step 3: fc 2H -> H (no bias).
        let mut x = self.fc.forward_on(&concat, target)?;

        // Step 4: feed through N DecoderLayers, each with its own KV cache slot.
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = mtp_cache.as_deref_mut().map(|mc| mc.layer_mut(i));
            x = layer.forward_on(&x, mrope, cos, sin, mask, layer_cache, target)?;
        }

        // Step 5: final norm.
        self.norm.forward_on(&x, target)
    }

    /// Pre-flight validation of input shapes and cache layout. Returns Err on first mismatch.
    fn validate_inputs(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mtp_cache: Option<&MtpCache>,
    ) -> Result<()> {
        if hidden_states.ndim() != 3 || next_token_embeds.ndim() != 3 {
            return Err(anyhow!(
                "Mtp::forward_on: hidden_states and next_token_embeds must be rank-3, \
                 got ranks {}/{}",
                hidden_states.ndim(),
                next_token_embeds.ndim(),
            ));
        }
        let hs = hidden_states.shape();
        let es = next_token_embeds.shape();
        let hs = hs.as_slice();
        let es = es.as_slice();
        if hs != es {
            return Err(anyhow!(
                "Mtp::forward_on: hidden_states {:?} and next_token_embeds {:?} \
                 must have identical shape",
                hs,
                es,
            ));
        }
        if hs[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "Mtp::forward_on: last-axis must equal cfg.hidden_size {}, got {}",
                self.cfg.hidden_size,
                hs[2],
            ));
        }
        if let Some(c) = mtp_cache {
            if c.num_layers() != self.layers.len() {
                return Err(anyhow!(
                    "Mtp::forward_on: mtp_cache.num_layers() = {} but Mtp has {} layers",
                    c.num_layers(),
                    self.layers.len(),
                ));
            }
        }
        Ok(())
    }

    /// Production constructor: load all components from a project [`Loader`].
    ///
    /// Reads (under prefix `mtp.`):
    ///
    /// - `{prefix}.pre_fc_norm_hidden.weight`           `[hidden_size]`
    /// - `{prefix}.pre_fc_norm_embedding.weight`        `[hidden_size]`
    /// - `{prefix}.fc.weight`                           `[hidden_size, 2 * hidden_size]` (no bias)
    /// - `{prefix}.layers.{0..N-1}.{...}`               (per [`DecoderLayer::from_loader`])
    /// - `{prefix}.norm.weight`                         `[hidden_size]`
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: MtpConfig) -> Result<Self> {
        let pre_fc_norm_hidden = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.pre_fc_norm_hidden"),
            cfg.layer.rms_norm_eps,
        )?;
        let pre_fc_norm_embedding = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.pre_fc_norm_embedding"),
            cfg.layer.rms_norm_eps,
        )?;
        let fc = Linear::from_loader(loader, &format!("{prefix}.fc"))?;

        // Validate fc weight shape. Unlike attention projections (where reshape /
        // matmul on downstream consumers will surface mismatches), the fc layer's
        // 2H -> H contract is MTP-specific — a misconfigured weight could silently
        // propagate wrong-rank features into the DecoderLayer chain. Catch it here
        // (production-grade stability — explicit bounds > trust caller).
        let expected_in = (cfg.hidden_size * 2) as usize;
        let expected_out = cfg.hidden_size as usize;
        if fc.in_features() != expected_in || fc.out_features() != expected_out {
            return Err(anyhow!(
                "Mtp.fc weight shape mismatch under prefix '{prefix}.fc': \
                 expected [in={expected_in}, out={expected_out}], got [in={}, out={}]",
                fc.in_features(),
                fc.out_features(),
            ));
        }

        let norm = RmsNorm::from_loader(loader, &format!("{prefix}.norm"), cfg.layer.rms_norm_eps)?;

        let mut layers = Vec::with_capacity(cfg.num_mtp_layers as usize);
        for i in 0..cfg.num_mtp_layers {
            layers.push(DecoderLayer::from_loader(
                loader,
                &format!("{prefix}.layers.{i}"),
                cfg.layer,
            )?);
        }

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    use crate::nn::{GatedAttention, GatedAttentionConfig, Mlp};

    fn rand_w(shape: &[i32], dtype: Dtype) -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin()).collect();
        let arr: Array = (data.as_slice(), shape).try_into().unwrap();
        mlx::ops::cast::astype(&arr, dtype).unwrap()
    }

    fn ones_w(dim: i32) -> Array {
        mlx::ops::constructors::ones((dim,), Dtype::Float32).unwrap()
    }

    fn small_layer_cfg() -> DecoderLayerConfig {
        DecoderLayerConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        }
    }

    fn build_decoder_layer(cfg: DecoderLayerConfig) -> DecoderLayer {
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let o_w = rand_w(
            &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
            Dtype::Bfloat16,
        );
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );
        DecoderLayer::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        )
    }

    fn build_mtp(num_layers: i32) -> Mtp {
        let layer_cfg = small_layer_cfg();
        let cfg = MtpConfig {
            hidden_size: layer_cfg.hidden_size,
            num_mtp_layers: num_layers,
            layer: layer_cfg,
        };
        let h = cfg.hidden_size;
        let layers = (0..num_layers)
            .map(|_| build_decoder_layer(layer_cfg))
            .collect();

        Mtp::from_components(
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            Linear::new_fp(rand_w(&[h, 2 * h], Dtype::Bfloat16), None),
            layers,
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            cfg,
        )
    }

    #[test]
    fn mtp_construction_components() {
        let mtp = build_mtp(1);
        assert_eq!(mtp.num_layers(), 1);
        assert_eq!(mtp.config().num_mtp_layers, 1);
        assert_eq!(mtp.config().hidden_size, 32);
    }

    #[test]
    fn forward_shape_and_dtype() {
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        // Inputs: bf16 hidden + bf16 next-token embeddings.
        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = mtp
            .forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None)
            .unwrap();

        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        // Precise dtype is not asserted (RmsNorm with fp32 weights promotes path);
        // the integration test in Task 4 verifies bit-exact-modulo-tol against Python.
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_validates_shape_mismatch() {
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        // Mismatched S between hidden (S=4) and next_embeds (S=3) → Err.
        let next_embeds = rand_w(&[1, 3, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let r = mtp.forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None);
        let err = r.expect_err("mismatched shapes must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("identical shape"),
            "expected shape-mismatch message, got: {msg}"
        );
    }

    #[test]
    fn forward_validates_cache_layers_mismatch() {
        let mtp = build_mtp(1); // 1 MTP layer
        let cfg = small_layer_cfg();

        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        // Cache with 2 layers, but Mtp has 1 layer → Err.
        let mut wrong_cache = MtpCache::new_with_cap(
            2,
            1,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.head_dim,
            Dtype::Bfloat16,
            16,
        )
        .unwrap();
        let r = mtp.forward(
            &hidden,
            &next_embeds,
            &mrope,
            &cos,
            &sin,
            None,
            Some(&mut wrong_cache),
        );
        let err = r.expect_err("cache num_layers mismatch must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("num_layers") && msg.contains('2') && msg.contains('1'),
            "expected cache-num_layers-mismatch message, got: {msg}"
        );
    }

    #[test]
    fn forward_validates_rank_mismatch() {
        // Rank-2 hidden_states (missing seq dim) → must Err with "rank-3" message.
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        // hidden_states is rank-2 [B*S, H] — invalid.
        let hidden = rand_w(&[4, cfg.hidden_size], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let r = mtp.forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None);
        let err = r.expect_err("rank-2 hidden_states must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("rank-3"),
            "expected rank-3 message, got: {msg}"
        );
    }

    #[test]
    fn forward_validates_hidden_size_mismatch() {
        // Last-axis mismatch (hidden_size in input != cfg.hidden_size) → must Err.
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        // Both inputs share the same wrong last-axis (so the shape-equality check passes
        // and we exercise the hidden_size branch specifically).
        let bad_h = cfg.hidden_size + 4; // 32 + 4 = 36 — does NOT match cfg.hidden_size=32
        let hidden = rand_w(&[1, 4, bad_h], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, bad_h], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let r = mtp.forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None);
        let err = r.expect_err("hidden_size mismatch must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("cfg.hidden_size") && msg.contains("32"),
            "expected hidden_size message containing cfg.hidden_size and 32, got: {msg}"
        );
    }

    #[test]
    fn forward_concat_layout_e_then_h() {
        // Pin concat order [e, h] (NOT [h, e]) — matches mlx-lm qwen3_5_mtp.py:380.
        //
        // Build pieces:
        //   fc weight (row-major [out=H, in=2H]):
        //     row i has W[i, i] = 1 (e-half coef) and W[i, H+i] = 3 (h-half coef), zeros elsewhere.
        //     ⇒ fc(concat([e, h])) = 1·e + 3·h.
        //     ⇒ fc(concat([h, e])) = 1·h + 3·e  (swap e/h).
        //   identity-DecoderLayer: zero o_proj + zero down_proj ⇒ out = x + 0 + 0 = x.
        //   pre_fc_norm and mtp.norm: RmsNorm with weight=ones (direction-preserving).
        //
        // Inputs:
        //   e = one-hot at channel 0 (broadcast over [B=1, S=4, H])  → pre_fc_norm(e)[k] = sqrt(H)·δ_{k,0}
        //   h = one-hot at channel 1                                 → pre_fc_norm(h)[k] = sqrt(H)·δ_{k,1}
        //
        // Under [e, h] ordering:
        //   fc out:    y[0] = 1·sqrt(H),  y[1] = 3·sqrt(H),  rest 0.
        //   identity:  unchanged.
        //   mtp.norm:  RMS(y) = sqrt((H + 9H)/H) = sqrt(10);  y'[0] = sqrt(H)/sqrt(10), y'[1] = 3·sqrt(H)/sqrt(10).
        //   ratio y'[1]/y'[0] = 3.0  ←— this PINS the order.
        // Under accidental [h, e] swap:
        //   ratio would be 1/3.
        let layer_cfg = small_layer_cfg();
        let cfg = MtpConfig {
            hidden_size: layer_cfg.hidden_size,
            num_mtp_layers: 1,
            layer: layer_cfg,
        };
        let h_dim = cfg.hidden_size as usize;

        // identity-on-zero-residual DecoderLayer.
        let q_w = rand_w(
            &[
                layer_cfg.num_heads * layer_cfg.head_dim * 2,
                layer_cfg.hidden_size,
            ],
            Dtype::Float32,
        );
        let k_w = rand_w(
            &[
                layer_cfg.num_kv_heads * layer_cfg.head_dim,
                layer_cfg.hidden_size,
            ],
            Dtype::Float32,
        );
        let v_w = rand_w(
            &[
                layer_cfg.num_kv_heads * layer_cfg.head_dim,
                layer_cfg.hidden_size,
            ],
            Dtype::Float32,
        );
        let o_w_zero = mlx::Array::zeros(
            (
                layer_cfg.hidden_size,
                layer_cfg.num_heads * layer_cfg.head_dim,
            ),
            Dtype::Float32,
        )
        .unwrap();
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w_zero, None),
            RmsNorm::new(ones_w(layer_cfg.head_dim), layer_cfg.rms_norm_eps),
            RmsNorm::new(ones_w(layer_cfg.head_dim), layer_cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: layer_cfg.num_heads,
                num_kv_heads: layer_cfg.num_kv_heads,
                head_dim: layer_cfg.head_dim,
                rms_norm_eps: layer_cfg.rms_norm_eps,
                attention_bias: layer_cfg.attention_bias,
            },
        );
        let gate_w = rand_w(
            &[layer_cfg.intermediate_size, layer_cfg.hidden_size],
            Dtype::Float32,
        );
        let up_w = rand_w(
            &[layer_cfg.intermediate_size, layer_cfg.hidden_size],
            Dtype::Float32,
        );
        let down_w_zero = mlx::Array::zeros(
            (layer_cfg.hidden_size, layer_cfg.intermediate_size),
            Dtype::Float32,
        )
        .unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w_zero, None),
        );
        let identity_layer = DecoderLayer::from_components(
            RmsNorm::new(ones_w(layer_cfg.hidden_size), layer_cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(layer_cfg.hidden_size), layer_cfg.rms_norm_eps),
            mlp,
            layer_cfg,
        );

        // Asymmetric fc weight: e-half coef = 1, h-half coef = 3.
        let mut fc_data = vec![0.0_f32; h_dim * 2 * h_dim];
        for i in 0..h_dim {
            fc_data[i * (2 * h_dim) + i] = 1.0;
            fc_data[i * (2 * h_dim) + (h_dim + i)] = 3.0;
        }
        let fc_w: Array = (
            fc_data.as_slice(),
            &[cfg.hidden_size, 2 * cfg.hidden_size][..],
        )
            .try_into()
            .unwrap();

        let mtp = Mtp::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            Linear::new_fp(fc_w, None),
            vec![identity_layer],
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            cfg,
        );

        // Inputs: e one-hot at channel 0; h one-hot at channel 1; broadcast over [B=1, S=4, H].
        let mut e_data = vec![0.0_f32; 4 * h_dim];
        let mut h_data = vec![0.0_f32; 4 * h_dim];
        for s in 0..4 {
            e_data[s * h_dim] = 1.0;
            h_data[s * h_dim + 1] = 1.0;
        }
        let next_embeds: Array = (e_data.as_slice(), &[1, 4, cfg.hidden_size][..])
            .try_into()
            .unwrap();
        let hidden: Array = (h_data.as_slice(), &[1, 4, cfg.hidden_size][..])
            .try_into()
            .unwrap();

        let mrope = Mrope::new(layer_cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = mtp
            .forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None)
            .unwrap();

        let v: Vec<f32> = out.to_vec().unwrap();
        // First row (b=0, s=0), channels 0 and 1.
        let c0 = v[0];
        let c1 = v[1];
        assert!(
            c0 > 0.0 && c1 > 0.0,
            "expected positive c0/c1, got c0={c0}, c1={c1}"
        );
        let ratio = c1 / c0;
        assert!(
            (ratio - 3.0_f32).abs() < 1e-3,
            "concat order broken: expected c1/c0 ≈ 3.0 ([e, h] order), got {ratio}",
        );
    }
}
