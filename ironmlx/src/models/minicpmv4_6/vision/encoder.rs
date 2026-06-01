//! SigLIP encoder: 27 × pre-norm MHA(+bias) / GELU-tanh MLP layers. No RoPE.

use anyhow::Result;
use mlx::fast::scaled_dot_product_attention;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;
use crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig;
use crate::nn::{gelu_tanh, LayerNorm};

// ---------------------------------------------------------------------------
// Multi-head attention (separate Q/K/V/out projections, with bias, no RoPE)
// ---------------------------------------------------------------------------

struct Mha {
    qw: Array,
    qb: Array,
    kw: Array,
    kb: Array,
    vw: Array,
    vb: Array,
    ow: Array,
    ob: Array,
    heads: i32,
    head_dim: i32,
}

impl Mha {
    fn from_loader(loader: &Loader, prefix: &str, hidden: i32, heads: i32) -> Result<Self> {
        let g = |n: &str| loader.tensor(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            qw: g("q_proj.weight")?,
            qb: g("q_proj.bias")?,
            kw: g("k_proj.weight")?,
            kb: g("k_proj.bias")?,
            vw: g("v_proj.weight")?,
            vb: g("v_proj.bias")?,
            ow: g("out_proj.weight")?,
            ob: g("out_proj.bias")?,
            heads,
            head_dim: hidden / heads,
        })
    }

    /// Linear projection: addmm(b, x, Wᵀ) — fused bias-matmul.
    ///
    /// // addmm (bias, x, Wᵀ) matches mlx nn.Linear — avoids 1-ULP drift vs the Python reference (see vision/block.rs).
    fn proj(x: &Array, w: &Array, b: &Array, t: StreamOrDevice) -> Result<Array> {
        let wt = w.transpose_on(t)?;
        Ok(ops::addmm_on(b, x, &wt, 1.0, 1.0, t)?)
    }

    fn forward_on(&self, x: &Array, t: StreamOrDevice) -> Result<Array> {
        let d = x.shape();
        let s = d.as_slice();
        let (bsz, seq) = (s[0], s[1]);

        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape_on(&[bsz, seq, self.heads, self.head_dim][..], t)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?)
        };

        let q = to_heads(Self::proj(x, &self.qw, &self.qb, t)?)?;
        let k = to_heads(Self::proj(x, &self.kw, &self.kb, t)?)?;
        let v = to_heads(Self::proj(x, &self.vw, &self.vb, t)?)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        // No causal mask for the vision encoder.
        let o = scaled_dot_product_attention(&q, &k, &v, scale, "", None, None)?;

        let o = o
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?
            .reshape_on(&[bsz, seq, self.heads * self.head_dim][..], t)?;

        Self::proj(&o, &self.ow, &self.ob, t)
    }
}

// ---------------------------------------------------------------------------
// One SigLIP encoder layer: pre-norm MHA + pre-norm MLP
// ---------------------------------------------------------------------------

pub struct SiglipEncoderLayer {
    ln1: LayerNorm,
    attn: Mha,
    ln2: LayerNorm,
    fc1w: Array,
    fc1b: Array,
    fc2w: Array,
    fc2b: Array,
}

impl SiglipEncoderLayer {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        hidden: i32,
        heads: i32,
        eps: f32,
    ) -> Result<Self> {
        let g = |n: &str| loader.tensor(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            ln1: LayerNorm::from_loader(loader, &format!("{prefix}.layer_norm1"), eps)?,
            attn: Mha::from_loader(loader, &format!("{prefix}.self_attn"), hidden, heads)?,
            ln2: LayerNorm::from_loader(loader, &format!("{prefix}.layer_norm2"), eps)?,
            fc1w: g("mlp.fc1.weight")?,
            fc1b: g("mlp.fc1.bias")?,
            fc2w: g("mlp.fc2.weight")?,
            fc2b: g("mlp.fc2.bias")?,
        })
    }

    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let t = target.into();

        // Attention sublayer: pre-norm + residual.
        let h = &self.attn.forward_on(&self.ln1.forward_on(x, t)?, t)? + x;

        // MLP sublayer: pre-norm → fc1 → gelu_tanh → fc2 → residual.
        let n = self.ln2.forward_on(&h, t)?;
        let wt1 = self.fc1w.transpose_on(t)?;
        let mlp = ops::addmm_on(&self.fc1b, &n, &wt1, 1.0, 1.0, t)?;
        let mlp = gelu_tanh(&mlp, t)?;
        let wt2 = self.fc2w.transpose_on(t)?;
        let mlp = ops::addmm_on(&self.fc2b, &mlp, &wt2, 1.0, 1.0, t)?;

        Ok(&h + &mlp)
    }

    /// Test-only constructor: build a layer from zero / unit Arrays without a
    /// checkpoint loader. Useful for shape-only unit tests.
    #[cfg(test)]
    pub fn new_for_test(hidden: i32, heads: i32, intermediate: i32) -> Self {
        use mlx::ops::constructors::ones;
        use mlx::Dtype;

        let zeros1d = |n: i32| Array::zeros(&[n][..], Dtype::Bfloat16).unwrap();
        let zeros2d = |r: i32, c: i32| Array::zeros(&[r, c][..], Dtype::Bfloat16).unwrap();

        let ln_weight = |n: i32| ones((n,), Dtype::Bfloat16).unwrap();
        let ln_bias = |n: i32| zeros1d(n);
        let ln = |n: i32| LayerNorm::new(ln_weight(n), Some(ln_bias(n)), 1e-6);

        let head_dim = hidden / heads;
        Self {
            ln1: ln(hidden),
            attn: Mha {
                qw: zeros2d(hidden, hidden),
                qb: zeros1d(hidden),
                kw: zeros2d(hidden, hidden),
                kb: zeros1d(hidden),
                vw: zeros2d(hidden, hidden),
                vb: zeros1d(hidden),
                ow: zeros2d(hidden, hidden),
                ob: zeros1d(hidden),
                heads,
                head_dim,
            },
            ln2: ln(hidden),
            fc1w: zeros2d(intermediate, hidden),
            fc1b: zeros1d(intermediate),
            fc2w: zeros2d(hidden, intermediate),
            fc2b: zeros1d(hidden),
        }
    }
}

// ---------------------------------------------------------------------------
// Full SigLIP encoder stack
// ---------------------------------------------------------------------------

pub struct SiglipEncoder {
    /// Public so the vision orchestration can run layers in two segments and
    /// insert the VitMerger resampler after `insert_layer_id` (mid-encoder
    /// downsample); a single all-layers forward would not allow that.
    pub layers: Vec<SiglipEncoderLayer>,
}

impl SiglipEncoder {
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(SiglipEncoderLayer::from_loader(
                loader,
                &format!("vision_tower.encoder.layers.{i}"),
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.layer_norm_eps,
            )?);
        }
        Ok(Self { layers })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn encoder_layer_preserves_shape() {
        let layer = SiglipEncoderLayer::new_for_test(1152, 16, 4304);
        let x = Array::zeros(&[1, 9, 1152][..], Dtype::Bfloat16).unwrap();
        let y = layer.forward_on(&x, ()).unwrap();
        assert_eq!(y.shape().as_slice(), &[1, 9, 1152]);
    }
}
