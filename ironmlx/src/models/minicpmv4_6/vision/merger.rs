//! Spatial resamplers for MiniCPM-V-4.6.
//!
//! - `VitMerger`: mid-encoder window (2×2) cross-attention resampler inserted
//!   after `insert_layer_id` in the SigLIP encoder; groups each non-overlapping
//!   (gh × gw) = (2 × 2) window of spatial tokens, runs a local self-attention
//!   followed by mean-pooling and a two-layer MLP, and returns a spatially
//!   down-sampled sequence at half resolution in each dim.
//! - `Merger`: final 2×2 reshape-flatten + MLP projection to the LM hidden size
//!   (`hidden_size * 4 = 4608` → `lm_hidden = 1024`).
//!
//! Both types match the corresponding `__call__` implementations in mlx-vlm's
//! `minicpmv4_6/minicpmv4_6.py`.

use anyhow::{ensure, Result};
use mlx::fast::scaled_dot_product_attention;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;
use crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig;
use crate::nn::{gelu_tanh, LayerNorm};

// ---------------------------------------------------------------------------
// Self-attention over grouped window tokens (q = k = v = normed windows)
// ---------------------------------------------------------------------------

// Structurally identical to encoder.rs::Mha but operates per-window (seq=4)
// rather than over the full image sequence. Extract to a shared crate::nn MHA
// helper only if a third call site appears (avoid premature abstraction).
struct SelfAttn {
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

impl SelfAttn {
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

    fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.qw);
        out.push(&self.qb);
        out.push(&self.kw);
        out.push(&self.kb);
        out.push(&self.vw);
        out.push(&self.vb);
        out.push(&self.ow);
        out.push(&self.ob);
    }

    /// Fused bias-matmul projection matching `nn.Linear` behaviour.
    fn proj(x: &Array, w: &Array, b: &Array, t: StreamOrDevice) -> Result<Array> {
        let wt = w.transpose_on(t)?;
        Ok(ops::addmm_on(b, x, &wt, 1.0, 1.0, t)?)
    }

    /// Self-attention where q = k = v = `x` of shape `[M, group_tokens, hidden]`.
    ///
    /// Batch dimension here is M (number of merged windows), not B (image batch).
    fn forward_on(&self, x: &Array, t: StreamOrDevice) -> Result<Array> {
        let s = x.shape();
        let sl = s.as_slice();
        let (bsz, seq) = (sl[0], sl[1]); // M, group_tokens

        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape_on(&[bsz, seq, self.heads, self.head_dim][..], t)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?)
        };

        let q = to_heads(Self::proj(x, &self.qw, &self.qb, t)?)?;
        let k = to_heads(Self::proj(x, &self.kw, &self.kb, t)?)?;
        let v = to_heads(Self::proj(x, &self.vw, &self.vb, t)?)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, "", None, None)?;

        let o = o
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?
            .reshape_on(&[bsz, seq, self.heads * self.head_dim][..], t)?;

        Self::proj(&o, &self.ow, &self.ob, t)
    }
}

// ---------------------------------------------------------------------------
// VitMerger
// ---------------------------------------------------------------------------

/// Window-resampling module inserted mid-encoder.
///
/// Reduces spatial resolution by `merge_group` (2 × 2) via local
/// self-attention followed by mean-pooling and a gated MLP.
pub struct VitMerger {
    /// Pre-norm on the flat [M, 4*hidden] windows before the MLP.
    pre_norm: LayerNorm,
    /// Per-token layer norm applied inside the attention sublayer.
    layer_norm1: LayerNorm,
    /// Local self-attention over `group_tokens` tokens per window.
    self_attn: SelfAttn,
    /// MLP weight/bias: hidden_size*group_tokens → merged_hidden_size.
    linear_1w: Array,
    linear_1b: Array,
    /// MLP weight/bias: merged_hidden_size → hidden_size.
    linear_2w: Array,
    linear_2b: Array,
    /// Merge window shape (gh, gw) — (2, 2) for 16× downsampling.
    merge_gh: i32,
    merge_gw: i32,
}

impl VitMerger {
    /// Load from checkpoint using the `vit_merger.` prefix.
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        let p = "vit_merger";
        let g = |n: &str| loader.tensor(&format!("{p}.{n}")).cloned();

        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let (gh, gw) = cfg.merge_group;

        Ok(Self {
            layer_norm1: LayerNorm::from_loader(
                loader,
                &format!("{p}.layer_norm1"),
                cfg.layer_norm_eps,
            )?,
            self_attn: SelfAttn::from_loader(loader, &format!("{p}.self_attn"), hidden, heads)?,
            pre_norm: LayerNorm::from_loader(loader, &format!("{p}.pre_norm"), cfg.layer_norm_eps)?,
            linear_1w: g("linear_1.weight")?,
            linear_1b: g("linear_1.bias")?,
            linear_2w: g("linear_2.weight")?,
            linear_2b: g("linear_2.bias")?,
            merge_gh: gh,
            merge_gw: gw,
        })
    }

    /// Push every weight tensor (norms, attention, MLP) onto `out` for eager
    /// materialization on the loading thread.
    pub(super) fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(self.pre_norm.weight());
        if let Some(b) = self.pre_norm.bias() {
            out.push(b);
        }
        out.push(self.layer_norm1.weight());
        if let Some(b) = self.layer_norm1.bias() {
            out.push(b);
        }
        self.self_attn.collect_weights(out);
        out.push(&self.linear_1w);
        out.push(&self.linear_1b);
        out.push(&self.linear_2w);
        out.push(&self.linear_2b);
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x`      — `[grid_h * grid_w, hidden_size]`, bf16.
    /// * `grid_h` — spatial height of the incoming token grid.
    /// * `grid_w` — spatial width  of the incoming token grid.
    ///
    /// # Returns
    /// `(merged, merged_h, merged_w)` where `merged` is `[merged_h*merged_w, hidden_size]`.
    pub fn forward_on(
        &self,
        x: &Array,
        grid_h: i32,
        grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, i32, i32)> {
        let t = target.into();
        let gh = self.merge_gh;
        let gw = self.merge_gw;

        ensure!(
            grid_h % gh == 0 && grid_w % gw == 0,
            "VitMerger requires grid divisible by merge_group ({gh}×{gw}), got ({grid_h}×{grid_w})"
        );

        let merged_h = grid_h / gh;
        let merged_w = grid_w / gw;
        let m = merged_h * merged_w; // number of windows
        let group_tokens = gh * gw; // tokens per window (4 for 2×2)
        let hidden = x.shape().as_slice()[1];

        // Step 2: tile-group reshape.
        // [grid_h*grid_w, hidden] → [grid_h, grid_w, hidden]
        // → [merged_h, gh, merged_w, gw, hidden]
        // → transpose(0,2,1,3,4) → [merged_h, merged_w, gh, gw, hidden]
        // → [M, group_tokens, hidden]
        let windows = x
            .reshape_on(&[grid_h, grid_w, hidden][..], t)?
            .reshape_on(&[merged_h, gh, merged_w, gw, hidden][..], t)?
            .transpose_axes_on(&[0_i32, 2, 1, 3, 4][..], t)?
            .reshape_on(&[m, group_tokens, hidden][..], t)?;

        // Step 3: local self-attention sublayer (pre-norm + residual).
        let normed = self.layer_norm1.forward_on(&windows, t)?;
        let attn = self.self_attn.forward_on(&normed, t)?;
        let windows = &windows + &attn;

        // Step 4: mean over group_tokens → residual for the MLP branch.
        let residual = ops::mean_on(&windows, 1_i32, false, t)?; // [M, hidden]

        // Step 5: MLP branch.
        // Flatten: [M, group_tokens, hidden] → [M, group_tokens*hidden]
        let merged = windows.reshape_on(&[m, group_tokens * hidden][..], t)?;
        // pre_norm
        let merged = self.pre_norm.forward_on(&merged, t)?;
        // linear_1: addmm(b, x, Wᵀ) → [M, merged_hidden]
        let wt1 = self.linear_1w.transpose_on(t)?;
        let merged = ops::addmm_on(&self.linear_1b, &merged, &wt1, 1.0, 1.0, t)?;
        // gelu (tanh-approx — `approx="precise"` in mlx == gelu_approx == gelu_tanh)
        let merged = gelu_tanh(&merged, t)?;
        // linear_2: addmm(b, x, Wᵀ) → [M, hidden]
        let wt2 = self.linear_2w.transpose_on(t)?;
        let merged = ops::addmm_on(&self.linear_2b, &merged, &wt2, 1.0, 1.0, t)?;

        // Step 6: add residual and return spatial dims.
        let out = &merged + &residual;
        Ok((out, merged_h, merged_w))
    }

    /// Zero-weight instance for shape-only unit tests (no checkpoint needed).
    #[cfg(test)]
    pub fn new_for_test(hidden: i32, heads: i32, merged_hidden: i32) -> Self {
        use mlx::ops::constructors::ones;
        use mlx::Dtype;

        let zeros1d = |n: i32| Array::zeros(&[n][..], Dtype::Bfloat16).unwrap();
        let zeros2d = |r: i32, c: i32| Array::zeros(&[r, c][..], Dtype::Bfloat16).unwrap();
        let ln_weight = |n: i32| ones((n,), Dtype::Bfloat16).unwrap();
        let ln = |n: i32| LayerNorm::new(ln_weight(n), Some(zeros1d(n)), 1e-6);

        let head_dim = hidden / heads;
        let merge_gh = 2_i32;
        let merge_gw = 2_i32;
        let group_tokens = merge_gh * merge_gw;
        let group_hidden = hidden * group_tokens;

        Self {
            layer_norm1: ln(hidden),
            self_attn: SelfAttn {
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
            pre_norm: ln(group_hidden),
            linear_1w: zeros2d(merged_hidden, group_hidden),
            linear_1b: zeros1d(merged_hidden),
            linear_2w: zeros2d(hidden, merged_hidden),
            linear_2b: zeros1d(hidden),
            merge_gh,
            merge_gw,
        }
    }
}

// ---------------------------------------------------------------------------
// Merger — final 2×2 → LM-hidden projection
// ---------------------------------------------------------------------------

/// Final spatial resampler that projects SigLIP tokens from vision-hidden
/// (`hidden_size * 4 = 4608`) to LM-hidden (1024).
///
/// Implements the Python `Merger.__call__` / `MergerBlock.__call__` for the
/// single-block case (`merger_times = 1`).
///
/// Weight prefix: `merger.mlp.0.{pre_norm,linear_1,linear_2}.{weight,bias}`.
pub struct Merger {
    /// Pre-norm on the flat `[M, 4*hidden]` windows.
    pre_norm: LayerNorm,
    /// 4608 → 4608 projection.
    linear_1w: Array,
    linear_1b: Array,
    /// 4608 → lm_hidden (1024) projection.
    linear_2w: Array,
    linear_2b: Array,
    /// Merge window shape (gh, gw) — (2, 2).
    merge_gh: i32,
    merge_gw: i32,
}

impl Merger {
    /// Load from checkpoint using the `merger.mlp.0.` prefix.
    ///
    /// The LM-hidden output dimension is derived from `linear_2.weight`'s row
    /// count (shape `[lm_hidden, 4608]`) — no extra config plumbing needed.
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        let p = "merger.mlp.0";
        let g = |n: &str| loader.tensor(&format!("{p}.{n}")).cloned();

        let (gh, gw) = cfg.merge_group;

        Ok(Self {
            pre_norm: LayerNorm::from_loader(loader, &format!("{p}.pre_norm"), cfg.layer_norm_eps)?,
            linear_1w: g("linear_1.weight")?,
            linear_1b: g("linear_1.bias")?,
            linear_2w: g("linear_2.weight")?,
            linear_2b: g("linear_2.bias")?,
            merge_gh: gh,
            merge_gw: gw,
        })
    }

    /// Push every weight tensor (pre-norm + 2-layer MLP) onto `out` for eager
    /// materialization on the loading thread.
    pub(super) fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(self.pre_norm.weight());
        if let Some(b) = self.pre_norm.bias() {
            out.push(b);
        }
        out.push(&self.linear_1w);
        out.push(&self.linear_1b);
        out.push(&self.linear_2w);
        out.push(&self.linear_2b);
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x`      — `[grid_h * grid_w, hidden_size]`, bf16.
    /// * `grid_h` — spatial height of the incoming token grid.
    /// * `grid_w` — spatial width  of the incoming token grid.
    ///
    /// # Returns
    /// `(out, merged_h, merged_w)` where `out` is `[merged_h*merged_w, lm_hidden]`.
    pub fn forward_on(
        &self,
        x: &Array,
        grid_h: i32,
        grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, i32, i32)> {
        let t = target.into();
        let gh = self.merge_gh;
        let gw = self.merge_gw;

        ensure!(
            grid_h % gh == 0 && grid_w % gw == 0,
            "Merger requires grid divisible by merge_group ({gh}×{gw}), got ({grid_h}×{grid_w})"
        );

        let mh = grid_h / gh;
        let mw = grid_w / gw;
        let inner_dim = x.shape().as_slice()[1]; // hidden_size (1152)

        // Reshape: [grid_h*grid_w, inner] → [mh, gh, mw, gw, inner]
        //   → transpose(0,2,1,3,4) → [mh, mw, gh, gw, inner]
        //   → [mh*mw, inner*gh*gw]  (flatten spatial group into feature axis)
        let flat = x
            .reshape_on(&[grid_h, grid_w, inner_dim][..], t)?
            .reshape_on(&[mh, gh, mw, gw, inner_dim][..], t)?
            .transpose_axes_on(&[0_i32, 2, 1, 3, 4][..], t)?
            .reshape_on(&[mh * mw, inner_dim * gh * gw][..], t)?;

        // MergerBlock: pre_norm → linear_1 → gelu_tanh → linear_2
        let x = self.pre_norm.forward_on(&flat, t)?;
        let wt1 = self.linear_1w.transpose_on(t)?;
        let x = ops::addmm_on(&self.linear_1b, &x, &wt1, 1.0, 1.0, t)?;
        let x = gelu_tanh(&x, t)?;
        let wt2 = self.linear_2w.transpose_on(t)?;
        let out = ops::addmm_on(&self.linear_2b, &x, &wt2, 1.0, 1.0, t)?;

        Ok((out, mh, mw))
    }

    /// Zero-weight instance for shape-only unit tests (no checkpoint needed).
    #[cfg(test)]
    pub fn new_for_test(group_hidden: i32, lm_hidden: i32) -> Self {
        use mlx::ops::constructors::ones;
        use mlx::Dtype;

        let zeros1d = |n: i32| Array::zeros(&[n][..], Dtype::Bfloat16).unwrap();
        let zeros2d = |r: i32, c: i32| Array::zeros(&[r, c][..], Dtype::Bfloat16).unwrap();
        let ln_weight = |n: i32| ones((n,), Dtype::Bfloat16).unwrap();
        let ln = |n: i32| LayerNorm::new(ln_weight(n), Some(zeros1d(n)), 1e-6);

        Self {
            pre_norm: ln(group_hidden),
            linear_1w: zeros2d(group_hidden, group_hidden),
            linear_1b: zeros1d(group_hidden),
            linear_2w: zeros2d(lm_hidden, group_hidden),
            linear_2b: zeros1d(lm_hidden),
            merge_gh: 2,
            merge_gw: 2,
        }
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
    fn vit_merger_halves_grid_and_errors_on_odd() {
        let m = VitMerger::new_for_test(1152, 16, 17216);
        let x = Array::zeros(&[6 * 6, 1152][..], Dtype::Bfloat16).unwrap();
        let (out, h, w) = m.forward_on(&x, 6, 6, ()).unwrap();
        assert_eq!((h, w), (3, 3));
        assert_eq!(out.shape().as_slice(), &[9, 1152]);
        assert!(m.forward_on(&x, 5, 6, ()).is_err());
    }

    #[test]
    fn merger_outputs_lm_hidden() {
        let m = Merger::new_for_test(4608, 1024);
        let x = Array::zeros(&[6 * 6, 1152][..], Dtype::Bfloat16).unwrap();
        let (out, h, w) = m.forward_on(&x, 6, 6, ()).unwrap();
        assert_eq!((h, w), (3, 3));
        assert_eq!(out.shape().as_slice(), &[9, 1024]);
        assert!(m.forward_on(&x, 5, 6, ()).is_err());
    }
}
