//! Inkling's vision tower: merge-ordered patches, bilinear positions and 2D RoPE.
//! Adapted from Apple's mlx-lm Inkling 2 reference (MIT, 2026).
use super::{config::VisionConfig, ops::*};
use crate::core::Loader;
use crate::nn::{LayerNorm, Linear};
use anyhow::{ensure, Result};
use mlx::ops::{indexing, shape};
use mlx::{Array, Dtype, StreamOrDevice};

struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    lambda1: Array,
    lambda2: Array,
}
pub(super) struct VisionTower {
    config: VisionConfig,
    patch: Linear,
    positions: Array,
    cls: Array,
    cls_position: Array,
    norm: LayerNorm,
    blocks: Vec<Block>,
    merge1: Linear,
    merge2: Linear,
    dtype: Dtype,
}
impl VisionTower {
    pub fn load(l: &Loader, c: &VisionConfig) -> Result<Self> {
        let p = "model.vision_tower";
        let mut blocks = Vec::new();
        for i in 0..c.num_hidden_layers {
            let b = format!("{p}.layers.{i}");
            let linear = |n: &str| Linear::from_loader(l, &format!("{b}.{n}"));
            blocks.push(Block {
                q: linear("self_attn.q_proj")?,
                k: linear("self_attn.k_proj")?,
                v: linear("self_attn.v_proj")?,
                o: linear("self_attn.out_proj")?,
                fc1: linear("mlp.fc1")?,
                fc2: linear("mlp.fc2")?,
                norm1: LayerNorm::from_loader(
                    l,
                    &format!("{b}.layernorm_before"),
                    c.layer_norm_eps,
                )?,
                norm2: LayerNorm::from_loader(
                    l,
                    &format!("{b}.layernorm_after"),
                    c.layer_norm_eps,
                )?,
                lambda1: l.tensor(&format!("{b}.lambda_1"))?.clone(),
                lambda2: l.tensor(&format!("{b}.lambda_2"))?.clone(),
            });
        }
        Ok(Self {
            config: c.clone(),
            patch: Linear::from_loader(l, &format!("{p}.embeddings.patch_embedding"))?,
            positions: l
                .tensor(&format!("{p}.embeddings.position_embedding.weight"))?
                .clone(),
            cls: l
                .tensor(&format!("{p}.embeddings.class_embedding"))?
                .clone(),
            cls_position: l
                .tensor(&format!("{p}.embeddings.cls_position_embedding"))?
                .clone(),
            norm: LayerNorm::from_loader(l, &format!("{p}.pre_layernorm"), c.layer_norm_eps)?,
            blocks,
            merge1: Linear::from_loader(l, &format!("{p}.merger.linear_1"))?,
            merge2: Linear::from_loader(l, &format!("{p}.merger.linear_2"))?,
            dtype: l
                .tensor(&format!("{p}.embeddings.patch_embedding.weight"))?
                .dtype(),
        })
    }
    pub fn forward(&self, pixels: &Array, t: StreamOrDevice) -> Result<Array> {
        let c = &self.config;
        let p = c.native_patch_size;
        let m = c.adapter_merge_size;
        ensure!(
            pixels.ndim() == 3 && pixels.shape_at(0) == 3,
            "Inkling vision expects [3,H,W]"
        );
        let (h, w) = (pixels.shape_at(1), pixels.shape_at(2));
        ensure!(
            h > 0 && w > 0 && h % (p * m) == 0 && w % (p * m) == 0,
            "Inkling image dimensions must be multiples of patch/merge size"
        );
        let (rows, cols) = (h / (p * m), w / (p * m));
        let (gh, gw) = (rows * m, cols * m);
        let n = gh * gw;
        let patches = pixels
            .transpose_axes_on([1, 2, 0], t)?
            .reshape_on(&[rows, m, p, cols, m, p, 3][..], t)?
            .transpose_axes_on(&[0, 3, 1, 4, 2, 5, 6][..], t)?
            .reshape_on((n, p * p * 3), t)?
            .astype_on(self.dtype, t)?;
        let mut taps = Vec::new();
        let mut weights = Vec::new();
        let mut row_pos = vec![0];
        let mut col_pos = vec![0];
        let side = c.posemb_grid_size;
        for i in 0..n {
            let row = i / (m * m * cols) * m + i / m % m;
            let col = i / (m * m) % cols * m + i % m;
            row_pos.push(row + 1);
            col_pos.push(col + 1);
            let sy = (row as f32 + 0.5) * side as f32 / gh as f32 - 0.5;
            let sx = (col as f32 + 0.5) * side as f32 / gw as f32 - 0.5;
            for dy in 0..2 {
                for dx in 0..2 {
                    let y = sy.floor() as i32 + dy;
                    let x = sx.floor() as i32 + dx;
                    taps.push(y.clamp(0, side - 1) * side + x.clamp(0, side - 1));
                    weights.push(
                        (1.0 - (sy - y as f32).abs()).max(0.0)
                            * (1.0 - (sx - x as f32).abs()).max(0.0),
                    );
                }
            }
        }
        let taps: Array = (taps.as_slice(), &[n, 4][..]).try_into()?;
        let weights: Array = (weights.as_slice(), &[n, 4, 1][..]).try_into()?;
        let position = indexing::take_on(&self.positions, &taps, 0, t)?
            .astype_on(Dtype::Float32, t)?
            .try_mul_on(&weights, t)?
            .sum_on(1, false, t)?
            .astype_on(self.dtype, t)?;
        let patches = self
            .patch
            .forward_on(&patches, t)?
            .try_add_on(&position, t)?;
        let cls = self
            .cls
            .try_add_on(&self.cls_position, t)?
            .astype_on(self.dtype, t)?
            .expand_dims_on(0, t)?;
        let mut hidden = self.norm.forward_on(&cat(&[&cls, &patches], 0, t)?, t)?;
        let d = c.hidden_size / c.num_attention_heads;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for i in 0..=n as usize {
            for pos in [col_pos[i], row_pos[i]] {
                for j in 0..d / 4 {
                    let angle = pos as f32 / c.rope_theta.powf((2 * j) as f32 / (d / 2) as f32);
                    cos.extend([angle.cos(); 2]);
                    sin.extend([angle.sin(); 2]);
                }
            }
        }
        let cos: Array = (cos.as_slice(), &[n + 1, 1, d][..]).try_into()?;
        let sin: Array = (sin.as_slice(), &[n + 1, 1, d][..]).try_into()?;
        let rotary = |x: Array| -> Result<Array> {
            let x = x.reshape_on((n + 1, c.num_attention_heads, d), t)?;
            let f = x.astype_on(Dtype::Float32, t)?;
            let pairs = f.reshape_on((n + 1, c.num_attention_heads, d / 2, 2), t)?;
            let even = slice(&pairs, 3, 0, 1, t)?.squeeze_on(3, t)?;
            let odd = slice(&pairs, 3, 1, 2, t)?.squeeze_on(3, t)?.try_neg_on(t)?;
            let rotated = shape::stack_on(&[&odd, &even], -1, t)?
                .reshape_on((n + 1, c.num_attention_heads, d), t)?;
            Ok(f.try_mul_on(&cos, t)?
                .try_add_on(&rotated.try_mul_on(&sin, t)?, t)?
                .astype_on(x.dtype(), t)?
                .transpose_axes_on([1, 0, 2], t)?
                .expand_dims_on(0, t)?)
        };
        for block in &self.blocks {
            let normed = block.norm1.forward_on(&hidden, t)?;
            let q = rotary(block.q.forward_on(&normed, t)?)?;
            let k = rotary(block.k.forward_on(&normed, t)?)?;
            let v = block
                .v
                .forward_on(&normed, t)?
                .reshape_on((n + 1, c.num_attention_heads, d), t)?
                .transpose_axes_on([1, 0, 2], t)?
                .expand_dims_on(0, t)?;
            let attn = mlx::fast::scaled_dot_product_attention_on(
                &q,
                &k,
                &v,
                (d as f32).powf(-0.5),
                "",
                None,
                None,
                t,
            )?
            .squeeze_on(0, t)?
            .transpose_axes_on([1, 0, 2], t)?
            .reshape_on((n + 1, c.hidden_size), t)?;
            hidden = hidden.try_add_on(
                &block
                    .o
                    .forward_on(&attn, t)?
                    .try_mul_on(&block.lambda1, t)?,
                t,
            )?;
            let mlp = block.fc2.forward_on(
                &gelu(
                    &block
                        .fc1
                        .forward_on(&block.norm2.forward_on(&hidden, t)?, t)?,
                    t,
                )?,
                t,
            )?;
            hidden = hidden.try_add_on(&mlp.try_mul_on(&block.lambda2, t)?, t)?;
        }
        let merged =
            slice(&hidden, 0, 1, n + 1, t)?.reshape_on((rows * cols, m * m * c.hidden_size), t)?;
        self.merge2
            .forward_on(&gelu(&self.merge1.forward_on(&merged, t)?, t)?, t)
    }
}
