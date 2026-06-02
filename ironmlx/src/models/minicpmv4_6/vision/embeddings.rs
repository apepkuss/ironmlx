//! SigLIP patch embedding + NaViT-style position-bucket interpolation.

use anyhow::Result;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;

use crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig;

pub struct SiglipEmbeddings {
    /// Conv weight reshaped to [hidden, patch*patch*channels] = [1152, 588].
    patch_w_2d: Array,
    patch_b: Array,
    /// [pos_grid_side^2, hidden] = [4900, 1152].
    pos_embed: Array,
    hidden_size: i32,
    pos_grid_side: i32,
}

/// Map each patch of a (grid_h, grid_w) image to a learned-position-table id
/// via fractional bucketing against `pos_grid_side` boundaries (mlx-vlm
/// `_build_position_buckets`). Row-major over (h, w).
///
/// Float semantics are an EXACT port of the mlx-vlm reference, which compares
/// `frac >= boundaries` where `boundaries = mx.arange(1/side, 1.0, 1/side)`.
/// MLX's Metal `arange` kernel computes `out[j] = start + j*step` in the array
/// dtype (here f32) — and the GPU fuses that into a single-rounded FMA, so
/// `boundaries[j] = fma(j, step, step)` where `step = (1.0 / side as f64) as f32` — NOT a freshly
/// recomputed `(j+1)/side`. The two formulations disagree at the exact tie
/// `frac == (j+1)/side` (e.g. `26/28 == 65/70`): the FMA boundary rounds
/// slightly *above* `frac`, so the tie does NOT increment the bucket.
/// Recomputing `k/side` per step (the naive port) gets that tie wrong and
/// shifts a whole grid row of position ids, corrupting the embeddings. We
/// therefore replicate the `start + j*step` FMA boundary arithmetic verbatim
/// (verified bit-identical to `mx.arange` across all `side-1` boundaries).
pub fn position_bucket_ids(grid_h: i32, grid_w: i32, side: i32) -> Vec<i32> {
    // boundaries[j] = fma(j, step, step), step = (1.0 / side as f64) as f32,
    // j in 0..side-1. Bit-matches `mx.arange(1/side, 1.0, 1/side)` on Metal.
    let step = (1.0_f64 / side as f64) as f32;
    let boundaries: Vec<f32> = (0..side - 1)
        .map(|j| (j as f32).mul_add(step, step))
        .collect();
    let bucket = |n: i32| -> Vec<i32> {
        let n = n.max(1);
        (0..n)
            .map(|i| {
                // Defensive clamp ported from mlx-vlm `_build_position_buckets`;
                // frac never reaches 1.0 for i in 0..n, but kept for parity.
                let frac = ((i as f32) / (n as f32)).min(1.0 - 1e-6);
                boundaries.iter().filter(|&&b| frac >= b).count() as i32
            })
            .collect()
    };
    let bh = bucket(grid_h);
    let bw = bucket(grid_w);
    let mut ids = Vec::with_capacity((grid_h * grid_w) as usize);
    for &h in &bh {
        for &w in &bw {
            ids.push(h * side + w);
        }
    }
    ids
}

impl SiglipEmbeddings {
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        let w = loader
            .tensor("vision_tower.embeddings.patch_embedding.weight")?
            .clone();
        let patch_elems = cfg.patch_size * cfg.patch_size * 3; // patch_size² × 3 (RGB) = 588
        let patch_w_2d = w.reshape(&[cfg.hidden_size, patch_elems][..])?;
        let patch_b = loader
            .tensor("vision_tower.embeddings.patch_embedding.bias")?
            .clone();
        let pos_embed = loader
            .tensor("vision_tower.embeddings.position_embedding.weight")?
            .clone();
        Ok(Self {
            patch_w_2d,
            patch_b,
            pos_embed,
            hidden_size: cfg.hidden_size,
            pos_grid_side: cfg.pos_grid_side,
        })
    }

    /// Push every weight tensor onto `out` for eager materialization on the
    /// loading thread (see [`super::MiniCpmV46Vision::eval_weights`]).
    pub(super) fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.patch_w_2d);
        out.push(&self.patch_b);
        out.push(&self.pos_embed);
    }

    /// `pixel_values`: patch-packed `[1, patch, n*patch, 3]`. Returns `[1, grid_h*grid_w, hidden]`.
    pub fn forward_on(
        &self,
        pixel_values: &Array,
        grid_h: i32,
        grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let dims = pixel_values.shape();
        let d = dims.as_slice();
        let (p, total_w, c) = (d[1], d[2], d[3]);
        anyhow::ensure!(
            total_w % p == 0,
            "SiglipEmbeddings: packed pixel width {total_w} not divisible by patch {p}"
        );
        let n = total_w / p;
        let x = pixel_values.reshape_on(&[1, p, n, p, c][..], target)?;
        let x = x.transpose_axes_on(&[0_i32, 2, 1, 3, 4][..], target)?;
        let x = x.reshape_on(&[1, n, p * p * c][..], target)?;
        let wt = self.patch_w_2d.transpose_on(target)?; // [588, 1152]
        let embeds = x.matmul_on(&wt, target)?; // [1, n, 1152]
        let embeds = &embeds + &self.patch_b;
        let ids = position_bucket_ids(grid_h, grid_w, self.pos_grid_side);
        let id_arr: Array = (ids.as_slice(), &[ids.len() as i32][..]).try_into()?;
        // ops::indexing::take requires uint32 indices
        let id_arr_u32 = ops::cast::astype(&id_arr, mlx::Dtype::Uint32)?;
        let pos = ops::indexing::take(&self.pos_embed, &id_arr_u32, 0)?; // [n, 1152]
        let pos = pos.reshape_on(&[1_i32, ids.len() as i32, self.hidden_size][..], target)?;
        Ok(&embeds + &pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn position_bucket_ids_match_grid() {
        // grid 4x4, pos_grid_side=70 → buckets in [0,70); ids = bh*70 + bw.
        let ids = position_bucket_ids(4, 4, 70);
        assert_eq!(ids.len(), 16);
        assert_eq!(ids[0], 0);
        assert!(ids.iter().all(|&v| v >= 0 && v < 70 * 70));
    }

    #[test]
    fn position_bucket_tie_matches_mlx_arange_fma() {
        // Regression for the FMA boundary tie bug found in P1 vision parity.
        // For grid_h=28, side=70: frac(26) = 26/28 == 65/70 exactly. The naive
        // `frac >= k/side` port counts this tie (bucket 65), but mlx-vlm's
        // `mx.arange(1/side, 1.0, 1/side)` boundary is FMA-rounded slightly
        // above frac, so the tie does NOT count (bucket 64). A row's worth of
        // position ids hinges on this; getting it wrong shifts the whole row.
        // grid_h=28, grid_w=1 → bw[0]=0, so each id is bh[h]*side + 0 = bh[h]*70.
        let ids = position_bucket_ids(28, 1, 70);
        assert_eq!(ids.len(), 28);
        // Full mlx-vlm reference height buckets for grid_h=28, side=70.
        let bh_ref: [i32; 28] = [
            0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55,
            57, 60, 62, 64, 67,
        ];
        let bh: Vec<i32> = ids.iter().map(|&v| v / 70).collect();
        assert_eq!(
            bh[26], 64,
            "frac==65/70 tie must NOT increment (mlx FMA boundary)"
        );
        assert_eq!(bh.as_slice(), &bh_ref);
    }

    #[test]
    fn siglip_embeddings_output_shape() {
        // hidden=1152, patch_size=14, pos_grid_side=70
        // grid 4x4 → 16 patches, pixel_values [1, 14, 4*14, 3] = [1, 14, 56, 3]
        let hidden = 1152_i32;
        let patch = 14_i32;
        let grid_h = 4_i32;
        let grid_w = 4_i32;
        let n = grid_h * grid_w;
        let patch_elems = patch * patch * 3; // 588

        let patch_w = Array::zeros(&[hidden, patch_elems], Dtype::Bfloat16).unwrap();
        let patch_b = Array::zeros(&[hidden], Dtype::Bfloat16).unwrap();
        // pos_embed: [70*70, 1152] = [4900, 1152]
        let pos_embed = Array::zeros(&[4900, hidden], Dtype::Bfloat16).unwrap();

        let emb = SiglipEmbeddings {
            patch_w_2d: patch_w,
            patch_b,
            pos_embed,
            hidden_size: hidden,
            pos_grid_side: 70,
        };

        // pixel_values: [1, 14, n*14, 3] — all n = grid_h*grid_w patches in one row
        let pixel_values = Array::zeros(&[1, patch, n * patch, 3], Dtype::Bfloat16).unwrap();
        let out = emb.forward_on(&pixel_values, grid_h, grid_w, ()).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, n, hidden]);
    }

    #[test]
    fn position_bucket_ids_rectangular_grid() {
        let ids = position_bucket_ids(2, 8, 70);
        assert_eq!(ids.len(), 16);
        // row-major: id = bucket_h*70 + bucket_w; first row all share bucket_h=0.
        assert_eq!(ids[0], 0);
        assert!(ids.iter().all(|&v| v >= 0 && v < 70 * 70));
    }
}
