//! SigLIP patch embedding + NaViT-style position-bucket interpolation.

use anyhow::Result;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;

use super::super::config::MiniCpmV46VisionConfig;

pub struct SiglipEmbeddings {
    /// Conv weight reshaped to [hidden, patch*patch*channels] = [1152, 588].
    patch_w_2d: Array,
    patch_b: Array,
    /// [pos_grid_side^2, hidden] = [4900, 1152].
    pos_embed: Array,
    hidden: i32,
    pos_grid_side: i32,
}

/// Map each patch of a (grid_h, grid_w) image to a learned-position-table id
/// via fractional bucketing against `pos_grid_side` boundaries (mlx-vlm
/// `_build_position_buckets`). Row-major over (h, w).
pub fn position_bucket_ids(grid_h: i32, grid_w: i32, side: i32) -> Vec<i32> {
    let bucket = |n: i32| -> Vec<i32> {
        let n = n.max(1);
        (0..n)
            .map(|i| {
                let frac = ((i as f32) / (n as f32)).min(1.0 - 1e-6);
                let mut b = 0;
                for k in 1..side {
                    if frac >= (k as f32) / (side as f32) {
                        b += 1;
                    }
                }
                b
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
        let patch_elems = cfg.patch_size * cfg.patch_size * 3;
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
            hidden: cfg.hidden_size,
            pos_grid_side: cfg.pos_grid_side,
        })
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
        let pos = pos.reshape_on(&[1_i32, ids.len() as i32, self.hidden][..], target)?;
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
            hidden,
            pos_grid_side: 70,
        };

        // pixel_values: [1, 14, n*14, 3] — all n = grid_h*grid_w patches in one row
        let pixel_values = Array::zeros(&[1, patch, n * patch, 3], Dtype::Bfloat16).unwrap();
        let out = emb.forward_on(&pixel_values, grid_h, grid_w, ()).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, n, hidden]);
    }
}
