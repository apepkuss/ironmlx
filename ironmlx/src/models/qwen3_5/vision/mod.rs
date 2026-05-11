//! Qwen3.5 vision tower (24-layer ViT) — see
//! `docs/superpowers/specs/2026-05-10-p6-vl-design.md` §4.2-4.5.

pub mod block;
pub mod dump;
pub mod merger;
pub mod patch_embed;

use anyhow::Result;
use mlx::{ops, Array};

use crate::core::Loader;
use crate::models::qwen3_5::VisionConfig;

use self::block::VitBlock;
use self::merger::PatchMerger;
use self::patch_embed::PatchEmbed;

// ---------------------------------------------------------------------------
// VisionTower
// ---------------------------------------------------------------------------

/// Qwen3.5 VisionTower: composes PatchEmbed + learned pos_embed + 24 ViT blocks
/// + PatchMerger into a single forward pass.
///
/// Weight loading requires [`Loader::open_multimodal`] so that `vision_tower.*`
/// keys are retained during sanitize.
pub struct VisionTower {
    patch_embed: PatchEmbed,
    /// Learned positional embedding table. Shape: `[num_position_embeddings, hidden_size]`
    /// e.g. `[2304, 1024]`. Used in `add_learned_pos_embed` for bilinear interpolation.
    pos_embed: Array,
    /// Half of head_dim — used as the rotary dimension. E.g. 32 for head_dim=64.
    rotary_dim: i32,
    /// Rotary base frequency (10000.0 for Qwen3.5 VL).
    rotary_theta: f32,
    blocks: Vec<VitBlock>,
    merger: PatchMerger,
    hidden_size: i32,
    /// Square root of `num_position_embeddings`, e.g. 48 for 2304 embeddings.
    num_grid_per_side: i32,
    spatial_merge_size: i32,
}

impl VisionTower {
    /// Construct VisionTower by loading all sub-module weights from `loader`.
    ///
    /// `loader` must have been opened with [`Loader::open_multimodal`] so that
    /// `vision_tower.*` tensor keys are available.
    pub fn from_loader(loader: &Loader, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let patch_embed =
            PatchEmbed::from_loader(loader, "vision_tower.patch_embed.proj", cfg.hidden_size)?;
        let pos_embed = loader.tensor("vision_tower.pos_embed.weight")?.clone();
        let mut blocks = Vec::with_capacity(cfg.depth as usize);
        for i in 0..cfg.depth {
            blocks.push(VitBlock::from_loader(
                loader,
                &format!("vision_tower.blocks.{i}"),
                cfg.num_heads,
                head_dim,
            )?);
        }
        let merger =
            PatchMerger::from_loader(loader, "vision_tower.merger", cfg.spatial_merge_size)?;
        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt() as i32;
        let tower = Self {
            patch_embed,
            pos_embed,
            rotary_dim: head_dim / 2,
            rotary_theta: 10_000.0,
            blocks,
            merger,
            hidden_size: cfg.hidden_size,
            num_grid_per_side,
            spatial_merge_size: cfg.spatial_merge_size,
        };
        // Eagerly evaluate every weight tensor held by the tower on the loading
        // thread. Constructors like `PatchEmbed::new` introduce lazy reshape ops
        // tagged with this thread's default MLX stream; if a later inference
        // call runs on a different thread (e.g. tokio blocking-pool), MLX errors
        // with "There is no Stream(gpu, N) in current thread." This mirrors the
        // pattern in `Loader::open_impl` for the raw weight map.
        tower.eval_weights()?;
        Ok(tower)
    }

    fn eval_weights(&self) -> Result<()> {
        let mut refs: Vec<&Array> = vec![&self.pos_embed];
        self.patch_embed.collect_weights(&mut refs);
        for blk in &self.blocks {
            blk.collect_weights(&mut refs);
        }
        self.merger.collect_weights(&mut refs);
        mlx::transforms::eval(&refs).map_err(|e| anyhow::anyhow!("VisionTower eval: {e}"))?;
        Ok(())
    }

    /// Returns the number of ViT blocks in the tower (e.g. 24 for Qwen3.5-VL).
    pub fn depth(&self) -> i32 {
        self.blocks.len() as i32
    }

    /// Full forward pass through the vision tower.
    ///
    /// `pixel_values`: pre-processed patches, shape `[N, T, C, H, W]`.
    /// `grid_thw`: per-image `(temporal, height, width)` grid dimensions.
    ///
    /// Returns merged patch features, shape `[total_patches / m², out_hidden]`.
    pub fn forward(&self, pixel_values: &Array, grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
        let mut x = self.patch_embed.forward(pixel_values)?;
        x = self.add_learned_pos_embed(&x, grid_thw)?;
        let rotary = self.compute_rotary_pos_emb(grid_thw)?;
        let cu_seqlens: Vec<i32> = {
            let mut v = vec![0_i32];
            let mut total = 0_i32;
            for (t, h, w) in grid_thw {
                total += t * h * w;
                v.push(total);
            }
            v
        };
        for blk in &self.blocks {
            x = blk.forward(&x, &rotary, &cu_seqlens)?;
        }
        self.merger.forward(&x, grid_thw)
    }

    /// Add learned positional embedding to patch features via bilinear interpolation.
    ///
    /// Translates `fast_pos_embed_interpolate` from mlx-vlm `vision.py:293-371`.
    /// For each grid (t, h, w), bilinearly interpolates the `[num_grid_per_side²,
    /// hidden]` embedding table to the target (h, w) size, tiles over t frames,
    /// then reorders patches into spatial-merge-block-consecutive order via a
    /// reshape + transpose matching the 2×2 spatial merge layout.
    fn add_learned_pos_embed(&self, x: &Array, grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
        let num_g = self.num_grid_per_side; // e.g. 48

        // Collect bilinear-interpolation indices and weights across all grids.
        // For each corner (top-left, top-right, bottom-left, bottom-right) we
        // accumulate flat indices into the [num_g*num_g, hidden] table.
        let total_hw: i32 = grid_thw.iter().map(|(_, h, w)| h * w).sum();
        let mut idx = [
            Vec::<i32>::with_capacity(total_hw as usize),
            Vec::<i32>::with_capacity(total_hw as usize),
            Vec::<i32>::with_capacity(total_hw as usize),
            Vec::<i32>::with_capacity(total_hw as usize),
        ];
        let mut wgt = [
            Vec::<f32>::with_capacity(total_hw as usize),
            Vec::<f32>::with_capacity(total_hw as usize),
            Vec::<f32>::with_capacity(total_hw as usize),
            Vec::<f32>::with_capacity(total_hw as usize),
        ];

        for &(_t, h, w) in grid_thw {
            // linspace(0, num_g-1, h) and linspace(0, num_g-1, w)
            let h_idxs: Vec<f32> = if h == 1 {
                vec![0.0_f32]
            } else {
                (0..h)
                    .map(|i| i as f32 * (num_g - 1) as f32 / (h - 1) as f32)
                    .collect()
            };
            let w_idxs: Vec<f32> = if w == 1 {
                vec![0.0_f32]
            } else {
                (0..w)
                    .map(|i| i as f32 * (num_g - 1) as f32 / (w - 1) as f32)
                    .collect()
            };

            let h_floor: Vec<i32> = h_idxs.iter().map(|v| *v as i32).collect();
            let w_floor: Vec<i32> = w_idxs.iter().map(|v| *v as i32).collect();
            let h_ceil: Vec<i32> = h_floor.iter().map(|&v| (v + 1).min(num_g - 1)).collect();
            let w_ceil: Vec<i32> = w_floor.iter().map(|&v| (v + 1).min(num_g - 1)).collect();

            let dh: Vec<f32> = h_idxs
                .iter()
                .zip(h_floor.iter())
                .map(|(v, &f)| v - f as f32)
                .collect();
            let dw: Vec<f32> = w_idxs
                .iter()
                .zip(w_floor.iter())
                .map(|(v, &f)| v - f as f32)
                .collect();

            let base_h: Vec<i32> = h_floor.iter().map(|&v| v * num_g).collect();
            let base_h_ceil: Vec<i32> = h_ceil.iter().map(|&v| v * num_g).collect();

            // Bilinear corners: iterate all (row, col) pairs in row-major order
            for r in 0..h as usize {
                for c in 0..w as usize {
                    idx[0].push(base_h[r] + w_floor[c]); // top-left
                    idx[1].push(base_h[r] + w_ceil[c]); // top-right
                    idx[2].push(base_h_ceil[r] + w_floor[c]); // bottom-left
                    idx[3].push(base_h_ceil[r] + w_ceil[c]); // bottom-right

                    wgt[0].push((1.0 - dh[r]) * (1.0 - dw[c]));
                    wgt[1].push((1.0 - dh[r]) * dw[c]);
                    wgt[2].push(dh[r] * (1.0 - dw[c]));
                    wgt[3].push(dh[r] * dw[c]);
                }
            }
        }

        // Build idx_tensor [4, total_hw] and weight_tensor [4, total_hw]
        let total = total_hw as usize;
        let idx_flat: Vec<i32> = idx[0]
            .iter()
            .chain(idx[1].iter())
            .chain(idx[2].iter())
            .chain(idx[3].iter())
            .copied()
            .collect();
        let idx_tensor: Array = (idx_flat.as_slice(), &[4_i32, total as i32][..]).try_into()?;

        let wgt_flat: Vec<f32> = wgt[0]
            .iter()
            .chain(wgt[1].iter())
            .chain(wgt[2].iter())
            .chain(wgt[3].iter())
            .copied()
            .collect();
        let wgt_tensor: Array = (wgt_flat.as_slice(), &[4_i32, total as i32][..]).try_into()?;
        // Cast weights to pos_embed dtype (bfloat16 for this model)
        let pe_dtype = self.pos_embed.dtype();
        let wgt_tensor = ops::cast::astype(&wgt_tensor, pe_dtype)?;
        let idx_tensor_i32 = ops::cast::astype(&idx_tensor, mlx::Dtype::Int32)?;

        // Look up pos_embed at each corner index: take(pe_w, idx_tensor_i32.reshape(-1), 0)
        // idx_tensor_i32 is [4, total_hw]; flatten to [4*total_hw], take, reshape back
        let pe_w = &self.pos_embed; // [num_g*num_g, hidden]
        let hidden_dim = pe_w.shape().as_slice()[1];
        let flat_idx = idx_tensor_i32.reshape(&[4 * total as i32][..])?;
        // take requires uint32 indices
        let flat_idx_u32 = ops::cast::astype(&flat_idx, mlx::Dtype::Uint32)?;
        let gathered_flat = ops::indexing::take(pe_w, &flat_idx_u32, 0)?;
        // gathered_flat: [4*total, hidden]
        let gathered = gathered_flat.reshape(&[4_i32, total as i32, hidden_dim][..])?;
        // gathered: [4, total_hw, hidden]

        // Weighted sum: gathered * wgt_tensor[:, :, None] → bilinear interpolated
        let wgt_bc = ops::shape::expand_dims(&wgt_tensor, &[2][..])?; // [4, total_hw, 1]
        let weighted = &gathered * &wgt_bc; // [4, total_hw, hidden]

        // Sum over 4 corners → [total_hw, hidden]
        let corner_parts = ops::shape::split_n(&weighted, 4, 0)?;
        let c0 = ops::shape::squeeze(&corner_parts[0], &[0][..])?;
        let c1 = ops::shape::squeeze(&corner_parts[1], &[0][..])?;
        let c2 = ops::shape::squeeze(&corner_parts[2], &[0][..])?;
        let c3 = ops::shape::squeeze(&corner_parts[3], &[0][..])?;
        let patch_pos_embeds = &(&c0 + &c1) + &(&c2 + &c3); // [total_hw, hidden]

        // For each grid, permute patches so that within each 2×2 merged block
        // the 4 sub-patches are consecutive (spatial-merge-block order).
        // This mirrors mlx-vlm's reshape+transpose(0,1,3,2,4,5)+reshape.
        let m = self.spatial_merge_size; // 2
        let hidden = self.hidden_size;
        let mut pieces: Vec<Array> = Vec::with_capacity(grid_thw.len());
        let mut offset: i32 = 0;

        for &(t, h, w) in grid_thw {
            let hw = h * w;
            // Slice this grid's embeddings from the flat buffer
            let pe = ops::slice(
                &patch_pos_embeds,
                &[offset, 0][..],
                &[offset + hw, hidden][..],
            )?; // [hw, hidden]

            // Tile over temporal frames (mx.tile(pe, (t,1)) == repeat along axis=0)
            let pe = if t > 1 {
                ops::shape::repeat(&pe, t, 0)?
            } else {
                pe
            };

            // Reorder into spatial-merge-block order:
            // reshape → [t, h/m, m, w/m, m, hidden]
            // transpose → [t, h/m, w/m, m, m, hidden] (axes: 0,1,3,2,4,5)
            // reshape → [-1, hidden]
            let pe = pe.reshape(&[t, h / m, m, w / m, m, hidden][..])?;
            let pe = ops::shape::transpose_axes(&pe, &[0_i32, 1, 3, 2, 4, 5][..])?;
            let thw = t * h * w;
            let pe = pe.reshape(&[thw, hidden][..])?;

            pieces.push(pe);
            offset += hw;
        }

        // Concatenate across all grids → [total_tokens, hidden]
        let refs: Vec<&Array> = pieces.iter().collect();
        let pos_embed_all = ops::concatenate(&refs, 0)?;

        // Add to patch features
        let x_cast = ops::cast::astype(x, pe_dtype)?;
        let result = &x_cast + &pos_embed_all;
        Ok(result)
    }

    /// Compute rotary positional embeddings for the given grid layout.
    ///
    /// Translates `rot_pos_emb` from mlx-vlm `vision.py:232-291`.
    ///
    /// For each grid (t, h, w) with `merge_size = 2`:
    ///   - Builds a 2D position index grid of shape `[merged_h, merged_w, 2, 2]`
    ///     where each entry carries its (row, col) absolute position in the
    ///     full patch grid.
    ///   - Flattens to `[merged_h * merged_w * 4, 2]` coordinate pairs.
    ///   - Looks up `freq_table[row]` and `freq_table[col]` and concatenates
    ///     along dim=-1 to yield `[tokens_this_grid, rotary_dim]`.
    ///
    /// Returns `[total_patches, rotary_dim]` float32. `rotary_dim = head_dim / 2`.
    fn compute_rotary_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
        let merge_size = self.spatial_merge_size; // 2

        // Max H or W across all grids — determines freq table length.
        let max_hw: i32 = grid_thw
            .iter()
            .flat_map(|&(_t, h, w)| [h, w])
            .max()
            .unwrap_or(1);

        // freq_table: [max_hw, rotary_dim/2] — half-dim table for one spatial axis.
        // `rotary_dim = head_dim/2` and `build_rotary_freqs(seqlen, dim, theta)`
        // returns [seqlen, dim/2]. To get [max_hw, rotary_dim/2]:
        //   build_rotary_freqs(max_hw, rotary_dim*2, theta) gives [max_hw, rotary_dim].
        // Wait — let's be precise:
        //   mlx-vlm: VisionRotaryEmbedding(dim = head_dim//2 = rotary_dim)
        //   its __call__: inv_freq length = dim/2 = rotary_dim/2
        //                 freq_table shape = [seqlen, rotary_dim/2]
        //   Our build_rotary_freqs(seqlen, dim, theta) returns [seqlen, dim/2].
        //   So build_rotary_freqs(max_hw, rotary_dim*2, theta) gives [max_hw, rotary_dim].
        // No wait: we want [max_hw, rotary_dim/2]:
        //   build_rotary_freqs(max_hw, rotary_dim, theta) gives [max_hw, rotary_dim/2].
        //   Then h_emb [tokens, rotary_dim/2] concat w_emb [tokens, rotary_dim/2]
        //   = [tokens, rotary_dim]. That matches.
        let freq_table = build_rotary_freqs(max_hw, self.rotary_dim, self.rotary_theta);
        // freq_table: [max_hw, rotary_dim/2]

        // Build row/col position indices for all grids, then gather from freq_table.
        let mut all_h_emb: Vec<Array> = Vec::with_capacity(grid_thw.len());
        let mut all_w_emb: Vec<Array> = Vec::with_capacity(grid_thw.len());

        for &(num_frames, height, width) in grid_thw {
            let merged_h = height / merge_size;
            let merged_w = width / merge_size;
            // Total tokens for this grid = num_frames * height * width
            let tokens_this = num_frames * height * width;

            // Build row_idx and col_idx each of shape [merged_h * merged_w * 4].
            // Index order (C-contiguous across (block_row, block_col, intra_row, intra_col)):
            //   row_pos(br, bc, ir, ic) = br * merge_size + ir
            //   col_pos(br, bc, ir, ic) = bc * merge_size + ic
            let inner = (merged_h * merged_w * merge_size * merge_size) as usize;
            let mut row_pos: Vec<i32> = Vec::with_capacity(inner);
            let mut col_pos: Vec<i32> = Vec::with_capacity(inner);

            for br in 0..merged_h {
                for bc in 0..merged_w {
                    for ir in 0..merge_size {
                        for ic in 0..merge_size {
                            row_pos.push(br * merge_size + ir);
                            col_pos.push(bc * merge_size + ic);
                        }
                    }
                }
            }

            // If multi-frame, tile the per-frame coords for each frame
            // (num_frames > 1 case: tile coords num_frames times).
            let (row_pos, col_pos) = if num_frames > 1 {
                let r_tiled: Vec<i32> = row_pos
                    .iter()
                    .cycle()
                    .take(tokens_this as usize)
                    .copied()
                    .collect();
                let c_tiled: Vec<i32> = col_pos
                    .iter()
                    .cycle()
                    .take(tokens_this as usize)
                    .copied()
                    .collect();
                (r_tiled, c_tiled)
            } else {
                (row_pos, col_pos)
            };

            let n = tokens_this as usize;
            let row_arr: Array = (row_pos.as_slice(), &[n as i32][..]).try_into()?;
            let col_arr: Array = (col_pos.as_slice(), &[n as i32][..]).try_into()?;

            // Take from freq_table: h_emb = freq_table[row_arr], shape [n, rotary_dim/2]
            // `take` requires uint32 indices.
            let row_u32 = ops::cast::astype(&row_arr, mlx::Dtype::Uint32)?;
            let col_u32 = ops::cast::astype(&col_arr, mlx::Dtype::Uint32)?;
            let h_emb = ops::indexing::take(&freq_table, &row_u32, 0)?;
            let w_emb = ops::indexing::take(&freq_table, &col_u32, 0)?;

            all_h_emb.push(h_emb);
            all_w_emb.push(w_emb);
        }

        // Concat all grids: h_emb [total, rotary_dim/2], w_emb [total, rotary_dim/2]
        let h_refs: Vec<&Array> = all_h_emb.iter().collect();
        let w_refs: Vec<&Array> = all_w_emb.iter().collect();
        let h_all = ops::concatenate(&h_refs, 0)?;
        let w_all = ops::concatenate(&w_refs, 0)?;

        // Concat h + w → [total, rotary_dim]
        ops::concatenate(&[&h_all, &w_all], 1).map_err(anyhow::Error::from)
    }
}

/// Vision rotary frequency table: `freqs[s, i] = s * (1 / theta^(2i/dim))`,
/// `i ∈ [0, dim/2)`. Output shape: `[seqlen, dim/2]`.
pub fn build_rotary_freqs(seqlen: i32, dim: i32, theta: f32) -> Array {
    use mlx::ops;

    let half = dim / 2;

    let exponents: Vec<f32> = (0..half).map(|i| (2 * i) as f32 / dim as f32).collect();
    let exponents_arr: Array = (exponents.as_slice(), &[half][..]).try_into().unwrap();

    let theta_arr: Array = (&[theta][..], ()).try_into().unwrap();
    let theta_pow = ops::power(&theta_arr, &exponents_arr).unwrap();
    let inv_freq = ops::reciprocal(&theta_pow).unwrap();

    let seq: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
    let seq_arr: Array = (seq.as_slice(), &[seqlen][..]).try_into().unwrap();
    let seq2 = ops::shape::reshape(&seq_arr, &[seqlen, 1][..]).unwrap();
    let inv2 = ops::shape::reshape(&inv_freq, &[1, half][..]).unwrap();

    &seq2 * &inv2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_pos_emb_shape() {
        let freqs = build_rotary_freqs(8, 32, 10000.0);
        assert_eq!(freqs.shape().as_slice(), &[8, 16]); // dim/2 = 16 entries
    }

    #[test]
    fn rotary_pos_emb_values_match_mlx_vlm() {
        let freqs = build_rotary_freqs(4, 32, 10000.0);
        let v: Vec<f32> = freqs.to_vec().unwrap();
        let expected_1_1 = 1.0_f32 / 10000.0_f32.powf(2.0 / 32.0);
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[16] - 1.0).abs() < 1e-5); // [1, 0]
        assert!((v[17] - expected_1_1).abs() < 1e-5); // [1, 1]
    }

    /// Load VisionTower from the real Qwen3.5-4B-MLX-4bit checkpoint and verify
    /// structural integrity (depth == 24). Requires the model on disk.
    ///
    /// Run with:
    /// ```
    /// QWEN35_MODEL=<path> cargo test -p ironmlx --lib --release vision_tower_load_qwen35_4b -- --ignored
    /// ```
    #[test]
    #[ignore] // real-model heavy
    fn vision_tower_load_qwen35_4b() {
        let env = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL not set");
        let loader =
            Loader::open_multimodal(std::path::Path::new(&env)).expect("loader open_multimodal");
        let cfg = crate::models::qwen3_5::Qwen35Config::from_loader(&loader).expect("Qwen35Config");
        let vc = cfg.vision_config.expect("vision_config present");
        let tower = VisionTower::from_loader(&loader, &vc).expect("VisionTower::from_loader");
        assert_eq!(tower.depth(), vc.depth);
    }
}
