//! Qwen3.5 vision tower (24-layer ViT) — see
//! `docs/superpowers/specs/2026-05-10-p6-vl-design.md` §4.2-4.5.

pub mod block;
pub mod merger;
pub mod patch_embed;

use anyhow::Result;
use mlx::Array;

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
    /// e.g. `[2304, 1024]`. Full interpolation is PLACEHOLDER (Task 21).
    #[allow(dead_code)] // used in Task 21 pos_embed interpolation
    pos_embed: Array,
    /// Half of head_dim — used as the rotary dimension. E.g. 32 for head_dim=64.
    rotary_dim: i32,
    /// Rotary base frequency (10000.0 for Qwen3.5 VL).
    #[allow(dead_code)] // used in Task 21 rotary_pos_emb
    rotary_theta: f32,
    blocks: Vec<VitBlock>,
    merger: PatchMerger,
    #[allow(dead_code)] // used in Task 21 pos_embed interpolation
    hidden_size: i32,
    /// Square root of `num_position_embeddings`, e.g. 48 for 2304 embeddings.
    #[allow(dead_code)] // used in Task 21 pos_embed interpolation
    num_grid_per_side: i32,
    #[allow(dead_code)] // used in Task 21 pos_embed interpolation
    spatial_merge_size: i32,
    #[allow(dead_code)] // used in Task 21 rotary_pos_emb
    head_dim: i32,
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
        Ok(Self {
            patch_embed,
            pos_embed,
            rotary_dim: head_dim / 2,
            rotary_theta: 10_000.0,
            blocks,
            merger,
            hidden_size: cfg.hidden_size,
            num_grid_per_side,
            spatial_merge_size: cfg.spatial_merge_size,
            head_dim,
        })
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
        // PLACEHOLDER (Task 21): bilinear pos_embed interpolation.
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

    /// Add learned positional embedding to patch features.
    ///
    /// PLACEHOLDER — Task 21 fills in fast_pos_embed_interpolate from mlx-vlm.
    /// Currently returns `x` unchanged (no position information added).
    fn add_learned_pos_embed(&self, x: &Array, _grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
        Ok(x.clone())
    }

    /// Compute rotary positional embeddings for the given grid layout.
    ///
    /// PLACEHOLDER — Task 21 fills in the full rot_pos_emb from mlx-vlm
    /// (2D spatial + temporal frequencies). Currently returns a zero tensor
    /// of the correct shape `[n_patches, rotary_dim]` so downstream blocks
    /// run without crashing.
    fn compute_rotary_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> Result<Array> {
        let n_patches: i32 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
        Ok(Array::zeros(
            [n_patches, self.rotary_dim],
            mlx::Dtype::Float32,
        )?)
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
