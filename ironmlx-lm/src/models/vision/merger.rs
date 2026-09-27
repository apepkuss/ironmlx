//! Qwen3.5 patch merger — norm + spatial 2×2 merge + 2-layer MLP.
//! Project from vision hidden=1024 to text hidden=2560. See spec §4.5.

use anyhow::Result;
use mlx::{ops, Array, StreamOrDevice};

use super::projection::VisionProjection;
use crate::nn::LayerNorm;
use ironmlx_core::weights::WeightSource;

/// Exact GELU activation (erf-based).
///
/// Formula: `0.5 * x * (1 + erf(x / sqrt(2)))`
///
/// This matches PyTorch `nn.GELU()` default (not the tanh approximation)
/// and mlx-vlm `nn.GELU()` used in `PatchMerger`.
fn gelu_exact(x: &Array, target: StreamOrDevice) -> Result<Array> {
    // x / sqrt(2)
    let x_scaled = x * std::f32::consts::FRAC_1_SQRT_2;
    // erf(x / sqrt(2))
    let e = x_scaled.erf_on(target)?;
    // 0.5 * x * (1 + erf(...))
    let out = x * 0.5_f32 * (&e + 1.0_f32);
    Ok(out)
}

// ---------------------------------------------------------------------------
// PatchMerger
// ---------------------------------------------------------------------------

/// Merges spatial-adjacent patches and projects to the LLM embedding dimension.
///
/// Architecture (matches `PatchMerger` in mlx-vlm/qwen3_vl/vision.py):
/// ```text
/// x: [N, hidden]            (N patches, hidden=1024 per-patch dim)
/// 1. norm(x)                → [N, 1024]      LayerNorm eps=1e-6
/// 2. reshape                → [N/m², m²·hidden]  = [N/4, 4096] for m=2
/// 3. fc1(x) + bias          → [N/4, 4096]
/// 4. exact GELU             → [N/4, 4096]
/// 5. fc2(x) + bias          → [N/4, out_hidden]  = [N/4, 2560]
/// ```
/// where `m = spatial_merge_size = 2`.
pub struct PatchMerger {
    norm: LayerNorm,
    fc1: VisionProjection,
    fc2: VisionProjection,
    /// Spatial merge size (default 2 → merges 2×2 = 4 patches per token).
    spatial_merge_size: i32,
    /// Per-patch hidden dim before merge (1024 for Qwen3.5-VL).
    hidden_size: i32,
    /// DeepStack mergers normalize the already shuffled 2x2 patch group.
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    /// Construct from pre-loaded weight Arrays.
    ///
    /// - `norm_w` / `norm_b`: shape `[hidden_size]`, e.g. `[1024]`.
    /// - `fc1_w`: shape `[merge_hidden, merge_hidden]`, e.g. `[4096, 4096]`.
    /// - `fc1_b`: shape `[merge_hidden]`, e.g. `[4096]`.
    /// - `fc2_w`: shape `[out_hidden, merge_hidden]`, e.g. `[2560, 4096]`.
    /// - `fc2_b`: shape `[out_hidden]`, e.g. `[2560]`.
    /// - `spatial_merge_size`: typically 2.
    pub fn new(
        norm_w: Array,
        norm_b: Array,
        fc1_w: Array,
        fc1_b: Array,
        fc2_w: Array,
        fc2_b: Array,
        spatial_merge_size: i32,
    ) -> Self {
        let hidden_size = norm_w.shape()[0];
        Self {
            norm: LayerNorm::new(norm_w, Some(norm_b), 1e-6),
            fc1: VisionProjection::new_fp(fc1_w, Some(fc1_b)),
            fc2: VisionProjection::new_fp(fc2_w, Some(fc2_b)),
            spatial_merge_size,
            hidden_size,
            use_postshuffle_norm: false,
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names (under `prefix`):
    /// - `{prefix}.norm.weight` / `.bias`
    /// - `{prefix}.linear_fc1.weight` / `.bias`
    /// - `{prefix}.linear_fc2.weight` / `.bias`
    pub fn from_loader(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        spatial_merge_size: i32,
    ) -> Result<Self> {
        Self::from_loader_with_norm_position(loader, prefix, spatial_merge_size, false)
    }

    /// Load a DeepStack merger whose LayerNorm is applied after the spatial
    /// 2x2 patches have been shuffled into a single feature vector.
    pub fn from_loader_postshuffle(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        spatial_merge_size: i32,
    ) -> Result<Self> {
        Self::from_loader_with_norm_position(loader, prefix, spatial_merge_size, true)
    }

    fn from_loader_with_norm_position(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        spatial_merge_size: i32,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let norm_w = loader.tensor(&format!("{prefix}.norm.weight"))?.clone();
        let norm_b = loader.tensor(&format!("{prefix}.norm.bias"))?.clone();
        let m2 = spatial_merge_size * spatial_merge_size;
        let hidden_size = if use_postshuffle_norm {
            norm_w.shape()[0] / m2
        } else {
            norm_w.shape()[0]
        };
        Ok(Self {
            norm: LayerNorm::new(norm_w, Some(norm_b), 1e-6),
            fc1: VisionProjection::from_loader(loader, &format!("{prefix}.linear_fc1"))?,
            fc2: VisionProjection::from_loader(loader, &format!("{prefix}.linear_fc2"))?,
            spatial_merge_size,
            hidden_size,
            use_postshuffle_norm,
        })
    }

    /// Forward pass on the default stream.
    ///
    /// `x` shape: `[N, hidden_size]` where `N` is total patch count and must
    /// satisfy `N % (spatial_merge_size²) == 0` (caller responsibility — the
    /// vision tower's block stack always produces this).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();

        // Step 1: spatial merge, with LayerNorm either before the shuffle for
        // the final merger or after it for Qwen3-VL DeepStack mergers.
        // m² = spatial_merge_size²
        let n = x.shape()[0];
        let m2 = self.spatial_merge_size * self.spatial_merge_size; // 4
        let merge_hidden = m2 * self.hidden_size; // 4096
        let n_merged = n / m2; // N/4
        let h = if self.use_postshuffle_norm {
            let grouped = ops::shape::reshape(x, &[n_merged, merge_hidden][..])?;
            self.norm.forward(&grouped)?
        } else {
            let normalized = self.norm.forward(x)?;
            ops::shape::reshape(&normalized, &[n_merged, merge_hidden][..])?
        };

        // Step 3: fc1  [N/4, 4096] @ [4096, 4096]^T + bias → [N/4, 4096]
        let h = self.fc1.forward_on(&h, target)?;

        // Step 4: exact GELU (erf-based)
        let h = gelu_exact(&h, target)?;

        // Step 5: fc2  [N/4, 4096] @ [out_hidden, 4096]^T + bias → [N/4, out_hidden]
        self.fc2.forward_on(&h, target)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::constructors;
    use mlx::{Array, Dtype};

    #[test]
    fn patch_merger_output_shape() {
        crate::models::vision::init_test_metallib();
        // grid 2×2 → after 2×2 merge → 1 token, 2560 dim
        let merger = PatchMerger::new(
            constructors::ones((1024_i32,), Dtype::Bfloat16).unwrap(),
            Array::zeros([1024], Dtype::Bfloat16).unwrap(),
            Array::zeros([4096, 4096], Dtype::Bfloat16).unwrap(),
            Array::zeros([4096], Dtype::Bfloat16).unwrap(),
            Array::zeros([2560, 4096], Dtype::Bfloat16).unwrap(),
            Array::zeros([2560], Dtype::Bfloat16).unwrap(),
            2, // spatial_merge_size
        );
        let x = constructors::ones((4_i32, 1024_i32), Dtype::Bfloat16).unwrap();
        let out = merger.forward(&x).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, 2560]);
    }

    #[test]
    fn gelu_exact_zero_maps_to_zero() {
        crate::models::vision::init_test_metallib();
        // gelu_exact(0) = 0.5 * 0 * (1 + erf(0)) = 0
        let zero = Array::try_from((&[0.0_f32][..], &[][..])).unwrap();
        let out = gelu_exact(&zero, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!(
            (v - 0.0_f32).abs() < 1e-6,
            "gelu_exact(0) should be 0, got {v}"
        );
    }

    #[test]
    fn gelu_exact_positive_passes_through() {
        crate::models::vision::init_test_metallib();
        // For large positive x, gelu_exact(x) ≈ x.
        let x = Array::try_from((&[10.0_f32][..], &[][..])).unwrap();
        let out = gelu_exact(&x, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!((v - 10.0_f32).abs() < 0.1, "gelu_exact(10) ≈ 10, got {v}");
    }

    #[test]
    fn patch_merger_zero_weights_produce_zeros() {
        crate::models::vision::init_test_metallib();
        // With all-zero fc1/fc2 weights and biases, output must be all zeros.
        let merger = PatchMerger::new(
            constructors::ones((1024_i32,), Dtype::Float32).unwrap(),
            Array::zeros([1024], Dtype::Float32).unwrap(),
            Array::zeros([4096, 4096], Dtype::Float32).unwrap(),
            Array::zeros([4096], Dtype::Float32).unwrap(),
            Array::zeros([2560, 4096], Dtype::Float32).unwrap(),
            Array::zeros([2560], Dtype::Float32).unwrap(),
            2,
        );
        let x = constructors::ones((4_i32, 1024_i32), Dtype::Float32).unwrap();
        let out = merger.forward(&x).unwrap();
        let vals: Vec<f32> = out.to_vec().unwrap();
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "zero-weight merger should produce zeros, max_abs={max_abs}"
        );
    }
}
