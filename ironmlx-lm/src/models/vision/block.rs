//! Qwen3.5 ViT block — norm1 → attn (rotary) → norm2 → mlp.
//! See spec §4.3.

use std::sync::OnceLock;

use anyhow::Result;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::fast::scaled_dot_product_attention;
use mlx::{ops, Array, StreamOrDevice};

use super::projection::VisionProjection;
use crate::nn::activations::{build_gelu_tanh, invoke_gelu_tanh};
use crate::nn::LayerNorm;
use ironmlx_core::weights::WeightSource;

/// Two-layer MLP inside each ViT block.
///
/// Architecture: `linear_fc1` (d_model→4*d_model) → GELU-tanh → `linear_fc2` (4*d_model→d_model).
/// Both layers have bias terms.
pub struct VitMLP {
    fc1: VisionProjection,
    fc2: VisionProjection,
    gelu: OnceLock<CompiledFn>,
}

impl VitMLP {
    /// Construct from pre-loaded weight Arrays.
    ///
    /// `fc1_w` shape: `[ffn_dim, d_model]`, e.g. `[4096, 1024]`.
    /// `fc1_b` shape: `[ffn_dim]`.
    /// `fc2_w` shape: `[d_model, ffn_dim]`, e.g. `[1024, 4096]`.
    /// `fc2_b` shape: `[d_model]`.
    pub fn new(fc1_w: Array, fc1_b: Array, fc2_w: Array, fc2_b: Array) -> Self {
        Self {
            fc1: VisionProjection::new_fp(fc1_w, Some(fc1_b)),
            fc2: VisionProjection::new_fp(fc2_w, Some(fc2_b)),
            gelu: OnceLock::new(),
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names:
    /// - `{prefix}.linear_fc1.weight` / `.bias`
    /// - `{prefix}.linear_fc2.weight` / `.bias`
    pub fn from_loader(loader: &(impl WeightSource + ?Sized), prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: VisionProjection::from_loader(loader, &format!("{prefix}.linear_fc1"))?,
            fc2: VisionProjection::from_loader(loader, &format!("{prefix}.linear_fc2"))?,
            gelu: OnceLock::new(),
        })
    }

    /// Forward pass on the default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    ///
    /// Computes: `fc2(gelu_tanh(fc1(x)))` where each linear is `addmm(bias, x, W^T)`.
    ///
    /// Uses `mx.addmm` (fused bias-matmul) to match mlx-vlm's `nn.Linear` which
    /// internally calls `mx.addmm(bias, x, W.T)`.  This avoids the 1-BF16-ULP
    /// rounding difference that arises when matmul and bias-add are separate ops.
    /// The GELU runs as one compiled kernel (on the default stream), like
    /// mlx-vlm's compiled `gelu_approx`.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();

        // fc1: addmm(b1, x, W1.T) → [T, ffn_dim]
        let h = self.fc1.forward_on(x, target)?;

        // GELU tanh approx
        let h = invoke_gelu_tanh(self.gelu.get_or_init(build_gelu_tanh), &h)?;

        // fc2: addmm(b2, h, W2.T) → [T, d_model]
        self.fc2.forward_on(&h, target)
    }
}

// ---------------------------------------------------------------------------
// Vision rotary helper
// ---------------------------------------------------------------------------

/// The vision rotary embedding of one forward pass, computed once and shared
/// by every block.
///
/// Matches `apply_rotary_pos_emb_vision` from mlx-vlm:
///   - `cos_full = tile(cos(freqs), 2)`, `sin_full = tile(sin(freqs), 2)`
///   - `rotate_half(x) = concat([-x[..., half:], x[..., :half]], axis=-1)`
///   - `out = (x * cos_full + rotate_half(x) * sin_full).astype(orig_dtype)`
///
/// The sign of `rotate_half` is folded into the sine table
/// (`sin_signed = concat([-sin, sin])`), which multiplies the swapped halves
/// `concat([x2, x1])`: `(-x2) * s` and `x2 * (-s)` are the same f32 product.
/// cos/sin stay f32 for the multiply, as in mlx-vlm (MLX promotes bf16×f32 to
/// f32, then the result is cast back), avoiding a premature bf16 cast.
pub struct VisionRotary {
    /// `[seq, 1, head_dim]` f32.
    cos: Array,
    /// `[seq, 1, head_dim]` f32, `concat([-sin, sin])`.
    sin_signed: Array,
}

impl VisionRotary {
    /// From the `[seq, head_dim/2]` f32 frequency table.
    pub fn new(freqs: &Array) -> Result<Self> {
        let shape = freqs.shape();
        let (seq, half) = (shape[0], shape[1]);
        let head_dim = 2 * half;
        let cos_half = ops::cos(freqs)?;
        let sin_half = ops::sin(freqs)?;
        let neg_sin_half = -&sin_half;
        let cos = ops::concatenate(&[&cos_half, &cos_half], 1)?;
        let sin_signed = ops::concatenate(&[&neg_sin_half, &sin_half], 1)?;
        Ok(Self {
            cos: ops::shape::reshape(&cos, &[seq, 1, head_dim][..])?,
            sin_signed: ops::shape::reshape(&sin_signed, &[seq, 1, head_dim][..])?,
        })
    }

    /// Rotates `tensor` (`[seq, num_heads, head_dim]`), with `apply` the
    /// compiled elementwise tail ([`build_rotary_apply`]).
    fn apply(&self, apply: &CompiledFn, tensor: &Array) -> Result<Array> {
        let shape = tensor.shape();
        let (seq, num_heads, head_dim) = (shape[0], shape[1], shape[2]);
        let half = head_dim / 2;
        let x1 = ops::slice(tensor, &[0, 0, 0][..], &[seq, num_heads, half][..])?;
        let x2 = ops::slice(tensor, &[0, 0, half][..], &[seq, num_heads, head_dim][..])?;
        let swapped = ops::concatenate(&[&x2, &x1], 2)?;
        let mut outs = apply
            .invoke(&[tensor, &swapped, &self.cos, &self.sin_signed])
            .map_err(|e| anyhow::anyhow!("vision rotary invoke failed: {e}"))?;
        outs.pop()
            .ok_or_else(|| anyhow::anyhow!("vision rotary returned no outputs"))
    }
}

/// `(x * cos + swapped * sin_signed).astype(x.dtype)` as one compiled kernel,
/// so the f32 products are never materialized.
fn build_rotary_apply() -> CompiledFn {
    compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let (x, swapped, cos, sin) = (inputs[0], inputs[1], inputs[2], inputs[3]);
            let out = &(x * cos) + &(swapped * sin);
            Ok(vec![ops::cast::astype(&out, x.dtype())?])
        },
        ShapeMode::Shapeless,
    )
    .expect("vision rotary compile")
}

/// Head dims MLX's NAX full-attention kernel takes
/// (`sdpa_full_self_attention_metal`, MLX 73ad5df2).
const NAX_FULL_ATTENTION_HEAD_DIMS: [i32; 3] = [64, 96, 128];

/// The head dim to zero-pad q, k and v to so that MLX runs its NAX
/// full-attention kernel (bf16/f16, query length > 8) instead of its
/// non-NAX one, which is far slower on chips with NAX: the smallest NAX head
/// dim above `head_dim`, when NAX is available and `head_dim` is not one.
/// Zero columns add nothing to `q·k`, so with the scale kept at
/// `1/sqrt(head_dim)` the first `head_dim` output columns are the attention
/// output (up to the kernels' accumulation order).
fn nax_padded_head_dim(head_dim: i32) -> Option<i32> {
    static NAX: OnceLock<bool> = OnceLock::new();
    if NAX_FULL_ATTENTION_HEAD_DIMS.contains(&head_dim)
        || !*NAX.get_or_init(|| mlx::metal::nax_available().unwrap_or(false))
    {
        return None;
    }
    NAX_FULL_ATTENTION_HEAD_DIMS
        .into_iter()
        .find(|&dim| dim > head_dim)
}

// ---------------------------------------------------------------------------
// VitAttention
// ---------------------------------------------------------------------------

/// Self-attention inside each ViT block.
///
/// Architecture (matches `Attention` in mlx-vlm/qwen3_vl/vision.py):
///   1. Fused QKV projection: `[seq, dim] → [seq, 3*dim]`
///   2. Reshape + transpose to `[3, seq, num_heads, head_dim]`, split Q/K/V
///   3. Apply vision rotary to Q and K (NOT V)
///   4. Transpose to `[num_heads, seq, head_dim]`
///   5. For each cu_seqlens segment: fused SDPA (`scale = 1/sqrt(head_dim)`)
///   6. Concat, reshape, output projection
pub struct VitAttention {
    qkv: VisionProjection,
    projection: VisionProjection,
    num_heads: i32,
    head_dim: i32,
    rotary_apply: OnceLock<CompiledFn>,
}

impl VitAttention {
    /// Construct from pre-loaded weight arrays.
    ///
    /// `qkv_w` shape: `[3*dim, dim]`, `qkv_b`: `[3*dim]`.
    /// `proj_w` shape: `[dim, dim]`, `proj_b`: `[dim]`.
    pub fn new(
        qkv_w: Array,
        qkv_b: Array,
        proj_w: Array,
        proj_b: Array,
        num_heads: i32,
        head_dim: i32,
    ) -> Self {
        Self {
            qkv: VisionProjection::new_fp(qkv_w, Some(qkv_b)),
            projection: VisionProjection::new_fp(proj_w, Some(proj_b)),
            num_heads,
            head_dim,
            rotary_apply: OnceLock::new(),
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names:
    /// - `{prefix}.qkv.weight` / `.bias`
    /// - `{prefix}.proj.weight` / `.bias`
    pub fn from_loader(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            qkv: VisionProjection::from_loader(loader, &format!("{prefix}.qkv"))?,
            projection: VisionProjection::from_loader(loader, &format!("{prefix}.proj"))?,
            num_heads,
            head_dim,
            rotary_apply: OnceLock::new(),
        })
    }

    /// Forward pass on the default stream.
    ///
    /// `rotary`: the forward pass's rotary embedding.
    /// `cu_seqlens`: cumulative sequence lengths, e.g. `[0, seq]` for single
    ///   image or `[0, s1, s1+s2, ...]` for multi-image batch.
    pub fn forward(&self, x: &Array, rotary: &VisionRotary, cu_seqlens: &[i32]) -> Result<Array> {
        let pad_to = nax_padded_head_dim(self.head_dim).filter(|_| {
            x.dtype() != mlx::Dtype::Float32 && cu_seqlens.windows(2).all(|w| w[1] - w[0] > 8)
        });
        self.forward_padded(x, rotary, cu_seqlens, pad_to)
    }

    /// [`Self::forward`] with the q/k/v head dim zero-padded to `pad_to`
    /// before attention (`None`: unpadded).
    fn forward_padded(
        &self,
        x: &Array,
        rotary: &VisionRotary,
        cu_seqlens: &[i32],
        pad_to: Option<i32>,
    ) -> Result<Array> {
        let seq = x.shape()[0];
        let nh = self.num_heads;
        let hd = self.head_dim;
        let scale = 1.0_f32 / (hd as f32).sqrt();

        // Step 1: fused QKV projection → [seq, 3*dim]
        // Use addmm (fused bias-matmul) to match mlx-vlm's nn.Linear which calls
        // mx.addmm(bias, x, W.T).  This eliminates the 1-ULP rounding difference
        // that would arise from separate matmul + bias-add.
        let qkv = self.qkv.forward_on(x, ().into())?;

        // Step 2: reshape to [seq, 3, nh, hd] then transpose(1,0,2,3) → [3, seq, nh, hd]
        let qkv = ops::shape::reshape(&qkv, &[seq, 3, nh, hd][..])?;
        let qkv = ops::shape::transpose_axes(&qkv, &[1, 0, 2, 3][..])?;

        // Step 3: split into Q, K, V each [1, seq, nh, hd] then squeeze axis 0 → [seq, nh, hd]
        let parts = ops::shape::split_n(&qkv, 3, 0)?;
        let q = ops::shape::squeeze(&parts[0], &[0][..])?;
        let k = ops::shape::squeeze(&parts[1], &[0][..])?;
        let v = ops::shape::squeeze(&parts[2], &[0][..])?;

        // Step 4: apply rotary to Q and K
        let rotary_apply = self.rotary_apply.get_or_init(build_rotary_apply);
        let q = rotary.apply(rotary_apply, &q)?;
        let k = rotary.apply(rotary_apply, &k)?;

        // Step 5: transpose to [nh, seq, hd] then unsqueeze batch → [1, nh, seq, hd] for SDPA
        // MLX fast SDPA requires rank-4 input: [batch, heads, seq, head_dim]
        let q = ops::shape::transpose_axes(&q, &[1, 0, 2][..])?;
        let k = ops::shape::transpose_axes(&k, &[1, 0, 2][..])?;
        let v = ops::shape::transpose_axes(&v, &[1, 0, 2][..])?;
        let q = ops::shape::expand_dims(&q, &[0][..])?;
        let k = ops::shape::expand_dims(&k, &[0][..])?;
        let v = ops::shape::expand_dims(&v, &[0][..])?;

        // Step 5b: zero-pad the head dim for MLX's NAX attention kernel.
        let (q, k, v) = match pad_to {
            Some(padded) => {
                let zeros = Array::zeros(&[1, nh, seq, padded - hd][..], q.dtype())?;
                (
                    ops::concatenate(&[&q, &zeros], 3)?,
                    ops::concatenate(&[&k, &zeros], 3)?,
                    ops::concatenate(&[&v, &zeros], 3)?,
                )
            }
            None => (q, k, v),
        };

        // Step 6: SDPA per cu_seqlens segment
        // q/k/v shape: [1, nh, seq, hd]
        // Split along axis=2 (seq axis) per cu_seqlens boundaries (cu_seqlens[1..-1])
        let n_segs = cu_seqlens.len() - 1;
        let output = if n_segs == 1 {
            // Single segment — no loop overhead
            scaled_dot_product_attention(&q, &k, &v, scale, "", None, None)?
        } else {
            // Multi-segment: split q/k/v along axis=2 (seq axis)
            let split_indices: Vec<i32> = cu_seqlens[1..cu_seqlens.len() - 1].to_vec();
            let qs = ops::shape::split_at(&q, &split_indices, 2)?;
            let ks = ops::shape::split_at(&k, &split_indices, 2)?;
            let vs = ops::shape::split_at(&v, &split_indices, 2)?;

            let mut seg_outputs: Vec<Array> = Vec::with_capacity(n_segs);
            for i in 0..n_segs {
                let seg_out =
                    scaled_dot_product_attention(&qs[i], &ks[i], &vs[i], scale, "", None, None)?;
                seg_outputs.push(seg_out);
            }
            let refs: Vec<&Array> = seg_outputs.iter().collect();
            ops::shape::concatenate(&refs, 2)?
        };

        // Drop the padded head-dim columns.
        let output = match pad_to {
            Some(_) => ops::slice(&output, &[0, 0, 0, 0][..], &[1, nh, seq, hd][..])?,
            None => output,
        };

        // Step 7: squeeze batch dim → [nh, seq, hd], transpose → [seq, nh, hd], reshape [seq, dim]
        let output = ops::shape::squeeze(&output, &[0][..])?;
        let output = ops::shape::transpose_axes(&output, &[1, 0, 2][..])?;
        let dim = nh * hd;
        let output = ops::shape::reshape(&output, &[seq, dim][..])?;

        // Step 8: output projection (fused addmm to match nn.Linear)
        self.projection.forward_on(&output, ().into())
    }
}

// ---------------------------------------------------------------------------
// VitBlock
// ---------------------------------------------------------------------------

/// One pre-norm transformer block inside the ViT.
///
/// Implements the standard residual pattern:
/// ```text
/// h = x + attn(norm1(x), rotary, cu_seqlens)
/// y = h + mlp(norm2(h))
/// ```
/// Both `norm1` and `norm2` are `LayerNorm` with eps = 1e-6 (Qwen3.5 config).
pub struct VitBlock {
    norm1: LayerNorm,
    attn: VitAttention,
    norm2: LayerNorm,
    mlp: VitMLP,
}

impl VitBlock {
    /// Construct from pre-built sub-modules.
    pub fn new(norm1: LayerNorm, attn: VitAttention, norm2: LayerNorm, mlp: VitMLP) -> Self {
        Self {
            norm1,
            attn,
            norm2,
            mlp,
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names (under `prefix`):
    /// - `{prefix}.norm1.weight` / `.bias`
    /// - `{prefix}.attn.qkv.weight` / `.bias`
    /// - `{prefix}.attn.proj.weight` / `.bias`
    /// - `{prefix}.norm2.weight` / `.bias`
    /// - `{prefix}.mlp.linear_fc1.weight` / `.bias`
    /// - `{prefix}.mlp.linear_fc2.weight` / `.bias`
    pub fn from_loader(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let norm1 = LayerNorm::from_loader(loader, &format!("{prefix}.norm1"), 1e-6)?;
        let attn =
            VitAttention::from_loader(loader, &format!("{prefix}.attn"), num_heads, head_dim)?;
        let norm2 = LayerNorm::from_loader(loader, &format!("{prefix}.norm2"), 1e-6)?;
        let mlp = VitMLP::from_loader(loader, &format!("{prefix}.mlp"))?;
        Ok(Self::new(norm1, attn, norm2, mlp))
    }

    /// Forward pass on the default stream.
    ///
    /// `rotary`: the forward pass's rotary embedding.
    /// `cu_seqlens`: cumulative sequence lengths, e.g. `[0, seq]` for a single image.
    pub fn forward(&self, x: &Array, rotary: &VisionRotary, cu_seqlens: &[i32]) -> Result<Array> {
        // h = x + attn(norm1(x))
        let normed1 = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed1, rotary, cu_seqlens)?;
        let h = x + &attn_out;

        // y = h + mlp(norm2(h))
        let normed2 = self.norm2.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(&h + &mlp_out)
    }

    /// Like [`Self::forward`] but emits 4 intra-block dumps (post-norm1, attn-residual,
    /// post-norm2, mlp-residual) under the given name prefix. The non-feature
    /// build erases the dump calls; in feature builds the dumps fire only if
    /// `IRONMLX_VISION_DUMP_DIR` is set. Used by the P6.3b op-level diff path.
    pub fn forward_with_name_prefix(
        &self,
        x: &Array,
        rotary: &VisionRotary,
        cu_seqlens: &[i32],
        name_prefix: &str,
    ) -> Result<Array> {
        use crate::models::vision::dump::dump_tensor;
        // h = x + attn(norm1(x))
        let normed1 = self.norm1.forward(x)?;
        dump_tensor(&format!("{name_prefix}_a_norm1_out"), &normed1);
        let attn_out = self.attn.forward(&normed1, rotary, cu_seqlens)?;
        let h = x + &attn_out;
        dump_tensor(&format!("{name_prefix}_b_attn_residual"), &h);

        // y = h + mlp(norm2(h))
        let normed2 = self.norm2.forward(&h)?;
        dump_tensor(&format!("{name_prefix}_c_norm2_out"), &normed2);
        let mlp_out = self.mlp.forward(&normed2)?;
        let out = &h + &mlp_out;
        dump_tensor(&format!("{name_prefix}_d_mlp_residual"), &out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::gelu_tanh;
    use mlx::{Array, Dtype};

    #[test]
    fn vit_mlp_output_shape() {
        crate::models::vision::init_test_metallib();
        let fc1_w = Array::zeros([4096, 1024], Dtype::Bfloat16).unwrap();
        let fc1_b = Array::zeros([4096], Dtype::Bfloat16).unwrap();
        let fc2_w = Array::zeros([1024, 4096], Dtype::Bfloat16).unwrap();
        let fc2_b = Array::zeros([1024], Dtype::Bfloat16).unwrap();
        let mlp = VitMLP::new(fc1_w, fc1_b, fc2_w, fc2_b);
        let x = Array::zeros([4, 1024], Dtype::Bfloat16).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape().as_slice(), &[4, 1024]);
    }

    #[test]
    fn gelu_tanh_zero_maps_to_zero() {
        crate::models::vision::init_test_metallib();
        // gelu_tanh(0) = 0.5 * 0 * (1 + tanh(0)) = 0
        let zero = Array::try_from((&[0.0_f32][..], &[][..])).unwrap();
        let out = gelu_tanh(&zero, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!(
            (v - 0.0_f32).abs() < 1e-6,
            "gelu_tanh(0) should be 0, got {v}"
        );
    }

    #[test]
    fn gelu_tanh_positive_passes_through() {
        crate::models::vision::init_test_metallib();
        // For large positive x, gelu_tanh(x) ≈ x.
        let x = Array::try_from((&[10.0_f32][..], &[][..])).unwrap();
        let out = gelu_tanh(&x, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!((v - 10.0_f32).abs() < 0.1, "gelu_tanh(10) ≈ 10, got {v}");
    }

    /// Helper: compute max absolute difference between two Arrays (both cast to f32).
    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let a32 = ops::astype(a, mlx::Dtype::Float32).expect("astype a");
        let b32 = ops::astype(b, mlx::Dtype::Float32).expect("astype b");
        let av: Vec<f32> = a32.to_vec().expect("a to_vec");
        let bv: Vec<f32> = b32.to_vec().expect("b to_vec");
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// VitBlock with all-zero attn/mlp weights must pass residuals through unchanged.
    ///
    /// Zero-weight attn proj → attn output = zeros → h = x + 0 = x.
    /// Zero-weight mlp fc1 → mlp output = zeros (post-GELU also zero) → y = h + 0 = x.
    #[test]
    fn vit_block_residual_connections() {
        crate::models::vision::init_test_metallib();
        let n1_w = ops::constructors::ones((1024_i32,), Dtype::Bfloat16).unwrap();
        let n1_b = Array::zeros([1024], Dtype::Bfloat16).unwrap();
        let norm1 = LayerNorm::new(n1_w, Some(n1_b), 1e-6);
        let norm2 = LayerNorm::new(
            ops::constructors::ones((1024_i32,), Dtype::Bfloat16).unwrap(),
            Some(Array::zeros([1024], Dtype::Bfloat16).unwrap()),
            1e-6,
        );

        let attn = VitAttention::new(
            Array::zeros([3072, 1024], Dtype::Bfloat16).unwrap(),
            Array::zeros([3072], Dtype::Bfloat16).unwrap(),
            Array::zeros([1024, 1024], Dtype::Bfloat16).unwrap(),
            Array::zeros([1024], Dtype::Bfloat16).unwrap(),
            16,
            64,
        );
        let mlp = VitMLP::new(
            Array::zeros([4096, 1024], Dtype::Bfloat16).unwrap(),
            Array::zeros([4096], Dtype::Bfloat16).unwrap(),
            Array::zeros([1024, 4096], Dtype::Bfloat16).unwrap(),
            Array::zeros([1024], Dtype::Bfloat16).unwrap(),
        );
        let block = VitBlock::new(norm1, attn, norm2, mlp);

        let x = ops::constructors::ones((8_i32, 1024_i32), Dtype::Bfloat16).unwrap();
        let rotary = VisionRotary::new(&Array::zeros([8, 32], Dtype::Float32).unwrap()).unwrap();
        let cu = vec![0_i32, 8];
        let out = block.forward(&x, &rotary, &cu).unwrap();
        // zero-weight attn + mlp ⇒ residual passes through ⇒ out == x
        let diff = max_abs_diff(&out, &x);
        println!("vit_block_residual max_diff = {diff:.6}");
        assert!(diff < 1e-3, "residual pass-through failed: max_diff={diff}");
    }

    /// Verify VitAttention forward matches mlx-vlm reference to within bf16 tolerance.
    ///
    /// Fixture format: safetensors (chosen because mlx::io::load_safetensors is available;
    /// mlx::io has no npz loader, so safetensors is the next best single-file format).
    ///
    /// Fixture generated with:
    ///   `~/.venvs/mlxvlm-ref/bin/python` + `mx.save_safetensors(...)`
    /// Keys: x, rotary, qkv_w, qkv_b, proj_w, proj_b, out
    #[test]
    fn vit_attention_matches_mlx_vlm_reference() {
        crate::models::vision::init_test_metallib();
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ironmlx/tests/fixtures/qwen35_vl/p6_vit_attn_ref.safetensors"
        );

        let (tensors, _meta) =
            mlx::io::load_safetensors(fixture_path).expect("failed to load vit_attn fixture");

        let x = tensors.get("x").expect("missing x");
        let rotary = tensors.get("rotary").expect("missing rotary");
        let qkv_w = tensors.get("qkv_w").expect("missing qkv_w").clone();
        let qkv_b = tensors.get("qkv_b").expect("missing qkv_b").clone();
        let proj_w = tensors.get("proj_w").expect("missing proj_w").clone();
        let proj_b = tensors.get("proj_b").expect("missing proj_b").clone();
        let expected = tensors.get("out").expect("missing out");

        let attn = VitAttention::new(qkv_w, qkv_b, proj_w, proj_b, 16, 64);

        // cu_seqlens = [0, 8] — single image with 8 tokens
        let cu_seqlens = [0_i32, 8];
        let got = attn
            .forward(x, &VisionRotary::new(rotary).unwrap(), &cu_seqlens)
            .expect("VitAttention::forward failed");

        let got_f32_arr = ops::astype(&got, mlx::Dtype::Float32).expect("astype got");
        let exp_f32_arr = ops::astype(expected, mlx::Dtype::Float32).expect("astype expected");
        let got_f32: Vec<f32> = got_f32_arr.to_vec::<f32>().expect("got to_vec");
        let exp_f32: Vec<f32> = exp_f32_arr.to_vec::<f32>().expect("expected to_vec");

        assert_eq!(
            got_f32.len(),
            exp_f32.len(),
            "output size mismatch: {} vs {}",
            got_f32.len(),
            exp_f32.len()
        );

        let max_diff = got_f32
            .iter()
            .zip(exp_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("VitAttention max_diff vs mlx-vlm ref: {max_diff:.6}");
        assert!(
            max_diff < 0.05,
            "VitAttention output diverges from mlx-vlm reference: max_diff={max_diff}"
        );
    }

    fn random(shape: &[i32], dtype: Dtype, seed: u64, scale: f64) -> Array {
        let key = mlx::random::key(seed).unwrap();
        let x = mlx::random::normal()
            .shape(shape)
            .dtype(Dtype::Float32)
            .scale(scale)
            .key(&key)
            .sample()
            .unwrap();
        ops::astype(&x, dtype).unwrap()
    }

    fn bits(a: &Array) -> Vec<u16> {
        let a = ops::astype(a, Dtype::Float32).unwrap();
        a.to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect()
    }

    /// The compiled rotary (cos/sin computed once, sign folded into the sine
    /// table) gives mlx-vlm's `apply_rotary_pos_emb_vision` bit for bit, as
    /// written eagerly with a per-call cos/sin.
    #[test]
    fn compiled_rotary_matches_the_eager_formula_bit_for_bit() {
        crate::models::vision::init_test_metallib();
        let (seq, heads, head_dim) = (40, 4, 72);
        let tensor = random(&[seq, heads, head_dim], Dtype::Bfloat16, 1, 1.0);
        let freqs = random(&[seq, head_dim / 2], Dtype::Float32, 2, 30.0);
        let rotary = VisionRotary::new(&freqs).unwrap();
        let got = rotary.apply(&build_rotary_apply(), &tensor).unwrap();

        let half = head_dim / 2;
        let cos_half = ops::cos(&freqs).unwrap();
        let sin_half = ops::sin(&freqs).unwrap();
        let cos = ops::concatenate(&[&cos_half, &cos_half], 1).unwrap();
        let sin = ops::concatenate(&[&sin_half, &sin_half], 1).unwrap();
        let cos = ops::shape::reshape(&cos, &[seq, 1, head_dim][..]).unwrap();
        let sin = ops::shape::reshape(&sin, &[seq, 1, head_dim][..]).unwrap();
        let x1 = ops::slice(&tensor, &[0, 0, 0][..], &[seq, heads, half][..]).unwrap();
        let x2 = ops::slice(&tensor, &[0, 0, half][..], &[seq, heads, head_dim][..]).unwrap();
        let rotated = ops::concatenate(&[&(-&x2), &x1], 2).unwrap();
        let eager = &(&tensor * &cos) + &(&rotated * &sin);
        let eager = ops::astype(&eager, Dtype::Bfloat16).unwrap();
        assert_eq!(got.dtype(), Dtype::Bfloat16);
        assert_eq!(bits(&got), bits(&eager));
    }

    /// The compiled GELU gives the eager op-by-op GELU bit for bit.
    #[test]
    fn compiled_gelu_matches_the_eager_gelu_bit_for_bit() {
        crate::models::vision::init_test_metallib();
        let x = random(&[64, 300], Dtype::Bfloat16, 3, 4.0);
        let compiled = invoke_gelu_tanh(&build_gelu_tanh(), &x).unwrap();
        let eager = gelu_tanh(&x, ().into()).unwrap();
        assert_eq!(compiled.dtype(), Dtype::Bfloat16);
        assert_eq!(bits(&compiled), bits(&eager));
    }

    /// Zero-padding the head dim (72 to 96, as for MLX's NAX attention) leaves
    /// the attention output unchanged up to bf16 accumulation noise, and
    /// padding is only chosen for head dims MLX's NAX kernel does not take.
    #[test]
    fn padded_head_dim_attention_matches_unpadded() {
        crate::models::vision::init_test_metallib();
        let (heads, head_dim, seq) = (4, 72, 48);
        let dim = heads * head_dim;
        let attn = VitAttention::new(
            random(&[3 * dim, dim], Dtype::Bfloat16, 4, 0.05),
            random(&[3 * dim], Dtype::Bfloat16, 5, 0.05),
            random(&[dim, dim], Dtype::Bfloat16, 6, 0.05),
            random(&[dim], Dtype::Bfloat16, 7, 0.05),
            heads,
            head_dim,
        );
        let x = random(&[seq, dim], Dtype::Bfloat16, 8, 1.0);
        let rotary =
            VisionRotary::new(&random(&[seq, head_dim / 2], Dtype::Float32, 9, 10.0)).unwrap();
        for cu in [vec![0, seq], vec![0, 20, seq]] {
            let plain = attn.forward_padded(&x, &rotary, &cu, None).unwrap();
            let padded = attn.forward_padded(&x, &rotary, &cu, Some(96)).unwrap();
            let diff = max_abs_diff(&plain, &padded);
            let peak = max_abs_diff(&plain, &ops::zeros_like(&plain).unwrap());
            assert!(
                diff <= 0.02 * peak,
                "padded attention differs: {diff} (output max {peak})"
            );
        }
        assert_eq!(nax_padded_head_dim(64), None);
        assert_eq!(nax_padded_head_dim(96), None);
        assert_eq!(nax_padded_head_dim(128), None);
        let nax = mlx::metal::nax_available().unwrap_or(false);
        assert_eq!(nax_padded_head_dim(72), nax.then_some(96));
        assert_eq!(nax_padded_head_dim(80), nax.then_some(96));
    }
}
