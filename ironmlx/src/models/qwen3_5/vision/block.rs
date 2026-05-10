//! Qwen3.5 ViT block — norm1 → attn (rotary) → norm2 → mlp.
//! See spec §4.3.

use anyhow::Result;
use mlx::fast::scaled_dot_product_attention;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;

// sqrt(2/π) = 0.7978845608028654  (tanh GELU approximation constant)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// GELU with tanh approximation (matches PyTorch `approximate="tanh"` / mlx-vlm `gelu_approx`).
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
///
/// There is no built-in `gelu_approx` in the mlx Rust bindings, so this is
/// hand-rolled from the exact polynomial formula used by mlx-vlm / PyTorch.
fn gelu_tanh(x: &Array, target: StreamOrDevice) -> Result<Array> {
    // x^3 via x * x * x  (avoid power() to stay on the fast path)
    let x2 = x * x;
    let x3 = &x2 * x;
    // inner = sqrt(2/π) * (x + 0.044715 * x^3)
    let inner = (&x3 * 0.044_715_f32 + x) * SQRT_2_OVER_PI;
    // tanh(inner)
    let t = inner.tanh_on(target)?;
    // 0.5 * x * (1 + t)
    let out = x * 0.5_f32 * (&t + 1.0_f32);
    Ok(out)
}

/// Two-layer MLP inside each ViT block.
///
/// Architecture: `linear_fc1` (d_model→4*d_model) → GELU-tanh → `linear_fc2` (4*d_model→d_model).
/// Both layers have bias terms.
pub struct VitMLP {
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
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
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names:
    /// - `{prefix}.linear_fc1.weight` / `.bias`
    /// - `{prefix}.linear_fc2.weight` / `.bias`
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let fc1_w = loader
            .tensor(&format!("{prefix}.linear_fc1.weight"))?
            .clone();
        let fc1_b = loader.tensor(&format!("{prefix}.linear_fc1.bias"))?.clone();
        let fc2_w = loader
            .tensor(&format!("{prefix}.linear_fc2.weight"))?
            .clone();
        let fc2_b = loader.tensor(&format!("{prefix}.linear_fc2.bias"))?.clone();
        Ok(Self::new(fc1_w, fc1_b, fc2_w, fc2_b))
    }

    /// Forward pass on the default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    ///
    /// Computes: `fc2(gelu_tanh(fc1(x)))` where each linear is `x @ W^T + b`.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();

        // fc1: [T, d_model] @ [d_model, ffn_dim] + bias  →  [T, ffn_dim]
        let wt1 = self.fc1_w.transpose_on(target)?;
        let h = x.matmul_on(&wt1, target)?;
        let h = &h + &self.fc1_b;

        // GELU tanh approx
        let h = gelu_tanh(&h, target)?;

        // fc2: [T, ffn_dim] @ [ffn_dim, d_model] + bias  →  [T, d_model]
        let wt2 = self.fc2_w.transpose_on(target)?;
        let out = h.matmul_on(&wt2, target)?;
        let out = &out + &self.fc2_b;

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Vision rotary helper
// ---------------------------------------------------------------------------

/// Apply vision rotary position embedding to a tensor of shape
/// `[seq, num_heads, head_dim]`.
///
/// `freqs` shape: `[seq, head_dim/2]` (float32 frequency table).
///
/// Matches `apply_rotary_pos_emb_vision` from mlx-vlm:
///   - Build `cos_full = tile(cos(freqs), 2)` → `[seq, 1, head_dim]`
///   - `rotate_half(x) = concat([-x[..., half:], x[..., :half]], axis=-1)`
///   - `out = x * cos_full + rotate_half(x) * sin_full`
fn apply_rotary_vision(tensor: &Array, freqs: &Array) -> Result<Array> {
    // tensor: [seq, num_heads, head_dim]
    // freqs:  [seq, head_dim/2]
    let shape = tensor.shape();
    let seq = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];
    let half = head_dim / 2;

    // cos_half: [seq, half], sin_half: [seq, half]
    let cos_half = ops::cos(freqs)?;
    let sin_half = ops::sin(freqs)?;

    // tile along axis=1: [seq, half] → [seq, half*2] via concat
    let cos_full = ops::concatenate(&[&cos_half, &cos_half], 1)?;
    let sin_full = ops::concatenate(&[&sin_half, &sin_half], 1)?;

    // expand to [seq, 1, head_dim] for broadcasting with [seq, num_heads, head_dim]
    let cos_bc = ops::shape::reshape(&cos_full, &[seq, 1, head_dim][..])?;
    let sin_bc = ops::shape::reshape(&sin_full, &[seq, 1, head_dim][..])?;

    // x1 = tensor[..., :half], x2 = tensor[..., half:]
    // slice requires ndim-length start/stop
    let x1 = ops::slice(tensor, &[0, 0, 0][..], &[seq, num_heads, half][..])?;
    let x2 = ops::slice(tensor, &[0, 0, half][..], &[seq, num_heads, head_dim][..])?;

    // rotate_half = concat([-x2, x1], axis=-1)
    let neg_x2 = -&x2;
    let rotated = ops::concatenate(&[&neg_x2, &x1], 2)?;

    // Cast cos/sin to tensor dtype for bf16 compat
    let cos_bc = ops::astype(&cos_bc, tensor.dtype())?;
    let sin_bc = ops::astype(&sin_bc, tensor.dtype())?;

    let out = tensor * &cos_bc + &rotated * &sin_bc;
    Ok(out)
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
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    num_heads: i32,
    head_dim: i32,
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
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
            num_heads,
            head_dim,
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names:
    /// - `{prefix}.qkv.weight` / `.bias`
    /// - `{prefix}.proj.weight` / `.bias`
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let qkv_w = loader.tensor(&format!("{prefix}.qkv.weight"))?.clone();
        let qkv_b = loader.tensor(&format!("{prefix}.qkv.bias"))?.clone();
        let proj_w = loader.tensor(&format!("{prefix}.proj.weight"))?.clone();
        let proj_b = loader.tensor(&format!("{prefix}.proj.bias"))?.clone();
        Ok(Self::new(qkv_w, qkv_b, proj_w, proj_b, num_heads, head_dim))
    }

    /// Forward pass on the default stream.
    ///
    /// `rotary_pos_emb`: `[seq, head_dim/2]` float32 frequency table.
    /// `cu_seqlens`: cumulative sequence lengths, e.g. `[0, seq]` for single
    ///   image or `[0, s1, s1+s2, ...]` for multi-image batch.
    pub fn forward(&self, x: &Array, rotary_pos_emb: &Array, cu_seqlens: &[i32]) -> Result<Array> {
        let seq = x.shape()[0];
        let nh = self.num_heads;
        let hd = self.head_dim;
        let scale = 1.0_f32 / (hd as f32).sqrt();

        // Step 1: fused QKV projection → [seq, 3*dim]
        let qkv_wt = self.qkv_w.transpose_on(())?;
        let qkv = x.matmul_on(&qkv_wt, ())?;
        let qkv = &qkv + &self.qkv_b;

        // Step 2: reshape to [seq, 3, nh, hd] then transpose(1,0,2,3) → [3, seq, nh, hd]
        let qkv = ops::shape::reshape(&qkv, &[seq, 3, nh, hd][..])?;
        let qkv = ops::shape::transpose_axes(&qkv, &[1, 0, 2, 3][..])?;

        // Step 3: split into Q, K, V each [1, seq, nh, hd] then squeeze axis 0 → [seq, nh, hd]
        let parts = ops::shape::split_n(&qkv, 3, 0)?;
        let q = ops::shape::squeeze(&parts[0], &[0][..])?;
        let k = ops::shape::squeeze(&parts[1], &[0][..])?;
        let v = ops::shape::squeeze(&parts[2], &[0][..])?;

        // Step 4: apply rotary to Q and K
        let q = apply_rotary_vision(&q, rotary_pos_emb)?;
        let k = apply_rotary_vision(&k, rotary_pos_emb)?;

        // Step 5: transpose to [nh, seq, hd] then unsqueeze batch → [1, nh, seq, hd] for SDPA
        // MLX fast SDPA requires rank-4 input: [batch, heads, seq, head_dim]
        let q = ops::shape::transpose_axes(&q, &[1, 0, 2][..])?;
        let k = ops::shape::transpose_axes(&k, &[1, 0, 2][..])?;
        let v = ops::shape::transpose_axes(&v, &[1, 0, 2][..])?;
        let q = ops::shape::expand_dims(&q, &[0][..])?;
        let k = ops::shape::expand_dims(&k, &[0][..])?;
        let v = ops::shape::expand_dims(&v, &[0][..])?;

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

        // Step 7: squeeze batch dim → [nh, seq, hd], transpose → [seq, nh, hd], reshape [seq, dim]
        let output = ops::shape::squeeze(&output, &[0][..])?;
        let output = ops::shape::transpose_axes(&output, &[1, 0, 2][..])?;
        let dim = nh * hd;
        let output = ops::shape::reshape(&output, &[seq, dim][..])?;

        // Step 8: output projection
        let proj_wt = self.proj_w.transpose_on(())?;
        let out = output.matmul_on(&proj_wt, ())?;
        let out = &out + &self.proj_b;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn vit_mlp_output_shape() {
        let fc1_w = Array::zeros(&[4096, 1024], Dtype::Bfloat16).unwrap();
        let fc1_b = Array::zeros(&[4096], Dtype::Bfloat16).unwrap();
        let fc2_w = Array::zeros(&[1024, 4096], Dtype::Bfloat16).unwrap();
        let fc2_b = Array::zeros(&[1024], Dtype::Bfloat16).unwrap();
        let mlp = VitMLP::new(fc1_w, fc1_b, fc2_w, fc2_b);
        let x = Array::zeros(&[4, 1024], Dtype::Bfloat16).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape().as_slice(), &[4, 1024]);
    }

    #[test]
    fn gelu_tanh_zero_maps_to_zero() {
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
        // For large positive x, gelu_tanh(x) ≈ x.
        let x = Array::try_from((&[10.0_f32][..], &[][..])).unwrap();
        let out = gelu_tanh(&x, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!((v - 10.0_f32).abs() < 0.1, "gelu_tanh(10) ≈ 10, got {v}");
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
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/p6_qwen35_vl/p6_vit_attn_ref.safetensors"
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
            .forward(x, rotary, &cu_seqlens)
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
}
