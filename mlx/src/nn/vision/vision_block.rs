use crate::array::Array;
use crate::error::Result;
use crate::fast;
use crate::nn::activations::gelu;
use crate::nn::linear::Linear;
use crate::nn::norm::LayerNorm;
use crate::ops;
use crate::stream::Stream;

/// Vision attention with merged QKV projection (non-causal, global attention)
pub struct VisionAttention {
    pub qkv: Linear,  // [3*hidden, hidden] with bias
    pub proj: Linear, // [hidden, hidden] with bias
    pub num_heads: usize,
    pub head_dim: usize,
}

impl VisionAttention {
    pub fn forward(&self, x: &Array, mask: Option<&Array>, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let hidden = shape[2];
        let n = self.num_heads as i32;
        let d = self.head_dim as i32;

        // QKV projection: [B, L, 3*hidden]
        let qkv = self.qkv.forward_with_stream(x, stream)?;

        // Reshape to [B, L, 3, num_heads, head_dim] then transpose
        let qkv = ops::reshape(&qkv, &[b, l, 3, n, d], stream)?;
        let qkv = ops::transpose_axes(&qkv, &[2, 0, 3, 1, 4], stream)?; // [3, B, n, L, d]

        // Split into Q, K, V
        let q = ops::slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[1, b, n, l, d],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let q = ops::reshape(&q, &[b, n, l, d], stream)?;
        let k = ops::slice(
            &qkv,
            &[1, 0, 0, 0, 0],
            &[2, b, n, l, d],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let k = ops::reshape(&k, &[b, n, l, d], stream)?;
        let v = ops::slice(
            &qkv,
            &[2, 0, 0, 0, 0],
            &[3, b, n, l, d],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let v = ops::reshape(&v, &[b, n, l, d], stream)?;

        // Scaled dot-product attention (non-causal, global)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, "none", mask, None, stream)?;

        // [B, n, L, d] → [B, L, hidden]
        let attn_out = ops::transpose_axes(&attn_out, &[0, 2, 1, 3], stream)?;
        let attn_out = ops::reshape(&attn_out, &[b, l, hidden], stream)?;

        // Output projection
        self.proj.forward_with_stream(&attn_out, stream)
    }
}

/// Vision MLP: fc1 → GELU → fc2
pub struct VisionMLP {
    pub fc1: Linear, // [intermediate_size, hidden_size] with bias
    pub fc2: Linear, // [hidden_size, intermediate_size] with bias
}

impl VisionMLP {
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let h = self.fc1.forward_with_stream(x, stream)?;
        let h = gelu(&h, stream)?;
        self.fc2.forward_with_stream(&h, stream)
    }
}

/// A single vision transformer block
pub struct VisionBlock {
    pub norm1: LayerNorm,
    pub attn: VisionAttention,
    pub norm2: LayerNorm,
    pub mlp: VisionMLP,
}

impl VisionBlock {
    pub fn forward(&self, x: &Array, mask: Option<&Array>, stream: &Stream) -> Result<Array> {
        // x + attn(norm1(x))
        let normed = self.norm1.forward_with_stream(x, stream)?;
        let attn_out = self.attn.forward(&normed, mask, stream)?;
        let h = ops::add(x, &attn_out, stream)?;

        // h + mlp(norm2(h))
        let normed = self.norm2.forward_with_stream(&h, stream)?;
        let mlp_out = self.mlp.forward(&normed, stream)?;
        ops::add(&h, &mlp_out, stream)
    }
}
