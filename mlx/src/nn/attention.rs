use super::linear::Linear;
use super::module::Module;
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::fast;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;
use std::collections::HashMap;

pub struct Attention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_traditional: bool,
    pub rope_base: Option<f32>,
    pub rope_scale: f32,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_traditional: bool,
        rope_base: Option<f32>,
        rope_scale: f32,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_traditional,
            rope_base,
            rope_scale,
        }
    }

    /// Forward with explicit cache. Returns (output, new_keys, new_values).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache_keys: Option<&Array>,
        cache_values: Option<&Array>,
        offset: i32,
        mask_mode: &str,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<(Array, Array, Array)> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // QKV projections
        let q = self.wq.forward_with_stream(x, stream)?;
        let k = self.wk.forward_with_stream(x, stream)?;
        let v = self.wv.forward_with_stream(x, stream)?;

        // Reshape: [B, L, n_heads * head_dim] -> [B, L, n_heads, head_dim]
        let q = ops::reshape(&q, &[batch, seq_len, self.n_heads, self.head_dim], stream)?;
        let k = ops::reshape(
            &k,
            &[batch, seq_len, self.n_kv_heads, self.head_dim],
            stream,
        )?;
        let v = ops::reshape(
            &v,
            &[batch, seq_len, self.n_kv_heads, self.head_dim],
            stream,
        )?;

        // Transpose to [B, n_heads, L, head_dim]
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3], stream)?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3], stream)?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3], stream)?;

        // RoPE
        let q = fast::rope(
            &q,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            self.rope_scale,
            offset,
            None,
            stream,
        )?;
        let k = fast::rope(
            &k,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            self.rope_scale,
            offset,
            None,
            stream,
        )?;

        // KV cache update
        let (k, v) = if let (Some(ck), Some(cv)) = (cache_keys, cache_values) {
            let arrays_k = VectorArray::from_arrays(&[ck, &k]);
            let arrays_v = VectorArray::from_arrays(&[cv, &v]);
            let new_k = ops::concatenate(&arrays_k, 2, stream)?; // concat along seq dim
            let new_v = ops::concatenate(&arrays_v, 2, stream)?;
            (new_k, new_v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, mask_mode, mask, None, stream)?;

        // Transpose back: [B, n_heads, L, head_dim] -> [B, L, n_heads, head_dim]
        let attn_out = ops::transpose_axes(&attn_out, &[0, 2, 1, 3], stream)?;
        // Reshape: [B, L, n_heads, head_dim] -> [B, L, n_heads * head_dim]
        let attn_out = ops::reshape(
            &attn_out,
            &[batch, seq_len, self.n_heads * self.head_dim],
            stream,
        )?;

        // Output projection
        let output = self.wo.forward_with_stream(&attn_out, stream)?;

        // Return output and updated KV cache
        Ok((output, k, v))
    }
}

impl Module for Attention {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        let (out, _, _) = self.forward_with_cache(x, None, None, 0, "none", None, &stream)?;
        Ok(out)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        for (name, arr) in self.wq.parameters() {
            params.push((format!("wq.{}", name), arr));
        }
        for (name, arr) in self.wk.parameters() {
            params.push((format!("wk.{}", name), arr));
        }
        for (name, arr) in self.wv.parameters() {
            params.push((format!("wv.{}", name), arr));
        }
        for (name, arr) in self.wo.parameters() {
            params.push((format!("wo.{}", name), arr));
        }
        params
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        let p = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", prefix, name)
            }
        };
        self.wq.load_weights(weights, &p("wq"))?;
        self.wk.load_weights(weights, &p("wk"))?;
        self.wv.load_weights(weights, &p("wv"))?;
        self.wo.load_weights(weights, &p("wo"))?;
        Ok(())
    }
}
