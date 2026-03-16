use super::linear::LinearLayer;
use super::module::Module;
use super::norm::RMSNorm;
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::fast;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;
use std::collections::HashMap;

pub struct Attention {
    pub wq: LinearLayer,
    pub wk: LinearLayer,
    pub wv: LinearLayer,
    pub wo: LinearLayer,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_traditional: bool,
    pub rope_base: Option<f32>,
    pub rope_scale: f32,
    pub partial_rotary_factor: f32,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: LinearLayer,
        wk: LinearLayer,
        wv: LinearLayer,
        wo: LinearLayer,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_traditional: bool,
        rope_base: Option<f32>,
        rope_scale: f32,
        partial_rotary_factor: f32,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            q_norm: None,
            k_norm: None,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_traditional,
            rope_base,
            rope_scale,
            partial_rotary_factor,
        }
    }

    /// Set QK normalization layers (used by Qwen3).
    pub fn with_qk_norm(mut self, q_norm: RMSNorm, k_norm: RMSNorm) -> Self {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
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
        let mut q = ops::reshape(&q, &[batch, seq_len, self.n_heads, self.head_dim], stream)?;
        let mut k = ops::reshape(
            &k,
            &[batch, seq_len, self.n_kv_heads, self.head_dim],
            stream,
        )?;
        let v = ops::reshape(
            &v,
            &[batch, seq_len, self.n_kv_heads, self.head_dim],
            stream,
        )?;

        // QK Norm (Qwen3): apply RMSNorm to Q and K before RoPE
        if let Some(ref qn) = self.q_norm {
            q = qn.forward_with_stream(&q, stream)?;
        }
        if let Some(ref kn) = self.k_norm {
            k = kn.forward_with_stream(&k, stream)?;
        }

        // Transpose to [B, n_heads, L, head_dim]
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3], stream)?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3], stream)?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3], stream)?;

        // RoPE (with optional partial rotary encoding)
        let (q, k) = if self.partial_rotary_factor < 1.0 {
            let rope_dim = (self.head_dim as f32 * self.partial_rotary_factor) as i32;
            let q_shape = q.shape();
            let k_shape = k.shape();

            // Split q along last axis: [B, n_heads, L, head_dim] -> rot + pass
            let q_rot = ops::slice(
                &q,
                &[0, 0, 0, 0],
                &[q_shape[0], q_shape[1], q_shape[2], rope_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let q_pass = ops::slice(
                &q,
                &[0, 0, 0, rope_dim],
                &[q_shape[0], q_shape[1], q_shape[2], q_shape[3]],
                &[1, 1, 1, 1],
                stream,
            )?;

            // Split k along last axis
            let k_rot = ops::slice(
                &k,
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], k_shape[2], rope_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let k_pass = ops::slice(
                &k,
                &[0, 0, 0, rope_dim],
                &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
                &[1, 1, 1, 1],
                stream,
            )?;

            // Apply RoPE only to the rotary portion
            let q_rot = fast::rope(
                &q_rot,
                rope_dim,
                self.rope_traditional,
                self.rope_base,
                self.rope_scale,
                offset,
                None,
                stream,
            )?;
            let k_rot = fast::rope(
                &k_rot,
                rope_dim,
                self.rope_traditional,
                self.rope_base,
                self.rope_scale,
                offset,
                None,
                stream,
            )?;

            // Concatenate rotary and pass-through portions back together
            let q_arr = VectorArray::from_arrays(&[&q_rot, &q_pass]);
            let k_arr = VectorArray::from_arrays(&[&k_rot, &k_pass]);
            (
                ops::concatenate(&q_arr, -1, stream)?,
                ops::concatenate(&k_arr, -1, stream)?,
            )
        } else {
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
            (q, k)
        };

        // KV cache update
        let (k, v) = if let (Some(ck), Some(cv)) = (cache_keys, cache_values) {
            let arrays_k = VectorArray::from_arrays(&[ck, &k]);
            let arrays_v = VectorArray::from_arrays(&[cv, &v]);
            let new_k = ops::concatenate(&arrays_k, 2, stream)?;
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

    fn load_weights(&mut self, _weights: &HashMap<String, Array>, _prefix: &str) -> Result<()> {
        unimplemented!("use LinearLayer::from_weights during model construction")
    }
}
