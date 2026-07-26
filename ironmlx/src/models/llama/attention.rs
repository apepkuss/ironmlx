//! Standard Llama GQA self-attention.
//!
//! `q_proj` / `k_proj` / `v_proj` / `o_proj` (no attention bias, no Q/K norm)
//! → reshape to per-head → full-rotary split-half RoPE (`traditional = false`,
//! the HF-Llama / MiniCPM5 convention) → KV cache write+fetch → fused SDPA
//! (GQA head expansion handled inside the kernel) → `o_proj`.
//!
//! No MLA / gating / sliding-window: this is the plainest GQA block in the
//! engine. The per-head `head_dim` is taken from config (MiniCPM5-1B uses 128,
//! which is NOT `hidden_size / num_heads = 96`).

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

/// Standard GQA full-attention block (LLaMA / Mistral / MiniCPM5 style).
pub struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
    scale: f32,
}

impl LlamaAttention {
    /// Wire the four projections from `loader` under `prefix`, expecting
    /// `{prefix}.{q,k,v,o}_proj`. Bias terms are loaded automatically by
    /// [`Linear::from_loader`] when present (absent for MiniCPM5-1B).
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
    ) -> Result<Self> {
        let q_proj = Linear::from_loader(loader, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_loader(loader, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_loader(loader, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_loader(loader, &format!("{prefix}.o_proj"))?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            scale,
        })
    }

    /// Attention forward over `x: [B, S, hidden]` → `[B, S, hidden]`.
    ///
    /// - `offset`: `[B]` i32 per-row pre-update cache length (= RoPE start
    ///   position for the new tokens).
    /// - `per_row_lens`: number of REAL tokens each row writes to the cache.
    /// - `mask`: additive attention mask broadcastable to `[B, N, T_q, T_kv]`;
    ///   `None` selects the fused SDPA `mask_mode = "causal"`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        offset: &Array,
        offset_values: &[i32],
        per_row_lens: &[i32],
        mask: Option<&Array>,
        cache: &mut KVCache,
        exact_batched_verify: bool,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        let batch = dims[0];
        let seq = dims[1];

        // Project Q, K, V.
        let q = self.q_proj.forward_on(x, target)?;
        let k = self.k_proj.forward_on(x, target)?;
        let v = self.v_proj.forward_on(x, target)?;

        // [B, S, heads, head_dim] → [B, heads, S, head_dim] (SDPA convention).
        let q = q
            .reshape_on((batch, seq, self.num_heads, self.head_dim), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let k = k
            .reshape_on((batch, seq, self.num_kv_heads, self.head_dim), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let v = v
            .reshape_on((batch, seq, self.num_kv_heads, self.head_dim), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;

        // Standard HF-Llama RoPE: full rotary (dims = head_dim), split-half
        // (`traditional = false`, rotate_half), per-row array offset.
        let q = mlx::fast::rope_with_array_offset_on(
            &q,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            offset,
            None,
            target,
        )?;
        let k = mlx::fast::rope_with_array_offset_on(
            &k,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            offset,
            None,
            target,
        )?;

        // Write post-RoPE K/V into the cache. Decode-time TurboQuant caches
        // can answer SDPA directly from packed K/V.
        let cached_attention = if exact_batched_verify {
            None
        } else {
            cache.try_update_and_attend_on(&q, &k, &v, per_row_lens, self.scale, mask, target)?
        };
        let out = if let Some(out) = cached_attention {
            out
        } else {
            let (k_full, v_full) =
                cache.update_and_fetch_for_attention_on(&k, &v, per_row_lens, target)?;
            if exact_batched_verify {
                query_position_isolated_attention_on(
                    &q,
                    &k_full,
                    &v_full,
                    mask,
                    per_row_lens,
                    offset_values,
                    self.scale,
                    target,
                )?
            } else {
                match mask {
                    None => mlx::fast::scaled_dot_product_attention_on(
                        &q, &k_full, &v_full, self.scale, "causal", None, None, target,
                    )?,
                    Some(m) => mlx::fast::scaled_dot_product_attention_on(
                        &q,
                        &k_full,
                        &v_full,
                        self.scale,
                        "",
                        Some(m),
                        None,
                        target,
                    )?,
                }
            }
        };

        // [B, heads, S, head_dim] → [B, S, hidden].
        let out = out
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.num_heads * self.head_dim), target)?;

        self.o_proj.forward_on(&out, target)
    }
}

#[allow(clippy::too_many_arguments)]
fn query_position_isolated_attention_on(
    queries: &Array,
    keys: &Array,
    values: &Array,
    mask: Option<&Array>,
    per_row_lens: &[i32],
    offsets: &[i32],
    scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    if q_dims.len() != 4 {
        return Err(anyhow!(
            "Llama exact verify attention expected rank-4 queries, got {q_dims:?}"
        ));
    }
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    if offsets.len() != batch as usize || per_row_lens.len() != batch as usize {
        return Err(anyhow!(
            "Llama exact verify attention expected {batch} offsets/lens, got {}/{}",
            offsets.len(),
            per_row_lens.len()
        ));
    }
    if query_len <= 1 {
        return Err(anyhow!(
            "Llama exact verify attention requires Q>1, got Q={query_len}"
        ));
    }
    for (row, &len) in per_row_lens.iter().enumerate() {
        if len < 0 || len > query_len {
            return Err(anyhow!(
                "Llama exact verify attention invalid row {row} length {len} for Q={query_len}"
            ));
        }
    }

    let key_dims = keys.shape();
    let key_dims = key_dims.as_slice();
    let value_dims = values.shape();
    let value_dims = value_dims.as_slice();
    if key_dims.len() != 4
        || value_dims.len() != 4
        || key_dims[0] != batch
        || value_dims[0] != batch
        || key_dims[2] != value_dims[2]
    {
        return Err(anyhow!(
            "Llama exact verify attention incompatible KV shapes: keys={key_dims:?}, values={value_dims:?}, batch={batch}"
        ));
    }
    let final_key_len = key_dims[2];
    let mask_dims = mask.map(|mask| mask.shape());
    if let Some(mask_dims) = mask_dims.as_ref() {
        let dims = mask_dims.as_slice();
        if dims.len() != 4
            || !matches!(dims[0], 1) && dims[0] != batch
            || dims[2] < query_len
            || dims[3] < final_key_len
        {
            return Err(anyhow!(
                "Llama exact verify attention mask {dims:?} cannot cover B={batch}, Q={query_len}, K={final_key_len}"
            ));
        }
    }

    let mut outputs = Vec::with_capacity(query_len as usize);
    for depth in 0..query_len {
        let key_end = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(&offset, &len)| offset + (depth + 1).min(len))
            .max()
            .unwrap_or(0);
        if key_end <= 0 || key_end > final_key_len {
            return Err(anyhow!(
                "Llama exact verify attention key end {key_end} outside (0,{final_key_len}] at depth {depth}"
            ));
        }
        let query = mlx::ops::indexing::slice_strided_on(
            queries,
            &[0_i32, 0, depth, 0][..],
            &[batch, heads, depth + 1, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let keys = mlx::ops::indexing::slice_strided_on(
            keys,
            &[0_i32, 0, 0, 0][..],
            &[batch, key_dims[1], key_end, key_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let values = mlx::ops::indexing::slice_strided_on(
            values,
            &[0_i32, 0, 0, 0][..],
            &[batch, value_dims[1], key_end, value_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let depth_mask = mask
            .map(|mask| {
                let dims = mask_dims
                    .as_ref()
                    .expect("mask dimensions exist when mask exists");
                let dims = dims.as_slice();
                mlx::ops::indexing::slice_strided_on(
                    mask,
                    &[0_i32, 0, depth, 0][..],
                    &[dims[0], dims[1], depth + 1, key_end][..],
                    &[1_i32, 1, 1, 1][..],
                    target,
                )
                .map_err(anyhow::Error::from)
            })
            .transpose()?;
        outputs.push(mlx::fast::scaled_dot_product_attention_on(
            &query,
            &keys,
            &values,
            scale,
            "",
            depth_mask.as_ref(),
            None,
            target,
        )?);
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    mlx::ops::shape::concatenate_on(&refs, 2, target).map_err(Into::into)
}
