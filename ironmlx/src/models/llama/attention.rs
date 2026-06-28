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
        per_row_lens: &[i32],
        mask: Option<&Array>,
        cache: &mut KVCache,
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
        let out = if let Some(out) =
            cache.try_update_and_attend_on(&q, &k, &v, per_row_lens, self.scale, mask, target)?
        {
            out
        } else {
            let (k_full, v_full) =
                cache.update_and_fetch_for_attention_on(&k, &v, per_row_lens, target)?;
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
        };

        // [B, heads, S, head_dim] → [B, S, hidden].
        let out = out
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.num_heads * self.head_dim), target)?;

        self.o_proj.forward_on(&out, target)
    }
}
