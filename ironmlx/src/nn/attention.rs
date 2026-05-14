//! Standard full attention with optional Q/K norm and MRoPE rotation,
//! routed through `mlx::fast::scaled_dot_product_attention` for the
//! attention math.
//!
//! Scope at P1 is **construction + parameter wiring**: `from_loader` reads
//! all four projections and (when `has_qk_norm`) the per-head Q/K RMSNorms,
//! then computes the attention scale. The `forward` body is fully wired
//! against the cxx-mlx fused SDPA kernel, but **calling it at P1 returns
//! `Err`** because [`crate::nn::Mrope::apply`] is stubbed: real position-ids
//! shapes only arrive once the Qwen3.5 model assembly (P3) drives the
//! attention block. KV-cache integration lands in P2.

use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, Mrope, RmsNorm};
use crate::Result;

/// Configuration knobs for [`Attention`].
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    /// Whether the architecture has per-head `q_norm` / `k_norm`
    /// (Qwen3+ style; absent in LLaMA / Mistral).
    pub has_qk_norm: bool,
}

/// Standard GQA full-attention block.
///
/// Layout: `q_proj`, `k_proj`, `v_proj` project the hidden state into per-head
/// Q/K/V; optional `q_norm`/`k_norm` normalize each head before rotation;
/// `Mrope::apply` rotates Q/K; the fused SDPA kernel does the
/// scale-softmax-matmul; `o_proj` projects back to the model dim.
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    cfg: AttentionConfig,
    scale: f32,
}

impl Attention {
    /// Wire all sub-layers from `loader` under `prefix`, expecting
    /// `{prefix}.q_proj`, `{prefix}.k_proj`, `{prefix}.v_proj`,
    /// `{prefix}.o_proj`, plus `{prefix}.q_norm` / `{prefix}.k_norm` when
    /// `cfg.has_qk_norm` is set.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: AttentionConfig) -> Result<Self> {
        let q_proj = Linear::from_loader(loader, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_loader(loader, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_loader(loader, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_loader(loader, &format!("{prefix}.o_proj"))?;

        let (q_norm, k_norm) = if cfg.has_qk_norm {
            (
                Some(RmsNorm::from_loader(
                    loader,
                    &format!("{prefix}.q_norm"),
                    cfg.rms_norm_eps,
                )?),
                Some(RmsNorm::from_loader(
                    loader,
                    &format!("{prefix}.k_norm"),
                    cfg.rms_norm_eps,
                )?),
            )
        } else {
            (None, None)
        };

        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cfg,
            scale,
        })
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &AttentionConfig {
        &self.cfg
    }

    /// Forward without KV cache (P1 prefill-only path; P2 adds cache + decode).
    ///
    /// `x: [batch, seq, hidden]`. Returns `[batch, seq, hidden]`.
    ///
    /// **As of P3b1 this is fully wired end-to-end**: rotary positions are
    /// applied via the fused MRoPE Q+K MetalKernel, then SDPA runs through
    /// `mlx::fast::scaled_dot_product_attention`. Caller supplies the
    /// pre-computed `cos`/`sin` tables (computed once per forward in the
    /// model assembly via `Mrope::cos_sin`).
    ///
    /// `cos`, `sin` are the precomputed rotary tables broadcastable against
    /// Q/K. `mask` is currently ignored — the kernel is always invoked with
    /// `mask_mode = "causal"`; explicit masks are folded in at P2 once the
    /// KV cache lands.
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        self.forward_on(x, mrope, cos, sin, mask, None, None, cache, ())
    }

    /// Stream-targeted forward pass — see [`Attention::forward`] for semantics.
    ///
    /// `kv_validity_mask: Option<&Array>` is the `[B, T]` boolean per-token
    /// validity mask consumed during batched prefill: when `Some`, the
    /// computed K, V tensors are multiplied by it (broadcast to
    /// `[B, num_kv_heads, T, head_dim]`) BEFORE the cache write, so pad
    /// positions land as zero K/V cells. Decode-time attention then reads
    /// a cache with no pad contamination. For single-stream callers (and
    /// for the `None` mask path) leave this `None` — no zeroing happens
    /// and the path is bit-identical to the pre-B1-p2.2 behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        kv_validity_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut KVCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        let shape = x.shape();
        let dims = shape.as_slice();
        let batch = dims[0];
        let seq = dims[1];

        // Project Q, K, V.
        let q = self.q_proj.forward_on(x, target)?;
        let k = self.k_proj.forward_on(x, target)?;
        let v = self.v_proj.forward_on(x, target)?;

        // Reshape to [batch, seq, heads, head_dim] then transpose to
        // [batch, heads, seq, head_dim] (SDPA convention).
        let q = q
            .reshape_on((batch, seq, self.cfg.num_heads, self.cfg.head_dim), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let k = k
            .reshape_on(
                (batch, seq, self.cfg.num_kv_heads, self.cfg.head_dim),
                target,
            )?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let v = v
            .reshape_on(
                (batch, seq, self.cfg.num_kv_heads, self.cfg.head_dim),
                target,
            )?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;

        // Per-head Q/K RMSNorm before rotation (Qwen3+ style).
        let q = if let Some(qn) = &self.q_norm {
            qn.forward_on(&q, target)?
        } else {
            q
        };
        let k = if let Some(kn) = &self.k_norm {
            kn.forward_on(&k, target)?
        } else {
            k
        };

        // Fused Q+K rotary application (T2 MetalKernel: one dispatch for both).
        let (q, k) = mrope.apply(&q, &k, cos, sin)?;

        // Zero out K, V at pad positions before writing to the cache. The
        // [B, T] boolean validity mask is broadcast to
        // [B, num_kv_heads=1 dim, T, head_dim=1 dim] then multiplied in.
        // Decode-time reads of the cache will see zero K, V at pad slots
        // → zero attention scores → no contamination of real-row outputs.
        let (k, v) = if let Some(vm) = kv_validity_mask {
            let vm_dtype = mlx::ops::cast::astype(vm, k.dtype())?;
            // [B, T] → [B, 1, T, 1]
            let vm_broadcast = vm_dtype.reshape_on((batch, 1_i32, seq, 1_i32), target)?;
            let k_masked = &k * &vm_broadcast;
            let v_masked = &v * &vm_broadcast;
            (k_masked, v_masked)
        } else {
            (k, v)
        };

        // Route post-RoPE K/V through KV cache when provided; otherwise pass
        // through unchanged. SDPA always consumes the full K/V history.
        let (k_full, v_full) = match cache {
            Some(c) => {
                let lens_owned: Vec<i32>;
                let lens_ref: &[i32] = match per_row_lens {
                    Some(l) => l,
                    None => {
                        // Non-batched single-stream caller (e.g., GenerationStream):
                        // construct lockstep-equivalent uniform lens from the K seq dim.
                        lens_owned = vec![seq; batch as usize];
                        &lens_owned
                    }
                };
                c.update_and_fetch_on(&k, &v, lens_ref, target)?
            }
            None => (k, v),
        };

        // Fused SDPA. mlx fast SDPA accepts either a string mask_mode
        // ("causal") with no mask_arr, or an explicit array mask
        // broadcast-compatible with [B, N, T_q, T_kv]. Pick based on
        // whether the caller passed an explicit attention_mask.
        let out = match mask {
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
        };

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden].
        let out = out
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.cfg.num_heads * self.cfg.head_dim), target)?;

        self.o_proj.forward_on(&out, target)
    }
}
