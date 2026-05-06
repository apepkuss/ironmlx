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

use mlx::Array;

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
    /// **At P1 this returns `Err`** because [`Mrope::apply`] is stubbed.
    /// The full path is verified in P3 once Qwen3.5 model assembly drives
    /// real position-id streams; this code is the wired structural skeleton
    /// the P3 path executes against.
    ///
    /// `cos`, `sin` are the precomputed rotary tables broadcastable against
    /// Q/K. `mask` is currently ignored — at P1 the kernel is invoked with
    /// `mask_mode = "causal"`; explicit masks are folded in at P2 once the
    /// KV cache lands.
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let _ = mask; // P1: always causal; explicit masks deferred to P2.

        let shape = x.shape();
        let dims = shape.as_slice();
        let batch = dims[0];
        let seq = dims[1];

        // Project Q, K, V.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, heads, head_dim] then transpose to
        // [batch, heads, seq, head_dim] (SDPA convention).
        let q = q
            .reshape((batch, seq, self.cfg.num_heads, self.cfg.head_dim))?
            .transpose_axes(&[0, 2, 1, 3][..])?;
        let k = k
            .reshape((batch, seq, self.cfg.num_kv_heads, self.cfg.head_dim))?
            .transpose_axes(&[0, 2, 1, 3][..])?;
        let v = v
            .reshape((batch, seq, self.cfg.num_kv_heads, self.cfg.head_dim))?
            .transpose_axes(&[0, 2, 1, 3][..])?;

        // Per-head Q/K RMSNorm before rotation (Qwen3+ style).
        let q = if let Some(qn) = &self.q_norm {
            qn.forward(&q)?
        } else {
            q
        };
        let k = if let Some(kn) = &self.k_norm {
            kn.forward(&k)?
        } else {
            k
        };

        // Apply rotary positions. Stubbed at P1 — surfaces a clear `Err`.
        let q = mrope.apply(&q, cos, sin)?;
        let k = mrope.apply(&k, cos, sin)?;

        // Fused SDPA — never compose softmax + matmul by hand.
        // P1 hard-codes causal masking; P2 layers in custom masks + KV cache.
        let out =
            mlx::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, "causal", None, None)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden].
        let out = out.transpose_axes(&[0, 2, 1, 3][..])?.reshape((
            batch,
            seq,
            self.cfg.num_heads * self.cfg.head_dim,
        ))?;

        self.o_proj.forward(&out)
    }
}
