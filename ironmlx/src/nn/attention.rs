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
    /// Q/K. `mask` routes through `mlx::fast::scaled_dot_product_attention`:
    /// `None` invokes `mask_mode = "causal"`; `Some(m)` invokes
    /// `mask_mode = ""` with `mask_arr = Some(m)` consuming an additive
    /// `[B, N, T_q, T_kv]`-broadcastable mask (typical shapes: `[B, 1, T, T]`
    /// for batched prefill, `[B, 1, 1, K]` for batched decode).
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
        // through unchanged. Decode-time TurboQuant caches can answer SDPA
        // directly from packed K/V; other cases read the dense history.
        let out = match cache {
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
                if let Some(out) = c.try_update_and_attend_decode_on(
                    &q, &k, &v, lens_ref, self.scale, mask, target,
                )? {
                    out
                } else {
                    let (k_full, v_full) =
                        c.update_and_fetch_for_attention_on(&k, &v, lens_ref, target)?;
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
            }
            None => match mask {
                None => mlx::fast::scaled_dot_product_attention_on(
                    &q, &k, &v, self.scale, "causal", None, None, target,
                )?,
                Some(m) => mlx::fast::scaled_dot_product_attention_on(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    "",
                    Some(m),
                    None,
                    target,
                )?,
            },
        };

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden].
        let out = out
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.cfg.num_heads * self.cfg.head_dim), target)?;

        self.o_proj.forward_on(&out, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::TurboQuantKVBits;
    use crate::nn::Linear;
    use mlx::ops::constructors;
    use mlx::{Array, Dtype};
    use serial_test::serial;

    fn identity(size: i32) -> Array {
        let mut data = vec![0.0_f32; (size * size) as usize];
        for i in 0..size as usize {
            data[i * size as usize + i] = 1.0;
        }
        (data.as_slice(), (size, size)).try_into().unwrap()
    }

    /// Risk #1 mitigation (b1-p2.3c-2 spec §9): verify mlx fast SDPA accepts
    /// a `[B, 1, 1, K]` additive bf16 mask passed via the existing
    /// `mask: Option<&Array>` parameter of `Attention::forward_on`.
    ///
    /// Decode-time `T_q = 1`; mask broadcasts against the
    /// `[B, n_heads, T_q=1, T_kv=K]` SDPA expected shape. The mask values
    /// here are 0 / -inf so the kernel applies them additively — we only
    /// assert the call succeeds and the output has the right shape.
    #[test]
    #[serial(mlx_metal)]
    fn attention_forward_on_accepts_decode_mask_shape() {
        // B=2, n_heads=2, n_kv_heads=2, head_dim=32, hidden=64
        let cfg = AttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-6,
            has_qk_norm: false,
        };

        let q_w = Array::zeros((64_i32, 64), Dtype::Bfloat16).unwrap();
        let k_w = Array::zeros((64_i32, 64), Dtype::Bfloat16).unwrap();
        let v_w = Array::zeros((64_i32, 64), Dtype::Bfloat16).unwrap();
        let o_w = Array::zeros((64_i32, 64), Dtype::Bfloat16).unwrap();

        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let attn = Attention {
            q_proj: Linear::new_fp(q_w, None),
            k_proj: Linear::new_fp(k_w, None),
            v_proj: Linear::new_fp(v_w, None),
            o_proj: Linear::new_fp(o_w, None),
            q_norm: None,
            k_norm: None,
            cfg,
            scale,
        };

        // rot_dim = 32; sections sum to 16.
        let mrope = Mrope::new(32, 1e7, 1.0, &[6, 5, 5], true).unwrap();

        // Pre-populate cache to offsets=[3, 3] via a 3-token write.
        let mut cache = KVCache::new(
            /* batch */ 2,
            /* n_kv_heads */ 2,
            /* head_dim */ 32,
            /* v_head_dim */ 32,
            Dtype::Bfloat16,
            /* cap */ 16,
        );

        let prefill_x = Array::zeros((2_i32, 3, 64), Dtype::Bfloat16).unwrap();
        let prefill_cos = constructors::ones((2_i32, 3, 32), Dtype::Float32).unwrap();
        let prefill_sin = Array::zeros((2_i32, 3, 32), Dtype::Float32).unwrap();
        let _ = attn
            .forward_on(
                &prefill_x,
                &mrope,
                &prefill_cos,
                &prefill_sin,
                None,
                None,
                Some(&[3, 3]),
                Some(&mut cache),
                (),
            )
            .expect("prefill warm-up failed");
        assert_eq!(cache.offsets(), &[3, 3]);

        // Decode step: x shape [B=2, S=1, hidden=64] bf16.
        let x = Array::zeros((2_i32, 1, 64), Dtype::Bfloat16).unwrap();
        let cos = constructors::ones((2_i32, 1, 32), Dtype::Float32).unwrap();
        let sin = Array::zeros((2_i32, 1, 32), Dtype::Float32).unwrap();

        // Build [B=2, 1, 1, K=4] bf16 mask. After the 1-token decode write,
        // K = max(offsets_after) = 4. Row 0 sees only positions [0, 1]
        // (positions [2, 3] are -inf); row 1 sees all 4 positions.
        let neg_inf = f32::NEG_INFINITY;
        let mask_data: Vec<f32> = vec![
            // batch 0: valid [0, 1], blocked [2, 3]
            0.0, 0.0, neg_inf, neg_inf, // batch 1: valid [0, 1, 2, 3]
            0.0, 0.0, 0.0, 0.0,
        ];
        let mask_f32: Array = (mask_data.as_slice(), (2_i32, 1, 1, 4))
            .try_into()
            .expect("build mask");
        let mask = mlx::ops::cast::astype(&mask_f32, Dtype::Bfloat16).expect("cast mask bf16");

        let out = attn
            .forward_on(
                &x,
                &mrope,
                &cos,
                &sin,
                Some(&mask),
                None,
                Some(&[1, 1]),
                Some(&mut cache),
                (),
            )
            .expect("decode-time forward_on with [B, 1, 1, K] mask must succeed");

        // Output shape must be [B, S, hidden] = [2, 1, 64].
        assert_eq!(out.shape().as_slice(), &[2, 1, 64]);
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        // Verify cache write happened.
        assert_eq!(cache.offsets(), &[4, 4]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn attention_uses_turboquant_cache_read_when_enabled() {
        let cfg = AttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            has_qk_norm: false,
        };
        let attn = Attention {
            q_proj: Linear::new_fp(identity(8), None),
            k_proj: Linear::new_fp(identity(8), None),
            v_proj: Linear::new_fp(identity(8), None),
            o_proj: Linear::new_fp(identity(8), None),
            q_norm: None,
            k_norm: None,
            cfg,
            scale: 1.0 / 8.0_f32.sqrt(),
        };
        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let x_data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.37).sin() * 1.5).collect();
        let x: Array = (x_data.as_slice(), (1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();
        let cos = constructors::ones((1_i32, 2, 8), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 2, 8), Dtype::Float32).unwrap();

        let mut dense_cache = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16).with_step(16);
        let mut turbo_cache = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");

        let dense = attn
            .forward_on(
                &x,
                &mrope,
                &cos,
                &sin,
                None,
                None,
                Some(&[2]),
                Some(&mut dense_cache),
                (),
            )
            .expect("dense attention");
        let turbo = attn
            .forward_on(
                &x,
                &mrope,
                &cos,
                &sin,
                None,
                None,
                Some(&[2]),
                Some(&mut turbo_cache),
                (),
            )
            .expect("turbo attention");

        let dense = dense.to_vec::<f32>().unwrap();
        let turbo = turbo.to_vec::<f32>().unwrap();
        let max_diff = dense
            .iter()
            .zip(turbo.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1.0e-4,
            "attention output should reflect TurboQuant materialized K/V, max_diff={max_diff}"
        );
    }
}
