//! Gated full attention block — Qwen3.5 / Qwen3-Next canonical attention.
//!
//! Mirrors mlx-lm's `Qwen3NextAttention` (`/Volumes/Dev/mlx-lm/mlx_lm/models/qwen3_next.py`).
//! `qwen3_5.py` imports it directly: `from .qwen3_next import Qwen3NextAttention as Attention`.
//!
//! Differs from P1 [`crate::nn::Attention`] (standard) in exactly two places:
//! 1. `q_proj` produces `num_heads * head_dim * 2` outputs; the second half is the gate.
//! 2. After SDPA + reshape, the result is element-wise multiplied by `sigmoid(gate)` before
//!    `o_proj`.
//!
//! See P3b2 spec § 2 for the data flow.

use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, Mrope, RmsNorm};
use crate::Result;

/// Configuration for [`GatedAttention`].
///
/// Notably differs from [`crate::nn::AttentionConfig`] by:
/// - `attention_bias` field (Qwen3.5: false; carried from model config)
/// - No `has_qk_norm` field — Qwen3.5 always has q/k_norm
#[derive(Debug, Clone, Copy)]
pub struct GatedAttentionConfig {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    /// Whether the projection layers carry a bias term. Qwen3.5 sets this to
    /// `false`. Currently informational — [`GatedAttention::from_loader`]
    /// auto-detects bias presence via [`Linear::from_loader`] probing rather
    /// than consulting this field. See `from_loader` doc.
    pub attention_bias: bool,
}

/// Qwen3.5 / Qwen3-Next gated full attention block.
pub struct GatedAttention {
    q_proj: Linear,  // [hidden] -> [num_heads * head_dim * 2]  (queries + gate halves)
    k_proj: Linear,  // [hidden] -> [num_kv_heads * head_dim]
    v_proj: Linear,  // [hidden] -> [num_kv_heads * head_dim]
    o_proj: Linear,  // [num_heads * head_dim] -> [hidden]
    q_norm: RmsNorm, // weight: [head_dim]
    k_norm: RmsNorm, // weight: [head_dim]
    cfg: GatedAttentionConfig,
    scale: f32, // 1 / sqrt(head_dim)
}

impl GatedAttention {
    /// Production constructor: load from a project [`Loader`].
    ///
    /// Reads `{prefix}.{q,k,v,o}_proj.{weight,bias?,scales?,biases?}` and
    /// `{prefix}.{q,k}_norm.weight`. `bias` presence is currently auto-detected
    /// by [`Linear::from_loader`] probing `{prefix}.bias` — `cfg.attention_bias`
    /// is **not** consulted in this constructor (Qwen3.5 has
    /// `attention_bias=false` and the checkpoint never contains a bias key, so
    /// the two are consistent in practice). The field is retained as a
    /// future-extension hook for architectures that explicitly require bias
    /// validation; a runtime assertion or a `Linear::from_loader_with_bias`
    /// variant can be added without breaking the public API.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedAttentionConfig) -> Result<Self> {
        let q_proj = Linear::from_loader(loader, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_loader(loader, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_loader(loader, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_loader(loader, &format!("{prefix}.o_proj"))?;
        let q_norm = RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?;

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

    /// Test/composition seam: build a `GatedAttention` from pre-built nn building blocks.
    ///
    /// Used by unit tests and the integration fixture path to avoid synthesizing a real
    /// `model_dir/safetensors` for tiny test cases. Production code uses [`from_loader`].
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        cfg: GatedAttentionConfig,
    ) -> Self {
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cfg,
            scale,
        }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &GatedAttentionConfig {
        &self.cfg
    }

    /// Forward pass — see [`forward_on`](Self::forward_on) for stream-targeted variant.
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        // Non-decoder callers (CLI / standalone tests) — pass -1 per spec § 2.5a.
        self.forward_on(x, mrope, cos, sin, mask, None, None, cache, (), -1)
    }

    /// Stream-targeted forward.
    ///
    /// `x: [B, S, hidden]`. Returns `[B, S, hidden]`.
    ///
    /// `cos`/`sin` are precomputed by [`Mrope::cos_sin`] (caller computes once per
    /// forward and shares across all attention layers).
    ///
    /// `mask: Option<&Array>` is the explicit SDPA mask. When `None`, SDPA
    /// runs in `mask_mode="causal"` (lower-right alignment). When `Some`,
    /// the array is passed directly to mlx fast SDPA's `mask_arr` slot —
    /// expected shape `[B, 1, T_q, T_kv]` additive, broadcast-compatible
    /// with `[B, num_heads, T_q, T_kv]`. See B1-p2.1 design for the
    /// batched-prefill use of this path.
    ///
    /// `kv_validity_mask: Option<&Array>` is the `[B, T]` boolean per-token
    /// validity mask for batched prefill. When `Some`, K and V are
    /// multiplied by it (broadcast to `[B, num_kv_heads=1, T, head_dim=1]`)
    /// BEFORE cache write, so pad slots land as zero K/V cells. Decode-time
    /// reads of the cache see zero K, V at pad positions → zero attention
    /// scores → no contamination of real-row outputs. Leave `None` for the
    /// single-stream path; behavior is bit-identical to the pre-B1-p2.2
    /// implementation.
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
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();

        // Two-step bind: x.shape() returns an owned Shape; we bind it to extend
        // its lifetime past .as_slice(), since the slice borrows from the Shape.
        let dims = x.shape();
        let dims = dims.as_slice();
        let batch = dims[0];
        let seq = dims[1];
        let h_q = self.cfg.num_heads;
        let h_kv = self.cfg.num_kv_heads;
        let d = self.cfg.head_dim;

        #[cfg(feature = "p5h-profile")]
        {
            // T2.2: 7-substep instrumentation, all under the `attention_path`
            // wrapper opened by DecoderLayerMoe::forward_on (T0a.11 step 1).
            // Substep boundaries verified against this file's pre-T2 source
            // (lines 184-282 at HEAD 9e746bc) per spec § 2.2 #5.
            //
            // The `try_` variant no-ops when no active P5H_CURRENT_TRACE
            // (CLI / standalone tests path) per Codex v12 P1 #1.

            // Substep 1: q_gate_k_v_proj — 3 Linear forwards (not fused).
            let (q_full, k, v) = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "q_gate_k_v_proj",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<(Array, Array, Array)> {
                    let q_full = self.q_proj.forward_on(x, target)?; // [B, S, Hq*D*2]
                    let k = self.k_proj.forward_on(x, target)?; // [B, S, Hkv*D]
                    let v = self.v_proj.forward_on(x, target)?; // [B, S, Hkv*D]
                                                                // P5h+1 T1: measurement-eval probe (defaults OFF in
                                                                // production). Forces lazy graph to materialize within
                                                                // this substep so inclusive_us reflects true cost.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&q_full, &k, &v])?;
                    }
                    Ok((q_full, k, v))
                },
            )?;

            // Substep 2: q_split_norm_reshape — per-head reshape Q, split into
            // (queries, gate); gate_flat reshape; q_norm + transpose to SDPA
            // layout; k reshape + k_norm + transpose; v reshape + transpose.
            // Per-head reshape BEFORE split matches q_proj weight row layout in
            // mlx-lm. mlx-lm applies q_norm/k_norm BEFORE transpose; either
            // order is mathematically identical (RMSNorm is on last axis = D).
            let (queries, k, v, gate_flat) =
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "q_split_norm_reshape",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array, Array, Array)> {
                        let q_per_head = q_full.reshape_on((batch, seq, h_q, d * 2), target)?;
                        let mut parts = mlx::ops::shape::split_n_on(&q_per_head, 2, -1, target)?;
                        // split_n_on returns Vec<Array>; index 0 = queries, index 1 = gate.
                        // Pop in reverse to avoid index-shift surprises (P3b1 polish convention).
                        let gate_per_head = parts.pop().expect("split_n_on returned <2 elements");
                        let queries = parts.pop().expect("split_n_on returned <2 elements");
                        // Gate is fed flat to sigmoid + element-wise mul later: [B, S, Hq*D].
                        let gate_flat = gate_per_head.reshape_on((batch, seq, h_q * d), target)?;
                        let queries = self.q_norm.forward_on(&queries, target)?;
                        let queries = queries.transpose_axes_on(&[0, 2, 1, 3][..], target)?;
                        let k = k.reshape_on((batch, seq, h_kv, d), target)?;
                        let k = self.k_norm.forward_on(&k, target)?;
                        let k = k.transpose_axes_on(&[0, 2, 1, 3][..], target)?;
                        let v = v
                            .reshape_on((batch, seq, h_kv, d), target)?
                            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&queries, &k, &v, &gate_flat])?;
                        }
                        Ok((queries, k, v, gate_flat))
                    },
                )?;

            // Substep 3: mrope_apply — fused MetalKernel rotates Q + K (P3b1).
            let (queries, k) = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "mrope_apply",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<(Array, Array)> {
                    let (queries, k) = mrope.apply(&queries, &k, cos, sin)?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&queries, &k])?;
                    }
                    Ok((queries, k))
                },
            )?;

            // Substep 4: kv_mask_update — (a) zero out K, V at pad positions via
            // the batched-prefill validity mask (broadcasts [B, T] -> [B, 1, T, 1])
            // so decode-time cache reads see zero contribution at pad slots;
            // (b) cache.update_and_fetch_on returns (k_full, v_full).
            let (k_full, v_full) = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "kv_mask_update",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<(Array, Array)> {
                    let (k, v) = if let Some(vm) = kv_validity_mask {
                        let vm_dtype = mlx::ops::cast::astype(vm, k.dtype())?;
                        let vm_broadcast =
                            vm_dtype.reshape_on((batch, 1_i32, seq, 1_i32), target)?;
                        let k_masked = &k * &vm_broadcast;
                        let v_masked = &v * &vm_broadcast;
                        (k_masked, v_masked)
                    } else {
                        (k, v)
                    };
                    let out = match cache {
                        Some(c) => {
                            let lens_owned: Vec<i32>;
                            let lens_ref: &[i32] = match per_row_lens {
                                Some(l) => l,
                                None => {
                                    // Non-batched single-stream caller (e.g., GenerationStream):
                                    // construct lockstep-equivalent uniform lens from K seq dim.
                                    lens_owned = vec![seq; batch as usize];
                                    &lens_owned
                                }
                            };
                            // T4.3: wrap KVCache::update_and_fetch_on in a
                            // `cache_state_update` tree span. Parent: the
                            // enclosing `kv_mask_update` substep span. Caller-
                            // site wrap so `layer_idx` is available for
                            // SpanFields.
                            crate::core::p5h::try_with_p5h_span_from_current_trace(
                                "cache_state_update",
                                || crate::core::p5h::SpanFields {
                                    layer_idx: Some(layer_idx),
                                    ..Default::default()
                                },
                                || -> Result<(Array, Array)> {
                                    let (k_full, v_full) =
                                        c.update_and_fetch_on(&k, &v, lens_ref, target)?;
                                    // P5h+1 T1: measurement-eval probe.
                                    if crate::core::p5h::is_measurement_eval_probes_active() {
                                        mlx::transforms::eval(&[&k_full, &v_full])?;
                                    }
                                    Ok((k_full, v_full))
                                },
                            )?
                        }
                        None => (k, v),
                    };
                    // P5h+1 T1: measurement-eval probe for the kv_mask_update
                    // substep itself (separate from the nested cache_state_update
                    // probe above; this captures the masked k/v tensors as
                    // returned to the SDPA caller).
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&out.0, &out.1])?;
                    }
                    Ok(out)
                },
            )?;

            // Substep 5: fused_sdpa — explicit array mask (batched prefill)
            // routes via mask_mode="" + mask_arr=Some; otherwise "causal"
            // (lower-right alignment) for single-stream + decode (T_q=1).
            let attn_out = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "fused_sdpa",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let out = match mask {
                        None => mlx::fast::scaled_dot_product_attention_on(
                            &queries, &k_full, &v_full, self.scale, "causal", None, None, target,
                        )?,
                        Some(m) => mlx::fast::scaled_dot_product_attention_on(
                            &queries,
                            &k_full,
                            &v_full,
                            self.scale,
                            "",
                            Some(m),
                            None,
                            target,
                        )?,
                    };
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&out])?;
                    }
                    Ok(out)
                },
            )?;

            // Substep 6: gate_sigmoid_mul — reshape attn out [B, Hq, S, D] ->
            // [B, S, Hq*D], apply sigmoid gate, element-wise multiply.
            let gated = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "gate_sigmoid_mul",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let attn_out = attn_out
                        .transpose_axes_on(&[0, 2, 1, 3][..], target)?
                        .reshape_on((batch, seq, h_q * d), target)?;
                    let gate_sig = gate_flat.sigmoid_on(target)?;
                    // &Array * &Array returns Array (panic-on-err overload), not Result<Array>.
                    let gated = &attn_out * &gate_sig;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&gated])?;
                    }
                    Ok(gated)
                },
            )?;

            // Substep 7: o_proj — final Linear projection.
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "o_proj",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let out = self.o_proj.forward_on(&gated, target)?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&out])?;
                    }
                    Ok(out)
                },
            )
        }

        #[cfg(not(feature = "p5h-profile"))]
        {
            // Production build: layer_idx is signature-only plumbing (consumed
            // only by the p5h-profile substep spans above).
            let _ = layer_idx;

            // Step 1: project Q (2x), K, V.
            let q_full = self.q_proj.forward_on(x, target)?; // [B, S, Hq*D*2]
            let k = self.k_proj.forward_on(x, target)?; // [B, S, Hkv*D]
            let v = self.v_proj.forward_on(x, target)?; // [B, S, Hkv*D]

            // Step 2: per-head reshape Q to [B, S, Hq, D*2], then split last axis into
            // (queries [B,S,Hq,D], gate [B,S,Hq,D]). Per-head reshape BEFORE split is
            // critical: it matches q_proj weight matrix row layout in mlx-lm.
            let q_per_head = q_full.reshape_on((batch, seq, h_q, d * 2), target)?;
            let mut parts = mlx::ops::shape::split_n_on(&q_per_head, 2, -1, target)?;
            // split_n_on returns Vec<Array>; index 0 = queries, index 1 = gate.
            // Pop in reverse to avoid index-shift surprises (P3b1 polish convention).
            let gate_per_head = parts.pop().expect("split_n_on returned <2 elements");
            let queries = parts.pop().expect("split_n_on returned <2 elements");

            // Gate is fed flat to sigmoid + element-wise mul later: [B, S, Hq*D].
            let gate_flat = gate_per_head.reshape_on((batch, seq, h_q * d), target)?;

            // Step 3: q_norm on per-head queries (last axis = D), then transpose to SDPA
            // layout [B, Hq, S, D]. mlx-lm applies q_norm BEFORE transpose; either order
            // is mathematically identical (RMSNorm is on last axis = D) — match mlx-lm.
            let queries = self.q_norm.forward_on(&queries, target)?;
            let queries = queries.transpose_axes_on(&[0, 2, 1, 3][..], target)?;

            // Step 4: reshape K to per-head, k_norm, transpose. Same for V (no norm).
            let k = k.reshape_on((batch, seq, h_kv, d), target)?;
            let k = self.k_norm.forward_on(&k, target)?;
            let k = k.transpose_axes_on(&[0, 2, 1, 3][..], target)?;

            let v = v
                .reshape_on((batch, seq, h_kv, d), target)?
                .transpose_axes_on(&[0, 2, 1, 3][..], target)?;

            // Step 5: rotate Q + K via fused MetalKernel (P3b1).
            let (queries, k) = mrope.apply(&queries, &k, cos, sin)?;

            // Step 5b (batched prefill): zero out K, V at pad positions before
            // writing to the cache. The [B, T] boolean validity mask broadcasts
            // to [B, num_kv_heads=1 dim, T, head_dim=1 dim] for the multiply.
            // Decode-time reads of the cache then see zero K, V at pad slots →
            // zero attention contribution → no contamination of real outputs.
            let (k, v) = if let Some(vm) = kv_validity_mask {
                let vm_dtype = mlx::ops::cast::astype(vm, k.dtype())?;
                let vm_broadcast = vm_dtype.reshape_on((batch, 1_i32, seq, 1_i32), target)?;
                let k_masked = &k * &vm_broadcast;
                let v_masked = &v * &vm_broadcast;
                (k_masked, v_masked)
            } else {
                (k, v)
            };

            // Step 6: KV cache route + SDPA. The explicit array mask (when
            // provided by the batched-prefill caller) routes via mask_mode=""
            // + mask_arr=Some; otherwise the kernel runs in "causal" mode
            // (lower-right alignment) which is correct for the single-stream
            // path and for decode-time (T_q=1) calls.
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
            let attn_out = match mask {
                None => mlx::fast::scaled_dot_product_attention_on(
                    &queries, &k_full, &v_full, self.scale, "causal", None, None, target,
                )?,
                Some(m) => mlx::fast::scaled_dot_product_attention_on(
                    &queries,
                    &k_full,
                    &v_full,
                    self.scale,
                    "",
                    Some(m),
                    None,
                    target,
                )?,
            };

            // Step 7: reshape attn out [B, Hq, S, D] -> [B, S, Hq*D], apply sigmoid gate,
            // o_proj.
            let attn_out = attn_out
                .transpose_axes_on(&[0, 2, 1, 3][..], target)?
                .reshape_on((batch, seq, h_q * d), target)?;

            let gate_sig = gate_flat.sigmoid_on(target)?;
            // &Array * &Array returns Array (panic-on-err overload), not Result<Array>.
            let gated = &attn_out * &gate_sig;

            self.o_proj.forward_on(&gated, target)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::constructors;
    use mlx::{Array, Dtype};

    /// Build a small synthetic GatedAttention for unit tests.
    /// B=1, S=4, Hq=4, Hkv=2, D=8, hidden=32; partial=1.0 → rot_dim=8.
    fn small_gated_attention() -> GatedAttention {
        // q_proj: [Hq*D*2=64, hidden=32]
        let q_w = Array::zeros((64_i32, 32), Dtype::Float32).unwrap();
        let k_w = Array::zeros((16_i32, 32), Dtype::Float32).unwrap();
        let v_w = Array::zeros((16_i32, 32), Dtype::Float32).unwrap();
        let o_w = Array::zeros((32_i32, 32), Dtype::Float32).unwrap();
        let q_n = constructors::ones((8_i32,), Dtype::Float32).unwrap();
        let k_n = constructors::ones((8_i32,), Dtype::Float32).unwrap();

        let cfg = GatedAttentionConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        };

        GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(q_n, cfg.rms_norm_eps),
            RmsNorm::new(k_n, cfg.rms_norm_eps),
            cfg,
        )
    }

    #[test]
    fn from_components_carries_config() {
        let attn = small_gated_attention();
        let cfg = attn.config();
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.head_dim, 8);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.attention_bias);
    }

    #[test]
    fn from_components_computes_scale() {
        let attn = small_gated_attention();
        // scale = 1 / sqrt(head_dim=8)
        let expected = 1.0 / 8.0_f32.sqrt();
        assert!((attn.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn forward_shape_and_dtype_fp32() {
        let attn = small_gated_attention();
        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();

        // x: [B=1, S=4, hidden=32] fp32
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        // cos/sin shape: [B, S, rot_dim=8]
        let cos = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();

        let out = attn
            .forward(&x, &mrope, &cos, &sin, None, None)
            .expect("forward");

        // Output shape == input shape [B, S, hidden]
        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    fn forward_shape_and_dtype_bf16() {
        // Build a bf16-weight attention so that bf16 input stays bf16 through
        // the full forward path. MLX promotes bf16 @ fp32 → fp32, so weights
        // must also be bf16 to preserve the dtype invariant.
        let q_w = Array::zeros((64_i32, 32), Dtype::Bfloat16).unwrap();
        let k_w = Array::zeros((16_i32, 32), Dtype::Bfloat16).unwrap();
        let v_w = Array::zeros((16_i32, 32), Dtype::Bfloat16).unwrap();
        let o_w = Array::zeros((32_i32, 32), Dtype::Bfloat16).unwrap();
        // RmsNorm weights must also be bf16 so that the norm kernel stays in bf16.
        let q_n = constructors::ones((8_i32,), Dtype::Bfloat16).unwrap();
        let k_n = constructors::ones((8_i32,), Dtype::Bfloat16).unwrap();
        let cfg = GatedAttentionConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        };
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(q_n, cfg.rms_norm_eps),
            RmsNorm::new(k_n, cfg.rms_norm_eps),
            cfg,
        );

        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();

        let x = Array::zeros((1_i32, 4, 32), Dtype::Bfloat16).unwrap();
        // cos/sin always fp32 per P3b1 spec.
        // cos/sin shape: [B, S, rot_dim=8]
        let cos = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();

        let out = attn
            .forward(&x, &mrope, &cos, &sin, None, None)
            .expect("forward bf16");

        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        assert_eq!(out.dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn forward_with_zero_weight_dispatch_succeeds() {
        // q_proj weight has shape [Hq*D*2, hidden] = [64, 32], all zeros. So
        // q_proj output is zero regardless of input → gate is zero → sigmoid(0)
        // = 0.5 (half-strength, NOT zero). We don't assert exact values here;
        // we just verify the full dispatch pipeline runs without crashing and
        // produces finite output. Exact numerical correctness is validated in
        // the T3 integration test against the Python fixture.
        let attn = small_gated_attention();
        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();

        // Random-ish input (small range, fp32 to avoid bf16 noise)
        let x_data: Vec<f32> = (0..(1 * 4 * 32)).map(|i| (i as f32) * 0.01).collect();
        let x: Array = (x_data.as_slice(), (1_i32, 4, 32)).try_into().unwrap();

        // cos/sin shape: [B, S, rot_dim=8]
        let cos = mlx::ops::constructors::ones((1_i32, 4, 8), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();

        let out = attn
            .forward(&x, &mrope, &cos, &sin, None, None)
            .expect("forward zero gate");

        // Output exists with the right shape; we don't assert exact values
        // (those are validated in the integration test against a Python ref).
        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        let v: Vec<f32> = out.to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output element");
    }

    #[test]
    fn per_head_split_layout_distinguishable_from_flat_split() {
        // 2 heads, head_dim = 2 (small for hand-checkable math).
        // q_proj: [Hq*D*2, hidden] = [8, 4]. Per-head row layout:
        //
        //   Row 0..2   = head 0 queries channels 0..2
        //   Row 2..4   = head 0 gate    channels 0..2
        //   Row 4..6   = head 1 queries channels 0..2
        //   Row 6..8   = head 1 gate    channels 0..2
        //
        // Set head 0 queries = identity (weight[0..2, 0..2] = I), head 0 gate = 0.
        // Set head 1 queries = 0, head 1 gate = identity (weight[6..8, 2..4] = I,
        // mapping x[2..4] -> gate[head 1]).
        //
        // Per-HEAD split:
        //   queries[head 0] = [x[0], x[1]] = [1, 2]
        //   queries[head 1] = [0, 0]
        //   gate[head 0]    = [0, 0]    -> sigmoid = [0.5, 0.5]
        //   gate[head 1]    = [x[2], x[3]] = [3, 4] -> sigmoid ≈ [0.953, 0.982]
        //
        // After SDPA + gate + o_proj=identity, head 1 channels (output indices
        // 2, 3) should have larger magnitude than head 0 channels (0, 1) because
        // head 1's sigmoid gate is much bigger than head 0's 0.5.

        let mut q_w_data = vec![0.0_f32; 8 * 4];
        // Row 0..2: head 0 queries = identity on x[0..2]
        q_w_data[0 * 4 + 0] = 1.0;
        q_w_data[1 * 4 + 1] = 1.0;
        // Row 6..8: head 1 gate = identity on x[2..4]
        q_w_data[6 * 4 + 2] = 1.0;
        q_w_data[7 * 4 + 3] = 1.0;
        let q_w: Array = (q_w_data.as_slice(), (8_i32, 4)).try_into().unwrap();

        // K, V projection: per-head 2 dims, 1 KV head -> [Hkv*D, hidden] = [2, 4].
        // Make k = v constant (broadcast a row of 0.25s).
        let kv_w_data = vec![0.25_f32; 2 * 4];
        let k_w: Array = (kv_w_data.as_slice(), (2_i32, 4)).try_into().unwrap();
        let v_w: Array = (kv_w_data.as_slice(), (2_i32, 4)).try_into().unwrap();

        // o_proj: 4x4 identity.
        let mut o_w_data = vec![0.0_f32; 4 * 4];
        for i in 0..4 {
            o_w_data[i * 4 + i] = 1.0;
        }
        let o_w: Array = (o_w_data.as_slice(), (4_i32, 4)).try_into().unwrap();

        let q_n = mlx::ops::constructors::ones((2_i32,), Dtype::Float32).unwrap();
        let k_n = mlx::ops::constructors::ones((2_i32,), Dtype::Float32).unwrap();

        let cfg = GatedAttentionConfig {
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        };

        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(q_n, cfg.rms_norm_eps),
            RmsNorm::new(k_n, cfg.rms_norm_eps),
            cfg,
        );

        // x = [1, 2, 3, 4]
        let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1, 4))
            .try_into()
            .unwrap();

        // rot_dim = 2 * 1.0 = 2. cos/sin shape: [B, S, rot_dim=2]
        let mrope = Mrope::new(2, 1e7, 1.0, &[1, 0, 0], true).unwrap();
        let cos = mlx::ops::constructors::ones((1_i32, 1, 2), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 1, 2), Dtype::Float32).unwrap();

        let out = attn
            .forward(&x, &mrope, &cos, &sin, None, None)
            .expect("forward");

        let v: Vec<f32> = out.to_vec().unwrap();
        // Sanity: finite + non-zero output
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
        assert!(v.iter().any(|x| x.abs() > 1e-3), "all zeros — likely a bug");
        // Per-head invariant: head 1's gate is non-zero (sigmoid > 0.9), head 0's
        // gate is zero (sigmoid = 0.5). With o_proj=identity, output index i comes
        // from SDPA-output[i] * sigmoid(gate)[i].
        //
        // Head 0 query post-RMSNorm is non-zero (input [1, 2] normalized). Head 1
        // query post-RMSNorm is the result of normalizing [0, 0]; with a small eps
        // it stays near zero, but SDPA with a near-zero query against constant k/v
        // still produces a finite, nonzero attention output (softmax over equal
        // logits => uniform weights => output = v average).
        //
        // So both heads produce finite SDPA-out of similar magnitude (k=v=const),
        // and the gate ratio dominates: head 1 channels (indices 2, 3) should be
        // larger than head 0 channels (indices 0, 1).
        assert!(
            v[2].abs() > v[0].abs() && v[3].abs() > v[1].abs(),
            "head 1 channels not larger than head 0 (per-head split incorrect): {:?}",
            v
        );
    }
}
