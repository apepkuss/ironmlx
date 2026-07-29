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

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, Mrope, RmsNorm};
use crate::Result;

#[derive(Clone, Copy)]
struct DecodeAttentionTurboProfileEvent {
    stage: &'static str,
    elapsed_us: u128,
    layer_idx: i32,
    batch: i32,
    seq: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

#[derive(Clone, Copy)]
struct DecodeAttentionTurboProfileShape {
    layer_idx: i32,
    batch: i32,
    seq: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

fn format_decode_attention_turbo_profile_line(event: DecodeAttentionTurboProfileEvent) -> String {
    format!(
        "{{\"event\":\"turboquant_gated_attention_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"layer_idx\":{},\"batch\":{},\"seq\":{},\"q_heads\":{},\"kv_heads\":{},\"head_dim\":{}}}",
        event.stage,
        event.elapsed_us,
        event.layer_idx,
        event.batch,
        event.seq,
        event.q_heads,
        event.kv_heads,
        event.head_dim,
    )
}

fn profile_decode_attention_turbo_stage(
    stage: &'static str,
    arrays: &[&Array],
    shape: DecodeAttentionTurboProfileShape,
) -> Result<()> {
    if shape.seq != 1 || std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_none() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    mlx::transforms::eval(arrays).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "{}",
        format_decode_attention_turbo_profile_line(DecodeAttentionTurboProfileEvent {
            stage,
            elapsed_us: start.elapsed().as_micros(),
            layer_idx: shape.layer_idx,
            batch: shape.batch,
            seq: shape.seq,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            head_dim: shape.head_dim,
        })
    );
    Ok(())
}

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
        let profile_shape = DecodeAttentionTurboProfileShape {
            layer_idx,
            batch,
            seq,
            q_heads: h_q,
            kv_heads: h_kv,
            head_dim: d,
        };
        profile_decode_attention_turbo_stage("decode_attention_input", &[x], profile_shape)?;

        {
            // Signature parity with other layer-aware forward paths.
            let _ = layer_idx;

            // Step 1: project Q (2x), K, V.
            let q_full = self.q_proj.forward_on(x, target)?; // [B, S, Hq*D*2]
            let k = self.k_proj.forward_on(x, target)?; // [B, S, Hkv*D]
            let v = self.v_proj.forward_on(x, target)?; // [B, S, Hkv*D]
            profile_decode_attention_turbo_stage(
                "decode_qkv_proj",
                &[&q_full, &k, &v],
                profile_shape,
            )?;

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
            profile_decode_attention_turbo_stage(
                "decode_q_split_norm_reshape",
                &[&queries, &k, &v, &gate_flat],
                profile_shape,
            )?;

            let mask_kv = |k: Array, v: Array| -> Result<(Array, Array)> {
                if let Some(vm) = kv_validity_mask {
                    let vm_dtype = mlx::ops::cast::astype(vm, k.dtype())?;
                    let vm_broadcast = vm_dtype.reshape_on((batch, 1_i32, seq, 1_i32), target)?;
                    let k_masked = &k * &vm_broadcast;
                    let v_masked = &v * &vm_broadcast;
                    Ok((k_masked, v_masked))
                } else {
                    Ok((k, v))
                }
            };

            // Step 5/6: MRoPE + KV cache route + SDPA. Decode-time TurboQuant
            // caches can answer SDPA directly from packed K/V. When the packed
            // parallel decode path is available, use the decode-only MRoPE +
            // query TurboQuant rotation kernel and skip the standalone q_rotate.
            let attn_out = match cache {
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

                    let exact_batched_verify = super::position_stable_qmm::is_armed() && seq > 1;
                    if exact_batched_verify {
                        let (k, v) = mask_kv(k, v)?;
                        query_position_isolated_attention_on(
                            c, mrope, &queries, &k, &v, cos, sin, mask, lens_ref, self.scale,
                            target,
                        )?
                    } else if let Some(signs) = c
                        .turboquant_pre_rotated_decode_query_signs(&queries, &k, &v, lens_ref, mask)
                    {
                        let (queries_tq, k_tq) = mrope
                            .apply_decode_query_turbo_rotation(&queries, &k, cos, sin, signs)?;
                        let (k_tq, v_tq) = mask_kv(k_tq, v.clone())?;
                        if let Some(out) = c.try_update_and_attend_decode_pre_rotated_on(
                            &queries_tq,
                            &k_tq,
                            &v_tq,
                            lens_ref,
                            self.scale,
                            mask,
                            queries.dtype(),
                            target,
                        )? {
                            out
                        } else {
                            let (queries, k) = mrope.apply(&queries, &k, cos, sin)?;
                            let (k, v) = mask_kv(k, v)?;
                            if let Some(out) = c.try_update_and_attend_on(
                                &queries, &k, &v, lens_ref, self.scale, mask, target,
                            )? {
                                out
                            } else {
                                let (k_full, v_full) =
                                    c.update_and_fetch_for_attention_on(&k, &v, lens_ref, target)?;
                                match mask {
                                    None => mlx::fast::scaled_dot_product_attention_on(
                                        &queries, &k_full, &v_full, self.scale, "causal", None,
                                        None, target,
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
                                }
                            }
                        }
                    } else {
                        let (queries, k) = mrope.apply(&queries, &k, cos, sin)?;
                        let (k, v) = mask_kv(k, v)?;
                        if let Some(out) = c.try_update_and_attend_on(
                            &queries, &k, &v, lens_ref, self.scale, mask, target,
                        )? {
                            out
                        } else {
                            let (k_full, v_full) =
                                c.update_and_fetch_for_attention_on(&k, &v, lens_ref, target)?;
                            match mask {
                                None => mlx::fast::scaled_dot_product_attention_on(
                                    &queries, &k_full, &v_full, self.scale, "causal", None, None,
                                    target,
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
                            }
                        }
                    }
                }
                None => {
                    let (queries, k) = mrope.apply(&queries, &k, cos, sin)?;
                    let (k, v) = mask_kv(k, v)?;
                    match mask {
                        None => mlx::fast::scaled_dot_product_attention_on(
                            &queries, &k, &v, self.scale, "causal", None, None, target,
                        )?,
                        Some(m) => mlx::fast::scaled_dot_product_attention_on(
                            &queries,
                            &k,
                            &v,
                            self.scale,
                            "",
                            Some(m),
                            None,
                            target,
                        )?,
                    }
                }
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

#[allow(clippy::too_many_arguments)]
fn query_position_isolated_attention_on(
    cache: &mut KVCache,
    mrope: &Mrope,
    queries: &Array,
    keys: &Array,
    values: &Array,
    cos: &Array,
    sin: &Array,
    mask: Option<&Array>,
    per_row_lens: &[i32],
    scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    if q_dims.len() != 4 {
        return Err(anyhow!(
            "Qwen exact verify attention expected rank-4 queries, got {q_dims:?}"
        ));
    }
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    if per_row_lens.len() != batch as usize {
        return Err(anyhow!(
            "Qwen exact verify attention expected {batch} row lengths, got {}",
            per_row_lens.len()
        ));
    }
    if query_len <= 1 {
        return Err(anyhow!(
            "Qwen exact verify attention requires Q>1, got Q={query_len}"
        ));
    }
    for (row, &len) in per_row_lens.iter().enumerate() {
        if len < 0 || len > query_len {
            return Err(anyhow!(
                "Qwen exact verify attention invalid row {row} length {len} for Q={query_len}"
            ));
        }
    }

    let key_shape = keys.shape();
    let key_dims = key_shape.as_slice();
    let value_shape = values.shape();
    let value_dims = value_shape.as_slice();
    if key_dims.len() != 4
        || value_dims.len() != 4
        || key_dims[0] != batch
        || value_dims[0] != batch
        || key_dims[2] != query_len
        || key_dims[2] != value_dims[2]
    {
        return Err(anyhow!(
            "Qwen exact verify attention incompatible KV shapes: keys={key_dims:?}, values={value_dims:?}, batch={batch}"
        ));
    }
    let mask_shape = mask.map(Array::shape);
    if let Some(mask_shape) = mask_shape.as_ref() {
        let dims = mask_shape.as_slice();
        if dims.len() != 4
            || (!matches!(dims[0], 1) && dims[0] != batch)
            || dims[2] < query_len
            || dims[3] < query_len
        {
            return Err(anyhow!(
                "Qwen exact verify attention mask {dims:?} cannot cover B={batch}, Q={query_len}"
            ));
        }
    }
    let cos_shape = cos.shape();
    let cos_dims = cos_shape.as_slice();
    let sin_shape = sin.shape();
    let sin_dims = sin_shape.as_slice();
    if cos_dims.len() != 3
        || sin_dims.len() != 3
        || cos_dims[0] != batch
        || sin_dims[0] != batch
        || cos_dims[1] != query_len
        || sin_dims[1] != query_len
        || cos_dims[2] != sin_dims[2]
    {
        return Err(anyhow!(
            "Qwen exact verify attention incompatible MRoPE shapes: cos={cos_dims:?}, sin={sin_dims:?}, B={batch}, Q={query_len}"
        ));
    }

    let mut outputs = Vec::with_capacity(query_len as usize);
    for depth in 0..query_len {
        let query = mlx::ops::indexing::slice_strided_on(
            queries,
            &[0_i32, 0, depth, 0][..],
            &[batch, heads, depth + 1, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let key = mlx::ops::indexing::slice_strided_on(
            keys,
            &[0_i32, 0, depth, 0][..],
            &[batch, key_dims[1], depth + 1, key_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let value = mlx::ops::indexing::slice_strided_on(
            values,
            &[0_i32, 0, depth, 0][..],
            &[batch, value_dims[1], depth + 1, value_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let depth_cos = mlx::ops::indexing::slice_strided_on(
            cos,
            &[0_i32, depth, 0][..],
            &[batch, depth + 1, cos_dims[2]][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let depth_sin = mlx::ops::indexing::slice_strided_on(
            sin,
            &[0_i32, depth, 0][..],
            &[batch, depth + 1, sin_dims[2]][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let step_lens = per_row_lens
            .iter()
            .map(|&len| i32::from(depth < len))
            .collect::<Vec<_>>();
        let key_end = cache
            .offsets()
            .iter()
            .zip(step_lens.iter())
            .map(|(&offset, &len)| offset + len)
            .max()
            .unwrap_or(0);
        let depth_mask = mask
            .map(|mask| {
                let dims = mask_shape
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
        outputs.push(update_and_attend_exact_position_on(
            cache,
            mrope,
            &query,
            &key,
            &value,
            &depth_cos,
            &depth_sin,
            &step_lens,
            scale,
            depth_mask.as_ref(),
            target,
        )?);
    }
    let output_refs = outputs.iter().collect::<Vec<_>>();
    mlx::ops::shape::concatenate_on(&output_refs, 2, target).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn update_and_attend_exact_position_on(
    cache: &mut KVCache,
    mrope: &Mrope,
    query: &Array,
    key: &Array,
    value: &Array,
    cos: &Array,
    sin: &Array,
    per_row_lens: &[i32],
    scale: f32,
    mask: Option<&Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    if let Some(signs) =
        cache.turboquant_pre_rotated_decode_query_signs(query, key, value, per_row_lens, mask)
    {
        let (query_tq, key_tq) =
            mrope.apply_decode_query_turbo_rotation(query, key, cos, sin, signs)?;
        if let Some(output) = cache.try_update_and_attend_decode_pre_rotated_on(
            &query_tq,
            &key_tq,
            value,
            per_row_lens,
            scale,
            mask,
            query.dtype(),
            target,
        )? {
            return Ok(output);
        }
    }

    let (query, key) = mrope.apply(query, key, cos, sin)?;
    if let Some(output) =
        cache.try_update_and_attend_on(&query, &key, value, per_row_lens, scale, mask, target)?
    {
        return Ok(output);
    }
    let (keys, values) =
        cache.update_and_fetch_for_attention_on(&key, value, per_row_lens, target)?;
    match mask {
        Some(mask) => mlx::fast::scaled_dot_product_attention_on(
            &query,
            &keys,
            &values,
            scale,
            "",
            Some(mask),
            None,
            target,
        )
        .map_err(Into::into),
        None => mlx::fast::scaled_dot_product_attention_on(
            &query, &keys, &values, scale, "causal", None, None, target,
        )
        .map_err(Into::into),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::{KVCache, TurboQuantKVBits};
    use mlx::ops::constructors;
    use mlx::{Array, Dtype};
    use serial_test::serial;

    #[test]
    fn format_decode_attention_turbo_profile_line_is_stable_json() {
        let line = format_decode_attention_turbo_profile_line(DecodeAttentionTurboProfileEvent {
            stage: "decode_qkv_proj",
            elapsed_us: 42,
            layer_idx: 3,
            batch: 1,
            seq: 1,
            q_heads: 16,
            kv_heads: 4,
            head_dim: 256,
        });

        assert_eq!(
            line,
            "{\"event\":\"turboquant_gated_attention_stage\",\"stage\":\"decode_qkv_proj\",\"elapsed_us\":42,\"layer_idx\":3,\"batch\":1,\"seq\":1,\"q_heads\":16,\"kv_heads\":4,\"head_dim\":256}"
        );
    }

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
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
    fn from_components_computes_scale() {
        let attn = small_gated_attention();
        // scale = 1 / sqrt(head_dim=8)
        let expected = 1.0 / 8.0_f32.sqrt();
        assert!((attn.scale - expected).abs() < 1e-6);
    }

    #[test]
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
    fn forward_decode_with_turboquant_cache_uses_packed_attention_path() {
        let attn = small_gated_attention();
        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let mut cache = KVCache::new(1, 2, 8, 8, Dtype::Float32, 8)
            .with_step(8)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable turboquant");

        let prefix = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let prefix_cos = constructors::ones((1_i32, 4, 8), Dtype::Float32).unwrap();
        let prefix_sin = Array::zeros((1_i32, 4, 8), Dtype::Float32).unwrap();
        attn.forward_on(
            &prefix,
            &mrope,
            &prefix_cos,
            &prefix_sin,
            None,
            None,
            Some(&[4]),
            Some(&mut cache),
            (),
            0,
        )
        .expect("prefix forward");
        assert_eq!(cache.offsets(), &[4]);

        let decode = Array::zeros((1_i32, 1, 32), Dtype::Float32).unwrap();
        let decode_cos = constructors::ones((1_i32, 1, 8), Dtype::Float32).unwrap();
        let decode_sin = Array::zeros((1_i32, 1, 8), Dtype::Float32).unwrap();
        let out = attn
            .forward_on(
                &decode,
                &mrope,
                &decode_cos,
                &decode_sin,
                None,
                None,
                Some(&[1]),
                Some(&mut cache),
                (),
                0,
            )
            .expect("decode forward with turboquant cache");

        assert_eq!(cache.offsets(), &[5]);
        assert!(cache.turboquant().is_some());
        assert_eq!(out.shape().as_slice(), &[1, 1, 32]);
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    #[serial(mlx_metal)]
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
