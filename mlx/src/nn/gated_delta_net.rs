//! Gated DeltaNet — SSM-based recurrent architecture used by Qwen3.5 linear attention layers.

use super::activations::silu;
use super::conv1d::Conv1d;
use super::linear::LinearLayer;
use super::module::{Module, get_weight};
use super::norm::RMSNormGated;
use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::fast;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;
use std::collections::HashMap;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// softplus(x) = log(1 + exp(x))
fn softplus(x: &Array, stream: &Stream) -> Result<Array> {
    let one = Array::from_float(1.0);
    let exp_x = ops::exp(x, stream)?;
    let sum = ops::add(&one, &exp_x, stream)?;
    ops::log(&sum, stream)
}

// ── Core functions ──────────────────────────────────────────────────────────

/// Compute gating factor: g = exp(-exp(A_log) * softplus(a + dt_bias))
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array, stream: &Stream) -> Result<Array> {
    // Cast A_log to float32 for precision
    let a_log_f32 = ops::astype(a_log, Dtype::Float32, stream)?;
    let exp_a_log = ops::exp(&a_log_f32, stream)?;
    let a_plus_bias = ops::add(a, dt_bias, stream)?;
    let sp = softplus(&a_plus_bias, stream)?;
    let prod = ops::multiply(&exp_a_log, &sp, stream)?;
    let neg_prod = ops::neg(&prod, stream)?;
    ops::exp(&neg_prod, stream)
}

/// Single recurrent step (reference ops implementation).
///
/// Shapes:
///   - q, k: [B, H, Dk]
///   - v: [B, H, Dv]
///   - g: [B, H]
///   - beta: [B, H]
///   - state: [B, H, Dv, Dk]
///
/// Returns: (y: [B, H, Dv], new_state: [B, H, Dv, Dk])
#[allow(clippy::too_many_arguments)]
fn gated_delta_step_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let old_state = state;

    // Decay: g has ndim==2 → expand [B,H] → [B,H,1,1]
    let decay = if g.ndim() == 2 {
        ops::expand_dims_axes(g, &[2, 3], stream)?
    } else if g.ndim() == 3 {
        // [B, H, Dk] → [B, H, 1, Dk]
        ops::expand_dims(g, 2, stream)?
    } else {
        return Err(crate::error::Error::Mlx(format!(
            "unsupported gating shape ndim={}",
            g.ndim()
        )));
    };
    let state_decayed = ops::multiply(state, &decay, stream)?;

    // k[..., None, :] → expand k [B,H,Dk] → [B,H,1,Dk]
    let k_expanded = ops::expand_dims(k, 2, stream)?;

    // kv_mem = sum(state * k[..., None, :], axis=-1) → [B, H, Dv]
    let state_k = ops::multiply(&state_decayed, &k_expanded, stream)?;
    let kv_mem = ops::sum(&state_k, &[-1], false, stream)?;

    // delta = (v - kv_mem) * beta[..., None]
    let v_minus_kv = ops::subtract(v, &kv_mem, stream)?;
    let beta_expanded = ops::expand_dims(beta, 2, stream)?; // [B,H,1]
    let delta = ops::multiply(&v_minus_kv, &beta_expanded, stream)?;

    // state = state_decayed + k[..., None, :] * delta[..., None]
    // delta [B,H,Dv] → [B,H,Dv,1]
    let delta_expanded = ops::expand_dims(&delta, 3, stream)?;
    let outer = ops::multiply(&k_expanded, &delta_expanded, stream)?;
    let new_state = ops::add(&state_decayed, &outer, stream)?;

    // y = sum(state * q[..., None, :], axis=-1) → [B, H, Dv]
    let q_expanded = ops::expand_dims(q, 2, stream)?;
    let state_q = ops::multiply(&new_state, &q_expanded, stream)?;
    let y = ops::sum(&state_q, &[-1], false, stream)?;

    // Cast y to match q dtype
    let y = ops::astype(&y, q.dtype(), stream)?;

    // Apply mask if present
    let final_state = if let Some(m) = mask {
        // mask [B] → [B,1,1,1]
        let mask_expanded = ops::expand_dims_axes(m, &[1, 2, 3], stream)?;
        ops::where_(&mask_expanded, &new_state, old_state, stream)?
    } else {
        new_state
    };

    Ok((y, final_state))
}

/// Prefill: loop over T timesteps using ops-based recurrence.
///
/// Shapes:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - g: [B, T, Hv]
///   - beta: [B, T, Hv]
///   - state: [B, Hv, Dv, Dk]
///
/// Returns: (y: [B, T, Hv, Dv], state: [B, Hv, Dv, Dk])
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    let b = q_shape[0];
    let t = q_shape[1];
    let hk = q_shape[2];
    let dk = q_shape[3];
    let hv = v_shape[2];
    let dv = v_shape[3];

    // Create zero state if none provided
    let zero_state;
    let state_ref = match state {
        Some(s) => s,
        None => {
            zero_state = ops::zeros(&[b, hv, dv, dk], Dtype::Float32, stream)?;
            &zero_state
        }
    };
    let mut current_state = state_ref.clone();

    // If Hv > Hk, repeat q and k along head dim
    let q_rep;
    let k_rep;
    let (q_use, k_use) = if hv > hk {
        let repeat_factor = hv / hk;
        q_rep = ops::repeat_axis(q, repeat_factor, 2, stream)?;
        k_rep = ops::repeat_axis(k, repeat_factor, 2, stream)?;
        (&q_rep, &k_rep)
    } else {
        (q, k)
    };

    let mut ys = Vec::with_capacity(t as usize);
    for ti in 0..t {
        // Slice timestep t: [:, t, :, :] → [B, H, D]
        let q_t = ops::slice(
            q_use,
            &[0, ti, 0, 0],
            &[b, ti + 1, hv, dk],
            &[1, 1, 1, 1],
            stream,
        )?;
        let q_t = ops::squeeze_axis(&q_t, 1, stream)?;
        let k_t = ops::slice(
            k_use,
            &[0, ti, 0, 0],
            &[b, ti + 1, hv, dk],
            &[1, 1, 1, 1],
            stream,
        )?;
        let k_t = ops::squeeze_axis(&k_t, 1, stream)?;
        let v_t = ops::slice(
            v,
            &[0, ti, 0, 0],
            &[b, ti + 1, hv, dv],
            &[1, 1, 1, 1],
            stream,
        )?;
        let v_t = ops::squeeze_axis(&v_t, 1, stream)?;

        // g[:, t, :] → [B, Hv]
        let g_t = ops::slice(g, &[0, ti, 0], &[b, ti + 1, hv], &[1, 1, 1], stream)?;
        let g_t = ops::squeeze_axis(&g_t, 1, stream)?;

        // beta[:, t, :] → [B, Hv]
        let beta_t = ops::slice(beta, &[0, ti, 0], &[b, ti + 1, hv], &[1, 1, 1], stream)?;
        let beta_t = ops::squeeze_axis(&beta_t, 1, stream)?;

        // mask[:, t] → [B] if present
        let mask_t = if let Some(m) = mask {
            let m_t = ops::slice(m, &[0, ti], &[b, ti + 1], &[1, 1], stream)?;
            Some(ops::squeeze_axis(&m_t, 1, stream)?)
        } else {
            None
        };

        let (y_t, new_state) = gated_delta_step_ops(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &current_state,
            mask_t.as_ref(),
            stream,
        )?;
        current_state = new_state;
        ys.push(y_t);
    }

    // Stack outputs: list of [B, Hv, Dv] → [B, T, Hv, Dv]
    let y_refs: Vec<&Array> = ys.iter().collect();
    let y_vec = VectorArray::from_arrays(&y_refs);
    let y = ops::stack(&y_vec, 1, stream)?;

    Ok((y, current_state))
}

/// Unified entry point: compute g from A_log/a/dt_bias, then call ops implementation.
///
/// Shapes:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - a: [B, T, Hv]
///   - b: [B, T, Hv]
// ── Metal Kernel implementation (kept for future optimization) ──────────────
// TODO: Metal kernel produces incorrect output — needs debugging.
// The shader source and wrapper are correct in structure but likely have
// a data layout or T parameter passing issue.
#[allow(dead_code)]
pub(crate) mod metal_kernel {
    use crate::array::Array;
    use crate::error::Result;
    use crate::fast_kernel::{MetalKernel, MetalKernelConfig};
    use crate::metal;
    use crate::stream::Stream;
    use std::sync::OnceLock;

    /// Metal shader source for gated delta step (scalar g, no mask).
    const METAL_SHADER_SCALAR: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y += Hv * Dv;
      g_ += Hv; beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
"#;

    /// Metal shader source for gated delta step (scalar g, with mask).
    const METAL_SHADER_SCALAR_MASKED: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      if (mask[b_idx * T + t]) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[hv_idx];
          kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[dv_idx] = static_cast<InT>(out);
        }
      }
      q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y += Hv * Dv;
      g_ += Hv; beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
"#;

    /// Cached Metal kernels (created once, reused).
    static KERNEL_SCALAR: OnceLock<Option<MetalKernel>> = OnceLock::new();
    static KERNEL_SCALAR_MASKED: OnceLock<Option<MetalKernel>> = OnceLock::new();

    fn get_kernel_scalar() -> Option<&'static MetalKernel> {
        KERNEL_SCALAR
            .get_or_init(|| {
                if !metal::is_available().unwrap_or(false) {
                    return None;
                }
                MetalKernel::new(
                    "gated_delta_step",
                    &["q", "k", "v", "g", "beta", "state_in", "T"],
                    &["y", "state_out"],
                    METAL_SHADER_SCALAR,
                    "",
                    true,
                    false,
                )
                .ok()
            })
            .as_ref()
    }

    fn get_kernel_scalar_masked() -> Option<&'static MetalKernel> {
        KERNEL_SCALAR_MASKED
            .get_or_init(|| {
                if !metal::is_available().unwrap_or(false) {
                    return None;
                }
                MetalKernel::new(
                    "gated_delta_step_mask",
                    &["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
                    &["y", "state_out"],
                    METAL_SHADER_SCALAR_MASKED,
                    "",
                    true,
                    false,
                )
                .ok()
            })
            .as_ref()
    }

    /// Execute gated delta update using Metal kernel (GPU-accelerated).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gated_delta_metal_kernel(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let k_shape = k.shape();
        let v_shape = v.shape();
        let b = k_shape[0];
        let t = k_shape[1];
        let hk = k_shape[2];
        let dk = k_shape[3];
        let hv = v_shape[2];
        let dv = v_shape[3];

        let input_dtype = q.dtype();
        let state_dtype = state.dtype();
        let t_arr = Array::from_int(t);

        let config = MetalKernelConfig::new();
        config.add_output_arg(&[b, t, hv, dv], input_dtype)?;
        config.add_output_arg(&state.shape(), state_dtype)?;
        config.set_grid([32, dv, b * hv])?;
        config.set_thread_group([32, 4, 1])?;
        config.add_template_arg_dtype("InT", input_dtype)?;
        config.add_template_arg_dtype("StT", state_dtype)?;
        config.add_template_arg_int("Dk", dk)?;
        config.add_template_arg_int("Dv", dv)?;
        config.add_template_arg_int("Hk", hk)?;
        config.add_template_arg_int("Hv", hv)?;

        let outputs = if let Some(m) = mask {
            let kernel = get_kernel_scalar_masked()
                .ok_or_else(|| crate::error::Error::Mlx("Metal not available".into()))?;
            kernel.apply(&[q, k, v, g, beta, state, &t_arr, m], &config, stream)?
        } else {
            let kernel = get_kernel_scalar()
                .ok_or_else(|| crate::error::Error::Mlx("Metal not available".into()))?;
            kernel.apply(&[q, k, v, g, beta, state, &t_arr], &config, stream)?
        };

        if outputs.len() < 2 {
            return Err(crate::error::Error::Mlx(
                "Metal kernel returned fewer than 2 outputs".into(),
            ));
        }

        Ok((outputs[0].clone(), outputs[1].clone()))
    }
} // mod metal_kernel

///   - a_log: [Hv]
///   - dt_bias: [Hv]
///   - state: Option<[B, Hv, Dv, Dk]>
///   - mask: Option<[B, T]>
///
/// Returns: (y: [B, T, Hv, Dv], state: [B, Hv, Dv, Dk])
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let beta = ops::sigmoid(b, stream)?;
    let g = compute_g(a_log, a, dt_bias, stream)?;

    let q_shape = q.shape();
    let v_shape = v.shape();

    let owned_state;
    let state_ref = match state {
        Some(s) => Some(s),
        None => {
            let b_dim = q_shape[0];
            let dk = q_shape[3];
            let hv = v_shape[2];
            let dv = v_shape[3];
            owned_state = ops::zeros(&[b_dim, hv, dv, dk], Dtype::Float32, stream)?;
            Some(&owned_state)
        }
    };

    let state_arr = state_ref.unwrap();

    let state_arr_ref: &Array = state_arr;

    // Ensure all inputs have consistent dtypes for Metal kernel
    let input_dtype = q.dtype();
    let v_cast = if v.dtype() != input_dtype {
        ops::astype(v, input_dtype, stream)?
    } else {
        v.clone()
    };
    let beta_cast = if beta.dtype() != input_dtype {
        ops::astype(&beta, input_dtype, stream)?
    } else {
        beta.clone()
    };
    let g_cast = if g.dtype() != input_dtype {
        ops::astype(&g, input_dtype, stream)?
    } else {
        g.clone()
    };

    // Use Metal kernel for prefill (T > 1) when available
    let t = q.shape()[1];
    if t > 1 && crate::metal::is_available().unwrap_or(false) && g_cast.ndim() <= 3 {
        let hk = q.shape()[2];
        let hv = v.shape()[2];
        if hv % hk == 0
            && let Ok(result) = metal_kernel::gated_delta_metal_kernel(
                q,
                k,
                &v_cast,
                &g_cast,
                &beta_cast,
                state_arr_ref,
                mask,
                stream,
            )
        {
            return Ok(result);
        }
    }

    gated_delta_ops(
        q,
        k,
        &v_cast,
        &g_cast,
        &beta_cast,
        Some(state_arr_ref),
        mask,
        stream,
    )
}

// ── GatedDeltaNet module ────────────────────────────────────────────────────

/// Gated DeltaNet module — SSM-based recurrent layer for Qwen3.5's linear attention.
pub struct GatedDeltaNet {
    pub in_proj_qkv: LinearLayer,
    pub in_proj_z: LinearLayer,
    pub in_proj_b: LinearLayer,
    pub in_proj_a: LinearLayer,
    pub conv1d: Conv1d,
    pub a_log: Array,
    pub dt_bias: Array,
    pub norm: RMSNormGated,
    pub out_proj: LinearLayer,
    // config
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_kernel_size: usize,
}

impl GatedDeltaNet {
    /// Forward pass with cache for incremental decoding.
    ///
    /// - `x`: [B, T, hidden_size]
    /// - `mask`: optional [B, T] SSM mask (bool)
    /// - `cache`: (conv_state, ssm_state) — both optional
    ///
    /// Returns output [B, T, hidden_size], mutates cache in-place.
    #[allow(clippy::type_complexity)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut (Option<Array>, Option<Array>),
        stream: &Stream,
    ) -> Result<Array> {
        let x_shape = x.shape();
        let b = x_shape[0];
        let t = x_shape[1];
        let hv = self.num_v_heads;
        let hk = self.num_k_heads;
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let conv_dim = self.key_dim * 2 + self.value_dim;

        // 1. Projections
        let qkv = self.in_proj_qkv.forward_with_stream(x, stream)?;
        let z_proj = self.in_proj_z.forward_with_stream(x, stream)?;
        let z = ops::reshape(&z_proj, &[b, t, hv, dv], stream)?;
        let b_proj = self.in_proj_b.forward_with_stream(x, stream)?;
        let a_proj = self.in_proj_a.forward_with_stream(x, stream)?;

        // 2. Conv state
        let conv_state = match &cache.0 {
            Some(cs) => cs.clone(),
            None => {
                let kernel_minus_1 = (self.conv_kernel_size as i32) - 1;
                ops::zeros(&[b, kernel_minus_1, conv_dim], x.dtype(), stream)?
            }
        };

        // 3. Apply mask to qkv if present
        let qkv = if let Some(m) = mask {
            let m_expanded = ops::expand_dims(m, 2, stream)?; // [B, T, 1]
            let zero = Array::from_float(0.0);
            ops::where_(&m_expanded, &qkv, &zero, stream)?
        } else {
            qkv
        };

        // 4. Conv input = cat([conv_state, qkv], axis=1)
        let cat_vec = VectorArray::from_arrays(&[&conv_state, &qkv]);
        let conv_input = ops::concatenate(&cat_vec, 1, stream)?;

        // Update cache[0] = last (kernel_size - 1) rows
        let kernel_minus_1 = (self.conv_kernel_size as i32) - 1;
        let conv_input_shape = conv_input.shape();
        let conv_t = conv_input_shape[1];
        let start_t = conv_t - kernel_minus_1;
        cache.0 = Some(ops::slice(
            &conv_input,
            &[0, start_t, 0],
            &[b, conv_t, conv_dim],
            &[1, 1, 1],
            stream,
        )?);

        // 5. conv_out = silu(conv1d(conv_input))
        let conv_raw = self.conv1d.forward_no_pad(&conv_input, stream)?;
        let conv_out = silu(&conv_raw, stream)?;

        // 6. Split conv_out → q, k, v
        let split_indices = &[self.key_dim, 2 * self.key_dim];
        let parts = ops::split_at_indices(&conv_out, split_indices, -1, stream)?;
        let q_raw = parts.get(0)?;
        let k_raw = parts.get(1)?;
        let v_raw = parts.get(2)?;

        let q = ops::reshape(&q_raw, &[b, t, hk, dk], stream)?;
        let k = ops::reshape(&k_raw, &[b, t, hk, dk], stream)?;
        let v = ops::reshape(&v_raw, &[b, t, hv, dv], stream)?;

        // 7. Q,K RMS scaling
        let inv_scale = (dk as f32).powf(-0.5);
        let inv_scale_sq = Array::from_float(inv_scale * inv_scale);
        let inv_scale_arr = Array::from_float(inv_scale);

        let q_norm = fast::rms_norm(&q, None, 1e-6, stream)?;
        let q = ops::multiply(&inv_scale_sq, &q_norm, stream)?;
        let k_norm = fast::rms_norm(&k, None, 1e-6, stream)?;
        let k = ops::multiply(&inv_scale_arr, &k_norm, stream)?;

        // 8. Gated delta update
        let (out, state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a_proj,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            cache.1.as_ref(),
            mask,
            stream,
        )?;

        // 9. Update cache[1] = state
        cache.1 = Some(state);

        // 10. out = norm(out, z) — RMSNormGated
        let out = self.norm.forward_with_stream(&out, &z, stream)?;

        // 11. out = out_proj(out.reshape(B, T, -1))
        let out = ops::reshape(&out, &[b, t, -1], stream)?;
        self.out_proj.forward_with_stream(&out, stream)
    }
}

impl Module for GatedDeltaNet {
    fn forward(&self, _x: &Array) -> Result<Array> {
        Err(crate::error::Error::Mlx(
            "GatedDeltaNet requires forward_with_cache; use forward_with_cache instead".into(),
        ))
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        for (name, arr) in self.in_proj_qkv.parameters() {
            params.push((format!("in_proj_qkv.{}", name), arr));
        }
        for (name, arr) in self.in_proj_z.parameters() {
            params.push((format!("in_proj_z.{}", name), arr));
        }
        for (name, arr) in self.in_proj_b.parameters() {
            params.push((format!("in_proj_b.{}", name), arr));
        }
        for (name, arr) in self.in_proj_a.parameters() {
            params.push((format!("in_proj_a.{}", name), arr));
        }
        for (name, arr) in self.conv1d.parameters() {
            params.push((format!("conv1d.{}", name), arr));
        }
        params.push(("A_log".to_string(), &self.a_log));
        params.push(("dt_bias".to_string(), &self.dt_bias));
        params.push(("norm.weight".to_string(), &self.norm.weight));
        for (name, arr) in self.out_proj.parameters() {
            params.push((format!("out_proj.{}", name), arr));
        }
        params
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        let pfx = |sub: &str| {
            if prefix.is_empty() {
                sub.to_string()
            } else {
                format!("{}.{}", prefix, sub)
            }
        };

        self.in_proj_qkv
            .load_weights(weights, &pfx("in_proj_qkv"))?;
        self.in_proj_z.load_weights(weights, &pfx("in_proj_z"))?;
        self.in_proj_b.load_weights(weights, &pfx("in_proj_b"))?;
        self.in_proj_a.load_weights(weights, &pfx("in_proj_a"))?;
        self.conv1d.load_weights(weights, &pfx("conv1d"))?;
        self.a_log = get_weight(weights, prefix, "A_log")?;
        self.dt_bias = get_weight(weights, prefix, "dt_bias")?;
        self.norm.weight = get_weight(weights, &pfx("norm"), "weight")?;
        self.out_proj.load_weights(weights, &pfx("out_proj"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// Compare ops vs Metal kernel output on a minimal known input.
    #[test]
    fn test_metal_kernel_vs_ops() {
        crate::init();
        let stream = Stream::new(&Device::gpu());

        // Minimal dims: B=1, T=2, Hk=1, Hv=2, Dk=32, Dv=4
        // Dk must be multiple of 32 (n_per_t = Dk/32 = 1)
        let b = 1i32;
        let t = 2i32;
        let hk = 1i32;
        let hv = 2i32;
        let dk = 32i32;
        let dv = 4i32;

        // Create simple test data
        let q_data: Vec<f32> = (0..b * t * hk * dk).map(|i| (i as f32) * 0.01).collect();
        let k_data: Vec<f32> = (0..b * t * hk * dk).map(|i| (i as f32) * 0.01).collect();
        let v_data: Vec<f32> = (0..b * t * hv * dv).map(|i| (i as f32) * 0.1).collect();
        let g_data: Vec<f32> = vec![0.9; (b * t * hv) as usize]; // decay ~0.9
        let beta_data: Vec<f32> = vec![0.5; (b * t * hv) as usize];
        let state_data: Vec<f32> = vec![0.0; (b * hv * dv * dk) as usize];

        let q = Array::from_slice_f32_shape(&q_data, &[b, t, hk, dk]);
        let k = Array::from_slice_f32_shape(&k_data, &[b, t, hk, dk]);
        let v = Array::from_slice_f32_shape(&v_data, &[b, t, hv, dv]);
        let g = Array::from_slice_f32_shape(&g_data, &[b, t, hv]);
        let beta = Array::from_slice_f32_shape(&beta_data, &[b, t, hv]);
        let state = Array::from_slice_f32_shape(&state_data, &[b, hv, dv, dk]);

        // Run ops version
        let (y_ops, state_ops) =
            gated_delta_ops(&q, &k, &v, &g, &beta, Some(&state), None, &stream)
                .expect("ops failed");
        y_ops.eval().unwrap();
        state_ops.eval().unwrap();

        println!("=== OPS ===");
        println!("y_ops shape: {:?}", y_ops.shape());
        println!("y_ops: {:?}", y_ops.to_vec_f32().unwrap());
        println!("state_ops shape: {:?}", state_ops.shape());

        // Run Metal kernel version
        let metal_result =
            metal_kernel::gated_delta_metal_kernel(&q, &k, &v, &g, &beta, &state, None, &stream);

        match metal_result {
            Ok((y_metal, state_metal)) => {
                y_metal.eval().unwrap();
                state_metal.eval().unwrap();

                println!("\n=== METAL ===");
                println!("y_metal shape: {:?}", y_metal.shape());
                println!("y_metal: {:?}", y_metal.to_vec_f32().unwrap());
                println!("state_metal shape: {:?}", state_metal.shape());

                // Compare
                let y_ops_vec = y_ops.to_vec_f32().unwrap();
                let y_metal_vec = y_metal.to_vec_f32().unwrap();
                assert_eq!(y_ops_vec.len(), y_metal_vec.len(), "y length mismatch");

                let mut max_diff: f32 = 0.0;
                for (i, (a, b)) in y_ops_vec.iter().zip(y_metal_vec.iter()).enumerate() {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if diff > 0.01 {
                        println!("  MISMATCH at {}: ops={}, metal={}, diff={}", i, a, b, diff);
                    }
                }
                println!("\nMax diff: {}", max_diff);
                assert!(
                    max_diff < 0.01,
                    "Metal kernel output differs from ops by {}",
                    max_diff
                );
            }
            Err(e) => {
                println!("Metal kernel failed: {}", e);
                println!("(This is expected if Metal is not available)");
            }
        }
    }
}
