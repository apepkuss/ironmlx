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

    gated_delta_ops(q, k, v, &g, &beta, state_ref, mask, stream)
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
