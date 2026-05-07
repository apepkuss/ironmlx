//! Qwen3.5 / Qwen3-Next GatedDeltaNet — recurrent SSM with delta rule.
//!
//! T5 (current): the `gated_delta_step` metal_kernel (2 variants — no-mask + masked).
//! Mirrors mlx-lm's `_make_gated_delta_kernel(has_mask)` from
//! `/Volumes/Dev/mlx-lm/mlx_lm/models/gated_delta.py:13-115`.
//!
//! T6 will add the `GatedDeltaNet` main struct around the kernel.
//!
//! Templates: `Dk, Dv, Hk, Hv` (i32), `InT, StT` (Dtype).
//! Grid: `(32, Dv, B * Hv)`; threadgroup: `(32, 4, 1)`.

use mlx::MetalKernel;

use crate::Result;

/// Build the `gated_delta_step` MetalKernel (no-mask or masked variant).
///
/// The shader source is identical between variants except for the per-token
/// guard expression (`mask_clause`). MLX's `metal_kernel` machinery auto-injects
/// `<name>_shape` / `<name>_strides` / `<name>_ndim` for input arrays referenced
/// in the source.
///
/// `T` is passed as a 0-dim int32 array, which MLX treats as `device const
/// int32_t& T` — usable directly as an integer in the shader (e.g.
/// `for (int t = 0; t < T; ++t)`).
// T6 will consume this from GatedDeltaNet; silence the unused-fn lint until then.
#[allow(dead_code)]
pub(crate) fn build_gated_delta_kernel(masked: bool) -> Result<MetalKernel> {
    let mask_clause = if masked {
        "mask[b_idx * T + t]"
    } else {
        "true"
    };
    let src = format!(
        r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_clause}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            // Note: all 32 simdgroup threads write the same zero value here
            // (no `thread_index_in_simdgroup == 0` guard). Matches mlx-lm
            // reference exactly; wasted write bandwidth is acceptable since
            // masked tokens are rare and all writes are identical.
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Advance pointers to the next time step.
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
        "#,
        mask_clause = mask_clause
    );

    let name = if masked {
        "ironmlx_gated_delta_masked"
    } else {
        "ironmlx_gated_delta"
    };

    let inputs: &[&str] = if masked {
        &["q", "k", "v", "g", "beta", "state_in", "T", "mask"]
    } else {
        &["q", "k", "v", "g", "beta", "state_in", "T"]
    };

    Ok(MetalKernel::builder(name)
        .inputs(inputs)
        .outputs(&["y", "state_out"])
        .source(&src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype, Shape};

    #[test]
    fn gated_delta_step_kernel_links() {
        // Dk must be >= 32 so that n_per_t = Dk/32 >= 1 (Metal C++ forbids
        // zero-length arrays). Use Dk=32, Dv=8, Hk=Hv=1, B=1, T=1.
        let kernel = build_gated_delta_kernel(false).expect("build kernel");

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 32), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch");

        let _y = outputs.take_at(0).expect("y");
        let _state = outputs.take_at(0).expect("state");
    }

    #[test]
    fn gated_delta_step_masked_zero_path() {
        // mask=0 everywhere: output should be 0, state unchanged.
        // Use non-zero state_in to verify state isn't accidentally modified.
        // Dk=32 (minimum) so n_per_t = 32/32 = 1 (Metal C++ forbids zero-length arrays).
        let kernel = build_gated_delta_kernel(true).expect("build masked kernel");

        // Initial state has values [1.0; 8*32] (Hv=1, Dv=8, Dk=32).
        let init_state_data: Vec<f32> = (0..256).map(|_| 1.0_f32).collect();
        let state_in: Array = (init_state_data.as_slice(), (1_i32, 1, 8, 32))
            .try_into()
            .unwrap();

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();
        // mask: [B*T] = [1*1 = 1] all-zero (masked out)
        let mask = Array::zeros((1_i32,), Dtype::Bool).unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr, &mask])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch masked");

        let y = outputs.take_at(0).expect("y");
        let state_out = outputs.take_at(0).expect("state_out");

        // y must be all-zero (else branch sets `y[dv_idx] = 0`).
        let y_f32 = mlx::ops::cast::astype(&y, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_f32.to_vec().unwrap();
        assert!(
            yv.iter().all(|x| x.abs() < 1e-6),
            "masked output not zero: {:?}",
            yv
        );

        // state_out must equal state_in (no update under mask=0 — kernel writes
        // back the unchanged register-cached state at the end).
        let sv: Vec<f32> = state_out.to_vec().unwrap();
        assert!(
            sv.iter().all(|x| (x - 1.0).abs() < 1e-6),
            "state changed under mask=0: {:?}",
            sv
        );
    }
}
