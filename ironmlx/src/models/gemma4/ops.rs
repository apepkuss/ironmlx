use anyhow::anyhow;
#[cfg(test)]
use mlx::ops;
use mlx::{Array, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

use crate::Result;

#[cfg(test)]
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

pub fn rms_norm_no_scale_on(
    x: &Array,
    eps: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    Ok(mlx::fast::rms_norm_on(x, None, eps, target)?)
}

#[cfg(test)]
fn gelu_approx_on(x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let target = target.into();
    let three: Array = (&[3_i32][..], ()).try_into()?;
    let x3 = x.power_on(&three, target)?;

    let dtype = x.dtype();
    let c_044715: Array = ops::cast::astype_on(&scalar_f32(0.044_715)?, dtype, target)?;
    let c_sqrt2pi: Array = ops::cast::astype_on(&scalar_f32(SQRT_2_OVER_PI)?, dtype, target)?;
    let c_half: Array = ops::cast::astype_on(&scalar_f32(0.5)?, dtype, target)?;
    let c_one: Array = ops::cast::astype_on(&scalar_f32(1.0)?, dtype, target)?;

    let inner = (&(&x3 * &c_044715) + x) * &c_sqrt2pi;
    let t = inner.tanh_on(target)?;
    Ok(x * &c_half * (&t + &c_one))
}

pub fn gelu_approx_mul_on(
    gate: &Array,
    up: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if gate.shape() != up.shape() {
        return Err(anyhow!(
            "Gemma4 GeGLU fused activation shape mismatch: gate {} vs up {}",
            gate.shape(),
            up.shape()
        ));
    }

    let target = target.into();
    let shape = gate.shape();
    let size = i32::try_from(shape.numel()).map_err(|_| {
        anyhow!(
            "Gemma4 GeGLU fused activation input too large: {} elements",
            shape.numel()
        )
    })?;
    if size == 0 {
        return Ok(Array::zeros_on(shape, gate.dtype(), target)?);
    }

    let kernel = geglu_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[gate, up])
        .output_shapes(&[shape])
        .output_dtypes(&[gate.dtype()])
        .grid(size, 1, 1)
        .threadgroup(256.min(size), 1, 1)
        .stream(target)
        .template_int("SIZE", size)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}

pub(crate) struct QuantizedGateUpGeGluDecode<'a> {
    pub(crate) weight: &'a Array,
    pub(crate) scales: &'a Array,
    pub(crate) biases: &'a Array,
    pub(crate) intermediate_size: i32,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
}

pub fn quantized_gate_up_geglu_decode_on(
    x: &Array,
    params: QuantizedGateUpGeGluDecode<'_>,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    if params.group_size != 64 || params.bits != 4 || params.intermediate_size <= 0 {
        return Ok(None);
    }

    let x_shape = x.shape();
    let x_dims = x_shape.as_slice();
    let Some((&k, leading)) = x_dims.split_last() else {
        return Ok(None);
    };
    let m_total: i32 = leading.iter().product();
    if m_total != 1 || k % 512 != 0 {
        return Ok(None);
    }

    let weight_dims = params.weight.shape();
    let weight_dims = weight_dims.as_slice();
    if weight_dims.len() != 2
        || weight_dims[0] != params.intermediate_size * 2
        || weight_dims[1] != k / 8
    {
        return Ok(None);
    }

    let scale_dims = params.scales.shape();
    let scale_dims = scale_dims.as_slice();
    let bias_dims = params.biases.shape();
    let bias_dims = bias_dims.as_slice();
    if scale_dims != [params.intermediate_size * 2, k / params.group_size]
        || bias_dims != [params.intermediate_size * 2, k / params.group_size]
    {
        return Ok(None);
    }

    let mut out_dims = leading.to_vec();
    out_dims.push(params.intermediate_size);
    let out_shape = Shape::from(out_dims);

    let target = target.into();
    let kernel = qmv_geglu_kernel()?;
    let row_tiles = (params.intermediate_size + 7) / 8;
    // MLX metal_kernel grid uses dispatch_threads: total threads, not threadgroups.
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, params.weight, params.scales, params.biases])
        .output_shapes(&[out_shape])
        .output_dtypes(&[x.dtype()])
        .grid(32, row_tiles * 2, 1)
        .threadgroup(32, 2, 1)
        .stream(target)
        .template_int("K", k)
        .template_int("N2", params.intermediate_size)
        .dispatch()?;
    Ok(Some(outputs.take_at(0)?))
}

pub(crate) struct RmsNormDefaultRopeDecode<'a> {
    pub(crate) weight: &'a Array,
    pub(crate) offsets: &'a Array,
    pub(crate) eps: f32,
    pub(crate) base: f32,
    pub(crate) traditional: bool,
    pub(crate) heads: i32,
    pub(crate) head_dim: i32,
}

pub(crate) fn rms_norm_default_rope_decode_on(
    x: &Array,
    params: RmsNormDefaultRopeDecode<'_>,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    if (params.eps - 1.0e-6).abs() > 1.0e-12 || (params.base - 10_000.0).abs() > 0.01 {
        return Ok(None);
    }
    if params.heads <= 0
        || params.head_dim <= 0
        || params.head_dim > 512
        || !(params.head_dim as u32).is_power_of_two()
    {
        return Ok(None);
    }
    let x_shape = x.shape();
    let x_dims = x_shape.as_slice();
    if x_dims != [1, 1, params.heads, params.head_dim] {
        return Ok(None);
    }
    if params.weight.shape().as_slice() != [params.head_dim] {
        return Ok(None);
    }
    if params.offsets.shape().as_slice() != [1] {
        return Ok(None);
    }

    let target = target.into();
    let out_shape = Shape::from(vec![1_i32, params.heads, 1, params.head_dim]);
    let kernel = rms_norm_default_rope_decode_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, params.weight, params.offsets])
        .output_shapes(&[out_shape])
        .output_dtypes(&[x.dtype()])
        .grid(params.head_dim * params.heads, 1, 1)
        .threadgroup(params.head_dim, 1, 1)
        .stream(target)
        .template_int("D", params.head_dim)
        .template_bool("TRADITIONAL", params.traditional)
        .dispatch()?;
    Ok(Some(outputs.take_at(0)?))
}

fn geglu_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        uint gid = thread_position_in_grid.x;
        if (gid >= SIZE) {
            return;
        }

        float x = float(gate[gid]);
        float u = float(up[gid]);
        float x2 = x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x2);
        float gelu = 0.5f * x * (1.0f + tanh(inner));
        out[gid] = static_cast<__typeof__(*out)>(gelu * u);
    "#;

    let kernel = MetalKernel::builder("ironmlx_gemma4_geglu")
        .inputs(&["gate", "up"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn rms_norm_default_rope_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        constexpr float EPS = 1.0e-6f;
        constexpr float BASE = 10000.0f;

        uint h = threadgroup_position_in_grid.x;
        uint d = thread_index_in_threadgroup;

        const device __typeof__(*x)* x_head = x + int(h) * D;
        device __typeof__(*out)* out_head = out + int(h) * D;

        threadgroup float vals[512];
        float xv = float(x_head[d]);
        vals[d] = xv * xv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = D / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                vals[d] += vals[d + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inv_rms = rsqrt(vals[0] / float(D) + EPS);
        vals[d] = xv * inv_rms * float(weight[d]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int pos = int(offsets[0]);
        float outv;
        if (TRADITIONAL) {
            uint pair = d >> 1;
            uint mate = (d & 1u) == 0u ? d + 1u : d - 1u;
            float freq = pow(BASE, -2.0f * float(pair) / float(D));
            float angle = float(pos) * freq;
            float cs = cos(angle);
            float sn = sin(angle);
            outv = (d & 1u) == 0u
                ? vals[d] * cs - vals[mate] * sn
                : vals[d] * cs + vals[mate] * sn;
        } else {
            uint half_dim = D / 2;
            uint j = d < half_dim ? d : d - half_dim;
            uint mate = d < half_dim ? d + half_dim : d - half_dim;
            float freq = pow(BASE, -2.0f * float(j) / float(D));
            float angle = float(pos) * freq;
            float cs = cos(angle);
            float sn = sin(angle);
            outv = d < half_dim
                ? vals[d] * cs - vals[mate] * sn
                : vals[d] * cs + vals[mate] * sn;
        }
        out_head[d] = static_cast<__typeof__(*out)>(outv);
    "#;

    let kernel = MetalKernel::builder("ironmlx_gemma4_rms_norm_default_rope_decode")
        .inputs(&["x", "weight", "offsets"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn qmv_geglu_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
        constexpr int SCALE_STEP_PER_THREAD = 64 / VALUES_PER_THREAD;
        constexpr int RESULTS_PER_SIMDGROUP = 4;
        constexpr int NUM_SIMDGROUPS = 2;

        uint m_idx = threadgroup_position_in_grid.x;
        uint row_tile = threadgroup_position_in_grid.y;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        int out_row_base = int(row_tile) * (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP) +
                           int(sgid) * RESULTS_PER_SIMDGROUP;
        int row_bytes = K / 2;
        int groups_per_row = K / 64;

        const device uchar* w_bytes = reinterpret_cast<const device uchar*>(w);
        const device __typeof__(*x)* x_row = x + int(m_idx) * K;
        device __typeof__(*out)* out_row = out + int(m_idx) * N2;

        thread float x_thread[VALUES_PER_THREAD];
        thread float gate_acc[RESULTS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float up_acc[RESULTS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            float x_sum = 0.0f;
            const int x_base = k0 + int(lane) * VALUES_PER_THREAD;

            #pragma clang loop unroll(full)
            for (int i = 0; i < VALUES_PER_THREAD; ++i) {
                float xv = float(x_row[x_base + i]);
                x_sum += xv;
                if ((i & 3) == 0) {
                    x_thread[i] = xv;
                } else if ((i & 3) == 1) {
                    x_thread[i] = xv / 16.0f;
                } else if ((i & 3) == 2) {
                    x_thread[i] = xv / 256.0f;
                } else {
                    x_thread[i] = xv / 4096.0f;
                }
            }

            const int byte_offset = (k0 / 2) + int(lane) * 8;
            const int group_offset = (k0 / 64) + int(lane) / SCALE_STEP_PER_THREAD;

            #pragma clang loop unroll(full)
            for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
                int n = out_row_base + r;
                if (n < N2) {
                    int gate_n = n;
                    int up_n = n + N2;

                    const device ushort* gate_w =
                        reinterpret_cast<const device ushort*>(
                            w_bytes + gate_n * row_bytes + byte_offset);
                    const device ushort* up_w =
                        reinterpret_cast<const device ushort*>(
                            w_bytes + up_n * row_bytes + byte_offset);

                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int p = 0; p < 4; ++p) {
                        ushort gw = gate_w[p];
                        ushort uw = up_w[p];
                        int xi = p * 4;
                        gate_dot +=
                            x_thread[xi + 0] * float(gw & 0x000fu) +
                            x_thread[xi + 1] * float(gw & 0x00f0u) +
                            x_thread[xi + 2] * float(gw & 0x0f00u) +
                            x_thread[xi + 3] * float(gw & 0xf000u);
                        up_dot +=
                            x_thread[xi + 0] * float(uw & 0x000fu) +
                            x_thread[xi + 1] * float(uw & 0x00f0u) +
                            x_thread[xi + 2] * float(uw & 0x0f00u) +
                            x_thread[xi + 3] * float(uw & 0xf000u);
                    }

                    int gate_sb = gate_n * groups_per_row + group_offset;
                    int up_sb = up_n * groups_per_row + group_offset;
                    gate_acc[r] += float(scales[gate_sb]) * gate_dot +
                                   x_sum * float(biases[gate_sb]);
                    up_acc[r] += float(scales[up_sb]) * up_dot +
                                 x_sum * float(biases[up_sb]);
                }
            }
        }

        #pragma clang loop unroll(full)
        for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
            int n = out_row_base + r;
            float gate = simd_sum(gate_acc[r]);
            float up = simd_sum(up_acc[r]);
            if (lane == 0 && n < N2) {
                float gate_rounded = float(static_cast<__typeof__(*out)>(gate));
                float up_rounded = float(static_cast<__typeof__(*out)>(up));
                float gate2 = gate_rounded * gate_rounded;
                float inner = 0.7978845608028654f *
                              (gate_rounded + 0.044715f * gate_rounded * gate2);
                float gelu = 0.5f * gate_rounded * (1.0f + tanh(inner));
                out_row[n] = static_cast<__typeof__(*out)>(gelu * up_rounded);
            }
        }
    "#;

    let kernel = MetalKernel::builder("ironmlx_gemma4_qmv_geglu_decode")
        .inputs(&["x", "w", "scales", "biases"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

pub fn logit_softcap_on(
    logits: &Array,
    softcap: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if softcap <= 0.0 {
        return Err(anyhow!("Gemma4 logit softcap must be > 0, got {softcap}"));
    }
    let target = target.into();
    let capped = (logits / softcap).tanh_on(target)?;
    Ok(&capped * softcap)
}

#[cfg(test)]
fn scalar_f32(v: f32) -> Result<Array> {
    Ok((&[v][..], ()).try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;
    use serial_test::serial;

    fn assert_all_close(got: &Array, expected: &Array, tol: f32) {
        let got_f32 = ops::cast::astype(got, Dtype::Float32).unwrap();
        let expected_f32 = ops::cast::astype(expected, Dtype::Float32).unwrap();
        let got_v = got_f32.to_vec::<f32>().unwrap();
        let expected_v = expected_f32.to_vec::<f32>().unwrap();
        assert_eq!(got_v.len(), expected_v.len());
        for (idx, (g, e)) in got_v.iter().zip(expected_v.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= tol,
                "idx={idx} got={g} expected={e} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn geglu_fused_matches_composed_f32() {
        let gate: Array = (
            &[-3.0_f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let up: Array = (
            &[0.5_f32, -2.0, 1.5, 3.0, -0.75, 2.5, -1.25, 0.25][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();

        let expected = &gelu_approx_on(&gate, ()).unwrap() * &up;
        let got = gelu_approx_mul_on(&gate, &up, ()).unwrap();
        assert_all_close(&got, &expected, 1e-5);
    }

    #[test]
    #[serial(mlx_metal)]
    fn geglu_fused_accepts_split_views() {
        let gate: Array = (
            &[0.25_f32, -1.0, 2.0, 0.5, -0.75, 1.25][..],
            &[2_i32, 3][..],
        )
            .try_into()
            .unwrap();
        let up: Array = (&[1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75][..], &[2_i32, 3][..])
            .try_into()
            .unwrap();
        let fused = ops::shape::concatenate(&[&gate, &up], 1).unwrap();
        let parts = ops::shape::split_at(&fused, &[3_i32][..], 1).unwrap();

        let expected = &gelu_approx_on(&parts[0], ()).unwrap() * &parts[1];
        let got = gelu_approx_mul_on(&parts[0], &parts[1], ()).unwrap();
        assert_all_close(&got, &expected, 1e-5);
    }

    #[test]
    #[serial(mlx_metal)]
    fn geglu_fused_matches_composed_bf16_with_final_bf16_tolerance() {
        let gate_f32: Array = (
            &[-3.0_f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let up_f32: Array = (
            &[0.5_f32, -2.0, 1.5, 3.0, -0.75, 2.5, -1.25, 0.25][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let gate = ops::cast::astype(&gate_f32, Dtype::Bfloat16).unwrap();
        let up = ops::cast::astype(&up_f32, Dtype::Bfloat16).unwrap();

        let expected = &gelu_approx_on(&gate, ()).unwrap() * &up;
        let got = gelu_approx_mul_on(&gate, &up, ()).unwrap();
        assert_all_close(&got, &expected, 0.02);
    }

    #[test]
    #[serial(mlx_metal)]
    fn qmv_geglu_decode_matches_composed_bf16() {
        let n = 8_i32;
        let k = 512_i32;
        let group_size = 64_i32;
        let bits = 4_i32;

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.0025).collect();
        let x_f32: Array = (x_data.as_slice(), (1_i32, k)).try_into().unwrap();
        let x = ops::cast::astype(&x_f32, Dtype::Bfloat16).unwrap();

        let w_data: Vec<f32> = (0..(2 * n * k))
            .map(|i| ((i % 37) as f32 - 18.0) * 0.003)
            .collect();
        let raw_w_f32: Array = (w_data.as_slice(), (2 * n, k)).try_into().unwrap();
        let raw_w = ops::cast::astype(&raw_w_f32, Dtype::Bfloat16).unwrap();
        let q = mlx::quantization::quantize(&raw_w, Some(group_size), Some(bits), "affine", None)
            .unwrap();
        let weight = &q[0];
        let scales = &q[1];
        let biases = &q[2];

        let projected = mlx::quantization::quantized_matmul(
            &x,
            weight,
            scales,
            Some(biases),
            /* transpose = */ true,
            Some(group_size),
            Some(bits),
            "affine",
        )
        .unwrap();
        let parts =
            mlx::ops::shape::split_at(&projected, &[n], projected.ndim() as i32 - 1).unwrap();
        let expected = gelu_approx_mul_on(&parts[0], &parts[1], ()).unwrap();

        let params = QuantizedGateUpGeGluDecode {
            weight,
            scales,
            biases,
            intermediate_size: n,
            group_size,
            bits,
        };
        let got = quantized_gate_up_geglu_decode_on(&x, params, ())
            .unwrap()
            .expect("shape should use fused qmv+geglu");

        assert_eq!(got.shape().as_slice(), expected.shape().as_slice());
        assert_all_close(&got, &expected, 0.02);
    }

    #[test]
    #[serial(mlx_metal)]
    fn rms_norm_default_rope_decode_matches_composed_non_traditional() {
        assert_rms_norm_default_rope_decode(false);
    }

    #[test]
    #[serial(mlx_metal)]
    fn rms_norm_default_rope_decode_matches_composed_traditional() {
        assert_rms_norm_default_rope_decode(true);
    }

    fn assert_rms_norm_default_rope_decode(traditional: bool) {
        let heads = 2_i32;
        let head_dim = 8_i32;
        let offset = 17_i32;
        let x_data: Vec<f32> = (0..(heads * head_dim))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
            .collect();
        let x: Array = (x_data.as_slice(), (1_i32, 1_i32, heads, head_dim))
            .try_into()
            .unwrap();
        let weight_data: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.03).collect();
        let weight: Array = (weight_data.as_slice(), &[head_dim][..])
            .try_into()
            .unwrap();
        let offsets: Array = (&[offset][..], &[1_i32][..]).try_into().unwrap();

        let expected = mlx::fast::rms_norm_on(&x, Some(&weight), 1.0e-6, ())
            .unwrap()
            .transpose_axes(&[0_i32, 2, 1, 3][..])
            .unwrap();
        let expected = mlx::fast::rope(
            &expected,
            head_dim,
            traditional,
            Some(10_000.0),
            1.0,
            offset,
            None,
        )
        .unwrap();

        let params = RmsNormDefaultRopeDecode {
            weight: &weight,
            offsets: &offsets,
            eps: 1.0e-6,
            base: 10_000.0,
            traditional,
            heads,
            head_dim,
        };
        let got = rms_norm_default_rope_decode_on(&x, params, ())
            .unwrap()
            .expect("shape should use fused decode RMSNorm+RoPE");

        assert_eq!(got.shape().as_slice(), expected.shape().as_slice());
        assert_all_close(&got, &expected, 0.001);
    }

    #[test]
    #[serial(mlx_metal)]
    fn geglu_fused_rejects_shape_mismatch() {
        let gate = Array::zeros((2, 3), Dtype::Float32).unwrap();
        let up = Array::zeros((3, 2), Dtype::Float32).unwrap();
        let err = gelu_approx_mul_on(&gate, &up, ()).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }
}
