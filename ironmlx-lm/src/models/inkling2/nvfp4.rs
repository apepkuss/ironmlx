//! B1 routed NVFP4 matrix-vector products on Apple GPUs.
use anyhow::Result;
use mlx::{Array, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

pub(super) fn decode(
    x: &Array,
    weight: &Array,
    scales: &Array,
    global: &Array,
    indices: &Array,
    rows_per_simd: i32,
    t: StreamOrDevice,
) -> Result<Array> {
    decode_activated(x, weight, scales, global, indices, rows_per_simd, None, t)
}

/// [`decode`], optionally followed by the routed gate/up activation of
/// interleaved output pairs (the `ActivationTables` lookups), written at half
/// the width. Each pair rounds to BF16 first, as the separate kernel reads it.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_activated(
    x: &Array,
    weight: &Array,
    scales: &Array,
    global: &Array,
    indices: &Array,
    rows_per_simd: i32,
    activation: Option<(&Array, &Array)>,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = super::metal::kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_nvfp4_decode")
            .inputs(&["x", "w", "scales", "global_scale", "indices", "gate", "up"])
            .outputs(&["out"])
            .header(r#"
                struct InklingFp4Pairs {
                    half2 values[256];
                };
                constexpr InklingFp4Pairs make_fp4_pairs() {
                    InklingFp4Pairs table{};
                    constexpr half levels[16] = {0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
                    for (int i = 0; i < 256; ++i) {
                        table.values[i] = half2(levels[i & 15], levels[i >> 4]);
                    }
                    return table;
                }
                constant InklingFp4Pairs inkling_e2m1 = make_fp4_pairs();
            "#)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint lane = thread_index_in_simdgroup;
                const uint route = thread_position_in_grid.y;
                const uint row = thread_position_in_grid.x / 32 * TILE;
                const uint packed_row = uint(indices[route]) * N + row;
                const uint input_row = SHARED ? route / ROUTES : route;
                float acc[TILE] = {};
                for (uint pack = lane; pack < K / 8; pack += 32) {
                    const uint start = input_row * K + pack * 8;
                    float values[8];
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 8; ++i) values[i] = float(x[start + i]);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < TILE; ++j) {
                        const uint word = w[(packed_row + j) * (K / 8) + pack];
                        const uchar sb = scales[(packed_row + j) * (K / 16) + pack / 2];
                        const float magnitude = float(as_type<half>(ushort((sb & 127) << 7))) * 256.0f;
                        const float scale = (sb & 128) ? -magnitude : magnitude;
                        float lo = 0, hi = 0;
                        #pragma clang loop unroll(full)
                        for (uint i = 0; i < 2; ++i) {
                            const float2 a = float2(inkling_e2m1.values[(word >> (8*i)) & 255]);
                            const float2 b = float2(inkling_e2m1.values[(word >> (16+8*i)) & 255]);
                            // E2M1 times a BF16 input is exact in FP32 over
                            // the bounded expert activations.
                            if (K == 2304 && TILE == 4) {
                                lo = fma(values[2*i], a.x, lo);
                                lo = fma(values[2*i+1], a.y, lo);
                                hi = fma(values[2*i+4], b.x, hi);
                                hi = fma(values[2*i+5], b.y, hi);
                            } else {
                                lo += values[2*i] * a.x;
                                lo += values[2*i+1] * a.y;
                                hi += values[2*i+4] * b.x;
                                hi += values[2*i+5] * b.y;
                            }
                        }
                        acc[j] += scale * (lo + hi);
                    }
                }
                const float factor = global_scale[indices[route]] * (1.f / 2688.f);
                if (ACT) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < TILE; j += 2) {
                        const T g = static_cast<T>(simd_sum(acc[j]) * factor);
                        const T u = static_cast<T>(simd_sum(acc[j + 1]) * factor);
                        if (lane == 0) {
                            out[route * (N / 2) + (row + j) / 2] =
                                bfloat(gate[as_type<ushort>(bfloat(float(g)))] * up[as_type<ushort>(bfloat(float(u)))]);
                        }
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < TILE; ++j) {
                        const float sum = simd_sum(acc[j]);
                        if (lane == 0) out[route * N + row + j] = static_cast<T>(sum * factor);
                    }
                }
            "#)
            .build()
    })?;
    let n = weight.shape_at(1);
    let routes = indices.shape_at(-1);
    let tokens = indices.shape_at(1);
    // Share inputs across output rows while preserving each row’s FP32 reduction.
    let tile = if n % rows_per_simd == 0 {
        rows_per_simd
    } else {
        1
    };
    let act = activation.is_some() && tile % 2 == 0;
    anyhow::ensure!(
        activation.is_none() || act,
        "the fused activation needs an even number of rows per SIMD group"
    );
    let (gate, up) = activation.unwrap_or((x, x));
    Ok(kernel
        .dispatch_builder()
        .inputs(&[x, weight, scales, global, indices, gate, up])
        .output_shapes(&[Shape::from(
            &[1, tokens, routes, 1, if act { n / 2 } else { n }][..],
        )])
        .output_dtypes(&[x.dtype()])
        .template_bool("ACT", act)
        .template_dtype("T", x.dtype())
        .template_int("N", n)
        .template_int("K", x.shape_at(-1))
        .template_int("ROUTES", routes)
        .template_int("TILE", tile)
        .template_bool("SHARED", x.shape_at(-3) == 1)
        .grid(32 * n / tile, tokens * routes, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{ops::indexing, Dtype};

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn routed_decode_matches_native_gather_with_per_expert_scales() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        let indices: Array = (&[3u32, 0, 1, 3, 2, 1, 0, 2][..], &[1, 1, 8][..]).try_into()?;
        let global: Array = (&[0.5f32, 1.25, 0.9, 3.0][..], &[4][..]).try_into()?;
        for (k, input_rows) in [(2304, 1), (1152, 8)] {
            let n = 2304;
            let key = mlx::random::key(k as u64)?;
            let values = mlx::random::normal()
                .shape(&[4, n, k][..])
                .key(&key)
                .sample()?;
            let packed = mlx::quantization::quantize(&values, Some(16), Some(4), "nvfp4", None)?;
            let x = mlx::random::normal()
                .shape(&[1, 1, input_rows, 1, k][..])
                .key(&mlx::random::split(&key)?.0)
                .sample()?
                .astype(Dtype::Bfloat16)?;
            let y = mlx::quantization::gather_quantized_matmul_on(
                &x.astype(Dtype::Float32)?,
                &packed[0],
                &packed[1],
                None,
                None,
                Some(&indices),
                true,
                Some(16),
                Some(4),
                "nvfp4",
                false,
                t,
            )?;
            let scale = indexing::take_on(&global, &indices, 0, t)?.expand_dims_on([-1, -2], t)?;
            let reference = y
                .try_mul(&super::super::ops::scale(&scale, 1.0 / 2688.0, t)?)?
                .astype(Dtype::Bfloat16)?;
            for rows_per_simd in [1, 2, 4] {
                let got = decode(
                    &x,
                    &packed[0],
                    &packed[1],
                    &global,
                    &indices,
                    rows_per_simd,
                    t,
                )?;
                let mismatches = got
                    .not_equal(&reference)?
                    .astype(Dtype::Int32)?
                    .sum(mlx::ops::All, false)?
                    .item::<i32>()?;
                assert_eq!(mismatches, 0, "K={k}, rows per SIMD={rows_per_simd}");
            }
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn activated_decode_matches_decode_then_activation_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        let indices: Array = (&[3u32, 0, 1, 3, 2, 1, 0, 2][..], &[1, 1, 8][..]).try_into()?;
        let global: Array = (&[0.5f32, 1.25, 0.9, 3.0][..], &[4][..]).try_into()?;
        let key = mlx::random::key(5)?;
        let values = mlx::random::normal()
            .shape(&[4, 2304, 2304][..])
            .key(&key)
            .sample()?;
        let packed = mlx::quantization::quantize(&values, Some(16), Some(4), "nvfp4", None)?;
        let x = mlx::random::normal()
            .shape(&[1, 1, 1, 1, 2304][..])
            .key(&mlx::random::split(&key)?.0)
            .sample()?
            .astype(Dtype::Bfloat16)?;
        let tables = super::super::metal::ActivationTables::new(4.0, 25.0, t)?;
        let plain = decode(&x, &packed[0], &packed[1], &global, &indices, 4, t)?;
        let want = tables.apply_scaled(&plain, None, t)?;
        let got = decode_activated(
            &x,
            &packed[0],
            &packed[1],
            &global,
            &indices,
            4,
            Some(tables.tables()),
            t,
        )?;
        assert_eq!(got.shape(), want.shape());
        let mismatches = got
            .not_equal(&want)?
            .astype(Dtype::Int32)?
            .sum(mlx::ops::All, false)?
            .item::<i32>()?;
        assert_eq!(mismatches, 0);
        Ok(())
    }
}
