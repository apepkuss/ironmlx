//! Kernels of the Dense prefill route.
use crate::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

/// Prefill GEMM over the M5 tiled weight layout (`m5_affine4::prepare`:
/// 32-column tiles, each 64-input group of a tile's columns contiguous, and
/// `(scale, bias)` pairs `[K/64, N]`), so a single resident tiled copy serves
/// prefill as well as decode. A threadgroup of four simdgroups owns a
/// `128 x 64` output tile; row tiles run fastest in the grid, so neighbouring
/// threadgroups read the same weight columns. Per 64-input step every thread
/// dequantizes 32 codes of one column into a shared bf16 stage (double
/// buffered) with MLX's values, each simdgroup multiplies its 32 rows on the
/// tensor unit with fp32 accumulation, and the next step's codes are read right
/// after the barrier. A half stage is ~4% faster but the bf16 x f16 matmul
/// loses accuracy (relative error against an f32 reference 2.2e-3, against
/// 1.66e-3 for MLX's dequantize + matmul and for this kernel), even with the
/// half values computed exactly.
///
/// `CHECK = 0` is the full-tile kernel: no bounds anywhere. On M5 Max any
/// conditional in its store makes the whole kernel ~15% slower (2048 x 5120 ->
/// 34816: 14.0 against 12.0 ms), so partial tiles take `CHECK = 1`: dynamic
/// activation extents (the tensor unit masks rows past M) and a branch-free
/// store whose out-of-range elements go to a spare output element.
///
/// `dims`: rows M, weight columns NW (the coefficient row stride), K (the
/// activation width), the first weight column NB of this output, its column
/// count NO, and the weight's input count KW (K < KW multiplies only the
/// weight's leading K inputs). `NTW` is the
/// layout's tile width (32, or 64 with `M5_QMM_COOP`).
///
/// `SWIGLU = 1` (a fused gate / up weight, gate columns first): a tile's stage
/// holds 32 gate columns and the 32 matching up columns, and the output is
/// MLX's compiled SwiGLU of the two rounded BF16 products (the same ops on
/// `bfloat16_t`, bit for bit) over NO = NW / 2 columns; the products round
/// through threadgroup memory so each lane pairs a gate with its up value.
const PREFILL_TILED_SOURCE: &str = r#"
  constexpr int RS = 32, NSG = 4, BM = RS * NSG, BN = 64, BK = 64;
  const int M = dims[0], NW = dims[1], K = dims[2], NB = dims[3], NO = dims[4], KW = dims[5];
  const int KG = KW / 64, S = K / BK;
  const int m0 = threadgroup_position_in_grid.x * BM;
  const int c0 = threadgroup_position_in_grid.y * BN;
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
  // Thread pairs share a column: half h of its 64 codes per step.
  const int c = int(tid) >> 1, h = int(tid) & 1;
#if SWIGLU
  const int g0 = threadgroup_position_in_grid.y * 32;
  const int n = NB + (c < 32 ? g0 + c : NW / 2 + g0 + c - 32);
#if CHECK
  const bool live = g0 + (c & 31) < NO;
#else
  constexpr bool live = true;
#endif
#else
#if CHECK
  const bool live = c0 + c < NO;
  const int n = NB + c0 + c;
#else
  constexpr bool live = true;
#if CLAMP
  // Columns past NW (a 64-padded output) recompute the last column.
  const int n = min(NB + c0 + c, NW - 1);
#else
  const int n = NB + c0 + c;
#endif
#endif
#endif
  threadgroup bfloat stage[2][BN * BK];
  // Column n's 32 codes of half h in group g: tile n / NTW, row n % NTW.
  const device uint4* wbase = (const device uint4*)W + (int64_t(n / NTW) * KG * NTW + (n % NTW)) * 2 + h;
  const device vec<bfloat, 2>* sbv = (const device vec<bfloat, 2>*)SB + n;
  auto fetch = [&](int g, thread uint4& q, thread float2& sbf) {
    if (live) {
      q = wbase[int64_t(g) * NTW * 2];
      const vec<bfloat, 2> v = sbv[int64_t(g) * NW];
      sbf = float2(float(v[0]), float(v[1]));
    } else {
      q = uint4(0);
      sbf = float2(0.0f);
    }
  };
  // MLX's dequantized values: scale * code + bias in f32, rounded once to bf16.
  auto dequant = [&](uint4 q, float2 sbf, threadgroup bfloat* dst) {
    for (int w = 0; w < 4; w++) {
      const uint word = q[w];
      vec<bfloat, 8> out;
      for (int e = 0; e < 8; e++) out[e] = bfloat(fma(float((word >> (4 * e)) & 0xFu), sbf.x, sbf.y));
      *(threadgroup vec<bfloat, 8>*)(dst + c * BK + (h * 4 + w) * 8) = out;
    }
  };
  constexpr auto desc = matmul2d_descriptor(RS, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  const int mr = m0 + int(sg) * RS;
#if CHECK
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(K, M));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tB0(stage[0], dextents<int32_t, 2>(BK, BN));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tB1(stage[1], dextents<int32_t, 2>(BK, BN));
  auto P = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
#else
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + int64_t(mr) * K, dextents<int32_t, 2>(K, RS));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> sB0(stage[0], dextents<int32_t, 2>(BK, BN));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> sB1(stage[1], dextents<int32_t, 2>(BK, BN));
  auto tB0 = sB0.template slice<BK, BN>(0, 0);
  auto tB1 = sB1.template slice<BK, BN>(0, 0);
  auto a0 = tA.template slice<BK, RS>(0, 0);
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB0), float>();
#endif
  for (uint16_t i = 0; i < P.get_capacity(); ++i) P[i] = 0.0f;
  uint4 q;
  float2 sbf;
  fetch(0, q, sbf);
  for (int st = 0; st < S; st++) {
    dequant(q, sbf, stage[st & 1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (st + 1 < S) fetch(st + 1, q, sbf);
#if CHECK
    auto a = tA.slice(st * BK, mr);
#else
    auto a = tA.template slice<BK, RS>(st * BK, 0);
#endif
    if (st & 1) op.run(a, tB1, P); else op.run(a, tB0, P);
  }
#if SWIGLU
  // Every simdgroup has read its last stage; each takes a 32 x 64 quarter.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup bfloat16_t* prod = (threadgroup bfloat16_t*)&stage[0][0] + int(sg) * (RS * BN);
  for (uint16_t i = 0; i < P.get_capacity(); ++i) {
    if (!P.is_valid_element(i)) continue;
    const auto idx = P.get_multidimensional_index(i);
    prod[int(idx[1]) * BN + int(idx[0])] = bfloat16_t(P[i]);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int l = int(thread_index_in_simdgroup);
  for (int r = 0; r < RS; r++) {
    const bfloat16_t gate = prod[r * BN + l], up = prod[r * BN + 32 + l];
    const bfloat16_t sig = IronSigmoid()(gate);
    const bfloat16_t silu = gate * sig;
    const bfloat16_t out = silu * up;
    const int m = mr + r;
#if CHECK
    device bfloat* dst = (m < M && g0 + l < NO) ? Y + (int64_t(m) * NO + g0 + l) : Spare;
    *dst = out;
#else
    Y[int64_t(m) * NO + g0 + l] = out;
#endif
  }
#else
  for (uint16_t i = 0; i < P.get_capacity(); ++i) {
    if (!P.is_valid_element(i)) continue;
    const auto idx = P.get_multidimensional_index(i);
    const int cc = c0 + int(idx[0]), m = mr + int(idx[1]);
#if CHECK
    device bfloat* dst = (m < M && cc < NO) ? Y + (int64_t(m) * NO + cc) : Spare;
    *dst = bfloat(P[i]);
#else
    Y[int64_t(m) * NO + cc] = bfloat(P[i]);
#endif
  }
#endif
"#;

/// MLX's `Sigmoid` unary op (`unary_ops.h`), for the SwiGLU epilogue.
const IRON_SIGMOID: &str = "struct IronSigmoid {\n\
  template <typename T>\n\
  T operator()(T x) thread {\n\
    auto y = 1 / (1 + metal::exp(metal::abs(x)));\n\
    return (x < 0) ? y : 1 - y;\n\
  }\n\
};\n";

/// `x [.., K] @ W^T` for a prepared (tiled) affine4 weight of `NW` columns,
/// with [`PREFILL_TILED_SOURCE`]. Whole 128-row blocks take the full-tile
/// kernel, into an output padded to a multiple of 64 columns when `NW` is not
/// one (the result is a view of its first `NW`); other row counts take the
/// checked kernel.
pub(crate) fn prefill_tiled_gemm(
    x: &Array,
    prepared: &super::m5_affine4::Prepared,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap_or(&0);
    let nw = prepared.weight.shape().as_slice()[0];
    anyhow::ensure!(
        k > 0
            && k % 64 == 0
            && prepared.weight.shape().as_slice()[1] * 8 == k
            && nw % 32 == 0
            && x.dtype() == Dtype::Bfloat16,
        "tiled prefill gemm of {:?} -> {nw}",
        shape.as_slice()
    );
    let m = i32::try_from(x.size() / k as usize)?;
    let width = if m % 128 == 0 {
        (nw + 63) / 64 * 64
    } else {
        nw
    };
    let y = tiled_dispatch(x, prepared, m, k, 0, width, m % 128 != 0, false, target)?;
    let mut out_shape = shape.as_slice().to_vec();
    *out_shape.last_mut().unwrap() = width;
    let y = mlx::ops::shape::reshape_on(&y, out_shape.as_slice(), target)?;
    if width == nw {
        return Ok(y);
    }
    let lo = vec![0; out_shape.len()];
    *out_shape.last_mut().unwrap() = nw;
    Ok(mlx::ops::indexing::slice_on(
        &y,
        &lo[..],
        out_shape.as_slice(),
        target,
    )?)
}

/// `x [.., K]` through a fused gate / up weight (`NW` columns, gate first) and
/// MLX's SwiGLU, `[.., NW / 2]`, with [`PREFILL_TILED_SOURCE`]'s `SWIGLU`
/// variant: bit for bit `swiglu(split(prefill_tiled_gemm(x)))` without the
/// products' round trip through memory.
pub(crate) fn prefill_tiled_swiglu(
    x: &Array,
    prepared: &super::m5_affine4::Prepared,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap_or(&0);
    let nw = prepared.weight.shape().as_slice()[0];
    anyhow::ensure!(
        k > 0
            && k % 64 == 0
            && prepared.weight.shape().as_slice()[1] * 8 == k
            && nw % 64 == 0
            && x.dtype() == Dtype::Bfloat16,
        "tiled prefill swiglu of {:?} -> {nw}",
        shape.as_slice()
    );
    let m = i32::try_from(x.size() / k as usize)?;
    let y = tiled_dispatch(x, prepared, m, k, 0, nw / 2, m % 128 != 0, true, target)?;
    let mut out_shape = shape.as_slice().to_vec();
    *out_shape.last_mut().unwrap() = nw / 2;
    Ok(mlx::ops::shape::reshape_on(
        &y,
        out_shape.as_slice(),
        target,
    )?)
}

/// [`prefill_tiled_swiglu`] over the leading `cols` gate / up column pairs
/// only (`[.., cols]`): the GPU's share of an MLP split by intermediate
/// channel.
pub(crate) fn prefill_tiled_swiglu_cols(
    x: &Array,
    prepared: &super::m5_affine4::Prepared,
    cols: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap_or(&0);
    let nw = prepared.weight.shape().as_slice()[0];
    anyhow::ensure!(
        k > 0
            && k % 64 == 0
            && prepared.weight.shape().as_slice()[1] * 8 == k
            && nw % 64 == 0
            && cols > 0
            && cols % 32 == 0
            && cols <= nw / 2
            && x.dtype() == Dtype::Bfloat16,
        "tiled prefill swiglu of {:?} -> {cols} of {nw}",
        shape.as_slice()
    );
    let m = i32::try_from(x.size() / k as usize)?;
    let y = tiled_dispatch(x, prepared, m, k, 0, cols, m % 128 != 0, true, target)?;
    let mut out_shape = shape.as_slice().to_vec();
    *out_shape.last_mut().unwrap() = cols;
    Ok(mlx::ops::shape::reshape_on(
        &y,
        out_shape.as_slice(),
        target,
    )?)
}

/// `x [.., K'] @ W[:, ..K']^T` for a prepared weight of K >= K' inputs: the
/// weight's leading K' inputs only (the GPU's channels of a split MLP down
/// projection). Same per-element arithmetic as [`prefill_tiled_gemm`] over
/// those inputs.
pub(crate) fn prefill_tiled_gemm_leading(
    x: &Array,
    prepared: &super::m5_affine4::Prepared,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap_or(&0);
    let nw = prepared.weight.shape().as_slice()[0];
    let kw = prepared.weight.shape().as_slice()[1] * 8;
    anyhow::ensure!(
        k > 0 && k % 64 == 0 && k <= kw && nw % 64 == 0 && x.dtype() == Dtype::Bfloat16,
        "tiled prefill gemm of {:?} over {kw} inputs -> {nw}",
        shape.as_slice()
    );
    let m = i32::try_from(x.size() / k as usize)?;
    let y = tiled_dispatch(x, prepared, m, k, 0, nw, m % 128 != 0, false, target)?;
    let mut out_shape = shape.as_slice().to_vec();
    *out_shape.last_mut().unwrap() = nw;
    Ok(mlx::ops::shape::reshape_on(
        &y,
        out_shape.as_slice(),
        target,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn tiled_dispatch(
    x: &Array,
    prepared: &super::m5_affine4::Prepared,
    m: i32,
    k: i32,
    first: i32,
    count: i32,
    check: bool,
    swiglu: bool,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNELS: [OnceLock<MetalKernel>; 16] = [const { OnceLock::new() }; 16];
    let nw = prepared.weight.shape().as_slice()[0];
    let ntw = super::m5_affine4::tile_width(nw);
    // The full-tile kernel clamps its columns only for a padded output: any
    // extra arithmetic there slows it (2048 x 5120 -> 14336: 4.90 -> 5.27 ms).
    let clamp = !check && !swiglu && first + count > nw;
    let slot = &KERNELS[usize::from(check)
        + if ntw == 64 { 2 } else { 0 }
        + if swiglu { 4 } else { 0 }
        + if clamp { 8 } else { 0 }];
    if slot.get().is_none() {
        let built = MetalKernel::builder(format!(
            "ironmlx_dense_prefill_tiled_gemm_v1_{}_{ntw}{}{}",
            u8::from(check),
            if swiglu { "_swiglu" } else { "" },
            if clamp { "_clamp" } else { "" }
        ))
        .inputs(&["X", "W", "SB", "dims"])
        .outputs(&["Y", "Spare"])
        .header(format!(
            "#define CHECK {}\n#define SWIGLU {}\n#define CLAMP {}\n#define NTW {ntw}\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n{IRON_SIGMOID}",
            u8::from(check),
            u8::from(swiglu),
            u8::from(clamp)
        ))
        .source(PREFILL_TILED_SOURCE)
        .build()?;
        let _ = slot.set(built);
    }
    let kw = prepared.weight.shape().as_slice()[1] * 8;
    let dims: Array = (&[m, nw, k, first, count, kw][..], &[6][..]).try_into()?;
    let mut outs = slot
        .get()
        .unwrap()
        .dispatch_builder()
        .inputs(&[x, &prepared.weight, &prepared.sb, &dims])
        .output_shapes(&[Shape::from(&[m, count][..]), Shape::from(&[1][..])])
        .output_dtypes(&[Dtype::Bfloat16, Dtype::Bfloat16])
        .grid(
            (m + 127) / 128 * 128,
            if swiglu {
                (count + 31) / 32
            } else {
                (count + 63) / 64
            },
            1,
        )
        .threadgroup(128, 1, 1)
        .stream(target)
        .dispatch()?;
    Ok(outs.take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::cast::astype;
    use serial_test::serial;

    /// The tiled prefill GEMM against an fp32 reference (dequantized in f32,
    /// f32 matmul), next to MLX's bf16 dequantize + matmul route, on the Dense
    /// prefill shapes at full, partial and small row counts; plus chained
    /// timing of both routes. Env `PREFILL_TILED_ROWS` (default `2048,1999,300`).
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn prefill_tiled_gemm_accuracy() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !crate::nn::m5_affine4::tensor_unit_available() {
            return Ok(());
        }
        let target = StreamOrDevice::default();
        let rows: Vec<i32> = std::env::var("PREFILL_TILED_ROWS")
            .unwrap_or_else(|_| "2048,1999,300".into())
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let rel = |a: &Array, b: &Array| -> Result<f32> {
            let d = &astype(a, Dtype::Float32)? - b;
            let num = mlx::ops::reduction::sum(&(&d * &d), mlx::ops::reduction::All, false)?
                .item::<f32>()?;
            let den = mlx::ops::reduction::sum(&(b * b), mlx::ops::reduction::All, false)?
                .item::<f32>()?;
            Ok((num / den).sqrt())
        };
        let time = |f: &dyn Fn() -> Result<Array>| -> Result<f64> {
            let mut v = Vec::new();
            for it in 0..7 {
                let t0 = std::time::Instant::now();
                let outs = (0..8).map(|_| f()).collect::<Result<Vec<_>>>()?;
                mlx::transforms::eval(&outs.iter().collect::<Vec<_>>())?;
                if it > 0 {
                    v.push(t0.elapsed().as_secs_f64() * 1e3 / 8.0);
                }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(v[v.len() / 2])
        };
        for (n, k) in [
            (34816, 5120),
            (5120, 17408),
            (16480, 5120),
            (14336, 5120),
            (5120, 6144),
        ] {
            let w = mlx::random::normal()
                .shape(&[n, k][..])
                .scale(0.02)
                .key(&mlx::random::key(11)?)
                .sample()?;
            let q = mlx::quantization::quantize(
                &astype(&w, Dtype::Bfloat16)?,
                Some(64),
                Some(4),
                "affine",
                None,
            )?;
            let parts = ironmlx_core::nn::linear::QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: crate::core::QuantMode::Affine,
            };
            let prepared = crate::nn::m5_affine4::prepare(&parts, target)?;
            let w32 = mlx::quantization::dequantize_on(
                &q[0],
                &q[1],
                Some(&q[2]),
                Some(64),
                Some(4),
                "affine",
                None,
                Some(Dtype::Float32),
                target,
            )?;
            let wbf = mlx::quantization::dequantize_on(
                &q[0],
                &q[1],
                Some(&q[2]),
                Some(64),
                Some(4),
                "affine",
                None,
                Some(Dtype::Bfloat16),
                target,
            )?;
            mlx::transforms::eval(&[&prepared.weight, &prepared.sb, &w32, &wbf])?;
            for &m in &rows {
                let x = astype(
                    &mlx::random::normal()
                        .shape(&[m, k][..])
                        .scale(0.5)
                        .key(&mlx::random::key(12)?)
                        .sample()?,
                    Dtype::Bfloat16,
                )?;
                let reference = mlx::ops::matmul::matmul(
                    &astype(&x, Dtype::Float32)?,
                    &mlx::ops::shape::transpose(&w32)?,
                )?;
                let mlx_route = mlx::ops::matmul::matmul(&x, &mlx::ops::shape::transpose(&wbf)?)?;
                let tiled = prefill_tiled_gemm(&x, &prepared, target)?;
                anyhow::ensure!(
                    tiled.shape().as_slice() == [m, n],
                    "tiled output shape {:?}",
                    tiled.shape().as_slice()
                );
                let (e_mlx, e_tiled) = (rel(&mlx_route, &reference)?, rel(&tiled, &reference)?);
                let t_mlx = time(&|| {
                    Ok(mlx::ops::matmul::matmul(
                        &x,
                        &mlx::ops::shape::transpose(&mlx::quantization::dequantize_on(
                            &q[0],
                            &q[1],
                            Some(&q[2]),
                            Some(64),
                            Some(4),
                            "affine",
                            None,
                            Some(Dtype::Bfloat16),
                            target,
                        )?)?,
                    )?)
                })?;
                let t_tiled = time(&|| prefill_tiled_gemm(&x, &prepared, target))?;
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                println!(
                    "TILED m={m} n={n} k={k}: rel vs f32 mlx {e_mlx:.3e} tiled {e_tiled:.3e} | mlx deq+mm {t_mlx:.2} ms ({:.0}T) tiled {t_tiled:.2} ms ({:.0}T)",
                    flops / t_mlx / 1e9,
                    flops / t_tiled / 1e9
                );
                // Same dequantized values as MLX: only the summation order differs.
                anyhow::ensure!(
                    e_tiled.is_finite() && e_tiled <= e_mlx * 1.01,
                    "tiled error {e_tiled} above the MLX route's {e_mlx}"
                );
            }
        }
        Ok(())
    }

    /// Needs a GPU: the fused SwiGLU route equals MLX's compiled SwiGLU of the
    /// split tiled GEMM bit for bit (full and checked kernels; every finite
    /// BF16 gate value through a crafted weight), and route times.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn prefill_tiled_swiglu_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !crate::nn::m5_affine4::tensor_unit_available() {
            return Ok(());
        }
        let target = StreamOrDevice::default();
        let swiglu = ironmlx_core::nn::activations::build_swiglu();
        let unfused = |x: &Array, prepared: &crate::nn::m5_affine4::Prepared| -> Result<Array> {
            let y = prefill_tiled_gemm(x, prepared, target)?;
            let parts = mlx::ops::shape::split_n_on(&y, 2, -1, target)?;
            ironmlx_core::nn::activations::invoke_swiglu(&swiglu, &parts[0], &parts[1])
        };
        let mismatches = |a: &Array, b: &Array| -> Result<u32> {
            // Exact through f32; NaN never equals itself, so NaNs count.
            let (a, b) = (astype(a, Dtype::Float32)?, astype(b, Dtype::Float32)?);
            Ok(mlx::ops::reduction::sum(
                &astype(&a.not_equal(&b)?, Dtype::Uint32)?,
                mlx::ops::reduction::All,
                false,
            )?
            .item::<u32>()?)
        };
        let time = |f: &dyn Fn() -> Result<Array>| -> Result<f64> {
            let mut v = Vec::new();
            for it in 0..7 {
                let t0 = std::time::Instant::now();
                let outs = (0..8).map(|_| f()).collect::<Result<Vec<_>>>()?;
                mlx::transforms::eval(&outs.iter().collect::<Vec<_>>())?;
                if it > 0 {
                    v.push(t0.elapsed().as_secs_f64() * 1e3 / 8.0);
                }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(v[v.len() / 2])
        };
        let prepare = |q: &[Array]| -> Result<crate::nn::m5_affine4::Prepared> {
            let parts = ironmlx_core::nn::linear::QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: crate::core::QuantMode::Affine,
            };
            let prepared = crate::nn::m5_affine4::prepare(&parts, target)?;
            mlx::transforms::eval(&[&prepared.weight, &prepared.sb])?;
            Ok(prepared)
        };
        let (n, k) = (34816, 5120);
        let w = mlx::random::normal()
            .shape(&[n, k][..])
            .scale(0.02)
            .key(&mlx::random::key(11)?)
            .sample()?;
        let q = mlx::quantization::quantize(
            &astype(&w, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        let prepared = prepare(&q)?;
        for m in [2048, 1300, 129] {
            let x = astype(
                &mlx::random::normal()
                    .shape(&[m, k][..])
                    .scale(0.5)
                    .key(&mlx::random::key(12)?)
                    .sample()?,
                Dtype::Bfloat16,
            )?;
            let want = unfused(&x, &prepared)?;
            let got = prefill_tiled_swiglu(&x, &prepared, target)?;
            anyhow::ensure!(
                got.shape().as_slice() == [m, n / 2],
                "fused output shape {:?}",
                got.shape().as_slice()
            );
            let bad = mismatches(&got, &want)?;
            let t_unfused = time(&|| unfused(&x, &prepared))?;
            let t_fused = time(&|| prefill_tiled_swiglu(&x, &prepared, target))?;
            println!("SWIGLU m={m}: mismatches {bad} | gemm+swiglu {t_unfused:.2} ms fused {t_fused:.2} ms");
            anyhow::ensure!(bad == 0, "{bad} fused SwiGLU outputs differ at {m} rows");
        }
        // Gates over many magnitudes (saturated sigmoid, tiny and huge
        // products): per-group weight scales spread as exp(3 N(0, 1)).
        let m = 2048;
        let spread = mlx::ops::unary::exp(
            &mlx::random::normal()
                .shape(&[n, k / 64][..])
                .scale(3.0)
                .key(&mlx::random::key(21)?)
                .sample()?,
        )?;
        let wq = &mlx::random::normal()
            .shape(&[n, k][..])
            .scale(0.02)
            .key(&mlx::random::key(23)?)
            .sample()?
            * &mlx::ops::shape::repeat(&spread, 64, 1)?;
        let q = mlx::quantization::quantize(
            &astype(&wq, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        let prepared = prepare(&q)?;
        let x = astype(
            &mlx::random::normal()
                .shape(&[m, k][..])
                .scale(3.0)
                .key(&mlx::random::key(24)?)
                .sample()?,
            Dtype::Bfloat16,
        )?;
        let want = unfused(&x, &prepared)?;
        let got = prefill_tiled_swiglu(&x, &prepared, target)?;
        let bad = mismatches(&got, &want)?;
        let gate = mlx::ops::shape::split_n_on(
            &prefill_tiled_gemm(&x, &prepared, target)?,
            2,
            -1,
            target,
        )?
        .remove(0);
        let absg = mlx::ops::unary::abs(&astype(&gate, Dtype::Float32)?)?;
        println!(
            "SWIGLU wide gates: mismatches {bad}, |gate| max {:.3e}",
            mlx::ops::reduction::max(&absg, mlx::ops::reduction::All, false)?.item::<f32>()?
        );
        anyhow::ensure!(bad == 0, "{bad} fused SwiGLU outputs differ on wide gates");
        Ok(())
    }

    /// Diagnostic: prefill attention at the Dense shapes (24 / 4 heads, D 256,
    /// 2048 causal query rows ending at the last key) through
    /// `scaled_dot_product_attention` (the D256 NAX library when configured),
    /// time and max error of head 0 against an f32 reference. Env
    /// `PREFILL_ATTN_KEYS` (default "8192,32768,65536,131072").
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn prefill_nax_attention_throughput() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let rows = 2048;
        let keys: Vec<i32> = std::env::var("PREFILL_ATTN_KEYS")
            .unwrap_or_else(|_| "8192,32768,65536,131072".into())
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        for keys in keys {
            let normal = |shape: &[i32], seed: u64| -> Result<Array> {
                let a = mlx::random::normal()
                    .shape(shape)
                    .scale(1.0)
                    .key(&mlx::random::key(seed)?)
                    .sample()?;
                Ok(astype(&a, Dtype::Bfloat16)?)
            };
            let q = normal(&[1, 24, rows, 256], 1)?;
            let k = normal(&[1, 4, keys, 256], 2)?;
            let v = normal(&[1, 4, keys, 256], 3)?;
            mlx::transforms::eval(&[&q, &k, &v])?;
            let pairs =
                rows as f64 * (keys - rows) as f64 + rows as f64 * (rows as f64 + 1.0) / 2.0;
            let flops = 4.0 * 24.0 * 256.0 * pairs;
            let mut runs = Vec::new();
            let mut out = None;
            for _ in 0..6 {
                let t0 = std::time::Instant::now();
                let o = mlx::fast::scaled_dot_product_attention(
                    &q, &k, &v, 0.0625, "causal", None, None,
                )?;
                mlx::transforms::eval(&[&o])?;
                runs.push(t0.elapsed().as_secs_f64());
                out = Some(o);
            }
            runs.sort_by(f64::total_cmp);
            let t = runs[runs.len() / 2];
            let head = |a: &Array, n: i32| -> Result<Array> {
                Ok(astype(
                    &mlx::ops::indexing::slice(a, &[0, 0, 0, 0][..], &[1, 1, n, 256][..])?,
                    Dtype::Float32,
                )?)
            };
            let reference = mlx::fast::scaled_dot_product_attention(
                &head(&q, rows)?,
                &head(&k, keys)?,
                &head(&v, keys)?,
                0.0625,
                "causal",
                None,
                None,
            )?;
            let d = &head(out.as_ref().unwrap(), rows)? - &reference;
            let err = mlx::ops::reduction::max(
                &mlx::ops::unary::abs(&d)?,
                mlx::ops::reduction::All,
                false,
            )?
            .item::<f32>()?;
            let rms = (mlx::ops::reduction::mean(&(&d * &d), mlx::ops::reduction::All, false)?
                .item::<f32>()?)
            .sqrt();
            println!(
                "ATTN keys={keys}: {:.2} ms {:.1} TF max err {err:.3e} rms {rms:.3e}",
                t * 1e3,
                flops / t / 1e12
            );
        }
        Ok(())
    }

    /// Needs a GPU: the column-limited SwiGLU equals the full route's leading
    /// columns bit for bit, and the leading-input GEMM's error against an f32
    /// reference matches MLX's dequantize + matmul over the same inputs.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn prefill_tiled_split_routes() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !crate::nn::m5_affine4::tensor_unit_available() {
            return Ok(());
        }
        let target = StreamOrDevice::default();
        let prepare =
            |n: i32, k: i32, seed: u64| -> Result<(crate::nn::m5_affine4::Prepared, Vec<Array>)> {
                let w = mlx::random::normal()
                    .shape(&[n, k][..])
                    .scale(0.02)
                    .key(&mlx::random::key(seed)?)
                    .sample()?;
                let q = mlx::quantization::quantize(
                    &astype(&w, Dtype::Bfloat16)?,
                    Some(64),
                    Some(4),
                    "affine",
                    None,
                )?;
                let parts = ironmlx_core::nn::linear::QuantizedLinearParts {
                    weight: &q[0],
                    scales: &q[1],
                    biases: Some(&q[2]),
                    bias: None,
                    group_size: 64,
                    bits: 4,
                    mode: crate::core::QuantMode::Affine,
                };
                let prepared = crate::nn::m5_affine4::prepare(&parts, target)?;
                mlx::transforms::eval(&[&prepared.weight, &prepared.sb])?;
                Ok((prepared, q))
            };
        let (gu, _) = prepare(34816, 5120, 41)?;
        let (dn, dq) = prepare(5120, 17408, 42)?;
        for (m, g) in [(2048, 12800), (1300, 12800), (2048, 9728)] {
            let x = astype(
                &mlx::random::normal()
                    .shape(&[m, 5120][..])
                    .scale(0.5)
                    .key(&mlx::random::key(43)?)
                    .sample()?,
                Dtype::Bfloat16,
            )?;
            let full = prefill_tiled_swiglu(&x, &gu, target)?;
            let lead = mlx::ops::indexing::slice(&full, &[0, 0][..], &[m, g][..])?;
            let cols = prefill_tiled_swiglu_cols(&x, &gu, g, target)?;
            let bad = mlx::ops::reduction::sum(
                &astype(&cols.not_equal(&lead)?, Dtype::Int32)?,
                mlx::ops::reduction::All,
                false,
            )?
            .item::<i32>()?;
            anyhow::ensure!(
                bad == 0,
                "{bad} column-limited SwiGLU outputs differ at {m} rows / {g}"
            );
            let a = astype(
                &mlx::random::normal()
                    .shape(&[m, g][..])
                    .scale(0.5)
                    .key(&mlx::random::key(44)?)
                    .sample()?,
                Dtype::Bfloat16,
            )?;
            let wl = |dtype| -> Result<Array> {
                let w = mlx::quantization::dequantize_on(
                    &dq[0],
                    &dq[1],
                    Some(&dq[2]),
                    Some(64),
                    Some(4),
                    "affine",
                    None,
                    Some(dtype),
                    target,
                )?;
                Ok(mlx::ops::indexing::slice(&w, &[0, 0][..], &[5120, g][..])?)
            };
            let reference = mlx::ops::matmul::matmul(
                &astype(&a, Dtype::Float32)?,
                &mlx::ops::shape::transpose(&wl(Dtype::Float32)?)?,
            )?;
            let mlx_route =
                mlx::ops::matmul::matmul(&a, &mlx::ops::shape::transpose(&wl(Dtype::Bfloat16)?)?)?;
            let tiled = prefill_tiled_gemm_leading(&a, &dn, target)?;
            let rel = |y: &Array| -> Result<f32> {
                let d = &astype(y, Dtype::Float32)? - &reference;
                let num = mlx::ops::reduction::sum(&(&d * &d), mlx::ops::reduction::All, false)?
                    .item::<f32>()?;
                let den = mlx::ops::reduction::sum(
                    &(&reference * &reference),
                    mlx::ops::reduction::All,
                    false,
                )?
                .item::<f32>()?;
                Ok((num / den).sqrt())
            };
            let (e_mlx, e_tiled) = (rel(&mlx_route)?, rel(&tiled)?);
            println!("SPLIT m={m} g={g}: swiglu cols bitwise, leading gemm rel {e_tiled:.3e} (mlx {e_mlx:.3e})");
            anyhow::ensure!(
                e_tiled.is_finite() && e_tiled <= e_mlx * 1.01,
                "leading gemm error {e_tiled} above MLX's {e_mlx}"
            );
        }
        Ok(())
    }
}
