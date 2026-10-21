//! GatedDeltaNet input preparation in one dispatch for decode / verify rows:
//! conv-state prepend, depthwise causal conv1d + SiLU, q/k RMS norms with
//! their scales, v, and the gates `g = exp(-exp(A_log) * softplus(a +
//! dt_bias))`, `beta = sigmoid(b)`, all read from the fused input
//! projection's output `[B, S, qkv | z | b | a]`.
//!
//! Each step rounds where the op-by-op path rounds (conv output and SiLU to
//! bf16, normalized q/k to bf16 before the float scale, `a + dt_bias` and
//! softplus to bf16); transcendental functions and the conv accumulation
//! order may differ from MLX's kernels, so results are close, not identical.
//! The q/k norm epsilon is the GatedDeltaNet's fixed 1e-6. M5 profile
//! setting `FUSED_GLUE` (see `nn::fused_norm`).

use std::sync::OnceLock;

use anyhow::ensure;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

/// Outputs of [`prep_on`].
pub(crate) struct Prepared {
    /// `[B, KS - 1 + S, C]` bf16: the conv state followed by this call's qkv.
    pub conv_input: Array,
    /// `[B, S, Hk, 128]` f32, scaled by `1/128` (q) and `1/sqrt(128)` (k).
    pub q: Array,
    pub k: Array,
    /// `[B, S, Hv, 128]` bf16.
    pub v: Array,
    /// `[B, S, Hv]` f32 and bf16.
    pub g: Array,
    pub beta: Array,
}

// MLX's prebuilt-kernel math for bf16 (precise transcendental functions,
// each bf16 op rounding its float result), as in the shim's exact gate chain.
const HEADER: &str = r#"
inline bfloat exp_bf16(bfloat x) { return bfloat(metal::precise::exp(float(x))); }
inline bfloat add_bf16(bfloat x, bfloat y) { return bfloat(float(x) + float(y)); }
inline bfloat sub_bf16(bfloat x, bfloat y) { return bfloat(float(x) - float(y)); }
inline bfloat mul_bf16(bfloat x, bfloat y) { return bfloat(float(x) * float(y)); }
inline bfloat div_bf16(bfloat x, bfloat y) { return bfloat(metal::precise::divide(float(x), float(y))); }
// MLX `Sigmoid`: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y, in bf16 ops.
inline bfloat sigmoid_bf16(bfloat x) {
  const bfloat one = bfloat(1.0f);
  const bfloat y = div_bf16(one, add_bf16(one, exp_bf16(bfloat(metal::precise::fabs(float(x))))));
  return float(x) < 0.0f ? y : sub_bf16(one, y);
}
// The shim's runtime-compiled gate chain, written as it is (bf16 operands,
// the library's default math for `/`).
inline bfloat shim_exp_bf16(bfloat x) {
  return static_cast<bfloat>(metal::precise::exp(static_cast<float>(x)));
}
inline bfloat shim_log1p_bf16(bfloat x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == FLT_MAX) {
    return as_type<bfloat>(ushort(0x7F7F));
  }
  if (xp1 == 1.0f) {
    return x;
  }
  return bfloat(x * (metal::precise::log(xp1) / (xp1 - 1.0f)));
}
inline bfloat shim_logaddexp_bf16(bfloat x, bfloat y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<bfloat>::quiet_NaN();
  }
  constexpr bfloat inf = metal::numeric_limits<bfloat>::infinity();
  bfloat maxval = metal::max(x, y);
  bfloat minval = metal::min(x, y);
  return (minval == -inf || maxval == inf) ? maxval
                                           : (maxval + shim_log1p_bf16(shim_exp_bf16(minval - maxval)));
}
inline bfloat shim_sigmoid_bf16(bfloat x) {
  auto y = 1 / (1 + shim_exp_bf16(static_cast<bfloat>(metal::precise::fabs(static_cast<float>(x)))));
  return (x < 0) ? y : 1 - y;
}
"#;

const SOURCE: &str = r#"
  constexpr int D = 128;
  constexpr int KEEP = KS - 1;
  const uint lane = thread_index_in_simdgroup;
  const int s = int(threadgroup_position_in_grid.x);
  const int hd = int(threadgroup_position_in_grid.y);  // q heads, then k heads, then v heads
  const int b = int(threadgroup_position_in_grid.z);
  const int S = int(proj_shape[1]);
  const device bfloat* st = (const device bfloat*)state + (int64_t)b * KEEP * C;
  const device bfloat* pr = (const device bfloat*)proj + (int64_t)b * S * NP;
  device bfloat* ci = (device bfloat*)conv_input + (int64_t)b * (KEEP + S) * C;
  const int c0 = hd * D + int(lane) * 4;
  float vals[4];
  for (int i = 0; i < 4; i++) {
    const int c = c0 + i;
    // MLX depthwise conv1d: float accumulation over the taps in order.
    float acc = 0.0f;
    for (int j = 0; j < KS; j++) {
      const int r = s + j;  // row of the conv input
      const float x = r < KEEP ? float(st[r * C + c]) : float(pr[(int64_t)(r - KEEP) * NP + c]);
      acc = metal::fma(x, float(((const device bfloat*)w)[c * KS + j]), acc);
    }
    const bfloat conv = bfloat(acc);
    vals[i] = float(mul_bf16(conv, sigmoid_bf16(conv)));
    ci[(int64_t)(KEEP + s) * C + c] = pr[(int64_t)s * NP + c];
    // State rows: row s by row-s threadgroups; rows past the last token by it.
    if (s < KEEP) ci[s * C + c] = st[s * C + c];
    if (s == S - 1)
      for (int r = S; r < KEEP; r++) ci[r * C + c] = st[r * C + c];
  }
  if (hd < 2 * HK) {
    // The shim's exact q/k norm (MLX rms_single_row, then the float scale).
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) acc += vals[i] * vals[i];
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / float(D) + 1e-6f);
    const bool is_q = hd < HK;
    // Host-computed f32 scales (inv_scale^2, inv_scale) as bit patterns.
    const float scale = is_q ? as_type<float>(int(QBITS)) : as_type<float>(int(KBITS));
    device float* out = (is_q ? (device float*)q : (device float*)k) +
        ((int64_t)(b * S + s) * HK + (is_q ? hd : hd - HK)) * D + lane * 4;
    for (int i = 0; i < 4; i++) out[i] = float(mul_bf16(bfloat(1.0f), bfloat(vals[i] * inv))) * scale;
  } else {
    const int hv = hd - 2 * HK;
    device bfloat* out = (device bfloat*)v + ((int64_t)(b * S + s) * HV + hv) * D + lane * 4;
    for (int i = 0; i < 4; i++) out[i] = bfloat(vals[i]);
    if (lane == 0) {
      // The shim's exact gate chain.
      const device bfloat* row = pr + (int64_t)s * NP;
      const bfloat x = row[A_OFF + hv] + ((const device bfloat*)dt_bias)[hv];
      const bfloat safe = shim_logaddexp_bf16(bfloat(0), x);
      const bfloat sp = static_cast<float>(x) > 20.0f ? x : safe;
      const float neg = -metal::precise::exp(static_cast<float>(((const device bfloat*)a_log)[hv]));
      ((device float*)g)[(b * S + s) * HV + hv] = metal::precise::exp(neg * static_cast<float>(sp));
      ((device bfloat*)beta)[(b * S + s) * HV + hv] = static_cast<bfloat>(shim_sigmoid_bf16(row[B_OFF + hv]));
    }
  }
"#;

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_on(
    proj: &Array,
    state: &Array,
    weight: &Array,
    a_log: &Array,
    dt_bias: &Array,
    conv_dim: i32,
    z_width: i32,
    heads_k: i32,
    heads_v: i32,
    target: StreamOrDevice,
) -> Result<Prepared> {
    let ps = proj.shape();
    let &[b, s, np] = ps.as_slice() else {
        anyhow::bail!("gdn prep: projection must be [B, S, N]");
    };
    let ws = weight.shape();
    let ks = ws.as_slice().get(1).copied().unwrap_or(0);
    let keep = ks - 1;
    ensure!(
        proj.dtype() == Dtype::Bfloat16
            && state.dtype() == Dtype::Bfloat16
            && weight.dtype() == Dtype::Bfloat16
            && a_log.dtype() == Dtype::Bfloat16
            && dt_bias.dtype() == Dtype::Bfloat16
            && ws.as_slice() == [conv_dim, ks, 1]
            && keep >= 1
            && state.shape().as_slice() == [b, keep, conv_dim]
            && conv_dim == (2 * heads_k + heads_v) * 128
            && np == conv_dim + z_width + 2 * heads_v
            && a_log.size() == heads_v as usize
            && dt_bias.size() == heads_v as usize,
        "gdn prep: unsupported shapes"
    );
    // The op-by-op path's scales: 1/sqrt(Dk) in f32, squared for q.
    let inv_scale = 1.0_f32 / 128.0_f32.sqrt();
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if KERNEL.get().is_none() {
        let built = MetalKernel::builder("ironmlx_gdn_prep_v2")
            .inputs(&["proj", "state", "w", "a_log", "dt_bias"])
            .outputs(&["conv_input", "q", "k", "v", "g", "beta"])
            .header(HEADER)
            .source(SOURCE)
            .build()?;
        let _ = KERNEL.set(built);
    }
    let mut out = KERNEL
        .get()
        .expect("built")
        .dispatch_builder()
        .inputs(&[proj, state, weight, a_log, dt_bias])
        .output_shapes(&[
            Shape::from(&[b, keep + s, conv_dim][..]),
            Shape::from(&[b, s, heads_k, 128][..]),
            Shape::from(&[b, s, heads_k, 128][..]),
            Shape::from(&[b, s, heads_v, 128][..]),
            Shape::from(&[b, s, heads_v][..]),
            Shape::from(&[b, s, heads_v][..]),
        ])
        .output_dtypes(&[
            Dtype::Bfloat16,
            Dtype::Float32,
            Dtype::Float32,
            Dtype::Bfloat16,
            Dtype::Float32,
            Dtype::Bfloat16,
        ])
        .grid(32 * s, 2 * heads_k + heads_v, b)
        .threadgroup(32, 1, 1)
        .template_int("C", conv_dim)
        .template_int("NP", np)
        .template_int("KS", ks)
        .template_int("HK", heads_k)
        .template_int("HV", heads_v)
        .template_int("B_OFF", conv_dim + z_width)
        .template_int("A_OFF", conv_dim + z_width + heads_v)
        .template_int("QBITS", (inv_scale * inv_scale).to_bits() as i32)
        .template_int("KBITS", inv_scale.to_bits() as i32)
        .stream(target)
        .dispatch()?;
    Ok(Prepared {
        conv_input: out.take_at(0)?,
        q: out.take_at(0)?,
        k: out.take_at(0)?,
        v: out.take_at(0)?,
        g: out.take_at(0)?,
        beta: out.take_at(0)?,
    })
}

// RmsNormGated of the op-by-op path: MLX `rms_single_row` over each 128-wide
// head (weight applied in bf16), then the compiled gate
// `bf16((z * sigmoid(z)) * normed)` in float, with z read in place from the
// fused input projection and the output in the out_proj input layout.
const GATED_NORM: &str = r#"
  constexpr int D = 128;
  const uint lane = thread_index_in_simdgroup;
  const int row = int(threadgroup_position_in_grid.x);  // b * S + s
  const int h = int(threadgroup_position_in_grid.y);
  const device bfloat* x = (const device bfloat*)y + ((int64_t)row * HV + h) * D + lane * 4;
  const device bfloat* zr = (const device bfloat*)proj + (int64_t)row * NP + Z_OFF + h * D + lane * 4;
  const device bfloat* wr = (const device bfloat*)w + lane * 4;
  float xs[4];
  float acc = 0;
  for (int i = 0; i < 4; i++) {
    xs[i] = x[i];
    acc += xs[i] * xs[i];
  }
  acc = simd_sum(acc);
  const float inv = metal::precise::rsqrt(acc / float(D) + as_type<float>(int(EPSBITS)));
  device bfloat* o = (device bfloat*)out + ((int64_t)row * HV + h) * D + lane * 4;
  for (int i = 0; i < 4; i++) {
    const bfloat normed = wr[i] * static_cast<bfloat>(xs[i] * inv);
    const float g = static_cast<float>(zr[i]);
    const float t = 1 / (1 + metal::precise::exp(metal::abs(g)));
    const float sig = (g < 0) ? t : 1 - t;
    o[i] = static_cast<bfloat>((g * sig) * static_cast<float>(normed));
  }
"#;

/// Gated RMS norm of the GDN recurrence output `y` `[B, S, Hv, 128]` with
/// the gate `z` at column `z_off` of the fused projection `proj` `[B, S, NP]`.
/// Returns `[B, S, Hv * 128]` bf16.
pub(crate) fn gated_norm_on(
    y: &Array,
    proj: &Array,
    z_off: i32,
    weight: &Array,
    eps: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let ys = y.shape();
    let &[b, s, hv, d] = ys.as_slice() else {
        anyhow::bail!("gdn gated norm: y must be [B, S, Hv, D]");
    };
    let np = proj.shape().as_slice()[2];
    ensure!(
        d == 128
            && y.dtype() == Dtype::Bfloat16
            && proj.dtype() == Dtype::Bfloat16
            && weight.dtype() == Dtype::Bfloat16
            && weight.size() == 128
            && proj.shape().as_slice()[..2] == [b, s]
            && z_off + hv * 128 <= np,
        "gdn gated norm: unsupported shapes"
    );
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if KERNEL.get().is_none() {
        let built = MetalKernel::builder("ironmlx_gdn_gated_norm_v1")
            .inputs(&["y", "proj", "w"])
            .outputs(&["out"])
            .source(GATED_NORM)
            .build()?;
        let _ = KERNEL.set(built);
    }
    Ok(KERNEL
        .get()
        .expect("built")
        .dispatch_builder()
        .inputs(&[y, proj, weight])
        .output_shapes(&[Shape::from(&[b, s, hv * 128][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid(32 * b * s, hv, 1)
        .threadgroup(32, 1, 1)
        .template_int("HV", hv)
        .template_int("NP", np)
        .template_int("Z_OFF", z_off)
        .template_int("EPSBITS", eps.to_bits() as i32)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::cast::astype;
    use serial_test::serial;

    fn rand(shape: &[i32], scale: f64, seed: u64) -> Result<Array> {
        let key = mlx::random::key(seed)?;
        let x = mlx::random::normal()
            .shape(shape)
            .scale(scale)
            .key(&key)
            .sample()?;
        Ok(astype(&x, Dtype::Bfloat16)?)
    }

    fn bits(a: &Array) -> Result<Vec<u32>> {
        Ok(astype(a, Dtype::Float32)?
            .to_vec::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    /// Bit-identical to the op-by-op MLX chain (conv1d, sigmoid, multiply,
    /// IronQkNorm, IronGdnGates) on Qwen3.6 GDN shapes.
    #[test]
    #[serial(mlx_metal)]
    fn matches_op_by_op_chain_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let target = StreamOrDevice::default();
        use mlx::ops::{indexing::slice_on, shape::*};
        let (hk, hv, zw) = (16, 32, 4096);
        let c = (2 * hk + hv) * 128;
        let np = c + zw + 2 * hv;
        let weight = rand(&[c, 4, 1], 0.3, 1)?;
        let a_log = rand(&[hv], 0.5, 2)?;
        let dt_bias = rand(&[hv], 1.0, 3)?;
        for (b, s) in [(1, 1), (1, 8), (2, 3), (1, 2)] {
            let proj = rand(&[b, s, np], 1.5, 10 + s as u64)?;
            let state = rand(&[b, 3, c], 1.5, 20 + s as u64)?;
            let got = prep_on(
                &proj, &state, &weight, &a_log, &dt_bias, c, zw, hk, hv, target,
            )?;
            let qkv = slice_on(&proj, &[0, 0, 0][..], &[b, s, c][..], target)?;
            let ba = |off: i32| slice_on(&proj, &[0, 0, off][..], &[b, s, off + hv][..], target);
            let (bb, aa) = (ba(c + zw)?, ba(c + zw + hv)?);
            let conv_input = concatenate_on(&[&state, &qkv], 1, target)?;
            let conv = mlx::ops::conv1d_on(&conv_input, &weight, 1, 0, 1, c, target)?;
            let conv = &conv * &conv.sigmoid()?;
            let inv = 1.0_f32 / 128.0_f32.sqrt();
            let (q, k) = mlx::quantization::qk_norm_fused_on(
                &conv,
                hk,
                128,
                0,
                hk * 128,
                inv * inv,
                inv,
                1e-6,
                target,
            )?;
            let v = slice_on(&conv, &[0, 0, 2 * hk * 128][..], &[b, s, c][..], target)?;
            let (g, beta) =
                mlx::quantization::gdn_gates_fused_on(&aa, &bb, &a_log, &dt_bias, target)?;
            let shape_k = [b, s, hk, 128];
            assert_eq!(
                bits(&got.conv_input)?,
                bits(&conv_input)?,
                "conv input b={b} s={s}"
            );
            assert_eq!(
                bits(&got.v)?,
                bits(&reshape(&v, &[b, s, hv, 128][..])?)?,
                "v b={b} s={s}"
            );
            assert_eq!(
                bits(&got.q)?,
                bits(&reshape(&q, &shape_k[..])?)?,
                "q b={b} s={s}"
            );
            assert_eq!(
                bits(&got.k)?,
                bits(&reshape(&k, &shape_k[..])?)?,
                "k b={b} s={s}"
            );
            assert_eq!(bits(&got.g)?, bits(&g)?, "g b={b} s={s}");
            assert_eq!(bits(&got.beta)?, bits(&beta)?, "beta b={b} s={s}");
            // Gated norm against RmsNormGated with the z slice.
            let y = rand(&[b, s, hv, 128], 1.0, 30 + s as u64)?;
            let nw = rand(&[128], 0.5, 31)?;
            let norm = ironmlx_core::nn::RmsNormGated::new(nw.clone(), 1e-6);
            let z = slice_on(&proj, &[0, 0, c][..], &[b, s, c + zw][..], target)?;
            let z = reshape(&z, &[b, s, hv, 128][..])?;
            let want = reshape(
                &norm.forward_on(&y, Some(&z), target)?,
                &[b, s, hv * 128][..],
            )?;
            let got = gated_norm_on(&y, &proj, c, &nw, 1e-6, target)?;
            assert_eq!(bits(&got)?, bits(&want)?, "gated norm b={b} s={s}");
        }
        Ok(())
    }
}
