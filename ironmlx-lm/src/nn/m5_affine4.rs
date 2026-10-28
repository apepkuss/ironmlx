//! DFlash2 affine4 tensor-unit projections (on by the M5 profile). The route is
//! scoped to a target forward, including Q1, so verify and serial use the same
//! arithmetic.
//! Packed values are unchanged; reduction order differs from ordinary MLX.
use super::linear::QuantizedLinearParts;
use crate::{core::QuantMode, Result};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::{
    cell::Cell,
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

thread_local! { static DEPTH: Cell<usize> = const { Cell::new(0) }; }
pub(crate) struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get() - 1));
    }
}
pub(crate) fn scope() -> Scope {
    DEPTH.with(|d| d.set(d.get() + 1));
    Scope
}
pub(crate) fn armed() -> bool {
    DEPTH.with(|d| d.get() > 0)
}

fn supported_arch(arch: &str) -> bool {
    arch.strip_prefix("applegpu_g")
        .and_then(|x| {
            x.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse::<u32>()
                .ok()
        })
        .is_some_and(|generation| generation >= 17)
}
/// Whether this GPU has the tensor unit the M5 kernels need.
pub(crate) fn tensor_unit_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| mlx::metal::architecture().is_ok_and(|arch| supported_arch(&arch)))
}
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_DFLASH2_QMM)
            && mlx::metal::architecture().is_ok_and(|arch| supported_arch(&arch))
    })
}

pub(crate) struct Prepared {
    pub(crate) weight: Array,
    pub(crate) sb: Array,
}

// Diagnostic accounting only: counts persistent decode-layout copies.
static PREPARED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PREPARED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (count, bytes) of decode-layout weight copies prepared in this process.
pub fn prepared_totals() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (PREPARED_COUNT.load(Relaxed), PREPARED_BYTES.load(Relaxed))
}
/// Columns per tile of the tiled layout of an `n`-column projection.
pub(crate) fn tile_width(n: i32) -> i32 {
    static COOP: OnceLock<bool> = OnceLock::new();
    if *COOP.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_QMM_COOP)
    }) && n % 64 == 0
    {
        64
    } else {
        32
    }
}
/// Split-K override for the per-shape sweep test (0 = per-shape choice).
static FORCED_SPLIT_K: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(test)]
pub(crate) fn force_split_k(sk: i32) {
    FORCED_SPLIT_K.store(sk, std::sync::atomic::Ordering::Relaxed);
}

/// K slices per threadgroup measured best on the M5 Max (applegpu_g17s) for
/// the Qwen3.6 MoE target and DFlash2 drafter projections, as one kernel in
/// a dependent chain (decode / verify rows): narrow outputs need more slices
/// to keep enough simdgroups streaming. M5 profile setting
/// `M5_TUNED_SPLIT_K`; other devices and shapes keep the default.
fn tuned_split_k(n: i32, k: i32) -> Option<i32> {
    static TUNED: OnceLock<bool> = OnceLock::new();
    if !*TUNED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_TUNED_SPLIT_K)
            && mlx::metal::architecture().is_ok_and(|arch| arch == "applegpu_g17s")
    }) {
        return None;
    }
    match (n, k) {
        (2048, 4096) | (2048, 6144) | (2048, 16384) => Some(8),
        (512, 2048) => Some(16),
        (12352, 2048) | (12288, 2048) | (9216, 2048) => Some(2),
        _ => None,
    }
}

fn split_k(n: i32, k: i32) -> i32 {
    let forced = FORCED_SPLIT_K.load(std::sync::atomic::Ordering::Relaxed);
    if forced > 0 {
        return forced.min((k / 64).max(1));
    }
    if let Some(sk) = tuned_split_k(n, k) {
        return sk;
    }
    if k >= 2048 {
        4
    } else {
        1
    }
}
pub(crate) fn fingerprint() -> String {
    format!(
        "experimental-m5-affine4-v2;tile64={};lane-attn={}",
        tile_width(64) == 64,
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_LANE_ATTN)
    )
}
fn compatible(x: &Array, p: &QuantizedLinearParts<'_>) -> bool {
    let w = p.weight.shape();
    x.ndim() >= 2
        && x.dtype() == Dtype::Bfloat16
        && p.scales.dtype() == Dtype::Bfloat16
        && p.weight.dtype() == Dtype::Uint32
        && w.as_slice().len() == 2
        && w.as_slice()[0] % 32 == 0
        && p.bits == 4
        && p.group_size == 64
        && p.mode == QuantMode::Affine
        && p.biases.is_some_and(|b| b.dtype() == Dtype::Bfloat16)
        && x.shape().as_slice()[x.ndim() - 1] % 64 == 0
        && w.as_slice()[1] * 8 == x.shape().as_slice()[x.ndim() - 1]
}

pub(crate) fn prepare(p: &QuantizedLinearParts<'_>, target: StreamOrDevice) -> Result<Prepared> {
    use mlx::ops::shape::*;
    let shape = p.weight.shape();
    let (n, kw) = (shape.as_slice()[0], shape.as_slice()[1]);
    let nt = tile_width(n);
    let w = reshape_on(p.weight, &[n / nt, nt, kw / 8, 8][..], target)?;
    let w = transpose_axes_on(&w, &[0, 2, 1, 3][..], target)?;
    let w = contiguous_on(&w, false, target)?;
    let weight = reshape_on(&w, &[n, kw][..], target)?;
    let scales = transpose_on(p.scales, target)?;
    let biases = transpose_on(p.biases.expect("validated"), target)?;
    let sb = contiguous_on(&stack_on(&[&scales, &biases], -1, target)?, false, target)?;
    mlx::transforms::eval(&[&weight, &sb])?;
    let bytes = weight.size() * weight.dtype().byte_size() + sb.size() * sb.dtype().byte_size();
    PREPARED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    PREPARED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(Prepared { weight, sb })
}

/// (N, K, row blocks, edge, fused group sums, K slices).
type KernelKey = (i32, i32, i32, bool, bool, i32);

/// Group sums of x inside the tensor kernel instead of a separate dependent
/// dispatch per projection (M5 profile setting `M5_FUSED_XSUM`); same
/// summation order, so the output bits are unchanged.
fn fused_xsum() -> bool {
    static FUSED: OnceLock<bool> = OnceLock::new();
    *FUSED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_FUSED_XSUM)
    })
}

fn kernel(n: i32, k: i32, tmr: i32, edge: bool, fxs: bool) -> Result<MetalKernel> {
    static CACHE: OnceLock<Mutex<HashMap<KernelKey, MetalKernel>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    let sk = split_k(n, k);
    let key = (n, k, tmr, edge, fxs, sk);
    if let Some(kernel) = cache.get(&key) {
        return Ok(kernel.clone());
    }
    // Fusion is identical for serial and verify before selecting this shape.
    let nt = tile_width(n);
    let body = format!(
        "constexpr int N={n}, K={k}, GS=64, NT={nt}, SK={sk}, TMR={tmr}, EDGE={}, FXS={};\n{}",
        i32::from(edge),
        i32::from(fxs),
        if nt == 64 {
            include_str!("m5_affine4_coop.metal")
        } else {
            include_str!("m5_affine4.metal")
        }
    );
    let kernel = MetalKernel::builder(format!(
        "ironmlx_m5_affine4_v2_{n}_{k}_{tmr}_{nt}_{sk}_{}{}",
        i32::from(edge),
        if fxs { "_fxs" } else { "" }
    ))
        .inputs(&["X","XS","Wq","SBt","mdims"]).outputs(&["Y"])
        .header("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;")
        .source(body).build()?;
    cache.insert(key, kernel.clone());
    Ok(kernel)
}

pub(crate) fn forward(
    x: &Array,
    p: QuantizedLinearParts<'_>,
    prepared: &OnceLock<Prepared>,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    if !armed() || !compatible(x, &p) {
        return Ok(None);
    }
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap();
    let m = i32::try_from(x.size() / k as usize)?;
    if m == 0 || m > 128 {
        return Ok(None);
    }
    let n = p.weight.shape().as_slice()[0];
    if prepared.get().is_none() {
        let _ = prepared.set(prepare(&p, target)?);
    }
    let p0 = prepared.get().unwrap();
    run_prepared(x, n, m, p0, p.bias, target).map(Some)
}

/// Whether the M5 route accepts `x` for a statically compatible affine4/g64
/// BF16 weight of `kw` packed words per row. Mirrors [`compatible`] for the
/// activation-dependent terms plus the `0 < M <= 128` row gate.
pub(crate) fn routes_activation(x: &Array, kw: i32) -> bool {
    if !armed() || x.ndim() < 2 || x.dtype() != Dtype::Bfloat16 {
        return false;
    }
    let k = x.shape().as_slice()[x.ndim() - 1];
    if k % 64 != 0 || kw * 8 != k {
        return false;
    }
    let m = x.size() / k as usize;
    m > 0 && m <= 128
}

/// Static weight-side terms of [`compatible`].
pub(crate) fn weight_compatible(p: &QuantizedLinearParts<'_>) -> bool {
    let w = p.weight.shape();
    p.scales.dtype() == Dtype::Bfloat16
        && p.weight.dtype() == Dtype::Uint32
        && w.as_slice().len() == 2
        && w.as_slice()[0] % 32 == 0
        && p.bits == 4
        && p.group_size == 64
        && p.mode == QuantMode::Affine
        && p.biases.is_some_and(|b| b.dtype() == Dtype::Bfloat16)
}

/// Lazily build the tiled decode layout (no evaluation, no accounting).
/// Byte permutation of the checkpoint arrays; arithmetic is unchanged.
pub(crate) fn tiled_from_native(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    target: StreamOrDevice,
) -> Result<Prepared> {
    use mlx::ops::shape::*;
    let shape = weight.shape();
    let (n, kw) = (shape.as_slice()[0], shape.as_slice()[1]);
    let nt = tile_width(n);
    let w = reshape_on(weight, &[n / nt, nt, kw / 8, 8][..], target)?;
    let w = transpose_axes_on(&w, &[0, 2, 1, 3][..], target)?;
    let w = contiguous_on(&w, false, target)?;
    let weight = reshape_on(&w, &[n, kw][..], target)?;
    let scales = transpose_on(scales, target)?;
    let biases = transpose_on(biases, target)?;
    let sb = contiguous_on(&stack_on(&[&scales, &biases], -1, target)?, false, target)?;
    Ok(Prepared { weight, sb })
}

/// Exact inverse of [`tiled_from_native`]: returns `(weight, scales, biases)`.
pub(crate) fn native_from_tiled(
    prepared: &Prepared,
    target: StreamOrDevice,
) -> Result<(Array, Array, Array)> {
    use mlx::ops::{indexing::slice_on, shape::*};
    let shape = prepared.weight.shape();
    let (n, kw) = (shape.as_slice()[0], shape.as_slice()[1]);
    let nt = tile_width(n);
    let w = reshape_on(&prepared.weight, &[n / nt, kw / 8, nt, 8][..], target)?;
    let w = transpose_axes_on(&w, &[0, 2, 1, 3][..], target)?;
    let weight = reshape_on(&contiguous_on(&w, false, target)?, &[n, kw][..], target)?;
    let kg = prepared.sb.shape().as_slice()[0];
    let column = |index: i32| -> Result<Array> {
        let plane = slice_on(
            &prepared.sb,
            &[0, 0, index][..],
            &[kg, n, index + 1][..],
            target,
        )?;
        let plane = squeeze_on(&plane, -1, target)?;
        Ok(contiguous_on(
            &transpose_on(&plane, target)?,
            false,
            target,
        )?)
    };
    Ok((weight, column(0)?, column(1)?))
}

/// Run the M5 kernel with an existing tiled layout. Caller has checked
/// [`routes_activation`] and the static weight terms.
pub(crate) fn forward_tiled(
    x: &Array,
    prepared: &Prepared,
    bias: Option<&Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    let k = *x.shape().as_slice().last().unwrap();
    let m = i32::try_from(x.size() / k as usize)?;
    let n = prepared.weight.shape().as_slice()[0];
    run_prepared(x, n, m, prepared, bias, target)
}

pub(crate) fn kernel_variant(m: i32) -> (i32, bool) {
    let mp = ((m + 15) / 16) * 16;
    let block = mp.min(32);
    (block / 16, mp % block != 0)
}

/// Smallest row count for every kernel variant reachable with 1..=`max_rows`
/// activation rows (rows above 128 never take this route).
pub(crate) fn variant_rows(max_rows: i32) -> Vec<i32> {
    let mut seen = Vec::new();
    let mut rows = Vec::new();
    for m in 1..=max_rows.min(128) {
        let variant = kernel_variant(m);
        if !seen.contains(&variant) {
            seen.push(variant);
            rows.push(m);
        }
    }
    rows
}

/// M5 profile setting (`IRONMLX_EXPERIMENTAL_M5_PRECOMPILE`).
pub fn precompile_requested() -> bool {
    ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_PRECOMPILE)
}

fn run_prepared(
    x: &Array,
    n: i32,
    m: i32,
    p0: &Prepared,
    bias: Option<&Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let k = *shape.as_slice().last().unwrap();
    let mp = ((m + 15) / 16) * 16;
    let block = mp.min(32);
    let (tmr, edge) = kernel_variant(m);
    let dims: Array = (&[m, mp][..], &[2][..]).try_into()?;
    let nt = tile_width(n);
    let fxs = fused_xsum() && nt == 32;
    let xs = if fxs {
        // The fused kernel never reads XS; x stands in for the input slot.
        x.clone()
    } else {
        static XSUM: OnceLock<MetalKernel> = OnceLock::new();
        if XSUM.get().is_none() {
            let built = MetalKernel::builder("ironmlx_m5_affine4_xsum_v1")
                .inputs(&["X","mdims"]).outputs(&["XS"])
                .source("const int M=mdims[0], MP=mdims[1]; const uint m=thread_position_in_grid.y, g=thread_position_in_grid.x; if(g>=K/64 || int(m)>=MP)return; float acc=0.0f; if(int(m)<M) for(int i=0;i<64;i++)acc+=float(X[m*K+g*64+i]); XS[g*MP+m]=acc;")
                .build()?;
            let _ = XSUM.set(built);
        }
        XSUM.get()
            .unwrap()
            .dispatch_builder()
            .inputs(&[x, &dims])
            .template_int("K", k)
            .output_shapes(&[Shape::from(&[k / 64, mp][..])])
            .output_dtypes(&[Dtype::Float32])
            .grid(k / 64, mp, 1)
            .threadgroup((k / 64).min(256), 1, 1)
            .stream(target)
            .dispatch()?
            .take_at(0)?
    };
    let sk = split_k(n, k);
    let y = kernel(n, k, tmr, edge, fxs)?
        .dispatch_builder()
        .inputs(&[x, &xs, &p0.weight, &p0.sb, &dims])
        .output_shapes(&[Shape::from(&[m, n][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid(n / 32 * 32 * sk, (mp + block - 1) / block, 1)
        .threadgroup(nt * sk, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?;
    let mut output_shape = shape.as_slice().to_vec();
    *output_shape.last_mut().unwrap() = n;
    let mut y = mlx::ops::shape::reshape_on(&y, output_shape.as_slice(), target)?;
    if let Some(bias) = bias {
        y = &y + bias;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    #[test]
    fn precompile_rows_cover_every_reachable_variant() {
        // Every row count the route accepts maps to a variant that the
        // start-up pre-compilation reaches with its representative rows.
        let decoder = variant_rows(128);
        assert_eq!(decoder, vec![1, 17, 33]);
        let covered: Vec<_> = decoder.iter().map(|&m| kernel_variant(m)).collect();
        for m in 1..=128 {
            assert!(covered.contains(&kernel_variant(m)), "m={m}");
        }
        assert_eq!(variant_rows(16), vec![1]);
        assert_eq!(kernel_variant(46), (2, true));
        assert_eq!(kernel_variant(40), (2, true));
        assert_eq!(kernel_variant(49), (2, false));
        assert_eq!(kernel_variant(15), (1, false));
        assert!(!(1..=128).any(|m| kernel_variant(m) == (1, true)));
    }

    #[test]
    fn architecture_gate_is_forward_compatible() {
        assert!(supported_arch("applegpu_g17s"));
        assert!(supported_arch("applegpu_g18g"));
        assert!(!supported_arch("applegpu_g16s"));
        assert!(!supported_arch("unknown"));
    }

    #[test]
    #[serial(mlx_metal)]
    fn tensor_qmm_matches_reference_and_is_row_invariant() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !mlx::metal::architecture().is_ok_and(|a| supported_arch(&a)) {
            return Ok(());
        }
        use mlx::ops::{cast::astype, shape::*};
        for k in [128, 2048] {
            let n = 64;
            let weights = (0..n * k)
                .map(|i| ((i * 17 % 79) as f32 - 39.0) * 0.004)
                .collect::<Vec<_>>();
            let w: Array = (weights.as_slice(), &[n, k][..]).try_into()?;
            let w = astype(&w, Dtype::Bfloat16)?;
            let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
            let layer = super::super::linear::Linear::new_quant(
                q[0].clone(),
                q[1].clone(),
                Some(q[2].clone()),
                None,
                64,
                4,
            );
            let values = (0..k)
                .map(|i| ((i * 13 % 61) as f32 - 30.0) * 0.009)
                .collect::<Vec<_>>();
            let x: Array = (values.as_slice(), &[1, 1, k][..]).try_into()?;
            let x = astype(&x, Dtype::Bfloat16)?;
            let reference = astype(&layer.forward(&x)?, Dtype::Float32)?.to_vec::<f32>()?;
            let _route = scope();
            let one = astype(&layer.forward(&x)?, Dtype::Float32)?.to_vec::<f32>()?;
            let max_error = one
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let rms = (reference.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
            eprintln!("K={k} max_abs={max_error} reference_rms={rms}");
            assert!(
                max_error < 0.02 * rms.max(0.1),
                "unexpected numerical error {max_error}"
            );
            for rows in [2, 8, 16, 17, 32, 33, 64, 128] {
                let x = concatenate(&vec![&x; rows], 1)?;
                let actual = astype(&layer.forward(&x)?, Dtype::Float32)?.to_vec::<f32>()?;
                for row in actual.chunks(n as usize) {
                    assert_eq!(row, one.as_slice(), "K={k} M={rows}");
                }
            }
        }
        Ok(())
    }

    /// Diagnostic: cost of converting between checkpoint and tiled decode
    /// layouts on production geometries (count-weighted). Run with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn layout_conversion_cost_on_production_shapes() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        use mlx::ops::{cast::astype, indexing::slice_on, shape::*};
        // (N, K, count per model)
        let shapes = [
            (34816, 5120, 64),
            (5120, 17408, 64),
            (16480, 5120, 48),
            (5120, 6144, 64),
            (14336, 5120, 16),
            (248320, 5120, 1),
        ];
        let target = StreamOrDevice::default();
        let mut total_fwd = 0.0_f64;
        let mut total_inv = 0.0_f64;
        for (seed, &(n, k, count)) in shapes.iter().enumerate() {
            let key = mlx::random::key(7 + seed as u64)?;
            let w = mlx::random::normal()
                .shape(&[n, k][..])
                .dtype(Dtype::Float32)
                .scale(0.02)
                .key(&key)
                .sample()?;
            let w = astype(&w, Dtype::Bfloat16)?;
            let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
            mlx::transforms::eval(&[&q[0], &q[1], &q[2]])?;
            let parts = QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: QuantMode::Affine,
            };
            let mut fwd = Vec::new();
            let mut inv = Vec::new();
            for _ in 0..7 {
                let started = std::time::Instant::now();
                let prepared = prepare(&parts, target)?;
                fwd.push(started.elapsed().as_secs_f64());
                let started = std::time::Instant::now();
                let kw = k / 8;
                let w4 = reshape_on(&prepared.weight, &[n / 32, kw / 8, 32, 8][..], target)?;
                let w4 = transpose_axes_on(&w4, &[0, 2, 1, 3][..], target)?;
                let native = reshape_on(&contiguous_on(&w4, false, target)?, &[n, kw][..], target)?;
                let sb = &prepared.sb;
                let scales = contiguous_on(
                    &transpose_on(
                        &squeeze_on(
                            &slice_on(sb, &[0, 0, 0][..], &[k / 64, n, 1][..], target)?,
                            -1,
                            target,
                        )?,
                        target,
                    )?,
                    false,
                    target,
                )?;
                let biases = contiguous_on(
                    &transpose_on(
                        &squeeze_on(
                            &slice_on(sb, &[0, 0, 1][..], &[k / 64, n, 2][..], target)?,
                            -1,
                            target,
                        )?,
                        target,
                    )?,
                    false,
                    target,
                )?;
                mlx::transforms::eval(&[&native, &scales, &biases])?;
                inv.push(started.elapsed().as_secs_f64());
                if fwd.len() == 1 {
                    let eq = |a: &Array, b: &Array| -> Result<bool> {
                        let same = mlx::ops::equal(a, b)?;
                        Ok(mlx::ops::all(&same, &[0, 1][..], false)?.item::<bool>()?)
                    };
                    assert!(eq(&native, &q[0])? && eq(&scales, &q[1])? && eq(&biases, &q[2])?);
                }
            }
            fwd.sort_by(f64::total_cmp);
            inv.sort_by(f64::total_cmp);
            let (f, i) = (fwd[3], inv[3]);
            total_fwd += f * count as f64;
            total_inv += i * count as f64;
            eprintln!(
                "convert N={n} K={k} count={count} to_tiled_ms={:.3} to_native_ms={:.3}",
                f * 1e3,
                i * 1e3
            );
        }
        eprintln!(
            "convert model_total to_tiled_ms={:.1} to_native_ms={:.1}",
            total_fwd * 1e3,
            total_inv * 1e3
        );
        Ok(())
    }

    /// Diagnostic: peak tensor-unit throughput of MPP matmul2d per operand
    /// type (bfloat x bfloat, half x half, int8 x int8 -> int32), each
    /// simdgroup repeating a rows x 32 x 32 multiply-accumulate from
    /// threadgroup operands. Run with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn tensor_unit_peak_by_type() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let target = StreamOrDevice::default();
        // (name, left type, right type, right in device memory, accumulator).
        for (name, in_t, right_t, right_device, acc_t) in [
            ("bf16", "bfloat", "bfloat", false, "float"),
            ("f16", "half", "half", false, "float"),
            ("i8", "int8_t", "int8_t", false, "int"),
            ("i8xi4_tg", "int8_t", "int4b_format", false, "int"),
            ("i8xi4_dv", "int8_t", "int4b_format", true, "int"),
            ("bf16xu4_dv", "bfloat", "uint4b_format", true, "float"),
            ("u8xu4_dv", "uint8_t", "uint4b_format", true, "int"),
        ] {
            for rows in [32, 64] {
                let iters = 4096;
                let packed = right_t.ends_with("4b_format");
                let right_decl = if right_device {
                    format!("tensor<device {right_t}, dextents<int32_t, 2>, tensor_inline> tb((device uchar*)X, dextents<int32_t, 2>(32, 32));")
                } else if packed {
                    format!(
                        "threadgroup uchar sb[4][32 * 32 / 2];\n\
                         for (int i = lane; i < 32 * 32 / 2; i += 32) sb[sg][i] = uchar(i * 37);\n\
                         tensor<threadgroup {right_t}, dextents<int32_t, 2>, tensor_inline> tb((threadgroup uchar*)sb[sg], dextents<int32_t, 2>(32, 32));"
                    )
                } else {
                    format!(
                        "threadgroup {right_t} sb[4][32 * 32];\n\
                         for (int i = lane; i < 32 * 32; i += 32) sb[sg][i] = {right_t}((i % 5) - 2);\n\
                         tensor<threadgroup {right_t}, dextents<int32_t, 2>, tensor_inline> tb(sb[sg], dextents<int32_t, 2>(32, 32));"
                    )
                };
                let src = format!(
                    "constexpr auto desc = matmul2d_descriptor({rows}, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);\n\
                     matmul2d<desc, execution_simdgroup> op;\n\
                     threadgroup {in_t} sa[4][{rows} * 32];\n\
                     const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;\n\
                     for (int i = lane; i < {rows} * 32; i += 32) sa[sg][i] = {in_t}((i % 7) - 3);\n\
                     {right_decl}\n\
                     simdgroup_barrier(mem_flags::mem_threadgroup);\n\
                     tensor<threadgroup {in_t}, dextents<int32_t, 2>, tensor_inline> ta(sa[sg], dextents<int32_t, 2>(32, {rows}));\n\
                     auto P = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), {acc_t}>();\n\
                     for (int i = 0; i < P.get_capacity(); i++) P[i] = 0;\n\
                     for (int it = 0; it < {iters}; it++) op.run(ta, tb, P);\n\
                     {acc_t} s = 0; for (int i = 0; i < P.get_capacity(); i++) s += P[i];\n\
                     Y[threadgroup_position_in_grid.x * 128 + sg * 32 + lane] = float(s);"
                );
                let kernel = MetalKernel::builder(format!("ironmlx_tensor_peak_{name}_{rows}"))
                    .inputs(&["X"])
                    .outputs(&["Y"])
                    .header("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;")
                    .source(src)
                    .build()?;
                let bytes: Vec<u8> = (0..512u32).map(|i| (i * 37) as u8).collect();
                let x: Array = (bytes.as_slice(), &[512][..]).try_into()?;
                let groups = 40 * 32;
                let mut runs = Vec::new();
                for _ in 0..5 {
                    let started = std::time::Instant::now();
                    let y = kernel
                        .dispatch_builder()
                        .inputs(&[&x])
                        .output_shapes(&[Shape::from(&[groups * 128][..])])
                        .output_dtypes(&[Dtype::Float32])
                        .grid(groups * 128, 1, 1)
                        .threadgroup(128, 1, 1)
                        .stream(target)
                        .dispatch()?
                        .take_at(0)?;
                    mlx::transforms::eval(&[&y])?;
                    runs.push(started.elapsed().as_secs_f64());
                }
                runs.sort_by(f64::total_cmp);
                let t = runs[runs.len() / 2];
                let ops = 2.0 * rows as f64 * 32.0 * 32.0 * iters as f64 * (groups * 4) as f64;
                eprintln!(
                    "PEAK {name} rows={rows}: {:.2} ms, {:.1} TOPS",
                    t * 1e3,
                    ops / t / 1e12
                );
            }
        }
        Ok(())
    }

    /// Diagnostic: Dense causal prefill attention of the last 2048-row chunk
    /// (24 query / 4 KV heads, D256) over long caches, under the installed
    /// M5 Dense profile (D256 NAX prefill attention). CHECK prints the
    /// output's max difference against an f32 reference of one query head.
    /// Run with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn dense_prefill_attention_throughput() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        ironmlx_core::m5_profile::install_for_target(
            ironmlx_core::m5_profile::M5ProfileMode::Auto,
            true,
            ironmlx_core::m5_profile::M5ProfileTarget::Qwen35Dense,
        );
        use mlx::ops::cast::astype;
        let rows = 2048;
        for keys in [8192, 32768, 65536, 131072] {
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
            let mut runs = Vec::new();
            let mut out = None;
            for _ in 0..7 {
                let started = std::time::Instant::now();
                let o = mlx::fast::scaled_dot_product_attention(
                    &q, &k, &v, 0.0625, "causal", None, None,
                )?;
                mlx::transforms::eval(&[&o])?;
                runs.push(started.elapsed().as_secs_f64());
                out = Some(o);
            }
            runs.sort_by(f64::total_cmp);
            let t = runs[runs.len() / 2];
            // Causal last chunk: row r sees keys - rows + r + 1 keys.
            let pairs =
                rows as f64 * (keys - rows) as f64 + rows as f64 * (rows as f64 + 1.0) / 2.0;
            let flops = 4.0 * 24.0 * 256.0 * pairs;
            // f32 reference of query head 0 (KV head 0).
            let q0 = astype(
                &mlx::ops::indexing::slice(&q, &[0, 0, 0, 0][..], &[1, 1, rows, 256][..])?,
                Dtype::Float32,
            )?;
            let k0 = astype(
                &mlx::ops::indexing::slice(&k, &[0, 0, 0, 0][..], &[1, 1, keys, 256][..])?,
                Dtype::Float32,
            )?;
            let v0 = astype(
                &mlx::ops::indexing::slice(&v, &[0, 0, 0, 0][..], &[1, 1, keys, 256][..])?,
                Dtype::Float32,
            )?;
            let reference = mlx::fast::scaled_dot_product_attention(
                &q0, &k0, &v0, 0.0625, "causal", None, None,
            )?;
            let o0 = astype(
                &mlx::ops::indexing::slice(
                    out.as_ref().unwrap(),
                    &[0, 0, 0, 0][..],
                    &[1, 1, rows, 256][..],
                )?,
                Dtype::Float32,
            )?;
            let diff = mlx::ops::reduction::max(
                &mlx::ops::unary::abs(&(&o0 - &reference))?,
                mlx::ops::reduction::All,
                false,
            )?
            .item::<f32>()?;
            eprintln!(
                "PREFILL attention keys={keys} rows={rows}: {:.2} ms {:.1} TFLOPS | CHECK max|o-ref| head0 {:.3e}",
                t * 1e3,
                flops / t / 1e12,
                diff
            );
        }
        Ok(())
    }

    /// Diagnostic: throughput of the Dense prefill projections at a 2048-row
    /// chunk, the installed quantized matmul (M5 Dense profile: MLX NAX QMM
    /// with the BM128 M tile) against a BF16 matmul of the same shape. Run
    /// with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn dense_prefill_projection_throughput() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        ironmlx_core::m5_profile::install_for_target(
            ironmlx_core::m5_profile::M5ProfileMode::Auto,
            true,
            ironmlx_core::m5_profile::M5ProfileTarget::Qwen35Dense,
        );
        use mlx::ops::cast::astype;
        for m in [2048, 4096] {
            for (n, k) in [
                (34816, 5120),
                (5120, 17408),
                (16480, 5120),
                (12288, 5120),
                (5120, 6144),
            ] {
                let key = mlx::random::key(11)?;
                let w = mlx::random::normal()
                    .shape(&[n, k][..])
                    .scale(0.02)
                    .key(&key)
                    .sample()?;
                let w = astype(&w, Dtype::Bfloat16)?;
                let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
                let x = mlx::random::normal()
                    .shape(&[m, k][..])
                    .scale(0.5)
                    .key(&mlx::random::key(12)?)
                    .sample()?;
                let x = astype(&x, Dtype::Bfloat16)?;
                let wt = mlx::ops::shape::transpose(&w)?;
                let wt = mlx::ops::shape::contiguous(&wt, false)?;
                mlx::transforms::eval(&[&q[0], &q[1], &q[2], &x, &wt])?;
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                let time = |f: &dyn Fn() -> Result<Array>| -> Result<f64> {
                    let mut runs = Vec::new();
                    for _ in 0..12 {
                        let started = std::time::Instant::now();
                        let mut outs = Vec::new();
                        for _ in 0..4 {
                            outs.push(f()?);
                            mlx::transforms::eval(&[outs.last().unwrap()])?;
                        }
                        runs.push(started.elapsed().as_secs_f64() / 4.0);
                    }
                    runs.sort_by(f64::total_cmp);
                    Ok(runs[runs.len() / 2])
                };
                let tq = time(&|| {
                    Ok(mlx::quantization::quantized_matmul(
                        &x,
                        &q[0],
                        &q[1],
                        Some(&q[2]),
                        true,
                        Some(64),
                        Some(4),
                        "affine",
                    )?)
                })?;
                let tb = time(&|| Ok(mlx::ops::matmul::matmul(&x, &wt)?))?;
                // Dequantize to BF16 inside the call, then a BF16 matmul.
                let td = time(&|| {
                    let w = mlx::quantization::dequantize(
                        &q[0],
                        &q[1],
                        Some(&q[2]),
                        Some(64),
                        Some(4),
                        "affine",
                        None,
                        None,
                    )?;
                    Ok(mlx::ops::matmul::matmul(
                        &x,
                        &mlx::ops::shape::transpose(&w)?,
                    )?)
                })?;
                eprintln!(
                "PREFILL n={n} k={k} m={m}: qmm {:.2} ms {:.1} TFLOPS | bf16 matmul {:.2} ms {:.1} TFLOPS | dequant+matmul {:.2} ms {:.1} TFLOPS",
                tq * 1e3,
                flops / tq / 1e12,
                tb * 1e3,
                flops / tb / 1e12,
                td * 1e3,
                flops / td / 1e12
            );
            }
        }
        Ok(())
    }

    /// Diagnostic: per-shape split-K sweep in a dependent chain of the
    /// production projection shapes (the source of `tuned_split_k`). Run
    /// with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn split_k_sweep_production_shapes() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        use mlx::ops::cast::astype;
        let target = StreamOrDevice::default();
        // SPLIT_K_SWEEP_SHAPES=dense sweeps the Qwen3.8 Dense target shapes
        // (fused gate/up, fused GatedDelta input, attention q+gate / k / v,
        // the three 5120-wide output projections, LM head) at its verify rows.
        let dense = std::env::var("SPLIT_K_SWEEP_SHAPES").as_deref() == Ok("dense");
        let shapes: &[(i32, i32)] = if dense {
            &[
                (34816, 5120),
                (16480, 5120),
                (12288, 5120),
                (1024, 5120),
                (5120, 6144),
                (5120, 17408),
                (248320, 5120),
            ]
        } else {
            &[
                (2048, 4096),
                (12352, 2048),
                (9216, 2048),
                (6144, 2048),
                (12288, 2048),
                (2048, 6144),
                (2048, 16384),
                (512, 2048),
                (248320, 2048),
            ]
        };
        let rows: &[i32] = if dense { &[1, 16, 32] } else { &[1, 8, 16] };
        for &(n, k) in shapes {
            let bytes_one = (n as f64) * (k as f64) * (0.5 + 4.0 / 64.0);
            let layers = ((1.5e9 / bytes_one) as usize).clamp(2, 40);
            let mut weights = Vec::new();
            for layer in 0..layers {
                let key = mlx::random::key(300 + layer as u64)?;
                let w = mlx::random::normal()
                    .shape(&[n, k][..])
                    .scale(0.05)
                    .key(&key)
                    .sample()?;
                let w = astype(&w, Dtype::Bfloat16)?;
                let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
                let p = tiled_from_native(&q[0], &q[1], &q[2], target)?;
                mlx::transforms::eval(&[&p.weight, &p.sb])?;
                weights.push(p);
            }
            for &m in rows {
                let x0: Vec<f32> = (0..m * k).map(|v| ((v % 13) as f32 - 6.0) * 0.05).collect();
                let x0: Array = (x0.as_slice(), &[m, k][..]).try_into()?;
                let x0 = astype(&x0, Dtype::Bfloat16)?;
                let mut line = format!("N=sk n={n} k={k} m={m}");
                for sk in [1, 2, 4, 8, 16] {
                    let sk_max: i32 = std::env::var("SPLIT_K_MAX")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(16);
                    if sk > k / 64 || sk > sk_max {
                        continue;
                    }
                    force_split_k(sk);
                    let mut runs = Vec::new();
                    for _ in 0..7 {
                        let started = std::time::Instant::now();
                        let mut x = x0.clone();
                        for w in &weights {
                            let y = forward_tiled(&x, w, None, target)?;
                            // Next input: the first k outputs (tiled when N < K).
                            let y = mlx::ops::indexing::slice_on(
                                &y,
                                &[0, 0][..],
                                &[m, k.min(n)][..],
                                target,
                            )?;
                            x = if n >= k {
                                y
                            } else {
                                let reps = vec![&y; ((k + n - 1) / n) as usize];
                                let tiled = mlx::ops::shape::concatenate_on(&reps, 1, target)?;
                                if k % n == 0 {
                                    tiled
                                } else {
                                    mlx::ops::indexing::slice_on(
                                        &tiled,
                                        &[0, 0][..],
                                        &[m, k][..],
                                        target,
                                    )?
                                }
                            };
                        }
                        mlx::transforms::eval(&[&x])?;
                        runs.push(started.elapsed().as_secs_f64() * 1e6 / layers as f64);
                    }
                    runs.sort_by(f64::total_cmp);
                    line.push_str(&format!(" sk{sk}={:.1}", runs[runs.len() / 2]));
                }
                force_split_k(0);
                eprintln!("{line} (us/layer, {layers} layers)");
            }
        }
        Ok(())
    }
}
