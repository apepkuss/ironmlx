//! Opt-in DFlash2 affine4 tensor-unit experiment. The route is scoped to a
//! target forward, including Q1, so verify and serial use the same arithmetic.
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
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_DFLASH2_QMM)
            && mlx::metal::architecture().is_ok_and(|arch| supported_arch(&arch))
    })
}

pub(crate) struct Prepared {
    weight: Array,
    sb: Array,
}

// Diagnostic accounting only: counts persistent decode-layout copies.
static PREPARED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PREPARED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (count, bytes) of decode-layout weight copies prepared in this process.
pub fn prepared_totals() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (PREPARED_COUNT.load(Relaxed), PREPARED_BYTES.load(Relaxed))
}
fn tile_width(n: i32) -> i32 {
    static COOP: OnceLock<bool> = OnceLock::new();
    if *COOP.get_or_init(|| std::env::var("IRONMLX_EXPERIMENTAL_M5_QMM_COOP").as_deref() == Ok("1"))
        && n % 64 == 0
    {
        64
    } else {
        32
    }
}
fn split_k(n: i32, k: i32) -> i32 {
    static SHAPE: OnceLock<bool> = OnceLock::new();
    if *SHAPE.get_or_init(|| {
        std::env::var("IRONMLX_EXPERIMENTAL_M5_QMM_SHAPE_SPLIT").as_deref() == Ok("1")
    }) {
        let mut sk = 1;
        while sk < 8 && (n / 32) * sk < 1024 && (k / 64) / (sk * 2) >= 8 {
            sk *= 2;
        }
        sk
    } else if k >= 2048 {
        4
    } else {
        1
    }
}
pub(crate) fn fingerprint() -> String {
    format!(
        "experimental-m5-affine4-v2;tile64={};shape-split={};grouped-attn={};lane-attn={}",
        tile_width(64) == 64,
        split_k(34816, 5120) == 1,
        std::env::var("IRONMLX_EXPERIMENTAL_GROUPED_VERIFY_ATTN").as_deref() == Ok("1"),
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

fn prepare(p: &QuantizedLinearParts<'_>, target: StreamOrDevice) -> Result<Prepared> {
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

type KernelKey = (i32, i32, i32, bool);

fn kernel(n: i32, k: i32, tmr: i32, edge: bool) -> Result<MetalKernel> {
    static CACHE: OnceLock<Mutex<HashMap<KernelKey, MetalKernel>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    let key = (n, k, tmr, edge);
    if let Some(kernel) = cache.get(&key) {
        return Ok(kernel.clone());
    }
    // Fusion is identical for serial and verify before selecting this shape.
    let sk = split_k(n, k);
    let nt = tile_width(n);
    let body = format!(
        "constexpr int N={n}, K={k}, GS=64, NT={nt}, SK={sk}, TMR={tmr}, EDGE={};\n{}",
        i32::from(edge),
        if nt == 64 {
            include_str!("m5_affine4_coop.metal")
        } else {
            include_str!("m5_affine4.metal")
        }
    );
    let kernel = MetalKernel::builder(format!("ironmlx_m5_affine4_v2_{n}_{k}_{tmr}_{nt}_{sk}_{}",i32::from(edge)))
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

/// Kernel variant `(tmr, edge)` used for `m` activation rows (part of the
/// compiled kernel key together with N and K).
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
    static XSUM: OnceLock<MetalKernel> = OnceLock::new();
    if XSUM.get().is_none() {
        let built = MetalKernel::builder("ironmlx_m5_affine4_xsum_v1")
            .inputs(&["X","mdims"]).outputs(&["XS"])
            .source("const int M=mdims[0], MP=mdims[1]; const uint m=thread_position_in_grid.y, g=thread_position_in_grid.x; if(g>=K/64 || int(m)>=MP)return; float acc=0.0f; if(int(m)<M) for(int i=0;i<64;i++)acc+=float(X[m*K+g*64+i]); XS[g*MP+m]=acc;")
            .build()?;
        let _ = XSUM.set(built);
    }
    let xs = XSUM
        .get()
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
        .take_at(0)?;
    let sk = split_k(n, k);
    let nt = tile_width(n);
    let y = kernel(n, k, tmr, edge)?
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
}
