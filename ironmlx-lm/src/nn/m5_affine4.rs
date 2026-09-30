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
        std::env::var("IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM").as_deref() == Ok("1")
            && mlx::metal::architecture().is_ok_and(|arch| supported_arch(&arch))
    })
}

pub(crate) struct Prepared {
    weight: Array,
    sb: Array,
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
        std::env::var("IRONMLX_EXPERIMENTAL_M5_LANE_ATTN").as_deref() == Ok("1")
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
    Ok(Prepared { weight, sb })
}

fn kernel(n: i32, k: i32, tmr: i32, edge: bool) -> Result<MetalKernel> {
    static CACHE: OnceLock<Mutex<HashMap<(i32, i32, i32, bool), MetalKernel>>> = OnceLock::new();
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
    let mp = ((m + 15) / 16) * 16;
    let block = mp.min(32);
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
    let y = kernel(n, k, block / 16, mp % block != 0)?
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
    if let Some(bias) = p.bias {
        y = &y + bias;
    }
    Ok(Some(y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
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
}
