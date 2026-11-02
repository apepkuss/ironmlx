//! Flash affine4/g32 projections on M5 tensor units, reading the original
//! checkpoint layout. No second resident weight layout is retained. This
//! candidate is qualification-only: the native-layout MPP path measured
//! slower than MLX QMV on M5 Max and is intentionally absent from routing.
use super::linear::QuantizedLinearParts;
use crate::{core::weights::QuantMode, Result};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
fn kernel(n: i32, k: i32, split: i32) -> Result<MetalKernel> {
    static CACHE: OnceLock<Mutex<HashMap<(i32, i32, i32), MetalKernel>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(kernel) = cache.get(&(n, k, split)) {
        return Ok(kernel.clone());
    }
    let source = format!(
        "constexpr int N={n}, K={k}, GS=32, NT=32, SK={split}, TMR=1, EDGE=0, FXS=1;\n{}",
        include_str!("flash_m5_qmm.metal")
    );
    let built=MetalKernel::builder(format!("ironmlx_flash_m5_g32_{n}_{k}_{split}"))
        .inputs(&["X","Wq","S","B","mdims"]).outputs(&["Y"])
        .header("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;")
        .source(source).build()?;
    cache.insert((n, k, split), built.clone());
    Ok(built)
}
pub(crate) fn forward(
    x: &Array,
    p: QuantizedLinearParts<'_>,
    split: i32,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    let shape = x.shape();
    let d = shape.as_slice();
    let w = p.weight.shape();
    let wd = w.as_slice();
    if d.len() < 2
        || wd.len() != 2
        || p.bits != 4
        || p.group_size != 32
        || p.mode != QuantMode::Affine
        || x.dtype() != Dtype::Bfloat16
        || p.scales.dtype() != Dtype::Bfloat16
        || p.weight.dtype() != Dtype::Uint32
        || !p.biases.is_some_and(|a| a.dtype() == Dtype::Bfloat16)
        || wd[0] % 32 != 0
        || d[d.len() - 1] % 32 != 0
        || wd[1] * 8 != d[d.len() - 1]
    {
        return Ok(None);
    }
    let k = d[d.len() - 1];
    let n = wd[0];
    let m = x.size() / k as usize;
    if !(1..=8).contains(&m) {
        return Ok(None);
    }
    let x = mlx::ops::shape::contiguous_on(x, false, target)?;
    let dims = Array::try_from((&[m as i32, 16][..], [2]))?;
    let output = kernel(n, k, split)?
        .dispatch_builder()
        .inputs(&[&x, p.weight, p.scales, p.biases.unwrap(), &dims])
        .output_shapes(&[Shape::from(&[m as i32, n][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid(n / 32 * 32 * split, 1, 1)
        .threadgroup(32 * split, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?;
    let mut out_shape = d.to_vec();
    *out_shape.last_mut().unwrap() = n;
    let mut output = output.reshape_on(out_shape.as_slice(), target)?;
    if let Some(bias) = p.bias {
        output = mlx::ops::binary::add_on(&output, bias, target)?;
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires generation-17 GPU; benchmarks and checks native g32 MPP"]
    fn flash_m5_native_g32_is_row_invariant_and_numerically_bounded() {
        assert!(mlx::metal::architecture()
            .unwrap()
            .starts_with("applegpu_g17"));
        for (n, k) in [
            (640, 2560),
            (2560, 640),
            (12288, 2560),
            (10240, 2560),
            (2560, 6144),
            (248320, 2560),
        ] {
            let key = mlx::random::key(47).unwrap();
            let weights = mlx::random::normal()
                .shape(&[n, k][..])
                .dtype(Dtype::Float32)
                .scale(0.02)
                .key(&key)
                .sample()
                .unwrap()
                .astype(Dtype::Bfloat16)
                .unwrap();
            let quant =
                mlx::quantization::quantize(&weights, Some(32), Some(4), "affine", None).unwrap();
            let x = mlx::random::normal()
                .shape(&[1, 8, k][..])
                .dtype(Dtype::Bfloat16)
                .key(&key)
                .sample()
                .unwrap();
            let parts = || QuantizedLinearParts {
                weight: &quant[0],
                scales: &quant[1],
                biases: Some(&quant[2]),
                bias: None,
                group_size: 32,
                bits: 4,
                mode: QuantMode::Affine,
            };
            let reference = mlx::quantization::quantized_matmul(
                &x,
                &quant[0],
                &quant[1],
                Some(&quant[2]),
                true,
                Some(32),
                Some(4),
                "affine",
            )
            .unwrap();
            reference.eval().unwrap();
            for split in [1, 2, 4, 8] {
                let y = forward(&x, parts(), split, ().into()).unwrap().unwrap();
                y.eval().unwrap();
                let a = y.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
                let b = reference
                    .astype(Dtype::Float32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let relative = (a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>()
                    / b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
                .sqrt();
                assert!(
                    relative < 0.01,
                    "g32 MPP N{n} K{k} split{split}: {relative}"
                );
                let first = x.slice(&[0, 0, 0][..], &[1, 1, k][..]).unwrap();
                let one = forward(&first, parts(), split, ().into()).unwrap().unwrap();
                let row = y.slice(&[0, 0, 0][..], &[1, 1, n][..]).unwrap();
                assert_eq!(
                    one.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap(),
                    row.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap()
                );
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    forward(&first, parts(), split, ().into())
                        .unwrap()
                        .unwrap()
                        .eval()
                        .unwrap();
                }
                eprintln!(
                    "g32 N={n} K={k} split={split} relative_rms={relative:.6} one_us={:.1}",
                    start.elapsed().as_secs_f64() * 1e6 / 20.
                );
            }
            let first = x.slice(&[0, 0, 0][..], &[1, 1, k][..]).unwrap();
            mlx::quantization::quantized_matmul(
                &first,
                &quant[0],
                &quant[1],
                Some(&quant[2]),
                true,
                Some(32),
                Some(4),
                "affine",
            )
            .unwrap()
            .eval()
            .unwrap();
            let start = std::time::Instant::now();
            for _ in 0..20 {
                mlx::quantization::quantized_matmul(
                    &first,
                    &quant[0],
                    &quant[1],
                    Some(&quant[2]),
                    true,
                    Some(32),
                    Some(4),
                    "affine",
                )
                .unwrap()
                .eval()
                .unwrap();
            }
            eprintln!(
                "native N={n} K={k} one_us={:.1}",
                start.elapsed().as_secs_f64() * 1e6 / 20.
            );
        }
    }
}
