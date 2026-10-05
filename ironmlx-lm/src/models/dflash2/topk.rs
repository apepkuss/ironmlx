//! Experimental exact BF16 top-k; no vocabulary restriction or logit rounding.
use crate::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

pub(super) fn candidates(logits: &Array, k: i32, target: StreamOrDevice) -> Result<Option<Array>> {
    if !ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::DFLASH2_RADIX_TOPK)
        || logits.dtype() != Dtype::Bfloat16
        || !(1..=64).contains(&k)
    {
        return Ok(None);
    }
    let shape = logits.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(dims.len() == 3 && dims[2] >= k, "invalid top-k shape");
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::builder("ironmlx_dflash2_radix_topk_v1")
            .inputs(&["X", "dims"])
            .outputs(&["IDX"])
            .header("\n[[max_total_threads_per_threadgroup(1024)]]\n")
            .source(include_str!("topk.metal"))
            .build()
            .expect("build radix topk")
    });
    let x = mlx::ops::shape::contiguous_on(logits, false, target)?;
    let meta: Array = (&[dims[2], k][..], &[2][..]).try_into()?;
    Ok(Some(
        kernel
            .dispatch_builder()
            .inputs(&[&x, &meta])
            .output_shapes(&[Shape::from(&[dims[0], dims[1], k][..])])
            .output_dtypes(&[Dtype::Uint32])
            .grid(dims[0] * dims[1] * 1024, 1, 1)
            .threadgroup(1024, 1, 1)
            .stream(target)
            .dispatch()?
            .take_at(0)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn radix_topk_handles_rows_ties_and_full_vocab() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        std::env::set_var("IRONMLX_EXPERIMENTAL_DFLASH2_RADIX_TOPK", "1");
        for n in [257, 8193, 248320] {
            let mut values = Vec::new();
            for row in 0..3 {
                for i in 0..n {
                    values.push(match row {
                        0 => 1.0,
                        1 => (i % 113) as f32 * 0.03 - 1.5,
                        _ => ((i * 7919) % 65521) as f32 * 0.0001 - 2.0,
                    });
                }
            }
            let x: Array = (values.as_slice(), &[1, 3, n][..]).try_into()?;
            let x = mlx::ops::cast::astype(&x, Dtype::Bfloat16)?;
            let reference = mlx::ops::cast::astype(&x, Dtype::Float32)?.to_vec::<f32>()?;
            for k in [1, 16, 64] {
                let actual = candidates(&x, k, ().into())?.unwrap().to_vec::<u32>()?;
                for row in 0..3_usize {
                    let values = &reference[row * n as usize..(row + 1) * n as usize];
                    let mut order = (0..n as u32).collect::<Vec<_>>();
                    order.sort_by(|&a, &b| {
                        values[b as usize]
                            .total_cmp(&values[a as usize])
                            .then(a.cmp(&b))
                    });
                    assert_eq!(
                        &actual[row * k as usize..(row + 1) * k as usize],
                        &order[..k as usize],
                        "N={n} K={k} row={row}"
                    );
                }
            }
        }
        std::env::remove_var("IRONMLX_EXPERIMENTAL_DFLASH2_RADIX_TOPK");
        Ok(())
    }
}
