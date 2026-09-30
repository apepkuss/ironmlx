//! Experimental B1 causal BF16 attention with position-stable key tiles.
//! Arithmetic differs from ordinary MLX; serial and speculative routes share it.
use crate::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

pub(crate) fn enabled() -> bool {
    super::m5_affine4::armed()
        && std::env::var("IRONMLX_EXPERIMENTAL_M5_LANE_ATTN").as_deref() == Ok("1")
}
pub(crate) fn group_tiles(tiles: i32) -> i32 {
    std::env::var("IRONMLX_EXPERIMENTAL_M5_ATTN_GROUP_TILES")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .filter(|n| (1..=16).contains(n))
        .unwrap_or(tiles.min(16))
}
pub(crate) fn compatible(q: &Array, k: &Array, v: &Array) -> bool {
    let qs = q.shape();
    let ks = k.shape();
    let q = qs.as_slice();
    let k = ks.as_slice();
    q.len() == 4
        && k.len() == 4
        && q[0] == 1
        && q[1] == 24
        && q[3] == 256
        && (1..=128).contains(&q[2])
        && k[0] == 1
        && k[1] == 4
        && k[3] == 256
        && k[2] >= q[2]
        && ks == v.shape()
}
pub(crate) fn attend(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    use mlx::ops::shape::*;
    anyhow::ensure!(
        compatible(q, k, v) && [q, k, v].iter().all(|a| a.dtype() == Dtype::Bfloat16),
        "incompatible M5 attention"
    );
    let t = q.shape().as_slice()[2];
    let l = k.shape().as_slice()[2];
    if let Some(plan) = super::dflash_tree::current() {
        return super::m5_tree_attention::attend(
            q,
            k,
            v,
            scale,
            plan.lane_attention(l - t)?,
            target,
        );
    }
    let r = t * 6;
    let rp = (r + 15) / 16 * 16;
    let sga = rp / 16;
    let sg = group_tiles(sga);
    let nch = (l + 511) / 512;
    let qp = reshape_on(q, &[4, 6, t, 256][..], target)?;
    let qp = transpose_axes_on(&qp, &[0, 2, 1, 3][..], target)?;
    let mut qp = reshape_on(&qp, &[4, r, 256][..], target)?;
    if rp > r {
        let z = Array::zeros_on(&[4, rp - r, 256][..], Dtype::Bfloat16, target)?;
        qp = concatenate_on(&[&qp, &z], 1, target)?;
    }
    let qp = contiguous_on(&qp, false, target)?;
    let dims: Array = (&[l, nch, t, 1, sga][..], &[5][..]).try_into()?;
    let scale: Array = (&[scale][..], &[1][..]).try_into()?;
    static PARTIAL: OnceLock<MetalKernel> = OnceLock::new();
    static MERGE: OnceLock<MetalKernel> = OnceLock::new();
    if PARTIAL.get().is_none() {
        let kernel=MetalKernel::builder("ironmlx_m5_lane_attention_partial_v1")
            .inputs(&["Qp","K","V","scale","dims"]).outputs(&["PO","PM","PL"])
            .header("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;")
            .source(include_str!("m5_attention_partial.metal")).ensure_row_contiguous(false).build()?;
        let _ = PARTIAL.set(kernel);
    }
    if MERGE.get().is_none() {
        let kernel = MetalKernel::builder("ironmlx_m5_lane_attention_merge_v1")
            .inputs(&["PO", "PM", "PL", "dims"])
            .outputs(&["OUT"])
            .source(include_str!("m5_attention_merge.metal"))
            .build()?;
        let _ = MERGE.set(kernel);
    }
    let mut parts = PARTIAL
        .get()
        .unwrap()
        .dispatch_builder()
        .inputs(&[&qp, k, v, &scale, &dims])
        .template_int("G", 6)
        .template_int("D", 256)
        .template_int("SG", sg)
        .template_int("CK", 512)
        .template_int("TK", 64)
        .output_shapes(&[
            Shape::from(&[4, nch, rp, 256][..]),
            Shape::from(&[4, nch, rp][..]),
            Shape::from(&[4, nch, rp][..]),
        ])
        .output_dtypes(&[Dtype::Float32, Dtype::Float32, Dtype::Float32])
        .grid(4 * 32 * sg, nch, (sga + sg - 1) / sg)
        .threadgroup(32 * sg, 1, 1)
        .stream(target)
        .dispatch()?;
    let po = parts.take_at(0)?;
    let pm = parts.take_at(0)?;
    let pl = parts.take_at(0)?;
    Ok(MERGE
        .get()
        .unwrap()
        .dispatch_builder()
        .inputs(&[&po, &pm, &pl, &dims])
        .template_int("G", 6)
        .template_int("D", 256)
        .output_shapes(&[Shape::from(&[1, 24, t, 256][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid(4 * 32, r, 1)
        .threadgroup(32, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn causal_tiles_match_reference_and_serial() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !mlx::metal::architecture()?.starts_with("applegpu_g17") {
            return Ok(());
        }
        let target = StreamOrDevice::default();
        use mlx::ops::{cast::astype, indexing::slice_strided_on};
        for prefix in [0, 63, 511, 512, 1024, 2048] {
            let t = 8;
            let l = prefix + t;
            let make = |h: i32, n: i32, salt: i32| -> Result<Array> {
                let data = (0..h * n * 256)
                    .map(|i| (((i * 13 + salt) % 113) as f32 - 56.) * 0.009)
                    .collect::<Vec<_>>();
                Ok(astype(
                    &Array::try_from((data.as_slice(), &[1, h, n, 256][..]))?,
                    Dtype::Bfloat16,
                )?)
            };
            let q = make(24, t, 3)?;
            let k = make(4, l, 11)?;
            let v = make(4, l, 17)?;
            let wide = attend(&q, &k, &v, 0.0625, target)?;
            let reference = mlx::fast::scaled_dot_product_attention_on(
                &q, &k, &v, 0.0625, "causal", None, None, target,
            )?;
            let wide_vec = astype(&wide, Dtype::Float32)?.to_vec::<f32>()?;
            let reference = astype(&reference, Dtype::Float32)?.to_vec::<f32>()?;
            let max = wide_vec
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("prefix={prefix} ordinary_mlx_max_abs={max}");
            assert!(max < 0.004, "reference mismatch {max}");
            for row in 0..t {
                let slice = |a: &Array, h: i32, start: i32, end: i32| {
                    slice_strided_on(
                        a,
                        &[0, 0, start, 0][..],
                        &[1, h, end, 256][..],
                        &[1, 1, 1, 1][..],
                        target,
                    )
                };
                let qr = slice(&q, 24, row, row + 1)?;
                let kr = slice(&k, 4, 0, prefix + row + 1)?;
                let vr = slice(&v, 4, 0, prefix + row + 1)?;
                let one = astype(&attend(&qr, &kr, &vr, 0.0625, target)?, Dtype::Float32)?
                    .to_vec::<f32>()?;
                let wr =
                    astype(&slice(&wide, 24, row, row + 1)?, Dtype::Float32)?.to_vec::<f32>()?;
                assert_eq!(one, wr, "prefix={prefix} row={row}");
            }
        }
        Ok(())
    }
}
