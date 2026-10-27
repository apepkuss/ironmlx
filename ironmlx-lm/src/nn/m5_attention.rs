//! Experimental B1 causal BF16 attention with position-stable key tiles.
//! Arithmetic differs from ordinary MLX; serial and speculative routes share it.
use crate::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

pub(crate) fn enabled() -> bool {
    super::m5_affine4::armed()
        && ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_LANE_ATTN)
}
pub(crate) fn group_tiles(tiles: i32) -> i32 {
    ironmlx_core::m5_profile::setting(ironmlx_core::m5_profile::settings::M5_ATTN_GROUP_TILES)
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
    let sga = (t * 6 + 15) / 16;
    attend_rows(q, k, v, scale, group_tiles(sga), 512, target)
}

/// [`attend`]'s lane kernels for any query row count: `sg` 16-row tiles per
/// threadgroup, keys in chunks of `ck` (one chunk when `ck >= L`, so the
/// merge only normalizes).
pub(crate) fn attend_rows(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    sg: i32,
    ck: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    use mlx::ops::shape::*;
    let t = q.shape().as_slice()[2];
    let l = k.shape().as_slice()[2];
    let r = t * 6;
    let rp = (r + 15) / 16 * 16;
    let sga = rp / 16;
    let nch = (l + ck - 1) / ck;
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
    let sg = sg.min(sga).max(1);
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
        .template_int("CK", ck)
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

    /// Diagnostic: the lane kernels at prefill shapes (2048 query rows) against
    /// MLX's SDPA, time and max error against an f32 reference of head 0.
    /// Env `LANE_PREFILL` = "sg,ck;..." (default "6,1048576;8,1048576").
    #[test]
    #[ignore]
    #[serial_test::serial(mlx_metal)]
    fn lane_attention_prefill_throughput() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        use mlx::ops::cast::astype;
        let target = StreamOrDevice::default();
        let rows = 2048;
        let spec = std::env::var("LANE_PREFILL").unwrap_or_else(|_| "6,1048576;8,1048576".into());
        let configs: Vec<(i32, i32)> = spec
            .split(';')
            .map(|c| {
                let v: Vec<i32> = c.split(',').map(|x| x.trim().parse().unwrap()).collect();
                (v[0], v[1])
            })
            .collect();
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
            let pairs =
                rows as f64 * (keys - rows) as f64 + rows as f64 * (rows as f64 + 1.0) / 2.0;
            let flops = 4.0 * 24.0 * 256.0 * pairs;
            let time = |f: &dyn Fn() -> Result<Array>| -> Result<(f64, Array)> {
                let mut runs = Vec::new();
                let mut out = None;
                for _ in 0..5 {
                    let started = std::time::Instant::now();
                    let o = f()?;
                    mlx::transforms::eval(&[&o])?;
                    runs.push(started.elapsed().as_secs_f64());
                    out = Some(o);
                }
                runs.sort_by(f64::total_cmp);
                Ok((runs[runs.len() / 2], out.unwrap()))
            };
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
            let err = |o: &Array| -> Result<f32> {
                let o0 = astype(
                    &mlx::ops::indexing::slice(o, &[0, 0, 0, 0][..], &[1, 1, rows, 256][..])?,
                    Dtype::Float32,
                )?;
                Ok(mlx::ops::reduction::max(
                    &mlx::ops::unary::abs(&(&o0 - &reference))?,
                    mlx::ops::reduction::All,
                    false,
                )?
                .item::<f32>()?)
            };
            let (tm, om) = time(&|| {
                Ok(mlx::fast::scaled_dot_product_attention(
                    &q, &k, &v, 0.0625, "causal", None, None,
                )?)
            })?;
            let mut line = format!(
                "LANE keys={keys}: mlx {:.2} ms {:.1} TF err {:.2e}",
                tm * 1e3,
                flops / tm / 1e12,
                err(&om)?
            );
            for &(sg, ck) in &configs {
                let (tl, ol) = time(&|| attend_rows(&q, &k, &v, 0.0625, sg, ck, target))?;
                line += &format!(
                    " | lane sg{sg} ck{ck} {:.2} ms {:.1} TF err {:.2e}",
                    tl * 1e3,
                    flops / tl / 1e12,
                    err(&ol)?
                );
            }
            eprintln!("{line}");
        }
        Ok(())
    }
}
