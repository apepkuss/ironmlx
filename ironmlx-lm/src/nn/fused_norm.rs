//! Residual add + RMS norm in one dispatch: `h = x + a` (bf16) and
//! `rms_norm(h, w, eps)`, written as MLX's `Add` and `rms_single_row`
//! compute them (same reduction order and roundings), so both outputs are
//! bit-identical to the two-op path.

use std::sync::OnceLock;

use anyhow::ensure;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

static AB_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Diagnostic in-process A/B (benchmarks only): take the op-by-op paths of
/// every fused glue dispatch again.
#[doc(hidden)]
pub fn set_ab_off(off: bool) {
    AB_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
}

/// M5 profile setting `FUSED_GLUE`: the bit-identical single-dispatch forms
/// of the decoder glue (this residual add + norm, the MoE residual add in
/// the expert combine, and the GatedDelta preparation and gated norm of
/// `nn::gdn_prep`).
pub(crate) fn requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::FUSED_GLUE)
    }) && !AB_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

const SOURCE: &str = r#"
  constexpr int N_READS = 4;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  const uint gid = threadgroup_position_in_grid.x;
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const device bfloat* xr = (const device bfloat*)x + gid * size_t(D) + lid * N_READS;
  const device bfloat* ar = (const device bfloat*)a + gid * size_t(D) + lid * N_READS;
  device bfloat* hr = (device bfloat*)h + gid * size_t(D) + lid * N_READS;
  const device bfloat* wr = (const device bfloat*)w + lid * N_READS;
  float acc = 0;
  float thread_x[N_READS];
  for (int i = 0; i < N_READS; i++) {
    const bfloat hv = xr[i] + ar[i];
    hr[i] = hv;
    thread_x[i] = hv;
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / uint(D) + as_type<float>(int(EPSBITS)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* o = (device bfloat*)normed + gid * size_t(D) + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    o[i] = wr[i] * static_cast<bfloat>(thread_x[i] * local_inv_mean[0]);
  }
"#;

/// `(x + a, rms_norm(x + a, weight, eps))` for bf16 `x`, `a` `[.., D]` with
/// `D` a multiple of 128 up to 4096 (MLX's single-row RMS kernel range).
pub(crate) fn add_rms_norm_on(
    x: &Array,
    a: &Array,
    weight: &Array,
    eps: f32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let shape = x.shape();
    let dims = shape.as_slice();
    let d = *dims.last().unwrap_or(&0);
    ensure!(
        x.dtype() == Dtype::Bfloat16
            && a.dtype() == Dtype::Bfloat16
            && weight.dtype() == Dtype::Bfloat16
            && a.shape().as_slice() == dims
            && weight.size() == d as usize
            && d % 128 == 0
            && (128..=4096).contains(&d),
        "fused add + rms norm: unsupported shapes"
    );
    let rows = i32::try_from(x.size() / d as usize)?;
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if KERNEL.get().is_none() {
        let built = MetalKernel::builder("ironmlx_add_rms_norm_v1")
            .inputs(&["x", "a", "w"])
            .outputs(&["h", "normed"])
            .source(SOURCE)
            .build()?;
        let _ = KERNEL.set(built);
    }
    // MLX: one simdgroup per 128 inputs (4 reads per lane).
    let threads = d / 4;
    let mut out = KERNEL
        .get()
        .expect("built")
        .dispatch_builder()
        .inputs(&[x, a, weight])
        .output_shapes(&[Shape::from(dims), Shape::from(dims)])
        .output_dtypes(&[Dtype::Bfloat16, Dtype::Bfloat16])
        .grid(threads * rows, 1, 1)
        .threadgroup(threads, 1, 1)
        .template_int("D", d)
        .template_int("EPSBITS", eps.to_bits() as i32)
        .stream(target)
        .dispatch()?;
    Ok((out.take_at(0)?, out.take_at(0)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::cast::astype;
    use serial_test::serial;

    fn bits(a: &Array) -> Result<Vec<u32>> {
        Ok(astype(a, Dtype::Float32)?
            .to_vec::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    #[test]
    #[serial(mlx_metal)]
    fn matches_add_then_rms_norm_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let target = StreamOrDevice::default();
        let rand = |shape: &[i32], scale: f64, seed: u64| -> Result<Array> {
            let key = mlx::random::key(seed)?;
            let x = mlx::random::normal()
                .shape(shape)
                .scale(scale)
                .key(&key)
                .sample()?;
            Ok(astype(&x, Dtype::Bfloat16)?)
        };
        for (rows, d) in [(1, 2048), (8, 2048), (32, 2048), (3, 512)] {
            let x = rand(&[1, rows, d], 3.0, rows as u64)?;
            let a = rand(&[1, rows, d], 1.0, 100 + rows as u64)?;
            let w = rand(&[d], 0.5, 7)?;
            let (h, normed) = add_rms_norm_on(&x, &a, &w, 1e-6, target)?;
            let want_h = &x + &a;
            let want = mlx::fast::rms_norm_on(&want_h, Some(&w), 1e-6, target)?;
            assert_eq!(bits(&h)?, bits(&want_h)?, "h rows={rows} d={d}");
            assert_eq!(bits(&normed)?, bits(&want)?, "normed rows={rows} d={d}");
        }
        Ok(())
    }
}
