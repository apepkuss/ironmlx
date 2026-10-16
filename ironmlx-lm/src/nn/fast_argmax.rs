//! Row argmax over a wide last axis (the LM head's vocabulary) with many
//! threadgroups per row. MLX's arg-reduce gives each row one threadgroup, so a
//! few verify rows over 248K logits run on a few GPU cores. Two dispatches:
//! per-chunk maxima, then one simdgroup per row. Ties pick the smallest index,
//! as MLX does, so the result is identical for finite logits.

use std::sync::OnceLock;

use anyhow::ensure;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

/// Chunks per row of the first pass.
const CHUNKS: i32 = 32;

const PARTIAL: &str = r#"
  const uint row = threadgroup_position_in_grid.y;
  const uint chunk = threadgroup_position_in_grid.x;
  const uint lid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint per = (V + CHUNKS - 1) / CHUNKS;
  const uint start = chunk * per;
  const uint stop = min(start + per, uint(V));
  const device bfloat* x = (const device bfloat*)X + ulong(row) * V;
  float best = -INFINITY;
  uint best_i = 0xffffffffu;
  for (uint i = start + lid; i < stop; i += 256) {
    const float v = float(x[i]);
    if (v > best) { best = v; best_i = i; }
  }
  threadgroup float vals[8];
  threadgroup uint idxs[8];
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(best, off);
    const uint oi = simd_shuffle_down(best_i, off);
    if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = best_i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    for (uint s = 1; s < 8; ++s)
      if (vals[s] > best || (vals[s] == best && idxs[s] < best_i)) { best = vals[s]; best_i = idxs[s]; }
    PV[row * CHUNKS + chunk] = best;
    PI[row * CHUNKS + chunk] = best_i;
  }
"#;

const FINAL: &str = r#"
  const uint row = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  float best = lane < CHUNKS ? PV[row * CHUNKS + lane] : -INFINITY;
  uint best_i = lane < CHUNKS ? PI[row * CHUNKS + lane] : 0xffffffffu;
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(best, off);
    const uint oi = simd_shuffle_down(best_i, off);
    if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
  }
  if (lane == 0) OUT[row] = best_i;
"#;

fn kernels() -> (&'static MetalKernel, &'static MetalKernel) {
    static PARTIAL_K: OnceLock<MetalKernel> = OnceLock::new();
    static FINAL_K: OnceLock<MetalKernel> = OnceLock::new();
    let partial = PARTIAL_K.get_or_init(|| {
        MetalKernel::builder("ironmlx_fast_argmax_partial_v1")
            .inputs(&["X"])
            .outputs(&["PV", "PI"])
            .source(PARTIAL)
            .build()
            .expect("build fast argmax partial")
    });
    let fin = FINAL_K.get_or_init(|| {
        MetalKernel::builder("ironmlx_fast_argmax_final_v1")
            .inputs(&["PV", "PI"])
            .outputs(&["OUT"])
            .source(FINAL)
            .build()
            .expect("build fast argmax final")
    });
    (partial, fin)
}

/// `argmax(x, -1)` (uint32, last axis dropped) of bf16 `x` with a wide last
/// axis; other inputs return `None` for the caller's MLX path.
pub fn argmax_last_on(x: &Array, target: StreamOrDevice) -> Result<Option<Array>> {
    let shape = x.shape();
    let dims = shape.as_slice();
    if x.dtype() != Dtype::Bfloat16 || dims.is_empty() || *dims.last().unwrap() < 4096 {
        return Ok(None);
    }
    let v = *dims.last().unwrap();
    let rows: i32 = dims[..dims.len() - 1].iter().product();
    ensure!(rows > 0, "fast argmax: empty rows");
    let x = x.reshape_on(&[rows, v][..], target)?;
    let (partial, fin) = kernels();
    let mut parts = partial
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[
            Shape::from(&[rows * CHUNKS][..]),
            Shape::from(&[rows * CHUNKS][..]),
        ])
        .output_dtypes(&[Dtype::Float32, Dtype::Uint32])
        .grid(CHUNKS * 256, rows, 1)
        .threadgroup(256, 1, 1)
        .template_int("V", v)
        .template_int("CHUNKS", CHUNKS)
        .stream(target)
        .dispatch()?;
    let pv = parts.take_at(0)?;
    let pi = parts.take_at(0)?;
    let out = fin
        .dispatch_builder()
        .inputs(&[&pv, &pi])
        .output_shapes(&[Shape::from(&[rows][..])])
        .output_dtypes(&[Dtype::Uint32])
        .grid(rows * 32, 1, 1)
        .threadgroup(32, 1, 1)
        .template_int("CHUNKS", CHUNKS)
        .stream(target)
        .dispatch()?
        .take_at(0)?;
    Ok(Some(out.reshape_on(&dims[..dims.len() - 1], target)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn matches_mlx_argmax_with_ties() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        for (rows, v) in [(1, 248320), (8, 248320), (3, 5000)] {
            let mut values = Vec::with_capacity((rows * v) as usize);
            for r in 0..rows {
                for i in 0..v {
                    // Coarse values so many ties land on the maximum.
                    values.push(((i * 7919 + r * 104729) % 509) as f32 * 0.25);
                }
            }
            let x: Array = (values.as_slice(), &[1, rows, v][..]).try_into()?;
            let x = mlx::ops::cast::astype(&x, Dtype::Bfloat16)?;
            let want = mlx::ops::reduction::argmax(&x, -1, false)?;
            let want = mlx::ops::cast::astype(&want, Dtype::Uint32)?.to_vec::<u32>()?;
            let got = argmax_last_on(&x, StreamOrDevice::default())?
                .expect("fast path")
                .to_vec::<u32>()?;
            assert_eq!(got, want, "rows {rows} v {v}");
        }
        Ok(())
    }
}
