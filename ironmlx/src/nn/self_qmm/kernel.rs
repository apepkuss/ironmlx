//! `mx::fast::metal_kernel` builder + dispatch for `self_qmm_t`.
//!
//! Stage 9 task 2: hardcoded BM=64, BN=64, BK=32 single kernel variant.
//! Task 3 generalizes via function constants / multi-tile dispatch.

use std::sync::OnceLock;

use mlx::{Array, MetalKernel, Shape};

use crate::Result;

/// Metal source for the qmm_t kernel. Read once at compile time.
const QMM_T_SOURCE: &str = include_str!("metal/qmm_t.metal.in");

/// Lazy `MetalKernel` for `ironmlx_self_qmm_t`. Built on first dispatch.
fn cached_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(k) = CELL.get() {
        return Ok(k);
    }
    let k = MetalKernel::builder("ironmlx_self_qmm_t")
        .inputs(&["x", "w", "scales", "biases"])
        .outputs(&["out"])
        .source(QMM_T_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| k))
}

/// Tile constants — must match `metal/qmm_t.metal.in`.
const BM: i32 = 64;
const BN: i32 = 64;
/// Threadgroup is 16x16 = 256 threads, each owning a 4x4 micro-tile.
const TG_THREADS: i32 = 256;

/// Dispatch the self-quant matmul kernel. See [`super::qmm_t_on`] for the
/// public entry point + parameter contract.
pub fn dispatch_qmm_t(x: &Array, w: &Array, scales: &Array, biases: &Array) -> Result<Array> {
    // x shape: [..., K]; flatten leading dims into M for kernel addressing.
    let x_shape = x.shape();
    let x_dims = x_shape.as_slice();
    assert!(
        !x_dims.is_empty(),
        "self_qmm: x must have at least 1 dim (got 0)"
    );
    let k = *x_dims.last().expect("checked non-empty");
    let m: i32 = x_dims[..x_dims.len() - 1].iter().product();

    // w shape: [N, K/8] — N is leading dim.
    let w_shape = w.shape();
    let w_dims = w_shape.as_slice();
    assert_eq!(
        w_dims.len(),
        2,
        "self_qmm: w must be rank-2 [N, K/8] (got rank {})",
        w_dims.len()
    );
    let n = w_dims[0];

    // Output shape: [..x.shape[:-1], N]
    let mut out_dims: Vec<i32> = x_dims[..x_dims.len() - 1].to_vec();
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let kernel = cached_kernel()?;

    // MLX `metal_kernel` uses `dispatch_threads` semantics: `grid` is the
    // *total thread count* (not a threadgroup count). The threadgroup size
    // is then clamped to grid per axis. To get `n_tiles_x` × `n_tiles_y`
    // threadgroups of `TG_THREADS` threads each, set the x dim of grid to
    // `n_tiles_x * TG_THREADS` and the y dim to `n_tiles_y`. Inside the
    // shader, `threadgroup_position_in_grid.{x,y}` then yields the tile
    // coordinates and `thread_index_in_threadgroup` yields the 0..255 lane.
    let n_tiles_x = (n + BN - 1) / BN;
    let n_tiles_y = (m + BM - 1) / BM;
    let grid_x = n_tiles_x * TG_THREADS;
    let grid_y = n_tiles_y;

    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, w, scales, biases])
        .output_shapes(&[out_shape])
        .output_dtypes(&[x.dtype()])
        .grid(grid_x, grid_y, 1)
        .threadgroup(TG_THREADS, 1, 1)
        .template_int("M", m)
        .template_int("N", n)
        .template_int("K", k)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}
