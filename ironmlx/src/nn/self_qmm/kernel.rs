//! `mx::fast::metal_kernel` builder + dispatch for `self_qmm_t`.
//!
//! Stage 9 task 3: tile dimensions (BM/BN/BK) parameterized via MLX
//! `template_int`. Each unique `(M, N, K, BM, BN, BK)` tuple triggers a
//! distinct C++ template specialization in MLX upstream, which dedupes /
//! caches PSOs internally — no Rust-side per-tile cache needed.

use std::sync::OnceLock;

use mlx::{Array, MetalKernel, Shape};

use crate::Result;

/// Metal source for the qmm_t kernel. Read once at compile time.
const QMM_T_SOURCE: &str = include_str!("metal/qmm_t.metal.in");

/// Lazy `MetalKernel` for `ironmlx_self_qmm_t`. Built on first dispatch.
/// One handle serves all `(BM, BN, BK)` variants — MLX upstream auto-
/// specializes per template-arg tuple at dispatch time.
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

/// Validate tile parameters against kernel invariants. Centralized so the
/// dispatch path stays clean.
fn validate_tile(bm: i32, bn: i32, bk: i32) {
    assert!(
        bm > 0 && bn > 0 && bk > 0,
        "self_qmm: tile dims must be positive (BM={bm} BN={bn} BK={bk})"
    );
    assert_eq!(
        (bm * bn) % 16,
        0,
        "self_qmm: BM*BN must be divisible by 16 (4x4 micro-tile per thread); got BM={bm} BN={bn}"
    );
    let threads_per_tg = (bm * bn) / 16;
    assert!(
        threads_per_tg <= 1024,
        "self_qmm: threads/TG = {threads_per_tg} exceeds Metal 1024 limit (BM={bm} BN={bn})"
    );
    assert_eq!(
        (bm * bk) % threads_per_tg,
        0,
        "self_qmm: (BM*BK) % threads_per_tg != 0 (BM={bm} BK={bk} threads/TG={threads_per_tg})"
    );
    assert_eq!(
        (bn * bk) % threads_per_tg,
        0,
        "self_qmm: (BN*BK) % threads_per_tg != 0 (BN={bn} BK={bk} threads/TG={threads_per_tg})"
    );
    assert_eq!(
        bn % 4,
        0,
        "self_qmm: BN must be divisible by 4 (col_in_tile stride); got BN={bn}"
    );
}

/// Dispatch the self-quant matmul kernel with the given tile dimensions.
/// See [`super::qmm_t_on`] for the public entry point + parameter contract.
pub fn dispatch_qmm_t(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bm: i32,
    bn: i32,
    bk: i32,
) -> Result<Array> {
    validate_tile(bm, bn, bk);

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
    // is then clamped to grid per axis. To get `n_tiles_x` x `n_tiles_y`
    // threadgroups of `threads_per_tg` threads each, set the x dim of grid
    // to `n_tiles_x * threads_per_tg` and the y dim to `n_tiles_y`.
    let threads_per_tg = (bm * bn) / 16;
    let n_tiles_x = (n + bn - 1) / bn;
    let n_tiles_y = (m + bm - 1) / bm;
    let grid_x = n_tiles_x * threads_per_tg;
    let grid_y = n_tiles_y;

    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, w, scales, biases])
        .output_shapes(&[out_shape])
        .output_dtypes(&[x.dtype()])
        .grid(grid_x, grid_y, 1)
        .threadgroup(threads_per_tg, 1, 1)
        .template_int("M", m)
        .template_int("N", n)
        .template_int("K", k)
        .template_int("BM", bm)
        .template_int("BN", bn)
        .template_int("BK", bk)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}
