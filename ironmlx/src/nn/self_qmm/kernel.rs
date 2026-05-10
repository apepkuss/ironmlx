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

/// Stage 9 fix kernel: ports llama.cpp's `kernel_mul_mm_q4_K_f32` structure
/// (NR1=32 batch rows, NR0=64 weight rows, NK=32 inner K). The 8×8 block
/// shmem layout, register-tile dequant staging, vec8 activation load, and
/// SG-MMA pattern (4 SGs in 2×2, mc[8] = 2 batch-frag × 4 wcol-frag) all
/// hardcode these dimensions — passing a different tile would write past
/// the 2048-half `sa` / 1024-half `sb` shmem buffers and produce wrong
/// SG-MMA addressing.
const KERNEL_BM: i32 = 32; // batch rows per TG (= llama.cpp NR1)
const KERNEL_BN: i32 = 64; // weight rows per TG (= llama.cpp NR0)
const KERNEL_BK: i32 = 32; // inner-K block (= llama.cpp NK)
const KERNEL_THREADS_PER_TG: i32 = 128; // 4 SGs × 32 lanes

fn validate_tile(bm: i32, bn: i32, bk: i32) {
    assert_eq!(
        bm, KERNEL_BM,
        "self_qmm stage 9 fix: kernel hardcodes BM={KERNEL_BM} (got {bm})"
    );
    assert_eq!(
        bn, KERNEL_BN,
        "self_qmm stage 9 fix: kernel hardcodes BN={KERNEL_BN} (got {bn})"
    );
    assert_eq!(
        bk, KERNEL_BK,
        "self_qmm stage 9 fix: kernel hardcodes BK={KERNEL_BK} (got {bk})"
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
    // threadgroups of `KERNEL_THREADS_PER_TG` threads each, set the x dim
    // of grid to `n_tiles_x * threads_per_tg` and the y dim to `n_tiles_y`.
    let threads_per_tg = KERNEL_THREADS_PER_TG;
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
