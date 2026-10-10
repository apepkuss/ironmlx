//! Expert-grouped gather qmv for multi-token MoE verification.
//!
//! MLX's sorted `gather_qmm` evaluates one qmv per (token, expert) slot, so an
//! expert chosen by several verify tokens has its weights streamed and
//! unpacked once per token. This kernel lets one threadgroup own every sorted
//! row of an expert, unpack each weight block once into exact integer terms
//! and reuse them for all rows, while each row evaluates the exact MLX
//! `qmv_fast_impl`/`qdot` expression sequence. The route is armed only by a target
//! scope and only for shapes MLX itself would run through `gather_qmv_fast`.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::{Array, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

/// Largest number of rows one expert can receive: one per verify token,
/// because a token's top-k experts are distinct.
pub(crate) const MAX_RUN: i32 = 16;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

static DISPATCHES: AtomicUsize = AtomicUsize::new(0);

/// Number of grouped kernel dispatches in this process (diagnostics and
/// qualification tests confirm the route actually ran).
pub fn dispatch_count() -> usize {
    DISPATCHES.load(Ordering::Relaxed)
}

pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn scope() -> Scope {
    DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Scope
}

pub(crate) fn is_armed() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

/// Shapes for which MLX dispatches `affine_gather_qmv_fast` and this kernel
/// reproduces its per-row arithmetic.
pub(crate) fn supported(bits: i32, group_size: i32, affine: bool, k: i32, n: i32) -> bool {
    affine && group_size == 64 && matches!(bits, 4 | 5 | 6 | 8) && k % 512 == 0 && n % 8 == 0
}

fn kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::builder("ironmlx_moe_grouped_gather_qmv_v2")
            .inputs(&["x", "w", "scales", "biases", "experts"])
            .outputs(&["y"])
            .header(include_str!("moe_grouped_qmv_header.metal"))
            .source(include_str!("moe_grouped_qmv.metal"))
            .build()
            .expect("build expert-grouped gather qmv kernel")
    })
}

/// `x`: `[S, 1, K]` or `[S, K]` rows sorted by expert; `experts`: `[S]`
/// non-decreasing expert ids; returns `[S, 1, N]` like MLX sorted
/// `gather_qmm`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_on(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    experts: &Array,
    bits: i32,
    group_size: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let x_shape = x.shape();
    let x_dims = x_shape.as_slice();
    let (rows, k) = match x_dims {
        [rows, one, k] if *one == 1 => (*rows, *k),
        [rows, k] => (*rows, *k),
        other => {
            return Err(anyhow!(
                "grouped gather qmv expects [S,1,K] or [S,K], got {other:?}"
            ))
        }
    };
    let w_shape = weight.shape();
    let w_dims = w_shape.as_slice();
    anyhow::ensure!(
        w_dims.len() == 3 && experts.shape().as_slice() == [rows],
        "grouped gather qmv expects weight [E,N,Kw] and experts [S], got {w_dims:?} and {:?}",
        experts.shape().as_slice()
    );
    let n = w_dims[1];
    anyhow::ensure!(
        supported(bits, group_size, true, k, n),
        "grouped gather qmv does not support bits={bits} group={group_size} K={k} N={n}"
    );
    let x = x.reshape_on((rows, k), target)?;
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut outputs = kernel()
        .dispatch_builder()
        .inputs(&[&x, weight, scales, biases, experts])
        .output_shapes(&[Shape::from(&[rows, n][..])])
        .output_dtypes(&[x.dtype()])
        .grid(32 * rows, 2 * (n / 8), 1)
        .threadgroup(32, 2, 1)
        .template_dtype("T", x.dtype())
        .template_int("BITS", bits)
        .template_int("GS", group_size)
        .template_int("K", k)
        .template_int("N", n)
        .template_int("MAXRUN", MAX_RUN)
        .stream(target)
        .dispatch()?;
    Ok(outputs.take_at(0)?.reshape_on((rows, 1, n), target)?)
}
