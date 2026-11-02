//! Pinned MLX-VLM Dk=128 prefill kernel. Its explicit butterfly reduction
//! matters for long Flash trajectories; decode keeps the shared Q1 kernel.
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;
fn kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = KERNEL.get() {
        return Ok(kernel);
    }
    let reference = include_str!("../../../vendor/mlx-vlm/mlx_vlm/models/qwen3_5/gated_delta.py");
    let source = reference
        .split_once("def _make_gated_delta_packed_kernel():")
        .and_then(|(_, x)| x.split_once("source = r\"\"\""))
        .and_then(|(_, x)| x.split_once("\"\"\"").map(|(body, _)| body))
        .context("pinned GDN prefill source missing")?;
    let built = MetalKernel::builder("ironmlx_flash_gdn_packed_btree")
        .inputs(&["q", "k", "v", "g", "beta", "state_in", "T"])
        .outputs(&["y", "state_out"])
        .source(source)
        .build()?;
    let _ = KERNEL.set(built);
    Ok(KERNEL.get().unwrap())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_on(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let qs = q.shape();
    let qd = qs.as_slice();
    let vs = v.shape();
    let vd = vs.as_slice();
    ensure!(
        qd.len() == 4
            && vd.len() == 4
            && qd[0] == vd[0]
            && qd[1] == vd[1]
            && qd[1] > 1
            && qd[3] == 128
            && vd[3] % 8 == 0
            && state.dtype() == Dtype::Float32
            && g.dtype() == Dtype::Float32,
        "invalid Flash GDN packed geometry"
    );
    let (b, t, hk, dk, hv, dv) = (qd[0], qd[1], qd[2], qd[3], vd[2], vd[3]);
    let length = Array::try_from((&[t][..], ()))?;
    let mut outputs = kernel()?
        .dispatch_builder()
        .inputs(&[q, k, v, g, beta, state, &length])
        .output_shapes(&[Shape::from(&[b, t, hv, dv][..]), state.shape()])
        .output_dtypes(&[q.dtype(), state.dtype()])
        .template_int("Dk", dk)
        .template_int("Dv", dv)
        .template_int("Hk", hk)
        .template_int("Hv", hv)
        .template_dtype("InT", q.dtype())
        .template_dtype("StT", state.dtype())
        .grid(32, dv / 8, b * hv)
        .threadgroup(32, 2, 1)
        .stream(target)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}
