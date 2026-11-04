//! MLX-VLM's pinned MIT-licensed QSA prefill reduction. Keeping the original
//! source also preserves sentinel placement and online-softmax reduction
//! geometry for prefixes shorter than the sparse token budget.
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

fn kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = KERNEL.get() {
        return Ok(kernel);
    }
    let reference = include_str!("../../../vendor/mlx-vlm/mlx_vlm/models/qwen4_exp/qsa_kernel.py");
    let body = reference
        .split_once("_QSA_SPARSE_ATTENTION_SOURCE = r\"\"\"")
        .and_then(|(_, body)| body.split_once("\"\"\"").map(|(body, _)| body))
        .context("pinned QSA source body missing")?;
    let kernel = MetalKernel::builder("ironmlx_flash_qsa_prefill")
        .inputs(&[
            "queries",
            "keys",
            "values",
            "block_indices",
            "query_ends",
            "scale",
            "k_size",
        ])
        .outputs(&["out"])
        .header("#include <metal_simdgroup>\nusing namespace metal;\n")
        .source(body)
        .build()?;
    let _ = KERNEL.set(kernel);
    Ok(KERNEL.get().expect("initialized QSA kernel"))
}

// Keep the native kernel's tensor/scalar inputs explicit, as in the GDN wrapper.
#[allow(clippy::too_many_arguments)]
pub fn prefill_on(
    q: &Array,
    k: &Array,
    v: &Array,
    selected: &Array,
    query_ends: &Array,
    scale: f32,
    block_size: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = q.shape();
    let dims = shape.as_slice();
    let kd = k.shape();
    let kd = kd.as_slice();
    let bd = selected.shape();
    let bd = bd.as_slice();
    ensure!(
        dims.len() == 4
            && kd.len() == 4
            && bd.len() == 3
            && dims[0] > 0
            && dims[1] > 0
            && dims[2] > 0
            && dims[3] > 0
            && dims[3] % 32 == 0
            && kd[0] == dims[0]
            && kd[1] > 0
            && dims[1] % kd[1] == 0
            && kd[3] == dims[3]
            && k.shape() == v.shape()
            && q.dtype() == k.dtype()
            && k.dtype() == v.dtype()
            && matches!(q.dtype(), Dtype::Bfloat16 | Dtype::Float16)
            && bd[0] == dims[0]
            && bd[1] == dims[2]
            && bd[2] > 0
            && block_size > 0
            && query_ends.shape().as_slice() == [dims[0], dims[2]],
        "unsupported QSA prefill geometry"
    );
    let selected_length = bd[2]
        .checked_mul(block_size)
        .context("QSA selection size overflow")?;
    let grid_y = dims[0]
        .checked_mul(dims[1])
        .and_then(|n| n.checked_mul(dims[2]))
        .context("QSA dispatch overflow")?;
    let contiguous = |a: &Array| mlx::ops::shape::contiguous_on(a, false, target);
    let q = contiguous(q)?;
    let k = contiguous(k)?;
    let v = contiguous(v)?;
    let selected = mlx::ops::sort::sort_on(&selected.astype_on(Dtype::Int32, target)?, -1, target)?;
    let selected = contiguous(&selected)?;
    let ends = contiguous(&query_ends.astype_on(Dtype::Int32, target)?)?;
    let scale_array = Array::try_from((&[scale][..], [1]))?;
    let key_length = Array::try_from((&[kd[2]][..], [1]))?;
    let mut output = kernel()?
        .dispatch_builder()
        .inputs(&[&q, &k, &v, &selected, &ends, &scale_array, &key_length])
        .output_shapes(&[Shape::from(dims)])
        .output_dtypes(&[q.dtype()])
        .template_dtype("T", q.dtype())
        .template_int("D_SIZE", dims[3])
        .template_int("Q_LEN", dims[2])
        .template_int("NUM_Q_HEADS", dims[1])
        .template_int("NUM_KV_HEADS", kd[1])
        .template_int("GQA_FACTOR", dims[1] / kd[1])
        .template_int("BLOCK_SIZE", block_size)
        .template_int("TOPK_BLOCKS", bd[2])
        .template_int("SELECTED_LENGTH", selected_length)
        .grid(1024, grid_y, 1)
        .threadgroup(1024, 1, 1)
        .stream(target)
        .dispatch()?;
    Ok(output.take_at(0)?)
}
