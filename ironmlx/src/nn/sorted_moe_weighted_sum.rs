//! Direct weighted reduction for expert outputs kept in sorted-route order.
//!
//! Sorted MoE dispatches group routed rows by expert so MLX can use its fast
//! `gather_qmm_rhs` path. The generic combine restores `[tokens, top_k, hidden]`
//! order before multiplying by router scores and reducing. This kernel reads
//! the sorted rows through the inverse permutation and writes `[tokens, hidden]`
//! directly, eliminating the scatter and expanded intermediate.
//!
//! The kernel topology is informed by oMLX v0.5.1's
//! `qwen35_moe_weighted_sum`; dispatch, dtype, integration, and gating are
//! implemented for ironmlx's MLX Rust execution path.

use std::sync::OnceLock;

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

const THREADS: i32 = 256;
/// Below this point the extra custom-kernel dispatch does not reliably repay
/// its fixed cost. Qwen3.5/3.6 layer-1 A/B measurements establish the crossover
/// at the 1024-token prefill cell; this also matches oMLX's production gate.
pub(crate) const MIN_TOKENS: i32 = 1024;

pub(crate) fn should_use(x_sorted: &Array, inverse_order: &Array, scores: &Array) -> bool {
    scores.shape().as_slice().first().copied().unwrap_or(0) >= MIN_TOKENS
        && supports(x_sorted, inverse_order, scores)
}

fn supports(x_sorted: &Array, inverse_order: &Array, scores: &Array) -> bool {
    let x_shape = x_sorted.shape();
    let x_dims = x_shape.as_slice();
    let order_shape = inverse_order.shape();
    let score_shape = scores.shape();
    let score_dims = score_shape.as_slice();

    x_dims.len() == 3
        && x_dims[1] == 1
        && x_dims[0] > 0
        && x_dims[2] > 0
        && matches!(x_sorted.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        && order_shape.as_slice().len() == 1
        && inverse_order.dtype() == Dtype::Uint32
        && score_dims.len() == 2
        && score_dims[0] > 0
        && matches!(score_dims[1], 4 | 6 | 8)
        && matches!(
            scores.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        )
        && x_dims[0] == score_dims[0] * score_dims[1]
        && order_shape.as_slice()[0] == score_dims[0] * score_dims[1]
}

pub(crate) fn apply_on(
    x_sorted: &Array,
    inverse_order: &Array,
    scores: &Array,
    cast_output_to_expert_dtype: bool,
    target: StreamOrDevice,
) -> Result<Array> {
    if !supports(x_sorted, inverse_order, scores) {
        return Err(anyhow!(
            "sorted MoE weighted-sum unsupported inputs: x={} {}, inverse_order={} {}, scores={} {}",
            x_sorted.shape(),
            x_sorted.dtype(),
            inverse_order.shape(),
            inverse_order.dtype(),
            scores.shape(),
            scores.dtype()
        ));
    }

    let x_shape = x_sorted.shape();
    let x_dims = x_shape.as_slice();
    let score_shape = scores.shape();
    let score_dims = score_shape.as_slice();
    let tokens = score_dims[0];
    let top_k = score_dims[1];
    let hidden = x_dims[2];
    let output_dtype = if cast_output_to_expert_dtype {
        x_sorted.dtype()
    } else {
        weighted_output_dtype(x_sorted.dtype(), scores.dtype())?
    };
    let grid_x = tokens
        .checked_mul(THREADS)
        .ok_or_else(|| anyhow!("sorted MoE weighted-sum grid overflow: {tokens} * {THREADS}"))?;

    let mut outputs = kernel()?
        .dispatch_builder()
        .inputs(&[x_sorted, inverse_order, scores])
        .output_shapes(&[Shape::from([tokens, hidden])])
        .output_dtypes(&[output_dtype])
        .grid(grid_x, 1, 1)
        .threadgroup(THREADS, 1, 1)
        .stream(target)
        .template_int("TOKENS", tokens)
        .template_int("TOP_K", top_k)
        .template_int("HIDDEN", hidden)
        .template_int("THREADS", THREADS)
        .dispatch()
        .context("dispatch sorted MoE weighted-sum kernel")?;
    outputs
        .take_at(0)
        .context("take sorted MoE weighted-sum output")
}

fn weighted_output_dtype(expert_dtype: Dtype, score_dtype: Dtype) -> Result<Dtype> {
    match (expert_dtype, score_dtype) {
        (Dtype::Float16, Dtype::Float16) => Ok(Dtype::Float16),
        (Dtype::Bfloat16, Dtype::Bfloat16) => Ok(Dtype::Bfloat16),
        (Dtype::Float16 | Dtype::Bfloat16, Dtype::Float32)
        | (Dtype::Float16, Dtype::Bfloat16)
        | (Dtype::Bfloat16, Dtype::Float16) => Ok(Dtype::Float32),
        _ => Err(anyhow!(
            "unsupported sorted MoE weighted-sum dtype promotion: {expert_dtype} * {score_dtype}"
        )),
    }
}

fn kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        uint token = threadgroup_position_in_grid.x;
        uint lid = thread_index_in_threadgroup;
        if (token >= TOKENS) {
            return;
        }

        threadgroup uint rows[TOP_K];
        threadgroup float route_scores[TOP_K];
        int route_base = int(token) * TOP_K;
        if (lid < TOP_K) {
            rows[lid] = inverse_order[route_base + int(lid)];
            route_scores[lid] = float(scores[route_base + int(lid)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int d = int(lid); d < HIDDEN; d += THREADS) {
            float acc = 0.0f;
            for (int k = 0; k < TOP_K; ++k) {
                acc += float(x_sorted[int(rows[k]) * HIDDEN + d]) * route_scores[k];
            }
            out[int(token) * HIDDEN + d] = static_cast<__typeof__(*out)>(acc);
        }
    "#;

    let built = MetalKernel::builder("ironmlx_sorted_moe_weighted_sum")
        .inputs(&["x_sorted", "inverse_order", "scores"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()
        .context("build sorted MoE weighted-sum kernel")?;
    Ok(CELL.get_or_init(|| built))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::indexing::take_on;
    use serial_test::serial;

    fn reference(
        x_sorted: &Array,
        inverse_order: &Array,
        scores: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let score_shape = scores.shape();
        let score_dims = score_shape.as_slice();
        let tokens = score_dims[0];
        let top_k = score_dims[1];
        let hidden = x_sorted.shape().as_slice()[2];
        let x_2d = mlx::ops::shape::reshape(x_sorted, [tokens * top_k, hidden])?;
        let restored = take_on(&x_2d, inverse_order, 0, target)?;
        let restored = mlx::ops::shape::reshape(&restored, [tokens, top_k, hidden])?;
        let weights = mlx::ops::shape::expand_dims_on(scores, -1, target)?;
        Ok(mlx::ops::sum_on(
            &(&restored * &weights),
            -2,
            false,
            target,
        )?)
    }

    #[test]
    #[serial(mlx_metal)]
    fn matches_scatter_then_reduce_for_top8_bfloat16() -> Result<()> {
        let tokens = 3_i32;
        let top_k = 8_i32;
        let hidden = 17_i32;
        let routes = tokens * top_k;
        let values: Vec<f32> = (0..routes * hidden)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 32.0)
            .collect();
        let x_f32: Array = (values.as_slice(), [routes, 1, hidden]).try_into()?;
        let x = mlx::ops::cast::astype_on(&x_f32, Dtype::Bfloat16, ())?;
        let inverse: Vec<u32> = (0..routes as u32).rev().collect();
        let inverse: Array = (inverse.as_slice(), [routes]).try_into()?;
        let score_values: Vec<f32> = (0..routes)
            .map(|i| ((i % top_k) + 1) as f32 / 36.0)
            .collect();
        let scores_f32: Array = (score_values.as_slice(), [tokens, top_k]).try_into()?;
        let scores = mlx::ops::cast::astype_on(&scores_f32, Dtype::Bfloat16, ())?;

        let got = apply_on(&x, &inverse, &scores, false, ().into())?;
        let want = reference(&x, &inverse, &scores, ().into())?;
        mlx::transforms::eval(&[&got, &want])?;

        assert_eq!(got.dtype(), want.dtype());
        let got = got.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let want = want.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 0.015625, "max_abs={max_abs}");
        Ok(())
    }

    #[test]
    #[serial(mlx_metal)]
    fn preserves_float32_promotion_for_float32_scores() -> Result<()> {
        let x: Array = (&[1.0_f32; 48][..], [6, 1, 8]).try_into()?;
        let x = mlx::ops::cast::astype_on(&x, Dtype::Float16, ())?;
        let inverse: Array = (&[0_u32, 1, 2, 3, 4, 5][..], [6]).try_into()?;
        let scores: Array = (&[0.1_f32, 0.2, 0.3, 0.15, 0.1, 0.15][..], [1, 6]).try_into()?;

        let out = apply_on(&x, &inverse, &scores, false, ().into())?;
        mlx::transforms::eval(&[&out])?;
        assert_eq!(out.dtype(), Dtype::Float32);
        for value in out.to_vec::<f32>()? {
            assert!((value - 1.0).abs() <= 1.0e-6, "value={value}");
        }
        Ok(())
    }

    #[test]
    #[serial(mlx_metal)]
    fn supports_top4_and_casts_float32_weighted_sum_to_expert_dtype() -> Result<()> {
        let x_values: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 8.0).collect();
        let x_f32: Array = (x_values.as_slice(), [4, 1, 6]).try_into()?;
        let x = mlx::ops::cast::astype_on(&x_f32, Dtype::Bfloat16, ())?;
        let inverse: Array = (&[2_u32, 0, 3, 1][..], [4]).try_into()?;
        let scores: Array = (&[0.1_f32, 0.2, 0.3, 0.4][..], [1, 4]).try_into()?;

        let got = apply_on(&x, &inverse, &scores, true, ().into())?;
        let want = reference(&x, &inverse, &scores, ().into())?.astype(Dtype::Bfloat16)?;
        mlx::transforms::eval(&[&got, &want])?;

        assert_eq!(got.dtype(), Dtype::Bfloat16);
        let got = got.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let want = want.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 0.0078125, "max_abs={max_abs}");
        Ok(())
    }

    #[test]
    fn production_gate_starts_at_1024_tokens() -> Result<()> {
        fn inputs(tokens: i32) -> Result<(Array, Array, Array)> {
            let routes = tokens * 8;
            Ok((
                Array::zeros([routes, 1, 1], Dtype::Bfloat16)?,
                Array::zeros([routes], Dtype::Uint32)?,
                Array::zeros([tokens, 8], Dtype::Bfloat16)?,
            ))
        }

        let (x, order, scores) = inputs(MIN_TOKENS - 1)?;
        assert!(!should_use(&x, &order, &scores));
        let (x, order, scores) = inputs(MIN_TOKENS)?;
        assert!(should_use(&x, &order, &scores));
        Ok(())
    }
}
