//! Numerically explicit operations used by Inkling's reference execution graph.
use anyhow::Result;
use mlx::ops::{indexing, shape};
use mlx::{Array, Dtype, StreamOrDevice};

pub(super) fn scalar(x: f32) -> Result<Array> {
    Ok((&[x][..], ()).try_into()?)
}
pub(super) fn slice(
    x: &Array,
    axis: usize,
    start: i32,
    end: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let mut lo = vec![0; x.ndim()];
    let mut hi = x.shape().as_slice().to_vec();
    lo[axis] = start;
    hi[axis] = end;
    Ok(indexing::slice_strided_on(
        x,
        lo.as_slice(),
        hi.as_slice(),
        vec![1; x.ndim()].as_slice(),
        target,
    )?)
}
pub(super) fn cat(xs: &[&Array], axis: i32, target: StreamOrDevice) -> Result<Array> {
    Ok(shape::concatenate_on(xs, axis, target)?)
}
pub(super) fn norm(x: &Array, target: StreamOrDevice) -> Result<Array> {
    Ok(mlx::fast::rms_norm_on(x, None, 1e-6, target)?)
}
pub(super) fn scale(x: &Array, a: f32, target: StreamOrDevice) -> Result<Array> {
    Ok(x.try_mul_on(&scalar(a)?, target)?)
}
pub(super) fn cap(x: &Array, a: f32, target: StreamOrDevice) -> Result<Array> {
    scale(&scale(x, 1.0 / a, target)?.tanh_on(target)?, a, target)
}
pub(super) fn l2_clip(
    x: &Array,
    weights: Option<&Array>,
    threshold: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let d = x.shape_at(-1);
    if x.shape_at(0) == 1
        && x.ndim() == 4
        && matches!(d, 16 | 128)
        && x.dtype() == Dtype::Bfloat16
        && weights.is_none_or(|w| w.shape().as_slice() == [x.shape_at(2), d])
        && super::layers::is_gpu(target)
    {
        return super::metal::clip_heads(x, weights, threshold, target);
    }
    reference_clip(x, weights, threshold, target)
}

fn reference_clip(
    x: &Array,
    weights: Option<&Array>,
    threshold: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let mut y = x.astype_on(Dtype::Float32, target)?;
    if let Some(w) = weights {
        y = y.try_mul_on(w, target)?;
    }
    let n = y
        .square_on(target)?
        .sum_on(-1, true, target)?
        .sqrt_on(target)?
        .try_add_on(&scalar(1e-6)?, target)?;
    let ratio = scalar(threshold)?.try_div_on(&n, target)?;
    let ratio = ratio.minimum_on(&scalar(1.0)?, target)?;
    Ok(y.try_mul_on(&ratio, target)?.astype_on(x.dtype(), target)?)
}
pub(super) fn depth_attention(
    blocks: &[Array],
    partial: &Array,
    query: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    if super::layers::FAST_OQ.get()
        && partial.shape_at(0) == 1
        && partial.shape_at(1) <= 8
        && partial.shape_at(-1) == 2304
        && query.size() == 2304
        && blocks.len() < 8
        && partial.dtype() == Dtype::Float32
        && query.dtype() == Dtype::Float32
        && super::layers::is_gpu(target)
    {
        return super::metal::single_depth_query(blocks, partial, query, target);
    }
    let states = depth_states(blocks, partial, target)?;
    mix_depth_states(&states, query, target)
}

pub(super) fn depth_states(
    blocks: &[Array],
    partial: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let refs: Vec<_> = blocks.iter().chain(std::iter::once(partial)).collect();
    Ok(shape::stack_on(&refs, -2, target)?)
}

pub(super) fn mix_depth_states(
    states: &Array,
    query: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let q = query
        .astype_on(Dtype::Float32, target)?
        .reshape_on((-1, states.shape_at(-1)), target)?;
    let logits = states.matmul_on(&q.transpose_on(target)?, target)?;
    let factor = (states.shape_at(-1) as f32).powf(-0.5);
    if super::layers::FAST_OQ.get()
        && states.shape_at(0) == 1
        && states.shape_at(1) <= 8
        && states.shape_at(2) <= 8
        && states.shape_at(-1) == 2304
        && matches!(q.shape_at(0), 1 | 3 | 4)
        && states.dtype() == Dtype::Float32
        && super::layers::is_gpu(target)
    {
        return super::metal::mix_depth(&logits, states, factor, target);
    }
    let logits = scale(&logits, factor, target)?;
    let weights =
        mlx::ops::softmax_on(&logits, -2, true, target)?.transpose_axes_on([0, 1, 3, 2], target)?;
    Ok(weights.matmul_on(states, target)?)
}
pub(super) fn gelu(x: &Array, target: StreamOrDevice) -> Result<Array> {
    let y = x.astype_on(Dtype::Float32, target)?;
    let factor = scale(&y, std::f32::consts::FRAC_1_SQRT_2, target)?
        .erf_on(target)?
        .try_add_on(&scalar(1.0)?, target)?;
    Ok(scale(&y.try_mul_on(&factor, target)?, 0.5, target)?.astype_on(x.dtype(), target)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn clipped_heads_preserve_bfloat16_rounding() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        for (tokens, heads, dim) in [(1, 24, 128), (512, 24, 128), (512, 24, 16)] {
            let key = mlx::random::key(342)?;
            let x = mlx::random::normal()
                .shape(&[1, tokens, heads, dim][..])
                .key(&key)
                .sample()?
                .astype(Dtype::Bfloat16)?;
            let w = mlx::random::normal()
                .shape(&[heads, dim][..])
                .key(&mlx::random::split(&key)?.0)
                .sample()?
                .astype(Dtype::Bfloat16)?;
            for weight in [None, Some(&w)] {
                let actual = super::super::metal::clip_heads(&x, weight, 3.0, t)?;
                let expected = reference_clip(&x, weight, 3.0, t)?;
                let diffs = actual
                    .not_equal(&expected)?
                    .astype(Dtype::Int32)?
                    .sum(mlx::ops::All, false)?
                    .item::<i32>()?;
                assert_eq!(
                    diffs,
                    0,
                    "tokens={tokens} heads={heads} dim={dim} weighted={}",
                    weight.is_some()
                );
            }
        }
        Ok(())
    }
}
