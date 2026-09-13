use anyhow::anyhow;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

pub(super) fn build_entropy_probs_chain() -> CompiledFn {
    compile(
        |inputs| {
            let logits = inputs[0].astype(Dtype::Float32)?;
            let lse = mlx::ops::logsumexp(&logits, -1_i32, true)?;
            let log_probs = &logits - &lse;
            let probs = log_probs.exp()?;
            let entropy_terms = &probs * &log_probs;
            let entropy = mlx::ops::sum(&entropy_terms, -1_i32, false)?;
            let entropy = entropy.try_neg()?;
            Ok(vec![entropy, probs])
        },
        ShapeMode::Shapeless,
    )
    .expect("DiffusionGemma entropy/probs compile")
}

pub(super) fn build_entropy_transfer_mask_chain() -> CompiledFn {
    compile(
        |inputs| {
            let entropy = inputs[0];
            let entropy_bound = inputs[1];
            let sorted_indices = mlx::ops::sort::argsort(entropy, -1_i32)?;
            let sorted_entropy =
                mlx::ops::indexing::take_along_axis(entropy, &sorted_indices, -1_i32)?;
            let prefix_entropy =
                mlx::ops::cumulative::cumsum(&sorted_entropy, -1_i32, false, false)?;
            let sorted_mask = prefix_entropy.less_equal(entropy_bound)?;
            let scattered_mask = mlx::ops::constructors::zeros_like(&sorted_mask)?;
            let mask = mlx::ops::indexing::put_along_axis(
                &scattered_mask,
                &sorted_indices,
                &sorted_mask,
                -1_i32,
            )?;
            Ok(vec![mask])
        },
        ShapeMode::Fixed,
    )
    .expect("DiffusionGemma entropy transfer mask compile")
}

pub(super) fn build_stable_confidence_chain() -> CompiledFn {
    compile(
        |inputs| {
            let current_canvas = inputs[0];
            let previous_canvas = inputs[1];
            let token_entropy = inputs[2];
            let confidence_threshold = inputs[3];

            let same_tokens = current_canvas.equal(previous_canvas)?;
            let stable = mlx::ops::all(&same_tokens, mlx::ops::All, false)?;
            let mean_entropy = mlx::ops::mean(token_entropy, mlx::ops::All, false)?;
            let confident = mean_entropy.less(confidence_threshold)?;
            let should_stop = mlx::ops::indexing::where_(&stable, &confident, &stable)?;
            Ok(vec![should_stop])
        },
        ShapeMode::Shapeless,
    )
    .expect("DiffusionGemma stable confidence compile")
}

pub(super) fn entropy_probs_chain_on(
    logits: &Array,
    compiled: Option<&CompiledFn>,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    if let Some(compiled) = compiled {
        let mut outputs = compiled.invoke(&[logits])?;
        if outputs.len() != 2 {
            return Err(anyhow!(
                "DiffusionGemma entropy/probs chain returned {} outputs",
                outputs.len()
            ));
        }
        let probs = outputs.pop().expect("checked output length");
        let entropy = outputs.pop().expect("checked output length");
        return Ok((entropy, probs));
    }

    let logits = logits.astype_on(Dtype::Float32, target)?;
    let lse = mlx::ops::logsumexp_on(&logits, -1_i32, true, target)?;
    let log_probs = &logits - &lse;
    let probs = log_probs.exp_on(target)?;
    let entropy_terms = &probs * &log_probs;
    let entropy = mlx::ops::sum_on(&entropy_terms, -1_i32, false, target)?;
    let entropy = entropy.try_neg_on(target)?;
    Ok((entropy, probs))
}

pub(super) fn build_geglu_tanh() -> CompiledFn {
    compile(
        |inputs| {
            let gate = inputs[0];
            let up = inputs[1];
            let gelu = gelu_tanh_expr(gate)?;
            Ok(vec![&gelu * up])
        },
        ShapeMode::Shapeless,
    )
    .expect("DiffusionGemma GeGLU compile")
}

pub(super) fn invoke_geglu_tanh(func: &CompiledFn, gate: &Array, up: &Array) -> Result<Array> {
    let mut outs = func.invoke(&[gate, up])?;
    outs.pop()
        .ok_or_else(|| anyhow!("DiffusionGemma GeGLU returned no outputs"))
}

pub(super) fn build_logit_softcap(softcap: f32) -> CompiledFn {
    compile(
        move |inputs| {
            let logits = inputs[0].astype(Dtype::Float32)?;
            let capped = (logits / softcap).tanh()?;
            Ok(vec![&capped * softcap])
        },
        ShapeMode::Shapeless,
    )
    .expect("DiffusionGemma logit softcap compile")
}

pub(super) fn invoke_logit_softcap(func: &CompiledFn, logits: &Array) -> Result<Array> {
    let mut outs = func.invoke(&[logits])?;
    outs.pop()
        .ok_or_else(|| anyhow!("DiffusionGemma logit softcap returned no outputs"))
}

pub(super) fn scalar_array_like_on(
    value: f32,
    like: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let scalar: Array = (&[value][..], ()).try_into()?;
    if scalar.dtype() == like.dtype() {
        Ok(scalar)
    } else {
        Ok(scalar.astype_on(like.dtype(), target)?)
    }
}

pub(super) fn mul_scalar_like_on(x: &Array, value: f32, target: StreamOrDevice) -> Result<Array> {
    Ok(x * &scalar_array_like_on(value, x, target)?)
}

pub(super) fn div_scalar_like_on(x: &Array, value: f32, target: StreamOrDevice) -> Result<Array> {
    Ok(x.try_div_on(&scalar_array_like_on(value, x, target)?, target)?)
}

#[cfg(test)]
pub(super) fn eager_logit_softcap_on(
    logits: &Array,
    softcap: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let logits = logits.astype_on(Dtype::Float32, target)?;
    let capped = (logits / softcap).tanh_on(target)?;
    Ok(&capped * softcap)
}

fn gelu_tanh_expr(x: &Array) -> mlx::Result<Array> {
    let three: Array = (&[3_i32][..], ()).try_into()?;
    let x3 = x.power(&three)?;
    let dtype = x.dtype();
    let c_044715: Array = mlx::ops::cast::astype(&(&[0.044_715_f32][..], ()).try_into()?, dtype)?;
    let c_sqrt2pi: Array = mlx::ops::cast::astype(&(&[SQRT_2_OVER_PI][..], ()).try_into()?, dtype)?;
    let c_half: Array = mlx::ops::cast::astype(&(&[0.5_f32][..], ()).try_into()?, dtype)?;
    let c_one: Array = mlx::ops::cast::astype(&(&[1.0_f32][..], ()).try_into()?, dtype)?;
    let inner = (&(&x3 * &c_044715) + x) * &c_sqrt2pi;
    let t = inner.tanh()?;
    Ok(x * &c_half * (&t + &c_one))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &Array, expected: &Array, tol: f32) {
        let got_v = got.to_vec::<f32>().unwrap();
        let expected_v = expected.to_vec::<f32>().unwrap();
        assert_eq!(got_v.len(), expected_v.len());
        for (idx, (g, e)) in got_v.iter().zip(expected_v.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= tol,
                "idx={idx} got={g} expected={e} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn scalar_multiply_preserves_bfloat16_dtype() {
        let target = StreamOrDevice::default();
        let x: Array = (&[1.0_f32, 2.0, 4.0][..], &[3_i32][..]).try_into().unwrap();
        let x = x.astype_on(Dtype::Bfloat16, target).unwrap();

        let y = mul_scalar_like_on(&x, 0.5, target).unwrap();

        assert_eq!(y.dtype(), Dtype::Bfloat16);
        let values = y.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(values, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn scalar_divide_preserves_bfloat16_dtype() {
        let target = StreamOrDevice::default();
        let x: Array = (&[1.0_f32, 2.0, 4.0][..], &[3_i32][..]).try_into().unwrap();
        let x = x.astype_on(Dtype::Bfloat16, target).unwrap();

        let y = div_scalar_like_on(&x, 2.0, target).unwrap();

        assert_eq!(y.dtype(), Dtype::Bfloat16);
        let values = y.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(values, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn compiled_geglu_tanh_matches_eager_formula() {
        let gate: Array = (&[-1.0_f32, 0.0, 1.0, 2.0][..], &[2_i32, 2_i32][..])
            .try_into()
            .unwrap();
        let up: Array = (&[0.5_f32, 1.0, 1.5, 2.0][..], &[2_i32, 2_i32][..])
            .try_into()
            .unwrap();
        let compiled = build_geglu_tanh();
        let got = invoke_geglu_tanh(&compiled, &gate, &up).unwrap();
        let expected = &gelu_tanh_expr(&gate).unwrap() * &up;
        assert_close(&got, &expected, 1.0e-6);
    }

    #[test]
    fn compiled_logit_softcap_matches_eager_fp32_softcap() {
        let logits: Array = (&[-60.0_f32, -1.5, 0.0, 3.0, 60.0][..], &[1_i32, 5_i32][..])
            .try_into()
            .unwrap();
        let compiled = build_logit_softcap(30.0);
        let got = invoke_logit_softcap(&compiled, &logits).unwrap();
        let expected = eager_logit_softcap_on(&logits, 30.0, StreamOrDevice::default()).unwrap();
        assert_close(&got, &expected, 1.0e-6);
    }
}
