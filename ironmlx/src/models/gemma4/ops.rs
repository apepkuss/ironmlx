use anyhow::anyhow;
use mlx::{ops, Array, MetalKernel, StreamOrDevice};
use std::sync::OnceLock;

use crate::Result;

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

pub fn rms_norm_no_scale_on(
    x: &Array,
    eps: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    Ok(mlx::fast::rms_norm_on(x, None, eps, target)?)
}

pub fn gelu_approx_on(x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let target = target.into();
    let three: Array = (&[3_i32][..], ()).try_into()?;
    let x3 = x.power_on(&three, target)?;

    let dtype = x.dtype();
    let c_044715: Array = ops::cast::astype_on(&scalar_f32(0.044_715)?, dtype, target)?;
    let c_sqrt2pi: Array = ops::cast::astype_on(&scalar_f32(SQRT_2_OVER_PI)?, dtype, target)?;
    let c_half: Array = ops::cast::astype_on(&scalar_f32(0.5)?, dtype, target)?;
    let c_one: Array = ops::cast::astype_on(&scalar_f32(1.0)?, dtype, target)?;

    let inner = (&(&x3 * &c_044715) + x) * &c_sqrt2pi;
    let t = inner.tanh_on(target)?;
    Ok(x * &c_half * (&t + &c_one))
}

pub fn gelu_approx_mul_on(
    gate: &Array,
    up: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if gate.shape() != up.shape() {
        return Err(anyhow!(
            "Gemma4 GeGLU fused activation shape mismatch: gate {} vs up {}",
            gate.shape(),
            up.shape()
        ));
    }

    let target = target.into();
    let shape = gate.shape();
    let size = i32::try_from(shape.numel()).map_err(|_| {
        anyhow!(
            "Gemma4 GeGLU fused activation input too large: {} elements",
            shape.numel()
        )
    })?;
    if size == 0 {
        return Ok(Array::zeros_on(shape, gate.dtype(), target)?);
    }

    let kernel = geglu_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[gate, up])
        .output_shapes(&[shape])
        .output_dtypes(&[gate.dtype()])
        .grid(size, 1, 1)
        .threadgroup(256.min(size), 1, 1)
        .stream(target)
        .template_int("SIZE", size)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}

fn geglu_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        uint gid = thread_position_in_grid.x;
        if (gid >= SIZE) {
            return;
        }

        float x = float(gate[gid]);
        float u = float(up[gid]);
        float x2 = x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x2);
        float gelu = 0.5f * x * (1.0f + tanh(inner));
        out[gid] = static_cast<__typeof__(*out)>(gelu * u);
    "#;

    let kernel = MetalKernel::builder("ironmlx_gemma4_geglu")
        .inputs(&["gate", "up"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

pub fn logit_softcap_on(
    logits: &Array,
    softcap: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if softcap <= 0.0 {
        return Err(anyhow!("Gemma4 logit softcap must be > 0, got {softcap}"));
    }
    let target = target.into();
    let capped = (logits / softcap).tanh_on(target)?;
    Ok(&capped * softcap)
}

fn scalar_f32(v: f32) -> Result<Array> {
    Ok((&[v][..], ()).try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    fn assert_all_close(got: &Array, expected: &Array, tol: f32) {
        let got_f32 = ops::cast::astype(got, Dtype::Float32).unwrap();
        let expected_f32 = ops::cast::astype(expected, Dtype::Float32).unwrap();
        let got_v = got_f32.to_vec::<f32>().unwrap();
        let expected_v = expected_f32.to_vec::<f32>().unwrap();
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
    fn geglu_fused_matches_composed_f32() {
        let gate: Array = (
            &[-3.0_f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let up: Array = (
            &[0.5_f32, -2.0, 1.5, 3.0, -0.75, 2.5, -1.25, 0.25][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();

        let expected = &gelu_approx_on(&gate, ()).unwrap() * &up;
        let got = gelu_approx_mul_on(&gate, &up, ()).unwrap();
        assert_all_close(&got, &expected, 1e-5);
    }

    #[test]
    fn geglu_fused_accepts_split_views() {
        let gate: Array = (
            &[0.25_f32, -1.0, 2.0, 0.5, -0.75, 1.25][..],
            &[2_i32, 3][..],
        )
            .try_into()
            .unwrap();
        let up: Array = (&[1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75][..], &[2_i32, 3][..])
            .try_into()
            .unwrap();
        let fused = ops::shape::concatenate(&[&gate, &up], 1).unwrap();
        let parts = ops::shape::split_at(&fused, &[3_i32][..], 1).unwrap();

        let expected = &gelu_approx_on(&parts[0], ()).unwrap() * &parts[1];
        let got = gelu_approx_mul_on(&parts[0], &parts[1], ()).unwrap();
        assert_all_close(&got, &expected, 1e-5);
    }

    #[test]
    fn geglu_fused_matches_composed_bf16_with_final_bf16_tolerance() {
        let gate_f32: Array = (
            &[-3.0_f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let up_f32: Array = (
            &[0.5_f32, -2.0, 1.5, 3.0, -0.75, 2.5, -1.25, 0.25][..],
            &[2_i32, 4][..],
        )
            .try_into()
            .unwrap();
        let gate = ops::cast::astype(&gate_f32, Dtype::Bfloat16).unwrap();
        let up = ops::cast::astype(&up_f32, Dtype::Bfloat16).unwrap();

        let expected = &gelu_approx_on(&gate, ()).unwrap() * &up;
        let got = gelu_approx_mul_on(&gate, &up, ()).unwrap();
        assert_all_close(&got, &expected, 0.02);
    }

    #[test]
    fn geglu_fused_rejects_shape_mismatch() {
        let gate = Array::zeros((2, 3), Dtype::Float32).unwrap();
        let up = Array::zeros((3, 2), Dtype::Float32).unwrap();
        let err = gelu_approx_mul_on(&gate, &up, ()).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }
}
