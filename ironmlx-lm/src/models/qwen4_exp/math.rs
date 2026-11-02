//! Flash-Next math differs from the Qwen3.5 graph: grouped zero-centered
//! RMS weights, hyper-connections and L2 q/k normalization with epsilon
//! after the sum. Keep explicit stream targets and dtype transitions.
use crate::Result;
use anyhow::ensure;
use mlx::ops::binary::{add_on, divide_on, multiply_on};
use mlx::{Array, Dtype, StreamOrDevice};

fn scalar(value: f32, like: &Array, target: StreamOrDevice) -> Result<Array> {
    Ok(Array::try_from((&[value][..], ()))?.astype_on(like.dtype(), target)?)
}

/// Normalize every hidden-width group in fp32, apply `1 + checkpoint_weight`
/// in fp32, then cast once. Pre-adding one in BF16 loses checkpoint precision.
pub fn grouped_residual_rms_on(
    x: &Array,
    weight: &Array,
    group: i32,
    eps: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = x.shape();
    let dims = shape.as_slice();
    ensure!(
        group > 0 && eps.is_finite() && eps > 0.0 && !dims.is_empty(),
        "invalid grouped RMS configuration"
    );
    let width = *dims.last().unwrap();
    ensure!(
        width > 0 && width % group == 0 && weight.shape().as_slice() == [width],
        "grouped RMS weight/input mismatch"
    );
    let y = x
        .astype_on(Dtype::Float32, target)?
        .reshape_on([-1, width / group, group], target)?;
    let mean = y.square_on(target)?.mean_on(-1, true, target)?;
    let inverse = add_on(&mean, &scalar(eps, &mean, target)?, target)?.rsqrt_on(target)?;
    let raw_weight = weight.astype_on(Dtype::Float32, target)?;
    let gamma = add_on(&raw_weight, &scalar(1.0, &raw_weight, target)?, target)?
        .reshape_on([width / group, group], target)?;
    Ok(
        multiply_on(&multiply_on(&y, &inverse, target)?, &gamma, target)?
            .reshape_on(dims, target)?
            .astype_on(x.dtype(), target)?,
    )
}

pub fn hc_activate_on(projected: &Array, streams: i32, target: StreamOrDevice) -> Result<Array> {
    ensure!(streams > 1, "hyper-connection requires multiple streams");
    let x = divide_on(
        projected,
        &scalar(streams as f32, projected, target)?,
        target,
    )?;
    Ok(multiply_on(&x, &x.sigmoid_on(target)?, target)?)
}

pub fn hc_mix_on(
    projected: &Array,
    normed: &Array,
    streams: i32,
    hidden: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    ensure!(
        streams > 1 && hidden > 0 && projected.shape() == normed.shape(),
        "hyper-connection mix shape mismatch"
    );
    let shape = normed.shape();
    let dims = shape.as_slice();
    ensure!(
        dims.last().copied() == streams.checked_mul(hidden),
        "hyper-connection width mismatch"
    );
    let mut view = dims[..dims.len() - 1].to_vec();
    view.extend([streams, hidden]);
    let gates = projected
        .sigmoid_on(target)?
        .reshape_on(view.as_slice(), target)?;
    let mixed = multiply_on(&gates, &normed.reshape_on(view.as_slice(), target)?, target)?;
    Ok(mixed.mean_on(-2, false, target)?)
}

pub fn hc_injection_gate_on(
    projected: &Array,
    streams: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    ensure!(streams > 1, "hyper-connection requires multiple streams");
    let scaled = divide_on(
        projected,
        &scalar(streams as f32, projected, target)?,
        target,
    )?;
    let gate = scaled.sigmoid_on(target)?;
    Ok(multiply_on(&gate, &scalar(2.0, &gate, target)?, target)?)
}

pub fn hc_inject_on(
    hyper: &Array,
    branch: &Array,
    weights: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let hs = hyper.shape();
    let bs = branch.shape();
    let ws = weights.shape();
    let h = hs.as_slice();
    let b = bs.as_slice();
    let w = ws.as_slice();
    ensure!(
        !h.is_empty()
            && h.len() == b.len()
            && b.len() == w.len()
            && h[..h.len() - 1] == b[..b.len() - 1]
            && h[..h.len() - 1] == w[..w.len() - 1]
            && b.last().unwrap().checked_mul(*w.last().unwrap()) == h.last().copied(),
        "hyper-connection injection shape mismatch"
    );
    let branch = mlx::ops::shape::expand_dims_on(branch, -2, target)?;
    let weights = mlx::ops::shape::expand_dims_on(weights, -1, target)?;
    Ok(add_on(
        hyper,
        &multiply_on(&branch, &weights, target)?.reshape_on(h, target)?,
        target,
    )?)
}

/// Drafter-author contract: completed zero-based layer outputs, averaged
/// over HC streams. This deliberately does not apply the learned final mixer.
pub fn capture_hc_mean_on(
    hyper: &Array,
    streams: i32,
    hidden: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = hyper.shape();
    let dims = shape.as_slice();
    ensure!(
        streams > 1 && hidden > 0 && dims.last().copied() == streams.checked_mul(hidden),
        "invalid HC capture shape"
    );
    let mut view = dims[..dims.len() - 1].to_vec();
    view.extend([streams, hidden]);
    Ok(hyper
        .reshape_on(view.as_slice(), target)?
        .mean_on(-2, false, target)?)
}

pub fn normalize_delta_qk_on(
    q: &Array,
    k: &Array,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let shape = q.shape();
    let head = shape.as_slice().last().copied().unwrap_or(0);
    ensure!(
        head > 0 && k.shape().as_slice().last() == Some(&head),
        "invalid GDN q/k head width"
    );
    let normalize = |x: &Array| -> Result<Array> {
        let sum = x.square_on(target)?.sum_on(-1, true, target)?;
        let inverse = add_on(&sum, &scalar(1e-6, &sum, target)?, target)?.rsqrt_on(target)?;
        Ok(multiply_on(x, &inverse, target)?)
    };
    let q = normalize(q)?;
    let scale = scalar(1.0 / (head as f32).sqrt(), &q, target)?;
    Ok((multiply_on(&q, &scale, target)?, normalize(k)?))
}

/// GDN uses direct gamma, unlike the zero-centered HC/QSA norms.
pub fn sigmoid_gated_rms_on(
    x: &Array,
    gate: &Array,
    weight: &Array,
    eps: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    ensure!(
        x.shape() == gate.shape(),
        "Flash GDN output gate shape mismatch"
    );
    let normed =
        mlx::fast::rms_norm_on(x, Some(weight), eps, target)?.astype_on(Dtype::Float32, target)?;
    let gate = gate.astype_on(Dtype::Float32, target)?.sigmoid_on(target)?;
    Ok(multiply_on(&normed, &gate, target)?.astype_on(x.dtype(), target)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn values(case: &serde_json::Value, field: &str) -> Vec<f32> {
        case[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn array(case: &serde_json::Value, field: &str, shape: &[i32], dtype: Dtype) -> Array {
        Array::try_from((values(case, field).as_slice(), shape))
            .unwrap()
            .astype(dtype)
            .unwrap()
    }
    fn compare(case: &serde_json::Value, field: &str, actual: Array, dtype: Dtype) {
        assert_eq!(actual.dtype(), dtype, "{field} dtype");
        let a = actual
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let b = values(case, field);
        assert_eq!(a.len(), b.len());
        for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
            let tolerance = if dtype == Dtype::Bfloat16 {
                b.abs().max(1e-3) * 0.008
            } else {
                3e-6
            };
            assert!(
                (a - b).abs() <= tolerance,
                "{field}[{i}]: actual {a}, reference {b}, tolerance {tolerance}"
            );
        }
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn flash_components_match_pinned_mlx_vlm_reference() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen38_flash_next/components.json"
        )))
        .unwrap();
        for case in fixture["component_cases"].as_array().unwrap() {
            let dtype = if case["dtype"] == "float32" {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            };
            let target = StreamOrDevice::default();
            let input = array(case, "input", &[1, 2, 16], dtype);
            compare(
                case,
                "capture_mean",
                capture_hc_mean_on(&input, 4, 4, target).unwrap(),
                dtype,
            );
            compare(
                case,
                "gdn_norm_output",
                sigmoid_gated_rms_on(
                    &array(case, "q", &[1, 2, 2, 4], dtype),
                    &array(case, "k", &[1, 2, 2, 4], dtype),
                    &array(case, "gdn_norm_weight", &[4], dtype),
                    1e-6,
                    target,
                )
                .unwrap(),
                dtype,
            );
            let norm = grouped_residual_rms_on(
                &input,
                &array(case, "weight", &[16], dtype),
                4,
                1e-6,
                target,
            )
            .unwrap();
            compare(case, "norm_output", norm.clone(), dtype);
            compare(
                case,
                "activated_low",
                hc_activate_on(&array(case, "low_projection", &[1, 2, 3], dtype), 4, target)
                    .unwrap(),
                dtype,
            );
            let mixed = hc_mix_on(
                &array(case, "mix_projection", &[1, 2, 16], dtype),
                &norm,
                4,
                4,
                target,
            )
            .unwrap();
            compare(case, "mixed_output", mixed.clone(), dtype);
            let gate = hc_injection_gate_on(
                &array(case, "injection_projection", &[1, 2, 4], dtype),
                4,
                target,
            )
            .unwrap();
            compare(case, "injection_output", gate.clone(), dtype);
            compare(
                case,
                "residual_output",
                hc_inject_on(&input, &mixed, &gate, target).unwrap(),
                dtype,
            );
            let (q, k) = normalize_delta_qk_on(
                &array(case, "q", &[1, 2, 2, 4], dtype),
                &array(case, "k", &[1, 2, 2, 4], dtype),
                target,
            )
            .unwrap();
            compare(case, "q_output", q, dtype);
            compare(case, "k_output", k, dtype);
        }
    }
}
