//! Scoped M1 target projections. Uses native packed weights, no resident copy
//! or full-weight dequantization, and no M5/global Metal settings.
use super::linear::QuantizedLinearParts;
use crate::{core::QuantMode, Result};
use ironmlx_core::m1_profile::M1ExecutionPlan;
use mlx::{Array, Dtype, StreamOrDevice};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    static PLAN: RefCell<Option<Arc<M1ExecutionPlan>>> = const { RefCell::new(None) };
}
pub(crate) struct Scope(Option<Arc<M1ExecutionPlan>>);
impl Scope {
    pub(crate) fn enter_on(
        plan: Option<&Arc<M1ExecutionPlan>>,
        target: StreamOrDevice,
    ) -> Option<Self> {
        if plan.is_none_or(|plan| !plan.is_active()) {
            return Self::enter(None);
        }
        Self::enter(if primary_gpu(target) { plan } else { None })
    }

    pub(crate) fn enter(plan: Option<&Arc<M1ExecutionPlan>>) -> Option<Self> {
        let plan = plan.filter(|plan| plan.is_active());
        if plan.is_none() && !active() {
            return None;
        }
        Some(Self(PLAN.with(|current| current.replace(plan.cloned()))))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        PLAN.with(|current| current.replace(self.0.take()));
    }
}
pub(crate) fn active() -> bool {
    PLAN.with(|current| current.borrow().is_some())
}

fn primary_gpu(target: StreamOrDevice) -> bool {
    let device = match target {
        StreamOrDevice::Default => mlx::default_device(),
        StreamOrDevice::Device(d) => d,
        StreamOrDevice::Stream(s) => s.device,
        StreamOrDevice::ThreadLocalStream(s) => s.device,
    };
    device.device_type == mlx::DeviceType::Gpu && device.index == 0
}

pub(crate) fn forward(
    x: &Array,
    parts: QuantizedLinearParts<'_>,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    let Some(plan) = PLAN.with(|current| current.borrow().clone()) else {
        return Ok(None);
    };
    // Exact speculative and batch-isolated arithmetic owns its dispatch.
    // In particular, a wide verification block must never become ordinary
    // prefill QMM merely because its flattened row count exceeds sixteen.
    if super::verify_qmm::is_armed()
        || super::position_stable_qmm::is_armed()
        || super::product_stable_qmm::is_armed()
        || super::batch_stable_qmm::linear_is_armed()
    {
        plan.record_fallback();
        return Ok(None);
    }
    let k = *x.shape().as_slice().last().unwrap_or(&0);
    let w = parts.weight.shape();
    let compatible = parts.bits == 4
        && parts.group_size == 64
        && parts.mode == QuantMode::Affine
        && parts.bias.is_none()
        && x.dtype() == Dtype::Bfloat16
        && k >= 512
        && k % 512 == 0
        && w.as_slice().len() == 2
        && w.as_slice()[0] >= 8
        && w.as_slice()[0] % 8 == 0
        && w.as_slice()[1] * 8 == k
        && parts.weight.dtype() == Dtype::Uint32
        && parts.scales.dtype() == Dtype::Bfloat16
        && parts.scales.shape().as_slice() == [w.as_slice()[0], k / 64]
        && parts
            .biases
            .is_some_and(|b| b.dtype() == Dtype::Bfloat16 && b.shape() == parts.scales.shape());
    if !compatible {
        plan.record_fallback();
        return Ok(None);
    }
    // CPU streams never enter a GPU-only primitive.
    if !primary_gpu(target) {
        plan.record_fallback();
        return Ok(None);
    }
    let rows = x.size() / k as usize;
    if rows == 0
        || x.size() > i32::MAX as usize
        || rows * w.as_slice()[0] as usize > i32::MAX as usize
    {
        plan.record_fallback();
        return Ok(None);
    }
    let result = if (2..=16).contains(&rows) {
        mlx::quantization::qmv_fast_wide_on(
            x,
            parts.weight,
            parts.scales,
            parts.biases.expect("checked"),
            plan.device.expect("active device").qmv_rows_per_group(),
            target,
        )?
    } else if rows == 1 {
        mlx::quantization::quantized_matmul_product_stable_on(
            x,
            parts.weight,
            parts.scales,
            parts.biases,
            true,
            Some(64),
            Some(4),
            "affine",
            target,
        )?
    } else {
        // Prefill retains MLX's dtype/rounding and bounded native QMM;
        // no M5 tensor-unit or dequantized-weight workspace is introduced.
        let y = mlx::quantization::quantized_matmul_on(
            x,
            parts.weight,
            parts.scales,
            parts.biases,
            true,
            Some(64),
            Some(4),
            "affine",
            target,
        )?;
        plan.record_prefill();
        return Ok(Some(y));
    };
    plan.record_narrow();
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmlx_core::m1_profile::M1Device;

    fn bits(x: &Array) -> Vec<u32> {
        mlx::ops::cast::astype(x, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn scopes_mask_other_models_and_restore_after_errors() {
        let a = Arc::new(M1ExecutionPlan::resolve(true, Some(M1Device::Pro)));
        let b = Arc::new(M1ExecutionPlan::resolve(true, Some(M1Device::Max)));
        assert!(!active());
        let _a = Scope::enter(Some(&a));
        {
            let _off = Scope::enter(None);
            assert!(!active());
        }
        assert!(active());
        {
            let _cpu = Scope::enter_on(Some(&b), mlx::Device::cpu().into());
            assert!(!active());
        }
        assert!(active());
        {
            let _b = Scope::enter(Some(&b));
            assert!(Arc::ptr_eq(&PLAN.with(|p| p.borrow().clone().unwrap()), &b));
        }
        assert!(Arc::ptr_eq(&PLAN.with(|p| p.borrow().clone().unwrap()), &a));
        let result = std::panic::catch_unwind(|| {
            let _b = Scope::enter(Some(&b));
            panic!("controlled scope unwind");
        });
        assert!(result.is_err());
        assert!(Arc::ptr_eq(&PLAN.with(|p| p.borrow().clone().unwrap()), &a));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn gpu_projections_preserve_serial_bits_and_prefill_baseline() -> Result<()> {
        let root = std::env::var("MLX_DIR").expect("MLX_DIR");
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        let key = mlx::random::key(173)?;
        let normal = |shape: &[i32]| -> Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .dtype(Dtype::Bfloat16)
                .key(&key)
                .sample()?)
        };
        for (n, k) in [(128, 512), (520, 2048), (128, 5120)] {
            let q =
                mlx::quantization::quantize(&normal(&[n, k])?, Some(64), Some(4), "affine", None)?;
            let linear = super::super::Linear::new_quant(
                q[0].clone(),
                q[1].clone(),
                Some(q[2].clone()),
                None,
                64,
                4,
            );
            for device in [M1Device::Pro, M1Device::Max] {
                let plan = Arc::new(M1ExecutionPlan::resolve(true, Some(device)));
                for rows in [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 64] {
                    let x = normal(&[1, rows, k])?;
                    let expected = if rows <= 16 {
                        mlx::quantization::quantized_matmul_product_stable_on(
                            &x,
                            &q[0],
                            &q[1],
                            Some(&q[2]),
                            true,
                            Some(64),
                            Some(4),
                            "affine",
                            (),
                        )?
                    } else {
                        mlx::quantization::quantized_matmul_on(
                            &x,
                            &q[0],
                            &q[1],
                            Some(&q[2]),
                            true,
                            Some(64),
                            Some(4),
                            "affine",
                            (),
                        )?
                    };
                    let actual = {
                        let _scope = Scope::enter(Some(&plan));
                        linear.forward_on(&x, ())?
                    };
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "device={device:?} shape={rows}x{k}->{n}"
                    );
                    if rows <= 16 {
                        for row in 0..rows {
                            let one = mlx::ops::indexing::slice(
                                &x,
                                &[0, row, 0][..],
                                &[1, row + 1, k][..],
                            )?;
                            let serial = mlx::quantization::quantized_matmul_product_stable_on(
                                &one,
                                &q[0],
                                &q[1],
                                Some(&q[2]),
                                true,
                                Some(64),
                                Some(4),
                                "affine",
                                (),
                            )?;
                            assert_eq!(
                                &bits(&actual)
                                    [row as usize * n as usize..(row + 1) as usize * n as usize],
                                bits(&serial)
                            );
                        }
                    }
                }
                let snapshot = plan.snapshot();
                assert_eq!(snapshot.narrow_projection_calls, 9);
                assert_eq!(snapshot.prefill_projection_calls, 2);
                assert_eq!(snapshot.fallback_projection_calls, 0);
                // Explicit CPU operands and incompatible metadata must be declined.
                let _scope = Scope::enter(Some(&plan));
                let x = normal(&[1, k])?;
                assert!(forward(
                    &x,
                    linear.quantized_parts().unwrap(),
                    mlx::Device::cpu().into()
                )?
                .is_none());
                let wrong_dtype = mlx::ops::cast::astype(&x, Dtype::Float32)?;
                assert!(
                    forward(&wrong_dtype, linear.quantized_parts().unwrap(), ().into())?.is_none()
                );
                assert_eq!(plan.snapshot().fallback_projection_calls, 2);
                drop(_scope);
                for kind in 0..3 {
                    let x = normal(&[2, 17, k])?;
                    let _product = (kind == 0).then(super::super::product_stable_qmm::scope);
                    let _position = (kind == 1).then(super::super::position_stable_qmm::scope);
                    let _batch = (kind == 2).then(super::super::batch_stable_qmm::linear_scope);
                    let expected = linear.forward_on(&x, ())?;
                    let actual = {
                        let _m1 = Scope::enter(Some(&plan));
                        linear.forward_on(&x, ())?
                    };
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "legacy isolation kind={kind}"
                    );
                }
                assert_eq!(plan.snapshot().fallback_projection_calls, 5);
            }
        }
        assert!(!active());
        Ok(())
    }
}
