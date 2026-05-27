use anyhow::anyhow;
use mlx::{ops, Array, StreamOrDevice};

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
