//! Learned hyper-connection mixing and branch injection. The final output
//! mixer has no injection projection. Draft capture uses raw stream means.
use anyhow::{ensure, Result};
use mlx::{Array, StreamOrDevice};

use super::{config::FlashTextConfig, math};
use crate::core::weights::WeightSource;
use crate::nn::Linear;

pub struct HyperConnection {
    weight: Array,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
    streams: i32,
    hidden: i32,
    eps: f32,
}

pub struct HyperBranch {
    pub input: Array,
    residual: Array,
    injection: Array,
}

impl HyperBranch {
    pub fn combine_on(&self, output: &Array, target: StreamOrDevice) -> Result<Array> {
        math::hc_inject_on(&self.residual, output, &self.injection, target)
    }
}

impl HyperConnection {
    pub fn from_weights(
        source: &(impl WeightSource + ?Sized),
        prefix: &str,
        cfg: &FlashTextConfig,
        branch: bool,
    ) -> Result<Self> {
        let weight = source.tensor(&format!("{prefix}.hc_norm.weight"))?.clone();
        let down = Linear::from_loader(source, &format!("{prefix}.input_mix_weight_down"))?;
        let up = Linear::from_loader(source, &format!("{prefix}.input_mix_weight_up"))?;
        let inject = branch
            .then(|| Linear::from_loader(source, &format!("{prefix}.block_inject_weight")))
            .transpose()?;
        let width = usize::try_from(
            cfg.hidden_size
                .checked_mul(cfg.hc_count)
                .ok_or_else(|| anyhow::anyhow!("HC width overflow"))?,
        )?;
        ensure!(
            weight.shape().as_slice() == [i32::try_from(width)?]
                && down.in_features() == width
                && down.out_features() == cfg.hc_lowrank as usize
                && up.in_features() == cfg.hc_lowrank as usize
                && up.out_features() == width,
            "HC checkpoint projection shape mismatch at {prefix}"
        );
        if let Some(inject) = &inject {
            ensure!(
                inject.in_features() == width && inject.out_features() == cfg.hc_count as usize,
                "HC injection shape mismatch at {prefix}"
            );
        }
        Ok(Self {
            weight,
            down,
            up,
            inject,
            streams: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }

    fn project_on(&self, hyper: &Array, target: StreamOrDevice) -> Result<(Array, Array)> {
        let normed =
            math::grouped_residual_rms_on(hyper, &self.weight, self.hidden, self.eps, target)?;
        let activated = math::hc_activate_on(
            &self.down.forward_on(&normed, target)?,
            self.streams,
            target,
        )?;
        let input = math::hc_mix_on(
            &self.up.forward_on(&activated, target)?,
            &normed,
            self.streams,
            self.hidden,
            target,
        )?;
        Ok((input, normed))
    }

    pub fn branch_on(&self, hyper: &Array, target: StreamOrDevice) -> Result<HyperBranch> {
        let inject = self
            .inject
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("final HC mixer has no branch"))?;
        let (input, normed) = self.project_on(hyper, target)?;
        let _stable = crate::nn::position_stable_qmm::scope();
        let injection =
            math::hc_injection_gate_on(&inject.forward_on(&normed, target)?, self.streams, target)?;
        Ok(HyperBranch {
            input,
            residual: hyper.clone(),
            injection,
        })
    }

    pub fn final_mix_on(&self, hyper: &Array, target: StreamOrDevice) -> Result<Array> {
        ensure!(
            self.inject.is_none(),
            "branch HC requires residual injection"
        );
        Ok(self.project_on(hyper, target)?.0)
    }
}
