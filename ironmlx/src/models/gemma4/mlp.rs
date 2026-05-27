use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};
use std::time::Instant;

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

use super::config::Gemma4LayerKind;
use super::ops::gelu_approx_mul_on;
use super::profile;

pub struct Gemma4GeGluMlp {
    gate_up: Linear,
    down: Linear,
    intermediate_size: i32,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
}

impl Gemma4GeGluMlp {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        intermediate_size: i32,
        layer_idx: usize,
        layer_kind: Gemma4LayerKind,
    ) -> Result<Self> {
        let gate_up = load_fused_gate_up(loader, prefix)
            .with_context(|| format!("loading fused Gemma4 GeGLU gate/up at `{prefix}`"))?;
        let down = Linear::from_loader(loader, &format!("{prefix}.down_proj"))
            .with_context(|| format!("loading Gemma4 GeGLU down at `{prefix}`"))?;
        Ok(Self {
            gate_up,
            down,
            intermediate_size,
            layer_idx,
            layer_kind,
        })
    }

    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let profile = profile::vl_layer_enabled();
        let t0 = Instant::now();
        let projected = self.gate_up.forward_on(x, target)?;
        profile::eval_layer(
            "gemma4_text_mlp_gate_up",
            self.layer_idx,
            self.layer_kind,
            &[&projected],
            t0,
            profile,
        )?;
        let t0 = Instant::now();
        let axis = projected.ndim() as i32 - 1;
        let parts =
            mlx::ops::shape::split_at_on(&projected, &[self.intermediate_size][..], axis, target)?;
        if parts.len() != 2 {
            return Err(anyhow!(
                "Gemma4GeGluMlp: split fused gate/up expected 2 parts, got {}",
                parts.len()
            ));
        }
        profile::eval_layer(
            "gemma4_text_mlp_split",
            self.layer_idx,
            self.layer_kind,
            &[&parts[0], &parts[1]],
            t0,
            profile,
        )?;
        let t0 = Instant::now();
        let activated = gelu_approx_mul_on(&parts[0], &parts[1], target)?;
        profile::eval_layer(
            "gemma4_text_mlp_geglu",
            self.layer_idx,
            self.layer_kind,
            &[&activated],
            t0,
            profile,
        )?;
        let t0 = Instant::now();
        let out = self.down.forward_on(&activated, target)?;
        profile::eval_layer(
            "gemma4_text_mlp_down",
            self.layer_idx,
            self.layer_kind,
            &[&out],
            t0,
            profile,
        )?;
        Ok(out)
    }
}

fn load_fused_gate_up(loader: &Loader, prefix: &str) -> Result<Linear> {
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let gate_w = loader.tensor(&format!("{gate}.weight"))?.clone();
    let up_w = loader.tensor(&format!("{up}.weight"))?.clone();
    let weight = mlx::ops::shape::concatenate(&[&gate_w, &up_w], 0)?;

    let gate_bias = loader.tensor_opt(&format!("{gate}.bias")).cloned();
    let up_bias = loader.tensor_opt(&format!("{up}.bias")).cloned();
    let bias = match (gate_bias, up_bias) {
        (Some(g), Some(u)) => Some(mlx::ops::shape::concatenate(&[&g, &u], 0)?),
        (None, None) => None,
        _ => {
            return Err(anyhow!(
                "Gemma4GeGluMlp: gate_proj/up_proj bias presence mismatch at `{prefix}`"
            ));
        }
    };

    let gate_scales_key = format!("{gate}.scales");
    let up_scales_key = format!("{up}.scales");
    if loader.contains(&gate_scales_key) || loader.contains(&up_scales_key) {
        if !loader.contains(&gate_scales_key) || !loader.contains(&up_scales_key) {
            return Err(anyhow!(
                "Gemma4GeGluMlp: gate_proj/up_proj quantization mismatch at `{prefix}`"
            ));
        }
        let qmeta = loader.quant_meta().ok_or_else(|| {
            anyhow!("Gemma4GeGluMlp: quantized gate/up present but Loader has no quant meta")
        })?;
        let gate_scales = loader.tensor(&gate_scales_key)?.clone();
        let up_scales = loader.tensor(&up_scales_key)?.clone();
        let scales = mlx::ops::shape::concatenate(&[&gate_scales, &up_scales], 0)?;

        let gate_biases = loader.tensor_opt(&format!("{gate}.biases")).cloned();
        let up_biases = loader.tensor_opt(&format!("{up}.biases")).cloned();
        let biases = match (gate_biases, up_biases) {
            (Some(g), Some(u)) => Some(mlx::ops::shape::concatenate(&[&g, &u], 0)?),
            (None, None) => None,
            _ => {
                return Err(anyhow!(
                    "Gemma4GeGluMlp: gate_proj/up_proj quantized biases mismatch at `{prefix}`"
                ));
            }
        };
        {
            let mut to_eval: Vec<&Array> = vec![&weight, &scales];
            if let Some(b) = &biases {
                to_eval.push(b);
            }
            if let Some(b) = &bias {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval).map_err(|e| {
                anyhow!("{prefix}: eager eval of fused gate/up tensors failed: {e}")
            })?;
        }
        Ok(Linear::new_quant(
            weight,
            scales,
            biases,
            bias,
            qmeta.group_size,
            qmeta.bits,
        ))
    } else {
        {
            let mut to_eval: Vec<&Array> = vec![&weight];
            if let Some(b) = &bias {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval).map_err(|e| {
                anyhow!("{prefix}: eager eval of fused gate/up tensors failed: {e}")
            })?;
        }
        Ok(Linear::new_fp(weight, bias))
    }
}
