use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

use super::ops::gelu_approx_on;

pub struct Gemma4GeGluMlp {
    gate_up: Linear,
    down: Linear,
    intermediate_size: i32,
}

impl Gemma4GeGluMlp {
    pub fn from_loader(loader: &Loader, prefix: &str, intermediate_size: i32) -> Result<Self> {
        let gate_up = load_fused_gate_up(loader, prefix)
            .with_context(|| format!("loading fused Gemma4 GeGLU gate/up at `{prefix}`"))?;
        let down = Linear::from_loader(loader, &format!("{prefix}.down_proj"))
            .with_context(|| format!("loading Gemma4 GeGLU down at `{prefix}`"))?;
        Ok(Self {
            gate_up,
            down,
            intermediate_size,
        })
    }

    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let projected = self.gate_up.forward_on(x, target)?;
        let axis = projected.ndim() as i32 - 1;
        let parts =
            mlx::ops::shape::split_at_on(&projected, &[self.intermediate_size][..], axis, target)?;
        if parts.len() != 2 {
            return Err(anyhow!(
                "Gemma4GeGluMlp: split fused gate/up expected 2 parts, got {}",
                parts.len()
            ));
        }
        let gate = gelu_approx_on(&parts[0], target)?;
        let activated = &gate * &parts[1];
        self.down.forward_on(&activated, target)
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
