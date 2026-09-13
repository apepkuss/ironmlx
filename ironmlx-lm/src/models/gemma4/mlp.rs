use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};
use std::time::Instant;

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

use super::config::Gemma4LayerKind;
use super::ops::{
    gelu_approx_mul_on, quantized_gate_up_geglu_decode_on, QuantizedGateUpGeGluDecode,
};
use super::profile;
use super::quant_fusion::{fused_quant_compatibility, FusedQuantCompatibility};

enum Gemma4GateUpProjection {
    Fused(Linear),
    Separate { gate: Linear, up: Linear },
}

pub struct Gemma4GeGluMlp {
    gate_up: Gemma4GateUpProjection,
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
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let compat = fused_quant_compatibility(
            loader,
            &[gate.as_str(), up.as_str()],
            "Gemma4GeGluMlp gate/up",
        )?;
        let gate_up = if compat == FusedQuantCompatibility::MixedQuantized {
            Gemma4GateUpProjection::Separate {
                gate: Linear::from_loader(loader, &gate)
                    .with_context(|| format!("loading Gemma4 GeGLU gate_proj at `{prefix}`"))?,
                up: Linear::from_loader(loader, &up)
                    .with_context(|| format!("loading Gemma4 GeGLU up_proj at `{prefix}`"))?,
            }
        } else {
            Gemma4GateUpProjection::Fused(
                load_fused_gate_up(loader, prefix, compat)
                    .with_context(|| format!("loading fused Gemma4 GeGLU gate/up at `{prefix}`"))?,
            )
        };
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

        match &self.gate_up {
            Gemma4GateUpProjection::Fused(gate_up) => {
                if let Some(parts) = gate_up.quantized_parts() {
                    if let (None, Some(qbiases)) = (parts.bias, parts.biases) {
                        let t0 = Instant::now();
                        let params = QuantizedGateUpGeGluDecode {
                            weight: parts.weight,
                            scales: parts.scales,
                            biases: qbiases,
                            intermediate_size: self.intermediate_size,
                            group_size: parts.group_size,
                            bits: parts.bits,
                            mode: parts.mode,
                        };
                        if let Some(activated) =
                            quantized_gate_up_geglu_decode_on(x, params, target)?
                        {
                            profile::eval_layer(
                                "gemma4_text_mlp_gate_up_geglu_fused",
                                self.layer_idx,
                                self.layer_kind,
                                &[&activated],
                                t0,
                                profile,
                            )?;
                            return self.forward_down_on(&activated, target, profile);
                        }
                    }
                }

                let t0 = Instant::now();
                let projected = gate_up.forward_on(x, target)?;
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
                let parts = mlx::ops::shape::split_at_on(
                    &projected,
                    &[self.intermediate_size][..],
                    axis,
                    target,
                )?;
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
                self.forward_down_on(&activated, target, profile)
            }
            Gemma4GateUpProjection::Separate { gate, up } => {
                let t0 = Instant::now();
                let gate_projected = gate.forward_on(x, target)?;
                let up_projected = up.forward_on(x, target)?;
                profile::eval_layer(
                    "gemma4_text_mlp_gate_up_separate",
                    self.layer_idx,
                    self.layer_kind,
                    &[&gate_projected, &up_projected],
                    t0,
                    profile,
                )?;
                let t0 = Instant::now();
                let activated = gelu_approx_mul_on(&gate_projected, &up_projected, target)?;
                profile::eval_layer(
                    "gemma4_text_mlp_geglu",
                    self.layer_idx,
                    self.layer_kind,
                    &[&activated],
                    t0,
                    profile,
                )?;
                self.forward_down_on(&activated, target, profile)
            }
        }
    }

    fn forward_down_on(
        &self,
        activated: &Array,
        target: StreamOrDevice,
        profile: bool,
    ) -> Result<Array> {
        let t0 = Instant::now();
        let out = self.down.forward_on(activated, target)?;
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

fn load_fused_gate_up(
    loader: &Loader,
    prefix: &str,
    compat: FusedQuantCompatibility,
) -> Result<Linear> {
    if compat == FusedQuantCompatibility::MixedQuantized {
        return Err(anyhow!(
            "Gemma4GeGluMlp: mixed gate_proj/up_proj quantization cannot be fused at `{prefix}`"
        ));
    }

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

    if let FusedQuantCompatibility::Quantized(qmeta) = compat {
        let gate_scales_key = format!("{gate}.scales");
        let up_scales_key = format!("{up}.scales");
        let gate_scales = loader.tensor(&gate_scales_key)?.clone();
        let up_scales = loader.tensor(&up_scales_key)?.clone();
        let scales = mlx::ops::shape::concatenate(&[&gate_scales, &up_scales], 0)?;

        let gate_biases = loader.tensor_opt(&format!("{gate}.biases")).cloned();
        let up_biases = loader.tensor_opt(&format!("{up}.biases")).cloned();
        qmeta.validate_storage(&gate, &gate_w, &gate_scales, gate_biases.as_ref())?;
        qmeta.validate_storage(&up, &up_w, &up_scales, up_biases.as_ref())?;
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
        Ok(Linear::new_quant_with_mode(
            weight,
            scales,
            biases,
            bias,
            qmeta.group_size,
            qmeta.bits,
            qmeta.mode,
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
