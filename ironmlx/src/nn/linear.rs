//! `nn::Linear` — single struct with private enum dispatch over fp and
//! quantized backends.
//!
//! Construction goes through [`Linear::from_loader`], which probes
//! `{prefix}.scales` to choose the variant. Forward computes
//! `y = x @ W^T + bias` for fp weights, or calls
//! [`mlx::quantization::quantized_matmul`] (with `transpose=true`) for
//! quantized weights, then optionally adds the (non-quantized) `bias`.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::{logical_width_from_packed, Loader, QuantMode};
use crate::Result;

/// Linear projection layer. Handles both full-precision and quantized
/// weight checkpoints transparently.
pub struct Linear {
    inner: LinearImpl,
}

/// Borrowed quantized Linear internals for architecture-specific fused paths.
#[derive(Clone, Copy)]
pub(crate) struct QuantizedLinearParts<'a> {
    pub(crate) weight: &'a Array,
    pub(crate) scales: &'a Array,
    pub(crate) biases: Option<&'a Array>,
    pub(crate) bias: Option<&'a Array>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    pub(crate) mode: QuantMode,
}

/// Internal backend variant. Private — callers use [`Linear`].
enum LinearImpl {
    Fp {
        /// `[out, in]` dense weight, dtype as stored in the checkpoint.
        weight: Array,
        /// Optional `[out]` bias.
        bias: Option<Array>,
    },
    Quant {
        /// Packed quantized weight (layout per `mlx::quantization`).
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group zero-points (affine quantization).
        biases: Option<Array>,
        /// Optional Linear bias term, applied after the quantized matmul.
        bias: Option<Array>,
        /// Group size from quantization metadata.
        group_size: i32,
        /// Bits per quantized weight.
        bits: i32,
        /// Quantization scheme from loader metadata.
        mode: QuantMode,
    },
}

impl Linear {
    /// Build a `Linear` from `loader`, looking for tensors at
    /// `{prefix}.weight` (required), `{prefix}.bias` (optional),
    /// `{prefix}.scales` (signals quantized variant), and
    /// `{prefix}.biases` (optional zero-points for affine quant).
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let bias_key = format!("{prefix}.bias");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");

        let weight = loader.tensor(&weight_key)?.clone();
        let bias = loader.tensor_opt(&bias_key).cloned();

        if loader.contains(&scales_key) {
            let qmeta = loader.quant_meta_for(prefix).ok_or_else(|| {
                anyhow!(
                    "Linear `{prefix}`: `{scales_key}` present but Loader has no quantization meta"
                )
            })?;
            let scales = loader.tensor(&scales_key)?.clone();
            let biases = loader.tensor_opt(&biases_key).cloned();
            qmeta.validate_storage(prefix, &weight, &scales, biases.as_ref())?;
            Ok(Linear {
                inner: LinearImpl::Quant {
                    weight,
                    scales,
                    biases,
                    bias,
                    group_size: qmeta.group_size,
                    bits: qmeta.bits,
                    mode: qmeta.mode,
                },
            })
        } else {
            Ok(Linear {
                inner: LinearImpl::Fp { weight, bias },
            })
        }
    }

    /// Test/composition seam: build an FP `Linear` from in-memory weight (and optional bias).
    /// Production code should use [`Linear::from_loader`]. This bypass lets `nn` building
    /// blocks be composed without writing a safetensors file (used by `GatedAttention`'s
    /// `from_components` constructor, unit tests, and integration tests).
    ///
    /// `weight` must be shape `[out, in]`; `bias` must be `[out]` if `Some`.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new_fp(weight: Array, bias: Option<Array>) -> Self {
        Self {
            inner: LinearImpl::Fp { weight, bias },
        }
    }

    /// Compose a quantized [`Linear`] from already-loaded Arrays. Used by
    /// callers that fuse multiple weight tensors at load time (e.g.
    /// [`GatedDeltaNet`](crate::nn::GatedDeltaNet)'s concatenated input
    /// projections). Production code that loads a single weight from a
    /// safetensors checkpoint should use [`Linear::from_loader`].
    ///
    /// `weight` is the packed quantized weight matrix; `scales` is per-group
    /// scales; `biases` is per-group zero-points (Some for affine
    /// quantization, None for symmetric); `bias` is the additive linear bias
    /// term separate from `biases` (typically None for Qwen3.5).
    /// `group_size` and `bits` are the quantization metadata (typically
    /// 64 / 4 for Qwen3.5 4-bit checkpoints).
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can
    /// use it. Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new_quant(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::new_quant_with_mode(
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
            QuantMode::Affine,
        )
    }

    /// Compose a quantized [`Linear`] with an explicit quantization mode.
    #[doc(hidden)]
    pub fn new_quant_with_mode(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: QuantMode,
    ) -> Self {
        Self {
            inner: LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            },
        }
    }

    /// Forward pass: `y = x @ W^T (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Number of input features (the trailing axis of the input the layer accepts).
    ///
    /// For fp weights stored as `[out, in]`, returns `weight.shape()[1]`.
    /// For quantized weights packed at `bits` bits per element into `u32`
    /// (32-bit) lanes, `in_features = weight.shape()[1] * 32 / bits`.
    pub fn in_features(&self) -> usize {
        match &self.inner {
            LinearImpl::Fp { weight, .. } => weight.shape().as_slice()[1] as usize,
            LinearImpl::Quant { weight, bits, .. } => {
                logical_width_from_packed(weight.shape().as_slice()[1], *bits)
                    .expect("quantized Linear must have a valid packed input width")
                    as usize
            }
        }
    }

    /// Number of output features (the trailing axis of the output the layer produces).
    pub fn out_features(&self) -> usize {
        match &self.inner {
            LinearImpl::Fp { weight, .. } => weight.shape().as_slice()[0] as usize,
            LinearImpl::Quant { weight, .. } => weight.shape().as_slice()[0] as usize,
        }
    }

    pub(crate) fn quantized_parts(&self) -> Option<QuantizedLinearParts<'_>> {
        match &self.inner {
            LinearImpl::Fp { .. } => None,
            LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            } => Some(QuantizedLinearParts {
                weight,
                scales,
                biases: biases.as_ref(),
                bias: bias.as_ref(),
                group_size: *group_size,
                bits: *bits,
                mode: *mode,
            }),
        }
    }

    /// Fuse output rows from matching quantized projections without retaining
    /// duplicate weights. Each output row keeps the same affine-4 or affine-8
    /// dot-product accumulation tree; callers split the fused result on the
    /// original row boundaries.
    pub(crate) fn fuse_quantized_outputs(projections: &[&Linear], context: &str) -> Result<Self> {
        let first = projections
            .first()
            .ok_or_else(|| anyhow!("{context} requires at least one projection"))?;
        let first_parts = first
            .quantized_parts()
            .ok_or_else(|| anyhow!("{context} requires quantized projections"))?;
        if first_parts.mode != QuantMode::Affine || !matches!(first_parts.bits, 4 | 8) {
            return Err(anyhow!(
                "{context} requires affine 4-bit or 8-bit projections"
            ));
        }
        let mut parts = Vec::with_capacity(projections.len());
        for projection in projections {
            let candidate = projection
                .quantized_parts()
                .ok_or_else(|| anyhow!("{context} requires quantized projections"))?;
            if projection.in_features() != first.in_features()
                || candidate.group_size != first_parts.group_size
                || candidate.bits != first_parts.bits
                || candidate.mode != first_parts.mode
                || candidate.weight.dtype() != first_parts.weight.dtype()
                || candidate.scales.dtype() != first_parts.scales.dtype()
            {
                return Err(anyhow!(
                    "{context} requires matching quantized input layouts"
                ));
            }
            parts.push(candidate);
        }

        let weight_refs = parts.iter().map(|part| part.weight).collect::<Vec<_>>();
        let scale_refs = parts.iter().map(|part| part.scales).collect::<Vec<_>>();
        let weight = mlx::ops::shape::concatenate_on(&weight_refs, 0, ())?;
        let scales = mlx::ops::shape::concatenate_on(&scale_refs, 0, ())?;
        let biases = if parts.iter().all(|part| part.biases.is_some()) {
            let refs = parts
                .iter()
                .map(|part| part.biases.expect("checked affine biases"))
                .collect::<Vec<_>>();
            Some(mlx::ops::shape::concatenate_on(&refs, 0, ())?)
        } else if parts.iter().all(|part| part.biases.is_none()) {
            None
        } else {
            return Err(anyhow!(
                "{context} requires matching quantization-bias presence"
            ));
        };
        let bias = if parts.iter().all(|part| part.bias.is_some()) {
            let refs = parts
                .iter()
                .map(|part| part.bias.expect("checked additive biases"))
                .collect::<Vec<_>>();
            Some(mlx::ops::shape::concatenate_on(&refs, 0, ())?)
        } else if parts.iter().all(|part| part.bias.is_none()) {
            None
        } else {
            return Err(anyhow!(
                "{context} requires matching additive-bias presence"
            ));
        };
        let mut arrays = vec![&weight, &scales];
        if let Some(biases) = &biases {
            arrays.push(biases);
        }
        if let Some(bias) = &bias {
            arrays.push(bias);
        }
        mlx::transforms::eval(&arrays)?;
        Ok(Self::new_quant_with_mode(
            weight,
            scales,
            biases,
            bias,
            first_parts.group_size,
            first_parts.bits,
            first_parts.mode,
        ))
    }

    /// Split a fused quantized projection into row views that share the fused
    /// storage. This lets DFlash2 retain the ordinary projection morphology
    /// for prefill and Q=1 work without keeping a second copy of the weights.
    pub(crate) fn split_quantized_outputs(
        &self,
        output_widths: &[usize],
        context: &str,
    ) -> Result<Vec<Self>> {
        let parts = self
            .quantized_parts()
            .ok_or_else(|| anyhow!("{context} requires a quantized projection"))?;
        if output_widths.is_empty() || output_widths.contains(&0) {
            return Err(anyhow!("{context} requires non-empty output widths"));
        }
        let total_width = output_widths.iter().try_fold(0_usize, |total, width| {
            total
                .checked_add(*width)
                .ok_or_else(|| anyhow!("{context} output width overflow"))
        })?;
        if total_width != self.out_features() {
            return Err(anyhow!(
                "{context} output widths total {total_width}, expected {}",
                self.out_features()
            ));
        }
        let mut cumulative = 0_usize;
        let cuts = output_widths
            .iter()
            .take(output_widths.len() - 1)
            .map(|width| {
                cumulative = cumulative
                    .checked_add(*width)
                    .ok_or_else(|| anyhow!("{context} output cut overflow"))?;
                i32::try_from(cumulative).map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let weights = mlx::ops::shape::split_at_on(parts.weight, &cuts, 0, ())?;
        let scales = mlx::ops::shape::split_at_on(parts.scales, &cuts, 0, ())?;
        let biases = parts
            .biases
            .map(|array| mlx::ops::shape::split_at_on(array, &cuts, 0, ()))
            .transpose()?;
        let bias = parts
            .bias
            .map(|array| mlx::ops::shape::split_at_on(array, &cuts, 0, ()))
            .transpose()?;
        let mut projections = Vec::with_capacity(output_widths.len());
        for index in 0..output_widths.len() {
            projections.push(Self::new_quant_with_mode(
                weights[index].clone(),
                scales[index].clone(),
                biases.as_ref().map(|arrays| arrays[index].clone()),
                bias.as_ref().map(|arrays| arrays[index].clone()),
                parts.group_size,
                parts.bits,
                parts.mode,
            ));
        }
        // `split_at_on` produces lazy row views on the loading thread's MLX
        // stream. DFlash2 moves the constructed target model into its actor
        // thread, so retaining those lazy views would make the first Q=1
        // projection try to use a command encoder owned by another thread.
        // Evaluate every view here while preserving the shared fused storage.
        let mut arrays = Vec::new();
        for projection in &projections {
            let projection_parts = projection
                .quantized_parts()
                .expect("split projections remain quantized");
            arrays.push(projection_parts.weight);
            arrays.push(projection_parts.scales);
            if let Some(biases) = projection_parts.biases {
                arrays.push(biases);
            }
            if let Some(bias) = projection_parts.bias {
                arrays.push(bias);
            }
        }
        mlx::transforms::eval(&arrays)?;
        Ok(projections)
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        if super::position_stable_qmm::exact_affine8_b4_q2_is_armed() {
            if let Some(parts) = self.quantized_parts() {
                if let Some(output) =
                    super::verify_qmm::forward_affine8_b4_q2_exact_on(x, parts, target)?
                {
                    return Ok(output);
                }
            }
        }
        if super::position_stable_linear::is_armed()
            && x.ndim() == 3
            && x.shape().as_slice()[1] > 1
            && matches!(self.inner, LinearImpl::Fp { .. })
        {
            return self.forward_fp_positions_isolated_on(x, target);
        }
        if super::position_stable_qmm::is_armed()
            && !super::product_stable_qmm::is_armed()
            && x.ndim() == 3
            && x.shape().as_slice()[1] > 1
            && self.quantized_parts().is_some()
        {
            return self.forward_positions_isolated_on(x, target);
        }
        if super::verify_qmm::is_armed() {
            if let Some(parts) = self.quantized_parts() {
                if let Some(output) = super::verify_qmm::forward_candidate_on(x, parts, target)? {
                    return Ok(output);
                }
            }
        }
        match &self.inner {
            LinearImpl::Fp { weight, bias } => {
                let wt = weight.transpose_on(target)?;
                let mut y = x.matmul_on(&wt, target)?;
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
            LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            } => {
                let product_stable = super::product_stable_qmm::is_armed()
                    && x.ndim() >= 2
                    && x.shape().as_slice()[..x.ndim() - 1].iter().product::<i32>() > 1
                    && matches!(*bits, 4 | 5 | 6 | 8)
                    && *mode == QuantMode::Affine;
                let mut y = if product_stable {
                    super::product_stable_qmm::forward_on(
                        x,
                        weight,
                        scales,
                        biases.as_ref(),
                        true,
                        *group_size,
                        *bits,
                        mode.mlx_backend_mode(),
                        target,
                    )?
                } else if super::batch_stable_qmm::linear_is_armed()
                    && x.ndim() == 3
                    && x.shape().as_slice()[0] > 1
                {
                    mlx::quantization::quantized_matmul_batch_isolated_on(
                        x,
                        weight,
                        scales,
                        biases.as_ref(),
                        true,
                        Some(*group_size),
                        Some(*bits),
                        mode.mlx_backend_mode(),
                        target,
                    )?
                } else {
                    mlx::quantization::quantized_matmul_on(
                        x,
                        weight,
                        scales,
                        biases.as_ref(),
                        /* transpose = */ true,
                        Some(*group_size),
                        Some(*bits),
                        mode.mlx_backend_mode(),
                        target,
                    )?
                };
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
        }
    }

    fn forward_fp_positions_isolated_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = x.shape();
        let Some(&[batch, sequence, _]) = <&[i32; 3]>::try_from(shape.as_slice()).ok() else {
            return self.forward_on(x, target);
        };
        if sequence <= 1 {
            return self.forward_on(x, target);
        }
        let mut outputs = Vec::with_capacity(sequence as usize);
        for position in 0..sequence {
            let position_x = x.slice_on(
                [0_i32, position, 0],
                [batch, position + 1, x.shape().as_slice()[2]],
                target,
            )?;
            outputs.push(self.forward_on(&position_x, target)?);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx::ops::concatenate_on(&output_refs, 1, target)?)
    }

    /// Project `[B, Q, K]` as Q independent `[B, K]` matrices, preserving the
    /// quantized matrix shape of a sequential `[B, 1, K]` call at each depth.
    /// Full-precision weights and non-sequence inputs retain the regular path.
    pub(crate) fn forward_positions_isolated_on(
        &self,
        x: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let shape = x.shape();
        let shape = shape.as_slice();
        let Some(&[batch, sequence, _]) = <&[i32; 3]>::try_from(shape).ok() else {
            return self.forward_on(x, target);
        };
        if sequence <= 1 {
            return self.forward_on(x, target);
        }
        let LinearImpl::Quant {
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
            mode,
        } = &self.inner
        else {
            return self.forward_on(x, target);
        };
        if batch == 4 && sequence == 2 && super::position_stable_qmm::exact_affine8_b4_q2_is_armed()
        {
            let parts = QuantizedLinearParts {
                weight,
                scales,
                biases: biases.as_ref(),
                bias: bias.as_ref(),
                group_size: *group_size,
                bits: *bits,
                mode: *mode,
            };
            if let Some(output) =
                super::verify_qmm::forward_affine8_b4_q2_exact_on(x, parts, target)?
            {
                return Ok(output);
            }
        }
        let product_stable =
            batch == 1 && matches!(*bits, 4 | 5 | 6 | 8) && *mode == QuantMode::Affine;
        let mut output = if product_stable {
            super::product_stable_qmm::forward_on(
                x,
                weight,
                scales,
                biases.as_ref(),
                true,
                *group_size,
                *bits,
                mode.mlx_backend_mode(),
                target,
            )?
        } else {
            // Affine8 B2/Q2 produces the same per-position morphology through
            // MLX's native flattened qmv-wide route as the transposed
            // batch-isolated route, while reusing each weight tile across all
            // four vectors. Keep every other qualified shape fail-closed on
            // the established isolated path.
            if batch == 2 && sequence == 2 && *bits == 8 && *mode == QuantMode::Affine {
                mlx::quantization::quantized_matmul_on(
                    x,
                    weight,
                    scales,
                    biases.as_ref(),
                    true,
                    Some(*group_size),
                    Some(*bits),
                    mode.mlx_backend_mode(),
                    target,
                )?
            } else {
                let isolated = x.transpose_axes_on(&[1_i32, 0, 2][..], target)?;
                let output = mlx::quantization::quantized_matmul_batch_isolated_on(
                    &isolated,
                    weight,
                    scales,
                    biases.as_ref(),
                    true,
                    Some(*group_size),
                    Some(*bits),
                    mode.mlx_backend_mode(),
                    target,
                )?;
                output.transpose_axes_on(&[1_i32, 0, 2][..], target)?
            }
        };
        if let Some(bias) = bias {
            output = &output + bias;
        }
        let output_width = output.shape().as_slice()[2];
        debug_assert_eq!(output.shape().as_slice(), &[batch, sequence, output_width]);
        Ok(output)
    }

    /// MTP verify projection for a small batch of speculative positions.
    ///
    /// Eligible affine quantized shapes use the dedicated verify QMM kernel;
    /// all other shapes and full-precision layers retain the standard
    /// [`Linear::forward_on`] path.
    #[doc(hidden)]
    pub fn forward_mtp_verify_on(
        &self,
        x: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if let Some(parts) = self.quantized_parts() {
            if let Some(output) = super::verify_qmm::forward_candidate_on(x, parts, target)? {
                return Ok(output);
            }
        }
        self.forward_on(x, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use mlx::Array;
    use serial_test::serial;

    fn fp_linear(weight: Array, bias: Option<Array>) -> Linear {
        Linear {
            inner: LinearImpl::Fp { weight, bias },
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn fp_forward_matches_manual_matmul() {
        // weight [out=2, in=3] = [[1,2,3],[4,5,6]]
        // x [batch=1, in=3] = [1,1,1]
        // y = x @ W^T = [[1+2+3, 4+5+6]] = [[6, 15]]
        let weight =
            Array::try_from((&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 1.0, 1.0][..], &[1, 3][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_eq!(v, vec![6.0, 15.0]);
        assert_eq!(y.shape().as_slice(), &[1, 2]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn fp_forward_with_bias() {
        // weight = identity [[1,0],[0,1]], x = [3, 4], bias = [10, 20]
        // y = [3*1+4*0, 3*0+4*1] + [10, 20] = [13, 24]
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let bias = Array::try_from((&[10.0f32, 20.0][..], &[2][..])).unwrap();
        let x = Array::try_from((&[3.0f32, 4.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, Some(bias));

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_abs_diff_eq!(v[0], 13.0, epsilon = 1e-6);
        assert_abs_diff_eq!(v[1], 24.0, epsilon = 1e-6);
    }

    #[test]
    #[serial(mlx_metal)]
    fn fp_dtype_preserved() {
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 2.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        assert_eq!(x.dtype(), mlx::Dtype::Float32);
        assert_eq!(y.dtype(), mlx::Dtype::Float32);
    }

    #[test]
    #[serial(mlx_metal)]
    fn position_stable_fp_forward_matches_sequential_q1_shapes() {
        let batch = 4_i32;
        let sequence = 5_i32;
        let out = 32_i32;
        let in_dim = 64_i32;
        let weight_data = (0..(out * in_dim))
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.015)
            .collect::<Vec<_>>();
        let input_data = (0..(batch * sequence * in_dim))
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.025)
            .collect::<Vec<_>>();
        let weight: Array = (weight_data.as_slice(), &[out, in_dim][..])
            .try_into()
            .unwrap();
        let input: Array = (input_data.as_slice(), &[batch, sequence, in_dim][..])
            .try_into()
            .unwrap();
        let weight = mlx::ops::cast::astype(&weight, mlx::Dtype::Bfloat16).unwrap();
        let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();
        let layer = fp_linear(weight, None);

        let mut expected = Vec::with_capacity(sequence as usize);
        for depth in 0..sequence {
            let position = mlx::ops::indexing::slice_strided(
                &input,
                &[0_i32, depth, 0][..],
                &[batch, depth + 1, in_dim][..],
                &[1_i32, 1, 1][..],
            )
            .unwrap();
            expected.push(layer.forward(&position).unwrap());
        }
        let expected_refs = expected.iter().collect::<Vec<_>>();
        let expected = mlx::ops::shape::concatenate(&expected_refs, 1).unwrap();
        let actual = {
            let _scope = crate::nn::position_stable_linear::scope();
            layer.forward(&input).unwrap()
        };
        let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32).unwrap();
        let actual = mlx::ops::cast::astype(&actual, mlx::Dtype::Float32).unwrap();

        assert_eq!(
            expected.to_vec::<f32>().unwrap(),
            actual.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn new_quant_round_trips_via_from_loader_shape() {
        // We cannot construct a real quantized weight from thin air without a
        // tokenizer / safetensors fixture. Instead verify the structural
        // contract: new_quant accepts the 6 fields exactly and stores them in
        // LinearImpl::Quant. Cross-check by inspecting in_features /
        // out_features which compute from the stored shapes.

        // Build a fake quantized weight matching MLX's packed layout for
        // 4-bit, group_size=64: weight shape [out, in/8] u32, scales shape
        // [out, in/64] f32, biases (zero-points) shape [out, in/64] f32.
        let out = 32_i32;
        let in_dim = 64_i32; // single q-group along input axis
        let weight_packed_dim = in_dim / 8; // 4 bits per weight, 8 weights per u32
        let weight_data = vec![0u32; (out * weight_packed_dim) as usize];
        let scales_data = vec![0.01_f32; (out * 1) as usize]; // in/group_size=1
        let weight: Array = (weight_data.as_slice(), &[out, weight_packed_dim][..])
            .try_into()
            .unwrap();
        let scales: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();
        let biases: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();

        let lin = Linear::new_quant(weight, scales, Some(biases), None, 64, 4);

        assert_eq!(lin.in_features(), in_dim as usize);
        assert_eq!(lin.out_features(), out as usize);
    }

    #[test]
    #[serial(mlx_metal)]
    fn fused_affine4_and_affine8_outputs_match_separate_product_stable_projections_exactly() {
        fn make_projection(out: i32, offset: i32, bits: i32) -> Linear {
            let input = 64_i32;
            let raw = (0..out * input)
                .map(|index| (((index + offset) % 31) as f32 - 15.0) * 0.0125)
                .collect::<Vec<_>>();
            let raw: Array = (raw.as_slice(), &[out, input][..]).try_into().unwrap();
            let raw = mlx::ops::cast::astype(&raw, mlx::Dtype::Bfloat16).unwrap();
            let quantized = mlx::quantization::quantize(&raw, Some(64), Some(bits), "affine", None)
                .expect("quantize affine projection");
            Linear::new_quant(
                quantized[0].clone(),
                quantized[1].clone(),
                Some(quantized[2].clone()),
                None,
                64,
                bits,
            )
        }

        for bits in [4, 8] {
            let first = make_projection(16, 0, bits);
            let second = make_projection(8, 7, bits);
            let fused =
                Linear::fuse_quantized_outputs(&[&first, &second], "test fused affine projections")
                    .expect("fuse projections");
            let input = (0..4 * 64)
                .map(|index| ((index % 23) as f32 - 11.0) * 0.02)
                .collect::<Vec<_>>();
            let input: Array = (input.as_slice(), &[1_i32, 4, 64][..]).try_into().unwrap();
            let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();

            let _scope = crate::nn::product_stable_qmm::scope();
            let first_output = first.forward(&input).expect("first projection");
            let second_output = second.forward(&input).expect("second projection");
            let expected =
                mlx::ops::shape::concatenate(&[&first_output, &second_output], -1).unwrap();
            let actual = fused.forward(&input).expect("fused projection");
            let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32).unwrap();
            let actual = mlx::ops::cast::astype(&actual, mlx::Dtype::Float32).unwrap();

            assert_eq!(actual.shape().as_slice(), &[1, 4, 24]);
            assert_eq!(
                expected.to_vec::<f32>().unwrap(),
                actual.to_vec::<f32>().unwrap(),
                "affine{bits} fused projection diverged"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn split_quantized_outputs_are_safe_to_use_on_an_actor_thread() {
        fn make_projection(out: i32, offset: i32, bits: i32) -> Linear {
            let input = 64_i32;
            let raw = (0..out * input)
                .map(|index| (((index + offset) % 31) as f32 - 15.0) * 0.0125)
                .collect::<Vec<_>>();
            let raw: Array = (raw.as_slice(), &[out, input][..]).try_into().unwrap();
            let raw = mlx::ops::cast::astype(&raw, mlx::Dtype::Bfloat16).unwrap();
            let quantized = mlx::quantization::quantize(&raw, Some(64), Some(bits), "affine", None)
                .expect("quantize affine projection");
            Linear::new_quant(
                quantized[0].clone(),
                quantized[1].clone(),
                Some(quantized[2].clone()),
                None,
                64,
                bits,
            )
        }

        for bits in [4, 8] {
            let first = make_projection(16, 0, bits);
            let second = make_projection(8, 7, bits);
            let fused =
                Linear::fuse_quantized_outputs(&[&first, &second], "actor-thread split test")
                    .expect("fuse projections");
            let mut split = fused
                .split_quantized_outputs(&[16, 8], "actor-thread split test")
                .expect("split projections");
            let projection = split.remove(0);

            let output = std::thread::spawn(move || {
                let input = (0..64)
                    .map(|index| ((index % 23) as f32 - 11.0) * 0.02)
                    .collect::<Vec<_>>();
                let input: Array = (input.as_slice(), &[1_i32, 1, 64][..]).try_into().unwrap();
                let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();
                projection
                    .forward(&input)
                    .and_then(|output| {
                        mlx::ops::cast::astype(&output, mlx::Dtype::Float32).map_err(Into::into)
                    })
                    .and_then(|output| output.to_vec::<f32>().map_err(Into::into))
            })
            .join()
            .expect("actor thread must not panic")
            .expect("split projection must execute on actor thread");

            assert_eq!(output.len(), 16, "affine{bits} actor-thread output");
        }
    }

    #[test]
    fn non_power_of_two_affine_widths_recover_exact_input_features() {
        let out = 2_i32;
        let logical_in = 2560_i32;
        for bits in [5_i32, 6_i32] {
            let packed_in = logical_in * bits / 32;
            let weight = Array::zeros((out, packed_in), mlx::Dtype::Uint32).unwrap();
            let scales = Array::zeros((out, logical_in / 64), mlx::Dtype::Bfloat16).unwrap();
            let biases = Array::zeros((out, logical_in / 64), mlx::Dtype::Bfloat16).unwrap();
            let linear = Linear::new_quant(weight, scales, Some(biases), None, 64, bits);

            assert_eq!(linear.in_features(), logical_in as usize);
        }
    }

    fn assert_quantized_forward_matches_mlx(bits: i32, raw_dtype: mlx::Dtype, rows: i32) {
        let out = 3_i32;
        let in_dim = 32_i32;
        let group_size = 32_i32;
        let w_data: Vec<f32> = (0..(out * in_dim))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
            .collect();
        let x_data: Vec<f32> = (0..(rows * in_dim))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        let raw_w_f32: Array = (w_data.as_slice(), &[out, in_dim][..]).try_into().unwrap();
        let x_f32: Array = (x_data.as_slice(), &[rows, in_dim][..]).try_into().unwrap();
        let raw_w = mlx::ops::cast::astype(&raw_w_f32, raw_dtype).unwrap();
        let x = mlx::ops::cast::astype(&x_f32, raw_dtype).unwrap();
        let q = mlx::quantization::quantize(&raw_w, Some(group_size), Some(bits), "affine", None)
            .unwrap();

        let layer = Linear::new_quant(
            q[0].clone(),
            q[1].clone(),
            Some(q[2].clone()),
            None,
            group_size,
            bits,
        );
        let got = layer.forward(&x).unwrap();
        let expected = mlx::quantization::quantized_matmul(
            &x,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            "affine",
        )
        .unwrap();

        let got = mlx::ops::cast::astype(&got, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= 0.001, "idx={idx} got={g} expected={e}");
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_8bit_forward_matches_mlx_bfloat16() {
        assert_quantized_forward_matches_mlx(8, mlx::Dtype::Bfloat16, 2);
    }

    #[test]
    #[serial(mlx_metal)]
    fn batch_stable_quantized_forward_matches_single_row_shape() {
        let out = 32_i32;
        let in_dim = 64_i32;
        let group_size = 64_i32;
        let rows = 3_i32;
        let weight_data = (0..(out * in_dim))
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.015)
            .collect::<Vec<_>>();
        let input_data = (0..(rows * in_dim))
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.025)
            .collect::<Vec<_>>();
        let weight: Array = (weight_data.as_slice(), &[out, in_dim][..])
            .try_into()
            .unwrap();
        let input: Array = (input_data.as_slice(), &[1_i32, rows, in_dim][..])
            .try_into()
            .unwrap();
        let quantized =
            mlx::quantization::quantize(&weight, Some(group_size), Some(4), "affine", None)
                .unwrap();
        let layer = Linear::new_quant(
            quantized[0].clone(),
            quantized[1].clone(),
            Some(quantized[2].clone()),
            None,
            group_size,
            4,
        );
        let expected = layer.forward(&input).unwrap().to_vec::<f32>().unwrap();
        let batch = mlx::ops::shape::concatenate(&[&input, &input, &input, &input], 0).unwrap();
        let actual = {
            let _scope = crate::nn::batch_stable_qmm::linear_scope();
            layer.forward(&batch).unwrap().to_vec::<f32>().unwrap()
        };

        assert_eq!(actual.len(), expected.len() * 4);
        for row in actual.chunks_exact(expected.len()) {
            assert_eq!(row, expected.as_slice());
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn product_stable_affine8_q1_matches_single_row_shape() {
        let out = 64_i32;
        let in_dim = 128_i32;
        let group_size = 64_i32;
        let weight_data = (0..(out * in_dim))
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0125)
            .collect::<Vec<_>>();
        let input_data = (0..in_dim)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.02)
            .collect::<Vec<_>>();
        let weight: Array = (weight_data.as_slice(), &[out, in_dim][..])
            .try_into()
            .unwrap();
        let input: Array = (input_data.as_slice(), &[1_i32, 1_i32, in_dim][..])
            .try_into()
            .unwrap();
        let weight = mlx::ops::cast::astype(&weight, mlx::Dtype::Bfloat16).unwrap();
        let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();
        let quantized =
            mlx::quantization::quantize(&weight, Some(group_size), Some(8), "affine", None)
                .unwrap();
        let layer = Linear::new_quant(
            quantized[0].clone(),
            quantized[1].clone(),
            Some(quantized[2].clone()),
            None,
            group_size,
            8,
        );
        let expected = mlx::ops::cast::astype(&layer.forward(&input).unwrap(), mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        for batch in [2_i32, 4] {
            let rows = std::iter::repeat_n(&input, batch as usize).collect::<Vec<_>>();
            let input = mlx::ops::shape::concatenate(&rows, 0).unwrap();
            let actual = {
                let _scope = crate::nn::product_stable_qmm::scope();
                layer.forward(&input).unwrap()
            };
            let actual = mlx::ops::cast::astype(&actual, mlx::Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            for row in actual.chunks_exact(expected.len()) {
                assert_eq!(row, expected.as_slice(), "B{batch}");
            }
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn position_stable_quantized_forward_matches_sequential_q1_shapes() {
        let sequence = 5_i32;
        let out = 32_i32;
        let in_dim = 64_i32;
        let group_size = 64_i32;
        let weight_data = (0..(out * in_dim))
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.015)
            .collect::<Vec<_>>();
        let weight: Array = (weight_data.as_slice(), &[out, in_dim][..])
            .try_into()
            .unwrap();
        let weight = mlx::ops::cast::astype(&weight, mlx::Dtype::Bfloat16).unwrap();
        for batch in [1_i32, 4] {
            let input_data = (0..(batch * sequence * in_dim))
                .map(|idx| ((idx % 19) as f32 - 9.0) * 0.025)
                .collect::<Vec<_>>();
            let input: Array = (input_data.as_slice(), &[batch, sequence, in_dim][..])
                .try_into()
                .unwrap();
            let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();
            let bit_widths: &[i32] = if batch == 1 { &[4, 5, 6, 8] } else { &[8] };
            for &bits in bit_widths {
                let quantized = mlx::quantization::quantize(
                    &weight,
                    Some(group_size),
                    Some(bits),
                    "affine",
                    None,
                )
                .unwrap();
                let layer = Linear::new_quant(
                    quantized[0].clone(),
                    quantized[1].clone(),
                    Some(quantized[2].clone()),
                    None,
                    group_size,
                    bits,
                );

                let mut expected = Vec::with_capacity(sequence as usize);
                for depth in 0..sequence {
                    let position = mlx::ops::indexing::slice_strided(
                        &input,
                        &[0_i32, depth, 0][..],
                        &[batch, depth + 1, in_dim][..],
                        &[1_i32, 1, 1][..],
                    )
                    .unwrap();
                    expected.push(layer.forward(&position).unwrap());
                }
                let expected_refs = expected.iter().collect::<Vec<_>>();
                let expected = mlx::ops::shape::concatenate(&expected_refs, 1).unwrap();
                let actual = {
                    let _scope = crate::nn::position_stable_qmm::scope();
                    layer.forward(&input).unwrap()
                };

                let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let actual = mlx::ops::cast::astype(&actual, mlx::Dtype::Float32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                assert_eq!(actual, expected, "batch={batch} bits={bits}");
            }
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn affine8_b2q2_native_matches_batch_isolated_exactly() {
        let (batch, sequence, out, in_dim, group_size) = (2_i32, 2_i32, 256_i32, 512_i32, 64_i32);
        let weight_data = (0..(out * in_dim))
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0125)
            .collect::<Vec<_>>();
        let input_data = (0..(batch * sequence * in_dim))
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.02)
            .collect::<Vec<_>>();
        let weight: Array = (weight_data.as_slice(), &[out, in_dim][..])
            .try_into()
            .unwrap();
        let input: Array = (input_data.as_slice(), &[batch, sequence, in_dim][..])
            .try_into()
            .unwrap();
        let weight = mlx::ops::cast::astype(&weight, mlx::Dtype::Bfloat16).unwrap();
        let input = mlx::ops::cast::astype(&input, mlx::Dtype::Bfloat16).unwrap();
        let quantized =
            mlx::quantization::quantize(&weight, Some(group_size), Some(8), "affine", None)
                .unwrap();
        let native = mlx::quantization::quantized_matmul(
            &input,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(group_size),
            Some(8),
            "affine",
        )
        .unwrap();
        let isolated_input = input.transpose_axes(&[1_i32, 0, 2][..]).unwrap();
        let isolated = mlx::quantization::quantized_matmul_batch_isolated(
            &isolated_input,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(group_size),
            Some(8),
            "affine",
        )
        .unwrap()
        .transpose_axes(&[1_i32, 0, 2][..])
        .unwrap();

        let native = mlx::ops::cast::astype(&native, mlx::Dtype::Float32).unwrap();
        let isolated = mlx::ops::cast::astype(&isolated, mlx::Dtype::Float32).unwrap();
        assert_eq!(
            native.to_vec::<f32>().unwrap(),
            isolated.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_2bit_forward_matches_mlx_float32() {
        assert_quantized_forward_matches_mlx(2, mlx::Dtype::Float32, 2);
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_5bit_and_6bit_forward_match_mlx_bfloat16() {
        for bits in [5, 6] {
            for rows in [1, 64] {
                assert_quantized_forward_matches_mlx(bits, mlx::Dtype::Bfloat16, rows);
            }
        }
    }

    fn assert_mxfp_forward_matches_mlx(mode: QuantMode, bits: i32) {
        let out = 3_i32;
        let in_dim = 32_i32;
        let group_size = 32_i32;
        let w_data: Vec<f32> = (0..(out * in_dim))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
            .collect();
        let x_data: Vec<f32> = (0..(2 * in_dim))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        let raw_w_f32: Array = (w_data.as_slice(), &[out, in_dim][..]).try_into().unwrap();
        let x_f32: Array = (x_data.as_slice(), &[2_i32, in_dim][..])
            .try_into()
            .unwrap();
        let raw_w = mlx::ops::cast::astype(&raw_w_f32, mlx::Dtype::Bfloat16).unwrap();
        let x = mlx::ops::cast::astype(&x_f32, mlx::Dtype::Bfloat16).unwrap();
        let q = mlx::quantization::quantize(
            &raw_w,
            Some(group_size),
            Some(bits),
            mode.mlx_backend_mode(),
            None,
        )
        .unwrap();
        assert_eq!(q.len(), 2, "MXFP quantization returns weight and scales");

        let layer = Linear::new_quant_with_mode(
            q[0].clone(),
            q[1].clone(),
            None,
            None,
            group_size,
            bits,
            mode,
        );
        let got = layer.forward(&x).unwrap();
        let expected = mlx::quantization::quantized_matmul(
            &x,
            &q[0],
            &q[1],
            None,
            true,
            Some(group_size),
            Some(bits),
            mode.mlx_backend_mode(),
        )
        .unwrap();

        assert_eq!(got.dtype(), mlx::Dtype::Bfloat16);
        let got = mlx::ops::cast::astype(&got, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= 0.001, "idx={idx} got={g} expected={e}");
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn mxfp4_forward_matches_native_mlx() {
        assert_mxfp_forward_matches_mlx(QuantMode::Mxfp4, 4);
    }

    #[test]
    #[serial(mlx_metal)]
    fn mxfp8_forward_matches_native_mlx() {
        assert_mxfp_forward_matches_mlx(QuantMode::Mxfp8, 8);
    }

    #[test]
    #[serial(mlx_metal)]
    fn optiq_quantized_forward_uses_independent_mode_with_affine_backend() {
        let out = 3_i32;
        let in_dim = 64_i32;
        let group_size = 64_i32;
        let bits = 4_i32;
        let w_data: Vec<f32> = (0..(out * in_dim))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.015)
            .collect();
        let x_data: Vec<f32> = (0..(2 * in_dim))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.025)
            .collect();
        let raw_w: Array = (w_data.as_slice(), &[out, in_dim][..]).try_into().unwrap();
        let x: Array = (x_data.as_slice(), &[2_i32, in_dim][..])
            .try_into()
            .unwrap();
        let q = mlx::quantization::quantize(&raw_w, Some(group_size), Some(bits), "affine", None)
            .unwrap();

        let layer = Linear::new_quant_with_mode(
            q[0].clone(),
            q[1].clone(),
            Some(q[2].clone()),
            None,
            group_size,
            bits,
            QuantMode::OptiQ,
        );
        let got = layer.forward(&x).unwrap();
        let expected = mlx::quantization::quantized_matmul(
            &x,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            "affine",
        )
        .unwrap();

        let got = mlx::ops::cast::astype(&got, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = mlx::ops::cast::astype(&expected, mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= 0.001, "idx={idx} got={g} expected={e}");
        }
    }
}
