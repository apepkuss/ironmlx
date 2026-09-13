//! Model-specific optimized projection over shared linear parameters.
use crate::core::weights::{QuantMode, WeightSource};
use crate::Result;
use anyhow::anyhow;
pub(crate) use ironmlx_core::nn::linear::QuantizedLinearParts;
use mlx::{Array, StreamOrDevice};

pub struct Linear {
    inner: ironmlx_core::nn::Linear,
}
impl Linear {
    pub fn from_loader(loader: &(impl WeightSource + ?Sized), prefix: &str) -> Result<Self> {
        Ok(Self {
            inner: ironmlx_core::nn::Linear::from_loader(loader, prefix)?,
        })
    }
    pub fn new_fp(weight: Array, bias: Option<Array>) -> Self {
        Self {
            inner: ironmlx_core::nn::Linear::new_fp(weight, bias),
        }
    }
    pub fn new_quant(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            inner: ironmlx_core::nn::Linear::new_quant(
                weight, scales, biases, bias, group_size, bits,
            ),
        }
    }
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
            inner: ironmlx_core::nn::Linear::new_quant_with_mode(
                weight, scales, biases, bias, group_size, bits, mode,
            ),
        }
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }
    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }
    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }
    pub(crate) fn quantized_parts(&self) -> Option<QuantizedLinearParts<'_>> {
        self.inner.quantized_parts()
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
            && self.quantized_parts().is_none()
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
        match self.quantized_parts() {
            None => self.inner.forward_on(x, target),
            Some(QuantizedLinearParts {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            }) => {
                let product_stable = super::product_stable_qmm::is_armed()
                    && x.ndim() >= 2
                    && x.shape().as_slice()[..x.ndim() - 1].iter().product::<i32>() > 1
                    && matches!(bits, 4 | 5 | 6 | 8)
                    && mode == QuantMode::Affine;
                let mut y = if product_stable {
                    super::product_stable_qmm::forward_on(
                        x,
                        weight,
                        scales,
                        biases,
                        true,
                        group_size,
                        bits,
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
                        biases,
                        true,
                        Some(group_size),
                        Some(bits),
                        mode.mlx_backend_mode(),
                        target,
                    )?
                } else {
                    return self.inner.forward_on(x, target);
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
        let Some(QuantizedLinearParts {
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
            mode,
        }) = self.quantized_parts()
        else {
            return self.forward_on(x, target);
        };
        if batch == 4 && sequence == 2 && super::position_stable_qmm::exact_affine8_b4_q2_is_armed()
        {
            let parts = QuantizedLinearParts {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            };
            if let Some(output) =
                super::verify_qmm::forward_affine8_b4_q2_exact_on(x, parts, target)?
            {
                return Ok(output);
            }
        }
        let product_stable =
            batch == 1 && matches!(bits, 4 | 5 | 6 | 8) && mode == QuantMode::Affine;
        let mut output = if product_stable {
            super::product_stable_qmm::forward_on(
                x,
                weight,
                scales,
                biases,
                true,
                group_size,
                bits,
                mode.mlx_backend_mode(),
                target,
            )?
        } else {
            // Affine8 B2/Q2 produces the same per-position morphology through
            // MLX's native flattened qmv-wide route as the transposed
            // batch-isolated route, while reusing each weight tile across all
            // four vectors. Keep every other qualified shape fail-closed on
            // the established isolated path.
            if batch == 2 && sequence == 2 && bits == 8 && mode == QuantMode::Affine {
                mlx::quantization::quantized_matmul_on(
                    x,
                    weight,
                    scales,
                    biases,
                    true,
                    Some(group_size),
                    Some(bits),
                    mode.mlx_backend_mode(),
                    target,
                )?
            } else {
                let isolated = x.transpose_axes_on(&[1_i32, 0, 2][..], target)?;
                let output = mlx::quantization::quantized_matmul_batch_isolated_on(
                    &isolated,
                    weight,
                    scales,
                    biases,
                    true,
                    Some(group_size),
                    Some(bits),
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
    use mlx::Array;
    use serial_test::serial;

    fn fp_linear(weight: Array, bias: Option<Array>) -> Linear {
        Linear::new_fp(weight, bias)
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
}
