use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};
use serde::Deserialize;

use ironmlx_core::nn::activations::gelu_tanh;
use ironmlx_lm::nn::{Linear, RmsNorm};

use crate::loader::ComponentLoader;
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct QwenImage21TransformerConfig {
    pub attention_head_dim: i32,
    pub axes_dims_rope: Vec<i32>,
    pub context_in_dim: i32,
    pub in_channels: i32,
    pub num_attention_heads: i32,
    pub num_layers: i32,
    pub out_channels: i32,
    pub mlp_ratio: i32,
    pub eps: f32,
    pub causal_condition: bool,
}

impl QwenImage21TransformerConfig {
    fn from_loader(loader: &ComponentLoader) -> Result<Self> {
        let config: Self = loader.config()?;
        if config.attention_head_dim != 128
            || config.axes_dims_rope != [16, 56, 56]
            || config.context_in_dim != 4096
            || config.in_channels != 64
            || config.num_attention_heads != 32
            || config.num_layers != 32
            || config.out_channels != 64
            || config.mlp_ratio != 3
            || !config.causal_condition
        {
            return Err(anyhow!("unsupported Qwen Image 2.1 transformer config"));
        }
        Ok(config)
    }
}

fn silu_on(value: &Array, target: StreamOrDevice) -> Result<Array> {
    Ok(value * &value.sigmoid_on(target)?)
}

fn affine_free_layer_norm_on(value: &Array, eps: f32, target: StreamOrDevice) -> Result<Array> {
    Ok(mlx::fast::layer_norm_on(value, None, None, eps, target)?)
}

fn slice_axis_1(value: &Array, start: i32, stop: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = value.shape();
    let dims = shape.as_slice();
    Ok(mlx::ops::indexing::slice_on(
        value,
        &[0_i32, start, 0][..],
        &[dims[0], stop, dims[2]][..],
        target,
    )?)
}

#[derive(Debug)]
struct AttentionSegment {
    start: i32,
    end: i32,
    is_text: bool,
}

struct JointLayout {
    gather_indices: Array,
    target_token_mask: Vec<bool>,
    segments: Vec<AttentionSegment>,
    prefix_length: i32,
    cos: Array,
    sin: Array,
}

struct QwenImage21Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QwenImage21Attention {
    fn from_loader(
        loader: &ComponentLoader,
        prefix: &str,
        config: &QwenImage21TransformerConfig,
    ) -> Result<Self> {
        Ok(Self {
            to_q: Linear::from_loader(loader, &format!("{prefix}.to_q"))?,
            to_k: Linear::from_loader(loader, &format!("{prefix}.to_k"))?,
            to_v: Linear::from_loader(loader, &format!("{prefix}.to_v"))?,
            to_out: Linear::from_loader(loader, &format!("{prefix}.to_out.0"))?,
            norm_q: RmsNorm::from_loader(loader, &format!("{prefix}.norm_q"), config.eps)?,
            norm_k: RmsNorm::from_loader(loader, &format!("{prefix}.norm_k"), config.eps)?,
            heads: config.num_attention_heads,
            head_dim: config.attention_head_dim,
            scale: 1.0 / (config.attention_head_dim as f32).sqrt(),
        })
    }

    fn apply_rope_on(
        &self,
        value: &Array,
        cos: &Array,
        sin: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = value.shape();
        let dims = shape.as_slice();
        let (batch, sequence) = (dims[0], dims[1]);
        let pairs = value.astype_on(Dtype::Float32, target)?.reshape_on(
            &[batch, sequence, self.heads, self.head_dim / 2, 2_i32][..],
            target,
        )?;
        let real = mlx::ops::indexing::slice_on(
            &pairs,
            &[0_i32, 0, 0, 0, 0][..],
            &[batch, sequence, self.heads, self.head_dim / 2, 1][..],
            target,
        )?
        .reshape_on((batch, sequence, self.heads, self.head_dim / 2), target)?;
        let imaginary = mlx::ops::indexing::slice_on(
            &pairs,
            &[0_i32, 0, 0, 0, 1][..],
            &[batch, sequence, self.heads, self.head_dim / 2, 2][..],
            target,
        )?
        .reshape_on((batch, sequence, self.heads, self.head_dim / 2), target)?;
        let cos = cos.reshape_on((1_i32, sequence, 1_i32, self.head_dim / 2), target)?;
        let sin = sin.reshape_on((1_i32, sequence, 1_i32, self.head_dim / 2), target)?;
        let out_real = &real * &cos - &imaginary * &sin;
        let out_imaginary = &real * &sin + &imaginary * &cos;
        let paired = mlx::ops::shape::stack_on(&[&out_real, &out_imaginary], -1, target)?;
        Ok(paired
            .reshape_on((batch, sequence, self.heads, self.head_dim), target)?
            .astype_on(value.dtype(), target)?)
    }

    fn forward_on(
        &self,
        hidden: &Array,
        segments: &[AttentionSegment],
        prefix_length: i32,
        cos: &Array,
        sin: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        let (batch, sequence) = (dims[0], dims[1]);
        let project = |linear: &Linear| -> Result<Array> {
            Ok(linear
                .forward_on(hidden, target)?
                .reshape_on((batch, sequence, self.heads, self.head_dim), target)?)
        };
        let query = self.norm_q.forward_on(&project(&self.to_q)?, target)?;
        let key = self.norm_k.forward_on(&project(&self.to_k)?, target)?;
        let value = project(&self.to_v)?;
        let query = self
            .apply_rope_on(&query, cos, sin, target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let key = self
            .apply_rope_on(&key, cos, sin, target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        let value = value.transpose_axes_on(&[0, 2, 1, 3][..], target)?;

        let mut outputs = Vec::with_capacity(segments.len() + 1);
        for segment in segments {
            let segment_query = mlx::ops::indexing::slice_on(
                &query,
                &[0_i32, 0, segment.start, 0][..],
                &[batch, self.heads, segment.end, self.head_dim][..],
                target,
            )?;
            let segment_key = mlx::ops::indexing::slice_on(
                &key,
                &[0_i32, 0, 0, 0][..],
                &[batch, self.heads, segment.end, self.head_dim][..],
                target,
            )?;
            let segment_value = mlx::ops::indexing::slice_on(
                &value,
                &[0_i32, 0, 0, 0][..],
                &[batch, self.heads, segment.end, self.head_dim][..],
                target,
            )?;
            outputs.push(mlx::fast::scaled_dot_product_attention_on(
                &segment_query,
                &segment_key,
                &segment_value,
                self.scale,
                if segment.is_text { "causal" } else { "" },
                None,
                None,
                target,
            )?);
        }
        let target_query = mlx::ops::indexing::slice_on(
            &query,
            &[0_i32, 0, prefix_length, 0][..],
            &[batch, self.heads, sequence, self.head_dim][..],
            target,
        )?;
        outputs.push(mlx::fast::scaled_dot_product_attention_on(
            &target_query,
            &key,
            &value,
            self.scale,
            "",
            None,
            None,
            target,
        )?);
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = mlx::ops::shape::concatenate_on(&output_refs, 2, target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((batch, sequence, self.heads * self.head_dim), target)?;
        self.to_out.forward_on(&output, target)
    }
}

struct QwenImage21Mlp {
    projection: Linear,
    gate: Linear,
    output: Linear,
}

impl QwenImage21Mlp {
    fn from_loader(loader: &ComponentLoader, prefix: &str) -> Result<Self> {
        Ok(Self {
            projection: Linear::from_loader(loader, &format!("{prefix}.proj"))?,
            gate: Linear::from_loader(loader, &format!("{prefix}.gate_layer"))?,
            output: Linear::from_loader(loader, &format!("{prefix}.out"))?,
        })
    }

    fn forward_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        let gate = silu_on(&self.gate.forward_on(hidden, target)?, target)?;
        self.output.forward_on(
            &(&gate * &self.projection.forward_on(hidden, target)?),
            target,
        )
    }
}

struct QwenImage21Block {
    attention: QwenImage21Attention,
    mlp: QwenImage21Mlp,
    eps: f32,
}

impl QwenImage21Block {
    fn from_loader(
        loader: &ComponentLoader,
        index: i32,
        config: &QwenImage21TransformerConfig,
    ) -> Result<Self> {
        let prefix = format!("transformer_blocks.{index}");
        Ok(Self {
            attention: QwenImage21Attention::from_loader(
                loader,
                &format!("{prefix}.attn"),
                config,
            )?,
            mlp: QwenImage21Mlp::from_loader(loader, &format!("{prefix}.img_mlp"))?,
            eps: config.eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_on(
        &self,
        hidden: &Array,
        scale1: &Array,
        gate1: &Array,
        scale2: &Array,
        gate2: &Array,
        segments: &[AttentionSegment],
        prefix_length: i32,
        cos: &Array,
        sin: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let normed = affine_free_layer_norm_on(hidden, self.eps, target)? * (scale1 + 1.0_f32);
        let attention =
            self.attention
                .forward_on(&normed, segments, prefix_length, cos, sin, target)?;
        let residual = hidden + &(&gate1.tanh_on(target)? * &attention);
        let normed = affine_free_layer_norm_on(&residual, self.eps, target)? * (scale2 + 1.0_f32);
        let mlp = self.mlp.forward_on(&normed, target)?;
        Ok(&residual + &(&gate2.tanh_on(target)? * &mlp))
    }
}

pub struct QwenImage21Transformer {
    config: QwenImage21TransformerConfig,
    text_norm: RmsNorm,
    text_in: Linear,
    text_out: Linear,
    image_in: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    modulation: Linear,
    blocks: Vec<QwenImage21Block>,
    norm_out: Linear,
    projection_out: Linear,
}

impl QwenImage21Transformer {
    pub fn from_loader(loader: &ComponentLoader) -> Result<Self> {
        let config = QwenImage21TransformerConfig::from_loader(loader)?;
        let shifted_text_norm = loader.tensor("txt_in.text_norm.weight")? + 1.0_f32;
        mlx::transforms::eval(&[&shifted_text_norm])?;
        let text_norm = RmsNorm::new(shifted_text_norm, config.eps);
        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for index in 0..config.num_layers {
            blocks.push(
                QwenImage21Block::from_loader(loader, index, &config)
                    .with_context(|| format!("loading Qwen Image 2.1 transformer block {index}"))?,
            );
        }
        Ok(Self {
            config,
            text_norm,
            text_in: Linear::from_loader(loader, "txt_in.in_layer")?,
            text_out: Linear::from_loader(loader, "txt_in.out_layer")?,
            image_in: Linear::from_loader(loader, "img_in")?,
            time_linear_1: Linear::from_loader(loader, "time_text_embed.linear_1")?,
            time_linear_2: Linear::from_loader(loader, "time_text_embed.linear_2")?,
            // The target MLX checkpoint stores the shared projection at
            // modulation.0.*, not the stale modulation.1.* mapping used by
            // older mflux releases.
            modulation: Linear::from_loader(loader, "modulation.0")?,
            blocks,
            norm_out: Linear::from_loader(loader, "norm_out.linear")?,
            projection_out: Linear::from_loader(loader, "proj_out")?,
        })
    }

    pub fn config(&self) -> &QwenImage21TransformerConfig {
        &self.config
    }

    fn timestep_embedding_on(&self, sigma: f32, target: StreamOrDevice) -> Result<Array> {
        let half = 128_usize;
        let frequencies = (0..half)
            .map(|index| (-10_000.0_f32.ln() * index as f32 / half as f32).exp())
            .collect::<Vec<_>>();
        let values = [sigma, 0.0_f32]
            .into_iter()
            .flat_map(|row| {
                let args = frequencies
                    .iter()
                    .map(|frequency| row * 1000.0 * frequency)
                    .collect::<Vec<_>>();
                args.iter()
                    .map(|value| value.cos())
                    .chain(args.iter().map(|value| value.sin()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let embedding: Array = (values.as_slice(), (2_i32, 256_i32)).try_into()?;
        let embedding = embedding.astype_on(Dtype::Bfloat16, target)?;
        let hidden = self.time_linear_1.forward_on(&embedding, target)?;
        self.time_linear_2
            .forward_on(&silu_on(&hidden, target)?, target)
    }

    fn joint_layout(
        &self,
        encoder_image_mask: &[bool],
        condition_shapes: &[(i32, i32)],
        target_height: i32,
        target_width: i32,
    ) -> Result<JointLayout> {
        let encoder_length = i32::try_from(encoder_image_mask.len())?;
        let condition_tokens =
            condition_shapes
                .iter()
                .try_fold(0_i32, |total, (height, width)| {
                    height
                        .checked_mul(*width)
                        .and_then(|tokens| total.checked_add(tokens))
                        .ok_or_else(|| anyhow!("Qwen Image 2.1 condition geometry overflow"))
                })?;
        let condition_slots = i32::try_from(
            encoder_image_mask
                .iter()
                .filter(|is_image| **is_image)
                .count(),
        )?;
        if condition_slots * 4 != condition_tokens {
            return Err(anyhow!(
                "Qwen Image 2.1 encoder mask has {condition_slots} image slots but condition geometry has {condition_tokens} latent tokens"
            ));
        }
        let target_tokens = target_height
            .checked_mul(target_width)
            .ok_or_else(|| anyhow!("Qwen Image 2.1 target geometry overflow"))?;
        if target_tokens % 4 != 0 {
            return Err(anyhow!(
                "Qwen Image 2.1 target latent token count must be divisible by four"
            ));
        }

        let capacity = usize::try_from(encoder_length + condition_slots * 3 + target_tokens)?;
        let mut gather_indices = Vec::with_capacity(capacity);
        let mut image_pad_mask = Vec::with_capacity(capacity);
        let mut condition_cursor = 0_i32;
        for (position, is_image) in encoder_image_mask.iter().copied().enumerate() {
            if is_image {
                for offset in 0..4_i32 {
                    gather_indices.push(u32::try_from(encoder_length + condition_cursor + offset)?);
                    image_pad_mask.push(true);
                }
                condition_cursor += 4;
            } else {
                gather_indices.push(u32::try_from(position)?);
                image_pad_mask.push(false);
            }
        }
        if condition_cursor != condition_tokens {
            return Err(anyhow!("Qwen Image 2.1 condition slot expansion mismatch"));
        }
        for offset in 0..target_tokens {
            gather_indices.push(u32::try_from(encoder_length + condition_tokens + offset)?);
            image_pad_mask.push(true);
        }

        let mut shapes = condition_shapes.to_vec();
        shapes.push((target_height, target_width));
        let image_positions = image_pad_mask
            .iter()
            .enumerate()
            .filter_map(|(position, is_image)| is_image.then_some(position))
            .collect::<Vec<_>>();
        let expected_image_tokens = shapes.iter().map(|(h, w)| h * w).sum::<i32>();
        if usize::try_from(expected_image_tokens)? != image_positions.len() {
            return Err(anyhow!("Qwen Image 2.1 joint image-token layout mismatch"));
        }
        let mut image_ids = vec![-1_i32; image_pad_mask.len()];
        let mut image_cursor = 0_usize;
        for (image_id, (height, width)) in shapes.iter().copied().enumerate() {
            let block_tokens = usize::try_from(height * width)?;
            for &position in &image_positions[image_cursor..image_cursor + block_tokens] {
                image_ids[position] = i32::try_from(image_id)?;
            }
            image_cursor += block_tokens;
        }

        let prefix_length = i32::try_from(image_pad_mask.len())? - target_tokens;
        let mut target_token_mask = vec![false; image_pad_mask.len()];
        target_token_mask[usize::try_from(prefix_length)?..].fill(true);
        let mut segments = Vec::new();
        let prefix = usize::try_from(prefix_length)?;
        if prefix > 0 {
            let mut start = 0_usize;
            for end in 1..=prefix {
                if end == prefix || image_ids[end] != image_ids[start] {
                    segments.push(AttentionSegment {
                        start: i32::try_from(start)?,
                        end: i32::try_from(end)?,
                        is_text: image_ids[start] < 0,
                    });
                    start = end;
                }
            }
        }

        let (cos, sin) = self.geometry(&image_pad_mask, &shapes)?;
        let gather_indices: Array = (
            gather_indices.as_slice(),
            (i32::try_from(gather_indices.len())?,),
        )
            .try_into()?;
        Ok(JointLayout {
            gather_indices,
            target_token_mask,
            segments,
            prefix_length,
            cos,
            sin,
        })
    }

    fn geometry(&self, image_pad_mask: &[bool], shapes: &[(i32, i32)]) -> Result<(Array, Array)> {
        let mut axes = [Vec::new(), Vec::new(), Vec::new()];
        let mut cursor = 0_usize;
        let mut position = 0_i32;
        for &(height, width) in shapes {
            let block_start = image_pad_mask[cursor..]
                .iter()
                .position(|is_image| *is_image)
                .map(|offset| cursor + offset)
                .ok_or_else(|| anyhow!("Qwen Image 2.1 image block missing from joint layout"))?;
            let text_length = i32::try_from(block_start - cursor)?;
            for text_position in position..position + text_length {
                for axis in &mut axes {
                    axis.push(text_position);
                }
            }
            position += text_length;
            let image_tokens = usize::try_from(height * width)?;
            axes[0].extend(std::iter::repeat_n(position, image_tokens));
            for row in -(height - height / 2)..height / 2 {
                axes[1].extend(std::iter::repeat_n(row, width as usize));
            }
            for _ in 0..height {
                axes[2].extend(-(width - width / 2)..width / 2);
            }
            cursor = block_start + image_tokens;
            position += height.max(width);
        }
        let trailing_text = i32::try_from(image_pad_mask.len() - cursor)?;
        for text_position in position..position + trailing_text {
            for axis in &mut axes {
                axis.push(text_position);
            }
        }
        if axes.iter().any(|axis| axis.len() != image_pad_mask.len()) {
            return Err(anyhow!("Qwen Image 2.1 rotary geometry length mismatch"));
        }

        let mut cos = Vec::with_capacity(image_pad_mask.len() * 64);
        let mut sin = Vec::with_capacity(image_pad_mask.len() * 64);
        for ((axis_0, axis_1), axis_2) in axes[0].iter().zip(&axes[1]).zip(&axes[2]) {
            let positions = [*axis_0, *axis_1, *axis_2];
            for (axis_position, dim) in positions.iter().zip(&self.config.axes_dims_rope) {
                for index in (0..*dim).step_by(2) {
                    let frequency = 1.0 / 10_000.0_f32.powf(index as f32 / *dim as f32);
                    let angle = *axis_position as f32 * frequency;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let sequence = i32::try_from(image_pad_mask.len())?;
        Ok((
            (cos.as_slice(), (sequence, 64_i32)).try_into()?,
            (sin.as_slice(), (sequence, 64_i32)).try_into()?,
        ))
    }

    fn selected_modulation_on(
        &self,
        value: &Array,
        target_token_mask: &[bool],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let row_indices = target_token_mask
            .iter()
            .map(|is_target| if *is_target { 0_u32 } else { 1_u32 })
            .collect::<Vec<_>>();
        let row_indices: Array =
            (row_indices.as_slice(), (i32::try_from(row_indices.len())?,)).try_into()?;
        Ok(
            mlx::ops::indexing::take_on(value, &row_indices, 0, target)?.reshape_on(
                &[
                    1_i32,
                    i32::try_from(target_token_mask.len())?,
                    value.shape().as_slice()[1],
                ][..],
                target,
            )?,
        )
    }

    pub fn forward(
        &self,
        latents: &Array,
        encoder_hidden: &Array,
        sigma: f32,
        latent_height: i32,
        latent_width: i32,
    ) -> Result<Array> {
        self.forward_on(
            latents,
            encoder_hidden,
            sigma,
            latent_height,
            latent_width,
            (),
        )
    }

    pub fn forward_on(
        &self,
        latents: &Array,
        encoder_hidden: &Array,
        sigma: f32,
        latent_height: i32,
        latent_width: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let encoder_length = usize::try_from(encoder_hidden.shape().as_slice()[1])?;
        self.forward_joint_on(
            latents,
            None,
            encoder_hidden,
            &vec![false; encoder_length],
            sigma,
            &[],
            latent_height,
            latent_width,
            target.into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_conditioned(
        &self,
        target_latents: &Array,
        condition_latents: &Array,
        encoder_hidden: &Array,
        encoder_image_mask: &[bool],
        sigma: f32,
        condition_height: i32,
        condition_width: i32,
        target_height: i32,
        target_width: i32,
    ) -> Result<Array> {
        self.forward_joint_on(
            target_latents,
            Some(condition_latents),
            encoder_hidden,
            encoder_image_mask,
            sigma,
            &[(condition_height, condition_width)],
            target_height,
            target_width,
            StreamOrDevice::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_joint_on(
        &self,
        target_latents: &Array,
        condition_latents: Option<&Array>,
        encoder_hidden: &Array,
        encoder_image_mask: &[bool],
        sigma: f32,
        condition_shapes: &[(i32, i32)],
        target_height: i32,
        target_width: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let encoder_dims = encoder_hidden.shape();
        let encoder_dims = encoder_dims.as_slice();
        if encoder_dims.len() != 3
            || encoder_dims[0] != 1
            || encoder_dims[2] != self.config.context_in_dim
            || encoder_image_mask.len() != usize::try_from(encoder_dims[1])?
        {
            return Err(anyhow!(
                "Qwen Image 2.1 encoder hidden/mask layout is invalid"
            ));
        }
        let target_tokens = target_height * target_width;
        if target_latents.shape().as_slice() != [1, target_tokens, self.config.in_channels] {
            return Err(anyhow!(
                "Qwen Image 2.1 latent shape {:?} does not match geometry {}x{}",
                target_latents.shape().as_slice(),
                target_height,
                target_width
            ));
        }
        let expected_condition_tokens = condition_shapes.iter().map(|(h, w)| h * w).sum::<i32>();
        match condition_latents {
            Some(latents)
                if latents.shape().as_slice()
                    == [1, expected_condition_tokens, self.config.in_channels] => {}
            None if expected_condition_tokens == 0 => {}
            Some(latents) => {
                return Err(anyhow!(
                    "Qwen Image 2.1 condition latent shape {:?} does not match {expected_condition_tokens} tokens",
                    latents.shape().as_slice()
                ));
            }
            None => return Err(anyhow!("Qwen Image 2.1 condition latents are missing")),
        }
        let layout = self.joint_layout(
            encoder_image_mask,
            condition_shapes,
            target_height,
            target_width,
        )?;

        let text = self.text_norm.forward_on(encoder_hidden, target)?;
        let text = self.text_in.forward_on(&text, target)?;
        let text = gelu_tanh(&text, target)?;
        let text = self.text_out.forward_on(&text, target)?;
        let image_latents = match condition_latents {
            Some(condition) => {
                mlx::ops::shape::concatenate_on(&[condition, target_latents], 1, target)?
            }
            None => target_latents.clone(),
        };
        let image = self.image_in.forward_on(&image_latents, target)?;
        let sources = mlx::ops::shape::concatenate_on(&[&text, &image], 1, target)?;
        let mut hidden = mlx::ops::indexing::take_on(&sources, &layout.gather_indices, 1, target)?;

        let time = self.timestep_embedding_on(sigma, target)?;
        let modulation = self
            .modulation
            .forward_on(&silu_on(&time, target)?, target)?;
        let half = modulation.shape().as_slice()[1] / 2;
        let mod1 =
            mlx::ops::indexing::slice_on(&modulation, &[0_i32, 0][..], &[2_i32, half][..], target)?;
        let mod2 = mlx::ops::indexing::slice_on(
            &modulation,
            &[0_i32, half][..],
            &[2_i32, half * 2][..],
            target,
        )?;
        let mod1 = self.selected_modulation_on(&mod1, &layout.target_token_mask, target)?;
        let mod2 = self.selected_modulation_on(&mod2, &layout.target_token_mask, target)?;
        let width = mod1.shape().as_slice()[2] / 2;
        let scale1 = slice_axis_1(
            &mod1.transpose_axes_on(&[0, 2, 1][..], target)?,
            0,
            width,
            target,
        )?
        .transpose_axes_on(&[0, 2, 1][..], target)?;
        let gate1 = slice_axis_1(
            &mod1.transpose_axes_on(&[0, 2, 1][..], target)?,
            width,
            width * 2,
            target,
        )?
        .transpose_axes_on(&[0, 2, 1][..], target)?;
        let scale2 = slice_axis_1(
            &mod2.transpose_axes_on(&[0, 2, 1][..], target)?,
            0,
            width,
            target,
        )?
        .transpose_axes_on(&[0, 2, 1][..], target)?;
        let gate2 = slice_axis_1(
            &mod2.transpose_axes_on(&[0, 2, 1][..], target)?,
            width,
            width * 2,
            target,
        )?
        .transpose_axes_on(&[0, 2, 1][..], target)?;
        for block in &self.blocks {
            hidden = block.forward_on(
                &hidden,
                &scale1,
                &gate1,
                &scale2,
                &gate2,
                &layout.segments,
                layout.prefix_length,
                &layout.cos,
                &layout.sin,
                target,
            )?;
        }

        let final_scale = self.norm_out.forward_on(&silu_on(&time, target)?, target)?;
        let final_scale =
            self.selected_modulation_on(&final_scale, &layout.target_token_mask, target)?;
        hidden =
            affine_free_layer_norm_on(&hidden, self.config.eps, target)? * (&final_scale + 1.0_f32);
        let output = self.projection_out.forward_on(&hidden, target)?;
        slice_axis_1(
            &output,
            layout.prefix_length,
            layout.prefix_length + target_tokens,
            target,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_metallib() {
        static METALLIB: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        METALLIB.get_or_init(|| {
            let metallib = std::path::PathBuf::from(
                std::env::var("MLX_DIR").expect("MLX_DIR must point to the local MLX install"),
            )
            .join("lib/mlx.metallib");
            mlx::metal::set_metallib_path(
                metallib
                    .to_str()
                    .expect("MLX_DIR/lib/mlx.metallib must be UTF-8"),
            )
            .expect("load MLX metallib");
        });
    }

    fn dummy_linear() -> Linear {
        Linear::new_fp(Array::zeros((1_i32, 1_i32), Dtype::Float32).unwrap(), None)
    }

    fn layout_test_transformer() -> QwenImage21Transformer {
        ensure_metallib();
        QwenImage21Transformer {
            config: QwenImage21TransformerConfig {
                attention_head_dim: 128,
                axes_dims_rope: vec![16, 56, 56],
                context_in_dim: 4096,
                in_channels: 64,
                num_attention_heads: 32,
                num_layers: 32,
                out_channels: 64,
                mlp_ratio: 3,
                eps: 1e-6,
                causal_condition: true,
            },
            text_norm: RmsNorm::new(
                mlx::ops::constructors::ones((1_i32,), Dtype::Float32).unwrap(),
                1e-6,
            ),
            text_in: dummy_linear(),
            text_out: dummy_linear(),
            image_in: dummy_linear(),
            time_linear_1: dummy_linear(),
            time_linear_2: dummy_linear(),
            modulation: dummy_linear(),
            blocks: Vec::new(),
            norm_out: dummy_linear(),
            projection_out: dummy_linear(),
        }
    }

    #[test]
    fn joint_layout_expands_condition_slot_and_appends_target_block() {
        let transformer = layout_test_transformer();
        let layout = transformer
            .joint_layout(&[false, true, false], &[(2, 2)], 2, 2)
            .unwrap();
        assert_eq!(
            layout.gather_indices.to_vec::<u32>().unwrap(),
            vec![0, 3, 4, 5, 6, 2, 7, 8, 9, 10]
        );
        assert_eq!(layout.prefix_length, 6);
        assert_eq!(
            layout.target_token_mask,
            vec![false, false, false, false, false, false, true, true, true, true]
        );
        assert_eq!(layout.segments.len(), 3);
        assert_eq!(
            layout
                .segments
                .iter()
                .map(|segment| (segment.start, segment.end, segment.is_text))
                .collect::<Vec<_>>(),
            vec![(0, 1, true), (1, 5, false), (5, 6, true)]
        );
        assert_eq!(layout.cos.shape().as_slice(), &[10, 64]);
        assert_eq!(layout.sin.shape().as_slice(), &[10, 64]);
    }

    #[test]
    fn joint_layout_rejects_condition_slot_geometry_mismatch() {
        let transformer = layout_test_transformer();
        assert!(transformer
            .joint_layout(&[false, true], &[(2, 4)], 2, 2)
            .is_err());
    }
}
