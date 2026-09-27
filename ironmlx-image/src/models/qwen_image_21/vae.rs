use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};
use serde::Deserialize;

use crate::loader::ComponentLoader;
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
struct QwenImage21VaeConfig {
    base_dim: i32,
    decoder_base_dim: i32,
    dim_mult: Vec<i32>,
    in_channels: i32,
    is_residual: bool,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    num_res_blocks: i32,
    out_channels: i32,
    scale_factor_spatial: i32,
    temperal_downsample: Vec<bool>,
    z_dim: i32,
}

struct Conv2d {
    weight: Array,
    bias: Option<Array>,
    stride: i32,
    padding: i32,
}

impl Conv2d {
    fn from_loader(
        loader: &ComponentLoader,
        prefix: &str,
        stride: i32,
        padding: i32,
    ) -> Result<Self> {
        let source = loader.tensor(&format!("{prefix}.weight"))?;
        if source.ndim() != 4 {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE convolution {prefix}.weight must be rank 4"
            ));
        }
        // The pinned MLX checkpoint has already converted Diffusers OIHW
        // kernels to the native MLX OHWI layout.
        let weight = source.clone();
        mlx::transforms::eval(&[&weight])?;
        Ok(Self {
            weight,
            bias: loader.tensor_opt(&format!("{prefix}.bias")).cloned(),
            stride,
            padding,
        })
    }

    fn forward_on(&self, input_nchw: &Array, target: StreamOrDevice) -> Result<Array> {
        let input = input_nchw.transpose_axes_on(&[0_i32, 2, 3, 1][..], target)?;
        let mut output = mlx::ops::conv2d_on(
            &input,
            &self.weight,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            1,
            target,
        )?;
        if let Some(bias) = &self.bias {
            output = &output + bias;
        }
        Ok(output.transpose_axes_on(&[0_i32, 3, 1, 2][..], target)?)
    }
}

struct ChannelRmsNorm {
    weight: Array,
    scale: f32,
}

impl ChannelRmsNorm {
    fn from_loader(loader: &ComponentLoader, prefix: &str, channels: i32) -> Result<Self> {
        let source = loader.tensor(&format!("{prefix}.gamma"))?;
        let weight = source.reshape((channels,))?;
        mlx::transforms::eval(&[&weight])?;
        Ok(Self {
            weight,
            scale: (channels as f32).sqrt(),
        })
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let input_f32 = input.astype_on(Dtype::Float32, target)?;
        let l2 = (&input_f32 * &input_f32)
            .sum_on(1, true, target)?
            .sqrt_on(target)?;
        let epsilon: Array = (&[1e-12_f32][..], ()).try_into()?;
        let l2 = l2.maximum_on(&epsilon, target)?;
        let normalized = (&input_f32 / &l2).astype_on(input.dtype(), target)?;
        let weight = self
            .weight
            .astype_on(input.dtype(), target)?
            .reshape_on((1_i32, -1_i32, 1_i32, 1_i32), target)?;
        Ok(&normalized * self.scale * &weight)
    }
}

fn silu_on(value: &Array, target: StreamOrDevice) -> Result<Array> {
    Ok(value * &value.sigmoid_on(target)?)
}

struct ResBlock {
    norm1: ChannelRmsNorm,
    conv1: Conv2d,
    norm2: ChannelRmsNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResBlock {
    fn from_loader(
        loader: &ComponentLoader,
        prefix: &str,
        input_channels: i32,
        output_channels: i32,
    ) -> Result<Self> {
        Ok(Self {
            norm1: ChannelRmsNorm::from_loader(loader, &format!("{prefix}.norm1"), input_channels)?,
            conv1: Conv2d::from_loader(loader, &format!("{prefix}.conv1"), 1, 1)?,
            norm2: ChannelRmsNorm::from_loader(
                loader,
                &format!("{prefix}.norm2"),
                output_channels,
            )?,
            conv2: Conv2d::from_loader(loader, &format!("{prefix}.conv2"), 1, 1)?,
            shortcut: (input_channels != output_channels)
                .then(|| Conv2d::from_loader(loader, &format!("{prefix}.conv_shortcut"), 1, 0))
                .transpose()?,
        })
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward_on(input, target)?,
            None => input.clone(),
        };
        let hidden = silu_on(&self.norm1.forward_on(input, target)?, target)?;
        let hidden = self.conv1.forward_on(&hidden, target)?;
        let hidden = silu_on(&self.norm2.forward_on(&hidden, target)?, target)?;
        Ok(self.conv2.forward_on(&hidden, target)? + &residual)
    }
}

struct SpatialAttention {
    norm: ChannelRmsNorm,
    qkv: Conv2d,
    projection: Conv2d,
    channels: i32,
}

impl SpatialAttention {
    fn from_loader(loader: &ComponentLoader, prefix: &str, channels: i32) -> Result<Self> {
        Ok(Self {
            norm: ChannelRmsNorm::from_loader(loader, &format!("{prefix}.norm"), channels)?,
            qkv: Conv2d::from_loader(loader, &format!("{prefix}.to_qkv"), 1, 0)?,
            projection: Conv2d::from_loader(loader, &format!("{prefix}.proj"), 1, 0)?,
            channels,
        })
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = input.shape();
        let dims = shape.as_slice();
        let (batch, height, width) = (dims[0], dims[2], dims[3]);
        let hidden = self.norm.forward_on(input, target)?;
        let qkv = self
            .qkv
            .forward_on(&hidden, target)?
            .transpose_axes_on(&[0_i32, 2, 3, 1][..], target)?
            .reshape_on((batch, height * width, 3_i32, self.channels), target)?;
        let mut parts = mlx::ops::shape::split_n_on(&qkv, 3, 2, target)?;
        let query = parts
            .remove(0)
            .reshape_on((batch, height * width, self.channels), target)?;
        let key = parts
            .remove(0)
            .reshape_on((batch, height * width, self.channels), target)?;
        let value = parts
            .remove(0)
            .reshape_on((batch, height * width, self.channels), target)?;
        let query = query.reshape_on((batch, 1_i32, height * width, self.channels), target)?;
        let key = key.reshape_on((batch, 1_i32, height * width, self.channels), target)?;
        let value = value.reshape_on((batch, 1_i32, height * width, self.channels), target)?;
        let attended = mlx::fast::scaled_dot_product_attention_on(
            &query,
            &key,
            &value,
            1.0 / (self.channels as f32).sqrt(),
            "",
            None,
            None,
            target,
        )?
        .reshape_on((batch, height, width, self.channels), target)?
        .transpose_axes_on(&[0_i32, 3, 1, 2][..], target)?;
        Ok(self.projection.forward_on(&attended, target)? + input)
    }
}

struct MidBlock {
    first: ResBlock,
    attention: SpatialAttention,
    second: ResBlock,
}

impl MidBlock {
    fn from_loader(loader: &ComponentLoader, prefix: &str, channels: i32) -> Result<Self> {
        Ok(Self {
            first: ResBlock::from_loader(
                loader,
                &format!("{prefix}.resnets.0"),
                channels,
                channels,
            )?,
            attention: SpatialAttention::from_loader(
                loader,
                &format!("{prefix}.attentions.0"),
                channels,
            )?,
            second: ResBlock::from_loader(
                loader,
                &format!("{prefix}.resnets.1"),
                channels,
                channels,
            )?,
        })
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let hidden = self.first.forward_on(input, target)?;
        let hidden = self.attention.forward_on(&hidden, target)?;
        self.second.forward_on(&hidden, target)
    }
}

struct UpBlock {
    resnets: Vec<ResBlock>,
    upsampler: Option<Conv2d>,
    input_channels: i32,
    output_channels: i32,
    temporal_factor: i32,
}

struct DownBlock {
    resnets: Vec<ResBlock>,
    downsampler: Option<Conv2d>,
    output_channels: i32,
    temporal_factor: i32,
}

impl DownBlock {
    fn from_loader(
        loader: &ComponentLoader,
        index: i32,
        input_channels: i32,
        output_channels: i32,
        downsample: bool,
        temporal_factor: i32,
    ) -> Result<Self> {
        let prefix = format!("encoder.down_blocks.{index}");
        let mut resnets = Vec::with_capacity(2);
        for resnet in 0..2 {
            resnets.push(ResBlock::from_loader(
                loader,
                &format!("{prefix}.resnets.{resnet}"),
                if resnet == 0 {
                    input_channels
                } else {
                    output_channels
                },
                output_channels,
            )?);
        }
        Ok(Self {
            resnets,
            downsampler: downsample
                .then(|| {
                    Conv2d::from_loader(loader, &format!("{prefix}.downsampler.resample.1"), 2, 0)
                })
                .transpose()?,
            output_channels,
            temporal_factor,
        })
    }

    fn average_shortcut_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = input.shape();
        let dims = shape.as_slice();
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let spatial_factor = if self.downsampler.is_some() {
            2_i32
        } else {
            1_i32
        };
        if height % spatial_factor != 0 || width % spatial_factor != 0 {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE downsample expects divisible geometry, got {height}x{width}"
            ));
        }
        let mut temporal =
            input.reshape_on(&[batch, channels, 1_i32, height, width][..], target)?;
        if self.temporal_factor == 2 {
            let zeros = Array::zeros(&[batch, channels, 1_i32, height, width][..], input.dtype())?;
            temporal = mlx::ops::shape::concatenate_on(&[&zeros, &temporal], 2, target)?;
        }
        let factor = self.temporal_factor * spatial_factor * spatial_factor;
        let grouped = temporal
            .reshape_on(
                &[
                    batch,
                    channels,
                    1_i32,
                    self.temporal_factor,
                    height / spatial_factor,
                    spatial_factor,
                    width / spatial_factor,
                    spatial_factor,
                ][..],
                target,
            )?
            .transpose_axes_on(&[0_i32, 1, 3, 5, 7, 2, 4, 6][..], target)?
            .reshape_on(
                &[
                    batch,
                    channels * factor,
                    1_i32,
                    height / spatial_factor,
                    width / spatial_factor,
                ][..],
                target,
            )?;
        let group_size = channels * factor / self.output_channels;
        Ok(grouped
            .reshape_on(
                &[
                    batch,
                    self.output_channels,
                    group_size,
                    1_i32,
                    height / spatial_factor,
                    width / spatial_factor,
                ][..],
                target,
            )?
            .mean_on(2, false, target)?
            .reshape_on(
                (
                    batch,
                    self.output_channels,
                    height / spatial_factor,
                    width / spatial_factor,
                ),
                target,
            )?)
    }

    fn downsample_main_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let Some(downsampler) = &self.downsampler else {
            return Ok(input.clone());
        };
        let shape = input.shape();
        let dims = shape.as_slice();
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let right = Array::zeros((batch, channels, height, 1_i32), input.dtype())?;
        let padded = mlx::ops::shape::concatenate_on(&[input, &right], 3, target)?;
        let bottom = Array::zeros((batch, channels, 1_i32, width + 1), input.dtype())?;
        let padded = mlx::ops::shape::concatenate_on(&[&padded, &bottom], 2, target)?;
        downsampler.forward_on(&padded, target)
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let shortcut = self.average_shortcut_on(input, target)?;
        let mut hidden = input.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward_on(&hidden, target)?;
        }
        let hidden = self.downsample_main_on(&hidden, target)?;
        Ok(hidden + &shortcut)
    }
}

impl UpBlock {
    fn from_loader(
        loader: &ComponentLoader,
        index: i32,
        input_channels: i32,
        output_channels: i32,
        upsample: bool,
        temporal_factor: i32,
    ) -> Result<Self> {
        let prefix = format!("decoder.up_blocks.{index}");
        let mut resnets = Vec::with_capacity(3);
        for resnet in 0..3 {
            resnets.push(ResBlock::from_loader(
                loader,
                &format!("{prefix}.resnets.{resnet}"),
                if resnet == 0 {
                    input_channels
                } else {
                    output_channels
                },
                output_channels,
            )?);
        }
        Ok(Self {
            resnets,
            upsampler: upsample
                .then(|| {
                    Conv2d::from_loader(loader, &format!("{prefix}.upsampler.resample.1"), 1, 1)
                })
                .transpose()?,
            input_channels,
            output_channels,
            temporal_factor,
        })
    }

    fn duplicate_shortcut_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = input.shape();
        let dims = shape.as_slice();
        let (batch, height, width) = (dims[0], dims[2], dims[3]);
        let spatial_factor = 2_i32;
        let factor = self.temporal_factor * spatial_factor * spatial_factor;
        let repeats = self.output_channels * factor / self.input_channels;
        let repeated = input.repeat_on(repeats, 1, target)?;
        let rearranged = repeated
            .reshape_on(
                &[
                    batch,
                    self.output_channels,
                    self.temporal_factor,
                    spatial_factor,
                    spatial_factor,
                    1_i32,
                    height,
                    width,
                ][..],
                target,
            )?
            .transpose_axes_on(&[0_i32, 1, 5, 2, 6, 3, 7, 4][..], target)?
            .reshape_on(
                &[
                    batch,
                    self.output_channels,
                    self.temporal_factor,
                    height * 2,
                    width * 2,
                ][..],
                target,
            )?;
        Ok(mlx::ops::indexing::slice_on(
            &rearranged,
            &[0_i32, 0, self.temporal_factor - 1, 0, 0][..],
            &[
                batch,
                self.output_channels,
                self.temporal_factor,
                height * 2,
                width * 2,
            ][..],
            target,
        )?
        .reshape_on((batch, self.output_channels, height * 2, width * 2), target)?)
    }

    fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let mut hidden = input.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward_on(&hidden, target)?;
        }
        let Some(upsampler) = &self.upsampler else {
            return Ok(hidden);
        };
        let shortcut = self.duplicate_shortcut_on(input, target)?;
        let hidden = hidden.repeat_on(2, 2, target)?.repeat_on(2, 3, target)?;
        Ok(upsampler.forward_on(&hidden, target)? + &shortcut)
    }
}

/// Qwen Image 2.1 image VAE used for condition-image encoding and output decoding.
pub struct QwenImage21Vae {
    mean: Array,
    std: Array,
    quant: Conv2d,
    encoder_conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    encoder_mid: MidBlock,
    encoder_norm_out: ChannelRmsNorm,
    encoder_conv_out: Conv2d,
    post_quant: Conv2d,
    conv_in: Conv2d,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: ChannelRmsNorm,
    conv_out: Conv2d,
}

impl QwenImage21Vae {
    pub fn from_loader(loader: &ComponentLoader) -> Result<Self> {
        let config: QwenImage21VaeConfig = loader.config()?;
        if config.base_dim != 96
            || config.decoder_base_dim != 144
            || config.dim_mult != [1, 2, 4, 8, 8]
            || config.in_channels != 4
            || !config.is_residual
            || config.num_res_blocks != 2
            || config.out_channels != 4
            || config.scale_factor_spatial != 16
            || config.temperal_downsample != [false, true, true, true]
            || config.z_dim != 64
            || config.latents_mean.len() != 64
            || config.latents_std.len() != 64
        {
            return Err(anyhow!("unsupported Qwen Image 2.1 VAE config"));
        }
        let mean: Array = (
            config.latents_mean.as_slice(),
            (1_i32, 64_i32, 1_i32, 1_i32),
        )
            .try_into()?;
        let std: Array =
            (config.latents_std.as_slice(), (1_i32, 64_i32, 1_i32, 1_i32)).try_into()?;
        let dims = [1152_i32, 1152, 1152, 576, 288, 144];
        let temporal = [2_i32, 2, 2, 1];
        let mut up_blocks = Vec::with_capacity(5);
        for index in 0..5 {
            up_blocks.push(
                UpBlock::from_loader(
                    loader,
                    index as i32,
                    dims[index],
                    dims[index + 1],
                    index < 4,
                    temporal.get(index).copied().unwrap_or(1),
                )
                .with_context(|| format!("loading Qwen Image 2.1 VAE up block {index}"))?,
            );
        }
        let encoder_dims = [96_i32, 96, 192, 384, 768, 768];
        let mut down_blocks = Vec::with_capacity(5);
        for index in 0..5 {
            down_blocks.push(
                DownBlock::from_loader(
                    loader,
                    index as i32,
                    encoder_dims[index],
                    encoder_dims[index + 1],
                    index < 4,
                    if index < 4 && config.temperal_downsample[index] {
                        2
                    } else {
                        1
                    },
                )
                .with_context(|| format!("loading Qwen Image 2.1 VAE down block {index}"))?,
            );
        }
        Ok(Self {
            mean,
            std,
            quant: Conv2d::from_loader(loader, "quant_conv", 1, 0)?,
            encoder_conv_in: Conv2d::from_loader(loader, "encoder.conv_in", 1, 1)?,
            down_blocks,
            encoder_mid: MidBlock::from_loader(loader, "encoder.mid_block", 768)?,
            encoder_norm_out: ChannelRmsNorm::from_loader(loader, "encoder.norm_out", 768)?,
            encoder_conv_out: Conv2d::from_loader(loader, "encoder.conv_out", 1, 1)?,
            post_quant: Conv2d::from_loader(loader, "post_quant_conv", 1, 0)?,
            conv_in: Conv2d::from_loader(loader, "decoder.conv_in", 1, 1)?,
            mid: MidBlock::from_loader(loader, "decoder.mid_block", 1152)?,
            up_blocks,
            norm_out: ChannelRmsNorm::from_loader(loader, "decoder.norm_out", 144)?,
            conv_out: Conv2d::from_loader(loader, "decoder.conv_out", 1, 1)?,
        })
    }

    /// Encode a normalized RGBA image in NCHW layout and return normalized
    /// 64-channel condition latents. Qwen Image 2.1 uses the posterior mode,
    /// so the log-variance half is intentionally discarded.
    pub fn encode_condition(&self, image: &Array) -> Result<Array> {
        if image.ndim() != 4 || image.shape().as_slice()[1] != 4 {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE expects an NCHW RGBA condition image, got {:?}",
                image.shape().as_slice()
            ));
        }
        let target = StreamOrDevice::default();
        let mut hidden = self.encoder_conv_in.forward_on(image, target)?;
        for block in &self.down_blocks {
            hidden = block.forward_on(&hidden, target)?;
        }
        hidden = self.encoder_mid.forward_on(&hidden, target)?;
        hidden = silu_on(&self.encoder_norm_out.forward_on(&hidden, target)?, target)?;
        let moments = self.encoder_conv_out.forward_on(&hidden, target)?;
        let moments = self.quant.forward_on(&moments, target)?;
        let dims = moments.shape();
        let dims = dims.as_slice();
        if dims[1] != 128 {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE posterior must have 128 channels, got {:?}",
                dims
            ));
        }
        let mode = mlx::ops::indexing::slice_on(
            &moments,
            &[0_i32, 0, 0, 0][..],
            &[dims[0], 64, dims[2], dims[3]][..],
            target,
        )?;
        Ok((&mode - &self.mean) / &self.std)
    }

    pub fn decode(&self, latents: &Array) -> Result<Array> {
        self.decode_on(latents, ())
    }

    pub fn decode_on(&self, latents: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        if latents.ndim() != 4 || latents.shape().as_slice()[1] != 64 {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE expects NCHW latents with 64 channels, got {:?}",
                latents.shape().as_slice()
            ));
        }
        let hidden = latents * &self.std + &self.mean;
        let hidden = self.post_quant.forward_on(&hidden, target)?;
        let hidden = self.conv_in.forward_on(&hidden, target)?;
        let mut hidden = self.mid.forward_on(&hidden, target)?;
        for block in &self.up_blocks {
            hidden = block.forward_on(&hidden, target)?;
        }
        let hidden = silu_on(&self.norm_out.forward_on(&hidden, target)?, target)?;
        let image = self.conv_out.forward_on(&hidden, target)?;
        let lower: Array = (&[-1.0_f32][..], ()).try_into()?;
        let upper: Array = (&[1.0_f32][..], ()).try_into()?;
        Ok(image.clip_on(Some(&lower), Some(&upper), target)?)
    }
}
