//! Checkpoint-specific Gemma4 Conformer audio encoder.
//! Numerical contract: mlx-vlm gemma4/audio.py at 3d87e88402f307efbf68e568971aa887ee7d9ed0.
use super::audio_processor;
use crate::models::gemma4::vision::{ClippableLinear, MultimodalEmbedder};
use crate::{
    core::audio_input::EmbeddingAudio,
    nn::{LayerNorm, Linear, RmsNorm},
    Loader, Result,
};
use anyhow::{ensure, Context};
use mlx::{Array, Dtype};
use std::path::Path;

fn scalar(value: f32, dtype: Dtype) -> Result<Array> {
    let value: Array = (&[value][..], ()).try_into()?;
    Ok(value.astype(dtype)?)
}
fn scale(x: &Array, value: f32) -> Result<Array> {
    Ok(x * &scalar(value, x.dtype())?)
}
fn clip(x: &Array) -> Result<Array> {
    Ok(mlx::ops::clip(
        x,
        Some(&scalar(-1e10, x.dtype())?),
        Some(&scalar(1e10, x.dtype())?),
    )?)
}
fn silu(x: &Array) -> Result<Array> {
    thread_local! { static SILU: std::cell::OnceCell<mlx::compile::CompiledFn> = const { std::cell::OnceCell::new() }; }
    SILU.with(|cache| {
        if cache.get().is_none() {
            let function = mlx::compile::compile(
                |inputs| Ok(vec![inputs[0] * &mlx::ops::sigmoid(inputs[0])?]),
                mlx::compile::ShapeMode::Shapeless,
            )?;
            cache
                .set(function)
                .map_err(|_| anyhow::anyhow!("audio SiLU compilation already initialized"))?;
        }
        let mut values = cache.get().expect("initialized SiLU").invoke(&[x])?;
        Ok(values.remove(0))
    })
}

fn norm(loader: &Loader, prefix: &str, name: &str) -> Result<RmsNorm> {
    RmsNorm::from_loader(loader, &format!("{prefix}.{name}"), 1e-6)
}
fn linear(loader: &Loader, prefix: &str, name: &str) -> Result<ClippableLinear> {
    ClippableLinear::from_loader(loader, &format!("{prefix}.{name}"), true)
}
fn pad(x: &Array, axis: usize, left: i32, right: i32) -> Result<Array> {
    let mut pieces = Vec::new();
    for count in [left, right] {
        let mut shape = x.shape().as_slice().to_vec();
        shape[axis] = count;
        pieces.push(Array::zeros(shape.as_slice(), x.dtype())?);
    }
    Ok(mlx::ops::concatenate(
        &[&pieces[0], x, &pieces[1]],
        axis as i32,
    )?)
}
struct FeedForward {
    pre: RmsNorm,
    first: ClippableLinear,
    second: ClippableLinear,
    post: RmsNorm,
}
impl FeedForward {
    fn load(l: &Loader, p: &str) -> Result<Self> {
        Ok(Self {
            pre: norm(l, p, "pre_layer_norm")?,
            first: linear(l, p, "ffw_layer_1")?,
            second: linear(l, p, "ffw_layer_2")?,
            post: norm(l, p, "post_layer_norm")?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.pre.forward(&clip(x)?)?;
        let h = silu(&self.first.forward_on(&h, ().into())?)?;
        let h = self.second.forward_on(&h, ().into())?;
        Ok(x + &scale(&self.post.forward(&clip(&h)?)?, 0.5)?)
    }
}
struct LightConv {
    pre: RmsNorm,
    start: ClippableLinear,
    weight: Array,
    norm: RmsNorm,
    end: ClippableLinear,
}
impl LightConv {
    fn load(l: &Loader, p: &str) -> Result<Self> {
        let weight = l.tensor(&format!("{p}.depthwise_conv1d.weight"))?.clone();
        let weight = if weight.shape().as_slice() == [1024, 1, 5] {
            weight.transpose_axes(&[0, 2, 1][..])?
        } else {
            weight
        };
        ensure!(
            weight.shape().as_slice() == [1024, 5, 1],
            "invalid audio depthwise convolution weight"
        );
        Ok(Self {
            pre: norm(l, p, "pre_layer_norm")?,
            start: linear(l, p, "linear_start")?,
            weight,
            norm: norm(l, p, "conv_norm")?,
            end: linear(l, p, "linear_end")?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.start.forward_on(&self.pre.forward(x)?, ().into())?;
        let parts = mlx::ops::split_n(&h, 2, -1)?;
        let h = &parts[0] * &mlx::ops::sigmoid(&parts[1])?;
        let h = mlx::ops::conv::conv1d(&pad(&h, 1, 4, 0)?, &self.weight, 1, 0, 1, 1024)?;
        let h = silu(&self.norm.forward(&clip(&h)?)?)?;
        Ok(x + &self.end.forward_on(&h, ().into())?)
    }
}
struct Attention {
    q: ClippableLinear,
    k: ClippableLinear,
    v: ClippableLinear,
    post: ClippableLinear,
    relative: Linear,
    position: Array,
    per_dim_scale: Array,
}
impl Attention {
    fn load(l: &Loader, p: &str) -> Result<Self> {
        let times: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let times: Array = (times.as_slice(), (1, 1, 512)).try_into()?;
        let inverse = mlx::ops::exp(&scale(&times, -(10000.0f64.ln() / 511.0) as f32)?)?;
        let positions: Vec<f32> = (0..13).map(|i| (12 - i) as f32).collect();
        let positions: Array = (positions.as_slice(), (1, 13, 1)).try_into()?;
        let angles = &positions * &inverse;
        let position =
            mlx::ops::concatenate(&[&mlx::ops::sin(&angles)?, &mlx::ops::cos(&angles)?], -1)?;
        Ok(Self {
            q: linear(l, p, "q_proj")?,
            k: linear(l, p, "k_proj")?,
            v: linear(l, p, "v_proj")?,
            post: linear(l, p, "post")?,
            relative: Linear::from_loader(l, &format!("{p}.relative_k_proj"))?,
            position: position.astype(Dtype::Bfloat16)?,
            per_dim_scale: l.tensor(&format!("{p}.per_dim_scale"))?.clone(),
        })
    }
    fn forward(&self, x: &Array, valid: usize) -> Result<Array> {
        let t = x.shape()[1];
        let u = (t + 11) / 12;
        let shape = (1, t, 8, 128);
        let q = self
            .q
            .forward_on(x, ().into())?
            .astype(Dtype::Float32)?
            .reshape(shape)?;
        let k = self
            .k
            .forward_on(x, ().into())?
            .astype(Dtype::Float32)?
            .reshape(shape)?;
        let v = self
            .v
            .forward_on(x, ().into())?
            .astype(Dtype::Float32)?
            .reshape(shape)?;
        let softplus = mlx::ops::logaddexp(
            &self.per_dim_scale,
            &scalar(0.0, self.per_dim_scale.dtype())?,
        )?;
        let q = &q * &scale(&softplus, (128.0f64.powf(-0.5) / 2.0f64.ln()) as f32)?;
        let k = scale(&k, ((1.0 + std::f64::consts::E).ln() / 2.0f64.ln()) as f32)?;
        let q = pad(&q, 1, 0, u * 12 - t)?
            .reshape(&[1, u, 12, 8, 128][..])?
            .transpose_axes(&[0, 3, 1, 2, 4][..])?;
        let indices: Vec<u32> = (0..u)
            .flat_map(|block| (0..24).map(move |j| (block * 12 + j) as u32))
            .collect();
        let indices: Array = (indices.as_slice(), (u, 24)).try_into()?;
        let k = mlx::ops::take(&pad(&k, 1, 12, 11)?, &indices, 1)?
            .transpose_axes(&[0, 3, 1, 4, 2][..])?;
        let v = mlx::ops::take(&pad(&v, 1, 12, 11)?, &indices, 1)?
            .transpose_axes(&[0, 3, 1, 2, 4][..])?;
        // Relative positions are projected in checkpoint dtype then promoted, as upstream.
        let pos = self
            .relative
            .forward(&self.position)?
            .astype(Dtype::Float32)?
            .reshape((13, 8, 128))?
            .transpose_axes(&[1, 2, 0][..])?;
        let relative = q
            .reshape((1, 8, u * 12, 128))?
            .matmul(&pos)?
            .reshape(&[1, 8, u, 12, 13][..])?;
        let relative = pad(&relative, 4, 0, 12)?
            .reshape((1, 8, u, 300))?
            .slice((0, 0, 0, 0), (1, 8, u, 288))?
            .reshape(&[1, 8, u, 12, 24][..])?;
        let logits = &q.matmul(&k)? + &relative;
        let logits = scale(
            &mlx::ops::tanh(&mlx::ops::divide(&logits, &scalar(50.0, Dtype::Float32)?)?)?,
            50.0,
        )?;
        let mask: Vec<bool> = (0..u)
            .flat_map(|block| {
                (0..12).flat_map(move |query| {
                    (0..24).map(move |key| {
                        let at = block * 12 + key - 12;
                        let distance = query + 12 - key;
                        at >= 0 && (at as usize) < valid && (0..12).contains(&distance)
                    })
                })
            })
            .collect();
        let mask: Array = (mask.as_slice(), &[1, 1, u, 12, 24][..]).try_into()?;
        let logits = mlx::ops::where_(&mask, &logits, &scalar(-1e9, Dtype::Float32)?)?;
        let probs = mlx::ops::softmax(&logits, -1, false)?;
        let context = mlx::ops::einsum::einsum(
            "bnuwc,bucnh->buwnh",
            &[&probs, &v.transpose_axes(&[0, 2, 3, 1, 4][..])?],
        )?
        .reshape((1, u * 12, 1024))?
        .slice((0, 0, 0), (1, t, 1024))?;
        self.post.forward_on(&context, ().into())
    }
}
struct Block {
    first: FeedForward,
    attn: Attention,
    pre: RmsNorm,
    post: RmsNorm,
    conv: LightConv,
    second: FeedForward,
    out: RmsNorm,
}
impl Block {
    fn load(l: &Loader, p: &str) -> Result<Self> {
        Ok(Self {
            first: FeedForward::load(l, &format!("{p}.feed_forward1"))?,
            attn: Attention::load(l, &format!("{p}.self_attn"))?,
            pre: norm(l, p, "norm_pre_attn")?,
            post: norm(l, p, "norm_post_attn")?,
            conv: LightConv::load(l, &format!("{p}.lconv1d"))?,
            second: FeedForward::load(l, &format!("{p}.feed_forward2"))?,
            out: norm(l, p, "norm_out")?,
        })
    }
    fn forward(&self, x: &Array, valid: usize) -> Result<Array> {
        let h = self.first.forward(x)?;

        let attn = self.attn.forward(&self.pre.forward(&clip(&h)?)?, valid)?;
        let h = &h + &self.post.forward(&clip(&attn)?)?;
        let mask: Vec<f32> = (0..h.shape()[1])
            .map(|i| if (i as usize) < valid { 1.0 } else { 0.0 })
            .collect();
        let mask: Array = (mask.as_slice(), (1, h.shape()[1], 1)).try_into()?;
        let h = self.conv.forward(&(&h * &mask))?;
        self.out.forward(&clip(&self.second.forward(&h)?)?)
    }
}
struct Subsample {
    weights: [Array; 2],
    norms: [LayerNorm; 2],
    projection: Linear,
}
impl Subsample {
    fn load(l: &Loader) -> Result<Self> {
        let p = "audio_tower.subsample_conv_projection";
        let mut weights = Vec::new();
        let mut norms = Vec::new();
        for i in 0..2 {
            let w = l.tensor(&format!("{p}.layer{i}.conv.weight"))?.clone();
            ensure!(w.ndim() == 4, "invalid audio subsampling weight rank");
            let channels = if i == 0 { 1 } else { 128 };
            let w = if w.shape()[3] != channels {
                w.transpose_axes(&[0, 2, 3, 1][..])?
            } else {
                w
            };
            ensure!(
                w.shape().as_slice() == [if i == 0 { 128 } else { 32 }, 3, 3, channels],
                "invalid audio subsampling weight"
            );
            weights.push(w);
            norms.push(LayerNorm::from_loader(
                l,
                &format!("{p}.layer{i}.norm"),
                1e-6,
            )?);
        }
        Ok(Self {
            weights: weights.try_into().expect("two convolution weights"),
            norms: norms.try_into().unwrap_or_else(|_| panic!("two norms")),
            projection: Linear::from_loader(l, &format!("{p}.input_proj_linear"))?,
        })
    }
    fn forward(&self, features: &audio_processor::AudioFeatures) -> Result<Array> {
        let mut h: Array = (
            features.values.as_slice(),
            (1, features.frames as i32, 128, 1),
        )
            .try_into()?;
        let mut valid = features.valid_frames;
        for i in 0..2 {
            let t = h.shape()[1];
            let mask: Vec<f32> = (0..t)
                .map(|at| if (at as usize) < valid { 1.0 } else { 0.0 })
                .collect();
            let mask: Array = (mask.as_slice(), (1, t, 1, 1)).try_into()?;
            h = mlx::ops::conv::conv2d(&(&h * &mask), &self.weights[i], (2, 2), (1, 1), (1, 1), 1)?;
            h = mlx::ops::maximum(&self.norms[i].forward(&h)?, &scalar(0.0, h.dtype())?)?;
            valid = valid.div_ceil(2);
        }
        self.projection
            .forward(&h.reshape((1, h.shape()[1], 1024))?)
    }
}

pub(super) struct AudioEncoder {
    subsample: Subsample,
    blocks: Vec<Block>,
    output: Linear,
    projection: MultimodalEmbedder,
}
impl AudioEncoder {
    pub fn load(l: &Loader, path: &Path) -> Result<Option<Self>> {
        if !validate_config(l.config_raw_value())? {
            return Ok(None);
        }
        let processor: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("processor_config.json"))
                .context("reading embedding audio processor")?,
        )?;
        validate_processor(&processor["feature_extractor"])?;
        Ok(Some(Self {
            subsample: Subsample::load(l)?,
            blocks: (0..12)
                .map(|i| Block::load(l, &format!("audio_tower.layers.{i}")))
                .collect::<Result<_>>()?,
            output: Linear::from_loader(l, "audio_tower.output_proj")?,
            projection: MultimodalEmbedder::from_loader(l, "embed_audio", 1e-6)?,
        }))
    }
    pub fn token_count(&self, audio: &EmbeddingAudio) -> usize {
        audio_processor::token_count(audio)
    }
    pub fn encode(&self, audio: &EmbeddingAudio) -> Result<Array> {
        let features = audio_processor::extract(audio)?;
        let mut h = self.subsample.forward(&features)?;

        for block in &self.blocks {
            h = block.forward(&h, features.valid_tokens)?;
            h.eval()?;
        }
        let h = self.output.forward(&h)?;

        let mask: Vec<f32> = (0..h.shape()[1])
            .map(|i| {
                if (i as usize) < features.valid_tokens {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mask: Array = (mask.as_slice(), (1, h.shape()[1], 1)).try_into()?;
        let h = self.projection.forward_on(&(&h * &mask), ())?;
        Ok(h.slice((0, 0, 0), (1, features.valid_tokens as i32, 512))?)
    }
}
pub(super) fn validate_config(raw: &serde_json::Value) -> Result<bool> {
    let Some(c) = raw.get("audio_config").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    for (name, expected) in [
        ("model_type", serde_json::json!("gemma4_audio")),
        ("dtype", serde_json::json!("bfloat16")),
        ("hidden_size", serde_json::json!(1024)),
        ("num_hidden_layers", serde_json::json!(12)),
        ("num_attention_heads", serde_json::json!(8)),
        ("output_proj_dims", serde_json::json!(1536)),
        ("subsampling_conv_channels", serde_json::json!([128, 32])),
        ("conv_kernel_size", serde_json::json!(5)),
        ("hidden_act", serde_json::json!("silu")),
        ("residual_weight", serde_json::json!(0.5)),
        ("rms_norm_eps", serde_json::json!(1e-6)),
        ("use_clipped_linears", serde_json::json!(true)),
        ("attention_chunk_size", serde_json::json!(12)),
        ("attention_context_left", serde_json::json!(13)),
        ("attention_context_right", serde_json::json!(0)),
        ("attention_invalid_logits_value", serde_json::json!(-1e9)),
        ("attention_logit_cap", serde_json::json!(50.0)),
        ("gradient_clipping", serde_json::json!(1e10)),
    ] {
        ensure!(
            c[name] == expected,
            "unsupported embedding audio configuration: {name}"
        );
    }
    ensure!(
        raw["audio_token_id"] == 258881
            && raw["boa_token_id"] == 256000
            && raw["eoa_token_index"] == 258883
            && raw["text_config"]["hidden_size"] == 512,
        "unsupported embedding audio token or projection"
    );
    Ok(true)
}
fn validate_processor(c: &serde_json::Value) -> Result<()> {
    for (name, expected) in [
        (
            "feature_extractor_type",
            serde_json::json!("Gemma4AudioFeatureExtractor"),
        ),
        ("sampling_rate", serde_json::json!(16000)),
        ("feature_size", serde_json::json!(128)),
        ("frame_length", serde_json::json!(320)),
        ("hop_length", serde_json::json!(160)),
        ("fft_length", serde_json::json!(512)),
        ("fft_overdrive", serde_json::json!(false)),
        ("min_frequency", serde_json::json!(0.0)),
        ("max_frequency", serde_json::json!(8000.0)),
        ("mel_floor", serde_json::json!(0.001)),
        ("dither", serde_json::json!(0.0)),
        ("input_scale_factor", serde_json::json!(1.0)),
        ("preemphasis", serde_json::json!(0.0)),
        ("per_bin_mean", serde_json::Value::Null),
        ("per_bin_stddev", serde_json::Value::Null),
        ("padding_side", serde_json::json!("right")),
        ("padding_value", serde_json::json!(0.0)),
        ("return_attention_mask", serde_json::json!(true)),
    ] {
        ensure!(
            c[name] == expected,
            "unsupported embedding audio preprocessing configuration: {name}"
        );
    }
    Ok(())
}
