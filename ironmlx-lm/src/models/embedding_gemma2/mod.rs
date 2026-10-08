//! Bidirectional text, image and audio embedding encoder for EmbeddingGemma 2.
//! BF16 and MLX affine4 checkpoints share this execution graph.
//! Numerical reference: mlx-vlm/models/embedding_gemma2/language.py at
//! 3d87e88402f307efbf68e568971aa887ee7d9ed0 (Blaizzy/mlx-vlm).
use std::{collections::HashMap, path::Path};

use anyhow::{bail, ensure};
use mlx::{Array, Dtype};
use serde::Deserialize;

use crate::{
    core::loader::QuantMode,
    nn::{gelu_tanh, Embedding, Linear, RmsNorm},
    Loader, Result, Tokenizer,
};

mod audio;
mod audio_processor;
mod vision;

/// Ordered embedding content. Images are encoded bytes; audio is bounded mono 16 kHz PCM.
#[derive(Clone, Debug)]
pub enum EmbeddingContent {
    Text(String),
    Image(Vec<u8>),
    Audio(crate::core::audio_input::EmbeddingAudio),
}
#[derive(Clone, Debug)]
pub struct EmbeddingSample {
    pub content: Vec<EmbeddingContent>,
}

pub const MAX_INPUT_TOKENS: usize = 8192;
pub const MAX_BATCH_SIZE: usize = 32;
pub const SUPPORTED_DIMENSIONS: &[usize] = &[128, 256, 512, 768];

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub hidden_size_per_layer_input: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub embedding_dim: i32,
    pub vocab_size: i32,
    pub pad_token_id: u32,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: HashMap<String, RopeConfig>,
    #[serde(default)]
    pub per_layer_config: HashMap<String, HeadConfig>,
    pub dtype: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f32,
    pub rope_type: String,
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct HeadConfig {
    pub head_dim: Option<i32>,
    pub num_attention_heads: Option<i32>,
    pub num_key_value_heads: Option<i32>,
}
#[derive(Deserialize)]
struct Config {
    model_type: String,
    text_config: TextConfig,
}

pub fn is_checkpoint(path: &Path) -> Result<bool> {
    let file = path.join("config.json");
    if !file.exists() {
        return Ok(false);
    }
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
    Ok(value["model_type"].as_str() == Some("embedding_gemma2"))
}

pub fn checkpoint_supports_images(path: &Path) -> Result<bool> {
    let value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    Ok(vision::validate_config(&value)?.is_some())
}

pub fn checkpoint_supports_audio(path: &Path) -> Result<bool> {
    let value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    audio::validate_config(&value)
}

pub fn validate_config(value: &serde_json::Value) -> Result<TextConfig> {
    vision::validate_config(value)?;
    audio::validate_config(value)?;
    let config: Config = serde_json::from_value(value.clone())?;
    ensure!(
        config.model_type == "embedding_gemma2",
        "unsupported embedding architecture"
    );
    let c = config.text_config;
    ensure!(
        c.model_type == "embedding_gemma2_text",
        "unsupported text embedding architecture"
    );
    ensure!(
        c.hidden_size > 0
            && c.intermediate_size > 0
            && c.hidden_size_per_layer_input > 0
            && c.vocab_size > 0
            && c.num_hidden_layers > 0,
        "invalid embedding encoder dimensions"
    );
    ensure!(
        c.embedding_dim == 768,
        "EmbeddingGemma 2 requires 768 output dimensions"
    );
    ensure!(
        c.dtype == "bfloat16" || c.dtype == "float32",
        "EmbeddingGemma 2 requires BF16 or FP32 activations"
    );
    ensure!(
        c.rms_norm_eps.is_finite() && c.rms_norm_eps > 0.0 && c.sliding_window > 0,
        "invalid embedding normalization or window"
    );
    ensure!(
        c.layer_types.len() == c.num_hidden_layers,
        "embedding layer_types count mismatch"
    );
    for (i, kind) in c.layer_types.iter().enumerate() {
        ensure!(
            kind == "full_attention" || kind == "sliding_attention",
            "unsupported embedding attention kind"
        );
        let rope = c
            .rope_parameters
            .get(kind)
            .ok_or_else(|| anyhow::anyhow!("missing embedding RoPE configuration"))?;
        ensure!(
            rope.rope_type == "default" && rope.rope_theta.is_finite() && rope.rope_theta > 0.0,
            "unsupported embedding RoPE configuration"
        );
        let (heads, kv, dim) = c.heads(i);
        ensure!(
            heads > 0 && kv > 0 && heads % kv == 0 && dim > 0 && dim % 2 == 0,
            "invalid embedding attention dimensions"
        );
    }
    if let Some(q) = value
        .get("quantization")
        .or_else(|| value.get("quantization_config"))
    {
        ensure!(
            q["mode"] == "affine" && q["bits"] == 4 && q["group_size"] == 64,
            "EmbeddingGemma 2 currently supports BF16 and affine 4bit group size 64"
        );
    }
    Ok(c)
}
impl TextConfig {
    fn heads(&self, index: usize) -> (i32, i32, i32) {
        let o = self.per_layer_config.get(&format!("{index:02}"));
        (
            o.and_then(|v| v.num_attention_heads)
                .unwrap_or(self.num_attention_heads),
            o.and_then(|v| v.num_key_value_heads)
                .unwrap_or(self.num_key_value_heads),
            o.and_then(|v| v.head_dim).unwrap_or(self.head_dim),
        )
    }
}
fn scaled(x: &Array, value: f32) -> Result<Array> {
    let scalar: Array = (&[value][..], ()).try_into()?;
    Ok(x * &scalar.astype(x.dtype())?)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: i32,
    kv: i32,
    dim: i32,
    theta: f32,
    eps: f32,
}
impl Attention {
    fn load(loader: &Loader, prefix: &str, c: &TextConfig, index: usize) -> Result<Self> {
        let (heads, kv, dim) = c.heads(index);
        Ok(Self {
            q: Linear::from_loader(loader, &format!("{prefix}.q_proj"))?,
            k: Linear::from_loader(loader, &format!("{prefix}.k_proj"))?,
            v: Linear::from_loader(loader, &format!("{prefix}.v_proj"))?,
            out: Linear::from_loader(loader, &format!("{prefix}.o_proj"))?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), c.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), c.rms_norm_eps)?,
            heads,
            kv,
            dim,
            theta: c.rope_parameters[&c.layer_types[index]].rope_theta,
            eps: c.rms_norm_eps,
        })
    }
    fn rotary(&self, x: &Array, seq: i32) -> Result<Array> {
        // Match the upstream FP32 angles followed by BF16 cos/sin rounding.
        let half = self.dim / 2;
        let powers: Vec<f32> = (0..half)
            .map(|j| (2 * j) as f32 / self.dim as f32)
            .collect();
        let powers: Array = (powers.as_slice(), (half,)).try_into()?;
        let base: Array = (&[self.theta][..], ()).try_into()?;
        let one: Array = (&[1.0_f32][..], ()).try_into()?;
        let frequencies = &one / &base.power(&powers)?;
        let positions: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let positions: Array = (positions.as_slice(), (1, seq, 1)).try_into()?;
        let angles = &positions * &frequencies;
        let angles = mlx::ops::shape::concatenate(&[&angles, &angles], -1)?
            .reshape((1, 1, seq, self.dim))?;
        let cos = angles.cos()?.astype(x.dtype())?;
        let sin = angles.sin()?.astype(x.dtype())?;
        let s = x.shape();
        let left = x.slice((0, 0, 0, 0), (s[0], s[1], seq, self.dim / 2))?;
        let right = scaled(
            &x.slice((0, 0, 0, self.dim / 2), (s[0], s[1], seq, self.dim))?,
            -1.0,
        )?;
        let rotated = mlx::ops::shape::concatenate(&[&right, &left], -1)?;
        Ok(x * &cos + &rotated * &sin)
    }
    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        let s = x.shape();
        let (batch, seq) = (s[0], s[1]);
        let q = self
            .q_norm
            .forward(
                &self
                    .q
                    .forward(x)?
                    .reshape((batch, seq, self.heads, self.dim))?,
            )?
            .transpose_axes((0, 2, 1, 3))?;
        let k = self
            .k_norm
            .forward(
                &self
                    .k
                    .forward(x)?
                    .reshape((batch, seq, self.kv, self.dim))?,
            )?
            .transpose_axes((0, 2, 1, 3))?;
        let v = mlx::fast::rms_norm(
            &self
                .v
                .forward(x)?
                .reshape((batch, seq, self.kv, self.dim))?,
            None,
            self.eps,
        )?
        .transpose_axes((0, 2, 1, 3))?;
        let result = mlx::fast::scaled_dot_product_attention(
            &self.rotary(&q, seq)?,
            &self.rotary(&k, seq)?,
            &v,
            1.0,
            "",
            Some(mask),
            None,
        )?;
        self.out
            .forward(&result.transpose_axes((0, 2, 1, 3))?.reshape((
                batch,
                seq,
                self.heads * self.dim,
            ))?)
    }
}
struct Layer {
    attention: Attention,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    ple_gate: Linear,
    ple_projection: Linear,
    ple_norm: RmsNorm,
    scalar: Array,
}
impl Layer {
    fn load(l: &Loader, p: &str, c: &TextConfig, i: usize) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(l, &format!("{p}.self_attn"), c, i)?,
            gate: Linear::from_loader(l, &format!("{p}.mlp.gate_proj"))?,
            up: Linear::from_loader(l, &format!("{p}.mlp.up_proj"))?,
            down: Linear::from_loader(l, &format!("{p}.mlp.down_proj"))?,
            input_norm: RmsNorm::from_loader(l, &format!("{p}.input_layernorm"), c.rms_norm_eps)?,
            post_attn_norm: RmsNorm::from_loader(
                l,
                &format!("{p}.post_attention_layernorm"),
                c.rms_norm_eps,
            )?,
            pre_ff_norm: RmsNorm::from_loader(
                l,
                &format!("{p}.pre_feedforward_layernorm"),
                c.rms_norm_eps,
            )?,
            post_ff_norm: RmsNorm::from_loader(
                l,
                &format!("{p}.post_feedforward_layernorm"),
                c.rms_norm_eps,
            )?,
            ple_gate: Linear::from_loader(l, &format!("{p}.ple_block.per_layer_input_gate"))?,
            ple_projection: Linear::from_loader(l, &format!("{p}.ple_block.per_layer_projection"))?,
            ple_norm: RmsNorm::from_loader(
                l,
                &format!("{p}.ple_block.post_per_layer_input_norm"),
                c.rms_norm_eps,
            )?,
            scalar: l.tensor(&format!("{p}.layer_scalar"))?.clone(),
        })
    }
    fn forward(&self, x: &Array, mask: &Array, ple: &Array) -> Result<Array> {
        let h = x + &self
            .post_attn_norm
            .forward(&self.attention.forward(&self.input_norm.forward(x)?, mask)?)?;
        let ff = self.pre_ff_norm.forward(&h)?;
        let activated = gelu_tanh(&self.gate.forward(&ff)?, ().into())? * self.up.forward(&ff)?;
        let h = &h + &self.post_ff_norm.forward(&self.down.forward(&activated)?)?;
        let gate = gelu_tanh(&self.ple_gate.forward(&h)?, ().into())? * ple;
        let h = &h
            + &self
                .ple_norm
                .forward(&self.ple_projection.forward(&gate)?)?;
        Ok(&h * &self.scalar)
    }
}

struct PreparedSample {
    ids: Vec<u32>,
    media_spans: Vec<(usize, usize)>,
}

pub struct EmbeddingGemma2Model {
    config: TextConfig,
    tokenizer: Tokenizer,
    embedding: Embedding,
    ple_projection: Linear,
    ple_norm: RmsNorm,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Linear,
    vision: Option<vision::ImageEncoder>,
    audio: Option<audio::AudioEncoder>,
    weight_bytes: usize,
}
#[derive(Debug)]
pub struct EmbeddingOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub input_tokens: usize,
}

impl EmbeddingGemma2Model {
    pub fn load(path: &Path) -> Result<Self> {
        let l = Loader::open_multimodal_embedding(path)?;
        let c = validate_config(l.config_raw_value())?;
        if let Some(q) = l.quant_meta() {
            ensure!(
                q.mode == QuantMode::Affine && q.bits == 4 && q.group_size == 64,
                "unsupported embedding quantization"
            );
        }
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            layers.push(Layer::load(&l, &format!("layers.{i}"), &c, i)?);
        }
        Ok(Self {
            vision: vision::ImageEncoder::load(&l, path)?,
            audio: audio::AudioEncoder::load(&l, path)?,
            tokenizer: Tokenizer::from_loader(&l)?,
            embedding: Embedding::from_loader(&l, "embed_tokens")?,
            ple_projection: Linear::from_loader(&l, "ple.per_layer_model_projection")?,
            ple_norm: RmsNorm::from_loader(&l, "ple.per_layer_projection_norm", c.rms_norm_eps)?,
            norm: RmsNorm::from_loader(&l, "norm", c.rms_norm_eps)?,
            output: Linear::from_loader(&l, "embedding_projection")?,
            weight_bytes: l.loaded_tensor_bytes(),
            layers,
            config: c,
        })
    }
    pub fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    pub fn supports_audio(&self) -> bool {
        self.audio.is_some()
    }

    pub fn encode_inputs(
        &self,
        inputs: &[EmbeddingSample],
        dimensions: usize,
    ) -> Result<EmbeddingOutput> {
        ensure!(
            !inputs.is_empty() && inputs.len() <= MAX_BATCH_SIZE,
            "embedding input requires 1..=32 samples"
        );
        ensure!(
            SUPPORTED_DIMENSIONS.contains(&dimensions),
            "unsupported embedding dimensions"
        );
        let mut audio_count = 0;
        let mut audio_samples = 0usize;
        let mut image_count = 0;
        let mut image_bytes = 0usize;
        let mut pixels = 0u64;
        for sample in inputs {
            ensure!(
                sample.content.len() <= 64,
                "embedding input content exceeds 64 parts"
            );
            for part in &sample.content {
                if let EmbeddingContent::Audio(audio) = part {
                    audio_count += 1;
                    audio_samples = audio_samples.saturating_add(audio.samples().len());
                }
                if let EmbeddingContent::Image(bytes) = part {
                    image_count += 1;
                    image_bytes = image_bytes.saturating_add(bytes.len());
                    let (w, h, _) = crate::core::image_input::inspect_image(bytes)
                        .map_err(|e| anyhow::anyhow!("invalid image input: {e}"))?;
                    pixels += u64::from(w) * u64::from(h);
                }
            }
        }
        ensure!(
            image_count <= crate::core::image_input::MAX_IMAGE_COUNT
                && image_bytes <= crate::core::image_input::MAX_TOTAL_IMAGE_BYTES
                && pixels <= crate::core::image_input::MAX_TOTAL_IMAGE_PIXELS,
            "embedding image input exceeds the aggregate image budget"
        );
        ensure!(
            audio_count <= crate::core::audio_input::MAX_AUDIO_COUNT
                && audio_samples <= crate::core::audio_input::MAX_TOTAL_AUDIO_SAMPLES,
            "embedding audio input exceeds the aggregate audio budget"
        );
        // Validate and tokenize every sample before allocating media activations.
        let prepared = inputs
            .iter()
            .map(|sample| self.prepare_sample(sample))
            .collect::<Result<Vec<_>>>()?;
        let mut input_tokens = 0;
        let mut embeddings = Vec::with_capacity(inputs.len());
        for (
            sample,
            PreparedSample {
                ids,
                media_spans: spans,
            },
        ) in inputs.iter().zip(prepared)
        {
            input_tokens += ids.len();
            if spans.is_empty() {
                embeddings.push(self.encode_tokens(&ids, dimensions)?);
                continue;
            }
            let text_ids: Vec<u32> = ids
                .iter()
                .map(|id| {
                    if [258880, 258881].contains(id) {
                        self.config.pad_token_id
                    } else {
                        *id
                    }
                })
                .collect();
            let tokens: Array = (text_ids.as_slice(), (1, ids.len() as i32)).try_into()?;
            let x = self
                .embedding
                .forward(&tokens)?
                .astype(self.activation_dtype())?;
            let text = scaled(&x, (self.config.hidden_size as f32).sqrt())?;
            let mut pieces = Vec::new();
            let mut cursor = 0;
            for ((start, count), part) in
                spans.into_iter().zip(sample.content.iter().filter(|part| {
                    matches!(
                        part,
                        EmbeddingContent::Image(_) | EmbeddingContent::Audio(_)
                    )
                }))
            {
                if start > cursor {
                    pieces.push(text.slice(
                        (0, cursor as i32, 0),
                        (1, start as i32, self.config.hidden_size),
                    )?);
                }
                let features = match part {
                    EmbeddingContent::Image(bytes) => self
                        .vision
                        .as_ref()
                        .expect("validated image encoder")
                        .encode(bytes)?,
                    EmbeddingContent::Audio(audio) => self
                        .audio
                        .as_ref()
                        .expect("validated audio encoder")
                        .encode(audio)?,
                    _ => unreachable!("filtered media"),
                }
                .astype(self.activation_dtype())?;
                // Materialize each media segment before preparing the next so a sample
                // never retains several lazy encoder activation graphs.
                features.eval()?;
                ensure!(
                    features.shape().as_slice() == [1, count as i32, self.config.hidden_size],
                    "media input feature count mismatch"
                );
                pieces.push(features);
                cursor = start + count;
            }
            if cursor < ids.len() {
                pieces.push(text.slice(
                    (0, cursor as i32, 0),
                    (1, ids.len() as i32, self.config.hidden_size),
                )?);
            }
            let h = mlx::ops::concatenate(&pieces.iter().collect::<Vec<_>>(), 1)?;
            embeddings.push(self.encode_hidden(h, dimensions)?);
        }
        Ok(EmbeddingOutput {
            embeddings,
            input_tokens,
        })
    }

    fn prepare_sample(&self, sample: &EmbeddingSample) -> Result<PreparedSample> {
        ensure!(
            !sample.content.is_empty(),
            "embedding input content must not be empty"
        );
        let mut prompt = String::new();
        let mut counts = Vec::new();
        let mut text_bytes = 0usize;
        for part in &sample.content {
            match part {
                EmbeddingContent::Text(text) => {
                    text_bytes = text_bytes.saturating_add(text.len());
                    ensure!(
                        text_bytes <= 128 * 1024,
                        "embedding input text is too large"
                    );
                    ensure!(
                        ![
                            "<|image|>",
                            "<|image>",
                            "<image|>",
                            "<|audio|>",
                            "<|audio>",
                            "<audio|>",
                            "<|video|>"
                        ]
                        .iter()
                        .any(|token| text.contains(token)),
                        "embedding input media placeholders must be supplied as typed content"
                    );
                    prompt.push_str(text);
                }
                EmbeddingContent::Image(bytes) => {
                    let encoder = self.vision.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("embedding input images are unsupported by this checkpoint")
                    })?;
                    let count = encoder.token_count(bytes)?;
                    counts.push((258880, count));
                    prompt.push_str("<|image>");
                    for _ in 0..count {
                        prompt.push_str("<|image|>");
                    }
                    prompt.push_str("<image|>");
                }
                EmbeddingContent::Audio(audio) => {
                    let encoder = self.audio.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("embedding input audio is unsupported by this checkpoint")
                    })?;
                    let count = encoder.token_count(audio);
                    counts.push((258881, count));
                    prompt.push_str("<|audio>");
                    for _ in 0..count {
                        prompt.push_str("<|audio|>");
                    }
                    prompt.push_str("<audio|>");
                }
            }
        }
        ensure!(
            !prompt.trim().is_empty(),
            "embedding input must not be empty"
        );
        let ids = self.tokenizer.encode(&prompt, true)?;
        ensure!(
            !ids.iter().any(|id| [258884].contains(id)),
            "embedding input video is unsupported"
        );
        ensure!(
            !ids.is_empty() && ids.len() <= MAX_INPUT_TOKENS,
            "embedding input exceeds the 8192 token limit"
        );
        let mut spans = Vec::new();
        let mut cursor = 0;
        for (token_id, count) in counts {
            let start = ids[cursor..]
                .iter()
                .position(|id| *id == token_id)
                .map(|offset| cursor + offset)
                .ok_or_else(|| anyhow::anyhow!("media input placeholder token missing"))?;
            ensure!(
                ids.get(start..start + count)
                    .is_some_and(|part| part.iter().all(|id| *id == token_id)),
                "media input placeholder count mismatch"
            );
            spans.push((start, count));
            cursor = start + count;
        }
        ensure!(
            !ids[cursor..].iter().any(|id| [258880, 258881].contains(id)),
            "media input placeholder count mismatch"
        );
        Ok(PreparedSample {
            ids,
            media_spans: spans,
        })
    }

    fn activation_dtype(&self) -> Dtype {
        if self.config.dtype == "float32" {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        }
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn encode(&self, texts: &[String], dimensions: usize) -> Result<EmbeddingOutput> {
        let samples: Vec<_> = texts
            .iter()
            .map(|text| EmbeddingSample {
                content: vec![EmbeddingContent::Text(text.clone())],
            })
            .collect();
        self.encode_inputs(&samples, dimensions)
    }
    pub fn encode_tokens(&self, ids: &[u32], dimensions: usize) -> Result<Vec<f32>> {
        ensure!(
            !ids.is_empty() && ids.len() <= MAX_INPUT_TOKENS,
            "embedding token count must be 1..=8192"
        );
        ensure!(
            SUPPORTED_DIMENSIONS.contains(&dimensions),
            "unsupported embedding dimensions"
        );
        ensure!(
            ids.iter()
                .all(|id| *id < self.config.vocab_size as u32 && *id != self.config.pad_token_id),
            "invalid embedding token ID or padding token"
        );
        let seq = ids.len() as i32;
        let tokens: Array = (ids, (1, seq)).try_into()?;
        let x = self
            .embedding
            .forward(&tokens)?
            .astype(if self.config.dtype == "float32" {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            })?;
        self.encode_hidden(
            scaled(&x, (self.config.hidden_size as f32).sqrt())?,
            dimensions,
        )
    }
    fn encode_hidden(&self, mut h: Array, dimensions: usize) -> Result<Vec<f32>> {
        let seq = h.shape().as_slice()[1];
        let ple = self.ple_projection.forward(&h)?;
        let ple = self.ple_norm.forward(
            &scaled(&ple, (self.config.hidden_size as f32).powf(-0.5))?.reshape((
                1,
                seq,
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input,
            ))?,
        )?;
        let mask = bidirectional_window_mask(seq, self.config.sliding_window)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let p = ple
                .slice(
                    (0, 0, i as i32, 0),
                    (
                        1,
                        seq,
                        i as i32 + 1,
                        self.config.hidden_size_per_layer_input,
                    ),
                )?
                .reshape((1, seq, self.config.hidden_size_per_layer_input))?;
            // Full attention uses an all-visible broadcast mask.
            let full: Array = (&[true][..], (1, 1, 1, 1)).try_into()?;
            h = layer.forward(
                &h,
                if self.config.layer_types[i] == "sliding_attention" {
                    &mask
                } else {
                    &full
                },
                &p,
            )?;
            h.eval()?;
        }
        let projected = self.output.forward(&self.norm.forward(&h)?)?;
        let pooled = projected
            .astype(Dtype::Float32)?
            .mean(1, false)?
            .slice((0, 0), (1, dimensions as i32))?;
        let norm = pooled.square()?.sum(-1, true)?.sqrt()?;
        let result = (&pooled / &norm).to_vec::<f32>()?;
        if result.iter().any(|x| !x.is_finite()) {
            bail!("embedding encoder returned non-finite values");
        }
        Ok(result)
    }
}
fn bidirectional_window_mask(length: i32, window: usize) -> Result<Array> {
    let mut values = Vec::with_capacity(length as usize * length as usize);
    for q in 0..length {
        for k in 0..length {
            values.push(q.abs_diff(k) as usize <= window);
        }
    }
    Ok((values.as_slice(), (1, 1, length, length)).try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_is_bidirectional_and_inclusive() {
        if let Ok(dir) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib")).unwrap();
        }
        let mask = bidirectional_window_mask(4, 1)
            .unwrap()
            .to_vec::<bool>()
            .unwrap();
        assert_eq!(
            mask,
            vec![
                true, true, false, false, true, true, true, false, false, true, true, true, false,
                false, true, true
            ]
        );
    }
}
