use super::{
    config::{AudioConfig, TextConfig},
    features, tokenizer, Config,
};
use crate::{PcmBuffer, SessionControl};
use anyhow::{anyhow, ensure, Context, Result};
use ironmlx_core::{
    nn::{Embedding, LayerNorm, Linear, RmsNorm},
    weights::{QuantMeta, QuantMode, WeightMap, WeightSource},
};
use mlx::{ops, Array, Dtype};
use serde::Serialize;
use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub language: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct Transcription {
    pub text: String,
    pub language: String,
    pub duration: f64,
    pub generated_tokens: usize,
    #[serde(skip)]
    pub segments: Vec<TranscriptSegment>,
    #[serde(skip)]
    pub raw_text: String,
    #[serde(skip)]
    pub stable_text: String,
}

/// One model belongs to one inference worker. Only owned PCM/results cross threads.
pub struct Qwen3Asr {
    pub(super) config: Config,
    weights: WeightMap,
    pub(super) tokenizer: tokenizers::Tokenizer,
    pub(super) embed: Embedding,
    encoder: Vec<EncoderLayer>,
    pub(super) decoder: Vec<DecoderLayer>,
    final_audio_norm: LayerNorm,
    pub(super) final_text_norm: RmsNorm,
    conv_out: Linear,
    proj1: Linear,
    proj2: Linear,
    weight_bytes: u64,
    pub(super) timestamp_head: Option<Linear>,
}

struct EncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    fc1: Linear,
    fc2: Linear,
}
pub(super) struct DecoderLayer {
    norm1: RmsNorm,
    norm2: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub(super) type Cache = Option<(Array, Array)>;

fn linear(w: &WeightMap, p: &str) -> Result<Linear> {
    Linear::from_loader(w, p)
}
fn gelu(x: &Array) -> Result<Array> {
    let scalar = |value: f32| Array::try_from((&[value][..], []))?.astype(x.dtype());
    let cdf = ops::add(
        &ops::erf(&ops::divide(x, &scalar(std::f32::consts::SQRT_2)?)?)?,
        &scalar(1.)?,
    )?;
    Ok(ops::divide(&ops::multiply(x, &cdf)?, &scalar(2.)?)?)
}

impl Qwen3Asr {
    pub fn load(directory: &Path) -> Result<Self> {
        super::preflight_model_metadata(directory)?;
        let config = Config::load(directory)?;
        let file = directory.join("model.safetensors");
        let weight_bytes = file.metadata()?.len();
        let (tensors, _) =
            mlx::io::load_safetensors(file.to_str().context("invalid ASR checkpoint path")?)?;
        ensure!(!tensors.is_empty(), "empty Qwen3-ASR weights");
        let weights = WeightMap::new(
            tensors,
            Some(QuantMeta {
                bits: config.quantization.bits,
                group_size: config.quantization.group_size,
                mode: QuantMode::Affine,
            }),
            HashMap::new(),
        );
        validate_weights(&weights, &config)?;
        let a = &config.thinker_config.audio_config;
        let t = &config.thinker_config.text_config;
        let mut encoder = Vec::with_capacity(a.encoder_layers);
        for i in 0..a.encoder_layers {
            let p = format!("audio_tower.layers.{i}");
            encoder.push(EncoderLayer {
                norm1: LayerNorm::from_loader(
                    &weights,
                    &format!("{p}.self_attn_layer_norm"),
                    1e-5,
                )?,
                norm2: LayerNorm::from_loader(&weights, &format!("{p}.final_layer_norm"), 1e-5)?,
                q: linear(&weights, &format!("{p}.self_attn.q_proj"))?,
                k: linear(&weights, &format!("{p}.self_attn.k_proj"))?,
                v: linear(&weights, &format!("{p}.self_attn.v_proj"))?,
                out: linear(&weights, &format!("{p}.self_attn.out_proj"))?,
                fc1: linear(&weights, &format!("{p}.fc1"))?,
                fc2: linear(&weights, &format!("{p}.fc2"))?,
            });
        }
        let mut decoder = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            decoder.push(DecoderLayer {
                norm1: RmsNorm::from_loader(
                    &weights,
                    &format!("{p}.input_layernorm"),
                    t.rms_norm_eps,
                )?,
                norm2: RmsNorm::from_loader(
                    &weights,
                    &format!("{p}.post_attention_layernorm"),
                    t.rms_norm_eps,
                )?,
                q_norm: RmsNorm::from_loader(
                    &weights,
                    &format!("{p}.self_attn.q_norm"),
                    t.rms_norm_eps,
                )?,
                k_norm: RmsNorm::from_loader(
                    &weights,
                    &format!("{p}.self_attn.k_norm"),
                    t.rms_norm_eps,
                )?,
                q: linear(&weights, &format!("{p}.self_attn.q_proj"))?,
                k: linear(&weights, &format!("{p}.self_attn.k_proj"))?,
                v: linear(&weights, &format!("{p}.self_attn.v_proj"))?,
                out: linear(&weights, &format!("{p}.self_attn.o_proj"))?,
                gate: linear(&weights, &format!("{p}.mlp.gate_proj"))?,
                up: linear(&weights, &format!("{p}.mlp.up_proj"))?,
                down: linear(&weights, &format!("{p}.mlp.down_proj"))?,
            });
        }
        let model = Self {
            tokenizer: tokenizer::load(directory)?,
            embed: Embedding::from_loader(&weights, "model.embed_tokens")?,
            final_audio_norm: LayerNorm::from_loader(&weights, "audio_tower.ln_post", 1e-5)?,
            final_text_norm: RmsNorm::from_loader(&weights, "model.norm", t.rms_norm_eps)?,
            conv_out: linear(&weights, "audio_tower.conv_out")?,
            proj1: linear(&weights, "audio_tower.proj1")?,
            proj2: linear(&weights, "audio_tower.proj2")?,
            timestamp_head: if config.is_aligner() {
                Some(linear(&weights, "lm_head")?)
            } else {
                None
            },
            config,
            weights,
            encoder,
            decoder,
            weight_bytes,
        };
        // Force reads while load errors still have model-loading context.
        for tensor in model.weights.tensors().values() {
            tensor.eval()?;
        }
        Ok(model)
    }

    pub fn is_aligner(&self) -> bool {
        self.config.is_aligner()
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn validate_language(&self, language: Option<&str>) -> Result<()> {
        self.config.language(language).map(|_| ())
    }

    pub(super) fn encode_audio(
        &self,
        features: &Array,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let cfg = &self.config.thinker_config.audio_config;
        let frames = features.shape().as_slice()[1] as usize;
        let chunk_size = cfg.n_window * 2;
        let max_chunk = frames.min(chunk_size);
        let mut convolution_chunks = Vec::new();
        for start in (0..frames).step_by(chunk_size) {
            control.check()?;
            let len = (frames - start).min(chunk_size);
            let mut x = features.slice([0, start as i32], [128, (start + len) as i32])?;
            if len < max_chunk {
                x = ops::concatenate(
                    &[
                        &x,
                        &Array::zeros([128, (max_chunk - len) as i32], x.dtype())?,
                    ],
                    1,
                )?;
            }
            x = x.reshape([1, 128, max_chunk as i32, 1])?;
            for i in 1..=3 {
                let p = format!("audio_tower.conv2d{i}");
                x = ops::conv2d(
                    &x,
                    self.weights.tensor(&format!("{p}.weight"))?,
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    1,
                )?;
                x = gelu(&ops::add(&x, self.weights.tensor(&format!("{p}.bias"))?)?)?;
            }
            let s = x.shape();
            let s = s.as_slice();
            x = self
                .conv_out
                .forward(
                    &x.transpose_axes([0, 2, 3, 1])?
                        .reshape([1, s[2], s[3] * s[1]])?,
                )?;
            let length = features::output_length(len) as i32;
            x = x.slice([0, 0, 0], [1, length, cfg.d_model])?;
            let half = cfg.d_model / 2;
            let mut positions = Vec::with_capacity(length as usize * cfg.d_model as usize);
            for position in 0..length {
                for cos in [false, true] {
                    for i in 0..half {
                        let angle =
                            position as f32 * (-10000f32.ln() * i as f32 / (half - 1) as f32).exp();
                        positions.push(if cos { angle.cos() } else { angle.sin() });
                    }
                }
            }
            let positions = Array::try_from((positions.as_slice(), [1, length, cfg.d_model]))?;
            x = ops::add(&x, &positions)?;
            x.eval()?;
            convolution_chunks.push(x);
        }
        let refs = convolution_chunks.iter().collect::<Vec<_>>();
        let x = ops::concatenate(&refs, 1)?;
        let total = x.shape().as_slice()[1];
        let window = features::output_length(max_chunk) * (cfg.n_window_infer / chunk_size);
        // Each attention window is independent; avoid a quadratic global mask.
        let mut windows = Vec::new();
        for start in (0..total).step_by(window) {
            let end = (start + window as i32).min(total);
            let mut hidden = x.slice([0, start, 0], [1, end, cfg.d_model])?;
            for layer in &self.encoder {
                control.check()?;
                hidden = layer.forward(&hidden, cfg)?;
                hidden.eval()?;
            }
            windows.push(hidden);
        }
        let refs = windows.iter().collect::<Vec<_>>();
        let hidden = self
            .final_audio_norm
            .forward(&ops::concatenate(&refs, 1)?)?;
        let hidden = self.proj2.forward(&gelu(&self.proj1.forward(&hidden)?)?)?;
        hidden.eval()?;
        Ok(hidden)
    }

    fn decode(
        &self,
        inputs: &Array,
        cache: &mut [Cache],
        offset: i32,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let mut hidden = inputs.clone();
        for (layer, cache) in self.decoder.iter().zip(cache.iter_mut()) {
            control.check()?;
            hidden = layer.forward(
                &hidden,
                &self.config.thinker_config.text_config,
                cache,
                offset,
            )?;
        }
        let n = hidden.shape().as_slice()[1];
        let hidden = self.final_text_norm.forward(&hidden.slice(
            [0, n - 1, 0],
            [1, n, self.config.thinker_config.text_config.hidden_size],
        )?)?;
        self.embed.as_output(&hidden)
    }

    /// Automatically segment long recordings without dropping or overlapping samples.
    pub fn transcribe(
        &mut self,
        pcm: &PcmBuffer,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: usize,
        control: &dyn SessionControl,
    ) -> Result<Transcription> {
        self.transcribe_with_progress(pcm, language, prompt, max_tokens, control, &mut |_| Ok(()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn transcribe_with_progress(
        &mut self,
        pcm: &PcmBuffer,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: usize,
        control: &dyn SessionControl,
        progress: &mut dyn FnMut(&Transcription) -> Result<()>,
    ) -> Result<Transcription> {
        let prepared = super::prepare_audio(pcm, control)?;
        let samples = prepared.samples;
        let mut output = Transcription {
            text: String::new(),
            language: String::new(),
            duration: samples.len() as f64 / 16000.,
            generated_tokens: 0,
            segments: Vec::new(),
            raw_text: String::new(),
            stable_text: String::new(),
        };
        for range in prepared.segments {
            control.check()?;
            let start = range.start as f64 / 16000.;
            let end = range.end as f64 / 16000.;
            let mut segment = samples[range].to_vec();
            segment.resize(segment.len().max(201), 0.);
            let pcm = PcmBuffer {
                format: crate::PcmFormat {
                    sample_rate: 16000,
                    channels: 1,
                },
                samples: segment,
            };
            let result = self.transcribe_window(&pcm, language, prompt, max_tokens, "", control)?;
            let delta = super::segments::append_text(&mut output.text, &result.text);
            output.generated_tokens += result.generated_tokens;
            output.segments.push(TranscriptSegment {
                start,
                end,
                text: result.text.clone(),
                language: result.language.clone(),
            });
            if !result.language.is_empty()
                && !output.language.split(',').any(|l| l == result.language)
            {
                if !output.language.is_empty() {
                    output.language.push(',');
                }
                output.language.push_str(&result.language);
            }
            let mut event = result;
            event.text = delta;
            progress(&event)?;
        }
        Ok(output)
    }

    /// Re-decode accumulating audio with Qwen's five-token rollback prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn transcribe_incremental(
        &mut self,
        pcm: &PcmBuffer,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: usize,
        previous: &str,
        update_index: usize,
        control: &dyn SessionControl,
    ) -> Result<Transcription> {
        let prefix = if update_index >= 2 {
            self.rollback_prefix(previous)?
        } else {
            String::new()
        };
        let prefix_tokens = self
            .tokenizer
            .encode(prefix.as_str(), false)
            .map_err(|e| anyhow!("ASR prefix budget: {e}"))?
            .len();
        let remaining = max_tokens
            .checked_sub(prefix_tokens)
            .filter(|value| *value > 0)
            .ok_or(crate::AudioError::GenerationLimitExceeded)?;
        let mut result =
            self.transcribe_window(pcm, language, prompt, remaining, &prefix, control)?;
        // The next update will preserve this prefix verbatim in its prompt.
        if update_index >= 1 {
            let stable = self.rollback_prefix(&result.raw_text)?;
            result.stable_text = if language.is_none() {
                stable.split_once("<asr_text>").map_or("", |(_, text)| text)
            } else {
                &stable
            }
            .trim()
            .to_owned();
        }
        Ok(result)
    }

    fn rollback_prefix(&self, previous: &str) -> Result<String> {
        let encoded = self
            .tokenizer
            .encode(previous, false)
            .map_err(|e| anyhow!("ASR prefix: {e}"))?;
        let mut end = encoded.len().saturating_sub(5);
        while end > 0 {
            let prefix = self
                .tokenizer
                .decode(&encoded.get_ids()[..end], false)
                .map_err(|e| anyhow!("ASR prefix decode: {e}"))?;
            if !prefix.contains('\u{fffd}') {
                return Ok(prefix);
            }
            end -= 1;
        }
        Ok(String::new())
    }

    /// Offline recognition; runtime owns cancellation, admission, and deadlines.
    fn transcribe_window(
        &mut self,
        pcm: &PcmBuffer,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: usize,
        prefix_text: &str,
        control: &dyn SessionControl,
    ) -> Result<Transcription> {
        control.check()?;
        ensure!(
            !self.is_aligner(),
            "ForcedAligner requires audio and text alignment, not transcription"
        );
        ensure!(
            pcm.format.channels > 0
                && pcm.format.channels <= 2
                && (8000..=96000).contains(&pcm.format.sample_rate),
            "unsupported ASR PCM format"
        );
        ensure!(
            pcm.samples
                .len()
                .is_multiple_of(usize::from(pcm.format.channels)),
            "incomplete ASR PCM frame"
        );
        ensure!(
            max_tokens > 0 && max_tokens <= 8192,
            "ASR max_tokens must be 1..8192"
        );
        ensure!(
            prompt.is_none_or(|p| p.len() <= 16384),
            "ASR prompt exceeds 16384 bytes"
        );
        let language = self.config.language(language)?;
        let mono = pcm
            .samples
            .chunks_exact(usize::from(pcm.format.channels))
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect::<Vec<_>>();
        let samples = crate::signal::resample_mono(&mono, pcm.format.sample_rate, 16000, control)?;
        ensure!(
            samples.len() <= 30 * features::SAMPLE_RATE,
            "ASR audio exceeds 30 seconds"
        );
        let duration = samples.len() as f64 / features::SAMPLE_RATE as f64;
        let features = features::extract(&samples)?;
        let audio = self
            .encode_audio(&features, control)?
            .astype(self.embed.output_dtype())?;
        let num_audio = audio.shape().as_slice()[1] as usize;
        let system = prompt
            .filter(|s| !s.is_empty())
            .map(|s| format!("{s}\n"))
            .unwrap_or_default();
        let before =
            format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>");
        let prefix = language
            .as_ref()
            .map(|s| format!("language {s}<asr_text>"))
            .unwrap_or_default();
        let after =
            format!("<|audio_end|><|im_end|>\n<|im_start|>assistant\n{prefix}{prefix_text}");
        let encode = |s: &str| {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow!("ASR encode: {e}"))
        };
        let before = encode(&before)?;
        let after = encode(&after)?;
        let token_array = |ids: &[u32]| Array::try_from((ids, [1, ids.len() as i32]));
        let left = self.embed.forward(&token_array(&before)?)?;
        let right = self.embed.forward(&token_array(&after)?)?;
        let mut inputs = ops::concatenate(&[&left, &audio, &right], 1)?;
        let prompt_len = before.len() + num_audio + after.len();
        ensure!(
            prompt_len + max_tokens
                <= self
                    .config
                    .thinker_config
                    .text_config
                    .max_position_embeddings,
            "ASR context capacity exceeded"
        );
        let mut caches = (0..self.decoder.len()).map(|_| None).collect::<Vec<_>>();
        let mut offset = 0i32;
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            control.check()?;
            let logits = self.decode(&inputs, &mut caches, offset, control)?;
            let token = logits.argmax(-1, false)?.item::<u32>()?;
            // Materialize KV before discarding the computation graph for this step.
            for (k, v) in caches.iter().flatten() {
                k.eval()?;
                v.eval()?;
            }
            if matches!(token, 151643 | 151645) {
                let raw = format!(
                    "{prefix_text}{}",
                    self.tokenizer
                        .decode(&generated, false)
                        .map_err(|e| anyhow!("ASR decode: {e}"))?
                );
                let (detected, text) = if let Some((detected, text)) = raw
                    .strip_prefix("language ")
                    .and_then(|s| s.split_once("<asr_text>"))
                {
                    (detected.trim().to_owned(), text.to_owned())
                } else {
                    (language.clone().unwrap_or_default(), raw.clone())
                };
                return Ok(Transcription {
                    text: text.trim().to_owned(),
                    language: detected.clone(),
                    duration,
                    generated_tokens: generated.len(),
                    segments: vec![TranscriptSegment {
                        start: 0.,
                        end: duration,
                        text: text.trim().to_owned(),
                        language: detected.clone(),
                    }],
                    raw_text: raw.clone(),
                    stable_text: String::new(),
                });
            }
            generated.push(token);
            offset += inputs.shape().as_slice()[1];
            inputs = self.embed.forward(&token_array(&[token])?)?;
        }
        Err(crate::AudioError::GenerationLimitExceeded.into())
    }
}

fn validate_weights(weights: &WeightMap, config: &Config) -> Result<()> {
    let a = &config.thinker_config.audio_config;
    let t = &config.thinker_config.text_config;
    let mut expected = HashMap::<String, (Vec<i32>, Dtype)>::new();
    let mut dense = |name: String, shape: Vec<i32>| {
        expected.insert(name, (shape, Dtype::Bfloat16));
    };
    for i in 1..=3 {
        dense(
            format!("audio_tower.conv2d{i}.weight"),
            vec![480, 3, 3, if i == 1 { 1 } else { 480 }],
        );
        dense(format!("audio_tower.conv2d{i}.bias"), vec![480]);
    }
    dense("audio_tower.conv_out.weight".into(), vec![a.d_model, 7680]);
    for i in 0..a.encoder_layers {
        let p = format!("audio_tower.layers.{i}");
        for (name, out, input) in [
            ("self_attn.q_proj", a.d_model, a.d_model),
            ("self_attn.k_proj", a.d_model, a.d_model),
            ("self_attn.v_proj", a.d_model, a.d_model),
            ("self_attn.out_proj", a.d_model, a.d_model),
            ("fc1", a.encoder_ffn_dim, a.d_model),
            ("fc2", a.d_model, a.encoder_ffn_dim),
        ] {
            dense(format!("{p}.{name}.weight"), vec![out, input]);
            dense(format!("{p}.{name}.bias"), vec![out]);
        }
        for name in ["self_attn_layer_norm", "final_layer_norm"] {
            for suffix in ["weight", "bias"] {
                dense(format!("{p}.{name}.{suffix}"), vec![a.d_model]);
            }
        }
    }
    for suffix in ["weight", "bias"] {
        dense(format!("audio_tower.ln_post.{suffix}"), vec![a.d_model]);
    }
    for (name, out) in [("proj1", a.d_model), ("proj2", a.output_dim)] {
        dense(format!("audio_tower.{name}.weight"), vec![out, a.d_model]);
        dense(format!("audio_tower.{name}.bias"), vec![out]);
    }
    dense("model.norm.weight".into(), vec![t.hidden_size]);
    for i in 0..t.num_hidden_layers {
        for name in ["input_layernorm", "post_attention_layernorm"] {
            dense(
                format!("model.layers.{i}.{name}.weight"),
                vec![t.hidden_size],
            );
        }
        for name in ["q_norm", "k_norm"] {
            dense(
                format!("model.layers.{i}.self_attn.{name}.weight"),
                vec![t.head_dim],
            );
        }
    }
    let mut quant = |name: String, out: i32, input: i32| {
        expected.insert(
            format!("{name}.weight"),
            (vec![out, input / 4], Dtype::Uint32),
        );
        for suffix in ["scales", "biases"] {
            expected.insert(
                format!("{name}.{suffix}"),
                (vec![out, input / 64], Dtype::Bfloat16),
            );
        }
    };
    quant("model.embed_tokens".into(), t.vocab_size, t.hidden_size);
    if config.is_aligner() {
        quant("lm_head".into(), 5000, t.hidden_size);
    }
    for i in 0..t.num_hidden_layers {
        for (name, out, input) in [
            (
                "self_attn.q_proj",
                t.num_attention_heads * t.head_dim,
                t.hidden_size,
            ),
            (
                "self_attn.k_proj",
                t.num_key_value_heads * t.head_dim,
                t.hidden_size,
            ),
            (
                "self_attn.v_proj",
                t.num_key_value_heads * t.head_dim,
                t.hidden_size,
            ),
            (
                "self_attn.o_proj",
                t.hidden_size,
                t.num_attention_heads * t.head_dim,
            ),
            ("mlp.gate_proj", t.intermediate_size, t.hidden_size),
            ("mlp.up_proj", t.intermediate_size, t.hidden_size),
            ("mlp.down_proj", t.hidden_size, t.intermediate_size),
        ] {
            quant(format!("model.layers.{i}.{name}"), out, input);
        }
    }
    ensure!(
        expected.len() == weights.tensors().len(),
        "Qwen3-ASR tensor inventory mismatch: expected {}, found {}",
        expected.len(),
        weights.tensors().len()
    );
    for (name, (shape, dtype)) in expected {
        let tensor = weights.tensor(&name)?;
        ensure!(
            tensor.shape().as_slice() == shape && tensor.dtype() == dtype,
            "Qwen3-ASR tensor {name}: expected {shape:?} {dtype:?}, found {:?} {:?}",
            tensor.shape(),
            tensor.dtype()
        );
    }
    Ok(())
}

impl EncoderLayer {
    fn forward(&self, x: &Array, cfg: &AudioConfig) -> Result<Array> {
        let s = x.shape().as_slice()[1];
        let h = cfg.encoder_attention_heads;
        let d = cfg.d_model / h;
        let normalized = self.norm1.forward(x)?;
        let shape = |a: Array| a.reshape([1, s, h, d])?.transpose_axes([0, 2, 1, 3]);
        let q = shape(self.q.forward(&normalized)?)?;
        let k = shape(self.k.forward(&normalized)?)?;
        let v = shape(self.v.forward(&normalized)?)?;
        let attended = mlx::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (d as f32).sqrt().recip(),
            "",
            None,
            None,
        )?;
        let attended = attended
            .transpose_axes([0, 2, 1, 3])?
            .reshape([1, s, cfg.d_model])?;
        let x = ops::add(x, &self.out.forward(&attended)?)?;
        Ok(ops::add(
            &x,
            &self
                .fc2
                .forward(&gelu(&self.fc1.forward(&self.norm2.forward(&x)?)?)?)?,
        )?)
    }
}
impl DecoderLayer {
    pub(super) fn forward(
        &self,
        x: &Array,
        cfg: &TextConfig,
        cache: &mut Cache,
        offset: i32,
    ) -> Result<Array> {
        let s = x.shape().as_slice()[1];
        let normalized = self.norm1.forward(x)?;
        let q = self
            .q_norm
            .forward(&self.q.forward(&normalized)?.reshape([
                1,
                s,
                cfg.num_attention_heads,
                cfg.head_dim,
            ])?)?
            .transpose_axes([0, 2, 1, 3])?;
        let k = self
            .k_norm
            .forward(&self.k.forward(&normalized)?.reshape([
                1,
                s,
                cfg.num_key_value_heads,
                cfg.head_dim,
            ])?)?
            .transpose_axes([0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&normalized)?
            .reshape([1, s, cfg.num_key_value_heads, cfg.head_dim])?
            .transpose_axes([0, 2, 1, 3])?;
        // All three ASR position axes are equal in the official processor, so
        // interleaved MRoPE reduces to this split-half one-dimensional RoPE.
        let q = mlx::fast::rope(
            &q,
            cfg.head_dim,
            false,
            Some(cfg.rope_theta),
            1.,
            offset,
            None,
        )?;
        let k = mlx::fast::rope(
            &k,
            cfg.head_dim,
            false,
            Some(cfg.rope_theta),
            1.,
            offset,
            None,
        )?;
        let (k, v) = if let Some((old_k, old_v)) = cache {
            (
                ops::concatenate(&[old_k, &k], 2)?,
                ops::concatenate(&[old_v, &v], 2)?,
            )
        } else {
            (k, v)
        };
        let y = mlx::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (cfg.head_dim as f32).sqrt().recip(),
            if s > 1 { "causal" } else { "" },
            None,
            None,
        )?;
        *cache = Some((k, v));
        let y = y.transpose_axes([0, 2, 1, 3])?.reshape([
            1,
            s,
            cfg.num_attention_heads * cfg.head_dim,
        ])?;
        let x = ops::add(x, &self.out.forward(&y)?)?;
        let normalized = self.norm2.forward(&x)?;
        let gate = self.gate.forward(&normalized)?;
        let y = ops::multiply(
            &ops::multiply(&gate, &gate.sigmoid()?)?,
            &self.up.forward(&normalized)?,
        )?;
        Ok(ops::add(&x, &self.down.forward(&y)?)?)
    }
}
