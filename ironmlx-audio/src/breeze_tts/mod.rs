//! Native Breeze TTS 2: T5Gemma2 text encoding, Qwen3 backbone, depth decoder
//! and Qwen3-TTS codec. No external inference process is used.
mod codec;
mod config;
mod transformer;
use crate::{
    AudioError, Language, OutputPolicy, PcmBuffer, PcmChunk, PcmFormat, SessionControl,
    StreamingGranularity, TtsCapabilities, TtsModel, TtsRequest, TtsSession, TtsStep, TtsSummary,
};
use anyhow::{ensure, Context, Result};
use config::Config;
pub use config::{preflight, preflight_model_metadata};
use ironmlx_core::{
    nn::{Embedding, Linear},
    weights::{QuantMeta, QuantMode, WeightMap, WeightSource},
};
use mlx::{ops, Array, Dtype};
use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};
use transformer::{scalar, scale, Cache, Stack};

#[derive(Clone, Debug)]
pub struct BreezeOptions {
    pub ref_text: Option<String>,
    pub instructions: Option<String>,
    pub cfg_scale: f32,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}
impl Default for BreezeOptions {
    fn default() -> Self {
        Self {
            ref_text: None,
            instructions: None,
            cfg_scale: 1.,
            max_tokens: 750,
            temperature: 0.9,
            top_p: 1.,
            top_k: 50,
        }
    }
}
impl BreezeOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.cfg_scale.is_finite() && (1. ..=10.).contains(&self.cfg_scale),
            "cfg_scale must be within 1–10"
        );
        ensure!(
            (1..=1500).contains(&self.max_tokens),
            "max_tokens must be within 1–1500"
        );
        ensure!(
            self.temperature.is_finite() && (0. ..=2.).contains(&self.temperature),
            "temperature must be within 0–2"
        );
        ensure!(
            self.top_p.is_finite() && (0. ..=1.).contains(&self.top_p),
            "top_p must be within 0–1"
        );
        ensure!(self.top_k <= 2052, "top_k must be within 0–2052");
        for s in [&self.instructions, &self.ref_text].into_iter().flatten() {
            ensure!(
                s.len() <= 16384 && !s.trim().is_empty(),
                "Breeze instructions/ref_text must be nonempty and at most 16384 UTF-8 bytes"
            );
        }
        Ok(())
    }
}
pub struct BreezeTts {
    _weights: WeightMap,
    config: Config,
    tokenizer: tokenizers::Tokenizer,
    text_embed: Embedding,
    eoi: Array,
    text: Stack,
    text_proj: Linear,
    audio_embed: Embedding,
    backbone: Stack,
    head: Linear,
    depth_embed: Embedding,
    depth_proj: Linear,
    depth: Stack,
    depth_head: Array,
    codec: codec::Codec,
    capabilities: TtsCapabilities,
    max_output_frames: u64,
}
fn model_error(e: anyhow::Error) -> AudioError {
    // Preserve control/IO errors rather than hiding cancellation as inference failure.
    match e.downcast::<AudioError>() {
        Ok(e) => e,
        Err(e) => AudioError::InferenceFailed {
            reason: format!("{e:#}"),
        },
    }
}
fn last(x: Array) -> Result<Array> {
    let s = x.shape();
    let a = s.as_slice();
    Ok(x.slice([0, a[1] - 1, 0], [1, a[1], a[2]])?)
}
impl BreezeTts {
    pub fn load(path: &Path, max_output_frames: u64) -> Result<Self> {
        preflight(path)?;
        ensure!(
            max_output_frames > 0 && max_output_frames <= 600 * 24000,
            "invalid Breeze output capacity"
        );
        let config = Config::load(path)?;
        let quant = config.quantization.as_ref().map(|_| QuantMeta {
            group_size: 32,
            bits: 8,
            mode: QuantMode::Mxfp8,
        });
        let mut tensors = HashMap::new();
        for file in config::shards(path)? {
            let (shard, _) = mlx::io::load_safetensors(
                path.join(file).to_str().context("invalid Breeze path")?,
            )?;
            for (k, v) in shard {
                ensure!(
                    tensors.insert(k.clone(), v).is_none(),
                    "duplicate Breeze tensor {k}"
                );
            }
        }
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
        let map = index["weight_map"]
            .as_object()
            .context("invalid Breeze weight index")?;
        ensure!(
            map.len() == tensors.len() && map.keys().all(|k| tensors.contains_key(k)),
            "Breeze index/tensor mismatch"
        );
        for (k, v) in &tensors {
            ensure!(
                matches!(v.dtype(), Dtype::Bfloat16 | Dtype::Uint32 | Dtype::Uint8),
                "{k}: unsupported Breeze dtype"
            );
        }
        // Shared embedding is canonical in depth_decoder; do not retain stale duplicates.
        for suffix in ["weight", "scales", "biases"] {
            let key = format!("depth_decoder.model.embed_tokens.{suffix}");
            if let Some(v) = tensors.get(&key).cloned() {
                tensors.insert(
                    format!("backbone_model.embed_tokens.embed_audio_tokens.{suffix}"),
                    v,
                );
            }
        }
        let w = WeightMap::new(tensors, quant, HashMap::new());
        let text_embed = Embedding::from_loader(&w, "text_encoder.embed_tokens")?;
        let audio_embed =
            Embedding::from_loader(&w, "backbone_model.embed_tokens.embed_audio_tokens")?;
        let depth_embed = Embedding::from_loader(&w, "depth_decoder.model.embed_tokens")?;
        let depth_head = w.tensor("depth_decoder.codebooks_head.weight")?.clone();
        ensure!(
            depth_head.shape().as_slice() == [15, 1024, 2051],
            "invalid Breeze depth heads"
        );
        let eoi = w.tensor("text_encoder.embed_tokens.eoi_embedding")?.clone();
        ensure!(
            eoi.shape().as_slice() == [1152],
            "invalid Breeze end-of-input embedding"
        );
        let model = Self {
            text: Stack::load(&w, "text_encoder", &config.text_encoder_config, true)?,
            backbone: Stack::load(&w, "backbone_model", &config.backbone_config, false)?,
            depth: Stack::load(
                &w,
                "depth_decoder.model",
                &config.depth_decoder_config,
                false,
            )?,
            text_proj: Linear::from_loader(&w, "text_encoder_proj")?,
            head: Linear::from_loader(&w, "lm_head")?,
            depth_proj: Linear::from_loader(&w, "depth_decoder.model.inputs_embeds_projector")?,
            tokenizer: tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("Breeze tokenizer: {e}"))?,
            codec: codec::Codec::load(path)?,
            _weights: w,
            config,
            text_embed,
            eoi,
            audio_embed,
            depth_embed,
            depth_head,
            capabilities: TtsCapabilities {
                languages: vec![Language::Zh, Language::En],
                requires_reference: false,
                output_format: PcmFormat {
                    sample_rate: 24000,
                    channels: 1,
                },
                streaming: StreamingGranularity::IncrementalWaveform,
                text_position_capacity: 8192,
                mel_position_capacity: 1500,
            },
            max_output_frames,
        };
        ensure!(
            model.text_proj.in_features() == 1152
                && model.text_proj.out_features() == 2048
                && model.head.in_features() == 2048
                && model.head.out_features() == 2052
                && model.depth_proj.in_features() == 2048
                && model.depth_proj.out_features() == 1024,
            "invalid Breeze projection dimensions"
        );
        Ok(model)
    }
    fn encode_text(&self, text: &str, control: &dyn SessionControl) -> Result<Array> {
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Breeze tokenizer: {e}"))?;
        ensure!(ids.len() <= 8192, "Breeze text exceeds 8192 tokens");
        let ids = Array::try_from((ids.get_ids(), [1, ids.len() as i32]))?;
        let mut x = scale(
            &self.text_embed.forward(&ids)?,
            (self.config.text_encoder_config.hidden_size as f32).sqrt(),
        )?;
        let eoi = Array::try_from((
            &[self.config.text_encoder_config.eoi_token_index as u32][..],
            [],
        ))?;
        x = ops::where_(&ops::equal(&ids, &eoi)?.reshape([1, -1, 1])?, &self.eoi, &x)?;
        self.text_proj
            .forward(&self.text.forward(x, &mut self.text.cache(), control)?)
    }
    fn audio_embedding(&self, codes: &Array) -> Result<Array> {
        let offsets: Vec<i32> = (0..16).map(|i| i * 2051).collect();
        let ids = ops::add(codes, &Array::try_from((offsets.as_slice(), [1, 1, 16]))?)?;
        Ok(ops::sum(&self.audio_embed.forward(&ids)?, 2, false)?)
    }
    fn prompt(
        &self,
        text: &str,
        instruction: Option<&str>,
        reference: Option<&Array>,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let target = match instruction {
            Some(i) => format!("[S0]<ins_bos>{i}<ins_eos>{text}"),
            None => format!("[S0]{text}"),
        };
        let target = self.encode_text(&target, control)?;
        let prompt = if let Some(reference) = reference {
            ops::concatenate(&[reference, &target], 1)?
        } else {
            target
        };
        ensure!(
            prompt.shape().as_slice()[1] <= 8192,
            "Breeze prompt exceeds 8192 positions"
        );
        Ok(prompt)
    }
    fn depth_frame(
        &self,
        first: i32,
        hidden: &Array,
        uncond: Option<&Array>,
        options: &BreezeOptions,
        key: &mut Array,
        control: &dyn SessionControl,
    ) -> Result<Vec<i32>> {
        let mut cache = self.depth.cache();
        let mut uc = self.depth.cache();
        self.depth
            .forward(self.depth_proj.forward(hidden)?, &mut cache, control)?
            .eval()?;
        if let Some(u) = uncond {
            self.depth
                .forward(self.depth_proj.forward(u)?, &mut uc, control)?
                .eval()?;
        }
        let mut codes = vec![first];
        for i in 0..15 {
            control.check()?;
            let ids = Array::try_from((&[codes[i] + i as i32 * 2051][..], [1, 1]))?;
            let input = self.depth_proj.forward(&self.depth_embed.forward(&ids)?)?;
            let h = self.depth.forward(input.clone(), &mut cache, control)?;
            let head = self
                .depth_head
                .slice([i as i32, 0, 0], [i as i32 + 1, 1024, 2051])?
                .reshape([1024, 2051])?;
            let mut logits = ops::matmul(&h, &head)?;
            if uncond.is_some() {
                let u = ops::matmul(&self.depth.forward(input, &mut uc, control)?, &head)?;
                logits = guided(&logits, &u, options.cfg_scale)?;
            }
            codes.push(sample(&logits, false, options, key)?);
        }
        Ok(codes)
    }
}
fn guided(cond: &Array, uncond: &Array, scale_value: f32) -> Result<Array> {
    Ok(ops::add(
        uncond,
        &scale(&ops::subtract(cond, uncond)?, scale_value)?,
    )?)
}
fn sample(logits: &Array, eos: bool, options: &BreezeOptions, key: &mut Array) -> Result<i32> {
    let width = if eos { 2052 } else { 2051 };
    let logits = logits.reshape([width])?.astype(Dtype::Float32)?;
    let mask: Vec<f32> = (0..width)
        .map(|i| {
            if (2048..2051).contains(&i) {
                f32::NEG_INFINITY
            } else {
                0.
            }
        })
        .collect();
    let mut logits = ops::add(&logits, &Array::try_from((mask.as_slice(), [width]))?)?;
    ensure!(
        ops::all(&ops::isfinite(&logits.slice([0], [2048])?)?, 0, false)?.item::<bool>()?,
        "non-finite Breeze logits"
    );
    if options.temperature == 0. {
        return Ok(logits.argmax(-1, false)?.item::<u32>()? as i32);
    }
    logits = scale(&logits, options.temperature.recip())?;
    // Sorting in ascending order follows mlx-lm's top-p sampler.
    let indices = ops::argsort(&logits, -1)?;
    let sorted = ops::take(&logits, &indices, 0)?;
    let mut keep = ops::ones([width], Dtype::Bool)?;
    if options.top_k > 0 && options.top_k < width as usize {
        let range = ops::arange(0., width as f64, 1., Dtype::Int32)?;
        keep = ops::greater_equal(
            &range,
            &Array::try_from((&[width - options.top_k as i32][..], []))?,
        )?;
    }
    if options.top_p > 0. && options.top_p < 1. {
        let cumulative = ops::cumsum(&sorted.softmax(-1, true)?, -1, false, true)?;
        keep = ops::minimum(
            &keep,
            &ops::greater(&cumulative, &scalar(1. - options.top_p, Dtype::Float32)?)?,
        )?;
    }
    let sorted = ops::where_(&keep, &sorted, &scalar(f32::NEG_INFINITY, Dtype::Float32)?)?;
    let (next, draw) = mlx::random::split(key)?;
    *key = next;
    let sampled = mlx::random::categorical(&sorted).key(&draw).sample()?;
    Ok(ops::take(&indices, &sampled, 0)?.item::<u32>()? as i32)
}
struct Generation {
    cache: Cache,
    uncond_cache: Cache,
    hidden: Array,
    uncond: Option<Array>,
    frames: usize,
    pending: Vec<i32>,
}
struct Session<'a> {
    model: &'a mut BreezeTts,
    control: &'a dyn SessionControl,
    options: BreezeOptions,
    segments: Vec<String>,
    segment: usize,
    reference: Option<Array>,
    reference_frames: u64,
    reference_rate: u32,
    key: Array,
    generation: Option<Generation>,
    output: VecDeque<PcmChunk>,
    collected: Vec<f32>,
    collect: bool,
    total: u64,
    finished: bool,
    language: Language,
}
impl TtsModel for BreezeTts {
    fn capabilities(&self) -> &TtsCapabilities {
        &self.capabilities
    }
    fn start<'a>(
        &'a mut self,
        request: TtsRequest,
        control: &'a dyn SessionControl,
    ) -> crate::Result<Box<dyn TtsSession + 'a>> {
        let prepare = || -> Result<_> {
            control.check()?;
            let options = request.breeze.unwrap_or_default();
            options.validate()?;
            ensure!(
                !request.text.trim().is_empty() && request.text.len() <= 65536,
                "Breeze text must be nonempty and at most 65536 UTF-8 bytes"
            );
            ensure!(
                matches!(
                    request.language,
                    Language::Auto | Language::Zh | Language::En
                ),
                "Breeze supports Chinese and English"
            );
            let source_frames = if request.reference.samples.is_empty() {
                0
            } else {
                request.reference.frames()? as u64
            };
            let source_rate = request.reference.format.sample_rate;
            let reference = if source_frames > 0 {
                let transcript = options
                    .ref_text
                    .as_ref()
                    .context("Breeze voice cloning requires ref_text")?;
                ensure!(
                    source_frames <= u64::from(source_rate) * 60,
                    "Breeze reference exceeds 60 seconds"
                );
                let text = self.encode_text(&format!("[S0]{transcript}"), control)?;
                let codes = self.codec.encode(&request.reference, control)?;
                let audio = self.audio_embedding(&codes)?;
                let eos = self.audio_embedding(&Array::zeros([1, 1, 16], Dtype::Int32)?)?;
                Some(ops::concatenate(&[&text, &audio, &eos], 1)?)
            } else {
                ensure!(
                    options.ref_text.is_none(),
                    "ref_text requires reference audio"
                );
                ensure!(
                    options.instructions.is_some(),
                    "Breeze voice design requires instructions"
                );
                None
            };
            let language = if request
                .text
                .chars()
                .any(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
            {
                Language::Zh
            } else {
                Language::En
            };
            let segments = split_text(&request.text);
            ensure!(segments.len() <= 256, "Breeze text exceeds 256 segments");
            Ok((
                options,
                reference,
                source_frames,
                source_rate,
                language,
                segments,
                request.seed,
                matches!(request.output_policy, OutputPolicy::Collect),
            ))
        };
        let (
            options,
            reference,
            reference_frames,
            reference_rate,
            language,
            segments,
            seed,
            collect,
        ) = prepare().map_err(|e| match e.downcast::<AudioError>() {
            Ok(e) => e,
            Err(e) => crate::error::invalid("breeze", format!("{e:#}")),
        })?;
        self.codec.reset_stream();
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        });
        Ok(Box::new(Session {
            model: self,
            control,
            options,
            segments,
            segment: 0,
            reference,
            reference_frames,
            reference_rate,
            key: mlx::random::key(seed)?,
            generation: None,
            output: VecDeque::new(),
            collected: Vec::new(),
            collect,
            total: 0,
            finished: false,
            language,
        }))
    }
}
fn split_text(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut segment = String::new();
    let mut n = 0;
    for c in text.chars() {
        segment.push(c);
        n += 1;
        if c == '\n' || n >= 600 {
            if !segment.trim().is_empty() {
                out.push(segment.trim().to_owned());
            }
            segment.clear();
            n = 0;
        }
    }
    if !segment.trim().is_empty() {
        out.push(segment.trim().to_owned());
    }
    out
}
impl Session<'_> {
    fn emit(&mut self, samples: Vec<f32>) -> Result<()> {
        ensure!(
            samples.iter().all(|x| x.is_finite()),
            "non-finite Breeze waveform"
        );
        if self.total + samples.len() as u64 > self.model.max_output_frames {
            return Err(AudioError::GenerationLimitExceeded.into());
        }
        if self.collect {
            self.collected.extend_from_slice(&samples);
            self.total += samples.len() as u64;
        } else {
            for s in samples.chunks(8192) {
                self.output.push_back(PcmChunk {
                    start_frame: self.total,
                    pcm: PcmBuffer {
                        format: self.model.capabilities.output_format,
                        samples: s.to_vec(),
                    },
                });
                self.total += s.len() as u64;
            }
        }
        Ok(())
    }
    fn step(&mut self) -> Result<TtsStep> {
        self.control.check()?;
        if let Some(c) = self.output.pop_front() {
            return Ok(TtsStep::Audio(c));
        }
        if self.segment == self.segments.len() {
            if self.collect && !self.collected.is_empty() {
                let samples = std::mem::take(&mut self.collected);
                let mut start = 0;
                for s in samples.chunks(8192) {
                    self.output.push_back(PcmChunk {
                        start_frame: start,
                        pcm: PcmBuffer {
                            format: self.model.capabilities.output_format,
                            samples: s.to_vec(),
                        },
                    });
                    start += s.len() as u64;
                }
                return self.step();
            }
            self.finished = true;
            return Ok(TtsStep::Finished(TtsSummary {
                total_frames: self.total,
                segments: self.segments.len(),
                resolved_language: self.language,
                language_ambiguous: false,
                reference_source_frames: self.reference_frames,
                reference_used_frames: self.reference_frames,
                reference_source_sample_rate: self.reference_rate,
            }));
        }
        if self.generation.is_none() {
            self.model.codec.reset_stream();
            let prompt = self.model.prompt(
                &self.segments[self.segment],
                self.options.instructions.as_deref(),
                self.reference.as_ref(),
                self.control,
            )?;
            let mut cache = self.model.backbone.cache();
            let hidden = last(
                self.model
                    .backbone
                    .forward(prompt, &mut cache, self.control)?,
            )?;
            let mut uncond_cache = self.model.backbone.cache();
            let uncond = if self.options.instructions.is_some() && self.options.cfg_scale != 1. {
                let prompt = self.model.prompt(
                    &self.segments[self.segment],
                    None,
                    self.reference.as_ref(),
                    self.control,
                )?;
                Some(last(self.model.backbone.forward(
                    prompt,
                    &mut uncond_cache,
                    self.control,
                )?)?)
            } else {
                None
            };
            self.generation = Some(Generation {
                cache,
                uncond_cache,
                hidden,
                uncond,
                frames: 0,
                pending: Vec::new(),
            });
            return Ok(TtsStep::Progress);
        }
        let g = self.generation.as_mut().unwrap();
        let mut logits = self.model.head.forward(&g.hidden)?;
        if let Some(u) = &g.uncond {
            logits = guided(
                &logits,
                &self.model.head.forward(u)?,
                self.options.cfg_scale,
            )?;
        }
        let first = sample(&logits, true, &self.options, &mut self.key)?;
        if first == 2051 {
            let g = self.generation.take().unwrap();
            ensure!(g.frames > 0, "Breeze returned EOS before any audio frame");
            if !g.pending.is_empty() {
                let codes = Array::try_from((
                    g.pending.as_slice(),
                    [1, (g.pending.len() / 16) as i32, 16],
                ))?;
                let audio = self
                    .model
                    .codec
                    .decode_step(&codes, self.control)?
                    .astype(Dtype::Float32)?
                    .to_vec::<f32>()?;
                self.emit(audio)?;
            }
            self.model.codec.reset_stream();
            self.segment += 1;
            return Ok(TtsStep::Progress);
        }
        if g.frames >= self.options.max_tokens {
            return Err(AudioError::GenerationLimitExceeded.into());
        }
        let frame = self.model.depth_frame(
            first,
            &g.hidden,
            g.uncond.as_ref(),
            &self.options,
            &mut self.key,
            self.control,
        )?;
        g.frames += 1;
        g.pending.extend_from_slice(&frame);
        let codes = Array::try_from((frame.as_slice(), [1, 1, 16]))?;
        let embeds = self.model.audio_embedding(&codes)?;
        g.hidden = last(self.model.backbone.forward(
            embeds.clone(),
            &mut g.cache,
            self.control,
        )?)?;
        if g.uncond.is_some() {
            g.uncond = Some(last(self.model.backbone.forward(
                embeds,
                &mut g.uncond_cache,
                self.control,
            )?)?);
        }
        if g.pending.len() >= 16 * 4 {
            let codes =
                Array::try_from((g.pending.as_slice(), [1, (g.pending.len() / 16) as i32, 16]))?;
            g.pending.clear();
            let audio = self
                .model
                .codec
                .decode_step(&codes, self.control)?
                .astype(Dtype::Float32)?
                .to_vec::<f32>()?;
            self.emit(audio)?;
        }
        Ok(TtsStep::Progress)
    }
}
impl TtsSession for Session<'_> {
    fn advance(&mut self) -> crate::Result<TtsStep> {
        if self.finished {
            return Err(AudioError::InvalidSessionState);
        }
        match self.step() {
            Ok(s) => Ok(s),
            Err(e) => {
                self.finished = true;
                self.model.codec.reset_stream();
                Err(model_error(e))
            }
        }
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        self.model.codec.reset_stream();
    }
}

#[cfg(test)]
mod tests;
