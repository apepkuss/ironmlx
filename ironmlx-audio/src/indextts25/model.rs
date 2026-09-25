//! Worker-local model loading and a pull-driven, bounded synthesis state machine.
use super::{
    cfm::{Cfm, CfmState},
    codec::Codec,
    conditioning::length_regulate,
    gpt::{self, Gpt, GptState},
    layers::Weights,
    vocoder::Vocoder,
    IndexTts25ReferenceEncoder, ReferenceConditioning,
};
use crate::{
    error::invalid,
    resources::{self, IndexTts25Component},
    text::{IndexTts25TextFrontend, PreparedText, TextLimits},
    AudioError, Language, OutputPolicy, PcmBuffer, PcmChunk, PcmFormat, ResolvedModelResources,
    ResourceIssue, ResourceReport, Result, SessionControl, StreamingGranularity, TtsCapabilities,
    TtsLoader, TtsModel, TtsRequest, TtsSession, TtsStep, TtsSummary,
};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array};
use std::{
    collections::HashMap,
    io::Read,
    path::{Path, PathBuf},
};

pub const INDEXTTS25_OUTPUT_SAMPLE_RATE_HZ: u32 = 22_050;
pub const INDEXTTS25_MAX_OUTPUT_FRAMES: u64 = 600 * INDEXTTS25_OUTPUT_SAMPLE_RATE_HZ as u64;

const FORMAT: PcmFormat = PcmFormat {
    sample_rate: INDEXTTS25_OUTPUT_SAMPLE_RATE_HZ,
    channels: 1,
};
const BLOCK_FRAMES: usize = 8192;
const GAP_FRAMES: usize = 4410;

/// Explicit local text resources; no host-environment discovery or downloads.
pub struct IndexTts25Loader {
    wetext_fsts: PathBuf,
    unidic_dir: PathBuf,
    text_limits: TextLimits,
    max_output_frames: u64,
}
impl IndexTts25Loader {
    pub fn new(wetext_fsts: PathBuf, unidic_dir: PathBuf) -> Self {
        Self {
            wetext_fsts,
            unidic_dir,
            text_limits: TextLimits::default(),
            max_output_frames: INDEXTTS25_MAX_OUTPUT_FRAMES,
        }
    }
    /// Service policies; model positions and per-segment generation limits remain fixed.
    pub fn with_limits(mut self, text: TextLimits, max_output_frames: u64) -> Result<Self> {
        if max_output_frames == 0
            || max_output_frames > usize::MAX as u64 / 4
            || text.segment_tokens == 0
            || !(4..=602).contains(&text.position_capacity)
            || text.max_segments == 0
            || text.max_total_tokens == 0
        {
            return Err(invalid(
                "synthesis limits",
                "invalid text or output capacity",
            ));
        }
        self.text_limits = text;
        self.max_output_frames = max_output_frames;
        Ok(self)
    }
    fn check_lock(path: &Path) -> Result<()> {
        if std::fs::metadata(path)?.len() > 2 * 1024 * 1024 {
            return Err(invalid("resource lock", "file too large"));
        }
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../../resources/indextts25/sources.json"))
                .expect("embedded resource lock");
        let actual: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| invalid("resource lock", e.to_string()))?;
        if actual != expected {
            return Err(AudioError::ResourceMismatch {
                component: "resource lock".into(),
                reason: "does not match the embedded fixed profile".into(),
            });
        }
        Ok(())
    }
    fn check_text(&self, snapshot: &Path) -> Result<()> {
        crate::text::IndexTts25Tokenizer::from_file(
            &snapshot.join("multilingual_zh_ja_yue_char_del.tiktoken"),
        )?;
        for language in ["zh", "en"] {
            for kind in ["tagger", "verbalizer"] {
                let member = format!("{language}/tn/{kind}.fst");
                crate::text::verify_resource(
                    &self.wetext_fsts.join(&member),
                    "wetext",
                    &format!("wetext/fsts/{member}"),
                )?;
            }
        }
        for name in ["sys.dic", "unk.dic", "matrix.bin", "char.bin", "dicrc"] {
            crate::text::verify_resource(
                &self.unidic_dir.join(name),
                "unidic-lite",
                &format!("unidic-lite-1.0.8/unidic_lite/dicdir/{name}"),
            )?;
        }
        Ok(())
    }
    /// Construct the concrete worker-local model after strict resource verification.
    pub fn load_model(&self, resources: &ResolvedModelResources) -> Result<IndexTts25> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
            return Err(invalid(
                "MLX_ENABLE_TF32",
                "set to 0 before process MLX initialization",
            ));
        }
        Self::check_lock(&resources.resource_lock)?;
        IndexTts25ReferenceEncoder::check_config(&resources.source_snapshot)?;
        let frontend = IndexTts25TextFrontend::load(
            &resources
                .source_snapshot
                .join("multilingual_zh_ja_yue_char_del.tiktoken"),
            &self.wetext_fsts,
            &self.unidic_dir,
        )?;
        let load = |component: IndexTts25Component| {
            resources::load_component(
                &resources.source_snapshot.join(component.file_name()),
                &component.spec()?,
            )
        };
        let gpt = load(IndexTts25Component::Gpt)?;
        let s2mel = load(IndexTts25Component::S2Mel)?;
        let copy = |w: &WeightMap| {
            WeightMap::new(
                w.tensors()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
                None,
                HashMap::new(),
            )
        };
        let reference = IndexTts25ReferenceEncoder::from_weights(
            &resources.source_snapshot,
            &resources.derived_resources,
            copy(&gpt),
            copy(&s2mel),
        )?;
        let mut length = copy(&s2mel);
        length.retain(|name, _| name.starts_with("length_regulator."));
        Ok(IndexTts25 {
            reference,
            frontend,
            gpt: Gpt::new(gpt),
            cfm: Cfm::new(s2mel),
            length: Weights(length),
            codec: Codec::new(load(IndexTts25Component::Codec)?)?,
            vocoder: Vocoder::new(load(IndexTts25Component::BigVgan)?)?,
            text_limits: self.text_limits.clone(),
            max_output_frames: self.max_output_frames,
            capabilities: TtsCapabilities {
                languages: vec![
                    Language::Zh,
                    Language::En,
                    Language::Ja,
                    Language::Es,
                    Language::Ar,
                ],
                requires_reference: true,
                output_format: FORMAT,
                streaming: StreamingGranularity::Segment,
                text_position_capacity: 602,
                mel_position_capacity: 1818,
            },
        })
    }
}
impl TtsLoader for IndexTts25Loader {
    fn inspect(&self, resources: &ResolvedModelResources) -> Result<ResourceReport> {
        let mut report = IndexTts25ReferenceEncoder::inspect(
            &resources.source_snapshot,
            &resources.derived_resources,
        );
        for component in [IndexTts25Component::Codec, IndexTts25Component::BigVgan] {
            match resources::inspect_component(
                &resources.source_snapshot.join(component.file_name()),
                &component.spec()?,
            ) {
                Ok(bytes) => report.static_tensor_bytes += bytes,
                Err(e) => report.issues.push(ResourceIssue {
                    component: component.file_name().into(),
                    reason: e.to_string(),
                }),
            }
        }
        for (name, result) in [
            ("resource lock", Self::check_lock(&resources.resource_lock)),
            (
                "text resources",
                self.check_text(&resources.source_snapshot),
            ),
        ] {
            if let Err(e) = result {
                report.issues.push(ResourceIssue {
                    component: name.into(),
                    reason: e.to_string(),
                });
            }
        }
        report.complete = report.issues.is_empty();
        Ok(report)
    }
    fn load(&self, resources: &ResolvedModelResources) -> Result<Box<dyn TtsModel>> {
        Ok(Box::new(self.load_model(resources)?))
    }
}

/// Native IndexTTS 2.5 synthesis, owned exclusively by one execution worker.
pub struct IndexTts25 {
    reference: IndexTts25ReferenceEncoder,
    frontend: IndexTts25TextFrontend,
    gpt: Gpt,
    codec: Codec,
    length: Weights,
    cfm: Cfm,
    vocoder: Vocoder,
    capabilities: TtsCapabilities,
    text_limits: TextLimits,
    max_output_frames: u64,
}
impl TtsModel for IndexTts25 {
    fn capabilities(&self) -> &TtsCapabilities {
        &self.capabilities
    }
    fn start<'a>(
        &'a mut self,
        request: TtsRequest,
        control: &'a dyn SessionControl,
    ) -> Result<Box<dyn TtsSession + 'a>> {
        control.check()?;
        if request.text.len() > self.text_limits.max_input_bytes {
            return Err(AudioError::CapacityExceeded {
                resource: "text input bytes",
            });
        }
        let collect = matches!(request.output_policy, OutputPolicy::Collect);
        let seed = match request.seed {
            Some(seed) => seed,
            None => {
                let mut bytes = [0; 8];
                std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
                u64::from_le_bytes(bytes)
            }
        };
        Ok(Box::new(Session {
            model: self,
            control,
            phase: Phase::Prepare(request),
            reference: None,
            text: None,
            key: mlx::random::key(seed)?,
            segment: 0,
            total_frames: 0,
            emitted_frames: 0,
            collect,
            collected: Vec::new(),
        }))
    }
}

enum Phase {
    Prepare(TtsRequest),
    StartSegment,
    Generate(GptState),
    Decode(Vec<u32>),
    Length(Array),
    Flow(CfmState),
    Vocoder(Array),
    Emit {
        samples: Vec<f32>,
        cursor: usize,
        final_output: bool,
    },
    Finish,
    Done,
}
struct Session<'a> {
    model: &'a mut IndexTts25,
    control: &'a dyn SessionControl,
    phase: Phase,
    reference: Option<ReferenceConditioning>,
    text: Option<PreparedText>,
    key: Array,
    segment: usize,
    total_frames: u64,
    emitted_frames: u64,
    collect: bool,
    collected: Vec<f32>,
}
impl Session<'_> {
    fn next_key(&mut self) -> Result<Array> {
        let (key, sample) = mlx::random::split(&self.key)?;
        self.key = key;
        Ok(sample)
    }
    fn reference(&self) -> Result<&ReferenceConditioning> {
        self.reference
            .as_ref()
            .ok_or(AudioError::InvalidSessionState)
    }
    fn text(&self) -> Result<&PreparedText> {
        self.text.as_ref().ok_or(AudioError::InvalidSessionState)
    }
    fn advance_inner(&mut self) -> Result<TtsStep> {
        self.control.check()?;
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        match phase {
            Phase::Prepare(request) => {
                self.text = Some(self.model.frontend.prepare(
                    &request.text,
                    request.language,
                    &self.model.text_limits,
                    self.control,
                )?);
                self.reference = Some(
                    self.model
                        .reference
                        .encode(&request.reference, self.control)?,
                );
                self.phase = Phase::StartSegment;
            }
            Phase::StartSegment => {
                let state = self.model.gpt.start(
                    &self.reference()?.gpt,
                    &self.text()?.token_ids[self.segment],
                    self.text()?.language_id,
                )?;
                self.phase = Phase::Generate(state);
            }
            Phase::Generate(mut state) => {
                let logits = self.model.gpt.logits(&mut state, self.control)?;
                let token = gpt::sample(&logits, &state.generated, &self.next_key()?)?;
                self.model.gpt.accept(&mut state, token)?;
                if state.finished {
                    if state.generated.is_empty() {
                        return Err(AudioError::InferenceFailed {
                            reason: "GPT ended without audio codes".into(),
                        });
                    }
                    gpt::compress_silence(&mut state.generated);
                    self.phase = Phase::Decode(state.generated);
                } else {
                    self.phase = Phase::Generate(state);
                }
            }
            Phase::Decode(codes) => {
                self.phase = Phase::Length(self.model.codec.decode(&codes, self.control)?);
            }
            Phase::Length(semantic) => {
                let frames = (f64::from(semantic.shape()[1]) * 1.72) as usize;
                // Reserve the known output size before CFM/vocoder allocation.
                let gap = if self.segment + 1 < self.text()?.segments.len() {
                    GAP_FRAMES
                } else {
                    0
                };
                check_output_capacity(
                    self.total_frames,
                    frames * 256 + gap,
                    self.model.max_output_frames,
                )?;
                let condition =
                    length_regulate(&self.model.length, &semantic, frames, self.control)?;
                let combined = ops::concatenate(&[&self.reference()?.prompt, &condition], 1)?;
                let noise = mlx::random::normal()
                    .shape([1, 80, combined.shape()[1]])
                    .key(&self.next_key()?)
                    .sample()?;
                self.phase = Phase::Flow(Cfm::start(
                    &combined,
                    &self.reference()?.mel,
                    &self.reference()?.style,
                    noise,
                )?);
            }
            Phase::Flow(mut state) => {
                self.model.cfm.advance(&mut state, self.control)?;
                if state.finished() {
                    let frames = state.x.shape()[2];
                    self.phase = Phase::Vocoder(ops::slice(
                        &state.x,
                        [0, 0, self.reference()?.mel_frames() as i32],
                        [1, 80, frames],
                    )?);
                } else {
                    self.phase = Phase::Flow(state);
                }
            }
            Phase::Vocoder(mel) => {
                let mut samples = self
                    .model
                    .vocoder
                    .synthesize(&mel, self.control)?
                    .to_vec::<f32>()?;
                postprocess(&mut samples)?;
                let last = self.segment + 1 == self.text()?.segments.len();
                if !last {
                    samples.try_reserve_exact(GAP_FRAMES).map_err(|_| {
                        AudioError::CapacityExceeded {
                            resource: "segment audio buffer",
                        }
                    })?;
                    samples.resize(samples.len() + GAP_FRAMES, 0.);
                }
                self.total_frames = check_output_capacity(
                    self.total_frames,
                    samples.len(),
                    self.model.max_output_frames,
                )?;
                self.segment += 1;
                if self.collect {
                    self.collected
                        .try_reserve_exact(samples.len())
                        .map_err(|_| AudioError::CapacityExceeded {
                            resource: "collected audio buffer",
                        })?;
                    self.collected.extend(samples);
                    self.phase = if last {
                        Phase::Emit {
                            samples: std::mem::take(&mut self.collected),
                            cursor: 0,
                            final_output: true,
                        }
                    } else {
                        Phase::StartSegment
                    };
                } else {
                    self.phase = Phase::Emit {
                        samples,
                        cursor: 0,
                        final_output: last,
                    };
                }
            }
            Phase::Emit {
                samples,
                cursor,
                final_output,
            } => {
                let end = (cursor + BLOCK_FRAMES).min(samples.len());
                if end == cursor {
                    return Err(AudioError::InferenceFailed {
                        reason: "empty output buffer".into(),
                    });
                }
                let chunk = PcmChunk {
                    start_frame: self.emitted_frames,
                    pcm: PcmBuffer {
                        format: FORMAT,
                        samples: samples[cursor..end].to_vec(),
                    },
                };
                self.emitted_frames += (end - cursor) as u64;
                self.phase = if end < samples.len() {
                    Phase::Emit {
                        samples,
                        cursor: end,
                        final_output,
                    }
                } else if final_output {
                    Phase::Finish
                } else {
                    Phase::StartSegment
                };
                return Ok(TtsStep::Audio(chunk));
            }
            Phase::Finish => {
                let reference = self.reference()?;
                let text = self.text()?;
                let summary = TtsSummary {
                    total_frames: self.total_frames,
                    segments: self.segment,
                    resolved_language: text.language,
                    language_ambiguous: text.language_ambiguous,
                    reference_source_frames: reference.source_frames() as u64,
                    reference_used_frames: reference.used_frames() as u64,
                    reference_source_sample_rate: reference.source_sample_rate(),
                };
                self.reference = None;
                self.text = None;
                return Ok(TtsStep::Finished(summary));
            }
            Phase::Done => return Err(AudioError::InvalidSessionState),
        }
        Ok(TtsStep::Progress)
    }
}
impl TtsSession for Session<'_> {
    fn advance(&mut self) -> Result<TtsStep> {
        if matches!(self.phase, Phase::Done) {
            return Err(AudioError::InvalidSessionState);
        }
        let result = self.advance_inner().and_then(|step| {
            self.control.check()?;
            Ok(step)
        });
        if result.is_err() {
            self.phase = Phase::Done;
            self.reference = None;
            self.text = None;
            self.collected = Vec::new();
        }
        result
    }
}
fn check_output_capacity(current: u64, added: usize, maximum: u64) -> Result<u64> {
    current
        .checked_add(added as u64)
        .filter(|&frames| frames <= maximum)
        .ok_or(AudioError::GenerationLimitExceeded)
}
fn postprocess(samples: &mut [f32]) -> Result<()> {
    if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
        return Err(AudioError::InferenceFailed {
            reason: "empty or non-finite waveform".into(),
        });
    }
    let peak = samples.iter().copied().map(f32::abs).fold(0., f32::max);
    for sample in samples {
        if peak > 1. {
            *sample /= peak;
        }
        *sample = sample.clamp(-0.99, 0.99);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned model/text/derived and IRONMLX_REFERENCE_ACOUSTIC/FEATURES; MLX_ENABLE_TF32=0"]
    fn real_session_error_boundaries() {
        use std::{cell::Cell, time::Instant};
        struct Budget {
            remaining: Cell<usize>,
            deadline: Cell<bool>,
        }
        impl SessionControl for Budget {
            fn check(&self) -> Result<()> {
                if self.deadline.get() {
                    return Err(AudioError::DeadlineExceeded);
                }
                let remaining = self.remaining.get();
                if remaining == 0 {
                    Err(AudioError::Cancelled)
                } else {
                    self.remaining.set(remaining - 1);
                    Ok(())
                }
            }
        }
        let path = |name: &str| PathBuf::from(std::env::var(name).unwrap());
        let loader = IndexTts25Loader::new(path("IRONMLX_WETEXT_FSTS"), path("IRONMLX_UNIDIC_DIR"));
        let resources = ResolvedModelResources {
            source_snapshot: path("IRONMLX_INDEXTTS25_SNAPSHOT"),
            derived_resources: path("IRONMLX_INDEXTTS25_DERIVED"),
            resource_lock: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("resources/indextts25/sources.json"),
        };
        let mut model = loader.load_model(&resources).unwrap();
        let (r, _) =
            mlx::io::load_safetensors(path("IRONMLX_REFERENCE_ACOUSTIC").to_str().unwrap())
                .unwrap();
        let (g, _) =
            mlx::io::load_safetensors(path("IRONMLX_REFERENCE_GPT").to_str().unwrap()).unwrap();
        let text = g["zh.text"]
            .astype(mlx::Dtype::Uint32)
            .unwrap()
            .to_vec::<u32>()
            .unwrap();
        let control = Budget {
            remaining: Cell::new(usize::MAX),
            deadline: Cell::new(false),
        };
        // Cancel inside the layer loops, after work has begun, and verify terminal cleanup.
        for stage in ["gpt", "codec", "cfm", "vocoder", "cap", "deadline"] {
            let phase = match stage {
                "gpt" | "cap" => {
                    let mut state = model.gpt.start(&g["conditioning"], &text, 1).unwrap();
                    if stage == "cap" {
                        state.generated = vec![52; 1500];
                    }
                    Phase::Generate(state)
                }
                "codec" => Phase::Decode(r["codes"].to_vec::<u32>().unwrap()),
                "cfm" => Phase::Flow(
                    Cfm::start(
                        &r["combined"],
                        &r["ref_mel"],
                        &r["style"],
                        r["noise"].clone(),
                    )
                    .unwrap(),
                ),
                _ => Phase::Vocoder(r["generated_mel"].clone()),
            };
            control
                .remaining
                .set(if stage == "cap" { usize::MAX } else { 5 });
            control.deadline.set(stage == "deadline");
            let mut session = Session {
                model: &mut model,
                control: &control,
                phase,
                reference: None,
                text: None,
                key: mlx::random::key(2025).unwrap(),
                segment: 0,
                total_frames: 0,
                emitted_frames: 0,
                collect: true,
                collected: vec![0.; 32],
            };
            let started = Instant::now();
            let error = session.advance().err().unwrap();
            eprintln!("{stage} stop latency: {:?}", started.elapsed());
            match stage {
                "cap" => assert!(matches!(error, AudioError::GenerationLimitExceeded)),
                "deadline" => assert!(matches!(error, AudioError::DeadlineExceeded)),
                _ => assert!(matches!(error, AudioError::Cancelled)),
            }
            assert!(matches!(
                session.advance(),
                Err(AudioError::InvalidSessionState)
            ));
            assert!(session.collected.is_empty());
        }
        control.deadline.set(false);
        control.remaining.set(usize::MAX);
        model.max_output_frames = 1;
        let (waves, _) =
            mlx::io::load_safetensors(path("IRONMLX_REFERENCE_FEATURES").to_str().unwrap())
                .unwrap();
        let request = TtsRequest {
            text: "你好。".into(),
            reference: PcmBuffer {
                format: PcmFormat {
                    sample_rate: 16000,
                    channels: 1,
                },
                samples: waves["speech_1s.wave16"].to_vec().unwrap(),
            },
            language: Language::Zh,
            seed: Some(2025),
            output_policy: OutputPolicy::Chunks,
        };
        let mut session = model.start(request, &control).unwrap();
        loop {
            match session.advance() {
                Ok(TtsStep::Progress) => {}
                Err(AudioError::GenerationLimitExceeded) => break,
                _ => panic!("output capacity must fail before audio or successful terminal"),
            }
        }
        assert!(matches!(
            session.advance(),
            Err(AudioError::InvalidSessionState)
        ));
    }
    #[test]
    fn output_capacity_includes_silence_and_overflow() {
        assert_eq!(
            check_output_capacity(10, GAP_FRAMES, 10 + GAP_FRAMES as u64).unwrap(),
            4420
        );
        assert!(matches!(
            check_output_capacity(10, GAP_FRAMES, 4419),
            Err(AudioError::GenerationLimitExceeded)
        ));
        assert!(matches!(
            check_output_capacity(u64::MAX, 1, u64::MAX),
            Err(AudioError::GenerationLimitExceeded)
        ));
    }
    #[test]
    fn waveform_processing_rejects_empty_nonfinite_and_normalizes_once() {
        for mut values in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(matches!(
                postprocess(&mut values),
                Err(AudioError::InferenceFailed { .. })
            ));
        }
        let mut values = [-2., -1., 0., 1., 2.];
        postprocess(&mut values).unwrap();
        assert_eq!(values, [-0.99, -0.5, 0., 0.5, 0.99]);
        let mut quiet = [0., 0.1, -0.2];
        postprocess(&mut quiet).unwrap();
        assert_eq!(quiet, [0., 0.1, -0.2]);
    }
    #[test]
    fn resource_lock_cannot_override_embedded_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sources.json");
        std::fs::write(
            &path,
            include_bytes!("../../resources/indextts25/sources.json"),
        )
        .unwrap();
        IndexTts25Loader::check_lock(&path).unwrap();
        std::fs::write(&path, "{}").unwrap();
        assert!(matches!(
            IndexTts25Loader::check_lock(&path),
            Err(AudioError::ResourceMismatch { .. })
        ));
    }
}
