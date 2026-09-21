//! Owned audio values and worker-local speech model contracts.
use crate::Result;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

/// Boundary validation is mandatory; interleaved finite f32 samples.
#[derive(Clone, Debug)]
pub struct PcmBuffer {
    pub format: PcmFormat,
    pub samples: Vec<f32>,
}

pub struct PcmChunk {
    pub start_frame: u64,
    pub pcm: PcmBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Auto,
    Zh,
    En,
    Ja,
    Es,
    Ar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingGranularity {
    None,
    Segment,
    IncrementalWaveform,
}

pub struct TtsCapabilities {
    pub languages: Vec<Language>,
    pub requires_reference: bool,
    pub output_format: PcmFormat,
    pub streaming: StreamingGranularity,
    pub text_position_capacity: usize,
    pub mel_position_capacity: usize,
}

pub enum OutputPolicy {
    Collect,
    Chunks,
}

/// Model configuration selects the fixed v1 synthesis profile.
/// No wire model identifier, Base64, WAV, or HTTP streaming flag.
pub struct TtsRequest {
    pub text: String,
    pub reference: PcmBuffer,
    pub language: Language,
    pub seed: Option<u64>,
    pub output_policy: OutputPolicy,
}

pub struct TtsSummary {
    pub total_frames: u64,
    pub segments: usize,
    pub resolved_language: Language,
    pub language_ambiguous: bool,
    pub reference_source_frames: u64,
    pub reference_used_frames: u64,
    pub reference_source_sample_rate: u32,
}

pub enum TtsStep {
    /// Computation advanced to a cancellation checkpoint; not an empty poll.
    Progress,
    Audio(PcmChunk),
    Finished(TtsSummary),
}

/// Supplied by runtime; deadline/cancellation policies do not enter the model.
pub trait SessionControl {
    fn check(&self) -> Result<()>;
}

pub trait TtsSession {
    /// Finished or Err is terminal; further calls return InvalidSessionState.
    fn advance(&mut self) -> Result<TtsStep>;
}

pub trait TtsModel {
    fn capabilities(&self) -> &TtsCapabilities;

    /// Session borrows the exclusively held model and stays on its worker.
    fn start<'a>(
        &'a mut self,
        request: TtsRequest,
        control: &'a dyn SessionControl,
    ) -> Result<Box<dyn TtsSession + 'a>>;
}

pub struct ResolvedModelResources {
    pub source_snapshot: PathBuf,
    pub derived_resources: PathBuf,
    pub resource_lock: PathBuf,
}

pub struct ResourceIssue {
    pub component: String,
    pub reason: String,
}

pub struct ResourceReport {
    pub complete: bool,
    pub issues: Vec<ResourceIssue>,
    pub static_tensor_bytes: u64,
}

/// Concrete IndexTts25 loader implements this contract; loading runs on worker.
pub trait TtsLoader {
    fn inspect(&self, resources: &ResolvedModelResources) -> Result<ResourceReport>;
    fn load(&self, resources: &ResolvedModelResources) -> Result<Box<dyn TtsModel>>;
}

pub struct DecodeLimits {
    pub max_encoded_bytes: usize,
    pub max_pcm_bytes: usize,
    pub max_duration_seconds: u32,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_channels: u16,
}

/// IO is independent from model inference and HTTP.
pub trait AudioIo {
    fn decode(&self, bytes: &[u8], limits: &DecodeLimits) -> Result<PcmBuffer>;
    fn encode_wav(&self, pcm: &PcmBuffer) -> Result<Vec<u8>>;
}
