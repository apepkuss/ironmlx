//! Native Qwen3-ASR audio recognition. Checkpoints are immutable local MLX snapshots.
mod alignment;
mod config;
mod features;
mod model;
mod segments;
mod tokenizer;

pub use config::{preflight_model_metadata, Config, Metadata};
pub use model::{Qwen3Asr, TranscriptSegment, Transcription};

pub use segments::{append_text as append_transcript, prepare_audio, PreparedAudio};

pub use alignment::{AlignedWord, Alignment};
