use anyhow::Context;
use clap::Args;
use ironmlx_audio::{qwen3_asr::Qwen3Asr, AudioIo, DecodeLimits, SessionControl};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct TranscribeArgs {
    /// Local immutable Qwen3-ASR MLX snapshot directory.
    #[arg(long)]
    model: PathBuf,
    /// WAV, FLAC, or MP3 recording.
    #[arg(long)]
    audio: PathBuf,
    /// Input language code or name; omit for automatic recognition.
    #[arg(long)]
    language: Option<String>,
    /// Vocabulary/topic context for recognition.
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,
}
struct Control;
impl SessionControl for Control {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}

pub fn run(args: TranscribeArgs) -> crate::Result<()> {
    let bytes = std::fs::read(&args.audio).context("reading ASR input audio")?;
    let limits = DecodeLimits {
        max_duration_seconds: 7200,
        max_encoded_bytes: 128 * 1024 * 1024,
        max_pcm_bytes: 512 * 1024 * 1024,
        ..DecodeLimits::default()
    };
    let pcm = ironmlx_audio::io::NativeAudioIo.decode(&bytes, &limits)?;
    let mut model = Qwen3Asr::load(&args.model)?;
    let result = model.transcribe(
        &pcm,
        args.language.as_deref(),
        args.prompt.as_deref(),
        args.max_tokens,
        &Control,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
