//! Bounded, file-free audio input shared by embedding transports and the adapter.
use ironmlx_audio::{AudioError, AudioIo, DecodeLimits, SessionControl};

pub const SAMPLE_RATE: usize = 16_000;
pub const MAX_AUDIO_SECONDS: usize = 30;
pub const MAX_AUDIO_COUNT: usize = 8;
pub const MAX_TOTAL_AUDIO_SAMPLES: usize = 60 * SAMPLE_RATE;
pub const MAX_AUDIO_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TOTAL_AUDIO_BYTES: usize = 24 * 1024 * 1024;

/// Mono, 16 kHz, finite samples. Construction enforces the audio adapter's bounds.
#[derive(Clone, Debug)]
pub struct EmbeddingAudio {
    samples: Vec<f32>,
}
impl EmbeddingAudio {
    pub fn from_mono_16k(samples: Vec<f32>) -> Result<Self, AudioError> {
        if samples.len() > MAX_AUDIO_SECONDS * SAMPLE_RATE {
            return Err(AudioError::CapacityExceeded {
                resource: "audio duration",
            });
        }
        if samples.len() <= 160 || samples.iter().any(|x| !x.is_finite() || x.abs() > 2.0) {
            return Err(AudioError::InvalidInput {
                field: "embedding audio",
                reason: "expected more than 10 ms of finite normalized PCM".into(),
            });
        }
        Ok(Self { samples })
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// No cropping: overlong audio is rejected rather than silently truncated.
    pub fn decode(bytes: &[u8]) -> Result<Self, AudioError> {
        let limits = DecodeLimits {
            max_encoded_bytes: MAX_AUDIO_BYTES,
            max_duration_seconds: MAX_AUDIO_SECONDS as u32,
            ..DecodeLimits::default()
        };
        let pcm = ironmlx_audio::io::NativeAudioIo.decode(bytes, &limits)?;
        if pcm.samples.iter().any(|x| x.abs() > 1.0) {
            return Err(AudioError::InvalidInput {
                field: "embedding audio",
                reason: "source PCM must be normalized to [-1, 1]".into(),
            });
        }
        let channels = usize::from(pcm.format.channels);
        let mono = pcm
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().map(|x| x / channels as f32).sum())
            .collect::<Vec<f32>>();
        let samples = ironmlx_audio::signal::resample_mono(
            &mono,
            pcm.format.sample_rate,
            SAMPLE_RATE as u32,
            &InputControl,
        )?;
        Self::from_mono_16k(samples)
    }
}
struct InputControl;
impl SessionControl for InputControl {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_audio_rejects_invalid_samples_and_duration() {
        assert!(EmbeddingAudio::from_mono_16k(vec![0.0; 160]).is_err());
        assert!(EmbeddingAudio::from_mono_16k(vec![f32::NAN; 161]).is_err());
        assert!(EmbeddingAudio::from_mono_16k(vec![f32::INFINITY; 161]).is_err());
        assert!(matches!(
            EmbeddingAudio::from_mono_16k(vec![0.0; 480001]),
            Err(AudioError::CapacityExceeded { .. })
        ));
        assert_eq!(
            EmbeddingAudio::from_mono_16k(vec![0.0; 480000])
                .unwrap()
                .samples()
                .len(),
            480000
        );
    }
    #[test]
    fn file_decoder_preserves_pcm_and_rejects_truncation() {
        let bytes = include_bytes!("../../tests/fixtures/embedding_gemma2/audio/tone.wav");
        let audio = EmbeddingAudio::decode(bytes).unwrap();
        assert_eq!(audio.samples().len(), 16000);
        assert!(EmbeddingAudio::decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(EmbeddingAudio::decode(b"not an audio file").is_err());
    }

    #[test]
    fn stereo_is_downmixed_and_resampled_before_duration_validation() {
        use ironmlx_audio::{PcmBuffer, PcmFormat};
        for sample_rate in [8000, 44100, 96000] {
            let bytes = ironmlx_audio::io::NativeAudioIo
                .encode_wav(&PcmBuffer {
                    format: PcmFormat {
                        sample_rate,
                        channels: 2,
                    },
                    samples: [0.25, -0.25].repeat(sample_rate as usize),
                })
                .unwrap();
            let audio = EmbeddingAudio::decode(&bytes).unwrap();
            assert_eq!(audio.samples().len(), SAMPLE_RATE);
            assert!(audio.samples().iter().all(|v| *v == 0.0));
        }
        for (sample_rate, channels) in [(7999, 1), (96001, 1), (16000, 3)] {
            let bytes = ironmlx_audio::io::NativeAudioIo
                .encode_wav(&PcmBuffer {
                    format: PcmFormat {
                        sample_rate,
                        channels,
                    },
                    samples: vec![0.0; sample_rate as usize * usize::from(channels)],
                })
                .unwrap();
            assert!(EmbeddingAudio::decode(&bytes).is_err());
        }
    }
}
