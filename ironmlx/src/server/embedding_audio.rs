//! Request-wide audio budgets applied before model acquisition or GPU work.
use axum::http::StatusCode;
use base64::{engine::general_purpose::STANDARD, Engine};
use ironmlx_lm::core::audio_input::{
    EmbeddingAudio, MAX_AUDIO_BYTES, MAX_AUDIO_COUNT, MAX_TOTAL_AUDIO_BYTES,
    MAX_TOTAL_AUDIO_SAMPLES,
};
use ironmlx_runtime::core::embedding_execution::EmbeddingAudioInput;

#[derive(Default)]
pub(super) struct AudioRequestBudget {
    count: usize,
    encoded_bytes: usize,
    samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmlx_audio::{io::NativeAudioIo, AudioIo, PcmBuffer, PcmFormat};

    fn wav(frames: usize) -> EmbeddingAudioInput {
        let bytes = NativeAudioIo
            .encode_wav(&PcmBuffer {
                format: PcmFormat {
                    sample_rate: 16000,
                    channels: 1,
                },
                samples: vec![0.0; frames],
            })
            .unwrap();
        EmbeddingAudioInput {
            data: STANDARD.encode(bytes),
            format: "wav".into(),
        }
    }

    #[test]
    fn malformed_audio_is_rejected_before_model_acquisition() {
        for input in [
            EmbeddingAudioInput {
                data: "not base64!".into(),
                format: "wav".into(),
            },
            EmbeddingAudioInput {
                data: STANDARD.encode(b"RIFF\0\0\0\0WAVE"),
                format: "wav".into(),
            },
            EmbeddingAudioInput {
                format: "mp3".into(),
                ..wav(161)
            },
            wav(160),
        ] {
            assert_eq!(
                AudioRequestBudget::default().decode(input).unwrap_err().0,
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn duration_clip_and_request_budgets_are_enforced_without_cropping() {
        let mut budget = AudioRequestBudget::default();
        assert_eq!(budget.decode(wav(480000)).unwrap().samples().len(), 480000);
        assert_eq!(budget.decode(wav(480000)).unwrap().samples().len(), 480000);
        assert_eq!(
            budget.decode(wav(161)).unwrap_err().0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            AudioRequestBudget::default()
                .decode(wav(480001))
                .unwrap_err()
                .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let mut budget = AudioRequestBudget::default();
        for _ in 0..MAX_AUDIO_COUNT {
            budget.decode(wav(161)).unwrap();
        }
        assert_eq!(
            budget.decode(wav(161)).unwrap_err().0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let mut budget = AudioRequestBudget {
            encoded_bytes: MAX_TOTAL_AUDIO_BYTES - 1,
            ..Default::default()
        };
        assert_eq!(
            budget.decode(wav(161)).unwrap_err().0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let mut budget = AudioRequestBudget::default();
        let oversized = EmbeddingAudioInput {
            data: "A".repeat(MAX_AUDIO_BYTES.div_ceil(3) * 4 + 4),
            format: "wav".into(),
        };
        assert_eq!(
            budget.decode(oversized).unwrap_err().0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[test]
    fn every_supported_codec_decodes_to_bounded_pcm() {
        let root = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ironmlx-lm/tests/fixtures/embedding_gemma2/audio/"
        );
        for format in ["wav", "flac", "mp3"] {
            let bytes = std::fs::read(format!("{root}sunny.{format}")).unwrap();
            let input = EmbeddingAudioInput {
                data: STANDARD.encode(bytes),
                format: format.into(),
            };
            let audio = AudioRequestBudget::default().decode(input).unwrap();
            assert!(audio.samples().len() > 160 && audio.samples().len() <= 480000);
        }
    }
}
impl AudioRequestBudget {
    pub fn decode(
        &mut self,
        input: EmbeddingAudioInput,
    ) -> Result<EmbeddingAudio, (StatusCode, String)> {
        self.count += 1;
        let remaining =
            MAX_AUDIO_BYTES.min(MAX_TOTAL_AUDIO_BYTES.saturating_sub(self.encoded_bytes));
        if self.count > MAX_AUDIO_COUNT || input.data.len() > remaining.div_ceil(3) * 4 {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "audio input exceeds the encoded-byte or clip-count budget".into(),
            ));
        }
        let bytes = STANDARD.decode(input.data).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                "audio input requires canonical standard base64".into(),
            )
        })?;
        self.encoded_bytes += bytes.len();
        if bytes.len() > MAX_AUDIO_BYTES || self.encoded_bytes > MAX_TOTAL_AUDIO_BYTES {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "audio input exceeds the encoded-byte budget".into(),
            ));
        }
        let matches_format = match input.format.as_str() {
            "wav" => bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"),
            "flac" => bytes.starts_with(b"fLaC"),
            "mp3" => {
                bytes.starts_with(b"ID3")
                    || (bytes.len() > 1 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
            }
            _ => false,
        };
        if !matches_format {
            return Err((
                StatusCode::BAD_REQUEST,
                "audio input format does not match its data".into(),
            ));
        }
        let audio = EmbeddingAudio::decode(&bytes).map_err(|e| {
            let status = if matches!(e, ironmlx_audio::AudioError::CapacityExceeded { .. }) {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, format!("invalid embedding audio input: {e}"))
        })?;
        self.samples += audio.samples().len();
        if self.samples > MAX_TOTAL_AUDIO_SAMPLES {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "audio input exceeds 60 seconds across the request".into(),
            ));
        }
        Ok(audio)
    }
}
