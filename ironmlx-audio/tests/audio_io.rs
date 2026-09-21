use ironmlx_audio::{
    io::{write_pcm_s16le, NativeAudioIo},
    signal::prepare_reference,
    AudioError, AudioIo, DecodeLimits, PcmBuffer, PcmFormat, Result, SessionControl,
};
struct Active;
impl SessionControl for Active {
    fn check(&self) -> Result<()> {
        Ok(())
    }
}
fn pcm(samples: Vec<f32>) -> PcmBuffer {
    PcmBuffer {
        format: PcmFormat {
            sample_rate: 22050,
            channels: 1,
        },
        samples,
    }
}
#[test]
fn wav_and_pcm_use_identical_quantization() {
    let pcm = pcm(vec![-1.5, -1., -0.5 / 32768., 0., 0.5 / 32768., 1., 1.5]);
    let wav = NativeAudioIo.encode_wav(&pcm).unwrap();
    let mut raw = Vec::new();
    write_pcm_s16le(&pcm, &mut raw).unwrap();
    assert_eq!(&wav[44..], raw);
    let values: Vec<i16> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(values, [-32768, -32768, -1, 0, 1, 32767, 32767]);
    let decoded = NativeAudioIo
        .decode(&wav, &DecodeLimits::default())
        .unwrap();
    assert_eq!(decoded.format, pcm.format);
    assert_eq!(
        decoded.samples,
        values
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect::<Vec<_>>()
    );
}
#[test]
fn rejects_bad_container_truncation_nonfinite_and_limits() {
    let wav = NativeAudioIo.encode_wav(&pcm(vec![0.; 22050])).unwrap();
    for len in [0, 4, 11, 20, 44, wav.len() - 1] {
        assert!(NativeAudioIo
            .decode(&wav[..len], &DecodeLimits::default())
            .is_err());
    }
    let mut trailing = wav.clone();
    trailing.push(0);
    assert!(NativeAudioIo
        .decode(&trailing, &DecodeLimits::default())
        .is_err());
    let limits = DecodeLimits {
        max_pcm_bytes: 4,
        ..Default::default()
    };
    assert!(matches!(
        NativeAudioIo.decode(&wav, &limits),
        Err(AudioError::CapacityExceeded { .. })
    ));
    assert!(NativeAudioIo.encode_wav(&pcm(vec![f32::NAN])).is_err());
    for text in [
        "https://example.com/a.wav",
        "data:audio/wav;base64,UklGRg==",
        "UklGRg==\n",
        "UklGRh==",
    ] {
        assert!(NativeAudioIo
            .decode_base64(text, &DecodeLimits::default())
            .is_err());
    }
}
#[test]
fn reference_checks_tail_before_crop_and_cancellation() {
    let mut audio = PcmBuffer {
        format: PcmFormat {
            sample_rate: 8000,
            channels: 2,
        },
        samples: vec![0.25; 8000 * 16 * 2],
    };
    let result = prepare_reference(&audio, &Active).unwrap();
    assert_eq!(result.source_frames, 8000 * 16);
    assert_eq!(result.used_frames, 8000 * 15);
    assert_eq!(result.semantic_16k.samples.len(), 16000 * 15);
    assert_eq!(result.mel_22050.samples.len(), 22050 * 15);
    assert!((result.semantic_16k.samples[16000] - 0.25).abs() < 1e-3);
    *audio.samples.last_mut().unwrap() = f32::NAN;
    assert!(prepare_reference(&audio, &Active).is_err());
    struct Cancelled;
    impl SessionControl for Cancelled {
        fn check(&self) -> Result<()> {
            Err(AudioError::Cancelled)
        }
    }
    assert!(matches!(
        prepare_reference(&audio, &Cancelled),
        Err(AudioError::Cancelled)
    ));
}
