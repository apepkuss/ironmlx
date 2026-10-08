//! Gemma4 semicausal magnitude log-mel features (not a power spectrogram).
//! Matches Transformers Gemma4AudioFeatureExtractor: 320-sample periodic Hann,
//! 160-sample hop, 512-point FFT, 128 HTK filters, log(magnitude + 0.001).
use crate::{core::audio_input::EmbeddingAudio, Result};

pub(super) struct AudioFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
    pub valid_tokens: usize,
}

pub(super) fn token_count(audio: &EmbeddingAudio) -> usize {
    // Frame i ends at source sample i*160 + 160. Two stride-2 mask slices
    // retain frames 0,4,8,...; a frame ending at the sample count is invalid.
    (audio.samples().len() - 161) / 640 + 1
}

pub(super) fn extract(audio: &EmbeddingAudio) -> Result<AudioFeatures> {
    let samples = audio.samples();
    let padded_samples = samples.len().div_ceil(128) * 128;
    let frames = (padded_samples - 161) / 160 + 1;
    let filters = mel_filters();
    let window: Vec<f32> = (0..320)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 320.0).cos()) as f32)
        .collect();
    let mut values = Vec::with_capacity(frames * 128);
    for frame in 0..frames {
        let mut real = [0.0f64; 512];
        let mut imag = [0.0f64; 512];
        for i in 0..320 {
            let source = frame * 160 + i;
            let sample = source
                .checked_sub(160)
                .and_then(|i| samples.get(i))
                .copied()
                .unwrap_or(0.0);
            real[i] = f64::from(sample * window[i]);
        }
        fft(&mut real, &mut imag);
        let valid = frame * 160 + 160 < samples.len();
        for filter in &filters {
            let mel: f64 = filter
                .iter()
                .enumerate()
                .map(|(bin, weight)| real[bin].hypot(imag[bin]) * weight)
                .sum();
            values.push(if valid {
                (mel + 0.001).ln() as f32
            } else {
                0.0
            });
        }
    }
    Ok(AudioFeatures {
        values,
        frames,
        valid_frames: (samples.len() - 161) / 160 + 1,
        valid_tokens: token_count(audio),
    })
}

fn mel_filters() -> Vec<Vec<f64>> {
    let max_mel = 2595.0 * (1.0 + 8000.0f64 / 700.0).log10();
    let frequencies: Vec<_> = (0..130)
        .map(|i| 700.0 * (10.0f64.powf(max_mel * i as f64 / 129.0 / 2595.0) - 1.0))
        .collect();
    (0..128)
        .map(|mel| {
            (0..257)
                .map(|bin| {
                    let frequency = bin as f64 * 16000.0 / 512.0;
                    let left =
                        (frequency - frequencies[mel]) / (frequencies[mel + 1] - frequencies[mel]);
                    let right = (frequencies[mel + 2] - frequency)
                        / (frequencies[mel + 2] - frequencies[mel + 1]);
                    left.min(right).max(0.0)
                })
                .collect()
        })
        .collect()
}

// Fixed-size radix-2 CPU FFT in float64, matching NumPy's FFT precision before
// the feature extractor casts its log-mel output to float32. No GPU allocations.
fn fft(real: &mut [f64; 512], imag: &mut [f64; 512]) {
    for i in 0..512usize {
        let j = i.reverse_bits() >> (usize::BITS - 9);
        if j > i {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= 512 {
        for start in (0..512).step_by(width) {
            for j in 0..width / 2 {
                let (sin, cos) = (-2.0 * std::f64::consts::PI * j as f64 / width as f64).sin_cos();
                let a = start + j;
                let b = a + width / 2;
                let r = real[b] * cos - imag[b] * sin;
                let v = real[b] * sin + imag[b] * cos;
                real[b] = real[a] - r;
                imag[b] = imag[a] - v;
                real[a] += r;
                imag[a] += v;
            }
        }
        width *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn log_mel_and_masks_match_official_feature_extractor() -> Result<()> {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/embedding_gemma2/audio-features-reference.json"
        ))?;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/embedding_gemma2/audio");
        for sample in reference.as_array().unwrap() {
            let audio = EmbeddingAudio::decode(&std::fs::read(
                root.join(sample["audio"].as_str().unwrap()),
            )?)?;
            let features = extract(&audio)?;
            assert_eq!(features.frames, sample["frames"].as_u64().unwrap() as usize);
            assert_eq!(
                features.valid_frames,
                sample["valid_frames"].as_u64().unwrap() as usize
            );
            assert_eq!(
                features.valid_tokens,
                sample["tokens"].as_u64().unwrap() as usize
            );
            for row in sample["rows"].as_array().unwrap() {
                let frame = row["frame"].as_u64().unwrap() as usize;
                for (actual, expected) in features.values[frame * 128..(frame + 1) * 128]
                    .iter()
                    .zip(row["values"].as_array().unwrap())
                {
                    assert!(
                        (f64::from(*actual) - expected.as_f64().unwrap()).abs() < 1e-6,
                        "{} frame {frame}",
                        sample["audio"]
                    );
                }
            }
        }
        for (samples, tokens) in [(161, 1), (800, 1), (801, 2), (480000, 750)] {
            assert_eq!(
                token_count(&EmbeddingAudio::from_mono_16k(vec![0.0; samples])?),
                tokens
            );
        }
        Ok(())
    }
}
