//! Whisper 128-bin Slaney power log-mel, with reflective centered STFT padding.
use anyhow::{ensure, Result};
use mlx::{ops, Array};

pub(super) const SAMPLE_RATE: usize = 16_000;

fn mel(hz: f64) -> f64 {
    if hz < 1000. {
        hz / (200. / 3.)
    } else {
        15. + (hz / 1000.).ln() / (6.4f64.ln() / 27.)
    }
}
fn hz(mel: f64) -> f64 {
    if mel < 15. {
        mel * (200. / 3.)
    } else {
        1000. * ((mel - 15.) * (6.4f64.ln() / 27.)).exp()
    }
}

pub(super) fn extract(samples: &[f32]) -> Result<Array> {
    ensure!(
        samples.len() > 200 && samples.iter().all(|s| s.is_finite()),
        "ASR audio must contain finite samples and exceed 12.5 ms"
    );
    // The mask counts sample positions 0,160,...; retain the final padded frame
    // for non-integral hops, matching WhisperFeatureExtractor's attention mask.
    let frames = samples.len().div_ceil(160);
    let padded_len = frames * 160;
    let mut windows = Vec::with_capacity(frames * 400);
    for frame in 0..frames {
        for i in 0..400 {
            let source = (frame * 160 + i) as isize - 200;
            let reflected = if source < 0 {
                -source
            } else if source >= padded_len as isize {
                2 * padded_len as isize - 2 - source
            } else {
                source
            } as usize;
            let sample = samples.get(reflected).copied().unwrap_or(0.);
            let window = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 400.).cos();
            windows.push(sample * window as f32);
        }
    }
    let waveform = Array::try_from((windows.as_slice(), [frames as i32, 400]))?;
    let spectrum = ops::fft::rfft(&waveform, 400, -1, ops::fft::FftNorm::Backward)?
        .abs()?
        .square()?;
    let points = (0..130)
        .map(|i| hz(mel(8000.) * i as f64 / 129.))
        .collect::<Vec<_>>();
    let mut filters = vec![0f32; 201 * 128];
    for bin in 0..201 {
        let frequency = bin as f64 * 40.;
        for m in 0..128 {
            let left = (frequency - points[m]) / (points[m + 1] - points[m]);
            let right = (points[m + 2] - frequency) / (points[m + 2] - points[m + 1]);
            filters[bin * 128 + m] =
                (left.min(right).max(0.) * 2. / (points[m + 2] - points[m])) as f32;
        }
    }
    let filters = Array::try_from((filters.as_slice(), [201, 128]))?;
    let power = ops::matmul(&spectrum, &filters)?;
    let floor = Array::try_from((&[1e-10f32][..], []))?;
    let logs = power.maximum(&floor)?.log()? / 10f32.ln();
    let max = ops::max(&logs, ops::All, false)?;
    let normalized = (logs.maximum(&(&max - 8f32))? + 4f32) / 4f32;
    Ok(normalized.transpose()?)
}

pub(super) fn output_length(frames: usize) -> usize {
    (frames / 100) * 13 + (frames % 100).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_lengths_keep_chunk_boundaries() {
        for (input, output) in [
            (1, 1),
            (8, 1),
            (9, 2),
            (99, 13),
            (100, 13),
            (101, 14),
            (3000, 390),
        ] {
            assert_eq!(output_length(input), output);
        }
    }

    #[test]
    fn whisper_mel_matches_transformers_reference() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let samples = (0..16000)
            .map(|i| (0.25 * (std::f64::consts::TAU * 440. * i as f64 / 16000.).sin()) as f32)
            .collect::<Vec<_>>();
        let features = extract(&samples)?;
        assert_eq!(features.shape().as_slice(), &[128, 100]);
        let actual = features.to_vec::<f32>()?;
        let expected = include_bytes!("../../tests/fixtures/qwen3_asr/whisper-mel-440hz.f32");
        let expected = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_error < 2e-4, "Whisper mel max error: {max_error}");
        Ok(())
    }
}
