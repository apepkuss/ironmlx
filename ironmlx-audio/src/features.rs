//! Fixed IndexTTS reference spectrograms, computed without a Python runtime.
use crate::{error::invalid, PcmBuffer, Result, SessionControl};
use std::f64::consts::PI;

#[derive(serde::Deserialize)]
struct Windows {
    hann1024: Vec<f32>,
    povey400: Vec<f32>,
}
fn windows() -> &'static Windows {
    static WINDOWS: std::sync::OnceLock<Windows> = std::sync::OnceLock::new();
    WINDOWS.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../resources/indextts25/analysis-windows.json"
        ))
        .expect("fixed analysis windows")
    })
}

/// Time-major dense features. Dimensions and storage are always validated on construction.
#[derive(Debug)]
pub struct FeatureMatrix {
    pub(crate) frames: usize,
    pub(crate) width: usize,
    pub(crate) values: Vec<f32>,
}
impl FeatureMatrix {
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Paired 80-bin frames and the corresponding stride-two padding mask.
pub struct SemanticFeatures {
    pub features: FeatureMatrix,
    pub attention_mask: Vec<i32>,
}

fn check_audio(pcm: &PcmBuffer, sample_rate: u32) -> Result<()> {
    pcm.validate()?;
    if pcm.format.channels != 1
        || pcm.format.sample_rate != sample_rate
        || pcm.samples.len() < sample_rate as usize
        || pcm.samples.len() > sample_rate as usize * 15
    {
        return Err(invalid(
            "reference_features",
            "expected 1–15 seconds at the fixed mono rate",
        ));
    }
    Ok(())
}

fn checked_matrix(frames: usize, width: usize, values: Vec<f32>) -> Result<FeatureMatrix> {
    if values.len() != frames * width || values.iter().any(|x| !x.is_finite()) {
        return Err(invalid(
            "reference_features",
            "non-finite feature computation",
        ));
    }
    Ok(FeatureMatrix {
        frames,
        width,
        values,
    })
}

// Radix-two FFT in f64 preserves the NumPy feature extractor's computation precision.
// Only the fixed 512/1024-point feature windows call this private routine.
fn spectrum(frame: &[f64]) -> Vec<(f64, f64)> {
    let n = frame.len();
    let mut out: Vec<_> = frame.iter().map(|&x| (x, 0.)).collect();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            out.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2. * PI / len as f64;
        let step = (angle.cos(), angle.sin());
        for base in (0..n).step_by(len) {
            let mut twiddle = (1., 0.);
            for k in 0..len / 2 {
                let a = out[base + k];
                let b = out[base + k + len / 2];
                let b = (
                    b.0 * twiddle.0 - b.1 * twiddle.1,
                    b.0 * twiddle.1 + b.1 * twiddle.0,
                );
                out[base + k] = (a.0 + b.0, a.1 + b.1);
                out[base + k + len / 2] = (a.0 - b.0, a.1 - b.1);
                twiddle = (
                    twiddle.0 * step.0 - twiddle.1 * step.1,
                    twiddle.0 * step.1 + twiddle.1 * step.0,
                );
            }
        }
        len *= 2;
    }
    out.truncate(n / 2 + 1);
    out
}

fn spectrum_f32(frame: &[f64]) -> Result<Vec<f32>> {
    let values: Vec<f32> = frame.iter().map(|&x| x as f32).collect();
    let n = values.len() as i32;
    let input = mlx::Array::try_from((values.as_slice(), [n]))?;
    let fft = mlx::ops::fft::rfft_on(
        &input,
        n,
        -1,
        mlx::ops::fft::FftNorm::Backward,
        mlx::Device::cpu(),
    )?;
    Ok(mlx::ops::abs_on(&fft, mlx::Device::cpu())?.to_vec::<f32>()?)
}

fn slaney_mel(hz: f64) -> f64 {
    if hz < 1000. {
        hz / (200. / 3.)
    } else {
        15. + (hz / 1000.).ln() / (6.4_f64.ln() / 27.)
    }
}
fn slaney_hz(mel: f64) -> f64 {
    if mel < 15. {
        mel * (200. / 3.)
    } else {
        1000. * ((mel - 15.) * (6.4_f64.ln() / 27.)).exp()
    }
}

/// 22050 Hz, FFT/window 1024, hop 256, Slaney 80-bin magnitude mel, log floor 1e-5.
pub fn reference_mel(pcm: &PcmBuffer, control: &dyn SessionControl) -> Result<FeatureMatrix> {
    control.check()?;
    check_audio(pcm, 22050)?;
    let frames = (pcm.samples.len() + 768 - 1024) / 256 + 1;
    let edges: Vec<_> = (0..82)
        .map(|i| slaney_hz(slaney_mel(11025.) * i as f64 / 81.))
        .collect();
    let mut filters = vec![vec![0f32; 513]; 80];
    for (m, filter) in filters.iter_mut().enumerate() {
        for (k, weight) in filter.iter_mut().enumerate() {
            let hz = k as f64 * 22050. / 1024.;
            let triangle = ((hz - edges[m]) / (edges[m + 1] - edges[m]))
                .min((edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))
                .max(0.) as f32;
            *weight = (f64::from(triangle) * 2. / (edges[m + 2] - edges[m])) as f32;
        }
    }
    let window = &windows().hann1024;
    let mut values = Vec::with_capacity(frames * 80);
    for t in 0..frames {
        control.check()?;
        let mut frame = vec![0.; 1024];
        for k in 0..1024 {
            let index = (t * 256 + k) as isize - 384;
            let index = if index < 0 {
                -index
            } else if index >= pcm.samples.len() as isize {
                2 * pcm.samples.len() as isize - index - 2
            } else {
                index
            } as usize;
            frame[k] = f64::from(pcm.samples[index] * window[k]);
        }
        // PyTorch's reference mel uses a float32 FFT, unlike SeamlessM4T's
        // float64 NumPy FFT. Preserve that distinction with MLX's CPU transform.
        let magnitudes: Vec<f32> = spectrum_f32(&frame)?
            .into_iter()
            .map(|x| (x * x + 1e-9).sqrt())
            .collect();
        if magnitudes.iter().any(|x| !x.is_finite()) {
            return Err(invalid("reference_features", "spectral magnitude overflow"));
        }
        for filter in &filters {
            let value: f32 = filter.iter().zip(&magnitudes).map(|(a, b)| a * b).sum();
            values.push(value.max(1e-5).ln());
        }
    }
    checked_matrix(frames, 80, values)
}

fn fbank(pcm: &PcmBuffer, seamless: bool, control: &dyn SessionControl) -> Result<FeatureMatrix> {
    control.check()?;
    check_audio(pcm, 16000)?;
    let frames = (pcm.samples.len() - 400) / 160 + 1;
    let low = 1127. * (1. + 20. / 700_f64).ln();
    let delta = (1127. * (1. + 8000. / 700_f64).ln() - low) / 81.;
    let mut filters = vec![vec![0.; 257]; 80];
    for (m, filter) in filters.iter_mut().enumerate() {
        for (k, weight) in filter.iter_mut().enumerate() {
            if seamless {
                let freq = 1127. * (1. + k as f64 * 31.25 / 700.).ln();
                let left = low + m as f64 * delta;
                *weight = ((freq - left) / delta)
                    .min((left + 2. * delta - freq) / delta)
                    .max(0.);
            } else if k < 256 {
                let freq = 1127_f32 * (1. + k as f32 * 31.25 / 700.).ln();
                let left = low as f32 + m as f32 * delta as f32;
                let center = low as f32 + (m + 1) as f32 * delta as f32;
                let right = low as f32 + (m + 2) as f32 * delta as f32;
                *weight = f64::from(
                    ((freq - left) / (center - left))
                        .min((right - freq) / (right - center))
                        .max(0.),
                );
            }
        }
    }
    let mut values = Vec::with_capacity(frames * 80);
    for t in 0..frames {
        control.check()?;
        let wave = &pcm.samples[t * 160..t * 160 + 400];
        let mean = wave.iter().map(|&v| f64::from(v)).sum::<f64>() / 400.;
        let mut frame = vec![0.; 512];
        for k in 0..400 {
            if seamless {
                let current = (f64::from(wave[k]) - mean) * 32768.;
                let previous = (f64::from(wave[k.saturating_sub(1)]) - mean) * 32768.;
                let window = (0.5 - 0.5 * (2. * PI * k as f64 / 399.).cos()).powf(0.85);
                frame[k] = (current - 0.97 * previous) * window;
            } else {
                let current = wave[k] - mean as f32;
                let previous = wave[k.saturating_sub(1)] - mean as f32;
                let window = windows().povey400[k];
                frame[k] = f64::from((current - 0.97 * previous) * window);
            }
        }
        let power: Vec<f64> = if seamless {
            spectrum(&frame)
                .into_iter()
                .map(|(re, im)| {
                    // NumPy stores the float64 FFT into a complex64 array, then
                    // evaluates magnitude/power in float64.
                    f64::from(re as f32).powi(2) + f64::from(im as f32).powi(2)
                })
                .collect()
        } else {
            spectrum_f32(&frame)?
                .into_iter()
                .map(|x| f64::from(x * x))
                .collect()
        };
        if power.iter().any(|x| !x.is_finite()) {
            return Err(invalid("reference_features", "spectral power overflow"));
        }
        for filter in &filters {
            let value = if seamless {
                filter
                    .iter()
                    .zip(&power)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .max(f64::from(f32::EPSILON))
                    .ln() as f32
            } else {
                filter
                    .iter()
                    .zip(&power)
                    .map(|(a, b)| (*a as f32) * (*b as f32))
                    .sum::<f32>()
                    .max(f32::EPSILON)
                    .ln()
            };
            values.push(value);
        }
    }
    checked_matrix(frames, 80, values)
}

/// CAMPPlus Kaldi fbank: dither 0, 80 bins, 25 ms Povey window, 10 ms hop,
/// snip edges, preemphasis 0.97 and per-bin time-mean subtraction.
pub fn speaker_fbank(pcm: &PcmBuffer, control: &dyn SessionControl) -> Result<FeatureMatrix> {
    let mut result = fbank(pcm, false, control)?;
    for bin in 0..80 {
        let mean = result
            .values
            .chunks_exact(80)
            .map(|row| row[bin])
            .sum::<f32>()
            / result.frames as f32;
        for row in result.values.chunks_exact_mut(80) {
            row[bin] -= mean;
        }
    }
    checked_matrix(result.frames, result.width, result.values)
}

/// SeamlessM4T per-bin sample-variance normalization, right padding with 1,
/// then pairs adjacent frames into 160-wide vectors and selects each second mask bit.
pub fn semantic_features(
    pcm: &PcmBuffer,
    control: &dyn SessionControl,
) -> Result<SemanticFeatures> {
    let mut result = fbank(pcm, true, control)?;
    for bin in 0..80 {
        let mean = result
            .values
            .chunks_exact(80)
            .map(|row| row[bin])
            .sum::<f32>()
            / result.frames as f32;
        let var = result
            .values
            .chunks_exact(80)
            .map(|row| (row[bin] - mean).powi(2))
            .sum::<f32>()
            / (result.frames - 1) as f32;
        for row in result.values.chunks_exact_mut(80) {
            row[bin] = (row[bin] - mean) / (var + 1e-7).sqrt();
        }
    }
    let needs_padding = !result.frames.is_multiple_of(2);
    let mut attention_mask = vec![1; result.frames.div_ceil(2)];
    if needs_padding {
        result.values.extend([1.; 80]);
        *attention_mask.last_mut().unwrap() = 0;
    }
    result.frames = result.frames.div_ceil(2);
    result.width = 160;
    Ok(SemanticFeatures {
        features: checked_matrix(result.frames, result.width, result.values)?,
        attention_mask,
    })
}
