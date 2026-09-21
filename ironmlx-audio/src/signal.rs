//! CPU reference-audio preparation. Resampling follows torchaudio's default
//! sinc-interpolation kernel (Hann, lowpass width 6, rolloff 0.99).
use crate::{
    error::invalid, AudioError, DecodeLimits, PcmBuffer, PcmFormat, Result, SessionControl,
};

pub struct ReferenceWaveforms {
    pub source_frames: usize,
    pub used_frames: usize,
    pub source_sample_rate: u32,
    pub semantic_16k: PcmBuffer,
    pub mel_22050: PcmBuffer,
}

/// Validate the entire decoded reference before downmixing and cropping its prefix.
pub fn prepare_reference(
    pcm: &PcmBuffer,
    control: &dyn SessionControl,
) -> Result<ReferenceWaveforms> {
    control.check()?;
    let source_frames = pcm.frames()?;
    let bounds = DecodeLimits::default();
    if !(bounds.min_sample_rate..=bounds.max_sample_rate).contains(&pcm.format.sample_rate)
        || pcm.format.channels > 2
    {
        return Err(invalid("reference", "expected 8–96 kHz mono or stereo"));
    }
    if source_frames < pcm.format.sample_rate as usize {
        return Err(invalid("reference", "at least one second is required"));
    }
    if source_frames as u64 > u64::from(pcm.format.sample_rate) * 60
        || pcm.samples.len() > bounds.max_pcm_bytes / 4
    {
        return Err(AudioError::CapacityExceeded {
            resource: "reference audio",
        });
    }
    let used_frames = source_frames.min(pcm.format.sample_rate as usize * 15);
    let channels = usize::from(pcm.format.channels);
    let mono: Vec<f32> = pcm.samples[..used_frames * channels]
        .chunks_exact(channels)
        // Divide before summation to keep valid finite stereo peaks finite.
        .map(|frame| frame.iter().map(|x| x / channels as f32).sum())
        .collect();
    let semantic = resample_mono(&mono, pcm.format.sample_rate, 16000, control)?;
    let mel = resample_mono(&mono, pcm.format.sample_rate, 22050, control)?;
    Ok(ReferenceWaveforms {
        source_frames,
        used_frames,
        source_sample_rate: pcm.format.sample_rate,
        semantic_16k: PcmBuffer {
            format: PcmFormat {
                sample_rate: 16000,
                channels: 1,
            },
            samples: semantic,
        },
        mel_22050: PcmBuffer {
            format: PcmFormat {
                sample_rate: 22050,
                channels: 1,
            },
            samples: mel,
        },
    })
}
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Resample mono finite samples, preserving torchaudio's ceil-length convention.
/// Input and output are bounded to the reference profile, including intermediate storage.
pub fn resample_mono(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
    control: &dyn SessionControl,
) -> Result<Vec<f32>> {
    control.check()?;
    if !(8000..=96000).contains(&source_rate)
        || !(8000..=96000).contains(&target_rate)
        || input.is_empty()
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "resample",
            "expected finite nonempty samples and 8–96 kHz rates",
        ));
    }
    if input.len() > source_rate as usize * 60 {
        return Err(AudioError::CapacityExceeded {
            resource: "resample input",
        });
    }
    if source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let divisor = gcd(source_rate, target_rate);
    let original = source_rate / divisor;
    let target = target_rate / divisor;
    let base = f64::from(original.min(target)) * 0.99;
    let width = (6. * f64::from(original) / base).ceil() as i64;
    let out_len = (input.len() as u64 * u64::from(target)).div_ceil(u64::from(original)) as usize;
    // Each phase stores only the nonzero sinc support, not a sparse full convolution kernel.
    let mut kernels = Vec::with_capacity(target as usize);
    for phase in 0..target {
        if phase % 256 == 0 {
            control.check()?;
        }
        let center = f64::from(phase) * f64::from(original) / f64::from(target);
        let radius = 6. * f64::from(original) / base;
        let first = ((center - radius).ceil() as i64).max(-width);
        let last = ((center + radius).floor() as i64).min(width + i64::from(original) - 1);
        let taps = (first..=last)
            .map(|j| {
                // torchaudio.functional.resample builds the kernel in the input's
                // dtype. Keep each arithmetic operation in f32 before convolution.
                let t = ((j as f32 / original as f32 - phase as f32 / target as f32) * base as f32)
                    .clamp(-6., 6.);
                let radians = t * std::f32::consts::PI;
                let window = (radians / 6. / 2.).cos().powi(2);
                let sinc = if radians == 0. {
                    1.
                } else {
                    radians.sin() / radians
                };
                sinc * (window * (base / f64::from(original)) as f32)
            })
            .collect::<Vec<_>>();
        kernels.push((first, taps));
    }
    if source_rate == 16000 && target_rate == 22050 {
        // This common reference-rate conversion uses pinned CPU float32
        // coefficients; libm ulp differences otherwise amplify in log-mel bins.
        #[derive(serde::Deserialize)]
        struct FixedKernel {
            phases: Vec<(i64, Vec<f32>)>,
        }
        static FIXED: std::sync::OnceLock<FixedKernel> = std::sync::OnceLock::new();
        kernels = FIXED
            .get_or_init(|| {
                serde_json::from_str(include_str!(
                    "../resources/indextts25/resample-16000-22050.json"
                ))
                .expect("fixed interpolation kernel")
            })
            .phases
            .clone();
    }
    let mut output = Vec::with_capacity(out_len);
    for index in 0..out_len {
        if index % 4096 == 0 {
            control.check()?;
        }
        let phase = index % target as usize;
        let start = (index / target as usize) as i64 * i64::from(original) + kernels[phase].0;
        let mut value = 0.;
        for (j, weight) in kernels[phase].1.iter().enumerate() {
            let at = start + j as i64;
            if at >= 0 && (at as usize) < input.len() {
                value = input[at as usize].mul_add(*weight, value);
            }
        }
        if !value.is_finite() {
            return Err(invalid("resample", "non-finite result"));
        }
        output.push(value);
    }
    Ok(output)
}
