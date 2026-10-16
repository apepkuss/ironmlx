//! Contiguous, bounded segmentation and script-aware transcript joining.
use std::ops::Range;

pub struct PreparedAudio {
    pub samples: Vec<f32>,
    pub segments: Vec<Range<usize>>,
}

pub fn prepare_audio(
    pcm: &crate::PcmBuffer,
    control: &dyn crate::SessionControl,
) -> anyhow::Result<PreparedAudio> {
    pcm.validate()?;
    anyhow::ensure!(
        pcm.format.channels <= 2 && (8000..=96000).contains(&pcm.format.sample_rate),
        "unsupported ASR PCM format"
    );
    anyhow::ensure!(
        pcm.frames()? as u64 <= u64::from(pcm.format.sample_rate) * 7200,
        "ASR recording exceeds two hours"
    );
    control.check()?;
    let mono = pcm
        .samples
        .chunks_exact(usize::from(pcm.format.channels))
        .map(|frame| frame.iter().map(|v| v / frame.len() as f32).sum::<f32>())
        .collect::<Vec<_>>();
    let samples = crate::signal::resample_mono_with_limit(
        &mono,
        pcm.format.sample_rate,
        16000,
        7200,
        control,
    )?;
    let segments = ranges(&samples);
    control.check()?;
    Ok(PreparedAudio { samples, segments })
}

pub(super) fn ranges(samples: &[f32]) -> Vec<Range<usize>> {
    const MAX: usize = 30 * 16000;
    const SEARCH: usize = 5 * 16000;
    let mut result = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let mut end = (start + MAX).min(samples.len());
        if end < samples.len() {
            // Ten-millisecond energy windows; prefer the latest boundary on ties.
            let search = end - SEARCH;
            let mut best = f64::INFINITY;
            for (i, window) in samples[search..end].chunks(160).enumerate() {
                let energy = window.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
                if energy <= best {
                    best = energy;
                    end = search + i * 160 + window.len();
                }
            }
        }
        result.push(start..end);
        start = end;
    }
    result
}

pub fn append_text(output: &mut String, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let space = output
        .chars()
        .last()
        .is_some_and(|c| c.is_ascii() && !c.is_whitespace())
        && text
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric());
    let delta = if space {
        format!(" {text}")
    } else {
        text.to_owned()
    };
    output.push_str(&delta);
    delta
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn segmentation_covers_every_sample_and_finds_quiet_boundaries() {
        let mut samples = vec![0.5; 95 * 16000 + 7];
        samples[27 * 16000..28 * 16000].fill(0.);
        let parts = ranges(&samples);
        assert_eq!(parts[0].end, 28 * 16000);
        assert_eq!(parts.first().unwrap().start, 0);
        assert_eq!(parts.last().unwrap().end, samples.len());
        assert!(parts.iter().all(|r| !r.is_empty() && r.len() <= 30 * 16000));
        assert!(parts.windows(2).all(|r| r[0].end == r[1].start));
        let mut text = String::new();
        append_text(&mut text, "Hello.");
        append_text(&mut text, "World.");
        assert_eq!(text, "Hello. World.");
        text.clear();
        append_text(&mut text, "你好。");
        append_text(&mut text, "世界。");
        assert_eq!(text, "你好。世界。");
    }
}
