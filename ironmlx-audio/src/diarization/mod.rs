//! Native Nemotron 3 speaker activity inference. Models and PCM stay on one worker.
mod cache;
mod config;
mod model;
pub use config::{preflight_model_metadata, Config};
pub use model::{NemotronDiarization, StreamingState};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const MODEL_ID: &str = "mlx-community/Nemotron-3-Diarization";
pub const SAMPLE_RATE: u32 = 16_000;
pub const MAX_DURATION_SECONDS: u32 = 7200;
pub type ProbabilityFrame = [f32; 8];
pub type ProbabilityChunk = (usize, Vec<ProbabilityFrame>);
pub type ProgressCallback<'a> = dyn FnMut(usize, &[ProbabilityFrame]) -> Result<()> + 'a;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Latency {
    #[default]
    Offline,
    Low,
    VeryLow,
    UltraLow,
}
impl Latency {
    pub(super) fn windows(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Offline => (340, 40, 40, 300),
            Self::Low => (9, 4, 264, 222),
            Self::VeryLow => (6, 2, 264, 222),
            Self::UltraLow => (3, 1, 264, 222),
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub threshold: f32,
    pub min_duration: f64,
    pub merge_gap: f64,
    pub latency: Latency,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_duration: 0.,
            merge_gap: 0.,
            latency: Latency::Offline,
        }
    }
}
impl Options {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.threshold.is_finite() && self.threshold > 0. && self.threshold < 1.,
            "threshold must be between 0 and 1 (exclusive)"
        );
        ensure!(
            self.min_duration.is_finite() && (0. ..=30.).contains(&self.min_duration),
            "min_duration must be between 0 and 30 seconds"
        );
        ensure!(
            self.merge_gap.is_finite() && (0. ..=5.).contains(&self.merge_gap),
            "merge_gap must be between 0 and 5 seconds"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diarization {
    pub duration: f64,
    pub num_speakers: usize,
    pub segments: Vec<Segment>,
}
impl Diarization {
    pub fn rttm(&self) -> String {
        self.segments
            .iter()
            .map(|s| {
                format!(
                    "SPEAKER audio 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                    s.start,
                    s.end - s.start,
                    s.speaker
                )
            })
            .collect()
    }
}

pub fn segments(probabilities: &[[f32; 8]], offset: usize, options: Options) -> Vec<Segment> {
    let mut segments = Vec::new();
    for speaker in 0..8 {
        let mut current: Option<usize> = None;
        let mut turns: Vec<Segment> = Vec::new();
        for frame in 0..=probabilities.len() {
            let active =
                frame < probabilities.len() && probabilities[frame][speaker] > options.threshold;
            if active && current.is_none() {
                current = Some(frame);
            }
            if !active {
                if let Some(start) = current.take() {
                    if (frame - start) as f64 * 0.01 + 1e-9 >= options.min_duration {
                        let start = (offset + start) as f64 * 0.01;
                        let end = (offset + frame) as f64 * 0.01;
                        if let Some(last) = turns.last_mut().filter(|last| {
                            options.merge_gap > 0. && start - last.end <= options.merge_gap + 1e-9
                        }) {
                            last.end = end;
                        } else {
                            turns.push(Segment {
                                speaker: format!("speaker_{speaker}"),
                                start,
                                end,
                            });
                        }
                    }
                }
            }
        }
        segments.extend(turns);
    }
    segments.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then_with(|| a.speaker.cmp(&b.speaker))
    });
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_activity_and_global_offsets_are_preserved() {
        let mut probabilities = vec![[0.; 8]; 4];
        probabilities[0][0] = 0.9;
        probabilities[1][0] = 0.8;
        probabilities[1][1] = 0.9;
        probabilities[2][1] = 0.9;
        let result = segments(&probabilities, 100, Options::default());
        assert_eq!(result.len(), 2);
        assert_eq!((result[0].start, result[0].end), (1., 1.02));
        assert_eq!((result[1].start, result[1].end), (1.01, 1.03));
        assert_eq!(result[1].speaker, "speaker_1");
    }
    #[test]
    fn short_turns_are_filtered_before_merging() {
        let mut probabilities = vec![[0.; 8]; 9];
        for i in [0, 1, 3, 4, 7] {
            probabilities[i][0] = 0.9;
        }
        let result = segments(
            &probabilities,
            0,
            Options {
                min_duration: 0.02,
                merge_gap: 0.02,
                ..Options::default()
            },
        );
        assert_eq!(result.len(), 1);
        assert_eq!((result[0].start, result[0].end), (0., 0.05));
    }
    #[test]
    fn invalid_options_and_empty_activity_do_not_create_speakers() {
        assert!(segments(&[], 0, Options::default()).is_empty());
        for threshold in [f32::NAN, f32::INFINITY, -1., 0., 1.] {
            assert!(Options {
                threshold,
                ..Options::default()
            }
            .validate()
            .is_err());
        }
        for value in [-1., f64::NAN, f64::INFINITY] {
            assert!(Options {
                min_duration: value,
                ..Options::default()
            }
            .validate()
            .is_err());
            assert!(Options {
                merge_gap: value,
                ..Options::default()
            }
            .validate()
            .is_err());
        }
    }
}
