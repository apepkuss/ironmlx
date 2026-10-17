//! Streaming CPU speech detection. Original 24 kHz PCM is retained for ASR.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const FRAME: usize = 384; // 16 ms at 24 kHz; Earshot consumes 256 samples at 16 kHz.

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VadSettings {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default = "threshold")]
    pub threshold: f32,
    #[serde(default = "padding")]
    pub prefix_padding_ms: usize,
    #[serde(default = "silence")]
    pub silence_duration_ms: usize,
}
fn threshold() -> f32 {
    0.5
}
fn padding() -> usize {
    300
}
fn silence() -> usize {
    500
}
impl VadSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != "server_vad"
            || !self.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.threshold)
            || self.prefix_padding_ms > 1000
            || !(100..=3000).contains(&self.silence_duration_ms)
        {
            return Err("Expected server_vad, threshold 0–1, prefix_padding_ms 0–1000, silence_duration_ms 100–3000".into());
        }
        Ok(())
    }
}

pub struct VadFrame {
    pub audio: Vec<i16>,
    pub started: Option<u64>,
    pub stopped: Option<u64>,
}

pub struct VoiceDetector {
    detector: earshot::Detector,
    settings: VadSettings,
    carry: Vec<i16>,
    prefix: VecDeque<i16>,
    active: bool,
    quiet: usize,
    active_samples: usize,
    position: u64,
}
impl VoiceDetector {
    pub fn new(settings: VadSettings, position: u64) -> Result<Self, String> {
        settings.validate()?;
        Ok(Self {
            detector: earshot::Detector::default(),
            settings,
            carry: Vec::new(),
            prefix: VecDeque::new(),
            active: false,
            quiet: 0,
            active_samples: 0,
            position,
        })
    }
    /// Samples retained by detection but not yet transferred to an ASR turn.
    pub fn buffered_samples(&self) -> usize {
        self.prefix.len() + self.carry.len()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<VadFrame> {
        let mut events = Vec::new();
        for sample in bytes.chunks_exact(2) {
            self.carry.push(i16::from_le_bytes([sample[0], sample[1]]));
            if self.carry.len() != FRAME {
                continue;
            }
            let mut downsampled = [0i16; 256];
            for (input, output) in self
                .carry
                .chunks_exact(3)
                .zip(downsampled.chunks_exact_mut(2))
            {
                output[0] = ((2 * i32::from(input[0]) + i32::from(input[1])) / 3) as i16;
                output[1] = ((i32::from(input[1]) + 2 * i32::from(input[2])) / 3) as i16;
            }
            let score = self.detector.predict_i16(&downsampled);
            let frame = std::mem::take(&mut self.carry);
            if let Some(event) = self.classify(frame, score) {
                events.push(event);
            }
        }
        events
    }
    fn classify(&mut self, frame: Vec<i16>, score: f32) -> Option<VadFrame> {
        let start = self.position;
        self.position += frame.len() as u64;
        if !self.active && score < self.settings.threshold {
            self.prefix.extend(frame);
            let keep = self.settings.prefix_padding_ms * 24;
            self.prefix.drain(..self.prefix.len().saturating_sub(keep));
            return None;
        }
        let started = (!self.active).then_some(start.saturating_sub(self.prefix.len() as u64));
        self.active = true;
        self.quiet = if score >= self.settings.threshold {
            0
        } else {
            self.quiet + frame.len()
        };
        let mut audio: Vec<_> = self.prefix.drain(..).collect();
        audio.extend(frame);
        self.active_samples += audio.len();
        let stopped = (self.active_samples >= 28 * 24000
            || self.quiet >= self.settings.silence_duration_ms * 24)
            .then_some(self.position);
        if stopped.is_some() {
            self.active = false;
            self.quiet = 0;
            self.active_samples = 0;
        }
        Some(VadFrame {
            audio,
            started,
            stopped,
        })
    }
    /// Flush incomplete detector frames without dropping accepted audio on explicit commit.
    pub fn finish(&mut self) -> Vec<i16> {
        self.position += self.carry.len() as u64;
        let audio = if self.active {
            std::mem::take(&mut self.carry)
        } else {
            self.carry.clear();
            Vec::new()
        };
        self.prefix.clear();
        self.active = false;
        self.quiet = 0;
        self.active_samples = 0;
        self.detector.reset();
        audio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn settings() -> VadSettings {
        serde_json::from_str(r#"{"type":"server_vad"}"#).unwrap()
    }
    #[test]
    fn silence_is_bounded_and_speech_preserves_prefix_and_hangover() {
        let mut vad = VoiceDetector::new(settings(), 0).unwrap();
        for _ in 0..1000 {
            assert!(vad.classify(vec![0; FRAME], 0.).is_none());
        }
        assert_eq!(vad.prefix.len(), 7200);
        let start = vad.classify(vec![1; FRAME], 1.).unwrap();
        assert_eq!(start.started, Some(384000 - 7200));
        assert_eq!(start.audio.len(), 7200 + FRAME);
        for _ in 0..31 {
            assert!(vad.classify(vec![0; FRAME], 0.).unwrap().stopped.is_none());
        }
        assert!(vad.classify(vec![0; FRAME], 0.).unwrap().stopped.is_some());
        assert!(vad.classify(vec![0; FRAME], 0.).is_none());
    }
    #[test]
    fn continuous_speech_rollover_covers_every_sample_and_commit_flushes_tail() {
        let mut vad = VoiceDetector::new(settings(), 0).unwrap();
        let mut total = 0;
        let mut starts = 0;
        let mut stops = 0;
        for _ in 0..2000 {
            let event = vad.classify(vec![1; FRAME], 1.).unwrap();
            total += event.audio.len();
            starts += usize::from(event.started.is_some());
            stops += usize::from(event.stopped.is_some());
        }
        assert_eq!((total, starts, stops), (2000 * FRAME, 2, 1));
        assert!(vad.push(&[1, 0, 2, 0]).is_empty());
        assert_eq!(vad.finish(), vec![1, 2]);
        assert!(vad.finish().is_empty());
        assert_eq!(vad.position, (2000 * FRAME + 2) as u64);
    }

    #[test]
    fn packet_boundaries_do_not_change_detection_and_silence_never_starts_speech() {
        let mut whole = VoiceDetector::new(settings(), 0).unwrap();
        let mut split = VoiceDetector::new(settings(), 0).unwrap();
        let samples: Vec<i16> = (0..24000).map(|i| ((i * 197) % 32000) as i16).collect();
        let bytes: Vec<_> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let summarize = |events: Vec<VadFrame>| {
            events
                .into_iter()
                .map(|e| (e.audio, e.started, e.stopped))
                .collect::<Vec<_>>()
        };
        let expected = summarize(whole.push(&bytes));
        let actual = summarize(
            bytes
                .chunks(202)
                .flat_map(|chunk| split.push(chunk))
                .collect(),
        );
        assert_eq!(expected, actual);
        let mut silent = VoiceDetector::new(settings(), 0).unwrap();
        assert!(silent.push(&vec![0; 48000]).is_empty());
        assert!(silent.carry.len() < FRAME);
    }
}
