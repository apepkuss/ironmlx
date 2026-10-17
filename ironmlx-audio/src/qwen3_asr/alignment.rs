//! Qwen3 timestamp classification and monotonic repair.
//! Algorithm adapted from QwenLM/Qwen3-ASR (Apache-2.0), Copyright 2026 Alibaba Qwen team.
use super::{features, model::Qwen3Asr};
use crate::{PcmBuffer, SessionControl};
use anyhow::{anyhow, ensure, Result};
use mlx::{ops, Array};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct AlignedWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Alignment {
    pub text: String,
    pub language: String,
    pub duration: f64,
    pub words: Vec<AlignedWord>,
}

// Chinese ideographs are individual units; Japanese uses explicit character units,
// Korean uses whitespace units. No hidden external tokenizer/model downloads.
pub(super) fn units(text: &str, language: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut word = String::new();
    let flush = |word: &mut String, result: &mut Vec<String>| {
        if !word.is_empty() {
            result.push(std::mem::take(word));
        }
    };
    for ch in text.chars() {
        let cjk = matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2ceaf);
        if cjk || (language == "Japanese" && ch.is_alphanumeric()) {
            flush(&mut word, &mut result);
            result.push(ch.to_string());
        } else if ch.is_whitespace() {
            flush(&mut word, &mut result);
        } else if ch.is_alphanumeric() || ch == '\'' {
            word.push(ch);
        }
    }
    flush(&mut word, &mut result);
    result
}

pub(super) fn validate_text(text: &str, language: &str) -> Result<()> {
    if text.trim().is_empty() || text.len() > 16384 || {
        let count = units(text, language).len();
        count == 0 || count > 2048
    } {
        return Err(crate::AudioError::InvalidInput {
            field: "text",
            reason: "Alignment text requires 1..16384 UTF-8 bytes and 1..2048 spoken units".into(),
        }
        .into());
    }
    Ok(())
}

fn repair(data: &[u32]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    // Match the official first longest nondecreasing subsequence and predecessor ties.
    let mut lengths = vec![1; data.len()];
    let mut parent = vec![None; data.len()];
    for i in 1..data.len() {
        for j in 0..i {
            if data[j] <= data[i] && lengths[j] + 1 > lengths[i] {
                lengths[i] = lengths[j] + 1;
                parent[i] = Some(j);
            }
        }
    }
    let best = *lengths.iter().max().unwrap();
    let mut index = lengths.iter().position(|v| *v == best);
    let mut normal = vec![false; data.len()];
    while let Some(i) = index {
        normal[i] = true;
        index = parent[i];
    }
    let mut result = data.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
    let mut i = 0;
    while i < data.len() {
        if normal[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < data.len() && !normal[i] {
            i += 1;
        }
        let left = start.checked_sub(1).map(|j| result[j]);
        let right = result.get(i).copied();
        let count = i - start;
        for (j, value) in result.iter_mut().enumerate().take(i).skip(start) {
            *value = match (left, right) {
                (Some(l), Some(r)) if count > 2 => {
                    l + (r - l) * (j - start + 1) as f64 / (count + 1) as f64
                }
                (Some(l), Some(r)) => {
                    if j - start < i - j {
                        l
                    } else {
                        r
                    }
                }
                (Some(l), None) => l,
                (None, Some(r)) => r,
                _ => unreachable!(),
            };
        }
    }
    result
}

impl Qwen3Asr {
    pub fn align(
        &self,
        pcm: &PcmBuffer,
        text: &str,
        language: &str,
        control: &dyn SessionControl,
    ) -> Result<Alignment> {
        control.check()?;
        ensure!(self.is_aligner(), "Alignment requires Qwen3-ForcedAligner");
        let language = self.config.language(Some(language))?.unwrap();
        validate_text(text, &language)?;
        pcm.validate()?;
        if pcm.frames()? as u64 > u64::from(pcm.format.sample_rate) * 300 {
            return Err(crate::AudioError::InvalidInput {
                field: "file", reason: "Standalone alignment audio exceeds 300 seconds; submit bounded audio/text pairs".into()
            }.into());
        }
        ensure!(
            (8000..=96000).contains(&pcm.format.sample_rate)
                && (1..=2).contains(&pcm.format.channels),
            "Unsupported alignment PCM format"
        );
        let mono = pcm
            .samples
            .chunks_exact(pcm.format.channels as usize)
            .map(|f| f.iter().sum::<f32>() / f.len() as f32)
            .collect::<Vec<_>>();
        let samples = crate::signal::resample_mono_with_limit(
            &mono,
            pcm.format.sample_rate,
            16000,
            300,
            control,
        )?;
        ensure!(
            samples.len() <= 300 * 16000,
            "Standalone alignment audio exceeds 300 seconds; submit bounded audio/text pairs"
        );
        let duration = samples.len() as f64 / 16000.;
        let words = units(text, &language);

        // Reject impossible audio/text contexts before submitting the audio encoder.
        let audio_count = features::output_length(samples.len().div_ceil(160));
        let after = format!(
            "<|audio_end|>{}<timestamp><timestamp>",
            words.join("<timestamp><timestamp>")
        );
        let right_ids = self
            .tokenizer
            .encode(after, false)
            .map_err(|e| anyhow!("Alignment tokenizer: {e}"))?
            .get_ids()
            .to_vec();
        let size = 1 + audio_count + right_ids.len();
        if size > 8192 {
            return Err(crate::AudioError::InvalidInput {
                field: "text",
                reason: "Alignment audio/text context exceeds 8192 tokens".into(),
            }
            .into());
        }
        let indices = right_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == 151705)
            .map(|(i, _)| (1 + audio_count + i) as i32)
            .collect::<Vec<_>>();
        ensure!(
            indices.len() == words.len() * 2,
            "Alignment timestamp token count mismatch"
        );
        let features = features::extract(&samples)?;
        let audio = self
            .encode_audio(&features, control)?
            .astype(self.embed.output_dtype())?;
        ensure!(
            audio.shape().as_slice()[1] as usize == audio_count,
            "Alignment audio token count mismatch"
        );
        let left = self
            .embed
            .forward(&Array::try_from((&[151669u32][..], [1, 1]))?)?;
        let right = self.embed.forward(&Array::try_from((
            right_ids.as_slice(),
            [1, right_ids.len() as i32],
        ))?)?;
        let mut hidden = ops::concatenate(&[&left, &audio, &right], 1)?;
        for layer in &self.decoder {
            control.check()?;
            hidden = layer.forward(
                &hidden,
                &self.config.thinker_config.text_config,
                &mut None,
                0,
            )?;
            hidden.eval()?;
        }
        let selected = ops::take(
            &hidden,
            &Array::try_from((indices.as_slice(), [indices.len() as i32]))?,
            1,
        )?;
        let selected = self.final_text_norm.forward(&selected)?;
        let logits = self.timestamp_head.as_ref().unwrap().forward(&selected)?;
        let ids = logits.argmax(-1, false)?.to_vec::<u32>()?;
        let times = repair(&ids);
        let words = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| AlignedWord {
                word,
                start: ((times[2 * i] * 0.08).min(duration) * 1000.).round() / 1000.,
                end: ((times[2 * i + 1] * 0.08).min(duration) * 1000.).round() / 1000.,
            })
            .collect();
        control.check()?;
        Ok(Alignment {
            text: text.to_owned(),
            language,
            duration,
            words,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_script_units_and_monotonic_repair() {
        assert_eq!(
            units("你好，IronMLX! It's good.", "Chinese"),
            ["你", "好", "IronMLX", "It's", "good"]
        );
        assert_eq!(
            units("こんにちは世界", "Japanese"),
            ["こ", "ん", "に", "ち", "は", "世", "界"]
        );
        assert_eq!(repair(&[0, 1, 8, 3, 4]), vec![0., 1., 1., 3., 4.]);
        for input in [vec![4, 3, 2, 1], vec![0, 8, 7, 6, 5, 10], vec![1, 1, 2, 2]] {
            assert!(repair(&input).windows(2).all(|v| v[0] <= v[1]));
        }
    }
}
