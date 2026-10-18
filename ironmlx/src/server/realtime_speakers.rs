//! IronMLX Realtime extension: continuous speaker state and confirmed word labels.
use super::{diarizations, transcriptions::FIXED_ALIGNER};
use ironmlx_audio::{
    diarization::{self, Diarization, Latency, Options, ProbabilityChunk, ProbabilityFrame},
    PcmBuffer, PcmFormat, SessionControl,
};
use ironmlx_runtime::core::{
    alignment_execution::AlignmentRequest,
    diarization_execution::DiarizationStream,
    engine_pool::{EnginePoolState, EngineVariant},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::VecDeque, future::Future, pin::Pin, sync::Arc};

pub(super) type FeedFuture =
    Pin<Box<dyn Future<Output = Result<(ProbabilityChunk, usize), String>> + Send>>;
pub(super) type AnnotationFuture = Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Settings {
    model: String,
    threshold: f32,
    latency: Latency,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            model: diarization::MODEL_ID.into(),
            threshold: 0.5,
            latency: Latency::Low,
        }
    }
}
impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        if self.model != diarization::MODEL_ID || self.latency == Latency::Offline {
            return Err("Realtime diarization requires mlx-community/Nemotron-3-Diarization and low, very_low or ultra_low latency".into());
        }
        self.options().validate().map_err(|e| e.to_string())
    }
    fn options(&self) -> Options {
        Options {
            threshold: self.threshold,
            latency: self.latency,
            ..Options::default()
        }
    }
}
struct NoCancel;
impl SessionControl for NoCancel {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}
/// Keep sinc right context and the 3:2 phase across arbitrarily sized PCM packets.
/// Reuse the native sinc implementation on a bounded, phase-aligned overlap.
#[derive(Default)]
struct Resampler {
    samples: Vec<f32>,
    offset: usize,
    received: usize,
    emitted: usize,
}
impl Resampler {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        self.samples.extend(
            bytes
                .chunks_exact(2)
                .map(|v| f32::from(i16::from_le_bytes([v[0], v[1]])) / 32768.),
        );
        self.received += bytes.len() / 2;
        let stable = self.received.saturating_sub(12) / 3 * 2;
        let output = self.resample()?;
        let base = self.offset / 3 * 2;
        let result = output[self.emitted - base..stable.max(self.emitted) - base].to_vec();
        self.emitted = stable.max(self.emitted);
        let keep = (self.emitted / 2 * 3).saturating_sub(24) / 3 * 3;
        self.samples.drain(..keep - self.offset);
        self.offset = keep;
        Ok(result)
    }
    fn resample(&self) -> Result<Vec<f32>, String> {
        if self.samples.is_empty() {
            return Ok(Vec::new());
        }
        ironmlx_audio::signal::resample_mono_with_limit(&self.samples, 24000, 16000, 5, &NoCancel)
            .map_err(|e| e.to_string())
    }
    fn tail(&self) -> Result<Vec<f32>, String> {
        let output = self.resample()?;
        Ok(output[self.emitted - self.offset / 3 * 2..].to_vec())
    }
}

pub(super) struct Annotation {
    pub item: String,
    pub pcm: Vec<f32>,
    pub text: String,
    pub language: String,
    pub start_sample: usize,
    pub is_final: bool,
}
pub(super) struct Speakers {
    id: String,
    epoch: usize,
    settings: Settings,
    stream: Arc<DiarizationStream>,
    resampler: Resampler,
    audio: VecDeque<(Vec<u8>, usize)>,
    fed_until: usize,
    history: VecDeque<ProbabilityFrame>,
    history_start: usize,
    annotations: VecDeque<Annotation>,
    input_memory: Arc<ironmlx_runtime::core::process_memory::MemoryReservation>,
}
impl Speakers {
    pub async fn open(
        pool: &EnginePoolState,
        settings: Settings,
        epoch: usize,
        input_memory: Arc<ironmlx_runtime::core::process_memory::MemoryReservation>,
    ) -> Result<Self, String> {
        settings.validate()?;
        if !pool
            .is_alignment_model(FIXED_ALIGNER)
            .await
            .map_err(|e| e.to_string())?
        {
            return Err(format!(
                "Register {FIXED_ALIGNER} for Realtime speaker labels"
            ));
        }
        if !pool
            .is_diarization_model(&settings.model)
            .await
            .map_err(|e| e.to_string())?
        {
            return Err(format!(
                "Register {} for Realtime speaker labels",
                settings.model
            ));
        }
        // Resolve the aligner now so unsupported/unloadable dependencies fail at configuration.
        let (_, aligner) = pool
            .resolve_engine(Some(FIXED_ALIGNER))
            .await
            .map_err(|e| e.to_string())?;
        drop(aligner);
        let (_, lease) = pool
            .resolve_engine(Some(&settings.model))
            .await
            .map_err(|e| e.to_string())?;
        let EngineVariant::Diarization(runtime) = lease.engine() else {
            return Err("Diarization model task mismatch".into());
        };
        let runtime = runtime.clone();
        let stream = runtime
            .start_stream(settings.options(), lease)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self {
            id: format!("diar_{}", uuid::Uuid::new_v4()),
            epoch,
            settings,
            stream: Arc::new(stream),
            resampler: Resampler::default(),
            audio: VecDeque::new(),
            fed_until: epoch,
            history: VecDeque::new(),
            history_start: 0,
            annotations: VecDeque::new(),
            input_memory,
        })
    }
    pub fn started(&self) -> Value {
        json!({"type":"input_audio_buffer.diarization.started","diarization_id":self.id,"audio_start_ms":self.epoch / 24})
    }
    pub fn idle(&self) -> bool {
        self.audio.is_empty() && self.annotations.is_empty()
    }
    pub fn accepts(&self, total: usize, additional: usize) -> bool {
        total + additional <= self.fed_until + 60 * 24000
            && self.annotations.len() < 16
            && self.annotations.iter().map(|a| a.pcm.len()).sum::<usize>() < 60 * 24000
    }
    pub fn append(&mut self, bytes: &[u8], until: usize) {
        // Coalesce short packets while bounding every worker feed to four seconds.
        // Resampling occurs only when this queued packet reaches the worker, so a
        // snapshot never includes a resampler tail from future queued audio.
        if let Some((last, end)) = self
            .audio
            .back_mut()
            .filter(|(last, _)| last.len() + bytes.len() <= 8 * 24000)
        {
            last.extend_from_slice(bytes);
            *end = until;
        } else {
            self.audio.push_back((bytes.to_vec(), until));
        }
    }
    pub fn prepare_feed(&mut self) -> Option<FeedFuture> {
        let (bytes, until) = self.audio.pop_front()?;
        let samples = self.resampler.push(&bytes);
        let stream = self.stream.clone();
        Some(Box::pin(async move {
            Ok((
                stream.feed(samples?).await.map_err(|e| e.to_string())?,
                until,
            ))
        }))
    }
    pub fn finish_feed(
        &mut self,
        chunk: ProbabilityChunk,
        until: usize,
    ) -> Result<Option<Value>, String> {
        let (offset, probabilities) = chunk;
        if offset != self.history_start + self.history.len() {
            return Err("Diarization activity timeline is discontinuous".into());
        }
        self.fed_until = until;
        let event = (!probabilities.is_empty()).then(|| {
            let mut segments = diarization::segments(&probabilities, offset, self.settings.options());
            for segment in &mut segments { segment.start += self.epoch as f64 / 24000.; segment.end += self.epoch as f64 / 24000.; }
            json!({"type":"input_audio_buffer.diarization.delta","diarization_id":self.id,"start":self.epoch as f64/24000.+offset as f64*0.01,"end":self.epoch as f64/24000.+(offset+probabilities.len()) as f64*0.01,"segments":segments})
        });
        self.history.extend(probabilities);
        // Enough for 60 seconds of backpressure plus one 30-second ASR window.
        while self.history.len() > 90 * 100 {
            self.history.pop_front();
            self.history_start += 1;
        }
        Ok(event)
    }
    pub fn enqueue(&mut self, annotation: Annotation) -> Result<(), String> {
        if self.annotations.len() >= 16
            || self.annotations.iter().map(|a| a.pcm.len()).sum::<usize>() + annotation.pcm.len()
                > 60 * 24000
        {
            return Err(
                "Realtime speaker annotation backlog exceeded; send audio at recording cadence"
                    .into(),
            );
        }
        self.annotations.push_back(annotation);
        Ok(())
    }
    pub fn prepare_annotation(&mut self, pool: &EnginePoolState) -> Option<AnnotationFuture> {
        let first = self.annotations.front()?;
        if first.start_sample + first.pcm.len() > self.fed_until {
            return None;
        }
        let annotation = self.annotations.pop_front().unwrap();
        let start_frame = (annotation.start_sample - self.epoch) / 240;
        let end_frame = (annotation.start_sample - self.epoch + annotation.pcm.len()).div_ceil(240);
        let from = start_frame.max(self.history_start);
        let to = end_frame
            .min(self.history_start + self.history.len())
            .max(from);
        let mut probabilities = self
            .history
            .iter()
            .skip(from - self.history_start)
            .take(to - from)
            .copied()
            .collect::<Vec<_>>();
        let tail = self.resampler.tail();
        let stream = self.stream.clone();
        let pool = pool.clone();
        let epoch = self.epoch;
        let id = self.id.clone();
        let options = self.settings.options();
        let input_memory = self.input_memory.clone();
        Some(Box::pin(async move {
            if start_frame < from {
                return Err("Realtime speaker history expired before word alignment".into());
            }
            if to < end_frame {
                let (offset, snapshot) = stream.snapshot(tail?).await.map_err(|e| e.to_string())?;
                if offset > to {
                    return Err("Realtime speaker snapshot has a timeline gap".into());
                }
                probabilities.extend(snapshot.into_iter().skip(to - offset).take(end_frame - to));
            }
            let mut segments = diarization::segments(&probabilities, start_frame, options);
            let duration = annotation.pcm.len() as f64 / 24000.;
            let relative_start = (annotation.start_sample - epoch) as f64 / 24000.;
            for segment in &mut segments {
                segment.start = (segment.start - relative_start).max(0.);
                segment.end = (segment.end - relative_start).min(duration);
            }
            let activity = Diarization {
                duration,
                num_speakers: segments
                    .iter()
                    .map(|s| &s.speaker)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                segments,
            };
            let words = if annotation.text.trim().is_empty() {
                Vec::new()
            } else {
                let (_, lease) = pool
                    .resolve_engine(Some(FIXED_ALIGNER))
                    .await
                    .map_err(|e| e.to_string())?;
                let EngineVariant::Alignment(runtime) = lease.engine() else {
                    return Err("Forced aligner task mismatch".into());
                };
                let runtime = runtime.clone();
                runtime
                    .align(
                        AlignmentRequest {
                            pcm: PcmBuffer {
                                format: PcmFormat {
                                    sample_rate: 24000,
                                    channels: 1,
                                },
                                samples: annotation.pcm,
                            },
                            text: annotation.text.clone(),
                            language: annotation.language.clone(),
                            input_memory: Some(input_memory),
                            input_slot: None,
                        },
                        lease,
                    )
                    .await
                    .map_err(|e| e.to_string())?
                    .words
            };
            let mut result = diarizations::attribute(&annotation.text, &words, duration, &activity)
                .map_err(|e| e.to_string())?;
            let base = annotation.start_sample as f64 / 24000.;
            for segment in result["segments"].as_array_mut().unwrap() {
                for key in ["start", "end"] {
                    segment[key] = (segment[key].as_f64().unwrap() + base).into();
                }
            }
            for segment in result["diarization"]["segments"].as_array_mut().unwrap() {
                for key in ["start", "end"] {
                    segment[key] = (segment[key].as_f64().unwrap() + base).into();
                }
            }
            Ok(
                json!({"type":"conversation.item.input_audio_transcription.diarized","item_id":annotation.item,"content_index":0,"diarization_id":id,"transcript":annotation.text,"language":annotation.language,"is_final":annotation.is_final,"start":base,"end":base+duration,"segments":result["segments"],"diarization":result["diarization"]}),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packet_boundaries_do_not_change_sinc_resampling_or_snapshots() {
        let bytes = (0..24013)
            .flat_map(|i| (((i as f32 * 0.13).sin() * 24000.) as i16).to_le_bytes())
            .collect::<Vec<_>>();
        let pcm = bytes
            .chunks_exact(2)
            .map(|v| f32::from(i16::from_le_bytes([v[0], v[1]])) / 32768.)
            .collect::<Vec<_>>();
        let expected = ironmlx_audio::signal::resample_mono(&pcm, 24000, 16000, &NoCancel).unwrap();
        for size in [2, 74, 1920, 48000] {
            let mut resampler = Resampler::default();
            let mut output = Vec::new();
            for bytes in bytes.chunks(size) {
                output.extend(resampler.push(bytes).unwrap());
                assert!(resampler.tail().unwrap().len() <= 10);
            }
            output.extend(resampler.tail().unwrap());
            assert_eq!(output, expected, "packet size {size}");
        }
    }
}
