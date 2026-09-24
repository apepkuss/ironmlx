//! Strict speech transport over the native audio execution lane.
use super::api_error::{ApiError, ApiProtocol};
use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Extension, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use ironmlx_audio::{
    io::{write_pcm_s16le, write_wav, NativeAudioIo},
    AudioError, AudioIo, DecodeLimits, PcmBuffer, PcmFormat,
};
use ironmlx_runtime::core::{
    audio_execution::{
        speech_request, AudioExecutionError, AudioOutput, AudioOutputChunk, AudioResponse,
    },
    engine_pool::{EnginePoolState, EngineVariant},
    process_memory::{global_process_memory_governor, MemoryReservation},
};
use serde::Deserialize;
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::Semaphore;

const INPUT_RESERVATION: usize = 256 * 1024 * 1024;
const MAX_OUTPUT_FRAMES: usize = 600 * 22050;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default)]
    voice: Option<VoiceSelector>,
    #[serde(default = "wav_format")]
    response_format: String,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VoiceSelector {
    Name(String),
    Id { id: String },
}

impl VoiceSelector {
    fn id(&self) -> &str {
        match self {
            Self::Name(value) => value,
            Self::Id { id } => id,
        }
    }
}
fn wav_format() -> String {
    "wav".into()
}
impl SpeechRequest {
    fn validate(&self) -> Result<(), Box<ApiError>> {
        if self.model.trim().is_empty() || self.input.trim().is_empty() {
            return Err(Box::new(ApiError::invalid_request(
                "invalid_audio_request",
                "model and input must be nonempty",
            )));
        }
        let has_reference = self
            .ref_audio
            .as_ref()
            .is_some_and(|value| !value.is_empty());
        let has_voice = self
            .voice
            .as_ref()
            .is_some_and(|value| !value.id().trim().is_empty());
        if has_reference == has_voice {
            return Err(Box::new(ApiError::invalid_request(
                "invalid_audio_reference",
                "provide exactly one of ref_audio or voice",
            )));
        }
        if self.input.len() > 65536 {
            return Err(Box::new(ApiError::from_status(
                StatusCode::PAYLOAD_TOO_LARGE,
                "audio_input_too_large",
                "input exceeds 65536 UTF-8 bytes",
            )));
        }
        if !matches!(
            (self.stream, self.response_format.as_str()),
            (false, "wav") | (true, "pcm")
        ) {
            return Err(Box::new(ApiError::invalid_request("unsupported_audio_response", "use response_format wav for complete output, or stream true with response_format pcm")));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("The selected model does not support this endpoint")]
pub(super) struct ModelTaskMismatch;

pub(super) fn task_mismatch(protocol: ApiProtocol) -> Response {
    ApiError::invalid_request(
        "model_task_mismatch",
        "The selected model does not support this endpoint.",
    )
    .into_response(protocol)
}
fn runtime_error(error: AudioExecutionError) -> ApiError {
    match error {
        AudioExecutionError::WrongEngine => {
            ApiError::invalid_request("model_task_mismatch", error.to_string())
        }
        AudioExecutionError::QueueFull => {
            ApiError::service_unavailable("audio_queue_full", error.to_string())
        }
        AudioExecutionError::Timeout(_) => ApiError::from_status(
            StatusCode::GATEWAY_TIMEOUT,
            "audio_timeout",
            error.to_string(),
        ),
        AudioExecutionError::Memory(_) => {
            ApiError::service_unavailable("audio_memory_unavailable", error.to_string())
        }
        AudioExecutionError::Model(error) => model_error(error),
        _ => ApiError::internal("audio_execution_failed", error.to_string()),
    }
}
fn model_error(error: AudioError) -> ApiError {
    let (status, code) = match &error {
        AudioError::InvalidInput { .. } | AudioError::UnsupportedFormat => {
            (StatusCode::BAD_REQUEST, "invalid_audio_input")
        }
        AudioError::CapacityExceeded { .. } => {
            (StatusCode::PAYLOAD_TOO_LARGE, "audio_input_too_large")
        }
        AudioError::ResourceMissing { .. } => {
            (StatusCode::SERVICE_UNAVAILABLE, "audio_resource_missing")
        }
        AudioError::DeadlineExceeded => (StatusCode::GATEWAY_TIMEOUT, "audio_timeout"),
        AudioError::GenerationLimitExceeded => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_limit_exceeded",
        ),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "audio_inference_failed"),
    };
    ApiError::from_status(status, code, error.to_string())
}
fn input_slots() -> Arc<Semaphore> {
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(Semaphore::new(5))).clone()
}
fn input_execution() -> Arc<Semaphore> {
    static EXECUTION: OnceLock<Arc<Semaphore>> = OnceLock::new();
    EXECUTION
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
}

pub(super) async fn speech(
    State(pool): State<EnginePoolState>,
    Extension(voices): Extension<super::voices::VoiceStore>,
    request: Request,
) -> Response {
    speech_with_pool(pool, voices, request).await
}
pub(super) async fn speech_with_pool(
    pool: EnginePoolState,
    voices: super::voices::VoiceStore,
    request: Request,
) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let mut response = match execute(pool, voices, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(ApiProtocol::OpenAi),
    };
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).expect("UUID"),
    );
    response
}

async fn execute(
    pool: EnginePoolState,
    voices: super::voices::VoiceStore,
    request: Request,
) -> Result<Response, ApiError> {
    // Admission and governor reservation precede reading any body bytes. The
    // common middleware delegates this route's bounded read here.
    let slot = input_slots().try_acquire_owned().map_err(|_| {
        ApiError::service_unavailable(
            "audio_input_queue_full",
            "Audio input preparation queue is full",
        )
    })?;
    let governor = global_process_memory_governor();
    governor.sample_process();
    let input_memory = governor
        .try_reserve(INPUT_RESERVATION, "audio input")
        .map_err(|e| runtime_error(e.into()))?;
    if !request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("application/json"))
        })
    {
        return Err(ApiError::invalid_request(
            "invalid_json",
            "Content-Type must be application/json",
        ));
    }
    let bytes = tokio::time::timeout(
        Duration::from_secs(60),
        to_bytes(request.into_body(), super::security::MAX_REQUEST_BODY_BYTES),
    )
    .await
    .map_err(|_| runtime_error(AudioExecutionError::Timeout("input")))?
    .map_err(|_| {
        ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "The request body exceeds 32 MiB",
        )
    })?;
    let permit = tokio::time::timeout(Duration::from_secs(60), input_execution().acquire_owned())
        .await
        .map_err(|_| runtime_error(AudioExecutionError::Timeout("input queue")))?
        .map_err(|_| ApiError::internal("audio_input_stopped", "Input executor stopped"))?;
    // CPU parsing owns the admission and memory guards even when the HTTP
    // future is dropped. Queued async acquisition above is cancel-safe.
    let (req, slot, permit, input_memory) = tokio::task::spawn_blocking(move || {
        let req: SpeechRequest = serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::invalid_request("invalid_json", e.to_string()))?;
        req.validate()?;
        Ok::<_, Box<ApiError>>((req, slot, permit, input_memory))
    })
    .await
    .map_err(|_| ApiError::internal("audio_input_failed", "Input worker failed"))?
    .map_err(|error| *error)?;
    let (reference, slot, permit, input_memory) = if let Some(encoded) = req.ref_audio {
        tokio::task::spawn_blocking(move || {
            NativeAudioIo
                .decode_base64(&encoded, &DecodeLimits::default())
                .map(|reference| (reference, slot, permit, input_memory))
        })
        .await
        .map_err(|_| ApiError::internal("audio_input_failed", "Input worker failed"))?
        .map_err(model_error)?
    } else {
        let voice = req.voice.expect("validated voice reference");
        let bytes = voices
            .reference_bytes(voice.id().trim())
            .await
            .map_err(|error| error.api_error())?;
        tokio::task::spawn_blocking(move || {
            NativeAudioIo
                .decode(&bytes, &DecodeLimits::default())
                .map(|reference| (reference, slot, permit, input_memory))
        })
        .await
        .map_err(|_| ApiError::internal("audio_input_failed", "Input worker failed"))?
        .map_err(model_error)?
    };
    // Input guards stay live through JSON parsing, profile lookup and audio decoding.
    let _guards = (slot, permit);
    let model = req.model;
    let stream = req.stream;
    let request = speech_request(req.input, reference, stream);
    if !pool
        .is_audio_model(Some(&model))
        .await
        .map_err(ApiError::engine_resolution)?
    {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "The selected model is not an audio model",
        ));
    }
    // Detached load survives disconnect/timeout, keeping EnginePool's load gate,
    // snapshot lease and reservation alive until the loader actually exits.
    let load = tokio::spawn(async move { pool.resolve_engine(Some(&model)).await });
    let (_, lease) = tokio::time::timeout(Duration::from_secs(300), load)
        .await
        .map_err(|_| runtime_error(AudioExecutionError::Timeout("load")))?
        .map_err(|_| ApiError::internal("audio_load_failed", "Audio loader task failed"))?
        .map_err(|e| {
            if let Some(audio) = e.downcast_ref::<AudioError>() {
                match audio {
                    AudioError::ResourceMissing { .. } => {
                        return ApiError::service_unavailable(
                            "audio_resource_missing",
                            audio.to_string(),
                        )
                    }
                    AudioError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
                        return ApiError::service_unavailable(
                            "audio_resource_missing",
                            audio.to_string(),
                        )
                    }
                    _ => return ApiError::internal("audio_load_failed", audio.to_string()),
                }
            }
            ApiError::engine_resolution(e)
        })?;
    let EngineVariant::Audio(runtime) = lease.engine() else {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "Selected model is not audio",
        ));
    };
    let runtime = runtime.clone();
    let mut output = runtime
        .submit(request, input_memory, lease)
        .await
        .map_err(runtime_error)?;
    if stream {
        let first = match output.next().await.map_err(runtime_error)? {
            Some(AudioOutput::Chunk(chunk)) => encode_chunk(chunk).map_err(model_error)?,
            _ => {
                return Err(ApiError::internal(
                    "empty_audio_output",
                    "Model finished without playable audio",
                ))
            }
        };
        let chunks = futures::stream::unfold(
            (Some(first), output, false),
            |(first, mut output, done)| async move {
                if done {
                    return None;
                }
                if let Some(first) = first {
                    return Some((Ok::<Bytes, std::io::Error>(first), (None, output, false)));
                }
                match output.next().await {
                    Ok(Some(AudioOutput::Chunk(chunk))) => match encode_chunk(chunk) {
                        Ok(bytes) => Some((Ok(bytes), (None, output, false))),
                        Err(error) => Some((
                            Err(std::io::Error::other(error.to_string())),
                            (None, output, true),
                        )),
                    },
                    Ok(Some(AudioOutput::Finished(_))) => None,
                    Ok(None) => Some((
                        Err(std::io::Error::other("missing audio completion")),
                        (None, output, true),
                    )),
                    Err(error) => Some((
                        Err(std::io::Error::other(error.to_string())),
                        (None, output, true),
                    )),
                }
            },
        );
        Ok(audio_headers(
            Response::new(Body::from_stream(chunks)),
            true,
        ))
    } else {
        collect_wav(output).await
    }
}

/// Bytes retains the reservation through hyper's consumption, cloning and drop.
struct ReservedBytes {
    bytes: Vec<u8>,
    _memory: MemoryReservation,
}
impl AsRef<[u8]> for ReservedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
fn encode_chunk(output: AudioOutputChunk) -> ironmlx_audio::Result<Bytes> {
    let mut bytes = Vec::with_capacity(output.chunk.pcm.samples.len() * 2);
    write_pcm_s16le(&output.chunk.pcm, &mut bytes)?;
    Ok(Bytes::from_owner(ReservedBytes {
        bytes,
        _memory: output.reservation,
    }))
}
async fn collect_wav(mut output: AudioResponse) -> Result<Response, ApiError> {
    let governor = global_process_memory_governor();
    governor.sample_process();
    let memory = governor
        .try_reserve(MAX_OUTPUT_FRAMES * 6 + 44, "audio complete response")
        .map_err(|e| runtime_error(e.into()))?;
    let mut pcm = PcmBuffer {
        format: PcmFormat {
            sample_rate: 22050,
            channels: 1,
        },
        samples: Vec::with_capacity(MAX_OUTPUT_FRAMES),
    };
    loop {
        match output.next().await.map_err(runtime_error)? {
            Some(AudioOutput::Chunk(chunk)) => {
                if chunk.chunk.pcm.samples.len() > MAX_OUTPUT_FRAMES - pcm.samples.len() {
                    return Err(model_error(AudioError::GenerationLimitExceeded));
                }
                pcm.samples.extend_from_slice(&chunk.chunk.pcm.samples);
            }
            Some(AudioOutput::Finished(summary))
                if summary.total_frames as usize == pcm.samples.len()
                    && !pcm.samples.is_empty() =>
            {
                break
            }
            _ => {
                return Err(ApiError::internal(
                    "invalid_audio_completion",
                    "Missing or inconsistent audio completion",
                ))
            }
        }
    }
    let mut bytes = Vec::with_capacity(pcm.samples.len() * 2 + 44);
    write_wav(&pcm, &mut bytes).map_err(model_error)?;
    let length = bytes.len();
    let mut response = audio_headers(
        Response::new(Body::from(Bytes::from_owner(ReservedBytes {
            bytes,
            _memory: memory,
        }))),
        false,
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    Ok(response)
}
fn audio_headers(mut response: Response, stream: bool) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(if stream { "audio/pcm" } else { "audio/wav" }),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(if stream {
            "no-store, no-transform"
        } else {
            "no-store"
        }),
    );
    headers.insert("x-audio-sample-rate", HeaderValue::from_static("22050"));
    headers.insert("x-audio-channels", HeaderValue::from_static("1"));
    headers.insert("x-audio-sample-format", HeaderValue::from_static("s16le"));
    if stream {
        headers.insert(
            "x-ironmlx-streaming-granularity",
            HeaderValue::from_static("segment"),
        );
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_contract_rejects_ambiguous_or_unknown_fields() {
        let ambiguous: SpeechRequest =
            serde_json::from_str(r#"{"model":"m","input":"x","ref_audio":"a","voice":"x"}"#)
                .expect("ambiguous request parses before validation");
        assert!(ambiguous.validate().is_err());
        for invalid in [
            r#"{"model":"m","model":"n","input":"x","ref_audio":"a"}"#,
            r#"{"model":"m","input":"x","ref_audio":"a","stream":null}"#,
            r#"{"model":"m","input":"x","ref_audio":"a","response_format":null}"#,
        ] {
            assert!(
                serde_json::from_str::<SpeechRequest>(invalid).is_err(),
                "{invalid}"
            );
        }
        for (stream, format, valid) in [
            (false, "wav", true),
            (true, "pcm", true),
            (true, "wav", false),
            (false, "pcm", false),
            (false, "mp3", false),
        ] {
            let req = SpeechRequest {
                model: "m".into(),
                input: "x".into(),
                ref_audio: Some("a".into()),
                voice: None,
                stream,
                response_format: format.into(),
            };
            assert_eq!(req.validate().is_ok(), valid);
        }
        for voice in [r#""speaker_a""#, r#"{"id":"speaker_a"}"#] {
            let request =
                format!(r#"{{"model":"m","input":"x","voice":{voice},"response_format":"wav"}}"#);
            let request: SpeechRequest = serde_json::from_str(&request).expect("voice request");
            assert!(request.validate().is_ok());
        }
    }
}
