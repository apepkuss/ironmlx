//! OpenAI-compatible multipart audio transcription on the native ASR worker.
use super::api_error::{ApiError, ApiProtocol};
use axum::{
    body::{to_bytes, Body},
    extract::{FromRequest, Multipart, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use ironmlx_audio::{AudioIo, DecodeLimits};
use ironmlx_runtime::core::{
    asr_execution::{AsrExecutionError, AsrRequest},
    engine_pool::{EnginePoolState, EngineVariant},
    process_memory::global_process_memory_governor,
};
use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::Semaphore;

pub(super) const MAX_ASR_REQUEST_BYTES: usize = 129 * 1024 * 1024;

fn input_slots() -> Arc<Semaphore> {
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(Semaphore::new(5))).clone()
}

pub(super) async fn transcriptions(
    State(pool): State<EnginePoolState>,
    request: Request,
) -> Response {
    transcriptions_with_pool(pool, request).await
}
pub(super) async fn transcriptions_with_pool(pool: EnginePoolState, request: Request) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let mut response = match execute(pool, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(ApiProtocol::OpenAi),
    };
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).expect("UUID"),
    );
    response
}
async fn execute(pool: EnginePoolState, request: Request) -> Result<Response, ApiError> {
    let slot = input_slots().try_acquire_owned().map_err(|_| {
        ApiError::service_unavailable("asr_input_queue_full", "ASR input queue is full")
    })?;
    let governor = global_process_memory_governor();
    governor.sample_process();
    let input_memory = governor
        .try_reserve(1536 * 1024 * 1024, "ASR input")
        .map_err(|e| ApiError::service_unavailable("asr_memory_unavailable", e.to_string()))?;
    let (parts, body) = request.into_parts();
    let bytes = tokio::time::timeout(
        Duration::from_secs(60),
        to_bytes(body, MAX_ASR_REQUEST_BYTES),
    )
    .await
    .map_err(|_| {
        ApiError::from_status(
            StatusCode::REQUEST_TIMEOUT,
            "asr_input_timeout",
            "Audio upload timed out",
        )
    })?
    .map_err(|_| {
        ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "Audio request exceeds 129 MiB",
        )
    })?;
    let request = Request::from_parts(parts, Body::from(bytes));
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|e| ApiError::invalid_request("invalid_multipart", e.to_string()))?;
    let mut seen = HashSet::new();
    let mut file = None;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut format = "json".to_string();
    let mut stream = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::invalid_request("invalid_multipart", e.to_string()))?
    {
        let name = field
            .name()
            .ok_or_else(|| {
                ApiError::invalid_request("invalid_multipart", "Multipart field must have a name")
            })?
            .to_string();
        if !seen.insert(name.clone()) {
            return Err(ApiError::invalid_request(
                "duplicate_parameter",
                format!("Duplicate field: {name}"),
            ));
        }
        if name == "file" {
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::invalid_request("invalid_audio_input", e.to_string()))?;
            if data.is_empty() || data.len() > 128 * 1024 * 1024 {
                return Err(ApiError::invalid_request(
                    "invalid_audio_input",
                    "Audio file must be nonempty and at most 128 MiB",
                ));
            }
            file = Some(data);
            continue;
        }
        let data = field
            .bytes()
            .await
            .map_err(|e| ApiError::invalid_request("invalid_multipart", e.to_string()))?;
        if data.len() > 16384 {
            return Err(ApiError::invalid_request(
                "invalid_value",
                format!("Field {name} exceeds 16384 bytes"),
            ));
        }
        let value = std::str::from_utf8(&data)
            .map_err(|_| {
                ApiError::invalid_request("invalid_value", format!("Field {name} is not UTF-8"))
            })?
            .to_owned();
        match name.as_str() {
            "model" => model = Some(value),
            "language" => language = Some(value),
            "prompt" => prompt = Some(value),
            "response_format" => format = value,
            "stream" if value == "false" => {}
            "stream" if value == "true" => stream = true,
            "temperature" if value.parse::<f32>().ok() == Some(0.) => {}
            _ => {
                return Err(ApiError::invalid_request(
                    "unsupported_parameter",
                    format!("Unsupported ASR parameter or value: {name}"),
                )
                .with_param(Some(name)))
            }
        }
    }
    if !matches!(format.as_str(), "json" | "text") {
        return Err(ApiError::invalid_request(
            "unsupported_response_format",
            "ASR supports json and text output",
        )
        .with_param(Some("response_format".into())));
    }
    if stream && format != "json" {
        return Err(ApiError::invalid_request(
            "unsupported_response_format",
            "Streaming transcription requires response_format=json",
        ));
    }
    let model = model.filter(|s| !s.trim().is_empty()).ok_or_else(|| {
        ApiError::invalid_request("missing_required_parameter", "model is required")
    })?;
    let file = file.ok_or_else(|| {
        ApiError::invalid_request("missing_required_parameter", "file is required")
    })?;
    if language.as_ref().is_some_and(|s| s.is_empty()) {
        return Err(ApiError::invalid_request(
            "invalid_value",
            "language must be nonempty",
        ));
    }
    if !pool
        .is_asr_model(&model)
        .await
        .map_err(ApiError::engine_resolution)?
    {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "The selected model does not support audio transcription",
        )
        .with_param(Some("model".into())));
    }
    let settings = pool
        .asr_settings(&model)
        .await
        .map_err(|e| ApiError::internal("asr_settings_failed", e.to_string()))?;
    let (pcm, input_memory, slot) = tokio::task::spawn_blocking(move || {
        let limits = DecodeLimits {
            max_duration_seconds: settings.max_audio_duration_seconds,
            max_encoded_bytes: 128 * 1024 * 1024,
            max_pcm_bytes: 512 * 1024 * 1024,
            ..DecodeLimits::default()
        };
        let pcm = ironmlx_audio::io::NativeAudioIo.decode(&file, &limits)?;
        if pcm.frames()? as u64 * 80 <= u64::from(pcm.format.sample_rate) {
            return Err(ironmlx_audio::AudioError::InvalidInput {
                field: "file",
                reason: "ASR audio must exceed 12.5 ms".into(),
            });
        }
        Ok::<_, ironmlx_audio::AudioError>((pcm, input_memory, slot))
    })
    .await
    .map_err(|e| ApiError::internal("audio_input_failed", e.to_string()))?
    .map_err(|e| ApiError::invalid_request("invalid_audio_input", e.to_string()))?;
    let (_, lease) = pool
        .resolve_engine(Some(&model))
        .await
        .map_err(ApiError::engine_resolution)?;
    let EngineVariant::Asr(runtime) = lease.engine() else {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "The selected model does not support ASR",
        ));
    };
    let runtime = runtime.clone();
    let (events, receiver) = tokio::sync::mpsc::channel(16);
    let request = AsrRequest {
        pcm,
        language,
        prompt,
        max_tokens: settings.max_output_tokens,
        streaming_context: None,
        progress: stream.then(|| events.clone()),
        input_memory: Some(Arc::new(input_memory)),
        input_slot: Some(Arc::new(slot)),
    };
    if stream {
        let (output, result) = tokio::sync::mpsc::channel::<
            Result<axum::response::sse::Event, std::convert::Infallible>,
        >(16);
        tokio::spawn(async move {
            let mut receiver = receiver;
            let inference = runtime.transcribe(request, lease);
            tokio::pin!(inference);
            loop {
                tokio::select! {
                    biased;
                    _ = output.closed() => break,
                    result = &mut inference => {
                        while let Ok(delta) = receiver.try_recv() {
                            if output.send(Ok(transcript_event("transcript.text.delta", serde_json::json!({"delta":delta})))).await.is_err() { return; }
                        }
                        let event = match result {
                            Ok(result) => transcript_event("transcript.text.done", serde_json::json!({"text":result.text})),
                            Err(error) => transcript_event("error", serde_json::json!({"error":{"message":error.to_string(),"type":"server_error"}})),
                        };
                        let _ = output.send(Ok(event)).await;
                        break;
                    }
                    Some(delta) = receiver.recv() => {
                        if output.send(Ok(transcript_event("transcript.text.delta", serde_json::json!({"delta":delta})))).await.is_err() { break; }
                    }
                }
            }
        });
        return Ok(
            axum::response::Sse::new(tokio_stream::wrappers::ReceiverStream::new(result))
                .keep_alive(axum::response::sse::KeepAlive::default())
                .into_response(),
        );
    }
    let result = runtime
        .transcribe(request, lease)
        .await
        .map_err(execution_error)?;
    Ok(if format == "text" {
        (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response()
    } else {
        Json(serde_json::json!({"text":result.text})).into_response()
    })
}
fn transcript_event(kind: &str, mut value: serde_json::Value) -> axum::response::sse::Event {
    value["type"] = kind.into();
    axum::response::sse::Event::default()
        .event(kind)
        .data(value.to_string())
}

pub(super) fn execution_error(error: AsrExecutionError) -> ApiError {
    match error {
        AsrExecutionError::InvalidInput(message) => {
            ApiError::invalid_request("invalid_value", message)
        }
        AsrExecutionError::QueueFull
        | AsrExecutionError::WorkerStopped
        | AsrExecutionError::Memory(_) => {
            ApiError::service_unavailable("asr_unavailable", error.to_string())
        }
        AsrExecutionError::WrongEngine => {
            ApiError::invalid_request("model_task_mismatch", error.to_string())
        }
        AsrExecutionError::Timeout => ApiError::from_status(
            StatusCode::GATEWAY_TIMEOUT,
            "asr_timeout",
            error.to_string(),
        ),
        AsrExecutionError::Inference(error) => {
            if let Some(audio) = error.downcast_ref::<ironmlx_audio::AudioError>() {
                if matches!(audio, ironmlx_audio::AudioError::GenerationLimitExceeded) {
                    return ApiError::invalid_request("asr_output_limit_exceeded",
                        "ASR output reached this model's configured token budget; increase max_output_tokens");
                }
                if matches!(audio, ironmlx_audio::AudioError::DeadlineExceeded) {
                    return ApiError::from_status(
                        StatusCode::GATEWAY_TIMEOUT,
                        "asr_timeout",
                        error.to_string(),
                    );
                }
            }
            ApiError::internal("asr_inference_failed", format!("{error:#}"))
        }
    }
}
