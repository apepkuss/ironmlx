//! Text, image and audio embeddings over the managed model pool.
use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine as _;
use ironmlx_runtime::core::{
    embedding_execution::{
        EmbeddingExecutionError, EmbeddingInput, EmbeddingItem, EmbeddingPart, EmbeddingRequest,
    },
    engine_pool::{EnginePoolState, EngineVariant},
};
use serde_json::json;

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error":{"message":message.to_string(),"type":if status.is_client_error(){"invalid_request_error"}else{"server_error"},"param":null,"code":null}}))).into_response()
}
pub(super) async fn embeddings(
    State(pool): State<EnginePoolState>,
    body: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> Response {
    embeddings_with_pool(pool, body).await
}
pub(super) async fn embeddings_with_pool(
    pool: EnginePoolState,
    body: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> Response {
    let mut request = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return error(
                if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    StatusCode::PAYLOAD_TOO_LARGE
                } else {
                    StatusCode::BAD_REQUEST
                },
                e.body_text(),
            )
        }
    };
    if let Err(e) = request.validate() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    // Decoding is CPU work with bounded concurrency. A cancelled caller cannot
    // release the admission permit until its blocking decoder actually finishes.
    static DECODE_SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    let permit = match DECODE_SLOTS
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(4)))
        .clone()
        .try_acquire_owned()
    {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "embedding input decoders are busy",
            )
        }
    };
    let input = request.input;
    let samples = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        decode_input(input)
    })
    .await
    {
        Ok(Ok(samples)) => samples,
        Ok(Err((status, message))) => return error(status, message),
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedding input decoder stopped",
            )
        }
    };
    request.input = EmbeddingInput::Prepared(samples);
    if !pool.is_embedding_model(&request.model).await {
        return error(
            StatusCode::BAD_REQUEST,
            "model is not registered as an embedding model",
        );
    }
    let (model, lease) = match pool.resolve_engine(Some(&request.model)).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let EngineVariant::Embedding(runtime) = lease.engine() else {
        return error(StatusCode::BAD_REQUEST, "model does not support embeddings");
    };
    let runtime = runtime.clone();
    let encoded = request.encoding_format.as_deref() == Some("base64");
    match runtime.encode(request, lease).await {
        Ok(output) => {
            let data: Vec<_> = output
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| {
                    let embedding = if encoded {
                        let bytes: Vec<u8> =
                            embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
                        json!(base64::engine::general_purpose::STANDARD.encode(bytes))
                    } else {
                        json!(embedding)
                    };
                    json!({"object":"embedding","index":index,"embedding":embedding})
                })
                .collect();
            Json(json!({"object":"list","data":data,"model":model,"usage":{"prompt_tokens":output.input_tokens,"total_tokens":output.input_tokens}})).into_response()
        }
        Err(e) => {
            let status = match &e {
                EmbeddingExecutionError::InvalidInput(_) | EmbeddingExecutionError::WrongEngine => {
                    StatusCode::BAD_REQUEST
                }
                EmbeddingExecutionError::QueueFull => StatusCode::TOO_MANY_REQUESTS,
                EmbeddingExecutionError::WorkerStopped => StatusCode::SERVICE_UNAVAILABLE,
                EmbeddingExecutionError::Timeout => StatusCode::GATEWAY_TIMEOUT,
                EmbeddingExecutionError::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            error(status, e)
        }
    }
}

fn decode_input(
    input: EmbeddingInput,
) -> Result<Vec<ironmlx_lm::models::embedding_gemma2::EmbeddingSample>, (StatusCode, String)> {
    let mut budget = super::image_input::ImageRequestBudget::default();
    let mut samples = Vec::new();
    let mut audio_budget = super::embedding_audio::AudioRequestBudget::default();
    let items = match input.into_items() {
        Ok(items) => items,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e.to_string())),
    };
    for item in items {
        let mut content = Vec::new();
        match item {
            EmbeddingItem::Text(text) => {
                content.push(ironmlx_lm::models::embedding_gemma2::EmbeddingContent::Text(text))
            }
            EmbeddingItem::Document(document) => {
                for part in document.content {
                    match part {
                        EmbeddingPart::Text { text } => content.push(
                            ironmlx_lm::models::embedding_gemma2::EmbeddingContent::Text(text),
                        ),
                        EmbeddingPart::ImageUrl { image_url } => {
                            match budget.decode_data_url(&image_url.url) {
                                Ok(bytes) => content.push(
                                    ironmlx_lm::models::embedding_gemma2::EmbeddingContent::Image(
                                        bytes,
                                    ),
                                ),
                                Err(e) => {
                                    return Err((
                                        super::image_input::image_error_status(e),
                                        format!("{}: {}", e.code(), e.message()),
                                    ))
                                }
                            }
                        }
                        EmbeddingPart::InputAudio { input_audio } => content.push(
                            ironmlx_lm::models::embedding_gemma2::EmbeddingContent::Audio(
                                audio_budget.decode(input_audio)?,
                            ),
                        ),
                    }
                }
            }
        }
        samples.push(ironmlx_lm::models::embedding_gemma2::EmbeddingSample { content });
    }
    Ok(samples)
}
