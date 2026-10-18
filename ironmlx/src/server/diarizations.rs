//! Independent diarization and word-time speaker attribution.
use super::api_error::{ApiError, ApiProtocol};
use axum::{
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use ironmlx_audio::{
    diarization::{self, Diarization, Options},
    PcmBuffer,
};
use ironmlx_runtime::core::{
    diarization_execution::{DiarizationExecutionError, DiarizationRequest},
    engine_pool::{EnginePoolState, EngineVariant},
};
use std::sync::Arc;

fn execution_error(error: DiarizationExecutionError) -> ApiError {
    match error {
        DiarizationExecutionError::WrongEngine => {
            ApiError::invalid_request("model_task_mismatch", error.to_string())
        }
        DiarizationExecutionError::Timeout => ApiError::from_status(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            "diarization_timeout",
            error.to_string(),
        ),
        DiarizationExecutionError::Inference(error) => {
            if matches!(
                error.downcast_ref::<ironmlx_audio::AudioError>(),
                Some(ironmlx_audio::AudioError::DeadlineExceeded)
            ) {
                return ApiError::from_status(
                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                    "diarization_timeout",
                    error.to_string(),
                );
            }
            ApiError::internal("diarization_inference_failed", format!("{error:#}"))
        }
        _ => ApiError::service_unavailable("diarization_unavailable", error.to_string()),
    }
}

pub(super) struct InputReservation {
    pub memory: Arc<ironmlx_runtime::core::process_memory::MemoryReservation>,
    pub slot: Arc<tokio::sync::OwnedSemaphorePermit>,
}
pub(super) async fn infer(
    pool: &EnginePoolState,
    model: &str,
    pcm: PcmBuffer,
    options: Options,
    input: InputReservation,
) -> Result<Diarization, ApiError> {
    let (_, lease) = pool
        .resolve_engine(Some(model))
        .await
        .map_err(ApiError::engine_resolution)?;
    let EngineVariant::Diarization(runtime) = lease.engine() else {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "The selected model does not support speaker diarization",
        ));
    };
    let runtime = runtime.clone();
    runtime
        .diarize(
            DiarizationRequest {
                pcm,
                options,
                progress: None,
                input_memory: Some(input.memory),
                input_slot: Some(input.slot),
            },
            lease,
        )
        .await
        .map_err(execution_error)
}
pub(super) async fn execute(
    pool: EnginePoolState,
    model: String,
    pcm: PcmBuffer,
    options: Options,
    format: String,
    stream: bool,
    input: InputReservation,
) -> Result<Response, ApiError> {
    if !stream {
        let result = infer(&pool, &model, pcm, options, input).await?;
        return Ok(if format == "rttm" {
            (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                result.rttm(),
            )
                .into_response()
        } else {
            Json(result).into_response()
        });
    }
    let (_, lease) = pool
        .resolve_engine(Some(&model))
        .await
        .map_err(ApiError::engine_resolution)?;
    let EngineVariant::Diarization(runtime) = lease.engine() else {
        return Err(ApiError::invalid_request(
            "model_task_mismatch",
            "The selected model does not support speaker diarization",
        ));
    };
    let runtime = runtime.clone();
    let (progress, mut chunks) = tokio::sync::mpsc::channel(2);
    let request = DiarizationRequest {
        pcm,
        options,
        progress: Some(progress),
        input_memory: Some(input.memory),
        input_slot: Some(input.slot),
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<
        Result<axum::response::sse::Event, std::convert::Infallible>,
    >(2);
    tokio::spawn(async move {
        let result = runtime.diarize(request, lease);
        tokio::pin!(result);
        let event = |kind: &str, value: serde_json::Value| {
            axum::response::sse::Event::default()
                .event(kind)
                .data(value.to_string())
        };
        let mut progress_open = true;
        let chunk_event = |offset: usize, probs: Vec<[f32; 8]>| {
            event(
                "diarization.chunk",
                serde_json::json!({"type":"diarization.chunk","start":offset as f64*0.01,"end":(offset+probs.len()) as f64*0.01,"segments":diarization::segments(&probs,offset,options)}),
            )
        };
        loop {
            tokio::select! {
                _=tx.closed()=>break,
                chunk=chunks.recv(), if progress_open=>{
                    if let Some((offset,probs))=chunk {
                        if tx.send(Ok(chunk_event(offset,probs))).await.is_err(){break;}
                    } else {progress_open=false;}
                }
                result=&mut result=>{
                    while let Ok((offset,probs))=chunks.try_recv() {
                        if tx.send(Ok(chunk_event(offset,probs))).await.is_err(){return;}
                    }
                    let (kind,value)=match result {
                        Ok(result)=>("diarization.done",serde_json::json!({"type":"diarization.done","result":result})),
                        Err(e)=>{
                            let response=execution_error(e).into_response(ApiProtocol::OpenAi);
                            let body=axum::body::to_bytes(response.into_body(),65536).await.unwrap_or_default();
                            ("error",serde_json::from_slice(&body).unwrap_or_else(|_|serde_json::json!({"error":{"code":"diarization_failed"}})))
                        }
                    };
                    let _=tx.send(Ok(event(kind,value))).await;
                    break;
                }
            }
        }
    });
    Ok(
        axum::response::Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmlx_audio::qwen3_asr::AlignedWord;
    #[test]
    fn speaker_attribution_keeps_text_and_exposes_overlap_and_unknown() {
        let text = "Hello,\n  world! Next.";
        let words = [
            AlignedWord {
                word: "Hello".into(),
                start: 0.,
                end: 1.,
            },
            AlignedWord {
                word: "world".into(),
                start: 1.,
                end: 2.,
            },
            AlignedWord {
                word: "Next".into(),
                start: 3.,
                end: 4.,
            },
        ];
        let turns = Diarization {
            duration: 4.,
            num_speakers: 2,
            segments: vec![
                diarization::Segment {
                    speaker: "speaker_0".into(),
                    start: 0.,
                    end: 2.,
                },
                diarization::Segment {
                    speaker: "speaker_1".into(),
                    start: 1.,
                    end: 2.,
                },
            ],
        };
        let output = attribute(text, &words, 4., &turns).unwrap();
        let segments = output["segments"].as_array().unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0]["speaker"], "speaker_0");
        assert_eq!(segments[1]["speaker"], "unknown");
        assert_eq!(segments[1]["ambiguous"], true);
        assert_eq!(
            segments[1]["speakers"],
            serde_json::json!(["speaker_0", "speaker_1"])
        );
        assert_eq!(segments[2]["speaker"], "unknown");
        assert_eq!(segments[2]["ambiguous"], false);
        assert_eq!(
            segments
                .iter()
                .map(|s| s["text"].as_str().unwrap())
                .collect::<String>(),
            text
        );
    }
}

/// Preserve ambiguous overlaps instead of claiming voice separation or identity.
pub(super) fn attribute(
    text: &str,
    words: &[ironmlx_audio::qwen3_asr::AlignedWord],
    duration: f64,
    turns: &Diarization,
) -> Result<serde_json::Value, super::subtitles::SubtitleError> {
    let units = super::subtitles::aligned_text(text, words, duration)?;
    let mut output: Vec<serde_json::Value> = Vec::new();
    for (text, start, end) in units {
        let mut speakers = Vec::new();
        for turn in &turns.segments {
            let overlap = end.min(turn.end) - start.max(turn.start);
            if overlap > 0. && overlap >= (end - start) * 0.2 && !speakers.contains(&turn.speaker) {
                speakers.push(turn.speaker.clone());
            }
        }
        speakers.sort();
        let speaker = if speakers.len() == 1 {
            speakers[0].clone()
        } else {
            "unknown".into()
        };
        let ambiguous = speakers.len() > 1;
        if let Some(previous) = output.last_mut().filter(|v| {
            v["speaker"] == speaker
                && v["speakers"] == serde_json::json!(speakers)
                && start - v["end"].as_f64().unwrap_or(start) < 0.6
        }) {
            let joined = format!("{}{}", previous["text"].as_str().unwrap_or(""), text);
            previous["text"] = joined.into();
            previous["end"] = end.into();
        } else {
            output.push(serde_json::json!({"type":"transcript.text.segment","id":format!("seg_{}",output.len()),"speaker":speaker,"speakers":speakers,"ambiguous":ambiguous,"start":start,"end":end,"text":text}));
        }
    }
    Ok(
        serde_json::json!({"task":"transcribe","duration":duration,"text":text,"segments":output,"diarization":turns}),
    )
}
