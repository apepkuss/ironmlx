//! Request-local fault injection; all response loops and wire formatting remain real.
use super::*;
use crate::core::generate::GenerateEvent;
use crate::core::scheduler::{RequestId, StepEvent};
use crate::core::scheduler_autotune::*;
use crate::core::server::scheduler_actor::tests::SchedulerActorFakeModel;
use tower::ServiceExt;

pub(super) enum FaultInjectableGenerationStream<'a, M: Model> {
    Real(Box<crate::core::generate::GenerationStream<'a, M>>),
    Injected(std::vec::IntoIter<GenerateEvent>),
}

impl<'a, M: Model + DenseVlMethods> FaultInjectableGenerationStream<'a, M> {
    pub(super) fn new(
        model: &'a M,
        tokenizer: &'a crate::core::Tokenizer,
        request: GenerateRequest,
        events: Option<Vec<GenerateEvent>>,
    ) -> anyhow::Result<Self> {
        match events {
            Some(events) => Ok(Self::Injected(events.into_iter())),
            None => crate::core::generate::GenerationStream::new(model, tokenizer, request)
                .map(Box::new)
                .map(Self::Real),
        }
    }

    pub(super) fn next_token(&mut self) -> anyhow::Result<Option<GenerateEvent>> {
        match self {
            Self::Real(stream) => stream.next_token(),
            Self::Injected(events) => Ok(events.next()),
        }
    }
}

pub(super) fn closed_scheduler_stream(events: Vec<GenerateEvent>) -> AdmitReply {
    let (tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let request_id = RequestId(1);
    for event in events {
        tx.send(StepEvent {
            id: request_id,
            token: event.token,
            finish_reason: event.finish_reason,
        })
        .unwrap();
    }
    // Closing the actual receiver is the fault, including after partial output.
    drop(tx);
    AdmitReply {
        request_id,
        event_rx,
    }
}

async fn state() -> AppState<SchedulerActorFakeModel> {
    let dir = std::env::temp_dir().join(format!("ironmlx-eof-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let tokenizer_path = dir.join("tokenizer.json");
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("hello".into(), 1),
                ("world".into(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    tokenizers::Tokenizer::new(model)
        .save(&tokenizer_path, false)
        .unwrap();
    let config = serde_json::from_value(serde_json::json!({"chat_template":"hello"})).unwrap();
    let tokenizer = crate::core::Tokenizer::from_files(&tokenizer_path, &config).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let profile = SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "eof-test".into(),
        hardware_label: "test".into(),
        runtime_context: SchedulerAutotuneRuntimeContext::local_default(256),
        config: SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 0,
            admission_deadline_ms: 0,
            admission_queue_max: 1,
            max_cache_cap: 256,
            decode_cadence_mid_chunk_cap: 256,
        },
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
    };
    super::super::build_plain_app_state(
        SchedulerActorFakeModel,
        tokenizer,
        "eof-test".into(),
        0,
        1,
        0,
        1,
        256,
        256,
        None,
        profile,
        false,
        None,
        None,
        None,
        Default::default(),
        crate::core::cache::active_kv::ActiveKvOffloadConfig::disabled(),
    )
    .await
    .unwrap()
}

async fn response(
    state: AppState<SchedulerActorFakeModel>,
    scheduler: bool,
    stream: bool,
    count: usize,
    terminal: Option<&'static str>,
) -> Response {
    // POST exercises JSON extraction, normalization, preparation, the production
    // dispatch and complete response body. Only the event producer is replaced.
    let router = axum::Router::new().route(
        "/v1/responses",
        axum::routing::post(move |Json(request): Json<ResponsesRequest>| {
            let state = state.clone();
            async move {
                let mut prepared =
                    prepare_response(&state, request.normalize().unwrap(), scheduler)
                        .await
                        .unwrap();
                prepared.use_scheduler = scheduler;
                let mut events: Vec<_> = (0..count)
                    .map(|index| GenerateEvent {
                        token: if index == 0 { 1 } else { 2 },
                        text: String::new(),
                        finish_reason: None,
                    })
                    .collect();
                if let Some(reason) = terminal {
                    events.push(GenerateEvent {
                        token: 2,
                        text: String::new(),
                        finish_reason: Some(reason),
                    });
                }
                prepared.injected_events = Some(events);
                serve_prepared_response(state, prepared).await
            }
        }),
    );
    router
        .oneshot(
            axum::http::Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model":"eof-test", "input":"hello", "store":false,
                        "stream":stream, "max_output_tokens":8, "reasoning":{"effort":"none"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
}

fn sse_events(wire: &str) -> Vec<serde_json::Value> {
    wire.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn endpoints_reject_eof_before_or_after_output() {
    let state = state().await;
    for scheduler in [false, true] {
        for count in [0, 2] {
            for stream in [false, true] {
                let response = response(state.clone(), scheduler, stream, count, None).await;
                assert_eq!(
                    response.status(),
                    if stream {
                        StatusCode::OK
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    },
                    "scheduler={scheduler}, count={count}, stream={stream}"
                );
                let content_type = response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .to_owned();
                let bytes = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    axum::body::to_bytes(response.into_body(), 65536),
                )
                .await
                .expect("response must close after EOF")
                .unwrap();
                if stream {
                    assert!(content_type.starts_with("text/event-stream"));
                    let events = sse_events(std::str::from_utf8(&bytes).unwrap());
                    assert_eq!(events.first().unwrap()["type"], "response.created");
                    assert_eq!(events.last().unwrap()["type"], "response.failed");
                    assert_eq!(
                        events
                            .iter()
                            .filter(|e| e["type"] == "response.failed")
                            .count(),
                        1
                    );
                    assert!(!events
                        .iter()
                        .any(|e| e["type"] == "response.completed"
                            || e["type"] == "response.incomplete"));
                    assert!(events
                        .last()
                        .unwrap()
                        .to_string()
                        .contains(MISSING_TERMINAL_EVENT));
                    if count > 0 {
                        assert!(
                            events
                                .iter()
                                .any(|e| e["type"] == "response.output_text.delta"
                                    && !e["delta"].as_str().unwrap().is_empty()),
                            "partial output must really reach the wire: {events:?}"
                        );
                    }
                } else {
                    assert!(content_type.starts_with("application/json"));
                    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(body["error"]["code"], "generation_error");
                    assert_eq!(body["error"]["type"], "server_error");
                    assert_eq!(body["error"]["message"], MISSING_TERMINAL_EVENT);
                    assert!(body.get("output").is_none());
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn endpoints_preserve_stop_and_length_with_terminal_events() {
    let state = state().await;
    for scheduler in [false, true] {
        for (reason, status) in [("stop", "completed"), ("length", "incomplete")] {
            for stream in [false, true] {
                let response = response(state.clone(), scheduler, stream, 2, Some(reason)).await;
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    axum::body::to_bytes(response.into_body(), 65536),
                )
                .await
                .unwrap()
                .unwrap();
                if stream {
                    let events = sse_events(std::str::from_utf8(&bytes).unwrap());
                    assert_eq!(events.last().unwrap()["type"], format!("response.{status}"));
                    assert!(!events.iter().any(|e| e["type"] == "response.failed"));
                } else {
                    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(body["status"], status);
                    assert!(body.get("error").is_none_or(serde_json::Value::is_null));
                }
            }
        }
    }
}
