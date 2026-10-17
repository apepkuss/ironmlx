//! Bounded transcription-only Realtime WebSocket sessions.
use super::api_error::{ApiError, ApiProtocol};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::Response,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ironmlx_audio::{
    qwen3_asr::Transcription,
    vad::{VadSettings, VoiceDetector},
    PcmBuffer, PcmFormat,
};
use ironmlx_runtime::core::{
    asr_execution::{AsrExecutionError, AsrRequest, AsrRuntime, AsrStreamingContext},
    engine_pool::{EngineLease, EnginePoolState, EngineVariant},
    process_memory::global_process_memory_governor,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

const RATE: usize = 24000;
const WINDOW: usize = 30 * RATE;
const STEP: usize = 2 * RATE;
type Inference = Pin<Box<dyn Future<Output = Result<Transcription, AsrExecutionError>> + Send>>;

#[derive(Default, Deserialize)]
pub(super) struct RealtimeQuery {
    model: Option<String>,
    intent: Option<String>,
}

pub(super) async fn realtime(
    State(pool): State<EnginePoolState>,
    Query(query): Query<RealtimeQuery>,
    upgrade: WebSocketUpgrade,
) -> Response {
    realtime_with_pool(pool, query, upgrade).await
}

pub(super) async fn realtime_with_pool(
    pool: EnginePoolState,
    query: RealtimeQuery,
    upgrade: WebSocketUpgrade,
) -> Response {
    if query
        .intent
        .as_deref()
        .is_some_and(|i| i != "transcription")
    {
        return ApiError::invalid_request(
            "unsupported_intent",
            "Only transcription sessions are supported",
        )
        .into_response(ApiProtocol::OpenAi);
    }
    if let Some(model) = &query.model {
        match pool.is_asr_model(model).await {
            Ok(true) => {}
            Ok(false) => {
                return ApiError::invalid_request(
                    "model_task_mismatch",
                    "Realtime transcription requires an ASR model",
                )
                .into_response(ApiProtocol::OpenAi)
            }
            Err(error) => {
                return ApiError::engine_resolution(error).into_response(ApiProtocol::OpenAi)
            }
        }
    }
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    let Ok(slot) = SLOTS
        .get_or_init(|| Arc::new(Semaphore::new(8)))
        .clone()
        .try_acquire_owned()
    else {
        return ApiError::service_unavailable(
            "realtime_capacity",
            "Realtime session capacity reached",
        )
        .into_response(ApiProtocol::OpenAi);
    };
    let governor = global_process_memory_governor();
    governor.sample_process();
    let memory = match governor.try_reserve(32 * 1024 * 1024, "ASR realtime input") {
        Ok(memory) => memory,
        Err(error) => {
            return ApiError::service_unavailable("asr_memory_unavailable", error.to_string())
                .into_response(ApiProtocol::OpenAi)
        }
    };
    upgrade
        .max_message_size(512 * 1024)
        .max_frame_size(512 * 1024)
        .on_upgrade(move |socket| async move {
            let (_slot, _memory) = (slot, memory);
            let _ = run(socket, pool, query.model).await;
        })
}

#[derive(Default, Clone)]
struct Configuration {
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    vad: Option<VadSettings>,
}
impl Configuration {
    fn view(&self, id: &str) -> Value {
        json!({"id":id,"type":"transcription","audio":{"input":{
            "format":{"type":"audio/pcm","rate":24000},
            "transcription":{"model":self.model,"language":self.language,"prompt":self.prompt},
            "turn_detection":self.vad
        }}})
    }
    fn update(&self, event: &Value) -> Result<Self, String> {
        let session = event
            .get("session")
            .and_then(Value::as_object)
            .ok_or("session must be an object")?;
        for key in session.keys() {
            if ![
                "type",
                "audio",
                "input_audio_format",
                "input_audio_transcription",
                "turn_detection",
                "input_audio_noise_reduction",
            ]
            .contains(&key.as_str())
            {
                return Err(format!("Unsupported session field: {key}"));
            }
        }
        if session.get("type").is_some_and(|v| v != "transcription") {
            return Err("Only transcription sessions are supported".into());
        }
        if session
            .get("input_audio_format")
            .is_some_and(|v| v != "pcm16")
        {
            return Err("Expected 24 kHz PCM16 mono little-endian".into());
        }
        let input = if let Some(audio) = session.get("audio") {
            let audio = audio.as_object().ok_or("audio must be an object")?;
            if audio.keys().any(|k| k != "input") {
                return Err("Only audio.input is supported".into());
            }
            let input = audio
                .get("input")
                .and_then(Value::as_object)
                .ok_or("audio.input must be an object")?;
            for key in input.keys() {
                if ![
                    "format",
                    "transcription",
                    "turn_detection",
                    "noise_reduction",
                ]
                .contains(&key.as_str())
                {
                    return Err(format!("Unsupported audio.input field: {key}"));
                }
            }
            if let Some(format) = input.get("format") {
                if format.as_object().is_none_or(|fields| {
                    fields
                        .keys()
                        .any(|key| !["type", "rate"].contains(&key.as_str()))
                }) {
                    return Err("format accepts only type and rate".into());
                }
                if format.get("type").and_then(Value::as_str) != Some("audio/pcm")
                    || format.get("rate").and_then(Value::as_u64) != Some(24000)
                {
                    return Err("Expected audio/pcm at 24000 Hz".into());
                }
            }
            Some(input)
        } else {
            None
        };
        for value in [
            session.get("input_audio_noise_reduction"),
            input.and_then(|i| i.get("noise_reduction")),
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_null() {
                return Err("Noise reduction is unsupported".into());
            }
        }
        let transcription = input
            .and_then(|i| i.get("transcription"))
            .or_else(|| session.get("input_audio_transcription"));
        let mut next = self.clone();
        let nested_vad = input.and_then(|i| i.get("turn_detection"));
        if nested_vad.is_some() && session.contains_key("turn_detection") {
            return Err("Specify turn_detection only once".into());
        }
        if let Some(value) = nested_vad.or_else(|| session.get("turn_detection")) {
            next.vad = if value.is_null() {
                None
            } else {
                let settings: VadSettings =
                    serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
                settings.validate()?;
                Some(settings)
            };
        }
        if let Some(transcription) = transcription {
            let fields = transcription
                .as_object()
                .ok_or("transcription must be an object")?;
            for key in fields.keys() {
                if !["model", "language", "languages", "prompt"].contains(&key.as_str()) {
                    return Err(format!("Unsupported transcription field: {key}"));
                }
            }
            for (key, target) in [
                ("model", &mut next.model),
                ("language", &mut next.language),
                ("prompt", &mut next.prompt),
            ] {
                if let Some(value) = fields.get(key) {
                    *target = if value.is_null() && key != "model" {
                        None
                    } else {
                        Some(
                            value
                                .as_str()
                                .ok_or_else(|| format!("{key} must be a string"))?
                                .to_owned(),
                        )
                    };
                }
            }
            if let Some(languages) = fields.get("languages") {
                let values = languages.as_array().ok_or("languages must be an array")?;
                if fields.contains_key("language") || values.len() > 1 {
                    return Err("Specify at most one language, using language or languages".into());
                }
                next.language = values
                    .first()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or("language must be a string")
                    })
                    .transpose()?;
            }
        }
        if next.model.as_ref().is_none_or(|s| s.is_empty()) {
            return Err("An ASR model is required".into());
        }
        if next.prompt.as_ref().is_some_and(|s| s.len() > 16384) {
            return Err("prompt exceeds 16384 bytes".into());
        }
        if next.language.as_ref().is_some_and(|s| s.is_empty()) {
            return Err("language must be nonempty".into());
        }
        Ok(next)
    }
}

struct Binding {
    runtime: AsrRuntime,
    lease: Arc<EngineLease>,
}
struct Pending {
    future: Inference,
    samples: usize,
    final_segment: bool,
}
struct Turn {
    audio: Vec<f32>,
    processed: usize,
    previous: String,
    updates: usize,
    published: String,
    item: String,
    committed: bool,
    acknowledged: bool,
}
impl Default for Turn {
    fn default() -> Self {
        Self {
            audio: Vec::new(),
            processed: 0,
            previous: String::new(),
            updates: 0,
            published: String::new(),
            item: format!("item_{}", uuid::Uuid::new_v4()),
            committed: false,
            acknowledged: false,
        }
    }
}
impl Turn {
    fn window_end(&self) -> usize {
        // Never move a boundary behind audio already represented by the prefix.
        let start = (28 * RATE).max(self.processed).min(WINDOW);
        let mut end = WINDOW;
        let mut energy = f64::INFINITY;
        for (i, window) in self.audio[start..WINDOW].chunks(RATE / 100).enumerate() {
            let value = window.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            if value <= energy {
                energy = value;
                end = start + i * (RATE / 100) + window.len();
            }
        }
        end
    }

    fn ready(&self) -> bool {
        !self.audio.is_empty()
            && (self.committed
                || self.audio.len().min(WINDOW).saturating_sub(self.processed) >= STEP
                || self.audio.len() >= WINDOW)
    }
    fn append(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.committed {
            return Err("Wait for the committed turn to complete before appending audio".into());
        }
        if bytes.is_empty() || !bytes.len().is_multiple_of(2) || bytes.len() > 8 * RATE {
            return Err(
                "Each append must contain nonempty PCM16 whole samples, at most four seconds"
                    .into(),
            );
        }
        if self.audio.len() + bytes.len() / 2 > 2 * WINDOW {
            return Err("Realtime input is too far ahead of inference; wait for results".into());
        }
        self.audio.extend(
            bytes
                .chunks_exact(2)
                .map(|v| f32::from(i16::from_le_bytes([v[0], v[1]])) / 32768.),
        );
        Ok(())
    }
}

fn vad_buffered_samples(turn: &Turn, queued: &VecDeque<Turn>, detector: &VoiceDetector) -> usize {
    turn.audio.len()
        + queued.iter().map(|t| t.audio.len()).sum::<usize>()
        + detector.buffered_samples()
}

async fn send(socket: &mut WebSocket, mut event: Value) -> Result<(), ()> {
    event["event_id"] = format!("event_{}", uuid::Uuid::new_v4()).into();
    tokio::time::timeout(
        Duration::from_secs(10),
        socket.send(Message::Text(event.to_string())),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}
async fn error(
    socket: &mut WebSocket,
    message: impl Into<String>,
    event_id: Option<&Value>,
) -> Result<(), ()> {
    send(socket, json!({"type":"error","error":{"type":"invalid_request_error","code":"invalid_realtime_request","message":message.into(),"event_id":event_id}})).await
}
async fn acknowledge(
    socket: &mut WebSocket,
    turn: &mut Turn,
    previous: &Option<String>,
) -> Result<(), ()> {
    if !turn.acknowledged {
        send(socket,json!({"type":"input_audio_buffer.committed","item_id":turn.item,"previous_item_id":previous})).await?;
        turn.acknowledged = true;
    }
    Ok(())
}

async fn run(
    mut socket: WebSocket,
    pool: EnginePoolState,
    model: Option<String>,
) -> Result<(), ()> {
    let session_id = format!("sess_{}", uuid::Uuid::new_v4());
    let mut config = Configuration {
        model,
        ..Default::default()
    };
    let mut settings = if let Some(model) = config.model.as_deref() {
        pool.asr_settings(model).await.map_err(|_| ())?
    } else {
        ironmlx_runtime::core::asr_execution::AsrSettings::default()
    };
    let mut binding: Option<Binding> = None;
    let mut turn = Turn::default();
    let mut queued: VecDeque<Turn> = VecDeque::new();
    let mut detector: Option<VoiceDetector> = None;
    let mut vad_item: Option<String> = None;
    let mut previous_item = None;
    let mut pending: Option<Pending> = None;
    let started = Instant::now();
    let mut total_samples = 0usize;
    let mut turn_samples = 0usize;
    send(
        &mut socket,
        json!({"type":"session.created","session":config.view(&session_id)}),
    )
    .await?;
    loop {
        if started.elapsed() > Duration::from_secs(7200) {
            error(
                &mut socket,
                "Session reached its two-hour duration limit",
                None,
            )
            .await?;
            break;
        }
        if pending.is_none() && turn.audio.is_empty() && !queued.is_empty() {
            turn = queued.pop_front().unwrap();
        }
        if pending.is_none() && turn.ready() {
            if binding.is_none() {
                let Some(model) = config.model.as_deref() else {
                    error(
                        &mut socket,
                        "Configure an ASR model before sending audio",
                        None,
                    )
                    .await?;
                    turn = Turn::default();
                    continue;
                };
                match pool.resolve_engine(Some(model)).await {
                    Ok((_, lease)) => {
                        let EngineVariant::Asr(runtime) = lease.engine() else {
                            error(&mut socket, "Selected model does not support ASR", None).await?;
                            if detector.is_some() {
                                return Ok(());
                            }
                            turn = Turn::default();
                            continue;
                        };
                        binding = Some(Binding {
                            runtime: runtime.clone(),
                            lease: Arc::new(lease),
                        });
                    }
                    Err(e) => {
                        error(&mut socket, e.to_string(), None).await?;
                        if detector.is_some() {
                            return Ok(());
                        }
                        turn = Turn::default();
                        continue;
                    }
                }
            }
            let final_segment = turn.audio.len() >= WINDOW;
            let samples = if final_segment {
                // Keep the rolling window bounded while avoiding a cut in a loud syllable.
                turn.window_end()
            } else {
                turn.audio.len()
            };
            let bound = binding.as_ref().unwrap();
            let request = AsrRequest {
                pcm: PcmBuffer {
                    format: PcmFormat {
                        sample_rate: RATE as u32,
                        channels: 1,
                    },
                    samples: turn.audio[..samples].to_vec(),
                },
                language: config.language.clone(),
                prompt: config.prompt.clone(),
                max_tokens: settings.max_output_tokens,
                streaming_context: Some(AsrStreamingContext {
                    raw_text: turn.previous.clone(),
                    update_index: turn.updates,
                }),
                progress: None,
                input_memory: None,
                input_slot: None,
            };
            let runtime = bound.runtime.clone();
            let lease = bound.lease.clone();
            pending = Some(Pending {
                future: Box::pin(async move { runtime.transcribe_leased(request, lease).await }),
                samples,
                final_segment,
            });
        }
        tokio::select! {
            result=async { pending.as_mut().unwrap().future.as_mut().await }, if pending.is_some() => {
                let work=pending.take().unwrap();
                match result {
                    Err(e) => {
                        let (kind,code)=if e.is_output_limit() {("invalid_request_error","asr_output_limit_exceeded")}else{("server_error","asr_inference_failed")};
                        send(&mut socket,json!({"type":"conversation.item.input_audio_transcription.failed","item_id":turn.item,"content_index":0,"error":{"type":kind,"code":code,"message":e.to_string()}})).await?;
                        if vad_item.as_deref()==Some(turn.item.as_str()) {
                            vad_item=None;
                            if let Some(vad)=detector.as_mut() {vad.finish();}
                        }
                        turn=Turn::default();
                        turn_samples=0;
                    }
                    Ok(result) => {
                        let final_now=work.final_segment || (turn.committed && work.samples==turn.audio.len());
                        let target=if final_now { &result.text } else { &result.stable_text };
                        if let Some(delta)=target.strip_prefix(&turn.published).filter(|d| !d.is_empty()) {
                            send(&mut socket,json!({"type":"conversation.item.input_audio_transcription.delta","item_id":turn.item,"content_index":0,"delta":delta})).await?;
                            turn.published=target.clone();
                        }
                        if final_now {
                            acknowledge(&mut socket,&mut turn,&previous_item).await?;
                            let is_final=turn.committed && work.samples==turn.audio.len();
                            send(&mut socket,json!({"type":"conversation.item.input_audio_transcription.completed","item_id":turn.item,"content_index":0,"transcript":result.text,"language":result.language,"is_final":is_final})).await?;
                            previous_item=Some(turn.item.clone());
                            let remaining=turn.audio.split_off(work.samples);
                            let committed=turn.committed && !remaining.is_empty();
                            turn=Turn { audio:remaining, committed, ..Default::default() };
                            if is_final { turn_samples=0; }
                        } else {
                            send(&mut socket,json!({"type":"conversation.item.input_audio_transcription.partial","item_id":turn.item,"content_index":0,"transcript":result.text,"revision":turn.updates+1})).await?;
                            turn.previous=result.raw_text; turn.updates+=1; turn.processed=work.samples;
                        }
                    }
                }
            }
            incoming=tokio::time::timeout(Duration::from_secs(90).min(Duration::from_secs(7200).saturating_sub(started.elapsed())),socket.recv()) => {
                let message=match incoming { Ok(Some(Ok(message))) => message, _ => break };
                let text=match message { Message::Text(text)=>text, Message::Close(_)=>break, Message::Ping(data)=>{ socket.send(Message::Pong(data)).await.map_err(|_| ())?; continue; }, Message::Pong(_)=>continue, _=>{error(&mut socket,"Send JSON events with base64 PCM16 audio",None).await?;continue;} };
                let event:Value=match serde_json::from_str(&text) { Ok(event)=>event,Err(_)=>{error(&mut socket,"Invalid JSON event",None).await?;continue;} };
                let failure: Option<String> = match event.get("type").and_then(Value::as_str).unwrap_or("") {
                    "session.update" | "transcription_session.update" => {
                        if pending.is_some() || !turn.audio.is_empty() || !queued.is_empty() || vad_item.is_some() { Some("Update the session between audio turns".into()) }
                        else { match config.update(&event) {
                            Err(e)=>Some(e),
                            Ok(next)=>{
                                match pool.is_asr_model(next.model.as_deref().unwrap()).await {
                                    Ok(true)=>{
                                        settings=pool.asr_settings(next.model.as_deref().unwrap()).await.map_err(|_| ())?;
                                        turn_samples=0;
                                        detector=next.vad.clone().map(|v| VoiceDetector::new(v,total_samples as u64)).transpose().map_err(|_| ())?;
                                        config=next; binding=None;
                                        let kind=if event["type"]=="transcription_session.update" {"transcription_session.updated"}else{"session.updated"};
                                        send(&mut socket,json!({"type":kind,"session":config.view(&session_id)})).await?; None
                                    },
                                    Ok(false)=>Some("Selected model does not support ASR".into()), Err(e)=>Some(e.to_string()),
                                }
                            }
                        } }
                    },
                    "input_audio_buffer.append" => {
                        if config.model.is_none() { Some("Configure an ASR model before sending audio".into()) }
                        else if let Some(encoded)=event.get("audio").and_then(Value::as_str) {
                            match STANDARD.decode(encoded) {
                                Err(_)=>Some("audio must be canonical base64 PCM16".into()),
                                Ok(bytes)=>{
                                    if total_samples+bytes.len()/2>7200*RATE { Some("Session reached its two-hour audio limit".into()) }
                                    else if bytes.is_empty() || !bytes.len().is_multiple_of(2) || bytes.len()>8*RATE {
                                        Some("Each append must contain nonempty PCM16 whole samples, at most four seconds".into())
                                    }
                                    else if detector.as_ref().is_some_and(|vad| vad_buffered_samples(&turn,&queued,vad)+bytes.len()/2>2*WINDOW) {
                                        Some("Realtime input is too far ahead of inference; wait for results".into())
                                    }
                                    else if detector.is_some() {
                                        total_samples+=bytes.len()/2;
                                        for frame in detector.as_mut().unwrap().push(&bytes) {
                                            if let Some(start)=frame.started {
                                                let next=Turn::default();
                                                let id=next.item.clone();
                                                if turn.audio.is_empty() && pending.is_none() && queued.is_empty() {turn=next;} else {queued.push_back(next);}
                                                vad_item=Some(id.clone());
                                                send(&mut socket,json!({"type":"input_audio_buffer.speech_started","item_id":id,"audio_start_ms":start/24})).await?;
                                            }
                                            let id=vad_item.as_deref().unwrap();
                                            let target=if turn.item==id {&mut turn} else {queued.iter_mut().find(|t|t.item==id).unwrap()};
                                            if target.audio.len()+frame.audio.len()>settings.max_audio_duration_seconds as usize*RATE {
                                                error(&mut socket,"VAD speech turn exceeds this model's configured audio duration",event.get("event_id")).await?;
                                                return Ok(());
                                            }
                                            target.audio.extend(frame.audio.iter().map(|s|f32::from(*s)/32768.));
                                            if let Some(end)=frame.stopped {
                                                target.committed=true;
                                                send(&mut socket,json!({"type":"input_audio_buffer.speech_stopped","item_id":id,"audio_end_ms":end/24})).await?;
                                                let predecessor=if queued.len()>1 {Some(queued[queued.len()-2].item.clone())} else if !queued.is_empty() {Some(turn.item.clone())} else {previous_item.clone()};
                                                let target=if turn.item==id {&mut turn} else {queued.iter_mut().find(|t|t.item==id).unwrap()};
                                                acknowledge(&mut socket,target,&predecessor).await?;
                                                vad_item=None;
                                            }
                                        }
                                        None
                                    }
                                    else if turn_samples+bytes.len()/2>settings.max_audio_duration_seconds as usize*RATE {
                                        Some(format!("Audio turn exceeds this model's configured duration limit of {} seconds",settings.max_audio_duration_seconds))
                                    }
                                    else { match turn.append(&bytes) { Ok(())=>{total_samples+=bytes.len()/2;turn_samples+=bytes.len()/2;None},Err(e)=>Some(e) } }
                                }
                            }
                        } else {Some("audio is required".into())}
                    },
                    "input_audio_buffer.commit" => {
                        if let Some(vad)=detector.as_mut() {
                            if let Some(id)=vad_item.take() {
                                let target=if turn.item==id {&mut turn} else {queued.iter_mut().find(|t|t.item==id).unwrap()};
                                let tail=vad.finish();
                                if target.audio.len()+tail.len()>settings.max_audio_duration_seconds as usize*RATE {
                                    error(&mut socket,"VAD speech turn exceeds this model's configured audio duration",event.get("event_id")).await?;
                                    return Ok(());
                                }
                                target.audio.extend(tail.iter().map(|s|f32::from(*s)/32768.));
                                target.committed=true;
                                send(&mut socket,json!({"type":"input_audio_buffer.speech_stopped","item_id":id,"audio_end_ms":total_samples/24})).await?;
                                let predecessor=if queued.len()>1 {Some(queued[queued.len()-2].item.clone())} else if !queued.is_empty() {Some(turn.item.clone())} else {previous_item.clone()};
                                let target=if turn.item==id {&mut turn} else {queued.iter_mut().find(|t|t.item==id).unwrap()};
                                acknowledge(&mut socket,target,&predecessor).await?;
                                None
                            } else {Some("No active speech turn to commit".into())}
                        }
                        else if turn.audio.len()<RATE/10 {Some("At least 100 ms of audio is required before commit".into())}
                        else if turn.committed {Some("This audio turn is already committed".into())}
                        else { turn.committed=true;acknowledge(&mut socket,&mut turn,&previous_item).await?;None }
                    },
                    "input_audio_buffer.clear" => {
                        pending=None;turn=Turn::default();turn_samples=0;queued.clear();vad_item=None;
                        detector=config.vad.clone().map(|v|VoiceDetector::new(v,total_samples as u64)).transpose().map_err(|_| ())?;
                        send(&mut socket,json!({"type":"input_audio_buffer.cleared"})).await?;None
                    },
                    _=>Some("Unsupported Realtime event".into()),
                };
                if let Some(message)=failure {error(&mut socket,message,event.get("event_id")).await?;}
            }
        }
    }
    // Dropping an in-flight future cancels its worker job; its lease survives GPU synchronization.
    drop(pending);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_configuration_is_atomic_and_rejects_unsupported_processing() {
        let config = Configuration {
            model: Some("local-asr".into()),
            ..Default::default()
        };
        let valid = json!({"session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"language":"zh"},"turn_detection":null}}}});
        assert_eq!(
            config.update(&valid).unwrap().language.as_deref(),
            Some("zh")
        );
        for session in [
            json!({"audio":{"input":{"format":{"type":"audio/pcm","rate":16000}}}}),
            json!({"turn_detection":{"type":"semantic_vad"}}),
            json!({"input_audio_transcription":{"languages":["zh","en"]}}),
            json!({"input_audio_transcription":{"stream":true}}),
            json!({"unknown":1}),
        ] {
            assert!(config.update(&json!({"session":session})).is_err());
        }
        assert!(config.language.is_none());
    }

    #[test]
    fn server_vad_settings_support_both_schemas_and_preserve_atomic_updates() {
        let config = Configuration {
            model: Some("local-asr".into()),
            ..Default::default()
        };
        for session in [
            json!({"turn_detection":{"type":"server_vad"}}),
            json!({"audio":{"input":{"turn_detection":{"type":"server_vad"}}}}),
        ] {
            let next = config.update(&json!({"session":session})).unwrap();
            assert_eq!(next.vad.as_ref().unwrap().silence_duration_ms, 500);
            assert_eq!(
                next.view("id")["audio"]["input"]["turn_detection"]["threshold"],
                0.5
            );
            assert!(next
                .update(&json!({"session":{"turn_detection":null}}))
                .unwrap()
                .vad
                .is_none());
        }
        for settings in [
            json!({"type":"semantic_vad"}),
            json!({"type":"server_vad","threshold":1.1}),
            json!({"type":"server_vad","threshold":-1}),
            json!({"type":"server_vad","prefix_padding_ms":1001}),
            json!({"type":"server_vad","silence_duration_ms":0}),
            json!({"type":"server_vad","create_response":true}),
        ] {
            assert!(config
                .update(&json!({"session":{"turn_detection":settings}}))
                .is_err());
        }
        assert!(config.vad.is_none());
    }

    #[test]
    fn vad_capacity_includes_retained_prefix_and_partial_frame() {
        let mut detector = VoiceDetector::new(
            serde_json::from_value(json!({"type":"server_vad","prefix_padding_ms":1000})).unwrap(),
            0,
        )
        .unwrap();
        assert!(detector.push(&vec![0; 2 * RATE]).is_empty());
        assert_eq!(detector.buffered_samples(), RATE);
        let current = Turn {
            audio: vec![0.; WINDOW],
            ..Default::default()
        };
        let queued = VecDeque::from([Turn {
            audio: vec![0.; WINDOW - RATE - 1],
            ..Default::default()
        }]);
        assert_eq!(
            vad_buffered_samples(&current, &queued, &detector),
            2 * WINDOW - 1
        );
        assert!(detector.push(&[0, 0]).is_empty());
        assert_eq!(
            vad_buffered_samples(&current, &queued, &detector),
            2 * WINDOW
        );
        assert!(vad_buffered_samples(&current, &queued, &detector) + 1 > 2 * WINDOW);
        detector.finish();
        assert_eq!(detector.buffered_samples(), 0);
    }

    #[test]
    fn rolling_boundary_cannot_rewind_recognized_audio() {
        let mut turn = Turn {
            audio: vec![0.5; WINDOW],
            processed: 29 * RATE,
            ..Default::default()
        };
        turn.audio[28 * RATE..29 * RATE].fill(0.);
        assert!(turn.window_end() >= turn.processed);
        turn.processed = 28 * RATE;
        assert_eq!(turn.window_end(), 29 * RATE);
    }

    #[test]
    fn realtime_pcm_admission_is_bounded_and_commit_blocks_further_audio() {
        let mut turn = Turn::default();
        assert!(turn.append(&[0]).is_err());
        assert!(turn.append(&vec![0; 8 * RATE + 2]).is_err());
        turn.append(&vec![0; 4 * RATE]).unwrap();
        assert!(turn.ready());
        assert_eq!(turn.audio.len(), STEP);
        turn.committed = true;
        assert!(turn.append(&[0, 0]).is_err());
        turn.committed = false;
        turn.audio.resize(2 * WINDOW, 0.);
        assert!(turn.append(&[0, 0]).is_err());
    }
}
