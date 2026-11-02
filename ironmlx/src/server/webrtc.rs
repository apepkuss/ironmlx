//! OpenAI-style WebRTC call establishment for IronMLX transcription sessions.
use super::{
    api_error::{ApiError, ApiProtocol},
    realtime::{self, Configuration, SessionTransport},
    webrtc_audio::{AudioDecoder, Packet},
};
use axum::{
    body::{to_bytes, Body},
    extract::{FromRequest, Multipart, Path, Query, Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ironmlx_runtime::core::engine_pool::EnginePoolState;
use rtc::peer_connection::configuration::setting_engine::SctpMaxMessageSize;
use rtc::rtp_transceiver::rtp_sender::{RTCRtpCodec, RTCRtpCodecParameters, RtpCodecKind};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};
use webrtc::{
    data_channel::{DataChannel, DataChannelEvent},
    media_stream::track_remote::{TrackRemote, TrackRemoteEvent},
    peer_connection::{
        register_default_interceptors, MediaEngine, PeerConnection, PeerConnectionBuilder,
        PeerConnectionEventHandler, RTCConfigurationBuilder, RTCIceGatheringState, RTCIceServer,
        RTCPeerConnectionState, RTCSessionDescription, Registry, SettingEngineBuilder,
    },
};

pub(super) const MAX_REQUEST_BYTES: usize = 128 * 1024;
const MAX_EVENT_BYTES: usize = 256 * 1024;
#[derive(Clone, Copy)]
pub(super) struct Network {
    pub ip: IpAddr,
}
impl Default for Network {
    fn default() -> Self {
        Self {
            ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
        }
    }
}
#[derive(Default, Deserialize)]
pub(super) struct CallQuery {
    model: Option<String>,
    intent: Option<String>,
}
struct Call {
    stop: watch::Sender<bool>,
}
fn calls() -> &'static Mutex<HashMap<String, Call>> {
    static CALLS: OnceLock<Mutex<HashMap<String, Call>>> = OnceLock::new();
    CALLS.get_or_init(Mutex::default)
}
#[cfg(test)]
pub(super) fn active_call_count() -> usize {
    calls().lock().unwrap().len()
}

pub(super) async fn create(
    State(pool): State<EnginePoolState>,
    Query(query): Query<CallQuery>,
    request: Request,
) -> Response {
    create_with_pool(pool, query, request).await
}
pub(super) async fn create_with_pool(
    pool: EnginePoolState,
    query: CallQuery,
    request: Request,
) -> Response {
    match tokio::time::timeout(Duration::from_secs(30), create_call(pool, query, request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => error.into_response(ApiProtocol::OpenAi),
        Err(_) => {
            ApiError::service_unavailable("webrtc_setup_timeout", "WebRTC call setup timed out")
                .into_response(ApiProtocol::OpenAi)
        }
    }
}
pub(super) async fn hangup(Path(id): Path<String>) -> Response {
    if let Some(call) = calls().lock().unwrap().get(&id) {
        let _ = call.stop.send(true);
        StatusCode::OK.into_response()
    } else {
        ApiError::from_status(
            StatusCode::NOT_FOUND,
            "call_not_found",
            "The WebRTC call does not exist",
        )
        .into_response(ApiProtocol::OpenAi)
    }
}
fn bad(message: impl Into<String>) -> ApiError {
    ApiError::invalid_request("invalid_webrtc_request", message)
}
async fn parse(request: Request) -> Result<(String, Value), ApiError> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, MAX_REQUEST_BYTES)
        .await
        .map_err(|_| bad("WebRTC call request exceeds 128 KiB"))?;
    if content_type
        .split(';')
        .next()
        .is_some_and(|v| v.trim() == "application/sdp")
    {
        return Ok((
            String::from_utf8(bytes.to_vec()).map_err(|_| bad("SDP must be UTF-8"))?,
            json!({}),
        ));
    }
    if !content_type.starts_with("multipart/form-data;") {
        return Err(bad(
            "Use multipart/form-data with sdp and session, or application/sdp",
        ));
    }
    let mut form = Multipart::from_request(Request::from_parts(parts, Body::from(bytes)), &())
        .await
        .map_err(|_| bad("Invalid multipart call request"))?;
    let mut sdp = None;
    let mut session = None;
    while let Some(field) = form
        .next_field()
        .await
        .map_err(|_| bad("Invalid multipart field"))?
    {
        let name = field.name().unwrap_or("").to_owned();
        let text = field
            .text()
            .await
            .map_err(|_| bad("Call fields must contain UTF-8 text"))?;
        match name.as_str() {
            "sdp" if sdp.is_none() => sdp = Some(text),
            "session" if session.is_none() => {
                if text.len() > 32768 {
                    return Err(bad("session exceeds 32 KiB"));
                }
                session = Some(
                    serde_json::from_str(&text)
                        .map_err(|_| bad("session must contain a JSON object"))?,
                );
            }
            _ => return Err(bad(format!("Unknown or repeated call field: {name}"))),
        }
    }
    Ok((
        sdp.ok_or_else(|| bad("sdp is required"))?,
        session.unwrap_or_else(|| json!({})),
    ))
}
fn validate_offer(sdp: &str) -> Result<(), String> {
    if sdp.len() > 65536 || !sdp.starts_with("v=0") {
        return Err(String::from("Expected an SDP offer of at most 64 KiB"));
    }
    let mut audio = 0;
    let mut data = 0;
    let mut section = "";
    let mut opus = false;
    let mut sending = true;
    for line in sdp.lines().map(str::trim) {
        if let Some(media) = line.strip_prefix("m=") {
            let fields = media.split_whitespace().collect::<Vec<_>>();
            if fields.len() < 4 {
                return Err(String::from("Malformed SDP media section"));
            }
            section = fields[0];
            if fields[1] == "0" {
                section = "rejected";
                continue;
            }
            match section {
                "audio" => audio += 1,
                "application" => data += 1,
                _ => return Err(String::from(
                    "Only one incoming Opus audio track and one data channel section are supported",
                )),
            }
        } else if section == "audio" {
            if line.starts_with("a=rtpmap:") && line.to_ascii_lowercase().contains(" opus/48000/2")
            {
                opus = true;
            }
            if matches!(line, "a=recvonly" | "a=inactive") {
                sending = false;
            }
        }
    }
    if audio != 1 || data != 1 || !opus || !sending {
        return Err(String::from(
            "Offer must send one Opus/48000/2 audio track and negotiate a DataChannel",
        ));
    }
    Ok(())
}

async fn create_call(
    pool: EnginePoolState,
    query: CallQuery,
    request: Request,
) -> Result<Response, ApiError> {
    if query
        .intent
        .as_deref()
        .is_some_and(|v| v != "transcription")
    {
        return Err(bad("Only transcription calls are supported"));
    }
    let network = request
        .extensions()
        .get::<Network>()
        .copied()
        .unwrap_or_default();
    let (sdp, session) = parse(request).await?;
    validate_offer(&sdp).map_err(bad)?;
    let config = Configuration::for_webrtc(&pool, session, query.model).await?;
    let ice_servers = match std::env::var("IRONMLX_WEBRTC_ICE_SERVERS") {
        Ok(value) => {
            let servers: Vec<RTCIceServer> = serde_json::from_str(&value)
                .map_err(|_| bad("Invalid IRONMLX_WEBRTC_ICE_SERVERS configuration"))?;
            if servers.len() > 8 {
                return Err(bad("At most eight ICE servers are allowed"));
            }
            servers
        }
        Err(_) => Vec::new(),
    };
    let ice_policy = match std::env::var("IRONMLX_WEBRTC_ICE_POLICY").as_deref() {
        Ok("relay") => webrtc::peer_connection::RTCIceTransportPolicy::Relay,
        Ok("all") | Err(_) => webrtc::peer_connection::RTCIceTransportPolicy::All,
        _ => return Err(bad("IRONMLX_WEBRTC_ICE_POLICY must be all or relay")),
    };
    let admission = realtime::admit().map_err(|e| *e)?;
    // Load and pin inference dependencies before accepting live media. A cold
    // checkpoint must not overflow the short transport queue after HTTP 201.
    let prepared = config.prepare(&pool, admission.memory.clone()).await?;
    let (stop, cancel) = watch::channel(false);
    let (announce_tx, announce_rx) = mpsc::channel(4);
    let (gather_tx, mut gather_rx) = mpsc::channel(1);
    let handler = Arc::new(Handler {
        announcements: announce_tx,
        gather: gather_tx,
        stop: stop.clone(),
        audio: AtomicBool::new(false),
        data: AtomicBool::new(false),
    });
    let mut media = MediaEngine::default();
    media
        .register_codec(
            RTCRtpCodecParameters {
                rtp_codec: RTCRtpCodec {
                    mime_type: "audio/opus".into(),
                    clock_rate: 48000,
                    channels: 2,
                    sdp_fmtp_line: "minptime=10;useinbandfec=1".into(),
                    rtcp_feedback: vec![],
                },
                payload_type: 111,
            },
            RtpCodecKind::Audio,
        )
        .map_err(|e| bad(e.to_string()))?;
    let registry = register_default_interceptors(Registry::new(), &mut media)
        .map_err(|e| bad(e.to_string()))?;
    let settings = SettingEngineBuilder::default()
        .with_sctp_max_message_size(SctpMaxMessageSize::Bounded(MAX_EVENT_BYTES as u32))
        .with_sctp_max_receive_buffer_size(MAX_EVENT_BYTES as u32 * 2)
        .build();
    let peer = PeerConnectionBuilder::new()
        .with_configuration(
            RTCConfigurationBuilder::default()
                .with_ice_servers(ice_servers)
                .with_ice_transport_policy(ice_policy)
                .build(),
        )
        .with_media_engine(media)
        .with_interceptor_registry(registry)
        .with_setting_engine(settings)
        .with_data_channel_send_buffer_limit(MAX_EVENT_BYTES)
        .with_handler(handler)
        .with_udp_addrs(vec![format!(
            "{}",
            std::net::SocketAddr::new(network.ip, 0)
        )])
        .build()
        .await
        .map_err(|e| ApiError::service_unavailable("webrtc_unavailable", e.to_string()))?;
    let guard = PeerGuard(Arc::new(peer));
    let offer: RTCSessionDescription = serde_json::from_value(json!({"type":"offer","sdp":sdp}))
        .map_err(|_| bad("Malformed SDP offer"))?;
    guard
        .0
        .set_remote_description(offer)
        .await
        .map_err(|e| bad(e.to_string()))?;
    let answer = guard
        .0
        .create_answer(None)
        .await
        .map_err(|e| bad(e.to_string()))?;
    guard
        .0
        .set_local_description(answer)
        .await
        .map_err(|e| bad(e.to_string()))?;
    tokio::time::timeout(Duration::from_secs(20), gather_rx.recv())
        .await
        .map_err(|_| {
            ApiError::service_unavailable("webrtc_ice_timeout", "ICE gathering timed out")
        })?
        .ok_or_else(|| bad("ICE gathering stopped"))?;
    let answer = guard
        .0
        .local_description()
        .await
        .ok_or_else(|| bad("No local SDP answer"))?;
    if !answer
        .sdp
        .lines()
        .any(|line| line.starts_with("a=candidate:"))
    {
        return Err(ApiError::service_unavailable(
            "webrtc_ice_unavailable",
            "No usable ICE candidates; check the selected network interface and STUN/TURN configuration",
        ));
    }
    let id = format!("call_{}", uuid::Uuid::new_v4());
    calls().lock().unwrap().insert(id.clone(), Call { stop });
    let call_id = id.clone();
    tokio::spawn(async move {
        let _registration = Registration(call_id);
        let _slot = admission.slot;
        bridge(
            guard,
            pool,
            config,
            prepared,
            admission.memory,
            announce_rx,
            cancel,
        )
        .await;
    });
    Ok((
        StatusCode::CREATED,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/sdp"),
            ),
            (
                header::LOCATION,
                HeaderValue::from_str(&format!("/v1/realtime/calls/{id}")).expect("call URI"),
            ),
        ],
        answer.sdp,
    )
        .into_response())
}
struct Registration(String);
impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(mut registry) = calls().lock() {
            registry.remove(&self.0);
        }
    }
}
struct PeerGuard(Arc<dyn PeerConnection>);
impl Drop for PeerGuard {
    fn drop(&mut self) {
        let peer = self.0.clone();
        tokio::spawn(async move {
            let _ = peer.close().await;
        });
    }
}
enum Announcement {
    Track(Arc<dyn TrackRemote>),
    Data(Arc<dyn DataChannel>),
}
struct Handler {
    announcements: mpsc::Sender<Announcement>,
    gather: mpsc::Sender<()>,
    stop: watch::Sender<bool>,
    audio: AtomicBool,
    data: AtomicBool,
}
#[async_trait::async_trait]
impl PeerConnectionEventHandler for Handler {
    async fn on_ice_gathering_state_change(&self, state: RTCIceGatheringState) {
        if state == RTCIceGatheringState::Complete {
            let _ = self.gather.try_send(());
        }
    }
    async fn on_connection_state_change(&self, state: RTCPeerConnectionState) {
        if matches!(
            state,
            RTCPeerConnectionState::Failed | RTCPeerConnectionState::Closed
        ) {
            let _ = self.stop.send(true);
        }
    }
    async fn on_track(&self, track: Arc<dyn TrackRemote>) {
        if self.audio.swap(true, Ordering::SeqCst)
            || self
                .announcements
                .try_send(Announcement::Track(track))
                .is_err()
        {
            let _ = self.stop.send(true);
        }
    }
    async fn on_data_channel(&self, data: Arc<dyn DataChannel>) {
        if self.data.swap(true, Ordering::SeqCst)
            || self
                .announcements
                .try_send(Announcement::Data(data))
                .is_err()
        {
            let _ = self.stop.send(true);
        }
    }
}
struct InferenceTask(Option<tokio::task::JoinHandle<Result<(), ()>>>);
impl Drop for InferenceTask {
    fn drop(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}
async fn dc_error(dc: &dyn DataChannel, code: &str, message: &str) {
    let value = json!({"type":"error","event_id":format!("event_{}",uuid::Uuid::new_v4()),"error":{"type":"invalid_request_error","code":code,"message":message}});
    let _ = tokio::time::timeout(Duration::from_secs(2), dc.send_text(&value.to_string())).await;
}
fn pcm_event(bytes: Vec<u8>) -> axum::extract::ws::Message {
    axum::extract::ws::Message::Text(
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(bytes)}).to_string(),
    )
}
async fn bridge(
    _peer: PeerGuard,
    pool: EnginePoolState,
    config: Configuration,
    prepared: realtime::Prepared,
    memory: Arc<ironmlx_runtime::core::process_memory::MemoryReservation>,
    mut announcements: mpsc::Receiver<Announcement>,
    mut cancel: watch::Receiver<bool>,
) {
    let (input, incoming) = mpsc::channel(128);
    let (outgoing, mut output) = mpsc::channel(32);
    let mut config = Some((config, prepared, incoming, outgoing));
    let mut inference = InferenceTask(None);
    let mut dc: Option<Arc<dyn DataChannel>> = None;
    let mut track: Option<Arc<dyn TrackRemote>> = None;
    let mut decoder = match AudioDecoder::new() {
        Ok(decoder) => decoder,
        Err(_) => return,
    };
    let mut opened = false;
    let started = Instant::now();
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    loop {
        if *cancel.borrow()
            || started.elapsed() > Duration::from_secs(7200)
            || (!opened && started.elapsed() > Duration::from_secs(30))
        {
            break;
        }
        let mut flush = false;
        tokio::select! {
            _ = cancel.changed() => break,
            announced = announcements.recv() => match announced {
                Some(Announcement::Data(data)) => {
                    if data.label().await.as_deref() != Ok("oai-events")
                        || data.ordered().await != Ok(true)
                        || data.max_retransmits().await.ok().flatten().is_some()
                        || data.max_packet_life_time().await.ok().flatten().is_some()
                    {
                        let _ = data.close().await;
                        break;
                    }
                    dc = Some(data);
                }
                Some(Announcement::Track(audio)) => {
                    let ssrcs = audio.ssrcs().await;
                    if audio.kind().await != RtpCodecKind::Audio
                        || ssrcs.len() != 1
                        || audio.codec(ssrcs[0]).await.is_none_or(|c| {
                            !c.mime_type.eq_ignore_ascii_case("audio/opus") || c.clock_rate != 48000
                        })
                    {
                        break;
                    }
                    track = Some(audio);
                }
                None => break,
            },
            event = async { dc.as_ref().unwrap().poll().await }, if dc.is_some() => match event {
                Some(DataChannelEvent::OnOpen) => {
                    if opened { break; }
                    opened = true;
                    let (initial, prepared, incoming, outgoing) = config.take().unwrap();
                    inference.0 = Some(tokio::spawn(realtime::run(
                        SessionTransport::WebRtc { incoming, outgoing },
                        pool.clone(), initial, memory.clone(), Some(prepared),
                    )));
                }
                Some(DataChannelEvent::OnMessage(message)) => {
                    let dc = dc.as_ref().unwrap();
                    if !message.is_string || message.data.len() > MAX_EVENT_BYTES {
                        dc_error(dc.as_ref(), "invalid_realtime_request", "Send JSON text events of at most 256 KiB").await;
                        continue;
                    }
                    let event: Value = match serde_json::from_slice(&message.data) {
                        Ok(event) => event,
                        Err(_) => {
                            dc_error(dc.as_ref(), "invalid_realtime_request", "Invalid JSON event").await;
                            continue;
                        }
                    };
                    match event.get("type").and_then(Value::as_str) {
                        Some("input_audio_buffer.append") => {
                            dc_error(dc.as_ref(), "unsupported_audio_transport", "Send audio through the WebRTC media track").await;
                            continue;
                        }
                        Some("input_audio_buffer.clear") => decoder.clear(),
                        Some("input_audio_buffer.commit") => {
                            match decoder.drain(Instant::now(), true) {
                                Ok(chunks) => {
                                    if chunks.into_iter().any(|pcm| input.try_send(pcm_event(pcm)).is_err()) {
                                        dc_error(dc.as_ref(), "realtime_capacity", "WebRTC input buffer is full").await;
                                        break;
                                    }
                                }
                                Err(error) => {
                                    dc_error(dc.as_ref(), "webrtc_audio_failed", &error.to_string()).await;
                                    break;
                                }
                            }
                        }
                        _ => {}
                    }
                    if input.try_send(axum::extract::ws::Message::Text(event.to_string())).is_err() {
                        dc_error(dc.as_ref(), "realtime_capacity", "WebRTC input buffer is full").await;
                        break;
                    }
                }
                Some(DataChannelEvent::OnClose) | None => break,
                _ => {},
            },
            event = async { track.as_ref().unwrap().poll().await }, if track.is_some() => match event {
                Some(TrackRemoteEvent::OnRtpPacket(packet)) => {
                    // RTP padding/keepalive packets carry no encoded audio.
                    if packet.payload.is_empty() { continue; }
                    if let Err(error) = decoder.push(Packet {
                        sequence: packet.header.sequence_number,
                        timestamp: packet.header.timestamp,
                        payload: packet.payload.to_vec(),
                        received: Instant::now(),
                    }) {
                        if let Some(dc) = &dc {
                            dc_error(dc.as_ref(), "webrtc_audio_failed", &error.to_string()).await;
                        }
                        break;
                    }
                    flush = true;
                }
                Some(TrackRemoteEvent::OnError) | Some(TrackRemoteEvent::OnEnded) | None => break,
                _ => {},
            },
            event = output.recv(), if opened => {
                let Some(axum::extract::ws::Message::Text(text)) = event else { break; };
                if tokio::time::timeout(Duration::from_secs(10), dc.as_ref().unwrap().send_text(&text))
                    .await.map_or(true, |r| r.is_err())
                {
                    break;
                }
            }
            _ = tick.tick() => flush = true,
            _ = async { inference.0.as_mut().unwrap().await }, if inference.0.is_some() => {
                while let Ok(axum::extract::ws::Message::Text(text)) = output.try_recv() {
                    let _ = tokio::time::timeout(Duration::from_secs(2), dc.as_ref().unwrap().send_text(&text)).await;
                }
                break;
            }
        }
        if opened && flush {
            match decoder.drain(Instant::now(), false) {
                Ok(chunks) => {
                    if chunks
                        .into_iter()
                        .any(|pcm| input.try_send(pcm_event(pcm)).is_err())
                    {
                        if let Some(dc) = &dc {
                            dc_error(
                                dc.as_ref(),
                                "realtime_capacity",
                                "WebRTC input buffer is full",
                            )
                            .await;
                        }
                        break;
                    }
                }
                Err(error) => {
                    if let Some(dc) = &dc {
                        dc_error(dc.as_ref(), "webrtc_audio_failed", &error.to_string()).await;
                    }
                    break;
                }
            }
        }
    }
}

pub(super) fn call_path(path: &str) -> bool {
    path == "/v1/realtime/calls" || path.starts_with("/v1/realtime/calls/")
}
fn allowed_origin(request: &Request) -> Option<HeaderValue> {
    let origin = request.headers().get(header::ORIGIN)?;
    let value = origin.to_str().ok()?;
    let url = url::Url::parse(value).ok()?;
    if url.origin().ascii_serialization() != value
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    let network = request
        .extensions()
        .get::<Network>()
        .copied()
        .unwrap_or_default();
    let same_interface = url
        .host_str()
        .and_then(|h| h.trim_matches(['[', ']']).parse::<IpAddr>().ok())
        == Some(network.ip);
    let listed = std::env::var("IRONMLX_WEBRTC_ALLOWED_ORIGINS")
        .ok()
        .is_some_and(|v| v.split(',').any(|entry| entry.trim() == value));
    if (url.scheme() == "https" || (url.scheme() == "http" && local))
        && (local || same_interface || listed)
    {
        Some(origin.clone())
    } else {
        None
    }
}
fn cors_headers(response: &mut Response, origin: HeaderValue) {
    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Authorization, Content-Type"),
    );
    headers.insert(
        header::ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static("Location"),
    );
    headers.insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static("600"),
    );
}
pub(super) fn preflight(request: &Request) -> Response {
    if let Some(origin) = allowed_origin(request) {
        let mut response = StatusCode::NO_CONTENT.into_response();
        cors_headers(&mut response, origin);
        response
    } else {
        ApiError::from_status(
            StatusCode::FORBIDDEN,
            "origin_not_allowed",
            "WebRTC request origin is not allowed",
        )
        .into_response(ApiProtocol::OpenAi)
    }
}
pub(super) async fn cors(request: Request, next: Next) -> Response {
    if !call_path(request.uri().path()) {
        return next.run(request).await;
    }
    if request.method() == Method::OPTIONS {
        return preflight(&request);
    }
    if request.headers().contains_key(header::ORIGIN) {
        let Some(origin) = allowed_origin(&request) else {
            return preflight(&request);
        };
        let mut response = next.run(request).await;
        cors_headers(&mut response, origin);
        response
    } else {
        next.run(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    #[tokio::test]
    async fn call_parser_rejects_unknown_duplicate_and_oversized_fields() {
        assert!(parse(
            Request::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap()
        )
        .await
        .is_err());
        assert!(parse(
            Request::builder()
                .header(header::CONTENT_TYPE, "application/sdp")
                .body(Body::from(vec![b'a'; MAX_REQUEST_BYTES + 1]))
                .unwrap()
        )
        .await
        .is_err());
        for fields in [
            vec![("sdp", "offer"), ("sdp", "again")],
            vec![("sdp", "offer"), ("session", "not-json")],
            vec![("sdp", "offer"), ("unknown", "ignored")],
        ] {
            let mut body = String::new();
            for (name, value) in fields {
                body.push_str(&format!(
                    "--call\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                ));
            }
            body.push_str("--call--\r\n");
            assert!(parse(
                Request::builder()
                    .header(header::CONTENT_TYPE, "multipart/form-data; boundary=call")
                    .body(Body::from(body))
                    .unwrap()
            )
            .await
            .is_err());
        }
    }
    #[test]
    fn offer_requires_one_sending_opus_track_and_a_data_channel() {
        let offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\na=sendonly\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n";
        validate_offer(offer).unwrap();
        for invalid in [
            offer.replace("opus/48000/2", "PCMU/8000"),
            offer.replace("a=sendonly", "a=recvonly"),
            offer.replace("m=application 9", "m=application 0"),
            format!("{offer}m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"),
            format!("{offer}m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"),
        ] {
            assert!(validate_offer(&invalid).is_err());
        }
    }
    #[tokio::test]
    async fn browser_preflight_is_limited_to_webrtc_and_rejects_unapproved_origins() {
        let app = axum::Router::new()
            .route(
                "/v1/realtime/calls",
                axum::routing::post(|| async { StatusCode::CREATED }),
            )
            .layer(axum::middleware::from_fn(cors));
        for (origin, status) in [
            ("http://localhost:3000", StatusCode::NO_CONTENT),
            ("http://127.0.0.1:9000", StatusCode::NO_CONTENT),
            ("https://untrusted.example", StatusCode::FORBIDDEN),
            ("null", StatusCode::FORBIDDEN),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("OPTIONS")
                        .uri("/v1/realtime/calls")
                        .header(header::ORIGIN, origin)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            if status == StatusCode::NO_CONTENT {
                assert_eq!(
                    response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
                    origin
                );
            }
        }
    }
}
