use std::fmt;
use std::io::Read;
use std::net::{IpAddr, SocketAddr};

use anyhow::{bail, Context};
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::Router;
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::Result;

pub const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum NetworkMode {
    #[default]
    Local,
    Lan,
}

#[derive(Clone)]
pub struct LanSecurity {
    api_key_digest: [u8; 32],
    tls_certificate_pem: Vec<u8>,
    tls_private_key_pem: Vec<u8>,
}

impl fmt::Debug for LanSecurity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LanSecurity")
            .field("api_key_digest", &"[REDACTED]")
            .field("tls_certificate_pem", &"[REDACTED]")
            .field("tls_private_key_pem", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct ServerNetworkConfig {
    pub local_addr: SocketAddr,
    pub lan_addr: Option<SocketAddr>,
    pub lan_security: Option<LanSecurity>,
}

impl ServerNetworkConfig {
    pub fn local(host: &str, port: u16) -> Result<Self> {
        let local_addr = parse_addr(host, port)?;
        if !local_addr.ip().is_loopback() {
            bail!("local_bind_address_required: local mode requires a loopback --host");
        }
        Ok(Self {
            local_addr,
            lan_addr: None,
            lan_security: None,
        })
    }

    pub fn resolve(
        mode: NetworkMode,
        host: &str,
        port: u16,
        lan_host: Option<IpAddr>,
        bootstrap_stdin: bool,
    ) -> Result<Self> {
        let mut config = Self::local(host, port)?;
        match mode {
            NetworkMode::Local => {
                if lan_host.is_some() || bootstrap_stdin {
                    bail!("lan_options_forbidden: LAN options require --network-mode lan");
                }
            }
            NetworkMode::Lan => {
                let lan_ip = lan_host.context(
                    "lan_bind_address_required: LAN mode requires a selected --lan-host IP",
                )?;
                if lan_ip.is_loopback() || lan_ip.is_unspecified() || lan_ip.is_multicast() {
                    bail!(
                        "lan_bind_address_unsafe: --lan-host must be a concrete unicast LAN address"
                    );
                }
                if !bootstrap_stdin {
                    bail!(
                        "lan_security_material_missing: LAN mode requires --security-bootstrap-stdin"
                    );
                }
                let security = LanSecurity::read_from_stdin()?;
                config.lan_addr = Some(SocketAddr::new(lan_ip, port));
                config.lan_security = Some(security);
            }
        }
        Ok(config)
    }
}

fn parse_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let normalized = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    let ip: IpAddr = normalized
        .parse()
        .with_context(|| format!("parsing bind IP address {host}"))?;
    Ok(SocketAddr::new(ip, port))
}

#[derive(Deserialize)]
struct SecurityBootstrap {
    api_key_sha256_base64: String,
    tls_certificate_pem_base64: String,
    tls_private_key_pem_base64: String,
}

impl LanSecurity {
    fn read_from_stdin() -> Result<Self> {
        let mut payload = Vec::new();
        std::io::stdin()
            .take((MAX_REQUEST_BODY_BYTES + 1) as u64)
            .read_to_end(&mut payload)
            .context("lan_security_material_invalid: reading security bootstrap from stdin")?;
        if payload.len() > MAX_REQUEST_BODY_BYTES {
            bail!("lan_security_material_invalid: security bootstrap is too large");
        }
        let bootstrap: SecurityBootstrap = serde_json::from_slice(&payload)
            .context("lan_security_material_invalid: parsing security bootstrap")?;
        let digest = decode_base64_field(&bootstrap.api_key_sha256_base64, "API key digest")?;
        let api_key_digest: [u8; 32] = digest.try_into().map_err(|_| {
            anyhow::anyhow!("lan_security_material_invalid: API key digest must be 32 bytes")
        })?;
        let tls_certificate_pem = decode_base64_field(
            &bootstrap.tls_certificate_pem_base64,
            "TLS certificate chain",
        )?;
        let tls_private_key_pem =
            decode_base64_field(&bootstrap.tls_private_key_pem_base64, "TLS private key")?;
        if tls_certificate_pem.is_empty() || tls_private_key_pem.is_empty() {
            bail!("lan_security_material_invalid: TLS material must not be empty");
        }
        Ok(Self {
            api_key_digest,
            tls_certificate_pem,
            tls_private_key_pem,
        })
    }
}

fn decode_base64_field(value: &str, field: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .with_context(|| format!("lan_security_material_invalid: invalid {field}"))
}

pub fn error_response(status: StatusCode, code: &'static str, message: &'static str) -> Response {
    protocol_error_response(status, code, message, super::api_error::ApiProtocol::OpenAi)
}

fn protocol_error_response(
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    protocol: super::api_error::ApiProtocol,
) -> Response {
    super::api_error::ApiError::from_status(status, code, message).into_response(protocol)
}

pub(crate) fn image_error_response(
    error: anyhow::Error,
    protocol: super::api_error::ApiProtocol,
) -> Response {
    if let Some(error) = error.downcast_ref::<ironmlx_lm::core::image_input::ImageInputError>() {
        return super::api_error::ApiError::from_status(
            super::image_input::image_error_status(*error),
            error.code(),
            error.message(),
        )
        .into_response(protocol);
    }
    super::api_error::ApiError::invalid_request(
        "image_decode_failed",
        "The image could not be decoded safely.",
    )
    .into_response(protocol)
}

async fn authenticate_lan(
    State(expected): State<[u8; 32]>,
    request: Request,
    next: Next,
) -> Response {
    let protocol = super::api_error::ApiProtocol::from_path(request.uri().path());
    if request.method() == axum::http::Method::OPTIONS
        && super::webrtc::call_path(request.uri().path())
    {
        return super::webrtc::preflight(&request);
    }
    let Some(value) = request.headers().get(header::AUTHORIZATION) else {
        return protocol_error_response(
            StatusCode::UNAUTHORIZED,
            "auth_invalid",
            "A valid Bearer API key is required.",
            protocol,
        );
    };
    let Ok(value) = value.to_str() else {
        return protocol_error_response(
            StatusCode::UNAUTHORIZED,
            "auth_invalid",
            "A valid Bearer API key is required.",
            protocol,
        );
    };
    let Some(api_key) = value.strip_prefix("Bearer ") else {
        return protocol_error_response(
            StatusCode::UNAUTHORIZED,
            "auth_invalid",
            "A valid Bearer API key is required.",
            protocol,
        );
    };
    let digest: [u8; 32] = Sha256::digest(api_key.as_bytes()).into();
    if digest.ct_eq(&expected).unwrap_u8() != 1 {
        return protocol_error_response(
            StatusCode::UNAUTHORIZED,
            "auth_invalid",
            "A valid Bearer API key is required.",
            protocol,
        );
    }
    next.run(request).await
}

async fn enforce_request_body_limit(request: Request, next: Next) -> Response {
    let protocol = super::api_error::ApiProtocol::from_path(request.uri().path());
    let body_limit = if super::webrtc::call_path(request.uri().path()) {
        super::webrtc::MAX_REQUEST_BYTES
    } else if matches!(
        request.uri().path(),
        "/v1/audio/transcriptions" | "/v1/audio/alignments" | "/v1/audio/diarizations"
    ) {
        super::transcriptions::MAX_ASR_REQUEST_BYTES
    } else {
        MAX_REQUEST_BODY_BYTES
    };
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > body_limit)
    {
        return super::api_error::ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            format!(
                "The request body exceeds the {} KiB limit.",
                body_limit / 1024
            ),
        )
        .into_response(protocol);
    }
    if matches!(
        request.uri().path(),
        "/v1/audio/speech"
            | "/v1/audio/transcriptions"
            | "/v1/audio/alignments"
            | "/v1/audio/diarizations"
    ) {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(body, body_limit).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return super::api_error::ApiError::from_status(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_body_too_large",
                format!(
                    "The request body exceeds the {} KiB limit.",
                    body_limit / 1024
                ),
            )
            .into_response(protocol);
        }
    };
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

fn bounded_router(router: Router) -> Router {
    router
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(middleware::from_fn(enforce_request_body_limit))
}

pub(super) fn lan_api_router(router: Router, key_digest: [u8; 32]) -> Router {
    bounded_router(router)
        .layer(middleware::from_fn_with_state(key_digest, authenticate_lan))
        .layer(middleware::from_fn(super::webrtc::cors))
}

fn local_management_router(router: Router) -> Router {
    bounded_router(router.merge(crate::logging::router()))
        .layer(middleware::from_fn(super::webrtc::cors))
}

/// The local listener and, in LAN mode, the authenticated LAN listener, bound
/// before anything is served so the ports are open while the process starts.
pub struct BoundListeners {
    local: std::net::TcpListener,
    local_addr: SocketAddr,
    lan: Option<BoundLanListener>,
}

struct BoundLanListener {
    listener: std::net::TcpListener,
    addr: SocketAddr,
    security: LanSecurity,
}

impl BoundListeners {
    pub fn bind(config: &ServerNetworkConfig) -> Result<Self> {
        let local = bind_listener(config.local_addr)
            .with_context(|| format!("binding local listener {}", config.local_addr))?;
        let local_addr = local
            .local_addr()
            .context("reading local listener address")?;
        let lan = match config.lan_addr {
            None => None,
            Some(lan_addr) => {
                let security = config
                    .lan_security
                    .clone()
                    .context("lan_security_material_missing: LAN mode has no security material")?;
                let listener = bind_listener(lan_addr)
                    .with_context(|| format!("binding selected LAN listener {lan_addr}"))?;
                let addr = listener
                    .local_addr()
                    .context("reading LAN listener address")?;
                Some(BoundLanListener {
                    listener,
                    addr,
                    security,
                })
            }
        };
        Ok(Self {
            local,
            local_addr,
            lan,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn lan_addr(&self) -> Option<SocketAddr> {
        self.lan.as_ref().map(|lan| lan.addr)
    }
}

fn bind_listener(addr: SocketAddr) -> Result<std::net::TcpListener> {
    let listener = std::net::TcpListener::bind(addr)?;
    listener
        .set_nonblocking(true)
        .context("configuring listener")?;
    Ok(listener)
}

pub async fn serve_router(router: Router, config: ServerNetworkConfig, label: &str) -> Result<()> {
    serve_listeners(BoundListeners::bind(&config)?, router, label).await
}

/// Serve `router` on listeners bound earlier; it answers with its startup
/// routes until [`SwitchableRouter::activate`] installs the full router.
///
/// [`SwitchableRouter::activate`]: super::startup_status::SwitchableRouter::activate
pub async fn serve_bound(
    listeners: BoundListeners,
    router: super::startup_status::SwitchableRouter,
    label: &str,
) -> Result<()> {
    serve_listeners(listeners, Router::new().fallback_service(router), label).await
}

async fn serve_listeners(listeners: BoundListeners, router: Router, label: &str) -> Result<()> {
    let local_router =
        local_management_router(router.clone()).layer(axum::Extension(super::webrtc::Network {
            ip: listeners.local_addr.ip(),
        }));
    let local_listener = tokio::net::TcpListener::from_std(listeners.local)
        .context("creating Tokio local listener")?;

    let Some(lan) = listeners.lan else {
        tracing::info!(
            "{label} listening locally on http://{}",
            listeners.local_addr
        );
        axum::serve(local_listener, local_router).await?;
        return Ok(());
    };
    let _ = rustls::crypto::ring::default_provider().install_default();
    let tls = axum_server::tls_rustls::RustlsConfig::from_pem(
        lan.security.tls_certificate_pem,
        lan.security.tls_private_key_pem,
    )
    .await
    .context("lan_security_material_invalid: loading TLS certificate and private key")?;
    let lan_router = lan_api_router(router, lan.security.api_key_digest).layer(axum::Extension(
        super::webrtc::Network { ip: lan.addr.ip() },
    ));
    tracing::info!(
        "{label} listening locally on http://{} and on authenticated LAN endpoint https://{}",
        listeners.local_addr,
        lan.addr
    );
    let local = axum::serve(local_listener, local_router);
    let lan = axum_server::from_tcp_rustls(lan.listener, tls).serve(lan_router.into_make_service());
    tokio::try_join!(local, lan)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use tower::ServiceExt;

    #[test]
    fn local_mode_rejects_non_loopback_bind() {
        let error = ServerNetworkConfig::local("0.0.0.0", 8080).unwrap_err();
        assert!(error.to_string().contains("local_bind_address_required"));
    }

    #[test]
    fn lan_mode_requires_selected_address_before_reading_stdin() {
        let error = ServerNetworkConfig::resolve(NetworkMode::Lan, "127.0.0.1", 8080, None, true)
            .unwrap_err();
        assert!(error.to_string().contains("lan_bind_address_required"));
    }

    #[test]
    fn lan_mode_rejects_wildcard_and_loopback() {
        for address in ["0.0.0.0", "127.0.0.1", "::", "::1"] {
            let error = ServerNetworkConfig::resolve(
                NetworkMode::Lan,
                "127.0.0.1",
                8080,
                Some(address.parse().unwrap()),
                false,
            )
            .unwrap_err();
            assert!(error.to_string().contains("lan_bind_address_unsafe"));
        }
    }

    #[test]
    fn lan_mode_without_authentication_material_is_rejected() {
        let error = ServerNetworkConfig::resolve(
            NetworkMode::Lan,
            "127.0.0.1",
            8080,
            Some("192.168.1.24".parse().unwrap()),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("lan_security_material_missing"));
    }

    fn protected_test_router(api_key: &str) -> Router {
        let expected: [u8; 32] = Sha256::digest(api_key.as_bytes()).into();
        bounded_router(
            Router::new()
                .route("/admin/api/test", get(|| async { "ok" }))
                .route("/v1/messages", get(|| async { "ok" }))
                .route(
                    "/v1/audio/voices",
                    get(|| async { "ok" }).post(|| async { "ok" }),
                )
                .route(
                    "/v1/audio/voices/:id",
                    get(|| async { "ok" })
                        .patch(|| async { "ok" })
                        .delete(|| async { "ok" }),
                )
                .route("/v1/audio/voices/:id/preview", get(|| async { "ok" })),
        )
        .layer(middleware::from_fn_with_state(expected, authenticate_lan))
    }

    #[tokio::test]
    async fn webrtc_browser_auth_and_limits_are_enforced_after_cors_preflight() {
        let router = lan_api_router(
            Router::new().route(
                "/v1/realtime/calls",
                axum::routing::post(|| async { StatusCode::CREATED }),
            ),
            Sha256::digest(b"imx_correct").into(),
        );
        for (method, token, size, status) in [
            ("OPTIONS", None, 0, StatusCode::NO_CONTENT),
            ("POST", None, 0, StatusCode::UNAUTHORIZED),
            ("POST", Some("Bearer imx_correct"), 0, StatusCode::CREATED),
            (
                "POST",
                Some("Bearer imx_correct"),
                super::super::webrtc::MAX_REQUEST_BYTES + 1,
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let mut request = Request::builder()
                .method(method)
                .uri("/v1/realtime/calls")
                .header(header::ORIGIN, "http://localhost:3000")
                .header(header::CONTENT_TYPE, "application/sdp");
            if let Some(token) = token {
                request = request.header(header::AUTHORIZATION, token);
            }
            // No Content-Length: exercise the streamed body size limit too.
            let response = router
                .clone()
                .oneshot(request.body(Body::from(vec![b'a'; size])).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
                "http://localhost:3000"
            );
            if status == StatusCode::PAYLOAD_TOO_LARGE {
                let body = to_bytes(response.into_body(), 4096).await.unwrap();
                assert!(String::from_utf8_lossy(&body).contains("128 KiB"));
            }
        }
    }

    #[tokio::test]
    async fn log_control_is_local_only_even_with_valid_lan_credentials() {
        let local = local_management_router(Router::new());
        let response = local
            .oneshot(
                Request::builder()
                    .uri("/admin/api/log-level")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Unit-test process has no global logger controller, but the route exists.
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        for method in ["GET", "POST"] {
            let response = protected_test_router("imx_correct")
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri("/admin/api/log-level")
                        .header(header::AUTHORIZATION, "Bearer imx_correct")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
        }
    }

    #[tokio::test]
    async fn every_lan_route_requires_the_correct_bearer_key() {
        let router = protected_test_router("imx_correct");
        for request in [
            Request::builder()
                .uri("/admin/api/test")
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .uri("/admin/api/test?api_key=imx_correct")
                .header(header::AUTHORIZATION, "Bearer imx_wrong")
                .body(Body::empty())
                .unwrap(),
        ] {
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(String::from_utf8_lossy(&bytes).contains("auth_invalid"));
        }

        let request = Request::builder()
            .uri("/admin/api/test")
            .header(header::AUTHORIZATION, "Bearer imx_correct")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn voice_routes_require_lan_bearer_authentication() {
        let router = protected_test_router("imx_correct");
        for (method, path) in [
            ("GET", "/v1/audio/voices"),
            ("POST", "/v1/audio/voices"),
            ("GET", "/v1/audio/voices/narrator/preview"),
            ("PATCH", "/v1/audio/voices/narrator"),
            ("DELETE", "/v1/audio/voices/narrator"),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {path}"
            );

            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .header(header::AUTHORIZATION, "Bearer imx_correct")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{method} {path}");
        }
    }

    #[tokio::test]
    async fn speech_auth_failure_precedes_body_processing() {
        let response = protected_test_router("imx_correct")
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("invalid JSON"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "auth_invalid");
    }

    #[tokio::test]
    async fn messages_auth_failure_uses_anthropic_error_contract() {
        let response = protected_test_router("imx_correct")
            .oneshot(
                Request::builder()
                    .uri("/v1/messages")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let request_id = response.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "authentication_error");
        assert_eq!(body["error"]["code"], "auth_invalid");
        assert_eq!(body["request_id"], request_id);
    }

    #[tokio::test]
    async fn oversized_body_is_rejected_with_stable_code() {
        let router = bounded_router(Router::new().route("/", get(|| async { "ok" })));
        let request = Request::builder()
            .uri("/")
            .header(header::CONTENT_LENGTH, MAX_REQUEST_BODY_BYTES + 1)
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("request_body_too_large"));
    }

    #[tokio::test]
    async fn startup_status_router_reports_503_until_ready_on_lan() {
        let certified = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_owned()]).unwrap();
        let config = ServerNetworkConfig {
            local_addr: "127.0.0.1:0".parse().unwrap(),
            lan_addr: Some("127.0.0.1:0".parse().unwrap()),
            lan_security: Some(LanSecurity {
                api_key_digest: Sha256::digest(b"imx_correct").into(),
                tls_certificate_pem: certified.cert.pem().into_bytes(),
                tls_private_key_pem: certified.signing_key.serialize_pem().into_bytes(),
            }),
        };
        let listeners = BoundListeners::bind(&config).unwrap();
        let local = format!("http://{}", listeners.local_addr());
        let lan = format!("https://{}", listeners.lan_addr().unwrap());
        let registry = ironmlx_runtime::core::load_registry::LoadRegistry::new();
        let router = super::super::startup_status::SwitchableRouter::starting(registry);
        let server = tokio::spawn(serve_bound(listeners, router.clone(), "startup LAN test"));
        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap();
        let fetch = |url: String, key: Option<&'static str>| {
            let mut request = client.get(url);
            if let Some(key) = key {
                request = request.bearer_auth(key);
            }
            request.send()
        };

        // Both listeners answer while starting; LAN still requires its key.
        let response = fetch(format!("{lan}/healthz"), None).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        for (base, key) in [(&lan, Some("imx_correct")), (&local, None)] {
            let response = fetch(format!("{base}/healthz"), key).await.unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                response.json::<serde_json::Value>().await.unwrap(),
                serde_json::json!({"status": "starting"})
            );
            let response = fetch(format!("{base}/admin/api/models/loads"), key)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.json::<serde_json::Value>().await.unwrap(),
                serde_json::json!({"loads": []})
            );
        }
        // Log control stays local-only.
        let response = fetch(format!("{lan}/admin/api/log-level"), Some("imx_correct"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "backend_starting");

        router.activate(Router::new().route("/healthz", get(|| async { "ready" })));
        for (base, key) in [(&lan, Some("imx_correct")), (&local, None)] {
            let response = fetch(format!("{base}/healthz"), key).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let response = fetch(format!("{lan}/healthz"), None).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        server.abort();
    }

    #[tokio::test]
    async fn messages_body_limit_uses_anthropic_error_contract() {
        let router =
            bounded_router(Router::new().route("/v1/messages", get(|| async { "unreachable" })));
        let request = Request::builder()
            .uri("/v1/messages")
            .header(header::CONTENT_LENGTH, MAX_REQUEST_BODY_BYTES + 1)
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
        let request_id = response.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "request_too_large");
        assert_eq!(body["error"]["code"], "request_body_too_large");
        assert_eq!(body["request_id"], request_id);
    }
}
