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
    if let Some(error) = error.downcast_ref::<crate::core::image_input::ImageInputError>() {
        return super::api_error::ApiError::from_status(
            error.status(),
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
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > MAX_REQUEST_BODY_BYTES)
    {
        return super::api_error::ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "The request body exceeds the 32 MiB limit.",
        )
        .into_response(protocol);
    }
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(body, MAX_REQUEST_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return super::api_error::ApiError::from_status(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_body_too_large",
                "The request body exceeds the 32 MiB limit.",
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

pub async fn serve_router(router: Router, config: ServerNetworkConfig, label: &str) -> Result<()> {
    let local_router = bounded_router(router.clone());
    let local_std = std::net::TcpListener::bind(config.local_addr)
        .with_context(|| format!("binding local listener {}", config.local_addr))?;
    local_std
        .set_nonblocking(true)
        .context("configuring local listener")?;
    let local_listener =
        tokio::net::TcpListener::from_std(local_std).context("creating Tokio local listener")?;

    let Some(lan_addr) = config.lan_addr else {
        tracing::info!("{label} listening locally on http://{}", config.local_addr);
        axum::serve(local_listener, local_router).await?;
        return Ok(());
    };
    let security = config
        .lan_security
        .context("lan_security_material_missing: LAN mode has no security material")?;
    let _ = rustls::crypto::ring::default_provider().install_default();
    let tls = axum_server::tls_rustls::RustlsConfig::from_pem(
        security.tls_certificate_pem,
        security.tls_private_key_pem,
    )
    .await
    .context("lan_security_material_invalid: loading TLS certificate and private key")?;
    let lan_std = std::net::TcpListener::bind(lan_addr)
        .with_context(|| format!("binding selected LAN listener {lan_addr}"))?;
    lan_std
        .set_nonblocking(true)
        .context("configuring LAN listener")?;
    let lan_router = bounded_router(router).layer(middleware::from_fn_with_state(
        security.api_key_digest,
        authenticate_lan,
    ));
    tracing::info!(
        "{label} listening locally on http://{} and on authenticated LAN endpoint https://{}",
        config.local_addr,
        lan_addr
    );
    let local = axum::serve(local_listener, local_router);
    let lan = axum_server::from_tcp_rustls(lan_std, tls).serve(lan_router.into_make_service());
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
                .route("/v1/messages", get(|| async { "ok" })),
        )
        .layer(middleware::from_fn_with_state(expected, authenticate_lan))
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
