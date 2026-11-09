use axum::{
    http::{header, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use ironmlx_runtime::core::load_error::{classify_load_error, LoadErrorKind};
use ironmlx_runtime::core::scheduler::{GenerationError, SchedulerError};

pub(crate) const RETRY_AFTER_SECONDS: u64 = 5;

/// Appended when a request's load left the model failed: requests do not
/// load a failed model again, so retrying the request cannot help.
pub(crate) const RELOAD_REQUIRED_MESSAGE: &str = "This model will not be loaded again automatically; load it again from the Dashboard or the load API after resolving this.";

/// HTTP status for a failed model load, shared by the admin load API and
/// inference requests that load a model lazily.
pub(crate) fn load_error_status(kind: LoadErrorKind) -> StatusCode {
    match kind {
        LoadErrorKind::Invalid => StatusCode::BAD_REQUEST,
        LoadErrorKind::Conflict => StatusCode::CONFLICT,
        LoadErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ApiProtocol {
    OpenAi,
    Anthropic,
    SystemOne,
}

impl ApiProtocol {
    pub(crate) fn from_path(path: &str) -> Self {
        if path == "/v1/messages" {
            Self::Anthropic
        } else if path == "/v1/systemone" {
            Self::SystemOne
        } else {
            Self::OpenAi
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApiErrorClass {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    Conflict,
    RequestTooLarge,
    Overloaded,
    Internal,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TokenCapacityDetails {
    required_total_tokens: usize,
    input_tokens: usize,
    requested_max_output_tokens: usize,
    server_max_context_tokens: usize,
    max_allowed_output_tokens: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    class: ApiErrorClass,
    code: &'static str,
    message: String,
    param: Option<String>,
    retry_after_seconds: Option<u64>,
    token_capacity: Option<TokenCapacityDetails>,
}

impl ApiError {
    pub(crate) fn from_status(
        status: StatusCode,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        let class = match status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
                ApiErrorClass::InvalidRequest
            }
            StatusCode::UNAUTHORIZED => ApiErrorClass::Authentication,
            StatusCode::FORBIDDEN => ApiErrorClass::Permission,
            StatusCode::NOT_FOUND => ApiErrorClass::NotFound,
            StatusCode::CONFLICT => ApiErrorClass::Conflict,
            StatusCode::PAYLOAD_TOO_LARGE => ApiErrorClass::RequestTooLarge,
            StatusCode::SERVICE_UNAVAILABLE => ApiErrorClass::Overloaded,
            _ if status.is_server_error() => ApiErrorClass::Internal,
            _ => ApiErrorClass::InvalidRequest,
        };
        Self {
            status,
            class,
            code,
            message: message.into(),
            param: None,
            retry_after_seconds: (status == StatusCode::SERVICE_UNAVAILABLE)
                .then_some(RETRY_AFTER_SECONDS),
            token_capacity: None,
        }
    }

    pub(crate) fn with_param(mut self, param: Option<String>) -> Self {
        self.param = param;
        self
    }

    pub(crate) fn invalid_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::from_status(StatusCode::BAD_REQUEST, code, message)
    }

    pub(crate) fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self::from_status(StatusCode::INTERNAL_SERVER_ERROR, code, message)
    }

    pub(crate) fn service_unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self::from_status(StatusCode::SERVICE_UNAVAILABLE, code, message)
    }

    pub(crate) fn request_token_capacity(error: &SchedulerError) -> Self {
        let SchedulerError::RequestTooLarge {
            required_total_tokens,
            input_tokens,
            requested_max_output_tokens,
            server_max_context_tokens,
            max_allowed_output_tokens,
        } = error
        else {
            unreachable!("request_token_capacity requires SchedulerError::RequestTooLarge")
        };
        let mut api_error = Self::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_token_capacity_exceeded",
            error.to_string(),
        );
        api_error.token_capacity = Some(TokenCapacityDetails {
            required_total_tokens: *required_total_tokens,
            input_tokens: *input_tokens,
            requested_max_output_tokens: *requested_max_output_tokens,
            server_max_context_tokens: *server_max_context_tokens,
            max_allowed_output_tokens: *max_allowed_output_tokens,
        });
        api_error
    }

    pub(crate) fn scheduler_admission(error: anyhow::Error) -> Self {
        let message = format!("{error:#}");
        if error
            .downcast_ref::<ironmlx_runtime::core::dflash2::DFlash2RequestError>()
            .is_some()
        {
            return Self::invalid_request("dflash2_request_rejected", message);
        }
        let scheduler_error = error.downcast_ref::<SchedulerError>().or_else(|| {
            error
                .downcast_ref::<GenerationError>()
                .and_then(|error| error.scheduler_error.as_ref())
        });
        match scheduler_error {
            Some(SchedulerError::QueueFull { .. }) => {
                Self::service_unavailable("scheduler_queue_full", message)
            }
            Some(error @ SchedulerError::RequestTooLarge { .. }) => {
                let mut api_error = Self::request_token_capacity(error);
                api_error.message = message;
                api_error
            }
            Some(SchedulerError::MemoryBudgetExceeded { .. }) => {
                Self::service_unavailable("memory_budget_exceeded", message)
            }
            Some(SchedulerError::MemoryPressure { .. }) => {
                Self::service_unavailable("memory_pressure", message)
            }
            Some(SchedulerError::PrefillPeakUnsafe { .. }) => {
                Self::service_unavailable("prefill_peak_unsafe", message)
            }
            Some(SchedulerError::VisionPrefillPeakUnsafe { .. }) => {
                Self::service_unavailable("vision_prefill_peak_unsafe", message)
            }
            Some(SchedulerError::ColdMaterializationUnsafe { .. }) => {
                Self::service_unavailable("cold_materialization_unsafe", message)
            }
            Some(SchedulerError::StoreBackpressure { .. }) => {
                Self::service_unavailable("prefix_store_backpressure", message)
            }
            None if error.is::<GenerationError>() => Self::internal("generation_error", message),
            None => Self::invalid_request("scheduler_rejected", message),
        }
    }

    pub(crate) fn generation(error: anyhow::Error) -> Self {
        if error.is::<SchedulerError>() || error.is::<GenerationError>() {
            Self::scheduler_admission(error)
        } else {
            Self::internal("generation_error", format!("{error:#}"))
        }
    }

    pub(crate) fn engine_resolution(error: anyhow::Error) -> Self {
        if error.is::<super::audio::ModelTaskMismatch>() {
            return Self::invalid_request("model_task_mismatch", error.to_string());
        }
        let message = format!("{error:#}");
        if let Some(registry) =
            error.downcast_ref::<ironmlx_runtime::core::engine_pool::EngineRegistryError>()
        {
            return match registry {
                ironmlx_runtime::core::engine_pool::EngineRegistryError::UnknownModel {
                    ..
                }
                | ironmlx_runtime::core::engine_pool::EngineRegistryError::ModelDisabled {
                    ..
                } => Self::from_status(StatusCode::NOT_FOUND, "model_not_found", message),
                ironmlx_runtime::core::engine_pool::EngineRegistryError::AmbiguousDefault => {
                    Self::invalid_request("model_required", message)
                }
                _ => Self::invalid_request("engine_pool_invalid_configuration", message),
            };
        }
        // A lazy load the request started or joined: answer like the admin
        // load API does for the same failure.
        let classified = classify_load_error(&error);
        let mut api_error = match classified.code {
            Some(code) => {
                Self::from_status(load_error_status(classified.kind), code, classified.message)
            }
            None => Self::service_unavailable("engine_unavailable", message),
        };
        // Requests never load a failed model again, so retrying cannot help
        // until someone loads it from the admin or load API.
        if error
            .downcast_ref::<ironmlx_runtime::core::load_error::RequestLoadFailed>()
            .is_some_and(|request| request.slot_failed)
        {
            api_error.message = format!("{} {RELOAD_REQUIRED_MESSAGE}", api_error.message);
            api_error.retry_after_seconds = None;
        }
        api_error
    }

    pub(crate) fn into_response(self, protocol: ApiProtocol) -> Response {
        let mut response = match protocol {
            ApiProtocol::OpenAi => self.openai_response(),
            ApiProtocol::Anthropic => self.anthropic_response(),
            ApiProtocol::SystemOne => (
                self.status,
                Json(serde_json::json!({"detail": self.message})),
            )
                .into_response(),
        };
        if let Some(seconds) = self.retry_after_seconds {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                HeaderValue::from_str(&seconds.to_string())
                    .expect("fixed Retry-After seconds are a valid header value"),
            );
        }
        response
    }

    pub(crate) fn openai_payload(&self) -> serde_json::Value {
        let kind = match self.class {
            ApiErrorClass::Authentication => "authentication_error",
            ApiErrorClass::Overloaded | ApiErrorClass::Internal => "server_error",
            _ => "invalid_request_error",
        };
        serde_json::to_value(OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: &self.message,
                kind,
                param: self.param.as_deref(),
                code: self.code,
                details: self.token_capacity.as_ref(),
            },
        })
        .expect("API error envelope is serializable")
    }

    fn openai_response(&self) -> Response {
        (self.status, Json(self.openai_payload())).into_response()
    }

    pub(crate) fn anthropic_payload(&self, request_id: &str) -> serde_json::Value {
        let kind = match self.class {
            ApiErrorClass::InvalidRequest => "invalid_request_error",
            ApiErrorClass::Authentication => "authentication_error",
            ApiErrorClass::Permission => "permission_error",
            ApiErrorClass::NotFound => "not_found_error",
            ApiErrorClass::Conflict => "conflict_error",
            ApiErrorClass::RequestTooLarge => "request_too_large",
            ApiErrorClass::Overloaded => "overloaded_error",
            ApiErrorClass::Internal => "api_error",
        };
        serde_json::to_value(AnthropicErrorEnvelope {
            kind: "error",
            error: AnthropicErrorBody {
                kind,
                message: &self.message,
                code: self.code,
                details: self.token_capacity.as_ref(),
            },
            request_id,
        })
        .expect("API error envelope is serializable")
    }

    pub(crate) fn code(&self) -> &'static str {
        self.code
    }
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    fn anthropic_response(&self) -> Response {
        let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let mut response = (self.status, Json(self.anthropic_payload(&request_id))).into_response();
        response.headers_mut().insert(
            HeaderName::from_static("request-id"),
            HeaderValue::from_str(&request_id).expect("generated request id is a valid header"),
        );
        response
    }
}

#[derive(Serialize)]
struct OpenAiErrorEnvelope<'a> {
    error: OpenAiErrorBody<'a>,
}

#[derive(Serialize)]
struct OpenAiErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<&'a str>,
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<&'a TokenCapacityDetails>,
}

#[derive(Serialize)]
struct AnthropicErrorEnvelope<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    error: AnthropicErrorBody<'a>,
    request_id: &'a str,
}

#[derive(Serialize)]
struct AnthropicErrorBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    message: &'a str,
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<&'a TokenCapacityDetails>,
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use serde_json::Value;

    use super::*;

    async fn json(response: Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn overload_renders_protocol_native_json_and_retry_after() {
        let openai = ApiError::service_unavailable("scheduler_queue_full", "busy")
            .into_response(ApiProtocol::OpenAi);
        assert_eq!(openai.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(openai.headers()[header::RETRY_AFTER], "5");
        let body = json(openai).await;
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "scheduler_queue_full");

        let anthropic = ApiError::service_unavailable("scheduler_queue_full", "busy")
            .into_response(ApiProtocol::Anthropic);
        assert_eq!(anthropic.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(anthropic.headers()[header::RETRY_AFTER], "5");
        let request_id = anthropic.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body = json(anthropic).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "overloaded_error");
        assert_eq!(body["error"]["code"], "scheduler_queue_full");
        assert_eq!(body["request_id"], request_id);
    }

    #[tokio::test]
    async fn body_and_token_413_have_distinct_stable_codes() {
        let body_limit = ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "body too large",
        )
        .into_response(ApiProtocol::OpenAi);
        let body_limit = json(body_limit).await;
        assert_eq!(body_limit["error"]["code"], "request_body_too_large");

        let scheduler = SchedulerError::RequestTooLarge {
            required_total_tokens: 65,
            input_tokens: 64,
            requested_max_output_tokens: 1,
            server_max_context_tokens: 32,
            max_allowed_output_tokens: 0,
        };
        let token_limit =
            ApiError::request_token_capacity(&scheduler).into_response(ApiProtocol::OpenAi);
        let token_limit = json(token_limit).await;
        assert_eq!(
            token_limit["error"]["code"],
            "request_token_capacity_exceeded"
        );
        assert_eq!(token_limit["error"]["details"]["input_tokens"], 64);
    }

    #[tokio::test]
    async fn dflash2_policy_rejection_has_stable_code() {
        let response = ApiError::scheduler_admission(anyhow::Error::new(
            ironmlx_runtime::core::dflash2::DFlash2RequestError::VisionUnsupported,
        ))
        .into_response(ApiProtocol::OpenAi);

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json(response).await;
        assert_eq!(body["error"]["code"], "dflash2_request_rejected");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("text-only"));
    }

    #[test]
    fn protocol_is_selected_from_public_request_path() {
        assert_eq!(
            ApiProtocol::from_path("/v1/messages"),
            ApiProtocol::Anthropic
        );
        assert_eq!(
            ApiProtocol::from_path("/v1/chat/completions"),
            ApiProtocol::OpenAi
        );
        assert_eq!(ApiProtocol::from_path("/v1/responses"), ApiProtocol::OpenAi);
        assert_eq!(ApiProtocol::from_path("/v1/decisions"), ApiProtocol::OpenAi);
    }

    #[tokio::test]
    async fn engine_resolution_errors_render_identically_for_routing_topologies() {
        let openai = ApiError::engine_resolution(
            ironmlx_runtime::core::engine_pool::EngineRegistryError::UnknownModel {
                id: "missing".to_owned(),
            }
            .into(),
        )
        .into_response(ApiProtocol::OpenAi);
        assert_eq!(openai.status(), StatusCode::NOT_FOUND);
        let body = json(openai).await;
        assert_eq!(body["error"]["code"], "model_not_found");

        let anthropic = ApiError::engine_resolution(anyhow::anyhow!("engine failed"))
            .into_response(ApiProtocol::Anthropic);
        assert_eq!(anthropic.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(anthropic.headers()[header::RETRY_AFTER], "5");
        let body = json(anthropic).await;
        assert_eq!(body["error"]["type"], "overloaded_error");
        assert_eq!(body["error"]["code"], "engine_unavailable");
    }

    #[tokio::test]
    async fn cancelled_lazy_load_is_a_conflict_without_retry_after() {
        use ironmlx_runtime::core::load_error::{LOAD_CANCELLED_CODE, LOAD_CANCELLED_MESSAGE};
        let cancelled = || {
            anyhow::Error::from(ironmlx_runtime::core::load_registry::LoadCancelled)
                .context("loading engine model `alpha`")
        };

        let openai = ApiError::engine_resolution(cancelled()).into_response(ApiProtocol::OpenAi);
        assert_eq!(openai.status(), StatusCode::CONFLICT);
        assert!(openai.headers().get(header::RETRY_AFTER).is_none());
        let body = json(openai).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], LOAD_CANCELLED_CODE);
        assert_eq!(body["error"]["message"], LOAD_CANCELLED_MESSAGE);

        let anthropic =
            ApiError::engine_resolution(cancelled()).into_response(ApiProtocol::Anthropic);
        assert_eq!(anthropic.status(), StatusCode::CONFLICT);
        assert!(anthropic.headers().get(header::RETRY_AFTER).is_none());
        let body = json(anthropic).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "conflict_error");
        assert_eq!(body["error"]["code"], LOAD_CANCELLED_CODE);
    }

    #[tokio::test]
    async fn misconfigured_lazy_load_is_a_bad_request_with_its_code() {
        use ironmlx_runtime::core::load_error::MTP_INCOMPATIBLE_MESSAGE;
        use ironmlx_runtime::core::model_management::{
            ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        };
        use ironmlx_runtime::core::model_validation::MTP_INCOMPATIBLE_CODE;

        let capability = anyhow::Error::from(ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        })
        .context("loading engine model `alpha`");
        let response = ApiError::engine_resolution(capability).into_response(ApiProtocol::OpenAi);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
        let body = json(response).await;
        assert_eq!(body["error"]["code"], DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE);
        assert_eq!(
            body["error"]["message"],
            DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE
        );

        let mtp = anyhow::anyhow!("{MTP_INCOMPATIBLE_CODE}: MTP head shape mismatch")
            .context("loading engine model `alpha`");
        let response = ApiError::engine_resolution(mtp).into_response(ApiProtocol::Anthropic);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], MTP_INCOMPATIBLE_CODE);
        assert_eq!(body["error"]["message"], MTP_INCOMPATIBLE_MESSAGE);
    }

    #[tokio::test]
    async fn resource_refused_lazy_load_stays_503_with_its_code() {
        use ironmlx_runtime::core::load_error::{
            MAX_LOADED_MODELS_REACHED_CODE, MAX_LOADED_MODELS_REACHED_MESSAGE,
        };
        let full = anyhow::anyhow!("engine pool capacity reached: 1 loaded, 1 pinned")
            .context("loading engine model `beta`");
        let response = ApiError::engine_resolution(full).into_response(ApiProtocol::OpenAi);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "5");
        let body = json(response).await;
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], MAX_LOADED_MODELS_REACHED_CODE);
        assert_eq!(body["error"]["message"], MAX_LOADED_MODELS_REACHED_MESSAGE);
    }

    /// `error` as a request's lazy load ends with it: classified before the
    /// model id is added.
    fn request_load_failed(
        model_id: &str,
        error: anyhow::Error,
        slot_failed: bool,
    ) -> anyhow::Error {
        let classified = classify_load_error(&error);
        error.context(ironmlx_runtime::core::load_error::RequestLoadFailed {
            model_id: model_id.to_owned(),
            classified,
            slot_failed,
        })
    }

    #[tokio::test]
    async fn load_failure_that_left_the_model_failed_asks_for_a_reload() {
        use ironmlx_runtime::core::load_error::{
            GPU_MEMORY_INSUFFICIENT_CODE, GPU_MEMORY_INSUFFICIENT_MESSAGE,
        };
        use ironmlx_runtime::core::model_management::{
            ModelCapabilityError, DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        };
        let oom = || anyhow::anyhow!("Metal OOM: command buffer failed");

        for protocol in [ApiProtocol::OpenAi, ApiProtocol::Anthropic] {
            let response = ApiError::engine_resolution(request_load_failed("alpha", oom(), true))
                .into_response(protocol);
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(response.headers().get(header::RETRY_AFTER).is_none());
            let body = json(response).await;
            assert_eq!(body["error"]["code"], GPU_MEMORY_INSUFFICIENT_CODE);
            assert_eq!(
                body["error"]["message"],
                format!("{GPU_MEMORY_INSUFFICIENT_MESSAGE} {RELOAD_REQUIRED_MESSAGE}")
            );
        }

        let capability = anyhow::Error::from(ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        });
        let response = ApiError::engine_resolution(request_load_failed("alpha", capability, true))
            .into_response(ApiProtocol::OpenAi);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json(response).await;
        assert_eq!(
            body["error"]["message"],
            format!("{DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE} {RELOAD_REQUIRED_MESSAGE}")
        );

        // A refusal that left the model unloaded is retried by the next
        // request, so the answer keeps Retry-After and its message.
        let response = ApiError::engine_resolution(request_load_failed("alpha", oom(), false))
            .into_response(ApiProtocol::OpenAi);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "5");
        let body = json(response).await;
        assert_eq!(body["error"]["message"], GPU_MEMORY_INSUFFICIENT_MESSAGE);
    }

    #[tokio::test]
    async fn request_load_failure_is_classified_without_the_model_id() {
        let error = request_load_failed(
            "foo-oom",
            anyhow::anyhow!("unsupported tensor layout"),
            false,
        );
        let response = ApiError::engine_resolution(error).into_response(ApiProtocol::OpenAi);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "5");
        let body = json(response).await;
        assert_eq!(body["error"]["code"], "engine_unavailable");
        assert_eq!(
            body["error"]["message"],
            "loading engine model `foo-oom`: unsupported tensor layout"
        );
    }

    #[tokio::test]
    async fn unclassified_failure_that_left_the_model_failed_asks_for_a_reload() {
        for protocol in [ApiProtocol::OpenAi, ApiProtocol::Anthropic] {
            let error =
                request_load_failed("alpha", anyhow::anyhow!("unsupported tensor layout"), true);
            let response = ApiError::engine_resolution(error).into_response(protocol);
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(response.headers().get(header::RETRY_AFTER).is_none());
            let body = json(response).await;
            assert_eq!(body["error"]["code"], "engine_unavailable");
            assert_eq!(
                body["error"]["message"],
                format!(
                    "loading engine model `alpha`: unsupported tensor layout {RELOAD_REQUIRED_MESSAGE}"
                )
            );
        }
    }
}
