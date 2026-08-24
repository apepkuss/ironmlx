use axum::{
    http::{header, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use crate::core::SchedulerError;

pub(crate) const RETRY_AFTER_SECONDS: u64 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ApiProtocol {
    OpenAi,
    Anthropic,
}

impl ApiProtocol {
    pub(crate) fn from_path(path: &str) -> Self {
        if path == "/v1/messages" {
            Self::Anthropic
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
            .downcast_ref::<crate::core::dflash2::DFlash2RequestError>()
            .is_some()
        {
            return Self::invalid_request("dflash2_request_rejected", message);
        }
        match error.downcast_ref::<SchedulerError>() {
            Some(SchedulerError::QueueFull { .. }) => {
                Self::service_unavailable("scheduler_queue_full", message)
            }
            Some(error @ SchedulerError::RequestTooLarge { .. }) => {
                Self::request_token_capacity(error)
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
            None => Self::invalid_request("scheduler_rejected", message),
        }
    }

    pub(crate) fn generation(error: anyhow::Error) -> Self {
        if error.downcast_ref::<SchedulerError>().is_some() {
            Self::scheduler_admission(error)
        } else {
            Self::internal("generation_error", format!("{error:#}"))
        }
    }

    pub(crate) fn engine_resolution(error: anyhow::Error) -> Self {
        let message = format!("{error:#}");
        if let Some(registry) = error.downcast_ref::<super::engine::EngineRegistryError>() {
            return match registry {
                super::engine::EngineRegistryError::UnknownModel { .. }
                | super::engine::EngineRegistryError::ModelDisabled { .. } => {
                    Self::from_status(StatusCode::NOT_FOUND, "model_not_found", message)
                }
                super::engine::EngineRegistryError::AmbiguousDefault => {
                    Self::invalid_request("model_required", message)
                }
                _ => Self::invalid_request("engine_pool_invalid_configuration", message),
            };
        }
        Self::service_unavailable("engine_unavailable", message)
    }

    pub(crate) fn into_response(self, protocol: ApiProtocol) -> Response {
        let mut response = match protocol {
            ApiProtocol::OpenAi => self.openai_response(),
            ApiProtocol::Anthropic => self.anthropic_response(),
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

    fn openai_response(&self) -> Response {
        let kind = match self.class {
            ApiErrorClass::Authentication => "authentication_error",
            ApiErrorClass::Overloaded | ApiErrorClass::Internal => "server_error",
            _ => "invalid_request_error",
        };
        (
            self.status,
            Json(OpenAiErrorEnvelope {
                error: OpenAiErrorBody {
                    message: &self.message,
                    kind,
                    param: self.param.as_deref(),
                    code: self.code,
                    details: self.token_capacity.as_ref(),
                },
            }),
        )
            .into_response()
    }

    fn anthropic_response(&self) -> Response {
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
        let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let mut response = (
            self.status,
            Json(AnthropicErrorEnvelope {
                kind: "error",
                error: AnthropicErrorBody {
                    kind,
                    message: &self.message,
                    code: self.code,
                    details: self.token_capacity.as_ref(),
                },
                request_id: &request_id,
            }),
        )
            .into_response();
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
            crate::core::dflash2::DFlash2RequestError::VisionUnsupported,
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
    }

    #[tokio::test]
    async fn engine_resolution_errors_render_identically_for_routing_topologies() {
        let openai = ApiError::engine_resolution(
            super::super::engine::EngineRegistryError::UnknownModel {
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
}
