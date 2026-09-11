//! Runtime diagnostics control for the App's loopback management channel.
use std::sync::{Mutex, OnceLock};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing_subscriber::{
    layer::SubscriberExt, reload, util::SubscriberInitExt, EnvFilter, Registry,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum LogLevel {
    #[serde(rename = "ALL", alias = "TRACE")]
    All,
    #[serde(rename = "DEBUG")]
    Debug,
    #[serde(rename = "INFO")]
    Info,
    #[serde(rename = "WARNING", alias = "WARN")]
    Warning,
    #[serde(rename = "ERROR")]
    Error,
}

impl LogLevel {
    fn filter(self) -> EnvFilter {
        let level = match self {
            Self::All => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warning => "warn",
            Self::Error => "error",
        };
        // Verbose diagnostics apply to our code, not dependency wire dumps.
        let dependencies = if self == Self::Error { "error" } else { "warn" };
        EnvFilter::new(format!("{dependencies},ironmlx={level}"))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LogSnapshot {
    pub level: Option<LogLevel>,
    pub revision: u64,
    pub process_id: u32,
}

struct Controller {
    handle: reload::Handle<EnvFilter, Registry>,
    snapshot: Mutex<LogSnapshot>,
}

static CONTROLLER: OnceLock<Controller> = OnceLock::new();

/// Preserve CLI RUST_LOG behavior. App launches provide a canonical level.
pub fn init() {
    let app_level = std::env::var("IRONMLX_LOG_LEVEL").ok().and_then(|value| {
        serde_json::from_value::<LogLevel>(serde_json::Value::String(value)).ok()
    });
    let filter = app_level.map(LogLevel::filter).unwrap_or_else(|| {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| LogLevel::Info.filter())
    });
    let (layer, handle) = reload::Layer::new(filter);
    tracing_subscriber::registry()
        .with(layer)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();
    let _ = CONTROLLER.set(Controller {
        handle,
        snapshot: Mutex::new(LogSnapshot {
            level: app_level,
            revision: 0,
            process_id: std::process::id(),
        }),
    });
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    level: LogLevel,
    expected_revision: u64,
    expected_process_id: u32,
}

fn unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, "log_control_unavailable").into_response()
}

async fn snapshot() -> Response {
    let Some(controller) = CONTROLLER.get() else {
        return unavailable();
    };
    match controller.snapshot.lock() {
        Ok(state) => Json(state.clone()).into_response(),
        Err(_) => unavailable(),
    }
}

async fn update(Json(request): Json<Update>) -> Response {
    let Some(controller) = CONTROLLER.get() else {
        return unavailable();
    };
    let Ok(mut state) = controller.snapshot.lock() else {
        return unavailable();
    };
    if state.revision != request.expected_revision
        || state.process_id != request.expected_process_id
    {
        return (StatusCode::CONFLICT, "log_control_changed").into_response();
    }
    if controller.handle.reload(request.level.filter()).is_err() {
        return unavailable();
    }
    state.level = Some(request.level);
    state.revision += 1;
    // No prompt, paths, credentials, or request payloads in these diagnostics.
    tracing::debug!(revision = state.revision, "runtime log filter applied");
    tracing::trace!(
        revision = state.revision,
        "runtime log filter trace enabled"
    );
    Json(state.clone()).into_response()
}

/// Merge only into the loopback listener; never into the LAN router.
pub(crate) fn router() -> Router {
    Router::new().route("/admin/api/log-level", get(snapshot).post(update))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::Arc;

    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn reload_changes_actual_output_without_replacing_subscriber() {
        let capture = Capture(Arc::new(Mutex::new(Vec::new())));
        let (layer, handle) = reload::Layer::new(LogLevel::Info.filter());
        let subscriber = tracing_subscriber::registry().with(layer).with(
            tracing_subscriber::fmt::layer()
                .without_time()
                .with_ansi(false)
                .with_writer(capture.clone()),
        );
        tracing::subscriber::with_default(subscriber, || {
            let emit = || {
                tracing::debug!("debug_marker");
                tracing::trace!("trace_marker");
                tracing::info!("info_marker");
                tracing::warn!("warn_marker");
                tracing::error!("error_marker");
            };
            for (level, debug, trace, info, warn) in [
                (LogLevel::Info, false, false, true, true),
                (LogLevel::Debug, true, false, true, true),
                (LogLevel::All, true, true, true, true),
                (LogLevel::Warning, false, false, false, true),
                (LogLevel::Error, false, false, false, false),
            ] {
                handle.reload(level.filter()).unwrap();
                capture.0.lock().unwrap().clear();
                emit();
                let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
                assert_eq!(output.contains("debug_marker"), debug);
                assert_eq!(output.contains("trace_marker"), trace);
                assert_eq!(output.contains("info_marker"), info);
                assert_eq!(output.contains("warn_marker"), warn);
                assert!(output.contains("error_marker"));
            }
        });
    }
}
