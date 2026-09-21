use thiserror::Error;
pub type Result<T> = std::result::Result<T, AudioError>;

/// Domain errors retain their classification across the runtime/transport boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioError {
    #[error("invalid {field}: {reason}")]
    InvalidInput { field: &'static str, reason: String },
    #[error("unsupported audio format")]
    UnsupportedFormat,
    #[error("missing resource: {component}")]
    ResourceMissing { component: String },
    #[error("resource {component}: {reason}")]
    ResourceMismatch { component: String, reason: String },
    #[error("capacity exceeded: {resource}")]
    CapacityExceeded { resource: &'static str },
    #[error("generation reached its limit without completing")]
    GenerationLimitExceeded,
    #[error("cancelled")]
    Cancelled,
    #[error("deadline exceeded")]
    DeadlineExceeded,
    #[error("inference failed: {reason}")]
    InferenceFailed { reason: String },
    #[error("session has already terminated")]
    InvalidSessionState,
    #[error("audio IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("MLX: {0}")]
    Mlx(#[from] mlx::Error),
}

pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> AudioError {
    AudioError::InvalidInput {
        field,
        reason: reason.into(),
    }
}
