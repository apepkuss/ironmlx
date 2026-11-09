//! What a failed model load tells its client: a stable code and a
//! user-facing message, the same whether the load answered an admin request
//! directly or is read back later from the load registry.

use crate::core::load_registry::{LoadCancelled, LoadInProgress};
use crate::core::model_management::{ModelCapabilityError, MODEL_CAPABILITY_CODES};
use crate::core::model_validation::{
    MTP_BASE_MODEL_NOT_FOUND_CODE, MTP_INCOMPATIBLE_CODE, MTP_INVALID_CONFIG_CODE,
    MTP_INVALID_DRAFT_TOKENS_CODE, MTP_INVALID_MODEL_TYPE_CODE, MTP_MODEL_NOT_FOUND_CODE,
    MTP_UNSUPPORTED_ARCHITECTURE_CODE,
};

pub const LOAD_IN_PROGRESS_CODE: &str = "load_in_progress";
pub const LOAD_IN_PROGRESS_MESSAGE: &str =
    "A load of this model with different settings is already in progress.";
pub const LOAD_CANCELLED_CODE: &str = "load_cancelled";
pub const LOAD_CANCELLED_MESSAGE: &str = "The model load was cancelled.";
pub const GPU_MEMORY_INSUFFICIENT_CODE: &str = "gpu_memory_insufficient";
pub const GPU_MEMORY_INSUFFICIENT_MESSAGE: &str =
    "Not enough available GPU memory to safely load this model. Unload one or more unused loaded models to free GPU memory, then try again.";
pub const MAX_LOADED_MODELS_REACHED_CODE: &str = "max_loaded_models_reached";
pub const MAX_LOADED_MODELS_REACHED_MESSAGE: &str =
    "Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model.";
pub const MODEL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "model_memory_limit_exceeded";
pub const MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Model memory limit reached. Unload one or more loaded models, raise Memory Limit (Model Only), or set it to Auto, then try again.";
pub const TOTAL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "total_memory_limit_exceeded";
pub const TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Total memory limit reached. Unload one or more loaded models, raise Memory Limit (Total), or set it to Auto, then try again.";
pub const KV_MEMORY_BUDGET_EXCEEDED_CODE: &str = "kv_memory_budget_exceeded";
pub const KV_MEMORY_BUDGET_EXCEEDED_MESSAGE: &str =
    "KV cache memory budget exceeded. Enable Active KV offload with paged prefix cache for long context, or lower MAX CONTEXT TOKENS or Max Sequences in the Dashboard, then try again.";
pub const MTP_MODEL_DIR_REQUIRED_CODE: &str = "mtp_model_dir_required";
pub const MTP_MODEL_DIR_REQUIRED_MESSAGE: &str = "MTP draft tokens require an MTP model directory.";
pub const MTP_INCOMPATIBLE_MESSAGE: &str =
    "MTP weights are not compatible with this model. Load the model without MTP or choose matching MTP weights.";

const MTP_CODES: [&str; 8] = [
    MTP_MODEL_DIR_REQUIRED_CODE,
    MTP_BASE_MODEL_NOT_FOUND_CODE,
    MTP_MODEL_NOT_FOUND_CODE,
    MTP_UNSUPPORTED_ARCHITECTURE_CODE,
    MTP_INVALID_MODEL_TYPE_CODE,
    MTP_INVALID_CONFIG_CODE,
    MTP_INVALID_DRAFT_TOKENS_CODE,
    MTP_INCOMPATIBLE_CODE,
];

const RESOURCE_CODES: [&str; 5] = [
    MODEL_MEMORY_LIMIT_EXCEEDED_CODE,
    TOTAL_MEMORY_LIMIT_EXCEEDED_CODE,
    KV_MEMORY_BUDGET_EXCEEDED_CODE,
    GPU_MEMORY_INSUFFICIENT_CODE,
    MAX_LOADED_MODELS_REACHED_CODE,
];

/// Which kind of answer a failed load deserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadErrorKind {
    /// The request or the model's files are wrong.
    Invalid,
    /// Another load or a cancel decided the outcome.
    Conflict,
    /// The machine or the pool cannot take the model now.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedLoadError {
    pub kind: LoadErrorKind,
    /// `None` for errors without a known reason; `message` is then the raw
    /// error chain.
    pub code: Option<&'static str>,
    pub message: String,
}

/// A load this caller waited for failed; carries the reason that load
/// recorded so the waiter reports it the same way.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct LoadFailed {
    pub code: Option<&'static str>,
    pub message: String,
}

impl LoadFailed {
    /// Rebuild the error from a registry entry's `code` and `message`.
    pub fn from_recorded(code: Option<&str>, message: Option<&str>) -> Self {
        Self {
            code: code.and_then(known_code),
            message: message.unwrap_or("model load failed").to_string(),
        }
    }

    /// What callers that joined the load failing with `error` are told: its
    /// classified code and message, or the raw chain when unclassified.
    pub fn from_error(error: &anyhow::Error) -> Self {
        let classified = classify_load_error(error);
        Self {
            code: classified.code,
            message: classified.message,
        }
    }
}

/// Context on the error an inference request's lazy load ends with: the model
/// it loaded, the error classified before the model id was added (so the id
/// cannot change the class), and whether the model's slot was left failed.
#[derive(Debug, Clone, thiserror::Error)]
#[error("loading engine model `{model_id}`")]
pub struct RequestLoadFailed {
    pub model_id: String,
    pub classified: ClassifiedLoadError,
    /// Requests never load a failed slot again; only a load from the admin
    /// or load API does.
    pub slot_failed: bool,
}

/// Every code [`classify_load_error`] produces.
fn known_code(code: &str) -> Option<&'static str> {
    MTP_CODES
        .into_iter()
        .chain(RESOURCE_CODES)
        .chain(MODEL_CAPABILITY_CODES)
        .chain([LOAD_IN_PROGRESS_CODE, LOAD_CANCELLED_CODE])
        .find(|known| *known == code)
}

fn kind_for_code(code: &str) -> LoadErrorKind {
    if RESOURCE_CODES.contains(&code) {
        LoadErrorKind::Unavailable
    } else if code == LOAD_IN_PROGRESS_CODE || code == LOAD_CANCELLED_CODE {
        LoadErrorKind::Conflict
    } else {
        LoadErrorKind::Invalid
    }
}

fn classified(code: &'static str, message: impl Into<String>) -> ClassifiedLoadError {
    ClassifiedLoadError {
        kind: kind_for_code(code),
        code: Some(code),
        message: message.into(),
    }
}

/// The code and user-facing message for a load error.
pub fn classify_load_error(error: &anyhow::Error) -> ClassifiedLoadError {
    if let Some(request) = error.downcast_ref::<RequestLoadFailed>() {
        return request.classified.clone();
    }
    if let Some(failed) = error.downcast_ref::<LoadFailed>() {
        return match failed.code {
            Some(code) => classified(code, failed.message.clone()),
            None => ClassifiedLoadError {
                kind: LoadErrorKind::Invalid,
                code: None,
                message: failed.message.clone(),
            },
        };
    }
    if error.downcast_ref::<LoadInProgress>().is_some() {
        return classified(LOAD_IN_PROGRESS_CODE, LOAD_IN_PROGRESS_MESSAGE);
    }
    if error.downcast_ref::<LoadCancelled>().is_some() {
        return classified(LOAD_CANCELLED_CODE, LOAD_CANCELLED_MESSAGE);
    }
    if let Some(error) = error.downcast_ref::<ModelCapabilityError>() {
        return ClassifiedLoadError {
            kind: LoadErrorKind::Invalid,
            code: Some(error.code),
            message: error.message.to_string(),
        };
    }
    let message = format!("{error:#}");
    if let Some(code) = MTP_CODES.into_iter().find(|code| message.contains(code)) {
        let user_message = if code == MTP_MODEL_DIR_REQUIRED_CODE {
            MTP_MODEL_DIR_REQUIRED_MESSAGE
        } else {
            MTP_INCOMPATIBLE_MESSAGE
        };
        return classified(code, user_message);
    }
    let lower = message.to_ascii_lowercase();
    if lower.contains("engine pool model memory limit exceeded") {
        return classified(
            MODEL_MEMORY_LIMIT_EXCEEDED_CODE,
            MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE,
        );
    }
    if lower.contains("engine pool total memory limit exceeded") {
        return classified(
            TOTAL_MEMORY_LIMIT_EXCEEDED_CODE,
            TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE,
        );
    }
    if lower.contains("memory budget") {
        return classified(
            KV_MEMORY_BUDGET_EXCEEDED_CODE,
            KV_MEMORY_BUDGET_EXCEEDED_MESSAGE,
        );
    }
    if lower.contains("out of memory")
        || contains_word(&lower, "oom")
        || lower.contains("failed to allocate")
        || lower.contains("memory allocation")
    {
        return classified(
            GPU_MEMORY_INSUFFICIENT_CODE,
            GPU_MEMORY_INSUFFICIENT_MESSAGE,
        );
    }
    if lower.contains("engine pool capacity reached") {
        return classified(
            MAX_LOADED_MODELS_REACHED_CODE,
            MAX_LOADED_MODELS_REACHED_MESSAGE,
        );
    }
    ClassifiedLoadError {
        kind: LoadErrorKind::Invalid,
        code: None,
        message,
    }
}

/// `word` occurs in `text` as a whole token: bounded by non-alphanumeric
/// characters or the ends of `text`, so `oom` does not match `bloom`.
fn contains_word(text: &str, word: &str) -> bool {
    text.split(|character: char| !character.is_alphanumeric())
        .any(|token| token == word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_waiter_reports_the_recorded_reason() {
        let recorded = LoadFailed::from_recorded(
            Some(MAX_LOADED_MODELS_REACHED_CODE),
            Some(MAX_LOADED_MODELS_REACHED_MESSAGE),
        );
        let classified = classify_load_error(&anyhow::Error::from(recorded));
        assert_eq!(classified.kind, LoadErrorKind::Unavailable);
        assert_eq!(classified.code, Some(MAX_LOADED_MODELS_REACHED_CODE));
        assert_eq!(classified.message, MAX_LOADED_MODELS_REACHED_MESSAGE);

        let unknown = LoadFailed::from_recorded(Some("load_failed"), Some("disk on fire"));
        let classified = classify_load_error(&anyhow::Error::from(unknown));
        assert_eq!(classified.code, None);
        assert_eq!(classified.message, "disk on fire");
    }

    #[test]
    fn oom_is_matched_only_as_a_word() {
        for text in [
            "loading engine model `bigscience/bloom`: unsupported tensor layout",
            "no room left in the staging area",
            "zoom level is invalid",
        ] {
            let classified = classify_load_error(&anyhow::anyhow!("{text}"));
            assert_eq!(classified.code, None, "{text}");
            assert_eq!(classified.message, text);
        }
        for text in [
            "OOM",
            "Metal OOM: command buffer failed",
            "oom during allocation",
            "loading engine model `bloom`: out of memory",
        ] {
            let classified = classify_load_error(&anyhow::anyhow!("{text}"));
            assert_eq!(
                classified.code,
                Some(GPU_MEMORY_INSUFFICIENT_CODE),
                "{text}"
            );
            assert_eq!(classified.kind, LoadErrorKind::Unavailable, "{text}");
        }
    }
}
