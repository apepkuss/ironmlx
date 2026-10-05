//! Default-off DFlash2 token-id diagnostic.
//!
//! With `IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS=<path>` the DFlash2 actor appends
//! one JSON line per completed request to `<path>`: the prompt token ids, the
//! token ids actually published to the client, how the request ended and its
//! DFlash2 metrics. The HTTP API exposes only text, so this is the reference
//! for raw token-id comparisons between builds. Unset, nothing is recorded.

use std::io::Write;

const TOKEN_IDS_ENV: &str = "IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS";

pub(crate) struct DFlash2TokenIdDiagnostic {
    file: std::fs::File,
}

impl DFlash2TokenIdDiagnostic {
    pub(crate) fn from_env() -> Option<Self> {
        let path = std::env::var_os(TOKEN_IDS_ENV)?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => {
                tracing::warn!(
                    target: "ironmlx::dflash2",
                    path = %std::path::Path::new(&path).display(),
                    "DFlash2 token-id diagnostic enabled"
                );
                Some(Self { file })
            }
            Err(error) => {
                tracing::error!(%error, "DFlash2 token-id diagnostic file cannot be opened");
                None
            }
        }
    }

    pub(crate) fn record(
        &mut self,
        request_id: u64,
        prompt_token_ids: &[u32],
        published_token_ids: &[u32],
        cancelled: bool,
        failure: Option<&str>,
        metrics: serde_json::Value,
    ) {
        let line = serde_json::json!({
            "request_id": request_id,
            "prompt_token_ids": prompt_token_ids,
            "published_token_ids": published_token_ids,
            "cancelled": cancelled,
            "failure": failure,
            "metrics": metrics,
        });
        if let Err(error) = writeln!(self.file, "{line}").and_then(|()| self.file.flush()) {
            tracing::error!(%error, "DFlash2 token-id diagnostic write failed");
        }
    }
}
