use tokio::sync::mpsc;

use ironmlx_core::generate::SamplerConfig;

use crate::engine::{EngineCommand, RequestOutput};

/// Async handle for submitting requests to the engine from HTTP handlers.
#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    pub fn new(cmd_tx: mpsc::Sender<EngineCommand>) -> Self {
        Self { cmd_tx }
    }

    /// Submit a request. Returns a receiver for streaming token outputs.
    pub async fn add_request(
        &self,
        request_id: String,
        prompt_token_ids: Vec<i32>,
        sampling_params: SamplerConfig,
        max_tokens: usize,
        eos_token_id: i32,
    ) -> mpsc::Receiver<RequestOutput> {
        let (token_tx, token_rx) = mpsc::channel(64);
        let _ = self
            .cmd_tx
            .send(EngineCommand::AddRequest {
                request_id,
                prompt_token_ids,
                sampling_params,
                max_tokens,
                eos_token_id,
                token_tx,
            })
            .await;
        token_rx
    }

    #[allow(dead_code)]
    pub async fn abort_request(&self, request_id: String) {
        let _ = self
            .cmd_tx
            .send(EngineCommand::AbortRequest { request_id })
            .await;
    }

    #[allow(dead_code)]
    pub async fn shutdown(&self) {
        let _ = self.cmd_tx.send(EngineCommand::Shutdown).await;
    }
}
