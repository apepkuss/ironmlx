//! Native Qwen Image 2.1 state and bounded serial admission.

use std::cell::OnceCell;
use std::sync::Arc;

use ironmlx_image::models::{QwenImage21GenerationConfig, QwenImage21Pipeline};
use mlx::Array;
use tokio::sync::Mutex;

use super::diffusion_execution::{
    DiffusionGemmaLane, DiffusionGemmaLaneError, DiffusionGemmaLaneGuard,
};

pub const QWEN_IMAGE_QUEUE_CAPACITY: usize = 2;

#[derive(Clone)]
pub struct QwenImageRuntime {
    pub pipeline: Arc<Mutex<QwenImage21Pipeline>>,
    pub model_id: String,
    pub model_weight_bytes: usize,
    pub lane: Arc<DiffusionGemmaLane>,
    pub runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
}

#[derive(Debug, Clone)]
pub struct QwenImageGenerateRequest {
    pub prompt: String,
    pub config: QwenImage21GenerationConfig,
    pub condition_image: Option<Vec<u8>>,
}

pub struct AdmittedQwenImageRequest {
    state: QwenImageRuntime,
    guard: DiffusionGemmaLaneGuard,
    request: QwenImageGenerateRequest,
}

pub struct QwenImageExecution<'a> {
    state: &'a QwenImageRuntime,
    pipeline: OnceCell<tokio::sync::MutexGuard<'a, QwenImage21Pipeline>>,
    request: Option<QwenImageGenerateRequest>,
}

impl QwenImageRuntime {
    pub fn new(pipeline: QwenImage21Pipeline, model_id: String, model_weight_bytes: usize) -> Self {
        Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            model_id,
            model_weight_bytes,
            lane: Arc::new(DiffusionGemmaLane::new(QWEN_IMAGE_QUEUE_CAPACITY)),
            runtime_usage: Arc::new(
                crate::core::runtime_usage::ModelRuntimeUsageCounters::default(),
            ),
        }
    }

    pub async fn admit(
        self,
        request: QwenImageGenerateRequest,
    ) -> Result<AdmittedQwenImageRequest, DiffusionGemmaLaneError> {
        let guard = self.lane.clone().enter().await?;
        Ok(AdmittedQwenImageRequest {
            state: self,
            guard,
            request,
        })
    }
}

impl AdmittedQwenImageRequest {
    pub fn spawn<R: Send + 'static>(
        self,
        consume: impl FnOnce(QwenImageExecution<'_>) -> R + Send + 'static,
    ) -> tokio::task::JoinHandle<R> {
        tokio::task::spawn_blocking(move || {
            let _guard = self.guard;
            consume(QwenImageExecution {
                state: &self.state,
                pipeline: OnceCell::new(),
                request: Some(self.request),
            })
        })
    }
}

impl QwenImageExecution<'_> {
    pub fn generate(mut self) -> Result<Array, String> {
        let request = self
            .request
            .take()
            .ok_or_else(|| "Qwen Image 2.1 request already executed".to_owned())?;
        let pipeline = self
            .pipeline
            .get_or_init(|| self.state.pipeline.blocking_lock());
        let image = match request.condition_image {
            Some(condition_image) => pipeline.generate_conditioned(
                &request.prompt,
                &condition_image,
                request.config,
                || false,
            ),
            None => pipeline.generate(&request.prompt, request.config, || false),
        }
        .map_err(|error| format!("Qwen Image 2.1 generation failed: {error:#}"))?;
        self.state.runtime_usage.record_output_tokens(1);
        Ok(image)
    }
}
