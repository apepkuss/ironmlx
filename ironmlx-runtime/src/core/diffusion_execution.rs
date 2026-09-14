//! Native DiffusionGemma model state and bounded serial admission.
//! Queue ownership and cancellation do not depend on an HTTP response type.

use ironmlx_lm::core::tokenizer::Tokenizer;
use ironmlx_lm::core::vision_input::VisionInputConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use {ironmlx_lm::models::DiffusionGemmaGenerationConfig, ironmlx_lm::models::DiffusionGemmaModel};

#[derive(Clone)]
pub struct DiffusionGemmaRuntime {
    pub model: Arc<Mutex<DiffusionGemmaModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub generation_config: DiffusionGemmaGenerationConfig,
    pub model_id: String,
    pub model_weight_bytes: usize,
    pub vision_input: VisionInputConfig,
    pub lane: Arc<DiffusionGemmaLane>,
    pub runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionGemmaLaneStats {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub queue_capacity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionGemmaLaneError {
    Overloaded,
    Closed,
}

pub struct DiffusionGemmaLane {
    queue_slots: Arc<Semaphore>,
    execution_slot: Arc<Semaphore>,
    active_requests: AtomicUsize,
    queued_requests: AtomicUsize,
    queue_capacity: usize,
}

impl DiffusionGemmaLane {
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue_slots: Arc::new(Semaphore::new(queue_capacity + 1)),
            execution_slot: Arc::new(Semaphore::new(1)),
            active_requests: AtomicUsize::new(0),
            queued_requests: AtomicUsize::new(0),
            queue_capacity,
        }
    }

    pub fn stats(&self) -> DiffusionGemmaLaneStats {
        DiffusionGemmaLaneStats {
            active_requests: self.active_requests.load(Ordering::SeqCst),
            queued_requests: self.queued_requests.load(Ordering::SeqCst),
            queue_capacity: self.queue_capacity,
        }
    }

    pub async fn enter(
        self: Arc<Self>,
    ) -> std::result::Result<DiffusionGemmaLaneGuard, DiffusionGemmaLaneError> {
        let queue_permit = match self.queue_slots.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => return Err(DiffusionGemmaLaneError::Overloaded),
            Err(TryAcquireError::Closed) => return Err(DiffusionGemmaLaneError::Closed),
        };
        self.queued_requests.fetch_add(1, Ordering::SeqCst);
        let queued = DiffusionGemmaQueuedSlot {
            lane: Arc::clone(&self),
            queue_permit: Some(queue_permit),
        };
        let execution_permit = self
            .execution_slot
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DiffusionGemmaLaneError::Closed)?;
        Ok(queued.promote(execution_permit))
    }
}

struct DiffusionGemmaQueuedSlot {
    lane: Arc<DiffusionGemmaLane>,
    queue_permit: Option<OwnedSemaphorePermit>,
}

impl DiffusionGemmaQueuedSlot {
    fn promote(mut self, execution_permit: OwnedSemaphorePermit) -> DiffusionGemmaLaneGuard {
        let queue_permit = self
            .queue_permit
            .take()
            .expect("queued slot must own its queue permit");
        self.lane.queued_requests.fetch_sub(1, Ordering::SeqCst);
        self.lane.active_requests.fetch_add(1, Ordering::SeqCst);
        DiffusionGemmaLaneGuard {
            lane: Arc::clone(&self.lane),
            _queue_permit: queue_permit,
            _execution_permit: execution_permit,
        }
    }
}

impl Drop for DiffusionGemmaQueuedSlot {
    fn drop(&mut self) {
        if self.queue_permit.is_some() {
            self.lane.queued_requests.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

pub struct DiffusionGemmaLaneGuard {
    lane: Arc<DiffusionGemmaLane>,
    _queue_permit: OwnedSemaphorePermit,
    _execution_permit: OwnedSemaphorePermit,
}

impl Drop for DiffusionGemmaLaneGuard {
    fn drop(&mut self) {
        self.lane.active_requests.fetch_sub(1, Ordering::SeqCst);
    }
}

const DEFAULT_DIFFUSION_GEMMA_QUEUE_CAPACITY: usize = 8;
pub fn build_diffusion_gemma_app_state(
    model: DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: DiffusionGemmaGenerationConfig,
    model_id: String,
    model_weight_bytes: usize,
    vision_input: VisionInputConfig,
) -> DiffusionGemmaRuntime {
    DiffusionGemmaRuntime {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        generation_config,
        model_id,
        model_weight_bytes,
        vision_input,
        lane: Arc::new(DiffusionGemmaLane::new(
            DEFAULT_DIFFUSION_GEMMA_QUEUE_CAPACITY,
        )),
        runtime_usage: Arc::new(crate::core::runtime_usage::ModelRuntimeUsageCounters::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn lane_tracks_active_queued_and_rejects_when_full() {
        let lane = Arc::new(DiffusionGemmaLane::new(1));
        let first = lane.clone().enter().await.expect("first request admitted");
        assert_eq!(
            lane.stats(),
            DiffusionGemmaLaneStats {
                active_requests: 1,
                queued_requests: 0,
                queue_capacity: 1,
            }
        );

        let second = {
            let lane = lane.clone();
            tokio::spawn(async move { lane.enter().await.expect("queued request admitted") })
        };

        for _ in 0..10 {
            if lane.stats().queued_requests == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(lane.stats().queued_requests, 1);
        assert!(matches!(
            lane.clone().enter().await,
            Err(DiffusionGemmaLaneError::Overloaded)
        ));

        drop(first);
        let second = second.await.expect("queued task joined");
        assert_eq!(lane.stats().active_requests, 1);
        drop(second);
        assert_eq!(
            lane.stats(),
            DiffusionGemmaLaneStats {
                active_requests: 0,
                queued_requests: 0,
                queue_capacity: 1,
            }
        );
    }

    #[tokio::test]
    async fn lane_releases_queued_count_when_waiter_is_cancelled() {
        let lane = Arc::new(DiffusionGemmaLane::new(1));
        let first = lane.clone().enter().await.expect("first request admitted");
        let waiter = {
            let lane = lane.clone();
            tokio::spawn(async move { lane.enter().await })
        };

        for _ in 0..10 {
            if lane.stats().queued_requests == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(lane.stats().queued_requests, 1);
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(lane.stats().queued_requests, 0);
        drop(first);
        assert_eq!(lane.stats().active_requests, 0);
    }
}

use ironmlx_lm::core::constrained::ConstraintPlan;
use ironmlx_lm::models::diffusion_gemma::{DiffusionGemmaEventSink, DiffusionGemmaGenerateEvent};
use mlx::Array;
use std::cell::{OnceCell, RefCell};

pub struct DiffusionGenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub pixel_values: Option<Vec<Array>>,
    pub image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    pub image_token_id: i32,
    pub constraint: Option<ConstraintPlan>,
    pub skip_special_tokens: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_generation_with_events(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    generation_config: &DiffusionGemmaGenerationConfig,
    request: DiffusionGenerateRequest,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    emit: ironmlx_lm::models::diffusion_gemma::DiffusionGemmaEventSink<'_>,
) -> std::result::Result<(), String> {
    let DiffusionGenerateRequest {
        prompt_ids,
        pixel_values,
        image_grid_thw,
        image_token_id,
        constraint,
        skip_special_tokens,
    } = request;
    match (pixel_values.as_deref(), image_grid_thw.as_deref()) {
        (Some(pixel_values), Some(image_grid_thw)) => match constraint.as_ref() {
            Some(constraint) => {
                ironmlx_lm::models::diffusion_gemma::generate_image_text_with_events_constrained(
                    model,
                    tokenizer,
                    &prompt_ids,
                    pixel_values,
                    image_grid_thw,
                    image_token_id,
                    generation_config,
                    max_tokens,
                    temperature,
                    seed,
                    constraint,
                    skip_special_tokens,
                    emit,
                )
                .map_err(|e| e.to_string())
            }
            None => ironmlx_lm::models::diffusion_gemma::generate_image_text_with_events(
                model,
                tokenizer,
                &prompt_ids,
                pixel_values,
                image_grid_thw,
                image_token_id,
                generation_config,
                max_tokens,
                temperature,
                seed,
                skip_special_tokens,
                emit,
            )
            .map_err(|e| e.to_string()),
        },
        (None, None) => match constraint.as_ref() {
            Some(constraint) => {
                ironmlx_lm::models::diffusion_gemma::generate_text_with_events_constrained(
                    model,
                    tokenizer,
                    &prompt_ids,
                    generation_config,
                    max_tokens,
                    temperature,
                    seed,
                    constraint,
                    skip_special_tokens,
                    emit,
                )
                .map_err(|e| e.to_string())
            }
            None => ironmlx_lm::models::diffusion_gemma::generate_text_with_events(
                model,
                tokenizer,
                &prompt_ids,
                generation_config,
                max_tokens,
                temperature,
                seed,
                skip_special_tokens,
                emit,
            )
            .map_err(|e| e.to_string()),
        },
        (Some(_), None) | (None, Some(_)) => {
            Err("DiffusionGemma image request missing image tensors or grids".to_string())
        }
    }
}

/// Sampling options for native block diffusion generation.
#[derive(Clone, Copy)]
pub struct DiffusionGenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: Option<u64>,
}

pub struct AdmittedDiffusionRequest {
    state: DiffusionGemmaRuntime,
    guard: DiffusionGemmaLaneGuard,
    request: DiffusionGenerateRequest,
}

/// Restricted event producer. Model locks and admission guards remain runtime-owned.
pub struct DiffusionExecution<'a> {
    state: &'a DiffusionGemmaRuntime,
    model: OnceCell<tokio::sync::MutexGuard<'a, DiffusionGemmaModel>>,
    request: RefCell<Option<DiffusionGenerateRequest>>,
}

impl DiffusionGemmaRuntime {
    pub async fn admit(
        self,
        request: DiffusionGenerateRequest,
    ) -> Result<AdmittedDiffusionRequest, DiffusionGemmaLaneError> {
        let guard = self.lane.clone().enter().await?;
        self.runtime_usage
            .record_input_tokens(request.prompt_ids.len() as u64);
        Ok(AdmittedDiffusionRequest {
            state: self,
            guard,
            request,
        })
    }
}

impl AdmittedDiffusionRequest {
    pub fn spawn<R: Send + 'static>(
        self,
        consume: impl FnOnce(DiffusionExecution<'_>) -> R + Send + 'static,
    ) -> tokio::task::JoinHandle<R> {
        tokio::task::spawn_blocking(move || {
            let _guard = self.guard;
            consume(DiffusionExecution {
                state: &self.state,
                model: OnceCell::new(),
                request: RefCell::new(Some(self.request)),
            })
        })
    }
}

impl DiffusionExecution<'_> {
    /// Called after any initial transport frame, at the model initialization boundary.
    pub fn initialize(&self) {
        self.model.get_or_init(|| self.state.model.blocking_lock());
    }

    pub fn generate(
        &self,
        options: DiffusionGenerationOptions,
        emit: DiffusionGemmaEventSink<'_>,
    ) -> Result<(), String> {
        self.initialize();
        let request = self
            .request
            .borrow_mut()
            .take()
            .ok_or_else(|| "DiffusionGemma request already executed".to_owned())?;
        run_generation_with_events(
            self.model.get().expect("model initialized"),
            &self.state.tokenizer,
            &self.state.generation_config,
            request,
            options.max_tokens,
            options.temperature,
            options.seed,
            emit,
        )
    }

    pub fn collect(
        &self,
        options: DiffusionGenerationOptions,
    ) -> Result<Vec<DiffusionGemmaGenerateEvent>, String> {
        let mut events = Vec::new();
        self.generate(options, &mut |event| {
            events.push(event);
            Ok(true)
        })?;
        mlx::transforms::clear_cache();
        Ok(events)
    }

    pub fn record_output_tokens(&self, count: u64) {
        self.state.runtime_usage.record_output_tokens(count);
    }
}
