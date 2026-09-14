//! Native DiffusionGemma model state and bounded serial admission.
//! Queue ownership and cancellation do not depend on an HTTP response type.

use crate::core::tokenizer::Tokenizer;
use crate::core::vision_input::VisionInputConfig;
use crate::models::{DiffusionGemmaGenerationConfig, DiffusionGemmaModel};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, TryAcquireError};

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
