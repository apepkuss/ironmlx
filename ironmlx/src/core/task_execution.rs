//! Native causal and DFlash2 task submission, admission errors and result receivers.
//! Dropping the admitted event receiver propagates cancellation to execution checkpoints.

use crate::core::{dflash2_actor, scheduler_actor};
use crate::Result;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::oneshot;

#[derive(Clone)]
pub enum RequestExecutionHandle {
    Scheduler(Arc<scheduler_actor::SchedulerActorHandle>),
    DFlash2(Arc<dflash2_actor::DFlash2ActorHandle>),
}

pub(crate) enum RequestAdmissionError {
    Rejected(anyhow::Error),
    Unavailable,
    ReplyLost,
}

impl RequestExecutionHandle {
    pub(crate) fn is_dflash2(&self) -> bool {
        matches!(self, Self::DFlash2(_))
    }

    pub(crate) fn active_and_queued(&self) -> (usize, usize) {
        let (active, queued) = match self {
            Self::Scheduler(handle) => (&handle.b_active, &handle.b_queued),
            Self::DFlash2(handle) => (&handle.b_active, &handle.b_queued),
        };
        (
            active.load(Ordering::Relaxed) as usize,
            queued.load(Ordering::Relaxed) as usize,
        )
    }

    pub(crate) async fn admit(
        &self,
        request: crate::core::generation_types::GenerateRequest,
    ) -> std::result::Result<scheduler_actor::AdmitReply, RequestAdmissionError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        match self {
            Self::Scheduler(handle) => handle
                .cmd_tx
                .send(scheduler_actor::SchedulerCommand::Admit { request, reply_tx })
                .await
                .map_err(|_| RequestAdmissionError::Unavailable)?,
            Self::DFlash2(handle) => match handle.enqueue(request, reply_tx) {
                Ok(()) => {}
                Err(dflash2_actor::DFlash2EnqueueError::QueueFull(error)) => {
                    return Err(RequestAdmissionError::Rejected(error));
                }
                Err(dflash2_actor::DFlash2EnqueueError::Unavailable) => {
                    return Err(RequestAdmissionError::Unavailable);
                }
            },
        }
        reply_rx
            .await
            .map_err(|_| RequestAdmissionError::ReplyLost)?
            .map_err(RequestAdmissionError::Rejected)
    }

    pub(crate) async fn clear_shared_prompt_lookup(&self) -> Result<usize> {
        match self {
            Self::Scheduler(handle) => handle.clear_shared_prompt_lookup().await,
            Self::DFlash2(_) => Ok(0),
        }
    }
}
