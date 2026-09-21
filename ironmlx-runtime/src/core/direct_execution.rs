//! Direct generation worker ownership, independent of transport framing.
//!
//! The consumer receives a native event source, never a model lock or scheduler.
//! The worker retains the model and resource guards through consumer finalization.

use std::time::Instant;

use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::tokenizer::Tokenizer;
use ironmlx_lm::core::vision::DenseVlMethods;
use tokio::task::JoinHandle;

use super::engine_state::{begin_direct_request_memory, CausalEngine, DirectRequestMemoryGuard};
use super::generate::GenerationStream;
use super::generation_types::{GenerateEvent, GenerateRequest};
use super::runtime_usage::ModelRuntimeRequestTracker;
use crate::Result;

pub struct DirectGeneration<'a, M: Model + DenseVlMethods + Send + 'static> {
    source: DirectSource<'a, M>,
    memory: Option<DirectRequestMemoryGuard>,
    state: &'a CausalEngine<M>,
}

enum DirectSource<'a, M: Model> {
    Real(Box<GenerationStream<'a, M>>),
    #[cfg(feature = "test-support")]
    Injected(std::vec::IntoIter<GenerateEvent>),
}

impl<'a, M: Model + DenseVlMethods + Send + 'static> DirectGeneration<'a, M> {
    pub fn tokenizer(&self) -> &'a Tokenizer {
        &self.state.tokenizer
    }

    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        match &mut self.source {
            DirectSource::Real(stream) => stream.next_token(),
            #[cfg(feature = "test-support")]
            DirectSource::Injected(events) => Ok(events.next()),
        }
    }

    /// Commit cold materialization once the consumer has observed successful initialization.
    /// Returns false on subsequent calls so first-event accounting remains exactly once.
    pub fn commit_memory(&mut self) -> bool {
        if let Some(memory) = self.memory.take() {
            memory.commit();
            true
        } else {
            false
        }
    }

    pub fn record_request_started(
        &self,
        input_tokens: u32,
        started_at: Instant,
    ) -> ModelRuntimeRequestTracker {
        self.state.record_request_started(input_tokens, started_at)
    }
}

pub fn spawn_direct<M, R>(
    state: CausalEngine<M>,
    request: GenerateRequest,
    consume: impl FnOnce(Result<DirectGeneration<'_, M>>) -> R + Send + 'static,
) -> JoinHandle<R>
where
    M: Model + DenseVlMethods + Send + 'static,
    R: Send + 'static,
{
    spawn_direct_inner(
        state,
        request,
        #[cfg(feature = "test-support")]
        None,
        consume,
    )
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub fn spawn_direct_with_events<M, R>(
    state: CausalEngine<M>,
    request: GenerateRequest,
    events: Option<Vec<GenerateEvent>>,
    consume: impl FnOnce(Result<DirectGeneration<'_, M>>) -> R + Send + 'static,
) -> JoinHandle<R>
where
    M: Model + DenseVlMethods + Send + 'static,
    R: Send + 'static,
{
    spawn_direct_inner(state, request, events, consume)
}

fn spawn_direct_inner<M, R>(
    state: CausalEngine<M>,
    request: GenerateRequest,
    #[cfg(feature = "test-support")] events: Option<Vec<GenerateEvent>>,
    consume: impl FnOnce(Result<DirectGeneration<'_, M>>) -> R + Send + 'static,
) -> JoinHandle<R>
where
    M: Model + DenseVlMethods + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let model = state.model.blocking_lock();
        let initialized = (|| {
            let memory = begin_direct_request_memory(&state, &*model, &request)?;
            #[cfg(feature = "test-support")]
            let source = match events {
                Some(events) => DirectSource::Injected(events.into_iter()),
                None => DirectSource::Real(Box::new(GenerationStream::new(
                    &*model,
                    &state.tokenizer,
                    request,
                )?)),
            };
            #[cfg(not(feature = "test-support"))]
            let source = DirectSource::Real(Box::new(GenerationStream::new(
                &*model,
                &state.tokenizer,
                request,
            )?));
            Ok(DirectGeneration {
                source,
                memory: Some(memory),
                state: &state,
            })
        })();
        consume(initialized)
    })
}
