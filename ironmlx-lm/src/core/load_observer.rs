//! Weight-load progress and cancellation shared between a [`Loader`] and the
//! code that requested the load.
//!
//! [`Loader`]: crate::core::Loader

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Which weights of a load the current progress belongs to. MTP heads, Gemma4
/// drafters and DFlash2 drafts are all `Draft`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightComponent {
    Main,
    Draft,
}

/// Point-in-time view of an observer. `bytes_*` describe the component most
/// recently started with [`LoadObserver::begin_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadObserverSnapshot {
    pub component: Option<WeightComponent>,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub cancelled: bool,
}

/// Returned when a load stops because its observer was cancelled. Recognise
/// it with `anyhow::Error::downcast_ref::<LoadCancelled>()`.
#[derive(Debug, thiserror::Error)]
#[error("model load cancelled")]
pub struct LoadCancelled;

#[derive(Debug, Default)]
struct WeightProgress {
    component: Option<WeightComponent>,
    bytes_done: u64,
    bytes_total: u64,
}

/// Progress sink and cancellation flag for weight loading. Shared through an
/// `Arc`; the loader updates it from the loading thread while other threads
/// read snapshots or cancel.
#[derive(Debug, Default)]
pub struct LoadObserver {
    cancelled: AtomicBool,
    progress: Mutex<WeightProgress>,
    #[cfg(test)]
    test_hooks: test_hooks::TestHooks,
}

impl LoadObserver {
    /// An observer nobody else holds: it is never cancelled and its progress
    /// is never read.
    pub fn noop() -> Arc<Self> {
        Self::new()
    }

    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Start reporting `component`, resetting the done count to zero.
    pub fn begin_weights(&self, component: WeightComponent, total_bytes: u64) {
        let mut progress = self.progress();
        *progress = WeightProgress {
            component: Some(component),
            bytes_done: 0,
            bytes_total: total_bytes,
        };
    }

    /// Record `bytes` more of the current component as evaluated.
    pub fn add_weight_bytes(&self, bytes: u64) {
        let mut progress = self.progress();
        progress.bytes_done = progress.bytes_done.saturating_add(bytes);
        #[cfg(test)]
        self.test_hooks.on_progress(self, progress.bytes_done);
    }

    pub fn snapshot(&self) -> LoadObserverSnapshot {
        let progress = self.progress();
        LoadObserverSnapshot {
            component: progress.component,
            bytes_done: progress.bytes_done,
            bytes_total: progress.bytes_total,
            cancelled: self.is_cancelled(),
        }
    }

    /// `Err(LoadCancelled)` once the observer has been cancelled.
    pub(crate) fn check_cancelled(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            return Err(LoadCancelled.into());
        }
        Ok(())
    }

    fn progress(&self) -> std::sync::MutexGuard<'_, WeightProgress> {
        self.progress.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod test_hooks {
    use super::*;

    /// Lets loader tests see every progress update and cancel at a chunk
    /// boundary without racing a second thread.
    #[derive(Debug, Default)]
    pub(super) struct TestHooks {
        cancel_after_first_progress: AtomicBool,
        progress_log: Mutex<Vec<u64>>,
    }

    impl TestHooks {
        pub(super) fn on_progress(&self, observer: &LoadObserver, bytes_done: u64) {
            self.progress_log
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(bytes_done);
            if self.cancel_after_first_progress.load(Ordering::SeqCst) {
                observer.cancel();
            }
        }
    }

    impl LoadObserver {
        /// Cancel as soon as the first chunk of weights has been reported.
        pub(crate) fn cancel_after_first_progress(&self) {
            self.test_hooks
                .cancel_after_first_progress
                .store(true, Ordering::SeqCst);
        }

        /// `bytes_done` after every `add_weight_bytes` call, in order.
        pub(crate) fn progress_log(&self) -> Vec<u64> {
            self.test_hooks
                .progress_log
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }
}
