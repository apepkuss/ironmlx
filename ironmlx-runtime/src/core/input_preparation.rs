//! Bounded native image preprocessing. A running worker owns its concurrency permit.

use ironmlx_lm::core::vision_input::{DecodedMessage, ExpandedVisionInputs, VisionInputConfig};
use tokio::sync::Semaphore;

static IMAGE_PREPROCESS_SEMAPHORE: Semaphore = Semaphore::const_new(2);

async fn run_bounded<T: Send + 'static>(
    semaphore: &'static Semaphore,
    work: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    let permit = semaphore
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("image preprocessing is unavailable"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
    .map_err(|error| anyhow::anyhow!("image preprocessing task failed: {error}"))?
}

pub async fn expand_decoded_messages_bounded(
    messages: Vec<DecodedMessage>,
    vision_input: VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    run_bounded(&IMAGE_PREPROCESS_SEMAPHORE, move || {
        ironmlx_lm::core::vision_input::expand_decoded_messages(messages, &vision_input)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_waiter_does_not_release_running_worker_permit() {
        static SEMAPHORE: Semaphore = Semaphore::const_new(1);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let waiter = tokio::spawn(run_bounded(&SEMAPHORE, move || {
            let _ = started_tx.send(());
            release_rx.recv()?;
            Ok(())
        }));
        started_rx.await.unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(SEMAPHORE.try_acquire().is_err());
        release_tx.send(()).unwrap();
        let permit = tokio::time::timeout(std::time::Duration::from_secs(5), SEMAPHORE.acquire())
            .await
            .expect("worker must return its permit")
            .unwrap();
        drop(permit);
        assert_eq!(SEMAPHORE.available_permits(), 1);
    }
}
