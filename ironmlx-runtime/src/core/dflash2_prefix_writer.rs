//! Background writer for the DFlash2 prefix store.
//!
//! A request hands over evaluated arrays and returns; one thread writes them.
//! The store's lock guards its index only: files are written and read
//! without it, so a lookup never waits for a large write.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use anyhow::Context;

use super::dflash2_prefix_store::{
    write_compaction, write_save, DFlash2DiskSave, DFlash2DiskStore,
};
use crate::Result;

/// Saves waiting to be written.
const MAX_PENDING_JOBS: usize = 8;
/// Bytes waiting to be written. One save larger than this is still accepted
/// when nothing else is waiting (a cold prompt's whole K/V).
const MAX_PENDING_BYTES: usize = 1 << 30;

#[derive(Debug, Default)]
struct Pending {
    jobs: usize,
    bytes: usize,
}

struct Job {
    save: DFlash2DiskSave,
    bytes: usize,
}

pub(crate) struct DFlash2PrefixWriter {
    store: Arc<Mutex<DFlash2DiskStore>>,
    sender: Option<Sender<Job>>,
    pending: Arc<(Mutex<Pending>, Condvar)>,
    /// Held by the worker while it writes one save; tests hold it to keep
    /// saves queued.
    #[cfg_attr(not(test), allow(dead_code))]
    gate: Arc<Mutex<()>>,
    max_jobs: usize,
    max_bytes: usize,
    worker: Option<JoinHandle<()>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl DFlash2PrefixWriter {
    pub(crate) fn spawn(store: DFlash2DiskStore) -> Result<Self> {
        Self::spawn_with_limits(store, MAX_PENDING_JOBS, MAX_PENDING_BYTES)
    }

    pub(crate) fn spawn_with_limits(
        store: DFlash2DiskStore,
        max_jobs: usize,
        max_bytes: usize,
    ) -> Result<Self> {
        let store = Arc::new(Mutex::new(store));
        let pending = Arc::new((Mutex::new(Pending::default()), Condvar::new()));
        let gate = Arc::new(Mutex::new(()));
        let (sender, receiver) = mpsc::channel::<Job>();
        let worker_store = Arc::clone(&store);
        let worker_pending = Arc::clone(&pending);
        let worker_gate = Arc::clone(&gate);
        let worker = std::thread::Builder::new()
            .name("ironmlx-dflash2-prefix-store".to_owned())
            .spawn(move || {
                // MLX stream registries are thread-local, and the arrays
                // written here were created on the request thread's stream.
                let device = mlx::Device::gpu(0);
                mlx::set_default_device(device);
                mlx::set_default_stream(mlx::default_stream(device));
                while let Ok(job) = receiver.recv() {
                    {
                        let _writing = lock(&worker_gate);
                        if let Err(error) = write_job(&worker_store, &job.save) {
                            lock(&worker_store).note_failed();
                            tracing::warn!(
                                tokens = job.save.token_ids.len(),
                                bytes = job.bytes,
                                error = %error,
                                "DFlash2 prefix artifact not written to disk"
                            );
                        }
                    }
                    drop(job.save);
                    let (state, idle) = &*worker_pending;
                    let mut state = lock(state);
                    state.jobs = state.jobs.saturating_sub(1);
                    state.bytes = state.bytes.saturating_sub(job.bytes);
                    idle.notify_all();
                }
                mlx::clear_streams();
            })
            .context("spawn DFlash2 prefix store writer")?;
        Ok(Self {
            store,
            sender: Some(sender),
            pending,
            gate,
            max_jobs,
            max_bytes,
            worker: Some(worker),
        })
    }

    /// Queue a save. `false` when the queue is full: the save is dropped and
    /// counted, never waited for.
    pub(crate) fn submit(&self, save: DFlash2DiskSave) -> bool {
        let bytes = save.payload_bytes();
        let accepted = {
            let mut state = lock(&self.pending.0);
            let fits = state.jobs < self.max_jobs
                && (state.jobs == 0 || state.bytes.saturating_add(bytes) <= self.max_bytes);
            if fits {
                state.jobs += 1;
                state.bytes += bytes;
            }
            fits
        };
        let sent = accepted
            && self
                .sender
                .as_ref()
                .is_some_and(|sender| sender.send(Job { save, bytes }).is_ok());
        if accepted && !sent {
            let mut state = lock(&self.pending.0);
            state.jobs = state.jobs.saturating_sub(1);
            state.bytes = state.bytes.saturating_sub(bytes);
        }
        if !sent {
            self.store().note_dropped();
        }
        sent
    }

    /// The store's index. Hold it briefly: the writer needs it to finish.
    pub(crate) fn store(&self) -> MutexGuard<'_, DFlash2DiskStore> {
        lock(&self.store)
    }

    /// The store, for a reader that outlives this writer's owner (health).
    pub(crate) fn shared_store(&self) -> Arc<Mutex<DFlash2DiskStore>> {
        Arc::clone(&self.store)
    }

    /// Wait until every queued save is written.
    #[cfg(test)]
    pub(crate) fn flush(&self) {
        let (state, idle) = &*self.pending;
        let mut state = lock(state);
        while state.jobs > 0 {
            state = idle.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Keep the worker from starting its next save while the guard lives.
    #[cfg(test)]
    pub(crate) fn hold(&self) -> MutexGuard<'_, ()> {
        lock(&self.gate)
    }
}

fn write_job(store: &Mutex<DFlash2DiskStore>, save: &DFlash2DiskSave) -> Result<()> {
    let Some(plan) = lock(store).plan_save(save)? else {
        return Ok(());
    };
    let written = write_save(&plan, save)?;
    lock(store).commit_save(written)?;
    compact(store, plan.key())
}

/// Rewrite a long chain of range files as one, when memory allows: the
/// rewrite holds the artifact once as read and once as written.
fn compact(store: &Mutex<DFlash2DiskStore>, key: &str) -> Result<()> {
    let Some(plan) = lock(store).begin_compaction(key) else {
        return Ok(());
    };
    let needed = usize::try_from(plan.payload_bytes().saturating_mul(2)).unwrap_or(usize::MAX);
    let snapshot = crate::core::process_memory::global_process_memory_governor().sample_process();
    let fits = snapshot.effective_ceiling_bytes == 0
        || snapshot.current_usage_bytes.saturating_add(needed) <= snapshot.soft_watermark_bytes;
    let written = if fits {
        write_compaction(&plan)
    } else {
        // The next save of this conversation tries again.
        Err(anyhow::anyhow!("no memory headroom"))
    };
    if let Err(error) = &written {
        tracing::info!(key, error = %error, "DFlash2 prefix artifact not compacted on disk");
    }
    lock(store).finish_compaction(key, written.ok())
}

impl Drop for DFlash2PrefixWriter {
    fn drop(&mut self) {
        // Closing the channel lets the worker finish what is queued.
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::core::dflash2_prefix_store::tests::{temp_root, test_identity, test_save};

    fn tokens(first: u32) -> Vec<u32> {
        (first..first + 40).collect()
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_writer_drops_a_save_over_the_queue_limit() {
        let dir = temp_root("dflash2-writer-limit");
        let store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("store");
        let writer = DFlash2PrefixWriter::spawn_with_limits(store, 8, 1000).expect("writer");
        let first = test_save(&writer.store(), &tokens(0), true);
        let second = test_save(&writer.store(), &tokens(100), true);
        assert!(first.payload_bytes() > 1000);
        {
            let _held = writer.hold();
            // Larger than the byte limit, but nothing else is waiting.
            assert!(writer.submit(first));
            assert!(!writer.submit(second));
        }
        writer.flush();
        let snapshot = writer.store().snapshot();
        assert_eq!(
            (snapshot.saves, snapshot.dropped, snapshot.entries),
            (1, 1, 1)
        );
        drop(writer);
        std::fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_writer_counts_a_failed_write_and_keeps_going() {
        let dir = temp_root("dflash2-writer-unwritable");
        let store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("store");
        let writer = DFlash2PrefixWriter::spawn(store).expect("writer");
        let segments = writer.store().root().join("segments");
        let set_mode = |mode: u32| {
            std::fs::set_permissions(&segments, std::fs::Permissions::from_mode(mode))
                .expect("permissions");
        };
        set_mode(0o555);
        let save = test_save(&writer.store(), &tokens(0), true);
        assert!(writer.submit(save));
        writer.flush();
        let snapshot = writer.store().snapshot();
        assert_eq!(
            (snapshot.saves, snapshot.failed, snapshot.entries),
            (0, 1, 0)
        );

        set_mode(0o755);
        let save = test_save(&writer.store(), &tokens(0), true);
        assert!(writer.submit(save));
        writer.flush();
        let snapshot = writer.store().snapshot();
        assert_eq!(
            (snapshot.saves, snapshot.failed, snapshot.entries),
            (1, 1, 1)
        );
        // A failed write leaves nothing behind.
        for sub in ["segments", "context", "artifacts"] {
            let dir = writer.store().root().join(sub);
            assert_eq!(std::fs::read_dir(dir).expect("dir").count(), 1, "{sub}");
        }
        drop(writer);
        std::fs::remove_dir_all(dir).expect("cleanup");
    }
}
