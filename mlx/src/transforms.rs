//! Computation graph transforms (sync + async evaluation).
//!
//! For lazy `Array::eval()` see `mlx/src/array.rs`. This module adds:
//!
//! - [`synchronize`] — block on the current thread's default stream
//! - [`synchronize_stream`] — block on a specific stream
//! - [`async_eval`] — submit + return a runtime-agnostic Future

use crate::{Array, Error, Result, Stream};

/// Block the current thread until all queued work on the current thread's
/// **default stream** completes. To synchronize on a specific stream
/// (regardless of which stream is currently the default), use
/// [`synchronize_stream`].
pub fn synchronize() -> Result<()> {
    mlx_sys::stream::ffi::synchronize().map_err(Error::from)
}

/// Block the current thread until all queued work on the **given stream**
/// completes, regardless of which thread queued the work or which stream
/// is currently the default.
pub fn synchronize_stream(s: Stream) -> Result<()> {
    mlx_sys::stream::ffi::synchronize_stream(s.into()).map_err(Error::from)
}

/// Asynchronously evaluate one or more arrays.
///
/// Submits the computation graph to MLX's stream worker on the **caller's
/// thread's default stream** (non-blocking, < 1µs), then returns a
/// `Future<Output = Result<()>>` that resolves when the work completes.
///
/// The future is **runtime-agnostic** — `.await` it under tokio,
/// async-std, smol, `futures_lite::future::block_on`, or any executor.
///
/// # Cancellation
///
/// Dropping the returned future without awaiting does **not** cancel the
/// submitted MLX work — MLX has no cancellation primitive. The work runs
/// to completion in the background, consuming GPU time and memory. Any
/// subsequent operation on the same arrays will implicitly synchronize.
///
/// # Implementation note
///
/// Submission runs synchronously on the caller's thread (so the work lands
/// on the caller's default stream, < 1µs). The returned future then waits
/// on each array's MLX [`Event`] inside [`blocking::unblock`]. Events are
/// MTLSharedEvent-backed and waitable from any thread, so the future polls
/// correctly under multi-threaded executors.
///
/// Stream-level synchronization is **not** used here because MLX's
/// per-stream `CommandEncoder` lookup is thread-local — calling
/// `synchronize_stream(s)` on a thread that did not register `s` throws.
/// Per-array events have no such constraint.
///
/// Scheduling overhead is ~5µs per call from the `blocking` global thread
/// pool, negligible vs typical MLX kernel times (µs–ms).
///
/// [`Event`]: https://github.com/ml-explore/mlx/blob/main/mlx/event.h
pub fn async_eval(arrays: &[&Array]) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
    // Clone each &Array into an owned Array (cheap: shared refcount on
    // mlx::core::array::array_desc_). Owned values move into the future's
    // closure so the wait calls have valid array references regardless of
    // when the caller drops the originals.
    let owned: Vec<Array> = arrays.iter().map(|a| (*a).clone()).collect();

    // Build raw pointer slice + submit (sync, fast). Pointers are valid for
    // the duration of this function (owned outlives them via stack).
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        owned.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: each pointer references an owned Array kept alive on this
    // stack frame; MLX async_eval copies the array refs internally so the
    // pointers need not outlive this call.
    let submit_result = unsafe { mlx_sys::stream::ffi::async_eval_many(&raw) };

    async move {
        submit_result.map_err(Error::from)?;
        blocking::unblock(move || -> Result<()> {
            // Wait on each submitted array's event. Events fire as the
            // underlying stream worker completes their kernels; total wall
            // time is bounded by the slowest array, not the sum.
            for a in &owned {
                mlx_sys::transforms::ffi::array_wait(a.as_inner()).map_err(Error::from)?;
            }
            Ok(())
        })
        .await
    }
}
