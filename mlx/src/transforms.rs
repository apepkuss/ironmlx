//! Computation graph transforms (sync + async evaluation).
//!
//! For lazy `Array::eval()` see `mlx/src/array.rs`. This module adds:
//!
//! - [`synchronize`] — block on the current thread's default stream
//! - [`synchronize_stream`] — block on a specific stream
//! - [`synchronize_thread_local_stream`] — block on a thread-local stream token
//! - [`eval`] — synchronously evaluate a batch of arrays (block until done)
//! - [`async_eval`] — submit a batch of arrays for async evaluation (returns
//!   immediately; subsequent reads block until materialized)
//! - [`async_eval_fut`] — submit + return a runtime-agnostic Future that
//!   resolves when the batch completes

use crate::{Array, Error, Result, Stream, ThreadLocalStream};

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

/// Block until all queued work on the current thread's concrete stream for a
/// thread-local stream token completes.
pub fn synchronize_thread_local_stream(s: ThreadLocalStream) -> Result<()> {
    mlx_sys::stream::ffi::synchronize_thread_local_stream(s.into()).map_err(Error::from)
}

/// Clear MLX's memory cache.
///
/// This releases cached allocator buffers; it does not clear compiled function
/// graphs. Use sparingly in long-running services where request-local temporary
/// buffers can otherwise accumulate and affect later requests.
pub fn clear_cache() {
    mlx_sys::stream::ffi::clear_cache();
}

/// Evaluate multiple arrays in one call. Blocks the current thread until
/// every array has been computed. More efficient than calling
/// [`Array::eval`](crate::Array::eval) per array because MLX can fuse and
/// schedule the combined graph in one pass.
///
/// `eval(&[])` is a no-op and returns `Ok(())`.
pub fn eval(arrays: &[&Array]) -> Result<()> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: each pointer borrows a live `&Array` kept alive by the caller
    // for the duration of this call; MLX `eval` copies the array refs
    // internally so the pointers need not outlive this call.
    unsafe { mlx_sys::stream::ffi::eval_many(&raw) }.map_err(Error::from)
}

/// Asynchronously evaluate multiple arrays. Submits the computation graph
/// to MLX's stream worker and returns immediately — does **not** block on
/// completion.
///
/// The arrays become observable as their computation finishes. Any later
/// operation that reads materialized data ([`Array::to_vec`],
/// [`Array::item`], [`Array::eval`], indexing, …) will implicitly wait for
/// the queued async work.
///
/// `async_eval(&[])` is a no-op and returns `Ok(())`.
///
/// To wait on the submitted batch via a runtime-agnostic Future, use
/// [`async_eval_fut`] instead.
pub fn async_eval(arrays: &[&Array]) -> Result<()> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: each pointer borrows a live `&Array` kept alive by the caller
    // for the duration of this call; MLX `async_eval` copies the array refs
    // internally so the pointers need not outlive this call.
    unsafe { mlx_sys::stream::ffi::async_eval_many(&raw) }.map_err(Error::from)
}

/// Asynchronously evaluate one or more arrays and return a Future that
/// resolves when the work completes.
///
/// Submits the computation graph to MLX's stream worker on the **caller's
/// thread's default stream** (non-blocking, < 1µs), then returns a
/// `Future<Output = Result<()>>` that resolves when the work completes.
///
/// The output is `()` — MLX evaluates arrays in-place via its refcount
/// model, so the caller's original `&Array` references read the
/// materialized data after `.await` (call `.to_vec()`, index, etc.).
///
/// The future is **runtime-agnostic** — `.await` it under tokio,
/// async-std, smol, `futures_lite::future::block_on`, or any executor.
///
/// If you don't need a Future (just want fire-and-forget submission and
/// rely on implicit waits when reading the data later), use the cheaper
/// [`async_eval`] variant which returns `Result<()>` directly.
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
pub fn async_eval_fut(
    arrays: &[&Array],
) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
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
