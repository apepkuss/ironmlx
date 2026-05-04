//! Computation graph transforms (sync + async evaluation).
//!
//! For lazy `Array::eval()` see `mlx/src/array.rs`. This module adds:
//!
//! - [`synchronize`] — block on the current thread's default stream
//! - [`synchronize_stream`] — block on a specific stream
//! - [`async_eval`] — submit + return a runtime-agnostic Future (Task 5)

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
/// The future captures the submission stream at construction time and
/// synchronizes on it explicitly via [`blocking::unblock`]. This works
/// correctly even when the future is polled on a different thread than
/// the submitter (MLX's bare `synchronize()` is thread-local; we use
/// `synchronize_stream(captured)` instead). Scheduling overhead is
/// ~5µs per call from the `blocking` global thread pool, negligible vs
/// typical MLX kernel times (µs–ms).
pub fn async_eval(arrays: &[&Array]) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
    // Capture the submission stream NOW (on the caller's thread, before
    // submission). MLX's async_eval queues work on the caller-thread's
    // default stream; we must wait on that exact stream regardless of
    // which thread polls the returned future.
    let device = mlx_sys::stream::ffi::default_device();
    let captured_stream = mlx_sys::stream::ffi::default_stream(device);

    // Build raw pointer slice + submit (sync, fast).
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: pointers valid for this fn (we hold &Array refs); MLX
    // async_eval copies arrays internally (refcount-share), so pointers
    // need not outlive THIS function — only the submission.
    let submit_result = unsafe { mlx_sys::stream::ffi::async_eval_many(&raw) };

    // Returned future: synchronize on the captured stream via blocking.
    // Stream is Copy (POD), moves into the closure with no lifetime issues.
    async move {
        submit_result.map_err(Error::from)?;
        blocking::unblock(move || {
            mlx_sys::stream::ffi::synchronize_stream(captured_stream).map_err(Error::from)
        })
        .await
    }
}
