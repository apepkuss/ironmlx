use std::sync::Mutex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// ── Global error capture ──────────────────────────────────────────────────────
//
// mlx-c surfaces errors through a user-installed callback.  We install one
// once (via `init`) that stores the last message in a thread-local so that
// callers can retrieve it after a non-zero return code.

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Take the last MLX error message recorded on this thread, if any.
pub(crate) fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|cell| cell.borrow_mut().take())
}

/// Convert a non-zero C return code into `Err(Error::Mlx(...))`.
#[inline]
pub(crate) fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        let msg = take_last_error().unwrap_or_else(|| format!("unknown MLX error (code {})", code));
        Err(Error::Mlx(msg))
    }
}

/// Install the MLX error handler.  Called once at crate initialisation.
pub(crate) fn install_error_handler() {
    unsafe extern "C" fn handler(msg: *const std::os::raw::c_char, _data: *mut std::os::raw::c_void) {
        if msg.is_null() {
            return;
        }
        let s = unsafe { std::ffi::CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned();
        LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(s));
    }

    unsafe {
        mlx_sys::mlx_set_error_handler(Some(handler), std::ptr::null_mut(), None);
    }
}
