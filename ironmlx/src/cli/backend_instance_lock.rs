use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub(crate) const INSTANCE_ALREADY_RUNNING_ERROR: &str = "ironmlx_instance_already_running";
const LOCK_DIRECTORY: &str = "run";
const LOCK_FILE: &str = "backend.lock";

#[derive(Debug)]
pub(crate) struct BackendInstanceLock {
    _file: File,
}

impl BackendInstanceLock {
    pub(crate) fn acquire() -> Result<Self> {
        let home = dirs::home_dir().context("locating home directory for IronMLX backend lock")?;
        Self::acquire_at(&home.join(".ironmlx").join(LOCK_DIRECTORY).join(LOCK_FILE))
    }

    fn acquire_at(path: &Path) -> Result<Self> {
        let parent = path
            .parent()
            .context("resolving IronMLX backend lock directory")?;
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating IronMLX backend lock directory {}",
                parent.display()
            )
        })?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .with_context(|| format!("opening IronMLX backend lock {}", path.display()))?;

        // flock is tied to the open file description. Holding `file` in this guard
        // releases the lock automatically on normal exit, crash, or SIGKILL.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            return Ok(Self { _file: file });
        }

        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            anyhow::bail!(
                "{INSTANCE_ALREADY_RUNNING_ERROR}: another IronMLX backend is already running for this macOS user"
            );
        }
        Err(error).with_context(|| format!("locking IronMLX backend lock {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendInstanceLock, INSTANCE_ALREADY_RUNNING_ERROR};
    use std::path::PathBuf;

    fn unique_lock_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("ironmlx-instance-lock-{}", uuid::Uuid::new_v4()))
            .join("backend.lock")
    }

    #[test]
    fn rejects_a_second_backend_and_releases_on_drop() {
        let path = unique_lock_path();
        let first = BackendInstanceLock::acquire_at(&path).expect("first lock");

        let error = BackendInstanceLock::acquire_at(&path).expect_err("second lock must fail");
        assert!(error.to_string().contains(INSTANCE_ALREADY_RUNNING_ERROR));

        drop(first);
        BackendInstanceLock::acquire_at(&path).expect("lock after release");
        std::fs::remove_dir_all(path.parent().expect("lock parent")).expect("remove test lock");
    }
}
