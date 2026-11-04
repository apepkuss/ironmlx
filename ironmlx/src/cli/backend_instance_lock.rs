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
    owner_pid: libc::pid_t,
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

        // flock references can survive fork until a child closes/execs. The
        // owning process explicitly unlocks on Drop; on crash the kernel
        // releases the lock when the final inherited reference closes.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            return Ok(Self {
                _file: file,
                owner_pid: unsafe { libc::getpid() },
            });
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

impl Drop for BackendInstanceLock {
    fn drop(&mut self) {
        // An inherited guard is not the owner. Unlocking it in the child would
        // also unlock the live parent's shared open file description.
        if unsafe { libc::getpid() } == self.owner_pid {
            while unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) } == -1
                && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
            {
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendInstanceLock, INSTANCE_ALREADY_RUNNING_ERROR};
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::path::PathBuf;

    struct InheritingChild {
        pid: libc::pid_t,
        // EOF normally releases the child if this process dies; a child-side
        // deadline also bounds its lifetime if another fork inherited the pipe.
        _keepalive: OwnedFd,
    }
    impl Drop for InheritingChild {
        fn drop(&mut self) {
            unsafe {
                libc::kill(self.pid, libc::SIGKILL);
                while libc::waitpid(self.pid, std::ptr::null_mut(), 0) == -1
                    && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                {
                }
            }
        }
    }

    fn pipe() -> (OwnedFd, OwnedFd) {
        let mut descriptors = [-1; 2];
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        let pair = unsafe {
            (
                OwnedFd::from_raw_fd(descriptors[0]),
                OwnedFd::from_raw_fd(descriptors[1]),
            )
        };
        for fd in [&pair.0, &pair.1] {
            assert_eq!(
                unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) },
                0
            );
        }
        pair
    }

    fn fork_with_lock(
        lock: BackendInstanceLock,
        drop_in_child: bool,
    ) -> (BackendInstanceLock, InheritingChild) {
        let (ready_read, ready_write) = pipe();
        let (release_read, release_write) = pipe();
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed: {}", std::io::Error::last_os_error());
        if pid == 0 {
            // A multithreaded test runner may have forked us. Only close,
            // getpid/close (inherited guard drop), poll/write and _exit are used here;
            // never enter MLX, allocate, format, log or unwind in this child.
            unsafe {
                libc::close(ready_read.as_raw_fd());
                libc::close(release_write.as_raw_fd());
                if drop_in_child {
                    drop(lock);
                }
                let byte = 1u8;
                if libc::write(ready_write.as_raw_fd(), (&byte as *const u8).cast(), 1) != 1 {
                    libc::_exit(2);
                }
                let mut release = libc::pollfd {
                    fd: release_read.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                libc::poll(&mut release, 1, 30_000);
                libc::_exit(0);
            }
        }
        drop(ready_write);
        drop(release_read);
        let child = InheritingChild {
            pid,
            _keepalive: release_write,
        };
        let mut ready = libc::pollfd {
            fd: ready_read.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            let timeout = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(!timeout.is_zero(), "child ready timeout");
            let result = unsafe { libc::poll(&mut ready, 1, timeout.as_millis().max(1) as i32) };
            if result == -1
                && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
            {
                continue;
            }
            assert_eq!(result, 1, "child ready timeout");
            break;
        }
        std::fs::File::from(ready_read)
            .read_exact(&mut [0u8])
            .unwrap();
        (lock, child)
    }

    fn unique_lock_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("ironmlx-instance-lock-{}", uuid::Uuid::new_v4()))
            .join("backend.lock")
    }

    #[test]
    fn rejects_a_second_backend_and_releases_on_drop() {
        let path = unique_lock_path();
        let first = BackendInstanceLock::acquire_at(&path).expect("first lock");
        assert_ne!(
            unsafe { libc::fcntl(first._file.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );

        let error = BackendInstanceLock::acquire_at(&path).expect_err("second lock must fail");
        assert!(error.to_string().contains(INSTANCE_ALREADY_RUNNING_ERROR));

        drop(first);
        BackendInstanceLock::acquire_at(&path).expect("lock after release");
        std::fs::remove_dir_all(path.parent().expect("lock parent")).expect("remove test lock");
    }

    #[test]
    fn owner_drop_releases_even_while_forked_child_holds_the_descriptor() {
        let path = unique_lock_path();
        let first = BackendInstanceLock::acquire_at(&path).unwrap();
        let (first, child) = fork_with_lock(first, false);
        drop(first);
        let reacquired = BackendInstanceLock::acquire_at(&path);
        let error = reacquired.as_ref().err().map(ToString::to_string);
        drop(reacquired);
        drop(child);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
        assert!(
            error.is_none(),
            "owner dropped but inherited descriptor retained lock: {error:?}"
        );
    }

    #[test]
    fn dropping_an_inherited_guard_in_child_does_not_unlock_the_parent() {
        let path = unique_lock_path();
        let first = BackendInstanceLock::acquire_at(&path).unwrap();
        let (first, child) = fork_with_lock(first, true);
        let blocked = BackendInstanceLock::acquire_at(&path);
        let rejected = blocked
            .as_ref()
            .is_err_and(|error| error.to_string().contains(INSTANCE_ALREADY_RUNNING_ERROR));
        drop(blocked);
        drop(first);
        let after = BackendInstanceLock::acquire_at(&path);
        let released = after.is_ok();
        drop(after);
        drop(child);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
        assert!(rejected, "child drop unlocked the live parent's guard");
        assert!(released, "parent drop did not release its lock");
    }
}
