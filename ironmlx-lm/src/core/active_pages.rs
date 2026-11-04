//! Private, disposable active-state pages. Immutable page leases survive forks
//! and rollback; only unreferenced slots may be reused. This is not a recovery
//! journal. Model/prefix files are never opened for writing.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, OpenOptionsExt, PermissionsExt},
    },
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const OWNER: &[u8] = b"ironmlx-flash-active-v1";
const HEADER: usize = 40;
const ENTRY_CHARGE: usize = 128;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ActivePagesConfig {
    pub root: PathBuf,
    pub cache_bytes: usize,
    pub pack_bytes: usize,
    pub disk_bytes: u64,
    pub min_free_bytes: u64,
}
impl ActivePagesConfig {
    pub fn for_memory(root: PathBuf, available: u64) -> Self {
        let (cache, pack) = if available <= 24 << 30 {
            (1, 32)
        } else if available <= 56 << 30 {
            (4, 64)
        } else {
            (16, 128)
        };
        Self {
            root,
            cache_bytes: cache << 20,
            pack_bytes: pack << 20,
            disk_bytes: 64 << 30,
            min_free_bytes: 1 << 30,
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (8192..=16 << 20).contains(&self.cache_bytes)
                && (4 << 20..=256 << 20).contains(&self.pack_bytes),
            "invalid Flash active-state RAM budget"
        );
        ensure!(
            self.disk_bytes > 0,
            "Flash active-state disk quota must be positive"
        );
        Ok(())
    }
}

#[derive(Default)]
pub struct ActivePagesStats {
    live_pages: AtomicU64,
    write_pages: AtomicU64,
    read_pages: AtomicU64,
    hot_window_tokens: AtomicU64,
    directory: std::sync::OnceLock<PathBuf>,
    live_files: AtomicU64,
    disk_bytes: AtomicU64,
    peak_disk_bytes: AtomicU64,
    cache_bytes: AtomicU64,
    read_bytes: AtomicU64,
    written_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    read_us: AtomicU64,
    write_us: AtomicU64,
    cache_per_layer_bytes: AtomicU64,
    gather_budget_bytes: AtomicU64,
    disk_quota_bytes: AtomicU64,
    failures: AtomicU64,
    pub prefix_export_skips: AtomicU64,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ActivePagesSnapshot {
    pub storage_dir: PathBuf,
    pub live_pages: u64,
    pub write_pages: u64,
    pub read_pages: u64,
    pub hot_window_tokens: Option<u64>,
    pub enabled: bool,
    pub live_files: u64,
    pub disk_bytes: u64,
    pub peak_disk_bytes: u64,
    pub cache_bytes: u64,
    pub read_bytes: u64,
    pub written_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: Option<f64>,
    pub read_us: u64,
    pub write_us: u64,
    pub cache_per_layer_bytes: u64,
    pub gather_budget_bytes: u64,
    pub disk_quota_bytes: u64,
    pub failures: u64,
    pub prefix_export_skips: u64,
}
impl ActivePagesStats {
    pub fn set_hot_window_tokens(&self, tokens: u64) {
        self.hot_window_tokens.store(tokens, Relaxed);
    }
    pub fn snapshot(&self) -> ActivePagesSnapshot {
        let hits = self.hits.load(Relaxed);
        let misses = self.misses.load(Relaxed);
        ActivePagesSnapshot {
            storage_dir: self.directory.get().cloned().unwrap_or_default(),
            live_pages: self.live_pages.load(Relaxed),
            write_pages: self.write_pages.load(Relaxed),
            read_pages: self.read_pages.load(Relaxed),
            hot_window_tokens: match self.hot_window_tokens.load(Relaxed) {
                0 => None,
                n => Some(n),
            },
            enabled: true,
            live_files: self.live_files.load(Relaxed),
            disk_bytes: self.disk_bytes.load(Relaxed),
            peak_disk_bytes: self.peak_disk_bytes.load(Relaxed),
            cache_bytes: self.cache_bytes.load(Relaxed),
            read_bytes: self.read_bytes.load(Relaxed),
            written_bytes: self.written_bytes.load(Relaxed),
            hits,
            misses,
            hit_rate: (hits + misses > 0).then(|| hits as f64 / (hits + misses) as f64),
            read_us: self.read_us.load(Relaxed),
            write_us: self.write_us.load(Relaxed),
            cache_per_layer_bytes: self.cache_per_layer_bytes.load(Relaxed),
            gather_budget_bytes: self.gather_budget_bytes.load(Relaxed),
            disk_quota_bytes: self.disk_quota_bytes.load(Relaxed),
            failures: self.failures.load(Relaxed),
            prefix_export_skips: self.prefix_export_skips.load(Relaxed),
        }
    }
}

pub struct ActivePagesPool {
    pub config: ActivePagesConfig,
    pub stats: Arc<ActivePagesStats>,
    path: PathBuf,
    _owner: File,
}
fn unique() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT.fetch_add(1, Relaxed)
    )
}
impl ActivePagesPool {
    pub fn new(config: ActivePagesConfig) -> Result<Arc<Self>> {
        config.validate()?;
        std::fs::create_dir_all(&config.root)?;
        ensure!(
            !std::fs::symlink_metadata(&config.root)?
                .file_type()
                .is_symlink(),
            "active-state root must not be a symlink"
        );
        // Reclaim only our recognizable, unlocked session directories. Never
        // infer process death from a PID or remove another live engine's files.
        for entry in std::fs::read_dir(&config.root)?.take(1024) {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with("session-")
                || !entry.file_type()?.is_dir()
            {
                continue;
            }
            let Ok(mut owner) = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(entry.path().join("owner"))
            else {
                continue;
            };
            if unsafe { libc::flock(owner.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                continue;
            }
            let mut magic = Vec::new();
            if (&mut owner)
                .take(OWNER.len() as u64 + 1)
                .read_to_end(&mut magic)
                .is_ok()
                && magic == OWNER
            {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        let path = config.root.join(format!("session-{}", unique()));
        std::fs::create_dir(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
        let result = (|| {
            let mut owner = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .mode(0o600)
                .open(path.join("owner"))?;
            ensure!(
                unsafe { libc::flock(owner.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
                "cannot lock active-state session"
            );
            owner.write_all(OWNER)?;
            let stats = Arc::new(ActivePagesStats {
                cache_per_layer_bytes: AtomicU64::new(config.cache_bytes as u64),
                gather_budget_bytes: AtomicU64::new(config.pack_bytes as u64),
                disk_quota_bytes: AtomicU64::new(config.disk_bytes),
                ..Default::default()
            });
            let _ = stats.directory.set(config.root.clone());
            let pool = Self {
                config,
                stats,
                path: path.clone(),
                _owner: owner,
            };
            pool.reserve(0)?;
            Ok(pool)
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&path);
        }
        result.map(Arc::new)
    }
    fn reserve(&self, bytes: u64) -> Result<()> {
        if self.config.min_free_bytes > 0 {
            let path = std::ffi::CString::new(self.path.as_os_str().as_encoded_bytes())?;
            let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
            ensure!(
                unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } == 0,
                "cannot sample active-state disk space"
            );
            let stat = unsafe { stat.assume_init() };
            ensure!(
                (stat.f_bavail as u64).saturating_mul(stat.f_frsize)
                    >= self.config.min_free_bytes.saturating_add(bytes),
                "insufficient free SSD space for active state"
            );
        }
        self.stats
            .disk_bytes
            .fetch_update(Relaxed, Relaxed, |n| {
                n.checked_add(bytes)
                    .filter(|n| *n <= self.config.disk_bytes)
            })
            .map_err(|_| anyhow::anyhow!("Flash active-state disk quota exhausted"))?;
        self.stats
            .peak_disk_bytes
            .fetch_max(self.stats.disk_bytes.load(Relaxed), Relaxed);
        Ok(())
    }
    pub fn file(self: &Arc<Self>, page_bytes: usize) -> Result<Arc<ActivePageFile>> {
        ensure!(
            page_bytes > 0 && page_bytes <= self.config.pack_bytes,
            "invalid active page size"
        );
        let path = self.path.join(unique());
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&path)?;
        #[cfg(target_os = "macos")]
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            let error = std::io::Error::last_os_error();
            let _ = std::fs::remove_file(&path);
            return Err(error.into());
        }
        self.stats.live_files.fetch_add(1, Relaxed);
        Ok(Arc::new(ActivePageFile {
            pool: Arc::clone(self),
            file,
            path,
            page_bytes,
            allocation: Mutex::default(),
            cache: Mutex::default(),
            next_id: AtomicU64::new(1),
        }))
    }
}
impl Drop for ActivePagesPool {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(%error, "active-state session cleanup failed");
        }
    }
}

#[derive(Default)]
struct Allocation {
    slots: u64,
    free: Vec<u64>,
}
#[derive(Default)]
struct ReadCache {
    clock: u64,
    entries: HashMap<u64, (Arc<Vec<u8>>, u64)>,
    charged: usize,
}
pub struct ActivePageFile {
    pool: Arc<ActivePagesPool>,
    file: File,
    path: PathBuf,
    page_bytes: usize,
    allocation: Mutex<Allocation>,
    cache: Mutex<ReadCache>,
    next_id: AtomicU64,
}
pub struct ActivePage {
    file: Arc<ActivePageFile>,
    slot: u64,
    id: u64,
    digest: [u8; 32],
}
impl ActivePageFile {
    fn cache(&self, id: u64, data: Arc<Vec<u8>>) {
        let charge = data.len() + ENTRY_CHARGE;
        if charge > self.pool.config.cache_bytes {
            return;
        }
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.entries.contains_key(&id) {
            return;
        }
        while cache.charged + charge > self.pool.config.cache_bytes {
            let oldest = cache
                .entries
                .iter()
                .min_by_key(|(_, (_, age))| *age)
                .map(|(&id, _)| id);
            let Some(id) = oldest else {
                break;
            };
            let (old, _) = cache.entries.remove(&id).unwrap();
            let bytes = old.len() + ENTRY_CHARGE;
            cache.charged -= bytes;
            self.pool.stats.cache_bytes.fetch_sub(bytes as u64, Relaxed);
        }
        cache.clock += 1;
        let clock = cache.clock;
        cache.entries.insert(id, (data, clock));
        cache.charged += charge;
        self.pool
            .stats
            .cache_bytes
            .fetch_add(charge as u64, Relaxed);
    }
    pub fn write(self: &Arc<Self>, payload: &[u8]) -> Result<Arc<ActivePage>> {
        Ok(self.write_batch(&[payload])?.remove(0))
    }
    pub fn write_batch(self: &Arc<Self>, payloads: &[&[u8]]) -> Result<Vec<Arc<ActivePage>>> {
        ensure!(
            !payloads.is_empty() && payloads.iter().all(|p| p.len() == self.page_bytes),
            "active-state page length mismatch"
        );
        ensure!(
            payloads
                .len()
                .checked_mul(self.page_bytes)
                .is_some_and(|n| n <= self.pool.config.pack_bytes),
            "active-state write batch exceeds staging budget"
        );
        let start = Instant::now();
        let result = (|| {
            let slots: Vec<u64> = {
                let mut a = self
                    .allocation
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let new = payloads.len().saturating_sub(a.free.len());
                self.pool
                    .reserve((new * (self.page_bytes + HEADER)) as u64)?;
                (0..payloads.len())
                    .map(|_| {
                        if let Some(slot) = a.free.pop() {
                            slot
                        } else {
                            let slot = a.slots;
                            a.slots += 1;
                            slot
                        }
                    })
                    .collect()
            };
            let mut records = Vec::with_capacity(payloads.len());
            let mut pages = Vec::with_capacity(payloads.len());
            for (&slot, payload) in slots.iter().zip(payloads) {
                let id = self.next_id.fetch_add(1, Relaxed);
                let mut hash = Sha256::new();
                hash.update(id.to_le_bytes());
                hash.update(payload);
                let digest: [u8; 32] = hash.finalize().into();
                self.pool.stats.live_pages.fetch_add(1, Relaxed);
                let page = Arc::new(ActivePage {
                    file: Arc::clone(self),
                    slot,
                    id,
                    digest,
                });
                let mut record = Vec::with_capacity(HEADER + payload.len());
                record.extend_from_slice(&id.to_le_bytes());
                record.extend_from_slice(&digest);
                record.extend_from_slice(payload);
                records.push((slot, record));
                pages.push(page);
            }
            records.sort_unstable_by_key(|(slot, _)| *slot);
            let stride = (self.page_bytes + HEADER) as u64;
            let mut index = 0;
            while index < records.len() {
                let first = index;
                let mut buffer = Vec::new();
                loop {
                    buffer.extend_from_slice(&records[index].1);
                    index += 1;
                    if index == records.len() || records[index].0 != records[index - 1].0 + 1 {
                        break;
                    }
                }
                self.file.write_all_at(&buffer, records[first].0 * stride)?;
                self.pool
                    .stats
                    .written_bytes
                    .fetch_add(buffer.len() as u64, Relaxed);
            }
            for (page, payload) in pages.iter().zip(payloads) {
                self.cache(page.id, Arc::new(payload.to_vec()));
            }
            self.pool
                .stats
                .write_pages
                .fetch_add(pages.len() as u64, Relaxed);
            Ok(pages)
        })();
        self.pool
            .stats
            .write_us
            .fetch_add(start.elapsed().as_micros() as u64, Relaxed);
        if result.is_err() {
            self.pool.stats.failures.fetch_add(1, Relaxed);
        }
        result
    }
}
impl ActivePage {
    fn cached(&self) -> Option<Arc<Vec<u8>>> {
        {
            let mut cache = self
                .file
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.clock += 1;
            let clock = cache.clock;
            if let Some((bytes, age)) = cache.entries.get_mut(&self.id) {
                *age = clock;
                self.file.pool.stats.hits.fetch_add(1, Relaxed);
                return Some(Arc::clone(bytes));
            }
        }
        None
    }
    pub fn read(&self) -> Result<Arc<Vec<u8>>> {
        if let Some(bytes) = self.cached() {
            return Ok(bytes);
        }
        self.file.pool.stats.misses.fetch_add(1, Relaxed);
        let start = Instant::now();
        let result = (|| {
            let mut record = vec![0; HEADER + self.file.page_bytes];
            let offset = self.slot * record.len() as u64;
            self.file.file.read_exact_at(&mut record, offset)?;
            ensure!(
                record[..8] == self.id.to_le_bytes() && record[8..HEADER] == self.digest,
                "active-state page identity mismatch"
            );
            let mut hash = Sha256::new();
            hash.update(self.id.to_le_bytes());
            hash.update(&record[HEADER..]);
            ensure!(
                <[u8; 32]>::from(hash.finalize()) == self.digest,
                "active-state page checksum mismatch"
            );
            self.file
                .pool
                .stats
                .read_bytes
                .fetch_add(record.len() as u64, Relaxed);
            self.file.pool.stats.read_pages.fetch_add(1, Relaxed);
            let data = Arc::new(record.split_off(HEADER));
            self.file.cache(self.id, Arc::clone(&data));
            Ok(data)
        })();
        self.file
            .pool
            .stats
            .read_us
            .fetch_add(start.elapsed().as_micros() as u64, Relaxed);
        if result.is_err() {
            self.file.pool.stats.failures.fetch_add(1, Relaxed);
        }
        result
    }
}
impl Drop for ActivePage {
    fn drop(&mut self) {
        self.file.pool.stats.live_pages.fetch_sub(1, Relaxed);
        self.file
            .allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .free
            .push(self.slot);
    }
}
impl Drop for ActivePageFile {
    fn drop(&mut self) {
        let bytes = self
            .allocation
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .slots
            * (self.page_bytes + HEADER) as u64;
        let cached = self
            .cache
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .charged;
        self.pool.stats.disk_bytes.fetch_sub(bytes, Relaxed);
        self.pool
            .stats
            .cache_bytes
            .fetch_sub(cached as u64, Relaxed);
        self.pool.stats.live_files.fetch_sub(1, Relaxed);
        if let Err(error) = std::fs::remove_file(&self.path) {
            tracing::warn!(%error, "active-state file cleanup failed");
        }
    }
}

/// Schedule only the pages selected by the next attention tile. The existing
/// bounded reader pool supplies look-ahead and parallel I/O without MLX arrays
/// crossing threads or keeping a store lock held while waiting for disk.
pub fn prefetch(pages: &[Arc<ActivePage>]) -> Result<Vec<Arc<Vec<u8>>>> {
    // Pin the entire batch's existing hits before misses start populating the
    // LRU. A gather larger than cache capacity must not evict all of its own
    // upcoming hits and degenerate into a zero-hit sequential scan.
    let mut result: Vec<_> = pages.iter().map(|p| p.cached()).collect();
    let misses: Vec<_> = pages
        .iter()
        .enumerate()
        .filter(|(i, _)| result[*i].is_none())
        .map(|(i, p)| (i, Arc::clone(p)))
        .collect();
    let jobs: Vec<_> = misses
        .chunks(32)
        .map(|pages| {
            let pages = pages.to_vec();
            super::tensor_file::io_task(move || {
                pages
                    .iter()
                    .map(|(i, page)| Ok((*i, page.read()?)))
                    .collect::<Result<Vec<_>>>()
            })
        })
        .collect();
    for job in jobs {
        for (index, data) in job.recv().context("active-state reader stopped")?? {
            result[index] = Some(data);
        }
    }
    result
        .into_iter()
        .map(|data| data.context("active-state prefetch returned incomplete batch"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ActivePagesConfig {
        ActivePagesConfig {
            root: std::env::temp_dir().join(unique()),
            cache_bytes: 8192,
            pack_bytes: 4 << 20,
            disk_bytes: 2 * (8192 + HEADER) as u64,
            min_free_bytes: 0,
        }
    }
    #[test]
    fn memory_tiers_keep_independent_payload_and_read_cache_budgets() {
        for (available, cache, pack) in [(24u64, 1usize, 32usize), (56, 4, 64), (120, 16, 128)] {
            let cfg = ActivePagesConfig::for_memory(PathBuf::from("unused"), available << 30);
            assert_eq!(cfg.cache_bytes, cache << 20);
            assert_eq!(cfg.pack_bytes, pack << 20);
            assert_eq!(cfg.disk_bytes, 64 << 30);
            assert_eq!(cfg.min_free_bytes, 1 << 30);
            cfg.validate().unwrap();
        }
    }
    #[test]
    fn leases_protect_forks_and_reused_slots_never_hit_old_cache_data() {
        let mut cfg = config();
        cfg.cache_bytes = 32768;
        let root = cfg.root.clone();
        let pool = ActivePagesPool::new(cfg).unwrap();
        let file = pool.file(8192).unwrap();
        let a = file.write(&vec![1; 8192]).unwrap();
        let fork = Arc::clone(&a);
        let b = file.write(&vec![2; 8192]).unwrap();
        drop(a);
        assert!(file.write(&vec![3; 8192]).is_err());
        assert_eq!(fork.read().unwrap().as_slice(), vec![1; 8192]);
        drop(fork);
        let c = file.write(&vec![3; 8192]).unwrap();
        assert_eq!(c.read().unwrap().as_slice(), vec![3; 8192]);
        assert_eq!(b.read().unwrap().as_slice(), vec![2; 8192]);
        assert_eq!(pool.stats.snapshot().disk_bytes, 2 * (8192 + HEADER) as u64);
        drop(file);
        assert_eq!(pool.stats.snapshot().live_files, 1);
        drop(b);
        drop(c);
        assert_eq!(pool.stats.snapshot().live_files, 0);
        assert_eq!(pool.stats.snapshot().disk_bytes, 0);
        assert_eq!(pool.stats.snapshot().cache_bytes, 0);
        drop(pool);
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn corruption_and_truncation_fail_closed() {
        let cfg = config();
        let root = cfg.root.clone();
        let pool = ActivePagesPool::new(cfg).unwrap();
        let file = pool.file(8192).unwrap();
        let a = file.write(&vec![7; 8192]).unwrap();
        file.file.write_all_at(&[8], HEADER as u64).unwrap();
        assert!(a.read().unwrap_err().to_string().contains("checksum"));
        file.file.set_len(4).unwrap();
        assert!(a.read().is_err());
        assert_eq!(pool.stats.snapshot().failures, 2);
        drop(a);
        drop(file);
        drop(pool);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn bounded_prefetch_preserves_order_and_cleanup() {
        let mut cfg = config();
        cfg.disk_bytes = 2 << 20;
        let root = cfg.root.clone();
        let pool = ActivePagesPool::new(cfg).unwrap();
        let file = pool.file(8192).unwrap();
        let pages: Vec<_> = (0..65)
            .map(|n| file.write(&vec![n; 8192]).unwrap())
            .collect();
        let result = prefetch(&pages).unwrap();
        for (n, bytes) in result.iter().enumerate() {
            assert!(bytes.iter().all(|&v| v == n as u8));
        }
        assert_eq!(pool.stats.snapshot().misses, 65);
        drop(pages);
        drop(file);
        drop(pool);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn gather_pins_hits_before_scan_misses_evict_them() {
        let mut cfg = config();
        cfg.cache_bytes = 32768;
        cfg.disk_bytes = 2 << 20;
        let root = cfg.root.clone();
        let pool = ActivePagesPool::new(cfg).unwrap();
        let file = pool.file(8192).unwrap();
        let pages: Vec<_> = (0..20)
            .map(|n| file.write(&vec![n; 8192]).unwrap())
            .collect();
        let result = prefetch(&pages).unwrap();
        assert_eq!(pool.stats.snapshot().hits, 3);
        for (n, data) in result.iter().enumerate() {
            assert!(data.iter().all(|&b| b == n as u8));
        }
        drop(pages);
        drop(file);
        drop(pool);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn session_collection_skips_live_and_unrecognized_directories() {
        let cfg = config();
        let root = cfg.root.clone();
        let first = ActivePagesPool::new(cfg.clone()).unwrap();
        let unknown = root.join("session-unknown");
        std::fs::create_dir(&unknown).unwrap();
        let stale = root.join("session-stale");
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("owner"), OWNER).unwrap();
        let second = ActivePagesPool::new(cfg).unwrap();
        assert!(first.path.is_dir());
        assert!(unknown.is_dir());
        assert!(!stale.exists());
        drop(first);
        assert!(second.path.is_dir());
        drop(second);
        std::fs::remove_dir(unknown).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn disk_headroom_failure_does_not_allocate_a_slot() {
        let cfg = config();
        let root = cfg.root.clone();
        let mut pool = ActivePagesPool::new(cfg).unwrap();
        Arc::get_mut(&mut pool).unwrap().config.min_free_bytes = u64::MAX;
        let file = pool.file(8192).unwrap();
        assert!(file.write(&vec![1; 8192]).is_err());
        assert_eq!(pool.stats.snapshot().disk_bytes, 0);
        drop(file);
        drop(pool);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn startup_headroom_failure_cleans_the_private_session() {
        let mut cfg = config();
        cfg.min_free_bytes = u64::MAX;
        let root = cfg.root.clone();
        assert!(ActivePagesPool::new(cfg).is_err());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        std::fs::remove_dir(root).unwrap();
    }
}
