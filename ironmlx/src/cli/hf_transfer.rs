use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use clap::Args;
use hf_hub::{Repo, RepoType};
use reqwest::header::{
    HeaderMap, HeaderName, CONTENT_LENGTH, CONTENT_RANGE, ETAG, LOCATION, RANGE,
};
use reqwest::redirect::Policy;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;

const TRANSFER_IDENTITY_VERSION: u32 = 1;
const HF_PARTIAL_EXTENSION: &str = "sync.part";
const HASH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const PROGRESS_MIN_BYTES: u64 = 4 * 1024 * 1024;
const PROGRESS_MAX_INTERVAL: Duration = Duration::from_millis(200);
/// hf-mirror.com -> huggingface.co -> CDN is two hops; leave headroom for
/// one more proxy without following loops.
const MAX_REDIRECTS: usize = 5;
const CHUNK_ATTEMPTS: u32 = 4;
const CHUNK_RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Args, Clone, Debug)]
pub struct HfTransferArgs {
    /// Hugging Face model repository, in organization/model form.
    #[arg(long)]
    repo_id: String,
    /// Immutable Hugging Face commit SHA.
    #[arg(long)]
    revision: String,
    /// Repository-relative file path.
    #[arg(long)]
    filename: String,
    /// Final file location inside the unpublished IronMLX staging snapshot.
    #[arg(long)]
    destination: PathBuf,
    /// Private cache directory used for this file's resumable transfer.
    #[arg(long)]
    cache_dir: PathBuf,
    /// Expected file size from the immutable repository manifest.
    #[arg(long)]
    expected_size: u64,
    /// Expected SHA-256 from the immutable repository manifest.
    #[arg(long)]
    expected_sha256: String,
    /// Hugging Face endpoint.
    #[arg(long, default_value = "https://huggingface.co")]
    endpoint: String,
    /// Maximum number of concurrent Range chunks.
    #[arg(long, default_value_t = 4)]
    parallelism: usize,
    /// Size of each Range chunk.
    #[arg(long, default_value_t = 10_000_000)]
    chunk_size: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TransferIdentity {
    version: u32,
    provider: String,
    repo_id: String,
    commit_sha: String,
    path: String,
    expected_size: u64,
    expected_sha256: String,
    etag: String,
}

#[derive(Debug)]
struct RemoteFileIdentity {
    commit_sha: String,
    size: u64,
    etag: String,
    /// Final location of the redirect chain; Range requests go here first.
    location: Url,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TransferEvent<'a> {
    Progress {
        bytes: u64,
        total: u64,
    },
    Complete {
        path: &'a Path,
        size: u64,
        sha256: &'a str,
        etag: &'a str,
    },
}

#[derive(Clone, Default)]
struct JsonProgress {
    state: Arc<Mutex<JsonProgressState>>,
}

struct JsonProgressState {
    bytes: u64,
    total: u64,
    last_emitted_bytes: u64,
    last_emitted_at: Instant,
    awaiting_baseline: bool,
    #[cfg(test)]
    first_emitted_bytes: Option<u64>,
}

impl Default for JsonProgressState {
    fn default() -> Self {
        Self {
            bytes: 0,
            total: 0,
            last_emitted_bytes: 0,
            last_emitted_at: Instant::now(),
            awaiting_baseline: false,
            #[cfg(test)]
            first_emitted_bytes: None,
        }
    }
}

impl JsonProgress {
    fn init(&self, size: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes = 0;
            state.total = size;
            state.last_emitted_bytes = 0;
            state.last_emitted_at = Instant::now();
            state.awaiting_baseline = true;
        }
    }

    fn update(&self, size: u64) {
        if let Ok(mut state) = self.state.lock() {
            // The first update after `init` is the committed starting offset,
            // before newly transferred bytes are reported.
            let baseline = std::mem::take(&mut state.awaiting_baseline);
            state.bytes = if baseline {
                size.min(state.total)
            } else {
                state.bytes.saturating_add(size).min(state.total)
            };
            let now = Instant::now();
            if baseline
                || state.bytes == state.total
                || state.bytes.saturating_sub(state.last_emitted_bytes) >= PROGRESS_MIN_BYTES
                || now.duration_since(state.last_emitted_at) >= PROGRESS_MAX_INTERVAL
            {
                #[cfg(test)]
                if state.first_emitted_bytes.is_none() {
                    state.first_emitted_bytes = Some(state.bytes);
                }
                emit_event(&TransferEvent::Progress {
                    bytes: state.bytes,
                    total: state.total,
                });
                state.last_emitted_bytes = state.bytes;
                state.last_emitted_at = now;
            }
        }
    }

    /// Withdraws bytes of a failed chunk attempt; the retry reports them again.
    fn rollback(&self, size: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes = state.bytes.saturating_sub(size);
        }
    }

    fn finish(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes = state.total;
            if state.last_emitted_bytes != state.total {
                #[cfg(test)]
                if state.first_emitted_bytes.is_none() {
                    state.first_emitted_bytes = Some(state.bytes);
                }
                emit_event(&TransferEvent::Progress {
                    bytes: state.bytes,
                    total: state.total,
                });
                state.last_emitted_bytes = state.total;
                state.last_emitted_at = Instant::now();
            }
        }
    }
}

struct TransferLock {
    file: File,
}

impl TransferLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating transfer lock directory {}", parent.display())
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening transfer lock {}", path.display()))?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            bail!(
                "another process is already downloading this Hugging Face file: {}",
                path.display()
            );
        }
        Ok(Self { file })
    }
}

impl Drop for TransferLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

struct HashCursor {
    digest: Sha256,
    bytes_hashed: u64,
}

pub fn run(args: HfTransferArgs) -> Result<()> {
    validate_args(&args)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("creating Hugging Face transfer runtime")?;
    runtime.block_on(run_async(args))
}

async fn run_async(args: HfTransferArgs) -> Result<()> {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    run_async_with_progress(args, token, JsonProgress::default()).await
}

async fn run_async_with_progress(
    args: HfTransferArgs,
    token: Option<String>,
    progress: JsonProgress,
) -> Result<()> {
    let expected_sha256 = args.expected_sha256.to_ascii_lowercase();
    if args.destination.exists() {
        let actual_size = args.destination.metadata()?.len();
        let actual_sha256 = sha256_file(&args.destination)?;
        if actual_size == args.expected_size && actual_sha256 == expected_sha256 {
            emit_event(&TransferEvent::Complete {
                path: &args.destination,
                size: actual_size,
                sha256: &actual_sha256,
                etag: "",
            });
            return Ok(());
        }
        bail!(
            "staging destination already exists with unexpected identity: {}",
            args.destination.display()
        );
    }

    let lock_path = transfer_lock_path(&args.cache_dir)?;
    let _lock = TransferLock::acquire(&lock_path)?;

    let repo = Repo::with_revision(args.repo_id.clone(), RepoType::Model, args.revision.clone());
    let url = format!(
        "{}/{}/resolve/{}/{}",
        args.endpoint.trim_end_matches('/'),
        repo.url(),
        repo.url_revision(),
        args.filename
    );
    let http = Arc::new(RedirectingClient::new(&url, token)?);
    let remote = resolve_remote_identity(&http).await?;
    if !remote.commit_sha.eq_ignore_ascii_case(&args.revision) {
        bail!(
            "Hugging Face returned commit {} for immutable revision {}",
            remote.commit_sha,
            args.revision
        );
    }
    if remote.size != args.expected_size {
        bail!(
            "Hugging Face returned size {} for {}, expected {}",
            remote.size,
            args.filename,
            args.expected_size
        );
    }

    let identity = TransferIdentity {
        version: TRANSFER_IDENTITY_VERSION,
        provider: "huggingface".to_string(),
        repo_id: args.repo_id.clone(),
        commit_sha: args.revision.to_ascii_lowercase(),
        path: args.filename.clone(),
        expected_size: args.expected_size,
        expected_sha256: expected_sha256.clone(),
        etag: remote.etag.clone(),
    };
    prepare_transfer_cache(&args.cache_dir, &identity)?;

    // Keep hf-hub's cache layout: the App reads this partial's committed marker
    // to show and resume progress across restarts.
    let partial_path = args
        .cache_dir
        .join(repo.folder_name())
        .join("blobs")
        .join(&remote.etag)
        .with_extension(HF_PARTIAL_EXTENSION);
    repair_invalid_partial(&partial_path, args.expected_size)?;

    let stop_hashing = Arc::new(AtomicBool::new(false));
    let hash_stop = Arc::clone(&stop_hashing);
    let hash_partial = partial_path.clone();
    let expected_size = args.expected_size;
    let hash_task = tokio::task::spawn_blocking(move || {
        hash_committed_prefix(&hash_partial, expected_size, &hash_stop)
    });

    let download_result = download_partial(
        http,
        remote.location.clone(),
        &partial_path,
        RangePlan {
            length: args.expected_size,
            chunk_size: args.chunk_size as u64,
            parallelism: args.parallelism,
        },
        progress,
    )
    .await;
    stop_hashing.store(true, Ordering::Release);
    let cursor = hash_task
        .await
        .context("joining incremental SHA-256 task")??;
    download_result.context("Hugging Face Range download failed")?;
    let blob = partial_path;
    let stored_size = blob.metadata()?.len();
    let marker_end = args.expected_size + std::mem::size_of::<u64>() as u64;
    if stored_size != marker_end {
        bail!(
            "partial for {} holds {} bytes, expected {}",
            args.filename,
            stored_size,
            marker_end
        );
    }
    let actual_sha256 = finish_sha256(&blob, cursor, args.expected_size)?;
    if actual_sha256 != expected_sha256 {
        let _ = fs::remove_dir_all(&args.cache_dir);
        bail!(
            "downloaded file SHA-256 {} does not match expected {}",
            actual_sha256,
            expected_sha256
        );
    }
    // Drop the committed marker only once the content is verified.
    let verified = OpenOptions::new().write(true).open(&blob)?;
    verified.set_len(args.expected_size)?;
    verified
        .sync_all()
        .with_context(|| format!("syncing verified partial {}", blob.display()))?;
    drop(verified);

    if let Some(parent) = args.destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating staging directory {}", parent.display()))?;
    }
    fs::rename(&blob, &args.destination).with_context(|| {
        format!(
            "atomically moving verified blob {} to {}",
            blob.display(),
            args.destination.display()
        )
    })?;
    sync_file(&args.destination)?;
    if let Some(parent) = args.destination.parent() {
        sync_directory(parent)?;
    }
    let _ = fs::remove_dir_all(&args.cache_dir);
    let _ = fs::remove_file(&lock_path);
    emit_event(&TransferEvent::Complete {
        path: &args.destination,
        size: args.expected_size,
        sha256: &actual_sha256,
        etag: &remote.etag,
    });
    Ok(())
}

fn validate_args(args: &HfTransferArgs) -> Result<()> {
    let repo_parts = args.repo_id.split('/').collect::<Vec<_>>();
    if repo_parts.len() != 2 || repo_parts.iter().any(|part| part.is_empty()) {
        bail!("repo-id must use organization/model form");
    }
    if args.revision.len() != 40 || !args.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("revision must be a full 40-character commit SHA");
    }
    if args.expected_sha256.len() != 64
        || !args
            .expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("expected-sha256 must contain 64 hexadecimal characters");
    }
    if args.parallelism == 0 {
        bail!("parallelism must be greater than zero");
    }
    if args.chunk_size == 0 {
        bail!("chunk-size must be greater than zero");
    }
    validate_repository_path(&args.filename)?;
    let cache_name = args
        .cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !cache_name.ends_with(".hf-transfer") {
        bail!("cache-dir must end with .hf-transfer");
    }
    Ok(())
}

fn validate_repository_path(path: &str) -> Result<()> {
    let candidate = Path::new(path);
    if candidate.is_absolute() || path.is_empty() {
        bail!("filename must be a non-empty repository-relative path");
    }
    if candidate
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("filename contains an unsafe path component");
    }
    Ok(())
}

/// HTTP client that follows redirects itself: every chain is bounded, Range
/// headers reach the final location, and the token is sent only to the origin
/// of the configured endpoint (never to a CDN or another host).
struct RedirectingClient {
    client: reqwest::Client,
    origin: Url,
    token: Option<String>,
}

struct RedirectedResponse {
    response: reqwest::Response,
    /// URL that produced `response`.
    url: Url,
    /// Headers of every redirect answer, in order.
    redirects: Vec<HeaderMap>,
}

impl RedirectingClient {
    fn new(url: &str, token: Option<String>) -> Result<Self> {
        let origin = Url::parse(url).context("parsing Hugging Face file URL")?;
        let client = reqwest::Client::builder()
            .user_agent(concat!("ironmlx/", env!("CARGO_PKG_VERSION")))
            .redirect(Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()
            .context("creating Hugging Face transfer client")?;
        Ok(Self {
            client,
            origin,
            token,
        })
    }

    async fn get_range(&self, start: &Url, range: &str) -> Result<RedirectedResponse> {
        let mut url = start.clone();
        let mut redirects = Vec::new();
        loop {
            let mut request = self.client.get(url.clone()).header(RANGE, range);
            if let Some(token) = self.token.as_deref() {
                if same_origin(&url, &self.origin) {
                    request = request.bearer_auth(token);
                }
            }
            let response = request
                .send()
                .await
                // Signed CDN URLs must not reach logs.
                .map_err(reqwest::Error::without_url)
                .with_context(|| format!("requesting {}", display_url(&url)))?;
            if !is_redirect(response.status()) {
                return Ok(RedirectedResponse {
                    response,
                    url,
                    redirects,
                });
            }
            if redirects.len() >= MAX_REDIRECTS {
                bail!(
                    "too many redirects: more than {MAX_REDIRECTS} following {}",
                    display_url(start)
                );
            }
            let location = required_header(response.headers(), &LOCATION)?;
            let next = url
                .join(&location)
                .with_context(|| format!("invalid redirect location from {}", display_url(&url)))?;
            check_redirect_target(&self.origin, &next)?;
            redirects.push(response.headers().clone());
            url = next;
        }
    }
}

/// An https endpoint never downgrades to http; plain http is followed only
/// when the configured endpoint is itself http (a local test server).
fn check_redirect_target(origin: &Url, next: &Url) -> Result<()> {
    match next.scheme() {
        "https" => Ok(()),
        "http" if origin.scheme() == "http" => Ok(()),
        "http" => bail!("refusing redirect from https to http {}", display_url(next)),
        other => bail!("refusing redirect to a {other} URL"),
    }
}

fn is_redirect(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

fn same_origin(lhs: &Url, rhs: &Url) -> bool {
    lhs.scheme() == rhs.scheme()
        && lhs.host_str() == rhs.host_str()
        && lhs.port_or_known_default() == rhs.port_or_known_default()
}

fn display_url(url: &Url) -> String {
    format!("{}{}", url.origin().ascii_serialization(), url.path())
}

async fn resolve_remote_identity(http: &RedirectingClient) -> Result<RemoteFileIdentity> {
    let reply = http
        .get_range(&http.origin, "bytes=0-0")
        .await
        .context("requesting immutable Hugging Face file identity")?;
    let status = reply.response.status();
    if !status.is_success() {
        bail!("Hugging Face identity request failed with HTTP {status}");
    }
    // A mirror may answer with a bare redirect to huggingface.co; the first
    // answer that names the commit carries the file identity.
    let final_headers = reply.response.headers();
    let commit_header = HeaderName::from_static("x-repo-commit");
    let linked_etag_header = HeaderName::from_static("x-linked-etag");
    let linked_size_header = HeaderName::from_static("x-linked-size");
    let headers = reply
        .redirects
        .iter()
        .chain(std::iter::once(final_headers))
        .find(|headers| headers.contains_key(&commit_header))
        .context("Hugging Face response is missing x-repo-commit")?;
    let commit_sha = required_header(headers, &commit_header)?;
    let etag = clean_etag(
        headers
            .get(&linked_etag_header)
            .or_else(|| headers.get(ETAG))
            .context("Hugging Face response is missing ETag")?
            .to_str()
            .context("Hugging Face returned a non-text ETag")?,
    )?;
    let size = if let Some(value) = headers.get(&linked_size_header) {
        value
            .to_str()
            .context("Hugging Face returned a non-text linked size")?
            .parse()
            .context("parsing Hugging Face linked size")?
    } else if let Some(value) = final_headers.get(CONTENT_RANGE) {
        parse_content_range_size(value.to_str()?)?
    } else {
        required_header(final_headers, &CONTENT_LENGTH)?
            .parse()
            .context("parsing Hugging Face content length")?
    };
    Ok(RemoteFileIdentity {
        commit_sha,
        size,
        etag,
        location: reply.url,
    })
}

struct RangePlan {
    length: u64,
    chunk_size: u64,
    parallelism: usize,
}

/// Shared state of one partial-file transfer.
struct RangeTransfer {
    http: Arc<RedirectingClient>,
    location: Mutex<Url>,
    file: File,
    length: u64,
    chunk_size: u64,
    /// Start of the next chunk to fetch. Workers take chunks strictly in file
    /// order, so the committed prefix trails the bytes on disk by at most
    /// `parallelism` chunks and a restart loses little.
    next_start: AtomicU64,
    ledger: Mutex<CommitLedger>,
    progress: JsonProgress,
}

struct CommitLedger {
    committed: u64,
    finished: BinaryHeap<Reverse<(u64, u64)>>,
}

/// Downloads the missing tail of the partial file (payload followed by an
/// 8-byte little-endian committed-prefix marker, as written by hf-hub) and
/// truncates the marker once every byte is present.
async fn download_partial(
    http: Arc<RedirectingClient>,
    location: Url,
    partial: &Path,
    plan: RangePlan,
    progress: JsonProgress,
) -> Result<()> {
    let marker_len = std::mem::size_of::<u64>() as u64;
    let stored_len = plan
        .length
        .checked_add(marker_len)
        .context("partial size overflow")?;
    if let Some(parent) = partial.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating transfer directory {}", parent.display()))?;
    }
    let committed = match committed_prefix(partial, plan.length)? {
        Some(committed) => committed,
        None => {
            let file = File::create(partial)
                .with_context(|| format!("creating partial {}", partial.display()))?;
            file.set_len(stored_len)?;
            0
        }
    };
    progress.init(plan.length);
    progress.update(committed);

    if committed < plan.length {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial)
            .with_context(|| format!("opening partial {}", partial.display()))?;
        let transfer = Arc::new(RangeTransfer {
            http,
            location: Mutex::new(location),
            file,
            length: plan.length,
            chunk_size: plan.chunk_size,
            next_start: AtomicU64::new(committed),
            ledger: Mutex::new(CommitLedger {
                committed,
                finished: BinaryHeap::new(),
            }),
            progress: progress.clone(),
        });
        let mut workers = tokio::task::JoinSet::new();
        for _ in 0..plan.parallelism {
            workers.spawn(Arc::clone(&transfer).run_worker());
        }
        // The first failure drops the set, which aborts the other workers.
        while let Some(joined) = workers.join_next().await {
            joined.context("joining Range transfer worker")??;
        }
        let committed = transfer
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("partial commit ledger poisoned"))?
            .committed;
        if committed != plan.length {
            bail!(
                "Range transfer committed {committed} of {} bytes",
                plan.length
            );
        }
    }

    // The marker stays until the caller has verified the SHA-256, so a quit
    // before the verified file is in place still resumes as complete.
    progress.finish();
    Ok(())
}

impl RangeTransfer {
    async fn run_worker(self: Arc<Self>) -> Result<()> {
        loop {
            let start = self.next_start.fetch_add(self.chunk_size, Ordering::SeqCst);
            if start >= self.length {
                return Ok(());
            }
            let end = start.saturating_add(self.chunk_size).min(self.length) - 1;
            let data = self.fetch_with_retries(start, end).await?;
            let transfer = Arc::clone(&self);
            tokio::task::spawn_blocking(move || -> Result<()> {
                transfer.file.write_all_at(&data, start)?;
                transfer.commit(start, end)
            })
            .await
            .context("joining partial writer")??;
        }
    }

    async fn fetch_with_retries(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        let mut attempt = 0;
        loop {
            // A retry resolves the chain again: CDN locations are signed and expire.
            let from = if attempt == 0 {
                self.current_location()?
            } else {
                self.http.origin.clone()
            };
            let mut received = 0;
            match self.fetch(&from, start, end, &mut received).await {
                Ok((data, location)) => {
                    if let Ok(mut current) = self.location.lock() {
                        *current = location;
                    }
                    return Ok(data);
                }
                Err(error) => {
                    self.progress.rollback(received);
                    attempt += 1;
                    if attempt >= CHUNK_ATTEMPTS {
                        return Err(error).with_context(|| {
                            format!("downloading bytes {start}-{end} after {attempt} attempts")
                        });
                    }
                    tokio::time::sleep(CHUNK_RETRY_BASE_DELAY * 2_u32.pow(attempt - 1)).await;
                }
            }
        }
    }

    fn current_location(&self) -> Result<Url> {
        self.location
            .lock()
            .map(|location| location.clone())
            .map_err(|_| anyhow::anyhow!("transfer location lock poisoned"))
    }

    async fn fetch(
        &self,
        from: &Url,
        start: u64,
        end: u64,
        received: &mut u64,
    ) -> Result<(Vec<u8>, Url)> {
        let reply = self
            .http
            .get_range(from, &format!("bytes={start}-{end}"))
            .await?;
        let mut response = reply.response;
        if response.status() != StatusCode::PARTIAL_CONTENT {
            bail!(
                "Hugging Face Range request returned HTTP {}",
                response.status()
            );
        }
        let content_range = required_header(response.headers(), &CONTENT_RANGE)?;
        if parse_content_range(&content_range)? != (start, end, self.length) {
            bail!("Hugging Face returned Content-Range {content_range} for bytes {start}-{end}");
        }
        let size = end - start + 1;
        let mut data = Vec::with_capacity(usize::try_from(size)?);
        while let Some(piece) = response
            .chunk()
            .await
            .map_err(reqwest::Error::without_url)
            .context("reading Hugging Face Range body")?
        {
            if data.len() as u64 + piece.len() as u64 > size {
                bail!("Hugging Face sent more than the requested bytes {start}-{end}");
            }
            data.extend_from_slice(&piece);
            *received += piece.len() as u64;
            self.progress.update(piece.len() as u64);
        }
        if data.len() as u64 != size {
            bail!(
                "Hugging Face ended bytes {start}-{end} after {} bytes",
                data.len()
            );
        }
        Ok((data, reply.url))
    }

    /// Advances the on-disk committed marker over every contiguous finished chunk.
    fn commit(&self, start: u64, end: u64) -> Result<()> {
        let mut ledger = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("partial commit ledger poisoned"))?;
        ledger.finished.push(Reverse((start, end)));
        let before = ledger.committed;
        while let Some(Reverse((first, last))) = ledger.finished.peek().copied() {
            if first != ledger.committed {
                break;
            }
            ledger.finished.pop();
            ledger.committed = last + 1;
        }
        if ledger.committed != before {
            self.file
                .write_all_at(&ledger.committed.to_le_bytes(), self.length)
                .context("writing partial committed marker")?;
        }
        Ok(())
    }
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64)> {
    let parsed = value
        .strip_prefix("bytes ")
        .and_then(|rest| rest.split_once('/'))
        .and_then(|(span, total)| {
            let (start, end) = span.split_once('-')?;
            Some((start.parse().ok()?, end.parse().ok()?, total.parse().ok()?))
        });
    parsed.with_context(|| format!("invalid Hugging Face Content-Range {value}"))
}

fn required_header(headers: &HeaderMap, name: &HeaderName) -> Result<String> {
    headers
        .get(name)
        .with_context(|| format!("Hugging Face response is missing {name}"))?
        .to_str()
        .with_context(|| format!("Hugging Face returned a non-text {name}"))
        .map(str::to_string)
}

fn parse_content_range_size(value: &str) -> Result<u64> {
    value
        .rsplit_once('/')
        .context("invalid Hugging Face Content-Range")?
        .1
        .parse()
        .context("parsing Hugging Face Content-Range size")
}

fn clean_etag(value: &str) -> Result<String> {
    let etag = value.trim().trim_matches('"').to_string();
    if etag.is_empty() || etag == "." || etag == ".." || etag.contains('/') || etag.contains('\\') {
        bail!("Hugging Face returned an unsafe ETag");
    }
    Ok(etag)
}

fn prepare_transfer_cache(cache_dir: &Path, identity: &TransferIdentity) -> Result<()> {
    let identity_path = cache_dir.join("identity.json");
    let matches = fs::read(&identity_path)
        .ok()
        .and_then(|data| serde_json::from_slice::<TransferIdentity>(&data).ok())
        .is_some_and(|stored| stored == *identity);
    if cache_dir.exists() && !matches {
        fs::remove_dir_all(cache_dir).with_context(|| {
            format!(
                "discarding partial with mismatched file identity {}",
                cache_dir.display()
            )
        })?;
    }
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating transfer cache {}", cache_dir.display()))?;
    atomic_write_json(&identity_path, identity)
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("identity path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".identity.{}.tmp", uuid::Uuid::new_v4()));
    let data = serde_json::to_vec_pretty(value)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&data)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    sync_directory(parent)
}

fn repair_invalid_partial(path: &Path, expected_size: u64) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut file = File::open(path)?;
    let expected_storage_size = expected_size
        .checked_add(std::mem::size_of::<u64>() as u64)
        .context("partial size overflow")?;
    if file.metadata()?.len() == expected_size {
        // Every byte arrived and the marker was dropped, but the process ended
        // before the file was moved into place. Mark it complete again; the
        // final SHA-256 check decides whether it is kept.
        drop(file);
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(expected_storage_size)?;
        file.write_all_at(&expected_size.to_le_bytes(), expected_size)?;
        file.sync_all()?;
        return Ok(());
    }
    if file.metadata()?.len() != expected_storage_size {
        drop(file);
        fs::remove_file(path)?;
        return Ok(());
    }
    file.seek(SeekFrom::Start(expected_size))?;
    let mut marker = [0_u8; std::mem::size_of::<u64>()];
    file.read_exact(&mut marker)?;
    let committed = u64::from_le_bytes(marker);
    if committed == expected_size.saturating_add(1) {
        drop(file);
        let mut file = OpenOptions::new().write(true).open(path)?;
        file.seek(SeekFrom::Start(expected_size))?;
        file.write_all(&expected_size.to_le_bytes())?;
        file.sync_all()?;
    } else if committed > expected_size {
        drop(file);
        fs::remove_file(path)?;
    }
    Ok(())
}

fn hash_committed_prefix(
    partial: &Path,
    expected_size: u64,
    stop: &AtomicBool,
) -> Result<HashCursor> {
    let mut cursor = HashCursor {
        digest: Sha256::new(),
        bytes_hashed: 0,
    };
    loop {
        if let Some(committed) = committed_prefix(partial, expected_size)? {
            if committed > cursor.bytes_hashed {
                hash_file_range(partial, &mut cursor.digest, cursor.bytes_hashed, committed)?;
                cursor.bytes_hashed = committed;
            }
        }
        if stop.load(Ordering::Acquire) {
            return Ok(cursor);
        }
        std::thread::sleep(HASH_POLL_INTERVAL);
    }
}

fn committed_prefix(partial: &Path, expected_size: u64) -> Result<Option<u64>> {
    let Ok(mut file) = File::open(partial) else {
        return Ok(None);
    };
    let marker_end = expected_size
        .checked_add(std::mem::size_of::<u64>() as u64)
        .context("partial marker offset overflow")?;
    if file.metadata()?.len() != marker_end {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(expected_size))?;
    let mut marker = [0_u8; std::mem::size_of::<u64>()];
    file.read_exact(&mut marker)?;
    let committed = u64::from_le_bytes(marker);
    if committed == expected_size.saturating_add(1) {
        return Ok(Some(expected_size));
    }
    if committed > expected_size {
        bail!("partial contains an invalid committed byte marker");
    }
    Ok(Some(committed))
}

fn finish_sha256(path: &Path, mut cursor: HashCursor, expected_size: u64) -> Result<String> {
    if cursor.bytes_hashed < expected_size {
        hash_file_range(path, &mut cursor.digest, cursor.bytes_hashed, expected_size)?;
        cursor.bytes_hashed = expected_size;
    }
    if cursor.bytes_hashed != expected_size {
        bail!("incremental SHA-256 did not cover the complete file");
    }
    Ok(format!("{:x}", cursor.digest.finalize()))
}

fn hash_file_range(path: &Path, digest: &mut Sha256, start: u64, end: u64) -> Result<()> {
    if start > end {
        bail!("invalid SHA-256 file range");
    }
    let mut file = File::open(path)
        .with_context(|| format!("opening {} for incremental SHA-256", path.display()))?;
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = end - start;
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    while remaining > 0 {
        let requested = usize::try_from(remaining.min(buffer.len() as u64))?;
        let read = file.read(&mut buffer[..requested])?;
        if read == 0 {
            bail!(
                "unexpected EOF hashing {} at byte {}",
                path.display(),
                end - remaining
            );
        }
        digest.update(&buffer[..read]);
        remaining -= read as u64;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let size = path.metadata()?.len();
    finish_sha256(
        path,
        HashCursor {
            digest: Sha256::new(),
            bytes_hashed: 0,
        },
        size,
    )
}

fn transfer_lock_path(cache_dir: &Path) -> Result<PathBuf> {
    let name = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("cache-dir has no valid final component")?;
    Ok(cache_dir.with_file_name(format!("{name}.lock")))
}

fn sync_file(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .open(path)?
        .sync_all()
        .with_context(|| format!("syncing verified file {}", path.display()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?
        .sync_all()
        .with_context(|| format!("syncing directory {}", path.display()))
}

fn emit_event(event: &TransferEvent<'_>) {
    if let Ok(json) = serde_json::to_string(event) {
        println!("{json}");
        let _ = std::io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, Response, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use reqwest::header::AUTHORIZATION;

    #[derive(Clone)]
    struct TestHubState {
        payload: Arc<Vec<u8>>,
        commit: String,
        etag: String,
        ranges: Arc<Mutex<Vec<String>>>,
    }

    fn args(cache_dir: PathBuf) -> HfTransferArgs {
        HfTransferArgs {
            repo_id: "org/model".to_string(),
            revision: "a".repeat(40),
            filename: "weights/model.safetensors".to_string(),
            destination: cache_dir
                .parent()
                .expect("cache parent")
                .join("model.safetensors"),
            cache_dir,
            expected_size: 16,
            expected_sha256: "b".repeat(64),
            endpoint: "https://huggingface.co".to_string(),
            parallelism: 4,
            chunk_size: 10_000_000,
        }
    }

    #[test]
    fn validates_immutable_identity_and_scoped_cache() {
        let valid = args(PathBuf::from("/tmp/model.safetensors.hf-transfer"));
        validate_args(&valid).expect("valid arguments");

        let mut mutable = valid;
        mutable.revision = "main".to_string();
        assert!(validate_args(&mutable).is_err());
    }

    #[test]
    fn changed_identity_discards_only_transfer_cache() {
        let root =
            std::env::temp_dir().join(format!("ironmlx-hf-transfer-{}", uuid::Uuid::new_v4()));
        let cache = root.join("model.safetensors.hf-transfer");
        fs::create_dir_all(&cache).expect("create cache");
        fs::write(cache.join("stale.sync.part"), b"stale").expect("write stale partial");
        let first = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: "a".repeat(40),
            path: "model.safetensors".to_string(),
            expected_size: 5,
            expected_sha256: "b".repeat(64),
            etag: "old".to_string(),
        };
        atomic_write_json(&cache.join("identity.json"), &first).expect("write first identity");
        let second = TransferIdentity {
            etag: "new".to_string(),
            ..first
        };

        prepare_transfer_cache(&cache, &second).expect("prepare changed identity");

        assert!(!cache.join("stale.sync.part").exists());
        let stored: TransferIdentity =
            serde_json::from_slice(&fs::read(cache.join("identity.json")).expect("read identity"))
                .expect("decode identity");
        assert_eq!(stored, second);
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn hashes_only_committed_prefix_then_finishes_without_rehashing() {
        let root = std::env::temp_dir().join(format!("ironmlx-hf-hash-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let partial = root.join("blob.sync.part");
        let payload = b"abcdefghijklmnop";
        let mut file = File::create(&partial).expect("create partial");
        file.set_len((payload.len() + std::mem::size_of::<u64>()) as u64)
            .expect("preallocate partial");
        file.seek(SeekFrom::Start(0)).expect("seek payload");
        file.write_all(payload).expect("write payload");
        file.seek(SeekFrom::Start(payload.len() as u64))
            .expect("seek marker");
        file.write_all(&8_u64.to_le_bytes())
            .expect("write committed marker");
        file.sync_all().expect("sync partial");

        let stop = AtomicBool::new(true);
        let cursor =
            hash_committed_prefix(&partial, payload.len() as u64, &stop).expect("hash prefix");
        assert_eq!(cursor.bytes_hashed, 8);
        file.set_len(payload.len() as u64).expect("truncate marker");
        let actual = finish_sha256(&partial, cursor, payload.len() as u64).expect("finish SHA-256");
        let expected = format!("{:x}", Sha256::digest(payload));
        assert_eq!(actual, expected);
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn invalid_partial_marker_is_removed_before_hf_hub_resume() {
        let root = std::env::temp_dir().join(format!("ironmlx-hf-marker-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let partial = root.join("blob.sync.part");
        let mut file = File::create(&partial).expect("create partial");
        file.set_len(24).expect("preallocate partial");
        file.seek(SeekFrom::Start(16)).expect("seek marker");
        file.write_all(&18_u64.to_le_bytes())
            .expect("write invalid marker");
        file.sync_all().expect("sync partial");

        repair_invalid_partial(&partial, 16).expect("repair partial");

        assert!(!partial.exists());
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test]
    async fn json_progress_uses_first_update_as_absolute_resume_baseline() {
        let progress = JsonProgress::default();

        progress.init(100);
        {
            let state = progress.state.lock().expect("progress state");
            assert!(state.awaiting_baseline);
            assert_eq!(state.bytes, 0);
            assert_eq!(state.last_emitted_bytes, 0);
        }

        progress.update(20);
        {
            let state = progress.state.lock().expect("progress state");
            assert!(!state.awaiting_baseline);
            assert_eq!(state.bytes, 20);
            assert_eq!(state.last_emitted_bytes, 20);
            assert_eq!(state.first_emitted_bytes, Some(20));
        }

        progress.update(7);
        let state = progress.state.lock().expect("progress state");
        assert_eq!(state.bytes, 27);
        assert_eq!(state.last_emitted_bytes, 20);
    }

    #[tokio::test]
    async fn json_progress_reports_zero_when_hf_hub_discards_partial() {
        let progress = JsonProgress::default();

        progress.init(100);
        progress.update(0);

        let state = progress.state.lock().expect("progress state");
        assert!(!state.awaiting_baseline);
        assert_eq!(state.bytes, 0);
        assert_eq!(state.last_emitted_bytes, 0);
        assert_eq!(state.first_emitted_bytes, Some(0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hf_hub_resumes_committed_ranges_and_publishes_verified_destination() {
        let payload = Arc::new(
            (0..(2 * 1024 * 1024 + 137))
                .map(|index| (index % 251) as u8)
                .collect::<Vec<_>>(),
        );
        let expected_sha256 = format!("{:x}", Sha256::digest(payload.as_slice()));
        let commit = "c".repeat(40);
        let etag = expected_sha256.clone();
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let state = TestHubState {
            payload: Arc::clone(&payload),
            commit: commit.clone(),
            etag: etag.clone(),
            ranges: Arc::clone(&ranges),
        };
        let app = Router::new()
            .route("/*path", get(test_hub_file))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test hub");
        let address = listener.local_addr().expect("test hub address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test hub");
        });

        let root = std::env::temp_dir().join(format!("ironmlx-hf-e2e-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let cache_dir = root.join("model.safetensors.hf-transfer");
        let destination = root.join("model.safetensors");
        let identity = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: commit.clone(),
            path: "model.safetensors".to_string(),
            expected_size: payload.len() as u64,
            expected_sha256: expected_sha256.clone(),
            etag: etag.clone(),
        };
        prepare_transfer_cache(&cache_dir, &identity).expect("prepare transfer cache");
        let partial = cache_dir
            .join("models--org--model")
            .join("blobs")
            .join(&etag)
            .with_extension(HF_PARTIAL_EXTENSION);
        fs::create_dir_all(partial.parent().expect("partial parent")).expect("create blob dir");
        let resume_offset = 512 * 1024_u64;
        let mut file = File::create(&partial).expect("create partial");
        file.set_len(payload.len() as u64 + std::mem::size_of::<u64>() as u64)
            .expect("preallocate partial");
        file.seek(SeekFrom::Start(0)).expect("seek partial");
        file.write_all(&payload[..resume_offset as usize])
            .expect("write committed prefix");
        file.seek(SeekFrom::Start(payload.len() as u64))
            .expect("seek committed marker");
        file.write_all(&resume_offset.to_le_bytes())
            .expect("write committed marker");
        file.sync_all().expect("sync partial");

        let progress = JsonProgress::default();
        run_async_with_progress(
            HfTransferArgs {
                repo_id: "org/model".to_string(),
                revision: commit,
                filename: "model.safetensors".to_string(),
                destination: destination.clone(),
                cache_dir: cache_dir.clone(),
                expected_size: payload.len() as u64,
                expected_sha256: expected_sha256.clone(),
                endpoint: format!("http://{address}"),
                parallelism: 4,
                chunk_size: 256 * 1024,
            },
            None,
            progress.clone(),
        )
        .await
        .expect("run resumed transfer");

        assert_eq!(
            progress
                .state
                .lock()
                .expect("progress state")
                .first_emitted_bytes,
            Some(resume_offset)
        );

        assert_eq!(fs::read(&destination).expect("read destination"), *payload);
        assert_eq!(
            sha256_file(&destination).expect("hash destination"),
            expected_sha256
        );
        assert!(!cache_dir.exists());
        assert!(!transfer_lock_path(&cache_dir).expect("lock path").exists());
        let requested = ranges.lock().expect("ranges lock").clone();
        assert!(
            requested
                .iter()
                .filter(|range| range.as_str() != "bytes=0-0")
                .all(|range| range_start(range) >= resume_offset),
            "unexpected pre-resume Range request: {requested:?}"
        );

        server.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }

    async fn test_hub_file(
        State(state): State<TestHubState>,
        request: Request<Body>,
    ) -> Response<Body> {
        let range = request
            .headers()
            .get(RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("bytes=0-0")
            .to_string();
        state
            .ranges
            .lock()
            .expect("ranges lock")
            .push(range.clone());
        let start = range_start(&range) as usize;
        let requested_end = range
            .strip_prefix("bytes=")
            .and_then(|value| value.split_once('-'))
            .and_then(|(_, end)| end.parse::<usize>().ok())
            .unwrap_or(start);
        let end = requested_end.min(state.payload.len() - 1);
        let body = Body::from(state.payload[start..=end].to_vec());
        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header("x-repo-commit", &state.commit)
            .header("x-linked-etag", format!("\"{}\"", state.etag))
            .header(
                CONTENT_RANGE,
                format!("bytes {start}-{end}/{}", state.payload.len()),
            )
            .header(CONTENT_LENGTH, end - start + 1)
            .body(body)
            .expect("test response")
    }

    fn range_start(value: &str) -> u64 {
        value
            .strip_prefix("bytes=")
            .and_then(|value| value.split_once('-'))
            .and_then(|(start, _)| start.parse().ok())
            .expect("valid byte range")
    }

    const TEST_RESOLVE_PATH: &str =
        "/org/model/resolve/cccccccccccccccccccccccccccccccccccccccc/model.safetensors";

    #[derive(Clone)]
    enum TestHop {
        Redirect {
            status: StatusCode,
            location: String,
            identity: bool,
        },
        Serve {
            identity: bool,
        },
    }

    #[derive(Clone, Debug)]
    struct ChainRequest {
        server: &'static str,
        path: String,
        authorization: Option<String>,
        range: String,
    }

    #[derive(Clone)]
    struct ChainState {
        name: &'static str,
        hops: Arc<Mutex<std::collections::HashMap<String, TestHop>>>,
        payload: Arc<Vec<u8>>,
        commit: String,
        etag: String,
        log: Arc<Mutex<Vec<ChainRequest>>>,
        /// Serving fails with HTTP 500 for data ranges starting at or after this offset.
        fail_from: Arc<std::sync::atomic::AtomicU64>,
    }

    struct ChainServer {
        address: std::net::SocketAddr,
        hops: Arc<Mutex<std::collections::HashMap<String, TestHop>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl ChainServer {
        fn url(&self, path: &str) -> String {
            format!("http://{}{path}", self.address)
        }

        fn route(&self, path: &str, hop: TestHop) {
            self.hops
                .lock()
                .expect("hops lock")
                .insert(path.to_string(), hop);
        }
    }

    struct ChainFixture {
        payload: Arc<Vec<u8>>,
        commit: String,
        sha256: String,
        log: Arc<Mutex<Vec<ChainRequest>>>,
        fail_from: Arc<std::sync::atomic::AtomicU64>,
    }

    impl ChainFixture {
        fn new(len: usize) -> Self {
            let payload = Arc::new((0..len).map(|i| (i % 253) as u8).collect::<Vec<_>>());
            let sha256 = format!("{:x}", Sha256::digest(payload.as_slice()));
            Self {
                payload,
                commit: "c".repeat(40),
                sha256,
                log: Arc::new(Mutex::new(Vec::new())),
                fail_from: Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX)),
            }
        }

        async fn server(&self, name: &'static str) -> ChainServer {
            let hops = Arc::new(Mutex::new(std::collections::HashMap::new()));
            let state = ChainState {
                name,
                hops: Arc::clone(&hops),
                payload: Arc::clone(&self.payload),
                commit: self.commit.clone(),
                etag: self.sha256.clone(),
                log: Arc::clone(&self.log),
                fail_from: Arc::clone(&self.fail_from),
            };
            let app = Router::new().fallback(chain_handler).with_state(state);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind chain server");
            let address = listener.local_addr().expect("chain server address");
            let task = tokio::spawn(async move {
                axum::serve(listener, app)
                    .await
                    .expect("serve chain server");
            });
            ChainServer {
                address,
                hops,
                task,
            }
        }

        fn args(&self, root: &Path, endpoint: String) -> HfTransferArgs {
            HfTransferArgs {
                repo_id: "org/model".to_string(),
                revision: self.commit.clone(),
                filename: "model.safetensors".to_string(),
                destination: root.join("model.safetensors"),
                cache_dir: root.join("model.safetensors.hf-transfer"),
                expected_size: self.payload.len() as u64,
                expected_sha256: self.sha256.clone(),
                endpoint,
                parallelism: 4,
                chunk_size: 64 * 1024,
            }
        }

        fn requests(&self) -> Vec<ChainRequest> {
            self.log.lock().expect("log lock").clone()
        }
    }

    async fn chain_handler(
        State(state): State<ChainState>,
        request: Request<Body>,
    ) -> Response<Body> {
        let path = request.uri().path().to_string();
        let range = request
            .headers()
            .get(RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        state.log.lock().expect("log lock").push(ChainRequest {
            server: state.name,
            path: path.clone(),
            authorization: request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string),
            range: range.clone(),
        });
        let hop = state.hops.lock().expect("hops lock").get(&path).cloned();
        match hop {
            Some(TestHop::Redirect {
                status,
                location,
                identity,
            }) => {
                let mut response = Response::builder()
                    .status(status)
                    .header(LOCATION, location);
                if identity {
                    response = response
                        .header("x-repo-commit", &state.commit)
                        .header("x-linked-etag", format!("\"{}\"", state.etag))
                        .header("x-linked-size", state.payload.len().to_string());
                }
                response.body(Body::empty()).expect("redirect response")
            }
            Some(TestHop::Serve { identity }) => {
                let start = range_start(&range);
                if range != "bytes=0-0"
                    && start >= state.fail_from.load(std::sync::atomic::Ordering::SeqCst)
                {
                    return Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .expect("failure response");
                }
                let requested_end = range
                    .strip_prefix("bytes=")
                    .and_then(|value| value.split_once('-'))
                    .and_then(|(_, end)| end.parse::<usize>().ok())
                    .unwrap_or(start as usize);
                let start = start as usize;
                let end = requested_end.min(state.payload.len() - 1);
                let mut response = Response::builder();
                if identity {
                    response = response
                        .header("x-repo-commit", &state.commit)
                        .header("x-linked-etag", format!("\"{}\"", state.etag));
                }
                response
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(
                        CONTENT_RANGE,
                        format!("bytes {start}-{end}/{}", state.payload.len()),
                    )
                    .header(CONTENT_LENGTH, end - start + 1)
                    .body(Body::from(state.payload[start..=end].to_vec()))
                    .expect("range response")
            }
            None => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::empty())
                .expect("not found response"),
        }
    }

    fn test_root(label: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ironmlx-hf-{label}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create test root");
        root
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn follows_mirror_hub_and_cdn_redirect_chain() {
        let fixture = ChainFixture::new(300 * 1024 + 17);
        let mirror = fixture.server("mirror").await;
        let hub = fixture.server("hub").await;
        let cdn = fixture.server("cdn").await;
        // hf-mirror.com answers with a bare 308 to huggingface.co, which in
        // turn redirects (with the file identity headers) to a CDN.
        mirror.route(
            TEST_RESOLVE_PATH,
            TestHop::Redirect {
                status: StatusCode::PERMANENT_REDIRECT,
                location: hub.url(TEST_RESOLVE_PATH),
                identity: false,
            },
        );
        hub.route(
            TEST_RESOLVE_PATH,
            TestHop::Redirect {
                status: StatusCode::FOUND,
                location: cdn.url("/xet-bridge/blob?Expires=1"),
                identity: true,
            },
        );
        cdn.route("/xet-bridge/blob", TestHop::Serve { identity: false });
        let root = test_root("chain");
        let args = fixture.args(&root, format!("http://{}", mirror.address));
        let destination = args.destination.clone();

        run_async_with_progress(args, None, JsonProgress::default())
            .await
            .expect("transfer through a two-hop redirect chain");

        assert_eq!(
            fs::read(&destination).expect("read destination"),
            *fixture.payload
        );
        let data_ranges = fixture
            .requests()
            .into_iter()
            .filter(|request| request.range != "bytes=0-0")
            .collect::<Vec<_>>();
        assert!(!data_ranges.is_empty());
        assert!(
            data_ranges.iter().any(|request| request.server == "cdn"),
            "data ranges never reached the final location: {data_ranges:?}"
        );
        for server in [mirror, hub, cdn] {
            server.task.abort();
        }
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn redirect_chain_longer_than_limit_fails_clearly() {
        let fixture = ChainFixture::new(1024);
        let mirror = fixture.server("mirror").await;
        mirror.route(
            TEST_RESOLVE_PATH,
            TestHop::Redirect {
                status: StatusCode::TEMPORARY_REDIRECT,
                location: "/loop/1".to_string(),
                identity: false,
            },
        );
        for hop in 1..=20 {
            mirror.route(
                &format!("/loop/{hop}"),
                TestHop::Redirect {
                    status: StatusCode::TEMPORARY_REDIRECT,
                    location: format!("/loop/{}", hop + 1),
                    identity: false,
                },
            );
        }
        let root = test_root("loop");
        let args = fixture.args(&root, format!("http://{}", mirror.address));

        let error = run_async_with_progress(args, None, JsonProgress::default())
            .await
            .expect_err("an endless redirect chain must fail");

        let message = format!("{error:#}");
        assert!(
            message.contains("too many redirects"),
            "unexpected error: {message}"
        );
        assert_eq!(fixture.requests().len(), MAX_REDIRECTS + 1);
        mirror.task.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn authorization_is_kept_on_same_origin_hops_and_stripped_across_hosts() {
        let fixture = ChainFixture::new(200 * 1024 + 3);
        let hub = fixture.server("hub").await;
        let cdn = fixture.server("cdn").await;
        hub.route(
            TEST_RESOLVE_PATH,
            TestHop::Redirect {
                status: StatusCode::TEMPORARY_REDIRECT,
                location: "/api/resolve-cache/model.safetensors".to_string(),
                identity: false,
            },
        );
        hub.route(
            "/api/resolve-cache/model.safetensors",
            TestHop::Redirect {
                status: StatusCode::FOUND,
                location: cdn.url("/blob"),
                identity: true,
            },
        );
        cdn.route("/blob", TestHop::Serve { identity: false });
        let root = test_root("auth");
        let args = fixture.args(&root, format!("http://{}", hub.address));
        let destination = args.destination.clone();

        run_async_with_progress(args, Some("secret".to_string()), JsonProgress::default())
            .await
            .expect("transfer with a token");

        assert_eq!(
            fs::read(&destination).expect("read destination"),
            *fixture.payload
        );
        let requests = fixture.requests();
        let hub_requests = requests
            .iter()
            .filter(|r| r.server == "hub")
            .collect::<Vec<_>>();
        let cdn_requests = requests
            .iter()
            .filter(|r| r.server == "cdn")
            .collect::<Vec<_>>();
        assert!(hub_requests
            .iter()
            .any(|r| r.path.starts_with("/api/resolve-cache")));
        assert!(!cdn_requests.is_empty());
        assert!(
            hub_requests
                .iter()
                .all(|r| r.authorization.as_deref() == Some("Bearer secret")),
            "same-origin hops lost the token: {hub_requests:?}"
        );
        assert!(
            cdn_requests.iter().all(|r| r.authorization.is_none()),
            "token leaked to another host: {cdn_requests:?}"
        );
        for server in [hub, cdn] {
            server.task.abort();
        }
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn https_endpoint_never_follows_a_redirect_to_http() {
        let https = Url::parse("https://hf-mirror.com/a").expect("https url");
        let http = Url::parse("http://127.0.0.1:8080/a").expect("http url");
        let error =
            check_redirect_target(&https, &Url::parse("http://cdn.example/b").expect("url"))
                .expect_err("https to http is a downgrade");
        assert!(
            format!("{error:#}").contains("https"),
            "unexpected error: {error:#}"
        );
        check_redirect_target(&https, &Url::parse("https://cdn.example/b").expect("url"))
            .expect("https to https");
        check_redirect_target(&http, &Url::parse("http://127.0.0.1:9090/b").expect("url"))
            .expect("an http endpoint may redirect to http");
        check_redirect_target(&http, &Url::parse("https://cdn.example/b").expect("url"))
            .expect("http to https");
        assert!(
            check_redirect_target(&https, &Url::parse("ftp://cdn.example/b").expect("url"))
                .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn complete_partial_left_by_a_quit_during_finishing_is_verified_not_discarded() {
        let fixture = ChainFixture::new(5 * 64 * 1024 + 9);
        let hub = fixture.server("hub").await;
        hub.route(TEST_RESOLVE_PATH, TestHop::Serve { identity: true });
        let root = test_root("finish");
        let args = fixture.args(&root, format!("http://{}", hub.address));
        let identity = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: fixture.commit.clone(),
            path: "model.safetensors".to_string(),
            expected_size: args.expected_size,
            expected_sha256: fixture.sha256.clone(),
            etag: fixture.sha256.clone(),
        };
        prepare_transfer_cache(&args.cache_dir, &identity).expect("prepare cache");
        let partial = args
            .cache_dir
            .join("models--org--model")
            .join("blobs")
            .join(&fixture.sha256)
            .with_extension(HF_PARTIAL_EXTENSION);
        fs::create_dir_all(partial.parent().expect("blob dir")).expect("create blob dir");
        // Every byte arrived and the marker was dropped, then the App quit
        // before the verified file was moved into place.
        fs::write(&partial, fixture.payload.as_slice()).expect("write finished partial");
        let destination = args.destination.clone();

        run_async_with_progress(args, None, JsonProgress::default())
            .await
            .expect("finish the interrupted transfer");

        assert_eq!(
            fs::read(&destination).expect("read destination"),
            *fixture.payload
        );
        let data_ranges = fixture
            .requests()
            .into_iter()
            .filter(|request| request.range != "bytes=0-0")
            .collect::<Vec<_>>();
        assert!(
            data_ranges.is_empty(),
            "complete file was downloaded again: {data_ranges:?}"
        );
        hub.task.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn corrupt_complete_partial_is_discarded_and_downloaded_again() {
        let fixture = ChainFixture::new(3 * 64 * 1024 + 1);
        let hub = fixture.server("hub").await;
        hub.route(TEST_RESOLVE_PATH, TestHop::Serve { identity: true });
        let root = test_root("finish-corrupt");
        let args = fixture.args(&root, format!("http://{}", hub.address));
        let identity = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: fixture.commit.clone(),
            path: "model.safetensors".to_string(),
            expected_size: args.expected_size,
            expected_sha256: fixture.sha256.clone(),
            etag: fixture.sha256.clone(),
        };
        prepare_transfer_cache(&args.cache_dir, &identity).expect("prepare cache");
        let partial = args
            .cache_dir
            .join("models--org--model")
            .join("blobs")
            .join(&fixture.sha256)
            .with_extension(HF_PARTIAL_EXTENSION);
        fs::create_dir_all(partial.parent().expect("blob dir")).expect("create blob dir");
        fs::write(&partial, vec![0_u8; fixture.payload.len()]).expect("write corrupt partial");
        let destination = args.destination.clone();
        let cache_dir = args.cache_dir.clone();

        run_async_with_progress(args.clone(), None, JsonProgress::default())
            .await
            .expect_err("a corrupt complete partial fails verification");
        assert!(!cache_dir.exists(), "corrupt partial must be discarded");
        run_async_with_progress(args, None, JsonProgress::default())
            .await
            .expect("the next attempt downloads again");
        assert_eq!(
            fs::read(&destination).expect("read destination"),
            *fixture.payload
        );
        hub.task.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn restarted_transfer_resumes_every_completed_chunk() {
        let chunk = 64 * 1024_u64;
        let fixture = ChainFixture::new(48 * chunk as usize + 101);
        let hub = fixture.server("hub").await;
        hub.route(TEST_RESOLVE_PATH, TestHop::Serve { identity: true });
        let root = test_root("restart");
        let endpoint = format!("http://{}", hub.address);
        // The first attempt dies once 20 chunks are on disk (the App quits).
        let interrupted_at = 20 * chunk;
        fixture
            .fail_from
            .store(interrupted_at, std::sync::atomic::Ordering::SeqCst);

        run_async_with_progress(
            fixture.args(&root, endpoint.clone()),
            None,
            JsonProgress::default(),
        )
        .await
        .expect_err("first attempt is interrupted");
        let args = fixture.args(&root, endpoint);
        let partial = args
            .cache_dir
            .join("models--org--model")
            .join("blobs")
            .join(&fixture.sha256)
            .with_extension(HF_PARTIAL_EXTENSION);
        assert_eq!(
            committed_prefix(&partial, args.expected_size).expect("read marker"),
            Some(interrupted_at),
            "the partial file must keep every chunk written before the interruption"
        );

        fixture.log.lock().expect("log lock").clear();
        fixture
            .fail_from
            .store(u64::MAX, std::sync::atomic::Ordering::SeqCst);
        let progress = JsonProgress::default();
        let destination = args.destination.clone();
        run_async_with_progress(args, None, progress.clone())
            .await
            .expect("second attempt resumes");

        assert_eq!(
            progress.state.lock().expect("progress").first_emitted_bytes,
            Some(interrupted_at)
        );
        let data_starts = fixture
            .requests()
            .into_iter()
            .filter(|request| request.range != "bytes=0-0")
            .map(|request| range_start(&request.range))
            .collect::<Vec<_>>();
        assert_eq!(data_starts.iter().min(), Some(&interrupted_at));
        assert_eq!(
            fs::read(&destination).expect("read destination"),
            *fixture.payload
        );
        hub.task.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }
}
