# Flash-Next active sparse KV tiering (stage 8)

Status: experimental implementation; local functional and regression checks passed.
Final Release service smoke passed; physical low-memory acceptance is pending. This is opt-in experimental
active state storage, independent of durable prefix persistence. Request resume
following a process crash is not supported.

## Storage and execution contract

`--active-kv-offload` enables Flash's model-owned sparse QSA history store for
ordinary and DFlash2 inference. App/EnginePool uses the same existing active-KV
setting. The default remains off. Generic whole-request parking stays disabled
for Flash; Flash does not serialize live requests into the generic dense cache.

The qualified geometry is BF16 KV, 24 query heads, 2 KV heads, D256 and QSA ratio
4. Other geometry is rejected before an enabled model starts serving. Raw
indexer history, pooled index keys, positions, recurrent states, PLE history and
DFlash2 sliding context stay in memory. Each sparse layer starts with resident
KV; once its history exceeds `max(4096, read_cache_bytes / token_kv_bytes)`
tokens, its KV history moves to immutable, 4-token SSD records. A small partial
page remains in RAM. Selected historical blocks are gathered, never the entire
long history on every token.

QSA prefill tiles remap selected block indices and causal query ends together,
and bound the union payload before dispatch. Outputs are evaluated between
tiles to release gather buffers. Decode preserves the pinned MLX two-pass
SDPA arithmetic: logical token positions retain their original reduction block,
FP32 accumulation and BF16 partial rounding. The local indexed kernel adapts
MIT-licensed MLX code from `65383d68c` and preserves its architecture-dependent
block choice. Rollback below 4096 temporarily uses bounded original-shape native
SDPA. The indexed kernel is specific to the pinned MLX reduction contract;
future MLX upgrades need this numerical gate again.

The initial compact-SDPA candidate was rejected: a full-target sample differed
by 6.2238% relative RMS (max absolute logit difference 0.546875). This was not
accepted by relaxing a tolerance. The replacement retains native logical
reduction placement; tested samples are required to match exactly.

## Budgets and configuration

Initial tiers use the planner's **available model budget**, not physical RAM
alone. For an otherwise idle 32/64/128 GiB machine reserving 8 GiB, this maps to:

| Available model budget | Per request, per sparse layer read cache | Selected gather payload |
| --- | ---: | ---: |
| <=24 GiB | 1 MiB | 32 MiB |
| >24, <=56 GiB | 4 MiB | 64 MiB |
| >56 GiB | 16 MiB | 128 MiB |

These are initial budgets, not measured hardware performance profiles. A query
union that exceeds the gather limit splits into smaller tiles; one query that
cannot fit fails validation. Read-cache pressure evicts LRU payloads, while
in-flight gathers retain their own page data. Snapshot/fork leases pin disk
records, and released slots can be reused without altering any referenced
state. The minimum resident threshold is 4096 tokens on every Apple GPU; the
128 GiB initial tier reaches 8192 tokens. Threshold and byte budgets are exposed
in health data.

Load/admission planning charges 16,384 bytes per logical token plus 3 GiB fixed
per sequence in tiered mode, compared with the existing conservative RAM-only
196,608 bytes/token plus 2 GiB. This is a reservation envelope, not a measurement
of process RSS or physical GPU residency. Model weights, expert/PLE caches,
vision weights and shared temporary workspaces remain separately charged. The
fixed tiered envelope includes an extra GiB so eight warm-KV generations plus
recurrent snapshots do not borrow the shared temporary reservation. It and the
bounded gather cover retained index/recurrent state,
rollback snapshots and transitional buffers. Gather payload limits do not mean
that total transient memory equals one payload: packing and GPU layouts may
briefly coexist.

Per-model SSD quota defaults to 64 GiB; free-space reserve is 1 GiB. Startup
checks configured context/concurrency against ratio-sized records plus checksum
headers and branch tail allowance. Runtime checks actual allocation/headroom;
quota/disk/read corruption failures fail the affected request, rather than
silently allocating an unbounded RAM fallback. They also increment health error
counters. Numeric overrides, only when active KV is enabled:

- `IRONMLX_FLASH_ACTIVE_CACHE_MIB` (8 KiB minimum, 16 MiB maximum; integer MiB via env).
- `IRONMLX_FLASH_ACTIVE_PACK_MIB` (up to 256 MiB; Flash requires at least
  8 MiB and enough for its configured warm history, so a 16 MiB read cache
  requires at least a 16 MiB gather budget).
- `IRONMLX_FLASH_ACTIVE_DISK_GIB` (positive; constrained by planned context).

These overrides change bounded storage parameters, not the model architecture.
The root is `<active-kv-offload-dir>/flash`. Model and prefix files are read-only.

## Transactions, cancellation and cleanup

Each record contains generation id, SHA256 and BF16 payload. A page batch is
published to logical history only after every write succeeds. Slot reuse uses a
new generation id; old cached contents cannot alias a new lease. COW page lists
and partial tails preserve independent B1/B2/B4 rows and speculative rollback.
Prefetch pins existing hits before reading misses on the shared bounded CPU I/O
pool; no MLX arrays cross those worker threads. macOS scratch files use
`F_NOCACHE`; writes are coalesced when slots are adjacent. There is no fsync
journal and no claim of crash durability.

Files and quota/cache accounting are released when their last request/snapshot
lease drops. Each model owns a private 0700 session, 0600 files and a flocked
owner marker. Startup removes only recognized, unlocked stale sessions, never
live sessions or unknown directories. A forced process kill may leave disposable
files until the next store startup reclaims them.

## Prefix persistence interaction

RAM-only prefix behavior is unchanged. Enabled models use a distinct execution
fingerprint, including storage version and budgets, preventing reuse across
unqualified configurations. Small complete prefixes can materialize bounded KV
and reuse the existing versioned durable codec. Long optional exports/lookups
are skipped when the full payload exceeds **512 MiB** or a sparse layer's KV
exceeds the gather budget. `prefix_export_skips` and a debug reason expose this
choice. In particular, active tiering does not promise large-prefix reuse on
128 GiB hardware. Streaming the durable codec directly from pages is future
work; this conservative limit avoids reconstructing long KV in memory merely
to save it. Prefix persistence is independent of active-state scratch files.

## Observability

`paging.active_state` includes enabled policy, live files/pages, allocated/peak
SSD bytes, charged read-cache RAM, cache/gather/quota budgets, resident-to-paged
threshold, cumulative read/write bytes/pages/time (reader-worker durations may exceed
wall time), hit/miss rate, failures and
skipped optional prefix exports. These are application I/O counters, not
physical SSD throughput, system page-cache misses or total device writes.
`active_kv_offload.mode=flash_sparse_history` reports idle/active/degraded from
live leases and failures; completed writes alone do not keep Flash marked
active. Memory policy is `flash_sparse_ssd`. Dashboard exposes the same storage
stats and limitations in all five supported languages. The read-cache budget is
per request/layer, while used RAM/SSD counters are model aggregates.

## Qualification

Host: M5 Max, 128 GiB, macOS 26.4; pinned MLX `65383d68c` on
`fix/main-thread-identity`. Target checkpoint
`d5fe7052703f142f839108bb1ace5486de238ab1`, draft
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`. Release builds. Logs and raw
measurements: `.build/flash-active-state/` (local artifacts, not source files).

The full-target fixed-trajectory comparison at 4099 tokens and eight decode
steps is bit-exact with RAM storage, including the native B2/B4 expert path
using a 2 GiB expert cache. The B2/B4 target verification test uses ragged base
lengths 4093/4096/4097/4100 and Q8, with independent B1 continuations. The B1
DFlash2 test checks accept lengths 0/1/4/8 at 4096 tokens. Canonical persistent
KV, raw index, positions and recurrent/PLE state are compared; pooled index
keys are a derived cache intentionally invalidated on trim, so their presence
is not itself a semantic state difference. Continuation hidden states, capture
states and logits must match exactly after reconstruction. Both DFlash2 tests
released all active files, disk reservations and read-cache charges, with no
I/O errors. The tests deliberately use a 64 KiB per-file read cache to force
misses; their aggregate runtimes and hit rates are not performance profiles.

Cross-process 4097-token prefix recovery passed both synchronous and adaptive
save modes, with exact B1/B2/B4 full-vocabulary continuations. The synchronous
sample's 232,653,290-byte payload took 0.285 s to commit and 0.112 s to load;
these local values exclude export materialization, are warm filesystem samples
and were not a controlled throughput benchmark. CPU/GPU regression jobs ran
alongside these functional tests, so their prefill durations are not used for
performance claims.

### Findings resolved during implementation

- Compacted decode changed native BF16 partial grouping and failed the full
  model numerical gate. It was replaced, not accepted under a looser tolerance.
- Direct array-count comparisons were invalid once RAM/SSD representation and
  a cleared derived index cache differed. The test now compares canonical
  persistent tensors and exact subsequent execution.
- A 64 MiB expert cache full-model stress run was stopped without completing
  after over six minutes. A legacy DFlash2 fixture also used a 64 MiB expert
  pack ceiling; sampling found expert preparation/packing on its active stack.
  Long-context qualification now uses the production 2 GiB pack ceiling and
  a 2 GiB expert cache. These incomplete runs remain in local artifacts and
  are not reported as passes or physical SSD throughput measurements.
- An initial planner assertion expected RAM-only 64K to reject at 24 GiB
  available. The planner can instead reduce expert residency to admit it.
  The corrected comparison uses 128K, where the RAM-state envelope alone
  exceeds that budget. At this boundary tiered planning admits a peak
  reservation of 19,154,317,304 bytes. This is not a real 32 GiB execution test.
- Tiered fixed reservation is 3 GiB per sequence, shared by ordinary and
  DFlash2 admission. Warm-state/verification generations are not charged to
  the same shared workspace as expert and gather packing.
- A gather budget sufficient for a sparse query can still be too small for
  bounded native-shape rollback across the paging threshold. Model preflight
  now checks both requirements. Large appends are written in bounded batches;
  no logical history is published if a later batch fails. Startup and runtime
  both check SSD free-space reserve.

Final automated counts, boundary tests and service measurements are listed
below. Real M1 Pro 32 GiB, real
64 GiB and cold physical SSD qualification are unavailable on this host.
Simulated memory planning and small-cache stress tests must not be represented
as those hardware results. Earlier intermittent prefix-latency investigation
remains independent and unresolved; its three local files are preserved.

### Remaining performance and hardware limits

The low-buffer component stress case (64 KiB read cache, 8 MiB gather,
6147-token prefix plus decode/rollback/forks) was exact but read
5,800,160,184 bytes for 12,743,136 written bytes and took 32.72 s on this host.
Its cumulative reader-worker time was 59.37 s, not wall time. A separate
8195-token component case with the large tier's 16 MiB cache and 128 MiB gather
was exact, read 559,776 bytes and took 0.90 s. Contexts and budgets differ;
this is evidence of budget-dependent read amplification, not a controlled
cache-policy speedup or full-model benchmark. In particular, the 64 KiB/8 MiB
case is not the default 32 GiB profile.

Prefill unions that require many gather tiles can repeatedly read the same
pages when the read cache is small. Streaming prefill accumulation or better
reuse across tiles remains a performance follow-up. The current implementation
bounds memory and fails safely; it does not promise high throughput for every
admitted context/budget. Genuine 32/64 GiB memory pressure, swap, cold device
I/O and long-context multi-request performance still need physical hardware
qualification. The full target was not run at 128K/256K in this round; synthetic
indexed-decode checks at 256K prove arithmetic only.

### Automated and service results

- Stable and pinned nightly formatting checks passed. CI-scope
  `cargo +nightly-2026-05-18 clippy --locked --all-features --workspace -- -D warnings`
  passed. Release all-feature workspace tests: **2553 passed, 0 failed,
  308 ignored** (real-model/opt-in tests are listed separately, not counted as
  automatic successes). Frontend Node tests: **85 passed**.
- The 6147-token narrow-budget and 8195-token large-tier real-weight QSA tests
  passed exact prefill, decode, four forks and rollback across 4096. The
  separate GPU boundary test passed a multi-batch append, two Q1 prefill tiles,
  later-batch quota failure without logical mutation, retry using freed slots,
  and final quota/file release. Indexed/native synthetic decode was exact at
  4096, 8193, 16384, 32769, 65537 and 262144 tokens.
- The final full-target 4099-token test passed exact RAM/SSD full-vocabulary
  logits, B2/B4 row isolation and native B4 expert microbatch execution using
  a 2 GiB expert cache. Ordinary and DFlash2 admission interfaces reported the
  same 16 KiB/token plus 3 GiB/sequence reservation. Metadata-only capacity
  tests passed 32/64/128 GiB budgets and reject undersized gather settings.
- Loaded EnginePool reservation and draining accounting now use the active
  storage contract, without taking the model mutex. A regression test covers
  both full expert residency with active KV and the RAM-only fallback; the
  earlier RAM constants would over-reserve an enabled model after loading.
- Initial Release HTTP matrix passed ordinary/DFlash2, text/images, concurrent
  submission of 2/4 requests, cancellation, a subsequent request and file
  cleanup. The 4834-token prompt's single-request output matched storage-off
  controls exactly. B2/B4 completed eight output tokens per request; rows with
  matching B1 controls matched those controls. Health polling observed
  `idle -> active -> idle`. After completion/cancellation, active files and SSD
  reservations were zero and I/O failures stayed zero. Concurrent submission
  is not a new qualification of the DFlash2 tensor-batch UI switch.

Initial HTTP binary SHA256:
`29ce4109b0efef26a3ba7e4360220077ed363c51f2b81a126ac94657409f0cb4`.
The binary and raw JSON/logs were retained locally. Final-boundary changes add
write batching, preflight checks and corrected EnginePool accounting; a final
binary service smoke repeats ordinary/DFlash2 text and image checks below.

### Bounded latency observations

4834 input tokens, eight output tokens, full expert residency, ANE off,
M5 DFlash2 auto profile, chunk 2048, logical cap 8192, max requests 4, prefix
caching off. Enabled cases force **1 MiB read cache / 16 MiB gather**. These
are pressure settings on a 128 GiB host, not that host's default 16/128 MiB
profile and not a real 32 GiB test. One health request after the first token
is included in total duration. No concurrent builds/tests ran during these
HTTP measurements. There are only two single-request samples per case;
"first" means first long request in that process, not cold physical SSD.

| Route / storage | First TTFT / total seconds | Subsequent TTFT / total seconds |
| --- | ---: | ---: |
| Ordinary / RAM | 15.126 / 25.108 | 6.352 / 16.314 |
| Ordinary / SSD | 23.036 / 34.013 | 6.812 / 17.585 |
| DFlash2 / RAM | 20.954 / 22.529 | 6.181 / 7.298 |
| DFlash2 / SSD | 20.365 / 29.622 | 9.018 / 14.857 |

This data does not establish a speedup. The warm DFlash2 pressure sample took
about twice the total time of its RAM control. Initialization/weight-residency
and previously unresolved latency variation also affect these observations;
not all differences can be causally attributed to page I/O. No uniform latency
regression bound is claimed. RAM-only inference remains the default; this
experimental mode trades I/O and latency for a smaller resident-state envelope.

Final Release binary SHA256:
`621660a62fd8f21627978214710504627ccf3e470b286058aee9b22e11481a10`.
Final text/image smoke passed for ordinary and DFlash2, with identical output
to the earlier RAM controls, zero active files/allocated SSD bytes after each
request and zero I/O failures. With the same pressure settings, final enabled
samples were:

| Route | First TTFT / total seconds | Subsequent TTFT / total seconds |
| --- | ---: | ---: |
| Ordinary / SSD | 25.299 / 36.438 | 7.101 / 18.096 |
| DFlash2 / SSD | 20.836 / 29.758 | 8.575 / 14.896 |

The earlier complete HTTP matrix and the final smoke have different binary
identities, recorded above; the final boundary fixes and pool accounting do
not change the allowed 1/16 MiB storage contract used by these samples. These
small runs remain observations, not a controlled performance acceptance or
a latency-tail fix.

No App bundle was installed/restarted and no commit or push was made. The
three held latency-investigation files were byte-for-byte preserved. The MLX
checkout was not changed. Release CLI and test artifacts are local under
`.build/flash-active-state/`; they are not included in the source change.
