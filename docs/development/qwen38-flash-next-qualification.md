# Qwen3.8 Flash-Next implementation and qualification

This branch follows the migration-first order: MLX pin and compatibility,
existing-model regression, then the Flash text graph, bounded storage,
DFlash2 transactions, request scheduling, hardware policies and multimodal work.
An implementation stage and its qualification are separate gates.

## Fixed identities

- MLX main: `a8f66b5901791a38ff527f249e1f1a84f46c03b6` (0.32.5).
- MLX-VLM reference: `ff788e98d82f6a931992a6a8ee0e74c7a7e6925e`.
- Target: `mlx-community/Qwen3.8-Flash-Next-4bit`, imported standalone snapshot
  `d5fe7052703f142f839108bb1ace5486de238ab1`. The upstream HF revision is
  `07b5dc6c54600a359b87f1e53e7adf6351c72a2c`; its config matches this import.
  The standalone snapshot ID is an import identity, not the HF commit.
- Draft: `incoai/Qwen3.8-Flash-Next-DFlash2`, snapshot
  `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
- Draft author's capture patch SHA-256:
  `91708e18c67f1c0e8f6c19cfbe7cc9f59ff257c05d61bb30b407ae4c36354169`.

The author draft consumes the arithmetic mean of the raw completed HC streams
at zero-based layers 5, 15, 25, 35 and 45. It uses its own RoPE theta 10,000;
the target uses 10,000,000. This is a pair-specific contract. Other model
families retain their existing matching requirements. The first qualified
draft precision is BF16; runtime quantization needs separate qualification.

## Reference comparison

| Implementation | Useful evidence | Boundary |
| --- | --- | --- |
| MLX-VLM, pinned above | Native MLX target, HC, PLE, GDN, QSA, Qwen3-VL vision | Full-model numerical oracle; author draft capture still needs its patch contract |
| Splash, `a56988e7` | Native Flash + DFlash serving and scheduling | Its target is GGUF; it does not prove this affine-4 MLX loader or arithmetic |
| oMLX, `5c6cbd63` | Vendored native MLX `qwen4_exp` target; separate GGUF adapter | Native MLX DFlash allowlist does not qualify `qwen4_exp` |
| TensorFold, `94b90c9d` | Native Zig Flash implementation; affine group 32/64/128; storage ideas | Primarily MTP evidence, not this DFlash2 pair |

These are source assessments at the pinned revisions, not cross-product
performance claims. Rival checkouts remain unmodified.

## Storage and capacity

The checkpoint contains 111,518,894,072 tensor payload bytes: roughly 29.80 GiB
of PLE rows and 70.31 GiB of routed experts. Text execution never materializes
the complete PLE table. It uses immutable positional reads with snapshot/file
identity checks and a separately bounded row cache and staging allocation.
PLE history and convolution state belong to requests.

Before creating weight arrays, `FlashLoadPlan` inventories tensor headers and
reserves fixed weights, the PLE cache, optional draft, KV/indexer/recurrent
state, transaction overlap, expert packing/staging and temporary work.
`auto` chooses full expert residency only if the whole envelope fits; otherwise
it chooses the bounded SSD expert adapter. Full residency fails before weight
materialization when the envelope does not fit. An expert lease stays pinned
until its submitted graph is evaluated. SSD is read-only and does not rewrite
or quantize checkpoint files.

Advanced process configuration (App and CLI share the same planner):

```sh
export IRONMLX_FLASH_EXPERT_RESIDENCY=auto  # auto | full | ssd
# Optional explicit upper bound in SSD mode:
# export IRONMLX_FLASH_EXPERT_CACHE_MIB=4096
export IRONMLX_FLASH_PAGED_BATCHING=auto  # auto | on | off (on is explicit diagnostic opt-in)
export IRONMLX_FLASH_PREFETCH=auto        # auto | off (one known next expert)
export IRONMLX_FLASH_IO_POLICY=cached     # cached | no_cache (macOS F_NOCACHE)
```

Default SSD expert caps are 4/8/16 GiB for usable load budgets of at most
24/56/more than 56 GiB respectively. These tiers use the remaining planner
budget, not the machine label; other models, host/process pressure and Metal's
recommended working set can reduce the actual cap further. Explicit overrides
still cannot exceed the complete plan. The existing conservative request-state
cost is retained. Expert packing remains capped at 2 GiB inside the existing
4 GiB temporary envelope; lowering only the packing cap would not reduce this
reservation and can increase repeated packing. An additional 32 MiB covers
bounded raw loading/prefetch overlap.

A physical M1 Pro with 32 GiB is **not qualified** by a dry run on this host.
The initial candidate is ordinary B1 with 4K context and a 2–4 GiB expert cap,
then 8K and DFlash2 only after measuring real headroom. The model/vision, draft,
request state, staging and compute budgets all still apply. Default B2/32K
cannot fit a 24 GiB load budget even before experts; rejection is intentional.
No context or concurrency is silently lowered to pass admission.

`POST /admin/api/models/load-plan` accepts the normal load request and returns
an advisory header-only Flash plan, without registration, eviction or weight
materialization. App loading displays its planned reservation and available
budget. Actual loading rechecks admission; the preview reserves nothing.

Expert eviction protects repeated decode accesses and inserts prefill scans
into the unprotected segment. Scan hits do not refresh decode recency. Pins
always win over eviction; pressure can still require eviction or a smaller
execution group. There is no claim that scan protection eliminates I/O when
the working set exceeds RAM. PLE rows share a packed payload slab rather than
one payload allocation per row. At width 160, a row has 100 payload bytes and
256 bytes of conservative metadata charge (previously 1,024). A 64 MiB budget
therefore provides 188,508 slots rather than 59,705; this is a capacity/accounting
comparison, not an RSS measurement.

Two bounded CPU readers perform positional reads. MLX array construction and
execution stay on the GPU owner thread. Expert reads/materialization and pack
evaluation occur outside the metadata lock. All-source validation is performed
at pack/gather boundaries, and individual reads retain before/after descriptor
stamp checks. PLE lookups deduplicate misses and coalesce adjacent rows within a
shard. Known-route prefetch holds at most one raw expert, drains on drop, and
never predicts a future layer's router. `no_cache` is an explicit A/B policy;
it does not establish controller-cold SSD throughput.

Native batching `auto` is enabled only for the validated Apple M5 Max GPU.
Other/unknown GPUs retain per-request execution, with an explicit health reason.
`on` is an experimental qualification override and is not validation of M1–M4
or other M5 models. SSD loading and low-memory budgeting remain available.

Paging accepts native B2/B4 inputs and shares expert packs/execution across
requests. Routing and shared projections retain each request's B1 shape.
Reference B1 working-set pieces are interleaved by request and grouped only
within compatible broadcast/QMV/QMM classes. This preserves dispatch thresholds
when unrelated requests are combined. Groups also obey the cache/packing cap;
a B4 input can execute as smaller groups. Attention, PLE history and recurrent
states remain request-owned. Verification records token-level snapshots and
supports ragged commits/rollback. Native target MoE batching does **not** certify
DFlash2 draft tensor groups; the existing separate draft capability gate stays.

Per-model `paging` snapshots are available in loaded-model/health APIs and CLI
health, including ordinary and DFlash2 engines. Reads of these atomic snapshots
take no paging-store lock and evaluate no MLX graph. Expert and PLE counters are
separate; prefill/decode/verify each expose hits, misses, evictions, successful
application payload bytes, cumulative read/packing/wait time, route counts,
input width, actual expert execution width, microbatches, budget/arithmetic
splits and prefetch scheduled/used/unused counts. Instance identity and sampling
clocks distinguish reloads. Gauges include charged/payload/peak/pinned/pending
bytes and budgets. Explicit disabled/token-major fallback reasons are counted.

The Dashboard transports these fields and localizes summaries/details in five
languages. Recent rates are deltas over observed samples, up to 30 seconds,
using monotonic elapsed time. Empty samples show no data; full expert residency
marks the expert cache inapplicable while PLE remains visible. Expert hits count
unique lookups **per execution microbatch**, whereas PLE counts row lookups.
Changing cohort boundaries changes the hit-rate denominator: compare bytes,
wait and request performance as well. Application reads can hit the OS cache;
read/pack timings can overlap or include each other and must not be summed as
an end-to-end latency breakdown.

The ordinary App path uses BF16 KV for Flash even when other models use a
global TurboQuant setting. Legacy Active KV swapping and persistent prefix
formats do not contain the complete Flash auxiliary state, so they are not
used for this architecture. The default App context remains 32K; explicit
larger requests must pass the complete memory plan.

## Completed numerical gates

Host: Apple M5 Max, 40 GPU cores, 128 GiB RAM, macOS 26.4. These results do not
claim validation on a physically smaller-memory machine.

- Existing Dense 27B and MoE 35B B1/B4 migration regression; Dense/MoE DFlash2
  rollback; g32/g64 NAX/QMM shapes; strided/windowed D256 attention; ANE bridge.
- Correct inference-mode (`eval`) primary oracle. All 48 prefill HC captures
  and 24 teacher-forced complete-model logits match exactly for the fixed
  short text trajectory, both resident and forced 64 MiB expert cache modes.
- Real HC, GDN and PLE components match exactly. A 2053-token QSA component
  crosses the 2048 sparse budget; output, KV, raw indexer and pooled blocks
  match exactly after using the primary reference's power-based RoPE frequencies.
- A 2,398-token prefill plus four fixed-trajectory logits: all 48 HC captures
  and complete logits match exactly in both full-resident and SSD modes.
  In forced SSD mode with a 2 GiB expert cache and 2 GiB packing budget,
  MLX peak was 12,090,880,688 bytes; the expert cache recorded 16,478 misses,
  15,780 evictions and 698 resident entries. This is an MLX allocation counter,
  not total process/machine RAM or a cold SSD throughput measurement.
- Forced 64 MiB expert cache: 13,963 misses and 13,942 evictions on the short
  trajectory, exact logits throughout. MLX peak was 3,483,641,240 bytes.
  This is a forced-budget test on a 128 GiB host, not an SSD cold-throughput test.
- B1 DFlash2 Q2/Q4/Q8 at contexts 8 and 64; acceptance 0/1/4/8; EOS in the
  proposal; all state arrays and continuation match ordinary Q1 execution.
- B2/B4 Q8 with independent contexts and ragged acceptance [0,1,4,8]: hidden
  states, author HC captures, committed state and continuation are exact.
- Actual ordinary/DFlash2 generation streams match tokens and finish reasons
  for budgets 17 and 3, including multiple real draft windows.

Reproduction tools live in `scripts/generate-qwen38-flash-next-*.py`,
`scripts/qualify-qwen38-flash-next-*.py` and the opt-in Rust `flash_*_real`
tests. Model paths and oracle artifacts are supplied through explicit
`IRONMLX_FLASH_*` variables. Generated large oracle tensors stay outside Git.

## Hardware policy

Flash g32 weights and two KV heads have an independent profile; Dense/MoE g64
settings are not inherited. Generic kernels remain the fallback. A native-layout
M5 MPP g32 projection candidate passed row-invariance and numerical checks but
measured slower than MLX QMV for all six tested projection geometries, including
the vocabulary head. It remains an ignored qualification benchmark, absent from
production routing. No acceleration claim follows from merely compiling on M5.

Flash's shared and routed experts execute on the GPU in ordinary and DFlash2
paths, including SSD-paged experts. The experimental Flash ANE implementation
and its diagnostic switches have been removed. The independent Dense g64 ANE
backend and global `auto`/`off` policy remain available for qualified 27B models.

## Serving and image gates

- Actual HTTP ordinary and DFlash2 (M5 off/auto) text serving passed independent
  B1/B4 output checks, SSE cancellation, immediate slot reuse and final idle
  accounting. The initial qualification used a forced 64 MiB expert cache.
  Concurrent qualification jobs make those HTTP timings unsuitable for a
  controlled throughput comparison.
- A pinned image oracle covers the Qwen3-VL tower, three-axis positions, all
  48 target HC captures and four complete logits: all match exactly.
  Reference tower attention pads D72 to D80; inherited D96 NAX padding and
  fused rotary arithmetic are not used for Flash. Flash uses native 3D patch
  convolution and MLX linspace interpolation; flattened GEMM can differ at
  BF16 boundaries for non-square images. The merger's exact GELU
  preserves the primary weak-scalar BF16/compiled arithmetic. Existing model
  families retain their prior vision paths.
- Unresized 320×224 and resized 321×225 patterned RGB images match the
  checkpoint-bound primary processor and the entire vision tower exactly.
  Flash uses NumPy-style float32 reciprocal rescaling; older Qwen families
  retain their existing division-based normalization. The 320×224 fixture also
  passes the complete language-model gate: three-axis positions, all 48 HC
  captures, four full logits, native tower-to-logit composition and speculative
  rollback continuation are exact in forced SSD mode.
- Image positions belong to QSA snapshots, row adoption and prefix rollback.
  Real speculative rollback after an image and continuation match Q1 exactly.
  A second 2,119-token image/text oracle crosses the sparse QSA budget: the
  tower, all three position streams, all 48 captures, four complete logits and
  speculative rollback continuation remain exact with bounded SSD experts.
  GenerationStream ordinary/DFlash outputs also match with image prefill split
  into eight-token chunks, for budgets 17 and 3.
- First cold image HTTP testing exposed a nested cold-materialization permit
  wait in chunked scheduler prefill. The outer synchronous prefill now owns the
  permit scope; nested chunk helpers reuse it. A regression exercises the cold
  vision component and verifies Warm state plus zero outstanding reservations.
- Full residency testing caught duplicate cold-weight admission: current MLX
  `Load::eval_cpu` allocates/reads buffers and `Loader::open_impl` eagerly evaluates
  retained weights. Flash text/draft buffers are already charged in MLX active
  bytes and its residency plan; only the deferred vision tower is an additional
  cold liability. Reserving the roughly 80 GB target a second time is rejected
  and is no longer done for Flash.
- Live App EnginePool testing additionally caught the large eager-load
  reservation being counted alongside the already materialized weight footprint.
  Flash now uses a model-load reservation class: the full envelope still blocks
  competing admission, while pressure/eviction classification uses actual
  footprint plus other reservation classes. Genuine physical pressure is still
  detected. Synthetic regression covers growth, commit, failure and admission
  conservation; fixed-model CLI serving alone did not exercise this pool path.
- Compact expert dispatch preserves the source expert count (512) when choosing
  the upstream `B/E >= 4` sorted-RHS QMM route. Merely reducing the resident RHS
  table must not change the QMV/QMM arithmetic during image-prefill chunks.
- The default App-sized image prefill exposed a second partition mismatch:
  ordinary scheduling uses `[N-1] + [1]`, while the shared image helper absorbed
  the final text token into `[N]` for DFlash. Flash now retains that scheduler
  suffix; truly chunked image-pad/tail absorption and GenerationStream behavior
  are unchanged. Explicitly different prefill partitions may change BF16
  trajectories, so parity claims compare the same partition contract.
- Flash's exact Qwen XML tool template now enables the existing byte-level
  constraint tokenizer and native tool parser. Live ordinary and DFlash tests
  cover sampled requests, strict JSON, forced tool calls and a tool-result turn.
- Actual ordinary HTTP with mixed text/image and multi-image requests passes
  B1/B4 token/message/finish-reason equality, image SSE cancellation and immediate
  row reuse. Flash image rows retain the B1 prefill chunk boundaries before
  their private caches are assembled into the scheduler batch. Model-management
  capability metadata exposes images only with the checkpoint's vision config.

The final mixed HTTP matrix also compares full-resident ordinary decoding,
full-resident DFlash2 and forced SSD DFlash2 (8 GiB expert cache, M5 auto
and off/ANE off): four B1
requests, the same four concurrently, image SSE cancellation and two immediate
reuse requests. Messages, token counts and finish reasons agree across modes;
all actors return idle without admission failures or outstanding ragged
reservations. The M5 profile actually built and scattered four ragged groups,
reached width four, emitted 44 tokens and accepted 30 draft tokens in those
ragged windows. Those counters establish execution, not a speedup claim.

Layer-major verification keeps each projection and attention at its Q1 shape;
B1/B2/B4, Q2/Q4/Q8, ragged acceptance, captures and continuation remain exact.
On one short full-resident trajectory, token-major DFlash took 1.898 seconds,
layer-major 1.169 seconds, and ordinary decode 0.663 seconds. The separate
sequential-verifier cost policy then measured ordinary decode before narrow
probes, reducing this DFlash run to 0.697 seconds versus 0.700 seconds ordinary.
These are bounded single-run observations, not a general acceleration claim.
A three-token request may remain ordinary except for a narrow recovery probe.

The production planner reserves a 2 GiB expert packing workspace, QSA scratch
and 4 GiB of temporary overlap plus 32 MiB of raw I/O overlap. It reserves the deferred vision tower before
advertising image input. EnginePool includes future bounded-cache/request
ceilings when planning another model and retains Flash liability while an
unloaded slot still has active request leases.

Video input is not exposed by the existing HTTP image contract. Native image
support must not be advertised as video support. Lower-memory physical hardware/older Apple GPUs still
need independent on-device qualification. No standalone manifest is rewritten
merely to upgrade an older download-only import into a newly supported model.

## Repository checks

The earlier Flash implementation checkpoint passed 1,755 Rust unit tests across CLI, core, LM and runtime,
569 Swift App tests, the workspace/all-targets Release check, supported-catalog
regeneration, reproducible third-party materials (17 verification tests), and
the deterministic CycloneDX SBOM including the pinned GDN reference kernel.
Model-backed GPU/HTTP gates above are separate opt-in checks; ignored tests
are not counted as passes. Release App packaging and live App acceptance use
the same pinned MLX source.

The App's fixed DFlash startup has a separate 300-second readiness ceiling:
weights load before HTTP opens, whereas the ordinary lazy daemon retains its
5-second budget. Process exit, generation replacement and planned stop interrupt
the readiness wait; waiting longer must not hide a dead helper or override Stop.
Live App startup is part of acceptance, beyond direct CLI serving.


## SSD paging stages 1–6 qualification (2026-11-03)

Scope: model/API/Dashboard paging observability, lower-memory load planning,
scan-resistant expert retention, bounded parallel reads/prefetch, packed PLE row
storage, and native paged target MoE B2/B4. Persistent Flash prefixes and active
KV/request-state offload (stages 7–8) remain unimplemented. KV and recurrent
state still require RAM. All runtime results here used the pinned target/draft
above on Apple M5 Max (40 GPU cores), 128 GiB, macOS 26.4, MLX main `a8f66b590`.

### Checks and actual-model gates

- Final workspace Release Rust suite: **2,512 passed, 0 failed, 301 ignored**
  across 209 test targets. Ignored hardware/model tests are not passes.
- Swift Release: **574 passed**. Dashboard paging Node tests: **4 passed**.
  Both stable and CI-pinned nightly formatting, CI-pinned nightly Clippy with
  warnings denied, reproducible third-party verification (17 tests), and SBOM
  verification passed.
- Separate opt-in model runs passed the full-model/reference, real vision,
  direct paged B2/B4 (sequence lengths 2/127/205/1024), ragged request-state,
  DFlash2 acceptance/rollback, ordinary/DFlash text and image generation,
  disabled-native control, `F_NOCACHE`, and prefetch-disabled gates.
  Dense 27B ANE still completed 64 jobs and released its surfaces after unload.
- B4 long prefill used independent lengths 63/127/1024/2049 and a 2 GiB expert
  cache. Prefill and two continuation steps matched independent B1 logits
  elementwise. Arithmetic compatibility split this input into actual B2
  prefill groups; subsequent decode reached actual B4. A direct B4 input does
  not promise a four-request GPU group for every cache budget or shape.
- Actual HTTP used 512 MiB expert cache, BF16 draft, context cap 4096,
  prefill chunk 128, max four sequences, and ANE off. Independent B1 references,
  B2/B4 messages/token counts/finish reasons, SSE cancellation and immediate
  slot reuse passed. Ordinary auto reached actual prefill/decode width four;
  DFlash2 default auto reached target verify width four. Draft tensor batching
  remains separately unqualified and unavailable. Ten-Hz health polling had
  zero errors; observed p95 was about 2.4–4.7 ms in these runs.

The final Release App was rebuilt with the same pinned MLX, then its default-
feature helpers were rebuilt/repackaged and ad-hoc signed after the scheduler
fix. Static bundle verification passed. The actual bundled helper passed both
ordinary and default DFlash2 mixed text/single-image/multi-image B1/B4 fixtures,
with an 8-token output cap, exact messages/token counts/finish reasons, image
SSE cancellation and immediate row reuse. Outputs also matched across routes.
Final counters showed ordinary decode width four and DFlash2 target verify
width four, image prefill width one, idle actors and zero admission failures.
Metadata-only preview edge cases (resource directory without `config.json`,
Flash before registration, and an already loaded Flash) passed. Source-side
Dashboard/bridge tests passed; this is not a manual visual Dashboard acceptance.

### Bounded measurements

These are sequential single-run observations, not a randomized benchmark or
cold SSD test. OS page cache was not purged; application read bytes are not
physical disk traffic. Comparing row fallback with native auto combines the
new batching/cohort policy, not just the replacement cache policy.

| Workload | Row fallback | Native auto | Interpretation |
| --- | ---: | ---: | --- |
| Long B4 prefill, 2 GiB experts | 41.919 s | 22.456 s | Same fixed inputs; actual prefill width 1 versus 2 |
| Long prefill expert application reads | 97,274,880,000 B | 38,913,024,000 B | Roughly 60% fewer bytes requested by the application |
| Ordinary HTTP B2, 512 MiB experts, 8-token cap | 12.294 s | 9.290 s | Same request/output fixture |
| Ordinary HTTP B4, same settings | 27.403 s | 17.058 s | Actual native width four |
| Default DFlash2 HTTP B4, same settings | 30.066 s | 28.549 s | Correct outputs; still slower than ordinary native |

The long-run MLX peak was about 12.45 GiB; this is an allocator counter, not
process RSS, machine memory, or proof of operation on a 32 GiB Mac. Hit-rate
changes also reflect a different unique-expert-per-microbatch denominator.
Do not attribute the observed difference to scan protection or prefetch alone.

### Problems found and remaining qualification

1. **Fixed numerical issue:** a first naive cross-request concatenation changed
   QMV/QMM dispatch thresholds and failed the long B4 exact-logit gate. Routing
   now preserves B1 shapes and combines only compatible reference pieces.
   The original failed log is retained alongside the passing reruns.
   Final bundled-helper mixed-image B4 testing additionally caught a scheduler
   gate coupling: enabling native text prefill skipped the independent
   multimodal row-isolation contract. A red-circle request changed output
   compared with B1; the disabled-native control passed. The scheduler now
   applies multimodal isolation independently, retaining B1 prefill boundaries
   while allowing native text batches and subsequent grouped decode/verify.
   A regression checks both mixed-image isolation and retained native text
   batching. The original failure remains in `bundle-http-first-failure/`.
2. **DFlash2 performance remains incomplete:** default M5 grouped draft windows
   cost more than ordinary native decoding on the 512 MiB/short-output fixture.
   Existing solo-request cost adaptation does not fully govern these grouped
   windows. A group-level SSD-aware cost/fallback policy remains future work;
   no general DFlash2 speedup or performance acceptance is claimed here.
3. **No physical M1 Pro 32 GiB, 64 GiB, other M5 or M1–M4 GPU qualification.**
   Header-only 24 GiB budget simulations with B1, no draft, 4 GiB expert cache
   reserve about 14.65 GiB at 4K and 15.40 GiB at 8K, including deferred vision.
   These demonstrate planner accounting only; live Metal limits and competing
   allocations can lower the usable budget. Default B2/32K is rejected at that
   budget. Start M1 Pro qualification with ordinary B1/4K, then expand only
   after on-device measurement. Native auto stays restricted to M5 Max.
4. **No controller-cold disk throughput, thermal endurance, real low-memory
   pressure/swap recovery, or long-duration SSD wear qualification.** Both I/O
   policies pass correctness, but forcing a small cache on this 128 GiB host
   cannot substitute for those hardware measurements. Read/prefetch benefits
   were not isolated in a statistically controlled A/B study.
5. **Existing diagnostics persist:** five CoreAnalytics “Context leak” messages
   appeared in the full Rust run. They are not newly declared fixed; the
   separately investigated MLX thread-identity issue remains deferred.
6. **One transient Swift test failure:**
   `plannedStopInterruptsDefaultReadinessWithoutMarkingFailure` once received
   `NSURLError -1004`; its isolated rerun and two full 574-test runs passed.
   The root cause is not established. No readiness/lifecycle fix is claimed.
7. **Test harness setup failure after bundle rebuilding:** the clean packaging
   rebuild produces MLX with a relative Metal-library lookup. The subsequently
   rebuilt audio test initially trapped during Metal device initialization
   because its colocated `mlx.metallib` had been removed with the build tree.
   Copying the exact newly built library beside the test binaries made the
   same audio test pass. The failure and recheck logs are retained; the App
   helper uses its explicit bundled library path. A new scheduler regression
   also initially asserted every cache allocation was B1; this overlooked
   the final B2 cache receiving isolated rows. The assertion now checks the
   intended final-B2 plus two temporary-B1 allocations.
8. Phase counters are independent atomics and observational snapshots, not a
   transactionally consistent memory-admission ledger. Protected experts can
   still be evicted when total working sets exceed the bounded cache. KV and
   prefix persistence are not provided by these stages.

Detailed manifests and original logs are retained locally under
`.build/flash-paging-optimization/` (not committed). Source edits remain on the
feature branch; qualification does not authorize integration or publication.
