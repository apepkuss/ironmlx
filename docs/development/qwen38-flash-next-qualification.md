# Qwen3.8 Flash-Next implementation and qualification

This branch follows the migration-first order: MLX pin and compatibility,
existing-model regression, then the Flash text graph, bounded storage,
DFlash2 transactions, request scheduling, hardware policies and multimodal work.
An implementation stage and its qualification are separate gates.

## Fixed identities

- Current MLX build: fork `65383d68c984ce4ef49f4b600820f3b9064c5a82`
  (0.32.5), integrating the main-thread identity fix.
- Frozen numeric-reference / upstream baseline MLX main:
  `a8f66b5901791a38ff527f249e1f1a84f46c03b6` (0.32.5).
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
global TurboQuant setting. Legacy Active KV swapping remains disabled. Stage 7
adds a versioned complete Flash prefix format for ordinary and DFlash2 reuse
(see the dedicated section below); legacy KV-only prefixes are never accepted.
The default App context remains 32K; explicit
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
   windows. The follow-up below adds a group cost policy, with a remaining
   performance gap; no general DFlash2 speedup is claimed.
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
   separately investigated MLX thread-identity issue was deferred at this gate;
   the independent follow-up below does not change the production MLX pin.
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


## SSD follow-up: measured group policy and isolated MLX investigation

This follow-up skips CI inspection and physical M1 Pro 32 GiB testing, as
requested. All measurements still use the same M5 Max / 128 GiB host. Stages
7–8 (persistent Flash prefixes and active KV/request-state offload) remain
out of scope. The production MLX pin remains main `a8f66b590`; an independent
MLX fix is tested separately and is not silently integrated into the App.

### Group cost and reference partition correctness

Native paged Flash DFlash2 actors now learn ordinary versus speculative full
window microseconds per emitted token, separately by width, context bucket and
draft budget. Two ordinary windows establish the baseline, followed by two
calibration draft windows when the remaining output horizon can amortize them.
Drafting must improve the measured cost by at least 3%; otherwise ordinary
windows continue, with bounded two-window recovery probes. Exploration credit
is capped; short requests cannot accumulate an unlimited later probe burst.
The policy periodically refreshes its ordinary baseline and resets a regime
after 60 seconds without observations. Cache construction is measured
separately and excluded from the steady cost comparison.

`IRONMLX_FLASH_DFLASH_GROUP_POLICY=auto|off` is a process diagnostic, not a
new Dashboard setting. Explicit existing fixed/ragged/probe/budget overrides
keep their existing behavior. Other target models keep their existing policy.
Unqualified speculative widths fall back to a complete native ordinary group,
including B3; this does not qualify speculative B3 or draft tensor batching.
Health diagnostics expose enabled, trained/ordinary/probe/reset and shape
fallback window counters under DFlash2 ragged-linear state.

The longer HTTP gate found a latent numerical partition defect missed by the
previous 8-token fixture: a 32-token native ordinary B4 answer changed from
“精准、高效” to “高效、精准”. Independent B1 prefill uses a prefix plus a final
single token; the native complete-prompt path previously used the entire
prompt at once. Flash now retains the B1 prefix/final-token partition. Chunked
or configured-prefix-cache prompts use conservative row-isolated initial
prefill, preserving their existing per-request partition. Images retain their
independent isolation contract. Subsequent native decode/target verification
and the direct native paged MoE B2/B4 implementation remain enabled. This means
not every server prefill runs as a native group.

After the fix, two rounds per configuration passed independent B1/B2/B4
messages, completion counts and finish reasons, ordinary-versus-DFlash2 exact
outputs, SSE cancellation and immediate capacity reuse. A separate unequal
4/20/48/64-token fixture passed the transition from B4 through ordinary B3,
restored B2 speculative verification and B1; outputs matched independent B1.
The original failure and intermediate controls are retained locally.

| 32-token B4 / 512 MiB experts | End-to-end observations |
| --- | ---: |
| Ordinary native | 44.71–44.79 s |
| DFlash2 group policy off | 61.57–61.59 s |
| DFlash2 measured auto, first run | 52.05–52.57 s |
| DFlash2 measured auto, repeated run | 52.90–53.14 s |

These are small sequential fixed-fixture observations, not broad performance
acceptance. Auto improves this DFlash2 case but remains slower than ordinary.
Admitting a complete ordinary B3 cohort did not prove a further material
throughput gain over the earlier measured-cost implementation. Sequential
DFlash2 request prefill and expert/weight movement still contribute; neither
all causes nor a general optimal policy have been established.

### Storage controls and limits of the measurements

The existing default segmented cache can be compared with a plain LRU control
using `IRONMLX_FLASH_EXPERT_CACHE_POLICY=segmented|lru`. Bounded reader workers
can be compared with `IRONMLX_FLASH_IO_WORKERS=1|2` (default two). Invalid worker
configuration or thread creation failure returns an I/O error rather than
panicking during pool initialization. These are diagnostic controls.

An opt-in actual-model regression (`flash_paging_ab_real`) restores the same
request snapshot, replays the single next token with ID 701 twice to promote
its decode experts, scans 2048 other tokens, then restores and replays that
same next token. This is a fixed hot expert set, not a 701-token trajectory. All four 4 GiB-cache ABBA runs matched
replay logits exactly. Segmented retention kept all 480 replay expert requests
hot (zero misses/read bytes), while plain LRU had 49 hits and 431 misses,
requesting 1,324,032,000 expert bytes. Replay took about 0.10–0.11 s versus
0.51–0.55 s. This demonstrates scan resistance for this bounded working set;
it does not guarantee protected entries survive an oversized working set.

All four 2048-token scans used two readers and took 15.90–17.76 s; their
earlier/later timing difference cannot be attributed to reader count. Separate
512 MiB / 63-token controls passed cached/no-cache, serial/parallel and
prefetch-disabled/auto correctness. That scan took 5.80 s with one reader and
5.33 s with two (prefetch off), 5.31 s with prefetch auto, and 5.28 s with cached
I/O. These are single observations; the prefetch difference is within noise.
Two readers helped this fixture, but no controller-cold disk throughput claim
is supported.

In the 512 MiB I/O controls, physical device counters increased only about
77–96 MB while application reads were tens of GB, including `F_NOCACHE` runs.
The data remained
largely served from system caches. `F_NOCACHE` and a small application cache do
not prove cold SSD I/O. No global page-cache purge was performed. Application
hit/read/pack counters, process RSS and MLX allocator peaks are distinct; none
of these results proves a 32 GiB Mac can run the same workload.

### Lifecycle and bounded additional memory load

Ordinary and DFlash2 serving passed independent B1/B4 output comparison, 65 seconds of
idle without target probes, first-request recovery, a held additional 16 GiB
anonymous allocation, cold image materialization, live SSE cancellation and
capacity release, B4 recovery, allocation release, and image/text rechecks.
The allocation stayed resident through the pressure/cancellation requests.
The first idle-recovery request took 11.49 s for ordinary and 10.18 s for
DFlash2. DFlash2 recorded a cost-regime idle reset. These lifecycle timings
are not controlled throughput benchmarks; a separate CPU-only candidate build
overlapped the initial ordinary warm fixture.

System pressure stayed normal. Existing host swap usage was approximately
14,885 MiB before testing and did not increase during either route. Thus this
qualifies bounded extra memory occupancy, not newly induced OS pressure,
swap-thrashing recovery or operation on a physical 32 GiB machine. Process RSS
increased after first image materialization; deferred vision weights remain
loaded for subsequent image requests. Allocator/process/machine counters are
retained separately in the lifecycle evidence.

### Validation evidence

The initial follow-up pinned-MLX workspace Release Rust suite passed **2,521 tests**, with
**302 ignored** across **210 targets**; ignored opt-in tests are not passes.
CI-pinned nightly Clippy with warnings denied and both formatting versions
passed. Swift Release passed **574 tests** and Dashboard Node passed **4**.
Five CoreAnalytics “Context leak” diagnostics still appeared with the pinned
MLX. The independently linked candidate below is evaluated separately.

A workflow error was caught before the special lifecycle gates: a workspace
`cargo test --all-features` rebuild overwrote the CLI artifact with a version
linked to test initialization, which requires `MLX_DIR`. A sanitized production
launch therefore aborted before serving. Rebuilding default-feature production
helpers after the test run resolved it; `ironmlx --version` also passed with
`MLX_DIR` unset. This was test artifact reuse, not a fix to hide the test
initializer. The original failure log is retained.

Detailed manifests, failed controls and passing reruns are retained locally
under `.build/flash-followup/` (not committed). No CI result, real M1/32 GiB,
64 GiB, controller-cold SSD, thermal endurance, critical pressure/swap or
long-duration wear qualification is claimed by this follow-up.


### Independent MLX main-thread identity fix

The candidate is isolated on MLX branch `fix/main-thread-identity`, based on
`a8f66b590`, with no changes to the production IronMLX pin or installed baseline
library. Darwin now asks `pthread_main_np()` for actual main-thread identity;
Linux uses kernel TID versus process PID. Other platforms retain the existing
fallback. Only macOS/Apple GPU execution is physically qualified here; Linux,
CUDA, Windows and other fallback-platform behavior are not validated.

The original implementation remembered the first calling C++ thread ID. On
this host, a worker-first probe failed actual-main/256-worker identity checking.
A worker-created shared-stream lazy GPU graph also failed after worker
`clear_streams()` and exit: `There is no Stream(gpu, 0) in current thread.`
The fixed library passes both cases. CPU/GPU pending asynchronous outputs from
64 sequential producer threads remain usable after their producers exit, and
CPU worker exception propagation passes. The separate six-case CTest gate
passes with the candidate.

LLDB sampled the same OS `kIOGPUQueueCount` symbol in both statically linked
probes, using 350 sequential workers and main-first/reused-worker controls:

| Initialization/lifetime | Original final live queues | Fixed final live queues |
| --- | ---: | ---: |
| Worker first, 350 workers exit | 350 | 0 |
| Main first, 350 workers exit | 1 | 1 |
| One worker reused 350 times, then exits | 1 | 0 |

The original worker-first process produced six CoreAnalytics “Context leak”
messages; the corresponding fixed probe produced zero. Both processes exited
successfully. Final RSS was about 120 MB versus 84 MB, a process observation
rather than a byte-accurate queue allocation ledger. Global stream registry
entries remain (350/351 for sequential-worker controls); the fix retires
thread-owned GPU queues, not the global stream index metadata or all CPU
scheduler resources. No general registry/allocator leak elimination is claimed.

The isolated candidate uses the same base and private NAX/QMM metallib hashes
as the production baseline; only its host MLX archive differs. Production App
qualification continues to use the unmodified pinned host archive.


The official C++ suite with the candidate and default GPU reported **313 of
319 cases passing**, with **six failing cases / 17 assertions**. Relinking the
exact same test objects against the untouched baseline reproduced the same
six failures and the same 17 assertion locations. The cases are QR, inversion,
Cholesky, pseudo-inverse, `eigh` and LU; all combine CPU linear algebra with
default-device checks. The isolated QR case fails in default GPU mode for both
libraries and passes with `DEVICE=cpu`. The full candidate suite with
`DEVICE=cpu` passed **319/319 cases and 5,854 assertions**. The broader six-case
GPU-default issue is therefore a reproduced baseline limitation in this
environment, not a newly established thread-fix regression. Its underlying
cause remains undetermined; no unrelated linear algebra patch or relaxed test
tolerance was added. The GPU-default official suite is not declared green.


The complete IronMLX Release workspace suite linked to the isolated candidate
passed **2,521 tests, 0 failed, 302 ignored across 210 targets**, matching the
baseline pass/ignore totals. It produced **zero** CoreAnalytics “Context leak”
diagnostics versus **five** with the pinned baseline in the same suite. This
qualifies these scenarios on this macOS host, not every possible framework
context lifetime. The production App still uses the pinned baseline until a
separate integration decision; its five-diagnostic baseline is not declared
fixed by testing another archive.


The opt-in NAX/QMM gate passed six actual tests across five targets: Flash MoE
QMM/gather reference shapes, real 27B gate/up/down QMM byte equality, D256
prefill, D256 split-K decode and sliding-window attention. The candidate also
passed actual Flash native B2/B4 ragged prefill/decode, DFlash2 verification and
transactional commit/rollback, real vision/non-square reference, resident
reference, text/image generation, and native-disabled/no-cache/prefetch-disabled
controls. These used actual local snapshots/oracles, not header-only simulation.

Dense 27B ANE ordinary and DFlash2 target execution each completed **64 jobs**,
with **zero surface bytes after model unload**. Ordinary candidate logits
matched the earlier pinned-library reference exactly; ordinary and DFlash2
candidate logits also matched each other exactly for this fixed 2048-token
teacher-forced prompt and four continuation steps. This does not qualify ANE
for Flash or new machines; Flash ANE remains removed.


### Additional existing-model service issue and final source recheck

The first real 35B MoE ordinary B4 HTTP fixture failed on the bundled
production-baseline helper, before a candidate-library comparison: prefill
reported incompatible attention shapes `(1,1,21,21)` and `(1,16,21,19)`, returning
HTTP 500. The original log and binary are retained in
`current-models-first-failure/` and `binaries/pre-facade-ironmlx`.

Static inspection found that the Qwen3.6 MoE facade did not forward the inner
Qwen3.5 MoE model's `requires_split_batched_prefill_for_token_parity` contract.
The facade now delegates that contract; its existing model-surface regression
asserts the requirement. This restores the existing B1 prefill partition
policy rather than changing MLX math or disabling subsequent grouped decode.
After this final source change, the entire model-library Release unit suite
passed again with both host archives: **693 passed, 40 ignored each**. The
2,521-test workspace runs above precede this last facade-only adjustment;
those complete runs are not relabeled as reruns of later source.

Both format checks and the exact CI-scope nightly Clippy command passed after
the adjustment. An extra, broader `--all-targets` Clippy attempt failed on
unchanged legacy `mlx/tests` warnings (needless generic borrows and approximate
constants). The CI command does not include that extra target scope. These
warnings were retained as a separate non-green exploratory check; no unrelated
test cleanup or warning suppression was added.


The facade adjustment alone was insufficient: it fixed initial prefill, then
ragged decode exposed another baseline defect. Batch-stable attention split the
cache into B1 rows while retaining the batch-wide mask key width. A shorter row
therefore saw `(1,1,1,22)` mask keys against `(1,16,1,20)` attention scores. The
second failure is retained in `current-models-facade-only-failure/`, with its
helper binary in `binaries/facade-only-ironmlx`.

`GatedAttention` row isolation now clips only the mask's trailing key dimension
to that row's actual post-append KV length, retaining broadcast masks. A new
nonzero-weight/nonzero-history B2/B4 regression compares unequal row histories
with independent B1 computation exactly, including resulting cache offsets.
Before the fix this regression reproduced `(1,1,1,6)` versus `(1,4,1,4)` and
failed; after it the model-library suite passed **694 tests, 40 ignored** with
each MLX archive. The bundled pinned-library helper then passed the previously
failing real 35B MoE ordinary B4 fixture and its DFlash2 counterpart, including
independent B1 outputs, completion counts, finish reasons and cancellation /
capacity reuse. No claim that the first facade patch alone fixed the service
error is made.


The final real-service matrix passed all **eight** pinned/candidate × Dense
27B/MoE 35B × ordinary/DFlash2 cases. Each waited 20 seconds after model loading
without target work before independent B1 requests, then B4, SSE cancellation
and immediate capacity reuse. The pinned cases executed the actual signed
Release App helper. Messages, completion token counts and finish reasons
matched independent B1 for each case, matched the corresponding pinned case
for the candidate, and matched across ordinary/DFlash2 for both models. This
qualifies these text fixtures; MoE 35B image and persistent-prefix reuse are not
newly qualified by this matrix.


The focused candidate Flash HTTP gate also passed ordinary and DFlash2 serving
with a 512 MiB expert cache: 20 seconds idle after loading, independent B1 and
concurrent B2, exact messages/counts/finish reasons, exact cross-route output,
and healthy idle schedulers. This covers real runtime transfer of the loaded
model to inference threads in addition to standalone graph-lifetime probes.

With both MoE service fixes included, the final **pinned-library** Release
workspace rerun passed **2,522 tests, 0 failed, 302 ignored across 210 targets**.
The additional passing test is the unequal-history row-mask regression. Five
CoreAnalytics diagnostics persist with this pinned library. The exact local
CI-scope nightly Clippy command passed again for this final source.


The final **candidate-library** workspace rerun on the same final source also
passed **2,522 tests, 0 failed, 302 ignored across 210 targets**, with **zero**
CoreAnalytics diagnostics. The final pinned run retained five. Production
helpers were rebuilt with default features after each full test run; the
sanitized production entry point, not a test-support-linked artifact, is used
for App restoration. The final local Clippy check uses exactly the repository
CI scope; no remote CI status was queried.

The Release bundle reuses the unchanged previously built Release Swift shell
and the exact pinned base/private metallibs, with freshly built production Rust
helpers, current feature-branch HEAD plus dirty-state metadata, and ad-hoc
signatures. Static bundle, resource parity and model-distribution-boundary
verification passed. The one-shot packaging script was not run against the
dirty independent MLX fix checkout: that script requires a clean pinned source.
The pinned installed archive was preserved unchanged, and the candidate was
built/linked separately. This is local build/qualification, not integration,
commit/push, signing/notarization for publication, or a new public release.


At the end of qualification, the updated Release App was started from this
worktree's `dist/IronMLX.app`. Port 9068 reported healthy with no loaded model
and no active/queued requests; the listening helper path matched that bundle.
The App metadata retained MLX `a8f66b590`, feature HEAD `462372701` with a dirty
source tree, version 0.2.0 / build 9. The three recorded model/App configuration
file hashes matched their pre-test values. The baseline `libmlx.a` hash also
matched its original value. New IronMLX and independent MLX source changes
remain uncommitted and unpushed.


## MLX thread-identity fix integration (2026-11-03)

The approved integration commits the four validated MLX source/test files on
`fix/main-thread-identity` as `65383d68c984ce4ef49f4b600820f3b9064c5a82`.
The MLX working tree is clean. IronMLX now pins this commit in the release
configuration and native dependency source verification. Regenerated third-
party inventory/notices and CycloneDX SBOM carry the new fork revision; the
upstream baseline remains `a8f66b590`. Frozen reference generators/oracle
identities retain their original baseline, rather than rewriting expected
numerical evidence to match the newly integrated library.

The full `scripts/build-app-bundle.sh` pipeline completed from clean build
outputs, using the clean commit-pinned MLX checkout: fresh MLX/AOT Metal/NAX,
production Rust helpers, Swift Release shell, bundle assembly/signatures,
third-party reproducibility, SBOM, static bundle and model-distribution gates.
This integration did not substitute the earlier manually repackaged bundle.
App metadata records MLX `65383d68`, IronMLX feature HEAD `462372701` plus dirty
state, version 0.2.0 / build 9.

The fresh archive SHA differs from the earlier candidate archive, but all
**128 object members**, in order, match byte-for-byte; the archive container
metadata is not an implementation difference. The complete AOT Metal library
also matches byte-for-byte. Source hashes of the four committed MLX files
matched the previously qualified candidate before committing. Six freshly
linked identity/error/CPU-GPU graph/pending-output probes passed; the fresh
350-worker LLDB run ended with **zero live GPU queues and zero Context leak
diagnostics**. Global stream index metadata is still retained as documented
above.

The MLX fix was initially qualified locally. Following approval on 2026-11-04
JST, `fix/main-thread-identity` was pushed to origin at `65383d68`, and the
remote branch head was verified before committing the IronMLX dependency
update. No merge, tag or public release was performed, and no remote CI status
was inspected. Prior
limitations (physical 32/64 GiB Macs, controller-cold SSD, critical memory
pressure/swap, other platforms, the six reproduced official GPU-default
linear algebra failures, and deferred Flash prefix/KV persistence) remain.


Integration completed on **2026-11-04 JST** (2026-11-03 UTC). The freshly built,
signed bundle helper passed six real-service cases: Dense 27B, MoE 35B and
Flash, each ordinary and DFlash2, with independent B1/B4 output/count/finish
checks, 20-second post-load idle and live cancellation/capacity reuse. The
27B/35B text results matched the previously qualified candidate exactly. Flash
used a forced 512 MiB expert cache and mixed text/single-image/multi-image
requests; its two routes matched exactly.

Freshly linked NAX/QMM tests passed **six tests across five targets**, including
real 27B QMM weight data. Fresh Dense 27B ANE ordinary and DFlash2 target runs
each completed **64 actual jobs**, returned to **zero live surface bytes after
unload**, and matched the qualified candidate logits exactly. These are the
post-integration component/runtime gates; the previous two 2,522-test workspace
runs remain prior same-implementation evidence, not newly claimed executions
of this integration step. The MLX object-member and Metal-library byte equality
and unchanged IronMLX numerical source support that evidence transfer.

The exact newly rebuilt App was launched and its own helper on port 9068
reported healthy, idle and no loaded model. Metadata confirms MLX `65383d68`;
the recorded user model/App configuration hashes are unchanged. The old App
and baseline MLX install are retained locally under
`.build/mlx-identity-integration/previous-IronMLX.app` and
`previous-mlx-install/`. Detailed new manifests, fresh artifact hashes, model
comparisons, thread/queue samples and startup identity are in
`.build/mlx-identity-integration/results.json`. The qualification App was built
before the feature changes were committed; its recorded dirty-source metadata
is retained as the tested artifact identity.

## Stage 7: durable Flash prefix reuse (2026-11-04)

This stage adds reusable **completed prefill boundaries**, not active request
checkpoints. It does not offload a live request's KV, recurrent state, sampler,
PRNG, SSE position or draft verification transaction. Stage 8 remains deferred.

### State and compatibility contract

The version-1 Flash payload includes every row's GatedDelta convolution and
FP32 recurrent state, PLE n-gram history (including EOS-reset history), PLE
convolution and history position, sparse-attention K/V, raw indexer keys,
MRoPE positions and completed compressed index blocks. Optional block summaries
retain their presence/count; rollback-invalidated summaries are not silently
invented on restore. Destination rows keep their own model instance identity;
ordinary rollback snapshots still reject foreign model/layer instances.

Keys bind token IDs, image-covered-boundary fingerprints, local checkpoint
identity, quantization, execution contract, tensor shapes/dtypes, prefix format,
IronMLX version and the pinned release backend configuration. Checkpoint identity
hashes config/index contents and checkpoint canonical paths, file identity,
length and nanosecond change/modification times. It deliberately does not hash
112 GB of weights on every load: copying/moving/replacing checkpoints causes a
conservative miss. The numerical reference remains MLX `a8f66b590`; the pinned
thread-identity-fix build is `65383d68c984ce4ef49f4b600820f3b9064c5a82`.
Flash keys have no dependence on the Dense ANE setting.

The canonical per-layer vector starts with `Int32[6]` metadata:
`[version=1, layer_index, kind, cached_len, ple_flags, block_count]`.
`kind=0` is linear and `kind=1` is sparse. PLE flags are bit 0 (PLE present)
and bit 1 (convolution present). PLE history position must equal the common
layer offset on export and is restored from that validated offset; EOS resets
the rolling token history, not the absolute position.

| Layer | Remaining tensor order | Canonical layout |
| --- | --- | --- |
| Linear | GDN conv, GDN recurrent, PLE token history, PLE conv | BF16 `[1,gdn_kernel-1,conv_dim]`; FP32 `[1,Hv,Dv,Dk]`; UInt32 `[ngram-1]`; BF16 `[1,(ple_kernel-1)*ngram,hidden*hc]` |
| Sparse | K, V, raw index keys, positions, compressed blocks | BF16 `[1,Hkv,N,D]` twice; BF16 `[1,N,Di]`; Int32 `[3,1,N]`; BF16 `[1,1,max(N/ratio,1),Di]` |

Layers without PLE use one-element zero placeholders for its two tensors.
Absent compressed blocks use zero padding and `block_count=0`; present blocks
use the validated count, preserving a shorter completed-block cache if needed.
The outer store records exact tensor dtypes/shapes and names Flash tensors
`main_{layer:04}_flash_{index}`. DFlash2 adds a separate final auxiliary layer
with context `[1,min(N,window-1),captures*hidden]` and final hidden
`[1,1,hidden]`. Legacy stores that do not understand these layer kinds miss
rather than interpreting the arrays as ordinary KV.

All Flash layer layouts and metadata are validated before a destination row is
changed. DFlash2 artifacts additionally store the unprojected target-layer
hidden-state tail and final target hidden state. Draft checkpoint identity,
precision, capture layers, sliding window and execution policy are part of the
DFlash2 key; an ordinary prefix alone is never used as a DFlash2 artifact.

### Storage and runtime behavior

Ordinary inference uses the existing prefix-cache scheduling and optional hot
LRU. DFlash2 retains its own artifact handling, with durable storage enabled
only for Flash. App launch qualification forwards SSD cache settings for Flash
DFlash2, while other DFlash2 targets keep their existing memory-only behavior.
The existing public `build_dflash2_engine_with_options` API is preserved; the
new `build_dflash2_engine_with_prefix_storage` entry point opts into storage.

A prefix is staged in a temporary directory, accompanied by SHA-256 payload
integrity data, flushed and committed by directory rename. Writers use an
exclusive per-root process lock; abandoned temporary transactions are reaped
under that lock. Readers ignore temporary directories. Disk capacity includes
serialized payload and metadata bytes; successful loads update advisory disk
LRU recency. The quota bounds committed logical file bytes; staging a transaction
additionally needs space for one record. Invalid/unavailable cache records fall back to computation, and
bad records can be repaired by a subsequent successful prefill/save.

Flash writes synchronously at a prefill boundary. The existing generic async
writer is limited to 512 MiB; placing long Flash snapshots there would skip
saves or require a larger retained-state budget. Synchronous writes avoid that
background retention, but add first-request I/O latency and form a safe boundary
that cancellation must wait to finish. They do not make prefill writes free or
provide streaming/incremental state persistence.

CLI example (explicit isolated cache quota):

```sh
ironmlx serve --model /path/to/Flash-Next --model-id flash \
  --paged-prefix-cache-dir /path/to/prefix-cache \
  --ssd-prefix-cache-max-gb 16
# Add the following for the qualified Flash draft:
# --dflash2-model-dir /path/to/Flash-Next-DFlash2 --dflash2-draft-bits 0
```

App uses its existing cache enable/capacity settings. Active KV offload and
TurboQuant remain disabled for Flash. Prefix reuse does not increase the
maximum live context or request-memory capacity of a 32/64 GiB machine.

### Qualification evidence

See `.build/flash-prefix/` for local logs and machine-readable artifacts. The
committed opt-in test is `ironmlx-runtime/tests/flash_prefix_real.rs`; it spawns
separate writer and reader processes. It includes EOS history, native B1/B2/B4
restoration and eight continuation steps with full-vocabulary logits compared
against uninterrupted independent B1 continuations. The 4097-token fixture
crosses the compressed sparse-index threshold. This is stronger than merely
comparing generated text, but is still a bounded fixture, not universal quality
or performance qualification.

The local evidence is from Apple M5 Max / 128 GiB. Physical M1 Pro / 32 GiB,
64 GiB, controller-cold SSD, memory pressure/swap and long-duration wear are not
qualified by these tests. Application expert read bytes can include OS page
cache hits and must not be described as physical SSD throughput.

### Measured results and remaining limits

Actual model tests used the standalone Flash checkpoint and BF16 draft listed
above, on the M5 Max / 128 GiB host, with a forced 2 GiB expert cache and cached
application I/O. No OS page-cache purge was performed. HTTP model fixtures
use `/v1/chat/completions`; other protocols share the runtime but did not get
separate real-model durable-prefix runs here. Production integration is in
`serve` and App cache settings; `generate` gains no new persistence option.

| Gate | Observed result |
| --- | --- |
| Cross-process ordinary state | 4097 and 16385 tokens; full-vocabulary logits exact against uninterrupted B1, eight steps at native widths B1/B2/B4 |
| Larger-than-queue record | 16385-token payload 583,598,570 bytes (556.6 MiB), saved/restored successfully; restored process had zero prefill expert read bytes |
| Ordinary HTTP | B1, mixed B4, cancellation then B2 reuse, single/multiple images, process restart, appended chat turn and corruption/repair passed |
| DFlash2 HTTP | Same mixed request/cancellation/restart/corruption gates passed, with outputs matching ordinary HTTP fixtures |
| Draft sliding-window boundary | 2420-token prompt; persisted context `[1,2047,12800]`, restored after process restart with unchanged output |

For the 820-token HTTP prompt, the first cache-building request took 20.301 s
on the ordinary route and 18.937 s on DFlash2. The first request after restarting
with the disk record took 2.019 s and 1.352 s respectively. In-process repeats
were 1.554–1.572 s and 1.459–1.511 s. These are single development samples,
including response/decode and first-write costs, not repeated steady-state
benchmarks; kernel preparation, expert residency and OS caching are additional
conditions. They do not isolate a universal speedup factor or physical SSD
throughput. No claim is made for short-prompt net benefit in full-resident mode.

The 4097-token direct fixture recorded 66,109,440,000 application expert bytes
in its prefill phase. Its restored process recorded zero prefill expert bytes.
The direct fixture also performs independent decode continuations: decode I/O
must not be confused with prefill I/O or with total machine disk traffic.

Issues found and resolved during implementation/validation:

- Flash was excluded from the generic prefix payload; merely persisting KV
  would omit recurrent/PLE/sparse-index state. A separately validated complete
  codec now binds the durable state to the new model instance.
- The 512 MiB generic background queue could not cover longer Flash records.
  Flash uses synchronous boundary writes, with the latency limits stated above.
- App DFlash2 launch previously omitted SSD flags for every target. An explicit
  Flash-only qualification now forwards them; the shared engine cache-enabled
  flag also recognizes disk-only configurations.
- Existing tests counted a new writer lock file as a cache entry (17 instead of
  16). The assertion now counts record directories and still requires all 16
  immutable blocks to persist.
- An initial security-startup test was blocked by the already-running idle App's
  instance lock. Tests run with that verified App stopped, and the original App
  is restored afterwards; the conflict was not a model/security-policy failure.

Remaining limits are material: synchronous writes can delay both the writing
request and other work on its actor until the safe boundary finishes; the disk
quota additionally requires one staged record of free space; small-memory and
controller-cold tests remain outstanding. Flash DFlash2 retains request-private
ragged verification and its existing Tensor Batch qualification restrictions.
This stage does not open that separate Tensor Batch setting.

DFlash2 `prefix_cache_entries`, `prefix_cache_bytes` and
`prefix_cache_evictions` remain **hot-memory** artifact metrics; logical
hits/misses/saves and reused-token counters also cover Flash disk lookups.
They are not physical SSD operation or per-record disk-eviction counters.
Detailed disk source/latency is available in logs; per-model durable-prefix
I/O/eviction telemetry is not separately exposed in Dashboard. Existing expert
and PLE paging telemetry is unchanged.

The opt-in cross-process test can be reproduced with the current pinned MLX
installation and an isolated artifact directory:

```sh
export IRONMLX_FLASH_SNAPSHOT=/absolute/path/to/Flash-Next
export IRONMLX_FLASH_PREFIX_ARTIFACTS=/absolute/path/to/test-output
export IRONMLX_FLASH_PREFIX_LEN=16385
cargo test --release -p ironmlx-runtime --test flash_prefix_real \
  -- --ignored --exact flash_prefix_cross_process_exact_b1_b2_b4 --nocapture
```

The artifact directory retains potentially large prefix and reference files.
It is test output, separate from the App's configured production cache.

Final validation summary: the full Release workspace suite passed **2529 tests**
with **303 conditional skips** across 211 targets. After correcting the final
shared cache-enabled status flag, all library suites passed again (**1941
passed, 106 skipped**) and the final production Release backend passed both
ordinary and DFlash2 disk-only smoke runs. The final image check used two
64×64 images with identical prompt token IDs but different red/blue pixels:
records had matching token hashes and different image fingerprints; replies
were `Red` and `Blue`, and the repeated red request remained exact on both
routes. Swift tests passed **575 tests**; pinned-nightly Clippy with
`-D warnings`, both formatters and whitespace checks passed.

No `Context leak` diagnostic was found in these Rust/model-run logs. Historical
unrelated MLX C++ GPU-default linear-algebra failures recorded above were not
reinvestigated or resolved by Stage 7. MLX native source is unchanged.

The original App bundle was restored healthy after isolated validation, with
its bundle files and user preferences unchanged. It has **not** been repackaged
with Stage 7; the new implementation was tested through the independently
built Release backend and App source/launch tests. At that Stage 7 checkpoint,
changes were still uncommitted; the subsequent commit is recorded below.

## Adaptive completed-prefix saving (2026-11-04)

The original Stage 7 implementation above was committed and pushed as
`1cbeb7554d535ab3fc4c2493b8ab9bec69571175` before this follow-up started.
Its description of exclusively synchronous Flash writes is historical. This
follow-up adds opt-in `--prefix-save-mode auto`; `sync` remains the default and
fallback. The cache master switch, hot-cache budget, source/contract identity,
opaque Flash format, integrity checks and committed disk quota remain in force.
This does **not** implement Stage 8 or enable active Flash KV/request offload.

### Admission and ownership

| Physical RAM | Snapshot peak cap, process-wide | Capturing/queued/active jobs | Write chunk |
| --- | --- | --- | --- |
| At most 32 GiB | 512 MiB | 1 | 8 MiB |
| Above 32 through 64 GiB | 2 GiB | 1 | 16 MiB |
| Above 64 GiB | `min(RAM / 32, 8 GiB)` | 2 | 32 MiB |

For example, the 128 GiB test host gets a 4 GiB cap. Caps are not preallocations.
For payload `S` and token count `T`, admission reserves a conservative peak of
`2*S + chunk + 12*T + 1 MiB`, covering owned bytes, compact materialization,
metadata and bounded I/O. It must fit both the remaining cap and one quarter
of safe unreserved governor headroom. Existing generic prefix work is accounted
for and the generic queue cannot enter concurrently with owned Flash snapshots.
This is not a promise that any particular model/context/concurrency fits 32 GiB.

The foreground thread explicitly materializes row-contiguous tensors and copies
byte-exact CPU storage, one tensor at a time. The worker owns byte vectors and a
governor lease, with no model, MLX Array, borrowed request state or lazy graph.
It still consults the thread-safe global memory governor. A pending snapshot can
be restored under another budget check; it is counted separately from SSD reuse.
Hot-memory retention is independent of whether optional persistence was skipped.

Writes stream safetensors and SHA-256 with bounded buffers, preserve the existing
transaction/flush/rename protocol and recheck cancellation before publication.
The macOS writer uses utility QoS and `F_NOCACHE` for the prefix output file;
model expert/PLE read policy is unchanged. A single process-wide worker is a
conservative serialization bound even for multiple volumes; parallel writers on
different volumes are not implemented. Switching volumes clears the write-rate
estimate instead of treating one SSD's observation as another SSD's throughput.

Soft pressure or unavailable telemetry stops new captures. Hard/emergency
pressure cancels optional work at safe checkpoints. Recovery holds admission for
five seconds after observed pressure. Estimated queue drain above two seconds
limits admission to one job; above five seconds stops admission. Copy/write wall
rates and completed workload-matched requests guide benefit decisions. Repeated
matched decode slowdowns above ten percent enable cooperative I/O throttling;
this requires at least three baseline observations and two slow windows, and is
a heuristic rather than a guarantee of causal attribution or a performance SLO.

Model unloading cancels captures and active writes and immediately drops queued
snapshots. Foreground reservation failure can preempt optional saves and retry
after a bounded wait. A native MLX evaluation or OS flush cannot be interrupted
mid-call; cancellation/reclamation is therefore not instantaneous. An already
complete, independent snapshot may finish after its originating request is
cancelled. It never retains that request's active generation state.

Adaptive ordinary serving prefers the reusable turn boundary and coalesces
adjacent ends; DFlash2 prefers the full prompt with its required draft context.
It does not repartition numerical prefill chunks. Short/low-benefit prefixes or
requests failing any budget are skipped rather than put in an unbounded queue.
Sync uses bounded tensor streaming as well, budgets its buffer/materialization
and waits for the durable transaction; its resource cost is not zero.

### Configuration and observability

The App forwards `prefix_save_mode` for ordinary serving and qualified Flash
DFlash2. Advanced settings expose Sync/Adaptive under the existing cache master
switch, with English, Simplified Chinese, Traditional Chinese, Japanese and
Korean text. Native `/healthz.prefix_persistence` and the Dashboard health bridge
expose process-wide status, reason, snapshot budgets, queued/committed/skipped/
cancelled/failed counters and distinct RAM/pending-snapshot/SSD lookup counts.
Accepted/queued is explicitly **not** a durable commit. These lookup counts are
not physical I/O operations or a per-model SSD hit rate; reused-token accounting
remains available separately. The worker's measured write time includes lock
wait and durable flush time, not merely physical device transfer.

### Validation record

Local artifacts are kept under `.build/prefix-adaptive/`. Performance runs use
explicit executable hashes and an isolated loopback service/cache directory;
user model files and App configuration are not test outputs. The first complete
workspace run passed 2,536 tests with 303 explicitly ignored across 211 targets.
Swift passed 576 tests, and Dashboard Node tests passed 84 tests. Final focused
checks and physical-model results are recorded below after qualification.

The CI-equivalent Clippy scope passed. An additional `--all-targets` attempt was
blocked by pre-existing diagnostics in unmodified MLX test files (`matmul`,
`async`, `paged_attention`, `tier2`, `quantization`, `nax_moe_regression`); these
are not being hidden as a successful all-target lint gate. No external MLX
native-source changes are included in this follow-up.

### Final adaptive qualification results

The physical host was **Apple M5 Max, 40 GPU cores, 128 GiB**, macOS 26.4.
The external MLX source remained on `fix/main-thread-identity` at
`65383d68c984ce4ef49f4b600820f3b9064c5a82`; it was not modified here.
The final production CLI SHA-256 is
`a64bd667148ed405d7a0d194c68d4019cf46a737ec19479e768451051d56ab54`.
The exhaustive HTTP fault/restart matrix used
`6d86daf62eb8ff61705c57a708b8bc34161a9251b91c48eca762dab24244aa57`;
the later changes were telemetry ownership and rejecting noncanonical BOOL
bytes in the general raw-array constructor. Flash snapshot tensor contents and
numerical paths were unchanged. The final CLI was separately used for the
existing-model HTTP regressions and final adaptive performance samples.

| Check | Actual result |
| --- | --- |
| Final Release workspace, all features, serial harness | **2,538 passed, 303 ignored, 0 failed**, 211 test targets |
| Swift App source tests | **576 passed** |
| Dashboard Node tests | **84 passed** |
| CI-equivalent pinned-nightly Clippy | Passed with `-D warnings` |
| Stable/pinned-nightly Rust formatters and whitespace | Passed |
| Array conversion, including final canonical BOOL guard | **17 passed** |
| Native window/prefill/decode D256 NAX, QMM-down and MoE NAX groups | All six recorded groups passed; QMM-down used a real 27B shard |
| Existing 27B and 35B-A3B HTTP regressions | Ordinary and DFlash2, all four cases passed; B1/B2/B4 and post-cancel outputs matched the prior qualified baseline |
| 27B ANE established paths | Ordinary and DFlash2 each completed 64 jobs; context lifetime test completed 192 jobs, surviving model remained bit-exact, surfaces returned to zero after both unloads |
| DFlash2 GPU auto profile plus ANE auto | 128 half-precision ANE jobs, passed |

The isolated cross-process model test used complete Flash prefixes of **4,097**
and **16,385** tokens. For each, B1/B2/B4 continuations matched the reference
**exactly across the full vocabulary for eight steps**, including independent
continuations and EOS inside the prompt. Payloads were 232,653,290 and
583,598,570 bytes. Capture plus waiting for background commit took 0.455 and
1.131 seconds; restoration took 0.386 and 1.013 seconds. These save times are
**not foreground-only blocking or TTFT measurements**. Tests forced a 64 GiB
planner/48 GiB wired limit and a 2 GiB expert cache on the 128 GiB host; this is
not physical 64 GiB hardware qualification.

The isolated HTTP fault/restart matrix passed on ordinary and DFlash2 routes:

- A separately held writer lock exposed a real pending snapshot. Reuse produced
  the same answer while committed count was still zero, with one pending hit.
  Pending jobs, frozen bytes and peak reservations returned to zero after drain.
- B2/B4 independent reuse, cancellation and subsequent requests passed. The
  DFlash2 prompt had 2,435 tokens, exceeding its 2,047-token retained draft tail.
- Same-token-count red/blue image requests remained distinct; replies were
  `Red`/`Blue`, and repeating red was exact. Ordinary and DFlash2 both passed.
- Restart hit SSD, not the pending map, and returned the exact same answer.
  Ordinary restored 834 of 835 prompt tokens and read 1,474,560,000 expert bytes
  for the remaining suffix; DFlash2 restored all 2,435 tokens with zero expert
  prefill reads. These are cache counters, not controller-cold SSD throughput.
- Bit corruption caused safe recomputation and repair. A real SIGKILL while a
  3,229,704-byte temporary payload existed exposed no committed entry; restart
  ignored/reaped the temporary transaction and recomputed successfully.
- The pre-adaptive helper read the new Auto file, and the new helper read the
  old Sync file, on both routes. State-format compatibility was preserved.
- Flash active KV offload remained disabled throughout.

Artifacts: `workspace-final.log`, `swift-final.log`, `dashboard-delivery-full.log`,
`clippy-delivery.log`, `native-final.json`, `array-checked.log`, `ane-final.json`,
`ane-auto-dflash.json`, `exact-auto-results.json`, `exact-auto-*/`,
`real/functional.json`, `functional-driver.log`, and `real/current-models.json`
under `.build/prefix-adaptive/`.

### Problems found and corrections

Implementation checks caught and corrected byte-offset handling for contiguous
views, large raw-array construction without a flat signed-32-bit byte dimension,
and noncanonical BOOL input. Raw F32/BF16/U32 bits, including NaNs, remain exact.
The final array conversion tests cover slices, strided rejection, empty arrays,
shape/length validation and ownership.

Resource/scheduling checks corrected a stalled writer with unknown initial
bandwidth being admitted indefinitely, stale process-memory sampling in library
use, and write-rate estimates being reused after a volume switch. Admission now
also observes oldest-job age, expires measured rates after 30 seconds, and
associates completed measurements with their originating volume. Cancellation
releases queued storage immediately; active work stops at bounded checkpoints.
Relative, not-yet-created cache roots resolve an existing ancestor for free-space
checks. Adjacent-prefix coalescing includes the execution contract and requires
an extant committed or owned pending entry. Worker status updates retain the
correct model/mode ownership.

One crash-test harness initially counted a hidden `.tmp` directory as a
published entry. Filtering transaction directories corrected the harness, and
the entire HTTP matrix was rerun successfully. An initial standalone instance
lock repeat omitted required `MLX_DIR` and aborted before the harness; the
properly configured repeat passed 100/100. These were validation setup errors,
not successful product checks at the time they failed.

### Remaining issues and limits

- **Latency stability is not fully qualified.** Preliminary adaptive samples
  included decode outliers around 21.5 seconds; the old synchronous baseline
  also exhibited DFlash2 TTFT/decode outliers. There is insufficient evidence to
  attribute all variation to the writer or certify a universal 5% latency/
  throughput bound. Auto remains opt-in; Sync remains the default fallback.
- One unmodified unique-temp backend instance-lock test failed a reacquire
  assertion in a parallel library run. The final full serial workspace run and
  100 configured serial repeats passed. The parallel failure's root cause is
  **not confirmed**; serial success does not rule out a concurrency issue.
- An extra diagnostic forcing the DFlash2 M5 GPU profile onto **ordinary** 27B
  execution reported zero ANE jobs and failed its assertion. The same forced
  combination failed with the pre-adaptive binary. Established ordinary ANE
  and DFlash2 ANE paths passed; the forced combination is not newly qualified.
- Additional all-target Clippy diagnostics and historical external MLX GPU
  default failures are not repaired by this work. Passing CI-equivalent checks
  must not be represented as those separate issues being resolved.
- No physical **M1 Pro 32 GiB**, 64 GiB, M1–M4 or other-chip qualification was
  performed. Tier/unit simulations and forced smaller planner limits do not
  replace hardware testing. Controller-cold SSD, actual extreme RAM/disk
  pressure, and multi-volume workloads have not been qualified.
- Hardware adaptation uses conservative RAM caps, live safe headroom and
  observed wall rates; it does not run a dedicated hardware bandwidth/SSD
  benchmark. Decode feedback is a delayed heuristic. It cannot guarantee zero
  interference or instant interruption of native evaluation/flush operations.
- One global writer serializes volumes. Oversized or low-benefit snapshots may
  be skipped, so not every first request creates a durable reusable prefix.
  A pending hit is not crash durability. This is completed-prefix persistence,
  **not** active KV tiering or request crash-resume (Stage 8 remains deferred).

### Measured foreground and background cost

The final CLI was run alone after compilation/regression work, using full expert
residency on the 128 GiB host, max sequences 4, 2,048-token prefill chunks,
BF16 state, ANE off, hot prefix retention disabled, and 4,036-token chat prompts.
Each route had a separate warm-up followed by five distinct first-miss prompts
and 16 generated tokens. The OS/model page cache was not forced cold. SSE TTFT
means first nonempty answer content with thinking disabled; decode wall time
runs from that content to completion. It is not a per-token GPU kernel timer.

| Final Auto path | Foreground capture mean (range) | Background commit mean (range) | TTFT median (range) | Decode wall median (range) |
| --- | --- | --- | --- | --- |
| Ordinary | **64.1 ms** (49.9–92.6) | **451.6 ms** (432.7–492.6) | **5.053 s** (4.932–5.272) | **0.499 s** (0.486–0.508) |
| DFlash2 | **32.3 ms** (28.5–41.6) | **538.3 ms** (529.1–556.3) | **4.665 s** (4.592–4.747) | **1.384 s** (1.279–1.634) |

Both routes committed all five measured snapshots with zero failures, and ended
with zero pending jobs/frozen bytes/reservations. Ordinary saved one reusable
4,035-token boundary instead of the old three 2,048/4,035/4,036 boundaries;
DFlash2 saved one full 4,036-token prompt instead of two boundaries. Measured
capture is the controlled persistence operation, not the entire foreground
request overhead. Background commit can overlap decoding and still compete for
memory/I/O resources. Different routes' decode wall times are not a comparative
claim about DFlash2 acceleration.

Earlier matched-configuration samples retained the pre-adaptive executable
`3a82dad1e6e7ebeb018e087a57f4ef7d038748165826f38be7b8db6416c0d5b0`
and implementation candidate
`303b61d902e9cfe866d6dc4e8998ed2daf87c85da84d5f888658858974721bcb`.
They are separate runs, not randomized paired trials:

| Route/mode | TTFT median (min–max), five requests | Decode wall median (min–max) |
| --- | --- | --- |
| Ordinary, old Sync | 5.986 s (5.863–6.051) | 1.552 s (1.540–1.653) |
| Ordinary, bounded Sync | 6.253 s (6.216–6.331) | 1.556 s (1.532–1.572) |
| Ordinary, early Auto | 5.940 s (5.350–6.166) | 0.848 s (0.522–21.665) |
| Ordinary, cache off | 4.802 s (4.665–5.387) | 1.478 s (1.429–1.556) |
| DFlash2, old Sync | 7.190 s (5.849–13.527) | 1.530 s (1.399–8.581) |
| DFlash2, bounded Sync | 5.683 s (5.601–5.764) | 1.471 s (1.390–1.563) |
| DFlash2, early Auto | 5.306 s (4.916–15.029) | 1.452 s (1.325–6.779) |
| DFlash2, cache off | 4.935 s (4.854–5.225) | 1.697 s (1.402–1.786) |

The bounded Sync fallback has a measurable cost: ordinary synchronous save
operations totaled a median **1.549 s/request**, versus **1.267 s** in the old
Sync run; DFlash2 bounded Sync totaled a median **1.029 s/request**. Bounded
materialization/buffering and uncached streaming trade predictable memory for
some write cost. This regression is retained explicitly, not presented as an
across-the-board speedup. These operation totals are not TTFT deltas.

The final Auto samples were stable over this small run, but do not explain away
the earlier long tails or qualify sustained load/P95. The first exploratory
2 GiB expert-cache timing run also overlapped compilation and is unsuitable for
clean latency attribution; it remains as raw evidence rather than a performance
claim. No universal throughput/TTFT improvement percentage is asserted.

Artifacts: `real/*-500-full-t16*.json`, matching server logs,
`performance-delivery-driver.log`, `delivery-binary.json`.

The original Release App was restored healthy on port 9068 after isolated CLI
validation. Both user preference files retained their original SHA-256 hashes,
and the original bundle's backend/benchmark/Metal payload hashes matched the
recorded bundle identity. The App was **not rebuilt with these adaptive changes**;
App source/UI changes were covered by Swift and Node checks, while physical
model tests used the isolated Release CLI. At that checkpoint adaptive changes
were still uncommitted and only Stage 7 commit `1cbeb7554` had been pushed;
the follow-up records below supersede that Git status.
No `Context leak` diagnostic was found in this turn's recorded Rust/model logs.

## Prefix persistence follow-up: latency investigation (2026-11-04)

The adaptive implementation was committed and pushed as `2480914b0` before
these follow-up changes. Stage 8 remains deferred. This section records a
latency investigation, **not a claim that the historical long tail is fixed**.

Added `scripts/qualify-flash-prefix-latency.py`, which retains all warmups and
measurement requests, records every nonempty SSE event, and optionally records
macOS `proc_pid_rusage` process counters. Missing terminal SSE events fail the
check. Small samples report median/min/max rather than a misleading P95.
Process I/O attribution is not controller-cold SSD throughput; process pageins
do not cover every possible GPU/driver stall. The qualifier does not purge
caches, inject memory pressure, or change server policy.

Runtime DEBUG `request latency breakdown` now distinguishes TTFT, maximum gap
between output events, and time between final output and request completion,
with route/concurrency and persistence overlap/pending status. An output event
may contain several speculative tokens; its gap is **not** per-token kernel
latency. Timing does not force additional graph evaluations or change decode.

### Isolated six-case comparison

On the same M5 Max 128 GiB host, the pre-follow-up Release executable
`a64bd667148ed405d7a0d194c68d4019cf46a737ec19479e768451051d56ab54`
ran ordinary/DFlash2 × cache off/Sync/Auto in separate processes, with no build
or other qualification job running concurrently. Full experts, b_max 4, chunk
2,048, ANE off, hot retention disabled, 8 GiB disk quota, 4,038–4,039 prompt
tokens, 16 output tokens. Each case retained two warmups and twelve measurements.
All 84 requests produced the same expected numeric continuation. This is a
single-host sequential comparison, not randomized hardware/SLO qualification.
The last ordinary Sync request also crossed the disk quota and exercised
eviction; the later dedicated save-cost comparison stays below the quota.

| Route/mode | TTFT median / max (s) | Decode wall median / max (s) | Largest measured content-event gap (s) |
| --- | --- | --- | --- |
| Ordinary off | 3.783 / 8.091 | 1.728 / 5.205 | **4.793** |
| Ordinary Sync | 6.304 / 6.486 | 1.555 / 1.609 | 1.135 |
| Ordinary Auto | 4.851 / 5.033 | 0.496 / 0.507 | 0.060 |
| DFlash2 off | 4.555 / 4.769 | 1.460 / 1.560 | 1.085 |
| DFlash2 Sync | 5.637 / 5.975 | 1.444 / 1.612 | 1.164 |
| DFlash2 Auto | 4.721 / 4.981 | 1.486 / 1.573 | 1.120 |

Ordinary off additionally had a **6.952 s** output-event gap in its first warmup.
Its warmed outlier (request index 4) had zero process pageins, zero process disk
writes, approximately 1.9 MiB attributed reads, and normal governor pressure.
The prefix writer was disabled. Therefore optional prefix writes are not a
necessary cause of this reproduced stall, and the counters do not support
attributing this particular event to process pageins. They do **not** identify
the ultimate GPU/driver/graph cause or exclude writer interference in other
workloads. The historical 21-second Auto decode stall was not reproduced here.
The issue remains open, and no universal latency-degradation bound is certified.

Artifacts: `.build/prefix-followup/latency-summary.json`, `latency-driver.log`,
`latency/*.json` and matching server logs. Model files and user configuration
were not modified.

### Profile-guided candidate and disposition

An additional 20-request ordinary-off diagnostic run used the new timing
instrumentation, executable SHA-256
`2a5ec44685cae4720f05592572a557cf77f0be8050fd5a2af0f1945594a79cd4`.
During its first warmup, a 2.088-second content gap triggered a one-second,
10-millisecond-interval macOS `sample` capture. The stack contained QSA
`selected.to_vec::<i32>()`, MLX graph evaluation/Metal encoding, and allocator
cache reclamation. This is a sampled call path, **not proof of the ultimate
stall cause**. The 20 subsequent measured requests had median decode wall
1.388 seconds and maximum event gap 1.181 seconds; completion tails in backend
DEBUG records were negligible. The profiler intervened in the warmup only.

A candidate kept sparse decode mask construction on the GPU using boolean
block flags, avoiding that host readback and keeping scratch linear in context.
Two component tests passed (boundaries, sentinels, duplicate blocks and a
262,145-position mask). Real 4,097/16,385-token prefix readers matched the old
reference **exactly across the full vocabulary for eight steps at B1/B2/B4**.
However, twenty ordinary-off requests produced median decode **1.396 seconds**
and maximum event gap **1.398 seconds**, versus the 1.388/1.181-second control.
That does not establish a latency improvement or resolve the historical tail.
The candidate was **withdrawn from the shipping source**, not presented as a
successful fix. Its patch, correctness tests and performance evidence remain
in `.build/prefix-followup/gpu-mask-candidate.patch`, `mask-*.log/json`, and
`mask-latency/`. No model arithmetic change is included in the latency commit.

The delivered diagnostic changes passed six runtime-usage tests and four
qualifier Python tests. The long-tail issue remains open; the practical result
is a reproducible first-miss comparison and better distinction between output
gaps and request teardown, without speculative changes to memory/GPU policy.

## Prefix persistence follow-up: synchronous save cost (2026-11-04)

The shipped change enables `sha2 0.10.9`'s existing `asm` feature **on AArch64
only**, adding locked `sha2-asm 0.6.4`. The upstream AArch64 implementation
detects CPU SHA instructions at runtime and retains its software fallback.
This accelerates CPU checksumming, not ANE/model computation. Cargo feature
unification also benefits other users of SHA-2 0.10 in the same ARM binary;
the digest algorithm, cache format and identity remain unchanged. Target-tree
checks show the feature on `aarch64-apple-darwin` and absent on
`x86_64-apple-darwin`; the latter is a resolution check, not an Intel build or
hardware test.

The existing immutable snapshot ownership, streaming buffer sizes, F_NOCACHE,
checksums, fsync/rename transaction, memory admission and Sync default remain
unchanged. A coalescing writer candidate was tested and **withdrawn**: ordinary
save median 1.587 → 1.526 seconds was insufficient, DFlash2 1.020 → 1.021 showed
no gain, and end-to-end timing did not establish a benefit. Its additional
buffering/code was not retained. Evidence remains in `coalescing-candidate.patch`,
`sync-coalescing-summary.json` and `sync/` under `.build/prefix-followup/`.

### Actual checksumming and model measurements

A standalone Release probe hashed the same 64 MiB input five times. Median
software time was **106.8 ms**, versus **21.8 ms** with the hardware feature.
Both digests matched each other and independent Python `hashlib`. This isolates
hash cost from GPU/prefill/disk timing; it does not measure SSD throughput.

Physical serving measurements again used the M5 Max 128 GiB host, full experts,
b_max 4, 2,048-token chunks, hot retention off, ANE off, 4,038–4,039 prompt
tokens, 16 output tokens, two retained warmups and six first-miss measurements
per route/version. All stores stayed below the 8 GiB quota. The software
control executable was `2a5ec44685cae4720f05592572a557cf77f0be8050fd5a2af0f1945594a79cd4`;
hardware executable was
`02c8adeee268f497f709ef3756534014c9815797e4f9aec50d316096a95c9ece`.

| Route | Software synchronous save median | Hardware synchronous save median (min–max) | Saves per measured request |
| --- | --- | --- | --- |
| Ordinary | **1.587 s** | **0.573 s** (0.543–0.621) | 3, no skips/failures |
| DFlash2 | **1.020 s** | **0.313 s** (0.301–0.323) | 2, no skips/failures |

Savings in the measured persistence operation were approximately 64%/69%.
Ordinary total save cost was also below the historical 1.267-second pre-adaptive
sample. A final replay with the unchanged software executable (three measured
requests) still took median **1.647 seconds**, so the save-cost improvement was
not obtained by silently disabling saves or only using a faster later control.

**End-to-end timing is less settled.** Hardware ordinary TTFT median was
5.358 seconds, decode 0.539; DFlash2 TTFT was 6.325 seconds, decode 1.541.
DFlash2 did not show a corresponding TTFT improvement over its software
control (6.279 seconds). The unchanged ordinary software replay itself ranged
6.351–10.104 seconds TTFT, with median 6.517. Save-time reduction must not be
reported as an equal TTFT/throughput gain or as a cure for all long tails.
Uncached small-write overhead still exists; this change removes substantial
CPU hash cost while retaining the resource/durability policy.

### Correctness, compatibility and limits

The hardware-feature prefix suite passed **74 tests, two ignored**. Independent
Python hashing validated all **40 new payload files / 9,168,594,576 bytes**.
The pre-adaptive Stage 7 executable read the new ordinary and DFlash2 files,
reported actual prefix reuse, and returned the exact expected continuations.
No model source, external MLX source, prefix schema or execution fingerprint
changed. The coalescing candidate's three extra tests are not counted as tests
of the delivered implementation.

No physical M1 Pro 32 GiB, 64 GiB or CPU lacking SHA instructions was tested.
The software probe exercises the portable implementation separately; it does
not substitute for testing runtime fallback on such a CPU. Native performance
results above apply to the actual M5 host and these workloads.

Artifacts under `.build/prefix-followup/`: `hash-{soft,hardware}.json`,
`hash-probe/`, `sha-features-{arm,x86}.txt`, `sha-summary.json`, `sha/*.json`,
`sha-prefix-tests.log`, `sha-compat/result.json` and matching logs.

## Prefix persistence follow-up: backend instance-lock lifetime (2026-11-04)

The CLI guard previously relied only on closing its `File`. BSD `flock` shares
an open-file-description lock across inherited descriptors, including the
fork-to-exec window even when FD_CLOEXEC is set. An unrelated child can
therefore retain the reference after the parent's guard is dropped, producing
the same `ironmlx_instance_already_running` reacquire failure as the earlier
parallel test. The original run did not record the inheriting child PID, so
its exact process history cannot be reconstructed retrospectively.

A controlled fork test reproduced the defect with **unmodified guard code**:
two lock tests passed, while reacquisition after owner Drop failed as expected
with the child still alive. The guard now records its acquiring PID and sends
LOCK_UN explicitly on Drop **in that process only**. Dropping an inherited
guard in a child only closes its descriptor; it must not unlock the live
parent's shared lock. No retries were added to hide a failed lock acquisition,
and the user-facing error, path, permissions and close-on-exec policy remain.

Validation passed:

- Three lock tests in parallel, including inherited-descriptor retention,
  child Drop protection and the original exclusivity/release test.
- **100 rounds × three parallel lock tests = 300 passed**.
- The entire CLI library at **16 threads: 379 passed, 14 ignored, 0 failed**.

Tests use unique temporary paths, not the user's production lock. Child code
avoids MLX, allocation, logging and unwinding after fork; parent teardown
kills/reaps its own test children. Ready waits are bounded and handle EINTR;
pipe close-on-exec, EOF and a child-side deadline limit leaked test processes.

The fixed boundary is normal guard destruction, including inherited-fd races.
SIGKILL cannot execute Drop: if a child still holds an inherited description,
kernel lock release still waits for its close/exec/exit. This OS lifetime rule
has not been replaced by a separate PID-file recovery protocol. External MLX
thread identity/native source was not changed.

Artifacts: `.build/prefix-followup/lock-before.log`, `lock-after.log`,
`lock-repeat.json`, `lock-repeat.log`, and `cli-parallel.log`.

## Follow-up closeout

The final Release workspace check passed **2,541 tests, 303 explicitly ignored,
zero failures**, across 211 test targets. CLI parallel verification and the
300 repeated lock checks are separately recorded above; they are not added to
the workspace count. Four qualifier Python tests passed. Stable and pinned
nightly format checks, whitespace checks and the CI-equivalent pinned-nightly
workspace Clippy scope passed with `-D warnings`.

An additional CLI **test-target** Clippy attempt failed on two existing
`items_after_test_module` diagnostics in unmodified files:
`ironmlx/src/server/embedding_audio.rs:18` and
`ironmlx/src/server/diarizations.rs:173`. They were not suppressed or repaired
as part of the three requested items. This failed expanded gate and the
previously recorded all-target MLX test diagnostics remain distinct from the
passing CI-equivalent lint scope.

The final production CLI SHA-256 was
`dc2fc193510831710c3b8ab92de23268cab7368ff52fc070fafd14a28da55066`.
Actual HTTP qualification passed on ordinary and DFlash2 with mixed text,
single-image/multi-image prompts, B1/B4 equivalence, cancellation and subsequent
B2 recovery. DFlash2 results matched the ordinary baseline, pending persistence
drained with zero failures, and Flash active KV offload stayed disabled.
The test did not requalify ANE forced-profile combinations or old external MLX
GPU-default failures. No new `Context leak` diagnostic appeared in these logs.

**The long-tail issue remains open.** A withdrawn GPU-mask candidate also
showed a 3.949-second DFlash2 Auto content gap despite removing the host readback
(`mask-latency/dflash-auto.json`). That reinforces the decision not to claim the
sync point as the sole cause. The delivered change improves checksum/save cost
and fixes a reproducible lock-lifetime race; it does not certify a general GPU
latency SLO. Physical M1 Pro 32 GiB/64 GiB, controller-cold SSD, sustained
pressure and no-SHA-instruction CPU qualification remain outstanding.

The original Release App was restored healthy on port 9068. Both preference
file hashes and the recorded backend/benchmark/Metal bundle hashes were
unchanged. The App was **not repackaged** with these follow-ups; the final CLI
was built/tested separately. No external MLX source or branch changed, and
Stage 8 was not implemented. The baseline adaptive change `2480914b0` was
pushed before work; latency diagnostics, checksum acceleration and the lock
fix are separate follow-up commits, with no further push in this task.

Final artifacts under `.build/prefix-followup/`: `workspace-final.log`,
`clippy-workspace.log`, `clippy-cli-tests.log`, `fmt-{stable,nightly}.log`,
`python-final.log`, `final-binary.json`, `final-serving/result.json`,
`final-serving-driver.log`, and `app-restored.json`.
