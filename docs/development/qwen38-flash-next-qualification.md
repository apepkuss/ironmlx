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
export IRONMLX_FLASH_EXPERT_CACHE_MIB=16384 # upper bound in SSD mode
```

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

ANE mode remains `auto`/`off`. The existing Dense g64 channel-sharing path is
not qualified for Flash g32/640-wide shared experts; `auto` must report
`not_applicable` until a measured end-to-end win passes the numerical and
memory gates. `off` explicitly disables ANE.

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
and 4 GiB of temporary overlap. It reserves the deferred vision tower before
advertising image input. EnginePool includes future bounded-cache/request
ceilings when planning another model and retains Flash liability while an
unloaded slot still has active request leases.

Video input is not exposed by the existing HTTP image contract. Native image
support must not be advertised as video support. Flash ANE remains explicitly
not applicable, and lower-memory physical hardware/older Apple GPUs still
need independent on-device qualification. No standalone manifest is rewritten
merely to upgrade an older download-only import into a newly supported model.

## Repository checks

The final source passed 1,755 Rust unit tests across CLI, core, LM and runtime,
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
