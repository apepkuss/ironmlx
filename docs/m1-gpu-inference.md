# M1 Pro/Max GPU execution

`ironmlx serve` and `ironmlx generate` accept `--m1-profile auto|off`.
The default is **off**. `auto` constructs an engine-owned GPU plan only for
Qwen3.8-27B affine4/group64 on a real Apple M1 Pro or M1 Max GPU. Unsupported
devices/models retain the established execution path. This does not change
`--m5-dflash2-profile`, its defaults, kernels or global Metal settings.

The first implementation selects existing native GPU primitives:

| Work | M1 route |
|---|---|
| One projection row | MLX product-stable packed affine4 QMV |
| 2–16 projection rows | Product-stable `qmv_fast_wide`, up to 2 rows/group on Pro and 4 on Max |
| More than 16 rows | Native MLX affine4 QMM, preserving the existing prefill arithmetic |
| GatedDeltaNet | Existing fused q/k normalization and gate kernels with their shape/dtype guards |
| Other operands or CPU streams | Existing fallback; no GPU-only primitive on a CPU stream |

Existing exact speculative and batch/position/product-isolated projection
scopes take priority over the M1 projection route, including wide verify blocks.

No extra resident weight format or full-layer dequantized weight workspace is
created. Chunk sizes, speculative budgets, attention and recurrent state
algorithms keep their existing policies. The row-group settings are opt-in
recipes, not a claim of measured speedup. Dedicated M1 tiled prefill GEMM,
attention kernels and further tuning are future work.

The plan is scoped to text forwards and target logit projections; the DFlash2
drafter is unchanged. Ordinary, MTP and DFlash2 target execution use the same
model-owned plan. Scoped state masks nested inactive models and restores its
previous value on return/error/unwind. M1 selection does not install global
environment variables or mutate the M5 `OnceLock` profile.

`healthz.m1_profiles`, omitted when no M1 plan was requested, reports each model's
status, device, fingerprint and narrow/prefill/fallback projection-call counts.
These count graph construction routes, not GPU timing measurements. The daemon
aggregates entries without combining different models' plans. DFlash2 execution
fingerprints and ordinary/MTP paged-prefix namespaces include the M1 recipe;
RAM prefix caches remain owned by their individual actors.

All existing load, cold-materialization, prefill, KV and workspace memory guards
remain in force. No memory ceilings, cache residency defaults or pinning policy
are changed.

ANE is **not implemented** in the M1 plan. Existing `--ane` behavior on M5 is
unchanged. M1 reports `ane: not_implemented` and executes on the GPU.
