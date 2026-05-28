# Qwen3.6 Performance Phase 4 Model Hotspots

**Goal:** Identify the model-graph source of the Qwen3.6 MoE single-request gap after Phase 3 showed the gap is already present inside ironmlx core execution.

**Artifact root:** `/tmp/ironmlx-qwen36-perf-phase4-latest`

## Context

Phase 3 measured the same rendered 524-token prompt:

- `mlx-lm-direct`: prompt p50 ~200.4 ms, decode p50 ~113.4 ms, wall p50 ~406.0 ms.
- `ironmlx-core gs-text`: TTFT p50 ~325.9 ms.
- `ironmlx-core scheduler-text`: TTFT p50 ~321.7 ms, E2E p50 ~443.5 ms.

The gap is therefore not primarily HTTP, SSE, admission, or scheduler split-prefill overhead. Phase 4 focuses on model-path evidence.

## Evidence

### P5H/P5G Directional Profile

P5H measurement-eval probes ranked MoE `gather_qmm_gate_up`, GDN projection/norm/kernel spans, and attention as large contributors. This ranking is useful directionally, but the forced `eval` probes split the lazy graph and can exaggerate intermediate costs.

P5G layer1 on Qwen3.6 GDN showed large boundary totals:

- earlier single request: prefill GDN total ~161.9 ms across 30 linear-attention layers.
- warm-server reproduction:
  - request 1 prefill GDN total: 153.9 ms, decode GDN total: 11.0 ms.
  - request 2 prefill GDN total: 181.5 ms, decode GDN total: 11.7 ms.
  - HTTP totals: warmup 710.5 ms, measured 563.9 ms.

Interpretation: layer1 boundary timings are noisy and barrier-heavy, but they keep pointing at GDN/materialization as a higher-risk area than HTTP/scheduler.

### Full MoE Path Microbench

Added `scripts/qwen36_moe_path_compare.py`.

This compares three full sparse-MoE block paths under the same MLX weights:

- mlx-lm `Qwen3NextSparseMoeBlock` reference.
- ironmlx-shaped split gate/up path.
- ironmlx-shaped fused gate/up path.

Results from `/tmp/ironmlx-qwen36-perf-phase4-latest/captures/mlx_moe_path_compare_seq521_seq1.json`:

| Shape | mlx-lm reference p50 | ironmlx split p50 | ironmlx fused p50 | Finding |
| --- | ---: | ---: | ---: | --- |
| `seq=521`, routes=4168 | 3.585 ms | 3.648 ms | 3.580 ms | fused/reference ratio 0.999x |
| `seq=1`, routes=8 | 0.257 ms | 0.261 ms | 0.256 ms | fused/reference ratio 0.996x |

Both ironmlx-shaped variants match reference numerically (`max_abs_diff=0.0`).

Conclusion: the current MoE routed-MLP execution shape is not the main steady-state single-request gap.

### Full GatedDeltaNet Path Microbench

Added `scripts/qwen36_gdn_path_compare.py`.

This compares mlx-lm's split-projection GDN reference with an ironmlx-shaped fused qkvz/ba projection path while still using MLX's Python-side `gated_delta_update`.

Results from `/tmp/ironmlx-qwen36-perf-phase4-latest/captures/mlx_gdn_path_compare_seq521_seq1.json`:

| Shape | Cache | mlx-lm split p50 | fused-shape p50 | Finding |
| --- | --- | ---: | ---: | --- |
| `seq=521` | no-cache | 2.067 ms | 1.766 ms | fused shape 0.855x |
| `seq=521` | fresh cache | 1.756 ms | 1.739 ms | parity |
| `seq=1` | no-cache | 0.259 ms | 0.261 ms | parity |
| `seq=1` | fresh cache | 0.256 ms | 0.257 ms | parity |

The fused-shape path is numerically equivalent to the reference within bf16 tolerance (`seq=521 max_abs_diff=0.03125`, `seq=1 max_abs_diff=0.0`).

Conclusion: GDN projection fusion itself is not the problem; it is neutral-to-positive under MLX. If GDN is the gap, the likely source is lower-level ironmlx execution around custom Metal dispatch, cache/materialization boundaries, or Rust MLX binding call shape.

### Rust GatedDeltaNet Core Microbench

Added `ironmlx/src/bin/ironmlx-gdn-bench.rs`.

This loads one real Qwen3.6 linear-attention layer directly from the checkpoint and measures `GatedDeltaNet::forward_on` without model assembly, HTTP, scheduler, tokenizer, or sampler overhead.

Results from `/tmp/ironmlx-qwen36-perf-phase4-latest/captures/ironmlx_gdn_bench_seq521_seq1.json`:

| Shape | Cache mode | ironmlx p50 | Finding |
| --- | --- | ---: | --- |
| `seq=521` | no-cache | 4.766 ms | materially slower than MLX no-cache full GDN at 2.067 ms |
| `seq=521` | fresh cache, output eval only | 3.982 ms | materially slower than MLX fresh-cache full GDN at 1.739 ms |
| `seq=521` | fresh cache, output + cache-state eval | 3.964 ms | cache-state materialization is not the main delta |
| `seq=1` | no-cache | 0.268 ms | roughly parity with MLX no-cache at 0.261 ms |
| `seq=1` | fresh cache, output eval only | 0.263 ms | roughly parity with MLX fresh-cache at 0.257 ms |
| `seq=1` | fresh cache, output + cache-state eval | 0.261 ms | cache-state materialization is not the decode-path issue |

Additional `seq=521` ablations under `p5g-profile`:

| Diagnostic | p50 | Finding |
| --- | ---: | --- |
| profile feature off-mode control | 4.037 ms | matches non-profile direct benchmark |
| substitute conv output | 3.949 ms | conv is not the main root cause |
| substitute `t_arr` scalar | 4.043 ms | scalar construction is not the root cause |
| substitute `compute_g` input | 4.877 ms | not a useful optimization direction; changed kernel input values can worsen timing |
| temporary force-regular-state kernel | 3.959 ms | zero-state kernel variant is not the root cause |

Conclusion: the long-prefill gap is reproduced inside Rust `GatedDeltaNet` itself. The shape-specific signature is important: `seq=1` is at parity while `seq=521` is about 2.3x slower per layer than the MLX Python reference. The remaining high-signal area is the custom gated-delta Metal dispatch path and Rust MLX binding/materialization behavior for long sequences.

## Lessons From P5h/P5i

- Do not continue the prior custom `gather_qmm_gate_up` kernel line: earlier P5i/P5i.c experiments showed that path underperformed MLX's steel implementation.
- Do not optimize based only on forced-eval subspan totals. Full-path MLX microbenches show MoE is effectively at parity despite P5H ranking it highly.
- Treat GDN as the next high-signal target. The Rust core microbench now confirms the gap is in long-prefill `GatedDeltaNet`, not in MoE, projection fusion, HTTP, scheduler, or cache-state eval.

## Next Tasks

- Add a Rust-side GDN stage breakdown benchmark that uses the same real layer and records forced-eval timings for projection, conv, q/k norm, compute_g, gated-delta kernel materialization, and output norm/proj in one process.
- Compare Rust custom Metal kernel dispatch against a same-source MLX/Python custom-kernel harness to determine whether the delta is in kernel source, MLX C++ binding wrappers, or upstream MLX graph scheduling.
- Investigate whether `MetalKernel::dispatch_builder` allocates avoidable per-call vectors/shapes/template args in the long-prefill path; only optimize this if stage evidence shows dispatch construction, not GPU kernel body, is material.
- Only after stage evidence is available choose a production optimization. Current evidence rules out rewriting MoE kernels and does not justify reverting GDN projection fusion.
