# Qwen3.6 Performance Phase 5 MLX QMM Root Cause

**Goal:** Close the remaining long-prefill `GatedDeltaNet` performance gap with evidence instead of continuing the failed P5h-style guess-and-patch loop.

**Artifact root:** `/tmp/ironmlx-qwen36-perf-phase5-latest`

## Prototype Prefill Review

Reviewed the earlier prototype at `/Users/xin/workspace/iron-rivals/ironmlx`.

High-signal ideas:

- `BatchGenerator::init_worker_stream()` promotes a worker-thread GPU stream to MLX's default stream. This was useful in the prototype because MLX 0.31.2 uses thread-local default-stream storage and compiled closures can otherwise capture a different stream.
- Prefill chunks evaluate cache state and final logits with one bundled `eval_many`, avoiding per-array dispatch overhead.
- Completed prefill cache arrays are detached to cut graph references and release old Metal resources.
- Chunked mid-admission interleaves prefill chunks and decode steps to bound active-stream stalls.

Current ironmlx already has scheduler admission queueing, chunked mid-admission, batched prefill surfaces, and bundled eval paths. The remaining question was whether prototype stream promotion explained the Qwen3.6 long-prefill gap.

## Stream Target Diagnostics

Added `--stream-mode` to `ironmlx-qlinear-bench`:

- `default`
- `explicit-default`
- `new-default`
- `new-explicit`

Artifacts:

- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_stream_default_seq521_seq1.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_stream_explicit_default_seq521_seq1.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_stream_new_default_seq521_seq1.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_stream_new_explicit_seq521_seq1.json`

Selected `seq=521` p50:

| Stream mode | qkvz direct qmm | qkvz linear+slice | out direct qmm | norm+out |
| --- | ---: | ---: | ---: | ---: |
| default | 2.414 ms | 2.316 ms | 0.971 ms | 1.023 ms |
| explicit-default | 2.423 ms | 2.301 ms | 0.961 ms | 0.987 ms |
| new-default | 2.380 ms | 2.318 ms | 0.968 ms | 1.010 ms |
| new-explicit | 2.612 ms | 2.335 ms | 0.993 ms | 1.336 ms |

Conclusion: prototype default-stream promotion is not the root cause of the current qlinear/GDN gap. Directly applying that production strategy would be unsupported by evidence.

## C++ QMM Loop Probe

Added a diagnostic-only C++ loop:

- `mlx::quantization::quantized_matmul_bench_ms`
- `--include-cxx-qmm` in `ironmlx-qlinear-bench`

This keeps repeated `mlx::core::quantized_matmul` calls inside C++, with `target`, optional arguments, and `mode` hoisted outside the loop. It separates Rust/C++ per-call bridge overhead from MLX's own quantized-matmul scheduling/kernel path.

Artifact:

- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_cxx_loop_seq521_rerun50.json`

Selected `seq=521` p50 with local MLX C++ 0.32.0:

| Case | p50 |
| --- | ---: |
| qkvz C++ loop qmm | 2.286 ms |
| qkvz linear+slice | 2.262 ms |
| out C++ loop qmm | 0.953 ms |
| out direct qmm | 0.951 ms |

Conclusion: Rust/C++ bridge overhead is not the main root. The C++ loop remains in the same performance band as Rust direct qmm.

## MLX Version Finding

The decisive difference was the MLX runtime:

- Rust default build used `/Users/xin/.local/mlx`, whose headers report MLX `0.32.0`.
- Python/MLX reference used the wheel-provided MLX `0.31.2`.

Built the same Rust bench against the wheel's MLX 0.31.2 into a separate target dir (`/tmp/ironmlx-target-mlx0312`) and ran the same model/layer/shape.

Artifacts:

- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_qlinear_cxx_loop_seq521_mlx0312_rerun50.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_gdn_seq521_mlx0312_rerun50.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/ironmlx_gdn_seq521_mlx0320_rerun50.json`
- `/tmp/ironmlx-qwen36-perf-phase5-latest/captures/mlx_qlinear_seq521_rerun50.json`

Focused qlinear `seq=521` p50:

| Runtime | qkvz direct qmm | qkvz linear+slice | out direct qmm |
| --- | ---: | ---: | ---: |
| Rust + MLX 0.32.0 | 3.444 ms | 2.262 ms | 0.951 ms |
| Rust + MLX 0.31.2 | 1.307 ms | 0.815 ms | 0.397 ms |
| Python + MLX 0.31.2 | 1.370 ms | 1.312 ms | 0.573 ms |

Full `GatedDeltaNet` `seq=521` p50:

| Runtime | no-cache | cache-out-only | cache-state-eval |
| --- | ---: | ---: | ---: |
| Rust + MLX 0.32.0 | 4.469 ms | 3.917 ms | 3.920 ms |
| Rust + MLX 0.31.2 | 3.439 ms | 1.883 ms | 1.883 ms |

Conclusion: the remaining long-prefill gap is primarily an MLX 0.32.0 quantized-matmul regression or local MLX build difference, not a Qwen3.6 MoE architecture flaw, scheduler/admission policy flaw, GDN kernel flaw, or Rust bridge overhead issue.

## Next Tasks

- Decide product policy for MLX runtime selection: pin/recommend MLX 0.31.2 for Qwen3.6 4-bit production until the 0.32.0 qmm regression is understood, or bisect/fix the local MLX 0.32.0 build.
- Add a lightweight runtime/version guard only after deciding the policy; do not silently paper over the dependency-level regression.
- Keep the qlinear C++ loop probe as a regression tool for future MLX upgrades.
- Re-run end-to-end Qwen3.6 core/serve performance after selecting the runtime policy.
