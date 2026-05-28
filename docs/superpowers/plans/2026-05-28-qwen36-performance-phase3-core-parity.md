# Qwen3.6 Performance Phase 3 Core Parity Plan

**Goal:** Measure Qwen3.6 MoE model-core latency against the mlx-lm/oMLX execution shape before changing kernels or scheduler policy.

**Architecture:** Keep the benchmark outside production serving. Use a fixed rendered prompt from the same Phase 1 request template, load the model once, then measure repeated request-local cache construction + prefill + first-token materialization + 16-token decode. Compare:

- `mlx-lm-direct`: `generate_step` external prefill shape, prompt tokens `0..N-2`, final prompt token through `_step`.
- `ironmlx-gs-text`: `GenerationStream::new_text_only`, one-shot full-prompt `forward_on` when prompt length is below `prefill_chunk_size`.
- `ironmlx-scheduler-text`: `Scheduler::prefill_admitted`, production short-prompt shape, external prefix hidden pass plus final-token `forward_on`.

**Artifacts:** `/tmp/ironmlx-qwen36-perf-phase3-latest`

## Tasks

- [x] Create Phase 3 artifact root and metadata.
- [x] Run mlx-lm direct benchmark with the same rendered chat prompt.
- [x] Add a small Rust core benchmark binary for ironmlx text-only paths.
- [x] Run `ironmlx-gs-text` and `ironmlx-scheduler-text` on Qwen3.6-35B-A3B-4bit.
- [x] Summarize parity gaps and choose the next optimization target from evidence.

## Current Evidence

`mlx-lm-direct` on the rendered 524-token prompt:

- prompt time p50: ~200.4 ms
- decode time p50: ~113.4 ms for 16 generated tokens
- wall p50: ~406.0 ms

`ironmlx-core` on the same rendered prompt:

- `gs-text` TTFT p50: ~325.9 ms
- `scheduler-text` TTFT p50: ~321.7 ms
- `scheduler-text` E2E p50: ~443.5 ms

Interpretation: oMLX HTTP `c=1 pp512 tg16` TTFT p50 (~221.9 ms) is close to mlx-lm direct execution plus API overhead. ironmlx's in-process core TTFT is already ~322-326 ms, matching the Phase 1/2 HTTP/P5H behavior. The single-request gap is therefore inside the Qwen3.6 MoE model graph/materialization, not HTTP parsing, SSE, scheduler admission, or the scheduler's prefix+last-token split.

Next optimization target: model-kernel call shape, especially MoE routed expert sort/gather/scatter, fused gate+up gather-qmm, down gather-qmm, GatedDeltaNet fused projections/conv/kernel/norm path, and materialization boundaries against mlx-lm's `SwitchGLU` + `GatedDeltaNet` reference shape.

## Guardrails

- Do not optimize yet; first establish whether the gap is in ironmlx `GenerationStream`, production scheduler split-prefill, or both.
- Treat `p5h-profile` probe timings as directional because sorted MoE profile mode differs from the current production rank-3 sorted path.
- If adding Rust code, run the required Rust validation gate before committing.
