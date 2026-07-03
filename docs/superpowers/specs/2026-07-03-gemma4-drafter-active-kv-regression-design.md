# Gemma4 Drafter Active KV Regression Design

## Goal

Add a reproducible heavy regression harness for Gemma4 target models with Gemma4 assistant drafter, paged prefix cache, prefix LRU, and Active KV offload. The harness must catch the class of regressions where App-mode dynamic loading succeeds but concurrent short or long prompt traffic later fails because scheduler health, MTP counters, Active KV counters, or memory-budget counters drift into an unsafe state.

## Scope

- In scope: App daemon mode, dynamic `/admin/api/models/load`, Gemma4 E4B and 12B dense target checkpoints, matching assistant drafter checkpoints, paged prefix cache, optional prefix LRU, Active KV offload, `max_cache_cap=262144`, `mtp_draft_tokens=2`, `b_max=1/2/4`, short prompt concurrency, and agent-like long prompt concurrency.
- Out of scope: replacing `iron-bench`, adding this heavy scenario to default CI, changing scheduler runtime behavior, or changing App UI defaults.
- No compatibility layer: this is a new opt-in regression runner. Existing benchmark artifacts are not migrated.

## Architecture

Keep `iron-bench` engine-neutral. Add an ironmlx-specific Python runner under `scripts/` that owns:

1. release binary discovery/build;
2. App daemon startup with active KV and paged prefix cache flags;
3. dynamic Gemma4 + drafter model load through `/admin/api/models/load`;
4. sequential and concurrent `iron-bench` runs for configured prompt lengths;
5. pre/post `/healthz` capture;
6. assertions over success status, memory budget counters, MTP counters, Active KV degradation, and logical/resident KV cap fields;
7. JSON, CSV, Markdown, metadata, and runnable command artifact output under `docs/benchmarks/gemma4-drafter-active-kv-regression/<timestamp>/`.

The runner should be safe to dry-run without local models. Real-model execution remains opt-in because it requires local Gemma4 checkpoints and enough Apple Silicon memory/GPU headroom.

## Default Matrix

- `e4b_b2`: E4B target + E4B assistant, `b_max=2`, `prompt_len=2048,24576`, `concurrent=2`.
- `e4b_b4`: E4B target + E4B assistant, `b_max=4`, `prompt_len=2048,24576`, `concurrent=4`.
- `12b_b2`: 12B target + 12B assistant, `b_max=2`, `prompt_len=2048,24576`, `concurrent=2`.

All default variants use `max_cache_cap=262144`, `mtp_draft_tokens=2`, `kv_quant=k3v4`, paged prefix cache, prefix LRU disabled by default, and Active KV offload enabled.

## Required Assertions

- Model load response succeeds and reports `mtp_enabled=true`.
- `/healthz.memory.kv_cache_budget_policy == "active_kv_offload"`.
- `/healthz.memory.kv_cache_logical_cap_tokens == min(262144, model.max_position_embeddings)`.
- `/healthz.memory.kv_cache_resident_cap_tokens < kv_cache_logical_cap_tokens`.
- `/healthz.scheduler.memory_budget_exceeded_count` does not increase during benchmark cells.
- `/healthz.active_kv_offload.degraded == false`.
- `/healthz.active_kv_offload.swap_error_count == 0`.
- `/healthz.mtp.enabled == true`.
- `/healthz.mtp.draft_tokens == 2`.
- `/healthz.mtp.prefill_count` or `/healthz.mtp.step_count` increases after greedy traffic.
- `iron-bench` reports no failed request and non-zero token throughput.

## Files

- Create `scripts/gemma4_drafter_active_kv_regression.py`: runner implementation.
- Create `scripts/test_gemma4_drafter_active_kv_regression.py`: dry-run/unit coverage for command construction, model path resolution, load payloads, health assertions, summary rendering, and run-plan filtering.
- Create `docs/benchmarks/gemma4-drafter-active-kv-regression/README.md`: usage, required local models, environment variables, and expected artifacts.
- Create `docs/superpowers/plans/2026-07-03-gemma4-drafter-active-kv-regression.md`: implementation plan.

## Validation

- `python3 scripts/test_gemma4_drafter_active_kv_regression.py`
- `cargo test -p iron-bench --release`
- `cargo fmt`
- `cargo +nightly fmt --all -- --check`
- `cargo +nightly clippy --all-features --workspace -- -D warnings`
- `cargo build --release`
- Optional real-model smoke: run the new script with E4B and 12B local checkpoints, then inspect `summary.md` for all `ok` rows.
