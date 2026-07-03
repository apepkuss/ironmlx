# Gemma4 Adaptive Admission Design

## Goal

Support a production default for Gemma4 drafter that keeps low-latency behavior for ordinary requests while allowing long, chunked concurrent prompts to use the b_max=4 throughput path.

## Current Problem

The old adaptive branch only raised Gemma4 drafter defaults to `b_max=4`. That exposes more scheduler slots, but it also lets fresh admission batch too many prompts at once. The measured result is higher long-prompt throughput with worse TTFT and ITL. Keeping `b_max=1` protects latency but cannot use the batched drafter decode work under concurrent agent-style prompts.

## Design

Separate physical scheduler capacity from runtime admission capacity:

- Physical capacity: Gemma4 drafter defaults to `b_max=4` when the user did not explicitly set `--max-sequences` and did not pass an explicit scheduler profile.
- Runtime admission capacity: a new adaptive policy limits fresh batches and rolling mid-admission based on request shape.
- Short or non-chunked requests use a latency-oriented effective cap of 2, so they do not behave like fixed `b_max=4`.
- Long chunked requests start with a fresh batch limit of 1, then can use additional slots through cadence-protected rolling mid-admission.
- Non-Gemma4 drafter paths keep the existing scheduler behavior.

The policy is implemented in a small server module and injected only by the Gemma4 drafter scheduler actor spawn functions. CLI single-model, engine-pool, and App dynamic model loading all use the same scheduler-default helper so text and VL routes share the same behavior.

## Invariants

- Explicit user `--max-sequences` is never changed.
- Explicit scheduler profile is never changed.
- Auto-loaded default/store profile may be raised to physical b_max=4 for Gemma4 drafter so the adaptive actor has capacity to use.
- Adaptive admission never exceeds physical `b_max`.
- Long prompt concurrency uses prefill chunking and decode cadence caps before consuming extra slots.
- Active KV and paged prefix cache semantics remain unchanged.

## Tests

- Unit-test the adaptive policy with short, long chunked, and non-chunked requests.
- Verify Gemma4 drafter default resolution raises physical `b_max` only when allowed.
- Verify scheduler queue drain respects adaptive caps.
- Run Rust formatting, clippy, release build, and Gemma4 active-KV regression smoke.
